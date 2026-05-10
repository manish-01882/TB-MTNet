"""Cell 7 – 3-Stage Progressive Training with 5-Fold Cross-Validation."""


# ── Layer-wise LR decay for Inception-v3 backbone ────────────────────
def _layerwise_backbone_params(backbone: nn.Module,
                                base_lr: float,
                                decay: float = 0.8) -> list:
    """Return AdamW-style param-group dicts with depth-aware LR scaling.

    Inception-v3 stage ordering (shallow → deep):
      Stage 0 (earliest) : Conv2d_1a … Conv2d_4a  →  LR × decay³  (≈ 0.51×)
      Stage 1 (mid)      : Mixed_5b … Mixed_6b    →  LR × decay²  (≈ 0.64×)
      Stage 2 (late-mid) : Mixed_6c … Mixed_6e    →  LR × decay¹  (≈ 0.80×)
      Stage 3 (deepest)  : Mixed_7a … Mixed_7c    →  LR × decay⁰  (= 1.00×)

    Any unmatched parameters (e.g. timm feature_info wrappers) fall back
    to base_lr so nothing is silently excluded from training.
    """
    stage_prefixes = [
        # stage 0 – earliest, most generic features
        ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
         "Conv2d_3b_1x1", "Conv2d_4a_3x3"],
        # stage 1
        ["Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b"],
        # stage 2
        ["Mixed_6c", "Mixed_6d", "Mixed_6e"],
        # stage 3 – deepest / most task-specific
        ["Mixed_7a", "Mixed_7b", "Mixed_7c"],
    ]
    n_stages   = len(stage_prefixes)
    assigned   = set()
    param_groups = []

    for stage_idx, prefixes in enumerate(stage_prefixes):
        lr_scale    = decay ** (n_stages - 1 - stage_idx)   # deeper → higher LR
        stage_params = []
        for name, param in backbone.named_parameters():
            if name not in assigned and any(name.startswith(pf) for pf in prefixes):
                stage_params.append(param)
                assigned.add(name)
        if stage_params:
            param_groups.append({"params": stage_params, "lr": base_lr * lr_scale})

    # Safety net: any unmatched params get base_lr
    remaining = [p for n, p in backbone.named_parameters() if n not in assigned]
    if remaining:
        param_groups.append({"params": remaining, "lr": base_lr})

    return param_groups


def make_optimizer(model: nn.Module, cfg: "Config",
                   stage: int, mtl_loss: nn.Module) -> torch.optim.Optimizer:
    core = model.module if hasattr(model, "module") else model
    if stage == 1:
        params = [
            # Backbone: layer-wise decay — early layers learn slower
            *_layerwise_backbone_params(core.backbone, cfg.S1_LR),
            {"params": core.eca.parameters(),        "lr": cfg.S1_LR},
            {"params": core.bridge.parameters(),     "lr": cfg.S1_LR},
            {"params": core.transformer.parameters(),"lr": cfg.S1_LR},
            {"params": core.norm.parameters(),       "lr": cfg.S1_LR},
            {"params": core.cls_head.parameters(),   "lr": cfg.S1_LR},
            {"params": core.cls_token,               "lr": cfg.S1_LR},
            {"params": core.pos_embed,               "lr": cfg.S1_LR},
            {"params": mtl_loss.parameters(),        "lr": cfg.S1_LR},
        ]
    elif stage == 2:
        params = [
            # Backbone: layer-wise decay — early layers learn slower
            *_layerwise_backbone_params(core.backbone, cfg.S2_LR),
            {"params": core.eca.parameters(),         "lr": cfg.S2_LR},
            {"params": core.bridge.parameters(),      "lr": cfg.S2_LR},
            {"params": core.transformer.parameters(), "lr": cfg.S2_LR},
            {"params": core.norm.parameters(),        "lr": cfg.S2_LR},
            {"params": core.cls_head.parameters(),    "lr": cfg.S2_LR},
            {"params": core.reg_head.parameters(),    "lr": cfg.S2_REG_LR},
            {"params": core.cls_token,                "lr": cfg.S2_LR},
            {"params": core.pos_embed,                "lr": cfg.S2_LR},
            {"params": mtl_loss.parameters(),         "lr": cfg.S2_LR},
        ]
    else:  # stage 3
        params = model.parameters()
        return torch.optim.AdamW(params, lr=cfg.S3_LR,
                                 weight_decay=cfg.WEIGHT_DECAY)
    return torch.optim.AdamW(params, weight_decay=cfg.WEIGHT_DECAY)


def make_scheduler(opt, n_epochs: int, cfg: "Config") -> CosineLRScheduler:
    return CosineLRScheduler(
        opt,
        t_initial=n_epochs,
        lr_min=cfg.ETA_MIN,
        warmup_t=cfg.WARMUP_EPOCHS,
        warmup_lr_init=1e-5,
        cycle_decay=cfg.CYCLE_DECAY,
    )


def train_epoch(model, loader, opt, scheduler, mtl_loss, scaler,
                cfg, epoch, freeze_reg=False):
    model.train()
    core = model.module if hasattr(model, "module") else model
    if freeze_reg:
        for p in core.reg_head.parameters():
            p.requires_grad_(False)
    else:
        for p in core.reg_head.parameters():
            p.requires_grad_(True)

    total_loss = cls_loss_sum = reg_loss_sum = 0.0
    for batch in loader:
        imgs  = batch["image"].to(cfg.DEVICE)
        y_cls = batch["tb_label"].to(cfg.DEVICE)
        y_sev = batch["severity"].to(cfg.DEVICE)
        s_msk = batch["severity_mask"].to(cfg.DEVICE)

        opt.zero_grad(set_to_none=True)
        amp_dtype = getattr(torch, cfg.AMP_DTYPE)  # bfloat16 or float16
        use_scaler = scaler is not None and cfg.AMP_DTYPE == "float16"
        if cfg.USE_AMP:
            with autocast("cuda", dtype=amp_dtype):
                logit, sev = model(imgs)
                loss, l_c, l_r = mtl_loss(logit, y_cls, sev, y_sev, s_msk)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                opt.step()
        else:
            logit, sev = model(imgs)       # DataParallel splits batch across GPUs
            loss, l_c, l_r = mtl_loss(logit, y_cls, sev, y_sev, s_msk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()

        total_loss   += loss.item()
        cls_loss_sum += l_c.item()
        reg_loss_sum += l_r.item()

    n = len(loader)
    scheduler.step(epoch + 1)
    return total_loss/n, cls_loss_sum/n, reg_loss_sum/n


@torch.no_grad()
def validate(model, loader, mtl_loss, cfg):
    model.eval()
    core = model.module if hasattr(model, "module") else model
    all_probs, all_labels, all_sev_pred, all_sev_true, all_sev_mask = [], [], [], [], []
    total_loss = 0.0

    for batch in loader:
        imgs  = batch["image"].to(cfg.DEVICE)
        y_cls = batch["tb_label"].to(cfg.DEVICE)
        y_sev = batch["severity"].to(cfg.DEVICE)
        s_msk = batch["severity_mask"].to(cfg.DEVICE)

        amp_dtype = getattr(torch, cfg.AMP_DTYPE)
        if cfg.USE_AMP:
            with autocast("cuda", dtype=amp_dtype):
                logit, sev = model(imgs)
                loss, _, _ = mtl_loss(logit, y_cls, sev, y_sev, s_msk)
        else:
            logit, sev = model(imgs)
            loss, _, _ = mtl_loss(logit, y_cls, sev, y_sev, s_msk)

        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logit).cpu())
        all_labels.append(y_cls.cpu())
        all_sev_pred.append(sev.cpu() * cfg.SEVERITY_MAX)
        all_sev_true.append(y_sev.cpu() * cfg.SEVERITY_MAX)
        all_sev_mask.append(s_msk.cpu())

    probs  = torch.cat(all_probs).numpy().ravel()
    labels = torch.cat(all_labels).numpy().ravel().astype(int)
    sp     = torch.cat(all_sev_pred).numpy().ravel()
    st     = torch.cat(all_sev_true).numpy().ravel()
    sm     = torch.cat(all_sev_mask).numpy().ravel().astype(bool)

    auroc = roc_auc_score(labels, probs) if labels.sum() > 0 else 0.0
    mae   = float(np.abs(sp[sm] - st[sm]).mean()) if sm.sum() > 0 else -1.0

    return total_loss / len(loader), auroc, mae


# ── Checkpoint helpers ────────────────────────────────────────────────

def _save_best_ckpt(fold: int, ep: int, auroc: float, mae: float,
                    model, loss, best_auroc: float, ckpt_path: Path) -> float:
    """Overwrite best checkpoint if current AUROC improved. Returns new best."""
    if auroc > best_auroc:
        core = model.module if hasattr(model, "module") else model
        torch.save({
            "fold":       fold,
            "epoch":      ep,
            "auroc":      auroc,
            "mae":        mae,
            "model":      core.state_dict(),
            "loss":       loss.state_dict(),
        }, ckpt_path)
        print(f"    ✓ Best checkpoint saved  (AUROC {auroc:.4f}, MAE {mae:.1f})")
        return auroc
    return best_auroc


def _save_resume_ckpt(fold: int, stage: int, ep: int, best_auroc: float,
                      model, loss, opt, sch, scaler, cfg: "Config") -> None:
    """Rolling resume checkpoint – overwritten after every epoch."""
    core = model.module if hasattr(model, "module") else model
    torch.save({
        # ── position ────────────────────────────────────────────────
        "fold":        fold,
        "stage":       stage,
        "epoch":       ep,          # last *completed* epoch (0-indexed)
        "best_auroc":  best_auroc,
        # ── weights ─────────────────────────────────────────────────
        "model":       core.state_dict(),
        "loss":        loss.state_dict(),
        # ── optimiser / scheduler ───────────────────────────────────
        "opt":         opt.state_dict(),
        "sch":         sch.state_dict(),
        "scaler":      scaler.state_dict() if scaler is not None else None,
    }, cfg.CKPT_DIR / f"fold{fold}_resume.pt")


def _save_stage_final_ckpt(fold: int, stage: int, model, loss, cfg: "Config") -> None:
    """Snapshot at the boundary between stages – useful for debugging / rollback."""
    core = model.module if hasattr(model, "module") else model
    torch.save({
        "fold":  fold,
        "stage": stage,
        "model": core.state_dict(),
        "loss":  loss.state_dict(),
    }, cfg.CKPT_DIR / f"fold{fold}_stage{stage}_final.pt")
    print(f"    ↳ Stage {stage} final snapshot saved.")


def _load_resume(fold: int, cfg: "Config") -> dict | None:
    """Return resume dict if a resume checkpoint exists for this fold, else None."""
    p = cfg.CKPT_DIR / f"fold{fold}_resume.pt"
    if p.exists():
        ckpt = torch.load(p, map_location=cfg.DEVICE, weights_only=False)
        print(f"  ↺ Resume found: fold {fold+1}  stage {ckpt['stage']}  "
              f"epoch {ckpt['epoch']+1}  best_AUROC {ckpt['best_auroc']:.4f}")
        return ckpt
    return None


# ── Main fold runner ──────────────────────────────────────────────────

def run_fold(fold: int, df: pd.DataFrame,
             train_idx, val_idx, cfg: "Config") -> dict:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1}/{cfg.N_FOLDS}")
    print(f"{'='*60}")

    train_loader, val_loader = make_loaders(df, train_idx, val_idx, cfg)
    print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    # ── Build model & loss ────────────────────────────────────────────
    _model = TBMTNet(cfg).to(cfg.DEVICE)
    _loss  = MultiTaskLoss(cfg.POS_WEIGHT, cfg.HUBER_BETA).to(cfg.DEVICE)

    scaler     = GradScaler("cuda") if (cfg.USE_AMP and cfg.AMP_DTYPE == "float16") else None
    ckpt_path  = cfg.CKPT_DIR / f"fold{fold}_best.pt"

    # ── Load resume state (if available) ─────────────────────────────
    resume     = _load_resume(fold, cfg)
    best_auroc = resume["best_auroc"] if resume else 0.0

    if resume is not None:
        _model.load_state_dict(resume["model"])
        _loss.load_state_dict(resume["loss"])

    # DataParallel must wrap AFTER loading state_dict
    if torch.cuda.device_count() > 1:
        _model = nn.DataParallel(_model)

    # Convenience: which stage/epoch did we last complete?
    resume_stage = resume["stage"] if resume else 0
    resume_epoch = resume["epoch"] if resume else -1   # -1 means "nothing done yet"

    # ── STAGE 1: classification only ─────────────────────────────────
    opt1 = make_optimizer(_model, cfg, stage=1, mtl_loss=_loss)
    sch1 = make_scheduler(opt1, cfg.S1_EPOCHS, cfg)

    if resume_stage == 1:
        # Restore exact optimiser + scheduler position mid-stage
        opt1.load_state_dict(resume["opt"])
        sch1.load_state_dict(resume["sch"])
        if resume["scaler"] and scaler:
            scaler.load_state_dict(resume["scaler"])

    if resume_stage <= 1:
        start_ep = (resume_epoch + 1) if resume_stage == 1 else 0
        print(f"\n  [Stage 1] Classification-only (reg head frozen)"
              f"{'  – resuming from ep ' + str(start_ep+1) if start_ep > 0 else ''}")
        for ep in range(start_ep, cfg.S1_EPOCHS):
            tl, cl, rl = train_epoch(_model, train_loader, opt1, sch1, _loss,
                                     scaler, cfg, ep, freeze_reg=True)
            vl, auroc, mae = validate(_model, val_loader, _loss, cfg)
            print(f"  S1 ep{ep+1:02d}  train={tl:.4f}  val={vl:.4f}  "
                  f"AUROC={auroc:.4f}  MAE={mae:.1f}")
            best_auroc = _save_best_ckpt(fold, ep, auroc, mae,
                                         _model, _loss, best_auroc, ckpt_path)
            _save_resume_ckpt(fold, 1, ep, best_auroc,
                              _model, _loss, opt1, sch1, scaler, cfg)

        _save_stage_final_ckpt(fold, 1, _model, _loss, cfg)

    # ── STAGE 2: full multi-task ──────────────────────────────────────
    opt2 = make_optimizer(_model, cfg, stage=2, mtl_loss=_loss)
    sch2 = make_scheduler(opt2, cfg.S2_EPOCHS, cfg)

    if resume_stage == 2:
        opt2.load_state_dict(resume["opt"])
        sch2.load_state_dict(resume["sch"])
        if resume["scaler"] and scaler:
            scaler.load_state_dict(resume["scaler"])

    if resume_stage <= 2:
        start_ep = (resume_epoch + 1) if resume_stage == 2 else 0
        print(f"\n  [Stage 2] Full multi-task (both heads)"
              f"{'  – resuming from ep ' + str(start_ep+1) if start_ep > 0 else ''}")
        for ep in range(start_ep, cfg.S2_EPOCHS):
            tl, cl, rl = train_epoch(_model, train_loader, opt2, sch2, _loss,
                                     scaler, cfg, ep, freeze_reg=False)
            vl, auroc, mae = validate(_model, val_loader, _loss, cfg)
            print(f"  S2 ep{ep+1:02d}  train={tl:.4f}  val={vl:.4f}  "
                  f"AUROC={auroc:.4f}  MAE={mae:.1f}  "
                  f"s_c={_loss.s_c.item():.3f} s_r={_loss.s_r.item():.3f}")
            best_auroc = _save_best_ckpt(fold, ep, auroc, mae,
                                         _model, _loss, best_auroc, ckpt_path)
            _save_resume_ckpt(fold, 2, ep, best_auroc,
                              _model, _loss, opt2, sch2, scaler, cfg)

        _save_stage_final_ckpt(fold, 2, _model, _loss, cfg)

    # ── STAGE 3: fine-tune + SWA ──────────────────────────────────────
    opt3      = make_optimizer(_model, cfg, stage=3, mtl_loss=_loss)
    sch3      = make_scheduler(opt3, cfg.S3_EPOCHS, cfg)
    swa_model = AveragedModel(_model)
    swa_sch   = SWALR(opt3, swa_lr=cfg.S3_LR)

    if resume_stage == 3:
        opt3.load_state_dict(resume["opt"])
        sch3.load_state_dict(resume["sch"])
        if resume["scaler"] and scaler:
            scaler.load_state_dict(resume["scaler"])

    start_ep = (resume_epoch + 1) if resume_stage == 3 else 0
    print(f"\n  [Stage 3] Fine-tune + SWA"
          f"{'  – resuming from ep ' + str(start_ep+1) if start_ep > 0 else ''}")
    for ep in range(start_ep, cfg.S3_EPOCHS):
        tl, cl, rl = train_epoch(_model, train_loader, opt3, sch3, _loss,
                                 scaler, cfg, ep, freeze_reg=False)
        if ep >= cfg.S3_EPOCHS - cfg.SWA_START:
            swa_model.update_parameters(_model)
            swa_sch.step()
        else:
            sch3.step(ep + 1)
        vl, auroc, mae = validate(_model, val_loader, _loss, cfg)
        print(f"  S3 ep{ep+1:02d}  train={tl:.4f}  val={vl:.4f}  "
              f"AUROC={auroc:.4f}  MAE={mae:.1f}")
        best_auroc = _save_best_ckpt(fold, ep, auroc, mae,
                                     _model, _loss, best_auroc, ckpt_path)
        _save_resume_ckpt(fold, 3, ep, best_auroc,
                          _model, _loss, opt3, sch3, scaler, cfg)

    _save_stage_final_ckpt(fold, 3, _model, _loss, cfg)

    # ── SWA: update batch-norm statistics ────────────────────────────
    class ImageOnlyLoader:
        """Thin wrapper so update_bn receives tensors, not dicts."""
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for b in self.loader: yield b["image"]
        def __len__(self): return len(self.loader)

    torch.optim.swa_utils.update_bn(ImageOnlyLoader(train_loader), swa_model,
                                    device=cfg.DEVICE)

    # ── Fix: evaluate the SWA model and save if it beats current best ─
    # Previously the SWA model's updated weights + BN stats were computed
    # but then discarded — the fold best was always the last non-SWA epoch.
    # SWA averaging is most effective when we actually keep the averaged model.
    print("  Evaluating SWA model on validation set …")
    swa_core = swa_model.module if hasattr(swa_model, "module") else swa_model
    swa_core.eval()

    # Temporarily patch predict path: swa_model wraps the original module;
    # validate() calls model(imgs) which triggers AveragedModel.__call__ → OK.
    swa_val_loss, swa_auroc, swa_mae = validate(swa_model, val_loader, _loss, cfg)
    print(f"  SWA val:  loss={swa_val_loss:.4f}  AUROC={swa_auroc:.4f}  MAE={swa_mae:.1f}")

    if swa_auroc > best_auroc:
        # Extract the underlying averaged parameters (module inside AveragedModel)
        inner = swa_model.module if hasattr(swa_model, "module") else swa_model
        torch.save({
            "fold":   fold,
            "epoch":  "swa",
            "auroc":  swa_auroc,
            "mae":    swa_mae,
            "model":  inner.state_dict(),
            "loss":   _loss.state_dict(),
        }, ckpt_path)
        best_auroc = swa_auroc
        print(f"    ✓ SWA checkpoint saved as fold best  (AUROC {swa_auroc:.4f})")
    else:
        print(f"    ↳ SWA AUROC ({swa_auroc:.4f}) did not beat current best "
              f"({best_auroc:.4f}) — keeping Stage-3 checkpoint.")

    # ── Fold complete – remove rolling resume file ────────────────────
    resume_p = cfg.CKPT_DIR / f"fold{fold}_resume.pt"
    if resume_p.exists():
        resume_p.unlink()
        print(f"  ✓ Resume checkpoint cleaned up.")

    print(f"\n  Fold {fold+1} best val AUROC = {best_auroc:.4f}")
    return {"fold": fold, "best_auroc": best_auroc, "ckpt": str(ckpt_path)}


# ── Cross-validation entry point ─────────────────────────────────────
def run_training(df: pd.DataFrame, cfg: "Config") -> List[dict]:
    cv = StratifiedGroupKFold(n_splits=cfg.N_FOLDS, shuffle=True,
                              random_state=cfg.SEED)
    X      = np.arange(len(df))
    y      = df["strat_key"].values
    groups = df["patient_id"].values

    results = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
        res = run_fold(fold, df, tr_idx, va_idx, cfg)
        results.append(res)
        print(f"\nFold {fold+1} summary: AUROC={res['best_auroc']:.4f}")

    mean_auroc = np.mean([r["best_auroc"] for r in results])
    print(f"\n{'='*60}")
    print(f"  5-Fold CV mean AUROC = {mean_auroc:.4f}")
    print(f"{'='*60}")
    return results


fold_results = run_training(labels_df, CFG)
