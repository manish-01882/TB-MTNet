"""Cell 9 – TTA + 5-Fold Ensemble Inference → submission.csv"""

import json
import torchvision.transforms.functional as TF   # for rotation TTA

# ── Four deterministic TTA augmentations ─────────────────────────────
# All keep spatial size at IMAGE_SIZE×IMAGE_SIZE so the fixed positional
# embedding (196 patch tokens) is never invalidated.
#   1) original
#   2) horizontal flip   – most useful for CXR (L/R mirror is clinically valid)
#   3) +5° rotation      – mild geometric jitter
#   4) −5° rotation
_TTA_FNS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[-1]),
    lambda x: TF.rotate(x, angle=5,  fill=0),
    lambda x: TF.rotate(x, angle=-5, fill=0),
]


@torch.no_grad()
def predict_tta(model: nn.Module, imgs: torch.Tensor,
                cfg: "Config", n_tta: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-augmentation TTA: original, h-flip, ±5° rotation (4 passes by default).

    All augmentations keep spatial dimensions at IMAGE_SIZE so the transformer's
    fixed pos-embed (196 patch tokens) stays valid throughout.
    Set n_tta=2 to fall back to original + h-flip only.
    """
    core = model.module if hasattr(model, "module") else model
    prob_sum = torch.zeros(imgs.size(0), 1, device=cfg.DEVICE)
    sev_sum  = torch.zeros(imgs.size(0), 1, device=cfg.DEVICE)

    augs = _TTA_FNS[:n_tta]
    for aug_fn in augs:
        x = aug_fn(imgs.clone())
        logit, sev = core(x)
        prob_sum += torch.sigmoid(logit)
        sev_sum  += sev * cfg.SEVERITY_MAX

    return prob_sum / len(augs), sev_sum / len(augs)


def _resolve_fold_ckpt(fold: int, cfg: "Config") -> Path:
    tier1_ckpt = cfg.CKPT_DIR / f"fold{fold}_tier1_reg.pt"
    if tier1_ckpt.exists():
        return tier1_ckpt
    return cfg.CKPT_DIR / f"fold{fold}_best.pt"


def load_fold_model(fold: int, cfg: "Config") -> nn.Module:
    ckpt_path = _resolve_fold_ckpt(fold, cfg)
    _model = TBMTNet(cfg).to(cfg.DEVICE)
    ckpt   = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
    state_dict = ckpt["model"]
    state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
    _model.load_state_dict(state_dict)
    _model.eval()
    return _model


def load_tier1_calibration(cfg: "Config") -> dict | None:
    path = cfg.CKPT_DIR / "tier1_calibration.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def ensemble_predict(df: pd.DataFrame, cfg: "Config",
                     n_tta: int = 2) -> pd.DataFrame:
    """
    Run all N_FOLDS models with TTA; average predictions.
    Returns df with columns: image_path, tb_prob, timika_score
    """
    ds     = TBXDataset(df, cfg, train=False)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)

    fold_probs = []
    fold_sevs  = []

    for fold in range(cfg.N_FOLDS):
        ckpt = _resolve_fold_ckpt(fold, cfg)
        if not ckpt.exists():
            print(f"  [WARN] Fold {fold} checkpoint not found — skipping")
            continue

        print(f"  Running fold {fold+1}/{cfg.N_FOLDS} …", end=" ", flush=True)
        _model = load_fold_model(fold, cfg)

        probs_list, sevs_list = [], []
        for batch in loader:
            imgs = batch["image"].to(cfg.DEVICE)
            if cfg.USE_AMP:
                with autocast("cuda", dtype=getattr(torch, cfg.AMP_DTYPE)):
                    prob, sev = predict_tta(_model, imgs, cfg, n_tta)
            else:
                prob, sev = predict_tta(_model, imgs, cfg, n_tta)
            probs_list.append(prob.cpu())
            sevs_list.append(sev.cpu())

        fold_probs.append(torch.cat(probs_list).numpy().ravel())
        fold_sevs.append(torch.cat(sevs_list).numpy().ravel())
        print("done")

    if not fold_probs:
        raise RuntimeError("No fold checkpoints found.")

    # Mean ensemble
    ens_probs_raw = np.stack(fold_probs, axis=0).mean(axis=0)
    ens_sevs_raw  = np.stack(fold_sevs,  axis=0).mean(axis=0)

    # 1. Calibrate Timika if Tier-1 calibration is available
    calib = load_tier1_calibration(cfg)
    if calib is not None:
        slope = float(calib.get("slope", 1.0))
        intercept = float(calib.get("intercept", 0.0))
        ens_sevs_raw = ens_sevs_raw * slope + intercept

    # 2. Recalibrate probabilities to undo POS_WEIGHT inflation
    # Math: p = q / (w * (1 - q) + q)
    w = cfg.POS_WEIGHT
    ens_probs_calibrated = ens_probs_raw / (w * (1 - ens_probs_raw) + ens_probs_raw)

    result_df = df[["image_path", "patient_id", "tb_label"]].copy()
    result_df["tb_prob"]      = ens_probs_calibrated
    result_df["tb_pred"]      = (ens_probs_calibrated >= 0.5).astype(int)
    
    # 3. Fix Timika Score (Model wasn't trained on normal lungs, so force to 0 if Normal)
    result_df["timika_score"] = ens_sevs_raw.clip(0, 140) * result_df["tb_pred"]
    return result_df


def save_submission(result_df: pd.DataFrame, cfg: "Config") -> None:
    sub = result_df[["image_path", "tb_prob", "timika_score"]].copy()
    sub["image_id"] = sub["image_path"].apply(lambda p: Path(p).stem)
    sub = sub[["image_id", "tb_prob", "timika_score"]]
    sub.to_csv(cfg.BASE / "submission.csv", index=False)
    print(f"submission.csv saved → {cfg.BASE / 'submission.csv'}")
    print(sub.head(10))


def print_summary(result_df: pd.DataFrame) -> None:
    pos = (result_df["tb_pred"] == 1).sum()
    neg = (result_df["tb_pred"] == 0).sum()
    print("\n" + "="*50)
    print("  ENSEMBLE INFERENCE SUMMARY")
    print("="*50)
    print(f"  Total images    : {len(result_df)}")
    print(f"  Predicted TB+   : {pos}  ({100*pos/len(result_df):.1f}%)")
    print(f"  Predicted TB-   : {neg}  ({100*neg/len(result_df):.1f}%)")
    if result_df["timika_score"].notna().any():
        s = result_df.loc[result_df["tb_pred"]==1, "timika_score"]
        print(f"  Timika (TB+)    : mean={s.mean():.1f}  "
              f"median={s.median():.1f}  "
              f"range=[{s.min():.1f}, {s.max():.1f}]")

    # GradCAM visualisation on top-5 uncertain cases
    result_df["uncertainty"] = (result_df["tb_prob"] - 0.5).abs()
    uncertain_idx = result_df["uncertainty"].nsmallest(5).index.tolist()
    print(f"\n  Top-5 most uncertain cases (prob closest to 0.5):")
    print(result_df.loc[uncertain_idx,
                        ["image_id" if "image_id" in result_df.columns
                         else "image_path", "tb_prob", "timika_score"]].to_string())


# ── Run inference on full dataset ─────────────────────────────────────
print("Running 5-fold TTA ensemble inference …")
result_df = ensemble_predict(labels_df, CFG, n_tta=4)   # 4 TTA passes: orig, h-flip, ±5° rot
print_summary(result_df)
save_submission(result_df, CFG)
