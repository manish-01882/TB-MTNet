"""Cell 8 – OOF Evaluation: AUROC, sensitivity/specificity/F1, MAE/RMSE/Pearson."""

def youden_threshold(fpr, tpr, thresholds):
    """Optimal threshold by Youden's J = TPR - FPR."""
    j = tpr - fpr
    return thresholds[np.argmax(j)]


def bootstrap_auroc(y_true, y_score, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


@torch.no_grad()
def predict_fold(fold: int, df: pd.DataFrame, val_idx, cfg: "Config"):
    """Load best checkpoint for a fold and run inference on val set."""
    ckpt_path = cfg.CKPT_DIR / f"fold{fold}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    _model = TBMTNet(cfg).to(cfg.DEVICE)
    ckpt   = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
    state_dict = ckpt["model"]
    state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
    _model.load_state_dict(state_dict)
    _model.eval()

    ds     = TBXDataset(df.iloc[val_idx], cfg, train=False)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)

    probs_list, labels_list, sev_pred_list, sev_true_list, sev_mask_list = \
        [], [], [], [], []
    core = _model.module if hasattr(_model, "module") else _model

    for batch in loader:
        imgs  = batch["image"].to(cfg.DEVICE)
        logit, sev = core(imgs)
        probs_list.append(torch.sigmoid(logit).cpu())
        labels_list.append(batch["tb_label"])
        sev_pred_list.append(sev.cpu() * cfg.SEVERITY_MAX)
        sev_true_list.append(batch["severity"] * cfg.SEVERITY_MAX)
        sev_mask_list.append(batch["severity_mask"])

    return {
        "probs":    torch.cat(probs_list).numpy().ravel(),
        "labels":   torch.cat(labels_list).numpy().ravel().astype(int),
        "sev_pred": torch.cat(sev_pred_list).numpy().ravel(),
        "sev_true": torch.cat(sev_true_list).numpy().ravel(),
        "sev_mask": torch.cat(sev_mask_list).numpy().ravel().astype(bool),
    }


def evaluate_oof(df: pd.DataFrame, fold_results: List[dict], cfg: "Config"):
    """Aggregate OOF predictions across all folds and compute metrics."""
    cv = StratifiedGroupKFold(n_splits=cfg.N_FOLDS, shuffle=True,
                              random_state=cfg.SEED)
    X      = np.arange(len(df))
    y      = df["strat_key"].values
    groups = df["patient_id"].values

    all_probs  = np.zeros(len(df))
    all_labels = np.zeros(len(df), dtype=int)
    all_sp     = np.zeros(len(df))
    all_st     = np.zeros(len(df))
    all_sm     = np.zeros(len(df), dtype=bool)

    for fold, (_, val_idx) in enumerate(cv.split(X, y, groups)):
        preds = predict_fold(fold, df, val_idx, cfg)
        all_probs[val_idx]  = preds["probs"]
        all_labels[val_idx] = preds["labels"]
        all_sp[val_idx]     = preds["sev_pred"]
        all_st[val_idx]     = preds["sev_true"]
        all_sm[val_idx]     = preds["sev_mask"]

    # ── Classification metrics ────────────────────────────────────────
    auroc         = roc_auc_score(all_labels, all_probs)
    ci_lo, ci_hi  = bootstrap_auroc(all_labels, all_probs)
    fpr, tpr, thr = roc_curve(all_labels, all_probs)
    best_thr      = youden_threshold(fpr, tpr, thr)
    preds_bin     = (all_probs >= best_thr).astype(int)
    cm            = confusion_matrix(all_labels, preds_bin)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    f1   = f1_score(all_labels, preds_bin)

    # Partial AUROC (specificity >= 0.70)
    fpr_thresh = 1 - 0.70
    mask_spec  = fpr <= fpr_thresh
    p_auroc    = np.trapz(tpr[mask_spec], fpr[mask_spec]) / fpr_thresh \
                 if mask_spec.sum() > 1 else 0.0

    print("\n" + "="*55)
    print("  OOF Classification Results")
    print("="*55)
    print(f"  AUROC          : {auroc:.4f}  (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  Partial AUROC  : {p_auroc:.4f}  (spec >= 0.70 region)")
    print(f"  Threshold (J)  : {best_thr:.4f}")
    print(f"  Sensitivity    : {sens:.4f}")
    print(f"  Specificity    : {spec:.4f}")
    print(f"  F1             : {f1:.4f}")
    print(f"  Confusion Mat  :\n    TN={tn} FP={fp}\n    FN={fn} TP={tp}")

    # ── Severity metrics ──────────────────────────────────────────────
    sv_pred = all_sp[all_sm]; sv_true = all_st[all_sm]
    if len(sv_pred) > 0:
        mae  = float(np.abs(sv_pred - sv_true).mean())
        rmse = float(np.sqrt(((sv_pred - sv_true)**2).mean()))
        pr, _ = pearsonr(sv_pred, sv_true)
        sp_r, _ = spearmanr(sv_pred, sv_true)
        print("\n  OOF Severity Results")
        print("="*55)
        print(f"  MAE            : {mae:.2f}  (target < 15)")
        print(f"  RMSE           : {rmse:.2f}")
        print(f"  Pearson r      : {pr:.4f}  (target > 0.75)")
        print(f"  Spearman rho   : {sp_r:.4f}")
        print(f"  N severity cases: {len(sv_pred)}")
    else:
        print("\n  [WARN] No valid severity cases for regression metrics.")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, lw=2, color="#4F46E5",
            label=f"AUROC = {auroc:.4f} [{ci_lo:.3f}–{ci_hi:.3f}]")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.axvline(1-0.70, color="grey", ls=":", label="spec=0.70 (WHO TPP)")
    ax.scatter([1-spec],[sens], color="red", zorder=5, label=f"Youden J ({best_thr:.3f})")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve (OOF)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Reliability diagram (severity)
    ax = axes[1]
    if len(sv_pred) > 5:
        bins = np.linspace(0, 140, 11)
        bin_centres, bin_means, bin_counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (sv_true >= lo) & (sv_true < hi)
            if mask.sum() > 0:
                bin_centres.append((lo + hi) / 2)
                bin_means.append(sv_pred[mask].mean())
                bin_counts.append(mask.sum())
        ax.scatter(bin_centres, bin_means, s=[c*3 for c in bin_counts],
                   color="#10B981", alpha=0.8, label="Pred mean per bin")
        ax.plot([0,140],[0,140],"k--", lw=1, label="Perfect calibration")
        ax.set(xlabel="True Timika", ylabel="Predicted Timika",
               title="Severity Reliability Diagram", xlim=(0,140), ylim=(0,140))
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.text(5, 130, f"MAE={mae:.1f}  r={pr:.3f}", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Insufficient severity data", ha="center",
                transform=ax.transAxes)

    plt.tight_layout()
    plot_path = CFG.BASE / "oof_evaluation.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\nPlot saved → {plot_path}")


evaluate_oof(labels_df, fold_results, CFG)
