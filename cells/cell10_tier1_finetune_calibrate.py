"""Cell 10 – Tier-1 reg-head fine-tune + OOF linear calibration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from cells.cell1_setup import CFG
from cells.cell4_severity import compute_tier1
from cells.cell5_dataset import TBXDataset
from cells.cell6_model import TBMTNet, pearson_loss


RUN_TIER1_FINETUNE = False  # set True to run the full pipeline
N_SPLITS = 5
EPOCHS = 6
LR = 3e-4


def build_tier1_df(labels_df: pd.DataFrame, cfg: "Config") -> pd.DataFrame:
    df = labels_df.copy()
    df = compute_tier1(df, cfg)
    tier1 = df[
        (df["source"] == "shenzhen") &
        (df["tb_label"] == 1) &
        (df["has_severity"] == 1)
    ].copy()
    if tier1.empty:
        raise RuntimeError("No Tier-1 Shenzhen TB+ samples found.")

    tier1["sev_quartile"] = pd.qcut(
        tier1["severity"], q=4, labels=False, duplicates="drop"
    )
    tier1["strat_key"] = tier1["sev_quartile"].astype(str)
    return tier1


class RegOnlyLoss(nn.Module):
    def __init__(self, huber_beta: float):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_beta, reduction="none")

    def forward(self, pred, target, mask):
        huber_elem = self.huber(pred, target)
        n_valid = mask.sum().clamp(min=1.0)
        l_huber = (huber_elem * mask).sum() / n_valid
        l_pearson = pearson_loss(pred, target, mask)
        return l_huber + 0.5 * l_pearson, l_huber, l_pearson


def _freeze_for_reg_head(model: nn.Module) -> None:
    core = model.module if hasattr(model, "module") else model
    for p in core.parameters():
        p.requires_grad_(False)
    for p in core.reg_head.parameters():
        p.requires_grad_(True)


def _make_loaders(df: pd.DataFrame, train_idx, val_idx, cfg: "Config"):
    train_ds = TBXDataset(df.iloc[train_idx], cfg, train=True)
    val_ds = TBXDataset(df.iloc[val_idx], cfg, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


@torch.no_grad()
def _eval_regression(model: nn.Module, loader: DataLoader, cfg: "Config"):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        imgs = batch["image"].to(cfg.DEVICE)
        y_sev = batch["severity"].to(cfg.DEVICE)
        s_msk = batch["severity_mask"].to(cfg.DEVICE)
        logit, sev = model(imgs)
        if s_msk.sum().item() == 0:
            continue
        preds.append((sev * cfg.SEVERITY_MAX).cpu())
        targets.append((y_sev * cfg.SEVERITY_MAX).cpu())

    if not preds:
        return float("inf"), 0.0
    p = torch.cat(preds).numpy().ravel()
    t = torch.cat(targets).numpy().ravel()
    mae = float(np.abs(p - t).mean())
    pr = float(pearsonr(p, t)[0]) if len(p) > 1 else 0.0
    return mae, pr


def fine_tune_reg_head(model: nn.Module, train_loader, val_loader, cfg: "Config",
                        epochs: int = 5, lr: float = 3e-4):
    _freeze_for_reg_head(model)
    core = model.module if hasattr(model, "module") else model
    opt = torch.optim.AdamW(core.reg_head.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = RegOnlyLoss(cfg.HUBER_BETA).to(cfg.DEVICE)
    scaler = GradScaler("cuda") if (cfg.USE_AMP and cfg.AMP_DTYPE == "float16") else None
    amp_dtype = getattr(torch, cfg.AMP_DTYPE)

    best_mae = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(cfg.DEVICE)
            y_sev = batch["severity"].to(cfg.DEVICE)
            s_msk = batch["severity_mask"].to(cfg.DEVICE)

            opt.zero_grad(set_to_none=True)
            if cfg.USE_AMP:
                with autocast("cuda", dtype=amp_dtype):
                    _, sev = model(imgs)
                    loss, _, _ = loss_fn(sev, y_sev, s_msk)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(core.reg_head.parameters(), cfg.GRAD_CLIP)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(core.reg_head.parameters(), cfg.GRAD_CLIP)
                    opt.step()
            else:
                _, sev = model(imgs)
                loss, _, _ = loss_fn(sev, y_sev, s_msk)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core.reg_head.parameters(), cfg.GRAD_CLIP)
                opt.step()

            total_loss += loss.item()

        if val_loader is not None:
            mae, pr = _eval_regression(model, val_loader, cfg)
            print(f"  FT ep{ep+1:02d}  loss={total_loss/len(train_loader):.4f}  "
                  f"MAE={mae:.2f}  r={pr:.3f}")
            if mae < best_mae:
                best_mae = mae
                best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}

    if best_state is not None:
        core.load_state_dict(best_state)
    return best_mae


@torch.no_grad()
def predict_severity(model: nn.Module, loader: DataLoader, cfg: "Config") -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        imgs = batch["image"].to(cfg.DEVICE)
        _, sev = model(imgs)
        preds.append((sev * cfg.SEVERITY_MAX).cpu())
    return torch.cat(preds).numpy().ravel()


def run_tier1_oof(tier1: pd.DataFrame, cfg: "Config",
                  base_ckpt: str | None = None,
                  n_splits: int = 5, epochs: int = 6, lr: float = 3e-4):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.SEED)
    X = np.arange(len(tier1))
    y = tier1["strat_key"].values
    groups = tier1["patient_id"].values

    preds = np.zeros(len(tier1))
    trues = tier1["severity"].values.astype(np.float32)

    ckpt_path = base_ckpt or str(cfg.CKPT_DIR / "fold0_best.pt")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n[Fold {fold+1}/{n_splits}] Tier-1 fine-tune")
        model = TBMTNet(cfg).to(cfg.DEVICE)
        ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        state_dict = ckpt["model"]
        state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        train_loader, val_loader = _make_loaders(tier1, train_idx, val_idx, cfg)
        fine_tune_reg_head(model, train_loader, val_loader, cfg, epochs=epochs, lr=lr)
        preds[val_idx] = predict_severity(model, val_loader, cfg)

    return preds, trues


def fit_linear_calibration(preds: np.ndarray, trues: np.ndarray) -> LinearRegression:
    reg = LinearRegression()
    reg.fit(preds.reshape(-1, 1), trues)
    return reg


def apply_linear_calibration(preds: np.ndarray, reg: LinearRegression) -> np.ndarray:
    return reg.predict(preds.reshape(-1, 1))


def save_calibration(reg: LinearRegression, out_path: Path) -> None:
    out = {
        "slope": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def finetune_full_tier1(
    tier1: pd.DataFrame,
    cfg: "Config",
    out_suffix: str = "tier1_reg",
    base_ckpt: str | None = None,
    epochs: int = 6,
    lr: float = 3e-4,
) -> None:
    full_loader = DataLoader(
        TBXDataset(tier1, cfg, train=True),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    for fold in range(cfg.N_FOLDS):
        ckpt_path = base_ckpt or str(cfg.CKPT_DIR / f"fold{fold}_best.pt")
        print(f"\n[Fold {fold+1}/{cfg.N_FOLDS}] Tier-1 full fine-tune")
        model = TBMTNet(cfg).to(cfg.DEVICE)
        ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        state_dict = ckpt["model"]
        state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        fine_tune_reg_head(model, full_loader, None, cfg, epochs=epochs, lr=lr)

        core = model.module if hasattr(model, "module") else model
        out_path = cfg.CKPT_DIR / f"fold{fold}_{out_suffix}.pt"
        torch.save({"model": core.state_dict()}, out_path)
        print(f"  Saved → {out_path}")


# ── Run ───────────────────────────────────────────────────────────────
if RUN_TIER1_FINETUNE:
    tier1_df = build_tier1_df(labels_df, CFG)
    oof_pred, oof_true = run_tier1_oof(
        tier1_df, CFG, base_ckpt=None, n_splits=N_SPLITS, epochs=EPOCHS, lr=LR
    )

    reg = fit_linear_calibration(oof_pred, oof_true)
    cal_pred = apply_linear_calibration(oof_pred, reg)

    mae_raw = float(np.abs(oof_pred - oof_true).mean())
    mae_cal = float(np.abs(cal_pred - oof_true).mean())
    print("\nCalibration results")
    print(f"  Linear: y = {reg.coef_[0]:.3f} * x + {reg.intercept_:.3f}")
    print(f"  MAE raw: {mae_raw:.2f}  |  MAE calibrated: {mae_cal:.2f}")

    cal_path = CFG.CKPT_DIR / "tier1_calibration.json"
    save_calibration(reg, cal_path)
    print(f"Saved calibration → {cal_path}")

    finetune_full_tier1(tier1_df, CFG, out_suffix="tier1_reg", epochs=EPOCHS, lr=LR)
