# TB-MTNet: Multi-Task Hybrid CNN-Transformer
### TB Detection + Timika Severity Regression on Chest X-Rays

---

## Quick Start on Kaggle

1. **Create a new Kaggle notebook** (GPU T4 ×2 accelerator)
2. **Add these datasets** via the sidebar → "Add Data":

| Slug | Purpose |
|------|---------|
| `raddar/tuberculosis-chest-xrays-shenzhen` | 662 CXRs, Tier-1 severity source |
| `yoctoman/shcxr-lung-mask` | Shenzhen lung masks |
| `raddar/tuberculosis-chest-xrays-montgomery` | 138 CXRs + L/R lung masks |
| `vbookshelf/tbx11k-simplified` | 11,200 CXRs + COCO bboxes |
| *(optional)* `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | 7,000 extra CXRs |

3. **Upload `tb_mtnet.ipynb`** to the notebook
4. **Enable Internet** (for first-run pip installs + NIH annotation download)
5. **Run all cells** — expect ~15–17 hrs across 2 sessions (checkpoint resume built-in)

---

## Architecture (TB-MTNet, ~25.4M params)

```
Input (B, 3, 512, 512)
  │
  ▼
Inception-v3 (timm, features_only, out_indices=(4,))
  → (B, 2048, 14, 14)
  │
  ▼
ECA channel attention (k=5, ~5 params)
  │
  ▼
Projection bridge:  Conv2d(2048→512)→BN→GELU→Conv2d(512→96)→BN→GELU
  → (B, 96, 14, 14)
  │
  ▼
Flatten → (B, 196, 96) + [CLS] token + learned pos-embed → (B, 197, 96)
  │
  ▼
Transformer encoder: 4 layers, d=96, heads=4, FFN=384, GELU, pre-LayerNorm
  │
  ▼
LayerNorm → CLS token → (B, 96)
  │
  ├─► Linear(96→1) → sigmoid        [TB probability]
  └─► Linear(96→1) → sigmoid × 140  [Timika score 0–140]
```

---

## Severity Label Pipeline (3 Tiers)

| Tier | Source | Method | Coverage |
|------|--------|--------|----------|
| 1 | Shenzhen polygon annotations (Yang et al. 2022) | Exact ALP + cavity flag from JSON masks | ~336 cases |
| 2 | TBX11K COCO bounding boxes | bbox-fill pseudo-mask → linear calibration (slope ~1.5) | ~1,200 cases |
| 3 | Unannotated TB+ | Grad-CAM++ on DenseNet-121 teacher + morphological refinement | Remainder |

**Timika formula:** `S = 100 × (A_lesion ∩ A_lung) / A_lung + 40 × cavity_present`

---

## Training Schedule

| Stage | Epochs | What trains | LR |
|-------|--------|-------------|-----|
| 1 | 10 | Trunk + Transformer + cls head (reg head **frozen**) | 3e-4 |
| 2 | 20 | Full multi-task — Kendall–Gal–Cipolla uncertainty weighting | trunk: 3e-4, reg: 3e-5 |
| 3 | 10 | Fine-tune + SWA (last 5 epochs) | 1e-5 |

- **Loss:** `L = exp(-s_c)·L_BCE + exp(-s_r)·L_Huber + s_c + s_r`
- **CV:** `StratifiedGroupKFold(5)` on patient ID (prevents leakage)
- **AMP:** fp16 — peak ~10–11 GB VRAM per T4 at batch 32 / 512²

---

## Expected Results

| Metric | Target | Paper Projection |
|--------|--------|-----------------|
| AUROC (TB) | ≥ 0.92 | 0.93–0.95 |
| Sensitivity | ≥ 0.90 | ~0.92 |
| Specificity | ≥ 0.80 | ~0.85 |
| MAE (Timika) | ≤ 15 | 12–14 |
| Pearson r | ≥ 0.75 | 0.75–0.85 |

---

## File Structure

```
tb_model/
├── tb_mtnet.ipynb          ← Main Kaggle notebook (upload this)
├── requirements.txt
├── assemble.py             ← Regenerates .ipynb from cell files
└── cells/
    ├── cell1_setup.py      ← Imports, seed, Config dataclass
    ├── cell2_csv.py        ← Unified CSV builder (all datasets)
    ├── cell3_lung.py       ← Lung U-Net (train + predict masks)
    ├── cell4_severity.py   ← 3-tier Timika severity labels
    ├── cell5_dataset.py    ← PyTorch Dataset + Albumentations
    ├── cell6_model.py      ← ECA + TB-MTNet + MultiTaskLoss
    ├── cell7_train.py      ← 3-stage training loop (5-fold CV)
    ├── cell8_eval.py       ← OOF evaluation + ROC + reliability plots
    └── cell9_infer.py      ← TTA + 5-fold ensemble → submission.csv
```

---

## References

- Liu et al. TBX11K dataset, CVPR 2020 / IEEE TPAMI 2023
- Ralph et al. Timika score, Thorax 2010
- Kantipudi et al. Automated Timika scoring, J. Imaging Inform. Med. 2024
- Kendall, Gal & Cipolla. Multi-task uncertainty weighting, CVPR 2018
- Wang et al. ECA-Net, CVPR 2020
- Yang et al. Shenzhen abnormality annotations, MDPI Data 7(7):95, 2022
