# TB-MTNet: Multi-Task CNN-Transformer for Tuberculosis Detection & Severity Scoring

A deep learning system that takes a chest X-ray and does two things at once: flags whether tuberculosis is present, and estimates how severe it is using the clinically-recognized Timika scoring system. Built as a multi-task hybrid CNN-Transformer trained across three public TB X-ray datasets with a custom 3-tier severity labeling pipeline to handle the fact that most TB datasets only have diagnosis labels, not severity scores.

## Why this exists

Most public TB chest X-ray datasets give you a binary label (TB / no TB) but not a severity score, even though severity is what actually drives clinical decisions. This project builds a labeling pipeline to recover severity signal from heterogeneous data (pixel-level annotations where available, bounding boxes where not, model-generated estimates as a last resort) and trains a single model to predict both diagnosis and severity jointly.

## Results

| Metric | Target | Paper Projection |
|---|---|---|
| AUROC (TB detection) | ≥ 0.92 | 0.93–0.95 |
| Sensitivity | ≥ 0.90 | ~0.92 |
| Specificity | ≥ 0.80 | ~0.85 |
| MAE (Timika severity) | ≤ 15 | 12–14 |
| Pearson r (severity) | ≥ 0.75 | 0.75–0.85 |

## How it works

**Architecture (~25.4M params):** An Inception-v3 backbone extracts features from 512×512 chest X-rays, refined through an efficient channel-attention module (ECA) and a lightweight projection bridge down to a 96-dim embedding space. That feature map is flattened into patch tokens, fed through a 4-layer Transformer encoder, and the resulting CLS token branches into two heads: a sigmoid classifier for TB probability and a scaled sigmoid regressor for the 0–140 Timika severity score.

**Severity labels (3-tier pipeline):** Since no single dataset has complete severity ground truth, labels are assembled in tiers — exact lesion-area calculations from pixel-level annotations where they exist (~336 cases), calibrated estimates from bounding-box datasets (~1,200 cases), and teacher-model-generated pseudo-labels (via Grad-CAM++) for the remainder.

**Training:** Three-stage schedule — warm up the classification head with the regression head frozen, then jointly fine-tune both heads using uncertainty-weighted multi-task loss (Kendall, Gal & Cipolla, 2018) so the model learns to balance the two objectives automatically, then a final fine-tune stage with stochastic weight averaging. Evaluated with 5-fold stratified group cross-validation, grouped by patient ID to prevent data leakage.

## Datasets

| Dataset | Purpose |
|---|---|
| Shenzhen Hospital CXR set | 662 X-rays, source of Tier-1 pixel-level severity annotations |
| Montgomery County CXR set | 138 X-rays with lung segmentation masks |
| TBX11K | 11,200 X-rays with bounding-box annotations |
| (optional) NIAID TB Portal subset | +7,000 X-rays for additional training data |

## Running it

This was built to run on Kaggle's free GPU tier (T4 ×2), since the full training run takes 15–17 hours.

1. Create a Kaggle notebook with the GPU T4 ×2 accelerator and internet access enabled
2. Add the datasets listed above via "Add Data"
3. Upload `tb_mtnet.ipynb` and run all cells (checkpointing means you can resume across sessions)

For local development, the model code is split into modular cell files under `cells/` (data pipeline, lung segmentation, severity labeling, model definition, training loop, evaluation, inference) and `assemble.py` stitches them back into the notebook — useful if you want to read or modify the logic without scrolling through a single giant notebook.

```
pip install -r requirements.txt
```

## Project structure

```
tb_mtnet.ipynb          ← main notebook (run this on Kaggle)
requirements.txt
assemble.py              ← regenerates the notebook from cells/
cells/
  cell1_setup.py         ← config, seeding, imports
  cell2_csv.py            ← unified dataset CSV builder
  cell3_lung.py            ← lung segmentation (U-Net)
  cell4_severity.py         ← 3-tier severity labeling
  cell5_dataset.py            ← PyTorch Dataset + augmentations
  cell6_model.py                ← ECA + TB-MTNet + multi-task loss
  cell7_train.py                  ← 3-stage training loop, 5-fold CV
  cell8_eval.py                     ← OOF evaluation, ROC, reliability plots
  cell9_infer.py                     ← TTA + ensemble inference
```

A full write-up of the methodology and results is in the included PDF/LaTeX report.

## References

- Liu et al., TBX11K dataset — CVPR 2020 / IEEE TPAMI 2023
- Ralph et al., Timika severity score — Thorax 2010
- Kantipudi et al., Automated Timika scoring — J. Imaging Inform. Med. 2024
- Kendall, Gal & Cipolla, Multi-task uncertainty weighting — CVPR 2018
- Wang et al., ECA-Net — CVPR 2020
- Yang et al., Shenzhen abnormality annotations — MDPI Data 7(7):95, 2022
