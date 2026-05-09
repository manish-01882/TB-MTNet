#!/usr/bin/env python3
"""Assemble individual cell files into tb_mtnet.ipynb"""
import json
from pathlib import Path

ROOT = Path(__file__).parent

def code_cell(src: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

def md_cell(src: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src,
    }

cells = []

# ── Title markdown ────────────────────────────────────────────────────
cells.append(md_cell(
    "# TB-MTNet: Multi-Task Hybrid CNN-Transformer\n"
    "## TB Detection + Timika Severity Regression on Chest X-Rays\n\n"
    "**Architecture:** Inception-v3 → ECA → 2048→512→96 bridge → 4-layer Transformer → dual heads  \n"
    "**Targets:** AUROC ≥ 0.92 | MAE ≤ 14 Timika points  \n"
    "**Compute:** Kaggle T4 ×2 | fp16 | batch 32 / 512²  \n\n"
    "### Required Kaggle datasets (Add Data → search each slug):\n"
    "```\n"
    "raddar/tuberculosis-chest-xrays-shenzhen\n"
    "yoctoman/shcxr-lung-mask\n"
    "raddar/tuberculosis-chest-xrays-montgomery\n"
    "vbookshelf/tbx11k-simplified\n"
    "(optional) tawsifurrahman/tuberculosis-tb-chest-xray-dataset\n"
    "```\n\n"
    "### Training schedule:\n"
    "| Stage | Epochs | What trains |\n"
    "|-------|--------|-------------|\n"
    "| 1 | 10 | Trunk + Transformer + cls head (reg head frozen) |\n"
    "| 2 | 20 | Full multi-task (Kendall–Gal–Cipolla uncertainty weighting) |\n"
    "| 3 | 10 | Fine-tune + SWA last 5 epochs |\n",
    "cell-title",
))

# ── Code cells ────────────────────────────────────────────────────────
cell_files = [
    ("cells/cell1_setup.py",    "cell-1-setup",    "## Cell 1 — Imports · Seed · Config"),
    ("cells/cell2_csv.py",      "cell-2-csv",      "## Cell 2 — Dataset CSV Builder"),
    ("cells/cell3_lung.py",     "cell-3-lung",     "## Cell 3 — Lung Segmentation (U-Net)"),
    ("cells/cell4_severity.py", "cell-4-severity", "## Cell 4 — Severity Labels (3-Tier Pipeline)"),
    ("cells/cell5_dataset.py",  "cell-5-dataset",  "## Cell 5 — PyTorch Dataset + Transforms"),
    ("cells/cell6_model.py",    "cell-6-model",    "## Cell 6 — TB-MTNet Architecture + Loss"),
    ("cells/cell7_train.py",    "cell-7-train",    "## Cell 7 — 3-Stage Training (5-Fold CV)"),
    ("cells/cell8_eval.py",     "cell-8-eval",     "## Cell 8 — OOF Evaluation + Plots"),
    ("cells/cell9_infer.py",    "cell-9-infer",    "## Cell 9 — TTA + Ensemble Inference"),
]

for fname, cell_id, header in cell_files:
    src = (ROOT / fname).read_text()
    cells.append(md_cell(header, f"md-{cell_id}"))
    cells.append(code_cell(src, cell_id))

# ── Notebook JSON ─────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.14",
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "acceleratorCount": 2,
            "dataSources": [
                {"sourceType": "datasetVersion",
                 "sourceId": "raddar/tuberculosis-chest-xrays-shenzhen"},
                {"sourceType": "datasetVersion",
                 "sourceId": "yoctoman/shcxr-lung-mask"},
                {"sourceType": "datasetVersion",
                 "sourceId": "raddar/tuberculosis-chest-xrays-montgomery"},
                {"sourceType": "datasetVersion",
                 "sourceId": "vbookshelf/tbx11k-simplified"},
            ],
            "isInternetEnabled": True,
            "isGpuEnabled": True,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "tb_mtnet.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Created: {out}")
print(f"Cells  : {len(cells)}")
print(f"Size   : {out.stat().st_size / 1024:.1f} KB")
