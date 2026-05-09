# !pip install -q timm albumentations segmentation-models-pytorch grad-cam

import os, json, math, random, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR

import timm
from timm.scheduler import CosineLRScheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SEED = 42
def seed_everything(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
seed_everything()

@dataclass
class Config:
    SEED:              int  = 42
    SHENZHEN_DIR:      Path = Path("/kaggle/input/tuberculosis-chest-xrays-shenzhen")
    SHENZHEN_MASK_DIR: Path = Path("/kaggle/input/shcxr-lung-mask")
    MONTGOMERY_DIR:    Path = Path("/kaggle/input/tuberculosis-chest-xrays-montgomery")
    TBX11K_DIR:        Path = Path("/kaggle/input/tbx11k-simplified")
    RAHMAN_DIR:        Path = Path("/kaggle/input/tuberculosis-tb-chest-xray-dataset")
    SHEN_ANNOT_JSON:   Path = Path("/kaggle/input/datasets/manishchoudhary9/shenzen-polygon-annotations/Annotations_AllinOne_json.json")
    BASE:       Path = Path("/kaggle/working")
    CKPT_DIR:   Path = Path("/kaggle/working/checkpoints")
    LABELS_CSV: Path = Path("/kaggle/working/labels.csv")
    LUNG_CKPT:  Path = Path("/kaggle/working/checkpoints/lung_unet.pt")
    IMAGE_SIZE: int   = 512
    SEG_SIZE:   int   = 256
    D_MODEL:    int   = 96
    N_HEADS:    int   = 4
    N_LAYERS:   int   = 4
    FFN_DIM:    int   = 384
    DROPOUT:    float = 0.1
    # ── Multi-GPU (T4 x2) settings ──────────────────────────────────
    NUM_GPUS:     int   = torch.cuda.device_count()          # 2 on T4 x2
    # 48 samples per GPU × 2 GPUs = 96 effective batch size
    # T4 has 16 GB VRAM; Inception-v3 + Transformer at 512^2 fits comfortably
    BATCH_SIZE:   int   = 96
    # 3 workers per GPU → 6 total; extra workers saturate CLAHE+ElasticTransform
    NUM_WORKERS:  int   = 6
    N_FOLDS:      int   = 5                                  # Full 5-fold CV
    POS_WEIGHT:   float = 5.95
    HUBER_BETA:   float = 0.1
    SEVERITY_MAX: float = 140.0
    GRAD_CLIP:    float = 1.0
    WEIGHT_DECAY: float = 1e-4
    SEG_EPOCHS:   int   = 10                                 # U-Net lung seg (proper)
    # LRs scaled by √2 ≈ 1.41 relative to single-GPU baseline (batch 32→64)
    # ── Epoch schedule: proper full-training run ───────────────────────
    # S1: classification warmup  – 10 eps is sufficient, cls AUROC already 0.98
    # S2: full multi-task        – 15 eps to push severity Pearson r → 0.75+
    # S3: fine-tune + SWA        – 10 eps; SWA kicks in last 3 eps for stability
    S1_EPOCHS: int=10; S1_LR: float=4e-4
    S2_EPOCHS: int=15; S2_LR: float=4e-4; S2_REG_LR: float=4e-5
    S3_EPOCHS: int=10; S3_LR: float=1.4e-5; SWA_START: int=3
    WARMUP_EPOCHS: int=2; CYCLE_DECAY: float=0.5; ETA_MIN: float=1e-6
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True
    # bfloat16: natively faster on T4 than fp16, no loss scaling needed,
    # and more stable for the regression head's small gradients.
    AMP_DTYPE: str = "bfloat16"

    def __post_init__(self):
        self.CKPT_DIR.mkdir(parents=True, exist_ok=True)

CFG = Config()
print(f"Device : {CFG.DEVICE} | GPUs: {CFG.NUM_GPUS} | AMP: {CFG.USE_AMP}")
print(f"Batch  : {CFG.BATCH_SIZE} ({CFG.BATCH_SIZE // max(CFG.NUM_GPUS,1)} per GPU) | Workers: {CFG.NUM_WORKERS}")
print(f"Image  : {CFG.IMAGE_SIZE}^2 | Folds: {CFG.N_FOLDS}")
print(f"Epochs : S1={CFG.S1_EPOCHS}  S2={CFG.S2_EPOCHS}  S3={CFG.S3_EPOCHS}  Seg={CFG.SEG_EPOCHS}")
print(f"LR     : S1={CFG.S1_LR}  S2={CFG.S2_LR}  S3={CFG.S3_LR}")
