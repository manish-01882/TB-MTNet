"""Cell 5 – PyTorch Dataset + Albumentations transforms."""

def get_transforms(cfg: "Config", train: bool = True) -> A.Compose:
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.25),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class TBXDataset(Dataset):
    """
    Returns dict with keys:
      image        : (3, H, W) float32 tensor
      tb_label     : (1,) float32
      severity     : (1,) float32  — normalised to [0,1] (÷140)
      severity_mask: (1,) float32  — 1 if severity is valid, else 0
      patient_id   : str
    """
    def __init__(self, df: pd.DataFrame, cfg: "Config", train: bool = True):
        self.df    = df.reset_index(drop=True)
        self.cfg   = cfg
        self.train = train
        self.tfm   = get_transforms(cfg, train)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        """Load grayscale CXR → CLAHE → 3-channel uint8 array."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE), dtype=np.uint8)
        # Per-image intensity normalisation
        img = img.astype(np.float32) / 255.0
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = np.stack([img, img, img], axis=-1)
        return img                              # (H, W, 3) uint8

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load and crop to lung
        img  = self._load_image(row["image_path"])
        if row["has_lung_mask"] and row["lung_mask_path"]:
            msk = cv2.imread(row["lung_mask_path"], cv2.IMREAD_GRAYSCALE)
            if msk is not None:
                img = crop_to_lung(img, msk, self.cfg.IMAGE_SIZE)
        img = cv2.resize(img, (self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE))

        # Albumentations
        aug   = self.tfm(image=img)
        image = aug["image"]                   # (3, H, W)

        # Labels
        tb_label = torch.tensor([float(row["tb_label"])], dtype=torch.float32)

        has_sev       = bool(row["has_severity"]) and float(row["severity"]) >= 0
        severity_raw  = float(row["severity"]) if has_sev else 0.0
        severity_norm = severity_raw / self.cfg.SEVERITY_MAX   # [0, 1]
        severity      = torch.tensor([severity_norm], dtype=torch.float32)
        severity_mask = torch.tensor([float(has_sev)], dtype=torch.float32)

        return {
            "image":         image,
            "tb_label":      tb_label,
            "severity":      severity,
            "severity_mask": severity_mask,
            "patient_id":    str(row["patient_id"]),
        }


def make_loaders(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: "Config",
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TBXDataset(df.iloc[train_idx], cfg, train=True)
    val_ds   = TBXDataset(df.iloc[val_idx],   cfg, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,          # pre-load more batches; feeds GPU during CLAHE+Elastic
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    return train_loader, val_loader


# ── Smoke-test ────────────────────────────────────────────────────────
_sample_ds = TBXDataset(labels_df.head(8), CFG, train=True)
_batch = _sample_ds[0]
print("Image shape     :", _batch["image"].shape)
print("tb_label        :", _batch["tb_label"])
print("severity (norm) :", _batch["severity"])
print("severity_mask   :", _batch["severity_mask"])
