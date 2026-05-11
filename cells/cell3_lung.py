"""Cell 3 – Lung Segmentation: train U-Net on Montgomery+Shenzhen masks,
then apply to all images lacking a lung mask (mainly TBX11K)."""

# ── Dataset for lung segmenter training ──────────────────────────────
class LungMaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 256):
        self.rows = df[df["has_lung_mask"] == 1].reset_index(drop=True)
        self.size = size
        self.tfm  = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r   = self.rows.iloc[idx]
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(r["lung_mask_path"], cv2.IMREAD_GRAYSCALE)
        # Guard against corrupted or missing files so the DataLoader never crashes
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
        if msk is None:
            msk = np.zeros((self.size, self.size), dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)          # H×W×3
        msk = (msk > 127).astype(np.float32)
        aug  = self.tfm(image=img, mask=msk)
        return aug["image"], aug["mask"].unsqueeze(0)     # (3,H,W), (1,H,W)


def get_lung_unet() -> nn.Module:
    """U-Net with ResNet-34 ImageNet encoder from segmentation_models_pytorch."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,          # raw logits; we apply sigmoid at inference
    )


def train_lung_segmenter(df: pd.DataFrame, cfg: "Config") -> nn.Module:
    """Train lung U-Net on available masks (Montgomery + Shenzhen)."""
    if cfg.LUNG_CKPT.exists():
        print(f"Loading cached lung U-Net: {cfg.LUNG_CKPT}")
        model = get_lung_unet().to(cfg.DEVICE)
        model.load_state_dict(torch.load(cfg.LUNG_CKPT, map_location=cfg.DEVICE, weights_only=False))
        model.eval()
        return model

    print("Training lung segmentation U-Net …")
    ds     = LungMaskDataset(df, size=cfg.SEG_SIZE)
    if len(ds) == 0:
        print("  [WARN] No lung masks found for training (len(ds)==0). Skipping segmentation.")
        return None

    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = get_lung_unet().to(cfg.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = smp.losses.DiceLoss(mode="binary")
    scaler  = GradScaler("cuda") if cfg.USE_AMP else None

    best_loss  = float("inf")
    amp_dtype  = getattr(torch, cfg.AMP_DTYPE, torch.float16)  # honour cfg (bfloat16 or float16)
    for epoch in range(cfg.SEG_EPOCHS):
        model.train(); epoch_loss = 0.0
        for imgs, msks in loader:
            imgs, msks = imgs.to(cfg.DEVICE), msks.to(cfg.DEVICE)
            opt.zero_grad(set_to_none=True)
            if cfg.USE_AMP:
                with autocast("cuda", dtype=amp_dtype):
                    pred = model(imgs)
                    loss = loss_fn(pred, msks)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                pred = model(imgs)
                loss = loss_fn(pred, msks)
                loss.backward(); opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        if epoch % 5 == 0 or epoch == cfg.SEG_EPOCHS - 1:   # always print the last epoch
            print(f"  Seg epoch {epoch+1:02d}/{cfg.SEG_EPOCHS}  loss={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            core = model.module if hasattr(model, "module") else model
            torch.save(core.state_dict(), cfg.LUNG_CKPT)

    print(f"Lung U-Net saved → {cfg.LUNG_CKPT}  (best Dice loss={best_loss:.4f})")
    core = model.module if hasattr(model, "module") else model
    core.eval()
    return core


@torch.no_grad()
def apply_lung_masks(df: pd.DataFrame, model: nn.Module, cfg: "Config") -> pd.DataFrame:
    """Predict lung masks for images that don't have one (e.g. TBX11K).
    Saves masks to /kaggle/working/predicted_masks/ and updates df."""
    out_dir = cfg.BASE / "predicted_masks"; out_dir.mkdir(exist_ok=True)
    needs   = df[df["has_lung_mask"] == 0].index.tolist()
    if not needs:
        print("All images already have lung masks.")
        return df

    if model is None:
        print("  [WARN] No lung model available. Skipping mask prediction.")
        return df

    print(f"Predicting lung masks for {len(needs)} images …")
    model.eval()
    tfm = A.Compose([
        A.Resize(cfg.SEG_SIZE, cfg.SEG_SIZE),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])

    for idx in tqdm(needs, desc="Lung masks"):
        img_path = df.at[idx, "image_path"]
        out_path = out_dir / (Path(img_path).stem + "_mask.png")
        if out_path.exists():
            df.at[idx, "lung_mask_path"] = str(out_path)
            df.at[idx, "has_lung_mask"]  = 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        img3 = np.stack([img, img, img], axis=-1)
        aug  = tfm(image=img3)
        inp  = aug["image"].unsqueeze(0).to(cfg.DEVICE)

        logit = model(inp)                                # (1,1,256,256)
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
        msk   = (prob > 0.5).astype(np.uint8) * 255
        msk   = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_path), msk)

        df.at[idx, "lung_mask_path"] = str(out_path)
        df.at[idx, "has_lung_mask"]  = 1

    df.to_csv(cfg.LABELS_CSV, index=False)
    print(f"Updated labels.csv with predicted masks.")
    return df


def crop_to_lung(img: np.ndarray, mask: np.ndarray,
                 target_size: int = 512, dilation_px: int = 10) -> np.ndarray:
    """Dilate lung mask, crop bounding box, pad to square, resize."""
    if mask.max() == 0:
        # fallback: centre-crop at 90% of image
        h, w = img.shape[:2]
        m = int(min(h, w) * 0.05)
        img = img[m:h-m, m:w-m]
    else:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        dilated = cv2.dilate(mask, kern)
        ys, xs  = np.where(dilated > 0)
        y0, y1  = max(ys.min() - dilation_px, 0), min(ys.max() + dilation_px, img.shape[0])
        x0, x1  = max(xs.min() - dilation_px, 0), min(xs.max() + dilation_px, img.shape[1])
        img = img[y0:y1, x0:x1]

    # Pad to square
    h, w = img.shape[:2]
    side  = max(h, w)
    padded = np.zeros((side, side, 3) if img.ndim == 3 else (side, side), dtype=img.dtype)
    ph, pw = (side - h) // 2, (side - w) // 2
    if img.ndim == 3:
        padded[ph:ph+h, pw:pw+w, :] = img
    else:
        padded[ph:ph+h, pw:pw+w]    = img
    return cv2.resize(padded, (target_size, target_size))


# ── Run ───────────────────────────────────────────────────────────────
lung_model   = train_lung_segmenter(labels_df, CFG)
labels_df    = apply_lung_masks(labels_df, lung_model, CFG)
print(f"\nImages with lung masks: {labels_df['has_lung_mask'].sum()} / {len(labels_df)}")
