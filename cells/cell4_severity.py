"""Cell 4 – 3-Tier Severity Label Generation (Timika score 0-140).

Timika formula:  S = 100 * (A_lesion ∩ A_lung) / A_lung  +  40 * cavity_present

Tier 1 – Shenzhen/ChinaSet VIA polygon annotations  (Annotations_AllinOne_json.json)
         Using the consolidated all-in-one VGG Image Annotator (VIA) JSON.
Tier 2 – TBX11K COCO bounding boxes  →  pseudo-ALP + linear calibration
Tier 3 – Grad-CAM++ pseudo-labels for remaining unannotated TB+ cases
"""

import json as _json
from pathlib import Path

# ── Tier 1: ChinaSet / Shenzhen VIA polygon annotations ─────────────────
# Labels considered as active lesions (contributing to ALP)
LESION_LABELS = {
    "cavity",
    "cavitation",
    "small_infiltrate_(non-linear)",
    "moderate_infiltrate_(non-linear)",
    "severe_infiltrate_(consolidation)",
    "clustered_nodule_(2mm-5mm_apart)",
    "single_nodule_(non-calcified)",
    "linear_density",
    "apical_thickening",
    "pleural_thickening_(non-apical)",
    "thickening_of_interlobar_fissure",
    "retraction",
    "adenopathy",
    "other",
}

# Labels specifically indicating cavitation (adds +40 to Timika score)
CAVITY_LABELS = {"cavity", "cavitation"}


def _load_via_annotations(json_path: Path) -> dict:
    """Load the VIA all-in-one JSON and return a dict keyed by filename."""
    with open(json_path, "r") as f:
        raw = _json.load(f)

    # VIA format: keys are "<filename><size>", value has 'filename' and 'regions'
    # Build a clean mapping: filename -> list of regions
    fname_map: dict = {}
    for entry in raw.values():
        fname = entry.get("filename", "")
        regions = entry.get("regions", [])
        if fname:
            # Multiple entries with same filename are rare but union them
            fname_map.setdefault(fname, []).extend(regions)
    return fname_map


def _alp_from_via_regions(regions: list, lung_mask) -> tuple:
    """Compute ALP and cavity flag from VIA region list for one image.

    Args:
        regions: list of VIA region dicts (each with shape_attributes & region_attributes)
        lung_mask: np.ndarray H×W grayscale lung segmentation mask

    Returns:
        (alp_percent, has_cavity): ALP in [0, 100] or -1 on error; cavity bool
    """
    h, w = lung_mask.shape[:2]
    lesion_mask = np.zeros((h, w), dtype=np.uint8)
    has_cavity = False

    for region in regions:
        shape = region.get("shape_attributes", {})
        attrs = region.get("region_attributes", {})

        # Determine the label from region_attributes (first key whose value == "good")
        label = ""
        for k, v in attrs.items():
            if v == "good":
                label = k.lower()
                break

        shape_name = shape.get("name", "")

        # ── Handle polygon and polyline shapes ──────────────────────────
        if shape_name in ("polygon", "polyline"):
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) < 3 or len(ys) < 3:
                continue

            # Scale coordinates to lung_mask dimensions
            # VIA annotations are in original image pixel space;
            # lung_mask may be a resized version → scale proportionally.
            # We use the bounding box of points to detect if scaling is needed.
            max_x, max_y = max(xs), max(ys)
            sx = w / max(max_x + 1, w)   # ≈1.0 if already at mask size
            sy = h / max(max_y + 1, h)

            pts = np.array(
                [[int(x * sx), int(y * sy)] for x, y in zip(xs, ys)],
                dtype=np.int32,
            )
            pts = np.clip(pts, [0, 0], [w - 1, h - 1])

            # Only draw lesion polygons (skip non-lesion labels like calcified nodules)
            if label in LESION_LABELS or label == "":
                cv2.fillPoly(lesion_mask, [pts], 255)

            if label in CAVITY_LABELS:
                has_cavity = True

        # ── Handle point markers (e.g. Pleural_Effusion point) ──────────
        elif shape_name == "point":
            # Points don't contribute area to ALP; only flag cavities if relevant
            if label in CAVITY_LABELS:
                has_cavity = True

    lung_area = (lung_mask > 127).sum()
    if lung_area == 0:
        return -1.0, False

    lesion_in_lung = ((lesion_mask > 0) & (lung_mask > 127)).sum()
    alp = float(lesion_in_lung) / float(lung_area) * 100.0
    return min(alp, 100.0), has_cavity


def compute_tier1(df: pd.DataFrame, cfg: "Config") -> pd.DataFrame:
    """Add Tier-1 severity labels using the ChinaSet VIA all-in-one JSON.

    Looks for the annotation file at:
      1. cfg.SHEN_ANNOT_JSON  (if attribute exists)
      2. cfg.BASE / 'Annotations_AllinOne_json.json'
      3. The tb_model project root alongside this script
    """
    # Resolve annotation JSON path
    annot_json: Path | None = None
    candidates = []
    if hasattr(cfg, "SHEN_ANNOT_JSON"):
        candidates.append(Path(cfg.SHEN_ANNOT_JSON))
    candidates.append(Path(cfg.BASE) / "Annotations_AllinOne_json.json")

    for c in candidates:
        if c.exists():
            annot_json = c
            break

    if annot_json is None:
        print(
            "[Tier-1] Annotations_AllinOne_json.json not found in any candidate path. "
            "Tier-1 skipped; Tier-2/3 will be used."
        )
        print(f"  Searched: {[str(c) for c in candidates]}")
        return df

    print(f"[Tier-1] Loading VIA annotations from: {annot_json}")
    fname_map = _load_via_annotations(annot_json)
    print(f"  Loaded {len(fname_map)} annotated images.")

    updated = 0
    skipped_no_mask = 0
    skipped_no_annot = 0

    for idx, row in df[df["source"] == "shenzhen"].iterrows():
        if row["tb_label"] != 1:
            continue

        fname = Path(row["image_path"]).name
        regions = fname_map.get(fname)

        if regions is None:
            skipped_no_annot += 1
            continue

        if not row["has_lung_mask"]:
            skipped_no_mask += 1
            continue

        lung_mask = cv2.imread(str(row["lung_mask_path"]), cv2.IMREAD_GRAYSCALE)
        if lung_mask is None:
            skipped_no_mask += 1
            continue

        alp, cavity = _alp_from_via_regions(regions, lung_mask)
        if alp < 0:
            continue

        timika = min(alp + 40.0 * float(cavity), 140.0)
        df.at[idx, "severity"]     = timika
        df.at[idx, "has_severity"] = 1
        updated += 1

    print(
        f"[Tier-1] Shenzhen VIA polygons: {updated} cases labelled "
        f"(skipped – no annotation: {skipped_no_annot}, no mask: {skipped_no_mask})"
    )
    return df


# ── Tier 2: TBX11K bounding-box pseudo-labels ────────────────────────
def compute_tier2(df: pd.DataFrame, cfg: "Config") -> pd.DataFrame:
    """Convert TBX11K data.csv bboxes → pseudo-ALP → Timika score.

    Bbox format in data.csv:  {'xmin': x, 'ymin': y, 'width': w, 'height': h}
    (Python dict stored as a string; parse with ast.literal_eval)

    Multiple bbox rows for the same image are unioned into one lesion mask.
    ALP = (lesion ∩ lung) / lung  *  100  → Timika = ALP (no cavity term for Tier-2).
    """
    import ast

    # Resolve TBX11K root (same logic as parse_tbx11k)
    from pathlib import Path as _Path
    tbx_dir = _resolve_tbx11k_dir(cfg)
    csv_path = tbx_dir / "data.csv"

    if not csv_path.exists():
        print("[Tier-2] data.csv not found — skipping")
        return df

    df_csv = pd.read_csv(csv_path)
    # tb_type column = 'active_tb' | 'latent_tb' | 'none'
    df_tb  = df_csv[(df_csv["tb_type"] == "active_tb") &
                    (df_csv["bbox"].notna()) &
                    (df_csv["bbox"] != "none")].copy()

    if df_tb.empty:
        print("[Tier-2] No active TB bboxes in data.csv — skipping")
        return df

    # Group bboxes by filename (one image may have multiple boxes)
    fname2boxes: Dict[str, list] = {}
    for _, row in df_tb.iterrows():
        fname = str(row["fname"]).strip()
        try:
            b = ast.literal_eval(str(row["bbox"]))
            fname2boxes.setdefault(fname, []).append(b)
        except Exception:
            continue

    updated = 0
    tier1_alp, raw_alp_paired = [], []

    for idx, row in df[df["source"] == "tbx11k"].iterrows():
        if row["tb_label"] != 1:
            continue
        fname = Path(row["image_path"]).name
        boxes = fname2boxes.get(fname, [])
        if not boxes:
            continue

        # ── Compute ALP ──────────────────────────────────────────────
        if row["has_lung_mask"] and row["lung_mask_path"]:
            # Preferred: lung-relative ALP
            lung_mask = cv2.imread(str(row["lung_mask_path"]), cv2.IMREAD_GRAYSCALE)
            if lung_mask is not None:
                h, w      = lung_mask.shape
                lung_area = (lung_mask > 127).sum()
                pseudo    = np.zeros((h, w), dtype=np.uint8)
                for b in boxes:
                    sx, sy = w / 512.0, h / 512.0
                    x0 = max(0, int(b["xmin"] * sx))
                    y0 = max(0, int(b["ymin"] * sy))
                    x1 = min(w - 1, int((b["xmin"] + b["width"])  * sx))
                    y1 = min(h - 1, int((b["ymin"] + b["height"]) * sy))
                    pseudo[y0:y1, x0:x1] = 255
                if lung_area == 0:
                    continue
                overlap = ((pseudo > 0) & (lung_mask > 127)).sum()
                raw_alp = float(overlap) / float(lung_area) * 100.0
            else:
                lung_mask = None
        else:
            lung_mask = None

        if lung_mask is None:
            # Fallback: image-relative ALP (lung ≈ 60% of 512×512 image)
            h, w = 512, 512
            std_lung_area = int(h * w * 0.60)
            bbox_area = sum(
                int(b["width"]) * int(b["height"]) for b in boxes
            )
            raw_alp = min(float(bbox_area) / float(std_lung_area) * 100.0, 100.0)

        df.at[idx, "_raw_alp"] = raw_alp
        if row["has_severity"]:
            raw_alp_paired.append(raw_alp)
            tier1_alp.append(row["severity"])
        updated += 1

    if not updated:
        print("[Tier-2] No TBX11K active-TB rows matched bboxes in data.csv — skipping")
        return df

    # Linear calibration against any Tier-1 labels (if available)
    slope = 1.5  # paper default
    if len(tier1_alp) >= 5:
        X = np.array(raw_alp_paired).reshape(-1, 1)
        y = np.array(tier1_alp)
        reg = LinearRegression().fit(X, y)
        slope = float(np.clip(reg.coef_[0], 1.0, 2.5))
        print(f"  [Tier-2] Calibration slope = {slope:.3f}")

    for idx in df[df["source"] == "tbx11k"].index:
        raw = df.at[idx, "_raw_alp"] if "_raw_alp" in df.columns else -1
        if raw < 0:
            continue
        timika = min(raw * slope, 100.0)   # no cavity bonus in data.csv
        df.at[idx, "severity"]     = timika
        df.at[idx, "has_severity"] = 1

    df.drop(columns=["_raw_alp"], errors="ignore", inplace=True)
    print(f"[Tier-2] TBX11K data.csv bbox: {updated} cases labelled  (slope={slope:.2f})")
    return df


# ── Tier 3: Grad-CAM++ pseudo-labels ─────────────────────────────────
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _train_teacher(df: pd.DataFrame, cfg: "Config") -> nn.Module:
    """Train a lightweight DenseNet-121 teacher on Tier-1+Tier-2 labelled data."""
    ckpt = cfg.CKPT_DIR / "teacher_densenet.pt"
    model = timm.create_model("densenet121", pretrained=True, num_classes=1)
    model = model.to(cfg.DEVICE)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE, weights_only=False))
        model.eval(); return model

    # Simple dataset of all images with known tb_label
    class SimpleDS(Dataset):
        def __init__(self, df):
            self.df  = df.reset_index(drop=True)
            self.tfm = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r   = self.df.iloc[i]
            img = cv2.imread(str(r["image_path"]), cv2.IMREAD_GRAYSCALE)
            if img is None: img = np.zeros((224, 224), dtype=np.uint8)
            img = np.stack([img] * 3, axis=-1)
            aug = self.tfm(image=img)
            lbl = torch.tensor([float(r["tb_label"])])
            return aug["image"], lbl

    ds = SimpleDS(df)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-4)
    pw     = torch.tensor([5.95]).to(cfg.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(10):
        model.train(); total = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(cfg.DEVICE), lbls.to(cfg.DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = loss_fn(out, lbls)
            loss.backward(); opt.step()
            total += loss.item()
        if ep % 3 == 0:
            print(f"  Teacher epoch {ep+1}/10  loss={total/len(loader):.4f}")
    torch.save(model.state_dict(), ckpt)
    model.eval(); return model


def compute_tier3(df: pd.DataFrame, cfg: "Config") -> pd.DataFrame:
    """Generate Grad-CAM++ pseudo severity labels for unannotated TB+ cases."""
    needs = df[
        (df["tb_label"] == 1) &
        (df["has_severity"] == 0) &
        (df["has_lung_mask"] == 1)
    ].index.tolist()
    if not needs:
        print("[Tier-3] No unannotated TB+ cases to pseudo-label")
        return df

    print(f"[Tier-3] Pseudo-labelling {len(needs)} unannotated TB+ cases …")
    teacher = _train_teacher(df, cfg)
    teacher.eval()

    # GradCAM++ targets the last denseblock
    target_layer = [teacher.features.denseblock4.denselayer16.conv2]
    cam = GradCAMPlusPlus(model=teacher, target_layers=target_layer)

    tfm_val = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    updated = 0
    for idx in tqdm(needs, desc="GradCAM++ pseudo-labels"):
        row      = df.iloc[idx]
        img_raw  = cv2.imread(str(row["image_path"]), cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(str(row["lung_mask_path"]), cv2.IMREAD_GRAYSCALE)
        if img_raw is None or mask_raw is None:
            continue

        img3 = np.stack([img_raw] * 3, axis=-1)
        aug  = tfm_val(image=img3)
        inp  = aug["image"].unsqueeze(0).to(cfg.DEVICE)

        grayscale_cam = cam(
            input_tensor=inp,
            targets=[ClassifierOutputTarget(0)]
        )[0]  # (224, 224)

        # Resize back to lung-mask size
        h, w = mask_raw.shape
        cam_resized = cv2.resize(grayscale_cam, (w, h))

        # Threshold at per-image quantile within lung mask
        lung_px = cam_resized[mask_raw > 127]
        if len(lung_px) == 0:
            continue
        thresh = np.quantile(lung_px, cfg.GRADCAM_Q if hasattr(cfg, "GRADCAM_Q") else 0.4)
        lesion = ((cam_resized >= thresh) & (mask_raw > 127)).astype(np.uint8)

        # Morphological refinement
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel)
        lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN,  kernel)

        lung_area = (mask_raw > 127).sum()
        if lung_area == 0:
            continue
        alp    = float(lesion.sum()) / float(lung_area) * 100.0
        alp    = min(alp, 100.0)
        # No cavity information → cavity = 0 for Tier-3 (conservative)
        timika = float(alp)
        df.at[idx, "severity"]     = timika
        df.at[idx, "has_severity"] = 1
        updated += 1

    cam.__del__()   # release hooks
    print(f"[Tier-3] GradCAM++: {updated} cases labelled")
    return df


def add_severity_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Add stratification key for StratifiedGroupKFold."""
    df["sev_quartile"] = -1
    mask = df["has_severity"] == 1
    df.loc[mask, "sev_quartile"] = pd.qcut(
        df.loc[mask, "severity"], q=4, labels=False, duplicates="drop"
    )
    # Combined stratification key: tb_label * 10 + sev_quartile+1
    df["strat_key"] = (
        df["tb_label"].astype(str) + "_" + df["sev_quartile"].astype(str)
    )
    return df


# ── Run severity pipeline ─────────────────────────────────────────────
if not (labels_df["has_severity"] == 1).any():
    labels_df = compute_tier1(labels_df, CFG)
    labels_df = compute_tier2(labels_df, CFG)
    labels_df = compute_tier3(labels_df, CFG)
    labels_df = add_severity_quartile(labels_df)
    labels_df.to_csv(CFG.LABELS_CSV, index=False)
    print(
        f"\nSeverity coverage: {labels_df['has_severity'].sum()} / "
        f"{labels_df['tb_label'].sum()} TB+ cases"
    )
else:
    print("Severity labels already present — skipping generation.")
    labels_df = add_severity_quartile(labels_df)
