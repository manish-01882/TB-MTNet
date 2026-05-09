"""Cell 2 – Build unified labels.csv from all source datasets.
Uses rglob() for robust path discovery regardless of subdirectory nesting."""


def _find_images(root: Path, pattern: str = "*.png") -> List[Path]:
    """Recursively find all images matching pattern under root."""
    return sorted(root.rglob(pattern))


def _resolve_tbx11k_dir(cfg: "Config") -> Path:
    """Handle both standard and double-nested Kaggle dataset slug paths.

    Standard:  /kaggle/input/tbx11k-simplified/
    Nested:    /kaggle/input/datasets/vbookshelf/tbx11k-simplified/tbx11k-simplified/
    Also tries the datasets/<owner>/<slug>/<slug> pattern automatically.
    """
    base = cfg.TBX11K_DIR
    if base.exists():
        # Check if images/ or data.csv live directly here
        if (base / "images").exists() or (base / "data.csv").exists():
            return base
        # One more level in (e.g. tbx11k-simplified/ inside the slug root)
        for child in base.iterdir():
            if child.is_dir() and (child / "images").exists():
                return child

    # Try the /kaggle/input/datasets/... path variant
    alt = Path("/kaggle/input/datasets")
    if alt.exists():
        for candidate in alt.rglob("data.csv"):
            return candidate.parent

    return base  # fall back; _diagnose will warn


def _diagnose(cfg: "Config") -> None:
    tbx_dir = _resolve_tbx11k_dir(cfg)
    print("\n────────────────────────── Dataset path diagnostics ──────────────────────────────────────────")
    for name, d in [
        ("Shenzhen",   cfg.SHENZHEN_DIR),
        ("Shen masks", cfg.SHENZHEN_MASK_DIR),
        ("Montgomery", cfg.MONTGOMERY_DIR),
        ("TBX11K",     tbx_dir),
        ("Rahman",     cfg.RAHMAN_DIR),
    ]:
        exists = d.exists()
        count  = len(list(d.rglob("*.png"))) if exists else 0
        print(f"  {name:<12}: {'EXISTS' if exists else 'MISSING':7}  "
              f"({count} PNGs found)  →  {d}")
    print("────────────────────────────────────────────────────────────────────────────────────────────────\n")



def parse_shenzhen(cfg: "Config") -> pd.DataFrame:
    """Shenzhen CXR: filename suffix _0=normal, _1=TB.
    Works with flat or ChinaSet_AllFiles/CXR_png/ nesting."""
    rows = []

    # Find ALL PNGs under the dataset root that match Shenzhen naming
    all_pngs = [p for p in _find_images(cfg.SHENZHEN_DIR)
                if p.stem.startswith("CHNCXR")]

    if not all_pngs:
        print(f"  [WARN] No Shenzhen images found under {cfg.SHENZHEN_DIR}")
        return pd.DataFrame()

    # Build mask lookup: stem → mask path (from yoctoman/shcxr-lung-mask)
    mask_lookup: Dict[str, str] = {}
    if cfg.SHENZHEN_MASK_DIR.exists():
        for mp in _find_images(cfg.SHENZHEN_MASK_DIR):
            # mask filenames may be CHNCXR_0001_0_mask.png or CHNCXR_0001_0.png
            key = mp.stem.replace("_mask", "")
            mask_lookup[key] = str(mp)

    for p in all_pngs:
        stem     = p.stem                        # e.g. CHNCXR_0001_0
        tb_label = int(stem.split("_")[-1])      # 0 or 1
        mask_p   = mask_lookup.get(stem, "")
        rows.append(dict(
            image_path=str(p),
            patient_id=f"SZ_{stem}",
            source="shenzhen",
            tb_label=tb_label,
            has_severity=0,
            severity=-1.0,
            has_lung_mask=int(bool(mask_p)),
            lung_mask_path=mask_p,
        ))

    print(f"  Shenzhen   : {len(rows)} images  "
          f"({sum(r['tb_label'] for r in rows)} TB+, "
          f"{sum(r['has_lung_mask'] for r in rows)} masks)")
    return pd.DataFrame(rows)


def parse_montgomery(cfg: "Config") -> pd.DataFrame:
    """Montgomery CXR: L+R masks merged via logical OR.
    Works with flat or MontgomerySet/CXR_png/ nesting."""
    rows = []

    all_pngs = [p for p in _find_images(cfg.MONTGOMERY_DIR)
                if p.stem.startswith("MCUCXR") and "Mask" not in str(p)]

    if not all_pngs:
        print(f"  [WARN] No Montgomery images found under {cfg.MONTGOMERY_DIR}")
        return pd.DataFrame()

    # Build left/right mask lookups
    left_lookup:  Dict[str, Path] = {}
    right_lookup: Dict[str, Path] = {}
    for mp in _find_images(cfg.MONTGOMERY_DIR):
        if "leftMask" in str(mp)  and mp.stem.startswith("MCUCXR"):
            left_lookup[mp.stem]  = mp
        if "rightMask" in str(mp) and mp.stem.startswith("MCUCXR"):
            right_lookup[mp.stem] = mp

    mc_mask_dir = cfg.BASE / "mc_masks"
    mc_mask_dir.mkdir(exist_ok=True)

    for p in all_pngs:
        stem     = p.stem                        # e.g. MCUCXR_0001_0
        tb_label = int(stem.split("_")[-1])      # 0 or 1

        mask_p = ""
        lm = left_lookup.get(stem)
        rm = right_lookup.get(stem)
        if lm and rm:
            merged_path = mc_mask_dir / f"{stem}.png"
            if not merged_path.exists():
                l_arr = cv2.imread(str(lm), cv2.IMREAD_GRAYSCALE)
                r_arr = cv2.imread(str(rm), cv2.IMREAD_GRAYSCALE)
                if l_arr is not None and r_arr is not None:
                    if l_arr.shape != r_arr.shape:
                        r_arr = cv2.resize(r_arr, (l_arr.shape[1], l_arr.shape[0]))
                    merged = np.clip(l_arr.astype(np.uint16) + r_arr, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(merged_path), merged)
            if merged_path.exists():
                mask_p = str(merged_path)

        rows.append(dict(
            image_path=str(p),
            patient_id=f"MC_{stem}",
            source="montgomery",
            tb_label=tb_label,
            has_severity=0,
            severity=-1.0,
            has_lung_mask=int(bool(mask_p)),
            lung_mask_path=mask_p,
        ))

    print(f"  Montgomery : {len(rows)} images  "
          f"({sum(r['tb_label'] for r in rows)} TB+, "
          f"{sum(r['has_lung_mask'] for r in rows)} masks)")
    return pd.DataFrame(rows)


def parse_tbx11k(cfg: "Config") -> pd.DataFrame:
    """TBX11K-simplified: read data.csv for precise labels.

    CSV target column values:
      'active_tb'    → tb_label = 1
      'no_tb'        → tb_label = 0
      'latent_tb'    → tb_label = 0 (not infectious active TB)
      'sick_not_tb'  → tb_label = 0

    Falls back to filename prefix (tb*/h*/s*) if data.csv not found.
    Handles nested slug paths automatically via _resolve_tbx11k_dir().
    """
    rows = []
    tbx_dir = _resolve_tbx11k_dir(cfg)

    if not tbx_dir.exists():
        print(f"  [WARN] TBX11K dir not found: {tbx_dir}")
        return pd.DataFrame()

    print(f"  TBX11K root: {tbx_dir}")

    # Build image path lookup: fname → full path
    img_dir = tbx_dir / "images"
    all_imgs = list(img_dir.glob("*.png")) if img_dir.exists() \
               else list(tbx_dir.rglob("*.png"))
    img_lookup = {p.name: str(p) for p in all_imgs}

    if not img_lookup:
        print(f"  [WARN] No PNG images found under {tbx_dir}")
        return pd.DataFrame()

    # Try reading data.csv for precise labels
    csv_path = tbx_dir / "data.csv"
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path)
        # One row per annotation (multiple rows for same image if multiple bboxes)
        # Deduplicate by fname keeping the first occurrence for label assignment
        df_unique = df_csv.drop_duplicates(subset=["fname"], keep="first")
        for _, row in df_unique.iterrows():
            fname    = str(row["fname"]).strip()
            img_path = img_lookup.get(fname)
            if not img_path:
                continue
            target   = str(row.get("target",  "")).strip().lower()
            tb_type  = str(row.get("tb_type", "")).strip().lower()
            # target column = 'tb' | 'no_tb'
            # tb_type column = 'active_tb' | 'latent_tb' | 'none'
            tb_label = 1 if tb_type == "active_tb" else 0
            rows.append(dict(
                image_path=img_path,
                patient_id=f"TBX_{Path(fname).stem.lower()}",
                source="tbx11k",
                tb_label=tb_label,
                has_severity=0,
                severity=-1.0,
                has_lung_mask=0,
                lung_mask_path="",
            ))
        print(f"  TBX11K     : {len(rows)} images (from data.csv)  "
              f"({sum(r['tb_label'] for r in rows)} active TB+)")
    else:
        # Fallback: filename-prefix heuristic
        print(f"  [WARN] data.csv not found, using filename prefix labels")
        for p in sorted(all_imgs):
            stem = p.stem.lower()
            tb_label = 1 if stem.startswith("tb") else 0
            rows.append(dict(
                image_path=str(p),
                patient_id=f"TBX_{stem}",
                source="tbx11k",
                tb_label=tb_label,
                has_severity=0,
                severity=-1.0,
                has_lung_mask=0,
                lung_mask_path="",
            ))
        print(f"  TBX11K     : {len(rows)} images (filename prefix)  "
              f"({sum(r['tb_label'] for r in rows)} TB+)")

    return pd.DataFrame(rows)



def parse_rahman(cfg: "Config") -> pd.DataFrame:
    """Optional Tawsifur Rahman dataset: folder-name labels."""
    rows = []
    if not cfg.RAHMAN_DIR.exists():
        return pd.DataFrame()

    for p in _find_images(cfg.RAHMAN_DIR):
        parts = p.parts
        if "Tuberculosis" in parts:
            tb_label = 1
        elif "Normal" in parts:
            tb_label = 0
        else:
            continue
        rows.append(dict(
            image_path=str(p),
            patient_id=f"RH_{p.stem}",
            source="rahman",
            tb_label=tb_label,
            has_severity=0,
            severity=-1.0,
            has_lung_mask=0,
            lung_mask_path="",
        ))

    if rows:
        print(f"  Rahman     : {len(rows)} images  "
              f"({sum(r['tb_label'] for r in rows)} TB+)")
    return pd.DataFrame(rows)


def build_labels_csv(cfg: "Config") -> pd.DataFrame:
    """Merge all datasets, drop duplicates, save labels.csv."""
    if cfg.LABELS_CSV.exists():
        print(f"Loading existing labels CSV: {cfg.LABELS_CSV}")
        df = pd.read_csv(cfg.LABELS_CSV)
        print(f"  {len(df)} rows | {df['tb_label'].sum()} TB+")
        return df

    _diagnose(cfg)
    print("Building labels.csv …")

    parts = [
        parse_shenzhen(cfg),
        parse_montgomery(cfg),
        parse_tbx11k(cfg),
        parse_rahman(cfg),
    ]

    valid = [p for p in parts if len(p) > 0]
    if not valid:
        raise RuntimeError(
            "No images found in any dataset!\n"
            "Make sure all 4 Kaggle dataset slugs are added via 'Add Data':\n"
            "  raddar/tuberculosis-chest-xrays-shenzhen\n"
            "  yoctoman/shcxr-lung-mask\n"
            "  raddar/tuberculosis-chest-xrays-montgomery\n"
            "  vbookshelf/tbx11k-simplified\n"
        )

    df = pd.concat(valid, ignore_index=True)
    df.drop_duplicates(subset=["image_path"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(cfg.LABELS_CSV, index=False)

    total  = len(df)
    tb_pos = df["tb_label"].sum()
    print(f"\n  Total  : {total} images")
    print(f"  TB+    : {int(tb_pos)}  ({100*tb_pos/total:.1f}%)")
    print(f"  Saved  → {cfg.LABELS_CSV}")
    return df


# Run
labels_df = build_labels_csv(CFG)
labels_df.head()
