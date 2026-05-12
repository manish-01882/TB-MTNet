"""Cell 6 – TB-MTNet architecture + MultiTaskLoss (Kendall–Gal–Cipolla)."""

# ── ECA Channel Attention ─────────────────────────────────────────────
class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020). ~5 params."""
    def __init__(self, channels: int, k: int = 5):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.gap(x).squeeze(-1).transpose(-1, -2)   # (B, 1, C)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        return x * torch.sigmoid(y)


# ── Projection Bridge: 2048 → 512 → 96 ───────────────────────────────
class ProjectionBridge(nn.Module):
    """2048→512→96 two-stage projection (paper §5.1)."""
    def __init__(self, in_ch: int = 2048, mid_ch: int = 512, out_ch: int = 96):
        super().__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bridge(x)   # (B, 96, 14, 14)


# ── TB-MTNet ─────────────────────────────────────────────────────────
class TBMTNet(nn.Module):
    """
    Forward path (paper §5.1):
      Input (B,3,512,512)
      → Inception-v3 features_only → (B, 2048, 14, 14)
      → ECA(k=5)
      → ProjectionBridge 2048→512→96 → (B, 96, 14, 14)
      → flatten → (B, 196, 96)  +  [CLS]  +  pos-embed  → (B, 197, 96)
      → 4-layer pre-norm Transformer (d=96, h=4, ffn=384, GELU)
      → LayerNorm → CLS token → (B, 96)
      → cls head: Linear(96→1) sigmoid   [TB probability]
      → reg head: Linear(96→1) sigmoid   [Timika / 140]
    """
    def __init__(self, cfg: "Config"):
        super().__init__()
        d = cfg.D_MODEL

        # Backbone
        self.backbone = timm.create_model(
            "inception_v3",
            pretrained=True,
            features_only=True,
            out_indices=(4,),
        )
        self.eca    = ECA(channels=2048, k=5)
        self.bridge = ProjectionBridge(2048, 512, d)

        # Transformer
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed  = nn.Parameter(torch.zeros(1, 197, d))  # 196 patches + 1 CLS
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.FFN_DIM,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,     # pre-norm (paper)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.N_LAYERS)
        self.norm        = nn.LayerNorm(d)

        # Heads
        self.cls_head = nn.Linear(d, 1)   # TB detection (logit → BCEWithLogits)
        self.reg_head = nn.Linear(d, 1)   # Timika/140  (sigmoid output)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        feats = self.backbone(x)[0]         # (B, 2048, 14, 14)
        feats = self.eca(feats)
        feats = self.bridge(feats)          # (B, 96, 14, 14)

        # Flatten spatial → tokens
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)   # (B, 196, 96)

        # Prepend CLS token
        cls   = self.cls_token.expand(B, -1, -1)    # (B, 1, 96)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, 197, 96)
        tokens = tokens + self.pos_embed

        # Transformer
        out = self.transformer(tokens)   # (B, 197, 96)
        out = self.norm(out)
        cls_out = out[:, 0]              # (B, 96)

        # Heads
        logit  = self.cls_head(cls_out)                # (B, 1)  — raw logit
        sev    = torch.sigmoid(self.reg_head(cls_out)) # (B, 1)  — in [0, 1]
        return logit, sev

    def predict(self, x: torch.Tensor):
        """Returns (tb_prob, timika_score) for inference."""
        logit, sev = self.forward(x)
        return torch.sigmoid(logit), sev * 140.0


# ── Pearson Correlation Loss ──────────────────────────────────────────
def pearson_loss(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """1 − Pearson r, computed only over masked (TB-positive) samples.
    Directly optimises the ranking metric the evaluation cares about."""
    pred_m   = pred[mask.bool()]
    target_m = target[mask.bool()]
    if len(pred_m) < 2:
        return torch.tensor(0.0, device=pred.device)
    vx  = pred_m   - pred_m.mean()
    vy  = target_m - target_m.mean()
    corr = (vx * vy).sum() / (
        (vx.norm() * vy.norm()).clamp(min=1e-8))
    return 1.0 - corr


# ── Multi-Task Loss (Kendall–Gal–Cipolla uncertainty weighting) ───────
class MultiTaskLoss(nn.Module):
    """
    L = exp(-s_c) * L_BCE + exp(-s_r) * (L_Huber + 0.5·L_Pearson) + s_c + s_r
    s_c, s_r are learnable log-variance parameters.
    Regression loss is masked to TB-positive cases only.
    """
    def __init__(self, pos_weight: float, huber_beta: float):
        super().__init__()
        self.s_c = nn.Parameter(torch.zeros(()))          # log-var for cls
        self.s_r = nn.Parameter(torch.tensor(-1.0))       # start high to prioritise severity
        pw = torch.tensor([pos_weight])
        self.bce   = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.huber = nn.SmoothL1Loss(beta=huber_beta, reduction="none")

    def forward(
        self,
        logit: torch.Tensor,        # (B, 1) raw logit
        y_cls: torch.Tensor,        # (B, 1) binary TB label
        sev_pred: torch.Tensor,     # (B, 1) sigmoid output [0, 1]
        y_sev: torch.Tensor,        # (B, 1) normalised severity [0, 1]
        sev_mask: torch.Tensor,     # (B, 1) 1 if severity valid
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        l_cls = self.bce(logit, y_cls)

        # Regression: Huber (absolute distance) + Pearson (ranking)
        huber_elem = self.huber(sev_pred, y_sev)        # (B, 1)
        n_valid    = sev_mask.sum().clamp(min=1.0)
        l_huber    = (huber_elem * sev_mask).sum() / n_valid
        l_pearson  = pearson_loss(sev_pred, y_sev, sev_mask)
        l_reg      = l_huber + 0.5 * l_pearson

        loss = (torch.exp(-self.s_c) * l_cls + self.s_c +
                torch.exp(-self.s_r) * l_reg  + self.s_r)
        return loss, l_cls, l_reg


# ── Utilities ─────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total/1e6:.2f}M")
    print(f"Trainable params : {trainable/1e6:.2f}M")
    return total


# ── Instantiate & sanity-check ────────────────────────────────────────
model    = TBMTNet(CFG).to(CFG.DEVICE)
mtl_loss = MultiTaskLoss(CFG.POS_WEIGHT, CFG.HUBER_BETA).to(CFG.DEVICE)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")

count_parameters(model)

# Forward-pass smoke test
_x = torch.randn(2, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE).to(CFG.DEVICE)
with torch.no_grad():
    _core = model.module if hasattr(model, "module") else model
    _logit, _sev = _core(_x)
print(f"Logit shape : {_logit.shape}  |  Sev shape : {_sev.shape}")
assert _logit.shape == (2, 1) and _sev.shape == (2, 1), "Shape mismatch!"
print("Architecture OK ✓")
