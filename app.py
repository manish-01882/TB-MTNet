import sys
import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Setup paths for importing from 'cells'
sys.path.append('.')
from cells.cell1_setup import CFG
from cells.cell6_model import TBMTNet
import segmentation_models_pytorch as smp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model on {device}...")
model = TBMTNet(CFG).to(device)

# Load weights
try:
    state_dict = torch.load('fold0_best.pt', map_location=device, weights_only=False)['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model weights: {e}")

# ── Load Lung Segmentation U-Net (cell3_lung.py architecture) ─────────
print("Loading lung segmentation U-Net...")
lung_unet = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # we load our own weights
    in_channels=3,
    classes=1,
    activation=None,
).to(device)

try:
    lung_unet.load_state_dict(torch.load('lung_unet.pt', map_location=device, weights_only=False))
    lung_unet.eval()
    print("Lung U-Net loaded successfully.")
except Exception as e:
    lung_unet = None
    print(f"Failed to load lung U-Net weights: {e}")

# Lung segmentation transform (matches cell3 apply_lung_masks)
lung_seg_transform = A.Compose([
    A.Resize(CFG.SEG_SIZE, CFG.SEG_SIZE),
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
])

# Custom Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x):
        self.model.zero_grad()
        
        # Forward pass (model returns logit, sev)
        logit, sev = self.model(x)
        raw_prob = torch.sigmoid(logit)
        
        # Calibrate probability to undo POS_WEIGHT inflation
        # Same formula as cell9_infer.py: p = q / (w * (1 - q) + q)
        w = CFG.POS_WEIGHT
        q = raw_prob.item()
        calibrated_prob = q / (w * (1 - q) + q)
        
        # Backward pass targeting the logit
        logit.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pool gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w_i in enumerate(weights):
            cam += w_i * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        sev_score = float(np.clip(sev.item() * 140.0, 0, 140))
        return calibrated_prob, sev_score, cam

# Target the bridge layer — last spatial conv before transformer tokenization
# This gives much more focused, task-relevant activations than ECA
cam_extractor = GradCAM(model, model.bridge)



# Match the VALIDATION transform from cell5_dataset.py (line 16-20)
# Training uses random augments, but inference uses only CLAHE + Normalize
inference_transform = A.Compose([
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ── Preprocessing to match training pipeline (cell3 + cell5) ─────────

def load_and_stretch(image_np):
    """Replicate cell5 _load_image(): grayscale → per-image min-max stretch → 3-ch uint8.
    This ensures the model sees the same intensity distribution it was trained on."""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
    
    # Per-image min-max normalisation (same as cell5 line 49-51)
    img_f = gray.astype(np.float32)
    lo, hi = img_f.min(), img_f.max()
    img = ((img_f - lo) / (hi - lo + 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
    img = np.stack([img, img, img], axis=-1)  # (H, W, 3) uint8
    return img


@torch.no_grad()
def predict_lung_mask(img_3ch):
    """Predict lung mask using the actual U-Net from cell3_lung.py.
    Input: (H, W, 3) uint8 image (grayscale stacked to 3 channels).
    Returns: binary mask (0/255) at original resolution."""
    if lung_unet is None:
        # Fallback to Otsu if U-Net failed to load
        gray = cv2.cvtColor(img_3ch, cv2.COLOR_RGB2GRAY) if len(img_3ch.shape) == 3 else img_3ch
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    
    h, w = img_3ch.shape[:2]
    aug = lung_seg_transform(image=img_3ch)
    inp = aug['image'].unsqueeze(0).to(device)       # (1, 3, 256, 256)
    
    logit = lung_unet(inp)                            # (1, 1, 256, 256)
    prob = torch.sigmoid(logit).squeeze().cpu().numpy()
    msk = (prob > 0.5).astype(np.uint8) * 255
    msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
    return msk


def crop_to_lung(img, mask, target_size=512, dilation_px=10):
    """Replicate cell3 crop_to_lung(): dilate mask → crop bbox → pad to square → resize.
    This is the exact same function from the training pipeline."""
    if mask.max() == 0:
        # Fallback: centre-crop at 90%
        h, w = img.shape[:2]
        m = int(min(h, w) * 0.05)
        img = img[m:h-m, m:w-m]
    else:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        dilated = cv2.dilate(mask, kern)
        ys, xs = np.where(dilated > 0)
        y0 = max(ys.min() - dilation_px, 0)
        y1 = min(ys.max() + dilation_px, img.shape[0])
        x0 = max(xs.min() - dilation_px, 0)
        x1 = min(xs.max() + dilation_px, img.shape[1])
        img = img[y0:y1, x0:x1]
    
    # Pad to square
    h, w = img.shape[:2]
    side = max(h, w)
    padded = np.zeros((side, side, 3) if img.ndim == 3 else (side, side), dtype=img.dtype)
    ph, pw = (side - h) // 2, (side - w) // 2
    if img.ndim == 3:
        padded[ph:ph+h, pw:pw+w, :] = img
    else:
        padded[ph:ph+h, pw:pw+w] = img
    return cv2.resize(padded, (target_size, target_size))


def preprocess_xray(image_np):
    """Full preprocessing pipeline matching training:
    1. Grayscale → min-max stretch → 3-channel
    2. Estimate lung mask (Otsu)
    3. Crop to lung region + pad to square + resize to 512
    4. CLAHE + ImageNet normalize + ToTensor
    """
    # Step 1: grayscale min-max stretch → 3-ch (cell5 _load_image)
    img = load_and_stretch(image_np)
    
    # Step 2: predict lung mask using U-Net (matches cell3 pipeline)
    lung_mask = predict_lung_mask(img)
    
    # Step 3: crop to lung (cell3 crop_to_lung)
    img_cropped = crop_to_lung(img, lung_mask, CFG.IMAGE_SIZE)
    
    # Step 4: CLAHE + Normalize + ToTensor (cell5 val transform)
    augmented = inference_transform(image=img_cropped)
    tensor = augmented['image']
    
    return tensor, img_cropped


def generate_heatmap_overlay(original_img_np, cam):
    """Generate a Grad-CAM heatmap overlay, masked to the lung region."""
    original = np.float32(original_img_np) / 255
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    
    # Mask the CAM to lung region only (on the cropped image)
    lung_mask = predict_lung_mask(original_img_np)
    lung_mask_float = cv2.GaussianBlur(lung_mask, (21, 21), 0).astype(np.float32) / 255.0
    
    cam_masked = cam * lung_mask_float
    if cam_masked.max() > 0:
        cam_masked = cam_masked / cam_masked.max()
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_masked), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    overlay = 0.5 * heatmap + 0.5 * original
    overlay = overlay / np.max(overlay)
    overlay = np.uint8(255 * overlay)
    
    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    
    # Full preprocessing matching training pipeline
    tensor, img_preprocessed = preprocess_xray(image_np)
    tensor = tensor.unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    
    # Run Grad-CAM
    prob, sev, cam = cam_extractor(tensor)
    
    # Generate heatmap on the preprocessed (cropped) image
    heatmap_b64 = generate_heatmap_overlay(img_preprocessed, cam)
    
    # Preprocessed image as base64
    _, orig_buffer = cv2.imencode('.png', cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2BGR))
    orig_b64 = base64.b64encode(orig_buffer).decode('utf-8')
    
    return {
        "tb_probability": prob,
        "timika_severity": sev,
        "original_image": f"data:image/png;base64,{orig_b64}",
        "heatmap_image": f"data:image/png;base64,{heatmap_b64}"
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
