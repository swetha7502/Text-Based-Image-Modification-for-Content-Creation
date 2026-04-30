"""
cv_pipeline.py — all CV ops, forced CPU-only
============================================
Operations:
  1. segment_object()    — Grounded-DINO + SAM → binary mask
  2. inpaint_remove()    — LaMa inpainting → erase object
  3. replace_object()    — SAM mask + texture synthesis → replace with description
  4. remove_background() — rembg U2-Net
  5. apply_filter()      — 12 local filters (NumPy/Pillow/OpenCV)
  6. recolor_object()    — HSV-based recoloring of segmented region
  7. auto_enhance()      — CLAHE + white balance + denoise
  8. sharpen_edges()     — Laplacian unsharp mask
  9. color_transfer()    — Reinhard color transfer between images
 10. depth_blur()        — Fake DOF using MiDaS depth estimation
"""

# Existing CV models (SAM, Grounded-DINO, LaMa, rembg) run on CPU.
# SD inpainting uses GPU when available — handled in sd_inpaint.py.
import os
import torch

DEVICE = torch.device("cpu")

# Patch SimpleLama's baked-in default (evaluated at class-definition time)
import simple_lama_inpainting.models.model as _lama_mod
_lama_mod.SimpleLama.__init__.__defaults__ = (torch.device("cpu"),)

import pathlib, urllib.request, warnings, colorsys
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance

warnings.filterwarnings("ignore")
_cv_device_label = "CUDA" if torch.cuda.is_available() else "CPU"
print(f"[cv] CV pipeline active (SAM/DINO/LaMa on CPU, SD on {_cv_device_label})")

# ── model paths ───────────────────────────────────────────────────────────────
MODELS_DIR     = pathlib.Path.home() / ".pixedit_models"
MODELS_DIR.mkdir(exist_ok=True)
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"
SAM_URL        = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
GDINO_WEIGHTS  = MODELS_DIR / "groundingdino_swint_ogc.pth"
GDINO_URL      = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# ── lazy singletons ───────────────────────────────────────────────────────────
_sam_predictor = None
_gdino_model   = None
_lama_model    = None

# ── downloaders ───────────────────────────────────────────────────────────────
def _download(url, dest, label):
    if dest.exists(): return
    print(f"[cv] Downloading {label} …")
    def _prog(b, bs, total):
        print(f"\r    {min(b*bs,total)/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_prog)
    print()

# ── model loaders ─────────────────────────────────────────────────────────────
def _load_gdino():
    global _gdino_model
    if _gdino_model: return _gdino_model
    _download(GDINO_URL, GDINO_WEIGHTS, "Grounded-DINO weights (~700 MB)")
    print("[cv] Loading Grounded-DINO …")
    import groundingdino
    from groundingdino.util.inference import load_model
    pkg = pathlib.Path(groundingdino.__file__).parent
    _gdino_model = load_model(
        model_config_path     = str(pkg / "config" / "GroundingDINO_SwinT_OGC.py"),
        model_checkpoint_path = str(GDINO_WEIGHTS),
        device                = "cpu",
    )
    return _gdino_model

def _load_sam():
    global _sam_predictor
    if _sam_predictor: return _sam_predictor
    _download(SAM_URL, SAM_CHECKPOINT, "SAM ViT-B weights (~375 MB)")
    print("[cv] Loading SAM ViT-B …")
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE); sam.eval()
    _sam_predictor = SamPredictor(sam)
    return _sam_predictor

def _load_lama():
    global _lama_model
    if _lama_model: return _lama_model
    print("[cv] Loading LaMa …")
    from simple_lama_inpainting import SimpleLama
    _lama_model = SimpleLama(device=DEVICE)
    return _lama_model

# ── helpers ───────────────────────────────────────────────────────────────────
def pil_to_bgr(img):  return np.array(img.convert("RGB"))[:,:,::-1].copy()
def bgr_to_pil(arr):  return Image.fromarray(arr[:,:,::-1])
def from_arr(arr):    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
def pil_to_rgb(img):  return np.array(img.convert("RGB"))


# ═════════════════════════════════════════════════════════════════════════════
# 1. TEXT-GUIDED SEGMENTATION  (Grounded-DINO + SAM)
# ═════════════════════════════════════════════════════════════════════════════

def segment_object(pil_img: Image.Image, text: str):
    """
    Returns binary mask (H×W uint8, 0/255) for the named object, or None.
    Pipeline: Grounded-DINO → bounding boxes → SAM → pixel masks
    """
    import groundingdino.datasets.transforms as GDT
    from groundingdino.util.inference import predict

    gdino     = _load_gdino()
    predictor = _load_sam()
    rgb       = pil_to_rgb(pil_img)
    h, w      = rgb.shape[:2]

    # MUST use groundingdino's own GDT transforms — torchvision causes
    # "ValueError: not supported" inside nested_tensor_from_tensor_list
    transform = GDT.Compose([
        GDT.RandomResize([800], max_size=1333),
        GDT.ToTensor(),
        GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(pil_img.convert("RGB"), None)
    img_tensor    = img_tensor.to(DEVICE)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gdino, image=img_tensor, caption=text,
            box_threshold=0.30, text_threshold=0.25,
            device="cpu",  # predict() defaults to "cuda" — must override
        )

    if boxes is None or len(boxes) == 0:
        print(f"[cv] '{text}' not found in image")
        return None

    b = boxes.clone()
    boxes_xyxy = torch.stack([
        (b[:,0]-b[:,2]/2)*w, (b[:,1]-b[:,3]/2)*h,
        (b[:,0]+b[:,2]/2)*w, (b[:,1]+b[:,3]/2)*h,
    ], dim=1).numpy()
    print(f"[cv] Found {len(boxes_xyxy)} box(es): {phrases}")

    predictor.set_image(rgb)
    combined = np.zeros((h, w), dtype=bool)
    for box in boxes_xyxy:
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        combined |= masks[np.argmax(scores)]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    return cv2.dilate(combined.astype(np.uint8)*255, kernel, iterations=2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. INPAINTING — REMOVE  (LaMa)
# ═════════════════════════════════════════════════════════════════════════════

def inpaint_remove(pil_img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Erase masked region, fill coherently with LaMa."""
    lama = _load_lama()
    return lama(pil_img.convert("RGB"), Image.fromarray(mask).convert("L"))


# ═════════════════════════════════════════════════════════════════════════════
# 3. REPLACE OBJECT
# ═════════════════════════════════════════════════════════════════════════════

def replace_object(pil_img: Image.Image, object_text: str, replacement_text: str) -> Image.Image:
    print(f"[cv] Replacing '{object_text}' with '{replacement_text}' ...")

    mask = segment_object(pil_img, object_text)
    if mask is None:
        return pil_img

    # ── SDEdit replacement (primary path) ────────────────────────────────────
    try:
        from sd_inpaint import sd_replace
        prompt = (
            f"{replacement_text}, photorealistic, sharp focus, "
            "natural lighting, high resolution, seamlessly integrated"
        )
        return sd_replace(pil_img, mask, prompt)
    except Exception as _sd_exc:
        print(f"[cv] SDEdit unavailable ({_sd_exc}), using synthesis fallback")

    # ── fallback: LaMa fill + color synthesis (unchanged) ─────────────────────
    rgb = pil_to_rgb(pil_img)
    h, w = rgb.shape[:2]

    # LaMa fills the object region with natural background
    lama        = _load_lama()
    mask_pil    = Image.fromarray(mask).convert("L")
    bg_img      = lama(pil_img.convert("RGB"), mask_pil)
    bg_img_resized = bg_img.resize((w, h), Image.LANCZOS)

    # Build replacement content
    replacement_patch = _synthesise_replacement(rgb, mask, replacement_text)

    if replacement_patch is None:
        # No color/texture inferable — return LaMa result (object erased)
        print(f"[cv] No color inferred from '{replacement_text}', returning inpainted result")
        return bg_img_resized

    bg           = pil_to_rgb(bg_img_resized)
    patch_resized = cv2.resize(replacement_patch, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_resized  = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask_resized > 0)
    if len(xs) == 0:
        return bg_img_resized

    cx = int((int(xs.min()) + int(xs.max())) / 2)
    cy = int((int(ys.min()) + int(ys.max())) / 2)
    cx = int(np.clip(cx, 1, w - 2))
    cy = int(np.clip(cy, 1, h - 2))

    try:
        result_bgr = cv2.seamlessClone(
            pil_to_bgr(Image.fromarray(patch_resized)),
            pil_to_bgr(bg_img_resized),
            mask_resized,
            (cx, cy),
            cv2.NORMAL_CLONE,
        )
        return bgr_to_pil(result_bgr)
    except cv2.error:
        soft    = cv2.GaussianBlur(mask_resized.astype(np.float32), (15, 15), 0) / 255.0
        soft    = soft[:, :, None]
        blended = (patch_resized.astype(np.float32) * soft +
                   bg.astype(np.float32) * (1 - soft))
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _synthesise_replacement(rgb: np.ndarray, mask: np.ndarray, description: str):
    """
    Returns an RGB patch (same size as rgb) with the mask region recolored to
    match `description`. Returns None if no color/texture can be inferred.

    Uses the original object pixels as the base (preserves shape and lighting)
    then shifts their color in LAB space toward the target.
    """
    desc = description.lower()

    # RGB color vocabulary — longest keywords first to avoid short-match shadowing
    COLOR_MAP_RGB = {
        "concrete": (169, 169, 169), "leather":  (101,  67,  33),
        "stormy":   ( 80,  80, 100), "golden":   (255, 200,  50),
        "cloudy":   (180, 180, 200), "wooden":   (160, 100,  50),
        "chrome":   (200, 200, 210), "crimson":  (220,  20,  60),
        "maroon":   (128,   0,   0), "silver":   (192, 192, 192),
        "denim":    ( 21,  96, 189), "coral":    (255, 127,  80),
        "azure":    (  0, 127, 255), "ocean":    (  0, 105, 148),
        "storm":    ( 70,  70,  90), "cloud":    (200, 200, 210),
        "steel":    (150, 160, 170), "grass":    ( 34, 139,  34),
        "beige":    (245, 228, 196), "stone":    (169, 169, 169),
        "metal":    (160, 160, 160), "neon":     (255, 255,   0),
        "rust":     (183,  65,  14), "sand":     (210, 200, 180),
        "snow":     (255, 250, 250), "fire":     (255,  69,   0),
        "teal":     (  0, 128, 128), "cyan":     (  0, 200, 200),
        "gold":     (255, 215,   0), "lime":     ( 50, 255,   0),
        "navy":     (  0,   0, 128), "pink":     (255, 105, 180),
        "wood":     (139,  82,  42), "dark":     ( 40,  40,  40),
        "dusk":     (180,  80,  50), "dawn":     (255, 150, 100),
        "red":      (200,   0,   0), "blue":     (  0, 100, 200),
        "sky":      (135, 206, 235), "sea":      (  0, 119, 190),
        "tan":      (210, 180, 140), "fog":      (200, 200, 210),
        "green":    ( 30, 150,  30), "brown":    (139,  82,  42),
        "water":    ( 70, 130, 180), "glass":    (200, 220, 240),
        "white":    (240, 240, 240), "black":    ( 20,  20,  20),
        "gray":     (128, 128, 128), "grey":     (128, 128, 128),
        "light":    (220, 220, 220), "bright":   (100, 200, 255),
        "orange":   (255, 140,   0), "yellow":   (255, 220,   0),
        "purple":   (130,   0, 180), "sunset":   (255, 120,  50),
    }

    target_rgb = None
    for kw, rgb_color in COLOR_MAP_RGB.items():
        if kw in desc:
            target_rgb = np.array(rgb_color, dtype=np.float32)
            break

    if target_rgb is None:
        return None

    # Use original pixels as base — preserves object shape and luminance structure
    patch = rgb.copy().astype(np.float32)
    mask_bool = mask > 0

    if mask_bool.sum() > 0:
        bgr      = patch[:, :, ::-1].astype(np.uint8)
        lab      = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

        target_bgr = target_rgb[::-1].reshape(1, 1, 3).astype(np.uint8)
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)

        # Full chroma shift (A, B channels) + 50/50 luminance blend
        lab[mask_bool, 1] = target_lab[1]
        lab[mask_bool, 2] = target_lab[2]
        lab[mask_bool, 0] = lab[mask_bool, 0] * 0.5 + target_lab[0] * 0.5

        result_bgr = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
        patch = result_bgr[:, :, ::-1].astype(np.float32)

    patch_u8  = np.clip(patch, 0, 255).astype(np.uint8)
    patch_bgr = patch_u8[:, :, ::-1].copy()

    # Texture modifiers
    if any(kw in desc for kw in ["smooth", "clean", "flat", "solid", "matte"]):
        if mask_bool.sum() > 0:
            mean_color = patch_bgr[mask_bool].mean(axis=0).astype(np.uint8)
            patch_bgr[mask_bool] = mean_color

    elif any(kw in desc for kw in ["rough", "textured", "grainy", "stone", "concrete",
                                    "brick", "leather", "rustic"]):
        noise     = np.random.normal(0, 20, patch_bgr.shape).astype(np.float32)
        patch_bgr = np.clip(patch_bgr.astype(np.float32) + noise * (mask[:, :, None] / 255.0),
                            0, 255).astype(np.uint8)

    elif any(kw in desc for kw in ["glossy", "shiny", "metallic", "glass", "chrome", "polished"]):
        patch_f       = patch_bgr.astype(np.float32)
        cy_m, cx_m    = np.where(mask > 0)
        if len(cy_m) > 0:
            spec_y    = int(cy_m.min() + (cy_m.max() - cy_m.min()) * 0.25)
            spec_x    = int(cx_m.min() + (cx_m.max() - cx_m.min()) * 0.50)
            spec_p    = np.zeros_like(patch_f)
            radius    = max(5, min(30, (cx_m.max() - cx_m.min()) // 4))
            cv2.circle(spec_p, (spec_x, spec_y), radius, (255, 255, 255), -1)
            spec_p    = cv2.GaussianBlur(spec_p, (31, 31), 0)
            patch_f  += spec_p * 0.5 * (mask[:, :, None] / 255.0)
        patch_bgr = np.clip(patch_f, 0, 255).astype(np.uint8)

    elif any(kw in desc for kw in ["blurry", "soft", "foggy", "hazy", "blur"]):
        blurred   = cv2.GaussianBlur(patch_bgr, (21, 21), 0)
        alpha     = mask[:, :, None] / 255.0
        patch_bgr = (blurred * alpha + patch_bgr * (1 - alpha)).astype(np.uint8)

    return patch_bgr[:, :, ::-1]  # return RGB


# ═════════════════════════════════════════════════════════════════════════════
# 4. BACKGROUND REMOVAL  (rembg)
# ═════════════════════════════════════════════════════════════════════════════

def remove_background(pil_img: Image.Image) -> Image.Image:
    """Returns RGBA image with transparent background."""
    print("[cv] Running rembg …")
    from rembg import remove as rembg_remove
    return rembg_remove(pil_img.convert("RGB"))


# ═════════════════════════════════════════════════════════════════════════════
# 5. OBJECT RECOLOR
# ═════════════════════════════════════════════════════════════════════════════

def recolor_object(pil_img: Image.Image, object_text: str, target_color: str) -> Image.Image:
    """
    Change the color of a named object.
    Uses SAM mask + HSV hue rotation in the masked region.

    Example: recolor_object(img, "shirt", "blue")
    """
    print(f"[cv] Recoloring '{object_text}' to '{target_color}' …")
    mask = segment_object(pil_img, object_text)
    if mask is None:
        return pil_img

    COLOR_HUE = {
        "red":0, "orange":20, "yellow":35, "green":75,
        "cyan":100, "blue":115, "purple":135, "pink":150,
        "magenta":155, "white":None, "black":None, "gray":None,
    }

    target = target_color.lower().strip()
    target_hue = None
    for kw, hue in COLOR_HUE.items():
        if kw in target:
            target_hue = hue
            break

    rgb  = pil_to_rgb(pil_img)
    hsv  = cv2.cvtColor(rgb[:,:,::-1].copy(), cv2.COLOR_BGR2HSV).astype(np.float32)

    if target_hue is None:
        # desaturate for white/black/gray
        hsv[:,:,1][mask > 0] *= 0.1
    else:
        hsv[:,:,0][mask > 0] = target_hue
        hsv[:,:,1][mask > 0] = np.clip(hsv[:,:,1][mask > 0] * 1.2, 0, 255)

    result_bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr_to_pil(result_bgr)


# ═════════════════════════════════════════════════════════════════════════════
# 6. AUTO ENHANCE
# ═════════════════════════════════════════════════════════════════════════════

def auto_enhance(pil_img: Image.Image) -> Image.Image:
    """
    Automatic image quality enhancement:
      - CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel
      - Gray world white balance
      - Fast NL-Means denoising
    """
    print("[cv] Auto-enhancing …")
    bgr = pil_to_bgr(pil_img)

    # Gray-world white balance
    b, g, r   = cv2.split(bgr.astype(np.float32))
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    gray_mean = (b_mean + g_mean + r_mean) / 3
    bgr = cv2.merge([
        np.clip(b * (gray_mean / (b_mean + 1e-6)), 0, 255),
        np.clip(g * (gray_mean / (g_mean + 1e-6)), 0, 255),
        np.clip(r * (gray_mean / (r_mean + 1e-6)), 0, 255),
    ]).astype(np.uint8)

    # CLAHE on L channel in LAB
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    bgr   = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_Lab2BGR)

    # Fast denoising
    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)

    return bgr_to_pil(bgr)


# ═════════════════════════════════════════════════════════════════════════════
# 7. EDGE-AWARE SHARPENING
# ═════════════════════════════════════════════════════════════════════════════

def sharpen_edges(pil_img: Image.Image, strength: float = 1.5) -> Image.Image:
    """
    Edge-aware unsharp mask sharpening.
    Detects edges with Laplacian, sharpens only along edges — avoids
    amplifying noise in flat regions.
    """
    bgr    = pil_to_bgr(pil_img)
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Laplacian edge map → use as sharpening weight
    lap    = cv2.Laplacian(gray, cv2.CV_64F)
    edge_w = np.abs(lap)
    edge_w = (edge_w / (edge_w.max() + 1e-6)).astype(np.float32)
    edge_w = cv2.GaussianBlur(edge_w, (5,5), 0)[:,:,None]

    # Unsharp mask
    blurred = cv2.GaussianBlur(bgr, (0,0), sigmaX=2)
    sharp   = cv2.addWeighted(bgr, 1 + strength, blurred, -strength, 0)

    # Blend: sharpen proportional to edge strength
    result = (sharp.astype(np.float32) * edge_w +
              bgr.astype(np.float32)   * (1 - edge_w))
    return bgr_to_pil(np.clip(result, 0, 255).astype(np.uint8))


# ═════════════════════════════════════════════════════════════════════════════
# 8. COLOR TRANSFER  (Reinhard et al.)
# ═════════════════════════════════════════════════════════════════════════════

def color_transfer(source: Image.Image, reference: Image.Image) -> Image.Image:
    """
    Transfer the color palette of `reference` onto `source`.
    Uses Reinhard's LAB color transfer:
      - Match mean and std of each LAB channel independently
    """
    src = cv2.cvtColor(pil_to_bgr(source),    cv2.COLOR_BGR2Lab).astype(np.float32)
    ref = cv2.cvtColor(pil_to_bgr(reference), cv2.COLOR_BGR2Lab).astype(np.float32)

    for ch in range(3):
        src_mean, src_std = src[:,:,ch].mean(), src[:,:,ch].std() + 1e-6
        ref_mean, ref_std = ref[:,:,ch].mean(), ref[:,:,ch].std() + 1e-6
        src[:,:,ch] = (src[:,:,ch] - src_mean) * (ref_std / src_std) + ref_mean

    result_bgr = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
    return bgr_to_pil(result_bgr)


# ═════════════════════════════════════════════════════════════════════════════
# 9. DEPTH-BASED BACKGROUND BLUR (fake shallow DOF)
# ═════════════════════════════════════════════════════════════════════════════

def depth_blur(pil_img: Image.Image, blur_strength: int = 21) -> Image.Image:
    """
    Fake depth-of-field: estimate a depth map, keep foreground sharp,
    progressively blur background.

    Depth estimation: uses a fast gradient-based proxy (no MiDaS download needed).
    Gradient magnitude ≈ high-frequency detail ≈ likely foreground.
    This is a lightweight approximation — rembg mask is used if available to
    cleanly separate foreground from background.
    """
    print("[cv] Applying depth-based blur …")

    bgr  = pil_to_bgr(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Gradient-based sharpness map (proxy for foreground)
    gx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)

    # Smooth and normalise → foreground weight
    fg_weight = cv2.GaussianBlur(grad.astype(np.float32), (51,51), 0)
    fg_weight = np.clip(fg_weight / (np.percentile(fg_weight, 90) + 1e-6), 0, 1)
    fg_weight = fg_weight[:,:,None]

    # Apply varying blur strength based on depth
    blurred = cv2.GaussianBlur(bgr, (blur_strength | 1, blur_strength | 1), 0)

    result = (bgr.astype(np.float32)     * fg_weight +
              blurred.astype(np.float32) * (1 - fg_weight))
    return bgr_to_pil(np.clip(result, 0, 255).astype(np.uint8))


# ═════════════════════════════════════════════════════════════════════════════
# 10. LOCAL FILTERS  (instant — no model)
# ═════════════════════════════════════════════════════════════════════════════

FILTER_NAMES = [
    "vintage","warm","cool","dramatic","bright",
    "noir","dreamy","cinematic","neon","lofi","hdr","sunset",
]

def apply_filter(pil_img: Image.Image, filter_name: str) -> Image.Image:
    img = pil_img.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    if filter_name == "vintage":
        r = arr[:,:,0]*0.393+arr[:,:,1]*0.769+arr[:,:,2]*0.189
        g = arr[:,:,0]*0.349+arr[:,:,1]*0.686+arr[:,:,2]*0.168
        b = arr[:,:,0]*0.272+arr[:,:,1]*0.534+arr[:,:,2]*0.131
        arr = np.stack([r,g,b],axis=2)
        arr += np.random.normal(0,12,arr.shape)
        H,W = arr.shape[:2]
        Y,X = np.ogrid[:H,:W]
        dist = np.sqrt(((X-W/2)/(W/2))**2+((Y-H/2)/(H/2))**2)
        arr *= (1-np.clip(dist*0.7,0,0.6))[:,:,None]
        return from_arr(arr)

    elif filter_name == "warm":
        arr[:,:,0]=np.clip(arr[:,:,0]*1.15,0,255)
        arr[:,:,1]=np.clip(arr[:,:,1]*1.05,0,255)
        arr[:,:,2]=np.clip(arr[:,:,2]*0.85,0,255)
        return ImageEnhance.Color(from_arr(arr)).enhance(1.3)

    elif filter_name == "cool":
        arr[:,:,0]=np.clip(arr[:,:,0]*0.85,0,255)
        arr[:,:,2]=np.clip(arr[:,:,2]*1.2,0,255)
        return from_arr(arr)

    elif filter_name == "dramatic":
        out=ImageEnhance.Contrast(from_arr(arr)).enhance(1.8)
        out=ImageEnhance.Brightness(out).enhance(0.85)
        return ImageEnhance.Color(out).enhance(0.7)

    elif filter_name == "bright":
        out=ImageEnhance.Brightness(from_arr(arr)).enhance(1.3)
        out=ImageEnhance.Color(out).enhance(1.4)
        return ImageEnhance.Contrast(out).enhance(1.1)

    elif filter_name == "noir":
        gray=np.mean(arr,axis=2,keepdims=True)
        arr=np.repeat(gray,3,axis=2)
        return ImageEnhance.Contrast(from_arr(arr)).enhance(1.5)

    elif filter_name == "dreamy":
        out=Image.blend(img,img.filter(ImageFilter.GaussianBlur(3)),0.4)
        out=ImageEnhance.Brightness(out).enhance(1.15)
        out=ImageEnhance.Color(out).enhance(0.8)
        arr2=np.array(out,dtype=np.float32)
        return from_arr(arr2*0.85+30)

    elif filter_name == "cinematic":
        arr[:,:,0]=np.clip(arr[:,:,0]*1.1+10,0,255)
        arr[:,:,1]=np.clip(arr[:,:,1]*0.95,0,255)
        arr[:,:,2]=np.clip(arr[:,:,2]*0.9+20,0,255)
        out=ImageEnhance.Contrast(from_arr(arr)).enhance(1.25)
        a=np.array(out); bar=max(1,int(a.shape[0]*0.055))
        a[:bar,:]=0; a[-bar:,:]=0
        return Image.fromarray(a)

    elif filter_name == "neon":
        out=ImageEnhance.Color(from_arr(arr)).enhance(2.5)
        out=ImageEnhance.Contrast(out).enhance(1.5)
        out=ImageEnhance.Brightness(out).enhance(0.7)
        edges=ImageEnhance.Brightness(out.filter(ImageFilter.FIND_EDGES)).enhance(3)
        return Image.blend(out,edges,0.4)

    elif filter_name == "lofi":
        arr=arr*0.88+18
        arr[:,:,0]=np.clip(arr[:,:,0]*1.05,0,255)
        arr[:,:,2]=np.clip(arr[:,:,2]*0.90,0,255)
        out=ImageEnhance.Color(from_arr(arr)).enhance(0.75)
        return ImageEnhance.Contrast(out).enhance(0.9)

    elif filter_name == "hdr":
        cv_img=pil_to_bgr(img)
        lab=cv2.cvtColor(cv_img,cv2.COLOR_BGR2Lab).astype(np.float32)
        lab[:,:,0]=np.clip(lab[:,:,0]*1.1,0,255)
        sharp=cv2.detailEnhance(cv2.cvtColor(lab.astype(np.uint8),cv2.COLOR_Lab2BGR),sigma_s=12,sigma_r=0.15)
        return bgr_to_pil(sharp)

    elif filter_name == "sunset":
        arr[:,:,0]=np.clip(arr[:,:,0]*1.2+20,0,255)
        arr[:,:,1]=np.clip(arr[:,:,1]*0.9+10,0,255)
        arr[:,:,2]=np.clip(arr[:,:,2]*0.6,0,255)
        return ImageEnhance.Color(from_arr(arr)).enhance(1.4)

    else:
        print(f"[cv] Unknown filter '{filter_name}', returning original")
        return img