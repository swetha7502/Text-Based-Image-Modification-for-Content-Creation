"""
cv_pipeline.py — all CV ops, forced CPU-only
"""

# ── Step 1: hide CUDA from the runtime before any other import ────────────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Step 2: import torch and neutralise every CUDA entry point ────────────────
import torch
torch.cuda.is_available   = lambda: False
torch.cuda.device_count   = lambda: 0
torch.cuda.current_device = lambda: 0
DEVICE = torch.device("cpu")

# ── Step 3: patch SimpleLama's baked-in default arg BEFORE it is used ─────────
import simple_lama_inpainting.models.model as _lama_mod
_lama_mod.SimpleLama.__init__.__defaults__ = (torch.device("cpu"),)

# ── Now safe to import everything else ────────────────────────────────────────
import pathlib, urllib.request, warnings
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance

warnings.filterwarnings("ignore")
print("[cv] CPU-only mode active")

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

def _download(url: str, dest: pathlib.Path, label: str):
    if dest.exists():
        return
    print(f"[cv] Downloading {label} …")
    def _prog(b, bs, total):
        print(f"\r    {min(b*bs,total)/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_prog)
    print()

# ── loaders ───────────────────────────────────────────────────────────────────

def _load_gdino():
    global _gdino_model
    if _gdino_model:
        return _gdino_model
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
    if _sam_predictor:
        return _sam_predictor
    _download(SAM_URL, SAM_CHECKPOINT, "SAM ViT-B weights (~375 MB)")
    print("[cv] Loading SAM ViT-B …")
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE)
    sam.eval()
    _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def _load_lama():
    global _lama_model
    if _lama_model:
        return _lama_model
    print("[cv] Loading LaMa …")
    from simple_lama_inpainting import SimpleLama
    _lama_model = SimpleLama(device=DEVICE)
    return _lama_model

# ── helpers ───────────────────────────────────────────────────────────────────

def pil_to_bgr(img):  return np.array(img.convert("RGB"))[:,:,::-1]
def bgr_to_pil(arr):  return Image.fromarray(arr[:,:,::-1])
def from_arr(arr):    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

# ─────────────────────────────────────────────────────────────────────────────
# 1. SEGMENTATION  (Grounded-DINO + SAM)
# ─────────────────────────────────────────────────────────────────────────────

def segment_object(pil_img: Image.Image, text: str):
    # MUST use groundingdino's own transforms — NOT torchvision.
    # groundingdino's T.Compose takes (image, target) pairs.
    # Using torchvision here causes "ValueError: not supported" inside nested_tensor_from_tensor_list.
    import groundingdino.datasets.transforms as GDT
    from groundingdino.util.inference import predict

    gdino     = _load_gdino()
    predictor = _load_sam()
    rgb       = np.array(pil_img.convert("RGB"))
    h, w      = rgb.shape[:2]

    transform = GDT.Compose([
        GDT.RandomResize([800], max_size=1333),
        GDT.ToTensor(),
        GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # groundingdino transforms take (PIL_image, target) — pass None for target
    img_tensor, _ = transform(pil_img.convert("RGB"), None)
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model          = gdino,
            image          = img_tensor,
            caption        = text,
            box_threshold  = 0.30,
            text_threshold = 0.25,
            device         = "cpu",   # predict() defaults to "cuda" — must override
        )

    if boxes is None or len(boxes) == 0:
        print(f"[cv] '{text}' not found in image")
        return None

    # cx,cy,w,h (normalised) → x1,y1,x2,y2 (pixels)
    b = boxes.clone()
    boxes_xyxy = torch.stack([
        (b[:,0] - b[:,2]/2) * w,
        (b[:,1] - b[:,3]/2) * h,
        (b[:,0] + b[:,2]/2) * w,
        (b[:,1] + b[:,3]/2) * h,
    ], dim=1).numpy()
    print(f"[cv] {len(boxes_xyxy)} box(es) found: {phrases}")

    predictor.set_image(rgb)
    combined = np.zeros((h, w), dtype=bool)
    for box in boxes_xyxy:
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        combined |= masks[np.argmax(scores)]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.dilate(combined.astype(np.uint8) * 255, kernel, iterations=2)

# ─────────────────────────────────────────────────────────────────────────────
# 2. INPAINTING  (LaMa)
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_remove(pil_img: Image.Image, mask: np.ndarray) -> Image.Image:
    lama = _load_lama()
    return lama(pil_img.convert("RGB"), Image.fromarray(mask).convert("L"))

# ─────────────────────────────────────────────────────────────────────────────
# 3. BACKGROUND REMOVAL  (rembg)
# ─────────────────────────────────────────────────────────────────────────────

def remove_background(pil_img: Image.Image) -> Image.Image:
    print("[cv] Running rembg …")
    from rembg import remove as rembg_remove
    return rembg_remove(pil_img.convert("RGB"))

# ─────────────────────────────────────────────────────────────────────────────
# 4. FILTERS  (local, instant — no model needed)
# ─────────────────────────────────────────────────────────────────────────────

FILTER_NAMES = [
    "vintage", "warm", "cool", "dramatic", "bright",
    "noir", "dreamy", "cinematic", "neon", "lofi", "hdr", "sunset",
]

def apply_filter(pil_img: Image.Image, filter_name: str) -> Image.Image:
    img = pil_img.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    if filter_name == "vintage":
        r = arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189
        g = arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168
        b = arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131
        arr = np.stack([r, g, b], axis=2)
        arr += np.random.normal(0, 12, arr.shape)
        H, W = arr.shape[:2]
        Y, X = np.ogrid[:H, :W]
        dist  = np.sqrt(((X-W/2)/(W/2))**2 + ((Y-H/2)/(H/2))**2)
        arr  *= (1 - np.clip(dist*0.7, 0, 0.6))[:,:,None]
        return from_arr(arr)

    elif filter_name == "warm":
        arr[:,:,0] = np.clip(arr[:,:,0]*1.15, 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1]*1.05, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2]*0.85, 0, 255)
        return ImageEnhance.Color(from_arr(arr)).enhance(1.3)

    elif filter_name == "cool":
        arr[:,:,0] = np.clip(arr[:,:,0]*0.85, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2]*1.2,  0, 255)
        return from_arr(arr)

    elif filter_name == "dramatic":
        out = ImageEnhance.Contrast(from_arr(arr)).enhance(1.8)
        out = ImageEnhance.Brightness(out).enhance(0.85)
        return ImageEnhance.Color(out).enhance(0.7)

    elif filter_name == "bright":
        out = ImageEnhance.Brightness(from_arr(arr)).enhance(1.3)
        out = ImageEnhance.Color(out).enhance(1.4)
        return ImageEnhance.Contrast(out).enhance(1.1)

    elif filter_name == "noir":
        gray = np.mean(arr, axis=2, keepdims=True)
        arr  = np.repeat(gray, 3, axis=2)
        return ImageEnhance.Contrast(from_arr(arr)).enhance(1.5)

    elif filter_name == "dreamy":
        out  = Image.blend(img, img.filter(ImageFilter.GaussianBlur(3)), 0.4)
        out  = ImageEnhance.Brightness(out).enhance(1.15)
        out  = ImageEnhance.Color(out).enhance(0.8)
        arr2 = np.array(out, dtype=np.float32)
        return from_arr(arr2*0.85 + 30)

    elif filter_name == "cinematic":
        arr[:,:,0] = np.clip(arr[:,:,0]*1.1  + 10, 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1]*0.95,      0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2]*0.9  + 20, 0, 255)
        out = ImageEnhance.Contrast(from_arr(arr)).enhance(1.25)
        a   = np.array(out)
        bar = max(1, int(a.shape[0]*0.055))
        a[:bar,:] = 0; a[-bar:,:] = 0
        return Image.fromarray(a)

    elif filter_name == "neon":
        out   = ImageEnhance.Color(from_arr(arr)).enhance(2.5)
        out   = ImageEnhance.Contrast(out).enhance(1.5)
        out   = ImageEnhance.Brightness(out).enhance(0.7)
        edges = ImageEnhance.Brightness(out.filter(ImageFilter.FIND_EDGES)).enhance(3)
        return Image.blend(out, edges, 0.4)

    elif filter_name == "lofi":
        arr   = arr*0.88 + 18
        arr[:,:,0] = np.clip(arr[:,:,0]*1.05, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2]*0.90, 0, 255)
        out = ImageEnhance.Color(from_arr(arr)).enhance(0.75)
        return ImageEnhance.Contrast(out).enhance(0.9)

    elif filter_name == "hdr":
        cv_img = pil_to_bgr(img)
        lab    = cv2.cvtColor(cv_img, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab[:,:,0] = np.clip(lab[:,:,0]*1.1, 0, 255)
        sharp  = cv2.detailEnhance(
            cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR),
            sigma_s=12, sigma_r=0.15,
        )
        return bgr_to_pil(sharp)

    elif filter_name == "sunset":
        arr[:,:,0] = np.clip(arr[:,:,0]*1.2 + 20, 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1]*0.9  + 10, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2]*0.6,       0, 255)
        return ImageEnhance.Color(from_arr(arr)).enhance(1.4)

    else:
        print(f"[cv] Unknown filter '{filter_name}', returning original")
        return img