"""
sd_inpaint.py — Crop-based SD inpainting for the replace_object pipeline.

Strategy:
  1. Crop the bounding box of the mask (+ padding) — reduces background
     context so the model has less "complete the scene" bias.
  2. Run StableDiffusionInpaintPipeline on the crop with guidance=15
     (strong prompt adherence) — model generates the replacement object.
  3. Resize crop result back, composite with a soft Gaussian mask.

Uses runwayml/stable-diffusion-inpainting (already cached locally).
"""

import cv2
import torch
import numpy as np
from PIL import Image

_sd_pipe  = None
_SD_MODEL = "runwayml/stable-diffusion-inpainting"
_SD_SIZE  = 512

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE  = torch.float16 if _DEVICE == "cuda" else torch.float32


def _load_pipeline():
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe

    from diffusers import StableDiffusionInpaintPipeline

    print(f"[sd] Loading {_SD_MODEL} on {_DEVICE.upper()} ...")
    _sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        _SD_MODEL,
        torch_dtype=_DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    _sd_pipe = _sd_pipe.to(_DEVICE)
    if _DEVICE == "cpu":
        _sd_pipe.enable_attention_slicing(1)
    print(f"[sd] Ready ({_DEVICE.upper()}, {'fp16' if _DTYPE == torch.float16 else 'fp32'}).")
    return _sd_pipe


def sd_replace(
    pil_img: Image.Image,
    mask_np: np.ndarray,
    prompt: str,
    negative_prompt: str = (
        "blurry, deformed, ugly, bad anatomy, watermark, text, "
        "duplicate, out of frame, artifacts, low quality, background"
    ),
    num_steps: int = 30,
    guidance_scale: float = 15.0,
) -> Image.Image:
    """
    Replace the masked object using crop-based SD inpainting.

    Crops to the mask bounding box so the model sees mostly the object
    region rather than the full background, then inpaints at guidance=15
    for strong text-prompt adherence.
    """
    pipe = _load_pipeline()

    orig_arr = np.array(pil_img.convert("RGB"))
    H, W = orig_arr.shape[:2]

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return pil_img

    # Crop to the mask bounding box + padding
    pad = 40
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(H, int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(W, int(xs.max()) + pad)

    crop_pil  = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    mask_crop = Image.fromarray(mask_np[y1:y2, x1:x2]).convert("L")
    crop_w, crop_h = crop_pil.size

    # Resize crop + mask to SD native resolution
    crop_sd = crop_pil.resize((_SD_SIZE, _SD_SIZE), Image.LANCZOS)
    mask_sd = mask_crop.resize((_SD_SIZE, _SD_SIZE), Image.NEAREST)

    print(f"[sd] Inpainting crop: '{prompt}'  cfg={guidance_scale}  steps={num_steps} ...")
    result_sd = pipe(
        prompt              = prompt,
        negative_prompt     = negative_prompt,
        image               = crop_sd,
        mask_image          = mask_sd,
        num_inference_steps = num_steps,
        guidance_scale      = guidance_scale,
    ).images[0]

    # Resize generated crop back to original crop dimensions
    result_crop = np.array(result_sd.resize((crop_w, crop_h), Image.LANCZOS), dtype=np.float32)
    orig_crop   = orig_arr[y1:y2, x1:x2].astype(np.float32)

    # Soft Gaussian mask for smooth blending at object boundary
    mask_f    = mask_np[y1:y2, x1:x2].astype(np.float32) / 255.0
    soft_mask = cv2.GaussianBlur(mask_f, (31, 31), 0)[:, :, None]

    blended           = result_crop * soft_mask + orig_crop * (1.0 - soft_mask)
    result_arr        = orig_arr.copy().astype(np.float32)
    result_arr[y1:y2, x1:x2] = blended

    return Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8))
