"""
sd_inpaint.py — Stable Diffusion inpainting for the replace_object pipeline.

Loaded lazily on first use. Falls back transparently if diffusers is not
installed or if the model download fails.

Input contract (from segment_object):
  pil_img  : PIL.Image RGB
  mask_np  : H×W uint8, 255 = region to replace, 0 = keep
  prompt   : plain-text description of what should appear in the masked region
"""

import torch
import numpy as np
from PIL import Image

_sd_pipe  = None
_SD_MODEL = "runwayml/stable-diffusion-inpainting"
_SD_SIZE  = 512   # SD 1.x native resolution

# Detect GPU once at import time
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE  = torch.float16 if _DEVICE == "cuda" else torch.float32


def _load_pipeline():
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe

    from diffusers import StableDiffusionInpaintPipeline

    print(f"[sd] Loading {_SD_MODEL} on {_DEVICE.upper()}")
    print("[sd] First run will download ~5 GB of model weights — please wait ...")

    _sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        _SD_MODEL,
        torch_dtype=_DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    _sd_pipe = _sd_pipe.to(_DEVICE)

    if _DEVICE == "cpu":
        _sd_pipe.enable_attention_slicing(1)

    print(f"[sd] Pipeline ready ({_DEVICE.upper()}, {'fp16' if _DTYPE == torch.float16 else 'fp32'}).")
    return _sd_pipe


def sd_replace(
    pil_img: Image.Image,
    mask_np: np.ndarray,
    prompt: str,
    negative_prompt: str = "blurry, deformed, ugly, bad anatomy, watermark, text, noise",
    num_steps: int = 20,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """
    Run SD inpainting on pil_img inside the region defined by mask_np.

    Args:
        pil_img:         Original image (PIL RGB).
        mask_np:         H×W uint8 mask from segment_object (255 = inpaint).
        prompt:          What to generate in the masked region.
        negative_prompt: What to avoid.
        num_steps:       Denoising steps (20 = fast, 50 = quality).
        guidance_scale:  Prompt adherence (7–9 is typical).

    Returns:
        PIL Image at the original resolution with the masked region replaced.
    """
    pipe = _load_pipeline()

    orig_w, orig_h = pil_img.size

    # Resize to SD native resolution
    img_sd   = pil_img.convert("RGB").resize((_SD_SIZE, _SD_SIZE), Image.LANCZOS)
    mask_sd  = Image.fromarray(mask_np).convert("L").resize((_SD_SIZE, _SD_SIZE), Image.NEAREST)

    print(f"[sd] Inpainting: '{prompt}' ({num_steps} steps) ...")
    result_sd = pipe(
        prompt          = prompt,
        negative_prompt = negative_prompt,
        image           = img_sd,
        mask_image      = mask_sd,
        num_inference_steps = num_steps,
        guidance_scale  = guidance_scale,
    ).images[0]

    # Restore original resolution
    return result_sd.resize((orig_w, orig_h), Image.LANCZOS)
