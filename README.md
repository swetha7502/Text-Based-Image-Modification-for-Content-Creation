# PixEdit — text-guided local image editor

> Upload image → type a prompt → get the edit. No cloud APIs required for core features.

---

## Project structure

```
pixedit/
├── app.py              ← Flask server (routes, image codec)
├── cv_pipeline.py      ← All CV operations (filters, segmentation, inpainting, bg removal)
├── intent_parser.py    ← Prompt → structured intent (regex → fuzzy → Claude)
├── requirements.txt
├── README.md
└── static/
    └── index.html      ← Frontend UI
```

---

## Setup

### 1. Install Python dependencies

```bash
cd pixedit
pip install -r requirements.txt
```

> **Note:** PyTorch is a transitive dependency of `segment-anything-py` and `simple-lama-inpainting`.
> If not already installed, pip will pull in the CPU-only version automatically.

### 2. (Optional) Smart intent parsing for unknown prompts

```bash
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

Without this, the parser uses keyword fuzzy-matching as fallback — still works well for most prompts.

### 3. Run

```bash
python app.py
```

Open **http://localhost:5000**

---

## What it does

| Prompt example | Action | Tech stack |
|---|---|---|
| `make it cinematic` | Filter | Local Pillow + NumPy |
| `vintage film grain` | Filter | Local Pillow + NumPy |
| `noir` / `dreamy` / `neon` | Filter | Local Pillow + NumPy |
| `remove the person` | Object removal | Grounded-DINO + SAM + LaMa |
| `erase the car in the back` | Object removal | Grounded-DINO + SAM + LaMa |
| `cut out background` | BG removal | rembg (U2-Net) |
| `lofi coffee shop vibe` | Filter (fuzzy matched) | Local |
| `dark stormy thriller mood` | Filter (Claude → dramatic) | Claude API |

---

## Model weights (downloaded automatically on first use)

| Model | Size | Purpose | Cache location |
|---|---|---|---|
| SAM ViT-B | ~375 MB | Pixel-level segmentation | `~/.pixedit_models/` |
| Grounded-DINO | ~700 MB | Text → bounding boxes | bundled with `groundingdino-py` |
| LaMa | ~200 MB | Inpainting / object removal | bundled with `simple-lama-inpainting` |
| rembg U2-Net | ~170 MB | Background removal | `~/.u2net/` |

Total first-run download: ~1.4 GB. Subsequent runs use cache.

---

## Intent parsing — 3 layers

```
User prompt
    │
    ▼
1. Regex            — instant; handles "remove X", "cut out background",
    │                  and exact filter keywords (noir, cinematic, etc.)
    │ (no match)
    ▼
2. Fuzzy scoring    — scores each filter by how many synonym keywords
    │                  appear in the prompt; works offline
    │ (Claude API available)
    ▼
3. Claude API       — handles anything: slang, vibes, metaphors, typos
                       e.g. "make it look like a rainy seoul night" → cinematic
```

---

## API endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/` | — | `index.html` |
| `POST` | `/edit` | `{image: dataURI, prompt: str}` | `{result: dataURI, action, detail?}` |
| `POST` | `/parse` | `{prompt: str}` | `{action, filter?/object?}` |
| `GET` | `/filters` | — | `{filters: [...]}` |

---

## Available filters

`vintage` · `warm` · `cool` · `dramatic` · `bright` · `noir` · `dreamy` · `cinematic` · `neon` · `lofi` · `hdr` · `sunset`

Each filter is implemented as a pure NumPy/Pillow/OpenCV transform — no model loading, instant results.

---

## Extending

**Add a new filter:**
1. Add it to `FILTER_MAP` in `intent_parser.py` with synonym keywords
2. Add a branch in `apply_filter()` in `cv_pipeline.py`

**Change port:**
```bash
PORT=8080 python app.py
```
