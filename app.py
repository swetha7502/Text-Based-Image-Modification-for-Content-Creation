"""
app.py — Flask backend for PixEdit
Routes:
  GET  /          → index.html
  POST /edit      → {image, prompt} → {result, action, detail?}
  POST /parse     → {prompt} → intent dict
  GET  /filters   → list of filter names
"""

import io, base64, os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from intent_parser import parse_intent, FILTER_MAP
from cv_pipeline import (
    apply_filter, segment_object, inpaint_remove, replace_object,
    remove_background, recolor_object, auto_enhance,
    sharpen_edges, depth_blur, FILTER_NAMES,
)

app = Flask(__name__, static_folder="static")
CORS(app)

# ── image codec ───────────────────────────────────────────────────────────────

def _decode(data_uri: str) -> Image.Image:
    b64 = data_uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def _encode(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    if img.mode == "RGBA" and fmt == "PNG":
        img.save(buf, format="PNG")
    else:
        img.convert("RGB").save(buf, format=fmt)
    b64  = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/filters")
def filters():
    return jsonify({"filters": FILTER_NAMES})

@app.route("/parse", methods=["POST"])
def parse_route():
    data = request.get_json(silent=True) or {}
    return jsonify(parse_intent(data.get("prompt", "")))

@app.route("/edit", methods=["POST"])
def edit():
    data   = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    uri    = data.get("image")

    if not prompt: return jsonify({"error": "prompt is required"}), 400
    if not uri:    return jsonify({"error": "image is required"}), 400

    try:
        pil_img = _decode(uri)
    except Exception as exc:
        return jsonify({"error": f"Could not decode image: {exc}"}), 400

    intent = parse_intent(prompt)
    action = intent.get("action", "filter")

    try:
        # ── filter ────────────────────────────────────────────────────────
        if action == "filter":
            fname  = intent.get("filter", "cinematic")
            result = apply_filter(pil_img, fname)
            return jsonify({"result": _encode(result), "action": action, "detail": fname})

        # ── remove ────────────────────────────────────────────────────────
        elif action == "remove":
            obj  = intent.get("object", "object")
            mask = segment_object(pil_img, obj)
            if mask is None:
                return jsonify({"error": f"Could not find '{obj}' in the image."}), 422
            result = inpaint_remove(pil_img, mask)
            return jsonify({"result": _encode(result), "action": action, "detail": obj})

        # ── replace ───────────────────────────────────────────────────────
        elif action == "replace":
            obj  = intent.get("object", "object")
            repl = intent.get("replacement", "")
            if not repl:
                return jsonify({"error": "Could not determine what to replace with."}), 422
            result = replace_object(pil_img, obj, repl)
            return jsonify({"result": _encode(result), "action": action,
                            "detail": f"{obj} → {repl}"})

        # ── recolor ───────────────────────────────────────────────────────
        elif action == "recolor":
            obj   = intent.get("object", "object")
            color = intent.get("color", "blue")
            result = recolor_object(pil_img, obj, color)
            return jsonify({"result": _encode(result), "action": action,
                            "detail": f"{obj} → {color}"})

        # ── background removal ────────────────────────────────────────────
        elif action == "bg_remove":
            result = remove_background(pil_img)
            return jsonify({"result": _encode(result, fmt="PNG"), "action": action})

        # ── auto enhance ──────────────────────────────────────────────────
        elif action == "enhance":
            result = auto_enhance(pil_img)
            return jsonify({"result": _encode(result), "action": action})

        # ── sharpen ───────────────────────────────────────────────────────
        elif action == "sharpen":
            result = sharpen_edges(pil_img)
            return jsonify({"result": _encode(result), "action": action})

        # ── depth blur ────────────────────────────────────────────────────
        elif action == "depth_blur":
            result = depth_blur(pil_img)
            return jsonify({"result": _encode(result), "action": action})

        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400

    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  PixEdit ->  http://localhost:{port}\n")
    app.run(debug=True, port=port, use_reloader=False)