"""
intent_parser.py
================
Maps any free-form user prompt to a structured intent dict.

Three layers (fastest → smartest):
  1. Regex       — instant, handles obvious patterns
  2. Fuzzy       — keyword scoring, handles vague/slang with no API
  3. Claude API  — handles anything creative/ambiguous (needs ANTHROPIC_API_KEY)
"""

import os, re, json, textwrap, urllib.request

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── filter vocabulary ─────────────────────────────────────────────────────────
FILTER_MAP = {
    "vintage"  : ["vintage","retro","old","aged","sepia","film grain","analog"],
    "warm"     : ["warm","sunny","golden","summer","cozy","cosy","orange","toasty"],
    "cool"     : ["cool","cold","icy","blue","winter","arctic","crisp"],
    "dramatic" : ["dramatic","moody","dark","intense","gritty","contrast","stormy","brooding"],
    "bright"   : ["bright","vivid","vibrant","pop","colourful","colorful","saturated","punchy"],
    "noir"     : ["noir","black and white","bw","grayscale","greyscale","monochrome","desaturated"],
    "dreamy"   : ["dreamy","soft","pastel","hazy","ethereal","romantic","airy","fairytale","whimsical"],
    "cinematic": ["cinematic","film","movie","letterbox","hollywood","teal orange","thriller","epic"],
    "neon"     : ["neon","cyberpunk","glow","synthwave","vaporwave","electric","glitch","futuristic"],
    "lofi"     : ["lofi","lo-fi","chill","muted","faded","washed","coffee","indie","instagram","cafe"],
    "hdr"      : ["hdr","sharp","detailed","crisp","ultra sharp","clear"],
    "sunset"   : ["sunset","dusk","golden hour","twilight","amber","magic hour"],
}

_KW_LOOKUP = {}
for _fname, _kws in FILTER_MAP.items():
    for _kw in _kws:
        _KW_LOOKUP[_kw] = _fname
_KW_SORTED = sorted(_KW_LOOKUP, key=len, reverse=True)

# ── regex patterns ────────────────────────────────────────────────────────────
_REMOVE_RE  = re.compile(r"\b(remove|erase|delete|get rid of|take out|eliminate|clean up|wipe)\b", re.I)
_REPLACE_RE = re.compile(r"\b(replace|swap|change|turn|make|convert)\b", re.I)
_RECOLOR_RE = re.compile(r"\b(recolor|colour|color|repaint|dye|tint)\b", re.I)
_BG_RE      = re.compile(r"\b(background|bg|backdrop)\b", re.I)
_CUT_RE     = re.compile(r"\b(cut\s*out|remove\s*back\w*|isolate|extract\s*subject|transparent)\b", re.I)
_ENHANCE_RE = re.compile(r"\b(enhance|improve|fix|auto|denoise|clean\s*up|restore|upscale)\b", re.I)
_SHARPEN_RE = re.compile(r"\b(sharpen|crisp\s*up|focus|unblur|sharpness)\b", re.I)
_BLUR_RE    = re.compile(r"\b(blur\s*background|bokeh|depth\s*of\s*field|dof|shallow\s*focus)\b", re.I)

# replace patterns: "replace X with Y" / "change X to Y" / "turn X into Y"
_REPLACE_FULL_RE = re.compile(
    r"\b(?:replace|swap|change|turn|convert|make)\b\s+(?:the\s+)?(.+?)\s+(?:with|to|into|as|by)\s+(.+)",
    re.I
)

def _regex_parse(prompt: str):
    p = prompt.lower().strip()

    if _CUT_RE.search(p):
        return {"action": "bg_remove"}

    if _BG_RE.search(p) and _REPLACE_RE.search(p):
        return {"action": "bg_remove"}

    # recolor: "recolor the shirt blue" / "change shirt color to red"
    if _RECOLOR_RE.search(p):
        # try to extract object and target color
        m = re.search(r"(?:recolor|colour|color|repaint|dye|tint)\s+(?:the\s+)?(.+?)\s+(?:to\s+)?(\w+)$", p, re.I)
        if m:
            return {"action": "recolor", "object": m.group(1).strip(), "color": m.group(2).strip()}
        return {"action": "recolor", "object": "object", "color": "blue"}

    # replace: needs both object AND replacement
    if _REPLACE_FULL_RE.search(p):
        m = _REPLACE_FULL_RE.search(p)
        obj  = m.group(1).strip().rstrip("the ").strip()
        repl = m.group(2).strip()
        return {"action": "replace", "object": obj, "replacement": repl}

    # remove
    if _REMOVE_RE.search(p):
        obj = _REMOVE_RE.sub("", p).strip().lstrip("the ").strip(" .")
        return {"action": "remove", "object": obj or "object"}

    # enhance
    if _ENHANCE_RE.search(p):
        return {"action": "enhance"}

    # sharpen
    if _SHARPEN_RE.search(p):
        return {"action": "sharpen"}

    # depth blur
    if _BLUR_RE.search(p):
        return {"action": "depth_blur"}

    # filter keyword match
    for kw in _KW_SORTED:
        if kw in p:
            return {"action": "filter", "filter": _KW_LOOKUP[kw]}

    return None   # unknown → escalate


def _fuzzy_fallback(prompt: str) -> dict:
    p = prompt.lower()
    scores = {f: 0 for f in FILTER_MAP}
    for kw, fname in _KW_LOOKUP.items():
        if kw in p:
            scores[fname] += len(kw)
    best = max(scores, key=scores.get)
    chosen = best if scores[best] > 0 else "cinematic"
    print(f"[intent] fuzzy fallback -> {chosen}")
    return {"action": "filter", "filter": chosen}


def _claude_parse(prompt: str) -> dict:
    if not ANTHROPIC_API_KEY:
        return _fuzzy_fallback(prompt)

    system = textwrap.dedent(f"""
        You are an image-editing intent classifier. Given a user's prompt, return ONLY
        a JSON object — no prose, no markdown fences.

        Possible JSON shapes:
          {{"action": "filter",     "filter": "<filter_name>"}}
          {{"action": "remove",     "object": "<what to remove>"}}
          {{"action": "replace",    "object": "<what to replace>", "replacement": "<replace with what>"}}
          {{"action": "recolor",    "object": "<what to recolor>", "color": "<target color>"}}
          {{"action": "bg_remove"}}
          {{"action": "enhance"}}
          {{"action": "sharpen"}}
          {{"action": "depth_blur"}}

        Available filter names (pick closest):
          {list(FILTER_MAP.keys())}

        Rules:
        - Style/vibe/mood/aesthetic/look            → action=filter
        - Delete/erase/remove a specific object     → action=remove
        - Replace/swap/change X with/to Y           → action=replace
        - Recolor/repaint/change color of object    → action=recolor
        - Remove/cut out background                 → action=bg_remove
        - Enhance/improve/fix quality               → action=enhance
        - Sharpen/make crisp                        → action=sharpen
        - Blur background/bokeh/depth of field      → action=depth_blur
        - When unsure                               → action=filter, pick best filter
        - NEVER return anything except the raw JSON
    """).strip()

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 150,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        import re as _re
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        raw    = data["content"][0]["text"].strip()
        raw    = _re.sub(r"^```[a-z]*\n?|```$", "", raw).strip()
        intent = json.loads(raw)
        print(f"[intent] Claude -> {intent}")
        return intent
    except Exception as exc:
        print(f"[intent] Claude failed ({exc}), fuzzy fallback")
        return _fuzzy_fallback(prompt)


def parse_intent(prompt: str) -> dict:
    intent = _regex_parse(prompt)
    if intent is None:
        print(f"[intent] regex miss -> escalating: {prompt!r}")
        intent = _claude_parse(prompt)
    print(f"[intent] final -> {intent}")
    return intent