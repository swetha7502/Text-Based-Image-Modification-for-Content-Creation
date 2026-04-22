"""
intent_parser.py
================
Maps any free-form user prompt to a structured intent dict.

Three layers (fastest → smartest):
  1. Regex       — instant, handles obvious patterns
  2. Fuzzy       — keyword scoring, handles vague/slang with no API
  3. Claude API  — handles anything creative/ambiguous (needs ANTHROPIC_API_KEY)
"""

import os, re, json, textwrap, urllib.request, urllib.error

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── filter vocabulary ─────────────────────────────────────────────────────────

FILTER_MAP: dict[str, list[str]] = {
    "vintage"  : ["vintage", "retro", "old", "aged", "sepia", "film grain", "analog"],
    "warm"     : ["warm", "sunny", "golden", "summer", "cozy", "cosy", "orange", "toasty"],
    "cool"     : ["cool", "cold", "icy", "blue", "winter", "arctic", "crisp"],
    "dramatic" : ["dramatic", "moody", "dark", "intense", "gritty", "contrast", "stormy", "brooding"],
    "bright"   : ["bright", "vivid", "vibrant", "pop", "colourful", "colorful", "saturated", "punchy"],
    "noir"     : ["noir", "black and white", "bw", "grayscale", "greyscale", "monochrome", "desaturated"],
    "dreamy"   : ["dreamy", "soft", "pastel", "hazy", "ethereal", "romantic", "airy", "fairytale", "whimsical"],
    "cinematic": ["cinematic", "film", "movie", "letterbox", "hollywood", "teal orange", "thriller", "epic"],
    "neon"     : ["neon", "cyberpunk", "glow", "synthwave", "vaporwave", "electric", "glitch", "futuristic"],
    "lofi"     : ["lofi", "lo-fi", "chill", "muted", "faded", "washed", "coffee", "indie", "instagram", "cafe"],
    "hdr"      : ["hdr", "sharp", "detailed", "crisp", "ultra sharp", "clear"],
    "sunset"   : ["sunset", "dusk", "golden hour", "twilight", "amber", "magic hour"],
}

# flat keyword → filter_name lookup (longer keywords checked first)
_KW_LOOKUP: dict[str, str] = {}
for _fname, _kws in FILTER_MAP.items():
    for _kw in _kws:
        _KW_LOOKUP[_kw] = _fname
_KW_SORTED = sorted(_KW_LOOKUP, key=len, reverse=True)

# ── regex patterns ────────────────────────────────────────────────────────────

_REMOVE_RE  = re.compile(r"\b(remove|erase|delete|get rid of|take out|eliminate|clean up|wipe)\b", re.I)
_BG_RE      = re.compile(r"\b(background|bg|backdrop|scene|sky|floor|ground|wall)\b", re.I)
_REPLACE_RE = re.compile(r"\b(replace|swap|change|turn)\b", re.I)
_ADD_RE     = re.compile(r"\b(add|insert|put|place|include)\b", re.I)
_CUT_RE     = re.compile(
    r"\b(cut\s*out|remove\s*back\w*|isolate|extract\s*subject|transparent\s*bg|transparent\s*background)\b", re.I
)


def _regex_parse(prompt):
    p = prompt.lower().strip()

    if _CUT_RE.search(p):
        return {"action": "bg_remove"}

    if _BG_RE.search(p) and (_REPLACE_RE.search(p) or _ADD_RE.search(p)):
        return {"action": "bg_remove"}

    if _REMOVE_RE.search(p):
        obj = _REMOVE_RE.sub("", p).strip().lstrip("the ").strip(" .")
        return {"action": "remove", "object": obj or "object"}

    for kw in _KW_SORTED:
        if kw in p:
            return {"action": "filter", "filter": _KW_LOOKUP[kw]}

    return None


def _fuzzy_fallback(prompt):
    """Score every filter by how many / how long its keywords appear in the prompt."""
    p = prompt.lower()
    scores: dict[str, int] = {f: 0 for f in FILTER_MAP}
    for kw, fname in _KW_LOOKUP.items():
        if kw in p:
            scores[fname] += len(kw)
    best = max(scores, key=scores.get)
    chosen = best if scores[best] > 0 else "cinematic"
    print(f"[intent] fuzzy fallback → {chosen}")
    return {"action": "filter", "filter": chosen}


def _claude_parse(prompt):
    """Ask Claude to interpret the prompt. Falls back to fuzzy if unavailable."""
    if not ANTHROPIC_API_KEY:
        return _fuzzy_fallback(prompt)

    system = textwrap.dedent(f"""
        You are an image-editing intent classifier. Given a user's prompt, return ONLY
        a JSON object — no prose, no markdown fences.

        Possible JSON shapes:
          {{"action": "filter",    "filter": "<filter_name>"}}
          {{"action": "remove",    "object": "<what to remove>"}}
          {{"action": "bg_remove"}}

        Available filter names (you MUST pick the closest one):
          {list(FILTER_MAP.keys())}

        Rules:
        - Style / vibe / mood / aesthetic / look → action=filter
        - Delete / erase / remove a specific object → action=remove
        - Remove / cut out background → action=bg_remove
        - When unsure → action=filter with the most relevant filter_name
        - NEVER return anything except the raw JSON object
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
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        raw = data["content"][0]["text"].strip()
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw).strip()
        intent = json.loads(raw)
        print(f"[intent] Claude → {intent}")
        return intent
    except Exception as exc:
        print(f"[intent] Claude failed ({exc}), using fuzzy fallback")
        return _fuzzy_fallback(prompt)


def parse_intent(prompt):
    """
    Main entry. Returns one of:
      {"action": "filter",    "filter": str}
      {"action": "remove",    "object": str}
      {"action": "bg_remove"}
    """
    intent = _regex_parse(prompt)
    if intent is None:
        print(f"[intent] regex miss — escalating to Claude for: {prompt!r}")
        intent = _claude_parse(prompt)
    print(f"[intent] final → {intent}")
    return intent
