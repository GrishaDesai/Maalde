import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
import base64
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFilter
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── Load .env file ────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Demand Oracle",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_clean")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── Secrets from environment (set in .env or system env) ─────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")
OPENROUTER_KEY  = os.getenv("OPENROUTER_KEY", "")

# ── Gemini: multiple fallback keys ───────────────────────────────────────────
# Set GEMINI_API_KEY for one key, or GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
# for multiple. All non-empty keys are collected and rotated automatically.
def _load_gemini_keys() -> list[str]:
    keys = []
    # Check numbered keys first: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
    for i in range(1, 11):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    # Also check plain GEMINI_API_KEY
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k and k not in keys:
        keys.append(k)
    return keys

GEMINI_KEYS     = _load_gemini_keys()   # list of all available Gemini keys
GEMINI_MODEL    = "gemini-2.0-flash"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"   # OpenRouter model string


# ─────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0c0c0f;
    --surface:  #131318;
    --surface2: #1a1a22;
    --border:   rgba(255,255,255,0.07);
    --accent:   #e8b86d;
    --accent2:  #c47f3a;
    --text:     #e8e4dc;
    --muted:    #7a7880;
    --success:  #6fcf8e;
    --radius:   14px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1400px; }

body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.6;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 300;
    letter-spacing: -0.02em;
    line-height: 1.05;
    color: var(--text);
    margin: 0;
}
.hero-title span { color: var(--accent); font-style: italic; }
.hero-sub {
    font-size: 0.82rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.4rem;
}
.hero-line {
    height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, transparent 70%);
    margin: 1.5rem 0 2rem;
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--accent2) !important;
    border-radius: var(--radius) !important;
    padding: 2rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 30px rgba(232,184,109,0.08) !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }

.img-card {
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.img-card img { width: 100%; display: block; }
.img-card-label {
    padding: 0.8rem 1rem;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    border-top: 1px solid var(--border);
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #1a0e00 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(196,127,58,0.3) !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(196,127,58,0.4) !important;
}

.metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
.metric-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.4); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.6;
}
.metric-card.highlight::before { background: linear-gradient(90deg, var(--accent2), var(--accent)); opacity: 1; }
.metric-label { font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.metric-value { font-family: 'Cormorant Garamond', serif; font-size: 2.4rem; font-weight: 300; color: var(--text); line-height: 1; }
.metric-card.highlight .metric-value { color: var(--accent); }
.metric-unit { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

.section-head {
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.section-head::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.tag-chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.25rem 0.8rem;
    font-size: 0.72rem;
    color: var(--muted);
    margin: 0.2rem;
}
.tag-chip.warm { border-color: rgba(232,184,109,0.3); color: var(--accent); background: rgba(232,184,109,0.06); }
.tag-chip.warn { border-color: rgba(255,100,100,0.3); color: #ff9090; background: rgba(255,80,80,0.06); }

.warn-box {
    background: rgba(232,184,109,0.05);
    border: 1px solid rgba(232,184,109,0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 1rem;
}
.warn-box-title { font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent2); margin-bottom: 0.4rem; }
.warn-item { font-size: 0.75rem; color: var(--muted); padding: 0.15rem 0; }

.divider { height: 1px; background: var(--border); margin: 2rem 0; }
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }
</style>
"""


# ─────────────────────────────────────────
# DRIVE HELPERS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_image_map_from_drive(folder_id: str, api_key: str) -> dict:
    if not api_key or not folder_id:
        return {}
    import urllib.request, urllib.parse
    image_map  = {}
    page_token = None
    while True:
        params = {
            "q"        : f"'{folder_id}' in parents and trashed=false",
            "fields"   : "nextPageToken,files(id,name)",
            "pageSize" : "1000",
            "key"      : api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            st.warning(f"Drive API error: {e}")
            break
        for f in data.get("files", []):
            code = os.path.splitext(f["name"])[0]
            image_map[code] = f["id"]
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return image_map


@st.cache_data(show_spinner=False)
def fetch_drive_image_b64(file_id: str, api_key: str) -> str | None:
    import urllib.request
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    except Exception:
        return None


def get_product_image_b64(product_code: str, image_map: dict) -> str | None:
    fid = image_map.get(str(product_code).strip())
    if fid and GOOGLE_API_KEY:
        return fetch_drive_image_b64(fid, GOOGLE_API_KEY)
    return None


# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = {
        "model.pkl"          : os.path.join(MODEL_DIR, "model.pkl"),
        "ohe.pkl"            : os.path.join(MODEL_DIR, "ohe.pkl"),
        "scaler.pkl"         : os.path.join(MODEL_DIR, "scaler.pkl"),
        "config.json"        : os.path.join(MODEL_DIR, "config.json"),
        "resnet_features.pkl": os.path.join(MODEL_DIR, "resnet_features.pkl"),
    }
    missing = [k for k, v in required.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")

    model        = joblib.load(required["model.pkl"])
    ohe          = joblib.load(required["ohe.pkl"])
    scaler       = joblib.load(required["scaler.pkl"])
    resnet_train = joblib.load(required["resnet_features.pkl"])
    with open(required["config.json"]) as f:
        config = json.load(f)

    # training_data.csv — check multiple locations
    csv_candidates = [
        os.path.join(MODEL_DIR, "training_data.csv"),
        os.path.join(BASE_DIR,  "final_dataset_training.csv"),
    ]
    train_df = None
    for p in csv_candidates:
        if os.path.exists(p):
            train_df = pd.read_csv(p)
            break
    if train_df is None:
        raise FileNotFoundError("training_data.csv not found in models_clean/ or project root")

    return model, ohe, scaler, config, train_df, resnet_train


# ─────────────────────────────────────────
# LOAD RESNET
# ─────────────────────────────────────────
RESNET_WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"

@st.cache_resource
def load_resnet():
    device       = torch.device("cpu")
    weights_path = os.path.join(DATA_DIR, "resnet50-11ad3fa6.pth")

    resnet = models.resnet50(weights=None)

    if os.path.exists(weights_path):
        # Local file found (not gitignored on this machine)
        resnet.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        # Not found locally — download via torchvision (uses torch hub cache)
        # Cache location: ~/.cache/torch/hub/checkpoints/
        # Downloaded once, reused on every subsequent run
        cache_dir  = os.path.join(torch.hub.get_dir(), "checkpoints")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "resnet50-11ad3fa6.pth")
        if not os.path.exists(cache_path):
            with st.spinner("⬇️ Downloading ResNet50 weights (~100 MB, once only)…"):
                torch.hub.download_url_to_file(
                    RESNET_WEIGHTS_URL,
                    cache_path,
                    progress=False
                )
        resnet.load_state_dict(torch.load(cache_path, map_location=device))

    extractor  = nn.Sequential(*list(resnet.children())[:-1])
    extractor.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return extractor, preprocess, device


# ─────────────────────────────────────────
# GEMINI — dynamic prompt + multi-key rotation + OpenRouter fallback
# ─────────────────────────────────────────

def build_gemini_prompt(ohe) -> str:
    """Build prompt using exact OHE category values — prevents unknown values."""
    cat_map = dict(zip(ohe.feature_names_in_, ohe.categories_))

    def opts(col):
        vals = [str(v) for v in cat_map.get(col, [])]
        return " or ".join(f'"{v}"' for v in vals) if vals else '"unknown"'

    return f"""
Analyze this kurti/dress product image and extract attributes.

Return ONLY a valid JSON object with EXACTLY these keys and ONLY these allowed values:
{{
  "fabric"        : "cotton" or "silk" or "georgette" or "rayon" or "linen" or "polyester" or "unknown",
  "length"        : {opts("length")},
  "sleeves"       : {opts("sleeves")},
  "neck"          : {opts("neck")},
  "work"          : {opts("work")},
  "pattern"       : {opts("pattern")},
  "occasion"      : {opts("occasion")},
  "price_tier"    : {opts("price_tier") if "price_tier" in cat_map else '"low" or "mid" or "high"'},
  "primary_color" : main color as simple lowercase (e.g. "red", "blue", "green", "pink", "yellow"),
  "dupatta"       : true or false,
  "pants_included": true or false
}}

CRITICAL: Use ONLY the exact string values listed above. No markdown. No backticks. Return ONLY JSON.
"""


def _is_quota_error(result: dict) -> bool:
    """Return True if the API response indicates quota/rate-limit exhaustion."""
    code = result.get("error", {}).get("code", 0)
    msg  = result.get("error", {}).get("message", "").lower()
    return code == 429 or any(w in msg for w in ("quota", "rate limit", "resource_exhausted"))


def _encode_image(image: Image.Image) -> str:
    """Encode PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _call_gemini_key(prompt: str, b64_img: str, api_key: str) -> tuple[str | None, bool]:
    """
    Call Gemini with a single API key.
    Returns (text, quota_exhausted).
      - text set, quota_exhausted=False  -> success
      - text None, quota_exhausted=True  -> rotate to next key
      - text None, quota_exhausted=False -> hard failure
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
        ]}],
        "generationConfig": {"temperature": 0}
    }
    backoff = [5, 15, 30]
    for attempt in range(3):
        try:
            r      = requests.post(url, json=payload, timeout=90)
            result = r.json()
            if "candidates" in result:
                return result["candidates"][0]["content"]["parts"][0]["text"].strip(), False
            if _is_quota_error(result):
                return None, True   # quota -> rotate
            return None, False      # other API error
        except requests.exceptions.Timeout:
            import time
            if attempt < 2:
                time.sleep(backoff[attempt])
            else:
                return None, False
        except Exception:
            return None, False
    return None, False


def call_gemini_with_rotation(prompt: str, image: Image.Image) -> tuple[str | None, str]:
    """
    Try each Gemini key in GEMINI_KEYS, rotating on quota errors.
    Returns (response_text, source_label).
    """
    if not GEMINI_KEYS:
        return None, "no_keys"

    b64_img = _encode_image(image)

    for idx, key in enumerate(GEMINI_KEYS):
        text, quota_hit = _call_gemini_key(prompt, b64_img, key)
        if text is not None:
            return text, f"gemini_key_{idx + 1}"
        if quota_hit and idx < len(GEMINI_KEYS) - 1:
            st.toast(f"Gemini key {idx + 1} quota exhausted -> trying key {idx + 2}...", icon="🔄")

    return None, "all_gemini_exhausted"


def call_openrouter(prompt: str, image: Image.Image) -> str | None:
    """
    OpenRouter fallback — uses google/gemini-2.0-flash-001 via OpenAI-compatible API.
    Called only when all Gemini keys are exhausted.
    """
    if not OPENROUTER_KEY:
        return None

    b64_img = _encode_image(image)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type" : "application/json",
        "HTTP-Referer" : "https://maalde.app",
        "X-Title"      : "Maalde Demand Oracle",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }],
        "temperature": 0,
        "max_tokens" : 500,
    }
    try:
        r      = requests.post("https://openrouter.ai/api/v1/chat/completions",
                               headers=headers, json=payload, timeout=90)
        result = r.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        err = result.get("error", {}).get("message", "Unknown OpenRouter error")
        st.warning(f"OpenRouter error: {err}")
        return None
    except requests.exceptions.Timeout:
        st.warning("OpenRouter timed out.")
        return None
    except Exception as e:
        st.warning(f"OpenRouter call failed: {e}")
        return None


def _parse_tags(raw: str) -> dict | None:
    """Parse JSON from model response. Returns dict or None."""
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        return None


def extract_tags_gemini(img: Image.Image, ohe) -> tuple[dict, str]:
    """
    Full extraction pipeline:
      1. Gemini key 1 -> key 2 -> ... -> key N  (rotate on quota exhaustion)
      2. OpenRouter                              (if all Gemini keys exhausted)
      3. Fallback defaults                       (if everything fails)

    Returns (tags_dict, source_label).
    """
    prompt = build_gemini_prompt(ohe)

    # Step 1: Try Gemini keys
    raw, source = call_gemini_with_rotation(prompt, img)

    if raw:
        tags = _parse_tags(raw)
        if tags:
            return tags, source
        st.warning("Gemini returned unparseable JSON — trying OpenRouter...")

    # Step 2: OpenRouter fallback — triggered whenever Gemini returned nothing
    gemini_failed = raw is None or source in ("all_gemini_exhausted", "no_keys")
    if gemini_failed:
        if OPENROUTER_KEY:
            st.toast("Switching to OpenRouter (all Gemini keys exhausted)...", icon="🌐")
            raw_or = call_openrouter(prompt, img)
            if raw_or:
                tags = _parse_tags(raw_or)
                if tags:
                    return tags, "openrouter"
            st.warning("OpenRouter also failed — using fallback tags")
        else:
            st.warning("All Gemini keys exhausted. Set OPENROUTER_KEY in .env as backup.")

    # Step 3: Hard fallback
    return _fallback_tags(ohe), "fallback"


def _fallback_tags(ohe) -> dict:
    """Safe defaults using first known value for each OHE column."""
    tags = {}
    for col, cats in zip(ohe.feature_names_in_, ohe.categories_):
        tags[col] = str(cats[0])
    tags.setdefault("dupatta",        False)
    tags.setdefault("pants_included", False)
    tags.setdefault("primary_color",  "unknown")
    tags.setdefault("fabric",         "unknown")
    return tags


# FEATURE EXTRACTION
# ─────────────────────────────────────────
def extract_resnet_features(img, extractor, preprocess, device):
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(tensor).squeeze()
    return feat.cpu().numpy()


def get_color_features(img):
    feats     = {}
    img_small = img.resize((100, 100))
    pixels    = np.array(img_small).reshape(-1, 3).astype(float)

    brightness = (0.299*pixels[:,0] + 0.587*pixels[:,1] + 0.114*pixels[:,2]) / 255.0
    feats["brightness_mean_y"] = float(brightness.mean())

    gray = np.array(img_small.convert("L")).astype(float) / 255.0
    feats["contrast_y"] = float(gray.std())

    rg = pixels[:,0] - pixels[:,1]
    yb = 0.5*(pixels[:,0]+pixels[:,1]) - pixels[:,2]
    feats["colorfulness_y"] = float(np.sqrt(rg.std()**2 + yb.std()**2)) / 255.0

    gray_img = img.resize((128, 128)).convert("L")
    edges    = gray_img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges).astype(float)

    feats["edge_density_y"]       = float(edge_arr.mean() / 255.0)
    feats["texture_complexity_y"] = float(np.array(gray_img).std() / 255.0)
    feats["color_variety_y"]      = min(feats["edge_density_y"] * 10, 1.0)
    return feats


# ─────────────────────────────────────────
# SAFE OHE + FEATURE VECTOR
# ─────────────────────────────────────────
def safe_ohe_transform(ohe, cat_row: dict) -> tuple[np.ndarray, list]:
    """
    Transform categorical row safely.
    - Known value  → use directly
    - Partial match → use closest
    - Unknown      → fallback to first known category
    Returns (encoded_array, warnings_list).
    """
    warnings = []
    safe_row = {}

    for col, known_cats in zip(ohe.feature_names_in_, ohe.categories_):
        raw      = str(cat_row.get(col, "")).lower().strip()
        known    = [str(k).lower() for k in known_cats]

        if raw in known:
            safe_row[col] = raw
        else:
            matched = next((k for k in known if raw in k or k in raw), None)
            if matched:
                safe_row[col] = matched
                warnings.append(f"<b>{col}</b>: '{raw}' → partial match → '{matched}'")
            else:
                safe_row[col] = known[0]
                warnings.append(f"<b>{col}</b>: '{raw}' unknown → fallback to '{known[0]}'")

    encoded = ohe.transform(pd.DataFrame([safe_row]))
    return encoded, warnings


def build_features(llm_tags: dict, color_feats: dict, resnet_feats: np.ndarray,
                   config: dict, ohe, scaler) -> tuple[np.ndarray, list]:
    cat_cols  = config["cat_cols"]
    bool_cols = config["bool_cols"]
    num_cols  = config["num_cols"]

    # ── Categorical → safe OHE ────────────────────────────────────────────────
    cat_row              = {col: str(llm_tags.get(col, "")).lower().strip() for col in cat_cols}
    X_cat, ohe_warnings = safe_ohe_transform(ohe, cat_row)

    # ── Boolean ───────────────────────────────────────────────────────────────
    X_bool = np.array([
        1 if llm_tags.get(col, False) else 0
        for col in bool_cols
    ]).reshape(1, -1)

    # ── Numeric — use training means for sales/price cols ─────────────────────
    num_vals = []
    for col in num_cols:
        if col in ("rate_avg", "days_on_market", "num_orders", "sales_velocity"):
            col_mean = config.get(f"{col}_mean", 1.0)
            num_vals.append(float(col_mean))
        else:
            num_vals.append(float(color_feats.get(col, 0.0)))

    X_num    = scaler.transform(np.array(num_vals).reshape(1, -1))
    X_resnet = resnet_feats.reshape(1, -1)
    X        = np.hstack([X_cat, X_bool, X_num, X_resnet])
    return X, ohe_warnings


# ─────────────────────────────────────────
# SIMILARITY PREDICTION
# ─────────────────────────────────────────
def similarity_prediction_resnet(query_feat, train_feats, train_df, top_k=5):
    sims       = cosine_similarity(query_feat.reshape(1, -1), train_feats)[0]
    top_idx    = np.argsort(sims)[-top_k:][::-1]
    similar_df = train_df.iloc[top_idx].copy()
    similar_df["similarity_score"] = sims[top_idx]
    sim_pred   = float(similar_df["qty_total"].mean())
    return sim_pred, similar_df


# ─────────────────────────────────────────
# SIMILAR PRODUCT CARDS
# ─────────────────────────────────────────
def render_sim_cards_grid(similar_df: pd.DataFrame, image_map: dict, cols_per_row: int = 5):
    rows = [similar_df.iloc[i:i+cols_per_row] for i in range(0, len(similar_df), cols_per_row)]

    for row_df in rows:
        cols = st.columns(len(row_df))
        for col, (_, row) in zip(cols, row_df.iterrows()):
            code  = str(row.get("product_code", "")).strip()
            name  = str(row.get("product_name", "Product"))
            price = row.get("rate_avg", None)
            qty   = row.get("qty_total", None)
            color = str(row.get("primary_color", "")).title()
            work  = str(row.get("work", "")).title()
            patt  = str(row.get("pattern", "")).title()
            score = float(row.get("similarity_score", 0))

            with col:
                img_b64 = get_product_image_b64(code, image_map)
                if img_b64:
                    st.markdown(
                        f'<div style="border-radius:10px;overflow:hidden;aspect-ratio:3/4;background:#1a1a22;">'
                        f'<img src="{img_b64}" style="width:100%;height:100%;object-fit:cover;display:block;" />'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="border-radius:10px;background:#1a1a22;aspect-ratio:3/4;'
                        'display:flex;align-items:center;justify-content:center;font-size:2.5rem;opacity:0.4;">👗</div>',
                        unsafe_allow_html=True
                    )

                tags_html = " ".join([
                    f'<span style="font-size:0.6rem;background:#1a1a22;border-radius:4px;padding:2px 6px;color:#7a7880;">{t}</span>'
                    for t in [color, work, patt] if t and t.lower() not in ("", "nan", "none")
                ])
                qty_html = (
                    f'<span style="font-size:0.65rem;color:#6fcf8e;margin-left:4px;">· {int(qty)} sold</span>'
                    if qty and not pd.isna(qty) else ""
                )
                st.markdown(f"""
                <div style="padding:0.5rem 0.2rem;">
                    <div style="font-size:0.78rem;font-weight:500;color:#e8e4dc;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                        title="{name}">{name[:28]}{"…" if len(name)>28 else ""}</div>
                    <div style="font-size:0.65rem;color:#7a7880;letter-spacing:0.06em;margin-top:2px;">{code}</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;">
                        <span style="font-family:'Cormorant Garamond',serif;font-size:1.1rem;color:#e8b86d;">
                            {"₹{:,.0f}".format(price) if price and not pd.isna(price) else "—"}
                        </span>
                        <span style="font-size:0.62rem;background:rgba(232,184,109,0.1);
                            border:1px solid rgba(232,184,109,0.25);color:#e8b86d;
                            border-radius:100px;padding:2px 7px;">{score:.2f}</span>
                    </div>
                    {qty_html}
                    <div style="margin-top:5px;display:flex;flex-wrap:wrap;gap:3px;">{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div>
        <p class="hero-sub">Fashion Intelligence · AI-Powered</p>
        <h1 class="hero-title">Demand <span>Oracle</span></h1>
        <div class="hero-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ───────────────────────────────────────────────────────────
    try:
        model, ohe, scaler, config, train_df, resnet_train = load_artifacts()
        extractor, preprocess, device = load_resnet()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    # ── Drive image map ───────────────────────────────────────────────────────
    with st.spinner("🔗 Connecting to Drive image library…"):
        image_map = build_image_map_from_drive(DRIVE_FOLDER_ID, GOOGLE_API_KEY)

    # ── Sidebar: connection status ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ System Status")

        # Drive
        if image_map:
            st.success(f"✅ {len(image_map)} images linked from Drive")
        elif GOOGLE_API_KEY and DRIVE_FOLDER_ID:
            st.error("❌ Drive images could not be loaded")
        else:
            st.warning("⚠️ Drive images disabled — set GOOGLE_API_KEY + DRIVE_FOLDER_ID in .env")

        # Gemini
        if GEMINI_KEYS:
            st.success(f"✅ {len(GEMINI_KEYS)} Gemini key(s) loaded")
        else:
            st.warning("⚠️ No Gemini keys set — tags will use fallback values")

        if OPENROUTER_KEY:
            st.success("✅ OpenRouter ready (backup)")
        else:
            st.info("💡 Set OPENROUTER_KEY for Gemini quota backup")

        st.markdown("---")
        st.markdown("**How to configure (.env):**")
        st.code("""# One Gemini key
GEMINI_API_KEY=your_key

# OR multiple fallback keys
GEMINI_API_KEY_1=key_one
GEMINI_API_KEY_2=key_two
GEMINI_API_KEY_3=key_three

# OpenRouter (backup when all Gemini exhausted)
OPENROUTER_KEY=your_openrouter_key

# Google Drive for product images
GOOGLE_API_KEY=your_google_key
DRIVE_FOLDER_ID=your_folder_id""", language="bash")

    # ── Layout ────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 1.8], gap="large")

    with left:
        st.markdown('<div class="section-head">Upload Product Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg", "png", "jpeg", "webp"], label_visibility="collapsed")

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64_preview = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div class="img-card">
                <img src="data:image/png;base64,{b64_preview}" />
                <div class="img-card-label">📷 {uploaded.name}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("✦  Run Demand Prediction")
        else:
            predict_btn = False
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:var(--muted);
                font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase;">
                Drop a product image to begin
            </div>
            """, unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    with right:
        if uploaded and predict_btn:

            with st.spinner("Analysing image with Gemini…"):
                color_feats  = get_color_features(img)
                resnet_feats = extract_resnet_features(img, extractor, preprocess, device)
                llm_tags, tag_source = extract_tags_gemini(img, ohe)

            with st.spinner("Computing demand forecast…"):
                X, ohe_warnings = build_features(
                    llm_tags, color_feats, resnet_feats, config, ohe, scaler
                )
                model_pred           = float(np.expm1(model.predict(X)[0]))
                sim_pred, similar_df = similarity_prediction_resnet(
                    resnet_feats, resnet_train, train_df
                )
                final_pred = 0.5 * model_pred + 0.5 * sim_pred

                st.session_state["similar_df"] = similar_df

            # ── Prediction metrics ────────────────────────────────────────
            st.markdown('<div class="section-head">Prediction Results</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-label">ML Model</div>
                    <div class="metric-value">{round(model_pred)}</div>
                    <div class="metric-unit">units predicted</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Similarity</div>
                    <div class="metric-value">{round(sim_pred)}</div>
                    <div class="metric-unit">units (neighbours)</div>
                </div>
                <div class="metric-card highlight">
                    <div class="metric-label">Final Forecast</div>
                    <div class="metric-value">{round(final_pred)}</div>
                    <div class="metric-unit">units · 50/50 blend</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Detected tags ─────────────────────────────────────────────
            if tag_source == "openrouter":
                source_badge = '<span class="tag-chip warm">✦ OpenRouter</span>'
            elif tag_source == "fallback":
                source_badge = '<span class="tag-chip warn">⚠ Fallback</span>'
            elif tag_source.startswith("gemini_key_"):
                key_num = tag_source.split("_")[-1]
                source_badge = f'<span class="tag-chip warm">✦ Gemini (key {key_num})</span>'
            else:
                source_badge = '<span class="tag-chip warm">✦ Gemini</span>'
            st.markdown(
                f'<div class="section-head">Detected Attributes {source_badge}</div>',
                unsafe_allow_html=True
            )

            # Build chips — green for categorical (known), plain for others
            cat_map = dict(zip(ohe.feature_names_in_, ohe.categories_))
            chips   = ""
            for k, v in llm_tags.items():
                if v is None or v is False:
                    continue
                label    = k.replace("_", " ").title()
                val_str  = str(v)
                is_cat   = k in cat_map
                known    = [str(c).lower() for c in cat_map.get(k, [])]
                is_known = str(v).lower() in known if is_cat else True
                chip_cls = "tag-chip warm" if is_known else "tag-chip warn"
                chips   += f'<span class="{chip_cls}">{label}: {val_str}</span>'

            st.markdown(f'<div style="margin-bottom:1rem">{chips}</div>', unsafe_allow_html=True)

            # ── Visual analytics ──────────────────────────────────────────
            st.markdown('<div class="section-head">Visual Analytics</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Brightness",   f"{color_feats['brightness_mean_y']:.2f}")
            c2.metric("Colorfulness", f"{color_feats['colorfulness_y']:.2f}")
            c3.metric("Edge Density", f"{color_feats['edge_density_y']:.2f}")

            # ── OHE warnings ──────────────────────────────────────────────
            if ohe_warnings:
                warn_items = "".join([f'<div class="warn-item">→ {w}</div>' for w in ohe_warnings])
                st.markdown(f"""
                <div class="warn-box">
                    <div class="warn-box-title">⚠ Encoding Adjustments</div>
                    {warn_items}
                </div>
                """, unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;height:300px;
                color:var(--muted);font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase;">
                Results will appear here
            </div>
            """, unsafe_allow_html=True)

    # ── Similar products (full width) ─────────────────────────────────────────
    if uploaded and predict_btn and st.session_state.get("similar_df") is not None:
        similar_df = st.session_state["similar_df"]
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Similar Products · Visual Neighbours</div>', unsafe_allow_html=True)

        render_sim_cards_grid(similar_df, image_map, cols_per_row=5)

        with st.expander("View raw data table"):
            disp_cols = [c for c in [
                "product_code", "product_name", "rate_avg", "qty_total",
                "primary_color", "work", "pattern", "similarity_score"
            ] if c in similar_df.columns]
            st.dataframe(
                similar_df[disp_cols].sort_values("qty_total", ascending=False),
                use_container_width=True
            )


if __name__ == "__main__":
    main()