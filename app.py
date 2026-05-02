import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
import base64
import requests
import colorsys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────
load_dotenv()

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Maalde — Demand Prediction Engine",
    page_icon="👗",
    layout="wide"
)

# ── Absolute paths ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODEL_DIR    = os.path.join(BASE_DIR, "modelsXG")
WEIGHTS_PATH = os.path.join(DATA_DIR, "resnet50-11ad3fa6.pth")

# ── Load API keys from .env ───────────────────────────────────────
def _load_keys(prefix):
    """Load all keys matching prefix_1, prefix_2... from env."""
    keys = []
    for i in range(1, 10):
        k = os.getenv(f"{prefix}_{i}", "").strip()
        if k and not k.startswith("your_"):
            keys.append(k)
    return keys

GEMINI_KEYS       = _load_keys("GEMINI_KEY")
OPENROUTER_KEYS   = _load_keys("OPENROUTER_KEY")
PREFERRED_PROVIDER= os.getenv("PREFERRED_PROVIDER", "gemini").lower()
OPENROUTER_MODEL  = os.getenv("OPENROUTER_MODEL", "openrouter/free")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE_URL   = "https://generativelanguage.googleapis.com/v1beta/models"

# ── Key rotation state (session-level) ───────────────────────────
def _init_key_state():
    if "gemini_key_idx"     not in st.session_state:
        st.session_state.gemini_key_idx     = 0
    if "openrouter_key_idx" not in st.session_state:
        st.session_state.openrouter_key_idx = 0
    if "provider"           not in st.session_state:
        st.session_state.provider           = PREFERRED_PROVIDER
    if "api_log"            not in st.session_state:
        st.session_state.api_log            = []

def _log(msg):
    st.session_state.api_log.append(msg)

def _next_gemini_key():
    keys = GEMINI_KEYS
    if not keys:
        return None
    idx = st.session_state.gemini_key_idx % len(keys)
    st.session_state.gemini_key_idx = (idx + 1) % len(keys)
    return keys[idx]

def _next_openrouter_key():
    keys = OPENROUTER_KEYS
    if not keys:
        return None
    idx = st.session_state.openrouter_key_idx % len(keys)
    st.session_state.openrouter_key_idx = (idx + 1) % len(keys)
    return keys[idx]

# ── Gemini API call ───────────────────────────────────────────────
def _call_gemini_single(key, prompt, image):
    """Single Gemini API call. Returns text or raises exception."""
    url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={key}"
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
        ]}],
        "generationConfig": {"temperature": 0}
    }
    r      = requests.post(url, json=payload, timeout=60)
    result = r.json()

    if "candidates" in result:
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    error  = result.get("error", {})
    code   = error.get("code", 0)
    msg    = error.get("message", str(result))

    if code in (429, 403) or "quota" in msg.lower():
        raise QuotaExhaustedException(f"Gemini quota: {msg}")
    raise Exception(f"Gemini error {code}: {msg}")


# ── OpenRouter API call ───────────────────────────────────────────
def _call_openrouter_single(key, prompt, image):
    """Single OpenRouter API call. Returns text or raises exception."""
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model"   : OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }}
        ]}],
        "temperature": 0,
        "max_tokens" : 1000,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type" : "application/json",
        "HTTP-Referer" : "https://maalde.com",
        "X-Title"      : "Maalde Demand Prediction",
    }
    r      = requests.post(OPENROUTER_URL, json=payload,
                           headers=headers, timeout=60)
    result = r.json()

    if "choices" in result:
        return result["choices"][0]["message"]["content"].strip()

    error = result.get("error", {})
    code  = error.get("code", 0)
    msg   = error.get("message", str(result))

    if code in (429, 402) or "quota" in msg.lower() or "credit" in msg.lower():
        raise QuotaExhaustedException(f"OpenRouter quota: {msg}")
    raise Exception(f"OpenRouter error {code}: {msg}")


class QuotaExhaustedException(Exception):
    pass


# ── Smart API caller with auto key + provider rotation ────────────
def call_vision_api(prompt, image):
    """
    Tries keys in this order:
    1. All keys of preferred provider
    2. All keys of fallback provider
    3. Returns None if all exhausted
    """
    _init_key_state()

    providers_order = (
        ["gemini", "openrouter"]
        if st.session_state.provider == "gemini"
        else ["openrouter", "gemini"]
    )

    for provider in providers_order:
        if provider == "gemini":
            keys     = GEMINI_KEYS
            call_fn  = _call_gemini_single
            label    = "Gemini"
        else:
            keys     = OPENROUTER_KEYS
            call_fn  = _call_openrouter_single
            label    = "OpenRouter"

        if not keys:
            _log(f"⚠️ No {label} keys configured — skipping")
            continue

        # Try each key of this provider
        for attempt in range(len(keys)):
            key = _next_gemini_key() if provider == "gemini" else _next_openrouter_key()
            try:
                result = call_fn(key, prompt, image)
                _log(f"✅ {label} key ***{key[-6:]} succeeded")
                st.session_state.provider = provider  # remember what worked
                return result

            except QuotaExhaustedException as e:
                _log(f"⚠️ {label} key ***{key[-6:]} quota exhausted → rotating")
                continue

            except Exception as e:
                _log(f"❌ {label} key ***{key[-6:]} error: {str(e)[:80]}")
                continue

        _log(f"❌ All {label} keys exhausted → switching provider")

    _log("❌ All providers and keys exhausted")
    return None


# ── Model loading ─────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    required = {
        "demand_model.pkl"   : os.path.join(MODEL_DIR, "demand_model.pkl"),
        "ohe_encoder.pkl"    : os.path.join(MODEL_DIR, "ohe_encoder.pkl"),
        "pca.pkl"            : os.path.join(MODEL_DIR, "pca.pkl"),
        "feature_config.json": os.path.join(MODEL_DIR, "feature_config.json"),
        "training_data.csv"  : os.path.join(MODEL_DIR, "training_data.csv"),
    }
    missing = [k for k, v in required.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"Missing files in modelsXG/: {', '.join(missing)}")

    model    = joblib.load(required["demand_model.pkl"])
    ohe      = joblib.load(required["ohe_encoder.pkl"])
    pca      = joblib.load(required["pca.pkl"])
    with open(required["feature_config.json"]) as f:
        config = json.load(f)
    train_df = pd.read_csv(required["training_data.csv"])
    return model, ohe, pca, config, train_df


@st.cache_resource
def load_resnet():
    device = torch.device('cpu')
    if os.path.exists(WEIGHTS_PATH):
        resnet = models.resnet50(weights=None)
        resnet.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    else:
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    extractor  = nn.Sequential(*list(resnet.children())[:-1])
    extractor.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    return extractor, preprocess, device


# ── Feature extraction ────────────────────────────────────────────
def get_color_features(img):
    feats     = {}
    img_small = img.resize((100, 100))
    pixels    = np.array(img_small).reshape(-1, 3).astype(float)

    feats['r_mean'] = pixels[:,0].mean() / 255.0
    feats['g_mean'] = pixels[:,1].mean() / 255.0
    feats['b_mean'] = pixels[:,2].mean() / 255.0
    feats['r_std']  = pixels[:,0].std()  / 255.0
    feats['g_std']  = pixels[:,1].std()  / 255.0
    feats['b_std']  = pixels[:,2].std()  / 255.0

    brightness = (0.299*pixels[:,0] + 0.587*pixels[:,1] + 0.114*pixels[:,2]) / 255.0
    feats['brightness_mean'] = float(brightness.mean())
    feats['brightness_std']  = float(brightness.std())

    gray = np.array(img_small.convert('L')).astype(float) / 255.0
    feats['contrast'] = float(gray.std())

    rg = pixels[:,0] - pixels[:,1]
    yb = 0.5*(pixels[:,0]+pixels[:,1]) - pixels[:,2]
    feats['colorfulness'] = float(
        np.sqrt(rg.std()**2 + yb.std()**2) + 0.3*np.sqrt(rg.mean()**2 + yb.mean()**2)
    ) / 255.0

    feats['warm_cool_bias'] = feats['r_mean'] - feats['b_mean']

    sample = pixels[:500].astype(int)
    hsv    = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) for r,g,b in sample])
    feats['hue_mean']        = float(hsv[:,0].mean())
    feats['hue_std']         = float(hsv[:,0].std())
    feats['saturation_mean'] = float(hsv[:,1].mean())
    feats['saturation_std']  = float(hsv[:,1].std())
    feats['value_mean']      = float(hsv[:,2].mean())

    gray_img = img.resize((128,128)).convert('L')
    edges    = gray_img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges).astype(float)
    feats['edge_density']       = float(edge_arr.mean() / 255.0)
    feats['edge_variance']      = float(edge_arr.var() / (255.0**2))
    feats['texture_complexity'] = float(np.array(gray_img).std() / 255.0)

    arr         = np.array(gray_img).astype(float)
    block_vars  = [arr[r:r+32, c:c+32].var()
                   for r in range(0,128,32) for c in range(0,128,32)]
    feats['local_contrast_mean'] = float(np.mean(block_vars) / (255.0**2))
    feats['local_contrast_std']  = float(np.std(block_vars)  / (255.0**2))
    feats['color_variety']       = min(feats['edge_density'] * 10, 1.0)
    return feats


def extract_resnet_features(img, extractor, preprocess, device):
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = extractor(tensor).squeeze(-1).squeeze(-1)
    return out.cpu().numpy()[0]


# ── Dynamic Gemini prompt from OHE categories ─────────────────────
def build_prompt(ohe):
    cat_map = dict(zip(ohe.feature_names_in_, ohe.categories_))
    def opts(col):
        vals = [str(v) for v in cat_map.get(col, [])]
        return ' or '.join(f'"{v}"' for v in vals) if vals else '"unknown"'
    return f"""
Analyze this kurti/dress product image and extract attributes.
Return ONLY a valid JSON object with EXACTLY these keys and ONLY these allowed values:
{{
  "length"         : {opts('length')},
  "sleeves"        : {opts('sleeves')},
  "neck"           : {opts('neck')},
  "work"           : {opts('work')},
  "pattern"        : {opts('pattern')},
  "occasion"       : {opts('occasion')},
  "dupatta"        : true or false,
  "pants_included" : true or false,
  "primary_color"  : main color as simple lowercase name (e.g. "teal", "red", "olive_green"),
  "secondary_color": second color or null,
  "fabric"         : fabric type
}}
CRITICAL: Use ONLY the exact string values listed. Return ONLY the JSON. No markdown. No backticks.
"""


# ── Safe OHE transform ────────────────────────────────────────────
def safe_ohe_transform(ohe, cat_row):
    warnings_list = []
    safe_row      = {}
    for col, known_cats in zip(ohe.feature_names_in_, ohe.categories_):
        raw   = str(cat_row.get(col, '')).lower().strip()
        known = [str(c).lower() for c in known_cats]
        if raw in known:
            safe_row[col] = raw
        else:
            matched = next((k for k in known if raw in k or k in raw), None)
            if matched:
                safe_row[col] = matched
                warnings_list.append(f"[{col}] '{raw}' → '{matched}'")
            else:
                fallback = str(known_cats[0])
                safe_row[col] = fallback
                warnings_list.append(f"[{col}] '{raw}' unknown → fallback '{fallback}'")
    encoded = ohe.transform(pd.DataFrame([safe_row]))
    return encoded, warnings_list


# ── Feature vector builder ────────────────────────────────────────
def build_feature_vector(llm_tags, color_feats, resnet_feats, config, ohe, pca):
    cat_cols        = config.get('cat_cols', [])
    bool_cols       = config.get('bool_cols', [])
    num_interp_cols = config.get('num_interp_cols', [])

    cat_row              = {col: str(llm_tags.get(col,'')).lower().strip()
                            for col in cat_cols}
    cat_encoded, warnings = safe_ohe_transform(ohe, cat_row)

    bool_vals = [1 if llm_tags.get(col, False) else 0 for col in bool_cols]

    num_vals = []
    for col in num_interp_cols:
        if col in ('rate_avg','num_orders','days_on_market','sales_velocity'):
            num_vals.append(float(config.get(f'{col}_mean', 1.0)))
        else:
            num_vals.append(float(color_feats.get(col, 0.0)))

    resnet_pca = pca.transform(resnet_feats.reshape(1, -1))

    X = np.hstack([
        cat_encoded,
        np.array(bool_vals).reshape(1,-1),
        np.array(num_vals).reshape(1,-1),
        resnet_pca
    ])
    return X, warnings


# ── Similar product finder ────────────────────────────────────────
def find_similar(llm_tags, train_df, n=5):
    score_cols = ['work','pattern','occasion','length','primary_color']
    available  = [c for c in score_cols if c in train_df.columns]
    if not available:
        return train_df.head(n)
    scores = [
        sum(str(row.get(c,'')).lower() == str(llm_tags.get(c,'')).lower()
            for c in available)
        for _, row in train_df.iterrows()
    ]
    df         = train_df.copy()
    df['_sim'] = scores
    return df.nlargest(n, '_sim').drop(columns='_sim')


# ── Sidebar ───────────────────────────────────────────────────────
def render_sidebar():
    _init_key_state()
    with st.sidebar:
        st.title("⚙️ Settings")

        # ── Provider status ───────────────────────────────────────
        st.markdown("**🔑 API Provider Status**")

        g_count  = len(GEMINI_KEYS)
        or_count = len(OPENROUTER_KEYS)

        st.markdown(f"**Gemini keys loaded:** {'✅ ' + str(g_count) if g_count else '❌ None'}")
        st.markdown(f"**OpenRouter keys loaded:** {'✅ ' + str(or_count) if or_count else '❌ None'}")
        st.markdown(f"**Active provider:** `{st.session_state.provider}`")
        st.markdown(f"**Preferred:** `{PREFERRED_PROVIDER}`")

        # Manual override
        override = st.selectbox("Override provider", ["auto","gemini","openrouter"])
        if override != "auto":
            st.session_state.provider = override

        # ── Model info ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**📊 Model Info**")
        try:
            _, _, _, config, train_df = load_model_artifacts()
            st.metric("Training Samples", config.get('n_training_samples', len(train_df)))
            st.metric("CV MAE",    f"±{config.get('cv_mae', 8.3):.1f} units")
            st.metric("CV R²",     f"{config.get('cv_r2', 0):.3f}")
            st.metric("Best Model", config.get('best_model', 'XGBoost'))
            st.success("✅ Model loaded")
        except Exception as e:
            st.error(f"❌ {e}")

        # ── API call log ──────────────────────────────────────────
        if st.session_state.api_log:
            st.markdown("---")
            st.markdown("**📋 API Log**")
            for entry in st.session_state.api_log[-8:]:
                st.caption(entry)
            if st.button("Clear log"):
                st.session_state.api_log = []

        st.markdown("---")
        st.caption("Maalde Demand Prediction Engine v3.0")


# ── Tab 1: Predict ────────────────────────────────────────────────
def render_predict_tab():
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Design Image")
        uploaded = st.file_uploader(
            "Choose a kurti image",
            type=['jpg','jpeg','png','webp'],
            label_visibility="collapsed"
        )
        if not uploaded:
            st.info("👆 Upload a kurti image to get started")

            # Show key config status
            if not GEMINI_KEYS and not OPENROUTER_KEYS:
                st.error("❌ No API keys configured. Add keys to your .env file.")
            return

        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Uploaded Design", use_column_width=True)

        # Show which provider will be used
        provider_label = st.session_state.provider
        if provider_label == "gemini" and GEMINI_KEYS:
            st.info(f"🔑 Using Gemini ({len(GEMINI_KEYS)} keys) → OpenRouter fallback")
        elif provider_label == "openrouter" and OPENROUTER_KEYS:
            st.info(f"🔑 Using OpenRouter ({len(OPENROUTER_KEYS)} keys) → Gemini fallback")
        else:
            st.warning("⚠️ No API keys in .env — prediction uses visual features only")

        predict_btn = st.button("🔮 Predict Demand", type="primary",
                                use_container_width=True)

    if not predict_btn or not uploaded:
        return

    # ── Load model ────────────────────────────────────────────────
    try:
        model, ohe, pca, config, train_df = load_model_artifacts()
    except Exception as e:
        st.error(f"❌ Model not found: {e}\n\nRun the training notebook first.")
        return

    # ── Extract visual features ───────────────────────────────────
    with st.spinner("Extracting visual features..."):
        try:
            extractor, preprocess, device = load_resnet()
            color_feats  = get_color_features(img)
            resnet_feats = extract_resnet_features(img, extractor, preprocess, device)
        except Exception as e:
            st.error(f"❌ Feature extraction failed: {e}")
            return

    # ── LLM tags via auto-rotating provider ──────────────────────
    llm_tags = {}
    has_keys = bool(GEMINI_KEYS or OPENROUTER_KEYS)

    if has_keys:
        with st.spinner(f"Analyzing design with AI ({st.session_state.provider})..."):
            prompt = build_prompt(ohe)
            raw    = call_vision_api(prompt, img)
            if raw:
                try:
                    cleaned  = raw.replace('```json','').replace('```','').strip()
                    llm_tags = json.loads(cleaned)
                except json.JSONDecodeError:
                    st.warning("⚠️ Could not parse AI response — using visual features only")
            else:
                st.warning("⚠️ All API keys exhausted — using visual features only")
    else:
        st.warning("⚠️ No API keys in .env file — prediction based on visual features only")

    # ── Predict ───────────────────────────────────────────────────
    with st.spinner("Predicting demand..."):
        try:
            X, ohe_warnings = build_feature_vector(
                llm_tags, color_feats, resnet_feats, config, ohe, pca
            )
            log_pred = float(model.predict(X)[0])
            pred_qty = max(1, round(np.expm1(log_pred)))
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    if ohe_warnings:
        with st.expander("⚠️ Attribute remapping warnings"):
            for w in ohe_warnings:
                st.write(f"• {w}")

    st.session_state.update({
        'prediction'  : pred_qty,
        'llm_tags'    : llm_tags,
        'color_feats' : color_feats,
        'train_df'    : train_df,
        'config'      : config,
        'log_pred'    : log_pred,
        'ohe'         : ohe,
        'ohe_warnings': ohe_warnings,
    })

    with col2:
        _render_results(pred_qty, log_pred, llm_tags, color_feats,
                        train_df, config, ohe, ohe_warnings)


# ── Results panel ─────────────────────────────────────────────────
def _render_results(pred_qty, log_pred, llm_tags, color_feats,
                    train_df, config, ohe, ohe_warnings):
    y_mean = config.get('y_mean', pred_qty)
    y_max  = config.get('y_max', max(pred_qty * 2, 1))

    st.subheader("📦 Prediction Result")

    if pred_qty >= y_mean * 1.5:
        level, color, emoji = "HIGH DEMAND",     "#28a745", "🔥"
    elif pred_qty >= y_mean * 0.7:
        level, color, emoji = "MODERATE DEMAND", "#ffc107", "📈"
    else:
        level, color, emoji = "LOW DEMAND",      "#dc3545", "📉"

    st.markdown(f"""
    <div style='background:{color}22; border-left:4px solid {color};
                padding:20px; border-radius:8px; margin-bottom:16px'>
        <h1 style='color:{color}; margin:0'>{emoji} {pred_qty} Units</h1>
        <p style='color:{color}; margin:4px 0 0 0; font-weight:bold'>{level}</p>
        <p style='color:gray; margin:4px 0 0 0'>Predicted total quantity sold</p>
    </div>
    """, unsafe_allow_html=True)

    pct = min(pred_qty / max(y_max, 1), 1.0)
    st.markdown("**Demand Strength**")
    st.progress(pct)
    st.caption(f"Avg: {y_mean:.0f} | Max: {y_max:.0f} | log_pred: {log_pred:.3f}")

    # LLM tags
    if llm_tags:
        st.subheader("🏷️ Detected Attributes")
        cat_map = dict(zip(ohe.feature_names_in_, ohe.categories_))
        tag_display = {
            '📏 Length'  : ('length',   llm_tags.get('length','—')),
            '👕 Sleeves' : ('sleeves',  llm_tags.get('sleeves','—')),
            '👔 Neck'    : ('neck',     llm_tags.get('neck','—')),
            '✨ Work'    : ('work',     llm_tags.get('work','—')),
            '🌸 Pattern' : ('pattern',  llm_tags.get('pattern','—')),
            '🎉 Occasion': ('occasion', llm_tags.get('occasion','—')),
            '🎨 Color'   : (None,       llm_tags.get('primary_color','—')),
            '🧵 Fabric'  : (None,       llm_tags.get('fabric','—')),
            '🧣 Dupatta' : (None,       '✅' if llm_tags.get('dupatta') else '❌'),
            '👖 Pants'   : (None,       '✅' if llm_tags.get('pants_included') else '❌'),
        }
        cols = st.columns(2)
        for i, (label, (ohe_col, val)) in enumerate(tag_display.items()):
            display_val = str(val).replace('_',' ').title()
            if ohe_col and ohe_col in cat_map:
                known = [str(c).lower() for c in cat_map[ohe_col]]
                display_val += " ✅" if str(val).lower() in known else " ⚠️"
            cols[i%2].metric(label, display_val)
    else:
        st.info("Prediction based on visual features only (no AI tags)")

    # Visual analysis
    with st.expander("🎨 Visual Feature Analysis"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brightness",   f"{color_feats.get('brightness_mean',0):.2f}")
        c2.metric("Colorfulness", f"{color_feats.get('colorfulness',0):.2f}")
        c3.metric("Edge Density", f"{color_feats.get('edge_density',0):.2f}")
        c4.metric("Texture",      f"{color_feats.get('texture_complexity',0):.2f}")

    # OHE debug
    with st.expander("🔍 Debug — OHE Validation"):
        cat_map = dict(zip(ohe.feature_names_in_, ohe.categories_))
        rows = []
        for col, known_cats in cat_map.items():
            gemini_val = str(llm_tags.get(col, 'NOT PROVIDED')).lower()
            known      = [str(c).lower() for c in known_cats]
            matched    = gemini_val in known
            rows.append({
                "Feature"      : col,
                "AI Value"     : gemini_val,
                "Status"       : "✅ Known" if matched else "❌ Fallback used",
                "Known Values" : ", ".join(known),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Similar designs
    st.subheader("🔍 Similar Past Designs")
    if llm_tags:
        similar   = find_similar(llm_tags, train_df, n=5)
        show_cols = [c for c in ['product_code','product_name','qty_total',
                                  'rate_avg','work','pattern','occasion',
                                  'primary_color'] if c in similar.columns]
        if 'qty_total' in similar.columns:
            similar = similar.sort_values('qty_total', ascending=False)
        out        = similar[show_cols].reset_index(drop=True)
        out.index += 1
        st.dataframe(out, use_container_width=True)
        if 'qty_total' in similar.columns:
            st.caption(f"Similar avg qty: {similar['qty_total'].mean():.0f} | Prediction: {pred_qty}")
    else:
        st.info("Add API keys to .env for similar design matching")


# ── Tab 2: Analytics ──────────────────────────────────────────────
def render_analytics_tab():
    st.subheader("📊 Sales Pattern Analysis")
    try:
        _, _, _, config, train_df = load_model_artifacts()
    except Exception as e:
        st.error(f"❌ {e}")
        return

    if 'qty_total' not in train_df.columns:
        st.warning("No qty_total in training data.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", len(train_df))
    c2.metric("Avg Qty Sold",   f"{train_df['qty_total'].mean():.0f}")
    c3.metric("Top Seller",     f"{train_df['qty_total'].max():.0f} units")
    c4.metric("Median Qty",     f"{train_df['qty_total'].median():.0f}")
    st.markdown("---")

    cat_cols_avail = [c for c in ['work','pattern','occasion','length',
                                   'sleeves','primary_color']
                      if c in train_df.columns]
    if cat_cols_avail:
        selected = st.selectbox("Analyze by attribute:", cat_cols_avail)
        grp = (train_df.groupby(selected)['qty_total']
                       .agg(['mean','count'])
                       .rename(columns={'mean':'Avg Qty','count':'Products'})
                       .sort_values('Avg Qty', ascending=False))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        grp['Avg Qty'].plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
        axes[0].set_title(f'Avg Qty Sold by {selected.title()}')
        axes[0].set_ylabel('Avg Units Sold')
        axes[0].tick_params(axis='x', rotation=45)
        grp['Products'].plot(kind='bar', ax=axes[1], color='coral', edgecolor='white')
        axes[1].set_title(f'Number of Products by {selected.title()}')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(grp.reset_index(), use_container_width=True)

    st.markdown("---")
    st.subheader("Qty Distribution")
    fig, ax = plt.subplots(figsize=(10, 3))
    train_df['qty_total'].plot(kind='hist', bins=30, ax=ax,
                               color='gold', edgecolor='black')
    ax.set_xlabel('Total Qty Sold')
    ax.set_title('Distribution of Sales Across Products')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("🏆 All Products — Sorted by Sales")
    show_cols = [c for c in ['product_code','product_name','qty_total',
                              'rate_avg','work','pattern','occasion',
                              'primary_color'] if c in train_df.columns]
    top        = (train_df[show_cols]
                  .sort_values('qty_total', ascending=False)
                  .reset_index(drop=True))
    top.index += 1
    st.dataframe(top, use_container_width=True)


# ── Tab 3: How It Works ───────────────────────────────────────────
def render_how_it_works_tab():
    st.subheader("ℹ️ How the Prediction System Works")

    g_status  = f"✅ {len(GEMINI_KEYS)} keys"  if GEMINI_KEYS  else "❌ Not configured"
    or_status = f"✅ {len(OPENROUTER_KEYS)} keys" if OPENROUTER_KEYS else "❌ Not configured"

    st.markdown(f"""
    ### 🔑 API Key Configuration
    | Provider | Status | Keys File |
    |----------|--------|-----------|
    | Gemini   | {g_status} | `.env` → `GEMINI_KEY_1`, `GEMINI_KEY_2`... |
    | OpenRouter | {or_status} | `.env` → `OPENROUTER_KEY_1`, `OPENROUTER_KEY_2`... |

    Keys rotate automatically when quota is exhausted.
    Provider switches automatically if all keys of current provider fail.

    ### 🧠 Prediction Pipeline
    1. **ResNet50** → 2048 deep visual features → PCA → 50 features
    2. **Gemini/OpenRouter Vision** → length, sleeves, neck, work, pattern, color, occasion
    3. **Color/texture stats** → brightness, colorfulness, edge density etc.
    4. **XGBoost** trained on log1p(qty_total) → expm1 to get final prediction

    ### 📊 Model Performance
    - **CV R²:** 0.660 | **CV MAE:** 8.3 units
    - **Features:** 154 total (OHE + color + PCA ResNet)
    - **Training samples:** 145 matched products

    ### ⚠️ Limitations
    - Small dataset → ±30% prediction uncertainty
    - No seasonal signals (April 2026 data only)
    - Cold start for truly novel designs

    ### 🚀 Future Improvements
    - More training data → better accuracy
    - Festival/season calendar features
    - Fine-tuned fashion vision model
    """)


# ── Main ──────────────────────────────────────────────────────────
def main():
    render_sidebar()
    st.title("👗 Maalde — Demand Prediction Engine")
    st.markdown("Upload a kurti design image to predict how many units it will sell.")

    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Analytics", "ℹ️ How It Works"])
    with tab1: render_predict_tab()
    with tab2: render_analytics_tab()
    with tab3: render_how_it_works_tab()


if __name__ == "__main__":
    main()