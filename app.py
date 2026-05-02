import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFilter
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Demand Oracle",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_clean")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── Google Drive config ───────────────────────────────────────────────────────
DRIVE_FOLDER_ID = "1tvA0sBjuQejPzkVJ0JNj8eBK9mTo34jb"

# ── Auto-build product_code → file_id map from public Drive folder ────────────
GOOGLE_API_KEY = "AIzaSyA3C5gEzh38AJ4zNgjFqvH22FKCFMKmy4g"  

@st.cache_resource(show_spinner=False)
def build_image_map_from_drive(folder_id: str, api_key: str) -> dict:
    """
    Fetches all filenames + IDs from a public Drive folder.
    Files must be named like  KA001.jpeg / KA001.jpg / KA001.png
    Returns dict: {"KA001": "<file_id>", ...}
    """
    if not api_key:
        return {}

    import urllib.request, urllib.parse

    image_map = {}
    page_token = None

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name)",
            "pageSize": "1000",
            "key": api_key,
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
            name = f["name"]
            # Strip extension → product code
            code = os.path.splitext(name)[0]
            image_map[code] = f["id"]

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return image_map

PRODUCT_IMAGE_MAP: dict = {}   # populated after page load (see main())

# ─────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
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

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1400px; }

/* ── Animated grain overlay ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.6;
}

/* ── Hero header ── */
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

/* ── Upload zone ── */
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

/* ── Image preview card ── */
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

/* ── Predict button ── */
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

/* ── Metric cards ── */
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
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.6;
}
.metric-card.highlight::before { background: linear-gradient(90deg, var(--accent2), var(--accent)); opacity: 1; }
.metric-label {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 300;
    color: var(--text);
    line-height: 1;
}
.metric-card.highlight .metric-value { color: var(--accent); }
.metric-unit { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Section heading ── */
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
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Tag chips ── */
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
.tag-chip.warm {
    border-color: rgba(232,184,109,0.3);
    color: var(--accent);
    background: rgba(232,184,109,0.06);
}

/* ── Similar product cards ── */
.sim-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem; }
.sim-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
}
.sim-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 50px rgba(0,0,0,0.5);
    border-color: rgba(232,184,109,0.2);
}
.sim-card img {
    width: 100%;
    aspect-ratio: 3/4;
    object-fit: cover;
    display: block;
    background: var(--surface2);
}
.sim-card-img-placeholder {
    width: 100%;
    aspect-ratio: 3/4;
    background: linear-gradient(135deg, var(--surface2), var(--surface));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    opacity: 0.4;
}
.sim-card-body { padding: 0.8rem 0.9rem; }
.sim-card-name {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 0.3rem;
}
.sim-card-code { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.08em; }
.sim-card-meta { display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; }
.sim-card-price {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    color: var(--accent);
}
.sim-score {
    font-size: 0.65rem;
    background: rgba(232,184,109,0.1);
    border: 1px solid rgba(232,184,109,0.2);
    color: var(--accent);
    border-radius: 100px;
    padding: 0.15rem 0.5rem;
}
.sim-card-tags { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.5rem; }
.sim-mini-tag {
    font-size: 0.6rem;
    background: var(--surface2);
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    color: var(--muted);
}

/* ── Warning box ── */
.warn-box {
    background: rgba(232,184,109,0.05);
    border: 1px solid rgba(232,184,109,0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 1rem;
}
.warn-box-title { font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent2); margin-bottom: 0.4rem; }
.warn-item { font-size: 0.75rem; color: var(--muted); padding: 0.15rem 0; }

/* ── Divider ── */
.divider { height: 1px; background: var(--border); margin: 2rem 0; }

/* ── Spinner overrides ── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }
</style>
"""


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_drive_image_b64(file_id: str, api_key: str) -> str | None:
    """
    Download image from Drive server-side using API key and return as base64.
    This bypasses the browser auth issue with thumbnail URLs.
    Cached so each file is only downloaded once per session.
    """
    import urllib.request
    # Use Drive API to download the file content directly
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        import base64
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    except Exception:
        return None


def get_product_image_b64(product_code: str) -> str | None:
    """Return base64 image string for a product code, or None if not found."""
    code = str(product_code).strip()
    fid  = PRODUCT_IMAGE_MAP.get(code)
    if fid and GOOGLE_API_KEY:
        return fetch_drive_image_b64(fid, GOOGLE_API_KEY)
    return None


# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model       = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    ohe         = joblib.load(os.path.join(MODEL_DIR, "ohe.pkl"))
    scaler      = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config  = json.load(f)
    train_df    = pd.read_csv("final_dataset_training.csv")
    resnet_train = joblib.load(os.path.join(MODEL_DIR, "resnet_features.pkl"))
    return model, ohe, scaler, config, train_df, resnet_train


# ─────────────────────────────────────────
# LOAD RESNET
# ─────────────────────────────────────────
@st.cache_resource
def load_resnet():
    device      = torch.device("cpu")
    weights_path = os.path.join(DATA_DIR, "resnet50-11ad3fa6.pth")
    resnet      = models.resnet50(weights=None)
    resnet.load_state_dict(torch.load(weights_path, map_location=device))
    extractor   = nn.Sequential(*list(resnet.children())[:-1])
    extractor.eval()
    preprocess  = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return extractor, preprocess, device


# ─────────────────────────────────────────
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
    feats['brightness_mean_y'] = float(brightness.mean())

    gray = np.array(img_small.convert('L')).astype(float) / 255.0
    feats['contrast_y'] = float(gray.std())

    rg  = pixels[:,0] - pixels[:,1]
    yb  = 0.5*(pixels[:,0]+pixels[:,1]) - pixels[:,2]
    feats['colorfulness_y'] = float(np.sqrt(rg.std()**2 + yb.std()**2)) / 255.0

    gray_img = img.resize((128,128)).convert('L')
    edges    = gray_img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges).astype(float)

    feats['edge_density_y']       = float(edge_arr.mean() / 255.0)
    feats['texture_complexity_y'] = float(np.array(gray_img).std() / 255.0)
    feats['color_variety_y']      = min(feats['edge_density_y'] * 10, 1.0)
    return feats


def safe_ohe_transform(ohe, cat_row):
    warnings = []
    safe_row = {}
    for col, known in zip(ohe.feature_names_in_, ohe.categories_):
        val       = str(cat_row.get(col, "unknown")).lower()
        known_list = [str(k).lower() for k in known]
        if val in known_list:
            safe_row[col] = val
        else:
            safe_row[col] = known_list[0]
            warnings.append(f"<b>{col}</b>: '{val}' → '{known_list[0]}'")
    encoded = ohe.transform(pd.DataFrame([safe_row]))
    return encoded, warnings


def build_features(llm_tags, color_feats, resnet_feats, config, ohe, scaler):
    cat_cols  = config["cat_cols"]
    bool_cols = config["bool_cols"]
    num_cols  = config["num_cols"]

    X_cat, warnings = safe_ohe_transform(ohe, llm_tags)

    X_bool = np.array([
        1 if llm_tags.get(col, False) else 0
        for col in bool_cols
    ]).reshape(1, -1)

    num_vals = []
    for col in num_cols:
        if col == "rate_avg":
            num_vals.append(3.5)
        elif col == "days_on_market":
            num_vals.append(30)
        else:
            num_vals.append(float(color_feats.get(col, 0)))

    X_num    = scaler.transform(np.array(num_vals).reshape(1, -1))
    X_resnet = resnet_feats.reshape(1, -1)
    X        = np.hstack([X_cat, X_bool, X_num, X_resnet])
    return X, warnings


def similarity_prediction_resnet(query_feat, train_feats, train_df, top_k=5):
    sims       = cosine_similarity(query_feat.reshape(1, -1), train_feats)[0]
    top_idx    = np.argsort(sims)[-top_k:][::-1]
    similar_df = train_df.iloc[top_idx].copy()
    similar_df["similarity_score"] = sims[top_idx]
    sim_pred   = similar_df["qty_total"].mean()
    return sim_pred, similar_df


def extract_tags():
    return {
        "fabric":        "cotton",
        "length":        "knee_length",
        "sleeves":       "three_quarter",
        "neck":          "round",
        "work":          "printed",
        "pattern":       "floral",
        "primary_color": "blue",
        "occasion":      "casual",
        "price_tier":    "mid",
        "dupatta":       False,
        "pants_included": True,
    }


# ─────────────────────────────────────────
# SIMILAR PRODUCT CARD (Streamlit native)
# Renders each card using st.columns so base64
# images are never concatenated into one huge HTML blob.
# ─────────────────────────────────────────
def render_sim_cards_grid(similar_df: pd.DataFrame, cols_per_row: int = 5):
    """Render all similar product cards using Streamlit native columns."""
    rows = [similar_df.iloc[i:i+cols_per_row] for i in range(0, len(similar_df), cols_per_row)]

    for row_df in rows:
        cols = st.columns(len(row_df))
        for col, (_, row) in zip(cols, row_df.iterrows()):
            code  = str(row.get("product_code", "")).strip()
            name  = str(row.get("product_name", "Product"))
            price = row.get("rate_avg", None)
            color = str(row.get("primary_color", "")).title()
            work  = str(row.get("work", "")).title()
            patt  = str(row.get("pattern", "")).title()
            score = float(row.get("similarity_score", 0))

            with col:
                # ── Image ──────────────────────────────────────────────────
                img_b64 = get_product_image_b64(code)
                if img_b64:
                    st.markdown(
                        f'''<div style="border-radius:10px;overflow:hidden;aspect-ratio:3/4;background:#1a1a22;">
                            <img src="{img_b64}" style="width:100%;height:100%;object-fit:cover;display:block;" />
                        </div>''',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '''<div style="border-radius:10px;background:#1a1a22;aspect-ratio:3/4;
                            display:flex;align-items:center;justify-content:center;font-size:2.5rem;opacity:0.4;">
                            👗
                        </div>''',
                        unsafe_allow_html=True
                    )

                # ── Info ───────────────────────────────────────────────────
                st.markdown(
                    f'''<div style="padding:0.5rem 0.2rem;">
                        <div style="font-size:0.78rem;font-weight:500;color:#e8e4dc;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                            title="{name}">{name[:28]}{"…" if len(name)>28 else ""}</div>
                        <div style="font-size:0.65rem;color:#7a7880;letter-spacing:0.06em;margin-top:2px;">{code}</div>
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;">
                            <span style="font-family:'Cormorant Garamond',serif;font-size:1.1rem;color:#e8b86d;">
                                {"₹{:,.0f}".format(price) if price else "—"}
                            </span>
                            <span style="font-size:0.62rem;background:rgba(232,184,109,0.1);
                                border:1px solid rgba(232,184,109,0.25);color:#e8b86d;
                                border-radius:100px;padding:2px 7px;">{score:.2f}</span>
                        </div>
                        <div style="margin-top:5px;display:flex;flex-wrap:wrap;gap:3px;">
                            {" ".join([
                                f'<span style="font-size:0.6rem;background:#1a1a22;border-radius:4px;padding:2px 6px;color:#7a7880;">{t}</span>'
                                for t in [color, work, patt] if t and t.lower() not in ("", "nan", "None")
                            ])}
                        </div>
                    </div>''',
                    unsafe_allow_html=True
                )
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # ── Hero Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div>
        <p class="hero-sub">Fashion Intelligence · AI-Powered</p>
        <h1 class="hero-title">Demand <span>Oracle</span></h1>
        <div class="hero-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ──────────────────────────────────────────────────────────
    model, ohe, scaler, config, train_df, resnet_train = load_artifacts()
    extractor, preprocess, device = load_resnet()

    # ── Build Drive image map ─────────────────────────────────────────────────
    global PRODUCT_IMAGE_MAP
    if not PRODUCT_IMAGE_MAP:
        with st.spinner("🔗 Connecting to Drive image library…"):
            PRODUCT_IMAGE_MAP = build_image_map_from_drive(DRIVE_FOLDER_ID, GOOGLE_API_KEY)

    # Show Drive connection status in sidebar
    with st.sidebar:
        st.markdown("### 🗂 Drive Images")
        if PRODUCT_IMAGE_MAP:
            st.success(f"✅ {len(PRODUCT_IMAGE_MAP)} images linked")
            st.caption(f"Folder: `{DRIVE_FOLDER_ID[:20]}…`")
        elif GOOGLE_API_KEY:
            st.error("❌ Could not load images")
        else:
            st.warning("⚠️ Add API key to enable Drive images")
            st.caption("Set `GOOGLE_API_KEY` in app.py")
            st.markdown("""
**How to get a free key:**
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Enable **Google Drive API**
3. Create → **API Key**
4. Paste into `GOOGLE_API_KEY`
""")

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([1, 1.8], gap="large")

    with left:
        st.markdown('<div class="section-head">Upload Product Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","png","jpeg","webp"], label_visibility="collapsed")

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            # Render image in styled card
            import base64, io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div class="img-card">
                <img src="data:image/png;base64,{b64}" />
                <div class="img-card-label">📷 {uploaded.name}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("✦  Run Demand Prediction")

        else:
            predict_btn = False
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem; color: var(--muted); font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase;">
                Drop a product image to begin
            </div>
            """, unsafe_allow_html=True)

    with right:
        if uploaded and predict_btn:
            with st.spinner("Analysing image & computing demand…"):

                # ── Feature extraction ──────────────────────────────────────
                color_feats  = get_color_features(img)
                resnet_feats = extract_resnet_features(img, extractor, preprocess, device)
                llm_tags     = extract_tags()

                X, warnings = build_features(
                    llm_tags, color_feats, resnet_feats, config, ohe, scaler
                )

                model_pred = np.expm1(model.predict(X)[0])
                sim_pred, similar_df = similarity_prediction_resnet(
                    resnet_feats, resnet_train, train_df
                )
                final_pred = 0.5 * model_pred + 0.5 * sim_pred
                # Store in session state so similar products section can access it
                st.session_state["similar_df"] = similar_df

            # ── Metrics ────────────────────────────────────────────────────
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

            # ── Detected tags ───────────────────────────────────────────────
            st.markdown('<div class="section-head">Detected Attributes</div>', unsafe_allow_html=True)
            chips = "".join([
                f'<span class="tag-chip {"warm" if v and v is not False else ""}">'
                f'{k.replace("_"," ").title()}: {v}</span>'
                for k, v in llm_tags.items()
            ])
            st.markdown(f'<div style="margin-bottom:1rem">{chips}</div>', unsafe_allow_html=True)

            # ── Image analytics ─────────────────────────────────────────────
            cf = color_feats
            st.markdown('<div class="section-head">Visual Analytics</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Brightness", f"{cf['brightness_mean_y']:.2f}")
            c2.metric("Colorfulness", f"{cf['colorfulness_y']:.2f}")
            c3.metric("Edge Density", f"{cf['edge_density_y']:.2f}")

            # ── Warnings ────────────────────────────────────────────────────
            if warnings:
                warn_items = "".join([f'<div class="warn-item">→ {w}</div>' for w in warnings])
                st.markdown(f"""
                <div class="warn-box">
                    <div class="warn-box-title">⚠ Encoding Adjustments</div>
                    {warn_items}
                </div>
                """, unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div style="display:flex; align-items:center; justify-content:center; height:300px; color:var(--muted); font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase;">
                Results will appear here
            </div>
            """, unsafe_allow_html=True)

    # ── Similar Products (full width) ─────────────────────────────────────────
    if uploaded and predict_btn and st.session_state.get("similar_df") is not None:
        similar_df = st.session_state["similar_df"]
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Similar Products · Visual Neighbours</div>', unsafe_allow_html=True)

        # Render cards using native Streamlit columns (avoids base64 blob truncation)
        render_sim_cards_grid(similar_df, cols_per_row=5)

        # Expandable raw data
        with st.expander("View raw data table"):
            disp_cols = [c for c in [
                "product_code","product_name","rate_avg","qty_total",
                "primary_color","work","pattern","similarity_score"
            ] if c in similar_df.columns]
            st.dataframe(similar_df[disp_cols], use_container_width=True)


if __name__ == "__main__":
    main()