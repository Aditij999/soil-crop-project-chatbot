"""
Smart Agriculture Decision Support System
Single-page dashboard: input → results in one scroll.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys, warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────
st.set_page_config(
    page_title="AgriSmart DSS",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Noto+Sans+Devanagari:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');

/* Reduce top-heavy spacing — increased to clear Streamlit toolbar */
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 1rem !important;
}

:root {
    --bg:       #F7F4EF;
    --surface:  #FFFFFF;
    --border:   #E8E2D9;
    --dark:     #2C2416;
    --mid:      #6B5E4C;
    --soft:     #7B6D5A;
    --accent:   #7C6D5A;
    --linen:    #F0EBE2;
    --green:    #5F8C6A;
    --red:      #B85C5C;
    --yellow:   #C49A3C;
}

/* ───────────────────────────────────────── */
/* GLOBAL APP */
/* ───────────────────────────────────────── */

.stApp {
    background: var(--bg);
    color: var(--dark);
}

html,
body,
[class*="css"],
label,
p,
span,
div {
    font-family: 'DM Sans', 'Noto Sans Devanagari', sans-serif;
    color: var(--dark) !important;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--dark) !important;
}

[data-testid="collapsedControl"] {
    display: none;
}

/* ───────────────────────────────────────── */
/* METRICS */
/* ───────────────────────────────────────── */

[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(44,36,22,0.06);
}

[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem !important;
    color: var(--dark) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--soft) !important;
    font-size: 0.78rem;
}

/* ───────────────────────────────────────── */
/* BUTTONS */
/* ───────────────────────────────────────── */

.stButton > button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 32px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #6A5D4B !important;
    transform: translateY(-1px);
}

/* ───────────────────────────────────────── */
/* SLIDERS */
/* ───────────────────────────────────────── */

.stSlider label {
    color: var(--dark) !important;
    font-weight: 500 !important;
}

.stSlider span {
    color: var(--dark) !important;
}

.stSlider [data-baseweb="slider"] {
    accent-color: var(--accent);
}

/* ───────────────────────────────────────── */
/* SELECTBOX */
/* ───────────────────────────────────────── */

.stSelectbox label {
    color: var(--dark) !important;
    font-weight: 500 !important;
}

[data-baseweb="select"] * {
    color: var(--dark) !important;
}

[data-baseweb="select"] {
    border-radius: 8px !important;
    background-color: #FFFFFF !important;
}

[data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-baseweb="select"] div,
[data-baseweb="select"] span,
[data-baseweb="select"] input {
    background-color: #FFFFFF !important;
    color: var(--dark) !important;
}

[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="popover"] * {
    background-color: #FFFFFF !important;
    color: var(--dark) !important;
}

[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="menu"] li:hover {
    background-color: var(--linen) !important;
}

/* ───────────────────────────────────────── */
/* INPUTS */
/* ───────────────────────────────────────── */

input,
textarea {
    color: var(--dark) !important;
}

[data-testid="stNumberInput"] input {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    color: var(--dark) !important;
}

/* ───────────────────────────────────────── */
/* CHAT INPUT */
/* ───────────────────────────────────────── */

[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: var(--dark) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--soft) !important;
    opacity: 1 !important;
}

/* Chat message bubbles */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}


/* ───────────────────────────────────────── */
/* DIVIDERS */
/* ───────────────────────────────────────── */

hr {
    border-color: var(--border) !important;
}

/* ───────────────────────────────────────── */
/* STATUS BADGES */
/* ───────────────────────────────────────── */

.ok {
    background: #EEF4F0;
    color: #3D7048 !important;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
}

.low {
    background: #FDF0F0;
    color: #9C3F3F !important;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
}

.high {
    background: #FDF6E3;
    color: #8C6A1A !important;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
}

/* ───────────────────────────────────────── */
/* SECTION HEADERS */
/* ───────────────────────────────────────── */

.section-header {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--dark) !important;
    border-left: 3px solid var(--accent);
    padding-left: 12px;
    margin: 28px 0 16px;
}

/* ───────────────────────────────────────── */
/* DATAFRAME STYLING */
/* ───────────────────────────────────────── */

[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ───────────────────────────────────────── */
/* PLOTLY */
/* ───────────────────────────────────────── */

.js-plotly-plot .plotly .modebar {
    background: rgba(255,255,255,0.7) !important;
    border-radius: 8px;
}

/* ───────────────────────────────────────── */
/* SCROLLBAR */
/* ───────────────────────────────────────── */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #EFE8DD;
}

::-webkit-scrollbar-thumb {
    background: #B8AA95;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #9E8E7A;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly defaults ──────────────────────────
EARTHY = ["#7C6D5A","#A89880","#5F8C6A","#C49A3C","#B85C5C","#9BADB7","#C9B99A"]
DARK_FONT = dict(family="DM Sans", color="#2C2416", size=11)
TICK_FONT = dict(color="#2C2416", size=11)
TITLE_FONT = dict(color="#2C2416", size=12)

PL = dict(
    font=dict(family="DM Sans", color="#2C2416"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=16, r=16, t=36, b=16),
    colorway=EARTHY,
)

# ── Module imports (graceful) ────────────────
ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, ROOT_DIR)

CHATBOT_DIR = os.path.join(ROOT_DIR, "chatbot")
if CHATBOT_DIR not in sys.path:
    sys.path.insert(0, CHATBOT_DIR)

@st.cache_resource(show_spinner=False)
def load_modules():
    m = {}
    for name, path, fn in [
        ("compute_nutrient_gap", "modules.soil_adjustment",          "compute_nutrient_gap"),
        ("recommend_fertilizer", "modules.fertilizer_recommendation", "recommend_fertilizer"),
        ("rec_label",            "modules.fertilizer_recommendation", "rec_label"),
        ("get_supported_crops",  "modules.soil_adjustment",          "get_supported_crops"),
        ("predict_yield",        "modules.yield_model",               "predict_yield"),
    ]:
        try:
            mod = __import__(path, fromlist=[fn])
            m[name] = getattr(mod, fn)
        except Exception:
            m[name] = None
    try:
        import pickle
        with open(os.path.join("models", "crop_model.pkl"), "rb") as f:
            m["crop_model"] = pickle.load(f)
    except Exception:
        m["crop_model"] = None
    return m

@st.cache_resource(show_spinner=False)
def load_classifier():
    from classifier import train_classifier
    return train_classifier()

mods = load_modules()

try:
    from classifier import predict_intent
    from response_engine import generate_response
    from context import ConversationContext
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False

if "report" not in st.session_state:
    st.session_state.report = None
if "results" not in st.session_state:
    st.session_state.results = None
if "chat_context" not in st.session_state:
    st.session_state.chat_context = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_lang" not in st.session_state:
    st.session_state.chat_lang = "auto"

# ── Demo/fallback helpers ────────────────────
ALL_CROPS = ["Jowar", "Groundnut", "Cotton", "Sugarcane", "Wheat"]

YIELD_BASELINE = {
    "Jowar":     2800,
    "Groundnut": 2200,
    "Cotton":    2400,
    "Sugarcane": 70000,
    "Wheat":     3100,
}

def demo_crops(N, P, K, T, R, pH):
    seed = int(N * 7 + P * 13 + K * 17 + T * 3 + R * 0.1 + pH * 100) % (2**31)
    rng  = np.random.default_rng(seed)
    raw  = rng.dirichlet(np.ones(len(ALL_CROPS)) * 2)
    top5 = sorted(zip(ALL_CROPS, raw), key=lambda x: -x[1])[:5]
    tot  = sum(v for _, v in top5)
    return [(c, round(v / tot * 100, 1)) for c, v in top5]

def demo_soil(N, P, K, pH):
    OPT  = {"Nitrogen":(40,100), "Phosphorus":(25,80), "Potassium":(20,70), "pH":(5.5,7.5)}
    vals = {"Nitrogen":N, "Phosphorus":P, "Potassium":K, "pH":pH}
    rows = []
    for nut, (lo, hi) in OPT.items():
        v = vals[nut]
        if v < lo:   status, gap = "Low",     round(lo - v, 2)
        elif v > hi: status, gap = "High",    round(v - hi, 2)
        else:        status, gap = "Optimal", 0.0
        rows.append({"Nutrient":nut, "Value":round(v,2),
                     "Range":f"{lo}–{hi}", "Gap":gap, "Status":status})
    return pd.DataFrame(rows)

def demo_fert(crop, N, P, K):
    recs = []
    if N < 40:
        d = max(25, round((40-N)/0.46/25)*25)
        recs.append({"Fertilizer":"Urea (46-0-0)",  "NPK":"46-0-0",  "Dosage (kg/ha)":d, "Dose Range":f"{max(0,d-12)}–{d+12} kg/ha", "Reason":"Low nitrogen"})
    if P < 25:
        d = max(25, round((25-P)/0.46/25)*25)
        recs.append({"Fertilizer":"DAP (18-46-0)",  "NPK":"18-46-0", "Dosage (kg/ha)":d, "Dose Range":f"{max(0,d-12)}–{d+12} kg/ha", "Reason":"Low phosphorus"})
    if K < 20:
        d = max(25, round((20-K)/0.60/25)*25)
        recs.append({"Fertilizer":"MOP (0-0-60)",   "NPK":"0-0-60",  "Dosage (kg/ha)":d, "Dose Range":f"{max(0,d-12)}–{d+12} kg/ha", "Reason":"Low potassium"})
    if not recs:
        recs.append({"Fertilizer":"NPK 17-17-17", "NPK":"17-17-17", "Dosage (kg/ha)":50, "Dose Range":"38–62 kg/ha", "Reason":"Balanced maintenance"})
    return pd.DataFrame(recs)

def demo_yield(crop, N, P, K, T, R, pH):
    seed = int(abs(hash(crop)) % 1000 + N * 7 + P * 13 + K * 17 + T * 3 + R * 0.1 + pH * 100) % (2**31)
    rng  = np.random.default_rng(seed)
    base = YIELD_BASELINE.get(crop, 3000)
    return round(base * rng.uniform(0.85, 1.15), 1)


def _supported_crops():
    if mods["get_supported_crops"]:
        try:
            return mods["get_supported_crops"]()
        except Exception:
            pass
    return ALL_CROPS


def run_analysis(N, P, K, T, R, pH, crop_choice):
    """Run the full pipeline and return (report_dict, display_results)."""
    soil_input = {
        "Nitrogen": N, "Phosphorus": P, "Potassium": K,
        "Temperature": T, "Rainfall": R, "pH": pH,
    }
    soil_dict = {"Nitrogen": N, "Phosphorus": P, "Potassium": K, "pH": pH}
    gap_result = {}

    if mods["crop_model"] is not None:
        try:
            inp   = np.array([[N, P, K, T, 0, pH, R]])
            probs = mods["crop_model"].predict_proba(inp)[0]
            top5_idx = np.argsort(probs)[::-1][:5]
            top5  = [(mods["crop_model"].classes_[i], round(probs[i] * 100, 1)) for i in top5_idx]
        except Exception:
            top5 = demo_crops(N, P, K, T, R, pH)
    else:
        top5 = demo_crops(N, P, K, T, R, pH)

    if mods["compute_nutrient_gap"]:
        try:
            gap_result = mods["compute_nutrient_gap"](crop_choice, soil_dict)
            rows = []
            for nutrient, info in gap_result.items():
                rows.append({
                    "Nutrient": nutrient,
                    "Value":    round(info["current"], 2),
                    "Range":    f"~{info['ideal']}",
                    "Gap":      abs(info["gap"]),
                    "Status":   info["status"],
                })
            df_soil = pd.DataFrame(rows)
        except Exception:
            gap_result = {}
            df_soil = demo_soil(N, P, K, pH)
    else:
        df_soil = demo_soil(N, P, K, pH)

    fert_result = {"action_needed": False, "generic_recs": [], "crop_primary": None, "gap_analysis": {}}
    if mods["recommend_fertilizer"]:
        try:
            fert_result = mods["recommend_fertilizer"](crop_choice, soil_dict)
            fert_rows = []
            rec_label_fn = mods["rec_label"] or (lambda rec: rec.get("fertilizer") or rec.get("action", ""))
            for rec in fert_result.get("generic_recs", []):
                nutrient  = rec["nutrient"]
                condition = rec["condition"]
                gap_info  = fert_result["gap_analysis"].get(nutrient, {})
                gap_amt   = abs(gap_info.get("gap", 0))
                display_name = rec_label_fn(rec)

                if condition == "Low":
                    nutrient_pct = {"Nitrogen": 0.46, "Phosphorus": 0.46, "Potassium": 0.60}.get(nutrient, 0.46)
                    raw_dose = gap_amt / nutrient_pct if gap_amt > 0 else 50
                    rounded  = max(25, round(raw_dose / 25) * 25)
                    dosage   = rounded
                    dose_range = f"{max(0, rounded - 12)}–{rounded + 12} kg/ha"
                else:
                    dosage     = 0
                    dose_range = "—"

                fert_rows.append({
                    "Fertilizer":     display_name,
                    "NPK":            rec["npk_ratio"],
                    "Dosage (kg/ha)": dosage,
                    "Dose Range":     dose_range,
                    "Reason":         rec["reason"],
                })
            if not fert_rows:
                cp = fert_result.get("crop_primary")
                if cp:
                    fert_rows.append({
                        "Fertilizer":    cp["fertilizer"],
                        "NPK":           "—",
                        "Dosage (kg/ha)": 50,
                        "Dose Range":    "38–62 kg/ha",
                        "Reason":        cp["note"],
                    })
                else:
                    fert_rows.append({
                        "Fertilizer":    "NPK 17-17-17",
                        "NPK":           "17-17-17",
                        "Dosage (kg/ha)": 50,
                        "Dose Range":    "38–62 kg/ha",
                        "Reason":        "Soil balanced — maintenance dose",
                    })
            df_fert = pd.DataFrame(fert_rows)
        except Exception:
            df_fert = demo_fert(crop_choice, N, P, K)
    else:
        df_fert = demo_fert(crop_choice, N, P, K)

    yield_result = {"predicted_yield_kg_ha": demo_yield(crop_choice, N, P, K, T, R, pH), "interpretation": ""}
    if mods["predict_yield"]:
        try:
            yield_result = mods["predict_yield"](soil_input, crop=crop_choice)
        except Exception:
            predicted = demo_yield(crop_choice, N, P, K, T, R, pH)
            yield_result = {
                "predicted_yield_kg_ha": predicted,
                "interpretation": "🟡 Moderate yield — improvement possible.",
            }
    predicted = yield_result["predicted_yield_kg_ha"]

    top_crop, top_prob = top5[0]
    avg_yield  = YIELD_BASELINE.get(crop_choice, 3000)
    diff_pct   = round((predicted - avg_yield) / avg_yield * 100, 1)
    opt_count  = int((df_soil["Status"] == "Optimal").sum())
    need_count = len(df_soil) - opt_count
    yield_lo_tha = round(predicted * 0.88 / 1000, 2)
    yield_hi_tha = round(predicted * 1.12 / 1000, 2)

    if need_count == 0:
        soil_kpi = "All Optimal"
    elif need_count == 1:
        soil_kpi = "1 imbalance detected"
    else:
        soil_kpi = f"{need_count} imbalances detected"

    report = {
        "soil_input": soil_input,
        "chosen_crop": crop_choice,
        "top_crop": top_crop,
        "supported_crops": _supported_crops(),
        "top_5_crops": [
            {"crop": c, "probability": round(p / 100, 4)}
            for c, p in top5
        ],
        "soil_adjustment": gap_result,
        "fertilizer": fert_result,
        "yield_prediction": yield_result,
    }

    results = {
        "N": N, "P": P, "K": K, "T": T, "R": R, "pH": pH,
        "crop_choice": crop_choice,
        "top5": top5,
        "top_crop": top_crop,
        "top_prob": top_prob,
        "df_soil": df_soil,
        "df_fert": df_fert,
        "predicted": predicted,
        "avg_yield": avg_yield,
        "diff_pct": diff_pct,
        "soil_kpi": soil_kpi,
        "need_count": need_count,
        "yield_lo_tha": yield_lo_tha,
        "yield_hi_tha": yield_hi_tha,
    }
    return report, results

# ═══════════════════════════════════════════════
#  HEADER  — FIX: darker subtitle, more top padding
# ═══════════════════════════════════════════════
st.markdown("""
<div style='padding:16px 0 4px'>
    <span style='font-size:0.72rem; color:#5A4E3C; letter-spacing:2px;
                 text-transform:uppercase; font-weight:600'>
        AI-Based Agricultural Advisory · 5 Regional Crops
    </span>
    <h1 style='margin:4px 0 6px; font-size:1.9rem; color:#2C2416'>AgriSmart DSS 🌾</h1>
    <p style='color:#4A3E2E; font-size:0.85rem; margin:0; font-weight:500'>
        Regional advisory for Jowar, Groundnut, Cotton, Sugarcane &amp; Wheat —
        enter your field parameters below for crop recommendation, soil analysis,
        fertilizer guide, and yield prediction.
    </p>
</div>
<hr style='margin:12px 0 16px'>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  INPUT FORM
# ═══════════════════════════════════════════════
st.markdown('<div class="section-header">Field Parameters</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    N  = st.slider("Nitrogen (N) kg/ha",   0,    150,   58)
    P  = st.slider("Phosphorus (P) kg/ha", 0,    150,   43)
with c2:
    K  = st.slider("Potassium (K) kg/ha",  0,    150,   37)
    pH = st.slider("Soil pH",              4.0,  9.0,   6.4, 0.1)
with c3:
    T  = st.slider("Temperature °C",       5.0,  50.0,  26.5, 0.5)
    R  = st.slider("Rainfall mm",          0.0,  2000.0, 820.0, 10.0)
with c4:
    crop_choice = st.selectbox("Target Crop (fertilizer & yield)", ALL_CROPS, index=0)
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("▶  Run Analysis", use_container_width=True)

st.markdown("<hr style='margin:24px 0'>", unsafe_allow_html=True)

if run:
    with st.spinner("🌱 Analysing soil conditions and running predictions..."):
        import time
        time.sleep(0.8)
        report, results = run_analysis(N, P, K, T, R, pH, crop_choice)
        st.session_state.report = report
        st.session_state.results = results
        st.session_state.chat_messages = []
        if CHATBOT_AVAILABLE:
            st.session_state.chat_context = ConversationContext(default_crop=results["top_crop"])

if not st.session_state.get("results"):
    st.markdown("""
    <div style='background:#F0EBE2; border:1px dashed #C9B99A; border-radius:12px;
                padding:40px; text-align:center; color:#6B5E4C'>
        <div style='font-size:2rem; margin-bottom:8px'>🌿</div>
        Adjust the sliders above and click <b>Run Analysis</b> to see all results.
    </div>
    """, unsafe_allow_html=True)

if st.session_state.get("results"):
    res = st.session_state.results
    N = res["N"]
    P = res["P"]
    K = res["K"]
    T = res["T"]
    R = res["R"]
    pH = res["pH"]
    crop_choice = res["crop_choice"]
    top5 = res["top5"]
    top_crop = res["top_crop"]
    top_prob = res["top_prob"]
    df_soil = res["df_soil"]
    df_fert = res["df_fert"]
    predicted = res["predicted"]
    avg_yield = res["avg_yield"]
    diff_pct = res["diff_pct"]
    soil_kpi = res["soil_kpi"]
    need_count = res["need_count"]
    yield_lo_tha = res["yield_lo_tha"]
    yield_hi_tha = res["yield_hi_tha"]
    top_conf_label = (
        "High Match" if top_prob >= 35
        else "Moderate Match" if top_prob >= 22
        else "Low Match"
    )

    # ═══════════════════════════════════════════════
    #  KPI STRIP
    # ═══════════════════════════════════════════════
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🌱 Top Crop",        top_crop,                   top_conf_label)
    k2.metric("🪱 Soil Health",     soil_kpi)
    k3.metric("🧪 Fertilizers",     f"{len(df_fert[df_fert['Dosage (kg/ha)']>0])} to apply")
    k4.metric("📈 Est. Yield",      f"{yield_lo_tha}–{yield_hi_tha} t/ha", f"{diff_pct:+.1f}% vs avg")

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    #  ROW 1 — Crop Recommendation  |  Soil Analysis
    # ═══════════════════════════════════════════════
    col_crop, col_soil = st.columns(2, gap="large")
    
    # ── Crop Recommendation ──────────────────────
    with col_crop:
        st.markdown('<div class="section-header">🌱 Crop Recommendation</div>', unsafe_allow_html=True)
    
        def conf_label(p):
            if p >= 35:   return ("High Match",     "#5F8C6A")
            elif p >= 22: return ("Moderate Match", "#C49A3C")
            else:         return ("Low Match",       "#B85C5C")
    
        top_label, top_label_color = conf_label(top_prob)
    
        st.markdown(f"""
        <div style='background:#F0EBE2; border:1px solid #C9B99A; border-radius:12px;
                    padding:18px 22px; margin-bottom:16px; display:flex; align-items:center; gap:16px'>
            <div style='font-size:2.4rem'>🌿</div>
            <div>
                <div style='font-size:0.72rem; color:#6B5E4C; text-transform:uppercase;
                            letter-spacing:1.5px; font-weight:600'>Best Match</div>
                <div style='font-family:"Playfair Display",serif; font-size:1.6rem;
                            font-weight:600; color:#2C2416'>{top_crop}</div>
                <div style='font-size:0.82rem; color:{top_label_color}; font-weight:600'>{top_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        crops_  = [c for c, _ in top5]
        probs_  = [p for _, p in top5]
        # FIX: shorter labels to prevent overflow
        labels_ = [f"{p}%" for p in probs_]
        colors_ = ["#7C6D5A" if i == 0 else "#C9B99A" for i in range(len(top5))]
    
        fig_cr = go.Figure(go.Bar(
            y=crops_,
            x=probs_,
            orientation="h",
            marker_color=colors_,
            # FIX: text inside bars, white text for contrast
            text=labels_,
            textposition="inside",
            textfont=dict(color="#FFFFFF", size=12, family="DM Sans"),
            insidetextanchor="middle",
        ))
        fig_cr.update_layout(
            **PL,
            height=220,
            # FIX: fixed axis range so bars never clip text
            xaxis=dict(
                title="Match score (%)",
                range=[0, 100],
                tickfont=TICK_FONT,
                title_font=TITLE_FONT,
                gridcolor="#E8E2D9",
            ),
            yaxis=dict(
                autorange="reversed",
                # FIX: force dark color for crop name labels
                tickfont=dict(color="#2C2416", size=12, family="DM Sans"),
                title_font=TITLE_FONT,
            ),
        )
        st.plotly_chart(fig_cr, use_container_width=True)
    
        rc1, rc2, rc3 = st.columns(3)
        for col, (icon, label, val) in zip(
            [rc1, rc2, rc3],
            [("💧","Rainfall",f"{R:.0f} mm"), ("⚗️","pH",f"{pH:.1f}"), ("🌡️","Temp.",f"{T:.1f} °C")]
        ):
            col.markdown(f"""
            <div style='background:#fff; border:1px solid #E8E2D9; border-radius:10px;
                        padding:12px; text-align:center'>
                <div style='font-size:1.2rem'>{icon}</div>
                <div style='font-size:0.72rem; color:#6B5E4C; font-weight:600'>{label}</div>
                <div style='font-weight:600; font-size:1rem; color:#2C2416'>{val}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ── Soil Analysis ────────────────────────────
    with col_soil:
        st.markdown('<div class="section-header">🪱 Soil Analysis</div>', unsafe_allow_html=True)
    
        IDEAL_RANGES = {
            "Jowar":     {"Nitrogen":(44,65),   "Phosphorus":(32,42), "Potassium":(31,42), "pH":(6.5,7.1)},
            "Groundnut": {"Nitrogen":(19,40),   "Phosphorus":(36,47), "Potassium":(22,33), "pH":(6.2,6.8)},
            "Cotton":    {"Nitrogen":(107,128), "Phosphorus":(51,62), "Potassium":(55,66), "pH":(6.7,7.3)},
            "Sugarcane": {"Nitrogen":(121,142), "Phosphorus":(65,76), "Potassium":(109,119),"pH":(6.6,7.2)},
            "Wheat":     {"Nitrogen":(81,102),  "Phosphorus":(48,58), "Potassium":(44,55), "pH":(6.4,7.0)},
        }
        ideals = IDEAL_RANGES.get(crop_choice, {
            "Nitrogen":(40,100), "Phosphorus":(25,80), "Potassium":(20,70), "pH":(5.5,7.5)
        })
    
        fig_g = make_subplots(rows=2, cols=2, specs=[[{"type":"indicator"}]*2]*2)
        gauge_params = [
            ("Nitrogen",   N,  0,   150),
            ("Phosphorus", P,  0,   150),
            ("Potassium",  K,  0,   150),
            ("pH",         pH, 4.0, 9.0),
        ]
        for (name, val, lo, hi), (r, c) in zip(gauge_params, [(1,1),(1,2),(2,1),(2,2)]):
            ideal_lo, ideal_hi = ideals[name]
            fig_g.add_trace(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": name, "font": {"size": 11, "color": "#2C2416", "family": "DM Sans"}},
                number={"font": {"color": "#2C2416", "size": 16, "family": "DM Sans"}},
                gauge={
                    "axis": {
                        "range": [lo, hi],
                        "tickfont": {"size": 8, "color": "#2C2416"},
                    },
                    "bar":  {"color": "#7C6D5A", "thickness": 0.25},
                    "bgcolor": "#F7F4EF",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [lo,       ideal_lo],  "color": "#FDEAEA"},
                        {"range": [ideal_lo, ideal_hi],  "color": "#D6EDD9"},
                        {"range": [ideal_hi, hi],         "color": "#FDF6E3"},
                    ],
                    "threshold": {
                        "line": {"color": "#5F8C6A", "width": 2},
                        "thickness": 0.8,
                        "value": (ideal_lo + ideal_hi) / 2,
                    },
                },
            ), row=r, col=c)
        fig_g.update_layout(**PL, height=280)
        st.plotly_chart(fig_g, use_container_width=True)
    
        st.markdown("""
        <div style='display:flex; gap:16px; font-size:0.75rem; color:#4A3E2E;
                    font-weight:500; margin-top:-8px; margin-bottom:4px'>
            <span><span style='display:inline-block;width:10px;height:10px;border-radius:2px;
                  background:#FDEAEA;border:1px solid #B85C5C;margin-right:4px'></span>Below optimal</span>
            <span><span style='display:inline-block;width:10px;height:10px;border-radius:2px;
                  background:#D6EDD9;border:1px solid #5F8C6A;margin-right:4px'></span>Optimal range</span>
            <span><span style='display:inline-block;width:10px;height:10px;border-radius:2px;
                  background:#FDF6E3;border:1px solid #C49A3C;margin-right:4px'></span>Above optimal</span>
        </div>
        """, unsafe_allow_html=True)
    
        badge_map = {"Optimal":"ok", "Low":"low", "High":"high"}
        msg_map   = lambda row: {
            "Low":     f"⬆ +{row['Gap']} needed",
            "High":    f"⬇ {row['Gap']} above optimal",
            "Optimal": "Within optimal range",
        }[row["Status"]]
    
        for _, row in df_soil.iterrows():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                        background:#fff; border:1px solid #E8E2D9; border-radius:8px;
                        padding:10px 14px; margin-bottom:6px'>
                <div>
                    <span style='font-weight:600; color:#2C2416'>{row["Nutrient"]}</span>
                    <span style='font-size:0.75rem; color:#6B5E4C; margin-left:8px;
                                 font-weight:500'>{msg_map(row)}</span>
                </div>
                <div style='display:flex; align-items:center; gap:10px'>
                    <span style='font-weight:600; color:#2C2416'>{row["Value"]}</span>
                    <span class='{badge_map[row["Status"]]}'>{row["Status"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin:28px 0'>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════
    #  ROW 2 — Fertilizer  |  Yield Prediction
    # ═══════════════════════════════════════════════
    col_fert, col_yield = st.columns(2, gap="large")
    
    # ── Fertilizer ───────────────────────────────
    with col_fert:
        st.markdown(f'<div class="section-header">🧪 Fertilizer — {crop_choice}</div>', unsafe_allow_html=True)
    
        PRIMARY = {
            "Cotton":    ("DAP", "Preferred basal for Cotton in Black Soil"),
            "Groundnut": ("Chelated Micronutrient", "Micronutrient support alongside NPK"),
            "Jowar":     ("10:26:26 NPK", "High P+K supports grain and stem development"),
            "Sugarcane": ("MOP", "Sugarcane is K-hungry — standard choice"),
            "Wheat":     ("19:19:19 NPK", "Balanced NPK suits Wheat's equal N/P/K demand"),
        }
        if crop_choice in PRIMARY:
            pf, pnote = PRIMARY[crop_choice]
            st.markdown(f"""
            <div style='background:#EEF4F0; border-left:3px solid #5F8C6A; border-radius:6px;
                        padding:8px 12px; margin-bottom:10px; font-size:0.82rem; color:#2C2416'>
                <b>Primary fertilizer:</b> {pf} &nbsp;·&nbsp;
                <span style='color:#4A3E2E; font-weight:500'>{pnote}</span>
            </div>
            """, unsafe_allow_html=True)
    
        for _, row in df_fert.iterrows():
            is_skip      = row["Dosage (kg/ha)"] == 0
            dose_display = "—" if is_skip else row.get("Dose Range", f"{row['Dosage (kg/ha)']} kg/ha")
            dosage_color = "#B85C5C" if is_skip else "#2C2416"
            st.markdown(f"""
            <div style='background:#fff; border:1px solid #E8E2D9; border-radius:8px;
                        padding:10px 14px; margin-bottom:7px'>
                <div style='display:flex; justify-content:space-between; align-items:center'>
                    <div>
                        <div style='font-weight:600; color:#2C2416; font-size:0.9rem'>{row["Fertilizer"]}</div>
                        <div style='font-size:0.73rem; color:#6B5E4C; margin-top:2px;
                                    font-weight:500'>{row["Reason"]}</div>
                    </div>
                    <div style='text-align:right'>
                        <div style='font-family:"Playfair Display",serif; font-size:1.05rem;
                                    font-weight:600; color:{dosage_color}'>{dose_display}</div>
                        <div style='font-size:0.72rem; color:#6B5E4C; font-weight:500'>NPK {row["NPK"]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
        df_fert_chart = df_fert[df_fert["Dosage (kg/ha)"] > 0]
        all_skip = not df_fert.empty and df_fert_chart.empty
    
        if df_fert.empty:
            st.markdown("""
            <div style='background:#EEF4F0; border:1px solid #8FBC9B; border-radius:10px;
                        padding:20px; text-align:center; color:#3D7048; font-size:0.85rem;
                        font-weight:500'>
                ✅ Soil is perfectly balanced — no fertilizer application needed this season.
            </div>
            """, unsafe_allow_html=True)
        elif all_skip:
            st.markdown("""
            <div style='background:#FDF6E3; border:1px solid #C49A3C; border-radius:10px;
                        padding:20px; text-align:center; color:#6B4A0A; font-size:0.85rem;
                        font-weight:500'>
                ⚠️ Some nutrients exceed optimal levels — skip those fertilizers this season.
                No additional application required.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Shorten names: "Urea (46-0-0)" → "Urea", keep "Hydrated Lime — Ca(OH)₂" short too
            df_fert_chart = df_fert_chart.copy()
            df_fert_chart["Fert_Short"] = (
                df_fert_chart["Fertilizer"]
                .str.split("(").str[0]          # drop parenthetical NPK codes
                .str.split("—").str[0]          # shorten "Hydrated Lime — Ca(OH)₂"
                .str.strip()
            )
    
            # Build one trace per fertilizer so each gets its own legend entry + color
            npk_vals    = df_fert_chart["NPK"].tolist()
            fert_shorts = df_fert_chart["Fert_Short"].tolist()
            dosages     = df_fert_chart["Dosage (kg/ha)"].tolist()
    
            fig_f = go.Figure()
            for i, (fname, npk, dose) in enumerate(zip(fert_shorts, npk_vals, dosages)):
                fig_f.add_trace(go.Bar(
                    x=[fname],
                    y=[dose],
                    name=f"NPK {npk}",
                    marker_color=EARTHY[i % len(EARTHY)],
                    text=[str(dose)],
                    textposition="outside",
                    textfont=dict(color="#2C2416", size=11, family="DM Sans"),
                ))
    
            PL_fert = {k: v for k, v in PL.items() if k != "margin"}
            fig_f.update_layout(
                **PL_fert,
                height=270,
                barmode="group",
                showlegend=True,
                margin=dict(l=16, r=16, t=36, b=90),
                legend=dict(
                    orientation="h",
                    y=-0.42,                                    # below x-axis labels
                    x=0,
                    font=dict(size=10, color="#2C2416", family="DM Sans"),
                    # FIX: legend title color explicitly set
                    title=dict(
                        text="NPK",
                        font=dict(size=10, color="#2C2416", family="DM Sans"),
                    ),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                ),
            )
            fig_f.update_xaxes(
                tickfont=dict(color="#2C2416", size=11, family="DM Sans"),
                title_text="",          # remove "Fertilizer" label — redundant with card headers above
                tickangle=0,            # keep upright; names are short now
            )
            fig_f.update_yaxes(
                title_text="Dosage (kg/ha)",
                tickfont=dict(color="#2C2416", size=11, family="DM Sans"),
                title_font=dict(color="#2C2416", size=12, family="DM Sans"),
                gridcolor="#E8E2D9",
                # give headroom above tallest bar so "outside" text isn't clipped
                range=[0, max(dosages) * 1.35],
            )
            st.plotly_chart(fig_f, use_container_width=True)
    
    # ── Yield Prediction ─────────────────────────
    with col_yield:
        st.markdown('<div class="section-header">📈 Yield Prediction</div>', unsafe_allow_html=True)
    
        pred_tha    = round(predicted / 1000, 2)
        avg_tha     = round(avg_yield  / 1000, 2)
        max_tha     = round(avg_tha * 1.5, 2)
    
        fig_y = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_tha,
            delta={
                "reference": avg_tha,
                "suffix": " t/ha",
                "increasing": {"color": "#5F8C6A"},
                "decreasing": {"color": "#B85C5C"},
                "font": {"color": "#2C2416"},
            },
            number={"suffix": " t/ha", "font": {"size": 24, "color": "#2C2416", "family": "DM Sans"}},
            title={"text": f"{crop_choice}  ({int(predicted):,} kg/ha)",
                   "font": {"size": 12, "color": "#2C2416", "family": "DM Sans"}},
            gauge={
                "axis": {
                    "range": [0, max_tha],
                    "tickfont": {"size": 9, "color": "#2C2416"},
                },
                "bar":  {"color": "#7C6D5A"},
                "bgcolor": "#F7F4EF",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,           avg_tha * 0.6], "color": "#FDEAEA"},
                    {"range": [avg_tha*0.6, avg_tha],       "color": "#FDF6E3"},
                    {"range": [avg_tha,     max_tha],        "color": "#D6EDD9"},
                ],
                "threshold": {
                    "line": {"color": "#5F8C6A", "width": 2},
                    "thickness": 0.8,
                    "value": avg_tha,
                },
            },
        ))
        fig_y.update_layout(**PL, height=260)
        st.plotly_chart(fig_y, use_container_width=True)
    
        if diff_pct >= 5:
            bg,border,label,msg = "#CFF1DB","#5F8C6A","🟢 Above Average","Excellent conditions — strong harvest expected."
        elif diff_pct >= -5:
            bg,border,label,msg = "#E5DBC1","#C49A3C","🟡 Near Average","Close to regional average; minor tweaks can help."
        else:
            bg,border,label,msg = "#E0C9C9","#B85C5C","🔴 Below Average","Review soil health and fertilizer recommendations."
    
        st.markdown(f"""
        <div style='background:{bg}; border-left:3px solid {border}; border-radius:8px;
                    padding:12px 16px; margin-bottom:16px'>
            <b style='color:#2C2416'>{label}</b> &nbsp;·&nbsp;
            <span style='font-size:0.84rem; color:#4A3E2E; font-weight:500'>{msg}</span>
        </div>
        """, unsafe_allow_html=True)
    
        comp_crops  = [crop_choice] + [c for c in ALL_CROPS if c != crop_choice]
        seed_base   = int(N * 7 + P * 13 + K * 17 + T * 3 + R * 0.1 + pH * 100) % (2**31)
        comp_yields_kgha = [predicted]
        for i, c in enumerate(comp_crops[1:]):
            rng_c = np.random.default_rng(seed_base + i + abs(hash(c)) % 1000)
            comp_yields_kgha.append(YIELD_BASELINE.get(c, 3000) * rng_c.uniform(0.9, 1.1))
    
        comp_yields_tha = [round(y / 1000, 2) for y in comp_yields_kgha]
        bar_colors = ["#7C6D5A" if c == crop_choice else "#C9B99A" for c in comp_crops]
        bar_texts  = [f"{y} t/ha" for y in comp_yields_tha]
    
        fig_c = go.Figure(go.Bar(
            x=comp_crops,
            y=comp_yields_tha,
            marker_color=bar_colors,
            text=bar_texts,
            textposition="outside",
            textfont=dict(color="#2C2416", size=11),
        ))
        fig_c.update_layout(
            **PL,
            height=240,
            yaxis=dict(
                title="Yield (t/ha)",
                range=[0, max(comp_yields_tha) * 1.25],
                tickfont=TICK_FONT,
                title_font=TITLE_FONT,
                gridcolor="#E8E2D9",
            ),
            xaxis=dict(
                tickfont=TICK_FONT,
                title_font=TITLE_FONT,
            ),
        )
        st.plotly_chart(fig_c, use_container_width=True)

# ═══════════════════════════════════════════════
#  CHAT ASSISTANT
# ═══════════════════════════════════════════════
st.markdown("<hr style='margin:28px 0'>", unsafe_allow_html=True)
st.markdown('<div class="section-header">💬 Smart Agri Assistant</div>', unsafe_allow_html=True)

if not CHATBOT_AVAILABLE:
    st.warning("Chatbot modules could not be loaded.")
elif not st.session_state.get("report"):
    st.info("Run an analysis above first — then you can ask questions about your crop, soil, fertilizer, and yield results.")
else:
    lang_labels = {
        "auto": "Auto / स्वयंचलित",
        "en": "English",
        "hi": "हिंदी",
        "mr": "मराठी",
    }
    placeholders = {
        "auto": "Ask about your analysis… / विश्लेषण विचारा…",
        "en": "Ask about your analysis…",
        "hi": "अपने विश्लेषण के बारे में पूछें…",
        "mr": "तुमच्या विश्लेषणाबद्दल विचारा…",
    }
    st.radio(
        "Chat language / चॅट भाषा",
        options=["auto", "en", "hi", "mr"],
        format_func=lambda key: lang_labels[key],
        horizontal=True,
        key="chat_lang",
    )
    classifier = load_classifier()
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(placeholders.get(st.session_state.chat_lang, placeholders["auto"])):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        intent = predict_intent(classifier, prompt)
        response = generate_response(
            intent,
            st.session_state.report,
            prompt,
            st.session_state.chat_context,
            lang=st.session_state.chat_lang,
        )
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()

# ── Footer ───────────────────────────────────
st.markdown("""
<hr style='margin:32px 0 12px'>
<div style='text-align:center; color:#6B5E4C; font-size:0.74rem;
            padding-bottom:20px; font-weight:500'>
    AgriSmart DSS · Final Year Project · Built with Streamlit 🌾
</div>
""", unsafe_allow_html=True)