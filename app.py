import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from utils.agent import run_farming_agent
from utils.crop_model import CROP_FEATURES, load_crop_dataset, recommend_crop
from utils.disease_model import detect_crop_disease, load_disease_artifacts
from utils.irrigation_model import load_irrigation_dataset, predict_irrigation
from utils.yield_model import (
    forecast_crop_yield,
    get_supported_countries,
    get_supported_crops,
    load_yield_dataset,
)


st.set_page_config(
    page_title="Precision Agriculture Intelligence Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    /* ── Tokens ── */
    :root {
        --field: #1a7a4a;
        --field-dark: #12603a;
        --field-glow: #22c66e;
        --leaf: #47a867;
        --leaf-soft: rgba(71,168,103,.12);
        --soil: #7a5a38;
        --soil-light: #a07c52;
        --sun: #e8a735;
        --sun-soft: rgba(232,167,53,.10);
        --sky: #2f80a7;
        --cream: #f7f5ee;
        --cream-warm: #faf8f0;
        --ink: #15302a;
        --ink-soft: #2c4a40;
        --muted: #5d7168;
        --line: #cddcc5;
        --card-bg: rgba(255,255,255,.82);
        --glass: rgba(255,255,255,.55);
    }

    /* ── Typography ── */
    html, body, .stApp,
    .stApp p, .stApp span, .stApp div, .stApp li, .stApp td, .stApp th,
    .stApp label, .stApp input, .stApp textarea, .stApp select, .stApp button,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        font-family: 'Inter', 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        font-size: 16px;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: var(--ink);
        letter-spacing: -0.01em;
        font-weight: 600;
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.25rem !important; }
    p, li, span, div { font-size: 1rem; }
    label { font-size: 0.95rem !important; font-weight: 500 !important; }
    .stMarkdown p { font-size: 1rem; line-height: 1.65; }

    /* ── Main App Background ── */
    .stApp {
        background-color: #f3f6ed;
        background-image:
            repeating-linear-gradient(
                90deg,
                rgba(26,122,74,.03) 0px,
                rgba(26,122,74,.03) 1px,
                transparent 1px,
                transparent 40px
            ),
            repeating-linear-gradient(
                0deg,
                rgba(122,90,56,.025) 0px,
                rgba(122,90,56,.025) 1px,
                transparent 1px,
                transparent 40px
            ),
            radial-gradient(circle at 20% 50%, rgba(71,168,103,.06) 0%, transparent 50%),
            radial-gradient(circle at 80% 30%, rgba(232,167,53,.05) 0%, transparent 50%),
            radial-gradient(circle at 50% 80%, rgba(122,90,56,.04) 0%, transparent 50%),
            linear-gradient(175deg, #eef3e6 0%, #f3f6ed 20%, #f7f5ee 45%, #faf7ef 65%, #f0ece0 85%, #e8e4d5 100%);
        color: var(--ink);
    }
    .stApp::after {
        content: '';
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 100px;
        pointer-events: none;
        z-index: 0;
        background: linear-gradient(0deg, rgba(26,122,74,.06) 0%, rgba(122,90,56,.03) 40%, transparent 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #e2ebd8;
        background-image:
            repeating-linear-gradient(
                135deg,
                rgba(26,122,74,.05) 0px,
                rgba(26,122,74,.05) 1px,
                transparent 1px,
                transparent 12px
            ),
            radial-gradient(circle at 30% 70%, rgba(71,168,103,.08) 0%, transparent 50%),
            radial-gradient(circle at 70% 30%, rgba(232,167,53,.06) 0%, transparent 40%),
            linear-gradient(195deg, #dfe9d3 0%, #dae5ce 35%, #e5e0d0 70%, #e0dbc8 100%);
        border-right: 2px solid var(--line);
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.35rem;
        background: linear-gradient(135deg, var(--field) 0%, var(--leaf) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    section[data-testid="stSidebar"] button[kind="icon"] {
        font-size: 0 !important;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
    }
    section[data-testid="stSidebar"] button[kind="icon"]::before {
        content: "☰";
        font-size: 1rem;
        line-height: 1;
    }
    section[data-testid="stSidebar"] button[kind="icon"] span,
    section[data-testid="stSidebar"] button[kind="icon"] svg {
        display: none !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: var(--leaf-soft);
        border-radius: 6px;
        transition: background .2s ease;
    }

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: var(--card-bg);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--line);
        border-left: 5px solid var(--field);
        padding: 16px 18px;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(26,122,74,.06);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26,122,74,.10);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
        color: var(--field);
    }

    /* ── Hero Banner ── */
    .hero {
        border: none;
        background:
            linear-gradient(105deg, rgba(21,47,38,.92) 0%, rgba(26,60,45,.78) 40%, rgba(26,60,45,.35) 100%),
            url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80");
        background-position: center;
        background-size: cover;
        min-height: 220px;
        padding: 38px 40px;
        border-radius: 14px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow:
            0 4px 24px rgba(21,47,38,.18),
            inset 0 -1px 0 rgba(255,255,255,.08);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(255,255,255,.04) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 30%, rgba(255,255,255,.03) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero h1 {
        color: #fffdf2;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 10px 0;
        text-shadow: 0 2px 12px rgba(0,0,0,.4);
        position: relative;
    }
    .hero p {
        color: #e5f0de;
        font-size: 1.05rem;
        margin: 0;
        max-width: 820px;
        text-shadow: 0 1px 6px rgba(0,0,0,.35);
        line-height: 1.6;
        position: relative;
    }

    /* ── Farm Strip (animated gradient) ── */
    .farm-strip {
        background: linear-gradient(90deg,
            #12603a 0%, #1a7a4a 18%,
            #47a867 32%, #22c66e 38%,
            #e8a735 42%, #d4952e 48%,
            #a07c52 55%, #7a5a38 70%,
            #5c3f24 100%);
        height: 6px;
        border-radius: 6px;
        margin: 2px 0 20px 0;
        background-size: 200% 100%;
        animation: farmStripShift 8s ease-in-out infinite alternate;
    }
    @keyframes farmStripShift {
        0%   { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* ── Farm Panel ── */
    .farm-panel {
        border: 1px solid var(--line);
        background: var(--card-bg);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-left: 5px solid var(--field);
        padding: 18px 20px;
        border-radius: 10px;
        margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(26,122,74,.05);
        transition: box-shadow .2s ease;
    }
    .farm-panel:hover {
        box-shadow: 0 4px 18px rgba(26,122,74,.09);
    }

    /* ── Status Band ── */
    .status-band {
        border: 1px solid #e8dfc0;
        border-left: 5px solid var(--sun);
        background: linear-gradient(135deg, #fffdf5 0%, #fef9e7 100%);
        padding: 14px 18px;
        border-radius: 10px;
        color: #4c4023;
        box-shadow: 0 2px 8px rgba(232,167,53,.08);
    }

    /* ── Result Box ── */
    .result-box {
        border: 1px solid var(--line);
        border-left: 5px solid var(--sky);
        background: var(--card-bg);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        padding: 18px 20px;
        border-radius: 10px;
        margin-top: 10px;
        box-shadow: 0 2px 10px rgba(47,128,167,.06);
    }

    /* ── Muted Text ── */
    .small-muted {
        color: var(--muted);
        font-size: .9rem;
        line-height: 1.5;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px;
        background: linear-gradient(135deg, var(--field) 0%, var(--leaf) 100%);
        color: white;
        border: none;
        min-height: 48px;
        min-width: 120px;
        padding: 10px 24px;
        white-space: normal;
        font-weight: 600;
        font-size: 0.95rem !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 2px 8px rgba(26,122,74,.15);
        transition: all .25s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--field-dark) 0%, var(--field) 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(26,122,74,.22);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 4px rgba(26,122,74,.12);
    }
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem !important;
        min-height: 48px;
        padding: 10px 24px;
    }
    [data-testid="stFormSubmitButton"] > button {
        font-size: 1rem !important;
        font-weight: 700;
        min-height: 52px;
        padding: 12px 32px;
        letter-spacing: 0.01em;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        font-family: 'Inter', sans-serif !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--leaf-soft);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: var(--card-bg);
    }

    /* ── Forms ── */
    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 20px;
        background: var(--card-bg);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        box-shadow: 0 2px 10px rgba(26,122,74,.04);
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] button[kind="secondary"] {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }
    [data-testid="stFileUploader"] button[kind="secondary"]::before {
        content: "📤";
        font-size: 1rem;
    }
    [data-testid="stFileUploader"] button[kind="secondary"]::after {
        content: "Upload";
        font-weight: 600;
    }
    [data-testid="stFileUploader"] button[kind="secondary"] span {
        display: none;
    }
    .stChatInput button[kind="primary"]::before {
        content: "➤";
        font-size: 1rem;
        margin-right: 0;
    }
    .stChatInput button[kind="primary"] span {
        display: none;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 7px; }
    ::-webkit-scrollbar-track { background: var(--cream); }
    ::-webkit-scrollbar-thumb {
        background: var(--line);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--muted);
    }

    /* ── Inputs & Widgets ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        font-size: 0.95rem !important;
        border-radius: 8px;
    }
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stMultiSelect > label,
    .stFileUploader > label {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: var(--ink-soft);
    }
    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] > div {
        font-size: 0.95rem;
    }
    .stRadio > label { font-size: 1rem !important; }
    .stRadio [role="radiogroup"] label {
        font-size: 1rem !important;
        padding: 6px 10px;
    }
    .stChatInput > div > textarea,
    .stChatInput > div > div > textarea {
        font-size: 1rem !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 1.05rem !important;
        font-weight: 600;
        list-style: none !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    [data-testid="stExpander"] summary::-webkit-details-marker {
        display: none !important;
    }
    [data-testid="stExpander"] summary > span > span {
        display: none !important;
    }
    [data-testid="stExpander"] summary > span > div {
        display: block !important;
    }
    [data-testid="stExpander"] summary::before {
        content: "🌾";
        font-size: 1rem;
        line-height: 1;
    }
    [data-testid="stExpander"] summary::after {
        content: "▾";
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1;
        transition: transform 0.2s ease;
    }
    [data-testid="stExpander"]:not([open]) summary::after {
        content: "▸";
    }
    .stAlert p { font-size: 0.95rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def crop_data():
    return load_crop_dataset()


@st.cache_data(show_spinner=False)
def irrigation_data():
    return load_irrigation_dataset()


@st.cache_data(show_spinner=False)
def yield_data():
    return load_yield_dataset()


@st.cache_data(show_spinner=False)
def disease_artifacts():
    return load_disease_artifacts()


@st.cache_data(show_spinner=False)
def supported_yield_countries():
    return get_supported_countries()


@st.cache_data(show_spinner=False)
def supported_yield_crops():
    return get_supported_crops()


def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        <div class="farm-strip"></div>
        """,
        unsafe_allow_html=True,
    )


def result_box(title, body):
    st.markdown(
        f"""
        <div class="result-box">
            <h3>{title}</h3>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def default_crop_context():
    return {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 24.0,
        "humidity": 78.0,
        "ph": 6.5,
        "rainfall": 210.0,
    }


def default_irrigation_context():
    return {
        "Soil_Type": "Loamy",
        "Soil_pH": 6.7,
        "Soil_Moisture": 34.0,
        "Organic_Carbon": 1.1,
        "Electrical_Conductivity": 1.8,
        "Temperature_C": 28.0,
        "Humidity": 64.0,
        "Rainfall_mm": 220.0,
        "Sunlight_Hours": 7.5,
        "Wind_Speed_kmh": 9.0,
        "Crop_Type": "Rice",
        "Crop_Growth_Stage": "Vegetative",
        "Season": "Kharif",
        "Irrigation_Type": "Drip",
        "Water_Source": "Reservoir",
        "Field_Area_hectare": 2.5,
        "Mulching_Used": "Yes",
        "Previous_Irrigation_mm": 45.0,
        "Region": "South",
    }


def default_yield_context():
    countries = supported_yield_countries()
    crops = supported_yield_crops()
    return {
        "country": "India" if "India" in countries else countries[0],
        "crop": "Rice" if "Rice" in crops else crops[0],
        "year": datetime.now().year,
        "rainfall_mm": 1200.0,
        "temperature_c": 27.5,
        "pesticides_tonnes": 45.2,
    }


def configured_gemini_key():
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("GEMINI_API_KEY", "")


def page_crop_recommendation():
    hero(
        "Crop Recommendation",
        "Enter soil nutrients and local weather conditions to recommend the most suitable crop.",
    )

    defaults = default_crop_context()
    with st.form("crop_form"):
        col1, col2, col3, col4 = st.columns(4)
        payload = {
            "N": col1.number_input("Nitrogen N", 0, 200, defaults["N"]),
            "P": col2.number_input("Phosphorus P", 0, 200, defaults["P"]),
            "K": col3.number_input("Potassium K", 0, 250, defaults["K"]),
            "ph": col4.number_input("Soil pH", 0.0, 14.0, defaults["ph"], step=0.1),
            "temperature": col1.number_input("Temperature C", -10.0, 60.0, defaults["temperature"], step=0.5),
            "humidity": col2.number_input("Humidity %", 0.0, 100.0, defaults["humidity"], step=0.5),
            "rainfall": col3.number_input("Rainfall mm", 0.0, 500.0, defaults["rainfall"], step=1.0),
        }
        submitted = st.form_submit_button("Recommend Crop")

    if submitted:
        result = recommend_crop(payload)
        top = result["top_predictions"]
        result_box(
            "Recommended Crop",
            f"<b style='font-size:1.45rem;color:#1f7a4d'>{result['recommended_crop'].title()}</b>",
        )
        if top:
            st.markdown("### Top Model Probabilities")
            st.dataframe(pd.DataFrame(top), width="stretch", hide_index=True)

    st.markdown("### Crop Dataset Explorer")
    cdf = crop_data()
    selected = st.multiselect(
        "Filter crops",
        sorted(cdf["label"].unique()),
        default=sorted(cdf["label"].unique())[:5],
    )
    filtered = cdf[cdf["label"].isin(selected)] if selected else cdf
    fig = px.scatter(
        filtered,
        x="rainfall",
        y="humidity",
        color="label",
        size="temperature",
        hover_data=CROP_FEATURES,
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, width="stretch")


def page_irrigation():
    hero(
        "Irrigation Prediction",
        "Estimate irrigation requirement from soil, climate, crop stage, water source, and field scale.",
    )

    d = default_irrigation_context()
    with st.form("irrigation_form"):
        col1, col2, col3, col4 = st.columns(4)
        payload = {
            "Soil_Type": col1.selectbox("Soil Type", ["Clay", "Loamy", "Sandy", "Silt"], index=1),
            "Crop_Type": col2.selectbox("Crop Type", ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Wheat"], index=3),
            "Crop_Growth_Stage": col3.selectbox("Growth Stage", ["Flowering", "Harvest", "Sowing", "Vegetative"], index=3),
            "Season": col4.selectbox("Season", ["Kharif", "Rabi", "Zaid"], index=0),
            "Irrigation_Type": col1.selectbox("Irrigation Type", ["Canal", "Drip", "Rainfed", "Sprinkler"], index=1),
            "Water_Source": col2.selectbox("Water Source", ["Groundwater", "Rainwater", "Reservoir", "River"], index=2),
            "Mulching_Used": col3.selectbox("Mulching Used", ["No", "Yes"], index=1),
            "Region": col4.selectbox("Region", ["Central", "East", "North", "South", "West"], index=3),
            "Soil_pH": col1.number_input("Soil pH", 0.0, 14.0, d["Soil_pH"], step=0.1),
            "Soil_Moisture": col2.number_input("Soil Moisture %", 0.0, 100.0, d["Soil_Moisture"], step=0.5),
            "Organic_Carbon": col3.number_input("Organic Carbon", 0.0, 5.0, d["Organic_Carbon"], step=0.05),
            "Electrical_Conductivity": col4.number_input("Electrical Conductivity", 0.0, 10.0, d["Electrical_Conductivity"], step=0.05),
            "Temperature_C": col1.number_input("Temperature C", -10.0, 60.0, d["Temperature_C"], step=0.5),
            "Humidity": col2.number_input("Humidity %", 0.0, 100.0, d["Humidity"], step=0.5),
            "Rainfall_mm": col3.number_input("Rainfall mm", 0.0, 3000.0, d["Rainfall_mm"], step=1.0),
            "Sunlight_Hours": col4.number_input("Sunlight Hours", 0.0, 15.0, d["Sunlight_Hours"], step=0.1),
            "Wind_Speed_kmh": col1.number_input("Wind Speed km/h", 0.0, 80.0, d["Wind_Speed_kmh"], step=0.5),
            "Field_Area_hectare": col2.number_input("Field Area hectare", 0.1, 100.0, d["Field_Area_hectare"], step=0.1),
            "Previous_Irrigation_mm": col3.number_input("Previous Irrigation mm", 0.0, 500.0, d["Previous_Irrigation_mm"], step=1.0),
        }
        submitted = st.form_submit_button("Predict Irrigation Need")

    if submitted:
        result = predict_irrigation(payload)
        color = {"Low": "#1f7a4d", "Medium": "#e8a735", "High": "#c6533d"}.get(
            result["irrigation_need"], "#2f80a7"
        )
        result_box(
            "Irrigation Need",
            f"<b style='font-size:1.45rem;color:{color}'>{result['irrigation_need']}</b>",
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Probability")
            st.dataframe(pd.DataFrame(result["probabilities"]), width="stretch", hide_index=True)
        with col2:
            st.markdown("### Water Stress Indicators")
            st.dataframe(
                pd.DataFrame([result["stress_indicators"]]).T.rename(columns={0: "value"}),
                width="stretch",
            )




def page_yield():
    hero(
        "Crop Yield Forecasting",
        "Forecast crop yield using country, crop, year, rainfall, temperature, and pesticide inputs.",
    )

    d = default_yield_context()
    countries = supported_yield_countries()
    crops = supported_yield_crops()
    with st.form("yield_form"):
        col1, col2, col3 = st.columns(3)
        payload = {
            "country": col1.selectbox("Country", countries, index=countries.index(d["country"])),
            "crop": col2.selectbox("Crop", crops, index=crops.index(d["crop"])),
            "year": col3.number_input("Year", 1991, 2050, int(d["year"])),
            "rainfall_mm": col1.number_input("Annual Rainfall mm", 0.0, 5000.0, d["rainfall_mm"], step=10.0),
            "temperature_c": col2.number_input("Average Temperature C", -10.0, 50.0, d["temperature_c"], step=0.2),
            "pesticides_tonnes": col3.number_input("Pesticides tonnes", 0.0, 100000.0, d["pesticides_tonnes"], step=1.0),
        }
        submitted = st.form_submit_button("Forecast Yield")

    if submitted:
        result = forecast_crop_yield(payload)
        result_box(
            "Forecasted Yield",
            f"<b style='font-size:1.45rem;color:#1f7a4d'>{result['predicted_yield_hg_per_ha']:,} {result['unit']}</b><br>"
            f"Confidence: <b>{result['confidence_tier']}</b> | R2: {result['confidence_r2']}",
        )
        if result["warnings"]:
            st.warning(" ".join(result["warnings"]))
        st.download_button(
            "Download Yield Report",
            data=json.dumps(result, indent=2),
            file_name="yield_forecast_report.json",
            mime="application/json",
        )

    st.markdown("### Yield Trends")
    ydf = yield_data()
    col1, col2 = st.columns([1, 1])
    country = col1.selectbox("Trend Country", countries, index=countries.index(d["country"]), key="trend_country")
    crop = col2.selectbox("Trend Crop", crops, index=crops.index(d["crop"]), key="trend_crop")
    trend = ydf[(ydf["Area"] == country) & (ydf["Item"] == crop)]
    if trend.empty:
        st.info("No trend records for this combination.")
    else:
        fig = px.line(
            trend,
            x="Year",
            y="hg/ha_yield",
            markers=True,
            color_discrete_sequence=["#1f7a4d"],
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, width="stretch")


def page_disease():
    hero(
        "Crop Disease Detection",
        "Review disease model performance and upload a crop leaf image when the trained PyTorch artifact is available.",
    )

    artifacts = disease_artifacts()
    if artifacts.get("fall_back_to_visual"):
        st.markdown(
            """
            <div class="status-band">
                📌 <b>Note:</b> Live image classification is using a lightweight visual fallback because the trained model checkpoint is not present locally.
                Disease analytics, model comparison, and class reference are available below.
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader("📤 Upload leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded crop leaf", width="stretch")
        result = detect_crop_disease(uploaded)
        st.info(result["message"])

    st.markdown("### Disease Model Comparison")
    comparison = artifacts["model_comparison"]
    fig = px.bar(
        comparison,
        x="model",
        y=["val_acc", "test_acc", "test_f1_macro"],
        barmode="group",
        color_discrete_sequence=["#47a867", "#2f80a7", "#e8a735"],
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Disease Classes")
        st.dataframe(artifacts["class_summary"], width="stretch", hide_index=True)
    with col2:
        st.markdown("### Classification Report")
        st.dataframe(artifacts["classification_report"], width="stretch", hide_index=True)


def agent_context_form():
    crop_defaults = default_crop_context()
    irrigation_defaults = default_irrigation_context()
    yield_defaults = default_yield_context()

    with st.expander("Farm Context — Soil, Weather & Crop Parameters", expanded=True):
        tab1, tab2, tab3 = st.tabs(["🌱 Soil & Crop", "💧 Irrigation", "📊 Yield Forecast"])
        with tab1:
            cols = st.columns(4)
            crop_payload = {
                "N": cols[0].number_input("Nitrogen (N)", 0, 200, crop_defaults["N"]),
                "P": cols[1].number_input("Phosphorus (P)", 0, 200, crop_defaults["P"]),
                "K": cols[2].number_input("Potassium (K)", 0, 250, crop_defaults["K"]),
                "ph": cols[3].number_input("Soil pH", 0.0, 14.0, crop_defaults["ph"], step=0.1),
                "temperature": cols[0].number_input("Temperature (°C)", -10.0, 60.0, crop_defaults["temperature"], step=0.5),
                "humidity": cols[1].number_input("Humidity (%)", 0.0, 100.0, crop_defaults["humidity"], step=0.5),
                "rainfall": cols[2].number_input("Rainfall (mm)", 0.0, 500.0, crop_defaults["rainfall"], step=1.0),
            }
        with tab2:
            cols = st.columns(4)
            irrigation_payload = {
                "Soil_Type": cols[0].selectbox("Soil Type", ["Clay", "Loamy", "Sandy", "Silt"], index=1, key="ctx_soil_type"),
                "Crop_Type": cols[1].selectbox("Crop Type", ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Wheat"], index=3, key="ctx_crop_type"),
                "Crop_Growth_Stage": cols[2].selectbox("Growth Stage", ["Flowering", "Harvest", "Sowing", "Vegetative"], index=3, key="ctx_growth_stage"),
                "Season": cols[3].selectbox("Season", ["Kharif", "Rabi", "Zaid"], index=0, key="ctx_season"),
                "Irrigation_Type": cols[0].selectbox("Irrigation Method", ["Canal", "Drip", "Rainfed", "Sprinkler"], index=1, key="ctx_irrigation_type"),
                "Water_Source": cols[1].selectbox("Water Source", ["Groundwater", "Rainwater", "Reservoir", "River"], index=2, key="ctx_water_source"),
                "Mulching_Used": cols[2].selectbox("Mulching Used", ["No", "Yes"], index=1, key="ctx_mulching"),
                "Region": cols[3].selectbox("Region", ["Central", "East", "North", "South", "West"], index=3, key="ctx_region"),
                "Soil_pH": cols[0].number_input("Soil pH Level", 0.0, 14.0, irrigation_defaults["Soil_pH"], step=0.1),
                "Soil_Moisture": cols[1].number_input("Soil Moisture (%)", 0.0, 100.0, irrigation_defaults["Soil_Moisture"], step=0.5),
                "Organic_Carbon": cols[2].number_input("Organic Carbon", 0.0, 5.0, irrigation_defaults["Organic_Carbon"], step=0.05),
                "Electrical_Conductivity": cols[3].number_input("Electrical Conductivity", 0.0, 10.0, irrigation_defaults["Electrical_Conductivity"], step=0.05),
                "Temperature_C": cols[0].number_input("Temperature (°C)", -10.0, 60.0, irrigation_defaults["Temperature_C"], step=0.5, key="ctx_irr_temp"),
                "Humidity": cols[1].number_input("Humidity (%)", 0.0, 100.0, irrigation_defaults["Humidity"], step=0.5, key="ctx_irr_humidity"),
                "Rainfall_mm": cols[2].number_input("Rainfall (mm)", 0.0, 3000.0, irrigation_defaults["Rainfall_mm"], step=1.0, key="ctx_irr_rainfall"),
                "Sunlight_Hours": cols[3].number_input("Sunlight Hours", 0.0, 15.0, irrigation_defaults["Sunlight_Hours"], step=0.1),
                "Wind_Speed_kmh": cols[0].number_input("Wind Speed (km/h)", 0.0, 80.0, irrigation_defaults["Wind_Speed_kmh"], step=0.5),
                "Field_Area_hectare": cols[1].number_input("Field Area (ha)", 0.1, 100.0, irrigation_defaults["Field_Area_hectare"], step=0.1),
                "Previous_Irrigation_mm": cols[2].number_input("Previous Irrigation (mm)", 0.0, 500.0, irrigation_defaults["Previous_Irrigation_mm"], step=1.0),
            }
        with tab3:
            countries = supported_yield_countries()
            crops = supported_yield_crops()
            cols = st.columns(3)
            yield_payload = {
                "country": cols[0].selectbox("Country", countries, index=countries.index(yield_defaults["country"]), key="ctx_country"),
                "crop": cols[1].selectbox("Crop", crops, index=crops.index(yield_defaults["crop"]), key="ctx_crop"),
                "year": cols[2].number_input("Year", 1991, 2050, int(yield_defaults["year"]), key="ctx_year"),
                "rainfall_mm": cols[0].number_input("Annual Rainfall (mm)", 0.0, 5000.0, yield_defaults["rainfall_mm"], step=10.0),
                "temperature_c": cols[1].number_input("Avg Temperature (°C)", -10.0, 50.0, yield_defaults["temperature_c"], step=0.2),
                "pesticides_tonnes": cols[2].number_input("Pesticides (tonnes)", 0.0, 100000.0, yield_defaults["pesticides_tonnes"], step=1.0),
            }

    return {"crop": crop_payload, "irrigation": irrigation_payload, "yield": yield_payload}


def page_agent():
    hero(
        "AI Farming Assistant",
        "A single agent routes farmer questions to the right local model, then optionally uses Gemini to produce the final advisory answer.",
    )

    saved_api_key = configured_gemini_key()
    st.markdown(
        f"""
        <div class="status-band">
            🔑 Gemini status: <b>{"Configured" if saved_api_key else "Not configured"}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    api_key = st.text_input(
        "Override Gemini API key",
        value="",
        type="password",
        help="Optional. Leave blank to use the local Streamlit secret.",
    )
    api_key = api_key or saved_api_key
    context = agent_context_form()

    st.markdown(
        """
        <div class="farm-panel">
            🌾 <b>Suggested farmer questions</b>
            <div class="small-muted">💡 Click one to run the assistant through the relevant farm model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    suggested_questions = [
        "Which crop is suitable for my current soil and weather?",
        "How much irrigation attention does my field need today?",
        "Forecast yield for this crop and explain the confidence.",
        "Analyze this crop leaf image for disease symptoms.",
    ]
    clicked_query = None
    qcols = st.columns(4)
    for index, question in enumerate(suggested_questions):
        if qcols[index].button(question, key=f"suggested_question_{index}"):
            clicked_query = question

    st.markdown("### 📸 Leaf Image Input")
    uploaded_leaf = st.file_uploader(
        "📤 Upload a crop leaf image for the disease module",
        type=["jpg", "jpeg", "png"],
        key="agent_leaf_image",
    )
    image_context = None
    image_result = None
    if uploaded_leaf:
        leaf_image = Image.open(uploaded_leaf)
        image_context = {
            "name": uploaded_leaf.name,
            "width": leaf_image.width,
            "height": leaf_image.height,
            "mode": leaf_image.mode,
            "format": leaf_image.format or uploaded_leaf.type,
        }
        image_result = detect_crop_disease(uploaded_leaf)
        col1, col2 = st.columns([0.7, 1.3])
        with col1:
            st.image(leaf_image, caption="Assistant leaf image", width="stretch")
        with col2:
            st.markdown(
                f"""
                <div class="status-band">
                    Image attached: <b>{image_context['name']}</b><br>
                    Size: {image_context['width']} x {image_context['height']} px<br>
                    Disease module response: {image_result['message']}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Analyze Uploaded Image", key="analyze_uploaded_leaf"):
                clicked_query = "Analyze this crop leaf image for disease symptoms."

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me about crop choice, irrigation need, yield forecast, or disease support.",
            }
        ]

    for message in st.session_state.messages:
        avatar = "👨‍🌾" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    query = clicked_query or st.chat_input("💬 Ask your farm question")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👨‍🌾"):
            st.write(query)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Routing query through farm intelligence tools..."):
                result = run_farming_agent(
                    query,
                    context,
                    api_key=api_key,
                    image_context=image_context,
                    image_result=image_result,
                    image_payload=leaf_image if uploaded_leaf else None,
                )
            st.write(result["answer"])
            with st.expander("Agent Tool Trace"):
                st.json(
                    {
                        "routes": result["routes"],
                        "llm_used": result["llm_used"],
                        "errors": result["errors"],
                        "tool_outputs": result["tool_outputs"],
                    }
                )
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


def page_analytics():
    hero(
        "Farm Analytics Dashboard",
        "Explore the agricultural database across crop suitability, irrigation behavior, disease performance, and yield history.",
    )

    cdf = crop_data()
    idf = irrigation_data()
    ydf = yield_data()
    disease = disease_artifacts()

    tab1, tab2, tab3, tab4 = st.tabs(["Crop Data", "Irrigation Data", "Yield Data", "Disease Data"])
    with tab1:
        col1, col2 = st.columns([1, 1])
        fig = px.box(cdf, x="label", y="ph", color="label")
        fig.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=25, b=10))
        col1.plotly_chart(fig, width="stretch")
        corr = cdf[CROP_FEATURES].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=["#c6533d", "#f7f4ea", "#1f7a4d"])
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10))
        col2.plotly_chart(fig, width="stretch")
    with tab2:
        sample = idf.sample(min(30000, max(1, len(idf))), random_state=11).copy()
        for column in ["Rainfall_mm", "Soil_Moisture", "Irrigation_Need"]:
            if column not in sample.columns:
                sample[column] = pd.NA
        sample = sample.dropna(subset=["Rainfall_mm", "Soil_Moisture", "Irrigation_Need"])
        if sample.empty:
            sample = pd.DataFrame(
                {
                    "Rainfall_mm": [500.0, 800.0, 1200.0],
                    "Soil_Moisture": [40.0, 55.0, 25.0],
                    "Irrigation_Need": ["Low", "Medium", "High"],
                    "Season": ["Kharif", "Rabi", "Zaid"],
                }
            )
        fig = px.bar(
            sample,
            x="Irrigation_Need",
            y="Soil_Moisture",
            color="Irrigation_Need",
            color_discrete_map={"Low": "#47a867", "Medium": "#e8a735", "High": "#c6533d"},
        )
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with tab3:
        country = st.selectbox("Country", sorted(ydf["Area"].unique()), index=sorted(ydf["Area"].unique()).index("India") if "India" in set(ydf["Area"]) else 0)
        data = ydf[ydf["Area"] == country]
        fig = px.line(data, x="Year", y="hg/ha_yield", color="Item")
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, width="stretch")
    with tab4:
        report = disease["classification_report"]
        class_rows = report[~report["Unnamed: 0"].isin(["accuracy", "macro avg", "weighted avg"])]
        fig = px.bar(
            class_rows,
            x="Unnamed: 0",
            y="f1-score",
            color="f1-score",
            color_continuous_scale=["#e8a735", "#47a867", "#1f7a4d"],
        )
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, width="stretch")


PAGES = {
    " Crop Recommendation": page_crop_recommendation,
    "🔬 Disease Detection": page_disease,
    "💧 Irrigation Prediction": page_irrigation,
    "📊 Yield Forecasting": page_yield,
    "🤖 AI Farming Assistant": page_agent,
    "📈 Farm Analytics": page_analytics,
}


with st.sidebar:
    st.markdown("## 🌾 AgriIntel Platform")
    st.caption("Smart farming · AI-powered advisory")
    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.markdown(
        """
        <div class="small-muted">
        Local models power predictions. Gemini is optional for natural-language final reasoning.
        </div>
        """,
        unsafe_allow_html=True,
    )

PAGES[page]()
