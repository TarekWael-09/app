import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Greenhouse",
    page_icon="🌿",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body, .main { background-color: #f4faf4; }
    h1 { color: #1b5e20; letter-spacing: 1px; }
    .stSelectbox label { font-weight: 600; color: #2e7d32; font-size: 0.95rem; }
    .section-header {
        background: #e8f5e9;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        color: #2e7d32;
        margin: 1.2rem 0 0.6rem 0;
    }
    .actuator-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 1rem;
        font-weight: 600;
    }
    .on-badge  { background:#d4edda; color:#155724; padding:4px 20px; border-radius:20px; font-weight:700; }
    .off-badge { background:#f8d7da; color:#721c24; padding:4px 20px; border-radius:20px; font-weight:700; }
    .tip-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 3px solid #2e7d32;
    }

    /* ── Plant Switch ── */
    .switch-wrapper {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    .switch-label {
        font-weight: 700;
        font-size: 1.1rem;
        color: #1b5e20;
        min-width: 130px;
    }
</style>
""", unsafe_allow_html=True)

# ── Train models from CSV ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    df = pd.read_csv("a.csv")
    df["Plant_enc"] = (df["Plant_Type"] == "Cucumber").astype(int)
    features = ["Plant_enc", "Temperature", "Soil_Moisture", "LDR"]
    models = {}
    for target in ["Heater", "Pump", "Light", "Fan"]:
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(df[features], df[target])
        models[target] = clf
    return models

models = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌿 Smart Greenhouse")
st.markdown("Select your **plant type** and enter **sensor readings** to predict the optimal actuator states.")
st.divider()

# ── Plant Type Switch ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌱 Plant Type</div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:white; border-radius:12px; padding:1rem 1.5rem;
            box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-bottom:0.5rem;">
<p style="font-weight:600; color:#2e7d32; margin-bottom:0.5rem;">Select plant type:</p>
</div>
""", unsafe_allow_html=True)

plant_choice = st.radio(
    "Select plant type:",
    options=["🥒  Cucumber", "🌾  Quinoa"],
    index=0,
    label_visibility="collapsed",
)
plant_type = "Cucumber" if "Cucumber" in plant_choice else "Quinoa"

# ── Sensor inputs ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌡️ Sensor Readings</div>', unsafe_allow_html=True)

TEMP_RANGES = {
    "Very Cold  (< 22 °C)":   20.0,
    "Cold       (22–24 °C)":  23.0,
    "Cool       (24–26 °C)":  25.0,
    "Moderate   (26–28 °C)":  27.0,
    "Warm       (28–30 °C)":  29.0,
    "Hot        (30–32 °C)":  31.0,
    "Very Hot   (> 32 °C)":   34.0,
}
SOIL_RANGES = {
    "Very Dry   (< 45%)":    42.0,
    "Dry        (45–55%)":   50.0,
    "Moderate   (55–65%)":   60.0,
    "Moist      (65–75%)":   70.0,
    "Very Moist (> 75%)":    78.0,
}
LDR_RANGES = {
    "Very Dark   (< 250)":   220.0,
    "Dark        (250–350)": 300.0,
    "Dim         (350–450)": 400.0,
    "Bright      (450–540)": 490.0,
    "Very Bright (> 540)":   565.0,
}

col1, col2, col3 = st.columns(3)
with col1:
    temp_label = st.selectbox("🌡️ Temperature",      list(TEMP_RANGES.keys()))
with col2:
    soil_label = st.selectbox("💧 Soil Moisture",     list(SOIL_RANGES.keys()))
with col3:
    ldr_label  = st.selectbox("☀️ Light Level (LDR)", list(LDR_RANGES.keys()))

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Actuator States", use_container_width=True, type="primary"):

    plant_enc = 1 if plant_type == "Cucumber" else 0
    X = np.array([[
        plant_enc,
        TEMP_RANGES[temp_label],
        SOIL_RANGES[soil_label],
        LDR_RANGES[ldr_label],
    ]])

    predictions = {act: models[act].predict(X)[0] for act in models}

    # ── Results ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Predicted Actuator States</div>', unsafe_allow_html=True)

    icons  = {"Heater": "🔥", "Pump": "💦", "Light": "💡", "Fan": "🌀"}
    col_a, col_b = st.columns(2)
    for i, (act, pred) in enumerate(predictions.items()):
        badge = f'<span class="on-badge">ON ✅</span>' if pred == "on" else f'<span class="off-badge">OFF ❌</span>'
        card  = f'<div class="actuator-card"><span>{icons[act]} {act}</span>{badge}</div>'
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(card, unsafe_allow_html=True)

    on_list  = [act for act, v in predictions.items() if v == "on"]
    off_list = [act for act, v in predictions.items() if v == "off"]
    if on_list:
        st.success(f"✅ Active: {' | '.join(on_list)}")
    if off_list:
        st.info(f"⏹️ Inactive: {' | '.join(off_list)}")

    # ── Care Tips ─────────────────────────────────────────────────────────────
    TIPS = {
        "Cucumber": [
            ("💧", "Watering",          "Cucumbers love moisture. Keep soil consistently moist (65–80%). Avoid letting it dry out between waterings."),
            ("🌡️", "Temperature",       "Ideal range is 22–32 °C. Activate the heater if temperature drops below 22 °C to protect growth."),
            ("☀️", "Light",             "Cucumbers need 8–10 hours of light daily. Turn on grow lights when natural light (LDR) is low."),
            ("🌀", "Ventilation",       "Good airflow prevents fungal diseases. Keep the fan running, especially in warm and humid conditions."),
            ("💦", "Pump / Irrigation", "Use the pump regularly to maintain soil moisture. Drip irrigation is highly recommended for cucumbers."),
            ("🌱", "Growth Tip",        "Cucumbers grow fast — expect fruit within 50–70 days. Train vines vertically to save space and improve air circulation."),
        ],
        "Quinoa": [
            ("💧", "Watering",          "Quinoa is drought-tolerant. Keep soil moderately dry (40–60%). Overwatering causes root rot."),
            ("🌡️", "Temperature",       "Quinoa thrives between 18–30 °C. It tolerates mild cold better than heat — avoid exceeding 35 °C."),
            ("☀️", "Light",             "Quinoa prefers moderate light. Avoid intense direct light for long periods; it can stress the plant."),
            ("🌀", "Ventilation",       "Adequate airflow helps prevent mold. Use the fan on low or mid speed, especially when humidity is high."),
            ("💦", "Pump / Irrigation", "Water sparingly. Allow the top layer of soil to dry between watering cycles to mimic natural conditions."),
            ("🌱", "Growth Tip",        "Quinoa matures in 90–120 days. It is a hardy crop — minimal intervention needed once established."),
        ],
    }

    st.markdown(f'<div class="section-header">💡 Care Tips for {plant_type}</div>', unsafe_allow_html=True)
    tips = TIPS[plant_type]
    col_c, col_d = st.columns(2)
    for i, (emoji, title, body) in enumerate(tips):
        with (col_c if i % 2 == 0 else col_d):
            st.markdown(f"""
            <div class="tip-card">
                <div style="font-size:1.4rem">{emoji}</div>
                <div style="font-weight:700; color:#1b5e20; margin:0.3rem 0 0.2rem">{title}</div>
                <div style="color:#444; font-size:0.88rem; line-height:1.5">{body}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Smart Greenhouse · Decision Tree Classifier · Built with Streamlit 🌱")