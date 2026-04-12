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
    h3 { color: #2e7d32; }
    .stSelectbox label { font-weight: 600; color: #2e7d32; font-size: 0.95rem; }
    .result-card {
        background: white;
        border-left: 5px solid #2e7d32;
        border-radius: 12px;
        padding: 1.4rem 2rem;
        margin-top: 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        text-align: center;
    }
    .plant-name {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1b5e20;
        margin: 0.2rem 0;
    }
    .plant-emoji { font-size: 3rem; }
    .confidence-label { color: #555; font-size: 0.95rem; margin-top: 0.5rem; }
    .section-header {
        background: #e8f5e9;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        color: #2e7d32;
        margin: 1.2rem 0 0.6rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Train model directly from CSV (no pickle needed) ──────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("a.csv")
    df["Heater_enc"] = (df["Heater"] == "on").astype(int)
    df["Pump_enc"]   = (df["Pump"]   == "on").astype(int)
    df["Light_enc"]  = (df["Light"]  == "on").astype(int)
    df["Fan_enc"]    = (df["Fan"]    == "on").astype(int)
    features = ["Temperature", "Soil_Moisture", "LDR", "PWM",
                "Heater_enc", "Pump_enc", "Light_enc", "Fan_enc"]
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(df[features], df["Plant_Type"])
    return clf

clf = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌿 Smart Greenhouse")
st.markdown("**Predict the plant type** based on sensor readings and actuator states using a **Decision Tree** model.")
st.divider()

# ── Dropdown option maps ──────────────────────────────────────────────────────
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

PWM_RANGES = {
    "Low Speed  (~120)":  120.0,
    "Mid Speed  (~180)":  180.0,
    "High Speed (~220)":  220.0,
}

# ── Sensor inputs ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌡️ Sensor Readings</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    temp_label = st.selectbox("🌡️ Temperature",      list(TEMP_RANGES.keys()))
    soil_label = st.selectbox("💧 Soil Moisture",     list(SOIL_RANGES.keys()))
with col2:
    ldr_label  = st.selectbox("☀️ Light Level (LDR)", list(LDR_RANGES.keys()))
    pwm_label  = st.selectbox("⚡ Fan Speed (PWM)",    list(PWM_RANGES.keys()))

# ── Actuator inputs ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔌 Actuator States</div>', unsafe_allow_html=True)

col3, col4, col5, col6 = st.columns(4)
with col3:
    heater = st.selectbox("🔥 Heater", ["off", "on"])
with col4:
    pump   = st.selectbox("💦 Pump",   ["off", "on"])
with col5:
    light  = st.selectbox("💡 Light",  ["off", "on"])
with col6:
    fan    = st.selectbox("🌀 Fan",    ["on",  "off"])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Plant Type", use_container_width=True, type="primary"):

    X = np.array([[
        TEMP_RANGES[temp_label],
        SOIL_RANGES[soil_label],
        LDR_RANGES[ldr_label],
        PWM_RANGES[pwm_label],
        1 if heater == "on" else 0,
        1 if pump   == "on" else 0,
        1 if light  == "on" else 0,
        1 if fan    == "on" else 0,
    ]])

    prediction = clf.predict(X)[0]
    proba      = clf.predict_proba(X)[0]
    confidence = max(proba) * 100

    plant_icons = {"Cucumber": "🥒", "Quinoa": "🌾"}
    icon = plant_icons.get(prediction, "🌱")

    st.markdown(f"""
    <div class="result-card">
        <div class="plant-emoji">{icon}</div>
        <div style="color:#555; font-size:0.9rem; margin-top:0.5rem;">Predicted Plant Type</div>
        <div class="plant-name">{prediction}</div>
        <div class="confidence-label">Confidence: <strong>{confidence:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Plant":           clf.classes_,
        "Probability (%)": [f"{p*100:.1f}%" for p in proba],
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # ── Care Tips ─────────────────────────────────────────────────────────────
    TIPS = {
        "Cucumber": [
            ("💧", "Watering",           "Cucumbers love moisture. Keep soil consistently moist (65–80%). Avoid letting it dry out between waterings."),
            ("🌡️", "Temperature",        "Ideal range is 22–32 °C. Activate the heater if temperature drops below 22 °C to protect growth."),
            ("☀️", "Light",              "Cucumbers need 8–10 hours of light daily. Turn on grow lights when natural light (LDR) is low."),
            ("🌀", "Ventilation",        "Good airflow prevents fungal diseases. Keep the fan running, especially in warm and humid conditions."),
            ("💦", "Pump / Irrigation",  "Use the pump regularly to maintain soil moisture. Drip irrigation is highly recommended for cucumbers."),
            ("🌱", "Growth Tip",         "Cucumbers grow fast — expect fruit within 50–70 days. Train vines vertically to save space and improve air circulation."),
        ],
        "Quinoa": [
            ("💧", "Watering",           "Quinoa is drought-tolerant. Keep soil moderately dry (40–60%). Overwatering causes root rot."),
            ("🌡️", "Temperature",        "Quinoa thrives between 18–30 °C. It tolerates mild cold better than heat — avoid exceeding 35 °C."),
            ("☀️", "Light",              "Quinoa prefers moderate light. Avoid intense direct light for long periods; it can stress the plant."),
            ("🌀", "Ventilation",        "Adequate airflow helps prevent mold. Use the fan on low or mid speed, especially when humidity is high."),
            ("💦", "Pump / Irrigation",  "Water sparingly. Allow the top layer of soil to dry between watering cycles to mimic natural conditions."),
            ("🌱", "Growth Tip",         "Quinoa matures in 90–120 days. It is a hardy crop — minimal intervention needed once established."),
        ],
    }

    st.markdown('<div class="section-header">💡 Care Tips for ' + prediction + '</div>', unsafe_allow_html=True)

    tips = TIPS[prediction]
    col_a, col_b = st.columns(2)
    for i, (emoji, title, body) in enumerate(tips):
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:1rem 1.2rem;
                        margin-bottom:0.8rem; box-shadow:0 2px 8px rgba(0,0,0,0.07);
                        border-top: 3px solid #2e7d32;">
                <div style="font-size:1.4rem">{emoji}</div>
                <div style="font-weight:700; color:#1b5e20; margin:0.3rem 0 0.2rem">{title}</div>
                <div style="color:#444; font-size:0.88rem; line-height:1.5">{body}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Smart Greenhouse · Decision Tree Classifier · Built with Streamlit 🌱")