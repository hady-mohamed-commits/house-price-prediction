import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Smart House Price Estimator",
    page_icon="🏠",
    layout="wide"
)

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Hero */

.hero {
    padding: 35px;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #d1d5db;
}

/* Section cards */

.section {
    background: white;
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid #e5e7eb;
}

/* Prediction */

.prediction-card {
    padding: 35px;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f766e, #115e59);
    color: white;
    text-align: center;
    margin-top: 25px;
}

.prediction-card h2 {
    font-size: 20px;
    margin-bottom: 10px;
}

.prediction-card .price {
    font-size: 42px;
    font-weight: 700;
}

/* Button */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 52px;
    font-size: 17px;
    font-weight: 600;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():

    model_data = joblib.load("house_price_model.pkl")

    return model_data["model"], model_data["features"]


model, features = load_model()


# =========================
# Hero Section
# =========================

st.markdown("""
<div class="hero">

<h1>🏠 Smart House Price Estimator</h1>

<p>
AI-powered house price prediction using Machine Learning.
Enter the property details below and get an estimated market price.
</p>

</div>
""", unsafe_allow_html=True)


# =========================
# Main Layout
# =========================

col1, col2 = st.columns([1, 1], gap="large")


# =========================
# Property Details
# =========================

with col1:

    st.markdown("""
    <div class="section">

    <h2>🏡 Property Details</h2>
    <p>Tell us about the property.</p>

    </div>
    """, unsafe_allow_html=True)

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=20,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.25
    )

    sqft_living = st.number_input(
        "Living Area (sqft)",
        min_value=100,
        max_value=20000,
        value=1500
    )

    sqft_lot = st.number_input(
        "Lot Area (sqft)",
        min_value=100,
        max_value=1000000,
        value=5000
    )

    floors = st.number_input(
        "Floors",
        min_value=1.0,
        max_value=4.0,
        value=1.0,
        step=0.5
    )

    sqft_above = st.number_input(
        "Above Ground Area (sqft)",
        min_value=100,
        max_value=20000,
        value=1500
    )

    sqft_basement = st.number_input(
        "Basement Area (sqft)",
        min_value=0,
        max_value=10000,
        value=0
    )


# =========================
# Location & Condition
# =========================

with col2:

    st.markdown("""
    <div class="section">

    <h2>📍 Location & Condition</h2>
    <p>Additional property information.</p>

    </div>
    """, unsafe_allow_html=True)

    waterfront = st.selectbox(
        "Waterfront",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    view = st.selectbox(
        "View Rating",
        [0, 1, 2, 3, 4]
    )

    condition = st.selectbox(
        "Property Condition",
        [1, 2, 3, 4, 5],
        index=2
    )

    yr_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    lat = st.number_input(
        "Latitude",
        value=47.500000,
        format="%.6f"
    )

    long = st.number_input(
        "Longitude",
        value=-122.200000,
        format="%.6f"
    )


# =========================
# Feature Engineering
# =========================

house_age = 2026 - yr_built

total_sqft = sqft_living + sqft_basement

living_lot_ratio = (
    sqft_living / sqft_lot
    if sqft_lot != 0 else 0
)

sqft_per_bedroom = (
    sqft_living / bedrooms
    if bedrooms != 0 else 0
)

bathroom_density = (
    bathrooms / sqft_living
    if sqft_living != 0 else 0
)


# =========================
# Create Input Data
# =========================

input_data = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "sqft_living": [sqft_living],
    "sqft_lot": [sqft_lot],
    "floors": [floors],
    "waterfront": [waterfront],
    "view": [view],
    "condition": [condition],
    "sqft_above": [sqft_above],
    "sqft_basement": [sqft_basement],
    "yr_built": [yr_built],
    "lat": [lat],
    "long": [long],
    "house_age": [house_age],
    "total_sqft": [total_sqft],
    "living_lot_ratio": [living_lot_ratio],
    "sqft_per_bedroom": [sqft_per_bedroom],
    "bathroom_density": [bathroom_density]
})


# Make sure the input matches training features
for feature in features:

    if feature not in input_data.columns:
        input_data[feature] = 0


input_data = input_data[features]


# =========================
# Prediction
# =========================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "✨ Estimate House Price"
)


if predict_button:

    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="prediction-card">

        <h2>Estimated Property Value</h2>

        <div class="price">
            ${prediction:,.0f}
        </div>

        <p>
        Estimated using the trained Gradient Boosting Machine Learning model.
        </p>

    </div>
    """, unsafe_allow_html=True)


# =========================
# Footer
# =========================

st.markdown("""
<br><br>

<div style="text-align:center; color:#6b7280;">

<p>
Built with Python • Scikit-learn • Gradient Boosting • Streamlit
</p>

</div>
""", unsafe_allow_html=True)