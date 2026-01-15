import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="California Home Price Predictor",
    layout="centered"
)

st.title("🏠 California Home Price Predictor")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb.pkl")

model = load_model()

# ----------------------------
# Feature order (MUST MATCH TRAINING)
# ----------------------------
FEATURES = [
    "DaysOnMarket",
    "Latitude",
    "Longitude",
    "BathroomsTotalInteger",
    "LivingArea",
    "FireplaceYN",
    "YearBuilt",
    "ParkingTotal",
    "BedroomsTotal",
    "PoolPrivateYN",
    "LotSizeAcres",
    "Stories",
]

# ----------------------------
# Inference UI
# ----------------------------
st.subheader("Property Information")

# ----------------------------
# Location
# ----------------------------
with st.expander("📍 Location", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        Latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            step=0.0001,
            value=34.05
        )
    with col2:
        Longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            step=0.0001,
            value=-118.25
        )

# ----------------------------
# Home Details
# ----------------------------
with st.expander("🏠 Home Details", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        BedroomsTotal = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            step=1,
            value=3
        )

        BathroomsTotalInteger = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=10.0,
            step=0.5,
            value=2.0
        )

        Stories = st.number_input(
            "Stories",
            min_value=1,
            max_value=5,
            step=1,
            value=1
        )

    with col2:
        LivingArea = st.number_input(
            "Living Area (sqft)",
            min_value=300,
            max_value=10000,
            step=50,
            value=1500
        )

        YearBuilt = st.number_input(
            "Year Built",
            min_value=1850,
            max_value=2025,
            step=1,
            value=1990
        )

# ----------------------------
# Amenities
# ----------------------------
with st.expander("✨ Amenities", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        FireplaceYN = st.checkbox("Fireplace")
    with col2:
        PoolPrivateYN = st.checkbox("Private Pool")

    ParkingTotal = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=10,
        step=1,
        value=2
    )

# ----------------------------
# Market & Lot
# ----------------------------
with st.expander("📊 Market & Lot", expanded=True):
    DaysOnMarket = st.number_input(
        "Days on Market",
        min_value=0,
        max_value=365,
        step=1,
        value=30
    )

    LotSizeAcres = st.number_input(
        "Lot Size (acres)",
        min_value=0.0,
        max_value=10.0,
        step=0.01,
        value=0.15
    )

# ----------------------------
# Assemble input (STRICT MODEL ORDER)
# ----------------------------
input_data = pd.DataFrame(
    [[
        DaysOnMarket,
        Latitude,
        Longitude,
        BathroomsTotalInteger,
        LivingArea,
        float(FireplaceYN),
        YearBuilt,
        ParkingTotal,
        BedroomsTotal,
        float(PoolPrivateYN),
        LotSizeAcres,
        Stories
    ]],
    columns=FEATURES
).astype(float)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Price"):
    log_price = model.predict(input_data)[0]
    price = np.exp(log_price)

    st.metric("Estimated Sale Price", f"${price:,.0f}")
    st.caption("Estimate is for informational purposes only.")