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
# Input UI
# ----------------------------
st.subheader("Property Details")

DaysOnMarket = st.number_input("Days on Market", min_value=0, value=30)
Latitude = st.number_input("Latitude", value=34.05)
Longitude = st.number_input("Longitude", value=-118.25)
BathroomsTotalInteger = st.number_input("Bathrooms", min_value=0.0, value=2.0)
LivingArea = st.number_input("Living Area (sqft)", min_value=0.0, value=1500.0)
FireplaceYN = st.checkbox("Fireplace")
YearBuilt = st.number_input("Year Built", min_value=1800, value=1990)
ParkingTotal = st.number_input("Parking Spaces", min_value=0.0, value=2.0)
BedroomsTotal = st.number_input("Bedrooms", min_value=0.0, value=3.0)
PoolPrivateYN = st.checkbox("Private Pool")
LotSizeAcres = st.number_input("Lot Size (acres)", min_value=0.0, value=0.15)
Stories = st.number_input("Stories", min_value=0.0, value=1.0)

# ----------------------------
# Assemble input (STRICT)
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

    st.success(f"Estimated Sale Price: ${price:,.0f}")
