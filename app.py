import streamlit as st
import numpy as np
import pandas as pd
import joblib

# app configuration
st.set_page_config(
    page_title="California Home Price Predictor",
    layout="centered"
)

# title
st.title("California Home Price Predictor")

# load model
@st.cache_resource
def load_model():
    return joblib.load("xgb.pkl")

model = load_model()

# feature order (MUST MATCH TRAINING)
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
# Sidebar routing
# ----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Inference", "Model Information"]
)

# ============================================================
# PAGE: Inference
# ============================================================
if page == "Inference":
    st.subheader("Property Information")

    # Location
    with st.expander("Location", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            Latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=34.05,
                help="Geographic latitude of the property. Restricted to California."
            )
        with col2:
            Longitude = st.number_input(
                "Longitude",
                min_value=-125.0,
                max_value=-114.0,
                value=-118.25,
                help="Geographic longitude of the property. Restricted to California."
            )

    # Home Details
    with st.expander("Home Details", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            BedroomsTotal = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=10,
                step=1,
                value=3,
                help="Total number of bedrooms in the home."
            )

            BathroomsTotalInteger = st.number_input(
                "Bathrooms",
                min_value=0.0,
                max_value=10.0,
                step=0.5,
                value=2.0,
                help="Total number of bathrooms. Half-baths (e.g., 2.5) are allowed."
            )

            Stories = st.number_input(
                "Stories",
                min_value=1,
                max_value=5,
                step=1,
                value=1,
                help="Number of floors in the home."
            )

        with col2:
            LivingArea = st.number_input(
                "Living Area (sqft)",
                min_value=300,
                max_value=10000,
                step=50,
                value=1500,
                help="Finished interior living space measured in square feet."
            )

            LotSizeAcres = st.number_input(
                "Lot Size (Acres)",
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                value=0.15,
                help="Total land area of the property measured in acres."
            )

            YearBuilt = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2025,
                step=1,
                value=1990,
                help="Year the home was originally constructed."
            )

    # Amenities
    with st.expander("Amenities", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            FireplaceYN = st.checkbox(
                "Fireplace",
                help="Indicates whether the home has at least one fireplace."
            )
        with col2:
            PoolPrivateYN = st.checkbox(
                "Private Pool",
                help="Indicates whether the property includes a private swimming pool."
            )

        ParkingTotal = st.number_input(
            "Parking Spaces",
            min_value=0,
            max_value=10,
            step=1,
            value=2,
            help="Total number of off-street parking spaces available."
        )

    # Market
    with st.expander("Market Information", expanded=True):
        DaysOnMarket = st.number_input(
            "Days on Market",
            min_value=0,
            max_value=365,
            step=1,
            value=30,
            help="Number of days the property has been listed before sale."
        )

    # Assemble user inputs (STRICT MODEL ORDER)
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

    # Prediction
    if st.button("Predict Price"):
        log_price = model.predict(input_data)[0]
        price = np.exp(log_price)

        st.metric("Estimated Sale Price", f"${price:,.0f}")
        st.caption("Note: This estimate is for demonstrational purposes only.")

# ============================================================
# PAGE: Model Information (placeholder)
# ============================================================
elif page == "Model Information":
    st.subheader("Model Information")

    st.write(
        "This page will summarize the dataset, modeling approach, and interpretability "
        "tools (feature importance, SHAP), plus limitations and stability checks."
    )

    st.markdown("### Coming next")
    st.markdown("- Data overview (high-level EDA)")
    st.markdown("- Global feature importance")
    st.markdown("- SHAP (global + local explanations)")
    st.markdown("- Stability / robustness and known limitations")
