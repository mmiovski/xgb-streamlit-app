import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

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

@st.cache_data
def load_shap_background():
    return pd.read_csv("shap_background.csv")

@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)

explainer = load_shap_explainer()

@st.cache_data
def compute_global_shap(background):
    return explainer.shap_values(background)

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
    "Stories"
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

    st.session_state["last_input"] = input_data

    # Prediction
    if st.button("Predict Price"):
        log_price = model.predict(input_data)[0]
        price = np.exp(log_price)

        st.metric("Estimated Sale Price", f"${price:,.0f}")
        st.caption("Note: This estimate is for demonstrational purposes only.")

# ============================================================
# PAGE: Model Information
# ============================================================
elif page == "Model Information":
    st.subheader("Model Information")

    st.markdown(
        """
        This application uses an **XGBoost regression model** trained on California MLS data.
        The target variable is **log-transformed sale price**, which stabilizes variance and
        improves predictive performance. Predictions shown in the app are converted back to
        dollar terms for interpretability.
        """
    )

    # ----------------------------
    # Global Feature Importance
    # ----------------------------
    st.markdown("### Global Feature Importance")

    importances = model.feature_importances_

    fi_df = (
        pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importances
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(fi_df, use_container_width=True)

    st.bar_chart(
        fi_df.set_index("Feature")["Importance"]
    )

    st.caption(
        "Feature importance reflects how much each feature contributes to reducing prediction "
        "error across all trees in the model (higher = more influence)."
    )

    # ----------------------------
    # SHAP: Global Explanation
    # ----------------------------
    st.markdown("### SHAP: Global Feature Impact")

    st.markdown(
        """
        This plot summarizes **global feature effects** across a representative
        sample of homes from the training distribution.

        Each point represents one home. The horizontal axis shows how a feature
        impacts the model’s **log-price prediction**, while color indicates whether
        the feature value is high (red) or low (blue).
        """
    )

    background = load_shap_background()
    global_shap_values = compute_global_shap(background)

    fig, ax = plt.subplots()
    shap.summary_plot(
        global_shap_values,
        background,
        feature_names=FEATURES,
        show=False
    )

    st.pyplot(fig)

    st.caption(
        "Global SHAP values are shown in log-price space. Features are ordered by "
        "overall importance across the dataset."
    )

    # ----------------------------
    # SHAP: Local Explanation
    # ----------------------------
    st.markdown("### SHAP: Local Explanation (Example)")

    st.markdown(
        """
        SHAP values explain how each feature contributes to a prediction.
        The model was trained on **log-transformed sale prices**, so all SHAP
        values are shown in **log(price) space**.
        """
    )

    if "last_input" not in st.session_state:
        st.warning("Run a prediction first to see a SHAP explanation.")
        st.stop()

    example_input = st.session_state["last_input"]

    baseline_log = explainer.expected_value
    baseline_price = np.exp(baseline_log)

    st.info(
        f"""
        **Understanding the baseline**

        The baseline value **E[f(X)] = {baseline_log:.3f}** represents the model’s
        average predicted **log(price)** across the training data.

        Converting this to dollars:

        **Baseline price ≈ ${baseline_price:,.0f}**

        SHAP values below explain how each feature moves the prediction
        *away from this baseline* to reach the final estimate.
        """
    )

    shap_values = explainer.shap_values(example_input)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=baseline_log,
            data=example_input.iloc[0],
            feature_names=FEATURES
        ),
        show=False
    )

    st.pyplot(fig)

    final_log = model.predict(example_input)[0]
    final_price = np.exp(final_log)

    st.success(
        f"""
        **Final prediction**

        log(price) = {final_log:.3f}  
        Estimated sale price ≈ **${final_price:,.0f}**
        """
    )

    st.caption(
        "Positive SHAP values increase the predicted price; negative values decrease it. "
        "All values are additive in log-price space."
    )






