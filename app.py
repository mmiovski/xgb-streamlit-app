"""Streamlit interface for the California home price XGBoost model."""

from __future__ import annotations

from datetime import date
import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from housing_price.charts import (
    DISPLAY_NAMES,
    importance_figure,
    local_contribution_figure,
    shap_summary_figure,
)
from housing_price.contract import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    property_inputs_from_mapping,
    validate_inputs,
)
from housing_price.explanations import (
    gain_importance,
    global_shap_values,
    load_background,
    local_shap_values,
    mean_absolute_shap,
)
from housing_price.inference import (
    ModelLoadError,
    PredictionError,
    build_feature_frame,
    format_currency,
    load_model_bundle,
    predict_price,
)
from housing_price.styles import APP_CSS, hero, metric_grid


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_nolist.ubj"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
BACKGROUND_PATH = ARTIFACT_DIR / "shap_background.csv"


st.set_page_config(
    page_title="California Home Price Model",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the verified model...")
def get_model_bundle():
    return load_model_bundle(MODEL_PATH, METADATA_PATH)


@st.cache_data(show_spinner=False)
def get_background() -> pd.DataFrame:
    return load_background(BACKGROUND_PATH)


@st.cache_data(show_spinner="Calculating SHAP summaries...")
def get_global_shap_summary() -> pd.DataFrame:
    background = get_background()
    values = global_shap_values(get_model_bundle(), background)
    return mean_absolute_shap(background, values)


def section_heading(title: str, copy: str) -> None:
    st.markdown(
        f'<h2 class="section-heading">{html.escape(title)}</h2>'
        f'<p class="section-copy">{html.escape(copy)}</p>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <p class="sidebar-kicker">Portfolio project</p>
          <p class="sidebar-title">California Home Price Model</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Page",
        ("Estimate", "Model and methodology"),
        label_visibility="collapsed",
    )
    st.sidebar.markdown(
        """
        <div class="sidebar-note">
          Built from 2025 California single-family-home sales. The deployed model does not use list price.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def render_performance_grid(metadata: dict) -> None:
    metrics = metadata["evaluation"]["metrics"]
    st.markdown(
        metric_grid(
            [
                (f'{metrics["r2"]:.2f}', "Held-out R²"),
                (f'{metrics["mape_percent"]:.2f}%', "Held-out MAPE"),
                (f'{metrics["mdape_percent"]:.2f}%', "Held-out MdAPE"),
                ("Sep-Oct 2025", "Evaluation period"),
            ]
        ),
        unsafe_allow_html=True,
    )


def render_estimate_page(bundle) -> None:
    st.markdown(
        hero(
            "Estimate a California home sale price",
            "Enter 12 property characteristics to generate a sale-price estimate. "
            "The XGBoost model is list-unaware, so it can estimate homes without a listed price.",
            "List-unaware XGBoost model",
        ),
        unsafe_allow_html=True,
    )
    render_performance_grid(bundle.metadata)
    st.markdown(
        """
        <div class="notice">
          This analytical estimate is for demonstration only. It is not an appraisal, offer, lending decision, or financial recommendation. Accuracy varies by property and price segment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading(
        "Property details",
        "Use decimal-degree coordinates and MLS-style property values. Required ranges are enforced before inference.",
    )
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("Location")
        location_left, location_right = st.columns(2)
        with location_left:
            latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=34.05,
                step=0.01,
                format="%.4f",
                help="California property latitude in decimal degrees.",
            )
        with location_right:
            longitude = st.number_input(
                "Longitude",
                min_value=-124.5,
                max_value=-114.0,
                value=-118.25,
                step=0.01,
                format="%.4f",
                help="California property longitude in decimal degrees. Use a negative value.",
            )

        st.subheader("Home and lot")
        home_one, home_two, home_three = st.columns(3)
        with home_one:
            living_area = st.number_input(
                "Living area (sq ft)",
                min_value=300,
                max_value=10000,
                value=1500,
                step=50,
                help="Finished interior living area.",
            )
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
            )
        with home_two:
            lot_size = st.number_input(
                "Lot size (acres)",
                min_value=0.01,
                max_value=10.0,
                value=0.15,
                step=0.01,
                format="%.2f",
            )
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                help="Whole bathroom count used by the MLS feature.",
            )
        with home_three:
            year_built = st.number_input(
                "Year built",
                min_value=1800,
                max_value=date.today().year,
                value=1990,
                step=1,
            )
            stories = st.number_input(
                "Stories",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
            )

        st.subheader("Listing and amenities")
        detail_one, detail_two, detail_three = st.columns(3)
        with detail_one:
            days_on_market = st.number_input(
                "Days on market",
                min_value=0,
                max_value=365,
                value=30,
                step=1,
            )
        with detail_two:
            parking = st.number_input(
                "Parking spaces",
                min_value=0.0,
                max_value=30.0,
                value=2.0,
                step=0.5,
                format="%.1f",
                help="Total parking capacity reported in the listing.",
            )
        with detail_three:
            fireplace = st.toggle("Fireplace", value=False)
            private_pool = st.toggle("Private pool", value=False)

        submitted = st.form_submit_button("Estimate sale price", use_container_width=True)

    if submitted:
        st.session_state.pop("latest_inputs", None)
        st.session_state.pop("latest_prediction", None)
        raw_values = {
            "DaysOnMarket": days_on_market,
            "Latitude": latitude,
            "Longitude": longitude,
            "BathroomsTotalInteger": bathrooms,
            "LivingArea": living_area,
            "FireplaceYN": fireplace,
            "YearBuilt": year_built,
            "ParkingTotal": parking,
            "BedroomsTotal": bedrooms,
            "PoolPrivateYN": private_pool,
            "LotSizeAcres": lot_size,
            "Stories": stories,
        }
        validation = validate_inputs(raw_values)
        if validation.errors:
            for error in validation.errors:
                st.error(error)
        else:
            for warning in validation.warnings:
                st.warning(warning)
            try:
                inputs = property_inputs_from_mapping(raw_values)
                prediction = predict_price(bundle, inputs)
                st.session_state["latest_inputs"] = inputs
                st.session_state["latest_prediction"] = prediction
            except (ValueError, PredictionError) as exc:
                st.error(str(exc))

    prediction = st.session_state.get("latest_prediction")
    inputs = st.session_state.get("latest_inputs")
    if prediction is not None and inputs is not None:
        st.markdown(
            f"""
            <div class="prediction-card">
              <p class="prediction-label">Estimated sale price</p>
              <p class="prediction-value">{format_currency(prediction.price)}</p>
              <p class="prediction-caption">Generated from the submitted property details. Review the model page for methodology, held-out performance, and feature-level explanations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Review submitted values"):
            summary = pd.DataFrame(
                {
                    "Feature": [DISPLAY_NAMES[name] for name in FEATURE_NAMES],
                    "Value": [inputs.as_dict()[name] for name in FEATURE_NAMES],
                }
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)


def render_methodology_page(bundle) -> None:
    metadata = bundle.metadata
    st.markdown(
        hero(
            "From monthly sales records to an interactive estimate",
            "The project covers exploratory analysis, leakage-aware preprocessing, chronological evaluation, native model packaging, and Streamlit deployment.",
            "Model and methodology",
        ),
        unsafe_allow_html=True,
    )
    render_performance_grid(metadata)

    section_heading(
        "Pipeline",
        "The public notebooks document the analysis and model build. Proprietary MLS source records are intentionally excluded.",
    )
    st.markdown(
        """
        <div class="pipeline">
          <div class="pipeline-step"><span class="pipeline-number">01</span><span class="pipeline-title">Monthly California sales records</span></div>
          <div class="pipeline-step"><span class="pipeline-number">02</span><span class="pipeline-title">Filtering, imputation, and outlier treatment</span></div>
          <div class="pipeline-step"><span class="pipeline-number">03</span><span class="pipeline-title">List-unaware XGBoost training</span></div>
          <div class="pipeline-step"><span class="pipeline-number">04</span><span class="pipeline-title">September-October holdout evaluation</span></div>
          <div class="pipeline-step"><span class="pipeline-number">05</span><span class="pipeline-title">Validated Streamlit inference</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Evaluation design")
        st.write(
            "Training uses January through August 2025. September and October 2025 are held out as a later-time test set. The model predicts the natural logarithm of sale price, and the application converts the result back to US dollars with the exponential function."
        )
        st.write(
            "The deployed variant excludes ListPrice and OriginalListPrice. This broadens its use to properties without an active listing and avoids relying on a near-direct proxy for sale price."
        )
    with right:
        st.subheader("Training frame")
        st.metric("Final training records", f'{metadata["training"]["final_rows"]:,}')
        st.caption("California single-family residences after documented preprocessing.")

    section_heading(
        "Held-out performance",
        "These metrics come from the modeling notebook's September-October evaluation. They are not results from the app's software smoke tests.",
    )
    price_bands = pd.DataFrame(metadata["evaluation"]["price_bands"])
    price_bands.columns = ["Price band", "MAPE (%)", "MdAPE (%)"]
    st.dataframe(price_bands, hide_index=True, use_container_width=True)
    st.caption(
        "R² measures explained variance. MAPE is mean absolute percentage error. MdAPE is median absolute percentage error. Performance varies across homes and price ranges."
    )

    section_heading(
        "Model features",
        "The application preserves the exact names and order embedded in the native model artifact.",
    )
    feature_table = pd.DataFrame(
        [
            {
                "Feature": spec.label,
                "Model column": spec.name,
                "Unit": spec.unit,
                "Type": spec.model_type,
            }
            for spec in FEATURE_SPECS
        ]
    )
    st.dataframe(feature_table, hide_index=True, use_container_width=True)

    section_heading(
        "How the model uses features",
        "Gain summarizes split improvement across the trained trees. It does not show direction and should not be interpreted causally.",
    )
    try:
        gain = gain_importance(bundle)
        figure = importance_figure(gain)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    except Exception:
        st.info("The stored model importance summary is temporarily unavailable.")

    if st.button("Calculate SHAP summaries", use_container_width=False):
        st.session_state["show_shap"] = True

    if st.session_state.get("show_shap", False):
        try:
            summary = get_global_shap_summary()
            figure = shap_summary_figure(summary)
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)
            st.markdown(
                "<p class='small-source'>SHAP magnitudes use the stored 300-row feature-only explanation sample. The original sampling procedure was not recorded, so this chart is descriptive rather than a population estimate.</p>",
                unsafe_allow_html=True,
            )

            inputs = st.session_state.get("latest_inputs")
            prediction = st.session_state.get("latest_prediction")
            if inputs is not None and prediction is not None:
                st.subheader("Latest estimate explanation")
                frame = build_feature_frame(inputs)
                local_values, _ = local_shap_values(bundle, frame)
                local_figure = local_contribution_figure(FEATURE_NAMES, local_values[0])
                st.pyplot(local_figure, use_container_width=True)
                plt.close(local_figure)
                st.caption(
                    f"Contributions are additive in log-price space for the latest {format_currency(prediction.price)} estimate. Positive values raise the estimate relative to the model reference value; negative values lower it."
                )
            else:
                st.info("Submit an estimate on the Estimate page to view a local explanation.")
        except Exception:
            st.info("SHAP explanations are temporarily unavailable. Price inference remains available.")

    section_heading(
        "Limitations",
        "The application is a portfolio demonstration of an offline model, not a production valuation service.",
    )
    st.markdown(
        """
        - The source data covers California single-family sales from 2025 and may not represent later market conditions.

        - The model does not include renovation quality, interior condition, school assignments, views, or current macroeconomic conditions.

        - Validation prevents obvious input errors but does not reproduce every MLS business rule or training-distribution constraint.

        - A point estimate does not express uncertainty and should not be used as an appraisal, lending decision, or financial recommendation.
        """
    )


def main() -> None:
    page = render_sidebar()
    try:
        bundle = get_model_bundle()
    except ModelLoadError:
        st.error("The verified model artifact could not be loaded. Check the repository artifacts and try again.")
        st.stop()

    if page == "Estimate":
        render_estimate_page(bundle)
    else:
        render_methodology_page(bundle)


if __name__ == "__main__":
    main()
