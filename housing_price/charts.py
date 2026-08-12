"""Accessible, presentation-aligned charts for the Streamlit application."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


CREAM = "#F7F2EA"
CHARCOAL = "#263438"
CORAL = "#D45B55"
MUTED = "#6B7476"
GOLD = "#D6A55C"

DISPLAY_NAMES: dict[str, str] = {
    "DaysOnMarket": "Days on market",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "BathroomsTotalInteger": "Bathrooms",
    "LivingArea": "Living area",
    "FireplaceYN": "Fireplace",
    "YearBuilt": "Year built",
    "ParkingTotal": "Parking spaces",
    "BedroomsTotal": "Bedrooms",
    "PoolPrivateYN": "Private pool",
    "LotSizeAcres": "Lot size",
    "Stories": "Stories",
}


def _finish_figure(fig: Figure, axis: plt.Axes, title: str, xlabel: str) -> Figure:
    fig.patch.set_alpha(0)
    axis.set_facecolor(CREAM)
    axis.set_title(title, loc="left", color=CHARCOAL, fontsize=13, fontweight="bold", pad=14)
    axis.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axis.tick_params(axis="both", colors=CHARCOAL, labelsize=9, length=0)
    axis.grid(axis="x", color="#DED7CC", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def importance_figure(frame: pd.DataFrame) -> Figure:
    """Plot normalized XGBoost gain in ascending order."""

    data = frame.copy().tail(10)
    labels = [DISPLAY_NAMES.get(name, name) for name in data["Feature"]]
    values = data["Gain"].to_numpy(dtype=float) * 100
    fig, axis = plt.subplots(figsize=(8, 4.6))
    axis.barh(labels, values, color=CORAL, height=0.62)
    return _finish_figure(fig, axis, "Feature importance by gain", "Share of total gain (%)")


def shap_summary_figure(frame: pd.DataFrame) -> Figure:
    """Plot mean absolute SHAP magnitude for the stored explanation sample."""

    data = frame.sort_values("Mean absolute SHAP", ascending=True).tail(10)
    labels = [DISPLAY_NAMES.get(name, name) for name in data["Feature"]]
    values = data["Mean absolute SHAP"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(8, 4.6))
    axis.barh(labels, values, color=GOLD, height=0.62)
    return _finish_figure(
        fig,
        axis,
        "Average SHAP magnitude",
        "Mean absolute contribution to predicted log price",
    )


def local_contribution_figure(features: Sequence[str], values: Sequence[float]) -> Figure:
    """Plot the largest local SHAP contributions in log-price space."""

    table = pd.DataFrame({"Feature": features, "Contribution": np.asarray(values, dtype=float)})
    table["Magnitude"] = table["Contribution"].abs()
    table = table.nlargest(8, "Magnitude").sort_values("Contribution")
    labels = [DISPLAY_NAMES.get(name, name) for name in table["Feature"]]
    contributions = table["Contribution"].to_numpy()
    colors = np.where(contributions >= 0, CORAL, CHARCOAL)
    fig, axis = plt.subplots(figsize=(8, 4.2))
    axis.barh(labels, contributions, color=colors, height=0.62)
    axis.axvline(0, color=MUTED, linewidth=0.8)
    return _finish_figure(
        fig,
        axis,
        "Largest contributions for this estimate",
        "SHAP contribution to predicted log price",
    )
