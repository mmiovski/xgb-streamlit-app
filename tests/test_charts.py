"""Tests for application chart presentation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from housing_price.charts import importance_figure, shap_summary_figure


def test_global_importance_charts_use_the_same_bar_color() -> None:
    gain = pd.DataFrame(
        {
            "Feature": ["LivingArea", "Latitude"],
            "Gain": [0.4, 0.6],
        }
    )
    shap = pd.DataFrame(
        {
            "Feature": ["LivingArea", "Latitude"],
            "Mean absolute SHAP": [0.2, 0.3],
        }
    )

    gain_figure = importance_figure(gain)
    shap_figure = shap_summary_figure(shap)
    try:
        gain_color = gain_figure.axes[0].patches[0].get_facecolor()
        shap_color = shap_figure.axes[0].patches[0].get_facecolor()
        assert np.allclose(gain_color, shap_color)
    finally:
        plt.close(gain_figure)
        plt.close(shap_figure)
