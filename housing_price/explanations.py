"""Native XGBoost explanation helpers kept separate from inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import DMatrix

from housing_price.contract import FEATURE_NAMES
from housing_price.inference import ModelBundle


class ExplanationError(RuntimeError):
    """Raised when stored explanation resources do not match the model."""


def load_background(path: Path) -> pd.DataFrame:
    try:
        background = pd.read_csv(Path(path).resolve())
    except Exception as exc:
        raise ExplanationError("The stored SHAP background sample could not be loaded.") from exc
    if tuple(background.columns) != FEATURE_NAMES:
        raise ExplanationError("SHAP background columns do not match the model feature contract.")
    if background.empty or background.isna().any().any():
        raise ExplanationError("SHAP background data must be non-empty and complete.")
    return background.astype(float)


def _native_shap_values(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    try:
        matrix = DMatrix(frame, feature_names=list(FEATURE_NAMES))
        contributions = np.asarray(
            bundle.model.predict(matrix, pred_contribs=True),
            dtype=float,
        )
    except Exception as exc:
        raise ExplanationError("XGBoost SHAP contributions could not be calculated.") from exc

    expected_shape = (len(frame), len(FEATURE_NAMES) + 1)
    if contributions.shape != expected_shape or not np.isfinite(contributions).all():
        raise ExplanationError("XGBoost SHAP contributions have an unexpected shape or value.")
    return contributions[:, :-1], contributions[:, -1]


def global_shap_values(bundle: ModelBundle, background: pd.DataFrame) -> np.ndarray:
    values, _ = _native_shap_values(bundle, background)
    if values.shape != background.shape or not np.isfinite(values).all():
        raise ExplanationError("Global SHAP values have an unexpected shape or value.")
    return values


def local_shap_values(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    values, biases = _native_shap_values(bundle, frame)
    if values.shape != frame.shape or not np.isfinite(values).all():
        raise ExplanationError("Local SHAP values have an unexpected shape or value.")
    if len(biases) != 1:
        raise ExplanationError("Local SHAP reference value has an unexpected shape.")
    return values, float(biases[0])


def gain_importance(bundle: ModelBundle) -> pd.DataFrame:
    raw = bundle.model.get_score(importance_type="gain")
    values = np.asarray([float(raw.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)
    total = float(values.sum())
    if total <= 0 or not np.isfinite(values).all():
        raise ExplanationError("XGBoost gain importance is unavailable.")
    return (
        pd.DataFrame({"Feature": FEATURE_NAMES, "Gain": values / total})
        .sort_values("Gain", ascending=True)
        .reset_index(drop=True)
    )


def mean_absolute_shap(background: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    means = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"Feature": background.columns, "Mean absolute SHAP": means})
        .sort_values("Mean absolute SHAP", ascending=False)
        .reset_index(drop=True)
    )
