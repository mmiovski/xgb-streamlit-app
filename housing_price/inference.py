"""Portable model loading and deterministic price inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

from housing_price.contract import FEATURE_NAMES, PropertyInputs


class ModelLoadError(RuntimeError):
    """Raised when the model bundle is absent, inconsistent, or unreadable."""


class PredictionError(RuntimeError):
    """Raised when inference does not return a usable sale-price estimate."""


@dataclass(frozen=True)
class ModelBundle:
    model: Booster
    metadata: dict[str, Any]
    model_path: Path


@dataclass(frozen=True)
class Prediction:
    log_price: float
    price: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_bundle(model_path: Path, metadata_path: Path) -> ModelBundle:
    """Load a native XGBoost model and verify its artifact and feature contract."""

    model_path = Path(model_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    if not model_path.is_file():
        raise ModelLoadError(f"Model artifact is unavailable: {model_path.name}.")
    if not metadata_path.is_file():
        raise ModelLoadError(f"Model metadata is unavailable: {metadata_path.name}.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError("Model metadata could not be read.") from exc

    expected_hash = str(metadata.get("artifact", {}).get("sha256", "")).lower()
    actual_hash = sha256_file(model_path)
    if not expected_hash or actual_hash != expected_hash:
        raise ModelLoadError("Model artifact checksum does not match its metadata.")

    try:
        model = Booster()
        model.load_model(model_path)
    except Exception as exc:
        raise ModelLoadError("The native XGBoost model could not be loaded.") from exc

    actual_names = tuple(model.feature_names or ())
    metadata_names = tuple(metadata.get("feature_names", ()))
    if actual_names != FEATURE_NAMES or metadata_names != FEATURE_NAMES:
        raise ModelLoadError("Model feature names or order do not match the application contract.")
    if model.num_features() != len(FEATURE_NAMES):
        raise ModelLoadError("Model feature count does not match the application contract.")

    return ModelBundle(model=model, metadata=metadata, model_path=model_path)


def build_feature_frame(inputs: PropertyInputs) -> pd.DataFrame:
    frame = pd.DataFrame([inputs.as_model_row()], columns=FEATURE_NAMES, dtype=float)
    if tuple(frame.columns) != FEATURE_NAMES:
        raise PredictionError("Feature construction did not preserve the model column order.")
    return frame


def predict_price(bundle: ModelBundle, inputs: PropertyInputs) -> Prediction:
    """Predict log price and reverse the training notebook's natural-log transform."""

    frame = build_feature_frame(inputs)
    try:
        matrix = DMatrix(frame, feature_names=list(FEATURE_NAMES))
        raw = bundle.model.predict(matrix)
    except Exception as exc:
        raise PredictionError("The model could not calculate a prediction.") from exc

    if np.asarray(raw).size != 1:
        raise PredictionError("The model returned an unexpected prediction shape.")
    log_price = float(np.asarray(raw).reshape(-1)[0])
    if not math.isfinite(log_price):
        raise PredictionError("The model returned a non-finite log-price prediction.")

    try:
        price = math.exp(log_price)
    except OverflowError as exc:
        raise PredictionError("The predicted price exceeded the supported numeric range.") from exc
    if not math.isfinite(price) or price <= 0:
        raise PredictionError("The model returned an invalid sale-price estimate.")
    return Prediction(log_price=log_price, price=price)


def format_currency(value: float) -> str:
    if not math.isfinite(value) or value < 0:
        raise ValueError("Currency value must be finite and non-negative.")
    return f"${value:,.0f}"
