"""Tests for artifact integrity, inference, and explanations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from housing_price.contract import FEATURE_NAMES, PropertyInputs
from housing_price.explanations import (
    gain_importance,
    load_background,
    local_shap_values,
)
from housing_price.inference import (
    ModelLoadError,
    build_feature_frame,
    format_currency,
    load_model_bundle,
    predict_price,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def default_inputs() -> PropertyInputs:
    return PropertyInputs(
        DaysOnMarket=30,
        Latitude=34.05,
        Longitude=-118.25,
        BathroomsTotalInteger=2,
        LivingArea=1500.0,
        FireplaceYN=False,
        YearBuilt=1990,
        ParkingTotal=2.0,
        BedroomsTotal=3,
        PoolPrivateYN=False,
        LotSizeAcres=0.15,
        Stories=1,
    )


def test_artifact_checksum_feature_contract_and_version(model_bundle) -> None:
    metadata = model_bundle.metadata
    assert sha256_file(model_bundle.model_path) == metadata["artifact"]["sha256"]
    assert tuple(model_bundle.model.feature_names or ()) == FEATURE_NAMES
    assert tuple(metadata["feature_names"]) == FEATURE_NAMES
    assert model_bundle.model.num_features() == 12
    assert metadata["artifact"]["xgboost_version"] == "3.1.2"
    assert model_bundle.model.num_boosted_rounds() == 1300


def test_tampered_metadata_checksum_is_rejected(tmp_path) -> None:
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact"]["sha256"] = "0" * 64
    altered_metadata = tmp_path / "model_metadata.json"
    altered_metadata.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="checksum"):
        load_model_bundle(ARTIFACT_DIR / "xgb_nolist.ubj", altered_metadata)


def test_controlled_prediction_is_finite_deterministic_and_in_dollars(model_bundle) -> None:
    first = predict_price(model_bundle, default_inputs())
    second = predict_price(model_bundle, default_inputs())

    assert first == second
    assert first.log_price == pytest.approx(13.555908203125, abs=1e-9)
    assert first.price == pytest.approx(771_358.25, abs=0.1)
    assert math.isfinite(first.price)
    assert format_currency(first.price) == "$771,358"


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_currency_formatter_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        format_currency(value)


def test_background_sample_has_exact_feature_contract_and_checksum(model_bundle) -> None:
    path = ARTIFACT_DIR / "shap_background.csv"
    background = load_background(path)
    metadata = model_bundle.metadata["explanations"]

    assert background.shape == (300, 12)
    assert tuple(background.columns) == FEATURE_NAMES
    assert sha256_file(path) == metadata["sha256"]


def test_deployed_gain_profile_matches_documented_artifact(model_bundle) -> None:
    gain = gain_importance(model_bundle).sort_values("Gain", ascending=False)
    assert gain.iloc[0]["Feature"] == "BathroomsTotalInteger"
    assert gain.iloc[0]["Gain"] * 100 == pytest.approx(29.56, abs=0.01)
    assert gain["Gain"].sum() == pytest.approx(1.0)


def test_local_shap_values_reconstruct_log_prediction(model_bundle) -> None:
    inputs = default_inputs()
    frame = build_feature_frame(inputs)
    prediction = predict_price(model_bundle, inputs)
    values, expected_value = local_shap_values(model_bundle, frame)

    assert expected_value + float(values.sum()) == pytest.approx(
        prediction.log_price,
        abs=1e-4,
    )
