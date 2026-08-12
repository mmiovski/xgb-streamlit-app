"""Tests for the 12-feature application input contract."""

from __future__ import annotations

import math

import pytest

from housing_price.contract import (
    FEATURE_NAMES,
    property_inputs_from_mapping,
    validate_inputs,
)
from housing_price.inference import build_feature_frame


EXPECTED_FEATURES = (
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
)


@pytest.fixture
def valid_values() -> dict[str, float | int | bool]:
    return {
        "DaysOnMarket": 30,
        "Latitude": 34.05,
        "Longitude": -118.25,
        "BathroomsTotalInteger": 2,
        "LivingArea": 1500,
        "FireplaceYN": False,
        "YearBuilt": 1990,
        "ParkingTotal": 2.0,
        "BedroomsTotal": 3,
        "PoolPrivateYN": False,
        "LotSizeAcres": 0.15,
        "Stories": 1,
    }


def test_feature_names_are_exact_and_ordered() -> None:
    assert FEATURE_NAMES == EXPECTED_FEATURES


def test_valid_values_build_exact_numeric_model_row(valid_values) -> None:
    result = validate_inputs(valid_values, current_year=2026)
    assert result.errors == ()

    inputs = property_inputs_from_mapping(valid_values)
    frame = build_feature_frame(inputs)

    assert tuple(frame.columns) == EXPECTED_FEATURES
    assert frame.shape == (1, 12)
    assert frame.loc[0, "FireplaceYN"] == 0.0
    assert frame.loc[0, "PoolPrivateYN"] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Latitude", 43.0, "Latitude must be no more than"),
        ("Longitude", -113.0, "Longitude must be no more than"),
        ("BathroomsTotalInteger", 2.5, "Bathrooms must be a whole number"),
        ("LivingArea", math.inf, "Living area must be a finite number"),
        ("YearBuilt", 2027, "Year built must be no more than 2026"),
        ("FireplaceYN", 1, "Fireplace must be yes or no"),
    ],
)
def test_invalid_values_are_rejected(valid_values, field, value, message) -> None:
    valid_values[field] = value
    result = validate_inputs(valid_values, current_year=2026)
    assert any(message in error for error in result.errors)


def test_missing_features_are_reported(valid_values) -> None:
    del valid_values["Stories"]
    result = validate_inputs(valid_values)
    assert result.errors == ("Missing required inputs: Stories.",)


def test_living_area_cannot_exceed_lot_area(valid_values) -> None:
    valid_values["LivingArea"] = 1000
    valid_values["LotSizeAcres"] = 0.01
    result = validate_inputs(valid_values)
    assert any("Living area cannot exceed lot area" in error for error in result.errors)


def test_unlikely_california_coordinate_is_warning_not_error(valid_values) -> None:
    valid_values["Latitude"] = 41.8
    valid_values["Longitude"] = -114.2
    result = validate_inputs(valid_values)
    assert result.errors == ()
    assert any("approximate mainland outline" in warning for warning in result.warnings)


def test_outside_observed_range_warns_without_blocking(valid_values) -> None:
    valid_values["DaysOnMarket"] = 300
    result = validate_inputs(valid_values)
    assert result.errors == ()
    assert any("Days on market is outside" in warning for warning in result.warnings)


def test_constructor_refuses_invalid_mapping(valid_values) -> None:
    valid_values["BedroomsTotal"] = 0
    with pytest.raises(ValueError, match="Bedrooms must be at least"):
        property_inputs_from_mapping(valid_values)
