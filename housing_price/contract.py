"""Input contract and validation for the deployed 12-feature model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    unit: str
    model_type: str
    default: float | int | bool
    minimum: float | int | None
    maximum: float | int | None
    step: float | int | None
    help_text: str
    observed_minimum: float | int | None = None
    observed_maximum: float | int | None = None


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "DaysOnMarket",
        "Days on market",
        "days",
        "integer",
        30,
        0,
        365,
        1,
        "Number of days the property was listed before sale.",
        0,
        263,
    ),
    FeatureSpec(
        "Latitude",
        "Latitude",
        "decimal degrees",
        "float",
        34.05,
        32.0,
        42.0,
        0.01,
        "Property latitude in decimal degrees. Coordinates must fall within California bounds.",
        32.12,
        41.89,
    ),
    FeatureSpec(
        "Longitude",
        "Longitude",
        "decimal degrees",
        "float",
        -118.25,
        -124.5,
        -114.0,
        0.01,
        "Property longitude in decimal degrees. California longitudes are negative.",
        -123.82,
        -114.35,
    ),
    FeatureSpec(
        "BathroomsTotalInteger",
        "Bathrooms",
        "whole bathrooms",
        "integer",
        2,
        1,
        10,
        1,
        "MLS integer bathroom count used during model training.",
        1,
        6,
    ),
    FeatureSpec(
        "LivingArea",
        "Living area",
        "square feet",
        "float",
        1500,
        300,
        10000,
        50,
        "Finished interior living area in square feet.",
        675,
        6493,
    ),
    FeatureSpec(
        "FireplaceYN",
        "Fireplace",
        "yes/no",
        "boolean",
        False,
        None,
        None,
        None,
        "Whether the property has at least one fireplace.",
    ),
    FeatureSpec(
        "YearBuilt",
        "Year built",
        "year",
        "integer",
        1990,
        1800,
        None,
        1,
        "Year the residence was built.",
        1801,
        2026,
    ),
    FeatureSpec(
        "ParkingTotal",
        "Parking spaces",
        "spaces",
        "float",
        2.0,
        0.0,
        30.0,
        0.5,
        "Total parking capacity reported by the listing, including fractional MLS values.",
        0.0,
        30.0,
    ),
    FeatureSpec(
        "BedroomsTotal",
        "Bedrooms",
        "rooms",
        "integer",
        3,
        1,
        10,
        1,
        "Total number of bedrooms.",
        2,
        6,
    ),
    FeatureSpec(
        "PoolPrivateYN",
        "Private pool",
        "yes/no",
        "boolean",
        False,
        None,
        None,
        None,
        "Whether a private pool is included with the property.",
    ),
    FeatureSpec(
        "LotSizeAcres",
        "Lot size",
        "acres",
        "float",
        0.15,
        0.01,
        10.0,
        0.01,
        "Total lot area in acres.",
        0.05,
        5.6,
    ),
    FeatureSpec(
        "Stories",
        "Stories",
        "floors",
        "integer",
        1,
        1,
        5,
        1,
        "Number of floors in the residence.",
        1,
        3,
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}

# A deliberately approximate mainland outline. The hard model contract uses the
# documented bounding box; this outline produces a warning for likely out-of-state
# coordinates without rejecting islands or unusual coastal records.
CALIFORNIA_MAINLAND_POLYGON: tuple[tuple[float, float], ...] = (
    (-124.41, 42.00),
    (-120.00, 42.00),
    (-120.00, 39.00),
    (-119.30, 38.00),
    (-118.40, 37.00),
    (-117.50, 36.00),
    (-116.60, 35.00),
    (-114.63, 35.00),
    (-114.55, 34.30),
    (-114.14, 34.26),
    (-114.46, 33.60),
    (-114.72, 32.72),
    (-117.13, 32.54),
    (-117.50, 33.00),
    (-117.75, 33.45),
    (-118.35, 33.75),
    (-119.10, 34.20),
    (-120.50, 34.45),
    (-121.00, 35.00),
    (-121.90, 36.50),
    (-122.50, 38.00),
    (-123.00, 38.70),
    (-123.75, 40.00),
    (-124.30, 41.00),
)


@dataclass(frozen=True)
class PropertyInputs:
    DaysOnMarket: int
    Latitude: float
    Longitude: float
    BathroomsTotalInteger: int
    LivingArea: float
    FireplaceYN: bool
    YearBuilt: int
    ParkingTotal: float
    BedroomsTotal: int
    PoolPrivateYN: bool
    LotSizeAcres: float
    Stories: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)

    def as_model_row(self) -> list[float]:
        values = self.as_dict()
        return [
            float(values[name]) if not isinstance(values[name], bool) else float(values[name])
            for name in FEATURE_NAMES
        ]


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_integer_value(value: Any) -> bool:
    return _is_finite_number(value) and float(value).is_integer()


def _point_in_polygon(longitude: float, latitude: float) -> bool:
    inside = False
    points = CALIFORNIA_MAINLAND_POLYGON
    previous = points[-1]
    for current in points:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > latitude) != (y2 > latitude)
        if crosses:
            boundary_x = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
            if longitude < boundary_x:
                inside = not inside
        previous = current
    return inside


def validate_inputs(values: Mapping[str, Any], *, current_year: int | None = None) -> ValidationResult:
    """Validate raw values against the documented inference contract."""

    errors: list[str] = []
    warnings: list[str] = []
    current_year = current_year or date.today().year

    missing = [name for name in FEATURE_NAMES if name not in values]
    if missing:
        errors.append(f"Missing required inputs: {', '.join(missing)}.")
        return ValidationResult(tuple(errors), tuple(warnings))

    for spec in FEATURE_SPECS:
        value = values[spec.name]
        if spec.model_type == "boolean":
            if not isinstance(value, bool):
                errors.append(f"{spec.label} must be yes or no.")
            continue

        if not _is_finite_number(value):
            errors.append(f"{spec.label} must be a finite number.")
            continue

        numeric = float(value)
        if spec.model_type == "integer" and not _is_integer_value(value):
            errors.append(f"{spec.label} must be a whole number.")
        if spec.minimum is not None and numeric < float(spec.minimum):
            errors.append(f"{spec.label} must be at least {spec.minimum} {spec.unit}.")
        maximum = current_year if spec.name == "YearBuilt" else spec.maximum
        if maximum is not None and numeric > float(maximum):
            errors.append(f"{spec.label} must be no more than {maximum} {spec.unit}.")

    if errors:
        return ValidationResult(tuple(dict.fromkeys(errors)), tuple(warnings))

    latitude = float(values["Latitude"])
    longitude = float(values["Longitude"])
    if not _point_in_polygon(longitude, latitude):
        warnings.append(
            "The coordinates are inside the model's California bounding box but do not "
            "appear to fall on the approximate mainland outline. Verify the location."
        )

    lot_square_feet = float(values["LotSizeAcres"]) * 43_560
    if float(values["LivingArea"]) > lot_square_feet:
        errors.append(
            "Living area cannot exceed lot area for this model. The training pipeline excluded "
            "records that violated this relationship."
        )

    for spec in FEATURE_SPECS:
        if spec.model_type == "boolean":
            continue
        numeric = float(values[spec.name])
        outside_observed = (
            spec.observed_minimum is not None and numeric < float(spec.observed_minimum)
        ) or (
            spec.observed_maximum is not None and numeric > float(spec.observed_maximum)
        )
        if outside_observed:
            warnings.append(
                f"{spec.label} is outside the trimmed training range documented in the EDA "
                f"({spec.observed_minimum} to {spec.observed_maximum} {spec.unit})."
            )

    return ValidationResult(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)))


def property_inputs_from_mapping(values: Mapping[str, Any]) -> PropertyInputs:
    """Construct typed inputs after validation succeeds."""

    result = validate_inputs(values)
    if not result.is_valid:
        raise ValueError(" ".join(result.errors))
    return PropertyInputs(
        DaysOnMarket=int(values["DaysOnMarket"]),
        Latitude=float(values["Latitude"]),
        Longitude=float(values["Longitude"]),
        BathroomsTotalInteger=int(values["BathroomsTotalInteger"]),
        LivingArea=float(values["LivingArea"]),
        FireplaceYN=bool(values["FireplaceYN"]),
        YearBuilt=int(values["YearBuilt"]),
        ParkingTotal=float(values["ParkingTotal"]),
        BedroomsTotal=int(values["BedroomsTotal"]),
        PoolPrivateYN=bool(values["PoolPrivateYN"]),
        LotSizeAcres=float(values["LotSizeAcres"]),
        Stories=int(values["Stories"]),
    )
