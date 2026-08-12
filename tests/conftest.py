"""Shared fixtures for application and model tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from housing_price.inference import load_model_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@pytest.fixture(scope="session")
def model_bundle():
    return load_model_bundle(
        ARTIFACT_DIR / "xgb_nolist.ubj",
        ARTIFACT_DIR / "model_metadata.json",
    )
