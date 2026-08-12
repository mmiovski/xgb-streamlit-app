"""Convert the trusted XGBoost pickle to native UBJSON with parity checks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix, __version__ as xgboost_version


TRUSTED_PICKLE_SHA256 = "875041cd4e0450cc05cfb2147652f9790e3ca73cdc4f503f4f2bbf4444a44f67"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pickle_path = args.pickle.resolve()
    background_path = args.background.resolve()
    output_path = args.output.resolve()

    actual_hash = sha256_file(pickle_path)
    if actual_hash != TRUSTED_PICKLE_SHA256:
        raise RuntimeError("Refusing to deserialize an unrecognized pickle artifact.")

    source_model = joblib.load(pickle_path)
    feature_names = list(source_model.get_booster().feature_names or ())
    if len(feature_names) != 12:
        raise RuntimeError("The trusted model does not expose the expected 12 features.")

    background = pd.read_csv(background_path)
    if list(background.columns) != feature_names:
        raise RuntimeError("Background data does not match the model feature order.")
    background = background.astype(float)

    reference = pd.DataFrame(
        [
            [30, 34.05, -118.25, 2, 1500, 0, 1990, 2, 3, 0, 0.15, 1],
            [0, 32.0, -124.0, 1, 300, 0, 1800, 0, 1, 0, 0.01, 1],
            [365, 42.0, -114.0, 10, 10000, 1, 2025, 30, 10, 1, 10, 5],
        ],
        columns=feature_names,
        dtype=float,
    )
    parity_frame = pd.concat([background, reference], ignore_index=True)
    source_predictions = source_model.predict(parity_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_booster = source_model.get_booster()
    source_booster.save_model(output_path)
    native_model = Booster()
    native_model.load_model(output_path)
    native_predictions = native_model.predict(
        DMatrix(parity_frame, feature_names=feature_names)
    )

    np.testing.assert_allclose(source_predictions, native_predictions, rtol=0.0, atol=1e-7)
    if native_model.feature_names != feature_names:
        raise RuntimeError("Native artifact did not preserve feature names.")

    maximum_delta = float(np.max(np.abs(source_predictions - native_predictions)))
    print(f"xgboost_version={xgboost_version}")
    print(f"source_sha256={actual_hash}")
    print(f"native_sha256={sha256_file(output_path)}")
    print(f"parity_rows={len(parity_frame)}")
    print(f"maximum_log_prediction_delta={maximum_delta:.12g}")
    print("feature_names=" + "|".join(feature_names))


if __name__ == "__main__":
    main()
