# Model Card: California Single-Family Home Price XGBoost

## Summary

This regression model estimates the sale price of a California single-family residence from 12 property and location features. It is list-unaware: neither current nor original list price is used.

The model is served as a native XGBoost UBJSON artifact. The application predicts natural-log sale price and applies the exponential function to return US dollars.

## Intended use

- Estimate sale prices for plausible California single-family-home inputs
- Inspect feature influence through gain importance and SHAP contributions
- Document the temporal evaluation and inference workflow

## Out-of-scope use

- Appraisals, offers, lending, underwriting, taxation, or financial recommendations
- Properties outside California
- Property types other than single-family residences
- High-stakes decisions about an individual home or person
- Claims about current market value after material market drift

## Data and split

- Population: recorded California residential single-family sales
- Training period: January-August 2025
- Temporal test period: September-October 2025
- Final training rows: 78,162
- Raw data availability: proprietary MLS records are excluded

The chronological split provides a more realistic later-time test than a random row split. Preprocessing thresholds are learned from training data only and then applied to the test months. September appeared in predecessor tuning diagnostics before October became available, so the combined period is not described as a strictly untouched evaluation set.

## Hyperparameter selection

Archived XGBoost experiments used exhaustive five-fold grid searches in two stages. The broad grid evaluated 125 depth, learning-rate, and estimator-count combinations; the refined grid evaluated 48. Each grid was run against mean and median percentage-error criteria on the natural-log target, for 1,730 total cross-validation fits.

The refined mean-error search selected `max_depth=7`, `learning_rate=0.05`, and `n_estimators=1300`, with `subsample=0.8` and `colsample_bytree=0.8` fixed throughout. The archived search used an earlier 18-feature list-unaware frame. The final model retained the selected settings after reducing the runtime contract to 12 features. A full retune on the reduced feature set has not been claimed. The extracted code and archived selections are documented in [`notebooks/xgboost_tuning.ipynb`](notebooks/xgboost_tuning.ipynb).

## Performance

| Metric | Temporal-test result |
| --- | ---: |
| R² | 0.90 |
| MAPE | 11.19% |
| MdAPE | 7.75% |

The executed modeling notebook reproduced these results on CPU with the pinned environment. Error differs by price band, and a single aggregate metric should not be interpreted as uniform accuracy.

## Features

| Order | Model column | Meaning | Runtime encoding |
| ---: | --- | --- | --- |
| 1 | `DaysOnMarket` | Days listed before sale | Integer |
| 2 | `Latitude` | Decimal-degree latitude | Float |
| 3 | `Longitude` | Decimal-degree longitude | Float |
| 4 | `BathroomsTotalInteger` | Whole MLS bathroom count | Integer-valued float |
| 5 | `LivingArea` | Finished interior square feet | Float |
| 6 | `FireplaceYN` | Fireplace indicator | 0 or 1 |
| 7 | `YearBuilt` | Construction year | Integer-valued float |
| 8 | `ParkingTotal` | Reported parking capacity | Float |
| 9 | `BedroomsTotal` | Bedroom count | Integer-valued float |
| 10 | `PoolPrivateYN` | Private-pool indicator | 0 or 1 |
| 11 | `LotSizeAcres` | Lot size in acres | Float |
| 12 | `Stories` | Floor count | Integer-valued float |

No separate preprocessing estimator is embedded in the artifact. The application constructs one ordered numeric frame and the booster predicts directly from it.

## Validation

Runtime checks enforce all required fields, finite numeric inputs, feature-specific types, broad model-safe ranges, California coordinate bounds, and a living-area-to-lot-area consistency rule. Plausible but out-of-distribution values produce warnings instead of being falsely treated as impossible.

Validation does not reproduce every rule or empirical dependency in the proprietary training records.

## Explainability

The app provides normalized XGBoost gain and optional SHAP explanations calculated through XGBoost's native TreeSHAP contribution mode. Gain measures aggregate split improvement and is not directional or causal. SHAP contributions are additive in log-price space and should not be interpreted as causal effects.

The 300-row feature-only SHAP sample was supplied with the original application. Its exact sampling procedure was not recorded, so global SHAP magnitudes are descriptive rather than population estimates.

## Artifact integrity

- Format: XGBoost UBJSON
- XGBoost artifact version: 3.1.2
- Trees: 1,300
- Native SHA-256: `04189265593b229ef5c650390940c3869bbf731f06f920bbe8911b7a41680114`
- Trusted source pickle SHA-256: `875041cd4e0450cc05cfb2147652f9790e3ca73cdc4f503f4f2bbf4444a44f67`
- Conversion parity: exact across 303 feature rows at the recorded tolerance

The application verifies the native checksum and feature contract before inference. Pickle is not used at runtime.

## Limitations and monitoring needs

- The 2025 training window does not represent all market regimes.
- Geographic coverage and feature distributions may be uneven.
- Important value drivers such as property condition, recent upgrades, views, school assignments, and macroeconomic conditions are omitted.
- The point estimate has no calibrated prediction interval.
- Later deployment should monitor input drift, error by geography and price band, and temporal performance before retraining.

## Project ownership

Marko Miovski built the list-unaware model and Streamlit application during an IDXExchange data science internship. The five-person team also developed other models. IDXExchange's later website integration was performed separately by others.
