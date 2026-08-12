# California Home Price Prediction

An end-to-end machine learning project that predicts sale prices for California single-family residences without using list price. The repository includes the exploratory analysis, preprocessing and model-building notebooks, a verified XGBoost artifact, focused tests, and a deployable Streamlit application.

**Live application:** [xgb-cali-houseprices.streamlit.app](https://xgb-cali-houseprices.streamlit.app/)

The primary model achieved **0.90 R²**, **11.19% MAPE**, and **7.75% MdAPE** on a later-time September-October 2025 holdout. The final training frame contained **78,162 records** and **12 property features**.

## Project scope

This work was completed during a remote Data Science Internship at IDXExchange from September 9 through December 9, 2025. The five-person team generally developed models individually and reviewed progress in weekly meetings.

My contribution covered California housing-data analysis, preprocessing, feature engineering, XGBoost modeling, validation, evaluation, model comparison, and the Streamlit application. The portfolio application serves my list-unaware XGBoost model. IDXExchange's later website integration was performed by others.

## Why list-unaware

The deployed model excludes `ListPrice` and `OriginalListPrice`. Those fields closely track final sale price but are unavailable for homes that are not actively listed. Removing them creates a harder and more useful task: estimating a sale price from property and location characteristics alone.

A list-aware comparison appears in the modeling notebook as project context. It is not the deployed portfolio model.

## Results

| Model | Held-out R² | MAPE | MdAPE | Portfolio role |
| --- | ---: | ---: | ---: | --- |
| List-unaware XGBoost | 0.90 | 11.19% | 7.75% | Primary deployed model |
| List-aware XGBoost | 0.99 | 3.30% | 2.10% | Context comparison only |

Training uses January through August 2025. September and October 2025 are held out as a later-time evaluation period. The metrics above were reproduced by the executed CPU publication notebook using the pinned environment. Software tests verify application behavior and do not constitute a second performance evaluation.

The list-unaware model's error varies across price segments:

| Held-out sale-price band | MAPE | MdAPE |
| --- | ---: | ---: |
| Below $500K | 12.96% | 7.40% |
| $500K to $1M | 9.64% | 6.54% |
| $1M to $2M | 11.85% | 9.07% |
| $2M to $5M | 13.89% | 11.52% |

These band results come from the executed CPU publication notebook. No held-out homes remained above $5 million after the documented training-derived filters.

## Modeling pipeline

1. Load January through October 2025 monthly sales records.
2. Use January-August for training and September-October for held-out evaluation.
3. Filter to California residential single-family sales.
4. Impute missing values by sampling observed training values with a fixed seed.
5. Remove duplicate listing keys, keeping the most recent close date.
6. Apply universal property-value checks.
7. Learn 0.5th and 99.5th percentile bounds from training data only and apply them to both periods.
8. Train fixed-configuration list-aware and list-unaware XGBoost regressors on `ln(ClosePrice)`.
9. Convert predictions back to US dollars with `exp` and evaluate on the held-out months.
10. Package the list-unaware booster in native XGBoost UBJSON format for Streamlit inference.

The complete analysis is in [notebooks/eda.ipynb](notebooks/eda.ipynb) and [notebooks/modeling.ipynb](notebooks/modeling.ipynb). Both publication copies were executed from start to finish with zero retained errors.

## Model features

The application preserves the exact feature names and order embedded in the artifact:

1. `DaysOnMarket`
2. `Latitude`
3. `Longitude`
4. `BathroomsTotalInteger`
5. `LivingArea`
6. `FireplaceYN`
7. `YearBuilt`
8. `ParkingTotal`
9. `BedroomsTotal`
10. `PoolPrivateYN`
11. `LotSizeAcres`
12. `Stories`

Input validation blocks missing, non-finite, type-invalid, out-of-range, and internally inconsistent values. It warns when otherwise valid inputs fall outside documented observed ranges or an approximate California mainland outline.

## Application

The Streamlit application provides:

- A guided 12-feature estimation form
- A clearly formatted sale-price estimate
- Visible non-appraisal and non-financial-use limitations
- Held-out metrics and chronological evaluation details
- Exact feature-contract documentation
- Native XGBoost gain importance
- On-demand global and local SHAP explanations
- Responsive desktop and mobile layouts

SHAP contributions are calculated on demand through XGBoost's native TreeSHAP mode. If explanations are unavailable, core price inference remains usable.

![California home price application preview](docs/app-preview.png)

## Run locally

Use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

To run the tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Deploy on Streamlit Community Cloud

The repository is organized for Streamlit Community Cloud and requires no secrets or external services:

- Repository: `mmiovski/xgb-streamlit-app`
- Branch: `main`
- Entrypoint: `app.py`
- Python: `3.12`

The application is deployed at [xgb-cali-houseprices.streamlit.app](https://xgb-cali-houseprices.streamlit.app/). Streamlit Community Cloud tracks `main` and rebuilds the existing application after approved pushes. Root `requirements.txt` contains all runtime dependencies, `.streamlit/config.toml` contains the visual theme, and model artifacts are addressed relative to `app.py`.

The deployment uses XGBoost's CPU-only wheel because the application performs CPU inference. This keeps the cloud installation substantially smaller without changing the `xgboost` API or native model format.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for deployment coordinates and the verified acceptance checklist.

## Repository structure

```text
.
|-- app.py                         Streamlit entrypoint
|-- artifacts/
|   |-- model_metadata.json        Model contract, metrics, and checksums
|   |-- shap_background.csv        Feature-only explanation sample
|   `-- xgb_nolist.ubj             Native XGBoost model
|-- docs/
|   `-- DEPLOYMENT.md              Streamlit Cloud deployment runbook
|-- housing_price/
|   |-- charts.py                  Accessible model charts
|   |-- contract.py                Feature schema and validation
|   |-- explanations.py            Optional SHAP and gain helpers
|   |-- inference.py               Artifact verification and inference
|   `-- styles.py                  Streamlit visual system
|-- notebooks/
|   |-- eda.ipynb                  Executed EDA and preprocessing
|   `-- modeling.ipynb             Executed training and evaluation
|-- scripts/                       Artifact and notebook verification tools
|-- tests/                         Contract, artifact, inference, and app tests
|-- MODEL_CARD.md                  Intended use and limitations
|-- requirements.txt               Pinned runtime dependencies
`-- requirements-dev.txt           Test dependencies
```

## Data availability

The project uses proprietary MLS records that are not included. Raw records contain addresses, agent details, brokerage information, listing identifiers, and other non-public fields. The notebooks retain aggregate outputs only. The application artifact and SHAP background contain the 12 model features but no target values, addresses, names, emails, or listing identifiers.

The notebooks expect the original ten monthly filenames when rerun. Without authorized access to those records, a recruiter can run the application and software tests but cannot independently reproduce the performance metrics.

## Model artifact and reproducibility

The deployed model uses XGBoost's native UBJSON format instead of pickle. The conversion script accepts only the audited source pickle SHA-256 and verified exact log-prediction parity across 303 rows. At startup, the application checks the native artifact checksum, feature names, order, and feature count before inference.

The controlled default-input smoke prediction is `$771,358`. This proves deterministic application inference, not real-world model accuracy.

See [MODEL_CARD.md](MODEL_CARD.md) and [artifacts/model_metadata.json](artifacts/model_metadata.json) for the full contract.

## Limitations

- The training period is limited to 2025 California single-family sales.
- The model can become stale as housing-market conditions change.
- It omits condition, renovations, views, school assignments, and current economic factors.
- Coordinate and property validation covers obvious errors, not every MLS rule.
- A point estimate does not quantify uncertainty.
- The output is not an appraisal, offer, lending decision, or financial recommendation.

## Limited future work

The bounded next step is periodic retraining with a newly authorized time window, followed by temporal drift and calibration checks. A prediction interval would also improve uncertainty communication. Neither is required for this portfolio renovation because the necessary later-period data and a validated interval method are outside the current artifact scope.

## License

Code and documentation are available under the [MIT License](LICENSE). The license does not grant rights to the excluded proprietary MLS source data.
