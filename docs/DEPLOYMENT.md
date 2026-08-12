# Streamlit Community Cloud Deployment

This repository is deployed on Streamlit Community Cloud. The existing application tracks the repository's `main` branch and updates in place after approved pushes.

## Deployment coordinates

- Repository: `mmiovski/xgb-streamlit-app`
- Branch: `main`
- Entrypoint: `app.py`
- Python version: `3.12`
- Public URL: <https://xgb-cali-houseprices.streamlit.app/>
- Secrets: none
- External Linux packages: none

## Repository prerequisites

- `app.py` is at the repository root.
- `requirements.txt` is at the repository root and contains pinned runtime dependencies.
- `.streamlit/config.toml` is at the repository root.
- `artifacts/xgb_nolist.ubj`, `artifacts/model_metadata.json`, and `artifacts/shap_background.csv` are tracked.
- Runtime paths are resolved from `app.py`, not the current working directory.
- XGBoost uses the CPU-only 3.1.2 wheel to reduce cloud build size.
- No raw MLS source data is tracked.

## Update the existing deployment

1. Complete local technical testing and user visual approval.
2. Separately authorize a commit and push of the approved changes.
3. Push the approved commit to `main`.
4. Allow the existing Streamlit application to rebuild in place.
5. Monitor startup until the public application reports a healthy state.
6. Complete the post-deployment acceptance checklist below.

## Verified post-deployment acceptance

The public deployment was verified on August 11, 2026:

- The build installs from `requirements.txt` without manual intervention.
- The landing page renders with no missing-artifact or checksum error.
- The default form submission returns `$771,358`.
- The Model and methodology page loads.
- Gain importance renders.
- On-demand global SHAP renders.
- A submitted estimate produces a local SHAP chart.
- No raw traceback, secret, local path, or proprietary record is displayed.
- The public URL is included in the README.

## Troubleshooting

If a rebuild fails, inspect the Community Cloud log for the first dependency or artifact error. Do not weaken checksum or feature-contract validation to bypass a deployment failure. Confirm that Python 3.12 remains selected and that all three files under `artifacts/` are present on `main`.

If the app loads but SHAP fails, core inference should remain available. Record the exact log error and resolve the explanation dependency without changing the verified model artifact.
