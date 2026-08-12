# Streamlit Community Cloud Deployment

This repository is prepared for Streamlit Community Cloud. Deployment is not part of the current local renovation and requires a separate commit, push, and deployment authorization.

## Deployment coordinates

- Repository: `mmiovski/xgb-streamlit-app`
- Branch: `main`
- Entrypoint: `app.py`
- Python version: `3.12`
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

## Deploy

1. Complete local technical testing and user visual approval.
2. Separately authorize a commit and push of the approved changes.
3. Open Streamlit Community Cloud and select **Create app**.
4. Choose `mmiovski/xgb-streamlit-app`, branch `main`, and entrypoint `app.py`.
5. Open Advanced settings and select Python 3.12.
6. Leave Secrets empty.
7. Deploy and monitor the build log until the app reports a healthy state.

## Post-deployment acceptance checklist

- The build installs from `requirements.txt` without manual intervention.
- The landing page renders with no missing-artifact or checksum error.
- The default form submission returns `$771,358`.
- A changed input returns a finite, formatted dollar estimate.
- The Model and methodology page loads.
- Gain importance renders.
- On-demand global SHAP renders.
- A submitted estimate produces a local SHAP chart.
- The layout is usable on desktop and a narrow mobile viewport.
- No raw traceback, secret, local path, or proprietary record is displayed.
- The public URL is added to the README only after it is confirmed healthy.

## Troubleshooting

If the build fails, inspect the Community Cloud log for the first dependency or artifact error. Do not weaken checksum or feature-contract validation to bypass a deployment failure. Confirm that Python 3.12 was selected and that all three files under `artifacts/` were pushed.

If the app loads but SHAP fails, core inference should remain available. Record the exact log error and resolve the explanation dependency without changing the verified model artifact.
