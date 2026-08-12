"""Create publication-ready notebook copies without changing the source notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


Notebook = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda-source", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _source_lines(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def _set_source(cell: dict[str, Any], text: str) -> None:
    cell["source"] = _source_lines(text)


def _set_markdown(cell: dict[str, Any], text: str) -> None:
    cell["cell_type"] = "markdown"
    cell["source"] = _source_lines(text)
    cell["metadata"] = {}
    cell.pop("execution_count", None)
    cell.pop("outputs", None)


def _load(path: Path, expected_cells: int) -> Notebook:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if len(notebook.get("cells", ())) != expected_cells:
        raise RuntimeError(f"Unexpected notebook structure: {path.name}")
    for cell in notebook["cells"]:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return notebook


def _repair_encoding(notebook: Notebook) -> None:
    replacements = {
        "RÂ²": "R²",
        "Ã—": "x",
        "â€“": " to ",
        "â€”": " - ",
        "â€œ": '"',
        "â€": '"',
        "â€™": "'",
    }
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", ()))
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)


def prepare_eda(path: Path) -> Notebook:
    notebook = _load(path, expected_cells=76)
    cells = notebook["cells"]

    _set_source(
        cells[0],
        """# California Housing Exploratory Analysis and Preprocessing

This notebook documents the exploratory analysis and preprocessing used for a 2025 California single-family-home price project. January through August form the training period; September and October form a later-time held-out period.

Raw MLS records are not included in the public repository. Retained outputs contain aggregate summaries and visualizations only. The final model build is documented in `modeling.ipynb`.""",
    )
    _set_source(
        cells[3],
        """## Load, verify, and merge monthly data

The chronological split keeps January through August 2025 in the training set and reserves September and October 2025 for held-out evaluation. All preprocessing decisions are learned from the training data and then applied to the held-out months.""",
    )
    _set_source(
        cells[5],
        """# January through August training months
df1 = pd.read_csv('CRMLSSold202503_filled.csv')    # March
df2 = pd.read_csv('CRMLSSold202504_filled.csv')    # April
df3 = pd.read_csv('CRMLSSold202505_filled.csv')    # May
df4 = pd.read_csv('CRMLSSold202506_filled.csv')    # June
df5 = pd.read_csv('CRMLSSold202507_filled.csv')    # July
df6 = pd.read_csv('CRMLSSold202508_filled-2.csv')  # August
df7 = pd.read_csv('CRMLSSold202501_filled.csv')    # January
df8 = pd.read_csv('CRMLSSold202502_filled.csv')    # February

# September and October held-out months
tst_sep = pd.read_csv('CRMLSSold202509.csv')
tst_oct = pd.read_csv('CRMLSSold202510.csv')""",
    )
    _set_source(
        cells[7],
        """training_months = [df7, df8, df1, df2, df3, df4, df5, df6]
held_out_months = [tst_sep, tst_oct]

training_columns = [list(frame.columns) for frame in training_months]
all_equal = all(training_columns[0] == columns for columns in training_columns[1:])
print('Training-month columns identical?', all_equal)

if not all_equal:
    for month_number, columns in enumerate(training_columns, start=1):
        print(f'Training frame {month_number} columns: {columns}')""",
    )
    _set_source(
        cells[8],
        """`training_months` and `held_out_months` keep the chronological roles explicit. They are processed with the same feature logic, but the groups are never mixed.""",
    )
    _set_source(
        cells[9],
        """The eight training files share an exact schema. The September and October files omit `LonFilled` and `LatFilled`; neither field is retained for analysis or modeling, so the difference does not affect the final feature contract.""",
    )
    _set_source(
        cells[10],
        """trn = pd.concat(training_months, ignore_index=True)
tst = pd.concat(held_out_months, ignore_index=True)

print('Training data')
trn.info()
print()
print('Held-out data')
tst.info()""",
    )
    _set_markdown(
        cells[46],
        """### Percentile trimming

The final pipeline does not use the earlier IQR experiment. For selected numeric fields, it learns the 0.5th and 99.5th percentile bounds from training data only, retaining the middle 99% of each feature. The same fixed bounds are then applied to the held-out data.""",
    )
    _set_markdown(
        cells[47],
        """This chronological, training-derived treatment limits extreme records without using information from September or October.""",
    )
    _set_source(cells[48], "### Apply training-derived percentile bounds")
    _set_source(
        cells[49],
        """# Numeric fields included in percentile trimming
num_cols = ['LotSizeSquareFeet',
            'LotSizeAcres',
            'LivingArea',
            'BathroomsTotalInteger',
            'BedroomsTotal',
            'ListPrice',
            'OriginalListPrice',
            'ClosePrice',
            'DaysOnMarket']

pct_bounds = {}
for col in num_cols:
    # Learn the 0.5th and 99.5th percentile thresholds from training data only.
    lower = trn[col].quantile(0.005)
    upper = trn[col].quantile(0.995)
    pct_bounds[col] = (lower, upper)
    trn = trn[(trn[col] >= lower) & (trn[col] <= upper)]

# Apply the fixed training thresholds to held-out data.
for col, (lower, upper) in pct_bounds.items():
    tst = tst[(tst[col] >= lower) & (tst[col] <= upper)]""",
    )
    _set_source(
        cells[54],
        """### Boxplots after percentile trimming

These plots provide a visual check of the retained distributions after applying the training-derived bounds.""",
    )
    _set_source(
        cells[65],
        """corr = trn[numeric_cols].corr()

# Use a deterministic sample and a focused feature set so the pairplot remains readable.
pair_cols = ['ClosePrice', 'LivingArea', 'LotSizeAcres',
             'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket']
plot_sample = trn[pair_cols].sample(n=min(3000, len(trn)), random_state=420)
sns.pairplot(plot_sample, corner=True, plot_kws={'alpha': 0.25, 's': 12})
plt.suptitle('Selected numeric feature relationships', y=1.02)
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation heatmap of numeric features')
plt.show()""",
    )
    _set_source(cells[70], "#### `LogPrice = ln(ClosePrice)`")
    _set_source(
        cells[71],
        """# Match the deployed model's natural-log target transformation.
trn['LogPrice'] = np.log(trn['ClosePrice'])
tst['LogPrice'] = np.log(tst['ClosePrice'])

sns.histplot(trn['LogPrice'], kde=True)
plt.title('Training distribution of natural-log sale price')
plt.show()""",
    )
    _set_source(cells[74], "### Reproducibility boundary")
    _set_markdown(
        cells[75],
        """Raw and processed MLS records are intentionally not exported or included in this repository. The executed outputs preserve aggregate evidence while protecting listing-level data.""",
    )
    _repair_encoding(notebook)
    return notebook


def prepare_model(path: Path) -> Notebook:
    notebook = _load(path, expected_cells=51)
    cells = notebook["cells"]

    _set_source(
        cells[0],
        """# California Home Price Modeling with XGBoost

This notebook trains and evaluates two XGBoost regressors on a chronological 2025 split. The deployed list-unaware model predicts sale price without `ListPrice` or `OriginalListPrice`; it is the primary portfolio artifact because it can estimate properties that are not actively listed.

The list-aware model is retained only as project context and a leakage-adjacent comparison. Raw MLS records are not included in the repository.

## Libraries""",
    )
    _set_source(
        cells[1],
        """import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# CPU is the portable default. Set XGB_DEVICE=cuda to accelerate a local verification run.
XGB_DEVICE = os.environ.get('XGB_DEVICE', 'cpu')
print(f'XGBoost device: {XGB_DEVICE}')""",
    )
    _set_source(
        cells[2],
        """## Load monthly data

January through August 2025 form the training period. September and October 2025 are reserved as a later-time held-out test set.""",
    )
    _set_source(
        cells[3],
        """# January through August training months
df1 = pd.read_csv('CRMLSSold202503_filled.csv')    # March
df2 = pd.read_csv('CRMLSSold202504_filled.csv')    # April
df3 = pd.read_csv('CRMLSSold202505_filled.csv')    # May
df4 = pd.read_csv('CRMLSSold202506_filled.csv')    # June
df5 = pd.read_csv('CRMLSSold202507_filled.csv')    # July
df6 = pd.read_csv('CRMLSSold202508_filled-2.csv')  # August
df7 = pd.read_csv('CRMLSSold202501_filled.csv')    # January
df8 = pd.read_csv('CRMLSSold202502_filled.csv')    # February

# September and October held-out months
tst = pd.read_csv('CRMLSSold202509.csv')
tst2 = pd.read_csv('CRMLSSold202510.csv')""",
    )
    _set_source(
        cells[24],
        """### Outlier treatment

Selected numeric fields are trimmed using the 0.5th and 99.5th percentile bounds learned from training data only. The same fixed bounds are applied to the held-out months.""",
    )
    _set_markdown(
        cells[25],
        """The final model does not use the earlier IQR experiment. Percentile trimming preserves the middle 99% of each selected training feature and avoids learning thresholds from held-out data.""",
    )
    _set_source(
        cells[26],
        """num_cols = ['LotSizeSquareFeet',
            'LotSizeAcres',
            'LivingArea',
            'BathroomsTotalInteger',
            'BedroomsTotal',
            'ListPrice',
            'OriginalListPrice',
            'ClosePrice',
            'DaysOnMarket']

pct_bounds = {}
for col in num_cols:
    # Learn the 0.5th and 99.5th percentile thresholds from training data only.
    lower = trn[col].quantile(0.005)
    upper = trn[col].quantile(0.995)
    pct_bounds[col] = (lower, upper)
    trn = trn[(trn[col] >= lower) & (trn[col] <= upper)]

# Apply the fixed training thresholds to held-out data.
for col, (lower, upper) in pct_bounds.items():
    tst = tst[(tst[col] >= lower) & (tst[col] <= upper)]""",
    )
    _set_source(
        cells[27],
        """## Retain the modeling features

Two variants are trained. The list-aware comparison includes listing-price fields, which closely track final sale price. The deployed list-unaware model removes both listing-price fields and uses the 12 property features exposed by the application.""",
    )
    _set_source(cells[29], "## Reproducibility boundary")
    _set_markdown(
        cells[30],
        """Processed train and test records are not exported from the publication notebook. The source MLS data and listing-level derivatives remain outside the repository.""",
    )
    _set_source(cells[31], "## Target and feature encoding")
    _set_source(cells[33], "## Context comparison: list-aware XGBoost")
    _set_source(cells[36], "## Deployed model: list-unaware XGBoost")
    _set_source(
        cells[34],
        """# Context comparison only: listing-price features are included.
xgb_list = XGBRegressor(max_depth=7,
                        learning_rate=0.01,
                        n_estimators=1000,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        device=XGB_DEVICE)

xgb_list.fit(X_trn_list, y_trn_log_list)
y_pred_log_list = xgb_list.predict(X_tst_list)
y_pred_list = np.exp(y_pred_log_list)

r2_xgb_list = r2_score(y_tst_list, y_pred_list)
mape_xgb_list = mean_absolute_percentage_error(y_tst_list, y_pred_list) * 100
mdape_xgb_list = np.median(np.abs((y_tst_list - y_pred_list) / y_tst_list)) * 100""",
    )
    _set_source(
        cells[38],
        """# Primary deployed model: no listing-price features.
xgb_nolist = XGBRegressor(max_depth=7,
                          learning_rate=0.05,
                          n_estimators=1300,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          random_state=42,
                          n_jobs=-1,
                          device=XGB_DEVICE)

xgb_nolist.fit(X_trn_nolist, y_trn_log_nolist)
y_pred_log_nolist = xgb_nolist.predict(X_tst_nolist)
y_pred_nolist = np.exp(y_pred_log_nolist)

r2_xgb_nolist = r2_score(y_tst_nolist, y_pred_nolist)
mape_xgb_nolist = mean_absolute_percentage_error(y_tst_nolist, y_pred_nolist) * 100
mdape_xgb_nolist = np.median(np.abs((y_tst_nolist - y_pred_nolist) / y_tst_nolist)) * 100""",
    )
    _set_source(
        cells[41],
        """results_df = pd.DataFrame({
    'Model': ['List-Aware XGBoost', 'List-Unaware XGBoost'],
    'R2': [round(r2_xgb_list, 2), round(r2_xgb_nolist, 2)],
    'MAPE (%)': [round(mape_xgb_list, 2), round(mape_xgb_nolist, 2)],
    'MdAPE (%)': [round(mdape_xgb_list, 2), round(mdape_xgb_nolist, 2)]
})

print()
print('FINAL MODEL COMPARISON')
print()
print(results_df.to_string(index=False))""",
    )
    _set_source(
        cells[47],
        """bands = {
    'Below $500K': (0, 500_000),
    '$500K to $1M': (500_000, 1_000_000),
    '$1M to $2M': (1_000_000, 2_000_000),
    '$2M to $5M': (2_000_000, 5_000_000),
    '$5M to $10M': (5_000_000, 10_000_000),
    '$10M and above': (10_000_000, np.inf),
}

def mdape_fn(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    return round(np.median(np.abs((y_true - y_pred) / y_true) * 100), 2)

def mape_fn(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    return round(np.mean(np.abs((y_true - y_pred) / y_true) * 100), 2)

rows = []
for band_name, (low, high) in bands.items():
    mask_list = (y_tst_list >= low) & (y_tst_list < high)
    mask_nolist = (y_tst_nolist >= low) & (y_tst_nolist < high)
    rows.append([
        band_name,
        mdape_fn(y_tst_nolist[mask_nolist], y_pred_nolist[mask_nolist]),
        mape_fn(y_tst_nolist[mask_nolist], y_pred_nolist[mask_nolist]),
        mdape_fn(y_tst_list[mask_list], y_pred_list[mask_list]),
        mape_fn(y_tst_list[mask_list], y_pred_list[mask_list]),
    ])

rows.append([
    'All price bands',
    mdape_fn(y_tst_nolist, y_pred_nolist),
    mape_fn(y_tst_nolist, y_pred_nolist),
    mdape_fn(y_tst_list, y_pred_list),
    mape_fn(y_tst_list, y_pred_list),
])

price_band_table = pd.DataFrame(
    rows,
    columns=['Price band', 'No-list MdAPE', 'No-list MAPE',
             'List-aware MdAPE', 'List-aware MAPE'],
)

print()
print('PRICE-BAND PERFORMANCE')
print()
print(price_band_table.to_string(index=False))""",
    )
    _set_source(cells[48], "## Artifact provenance")
    _set_markdown(
        cells[49],
        """The deployed model is stored as `artifacts/xgb_nolist.ubj`, XGBoost's native UBJSON format. It was converted from the repository's trusted pickle with an exact parity check across 303 feature rows. The application verifies the native artifact checksum and embedded feature order before inference.""",
    )
    _set_source(
        cells[50],
        """The list-unaware model is the deployed portfolio artifact. Reported performance is tied to the chronological September-October 2025 holdout shown above; application smoke tests verify software behavior, not predictive performance.""",
    )
    _repair_encoding(notebook)
    return notebook


def _write(notebook: Notebook, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    _write(prepare_eda(args.eda_source.resolve()), output_dir / "eda.ipynb")
    _write(prepare_model(args.model_source.resolve()), output_dir / "modeling.ipynb")
    print(f"notebooks_written={output_dir}")


if __name__ == "__main__":
    main()
