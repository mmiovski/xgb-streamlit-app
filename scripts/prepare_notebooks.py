"""Create cleaned, reproducible notebook copies without changing the sources."""

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


def _markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(text),
    }


def _code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
    }


def _assign_cell_ids(notebook: Notebook, prefix: str) -> None:
    for index, cell in enumerate(notebook.get("cells", ())):
        cell["id"] = f"{prefix}-{index:03d}"


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


def _replace_markdown_lines(
    cell: dict[str, Any], replacements: dict[str, str]
) -> None:
    lines = []
    for line in cell.get("source", ()):
        replacement = next(
            (value for prefix, value in replacements.items() if line.startswith(prefix)),
            None,
        )
        lines.append(f"{replacement}\n" if replacement is not None else line)
    cell["source"] = lines


def prepare_eda(path: Path) -> Notebook:
    notebook = _load(path, expected_cells=76)
    cells = notebook["cells"]

    _set_markdown(
        cells[6],
        """### Verify Housing Data Types and Structure

Matching schemas are required before concatenating the monthly training files.""",
    )
    _set_markdown(
        cells[12],
        """Filter the records to:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`
- `StateOrProvince = CA`

`PropertyType=Residential` defines the broad residential category. `PropertySubType=SingleFamilyResidence` removes condos, townhouses, multifamily properties, and mobile homes, leaving detached houses. `StateOrProvince=CA` restricts the target population to California.""",
    )
    _replace_markdown_lines(
        cells[19],
        {
            "`UnparsedAddress`": "`UnparsedAddress`, String, 0.109221, Text representation of the address with the full civic location and may OPTIONALLY include any of the City, StateOrProvince, PostcalCode, County, Remove; structured location features are already available.",
            "`LivingArea`": "`LivingArea`, Decimal, 0.055943, Total livable area in the house, Keep; physical feature likely to predict price; units require standardization.",
            "`PurchaseContractDate`": "`PurchaseContractDate`, DateTime, 0.003996, date offer is accepted and listing is no longer on market, Remove; used only to calculate `DaysOnMarket`, which is already available",
            "`PropertySubType`": "`PropertySubType`, String, 0.000000, subtypes to the PropertyType variable, Remove; constant after filtering on `PropertySubType = SingleFamilyResidence`.",
            "`ListingKeyNumeric`": "`ListingKeyNumeric`, Integral, 0.000000, Remove; redundant because `ListingKey` is the primary key.",
            "`PropertyType`": "`PropertyType`, String, 0.000000, the type of property, Remove; constant after filtering on `PropertyType = Residential`.",
            "`StateOrProvince`": "`StateOrProvince`, String, 0.000000, state the listing is in, Remove; constant after filtering on `StateOrProvince = CA`.",
            "`ListingContractDate`": "`ListingContractDate`, DateTime, 0.000000, date the listing agreement was signed between the seller and the listing agent i.e., the date the house is put on the market, Remove; redundant with `DaysOnMarket`",
            "`ListingId`": "`ListingId`, String, 0.000000, Remove; unique key for a specific listing, but `ListingKey` is already the primary key.",
            "`Latfilled`": "`Latfilled`, Boolean, 0.000000, Remove; redundant latitude-completeness flag.",
            "`Lonfilled`": "`Lonfilled`, Boolean, 0.000000, Remove; redundant longitude-completeness flag.",
        },
    )
    _set_markdown(
        cells[20],
        """`Maybe Keep` identifies potentially useful fields whose missingness or inconsistency complicates modeling. They are excluded from the current feature set and can be reconsidered if later experiments justify the additional processing.""",
    )
    _set_markdown(
        cells[25],
        """`AssociationFee` is nearly 30% missing, so extensive imputation could distort its distribution.""",
    )
    _set_markdown(
        cells[27],
        """`Flooring`, `AssociationFee`, and `Levels` are excluded from the current feature set.""",
    )
    _set_markdown(
        cells[30],
        """### Approaches for Missing Value Handling

Features with complete or near-complete missingness are removed.

For features below roughly 10% missingness, mean, median, or mode imputation can alter relationships and reduce variance. Missing entries are instead filled with fixed-seed random draws from observed training values, preserving the empirical center and spread.

Test-set values are sampled only from training observations to prevent preprocessing leakage.""",
    )
    _set_markdown(
        cells[37],
        """### Duplicates

Duplicate listing keys are resolved by retaining the record with the most recent closing date.""",
    )
    _set_markdown(
        cells[39],
        """The same duplicate-removal rule is applied to `tst`, preventing repeated properties from distorting test metrics.""",
    )
    _set_markdown(
        cells[41],
        """After deduplication, `ListingKey` and `CloseDate` are removed because they are no longer needed.""",
    )
    _set_markdown(
        cells[43],
        """### Impossible Values

Domain rules remove records with impossible bedrooms, bathrooms, living area, lot size, parking, construction year, price, or California coordinates. Zero-day listings remain valid because a residence can be listed and closed on the same day.

Living area cannot exceed lot area, and garage capacity cannot exceed total parking. IQR filtering made the discrete parking fields deterministic, so a domain-informed upper bound of 30 spaces is used instead.""",
    )
    _set_markdown(
        cells[58],
        """`City` and `CountyOrParish` are excluded because their high cardinality would substantially expand the encoded feature space.

`ContractStatusChangeDate` is also excluded because it does not add necessary model information.""",
    )

    _set_source(
        cells[0],
        """# California Housing Exploratory Analysis and Preprocessing

This notebook documents the exploratory analysis and preprocessing used for a 2025 California single-family-home price project. January through August form the training period; September and October form a later-time test period.

Raw MLS records are not included in the public repository. Retained outputs contain aggregate summaries and visualizations only. The final model build is documented in `modeling.ipynb`.""",
    )
    _set_source(
        cells[3],
        """## Load, verify, and merge monthly data

The chronological split keeps January through August 2025 in the training set and uses September and October 2025 for later-time testing. All preprocessing decisions are learned from the training data and then applied to the test months.""",
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

# September and October test months
tst_sep = pd.read_csv('CRMLSSold202509.csv')
tst_oct = pd.read_csv('CRMLSSold202510.csv')""",
    )
    _set_source(
        cells[7],
        """training_months = [df7, df8, df1, df2, df3, df4, df5, df6]
test_months = [tst_sep, tst_oct]

training_columns = [list(frame.columns) for frame in training_months]
all_equal = all(training_columns[0] == columns for columns in training_columns[1:])
print('Training-month columns identical?', all_equal)

if not all_equal:
    for month_number, columns in enumerate(training_columns, start=1):
        print(f'Training frame {month_number} columns: {columns}')""",
    )
    _set_source(
        cells[8],
        """`training_months` and `test_months` keep the chronological roles explicit. They are processed with the same feature logic, but the groups are never mixed.""",
    )
    _set_source(
        cells[9],
        """The eight training files share an exact schema. The September and October files omit `LonFilled` and `LatFilled`; neither field is retained for analysis or modeling, so the difference does not affect the final feature contract.""",
    )
    _set_source(
        cells[10],
        """trn = pd.concat(training_months, ignore_index=True)
tst = pd.concat(test_months, ignore_index=True)

print('Training data')
trn.info()
print()
print('Test data')
tst.info()""",
    )
    _set_markdown(
        cells[46],
        """### Percentile trimming

The final pipeline does not use the earlier IQR experiment. For selected numeric fields, it learns the 0.5th and 99.5th percentile bounds from training data only, retaining the middle 99% of each feature. The same fixed bounds are then applied to the test data.""",
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

# Apply the fixed training thresholds to test data.
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
    _assign_cell_ids(notebook, "eda")
    return notebook


def prepare_model(path: Path) -> Notebook:
    notebook = _load(path, expected_cells=51)
    cells = notebook["cells"]

    _set_source(
        cells[0],
        """# California Home Price Modeling with XGBoost

This notebook trains and evaluates two XGBoost regressors on a chronological 2025 split. The deployed list-unaware model predicts sale price without `ListPrice` or `OriginalListPrice`, allowing estimates for properties that are not actively listed.

The list-aware model serves as a benchmark for the predictive advantage of listing-price fields. Raw MLS records are not included in the repository.

## Libraries""",
    )
    _set_markdown(
        cells[11],
        """## Retain Only Relevant Features

The deployed feature set contains:

- `DaysOnMarket`
- `Latitude`
- `Longitude`
- `BathroomsTotalInteger`
- `LivingArea`
- `FireplaceYN`
- `YearBuilt`
- `ParkingTotal`
- `BedroomsTotal`
- `PoolPrivateYN`
- `LotSizeAcres`
- `Stories`

Several additional columns remain temporarily to support preprocessing and are removed before training.""",
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

January through August 2025 form the training period. September and October 2025 form a later-time test set.""",
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

# September and October test months
tst = pd.read_csv('CRMLSSold202509.csv')
tst2 = pd.read_csv('CRMLSSold202510.csv')""",
    )
    _set_source(
        cells[24],
        """### Outlier treatment

Selected numeric fields are trimmed using the 0.5th and 99.5th percentile bounds learned from training data only. The same fixed bounds are applied to the test months.""",
    )
    _set_markdown(
        cells[25],
        """The final model does not use the earlier IQR experiment. Percentile trimming preserves the middle 99% of each selected training feature and avoids learning thresholds from test data.""",
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

# Apply the fixed training thresholds to test data.
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
        """Processed train and test records are not exported from this notebook. The source MLS data and listing-level derivatives remain outside the repository.""",
    )
    _set_source(cells[31], "## Target and feature encoding")
    _set_source(cells[33], "## Benchmark: list-aware XGBoost")
    _set_source(
        cells[36],
        """## Hyperparameter selection

The deployed settings came from a two-stage exhaustive five-fold search on a predecessor list-unaware feature frame and were retained after the runtime contract was reduced to the 12 inputs below. `xgboost_tuning.ipynb` contains the grids, archived selections, and reproducibility boundary.

## Deployed model: list-unaware XGBoost""",
    )
    _set_source(
        cells[34],
        """# Benchmark model: listing-price features are included.
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
        """The list-unaware model is deployed by the Streamlit application. Reported performance is tied to the chronological September-October 2025 test period shown above. September appeared in predecessor tuning diagnostics before October became available, so the combined period is not a strictly untouched evaluation set. Application smoke tests verify software behavior, not predictive performance.""",
    )
    _repair_encoding(notebook)
    _assign_cell_ids(notebook, "modeling")
    return notebook


def build_tuning_notebook() -> Notebook:
    """Build the focused XGBoost tuning and selection record."""

    cells = [
        _markdown_cell(
            """# XGBoost Hyperparameter Tuning

This notebook extracts the XGBoost search that produced the configuration retained by the deployed California home-price model. It excludes unfinished deployment cells and alternative-model experiments from the source notebooks.

The archived searches ran on an earlier 18-feature list-unaware frame. The final application uses a reduced 12-feature contract, so the archived selections are model-development evidence rather than a claim that the complete search was rerun after feature reduction. The selected configuration is re-fitted and evaluated on the final processed frames below."""
        ),
        _code_cell(
            """import os
from math import prod

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

FEATURES = [
    'DaysOnMarket',
    'Latitude',
    'Longitude',
    'BathroomsTotalInteger',
    'LivingArea',
    'FireplaceYN',
    'YearBuilt',
    'ParkingTotal',
    'BedroomsTotal',
    'PoolPrivateYN',
    'LotSizeAcres',
    'Stories',
]

XGB_DEVICE = os.environ.get('XGB_DEVICE', 'cpu')
RUN_FULL_TUNING = os.environ.get('RUN_FULL_TUNING', '0') == '1'
SEARCH_JOBS = 1 if XGB_DEVICE == 'cuda' else -1

print(f'XGBoost device: {XGB_DEVICE}')
print(f'Full tuning enabled: {RUN_FULL_TUNING}')"""
        ),
        _markdown_cell(
            """## Final processed data contract

`trn.csv` contains the January-August training frame and `tst.csv` contains the September-October temporal test frame. These processed listing-level files are excluded from the repository. Only the 12 deployed features and the sale-price target are used below; listing-price fields remain available solely for the separate benchmark in `modeling.ipynb`."""
        ),
        _code_cell(
            """trn = pd.read_csv('trn.csv')
tst = pd.read_csv('tst.csv')

required_columns = FEATURES + ['ClosePrice', 'OriginalListPrice', 'ListPrice']
for name, frame in [('training', trn), ('test', tst)]:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{name} frame is missing required columns: {missing}')
    if frame[FEATURES + ['ClosePrice']].isna().any().any():
        raise ValueError(f'{name} frame contains missing model values')
    if (frame['ClosePrice'] <= 0).any():
        raise ValueError(f'{name} frame contains non-positive sale prices')

for frame in (trn, tst):
    frame[['FireplaceYN', 'PoolPrivateYN']] = frame[
        ['FireplaceYN', 'PoolPrivateYN']
    ].astype(int)

X_trn = trn[FEATURES].copy()
X_tst = tst[FEATURES].copy()
y_trn = trn['ClosePrice'].to_numpy(dtype=float)
y_tst = tst['ClosePrice'].to_numpy(dtype=float)
y_trn_log = np.log(y_trn)

print(f'Training shape: {X_trn.shape}')
print(f'Test shape: {X_tst.shape}')
print(f'Feature order: {list(X_trn.columns)}')"""
        ),
        _markdown_cell(
            """## Search objectives

The source tuning code compared mean and median absolute percentage error on the natural-log target used for training. These are selection criteria in log space, not dollar-space MAPE and MdAPE. Dollar-space metrics are calculated only after applying the exponential inverse transform.

The cleaned code retains the archived objectives so the documented parameter selections remain interpretable."""
        ),
        _code_cell(
            """def median_log_target_ape(y_true_log, y_pred_log):
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    return float(np.median(np.abs((y_true_log - y_pred_log) / y_true_log)))


mean_log_ape_scorer = make_scorer(
    mean_absolute_percentage_error,
    greater_is_better=False,
)
median_log_ape_scorer = make_scorer(
    median_log_target_ape,
    greater_is_better=False,
)"""
        ),
        _markdown_cell(
            """## Stage 1: broad search

The broad grid varies tree depth, learning rate, and estimator count while holding row and column sampling at 0.8. It contains 125 candidate combinations and requires 625 fits per scoring criterion with five-fold cross-validation."""
        ),
        _code_cell(
            """broad_grid = {
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [100, 300, 500, 800, 1000],
}

assert prod(len(values) for values in broad_grid.values()) == 125"""
        ),
        _markdown_cell(
            """## Stage 2: refined search

The refined grid concentrates on the plateau identified by the broad searches. It contains 48 combinations and requires 240 fits per scoring criterion."""
        ),
        _code_cell(
            """refined_grid = {
    'max_depth': [7, 9, 11, 13],
    'learning_rate': [0.01, 0.05, 0.10],
    'n_estimators': [1000, 1100, 1200, 1300],
}

assert prod(len(values) for values in refined_grid.values()) == 48"""
        ),
        _markdown_cell(
            """## Archived search selections

The source notebooks retain the following completed `GridSearchCV` outputs:

| Stage | Selection criterion | Depth | Learning rate | Estimators |
| --- | --- | ---: | ---: | ---: |
| Broad | Mean log-target percentage error | 7 | 0.05 | 1,000 |
| Broad | Median log-target percentage error | 11 | 0.01 | 1,000 |
| Refined | Mean log-target percentage error | 7 | 0.05 | 1,300 |
| Refined | Median log-target percentage error | 11 | 0.01 | 1,300 |

The refined mean-error result is the configuration retained by the deployed model. Because the archived searches used the earlier expanded feature frame, this table is kept separate from the final 12-feature temporal test metrics."""
        ),
        _markdown_cell(
            """## Optional exhaustive execution

The four searches total 1,730 model fits. They are disabled during normal notebook execution so routine verification does not repeat a long model-selection run. Set `RUN_FULL_TUNING=1` before starting the kernel to execute them. On a CUDA system, set `XGB_DEVICE=cuda`; search-level parallelism is then limited to one job to avoid competing GPU fits."""
        ),
        _code_cell(
            """def make_search(param_grid, scorer):
    estimator = XGBRegressor(
        objective='reg:squarederror',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        device=XGB_DEVICE,
    )
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=SEARCH_JOBS,
        verbose=2,
    )


search_plan = {
    'broad_mean_log_ape': (broad_grid, mean_log_ape_scorer),
    'broad_median_log_ape': (broad_grid, median_log_ape_scorer),
    'refined_mean_log_ape': (refined_grid, mean_log_ape_scorer),
    'refined_median_log_ape': (refined_grid, median_log_ape_scorer),
}

search_results = {}
if RUN_FULL_TUNING:
    for name, (grid, scorer) in search_plan.items():
        search = make_search(grid, scorer)
        search.fit(X_trn, y_trn_log)
        search_results[name] = {
            'best_params': search.best_params_,
            'best_score': -float(search.best_score_),
        }
    print(pd.DataFrame(search_results).T.to_string())
else:
    print('Full 1,730-fit search skipped. Set RUN_FULL_TUNING=1 to execute it.')"""
        ),
        _markdown_cell(
            """## Selected configuration on the final feature contract

This bounded step re-fits only the selected configuration on the final 12-feature training frame and reports dollar-space performance on the September-October temporal test frame. It verifies code compatibility without repeating model selection."""
        ),
        _code_cell(
            """selected_params = {
    'objective': 'reg:squarederror',
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 1300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'device': XGB_DEVICE,
}

selected_model = XGBRegressor(**selected_params)
selected_model.fit(X_trn, y_trn_log)
y_pred = np.exp(selected_model.predict(X_tst))

evaluation = pd.DataFrame(
    {
        'Metric': ['R²', 'MAPE', 'MdAPE'],
        'Temporal-test result': [
            r2_score(y_tst, y_pred),
            mean_absolute_percentage_error(y_tst, y_pred) * 100,
            np.median(np.abs((y_tst - y_pred) / y_tst)) * 100,
        ],
    }
)
evaluation['Temporal-test result'] = evaluation['Temporal-test result'].round(2)
evaluation"""
        ),
        _markdown_cell(
            """## Interpretation boundary

The selected-configuration result above validates the final feature order, training call, inverse transform, and aggregate metric calculations. It does not represent a new exhaustive search. The deployed UBJSON artifact remains the authoritative runtime model and is independently protected by checksum, feature-order, and deterministic-prediction tests."""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _assign_cell_ids(notebook, "xgb-tuning")
    return notebook


def _write(notebook: Notebook, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    _write(prepare_eda(args.eda_source.resolve()), output_dir / "eda.ipynb")
    _write(prepare_model(args.model_source.resolve()), output_dir / "modeling.ipynb")
    _write(build_tuning_notebook(), output_dir / "xgboost_tuning.ipynb")
    print(f"notebooks_written={output_dir}")


if __name__ == "__main__":
    main()
