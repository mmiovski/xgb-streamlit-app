"""Tests for the published notebook contracts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_notebooks import build_tuning_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_notebook(name: str) -> dict:
    path = PROJECT_ROOT / "notebooks" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _all_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", ())) for cell in notebook.get("cells", ())
    )


def test_tuning_notebook_matches_the_generated_source_contract() -> None:
    committed = _load_notebook("xgboost_tuning.ipynb")
    generated = build_tuning_notebook()

    assert [cell["cell_type"] for cell in committed["cells"]] == [
        cell["cell_type"] for cell in generated["cells"]
    ]
    assert [cell["source"] for cell in committed["cells"]] == [
        cell["source"] for cell in generated["cells"]
    ]

    error_outputs = [
        output
        for cell in committed["cells"]
        for output in cell.get("outputs", ())
        if output.get("output_type") == "error"
    ]
    assert not error_outputs


def test_tuning_notebook_records_search_scope_and_execution_guard() -> None:
    source = _all_source(_load_notebook("xgboost_tuning.ipynb"))

    assert "RUN_FULL_TUNING" in source
    assert "Full 1,730-fit search skipped" in source
    assert "prod(len(values) for values in broad_grid.values()) == 125" in source
    assert "prod(len(values) for values in refined_grid.values()) == 48" in source
    assert "'max_depth': 7" in source
    assert "'learning_rate': 0.05" in source
    assert "'n_estimators': 1300" in source


def test_modeling_notebook_links_selection_to_the_tuning_record() -> None:
    source = _all_source(_load_notebook("modeling.ipynb"))

    assert "## Hyperparameter selection" in source
    assert "xgboost_tuning.ipynb" in source


def test_metadata_records_tuning_and_temporal_test_boundaries() -> None:
    metadata_path = PROJECT_ROOT / "artifacts" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["evaluation"]["strictly_untouched"] is False
    assert metadata["tuning"]["candidate_counts"] == {
        "broad_per_criterion": 125,
        "refined_per_criterion": 48,
        "cross_validation_fits_total": 1730,
    }
    assert metadata["tuning"]["source_feature_count"] == 18
    assert metadata["tuning"]["final_feature_count"] == 12
