"""Streamlit application smoke tests."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_estimate_page_loads_and_predicts() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=45).run()
    assert not app.exception
    assert "Estimate sale price" in [button.label for button in app.button]

    submit = next(button for button in app.button if button.label == "Estimate sale price")
    submit.click()
    app.run()

    assert not app.exception
    rendered_markdown = "\n".join(item.value for item in app.markdown)
    assert "Estimated sale price" in rendered_markdown
    assert "$771,358" in rendered_markdown


def test_methodology_page_loads_without_calculating_shap() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=45).run()
    app.radio[0].set_value("Methodology")
    app.run()

    assert not app.exception
    assert "Calculate SHAP summaries" in [button.label for button in app.button]
    assert "Final training records" in [metric.label for metric in app.metric]
