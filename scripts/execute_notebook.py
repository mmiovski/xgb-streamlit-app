"""Execute a notebook in a specified data directory and save verified outputs."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cell-timeout", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    args = parse_args()
    notebook_path = args.notebook.resolve()
    data_dir = args.data_dir.resolve()
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    with notebook_path.open("r", encoding="utf-8") as stream:
        notebook = nbformat.read(stream, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=args.cell_timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(data_dir)}},
        allow_errors=False,
        record_timing=True,
    )
    client.execute()

    with notebook_path.open("w", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", ())
        if output.output_type == "error"
    ]
    if error_outputs:
        raise RuntimeError(f"Notebook retained {len(error_outputs)} error outputs.")
    print(f"notebook={notebook_path.name}")
    print(f"executed_code_cells={len(code_cells)}")
    print("error_outputs=0")


if __name__ == "__main__":
    main()
