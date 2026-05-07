"""Structural sanity over the notebook files: parse-able JSON, expected
shape (one H1 + at least one code cell), no nbformat surprises. Catches
broken notebooks before students try to open them."""

import json
from pathlib import Path
import pytest


NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))

assert NOTEBOOKS, f"no notebooks found under {NB_DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_is_valid_json(path):
    with open(path) as f:
        nb = json.load(f)
    assert "cells" in nb, f"{path.name} has no 'cells' key"
    assert nb.get("nbformat", 0) >= 4, f"{path.name} has nbformat < 4"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_starts_with_h1(path):
    with open(path) as f:
        nb = json.load(f)
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown", f"{path.name}: first cell not markdown"
    src = "".join(first["source"]) if isinstance(first["source"], list) else first["source"]
    assert src.lstrip().startswith("# "), f"{path.name}: first cell not H1"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_has_code_cells(path):
    with open(path) as f:
        nb = json.load(f)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, f"{path.name}: no code cells"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_no_outputs_committed(path):
    """We expect students to produce outputs locally — committed outputs
    bloat diffs and leak data. Fail loudly if any code cell has outputs."""
    with open(path) as f:
        nb = json.load(f)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and c.get("outputs"):
            pytest.fail(f"{path.name} cell {i} has committed outputs")
