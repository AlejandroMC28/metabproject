"""Tests for explore/structures.py."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit not installed")


def test_render_valid_smiles():
    from metabproject.explore.structures import render_structures

    smi = "OC(=O)/C=C/c1ccc(O)c(O)c1"  # caffeic acid
    result = render_structures(pd.Series([smi]))
    assert smi in result
    assert "<svg" in result[smi].lower()


def test_render_deduplicates():
    from metabproject.explore.structures import render_structures

    smi = "C"  # methane
    result = render_structures(pd.Series([smi, smi, smi]))
    assert len(result) == 1


def test_render_invalid_smiles_omitted():
    from metabproject.explore.structures import render_structures

    result = render_structures(pd.Series(["not_a_smiles_XXXX", "C"]))
    assert "not_a_smiles_XXXX" not in result
    assert "C" in result


def test_render_null_omitted():
    from metabproject.explore.structures import render_structures

    import numpy as np

    result = render_structures(pd.Series([None, float("nan"), "C"]))
    assert "C" in result
    assert None not in result


def test_render_empty_series():
    from metabproject.explore.structures import render_structures

    result = render_structures(pd.Series([], dtype=str))
    assert result == {}
