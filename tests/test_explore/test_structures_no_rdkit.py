"""Test render_structures graceful degradation when RDKit is absent."""

from __future__ import annotations

import sys
import warnings

import pandas as pd
import pytest


def test_render_without_rdkit_returns_empty_with_warning(monkeypatch):
    # Block rdkit from being imported.
    monkeypatch.setitem(sys.modules, "rdkit", None)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", None)

    # Force reimport of structures module.
    if "metabproject.explore.structures" in sys.modules:
        del sys.modules["metabproject.explore.structures"]

    from metabproject.explore.structures import render_structures

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = render_structures(pd.Series(["C"]))

    assert result == {}
    assert any("RDKit" in str(warning.message) for warning in w)
