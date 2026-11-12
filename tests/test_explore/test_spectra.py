"""Tests for metabproject.explore.spectra.process_spectrum_for_display."""

from __future__ import annotations

import sys

import pytest

from metabproject.explore.spectra import process_spectrum_for_display


def test_returns_base_peak_normalised():
    out = process_spectrum_for_display(
        mz=[100.0, 150.0, 200.0],
        intensity=[500.0, 1000.0, 250.0],
        precursor_mz=345.0,
        precursor_charge=1,
    )
    assert out is not None
    assert out["precursor_mz"] == pytest.approx(345.0)
    assert out["precursor_charge"] == 1
    # Base peak scaled to 1.0.
    assert max(out["intensity_rel"]) == pytest.approx(1.0)
    assert out["base_peak_mz"] == pytest.approx(150.0)
    assert len(out["mz"]) == out["n_peaks"]


def test_none_charge_defaults_to_one():
    out = process_spectrum_for_display(
        mz=[100.0], intensity=[10.0], precursor_mz=200.0, precursor_charge=None
    )
    assert out is not None
    assert out["precursor_charge"] == 1


def test_empty_peaks_returns_none():
    assert (
        process_spectrum_for_display(
            mz=[], intensity=[], precursor_mz=200.0, precursor_charge=1
        )
        is None
    )


def test_falls_back_to_raw_without_spectrum_utils(monkeypatch):
    # Block spectrum_utils from importing.
    monkeypatch.setitem(sys.modules, "spectrum_utils", None)
    monkeypatch.setitem(sys.modules, "spectrum_utils.spectrum", None)

    out = process_spectrum_for_display(
        mz=[100.0, 150.0],
        intensity=[400.0, 800.0],
        precursor_mz=345.0,
        precursor_charge=1,
    )
    assert out is not None
    # Raw peaks kept, normalised to base peak.
    assert out["n_peaks"] == 2
    assert out["mz"] == [100.0, 150.0]
    assert out["intensity_rel"] == pytest.approx([0.5, 1.0])
