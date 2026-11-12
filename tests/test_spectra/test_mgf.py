"""Tests for metabproject.spectra.mgf."""

import textwrap

import pandas as pd
import pytest

pytest.importorskip("pyteomics")

from metabproject.spectra.mgf import load_mgf_spectra

# ---------------------------------------------------------------------------
# Synthetic fixture data
# ---------------------------------------------------------------------------

# Three blocks mirroring the MZmine IIMN-FBMN export: a multi-peak spectrum,
# a spectrum with no CHARGE line (charge absent → nullable), and a spectrum
# keyed only by SCANS (no FEATURE_ID → falls back to SCANS).
_MGF = textwrap.dedent("""\
    BEGIN IONS
    FEATURE_ID=10
    PEPMASS=345.1234
    CHARGE=1+
    MSLEVEL=2
    SCANS=10
    Num peaks=3
    100.05 500
    150.10 1000
    200.20 250
    END IONS
    BEGIN IONS
    FEATURE_ID=20
    PEPMASS=200.9876
    MSLEVEL=2
    SCANS=20
    Num peaks=1
    99.0 42
    END IONS
    BEGIN IONS
    PEPMASS=410.0
    CHARGE=2+
    MSLEVEL=2
    SCANS=30
    Num peaks=2
    120.0 10
    130.0 20
    END IONS
""")


@pytest.fixture
def mgf_file(tmp_path):
    p = tmp_path / "spectra.mgf"
    p.write_text(_MGF)
    return p


def test_schema_and_index(mgf_file):
    df = load_mgf_spectra(mgf_file)
    assert df.index.name == "feature_id"
    assert list(df.columns) == [
        "precursor_mz",
        "precursor_charge",
        "scan",
        "n_peaks",
        "mz",
        "intensity",
    ]
    assert df.index.tolist() == [10, 20, 30]


def test_precursor_and_peaks(mgf_file):
    df = load_mgf_spectra(mgf_file)
    row = df.loc[10]
    assert row["precursor_mz"] == pytest.approx(345.1234)
    assert row["n_peaks"] == 3
    assert row["mz"] == [100.05, 150.10, 200.20]
    assert row["intensity"] == [500.0, 1000.0, 250.0]


def test_missing_charge_is_nullable(mgf_file):
    df = load_mgf_spectra(mgf_file)
    assert df.loc[10, "precursor_charge"] == 1
    assert pd.isna(df.loc[20, "precursor_charge"])


def test_id_falls_back_to_scans(mgf_file):
    df = load_mgf_spectra(mgf_file)
    # Third block has no FEATURE_ID → keyed by SCANS=30.
    assert 30 in df.index
    assert df.loc[30, "precursor_charge"] == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mgf_spectra(tmp_path / "nope.mgf")


def test_empty_mgf_raises(tmp_path):
    p = tmp_path / "empty.mgf"
    p.write_text("")
    with pytest.raises(ValueError, match="No spectra found"):
        load_mgf_spectra(p)
