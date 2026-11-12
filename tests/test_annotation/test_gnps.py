"""Tests for metabproject.annotation.gnps."""

import textwrap

import pandas as pd
import pytest

from metabproject.annotation.gnps import _parse_suspect_name, load_gnps_library, load_gnps_suspect

# ---------------------------------------------------------------------------
# Synthetic fixture data
# ---------------------------------------------------------------------------

_GNPS_LIBRARY_TSV = textwrap.dedent("""\
    #Scan#\tCompound_Name\tSmiles\tINSCHI\tInChIKey-Planar\tMQScore\tnpclassifier_pathway\tnpclassifier_superclass\tnpclassifier_class\tCF_NPC_class
    10\tCaffeine\tCc1cnc2n1c(=O)n(C)c(=O)n2C\tInChI=1/C8H10N4O2\tGSVVHLFALELDLRNL\t0.85\tAlkaloids\tPurine alkaloids\tMethylxanthines\tMethylxanthines
    20\tGlucose\tOC[C@H]1OC(O)\tInChI=1/C6H12O6\tWQZGKKKJIJFFOK-H\t0.72\tCarbohydrates\tMonosaccharides\tHexoses\tHexoses
""")

_GNPS_SUSPECT_TSV = textwrap.dedent("""\
    #Scan#\tCompound_Name\tLibraryName
    10\tCaffeine\tGNPS-LIBRARY.mgf
    30\tMyCompound | SIRIUS: C10H12O | BUDDY: C10H12O2 | Δmz: 0.002 | adduct: [M+H]+\tGNPS-SUSPECTLIST.mgf
    40\tSimpleName | SIRIUS: C5H6\tGNPS-SUSPECTLIST.mgf
    50\tNoFormula\tGNPS-SUSPECTLIST.mgf
""")


@pytest.fixture()
def gnps_library_file(tmp_path):
    p = tmp_path / "gnps_library.tsv"
    p.write_text(_GNPS_LIBRARY_TSV)
    return p


@pytest.fixture()
def gnps_suspect_file(tmp_path):
    p = tmp_path / "gnps_suspect.tsv"
    p.write_text(_GNPS_SUSPECT_TSV)
    return p


# -- Library tests -----------------------------------------------------------

def test_library_feature_id_rename(gnps_library_file):
    df = load_gnps_library(gnps_library_file)
    assert df.index.name == "feature_id"
    assert 10 in df.index
    assert 20 in df.index


def test_library_column_prefixing(gnps_library_file):
    df = load_gnps_library(gnps_library_file)
    assert all(c.startswith("gnps_") for c in df.columns)


def test_library_inchikey_rename(gnps_library_file):
    df = load_gnps_library(gnps_library_file)
    assert "gnps_InChIkey2D" in df.columns
    assert "InChIKey-Planar" not in df.columns


def test_library_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gnps_library(tmp_path / "nope.tsv")


def test_library_missing_required_columns_raises(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("col_a\tcol_b\n1\t2\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_gnps_library(bad)


# -- Suspect tests -----------------------------------------------------------

def test_suspect_library_name_filter(gnps_suspect_file):
    # Feature 10 is in GNPS-LIBRARY.mgf — should be excluded.
    df = load_gnps_suspect(gnps_suspect_file)
    assert 10 not in df.index
    assert 30 in df.index


def test_suspect_dedup_against_library(gnps_suspect_file):
    # Exclude feature 30 as if it were already in the library.
    exclude = pd.Index([30])
    df = load_gnps_suspect(gnps_suspect_file, exclude_feature_ids=exclude)
    assert 30 not in df.index
    assert 40 in df.index


def test_suspect_column_prefixing(gnps_suspect_file):
    df = load_gnps_suspect(gnps_suspect_file)
    assert all(c.startswith("suspect_") for c in df.columns)


# -- Compound_Name regex parsing tests ---------------------------------------

def test_parse_full():
    result = _parse_suspect_name(
        "MyCompound | SIRIUS: C10H12O | BUDDY: C10H12O2 | Δmz: 0.002 | adduct: [M+H]+"
    )
    assert result["compound_name"] == "MyCompound"
    assert result["formula_sirius"] == "C10H12O"
    assert result["formula_buddy"] == "C10H12O2"
    assert abs(result["delta_mz"] - 0.002) < 1e-9
    assert result["adduct"] == "[M+H]+"


def test_parse_partial():
    result = _parse_suspect_name("SimpleName | SIRIUS: C5H6")
    assert result["compound_name"] == "SimpleName"
    assert result["formula_sirius"] == "C5H6"
    assert result["formula_buddy"] is None
    assert result["delta_mz"] is None


def test_parse_no_pipes():
    result = _parse_suspect_name("NoFormula")
    assert result["compound_name"] == "NoFormula"
    assert result["formula_sirius"] is None
