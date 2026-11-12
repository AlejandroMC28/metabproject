"""Tests for metabproject.annotation.sirius (structural-only)."""

import textwrap

import pytest

from metabproject.annotation.sirius import load_sirius

_STRUCTURES_TSV = textwrap.dedent("""\
    mappingFeatureId\tstructurePerIdRank\tConfidenceScoreApproximate\tInChIkey2D\tInChI\tname\tsmiles\tmolecularFormula\tadduct
    1\t1\t0.80\tAAAAAAAAAAAAAAA\tInChI=1/A\tCompound A\tCC\tC2H6\t[M+H]+
    1\t2\t0.50\tBBBBBBBBBBBBBBB\tInChI=1/B\tCompound B\tCCC\tC3H8\t[M+H]+
    2\t1\t0.40\tCCCCCCCCCCCCCCC\tInChI=1/C\tCompound C\tO\tH2O\t[M+Na]+
    3\t1\t0.70\tDDDDDDDDDDDDDDD\tInChI=1/D\tCompound D\tCC=O\tC2H4O\t[M+H]+
    4\t1\t-inf\tEEEEEEEEEEEEEEE\tInChI=1/E\tCompound E\tCO\tCH4O\t[M+H]+
""")


@pytest.fixture()
def structures_file(tmp_path):
    p = tmp_path / "structure_identifications.tsv"
    p.write_text(_STRUCTURES_TSV)
    return p


def test_rank1_filter(structures_file):
    df = load_sirius(structures_file)
    assert 1 in df.index
    assert len(df) == 4  # features 1, 2, 3, 4


def test_high_confidence_flag(structures_file):
    df = load_sirius(structures_file, min_confidence=0.64)
    assert df.loc[1, "sirius_high_confidence"] == True
    assert df.loc[2, "sirius_high_confidence"] == False
    # -inf confidence → NA → high_confidence False
    assert df.loc[4, "sirius_high_confidence"] == False


def test_inf_confidence_is_na(structures_file):
    import pandas as pd
    df = load_sirius(structures_file)
    assert pd.isna(df.loc[4, "sirius_confidence"])


def test_column_prefixing(structures_file):
    df = load_sirius(structures_file)
    expected = {"sirius_confidence", "sirius_InChIkey2D", "sirius_compound_name",
                "sirius_smiles", "sirius_mol_formula", "sirius_adduct",
                "sirius_high_confidence", "sirius_InChI"}
    assert expected.issubset(set(df.columns))
    # No taxonomy columns from CANOPUS should appear
    assert not any("npc" in c or "classyfire" in c for c in df.columns)


def test_index_is_feature_id(structures_file):
    df = load_sirius(structures_file)
    assert df.index.name == "feature_id"
    assert df.index.dtype == int


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sirius(tmp_path / "missing.tsv")


def test_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("col_a\tcol_b\n1\t2\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_sirius(bad)
