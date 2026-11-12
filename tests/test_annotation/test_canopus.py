"""Tests for metabproject.annotation.canopus."""

import textwrap

import pandas as pd
import pytest

from metabproject.annotation.canopus import load_canopus

_CANOPUS_TSV = textwrap.dedent("""\
    mappingFeatureId\tNPC#pathway\tNPC#pathway Probability\tNPC#superclass\tNPC#superclass Probability\tNPC#class\tNPC#class Probability\tClassyFire#most specific class\tClassyFire#most specific class Probability
    1\tLipids\t0.90\tFatty acids\t0.85\tFA esters\t0.75\tWax esters\t0.72
    2\tAlkaloids\t0.80\tPyridine alkaloids\t0.65\tPiperidine\t0.50\tPiperidine alkaloids\t0.40
    3\tFlavonoids\t0.60\tFlavones\t0.55\tFlavone\t0.45\tFlavone C-glycosides\t0.30
    4\tTerpenoids\t0.95\tSesquiterpenes\t0.90\tEudesmane\t0.88\tEudesmane sesquiterpenes\t0.85
""")


@pytest.fixture()
def canopus_file(tmp_path):
    p = tmp_path / "canopus_structure_summary.tsv"
    p.write_text(_CANOPUS_TSV)
    return p


def test_column_renaming(canopus_file):
    df = load_canopus(canopus_file)
    assert "canopus_npc_pathway" in df.columns
    assert "canopus_npc_pathway_probability" in df.columns
    assert "canopus_npc_class" in df.columns
    assert "canopus_classyfire_class" in df.columns
    assert "NPC#pathway" not in df.columns


def test_passed_flags_default_threshold(canopus_file):
    df = load_canopus(canopus_file, min_probability=0.7)
    # Feature 1: pathway 0.90 → passed; class 0.75 → passed
    assert df.loc[1, "canopus_npc_pathway_passed"] == True
    assert df.loc[1, "canopus_npc_class_passed"] == True
    # Feature 2: pathway 0.80 → passed; class 0.50 → not passed
    assert df.loc[2, "canopus_npc_pathway_passed"] == True
    assert df.loc[2, "canopus_npc_class_passed"] == False
    # Feature 3: pathway 0.60 → not passed
    assert df.loc[3, "canopus_npc_pathway_passed"] == False


def test_threshold_is_configurable(canopus_file):
    df_low = load_canopus(canopus_file, min_probability=0.5)
    df_high = load_canopus(canopus_file, min_probability=0.9)
    # At 0.5, feature 2 class should pass
    assert df_low.loc[2, "canopus_npc_class_passed"] == True
    # At 0.9, feature 4 (0.95) passes; feature 3 (0.60) does not
    assert df_high.loc[4, "canopus_npc_pathway_passed"] == True
    assert df_high.loc[3, "canopus_npc_pathway_passed"] == False  # 0.60 < 0.9


def test_raw_values_always_stored(canopus_file):
    """Raw taxonomy is stored regardless of threshold — gating is in passed flags."""
    df = load_canopus(canopus_file, min_probability=0.99)
    # All passed flags should be False but taxonomy values still present
    assert df["canopus_npc_pathway_passed"].sum() == 0
    assert df["canopus_npc_pathway"].notna().sum() == 4


def test_index_is_feature_id(canopus_file):
    df = load_canopus(canopus_file)
    assert df.index.name == "feature_id"
    assert df.index.dtype == int


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_canopus(tmp_path / "missing.tsv")


def test_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("col_a\tcol_b\n1\t2\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_canopus(bad)


def test_loads_classyfire_subclass_and_level5_when_present(tmp_path):
    """Newer CANOPUS exports include ClassyFire subclass + level 5 — they
    are picked up with their probability + passed flags."""
    tsv = textwrap.dedent("""\
        mappingFeatureId\tNPC#pathway\tNPC#pathway Probability\tNPC#superclass\tNPC#superclass Probability\tNPC#class\tNPC#class Probability\tClassyFire#most specific class\tClassyFire#most specific class Probability\tClassyFire#subclass\tClassyFire#subclass Probability\tClassyFire#level 5\tClassyFire#level 5 Probability
        1\tLipids\t0.90\tFatty acids\t0.85\tFA esters\t0.75\tWax esters\t0.72\tFatty acid esters\t0.80\tLong-chain wax esters\t0.65
    """)
    p = tmp_path / "canopus_structure_summary.tsv"
    p.write_text(tsv)
    df = load_canopus(p, min_probability=0.7)
    assert "canopus_classyfire_subclass" in df.columns
    assert "canopus_classyfire_level5" in df.columns
    assert df.loc[1, "canopus_classyfire_subclass"] == "Fatty acid esters"
    assert df.loc[1, "canopus_classyfire_level5"] == "Long-chain wax esters"
    assert df.loc[1, "canopus_classyfire_subclass_passed"] == True   # 0.80 >= 0.7
    assert df.loc[1, "canopus_classyfire_level5_passed"] == False    # 0.65 <  0.7
