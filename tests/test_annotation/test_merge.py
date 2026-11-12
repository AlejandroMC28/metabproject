"""Tests for metabproject.annotation.merge."""

import pandas as pd
import pytest

from metabproject.annotation.merge import merge_annotations


# ---------------------------------------------------------------------------
# Helpers to build minimal per-source DataFrames
# ---------------------------------------------------------------------------

def _sirius(feature_id, compound_name, inchikey2d, mol_formula, adduct,
            confidence, high_confidence):
    idx = pd.Index([feature_id], name="feature_id")
    return pd.DataFrame({
        "sirius_compound_name": [compound_name],
        "sirius_InChIkey2D": [inchikey2d],
        "sirius_mol_formula": [mol_formula],
        "sirius_adduct": [adduct],
        "sirius_confidence": [confidence],
        "sirius_high_confidence": [high_confidence],
        "sirius_InChI": [None],
        "sirius_smiles": [None],
    }, index=idx)


def _canopus(feature_id, pathway=None, pathway_prob=0.9,
             npc_class=None, class_prob=0.8,
             classyfire=None, cf_prob=0.8):
    idx = pd.Index([feature_id], name="feature_id")
    return pd.DataFrame({
        "canopus_npc_pathway": [pathway],
        "canopus_npc_pathway_probability": [pathway_prob],
        "canopus_npc_pathway_passed": [pathway is not None and pathway_prob >= 0.7],
        "canopus_npc_superclass": [None],
        "canopus_npc_superclass_probability": [None],
        "canopus_npc_superclass_passed": [False],
        "canopus_npc_class": [npc_class],
        "canopus_npc_class_probability": [class_prob],
        "canopus_npc_class_passed": [npc_class is not None and class_prob >= 0.7],
        "canopus_classyfire_class": [classyfire],
        "canopus_classyfire_class_probability": [cf_prob],
        "canopus_classyfire_class_passed": [classyfire is not None and cf_prob >= 0.7],
    }, index=idx)


def _gnps(feature_id, compound_name, inchikey2d, npc_class=None,
          classyfire=None, mq_score=0.8):
    idx = pd.Index([feature_id], name="feature_id")
    return pd.DataFrame({
        "gnps_compound_name": [compound_name],
        "gnps_InChIkey2D": [inchikey2d],
        "gnps_smiles": [None],
        "gnps_InChI": [None],
        "gnps_MQScore": [mq_score],
        "gnps_npc_pathway": [None],
        "gnps_npc_superclass": [None],
        "gnps_npc_class": [npc_class],
        "gnps_classyfire_class": [classyfire],
    }, index=idx)


def _suspect(feature_id, compound_name):
    idx = pd.Index([feature_id], name="feature_id")
    return pd.DataFrame({
        "suspect_compound_name": [compound_name],
        "suspect_formula_sirius": [None],
        "suspect_formula_buddy": [None],
        "suspect_delta_mz": [None],
        "suspect_adduct": [None],
    }, index=idx)


def _empty_sirius():
    idx = pd.Index([], name="feature_id")
    return pd.DataFrame(columns=["sirius_compound_name", "sirius_high_confidence",
                                  "sirius_InChIkey2D"], index=idx)


def _empty_gnps():
    idx = pd.Index([], name="feature_id")
    return pd.DataFrame(columns=["gnps_compound_name", "gnps_InChIkey2D"], index=idx)


def _empty_canopus():
    idx = pd.Index([], name="feature_id")
    return pd.DataFrame(columns=["canopus_npc_pathway", "canopus_npc_pathway_passed",
                                  "canopus_npc_class", "canopus_npc_class_passed",
                                  "canopus_classyfire_class", "canopus_classyfire_class_passed"],
                        index=idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sirius_only_high_conf():
    s = _sirius(1, "CompA", "AAAA", "C2H6", "[M+H]+", 0.80, True)
    c = _empty_canopus()
    g = _empty_gnps()
    result = merge_annotations(s, c, g)
    assert 1 in result.index
    assert result.loc[1, "annotation_source"] == "sirius"
    assert result.loc[1, "compound_name"] == "CompA"
    assert result.loc[1, "inchikey_agreement"] == "single_source"
    assert result.loc[1, "is_canopus_only"] == False


def test_sirius_only_low_conf_excluded():
    s = _sirius(2, "LowConf", "BBBB", "C3H8", "[M+Na]+", 0.40, False)
    c = _empty_canopus()
    g = _empty_gnps()
    result = merge_annotations(s, c, g)
    assert 2 not in result.index


def test_gnps_only():
    s = _empty_sirius()
    c = _empty_canopus()
    g = _gnps(3, "CompB", "CCCC")
    result = merge_annotations(s, c, g)
    assert 3 in result.index
    assert result.loc[3, "annotation_source"] == "gnps"
    assert result.loc[3, "inchikey_agreement"] == "single_source"


def test_canopus_only():
    s = _empty_sirius()
    c = _canopus(4, pathway="Lipids", npc_class="FA esters", classyfire="Wax esters")
    g = _empty_gnps()
    result = merge_annotations(s, c, g)
    assert 4 in result.index
    assert result.loc[4, "annotation_source"] == "canopus"
    assert result.loc[4, "is_canopus_only"] == True
    assert result.loc[4, "npc_pathway"] == "Lipids"
    assert result.loc[4, "npc_class"] == "FA esters"
    # No structural data from CANOPUS
    assert pd.isna(result.loc[4, "compound_name"])
    assert pd.isna(result.loc[4, "inchikey_agreement"])


def test_canopus_low_conf_excluded():
    """Feature with all CANOPUS probabilities below threshold → not in merged."""
    s = _empty_sirius()
    c = _canopus(5, pathway="Lipids", pathway_prob=0.3, npc_class="FA esters",
                 class_prob=0.2, classyfire="Wax esters", cf_prob=0.1)
    g = _empty_gnps()
    result = merge_annotations(s, c, g)
    assert 5 not in result.index


def test_canopus_partial_level_gating():
    """Feature with pathway passing but class failing → pathway populated, class NA."""
    s = _empty_sirius()
    c = _canopus(6, pathway="Alkaloids", pathway_prob=0.85,
                 npc_class="Piperidine", class_prob=0.50)  # class below 0.7
    g = _empty_gnps()
    result = merge_annotations(s, c, g)
    assert 6 in result.index
    assert result.loc[6, "npc_pathway"] == "Alkaloids"
    assert pd.isna(result.loc[6, "npc_class"])


def test_low_conf_sirius_with_canopus():
    """Low-confidence SIRIUS structural ID + CANOPUS taxonomy → included via canopus."""
    s = _sirius(7, "LowConf", "LLLL", "C4H8", "[M+H]+", 0.30, False)
    c = _canopus(7, pathway="Terpenoids", npc_class="Monoterpenes")
    g = _empty_gnps()
    result = merge_annotations(s, c, g)
    assert 7 in result.index
    assert result.loc[7, "annotation_source"] == "canopus"
    assert result.loc[7, "npc_pathway"] == "Terpenoids"


def test_suspect_only():
    s = _empty_sirius()
    c = _empty_canopus()
    g = _empty_gnps()
    sus = _suspect(8, "TentativeCompound")
    result = merge_annotations(s, c, g, sus)
    assert 8 in result.index
    assert result.loc[8, "annotation_source"] == "suspect"
    assert result.loc[8, "is_suspect_only"] == True
    assert pd.isna(result.loc[8, "inchikey_agreement"])


def test_gnps_and_sirius_agree():
    s = _sirius(9, "CompC_S", "SAME_KEY", "C10H20O2", "[M+H]+", 0.80, True)
    c = _empty_canopus()
    g = _gnps(9, "CompC_G", "SAME_KEY")
    result = merge_annotations(s, c, g)
    assert result.loc[9, "inchikey_agreement"] == "agree"
    assert result.loc[9, "compound_name"] == "CompC_G"  # GNPS preferred
    assert result.loc[9, "compound_name_gnps"] == "CompC_G"
    assert result.loc[9, "compound_name_sirius"] == "CompC_S"
    assert result.loc[9, "inchikey2d_gnps"] == "SAME_KEY"
    assert result.loc[9, "inchikey2d_sirius"] == "SAME_KEY"


def test_gnps_and_sirius_disagree():
    s = _sirius(10, "SIRIUS_compound", "SIRIUS_KEY", "C10H20O", "[M+H]+", 0.80, True)
    c = _empty_canopus()
    g = _gnps(10, "GNPS_compound", "GNPS_KEY")
    result = merge_annotations(s, c, g)
    assert result.loc[10, "inchikey_agreement"] == "disagree"
    assert result.loc[10, "compound_name"] == "GNPS_compound"
    assert result.loc[10, "compound_name_gnps"] == "GNPS_compound"
    assert result.loc[10, "compound_name_sirius"] == "SIRIUS_compound"
    assert result.loc[10, "inchikey2d_gnps"] == "GNPS_KEY"
    assert result.loc[10, "inchikey2d_sirius"] == "SIRIUS_KEY"


def test_per_source_columns_populated_when_both_present():
    s = _sirius(20, "SIRIUS_name", "ABCD", "C5H10", "[M+H]+", 0.80, True)
    c = _empty_canopus()
    g = _gnps(20, "GNPS_name", "EFGH")
    result = merge_annotations(s, c, g)
    assert result.loc[20, "compound_name_gnps"] == "GNPS_name"
    assert result.loc[20, "compound_name_sirius"] == "SIRIUS_name"
    assert result.loc[20, "compound_name"] == "GNPS_name"
    assert result.loc[20, "sirius_high_confidence"] == True


def test_sirius_column_shows_low_conf_value():
    """Low-conf SIRIUS value should appear in _sirius columns; best-pick stays NA."""
    s = _sirius(21, "LowConfName", "LLLL", "C4H8", "[M+H]+", 0.30, False)
    c = _empty_canopus()
    g = _empty_gnps()
    result = merge_annotations(s, c, g)
    # Feature not in output (no high-conf SIRIUS, no GNPS, no CANOPUS)
    assert 21 not in result.index


def test_sirius_column_shows_low_conf_value_when_gnps_present():
    """With GNPS + low-conf SIRIUS: _sirius column shows low-conf value, best-pick = GNPS."""
    s = _sirius(22, "LowConfSIRIUS", "SIRIUS_LC", "C4H8", "[M+H]+", 0.30, False)
    c = _empty_canopus()
    g = _gnps(22, "GNPS_name", "GNPS_KEY")
    result = merge_annotations(s, c, g)
    assert result.loc[22, "compound_name_gnps"] == "GNPS_name"
    assert result.loc[22, "compound_name_sirius"] == "LowConfSIRIUS"
    assert result.loc[22, "compound_name"] == "GNPS_name"
    assert result.loc[22, "sirius_high_confidence"] == False


def test_sirius_column_na_when_no_sirius_hit():
    s = _empty_sirius()
    c = _empty_canopus()
    g = _gnps(23, "GNPS_only", "GNPS_KEY")
    result = merge_annotations(s, c, g)
    assert result.loc[23, "compound_name_gnps"] == "GNPS_only"
    assert pd.isna(result.loc[23, "compound_name_sirius"])


def test_inchikey_agreement_ignores_low_conf_sirius():
    """GNPS + low-conf SIRIUS → single_source, not agree/disagree."""
    s = _sirius(24, "LowConf", "SIRIUS_IK", "C4H8", "[M+H]+", 0.30, False)
    c = _empty_canopus()
    g = _gnps(24, "GNPS_name", "GNPS_IK")
    result = merge_annotations(s, c, g)
    assert result.loc[24, "inchikey_agreement"] == "single_source"


def test_structural_alt_columns_removed():
    """Structural *_alt columns must NOT be present; taxonomy *_alt still are."""
    s = _sirius(25, "S", "SIRIUS_IK", "C2H6", "[M+H]+", 0.80, True)
    c = _empty_canopus()
    g = _gnps(25, "G", "GNPS_IK")
    result = merge_annotations(s, c, g)
    for col in ("compound_name_alt", "inchikey2d_alt", "smiles_alt", "mol_formula_alt"):
        assert col not in result.columns, f"unexpected column: {col}"
    assert "npc_class_alt" in result.columns
    assert "classyfire_class_alt" in result.columns


def test_gnps_preferred_over_canopus_taxonomy():
    s = _empty_sirius()
    c = _canopus(11, pathway="Alkaloids", npc_class="Piperidine")
    g = _gnps(11, "CompD", "DDDD", npc_class="Lipids")
    result = merge_annotations(s, c, g)
    # GNPS taxonomy preferred; CANOPUS goes to alt
    assert result.loc[11, "npc_class"] == "Lipids"
    assert result.loc[11, "npc_class_alt"] == "Piperidine"


def test_all_sources():
    s = _sirius(12, "CompE_S", "EEEE", "C8H8", "[M+H]+", 0.75, True)
    c = _canopus(12, pathway="Flavonoids", npc_class="Flavones")
    g = _gnps(12, "CompE_G", "EEEE")
    sus = _suspect(12, "TentativeE")
    result = merge_annotations(s, c, g, sus)
    sources = set(result.loc[12, "annotation_source"].split(":"))
    assert sources == {"gnps", "sirius", "canopus", "suspect"}


def test_unannotated_absent():
    s = _sirius(13, "LowConf", "XXXX", "C4H8", "[M+H]+", 0.20, False)
    c = _canopus(13, pathway="Terpenes", pathway_prob=0.20)  # below threshold
    g = _empty_gnps()
    result = merge_annotations(s, c, g)
    assert 13 not in result.index
