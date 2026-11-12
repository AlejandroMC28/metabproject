"""Tests for compute_gnps_sirius_topk_match and its merge integration."""

import pandas as pd
import pytest

from metabproject.annotation.sirius import compute_gnps_sirius_topk_match


def _gnps_keys(mapping: dict[int, str | None]) -> pd.DataFrame:
    idx = pd.Index(list(mapping.keys()), name="feature_id")
    return pd.DataFrame({"gnps_InChIkey2D": list(mapping.values())}, index=idx)


def _topk(rows: list[tuple[int, int, str, float | None]]) -> pd.DataFrame:
    """Build a top-k DataFrame from (feature_id, rank, inchikey2d, confidence) rows."""
    fids = [r[0] for r in rows]
    idx = pd.Index(fids, name="feature_id")
    return pd.DataFrame(
        {
            "structurePerIdRank": [r[1] for r in rows],
            "sirius_InChIkey2D": [r[2] for r in rows],
            "sirius_confidence": [r[3] for r in rows],
        },
        index=idx,
    )


def test_match_at_rank1_highconf():
    gnps = _gnps_keys({1: "KEYAAA"})
    topk = _topk([(1, 1, "KEYAAA", 0.90), (1, 2, "KEYBBB", 0.40)])
    out = compute_gnps_sirius_topk_match(gnps, topk, min_confidence=0.64)
    assert out.loc[1, "gnps_sirius_rank"] == 1
    assert out.loc[1, "gnps_sirius_highconf"] == True


def test_match_at_rank1_lowconf():
    gnps = _gnps_keys({1: "KEYAAA"})
    topk = _topk([(1, 1, "KEYAAA", 0.50)])
    out = compute_gnps_sirius_topk_match(gnps, topk, min_confidence=0.64)
    assert out.loc[1, "gnps_sirius_rank"] == 1
    assert out.loc[1, "gnps_sirius_highconf"] == False


def test_match_at_rank3():
    gnps = _gnps_keys({2: "KEYCCC"})
    topk = _topk([
        (2, 1, "KEYAAA", 0.80),
        (2, 2, "KEYBBB", 0.70),
        (2, 3, "KEYCCC", 0.65),
    ])
    out = compute_gnps_sirius_topk_match(gnps, topk, min_confidence=0.64)
    assert out.loc[2, "gnps_sirius_rank"] == 3
    assert out.loc[2, "gnps_sirius_highconf"] == True


def test_no_match_in_topk():
    gnps = _gnps_keys({3: "KEYZZZ"})
    topk = _topk([(3, 1, "KEYAAA", 0.80), (3, 2, "KEYBBB", 0.70)])
    out = compute_gnps_sirius_topk_match(gnps, topk)
    assert out.loc[3, "gnps_sirius_rank"] == 0
    assert out.loc[3, "gnps_sirius_highconf"] == False


def test_feature_absent_from_topk():
    gnps = _gnps_keys({4: "KEYAAA"})
    topk = _topk([(99, 1, "KEYAAA", 0.80)])
    out = compute_gnps_sirius_topk_match(gnps, topk)
    assert pd.isna(out.loc[4, "gnps_sirius_rank"])
    assert out.loc[4, "gnps_sirius_highconf"] == False


def test_feature_with_no_gnps_key_excluded():
    gnps = _gnps_keys({1: "KEYAAA", 2: None})
    topk = _topk([(1, 1, "KEYAAA", 0.90), (2, 1, "ANYTHING", 0.90)])
    out = compute_gnps_sirius_topk_match(gnps, topk)
    assert 1 in out.index
    assert 2 not in out.index


def test_empty_gnps_returns_empty():
    gnps = pd.DataFrame({"gnps_InChIkey2D": []}, index=pd.Index([], name="feature_id"))
    topk = _topk([(1, 1, "KEYAAA", 0.80)])
    out = compute_gnps_sirius_topk_match(gnps, topk)
    assert out.empty
    assert list(out.columns) == ["gnps_sirius_rank", "gnps_sirius_highconf"]


def test_gnps_missing_inchikey_column():
    gnps = pd.DataFrame(
        {"gnps_compound_name": ["X"]}, index=pd.Index([1], name="feature_id")
    )
    topk = _topk([(1, 1, "KEYAAA", 0.80)])
    out = compute_gnps_sirius_topk_match(gnps, topk)
    assert out.empty


def test_matched_candidate_confidence_nan_is_lowconf():
    gnps = _gnps_keys({1: "KEYAAA"})
    topk = _topk([(1, 1, "KEYAAA", None)])
    out = compute_gnps_sirius_topk_match(gnps, topk, min_confidence=0.64)
    assert out.loc[1, "gnps_sirius_rank"] == 1
    assert out.loc[1, "gnps_sirius_highconf"] == False


# --------------------------------------------------------------------------
# Integration with merge_annotations
# --------------------------------------------------------------------------

def test_merge_adds_cross_check_columns():
    from metabproject.annotation.merge import merge_annotations

    sirius_idx = pd.Index([10], name="feature_id")
    sirius_df = pd.DataFrame({
        "sirius_compound_name": ["X"],
        "sirius_InChIkey2D": ["KEYAAA"],
        "sirius_mol_formula": ["C"],
        "sirius_adduct": ["[M+H]+"],
        "sirius_confidence": [0.9],
        "sirius_high_confidence": [True],
        "sirius_InChI": [None],
        "sirius_smiles": [None],
    }, index=sirius_idx)

    canopus_df = pd.DataFrame(index=pd.Index([], name="feature_id"))

    gnps_df = pd.DataFrame({
        "gnps_compound_name": ["X"],
        "gnps_InChIkey2D": ["KEYAAA"],
        "gnps_MQScore": [0.95],
        "gnps_smiles": [None],
        "gnps_InChI": [None],
    }, index=sirius_idx)

    topk_df = pd.DataFrame({
        "structurePerIdRank": [1, 2],
        "sirius_InChIkey2D": ["KEYAAA", "KEYBBB"],
        "sirius_confidence": [0.9, 0.4],
    }, index=pd.Index([10, 10], name="feature_id"))

    merged = merge_annotations(
        sirius_df, canopus_df, gnps_df, suspect_df=None,
        sirius_topk_df=topk_df, sirius_min_confidence=0.64,
    )
    assert "gnps_sirius_rank" in merged.columns
    assert "gnps_sirius_highconf" in merged.columns
    assert merged.loc[10, "gnps_sirius_rank"] == 1
    assert merged.loc[10, "gnps_sirius_highconf"] == True


def test_merge_without_topk_omits_columns():
    from metabproject.annotation.merge import merge_annotations

    sirius_idx = pd.Index([10], name="feature_id")
    sirius_df = pd.DataFrame({
        "sirius_compound_name": ["X"],
        "sirius_InChIkey2D": ["KEYAAA"],
        "sirius_mol_formula": ["C"],
        "sirius_adduct": ["[M+H]+"],
        "sirius_confidence": [0.9],
        "sirius_high_confidence": [True],
        "sirius_InChI": [None],
        "sirius_smiles": [None],
    }, index=sirius_idx)
    canopus_df = pd.DataFrame(index=pd.Index([], name="feature_id"))
    gnps_df = pd.DataFrame({
        "gnps_compound_name": ["X"],
        "gnps_InChIkey2D": ["KEYAAA"],
        "gnps_MQScore": [0.95],
        "gnps_smiles": [None],
        "gnps_InChI": [None],
    }, index=sirius_idx)

    merged = merge_annotations(sirius_df, canopus_df, gnps_df)
    assert "gnps_sirius_rank" not in merged.columns
    assert "gnps_sirius_highconf" not in merged.columns
