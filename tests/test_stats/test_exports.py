"""Tests for stats/exports.py — bucket / pair / count-matrix exporters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metabproject.stats.comparison_runner import run_comparison
from metabproject.stats.comparisons import Comparison
from metabproject.stats.exports import (
    PAIR_TSV_COLUMNS,
    _attach_annotation,
    _orient_pairs,
    combined_factor_name,
    export_buckets,
    export_pairwise_count_matrix,
    export_tukey_pair_tsvs,
    pairwise_count_matrix,
)


SEED = 0


def _two_way_with_3_levels() -> tuple[pd.DataFrame, pd.DataFrame, Comparison]:
    rng = np.random.default_rng(SEED)
    samples, l1_col, l2_col = [], [], []
    for l1 in ("a", "b", "c"):
        for l2 in ("p", "q", "r"):
            for r in range(4):
                samples.append(f"{l1}{l2}{r + 1}")
                l1_col.append(l1)
                l2_col.append(l2)
    meta = pd.DataFrame(
        {"variety": l1_col, "treatment": l2_col},
        index=pd.Index(samples, name="sample"),
    )
    l1 = np.array(l1_col)
    l2 = np.array(l2_col)
    n = len(samples)
    rows = np.vstack([
        # factor1_only
        np.where(l1 == "a", 5.0, 0.0) + rng.normal(scale=0.3, size=n),
        # factor2_only
        np.where(l2 == "p", 5.0, 0.0) + rng.normal(scale=0.3, size=n),
        # interaction
        np.where((l1 == "a") & (l2 == "p"), 5.0, 0.0)
        + np.where((l1 == "c") & (l2 == "r"), 5.0, 0.0)
        + rng.normal(scale=0.3, size=n),
        # ns
        rng.normal(scale=1.0, size=n),
    ])
    ft = pd.DataFrame(
        rows,
        index=pd.Index(["f1", "f2", "f3", "f4"], name="feature_id"),
        columns=samples,
    )
    cmp = Comparison(
        name="x", kind="anova_2way",
        factors=["variety", "treatment"], posthoc="tukey",
    )
    return ft, meta, cmp


# ------------------------------------------------------------------------ #
# Pure helpers
# ------------------------------------------------------------------------ #

def test_combined_factor_name() -> None:
    assert combined_factor_name("variety", "treatment") == "_combined_variety_x_treatment"


def test_orient_pairs_canonical_order_and_sign_flip() -> None:
    df = pd.DataFrame({
        "group_A": ["z_p", "a_p"],
        "group_B": ["a_q", "b_q"],
        "mean_A": [10.0, 1.0],
        "mean_B": [5.0, 2.0],
        "diff": [-5.0, 1.0],
    })
    out = _orient_pairs(df)
    # First row should be flipped (z > a); second row stays.
    assert out.iloc[0]["group_A"] == "a_q"
    assert out.iloc[0]["group_B"] == "z_p"
    assert out.iloc[0]["mean_A"] == 5.0
    assert out.iloc[0]["mean_B"] == 10.0
    assert out.iloc[0]["diff"] == 5.0
    assert out.iloc[1]["group_A"] == "a_p"
    assert out.iloc[1]["diff"] == 1.0


def test_attach_annotation_missing_table_uses_NA() -> None:
    df = pd.DataFrame({"feature": ["a", "b"]})
    out = _attach_annotation(df, None)
    for col in ("ann_source", "ann_compound_name", "ann_inchikey_2d"):
        assert col in out.columns
        assert out[col].isna().all()


def test_attach_annotation_left_join() -> None:
    df = pd.DataFrame({"feature": ["a", "b", "c"]})
    ann = pd.DataFrame(
        {"annotation_source": ["GNPS", "SIRIUS"],
         "compound_name": ["abrine", "phenylalanine"],
         "inchikey2d": ["AAA", "BBB"]},
        index=pd.Index(["a", "c"]),
    )
    out = _attach_annotation(df, ann)
    assert out.set_index("feature").loc["a", "ann_source"] == "GNPS"
    assert out.set_index("feature").loc["c", "ann_compound_name"] == "phenylalanine"
    assert pd.isna(out.set_index("feature").loc["b", "ann_source"])


def test_attach_annotation_carries_taxonomy_columns() -> None:
    df = pd.DataFrame({"feature": ["a", "b"]})
    ann = pd.DataFrame(
        {
            "annotation_source": ["GNPS", "canopus"],
            "compound_name": ["abrine", pd.NA],
            "inchikey2d": ["AAA", pd.NA],
            "npc_pathway": ["Alkaloids", "Lipids"],
            "npc_superclass": ["Indole", "Fatty acids"],
            "npc_class": ["Tryptamines", "FA esters"],
            "classyfire_class": ["Indolyl", "Wax esters"],
            "classyfire_subclass": ["Indol", "Long-chain"],
            "classyfire_level5": [pd.NA, "C20+ wax"],
        },
        index=pd.Index(["a", "b"]),
    )
    out = _attach_annotation(df, ann).set_index("feature")
    assert out.loc["a", "ann_npc_pathway"] == "Alkaloids"
    assert out.loc["b", "ann_classyfire_subclass"] == "Long-chain"
    assert out.loc["b", "ann_classyfire_level5"] == "C20+ wax"
    assert pd.isna(out.loc["a", "ann_classyfire_level5"])


def test_attach_annotation_fills_missing_taxonomy_with_NA() -> None:
    """An older annotation table without classyfire_subclass / level5 still
    yields a stable pair-TSV schema with the missing columns filled NA."""
    df = pd.DataFrame({"feature": ["a"]})
    ann = pd.DataFrame(
        {"annotation_source": ["GNPS"], "compound_name": ["x"], "inchikey2d": ["AAA"]},
        index=pd.Index(["a"]),
    )
    out = _attach_annotation(df, ann)
    for col in (
        "ann_npc_pathway", "ann_npc_superclass", "ann_npc_class",
        "ann_classyfire_class", "ann_classyfire_subclass", "ann_classyfire_level5",
    ):
        assert col in out.columns
        assert out[col].isna().all()


# ------------------------------------------------------------------------ #
# export_buckets
# ------------------------------------------------------------------------ #

def test_export_buckets_writes_per_bucket_tsv(tmp_path: Path) -> None:
    ft, meta, cmp = _two_way_with_3_levels()
    result = run_comparison(cmp, ft, meta)
    counts = export_buckets(
        result.omnibus, ft, meta, ["variety", "treatment"], tmp_path,
    )
    assert sum(counts.values()) == len(result.omnibus)
    # Read back any non-empty bucket file and verify enrichment cols.
    for bucket, n in counts.items():
        path = tmp_path / f"bucket_{bucket}.tsv"
        if n == 0:
            assert not path.exists()
            continue
        df = pd.read_csv(path, sep="\t")
        assert (df["classification"] == bucket).all()
        assert any(col.startswith("a_") for col in df.columns) or any(
            col in df.columns for col in ["max_cell", "min_cell", "range_cell"]
        )


# ------------------------------------------------------------------------ #
# export_tukey_pair_tsvs
# ------------------------------------------------------------------------ #

def test_export_tukey_pair_tsvs_writes_schema(tmp_path: Path) -> None:
    ft, meta, cmp = _two_way_with_3_levels()
    result = run_comparison(cmp, ft, meta)
    ann = pd.DataFrame(
        {"annotation_source": ["GNPS"], "compound_name": ["foo"], "inchikey2d": ["X"]},
        index=pd.Index(["f3"]),
    )
    written = export_tukey_pair_tsvs(
        result.posthoc, result.omnibus, tmp_path,
        ["variety", "treatment"], annotation_table=ann,
    )
    assert (tmp_path / "crosscorner_all.tsv").exists()
    assert any(name.startswith("pair_within_variety_") for name in written)
    assert any(name.startswith("pair_within_treatment_") for name in written)

    # Pick any within-* file and check the schema.
    sample = next(
        p for p in tmp_path.glob("pair_within_variety_*.tsv")
        if p.stat().st_size > 0
    )
    df = pd.read_csv(sample, sep="\t")
    assert list(df.columns) == PAIR_TSV_COLUMNS

    cc = pd.read_csv(tmp_path / "crosscorner_all.tsv", sep="\t")
    assert "pair" in cc.columns
    # group_A < group_B canonicalisation.
    assert (cc["group_A"] < cc["group_B"]).all()


# ------------------------------------------------------------------------ #
# pairwise_count_matrix
# ------------------------------------------------------------------------ #

def test_pairwise_count_matrix_symmetric_zero_diagonal(tmp_path: Path) -> None:
    ft, meta, cmp = _two_way_with_3_levels()
    result = run_comparison(cmp, ft, meta)
    mat = pairwise_count_matrix(result.posthoc, ["variety", "treatment"])
    assert mat.shape[0] == mat.shape[1]
    np.testing.assert_array_equal(mat.values, mat.values.T)
    assert (np.diag(mat.values) == 0).all()

    out = tmp_path / "pwc.tsv"
    saved = export_pairwise_count_matrix(
        result.posthoc, out, ["variety", "treatment"],
    )
    assert out.exists()
    pd.testing.assert_frame_equal(saved, mat)


# ------------------------------------------------------------------------ #
# run_comparison with export flags
# ------------------------------------------------------------------------ #

def test_run_comparison_dispatches_exports(tmp_path: Path) -> None:
    ft, meta, cmp = _two_way_with_3_levels()
    out = tmp_path / "comp"
    result = run_comparison(
        cmp, ft, meta,
        output_dir=out,
        export_buckets=True, export_pairs=True, count_matrix=True,
        plot_features=False,
    )
    assert result.bucket_dir == out / "buckets"
    assert result.pair_dir == out / "pairs"
    assert result.count_matrix_path == out / "pairwise_count_matrix.tsv"
    assert result.bucket_dir.is_dir()
    assert (result.pair_dir / "crosscorner_all.tsv").exists()
    assert result.count_matrix_path.exists()
    assert "buckets" in result.export_summary
    assert "pairs" in result.export_summary


def test_run_comparison_export_requires_2way() -> None:
    ft, meta, _ = _two_way_with_3_levels()
    cmp1 = Comparison(
        name="x", kind="anova_1way", factors=["variety"], posthoc="tukey",
    )
    with pytest.raises(ValueError, match="anova_2way"):
        run_comparison(
            cmp1, ft, meta, output_dir="x", export_buckets=True,
        )


def test_run_comparison_export_requires_output_dir() -> None:
    ft, meta, cmp = _two_way_with_3_levels()
    with pytest.raises(ValueError, match="output_dir"):
        run_comparison(cmp, ft, meta, export_buckets=True)
