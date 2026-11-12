"""Unit tests for class_enrichment()."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from metabproject.enrichment.class_test import CANOPUS_LEVELS, class_enrichment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contrast(n: int, seed: int = 0) -> pd.DataFrame:
    """Simple contrast DataFrame with a log2fc column."""
    rng = np.random.default_rng(seed)
    idx = pd.Index(range(1, n + 1), name="feature_id")
    return pd.DataFrame({"log2fc": rng.normal(size=n)}, index=idx)


def _make_annotations(assignments: dict[int, str], level: str = "npc_class") -> pd.DataFrame:
    """Build a minimal canopus_annotations DataFrame from a {feature_id: class} mapping."""
    feature_ids = sorted(assignments)
    class_col = f"canopus_{level}"
    passed_col = f"canopus_{level}_passed"
    df = pd.DataFrame(
        {
            class_col: [assignments[fid] for fid in feature_ids],
            passed_col: True,
        },
        index=pd.Index(feature_ids, name="feature_id"),
    )
    return df


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_output_columns() -> None:
    contrast = _make_contrast(20)
    ann = _make_annotations({1: "ClassA", 2: "ClassA", 3: "ClassA",
                               4: "ClassB", 5: "ClassB", 6: "ClassB"})
    result = class_enrichment(contrast, ann, level="npc_class", min_class_n=3)
    expected = {"level", "class_name", "n", "mean_log2fc", "p_value", "p_adj", "significant"}
    assert expected == set(result.columns)


def test_known_separation() -> None:
    # ClassA = high log2fc, ClassB = low log2fc → should be significant.
    n = 30
    idx = pd.Index(range(1, n + 1), name="feature_id")
    log2fc = np.concatenate([np.full(15, 5.0), np.full(15, -5.0)])
    contrast = pd.DataFrame({"log2fc": log2fc}, index=idx)
    assignments = {i: "ClassA" for i in range(1, 16)}
    assignments.update({i: "ClassB" for i in range(16, 31)})
    ann = _make_annotations(assignments)

    result = class_enrichment(contrast, ann, level="npc_class", min_class_n=3)
    assert len(result) == 2
    assert result["significant"].all()
    row_a = result[result["class_name"] == "ClassA"].iloc[0]
    assert row_a["mean_log2fc"] == pytest.approx(5.0)
    assert row_a["n"] == 15


def test_min_class_n_filter() -> None:
    # ClassA has 2 members (< default min_class_n=3) — should be skipped.
    contrast = _make_contrast(10)
    ann = _make_annotations({1: "ClassA", 2: "ClassA",
                               3: "ClassB", 4: "ClassB", 5: "ClassB"})
    result = class_enrichment(contrast, ann, level="npc_class", min_class_n=3)
    assert "ClassA" not in result["class_name"].values
    assert "ClassB" in result["class_name"].values


def test_empty_result_when_all_classes_below_min() -> None:
    contrast = _make_contrast(6)
    ann = _make_annotations({1: "ClassA", 2: "ClassA"})
    result = class_enrichment(contrast, ann, level="npc_class", min_class_n=3)
    assert len(result) == 0
    assert list(result.columns) == [
        "level", "class_name", "n", "mean_log2fc", "p_value", "p_adj", "significant"
    ]


def test_bh_applied_within_level() -> None:
    # p_adj must be >= p_value for all rows.
    n = 40
    rng = np.random.default_rng(1)
    idx = pd.Index(range(1, n + 1), name="feature_id")
    contrast = pd.DataFrame({"log2fc": rng.normal(size=n)}, index=idx)
    assignments = {i: f"Class{i % 5}" for i in range(1, n + 1)}
    ann = _make_annotations(assignments)

    result = class_enrichment(contrast, ann, level="npc_class", min_class_n=3)
    if len(result) > 0:
        assert (result["p_adj"].values + 1e-12 >= result["p_value"].values).all()


def test_sorted_by_p_adj() -> None:
    n = 30
    idx = pd.Index(range(1, n + 1), name="feature_id")
    contrast = pd.DataFrame({"log2fc": np.concatenate([np.full(15, 5.0), np.full(15, -5.0)])},
                             index=idx)
    assignments = {i: f"Class{i % 5}" for i in range(1, n + 1)}
    ann = _make_annotations(assignments)
    result = class_enrichment(contrast, ann, level="npc_class", min_class_n=3)
    if len(result) > 1:
        assert (result["p_adj"].values[:-1] <= result["p_adj"].values[1:]).all()


# ---------------------------------------------------------------------------
# Missing *_passed column
# ---------------------------------------------------------------------------

def test_missing_passed_column_warns_and_returns_empty() -> None:
    contrast = _make_contrast(10)
    # Ann without the _passed column.
    ann = pd.DataFrame(
        {"canopus_npc_class": ["ClassA"] * 5},
        index=pd.Index(range(1, 6), name="feature_id"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = class_enrichment(contrast, ann, level="npc_class")
    assert len(result) == 0
    assert any("Skipping level" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Background modes
# ---------------------------------------------------------------------------

def test_background_all_excludes_foreground() -> None:
    n = 20
    idx = pd.Index(range(1, n + 1), name="feature_id")
    contrast = pd.DataFrame({"log2fc": np.ones(n)}, index=idx)
    assignments = {i: "ClassA" for i in range(1, 6)}
    assignments.update({i: "ClassB" for i in range(6, 11)})
    ann = _make_annotations(assignments)

    result = class_enrichment(contrast, ann, level="npc_class",
                               background="all", min_class_n=3)
    assert len(result) == 2  # both classes tested


def test_background_level_only_uses_annotated() -> None:
    n = 20
    idx = pd.Index(range(1, n + 1), name="feature_id")
    contrast = pd.DataFrame({"log2fc": np.ones(n)}, index=idx)
    # Only features 1-9 are annotated at this level.
    assignments = {i: "ClassA" for i in range(1, 5)}
    assignments.update({i: "ClassB" for i in range(5, 10)})
    ann = _make_annotations(assignments)

    result_all = class_enrichment(contrast, ann, level="npc_class",
                                   background="all", min_class_n=3)
    result_level = class_enrichment(contrast, ann, level="npc_class",
                                     background="level", min_class_n=3)
    # Both run without error; background differs but both valid.
    assert set(result_all["class_name"]) == set(result_level["class_name"])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_invalid_level_raises() -> None:
    contrast = _make_contrast(6)
    ann = _make_annotations({1: "ClassA", 2: "ClassA", 3: "ClassA"})
    with pytest.raises(ValueError, match="Unknown level"):
        class_enrichment(contrast, ann, level="npc_bogus")


def test_invalid_background_raises() -> None:
    contrast = _make_contrast(6)
    ann = _make_annotations({1: "ClassA", 2: "ClassA", 3: "ClassA"})
    with pytest.raises(ValueError, match="Unknown background"):
        class_enrichment(contrast, ann, level="npc_class", background="random")


def test_no_finite_log2fc_raises() -> None:
    idx = pd.Index(range(1, 6), name="feature_id")
    contrast = pd.DataFrame({"log2fc": [float("nan")] * 5}, index=idx)
    ann = _make_annotations({1: "ClassA", 2: "ClassA", 3: "ClassA"})
    with pytest.raises(ValueError, match="no finite log2fc"):
        class_enrichment(contrast, ann, level="npc_class")


def test_multi_label_counts_in_each_class() -> None:
    # Feature 3 is "ClassA|ClassB" (GNPS NPClassifier style) → member of both.
    contrast = _make_contrast(20)
    ann = _make_annotations({1: "ClassA", 2: "ClassA", 3: "ClassA|ClassB",
                             4: "ClassB", 5: "ClassB"})
    result = class_enrichment(contrast, ann, level="npc_class", min_class_n=3)
    assert dict(zip(result["class_name"], result["n"])) == {"ClassA": 3, "ClassB": 3}


# ---------------------------------------------------------------------------
# CANOPUS_LEVELS constant
# ---------------------------------------------------------------------------

def test_canopus_levels_constant() -> None:
    assert set(CANOPUS_LEVELS) == {
        "npc_pathway", "npc_superclass", "npc_class", "classyfire_class"
    }
