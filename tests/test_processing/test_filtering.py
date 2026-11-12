"""Tests for the per-group prevalence filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.processing.filtering import prevalence_filter


def _make_table(areas: dict[str, list[float]]) -> pd.DataFrame:
    """Build a small feature table from a sample→areas dict."""
    n = len(next(iter(areas.values())))
    return pd.DataFrame(areas, index=pd.Index(range(1, n + 1), name="feature_id"))


def _make_meta(groups: dict[str, str]) -> pd.DataFrame:
    """Build a minimal metadata frame from sample_name→group dict."""
    return pd.DataFrame(
        {"group": list(groups.values())},
        index=pd.Index(list(groups.keys()), name="sample_name"),
    )


# ---------------------------------------------------------------------------
# Two-group scenarios
# ---------------------------------------------------------------------------


def test_two_group_basic():
    """Feature kept if >= 2 detected in control OR fermented (basic case)."""
    df = _make_table({
        "ctrl_1": [100.0, 0.0, np.nan],
        "ctrl_2": [100.0, 0.0, 100.0],
        "ctrl_3": [100.0, 0.0, 0.0],
        "ferm_1": [0.0,   100.0, 0.0],
        "ferm_2": [0.0,   100.0, 0.0],
        "ferm_3": [0.0,   0.0,   0.0],
    })
    meta = _make_meta({
        "ctrl_1": "control", "ctrl_2": "control", "ctrl_3": "control",
        "ferm_1": "fermented", "ferm_2": "fermented", "ferm_3": "fermented",
    })
    result = prevalence_filter(df, meta, min_prevalence=2)
    # Feature 1: ctrl has 3 ≥ 2 → keep; feature 2: ferm has 2 ≥ 2 → keep;
    # feature 3: ctrl has 1, ferm has 0 → drop
    assert list(result.index) == [1, 2]


def test_two_group_prevalence_passes_either_group():
    """A feature detected only in fermented (not control) should be kept."""
    df = _make_table({
        "ctrl_1": [0.0], "ctrl_2": [0.0], "ctrl_3": [0.0],
        "ferm_1": [100.0], "ferm_2": [100.0], "ferm_3": [100.0],
    })
    meta = _make_meta({
        "ctrl_1": "control", "ctrl_2": "control", "ctrl_3": "control",
        "ferm_1": "fermented", "ferm_2": "fermented", "ferm_3": "fermented",
    })
    result = prevalence_filter(df, meta, min_prevalence=2)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Nine-group scenario
# ---------------------------------------------------------------------------


def test_nine_group_prevalence():
    """Feature kept if >= 2 detected in any one of nine groups."""
    # 9 groups × 3 reps = 27 samples. One feature detected in group 5 only.
    rng = np.random.default_rng(0)
    n_groups = 9
    reps_per_group = 3
    samples = [f"g{g}_r{r}" for g in range(n_groups) for r in range(reps_per_group)]
    groups = {s: f"group_{s[:2]}" for s in samples}

    # Feature 1: all zeros except group_g5 (≥2 detected there)
    areas_feat1 = [0.0] * len(samples)
    g5_idxs = [i for i, s in enumerate(samples) if s.startswith("g5")]
    for i in g5_idxs[:2]:
        areas_feat1[i] = 100.0

    # Feature 2: completely absent everywhere
    areas_feat2 = [0.0] * len(samples)

    df = pd.DataFrame(
        {"feat1": areas_feat1, "feat2": areas_feat2},
        index=pd.Index(samples, name="sample_name"),
    ).T
    df.index = pd.Index([1, 2], name="feature_id")

    meta = _make_meta(groups)
    result = prevalence_filter(df, meta, min_prevalence=2)
    assert list(result.index) == [1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_edge_exactly_n_detections():
    """Feature with exactly min_prevalence detections in a group passes."""
    # Feature 1: 2 detections (exactly min_prevalence=2) → keep
    # Feature 2: 1 detection → drop
    df = _make_table({
        "ctrl_1": [100.0, 100.0],
        "ctrl_2": [100.0, 0.0],
        "ctrl_3": [0.0,   0.0],
    })
    meta = _make_meta({"ctrl_1": "ctrl", "ctrl_2": "ctrl", "ctrl_3": "ctrl"})
    result = prevalence_filter(df, meta, min_prevalence=2)
    assert len(result) == 1
    assert result.index[0] == 1


def test_edge_n_minus_one_detections():
    """Feature with min_prevalence-1 detections in every group is removed."""
    df = _make_table({
        "ctrl_1": [100.0], "ctrl_2": [0.0], "ctrl_3": [0.0],
        "ferm_1": [100.0], "ferm_2": [0.0], "ferm_3": [0.0],
    })
    meta = _make_meta({
        "ctrl_1": "control", "ctrl_2": "control", "ctrl_3": "control",
        "ferm_1": "fermented", "ferm_2": "fermented", "ferm_3": "fermented",
    })
    result = prevalence_filter(df, meta, min_prevalence=2)
    assert len(result) == 0


def test_nan_treated_as_missing():
    """NaN values are not counted as detections."""
    df = _make_table({
        "ctrl_1": [np.nan], "ctrl_2": [np.nan], "ctrl_3": [100.0],
    })
    meta = _make_meta({"ctrl_1": "ctrl", "ctrl_2": "ctrl", "ctrl_3": "ctrl"})
    result = prevalence_filter(df, meta, min_prevalence=2)
    assert len(result) == 0


def test_returns_copy():
    """Result is a copy; modifying it does not affect the input."""
    df = _make_table({"s1": [100.0], "s2": [100.0]})
    meta = _make_meta({"s1": "g", "s2": "g"})
    result = prevalence_filter(df, meta, min_prevalence=1)
    result.iloc[0, 0] = -999.0
    assert df.iloc[0, 0] == 100.0


# ---------------------------------------------------------------------------
# detected-mode and min_groups_meeting (3 groups × 3 reps fixture)
# ---------------------------------------------------------------------------


def _three_group_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """3 groups × 3 replicates = 9 samples; 6 features covering each test case.

    Feature plan (rows = features 1..6):
      1: detected in 2/3 reps of group A only
      2: detected in 1 sample of A only (insufficient)
      3: present as area in 2 reps of A but flagged ESTIMATED (gap-filled)
      4: detected in 2 reps of A AND 2 reps of B (≥2 groups)
      5: detected only in group A (1 group meets threshold)
      6: detected in all 3 reps of every group (universally detected)
    """
    samples = [
        "a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "c3",
    ]
    groups_map = {
        "a1": "A", "a2": "A", "a3": "A",
        "b1": "B", "b2": "B", "b3": "B",
        "c1": "C", "c2": "C", "c3": "C",
    }
    meta = _make_meta(groups_map)

    # Areas: every nonzero value is "present"; the detected mask further
    # constrains which are truly DETECTED vs ESTIMATED.
    areas = pd.DataFrame(
        [
            #  a1     a2     a3     b1     b2     b3     c1     c2     c3
            [100.0, 100.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 1
            [100.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 2
            [ 50.0,  50.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 3 ESTIMATED in a1,a2
            [100.0, 100.0,   0.0, 100.0, 100.0,   0.0,   0.0,   0.0,   0.0],  # 4
            [100.0, 100.0, 100.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 5
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],  # 6
        ],
        columns=samples,
        index=pd.Index([1, 2, 3, 4, 5, 6], name="feature_id"),
    )

    detection_mask = pd.DataFrame(
        [
            [True,  True,  False, False, False, False, False, False, False],  # 1
            [True,  False, False, False, False, False, False, False, False],  # 2
            [False, False, False, False, False, False, False, False, False],  # 3 all ESTIMATED
            [True,  True,  False, True,  True,  False, False, False, False],  # 4
            [True,  True,  True,  False, False, False, False, False, False],  # 5
            [True,  True,  True,  True,  True,  True,  True,  True,  True ],  # 6
        ],
        columns=samples,
        index=pd.Index([1, 2, 3, 4, 5, 6], name="feature_id"),
    )

    return areas, meta, detection_mask


def test_any_value_mode_default_matches_legacy_behavior():
    """Calling without mode kwarg reproduces legacy any_value semantics."""
    # Same input as test_two_group_basic above; identical expected output.
    df = _make_table({
        "ctrl_1": [100.0, 0.0, np.nan],
        "ctrl_2": [100.0, 0.0, 100.0],
        "ctrl_3": [100.0, 0.0, 0.0],
        "ferm_1": [0.0,   100.0, 0.0],
        "ferm_2": [0.0,   100.0, 0.0],
        "ferm_3": [0.0,   0.0,   0.0],
    })
    meta = _make_meta({
        "ctrl_1": "control", "ctrl_2": "control", "ctrl_3": "control",
        "ferm_1": "fermented", "ferm_2": "fermented", "ferm_3": "fermented",
    })
    result = prevalence_filter(df, meta, min_prevalence=2)
    assert list(result.index) == [1, 2]


def test_detected_mode_keeps_feature_detected_in_two_replicates_of_one_group():
    df, meta, mask = _three_group_fixture()
    result = prevalence_filter(
        df, meta, min_prevalence=2,
        mode="detected", detection_mask=mask,
    )
    assert 1 in result.index


def test_detected_mode_drops_feature_detected_only_in_one_sample():
    df, meta, mask = _three_group_fixture()
    result = prevalence_filter(
        df, meta, min_prevalence=2,
        mode="detected", detection_mask=mask,
    )
    assert 2 not in result.index


def test_detected_mode_drops_feature_present_only_as_estimated():
    """Feature 3 has area in 2 A-reps but all are ESTIMATED (gap-filled)."""
    df, meta, mask = _three_group_fixture()
    # any_value would keep it...
    any_value_result = prevalence_filter(df, meta, min_prevalence=2)
    assert 3 in any_value_result.index
    # ...detected mode correctly drops it.
    detected_result = prevalence_filter(
        df, meta, min_prevalence=2,
        mode="detected", detection_mask=mask,
    )
    assert 3 not in detected_result.index


def test_min_groups_meeting_two_drops_single_group_feature():
    df, meta, mask = _three_group_fixture()
    # any_value: feature 5 (A only) drops; feature 4 (A+B) and 6 (all) stay.
    any_value_result = prevalence_filter(
        df, meta, min_prevalence=2, min_groups_meeting=2,
    )
    assert 4 in any_value_result.index
    assert 6 in any_value_result.index
    assert 5 not in any_value_result.index
    assert 1 not in any_value_result.index

    detected_result = prevalence_filter(
        df, meta, min_prevalence=2, min_groups_meeting=2,
        mode="detected", detection_mask=mask,
    )
    assert 4 in detected_result.index
    assert 6 in detected_result.index
    assert 5 not in detected_result.index


def test_detection_mask_with_mismatched_columns_raises():
    df, meta, _ = _three_group_fixture()
    bad_mask = pd.DataFrame(
        True,
        index=df.index,
        columns=["x", "y", "z"],
    )
    with pytest.raises(ValueError, match="shares no columns"):
        prevalence_filter(
            df, meta, min_prevalence=2,
            mode="detected", detection_mask=bad_mask,
        )


def test_detected_mode_without_mask_raises():
    df, meta, _ = _three_group_fixture()
    with pytest.raises(ValueError, match="requires a detection_mask"):
        prevalence_filter(df, meta, min_prevalence=2, mode="detected")
