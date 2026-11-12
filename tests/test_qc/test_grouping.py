"""Tests for the per-group alignment breakdown in
``metabproject.project._run_alignment_by_group``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metabproject.project import _run_alignment_by_group


def _build_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Eight samples split A1..A4 (group A) / B1..B4 (group B), four features.

    Feature pair (1, 2) is an alignment_like failure that ONLY manifests
    inside group A (disjoint detection across A samples, ΔRT well above
    FWHM). Inside group B, features 1 and 2 are both detected together,
    so the pair won't qualify as alignment_like there.

    Features 3 and 4 are isolated controls.
    """
    samples_a = ["A1", "A2", "A3", "A4"]
    samples_b = ["B1", "B2", "B3", "B4"]
    samples = samples_a + samples_b

    feat_ids = [1, 2, 3, 4]
    mz = [200.0000, 200.0010, 500.0, 600.0]
    rt = [5.00, 5.06, 3.00, 7.00]
    ms1 = pd.DataFrame(
        {"mz": mz, "rt": rt},
        index=pd.Index(feat_ids, name="feature_id"),
    )

    # Detection per feature × sample. Inside group A, feature 1 is detected
    # only in A1/A2 and feature 2 only in A3/A4 (disjoint sets, ΔRT = 0.06,
    # fwhm = 0.02). Inside group B, both 1 and 2 are detected in every
    # sample so jaccard = 1.0 → not alignment_like.
    detected = {
        1: [True,  True,  False, False,  True, True, True, True],
        2: [False, False, True,  True,   True, True, True, True],
        3: [True,  True,  True,  False,  True, True, True, False],
        4: [False, True,  True,  True,   True, True, False, True],
    }

    areas = {
        1: [1e6, 1.1e6, np.nan, np.nan, 1.0e6, 1.1e6, 1.2e6, 1.05e6],
        2: [np.nan, np.nan, 9.0e5, 1.0e6, 1.0e6, 1.1e6, 1.2e6, 1.05e6],
        3: [2e6, 2.5e6, 2.1e6, np.nan, 1.8e6, 2.0e6, 2.2e6, np.nan],
        4: [np.nan, 1e6, 1.2e6, 1.1e6, 1.0e6, 1.1e6, np.nan, 1.2e6],
    }
    areas_df = pd.DataFrame(
        {fid: pd.Series(areas[fid], index=samples) for fid in feat_ids}
    ).T
    areas_df.index = pd.Index(feat_ids, name="feature_id")

    detail_cols: list[tuple[str, str]] = []
    detail_vals: list[list] = []
    for i, s in enumerate(samples):
        detail_cols.append((s, "feature_state"))
        detail_vals.append([
            "DETECTED" if detected[fid][i] else "MISSING"
            for fid in feat_ids
        ])
        detail_cols.append((s, "fwhm"))
        detail_vals.append([
            0.02 if detected[fid][i] else np.nan
            for fid in feat_ids
        ])

    details = pd.DataFrame(
        np.array(detail_vals, dtype=object).T,
        index=pd.Index(feat_ids, name="feature_id"),
        columns=pd.MultiIndex.from_tuples(
            detail_cols, names=["filename", "metric"]
        ),
    )

    metadata = pd.DataFrame({
        "sample_file": samples,
        "variety": ["A"] * 4 + ["B"] * 4,
    }, index=pd.Index(samples, name="sample_name"))

    return ms1, areas_df, details, metadata


def test_group_a_is_worst_aligned() -> None:
    ms1, areas, details, metadata = _build_inputs()

    rows, skipped = _run_alignment_by_group(
        ms1=ms1, areas=areas, sample_details=details,
        metadata=metadata,
        group_by="variety",
        min_samples_per_group=3,
        mz_tol_ppm=10.0,
        rt_tol=0.1,
        min_correlation=0.7,
        use_fwhm=True,
    )

    assert skipped == []
    assert [r["group_value"] for r in rows] == ["A", "B"]

    by_group = {r["group_value"]: r for r in rows}
    # Group A surfaces the engineered alignment_like pair; B does not.
    assert by_group["A"]["n_alignment_like"] == 1
    assert by_group["B"]["n_alignment_like"] == 0
    assert by_group["A"]["alignment_like_rate"] > 0
    assert by_group["B"]["alignment_like_rate"] == 0
    # Headline: A ranks first because rows are sorted by alignment_like_rate.
    assert rows[0]["group_value"] == "A"
    # Rates are normalized by features detected in the group.
    assert by_group["A"]["n_features_detected"] >= 2
    assert by_group["A"]["flagged_feature_rate"] > 0


def test_small_group_is_skipped() -> None:
    ms1, areas, details, metadata = _build_inputs()
    # Reassign one A sample to a tiny new group → only 1 sample, below threshold.
    metadata = metadata.copy()
    metadata.loc["A4", "variety"] = "C"

    rows, skipped = _run_alignment_by_group(
        ms1=ms1, areas=areas, sample_details=details,
        metadata=metadata,
        group_by="variety",
        min_samples_per_group=3,
        mz_tol_ppm=10.0,
        rt_tol=0.1,
        min_correlation=0.7,
        use_fwhm=True,
    )

    assert any(s["group_value"] == "C" for s in skipped)
    assert all(r["group_value"] != "C" for r in rows)
