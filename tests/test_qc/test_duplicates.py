"""Tests for ``metabproject.qc.duplicates.detect_duplicates``.

Hand-built 3-sample × 6-feature dataset containing one each of an
alignment_like, split_like, and isomer_like pair, plus two isolated
features that must not be flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabproject.qc.duplicates import detect_duplicates


SAMPLES = ["S1", "S2", "S3"]


def _toy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Six features:
    - 1, 2 → alignment_like:  same nominal mz, ΔRT = 0.06 (big vs fwhm 0.02),
      and detection sets are disjoint (S1 only / S2,S3 only).
    - 3, 4 → split_like: same mz, ΔRT = 0.01 (≤ 0.02 fwhm), full overlap,
      highly correlated areas.
    - 5, 6 → isomer_like: same mz, ΔRT = 0.08 (4× fwhm 0.02), full overlap,
      uncorrelated.
    - 7    → isolated (mz far from all others).
    - 8    → isolated (mz far from all others).
    """
    feat_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    mz = [200.0000, 200.0010, 300.0000, 300.0015, 400.0000, 400.0020, 500.0, 600.0]
    rt = [5.00,     5.06,     8.00,     8.01,     12.00,    12.08,    3.00,  7.00]
    ms1 = pd.DataFrame({"mz": mz, "rt": rt}, index=pd.Index(feat_ids, name="feature_id"))

    # Detection masks per feature.
    detected = {
        1: [True,  False, False],
        2: [False, True,  True],
        3: [True,  True,  True],
        4: [True,  True,  True],
        5: [True,  True,  True],
        6: [True,  True,  True],
        7: [True,  True,  False],
        8: [False, True,  True],
    }

    # Areas: NaN where MISSING, otherwise positive. For features 3 & 4 the
    # areas are perfectly correlated (split_like). For 5 & 6 they are
    # uncorrelated so the pair only qualifies as isomer_like (which doesn't
    # require Pearson).
    areas = {
        1: [1e6,  np.nan, np.nan],
        2: [np.nan, 1.1e6, 9.0e5],
        3: [1e6, 2e6, 3e6],
        4: [1.1e6, 2.2e6, 3.3e6],   # perfectly correlated with feature 3
        5: [1e6, 5e6, 2e6],
        6: [4e6, 1.0e6, 6e6],       # weak/negative correlation
        7: [2e6, 2.5e6, np.nan],
        8: [np.nan, 1e6, 1.2e6],
    }

    areas_df = pd.DataFrame(
        {fid: pd.Series(areas[fid], index=SAMPLES) for fid in feat_ids}
    ).T
    areas_df.index = pd.Index(feat_ids, name="feature_id")

    # Zone 3 MultiIndex columns: (sample, metric).
    detail_cols: list[tuple[str, str]] = []
    detail_vals: list[list] = []
    for s in SAMPLES:
        # feature_state column for sample s
        detail_cols.append((s, "feature_state"))
        detail_vals.append([
            "DETECTED" if detected[fid][SAMPLES.index(s)] else "MISSING"
            for fid in feat_ids
        ])
        # fwhm column for sample s (constant 0.02 min where detected, else NaN)
        detail_cols.append((s, "fwhm"))
        detail_vals.append([
            0.02 if detected[fid][SAMPLES.index(s)] else np.nan
            for fid in feat_ids
        ])

    details = pd.DataFrame(
        np.array(detail_vals, dtype=object).T,
        index=pd.Index(feat_ids, name="feature_id"),
        columns=pd.MultiIndex.from_tuples(detail_cols, names=["filename", "metric"]),
    )
    return ms1, areas_df, details


def test_detect_duplicates_classifies_three_pairs() -> None:
    ms1, areas, details = _toy_inputs()

    pairs, flags, summary = detect_duplicates(
        ms1, areas, details,
        mz_tol_ppm=10.0,
        rt_tol=0.1,
        min_correlation=0.7,
        use_fwhm=True,
    )

    # Exactly the three engineered pairs should be candidates.
    assert len(pairs) == 3, pairs

    by_pair = {
        (row.feature_id_i, row.feature_id_j): row.classification
        for row in pairs.itertuples()
    }
    assert by_pair[(1, 2)] == "alignment_like"
    assert by_pair[(3, 4)] == "split_like"
    assert by_pair[(5, 6)] == "isomer_like"

    # Counts in summary should reflect the same.
    assert summary["counts"]["alignment_like"] == 1
    assert summary["counts"]["split_like"] == 1
    assert summary["counts"]["isomer_like"] == 1
    assert summary["counts"]["unclassified"] == 0

    # Six features participate in pairs; isolated 7 & 8 must not.
    flagged = set(flags["feature_id"])
    assert flagged == {1, 2, 3, 4, 5, 6}

    # Recommended actions per the rule table.
    action = dict(zip(flags["feature_id"], flags["recommended_action"]))
    assert action[3] == "drop_lower_intensity"
    assert action[4] == "drop_lower_intensity"
    assert action[5] == "keep_both"
    assert action[6] == "keep_both"
    assert action[1] == "review"
    assert action[2] == "review"

    # Per-feature pair count is 1 for each flagged feature.
    assert (flags["n_pairs"] == 1).all()


def test_no_pairs_when_tolerances_too_tight() -> None:
    ms1, areas, details = _toy_inputs()
    pairs, flags, summary = detect_duplicates(
        ms1, areas, details,
        mz_tol_ppm=0.1,   # impossibly tight ppm window
        rt_tol=0.1,
        min_correlation=0.7,
        use_fwhm=True,
    )
    assert pairs.empty
    assert flags.empty
    assert all(c == 0 for c in summary["counts"].values())


def test_use_fwhm_false_falls_back_to_drt_rule() -> None:
    ms1, areas, details = _toy_inputs()
    pairs, _, _ = detect_duplicates(
        ms1, areas, details,
        mz_tol_ppm=10.0,
        rt_tol=0.1,
        min_correlation=0.7,
        use_fwhm=False,
    )
    # The disjoint-detection pair (1,2) should still be alignment_like
    # without FWHM, via the |ΔRT| > 0 fallback.
    by_pair = {
        (row.feature_id_i, row.feature_id_j): row.classification
        for row in pairs.itertuples()
    }
    assert by_pair[(1, 2)] == "alignment_like"
    # drt_over_fwhm should be NaN throughout when use_fwhm=False.
    assert pairs["drt_over_fwhm"].isna().all()
