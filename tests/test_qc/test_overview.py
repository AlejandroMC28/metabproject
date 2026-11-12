"""Tests for ``metabproject.qc.overview.compute_overview``.

Synthetic 3-sample × 5-feature dataset covering:
- A sparse feature (1/3 detected) → singleton bin + drags density down.
- A fully detected feature (3/3) → top bin.
- A feature with a low alignment_scores:rate → low_alignment_rate group.
- A feature with poor peak shape (tailing/asymmetry > threshold) →
  poor_peak_shape group.
- A baseline feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metabproject.qc.overview import compute_overview


SAMPLES = ["S1", "S2", "S3"]


def _toy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat_ids = [1, 2, 3, 4, 5]

    # Detection masks per feature.
    detected = {
        1: [True, False, False],   # singleton
        2: [True, True, True],     # fully detected, baseline
        3: [True, True, True],     # fully detected, low alignment rate
        4: [True, True, True],     # fully detected, poor peak shape
        5: [True, True, False],    # partial
    }

    ms1 = pd.DataFrame(
        {
            "mz": [200.0, 300.0, 400.0, 500.0, 600.0],
            "rt": [1.0, 2.0, 3.0, 4.0, 5.0],
            "charge": [1, 1, 2, 1, 3],
            "fragment_scans": [0, 1, 1, 1, 0],
            "alignment_scores:rate": [0.9, 0.95, 0.05, 0.9, 0.88],
            "alignment_scores:weighted_distance_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "alignment_scores:rt_absolute_error": [0.01, 0.02, 0.03, 0.04, 0.05],
            "alignment_scores:mz_diff_ppm": [1.0, 2.0, 3.0, 4.0, 5.0],
            "alignment_scores:aligned_features_n": [3, 3, 3, 3, 3],
            "preferred_annotation:compound_name": [None, "X", None, None, None],
            "preferred_annotation:annotation_method": [None, "gnps", None, None, None],
        },
        index=pd.Index(feat_ids, name="feature_id"),
    )

    # Areas (positive where DETECTED).
    area_vals = {
        1: [1e6, np.nan, np.nan],
        2: [1e6, 1.1e6, 1.2e6],
        3: [2e6, 2.1e6, 2.2e6],
        4: [3e6, 3.1e6, 3.2e6],
        5: [4e6, 4.1e6, np.nan],
    }
    areas = pd.DataFrame(
        {fid: pd.Series(area_vals[fid], index=SAMPLES) for fid in feat_ids}
    ).T
    areas.index = pd.Index(feat_ids, name="feature_id")

    # Zone 3: feature_state, fwhm, tailing_factor, asymmetry_factor per sample.
    detail_cols: list[tuple[str, str]] = []
    detail_vals: list[list] = []
    for s_idx, s in enumerate(SAMPLES):
        detail_cols.append((s, "feature_state"))
        detail_vals.append([
            "DETECTED" if detected[fid][s_idx] else "MISSING"
            for fid in feat_ids
        ])
        detail_cols.append((s, "fwhm"))
        detail_vals.append([
            0.05 if detected[fid][s_idx] else np.nan for fid in feat_ids
        ])
        detail_cols.append((s, "tailing_factor"))
        # Feature 4 has tailing > 2 (poor); others ~1.1.
        detail_vals.append([
            (3.0 if fid == 4 else 1.1) if detected[fid][s_idx] else np.nan
            for fid in feat_ids
        ])
        detail_cols.append((s, "asymmetry_factor"))
        detail_vals.append([
            (3.5 if fid == 4 else 1.0) if detected[fid][s_idx] else np.nan
            for fid in feat_ids
        ])

    details = pd.DataFrame(
        np.array(detail_vals, dtype=object).T,
        index=pd.Index(feat_ids, name="feature_id"),
        columns=pd.MultiIndex.from_tuples(detail_cols, names=["filename", "metric"]),
    )
    return ms1, areas, details


def test_compute_overview_metric_groups_and_issues() -> None:
    ms1, areas, details = _toy_inputs()

    metrics, issues = compute_overview(
        ms1, areas, details,
        low_alignment_rate_threshold=0.10,
        low_detection_sample_zscore=-2.0,
        poor_peak_shape_threshold=2.0,
    )

    # ---- Table shape -----------------------------------------------------
    shape = metrics["table_shape"]
    assert shape["n_features"] == 5
    assert shape["n_samples"] == 3
    # 12 detected cells out of 15.
    assert shape["density"] == 12 / 15

    # ---- Per-feature detection histogram --------------------------------
    hist = metrics["per_feature_detection"]["detection_histogram"]
    # n_samples=3 < 27: bins above the saturation count are dropped and the
    # top bin is relabeled to "2_to_3_or_all".
    assert set(hist.keys()) == {"1", "2_to_3_or_all"}
    # Feature 1 = 1 detection; features 2,3,4 = 3 detections; feature 5 = 2.
    assert hist["1"]["count"] == 1
    assert hist["2_to_3_or_all"]["count"] == 4

    # ---- Per-sample detection -------------------------------------------
    ps = metrics["per_sample_detection"]
    samples_sorted = [r["sample"] for r in ps["per_sample_detected"]]
    # S3 has 2 detections, others have 5 and 4.
    assert samples_sorted[0] == "S3"
    # With n_samples=3 the z-score band is too wide to flag any sample,
    # so the list stays empty in this fixture.
    assert ps["flagged_samples"] == []

    # ---- Alignment quality + issue --------------------------------------
    aq = metrics["alignment_quality"]
    assert aq["n_low_alignment_rate"] == 1  # feature 3 has rate < 0.10
    assert aq["fraction_low_alignment_rate"] == 1 / 5
    # fraction (0.20) is below the 0.25 warning cutoff → info-severity issue.
    align_issues = [i for i in issues if i["type"] == "low_alignment_rate"]
    assert len(align_issues) == 1
    assert align_issues[0]["severity"] == "info"

    # ---- Chromatographic quality + issue --------------------------------
    cq = metrics["chromatographic_quality"]
    # 3 of 12 DETECTED cells exceed tailing=2 → fraction = 3/12 > 0.10.
    assert cq["fraction_poor_tailing"] > 0.10
    assert cq["fraction_poor_asymmetry"] > 0.10
    shape_issues = [i for i in issues if i["type"] == "poor_peak_shape"]
    assert len(shape_issues) == 1
    assert shape_issues[0]["severity"] == "warning"

    # ---- MS2 coverage ----------------------------------------------------
    ms2 = metrics["ms2_coverage"]
    # 3 of 5 features have fragment_scans > 0.
    assert ms2["fraction_with_ms2"] == 3 / 5
    # Above the 0.30 info threshold so no low_ms2_coverage issue.
    assert not any(i["type"] == "low_ms2_coverage" for i in issues)

    # ---- Mass / RT coverage ---------------------------------------------
    mr = metrics["mass_rt_coverage"]
    assert mr["mz_range"] == {"min": 200.0, "max": 600.0}
    assert mr["rt_range"] == {"min": 1.0, "max": 5.0}
    assert len(mr["mz_rt_density"]) == 5  # one cell per feature on a 50×50 grid

    # ---- Annotation coverage --------------------------------------------
    ann = metrics["annotation_coverage"]
    assert ann["fraction_preferred_annotated"] == 1 / 5
    assert ann["by_method"] == {"gnps": 1}

    # ---- Charge state ----------------------------------------------------
    cs = metrics["charge_state"]
    assert cs["counts"]["1"] == 3
    assert cs["counts"]["2"] == 1
    assert cs["counts"]["3+"] == 1
    assert cs["counts"]["missing"] == 0

    # ---- Singleton-fraction issue ---------------------------------------
    # 1 singleton / 5 features = 0.20 > 0.15 → warning.
    singleton_issues = [i for i in issues if i["type"] == "high_singleton_fraction"]
    assert len(singleton_issues) == 1
    assert singleton_issues[0]["severity"] == "warning"


def test_compute_overview_low_ms2_triggers_info_issue() -> None:
    ms1, areas, details = _toy_inputs()
    ms1 = ms1.copy()
    # Knock out all MS2 coverage.
    ms1["fragment_scans"] = 0

    _, issues = compute_overview(
        ms1, areas, details,
        low_alignment_rate_threshold=0.10,
        low_detection_sample_zscore=-2.0,
        poor_peak_shape_threshold=2.0,
    )
    low_ms2 = [i for i in issues if i["type"] == "low_ms2_coverage"]
    assert len(low_ms2) == 1
    assert low_ms2[0]["severity"] == "info"


def test_compute_overview_sparse_table_triggers_warning() -> None:
    """Force density below 0.30 by replacing most DETECTED cells with MISSING."""
    ms1, areas, details = _toy_inputs()
    # Wipe S2/S3 entirely and most of S1 to drive density below 0.30.
    new_details = details.copy()
    for s in ("S2", "S3"):
        new_details[(s, "feature_state")] = "MISSING"
    new_details[("S1", "feature_state")] = ["DETECTED"] + ["MISSING"] * 4

    metrics, issues = compute_overview(
        ms1, areas, new_details,
        low_alignment_rate_threshold=0.10,
        low_detection_sample_zscore=-2.0,
        poor_peak_shape_threshold=2.0,
    )
    assert metrics["table_shape"]["density"] < 0.30
    assert any(i["type"] == "sparse_table" and i["severity"] == "warning"
               for i in issues)


def test_compute_overview_sample_name_map_applied() -> None:
    ms1, areas, details = _toy_inputs()
    metrics, _ = compute_overview(
        ms1, areas, details,
        low_alignment_rate_threshold=0.10,
        low_detection_sample_zscore=-2.0,
        poor_peak_shape_threshold=2.0,
        sample_name_map={"S1": "control_rep1", "S2": "control_rep2",
                         "S3": "fermented_rep1"},
    )
    samples = {r["sample"] for r in metrics["per_sample_detection"][
        "per_sample_detected"
    ]}
    assert samples == {"control_rep1", "control_rep2", "fermented_rep1"}
