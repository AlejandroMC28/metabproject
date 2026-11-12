"""Feature-table overview profiling.

Computes table-level health metrics from the three MZmine zones:

- Shape, density, per-feature / per-sample detection profiles.
- MZmine alignment-score distributions.
- Chromatographic quality (FWHM, tailing, asymmetry).
- MS2 coverage (overall and by intensity decile).
- m/z × RT density grid.
- Annotation coverage (when present).
- Charge-state counts.

Independent of CLI/IO so it is straightforward to unit-test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_DETECTION_BINS = [
    ("1", 1, 1),
    ("2_3", 2, 3),
    ("4_9", 4, 9),
    ("10_18", 10, 18),
    ("19_26", 19, 26),
    ("27_or_all", 27, None),  # upper bound replaced with n_samples
]

_ALIGNMENT_METRICS = [
    "alignment_scores:rate",
    "alignment_scores:weighted_distance_score",
    "alignment_scores:rt_absolute_error",
    "alignment_scores:mz_diff_ppm",
    "alignment_scores:aligned_features_n",
]

# Issue thresholds (the *issue* thresholds, not the per-feature flags which
# come from CLI args).
SPARSE_TABLE_DENSITY = 0.30
LOW_ALIGN_FRACTION = 0.25
LOW_MS2_FRACTION = 0.30
POOR_SHAPE_FRACTION = 0.10
SINGLETON_FRACTION = 0.15


def compute_overview(
    ms1_features: pd.DataFrame,
    areas: pd.DataFrame,
    sample_details: pd.DataFrame,
    *,
    low_alignment_rate_threshold: float,
    low_detection_sample_zscore: float,
    poor_peak_shape_threshold: float,
    sample_name_map: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Profile the full MZmine feature table.

    Args:
        ms1_features: Zone 1 consensus metadata.
        areas: Zone 2 area matrix (features × datafile filenames).
        sample_details: Zone 3 per-sample metrics with (filename, metric)
            MultiIndex columns. ``feature_state`` is required.
        low_alignment_rate_threshold: ``alignment_scores:rate`` cutoff
            below which features are flagged as low-alignment.
        low_detection_sample_zscore: z-score cutoff (a negative number)
            for flagging samples with low detection counts.
        poor_peak_shape_threshold: tailing/asymmetry cutoff above which
            a DETECTED cell counts as poor shape.
        sample_name_map: Optional ``{datafile_filename: sample_name}`` for
            user-facing labels in the per-sample profile. Filenames absent
            from the map are kept as-is.

    Returns:
        ``(metrics, issues)``.
    """
    metrics: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []

    if "feature_state" not in sample_details.columns.get_level_values("metric"):
        raise ValueError(
            "sample_details must contain a 'feature_state' metric "
            "(Zone 3 from parse_feature_table)"
        )

    detected = (
        sample_details.xs("feature_state", axis=1, level="metric") == "DETECTED"
    )
    sample_filenames = list(detected.columns)
    n_features = int(len(ms1_features))
    n_samples = int(len(sample_filenames))

    name_map = sample_name_map or {}

    def _label(fname: str) -> str:
        return name_map.get(fname, fname)

    # ---- Table shape ------------------------------------------------------
    if n_samples > 0 and n_features > 0:
        density = float(detected.to_numpy().mean())
    else:
        density = float("nan")

    per_sample_detected = detected.sum(axis=0).astype(int)  # per sample
    per_feature_detected = detected.sum(axis=1).astype(int)  # per feature

    metrics["table_shape"] = {
        "n_features": n_features,
        "n_samples": n_samples,
        "density": density,
        "features_per_sample": _five_number(per_sample_detected),
    }

    # ---- Per-feature detection profile -----------------------------------
    metrics["per_feature_detection"] = {
        "detection_histogram": _detection_histogram(
            per_feature_detected, n_features=n_features, n_samples=n_samples
        ),
        "n_detected_stats": _five_number(per_feature_detected, with_min_max=False),
    }

    # ---- Per-sample detection profile ------------------------------------
    sample_rows = []
    for fname, count in per_sample_detected.sort_values().items():
        frac = float(count / n_features) if n_features > 0 else float("nan")
        sample_rows.append(
            {
                "sample": _label(fname),
                "n_detected": int(count),
                "fraction_of_features": frac,
            }
        )
    flagged_samples: list[str] = []
    if n_samples >= 2:
        mean_det = float(per_sample_detected.mean())
        std_det = float(per_sample_detected.std(ddof=0))
        if std_det > 0:
            zscore = (per_sample_detected - mean_det) / std_det
            cutoff = -abs(low_detection_sample_zscore)
            flagged_filenames = list(zscore[zscore < cutoff].index)
            flagged_samples = [_label(f) for f in flagged_filenames]
    metrics["per_sample_detection"] = {
        "per_sample_detected": sample_rows,
        "flagged_samples": flagged_samples,
    }
    for s in flagged_samples:
        issues.append(
            {
                "type": "low_detection_sample",
                "severity": "warning",
                "details": {"sample": s},
            }
        )

    # ---- Alignment quality -----------------------------------------------
    alignment_stats: dict[str, dict[str, float]] = {}
    for col in _ALIGNMENT_METRICS:
        if col in ms1_features.columns:
            alignment_stats[col] = _quantile_stats(
                pd.to_numeric(ms1_features[col], errors="coerce"),
                quantiles=(0.25, 0.5, 0.75, 0.95),
            )

    n_low_align = 0
    frac_low_align = float("nan")
    rate_col = "alignment_scores:rate"
    if rate_col in ms1_features.columns and n_features > 0:
        rates = pd.to_numeric(ms1_features[rate_col], errors="coerce")
        n_low_align = int((rates < low_alignment_rate_threshold).sum())
        frac_low_align = float(n_low_align / n_features)
    metrics["alignment_quality"] = {
        "stats": alignment_stats,
        "n_low_alignment_rate": n_low_align,
        "fraction_low_alignment_rate": frac_low_align,
        "low_alignment_rate_threshold": float(low_alignment_rate_threshold),
    }
    if not np.isnan(frac_low_align):
        issues.append(
            {
                "type": "low_alignment_rate",
                "severity": "warning" if frac_low_align > LOW_ALIGN_FRACTION else "info",
                "details": {
                    "fraction": frac_low_align,
                    "n_features": n_low_align,
                    "threshold": float(low_alignment_rate_threshold),
                },
            }
        )

    # ---- Chromatographic quality -----------------------------------------
    detected_mask_arr = detected.to_numpy()

    def _detected_cells(metric: str) -> np.ndarray:
        if metric not in sample_details.columns.get_level_values("metric"):
            return np.array([], dtype=float)
        mat = sample_details.xs(metric, axis=1, level="metric")
        mat = mat.reindex(columns=sample_filenames)
        arr = mat.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        vals = arr[detected_mask_arr]
        return vals[np.isfinite(vals)]

    fwhm_vals = _detected_cells("fwhm")
    fwhm_per_sample: list[dict[str, Any]] = []
    if "fwhm" in sample_details.columns.get_level_values("metric"):
        fwhm_mat = sample_details.xs("fwhm", axis=1, level="metric")
        fwhm_mat = fwhm_mat.reindex(columns=sample_filenames)
        fwhm_arr = fwhm_mat.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        masked = np.where(detected_mask_arr, fwhm_arr, np.nan)
        for j, fname in enumerate(sample_filenames):
            col_vals = masked[:, j]
            col_vals = col_vals[np.isfinite(col_vals)]
            med = float(np.median(col_vals)) if col_vals.size else float("nan")
            fwhm_per_sample.append({"sample": _label(fname), "median_fwhm": med})

    tailing_vals = _detected_cells("tailing_factor")
    asym_vals = _detected_cells("asymmetry_factor")

    metrics["chromatographic_quality"] = {
        "fwhm_global": _quantile_stats(
            pd.Series(fwhm_vals), quantiles=(0.25, 0.5, 0.75)
        ),
        "fwhm_per_sample": fwhm_per_sample,
        "tailing_factor": _quantile_stats(
            pd.Series(tailing_vals), quantiles=(0.25, 0.5, 0.75, 0.95)
        ),
        "asymmetry_factor": _quantile_stats(
            pd.Series(asym_vals), quantiles=(0.25, 0.5, 0.75, 0.95)
        ),
        "fraction_poor_tailing": _fraction_above(
            tailing_vals, poor_peak_shape_threshold
        ),
        "fraction_poor_asymmetry": _fraction_above(
            asym_vals, poor_peak_shape_threshold
        ),
        "poor_peak_shape_threshold": float(poor_peak_shape_threshold),
    }
    cq = metrics["chromatographic_quality"]
    if (
        (not np.isnan(cq["fraction_poor_tailing"])
         and cq["fraction_poor_tailing"] > POOR_SHAPE_FRACTION)
        or (not np.isnan(cq["fraction_poor_asymmetry"])
            and cq["fraction_poor_asymmetry"] > POOR_SHAPE_FRACTION)
    ):
        issues.append(
            {
                "type": "poor_peak_shape",
                "severity": "warning",
                "details": {
                    "fraction_poor_tailing": cq["fraction_poor_tailing"],
                    "fraction_poor_asymmetry": cq["fraction_poor_asymmetry"],
                    "threshold": float(poor_peak_shape_threshold),
                },
            }
        )

    # ---- MS2 coverage -----------------------------------------------------
    ms2_metrics: dict[str, Any] = {}
    frac_with_ms2 = float("nan")
    if "fragment_scans" in ms1_features.columns and n_features > 0:
        scans = pd.to_numeric(ms1_features["fragment_scans"], errors="coerce").fillna(0)
        has_ms2 = scans > 0
        frac_with_ms2 = float(has_ms2.mean())
        ms2_metrics["fraction_with_ms2"] = frac_with_ms2
        # Decile by mean detected area (intensity); rank features by area
        # so deciles map directly to relative intensity bins.
        feature_intensity = areas.mean(axis=1, skipna=True)
        feature_intensity = feature_intensity.reindex(ms1_features.index)
        decile_table = _ms2_by_intensity_decile(feature_intensity, has_ms2)
        ms2_metrics["ms2_by_intensity_decile"] = decile_table
    else:
        ms2_metrics["fraction_with_ms2"] = frac_with_ms2
        ms2_metrics["ms2_by_intensity_decile"] = {}
    metrics["ms2_coverage"] = ms2_metrics
    if not np.isnan(frac_with_ms2) and frac_with_ms2 < LOW_MS2_FRACTION:
        issues.append(
            {
                "type": "low_ms2_coverage",
                "severity": "info",
                "details": {"fraction_with_ms2": frac_with_ms2},
            }
        )

    # ---- Mass and RT coverage --------------------------------------------
    mz_range = {"min": float("nan"), "max": float("nan")}
    rt_range = {"min": float("nan"), "max": float("nan")}
    mz_rt_density: list[dict[str, Any]] = []
    if {"mz", "rt"}.issubset(ms1_features.columns) and n_features > 0:
        mz = pd.to_numeric(ms1_features["mz"], errors="coerce").to_numpy()
        rt = pd.to_numeric(ms1_features["rt"], errors="coerce").to_numpy()
        finite = np.isfinite(mz) & np.isfinite(rt)
        mz_f = mz[finite]
        rt_f = rt[finite]
        if mz_f.size > 0:
            mz_range = {"min": float(mz_f.min()), "max": float(mz_f.max())}
            rt_range = {"min": float(rt_f.min()), "max": float(rt_f.max())}
            mz_rt_density = _mz_rt_density_grid(mz_f, rt_f, n_bins=50)
    metrics["mass_rt_coverage"] = {
        "mz_range": mz_range,
        "rt_range": rt_range,
        "mz_rt_density": mz_rt_density,
    }

    # ---- Annotation coverage ---------------------------------------------
    pref_col = "preferred_annotation:compound_name"
    lipid_col = "lipid_annotations:lipid_annotations"
    method_col = "preferred_annotation:annotation_method"
    has_pref = pref_col in ms1_features.columns
    has_lipid = lipid_col in ms1_features.columns
    has_method = method_col in ms1_features.columns
    pref_pop = has_pref and ms1_features[pref_col].notna().any()
    lipid_pop = has_lipid and ms1_features[lipid_col].notna().any()
    method_pop = has_method and ms1_features[method_col].notna().any()
    if pref_pop or lipid_pop or method_pop:
        ann: dict[str, Any] = {}
        if has_pref and n_features > 0:
            ann["fraction_preferred_annotated"] = float(
                ms1_features[pref_col].notna().mean()
            )
        if has_lipid and n_features > 0:
            ann["fraction_lipid_annotated"] = float(
                ms1_features[lipid_col].notna().mean()
            )
        if has_method:
            counts = ms1_features[method_col].dropna().value_counts()
            ann["by_method"] = {str(k): int(v) for k, v in counts.items()}
        metrics["annotation_coverage"] = ann

    # ---- Charge state -----------------------------------------------------
    if "charge" in ms1_features.columns and n_features > 0:
        charge = pd.to_numeric(ms1_features["charge"], errors="coerce")
        n_missing = int(charge.isna().sum())
        n_one = int((charge == 1).sum())
        n_two = int((charge == 2).sum())
        n_three_plus = int((charge >= 3).sum())
        denom = float(n_features)
        metrics["charge_state"] = {
            "counts": {
                "1": n_one,
                "2": n_two,
                "3+": n_three_plus,
                "missing": n_missing,
            },
            "fractions": {
                "1": n_one / denom,
                "2": n_two / denom,
                "3+": n_three_plus / denom,
                "missing": n_missing / denom,
            },
        }

    # ---- Density / singleton-fraction issues -----------------------------
    if not np.isnan(density) and density < SPARSE_TABLE_DENSITY:
        issues.append(
            {
                "type": "sparse_table",
                "severity": "warning",
                "details": {"density": density},
            }
        )

    if n_features > 0:
        singleton_frac = float((per_feature_detected == 1).sum() / n_features)
        if singleton_frac > SINGLETON_FRACTION:
            issues.append(
                {
                    "type": "high_singleton_fraction",
                    "severity": "warning",
                    "details": {"fraction": singleton_frac},
                }
            )

    return metrics, issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _five_number(s: pd.Series, *, with_min_max: bool = True) -> dict[str, float]:
    """Return summary stats. NaN if empty."""
    if len(s) == 0:
        out = {"median": float("nan"), "mean": float("nan"),
               "p25": float("nan"), "p75": float("nan")}
        if with_min_max:
            out["min"] = float("nan")
            out["max"] = float("nan")
        return out
    arr = np.asarray(s, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = {"median": float("nan"), "mean": float("nan"),
               "p25": float("nan"), "p75": float("nan")}
        if with_min_max:
            out["min"] = float("nan")
            out["max"] = float("nan")
        return out
    out = {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
    }
    if with_min_max:
        out["min"] = float(arr.min())
        out["max"] = float(arr.max())
    return out


def _quantile_stats(
    s: pd.Series, *, quantiles: tuple[float, ...]
) -> dict[str, float]:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    out: dict[str, float] = {}
    name_for = {0.25: "p25", 0.5: "median", 0.75: "p75", 0.95: "p95"}
    if arr.size == 0:
        for q in quantiles:
            out[name_for.get(q, f"q{int(q * 100)}")] = float("nan")
        return out
    for q in quantiles:
        out[name_for.get(q, f"q{int(q * 100)}")] = float(np.quantile(arr, q))
    return out


def _fraction_above(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return float("nan")
    return float((values > threshold).mean())


def _detection_histogram(
    per_feature_detected: pd.Series,
    *,
    n_features: int,
    n_samples: int,
) -> dict[str, dict[str, float]]:
    """Bin per-feature detection counts; relabel the top bin from n_samples.

    For n_samples >= 27 the canonical bins apply unchanged. For smaller
    tables, bins whose lower bound is above n_samples are dropped and the
    last surviving bin is relabeled to reflect saturation at n_samples.
    """
    if n_samples >= 27:
        bins = list(_DETECTION_BINS)
    else:
        bins = [b for b in _DETECTION_BINS if b[1] <= max(n_samples, 1)]
        if not bins:
            bins = [("1", 1, 1)]
        # Top bin extends up to (and including) n_samples; label includes
        # "_or_all" so the saturation read is unambiguous.
        last_label, last_lo, _ = bins[-1]
        if last_lo == n_samples:
            new_label = f"{n_samples}_or_all"
        else:
            new_label = f"{last_lo}_to_{n_samples}_or_all"
        bins[-1] = (new_label, last_lo, None)

    counts = per_feature_detected.to_numpy(dtype=int)
    out: dict[str, dict[str, float]] = {}
    denom = float(n_features) if n_features > 0 else float("nan")
    for label, lo, hi in bins:
        if hi is None:
            mask = counts >= lo
        else:
            mask = (counts >= lo) & (counts <= hi)
        n = int(mask.sum())
        frac = float(n / denom) if denom and not np.isnan(denom) else float("nan")
        out[label] = {"count": n, "fraction": frac}
    return out


def _ms2_by_intensity_decile(
    feature_intensity: pd.Series, has_ms2: pd.Series
) -> dict[str, float]:
    """Fraction with MS2 in each intensity decile, deciles labeled 1..10."""
    df = pd.DataFrame(
        {
            "intensity": pd.to_numeric(feature_intensity, errors="coerce"),
            "has_ms2": has_ms2.astype(bool),
        }
    ).dropna(subset=["intensity"])
    if df.empty:
        return {}
    # qcut with duplicates="drop" handles intensity ties.
    try:
        df["decile"] = pd.qcut(
            df["intensity"], q=10, labels=False, duplicates="drop"
        ) + 1
    except ValueError:
        return {}
    grouped = df.groupby("decile")["has_ms2"].mean()
    return {str(int(k)): float(v) for k, v in grouped.items()}


def _mz_rt_density_grid(
    mz: np.ndarray, rt: np.ndarray, *, n_bins: int
) -> list[dict[str, Any]]:
    """Return non-empty 2D histogram cells as a list of dicts."""
    if mz.size == 0:
        return []
    counts, mz_edges, rt_edges = np.histogram2d(
        mz, rt, bins=n_bins,
    )
    cells: list[dict[str, Any]] = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            n = int(counts[i, j])
            if n == 0:
                continue
            cells.append(
                {
                    "mz_bin": float((mz_edges[i] + mz_edges[i + 1]) / 2.0),
                    "rt_bin": float((rt_edges[j] + rt_edges[j + 1]) / 2.0),
                    "n_features": n,
                }
            )
    return cells
