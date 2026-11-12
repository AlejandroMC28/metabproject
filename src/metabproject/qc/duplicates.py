"""Duplicate-feature detection for MZmine alignment/deconvolution failures.

Pairs of features that lie within an m/z + RT window are scored on detection
overlap, peak-area correlation, and FWHM-scaled RT separation, then bucketed
into ``alignment_like``, ``split_like``, ``isomer_like`` or ``unclassified``.

Inputs are the raw three-zone DataFrames returned by
:func:`metabproject.io.mzmine.parse_feature_table` — no imputation or
transformation should have been applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Classification thresholds (module-level so tuning is explicit and visible).
ALIGNMENT_JACCARD_MAX = 0.3
ALIGNMENT_DRT_OVER_FWHM_MIN = 0.5
SPLIT_JACCARD_MIN = 0.7
SPLIT_DRT_OVER_FWHM_MAX = 1.0
SPLIT_PEARSON_MIN = 0.7
ISOMER_JACCARD_MIN = 0.7
ISOMER_DRT_OVER_FWHM_MIN = 1.5
MIN_SHARED_FOR_PEARSON = 3


def detect_duplicates(
    ms1_features: pd.DataFrame,
    areas: pd.DataFrame,
    sample_details: pd.DataFrame,
    *,
    mz_tol_ppm: float,
    rt_tol: float,
    min_correlation: float,
    use_fwhm: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Identify suspect feature pairs from m/z + RT proximity and score them.

    Args:
        ms1_features: Zone 1 consensus metadata indexed by ``feature_id``,
            requires ``mz`` and ``rt`` columns.
        areas: Zone 2 peak areas (features × samples). Column labels must
            match the first level of ``sample_details`` columns.
        sample_details: Zone 3 per-sample metrics with ``(filename, metric)``
            MultiIndex columns. ``feature_state`` is required;
            ``fwhm`` is used when ``use_fwhm`` is True.
        mz_tol_ppm: m/z window (ppm) for candidate pairing.
        rt_tol: RT window (minutes) for candidate pairing.
        min_correlation: Pearson cutoff for the ``split_like`` classification
            (overrides :data:`SPLIT_PEARSON_MIN` when stricter).
        use_fwhm: If True, compute ``drt_over_fwhm`` from Zone 3 ``fwhm``;
            otherwise leave it NaN and use ΔRT alone for the alignment-like
            rule.

    Returns:
        ``(pairs_df, flags_df, summary_dict)``.
    """
    if "mz" not in ms1_features.columns or "rt" not in ms1_features.columns:
        raise ValueError("ms1_features must contain 'mz' and 'rt' columns")

    feat = ms1_features[["mz", "rt"]].copy()
    # Carry preferred_annotation through when present.
    ann_col = "preferred_annotation:compound_name"
    if ann_col in ms1_features.columns:
        feat[ann_col] = ms1_features[ann_col]

    # Index sort by m/z for the sliding-window pair enumeration.
    feat = feat.sort_values("mz")
    ids = feat.index.to_numpy()
    mz = feat["mz"].to_numpy(dtype=float)
    rt = feat["rt"].to_numpy(dtype=float)

    # Build per-feature detection mask + area row once, aligned to a common
    # sample column order.
    detected = _build_detected_mask(sample_details, ids)
    aligned_areas = areas.reindex(index=ids, columns=detected.columns)
    area_arr = aligned_areas.to_numpy(dtype=float)
    detected_arr = detected.to_numpy(dtype=bool)

    fwhm_arr = None
    if use_fwhm:
        fwhm_arr = _build_fwhm_matrix(sample_details, ids, detected.columns)

    pearson_min = max(min_correlation, SPLIT_PEARSON_MIN)

    rows: list[dict] = []
    n = len(ids)
    j_start = 1
    for i in range(n - 1):
        mz_i = mz[i]
        rt_i = rt[i]
        ppm_window = mz_i * mz_tol_ppm / 1e6
        mz_upper = mz_i + ppm_window
        j = max(j_start, i + 1)
        while j < n and mz[j] <= mz_upper:
            if abs(rt[j] - rt_i) <= rt_tol:
                rows.append(
                    _score_pair(
                        i, j, ids, mz, rt,
                        area_arr, detected_arr, fwhm_arr,
                        pearson_min=pearson_min,
                        use_fwhm=use_fwhm,
                    )
                )
            j += 1

    pairs_df = pd.DataFrame(rows, columns=_pair_columns())
    if rows and ann_col in feat.columns:
        ann_map = feat[ann_col].to_dict()
        pairs_df["preferred_annotation_i"] = pairs_df["feature_id_i"].map(ann_map)
        pairs_df["preferred_annotation_j"] = pairs_df["feature_id_j"].map(ann_map)
    else:
        pairs_df["preferred_annotation_i"] = pd.NA
        pairs_df["preferred_annotation_j"] = pd.NA

    flags_df = _build_flags(pairs_df)
    summary = _build_summary(pairs_df, flags_df)
    return pairs_df, flags_df, summary


def _pair_columns() -> list[str]:
    return [
        "feature_id_i", "feature_id_j",
        "mz_i", "mz_j", "rt_i", "rt_j",
        "delta_rt", "delta_ppm",
        "n_both_detected", "n_only_i", "n_only_j",
        "jaccard_detected", "pearson_area",
        "median_fwhm", "drt_over_fwhm",
        "classification",
    ]


def _build_detected_mask(
    sample_details: pd.DataFrame, feature_order: np.ndarray
) -> pd.DataFrame:
    """Return a ``features × samples`` bool DataFrame of DETECTED status."""
    if "feature_state" not in sample_details.columns.get_level_values("metric"):
        raise ValueError(
            "sample_details must contain a 'feature_state' metric "
            "(Zone 3 from parse_feature_table)"
        )
    state = sample_details.xs("feature_state", axis=1, level="metric")
    detected = (state == "DETECTED")
    return detected.reindex(feature_order)


def _build_fwhm_matrix(
    sample_details: pd.DataFrame,
    feature_order: np.ndarray,
    sample_order: pd.Index,
) -> np.ndarray | None:
    """Return a ``features × samples`` FWHM array, or None if absent."""
    metrics = sample_details.columns.get_level_values("metric")
    if "fwhm" not in metrics:
        return None
    fwhm = sample_details.xs("fwhm", axis=1, level="metric")
    fwhm = fwhm.reindex(index=feature_order, columns=sample_order)
    return fwhm.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _score_pair(
    i: int,
    j: int,
    ids: np.ndarray,
    mz: np.ndarray,
    rt: np.ndarray,
    area_arr: np.ndarray,
    detected_arr: np.ndarray,
    fwhm_arr: np.ndarray | None,
    *,
    pearson_min: float,
    use_fwhm: bool,
) -> dict:
    di = detected_arr[i]
    dj = detected_arr[j]
    both = di & dj
    only_i = di & ~dj
    only_j = dj & ~di

    n_both = int(both.sum())
    n_only_i = int(only_i.sum())
    n_only_j = int(only_j.sum())

    denom = n_both + n_only_i + n_only_j
    jaccard = float(n_both / denom) if denom > 0 else float("nan")

    if n_both >= MIN_SHARED_FOR_PEARSON:
        a = area_arr[i, both]
        b = area_arr[j, both]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) \
                and np.std(a) > 0 and np.std(b) > 0:
            pearson = float(np.corrcoef(a, b)[0, 1])
        else:
            pearson = float("nan")
    else:
        pearson = float("nan")

    median_fwhm = float("nan")
    drt_over_fwhm = float("nan")
    delta_rt = float(rt[j] - rt[i])
    if use_fwhm and fwhm_arr is not None:
        # FWHM is a per-feature peak-width attribute; take the median across
        # each feature's own DETECTED samples (not just the shared subset,
        # which is empty for disjoint-detection alignment-failure pairs).
        vals = np.concatenate([fwhm_arr[i, di], fwhm_arr[j, dj]])
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            median_fwhm = float(np.median(vals))
            if median_fwhm > 0:
                drt_over_fwhm = abs(delta_rt) / median_fwhm

    classification = _classify(
        jaccard=jaccard,
        drt_over_fwhm=drt_over_fwhm,
        pearson=pearson,
        delta_rt=delta_rt,
        use_fwhm=use_fwhm,
        pearson_min=pearson_min,
    )

    delta_ppm = abs(mz[j] - mz[i]) / mz[i] * 1e6 if mz[i] > 0 else float("nan")

    return {
        "feature_id_i": ids[i],
        "feature_id_j": ids[j],
        "mz_i": mz[i],
        "mz_j": mz[j],
        "rt_i": rt[i],
        "rt_j": rt[j],
        "delta_rt": delta_rt,
        "delta_ppm": delta_ppm,
        "n_both_detected": n_both,
        "n_only_i": n_only_i,
        "n_only_j": n_only_j,
        "jaccard_detected": jaccard,
        "pearson_area": pearson,
        "median_fwhm": median_fwhm,
        "drt_over_fwhm": drt_over_fwhm,
        "classification": classification,
    }


def _classify(
    *,
    jaccard: float,
    drt_over_fwhm: float,
    pearson: float,
    delta_rt: float,
    use_fwhm: bool,
    pearson_min: float,
) -> str:
    if np.isnan(jaccard):
        return "unclassified"

    if jaccard < ALIGNMENT_JACCARD_MAX:
        if not np.isnan(drt_over_fwhm):
            if drt_over_fwhm > ALIGNMENT_DRT_OVER_FWHM_MIN:
                return "alignment_like"
        elif not use_fwhm and abs(delta_rt) > 0:
            return "alignment_like"

    if jaccard > ISOMER_JACCARD_MIN \
            and not np.isnan(drt_over_fwhm) \
            and drt_over_fwhm > ISOMER_DRT_OVER_FWHM_MIN:
        return "isomer_like"

    if jaccard > SPLIT_JACCARD_MIN \
            and not np.isnan(pearson) \
            and pearson > pearson_min:
        drt_ok = np.isnan(drt_over_fwhm) or drt_over_fwhm < SPLIT_DRT_OVER_FWHM_MAX
        if drt_ok:
            return "split_like"

    return "unclassified"


def _build_flags(pairs_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "feature_id", "n_pairs", "classifications", "recommended_action",
    ]
    if pairs_df.empty:
        return pd.DataFrame(columns=cols)

    long = pd.concat(
        [
            pairs_df[["feature_id_i", "classification"]].rename(
                columns={"feature_id_i": "feature_id"}
            ),
            pairs_df[["feature_id_j", "classification"]].rename(
                columns={"feature_id_j": "feature_id"}
            ),
        ],
        ignore_index=True,
    )

    grouped = long.groupby("feature_id", sort=False)
    flags = grouped.agg(
        n_pairs=("classification", "size"),
        classifications=(
            "classification",
            lambda s: ",".join(sorted(set(s))),
        ),
    ).reset_index()
    flags["recommended_action"] = flags["classifications"].map(_recommended_action)
    return flags[cols]


def _recommended_action(classes_csv: str) -> str:
    classes = set(classes_csv.split(","))
    if classes == {"split_like"}:
        return "drop_lower_intensity"
    if classes == {"isomer_like"}:
        return "keep_both"
    return "review"


def _build_summary(pairs_df: pd.DataFrame, flags_df: pd.DataFrame) -> dict:
    classes = ["alignment_like", "split_like", "isomer_like", "unclassified"]
    counts = {c: int((pairs_df["classification"] == c).sum()) for c in classes} \
        if not pairs_df.empty else {c: 0 for c in classes}

    top_flagged: list[dict] = []
    if not flags_df.empty:
        top = flags_df.sort_values("n_pairs", ascending=False).head(10)
        top_flagged = top.to_dict(orient="records")

    suggestions: list[str] = []
    if counts["split_like"] > 0:
        median_drt = float(
            pairs_df.loc[pairs_df["classification"] == "split_like", "delta_rt"]
            .abs().median()
        )
        suggestions.append(
            f"Median |ΔRT| among split_like pairs is {median_drt:.3f} min; "
            f"consider increasing MZmine deconvolution peak duration to "
            f">{median_drt * 2:.3f} min."
        )
    if counts["alignment_like"] > 0:
        median_drt_over = pairs_df.loc[
            pairs_df["classification"] == "alignment_like", "drt_over_fwhm"
        ].median()
        if pd.notna(median_drt_over):
            suggestions.append(
                f"Median ΔRT/FWHM among alignment_like pairs is "
                f"{median_drt_over:.2f}; consider widening MZmine RT tolerance "
                f"during alignment."
            )

    return {
        "counts": counts,
        "top_flagged": top_flagged,
        "suggestions": suggestions,
    }
