"""Per-class Mann-Whitney U enrichment test against a log2FC background.

For each CANOPUS class at a given level, tests whether the foreground
(features assigned to the class with probability ≥ the annotate threshold)
has a significantly different log2FC distribution from the background,
using a two-sided Mann-Whitney U test with BH-FDR correction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, norm

CANOPUS_LEVELS: tuple[str, ...] = (
    "npc_pathway",
    "npc_superclass",
    "npc_class",
    "classyfire_class",
)

_RESULT_COLUMNS = [
    "level", "class_name", "n", "mean_log2fc", "p_value", "p_adj", "significant",
]


def class_enrichment(
    contrast_df: pd.DataFrame,
    annotations: pd.DataFrame,
    level: str,
    background: str = "all",
    min_class_n: int = 3,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Mann-Whitney U enrichment for all classes at one CANOPUS level.

    Ranks the universe once then derives per-class U statistics from rank
    sums, avoiding repeated O(n log n) ranking per class.

    Args:
        contrast_df: Per-feature stats DataFrame (``feature_id`` index, must
            have a ``log2fc`` column). Typically ``project.contrast(name)``.
        annotations: CANOPUS annotations (``feature_id`` index with
            ``canopus_<level>`` and ``canopus_<level>_passed`` columns).
            Typically ``project.canopus_annotations``.
        level: One of :data:`CANOPUS_LEVELS`.
        background: Universe used as the background for each class test:

            - ``"all"`` — all features with finite ``log2fc``, excluding the
              foreground. Reproduces the reference pipeline.
            - ``"level"`` — features with ``canopus_<level>_passed == True``,
              excluding the foreground (GSEA-style).
            - ``"any"`` — features with at least one ``canopus_*_passed ==
              True``, excluding the foreground.
        min_class_n: Minimum foreground size to test a class. Classes below
            this threshold are silently skipped.
        alpha: Significance threshold for the ``significant`` column.

    Returns:
        DataFrame with columns ``level, class_name, n, mean_log2fc,
        p_value, p_adj, significant``, one row per tested class, sorted by
        ``p_adj``. Empty DataFrame (same columns) if no class passes
        ``min_class_n``.

    Raises:
        ValueError: If ``level`` is not in :data:`CANOPUS_LEVELS`, if
            ``background`` is not a recognised value, or if ``contrast_df``
            has no finite ``log2fc`` values.
    """
    if level not in CANOPUS_LEVELS:
        raise ValueError(
            f"Unknown level {level!r}. Must be one of {CANOPUS_LEVELS}."
        )
    if background not in ("all", "level", "any"):
        raise ValueError(
            f"Unknown background {background!r}. Must be 'all', 'level', or 'any'."
        )

    class_col = f"canopus_{level}"
    passed_col = f"canopus_{level}_passed"

    _EMPTY = pd.DataFrame(columns=_RESULT_COLUMNS)

    if passed_col not in annotations.columns:
        warnings.warn(
            f"[enrich] Skipping level '{level}': column '{passed_col}' not found "
            f"in annotations. Re-run 'metab annotate' to generate it.",
            stacklevel=2,
        )
        return _EMPTY

    # Build working table: log2fc joined with class annotation.
    fc_series = contrast_df["log2fc"]
    finite_mask = np.isfinite(fc_series)
    if not finite_mask.any():
        raise ValueError(
            "contrast_df has no finite log2fc values — cannot run enrichment."
        )
    fc_all = fc_series[finite_mask]

    # Build background universe for "any" mode once.
    any_passed_ids: set | None = None
    if background == "any":
        any_passed_mask = pd.Series(False, index=annotations.index)
        for lvl in CANOPUS_LEVELS:
            pc = f"canopus_{lvl}_passed"
            if pc in annotations.columns:
                any_passed_mask |= annotations[pc].fillna(False)
        any_passed_ids = set(annotations.index[any_passed_mask])

    # Passed annotations joined with finite fc.
    ann_passed = annotations.loc[
        annotations.index.isin(fc_all.index)
        & annotations[passed_col].fillna(False)
    ]

    # Multi-label values ("Chromanes|Coumarins", from GNPS NPClassifier) count
    # the feature in each of their classes.
    class_series = ann_passed[class_col].dropna().astype(str).str.split("|").explode()
    if class_series.empty:
        return _EMPTY

    # ------------------------------------------------------------------
    # Determine universe and rank once
    # ------------------------------------------------------------------
    if background == "all":
        universe_fc = fc_all
    elif background == "level":
        universe_fc = fc_all.loc[fc_all.index.isin(ann_passed.index)]
    else:  # "any"
        universe_fc = fc_all.loc[fc_all.index.isin(any_passed_ids)]  # type: ignore[arg-type]

    N = len(universe_fc)
    if N < 2:
        return _EMPTY

    rank_arr = universe_fc.rank().to_numpy(dtype=np.float64)
    fc_arr = universe_fc.to_numpy(dtype=np.float64)
    pos_map = {fid: i for i, fid in enumerate(universe_fc.index)}

    # Tie correction: sum(t^3 - t) over groups of tied values.
    _, tie_counts = np.unique(fc_arr, return_counts=True)
    tie_term = float(np.sum(tie_counts.astype(np.int64) ** 3 - tie_counts))

    # ------------------------------------------------------------------
    # Per-class U statistic from rank sums
    # ------------------------------------------------------------------
    valid_classes = class_series[class_series.index.isin(pos_map)]

    rows: list[dict] = []
    for class_name, group in valid_classes.groupby(valid_classes):
        positions = np.array(
            [pos_map[fid] for fid in group.index], dtype=np.intp,
        )
        n1 = len(positions)
        if n1 < min_class_n:
            continue
        n2 = N - n1
        if n2 < 1:
            continue

        R1 = float(rank_arr[positions].sum())
        U1 = R1 - n1 * (n1 + 1) / 2.0

        mu = n1 * n2 / 2.0
        sigma2 = (n1 * n2 / 12.0) * (
            (N + 1) - tie_term / (N * (N - 1))
        )
        # Two-sided normal approximation (matches scipy asymptotic).
        if sigma2 <= 0:
            p = 1.0
        else:
            z = abs(U1 - mu) / np.sqrt(sigma2)
            p = min(2.0 * norm.sf(z), 1.0)

        rows.append(
            {
                "level": level,
                "class_name": class_name,
                "n": n1,
                "mean_log2fc": float(fc_arr[positions].mean()),
                "p_value": float(p),
            }
        )

    if not rows:
        return _EMPTY

    result = pd.DataFrame(rows)

    # BH-FDR within this (contrast × level) block.
    p_adj = false_discovery_control(result["p_value"].to_numpy(), method="bh")
    result["p_adj"] = p_adj
    result["significant"] = result["p_adj"] < alpha

    return result.sort_values("p_adj").reset_index(drop=True)
