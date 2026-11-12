"""Shared helpers for the ``metabproject.stats`` submodules.

Internal — not part of the public surface.
"""

from __future__ import annotations

import os
from typing import Callable, Sequence, TypeVar

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import false_discovery_control, studentized_range

T = TypeVar("T")
R = TypeVar("R")


def parallel_per_feature(
    items: Sequence[T],
    worker: Callable[[T], R],
    n_jobs: int | None,
) -> list[R]:
    """Apply ``worker(item)`` over ``items``, optionally in parallel.

    Each per-feature stats call (ANOVA, Kruskal-Wallis, Tukey, ...) is
    independent, so this is a trivial map. We use joblib's ``loky``
    backend (processes) because scipy / scikit_posthocs spend
    non-trivial time in pure-Python paths and integration routines that
    don't reliably release the GIL — threads underperform in practice.
    Falls back to a plain serial loop for ``n_jobs`` in ``{None, 1}`` or
    when there's only one item.

    Args:
        items: Sequence to map over (typically feature indices).
        worker: Pure function called as ``worker(item)``.
        n_jobs: ``None`` / ``1`` → serial. ``-1`` → all cores.

    Returns:
        A list of ``worker(item)`` results in input order.
    """
    if n_jobs in (None, 1) or len(items) < 2:
        return [worker(it) for it in items]
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(worker)(it) for it in items)


def bh_adjust(p_value: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment that preserves NaNs.

    Args:
        p_value: 1-D array of raw p-values; NaNs are allowed and propagate.

    Returns:
        Array of BH-adjusted p-values the same length as ``p_value``.
    """
    p_adj = np.full_like(p_value, np.nan, dtype=float)
    valid = ~np.isnan(p_value)
    if valid.any():
        p_adj[valid] = false_discovery_control(p_value[valid], method="bh")
    return p_adj


def validate_factor_columns(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
) -> None:
    """Validate that factor columns exist with ≥2 levels and samples match.

    Args:
        feature_table: Features × samples table.
        metadata: Sample metadata indexed by sample name.
        factors: Metadata column names to check.

    Raises:
        ValueError: If any factor is missing, has fewer than 2 distinct
            non-NaN levels, or ``metadata.index`` references samples
            absent from ``feature_table.columns``.
    """
    for factor in factors:
        if factor not in metadata.columns:
            raise ValueError(
                f"factor {factor!r} not found in metadata.columns "
                f"(available: {list(metadata.columns)})"
            )
        levels = metadata[factor].dropna().unique()
        if len(levels) < 2:
            raise ValueError(
                f"factor {factor!r} must have at least 2 distinct non-NaN "
                f"levels (got {len(levels)}: {list(levels)})"
            )
    missing = [s for s in metadata.index if s not in feature_table.columns]
    if missing:
        raise ValueError(
            "metadata.index contains samples absent from feature_table.columns: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )


def _sf_batched(q: np.ndarray, k: int, df_resid) -> np.ndarray:
    """Studentized-range survival function, deduplicated by unique df.

    ``scipy.stats.studentized_range.sf`` iterates element-wise over array
    inputs, calling ``scipy.integrate.quad`` once per ``(q, df)`` pair.
    For balanced designs ``df_resid`` is constant across thousands of
    features, so collapsing to the unique df values turns thousands of
    integrations into a handful.

    Args:
        q: 1-D array of studentized-range statistics.
        k: Number of groups in the omnibus (studentized-range parameter).
        df_resid: Residual degrees of freedom. Scalar (broadcast) or
            1-D array matching ``q``.

    Returns:
        1-D array of p-values the same length as ``q``.
    """
    df_arr = np.asarray(df_resid)
    if df_arr.ndim == 0 or np.unique(df_arr).size == 1:
        df_scalar = float(df_arr) if df_arr.ndim == 0 else float(df_arr.flat[0])
        return np.asarray(studentized_range.sf(q, k, df_scalar))
    out = np.empty_like(q, dtype=float)
    uniq, inv = np.unique(df_arr, return_inverse=True)
    for u_idx, u_df in enumerate(uniq):
        mask = inv == u_idx
        out[mask] = studentized_range.sf(q[mask], k, float(u_df))
    return out


def _sf_parallel(
    q: np.ndarray,
    k: int,
    df_resid,
    n_jobs: int | None,
) -> np.ndarray:
    """Parallelised version of :func:`_sf_batched` for large q-arrays.

    Uses ``loky`` (processes) because ``scipy.integrate.quad`` doesn't
    reliably release the GIL. Small arrays (< 2000 elements) and
    ``n_jobs in {None, 0, 1}`` fall through to the serial batched path.

    Args:
        q: 1-D array of studentized-range statistics.
        k: Number of groups.
        df_resid: Residual degrees of freedom (scalar or matching ``q``).
        n_jobs: ``-1`` → all cores. ``None`` / ``1`` → serial.

    Returns:
        1-D array of p-values the same length as ``q``.
    """
    if n_jobs in (None, 0, 1) or q.size < 2000:
        return _sf_batched(q, k, df_resid)
    n_workers = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    df_arr = (
        np.broadcast_to(df_resid, q.shape)
        if np.ndim(df_resid) == 0
        else np.asarray(df_resid)
    )
    chunks = [c for c in np.array_split(np.arange(q.size), n_workers) if c.size]
    parts = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_sf_batched)(q[c], k, df_arr[c]) for c in chunks
    )
    out = np.empty_like(q, dtype=float)
    for c, p in zip(chunks, parts):
        out[c] = p
    return out
