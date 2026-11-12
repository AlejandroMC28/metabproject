"""Diagnostic figures for ``metab feature-qc`` output.

Two figures generated off a ``__pairs.tsv``:

- Scatter of ΔRT vs Pearson(area), colored by classification, size encodes
  shared-detection count.
- Histogram of ΔRT / median-FWHM faceted by classification.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_CLASS_ORDER = ["alignment_like", "split_like", "isomer_like", "unclassified"]


def feature_qc_overview_figure(
    metrics: dict,
    issues: list[dict] | None = None,
) -> plt.Figure:
    """Build the 2×3 panel overview figure.

    Panels: detection histogram, per-sample detection, alignment-rate
    distribution, FWHM distribution, MS2 coverage by intensity decile,
    m/z × RT density.

    Args:
        metrics: ``metrics`` dict produced by
            :func:`metabproject.qc.overview.compute_overview`.
        issues: Issues list (used to highlight flagged samples in panel 2).

    Returns:
        The matplotlib Figure.
    """
    issues = issues or []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    _panel_detection_histogram(axes[0, 0], metrics)
    _panel_per_sample_detection(axes[0, 1], metrics)
    _panel_alignment_rate(axes[0, 2], metrics)
    _panel_fwhm_distribution(axes[1, 0], metrics)
    _panel_ms2_deciles(axes[1, 1], metrics)
    _panel_mz_rt_density(axes[1, 2], metrics)

    fig.suptitle("Feature-QC overview", fontsize=13)
    fig.tight_layout()
    return fig


def _panel_detection_histogram(ax: plt.Axes, metrics: dict) -> None:
    hist = metrics.get("per_feature_detection", {}).get("detection_histogram", {})
    labels = list(hist.keys())
    counts = [hist[k].get("count", 0) for k in labels]
    if labels:
        ax.bar(labels, counts, color="#4c72b0", edgecolor="white")
    ax.set_title("Per-feature detection histogram", fontsize=10)
    ax.set_xlabel("n samples detected")
    ax.set_ylabel("n features")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_per_sample_detection(ax: plt.Axes, metrics: dict) -> None:
    block = metrics.get("per_sample_detection", {})
    rows = block.get("per_sample_detected", [])
    flagged = set(block.get("flagged_samples", []))
    samples = [r["sample"] for r in rows]
    counts = [r["n_detected"] for r in rows]
    colors = ["#c44e52" if s in flagged else "#55a868" for s in samples]
    if samples:
        ax.bar(range(len(samples)), counts, color=colors, edgecolor="white")
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(samples, rotation=60, ha="right", fontsize=7)
    ax.set_title("Per-sample n_detected (red = flagged)", fontsize=10)
    ax.set_ylabel("n features")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_alignment_rate(ax: plt.Axes, metrics: dict) -> None:
    aq = metrics.get("alignment_quality", {})
    stats = aq.get("stats", {}).get("alignment_scores:rate", {})
    median = stats.get("median")
    threshold = aq.get("low_alignment_rate_threshold")
    # We don't have the raw rates here. Plot the summary stats as vertical
    # markers on a 0..1 axis so the panel still tells a story.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for q_key, q_label, color in (
        ("p25", "p25", "#dd8452"),
        ("median", "median", "#4c72b0"),
        ("p75", "p75", "#dd8452"),
        ("p95", "p95", "#a55194"),
    ):
        val = stats.get(q_key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.axvline(val, color=color, linestyle="--", linewidth=1.2,
                   label=f"{q_label}={val:.2f}")
    if threshold is not None:
        ax.axvline(threshold, color="#c44e52", linewidth=1.5,
                   label=f"threshold={threshold}")
    ax.set_title("Alignment-rate quartiles", fontsize=10)
    ax.set_xlabel("alignment_scores:rate")
    ax.set_yticks([])
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if median is not None and not (isinstance(median, float) and np.isnan(median)):
        ax.text(
            0.98, 0.05,
            f"frac_low={metrics['alignment_quality'].get('fraction_low_alignment_rate', float('nan')):.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#555",
        )


def _panel_fwhm_distribution(ax: plt.Axes, metrics: dict) -> None:
    cq = metrics.get("chromatographic_quality", {})
    per_sample = cq.get("fwhm_per_sample", [])
    medians = [r.get("median_fwhm") for r in per_sample
               if r.get("median_fwhm") is not None
               and not (isinstance(r.get("median_fwhm"), float)
                        and np.isnan(r.get("median_fwhm")))]
    if medians:
        ax.hist(medians, bins=12, color="#8172b3", edgecolor="white")
    fg = cq.get("fwhm_global", {})
    med = fg.get("median")
    if med is not None and not (isinstance(med, float) and np.isnan(med)):
        ax.axvline(med, color="#c44e52", linewidth=1.2,
                   label=f"global median={med:.3f}")
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("Per-sample median FWHM", fontsize=10)
    ax.set_xlabel("FWHM (min)")
    ax.set_ylabel("n samples")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_ms2_deciles(ax: plt.Axes, metrics: dict) -> None:
    deciles = metrics.get("ms2_coverage", {}).get("ms2_by_intensity_decile", {})
    items = sorted(deciles.items(), key=lambda kv: int(kv[0]))
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    if labels:
        ax.bar(labels, vals, color="#937860", edgecolor="white")
    ax.set_title("MS2 coverage by intensity decile", fontsize=10)
    ax.set_xlabel("decile (1=low, 10=high)")
    ax.set_ylabel("fraction with MS2")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_mz_rt_density(ax: plt.Axes, metrics: dict) -> None:
    grid = metrics.get("mass_rt_coverage", {}).get("mz_rt_density", [])
    if not grid:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="#888")
        ax.set_axis_off()
        return
    mz = [c["mz_bin"] for c in grid]
    rt = [c["rt_bin"] for c in grid]
    n = [c["n_features"] for c in grid]
    sc = ax.scatter(rt, mz, c=n, cmap="viridis", s=18, edgecolor="none")
    ax.set_title("m/z × RT density", fontsize=10)
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("m/z")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("n features", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def feature_qc_figures(
    pairs_df: pd.DataFrame,
    *,
    palette: dict | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Build the scatter + histogram diagnostic figures from a pairs table.

    Args:
        pairs_df: Output of :func:`metabproject.qc.detect_duplicates`.
        palette: Optional ``{classification: color}`` override.

    Returns:
        ``(scatter_fig, hist_fig)``.
    """
    classes = [c for c in _CLASS_ORDER if c in set(pairs_df.get("classification", []))]
    if not classes:
        classes = _CLASS_ORDER
    default_colors = dict(zip(_CLASS_ORDER, sns.color_palette("Set2", n_colors=4)))
    if palette:
        default_colors.update(palette)
    color_map = {c: default_colors[c] for c in classes}

    scatter_fig = _scatter(pairs_df, color_map, classes)
    hist_fig = _histogram(pairs_df, color_map, classes)
    return scatter_fig, hist_fig


def _scatter(
    pairs_df: pd.DataFrame,
    color_map: dict,
    classes: list[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    if pairs_df.empty:
        ax.text(0.5, 0.5, "No suspect pairs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    sizes_raw = pairs_df["n_both_detected"].fillna(0).to_numpy(dtype=float)
    size_floor, size_scale = 20.0, 8.0
    sizes = size_floor + size_scale * sizes_raw

    for cls in classes:
        mask = pairs_df["classification"] == cls
        if not mask.any():
            continue
        ax.scatter(
            pairs_df.loc[mask, "delta_rt"],
            pairs_df.loc[mask, "pearson_area"],
            s=sizes[mask.to_numpy()],
            color=color_map[cls],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            label=cls,
        )

    ax.axhline(0, color="#888", linewidth=0.5)
    ax.axvline(0, color="#888", linewidth=0.5)
    ax.set_xlabel("ΔRT (min)")
    ax.set_ylabel("Pearson(area) on shared-detected samples")
    ax.set_title("Feature-QC: pair area-correlation vs ΔRT")
    ax.legend(title="classification", frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _histogram(
    pairs_df: pd.DataFrame,
    color_map: dict,
    classes: list[str],
) -> plt.Figure:
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]

    series = pairs_df.get("drt_over_fwhm", pd.Series(dtype=float))
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        bins = np.linspace(0, 1, 11)
    else:
        upper = max(finite.max(), 1e-6)
        bins = np.linspace(0, upper * 1.05, 21)

    for ax, cls in zip(axes, classes):
        sub = pairs_df.loc[pairs_df["classification"] == cls, "drt_over_fwhm"]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        if not sub.empty:
            ax.hist(sub, bins=bins, color=color_map[cls], edgecolor="white")
        else:
            ax.text(0.5, 0.5, "n=0", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel("|ΔRT| / median FWHM")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Pairs")
    fig.suptitle("Feature-QC: ΔRT / FWHM by classification")
    fig.tight_layout()
    return fig
