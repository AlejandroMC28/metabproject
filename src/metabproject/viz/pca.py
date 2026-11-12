"""PCA scatter for processed feature tables.

Fits ``sklearn.decomposition.PCA`` on samples × features, scatters two (or three)
principal components colored by a metadata group column, and annotates each
axis with the variance explained. Optionally encodes a second metadata column
via marker shape (e.g. variety=colour × treatment=shape).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from metabproject.viz._style import group_palette

# Marker cycle used when --marker-col is set. Falls back through these in
# order of metadata.<marker_col>.unique().
_MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def pca_figure(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str = "group",
    components: tuple[int, int] = (1, 2),
    three_d: bool = False,
    palette: dict | None = None,
    marker_col: str | None = None,
    markers: dict | None = None,
    title: str | None = None,
    scale: bool = True,
) -> tuple[plt.Figure, dict[str, list[float]]]:
    """Build a PCA scatter plot from a processed feature table.

    Args:
        feature_table: Processed feature table, features × samples.
        metadata: Sample metadata indexed by sample name, with ``group_col``
            (and ``marker_col`` if provided).
        group_col: Column to colour points by.
        components: Two 1-indexed PC numbers (ignored when ``three_d``).
        three_d: If True, return a 3D scatter of PC1/PC2/PC3.
        palette: Optional ``{group_label: colour}`` mapping. Missing keys fall
            back to the Set2 default.
        marker_col: Optional second metadata column whose values are encoded
            via marker shape. Triggers a second legend.
        markers: Optional ``{label: matplotlib_marker}`` override (e.g.
            ``{"control": "o", "malted": "^"}``). Labels missing from this
            dict fall back to ``_MARKER_CYCLE`` in encounter order.
        title: Optional figure title.
        scale: If True, autoscale features (``StandardScaler``) before PCA,
            matching ``stats.multivariate.run_pca(scale=True)``. If False,
            PCA only centres the data.

    Returns:
        ``(fig, info)`` with ``info = {"variance_explained": [...]}``.
    """
    samples = [s for s in feature_table.columns if s in metadata.index]
    if not samples:
        raise ValueError(
            "No overlap between feature_table columns and metadata index."
        )
    X = feature_table[samples].to_numpy(dtype=float).T

    # Cheap fallback: mean-impute any non-finite values per feature.
    if not np.all(np.isfinite(X)):
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        bad = ~np.isfinite(X)
        X = np.where(bad, np.broadcast_to(col_means, X.shape), X)

    if scale:
        X = StandardScaler().fit_transform(X)

    n_pc = 3 if three_d else max(components)
    n_pc = min(n_pc, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_pc)
    scores = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.tolist()

    colour_map = group_palette(metadata, group_col=group_col, custom=palette)
    group_labels = metadata.loc[samples, group_col].tolist()

    if marker_col is not None:
        if marker_col not in metadata.columns:
            raise KeyError(
                f"marker_col {marker_col!r} not in metadata "
                f"(have: {metadata.columns.tolist()})"
            )
        marker_values = metadata.loc[samples, marker_col].tolist()
        marker_unique = list(dict.fromkeys(marker_values))
        overrides = {str(k): v for k, v in (markers or {}).items()}
        fallback_iter = iter(
            m for m in _MARKER_CYCLE if m not in set(overrides.values())
        )
        marker_map = {}
        for m in marker_unique:
            if str(m) in overrides:
                marker_map[m] = overrides[str(m)]
            else:
                marker_map[m] = next(fallback_iter, _MARKER_CYCLE[-1])
    else:
        marker_values = ["o"] * len(samples)
        marker_unique = []
        marker_map = {"o": "o"}

    if three_d:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        idx_components = (0, 1, 2)
    else:
        i, j = components[0] - 1, components[1] - 1
        if max(i, j) >= n_pc:
            raise ValueError(
                f"Requested PC{max(components)} but only {n_pc} components are available."
            )
        fig, ax = plt.subplots(figsize=(7, 6))
        idx_components = (i, j)

    # Plot one scatter call per (group, marker) combination so legend handles
    # stay consistent and orthogonal.
    seen_marker_keys = marker_unique if marker_col else ["__single__"]
    for g, c in colour_map.items():
        for mk in seen_marker_keys:
            if marker_col:
                mask = np.array(
                    [(gl == g) and (mv == mk) for gl, mv in zip(group_labels, marker_values)]
                )
                marker = marker_map[mk]
            else:
                mask = np.array([gl == g for gl in group_labels])
                marker = "o"
            if not mask.any():
                continue
            xs = scores[mask][:, idx_components[0]]
            ys = scores[mask][:, idx_components[1]]
            kwargs = dict(
                color=c, marker=marker, s=80, alpha=0.9,
                edgecolor="black", linewidth=0.7,
            )
            if three_d:
                zs = scores[mask][:, idx_components[2]]
                ax.scatter(xs, ys, zs, **kwargs)
            else:
                ax.scatter(xs, ys, **kwargs)

    if three_d:
        ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}%)")
        ax.set_zlabel(f"PC3 ({var_exp[2] * 100:.1f}%)")
    else:
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xlabel(f"PC{components[0]} ({var_exp[idx_components[0]] * 100:.1f}%)")
        ax.set_ylabel(f"PC{components[1]} ({var_exp[idx_components[1]] * 100:.1f}%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Group legend (colour).
    group_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colour_map[g],
               markersize=9, markeredgecolor="black", label=str(g))
        for g in colour_map
    ]
    leg1 = ax.legend(
        handles=group_handles, title=group_col,
        loc="upper left", fontsize=8, title_fontsize=9, frameon=True,
    )

    # Optional marker legend (shape) — second legend added separately.
    if marker_col and marker_unique:
        ax.add_artist(leg1)
        marker_handles = [
            Line2D([0], [0], marker=marker_map[m], color="w",
                   markerfacecolor="lightgray", markersize=9,
                   markeredgecolor="black", label=str(m))
            for m in marker_unique
        ]
        ax.legend(
            handles=marker_handles, title=marker_col,
            loc="lower left", fontsize=8, title_fontsize=9, frameon=True,
        )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    return fig, {"variance_explained": var_exp}
