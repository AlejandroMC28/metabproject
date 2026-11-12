"""Dispatch a :class:`Comparison` to the right omnibus + post-hoc.

The runner is the only place that knows how the four omnibus tests and
three post-hoc tests fit together. It owns the bucket-routing for
two-way ANOVA + Tukey, where features in different ``classification``
buckets get post-hocs on different (or combined) factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from metabproject.stats.anova import anova_1way, anova_2way, welch_anova
from metabproject.stats.comparisons import Comparison
from metabproject.stats.kruskal import kruskal_wallis
from metabproject.stats.posthoc import dunn, games_howell, tukey_hsd


@dataclass
class ComparisonResult:
    """Outcome of a :func:`run_comparison` call.

    Attributes:
        comparison_name: Echo of the input :class:`Comparison` name.
        kind: Echo of the input :class:`Comparison` kind.
        omnibus: Per-feature omnibus DataFrame indexed by ``feature_id``.
        posthoc: Long-format post-hoc DataFrame (one row per
            feature × pairwise comparison), or ``None`` if no post-hoc
            was requested. The schema differs by post-hoc family — see
            :mod:`metabproject.stats.posthoc`.
        output_dir: Directory under which export artefacts were written
            (``None`` when no exports were requested).
        bucket_dir, pair_dir, feature_plots_dir, count_matrix_path,
        residuals_path:
            Filled in by the export side-effects when the corresponding
            export flag was set.
        export_summary: Per-export counts (bucket → row counts, pair file
            → row counts, etc.) for downstream reporting / manifest.
    """

    comparison_name: str
    kind: str
    omnibus: pd.DataFrame
    posthoc: pd.DataFrame | None
    output_dir: Path | None = None
    bucket_dir: Path | None = None
    pair_dir: Path | None = None
    feature_plots_dir: Path | None = None
    count_matrix_path: Path | None = None
    residuals_path: Path | None = None
    residual_per_feature_path: Path | None = None
    export_summary: dict = field(default_factory=dict)


_POSTHOC_FNS = {
    "tukey": tukey_hsd,
    "games_howell": games_howell,
    "dunn": dunn,
}


def _features_with(omnibus: pd.DataFrame, mask_col: str) -> list[str]:
    return omnibus.index[omnibus[mask_col].astype(bool)].tolist()


def _run_2way_tukey(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    factors: list[str],
    omnibus: pd.DataFrame,
    alpha: float,
    exclude_cross_corner: bool = True,
) -> pd.DataFrame:
    """Stack Tukey rows across factor1, factor2, and the combined factor.

    When ``exclude_cross_corner`` is True (default), pairs from the
    combined-factor Tukey where **both** factor1 and factor2 differ are
    dropped — those comparisons confound the two main effects and aren't
    biologically interpretable in a factorial design.
    """
    f1, f2 = factors
    classification = omnibus["classification"]

    feats_for_f1 = omnibus.index[
        classification.isin({"factor1_only", "both_no_interaction"})
    ].tolist()
    feats_for_f2 = omnibus.index[
        classification.isin({"factor2_only", "both_no_interaction"})
    ].tolist()
    feats_for_combined = omnibus.index[
        classification.isin({"interaction", "interaction_only"})
    ].tolist()

    frames: list[pd.DataFrame] = []
    if feats_for_f1:
        frames.append(tukey_hsd(feature_table, metadata, f1, feats_for_f1, alpha=alpha))
    if feats_for_f2:
        frames.append(tukey_hsd(feature_table, metadata, f2, feats_for_f2, alpha=alpha))
    if feats_for_combined:
        combined_col = f"_combined_{f1}_x_{f2}"
        combined_meta = metadata.copy()
        combined_meta[combined_col] = (
            metadata[f1].astype(str) + "_" + metadata[f2].astype(str)
        )
        combined_frame = tukey_hsd(
            feature_table, combined_meta, combined_col, feats_for_combined,
            alpha=alpha,
        )
        if exclude_cross_corner and not combined_frame.empty:
            cell_to_pair = {
                f"{row[f1]}_{row[f2]}": (str(row[f1]), str(row[f2]))
                for _, row in metadata[[f1, f2]].drop_duplicates().iterrows()
            }
            a = combined_frame["group_A"].map(cell_to_pair)
            b = combined_frame["group_B"].map(cell_to_pair)
            same_f1 = a.str[0] == b.str[0]
            same_f2 = a.str[1] == b.str[1]
            keep = same_f1 | same_f2
            combined_frame = combined_frame[keep].reset_index(drop=True)
        frames.append(combined_frame)

    if not frames:
        return tukey_hsd(feature_table, metadata, f1, [], alpha=alpha)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["feature", "factor", "group_A"], kind="stable"
    ).reset_index(drop=True)


def run_comparison(
    comparison: Comparison,
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    export_buckets: bool = False,
    export_pairs: bool = False,
    plot_features: bool = False,
    count_matrix: bool = False,
    residual_plots: bool = False,
    annotation_table: pd.DataFrame | None = None,
    max_feature_plots: int | None = None,
    exclude_cross_corner: bool = True,
) -> ComparisonResult:
    """Run the omnibus and (optionally) post-hoc for a :class:`Comparison`.

    Dispatches on ``comparison.kind``:

    - ``anova_1way`` → :func:`anova_1way` (+ ``tukey`` / ``games_howell``)
    - ``welch_anova`` → :func:`welch_anova` (+ ``games_howell``)
    - ``kruskal`` → :func:`kruskal_wallis` (+ ``dunn``)
    - ``anova_2way`` → :func:`anova_2way` (+ bucket-routed ``tukey``)

    For one-factor kinds the post-hoc runs on features whose omnibus
    flagged ``significant=True``. For ``anova_2way`` the post-hoc routing
    follows :attr:`anova_2way`'s ``classification`` column: ``factor1_only``
    / ``both_no_interaction`` features get Tukey on factor1;
    ``factor2_only`` / ``both_no_interaction`` features get Tukey on
    factor2; ``interaction`` / ``interaction_only`` features get Tukey on
    a combined ``{factor1}_{factor2}`` factor so all cells are compared
    pairwise. ``ns`` features get no post-hoc.

    Args:
        comparison: The :class:`Comparison` to execute.
        feature_table: Features × samples processed table.
        metadata: Sample metadata indexed by sample name.

    Returns:
        :class:`ComparisonResult` with the omnibus DataFrame and the
        post-hoc DataFrame (or ``None`` when ``comparison.posthoc`` is
        ``None``).

    Raises:
        ValueError: If ``comparison.kind`` is unsupported, or factor
            columns fail validation (missing, single-level, or metadata
            samples absent from ``feature_table.columns``).
    """
    if comparison.kind == "anova_1way":
        omnibus = anova_1way(
            feature_table, metadata, comparison.factors[0], alpha=comparison.alpha,
        )
        posthoc_df: pd.DataFrame | None
        if comparison.posthoc is None:
            posthoc_df = None
        else:
            sig_features = _features_with(omnibus, "significant")
            posthoc_df = _POSTHOC_FNS[comparison.posthoc](
                feature_table, metadata, comparison.factors[0], sig_features,
                alpha=comparison.alpha,
            )

    elif comparison.kind == "welch_anova":
        omnibus = welch_anova(
            feature_table, metadata, comparison.factors[0], alpha=comparison.alpha,
        )
        if comparison.posthoc is None:
            posthoc_df = None
        else:
            sig_features = _features_with(omnibus, "significant")
            posthoc_df = games_howell(
                feature_table, metadata, comparison.factors[0], sig_features,
                alpha=comparison.alpha,
            )

    elif comparison.kind == "kruskal":
        omnibus = kruskal_wallis(
            feature_table, metadata, comparison.factors[0], alpha=comparison.alpha,
        )
        if comparison.posthoc is None:
            posthoc_df = None
        else:
            sig_features = _features_with(omnibus, "significant")
            posthoc_df = dunn(
                feature_table, metadata, comparison.factors[0], sig_features,
                alpha=comparison.alpha,
            )

    elif comparison.kind == "anova_2way":
        omnibus = anova_2way(
            feature_table, metadata, comparison.factors, alpha=comparison.alpha,
        )
        if comparison.posthoc is None:
            posthoc_df = None
        else:
            posthoc_df = _run_2way_tukey(
                feature_table, metadata, comparison.factors, omnibus,
                alpha=comparison.alpha,
                exclude_cross_corner=exclude_cross_corner,
            )

    else:
        raise ValueError(f"Unsupported Comparison.kind: {comparison.kind!r}")

    result = ComparisonResult(
        comparison_name=comparison.name,
        kind=comparison.kind,
        omnibus=omnibus,
        posthoc=posthoc_df,
    )

    any_export = (
        export_buckets or export_pairs or plot_features
        or count_matrix or residual_plots
    )
    if not any_export:
        return result

    if output_dir is None:
        raise ValueError("output_dir is required when any export flag is set")
    if comparison.kind != "anova_2way":
        raise ValueError(
            "exports (buckets / pairs / count_matrix / plot_features / "
            "residual_plots) are only implemented for kind='anova_2way'"
        )

    from metabproject.stats.exports import (
        export_buckets as _export_buckets,
        export_pairwise_count_matrix as _export_count_matrix,
        export_tukey_pair_tsvs as _export_pairs,
    )

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    result.output_dir = out_root
    factors = comparison.factors

    if export_buckets:
        bucket_dir = out_root / "buckets"
        counts = _export_buckets(omnibus, feature_table, metadata, factors, bucket_dir)
        result.bucket_dir = bucket_dir
        result.export_summary["buckets"] = counts

    if export_pairs:
        if posthoc_df is None:
            raise ValueError("export_pairs requires a post-hoc on the Comparison")
        pair_dir = out_root / "pairs"
        written = _export_pairs(
            posthoc_df, omnibus, pair_dir, factors,
            annotation_table=annotation_table,
        )
        result.pair_dir = pair_dir
        result.export_summary["pairs"] = written

    if count_matrix:
        if posthoc_df is None:
            raise ValueError("count_matrix requires a post-hoc on the Comparison")
        cm_path = out_root / "pairwise_count_matrix.tsv"
        _export_count_matrix(posthoc_df, cm_path, factors)
        result.count_matrix_path = cm_path

    if residual_plots:
        from metabproject.stats.anova_diagnostics import (
            anova_2way_residual_diagnostics,
        )
        from metabproject.viz._style import save_figure
        import matplotlib.pyplot as _plt

        diag = anova_2way_residual_diagnostics(
            feature_table, metadata, factors,
        )
        png_path, _ = save_figure(diag.figure, out_root, "residuals_diagnostics")
        _plt.close(diag.figure)
        result.residuals_path = png_path
        per_feature_path = out_root / "residual_diagnostics_per_feature.parquet"
        diag.per_feature.to_parquet(per_feature_path)
        result.residual_per_feature_path = per_feature_path
        result.export_summary["residuals"] = {
            "n_features": diag.n_features,
            "n_samples": diag.n_samples,
            "median_r2": diag.median_r2,
            "frac_shapiro_reject": diag.frac_shapiro_reject,
            "per_feature_table": str(per_feature_path),
        }

    if plot_features:
        from metabproject.viz._style import save_figure
        from metabproject.viz.boxplot import feature_cell_means_figure

        feat_dir = out_root / "feature_plots"
        feat_dir.mkdir(parents=True, exist_ok=True)
        result.feature_plots_dir = feat_dir

        sig_mask = omnibus["classification"].astype(str) != "ns"
        feats = list(omnibus.index[sig_mask])
        if max_feature_plots is not None:
            feats = feats[:max_feature_plots]

        plotted = 0
        for fid in feats:
            ann_row = (
                annotation_table.loc[fid]
                if annotation_table is not None and fid in annotation_table.index
                else None
            )
            try:
                import matplotlib.pyplot as _plt
                fig = feature_cell_means_figure(
                    fid, feature_table, metadata, factors,
                    omnibus_row=omnibus.loc[fid],
                    annotation_row=ann_row,
                )
                save_figure(fig, feat_dir, f"feature_{fid}")
                _plt.close(fig)
                plotted += 1
            except (ValueError, KeyError):
                continue
        result.export_summary["feature_plots"] = plotted

    return result
