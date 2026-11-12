"""Click CLI entry point and subcommands.

Each subcommand is a thin wrapper that parses options and delegates to a
:class:`~metabproject.project.MetabProject` method. Entry point: ``metab``.
"""

from __future__ import annotations

import sys

import click

from metabproject import __version__


@click.group()
@click.version_option(__version__, prog_name="metab")
def main() -> None:
    """metab — project-based untargeted LC-MS/MS metabolomics processing."""


@main.command("init")
@click.argument("project", type=click.Path())
@click.option(
    "--feature-table",
    required=True,
    type=click.Path(exists=True),
    help="MZmine full feature table CSV.",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True),
    help="Sample metadata CSV (sample_file, sample_name, group[, weight]).",
)
def init_cmd(project: str, feature_table: str, metadata: str) -> None:
    """Create a new project from MZmine output. See specs/init.md."""
    from metabproject.project import MetabProject

    try:
        MetabProject.init(path=project, feature_table=feature_table, metadata=metadata)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("process")
@click.argument("project", type=click.Path(exists=True))
@click.option("--min-prevalence", type=int, default=2, show_default=True)
@click.option(
    "--prevalence-mode",
    type=click.Choice(["any_value", "detected"]),
    default="any_value",
    show_default=True,
    help="Detection criterion. 'any_value' uses (area > 0) & notna(); "
    "'detected' uses MZmine's feature_state == 'DETECTED' mask "
    "(excludes gap-filled values).",
)
@click.option(
    "--min-groups-meeting",
    type=int,
    default=1,
    show_default=True,
    help="Number of groups that must each meet --min-prevalence.",
)
@click.option(
    "--impute",
    type=click.Choice(["half-min", "second-min", "min", "lod", "knn", "none"]),
    default="half-min",
    show_default=True,
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--lod-value", type=float, default=None)
@click.option("--knn-k", type=int, default=5, show_default=True)
@click.option("--weight-normalize", is_flag=True, default=False)
@click.option(
    "--normalize",
    type=click.Choice(["none", "tic", "pqn", "median"]),
    default="none",
    show_default=True,
)
@click.option(
    "--transform",
    type=click.Choice(["log2", "log10", "sqrt", "none"]),
    default="log2",
    show_default=True,
)
@click.option(
    "--scale",
    type=click.Choice(["none", "pareto", "auto"]),
    default="none",
    show_default=True,
)
@click.option("--force", is_flag=True, default=False)
def process_cmd(
    project: str,
    min_prevalence: int,
    prevalence_mode: str,
    min_groups_meeting: int,
    impute: str,
    seed: int,
    lod_value: float | None,
    knn_k: int,
    weight_normalize: bool,
    normalize: str,
    transform: str,
    scale: str,
    force: bool,
) -> None:
    """Run the processing pipeline. See specs/process.md."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.process(
            min_prevalence=min_prevalence,
            prevalence_mode=prevalence_mode,
            min_groups_meeting=min_groups_meeting,
            impute=impute,
            seed=seed,
            lod_value=lod_value,
            knn_k=knn_k,
            weight_normalize=weight_normalize,
            normalize=normalize,
            transform=transform,
            scale=scale,
            force=force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("annotate")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--sirius-structures",
    required=True,
    type=click.Path(exists=True),
    help="SIRIUS structure_identifications.tsv.",
)
@click.option(
    "--canopus",
    required=True,
    type=click.Path(exists=True),
    help="CANOPUS canopus_structure_summary.tsv.",
)
@click.option(
    "--gnps-library",
    required=True,
    type=click.Path(exists=True),
    help="GNPS library TSV (merged_results_with_gnps.tsv).",
)
@click.option(
    "--gnps-suspect",
    default=None,
    type=click.Path(exists=True),
    help="GNPS suspect-list TSV (optional).",
)
@click.option(
    "--sirius-min-confidence",
    type=float,
    default=0.64,
    show_default=True,
    help="Minimum ConfidenceScoreApproximate for sirius_high_confidence flag.",
)
@click.option(
    "--canopus-min-probability",
    type=float,
    default=0.7,
    show_default=True,
    help="Minimum per-level probability for CANOPUS class predictions.",
)
@click.option(
    "--sirius-top-k",
    type=int,
    default=1,
    show_default=True,
    help="Store top-k SIRIUS candidates for cross-adduct validation (default 1 = rank-1 only).",
)
@click.option(
    "--gnps-polarity",
    type=click.Choice(["positive", "negative"], case_sensitive=False),
    default=None,
    help=(
        "Experiment polarity. If set, GNPS library matches whose IonMode is "
        "the opposite polarity are dropped (rows with missing IonMode are kept)."
    ),
)
@click.option(
    "--classify-sirius-structures",
    is_flag=True,
    default=False,
    help=(
        "For features with high-confidence SIRIUS structures (and no GNPS hit), "
        "query the NP Classifier web API on the rank-1 SMILES and use the "
        "result for NPC pathway/superclass/class — overrides CANOPUS predictions."
    ),
)
@click.option(
    "--np-cache",
    type=click.Path(),
    default=None,
    help=(
        "Path to a NP Classifier cache parquet (keyed by InChIkey 2D). "
        "Defaults to <project>/annotations/np_classifier_cache.parquet."
    ),
)
@click.option("--force", is_flag=True, default=False)
def annotate_cmd(
    project: str,
    sirius_structures: str,
    canopus: str,
    gnps_library: str,
    gnps_suspect: str | None,
    sirius_min_confidence: float,
    canopus_min_probability: float,
    sirius_top_k: int,
    gnps_polarity: str | None,
    classify_sirius_structures: bool,
    np_cache: str | None,
    force: bool,
) -> None:
    """Merge SIRIUS, CANOPUS, and GNPS annotations. See specs/annotate.md."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.annotate(
            sirius_structures=sirius_structures,
            canopus=canopus,
            gnps_library=gnps_library,
            gnps_suspect=gnps_suspect,
            sirius_min_confidence=sirius_min_confidence,
            canopus_min_probability=canopus_min_probability,
            sirius_top_k=sirius_top_k,
            gnps_polarity=gnps_polarity.lower() if gnps_polarity else None,
            classify_sirius_structures=classify_sirius_structures,
            np_cache=np_cache,
            force=force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("network")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--pairs",
    required=True,
    type=click.Path(exists=True),
    help="GNPS FBMN filtered_pairs.tsv edge list.",
)
@click.option(
    "--min-cosine", type=float, default=0.0, show_default=True,
    help="Drop edges with cosine below this threshold.",
)
@click.option(
    "--keep-singletons", is_flag=True, default=False,
    help="Keep ComponentIndex == -1 singleton edges.",
)
@click.option("--force", is_flag=True, default=False)
def network_cmd(
    project: str,
    pairs: str,
    min_cosine: float,
    keep_singletons: bool,
    force: bool,
) -> None:
    """Import a GNPS FBMN molecular network. See specs/network.md."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.import_network(
            pairs=pairs,
            min_cosine=min_cosine,
            drop_singletons=not keep_singletons,
            force=force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("spectra")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--mgf",
    required=True,
    type=click.Path(exists=True),
    help="MZmine FBMN / GNPS MGF of MS2 spectra (keyed by feature id).",
)
@click.option("--force", is_flag=True, default=False)
def spectra_cmd(project: str, mgf: str, force: bool) -> None:
    """Import MS2 spectra from an MGF. See specs/spectra.md."""
    from metabproject.project import MetabProject

    try:
        MetabProject(project).import_spectra(mgf=mgf, force=force)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("similarity")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--tolerance",
    type=float,
    default=0.1,
    show_default=True,
    help="m/z match tolerance in Da.",
)
@click.option(
    "--score-type",
    type=click.Choice(["modified", "direct", "both"]),
    default="both",
    show_default=True,
    help="Cosine variant(s) to compute.",
)
@click.option(
    "--scaling",
    "scalings",
    multiple=True,
    type=click.Choice(["linear", "sqrt", "massbank"]),
    help="Intensity/m-z scaling preset(s). Repeatable; default: all.",
)
@click.option("--force", is_flag=True, default=False)
def similarity_cmd(
    project: str,
    tolerance: float,
    score_type: str,
    scalings: tuple[str, ...],
    force: bool,
) -> None:
    """Compute matchms spectral similarity (cosine / modified cosine).

    Scores MS2 spectrum pairs within GNPS molecular families and records the
    matched peaks for the explorer's compare panel. Requires `metab spectra`
    and `metab network` first. See specs/similarity.md.
    """
    from metabproject.project import MetabProject

    score_types = ("modified", "direct") if score_type == "both" else (score_type,)
    try:
        MetabProject(project).compute_similarity(
            tolerance=tolerance,
            score_types=score_types,
            scalings=scalings or None,
            force=force,
        )
    except (FileNotFoundError, FileExistsError, ValueError, ImportError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("stats")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--contrasts",
    "contrasts_file",
    required=True,
    type=click.Path(exists=True),
    help="YAML file defining the contrasts to run.",
)
@click.option(
    "--alpha", type=float, default=0.05, show_default=True,
    help="Significance threshold for 'direction' and 'significant' columns.",
)
@click.option(
    "--min-samples", type=int, default=3, show_default=True,
    help="Per-group minimum of non-NaN values required to test a feature.",
)
@click.option("--force", is_flag=True, default=False)
def stats_cmd(
    project: str,
    contrasts_file: str,
    alpha: float,
    min_samples: int,
    force: bool,
) -> None:
    """Run per-feature Welch t-tests for declared contrasts. See specs/stats.md."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.stats(
            contrasts=contrasts_file,
            alpha=alpha,
            min_samples=min_samples,
            force=force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("enrich")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--contrast",
    "contrast_names",
    multiple=True,
    help="Contrast name to run (repeatable). Defaults to all contrasts in stats/.",
)
@click.option(
    "--level",
    "level_names",
    multiple=True,
    type=click.Choice(["npc_pathway", "npc_superclass", "npc_class", "classyfire_class"]),
    help="CANOPUS level to test (repeatable). Defaults to all four levels.",
)
@click.option(
    "--background",
    type=click.Choice(["all", "level", "any"]),
    default="all",
    show_default=True,
    help=(
        "Background universe for each class test. "
        "'all' = all features with finite log2fc (default); "
        "'level' = features annotated at this level; "
        "'any' = features annotated at any level."
    ),
)
@click.option(
    "--min-class-n", type=int, default=3, show_default=True,
    help="Minimum foreground size to test a class.",
)
@click.option(
    "--alpha", type=float, default=0.05, show_default=True,
    help="Significance threshold for 'significant' column.",
)
@click.option("--force", is_flag=True, default=False)
def enrich_cmd(
    project: str,
    contrast_names: tuple[str, ...],
    level_names: tuple[str, ...],
    background: str,
    min_class_n: int,
    alpha: float,
    force: bool,
) -> None:
    """Run CANOPUS class enrichment (Mann-Whitney U + BH-FDR). See specs/enrich.md."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.enrich(
            contrasts=list(contrast_names) if contrast_names else None,
            levels=list(level_names) if level_names else None,
            background=background,
            min_class_n=min_class_n,
            alpha=alpha,
            force=force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("validate")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--check",
    "check_names",
    multiple=True,
    type=click.Choice(["iin"]),
    help="Validation check to run (repeatable). Defaults to all checks.",
)
@click.option(
    "--sirius-top-k",
    type=int,
    default=5,
    show_default=True,
    help="Top-k SIRIUS candidates for annotation comparison (requires annotate --sirius-top-k).",
)
@click.option("--force", is_flag=True, default=False)
def validate_cmd(
    project: str,
    check_names: tuple[str, ...],
    sirius_top_k: int,
    force: bool,
) -> None:
    """Run validation checks (IIN consistency). See specs/validate.md."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.validate(
            checks=list(check_names) if check_names else None,
            sirius_top_k=sirius_top_k,
            force=force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        import sys
        sys.exit(1)


@main.command("explore")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output HTML path. Defaults to <project>/exports/explore.html.",
)
@click.option(
    "--no-structures",
    is_flag=True,
    default=False,
    help="Skip RDKit 2D structure rendering (faster, smaller file).",
)
@click.option(
    "--cdn",
    is_flag=True,
    default=False,
    help="Use Plotly CDN instead of embedding (~3.5 MB savings, requires internet).",
)
def explore_cmd(
    project: str,
    output: str | None,
    no_structures: bool,
    cdn: bool,
) -> None:
    """Generate an interactive HTML viewer for browsing annotated features."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.explore(
            output=output,
            render_structures=not no_structures,
            embed_plotly=not cdn,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.group("viz")
def viz_group() -> None:
    """Generate static figures (PCA, volcano, heatmap). See specs/viz.md."""


def _parse_components(value: str) -> tuple[int, int]:
    """Parse a ``"1,2"``-style components option into a (i, j) tuple."""
    try:
        parts = [int(p.strip()) for p in value.split(",")]
    except ValueError as exc:
        raise click.BadParameter(f"Expected 'i,j' integers, got {value!r}") from exc
    if len(parts) != 2:
        raise click.BadParameter(f"Expected exactly two components, got {value!r}")
    return parts[0], parts[1]


@viz_group.command("pca")
@click.argument("project", type=click.Path(exists=True))
@click.option("--group-col", default="group", show_default=True)
@click.option(
    "--components", "components_str", default="1,2", show_default=True,
    help="Two PCs to plot, e.g. '1,2' or '1,3' (ignored with --3d).",
)
@click.option("--3d", "three_d", is_flag=True, default=False, help="Primary output is the 3D PC1/PC2/PC3 plot.")
@click.option("--no-3d", "skip_3d", is_flag=True, default=False,
              help="Don't also write the 3D companion figure alongside the 2D one.")
@click.option("--palette", "palette_str", default=None,
              help="Custom palette as 'key=color,key=color' (e.g. 'red=#C0392B,cream=#E8C999').")
@click.option("--marker-col", default=None,
              help="Second metadata column to encode via marker shape.")
@click.option("--basename", default=None, help="Filename stem under <project>/viz/.")
@click.option("--scale/--no-scale", default=True, show_default=True,
              help="Autoscale features before PCA (same as run_pca).")
def viz_pca_cmd(
    project: str, group_col: str, components_str: str, three_d: bool,
    skip_3d: bool, palette_str: str | None, marker_col: str | None,
    basename: str | None, scale: bool,
) -> None:
    """Render a PCA scatter (writes a 3D companion by default)."""
    from metabproject.project import MetabProject
    from metabproject.viz._style import (
        load_markers_yaml,
        load_palette_yaml,
        parse_palette_string,
    )

    try:
        palette = load_palette_yaml(project)
        if palette_str:
            palette.update(parse_palette_string(palette_str))
        palette = palette or None
        markers = load_markers_yaml(project) or None
        p = MetabProject(project)
        p.viz_pca(
            group_col=group_col,
            components=_parse_components(components_str),
            three_d=three_d,
            include_3d=not skip_3d,
            palette=palette,
            marker_col=marker_col,
            markers=markers,
            basename=basename,
            scale=scale,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@viz_group.command("volcano")
@click.argument("project", type=click.Path(exists=True))
@click.argument("contrast")
@click.option("--fc", "fc_threshold", type=float, default=1.0, show_default=True,
              help="|log2FC| threshold for the up/down split.")
@click.option("--q", "q_threshold", type=float, default=0.05, show_default=True,
              help="BH-adjusted p-value threshold.")
@click.option("--label-top-n", type=int, default=0, show_default=True,
              help="Annotate the N most-significant feature IDs.")
@click.option("--basename", default=None, help="Filename stem under <project>/viz/.")
def viz_volcano_cmd(
    project: str, contrast: str, fc_threshold: float, q_threshold: float,
    label_top_n: int, basename: str | None,
) -> None:
    """Render a volcano plot for one contrast."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.viz_volcano(
            contrast=contrast,
            fc_threshold=fc_threshold,
            q_threshold=q_threshold,
            label_top_n=label_top_n,
            basename=basename,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@viz_group.command("heatmap")
@click.argument("project", type=click.Path(exists=True))
@click.option("--features-from-contrast", "features_from_contrast", default=None,
              help="Contrast name; plots features with p_adj < --q.")
@click.option("--q", "q_threshold", type=float, default=0.05, show_default=True,
              help="Significance threshold for --features-from-contrast.")
@click.option("--group-col", default="group", show_default=True)
@click.option("--group-col-2", default=None,
              help="Optional second metadata column shown as an extra colour bar.")
@click.option("--palette", "palette_str", default=None,
              help="Custom palette as 'key=color,key=color'.")
@click.option("--z-score", type=click.Choice(["rows", "none"]), default="rows", show_default=True)
@click.option("--cluster-rows/--no-cluster-rows", default=True, show_default=True)
@click.option("--cluster-cols/--no-cluster-cols", default=True, show_default=True)
@click.option("--cmap", default="vlag", show_default=True)
@click.option("--basename", default="heatmap", show_default=True)
def viz_heatmap_cmd(
    project: str, features_from_contrast: str | None, q_threshold: float,
    group_col: str, group_col_2: str | None, palette_str: str | None, z_score: str,
    cluster_rows: bool, cluster_cols: bool, cmap: str, basename: str,
) -> None:
    """Render a clustered heatmap of the processed feature table."""
    from metabproject.project import MetabProject
    from metabproject.viz._style import load_palette_yaml, parse_palette_string

    try:
        palette = load_palette_yaml(project)
        if palette_str:
            palette.update(parse_palette_string(palette_str))
        palette = palette or None
        p = MetabProject(project)
        p.viz_heatmap(
            features_from_contrast=features_from_contrast,
            q_threshold=q_threshold,
            group_col=group_col,
            group_col_2=group_col_2,
            palette=palette,
            z_score=None if z_score == "none" else z_score,
            cluster_rows=cluster_rows,
            cluster_cols=cluster_cols,
            cmap=cmap,
            basename=basename,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@viz_group.command("heatmap-annotated")
@click.argument("project", type=click.Path(exists=True))
@click.option("--features-from-contrast", "features_from_contrast", default=None,
              help="Contrast name; further restricts to features with p_adj < --q.")
@click.option("--q", "q_threshold", type=float, default=0.05, show_default=True,
              help="Significance threshold for --features-from-contrast.")
@click.option("--group-col", default="group", show_default=True)
@click.option("--group-col-2", default=None,
              help="Optional second metadata column shown as an extra colour bar.")
@click.option("--palette", "palette_str", default=None,
              help="Custom palette as 'key=color,key=color'.")
@click.option("--z-score", type=click.Choice(["rows", "none"]), default="rows", show_default=True)
@click.option("--cluster-rows/--no-cluster-rows", default=True, show_default=True)
@click.option("--cluster-cols/--no-cluster-cols", default=True, show_default=True)
@click.option("--cmap", default="vlag", show_default=True)
@click.option("--basename", default="heatmap_annotated", show_default=True)
def viz_heatmap_annotated_cmd(
    project: str, features_from_contrast: str | None, q_threshold: float,
    group_col: str, group_col_2: str | None, palette_str: str | None, z_score: str,
    cluster_rows: bool, cluster_cols: bool, cmap: str, basename: str,
) -> None:
    """Heatmap restricted to features with high-confidence structure annotations."""
    from metabproject.project import MetabProject
    from metabproject.viz._style import load_palette_yaml, parse_palette_string

    try:
        palette = load_palette_yaml(project)
        if palette_str:
            palette.update(parse_palette_string(palette_str))
        palette = palette or None
        p = MetabProject(project)
        p.viz_heatmap_annotated(
            features_from_contrast=features_from_contrast,
            q_threshold=q_threshold,
            group_col=group_col,
            group_col_2=group_col_2,
            palette=palette,
            z_score=None if z_score == "none" else z_score,
            cluster_rows=cluster_rows,
            cluster_cols=cluster_cols,
            cmap=cmap,
            basename=basename,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@viz_group.command("canopus-violin")
@click.argument("project", type=click.Path(exists=True))
@click.argument("contrast")
@click.option(
    "--level", default="npc_class", show_default=True,
    type=click.Choice(["npc_pathway", "npc_superclass", "npc_class", "classyfire_class"]),
    help="CANOPUS taxonomy level.",
)
@click.option("--top-n", type=int, default=15, show_default=True,
              help="Max classes to render.")
@click.option(
    "--direction", default="both", show_default=True,
    type=click.Choice(["up", "down", "both"]),
    help="Filter classes by sign of mean log2fc (enrichment-based selection only).",
)
@click.option("--q", "q_threshold", type=float, default=0.05, show_default=True,
              help="Enrichment q-value gate; pass 1.0 to ignore.")
@click.option("--min-class-size", type=int, default=3, show_default=True,
              help="Drop classes with fewer foreground features.")
@click.option("--classes", "classes_str", default=None,
              help="Explicit comma-separated class list (overrides selection).")
@click.option("--fg-color", default=None,
              help="Foreground violin colour (defaults to first viz.yaml palette entry).")
@click.option("--basename", default=None, help="Filename stem under <project>/viz/.")
def viz_canopus_violin_cmd(
    project: str, contrast: str, level: str, top_n: int, direction: str,
    q_threshold: float, min_class_size: int, classes_str: str | None,
    fg_color: str | None, basename: str | None,
) -> None:
    """Render paired CANOPUS-class vs background log2fc violins."""
    from metabproject.project import MetabProject
    from metabproject.viz._style import load_palette_yaml

    try:
        if fg_color is None:
            palette = load_palette_yaml(project)
            if palette:
                fg_color = next(iter(palette.values()))
        classes = None
        if classes_str:
            classes = [c.strip() for c in classes_str.split(",") if c.strip()]
        p = MetabProject(project)
        p.viz_canopus_violin(
            contrast=contrast,
            level=level,
            top_n=top_n,
            direction=direction,
            q_threshold=None if q_threshold >= 1.0 else q_threshold,
            classes=classes,
            min_class_size=min_class_size,
            fg_color=fg_color,
            basename=basename,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("feature-qc")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--check", type=click.Choice(["alignment", "overview"]),
    default="alignment", show_default=True, help="Which QC check to run.",
)
@click.option("--name", required=True, help="Run name (output file stem).")
@click.option("--mz-tol-ppm", type=float, default=10.0, show_default=True,
              help="(alignment) m/z window for candidate pairs (ppm).")
@click.option("--rt-tol", type=float, default=0.1, show_default=True,
              help="(alignment) RT window for candidate pairs (minutes).")
@click.option("--min-correlation", type=float, default=0.7, show_default=True,
              help="(alignment) Pearson(area) cutoff for split-like.")
@click.option("--use-fwhm/--no-use-fwhm", default=True, show_default=True,
              help="(alignment) Use Zone 3 FWHM to scale ΔRT.")
@click.option("--group-by", type=str, default=None,
              help="(alignment) Metadata column to slice misalignment rates "
                   "by (e.g. 'variety' or 'treatment'). Emits a per-group "
                   "TSV and ranks groups by alignment_like_rate.")
@click.option("--min-samples-per-group", type=int, default=3, show_default=True,
              help="(alignment) Skip groups with fewer than this many samples.")
@click.option("--low-alignment-rate-threshold", type=float, default=0.10,
              show_default=True,
              help="(overview) alignment_scores:rate cutoff for low-alignment "
                   "features.")
@click.option("--low-detection-sample-zscore", type=float, default=-2.0,
              show_default=True,
              help="(overview) z-score cutoff for flagging samples with low "
                   "detection counts.")
@click.option("--poor-peak-shape-threshold", type=float, default=2.0,
              show_default=True,
              help="(overview) tailing/asymmetry cutoff above which a "
                   "DETECTED cell counts as poor shape.")
@click.option("--mzmine-raw", type=click.Path(exists=True), default=None,
              help="Override the raw MZmine CSV path stored in the init op.")
@click.option("--force", is_flag=True, default=False)
def feature_qc_cmd(
    project: str, check: str, name: str, mz_tol_ppm: float, rt_tol: float,
    min_correlation: float, use_fwhm: bool,
    group_by: str | None, min_samples_per_group: int,
    low_alignment_rate_threshold: float,
    low_detection_sample_zscore: float,
    poor_peak_shape_threshold: float,
    mzmine_raw: str | None, force: bool,
) -> None:
    """Run a feature-table QC check (alignment duplicates, or overview profile)."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.feature_qc(
            check=check,
            name=name,
            mz_tol_ppm=mz_tol_ppm,
            rt_tol=rt_tol,
            min_correlation=min_correlation,
            use_fwhm=use_fwhm,
            group_by=group_by,
            min_samples_per_group=min_samples_per_group,
            low_alignment_rate_threshold=low_alignment_rate_threshold,
            low_detection_sample_zscore=low_detection_sample_zscore,
            poor_peak_shape_threshold=poor_peak_shape_threshold,
            mzmine_raw=mzmine_raw,
            force=force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@viz_group.command("feature-qc")
@click.argument("project", type=click.Path(exists=True))
@click.option("--name", required=True, help="Feature-QC run name to visualize.")
@click.option("--basename", default=None,
              help="Filename stem under <project>/viz/ (defaults to --name).")
def viz_feature_qc_cmd(project: str, name: str, basename: str | None) -> None:
    """Render diagnostic figures for a feature_qc run."""
    from metabproject.project import MetabProject

    try:
        p = MetabProject(project)
        p.viz_feature_qc(name=name, basename=basename)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
