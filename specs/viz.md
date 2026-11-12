# `metab viz` — static figures (v1)

Three figure types share the same shape: a pure function in
`src/metabproject/viz/<name>.py` that takes DataFrames and returns a
`matplotlib.figure.Figure`, a `MetabProject.viz_<name>` method that saves it
under `<project>/viz/` as both PNG and SVG and appends a manifest entry, and a
Click subcommand under `metab viz`.

## Output

- `<project>/viz/<basename>.png` (300 DPI) + `<basename>.svg` (vector).
- Manifest entry with `command: "viz_<name>"`, the parameters used, and an
  `outputs` list of relative paths.

## Shared style — `viz/_style.py`

- `group_palette(metadata, group_col="group", custom=None) -> {label: colour}`
  — Set2 palette by default, in first-appearance order. The optional
  `custom` dict overrides individual entries; missing keys fall back to
  Set2. Same palette used by `viz/qc.py` so QC and downstream figures share
  group colours.
- `parse_palette_string("k1=#hex,k2=#hex") -> {k: hex}` — parses the CLI
  `--palette` flag.
- `load_palette_yaml(project_root) -> {label: colour}` — loads the optional
  `<project>/viz.yaml` and returns its `palette:` mapping (or `{}` if the
  file or block is absent). See "Project-level palette" below.
- `save_figure(fig, out_dir, basename, dpi=300) -> (png, svg)` — writes both
  formats and creates the directory if missing.

### Project-level palette — `viz.yaml`

To avoid re-typing `--palette` on every `pca` / `heatmap` call, drop a
`viz.yaml` at the project root:

```yaml
palette:
  red: "#C0392B"
  cream: "#E8C999"
  brown: "#6D4C41"
```

`metab viz pca`, `metab viz heatmap`, and `metab viz heatmap-annotated` read
this file automatically. The CLI `--palette` flag still works and overrides
matching keys from `viz.yaml`. Groups missing from both fall back to Set2.
`metab viz volcano` does not use group colours and ignores `viz.yaml`.

`viz.yaml` may also include a `markers:` block, used by `metab viz pca`
together with `--marker-col` to pin specific matplotlib markers to labels:

```yaml
palette:
  red:    "#C0392B"
  cream:  "#E8C999"
  brown:  "#6D4C41"
markers:
  control:  o
  malted:   ^
  dehulled: s
```

Labels not listed under `markers:` fall back to the default cycle
(`o, s, ^, D, v, P, X, *, <, >`) in encounter order, skipping any shapes
already claimed by an override. `heatmap` has no marker concept and ignores
this block.

## `metab viz pca PROJECT`

Fits `sklearn.decomposition.PCA` on `feature_table.T` (samples × features),
plots PC_i vs PC_j coloured by `metadata[group_col]`, annotates axes with
variance explained. By default also writes a 3D companion figure
(`pca_3d.{png,svg}`) of PC1/PC2/PC3.

| Option | Default | Notes |
| --- | --- | --- |
| `--group-col` | `group` | Metadata column for colour. |
| `--components 1,2` | `1,2` | Two 1-indexed PCs. |
| `--3d` | off | Make the 3D plot the primary output. |
| `--no-3d` | off | Skip the 3D companion when running with 2D primary. |
| `--palette "k=#hex,..."` | none | Override Set2 colours per group. |
| `--marker-col` | none | Second metadata column → marker shape (orthogonal legend). |
| `--basename` | `pca` (or `pca_3d`, `pca_pc1_pc3`) | Filename stem. |
| `--scale/--no-scale` | `--scale` | Autoscale features before PCA, same as `stats.multivariate.run_pca(scale=True)`, so the figure matches the reported PCA. `--no-scale` only centres. |

The `--palette` + `--marker-col` combination gives a two-factor
style — variety encoded by colour, treatment by
marker shape, two stacked legends.

Non-finite values in `feature_table` are mean-imputed per feature (cheap
fallback — log-transformed processed data should already be finite).

## `metab viz volcano PROJECT CONTRAST`

Reads `stats/<contrast>.parquet`. Plots `log2fc` (x) vs `-log10(p_adj)` (y).
Points coloured by direction at the **current** `--fc` / `--q` thresholds, so
re-running with different cutoffs recolours without re-running stats. Zero or
negative `p_adj` values are clipped to the smallest positive value before
`-log10`.

| Option | Default | Notes |
| --- | --- | --- |
| `--fc` | `1.0` | \|log2FC\| guideline and up/down split. |
| `--q` | `0.05` | BH-adjusted p-value guideline. |
| `--label-top-n` | `0` | Label the N most-significant features by ID. |
| `--basename` | `volcano_<contrast>` | Filename stem. |

## `metab viz heatmap PROJECT`

Wraps `seaborn.clustermap`. Row-z-scoring on by default. Column colour bar
from `metadata[group_col]`. Zero-variance rows are dropped before z-scoring.

| Option | Default | Notes |
| --- | --- | --- |
| `--features-from-contrast` | `None` | Stats contrast name; selects features with `p_adj < --q`. |
| `--q` | `0.05` | Threshold for `--features-from-contrast`. |
| `--group-col` | `group` | Metadata column for the (primary) column colour bar. |
| `--group-col-2` | `None` | Optional second metadata column rendered as an extra colour bar above the primary one. |
| `--palette "k=#hex,..."` | none | Override Set2 colours per group. |
| `--z-score` | `rows` | Or `none`. |
| `--cluster-rows / --no-cluster-rows` | on | |
| `--cluster-cols / --no-cluster-cols` | on | |
| `--cmap` | `vlag` | Matplotlib colormap. |
| `--basename` | `heatmap` | Filename stem. |

## `metab viz heatmap-annotated PROJECT`

Same renderer as `heatmap`, restricted to features carrying a **high-confidence
structure annotation** in the merged annotation table: feature is kept when
either `annotation_source` contains `"gnps"` **or** `sirius_high_confidence` is
`True`, and `compound_name` is non-null. Suspect-only and CANOPUS-only matches
are excluded. Row labels are the annotated compound name (`feature_id` kept as
fallback when missing). Requires `metab annotate` to have been run.

| Option | Default | Notes |
| --- | --- | --- |
| `--features-from-contrast` | `None` | Further restricts to features with `p_adj < --q` in that contrast. |
| `--q` | `0.05` | Threshold for `--features-from-contrast`. |
| `--group-col` | `group` | Metadata column for the (primary) column colour bar. |
| `--group-col-2` | `None` | Optional second metadata column rendered as an extra colour bar above the primary one. |
| `--palette "k=#hex,..."` | none | Override Set2 colours per group. |
| `--z-score` | `rows` | Or `none`. |
| `--cluster-rows / --no-cluster-rows` | on | |
| `--cluster-cols / --no-cluster-cols` | on | |
| `--cmap` | `vlag` | Matplotlib colormap. |
| `--basename` | `heatmap_annotated` | Filename stem. |

## Python API

```python
from metabproject.project import MetabProject

p = MetabProject("my_project")

# 2D + 3D PCA in one call, with the original-notebook palette.
fig = p.viz_pca(
    group_col="variety",
    marker_col="treatment",
    palette={"red": "#C0392B", "cream": "#E8C999", "brown": "#6D4C41"},
)

# Volcano for one contrast.
fig = p.viz_volcano("red_malted_vs_control")

# Heatmap of significant features for that contrast.
fig = p.viz_heatmap(features_from_contrast="red_malted_vs_control")
```

Each method returns the primary `Figure` so it can be tweaked in a notebook
before re-saving.

## `metab viz canopus-violin PROJECT CONTRAST`

Paired class-vs-background log2fc violins — the visual companion to
`metab enrich`. For each selected CANOPUS class at the chosen level, draws
a coloured violin of the class's per-feature `log2fc` (from
`stats/<contrast>.parquet`) next to a grey violin of the background log2fc
distribution. Class selection prefers `enrichment/<contrast>.parquet`
filtered by `--q` and `--direction`, ordered by `mean_log2fc` ascending
(down-regulated leftmost). If enrichment hasn't been run, falls back to
ranking classes by the mean |log2fc| of their features and emits a warning.

The background is "all annotated features at this level that are not in
any plotted class" — a deliberate viz simplification of the per-class
"universe minus this class" background that the Mann-Whitney test in
`metab enrich` uses. The shape of the difference between class and
background is the same; only the exact denominator differs.

| Option | Default | Notes |
| --- | --- | --- |
| `CONTRAST` | required positional | Stats contrast name. |
| `--level` | `npc_class` | One of `npc_pathway`, `npc_superclass`, `npc_class`, `classyfire_class`. |
| `--top-n` | `15` | Max classes to render. |
| `--direction` | `both` | `up` / `down` / `both` (enrichment selection only). |
| `--q` | `0.05` | Enrichment q-value gate; pass `--q 1.0` to ignore. |
| `--min-class-size` | `3` | Drop classes with fewer foreground features. |
| `--classes "A,B,C"` | none | Explicit class list (overrides selection logic). |
| `--fg-color` | first entry in `viz.yaml` palette, else Set2 | Foreground violin colour. |
| `--basename` | `canopus_violin_<level>_<contrast>` | Filename stem. |

Requires `metab annotate` (for `canopus_annotations`) and `metab stats`
(for the contrast). `metab enrich` is recommended but not required.

## Future extensions (not in v1)

- Spotlight: top-N per-feature boxplots with significance brackets.
- Annotation views: InChIKey Venn, SIRIUS-database UpSet.
- Class forest plot: multi-level NPC summary.

---

## v1.1 additions — multivariate helpers

Three new figure helpers in `viz/` and matching `MetabProject` methods:

- `viz/boxplot.py :: feature_cell_means_figure(feature_id, feature_table,
  metadata, factors, omnibus_row=None, annotation_row=None, palette=None)`
  — per-feature box+strip plot grouped by `factor1` on the x-axis,
  coloured by `factor2`. Adjusted p-values and `classification` from
  `omnibus_row` go into the suptitle; `annotation_row` adds the compound
  name + source. Exposed as `MetabProject.viz_feature_boxplot(...)`.
- `viz/volcano.py :: volcano_grid(panels, fc_threshold, q_threshold,
  annotated_features=None, label_lookup=None, ncols=3, row_labels=None,
  suptitle=None)` — multi-panel volcano grid sharing
  `volcano_figure`'s single-panel renderer (now factored into
  `_render_volcano_ax`). `volcano_figure` itself gains
  `annotated_features` (black-edged points) and `label_lookup` (label
  text by feature id, e.g. compound name). Adapter:
  `tukey_pair_to_contrast(posthoc, factor, group_A, group_B)` converts a
  Tukey post-hoc subframe to the `[log2fc, p_adj]` schema expected by the
  volcanoes. Exposed as `MetabProject.viz_volcano_pairs(comparison_name,
  factor, pairs, annotated=True, restrict_to_buckets=None, row_labels=None,
  ...)` — pulls `posthoc.parquet` / `omnibus.parquet` from
  `<project>/stats/<comparison_name>/`.
- `viz/heatmap.py :: pairwise_count_heatmap(count_matrix,
  group_separators=None, ...)` — square heatmap of a symmetric cell-pair
  count matrix with masked diagonal and optional separator lines
  (e.g. `[3, 6]` for a 3 × 3-block layout). Exposed as
  `MetabProject.viz_pairwise_count_heatmap(comparison_name, factors,
  group_separators=None, cells=None)`.

Plus three small helpers wrapping `stats.multivariate` results:
`MetabProject.viz_rf_importance(rf_result, top_n)`,
`viz_rf_oob_curve(rf_result)`, `viz_hca_dendrogram(hca_result,
label_col=None)` — each saves PNG + SVG under `viz/` and appends a
manifest entry.

Volcano colours and the annotated-feature edge style continue to use the
single-panel `_COLORS` palette (`#d6604d` / `#4393c3` / `#bdbdbd`).
