# `stats.comparisons` Specification

## Overview

`Comparison` is the **factorial / multi-level** counterpart to
:class:`metabproject.stats.contrasts.Contrast`. A `Contrast` names a
pairwise A-vs-B comparison; a `Comparison` names an omnibus statistical
question parameterised by one or more factors (metadata columns), with
an optional post-hoc. Both YAML keys (`contrasts:` and `comparisons:`)
can live in the same file — the two loaders ignore each other's
top-level key.

Library-only — no CLI subcommand, no manifest entry, no plotting
helpers. The user picks `kind` explicitly in the YAML; the
diagnostics module (`shapiro_per_feature` / `levene_per_feature`)
informs that choice but does not make it. No automatic
parametric/non-parametric dispatch.

`Contrast` and `welch_ttest` are untouched.

---

## Public surface

```python
from metabproject.stats import (
    Comparison,
    ComparisonResult,
    load_comparisons,
    run_comparison,
    # omnibus runners
    anova_1way, anova_2way, welch_anova, kruskal_wallis,
    # post-hocs
    tukey_hsd, games_howell, dunn,
)
```

---

## `Comparison` dataclass

```python
@dataclass
class Comparison:
    name: str                       # filename-safe slug
    kind: str                       # 'anova_1way' | 'anova_2way' |
                                    # 'welch_anova' | 'kruskal'
    factors: list[str]              # metadata column names
    posthoc: str | None = None      # 'tukey' | 'games_howell' | 'dunn' | None
    alpha: float = 0.05
```

### Kind → factor count → allowed post-hoc

| `kind`        | `len(factors)` | allowed `posthoc`              |
|---------------|----------------|--------------------------------|
| `anova_1way`  | 1              | `None`, `tukey`, `games_howell`|
| `anova_2way`  | 2              | `None`, `tukey`                |
| `welch_anova` | 1              | `None`, `games_howell`         |
| `kruskal`     | 1              | `None`, `dunn`                 |

---

## `load_comparisons(path)`

```yaml
comparisons:
  - name: variety_main
    kind: anova_1way
    factors: [variety]
    posthoc: tukey
  - name: full_factorial
    kind: anova_2way
    factors: [variety, treatment]
    posthoc: tukey
  - name: variety_nonparam
    kind: kruskal
    factors: [variety]
    posthoc: dunn
```

### Validation (at load)

- `name`: non-empty string matching `^[A-Za-z0-9_.-]+$`; globally unique.
- `kind`: one of the four supported values.
- `factors`: list of strings, no duplicates, exact length for the kind.
- `posthoc`: `None` or a value allowed for the kind.
- `alpha`: float in `(0, 1)` — defaults to `0.05`.

Factor column **existence in metadata** is validated at runtime by
`run_comparison`, not at load time (metadata isn't known yet).

---

## Omnibus runners

All return a per-feature DataFrame indexed by `feature_id`. BH-FDR is
applied via the shared `_util.bh_adjust` helper.

### `anova_1way(feature_table, metadata, factor, min_samples_per_group=3, alpha=0.05, n_jobs=-1)`

Classical one-way ANOVA via `pingouin.anova`. Columns:
`F, p_value, p_adj, significant, group_means` (dict per feature).

### `welch_anova(feature_table, metadata, factor, min_samples_per_group=3, alpha=0.05, n_jobs=-1)`

Welch's one-way ANOVA via `pingouin.welch_anova` (heteroscedasticity-
robust). Same shape as `anova_1way`.

### `kruskal_wallis(feature_table, metadata, factor, min_samples_per_group=3, alpha=0.05, n_jobs=-1)`

Non-parametric one-way via `pingouin.kruskal`. Columns:
`H, p_value, p_adj, significant, group_medians`.

### `anova_2way(feature_table, metadata, factors, min_samples_per_cell=2, alpha=0.05, n_jobs=-1)`

Two-way ANOVA with interaction via `pingouin.anova`. Three p-values
(factor1 main, factor2 main, interaction) are BH-corrected
**independently** across features. Columns (with `{f1}`/`{f2}` replaced
by the literal factor names):

```
F_{f1}, p_{f1}, p_{f1}_adj, sig_{f1},
F_{f2}, p_{f2}, p_{f2}_adj, sig_{f2},
F_interaction, p_interaction, p_interaction_adj, sig_interaction,
classification
```

`classification` is one of `ns`, `factor1_only`, `factor2_only`,
`both_no_interaction`, `interaction_only`, `interaction`. Order-
sensitive resolution: `interaction_only` requires *only* the
interaction significant; `interaction` requires the interaction AND at
least one main effect.

### Skip behaviour (all omnibus runners)

A feature is skipped — NaN cells, `significant=False`, empty
means/medians dict — when any group (or cell, for 2-way) has fewer than
`min_samples_per_group` / `min_samples_per_cell` non-NaN values. Skipped
features are excluded from BH-FDR.

### Parallelism (all omnibus + post-hoc runners)

Every per-feature routine accepts `n_jobs: int | None = -1`. Fan-out is
done by the shared helper `metabproject.stats._util.parallel_per_feature`,
which uses joblib's **threading** backend. Threads are the right choice
here because:

- pingouin / scipy / scikit-posthocs spend most of their time in numpy
  C code that releases the GIL, so threads scale;
- threading avoids pickling the whole feature table to subprocesses,
  which would dominate runtime on wide tables.

`n_jobs=-1` (default) uses all cores. `n_jobs=None` or `n_jobs=1`
forces the serial loop (useful for benchmarking or debugging). Results
are bit-identical to the serial path.

---

## Post-hoc runners

All take `features_to_test: list[str]` and return a **long-format**
DataFrame (one row per feature × pairwise comparison). The common
`factor` column lets the runner stack 2-way Tukey rows from different
factors into a single frame. All accept `n_jobs: int | None = -1` with
the same semantics as the omnibus runners.

### `tukey_hsd` / `games_howell`

Columns: `feature, factor, group_A, group_B, mean_A, mean_B, diff,
p_value, significant, hedges_g`. The `p_value` column carries Tukey's
`p_tukey` (Tukey HSD) or Games-Howell's `pval` (Welch-corrected
variant). Both paths use `scipy.stats.studentized_range.sf(q, k, df)`
for the tail probability — continuous down to numerical precision, with
no ~1e-3 floor (statsmodels' `psturng` lookup table is intentionally
avoided so highly-significant pairs don't pile onto the same
`-log10(p) = 3` ceiling on volcano plots).

### `dunn`

Columns: `feature, factor, group_A, group_B, rank_mean_diff, p_value,
significant`. `rank_mean_diff = mean_rank(A) − mean_rank(B)` after
ranking all non-NaN observations across the groups present for that
feature. `p_adjust` (default `'fdr_bh'`) is applied across pairwise
comparisons **within each feature** (not across features).

Empty `features_to_test` returns a zero-row DataFrame with the correct
columns.

---

## `run_comparison(comparison, feature_table, metadata) -> ComparisonResult`

Dispatches on `comparison.kind`:

| `kind`        | omnibus           | post-hoc routing                                                      |
|---------------|-------------------|-----------------------------------------------------------------------|
| `anova_1way`  | `anova_1way`      | features with `significant` → `tukey_hsd` or `games_howell`           |
| `welch_anova` | `welch_anova`     | features with `significant` → `games_howell`                          |
| `kruskal`     | `kruskal_wallis`  | features with `significant` → `dunn`                                  |
| `anova_2way`  | `anova_2way`      | bucket-routed Tukey (see below)                                       |

### 2-way Tukey routing

Per `classification` bucket:

| Bucket                | Tukey on                                                  |
|-----------------------|-----------------------------------------------------------|
| `factor1_only`        | factor1                                                   |
| `factor2_only`        | factor2                                                   |
| `both_no_interaction` | factor1 AND factor2 (two stacked sets of rows)            |
| `interaction`         | combined `{factor1}_{factor2}` column (all cells pairwise)|
| `interaction_only`    | combined `{factor1}_{factor2}` column                     |
| `ns`                  | no post-hoc rows                                          |

The combined factor is a temporary column
`_combined_{factor1}_x_{factor2}` added to a copy of the metadata.
Rows from all three Tukey calls are concatenated, sorted by
`feature, factor, group_A`, and returned as a single DataFrame.

### `ComparisonResult`

```python
@dataclass
class ComparisonResult:
    comparison_name: str
    kind: str
    omnibus: pd.DataFrame                  # indexed by feature_id
    posthoc: pd.DataFrame | None           # long-format, or None
```

### Runtime validation (in `run_comparison`)

- Factor columns exist in `metadata.columns`.
- Each factor has ≥2 distinct non-NaN levels.
- `metadata.index ⊆ feature_table.columns`.

All raise `ValueError` with an informative message.

---

## Idempotency / reproducibility

All routines are pure functions of their inputs; pingouin and
scikit-posthocs are deterministic for these tests, so no seed is
needed. No project mutation, no manifest entry.

---

## Dependencies

Added to `pyproject.toml`:

- `pingouin>=0.5` — `pg.anova`, `pg.welch_anova`, `pg.kruskal`,
  `pg.pairwise_tukey`, `pg.pairwise_gameshowell`.
- `scikit-posthocs>=0.8` — `sp.posthoc_dunn`.

---

## Out of scope

- No changes to `Contrast`, `contrasts.py`, or `welch_ttest`.
- No PERMANOVA dispatch via `Comparison` (stays in
  `stats.multivariate`).
- No CLI subcommand, no manifest entry, no plotting helpers.
- No mixed-effects / repeated-measures ANOVA.
- No automatic parametric/non-parametric switching from a normality
  test — the user picks `kind` explicitly.

---

## Export integration (v1.1)

`run_comparison` now optionally drives a full export step in one call (only
for `kind="anova_2way"`). Set `output_dir` plus any of:

- `export_buckets=True` — writes one `bucket_<classification>.tsv` per
  ANOVA bucket under `<output_dir>/buckets/`, each enriched with per-cell
  mean abundance, `max_cell`, `min_cell`, `range_cell`.
- `export_pairs=True` — splits the combined-factor Tukey rows into
  `pair_within_<f1>_<level>_<lvlA>_vs_<lvlB>.tsv`,
  `pair_within_<f2>_<level>_<lvlA>_vs_<lvlB>.tsv`, and
  `crosscorner_all.tsv`. Columns are canonicalised to `group_A < group_B`
  with `diff = mean_B - mean_A`, joined with `annotation_table`
  (source / compound_name / inchikey_2d plus the full taxonomy —
  `npc_pathway`, `npc_superclass`, `npc_class`, `classyfire_class`,
  `classyfire_subclass`, `classyfire_level5` — all surfaced as `ann_*`),
  classified by ANOVA bucket, and only significant rows are written.
  Missing annotation columns are left as `NA` so the schema stays stable
  across projects with partial annotations. Schema matches
  `PAIR_TSV_COLUMNS` in `stats/exports.py`.
- `count_matrix=True` — writes `<output_dir>/pairwise_count_matrix.tsv`,
  the symmetric integer matrix of significant Tukey pairs per cell pair.
- `plot_features=True` — writes one boxplot PNG (`feature_<id>.png/svg`)
  per non-`ns` feature using `viz.boxplot.feature_cell_means_figure`.
- `residual_plots=True` — writes a single aggregate diagnostic figure
  `<output_dir>/residuals_diagnostics.png` (+ `.svg`) for the full 2-way
  model fit. Four panels: pooled standardised residual QQ plot
  (overall normality), per-feature Shapiro-Wilk p-value histogram
  (distribution of normality across features — uniform under H0,
  spike near 0 quantifies violations), residuals-vs-fitted hexbin
  (homoscedasticity), and per-feature R² histogram (validity of fit).
  In the same step, a per-feature parquet
  `<output_dir>/residual_diagnostics_per_feature.parquet` is written
  with columns `in_full_model`, `r2`, `residual_sigma`,
  `max_abs_std_resid`, `shapiro_p`, `shapiro_p_adj` (BH across features),
  and `normal_residuals` — sort it to find features whose 2-way ANOVA
  assumptions look shaky. Headline stats (`n_features`, `n_samples`,
  `median_r2`, `frac_shapiro_reject`) are recorded in
  `export_summary["residuals"]`. Implementation in
  `metabproject.stats.anova_diagnostics.anova_2way_residual_diagnostics`.

For single-feature follow-up, the same module exposes
`feature_residual_diagnostics_figure(feature_id, feature_table,
metadata, factors)` — a 2-panel figure (standardised-residual QQ +
residual histogram with N(0, σ̂) overlay) used to eyeball whether a
flagged feature's Shapiro rejection is one outlier point, a heavy
tail, or genuine non-normality. `MetabProject.viz_feature_residuals(
feature_id, factors, basename=None)` wraps it, saves PNG + SVG to
`viz/`, and appends a manifest entry.

The returned `ComparisonResult` carries the resulting paths
(`bucket_dir`, `pair_dir`, `feature_plots_dir`, `count_matrix_path`,
`residuals_path`, `residual_per_feature_path`, `output_dir`) and an
`export_summary` dict with
per-bucket, per-file, and residual-diagnostic counts. Helper functions
live in `metabproject.stats.exports` (`export_buckets`,
`export_tukey_pair_tsvs`, `pairwise_count_matrix`,
`export_pairwise_count_matrix`, `combined_factor_name`).

`MetabProject.run_comparison(...)` exposes the same flags, defaults
`output_dir` to `<project>/stats/<comparison.name>/`, persists
`omnibus.parquet` + `posthoc.parquet` so the viz wrappers can be re-run
later, and writes a single manifest entry.
