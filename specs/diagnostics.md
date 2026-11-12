# `stats.diagnostics` Specification

## Overview

Sibling submodule to `stats/univariate.py` and `stats/multivariate/`.
Provides per-feature **diagnostic** tests that inform the choice between
parametric and non-parametric univariate methods but do **not** perform
inference themselves:

- **Shapiro-Wilk** normality — `shapiro_per_feature`
- **Levene's test** for variance homogeneity (Brown-Forsythe variant by
  default) — `levene_per_feature`

Pure functions, no I/O, no plotting, no CLI awareness. Both return a
DataFrame indexed by `feature_id`, with BH-FDR adjustment via the same
`scipy.stats.false_discovery_control` helper used by
`stats/univariate.py`. This is a library API only — there is **no
`metab diagnostics` command** and no automatic dispatch to a parametric
or non-parametric test based on these results. That decision stays with
the user.

---

## Public surface

```python
from metabproject.stats import shapiro_per_feature, levene_per_feature
```

Both functions take the processed `feature_table` (features × samples).
NaN cells are tolerated and excluded per feature/group; features (or
feature/group cells) with fewer than 3 non-NaN values are skipped rather
than tested.

---

## `shapiro_per_feature`

```python
shapiro_per_feature(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    group_col: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame
```

Two modes, selected by `group_col`:

### Across-all-samples mode (`group_col is None`)

Tests each feature's distribution across every column of `feature_table`.
`metadata` is ignored. BH-FDR is applied across features. Returns a
DataFrame indexed by `feature_id`:

| Column | Type | Description |
|--------|------|-------------|
| `n` | int | Non-NaN sample count for the feature |
| `statistic` | float | Shapiro W (NaN if skipped) |
| `p_value` | float | Two-sided Shapiro p (NaN if skipped) |
| `p_adj` | float | BH-FDR adjusted p across features (NaN if skipped) |
| `normal` | bool | `p_adj >= alpha` and not skipped |
| `skipped_reason` | str | `""` or `"n<3"` |

A feature is skipped when it has fewer than 3 non-NaN values.

### Per-group mode (`group_col` given)

Tests each feature's distribution within each group separately. The
groups are the unique non-NaN values of `metadata[group_col]`, sorted by
string form. BH-FDR is applied *within each group* across features.
Returns a DataFrame indexed by `feature_id`:

| Column | Type | Description |
|--------|------|-------------|
| `{g}_n` | int | Non-NaN samples in group `g` for the feature |
| `{g}_statistic` | float | Shapiro W within group `g` |
| `{g}_p_value` | float | Shapiro p within group `g` |
| `{g}_p_adj` | float | BH-FDR adjusted p within group `g`, across features |
| `{g}_normal` | bool | `p_adj >= alpha` and not skipped |
| `all_groups_normal` | bool | True iff every group's `{g}_normal` is True for that feature |
| `n_groups_normal` | int | Count of groups for which `{g}_normal` is True |
| `skipped_reason` | str | Semicolon-joined list of `"{g}:n<3"` entries; `""` when no group was skipped |

Per-group mode is the right tool when groups have different means: a
shift between groups inflates the across-all-samples variance and can
falsely flag a per-group-normal feature as non-normal.

---

## `levene_per_feature`

```python
levene_per_feature(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    center: str = "median",
    alpha: float = 0.05,
) -> pd.DataFrame
```

Per-feature Levene's test for variance homogeneity across the groups
defined by `metadata[group_col]`. `center='median'` selects the
Brown-Forsythe variant, which is more robust to non-normality than the
mean-centered version; `"mean"` and `"trimmed"` are also accepted.
BH-FDR is applied across features. Returns a DataFrame indexed by
`feature_id`:

| Column | Type | Description |
|--------|------|-------------|
| `statistic` | float | Levene statistic (NaN if skipped) |
| `p_value` | float | Levene p (NaN if skipped) |
| `p_adj` | float | BH-FDR adjusted p across features (NaN if skipped) |
| `homoscedastic` | bool | `p_adj >= alpha` and not skipped |
| `skipped_reason` | str | Semicolon-joined list of `"{g}:n<3"` entries; `""` when none skipped |

A feature is skipped when any group has fewer than 3 non-NaN values.

---

## Validation

Both functions validate inputs up front and raise `ValueError` with an
informative message on:

- `group_col` (when used) missing from `metadata.columns`
- `group_col` having fewer than 2 distinct non-NaN levels
- `metadata.index` containing samples absent from `feature_table.columns`
- `shapiro_per_feature(..., group_col=...)` called with `metadata=None`
- `levene_per_feature(..., center=...)` not in `{"median", "mean", "trimmed"}`

---

## Idempotency / reproducibility

Shapiro and Levene are deterministic — no seed is required. Both
functions are pure (no project mutation, no manifest entry).

---

## Out of scope

- No inference. These diagnostics inform the choice of downstream test
  but do not produce p-values for biological differences.
- No automatic dispatch to a parametric or non-parametric inference
  routine based on the diagnostic outcome.
- No CLI subcommand and no manifest entry.
- No plotting helpers.
