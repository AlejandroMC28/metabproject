# `metab stats` Specification

## Overview

Per-feature Welch t-tests with Benjamini-Hochberg FDR adjustment, for one or
more user-declared contrasts. Each contrast names a comparison and defines
its two groups as filters on metadata columns, so factorial designs
(variety × treatment, etc.) are expressed directly without rebuilding a
combined group column.

Tests run on the processed feature table. `log2fc` follows the convention
`log2(B / A)` (positive = up in group B), matching the standard
volcano-plot convention "treatment vs control".

How `log2fc` is computed depends on the transform chosen at `process` time:

- **`log2` / `log10`** (and `--scale none`): the processed table is already
  on a pure log scale, so `log2fc = mean_b - mean_a` directly (multiplied by
  `log2(10)` when the transform was `log10`). No raw-scale fold-change
  matrix is built — the difference of means on the log scale *is* the log2
  fold change. This is what the reference pipeline does and is the
  recommended setup for differential abundance.
- **Any other transform** (`sqrt`, `none`), **or any non-`none` scaling**
  (`pareto`, `auto`): the processed table no longer encodes log fold change
  directly, so `log2fc` is computed from a freshly-imputed copy of the raw
  feature table (no normalization, no transformation, no scaling) as
  `log2(arith_mean_b / arith_mean_a)`. Fold change stays biologically
  interpretable regardless of the processing choices.

Writes one parquet per contrast to `<project>/stats/`. Appends one manifest
entry per `stats` run, summarising every contrast.

---

## Usage

```bash
metab stats my_project/ --contrasts contrasts.yaml \
    [--alpha 0.05] [--min-samples 3] [--force]
```

Prerequisite: `metab process` must have been run on the project.

---

## Contrasts file (`--contrasts`)

YAML. Top-level key `contrasts` is a list. Each entry has `name`, `a`, `b`.
`a` and `b` are filter mappings: keys are metadata column names; values are
either a scalar or a list. Multiple columns in one filter are AND-ed; list
values within a column are OR-ed.

```yaml
contrasts:
  - name: varA_control_vs_fermented
    a: { variety: varA, treatment: control }
    b: { variety: varA, treatment: fermented }

  - name: varA_control_vs_roasted
    a: { variety: varA, treatment: control }
    b: { variety: varA, treatment: roasted }

  - name: control_varA_vs_varB
    a: { variety: varA, treatment: control }
    b: { variety: varB, treatment: control }

  - name: varA_vs_varB_pooled
    a: { variety: varA }
    b: { variety: varB }
```

### Validation (fail-fast before any test runs)

- `name` is a unique, filename-safe slug (`[A-Za-z0-9_.-]+`).
- Every filter column exists in `metadata`.
- Every filter value matches at least one sample in `metadata`.
- `a` and `b` sample sets are disjoint.
- Both sample sets have ≥ `--min-samples`; contrasts that fail this are
  **skipped** (recorded in the manifest with `skipped: true` and a reason)
  rather than erroring the whole run.

---

## Processing Steps

1. **Parse + resolve contrasts.** All YAML validation and filter resolution
   happen up front; if any contrast is malformed the run aborts before any
   file is written.
2. **(Non-log-shortcut path) Build the fold-change matrix.** From the
   latest `process` manifest entry, read `min_prevalence`, `impute`,
   `seed`, `lod_value`, `knn_k`. On `raw_feature_table` (after `0 → NaN`):
   apply the same prevalence filter, then re-run **only** imputation. No
   normalization, no transformation, no scaling. Skipped when the processed
   table is already pure-log (i.e. `transform` is `log2`/`log10` *and*
   `scale` is `none`).
3. **Per contrast: per-feature Welch t-test.**
   `scipy.stats.ttest_ind(..., equal_var=False, nan_policy="omit")` on each
   row of the processed feature table restricted to the contrast's samples.
   Features with fewer than `--min-samples` non-NaN values in either group
   get NaN p-values and are excluded from FDR.
4. **BH-FDR.** `scipy.stats.false_discovery_control(..., method="bh")` over
   the contrast's tested features.
5. **`log2fc`.**
   - `log2` transform + `none` scaling: `mean_b - mean_a` on the processed table.
   - `log10` transform + `none` scaling: `(mean_b - mean_a) * log2(10)`.
   - All other (transform, scale) combinations: `log2(arith_mean_b / arith_mean_a)`
     from the FC matrix. NaN only when either group mean is zero (rare after
     imputation).
6. **`direction`.** `"up"` if `log2fc > 0` and `p_adj < alpha`; `"down"`
   if `log2fc < 0` and `p_adj < alpha`; otherwise `"ns"`.

---

## Output Files

Per contrast in `<project>/stats/<name>.parquet`, indexed by `feature_id`:

| Column | Type | Description |
|--------|------|-------------|
| `mean_a` | float | Mean of group A on the processed scale |
| `mean_b` | float | Mean of group B on the processed scale |
| `n_a` | int | Non-NaN samples used from group A |
| `n_b` | int | Non-NaN samples used from group B |
| `log2fc` | float | `log2(B / A)`. From the processed table for log transforms, otherwise from the raw-scale FC matrix. |
| `t_stat` | float | Welch t statistic |
| `p_value` | float | Two-sided Welch p (NaN when feature not tested) |
| `p_adj` | float | BH-FDR adjusted p (NaN when not tested) |
| `direction` | str | `"up"`, `"down"`, or `"ns"` |
| `significant` | bool | `p_adj < alpha` |

---

## Manifest Entry

### Parameters

```json
{
  "contrasts_file": "<path>",
  "alpha": 0.05,
  "min_samples": 3,
  "test": "welch_ttest",
  "fdr_method": "bh",
  "log2fc_source": "processed_difference"   // or "raw_fc_matrix"
}
```

### Results

Keyed by contrast name:

```json
{
  "varA_control_vs_fermented": {
    "n_a": 3,
    "n_b": 3,
    "n_tested": 1842,
    "n_significant": 217,
    "n_up": 109,
    "n_down": 108
  },
  "skipped_contrast": {
    "skipped": true,
    "reason": "insufficient samples (a=1, b=3, min=3)",
    "n_a": 1,
    "n_b": 3
  }
}
```

---

## Python API

```python
from metabproject.project import MetabProject

p = MetabProject("my_project/")

p.stats(
    contrasts="contrasts.yaml",
    alpha=0.05,
    min_samples=3,
    force=False,
)

p.contrast("varA_control_vs_fermented")  # one DataFrame
p.contrasts                                  # dict of all contrast DataFrames
```

---

## Idempotency

Running `metab stats` raises `FileExistsError` if **any** target parquet
already exists in `stats/`. With `--force`, only the parquets that will be
overwritten are deleted (other files in `stats/` are untouched), and a new
manifest entry is appended (the manifest is append-only).

---

## Errors

| Condition | Exception |
|-----------|-----------|
| Contrasts file not found | `FileNotFoundError` |
| Malformed YAML / missing keys / unsafe name / duplicate names | `ValueError` |
| Filter column not in metadata, or value matches no samples | `ValueError` |
| Group A and B share samples | `ValueError` |
| Target parquet exists and `--force` not passed | `FileExistsError` |
| No `process` entry in the manifest | `FileNotFoundError` |
