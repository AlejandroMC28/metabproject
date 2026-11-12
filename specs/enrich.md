# `metab enrich` Specification

## Overview

Per-CANOPUS-class Mann-Whitney U test on the per-feature `log2fc` distribution from a contrast. For each class at a given level, tests whether the foreground (features assigned to that class with probability ≥ the threshold set at `annotate` time) has a significantly different `log2fc` distribution from the background, using BH-FDR correction within each (contrast × level) block.

Produces one parquet per contrast under `<project>/enrichment/`. Appends one manifest entry per `enrich` run.

---

## Usage

```bash
metab enrich my_project/ \
    [--contrast control_vs_fermented] \
    [--level npc_class] \
    [--background all|level|any] \
    [--min-class-n 3] [--alpha 0.05] [--force]
```

Prerequisites: `metab process`, `metab annotate`, and `metab stats` must all have been run on the project.

`--contrast` and `--level` are repeatable. Defaults to all contrasts × all four levels.

---

## Algorithm

For each `(contrast, level, class_name)`:

1. **Foreground** = features with `canopus_<level> == class_name` AND `canopus_<level>_passed == True` (probability flag set at `annotate` time) AND finite `log2fc` in the contrast.
2. **Background** (controlled by `--background`):
   - `all` *(default)* — all features with finite `log2fc`, excluding the foreground. Reproduces the reference pipeline.
   - `level` — features with `canopus_<level>_passed == True`, excluding the foreground. GSEA-style: "different from other identified chemistry at this level."
   - `any` — features with at least one `canopus_*_passed == True`, excluding the foreground.
3. Skip class if `len(foreground) < min_class_n`.
4. **Test**: `scipy.stats.mannwhitneyu(foreground, background, alternative="two-sided", method="auto").pvalue`.
5. **Effect size**: `mean(foreground log2fc)` — positive = foreground enriched in group B.
6. **BH-FDR**: `scipy.stats.false_discovery_control(pvalues, method="bh")` applied within each (contrast × level) block before concatenation.

---

## Inputs

- `<project>/stats/<contrast>.parquet` — per-feature `log2fc` column.
- `<project>/annotations/canopus.parquet` — CANOPUS class predictions created by `metab annotate`, with per-level `canopus_<level>`, `canopus_<level>_probability`, and `canopus_<level>_passed` columns. The `*_passed` flags are set at annotate time based on the `--canopus-min-probability` threshold (default 0.7) and determine which features enter the foreground for each class.

---

## Output Files

One parquet per contrast: `<project>/enrichment/<contrast>.parquet`.

| Column | Type | Description |
|--------|------|-------------|
| `level` | str | CANOPUS level (`npc_pathway`, `npc_superclass`, `npc_class`, `classyfire_class`) |
| `class_name` | str | Class label |
| `n` | int | Foreground feature count |
| `mean_log2fc` | float | Mean log2fc of foreground features |
| `p_value` | float | Raw Mann-Whitney U p-value |
| `p_adj` | float | BH-FDR adjusted p-value (within contrast × level) |
| `significant` | bool | `p_adj < alpha` |

Sorted by `p_adj` ascending within each level block.

---

## Manifest Entry

### Parameters

```json
{
  "contrasts": ["control_vs_fermented"],
  "levels": ["npc_pathway", "npc_superclass", "npc_class", "classyfire_class"],
  "background": "all",
  "min_class_n": 3,
  "alpha": 0.05,
  "test": "mannwhitneyu",
  "fdr_method": "bh"
}
```

### Results

Keyed by contrast name:

```json
{
  "control_vs_fermented": {
    "n_total_tested": 87,
    "n_significant": 12,
    "per_level": {
      "npc_pathway":    {"n_tested": 7,  "n_significant": 2},
      "npc_superclass": {"n_tested": 18, "n_significant": 3},
      "npc_class":      {"n_tested": 54, "n_significant": 7},
      "classyfire_class": {"n_tested": 8, "n_significant": 0}
    }
  }
}
```

If a level was skipped (missing `*_passed` column), it appears under `"skipped_levels": [...]` in the contrast entry.

---

## Python API

```python
from metabproject.project import MetabProject

p = MetabProject("my_project/")

p.enrich(
    contrasts=["control_vs_fermented"],   # default: all
    levels=["npc_class"],                  # default: all four
    background="all",
    min_class_n=3,
    alpha=0.05,
    force=False,
)

p.enrichment("control_vs_fermented")   # one DataFrame
p.enrichment_results                    # dict of all enrichment DataFrames
```

---

## Idempotency

Running `metab enrich` raises `FileExistsError` if **any** target parquet already exists in `enrichment/`. With `--force`, only colliding parquets are deleted (other files are untouched), and a new manifest entry is appended.

---

## Errors

| Condition | Exception |
|-----------|-----------|
| `canopus.parquet` not found | `FileNotFoundError` — "Run 'metab annotate' first." |
| No stats parquets found | `FileNotFoundError` — "Run 'metab stats' first." |
| Named contrast not in `stats/` | `FileNotFoundError` listing missing names |
| `*_passed` column absent for a level | Warning printed; level skipped; recorded in manifest |
| Target parquet exists and `--force` not set | `FileExistsError` |
| Contrast has no finite `log2fc` | `ValueError` |
