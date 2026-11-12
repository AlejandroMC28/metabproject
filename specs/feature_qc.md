# `metab feature-qc` Specification

## Overview

Runs feature-table QC checks against the **raw, un-imputed, un-transformed** MZmine output (re-parsed from the CSV referenced by the `init` op) so QC sees the same data MZmine emitted. Writes artifacts to `<project>/feature_qc/` and appends one manifest entry per run. A paired `metab viz feature-qc` renders diagnostic figures.

Two checks are implemented; dispatch is by `--check`:

- **`alignment`** — duplicate-feature detection from MZmine alignment / deconvolution failures.
- **`overview`** — table-level health profile (shape, detection, alignment scores, chromatographic quality, MS2 coverage, m/z×RT density, annotation coverage, charge state).
- *Reserved:* `blanks`, `cv`.

Both checks share the load step (re-parse via `parse_feature_table`) and the manifest-logging conventions; they differ in what they compute and which files they write.

---

## Algorithm (`--check alignment`)

1. Re-parse the raw MZmine CSV via `parse_feature_table` into Zone 1 (`ms1_features`), Zone 2 (`areas`), Zone 3 (`sample_details`).
2. Sort features by `mz`. Enumerate candidate pairs via a single sliding window: for feature *i*, include each feature *j > i* while `(mz_j - mz_i) / mz_i * 1e6 <= mz_tol_ppm`, then keep only pairs with `|rt_j - rt_i| <= rt_tol`.
3. Build per-feature detection masks once (Zone 3 `feature_state == "DETECTED"`) and the per-feature area / FWHM matrices, aligned to a common sample order.
4. For each candidate pair compute:
   - `delta_rt`, `delta_ppm`
   - `n_both_detected`, `n_only_i`, `n_only_j`
   - `jaccard_detected = n_both / (n_both + n_only_i + n_only_j)`
   - `pearson_area` on samples where both are detected; NaN if shared count < `MIN_SHARED_FOR_PEARSON`
   - `median_fwhm` — median of Zone 3 `fwhm` across each feature's own DETECTED samples (per-feature peak width, not just the shared subset, so disjoint-detection alignment failures still get a width). NaN when `--no-use-fwhm` or no FWHM available.
   - `drt_over_fwhm = |delta_rt| / median_fwhm` when defined.
5. Classify each pair against the module-level constants in `metabproject/qc/duplicates.py`.
6. Build the per-feature flag table by exploding pairs by participant.

### Classification rules

| Class | Rule |
|---|---|
| `alignment_like` | `jaccard_detected < ALIGNMENT_JACCARD_MAX (0.3)` **and** `drt_over_fwhm > ALIGNMENT_DRT_OVER_FWHM_MIN (0.5)` — or `|delta_rt| > 0` when `use_fwhm=False`. |
| `split_like` | `jaccard_detected > SPLIT_JACCARD_MIN (0.7)` **and** `pearson_area > max(min_correlation, SPLIT_PEARSON_MIN=0.7)` **and** (`drt_over_fwhm < SPLIT_DRT_OVER_FWHM_MAX (1.0)` or NaN). |
| `isomer_like` | `jaccard_detected > ISOMER_JACCARD_MIN (0.7)` **and** `drt_over_fwhm > ISOMER_DRT_OVER_FWHM_MIN (1.5)`. |
| `unclassified` | Anything else. |

`min_correlation` is a runtime knob that can only *tighten* the split-like Pearson cutoff above `SPLIT_PEARSON_MIN`. All other thresholds are module-level constants — change them in `qc/duplicates.py`, not at the call site.

### Per-group analysis (optional)

When `--group-by <metadata_col>` is supplied, the same scoring is re-run independently inside each level of the chosen metadata column. The global pair/flag outputs are unchanged; a new per-group TSV and a "Per-group misalignment" section in `__summary.txt` rank groups by misalignment rate so the user can answer *which group is MZmine aligning worst*.

For each level (after restricting Zone 2 areas and Zone 3 `sample_details` columns to that level's `sample_file`s):

- `n_features_detected` — features DETECTED in ≥1 sample of the group.
- Per-class pair counts: `n_alignment_like`, `n_split_like`, `n_isomer_like`, `n_unclassified`.
- `n_features_flagged` — unique features appearing in any suspect pair.
- `alignment_like_rate = n_alignment_like / n_features_detected` — **headline metric** (RT-misalignment failures).
- `any_suspect_rate = n_pairs / n_features_detected`.
- `flagged_feature_rate = n_features_flagged / n_features_detected`.

Rates are normalized by features-detected-in-group, not the global feature count, so groups with sparse detection are not double-penalized.

`--min-samples-per-group` (default 3, matches `MIN_SHARED_FOR_PEARSON`) gates the analysis: groups with fewer usable samples are listed under `skipped_groups` with their sample count and not scored. Without this, two-sample groups make Jaccard binary and Pearson undefined.

### Recommended-action rule (per feature)

| Classifications across all pairs the feature appears in | `recommended_action` |
|---|---|
| `{split_like}` | `drop_lower_intensity` |
| `{isomer_like}` | `keep_both` |
| anything else (mixed, contains `alignment_like` or `unclassified`) | `review` |

---

## Output

### `feature_qc/<name>__pairs.tsv`

One row per suspect pair.

| Column | Type | Description |
|---|---|---|
| `feature_id_i`, `feature_id_j` | int | Pair members (i < j by m/z) |
| `mz_i`, `mz_j`, `rt_i`, `rt_j` | float | Consensus m/z and RT |
| `delta_rt` | float | `rt_j - rt_i` (minutes) |
| `delta_ppm` | float | m/z separation in ppm |
| `n_both_detected`, `n_only_i`, `n_only_j` | int | Detection overlap counts |
| `jaccard_detected` | float | Detection-set Jaccard |
| `pearson_area` | float / NaN | Area correlation on shared-detected samples |
| `median_fwhm` | float / NaN | Per-feature FWHM median (NaN when `--no-use-fwhm`) |
| `drt_over_fwhm` | float / NaN | `|delta_rt| / median_fwhm` |
| `classification` | str | One of the four classes |
| `preferred_annotation_i`, `preferred_annotation_j` | str / NA | From Zone 1 `preferred_annotation:compound_name` when present |

### `feature_qc/<name>__feature_flags.tsv`

One row per feature appearing in any suspect pair.

| Column | Type | Description |
|---|---|---|
| `feature_id` | int | Feature ID |
| `n_pairs` | int | Number of suspect pairs the feature participates in |
| `classifications` | str | Sorted comma-separated unique classes across its pairs |
| `recommended_action` | str | See table above |

### `feature_qc/<name>__summary.txt`

Plain-text summary with per-class pair counts, the top 10 most-flagged features, and parameter-adjustment hints derived from the distributions of `delta_rt` (split_like) and `drt_over_fwhm` (alignment_like). Hints are only emitted when ≥1 pair of that class exists.

When `--group-by` is set, a final "Per-group misalignment" section appends a worst-aligned-group headline, a ranked table (one row per level, sorted descending by `alignment_like_rate`), and a `skipped_groups` listing.

### `feature_qc/<name>__by_group.tsv` (only when `--group-by` is set)

One row per scored level, sorted descending by `alignment_like_rate`.

| Column | Type | Description |
|---|---|---|
| `group_value` | str | Level of the `--group-by` metadata column. |
| `n_samples` | int | Samples in the level (after intersection with Zone 2/3 columns). |
| `n_features_detected` | int | Features DETECTED in ≥1 sample of the level. |
| `n_pairs` | int | Suspect pairs found inside the level. |
| `n_alignment_like`, `n_split_like`, `n_isomer_like`, `n_unclassified` | int | Per-class pair counts. |
| `n_features_flagged` | int | Unique features in any suspect pair. |
| `alignment_like_rate` | float | `n_alignment_like / n_features_detected`. |
| `any_suspect_rate` | float | `n_pairs / n_features_detected`. |
| `flagged_feature_rate` | float | `n_features_flagged / n_features_detected`. |

### Manifest entry

```json
{
  "command": "feature_qc",
  "timestamp": "...",
  "parameters": {
    "check": "alignment",
    "name": "baseline",
    "mz_tol_ppm": 10.0,
    "rt_tol": 0.1,
    "min_correlation": 0.7,
    "use_fwhm": true,
    "group_by": "variety",
    "min_samples_per_group": 3,
    "mzmine_raw": "/abs/path/to/feature_table.csv",
    "force": false
  },
  "results": {
    "check": "alignment",
    "n_features_input": 11342,
    "n_suspect_pairs": 218,
    "n_alignment_like": 31,
    "n_split_like": 96,
    "n_isomer_like": 47,
    "n_unclassified": 44,
    "n_features_flagged": 401,
    "output_paths": {
      "pairs":   "feature_qc/baseline__pairs.tsv",
      "flags":   "feature_qc/baseline__feature_flags.tsv",
      "summary": "feature_qc/baseline__summary.txt",
      "by_group":"feature_qc/baseline__by_group.tsv"
    },
    "by_group": {
      "column": "variety",
      "min_samples_per_group": 3,
      "worst_group_alignment_like": {"group": "cream", "alignment_like_rate": 0.018},
      "rows": [ { "group_value": "cream", "n_samples": 9, "...": "..." } ],
      "skipped_groups": []
    }
  }
}
```

---

## Algorithm (`--check overview`)

A table-level health profile. No per-pair or per-feature scoring; instead each metric group describes the **whole table** so two MZmine exports can be compared side-by-side (e.g. original vs dedup, or before/after a parameter change).

1. Re-parse the raw MZmine CSV (same loader as the alignment check).
2. Build the boolean `features × samples` detection mask from Zone 3 `feature_state == "DETECTED"`.
3. Compute the metric groups below in `metabproject.qc.overview.compute_overview(...)`. The function is library-only and pure — given the three DataFrames it returns `(metrics_dict, issues_list)` with no I/O.
4. The `MetabProject` wrapper resolves user-facing sample names from `metadata.parquet` (`sample_file → sample_name`) and passes them as `sample_name_map`. The per-sample profile is then labeled with `sample_name`, not raw datafile filenames.

### Metric groups

All percentages are stored as floats in `[0, 1]`.

| Group | Key fields |
|---|---|
| `table_shape` | `n_features`, `n_samples`, `density` (fraction of DETECTED cells), `features_per_sample` (median / mean / p25 / p75 / min / max of DETECTED count per sample). |
| `per_feature_detection` | `detection_histogram` — bins `1`, `2_3`, `4_9`, `10_18`, `19_26`, `27_or_all` (top bin relabeled to `{lo}_to_{n_samples}_or_all` when `n_samples < 27`); each entry is `{count, fraction}`. `n_detected_stats` — median / mean / p25 / p75 of per-feature DETECTED count. |
| `per_sample_detection` | `per_sample_detected` — list of `{sample, n_detected, fraction_of_features}` ascending by `n_detected`. `flagged_samples` — samples whose detection count is more than `|--low-detection-sample-zscore|` SDs below the across-sample mean. |
| `alignment_quality` | `stats` — `{median, p25, p75, p95}` for each of `alignment_scores:rate`, `alignment_scores:weighted_distance_score`, `alignment_scores:rt_absolute_error`, `alignment_scores:mz_diff_ppm`, `alignment_scores:aligned_features_n` (silently omitted when the column is absent). `n_low_alignment_rate`, `fraction_low_alignment_rate`, `low_alignment_rate_threshold`. |
| `chromatographic_quality` | `fwhm_global` (median / p25 / p75 across all DETECTED cells), `fwhm_per_sample` — `{sample, median_fwhm}` list, `tailing_factor` and `asymmetry_factor` (median / p25 / p75 / p95), `fraction_poor_tailing` and `fraction_poor_asymmetry` (DETECTED cells exceeding `--poor-peak-shape-threshold`), and the threshold itself. |
| `ms2_coverage` | `fraction_with_ms2` — features with `fragment_scans > 0`. `ms2_by_intensity_decile` — `{decile: fraction_with_ms2}` over deciles 1..10 of mean detected area (high decile = high intensity). Surfaces MS2 selection bias. |
| `mass_rt_coverage` | `mz_range`, `rt_range`, and `mz_rt_density` — non-empty cells of a 50×50 2-D histogram (`mz_bin`, `rt_bin`, `n_features`). |
| `annotation_coverage` | Present **only when** any annotation column is populated. `fraction_preferred_annotated`, `fraction_lipid_annotated`, and `by_method` (value counts of `preferred_annotation:annotation_method`). |
| `charge_state` | `counts` and `fractions` for `1`, `2`, `3+`, `missing`. |

### Issues list

Computed alongside the metrics. Each issue is `{type, severity, details}`:

| Type | Severity rule |
|---|---|
| `low_alignment_rate` | `warning` if `fraction_low_alignment_rate > 0.25`, else `info`. Always emitted when the alignment-rate column is present. |
| `low_detection_sample` | One `warning` per sample in `flagged_samples`. |
| `sparse_table` | `warning` when `density < 0.30`. |
| `poor_peak_shape` | `warning` when `fraction_poor_tailing > 0.10` or `fraction_poor_asymmetry > 0.10`. |
| `low_ms2_coverage` | `info` when `fraction_with_ms2 < 0.30`. |
| `high_singleton_fraction` | `warning` when the fraction of features detected in exactly one sample exceeds `0.15`. |

Thresholds in the rules above are module-level constants in `metabproject/qc/overview.py`; the three CLI-tunable knobs are the per-feature/per-sample inputs (`--low-alignment-rate-threshold`, `--low-detection-sample-zscore`, `--poor-peak-shape-threshold`).

### Overview outputs

#### `feature_qc/<name>__overview.json`

The full structured dict, pretty-printed (indent 2):

```json
{
  "metrics": { "table_shape": {...}, "per_feature_detection": {...}, ... },
  "issues":  [ { "type": "...", "severity": "warning|info", "details": {...} } ]
}
```

NaN values are serialised as `null` so the JSON is portable.

#### `feature_qc/<name>__overview.txt`

Human-readable summary: one heading per metric group with the headline numbers, followed by an `Issues` section listing warnings then info-level issues.

#### Manifest entry

The state holds the headline numbers and the output paths; the full metrics dict is not duplicated (the JSON file is the source of truth).

```json
{
  "command": "feature_qc",
  "timestamp": "...",
  "parameters": {
    "check": "overview",
    "name": "baseline",
    "low_alignment_rate_threshold": 0.10,
    "low_detection_sample_zscore": -2.0,
    "poor_peak_shape_threshold": 2.0,
    "mzmine_raw": "/abs/path/to/feature_table.csv",
    "force": false
  },
  "results": {
    "check": "overview",
    "n_features": 11237,
    "n_samples": 27,
    "density": 0.338,
    "n_issues_total": 3,
    "n_issues_warning": 3,
    "output_paths": {
      "overview_json": "feature_qc/baseline__overview.json",
      "overview_txt":  "feature_qc/baseline__overview.txt"
    }
  }
}
```

---

## CLI

```
metab feature-qc <project>
    --check {alignment,overview}             # default: alignment
    --name <run_name>                        # output stem under feature_qc/

    # --- alignment-only --------------------------------------------------
    --mz-tol-ppm 10.0
    --rt-tol 0.1                              # minutes
    --min-correlation 0.7
    --use-fwhm / --no-use-fwhm                # default --use-fwhm
    --group-by <metadata_col>                 # optional; per-group misalignment rates
    --min-samples-per-group 3                 # min sample count to score a level

    # --- overview-only ---------------------------------------------------
    --low-alignment-rate-threshold 0.10
    --low-detection-sample-zscore -2.0
    --poor-peak-shape-threshold 2.0

    [--mzmine-raw <path>]                     # overrides the path stored in the init op
    [--force]                                 # overwrite existing run with the same name
```

`--mzmine-raw` is only needed when the original CSV recorded by `metab init` has moved. Without it the path is read from the `init` manifest entry; a missing file raises `FileNotFoundError`.

Re-running the same `--name` without `--force` raises `FileExistsError`. The existence check uses `__pairs.tsv` for alignment runs and `__overview.json` for overview runs; alignment and overview runs can therefore safely share a `--name`. (The paired viz, however, dispatches on which file exists *first* — see below — so distinct names are recommended when both checks are kept around.)

### Paired viz

```
metab viz feature-qc <project>
    --name <run_name>
    [--basename <stem>]                       # defaults to --name
```

Dispatches by which run artefact is present under `<project>/feature_qc/`:

- **Alignment** (`<name>__pairs.tsv`): writes two figures under `<project>/viz/` (PNG + SVG each):
  - `<stem>__feature_qc_scatter` — ΔRT (x) vs Pearson(area) (y), colored by classification, point size encodes `n_both_detected`.
  - `<stem>__feature_qc_fwhm_hist` — histogram of `drt_over_fwhm` faceted by classification.
- **Overview** (`<name>__overview.json`): writes one figure (PNG + SVG) with a 2×3 panel grid:
  1. Detection histogram (bars by bin label, y = n_features)
  2. Per-sample detection bar (samples ascending by `n_detected`, flagged samples in red)
  3. Alignment-rate quartiles + the low-alignment threshold marker
  4. Per-sample median FWHM distribution + global-median line
  5. MS2 coverage by intensity decile (bars)
  6. m/z × RT density (scatter of non-empty grid cells, colored by `n_features`)

When both a `__pairs.tsv` and a `__overview.json` exist for the same `--name`, the alignment branch wins. Logs a `viz_feature_qc` manifest entry with the output paths and the relevant run summary (`n_pairs` for alignment; `n_features` and `n_samples` for overview).

---

## Python API

```python
# alignment
project.feature_qc(
    check="alignment",
    name="baseline_alignment",
    mz_tol_ppm=10.0,
    rt_tol=0.1,
    min_correlation=0.7,
    use_fwhm=True,
    mzmine_raw=None,        # default: pull from init manifest entry
    force=False,
)

# overview
project.feature_qc(
    check="overview",
    name="baseline_overview",
    low_alignment_rate_threshold=0.10,
    low_detection_sample_zscore=-2.0,
    poor_peak_shape_threshold=2.0,
    mzmine_raw=None,
    force=False,
)

project.viz_feature_qc(name="baseline_alignment")   # → scatter + ΔRT/FWHM hist
project.viz_feature_qc(name="baseline_overview")    # → 2×3 panel grid

# Library entry points (no I/O, no manifest):
from metabproject.qc import detect_duplicates, compute_overview

pairs_df, flags_df, summary_dict = detect_duplicates(
    ms1_features, areas, sample_details,
    mz_tol_ppm=10.0, rt_tol=0.1,
    min_correlation=0.7, use_fwhm=True,
)

metrics, issues = compute_overview(
    ms1_features, areas, sample_details,
    low_alignment_rate_threshold=0.10,
    low_detection_sample_zscore=-2.0,
    poor_peak_shape_threshold=2.0,
    sample_name_map=None,   # optional {datafile_filename: sample_name}
)
```

`detect_duplicates` and `compute_overview` take plain DataFrames in and return plain dicts / DataFrames out, with no project, manifest, or filesystem dependency. They are the right entry points for tolerance sweeps, side-by-side comparisons across MZmine exports, or unit tests.

---

## Implementation

```
src/metabproject/qc/
    __init__.py
    duplicates.py            # detect_duplicates + threshold constants
    overview.py              # compute_overview + issue thresholds
src/metabproject/viz/
    feature_qc.py            # feature_qc_figures(pairs_df) -> (scatter, hist)
                             # feature_qc_overview_figure(metrics, issues)
tests/test_qc/
    test_duplicates.py       # synthetic 3-sample × 6-feature dataset
    test_overview.py         # synthetic 3-sample × 5-feature dataset
```

CLI commands live in `src/metabproject/cli.py` (`feature_qc_cmd`, `viz_feature_qc_cmd`). Both are thin wrappers that delegate to `MetabProject.feature_qc(...)` / `viz_feature_qc(...)`. `MetabProject.feature_qc` shares the load step (re-parse via `parse_feature_table`) then dispatches to `_feature_qc_alignment(...)` or `_feature_qc_overview(...)`. Each branch owns its own output files, manifest-entry shape, and existence check.

The `feature_qc/` output directory is created lazily on first run, not by `metab init` — existing projects gain the command without re-initialization.

### Why re-parse the raw CSV

`feature_table_raw.parquet` only carries Zone 2 areas (renamed to `sample_name`). Zone 3 metrics (`feature_state`, `fwhm`) and Zone 1 `mz`/`rt`/annotation columns are needed in full, so `feature_qc` re-runs `parse_feature_table` on the original CSV. This is also why `--mzmine-raw` exists as an override.
