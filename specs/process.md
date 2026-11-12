# Spec: `metab process`

Apply preprocessing to the raw feature table. Runs the full pipeline in a fixed order: prevalence filter → imputation → weight correction → normalization → transformation → scaling. Always generates a QC figure.

## Usage

```bash
metab process my_project \
  --min-prevalence 2 \
  --prevalence-mode any_value \
  --min-groups-meeting 1 \
  --impute half-min \
  --seed 42 \
  --weight-normalize \
  --normalize none \
  --transform log2 \
  --scale none
```

## Prerequisites

- Project must have been initialized with `metab init`
- `features/feature_table_raw.parquet` and `features/metadata.parquet` must exist
- If `feature_table.parquet` already exists, error unless `--force` is passed

## Processing Pipeline

Steps are executed in this fixed order. The order is designed so that each step operates on the best possible input: prevalence filtering works on raw detection counts, imputation fills missing values on the raw (uncorrected) scale closest to the instrument detection limit, and normalization sees a complete matrix with no missing values.

### Step 1: Prevalence Filter

Remove features not consistently detected across replicates. Operates on raw areas before any correction, since detection is a binary property of the raw signal.

**Parameters:**

- `--min-prevalence N` (default: 2) — minimum detections required within a single group.
- `--min-groups-meeting K` (default: 1) — number of groups that must each meet `--min-prevalence`. Default 1 reproduces the legacy "passes in any group" semantics.
- `--prevalence-mode {any_value,detected}` (default: `any_value`) — choice of detection criterion (see below).

**Detection criterion modes:**

- **`any_value`** (default, legacy): a sample is considered detected when its peak area is `> 0` and not NaN. Cannot distinguish a true MZmine detection from a gap-filled (`ESTIMATED`) value — features that are gap-filled in enough replicates will pass.
- **`detected`**: uses MZmine's own `feature_state == "DETECTED"` mask (persisted at init as `features/detection_mask.parquet`). Drops features that survive only because of gap-filling. Recommended when reporting a "truly observed" feature count.

**Logic:** Keep a feature if it is detected in at least `--min-prevalence` samples within **at least `--min-groups-meeting`** groups. Groups are defined by the `group` column in metadata.

**Examples:**
- 2 groups × 3 reps, defaults): feature kept if ≥2/3 in control OR ≥2/3 in fermented
- 9 groups × 3 reps, defaults): feature kept if ≥2/3 in any one of the 9 groups
- `--min-groups-meeting 2`: feature must clear `--min-prevalence` in at least two groups
- `--prevalence-mode detected`: any/all of the above, but counting only MZmine `DETECTED` peaks (not gap-filled `ESTIMATED` ones)

**Detection mask provenance.** The mask is built at `metab init` from MZmine Zone 3 `sample_details`:

```python
detection_mask = (sample_details.xs('feature_state', level='metric', axis=1) == 'DETECTED')
```

Columns are renamed from raw `.mzML` filenames to `sample_name` (same rename applied to the areas matrix) and the result is written to `features/detection_mask.parquet`. Projects initialized before this file existed are recovered on demand by re-parsing the original MZmine CSV recorded in the `init` manifest entry; if that file is no longer reachable, `--prevalence-mode detected` errors with a clear message and the user is told to re-init or fall back to `any_value`.

**Implementation:**
```python
def prevalence_filter(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    min_prevalence: int = 2,
    min_groups_meeting: int = 1,
    mode: str = "any_value",
    detection_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Filter features by per-group detection prevalence.

    Keeps a feature if it is detected in at least ``min_prevalence`` samples
    within at least ``min_groups_meeting`` groups, using either the
    area-based proxy (``mode='any_value'``) or MZmine's DETECTED mask
    (``mode='detected'``, requires ``detection_mask``).
    """
```

### Step 2: Imputation

Fill remaining zeros/NaN values for features that passed the filter but are missing in some samples. Operates on raw, uncorrected areas so the imputation threshold reflects the actual instrument detection limit.

**Parameter:** `--impute METHOD` (default: `half-min`)

**Methods:**

- **`half-min`** — Compute the global minimum positive value across the entire filtered dataset. Fill each missing value (zero or NaN) with a random draw from `Uniform(0.1 × min, 0.5 × min)`. Seeded via `--seed` (default: 42) for reproducibility.

  ```python
  positive_vals = df.values[np.isfinite(df.values) & (df.values > 0)]
  global_min = positive_vals.min()
  rng = np.random.default_rng(seed)
  mask = (df == 0) | df.isna()
  n_missing = mask.sum().sum()
  imputed_values = rng.uniform(low=global_min * 0.1, high=global_min * 0.5, size=n_missing)
  # Fill masked positions with imputed_values
  ```

- **`second-min`** — Compute the *second-smallest* unique positive value across the entire filtered dataset. Fill each missing value with a random draw from `Uniform(0, second_min)`, clipped to a strict positive floor of `1e-10` so log transforms remain defined. Seeded via `--seed`. Matches the imputation scheme used in the reference pipeline.

  ```python
  positive_vals = df.values[np.isfinite(df.values) & (df.values > 0)]
  second_min = np.unique(positive_vals)[1]
  rng = np.random.default_rng(seed)
  mask = (df == 0) | df.isna()
  n_missing = mask.sum().sum()
  imputed_values = np.clip(rng.uniform(0, second_min, size=n_missing), 1e-10, None)
  # Fill masked positions with imputed_values
  ```

- **`min`** — Replace all missing values with the global minimum positive value. Deterministic.

- **`lod`** — Replace with a user-specified limit-of-detection value via `--lod-value FLOAT`. Error if `--lod-value` not provided.

- **`knn`** — K-nearest neighbors imputation using `sklearn.impute.KNNImputer`. Parameter: `--knn-k INT` (default: 5). Operates on features as rows, samples as columns.

- **`none`** — Leave zeros/NaN as-is. Print a warning that this will cause errors with log transforms.

### Step 3: Sample Weight Correction (optional)

Divide each sample's areas by its weight from the metadata file.

**Parameter:** `--weight-normalize` flag (default: off)

**Requirements:**
- `weight` column must exist in metadata. Error with clear message if missing.
- All weight values must be > 0. Error if any are zero or negative.

**Implementation:**
```python
for sample_name in df.columns:
    weight = metadata.loc[sample_name, 'weight']
    df[sample_name] = df[sample_name] / weight
```

### Step 4: Sample Normalization (optional)

Correct for systematic differences in total signal across samples (e.g., spray instability, ionization drift). Operates on the complete, weight-corrected matrix.

**Parameter:** `--normalize METHOD` (default: `none`)

**Methods:**

- **`none`** — Skip this step.

- **`tic`** — Total Ion Current normalization. For each sample: divide all feature areas by the sample's total ion current (sum of all features), then multiply by the median TIC across all samples to keep values in a reasonable range.

  ```python
  tic_per_sample = df.sum(axis=0)
  median_tic = tic_per_sample.median()
  df_normalized = df.div(tic_per_sample, axis=1) * median_tic
  ```

- **`pqn`** — Probabilistic Quotient Normalization.
  1. Compute a reference sample: median value across all samples for each feature.
  2. For each sample: divide each feature by the corresponding reference value to get ratios.
  3. Take the median of those ratios as the normalization factor for that sample.
  4. Divide all features in that sample by the normalization factor.

  ```python
  reference = df.median(axis=1)
  quotients = df.div(reference, axis=0)
  norm_factors = quotients.median(axis=0)
  df_normalized = df.div(norm_factors, axis=1)
  ```

- **`median`** — Median normalization. Divide each sample by its median feature area, then multiply by the global median of per-sample medians.

  ```python
  sample_medians = df.median(axis=0)
  global_median = sample_medians.median()
  df_normalized = df.div(sample_medians, axis=1) * global_median
  ```

### Step 5: Transformation

Variance-stabilizing transformation. Pulls right-skewed area distributions toward symmetry so downstream statistics (Welch t-tests, PCA, distance metrics) behave well.

**Parameter:** `--transform METHOD` (default: `log2`)

**Methods:**

- **`log2`** — `log2(x)`. Requires all values > 0; error if any zeros or NaN remain (which means imputation was skipped or failed).

- **`log10`** — `log10(x)`. Same requirement as log2.

- **`sqrt`** — `sqrt(x)`. Requires all values ≥ 0.

- **`none`** — No transformation.

### Step 6: Scaling

Optional centering and standardization applied **after** transformation. Centers each feature across samples and divides by a measure of spread so that features with different magnitudes contribute on a comparable scale.

**Parameter:** `--scale METHOD` (default: `none`)

**Methods:**

- **`none`** — No scaling. Use this when downstream analysis depends on the original (or log-transformed) magnitude — univariate testing, fold-change reporting, enrichment.

- **`pareto`** — Pareto scaling. For each feature: subtract the mean across samples, then divide by the square root of the standard deviation. Less aggressive than autoscaling; preserves some magnitude information.

  ```python
  means = df.mean(axis=1)
  stds = df.std(axis=1)
  df_pareto = df.sub(means, axis=0).div(np.sqrt(stds), axis=0)
  ```

- **`auto`** — Autoscaling (z-score). For each feature: subtract mean and divide by standard deviation.

  ```python
  df_auto = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
  ```

**When scaling helps:** PCA, distance-based clustering, heatmaps — anywhere a few high-intensity features would otherwise dominate the variance.

**When scaling hurts:** univariate testing (fold change loses biological meaning), non-Euclidean PERMANOVA, and class-level enrichment tests like the CANOPUS Mann-Whitney enrichment in `metab enrich` — these expect the feature scale to carry signal.

Reference: van den Berg et al. 2006, *BMC Genomics* 7:142 — standard treatment of the transformation vs scaling distinction in metabolomics.

## QC Figure

Always generated. Saved to `qc/processing_qc.png`.

### Layout

Two rows. One column per processing stage that was actually applied (3–6 columns depending on which optional steps are active).

**Top row: Overall intensity distribution.**
Density plot (KDE) of all values pooled across all features and all samples. One panel per stage. Panel titles: "Raw", "Filtered", "Imputed", "Weight-corrected", "Normalized", "Transformed" (only those that apply).

Shows how the distribution shape changes through the pipeline: the initial right skew, imputed values appearing at the low end, the effect of corrections, and the log transform pulling it toward symmetry.

**Bottom row: Per-sample boxplots.**
One boxplot per sample, colored by group. Same columns/stages as the top row.

Shows whether samples within a group are consistent, whether normalization brings outlier samples in line, and whether groups separate or overlap.

**Panels to skip:**
- If `--weight-normalize` is not set: skip the "Weight-corrected" panel
- If `--normalize none`: skip the "Normalized" panel
- The "Raw", "Filtered", "Imputed", and "Transformed" panels are always shown (unless `--transform none`, in which case skip "Transformed")
- If `--scale none`: skip the "Scaled" panel; otherwise show it as the rightmost stage

### Figure Style

- Use matplotlib/seaborn
- Clean, publication-adjacent style (no grid clutter)
- Group colors should be distinct and consistent across all panels
- Figure size should adapt to the number of panels (wider with more panels)
- DPI: 150 for the saved PNG

## Idempotency

- If `feature_table.parquet` already exists: error with message "Project already processed. Use --force to reprocess."
- With `--force`: delete existing `feature_table.parquet` and `qc/processing_qc.png`, reprocess from `feature_table_raw.parquet`
- Each run appends a new entry to the manifest, so reprocessing creates a second `process` entry (the latest one reflects the current state)

## Output Files

```
my_project/
├── features/
│   ├── feature_table_raw.parquet  # unchanged from init
│   └── feature_table.parquet      # processed table, ready for downstream analysis
├── qc/
│   └── processing_qc.png         # before/after QC figure
└── metabproject.json              # updated with process entry
```

## Manifest Entry

```json
{
  "command": "process",
  "timestamp": "2026-05-28T14:30:00Z",
  "parameters": {
    "min_prevalence": 2,
    "prevalence_scope": "per_group",
    "impute": "half-min",
    "seed": 42,
    "weight_normalize": true,
    "normalize": "none",
    "transform": "log2",
    "scale": "none"
  },
  "results": {
    "features_before_filter": 2341,
    "features_after_filter": 1892,
    "features_removed": 449,
    "n_features_before_prevalence": 2341,
    "n_features_after_prevalence": 1892,
    "prevalence_mode_used": "any_value",
    "min_prevalence_used": 2,
    "min_groups_meeting_used": 1,
    "values_imputed": 234,
    "values_total": 11352
  }
}
```

## CLI Parameters Summary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | argument | required | Path to project directory |
| `--min-prevalence` | int | 2 | Min detections required within a single group |
| `--prevalence-mode` | choice | `any_value` | Detection criterion: `any_value` (area > 0 & notna) or `detected` (MZmine DETECTED mask) |
| `--min-groups-meeting` | int | 1 | Number of groups that must each meet `--min-prevalence` |
| `--impute` | choice | `half-min` | Imputation method: `half-min`, `second-min`, `min`, `lod`, `knn`, `none` |
| `--seed` | int | 42 | Random seed for `half-min` / `second-min` imputation |
| `--lod-value` | float | — | LOD value (required if `--impute lod`) |
| `--knn-k` | int | 5 | Neighbors for KNN imputation |
| `--weight-normalize` | flag | off | Enable sample weight correction |
| `--normalize` | choice | `none` | Normalization method: `none`, `tic`, `pqn`, `median` |
| `--transform` | choice | `log2` | Variance-stabilizing transform: `log2`, `log10`, `sqrt`, `none` |
| `--scale` | choice | `none` | Scaling applied after transform: `none`, `pareto`, `auto` |
| `--force` | flag | off | Overwrite existing processed data |

## Error Handling

- Project not initialized (no `metabproject.json`) → error: "Not a metabproject directory"
- Already processed without `--force` → error: "Project already processed. Use --force to reprocess."
- `--weight-normalize` but no `weight` column in metadata → error: "Metadata missing 'weight' column required for weight normalization"
- `--impute lod` without `--lod-value` → error: "LOD imputation requires --lod-value"
- Zeros remaining after imputation + log transform requested → error: "Cannot apply log2 transform: N zero values remain. Impute or filter zero/missing values before applying this transform."
- No features pass prevalence filter → error: "No features passed prevalence filter (mode=…, min_prevalence=N, min_groups_meeting=K). Check your data or lower the threshold."
- `--prevalence-mode detected` but the project has no persisted detection mask AND the original MZmine CSV is no longer reachable → error directing the user to re-init or use `--prevalence-mode any_value`
- `--prevalence-mode detected` but the MZmine export has no `feature_state` column → error: "MZmine feature table has no 'feature_state' column; cannot build a detection mask. Use --prevalence-mode any_value."

## Python API

```python
from metabproject import MetabProject

project = MetabProject("my_project")
project.process(
    min_prevalence=2,
    prevalence_mode="any_value",   # or "detected" for MZmine DETECTED-only counts
    min_groups_meeting=1,
    impute="half-min",
    seed=42,
    weight_normalize=True,
    normalize="none",
    transform="log2",
    scale="none",
)

# Access processed data
project.feature_table       # processed DataFrame
project.raw_feature_table   # original DataFrame (unchanged)
```

## Testing

- Test each processing step independently with known inputs/outputs:
  - Prevalence filter: 2-group and 9-group scenarios, edge cases (feature detected in exactly N, N-1 samples)
  - Imputation: verify seeded reproducibility, verify value range for half-min, verify no zeros remain after imputation
  - Weight correction: verify division by known weights
  - Normalization: verify TIC (row sums equalized), PQN (quotient-based), median
  - Transformation: verify log2 values against manual calculation
- Test pipeline order: intermediate values at each stage match expected sequence
- Test QC figure generation (file created, correct number of panels)
- Test idempotency: second run errors without `--force`, succeeds with `--force`
- Test error cases: missing weight column, zeros with log transform, no features passing filter
- Integration test: init → process → verify `feature_table.parquet` shape and manifest
