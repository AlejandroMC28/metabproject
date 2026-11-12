# Spec: `metab init`

Create a new project from MZmine output.

## Usage

```bash
metab init my_project \
  --feature-table mzmine_feature_table.csv \
  --metadata samples.csv
```

## Inputs

### `--feature-table` (required)

The MZmine full feature table CSV. This single file contains all feature metadata and per-sample quantification. The parser must handle its three-zone column structure:

**Zone 1: Consensus feature metadata** (columns before any `datafile:*` column):

- `id` — canonical feature identifier (becomes `feature_id` throughout the package)
- `mz`, `mz_range:min`, `mz_range:max` — precursor m/z
- `rt`, `rt_range:min`, `rt_range:max` — retention time (minutes)
- `area`, `height` — consensus (aligned) area and height
- `intensity_range:min`, `intensity_range:max`
- `charge`, `fragment_scans`
- `alignment_scores:*` — `rate`, `aligned_features_n`, `align_extra_features`, `weighted_distance_score`, `mz_diff_ppm`, `mz_diff`, `rt_absolute_error`, `ion_mobility_absolute_error`
- `molecular_networking:*` — `net_cluster_id`, `net_community_id`, `edges`, `net_cluster_size`, `net_community_size`
- `ion_identities:*` — `iin_id`, `ion_identities`, `list_size`, `neutral_mass`, `partner_row_ids`, `iin_relationship`, `consensus_formulas`, `simple_formulas`
- `preferred_annotation:*` — `compound_name`, `mol_formula`, `adduct`, `score`, `precursor_mz`, `annotation_method`
- `lipid_annotations:*` — `lipid_annotations`, `ion_adduct`, `mol_formula`, `mz_diff_ppm`, `explained_intensity_percent`
- `feature_group`

**Zone 2: Per-sample peak areas** (columns matching `datafile:*.mzML:area`):

These are the quantification values. Each column corresponds to one sample. Example column name:
```
datafile:Mots_18102024_Foodomics_CSF1_S46.mzML:area
```

The filename embedded in the column header (`Mots_18102024_Foodomics_CSF1_S46.mzML`) is the key that maps to the metadata file.

**Zone 3: Other per-sample data** (all other `datafile:*` columns):

For each sample, MZmine exports additional columns:
- `datafile:{filename}.mzML:feature_state` — `DETECTED`, `ESTIMATED`, or `MISSING`
- `datafile:{filename}.mzML:height` — peak height
- `datafile:{filename}.mzML:mz` — per-sample measured m/z
- `datafile:{filename}.mzML:rt` — per-sample retention time
- `datafile:{filename}.mzML:fwhm` — peak width at half maximum
- `datafile:{filename}.mzML:tailing_factor` — peak tailing
- `datafile:{filename}.mzML:asymmetry_factor` — peak asymmetry
- `datafile:{filename}.mzML:charge`
- `datafile:{filename}.mzML:fragment_scans`
- `datafile:{filename}.mzML:isotopes`
- `datafile:{filename}.mzML:rt_ms2_apex_distance`
- `datafile:{filename}.mzML:mz_range:min`, `mz_range:max`
- `datafile:{filename}.mzML:rt_range:min`, `rt_range:max`
- `datafile:{filename}.mzML:intensity_range:min`, `intensity_range:max`

These are stored for QC purposes but not used in the main processing pipeline.

### `--metadata` (required)

A user-provided CSV mapping sample filenames to groups and optional sample weights.

**Required columns:**
- `sample_file` — must match the filename inside the MZmine `datafile:...:area` column headers (e.g., `Mots_18102024_Foodomics_CSF1_S46.mzML`)
- `sample_name` — clean name used throughout the project (e.g., `CSF1_rep1`)
- `group` — group assignment for prevalence filtering and statistical comparisons

**Optional columns:**
- `weight` — sample weight in grams (for sample-weight normalization in `metab process`)
- Any additional columns (e.g., `variety`, `processing` for factorial designs) — stored in metadata but not used by processing commands

**Example (2 groups):**
```csv
sample_file,sample_name,group,weight
Mots_18102024_Foodomics_CSF1_S46.mzML,CSF1_rep1,control,1.50
Mots_18102024_Foodomics_CSF1_S47.mzML,CSF1_rep2,control,1.50
Mots_18102024_Foodomics_CSF1_S48.mzML,CSF1_rep3,control,1.50
Mots_18102024_Foodomics_CSF2_S49.mzML,CSF2_rep1,fermented,2.06
Mots_18102024_Foodomics_CSF2_S50.mzML,CSF2_rep2,fermented,2.06
Mots_18102024_Foodomics_CSF2_S51.mzML,CSF2_rep3,fermented,2.06
```

**Example (9 groups × 3 reps):**
```csv
sample_file,sample_name,group,variety,processing
file_01.mzML,red_ctrl_rep1,red_control,red,control
file_02.mzML,red_ctrl_rep2,red_control,red,control
file_03.mzML,red_ctrl_rep3,red_control,red,control
file_04.mzML,red_malt_rep1,red_malted,red,malted
file_05.mzML,red_malt_rep2,red_malted,red,malted
file_06.mzML,red_malt_rep3,red_malted,red,malted
file_07.mzML,red_dehull_rep1,red_dehulled,red,dehulled
...
```

## Implementation

### Step 1: Parse the Feature Table

Read the CSV and split into three zones:

1. **Detect zone boundaries.** Scan column names for the first column matching the regex `^datafile:.*\.mzML:`. Everything before that is Zone 1 (consensus). Everything matching `datafile:*.mzML:area` is Zone 2 (quantification). Everything else matching `datafile:*` is Zone 3 (per-sample details).

2. **Zone 1 → `ms1_features.parquet`.** Extract all consensus columns. Rename `id` to `feature_id`. Set `feature_id` as the index. Store types as-is (MZmine exports mixed types: numeric for mz/rt/area, string for ion identities, etc.).

3. **Zone 2 → `feature_table_raw.parquet`.** Extract `datafile:*:area` columns. For each column, parse the filename from the pattern `datafile:{filename}:area`. Map each filename to `sample_name` using the metadata. Set `feature_id` as the index, `sample_name` values as columns. Values are peak areas (float64, with NaN for missing/undetected features).

4. **Zone 3 → `sample_details.parquet`.** Extract all remaining `datafile:*` columns. Restructure into a multi-level column index: `(sample_name, metric)` where metric is `feature_state`, `height`, `mz`, `rt`, `fwhm`, etc. Set `feature_id` as the index.

### Step 2: Validate Metadata

- Check that `sample_file`, `sample_name`, and `group` columns exist. Error if any are missing.
- Check that every `sample_file` in the metadata matches a `datafile:*:area` column in the feature table. Error if any metadata entry has no matching column.
- If the feature table has `datafile:*:area` columns not represented in the metadata, warn and drop those samples from the quantification table.
- Check that `sample_name` values are unique. Error if duplicates exist.

### Step 3: Store Metadata

Save as `metadata.parquet` with `sample_name` as the index.

### Step 4: Create Manifest

Initialize `metabproject.json`:
```json
{
  "project_name": "my_project",
  "created": "2026-05-28T14:00:00Z",
  "operations": [
    {
      "command": "init",
      "timestamp": "2026-05-28T14:00:00Z",
      "parameters": {
        "feature_table": "mzmine_feature_table.csv",
        "metadata": "samples.csv"
      },
      "results": {
        "n_features": 2341,
        "n_samples": 6,
        "groups": {"control": 3, "fermented": 3}
      }
    }
  ]
}
```

### Step 5: Print Summary

```
Project 'my_project' initialized
Features: 2,341
Samples: 6
Groups: control (3), fermented (3)
```

## Output Files

```
my_project/
├── metabproject.json
└── features/
    ├── feature_table_raw.parquet   # shape: (n_features, n_samples), values = peak areas
    ├── metadata.parquet            # shape: (n_samples, n_metadata_cols)
    ├── ms1_features.parquet        # shape: (n_features, n_consensus_cols)
    ├── sample_details.parquet      # shape: (n_features, n_samples × n_detail_metrics)
    └── detection_mask.parquet      # shape: (n_features, n_samples), boolean — True = MZmine DETECTED
```

The detection mask is derived from Zone 3 `sample_details` as
`sample_details.xs('feature_state', level='metric', axis=1) == 'DETECTED'`,
with raw `.mzML` filenames renamed to `sample_name` (same mapping applied to
the areas matrix). It powers `metab process --prevalence-mode detected`. The
file is only written when the MZmine export includes a `feature_state` column.

## Error Handling

- Project directory already exists → error (use a different name or delete first)
- Feature table file not found → error with path
- Metadata file not found → error with path
- Metadata missing required columns → error listing which columns are missing
- Metadata `sample_file` not found in feature table → error listing unmatched entries
- Duplicate `sample_name` in metadata → error

## Python API

```python
from metabproject import MetabProject

project = MetabProject.init(
    path="my_project",
    feature_table="mzmine_feature_table.csv",
    metadata="samples.csv"
)

# Access data
project.raw_feature_table   # pandas DataFrame (n_features × n_samples)
project.metadata             # pandas DataFrame (n_samples × n_metadata_cols)
project.ms1_features         # pandas DataFrame (n_features × n_consensus_cols)
```

## Testing

- Fixture: synthetic MZmine-format CSV with realistic column names (50 features × 6 samples, 2 groups × 3 reps), including consensus columns, `datafile:*:area` columns, and `datafile:*:feature_state` columns
- Test zone detection (consensus vs. area vs. detail columns)
- Test column renaming (MZmine verbose names → sample names via metadata)
- Test metadata validation (missing columns, unmatched files, duplicate names)
- Test that unmatched feature table samples are dropped with warning
- Test manifest creation and content
- Test that re-running init on existing directory errors
