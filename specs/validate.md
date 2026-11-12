# `metab validate` Specification

## Overview

Runs data quality checks on a project. Currently implements one check:

- **`iin`** — Ion Identity Network (IIN) consistency: verifies that features MZmine grouped as the same compound (different adducts) are internally consistent in fold-change direction and structural annotation.

Writes results to `<project>/validation/`. Appends one manifest entry per run.

Prerequisites for the `iin` check:
- `ms1_features.parquet` must contain IIN columns (requires Ion Identity Networking to have been enabled in MZmine at export time)
- At least one contrast parquet must exist in `stats/`
- `annotations/merged.parquet` must exist

---

## Ion Identity Network (IIN) Check

MZmine's IIN groups features that are believed to be the same compound ionised as different adducts (e.g., `[M+H]+`, `[M+Na]+`, `[M+K]+`). Each group shares an `ion_identities:iin_id`. This check asks:

1. **Trend consistency** — do all members agree on fold-change direction per contrast?
2. **Annotation consistency** — do members with high-confidence structural annotations point to the same compound?

### IIN columns in `ms1_features.parquet`

| Column | Description |
|--------|-------------|
| `ion_identities:iin_id` | Network ID (NaN if not in an IIN) |
| `ion_identities:partner_row_ids` | Comma-separated feature IDs of partners |
| `ion_identities:ion_identities` | Adduct label for this feature |
| `ion_identities:neutral_mass` | Consensus neutral mass |

If these columns are absent, `validate` raises a clear error.

### Trend consistency logic

For each IIN group × contrast:

- Only features that appear in the stats table (survived the prevalence filter) are evaluated. Features absent from stats are treated as untested and ignored.
- `trend_consistent = True` if all **tested, non-ns** members agree on direction (all "up" or all "down"), or if all tested members are "ns".
- A mix of "up" and "down" within a group is inconsistent.

### Annotation consistency logic

Only **high-confidence structural annotations** are compared. A feature is HC-annotated if its `inchikey2d` in `merged.parquet` is non-null — which means it has a GNPS library match or a high-confidence SIRIUS prediction (≥ `--sirius-min-confidence`). CANOPUS-only, suspect-only, and unannotated features are excluded from the comparison.

- If `<2` members in the group have HC structural annotations, `annotation_consistent = None` (insufficient data).
- **Rank-1 check**: if all HC-annotated members share the same `inchikey2d`, `annotation_consistent = True`.
- **Top-k check** (when `annotations/sirius_topk.parquet` exists): extend each HC-annotated member's candidate set with their top-k SIRIUS `sirius_InChIkey2D` values. If any key is shared across all members' candidate sets, `annotation_consistent = True`. InChIKey2D is adduct-independent so the same compound should yield the same key regardless of adduct.
- Otherwise `annotation_consistent = False`.

`sirius_topk.parquet` is written by `metab annotate --sirius-top-k N` (N > 1). Default is 1 (rank-1 only, no extra file).

---

## Output

### `validation/iin_consistency.parquet`

One row per IIN group, indexed by `iin_id`.

| Column | Type | Description |
|--------|------|-------------|
| `iin_id` | int | Ion identity network ID (index) |
| `n_features` | int | Total adduct members in the group |
| `feature_ids` | str | Comma-separated feature IDs |
| `neutral_mass` | float | Consensus neutral mass from ms1_features |
| `ion_identities` | str | Adduct labels, semicolon-separated |
| `n_hc_annotated` | int | Members with a HC structural annotation (non-null inchikey2d) |
| `annotation_sources` | str | Unique annotation sources across HC-annotated members |
| `unique_inchikeys_rank1` | int | Distinct rank-1 InChIKey2D values across HC-annotated members |
| `shared_inchikey_rank1` | str/NA | The shared InChIKey2D if all HC-annotated rank-1 agree |
| `shared_inchikey_topk` | str/NA | Any shared InChIKey2D found across top-k sets |
| `annotation_consistent` | bool/NA | True = agree, False = disagree, NA = insufficient |
| `trend_<contrast>_consistent` | bool | Per-contrast direction agreement |
| `trend_<contrast>_directions` | str | Observed directions in the group (e.g., "down,up") |
| `all_consistent` | bool | True if annotation_consistent (or NA) AND all trend columns are True |

The number of `trend_*` columns equals the number of contrasts available in `stats/`.

### Manifest entry

```json
{
  "command": "validate",
  "timestamp": "...",
  "parameters": {
    "checks": ["iin"],
    "sirius_top_k": 5
  },
  "results": {
    "iin": {
      "n_iin_groups": 42,
      "n_features_in_iins": 108,
      "n_trend_consistent": 35,
      "n_trend_inconsistent": 7,
      "n_annotation_consistent": 10,
      "n_annotation_inconsistent": 3,
      "n_annotation_insufficient": 29,
      "n_all_consistent": 9
    }
  }
}
```

---

## CLI

```
metab validate <project>
    [--check iin]            # repeatable; defaults to all checks
    [--sirius-top-k N]       # candidates for top-k annotation comparison (default 5)
    [--force]
```

`--sirius-top-k` only has effect when `annotations/sirius_topk.parquet` exists (written by `metab annotate --sirius-top-k N`). If the file is absent, the check falls back to rank-1 comparison only.

---

## Python API

```python
project.validate(checks=["iin"], sirius_top_k=5, force=False)
project.iin_validation  # lazy property → validation/iin_consistency.parquet
```

---

## Implementation

```
src/metabproject/validation/
    __init__.py
    ion_identity.py     # parse_iin_groups, check_trend_consistency,
                        # check_annotation_consistency, build_iin_validation_table
tests/test_validation/
    test_ion_identity.py
```

`load_sirius_topk(structures_path, top_k)` in `annotation/sirius.py` loads ranks 1–k with a non-unique `feature_id` index. Called by `annotate()` when `sirius_top_k > 1`.
