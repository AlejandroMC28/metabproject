# `metab annotate` Specification

## Overview

Merges external compound annotations (SIRIUS structure predictions, CANOPUS
compound class predictions, and GNPS spectral library / suspect-list matches)
onto the project's feature set. Writes per-source parquets and a skinny
merged table to `<project>/annotations/`. Appends one manifest entry.

---

## Input Files

All four sources are tab-separated (`.tsv`). `--gnps-suspect` is optional.

### 1. `structure_identifications.tsv` (SIRIUS)

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `mappingFeatureId` | int | MZmine `feature_id` |
| `structurePerIdRank` | int | Prediction rank (1 = best) |
| `ConfidenceScoreApproximate` | float | SIRIUS CSI:FingerID confidence (0–1) |
| `InChIkey2D` | str | 2D InChIKey |
| `InChI` | str | Full InChI |
| `name` | str | Compound name |
| `smiles` | str | SMILES string |
| `molecularFormula` | str | Molecular formula |
| `adduct` | str | Adduct type |

Only rank-1 rows are kept (`structurePerIdRank == 1`).

### 2. `canopus_structure_summary.tsv` (CANOPUS)

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `mappingFeatureId` | int | MZmine `feature_id` |
| `NPC#pathway` | str | NPC pathway |
| `NPC#superclass` | str | NPC superclass |
| `NPC#class` | str | NPC class |
| `ClassyFire#most specific class` | str | ClassyFire most-specific (deepest assigned) class |

Optional columns (loaded if present — newer SIRIUS/CANOPUS exports):

| Column | Type | Description |
|--------|------|-------------|
| `ClassyFire#subclass` | str | ClassyFire subclass |
| `ClassyFire#subclass Probability` | float | Probability for the subclass call |
| `ClassyFire#level 5` | str | ClassyFire level-5 (below subclass) |
| `ClassyFire#level 5 Probability` | float | Probability for the level-5 call |

### 3. GNPS library TSV (`merged_results_with_gnps.tsv`)

Required columns: `#Scan#`, `Compound_Name`. Additional columns used when present:

| Column | Renamed to |
|--------|------------|
| `#Scan#` | `feature_id` |
| `Compound_Name` | `gnps_compound_name` |
| `Smiles` | `gnps_smiles` |
| `INCHI` | `gnps_InChI` |
| `InChIKey-Planar` | `gnps_InChIkey2D` |
| `MQScore` | `gnps_MQScore` |
| `IonMode` | `gnps_ion_mode` |
| `npclassifier_pathway` | `gnps_npc_pathway` |
| `npclassifier_superclass` | `gnps_npc_superclass` |
| `npclassifier_class` | `gnps_npc_class` |
| `CF_NPC_class` | `gnps_classyfire_class` |

### 4. GNPS suspect-list TSV (optional)

Required columns:

| Column | Description |
|--------|-------------|
| `#Scan#` | MZmine `feature_id` |
| `Compound_Name` | Parsed compound name string (see Suspect parsing below) |
| `LibraryName` | Used to filter to `GNPS-SUSPECTLIST.mgf` entries only |

---

## Processing Steps

### SIRIUS loading (`annotation/sirius.py`)

1. Read `structure_identifications.tsv` and `canopus_structure_summary.tsv`.
2. Filter SIRIUS to `structurePerIdRank == 1`.
3. Left-merge CANOPUS onto SIRIUS on `mappingFeatureId` (CANOPUS is the left table so all CANOPUS features are kept even if SIRIUS has no match for that feature).
4. Rename columns per the rename map (`ConfidenceScoreApproximate` → `confidence`, `name` → `compound_name`, etc.).
5. Compute `high_confidence = (confidence >= min_confidence) & confidence.notna()`.
6. Rename `mappingFeatureId` → `feature_id`, cast to `int`, set as index.
7. Prefix all columns with `sirius_`.

### GNPS library loading (`annotation/gnps.py`)

1. Read the TSV.
2. Keep only available columns from the defined set (tolerates missing taxonomy columns from jobs that didn't run NPC/ClassyFire classification).
3. Rename `#Scan#` → `feature_id`, `InChIKey-Planar` → `InChIkey2D`, etc.
4. Cast `feature_id` to `int`, set as index.
5. **Polarity filter** (optional). If `--gnps-polarity` is set to `positive` or `negative`, drop rows whose `IonMode` (case-insensitive) is the opposite polarity. Rows with missing or unrecognized `IonMode` are kept.
6. Prefix all columns with `gnps_`.

### GNPS suspect-list loading (`annotation/gnps.py`)

1. Read the TSV.
2. Filter rows to `LibraryName == "GNPS-SUSPECTLIST.mgf"`.
3. Rename `#Scan#` → `feature_id`, cast to `int`, set as index.
4. Remove features whose `feature_id` is already in the GNPS library index (avoids double-annotation).
5. Parse `Compound_Name` via regex into: `compound_name`, `formula_sirius`, `formula_buddy`, `delta_mz`, `adduct`.
6. Prefix all parsed columns with `suspect_`.

#### Suspect `Compound_Name` regex

The regex handles the pipe-delimited format produced by SIRIUS+CANOPUS suspect-list hits:

```
^(?P<compound_name>[^|]+?)
  (?:\s*\|\s*SIRIUS:\s*(?P<formula_sirius>[A-Za-z0-9]+))?
  (?:\s*\|\s*BUDDY:\s*(?P<formula_buddy>[A-Za-z0-9]+))?
  (?:\s*\|\s*[Δδ]mz:\s*(?P<delta_mz>[-\d.]+))?
  (?:\s*\|\s*adduct:\s*(?P<adduct>\S+))?
\s*$
```

Example input: `"MyMolecule | SIRIUS: C10H12O | BUDDY: C10H12O2 | Δmz: 0.002 | adduct: [M+H]+"`

### SIRIUS top-k loading (`annotation/sirius.py`)

Activated when `--sirius-top-k > 1`.

1. Read `structure_identifications.tsv` again, keeping rows with `structurePerIdRank <= top_k` (instead of only rank-1).
2. Same column rename as rank-1 loader; `structurePerIdRank` is preserved.
3. Index is non-unique (one row per `(feature_id, rank)` pair).
4. Written to `annotations/sirius_topk.parquet`. Consumed by IIN validation and by the GNPS↔SIRIUS cross-check (`compute_gnps_sirius_topk_match`).

### NP Classifier (`annotation/np_classifier.py`)

Activated when `--classify-sirius-structures` is passed.

1. Select sirius rows with `sirius_high_confidence == True` whose `feature_id` is **not** in the GNPS library index (GNPS NPC is preferred and computed offline from a curated structure).
2. Deduplicate by `InChIkey2D` — the API is queried once per unique structure.
3. For each unseen InChIkey, GET `https://npclassifier.gnps2.org/classify?smiles=<sirius_smiles>`. The first non-empty entry in `pathway_results`, `superclass_results`, `class_results` becomes the prediction.
4. Cache (hit + miss) is written to `annotations/np_classifier_cache.parquet` keyed by `InChIkey2D`. A custom path can be passed via `--np-cache`; the same cache can be shared across projects.
5. API failures (network, non-200, bad JSON) cache nulls so reruns don't re-hit. Affected features fall back to CANOPUS in the merge.
6. Polite default: `request_delay=0.1s` between calls; per-request timeout `10s`.

The result is a per-feature DataFrame with `sirius_struct_npc_pathway`, `sirius_struct_npc_superclass`, `sirius_struct_npc_class`, passed to `merge_annotations`.

---

## Merge Logic (`annotation/merge.py`)

### Outer join

All three source DataFrames are outer-joined on `feature_id`.

### Presence masks

- `gnps_has` — `gnps_compound_name` is not null.
- `sirius_has` — `sirius_InChIkey2D` is not null **AND** `sirius_high_confidence` is True.
- `suspect_has` — `suspect_compound_name` is not null.

Note: low-confidence SIRIUS hits (below `--sirius-min-confidence`) **do not** appear in any curated merged column or in `annotation_source`. They are preserved in `sirius.parquet` for inspection.

### Curated columns

#### Structural (priority: GNPS > SIRIUS high-conf > suspect)

| Merged column | Priority chain |
|---------------|----------------|
| `compound_name` | `gnps_compound_name` → `sirius_compound_name` → `suspect_compound_name` |
| `mol_formula` | `sirius_mol_formula` → `suspect_formula_sirius` |
| `adduct` | `sirius_adduct` → `suspect_adduct` |
| `inchikey2d` | `gnps_InChIkey2D` → `sirius_InChIkey2D` |
| `smiles` | `gnps_smiles` → `sirius_smiles` |
| `confidence` | `gnps_MQScore` (when gnps_has) or `sirius_confidence` (when sirius_has only) |

#### Taxonomy

NPC priority: **GNPS library → SIRIUS-structure-derived (when high-confidence, opt-in via `--classify-sirius-structures`) → CANOPUS** (per-level, probability-gated).
ClassyFire priority: **GNPS library → CANOPUS** (no structure-derived source).

| Merged column | Priority chain |
|---------------|----------------|
| `npc_pathway` | `gnps_npc_pathway` → `sirius_struct_npc_pathway` (when `sirius_high_confidence`) → `canopus_npc_pathway` (when `canopus_npc_pathway_passed`) |
| `npc_superclass` | `gnps_npc_superclass` → `sirius_struct_npc_superclass` (when `sirius_high_confidence`) → `canopus_npc_superclass` (when `canopus_npc_superclass_passed`) |
| `npc_class` | `gnps_npc_class` → `sirius_struct_npc_class` (when `sirius_high_confidence`) → `canopus_npc_class` (when `canopus_npc_class_passed`) |
| `classyfire_class` | `gnps_classyfire_class` → `canopus_classyfire_class` (when `canopus_classyfire_class_passed`) — note: this is the ClassyFire **most specific class** (deepest assigned level), kept under this name for backward compatibility |
| `classyfire_subclass` | `canopus_classyfire_subclass` (when `canopus_classyfire_subclass_passed`) — CANOPUS-only, omitted from older exports |
| `classyfire_level5` | `canopus_classyfire_level5` (when `canopus_classyfire_level5_passed`) — CANOPUS-only, omitted from older exports |

The structure-derived NPC columns come from the GNPS2 NP Classifier (`https://npclassifier.gnps2.org`), queried per unique InChIKey-2D and cached at `<project>/annotations/np_classifier_cache.parquet` (or a path provided via `--np-cache`). Features already annotated by a GNPS library hit are skipped (GNPS preferred).

### `annotation_source`

Colon-joined string of every contributing source per feature:

- Includes `"gnps"` if `gnps_has`
- Includes `"sirius"` if `sirius_has` (high-confidence only)
- Includes `"sirius_struct_npc"` if structure-derived NPC fired for this feature
- Includes `"canopus"` if any CANOPUS level has `*_passed == True`
- Includes `"suspect"` if `suspect_has`

Examples: `"gnps"`, `"sirius:sirius_struct_npc"`, `"canopus"`, `"gnps:sirius:canopus"`, `"gnps:sirius:canopus:suspect"`.

### `is_suspect_only`

Boolean. `True` iff `annotation_source == "suspect"`. Convenient filter for downstream steps that want to exclude tentative-only annotations.

### `is_canopus_only`

Boolean. `True` iff `annotation_source == "canopus"`. Features with only CANOPUS class predictions and no structural identification.

### `inchikey_agreement`

Only defined when both GNPS and SIRIUS (high-confidence) annotate the same feature:

| Value | Condition |
|-------|-----------|
| `"agree"` | Both present, `gnps_InChIkey2D == sirius_InChIkey2D` |
| `"disagree"` | Both present, InChIKeys differ |
| `"single_source"` | Only one of GNPS/SIRIUS present |
| `NA` | Neither present (suspect-only feature) |

### `*_alt` columns (conflict preservation)

Populated only when `inchikey_agreement == "disagree"` (GNPS preferred; SIRIUS values go to alt):

- `compound_name_alt` — `sirius_compound_name`
- `inchikey2d_alt` — `sirius_InChIkey2D`
- `smiles_alt` — `sirius_smiles`
- `mol_formula_alt` — `sirius_mol_formula`

Taxonomy alt columns are populated whenever GNPS and CANOPUS taxonomies differ:

- `npc_class_alt` — `canopus_npc_class` when `gnps_npc_class != canopus_npc_class`
- `classyfire_class_alt` — `canopus_classyfire_class` when `gnps_classyfire_class != canopus_classyfire_class`

### GNPS↔SIRIUS top-k cross-check

When `--sirius-top-k > 1`, two additional columns are populated for every feature with a GNPS library match:

| Column | Meaning |
|--------|---------|
| `gnps_sirius_rank` | `Int64`. Lowest SIRIUS rank whose `InChIkey2D` equals the GNPS planar key; `0` if no candidate matches; `<NA>` if SIRIUS produced no candidates for the feature. |
| `gnps_sirius_highconf` | `bool`. `True` iff the matched candidate's `sirius_confidence` is ≥ `sirius_min_confidence`. |

These columns are an orthogonal confidence signal for library matches — a GNPS hit confirmed by SIRIUS at rank 1 with high confidence is the most trustworthy outcome.

### Unannotated feature exclusion

Features with no qualifying annotation (neither `gnps_has`, `sirius_has`, nor `suspect_has`) are **absent** from `merged.parquet`. Rejoin to `ms1_features` on `feature_id` to recover the full feature list.

---

## Output Files

All written to `<project>/annotations/`:

| File | Contents |
|------|----------|
| `sirius.parquet` | All SIRIUS rank-1 predictions, prefixed `sirius_*`, indexed by `feature_id` |
| `canopus.parquet` | CANOPUS class predictions with per-level `canopus_*_probability` and `canopus_*_passed` columns, indexed by `feature_id`. Used by `metab enrich`. |
| `gnps.parquet` | Cleaned GNPS library matches, prefixed `gnps_*`, indexed by `feature_id` |
| `gnps_suspect.parquet` | Cleaned suspect-list matches, prefixed `suspect_*`, indexed by `feature_id`. Written only when `--gnps-suspect` is provided. |
| `sirius_topk.parquet` | All SIRIUS rank-1..k candidates, indexed by `feature_id` (non-unique). Written when `--sirius-top-k > 1`. Consumed by IIN validation and by the GNPS↔SIRIUS top-k cross-check. |
| `np_classifier_cache.parquet` | NP Classifier cache keyed by InChIKey-2D. Written/read when `--classify-sirius-structures` is used. Reused across reruns; safe to share between projects via `--np-cache`. |
| `merged.parquet` | Skinny curated table with annotated features only (see column schema above) |

---

## Manifest Entry

### Parameters

```json
{
  "sirius_structures": "<path>",
  "canopus": "<path>",
  "gnps_library": "<path>",
  "gnps_suspect": "<path or null>",
  "sirius_min_confidence": 0.64,
  "canopus_min_probability": 0.7,
  "sirius_top_k": 1,
  "gnps_polarity": "positive",
  "classify_sirius_structures": false,
  "np_cache": null,
  "priority_chain": ["gnps", "sirius_high_confidence", "sirius_struct_npc", "canopus", "suspect"]
}
```

### Results

```json
{
  "n_sirius_rank1": 694,
  "n_sirius_high_confidence": 93,
  "n_gnps_library": 26,
  "n_gnps_suspect_unique": 49,
  "n_canopus_features_total": 1200,
  "n_canopus_pathway_passed": 850,
  "n_canopus_superclass_passed": 780,
  "n_canopus_class_passed": 620,
  "n_canopus_classyfire_passed": 540,
  "n_features_annotated": 168,
  "n_features_canopus_only": 35,
  "n_features_suspect_only": 41,
  "n_inchikey_agree": 3,
  "n_inchikey_disagree": 2,
  "n_taxonomy_disagree": 7,
  "n_gnps_sirius_rank1": 12,
  "n_gnps_sirius_topk_match": 18,
  "n_gnps_sirius_no_match": 5,
  "n_gnps_sirius_highconf": 10,
  "n_sirius_struct_npc_classified": 67
}
```

---

## Python API

```python
from metabproject.project import MetabProject

p = MetabProject("my_project/")

p.annotate(
    sirius_structures="structure_identifications.tsv",
    canopus="canopus_structure_summary.tsv",
    gnps_library="merged_results_with_gnps.tsv",
    gnps_suspect="merged_results_with_gnps_suspect.tsv",  # optional
    sirius_min_confidence=0.64,
    sirius_top_k=5,                       # enables GNPS↔SIRIUS top-k cross-check columns
    gnps_polarity="positive",             # optional: drop opposite-polarity library matches
    classify_sirius_structures=True,      # opt-in: NP Classifier on high-conf SIRIUS SMILES
    np_cache=None,                        # defaults to <project>/annotations/np_classifier_cache.parquet
    force=False,
)

# Lazy-loaded properties:
p.annotations            # merged.parquet — skinny curated table
p.sirius_annotations     # sirius.parquet
p.gnps_annotations       # gnps.parquet
p.gnps_suspect_annotations  # gnps_suspect.parquet or None
```

---

## Idempotency

Running `metab annotate` a second time without `--force` raises `FileExistsError`. With `--force`, existing parquets in `annotations/` are deleted and a new manifest entry is appended (the old entry is not removed — the manifest is append-only).

---

## Errors

| Condition | Exception |
|-----------|-----------|
| `annotations/merged.parquet` exists, no `--force` | `FileExistsError` |
| Any required input file not found | `FileNotFoundError` |
| Required columns missing from input TSV | `ValueError` |
