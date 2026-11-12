# `metab explore` Specification

## Overview

Generates a self-contained interactive HTML viewer for browsing annotated features.
Pulls together all available project data — processed abundances, annotations,
statistical results, IIN groups, and database source info — into a single file
that opens in any browser with no server required.

Works at any pipeline stage. Sections populate based on what data is available;
`process` is the only prerequisite.

Writes to `<project>/exports/explore.html` by default. Appends one manifest entry per run.
Unlike other commands, overwriting is always allowed (no `--force` needed — this is a
read-only export).

---

## Views

### 1. Feature List (default)

A filterable, paginated table (50 rows/page) of all features. Columns: feature ID,
m/z, RT, compound name, annotation source, NPC class, log2FC, p_adj, direction.

**Filters:**
- Feature ID exact-match search
- Compound name text search
- Annotation source checkboxes (gnps, sirius, canopus, suspect, unannotated)
- Contrast selector + max p_adj threshold
- Direction (up / down / ns)
- NPC pathway dropdown
- NPC class dropdown
- Factorial comparison selector + classification-bucket checkboxes
  (e.g. `interaction`, `interaction_only`, `factor1_only`, …) — populated
  from `filter_options.comparisons[<name>].classifications`
- IIN members only checkbox
- Derived ions dropdown (only with `feature_relations`): hide / only confident
  in-source fragments, only / hide derived ions of any kind

With `feature_relations`, a derived ion carries a badge before its compound
name: `ISF→<parent>` (confident in-source fragment) or `<role>→<parent>`
(isotope, adduct, unconfirmed loss).

The table includes a `Bucket` column showing the per-feature
classification for the selected comparison.

Clicking a row navigates to the Feature Detail view.

### 2. Feature Detail

Full view for a single feature, reached by clicking the list or via the tab.

**Panels:**

| Panel | Content |
|-------|---------|
| Annotation card | compound name, formula, adduct, SMILES, InChIKey2D, annotation source, confidence |
| 2D structure | SVG rendered from SMILES via RDKit (if installed), else placeholder |
| Compound class taxonomy | NPC pathway → superclass → class, ClassyFire class as a breadcrumb chain |
| Annotation sources | Per-source breakdown: GNPS (MQScore), SIRIUS (confidence, high_confidence flag), CANOPUS (class probabilities), suspect list |
| Boxplots | One Plotly boxplot per statistical contrast. Shows both groups with individual sample points (jittered). Annotated with log2FC and p_adj. |
| MS2 spectrum | If the feature has an ingested MS2 spectrum (`metab spectra`): an interactive Plotly centroid stem plot of the cleaned peak list (base-peak normalised), with the precursor m/z marked and hover showing m/z + relative intensity. Hidden when no spectrum is stored. |
| Factorial comparisons | One grouped Plotly boxplot per persisted `Comparison` (`anova_1way` / `anova_2way` / `welch_anova` / `kruskal`). For a 2-factor design, x-axis = factor1 levels, one box per (factor1, factor2) cell, individual sample points jittered (e.g. 3 varieties × 3 treatments → 9 boxes in one panel). Header carries the classification badge (2-way), per-term p-value chips, and the per-feature post-hoc pairs table (significant-only by default, toggle to show all). |
| Related ions | Only with `feature_relations`. On a derived ion: its role (in-source fragment / isotope / adduct), the clickable parent, the relation (e.g. `-hexose`), MZmine correlation r and, for losses, MS2 support, across-sample r, fragment/parent ratio CV and the confident flag. On a parent: the table of ions derived from it. Parents not in the processed table are shown unlinked. |
| IIN context | If the feature belongs to an IIN group: group ID, neutral mass, table of partner features (clickable), link to IIN Group view |
| Molecular network | If the feature belongs to a GNPS molecular family: family ID, member count, this node's degree, and a "View molecular network →" button linking to the Network view |

### 3. Normality Visualization

Per-feature 2-way ANOVA residual diagnostics for any feature with a
`residual_diagnostics_per_feature.parquet` row in the active comparison.
Reached via the top-level **Normality Visualization** tab or from a
Feature Detail card. Shows the headline numbers (`r2`, `residual_sigma`,
`max_abs_std_resid`, `shapiro_p`, `shapiro_p_adj`, `normal_residuals`)
and the QQ / residual-distribution figure (rendered by
`viz_feature_residuals` or inline from the parquet row).

### 4. IIN Group View

A sidebar lists all IIN groups (searchable by neutral mass or group ID). Selecting a
group shows:

- Group header: IIN ID, neutral mass, member count, adduct labels
- Annotation consistency badge (consistent / inconsistent / insufficient data)
- Per-contrast trend consistency badges
- Member table: feature ID, ion identity, compound name, annotation source, per-contrast log2FC and direction
- Side-by-side Plotly boxplots for all members per contrast (each member = distinct color)

### 5. Network View

Present only when a GNPS molecular network has been ingested (`metab network`);
the tab is hidden otherwise. A sidebar lists all molecular families (searchable
by family ID or member compound name, sorted by size). Selecting a family shows:

- Family header: family ID, feature count, edge count
- **Node-link graph** — the family drawn as an interactive Plotly graph. Node
  positions are computed in-browser with a deterministic Fruchterman-Reingold
  force layout (circular seeding; circular fallback and no labels for families
  `> 60` nodes). Edge width ∝ cosine similarity; node colour = annotation source.
  Hovering shows edge cosine / Δm/z and node id / compound / m/z / RT.
- **Per-treatment boxplots** — clicking any node (or member-table row) renders
  that feature's abundances grouped by treatment, using the first factorial
  comparison's design (`boxmode:'group'`); falls back to one box per
  `groups` when no comparison is defined.
- Member table: feature ID, compound, source, m/z, RT (clickable → boxplot).
- **Compare MS2 spectra** — a head-to-tail (mirror) plot of any two family
  members that carry an ingested MS2 spectrum (top upright, bottom inverted, each
  base-peak normalised). When `metab similarity` has been run, **Score type**
  (modified / direct cosine) and **Scaling** (linear / sqrt / MassBank) `<select>`s
  drive the plot: the matchms score + matched-peak count are shown, matched peaks
  are drawn in a highlight colour with faint connectors across the mirror axis, and
  unmatched peaks are dimmed. The pre-ingested GNPS cosine is kept as a secondary
  note when the two nodes share an edge. The toggles are hidden (GNPS-cosine-only
  behaviour) when similarity has not been computed. See `specs/similarity.md`.

Reached via the top-level **Network** tab or the Feature Detail "View molecular
network →" button.

---

## Output

### `exports/explore.html`

A fully self-contained HTML file. By default, Plotly.js (~3.5 MB) is embedded inline.
All project data is serialised as a JSON object and embedded in a `<script>` block.
The file has no external dependencies and can be opened offline.

File size is typically:
- 4–6 MB with `--cdn` and no structures
- 7–10 MB with embedded Plotly and no structures
- Larger when RDKit structure SVGs are included

### Manifest entry

```json
{
  "command": "explore",
  "timestamp": "...",
  "parameters": {
    "output": "<project>/exports/explore.html",
    "render_structures": true,
    "embed_plotly": true,
    "feature_relations": null
  },
  "results": {
    "n_features": 3466,
    "n_annotated": 821,
    "n_contrasts": 2,
    "n_iin_groups": 142,
    "file_size_mb": 8.4
  }
}
```

---

## CLI

```
metab explore <project>
    [-o / --output PATH]    # output HTML path (default: <project>/exports/explore.html)
    [--no-structures]       # skip RDKit rendering (faster, no rdkit required)
    [--cdn]                 # use Plotly CDN instead of embedding (~3.5 MB smaller, needs internet)
```

---

## Python API

```python
path = project.explore(
    output=None,             # defaults to <project>/exports/explore.html
    render_structures=True,  # False if rdkit not installed
    embed_plotly=True,       # False for CDN mode
    feature_relations=None,  # optional TSV of derived-ion links (API only, no CLI flag)
)
```

Returns the `Path` to the generated file.

`feature_relations` is a TSV with one row per derived feature: `feature_id`,
`parent`, `role` (`isotope` / `adduct` / `loss`), `relation`, and optionally
`r`, `ms2_support`, `r_samples`, `ratio_cv`, `isf_confident`. The package does
not compute it; it is supplied by the analysis (e.g. an analysis
repo's `feature_relations.py`). Each feature gets `derived_from` (dict or null) and
`derived_ions` (list) in the viewer data.

---

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `plotly>=5.18` | Yes | Interactive boxplots + JS bundle |
| `jinja2>=3.1` | Yes | HTML templating |
| `rdkit` | Optional | 2D structure rendering from SMILES |

If `rdkit` is not installed, `render_structures` degrades gracefully — structures
show a placeholder and a `UserWarning` is emitted. Install with:
```
pip install rdkit
```

---

## Implementation

```
src/metabproject/explore/
    __init__.py             # public API: assemble_explore_data, generate_explore_html
    data.py                 # assemble_explore_data() — builds JSON-serialisable dict
    structures.py           # render_structures() — SMILES → SVG via RDKit
    report.py               # generate_explore_html() — renders Jinja2 template
    templates/
        viewer.html         # self-contained HTML/CSS/JS template

tests/test_explore/
    test_data.py
    test_structures.py
    test_structures_no_rdkit.py
    test_report.py
tests/test_integration/
    test_explore_integration.py
```

### Data assembly (`data.py`)

`assemble_explore_data(project, render_structures)` builds a dict with:

- `features` — keyed by `feature_id` (string). Each entry contains: mz, rt, charge,
  processed abundances (sample → value), merged annotation, per-source breakdown,
  structure SVG, per-contrast stats, IIN membership, and `network` membership
  (`{component_id, degree, n_component_members}` or null).
- `network_components` — GNPS molecular families keyed by `component_id` (string).
  Each carries `feature_ids`, `n_features`, `n_edges`, compact `[id1, id2, cosine]`
  edge triples, a light `members` table, and a `similarity` block (keyed by
  `"<lo>|<hi>"` → `score_type` → `scaling` → `{score, matches}`, where `matches`
  is a list of `[idx_a, idx_b]` peak-index pairs; empty when `metab similarity`
  has not run). Empty dict when no network ingested. Abundances are not
  duplicated — the viewer reads `features[fid].abundances`.
- `similarity_meta` — `{score_types, scalings, tolerance}` describing the computed
  matchms similarity, or `null` when none exists. Drives the compare-panel score /
  scaling toggles (and hides them when absent). Built from
  `project.spectral_similarity`; see `specs/similarity.md`.
- `features[<id>].ms2` — the feature's processed MS2 spectrum
  (`{precursor_mz, precursor_charge, n_peaks, base_peak_mz, mz, intensity_rel}`),
  or `null` when no spectrum was ingested for that feature. Built from
  `project.ms2_spectra` via `explore/spectra.process_spectrum_for_display`
  (`spectrum_utils` cleanup at display time, falling back to raw peaks).
- `contrasts_meta` — group labels and sample lists per contrast, recovered by
  re-parsing the contrasts YAML path stored in the manifest's last `stats` entry.
- `iin_groups` — member lists, consistency flags, denormalised member data for fast
  rendering. Built from `iin_consistency.parquet` if available, else from
  `ms1_features` IIN columns directly.
- `comparisons_meta` — per-comparison summary loaded from
  `project.comparisons` (which scans `<project>/stats/<name>/omnibus.parquet`
  and `posthoc.parquet`). Each entry carries `kind`, `factors`, `posthoc`,
  `alpha`, the sorted `levels` per factor, and a `sample → {factor → level}`
  map used by the viewer to bucket samples into the grouped boxplot's cells.
  Per-feature comparison data (omnibus terms, classification, filtered post-
  hoc rows) is attached under `features[<id>].comparisons`. Empty/absent if
  no `Comparison` has been run.
- `filter_options` — precomputed unique values for all filter dropdowns,
  including `comparisons[<name>] = {kind, factors, classifications}` for
  the comparison + bucket filter on the list view.
- `palette` — optional `{level → hex}` map loaded from `<project>/viz.yaml`
  (`palette:` block). Used to colour grouped boxplots and detail-view
  badges consistently with `metab viz` outputs.
- Per-comparison `comparisons_meta[<name>].has_residual_diag` and
  per-feature `comparisons[<name>].residual_diag` (with `in_full_model`,
  `r2`, `residual_sigma`, `max_abs_std_resid`, `shapiro_p`,
  `shapiro_p_adj`, `normal_residuals`) — populated when
  `stats/<name>/residual_diagnostics_per_feature.parquet` exists.

All sections use try/except to load optional data — the function never raises if a
pipeline stage hasn't been run.

### Structure rendering (`structures.py`)

Deduplicates SMILES before rendering (unique SMILES only). Each SVG is 2–5 KB; for
datasets with thousands of annotated features, this is the primary driver of file size.
Use `--no-structures` to skip.
