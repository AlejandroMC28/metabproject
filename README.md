# metabproject

A Python CLI package for persistent, project-based untargeted LC-MS/MS
metabolomics data processing. **One MZmine run = one project.**

Initialize a project from MZmine outputs, process the feature table, then
incrementally add annotations, stats, and enrichment analyses — all tracked in a
single project directory you can return to later.

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
# Create a project from an MZmine feature table + sample metadata
metab init my_project \
  --feature-table mzmine_feature_table.csv \
  --metadata samples.csv

# Process the feature table (filter → impute → correct → normalize → transform)
metab process my_project \
  --min-prevalence 2 \
  --impute half-min \
  --weight-normalize \
  --transform log2
```

## Project layout

```
my_project/
├── metabproject.json      # Manifest: append-only operations log
├── features/              # Feature tables + metadata (parquet)
├── qc/                    # Processing QC figure
├── annotations/           # metab annotate (SIRIUS / CANOPUS / GNPS merge)
├── stats/                 # metab stats (per-contrast Welch t-tests)
├── enrichment/            # metab enrich (CANOPUS class enrichment)
├── validation/            # metab validate (IIN consistency)
├── viz/                   # metab viz (PCA, volcano, heatmap)
└── exports/               # metab explore (interactive HTML viewer)
```

## Python API

Every CLI command maps to a `MetabProject` method:

```python
from metabproject import MetabProject

project = MetabProject.init("my_project", feature_table="...", metadata="...")
project.process(min_prevalence=2, impute="half-min", transform="log2")

project.feature_table       # processed DataFrame
project.raw_feature_table   # original DataFrame (never modified)
project.metadata
project.ms1_features
```

## Documentation

- [`specs/init.md`](specs/init.md) — `metab init` (MZmine parsing, validation)
- [`specs/process.md`](specs/process.md) — `metab process` (pipeline + QC)
- [`specs/annotate.md`](specs/annotate.md) — `metab annotate` (SIRIUS / CANOPUS / GNPS merge)
- [`specs/stats.md`](specs/stats.md) — `metab stats` (Welch t-tests + BH-FDR)
- [`specs/enrich.md`](specs/enrich.md) — `metab enrich` (CANOPUS class enrichment)
- [`specs/validate.md`](specs/validate.md) — `metab validate` (IIN consistency)
- [`specs/viz.md`](specs/viz.md) — `metab viz` (PCA, volcano, heatmap)
- [`specs/explore.md`](specs/explore.md) — `metab explore` (interactive HTML viewer)

## Development

```bash
pytest
```
