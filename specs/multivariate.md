# `stats.multivariate` Specification

## Overview

Sibling submodule to `stats/univariate.py`. Provides four multivariate
methods that operate on the processed feature table:

- **PCA** — `run_pca`
- **PCoA** with PERMANOVA + PERMDISP — `run_pcoa`
- **Hierarchical clustering** — `run_hca`
- **Random Forest** with OOB curve and permutation importance —
  `run_random_forest`

Each routine is a pure function returning a `@dataclass`. The module has
no I/O, no plotting, no CLI awareness; assembly into a project lives
elsewhere. This is a library API only — there is **no `metab multivariate`
command** at this time. CLI wiring and `viz/` plotting helpers are
deliberate follow-ups.

The behavioural reference is the
[FBMN-STATS Python notebook](https://github.com/Functional-Metabolomics-Lab/FBMN-STATS).
Defaults match it where reasonable.

---

## Public surface

```python
from metabproject.stats.multivariate import (
    run_pca, PCAResult,
    run_pcoa, PCoAResult,
    run_hca, HCAResult,
    run_random_forest, RFResult,
)
```

Every function takes the processed `feature_table` (features × samples)
and returns a dataclass. NaN inputs are rejected — impute before calling.
All randomness is seeded.

---

## `run_pca`

```python
run_pca(
    feature_table: pd.DataFrame,
    n_components: int = 10,
    scale: bool = True,
) -> PCAResult
```

PCA is fit on the transposed (samples × features) matrix. With
`scale=True`, a `StandardScaler` is applied internally before fitting —
the caller's table is **not** mutated. The number of components is
clamped to `min(n_components, n_samples, n_features)` and reported in
`PCAResult.n_components_used`.

**`PCAResult`**: `scores` (samples × PC), `loadings` (features × PC),
`explained_variance_ratio` (np.ndarray), `n_components_used` (int).

---

## `run_pcoa`

```python
run_pcoa(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,        # indexed by sample name
    group_col: str,
    metric: str = "euclidean",
    permutations: int = 999,
    seed: int = 0,
) -> PCoAResult
```

Distances are computed by `scipy.spatial.distance.pdist`; the ordination,
PERMANOVA, and PERMDISP come from `scikit-bio`.

- **Allowed metrics**: `euclidean`, `cityblock`, `canberra`, `braycurtis`,
  `jaccard`, `minkowski`.
- **PERMANOVA R²** is recomputed manually from the pseudo-F because
  `skbio.stats.distance.permanova` does not return it:
  `R² = (F·(k−1)/(n−k)) / (1 + F·(k−1)/(n−k))`.
- `group_col` must exist in `metadata` and resolve to ≥2 levels with ≥2
  samples each.

**Optional dependency.** scikit-bio is in the `pcoa` extra (see
`pyproject.toml`). `run_pcoa` imports it lazily and raises `ImportError`
with an installation hint when the extra is not installed.

**`PCoAResult`**: `scores`, `explained_variance_ratio`,
`distance_matrix` (skbio `DistanceMatrix`),
`permanova` dict `{test_statistic, p_value, R2, sample_size, number_of_groups}`,
`permdisp` dict `{test_statistic, p_value}`.

---

## `run_hca`

```python
run_hca(
    feature_table: pd.DataFrame,
    metric: str = "euclidean",
    linkage_method: str = "complete",
    n_clusters: int | None = None,
) -> HCAResult
```

Clusters samples via `scipy.spatial.distance.pdist` →
`scipy.cluster.hierarchy.linkage`. Defaults match the FBMN-STATS
notebook: complete linkage on euclidean distances.

- **Allowed linkage methods**: `single`, `complete`, `average`,
  `weighted`, `centroid`, `median`, `ward`. `ward` requires
  `metric='euclidean'`.
- When `n_clusters` is given, a flat assignment is produced via
  `fcluster(..., criterion='maxclust')` and returned as a `pd.Series`
  indexed by sample name. When `None`, `cluster_labels` is `None`.

**`HCAResult`**: `linkage_matrix` (np.ndarray, scipy format),
`cluster_labels` (pd.Series | None), `metric` (str),
`linkage_method` (str).

---

## `run_random_forest`

```python
run_random_forest(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,             # indexed by sample name
    target_col: str,
    n_estimators: int = 500,
    test_size: float = 0.25,
    n_permutation_repeats: int = 10,
    mda_top_k: int | None = None,
    seed: int = 123,
    n_jobs: int | None = -1,
) -> RFResult
```

Stratified `train_test_split` (`random_state=seed`). Class weights are
computed via `sklearn.utils.class_weight.compute_class_weight('balanced',
...)` on the **training** labels and passed to the forest — never on the
full label vector, to avoid leakage. The OOB curve is recorded by
warm-starting a single `RandomForestClassifier(oob_score=True,
warm_start=True, n_jobs=n_jobs)` and reading `oob_score_` as the tree
count grows. Permutation importance (MDA) is run on the held-out test
set with `sklearn.inspection.permutation_importance(...,
scoring='accuracy', n_jobs=n_jobs)`.

**Parallelism / runtime knobs.** `n_jobs=-1` (default) uses every core
for both tree fitting and permutation importance. `mda_top_k` limits
MDA to the top-K features by MDI; the rest get `NaN` for
`mda_mean`/`mda_std`. Permutation importance scales as `n_features ×
n_repeats`, so on wide tables (>>1k features) this is the dominant
cost — `mda_top_k` is the right knob. When set, MDA is computed with
an explicit per-feature shuffle loop (no `permutation_importance`
joblib dispatch overhead), and the result table sort key falls back to
`mdi_importance` for the un-permuted features so they still rank
sensibly.

**`RFResult`**:
- `classifier` (fitted `RandomForestClassifier`)
- `classes_` (np.ndarray)
- `accuracy` (float — held-out test set)
- `classification_report` (str)
- `feature_importance` — DataFrame with columns `feature`,
  `mdi_importance`, `mda_mean`, `mda_std`, sorted by `mda_mean` desc
- `oob_curve` — DataFrame with columns `n_estimators`, `oob_score`

---

## Validation contract (all functions)

Each function validates inputs at the top and raises `ValueError` on
failure with the offending value formatted via `{x!r}`:

- NaN in `feature_table` → reject (impute first).
- Required metadata column missing → reject with the list of available
  columns.
- Sample names in `feature_table.columns` that are absent from
  `metadata.index` → reject with the missing list.
- Group/class with <2 samples, or <2 distinct levels where the method
  needs them, → reject.
- Out-of-range parameters (`n_components < 1`, `test_size ∉ (0, 1)`,
  unsupported metric/linkage, etc.) → reject.

No `print`, no `logging`, no file I/O.

---

## Dependencies

Top-level (already required): `numpy`, `pandas`, `scipy`, `scikit-learn`.

Optional extra: `pcoa = ["scikit-bio>=0.5"]`. Install with
`pip install 'metabproject[pcoa]'`. Without the extra, `run_pcoa` raises
`ImportError` on call; PCA, HCA, and Random Forest work unaffected.

---

## Tests

Under `tests/test_stats/test_multivariate/`. Shared local `conftest.py`
provides a seeded 50 × 20 synthetic feature table with planted 2-group
structure (features 1–10 elevated in samples 1–10, features 11–20
elevated in samples 11–20, 30 noise features). Each module's tests check:

- Behaviour on planted structure (PC1 sign split, PERMANOVA p < 0.05,
  cluster purity ≥ 0.9, RF accuracy > 0.8).
- Output shape and column contract.
- Validation paths (`pytest.raises(ValueError, match=...)` for every
  rejected input class).
- Reproducibility under a fixed seed (Random Forest).

PCoA tests use `pytest.importorskip("skbio")` so they skip cleanly on
installs without the `pcoa` extra.

---

## Out of scope (deliberate)

- **CLI wiring** (`metab multivariate ...`) — separate change.
- **Plotting helpers** under `viz/` (PCA biplot, PCoA scatter, dendrogram,
  RF importance bar, OOB curve) — separate change.
- Any rework of `stats/univariate.py` or a unified `Comparison`/`Test`
  abstraction — separate planned refactor.

---

## v1.1 — `MetabProject` wrappers

Pure library functions remain unchanged. Each is now also exposed as a
`MetabProject` method that writes a manifest entry on completion:

- `project.run_pca(n_components=10, scale=True) -> PCAResult`
- `project.run_pcoa(group_col, metric='euclidean', permutations=999, seed=0) -> PCoAResult`
- `project.run_hca(metric='euclidean', linkage_method='complete', n_clusters=None) -> HCAResult`
- `project.run_random_forest(target_col, n_estimators=500, test_size=0.25, n_permutation_repeats=10, mda_top_k=None, seed=123, n_jobs=-1) -> RFResult`

And the matching plot wrappers (saved under `<project>/viz/`):
`project.viz_rf_importance(rf_result, top_n=30)`,
`project.viz_rf_oob_curve(rf_result)`,
`project.viz_hca_dendrogram(hca_result, label_col=None)`.

Diagnostics (`stats/diagnostics.py`) are also wrapped:
`project.run_shapiro(group_col=None, alpha=0.05)` and
`project.run_levene(group_col, center='median', alpha=0.05)`.

A 2-way ANOVA + Tukey workflow is now a one-call API:
`project.run_comparison(Comparison(..., kind='anova_2way', ...),
export_buckets=True, export_pairs=True, count_matrix=True,
plot_features=False)` — see `specs/comparisons.md` for details.

CLI wiring is still out of scope.
