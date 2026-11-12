"""Random Forest classification with OOB curve and permutation importance.

A :class:`~sklearn.ensemble.RandomForestClassifier` is fit on the
processed feature table with class weights computed from the *training*
labels via ``compute_class_weight('balanced', ...)``. The OOB curve is
recorded by warm-starting the estimator and reading ``oob_score_`` at each
intermediate tree count. Feature importance combines two perspectives:
Mean Decrease in Impurity (MDI) from the fitted forest and Mean Decrease
in Accuracy (MDA) from
:func:`sklearn.inspection.permutation_importance` on the held-out test
set. Outputs are sorted by ``mda_mean`` descending so the most informative
features lead the table.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def _permute_one_feature(
    col: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf: RandomForestClassifier,
    baseline: float,
    n_repeats: int,
    seed: int,
) -> tuple[float, float]:
    """Return ``(mean_drop, std_drop)`` for permuting one feature column."""
    rng = np.random.default_rng(seed)
    X_perm = X_test.copy()
    original = X_perm[:, col].copy()
    drops = np.empty(n_repeats)
    for r in range(n_repeats):
        rng.shuffle(X_perm[:, col])
        drops[r] = baseline - clf.score(X_perm, y_test)
        X_perm[:, col] = original
    return float(drops.mean()), float(drops.std())


@dataclass
class RFResult:
    """Result of a Random Forest fit.

    Attributes:
        classifier: The fitted :class:`RandomForestClassifier`.
        classes_: Class labels in the order used by ``classifier``.
        accuracy: Accuracy on the held-out test split.
        classification_report: Per-class precision/recall/F1 text report
            from :func:`sklearn.metrics.classification_report`.
        feature_importance: DataFrame with columns ``feature``,
            ``mdi_importance``, ``mda_mean``, ``mda_std`` sorted by
            ``mda_mean`` descending.
        oob_curve: DataFrame with columns ``n_estimators`` and
            ``oob_score`` recorded as the forest grows.
    """

    classifier: RandomForestClassifier
    classes_: np.ndarray
    accuracy: float
    classification_report: str
    feature_importance: pd.DataFrame
    oob_curve: pd.DataFrame


def run_random_forest(
    feature_table: pd.DataFrame,
    metadata: pd.DataFrame,
    target_col: str,
    subset: dict[str, Any] | None = None,
    n_estimators: int = 500,
    test_size: float = 0.25,
    n_permutation_repeats: int = 10,
    mda_top_k: int | None = None,
    seed: int = 123,
    n_jobs: int | None = -1,
) -> RFResult:
    """Fit a Random Forest classifier with OOB curve and MDA importance.

    Args:
        feature_table: Processed feature table (features × samples). The
            table is transposed internally so samples become rows. Must
            contain no NaN values.
        metadata: Sample metadata indexed by sample name. Must contain
            ``target_col`` and cover every column of ``feature_table``.
        target_col: Column in ``metadata`` holding the class label.
        subset: Optional ``{column: value}`` (or ``{column: [value, ...]}``)
            mapping used to restrict the analysis to a subset of samples
            *before* training. Useful for controlling for a condition — for
            example, ``subset={"variety": "red"}`` predicts ``target_col``
            within a single variety only. All keys must exist in
            ``metadata.columns``.
        n_estimators: Final number of trees. The OOB curve grows the
            forest from a small base to this value.
        test_size: Fraction of samples held out for the test set and for
            permutation importance. Passed to
            :func:`sklearn.model_selection.train_test_split` (stratified).
        n_permutation_repeats: ``n_repeats`` for
            :func:`sklearn.inspection.permutation_importance`.
        mda_top_k: If set, compute MDA only on the top ``mda_top_k``
            features by MDI; the rest get ``NaN`` for ``mda_mean`` /
            ``mda_std``. Permutation importance scales with
            ``n_features × n_repeats`` and dominates the runtime on wide
            tables (>>1k features), so restricting it here is the main
            knob for keeping the function snappy. ``None`` (default)
            permutes every feature.
        seed: Seed shared between the split, the forest, and the
            permutation test for reproducibility.
        n_jobs: Parallelism for tree fitting and permutation importance.
            ``-1`` uses all available cores (default), ``None`` or ``1``
            runs single-threaded.

    Returns:
        :class:`RFResult` with the fitted classifier, held-out accuracy,
        classification report, combined MDI/MDA importance table, and the
        OOB curve.

    Raises:
        ValueError: If ``target_col`` is missing, the feature table
            contains NaNs, samples cannot be aligned with ``metadata``,
            fewer than 2 classes are present, any class has <2 samples, or
            ``n_estimators`` / ``test_size`` are out of range.
    """
    if n_estimators < 2:
        raise ValueError(f"n_estimators must be >= 2 (got {n_estimators!r})")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1) (got {test_size!r})")
    if n_permutation_repeats < 1:
        raise ValueError(
            f"n_permutation_repeats must be >= 1 (got {n_permutation_repeats!r})"
        )

    nan_features = int(feature_table.isna().any(axis=1).sum())
    if nan_features:
        raise ValueError(
            f"feature_table contains NaN values in {nan_features} feature(s); "
            "impute before running Random Forest"
        )

    if target_col not in metadata.columns:
        raise ValueError(
            f"metadata has no column {target_col!r} "
            f"(available: {metadata.columns.tolist()})"
        )

    if subset:
        bad = [c for c in subset if c not in metadata.columns]
        if bad:
            raise ValueError(
                f"subset references unknown metadata column(s) {bad} "
                f"(available: {metadata.columns.tolist()})"
            )
        mask = pd.Series(True, index=metadata.index)
        for col, val in subset.items():
            values = list(val) if isinstance(val, (list, tuple, set)) else [val]
            mask &= metadata[col].isin(values)
        kept = metadata.index[mask]
        if len(kept) == 0:
            raise ValueError(f"subset {subset!r} removes every sample")
        feature_table = feature_table.loc[:, feature_table.columns.intersection(kept)]
        metadata = metadata.loc[feature_table.columns]

    sample_names = list(feature_table.columns)
    missing = [s for s in sample_names if s not in metadata.index]
    if missing:
        raise ValueError(f"Sample(s) absent from metadata index: {missing}")

    y = metadata.loc[sample_names, target_col].to_numpy()
    counts = pd.Series(y).value_counts()
    if counts.size < 2:
        raise ValueError(
            f"target_col {target_col!r} must have >=2 distinct classes "
            f"(got {counts.index.tolist()})"
        )
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(
            f"target_col {target_col!r} classes with <2 samples: "
            f"{too_small.to_dict()}"
        )

    feature_names = list(feature_table.index)
    X = feature_table.to_numpy(dtype=float).T

    n_samples = len(y)
    n_classes = int(counts.size)
    n_test = int(np.ceil(test_size * n_samples))
    n_train = n_samples - n_test
    if n_test < n_classes or n_train < n_classes:
        min_frac = n_classes / n_samples
        raise ValueError(
            f"test_size={test_size!r} yields a stratified split of "
            f"{n_train} train / {n_test} test from {n_samples} samples, but "
            f"both sides must hold at least one of each of the {n_classes} "
            f"classes. Increase test_size to at least "
            f"{min_frac:.3f} (and at most {1 - min_frac:.3f}), reduce the "
            f"number of classes, or collect more samples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train_classes = np.unique(y_train)
    class_weight_values = compute_class_weight(
        "balanced", classes=train_classes, y=y_train
    )
    class_weight = {
        cls: float(w) for cls, w in zip(train_classes, class_weight_values)
    }

    # Step over n_estimators to record an OOB curve. We use warm_start to grow
    # the same forest rather than refitting from scratch each step.
    step = max(10, n_estimators // 20)
    checkpoints = list(range(step, n_estimators + 1, step))
    if checkpoints[-1] != n_estimators:
        checkpoints.append(n_estimators)

    clf = RandomForestClassifier(
        n_estimators=checkpoints[0],
        class_weight=class_weight,
        oob_score=True,
        warm_start=True,
        bootstrap=True,
        random_state=seed,
        n_jobs=n_jobs,
    )
    oob_rows: list[dict] = []
    for n in checkpoints:
        clf.n_estimators = n
        clf.fit(X_train, y_train)
        oob_rows.append({"n_estimators": n, "oob_score": float(clf.oob_score_)})
    oob_curve = pd.DataFrame(oob_rows)

    y_pred = clf.predict(X_test)
    accuracy = float((y_pred == y_test).mean())
    report = classification_report(y_test, y_pred, zero_division=0)

    if mda_top_k is not None and mda_top_k < len(feature_names):
        perm_idx = np.argsort(clf.feature_importances_)[::-1][:mda_top_k]
    else:
        perm_idx = np.arange(len(feature_names))

    with warnings.catch_warnings():
        # sklearn emits a noisy `sklearn.utils.parallel.delayed` UserWarning
        # from inside permutation_importance when n_jobs is set; the result
        # is correct and parallel, so we silence the cosmetic warning here.
        warnings.filterwarnings(
            "ignore",
            message=r".*sklearn\.utils\.parallel\.delayed.*",
            category=UserWarning,
        )
        if perm_idx.size == len(feature_names):
            perm = permutation_importance(
                clf, X_test, y_test,
                n_repeats=n_permutation_repeats, random_state=seed,
                scoring="accuracy", n_jobs=n_jobs,
            )
            mda_mean = perm.importances_mean
            mda_std = perm.importances_std
        else:
            mda_mean = np.full(len(feature_names), np.nan)
            mda_std = np.full(len(feature_names), np.nan)
            baseline = float(clf.score(X_test, y_test))
            # deepcopy + n_jobs=1 on the worker classifier so loky workers
            # don't recursively fan out their own thread pools.
            clf_worker = copy.deepcopy(clf)
            clf_worker.n_jobs = 1
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_permute_one_feature)(
                    int(col), X_test, y_test, clf_worker, baseline,
                    n_permutation_repeats, seed + j,
                )
                for j, col in enumerate(perm_idx)
            )
            for j, (mean_drop, std_drop) in enumerate(results):
                mda_mean[perm_idx[j]] = mean_drop
                mda_std[perm_idx[j]] = std_drop

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mdi_importance": clf.feature_importances_,
            "mda_mean": mda_mean,
            "mda_std": mda_std,
        }
    ).sort_values(
        ["mda_mean", "mdi_importance"], ascending=False, kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return RFResult(
        classifier=clf,
        classes_=clf.classes_,
        accuracy=accuracy,
        classification_report=report,
        feature_importance=feature_importance,
        oob_curve=oob_curve,
    )
