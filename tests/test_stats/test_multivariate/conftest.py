"""Shared fixtures for multivariate tests.

The fixture builds a 50 × 20 feature table with a planted 2-group split:
features 1–10 are elevated in samples 1–10 (group A) and features 11–20 are
elevated in samples 11–20 (group B); the remaining 30 features are noise.
All four multivariate methods (PCA, PCoA, HCA, RF) should recover that
split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_FEATURES = 50
N_PER_GROUP = 10
SEED = 0


def _build_planted_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    n_samples = 2 * N_PER_GROUP

    sample_names = [f"a{i + 1}" for i in range(N_PER_GROUP)] + [
        f"b{i + 1}" for i in range(N_PER_GROUP)
    ]
    feature_names = [f"f{i + 1}" for i in range(N_FEATURES)]

    # Background noise on a log-like scale.
    data = rng.normal(loc=10.0, scale=0.3, size=(N_FEATURES, n_samples))

    # Plant discriminative signal in features 1-10 (up in A) and 11-20 (up in B).
    data[0:10, 0:N_PER_GROUP] += 4.0
    data[10:20, N_PER_GROUP:] += 4.0

    ft = pd.DataFrame(
        data,
        index=pd.Index(feature_names, name="feature_id"),
        columns=sample_names,
    )

    metadata = pd.DataFrame(
        {"group": ["A"] * N_PER_GROUP + ["B"] * N_PER_GROUP},
        index=pd.Index(sample_names, name="sample_name"),
    )

    return ft, metadata


@pytest.fixture
def planted_feature_table() -> pd.DataFrame:
    ft, _ = _build_planted_table()
    return ft


@pytest.fixture
def planted_metadata() -> pd.DataFrame:
    _, md = _build_planted_table()
    return md


@pytest.fixture
def planted_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _build_planted_table()
