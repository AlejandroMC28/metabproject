"""Shared pytest fixtures: synthetic MZmine feature table + metadata.

The synthetic dataset is 50 features × 6 samples (2 groups × 3 reps) in the real
MZmine column layout, with known zeros for imputation testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_FEATURES = 50
GROUPS = {"control": 3, "fermented": 3}
SEED = 0


def _sample_files() -> list[str]:
    """Synthetic MZmine datafile filenames, one per sample."""
    files = []
    idx = 46
    for label, n in (("CSF1", GROUPS["control"]), ("CSF2", GROUPS["fermented"])):
        for _ in range(n):
            files.append(f"Mots_18102024_Foodomics_{label}_S{idx}.mzML")
            idx += 1
    return files


@pytest.fixture
def metadata_df() -> pd.DataFrame:
    """Sample metadata: sample_file, sample_name, group, weight, variety.

    ``variety`` alternates per replicate so each group has both varieties,
    enabling factorial contrast tests (e.g. variety×treatment).
    """
    files = _sample_files()
    rows = []
    for i, f in enumerate(files):
        group = "control" if i < GROUPS["control"] else "fermented"
        rep = (i % 3) + 1
        variety = "varA" if rep <= 2 else "varB"
        rows.append(
            {
                "sample_file": f,
                "sample_name": f"{group}_rep{rep}",
                "group": group,
                "weight": 1.50 if group == "control" else 2.06,
                "variety": variety,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def feature_table_df() -> pd.DataFrame:
    """Synthetic MZmine full feature table with all three column zones.

    Contains Zone 1 consensus columns (``id``, ``mz``, ``rt`` …), Zone 2
    ``datafile:*:area`` columns, and a Zone 3 ``datafile:*:feature_state``
    column. Some areas are zero to exercise the prevalence filter and imputation.
    """
    rng = np.random.default_rng(SEED)
    files = _sample_files()

    consensus = pd.DataFrame(
        {
            "id": np.arange(1, N_FEATURES + 1),
            "mz": rng.uniform(100, 1000, N_FEATURES),
            "rt": rng.uniform(0.5, 20, N_FEATURES),
            "charge": 1,
        }
    )

    areas = rng.lognormal(mean=12, sigma=1.5, size=(N_FEATURES, len(files)))
    # Punch in known zeros (missing detections) for imputation tests.
    zero_mask = rng.random((N_FEATURES, len(files))) < 0.1
    areas[zero_mask] = 0.0

    cols: dict[str, np.ndarray] = {}
    for j, f in enumerate(files):
        cols[f"datafile:{f}:area"] = areas[:, j]
        state = np.where(areas[:, j] > 0, "DETECTED", "MISSING")
        cols[f"datafile:{f}:feature_state"] = state

    return pd.concat([consensus, pd.DataFrame(cols)], axis=1)


@pytest.fixture
def feature_table_csv(tmp_path: Path, feature_table_df: pd.DataFrame) -> Path:
    """Write the synthetic feature table to a CSV and return its path."""
    path = tmp_path / "feature_table.csv"
    feature_table_df.to_csv(path, index=False)
    return path


@pytest.fixture
def metadata_csv(tmp_path: Path, metadata_df: pd.DataFrame) -> Path:
    """Write the synthetic metadata to a CSV and return its path."""
    path = tmp_path / "metadata.csv"
    metadata_df.to_csv(path, index=False)
    return path
