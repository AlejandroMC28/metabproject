"""Multivariate analyses on the processed feature table.

Sibling submodule to :mod:`metabproject.stats.univariate`; provides PCA,
PCoA (with PERMANOVA + PERMDISP), hierarchical clustering, and Random
Forest classification. Each routine is a pure function returning a
dataclass — no I/O, no plotting, no CLI awareness.
"""

from metabproject.stats.multivariate.hca import HCAResult, run_hca
from metabproject.stats.multivariate.pca import PCAResult, run_pca
from metabproject.stats.multivariate.pcoa import PCoAResult, run_pcoa
from metabproject.stats.multivariate.random_forest import RFResult, run_random_forest

__all__ = [
    "HCAResult",
    "PCAResult",
    "PCoAResult",
    "RFResult",
    "run_hca",
    "run_pca",
    "run_pcoa",
    "run_random_forest",
]
