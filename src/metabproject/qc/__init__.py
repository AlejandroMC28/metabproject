"""Feature-table QC checks (alignment duplicates, overview profile)."""

from metabproject.qc.duplicates import detect_duplicates
from metabproject.qc.overview import compute_overview

__all__ = ["detect_duplicates", "compute_overview"]
