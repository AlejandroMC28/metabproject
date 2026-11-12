"""Load and clean GNPS library and suspect-list annotations."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd


def _inchikey2d_from_smiles(smiles: pd.Series) -> pd.Series:
    """First InChIKey block for each SMILES (None where unparsable).

    Newer GNPS2 FBMN outputs no longer carry ``InChIKey-Planar``; this fills it
    so GNPS hits can still be matched against SIRIUS. Needs RDKit.
    """
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        warnings.warn("RDKit not installed — GNPS InChIKey2D cannot be derived from SMILES.",
                      UserWarning, stacklevel=3)
        return pd.Series(None, index=smiles.index, dtype=object)
    RDLogger.DisableLog("rdApp.*")

    def key(smi):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None
        return Chem.MolToInchiKey(mol).split("-")[0] if mol is not None else None

    return smiles.map(key)


# Columns kept from GNPS library TSV (before prefixing).
_GNPS_LIBRARY_COLS = [
    "#Scan#",
    "Compound_Name",
    "Smiles",
    "INCHI",
    "InChIKey-Planar",
    "MQScore",
    "IonMode",
    "npclassifier_pathway",
    "npclassifier_superclass",
    "npclassifier_class",
    "CF_NPC_class",
]

# Map raw GNPS column names → clean names (before gnps_ prefix).
_GNPS_RENAME = {
    "#Scan#": "feature_id",
    "Compound_Name": "compound_name",
    "Smiles": "smiles",
    "INCHI": "InChI",
    "InChIKey-Planar": "InChIkey2D",
    "MQScore": "MQScore",
    "IonMode": "ion_mode",
    "npclassifier_pathway": "npc_pathway",
    "npclassifier_superclass": "npc_superclass",
    "npclassifier_class": "npc_class",
    "CF_NPC_class": "classyfire_class",
}

_SUSPECT_LIBRARY_NAME = "GNPS-SUSPECTLIST.mgf"

# Regex to parse suspect Compound_Name field.
# Example: "compound name | SIRIUS: C10H12O | BUDDY: C10H12O | Δmz: 0.002 | adduct: [M+H]+"
_SUSPECT_FORMULA_RE = re.compile(
    r"^(?P<compound_name>[^|]+?)"
    r"(?:\s*\|\s*SIRIUS:\s*(?P<formula_sirius>[A-Za-z0-9]+))?"
    r"(?:\s*\|\s*BUDDY:\s*(?P<formula_buddy>[A-Za-z0-9]+))?"
    r"(?:\s*\|\s*[Δδ]mz:\s*(?P<delta_mz>[-\d.]+))?"
    r"(?:\s*\|\s*adduct:\s*(?P<adduct>\S+))?"
    r"\s*$"
)


def load_gnps_library(
    path: str | Path,
    polarity: str | None = None,
) -> pd.DataFrame:
    """Load and clean GNPS spectral library matches.

    Reads a GNPS library annotation TSV (``merged_results_with_gnps.tsv``),
    selects relevant columns, renames them, prefixes with ``gnps_``, and
    returns a DataFrame indexed by ``feature_id``.

    Args:
        path: Path to the GNPS library TSV.
        polarity: If ``"positive"`` or ``"negative"``, drop library matches
            whose ``IonMode`` is the opposite polarity. Rows with missing or
            unrecognized ``IonMode`` are kept. ``None`` (default) keeps all
            rows regardless of IonMode.

    Returns:
        DataFrame indexed by ``feature_id`` with ``gnps_*`` columns.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If required columns are absent, or ``polarity`` is not one
            of ``None``, ``"positive"``, ``"negative"``.
    """
    if polarity is not None:
        polarity = polarity.lower()
        if polarity not in ("positive", "negative"):
            raise ValueError(
                f"polarity must be 'positive', 'negative', or None; got {polarity!r}"
            )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GNPS library file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    # Keep only available required columns — some GNPS jobs omit taxonomy.
    available = [c for c in _GNPS_LIBRARY_COLS if c in df.columns]
    missing_required = [c for c in ["#Scan#", "Compound_Name"] if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing_required}\n"
            f"  Found: {df.columns.tolist()}"
        )

    df = df[available].copy()
    df = df.rename(columns={k: v for k, v in _GNPS_RENAME.items() if k in df.columns})
    df["feature_id"] = df["feature_id"].astype(int)
    df = df.set_index("feature_id")

    if ("InChIkey2D" not in df.columns or df["InChIkey2D"].isna().all()) and "smiles" in df.columns:
        df["InChIkey2D"] = _inchikey2d_from_smiles(df["smiles"])

    # Polarity filter: drop only rows whose IonMode is the explicit opposite.
    if polarity is not None and "ion_mode" in df.columns:
        opposite = "negative" if polarity == "positive" else "positive"
        ion_mode_norm = df["ion_mode"].astype(str).str.strip().str.lower()
        df = df[ion_mode_norm != opposite]

    # Prefix all columns.
    df.columns = [f"gnps_{c}" for c in df.columns]

    return df


def load_gnps_suspect(
    path: str | Path,
    exclude_feature_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """Load and clean GNPS suspect-list annotations.

    Reads a suspect-list TSV, filters to ``LibraryName == GNPS-SUSPECTLIST.mgf``,
    removes features already matched in the library, parses the ``Compound_Name``
    field via regex, and returns a DataFrame indexed by ``feature_id`` with
    ``suspect_*`` columns.

    Args:
        path: Path to the suspect-list TSV.
        exclude_feature_ids: Feature IDs to exclude (typically from GNPS library
            to prevent double-annotation).

    Returns:
        DataFrame indexed by ``feature_id`` with ``suspect_*`` columns.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GNPS suspect file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    required = ["#Scan#", "Compound_Name", "LibraryName"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}\n"
            f"  Found: {df.columns.tolist()}"
        )

    # Filter to suspect-list entries only.
    df = df[df["LibraryName"] == _SUSPECT_LIBRARY_NAME].copy()

    df["feature_id"] = df["#Scan#"].astype(int)

    # Deduplicate against GNPS library matches.
    if exclude_feature_ids is not None and len(exclude_feature_ids) > 0:
        df = df[~df["feature_id"].isin(exclude_feature_ids)]

    df = df.set_index("feature_id")

    # Parse Compound_Name into structured fields.
    parsed = df["Compound_Name"].apply(_parse_suspect_name)
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # Prefix all parsed columns.
    parsed_df.columns = [f"suspect_{c}" for c in parsed_df.columns]

    return parsed_df


def _parse_suspect_name(name: str) -> dict:
    """Parse a GNPS suspect-list Compound_Name string into components.

    Args:
        name: Raw Compound_Name value from suspect-list TSV.

    Returns:
        Dict with keys: compound_name, formula_sirius, formula_buddy,
        delta_mz, adduct. Missing components are None.
    """
    m = _SUSPECT_FORMULA_RE.match(str(name))
    if m:
        return {
            "compound_name": m.group("compound_name").strip() or None,
            "formula_sirius": m.group("formula_sirius"),
            "formula_buddy": m.group("formula_buddy"),
            "delta_mz": float(m.group("delta_mz")) if m.group("delta_mz") else None,
            "adduct": m.group("adduct"),
        }
    return {
        "compound_name": str(name).strip() or None,
        "formula_sirius": None,
        "formula_buddy": None,
        "delta_mz": None,
        "adduct": None,
    }
