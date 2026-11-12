"""Read, write, and append to the project manifest (``metabproject.json``).

The manifest is an append-only JSON log: every command records its name,
timestamp, parameters, and result counts. It serves as the audit trail and as
the source of truth for project state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class _ManifestJSONEncoder(json.JSONEncoder):
    """JSON encoder that coerces numpy scalars and Paths to native types."""

    def default(self, o: Any) -> Any:
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)

_MANIFEST_NAME = "metabproject.json"


def create_manifest(
    path: str | Path,
    project_name: str,
    init_entry: dict[str, Any],
) -> dict[str, Any]:
    """Initialize a new manifest with a single ``init`` operation entry.

    Args:
        path: Project directory in which to write ``metabproject.json``.
        project_name: Human-readable project name.
        init_entry: The operation entry for the ``init`` command.

    Returns:
        The manifest dictionary that was written.
    """
    manifest: dict[str, Any] = {
        "project_name": project_name,
        "created": init_entry["timestamp"],
        "operations": [init_entry],
    }
    _write(Path(path), manifest)
    return manifest


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Load the manifest from a project directory.

    Args:
        path: Project directory containing ``metabproject.json``.

    Returns:
        The parsed manifest dictionary.

    Raises:
        FileNotFoundError: If no manifest exists (not a metabproject directory).
    """
    manifest_path = Path(path) / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Not a metabproject directory: {path}\n"
            f"  (Expected {manifest_path} to exist)"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def append_operation(path: str | Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append an operation entry to the manifest and persist it.

    Args:
        path: Project directory containing ``metabproject.json``.
        entry: Operation entry with ``command``, ``timestamp``, ``parameters``,
            and ``results`` keys.

    Returns:
        The updated manifest dictionary.
    """
    manifest = read_manifest(path)
    manifest["operations"].append(entry)
    _write(Path(path), manifest)
    return manifest


def make_operation_entry(
    command: str,
    parameters: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Build an operation entry with an ISO 8601 timestamp.

    Args:
        command: Command name (e.g. ``"init"``, ``"process"``).
        parameters: Parameters the command was invoked with.
        results: Result summary (counts, files created).

    Returns:
        A manifest operation entry.
    """
    return {
        "command": command,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "parameters": parameters,
        "results": results,
    }


def _write(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest dict to disk as pretty-printed JSON."""
    (path / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, cls=_ManifestJSONEncoder),
        encoding="utf-8",
    )
