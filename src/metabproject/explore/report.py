"""Generate the self-contained interactive HTML viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_explore_html(
    data: dict[str, Any],
    output_path: str | Path,
    embed_plotly: bool = True,
) -> Path:
    """Render the Jinja2 viewer template with embedded JSON data.

    Args:
        data: Output of :func:`~metabproject.explore.data.assemble_explore_data`.
        output_path: Where to write the HTML file.
        embed_plotly: If ``True`` (default), embed plotly.js inline (~3.5 MB)
            for a fully self-contained file. If ``False``, use a CDN link
            (smaller file but requires internet access).

    Returns:
        Path to the written HTML file.
    """
    try:
        import jinja2
    except ImportError as exc:
        raise ImportError(
            "jinja2 is required for the HTML report. "
            "Install with: pip install jinja2"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    templates_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        autoescape=False,  # data_json is already safe JSON
    )
    template = env.get_template("viewer.html")

    plotly_js = ""
    if embed_plotly:
        try:
            import plotly.offline as po

            plotly_js = po.get_plotlyjs()
        except ImportError as exc:
            raise ImportError(
                "plotly is required for the HTML viewer. "
                "Install with: pip install plotly"
            ) from exc

    data_json = json.dumps(data, default=_json_default, ensure_ascii=False)

    html = template.render(
        project_name=data.get("project_name", ""),
        data_json=data_json,
        plotly_js=plotly_js,
        embed_plotly=embed_plotly,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for JSON types that aren't natively serialisable."""
    import math

    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
