"""Interactive HTML viewer for browsing annotated metabolomics features.

Public API::

    from metabproject.explore import assemble_explore_data, generate_explore_html
"""

from metabproject.explore.data import assemble_explore_data
from metabproject.explore.report import generate_explore_html

__all__ = ["assemble_explore_data", "generate_explore_html"]
