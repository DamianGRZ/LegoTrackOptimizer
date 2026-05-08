"""Visualization package.

Re-exports the public API from the submodules so callsites using
``from src.visualization import plot_layout`` keep working.
"""

from .track_renderer import (
    get_piece_color,
    get_piece_short_name,
    plot_layout,
    plot_multi_path_layout,
)
from .pareto_plot import plot_pareto_front
from .serialize import catalog_to_json, port_graph_to_json
from .showcase import build_showcase_layout
from .topology_showcase import build_topology_showcase

__all__ = [
    "get_piece_color",
    "get_piece_short_name",
    "plot_layout",
    "plot_multi_path_layout",
    "plot_pareto_front",
    "catalog_to_json",
    "port_graph_to_json",
    "build_showcase_layout",
    "build_topology_showcase",
]
