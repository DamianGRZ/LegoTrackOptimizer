"""Visualization package.

Re-exports the public API from the submodules so callsites using
``from src.visualization import plot_layout`` keep working.
"""

from src.visualization.track_renderer import (
    get_piece_color,
    get_piece_short_name,
    plot_layout,
    plot_multi_path_layout,
)
from src.visualization.pareto_plot import plot_pareto_front

__all__ = [
    "get_piece_color",
    "get_piece_short_name",
    "plot_layout",
    "plot_multi_path_layout",
    "plot_pareto_front",
]
