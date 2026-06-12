"""Pareto plot: duplicated F points must be visible as multiplicity labels.

A converged terminal population holds hundreds of coincident F vectors;
plotting them as overlapping markers reads as 'two lonely dots'. Distinct
points get one marker each, annotated with their multiplicity.
"""

import numpy as np

from src.visualization.pareto_plot import plot_pareto_front


def test_duplicate_points_annotated_with_multiplicity(tmp_path):
    F = np.array([[-0.7, -1.0]] * 5 + [[-0.5, -0.9]])
    G = np.array([[-1.0]] * 6)
    fig = plot_pareto_front(F, G, save_path=tmp_path / "p.png")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "×5" in texts, texts
    assert (tmp_path / "p.png").exists()


def test_archive_front_drawn_as_run_level_front(tmp_path):
    """The run-cumulative feasible front (monitor archive) must be drawn as
    the Pareto front — the final population alone is a converged monoculture
    that hides every mid-run trade-off solution."""
    F = np.array([[-0.7, -1.0]] * 3)
    G = np.array([[-1.0]] * 3)
    archive = np.array([[-0.7, -1.0], [-0.5, -1.02], [-0.72, -0.9]])
    fig = plot_pareto_front(F, G, archive_F=archive, save_path=tmp_path / "p.png")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("Run Pareto front" in label for label in labels), labels
