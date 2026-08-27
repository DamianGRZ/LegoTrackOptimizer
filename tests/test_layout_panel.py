"""Tests for the unified single-figure renderer (`plot_layout`).

The info panel and legend must report the SAME numbers as run_info.md
(`count_pieces`), list every catalog type including unused ones, mark one
square per PHYSICAL crossing (records, not slots), and gate the FK-drift
overlay by the optimizer's own closure tolerances.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.catalog import TrackCatalog  # noqa: E402
from src.config import OptimizationConfig  # noqa: E402
from src.decoder import decode_chromosome  # noqa: E402
from src.encoding import (  # noqa: E402
    R40_CURVE,
    compute_dimensions,
    create_chromosome_from_pieces,
)
from src.intersection import _cross_midpoint  # noqa: E402
from src.run_info import count_pieces  # noqa: E402
from src.sampling import _gen_figure_eight_cross  # noqa: E402
from src.types import MultiPathLayout, TraversalPath  # noqa: E402
from src.visualization.track_renderer import (  # noqa: E402
    _cross_marker_points,
    _drift_segments,
    plot_layout,
)


@pytest.fixture(scope="module")
def catalog():
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture(scope="module")
def config():
    return OptimizationConfig.load("configs/all_pieces.yaml")


def _circle_layout(catalog, config):
    """Minimal closed loop: 16 R40 curves, one path, no S24/switches/crossings."""
    dims = compute_dimensions(config, catalog)
    x = create_chromosome_from_pieces(dims, [int(R40_CURVE)] * 16)
    return decode_chromosome(x, catalog, config.inventory, dims=dims)


def _render(layout, catalog, config, **kwargs):
    return plot_layout(
        layout, catalog, config.boundary, "test",
        closure_tolerance=config.closure_tolerance,
        angle_tolerance=config.angle_tolerance,
        **kwargs,
    )


def test_legend_lists_every_catalog_type_including_zeros(catalog, config):
    layout = _circle_layout(catalog, config)
    fig = _render(layout, catalog, config, inventory=config.inventory)
    legend = fig.axes[1].get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    plt.close(fig)

    counts = count_pieces(layout)
    for piece_idx, piece_id in catalog.index_to_id.items():
        used = counts.get(piece_idx, 0)
        cap = config.inventory[piece_id]
        expected = [piece_id, f"{used}/{cap}"]
        assert any(label.split() == expected for label in labels), (
            f"missing legend row {expected}, got {labels}"
        )
    # A pure-curve circle uses no STRAIGHT_24: the zero row must still show.
    assert any(label.split() == ["STRAIGHT_24", "0/8"] for label in labels)


def test_empty_layout_renders_and_saves(catalog, config, tmp_path):
    """A degenerate (0-path) layout still yields a titled figure and writes its
    PNG: the early return carries its own save path."""
    out = tmp_path / "empty.png"
    fig = plot_layout(
        MultiPathLayout(), catalog, config.boundary, "empty case", out,
        closure_tolerance=config.closure_tolerance,
        angle_tolerance=config.angle_tolerance,
    )
    texts = [t.get_text() for t in fig.axes[0].texts]
    plt.close(fig)

    assert "Empty Layout" in texts
    assert out.exists()


def test_panel_pieces_row_matches_count_pieces(catalog, config):
    layout = _circle_layout(catalog, config)
    fig = _render(layout, catalog, config, inventory=config.inventory)
    panel_lines = [
        line for text in fig.axes[1].texts for line in text.get_text().splitlines()
    ]
    plt.close(fig)

    used = sum(count_pieces(layout).values())
    total = sum(config.inventory.values())
    assert used == layout.n_physical_pieces
    assert f"Pieces: {used}/{total}" in panel_lines


def test_one_physical_cross_yields_one_marker(catalog, config):
    dims = compute_dimensions(config, catalog)
    inv = catalog.inventory_by_index(config.inventory)
    variants = _gen_figure_eight_cross(inv, dims)
    assert variants, "expected the figure-8 cross seed for all_pieces"
    pieces, flips, junctions, cross_junctions, dcs = variants[0]
    x = create_chromosome_from_pieces(
        dims, pieces, main_loop_flips=flips,
        junctions=junctions, cross_junctions=cross_junctions,
        double_crossovers=dcs,
    )
    layout = decode_chromosome(x, catalog, config.inventory, dims=dims)

    assert layout.n_cross_pieces == 1
    points = _cross_marker_points(layout)
    assert len(points) == 1  # one square per physical piece

    # The marker must sit on the DRAWN crossing: the validator's midpoint of
    # the auto-centered slot state, never the pre-centering record frame.
    entry_pos = layout.cross_junctions[0].positions[0]
    exp_x, exp_y, _ = _cross_midpoint(layout.paths[0].states[entry_pos])
    (marker_x, marker_y), = points
    assert (round(marker_x, 1), round(marker_y, 1)) == (round(exp_x, 1), round(exp_y, 1))


def _siding_path(closure_error):
    """Synthetic branched path: divergent range ends at slot 1, drift after it."""
    states = np.array(
        [[0.0, 0.0, 0.0], [16.0, 0.0, 0.0], [32.0, 0.0, 0.0], [48.0, 0.0, 0.0]]
    )
    return TraversalPath(
        path_id=1, route_choices=(1,), piece_sequence=[4, 0, 5], states=states,
        closure_error=closure_error, angle_error=0.0, divergent_ranges={0: (0, 1)},
    )


def _drift_line_count(fig):
    return sum(
        1 for line in fig.axes[0].lines
        if line.get_linestyle() == ":" and line.get_color() == "#888888"
    )


@pytest.mark.parametrize("closure_error,expected_segments", [(0.5, 0), (10.0, 1)])
def test_drift_gated_by_config_tolerances(catalog, config,
                                          closure_error, expected_segments):
    layout = MultiPathLayout(
        main_loop_pieces=[4, 0, 5], paths=[_siding_path(closure_error)],
    )
    segments = _drift_segments(
        layout, config.closure_tolerance, config.angle_tolerance,
    )
    assert len(segments) == expected_segments

    fig = _render(layout, catalog, config)
    n_drawn = _drift_line_count(fig)
    plt.close(fig)
    assert n_drawn == expected_segments
