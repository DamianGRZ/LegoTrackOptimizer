"""Render-geometry tests for the DOUBLE_CROSSOVER body.

A physical DOUBLE_CROSSOVER is ONE piece, but a closed loop threads it twice
(a both-cross or both-through cover of its four ports), so piece index 6 shows
up at two non-adjacent slots in a path's piece_sequence. The renderer must draw
the body exactly ONCE, anchored at port A, regardless of which port each
traversal entered through -- drawing it per-occurrence paints it twice and, for
the traversal that enters at port C, 16 studs off to the wrong side.

These tests pin the pure pose-recovery helper (`_dc_body_poses`) so the visual
bug can't silently regress.
"""
import numpy as np

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import compute_dimensions, create_chromosome_from_pieces
from src.sampling import _gen_figure_eight_dbl_crossover
from src.visualization.track_renderer import _dc_body_poses, DC_LATERAL_STUDS


def _figure8_dc_layout():
    cat = TrackCatalog.load("data/track_pieces_v2.yaml")
    cfg = OptimizationConfig.load("configs/all_pieces.yaml")
    dims = compute_dimensions(cfg, cat)
    inv = {cat._id_to_index[k]: v for k, v in cfg.inventory.items()
           if k in cat._id_to_index}
    variants = _gen_figure_eight_dbl_crossover(inv, dims)
    assert variants, "expected the figure-8 DC seed for all_pieces"
    pieces, flips, junc, crossj, dcs = variants[0]
    x = create_chromosome_from_pieces(
        dims, pieces, main_loop_flips=flips,
        junctions=junc, cross_junctions=crossj, double_crossovers=dcs,
    )
    layout = decode_chromosome(x, cat, cfg.inventory, dims=dims)
    assert layout.n_dbl_crossovers == 1
    return layout


def test_dc_body_drawn_once_at_port_a():
    """One physical DC -> exactly one body, anchored at the lower-left port A,
    even though the figure-8 path traverses it twice (ports A and C)."""
    path = _figure8_dc_layout().paths[0]
    assert sum(1 for p in path.piece_sequence if p == 6) == 2  # threaded twice

    poses = _dc_body_poses(path.piece_sequence, path.states)

    assert len(poses) == 1, "DC body must be drawn once, not once per traversal"
    x0, y0, th = poses[0]
    assert (round(x0, 1), round(y0, 1)) == (-24.0, -8.0)
    assert round(th % 360, 1) in (0.0, 360.0)


def test_dc_body_four_ports_match_lobe_straights():
    """Port-A pose places the four DC ports flush against the lobe straights at
    (+-24, +-8) -- i.e. the body straddles y in [-8, +8], not floating above."""
    path = _figure8_dc_layout().paths[0]
    (x0, y0, th), = _dc_body_poses(path.piece_sequence, path.states)

    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))

    def tp(lx, ly):
        return (round(x0 + lx * c - ly * s, 1), round(y0 + lx * s + ly * c, 1))

    ports = {tp(0, 0), tp(48, 0), tp(0, DC_LATERAL_STUDS), tp(48, DC_LATERAL_STUDS)}
    assert ports == {(-24.0, -8.0), (24.0, -8.0), (-24.0, 8.0), (24.0, 8.0)}


def test_plot_layout_draws_dc_body_once(monkeypatch):
    """Switchless DC layouts (0 switch pairs) render through `plot_layout`, not
    `plot_multi_path_layout` (runner._render_one / save_results dispatch on
    n_switch_pairs). plot_layout must also draw each physical DC exactly once,
    anchored at port A -- else snapshots/best_layout keep the double-draw bug."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import src.visualization.track_renderer as tr

    layout = _figure8_dc_layout()
    assert layout.n_switch_pairs == 0  # this layout takes the plot_layout branch
    cat = TrackCatalog.load("data/track_pieces_v2.yaml")

    calls = []
    real = tr._draw_double_crossover_piece

    def spy(ax, x0, y0, th, color, *a, **k):
        calls.append((round(float(x0), 1), round(float(y0), 1)))
        return real(ax, x0, y0, th, color, *a, **k)

    monkeypatch.setattr(tr, "_draw_double_crossover_piece", spy)
    fig = tr.plot_layout(layout, cat, boundary=None, title="t")
    plt.close(fig)

    assert calls == [(-24.0, -8.0)], f"DC body drawn {len(calls)}x at {calls}"
