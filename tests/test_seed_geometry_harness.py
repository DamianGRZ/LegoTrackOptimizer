"""Reusable seed-geometry oracle: decode a Pattern and assert it is a feasible,
committed, boundary-fitting closed loop. Also used as the derivation harness for
new crossing seeds (run candidate geometries through `decode_seed` until it passes).

Two checks:
- `assert_valid_closed` — closure error, committed crossing counts, bounding box fit.
- `assert_seed_feasible` — the STRONG oracle: the seed must decode to a fully FEASIBLE
  layout (every constraint <= 0, which crucially includes the collision constraint G[4]),
  so an over-wound self-overlapping coil is rejected even though it "closes". Always use
  this for new seed families.
"""
import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import (
    CROSS_90, DOUBLE_CROSSOVER, PartitionedDimensions,
    compute_dimensions, create_chromosome_from_pieces,
)
from src.problem import TrackOptimizationProblem


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def cfg() -> OptimizationConfig:
    return OptimizationConfig.load("configs/all_pieces.yaml")


@pytest.fixture
def dims(cfg, cat) -> PartitionedDimensions:
    return compute_dimensions(cfg, cat)


@pytest.fixture
def prob(cat, cfg) -> TrackOptimizationProblem:
    return TrackOptimizationProblem(catalog=cat, config=cfg)


@pytest.fixture
def cfg_crossing() -> OptimizationConfig:
    """The CROSS_90 config (with_crossing) — semantic home of the figure-8-cross seed."""
    return OptimizationConfig.load("configs/with_crossing.yaml")


@pytest.fixture
def dims_crossing(cfg_crossing, cat) -> PartitionedDimensions:
    return compute_dimensions(cfg_crossing, cat)


@pytest.fixture
def prob_crossing(cat, cfg_crossing) -> TrackOptimizationProblem:
    return TrackOptimizationProblem(catalog=cat, config=cfg_crossing)


def _chromosome(pattern, dims):
    main_pieces, main_flips, junctions, cross_junctions, dbl_crossovers = pattern
    return create_chromosome_from_pieces(
        dims, main_pieces, main_loop_flips=main_flips,
        junctions=junctions, cross_junctions=cross_junctions,
        double_crossovers=dbl_crossovers,
    )


def decode_seed(pattern, cfg, cat, dims):
    """Decode one Pattern tuple -> MultiPathLayout (using config inventory + boundary)."""
    return decode_chromosome(_chromosome(pattern, dims), cat, cfg.inventory, dims=dims)


def assert_valid_closed(layout, cfg, *, n_cross=0, n_dc=0):
    """Weak oracle: closed, committed crossings, fits boundary. Does NOT check collisions."""
    assert layout.max_closure_error < 4.0, f"not closed: {layout.max_closure_error}"
    assert len(layout.cross_junctions) == n_cross, \
        f"CROSS_90 committed {len(layout.cross_junctions)} != {n_cross}"
    assert layout.n_dbl_crossovers == n_dc, \
        f"DC committed {layout.n_dbl_crossovers} != {n_dc}"
    xs, ys = [], []
    for p in layout.paths:
        if len(p.states):
            xs.append(p.states[:, 0]); ys.append(p.states[:, 1])
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    b = cfg.boundary
    assert (xs.max() - xs.min()) <= (b.max_x - b.min_x), "too wide for boundary"
    assert (ys.max() - ys.min()) <= (b.max_y - b.min_y), "too tall for boundary"


def assert_seed_feasible(pattern, prob, cfg, cat, dims, *, n_cross=0, n_dc=0):
    """STRONG oracle: the seed must decode to a fully FEASIBLE layout (all constraints
    <= 0, including the collision constraint G[4]) AND commit the expected crossings.

    Rejects over-wound self-overlapping coils that pass closure but collide. Returns the
    decoded layout so callers can check piece count.
    """
    x = _chromosome(pattern, dims)
    out: dict = {}
    prob._evaluate(x, out)
    g = np.asarray(out["G"], dtype=float)
    assert np.all(g <= 1e-9), (
        f"infeasible seed: collisions(G4)={g[4]:.3f}, "
        f"closure(G0..2)={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}, boundary(G3)={g[3]:.2f}"
    )
    layout = decode_chromosome(x, cat, cfg.inventory, dims=dims)
    assert len(layout.cross_junctions) == n_cross, \
        f"CROSS_90 committed {len(layout.cross_junctions)} != {n_cross}"
    assert layout.n_dbl_crossovers == n_dc, \
        f"DC committed {layout.n_dbl_crossovers} != {n_dc}"
    return layout


def _inv_by_index(cfg, cat):
    return {cat._id_to_index[k]: v for k, v in cfg.inventory.items() if k in cat._id_to_index}


def test_existing_figure_eight_cross_validates(cfg_crossing, cat, dims_crossing, prob_crossing):
    """Sanity: the existing 1-cross seed passes BOTH oracles on with_crossing (the
    CROSS_90 config). Proves the oracles are correct against known-good geometry."""
    from src.sampling import _gen_figure_eight_cross
    variants = _gen_figure_eight_cross(_inv_by_index(cfg_crossing, cat), dims_crossing)
    assert variants, "expected the existing 1-cross seed to be emitted for with_crossing"
    assert_valid_closed(decode_seed(variants[0], cfg_crossing, cat, dims_crossing), cfg_crossing, n_cross=1, n_dc=0)
    assert_seed_feasible(variants[0], prob_crossing, cfg_crossing, cat, dims_crossing, n_cross=1, n_dc=0)


def test_existing_figure_eight_dc_validates(cfg, cat, dims, prob):
    """Sanity: the existing DC figure-8 seed passes the strong oracle."""
    from src.sampling import _gen_figure_eight_dbl_crossover
    variants = _gen_figure_eight_dbl_crossover(_inv_by_index(cfg, cat), dims)
    assert variants, "expected the existing DC seed to be emitted for all_pieces"
    assert_seed_feasible(variants[0], prob, cfg, cat, dims, n_cross=0, n_dc=1)


def test_strong_oracle_rejects_overwound_coil(cfg, cat, dims, prob):
    """Guard: a 28-curve-per-lobe 'figure-8' closes but self-collides (50 crossings);
    the strong oracle MUST reject it (the weak closure check alone would pass)."""
    from src.encoding import STRAIGHT_16, R40_CURVE
    run = 5
    pieces = ([int(STRAIGHT_16)] * run + [int(R40_CURVE)] * 28
              + [int(STRAIGHT_16)] * run + [int(R40_CURVE)] * 28)
    flips = [0] * run + [0] * 28 + [0] * run + [1] * 28
    pat = (pieces, flips, None, [(1, 2, run + 28 + 2)], None)
    with pytest.raises(AssertionError, match="infeasible"):
        assert_seed_feasible(pat, prob, cfg, cat, dims, n_cross=1, n_dc=0)
