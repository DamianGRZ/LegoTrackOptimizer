"""Crossing seed family tests (family C: DOUBLE_CROSSOVER weave).

Family C reduces to removing a hardcoded size cap on the existing
``_gen_figure_eight_dbl_crossover`` (a 2-DC weave is not geometrically derivable
from the R40-only kit — every chained-unit candidate self-overlaps, G[4] >= 0.4).
Uncapped, the DC figure-8 scales from inventory + boundary to a ~128-piece /
~61%-utilisation seed that competes with the racetrack instead of being bred out.

Each emitted variant must pass the STRONG oracle (`assert_seed_feasible`): the seed
decodes to a fully FEASIBLE layout (all constraints <= 0, including the collision
constraint G[4]) and commits exactly one DOUBLE_CROSSOVER. The weak closure-only
check is intentionally NOT used here.
"""
import pytest

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.encoding import compute_dimensions
from src.problem import TrackOptimizationProblem
from src.sampling import _gen_figure_eight_dbl_crossover
from tests.test_seed_geometry_harness import assert_seed_feasible


@pytest.fixture
def cat() -> TrackCatalog:
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def cfg() -> OptimizationConfig:
    return OptimizationConfig.load("configs/all_pieces.yaml")


@pytest.fixture
def dims(cfg, cat):
    return compute_dimensions(cfg, cat)


@pytest.fixture
def prob(cat, cfg) -> TrackOptimizationProblem:
    return TrackOptimizationProblem(catalog=cat, config=cfg)


@pytest.fixture
def inv(cfg, cat):
    return {cat._id_to_index[k]: v for k, v in cfg.inventory.items() if k in cat._id_to_index}


def test_dc_weave_variants_all_feasible(inv, dims, prob, cfg, cat):
    """Every emitted DC figure-8 variant decodes to a fully feasible, 1-DC layout."""
    variants = _gen_figure_eight_dbl_crossover(inv, dims)
    assert variants, "expected the DC figure-8 seed to emit >= 1 variant for all_pieces"
    for pat in variants:
        assert_seed_feasible(pat, prob, cfg, cat, dims, n_cross=0, n_dc=1)


def test_uncapped_dc_weave_reaches_competitive_size(inv, dims, prob, cfg, cat):
    """With the hardcoded k<=6 cap removed, the largest variant is boundary-limited
    (~128 pieces), not the old ~88. This is what lets the DC seed compete with the
    ~60% racetrack instead of being bred out at 42%."""
    variants = _gen_figure_eight_dbl_crossover(inv, dims)
    best = max(
        assert_seed_feasible(pat, prob, cfg, cat, dims, n_cross=0, n_dc=1).n_pieces
        for pat in variants
    )
    assert best >= 120, f"largest DC figure-8 only {best} pieces (expected boundary-limited >= 120)"


# =============================================================================
# Family D: scaled single-CROSS_90 figure-8 (compensated straight pairs)
# =============================================================================

@pytest.fixture
def cfg_crossing() -> OptimizationConfig:
    return OptimizationConfig.load("configs/with_crossing.yaml")


@pytest.fixture
def dims_crossing(cfg_crossing, cat):
    return compute_dimensions(cfg_crossing, cat)


@pytest.fixture
def prob_crossing(cat, cfg_crossing) -> TrackOptimizationProblem:
    return TrackOptimizationProblem(catalog=cat, config=cfg_crossing)


@pytest.fixture
def inv_crossing(cfg_crossing, cat):
    return {cat._id_to_index[k]: v for k, v in cfg_crossing.inventory.items()
            if k in cat._id_to_index}


def test_cross_seed_emits_scaled_variants(inv_crossing, dims_crossing):
    """The 1-cross figure-8 must emit size variants, not only the 34-piece base."""
    from src.sampling import _gen_figure_eight_cross
    variants = _gen_figure_eight_cross(inv_crossing, dims_crossing)
    assert len(variants) >= 2, "expected scaled variants beyond the base figure-8"


def test_cross_seed_variants_all_feasible(inv_crossing, dims_crossing, prob_crossing,
                                          cfg_crossing, cat):
    """Every emitted variant decodes fully feasible with exactly 1 committed CROSS_90."""
    from src.sampling import _gen_figure_eight_cross
    for pat in _gen_figure_eight_cross(inv_crossing, dims_crossing):
        assert_seed_feasible(pat, prob_crossing, cfg_crossing, cat, dims_crossing,
                             n_cross=1, n_dc=0)


def test_cross_seed_scales_to_boundary_limit(inv_crossing, dims_crossing, prob_crossing,
                                             cfg_crossing, cat):
    """In the 500x500 box with 120 straights the largest variant is boundary-limited
    (~118 pieces), lifting the cross family from 17% toward racetrack-level density."""
    from src.sampling import _gen_figure_eight_cross
    variants = _gen_figure_eight_cross(inv_crossing, dims_crossing)
    best = max(
        assert_seed_feasible(pat, prob_crossing, cfg_crossing, cat, dims_crossing,
                             n_cross=1, n_dc=0).n_pieces
        for pat in variants
    )
    assert best >= 110, f"largest cross figure-8 only {best} pieces (expected ~118)"


def test_cross_seed_respects_small_boundary(inv_crossing, cfg_crossing, cat):
    """A 200x200 box still admits the base plus small variants — every emitted
    variant must fit and stay feasible (no oversized emissions)."""
    from src.sampling import _gen_figure_eight_cross
    small = OptimizationConfig(
        inventory=dict(cfg_crossing.inventory),
        boundary={"min_x": -100.0, "max_x": 100.0, "min_y": -100.0, "max_y": 100.0},
    )
    dims_small = compute_dimensions(small, cat)
    prob_small = TrackOptimizationProblem(cat, small)
    variants = _gen_figure_eight_cross(
        {cat._id_to_index[k]: v for k, v in small.inventory.items()
         if k in cat._id_to_index},
        dims_small,
    )
    assert variants, "base figure-8 (160x160) must still be emitted in a 200x200 box"
    for pat in variants:
        assert_seed_feasible(pat, prob_small, small, cat, dims_small, n_cross=1, n_dc=0)
