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
