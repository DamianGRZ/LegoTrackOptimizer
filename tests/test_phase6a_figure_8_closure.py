"""Regression test for the Phase 6a closure-search-driven _emit_figure_8 fix.

Per the architectural Decision log added to docs/PLAN.md on 2026-05-07,
the figure-8 emitter must use a closing parametrization. The closure
search at tools/figure8_closure_search.py established that diagonal-
quadrant lobes (port pairs B->C, B->D, D->A, C->A) close with
``M1 STR + 12 R40 same-handed + M2 STR`` shape, with M1 ∈ {2}, M2 ∈
{2, 3}. The pre-fix perpendicular topology (B<->A horizontal lobe,
C<->D vertical lobe) admits zero closures because the lobes physically
intersect outside the cross.

This test asserts that the layout produced by ``_emit_figure_8``
actually closes within 2 stud / 2 deg on both lobes for both
diagonal-pair variants.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import DecoderConfig, decode_chromosome
from src_v2.encoding import (
    compute_port_pair_dimensions,
    create_empty_chromosome,
    set_anchor,
)
from src_v2.operators import PortPairSampling, _emit_figure_8


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
WITH_CROSSING_CFG = Path(__file__).parent.parent / "configs" / "with_crossing.yaml"

_POS_TOL = 2.0
_ANGLE_TOL_DEG = 2.0


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(WITH_CROSSING_CFG)
    dims = compute_port_pair_dimensions(config.boundary, catalog, config.inventory)
    decoder_config = DecoderConfig(
        boundary_min_x=config.boundary.min_x,
        boundary_max_x=config.boundary.max_x,
        boundary_min_y=config.boundary.min_y,
        boundary_max_y=config.boundary.max_y,
    )
    return catalog, config, dims, decoder_config


def test_emit_figure_8_emits_two_diagonal_variants(setup) -> None:
    """The fixed emitter ships exactly two variants: top-left+bottom-right
    (LEFT-handed) and top-right+bottom-left (RIGHT-handed). Pre-fix it
    emitted four (including non-closing perpendicular mixes)."""
    catalog, config, dims, _decoder_config = setup
    patterns = _emit_figure_8(catalog, config.inventory, dims)
    assert len(patterns) == 2


def test_emit_figure_8_lobes_close_within_tolerance(setup) -> None:
    """Each pattern from ``_emit_figure_8`` decodes (via ``decode_chromosome``
    directly, BYPASSING the ``PortPairProblem._evaluate`` pipeline) to a
    layout whose closure residuals satisfy ``|dx|<2 stud, |dy|<2 stud,
    |dtheta|<2 deg`` on both cycles.

    Why direct decode: the ``CycleClosureRepair`` inside ``_evaluate``
    targets a fixed 16-R40 cycle budget (matching V2's typical mainline
    oval) and tries to "repair" the 12-R40 figure-8 lobes by splicing in
    extra R40s -- which structurally breaks the closure designed by the
    closure search. Integrating the repair pipeline with figure-8 lobes
    is a follow-up; here we verify the FK closure of the EMITTED layout."""
    catalog, config, dims, decoder_config = setup
    patterns = _emit_figure_8(catalog, config.inventory, dims)
    assert patterns, "expected at least one figure-8 pattern"

    sampler = PortPairSampling(dims, catalog, config, heuristic_ratio=0.30)

    for pattern_idx, pattern in enumerate(patterns):
        x = create_empty_chromosome(dims)
        sampler._populate_from_pattern(x, pattern)
        set_anchor(x, dims, 0, 0, 0)

        graph = decode_chromosome(x, dims, catalog, decoder_config)
        assert graph.n_cycles >= 2, (
            f"pattern {pattern_idx}: figure-8 should yield >=2 cycles; "
            f"got {graph.n_cycles}"
        )
        assert len(graph.closure_residuals) >= 2

        for r_idx, residual in enumerate(graph.closure_residuals):
            dx, dy = float(residual.dx), float(residual.dy)
            dtheta_deg = math.degrees(float(residual.dtheta))
            assert abs(dx) < _POS_TOL, (
                f"pattern {pattern_idx} residual {r_idx}: dx={dx:.4f} "
                f"exceeds {_POS_TOL} stud tolerance"
            )
            assert abs(dy) < _POS_TOL, (
                f"pattern {pattern_idx} residual {r_idx}: dy={dy:.4f} "
                f"exceeds {_POS_TOL} stud tolerance"
            )
            assert abs(dtheta_deg) < _ANGLE_TOL_DEG, (
                f"pattern {pattern_idx} residual {r_idx}: "
                f"dtheta={dtheta_deg:.4f} deg exceeds "
                f"{_ANGLE_TOL_DEG} deg tolerance"
            )


def test_emit_figure_8_inventory_requirements(setup) -> None:
    """The diagonal-lobe parametrization needs 24 R40_CURVE + 9 STRAIGHT_16
    + 1 CROSS_90."""
    catalog, _config, dims, _decoder_config = setup

    inv_short_r40 = {"CROSS_90": 1, "R40_CURVE": 23, "STRAIGHT_16": 9}
    assert _emit_figure_8(catalog, inv_short_r40, dims) == []

    inv_short_str = {"CROSS_90": 1, "R40_CURVE": 24, "STRAIGHT_16": 8}
    assert _emit_figure_8(catalog, inv_short_str, dims) == []

    inv_ok = {"CROSS_90": 1, "R40_CURVE": 24, "STRAIGHT_16": 9}
    assert len(_emit_figure_8(catalog, inv_ok, dims)) == 2
