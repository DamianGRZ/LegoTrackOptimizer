# tests/test_boundary_repair.py
import numpy as np
import pytest

from src.encoding import (
    PartitionedDimensions, INACTIVE, PieceIndex,
    create_empty_chromosome, set_main_loop_type, set_flip,
)

# Catalog FK rows (index = piece index): [dx, dy, dtheta_deg]
FK = np.array([
    [16.0, 0.0, 0.0],       # 0 STRAIGHT_16
    [24.0, 0.0, 0.0],       # 1 STRAIGHT_24
    [15.307, 3.045, 22.5],  # 2 R40_CURVE
    [0.0, 0.0, 90.0],       # 3 CROSS_90 (unused here)
    [0.0, 0.0, 0.0],        # 4 R40_SWITCH_LEFT
    [0.0, 0.0, 0.0],        # 5 R40_SWITCH_RIGHT
    [0.0, 0.0, 0.0],        # 6 DOUBLE_CROSSOVER
], dtype=np.float64)

S16 = int(PieceIndex.STRAIGHT_16)
R40 = int(PieceIndex.R40_CURVE)


def make_dims(n_main=24, max_junctions=0, box=250.0):
    return PartitionedDimensions(
        n_main=n_main, max_junctions=max_junctions,
        max_cross_junctions=0, max_double_crossovers=0,
        n_straights_16=n_main, n_straights_24=0,
        boundary_min_x=-box, boundary_max_x=box,
        boundary_min_y=-box, boundary_max_y=box,
    )


def test_main_loop_states_single_straight():
    from src.repair import _main_loop_states
    dims = make_dims(n_main=4)
    x = create_empty_chromosome(dims)
    set_main_loop_type(x, dims, 0, S16)  # one 16-stud straight at heading 0
    states = _main_loop_states(x, dims, FK)
    # origin + one straight along +x
    assert states.shape == (2, 3)
    np.testing.assert_allclose(states[1], [16.0, 0.0, 0.0], atol=1e-6)


def test_antiparallel_pair_detected_for_S16():
    from src.repair import _active_straight_headings, _find_antiparallel_pairs
    dims = make_dims(n_main=24)
    x = create_empty_chromosome(dims)
    # straight @0, 8 R40 left turns (=> heading 180), straight @180
    set_main_loop_type(x, dims, 0, S16)
    for i in range(1, 9):
        set_main_loop_type(x, dims, i, R40)
        set_flip(x, dims, i, 0)  # +22.5 each => 180 total
    set_main_loop_type(x, dims, 9, S16)

    headings = _active_straight_headings(x, dims, FK)
    # two straights: slot 0 @ ~0deg, slot 9 @ ~180deg
    by_slot = {slot: round(h % 360, 1) for slot, ptype, h in headings}
    assert by_slot[0] == pytest.approx(0.0, abs=0.5)
    assert by_slot[9] == pytest.approx(180.0, abs=0.5)

    pairs = _find_antiparallel_pairs(headings, axis="x")  # 0/180 axis
    assert (0, 9) in [tuple(sorted(p)) for p in pairs]


def test_translate_zeros_offset_when_loop_fits():
    from src.repair import BoundaryAwareRepair
    dims = make_dims(n_main=24, box=250.0)
    x = create_empty_chromosome(dims)
    # Small loop that easily fits the 500x500 box: a few straights + curves.
    for i in range(8):
        set_main_loop_type(x, dims, i, R40)  # 8 curves, ~half circle
    set_main_loop_type(x, dims, 8, S16)
    # Push it out of bounds via a large fine-tuning offset.
    x[dims.start_pos_start] = 240
    x[dims.start_pos_start + 1] = 0

    rep = BoundaryAwareRepair(dims, FK, boundary_tolerance=0.0)
    X = np.array([x])
    rep._do(None, X)
    # Fits => translate => offset zeroed.
    assert int(X[0, dims.start_pos_start]) == 0
    assert int(X[0, dims.start_pos_start + 1]) == 0
    # No straights removed (still active).
    assert int(X[0, 8]) == S16


def test_shrink_removes_antiparallel_pair_when_too_wide():
    from src.repair import BoundaryAwareRepair
    # Tiny box so the loop is "too wide" on x and must shrink.
    dims = make_dims(n_main=24, box=20.0)  # box width = 40 studs
    x = create_empty_chromosome(dims)
    # Degenerate loop: S16 @0, 8 R40 (=>180), S16 @180, 8 R40 (=>360).
    set_main_loop_type(x, dims, 0, S16)
    for i in range(1, 9):
        set_main_loop_type(x, dims, i, R40)
    set_main_loop_type(x, dims, 9, S16)
    for i in range(10, 18):
        set_main_loop_type(x, dims, i, R40)

    rep = BoundaryAwareRepair(dims, FK, boundary_tolerance=0.0, siding_margin=0.0)
    X = np.array([x])
    rep._do(None, X)
    # The anti-parallel S16 pair (slots 0 and 9) should be deactivated.
    assert int(X[0, 0]) == INACTIVE
    assert int(X[0, 9]) == INACTIVE
    # Offset zeroed too.
    assert int(X[0, dims.start_pos_start]) == 0


def test_shrink_declines_when_no_antiparallel_pair():
    from src.repair import BoundaryAwareRepair
    dims = make_dims(n_main=8, box=5.0)  # box width 10 -> too small
    x = create_empty_chromosome(dims)
    # Two straights both heading 0 (no anti-parallel partner).
    set_main_loop_type(x, dims, 0, S16)
    set_main_loop_type(x, dims, 1, S16)
    rep = BoundaryAwareRepair(dims, FK, boundary_tolerance=0.0, siding_margin=0.0)
    X = np.array([x])
    rep._do(None, X)  # must not raise, must not remove (no valid pair)
    assert int(X[0, 0]) == S16
    assert int(X[0, 1]) == S16


def test_pipeline_includes_boundary_repair_and_runs():
    from src.repair import TrackRepairPipeline, BoundaryAwareRepair
    dims = make_dims(n_main=24, max_junctions=0, box=250.0)
    inv = {S16: 24, R40: 24}
    pipe = TrackRepairPipeline(
        dims=dims, inventory_by_index=inv, catalog_fk_table=FK,
        boundary_tolerance=0.0, enable_boundary_repair=True,
    )
    assert isinstance(pipe.boundary_repair, BoundaryAwareRepair)
    # Out-of-bounds-by-offset individual gets re-centered by the full pipeline.
    x = create_empty_chromosome(dims)
    for i in range(8):
        set_main_loop_type(x, dims, i, R40)
    set_main_loop_type(x, dims, 8, S16)
    x[dims.start_pos_start] = 240
    X = pipe._do(None, np.array([x]))
    assert int(X[0, dims.start_pos_start]) == 0


def _wide_loop(dims):
    """S16 @0, 8 R40 (=>180), S16 @180, 8 R40 (=>360). Spans 96 x 80 studs."""
    x = create_empty_chromosome(dims)
    set_main_loop_type(x, dims, 0, S16)
    for i in range(1, 9):
        set_main_loop_type(x, dims, i, R40)
    set_main_loop_type(x, dims, 9, S16)
    for i in range(10, 18):
        set_main_loop_type(x, dims, i, R40)
    return x


def _repaired(dims, x, tol):
    from src.repair import BoundaryAwareRepair
    population = np.array([x])
    BoundaryAwareRepair(dims, FK, boundary_tolerance=tol)._do(None, population)
    return population[0]


class TestToleranceMatchesTheBoundaryConstraint:
    """The repair grants the same per-edge overshoot allowance as G[3], so it never
    rewrites a layout the constraint accepts."""

    def test_loop_inside_the_band_is_left_untouched(self):
        # A 94-wide box leaves the 96-stud loop 1 stud over each x edge, inside a
        # 2-stud allowance — the constraint scores this feasible, so must the repair.
        dims = make_dims(n_main=24, box=47.0)
        x = _wide_loop(dims)
        assert np.array_equal(_repaired(dims, x, tol=2.0), x)

    def test_same_loop_shrinks_when_no_allowance_is_granted(self):
        # Identical geometry, tolerance the only difference: this is what gates it.
        dims = make_dims(n_main=24, box=47.0)
        repaired = _repaired(dims, _wide_loop(dims), tol=0.0)
        assert int(repaired[0]) == INACTIVE
        assert int(repaired[9]) == INACTIVE

    def test_loop_past_the_band_still_shrinks(self):
        # A 90-wide box leaves it 3 studs over each edge — past the allowance.
        dims = make_dims(n_main=24, box=45.0)
        repaired = _repaired(dims, _wide_loop(dims), tol=2.0)
        assert int(repaired[0]) == INACTIVE
        assert int(repaired[9]) == INACTIVE


class TestEdgesAreInsetIndividually:
    """The siding reserve moves each edge, replacing a span-level multiplier."""

    def test_inset_pulls_both_edges_in_by_the_margin(self):
        from src.repair import _inset
        # Asymmetric on purpose: the reserve is per edge, not a share of the width.
        assert _inset(-100.0, 60.0, 16.0) == (-84.0, 44.0)

    def test_inset_collapses_to_the_midpoint_when_edges_would_cross(self):
        from src.repair import _inset
        assert _inset(-5.0, 5.0, 20.0) == (0.0, 0.0)


def test_pipeline_carries_the_configs_boundary_tolerance(catalog, default_config):
    """The value the constraint reads must be the value the repair acts on."""
    from src.algorithm.runner import _build_search_components
    from src.problem import TrackOptimizationProblem

    problem = TrackOptimizationProblem(catalog, default_config)
    _, _, _, repair = _build_search_components(default_config, problem, catalog)
    assert repair.boundary_repair.boundary_tolerance == default_config.boundary_tolerance
