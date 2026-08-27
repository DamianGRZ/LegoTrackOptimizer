# tests/test_closure_translation.py
"""Translational closure: drop straights to kill the dx/dy gap of a loop that
is already angularly closed (sum of turns = 360 deg)."""
import numpy as np
import pytest

from src.encoding import (
    PartitionedDimensions, INACTIVE, PieceIndex,
    create_empty_chromosome, set_junction, set_main_loop_type,
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


def _angularly_closed_lens(x, dims):
    """S16 @0, 8x R40 (=>180), S16 @180, 8x R40 (=>360): closed loop, gap ~0."""
    set_main_loop_type(x, dims, 0, S16)
    for i in range(1, 9):
        set_main_loop_type(x, dims, i, R40)
    set_main_loop_type(x, dims, 9, S16)
    for i in range(10, 18):
        set_main_loop_type(x, dims, i, R40)


def _gap(x, dims):
    from src.repair import _main_loop_states
    states = _main_loop_states(x, dims, FK)
    return float(np.hypot(states[-1, 0] - states[0, 0],
                          states[-1, 1] - states[0, 1]))


def test_close_translation_drops_straight_to_close_positional_gap():
    from src.repair import MainLoopClosureRepair
    dims = make_dims(n_main=24)
    inv = {S16: 24, R40: 24}
    x = create_empty_chromosome(dims)
    _angularly_closed_lens(x, dims)
    # One EXTRA forward straight (heading 0): angle untouched, +16 stud gap.
    set_main_loop_type(x, dims, 18, S16)

    assert _gap(x, dims) == pytest.approx(16.0, abs=0.5)  # precondition

    MainLoopClosureRepair(dims, FK, inv)._repair_chromosome(x)

    # Repair removed a forward straight: loop is positionally closed again.
    assert _gap(x, dims) == pytest.approx(0.0, abs=0.5)
    n_straights = int(np.sum(x[:dims.n_main] == S16))
    assert n_straights == 2  # 3 placed, 1 dropped


def test_balanced_loop_keeps_all_straights():
    """An already-closed loop (gap ~0) must not lose any straights."""
    from src.repair import MainLoopClosureRepair
    dims = make_dims(n_main=24)
    x = create_empty_chromosome(dims)
    _angularly_closed_lens(x, dims)  # gap ~0, two balanced straights

    assert _gap(x, dims) == pytest.approx(0.0, abs=0.5)
    MainLoopClosureRepair(dims, FK, {S16: 24, R40: 24})._repair_chromosome(x)

    assert _gap(x, dims) == pytest.approx(0.0, abs=0.5)
    assert int(np.sum(x[:dims.n_main] == S16)) == 2  # nothing removed


def test_skew_gap_closed_on_two_headings():
    """A diagonal gap (x and y) closes by dropping a straight on each axis."""
    from src.repair import MainLoopClosureRepair
    dims = make_dims(n_main=24)
    x = create_empty_chromosome(dims)
    # Square: 4 corners of 4x R40 (=90 deg each), with an extra forward straight
    # on the heading-0 and heading-90 sides => gap (16, 16).
    set_main_loop_type(x, dims, 0, S16)   # h0
    set_main_loop_type(x, dims, 1, S16)   # h0 (extra)
    for i in range(2, 6):
        set_main_loop_type(x, dims, i, R40)   # -> h90
    set_main_loop_type(x, dims, 6, S16)   # h90
    set_main_loop_type(x, dims, 7, S16)   # h90 (extra)
    for i in range(8, 12):
        set_main_loop_type(x, dims, i, R40)   # -> h180
    set_main_loop_type(x, dims, 12, S16)  # h180
    for i in range(13, 17):
        set_main_loop_type(x, dims, i, R40)   # -> h270
    set_main_loop_type(x, dims, 17, S16)  # h270
    for i in range(18, 22):
        set_main_loop_type(x, dims, i, R40)   # -> h360

    assert _gap(x, dims) == pytest.approx(np.hypot(16.0, 16.0), abs=0.5)

    MainLoopClosureRepair(dims, FK, {S16: 24, R40: 24})._repair_chromosome(x)

    assert _gap(x, dims) == pytest.approx(0.0, abs=0.5)
    assert int(np.sum(x[:dims.n_main] == S16)) == 4  # 6 placed, 2 dropped


def test_active_siding_skips_translational_closure():
    """A genome with an active passing-siding junction keeps all its straights.

    The switch pair re-lengthens the loop at decode (32-stud bodies vs 16-stud
    straights), so the raw main-loop dx/dy gap is an artifact — Stage-2 must be
    skipped, mirroring the double-crossover exemption.
    """
    from src.repair import MainLoopClosureRepair
    dims = make_dims(n_main=24, max_junctions=1)
    x = create_empty_chromosome(dims)
    _angularly_closed_lens(x, dims)
    set_main_loop_type(x, dims, 18, S16)  # extra forward straight -> +16 raw gap
    set_junction(x, dims, 0, active=1, position=2, handedness=0, n_straights=0)

    assert _gap(x, dims) == pytest.approx(16.0, abs=0.5)  # precondition: raw gap
    before = int(np.sum(x[:dims.n_main] == S16))

    MainLoopClosureRepair(dims, FK, {S16: 24, R40: 24})._repair_chromosome(x)

    # Stage-2 skipped: no straight dropped, raw gap left for the decoder to absorb.
    assert int(np.sum(x[:dims.n_main] == S16)) == before
    assert _gap(x, dims) == pytest.approx(16.0, abs=0.5)
