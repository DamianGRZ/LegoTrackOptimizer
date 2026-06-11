"""Characterization tests for the vectorized FK chain and intersection scans.

The ``_ref_*`` functions below are VERBATIM copies of the pre-vectorization
(pure-Python loop) implementations. They pin the exact semantics; the live
implementations in ``src.geometry`` / ``src.intersection`` must match them on
randomized and edge-case inputs — before and after vectorization.
"""

import numpy as np
import pytest

from src.geometry import compute_fk_chain
from src.intersection import count_segment_crossings, find_crossing_pairs

RNG = np.random.default_rng(42)

# Catalog-like FK rows: STR16, STR24, R40 left, R40 right (flip pre-applied).
_PIECE_DELTAS = np.array([
    [16.0, 0.0, 0.0],
    [24.0, 0.0, 0.0],
    [15.307, 3.045, 22.5],
    [15.307, -3.045, -22.5],
], dtype=np.float64)


# =============================================================================
# Reference implementations (verbatim pre-vectorization copies)
# =============================================================================

def _ref_fk_chain(fk_deltas):
    n = len(fk_deltas)
    states = np.zeros((n + 1, 3), dtype=np.float64)
    for i in range(n):
        dx, dy, dtheta = fk_deltas[i]
        theta_rad = np.radians(states[i, 2])
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        states[i + 1, 0] = states[i, 0] + dx * cos_t - dy * sin_t
        states[i + 1, 1] = states[i, 1] + dx * sin_t + dy * cos_t
        states[i + 1, 2] = states[i, 2] + dtheta
    return states


def _ref_segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    def cross(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = cross(cx, cy, dx, dy, ax, ay)
    d2 = cross(cx, cy, dx, dy, bx, by)
    d3 = cross(ax, ay, bx, by, cx, cy)
    d4 = cross(ax, ay, bx, by, dx, dy)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


_EXEMPT = frozenset({3, 6})  # CROSS_90_INDEX, DOUBLE_CROSSOVER_INDEX


def _ref_exempt_positions(piece_indices):
    if not piece_indices:
        return set()
    return {i for i, idx in enumerate(piece_indices) if idx in _EXEMPT}


def _ref_count(states, piece_indices=None, min_separation=3):
    n = len(states) - 1
    if n < min_separation + 1:
        return 0
    cross_positions = _ref_exempt_positions(piece_indices)
    x = states[:, 0]
    y = states[:, 1]
    crossings = 0
    for i in range(n - min_separation):
        ax, ay = x[i], y[i]
        bx, by = x[i + 1], y[i + 1]
        for j in range(i + min_separation, n):
            cx, cy = x[j], y[j]
            dx, dy = x[j + 1], y[j + 1]
            if _ref_segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                if i in cross_positions or j in cross_positions:
                    continue
                crossings += 1
    return crossings


def _ref_pairs(states, piece_indices=None, min_separation=3):
    n = len(states) - 1
    if n < min_separation + 1:
        return []
    cross_positions = _ref_exempt_positions(piece_indices)
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    pairs = []
    for i in range(n - min_separation):
        if i in cross_positions:
            continue
        for j in range(i + min_separation, n):
            if j in cross_positions:
                continue
            if _ref_segments_intersect(x[i], y[i], x[i + 1], y[i + 1],
                                       x[j], y[j], x[j + 1], y[j + 1]):
                raw_diff = abs(theta[i] - theta[j]) % 180
                angle_diff = min(raw_diff, 180 - raw_diff)
                pairs.append((i, j, angle_diff))
    pairs.sort(key=lambda t: abs(t[2] - 90.0))
    return pairs


# =============================================================================
# Input generators
# =============================================================================

def _random_piece_deltas(n, rng):
    """Catalog-like FK rows sampled with replacement."""
    return _PIECE_DELTAS[rng.integers(0, len(_PIECE_DELTAS), size=n)]


def _wavy_loop_states(n, waves=7, amp=5.0):
    """Closed wavy loop with self-near-passes — exercises the crossing scan."""
    t = np.linspace(0, 2 * np.pi, n + 1)
    x = 200 * np.cos(t) + amp * np.cos(waves * t)
    y = 180 * np.sin(t) + amp * np.sin(waves * t)
    th = np.degrees(np.arctan2(np.gradient(y), np.gradient(x)))
    return np.column_stack([x, y, th]).astype(np.float64)


def _pinched_loop_states():
    """Small figure-8-like trajectory that genuinely self-crosses.

    Phase offset keeps the two crossing passes strictly inside segment
    interiors — the scan's strict orientation test ignores crossings that
    coincide with segment endpoints.
    """
    t = np.linspace(0, 2 * np.pi, 41) + 0.07
    x = 100 * np.sin(t)
    y = 60 * np.sin(2 * t)
    th = np.degrees(np.arctan2(np.gradient(y), np.gradient(x)))
    return np.column_stack([x, y, th]).astype(np.float64)


# =============================================================================
# compute_fk_chain equivalence
# =============================================================================

class TestFKChainEquivalence:

    @pytest.mark.parametrize("n", [0, 1, 2, 5, 16, 50, 200])
    def test_piece_like_deltas(self, n):
        deltas = _random_piece_deltas(n, np.random.default_rng(n))
        np.testing.assert_allclose(
            compute_fk_chain(deltas), _ref_fk_chain(deltas),
            rtol=1e-12, atol=1e-9,
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_arbitrary_float_deltas(self, seed):
        rng = np.random.default_rng(seed)
        deltas = rng.uniform(-50, 50, size=(rng.integers(1, 120), 3))
        np.testing.assert_allclose(
            compute_fk_chain(deltas), _ref_fk_chain(deltas),
            rtol=1e-12, atol=1e-9,
        )

    def test_empty_returns_single_origin_state(self):
        out = compute_fk_chain(np.zeros((0, 3)))
        assert out.shape == (1, 3)
        np.testing.assert_array_equal(out, np.zeros((1, 3)))

    def test_full_circle_closes(self):
        deltas = np.tile([15.307, 3.045, 22.5], (16, 1))
        states = compute_fk_chain(deltas)
        np.testing.assert_allclose(states[-1, :2], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(states[-1, 2], 360.0, atol=1e-9)

    def test_accepts_python_list_input(self):
        deltas = [[16.0, 0.0, 0.0], [15.307, 3.045, 22.5]]
        np.testing.assert_allclose(
            compute_fk_chain(np.asarray(deltas)),
            _ref_fk_chain(np.asarray(deltas)),
            rtol=1e-12, atol=1e-9,
        )


# =============================================================================
# count_segment_crossings equivalence
# =============================================================================

class TestCountCrossingsEquivalence:

    @pytest.mark.parametrize("n", [4, 20, 40, 80, 160])
    def test_wavy_loops(self, n):
        states = _wavy_loop_states(n)
        pieces = [0] * n
        assert (count_segment_crossings(states, pieces)
                == _ref_count(states, pieces))

    def test_pinched_loop_has_crossings_and_matches(self):
        states = _pinched_loop_states()
        n = len(states) - 1
        pieces = [0] * n
        ref = _ref_count(states, pieces)
        assert ref > 0, "fixture must genuinely self-cross"
        assert count_segment_crossings(states, pieces) == ref

    def test_exempt_positions_skip_pairs(self):
        states = _pinched_loop_states()
        n = len(states) - 1
        ref_plain = _ref_pairs(states, [0] * n)
        assert ref_plain, "fixture must self-cross"
        # Exempt one side of the first crossing pair via CROSS_90 index.
        i0, j0, _ = ref_plain[0]
        pieces = [0] * n
        pieces[i0] = 3  # CROSS_90_INDEX
        assert (count_segment_crossings(states, pieces)
                == _ref_count(states, pieces))
        pieces[j0] = 6  # DOUBLE_CROSSOVER_INDEX too
        assert (count_segment_crossings(states, pieces)
                == _ref_count(states, pieces))

    @pytest.mark.parametrize("min_sep", [1, 2, 3, 5])
    def test_min_separation_variants(self, min_sep):
        states = _pinched_loop_states()
        n = len(states) - 1
        pieces = [0] * n
        assert (count_segment_crossings(states, pieces, min_separation=min_sep)
                == _ref_count(states, pieces, min_separation=min_sep))

    def test_too_short_returns_zero(self):
        states = _wavy_loop_states(3)
        assert count_segment_crossings(states, [0, 0, 0]) == 0

    def test_none_piece_indices(self):
        states = _pinched_loop_states()
        assert (count_segment_crossings(states, None)
                == _ref_count(states, None))


# =============================================================================
# find_crossing_pairs equivalence (slots, angles AND order)
# =============================================================================

def _assert_pairs_equal(live, ref):
    assert len(live) == len(ref)
    for (li, lj, la), (ri, rj, ra) in zip(live, ref):
        assert (li, lj) == (ri, rj)
        assert la == pytest.approx(ra, abs=1e-9)


class TestFindCrossingPairsEquivalence:

    def test_pinched_loop_pairs_and_order(self):
        states = _pinched_loop_states()
        n = len(states) - 1
        _assert_pairs_equal(
            find_crossing_pairs(states, [0] * n),
            _ref_pairs(states, [0] * n),
        )

    @pytest.mark.parametrize("n", [20, 40, 80, 160])
    def test_wavy_loops(self, n):
        states = _wavy_loop_states(n, waves=9, amp=12.0)
        pieces = [0] * n
        _assert_pairs_equal(
            find_crossing_pairs(states, pieces),
            _ref_pairs(states, pieces),
        )

    def test_exempt_positions_pre_skip(self):
        states = _pinched_loop_states()
        n = len(states) - 1
        ref_plain = _ref_pairs(states, [0] * n)
        assert ref_plain
        i0, j0, _ = ref_plain[0]
        pieces = [0] * n
        pieces[i0] = 3
        pieces[j0] = 6
        _assert_pairs_equal(
            find_crossing_pairs(states, pieces),
            _ref_pairs(states, pieces),
        )

    def test_too_short_returns_empty(self):
        states = _wavy_loop_states(3)
        assert find_crossing_pairs(states, [0, 0, 0]) == []

    @pytest.mark.parametrize("min_sep", [1, 2, 4])
    def test_min_separation_variants(self, min_sep):
        states = _pinched_loop_states()
        n = len(states) - 1
        _assert_pairs_equal(
            find_crossing_pairs(states, [0] * n, min_separation=min_sep),
            _ref_pairs(states, [0] * n, min_separation=min_sep),
        )

    def test_none_piece_indices(self):
        states = _pinched_loop_states()
        _assert_pairs_equal(
            find_crossing_pairs(states, None),
            _ref_pairs(states, None),
        )
