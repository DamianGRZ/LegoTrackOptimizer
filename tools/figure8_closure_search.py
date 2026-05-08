"""Phase 6a closure search v2: enumerate figure-8 lobe parametrizations
with straights anchored at BOTH cross ports.

Per the user-supplied geometric insight, a figure-8 lobe between two
adjacent (diagonal-quadrant) ports of CROSS_90 is shaped like a
passing-siding branch: STRs extend the port outward, R40 curves then
turn the chain to align with the next port's extension, more STRs bring
it home. This script enumerates::

    [STR x M1] + [curve sequence] + [STR x M2]

over ``M1, M2 in [0, max_str)`` and curve-sequence patterns indexed by
total R40 count + handedness/interleaving. The "curve sequence" itself
is sub-enumerated over a richer space than the v1 search:

- ``n_R40 in [4, 16]``
- ``handedness`` is enumerated over ALL 2^n_R40 binary L/R strings up to
  a cap (n_R40 = 4..8 enumerated fully, larger n_R40 sampled by the
  4-segment patterns from the previous version to stay tractable)
- inner straights distributed at every gap between consecutive R40s

Output: for each diagonal lobe pairing (B->C, B->D, D->A, C->A), the
count of closing tuples and the first 10 closing tuples with residuals.

Usage::

    .venv/Scripts/python.exe tools/figure8_closure_search.py
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src_v2.catalog import TrackCatalog  # noqa: E402
from src_v2.se2 import Pose, pose_compose  # noqa: E402


_POS_TOL: float = 2.0
_ANGLE_TOL_DEG: float = 2.0


def _normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta <= -math.pi:
        theta += 2 * math.pi
    return theta


def _propagate_fk(start: Pose, deltas: Iterable[Pose]) -> Pose:
    state = start
    for d in deltas:
        state = pose_compose(state, d)
    return state


def _residual(end: Pose, target: Pose) -> Tuple[float, float, float, bool]:
    dx = end[0] - target[0]
    dy = end[1] - target[1]
    dtheta = _normalize_angle(end[2] - target[2])
    closed = (
        abs(dx) < _POS_TOL
        and abs(dy) < _POS_TOL
        and abs(math.degrees(dtheta)) < _ANGLE_TOL_DEG
    )
    return dx, dy, dtheta, closed


def _enumerate_curve_strings(n_r40: int) -> Iterable[str]:
    """For small n_r40 (<= 8), enumerate all 2^n_r40 L/R strings.
    For larger n_r40, restrict to 4-segment same-handedness-per-segment
    patterns to stay tractable."""
    if n_r40 <= 8:
        for bits in itertools.product("LR", repeat=n_r40):
            yield "".join(bits)
        return
    seg_size = n_r40 // 4
    rem = n_r40 % 4
    seg_sizes = [seg_size + (1 if i < rem else 0) for i in range(4)]
    for pattern in itertools.product("LR", repeat=4):
        s = "".join(c * sz for c, sz in zip(pattern, seg_sizes))
        yield s


def _curve_chain_deltas(
    handedness: str, r40_l: Pose, r40_r: Pose, str_fk: Pose, inner_strs: Tuple[int, ...],
) -> List[Pose]:
    """Build a list of FK deltas for a curve sequence with optional STR
    insertions at every gap between consecutive R40s. ``inner_strs`` has
    length ``len(handedness) - 1`` if non-empty; otherwise no insertions."""
    deltas: List[Pose] = []
    for i, h in enumerate(handedness):
        deltas.append(r40_l if h == "L" else r40_r)
        if inner_strs and i < len(inner_strs):
            for _ in range(inner_strs[i]):
                deltas.append(str_fk)
    return deltas


def _search_pairing(
    label: str,
    start: Pose,
    target: Pose,
    r40_l: Pose,
    r40_r: Pose,
    str_fk: Pose,
    n_r40_max: int,
    outer_str_max: int,
    inner_str_max: int,
) -> Tuple[int, List[Tuple[int, int, int, str, Tuple[int, ...], Tuple[float, float, float]]]]:
    """Search closures of form ``[STR x M1] + curves + [STR x M2]`` plus an
    optional uniform inner-STR count between curves."""
    tested = 0
    closing: List[Tuple[int, int, int, str, Tuple[int, ...], Tuple[float, float, float]]] = []

    for n_r40 in range(2, n_r40_max + 1):
        for handedness in _enumerate_curve_strings(n_r40):
            # Inner-STR count per gap (uniform across gaps to bound enumeration).
            for inner_str_each in range(0, inner_str_max + 1):
                inner_strs = tuple(inner_str_each for _ in range(n_r40 - 1))
                curve_deltas = _curve_chain_deltas(
                    handedness, r40_l, r40_r, str_fk, inner_strs,
                )
                for m1 in range(0, outer_str_max + 1):
                    for m2 in range(0, outer_str_max + 1):
                        tested += 1
                        deltas = (
                            [str_fk] * m1
                            + curve_deltas
                            + [str_fk] * m2
                        )
                        end = _propagate_fk(start, deltas)
                        dx, dy, dtheta, ok = _residual(end, target)
                        if ok:
                            closing.append((
                                m1, m2, inner_str_each, handedness,
                                inner_strs, (dx, dy, dtheta),
                            ))
    return tested, closing


def main() -> None:
    catalog = TrackCatalog.load(_PROJECT_ROOT / "data" / "track_pieces_v2.yaml")
    spec = catalog.spec
    if spec is None:
        raise RuntimeError("V2 catalog spec required")
    cross_spec = spec.by_id["CROSS_90"]
    r40_spec = spec.by_id["R40_CURVE"]
    str_spec = spec.by_id["STRAIGHT_16"]

    port_a = cross_spec.ports["A"]
    port_b = cross_spec.ports["B"]
    port_c = cross_spec.ports["C"]
    port_d = cross_spec.ports["D"]

    r40_b = r40_spec.ports["B"]
    r40_l: Pose = (float(r40_b.dx), float(r40_b.dy), float(r40_b.dtheta))
    r40_r: Pose = (float(r40_b.dx), -float(r40_b.dy), -float(r40_b.dtheta))
    str_b = str_spec.ports["B"]
    str_fk: Pose = (float(str_b.dx), float(str_b.dy), float(str_b.dtheta))

    pairings: Dict[str, Tuple[Pose, Pose]] = {
        "B->C (lower-right)": (
            (float(port_b.dx), float(port_b.dy), float(port_b.dtheta)),
            (float(port_c.dx), float(port_c.dy), float(port_c.dtheta)),
        ),
        "B->D (upper-right)": (
            (float(port_b.dx), float(port_b.dy), float(port_b.dtheta)),
            (float(port_d.dx), float(port_d.dy), float(port_d.dtheta)),
        ),
        "D->A (upper-left)": (
            (float(port_d.dx), float(port_d.dy), float(port_d.dtheta)),
            (float(port_a.dx), float(port_a.dy), float(port_a.dtheta)),
        ),
        "C->A (lower-left)": (
            (float(port_c.dx), float(port_c.dy), float(port_c.dtheta)),
            (float(port_a.dx), float(port_a.dy), float(port_a.dtheta)),
        ),
    }

    print("=" * 72)
    print("Figure-8 lobe closure search v2 (port-extension + curve chain)")
    print("=" * 72)
    print("CROSS_90 ports (piece-local frame):")
    for label, p in [("A", port_a), ("B", port_b), ("C", port_c), ("D", port_d)]:
        print(f"  port {label}: x={float(p.dx):+7.3f}, y={float(p.dy):+7.3f}, "
              f"theta={math.degrees(float(p.dtheta)):+7.3f} deg")
    print(f"R40_L: dx={r40_l[0]:.3f}, dy={r40_l[1]:.3f}, "
          f"dtheta={math.degrees(r40_l[2]):+.3f} deg")
    print(f"R40_R: dx={r40_r[0]:.3f}, dy={r40_r[1]:.3f}, "
          f"dtheta={math.degrees(r40_r[2]):+.3f} deg")
    print(f"STR16: dx={str_fk[0]:.3f}, dy={str_fk[1]:.3f}, "
          f"dtheta={math.degrees(str_fk[2]):+.3f} deg")
    print(f"Tolerance: |dx|<{_POS_TOL}, |dy|<{_POS_TOL}, "
          f"|dtheta|<{_ANGLE_TOL_DEG} deg")
    print()

    # Search bounds (wide v3 search): n_R40 up to 8 enumerates all 2^8 = 256
    # handedness strings; n_R40 9..24 falls back to 4-segment patterns.
    # Outer M1/M2 up to 24 each side; inner per-gap STR up to 8.
    n_r40_max = 24
    outer_str_max = 24
    inner_str_max = 8

    grand_tested = 0
    grand_closing = 0

    for label, (start, target) in pairings.items():
        print("-" * 72)
        print(f"Lobe pairing: {label}")
        print(f"  start:  x={start[0]:+7.3f}, y={start[1]:+7.3f}, "
              f"theta={math.degrees(start[2]):+7.3f} deg")
        print(f"  target: x={target[0]:+7.3f}, y={target[1]:+7.3f}, "
              f"theta={math.degrees(target[2]):+7.3f} deg")
        tested, closing = _search_pairing(
            label, start, target, r40_l, r40_r, str_fk,
            n_r40_max=n_r40_max,
            outer_str_max=outer_str_max,
            inner_str_max=inner_str_max,
        )
        grand_tested += tested
        grand_closing += len(closing)
        print(f"  tuples tested:  {tested}")
        print(f"  closing tuples: {len(closing)}")
        if closing:
            n_show = min(10, len(closing))
            print(f"  first {n_show}:")
            for m1, m2, inner, hand, _inner_strs, (dx, dy, dtheta) in closing[:n_show]:
                n_str_total = m1 + m2 + inner * (len(hand) - 1)
                print(f"    M1={m1:>2}, M2={m2:>2}, inner_per_gap={inner}, "
                      f"hand={hand} (len={len(hand)}, total_STR={n_str_total}): "
                      f"dx={dx:+8.4f}, dy={dy:+8.4f}, "
                      f"dtheta={math.degrees(dtheta):+8.4f} deg")
        print()

    print("=" * 72)
    print(f"GRAND TOTAL tuples tested: {grand_tested}")
    print(f"GRAND TOTAL closing tuples: {grand_closing}")
    print("=" * 72)


if __name__ == "__main__":
    main()
