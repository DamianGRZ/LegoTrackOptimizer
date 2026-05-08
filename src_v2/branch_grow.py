"""Angular-budget-aware A* search for closing a switch's port C back to
another switch's port C on the same ``through`` cycle.

Cellular-Encoding-style branch growth grammar with closure verification.

A* state space: append one of three successors at the chain tail:

- ``STRAIGHT_16`` (FK delta = (+16, 0, 0))
- ``R40_CURVE`` flip=0 (left turn,  delta = (+15.307, +3.045, +π/8))
- ``R40_CURVE`` flip=1 (right turn, delta = (+15.307, −3.045, −π/8))

Heuristic ``h(s) = euclidean(s.pose, target) / 16`` is admissible (one piece
covers ≤ 16 studs of straight-line progress) and consistent (triangle
inequality), so A* with a closed set on quantized state finds an optimal-
length closure when one exists in budget.

Mating convention follows the V2 decoder: a successor's port A world pose
coincides with the parent's port B world pose, both facing the same
direction (no 180° flip — see ``decoder.py`` file header).

Pure function — no chromosome mutation. The mutation operator wraps it.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from .catalog import TrackCatalog
from .se2 import pose_compose


# Catalog piece-IDs the branch-growth A* is allowed to place.
# Switches and crossings are not branch material.
_BRANCH_PIECE_IDS: tuple = ("STRAIGHT_16", "R40_CURVE")


class BranchStep(NamedTuple):
    """A single piece in the closing branch path.

    ``flip`` is a Y-mirror bit (only meaningful for symmetric pieces such
    as R40_CURVE). ``rotate`` is reserved for future rotatable pieces and
    is always 0 for straights and curves.
    """

    piece_id: str
    flip: int
    rotate: int


class _Successor(NamedTuple):
    """Pre-computed FK delta for one valid A* successor action."""

    piece_id: str
    flip: int
    rotate: int
    delta: tuple  # (dx, dy, dtheta) in piece-local frame, flip already applied


class _State(NamedTuple):
    """A* state. Hashable for closed-set lookup."""

    f: float        # g + h
    g: int          # depth (= pieces placed so far)
    pose: tuple     # (x, y, theta) — quantized for hashing
    inventory: tuple  # sorted ((piece_id, count), ...) — hashable
    path: tuple     # tuple of BranchStep so far


def find_branch_path(
    start_pose: tuple,
    target_pose: tuple,
    inventory: Mapping[str, int],
    catalog: TrackCatalog,
    *,
    max_depth: int,
    tolerance: float,
    rng: np.random.Generator,
) -> list | None:
    """A*-search a piece sequence whose end-pose ≈ ``target_pose``.

    Args:
        start_pose: world pose of the source port (e.g. IN switch port C).
        target_pose: world pose the chain end must match (e.g. OUT switch
            port C). Position is checked under ``tolerance``; orientation
            is not separately tested but the heuristic biases toward it
            via the position metric.
        inventory: piece-id → remaining count budget (after accounting for
            pieces already in the chromosome).
        catalog: V2 :class:`TrackCatalog` (used for FK port-B offsets).
        max_depth: hard cap on branch piece count (typical: 16).
        tolerance: stud distance under which the goal is considered
            reached. Match to ``branch_closure_tolerance`` (default 8.0).
        rng: numpy generator for tie-breaking on equal-f successors.

    Returns:
        A list of :class:`BranchStep`, or ``None`` if no closure was found
        within the depth + inventory budget.
    """
    if catalog.spec is None:
        return None

    successors = _build_successor_specs(catalog)
    start_q = _quantize_pose(start_pose)
    target_xy = (target_pose[0], target_pose[1])

    initial_inv = _sorted_inv(inventory)
    initial_h = _h_pose(start_pose, target_pose)
    initial = _State(
        f=initial_h,
        g=0,
        pose=start_q,
        inventory=initial_inv,
        path=(),
    )

    # Trivial closure: already at target pose.
    if _euclidean_xy(start_q, target_xy) <= tolerance:
        return list(initial.path)

    heap: list = [(initial.f, float(rng.random()), initial)]
    closed: set = set()

    while heap:
        _f, _tie, state = heapq.heappop(heap)
        key = (state.pose, state.inventory, state.g)
        if key in closed:
            continue
        closed.add(key)

        if _euclidean_xy(state.pose, target_xy) <= tolerance:
            return list(state.path)

        if state.g >= max_depth:
            continue

        inv_dict = dict(state.inventory)
        for succ in successors:
            if inv_dict.get(succ.piece_id, 0) <= 0:
                continue
            new_pose_raw = pose_compose(state.pose, succ.delta)
            new_pose_q = _quantize_pose(new_pose_raw)
            new_inv = _decrement_inv(state.inventory, succ.piece_id)
            new_g = state.g + 1
            new_h = _h_pose(new_pose_raw, target_pose)
            new_state = _State(
                f=new_g + new_h,
                g=new_g,
                pose=new_pose_q,
                inventory=new_inv,
                path=state.path + (BranchStep(succ.piece_id, succ.flip, succ.rotate),),
            )
            new_key = (new_state.pose, new_state.inventory, new_state.g)
            if new_key in closed:
                continue
            heapq.heappush(heap, (new_state.f, float(rng.random()), new_state))

    return None


# =============================================================================
# Helpers
# =============================================================================


def _h_pose(pose, target) -> float:
    """Admissible, consistent heuristic: euclidean distance / 16.

    A single piece covers AT MOST 16 studs of forward progress
    (STRAIGHT_16 = 16 exactly; R40_CURVE chord ≈ 15.5). So ``h ≤ pieces
    remaining`` and the triangle inequality on euclidean distance gives
    consistency: ``h(s) ≤ 1 + h(s')`` for any successor s' reached by one
    piece.
    """
    return _euclidean_xy(pose, target) / 16.0


def _euclidean_xy(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _quantize_pose(pose) -> tuple:
    """Round to quarter-stud x/y and 4-decimal theta — makes float poses
    hashable for the closed set without losing meaningful precision."""
    return (
        round(pose[0] * 4) / 4,
        round(pose[1] * 4) / 4,
        round(pose[2], 4),
    )


def _sorted_inv(inv: Mapping[str, int]) -> tuple:
    """Hashable sorted-tuple form, dropping zero-count entries."""
    return tuple(sorted((pid, c) for pid, c in inv.items() if c > 0))


def _decrement_inv(inv_tuple: tuple, piece_id: str) -> tuple:
    """Return a new sorted inventory tuple with one ``piece_id`` deducted.

    Drops the entry entirely when count reaches zero so the canonical
    sorted form is preserved (no zero-count entries in the tuple).
    """
    return tuple(
        (pid, c - 1) if pid == piece_id else (pid, c)
        for pid, c in inv_tuple
        if not (pid == piece_id and c <= 1)
    )


def _build_successor_specs(catalog: TrackCatalog) -> list:
    """Three branch-growth successors with FK deltas read from catalog spec.

    The delta is the port-B local pose (port A is the piece origin), with
    the ``flip`` Y-mirror applied for the right-turn curve variant.
    """
    spec = catalog.spec
    out: list = []
    for piece_id in _BRANCH_PIECE_IDS:
        ps = spec.by_id.get(piece_id) if spec else None
        if ps is None or "B" not in ps.ports:
            continue
        port_b = ps.ports["B"]
        if piece_id == "R40_CURVE":
            # Two curve variants: flip=0 (left turn) and flip=1 (right turn).
            out.append(_Successor(
                piece_id=piece_id, flip=0, rotate=0,
                delta=(port_b.dx, port_b.dy, port_b.dtheta),
            ))
            out.append(_Successor(
                piece_id=piece_id, flip=1, rotate=0,
                delta=(port_b.dx, -port_b.dy, -port_b.dtheta),
            ))
        else:
            out.append(_Successor(
                piece_id=piece_id, flip=0, rotate=0,
                delta=(port_b.dx, port_b.dy, port_b.dtheta),
            ))
    return out
