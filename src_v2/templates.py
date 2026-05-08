"""Template definitions for parametric junction expansion (Phase 5a+).

A *template* is a frozen, picklable description of how a single junction
descriptor expands into multiple chromosome slots and port-pair edges. The
decoder's ``_materialize_junctions`` step reads an active junction
``(active=1, anchor_slot, kind, param_a, param_b)`` from the chromosome and
asks the matching template to:

1. Compute branch piece sequence (for FK validation) given ``param_a``.
2. Compute branch endpoint pose given the IN-switch entry pose.
3. Test geometric validity (branch endpoint aligns with OUT-switch port C).
4. Report inventory requirements (per-piece-id counts).

Templates contain NO mutable state, NO RNG, NO closures, and NO logger refs
so they round-trip cleanly through ``multiprocessing.Pool`` workers
(Rules 11, 17). All FK constants are derived from the V2 catalog YAML
(``data/track_pieces_v2.yaml``) -- if the catalog ever changes, update
the constants below in lockstep.

Phase 5a ships :data:`PASSING_SIDING_LEFT` and :data:`PASSING_SIDING_RIGHT`
only. Phase 6a adds ``FIGURE_8_CROSS``; Phase 7a adds ``PARALLEL_DC_BRIDGE``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from .encoding import (
    JUNCTION_KIND_FIGURE_8_CROSS,
    JUNCTION_KIND_PARALLEL_DC_BRIDGE,
    JUNCTION_KIND_PASSING_SIDING,
)
from .se2 import Pose, pose_compose


# =============================================================================
# Catalog-derived FK constants (must mirror data/track_pieces_v2.yaml exactly)
# =============================================================================

# R40_CURVE port B in piece-local frame (radius 40 studs, sector pi/8 = 22.5 deg).
# flip=0 -> right turn (-22.5 deg), flip=1 -> left turn (+22.5 deg).
# Computed: dx = R*sin(theta), dy = R*(1-cos(theta)) for left turn.
_R40_LEFT_FK: Pose = (
    40.0 * math.sin(math.pi / 8),         # dx ~ 15.307
    40.0 * (1.0 - math.cos(math.pi / 8)), # dy ~ 3.045
    math.pi / 8,                          # +22.5 deg
)
_R40_RIGHT_FK: Pose = (_R40_LEFT_FK[0], -_R40_LEFT_FK[1], -_R40_LEFT_FK[2])

# STRAIGHT_16 axial extent.
_STRAIGHT_16_FK: Pose = (16.0, 0.0, 0.0)

# Switch port C poses (from the catalog YAML; 32-stud switches).
# R40_SWITCH_LEFT.C: (31.0, 6.2, +pi/8) -- diverges to +y at +22.5 deg.
# R40_SWITCH_RIGHT.C: (31.0, -6.2, -pi/8) -- diverges to -y at -22.5 deg.
_SWITCH_LEFT_DIVERGE_FK: Pose = (31.0, 6.2, math.pi / 8)
_SWITCH_RIGHT_DIVERGE_FK: Pose = (31.0, -6.2, -math.pi / 8)

# Switch port C poses AFTER rotate=1 (180-deg rotation around piece body center).
# Decoder convention: ``(dx, dy, dtheta) -> (L - dx, -dy, dtheta + pi)`` where
# ``L = body_length = 32`` for switches. Used by the OUT switch in a passing
# siding (rotate=1 places the frog on the throat side).
_SWITCH_LEFT_DIVERGE_FK_ROTATED: Pose = (
    32.0 - 31.0, -6.2, math.pi / 8 + math.pi,
)  # (1.0, -6.2, +9pi/8) -- LEFT switch port C after 180-deg rotation
_SWITCH_RIGHT_DIVERGE_FK_ROTATED: Pose = (
    32.0 - 31.0, 6.2, -math.pi / 8 + math.pi,
)  # (1.0, +6.2, +7pi/8) -- RIGHT switch port C after 180-deg rotation


# =============================================================================
# Template dataclass
# =============================================================================


@dataclass(frozen=True)
class PassingSidingTemplate:
    """Geometry + inventory description of one passing-siding orientation.

    A passing siding is two switches plus a parallel branch that rejoins
    the mainline. V2 unifies switch handedness via the ``rotate`` bit:
    rotate=0 places the switch in IN orientation (port C diverges away
    from mainline), rotate=1 places it 180-degree-rotated as OUT
    (port C diverges back toward mainline). The IN and OUT switches use
    OPPOSITE handedness so port C lands on the same side of the mainline
    on entry and exit (see _emit_simple_oval_with_siding for derivation).
    """

    name: str
    handedness: str                     # "LEFT" (branch on +y) or "RIGHT" (-y)
    in_switch_id: str                   # "R40_SWITCH_LEFT" or "R40_SWITCH_RIGHT"
    in_switch_rotate: int               # 0 = IN placement
    out_switch_id: str
    out_switch_rotate: int              # 1 = OUT placement (180-deg rotated)
    branch_curve_id: str                # "R40_CURVE"
    branch_curve_flip: int              # 0 = right turn, 1 = left turn
    straight_id: str                    # "STRAIGHT_16"
    diverge_fk: Pose                    # IN switch port C in piece-local frame
    merge_fk: Pose                      # OUT switch port C (same side of mainline)


# =============================================================================
# Module-level template constants (Rule 11: frozen, no lazy construction)
# =============================================================================

PASSING_SIDING_LEFT = PassingSidingTemplate(
    name="passing_siding_left",
    handedness="LEFT",
    # IN: LEFT switch with rotate=0 -> port C at NE, +pi/8 outward.
    in_switch_id="R40_SWITCH_LEFT",
    in_switch_rotate=0,
    # OUT: RIGHT switch rotate=1 -> port C at NW, +7pi/8 outward (back into +y).
    out_switch_id="R40_SWITCH_RIGHT",
    out_switch_rotate=1,
    # Branch descends from +pi/8 (after IN.C) back to -pi/8 (before OUT.C):
    # uses R40_CURVE flip=0 (right-turning, dtheta=-pi/8 each).
    branch_curve_id="R40_CURVE",
    branch_curve_flip=0,
    straight_id="STRAIGHT_16",
    diverge_fk=_SWITCH_LEFT_DIVERGE_FK,
    merge_fk=_SWITCH_RIGHT_DIVERGE_FK_ROTATED,   # OUT = RIGHT switch rotate=1
)

PASSING_SIDING_RIGHT = PassingSidingTemplate(
    name="passing_siding_right",
    handedness="RIGHT",
    in_switch_id="R40_SWITCH_RIGHT",
    in_switch_rotate=0,
    out_switch_id="R40_SWITCH_LEFT",
    out_switch_rotate=1,
    branch_curve_id="R40_CURVE",
    branch_curve_flip=1,                # left-turning to ascend then descend
    straight_id="STRAIGHT_16",
    diverge_fk=_SWITCH_RIGHT_DIVERGE_FK,
    merge_fk=_SWITCH_LEFT_DIVERGE_FK_ROTATED,    # OUT = LEFT switch rotate=1
)

PASSING_SIDING_TEMPLATES: Dict[int, Tuple[PassingSidingTemplate, ...]] = {
    JUNCTION_KIND_PASSING_SIDING: (PASSING_SIDING_LEFT, PASSING_SIDING_RIGHT),
}
"""Map kind -> tuple of orientation variants. Phase 5c picks one variant
via ``param_b`` (0=LEFT, 1=RIGHT)."""


# =============================================================================
# Figure-8 template (Phase 6a)
# =============================================================================
#
# A figure-8 layout is two cycles sharing a CROSS_90 piece. The cross's
# horizontal route (A-B) carries the mainline; the vertical route (C-D)
# carries the secondary lobe. Phase 6a materializes the secondary lobe
# parametrically: replace the anchor with CROSS_90, then splice 16 R40 +
# 2*m STR forming a stadium that connects port D back to port C externally.
#
# CROSS_90 port poses in piece-local frame (from data/track_pieces_v2.yaml):
#   C: (8, -8, -pi/2) -- south edge, outward heading -pi/2 (going south)
#   D: (8,  8, +pi/2) -- north edge, outward heading +pi/2 (going north)
# A train exits port D heading +y (north into the lobe) and re-enters
# port C heading +y (north into the cross from south).

_CROSS_PORT_C_FK: Pose = (8.0, -8.0, -math.pi / 2)
_CROSS_PORT_D_FK: Pose = (8.0, 8.0, math.pi / 2)


@dataclass(frozen=True)
class Figure8Template:
    """Geometry + inventory for one figure-8 lobe orientation.

    The lobe is a 16-R40 secondary cycle attached to the cross's C and D
    ports. Same-handed R40s give a closed 360-degree path; param_a (=
    n_straights) lets the lobe stretch axially so it doesn't collide with
    the mainline. param_b selects lobe handedness (0 = right-turning,
    1 = left-turning) -- visually flips the lobe to north or south of the
    cross.
    """

    name: str
    cross_id: str                       # "CROSS_90"
    cross_rotate: int                   # 0 -- cross is symmetric under rotation
    lobe_curve_id: str                  # "R40_CURVE"
    lobe_curve_flip: int                # 0 = right turn, 1 = left turn
    straight_id: str                    # "STRAIGHT_16"
    port_c_fk: Pose                     # cross's port C local pose
    port_d_fk: Pose                     # cross's port D local pose


FIGURE_8_LEFT_LOBE = Figure8Template(
    name="figure_8_left_lobe",
    cross_id="CROSS_90",
    cross_rotate=0,
    lobe_curve_id="R40_CURVE",
    lobe_curve_flip=1,                  # left-turning -> lobe to north of cross
    straight_id="STRAIGHT_16",
    port_c_fk=_CROSS_PORT_C_FK,
    port_d_fk=_CROSS_PORT_D_FK,
)

FIGURE_8_RIGHT_LOBE = Figure8Template(
    name="figure_8_right_lobe",
    cross_id="CROSS_90",
    cross_rotate=0,
    lobe_curve_id="R40_CURVE",
    lobe_curve_flip=0,                  # right-turning -> lobe to south of cross
    straight_id="STRAIGHT_16",
    port_c_fk=_CROSS_PORT_C_FK,
    port_d_fk=_CROSS_PORT_D_FK,
)

FIGURE_8_TEMPLATES: Dict[int, Tuple[Figure8Template, ...]] = {
    JUNCTION_KIND_FIGURE_8_CROSS: (FIGURE_8_LEFT_LOBE, FIGURE_8_RIGHT_LOBE),
}
"""Map kind -> figure-8 orientation variants. ``param_b`` picks LEFT (0)
or RIGHT (1) lobe."""


# Figure-8 lobe geometry: 16 same-handed R40 = 360 degrees = closed loop.
# Adding ``n_straights`` STR after each 8-R40 bank stretches the loop
# axially without breaking closure (the resulting shape is a stadium).
_LOBE_R40_COUNT: int = 16
_LOBE_BANK_SIZE: int = 8


def compute_lobe_pieces(
    template: Figure8Template, n_straights: int,
) -> Tuple[Tuple[str, int, int], ...]:
    """Lobe piece sequence: ``[R40 x 8, STR x m, R40 x 8, STR x m]``.

    16 same-handed R40s close the angular budget; the 2*m STRs stretch the
    stadium so it sits adjacent to the cross instead of overlapping it.
    """
    curve = (template.lobe_curve_id, template.lobe_curve_flip, 0)
    straight = (template.straight_id, 0, 0)
    m = max(0, int(n_straights))
    return (
        *(curve for _ in range(_LOBE_BANK_SIZE)),
        *(straight for _ in range(m)),
        *(curve for _ in range(_LOBE_BANK_SIZE)),
        *(straight for _ in range(m)),
    )


def compute_lobe_endpoint(
    cross_world_pose: Pose,
    template: Figure8Template,
    n_straights: int,
) -> Pose:
    """Walk FK from the cross's port D, through the lobe, back to port C
    (world frame). Used by :func:`is_valid_figure8`."""
    curve_fk = (
        (40.0 * math.sin(math.pi / 8),
         40.0 * (1.0 - math.cos(math.pi / 8)),
         math.pi / 8)
        if template.lobe_curve_flip == 1
        else (40.0 * math.sin(math.pi / 8),
              -40.0 * (1.0 - math.cos(math.pi / 8)),
              -math.pi / 8)
    )
    straight_fk: Pose = (16.0, 0.0, 0.0)
    state = pose_compose(cross_world_pose, template.port_d_fk)
    for _ in range(_LOBE_BANK_SIZE):
        state = pose_compose(state, curve_fk)
    for _ in range(max(0, int(n_straights))):
        state = pose_compose(state, straight_fk)
    for _ in range(_LOBE_BANK_SIZE):
        state = pose_compose(state, curve_fk)
    for _ in range(max(0, int(n_straights))):
        state = pose_compose(state, straight_fk)
    return state


def is_valid_figure8(
    cross_world_pose: Pose,
    template: Figure8Template,
    n_straights: int,
    position_tolerance: float = 4.0,
    angle_tolerance_deg: float = 5.0,
) -> bool:
    """The lobe must close: walking from port D through 16 R40 + 2*m STR
    returns to within tolerance of port C's world pose (entering the cross
    from the south, anti-parallel to C's outward heading)."""
    end_pose = compute_lobe_endpoint(cross_world_pose, template, n_straights)
    port_c_world = pose_compose(cross_world_pose, template.port_c_fk)
    dx = end_pose[0] - port_c_world[0]
    dy = end_pose[1] - port_c_world[1]
    pos_err = math.hypot(dx, dy)
    angle_err_deg = math.degrees(abs(
        _normalize_angle(end_pose[2] - port_c_world[2] + math.pi)
    ))
    return pos_err <= position_tolerance and angle_err_deg <= angle_tolerance_deg


def get_figure8_inventory_requirements(
    template: Figure8Template, n_straights: int,
) -> Dict[str, int]:
    """Per-piece-id counts needed to materialize a single figure-8 lobe."""
    reqs: Dict[str, int] = {}
    reqs[template.cross_id] = reqs.get(template.cross_id, 0) + 1
    reqs[template.lobe_curve_id] = (
        reqs.get(template.lobe_curve_id, 0) + _LOBE_R40_COUNT
    )
    if n_straights > 0:
        reqs[template.straight_id] = (
            reqs.get(template.straight_id, 0) + 2 * int(n_straights)
        )
    return reqs


def check_figure8_inventory(
    template: Figure8Template,
    n_straights: int,
    available: Dict[str, int],
    used: Dict[str, int],
) -> bool:
    """True iff the figure-8's pieces still fit in remaining inventory."""
    for piece_id, needed in get_figure8_inventory_requirements(
        template, n_straights,
    ).items():
        if used.get(piece_id, 0) + needed > available.get(piece_id, 0):
            return False
    return True


# =============================================================================
# Parallel DC bridge template (Phase 7a, minimal scope)
# =============================================================================
#
# The DOUBLE_CROSSOVER (DC) is a 4-port piece (48 stud long, 16 stud port
# spacing) that lets two parallel tracks switch lanes mid-run. Per the
# plan's Phase 7a spec, *parallel-section detection* (the materializer
# step that identifies which two main-loop straights are parallel and
# fits a DC across them) is "non-trivial" and "high risk". Until Phase 6a
# clears [?] state, Phase 7a ships only the template + dispatch
# infrastructure: the materializer's job is to validate the anchor and
# replace its piece with DOUBLE_CROSSOVER. The Phase 7b heuristic seed
# (when implemented) is responsible for delivering chromosomes whose
# port-pair edges already wire a sensible parallel-track context.
#
# DOUBLE_CROSSOVER ports (from data/track_pieces_v2.yaml):
#   A: (0, 0, 0)    -- entry track1 (west, lane 1)
#   B: (48, 0, 0)   -- exit track1 (east, lane 1)
#   C: (0, 16, 0)   -- entry track2 (west, lane 2, 16 stud north of lane 1)
#   D: (48, 16, 0)  -- exit track2 (east, lane 2)


@dataclass(frozen=True)
class ParallelBridgeTemplate:
    """Minimal Phase 7a descriptor: piece id + rotate bit only.

    The DOUBLE_CROSSOVER's geometry is fully specified by its catalog
    entry; no FK constants need duplicating into the template. Future
    Phase 7 refinements may add more parameters (e.g. handedness for the
    cross routes, parallel-section reservation length) here."""

    name: str
    dc_id: str          # "DOUBLE_CROSSOVER"
    dc_rotate: int      # 0 or 1 (rotatable=False in catalog -> always 0)


PARALLEL_DC_BRIDGE_PRIMARY = ParallelBridgeTemplate(
    name="parallel_dc_bridge_primary",
    dc_id="DOUBLE_CROSSOVER",
    dc_rotate=0,
)


PARALLEL_DC_BRIDGE_TEMPLATES: Dict[int, Tuple[ParallelBridgeTemplate, ...]] = {
    JUNCTION_KIND_PARALLEL_DC_BRIDGE: (PARALLEL_DC_BRIDGE_PRIMARY,),
}


def get_dc_bridge_inventory_requirements(
    template: ParallelBridgeTemplate,
) -> Dict[str, int]:
    """One DOUBLE_CROSSOVER per active bridge junction. The seeded
    parallel-track straights aren't claimed here; the seed counts them
    against the chromosome's slot region."""
    return {template.dc_id: 1}


def check_dc_bridge_inventory(
    template: ParallelBridgeTemplate,
    available: Dict[str, int],
    used: Dict[str, int],
) -> bool:
    """True iff a DOUBLE_CROSSOVER piece is still free."""
    for piece_id, needed in get_dc_bridge_inventory_requirements(template).items():
        if used.get(piece_id, 0) + needed > available.get(piece_id, 0):
            return False
    return True


# =============================================================================
# Branch geometry helpers
# =============================================================================


def compute_branch_pieces(
    template: PassingSidingTemplate, n_straights: int,
) -> Tuple[Tuple[str, int, int], ...]:
    """Return the branch piece sequence as ``(piece_id, flip, rotate)`` triples.

    Branch structure: ``[approach_curve, straight x N, return_curve]`` where
    both curves use ``branch_curve_flip``. The returned tuple is suitable for
    splicing into the chromosome's slot region (one slot per element).
    """
    curve = (template.branch_curve_id, template.branch_curve_flip, 0)
    straight = (template.straight_id, 0, 0)
    return (curve, *(straight for _ in range(max(0, int(n_straights)))), curve)


def compute_branch_endpoint(
    in_switch_state: Pose,
    template: PassingSidingTemplate,
    n_straights: int,
) -> Pose:
    """Walk FK from IN-switch entry to the branch's last-piece exit pose.

    Composes IN-switch diverge -> approach curve -> N straights -> return
    curve. Uses ``pose_compose`` (radians, V2 convention) -- distinct from
    V1's degrees-based implementation.
    """
    # Map (piece_id, flip) -> piece-local FK delta (port A -> port B).
    state = pose_compose(in_switch_state, template.diverge_fk)
    curve_fk = (
        _R40_LEFT_FK if template.branch_curve_flip == 1 else _R40_RIGHT_FK
    )
    state = pose_compose(state, curve_fk)
    for _ in range(max(0, int(n_straights))):
        state = pose_compose(state, _STRAIGHT_16_FK)
    state = pose_compose(state, curve_fk)
    return state


def compute_required_main_distance(
    template: PassingSidingTemplate, n_straights: int,
) -> float:
    """X-distance the siding spans along the mainline axis.

    Used to locate the OUT-switch slot when walking cycle edges from the
    anchor: the OUT switch must be roughly this many studs ahead. Computed
    by walking the branch from origin and reading the X coordinate at exit.
    """
    end = compute_branch_endpoint((0.0, 0.0, 0.0), template, n_straights)
    return float(end[0])


def is_valid_siding(
    in_switch_state: Pose,
    out_switch_state: Pose,
    template: PassingSidingTemplate,
    n_straights: int,
    position_tolerance: float = 2.0,
    angle_tolerance_deg: float = 5.0,
) -> bool:
    """Geometric closure test: branch endpoint aligns with OUT-switch port C.

    Returns True iff the branch (from IN-switch port C through approach,
    straights, return curve) lands within ``position_tolerance`` studs and
    ``angle_tolerance_deg`` degrees of the OUT-switch port C world pose.
    """
    branch_end = compute_branch_endpoint(in_switch_state, template, n_straights)
    out_port_c = pose_compose(out_switch_state, template.merge_fk)
    dx = branch_end[0] - out_port_c[0]
    dy = branch_end[1] - out_port_c[1]
    pos_err = math.hypot(dx, dy)
    # The branch arrives INTO OUT's port C (anti-parallel to OUT.C's outward
    # heading), so the closure-aligned angle is the diff offset by pi.
    angle_err_deg = math.degrees(abs(
        _normalize_angle(branch_end[2] - out_port_c[2] + math.pi)
    ))
    return pos_err <= position_tolerance and angle_err_deg <= angle_tolerance_deg


def _normalize_angle(theta: float) -> float:
    """Wrap radians into [-pi, pi]."""
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


# =============================================================================
# Inventory helpers (piece_id keyed, V2 convention)
# =============================================================================


def get_siding_inventory_requirements(
    template: PassingSidingTemplate, n_straights: int,
) -> Dict[str, int]:
    """Per-piece-id counts needed to materialize a single passing siding."""
    reqs: Dict[str, int] = {}
    reqs[template.in_switch_id] = reqs.get(template.in_switch_id, 0) + 1
    reqs[template.out_switch_id] = reqs.get(template.out_switch_id, 0) + 1
    # Branch curves: 2 per siding (approach + return).
    reqs[template.branch_curve_id] = reqs.get(template.branch_curve_id, 0) + 2
    if n_straights > 0:
        reqs[template.straight_id] = (
            reqs.get(template.straight_id, 0) + int(n_straights)
        )
    return reqs


def check_siding_inventory(
    template: PassingSidingTemplate,
    n_straights: int,
    available: Dict[str, int],
    used: Dict[str, int],
) -> bool:
    """True iff the siding's pieces still fit in remaining inventory."""
    for piece_id, needed in get_siding_inventory_requirements(
        template, n_straights,
    ).items():
        if used.get(piece_id, 0) + needed > available.get(piece_id, 0):
            return False
    return True
