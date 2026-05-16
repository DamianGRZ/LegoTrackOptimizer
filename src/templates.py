"""Template-based passing siding definitions for guaranteed geometric closure.

Standard LEGO passing siding pattern:
    [IN_switch] -> [approach_curve] -> [straights×N] -> [return_curve] -> [OUT_switch]

The template approach guarantees that branch paths geometrically connect
the IN switch diverge exit to the OUT switch merge entry, eliminating
the ~0% feasibility rate of random branch piece placement.

Key insight from Monty's Trains (Track Planning Part 1):
    "a curved track per switch allows diverging route to align parallel to main route"
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# TEMPLATE DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class PassingSidingTemplate:
    """Defines geometry of a standard passing siding.

    A passing siding diverges from the main line, runs parallel for some
    distance, then merges back. Standard configuration uses one LEFT and one
    RIGHT switch (opposite-handed pair, confirmed by LEGO 9V track-planning
    references). For a siding diverging to side `handedness`:
      - The entry switch matches `handedness` (LEFT siding -> LEFT switch).
      - The exit switch is opposite handedness, installed REVERSED on main.

    R40 curves come from one physical SKU; the per-placement flip bit picks
    direction at decode time. ``*_curve_flip`` is the flip applied to the
    catalog FK when generating that curve slot — flip=0 means LEFT (+22.5°),
    flip=1 means RIGHT (-22.5°).
    """

    name: str
    handedness: str
    approach_curve_idx: int
    approach_curve_flip: int
    return_curve_idx: int
    return_curve_flip: int
    straight_idx: int
    diverge_fk: Tuple[float, float, float]
    merge_fk: Tuple[float, float, float]


# =============================================================================
# STANDARD TEMPLATES
# =============================================================================

# Piece indices (must match src/encoding.py PieceIndex enum)
STRAIGHT_16 = 0
R40_CURVE = 2
R40_SWITCH_LEFT = 4
R40_SWITCH_RIGHT = 5


def _compute_reversed_out_merge_fk(
    port_c_dx: float, port_c_dy: float, port_c_dtheta_deg: float,
) -> Tuple[float, float, float]:
    """Derive C->A traversal FK for an OUT switch installed REVERSED on main.

    Geometry: the OUT switch is rotated 180° around its body center in the
    main loop. Its natural port A (at switch-local (0,0)) ends up at the
    forward end of the switch in main frame; port C (naturally at port_c_dx,
    port_c_dy) ends up on the side facing the siding. The train arrives at
    port C from the siding and exits at port A onto main.

    In MAIN frame, the displacement from port C to port A equals
    (port_c_dx, port_c_dy) — both positive in this representation because
    the 180° rotation flips port C's offset to the OPPOSITE side of port A,
    and we measure C->A as (port_A - port_C) where port A's flipped position
    minus port C's flipped position works out to (port_c_dx, port_c_dy).

    The train arrives heading port_c_dtheta_deg in main (anti-parallel to
    port C's natural exit direction after reversal), and exits heading 0°
    (parallel to main). Result is expressed in the train's entry-local frame.
    """
    # Displacement port C -> port A in MAIN frame (after reversal):
    dx_world = port_c_dx
    dy_world = port_c_dy
    # Train enters port C heading port_c_dtheta_deg in main:
    entry_theta = port_c_dtheta_deg
    c, s = np.cos(np.radians(entry_theta)), np.sin(np.radians(entry_theta))
    # Express world displacement in train entry-local frame (rotate by -entry):
    dx_local = dx_world * c + dy_world * s
    dy_local = -dx_world * s + dy_world * c
    # Train heading change: from entry_theta to 0 = -entry_theta
    dtheta_local = -port_c_dtheta_deg
    return (float(dx_local), float(dy_local), float(dtheta_local))


# Catalog port C values per handedness (must match data/track_pieces_v2.yaml)
_LEFT_C_DX, _LEFT_C_DY, _LEFT_C_DTHETA_DEG = 32.75, 13.0, 22.5
_RIGHT_C_DX, _RIGHT_C_DY, _RIGHT_C_DTHETA_DEG = 32.75, -13.0, -22.5


# A LEFT siding (above main) uses LEFT_IN + RIGHT_OUT (reversed). The train
# exiting LEFT_IN's port C is heading +22.5°; it runs parallel above main;
# returns heading -22.5° to enter the reversed RIGHT_OUT's port C. The OUT
# switch's port C in main frame is at +y (same side as siding) because of the
# reversal; the merge_fk is computed from RIGHT switch's natural port C
# (-13 dy in switch frame, flipped by the reversal).
# LEFT_SIDING (siding above main) historically used R40_RIGHT curves, which
# now correspond to R40_CURVE with flip=1 (negated dy/dtheta).
LEFT_SIDING = PassingSidingTemplate(
    name="left_passing_siding",
    handedness="LEFT",
    approach_curve_idx=R40_CURVE,
    approach_curve_flip=1,
    return_curve_idx=R40_CURVE,
    return_curve_flip=1,
    straight_idx=STRAIGHT_16,
    diverge_fk=(_LEFT_C_DX, _LEFT_C_DY, _LEFT_C_DTHETA_DEG),
    merge_fk=_compute_reversed_out_merge_fk(
        _RIGHT_C_DX, _RIGHT_C_DY, _RIGHT_C_DTHETA_DEG,
    ),
)

# RIGHT_SIDING (siding below main) historically used R40_LEFT curves → flip=0.
RIGHT_SIDING = PassingSidingTemplate(
    name="right_passing_siding",
    handedness="RIGHT",
    approach_curve_idx=R40_CURVE,
    approach_curve_flip=0,
    return_curve_idx=R40_CURVE,
    return_curve_flip=0,
    straight_idx=STRAIGHT_16,
    diverge_fk=(_RIGHT_C_DX, _RIGHT_C_DY, _RIGHT_C_DTHETA_DEG),
    merge_fk=_compute_reversed_out_merge_fk(
        _LEFT_C_DX, _LEFT_C_DY, _LEFT_C_DTHETA_DEG,
    ),
)


def switch_indices_for(template: PassingSidingTemplate) -> Tuple[int, int]:
    """Return (entry_switch_idx, exit_switch_idx) given a template.

    The entry switch matches the siding's handedness; the exit switch is
    the opposite handedness (installed reversed). For a LEFT siding this
    yields (SWITCH_LEFT, SWITCH_RIGHT); for RIGHT it yields (SWITCH_RIGHT,
    SWITCH_LEFT).
    """
    if template.handedness == "LEFT":
        return (R40_SWITCH_LEFT, R40_SWITCH_RIGHT)
    return (R40_SWITCH_RIGHT, R40_SWITCH_LEFT)

# Template lookup by handedness index (0=LEFT, 1=RIGHT)
TEMPLATES = {
    0: LEFT_SIDING,
    1: RIGHT_SIDING,
}


# =============================================================================
# CROSS-JUNCTION TEMPLATE (4 switches + CROSS_90)
# =============================================================================
# Geometry: cross at origin, 4 switches arranged in 90° rotational symmetry.
# Each switch's port C connects to one cross port via a single R40 spur curve.
# All-LEFT configuration: 4 LEFT switches + 4 R40_CURVE flip=1 spurs + 1 CROSS_90.
# Geometry validated by spike script — all 4 spurs land exactly on cross ports.
CROSS_90_PIECE_IDX = 3

# Cross ports in cross local frame: (dx, dy, entry_heading_degrees).
# Entry heading is the train heading when entering the cross at this port.
CROSS_PORTS_LOCAL = {
    "W": (0.0,  0.0,    0.0),    # port A — train enters going east
    "E": (16.0, 0.0,  180.0),    # port B — train enters going west
    "S": (8.0, -8.0,   90.0),    # port C — train enters going north
    "N": (8.0,  8.0,  -90.0),    # port D — train enters going south
}


@dataclass(frozen=True)
class CrossJunctionTemplate:
    """4-switch + CROSS_90 junction template.

    A junction places 4 switches at the 4 sides of a CROSS_90, each connected
    to one cross port via a 1-curve spur. The switches sit on the main loop;
    the cross + spurs are an internal sub-structure.

    Train traversal modes per junction (3):
      - bypass: stay on main loop (use switch's port B, skip the cross)
      - cross W↔E: divert via switch_for_W's port C, traverse cross W→E,
                    re-merge via switch_for_E's port C (and reverse)
      - cross N↔S: similar via N and S switches

    Attributes:
        name: Human-readable.
        handedness: "LEFT" or "RIGHT" — handedness of all 4 switches.
        switch_idx: catalog piece index of the switches (all same handedness).
        spur_curve_idx: catalog piece index of the spur curve (R40_CURVE).
        spur_curve_flip: flip bit for the spur curve. LEFT switches use flip=1
                         (curves back to align); RIGHT switches use flip=0.
        cross_idx: catalog piece index of CROSS_90 (= 3 post-refactor).
    """

    name: str
    handedness: str
    switch_idx: int
    spur_curve_idx: int
    spur_curve_flip: int
    cross_idx: int = CROSS_90_PIECE_IDX


CROSS_JUNCTION_LEFT = CrossJunctionTemplate(
    name="cross_junction_left",
    handedness="LEFT",
    switch_idx=R40_SWITCH_LEFT,
    spur_curve_idx=R40_CURVE,
    spur_curve_flip=1,
)

CROSS_JUNCTION_RIGHT = CrossJunctionTemplate(
    name="cross_junction_right",
    handedness="RIGHT",
    switch_idx=R40_SWITCH_RIGHT,
    spur_curve_idx=R40_CURVE,
    spur_curve_flip=0,
)

CROSS_JUNCTION_TEMPLATES = {
    0: CROSS_JUNCTION_LEFT,
    1: CROSS_JUNCTION_RIGHT,
}


def switch_position_for_cross_port(
    port_name: str,
    cross_position: Tuple[float, float] = (0.0, 0.0),
    template: CrossJunctionTemplate = CROSS_JUNCTION_LEFT,
) -> Tuple[float, float, float]:
    """Compute the (x, y, theta) where a switch must be placed in the main
    loop so that its port C + spur lands exactly on the named cross port.

    The 1-curve spur (R40 of opposite handedness from the switch) cancels
    the switch's ±22.5° divergence, so the train heading at spur end equals
    the switch's main heading. That heading must equal the cross port's
    entry heading.
    """
    px_local, py_local, pt_local = CROSS_PORTS_LOCAL[port_name]
    target_x = cross_position[0] + px_local
    target_y = cross_position[1] + py_local
    switch_heading = pt_local

    # Forward-displacement from switch port A through diverge + spur, in main:
    sh_rad = np.radians(switch_heading)
    if template.handedness == "LEFT":
        port_c_local_dy = 13.0
        port_c_dtheta = 22.5
        spur_local_dy = -3.045  # R40_CURVE flip=1
        spur_local_dtheta = -22.5
    else:
        port_c_local_dy = -13.0
        port_c_dtheta = -22.5
        spur_local_dy = 3.045   # R40_CURVE flip=0
        spur_local_dtheta = 22.5

    pc_rad = np.radians(switch_heading + port_c_dtheta)
    dx = (32.75 * np.cos(sh_rad) - port_c_local_dy * np.sin(sh_rad)
          + 15.307 * np.cos(pc_rad) - spur_local_dy * np.sin(pc_rad))
    dy = (32.75 * np.sin(sh_rad) + port_c_local_dy * np.cos(sh_rad)
          + 15.307 * np.sin(pc_rad) + spur_local_dy * np.cos(pc_rad))
    return (target_x - dx, target_y - dy, switch_heading)


def get_cross_junction_inventory_requirements(
    template: CrossJunctionTemplate,
) -> dict:
    """Pieces consumed by one cross-junction instance.

    4 switches of the chosen handedness + 4 spur curves (opposite handedness)
    + 1 CROSS_90.
    """
    return {
        template.switch_idx: 4,
        template.spur_curve_idx: 4,
        template.cross_idx: 1,
    }


# =============================================================================
# DOUBLE_CROSSOVER TEMPLATE
# =============================================================================
# The piece is 48 stud long (3x straight_16) with two parallel tracks at 16
# stud lateral separation. Four ports + four routes:
#
#   route 0 (track1_through):  A(0, 0)   -> B(48, 0)    — straight on track 1
#   route 1 (track2_through):  C(0, 16)  -> D(48, 16)   — straight on track 2
#   route 2 (cross_1_to_2):    A(0, 0)   -> D(48, 16)   — diagonal up
#   route 3 (cross_2_to_1):    C(0, 16)  -> B(48, 0)    — diagonal down
#
# Train-frame FK per route (heading preserved through every route):

DC_PIECE_IDX = 6  # DOUBLE_CROSSOVER catalog index (post R40 collapse)
DC_LENGTH_STUDS = 48.0
DC_LATERAL_STUDS = 16.0

# Catalog-route indices (must match yaml order in track_pieces_v2.yaml)
DC_R_TRACK1_THROUGH = 0
DC_R_TRACK2_THROUGH = 1
DC_R_CROSS_1_TO_2 = 2
DC_R_CROSS_2_TO_1 = 3

# Train-frame FK (dx, dy, dtheta_deg) per route, with entry port as origin and
# entry heading = 0. dtheta is in DEGREES to match catalog _fk_table convention.
DC_ROUTE_FK: Dict[int, Tuple[float, float, float]] = {
    DC_R_TRACK1_THROUGH: (DC_LENGTH_STUDS,  0.0,                0.0),
    DC_R_TRACK2_THROUGH: (DC_LENGTH_STUDS,  0.0,                0.0),
    DC_R_CROSS_1_TO_2:   (DC_LENGTH_STUDS,  DC_LATERAL_STUDS,   0.0),
    DC_R_CROSS_2_TO_1:   (DC_LENGTH_STUDS, -DC_LATERAL_STUDS,   0.0),
}

# Which port the train enters at for each route.
DC_ROUTE_ENTRY_PORT: Dict[int, str] = {
    DC_R_TRACK1_THROUGH: "A",
    DC_R_TRACK2_THROUGH: "C",
    DC_R_CROSS_1_TO_2:   "A",
    DC_R_CROSS_2_TO_1:   "C",
}

# Port positions in piece-local frame (port_A at origin, piece heading = +x).
DC_PORT_LOCAL: Dict[str, Tuple[float, float]] = {
    "A": (0.0,              0.0),
    "B": (DC_LENGTH_STUDS,  0.0),
    "C": (0.0,              DC_LATERAL_STUDS),
    "D": (DC_LENGTH_STUDS,  DC_LATERAL_STUDS),
}

# Port set covered by each catalog route (0=A, 1=B, 2=C, 3=D).
DC_PORT_INDEX: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}
DC_ROUTE_PORT_SET: Dict[int, frozenset] = {
    DC_R_TRACK1_THROUGH: frozenset({0, 1}),
    DC_R_TRACK2_THROUGH: frozenset({2, 3}),
    DC_R_CROSS_1_TO_2:   frozenset({0, 3}),
    DC_R_CROSS_2_TO_1:   frozenset({2, 1}),
}

# Valid 2-route covers of {A,B,C,D} — these are the only route pairs that
# leave NO dangling ports on a physical DBL_CROSSOVER.
DC_VALID_ROUTE_PAIRS: frozenset = frozenset({
    frozenset({DC_R_TRACK1_THROUGH, DC_R_TRACK2_THROUGH}),  # both-through
    frozenset({DC_R_CROSS_1_TO_2,   DC_R_CROSS_2_TO_1}),    # both-cross
})


def dbl_crossover_piece_origin(
    train_pose: Tuple[float, float, float],
    route: int,
) -> Tuple[float, float, float]:
    """World pose of port A given the train pose at its entry port for `route`.

    Train pose is (x, y, theta_deg). Returned tuple is (x_A, y_A, theta_deg)
    — port A's world coordinates and the piece heading. The piece heading
    equals the train heading because all routes preserve heading.
    """
    x, y, theta_deg = train_pose
    entry_port = DC_ROUTE_ENTRY_PORT[route]
    if entry_port == "A":
        return (x, y, theta_deg)
    # Entry port is C: port A sits at piece-local (0, 0); port C at (0, 16).
    # Piece-local +y is the lateral axis; in world, that direction is
    # rotated by theta. Port A is 16 stud back along that lateral axis.
    theta_rad = np.radians(theta_deg)
    return (
        x + DC_LATERAL_STUDS * np.sin(theta_rad),
        y - DC_LATERAL_STUDS * np.cos(theta_rad),
        theta_deg,
    )


def dbl_crossover_routes_cover_all_ports(route_a: int, route_b: int) -> bool:
    """True iff the two-route pair (route_a, route_b) covers {A,B,C,D}."""
    return frozenset({route_a, route_b}) in DC_VALID_ROUTE_PAIRS


def get_dbl_crossover_inventory_requirements() -> Dict[int, int]:
    """Pieces consumed by one physical DBL_CROSSOVER instance."""
    return {DC_PIECE_IDX: 1}


# =============================================================================
# BRANCH PIECE COMPUTATION
# =============================================================================


def compute_branch_pieces(
    template: PassingSidingTemplate, n_straights: int,
) -> Tuple[List[int], List[int]]:
    """Generate piece sequence + per-slot flips for a passing siding branch.

    The branch structure is:
        [approach_curve] + [straight × N] + [return_curve]

    Args:
        template: Passing siding template defining piece types + flips.
        n_straights: Number of straight pieces in parallel section.

    Returns:
        (pieces, flips) — piece indices and matching flip bits. Flips on
        straights are 0 (irrelevant for symmetric pieces); approach/return
        curves use the template's per-curve flip.
    """
    pieces = (
        [template.approach_curve_idx]
        + [template.straight_idx] * n_straights
        + [template.return_curve_idx]
    )
    flips = (
        [template.approach_curve_flip]
        + [0] * n_straights
        + [template.return_curve_flip]
    )
    return pieces, flips


# =============================================================================
# GEOMETRY COMPUTATION
# =============================================================================

# R40 curve FK (catalog default — flip=0 = LEFT direction)
R40_CURVE_FK = (15.307, 3.045, 22.5)
STRAIGHT_16_FK = (16.0, 0.0, 0.0)

# FK lookup by piece index (default route — switches use through-route here).
# R40_CURVE entry is the LEFT direction; flip-aware callers use piece_fk(idx, flip).
PIECE_FK = {
    STRAIGHT_16: STRAIGHT_16_FK,
    R40_CURVE: R40_CURVE_FK,
    R40_SWITCH_LEFT: (32.0, 0.0, 0.0),
    R40_SWITCH_RIGHT: (32.0, 0.0, 0.0),
}


def piece_fk(piece_idx: int, flip: int = 0) -> Tuple[float, float, float]:
    """FK for ``piece_idx`` with optional flip for symmetric pieces.

    R40_CURVE with flip=1 negates dy and dtheta (LEFT → RIGHT). Other pieces
    ignore the flip bit (they are symmetric or have no useful mirror).
    """
    fk = PIECE_FK[piece_idx]
    if piece_idx == R40_CURVE and flip:
        dx, dy, dtheta = fk
        return (dx, -dy, -dtheta)
    return fk


def apply_fk(state: Tuple[float, float, float], fk: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Apply FK delta to a state (x, y, theta).

    FK transformation:
        x' = x + dx * cos(theta) - dy * sin(theta)
        y' = y + dx * sin(theta) + dy * cos(theta)
        theta' = theta + dtheta

    Args:
        state: Current pose (x, y, theta) in degrees.
        fk: FK delta (dx, dy, dtheta).

    Returns:
        New pose (x', y', theta').
    """
    x, y, theta = state
    dx, dy, dtheta = fk

    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x_new = x + dx * cos_t - dy * sin_t
    y_new = y + dx * sin_t + dy * cos_t
    theta_new = theta + dtheta

    return (float(x_new), float(y_new), float(theta_new))


def compute_branch_endpoint(
    in_switch_state: Tuple[float, float, float],
    template: PassingSidingTemplate,
    n_straights: int,
) -> Tuple[float, float, float]:
    """Compute where the branch ends (before merge).

    Traces FK through:
    1. IN switch diverge
    2. Approach curve
    3. N straights
    4. Return curve

    Args:
        in_switch_state: Pose at IN switch entry (x, y, theta).
        template: Passing siding template.
        n_straights: Number of straights in parallel section.

    Returns:
        Branch endpoint pose (x, y, theta) ready for OUT switch merge.
    """
    state = in_switch_state

    # 1. IN switch diverge
    state = apply_fk(state, template.diverge_fk)

    # 2. Approach curve (flip-aware: LEFT_SIDING uses flip=1 i.e. -22.5°)
    state = apply_fk(state, piece_fk(template.approach_curve_idx, template.approach_curve_flip))

    # 3. N straights
    straight_fk = PIECE_FK[template.straight_idx]
    for _ in range(n_straights):
        state = apply_fk(state, straight_fk)

    # 4. Return curve (flip-aware)
    state = apply_fk(state, piece_fk(template.return_curve_idx, template.return_curve_flip))

    return state


def compute_required_main_distance(
    template: PassingSidingTemplate,
    n_straights: int,
    body_length: float = 32.0,
    out_switch_port_c_dx: float = 32.75,
) -> float:
    """X contribution of main-loop pieces strictly between IN and OUT switches
    (NOT counting the two switch bodies themselves) needed so that main and
    branch paths through this siding traverse the same X distance.

    Derivation in IN-entry frame:
      branch_end_x = compute_branch_endpoint(...)[0] = X just before OUT.
      merge_fk takes the train from branch_end through the reversed-install
      OUT switch, ending at OUT body END (in main frame). The world-frame X
      delta of that traversal works out to +port_c_dx, so:
          OUT_body_END = branch_end_x + port_c_dx
          OUT_body_start = OUT_body_END - body_length
          main_pieces_x = OUT_body_start - body_length    (IN body end = body_length)
                        = branch_end_x + port_c_dx - 2 * body_length
    """
    start_state = (0.0, 0.0, 0.0)
    branch_end = compute_branch_endpoint(start_state, template, n_straights)
    return float(branch_end[0] + out_switch_port_c_dx - 2.0 * body_length)


def compute_out_switch_alignment_error(
    in_switch_state: Tuple[float, float, float],
    out_switch_state: Tuple[float, float, float],
    template: PassingSidingTemplate,
    n_straights: int,
) -> Tuple[float, float]:
    """Compute error between branch endpoint and OUT switch merge entry.

    For a valid passing siding, the branch must arrive at the OUT switch's
    merge entry port (port 2). This function computes the position and
    angle error.

    Args:
        in_switch_state: Pose at IN switch entry.
        out_switch_state: Pose at OUT switch entry.
        template: Passing siding template.
        n_straights: Number of straights in parallel section.

    Returns:
        (position_error, angle_error) tuple in (studs, degrees).
    """
    # Strategy: simulate the full branch traversal, including the reversed-
    # install OUT switch via template.merge_fk, and compare the train's exit
    # state to where the OUT switch's main-side exit (port A in main frame,
    # body_length forward of out_switch_state) actually sits.
    branch_end = compute_branch_endpoint(in_switch_state, template, n_straights)
    after_merge = apply_fk(branch_end, template.merge_fk)

    out_x, out_y, out_theta = out_switch_state
    body_length = 32.0
    out_theta_rad = np.radians(out_theta)
    target_x = out_x + body_length * np.cos(out_theta_rad)
    target_y = out_y + body_length * np.sin(out_theta_rad)
    target_theta = out_theta  # exit on main heading

    position_error = float(np.sqrt(
        (after_merge[0] - target_x) ** 2 + (after_merge[1] - target_y) ** 2
    ))
    angle_diff = after_merge[2] - target_theta
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    angle_error = float(abs(angle_diff))

    return (position_error, angle_error)


def is_valid_siding(
    in_switch_state: Tuple[float, float, float],
    out_switch_state: Tuple[float, float, float],
    template: PassingSidingTemplate,
    n_straights: int,
    position_tolerance: float = 2.0,
    angle_tolerance: float = 5.0,
) -> bool:
    """Check if a passing siding configuration is geometrically valid.

    Args:
        in_switch_state: Pose at IN switch entry.
        out_switch_state: Pose at OUT switch entry.
        template: Passing siding template.
        n_straights: Number of straights in parallel section.
        position_tolerance: Max position error (studs).
        angle_tolerance: Max angle error (degrees).

    Returns:
        True if siding geometry closes within tolerances.
    """
    pos_error, angle_error = compute_out_switch_alignment_error(
        in_switch_state, out_switch_state, template, n_straights
    )
    return pos_error <= position_tolerance and angle_error <= angle_tolerance


# =============================================================================
# INVENTORY HELPERS
# =============================================================================


def get_siding_inventory_requirements(
    template: PassingSidingTemplate,
    n_straights: int,
) -> dict[int, int]:
    """Get piece counts needed for a passing siding.

    Args:
        template: Passing siding template.
        n_straights: Number of straights in parallel section.

    Returns:
        Dict mapping piece_index -> required count.
    """
    requirements: dict[int, int] = {}

    # Every passing siding consumes 1 LEFT + 1 RIGHT switch (opposite-handed).
    requirements[R40_SWITCH_LEFT] = requirements.get(R40_SWITCH_LEFT, 0) + 1
    requirements[R40_SWITCH_RIGHT] = requirements.get(R40_SWITCH_RIGHT, 0) + 1

    # Curves (approach + return are same handedness, so two of the same curve type)
    requirements[template.approach_curve_idx] = requirements.get(template.approach_curve_idx, 0) + 1
    requirements[template.return_curve_idx] = requirements.get(template.return_curve_idx, 0) + 1

    # Straights
    if n_straights > 0:
        requirements[template.straight_idx] = requirements.get(template.straight_idx, 0) + n_straights

    return requirements


def check_siding_inventory(
    template: PassingSidingTemplate,
    n_straights: int,
    available_inventory: dict[int, int],
    used_inventory: dict[int, int],
) -> bool:
    """Check if inventory has pieces for a passing siding.

    Args:
        template: Passing siding template.
        n_straights: Number of straights in parallel section.
        available_inventory: Available pieces by index.
        used_inventory: Already used pieces by index.

    Returns:
        True if sufficient inventory remains.
    """
    requirements = get_siding_inventory_requirements(template, n_straights)

    for piece_idx, needed in requirements.items():
        available = available_inventory.get(piece_idx, 0)
        used = used_inventory.get(piece_idx, 0)
        if used + needed > available:
            return False

    return True
