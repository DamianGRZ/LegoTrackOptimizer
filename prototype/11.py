#!/usr/bin/env python3
"""
LEGO Train Track Layout Genetic Algorithm Optimizer

Implements mutation-only GA approach based on research findings:
- ADD, MUTATE, DELETE operators for incremental construction
- Forward propagation maintains connectivity between track pieces
- IDEA (Infeasibility-Driven EA) maintaining 5-20% infeasible solutions
- Two-phase optimization: 1) Maximize curves/straights, 2) Add switches

LEGO Track Geometry (from bricksmcgee.com):
- Straight: 16 studs length
- Curve: 22.5° per piece, 4 pieces = 90°, radius ~40 studs
- Switch: 32 studs length, branches 8 studs apart
- Standard parallel track spacing: 8 studs
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from enum import IntEnum
from copy import deepcopy


# ============================================================================
# CONSTANTS - LEGO TRACK GEOMETRY
# ============================================================================

class PieceType(IntEnum):
    STRAIGHT = 0
    CURVE_LEFT = 1
    CURVE_RIGHT = 2
    SWITCH_LEFT = 3
    SWITCH_RIGHT = 4


# Track geometry constants
STRAIGHT_LENGTH = 16.0  # studs
CURVE_ANGLE = 22.5  # degrees per curve piece
CURVE_RADIUS = 40.0  # studs (approximation for standard LEGO curves)
SWITCH_LENGTH = 32.0  # studs
SWITCH_BRANCH_OFFSET = 8.0  # studs - standard parallel track spacing
SWITCH_ANGLE = 22.5  # degrees - angle of branch divergence

# For a closed loop: 16 curves * 22.5° = 360° (perfect circle)
# Actual may differ slightly
CURVES_FOR_FULL_CIRCLE = 16  # Standard LEGO simple loop

# Connection tolerances - LEGO has some play
POSITION_TOLERANCE = 8.0  # studs - LEGO connections have tolerance
ANGLE_TOLERANCE = 15.0  # degrees - for discrete 22.5° increments
# Siding branch tolerance: opposite-type switch pairing (SL→SR, SR→SL) creates
# a ~24 stud gap because branch stays on one side while closing switch expects
# from the other side. This is a geometric limitation of LEGO on straight sections.
BRANCH_POSITION_TOLERANCE = 35.0  # studs - allows opposite-type siding configurations


# ============================================================================
# SWITCH CONFIGURATION ANALYSIS
# ============================================================================

def analyze_switch_feasibility(inventory: 'PieceInventory',
                               main_track_curves_used: int = None) -> Dict:
    """
    Analyze what switch configurations are possible with given inventory.

    CORRECT SIDING PATTERNS (switches must be OPPOSITE types):
    - LEFT siding: SL (Normal) → SR (Reversed) - needs R-S-R branch pattern
    - RIGHT siding: SR (Normal) → SL (Reversed) - needs L-S-L branch pattern

    Branch pattern explanation (R-S-R for left siding):
    - SL Normal outputs branch at +22.5° (left side)
    - R curve: +22.5° → 0° (now parallel to main)
    - Straights: travel parallel
    - R curve: 0° → -22.5° (now approaching from right side)
    - SR Reversed receives branch at -22.5° (right side)
    - The branch CROSSES OVER from left to right side!

    Returns dict with:
    - 'passing_siding_left': Can make LEFT siding (1 SL + 1 SR)?
    - 'passing_siding_right': Can make RIGHT siding (1 SR + 1 SL)?
    - 'crossover': Can make crossover with 1 left + 1 right switch?
    - 'required_branch_pieces': What pieces are needed for each config
    - 'recommendation': Best configuration for this inventory
    """
    result = {
        'passing_siding_left': False,
        'passing_siding_right': False,
        'crossover': False,
        'required_branch_pieces': {},
        'recommendation': None,
        'reason': "",
        'spare_curves': {}
    }

    sl = inventory.switch_left
    sr = inventory.switch_right
    cl = inventory.curve_left
    cr = inventory.curve_right
    s = inventory.straight

    # Calculate curves needed for main loop (16 curves of same type)
    main_curves_needed = 16

    # Calculate spare curves (beyond main loop needs)
    if cl >= main_curves_needed and cr >= main_curves_needed:
        spare_left = cl
        spare_right = cr
    elif cl >= main_curves_needed:
        spare_left = cl - main_curves_needed
        spare_right = cr
    elif cr >= main_curves_needed:
        spare_left = cl
        spare_right = cr - main_curves_needed
    else:
        spare_left = 0
        spare_right = 0

    result['spare_curves'] = {'left': spare_left, 'right': spare_right}

    # Both siding types require 1 SL + 1 SR (opposite switch types)
    has_switch_pair = (sl >= 1 and sr >= 1)

    # LEFT siding: SL (Normal) → SR (Reversed) with R-S-R branch
    # R-S-R pattern needs 2+ RIGHT curves
    if has_switch_pair:
        if spare_right >= 2:
            result['passing_siding_left'] = True
            result['required_branch_pieces']['passing_siding_left'] = {
                'switches': '1 SL (normal) + 1 SR (reversed)',
                'branch_pattern': 'R-S-R (2+ right curves)',
                'spare_curves_right': 2,
                'straights_min': 1
            }
        else:
            result['required_branch_pieces']['passing_siding_left'] = {
                'missing': f"need 2+ spare RIGHT curves for R-S-R branch (have {spare_right})"
            }

    # RIGHT siding: SR (Normal) → SL (Reversed) with L-S-L branch
    # L-S-L pattern needs 2+ LEFT curves
    if has_switch_pair:
        if spare_left >= 2:
            result['passing_siding_right'] = True
            result['required_branch_pieces']['passing_siding_right'] = {
                'switches': '1 SR (normal) + 1 SL (reversed)',
                'branch_pattern': 'L-S-L (2+ left curves)',
                'spare_curves_left': 2,
                'straights_min': 1
            }
        else:
            result['required_branch_pieces']['passing_siding_right'] = {
                'missing': f"need 2+ spare LEFT curves for L-S-L branch (have {spare_left})"
            }

    # Crossover uses both switches directly connected
    if sl >= 1 and sr >= 1:
        result['crossover'] = True
        result['required_branch_pieces']['crossover'] = {
            'curves_left': 0, 'curves_right': 0, 'note': 'Direct crossover'
        }

    # Determine recommendation
    if result['passing_siding_left'] and spare_right >= 2:
        result['recommendation'] = 'passing_siding_left'
        result['reason'] = "Left siding: SL(Normal)→SR(Reversed) with R-S-R branch"
    elif result['passing_siding_right'] and spare_left >= 2:
        result['recommendation'] = 'passing_siding_right'
        result['reason'] = "Right siding: SR(Normal)→SL(Reversed) with L-S-L branch"
    elif result['crossover']:
        result['recommendation'] = 'crossover'
        result['reason'] = "Crossover with 1 SL + 1 SR"
    elif sl > 0 or sr > 0:
        result['recommendation'] = 'none_feasible'
        reasons = []
        if not has_switch_pair:
            reasons.append(f"Need 1 SL + 1 SR for siding (have {sl} SL, {sr} SR)")
        else:
            if 'missing' in result['required_branch_pieces'].get('passing_siding_left', {}):
                reasons.append(f"Left siding: {result['required_branch_pieces']['passing_siding_left']['missing']}")
            if 'missing' in result['required_branch_pieces'].get('passing_siding_right', {}):
                reasons.append(f"Right siding: {result['required_branch_pieces']['passing_siding_right']['missing']}")
        result['reason'] = "Cannot form closed branch. " + "; ".join(reasons)
    else:
        result['recommendation'] = 'no_switches'
        result['reason'] = "No switches in inventory"

    return result


def calculate_passing_siding_geometry(switch_type: PieceType,
                                      main_track_straights: int = 2) -> Dict:
    """
    Calculate geometry for a passing siding.

    Args:
        switch_type: SWITCH_LEFT or SWITCH_RIGHT
        main_track_straights: Number of straights on main between switches

    Returns geometry details for constructing the passing siding.
    """
    is_left = switch_type == PieceType.SWITCH_LEFT
    diverge_angle = SWITCH_ANGLE if is_left else -SWITCH_ANGLE

    # Main track length between switches
    main_length = main_track_straights * STRAIGHT_LENGTH

    # Branch geometry:
    # Start: diverges at ±22.5°
    # Correction curve: ∓22.5° to become parallel
    # Parallel section: must cover same X distance as main
    # Return curve: ±22.5° to approach second switch
    # End: converges at ±22.5°

    # The parallel branch is offset from main track
    # Calculate exact offset based on switch + curve geometry

    # First, switch branch endpoint (relative to switch start at 0,0 heading 0°)
    if is_left:
        # Left switch branch goes up-left
        branch_start = Point(29.56, 12.25)  # From earlier calculation
        branch_angle = 22.5
    else:
        # Right switch branch goes down-right
        branch_start = Point(29.56, -12.25)
        branch_angle = -22.5

    # After correction curve, branch is parallel but offset
    # Curve has radius 40, angle 22.5°
    # Endpoint offset in perpendicular direction

    return {
        'switch_type': switch_type,
        'main_straights': main_track_straights,
        'branch_pieces': {
            'correction_curve': PieceType.CURVE_RIGHT if is_left else PieceType.CURVE_LEFT,
            'parallel_straights': max(1, main_track_straights - 1),  # Slightly fewer due to switch length
            'return_curve': PieceType.CURVE_LEFT if is_left else PieceType.CURVE_RIGHT
        },
        'total_branch_length': main_length,  # Approximately
        'branch_offset': 12.25  # Approximate studs from main
    }


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def rotate(self, angle_deg: float, center: 'Point' = None) -> 'Point':
        """Rotate point around center (default origin)"""
        if center is None:
            center = Point(0, 0)
        angle_rad = math.radians(angle_deg)
        dx = self.x - center.x
        dy = self.y - center.y
        new_x = center.x + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        new_y = center.y + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        return Point(new_x, new_y)


@dataclass
class TrackPiece:
    """Represents a single track piece with its computed geometry"""
    piece_type: PieceType
    start_pos: Point
    start_angle: float  # degrees
    end_pos: Point
    end_angle: float  # degrees

    # For switches - secondary endpoint
    branch_end_pos: Optional[Point] = None
    branch_end_angle: Optional[float] = None
    is_reversed: bool = False  # True for reversed/trailing switch
    is_branch_piece: bool = False  # True if this piece is part of the branch track

    def get_length(self) -> float:
        if self.piece_type == PieceType.STRAIGHT:
            return STRAIGHT_LENGTH
        elif self.piece_type in (PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT):
            return CURVE_RADIUS * math.radians(CURVE_ANGLE)
        else:  # Switch
            return SWITCH_LENGTH


@dataclass
class Gene:
    """A single gene representing a track piece type"""
    piece_type: PieceType
    is_branch: bool = False  # True if this piece is part of the branch track
    is_reversed: bool = False  # True for reversed switch (closing switch)

    def copy(self) -> 'Gene':
        return Gene(self.piece_type, self.is_branch, self.is_reversed)


@dataclass
class PieceInventory:
    """Manages available track pieces"""
    straight: int = 0
    curve_left: int = 0
    curve_right: int = 0
    switch_left: int = 0
    switch_right: int = 0

    def get_count(self, piece_type: PieceType) -> int:
        counts = {
            PieceType.STRAIGHT: self.straight,
            PieceType.CURVE_LEFT: self.curve_left,
            PieceType.CURVE_RIGHT: self.curve_right,
            PieceType.SWITCH_LEFT: self.switch_left,
            PieceType.SWITCH_RIGHT: self.switch_right
        }
        return counts.get(piece_type, 0)

    def total_basic(self) -> int:
        """Total non-switch pieces"""
        return self.straight + self.curve_left + self.curve_right

    def total_switches(self) -> int:
        return self.switch_left + self.switch_right

    def total(self) -> int:
        return self.total_basic() + self.total_switches()

    def copy(self) -> 'PieceInventory':
        return PieceInventory(
            self.straight, self.curve_left, self.curve_right,
            self.switch_left, self.switch_right
        )

    def can_use(self, piece_type: PieceType) -> bool:
        return self.get_count(piece_type) > 0

    def use_piece(self, piece_type: PieceType) -> bool:
        if piece_type == PieceType.STRAIGHT and self.straight > 0:
            self.straight -= 1
            return True
        elif piece_type == PieceType.CURVE_LEFT and self.curve_left > 0:
            self.curve_left -= 1
            return True
        elif piece_type == PieceType.CURVE_RIGHT and self.curve_right > 0:
            self.curve_right -= 1
            return True
        elif piece_type == PieceType.SWITCH_LEFT and self.switch_left > 0:
            self.switch_left -= 1
            return True
        elif piece_type == PieceType.SWITCH_RIGHT and self.switch_right > 0:
            self.switch_right -= 1
            return True
        return False

    def return_piece(self, piece_type: PieceType):
        if piece_type == PieceType.STRAIGHT:
            self.straight += 1
        elif piece_type == PieceType.CURVE_LEFT:
            self.curve_left += 1
        elif piece_type == PieceType.CURVE_RIGHT:
            self.curve_right += 1
        elif piece_type == PieceType.SWITCH_LEFT:
            self.switch_left += 1
        elif piece_type == PieceType.SWITCH_RIGHT:
            self.switch_right += 1


# ============================================================================
# TRACK GEOMETRY COMPUTATION (Forward Propagation)
# ============================================================================

def compute_piece_endpoint(piece_type: PieceType, start_pos: Point,
                           start_angle: float,
                           is_reversed: bool = False) -> Tuple[Point, float, Optional[Point], Optional[float]]:
    """
    Compute endpoint of a track piece given start position and angle.
    Returns (end_pos, end_angle, branch_end_pos, branch_end_angle)

    Forward propagation: Each piece's output becomes the next piece's input.

    For REVERSED switches (closing/trailing position):
    - Main track still goes straight through (same as normal)
    - Branch port is at the END, not the START (branch converges to switch)
    - branch_end_pos represents where branch track connects to this switch
    """
    angle_rad = math.radians(start_angle)

    if piece_type == PieceType.STRAIGHT:
        end_x = start_pos.x + STRAIGHT_LENGTH * math.cos(angle_rad)
        end_y = start_pos.y + STRAIGHT_LENGTH * math.sin(angle_rad)
        return Point(end_x, end_y), start_angle, None, None

    elif piece_type == PieceType.CURVE_LEFT:
        # Left turn: center is perpendicular left
        perp_angle = angle_rad + math.pi / 2
        center = Point(
            start_pos.x + CURVE_RADIUS * math.cos(perp_angle),
            start_pos.y + CURVE_RADIUS * math.sin(perp_angle)
        )
        end_angle = start_angle + CURVE_ANGLE
        end_angle_rad = math.radians(end_angle)
        end_pos = Point(
            center.x + CURVE_RADIUS * math.cos(end_angle_rad - math.pi / 2),
            center.y + CURVE_RADIUS * math.sin(end_angle_rad - math.pi / 2)
        )
        return end_pos, end_angle, None, None

    elif piece_type == PieceType.CURVE_RIGHT:
        # Right turn: center is perpendicular right
        perp_angle = angle_rad - math.pi / 2
        center = Point(
            start_pos.x + CURVE_RADIUS * math.cos(perp_angle),
            start_pos.y + CURVE_RADIUS * math.sin(perp_angle)
        )
        end_angle = start_angle - CURVE_ANGLE
        end_angle_rad = math.radians(end_angle)
        end_pos = Point(
            center.x + CURVE_RADIUS * math.cos(end_angle_rad + math.pi / 2),
            center.y + CURVE_RADIUS * math.sin(end_angle_rad + math.pi / 2)
        )
        return end_pos, end_angle, None, None

    elif piece_type == PieceType.SWITCH_LEFT:
        # Main track continues straight
        main_end_x = start_pos.x + SWITCH_LENGTH * math.cos(angle_rad)
        main_end_y = start_pos.y + SWITCH_LENGTH * math.sin(angle_rad)
        main_end = Point(main_end_x, main_end_y)

        if is_reversed:
            # REVERSED switch: branch converges TO the switch near START
            # The frog point (where tracks merge) is at the start_pos end
            # branch_end_pos = where branch track connects (SWITCH_LENGTH back from start at branch angle)
            branch_angle = start_angle + SWITCH_ANGLE
            branch_angle_rad = math.radians(branch_angle)
            # Branch comes FROM this direction, so we place end point back along that angle
            branch_end_x = start_pos.x + SWITCH_LENGTH * math.cos(branch_angle_rad)
            branch_end_y = start_pos.y + SWITCH_LENGTH * math.sin(branch_angle_rad)
            branch_end = Point(branch_end_x, branch_end_y)
            # Return branch_end as where branch comes from, branch_angle as its approach direction
            return main_end, start_angle, branch_end, branch_angle
        else:
            # NORMAL switch: branch diverges FROM the switch at the START
            branch_angle = start_angle + SWITCH_ANGLE
            branch_angle_rad = math.radians(branch_angle)
            branch_end_x = start_pos.x + SWITCH_LENGTH * math.cos(branch_angle_rad)
            branch_end_y = start_pos.y + SWITCH_LENGTH * math.sin(branch_angle_rad)
            branch_end = Point(branch_end_x, branch_end_y)
            return main_end, start_angle, branch_end, branch_angle

    elif piece_type == PieceType.SWITCH_RIGHT:
        # Main track continues straight
        main_end_x = start_pos.x + SWITCH_LENGTH * math.cos(angle_rad)
        main_end_y = start_pos.y + SWITCH_LENGTH * math.sin(angle_rad)
        main_end = Point(main_end_x, main_end_y)

        if is_reversed:
            # REVERSED switch: branch converges TO the switch near START
            branch_angle = start_angle - SWITCH_ANGLE
            branch_angle_rad = math.radians(branch_angle)
            branch_end_x = start_pos.x + SWITCH_LENGTH * math.cos(branch_angle_rad)
            branch_end_y = start_pos.y + SWITCH_LENGTH * math.sin(branch_angle_rad)
            branch_end = Point(branch_end_x, branch_end_y)
            return main_end, start_angle, branch_end, branch_angle
        else:
            # NORMAL switch: branch diverges FROM the switch at the START
            branch_angle = start_angle - SWITCH_ANGLE
            branch_angle_rad = math.radians(branch_angle)
            branch_end_x = start_pos.x + SWITCH_LENGTH * math.cos(branch_angle_rad)
            branch_end_y = start_pos.y + SWITCH_LENGTH * math.sin(branch_angle_rad)
            branch_end = Point(branch_end_x, branch_end_y)
            return main_end, start_angle, branch_end, branch_angle

    raise ValueError(f"Unknown piece type: {piece_type}")


def calculate_track_bounds(pieces: List[TrackPiece]) -> Tuple[float, float, float, float]:
    """Calculate bounding box of track layout"""
    if not pieces:
        return (0, 0, 0, 0)

    x_coords = []
    y_coords = []

    for piece in pieces:
        x_coords.extend([piece.start_pos.x, piece.end_pos.x])
        y_coords.extend([piece.start_pos.y, piece.end_pos.y])
        if piece.branch_end_pos:
            x_coords.append(piece.branch_end_pos.x)
            y_coords.append(piece.branch_end_pos.y)

    return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))


def build_track_centered(genes: List[Gene],
                         boundary: Tuple[float, float, float, float]) -> List[TrackPiece]:
    """Build track and center it within the boundary"""
    # First build at origin
    pieces = build_track_from_genes(genes)

    if not pieces:
        return pieces

    # Calculate track bounds
    track_x_min, track_x_max, track_y_min, track_y_max = calculate_track_bounds(pieces)
    track_center_x = (track_x_min + track_x_max) / 2
    track_center_y = (track_y_min + track_y_max) / 2

    # Calculate boundary center
    bx_min, bx_max, by_min, by_max = boundary
    boundary_center_x = (bx_min + bx_max) / 2
    boundary_center_y = (by_min + by_max) / 2

    # Calculate offset to center track
    offset_x = boundary_center_x - track_center_x
    offset_y = boundary_center_y - track_center_y

    # Rebuild with offset
    start_pos = Point(offset_x, offset_y)
    return build_track_from_genes(genes, start_pos=start_pos)


def build_track_from_genes(genes: List[Gene],
                           start_pos: Point = None,
                           start_angle: float = 0.0) -> List[TrackPiece]:
    """
    Build complete track layout using forward propagation.
    Each piece's endpoint becomes the next piece's start point.

    BRANCH HANDLING:
    - Main track pieces (is_branch=False) are built sequentially
    - Branch pieces (is_branch=True) are built starting from the first switch's branch port
    - For reversed switches, the branch port is at the END (converging)
    """
    if start_pos is None:
        start_pos = Point(0.0, 0.0)

    pieces = []
    current_pos = start_pos
    current_angle = start_angle

    # Separate main track and branch genes
    main_genes = [g for g in genes if not g.is_branch]
    branch_genes = [g for g in genes if g.is_branch]

    # Track switches: (piece, gene) pairs to know which are reversed
    switch_info = []  # List of (piece, is_reversed)

    # Build main track
    for gene in main_genes:
        # Pass is_reversed for switches
        is_reversed = gene.is_reversed if gene.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT) else False

        end_pos, end_angle, branch_end, branch_angle = compute_piece_endpoint(
            gene.piece_type, current_pos, current_angle, is_reversed
        )

        piece = TrackPiece(
            piece_type=gene.piece_type,
            start_pos=current_pos,
            start_angle=current_angle,
            end_pos=end_pos,
            end_angle=end_angle,
            branch_end_pos=branch_end,
            branch_end_angle=branch_angle,
            is_reversed=is_reversed
        )
        pieces.append(piece)

        # Track switch info for branch building
        if gene.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT):
            switch_info.append((piece, gene.is_reversed))

        # Forward propagation: next piece starts where this one ends
        current_pos = end_pos
        current_angle = end_angle

    # Build branch track if we have branch genes and at least two switches
    if branch_genes and len(switch_info) >= 2:
        # Find first normal (opening) switch and first reversed (closing) switch
        normal_switches = [(p, rev) for p, rev in switch_info if not rev]
        reversed_switches = [(p, rev) for p, rev in switch_info if rev]

        if normal_switches:
            # Branch starts from first normal switch's branch output
            first_switch = normal_switches[0][0]
            branch_start_pos = first_switch.branch_end_pos
            branch_start_angle = first_switch.branch_end_angle

            if branch_start_pos and branch_start_angle is not None:
                branch_pos = branch_start_pos
                branch_angle = branch_start_angle

                for gene in branch_genes:
                    end_pos, end_angle, br_end, br_angle = compute_piece_endpoint(
                        gene.piece_type, branch_pos, branch_angle
                    )

                    piece = TrackPiece(
                        piece_type=gene.piece_type,
                        start_pos=branch_pos,
                        start_angle=branch_angle,
                        end_pos=end_pos,
                        end_angle=end_angle,
                        branch_end_pos=br_end,
                        branch_end_angle=br_angle,
                        is_branch_piece=True  # Mark as branch track
                    )
                    pieces.append(piece)

                    branch_pos = end_pos
                    branch_angle = end_angle

    return pieces


def calculate_closure_error(pieces: List[TrackPiece],
                            start_pos: Point = None,
                            genes: List[Gene] = None) -> Tuple[float, float]:
    """
    Calculate how well the track closes back to start.
    Returns (position_error, angle_error) in studs and degrees.

    For a valid closed loop:
    - Final position should equal start position
    - Final angle should point back to start (or equal start angle)

    NOTE: If genes are provided, only considers main track pieces (is_branch=False)
    for closure calculation. Branch pieces form a separate parallel path.
    """
    if not pieces:
        return float('inf'), float('inf')

    # If genes provided, filter to main track pieces only
    if genes:
        main_gene_count = sum(1 for g in genes if not g.is_branch)
        # Only consider main track pieces (first main_gene_count pieces)
        main_pieces = pieces[:main_gene_count]
    else:
        main_pieces = pieces

    if not main_pieces:
        return float('inf'), float('inf')

    if start_pos is None:
        start_pos = main_pieces[0].start_pos

    start_angle = main_pieces[0].start_angle
    final_pos = main_pieces[-1].end_pos
    final_angle = main_pieces[-1].end_angle

    # Position error: distance from final position to start
    pos_error = final_pos.distance_to(start_pos)

    # Angle error: For a closed loop, total angle change should be multiple of 360
    # Calculate total angle change
    total_angle_change = 0.0
    for piece in main_pieces:
        angle_diff = piece.end_angle - piece.start_angle
        # Normalize to -180 to 180
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        total_angle_change += angle_diff

    # For closure, total should be 360, -360, or 0 (depending on loop direction)
    # Check how close we are to a multiple of 360
    angle_mod = total_angle_change % 360
    angle_error = min(abs(angle_mod), abs(360 - abs(angle_mod)))

    # Also check if final angle matches start angle (needed for proper closure)
    final_angle_norm = final_angle % 360
    start_angle_norm = start_angle % 360
    angle_match_error = abs(final_angle_norm - start_angle_norm)
    angle_match_error = min(angle_match_error, 360 - angle_match_error)

    # Combined angle error - track should both complete a full rotation and match angles
    combined_angle_error = min(angle_error, angle_match_error)

    return pos_error, combined_angle_error


def check_boundary_violations(pieces: List[TrackPiece],
                              boundary: Tuple[float, float, float, float],
                              num_samples: int = 5) -> Tuple[int, float]:
    """
    Check how many points violate boundary constraints.
    For curves, samples multiple points along the arc.
    Returns (violation_count, total_violation_distance)
    """
    x_min, x_max, y_min, y_max = boundary
    violations = 0
    total_distance = 0.0

    # Small tolerance for floating point comparisons
    tolerance = 0.1

    for piece in pieces:
        # Collect all points to check
        points_to_check = [piece.start_pos, piece.end_pos]

        # For curves, sample along the arc
        if piece.piece_type in [PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT]:
            # Calculate arc center
            angle_rad = math.radians(piece.start_angle)
            if piece.piece_type == PieceType.CURVE_RIGHT:
                perp_angle = angle_rad - math.pi / 2
            else:
                perp_angle = angle_rad + math.pi / 2

            center = Point(
                piece.start_pos.x + CURVE_RADIUS * math.cos(perp_angle),
                piece.start_pos.y + CURVE_RADIUS * math.sin(perp_angle)
            )

            # Sample points along arc
            for i in range(1, num_samples):
                t = i / num_samples
                mid_angle = piece.start_angle + t * (piece.end_angle - piece.start_angle)
                mid_angle_rad = math.radians(mid_angle)

                if piece.piece_type == PieceType.CURVE_RIGHT:
                    arc_x = center.x + CURVE_RADIUS * math.cos(mid_angle_rad + math.pi / 2)
                    arc_y = center.y + CURVE_RADIUS * math.sin(mid_angle_rad + math.pi / 2)
                else:
                    arc_x = center.x + CURVE_RADIUS * math.cos(mid_angle_rad - math.pi / 2)
                    arc_y = center.y + CURVE_RADIUS * math.sin(mid_angle_rad - math.pi / 2)

                points_to_check.append(Point(arc_x, arc_y))

        # Check all points
        for point in points_to_check:
            violation = 0.0
            if point.x < x_min - tolerance:
                violation += (x_min - point.x)
            elif point.x > x_max + tolerance:
                violation += (point.x - x_max)
            if point.y < y_min - tolerance:
                violation += (y_min - point.y)
            elif point.y > y_max + tolerance:
                violation += (point.y - y_max)

            if violation > tolerance:
                violations += 1
                total_distance += violation

        # Check branch endpoint for switches
        if piece.branch_end_pos:
            point = piece.branch_end_pos
            violation = 0.0
            if point.x < x_min - tolerance:
                violation += (x_min - point.x)
            elif point.x > x_max + tolerance:
                violation += (point.x - x_max)
            if point.y < y_min - tolerance:
                violation += (y_min - point.y)
            elif point.y > y_max + tolerance:
                violation += (point.y - y_max)

            if violation > tolerance:
                violations += 1
                total_distance += violation

    return violations, total_distance


def check_collisions(pieces: List[TrackPiece], min_distance: float = 6.0,
                     genes: List[Gene] = None) -> int:
    """
    Check for track collisions (pieces too close to each other).
    Returns collision count.

    Rules:
    - Branch pieces (is_branch_piece=True) are allowed to be close to main track
      pieces near switches (this is the passing siding)
    - Pieces near the main track closure point are allowed to be close
    - Collisions between other piece pairs are counted
    """
    collision_count = 0
    n = len(pieces)

    if n < 3:
        return 0

    # Identify switch positions and adjacent areas in pieces list
    switch_adjacent = set()
    for i, piece in enumerate(pieces):
        if piece.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT):
            # Pieces within 3 positions of switch are adjacent to siding
            for offset in range(-3, 4):
                if 0 <= i + offset < n:
                    switch_adjacent.add(i + offset)

    # Get main track pieces for closure detection
    main_piece_indices = [i for i, p in enumerate(pieces) if not p.is_branch_piece]
    n_main = len(main_piece_indices)

    for i in range(n):
        for j in range(i + 2, n):
            # Skip adjacent pieces
            if abs(i - j) <= 1:
                continue

            pi = pieces[i]
            pj = pieces[j]

            # Skip pieces near the closure point on MAIN track
            if not pi.is_branch_piece and not pj.is_branch_piece:
                if i in main_piece_indices and j in main_piece_indices:
                    i_main_idx = main_piece_indices.index(i)
                    j_main_idx = main_piece_indices.index(j)

                    # Allow main track closure - first and last ~5 main pieces
                    if i_main_idx < 5 and j_main_idx >= n_main - 5:
                        continue
                    if j_main_idx < 5 and i_main_idx >= n_main - 5:
                        continue

            # Allow branch pieces to be close to switch-adjacent main track pieces
            # This is expected for a passing siding
            i_is_branch = pi.is_branch_piece
            j_is_branch = pj.is_branch_piece
            i_is_siding = i in switch_adjacent
            j_is_siding = j in switch_adjacent

            # Branch piece near siding area - this is the passing siding, skip
            if (i_is_branch and j_is_siding) or (j_is_branch and i_is_siding):
                continue
            # Both are branch pieces - skip collision between them too
            if i_is_branch and j_is_branch:
                continue
            # Branch piece near any main track piece in switch area
            if i_is_branch and not pj.is_branch_piece:
                continue  # Branch is allowed near main track
            if j_is_branch and not pi.is_branch_piece:
                continue  # Branch is allowed near main track

            # Simple point-to-point distance check
            dist = min(
                pi.end_pos.distance_to(pj.start_pos),
                pi.end_pos.distance_to(pj.end_pos),
                pi.start_pos.distance_to(pj.start_pos),
                pi.start_pos.distance_to(pj.end_pos)
            )

            if dist < min_distance:
                collision_count += 1

    return collision_count


# ============================================================================
# INDIVIDUAL AND FITNESS
# ============================================================================

@dataclass
class FitnessResult:
    """Detailed fitness evaluation results"""
    fitness: float
    is_feasible: bool
    num_pieces: int
    pos_error: float
    angle_error: float
    boundary_violations: int
    boundary_distance: float
    collision_count: int
    inventory_violation: bool
    pieces_by_type: Dict[PieceType, int] = field(default_factory=dict)
    # Branch track info (for switches)
    has_switches: bool = False
    branch_closed: bool = True  # True if no switches or branch properly closes
    branch_pos_error: float = 0.0
    branch_angle_error: float = 0.0
    # Space utilization metrics
    x_spread: float = 0.0  # Track width as fraction of boundary width
    y_spread: float = 0.0  # Track height as fraction of boundary height
    space_utilization: float = 0.0  # Combined metric (0-1)


@dataclass
class Individual:
    """Individual in the population representing a track layout"""
    genes: List[Gene]
    fitness_result: Optional[FitnessResult] = None
    age: int = 0

    def copy(self) -> 'Individual':
        return Individual(
            genes=[g.copy() for g in self.genes],
            fitness_result=self.fitness_result,
            age=self.age
        )

    def count_pieces(self) -> Dict[PieceType, int]:
        counts = {pt: 0 for pt in PieceType}
        for gene in self.genes:
            counts[gene.piece_type] += 1
        return counts


def validate_branch_closure(pieces: List[TrackPiece],
                            inventory: Optional['PieceInventory'] = None,
                            used_pieces: Optional[Dict[PieceType, int]] = None,
                            genes: Optional[List['Gene']] = None) -> Tuple[bool, float, float]:
    """
    Validate that switch branches form a closed loop by computing actual geometry.

    For a proper passing siding:
    - Opening switch (normal) sends branch out
    - Branch track travels parallel to main
    - Closing switch (reversed) receives branch back

    Returns: (is_closed, pos_error, angle_error)
    """
    # Find main track pieces (non-branch)
    main_pieces = [p for p in pieces if not p.is_branch_piece]
    branch_pieces = [p for p in pieces if p.is_branch_piece]

    # Find switches in main track
    main_switches = [(i, p) for i, p in enumerate(main_pieces)
                     if p.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT)]

    if len(main_switches) == 0:
        # No switches - no branch to validate
        return True, 0.0, 0.0

    if len(main_switches) == 1:
        # Single switch creates open branch - not closed
        return False, float('inf'), float('inf')

    if not branch_pieces:
        # Switches but no branch track
        return False, float('inf'), float('inf')

    # For 2 switches (passing siding)
    if len(main_switches) == 2:
        (_, sw1), (_, sw2) = main_switches

        # Find normal (opening) and reversed (closing) switches
        opening_sw = sw1 if not sw1.is_reversed else sw2
        closing_sw = sw2 if sw2.is_reversed else sw1

        if opening_sw.is_reversed or not closing_sw.is_reversed:
            # No proper normal/reversed pair
            return False, float('inf'), float('inf')

        # Branch start: opening switch's branch output
        if not opening_sw.branch_end_pos:
            return False, float('inf'), float('inf')

        branch_start = opening_sw.branch_end_pos
        branch_start_angle = opening_sw.branch_end_angle

        # Branch end: last branch piece's end position
        branch_end = branch_pieces[-1].end_pos
        branch_end_angle = branch_pieces[-1].end_angle

        # Closing switch's branch entry point
        # For a reversed switch, the branch entry is near start_pos (frog at start)
        # Calculate where branch should connect based on closing switch geometry
        closing_angle_rad = math.radians(closing_sw.start_angle)
        if closing_sw.piece_type == PieceType.SWITCH_LEFT:
            # Branch approaches from left at +SWITCH_ANGLE from main
            branch_approach_angle = closing_sw.start_angle + SWITCH_ANGLE
        else:
            # Branch approaches from right at -SWITCH_ANGLE from main
            branch_approach_angle = closing_sw.start_angle - SWITCH_ANGLE

        # The branch should end up at position that aligns with closing switch
        # For a reversed switch, branch enters at the "frog" which is at start_pos
        # The branch endpoint should be at: start_pos + offset at branch angle
        branch_offset_angle_rad = math.radians(branch_approach_angle)
        expected_branch_end = Point(
            closing_sw.start_pos.x + SWITCH_LENGTH * math.cos(branch_offset_angle_rad) -
            SWITCH_LENGTH * math.cos(closing_angle_rad),
            closing_sw.start_pos.y + SWITCH_LENGTH * math.sin(branch_offset_angle_rad) -
            SWITCH_LENGTH * math.sin(closing_angle_rad)
        )

        # Simpler: for same-type switches, the branch ports should be at same relative position
        # Opening switch branch port offset from switch start
        # Closing switch (reversed) expects branch to arrive at similar offset from ITS start

        # Actually use the directly computed branch positions from the switches
        if opening_sw.branch_end_pos and closing_sw.branch_end_pos:
            # For reversed switch, branch_end_pos was computed for where branch should connect
            expected_end = closing_sw.branch_end_pos

            pos_error = branch_end.distance_to(expected_end)

            # Angle: branch end angle should match closing switch's branch_end_angle
            # Both represent the direction of the branch port/track
            expected_angle = closing_sw.branch_end_angle
            if expected_angle is not None:
                # Normalize angles to 0-360 range for comparison
                expected_norm = expected_angle % 360
                actual_norm = branch_end_angle % 360
                angle_diff = abs(actual_norm - expected_norm)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                angle_error = angle_diff
            else:
                angle_error = 0.0

            # Use more lenient tolerance for siding branch closure
            is_closed = pos_error < BRANCH_POSITION_TOLERANCE and angle_error < ANGLE_TOLERANCE
            return is_closed, pos_error, angle_error

    # More than 2 switches or complex topology
    return False, float('inf'), float('inf')


def evaluate_individual(individual: Individual,
                        boundary: Tuple[float, float, float, float],
                        inventory: Optional[PieceInventory] = None,
                        phase: int = 1) -> FitnessResult:
    """
    Evaluate fitness of an individual.

    Phase 1: Optimize basic track (straights + curves only)
    Phase 2: Add switches to maximize piece usage

    Fitness strategy:
    - Feasible solutions ALWAYS beat infeasible (Deb's constraint handling)
    - Among feasible: maximize pieces, then minimize closure error
    - Among infeasible: minimize constraint violations while rewarding pieces
    """
    genes = individual.genes
    num_pieces = len(genes)

    if num_pieces == 0:
        return FitnessResult(
            fitness=-1e10, is_feasible=False, num_pieces=0,
            pos_error=float('inf'), angle_error=float('inf'),
            boundary_violations=0, boundary_distance=0,
            collision_count=0, inventory_violation=True
        )

    # Build track using forward propagation, centered in boundary
    pieces = build_track_centered(genes, boundary)

    # Check inventory constraints
    piece_counts = individual.count_pieces()
    inventory_violation = False
    if inventory:
        for pt, count in piece_counts.items():
            if count > inventory.get_count(pt):
                inventory_violation = True
                break

    # Calculate closure error (pass genes to filter out branch pieces)
    pos_error, angle_error = calculate_closure_error(pieces, genes=genes)

    # Check boundary violations
    boundary_violations, boundary_distance = check_boundary_violations(pieces, boundary)

    # Check collisions (pass genes to handle branch pieces properly)
    collision_count = check_collisions(pieces, genes=genes)

    # Determine feasibility - all hard constraints must be satisfied
    is_closed = pos_error < POSITION_TOLERANCE and angle_error < ANGLE_TOLERANCE
    is_in_bounds = boundary_violations == 0
    is_collision_free = collision_count == 0
    has_valid_inventory = not inventory_violation

    is_feasible = is_closed and is_in_bounds and is_collision_free and has_valid_inventory

    # Calculate weighted piece count
    # Switches are 32 studs (2x straight length), so count as 2 pieces
    # This ensures tracks with switches aren't penalized for "fewer pieces"
    weighted_pieces = 0
    switch_count = 0
    for pt, count in piece_counts.items():
        if pt in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT):
            weighted_pieces += count * 2  # Switches count as 2
            switch_count += count
        else:
            weighted_pieces += count

    # Calculate fitness using Deb's constraint handling
    # Feasible solutions always dominate infeasible ones
    if is_feasible:
        # Maximize weighted piece count with bonuses for good closure
        closure_quality = max(0, 1 - pos_error / POSITION_TOLERANCE) * 1000
        # Add bonus for using switches (encourages switch usage when possible)
        switch_bonus = switch_count * 500  # Bonus per switch
        fitness = weighted_pieces * 1000 + closure_quality + switch_bonus
    else:
        # Infeasible: minimize constraint violations
        # Use negative fitness to ensure feasible always wins
        constraint_violation = 0.0

        if not is_closed:
            constraint_violation += pos_error * 100 + angle_error * 10

        if not is_in_bounds:
            constraint_violation += boundary_violations * 1000 + boundary_distance * 100

        if not is_collision_free:
            constraint_violation += collision_count * 500

        if not has_valid_inventory:
            constraint_violation += 10000

        # Negative fitness, but still reward more pieces slightly
        fitness = -constraint_violation + weighted_pieces * 10

    # Check for switches and branch closure
    has_switches = any(g.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT)
                       for g in genes)
    branch_closed = True
    branch_pos_error = 0.0
    branch_angle_error = 0.0

    if has_switches:
        branch_closed, branch_pos_error, branch_angle_error = validate_branch_closure(
            pieces, inventory, piece_counts, genes
        )

        # If branch doesn't close, solution is infeasible
        if not branch_closed:
            is_feasible = False
            # Add branch violation to constraint score
            if not is_feasible:
                constraint_violation = pos_error * 100 + angle_error * 10
                constraint_violation += branch_pos_error * 50 + branch_angle_error * 5
                constraint_violation += boundary_violations * 1000 + boundary_distance * 100
                constraint_violation += collision_count * 500
                if inventory_violation:
                    constraint_violation += 10000
                fitness = -constraint_violation + num_pieces * 10

    # Calculate space utilization metrics
    track_bounds = calculate_track_bounds(pieces)
    track_x_min, track_x_max, track_y_min, track_y_max = track_bounds
    track_width = track_x_max - track_x_min
    track_height = track_y_max - track_y_min

    bx_min, bx_max, by_min, by_max = boundary
    boundary_width = bx_max - bx_min
    boundary_height = by_max - by_min

    # Calculate spread as fraction of usable boundary (with margin)
    margin = 20.0  # Safety margin from edges
    usable_width = max(1, boundary_width - 2 * margin)
    usable_height = max(1, boundary_height - 2 * margin)

    x_spread = min(1.0, track_width / usable_width)
    y_spread = min(1.0, track_height / usable_height)

    # Space utilization: geometric mean emphasizes balanced use of both dimensions
    # This rewards tracks that use BOTH dimensions, not just one
    space_utilization = math.sqrt(x_spread * y_spread)

    # Add space utilization bonus for feasible solutions
    if is_feasible:
        # Bonus for utilizing vertical space (Y-axis)
        # Weight Y-spread higher to encourage vertical exploration
        vertical_bonus = y_spread * 2000  # Up to 2000 points for full Y utilization
        balanced_bonus = space_utilization * 1000  # Up to 1000 for balanced use
        fitness += vertical_bonus + balanced_bonus

    return FitnessResult(
        fitness=fitness,
        is_feasible=is_feasible,
        num_pieces=num_pieces,
        pos_error=pos_error,
        angle_error=angle_error,
        boundary_violations=boundary_violations,
        boundary_distance=boundary_distance,
        collision_count=collision_count,
        inventory_violation=inventory_violation,
        pieces_by_type=piece_counts,
        has_switches=has_switches,
        branch_closed=branch_closed,
        branch_pos_error=branch_pos_error,
        branch_angle_error=branch_angle_error,
        x_spread=x_spread,
        y_spread=y_spread,
        space_utilization=space_utilization
    )


# ============================================================================
# MUTATION OPERATORS (Mutation-Only GA)
# ============================================================================

class MutationOperator:
    """
    Mutation-only genetic algorithm operators.

    Three main operators:
    - ADD: Insert new piece at strategic position
    - MUTATE: Change piece type while maintaining connectivity
    - DELETE: Remove low-quality pieces
    """

    def __init__(self, inventory: Optional[PieceInventory] = None,
                 boundary: Tuple[float, float, float, float] = None,
                 phase: int = 1):
        self.inventory = inventory
        self.boundary = boundary
        self.phase = phase

        # Adaptive mutation rates
        self.add_rate = 0.3
        self.mutate_rate = 0.4
        self.delete_rate = 0.3

    def insert_passing_siding(self, individual: Individual) -> Individual:
        """
        Insert a passing siding with OPPOSITE switch types for opening/closing.

        CORRECT SIDING PATTERNS (opening → closing must be OPPOSITE types):

        Left Siding:
          - Opening: SL (Normal) - branch diverges LEFT at +22.5°
          - Closing: SR (Reversed) - branch converges from RIGHT at -22.5°
          - Branch pattern: R → S → R (crosses from left to right side)

          Angular transformation:
            +22.5° → R(-22.5°) → 0° → S(0°) → 0° → R(-22.5°) → -22.5°

          Main:   --[SL Normal]----S----S----[SR Reversed]--
                       \\                         //
          Branch:       R------S------R
                    (left→parallel→right: CROSSES OVER)

        Right Siding:
          - Opening: SR (Normal) - branch diverges RIGHT at -22.5°
          - Closing: SL (Reversed) - branch converges from LEFT at +22.5°
          - Branch pattern: L → S → L (crosses from right to left side)

          Angular transformation:
            -22.5° → L(+22.5°) → 0° → S(0°) → 0° → L(+22.5°) → +22.5°

          Main:   --[SR Normal]----S----S----[SL Reversed]--
                       //                         \\
          Branch:       L------S------L
                    (right→parallel→left: CROSSES OVER)

        Key: Both curves in branch are SAME type (R+R or L+L), creating crossover
        """
        if not self.inventory:
            return individual.copy()

        # Check if we can insert a passing siding
        switch_analysis = analyze_switch_feasibility(self.inventory)

        genes = [g.copy() for g in individual.genes]
        counts = {pt: 0 for pt in PieceType}
        for g in genes:
            counts[g.piece_type] += 1

        # Check feasibility based on overall inventory
        can_left_siding = switch_analysis['passing_siding_left']
        can_right_siding = switch_analysis['passing_siding_right']

        if not can_left_siding and not can_right_siding:
            return individual.copy()

        # Both siding types need 1 SL + 1 SR (opposite switch types)
        switches_available_left = self.inventory.switch_left - counts.get(PieceType.SWITCH_LEFT, 0)
        switches_available_right = self.inventory.switch_right - counts.get(PieceType.SWITCH_RIGHT, 0)

        # Need at least 1 of each switch type for any siding
        if switches_available_left < 1 or switches_available_right < 1:
            return individual.copy()

        # Check spare curves for branch pattern
        spare_right = self.inventory.curve_right - counts.get(PieceType.CURVE_RIGHT, 0)
        spare_left = self.inventory.curve_left - counts.get(PieceType.CURVE_LEFT, 0)

        # Left siding needs R-S-R (2+ right curves)
        can_left_siding = can_left_siding and spare_right >= 2
        # Right siding needs L-S-L (2+ left curves)
        can_right_siding = can_right_siding and spare_left >= 2

        if not can_left_siding and not can_right_siding:
            return individual.copy()

        # Choose siding type based on available curves
        if can_left_siding and can_right_siding:
            is_left = spare_right >= spare_left
        elif can_left_siding:
            is_left = True
        else:
            is_left = False

        # OPPOSITE switch types for opening and closing
        if is_left:
            # Left siding: SL opens, SR closes
            opening_switch = PieceType.SWITCH_LEFT
            closing_switch = PieceType.SWITCH_RIGHT
            branch_curve = PieceType.CURVE_RIGHT  # R-S-R pattern (same curve type)
        else:
            # Right siding: SR opens, SL closes
            opening_switch = PieceType.SWITCH_RIGHT
            closing_switch = PieceType.SWITCH_LEFT
            branch_curve = PieceType.CURVE_LEFT  # L-S-L pattern (same curve type)

        # Find a run of consecutive straights (need at least 6)
        straight_runs = []
        i = 0
        while i < len(genes):
            if genes[i].piece_type == PieceType.STRAIGHT:
                run_start = i
                run_length = 0
                while i < len(genes) and genes[i].piece_type == PieceType.STRAIGHT:
                    run_length += 1
                    i += 1
                if run_length >= 6:
                    straight_runs.append((run_start, run_length))
            else:
                i += 1

        if not straight_runs:
            return individual.copy()

        # Choose a run to use
        run_start, run_length = random.choice(straight_runs)

        # Calculate straights allocation
        middle_straights = min(run_length - 4, 4)
        if middle_straights < 2:
            middle_straights = 2

        # Branch straights for the parallel section
        branch_straights = max(1, middle_straights - 1)

        non_run_straights = counts.get(PieceType.STRAIGHT, 0) - run_length
        straights_needed = middle_straights + branch_straights
        straights_available = self.inventory.straight - non_run_straights

        if straights_available < straights_needed:
            branch_straights = max(1, straights_available - middle_straights)
            if branch_straights < 1:
                return individual.copy()

        # Build the new gene sequence
        new_genes = genes[:run_start]

        # === MAIN TRACK SECTION ===
        # Opening switch (Normal orientation)
        new_genes.append(Gene(opening_switch, is_reversed=False))

        # Middle straights between switches on main track
        for _ in range(middle_straights):
            new_genes.append(Gene(PieceType.STRAIGHT))

        # Closing switch (Reversed orientation - OPPOSITE type from opening)
        new_genes.append(Gene(closing_switch, is_reversed=True))

        # === BRANCH SECTION ===
        # Pattern: R-S-R for left siding, L-S-L for right siding
        # Both curves are SAME type - this creates the crossover effect

        # First curve (turns toward parallel)
        new_genes.append(Gene(branch_curve, is_branch=True))

        # Straights (parallel section)
        for _ in range(branch_straights):
            new_genes.append(Gene(PieceType.STRAIGHT, is_branch=True))

        # Second curve (turns to approach closing switch from opposite side)
        new_genes.append(Gene(branch_curve, is_branch=True))

        # Add any remaining straights from original run
        used_straights = 4 + middle_straights
        remaining = run_length - used_straights
        for _ in range(max(0, remaining)):
            new_genes.append(Gene(PieceType.STRAIGHT))

        # Add rest of original track
        new_genes.extend(genes[run_start + run_length:])

        return Individual(genes=new_genes)

    def get_available_piece_types(self, current_genes: List[Gene]) -> List[PieceType]:
        """Get piece types that can still be used based on inventory"""
        if not self.inventory:
            if self.phase == 1:
                return [PieceType.STRAIGHT, PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT]
            else:
                return list(PieceType)

        # Count current usage
        counts = {pt: 0 for pt in PieceType}
        for gene in current_genes:
            counts[gene.piece_type] += 1

        # Find available types
        available = []
        inventory_counts = {
            PieceType.STRAIGHT: self.inventory.straight,
            PieceType.CURVE_LEFT: self.inventory.curve_left,
            PieceType.CURVE_RIGHT: self.inventory.curve_right,
            PieceType.SWITCH_LEFT: self.inventory.switch_left,
            PieceType.SWITCH_RIGHT: self.inventory.switch_right
        }

        for pt, max_count in inventory_counts.items():
            if counts[pt] < max_count:
                # Phase 1: only basic pieces
                if self.phase == 1 and pt in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT):
                    continue
                available.append(pt)

        return available

    def mutate_add(self, individual: Individual) -> Individual:
        """
        ADD operator: Insert a new piece at a strategic position.
        Prioritizes positions that improve closure.
        """
        genes = [g.copy() for g in individual.genes]
        available = self.get_available_piece_types(genes)

        if not available:
            return Individual(genes=genes)

        # Choose position - favor positions near closure issues
        # PROTECT SWITCH STRUCTURE: Don't insert between paired switches
        if len(genes) == 0:
            insert_pos = 0
        elif len(genes) < 8:
            # For short tracks, add at end to grow
            insert_pos = len(genes)
        else:
            # Find switch positions to avoid inserting between them
            switch_indices = [i for i, g in enumerate(genes)
                              if g.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT)]

            # Strategically choose: near end to fix closure, or middle for variety
            if random.random() < 0.6:
                # Near end - affects closure
                insert_pos = random.randint(max(0, len(genes) - 4), len(genes))
            else:
                # Random position for exploration
                insert_pos = random.randint(0, len(genes))

            # Avoid inserting directly between paired switches
            if len(switch_indices) >= 2:
                # If inserting between first and second switch, adjust
                first_sw, last_sw = switch_indices[0], switch_indices[-1]
                if first_sw < insert_pos <= last_sw:
                    # Move insertion to after switch pair
                    if random.random() < 0.5:
                        insert_pos = last_sw + 1
                    else:
                        insert_pos = first_sw

        # Choose piece type - bias towards curves if closure is needed
        # Never add switches through normal mutation (use insert_passing_siding)
        non_switch_available = [pt for pt in available
                                if pt not in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT)]

        if not non_switch_available:
            return Individual(genes=genes)

        if random.random() < 0.7:
            # Prefer curves for loop closure
            curve_types = [pt for pt in non_switch_available
                           if pt in (PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT)]
            if curve_types:
                new_type = random.choice(curve_types)
            else:
                new_type = random.choice(non_switch_available)
        else:
            new_type = random.choice(non_switch_available)

        new_gene = Gene(new_type)
        genes.insert(insert_pos, new_gene)

        return Individual(genes=genes)

    def mutate_change(self, individual: Individual) -> Individual:
        """
        MUTATE operator: Change piece type at a position.
        Uses forward propagation to maintain connectivity.
        """
        if len(individual.genes) == 0:
            return individual.copy()

        genes = [g.copy() for g in individual.genes]

        # Find positions with "worst" genes (contributing to constraint violations)
        pieces = build_track_from_genes(genes)
        pos_error, angle_error = calculate_closure_error(pieces)

        # Prioritize mutation positions near the end if closure is bad
        if pos_error > POSITION_TOLERANCE or angle_error > ANGLE_TOLERANCE:
            # Focus on last third of track
            start_idx = max(0, len(genes) * 2 // 3)
            mut_idx = random.randint(start_idx, len(genes) - 1)
        else:
            # Random position when track is already closing well
            mut_idx = random.randint(0, len(genes) - 1)

        # PROTECT SWITCHES: Don't change switch genes (they're carefully placed)
        if genes[mut_idx].piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT):
            # Find a non-switch gene to mutate instead
            non_switch_indices = [i for i, g in enumerate(genes)
                                  if g.piece_type not in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT)]
            if non_switch_indices:
                mut_idx = random.choice(non_switch_indices)
            else:
                return Individual(genes=genes)  # All switches, don't mutate

        # Get available types for swap
        old_type = genes[mut_idx].piece_type
        temp_genes = [g.copy() for g in genes]
        temp_genes.pop(mut_idx)
        available = self.get_available_piece_types(temp_genes)

        # Remove current type from options if we want to actually change
        # Also don't change non-switches into switches (preserve siding structure)
        available = [pt for pt in available
                     if pt != old_type and pt not in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT)]

        if available:
            new_type = random.choice(available)
            genes[mut_idx].piece_type = new_type

        return Individual(genes=genes)

    def mutate_delete(self, individual: Individual) -> Individual:
        """
        DELETE operator: Remove a piece that contributes least to solution quality.
        Maintains minimum viable track length.
        BOUNDARY-AWARE: Prioritizes removing pieces causing boundary violations.
        """
        min_length = 8  # Minimum for any viable closed track

        if len(individual.genes) <= min_length:
            return individual.copy()

        genes = [g.copy() for g in individual.genes]

        # Find candidate positions for deletion
        # Prefer deleting pieces that don't contribute much
        pieces = build_track_from_genes(genes)

        # Check for boundary violations
        if self.boundary:
            _, boundary_dist = check_boundary_violations(pieces, self.boundary)

            # If we have boundary violations, prioritize removing straights
            # (they extend the track length)
            if boundary_dist > 0:
                straight_indices = [i for i, g in enumerate(genes)
                                    if g.piece_type == PieceType.STRAIGHT]
                if straight_indices:
                    del_idx = random.choice(straight_indices)
                    genes.pop(del_idx)
                    return Individual(genes=genes)

        # Score each piece - lower is better for deletion
        # PROTECT SWITCHES: Give them very low score so they're never deleted
        scores = []
        for i, gene in enumerate(genes):
            # Switches should NEVER be deleted (they're paired)
            if gene.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT):
                scores.append((-1000, i))  # Very low score = never selected
                continue

            score = 0
            # Straights in middle of long straight runs are good deletion candidates
            if gene.piece_type == PieceType.STRAIGHT:
                score += 1
            # Check if piece is contributing to boundary violations
            if self.boundary and i < len(pieces):
                piece = pieces[i]
                x_min, x_max, y_min, y_max = self.boundary
                for pt in [piece.start_pos, piece.end_pos]:
                    dist_to_edge = min(
                        pt.x - x_min, x_max - pt.x,
                        pt.y - y_min, y_max - pt.y
                    )
                    if dist_to_edge < 20:  # Close to boundary
                        score += 2
            scores.append((score, i))

        # Select from worst candidates (but never switches)
        scores.sort(reverse=True)
        top_candidates = [idx for score, idx in scores[:max(1, len(scores) // 3)] if score >= 0]

        if not top_candidates:
            # All candidates are switches, don't delete anything
            return Individual(genes=genes)

        del_idx = random.choice(top_candidates)

        genes.pop(del_idx)
        return Individual(genes=genes)

    def mutate_shrink_for_boundary(self, individual: Individual) -> Individual:
        """
        SHRINK operator: Remove straights to fit within boundary.
        Used when track exceeds boundary limits.
        """
        if not self.boundary:
            return individual.copy()

        genes = [g.copy() for g in individual.genes]
        pieces = build_track_from_genes(genes)

        # Check if we violate boundary
        violations, _ = check_boundary_violations(pieces, self.boundary)

        if violations == 0:
            return Individual(genes=genes)

        # Count straights
        straight_indices = [i for i, g in enumerate(genes)
                            if g.piece_type == PieceType.STRAIGHT]

        if not straight_indices:
            return Individual(genes=genes)

        # Remove straights one at a time until we fit
        # Remove from both ends to maintain symmetry
        while straight_indices and violations > 0:
            # Remove last straight
            if straight_indices:
                del_idx = straight_indices.pop()
                genes.pop(del_idx)

                # Recalculate
                pieces = build_track_from_genes(genes)
                violations, _ = check_boundary_violations(pieces, self.boundary)

                # Update indices
                straight_indices = [i for i, g in enumerate(genes)
                                    if g.piece_type == PieceType.STRAIGHT]

        return Individual(genes=genes)

    def mutate_swap(self, individual: Individual) -> Individual:
        """
        SWAP operator: Swap positions of two pieces.
        Helps explore different orderings.
        PROTECTS SWITCHES: Never swaps switch positions (they're paired).
        """
        if len(individual.genes) < 2:
            return individual.copy()

        genes = [g.copy() for g in individual.genes]

        # Find non-switch indices (only swap non-switch pieces)
        non_switch_indices = [i for i, g in enumerate(genes)
                              if g.piece_type not in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT)]

        if len(non_switch_indices) < 2:
            return Individual(genes=genes)  # Not enough non-switch pieces to swap

        idx1 = random.choice(non_switch_indices)
        idx2 = random.choice([i for i in non_switch_indices if i != idx1])

        genes[idx1], genes[idx2] = genes[idx2], genes[idx1]
        return Individual(genes=genes)

    def mutate_for_vertical_spread(self, individual: Individual) -> Individual:
        """
        VERTICAL SPREAD mutation: Reorganize track to increase Y-axis utilization.

        Strategy: Move straights from horizontal sections (near 0°/180° angle)
        to vertical sections (near 90°/270° angle).

        This is done by:
        1. Finding straights that are in horizontal sections
        2. Moving them to positions after 4 curves (which creates 90° angle)
        """
        genes = [g.copy() for g in individual.genes]

        if len(genes) < 8:
            return Individual(genes=genes)

        # Find curve runs of 4+ (creates 90° corners)
        curve_runs = []
        i = 0
        while i < len(genes):
            if genes[i].piece_type in (PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT):
                run_start = i
                curve_type = genes[i].piece_type
                run_length = 0
                while i < len(genes) and genes[i].piece_type == curve_type:
                    run_length += 1
                    i += 1
                if run_length >= 4:
                    curve_runs.append((run_start, run_length, curve_type))
            else:
                i += 1

        # Find straight runs
        straight_runs = []
        i = 0
        while i < len(genes):
            if genes[i].piece_type == PieceType.STRAIGHT:
                run_start = i
                run_length = 0
                while i < len(genes) and genes[i].piece_type == PieceType.STRAIGHT:
                    run_length += 1
                    i += 1
                straight_runs.append((run_start, run_length))
            else:
                i += 1

        if not curve_runs or not straight_runs:
            return Individual(genes=genes)

        # Try to create a pattern with straights after 4 curves
        # This puts straights in vertical sections

        # Find a curve run that can be split
        for curve_start, curve_len, curve_type in curve_runs:
            if curve_len >= 8:
                # Split an 8-curve section into 4-straight-4
                # This converts: CCCCCCCC to CCCC-SS-CCCC

                # Find straights we can move here
                for str_start, str_len in straight_runs:
                    # Don't move straights that are already after a 4-curve run
                    # Check if this straight run is in a "horizontal" position

                    # Move 1-2 straights to after the 4th curve
                    num_to_move = min(2, str_len)

                    # Remove straights from original position
                    for _ in range(num_to_move):
                        if str_start < len(genes):
                            genes.pop(str_start)

                    # Insert after 4th curve
                    insert_pos = min(curve_start + 4, len(genes))
                    for _ in range(num_to_move):
                        genes.insert(insert_pos, Gene(PieceType.STRAIGHT))

                    return Individual(genes=genes)

        return Individual(genes=genes)

    def mutate_for_closure(self, individual: Individual) -> Individual:
        """
        Specialized mutation targeting track closure.
        Analyzes current track and makes directed changes to improve closure.
        """
        genes = [g.copy() for g in individual.genes]

        if len(genes) < 4:
            return Individual(genes=genes)

        # Build current track and analyze
        pieces = build_track_from_genes(genes)
        pos_error, angle_error = calculate_closure_error(pieces)

        # Calculate net angle
        total_angle = sum(p.end_angle - p.start_angle for p in pieces)
        net_angle = total_angle % 360

        # Determine what we need to close
        # For closure, we want net_angle close to 0 or 360
        if net_angle > 180:
            # We have too many right curves (negative angle)
            # Need more left curves or remove right curves
            need_more_left = True
        else:
            # We have too many left curves (positive angle)
            # Need more right curves or remove left curves
            need_more_left = False

        # Strategy 1: Change a curve type in the latter half of track
        changed = False
        for i in range(len(genes) - 1, len(genes) // 2, -1):
            gene = genes[i]
            if gene.piece_type == PieceType.CURVE_LEFT and not need_more_left:
                # Try to change to right
                if self._can_change_to(genes, i, PieceType.CURVE_RIGHT):
                    genes[i].piece_type = PieceType.CURVE_RIGHT
                    changed = True
                    break
            elif gene.piece_type == PieceType.CURVE_RIGHT and need_more_left:
                # Try to change to left
                if self._can_change_to(genes, i, PieceType.CURVE_LEFT):
                    genes[i].piece_type = PieceType.CURVE_LEFT
                    changed = True
                    break

        # Strategy 2: If position error is large, try adding curves at end
        if not changed and pos_error > POSITION_TOLERANCE:
            available = self.get_available_piece_types(genes)
            if PieceType.CURVE_RIGHT in available and need_more_left == False:
                genes.append(Gene(PieceType.CURVE_RIGHT))
            elif PieceType.CURVE_LEFT in available and need_more_left:
                genes.append(Gene(PieceType.CURVE_LEFT))

        return Individual(genes=genes)

    def _can_change_to(self, genes: List[Gene], idx: int, new_type: PieceType) -> bool:
        """Check if we can change gene at idx to new_type given inventory"""
        if not self.inventory:
            return True

        # Count current usage excluding the gene we want to change
        counts = {pt: 0 for pt in PieceType}
        for i, g in enumerate(genes):
            if i != idx:
                counts[g.piece_type] += 1

        # Check if new_type is available
        return counts[new_type] < self.inventory.get_count(new_type)

    def repair_boundary_violations(self, individual: Individual) -> Individual:
        """
        Repair operator: Remove straights to ensure track fits within boundary.
        Called after any mutation to ensure boundary compliance.
        """
        if not self.boundary:
            return individual

        genes = [g.copy() for g in individual.genes]

        # Build centered track and check violations
        pieces = build_track_centered(genes, self.boundary)
        violations, _ = check_boundary_violations(pieces, self.boundary)

        if violations == 0:
            return Individual(genes=genes)

        # Remove straights until we fit (they extend the track most)
        min_length = 8  # Minimum viable loop

        while violations > 0 and len(genes) > min_length:
            # Find straights
            straight_indices = [i for i, g in enumerate(genes)
                                if g.piece_type == PieceType.STRAIGHT]

            if not straight_indices:
                break  # No straights to remove

            # Remove the last straight (affects closure least)
            del_idx = straight_indices[-1]
            genes.pop(del_idx)

            # Rebuild and check
            pieces = build_track_centered(genes, self.boundary)
            violations, _ = check_boundary_violations(pieces, self.boundary)

        return Individual(genes=genes)

    def mutate(self, individual: Individual) -> Individual:
        """
        Apply a mutation operator probabilistically.
        Implements multi-mutation selection for better exploration.
        BOUNDARY-AWARE: Repairs boundary violations after every mutation.
        PHASE-2: Can insert passing sidings with switches.
        """
        # First check if we have boundary violations - use shrink if so
        if individual.fitness_result and self.boundary:
            if individual.fitness_result.boundary_violations > 0:
                # High probability of shrink mutation
                if random.random() < 0.7:
                    mutated = self.mutate_shrink_for_boundary(individual)
                    return self.repair_boundary_violations(mutated)

        # In Phase 2, try to insert passing siding with significant probability
        if self.phase == 2 and random.random() < 0.3:
            # Check if we already have switches
            switch_count = sum(1 for g in individual.genes
                               if g.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT))
            if switch_count == 0:
                # Try to insert a passing siding
                mutated = self.insert_passing_siding(individual)
                if mutated.genes != individual.genes:
                    return self.repair_boundary_violations(mutated)

        # Check if we need closure-focused mutation
        if individual.fitness_result:
            fr = individual.fitness_result
            if not fr.is_feasible and (fr.pos_error > POSITION_TOLERANCE or
                                       fr.angle_error > ANGLE_TOLERANCE):
                # High probability of closure-focused mutation
                if random.random() < 0.5:
                    mutated = self.mutate_for_closure(individual)
                    return self.repair_boundary_violations(mutated)

            # Check if we need vertical spread improvement
            if fr.is_feasible and fr.y_spread < 0.5:
                # Track is too horizontally biased - try vertical spread mutation
                if random.random() < 0.3:
                    mutated = self.mutate_for_vertical_spread(individual)
                    return self.repair_boundary_violations(mutated)

        # Adaptive operator selection with vertical spread option
        r = random.random()

        if r < self.add_rate:
            mutated = self.mutate_add(individual)
        elif r < self.add_rate + self.mutate_rate:
            mutated = self.mutate_change(individual)
        elif r < self.add_rate + self.mutate_rate + self.delete_rate:
            mutated = self.mutate_delete(individual)
        elif r < self.add_rate + self.mutate_rate + self.delete_rate + 0.1:
            # 10% chance of vertical spread mutation
            mutated = self.mutate_for_vertical_spread(individual)
        else:
            mutated = self.mutate_swap(individual)

        # Always repair boundary violations after mutation
        return self.repair_boundary_violations(mutated)

    def apply_multiple_mutations(self, individual: Individual,
                                 num_mutations: int = 3) -> List[Individual]:
        """
        Select Best Mutation (SBM): Apply multiple mutations and return all.
        Caller selects the best offspring.
        """
        offspring = []
        for _ in range(num_mutations):
            child = self.mutate(individual)
            offspring.append(child)
        return offspring


# ============================================================================
# TRACK REPAIR OPERATOR
# ============================================================================

class TrackRepairer:
    """
    Repair operator for fixing infeasible tracks.
    Uses greedy heuristics to restore closure while maintaining connectivity.
    """

    def __init__(self, inventory: Optional[PieceInventory] = None,
                 boundary: Tuple[float, float, float, float] = None):
        self.inventory = inventory
        self.boundary = boundary

    def repair_closure(self, individual: Individual) -> Individual:
        """
        Attempt to repair track closure by adjusting ending pieces.
        Uses greedy approach to minimize closure error.
        """
        genes = [g.copy() for g in individual.genes]

        if len(genes) < 4:
            return Individual(genes=genes)

        # Build current track and check closure
        pieces = build_track_from_genes(genes)
        pos_error, angle_error = calculate_closure_error(pieces)

        if pos_error < POSITION_TOLERANCE and angle_error < ANGLE_TOLERANCE:
            return Individual(genes=genes)  # Already good

        # Try modifying last few pieces to improve closure
        best_genes = genes
        best_error = pos_error + angle_error

        # Calculate needed angle change to close
        total_angle = sum(p.end_angle - p.start_angle for p in pieces)
        angle_to_360 = (360 - (total_angle % 360)) % 360

        # Determine if we need more left or right curves
        need_left = angle_to_360 > 0 and angle_to_360 < 180
        need_right = angle_to_360 >= 180 or angle_to_360 < 0

        # Try different modifications to last few pieces
        for i in range(min(4, len(genes))):
            idx = len(genes) - 1 - i
            original_type = genes[idx].piece_type

            for new_type in [PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT, PieceType.STRAIGHT]:
                if new_type == original_type:
                    continue

                # Check inventory
                if self.inventory:
                    counts = {pt: 0 for pt in PieceType}
                    for j, g in enumerate(genes):
                        if j != idx:
                            counts[g.piece_type] += 1
                    counts[new_type] += 1

                    if counts[new_type] > self.inventory.get_count(new_type):
                        continue

                test_genes = [g.copy() for g in genes]
                test_genes[idx].piece_type = new_type

                test_pieces = build_track_from_genes(test_genes)
                test_pos_error, test_angle_error = calculate_closure_error(test_pieces)
                total_error = test_pos_error + test_angle_error

                if total_error < best_error:
                    best_error = total_error
                    best_genes = test_genes

        return Individual(genes=best_genes)

    def repair_boundary(self, individual: Individual) -> Individual:
        """
        Attempt to repair boundary violations by modifying problematic sections.
        """
        genes = [g.copy() for g in individual.genes]

        if len(genes) < 2 or self.boundary is None:
            return Individual(genes=genes)

        x_min, x_max, y_min, y_max = self.boundary

        # Find pieces causing boundary violations
        pieces = build_track_from_genes(genes)
        violation_indices = []

        for i, piece in enumerate(pieces):
            for pt in [piece.start_pos, piece.end_pos]:
                if (pt.x < x_min or pt.x > x_max or
                        pt.y < y_min or pt.y > y_max):
                    violation_indices.append(i)
                    break

        if not violation_indices:
            return Individual(genes=genes)

        # Try to fix by changing piece types near violations
        for idx in violation_indices[:3]:  # Fix up to 3 violations
            if idx >= len(genes):
                continue

            # Try changing direction to bring track back
            old_type = genes[idx].piece_type

            # If going out of bounds, try turning the other way
            for new_type in [PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT]:
                if new_type == old_type:
                    continue

                test_genes = [g.copy() for g in genes]
                test_genes[idx].piece_type = new_type

                test_pieces = build_track_from_genes(test_genes)
                test_violations, _ = check_boundary_violations(test_pieces, self.boundary)

                orig_violations, _ = check_boundary_violations(pieces, self.boundary)

                if test_violations < orig_violations:
                    genes = test_genes
                    pieces = test_pieces
                    break

        return Individual(genes=genes)


# ============================================================================
# POPULATION AND EVOLUTION
# ============================================================================

class Population:
    """
    Population management implementing IDEA (Infeasibility-Driven EA).
    Maintains 5-20% infeasible solutions for faster convergence.
    """

    def __init__(self, size: int,
                 boundary: Tuple[float, float, float, float],
                 inventory: Optional[PieceInventory] = None,
                 phase: int = 1,
                 infeasible_ratio: float = 0.15):
        self.size = size
        self.boundary = boundary
        self.inventory = inventory
        self.phase = phase
        self.infeasible_ratio = infeasible_ratio

        self.feasible: List[Individual] = []
        self.infeasible: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None

        self.mutator = MutationOperator(inventory, boundary, phase)
        self.repairer = TrackRepairer(inventory, boundary)

    def initialize(self):
        """Initialize population with mix of feasible templates and random individuals"""
        self.feasible = []
        self.infeasible = []

        # Analyze inventory to choose best initialization strategy
        if self.inventory:
            total_curves = self.inventory.curve_left + self.inventory.curve_right
            max_same_dir = max(self.inventory.curve_left, self.inventory.curve_right)

            # Priority 1: Seed with max-piece patterns (try to use all pieces)
            num_max_piece = min(self.size // 6, 15)
            for _ in range(num_max_piece):
                individual = self._create_max_piece_pattern()
                self._evaluate(individual)
                self._add_to_population(individual)

            # Priority 2: Add VERTICAL patterns for Y-axis utilization
            num_vertical = min(self.size // 4, 15)
            for _ in range(num_vertical):
                if random.random() < 0.5:
                    individual = self._create_vertical_pattern()
                else:
                    individual = self._create_tall_rectangle()
                self._evaluate(individual)
                self._add_to_population(individual)

            if max_same_dir >= 16:
                # Can make a simple circle - add some of these too
                num_templates = min(5, self.size // 6)
                for _ in range(num_templates):
                    individual = self._create_simple_loop()
                    self._evaluate(individual)
                    self._add_to_population(individual)
            elif total_curves >= 16:
                # Have enough curves but split - use oval patterns
                num_templates = min(5, self.size // 6)
                for _ in range(num_templates):
                    individual = self._create_oval_pattern()
                    self._evaluate(individual)
                    self._add_to_population(individual)
            else:
                # Not enough curves for closure - will need straights in pattern
                num_templates = min(5, self.size // 6)
                for _ in range(num_templates):
                    individual = self._create_varied_pattern()
                    self._evaluate(individual)
                    self._add_to_population(individual)
        else:
            # No inventory - use mix of patterns including vertical
            num_vertical = min(10, self.size // 3)
            for _ in range(num_vertical):
                if random.random() < 0.5:
                    individual = self._create_vertical_pattern()
                else:
                    individual = self._create_tall_rectangle()
                self._evaluate(individual)
                self._add_to_population(individual)

            num_templates = min(10, self.size // 3)
            for _ in range(num_templates):
                individual = self._create_simple_loop()
                self._evaluate(individual)
                self._add_to_population(individual)

        # Create varied patterns (including some vertical variants)
        num_varied = self.size // 6
        for _ in range(num_varied):
            individual = self._create_varied_pattern()
            self._evaluate(individual)
            self._add_to_population(individual)

        # Fill rest with random
        while len(self.feasible) + len(self.infeasible) < self.size:
            individual = self._create_random()
            self._evaluate(individual)
            self._add_to_population(individual)

        self._update_best()

    def _create_oval_pattern(self) -> Individual:
        """Create oval pattern optimized for split curve inventory"""
        genes = []

        if not self.inventory:
            return self._create_simple_loop()

        # Use all curves in one direction for half the oval,
        # then use straights to span distance, repeat
        left_curves = self.inventory.curve_left
        right_curves = self.inventory.curve_right
        straights = self.inventory.straight

        # Determine dominant direction
        if right_curves >= left_curves:
            main_curve = PieceType.CURVE_RIGHT
            main_count = right_curves
        else:
            main_curve = PieceType.CURVE_LEFT
            main_count = left_curves

        # Build oval: 8 curves, straights, 8 curves, straights
        curves_per_end = min(8, main_count // 2)
        straights_per_side = min(2, straights // 2)

        for _ in range(curves_per_end):
            genes.append(Gene(main_curve))
        for _ in range(straights_per_side):
            genes.append(Gene(PieceType.STRAIGHT))
        for _ in range(curves_per_end):
            genes.append(Gene(main_curve))
        for _ in range(straights_per_side):
            genes.append(Gene(PieceType.STRAIGHT))

        return Individual(genes=genes)

    def _create_max_piece_pattern(self, safety_margin: float = 10.0) -> Individual:
        """
        Create pattern attempting to use ALL available pieces.
        Uses symmetric oval structure to maximize straight usage.
        BOUNDARY-AWARE: calculates maximum straights that can fit with safety margin.

        Args:
            safety_margin: Extra margin from boundary edge (default 10 studs)
        """
        genes = []

        if not self.inventory:
            return self._create_simple_loop()

        left_curves = self.inventory.curve_left
        right_curves = self.inventory.curve_right
        straights = self.inventory.straight

        # Determine which curve direction to use (need 16 of one type)
        if right_curves >= 16:
            main_curve = PieceType.CURVE_RIGHT
            curves_to_use = 16
        elif left_curves >= 16:
            main_curve = PieceType.CURVE_LEFT
            curves_to_use = 16
        else:
            # Can't form closed loop, use what we have
            if right_curves > left_curves:
                main_curve = PieceType.CURVE_RIGHT
                curves_to_use = right_curves
            else:
                main_curve = PieceType.CURVE_LEFT
                curves_to_use = left_curves

        # BOUNDARY-AWARE: Calculate max straights that fit with safety margin
        boundary_width = self.boundary[1] - self.boundary[0]
        boundary_height = self.boundary[3] - self.boundary[2]

        # Available space after margins
        total_margin = 2 * safety_margin
        available_width = boundary_width - total_margin
        available_height = boundary_height - total_margin

        # Base circle is ~80 studs wide, remaining for straights
        base_track_width = 80.0
        extra_width_for_straights = max(0, available_width - base_track_width)

        # Each straight is 16 studs, and we split them between two sides
        max_straights_per_side = int(extra_width_for_straights / STRAIGHT_LENGTH)
        max_straights_total = max_straights_per_side * 2

        # Use minimum of available inventory and what fits
        straights_to_use = min(straights, max_straights_total)
        straights_per_side = straights_to_use // 2
        remaining_straights = straights_to_use % 2

        # Build symmetric pattern
        # First half-circle (8 curves)
        for _ in range(min(8, curves_to_use // 2)):
            genes.append(Gene(main_curve))

        # First straight section
        for _ in range(straights_per_side + remaining_straights):
            genes.append(Gene(PieceType.STRAIGHT))

        # Second half-circle (8 curves)
        for _ in range(min(8, curves_to_use - curves_to_use // 2)):
            genes.append(Gene(main_curve))

        # Second straight section
        for _ in range(straights_per_side):
            genes.append(Gene(PieceType.STRAIGHT))

        return Individual(genes=genes)

    def _create_simple_loop(self) -> Individual:
        """Create a simple closed loop - guaranteed to close"""
        genes = []

        if self.inventory:
            # Use available curves, prioritizing balance
            available_left = self.inventory.curve_left
            available_right = self.inventory.curve_right
            total_curves = available_left + available_right

            # 16 curves of same type = full circle (16 * 22.5° = 360°)
            # Strategy 1: If we have 16+ of one type, use simple circle
            if available_right >= 16:
                genes = [Gene(PieceType.CURVE_RIGHT) for _ in range(16)]
            elif available_left >= 16:
                genes = [Gene(PieceType.CURVE_LEFT) for _ in range(16)]
            elif total_curves >= 16:
                # Strategy 2: We need NET 16 curves in one direction
                # If we have 8L and 8R, net = 0, won't close
                # Use more of dominant direction
                # For net 360°: (right_count - left_count) * 22.5 = 360
                # or (left_count - right_count) * 22.5 = 360
                # Need net 16 curves one direction

                # Option A: Use all curves in same direction as majority
                if available_right >= available_left:
                    # Use all right curves first
                    for _ in range(available_right):
                        genes.append(Gene(PieceType.CURVE_RIGHT))
                    # Use remaining budget from left (will reduce net angle)
                    remaining_needed = 16 - available_right
                    if remaining_needed > 0 and available_left > 0:
                        # This won't help - opposite direction reduces
                        # Instead, just add straights for variety
                        for _ in range(min(4, self.inventory.straight)):
                            genes.append(Gene(PieceType.STRAIGHT))
                else:
                    for _ in range(available_left):
                        genes.append(Gene(PieceType.CURVE_LEFT))
                    for _ in range(min(4, self.inventory.straight)):
                        genes.append(Gene(PieceType.STRAIGHT))
            else:
                # Not enough curves for full circle - use all available
                for _ in range(available_right):
                    genes.append(Gene(PieceType.CURVE_RIGHT))
                for _ in range(available_left):
                    genes.append(Gene(PieceType.CURVE_LEFT))
        else:
            # No inventory - simple 16-curve circle
            genes = [Gene(PieceType.CURVE_RIGHT) for _ in range(16)]

        # Ensure minimum length
        if len(genes) < 8:
            genes.extend([Gene(PieceType.CURVE_RIGHT) for _ in range(8 - len(genes))])

        return Individual(genes=genes)

    def _create_varied_pattern(self) -> Individual:
        """Create varied pattern with straights and curves"""
        genes = []

        # Choose a pattern - all designed to close
        patterns = [
            # Oval: 8 curves each end + straights in between (needs 16 curves same dir)
            # This creates an oval shape
            [PieceType.CURVE_RIGHT] * 8 + [PieceType.STRAIGHT] * 2 +
            [PieceType.CURVE_RIGHT] * 8 + [PieceType.STRAIGHT] * 2,

            # Simple circle - 16 curves
            [PieceType.CURVE_RIGHT] * 16,
            [PieceType.CURVE_LEFT] * 16,

            # Rectangle with curved corners: 4 curves (90°) + straight + repeat
            [PieceType.CURVE_RIGHT] * 4 + [PieceType.STRAIGHT] +
            [PieceType.CURVE_RIGHT] * 4 + [PieceType.STRAIGHT] +
            [PieceType.CURVE_RIGHT] * 4 + [PieceType.STRAIGHT] +
            [PieceType.CURVE_RIGHT] * 4 + [PieceType.STRAIGHT],

            # Longer oval with more straights
            [PieceType.CURVE_RIGHT] * 8 + [PieceType.STRAIGHT] * 4 +
            [PieceType.CURVE_RIGHT] * 8 + [PieceType.STRAIGHT] * 4,
        ]

        pattern = random.choice(patterns)

        if self.inventory:
            available = self.inventory.copy()
            temp_genes = []
            for pt in pattern:
                if available.can_use(pt):
                    temp_genes.append(Gene(pt))
                    available.use_piece(pt)
            genes = temp_genes
        else:
            genes = [Gene(pt) for pt in pattern]

        # Ensure minimum length for closure attempt
        if len(genes) < 12:
            # Try to add more curves to help closure
            remaining_right = (self.inventory.curve_right -
                               sum(1 for g in genes if g.piece_type == PieceType.CURVE_RIGHT)) if self.inventory else 8
            remaining_left = (self.inventory.curve_left -
                              sum(1 for g in genes if g.piece_type == PieceType.CURVE_LEFT)) if self.inventory else 8

            while len(genes) < 16 and (remaining_right > 0 or remaining_left > 0):
                if remaining_right > 0:
                    genes.append(Gene(PieceType.CURVE_RIGHT))
                    remaining_right -= 1
                elif remaining_left > 0:
                    genes.append(Gene(PieceType.CURVE_LEFT))
                    remaining_left -= 1

        return Individual(genes=genes)

    def _create_vertical_pattern(self) -> Individual:
        """
        Create VERTICALLY-oriented pattern to maximize Y-axis utilization.

        Key insight: To get vertical straights, we need the track direction
        to be ~90° when placing straights. This means:
        - Start with 4 right curves (90° turn) to point upward
        - Place straights while going up
        - Continue with curves to complete the loop
        """
        genes = []

        if not self.inventory:
            # No inventory - create simple vertical oval
            # 4R to turn up, straights going up, 8R to turn around, straights down, 4R
            pattern = (
                    [PieceType.CURVE_RIGHT] * 4 +  # Turn from 0° to -90° (pointing down-ish)
                    [PieceType.STRAIGHT] * 2 +  # Vertical segment
                    [PieceType.CURVE_RIGHT] * 8 +  # Turn 180°
                    [PieceType.STRAIGHT] * 2 +  # Vertical segment back
                    [PieceType.CURVE_RIGHT] * 4  # Complete the turn
            )
            return Individual(genes=[Gene(pt) for pt in pattern])

        # Get available pieces
        left_curves = self.inventory.curve_left
        right_curves = self.inventory.curve_right
        straights = self.inventory.straight

        # Choose curve direction based on inventory
        if right_curves >= 16:
            curve_type = PieceType.CURVE_RIGHT
            available_curves = right_curves
        elif left_curves >= 16:
            curve_type = PieceType.CURVE_LEFT
            available_curves = left_curves
        else:
            # Not enough curves for full loop
            curve_type = PieceType.CURVE_RIGHT if right_curves > left_curves else PieceType.CURVE_LEFT
            available_curves = max(right_curves, left_curves)

        # Calculate how many straights can fit vertically
        # Boundary height available for straights (after curves take some space)
        boundary_height = self.boundary[3] - self.boundary[2]
        margin = 30.0  # Leave room for curves at top/bottom
        available_height = boundary_height - 2 * margin - 80  # 80 for curves at ends
        max_straights_vertical = max(0, int(available_height / STRAIGHT_LENGTH))

        # Use half the straights on each vertical section
        straights_per_side = min(straights // 2, max_straights_vertical)

        # Build vertical oval pattern
        # First 4 curves to turn toward vertical direction
        for _ in range(min(4, available_curves)):
            genes.append(Gene(curve_type))

        # First vertical straight section
        for _ in range(straights_per_side):
            genes.append(Gene(PieceType.STRAIGHT))

        # Middle curves (8 total for 180° turn at top)
        remaining_curves = available_curves - 4
        middle_curves = min(8, remaining_curves)
        for _ in range(middle_curves):
            genes.append(Gene(curve_type))

        # Second vertical straight section
        remaining_straights = straights - straights_per_side
        for _ in range(min(straights_per_side, remaining_straights)):
            genes.append(Gene(PieceType.STRAIGHT))

        # Final 4 curves to complete the loop
        remaining_curves = available_curves - 4 - middle_curves
        for _ in range(min(4, remaining_curves)):
            genes.append(Gene(curve_type))

        return Individual(genes=genes)

    def _create_tall_rectangle(self) -> Individual:
        """
        Create a tall rectangular pattern that maximizes vertical extent.
        Pattern: 4 curves (corner) + vertical straights + 4 curves + horizontal + repeat
        """
        genes = []

        if not self.inventory:
            # Default tall rectangle
            pattern = (
                    [PieceType.CURVE_RIGHT] * 4 +  # First corner (90° right turn)
                    [PieceType.STRAIGHT] * 4 +  # Vertical section going up
                    [PieceType.CURVE_RIGHT] * 4 +  # Second corner
                    [PieceType.STRAIGHT] * 1 +  # Short horizontal
                    [PieceType.CURVE_RIGHT] * 4 +  # Third corner
                    [PieceType.STRAIGHT] * 4 +  # Vertical section going down
                    [PieceType.CURVE_RIGHT] * 4 +  # Fourth corner
                    [PieceType.STRAIGHT] * 1  # Short horizontal back
            )
            return Individual(genes=[Gene(pt) for pt in pattern])

        # Calculate dimensions
        left_curves = self.inventory.curve_left
        right_curves = self.inventory.curve_right
        straights = self.inventory.straight

        # Need 16 curves for 4 corners
        if right_curves >= 16:
            curve_type = PieceType.CURVE_RIGHT
        elif left_curves >= 16:
            curve_type = PieceType.CURVE_LEFT
        else:
            # Fall back to oval
            return self._create_oval_pattern()

        # Distribute straights: more on vertical sides, fewer on horizontal
        # Aim for 3:1 ratio vertical:horizontal
        total_straights = straights
        vertical_straights = min((total_straights * 3) // 4, straights)
        horizontal_straights = total_straights - vertical_straights

        # Split between two sides
        vert_per_side = vertical_straights // 2
        horiz_per_side = max(1, horizontal_straights // 2)

        # Build tall rectangle
        # Corner 1
        for _ in range(4):
            genes.append(Gene(curve_type))
        # Vertical side 1 (going up or down depending on curve direction)
        for _ in range(vert_per_side):
            genes.append(Gene(PieceType.STRAIGHT))
        # Corner 2
        for _ in range(4):
            genes.append(Gene(curve_type))
        # Horizontal side 1 (short)
        for _ in range(horiz_per_side):
            genes.append(Gene(PieceType.STRAIGHT))
        # Corner 3
        for _ in range(4):
            genes.append(Gene(curve_type))
        # Vertical side 2
        for _ in range(vert_per_side):
            genes.append(Gene(PieceType.STRAIGHT))
        # Corner 4
        for _ in range(4):
            genes.append(Gene(curve_type))
        # Horizontal side 2 (short)
        remaining = horizontal_straights - horiz_per_side
        for _ in range(remaining):
            genes.append(Gene(PieceType.STRAIGHT))

        return Individual(genes=genes)

    def _create_random(self) -> Individual:
        """Create random individual for diversity"""
        length = random.randint(12, 24)
        genes = []

        if self.inventory:
            available = self.inventory.copy()
            for _ in range(length):
                basic_types = [PieceType.STRAIGHT, PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT]
                if self.phase == 2:
                    basic_types.extend([PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT])

                random.shuffle(basic_types)
                for pt in basic_types:
                    if available.can_use(pt):
                        genes.append(Gene(pt))
                        available.use_piece(pt)
                        break
        else:
            for _ in range(length):
                pt = random.choice([PieceType.STRAIGHT, PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT])
                genes.append(Gene(pt))

        return Individual(genes=genes)

    def _evaluate(self, individual: Individual):
        """Evaluate fitness of individual"""
        individual.fitness_result = evaluate_individual(
            individual, self.boundary, self.inventory, self.phase
        )

    def _add_to_population(self, individual: Individual):
        """Add individual to appropriate subpopulation"""
        if individual.fitness_result and individual.fitness_result.is_feasible:
            self.feasible.append(individual)
        else:
            self.infeasible.append(individual)

    def _update_best(self):
        """Update best-ever individual"""
        all_individuals = self.feasible + self.infeasible
        for ind in all_individuals:
            if ind.fitness_result:
                if self.best_ever is None:
                    self.best_ever = ind.copy()
                elif ind.fitness_result.is_feasible:
                    if not self.best_ever.fitness_result.is_feasible:
                        self.best_ever = ind.copy()
                    elif ind.fitness_result.fitness > self.best_ever.fitness_result.fitness:
                        self.best_ever = ind.copy()
                elif not self.best_ever.fitness_result.is_feasible:
                    if ind.fitness_result.fitness > self.best_ever.fitness_result.fitness:
                        self.best_ever = ind.copy()

    def evolve(self, num_offspring: int = None):
        """
        Evolve population using mutation-only approach.
        Implements IDEA by maintaining infeasible subpopulation.
        """
        if num_offspring is None:
            num_offspring = self.size

        new_feasible = []
        new_infeasible = []

        # Elitism: keep best feasible individuals
        if self.feasible:
            elite_count = max(2, len(self.feasible) // 10)
            sorted_feasible = sorted(
                self.feasible,
                key=lambda x: x.fitness_result.fitness if x.fitness_result else -1e10,
                reverse=True
            )
            for ind in sorted_feasible[:elite_count]:
                ind_copy = ind.copy()
                ind_copy.age += 1
                new_feasible.append(ind_copy)

        # Selection pool: mostly feasible, some infeasible for IDEA
        all_individuals = self.feasible + self.infeasible
        if not all_individuals:
            self.initialize()
            return

        # Generate offspring through mutation
        offspring_count = 0
        max_attempts = num_offspring * 3
        attempts = 0

        while offspring_count < num_offspring and attempts < max_attempts:
            attempts += 1

            # Tournament selection
            tournament_size = 3
            tournament = random.sample(all_individuals, min(tournament_size, len(all_individuals)))
            parent = max(tournament, key=lambda x: (
                x.fitness_result.is_feasible if x.fitness_result else False,
                x.fitness_result.fitness if x.fitness_result else -1e10
            ))

            # Apply mutation
            child = self.mutator.mutate(parent)

            if random.random() < 0.15:  # 15% repair rate
                child = self.repairer.repair_closure(child)
                child = self.repairer.repair_boundary(child)

            # Evaluate
            self._evaluate(child)

            # Add to appropriate subpopulation
            if child.fitness_result.is_feasible:
                new_feasible.append(child)
            else:
                new_infeasible.append(child)

            offspring_count += 1

        # Maintain population size and infeasible ratio
        target_infeasible = int(self.size * self.infeasible_ratio)
        target_feasible = self.size - target_infeasible

        # Trim populations if needed
        if len(new_feasible) > target_feasible:
            new_feasible.sort(
                key=lambda x: x.fitness_result.fitness if x.fitness_result else -1e10,
                reverse=True
            )
            new_feasible = new_feasible[:target_feasible]

        if len(new_infeasible) > target_infeasible:
            new_infeasible.sort(
                key=lambda x: x.fitness_result.fitness if x.fitness_result else -1e10,
                reverse=True
            )
            new_infeasible = new_infeasible[:target_infeasible]

        self.feasible = new_feasible
        self.infeasible = new_infeasible
        self.generation += 1
        self._update_best()

    def get_best(self) -> Optional[Individual]:
        """Get best individual (prefer feasible)"""
        if self.feasible:
            return max(self.feasible,
                       key=lambda x: x.fitness_result.fitness if x.fitness_result else -1e10)
        elif self.infeasible:
            return max(self.infeasible,
                       key=lambda x: x.fitness_result.fitness if x.fitness_result else -1e10)
        return None

    def get_stats(self) -> Dict:
        """Get population statistics"""
        return {
            'generation': self.generation,
            'feasible_count': len(self.feasible),
            'infeasible_count': len(self.infeasible),
            'total': len(self.feasible) + len(self.infeasible),
            'best_fitness': (self.best_ever.fitness_result.fitness
                             if self.best_ever and self.best_ever.fitness_result else None),
            'best_pieces': (self.best_ever.fitness_result.num_pieces
                            if self.best_ever and self.best_ever.fitness_result else 0),
            'best_feasible': (self.best_ever.fitness_result.is_feasible
                              if self.best_ever and self.best_ever.fitness_result else False)
        }


# ============================================================================
# TWO-PHASE GENETIC ALGORITHM
# ============================================================================

class TwoPhaseTrackOptimizer:
    """
    Two-phase optimization:
    Phase 1: Maximize usage of straights and curves to create basic closed loop
    Phase 2: Add switches to further maximize piece usage
    """

    def __init__(self,
                 boundary: Tuple[float, float, float, float],
                 inventory: PieceInventory,
                 population_size: int = 100,
                 phase1_generations: int = 200,
                 phase2_generations: int = 100):
        self.boundary = boundary
        self.inventory = inventory
        self.population_size = population_size
        self.phase1_generations = phase1_generations
        self.phase2_generations = phase2_generations

        self.phase1_result: Optional[Individual] = None
        self.phase2_result: Optional[Individual] = None
        self.history: List[Dict] = []

    def check_inventory_feasibility(self) -> Tuple[bool, str]:
        """
        Check if the inventory can mathematically form a closed loop.

        For closure: net angle must be ±360° (or multiple)
        - Left curves: +22.5° each
        - Right curves: -22.5° each
        - Straights: 0°

        Returns (is_feasible, explanation)
        """
        left = self.inventory.curve_left
        right = self.inventory.curve_right

        # Calculate maximum possible net angle
        # Best case: use all of one type, none of other
        max_positive = left * CURVE_ANGLE  # All lefts
        max_negative = right * CURVE_ANGLE  # All rights

        # For closure, we need |net_angle| = 360
        # Using L left curves and R right curves: net = L*22.5 - R*22.5
        # We need: L*22.5 - R*22.5 = ±360
        # So: L - R = ±16

        # Check if we can achieve 360° in either direction
        can_close_right = right >= 16  # 16 right curves = -360°
        can_close_left = left >= 16  # 16 left curves = +360°

        # Check mixed case: if |left - right| >= 16 with sufficient curves
        diff = abs(left - right)
        can_close_mixed = diff >= 16

        if can_close_right or can_close_left:
            return True, f"Feasible: {'≥16 right' if can_close_right else '≥16 left'} curves available"
        elif can_close_mixed:
            if left > right:
                return True, f"Feasible: Can use {16 + right} left + {right} right = net +360°"
            else:
                return True, f"Feasible: Can use {left} left + {16 + left} right = net -360°"
        else:
            # Calculate what we'd need
            if left >= right:
                needed_left = 16 + right
                shortage = needed_left - left
                return False, (f"INFEASIBLE: {left}L + {right}R curves cannot close.\n"
                               f"  Max net angle: {(left - right) * 22.5:.1f}° (need 360°)\n"
                               f"  Need {shortage} more left curves, OR {16 - right} more right curves")
            else:
                needed_right = 16 + left
                shortage = needed_right - right
                return False, (f"INFEASIBLE: {left}L + {right}R curves cannot close.\n"
                               f"  Max net angle: {(right - left) * 22.5:.1f}° (need 360°)\n"
                               f"  Need {shortage} more right curves, OR {16 - left} more left curves")

    def estimate_track_size(self) -> Tuple[float, float]:
        """Estimate minimum track size needed for inventory"""
        # Base circle with 16 curves: ~80x80 studs
        base_width = 80.0
        base_height = 80.0

        # Each pair of straights on opposite sides adds 16 studs to one dimension
        num_straights = self.inventory.straight
        # Assume straights split between two sides of oval
        straights_per_side = num_straights // 2
        extra_length = straights_per_side * STRAIGHT_LENGTH

        return base_width + extra_length, base_height

    def calculate_max_pieces_for_boundary(self, safety_margin: float = 10.0) -> Dict[str, int]:
        """
        Calculate maximum number of each piece type that can fit in boundary.
        Returns dict with 'curves', 'straights', 'total' keys.

        Args:
            safety_margin: Extra margin from boundary edge (default 10 studs)
        """
        boundary_width = self.boundary[1] - self.boundary[0]
        boundary_height = self.boundary[3] - self.boundary[2]

        # Total margin (both sides + safety)
        margin = 2 * safety_margin
        available_width = boundary_width - margin
        available_height = boundary_height - margin

        # Curves: need 16 for closure, they take ~80 studs
        min_for_curves = 80.0
        can_fit_curves = min(available_width, available_height) >= min_for_curves
        max_curves = 16 if can_fit_curves else 0

        # Straights: each pair adds 16 studs to width
        # Available width for straights = available_width - base_circle_width
        base_width = 80.0
        extra_for_straights = available_width - base_width
        max_straights = int(extra_for_straights / STRAIGHT_LENGTH) * 2  # pairs
        max_straights = max(0, max_straights)

        return {
            'curves': max_curves,
            'straights': max_straights,
            'total': max_curves + max_straights
        }

    def run_phase1(self, verbose: bool = True) -> Individual:
        """
        Phase 1: Optimize basic track (straights + curves only)
        Goal: Create closed loop using maximum pieces
        """
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 1: Optimizing Basic Track (Straights + Curves)")
            print("=" * 60)

        # Check mathematical feasibility first
        is_feasible, explanation = self.check_inventory_feasibility()
        if verbose:
            print(f"\nInventory Analysis: {explanation}")

        if not is_feasible:
            if verbose:
                print("\n⚠️  WARNING: This inventory CANNOT form a closed loop!")
                print("   The algorithm will attempt to find the best partial solution.")

        # Estimate required track size
        est_width, est_height = self.estimate_track_size()
        boundary_width = self.boundary[1] - self.boundary[0]
        boundary_height = self.boundary[3] - self.boundary[2]

        if verbose:
            print(f"\nTrack Size Estimate: {est_width:.0f} × {est_height:.0f} studs")
            print(f"Boundary Size:       {boundary_width:.0f} × {boundary_height:.0f} studs")

        if est_width > boundary_width or est_height > boundary_height:
            # Calculate how many pieces can actually fit
            max_pieces = self.calculate_max_pieces_for_boundary()
            if verbose:
                print(f"\n⚠️  WARNING: Full inventory needs {est_width:.0f}×{est_height:.0f} studs")
                print(f"   but boundary is only {boundary_width:.0f}×{boundary_height:.0f} studs!")
                print(f"   Maximum that can fit: ~{max_pieces['total']} pieces")
                print(f"   ({max_pieces['curves']} curves + {max_pieces['straights']} straights)")

        # Create phase 1 inventory (no switches)
        phase1_inventory = PieceInventory(
            straight=self.inventory.straight,
            curve_left=self.inventory.curve_left,
            curve_right=self.inventory.curve_right,
            switch_left=0,
            switch_right=0
        )

        population = Population(
            size=self.population_size,
            boundary=self.boundary,
            inventory=phase1_inventory,
            phase=1,
            infeasible_ratio=0.15
        )
        population.initialize()

        start_time = time.time()
        best_fitness = -float('inf')
        stagnation = 0
        early_stop = 50

        for gen in range(self.phase1_generations):
            population.evolve()

            stats = population.get_stats()
            current_best = stats['best_fitness'] or -float('inf')

            if current_best > best_fitness:
                best_fitness = current_best
                stagnation = 0
            else:
                stagnation += 1

            self.history.append({
                'phase': 1,
                'generation': gen,
                **stats
            })

            if verbose and (gen % 20 == 0 or gen == self.phase1_generations - 1):
                elapsed = time.time() - start_time
                status = "FEASIBLE" if stats['best_feasible'] else "INFEASIBLE"
                print(f"Gen {gen:4d} | {status:10s} | Pieces: {stats['best_pieces']:3d} | "
                      f"Feasible: {stats['feasible_count']:3d}/{stats['total']:3d} | "
                      f"Time: {elapsed:.1f}s")

            if stagnation >= early_stop:
                if verbose:
                    print(f"Early stopping at generation {gen} (no improvement for {early_stop} gens)")
                break

        self.phase1_result = population.get_best()

        if verbose and self.phase1_result:
            fr = self.phase1_result.fitness_result
            print(f"\nPhase 1 Complete:")
            print(f"  Pieces: {fr.num_pieces}")
            print(f"  Feasible: {fr.is_feasible}")
            print(f"  Position Error: {fr.pos_error:.2f} studs")
            print(f"  Angle Error: {fr.angle_error:.2f}°")

        return self.phase1_result

    def run_phase2(self, verbose: bool = True) -> Optional[Individual]:
        """
        Phase 2: Add switches to the track
        Goal: Maximize total piece usage while maintaining closed loop

        IMPORTANT: Switches require specific geometry to form closed branches.
        - 2 same-direction switches need 1L + 1R curve for branch track
        - 1 left + 1 right switch can form a crossover
        """
        if not self.phase1_result:
            if verbose:
                print("Phase 1 must be run first!")
            return None

        if self.inventory.total_switches() == 0:
            if verbose:
                print("\nNo switches in inventory, skipping Phase 2")
            return self.phase1_result

        # Check switch feasibility
        switch_analysis = analyze_switch_feasibility(self.inventory)

        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 2: Adding Switches")
            print("=" * 60)
            print(f"\nSwitch Configuration Analysis:")
            print(f"  Inventory: {self.inventory.switch_left} left, {self.inventory.switch_right} right switches")
            print(f"  Available curves: {self.inventory.curve_left} left, {self.inventory.curve_right} right")
            print(
                f"\n  Passing siding (2 left switches): {'✓ Possible' if switch_analysis['passing_siding_left'] else '✗ Not possible'}")
            print(
                f"  Passing siding (2 right switches): {'✓ Possible' if switch_analysis['passing_siding_right'] else '✗ Not possible'}")
            print(
                f"  Crossover (1 left + 1 right): {'✓ Possible' if switch_analysis['crossover'] else '✗ Not possible'}")
            print(f"\n  Recommendation: {switch_analysis['recommendation']}")
            print(f"  Reason: {switch_analysis['reason']}")

        # If no valid switch configuration is possible, skip phase 2
        if switch_analysis['recommendation'] == 'none_feasible':
            if verbose:
                print("\n⚠️  Cannot add switches - no valid closed configuration possible!")
                print("   Switches would create open branches that don't reconnect.")
                print("   Keeping Phase 1 result (without switches).")
            return self.phase1_result

        # Calculate remaining inventory after phase 1
        phase1_counts = self.phase1_result.count_pieces()
        phase2_inventory = PieceInventory(
            straight=self.inventory.straight - phase1_counts.get(PieceType.STRAIGHT, 0),
            curve_left=self.inventory.curve_left - phase1_counts.get(PieceType.CURVE_LEFT, 0),
            curve_right=self.inventory.curve_right - phase1_counts.get(PieceType.CURVE_RIGHT, 0),
            switch_left=self.inventory.switch_left,
            switch_right=self.inventory.switch_right
        )

        # Initialize population with phase 1 result as seed
        population = Population(
            size=self.population_size,
            boundary=self.boundary,
            inventory=self.inventory,  # Full inventory for phase 2
            phase=2,
            infeasible_ratio=0.2
        )

        # Create switch-included versions of phase 1 result
        mutator = MutationOperator(
            inventory=self.inventory,
            boundary=self.boundary,
            phase=2
        )

        # Seed half with original phase 1 result
        original_count = self.population_size // 4
        population.feasible = []
        for _ in range(original_count):
            ind = self.phase1_result.copy()
            population._evaluate(ind)
            if ind.fitness_result.is_feasible:
                population.feasible.append(ind)
            else:
                population.infeasible.append(ind)

        # Seed the rest with switch-inserted versions
        switch_inserted_count = self.population_size - original_count
        for _ in range(switch_inserted_count):
            # Start with phase 1 result
            ind = self.phase1_result.copy()

            # Try to insert passing siding
            ind = mutator.insert_passing_siding(ind)

            population._evaluate(ind)
            population._add_to_population(ind)

        population._update_best()

        if verbose:
            feasible_with_switches = sum(
                1 for ind in population.feasible
                if any(g.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT)
                       for g in ind.genes)
            )
            print(f"\nInitial population: {len(population.feasible)} feasible, "
                  f"{feasible_with_switches} with switches")

        # Update mutator for phase 2
        population.mutator.phase = 2

        start_time = time.time()
        best_fitness = -float('inf')
        stagnation = 0
        early_stop = 30

        for gen in range(self.phase2_generations):
            population.evolve()

            stats = population.get_stats()
            current_best = stats['best_fitness'] or -float('inf')

            if current_best > best_fitness:
                best_fitness = current_best
                stagnation = 0
            else:
                stagnation += 1

            self.history.append({
                'phase': 2,
                'generation': gen,
                **stats
            })

            if verbose and (gen % 10 == 0 or gen == self.phase2_generations - 1):
                elapsed = time.time() - start_time
                status = "FEASIBLE" if stats['best_feasible'] else "INFEASIBLE"
                print(f"Gen {gen:4d} | {status:10s} | Pieces: {stats['best_pieces']:3d} | "
                      f"Feasible: {stats['feasible_count']:3d}/{stats['total']:3d} | "
                      f"Time: {elapsed:.1f}s")

            if stagnation >= early_stop:
                if verbose:
                    print(f"Early stopping at generation {gen}")
                break

        self.phase2_result = population.get_best()

        # Return better of phase 1 or phase 2 based on FITNESS (not piece count)
        # This ensures switch tracks with higher fitness are selected
        if self.phase2_result and self.phase2_result.fitness_result:
            if self.phase2_result.fitness_result.is_feasible:
                phase2_fitness = self.phase2_result.fitness_result.fitness
                phase1_fitness = self.phase1_result.fitness_result.fitness

                if phase2_fitness > phase1_fitness:
                    if verbose:
                        print(f"\n✓ Phase 2 improved: fitness {phase1_fitness:.0f} -> {phase2_fitness:.0f}")
                    return self.phase2_result

        if verbose:
            print(f"\n  Phase 1 result retained (fitness: {self.phase1_result.fitness_result.fitness:.0f})")
        return self.phase1_result

    def run(self, verbose: bool = True) -> Individual:
        """Run complete two-phase optimization"""
        phase1_best = self.run_phase1(verbose)

        if self.inventory.total_switches() > 0:
            final_best = self.run_phase2(verbose)
        else:
            final_best = phase1_best

        return final_best


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_track(individual: Individual,
                    boundary: Tuple[float, float, float, float],
                    title: str = "Track Layout",
                    save_path: str = None):
    """Visualize track layout using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Arc, FancyBboxPatch
        import numpy as np
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw boundary
    x_min, x_max, y_min, y_max = boundary
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='gray', facecolor='lightgray', alpha=0.3
    )
    ax.add_patch(rect)

    # Build track centered
    pieces = build_track_centered(individual.genes, boundary)

    # Color map for piece types
    colors = {
        PieceType.STRAIGHT: 'blue',
        PieceType.CURVE_LEFT: 'green',
        PieceType.CURVE_RIGHT: 'red',
        PieceType.SWITCH_LEFT: 'purple',
        PieceType.SWITCH_RIGHT: 'orange'
    }

    # Lighter colors for branch track pieces
    branch_colors = {
        PieceType.STRAIGHT: 'deepskyblue',
        PieceType.CURVE_LEFT: 'lightgreen',
        PieceType.CURVE_RIGHT: 'salmon',
        PieceType.SWITCH_LEFT: 'plum',
        PieceType.SWITCH_RIGHT: 'moccasin'
    }

    # Find branch pieces and their endpoints
    branch_pieces = [p for p in pieces if p.is_branch_piece]
    branch_end_pos = branch_pieces[-1].end_pos if branch_pieces else None

    # Find switches
    normal_switches = [p for p in pieces if
                       p.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT) and not p.is_reversed]
    reversed_switches = [p for p in pieces if
                         p.piece_type in (PieceType.SWITCH_LEFT, PieceType.SWITCH_RIGHT) and p.is_reversed]

    # Draw each piece
    for i, piece in enumerate(pieces):
        # Use lighter colors for branch pieces to distinguish from main track
        if piece.is_branch_piece:
            color = branch_colors.get(piece.piece_type, 'cyan')
        else:
            color = colors.get(piece.piece_type, 'black')

        if piece.piece_type == PieceType.STRAIGHT:
            # Straight line
            ax.plot([piece.start_pos.x, piece.end_pos.x],
                    [piece.start_pos.y, piece.end_pos.y],
                    color=color, linewidth=4, solid_capstyle='round')
        elif piece.piece_type in [PieceType.CURVE_LEFT, PieceType.CURVE_RIGHT]:
            # Draw curve as arc - sample points along the curve
            angle_rad = math.radians(piece.start_angle)
            if piece.piece_type == PieceType.CURVE_RIGHT:
                perp_angle = angle_rad - math.pi / 2
            else:
                perp_angle = angle_rad + math.pi / 2

            center = Point(
                piece.start_pos.x + CURVE_RADIUS * math.cos(perp_angle),
                piece.start_pos.y + CURVE_RADIUS * math.sin(perp_angle)
            )

            # Sample points along arc
            num_points = 10
            arc_x = []
            arc_y = []
            for j in range(num_points + 1):
                t = j / num_points
                mid_angle = piece.start_angle + t * (piece.end_angle - piece.start_angle)
                mid_angle_rad = math.radians(mid_angle)

                if piece.piece_type == PieceType.CURVE_RIGHT:
                    px = center.x + CURVE_RADIUS * math.cos(mid_angle_rad + math.pi / 2)
                    py = center.y + CURVE_RADIUS * math.sin(mid_angle_rad + math.pi / 2)
                else:
                    px = center.x + CURVE_RADIUS * math.cos(mid_angle_rad - math.pi / 2)
                    py = center.y + CURVE_RADIUS * math.sin(mid_angle_rad - math.pi / 2)

                arc_x.append(px)
                arc_y.append(py)

            ax.plot(arc_x, arc_y, color=color, linewidth=4, solid_capstyle='round')
        else:
            # Switch - draw main line
            ax.plot([piece.start_pos.x, piece.end_pos.x],
                    [piece.start_pos.y, piece.end_pos.y],
                    color=color, linewidth=4, solid_capstyle='round')

            # Draw branch divergence/convergence line
            if piece.is_reversed:
                # REVERSED switch: branch converges TO end_pos
                # For opposite-type sidings, draw from actual branch end to switch end
                if branch_end_pos:
                    # Connect actual branch track endpoint to switch end (frog)
                    ax.plot([branch_end_pos.x, piece.end_pos.x],
                            [branch_end_pos.y, piece.end_pos.y],
                            color=color, linewidth=2, linestyle='--', alpha=0.7)
                else:
                    # No branch track - draw theoretical entry based on switch geometry
                    angle_rad = math.radians(piece.start_angle)
                    if piece.piece_type == PieceType.SWITCH_LEFT:
                        branch_offset_angle = angle_rad + math.radians(SWITCH_ANGLE) + math.pi
                    else:  # SWITCH_RIGHT
                        branch_offset_angle = angle_rad - math.radians(SWITCH_ANGLE) + math.pi
                    branch_entry_x = piece.end_pos.x + SWITCH_LENGTH * math.cos(branch_offset_angle)
                    branch_entry_y = piece.end_pos.y + SWITCH_LENGTH * math.sin(branch_offset_angle)
                    ax.plot([branch_entry_x, piece.end_pos.x],
                            [branch_entry_y, piece.end_pos.y],
                            color=color, linewidth=2, linestyle='--', alpha=0.7)
                # Mark reversed switch with hollow circle at frog (end)
                ax.plot(piece.end_pos.x, piece.end_pos.y, 'o',
                        color=color, markersize=8, markerfacecolor='white', markeredgewidth=2)
            elif piece.branch_end_pos:
                # NORMAL switch: branch diverges FROM start_pos to branch_end_pos
                ax.plot([piece.start_pos.x, piece.branch_end_pos.x],
                        [piece.start_pos.y, piece.branch_end_pos.y],
                        color=color, linewidth=2, linestyle='--', alpha=0.7)
                # Mark normal switch with filled circle at branch end
                ax.plot(piece.branch_end_pos.x, piece.branch_end_pos.y, 'o',
                        color=color, markersize=6)

        # Draw start point for non-branch pieces
        if not piece.is_branch_piece:
            ax.plot(piece.start_pos.x, piece.start_pos.y, 'ko', markersize=3)

        # Mark first piece
        if i == 0:
            ax.plot(piece.start_pos.x, piece.start_pos.y, 'g*', markersize=15, label='Start')

    # Draw closure line (dashed) showing the gap - use MAIN TRACK only, not branch
    main_pieces = [p for p in pieces if not p.is_branch_piece]
    if main_pieces:
        gap_dist = main_pieces[-1].end_pos.distance_to(main_pieces[0].start_pos)
        if gap_dist > 0.1:  # Only draw if there's a visible gap
            ax.plot([main_pieces[-1].end_pos.x, main_pieces[0].start_pos.x],
                    [main_pieces[-1].end_pos.y, main_pieces[0].start_pos.y],
                    'r--', linewidth=1, alpha=0.5, label=f'Gap: {gap_dist:.1f}')

    # Legend for piece types
    for pt, color in colors.items():
        ax.plot([], [], color=color, linewidth=4, label=pt.name)

    # Set axis limits with margin
    margin = 20
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (studs)')
    ax.set_ylabel('Y (studs)')

    # Add boundary dimensions
    ax.text(x_min + 5, y_max - 5, f'Boundary: {x_max - x_min:.0f}×{y_max - y_min:.0f}',
            fontsize=9, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def get_user_input() -> Tuple[Tuple[float, float, float, float], PieceInventory, int, int]:
    """Get optimization parameters from user input"""
    print("\n" + "=" * 60)
    print("LEGO Train Track Layout Optimizer")
    print("Mutation-Only GA with Two-Phase Optimization")
    print("=" * 60)

    # Area size
    print("\n--- Area Configuration ---")
    try:
        width = float(input("Enter area width in studs (default 160): ") or "160")
        height = float(input("Enter area height in studs (default 160): ") or "160")
        boundary = (-width / 2, width / 2, -height / 2, height / 2)
    except ValueError:
        print("Invalid input, using defaults (160x160)")
        boundary = (-80, 80, -80, 80)

    # Piece inventory
    print("\n--- Piece Inventory ---")
    try:
        straight = int(input("Number of straight pieces (default 8): ") or "8")
        curve_left = int(input("Number of left curve pieces (default 8): ") or "8")
        curve_right = int(input("Number of right curve pieces (default 8): ") or "8")
        switch_left = int(input("Number of left switches (default 0): ") or "0")
        switch_right = int(input("Number of right switches (default 0): ") or "0")

        inventory = PieceInventory(
            straight=straight,
            curve_left=curve_left,
            curve_right=curve_right,
            switch_left=switch_left,
            switch_right=switch_right
        )
    except ValueError:
        print("Invalid input, using defaults")
        inventory = PieceInventory(straight=8, curve_left=8, curve_right=8)

    # GA parameters
    print("\n--- Algorithm Parameters ---")
    try:
        pop_size = int(input("Population size (default 100): ") or "100")
        pop_size = max(20, min(500, pop_size))
    except ValueError:
        pop_size = 100

    try:
        max_gens = int(input("Maximum generations (default 300): ") or "300")
        max_gens = max(50, min(2000, max_gens))
    except ValueError:
        max_gens = 300

    return boundary, inventory, pop_size, max_gens


def format_track_result(individual: Individual, inventory: PieceInventory) -> str:
    """Format track result for display"""
    if not individual or not individual.fitness_result:
        return "No valid result"

    fr = individual.fitness_result
    pieces = build_track_from_genes(individual.genes)

    lines = [
        "\n" + "=" * 60,
        "OPTIMIZATION RESULT",
        "=" * 60,
        f"\nTrack Statistics:",
        f"  Total Pieces: {fr.num_pieces}",
        f"  Feasible: {'Yes' if fr.is_feasible else 'No'}",
        f"  Position Error: {fr.pos_error:.2f} studs",
        f"  Angle Error: {fr.angle_error:.2f}°",
        f"  Boundary Violations: {fr.boundary_violations}",
        f"  Collisions: {fr.collision_count}",
        "\nPiece Usage:",
    ]

    for pt in PieceType:
        used = fr.pieces_by_type.get(pt, 0)
        available = inventory.get_count(pt)
        lines.append(f"  {pt.name}: {used}/{available}")

    # Track sequence
    lines.append("\nTrack Sequence:")
    seq = []
    for gene in individual.genes:
        if gene.piece_type == PieceType.STRAIGHT:
            seq.append('S')
        elif gene.piece_type == PieceType.CURVE_LEFT:
            seq.append('L')
        elif gene.piece_type == PieceType.CURVE_RIGHT:
            seq.append('R')
        elif gene.piece_type == PieceType.SWITCH_LEFT:
            seq.append('SL')
        elif gene.piece_type == PieceType.SWITCH_RIGHT:
            seq.append('SR')
    # Break into lines of 15 items
    for i in range(0, len(seq), 15):
        lines.append("  " + "-".join(seq[i:i + 15]))

    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Main entry point"""
    boundary, inventory, pop_size, max_gens = get_user_input()

    print(f"\nConfiguration:")
    print(f"  Area: {boundary[1] - boundary[0]:.0f} x {boundary[3] - boundary[2]:.0f} studs")
    print(f"  Total Pieces: {inventory.total()} "
          f"({inventory.straight}S, {inventory.curve_left}L, {inventory.curve_right}R, "
          f"{inventory.switch_left}SL, {inventory.switch_right}SR)")
    print(f"  Population: {pop_size}")
    print(f"  Max Generations: {max_gens}")

    # Run optimizer
    optimizer = TwoPhaseTrackOptimizer(
        boundary=boundary,
        inventory=inventory,
        population_size=pop_size,
        phase1_generations=max_gens,
        phase2_generations=max_gens // 2
    )

    best = optimizer.run(verbose=True)

    # Display results
    print(format_track_result(best, inventory))

    # Visualization option
    try:
        visualize = input("\nVisualize result? (y/n, default n): ").lower().startswith('y')
        if visualize:
            visualize_track(
                best, boundary,
                title=f"Optimized Track - {best.fitness_result.num_pieces} Pieces",
                save_path=None
            )
    except:
        pass

    return best


if __name__ == "__main__":
    main()