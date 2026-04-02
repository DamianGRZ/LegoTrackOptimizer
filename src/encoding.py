"""CGP-inspired integer chromosome encoding for track layout optimization.

Each node in the chromosome represents a track piece placement with:
- piece_type: integer piece index (-1=inactive, 0-9=piece type)
- port2_conn: which node does port 2 connect to (-1=not used)
- port3_conn: which node does port 3 connect to (-1=not used)

Ports 0 and 1 are implicit (sequential chain: node i port 1 → node i+1 port 0).
Extra ports (2, 3) use explicit connection genes for branches and secondary loops.

Chromosome layout (flat integer array, N_VAR = n_nodes * GENES_PER_NODE):
    [type_0, p2conn_0, p3conn_0,  type_1, p2conn_1, p3conn_1,  ...]

Dimensions scale dynamically from the user's inventory configuration.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Constants
# =============================================================================

GENES_PER_NODE = 3  # (piece_type, port2_conn, port3_conn)
INACTIVE = -1       # Sentinel for inactive node or unused connection


# Gene offsets within a node tuple
TYPE_OFFSET = 0
PORT2_OFFSET = 1
PORT3_OFFSET = 2


# =============================================================================
# Piece Index Constants (from track_pieces.yaml piece_index mapping)
# =============================================================================

class PieceIndex(IntEnum):
    """Track piece indices from catalog."""
    STRAIGHT_16 = 0
    STRAIGHT_24 = 1
    R40_LEFT = 2
    R40_RIGHT = 3
    CROSS_90 = 4
    SWITCH_LEFT_IN = 5
    SWITCH_LEFT_OUT = 6
    SWITCH_RIGHT_IN = 7
    SWITCH_RIGHT_OUT = 8
    DOUBLE_CROSSOVER = 9


# Convenience aliases
STRAIGHT_16 = PieceIndex.STRAIGHT_16
STRAIGHT_24 = PieceIndex.STRAIGHT_24
R40_LEFT = PieceIndex.R40_LEFT
R40_RIGHT = PieceIndex.R40_RIGHT
CROSS_90 = PieceIndex.CROSS_90
SWITCH_LEFT_IN = PieceIndex.SWITCH_LEFT_IN
SWITCH_LEFT_OUT = PieceIndex.SWITCH_LEFT_OUT
SWITCH_RIGHT_IN = PieceIndex.SWITCH_RIGHT_IN
SWITCH_RIGHT_OUT = PieceIndex.SWITCH_RIGHT_OUT
DOUBLE_CROSSOVER = PieceIndex.DOUBLE_CROSSOVER

# Piece categories
SIMPLE_PIECE_INDICES = {STRAIGHT_16, STRAIGHT_24, R40_LEFT, R40_RIGHT}
COMPLEX_PIECE_INDICES = {CROSS_90, SWITCH_LEFT_IN, SWITCH_LEFT_OUT,
                         SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT, DOUBLE_CROSSOVER}
SWITCH_INDICES = {SWITCH_LEFT_IN, SWITCH_LEFT_OUT, SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT}
IN_SWITCH_INDICES = {SWITCH_LEFT_IN, SWITCH_RIGHT_IN}
OUT_SWITCH_INDICES = {SWITCH_LEFT_OUT, SWITCH_RIGHT_OUT}
CROSSING_INDICES = {CROSS_90, DOUBLE_CROSSOVER}

# Switch pairs for balancing: (IN_index, OUT_index)
SWITCH_PAIRS = [
    (SWITCH_LEFT_IN, SWITCH_LEFT_OUT),
    (SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT),
]

# Pieces with 3 ports (switches) — need port2_conn gene
THREE_PORT_PIECES = SWITCH_INDICES
# Pieces with 4 ports (crossings) — need port2_conn and port3_conn genes
FOUR_PORT_PIECES = CROSSING_INDICES


# =============================================================================
# Chromosome Dimensions (Dynamic, Computed From Config)
# =============================================================================

@dataclass(frozen=True)
class ChromosomeDimensions:
    """Chromosome layout dimensions, computed from inventory at runtime.

    All segment boundaries are derived from n_nodes and GENES_PER_NODE.
    """
    n_nodes: int    # Total node count = total_inventory
    n_var: int      # Total genes = n_nodes * GENES_PER_NODE

    @property
    def genes_per_node(self) -> int:
        return GENES_PER_NODE

    def node_start(self, node_idx: int) -> int:
        """Gene index of first gene for node node_idx."""
        return node_idx * GENES_PER_NODE

    def type_index(self, node_idx: int) -> int:
        """Gene index of piece_type for node node_idx."""
        return node_idx * GENES_PER_NODE + TYPE_OFFSET

    def port2_index(self, node_idx: int) -> int:
        """Gene index of port2_conn for node node_idx."""
        return node_idx * GENES_PER_NODE + PORT2_OFFSET

    def port3_index(self, node_idx: int) -> int:
        """Gene index of port3_conn for node node_idx."""
        return node_idx * GENES_PER_NODE + PORT3_OFFSET


def compute_dimensions(total_inventory: int) -> ChromosomeDimensions:
    """Compute chromosome dimensions from total inventory count.

    Args:
        total_inventory: Sum of all piece counts in inventory.

    Returns:
        ChromosomeDimensions with n_nodes = total_inventory.
    """
    n_nodes = max(1, total_inventory)
    return ChromosomeDimensions(
        n_nodes=n_nodes,
        n_var=n_nodes * GENES_PER_NODE,
    )


# =============================================================================
# Bounds Generation
# =============================================================================

def generate_bounds(
    dims: ChromosomeDimensions,
    max_piece_index: int = 9,
) -> Tuple[NDArray, NDArray]:
    """Generate lower and upper bounds for the chromosome.

    Args:
        dims: Chromosome dimensions.
        max_piece_index: Maximum valid piece index from catalog.

    Returns:
        Tuple of (xl, xu) integer bound arrays of length n_var.
    """
    xl = np.full(dims.n_var, INACTIVE, dtype=np.int16)
    xu = np.full(dims.n_var, INACTIVE, dtype=np.int16)

    for i in range(dims.n_nodes):
        base = i * GENES_PER_NODE
        # piece_type: -1 (inactive) to max_piece_index
        xl[base + TYPE_OFFSET] = INACTIVE
        xu[base + TYPE_OFFSET] = max_piece_index
        # port2_conn: -1 (unused) to n_nodes - 1
        xl[base + PORT2_OFFSET] = INACTIVE
        xu[base + PORT2_OFFSET] = dims.n_nodes - 1
        # port3_conn: -1 (unused) to n_nodes - 1
        xl[base + PORT3_OFFSET] = INACTIVE
        xu[base + PORT3_OFFSET] = dims.n_nodes - 1

    return xl, xu


# =============================================================================
# Node Access Functions
# =============================================================================

def get_node(x: NDArray, node_idx: int) -> Tuple[int, int, int]:
    """Read a node tuple from the chromosome.

    Args:
        x: Chromosome array.
        node_idx: Node index.

    Returns:
        Tuple of (piece_type, port2_conn, port3_conn).
    """
    base = node_idx * GENES_PER_NODE
    return (int(x[base + TYPE_OFFSET]),
            int(x[base + PORT2_OFFSET]),
            int(x[base + PORT3_OFFSET]))


def set_node(x: NDArray, node_idx: int,
             piece_type: int, port2_conn: int = INACTIVE,
             port3_conn: int = INACTIVE) -> None:
    """Write a node tuple to the chromosome (in-place).

    Args:
        x: Chromosome array.
        node_idx: Node index.
        piece_type: Piece type index or INACTIVE.
        port2_conn: Port 2 connection target or INACTIVE.
        port3_conn: Port 3 connection target or INACTIVE.
    """
    base = node_idx * GENES_PER_NODE
    x[base + TYPE_OFFSET] = piece_type
    x[base + PORT2_OFFSET] = port2_conn
    x[base + PORT3_OFFSET] = port3_conn


def get_piece_type(x: NDArray, node_idx: int) -> int:
    """Read piece type for a node."""
    return int(x[node_idx * GENES_PER_NODE + TYPE_OFFSET])


def set_piece_type(x: NDArray, node_idx: int, piece_type: int) -> None:
    """Set piece type for a node (in-place)."""
    x[node_idx * GENES_PER_NODE + TYPE_OFFSET] = piece_type


def get_port2_conn(x: NDArray, node_idx: int) -> int:
    """Read port 2 connection target for a node."""
    return int(x[node_idx * GENES_PER_NODE + PORT2_OFFSET])


def set_port2_conn(x: NDArray, node_idx: int, target: int) -> None:
    """Set port 2 connection target for a node (in-place)."""
    x[node_idx * GENES_PER_NODE + PORT2_OFFSET] = target


def get_port3_conn(x: NDArray, node_idx: int) -> int:
    """Read port 3 connection target for a node."""
    return int(x[node_idx * GENES_PER_NODE + PORT3_OFFSET])


def set_port3_conn(x: NDArray, node_idx: int, target: int) -> None:
    """Set port 3 connection target for a node (in-place)."""
    x[node_idx * GENES_PER_NODE + PORT3_OFFSET] = target


# =============================================================================
# Bulk Access (Vectorized)
# =============================================================================

def get_all_piece_types(x: NDArray, n_nodes: int) -> NDArray:
    """Extract all piece types as a 1D array.

    Args:
        x: Chromosome array.
        n_nodes: Number of nodes.

    Returns:
        Array of piece types, shape (n_nodes,).
    """
    return x[TYPE_OFFSET::GENES_PER_NODE][:n_nodes].astype(np.int16)


def get_all_port2_conns(x: NDArray, n_nodes: int) -> NDArray:
    """Extract all port 2 connections as a 1D array."""
    return x[PORT2_OFFSET::GENES_PER_NODE][:n_nodes].astype(np.int16)


def get_all_port3_conns(x: NDArray, n_nodes: int) -> NDArray:
    """Extract all port 3 connections as a 1D array."""
    return x[PORT3_OFFSET::GENES_PER_NODE][:n_nodes].astype(np.int16)


def get_active_node_indices(x: NDArray, n_nodes: int) -> NDArray:
    """Get indices of active nodes (piece_type != INACTIVE).

    Returns:
        Integer array of active node indices.
    """
    types = get_all_piece_types(x, n_nodes)
    return np.where(types != INACTIVE)[0]


def get_main_loop_pieces(x: NDArray, n_nodes: int,
                         branch_nodes: Optional[Set[int]] = None) -> List[Tuple[int, int]]:
    """Extract main loop piece sequence (excluding branch nodes).

    The main loop is the sequential chain of active nodes that are NOT
    claimed by branch connections.

    Args:
        x: Chromosome array.
        n_nodes: Number of nodes.
        branch_nodes: Set of node indices used in branches (to exclude).

    Returns:
        List of (node_idx, piece_type) for main loop pieces in order.
    """
    if branch_nodes is None:
        branch_nodes = set()

    pieces = []
    for i in range(n_nodes):
        pt = get_piece_type(x, i)
        if pt != INACTIVE and i not in branch_nodes:
            pieces.append((i, pt))
    return pieces


# =============================================================================
# Connection Analysis
# =============================================================================

def find_switch_connections(x: NDArray, n_nodes: int) -> List[Tuple[int, int, int, int]]:
    """Find switch IN→OUT pairs linked by port 2 connections.

    Scans for IN switches whose port2_conn points to a chain of nodes
    that terminates at an OUT switch's port2_conn.

    Returns:
        List of (in_node, out_node, branch_start, branch_end) tuples.
    """
    connections = []
    for i in range(n_nodes):
        pt = get_piece_type(x, i)
        if pt in IN_SWITCH_INDICES:
            branch_start = get_port2_conn(x, i)
            if branch_start == INACTIVE or branch_start >= n_nodes:
                continue
            # Find matching OUT switch that references the branch end
            for j in range(n_nodes):
                pt_j = get_piece_type(x, j)
                if pt_j in OUT_SWITCH_INDICES:
                    branch_end = get_port2_conn(x, j)
                    if branch_end != INACTIVE and branch_start <= branch_end < n_nodes:
                        # Verify compatible pair (LEFT_IN↔LEFT_OUT or RIGHT_IN↔RIGHT_OUT)
                        if _are_compatible_switches(pt, pt_j):
                            connections.append((i, j, branch_start, branch_end))
    return connections


def _are_compatible_switches(in_type: int, out_type: int) -> bool:
    """Check if IN and OUT switch types are compatible for a siding."""
    for in_idx, out_idx in SWITCH_PAIRS:
        if in_type == in_idx and out_type == out_idx:
            return True
        # Reverse variants: OUT acts as IN, IN acts as OUT
        if in_type == out_idx and out_type == in_idx:
            return True
    return False


def find_crossing_connections(x: NDArray, n_nodes: int) -> List[Tuple[int, int, int]]:
    """Find CROSS_90 / DOUBLE_CROSSOVER nodes with port 2/3 connections.

    Returns:
        List of (node_idx, port2_target, port3_target) tuples.
    """
    crossings = []
    for i in range(n_nodes):
        pt = get_piece_type(x, i)
        if pt in CROSSING_INDICES:
            p2 = get_port2_conn(x, i)
            p3 = get_port3_conn(x, i)
            if p2 != INACTIVE or p3 != INACTIVE:
                crossings.append((i, p2, p3))
    return crossings


def get_branch_nodes(x: NDArray, n_nodes: int,
                     branch_start: int, branch_end: int) -> List[int]:
    """Get sequential chain of active nodes forming a branch.

    Branch nodes are the sequential range [branch_start, branch_end]
    that have active piece types.

    Args:
        x: Chromosome array.
        n_nodes: Number of nodes.
        branch_start: First node of branch.
        branch_end: Last node of branch (inclusive).

    Returns:
        List of active node indices in the branch.
    """
    nodes = []
    for i in range(branch_start, min(branch_end + 1, n_nodes)):
        if get_piece_type(x, i) != INACTIVE:
            nodes.append(i)
    return nodes


# =============================================================================
# Chromosome Construction
# =============================================================================

def create_empty_chromosome(dims: ChromosomeDimensions) -> NDArray:
    """Create a chromosome with all nodes inactive.

    Args:
        dims: Chromosome dimensions.

    Returns:
        Integer array of length n_var, all INACTIVE.
    """
    return np.full(dims.n_var, INACTIVE, dtype=np.int16)


def create_chromosome_from_pieces(
    dims: ChromosomeDimensions,
    main_loop_pieces: List[int],
    branch_specs: Optional[List[Tuple[int, int, int, List[int]]]] = None,
) -> NDArray:
    """Create a chromosome from a piece sequence and optional branches.

    Args:
        dims: Chromosome dimensions.
        main_loop_pieces: List of piece type indices for the main loop.
        branch_specs: Optional list of (in_switch_pos, out_switch_pos,
                      handedness, branch_pieces) tuples. Positions are
                      indices into main_loop_pieces.

    Returns:
        Chromosome array.
    """
    x = create_empty_chromosome(dims)
    n_main = min(len(main_loop_pieces), dims.n_nodes)

    # Place main loop pieces
    for i in range(n_main):
        set_node(x, i, main_loop_pieces[i])

    if not branch_specs:
        return x

    # Place branches after main loop pieces
    next_free = n_main
    for in_pos, out_pos, _handedness, branch_pieces in branch_specs:
        n_branch = len(branch_pieces)
        if next_free + n_branch > dims.n_nodes:
            break  # No room

        branch_start = next_free
        branch_end = next_free + n_branch - 1

        # Place branch pieces
        for j, bp in enumerate(branch_pieces):
            set_node(x, next_free + j, bp)

        # Wire connections: IN switch port2 → branch start
        if 0 <= in_pos < n_main:
            set_port2_conn(x, in_pos, branch_start)
        # OUT switch port2 → branch end
        if 0 <= out_pos < n_main:
            set_port2_conn(x, out_pos, branch_end)

        next_free += n_branch

    return x


# =============================================================================
# Validation
# =============================================================================

def validate_chromosome(x: NDArray, dims: ChromosomeDimensions,
                        max_piece_index: int = 9) -> List[str]:
    """Validate chromosome gene values are within bounds.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    if len(x) != dims.n_var:
        errors.append(f"Length {len(x)} != expected {dims.n_var}")
        return errors

    for i in range(dims.n_nodes):
        pt, p2, p3 = get_node(x, i)

        if pt < INACTIVE or pt > max_piece_index:
            errors.append(f"Node {i}: piece_type {pt} out of range [{INACTIVE}, {max_piece_index}]")

        if p2 < INACTIVE or p2 >= dims.n_nodes:
            errors.append(f"Node {i}: port2_conn {p2} out of range [{INACTIVE}, {dims.n_nodes - 1}]")

        if p3 < INACTIVE or p3 >= dims.n_nodes:
            errors.append(f"Node {i}: port3_conn {p3} out of range [{INACTIVE}, {dims.n_nodes - 1}]")

        # Connection genes should be -1 for pieces that don't have extra ports
        if pt != INACTIVE and pt not in THREE_PORT_PIECES and pt not in FOUR_PORT_PIECES:
            if p2 != INACTIVE:
                errors.append(f"Node {i}: 2-port piece {pt} has port2_conn={p2} (should be -1)")
            if p3 != INACTIVE:
                errors.append(f"Node {i}: 2-port piece {pt} has port3_conn={p3} (should be -1)")

        if pt in THREE_PORT_PIECES and p3 != INACTIVE:
            errors.append(f"Node {i}: 3-port piece {pt} has port3_conn={p3} (should be -1)")

    return errors


# =============================================================================
# Chromosome Statistics
# =============================================================================

def chromosome_stats(x: NDArray, dims: ChromosomeDimensions) -> Dict:
    """Compute summary statistics for a chromosome.

    Returns:
        Dict with piece counts, connection counts, etc.
    """
    types = get_all_piece_types(x, dims.n_nodes)
    p2 = get_all_port2_conns(x, dims.n_nodes)
    p3 = get_all_port3_conns(x, dims.n_nodes)

    active_mask = types != INACTIVE
    n_active = int(np.sum(active_mask))

    # Count piece types
    piece_counts: Dict[int, int] = {}
    for pt in types[active_mask]:
        pt_int = int(pt)
        piece_counts[pt_int] = piece_counts.get(pt_int, 0) + 1

    n_connections_p2 = int(np.sum(p2 != INACTIVE))
    n_connections_p3 = int(np.sum(p3 != INACTIVE))
    n_switches = sum(piece_counts.get(s, 0) for s in SWITCH_INDICES)
    n_crossings = sum(piece_counts.get(c, 0) for c in CROSSING_INDICES)

    return {
        "n_nodes": dims.n_nodes,
        "n_active": n_active,
        "n_inactive": dims.n_nodes - n_active,
        "piece_counts": piece_counts,
        "n_switches": n_switches,
        "n_crossings": n_crossings,
        "n_port2_connections": n_connections_p2,
        "n_port3_connections": n_connections_p3,
    }
