"""Track catalog loading and vectorized piece access."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from numpy.typing import NDArray

from .topology import FKRoute, PieceClass, PieceTopology


@dataclass(frozen=True)
class FKDeltas:
    """Forward kinematics deltas for a single piece."""

    dx: float
    dy: float
    dtheta: float

    def to_array(self) -> NDArray[np.float64]:
        """Return [dx, dy, dtheta] as numpy array."""
        return np.array([self.dx, self.dy, self.dtheta], dtype=np.float64)


@dataclass(frozen=True)
class Port:
    """Connection point on a track piece."""

    x: float
    y: float
    heading: float
    gender: str  # "M" or "F"


@dataclass
class TrackPiece:
    """Single track piece definition."""

    id: str
    name: str
    piece_type: str  # 'straight', 'curve', 'crossing', 'switch', 'bumper'
    fk: FKDeltas
    ports: Tuple[Port, ...]
    index: int
    length: float = 16.0
    radius: Optional[float] = None
    angle: Optional[float] = None
    direction: Optional[str] = None  # 'left' or 'right'
    radius_mm: Optional[float] = None
    speed_limit_ms: float = 1.57  # Motor top speed default
    is_terminator: bool = False
    routes_data: Optional[List[Dict[str, Any]]] = None

    @property
    def is_straight(self) -> bool:
        """Check if piece is a straight."""
        return self.piece_type == "straight"

    @property
    def is_curve(self) -> bool:
        """Check if piece is a curve."""
        return self.piece_type == "curve"

    @property
    def arc_length(self) -> float:
        """Arc length in studs."""
        if self.is_straight:
            return self.length
        elif self.is_curve and self.radius and self.angle:
            return self.radius * math.radians(abs(self.angle))
        else:
            return self.length


class TrackCatalog:
    """Manages track piece inventory with vectorized access."""

    SECTION_TYPES = {
        "straights": "straight",
        "curves": "curve",
        "crossings": "crossing",
        "r40_switch_components": "switch",
        "bumpers": "bumper",
    }

    PIECE_TYPE_TO_CLASS = {
        "straight": PieceClass.SIMPLE_2PORT,
        "curve": PieceClass.SIMPLE_2PORT,
        "crossing": PieceClass.CROSSING_4PORT,
        "switch": PieceClass.SWITCH_3PORT,
        "bumper": PieceClass.BUMPER_1PORT,
    }

    DEFAULT_SPEED = 1.57  # Motor top speed

    def __init__(self) -> None:
        self._pieces: Dict[str, TrackPiece] = {}
        self._index_to_piece: Dict[int, TrackPiece] = {}
        self._id_to_index: Dict[str, int] = {}
        self._fk_table: NDArray[np.float64] = np.zeros((0, 3))
        self._speed_table: NDArray[np.float64] = np.zeros(0)
        self._radius_table: NDArray[np.float64] = np.zeros(0)
        self._arc_length_table: NDArray[np.float64] = np.zeros(0)
        self._topologies: Dict[int, PieceTopology] = {}
        self._route_fk_tables: Dict[int, List[NDArray[np.float64]]] = {}
        self._max_index: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "TrackCatalog":
        """Load catalog from YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        catalog = cls()
        catalog._parse_yaml(data)
        catalog._build_tables()
        return catalog

    def _parse_yaml(self, data: Dict[str, Any]) -> None:
        """Parse YAML structure."""
        # First, get the piece_index mapping
        piece_index = data.get("piece_index", {})
        id_to_index: Dict[str, int] = {}
        for idx, piece_id in piece_index.items():
            if piece_id is not None and idx != -1:
                id_to_index[piece_id] = int(idx)
                self._max_index = max(self._max_index, int(idx))

        self._id_to_index = id_to_index

        # Parse each section
        for section, piece_type in self.SECTION_TYPES.items():
            if section in data:
                for piece_data in data[section]:
                    piece = self._parse_piece(piece_data, piece_type, id_to_index)
                    if piece:
                        self._register_piece(piece)

    def _parse_piece(
        self, data: Dict[str, Any], piece_type: str, id_to_index: Dict[str, int]
    ) -> Optional[TrackPiece]:
        """Parse single piece from YAML data."""
        piece_id = data.get("id")
        if not piece_id or piece_id not in id_to_index:
            return None

        index = id_to_index[piece_id]

        # Parse FK
        fk_data = data.get("fk", {})
        fk = FKDeltas(
            dx=fk_data.get("dx", 0.0),
            dy=fk_data.get("dy", 0.0),
            dtheta=fk_data.get("dtheta", 0.0),
        )

        # Parse ports
        ports_data = data.get("ports", [])
        ports: List[Port] = []
        for p in ports_data:
            pos = p.get("pos", [0, 0])
            ports.append(
                Port(
                    x=pos[0],
                    y=pos[1],
                    heading=p.get("heading", 0.0),
                    gender=p.get("gender", "M"),
                )
            )

        # Parse physics
        physics = data.get("physics", {})
        radius_mm = physics.get("radius_mm")
        speed_limit = physics.get("speed_limit_ms", self.DEFAULT_SPEED)

        # Handle closure data (for curves)
        closure = data.get("closure", {})

        return TrackPiece(
            id=piece_id,
            name=data.get("name", piece_id),
            piece_type=piece_type,
            fk=fk,
            ports=tuple(ports),
            index=index,
            length=data.get("length", data.get("footprint", {}).get("length", 16.0)),
            radius=data.get("radius"),
            angle=data.get("angle"),
            direction=data.get("direction"),
            radius_mm=radius_mm,
            speed_limit_ms=speed_limit if speed_limit else self.DEFAULT_SPEED,
            is_terminator=(piece_type == "bumper"),
            routes_data=data.get("routes"),
        )

    def _register_piece(self, piece: TrackPiece) -> None:
        """Add piece to catalog."""
        self._pieces[piece.id] = piece
        self._index_to_piece[piece.index] = piece

        # Create topology
        topology = self._create_topology(piece)
        self._topologies[piece.index] = topology

    def _create_topology(self, piece: TrackPiece) -> PieceTopology:
        """Create topology metadata for piece."""
        piece_class = self.PIECE_TYPE_TO_CLASS.get(piece.piece_type, PieceClass.SIMPLE_2PORT)

        # Determine num_ports from ports list or piece class
        num_ports = len(piece.ports) if piece.ports else 2

        # Parse routes
        routes = self._parse_routes(piece.routes_data, piece)

        # Port positions and headings
        port_positions = tuple((p.x, p.y) for p in piece.ports)
        port_headings = tuple(p.heading for p in piece.ports)

        return PieceTopology(
            piece_id=piece.id,
            piece_index=piece.index,
            piece_class=piece_class,
            num_ports=num_ports,
            routes=routes,
            default_route_idx=0,
            port_positions=port_positions,
            port_headings=port_headings,
        )

    def _parse_routes(
        self, routes_data: Optional[List[Dict[str, Any]]], piece: TrackPiece
    ) -> Tuple[FKRoute, ...]:
        """Parse multi-route data from YAML."""
        if not routes_data:
            # Single default route
            return (
                FKRoute(
                    entry_port=0,
                    exit_port=1,
                    dx=piece.fk.dx,
                    dy=piece.fk.dy,
                    dtheta=piece.fk.dtheta,
                    arc_length=piece.arc_length,
                    radius_mm=piece.radius_mm,
                    speed_limit=piece.speed_limit_ms,
                ),
            )

        routes: List[FKRoute] = []
        for route in routes_data:
            fk_data = route.get("fk", {})
            physics = route.get("physics", {})

            routes.append(
                FKRoute(
                    entry_port=route.get("entry_port", 0),
                    exit_port=route.get("exit_port", 1),
                    dx=fk_data.get("dx", 0.0),
                    dy=fk_data.get("dy", 0.0),
                    dtheta=fk_data.get("dtheta", 0.0),
                    arc_length=piece.arc_length,
                    radius_mm=physics.get("radius_mm"),
                    speed_limit=physics.get("speed_limit_ms", self.DEFAULT_SPEED),
                )
            )

        return tuple(routes)

    def _build_tables(self) -> None:
        """Build numpy lookup tables for vectorized access."""
        n = self._max_index + 1

        self._fk_table = np.zeros((n, 3), dtype=np.float64)
        self._speed_table = np.full(n, self.DEFAULT_SPEED, dtype=np.float64)
        self._radius_table = np.full(n, np.inf, dtype=np.float64)  # inf for straights
        self._arc_length_table = np.zeros(n, dtype=np.float64)

        for idx, piece in self._index_to_piece.items():
            if 0 <= idx < n:
                self._fk_table[idx] = piece.fk.to_array()
                self._speed_table[idx] = piece.speed_limit_ms
                if piece.radius_mm:
                    self._radius_table[idx] = piece.radius_mm
                self._arc_length_table[idx] = piece.arc_length

        # Build route FK tables for multi-route pieces
        self._build_route_tables()

        # Build piece role classification tables (fork/merge/crossing)
        self._build_role_tables()

    def _build_route_tables(self) -> None:
        """Build multi-route FK tables."""
        for idx, topology in self._topologies.items():
            if len(topology.routes) > 1:
                self._route_fk_tables[idx] = [route.to_array() for route in topology.routes]

    def _vectorized_lookup(
        self, indices: NDArray, table: NDArray, default: float = 0.0
    ) -> NDArray[np.float64]:
        """Generic vectorized lookup with bounds checking."""
        result = np.full(len(indices), default, dtype=np.float64)
        valid = (indices >= 0) & (indices < len(table))
        result[valid] = table[indices[valid]]
        return result

    def get_fk(self, indices: NDArray) -> NDArray[np.float64]:
        """Get FK deltas for piece indices.

        Args:
            indices: Array of piece indices.

        Returns:
            (n, 3) array of [dx, dy, dtheta] for each piece.
        """
        indices = np.asarray(indices, dtype=np.int32)
        result = np.zeros((len(indices), 3), dtype=np.float64)

        valid = (indices >= 0) & (indices < len(self._fk_table))
        result[valid] = self._fk_table[indices[valid]]

        return result

    def get_radii(self, indices: NDArray) -> NDArray[np.float64]:
        """Get radii in mm for piece indices.

        Args:
            indices: Array of piece indices.

        Returns:
            Array of radii (inf for straights).
        """
        indices = np.asarray(indices, dtype=np.int32)
        return self._vectorized_lookup(indices, self._radius_table, np.inf)

    def get_arc_lengths(self, indices: NDArray) -> NDArray[np.float64]:
        """Get arc lengths in studs for piece indices.

        Args:
            indices: Array of piece indices.

        Returns:
            Array of arc lengths.
        """
        indices = np.asarray(indices, dtype=np.int32)
        return self._vectorized_lookup(indices, self._arc_length_table, 0.0)

    def get_speed_limits(self, indices: NDArray) -> NDArray[np.float64]:
        """Get speed limits in m/s for piece indices.

        Args:
            indices: Array of piece indices.

        Returns:
            Array of speed limits.
        """
        indices = np.asarray(indices, dtype=np.int32)
        return self._vectorized_lookup(indices, self._speed_table, self.DEFAULT_SPEED)

    def get_topology(self, piece_idx: int) -> Optional[PieceTopology]:
        """Get topology metadata for piece index."""
        return self._topologies.get(piece_idx)

    def get_topologies(self, indices: NDArray) -> List[Optional[PieceTopology]]:
        """Get topology metadata for multiple piece indices."""
        return [self.get_topology(int(idx)) for idx in indices]

    def get_fk_route(self, piece_idx: int, route_idx: int = 0) -> NDArray[np.float64]:
        """Get FK for specific route.

        Args:
            piece_idx: Piece index.
            route_idx: Route index (0 = default).

        Returns:
            [dx, dy, dtheta] array.
        """
        if piece_idx in self._route_fk_tables:
            routes = self._route_fk_tables[piece_idx]
            if 0 <= route_idx < len(routes):
                return routes[route_idx]

        # Fall back to default FK
        if 0 <= piece_idx < len(self._fk_table):
            return self._fk_table[piece_idx]

        return np.zeros(3, dtype=np.float64)

    def get_fk_with_routes(
        self, piece_indices: NDArray, route_indices: NDArray
    ) -> NDArray[np.float64]:
        """Get FK deltas with route selection for each piece.

        Args:
            piece_indices: Array of piece indices.
            route_indices: Array of route indices (same length).

        Returns:
            (n, 3) array of [dx, dy, dtheta] for each piece.
        """
        piece_indices = np.asarray(piece_indices, dtype=np.int32)
        route_indices = np.asarray(route_indices, dtype=np.int32)

        n = len(piece_indices)
        result = np.zeros((n, 3), dtype=np.float64)

        for i in range(n):
            result[i] = self.get_fk_route(int(piece_indices[i]), int(route_indices[i]))

        return result

    def get_piece_class(self, piece_idx: int) -> Optional[PieceClass]:
        """Get piece classification."""
        topology = self.get_topology(piece_idx)
        return topology.piece_class if topology else None

    def classify_pieces(self) -> Dict[PieceClass, List[int]]:
        """Group pieces by class."""
        result: Dict[PieceClass, List[int]] = {pc: [] for pc in PieceClass}
        for idx, topology in self._topologies.items():
            result[topology.piece_class].append(idx)
        return result

    def get_simple_pieces(self) -> List[int]:
        """Get 2-port piece indices (straights, curves)."""
        return [
            idx
            for idx, t in self._topologies.items()
            if t.piece_class == PieceClass.SIMPLE_2PORT
        ]

    def get_switch_pieces(self) -> List[int]:
        """Get switch piece indices (3-4 ports)."""
        return [
            idx
            for idx, t in self._topologies.items()
            if t.piece_class in (PieceClass.SWITCH_3PORT, PieceClass.SWITCH_4PORT)
        ]

    # =========================================================================
    # Port-based piece role classification (Phase B)
    # =========================================================================

    def get_piece_role(self, piece_idx: int) -> str:
        """Classify piece by its route structure: 'fork', 'merge', 'crossing', or 'simple'.

        Derived programmatically from routes — no hardcoded piece lists.
        - Fork: same entry port, multiple exit ports (e.g., SWITCH_LEFT_IN)
        - Merge: multiple entry ports, same exit port (e.g., SWITCH_LEFT_OUT)
        - Crossing: disjoint port sets per route (e.g., CROSS_90)
        """
        topo = self.get_topology(piece_idx)
        if topo is None or len(topo.routes) < 2:
            return "simple"

        entry_ports = {r.entry_port for r in topo.routes}
        exit_ports = {r.exit_port for r in topo.routes}

        if len(entry_ports) == 1 and len(exit_ports) > 1:
            return "fork"
        if len(entry_ports) > 1 and len(exit_ports) == 1:
            return "merge"
        return "crossing"

    def get_alternate_route(self, piece_idx: int) -> Optional[FKRoute]:
        """Get the non-default route for a multi-route piece.

        For forks: the diverge route. For merges: the merge route.
        Returns None for simple pieces or pieces without alternate routes.
        """
        topo = self.get_topology(piece_idx)
        if topo is None or len(topo.routes) < 2:
            return None
        return topo.routes[1]  # Route 0 is default, route 1 is alternate

    def can_pair(self, fork_idx: int, merge_idx: int) -> bool:
        """Check if fork piece can pair with merge piece for a branch.

        Pairing requires their alternate route angles to cancel
        (fork diverges +θ, merge converges -θ, or vice versa).
        """
        fork_alt = self.get_alternate_route(fork_idx)
        merge_alt = self.get_alternate_route(merge_idx)
        if not fork_alt or not merge_alt:
            return False
        # Angles must cancel: LEFT_IN (+22.5) + LEFT_OUT (-22.5) = 0
        return abs(fork_alt.dtheta + merge_alt.dtheta) < 1.0

    def _build_role_tables(self) -> None:
        """Build lookup tables for piece roles and compatibility."""
        self._fork_indices: set = set()
        self._merge_indices: set = set()
        self._crossing_indices: set = set()
        self._compatible_pairs: Dict[int, List[int]] = {}

        for idx in self._topologies:
            role = self.get_piece_role(idx)
            if role == "fork":
                self._fork_indices.add(idx)
            elif role == "merge":
                self._merge_indices.add(idx)
            elif role == "crossing":
                self._crossing_indices.add(idx)

        # Build compatibility map: which merges can each fork pair with?
        for fork_idx in self._fork_indices:
            compatible = [
                merge_idx for merge_idx in self._merge_indices
                if self.can_pair(fork_idx, merge_idx)
            ]
            if compatible:
                self._compatible_pairs[fork_idx] = compatible

    def validate_inventory(self, usage: Dict[int, int], inventory: Dict[str, int]) -> bool:
        """Check if usage is within inventory limits.

        Args:
            usage: {piece_index: count} of used pieces.
            inventory: {piece_id: count} of available pieces.

        Returns:
            True if all usage is within limits.
        """
        for idx, count in usage.items():
            piece = self._index_to_piece.get(idx)
            if piece:
                available = inventory.get(piece.id, 0)
                if count > available:
                    return False
        return True

    @property
    def n_pieces(self) -> int:
        """Total number of piece types."""
        return len(self._pieces)

    @property
    def fk_table(self) -> NDArray[np.float64]:
        """Raw FK lookup table."""
        return self._fk_table

    @property
    def id_to_index(self) -> Dict[str, int]:
        """ID to index mapping."""
        return self._id_to_index

    @property
    def index_to_id(self) -> Dict[int, str]:
        """Index to ID mapping (dynamically generated from catalog).

        This replaces hardcoded INDEX_TO_ID constants, making the system
        automatically adapt when new pieces are added to track_pieces.yaml.
        """
        return {idx: piece.id for idx, piece in self._index_to_piece.items()}

    def __getitem__(self, index: int) -> Optional[TrackPiece]:
        """Get piece by index."""
        return self._index_to_piece.get(index)


def load_inventory(path: str | Path) -> Dict[str, int]:
    """Load inventory from YAML config file."""
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("inventory", {})
