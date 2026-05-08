"""Track catalog: registry of track pieces with vectorized access."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from numpy.typing import NDArray

from ..types import FKRoute, PieceClass, PieceTopology
from .loader import load_catalog_spec
from .pieces import FKDeltas, Port, TrackPiece
from .specs import TrackCatalogSpec, TrackPieceSpec


# Canonical piece_id -> chromosome index mapping (legacy v1 encoding).
# V2 YAML is order-agnostic; this map preserves the chromosome-index contract
# so existing GA code (sampling, operators, repair) keeps working unchanged.
_LEGACY_PIECE_INDEX: Dict[str, int] = {
    "STRAIGHT_16": 0,
    "STRAIGHT_24": 1,
    "R40_LEFT": 2,
    "R40_RIGHT": 3,
    "CROSS_90": 4,
    "R40_SWITCH_LEFT_IN": 5,
    "R40_SWITCH_LEFT_OUT": 6,
    "R40_SWITCH_RIGHT_IN": 7,
    "R40_SWITCH_RIGHT_OUT": 8,
    "DOUBLE_CROSSOVER": 9,
}

# V2 kind -> legacy piece_type string used by PIECE_TYPE_TO_CLASS.
_V2_KIND_TO_LEGACY_TYPE: Dict[str, str] = {
    "straight": "straight",
    "curve": "curve",
    "crossing": "crossing",
    "switch": "switch",
    "wye": "switch",
}

# Per-piece (v1) radius_mm / speed_limit for multi-route pieces on their
# *default* route (the one whose exit port is the FK row of the legacy table).
# V2 does not encode per-route physics; we supply v1-equivalent values so
# fk_table/radius_table/speed_table stay bit-identical across schemas.
#
# Default route for each piece (see _default_route_name):
#   switches:  "through"  -> straight-through -> speed 1.57, radius=inf
#   CROSS_90:  "horizontal" -> speed 1.57, radius=inf
#   DBL_CROSS: "track1_through" -> speed 1.57, radius=inf
# Curves keep their 0.97 / 320mm from v1.
_V2_DEFAULT_PHYSICS: Dict[str, Tuple[Optional[float], float]] = {
    # piece_id -> (radius_mm, speed_limit_ms) on the default route
    "R40_LEFT": (320.0, 0.97),
    "R40_RIGHT": (320.0, 0.97),
}

# Per-route physics overrides for multi-route pieces, used to reconstruct the
# legacy routes_data list in v1-parity form.
_V2_ROUTE_PHYSICS: Dict[str, Dict[str, Tuple[Optional[float], float]]] = {
    "R40_SWITCH_LEFT_IN":  {"through": (None, 1.57), "diverging": (320.0, 0.97)},
    "R40_SWITCH_LEFT_OUT": {"through": (None, 1.57), "diverging": (320.0, 0.97)},
    "R40_SWITCH_RIGHT_IN": {"through": (None, 1.57), "diverging": (320.0, 0.97)},
    "R40_SWITCH_RIGHT_OUT":{"through": (None, 1.57), "diverging": (320.0, 0.97)},
    "CROSS_90": {
        "horizontal": (None, 1.57),
        "vertical":   (None, 1.57),
    },
    "DOUBLE_CROSSOVER": {
        "track1_through": (None, 1.57),
        "track2_through": (None, 1.57),
        "cross_1_to_2":   (320.0, 0.97),
        "cross_2_to_1":   (320.0, 0.97),
    },
}


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
        self.stud_mm: float = 8.0

    @classmethod
    def load(cls, path: str | Path) -> "TrackCatalog":
        """Load catalog from YAML. Auto-detects v1 (section-keyed) vs v2 (port-centric)."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if _is_v2_schema(data):
            spec = load_catalog_spec(path)
            return cls._from_v2_spec(spec)

        import warnings
        warnings.warn(
            f"Loading legacy v1 catalog format from {path}. "
            f"Migrate to V2 port-centric schema (see data/track_pieces_v2.yaml). "
            f"v1 support will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        catalog = cls()
        catalog._parse_yaml(data)
        catalog._build_tables()
        return catalog

    @classmethod
    def _from_v2_spec(cls, spec: TrackCatalogSpec) -> "TrackCatalog":
        """Build a legacy-surface TrackCatalog from a V2 TrackCatalogSpec.

        Translates port-centric V2 data into the v1 numpy tables by picking
        each piece's default route and converting radians -> degrees for the
        legacy angle convention used by geometry.compute_fk_chain.
        """
        catalog = cls()
        catalog.stud_mm = spec.meta.stud_mm

        for ps in spec.pieces:
            if ps.piece_id not in _LEGACY_PIECE_INDEX:
                # Unknown piece — skip rather than invent an index; the chromosome
                # encoding is tied to the canonical map.
                continue
            index = _LEGACY_PIECE_INDEX[ps.piece_id]
            catalog._max_index = max(catalog._max_index, index)

            main_route_name = _default_route_name(ps)
            main_port_seq = ps.routes[main_route_name]
            exit_port = ps.ports[main_port_seq[-1]]

            fk = FKDeltas(
                dx=exit_port.dx,
                dy=exit_port.dy,
                # Legacy _fk_table stores DEGREES (geometry.py converts via np.radians).
                dtheta=math.degrees(exit_port.dtheta),
            )

            ports = tuple(
                Port(
                    x=port.dx,
                    y=port.dy,
                    heading=math.degrees(port.dtheta),
                    gender="F" if name == "A" else "M",
                )
                for name, port in ps.ports.items()
            )

            piece_type = _V2_KIND_TO_LEGACY_TYPE[ps.kind]
            radius_mm, speed_limit = _default_physics_for(ps)

            piece = TrackPiece(
                id=ps.piece_id,
                name=ps.piece_id,
                piece_type=piece_type,
                fk=fk,
                ports=ports,
                index=index,
                length=(ps.length_studs
                        if ps.length_studs is not None
                        else (ps.body_length_studs if ps.body_length_studs is not None
                              else 16.0)),
                radius=(ps.radius_studs
                        if ps.radius_studs is not None
                        else ps.diverging_radius_studs),
                angle=(math.degrees(ps.sector_angle_rad)
                       if ps.sector_angle_rad is not None else None),
                direction=ps.hand,
                radius_mm=radius_mm,
                speed_limit_ms=speed_limit,
                is_terminator=False,
                routes_data=_build_legacy_routes(ps),
            )

            catalog._pieces[ps.piece_id] = piece
            catalog._index_to_piece[index] = piece
            catalog._id_to_index[ps.piece_id] = index

        catalog._rebuild_topologies()
        catalog._build_tables()
        return catalog

    def _rebuild_topologies(self) -> None:
        """Construct PieceTopology entries for all registered pieces."""
        for piece in self._index_to_piece.values():
            self._topologies[piece.index] = self._create_topology(piece)

    def _parse_yaml(self, data: Dict[str, Any]) -> None:
        """Parse YAML structure."""
        metadata = data.get("metadata", {}) or {}
        self.stud_mm = float(metadata.get("stud_mm", 8.0))

        piece_index = data.get("piece_index", {})
        id_to_index: Dict[str, int] = {}
        for idx, piece_id in piece_index.items():
            if piece_id is not None and idx != -1:
                id_to_index[piece_id] = int(idx)
                self._max_index = max(self._max_index, int(idx))

        self._id_to_index = id_to_index

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

        fk_data = data.get("fk", {})
        fk = FKDeltas(
            dx=fk_data.get("dx", 0.0),
            dy=fk_data.get("dy", 0.0),
            dtheta=fk_data.get("dtheta", 0.0),
        )

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

        physics = data.get("physics", {})
        radius_mm = physics.get("radius_mm")
        speed_limit = physics.get("speed_limit_ms", self.DEFAULT_SPEED)

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

        topology = self._create_topology(piece)
        self._topologies[piece.index] = topology

    def _create_topology(self, piece: TrackPiece) -> PieceTopology:
        """Create topology metadata for piece."""
        piece_class = self.PIECE_TYPE_TO_CLASS.get(piece.piece_type, PieceClass.SIMPLE_2PORT)
        num_ports = len(piece.ports) if piece.ports else 2
        routes = self._parse_routes(piece.routes_data, piece)
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
        self._radius_table = np.full(n, np.inf, dtype=np.float64)
        self._arc_length_table = np.zeros(n, dtype=np.float64)

        for idx, piece in self._index_to_piece.items():
            if 0 <= idx < n:
                self._fk_table[idx] = piece.fk.to_array()
                self._speed_table[idx] = piece.speed_limit_ms
                if piece.radius_mm:
                    self._radius_table[idx] = piece.radius_mm
                self._arc_length_table[idx] = piece.arc_length

        self._build_route_tables()

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
        """Get FK deltas for piece indices."""
        indices = np.asarray(indices, dtype=np.int32)
        result = np.zeros((len(indices), 3), dtype=np.float64)
        valid = (indices >= 0) & (indices < len(self._fk_table))
        result[valid] = self._fk_table[indices[valid]]
        return result

    def get_radii(self, indices: NDArray) -> NDArray[np.float64]:
        """Get radii in mm for piece indices (inf for straights)."""
        indices = np.asarray(indices, dtype=np.int32)
        return self._vectorized_lookup(indices, self._radius_table, np.inf)

    def get_arc_lengths(self, indices: NDArray) -> NDArray[np.float64]:
        """Get arc lengths in studs for piece indices."""
        indices = np.asarray(indices, dtype=np.int32)
        return self._vectorized_lookup(indices, self._arc_length_table, 0.0)

    def get_speed_limits(self, indices: NDArray) -> NDArray[np.float64]:
        """Get speed limits in m/s for piece indices."""
        indices = np.asarray(indices, dtype=np.int32)
        return self._vectorized_lookup(indices, self._speed_table, self.DEFAULT_SPEED)

    def get_topology(self, piece_idx: int) -> Optional[PieceTopology]:
        """Get topology metadata for piece index."""
        return self._topologies.get(piece_idx)

    def get_fk_route(self, piece_idx: int, route_idx: int = 0) -> NDArray[np.float64]:
        """Get FK for specific route."""
        if piece_idx in self._route_fk_tables:
            routes = self._route_fk_tables[piece_idx]
            if 0 <= route_idx < len(routes):
                return routes[route_idx]
        if 0 <= piece_idx < len(self._fk_table):
            return self._fk_table[piece_idx]
        return np.zeros(3, dtype=np.float64)

    def get_fk_with_routes(
        self, piece_indices: NDArray, route_indices: NDArray
    ) -> NDArray[np.float64]:
        """Get FK deltas with route selection for each piece."""
        piece_indices = np.asarray(piece_indices, dtype=np.int32)
        route_indices = np.asarray(route_indices, dtype=np.int32)
        n = len(piece_indices)
        result = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            result[i] = self.get_fk_route(int(piece_indices[i]), int(route_indices[i]))
        return result

    def classify_pieces(self) -> Dict[PieceClass, List[int]]:
        """Group pieces by class."""
        result: Dict[PieceClass, List[int]] = {pc: [] for pc in PieceClass}
        for idx, topology in self._topologies.items():
            result[topology.piece_class].append(idx)
        return result

    def get_simple_pieces(self) -> List[int]:
        """Get 2-port piece indices (straights, curves)."""
        return [
            idx for idx, t in self._topologies.items()
            if t.piece_class == PieceClass.SIMPLE_2PORT
        ]

    def get_switch_pieces(self) -> List[int]:
        """Get switch piece indices (3-4 ports)."""
        return [
            idx for idx, t in self._topologies.items()
            if t.piece_class in (PieceClass.SWITCH_3PORT, PieceClass.SWITCH_4PORT)
        ]

    @property
    def n_pieces(self) -> int:
        return len(self._pieces)

    @property
    def fk_table(self) -> NDArray[np.float64]:
        return self._fk_table

    @property
    def radius_table(self) -> NDArray[np.float64]:
        return self._radius_table

    @property
    def speed_table(self) -> NDArray[np.float64]:
        return self._speed_table

    @property
    def arc_length_table(self) -> NDArray[np.float64]:
        return self._arc_length_table

    @property
    def id_to_index(self) -> Dict[str, int]:
        return self._id_to_index

    @property
    def index_to_id(self) -> Dict[int, str]:
        return {idx: piece.id for idx, piece in self._index_to_piece.items()}

    def __getitem__(self, index: int) -> Optional[TrackPiece]:
        return self._index_to_piece.get(index)


def load_inventory(path: str | Path) -> Dict[str, int]:
    """Load inventory from YAML config file."""
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("inventory", {})


# =============================================================================
# V2 schema compatibility shim
# =============================================================================


def _is_v2_schema(data: Any) -> bool:
    """V2 has top-level `pieces:` list and `meta.schema_version`; v1 has `straights:` etc."""
    if not isinstance(data, dict):
        return False
    meta = data.get("meta")
    return (
        "pieces" in data
        and isinstance(meta, dict)
        and "schema_version" in meta
    )


def _default_route_name(ps: TrackPieceSpec) -> str:
    """Pick the route whose exit port is the 'main FK' row for the legacy table.

    Preferences match v1 behavior:
      - simple 2-port pieces use "main"
      - switches use "through" (straight-through FK is the default)
      - crossings use whichever of "horizontal"/"track1_through" exists first
    """
    preference = ("main", "through", "horizontal", "track1_through")
    for name in preference:
        if name in ps.routes:
            return name
    return next(iter(ps.routes))


def _default_physics_for(ps: TrackPieceSpec) -> Tuple[Optional[float], float]:
    """Return (radius_mm, speed_limit_ms) for ps's default route.

    V2 does not encode per-route physics; we map known piece_ids to v1 values.
    Falls back to (None, DEFAULT_SPEED) for pieces without an explicit entry.
    """
    default_speed = TrackCatalog.DEFAULT_SPEED
    default_route = _default_route_name(ps)
    per_route = _V2_ROUTE_PHYSICS.get(ps.piece_id)
    if per_route and default_route in per_route:
        return per_route[default_route]
    if ps.piece_id in _V2_DEFAULT_PHYSICS:
        return _V2_DEFAULT_PHYSICS[ps.piece_id]
    return (None, default_speed)


def _build_legacy_routes(ps: TrackPieceSpec) -> Optional[List[Dict[str, Any]]]:
    """Reconstruct the legacy routes_data list the v1 parser produces.

    Single-route 2-port pieces: return None (v1 emits a synthesized default
    route in _parse_routes when routes_data is falsy). Multi-route pieces
    emit one dict per route with entry_port/exit_port as integer indices
    matching the order ports appear in ps.ports, and dtheta converted to
    degrees for legacy parity.
    """
    if len(ps.routes) <= 1:
        return None

    port_names = list(ps.ports)
    per_route_physics = _V2_ROUTE_PHYSICS.get(ps.piece_id, {})

    out: List[Dict[str, Any]] = []
    for name, port_seq in ps.routes.items():
        entry_name = port_seq[0]
        exit_name = port_seq[-1]
        entry_port_spec = ps.ports[entry_name]
        exit_port_spec = ps.ports[exit_name]

        # Route FK = exit pose in piece-local frame with entry port as origin.
        # All V2 routes start from port A (origin), so exit pose equals the
        # exit port's pose; when entry is non-A (e.g. CROSS_90 vertical),
        # the legacy fk is still defined in the train's own frame as a
        # straight-through (dx=length, dy=0, dtheta=0). V1 encodes this by
        # storing dx=length,dy=0,dtheta=0 for vertical route of CROSS_90;
        # we preserve that by using exit-relative-to-entry in the train's
        # local frame along the route.
        dx, dy, dtheta_rad = _route_fk_in_train_frame(
            entry_port_spec, exit_port_spec
        )

        radius_mm, speed_limit = per_route_physics.get(name, (None, TrackCatalog.DEFAULT_SPEED))

        out.append({
            "name": name,
            "entry_port": port_names.index(entry_name),
            "exit_port": port_names.index(exit_name),
            "fk": {
                "dx": dx,
                "dy": dy,
                "dtheta": math.degrees(dtheta_rad),
            },
            "physics": {
                "radius_mm": radius_mm,
                "speed_limit_ms": speed_limit,
            },
        })
    return out


def _route_fk_in_train_frame(entry, exit) -> Tuple[float, float, float]:
    """Compute (dx, dy, dtheta_rad) of `exit` relative to `entry` pose.

    Both arguments are PortDef instances giving SE(2) pose in the piece-local
    frame. The result is the rigid transform from entry frame to exit frame,
    which is what the legacy fk_table stores for each route.
    """
    c = math.cos(-entry.dtheta)
    s = math.sin(-entry.dtheta)
    dx_world = exit.dx - entry.dx
    dy_world = exit.dy - entry.dy
    dx = dx_world * c - dy_world * s
    dy = dx_world * s + dy_world * c
    dtheta = exit.dtheta - entry.dtheta
    return (dx, dy, dtheta)
