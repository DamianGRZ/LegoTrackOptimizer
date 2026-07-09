"""Track catalog: registry of track pieces with vectorized access."""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


# Canonical piece_id -> chromosome index. The GA encoding is tied to these exact
# indices, so the mapping is pinned here rather than read from YAML piece order;
# sampling, operators and repair all rely on it staying stable.
_CANONICAL_PIECE_INDEX: Dict[str, int] = {
    "STRAIGHT_16": 0,
    "STRAIGHT_24": 1,
    "R40_CURVE": 2,
    "CROSS_90": 3,
    "R40_SWITCH_LEFT": 4,
    "R40_SWITCH_RIGHT": 5,
    "DOUBLE_CROSSOVER": 6,
}

# Default-route radius_mm / speed_limit per piece. The runtime tables store one
# physics row per piece — its *default* route (see _default_route_name), whose
# exit port is the FK row. The port-centric YAML doesn't carry per-route
# physics, so these constants fill fk_table / radius_table / speed_table.
_DEFAULT_PHYSICS: Dict[str, Tuple[Optional[float], float]] = {
    # piece_id -> (radius_mm, speed_limit_ms) on the default route
    "R40_CURVE": (320.0, 0.97),
}

# Per-route physics for multi-route pieces, used to build each piece's
# routes_data list (radius/speed keyed by route name).
_ROUTE_PHYSICS: Dict[str, Dict[str, Tuple[Optional[float], float]]] = {
    "R40_SWITCH_LEFT":  {"through": (None, 1.57), "diverging": (320.0, 0.97)},
    "R40_SWITCH_RIGHT": {"through": (None, 1.57), "diverging": (320.0, 0.97)},
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

    PIECE_TYPE_TO_CLASS = {
        "straight": PieceClass.SIMPLE_2PORT,
        "curve": PieceClass.SIMPLE_2PORT,
        "crossing": PieceClass.CROSSING_4PORT,
        "switch": PieceClass.SWITCH_3PORT,
        "bumper": PieceClass.BUMPER_1PORT,
    }

    DEFAULT_SPEED = 1.57

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
        self._warned_inventory_ids: set = set()
        self.stud_mm: float = 8.0

    @classmethod
    def load(cls, path: str | Path) -> "TrackCatalog":
        """Load the port-centric catalog YAML into runtime numpy tables."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not _is_port_centric_schema(data):
            raise ValueError(
                f"{path}: not a port-centric catalog. Expected a top-level "
                f"`pieces:` list and `meta.schema_version` (see data/track_pieces_v2.yaml)."
            )
        return cls._from_spec(load_catalog_spec(path))

    @classmethod
    def _from_spec(cls, spec: TrackCatalogSpec) -> "TrackCatalog":
        """Build a TrackCatalog (runtime numpy tables) from a validated TrackCatalogSpec.

        Flattens the port-centric spec into the runtime tables by picking each
        piece's default route and converting radians -> degrees for the angle
        convention geometry.compute_fk_chain expects.
        """
        catalog = cls()
        catalog.stud_mm = spec.meta.stud_mm

        for ps in spec.pieces:
            if ps.piece_id not in _CANONICAL_PIECE_INDEX:
                # Unknown piece — skip rather than invent an index; the chromosome
                # encoding is tied to the canonical map.
                continue
            index = _CANONICAL_PIECE_INDEX[ps.piece_id]
            catalog._max_index = max(catalog._max_index, index)

            main_route_name = _default_route_name(ps)
            main_port_seq = ps.routes[main_route_name]
            exit_port = ps.ports[main_port_seq[-1]]

            fk = FKDeltas(
                dx=exit_port.dx,
                dy=exit_port.dy,
                # The runtime _fk_table stores angles in DEGREES
                # (geometry.py converts via np.radians).
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

            piece_type = ps.kind
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
                routes_data=_build_routes_data(ps),
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

    def inventory_by_index(self, inventory: Dict[str, int]) -> Dict[int, int]:
        """Convert a {piece_id: count} inventory to {piece_index: count}.

        Piece IDs missing from the catalog cannot be placed, so they are
        dropped from the result. Dropping is loud: each unknown-ID set is
        logged once per catalog instance, not per call — the decoder converts
        on every evaluation.
        """
        unknown = tuple(sorted(pid for pid in inventory if pid not in self._id_to_index))
        if unknown and unknown not in self._warned_inventory_ids:
            self._warned_inventory_ids.add(unknown)
            log.warning("Inventory IDs not in catalog, ignored: %s", ", ".join(unknown))
        return {
            self._id_to_index[pid]: count
            for pid, count in inventory.items()
            if pid in self._id_to_index
        }

    def __getitem__(self, index: int) -> Optional[TrackPiece]:
        return self._index_to_piece.get(index)


# =============================================================================
# Port-centric schema: detection + translation into the runtime tables
# =============================================================================


def _is_port_centric_schema(data: Any) -> bool:
    """A valid catalog has a top-level `pieces:` list and `meta.schema_version`."""
    if not isinstance(data, dict):
        return False
    meta = data.get("meta")
    return (
        "pieces" in data
        and isinstance(meta, dict)
        and "schema_version" in meta
    )


def _default_route_name(ps: TrackPieceSpec) -> str:
    """Pick the route whose exit port is the 'main FK' row for the runtime table.

    Preference order:
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

    The port-centric YAML doesn't encode per-route physics; we map known
    piece_ids to their physics values, falling back to (None, DEFAULT_SPEED).
    """
    default_speed = TrackCatalog.DEFAULT_SPEED
    default_route = _default_route_name(ps)
    per_route = _ROUTE_PHYSICS.get(ps.piece_id)
    if per_route and default_route in per_route:
        return per_route[default_route]
    if ps.piece_id in _DEFAULT_PHYSICS:
        return _DEFAULT_PHYSICS[ps.piece_id]
    return (None, default_speed)


def _build_routes_data(ps: TrackPieceSpec) -> Optional[List[Dict[str, Any]]]:
    """Build the routes_data list the runtime parser consumes.

    Single-route 2-port pieces: return None (_parse_routes then synthesizes a
    default route when routes_data is falsy). Multi-route pieces emit one dict
    per route with entry_port/exit_port as integer indices matching the order
    ports appear in ps.ports, and dtheta converted to degrees.
    """
    if len(ps.routes) <= 1:
        return None

    port_names = list(ps.ports)
    per_route_physics = _ROUTE_PHYSICS.get(ps.piece_id, {})

    out: List[Dict[str, Any]] = []
    for name, port_seq in ps.routes.items():
        entry_name = port_seq[0]
        exit_name = port_seq[-1]
        entry_port_spec = ps.ports[entry_name]
        exit_port_spec = ps.ports[exit_name]

        # Route FK = exit pose in the entering train's frame, so e.g. the
        # CROSS_90 vertical route is a straight-through (length, 0, 0) even
        # though its entry port is authored as an outward normal.
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


def _route_fk_in_train_frame(entry, exit_port) -> Tuple[float, float, float]:
    """Compute (dx, dy, dtheta_rad) of the route in the entering train's frame.

    Port dtheta may be authored as travel direction (A/B-style entries) or as
    the outward normal (CROSS_90 C/D); an entry heading opposing the route
    chord is flipped by pi so the frame always follows the train.
    """
    dx_world = exit_port.dx - entry.dx
    dy_world = exit_port.dy - entry.dy
    entry_heading = entry.dtheta
    if math.cos(math.atan2(dy_world, dx_world) - entry_heading) < 0:
        entry_heading += math.pi
    c = math.cos(-entry_heading)
    s = math.sin(-entry_heading)
    dx = dx_world * c - dy_world * s
    dy = dx_world * s + dy_world * c
    dtheta = exit_port.dtheta - entry_heading
    return (dx, dy, dtheta)
