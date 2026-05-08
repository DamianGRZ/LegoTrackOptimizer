"""JSON serializers for the browser-side canvas renderer.

Two contracts:

- ``catalog_to_json(catalog)`` returns the V2 port-centric catalog as a plain
  dict tree the JS renderer can drive piece geometry from. Studs + radians
  preserved verbatim; ``stud_mm`` is included so the JS side can convert.
- ``port_graph_to_json(graph, catalog, config)`` returns a layout payload:
  per-active-slot world pose at port A (studs + radians), the port-pair edge
  list, the boundary in studs and mm, and a few aggregate stats the
  scorecard already shows.

Both are pure data; no numpy types leak — pymoo callbacks/runners frequently
get JSON-serialized through ``json.dumps`` directly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..catalog import TrackCatalog
from ..config import OptimizationConfig
from ..types import PortGraph


def catalog_to_json(catalog: TrackCatalog) -> Dict[str, Any]:
    """Serialize the V2 catalog spec to a render-ready dict tree.

    Requires the catalog was loaded from a V2 yaml (``catalog.spec`` is set).
    Falls back to an empty pieces list if the catalog is V1-only — the JS
    renderer will then render nothing rather than crash.
    """
    spec = catalog.spec
    if spec is None:
        return {"stud_mm": catalog.stud_mm, "pieces": []}

    pieces: List[Dict[str, Any]] = []
    for ps in spec.pieces:
        ports = {
            name: {"dx": float(p.dx), "dy": float(p.dy), "dtheta": float(p.dtheta)}
            for name, p in ps.ports.items()
        }
        routes = {name: list(seq) for name, seq in ps.routes.items()}
        pieces.append({
            "id": ps.piece_id,
            "kind": ps.kind,
            "hand": ps.hand,
            "manufacturer": ps.manufacturer,
            "length_studs": ps.length_studs,
            "radius_studs": ps.radius_studs,
            "sector_angle_rad": ps.sector_angle_rad,
            "body_length_studs": ps.body_length_studs,
            "diverging_radius_studs": ps.diverging_radius_studs,
            "ports": ports,
            "routes": routes,
        })

    return {
        "stud_mm": float(spec.meta.stud_mm),
        "atomic_angle_rad": float(spec.meta.atomic_angle_rad),
        "pieces": pieces,
    }


def port_graph_to_json(
    graph: PortGraph,
    catalog: TrackCatalog,
    config: OptimizationConfig,
) -> Dict[str, Any]:
    """Serialize a decoded PortGraph to a render-ready layout dict.

    Slot poses come from ``graph.slot_poses`` directly (studs + radians at
    port A). Boundaries are emitted in both studs and mm so the JS renderer
    can pick whichever it prefers; the scale shown to users is mm.
    """
    stud_mm = float(catalog.stud_mm)
    b = config.boundary

    spec = catalog.spec
    pieces: List[Dict[str, Any]] = []
    for slot_idx in sorted(graph.slot_pieces.keys()):
        if slot_idx not in graph.slot_poses:
            continue
        x, y, theta = graph.slot_poses[slot_idx]
        piece_id = graph.slot_pieces[slot_idx]
        # Per-route cycle id (None when route's port set isn't on a cycle).
        piece_spec = spec.by_id.get(piece_id) if spec else None
        if piece_spec is not None:
            branch_labels = {
                route_name: graph.branch_labels.get((slot_idx, route_name))
                for route_name in piece_spec.routes
            }
        else:
            branch_labels = {}
        pieces.append({
            "slot": int(slot_idx),
            "piece_id": piece_id,
            "pose_studs": {"x": float(x), "y": float(y), "theta": float(theta)},
            "flip": int(graph.slot_flips.get(slot_idx, 0)),
            "rotate": int(graph.slot_rotates.get(slot_idx, 0)),
            "branch_labels": branch_labels,
        })

    edges = [
        {
            "slot_a": int(e.slot_a), "port_a": e.port_a,
            "slot_b": int(e.slot_b), "port_b": e.port_b,
        }
        for e in graph.edges
    ]

    n_branches = (
        max(graph.branch_labels.values()) + 1 if graph.branch_labels else 0
    )

    boundary_studs = {
        "min_x": float(b.min_x), "max_x": float(b.max_x),
        "min_y": float(b.min_y), "max_y": float(b.max_y),
        "width": float(b.width), "height": float(b.height),
    }
    boundary_mm = {k: v * stud_mm for k, v in boundary_studs.items()}

    return {
        "stud_mm": stud_mm,
        "boundary_studs": boundary_studs,
        "boundary_mm": boundary_mm,
        "pieces": pieces,
        "edges": edges,
        "stats": {
            "n_slots": int(graph.n_slots),
            "n_edges": int(graph.n_edges),
            "n_components": int(graph.n_components),
            "n_cycles": int(graph.n_cycles),
            "n_loose_ports": int(graph.n_loose_ports),
            "n_branches": int(n_branches),
            "max_closure_pos_studs": float(graph.max_closure_position),
            "max_closure_angle_deg": float(graph.max_closure_angle_deg),
        },
    }
