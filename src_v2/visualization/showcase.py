"""Reference layout: one of each catalog piece, laid out in a 2×5 grid.

Used by ``GET /api/showcase`` so the canvas renderer can be eyeballed against
every kind (straight / curve / switch / crossing / double-crossover) without
running an optimization. The layout is purely visual — no edges, no closure
constraint, no inventory check. Each piece is anchored at port A facing +x at
its grid cell origin.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..catalog import TrackCatalog


# Display order. Each entry is (piece_id, flip_bit, rotate_bit). Missing
# piece_ids in the catalog are silently skipped so this survives partial
# catalogs. Variants demoed:
#   R40_CURVE appears as flip=0 (left turn) and flip=1 (right turn).
#   R40_SWITCH_LEFT/RIGHT appear as rotate=0 (IN, frog at far end) and
#   rotate=1 (OUT, frog at throat-end). This is the IN/OUT pair from one
#   physical SKU (LEGO sells 2 switch parts: left + right; IN/OUT is a
#   placement orientation, not a separate part).
SHOWCASE_ORDER: tuple[tuple[str, int, int], ...] = (
    ("STRAIGHT_16",       0, 0),
    ("STRAIGHT_24",       0, 0),
    ("R40_CURVE",         0, 0),     # left turn (canonical)
    ("R40_CURVE",         1, 0),     # right turn (flipped)
    ("CROSS_90",          0, 0),
    ("R40_SWITCH_LEFT",   0, 0),     # IN
    ("R40_SWITCH_LEFT",   0, 1),     # OUT (rotated 180°)
    ("R40_SWITCH_RIGHT",  0, 0),     # IN
    ("R40_SWITCH_RIGHT",  0, 1),     # OUT (rotated 180°)
    ("DOUBLE_CROSSOVER",  0, 0),
)

# Cell dimensions in studs. Generous enough to contain every piece's footprint
# (DOUBLE_CROSSOVER is the biggest at 48×16 studs).
CELL_W_STUDS: float = 70.0
CELL_H_STUDS: float = 50.0
COLS: int = 5
MARGIN_STUDS: float = 20.0


def build_showcase_layout(catalog: TrackCatalog) -> Dict[str, Any]:
    """Return a layout-JSON-compatible dict with one of each piece in a grid.

    Coordinates use the same convention as ``port_graph_to_json``:
    studs internally with theta in radians; mm derived via ``catalog.stud_mm``.
    """
    stud_mm = float(catalog.stud_mm)
    spec = catalog.spec
    available = set(p.piece_id for p in spec.pieces) if spec else set()

    pieces: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []
    slot = 0
    for piece_id, flip, rotate in SHOWCASE_ORDER:
        if piece_id not in available:
            continue
        col = slot % COLS
        row = slot // COLS
        x = MARGIN_STUDS + col * CELL_W_STUDS
        y = MARGIN_STUDS + row * CELL_H_STUDS
        pieces.append({
            "slot": slot,
            "piece_id": piece_id,
            "pose_studs": {"x": float(x), "y": float(y), "theta": 0.0},
            "flip": int(flip),
            "rotate": int(rotate),
        })
        suffix = ""
        if flip and rotate:
            suffix = " (flip+rot)"
        elif flip:
            suffix = " (flipped)"
        elif rotate:
            suffix = " (rotated)"
        labels.append({
            "x_studs": float(x),
            "y_studs": float(y - 6.0),
            "text": f"{piece_id}{suffix}",
        })
        slot += 1

    n_rows = (slot + COLS - 1) // COLS
    width_studs = MARGIN_STUDS * 2 + COLS * CELL_W_STUDS
    height_studs = MARGIN_STUDS * 2 + n_rows * CELL_H_STUDS

    boundary_studs = {
        "min_x": 0.0, "max_x": width_studs,
        "min_y": 0.0, "max_y": height_studs,
        "width": width_studs, "height": height_studs,
    }
    boundary_mm = {k: v * stud_mm for k, v in boundary_studs.items()}

    return {
        "stud_mm": stud_mm,
        "boundary_studs": boundary_studs,
        "boundary_mm": boundary_mm,
        "pieces": pieces,
        "edges": [],
        "labels": labels,
        "stats": {
            "n_slots": slot,
            "n_edges": 0,
            "n_components": slot,
            "n_cycles": 0,
            "n_loose_ports": 0,
            "max_closure_pos_studs": 0.0,
            "max_closure_angle_deg": 0.0,
        },
        "is_showcase": True,
    }
