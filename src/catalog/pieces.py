"""Track piece data types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FKDeltas:
    """Forward kinematics deltas for a single piece."""

    dx: float
    dy: float
    dtheta: float

    def to_array(self) -> NDArray[np.float64]:
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
    piece_type: str  # 'straight', 'curve', 'crossing', 'switch'
    fk: FKDeltas
    ports: Tuple[Port, ...]
    index: int
    length: float = 16.0
    radius: Optional[float] = None
    angle: Optional[float] = None
    direction: Optional[str] = None  # 'left' or 'right'
    radius_mm: Optional[float] = None
    routes_data: Optional[List[Dict[str, Any]]] = None

    @property
    def is_straight(self) -> bool:
        return self.piece_type == "straight"

    @property
    def is_curve(self) -> bool:
        return self.piece_type == "curve"

    @property
    def arc_length(self) -> float:
        """Arc length in studs."""
        if self.is_curve and self.radius and self.angle:
            return self.radius * math.radians(abs(self.angle))
        return self.length
