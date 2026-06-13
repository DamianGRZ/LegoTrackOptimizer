"""Decoder-internal dataclasses and helper types.

These types are produced and consumed by the construction decoder.
They are separated from construction logic so that test code, the
problem class, and (potentially) alternative decoders can import
types without pulling in the full construction algorithm.
"""

from dataclasses import dataclass
from typing import Dict, List

from src.catalog import TrackCatalog
from src.templates import PassingSidingTemplate


# =============================================================================
# Decoder Configuration
# =============================================================================

@dataclass
class DecoderConfig:
    """Configuration for the partitioned decoder."""

    position_tolerance: float = 8.0    # studs — closure tolerance
    angle_tolerance: float = 15.0      # degrees — closure tolerance
    siding_position_tolerance: float = 8.0  # studs — siding alignment
    siding_angle_tolerance: float = 10.0    # degrees — siding alignment
    crossing_angle_tolerance: float = 15.0  # degrees — ~90° crossing detection
    boundary_min_x: float = -100.0
    boundary_max_x: float = 100.0
    boundary_min_y: float = -100.0
    boundary_max_y: float = 100.0


# =============================================================================
# Inventory Tracker
# =============================================================================

class InventoryTracker:
    """Tracks piece usage against available inventory.

    Centralizes all inventory bookkeeping so the decoder never
    exceeds physical piece counts.
    """

    def __init__(self, inventory: Dict[str, int], catalog: TrackCatalog) -> None:
        self._available: Dict[int, int] = catalog.inventory_by_index(inventory)
        self._used: Dict[int, int] = {}

    def remaining(self, piece_idx: int) -> int:
        """Pieces remaining for a given type."""
        return self._available.get(piece_idx, 0) - self._used.get(piece_idx, 0)

    def can_use(self, piece_idx: int, count: int = 1) -> bool:
        """Check if count pieces are available."""
        return self.remaining(piece_idx) >= count

    def use(self, piece_idx: int, count: int = 1) -> None:
        """Consume pieces from inventory."""
        self._used[piece_idx] = self._used.get(piece_idx, 0) + count

    def release(self, piece_idx: int, count: int = 1) -> None:
        """Return pieces to inventory."""
        self._used[piece_idx] = max(0, self._used.get(piece_idx, 0) - count)

    def can_use_batch(self, requirements: Dict[int, int]) -> bool:
        """Check if all pieces in a requirements dict are available."""
        for idx, needed in requirements.items():
            if self.remaining(idx) < needed:
                return False
        return True

    def use_batch(self, requirements: Dict[int, int]) -> None:
        """Consume a batch of pieces."""
        for idx, count in requirements.items():
            self.use(idx, count)

    @property
    def used(self) -> Dict[int, int]:
        """Current usage snapshot."""
        return dict(self._used)

    @property
    def available(self) -> Dict[int, int]:
        """Available inventory snapshot."""
        return dict(self._available)


# =============================================================================
# Validated Junction
# =============================================================================

@dataclass
class ValidatedJunction:
    """A junction that has passed inventory and position checks."""

    slot: int
    position: int           # Position in main loop (clamped to valid range)
    handedness: int          # 0-3, maps to TEMPLATES
    n_straights: int
    template: PassingSidingTemplate
    branch_pieces: List[int]
    branch_flips: List[int]
    siding_requirements: Dict[int, int]
