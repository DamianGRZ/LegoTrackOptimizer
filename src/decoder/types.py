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

    # How much track may be missing at a joint for it to still count as joined.
    # Checked where a siding returns to its OUT switch, and where a double
    # crossover's two slots must land on one physical piece.
    siding_position_tolerance: float = 8.0  # studs
    siding_angle_tolerance: float = 10.0    # degrees
    boundary_min_x: float = -100.0
    boundary_max_x: float = 100.0
    boundary_min_y: float = -100.0
    boundary_max_y: float = 100.0

    @classmethod
    def from_optimization_config(cls, config) -> "DecoderConfig":
        """Decoder geometry mirroring what evaluation uses.

        Every decode outside the evaluation pipeline (renderers, reports,
        run summaries) must build its config here: the default boundary
        (±100) is NOT the configured one, so a default-constructed config
        auto-centers layouts differently than the constraints judged them.

        ``config`` is an ``OptimizationConfig`` (duck-typed to avoid a
        circular import): needs a ``boundary`` with min/max x/y and the
        closure tolerances. A branch joint and a loop seam are the same
        physical gap, so both read the one tolerance the config states.
        """
        return cls(
            siding_position_tolerance=config.closure_tolerance,
            siding_angle_tolerance=config.angle_tolerance,
            boundary_min_x=config.boundary.min_x,
            boundary_max_x=config.boundary.max_x,
            boundary_min_y=config.boundary.min_y,
            boundary_max_y=config.boundary.max_y,
        )


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
        return all(self.remaining(idx) >= needed for idx, needed in requirements.items())

    def use_batch(self, requirements: Dict[int, int]) -> None:
        """Consume a batch of pieces."""
        for idx, count in requirements.items():
            self.use(idx, count)

    @property
    def used(self) -> Dict[int, int]:
        """Current usage snapshot."""
        return dict(self._used)


# =============================================================================
# Validated Junction
# =============================================================================

@dataclass
class ValidatedJunction:
    """A junction that has passed inventory and position checks."""

    slot: int
    position: int           # Position in main loop (clamped to valid range)
    handedness: int          # 0=LEFT, 1=RIGHT, maps to TEMPLATES
    n_straights: int
    template: PassingSidingTemplate
    branch_pieces: List[int]
    branch_flips: List[int]
    siding_requirements: Dict[int, int]
