"""NEAT-style innovation tracking for topology evolution.

Tracks when structural innovations (new switch pairs, branches, crossings)
are first discovered in the population. Innovation numbers enable proper
crossover alignment between topologically different chromosomes.

Based on Stanley & Miikkulainen's NEAT (2002) innovation number concept.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Innovation:
    """Record of a structural innovation in the population.

    Attributes:
        innovation_id: Unique identifier assigned when first discovered.
        innovation_type: Category of structural change (switch_pair, branch, crossing).
        created_generation: Generation when this innovation first appeared.
        parameters: Configuration details of the innovation.
    """
    innovation_id: int
    innovation_type: str
    created_generation: int
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return self.innovation_id

    def __eq__(self, other) -> bool:
        if isinstance(other, Innovation):
            return self.innovation_id == other.innovation_id
        return False


class InnovationTracker:
    """Global innovation number assignment for NEAT-style crossover.

    Ensures that structurally equivalent innovations get the same ID,
    enabling meaningful crossover between chromosomes with different topologies.

    Example:
        tracker = InnovationTracker()

        # First switch pair in position 10 with LEFT handedness
        id1 = tracker.get_or_create("switch_pair", (10, 0, 2))  # Returns 0

        # Same configuration later gets same ID
        id2 = tracker.get_or_create("switch_pair", (10, 0, 2))  # Returns 0

        # Different configuration gets new ID
        id3 = tracker.get_or_create("switch_pair", (15, 1, 3))  # Returns 1
    """

    def __init__(self):
        """Initialize innovation tracker."""
        self._next_id: int = 0
        self._innovations: Dict[Tuple[str, Tuple], Innovation] = {}
        self._current_generation: int = 0

    def set_generation(self, generation: int) -> None:
        """Update current generation for new innovation records.

        Args:
            generation: Current generation number.
        """
        self._current_generation = generation

    def get_or_create(
        self,
        innovation_type: str,
        params: Tuple,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Get existing innovation ID or create new one.

        Args:
            innovation_type: Type of innovation (switch_pair, branch, crossing).
            params: Tuple of parameters defining the innovation.
            extra_data: Optional additional data to store with new innovations.

        Returns:
            Innovation ID (existing or newly assigned).
        """
        key = (innovation_type, params)

        if key not in self._innovations:
            innovation = Innovation(
                innovation_id=self._next_id,
                innovation_type=innovation_type,
                created_generation=self._current_generation,
                parameters=dict(extra_data) if extra_data else {},
            )
            self._innovations[key] = innovation
            self._next_id += 1

        return self._innovations[key].innovation_id

    def get_innovation(self, innovation_type: str, params: Tuple) -> Optional[Innovation]:
        """Get innovation record if it exists.

        Args:
            innovation_type: Type of innovation.
            params: Parameters defining the innovation.

        Returns:
            Innovation record or None if not found.
        """
        key = (innovation_type, params)
        return self._innovations.get(key)

    def get_by_id(self, innovation_id: int) -> Optional[Innovation]:
        """Get innovation record by ID.

        Args:
            innovation_id: Innovation identifier.

        Returns:
            Innovation record or None if not found.
        """
        for innovation in self._innovations.values():
            if innovation.innovation_id == innovation_id:
                return innovation
        return None

    @property
    def total_innovations(self) -> int:
        """Total number of innovations discovered."""
        return len(self._innovations)

    def get_innovations_by_type(self, innovation_type: str) -> Dict[Tuple, Innovation]:
        """Get all innovations of a specific type.

        Args:
            innovation_type: Type to filter by.

        Returns:
            Dictionary mapping parameters to innovation records.
        """
        return {
            params: inn
            for (itype, params), inn in self._innovations.items()
            if itype == innovation_type
        }

    def reset(self) -> None:
        """Reset all tracked innovations."""
        self._next_id = 0
        self._innovations.clear()
        self._current_generation = 0


def extract_innovations_from_chromosome(
    chromosome: NDArray,
    tracker: InnovationTracker,
) -> Dict[int, Innovation]:
    """Extract innovation records from a chromosome.

    Analyzes the chromosome's branch slots and identifies which
    innovations are active.

    Args:
        chromosome: RK chromosome array.
        tracker: Innovation tracker for ID lookup/assignment.

    Returns:
        Dictionary mapping slot indices to innovation records.
    """
    from .encoding import (
        B_MAX,
        BRANCH_START,
        B_SLOT,
        get_branch_template_params,
        is_branch_active,
    )

    innovations = {}

    for slot_idx in range(B_MAX):
        if is_branch_active(chromosome, slot_idx):
            in_pos, handedness, n_straights, _ = get_branch_template_params(
                chromosome, slot_idx
            )

            # Create innovation key from branch parameters
            params = (in_pos, handedness, n_straights)
            inn_id = tracker.get_or_create("branch", params)
            innovation = tracker.get_by_id(inn_id)

            if innovation:
                innovations[slot_idx] = innovation

    return innovations


def count_active_innovations(chromosome: NDArray) -> int:
    """Count active structural innovations in a chromosome.

    Args:
        chromosome: RK chromosome array.

    Returns:
        Number of active branch slots.
    """
    from .encoding import B_MAX, is_branch_active

    return sum(
        1 for slot_idx in range(B_MAX)
        if is_branch_active(chromosome, slot_idx)
    )


def compute_innovation_distance(
    x1: NDArray,
    x2: NDArray,
    tracker: InnovationTracker,
) -> Tuple[int, int, float]:
    """Compute structural distance between two chromosomes.

    Based on NEAT's compatibility distance:
    - Disjoint genes: innovations in one but not the other
    - Excess genes: innovations beyond the range of the other
    - Weight differences: RK value differences for matching genes

    Args:
        x1: First chromosome.
        x2: Second chromosome.
        tracker: Innovation tracker.

    Returns:
        Tuple of (disjoint_count, excess_count, avg_weight_diff).
    """
    inn1 = extract_innovations_from_chromosome(x1, tracker)
    inn2 = extract_innovations_from_chromosome(x2, tracker)

    ids1 = {inn.innovation_id for inn in inn1.values()}
    ids2 = {inn.innovation_id for inn in inn2.values()}

    # Find max innovation ID in each
    max1 = max(ids1) if ids1 else -1
    max2 = max(ids2) if ids2 else -1

    # Matching, disjoint, and excess
    matching = ids1 & ids2
    all_ids = ids1 | ids2

    if max1 < max2:
        excess = {i for i in ids2 if i > max1}
        disjoint = all_ids - matching - excess
    elif max2 < max1:
        excess = {i for i in ids1 if i > max2}
        disjoint = all_ids - matching - excess
    else:
        excess = set()
        disjoint = all_ids - matching

    # Weight difference for main loop genes
    from .encoding import MAIN_LOOP_START, MAIN_LOOP_END

    ml1 = x1[MAIN_LOOP_START:MAIN_LOOP_END]
    ml2 = x2[MAIN_LOOP_START:MAIN_LOOP_END]
    avg_weight_diff = float(np.mean(np.abs(ml1 - ml2)))

    return len(disjoint), len(excess), avg_weight_diff


# Global singleton for convenience (optional)
_global_tracker: Optional[InnovationTracker] = None


def get_global_tracker() -> InnovationTracker:
    """Get or create global innovation tracker singleton."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = InnovationTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global innovation tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()
