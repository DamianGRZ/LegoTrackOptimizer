"""Genetic operators for multi-segment track layout optimization.

Mutation operators:
- MUTATE (p=0.30): Swap/replace a random active gene
- ADD (p=0.20): Insert a new piece, shift downstream right
- DELETE (p=0.10): Remove a piece, shift left
- BRANCH (p=0.20): Branch topology operations (stub)
- COMPOUND (p=0.20): Segment replacement + local search (stub)

Crossover:
- NoOpCrossover: Identity operator returning parents unchanged (Phase 1)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

from .encoding import (
    B_MAX,
    B_SLOT,
    BRANCH_SLOTS_END,
    BRANCH_SLOTS_START,
    INACTIVE,
    L_MAX,
    MAIN_LOOP_END,
    MAIN_LOOP_START,
    N_VAR,
    RK_INACTIVE,
    RK_INACTIVE_THRESHOLD,
    STRAIGHT_16,
    SWITCH_INDICES,
    SWITCH_LEFT_IN,
    SWITCH_LEFT_OUT,
    SWITCH_MASK_END,
    SWITCH_MASK_START,
    SWITCH_PAIRS,
    SWITCH_RIGHT_IN,
    SWITCH_RIGHT_OUT,
    find_active_branch_slots,
    find_inactive_branch_slots,
    get_branch_pieces,
    get_branch_slot,
    get_branch_src_switch,
    get_branch_template_params,
    get_main_loop,
    get_start_position,
    get_switch_mask_value,
    is_branch_active,
    piece_index_to_rk,
    rk_to_piece_index,
    set_branch_slot,
    set_branch_template_params,
    set_switch_mask_value,
)


# =============================================================================
# Operator Probabilities
# =============================================================================

@dataclass
class OperatorProbabilities:
    """Probabilities for each mutation operator."""
    mutate: float = 0.30   # Swap/replace
    add: float = 0.20      # Insert piece
    delete: float = 0.10   # Remove piece
    branch: float = 0.20   # Branch operations
    compound: float = 0.20 # Segment replacement + local search

    def __post_init__(self):
        total = self.mutate + self.add + self.delete + self.branch + self.compound
        if abs(total - 1.0) > 0.01:
            # Normalize probabilities
            self.mutate /= total
            self.add /= total
            self.delete /= total
            self.branch /= total
            self.compound /= total

    def select_operator(self) -> str:
        """Randomly select an operator based on probabilities."""
        r = np.random.random()
        cumsum = 0.0
        for name, prob in [
            ('mutate', self.mutate),
            ('add', self.add),
            ('delete', self.delete),
            ('branch', self.branch),
            ('compound', self.compound),
        ]:
            cumsum += prob
            if r < cumsum:
                return name
        return 'mutate'  # Fallback


@dataclass
class BranchSubOperatorProbabilities:
    """Probabilities for BRANCH sub-operators."""
    add_branch: float = 0.35     # ADD_BRANCH (p=0.07 of 0.20 total)
    extend_branch: float = 0.30  # EXTEND_BRANCH (p=0.06 of 0.20)
    shorten_branch: float = 0.20 # SHORTEN_BRANCH (p=0.04 of 0.20)
    remove_branch: float = 0.15  # REMOVE_BRANCH (p=0.03 of 0.20)

    def select_sub_operator(self) -> str:
        """Randomly select a branch sub-operator."""
        r = np.random.random()
        if r < self.add_branch:
            return 'add_branch'
        elif r < self.add_branch + self.extend_branch:
            return 'extend_branch'
        elif r < self.add_branch + self.extend_branch + self.shorten_branch:
            return 'shorten_branch'
        else:
            return 'remove_branch'


@dataclass
class CompoundSubOperatorProbabilities:
    """Probabilities for COMPOUND sub-operators."""
    segment_replace: float = 0.60  # Segment replacement (p=0.12 of 0.20)
    local_search: float = 0.40     # Local search (p=0.08 of 0.20)

    def select_sub_operator(self) -> str:
        """Randomly select a compound sub-operator."""
        r = np.random.random()
        if r < self.segment_replace:
            return 'segment_replace'
        else:
            return 'local_search'


# =============================================================================
# Track Layout Mutation
# =============================================================================

# Switch piece indices (from track_pieces.yaml)
SWITCH_INDICES = {5, 6, 7, 8}  # LEFT_IN, LEFT_OUT, RIGHT_IN, RIGHT_OUT
STRAIGHT_16_INDEX = 0  # Piece that switches replace


class TrackLayoutMutation(Mutation):
    """Five-operator mutation suite for track layout chromosomes.

    Operators:
    - MUTATE: Swap/replace a random active gene
    - ADD: Insert a new piece
    - DELETE: Remove a piece
    - BRANCH: Branch topology operations (4 sub-operators)
    - COMPOUND: Segment replacement + local search (2 sub-operators)
    """

    def __init__(
        self,
        max_piece_index: int,
        inventory_by_index: Optional[Dict[int, int]] = None,
        probabilities: Optional[OperatorProbabilities] = None,
        branch_probabilities: Optional[BranchSubOperatorProbabilities] = None,
        compound_probabilities: Optional[CompoundSubOperatorProbabilities] = None,
        mutation_prob: float = 1.0,  # Per-individual mutation probability
    ):
        """Initialize mutation operator.

        Args:
            max_piece_index: Maximum valid piece index from catalog.
            inventory_by_index: Available inventory {piece_index: count}.
            probabilities: Operator selection probabilities.
            branch_probabilities: BRANCH sub-operator probabilities.
            compound_probabilities: COMPOUND sub-operator probabilities.
            mutation_prob: Probability of mutating each individual.
        """
        super().__init__()
        self.max_piece_index = max_piece_index
        self.inventory_by_index = inventory_by_index or {}
        self.probabilities = probabilities or OperatorProbabilities()
        self.branch_probs = branch_probabilities or BranchSubOperatorProbabilities()
        self.compound_probs = compound_probabilities or CompoundSubOperatorProbabilities()
        self.mutation_prob = mutation_prob

        # Build sorted list of available piece indices for RK encoding/decoding
        self.available_pieces = sorted(self.inventory_by_index.keys())
        if not self.available_pieces:
            self.available_pieces = sorted(range(max_piece_index + 1))

        # Alias for backward compatibility
        self.valid_pieces = self.available_pieces

        # Build list of switch indices available in inventory
        self.valid_switches = [idx for idx in SWITCH_INDICES if idx in self.inventory_by_index]

        # Build list of non-switch pieces (for branch construction)
        self.valid_non_switch_pieces = [
            idx for idx in self.available_pieces if idx not in SWITCH_INDICES
        ]

    def _do(self, problem, X, **kwargs) -> NDArray:
        """Apply mutation to population.

        Args:
            problem: Optimization problem.
            X: Population array of shape (n_pop, n_var).
            **kwargs: Additional arguments.

        Returns:
            Mutated population array.
        """
        X_mutated = X.copy()

        for i in range(len(X)):
            if np.random.random() < self.mutation_prob:
                X_mutated[i] = self._mutate_individual(X[i])

        return X_mutated

    def _mutate_individual(self, x: NDArray) -> NDArray:
        """Apply mutation to single chromosome.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        x = x.copy()
        operator = self.probabilities.select_operator()

        if operator == 'mutate':
            x = self._op_mutate(x)
        elif operator == 'add':
            x = self._op_add(x)
        elif operator == 'delete':
            x = self._op_delete(x)
        elif operator == 'branch':
            x = self._op_branch(x)
        elif operator == 'compound':
            x = self._op_compound(x)

        return x

    # =========================================================================
    # Operator 1: MUTATE (swap/replace)
    # =========================================================================

    def _op_mutate(self, x: NDArray) -> NDArray:
        """MUTATE: Replace a random active gene with another valid piece.

        Uses RK encoding: decodes current piece, selects a different one,
        re-encodes as RK value in [0, 1].

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END].copy()

        # Find active positions (RK value above inactive threshold)
        active_positions = np.where(main_loop >= RK_INACTIVE_THRESHOLD)[0]
        if len(active_positions) == 0:
            return x

        # Select random active position
        pos = np.random.choice(active_positions)

        # Decode current piece and select a different one
        current_piece = rk_to_piece_index(float(main_loop[pos]), self.available_pieces)
        candidates = [p for p in self.available_pieces if p != current_piece]
        if not candidates:
            candidates = self.available_pieces

        if candidates:
            new_piece = np.random.choice(candidates)
            main_loop[pos] = piece_index_to_rk(new_piece, self.available_pieces)

        x[MAIN_LOOP_START:MAIN_LOOP_END] = main_loop
        return x

    # =========================================================================
    # Operator 2: ADD (insertion)
    # =========================================================================

    def _op_add(self, x: NDArray) -> NDArray:
        """ADD: Insert a new piece, shifting downstream genes right.

        Position selection:
        - 40% near end (append-like)
        - 30% random position
        - 30% at first inactive slot

        Uses RK encoding for piece values.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END].copy()

        # Find active and inactive positions using RK threshold
        active_mask = main_loop >= RK_INACTIVE_THRESHOLD
        n_active = int(np.sum(active_mask))

        # If main loop is full, can't add
        if n_active >= L_MAX:
            return x

        # Select insertion position
        r = np.random.random()
        if r < 0.4:
            # Near end - after last active position
            if n_active > 0:
                last_active = np.max(np.where(active_mask)[0])
                pos = min(last_active + 1, L_MAX - 1)
            else:
                pos = 0
        elif r < 0.7:
            # Random position among active region
            pos = np.random.randint(0, max(1, n_active + 1))
        else:
            # First inactive slot
            inactive_positions = np.where(~active_mask)[0]
            if len(inactive_positions) > 0:
                pos = inactive_positions[0]
            else:
                return x  # No inactive slots

        # Select piece and encode as RK value
        new_piece = np.random.choice(self.available_pieces)
        new_rk = piece_index_to_rk(new_piece, self.available_pieces)

        # Shift genes right and insert
        if pos < L_MAX - 1:
            main_loop[pos + 1:] = main_loop[pos:-1]
        main_loop[pos] = new_rk

        x[MAIN_LOOP_START:MAIN_LOOP_END] = main_loop
        return x

    # =========================================================================
    # Operator 3: DELETE (removal)
    # =========================================================================

    def _op_delete(self, x: NDArray) -> NDArray:
        """DELETE: Remove a piece, shifting downstream genes left.

        Position selection:
        - 50% random active position
        - 30% last active position
        - 20% first active position

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END].copy()

        # Find active positions using RK threshold
        active_positions = np.where(main_loop >= RK_INACTIVE_THRESHOLD)[0]
        if len(active_positions) == 0:
            return x

        # Select position to delete
        r = np.random.random()
        if r < 0.5:
            pos = np.random.choice(active_positions)
        elif r < 0.8:
            pos = active_positions[-1]
        else:
            pos = active_positions[0]

        # Shift genes left and fill last with RK inactive
        main_loop[pos:-1] = main_loop[pos + 1:]
        main_loop[-1] = RK_INACTIVE

        x[MAIN_LOOP_START:MAIN_LOOP_END] = main_loop
        return x

    # =========================================================================
    # Operator 4: BRANCH (4 sub-operators)
    # =========================================================================

    def _op_branch(self, x: NDArray) -> NDArray:
        """BRANCH: Branch topology operations.

        Sub-operators:
        - ADD_BRANCH: Convert 2-port piece to switch, init minimal branch
        - EXTEND_BRANCH: Add 1-2 pieces to existing branch
        - SHORTEN_BRANCH: Remove 1-2 pieces from branch
        - REMOVE_BRANCH: Delete branch, revert switch to 2-port

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        sub_op = self.branch_probs.select_sub_operator()

        if sub_op == 'add_branch':
            return self._branch_add(x)
        elif sub_op == 'extend_branch':
            return self._branch_extend(x)
        elif sub_op == 'shorten_branch':
            return self._branch_shorten(x)
        else:  # remove_branch
            return self._branch_remove(x)

    def _branch_add(self, x: NDArray) -> NDArray:
        """ADD_BRANCH: Activate a branch slot with template-based encoding.

        Template-based encoding: [IN_pos, handedness, n_straights, active]
        - IN_pos: Main loop position where branch diverges
        - handedness: 0=LEFT, 1=RIGHT (determines switch types and curve directions)
        - n_straights: Number of straight pieces in parallel section (0-8)
        - active: 1 to enable branch

        The decoder will:
        1. Inject IN/OUT switches at computed positions
        2. Generate branch pieces from template (approach + straights + return)

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        # Check if we have available switch pieces
        if not self.valid_switches:
            return x

        # Find inactive branch slots
        inactive_slots = find_inactive_branch_slots(x)
        if not inactive_slots:
            return x  # All branch slots are used

        # Find valid positions in main loop (need room for OUT switch after IN)
        main_loop = get_main_loop(x)
        n_active = int(np.sum(main_loop >= RK_INACTIVE_THRESHOLD))

        if n_active < 8:
            return x  # Need minimum pieces for a valid siding

        # Select random IN position (not too close to end - need room for OUT)
        max_in_pos = max(0, n_active - 4)  # Leave room for OUT switch
        if max_in_pos <= 0:
            return x

        in_pos = np.random.randint(0, max_in_pos)

        # Select handedness (0=LEFT, 1=RIGHT)
        handedness = np.random.randint(0, 2)

        # Select number of straights (start small, 1-4)
        n_straights = np.random.randint(1, 5)

        # Use first available slot
        slot_idx = inactive_slots[0]

        # Set template parameters using the correct encoding
        set_branch_template_params(x, slot_idx, in_pos, handedness, n_straights, active=1)

        return x

    def _branch_extend(self, x: NDArray) -> NDArray:
        """EXTEND_BRANCH: Add straights to an existing branch.

        In template-based encoding, this increases n_straights for a branch slot.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        active_slots = find_active_branch_slots(x)
        if not active_slots:
            return x

        # Select random active branch
        slot_idx = np.random.choice(active_slots)
        in_pos, handedness, n_straights, active = get_branch_template_params(x, slot_idx)

        # Increase n_straights (max 8)
        if n_straights < 8:
            new_n_straights = min(8, n_straights + np.random.randint(1, 3))
            set_branch_template_params(x, slot_idx, in_pos, handedness, new_n_straights, active=1)

        return x

    def _branch_shorten(self, x: NDArray) -> NDArray:
        """SHORTEN_BRANCH: Remove straights from an existing branch.

        In template-based encoding, this decreases n_straights for a branch slot.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        active_slots = find_active_branch_slots(x)
        if not active_slots:
            return x

        # Select random active branch
        slot_idx = np.random.choice(active_slots)
        in_pos, handedness, n_straights, active = get_branch_template_params(x, slot_idx)

        # Decrease n_straights (min 0)
        if n_straights > 0:
            n_to_remove = min(np.random.randint(1, 3), n_straights)
            new_n_straights = max(0, n_straights - n_to_remove)
            set_branch_template_params(x, slot_idx, in_pos, handedness, new_n_straights, active=1)

        return x

    def _branch_remove(self, x: NDArray) -> NDArray:
        """REMOVE_BRANCH: Delete entire branch by deactivating a branch slot.

        In template-based encoding, this sets active=0 for the branch slot.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        active_slots = find_active_branch_slots(x)
        if not active_slots:
            return x

        # Select random active branch
        slot_idx = np.random.choice(active_slots)

        # Deactivate the branch slot by setting active=0
        # Keep other params so they can potentially be reused
        in_pos, handedness, n_straights, _ = get_branch_template_params(x, slot_idx)
        set_branch_template_params(x, slot_idx, in_pos, handedness, n_straights, active=0)

        return x

    # =========================================================================
    # Operator 5: COMPOUND (2 sub-operators)
    # =========================================================================

    def _op_compound(self, x: NDArray) -> NDArray:
        """COMPOUND: Segment replacement and local search.

        Sub-operators:
        - Segment replacement: Remove N consecutive pieces, replace with M new
        - Local search: Hill-climb on 3-5 piece subsection

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        sub_op = self.compound_probs.select_sub_operator()

        if sub_op == 'segment_replace':
            return self._compound_segment_replace(x)
        else:  # local_search
            return self._compound_local_search(x)

    def _compound_segment_replace(self, x: NDArray) -> NDArray:
        """Segment replacement: Remove N consecutive pieces, replace with M.

        ALNS-style destroy-repair: removes a segment and replaces with
        greedy-selected pieces. Uses RK encoding for new piece values.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END].copy()

        # Find active positions using RK threshold
        active_positions = np.where(main_loop >= RK_INACTIVE_THRESHOLD)[0]
        if len(active_positions) < 5:
            return x

        # Select segment size (2-5 pieces)
        max_segment = min(5, len(active_positions) - 2)
        if max_segment < 2:
            return x
        segment_size = np.random.randint(2, max_segment + 1)

        # Select start position (not at the very end)
        max_start = len(active_positions) - segment_size
        if max_start <= 0:
            return x
        start_idx = np.random.randint(0, max_start)
        start_pos = active_positions[start_idx]
        end_pos = active_positions[start_idx + segment_size - 1]

        # Clear the segment with RK inactive
        main_loop[start_pos:end_pos + 1] = RK_INACTIVE

        # Replace with new pieces (M = segment_size ± 1), encoded as RK
        n_new = segment_size + np.random.randint(-1, 2)
        n_new = max(1, min(n_new, L_MAX - start_pos))

        for i in range(n_new):
            pos = start_pos + i
            if pos < L_MAX and self.available_pieces:
                new_piece = np.random.choice(self.available_pieces)
                main_loop[pos] = piece_index_to_rk(new_piece, self.available_pieces)

        x[MAIN_LOOP_START:MAIN_LOOP_END] = main_loop
        return x

    def _compound_local_search(self, x: NDArray) -> NDArray:
        """Local search: Apply multiple mutations to a subsection.

        Memetic-style: selects a 3-5 piece subsection and applies
        several mutations. Uses RK encoding for new piece values.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END].copy()

        # Find active positions using RK threshold
        active_positions = np.where(main_loop >= RK_INACTIVE_THRESHOLD)[0]
        if len(active_positions) < 4:
            return x

        # Select subsection size (3-5 pieces, but at most len-1 to leave room)
        max_subsection = min(5, len(active_positions) - 1)
        if max_subsection < 3:
            return x
        subsection_size = np.random.randint(3, max_subsection + 1)

        # Select start position
        max_start = len(active_positions) - subsection_size + 1
        if max_start <= 0:
            return x
        start_idx = np.random.randint(0, max_start)

        # Apply multiple point mutations, encoding as RK values
        for i in range(subsection_size):
            pos = active_positions[start_idx + i]
            if np.random.random() < 0.5 and self.available_pieces:
                new_piece = np.random.choice(self.available_pieces)
                main_loop[pos] = piece_index_to_rk(new_piece, self.available_pieces)

        x[MAIN_LOOP_START:MAIN_LOOP_END] = main_loop
        return x


# =============================================================================
# NoOp Crossover (Phase 1)
# =============================================================================

class NoOpCrossover(Crossover):
    """Identity crossover returning parents unchanged.

    Used in Phase 1 mutation-only evolution.
    Phase 3 will implement segment-selective BRKGA-style crossover.
    """

    def __init__(self):
        """Initialize no-op crossover."""
        super().__init__(n_parents=2, n_offsprings=2)

    def _do(self, problem, X, **kwargs) -> NDArray:
        """Return parents unchanged.

        Args:
            problem: Optimization problem.
            X: Parents array of shape (n_parents, n_matings, n_var).
            **kwargs: Additional arguments.

        Returns:
            Offspring array (copy of parents).
        """
        # X shape: (n_parents=2, n_matings, n_var)
        # Return shape: (n_offsprings=2, n_matings, n_var)
        return X.copy()


# =============================================================================
# Switch-Preserving Crossover
# =============================================================================

class SwitchPreservingCrossover(Crossover):
    """Crossover that preserves switch pair integrity.

    Standard crossover can split switch pairs (IN goes to child, OUT stays
    in parent), creating orphan switches that violate constraints.

    Strategy:
    1. Identify valid cut points that don't split switch pairs
    2. Perform segment swap at valid cut points
    3. Apply switch repair to fix any remaining imbalances

    A cut point is INVALID if:
    - It's immediately after an IN switch (would orphan the IN)
    - It's immediately before an OUT switch (would orphan the OUT)
    """

    def __init__(
        self,
        inventory_by_index: Optional[Dict[int, int]] = None,
        prob: float = 0.9,
    ):
        """Initialize switch-preserving crossover.

        Args:
            inventory_by_index: Available inventory {piece_index: count}.
                Used for switch repair after crossover.
            prob: Crossover probability (default 0.9).
        """
        super().__init__(n_parents=2, n_offsprings=2)
        self.inventory_by_index = inventory_by_index or {}
        self.prob = prob

        # Build sorted available pieces list for RK encoding/decoding
        self.available_pieces = sorted(self.inventory_by_index.keys())
        if not self.available_pieces:
            self.available_pieces = list(range(10))  # Fallback

    def _do(self, problem, X, **kwargs) -> NDArray:
        """Perform switch-preserving crossover on population.

        Args:
            problem: Optimization problem.
            X: Parents array of shape (n_parents, n_matings, n_var).
            **kwargs: Additional arguments.

        Returns:
            Offspring array of shape (n_offsprings, n_matings, n_var).
        """
        n_parents, n_matings, n_var = X.shape
        Y = np.zeros((self.n_offsprings, n_matings, n_var), dtype=X.dtype)

        for k in range(n_matings):
            p1 = X[0, k]
            p2 = X[1, k]

            # Skip crossover with probability (1 - prob)
            if np.random.random() > self.prob:
                Y[0, k] = p1.copy()
                Y[1, k] = p2.copy()
                continue

            # Perform switch-preserving crossover
            c1, c2 = self._cross_pair(p1, p2)
            Y[0, k] = c1
            Y[1, k] = c2

        return Y

    def _cross_pair(self, p1: NDArray, p2: NDArray) -> Tuple[NDArray, NDArray]:
        """Perform crossover on a single pair of parents.

        Args:
            p1: First parent chromosome.
            p2: Second parent chromosome.

        Returns:
            Tuple of two offspring chromosomes.
        """
        c1, c2 = p1.copy(), p2.copy()

        # Get main loops
        ml1 = p1[MAIN_LOOP_START:MAIN_LOOP_END]
        ml2 = p2[MAIN_LOOP_START:MAIN_LOOP_END]

        # Find valid cut points for each parent
        valid1 = self._find_valid_cut_points(ml1)
        valid2 = self._find_valid_cut_points(ml2)

        # Need at least 2 valid cut points (start and one interior)
        if len(valid1) < 2 or len(valid2) < 2:
            # Fall back to simple repair-based crossover
            return self._simple_crossover_with_repair(p1, p2)

        # Select a random valid cut point from each (excluding 0 and end)
        interior1 = valid1[(valid1 > 0) & (valid1 < L_MAX)]
        interior2 = valid2[(valid2 > 0) & (valid2 < L_MAX)]

        if len(interior1) == 0 or len(interior2) == 0:
            return self._simple_crossover_with_repair(p1, p2)

        cut1 = np.random.choice(interior1)
        cut2 = np.random.choice(interior2)

        # Create offspring by swapping segments
        new_ml1 = self._swap_segments(ml1, ml2, cut1, cut2)
        new_ml2 = self._swap_segments(ml2, ml1, cut2, cut1)

        # Update offspring
        c1[MAIN_LOOP_START:MAIN_LOOP_END] = new_ml1
        c2[MAIN_LOOP_START:MAIN_LOOP_END] = new_ml2

        # Apply switch repair to balance pairs
        c1 = self._repair_switches(c1)
        c2 = self._repair_switches(c2)

        return c1, c2

    def _find_valid_cut_points(self, main_loop: NDArray) -> NDArray:
        """Find positions where cutting won't split a switch pair.

        A cut is INVALID if:
        - It's immediately after an IN switch (position i+1 where gene[i] is IN)
        - It's immediately before an OUT switch (position i where gene[i] is OUT)

        Uses RK decoding to identify switch pieces from [0,1] gene values.

        Args:
            main_loop: Main loop gene array (RK encoded).

        Returns:
            Array of valid cut point indices (0 to L_MAX inclusive).
        """
        n = len(main_loop)
        valid = np.ones(n + 1, dtype=bool)  # n+1 possible cut positions

        for i in range(n):
            if main_loop[i] < RK_INACTIVE_THRESHOLD:
                continue

            # Decode RK value to piece index
            piece_idx = rk_to_piece_index(float(main_loop[i]), self.available_pieces)

            # IN switches: can't cut immediately after
            if piece_idx in {SWITCH_LEFT_IN, SWITCH_RIGHT_IN}:
                if i + 1 <= n:
                    valid[i + 1] = False

            # OUT switches: can't cut immediately before
            if piece_idx in {SWITCH_LEFT_OUT, SWITCH_RIGHT_OUT}:
                valid[i] = False

        return np.where(valid)[0]

    def _swap_segments(
        self,
        ml1: NDArray,
        ml2: NDArray,
        cut1: int,
        cut2: int,
    ) -> NDArray:
        """Create new main loop by swapping segments.

        Takes prefix from ml1 (up to cut1) and suffix from ml2 (from cut2).

        Args:
            ml1: First main loop.
            ml2: Second main loop.
            cut1: Cut point in ml1.
            cut2: Cut point in ml2.

        Returns:
            New main loop array of length L_MAX.
        """
        prefix = ml1[:cut1]
        suffix = ml2[cut2:]

        # Concatenate and fit to length
        combined = np.concatenate([prefix, suffix])
        result = np.full(L_MAX, RK_INACTIVE, dtype=ml1.dtype)

        n = min(len(combined), L_MAX)
        result[:n] = combined[:n]

        return result

    def _simple_crossover_with_repair(
        self,
        p1: NDArray,
        p2: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Fallback: simple one-point crossover with repair.

        Args:
            p1: First parent.
            p2: Second parent.

        Returns:
            Two offspring with switch repair applied.
        """
        c1, c2 = p1.copy(), p2.copy()

        ml1 = p1[MAIN_LOOP_START:MAIN_LOOP_END]
        ml2 = p2[MAIN_LOOP_START:MAIN_LOOP_END]

        # Find active lengths using RK threshold
        active1 = int(np.sum(ml1 >= RK_INACTIVE_THRESHOLD))
        active2 = int(np.sum(ml2 >= RK_INACTIVE_THRESHOLD))

        if active1 < 2 or active2 < 2:
            return c1, c2

        # Random cut point in active region
        cut = np.random.randint(1, min(active1, active2))

        # Swap suffixes
        new_ml1 = np.full(L_MAX, RK_INACTIVE, dtype=ml1.dtype)
        new_ml2 = np.full(L_MAX, RK_INACTIVE, dtype=ml2.dtype)

        new_ml1[:cut] = ml1[:cut]
        new_ml1[cut:cut + active2 - cut] = ml2[cut:active2]

        new_ml2[:cut] = ml2[:cut]
        new_ml2[cut:cut + active1 - cut] = ml1[cut:active1]

        c1[MAIN_LOOP_START:MAIN_LOOP_END] = new_ml1
        c2[MAIN_LOOP_START:MAIN_LOOP_END] = new_ml2

        # Apply switch repair
        c1 = self._repair_switches(c1)
        c2 = self._repair_switches(c2)

        return c1, c2

    def _repair_switches(self, x: NDArray) -> NDArray:
        """Repair switch imbalances in chromosome.

        Ensures IN/OUT switches are balanced for both left and right pairs.

        Args:
            x: Chromosome array.

        Returns:
            Repaired chromosome.
        """
        x = x.copy()
        main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END].copy()

        # Count current usage
        current_usage = self._count_usage(main_loop)

        # Repair each switch pair (left and right)
        for in_idx, out_idx in SWITCH_PAIRS:
            main_loop = self._balance_pair(
                main_loop, in_idx, out_idx, current_usage
            )

        x[MAIN_LOOP_START:MAIN_LOOP_END] = main_loop
        return x

    def _count_usage(self, main_loop: NDArray) -> Dict[int, int]:
        """Count piece usage in main loop using RK decoding."""
        usage: Dict[int, int] = {}
        for gene in main_loop:
            if gene < RK_INACTIVE_THRESHOLD:
                continue
            piece_idx = rk_to_piece_index(float(gene), self.available_pieces)
            usage[piece_idx] = usage.get(piece_idx, 0) + 1
        return usage

    def _balance_pair(
        self,
        main_loop: NDArray,
        in_idx: int,
        out_idx: int,
        current_usage: Dict[int, int],
    ) -> NDArray:
        """Balance IN and OUT switch counts for one pair.

        Strategy:
        1. If inventory allows, add missing switch by converting STRAIGHT_16
        2. Otherwise, remove excess switches by converting to STRAIGHT_16

        Uses RK decoding to find piece positions and RK encoding for replacements.

        Args:
            main_loop: Main loop array (RK encoded).
            in_idx: IN switch piece index.
            out_idx: OUT switch piece index.
            current_usage: Current piece usage counts (already RK-decoded).

        Returns:
            Modified main loop.
        """
        # Decode all positions to find switches and straights
        in_positions = []
        out_positions = []
        straight_positions = []

        for i in range(len(main_loop)):
            if main_loop[i] < RK_INACTIVE_THRESHOLD:
                continue
            piece = rk_to_piece_index(float(main_loop[i]), self.available_pieces)
            if piece == in_idx:
                in_positions.append(i)
            elif piece == out_idx:
                out_positions.append(i)
            elif piece == STRAIGHT_16:
                straight_positions.append(i)

        n_in = len(in_positions)
        n_out = len(out_positions)

        if n_in == n_out:
            return main_loop

        deficit_type = in_idx if n_in < n_out else out_idx
        excess_type = out_idx if n_in < n_out else in_idx
        excess_positions = out_positions if n_in < n_out else in_positions
        deficit = abs(n_in - n_out)

        # Try to add missing switches (encoded as RK)
        available = self.inventory_by_index.get(deficit_type, 0) - current_usage.get(deficit_type, 0)

        if available > 0 and len(straight_positions) > 0:
            n_add = min(deficit, available, len(straight_positions))
            for i in range(n_add):
                pos = straight_positions[i]
                main_loop[pos] = piece_index_to_rk(deficit_type, self.available_pieces)
                current_usage[deficit_type] = current_usage.get(deficit_type, 0) + 1
                current_usage[STRAIGHT_16] = current_usage.get(STRAIGHT_16, 1) - 1
            deficit -= n_add

        # Remove excess if still unbalanced (replace with STRAIGHT_16 RK)
        if deficit > 0 and len(excess_positions) > 0:
            straight_rk = piece_index_to_rk(STRAIGHT_16, self.available_pieces)
            n_remove = min(deficit, len(excess_positions))
            for i in range(n_remove):
                pos = excess_positions[-(i + 1)]  # Remove from end
                main_loop[pos] = straight_rk
                current_usage[excess_type] = max(0, current_usage.get(excess_type, 0) - 1)
                current_usage[STRAIGHT_16] = current_usage.get(STRAIGHT_16, 0) + 1

        return main_loop


# =============================================================================
# Helper Functions
# =============================================================================

def create_mutation_operator(
    catalog,
    inventory: Dict[str, int],
    mutation_prob: float = 1.0,
) -> TrackLayoutMutation:
    """Create mutation operator from catalog and inventory.

    Args:
        catalog: Track catalog.
        inventory: Available inventory {piece_id: count}.
        mutation_prob: Probability of mutating each individual.

    Returns:
        Configured TrackLayoutMutation instance.
    """
    # Convert inventory to index-based
    inventory_by_index = {}
    for piece_id, count in inventory.items():
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None:
            inventory_by_index[idx] = count

    return TrackLayoutMutation(
        max_piece_index=catalog._max_index,
        inventory_by_index=inventory_by_index,
        mutation_prob=mutation_prob,
    )


def create_crossover_operator(
    catalog,
    inventory: Dict[str, int],
    prob: float = 0.9,
) -> SwitchPreservingCrossover:
    """Create switch-preserving crossover operator.

    Args:
        catalog: Track catalog.
        inventory: Available inventory {piece_id: count}.
        prob: Crossover probability.

    Returns:
        Configured SwitchPreservingCrossover instance.
    """
    # Convert inventory to index-based
    inventory_by_index = {}
    for piece_id, count in inventory.items():
        idx = catalog._id_to_index.get(piece_id)
        if idx is not None:
            inventory_by_index[idx] = count

    return SwitchPreservingCrossover(
        inventory_by_index=inventory_by_index,
        prob=prob,
    )


# =============================================================================
# NEAT-Style Complexification Mutation
# =============================================================================

class ComplexificationMutation(Mutation):
    """NEAT-style mutation that gradually adds structural complexity.

    Unlike standard mutation that operates on existing genes, complexification
    mutations add new structural features (switch pairs, branches) that
    enable the evolution of more complex track topologies.

    Operations:
    - ADD_SWITCH_PAIR: Activate an inactive branch slot (increases topology)
    - ADJUST_PARAMS: Modify parameters of existing branches
    - PERTURB_GENES: Small random changes to RK gene values

    Based on Stanley & Miikkulainen's NEAT (2002) structural mutation.
    """

    def __init__(
        self,
        add_switch_prob: float = 0.03,
        adjust_params_prob: float = 0.15,
        perturb_prob: float = 0.80,
        perturb_strength: float = 0.1,
        innovation_tracker=None,
    ):
        """Initialize complexification mutation.

        Args:
            add_switch_prob: Probability of adding a new branch (structural).
            adjust_params_prob: Probability of adjusting existing branch params.
            perturb_prob: Probability of small RK gene perturbation.
            perturb_strength: Standard deviation of gene perturbation.
            innovation_tracker: Optional InnovationTracker for NEAT-style tracking.
        """
        super().__init__()
        self.add_switch_prob = add_switch_prob
        self.adjust_params_prob = adjust_params_prob
        self.perturb_prob = perturb_prob
        self.perturb_strength = perturb_strength
        self.innovation_tracker = innovation_tracker

    def _do(self, problem, X, **kwargs) -> NDArray:
        """Apply complexification mutation to population.

        Args:
            problem: Optimization problem.
            X: Population array of shape (n_pop, n_var).
            **kwargs: Additional arguments.

        Returns:
            Mutated population array.
        """
        X_mutated = X.copy()

        for i in range(len(X)):
            X_mutated[i] = self._mutate_individual(X[i])

        return X_mutated

    def _mutate_individual(self, x: NDArray) -> NDArray:
        """Apply complexification mutation to single chromosome.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        x = x.copy()

        # Select mutation type based on probabilities
        r = np.random.random()

        if r < self.add_switch_prob:
            x = self._add_branch_slot(x)
        elif r < self.add_switch_prob + self.adjust_params_prob:
            x = self._adjust_branch_params(x)
        elif r < self.add_switch_prob + self.adjust_params_prob + self.perturb_prob:
            x = self._perturb_genes(x)

        return x

    def _add_branch_slot(self, x: NDArray) -> NDArray:
        """ADD_SWITCH_PAIR: Activate an inactive branch slot.

        This is a structural mutation that increases topology complexity.
        The innovation tracker records when this configuration was first seen.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome with new branch.
        """
        # Find inactive branch slots
        inactive_slots = find_inactive_branch_slots(x)
        if not inactive_slots:
            return x  # All slots are active

        # Select first inactive slot
        slot_idx = inactive_slots[0]

        # Generate random branch parameters
        in_pos = np.random.randint(4, 80)  # Position in main loop
        handedness = np.random.randint(0, 2)  # 0=LEFT, 1=RIGHT
        n_straights = np.random.randint(0, 5)  # Number of straights

        # Record innovation if tracker is available
        if self.innovation_tracker is not None:
            params = (in_pos, handedness, n_straights)
            self.innovation_tracker.get_or_create("branch", params)

        # Activate the branch slot
        set_branch_template_params(x, slot_idx, in_pos, handedness, n_straights, active=1)

        return x

    def _adjust_branch_params(self, x: NDArray) -> NDArray:
        """ADJUST_PARAMS: Modify parameters of an existing branch.

        Makes small adjustments to branch configuration without adding
        new structural complexity.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        active_slots = find_active_branch_slots(x)
        if not active_slots:
            return x

        # Select random active branch
        slot_idx = np.random.choice(active_slots)
        in_pos, handedness, n_straights, _ = get_branch_template_params(x, slot_idx)

        # Randomly adjust one parameter
        param_choice = np.random.randint(0, 3)

        if param_choice == 0:
            # Shift IN position by ±1-5
            delta = np.random.randint(-5, 6)
            in_pos = max(0, min(99, in_pos + delta))
        elif param_choice == 1:
            # Flip handedness
            handedness = 1 - handedness
        else:
            # Adjust n_straights by ±1-2
            delta = np.random.randint(-2, 3)
            n_straights = max(0, min(8, n_straights + delta))

        set_branch_template_params(x, slot_idx, in_pos, handedness, n_straights, active=1)

        return x

    def _perturb_genes(self, x: NDArray) -> NDArray:
        """PERTURB_GENES: Small random changes to RK gene values.

        Adds Gaussian noise to existing gene values, similar to
        polynomial mutation but lighter.

        Args:
            x: Chromosome array.

        Returns:
            Mutated chromosome.
        """
        # Only perturb main loop and priority genes
        mask = np.zeros(len(x), dtype=bool)
        mask[MAIN_LOOP_START:MAIN_LOOP_END] = True

        # Apply Gaussian perturbation
        noise = np.random.normal(0, self.perturb_strength, size=len(x))
        x[mask] = x[mask] + noise[mask]

        # Clip to [0, 1]
        x = np.clip(x, 0, 1)

        return x


def create_complexification_mutation(
    add_switch_prob: float = 0.03,
    adjust_params_prob: float = 0.15,
    perturb_prob: float = 0.80,
    innovation_tracker=None,
) -> ComplexificationMutation:
    """Create a complexification mutation operator.

    Args:
        add_switch_prob: Probability of structural mutation.
        adjust_params_prob: Probability of parameter adjustment.
        perturb_prob: Probability of gene perturbation.
        innovation_tracker: Optional innovation tracker.

    Returns:
        Configured ComplexificationMutation instance.
    """
    return ComplexificationMutation(
        add_switch_prob=add_switch_prob,
        adjust_params_prob=adjust_params_prob,
        perturb_prob=perturb_prob,
        innovation_tracker=innovation_tracker,
    )
