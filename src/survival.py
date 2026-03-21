"""Structural niching survival operator for track layout optimization.

Ensures diversity by preserving solutions with complex pieces (switches, crossings)
even when simple solutions (circles, ovals) dominate.

The survival operator guarantees at least 1/3 of the surviving population contains
"complex" pieces: switches, 90-degree crossings, double crossovers, etc.
"""

from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

from .encoding import (
    COMPLEX_PIECE_INDICES,
    MAIN_LOOP_END,
    MAIN_LOOP_START,
    RK_INACTIVE_THRESHOLD,
    SIMPLE_PIECE_INDICES,
    STRAIGHT_16,
    SWITCH_LEFT_IN,
    SWITCH_LEFT_OUT,
    SWITCH_RIGHT_IN,
    SWITCH_RIGHT_OUT,
    SWITCH_PAIRS,
    rk_to_piece_index,
)


def has_complex_pieces(x: NDArray, available_pieces: list = None) -> bool:
    """Check if chromosome contains any complex pieces.

    Args:
        x: Chromosome array with RK values.
        available_pieces: Sorted list of available piece indices for RK decoding.
            If None, uses hardcoded ranges (less accurate).

    Returns:
        True if chromosome has switches, crossings, or other complex pieces.
    """
    main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END]

    for gene in main_loop:
        if gene < RK_INACTIVE_THRESHOLD:
            continue  # Skip inactive genes

        if available_pieces is not None:
            piece_idx = rk_to_piece_index(float(gene), available_pieces)
        else:
            # Fallback: assume complex pieces are in upper RK range [0.4, 1.0]
            # This is approximate but better than int(gene)
            piece_idx = -1 if gene < 0.4 else 5  # Assume complex if > 0.4

        if piece_idx in COMPLEX_PIECE_INDICES:
            return True
    return False


def count_complex_pieces(x: NDArray, available_pieces: list = None) -> int:
    """Count number of complex pieces in chromosome.

    Args:
        x: Chromosome array with RK values.
        available_pieces: Sorted list of available piece indices for RK decoding.

    Returns:
        Count of complex pieces.
    """
    main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END]
    count = 0

    for gene in main_loop:
        if gene < RK_INACTIVE_THRESHOLD:
            continue

        if available_pieces is not None:
            piece_idx = rk_to_piece_index(float(gene), available_pieces)
        else:
            piece_idx = -1 if gene < 0.4 else 5

        if piece_idx in COMPLEX_PIECE_INDICES:
            count += 1
    return count


def get_complexity_category(x: NDArray, available_pieces: list = None) -> int:
    """Get complexity category for niching.

    Categories:
        0: Simple only (straights + curves)
        1: Has 1-2 complex pieces
        2: Has 3-4 complex pieces
        3: Has 5+ complex pieces

    Args:
        x: Chromosome array.
        available_pieces: Sorted list of available piece indices.

    Returns:
        Complexity category (0-3).
    """
    count = count_complex_pieces(x, available_pieces)
    if count == 0:
        return 0
    elif count <= 2:
        return 1
    elif count <= 4:
        return 2
    else:
        return 3


def count_switch_pairs(x: NDArray, available_pieces: list = None) -> Tuple[int, int, int, int]:
    """Count switch pieces by type.

    Args:
        x: Chromosome array with RK values.
        available_pieces: Sorted list of available piece indices.

    Returns:
        Tuple of (left_in, left_out, right_in, right_out) counts.
    """
    main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END]
    left_in = left_out = right_in = right_out = 0

    for gene in main_loop:
        if gene < RK_INACTIVE_THRESHOLD:
            continue

        if available_pieces is not None:
            piece_idx = rk_to_piece_index(float(gene), available_pieces)
        else:
            continue  # Can't decode without available_pieces

        if piece_idx == SWITCH_LEFT_IN:
            left_in += 1
        elif piece_idx == SWITCH_LEFT_OUT:
            left_out += 1
        elif piece_idx == SWITCH_RIGHT_IN:
            right_in += 1
        elif piece_idx == SWITCH_RIGHT_OUT:
            right_out += 1

    return left_in, left_out, right_in, right_out


def count_balanced_switch_pairs(x: NDArray, available_pieces: list = None) -> int:
    """Count number of balanced switch pairs.

    A balanced pair has equal IN and OUT switches for same handedness.

    Args:
        x: Chromosome array.
        available_pieces: Sorted list of available piece indices.

    Returns:
        Number of balanced pairs (min(left_in, left_out) + min(right_in, right_out)).
    """
    left_in, left_out, right_in, right_out = count_switch_pairs(x, available_pieces)
    return min(left_in, left_out) + min(right_in, right_out)


def inject_switch_pair(
    x: NDArray,
    available_pieces: list = None,
    rng: np.random.Generator = None,
) -> bool:
    """Inject a balanced switch pair into a simple chromosome.

    Finds two STRAIGHT_16 pieces and converts them to a switch IN/OUT pair.
    The OUT switch is placed after the IN switch to create proper ordering.

    Args:
        x: Chromosome array (modified in place) with RK values.
        available_pieces: Sorted list of available piece indices for RK encoding.
        rng: Random number generator.

    Returns:
        True if injection succeeded, False if not enough straights found.
    """
    if rng is None:
        rng = np.random.default_rng()

    if available_pieces is None:
        return False  # Can't inject without knowing available pieces

    main_loop = x[MAIN_LOOP_START:MAIN_LOOP_END]

    # Find all positions with STRAIGHT_16
    straight_positions = []
    for i, gene in enumerate(main_loop):
        if gene < RK_INACTIVE_THRESHOLD:
            continue
        piece_idx = rk_to_piece_index(float(gene), available_pieces)
        if piece_idx == STRAIGHT_16:
            straight_positions.append(i)

    # Need at least 2 straights, preferably separated by some distance
    if len(straight_positions) < 2:
        return False

    # Choose handedness randomly (0 = left, 1 = right)
    handedness = rng.integers(0, 2)
    in_switch, out_switch = SWITCH_PAIRS[handedness]

    # Check if switches are available
    if in_switch not in available_pieces or out_switch not in available_pieces:
        return False

    # Pick first straight for IN switch
    in_idx = rng.choice(len(straight_positions))
    in_pos = straight_positions[in_idx]

    # Pick second straight for OUT switch (different position)
    remaining = [p for p in straight_positions if p != in_pos]
    if not remaining:
        return False

    # Try to find a position at least 3 steps away
    far_positions = [p for p in remaining if abs(p - in_pos) >= 3]
    if far_positions:
        out_pos = rng.choice(far_positions)
    else:
        out_pos = rng.choice(remaining)

    # Ensure proper ordering: IN before OUT in the loop
    if out_pos < in_pos:
        in_pos, out_pos = out_pos, in_pos

    # Convert piece indices to RK values
    from .encoding import piece_index_to_rk
    in_rk = piece_index_to_rk(in_switch, available_pieces)
    out_rk = piece_index_to_rk(out_switch, available_pieces)

    # Inject the switch pair as RK values
    x[MAIN_LOOP_START + in_pos] = in_rk
    x[MAIN_LOOP_START + out_pos] = out_rk

    return True


# =============================================================================
# Structural Niching Survival
# =============================================================================

class StructuralNichingSurvival(Survival):
    """Survival operator that preserves structural diversity.

    Ensures at least `complex_ratio` of surviving population contains
    complex pieces (switches, crossings, etc.). This prevents simple
    circles from eliminating all switch-based layouts.

    CRITICAL: Complex solutions use SOFT feasibility - they're kept alive
    but those closer to feasible are preferred. This allows switch-based
    layouts to evolve toward valid configurations over time.

    Algorithm:
    1. Separate population into "simple" and "complex" groups
    2. Allocate survival slots: complex_ratio to complex, rest to simple
    3. Complex group: rank by fitness only (IGNORE feasibility)
    4. Simple group: rank by feasibility first, then fitness
    5. If complex group is too small, fill remaining slots from simple
    """

    def __init__(
        self,
        complex_ratio: float = 0.33,
        prefer_balanced_switches: bool = True,
        inject_switches: bool = True,
        injection_ratio: float = 0.5,
        seed: int = None,
        available_pieces: list = None,
    ):
        """Initialize structural niching survival.

        Args:
            complex_ratio: Minimum fraction of survivors with complex pieces.
            prefer_balanced_switches: If True, prefer solutions with balanced
                switch pairs (equal IN/OUT) over unbalanced ones.
            inject_switches: If True, proactively inject switch pairs into
                simple solutions when complex population is below target.
            injection_ratio: Fraction of the deficit to fill via injection
                (0.5 = inject into half of needed conversions).
            seed: Random seed for injection reproducibility.
            available_pieces: Sorted list of available piece indices for RK decoding.
        """
        super().__init__(filter_infeasible=False)
        self.complex_ratio = complex_ratio
        self.prefer_balanced_switches = prefer_balanced_switches
        self.inject_switches = inject_switches
        self.injection_ratio = injection_ratio
        self.nds = NonDominatedSorting()
        self.rng = np.random.default_rng(seed)
        self.available_pieces = available_pieces

    def _do(
        self,
        problem,
        pop,
        *args,
        n_survive: int = None,
        **kwargs,
    ) -> Population:
        """Select survivors with structural diversity preservation.

        Args:
            problem: Optimization problem.
            pop: Current population.
            n_survive: Number of individuals to survive.
            **kwargs: Additional arguments.

        Returns:
            Surviving population.
        """
        if n_survive is None:
            n_survive = len(pop)

        # Get population data
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")

        n_pop = len(pop)

        # Classify individuals as simple or complex
        is_complex = np.array([has_complex_pieces(x, self.available_pieces) for x in X])
        complex_indices = np.where(is_complex)[0]
        simple_indices = np.where(~is_complex)[0]

        # Calculate target slots for each group
        n_complex_target = int(np.ceil(n_survive * self.complex_ratio))
        n_simple_target = n_survive - n_complex_target

        # PROACTIVE INJECTION: If not enough complex individuals, inject switches
        n_complex_available = len(complex_indices)
        if self.inject_switches and n_complex_available < n_complex_target:
            deficit = n_complex_target - n_complex_available
            n_to_inject = int(np.ceil(deficit * self.injection_ratio))

            # Select simple individuals to convert (prefer better fitness)
            if len(simple_indices) > 0 and n_to_inject > 0:
                simple_F = F[simple_indices]
                # Sort by fitness (lower = better for minimization)
                sorted_simple = np.argsort(simple_F[:, 0])
                # Take top candidates (skip the very best to preserve diversity)
                skip_best = min(5, len(sorted_simple) // 4)
                candidates = sorted_simple[skip_best : skip_best + n_to_inject * 2]

                n_injected = 0
                for local_idx in candidates:
                    if n_injected >= n_to_inject:
                        break
                    pop_idx = simple_indices[local_idx]
                    # Inject switch pair (modifies X in place)
                    if inject_switch_pair(X[pop_idx], self.available_pieces, self.rng):
                        n_injected += 1

                # Re-classify after injection
                if n_injected > 0:
                    is_complex = np.array([has_complex_pieces(x, self.available_pieces) for x in X])
                    complex_indices = np.where(is_complex)[0]
                    simple_indices = np.where(~is_complex)[0]

        # Adjust if still not enough complex individuals
        n_complex_available = len(complex_indices)
        if n_complex_available < n_complex_target:
            n_complex_target = n_complex_available
            n_simple_target = n_survive - n_complex_target

        survivors = []

        # Select from complex group - SOFT FEASIBILITY to keep them alive
        # while preferring those closer to being feasible
        if len(complex_indices) > 0 and n_complex_target > 0:
            complex_survivors = self._select_from_group(
                pop, complex_indices, n_complex_target, F, G,
                soft_feasibility=True,  # Keep complex solutions, prefer closer to feasible
            )
            survivors.extend(complex_survivors)

        # Select from simple group - use normal feasibility ranking
        if len(simple_indices) > 0 and n_simple_target > 0:
            simple_survivors = self._select_from_group(
                pop, simple_indices, n_simple_target, F, G,
                soft_feasibility=False,  # Simple solutions: hard feasibility
            )
            survivors.extend(simple_survivors)

        # If we don't have enough survivors (unlikely), fill from either group
        if len(survivors) < n_survive:
            remaining_indices = [i for i in range(n_pop) if i not in survivors]
            n_needed = n_survive - len(survivors)
            if remaining_indices:
                additional = self._select_from_group(
                    pop, np.array(remaining_indices), n_needed, F, G
                )
                survivors.extend(additional)

        # Create surviving population
        return pop[survivors[:n_survive]]

    def _select_from_group(
        self,
        pop: Population,
        indices: NDArray,
        n_select: int,
        F: NDArray,
        G: NDArray,
        soft_feasibility: bool = False,
    ) -> List[int]:
        """Select best individuals from a group.

        Selection criteria:
        - Simple group (soft_feasibility=False): feasibility first, then fitness
        - Complex group (soft_feasibility=True): fitness first, with soft penalty
          for constraint violation (keeps them alive but prefers closer to feasible)

        Args:
            pop: Full population.
            indices: Indices of individuals in this group.
            n_select: Number to select.
            F: Fitness values for full population.
            G: Constraint values for full population.
            soft_feasibility: If True, use soft penalty instead of hard feasibility.
                Complex solutions stay alive but those closer to feasible are preferred.

        Returns:
            List of selected indices (from original population).
        """
        if len(indices) == 0:
            return []

        n_select = min(n_select, len(indices))

        # Get data for this group
        X = pop.get("X")
        group_F = F[indices]
        group_G = G[indices] if G is not None else None

        # Calculate composite score for ranking
        # Lower score = better
        scores = np.zeros(len(indices))

        if soft_feasibility and group_G is not None:
            # SOFT FEASIBILITY for complex group:
            # Strong gradient toward feasibility while keeping them alive.
            # Feasible complex solutions get a large bonus to compete with simple ovals.

            # Sum of positive constraint violations (g > 0 means violated)
            cv = np.maximum(group_G, 0).sum(axis=1)  # Total constraint violation
            is_feasible = np.all(group_G <= 0, axis=1)

            # Soft penalty: 50 per unit of CV (strong gradient toward feasibility)
            scores += cv * 50

            # Feasibility bonus: make feasible complex solutions strongly competitive
            scores[is_feasible] -= 1500

        elif not soft_feasibility and group_G is not None:
            # HARD FEASIBILITY for simple group:
            # Infeasible solutions get massive penalty
            is_feasible = np.all(group_G <= 0, axis=1)
            scores[~is_feasible] += 1e6

        # Primary: fitness (assuming minimization, F is already negative of what we want)
        # So lower F = better (more pieces = more negative F = lower score)
        scores += group_F[:, 0]  # Single objective

        # For complex individuals: bonus for balanced switches
        # This helps evolve toward valid switch pairs
        if self.prefer_balanced_switches and self.available_pieces:
            for i, idx in enumerate(indices):
                if has_complex_pieces(X[idx], self.available_pieces):
                    n_balanced = count_balanced_switch_pairs(X[idx], self.available_pieces)
                    n_complex = count_complex_pieces(X[idx], self.available_pieces)
                    # Bonus for balanced switches (negative = bonus)
                    balanced_bonus = n_balanced * -500  # Reward balanced pairs
                    unbalanced_penalty = (n_complex - n_balanced * 2) * 200
                    scores[i] += balanced_bonus + unbalanced_penalty

        # Sort by score (ascending = best first)
        sorted_local_indices = np.argsort(scores)

        # Return original population indices
        return [indices[i] for i in sorted_local_indices[:n_select]]


# =============================================================================
# Alternative: Category-Based Niching
# =============================================================================

class CategoryNichingSurvival(Survival):
    """Survival operator with explicit complexity categories.

    Divides population into 4 categories by complexity level and ensures
    representation from each category in the surviving population.

    Categories:
        0: Simple only (0 complex pieces)
        1: Low complexity (1-2 complex pieces)
        2: Medium complexity (3-4 complex pieces)
        3: High complexity (5+ complex pieces)
    """

    def __init__(
        self,
        category_ratios: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    ):
        """Initialize category niching survival.

        Args:
            category_ratios: Target ratio for each category. Will be normalized
                if doesn't sum to 1.0.
        """
        super().__init__(filter_infeasible=False)

        # Normalize ratios
        total = sum(category_ratios)
        self.category_ratios = tuple(r / total for r in category_ratios)

    def _do(
        self,
        problem,
        pop,
        *args,
        n_survive: int = None,
        **kwargs,
    ) -> Population:
        """Select survivors with category-based diversity.

        Args:
            problem: Optimization problem.
            pop: Current population.
            n_survive: Number of individuals to survive.
            **kwargs: Additional arguments.

        Returns:
            Surviving population.
        """
        if n_survive is None:
            n_survive = len(pop)

        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")

        # Categorize individuals
        categories: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
        for i, x in enumerate(X):
            cat = get_complexity_category(x)
            categories[cat].append(i)

        # Calculate target slots per category
        targets = [int(np.ceil(n_survive * r)) for r in self.category_ratios]

        # Adjust for available individuals
        available = [len(categories[c]) for c in range(4)]
        actual_targets = [min(t, a) for t, a in zip(targets, available)]

        # Redistribute excess slots
        total_actual = sum(actual_targets)
        if total_actual < n_survive:
            deficit = n_survive - total_actual
            # Give to categories with excess available
            for c in range(4):
                if available[c] > actual_targets[c]:
                    can_add = min(deficit, available[c] - actual_targets[c])
                    actual_targets[c] += can_add
                    deficit -= can_add
                    if deficit <= 0:
                        break

        # Select from each category
        survivors = []
        for c in range(4):
            if actual_targets[c] > 0 and categories[c]:
                selected = self._select_best(
                    pop, np.array(categories[c]), actual_targets[c], F, G
                )
                survivors.extend(selected)

        return pop[survivors[:n_survive]]

    def _select_best(
        self,
        pop: Population,
        indices: NDArray,
        n_select: int,
        F: NDArray,
        G: NDArray,
    ) -> List[int]:
        """Select best individuals by feasibility then fitness."""
        if len(indices) == 0:
            return []

        n_select = min(n_select, len(indices))
        group_F = F[indices]
        group_G = G[indices] if G is not None else None

        # Feasibility
        if group_G is not None:
            is_feasible = np.all(group_G <= 0, axis=1)
        else:
            is_feasible = np.ones(len(indices), dtype=bool)

        # Score: infeasible penalty + fitness
        scores = np.zeros(len(indices))
        scores[~is_feasible] += 1e6
        scores += group_F[:, 0]

        sorted_local = np.argsort(scores)
        return [indices[i] for i in sorted_local[:n_select]]


# =============================================================================
# NEAT-Style Speciated Survival
# =============================================================================

class SpeciatedSurvival(Survival):
    """NEAT-style survival with species-based selection.

    Each species gets offspring proportional to its adjusted fitness.
    Stagnant species (no improvement for N generations) can be culled.
    This preserves topological diversity while maintaining evolutionary pressure.

    Key features:
    - Fitness sharing within species (prevents large species from dominating)
    - Offspring allocation proportional to species fitness
    - Stagnation detection and removal
    - Representative selection for species compatibility

    Based on Stanley & Miikkulainen's NEAT (2002).
    """

    def __init__(
        self,
        speciation=None,
        stagnation_limit: int = 15,
        elitism_per_species: int = 1,
        min_species_size: int = 2,
    ):
        """Initialize speciated survival.

        Args:
            speciation: TopologySpeciation instance for species management.
            stagnation_limit: Generations without improvement before culling.
            elitism_per_species: Keep best N individuals from each species.
            min_species_size: Minimum individuals to keep per species.
        """
        super().__init__(filter_infeasible=False)
        self.speciation = speciation
        self.stagnation_limit = stagnation_limit
        self.elitism_per_species = elitism_per_species
        self.min_species_size = min_species_size

    def _do(
        self,
        problem,
        pop,
        *args,
        n_survive: int = None,
        **kwargs,
    ) -> Population:
        """Select survivors with species-proportional allocation.

        Args:
            problem: Optimization problem.
            pop: Current population.
            n_survive: Number of individuals to survive.
            **kwargs: Additional arguments.

        Returns:
            Surviving population.
        """
        if n_survive is None:
            n_survive = len(pop)

        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")

        # If no speciation system, fall back to standard selection
        if self.speciation is None:
            return self._fallback_selection(pop, n_survive, F, G)

        # Update species with current population
        species_list = self.speciation.speciate(X, F[:, 0])

        # Remove stagnant species (keep minimum)
        self.speciation.remove_stagnant_species(
            threshold=self.stagnation_limit,
            keep_min=2
        )

        # Allocate offspring to species
        allocation = self.speciation.allocate_offspring(
            total_offspring=n_survive,
            min_per_species=self.min_species_size
        )

        survivors = []

        for species in self.speciation.species:
            n_alloc = allocation.get(species.species_id, 0)
            if n_alloc == 0 or not species.members:
                continue

            # Select best individuals from this species
            selected = self._select_from_species(
                species.members,
                n_alloc,
                F,
                G,
            )
            survivors.extend(selected)

        # If we don't have enough, fill from best overall
        if len(survivors) < n_survive:
            all_indices = set(range(len(pop)))
            remaining = list(all_indices - set(survivors))
            n_needed = n_survive - len(survivors)

            if remaining:
                # Sort by fitness and take best
                remaining_F = F[remaining, 0]
                sorted_idx = np.argsort(remaining_F)
                additional = [remaining[i] for i in sorted_idx[:n_needed]]
                survivors.extend(additional)

        return pop[survivors[:n_survive]]

    def _select_from_species(
        self,
        member_indices: List[int],
        n_select: int,
        F: NDArray,
        G: NDArray,
    ) -> List[int]:
        """Select best individuals from a species.

        Uses constraint-weighted fitness: feasible solutions preferred,
        but infeasible ones can survive if they're close.

        Args:
            member_indices: Population indices of species members.
            n_select: Number to select.
            F: Fitness values.
            G: Constraint values.

        Returns:
            List of selected population indices.
        """
        if not member_indices:
            return []

        n_select = min(n_select, len(member_indices))
        indices = np.array(member_indices)

        # Compute scores (lower = better)
        scores = F[indices, 0].copy()

        if G is not None:
            # Add soft constraint violation penalty
            cv = np.maximum(G[indices], 0).sum(axis=1)
            scores += cv * 10  # Soft penalty

        # Sort and select best
        sorted_local = np.argsort(scores)
        return [indices[i] for i in sorted_local[:n_select]]

    def _fallback_selection(
        self,
        pop: Population,
        n_survive: int,
        F: NDArray,
        G: NDArray,
    ) -> Population:
        """Fallback selection when speciation is not available.

        Uses standard feasibility-first, then fitness selection.

        Args:
            pop: Population.
            n_survive: Number to survive.
            F: Fitness values.
            G: Constraint values.

        Returns:
            Surviving population.
        """
        n_pop = len(pop)
        scores = F[:, 0].copy()

        if G is not None:
            is_feasible = np.all(G <= 0, axis=1)
            scores[~is_feasible] += 1e6

        sorted_indices = np.argsort(scores)
        return pop[sorted_indices[:n_survive]]

    def get_species_stats(self) -> Dict:
        """Get statistics about current species state.

        Returns:
            Dictionary with species statistics.
        """
        if self.speciation is None:
            return {"n_species": 0}

        return self.speciation.get_summary()


def create_speciated_survival(
    compatibility_threshold: float = 3.0,
    stagnation_limit: int = 15,
    innovation_tracker=None,
) -> SpeciatedSurvival:
    """Create a speciated survival operator with default settings.

    Args:
        compatibility_threshold: Threshold for species membership.
        stagnation_limit: Generations before culling stagnant species.
        innovation_tracker: Optional innovation tracker.

    Returns:
        Configured SpeciatedSurvival instance.
    """
    from .speciation import TopologySpeciation

    speciation = TopologySpeciation(
        compatibility_threshold=compatibility_threshold,
        innovation_tracker=innovation_tracker,
    )

    return SpeciatedSurvival(
        speciation=speciation,
        stagnation_limit=stagnation_limit,
    )