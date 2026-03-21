"""NEAT-style speciation for topology-aware evolution.

Groups chromosomes into species based on structural similarity,
enabling parallel exploration of different track topologies.

Based on Stanley & Miikkulainen's NEAT (2002) speciation approach.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .encoding import (
    B_MAX,
    MAIN_LOOP_START,
    MAIN_LOOP_END,
    BRANCH_START,
    B_SLOT,
    is_branch_active,
    get_branch_template_params,
)
from .innovation import InnovationTracker, compute_innovation_distance


@dataclass
class Species:
    """A group of topologically similar individuals.

    Attributes:
        species_id: Unique identifier for this species.
        representative: Chromosome of the species representative.
        members: Indices into the population array.
        best_fitness: Best fitness achieved by any member.
        fitness_history: Historical best fitness per generation.
        stagnation_count: Generations without improvement.
        age: Number of generations this species has existed.
    """
    species_id: int
    representative: NDArray
    members: List[int] = field(default_factory=list)
    best_fitness: float = float('inf')  # Lower is better (pymoo minimization)
    fitness_history: List[float] = field(default_factory=list)
    stagnation_count: int = 0
    age: int = 0

    def update_fitness(self, new_best: float) -> None:
        """Update fitness and track stagnation.

        Args:
            new_best: Best fitness from current generation.
        """
        self.fitness_history.append(new_best)

        if new_best < self.best_fitness - 0.001:  # Improvement threshold
            self.best_fitness = new_best
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        self.age += 1

    @property
    def size(self) -> int:
        """Number of members in this species."""
        return len(self.members)

    @property
    def is_stagnant(self) -> bool:
        """Check if species is stagnant (configurable threshold)."""
        return self.stagnation_count > 15

    def select_representative(self, population: NDArray) -> None:
        """Select new representative from current members.

        Chooses the member closest to the species centroid.

        Args:
            population: Full population array.
        """
        if not self.members:
            return

        member_chromosomes = population[self.members]
        centroid = np.mean(member_chromosomes, axis=0)

        distances = np.linalg.norm(member_chromosomes - centroid, axis=1)
        best_idx = np.argmin(distances)
        self.representative = member_chromosomes[best_idx].copy()


class TopologySpeciation:
    """NEAT-style speciation based on structural similarity.

    Groups individuals with similar topologies together, allowing
    different track configurations to evolve in parallel without
    competing directly.

    Example:
        speciation = TopologySpeciation(
            compatibility_threshold=3.0,
            disjoint_coeff=1.0,
            weight_coeff=0.4,
        )

        species_list = speciation.speciate(population, fitness)
    """

    def __init__(
        self,
        compatibility_threshold: float = 3.0,
        disjoint_coeff: float = 1.0,
        excess_coeff: float = 1.0,
        weight_coeff: float = 0.4,
        innovation_tracker: Optional[InnovationTracker] = None,
    ):
        """Initialize speciation system.

        Args:
            compatibility_threshold: Max distance for same species.
            disjoint_coeff: Weight for disjoint gene count (c1).
            excess_coeff: Weight for excess gene count (c2).
            weight_coeff: Weight for gene value differences (c3).
            innovation_tracker: Tracker for innovation IDs.
        """
        self.threshold = compatibility_threshold
        self.c1 = disjoint_coeff
        self.c2 = excess_coeff
        self.c3 = weight_coeff
        self.tracker = innovation_tracker or InnovationTracker()

        self.species: List[Species] = []
        self._next_species_id: int = 0

    def compute_distance(self, x1: NDArray, x2: NDArray) -> float:
        """Compute topological distance between two chromosomes.

        Distance formula (from NEAT):
        d = c1 * D/N + c2 * E/N + c3 * W

        Where:
        - D = disjoint gene count
        - E = excess gene count
        - N = normalizing factor (max genes)
        - W = average weight difference

        Args:
            x1: First chromosome.
            x2: Second chromosome.

        Returns:
            Compatibility distance.
        """
        # Count structural differences (active branches)
        branches1 = self._count_active_branches(x1)
        branches2 = self._count_active_branches(x2)

        # Simple structural distance based on branch count difference
        structural_diff = abs(branches1 - branches2)

        # Get branch configuration similarity
        config_diff = self._branch_config_distance(x1, x2)

        # Gene value distance (for main loop genes)
        ml1 = x1[MAIN_LOOP_START:MAIN_LOOP_END]
        ml2 = x2[MAIN_LOOP_START:MAIN_LOOP_END]
        weight_diff = float(np.mean(np.abs(ml1 - ml2)))

        # Combined distance
        N = max(B_MAX, 1)
        distance = (
            self.c1 * structural_diff / N +
            self.c2 * config_diff / N +
            self.c3 * weight_diff
        )

        return distance

    def _count_active_branches(self, x: NDArray) -> int:
        """Count active branch slots in chromosome."""
        return sum(1 for i in range(B_MAX) if is_branch_active(x, i))

    def _branch_config_distance(self, x1: NDArray, x2: NDArray) -> float:
        """Compute distance based on branch configurations."""
        diff = 0.0

        for i in range(B_MAX):
            active1 = is_branch_active(x1, i)
            active2 = is_branch_active(x2, i)

            if active1 != active2:
                diff += 1.0
            elif active1 and active2:
                # Both active - compare parameters
                params1 = get_branch_template_params(x1, i)
                params2 = get_branch_template_params(x2, i)

                # Handedness mismatch is significant
                if params1[1] != params2[1]:
                    diff += 0.5

                # Position and straights difference
                pos_diff = abs(params1[0] - params2[0]) / 100.0
                str_diff = abs(params1[2] - params2[2]) / 8.0
                diff += 0.25 * (pos_diff + str_diff)

        return diff

    def speciate(
        self,
        population: NDArray,
        fitness: Optional[NDArray] = None,
    ) -> List[Species]:
        """Assign individuals to species based on compatibility.

        Args:
            population: Array of chromosomes (n_pop, n_var).
            fitness: Optional fitness values for updating species fitness.

        Returns:
            List of species with assigned members.
        """
        n_pop = len(population)

        # Clear member lists
        for species in self.species:
            species.members.clear()

        # Assign each individual to a species
        for i in range(n_pop):
            chromosome = population[i]
            assigned = False

            # Try to assign to existing species
            for species in self.species:
                if species.representative is not None:
                    distance = self.compute_distance(chromosome, species.representative)
                    if distance < self.threshold:
                        species.members.append(i)
                        assigned = True
                        break

            # Create new species if no match
            if not assigned:
                new_species = Species(
                    species_id=self._next_species_id,
                    representative=chromosome.copy(),
                    members=[i],
                )
                self.species.append(new_species)
                self._next_species_id += 1

        # Remove empty species
        self.species = [s for s in self.species if s.members]

        # Update representatives and fitness
        for species in self.species:
            species.select_representative(population)

            if fitness is not None and species.members:
                member_fitness = fitness[species.members]
                best_fit = float(np.min(member_fitness))
                species.update_fitness(best_fit)

        return self.species

    def get_species_count(self) -> int:
        """Get current number of species."""
        return len(self.species)

    def get_largest_species(self) -> Optional[Species]:
        """Get species with most members."""
        if not self.species:
            return None
        return max(self.species, key=lambda s: s.size)

    def get_stagnant_species(self, threshold: int = 15) -> List[Species]:
        """Get species that haven't improved recently.

        Args:
            threshold: Generations without improvement to consider stagnant.

        Returns:
            List of stagnant species.
        """
        return [s for s in self.species if s.stagnation_count > threshold]

    def remove_stagnant_species(self, threshold: int = 15, keep_min: int = 2) -> int:
        """Remove stagnant species, keeping at least keep_min.

        Args:
            threshold: Stagnation threshold.
            keep_min: Minimum species to keep.

        Returns:
            Number of species removed.
        """
        if len(self.species) <= keep_min:
            return 0

        # Sort by stagnation (most stagnant first)
        sorted_species = sorted(self.species, key=lambda s: -s.stagnation_count)

        removed = 0
        new_species = []

        for species in sorted_species:
            if (species.stagnation_count <= threshold or
                    len(new_species) < keep_min):
                new_species.append(species)
            else:
                removed += 1

        self.species = new_species
        return removed

    def compute_adjusted_fitness(
        self,
        population: NDArray,
        fitness: NDArray,
    ) -> NDArray:
        """Compute adjusted fitness for explicit fitness sharing.

        Adjusted fitness = raw_fitness / species_size

        This prevents large species from dominating selection.

        Args:
            population: Population array.
            fitness: Raw fitness values.

        Returns:
            Adjusted fitness values.
        """
        adjusted = fitness.copy()

        for species in self.species:
            if species.members:
                species_size = len(species.members)
                # Divide fitness by species size (sharing)
                adjusted[species.members] = fitness[species.members] / species_size

        return adjusted

    def allocate_offspring(
        self,
        total_offspring: int,
        min_per_species: int = 1,
    ) -> Dict[int, int]:
        """Allocate offspring to species proportionally.

        Species with better average fitness get more offspring.

        Args:
            total_offspring: Total offspring to allocate.
            min_per_species: Minimum offspring per species.

        Returns:
            Dictionary mapping species_id to offspring count.
        """
        if not self.species:
            return {}

        # Compute adjusted fitness sum per species
        # Lower fitness is better, so invert for allocation
        fitness_sums = []
        for species in self.species:
            if species.best_fitness < float('inf'):
                # Invert so lower (better) fitness gets higher allocation
                inv_fitness = 1.0 / (species.best_fitness + 1e-6)
            else:
                inv_fitness = 0.0
            fitness_sums.append(inv_fitness)

        total_fitness = sum(fitness_sums)

        allocation = {}
        allocated = 0

        for species, inv_fit in zip(self.species, fitness_sums):
            if total_fitness > 0:
                proportion = inv_fit / total_fitness
                count = max(min_per_species, int(proportion * total_offspring))
            else:
                count = total_offspring // len(self.species)

            allocation[species.species_id] = count
            allocated += count

        # Adjust to match total_offspring
        diff = total_offspring - allocated
        if diff != 0 and self.species:
            # Add/remove from largest species
            largest = self.get_largest_species()
            if largest:
                allocation[largest.species_id] = max(
                    min_per_species,
                    allocation[largest.species_id] + diff
                )

        return allocation

    def get_summary(self) -> Dict:
        """Get summary statistics about current speciation state."""
        if not self.species:
            return {"n_species": 0}

        sizes = [s.size for s in self.species]
        stagnations = [s.stagnation_count for s in self.species]
        ages = [s.age for s in self.species]

        return {
            "n_species": len(self.species),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "mean_size": np.mean(sizes),
            "max_stagnation": max(stagnations),
            "mean_stagnation": np.mean(stagnations),
            "oldest_species": max(ages),
        }
