"""Island Model implementation for improved diversity preservation.

Based on research best practices:
- 4-8 subpopulations evolving independently
- Periodic migration (5-10% of individuals every 10-20 generations)
- Uses pymoo's ask-and-tell interface for flexible control

This is particularly effective for combinatorial problems with many local optima,
like LEGO track layout optimization.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.core.termination import Termination
from pymoo.termination.default import DefaultSingleObjectiveTermination


@dataclass
class IslandConfig:
    """Configuration for island model optimization.

    Attributes:
        n_islands: Number of islands (subpopulations). Research recommends 4-8.
        pop_size_per_island: Population size for each island.
        migration_interval: Generations between migrations. Research recommends 10-20.
        migration_rate: Fraction of population to migrate. Research recommends 0.05-0.10.
        migration_policy: How to select migrants ('best', 'random', 'tournament').
        topology: Migration topology ('ring', 'all_to_all', 'random').
    """

    n_islands: int = 4
    pop_size_per_island: int = 50
    migration_interval: int = 15
    migration_rate: float = 0.10
    migration_policy: str = "best"
    topology: str = "ring"


@dataclass
class IslandState:
    """State tracking for a single island."""

    island_id: int
    algorithm: Algorithm
    n_gen: int = 0
    best_f: float = float("inf")
    feasible_count: int = 0


class IslandOptimizer:
    """Island Model optimizer using pymoo's ask-and-tell interface.

    Maintains multiple independent subpopulations (islands) with periodic
    migration to preserve diversity while allowing local exploitation.

    Example:
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from src.island_model import IslandOptimizer, IslandConfig

        config = IslandConfig(n_islands=4, pop_size_per_island=50)
        optimizer = IslandOptimizer(
            problem=problem,
            algorithm_factory=lambda: GA(pop_size=50, ...),
            config=config,
        )
        result = optimizer.run(n_gen=200)
    """

    def __init__(
        self,
        problem: Problem,
        algorithm_factory: Callable[[], Algorithm],
        config: Optional[IslandConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize island model optimizer.

        Args:
            problem: pymoo Problem to solve.
            algorithm_factory: Callable that returns a new Algorithm instance.
                Called once per island.
            config: Island model configuration.
            seed: Random seed for reproducibility.
        """
        self.problem = problem
        self.algorithm_factory = algorithm_factory
        self.config = config or IslandConfig()
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Initialize islands
        self.islands: List[IslandState] = []
        self._setup_islands()

        # Tracking
        self.global_gen = 0
        self.history: List[Dict[str, Any]] = []

    def _setup_islands(self) -> None:
        """Initialize all islands with their algorithms."""
        for i in range(self.config.n_islands):
            algorithm = self.algorithm_factory()
            algorithm.setup(
                self.problem,
                termination=DefaultSingleObjectiveTermination(
                    n_max_gen=1,  # We control generations externally
                ),
                verbose=False,
            )
            self.islands.append(IslandState(island_id=i, algorithm=algorithm))

    def run(
        self,
        n_gen: int = 100,
        verbose: bool = False,
        callback: Optional[Callable[[int, List[IslandState]], None]] = None,
    ) -> Result:
        """Run island model optimization.

        Args:
            n_gen: Total number of generations to run.
            verbose: Print progress information.
            callback: Optional callback(generation, islands) called each generation.

        Returns:
            pymoo Result object with best solution across all islands.
        """
        for gen in range(n_gen):
            self.global_gen = gen

            # Evolve each island for one generation
            for island in self.islands:
                self._evolve_island(island)

            # Periodic migration
            if gen > 0 and gen % self.config.migration_interval == 0:
                self._migrate()
                if verbose:
                    best_f = min(island.best_f for island in self.islands)
                    print(f"Gen {gen}: Migration complete. Best F = {best_f:.4f}")

            # Record history
            self._record_history()

            # Callback
            if callback is not None:
                callback(gen, self.islands)

            # Verbose output
            if verbose and gen % 10 == 0:
                self._print_status(gen)

        return self._build_result()

    def _evolve_island(self, island: IslandState) -> None:
        """Evolve a single island for one generation using ask-and-tell.

        Args:
            island: Island state to evolve.
        """
        algorithm = island.algorithm

        # Ask for next population to evaluate
        pop = algorithm.ask()

        # Evaluate population
        algorithm.evaluator.eval(self.problem, pop)

        # Tell algorithm the results
        algorithm.tell(infills=pop)

        # Update island state
        island.n_gen += 1

        F = pop.get("F")
        G = pop.get("G")

        if F is not None and len(F) > 0:
            island.best_f = float(np.min(F))

        if G is not None and len(G) > 0:
            island.feasible_count = int(np.sum(np.all(G <= 0, axis=1)))

    def _migrate(self) -> None:
        """Perform migration between islands based on configured topology."""
        n_migrants = max(1, int(self.config.pop_size_per_island * self.config.migration_rate))

        if self.config.topology == "ring":
            self._migrate_ring(n_migrants)
        elif self.config.topology == "all_to_all":
            self._migrate_all_to_all(n_migrants)
        elif self.config.topology == "random":
            self._migrate_random(n_migrants)
        else:
            self._migrate_ring(n_migrants)  # Default

    def _migrate_ring(self, n_migrants: int) -> None:
        """Ring topology: each island sends to the next."""
        migrants = []

        # Collect migrants from each island
        for island in self.islands:
            migrants.append(self._select_migrants(island, n_migrants))

        # Send to next island in ring
        for i, island in enumerate(self.islands):
            source_idx = (i - 1) % len(self.islands)
            self._inject_migrants(island, migrants[source_idx])

    def _migrate_all_to_all(self, n_migrants: int) -> None:
        """All-to-all: broadcast best individuals to all islands."""
        # Collect migrants from all islands
        all_migrants = []
        for island in self.islands:
            all_migrants.extend(self._select_migrants(island, n_migrants // self.config.n_islands + 1))

        # Inject into all islands
        for island in self.islands:
            self._inject_migrants(island, all_migrants[:n_migrants])

    def _migrate_random(self, n_migrants: int) -> None:
        """Random pairs: randomly pair islands for migration."""
        indices = np.random.permutation(len(self.islands))

        for i in range(0, len(indices) - 1, 2):
            island_a = self.islands[indices[i]]
            island_b = self.islands[indices[i + 1]]

            migrants_a = self._select_migrants(island_a, n_migrants)
            migrants_b = self._select_migrants(island_b, n_migrants)

            self._inject_migrants(island_a, migrants_b)
            self._inject_migrants(island_b, migrants_a)

    def _select_migrants(self, island: IslandState, n: int) -> List[NDArray]:
        """Select migrants from island based on policy.

        Args:
            island: Source island.
            n: Number of migrants to select.

        Returns:
            List of chromosome arrays.
        """
        pop = island.algorithm.pop
        if pop is None or len(pop) == 0:
            return []

        X = pop.get("X")
        F = pop.get("F")

        if X is None or len(X) == 0:
            return []

        n = min(n, len(X))

        if self.config.migration_policy == "best":
            # Select best individuals by objective
            if F is not None and len(F) > 0:
                indices = np.argsort(F[:, 0])[:n]
            else:
                indices = np.arange(n)
        elif self.config.migration_policy == "random":
            indices = np.random.choice(len(X), n, replace=False)
        elif self.config.migration_policy == "tournament":
            # Tournament selection
            indices = []
            for _ in range(n):
                candidates = np.random.choice(len(X), min(3, len(X)), replace=False)
                if F is not None:
                    winner = candidates[np.argmin(F[candidates, 0])]
                else:
                    winner = candidates[0]
                indices.append(winner)
            indices = np.array(indices)
        else:
            indices = np.arange(n)

        return [X[i].copy() for i in indices]

    def _inject_migrants(self, island: IslandState, migrants: List[NDArray]) -> None:
        """Inject migrants into island, replacing worst individuals.

        Args:
            island: Destination island.
            migrants: List of chromosome arrays to inject.
        """
        if not migrants:
            return

        pop = island.algorithm.pop
        if pop is None or len(pop) == 0:
            return

        X = pop.get("X")
        F = pop.get("F")

        if X is None or len(X) == 0:
            return

        n_replace = min(len(migrants), len(X))

        # Find worst individuals to replace
        if F is not None and len(F) > 0:
            worst_indices = np.argsort(F[:, 0])[-n_replace:]
        else:
            worst_indices = np.arange(len(X) - n_replace, len(X))

        # Replace worst with migrants
        for i, idx in enumerate(worst_indices):
            if i < len(migrants):
                X[idx] = migrants[i]

        # Update population (need to re-evaluate replaced individuals)
        # Note: This is a simplified approach - migrants keep their original fitness
        # until next generation evaluation

    def _record_history(self) -> None:
        """Record current state to history."""
        record = {
            "generation": self.global_gen,
            "island_stats": [],
        }

        for island in self.islands:
            record["island_stats"].append(
                {
                    "island_id": island.island_id,
                    "best_f": island.best_f,
                    "feasible_count": island.feasible_count,
                    "n_gen": island.n_gen,
                }
            )

        self.history.append(record)

    def _print_status(self, gen: int) -> None:
        """Print current optimization status."""
        best_f = min(island.best_f for island in self.islands)
        total_feasible = sum(island.feasible_count for island in self.islands)
        avg_f = np.mean([island.best_f for island in self.islands])

        print(
            f"Gen {gen:4d}: Best F = {best_f:.4f}, "
            f"Avg F = {avg_f:.4f}, "
            f"Total Feasible = {total_feasible}"
        )

    def _build_result(self) -> Result:
        """Build pymoo Result from best solution across all islands.

        Returns:
            Result object with best solution.
        """
        # Find best individual across all islands
        best_island = min(self.islands, key=lambda i: i.best_f)
        best_pop = best_island.algorithm.pop

        if best_pop is None or len(best_pop) == 0:
            return Result()

        F = best_pop.get("F")
        if F is None or len(F) == 0:
            return Result()

        best_idx = np.argmin(F[:, 0])

        # Create result
        result = Result()
        result.X = best_pop.get("X")[best_idx]
        result.F = F[best_idx]
        result.G = best_pop.get("G")[best_idx] if best_pop.get("G") is not None else None

        return result

    def get_best_per_island(self) -> List[Tuple[int, float, NDArray]]:
        """Get best solution from each island.

        Returns:
            List of (island_id, best_f, best_X) tuples.
        """
        results = []

        for island in self.islands:
            pop = island.algorithm.pop
            if pop is None or len(pop) == 0:
                continue

            F = pop.get("F")
            X = pop.get("X")

            if F is not None and X is not None and len(F) > 0:
                best_idx = np.argmin(F[:, 0])
                results.append((island.island_id, float(F[best_idx, 0]), X[best_idx].copy()))

        return results

    def get_diversity_metrics(self) -> Dict[str, float]:
        """Compute diversity metrics across all islands.

        Returns:
            Dictionary with diversity statistics.
        """
        all_X = []

        for island in self.islands:
            pop = island.algorithm.pop
            if pop is not None:
                X = pop.get("X")
                if X is not None:
                    all_X.extend(X)

        if not all_X:
            return {"std": 0.0, "range": 0.0}

        all_X = np.array(all_X)

        return {
            "std": float(np.mean(np.std(all_X, axis=0))),
            "range": float(np.mean(np.ptp(all_X, axis=0))),
        }


def run_island_optimization(
    problem: Problem,
    algorithm_factory: Callable[[], Algorithm],
    n_gen: int = 200,
    config: Optional[IslandConfig] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Result:
    """Convenience function to run island model optimization.

    Args:
        problem: pymoo Problem to solve.
        algorithm_factory: Factory function returning GA instances.
        n_gen: Number of generations.
        config: Island model configuration.
        verbose: Print progress.
        seed: Random seed.

    Returns:
        Best Result from all islands.

    Example:
        from pymoo.algorithms.soo.nonconvex.ga import GA

        result = run_island_optimization(
            problem=problem,
            algorithm_factory=lambda: GA(
                pop_size=50,
                sampling=my_sampling,
                mutation=my_mutation,
                crossover=my_crossover,
            ),
            n_gen=200,
            config=IslandConfig(n_islands=4, migration_interval=20),
        )
    """
    optimizer = IslandOptimizer(
        problem=problem,
        algorithm_factory=algorithm_factory,
        config=config,
        seed=seed,
    )

    return optimizer.run(n_gen=n_gen, verbose=verbose)
