"""pymoo optimization problem definition for track layout optimization.

Single problem class with:
- ONE objective: maximize utilization (-utilization for minimization)
- Constraints via Deb's CV (normalized to same scale)
- Decoder as single source of truth for layout evaluation
"""

import logging
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from pymoo.core.problem import ElementwiseProblem

from .config import OptimizationConfig
from .data import TrackCatalog
from .decoder import DecoderConfig, decode_chromosome
from .encoding import N_VAR, generate_bounds


class TrackOptimizationProblem(ElementwiseProblem):
    """Single-objective track layout optimization.

    Objective: Maximize piece utilization (-utilization for pymoo minimization)
    Constraints: 5 normalized constraints via Deb's CV rules
        G[0]: closure error (position)
        G[1]: angle error
        G[2]: boundary violation
        G[3]: inventory violation
        G[4]: loose port count (switches/crossings must have all ports connected)

    Deb's Rules (pymoo default):
    1. Feasible solutions always beat infeasible
    2. Among infeasible, lower total CV wins
    3. Among feasible, better objective wins
    """

    def __init__(
        self,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        closure_tolerance: float = None,
        angle_tolerance: float = None,
        **kwargs,
    ):
        """Initialize track optimization problem.

        Args:
            catalog: Track catalog with piece properties
            config: Optimization configuration
            closure_tolerance: Position closure tolerance in studs (default from config)
            angle_tolerance: Angle closure tolerance in degrees (default from config)
            **kwargs: Additional arguments passed to ElementwiseProblem
        """
        # Use config tolerances or provided values
        self.closure_tolerance = closure_tolerance or config.closure_tolerance
        self.angle_tolerance = angle_tolerance or config.angle_tolerance

        # Generate segment-specific bounds
        bounds = generate_bounds(
            max_piece_index=catalog._max_index,
            boundary_min_x=config.boundary.min_x,
            boundary_max_x=config.boundary.max_x,
            boundary_min_y=config.boundary.min_y,
            boundary_max_y=config.boundary.max_y,
        )

        # Compute boundary diagonal for normalization
        self.diagonal = np.sqrt(
            (config.boundary.max_x - config.boundary.min_x) ** 2
            + (config.boundary.max_y - config.boundary.min_y) ** 2
        )

        super().__init__(
            n_var=N_VAR,
            n_obj=1,           # ONE objective: -(utilization - loose_port_penalty)
            n_ieq_constr=5,    # 5 constraints: closure, angle, boundary, inventory, secondary_closure
            xl=bounds.xl,
            xu=bounds.xu,
            **kwargs,
        )

        self.catalog = catalog
        self.config = config
        self.total_inventory = sum(config.inventory.values())

        # Convert inventory to index-based for decoder
        self.inventory_by_index = self._convert_inventory(config.inventory)

        # Decoder configuration with boundary for RK position scaling
        self.decoder_config = DecoderConfig(
            position_tolerance=self.closure_tolerance,
            angle_tolerance=self.angle_tolerance,
            boundary_min_x=config.boundary.min_x,
            boundary_max_x=config.boundary.max_x,
            boundary_min_y=config.boundary.min_y,
            boundary_max_y=config.boundary.max_y,
        )

    def _convert_inventory(self, inventory: Dict[str, int]) -> Dict[int, int]:
        """Convert inventory from piece_id to piece_index."""
        result = {}
        for piece_id, count in inventory.items():
            idx = self.catalog._id_to_index.get(piece_id)
            if idx is not None:
                result[idx] = count
        return result

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate single chromosome.

        Decoder is the single source of truth - if it produces a layout, evaluate it.

        Args:
            x: Chromosome array (1D, length N_VAR)
            out: Dictionary to populate with "F" and "G"
        """
        # Decoder is the single source of truth
        layout = decode_chromosome(
            x,
            self.catalog,
            self.config.inventory,
            self.decoder_config,
        )

        # Handle empty layouts
        if layout.n_pieces == 0:
            out["F"] = [0.0]
            out["G"] = [1.0, 1.0, 1.0, 1.0, -1.0]  # 5 constraints (secondary = feasible)
            return

        # ONE objective: maximize utilization scaled by closure quality.
        # Layouts that can't close (wrong angle OR far position) get diminished score.
        utilization = layout.n_pieces / self.total_inventory
        loose_port_penalty = layout.loose_port_count * (2.0 / self.total_inventory)

        # Closure quality from ANGLE: 0° error → 1.0, 360° error → 0.0
        angle_err = layout.angle_error if hasattr(layout, 'angle_error') else 360.0
        angle_quality = max(0.0, 1.0 - angle_err / 360.0)

        # Closure quality from POSITION: 0 studs → 1.0, diagonal → 0.0
        closure_err = layout.closure_error if hasattr(layout, 'closure_error') else self.diagonal
        position_quality = max(0.0, 1.0 - closure_err / max(self.diagonal, 1.0))

        # Boundary quality: 0 violation → 1.0, large violation → 0.0
        # Penalizes layouts that exceed boundary — pushes toward compact shapes
        boundary_violation = self._compute_boundary_violation(layout)
        half_boundary = min(
            self.config.boundary.max_x - self.config.boundary.min_x,
            self.config.boundary.max_y - self.config.boundary.min_y,
        ) * 0.5
        boundary_quality = max(0.0, 1.0 - boundary_violation / max(half_boundary, 1.0))

        # Combined closure scale: ALL qualities must be good for full credit
        closure_scale = 0.3 + 0.7 * min(angle_quality, position_quality, boundary_quality)

        out["F"] = [-(utilization * closure_scale - loose_port_penalty)]

        # 4 hard constraints via Deb's CV (g <= 0 feasible)
        # Use main path closure (straight-through) for closure constraint.
        # Branch path closure depends on template geometry, not GA piece selection.
        main_path = layout.get_main_path() if hasattr(layout, 'get_main_path') else None
        closure_err = main_path.closure_error if main_path else layout.closure_error
        angle_err = main_path.angle_error if main_path else layout.angle_error
        g_closure = (closure_err - self.closure_tolerance) / self.closure_tolerance
        g_angle = (angle_err - self.angle_tolerance) / self.angle_tolerance
        g_boundary = self._compute_boundary_violation(layout) / max(self.diagonal, 1.0)
        g_inventory = float(self._compute_inventory_violation(layout))

        # 5th constraint: secondary loop closure (crossings)
        sec_err = getattr(layout, 'secondary_closure_error', 0.0)
        if sec_err > 0:
            g_secondary = (sec_err - self.closure_tolerance) / self.closure_tolerance
        else:
            g_secondary = -1.0  # Feasible when no crossing or no secondary loop

        out["G"] = [g_closure, g_angle, g_boundary, g_inventory, g_secondary]

    def _compute_boundary_violation(self, layout) -> float:
        """Compute max boundary violation across all paths.

        Args:
            layout: Decoded layout object.

        Returns:
            Maximum violation distance in studs.
        """
        max_violation = 0.0
        boundary = self.config.boundary

        for path in layout.paths:
            if len(path.states) == 0:
                continue

            x = path.states[:, 0]
            y = path.states[:, 1]

            path_max = max(
                np.max(np.maximum(0, boundary.min_x - x)),
                np.max(np.maximum(0, x - boundary.max_x)),
                np.max(np.maximum(0, boundary.min_y - y)),
                np.max(np.maximum(0, y - boundary.max_y)),
            )
            max_violation = max(max_violation, path_max)

        return float(max_violation)

    def _compute_inventory_violation(self, layout) -> int:
        """Compute inventory violation from layout pieces.

        Args:
            layout: Decoded layout object.

        Returns:
            Total count of pieces used beyond available inventory.
        """
        # Count pieces used in the layout
        piece_counts: Dict[int, int] = {}

        # Count main loop pieces
        for piece_idx in layout.main_loop_pieces:
            if piece_idx >= 0:
                piece_counts[piece_idx] = piece_counts.get(piece_idx, 0) + 1

        # Count branch pieces from switch pairs
        for switch_pair in layout.switch_pairs:
            for piece_idx in switch_pair.branch_pieces:
                if piece_idx >= 0:
                    piece_counts[piece_idx] = piece_counts.get(piece_idx, 0) + 1

        # Compute violation (excess usage)
        total_violation = 0
        for piece_idx, used in piece_counts.items():
            available = self.inventory_by_index.get(piece_idx, 0)
            if used > available:
                total_violation += used - available

        return total_violation


# =============================================================================
# Epsilon-Constraint Survival for BRKGA
# =============================================================================

from pymoo.core.callback import Callback
from pymoo.core.duplicate import DefaultDuplicateElimination, DuplicateElimination
from pymoo.core.survival import Survival


class EpsilonEliteSurvival(Survival):
    """BRKGA elite survival with ε-constraint relaxation.

    Mirrors pymoo's EliteSurvival but adjusts CV before ranking:
    solutions with cv <= epsilon are treated as ε-feasible (adjusted_cv=0)
    and compete by objective alone. This prevents premature convergence
    to trivial feasible solutions when feasibility is rare.

    When epsilon=0, behavior is identical to pymoo's EliteSurvival.
    """

    def __init__(self, n_elites, eliminate_duplicates=None):
        super().__init__(False)
        self.n_elites = n_elites
        self.eliminate_duplicates = eliminate_duplicates
        self.epsilon = 0.0  # Controlled by EpsilonDecayCallback

    def _do(self, problem, pop, n_survive=None, algorithm=None, **kwargs):
        # Duplicate elimination (mirrors pymoo EliteSurvival)
        if isinstance(self.eliminate_duplicates, bool) and self.eliminate_duplicates:
            pop = DefaultDuplicateElimination(func=lambda p: p.get("F")).do(pop)
        elif isinstance(self.eliminate_duplicates, DuplicateElimination):
            _, no_candidates, candidates = DefaultDuplicateElimination(
                func=lambda p: p.get("F")
            ).do(pop, return_indices=True)
            _, _, is_duplicate = self.eliminate_duplicates.do(
                pop[candidates], pop[no_candidates],
                return_indices=True, to_itself=False,
            )
            elim = set(np.array(candidates)[is_duplicate])
            pop = pop[[k for k in range(len(pop)) if k not in elim]]

        F = pop.get("F")
        cv = pop.get("cv")

        # ε-adjusted ranking: solutions with cv <= epsilon compete by objective
        adjusted_cv = np.where(cv <= self.epsilon, 0.0, cv)
        S = np.lexsort([F[:, 0], adjusted_cv])
        pop = pop[S]

        # Tag elite/non-elite for BRKGA's biased crossover
        elites = pop[:self.n_elites]
        non_elites = pop[self.n_elites:]
        elites.set("type", ["elite"] * len(elites))
        non_elites.set("type", ["non_elite"] * len(non_elites))

        return pop


class EpsilonDecayCallback(Callback):
    """Decay ε from initial value to 0 over generations.

    Schedule (from research Topic 2 - adaptive ε-constraint handling):
    - Phase A (0 to hold_until): Hold at ε₀ — broad exploration
    - Phase B (hold_until to decay_until): Linear decay to 0
    - Phase C (decay_until to end): Hold at 0 — strict feasibility

    ε₀ is auto-computed from the initial population's CV distribution
    (80th percentile of positive CVs, per Takahama & Sakai 2006).
    """

    def __init__(self, n_max_gen, hold_until=0.4, decay_until=0.8, min_epsilon_ratio=0.02):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.hold_until = hold_until
        self.decay_until = decay_until
        self.min_epsilon_ratio = min_epsilon_ratio  # Keep 2% of ε₀ to maintain objective pressure
        self.initial_epsilon = None
        self._logger = logging.getLogger(__name__)

    def notify(self, algorithm):
        # Access survival via algorithm (pymoo deep-copies the algorithm in minimize())
        survival = algorithm.survival
        if not hasattr(survival, 'epsilon'):
            return

        # Auto-initialize epsilon from population's CV distribution
        if self.initial_epsilon is None:
            cv = algorithm.pop.get("cv")
            if cv is not None and len(cv) > 0:
                positive_cv = cv[cv > 0]
                if len(positive_cv) > 0:
                    self.initial_epsilon = float(np.percentile(positive_cv, 80))
                else:
                    self.initial_epsilon = 1.0
            else:
                self.initial_epsilon = 1.0
            self._logger.info(f"ε-constraint: initial ε₀={self.initial_epsilon:.3f}")

        n_gen = algorithm.n_gen
        progress = n_gen / self.n_max_gen

        min_eps = self.initial_epsilon * self.min_epsilon_ratio

        if progress < self.hold_until:
            epsilon = self.initial_epsilon
        elif progress < self.decay_until:
            t = (progress - self.hold_until) / (self.decay_until - self.hold_until)
            epsilon = self.initial_epsilon * (1 - t)
        else:
            epsilon = min_eps  # Keep small ε to maintain objective-driven pressure

        survival.epsilon = max(epsilon, min_eps)


# =============================================================================
# Legacy Epsilon-Tightening Callback (kept for backward compatibility)
# =============================================================================


class EpsilonTightening(Callback):
    """Tighten closure tolerance over generations.

    Three-phase schedule designed to give complex layouts (switches, crossings)
    enough exploration time before tightening constraints:

    - Phase A (0-40%): Hold at initial tolerance — broad exploration
    - Phase B (40-80%): Linear tighten to final tolerance
    - Phase C (80-100%): Hold at final tolerance — polishing

    Example:
        callback = EpsilonTightening(initial_tol=8.0, final_tol=1.0)
        result = minimize(problem, algorithm, callback=callback)
    """

    def __init__(
        self,
        initial_tol: float = 8.0,
        final_tol: float = 1.0,
        hold_until: float = 0.4,
        tighten_until: float = 0.8,
    ):
        """Initialize epsilon-tightening callback.

        Args:
            initial_tol: Starting closure tolerance in studs (loose).
            final_tol: Final closure tolerance in studs (tight).
            hold_until: Progress fraction to hold at initial tolerance.
            tighten_until: Progress fraction at which tightening completes.
        """
        super().__init__()
        self.initial_tol = initial_tol
        self.final_tol = final_tol
        self.hold_until = hold_until
        self.tighten_until = tighten_until

    def notify(self, algorithm):
        """Called after each generation."""
        n_gen = algorithm.n_gen
        n_max_gen = getattr(algorithm.termination, 'n_max_gen', None)

        if n_max_gen is None:
            return

        progress = n_gen / n_max_gen

        if progress < self.hold_until:
            # Phase A: hold at initial (broad exploration)
            tolerance = self.initial_tol
        elif progress < self.tighten_until:
            # Phase B: linear tighten
            t = (progress - self.hold_until) / (self.tighten_until - self.hold_until)
            tolerance = self.initial_tol * (1 - t) + self.final_tol * t
        else:
            # Phase C: hold at final (polishing)
            tolerance = self.final_tol

        if hasattr(algorithm, 'problem'):
            algorithm.problem.closure_tolerance = tolerance


class StagnationCallback(Callback):
    """Detect stagnation and trigger hypermutation response.

    Monitors best feasible fitness over a sliding window. When no improvement
    is detected for `patience` generations, triggers population restart by
    injecting fresh random individuals into the worst portion of the population.

    Based on research recommendation: 50 gens no improvement -> hypermutation.
    """

    def __init__(
        self,
        patience: int = 50,
        inject_ratio: float = 0.10,
    ):
        """Initialize stagnation detector.

        Args:
            patience: Generations without improvement before triggering.
            inject_ratio: Fraction of population to replace with random.
        """
        super().__init__()
        self.patience = patience
        self.inject_ratio = inject_ratio
        self._best_fitness = float('inf')
        self._gens_without_improvement = 0
        self._logger = logging.getLogger(__name__)

    def notify(self, algorithm):
        """Called after each generation."""
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")

        if F is None or len(F) == 0:
            return

        # Track best OVERALL objective (not just feasible) to detect
        # improvement during ε-exploration when no feasible solutions exist
        current_best = float(np.min(F[:, 0]))

        # Check for improvement
        if current_best < self._best_fitness - 1e-8:
            self._best_fitness = current_best
            self._gens_without_improvement = 0
        else:
            self._gens_without_improvement += 1

        # Escalating stagnation response (research: shake + restart)
        stag = self._gens_without_improvement

        if stag == self.patience:
            # Tier 1: mild perturbation
            self._logger.info(
                f"Gen {algorithm.n_gen}: STAGNATION ({stag} gens). "
                f"Mild perturbation + injection."
            )
            self._inject_random(algorithm)
            self._perturb_elites(algorithm, n_perturb=10, n_genes=5, sigma=0.15)

        elif stag == self.patience * 2:
            # Tier 2: SHAKE — strong perturbation to escape local optimum
            self._logger.info(
                f"Gen {algorithm.n_gen}: DEEP STAGNATION ({stag} gens). "
                f"SHAKE: perturbing 50% of elites (20 genes) + re-randomizing non-elites."
            )
            self._shake(algorithm)

        elif stag == self.patience * 4:
            # Tier 3: RESTART — keep best 10, regenerate everything else
            self._logger.info(
                f"Gen {algorithm.n_gen}: EXTREME STAGNATION ({stag} gens). "
                f"RESTART: keeping top 10 elites, regenerating population."
            )
            self._restart(algorithm)
            self._gens_without_improvement = 0  # Reset counter after restart

        elif stag > self.patience and stag % self.patience == 0:
            # Recurring: inject fresh between escalation tiers
            self._logger.info(
                f"Gen {algorithm.n_gen}: Continued stagnation ({stag} gens). Re-injecting."
            )
            self._inject_random(algorithm)
            self._perturb_elites(algorithm)

    def _inject_random(self, algorithm):
        """Replace worst individuals with fresh random chromosomes.

        BRKGA-aware: only injects into non-elite positions to preserve
        BRKGA's elite partition integrity.
        """
        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")

        if X is None or F is None:
            return

        n_inject = max(1, int(len(X) * self.inject_ratio))

        # BRKGA tags individuals as "elite"/"non_elite" — only inject into non-elite
        try:
            types = pop.get("type")
        except Exception:
            types = None

        if types is not None:
            non_elite_mask = np.array([t == "non_elite" for t in types])
            non_elite_indices = np.where(non_elite_mask)[0]
            if len(non_elite_indices) == 0:
                return
            n_inject = min(n_inject, len(non_elite_indices))
            non_elite_F = F[non_elite_indices, 0]
            worst_local = np.argsort(non_elite_F)[-n_inject:]
            worst_indices = non_elite_indices[worst_local]
        else:
            # Fallback for non-BRKGA algorithms
            worst_indices = np.argsort(F[:, 0])[-n_inject:]

        # Generate closure-aware chromosomes (research: structured injection)
        # Falls back to uniform random if no sampler available
        sampler = getattr(algorithm, '_sampler_ref', None)
        xl = algorithm.problem.xl
        xu = algorithm.problem.xu

        for idx in worst_indices:
            if sampler is not None and hasattr(sampler, '_closure_aware_chromosome'):
                X[idx] = sampler._closure_aware_chromosome()
            else:
                X[idx] = np.random.uniform(xl, xu)

    def _perturb_elites(self, algorithm, n_perturb=10, n_genes=5, sigma=0.15):
        """Perturb a subset of elite chromosomes to escape local optima."""
        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")
        if X is None or F is None:
            return

        try:
            types = pop.get("type")
        except Exception:
            return

        if types is None:
            return

        elite_mask = np.array([t == "elite" for t in types])
        elite_indices = np.where(elite_mask)[0]
        if len(elite_indices) < 2:
            return

        # Perturb the WORST elites (not the best — preserve the best solution)
        elite_F = F[elite_indices, 0]
        worst_elite_order = np.argsort(elite_F)  # Best first (most negative)
        # Skip the best 5, perturb the next n_perturb
        start = min(5, len(elite_indices) - 1)
        perturb_indices = elite_indices[worst_elite_order[start:start + n_perturb]]

        for idx in perturb_indices:
            # Pick random gene positions to perturb (main loop genes 0-99)
            gene_positions = np.random.choice(100, size=min(n_genes, 100), replace=False)
            noise = np.random.normal(0, sigma, size=len(gene_positions))
            X[idx, gene_positions] = np.clip(X[idx, gene_positions] + noise, 0.0, 1.0)

    def _shake(self, algorithm):
        """SHAKE: strong perturbation of 50% of elites + re-randomize non-elites."""
        pop = algorithm.pop
        X = pop.get("X")
        if X is None:
            return

        try:
            types = pop.get("type")
        except Exception:
            return
        if types is None:
            return

        elite_mask = np.array([t == "elite" for t in types])
        elite_indices = np.where(elite_mask)[0]
        non_elite_indices = np.where(~elite_mask)[0]

        # Perturb 50% of elites with strong noise (20 genes, σ=0.25)
        n_shake = max(1, len(elite_indices) // 2)
        # Keep best 5 untouched
        F = pop.get("F")
        if F is not None:
            elite_F = F[elite_indices, 0]
            order = np.argsort(elite_F)
            shake_indices = elite_indices[order[5:5 + n_shake]]
        else:
            shake_indices = elite_indices[:n_shake]

        for idx in shake_indices:
            gene_positions = np.random.choice(100, size=20, replace=False)
            noise = np.random.normal(0, 0.25, size=20)
            X[idx, gene_positions] = np.clip(X[idx, gene_positions] + noise, 0.0, 1.0)

        # Re-randomize all non-elites
        sampler = getattr(algorithm, '_sampler_ref', None)
        xl = algorithm.problem.xl
        xu = algorithm.problem.xu
        for idx in non_elite_indices:
            if sampler is not None and hasattr(sampler, '_closure_aware_chromosome'):
                X[idx] = sampler._closure_aware_chromosome()
            else:
                X[idx] = np.random.uniform(xl, xu)

    def _restart(self, algorithm):
        """RESTART: keep top 10 elites, regenerate everything else."""
        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")
        if X is None or F is None:
            return

        # Keep top 10 by objective
        top_indices = np.argsort(F[:, 0])[:10]
        top_set = set(top_indices)

        # Regenerate all others
        sampler = getattr(algorithm, '_sampler_ref', None)
        xl = algorithm.problem.xl
        xu = algorithm.problem.xu
        for idx in range(len(X)):
            if idx not in top_set:
                if sampler is not None and hasattr(sampler, '_closure_aware_chromosome'):
                    X[idx] = sampler._closure_aware_chromosome()
                else:
                    X[idx] = np.random.uniform(xl, xu)


# =============================================================================
# RK-Space Local Search Callback (Research Topic 3 / Improvements Rank 1)
# =============================================================================


class LocalSearchCallback(Callback):
    """Lamarckian local search on elite RK vectors.

    After each generation, applies neighborhood moves to elite chromosomes.
    If a move improves penalized fitness, the chromosome is updated in-place
    (Lamarckian writeback). This helps BRKGA make the fine-grained changes
    its biased crossover can't achieve.

    Four RK-space moves (from Chaves, Resende 2024 — RKO framework):
    - SWAP: exchange two gene values
    - CHANGE: replace one gene with random [0,1]
    - SHIFT: move gene from pos i to pos j
    - INVERSION: reverse subsequence [i..j]
    """

    def __init__(
        self,
        problem,
        every_n_gen: int = 1,
        n_elite_frac: float = 0.1,
        max_steps: int = 3,
    ):
        super().__init__()
        self.problem = problem
        self.every_n_gen = every_n_gen
        self.n_elite_frac = n_elite_frac
        self.max_steps = max_steps
        self._logger = logging.getLogger(__name__)
        self._improvements = 0
        self._total_evals = 0

    def notify(self, algorithm):
        if algorithm.n_gen % self.every_n_gen != 0:
            return

        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")
        if X is None or F is None:
            return

        try:
            types = pop.get("type")
        except Exception:
            return
        if types is None:
            return

        elite_mask = np.array([t == "elite" for t in types])
        elite_indices = np.where(elite_mask)[0]
        if len(elite_indices) == 0:
            return

        # Select top fraction of elites for LS (best fitness first)
        n_select = max(1, int(len(elite_indices) * self.n_elite_frac))
        elite_F = F[elite_indices, 0]
        best_order = np.argsort(elite_F)
        selected = elite_indices[best_order[:n_select]]

        improvements_this_gen = 0
        evals_this_gen = 0

        for idx in selected:
            current_x = X[idx].copy()
            current_fitness = self._evaluate_penalized(current_x)
            evals_this_gen += 1

            for _ in range(self.max_steps):
                # Choose random move (biased toward simple moves)
                move = np.random.choice(
                    ['swap', 'change', 'shift', 'inversion'],
                    p=[0.35, 0.35, 0.20, 0.10],
                )
                candidate = self._apply_move(current_x, move)
                candidate_fitness = self._evaluate_penalized(candidate)
                evals_this_gen += 1

                if candidate_fitness < current_fitness - 1e-8:
                    # Improvement — Lamarckian writeback
                    current_x = candidate
                    current_fitness = candidate_fitness
                    improvements_this_gen += 1

            # Write back improved chromosome
            X[idx] = current_x

        self._improvements += improvements_this_gen
        self._total_evals += evals_this_gen

        if improvements_this_gen > 0 and algorithm.n_gen % 50 == 0:
            self._logger.info(
                f"Gen {algorithm.n_gen}: LS improved {improvements_this_gen}/{n_select} elites "
                f"({self._total_evals} total LS evals)"
            )

    def _evaluate_penalized(self, x: NDArray) -> float:
        """Evaluate chromosome with penalized fitness for LS comparison."""
        out = {}
        self.problem._evaluate(x, out)
        f = out["F"][0]
        g = out.get("G", [])
        cv = sum(max(0, gi) for gi in g) if g else 0.0
        return f + 10.0 * cv  # Penalized: objective + 10× constraint violation

    def _apply_move(self, x: NDArray, move: str) -> NDArray:
        """Apply an RK-space neighborhood move to chromosome copy.

        Biased toward later gene positions (30-99) where changes affect
        fewer downstream construction decisions.
        """
        candidate = x.copy()
        n_genes = 100  # Main loop genes only

        # Bias position selection toward later genes
        weights = np.arange(1, n_genes + 1, dtype=np.float64)
        weights /= weights.sum()

        if move == 'swap':
            i = np.random.choice(n_genes, p=weights)
            j = np.random.choice(n_genes, p=weights)
            candidate[i], candidate[j] = candidate[j], candidate[i]

        elif move == 'change':
            i = np.random.choice(n_genes, p=weights)
            candidate[i] = np.random.uniform(0, 1)

        elif move == 'shift':
            i = np.random.choice(n_genes, p=weights)
            j = np.random.choice(n_genes, p=weights)
            val = candidate[i]
            candidate = np.delete(candidate, i)
            candidate = np.insert(candidate, min(j, len(candidate)), val)

        elif move == 'inversion':
            i = np.random.choice(n_genes, p=weights)
            j = np.random.choice(n_genes, p=weights)
            lo, hi = min(i, j), max(i, j)
            candidate[lo:hi+1] = candidate[lo:hi+1][::-1]

        return candidate


# =============================================================================
# Convergence Tracker Callback
# =============================================================================


class ConvergenceTracker(Callback):
    """Track optimization convergence metrics per generation.

    Records best objective value and feasibility percentage at each generation.
    Based on research best practices for monitoring GA progress.

    Attributes:
        data: Dictionary with tracked metrics:
            - 'generation': List of generation numbers
            - 'best_f': List of best objective values (negated for maximization)
            - 'feasible_pct': List of feasibility percentages (0-1)
            - 'n_pieces': List of best piece counts
            - 'avg_cv': List of average constraint violations

    Example:
        tracker = ConvergenceTracker()
        result = minimize(problem, algorithm, callback=tracker)
        # Access data after optimization:
        print(tracker.data['best_f'])
        tracker.plot_convergence()  # Optional visualization
    """

    def __init__(self):
        """Initialize convergence tracker."""
        super().__init__()
        self.data: Dict[str, List] = {
            "generation": [],
            "best_f": [],
            "feasible_pct": [],
            "n_pieces": [],
            "avg_cv": [],
        }

    def notify(self, algorithm):
        """Called after each generation to record metrics.

        Args:
            algorithm: The pymoo algorithm instance.
        """
        pop = algorithm.pop
        n_gen = algorithm.n_gen

        # Get objective and constraint values
        F = pop.get("F")
        G = pop.get("G")

        if F is None or len(F) == 0:
            return

        # Best objective (F is negated utilization, so min is best)
        best_f = float(np.min(F[:, 0]))

        # Best piece count (F[0] = -utilization = -n_pieces/total)
        # Assuming we can access problem's total_inventory
        if hasattr(algorithm, 'problem') and hasattr(algorithm.problem, 'total_inventory'):
            total = algorithm.problem.total_inventory
            best_pieces = int(-best_f * total)
        else:
            best_pieces = 0

        # Feasibility percentage
        if G is not None and len(G) > 0:
            feasible = np.all(G <= 0, axis=1)
            feasible_pct = float(np.mean(feasible))
            avg_cv = float(np.mean(np.maximum(0, G).sum(axis=1)))
        else:
            feasible_pct = 1.0
            avg_cv = 0.0

        # Record metrics
        self.data["generation"].append(n_gen)
        self.data["best_f"].append(best_f)
        self.data["feasible_pct"].append(feasible_pct)
        self.data["n_pieces"].append(best_pieces)
        self.data["avg_cv"].append(avg_cv)

    def get_summary(self) -> Dict:
        """Get summary statistics from tracked data.

        Returns:
            Dictionary with summary metrics.
        """
        if not self.data["generation"]:
            return {}

        return {
            "final_generation": self.data["generation"][-1],
            "best_objective": min(self.data["best_f"]),
            "final_feasible_pct": self.data["feasible_pct"][-1],
            "max_feasible_pct": max(self.data["feasible_pct"]),
            "best_n_pieces": max(self.data["n_pieces"]),
            "convergence_generation": self._find_convergence_gen(),
        }

    def _find_convergence_gen(self, window: int = 10, threshold: float = 0.001) -> int:
        """Find generation where optimization converged.

        Convergence defined as when best_f stops improving significantly.

        Args:
            window: Number of generations to check for improvement.
            threshold: Minimum improvement to consider as progress.

        Returns:
            Generation number where convergence was detected, or -1 if not converged.
        """
        if len(self.data["best_f"]) < window:
            return -1

        for i in range(window, len(self.data["best_f"])):
            recent = self.data["best_f"][i - window:i]
            improvement = abs(max(recent) - min(recent))
            if improvement < threshold:
                return self.data["generation"][i - window]

        return -1


# =============================================================================
# Aliases for backward compatibility (will be removed in future)
# =============================================================================

# These are kept temporarily so existing code doesn't break immediately
MultiSegmentProblem = TrackOptimizationProblem
SingleObjectiveProblem = TrackOptimizationProblem
TrackLayoutProblem = TrackOptimizationProblem
