"""Tests for bi-objective NSGA-II problem definition."""

import numpy as np
import pytest

from src.problem import TrackOptimizationProblem
from src.encoding import (
    compute_dimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    PartitionedDimensions,
    PieceIndex,
)


class TestTrackOptimizationProblem:
    """Tests for TrackOptimizationProblem (bi-objective with Deb's CV)."""

    def test_problem_dimensions(self, catalog, default_config):
        """Verify problem has correct dimensions."""
        problem = TrackOptimizationProblem(catalog, default_config)

        assert problem.n_var == problem.dims.n_var
        assert problem.n_obj == 2  # utilization + speed
        # Stage B: 5 base constraints + per-type inventory (one per catalog piece)
        assert problem.n_ieq_constr == 5 + catalog.n_pieces

    def test_evaluate_valid_circle(self, catalog, default_config):
        """16 R40 circle evaluates without error."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        assert "F" in out
        assert "G" in out
        assert len(out["F"]) == 2  # Two objectives
        # Stage B: G has 5 + n_piece_types entries
        assert len(out["G"]) == 5 + catalog.n_pieces

    def test_evaluate_empty_chromosome(self, catalog, default_config):
        """Empty chromosome evaluates without error and emits infeasibility sentinel."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        chromosome = create_empty_chromosome(dims)
        out = {}

        problem._evaluate(chromosome, out)

        assert "F" in out
        assert "G" in out
        # Infeasibility sentinel: empty layout → +inf F (infeasibles never dominate)
        assert np.isinf(out["F"][0]) and out["F"][0] > 0
        assert np.isinf(out["F"][1]) and out["F"][1] > 0

    def test_objectives_shape(self, catalog, default_config):
        """out['F'] has shape (2,) for bi-objective."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        F = np.array(out["F"])
        assert F.shape == (2,)

    def test_constraints_shape(self, catalog, default_config):
        """out['G'] has shape (5 + n_piece_types,) after Stage B."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        G = np.array(out["G"])
        assert G.shape == (5 + catalog.n_pieces,)

    def test_objective_correct_sign(self, catalog, default_config):
        """Both objectives have correct sign for minimization (negative)."""
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        # F[0] = -utilization (negative to maximize)
        assert out["F"][0] < 0
        # F[1] = -(slowest route's avg speed): negative for any closed loop.
        assert out["F"][1] < 0

    def test_feasible_circle_constraints(self, catalog, default_config):
        """Closed circle should satisfy closure, boundary, collision, and inventory constraints.

        Stage B G layout:
          G[0..2]: closure_x / closure_y / closure_theta
          G[3]: boundary
          G[4]: collisions
          G[5..4+T]: per-type inventory excess
        """
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        pattern = [PieceIndex.R40_CURVE] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern)
        out = {}

        problem._evaluate(chromosome, out)

        G = np.asarray(out["G"])
        assert G[0] <= 0  # Closure X satisfied
        assert G[1] <= 0  # Closure Y satisfied
        assert G[2] <= 0  # Closure theta satisfied
        assert G[3] <= 0  # Boundary satisfied
        assert G[4] <= 0  # No collisions
        # All per-type inventory entries satisfied (16 R40_CURVE within 20-piece cap)
        assert np.all(G[5:] <= 0)

    def test_closure_tolerance_can_be_set(self, catalog, default_config):
        """Closure tolerance can be overridden."""
        loose = TrackOptimizationProblem(catalog, default_config, closure_tolerance=10.0)
        tight = TrackOptimizationProblem(catalog, default_config, closure_tolerance=0.1)

        assert loose.closure_tolerance == 10.0
        assert tight.closure_tolerance == 0.1

    def test_oval_route_pace_exceeds_circle(self, catalog, default_config):
        """Adding straights raises the route PACE under the slowest-route objective.

        F[1] = -(slowest route's avg speed). The oval accelerates along its
        straights, so its overall pace exceeds the all-curve circle's. (Under
        the old per-segment-bottleneck objective the two were equal; under pace
        semantics they differ.)
        """
        problem = TrackOptimizationProblem(catalog, default_config)
        dims = problem.dims

        circle = create_chromosome_from_pieces(dims, [PieceIndex.R40_CURVE] * 16)
        out_circle = {}
        problem._evaluate(circle, out_circle)

        oval = create_chromosome_from_pieces(dims, [
            PieceIndex.R40_CURVE] * 8 +
            [PieceIndex.STRAIGHT_16] * 4 +
            [PieceIndex.R40_CURVE] * 8 +
            [PieceIndex.STRAIGHT_16] * 4,
        )
        out_oval = {}
        problem._evaluate(oval, out_oval)

        # If oval didn't close (Stage B: G[0]=closure_x, G[1]=closure_y), comparison doesn't apply.
        if out_oval["G"][0] <= 0 and out_oval["G"][1] <= 0:
            # -F[1] is speed; a faster route is a more-negative F[1].
            assert out_oval["F"][1] < out_circle["F"][1] - 1e-6, (
                f"oval pace must exceed circle pace: circle F[1]={out_circle['F'][1]}, "
                f"oval F[1]={out_oval['F'][1]}"
            )


class TestInfeasibilitySentinel:
    """Empty/infeasible layouts must emit +inf F, not 0.0.

    F=[+inf, +inf] keeps a zero-piece layout from looking like a valid
    solution and dominating real ones in F-space.
    """

    def test_empty_layout_f_is_positive_infinity(self, catalog, default_config):
        import numpy as np
        from src.problem import TrackOptimizationProblem

        problem = TrackOptimizationProblem(catalog, default_config)
        # All-sentinel chromosome → empty main loop
        x = problem.xl.astype(int)
        out = {}
        problem._evaluate(x, out)
        assert np.isinf(out["F"][0]), f"F[0] should be +inf, got {out['F'][0]}"
        assert np.isinf(out["F"][1]), f"F[1] should be +inf, got {out['F'][1]}"
        assert out["F"][0] > 0 and out["F"][1] > 0, "sentinel must be POSITIVE infinity"

    def test_empty_layout_g_is_large_finite(self, catalog, default_config):
        import numpy as np
        from src.problem import TrackOptimizationProblem

        problem = TrackOptimizationProblem(catalog, default_config)
        x = problem.xl.astype(int)
        out = {}
        problem._evaluate(x, out)
        G = np.asarray(out["G"])
        assert np.all(np.isfinite(G)), f"G must be finite: {G}"
        assert np.all(G > 0), f"All G entries must be positive for infeasible: {G}"
        assert G.max() >= 1e5, (
            f"Sentinel too small; CV won't dominate real violations: max={G.max()}"
        )
        assert len(G) == problem.n_ieq_constr, f"G length mismatch"

    def test_infeasibility_sentinel_dominated_by_feasible(self):
        """An infeasible individual (F=+inf) must lose to feasible in dominance comparison."""
        import numpy as np
        from pymoo.util.dominator import Dominator

        F_feas = np.array([-0.5, -1.0])       # feasible
        F_infeas = np.array([np.inf, np.inf])  # sentinel
        dom = Dominator.get_relation(F_feas, F_infeas)
        assert dom == 1, (
            f"Feasible should dominate infeasible, got relation={dom}"
        )


class TestF1SlowestRoutePace:
    """F[1] = -(avg speed of the slowest traversal route).

    Per-segment derailment safety is enforced inside the speed profile, so the
    objective shapes throughput: each route's overall pace (avg_speed), worst
    route wins. For a single-route layout F[1] is just that loop's pace.
    """

    def test_f1_reflects_route_pace_not_bottleneck(self, catalog, default_config):
        """F[1] for a single-route closed layout equals -avg_speed at the margin.

        Uses an oval (curves + straights) so avg_speed and min_speed diverge —
        the forward-backward pass lets the train accelerate on the straights,
        pulling the route PACE (avg) above the per-segment bottleneck (min).
        F[1] must track the pace (avg), proving it is the route speed and not
        the worst-segment speed.
        """
        import numpy as np
        from src.problem import TrackOptimizationProblem, SPEED_SAFETY_MARGIN
        from src.train import compute_speed_profile
        from src.decoder import decode_chromosome
        from src.encoding import create_chromosome_from_pieces, PieceIndex

        problem = TrackOptimizationProblem(catalog, default_config)
        # Oval: 8 R40 + 4 straight + 8 R40 + 4 straight (closes, mixes curve/straight).
        pattern = (
            [PieceIndex.R40_CURVE] * 8
            + [PieceIndex.STRAIGHT_16] * 4
            + [PieceIndex.R40_CURVE] * 8
            + [PieceIndex.STRAIGHT_16] * 4
        )
        x = create_chromosome_from_pieces(problem.dims, pattern)
        out = {}
        problem._evaluate(x, out)

        if not np.isfinite(out["F"][1]):
            pytest.skip("Test chromosome didn't decode to a non-empty layout")

        layout = decode_chromosome(
            x, catalog, default_config.inventory,
            dims=problem.dims, config=problem.decoder_config,
        )
        profile = compute_speed_profile(
            layout, catalog, train_config=problem._train_config,
            safety_margin=SPEED_SAFETY_MARGIN,
        )
        # Precondition: avg and min differ so the test distinguishes pace from bottleneck.
        assert profile.avg_speed > profile.min_speed + 1e-6, (
            f"Test precondition broken: avg={profile.avg_speed}, min={profile.min_speed} "
            f"are too close to distinguish pace vs bottleneck."
        )
        expected = -profile.avg_speed
        assert abs(out["F"][1] - expected) < 1e-9, (
            f"F[1]={out['F'][1]}, expected -avg_speed={expected} "
            f"(per-segment min was {-profile.min_speed})."
        )

    def test_f1_for_closed_r40_circle_is_friction_bound(self, catalog, default_config):
        """For an all-R40 circle avg==min (no straights to accelerate on), so
        F[1] = -avg_speed lands in the curve-bound range, strictly below the
        1.57 m/s straight speed cap.
        """
        import numpy as np
        from src.problem import TrackOptimizationProblem, SPEED_SAFETY_MARGIN
        from src.train import compute_speed_profile
        from src.decoder import decode_chromosome

        problem = TrackOptimizationProblem(catalog, default_config)
        x = problem.xl.astype(np.int32).copy()
        x[:16] = 2  # R40_CURVE
        out = {}
        problem._evaluate(x, out)
        if not np.isfinite(out["F"][1]):
            pytest.skip("Test chromosome didn't decode to a non-empty layout")

        layout = decode_chromosome(
            x, catalog, default_config.inventory,
            dims=problem.dims, config=problem.decoder_config,
        )
        profile = compute_speed_profile(
            layout, catalog, train_config=problem._train_config,
            safety_margin=SPEED_SAFETY_MARGIN,
        )
        assert abs(out["F"][1] - (-profile.avg_speed)) < 1e-9, (
            f"F[1]={out['F'][1]} must equal -avg_speed={-profile.avg_speed}; "
            f"min_speed was {profile.min_speed}"
        )
        # Friction-bound curves are strictly slower than the straight cap (1.57 m/s).
        assert out["F"][1] > -1.5, (
            f"F[1]={out['F'][1]}: a pure-curve circle cannot approach the straight cap."
        )


class TestF1MultiPathSlowestRoute:
    """F[1] is the average speed of the layout's SLOWEST traversal route.

    The speed objective must "see" branch geometry: every one of the 2^J routes
    (take/skip each siding) is scored, and F[1] is the slowest route's overall
    pace (avg_speed at the operating safety margin). A switchless layout has one
    route, so F[1] reduces to that loop's pace.
    """

    def test_f1_equals_slowest_route_pace(self, switches_config, catalog):
        import numpy as np
        from src.problem import TrackOptimizationProblem, SPEED_SAFETY_MARGIN
        from src.decoder import decode_chromosome
        from src.geometry import Layout
        from src.train import compute_speed_profile
        from src.encoding import create_chromosome_from_pieces
        from src.sampling import _gen_oval_with_siding

        problem = TrackOptimizationProblem(catalog, switches_config)
        dims = problem.dims

        inv = {
            catalog._id_to_index[k]: v
            for k, v in switches_config.inventory.items()
            if k in catalog._id_to_index
        }
        variants = _gen_oval_with_siding(inv, dims)
        assert variants, "seeder produced no oval+siding variant for this inventory"
        pieces, flips, junctions, *_ = variants[0]
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, junctions=junctions,
        )

        out = {}
        problem._evaluate(x, out)
        if not np.isfinite(out["F"][1]):
            pytest.skip("siding chromosome didn't decode to a non-empty layout")

        layout = decode_chromosome(
            x, catalog, switches_config.inventory,
            dims=dims, config=problem.decoder_config,
        )
        assert layout.n_switch_pairs >= 1 and layout.n_paths >= 2, (
            "test needs a genuine multipath layout to distinguish all-routes scoring"
        )

        # Independently recompute each route's overall pace at the problem's margin.
        route_paces = []
        for p in layout.paths:
            if len(p.piece_sequence) == 0:
                continue
            view = Layout(
                indices=np.asarray(p.piece_sequence, dtype=np.int32),
                states=p.states,
            )
            prof = compute_speed_profile(
                view, catalog, train_config=problem._train_config,
                safety_margin=SPEED_SAFETY_MARGIN,
            )
            route_paces.append(prof.avg_speed)

        expected = -min(route_paces)
        assert abs(out["F"][1] - expected) < 1e-9, (
            f"F[1]={out['F'][1]} must equal -min(per-route avg pace)={expected}; "
            f"route paces={route_paces}"
        )

    def test_single_path_reduces_to_its_own_pace(self, switches_config, catalog):
        """A switchless layout under the slowest-route objective == its own pace."""
        import numpy as np
        from src.problem import TrackOptimizationProblem, SPEED_SAFETY_MARGIN
        from src.decoder import decode_chromosome
        from src.train import compute_speed_profile
        from src.encoding import create_chromosome_from_pieces, PieceIndex

        problem = TrackOptimizationProblem(catalog, switches_config)
        x = create_chromosome_from_pieces(problem.dims, [PieceIndex.R40_CURVE] * 16)
        out = {}
        problem._evaluate(x, out)
        if not np.isfinite(out["F"][1]):
            pytest.skip("circle didn't decode to a non-empty layout")

        layout = decode_chromosome(
            x, catalog, switches_config.inventory,
            dims=problem.dims, config=problem.decoder_config,
        )
        assert layout.n_paths == 1
        profile = compute_speed_profile(
            layout, catalog, train_config=problem._train_config,
            safety_margin=SPEED_SAFETY_MARGIN,
        )
        assert abs(out["F"][1] - (-profile.avg_speed)) < 1e-9


class TestGShapeV2:
    """G has 5 + n_piece_types entries."""

    def test_n_ieq_constr_is_5_plus_piece_types(self, catalog, default_config):
        from src.problem import TrackOptimizationProblem
        problem = TrackOptimizationProblem(catalog, default_config)
        expected = 5 + catalog.n_pieces
        assert problem.n_ieq_constr == expected, (
            f"n_ieq_constr={problem.n_ieq_constr}, expected {expected}"
        )

    def test_g_length_matches_n_ieq_constr(self, catalog, default_config):
        """Every evaluation must produce a G of length n_ieq_constr."""
        import numpy as np
        from src.problem import TrackOptimizationProblem

        problem = TrackOptimizationProblem(catalog, default_config)
        pattern = [PieceIndex.R40_CURVE] * 16
        x = create_chromosome_from_pieces(problem.dims, pattern)
        out = {}
        problem._evaluate(x, out)
        G = np.asarray(out["G"])
        assert len(G) == problem.n_ieq_constr

    def test_g_entries_order_and_ranges(self, catalog, default_config):
        """Verify G[0..2]=closure_x/y/theta, G[3]=boundary, G[4]=collisions,
        G[5:]=per-type inventory."""
        import numpy as np
        from src.problem import TrackOptimizationProblem

        problem = TrackOptimizationProblem(catalog, default_config)
        # Use helper so start position defaults to (0, 0) — centers the circle
        # well inside the +/-150 boundary box.
        pattern = [PieceIndex.R40_CURVE] * 16
        x = create_chromosome_from_pieces(problem.dims, pattern)
        out = {}
        problem._evaluate(x, out)
        G = np.asarray(out["G"])

        # Closed circle -> all closure entries satisfied
        assert G[0] <= 0.0, f"G[0] closure_x={G[0]} should be <= 0 for closed circle"
        assert G[1] <= 0.0, f"G[1] closure_y={G[1]} should be <= 0 for closed circle"
        assert G[2] <= 0.0, f"G[2] closure_theta={G[2]} should be <= 0 for closed circle"
        # Default config has plus/minus 150 boundary; R40 circle (radius 40) fits comfortably
        assert G[3] <= 0.0, f"G[3] boundary={G[3]} should be <= 0 for small circle in large box"
        # No collisions in a simple circle
        assert G[4] <= 0.0, f"G[4] collisions={G[4]} should be <= 0"
        # Inventory entries: R40_CURVE (index 2) uses 16 pieces; default config has 20 -> feasible
        inv_entries = G[5:]
        assert len(inv_entries) == catalog.n_pieces, (
            f"inventory entries count {len(inv_entries)} != n_pieces {catalog.n_pieces}"
        )
        assert np.all(inv_entries <= 0.0), (
            f"All per-type inventory entries should be <= 0 for feasible circle; got {inv_entries}"
        )

    def test_empty_layout_sentinel_g_matches_new_length(self, catalog, default_config):
        """The +inf sentinel must fill a G of the new length."""
        import numpy as np
        from src.problem import TrackOptimizationProblem

        problem = TrackOptimizationProblem(catalog, default_config)
        x = problem.xl.astype(np.int32)
        out = {}
        problem._evaluate(x, out)
        G = np.asarray(out["G"])
        assert len(G) == 5 + catalog.n_pieces
        assert np.all(G == 1e6), f"Sentinel should fill all G entries with 1e6, got {G}"


class TestFeasibilityParity:
    """Stage B must NOT regress existing configs."""

    @pytest.mark.parametrize(
        "config_name", ["default", "with_switches", "with_crossing", "compact"]
    )
    def test_config_smoke_still_runs(self, config_name):
        """Each config's --quick-test must exit 0 (feasible count is config-dependent)."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "main.py", "--config", f"configs/{config_name}.yaml", "--quick-test"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"{config_name} quick-test failed: returncode={result.returncode}\n"
            f"stdout (last 500):\n{result.stdout[-500:]}\n"
            f"stderr (last 300):\n{result.stderr[-300:]}"
        )
