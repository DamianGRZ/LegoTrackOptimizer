"""Tests for bi-objective NSGA-II problem definition."""

import numpy as np
import pytest

from src.config import BoundaryConfig
from src.problem import TrackOptimizationProblem
from src.encoding import (
    create_chromosome_from_pieces,
    create_empty_chromosome,
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
        # F[1] = expected traversal time in seconds: positive, minimized directly.
        assert out["F"][1] > 0
        assert np.isfinite(out["F"][1])

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

    def test_oval_takes_longer_than_circle(self, catalog, default_config):
        """Adding straights lengthens the network, so traversal time grows.

        F[1] = expected traversal time. The oval is the circle plus 8 straights
        — more track to cover — so its F[1] must exceed the circle's. This is
        the conflict with utilization the objective exists to create.
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
            assert out_oval["F"][1] > out_circle["F"][1] + 1e-6, (
                f"more track must take longer: circle F[1]={out_circle['F'][1]}, "
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


class TestF1SingleRouteTime:
    """F[1] = expected traversal time; on a plain loop it is that loop's lap time.

    Per-segment derailment safety is enforced inside the speed profile: the
    3-pass profiler brakes for curves and accelerates on straights, and F[1]
    integrates the resulting per-segment times over the network.
    """

    def test_f1_is_profile_time_not_bottleneck_time(self, catalog, default_config):
        """F[1] for a single-route closed layout equals the profile's lap_time.

        Uses an oval (curves + straights) so avg_speed and min_speed diverge —
        covering the distance at the bottleneck (min) speed would take strictly
        longer than the accel/brake-aware profile time. F[1] must equal the
        profile's lap_time, proving it integrates the 3-pass dynamics rather
        than dividing distance by the worst-segment speed.
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
        # Precondition: avg and min differ so profile time and bottleneck time differ.
        assert profile.avg_speed > profile.min_speed + 1e-6, (
            f"Test precondition broken: avg={profile.avg_speed}, min={profile.min_speed} "
            f"are too close to distinguish profile time from bottleneck time."
        )
        assert abs(out["F"][1] - profile.lap_time) < 1e-9, (
            f"F[1]={out['F'][1]}, expected lap_time={profile.lap_time}."
        )
        bottleneck_time = profile.total_distance / profile.min_speed
        assert out["F"][1] < bottleneck_time - 1e-6, (
            f"F[1]={out['F'][1]} must beat distance/min_speed={bottleneck_time}: "
            f"the profiler accelerates on straights."
        )

    def test_f1_for_closed_r40_circle_is_friction_bound(self, catalog, default_config):
        """An all-R40 circle runs curve-bound, so its time exceeds the
        straight-cap time for the same distance (no straights to accelerate on).
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
        assert abs(out["F"][1] - profile.lap_time) < 1e-9, (
            f"F[1]={out['F'][1]} must equal lap_time={profile.lap_time}; "
            f"min_speed was {profile.min_speed}"
        )
        # Curve-bound: strictly slower than covering the distance at the
        # motor-capped straight speed.
        straight_cap_time = profile.total_distance / (1.57 * SPEED_SAFETY_MARGIN)
        assert out["F"][1] > straight_cap_time, (
            f"F[1]={out['F'][1]}: a pure-curve circle cannot approach "
            f"straight-cap pace ({straight_cap_time})."
        )


class TestF1WholeGraphTime:
    """F[1] covers the WHOLE physical network, not just one route.

    Every one of the 2^J routes (take/skip each siding) is profiled; each
    physical piece is charged the mean of its traversal times across all
    passages (identity via piece_uids), and F[1] sums over distinct pieces.
    A plain loop has one route covering each piece once, so F[1] reduces to
    its lap time; a self-crossing one charges the crossing once for two
    passages and lands below it.
    """

    def test_f1_equals_per_piece_mean_over_routes(self, switches_config, catalog):
        """Definition check: F[1] == independent per-piece mean recomputation.

        On an oval+siding both routes and both exclusive piece sets (branch,
        bypassed straights) are charged, so F[1] strictly exceeds every single
        route's lap time — the whole-graph property the slowest-route
        objective lacked.
        """
        import numpy as np
        from src.problem import TrackOptimizationProblem, SPEED_SAFETY_MARGIN
        from src.decoder import decode_chromosome
        from src.geometry import Layout
        from src.train import compute_speed_profile
        from src.encoding import create_chromosome_from_pieces
        from src.sampling import _gen_oval_with_siding

        problem = TrackOptimizationProblem(catalog, switches_config)
        dims = problem.dims

        inv = catalog.inventory_by_index(switches_config.inventory)
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
            "test needs a genuine multipath layout to distinguish whole-graph scoring"
        )

        # Independent recomputation: per-route profiles at the problem's margin,
        # segment times pooled per physical piece, mean per piece, summed.
        sums: dict = {}
        passages: dict = {}
        route_times = []
        for p in layout.paths:
            if len(p.piece_sequence) == 0:
                continue
            indices = np.asarray(p.piece_sequence, dtype=np.int32)
            view = Layout(
                indices=indices,
                states=p.states,
                route_indices=np.asarray(p.route_indices, dtype=np.int32),
            )
            prof = compute_speed_profile(
                view, catalog, train_config=problem._train_config,
                safety_margin=SPEED_SAFETY_MARGIN,
                closure_pos_tol=problem.closure_tolerance,
                closure_angle_tol=problem.angle_tolerance,
            )
            route_times.append(prof.lap_time)
            arc_m = catalog.get_route_arc_lengths(
                indices, np.asarray(p.route_indices, dtype=np.int32),
            ) * catalog.stud_mm / 1000.0
            seg_times = arc_m / np.where(prof.speeds > 0, prof.speeds, 0.001)
            for uid, seg_time in zip(p.piece_uids, seg_times, strict=True):
                sums[uid] = sums.get(uid, 0.0) + float(seg_time)
                passages[uid] = passages.get(uid, 0) + 1

        expected = sum(sums[uid] / passages[uid] for uid in sums)
        assert abs(out["F"][1] - expected) < 1e-9, (
            f"F[1]={out['F'][1]} must equal per-piece mean sum={expected}"
        )
        assert out["F"][1] > max(route_times) + 1e-6, (
            f"whole-graph time {out['F'][1]} must exceed every single route's "
            f"lap time {route_times}: bypassed straights and branch both count"
        )

    def test_single_path_reduces_to_lap_time(self, switches_config, catalog):
        """A plain loop's F[1] is exactly its one route's lap time."""
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
        assert abs(out["F"][1] - profile.lap_time) < 1e-9


class TestF1ShapeSensitivity:
    """Layout SHAPE moves F[1] at a fixed piece count (equal utilization).

    This is what makes the problem genuinely bi-objective: time is not a
    monotone function of piece count, so the GA has a second dimension to
    search at every utilization level.
    """

    def test_chicane_slows_equal_piece_loop(self, catalog, default_config):
        """Two 32-piece loops, same F[0]; the curve-heavy one takes longer.

        Both loops are centrally symmetric [arc, run, arc, run] x 2 shapes, so
        they close exactly. The plain variant's runs are 4 straights; the
        chicane variant replaces two opposite runs with zero-net-turn R+R+L+L
        chicanes — same piece count, more braking.
        """
        problem = TrackOptimizationProblem(catalog, default_config)
        arc_pieces, arc_flips = [PieceIndex.R40_CURVE] * 4, [0] * 4
        straight_run = ([PieceIndex.STRAIGHT_16] * 4, [0] * 4)
        chicane_run = ([PieceIndex.R40_CURVE] * 4, [1, 1, 0, 0])

        def loop(run_a, run_b):
            pieces_a, flips_a = run_a
            pieces_b, flips_b = run_b
            pieces = (arc_pieces + pieces_a + arc_pieces + pieces_b) * 2
            flips = (arc_flips + flips_a + arc_flips + flips_b) * 2
            return create_chromosome_from_pieces(problem.dims, pieces, main_loop_flips=flips)

        out_plain, out_chicane = {}, {}
        problem._evaluate(loop(straight_run, straight_run), out_plain)
        problem._evaluate(loop(chicane_run, straight_run), out_chicane)

        for out in (out_plain, out_chicane):
            assert np.all(np.asarray(out["G"][:3]) <= 0), "loop must close exactly"
        assert out_plain["F"][0] == pytest.approx(out_chicane["F"][0]), (
            "equal piece count must give equal utilization"
        )
        assert out_chicane["F"][1] > out_plain["F"][1] + 1e-3, (
            f"chicanes must cost time at equal utilization: "
            f"plain={out_plain['F'][1]}, chicane={out_chicane['F'][1]}"
        )


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


class TestG3BoundaryTolerance:
    """G[3] = (violation - boundary_tolerance) / diagonal: for overshoots up to
    the tolerance, the tolerance term (not the raw violation) decides the sign."""

    def _g3(self, catalog, default_config, tolerance: float) -> float:
        config = default_config.model_copy(deep=True)
        # 70x70 box; a 16-R40 circle spans ~78 studs, so it overshoots by a few
        # studs no matter how the decoder centers it.
        config.boundary = BoundaryConfig(min_x=-35.0, max_x=35.0, min_y=-35.0, max_y=35.0)
        config.boundary_tolerance = tolerance
        problem = TrackOptimizationProblem(catalog, config)
        x = create_chromosome_from_pieces(problem.dims, [PieceIndex.R40_CURVE] * 16)
        out = {}
        problem._evaluate(x, out)
        return float(out["G"][3])

    def test_overshoot_beyond_tolerance_violates(self, catalog, default_config):
        assert self._g3(catalog, default_config, 0.0) > 0.0

    def test_overshoot_within_tolerance_is_feasible(self, catalog, default_config):
        assert self._g3(catalog, default_config, 20.0) <= 0.0


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


class TestInventoryNamesMustBeInTheCatalog:
    """A misspelled piece id cannot be built, but would still enlarge the kit, so
    every utilization figure for the whole run would be understated."""

    def _with_extra_piece(self, default_config, piece_id, count=20):
        from src.config import OptimizationConfig
        inventory = {**default_config.inventory, piece_id: count}
        config = OptimizationConfig.model_validate(
            {**default_config.model_dump(exclude={"inventory"}), "inventory": inventory}
        )
        config._base_dir = default_config._base_dir
        return config

    def test_unknown_piece_id_is_rejected(self, catalog, default_config):
        config = self._with_extra_piece(default_config, "STRAIGHT_25")
        with pytest.raises(ValueError, match="STRAIGHT_25"):
            TrackOptimizationProblem(catalog, config)

    def test_message_lists_the_known_pieces(self, catalog, default_config):
        config = self._with_extra_piece(default_config, "STRAIGHT_25")
        with pytest.raises(ValueError, match="R40_SWITCH_LEFT"):
            TrackOptimizationProblem(catalog, config)

    def test_kit_size_counts_only_placeable_pieces(self, catalog, default_config):
        problem = TrackOptimizationProblem(catalog, default_config)
        assert problem.total_inventory == sum(problem.inventory_by_index.values())
