"""Test script for unified track optimization approach."""

import numpy as np
from pathlib import Path

from src.data import TrackCatalog
from src.unified_problem import (
    UnifiedTrackProblem,
    UnifiedSampling,
    UnifiedConfig,
    decode_unified,
    U_N_VAR,
)
from src.intersection import find_self_intersections, analyze_switch_connections


def test_basic_decode():
    """Test basic chromosome decoding."""
    print("=" * 60)
    print("TEST: Basic Decode")
    print("=" * 60)

    # Load catalog
    catalog = TrackCatalog.load("data/track_pieces.yaml")
    print(f"Loaded catalog with {catalog.n_pieces} pieces")

    # Create sampling helper
    inventory = {
        "STRAIGHT_16": 20,
        "R40_LEFT": 16,
        "R40_RIGHT": 16,
    }
    sampler = UnifiedSampling(catalog, inventory)

    # Test circle
    print("\n--- Circle (16 left curves) ---")
    x_circle = sampler.sample_circle()
    layout = decode_unified(x_circle, catalog)
    print(f"Pieces: {layout.n_pieces}")
    print(f"Closure error: {layout.closure_error:.3f} studs")
    print(f"Angle error: {layout.angle_error:.3f} degrees")
    print(f"Is closed: {layout.is_closed}")

    # Test oval
    print("\n--- Oval (8+8 curves + straights) ---")
    x_oval = sampler.sample_oval(n_straights=4)
    layout = decode_unified(x_oval, catalog)
    print(f"Pieces: {layout.n_pieces}")
    print(f"Closure error: {layout.closure_error:.3f} studs")
    print(f"Angle error: {layout.angle_error:.3f} degrees")
    print(f"Is closed: {layout.is_closed}")

    # Test figure-8 base
    print("\n--- Figure-8 base (16L + 16R curves) ---")
    x_fig8 = sampler.sample_figure_8_base()
    layout = decode_unified(x_fig8, catalog)
    print(f"Pieces: {layout.n_pieces}")
    print(f"Closure error: {layout.closure_error:.3f} studs")
    print(f"Angle error: {layout.angle_error:.3f} degrees")
    print(f"Is closed: {layout.is_closed}")


def test_self_intersection_detection():
    """Test self-intersection detection on an oval."""
    print("\n" + "=" * 60)
    print("TEST: Self-Intersection Detection")
    print("=" * 60)

    catalog = TrackCatalog.load("data/track_pieces.yaml")

    # Create a large oval with many straights
    inventory = {"STRAIGHT_16": 20, "R40_LEFT": 16, "R40_RIGHT": 16}
    sampler = UnifiedSampling(catalog, inventory)

    x_oval = sampler.sample_oval_with_siding_opportunity(n_straights=8)
    layout = decode_unified(x_oval, catalog)

    print(f"Layout has {layout.n_pieces} pieces")
    print(f"Closure error: {layout.closure_error:.3f}")

    # Find self-intersections
    opportunities = find_self_intersections(
        layout.states,
        layout.indices,
        position_tolerance=20.0,  # Wider search
        angle_tolerance=30.0,
    )

    print(f"\nFound {len(opportunities)} potential switch opportunities:")
    for i, opp in enumerate(opportunities[:5]):  # Show top 5
        print(f"  {i+1}. pos {opp.in_position} -> {opp.out_position}: "
              f"{opp.handedness}, dist={opp.position_error:.1f}, "
              f"angle_err={opp.heading_error:.1f}")


def test_switch_connections():
    """Test switch connection analysis with actual switches."""
    print("\n" + "=" * 60)
    print("TEST: Switch Connection Analysis")
    print("=" * 60)

    catalog = TrackCatalog.load("data/track_pieces.yaml")

    from src.encoding import (
        R40_LEFT, R40_RIGHT, STRAIGHT_16,
        SWITCH_LEFT_IN, SWITCH_LEFT_OUT,
    )

    # Manually create a closed oval with switches
    # Key insight: switches have same straight-through FK as STRAIGHT_16
    # So we can replace 2 straights with IN+OUT switches
    # Pattern: 8 curves + [SW_IN, 4 straights, SW_OUT] + 8 curves + 6 straights
    # Total straights equivalent: 1(IN) + 4 + 1(OUT) + 6 = 12 on both sides? No...
    #
    # Actually for closure: first side has IN(16) + 4*STRAIGHT(64) + OUT(16) = 96 studs
    # Second side needs 6*STRAIGHT = 96 studs. That's 6 straights.

    pieces = []
    # First half circle
    for _ in range(8):
        pieces.append(R40_LEFT)
    # Straight section with switches (total length = 6 straights = 96 studs)
    pieces.append(SWITCH_LEFT_IN)   # 16 studs (same as straight)
    for _ in range(4):
        pieces.append(STRAIGHT_16)  # 4 * 16 = 64 studs
    pieces.append(SWITCH_LEFT_OUT)  # 16 studs
    # Second half circle
    for _ in range(8):
        pieces.append(R40_LEFT)
    # Return straights - must equal total length of switch section = 6 straights
    for _ in range(6):
        pieces.append(STRAIGHT_16)

    # Create chromosome
    x = np.full(U_N_VAR, -1.0)
    x[:len(pieces)] = pieces

    layout = decode_unified(x, catalog)

    print(f"Layout: {layout.n_pieces} pieces, {layout.n_switches} switches")
    print(f"Closure error: {layout.closure_error:.3f}")
    print(f"Angle error: {layout.angle_error:.3f}")
    print(f"Connected pairs: {layout.n_connected_pairs}")
    print(f"Loose ports: {layout.n_loose_ports}")
    print(f"Has valid switches: {layout.has_valid_switches}")

    if layout.intersection_result.connected_switches:
        print("\nConnected switch pairs:")
        for in_pos, out_pos in layout.intersection_result.connected_switches:
            print(f"  IN@{in_pos} <-> OUT@{out_pos}")
    else:
        print("\nNo connected switches found.")
        print("Note: Switches on same straight section won't connect via self-intersection")
        print("      (they need a parallel branch, not same-track placement)")


def test_problem_evaluation():
    """Test problem evaluation."""
    print("\n" + "=" * 60)
    print("TEST: Problem Evaluation")
    print("=" * 60)

    catalog = TrackCatalog.load("data/track_pieces.yaml")

    inventory = {
        "STRAIGHT_16": 20,
        "R40_LEFT": 16,
        "R40_RIGHT": 16,
        "R40_SWITCH_LEFT_IN": 2,
        "R40_SWITCH_LEFT_OUT": 2,
        "R40_SWITCH_RIGHT_IN": 2,
        "R40_SWITCH_RIGHT_OUT": 2,
    }

    config = UnifiedConfig(
        position_tolerance=2.0,
        angle_tolerance=5.0,
    )

    problem = UnifiedTrackProblem(catalog, inventory, config)
    sampler = UnifiedSampling(catalog, inventory)

    print(f"Problem: {problem.n_var} variables, {problem.n_obj} objectives, "
          f"{problem.n_ieq_constr} constraints")

    # Evaluate circle
    print("\n--- Evaluating circle ---")
    x_circle = sampler.sample_circle()
    out = {}
    problem._evaluate(x_circle, out)
    print(f"Objectives: {out['F']}")
    print(f"Constraints: {out['G']}")
    print(f"Feasible: {all(g <= 0 for g in out['G'])}")

    # Evaluate oval
    print("\n--- Evaluating oval ---")
    x_oval = sampler.sample_oval(n_straights=4)
    out = {}
    problem._evaluate(x_oval, out)
    print(f"Objectives: {out['F']}")
    print(f"Constraints: {out['G']}")
    print(f"Feasible: {all(g <= 0 for g in out['G'])}")


def test_pymoo_integration():
    """Test pymoo optimization (short run)."""
    print("\n" + "=" * 60)
    print("TEST: pymoo Integration (10 generations)")
    print("=" * 60)

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.core.sampling import Sampling

    catalog = TrackCatalog.load("data/track_pieces.yaml")

    inventory = {
        "STRAIGHT_16": 20,
        "R40_LEFT": 16,
        "R40_RIGHT": 16,
    }

    problem = UnifiedTrackProblem(catalog, inventory)

    # Custom sampling that uses our heuristics
    class HeuristicSampling(Sampling):
        def __init__(self, catalog, inventory):
            super().__init__()
            self.sampler = UnifiedSampling(catalog, inventory)

        def _do(self, problem, n_samples, **kwargs):
            X = np.zeros((n_samples, problem.n_var))
            rng = np.random.default_rng()

            for i in range(n_samples):
                if i % 4 == 0:
                    X[i] = self.sampler.sample_circle()
                elif i % 4 == 1:
                    X[i] = self.sampler.sample_oval(n_straights=4)
                elif i % 4 == 2:
                    X[i] = self.sampler.sample_figure_8_base()
                else:
                    X[i] = self.sampler.random_sample(rng)

            return X

    algorithm = NSGA2(
        pop_size=20,
        sampling=HeuristicSampling(catalog, inventory),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 10),
        seed=42,
        verbose=True,
    )

    print(f"\nOptimization complete!")
    print(f"Best objectives: {result.F}")
    print(f"Feasible solutions: {sum(1 for cv in result.CV if cv <= 0) if result.CV is not None else 'N/A'}")

    # Decode best solution
    if result.X is not None:
        best_x = result.X[0] if result.X.ndim > 1 else result.X
        layout = decode_unified(best_x, catalog)
        print(f"\nBest layout:")
        print(f"  Pieces: {layout.n_pieces}")
        print(f"  Closure: {layout.closure_error:.3f} studs")
        print(f"  Angle: {layout.angle_error:.3f} degrees")


if __name__ == "__main__":
    test_basic_decode()
    test_self_intersection_detection()
    test_switch_connections()
    test_problem_evaluation()
    test_pymoo_integration()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
