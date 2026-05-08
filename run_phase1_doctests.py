"""Phase 1 DoD: tests 1.8 (mini-opt before/after) and 1.10 (full default 3-seed)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.repair import CycleClosureRepair
from src_v2.runner import run_optimization
from tests.fixtures.mini_problem import DEFAULT_MINI_INVENTORY
from tests.fixtures.regression_baselines import extract_metrics

ROOT = Path(__file__).resolve().parent
RUNTIME_CX = 0.0
RUNTIME_MUT = 1.0


def _runtime_knobs(config: OptimizationConfig) -> None:
    """Apply §9.7 runtime knobs (Rule 25 revised): cx=0.0, mutation=1.0."""
    config.algorithm.crossover_prob = RUNTIME_CX
    config.algorithm.mutation_prob = RUNTIME_MUT


def _mini_opt_with_runtime_knobs(
    out: Path, seed: int = 42, heuristic_ratio: float = 0.0,
) -> object:
    """Mini optimization on configs/default.yaml 64-piece inventory with
    §9.7 runtime knobs applied (cx=0.0, mutation=1.0). Standard T3 mini-opt
    sizing (n_gen=50, pop=50). ``heuristic_ratio=0.0`` forces random init so
    Phase 1's closure repair has work to do (with heuristic seeds the
    initial population is already closed and Phase 1's effect is masked)."""
    config = OptimizationConfig.load(ROOT / "configs" / "default.yaml")
    config.algorithm.seed = seed
    config.algorithm.n_gen = 50
    config.algorithm.pop_size = 50
    config.algorithm.heuristic_ratio = heuristic_ratio
    config.n_workers = 1
    _runtime_knobs(config)
    catalog = TrackCatalog.load(ROOT / "data" / "track_pieces_v2.yaml")
    out.mkdir(parents=True, exist_ok=True)
    return run_optimization(config, catalog, output_dir=out, seed=seed)


# =============================================================================
# Test 1.8: Mini-opt before/after Phase 1
# =============================================================================

def test_1_8() -> dict:
    print("=" * 72)
    print("Test 1.8: Mini-opt before/after Phase 1 (cx=0.0, mutation=1.0)")
    print("=" * 72)

    original_repair = CycleClosureRepair.repair_one

    # ----- BEFORE: monkeypatch CycleClosureRepair.repair_one to no-op -----
    CycleClosureRepair.repair_one = lambda self, x, skip_anchor_slots=None: x
    out_before = ROOT / "outputs_v2" / "_phase1_doctest_before"
    t0 = time.time()
    res_before = _mini_opt_with_runtime_knobs(out_before, seed=42)
    t_before = time.time() - t0
    m_before = extract_metrics(res_before)

    # ----- AFTER: restore Phase 1 repair -----
    CycleClosureRepair.repair_one = original_repair
    out_after = ROOT / "outputs_v2" / "_phase1_doctest_after"
    t0 = time.time()
    res_after = _mini_opt_with_runtime_knobs(out_after, seed=42)
    t_after = time.time() - t0
    m_after = extract_metrics(res_after)

    print(f"BEFORE Phase 1: {m_before} | runtime={t_before:.1f}s")
    print(f"AFTER  Phase 1: {m_after} | runtime={t_after:.1f}s")

    feas_delta = m_after["feasibility_rate"] - m_before["feasibility_rate"]
    if m_before["mean_util_feasible"] is None or m_after["mean_util_feasible"] is None:
        util_delta = None
    else:
        util_delta = m_after["mean_util_feasible"] - m_before["mean_util_feasible"]
    runtime_ratio = t_after / t_before if t_before > 0 else float("inf")

    print(f"delta_feasibility = {feas_delta:+.4f}  (need >= +0.05)")
    print(f"delta_mean_util   = {util_delta if util_delta is None else f'{util_delta:+.4f}'}  (need >= +0.05)")
    print(f"runtime ratio = {runtime_ratio:.2f}x  (need < 1.5x)")

    pass_feas = feas_delta >= 0.05
    pass_util = util_delta is not None and util_delta >= 0.05
    pass_rt = runtime_ratio < 1.5
    pass_all = pass_feas and pass_util and pass_rt

    print(f"\n1.8 verdict: feas={pass_feas} util={pass_util} runtime={pass_rt} -> {'PASS' if pass_all else 'FAIL'}")
    return {
        "before": m_before,
        "after": m_after,
        "feas_delta": feas_delta,
        "util_delta": util_delta,
        "runtime_ratio": runtime_ratio,
        "runtime_before_s": t_before,
        "runtime_after_s": t_after,
        "pass": pass_all,
    }


# =============================================================================
# Test 1.10: Full `default` 200-gen x 3-seed
# =============================================================================

def test_1_10() -> dict:
    print("\n" + "=" * 72)
    print("Test 1.10: Full default 200-gen x 3-seed (cx=0.0, mutation=1.0)")
    print("=" * 72)

    catalog = TrackCatalog.load(ROOT / "data" / "track_pieces_v2.yaml")
    seeds = [42, 43, 44]
    per_seed: list[dict] = []
    best_piece_counts: list[int] = []

    for seed in seeds:
        config = OptimizationConfig.load(ROOT / "configs" / "default.yaml")
        config.algorithm.n_gen = 200
        config.algorithm.pop_size = 1000
        config.algorithm.heuristic_ratio = 0.30
        config.algorithm.seed = seed
        _runtime_knobs(config)
        config.n_workers = 8

        out = ROOT / "outputs_v2" / f"_phase1_doctest_full_seed{seed}"
        out.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        res = run_optimization(config, catalog, output_dir=out, seed=seed)
        elapsed = time.time() - t0

        F = res.pop.get("F")
        G = res.pop.get("G")
        feas_mask = np.all(G <= 0, axis=1)
        n_feas = int(feas_mask.sum())
        total_inv = sum(config.inventory.values())

        if n_feas > 0:
            utils = -F[feas_mask, 0]
            best_pieces = int(round(float(utils.max()) * total_inv))
        else:
            best_pieces = 0

        best_piece_counts.append(best_pieces)
        per_seed.append({
            "seed": seed,
            "n_feasible": n_feas,
            "n_pop": int(F.shape[0]),
            "best_pieces": best_pieces,
            "total_inventory": total_inv,
            "runtime_s": elapsed,
        })
        print(f"  seed={seed}: feas={n_feas}/{F.shape[0]}, best_pieces={best_pieces}/{total_inv}, time={elapsed:.1f}s")

    best_overall = max(best_piece_counts)
    pass_all = best_overall >= 58
    print(f"\nbest_piece_counts per seed: {best_piece_counts}")
    print(f"max best_pieces across seeds = {best_overall}  (need >= 58)")
    print(f"\n1.10 verdict: {'PASS' if pass_all else 'FAIL'}")

    return {
        "per_seed": per_seed,
        "best_pieces_per_seed": best_piece_counts,
        "best_overall": best_overall,
        "pass": pass_all,
    }


def main() -> None:
    results = {
        "1.8": test_1_8(),
        "1.10": test_1_10(),
    }
    print("\n" + "=" * 72)
    print("PHASE 1 DOD SUMMARY")
    print("=" * 72)
    print(json.dumps(results, indent=2, default=str))
    out_path = ROOT / "outputs_v2" / "phase1_doctest_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
