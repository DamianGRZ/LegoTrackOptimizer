"""Phase 1 DoD fast verification: 1.8 + single-seed 1.10. Targets <10 min wall."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.repair import CycleClosureRepair
from src_v2.runner import run_optimization
from tests.fixtures.regression_baselines import extract_metrics
import src_v2.runner as runner_mod
from pymoo.core.termination import Termination

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


# ----- Stagnation termination (same as run_phase1_seed.py) -----------------

class _Stagnation(Termination):
    def __init__(self, n_max_gen: int, patience: int) -> None:
        super().__init__()
        self.n_max_gen = n_max_gen
        self.patience = patience
        self._best = -float("inf")
        self._last_imp = 0

    def _update(self, algorithm) -> float:
        gen = int(algorithm.n_gen or 0)
        F = algorithm.pop.get("F")
        G = algorithm.pop.get("G")
        if F is not None and G is not None:
            feas = np.all(G <= 0, axis=1)
            cur = float((-F[feas, 0]).max()) if feas.any() else -float("inf")
            if cur > self._best + 1e-9:
                self._best = cur
                self._last_imp = gen
        stale = gen - self._last_imp
        if stale >= self.patience or gen >= self.n_max_gen:
            return 1.0
        return max(gen / self.n_max_gen, stale / self.patience)


class _StagnationFactory:
    """Drop-in for ``DefaultMultiObjectiveTermination`` mirroring its kwargs."""
    _patience = 30
    _n_max_gen_default = 200

    def __new__(cls, *, xtol=None, cvtol=None, ftol=None, period=None,
                n_skip=None, n_max_gen=200, **_kwargs):
        return _Stagnation(n_max_gen=n_max_gen, patience=cls._patience)


def _patch_termination(patience: int) -> None:
    _StagnationFactory._patience = patience
    runner_mod.DefaultMultiObjectiveTermination = _StagnationFactory


# ----- Test 1.8 ---------------------------------------------------------------

def _mini_run(seed: int, *, with_phase1: bool, pop: int, n_gen: int,
              heuristic_ratio: float, out: Path) -> tuple[dict, float]:
    config = OptimizationConfig.load(ROOT / "configs" / "default.yaml")
    config.algorithm.seed = seed
    config.algorithm.n_gen = n_gen
    config.algorithm.pop_size = pop
    config.algorithm.heuristic_ratio = heuristic_ratio
    config.algorithm.crossover_prob = 0.0
    config.algorithm.mutation_prob = 1.0
    config.n_workers = 1
    catalog = TrackCatalog.load(ROOT / "data" / "track_pieces_v2.yaml")
    out.mkdir(parents=True, exist_ok=True)

    original = CycleClosureRepair.repair_one
    if not with_phase1:
        CycleClosureRepair.repair_one = (
            lambda self, x, skip_anchor_slots=None: x
        )
    try:
        t0 = time.time()
        res = run_optimization(config, catalog, output_dir=out, seed=seed)
        elapsed = time.time() - t0
    finally:
        CycleClosureRepair.repair_one = original
    return extract_metrics(res), elapsed


def test_1_8(pop: int = 100, n_gen: int = 80) -> dict:
    print(f"\n[1.8] pop={pop} n_gen={n_gen} heuristic=0.0 inv=64-piece default")
    _patch_termination(patience=n_gen)  # don't early-terminate; full budget
    out_b = ROOT / "outputs_v2" / "_p1dod_1_8_before"
    out_a = ROOT / "outputs_v2" / "_p1dod_1_8_after"
    m_b, t_b = _mini_run(42, with_phase1=False, pop=pop, n_gen=n_gen,
                         heuristic_ratio=0.0, out=out_b)
    m_a, t_a = _mini_run(42, with_phase1=True, pop=pop, n_gen=n_gen,
                         heuristic_ratio=0.0, out=out_a)
    feas_d = m_a["feasibility_rate"] - m_b["feasibility_rate"]
    if m_b["mean_util_feasible"] is None or m_a["mean_util_feasible"] is None:
        util_d = None
    else:
        util_d = m_a["mean_util_feasible"] - m_b["mean_util_feasible"]
    rt = t_a / t_b if t_b > 0 else float("inf")
    print(f"  BEFORE: feas={m_b['feasibility_rate']:.3f} util={m_b['mean_util_feasible']} t={t_b:.1f}s")
    print(f"  AFTER : feas={m_a['feasibility_rate']:.3f} util={m_a['mean_util_feasible']} t={t_a:.1f}s")
    print(f"  delta_feas={feas_d:+.4f} (>=+0.05) | delta_util={util_d} (>=+0.05) | runtime={rt:.2f}x (<1.5x)")
    pass_feas = feas_d >= 0.05
    pass_util = util_d is not None and util_d >= 0.05
    pass_rt = rt < 1.5
    verdict = "PASS" if (pass_feas and pass_util and pass_rt) else "FAIL"
    print(f"  1.8 verdict: feas={pass_feas} util={pass_util} runtime={pass_rt} -> {verdict}")
    return {
        "before": m_b, "after": m_a, "delta_feas": feas_d, "delta_util": util_d,
        "runtime_ratio": rt, "verdict": verdict,
    }


# ----- Test 1.10 (single seed via subprocess for clean process state) --------

def test_1_10_one_seed(seed: int, *, pop: int, patience: int) -> dict:
    print(f"\n[1.10] with_switches seed={seed} pop={pop} patience={patience}")
    run_dir = ROOT / "outputs_v2" / f"_p1dod_1_10_seed{seed}"
    out = run_dir / "result.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PHASE1_PATIENCE=str(patience))
    cmd = [
        str(PYTHON), "run_phase1_seed.py",
        "--config", "with_switches", "--seed", str(seed),
        "--n-gen", "200", "--pop", str(pop), "--workers", "1",
        "--out", str(out), "--run-dir", str(run_dir),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0 or not out.exists():
        print(f"  ERROR rc={proc.returncode} elapsed={elapsed:.1f}s")
        print(proc.stderr[-1000:])
        return {"seed": seed, "error": True, "runtime_s": elapsed}
    payload = json.loads(out.read_text(encoding="utf-8"))
    print(f"  feas={payload['n_feasible']}/{payload['n_pop']} "
          f"best_pieces={payload['best_pieces']}/{payload['total_inventory']} "
          f"runtime={elapsed:.1f}s")
    print(f"  1.10 verdict: {'PASS' if payload['best_pieces'] >= 58 else 'FAIL'} (>=58)")
    return payload


def main() -> None:
    t0 = time.time()
    r18 = test_1_8(pop=100, n_gen=80)
    r110 = test_1_10_one_seed(42, pop=300, patience=15)
    total = time.time() - t0
    print(f"\n{'=' * 60}\nTotal wall-clock: {total:.1f}s")
    summary = {"1.8": r18, "1.10": r110, "wall_s": total}
    (ROOT / "outputs_v2" / "phase1_dod_fast_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
