"""Run a single seed of test 1.10 (full default 200-gen) with early
termination disabled. Writes per-seed JSON. Invoked from the orchestrator
in a fresh subprocess to dodge the Windows multiprocessing.Pool reuse
BrokenPipeError that hits when several pool-bearing runs share one process."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import src_v2.runner as runner_mod
from pymoo.core.termination import Termination
from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.runner import run_optimization

ROOT = Path(__file__).resolve().parent
import os as _os
_STAGNATION_PATIENCE = int(_os.environ.get("PHASE1_PATIENCE", "30"))


class _StagnationOrMaxGenTermination(Termination):
    """Stop when either (a) ``n_max_gen`` reached or (b) no improvement in
    the population's best feasible utilization for ``patience`` consecutive
    generations -- whichever fires first."""

    def __init__(self, n_max_gen: int = 200, patience: int = _STAGNATION_PATIENCE):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.patience = patience
        self._best = -float("inf")
        self._last_improvement_gen = 0

    @staticmethod
    def _best_util(algorithm) -> float:
        F = algorithm.pop.get("F")
        G = algorithm.pop.get("G")
        if F is None or G is None:
            return -float("inf")
        feas = np.all(G <= 0, axis=1)
        if not feas.any():
            return -float("inf")
        return float((-F[feas, 0]).max())

    def _update(self, algorithm) -> float:
        gen = int(algorithm.n_gen or 0)
        cur = self._best_util(algorithm)
        if cur > self._best + 1e-9:
            self._best = cur
            self._last_improvement_gen = gen
        stale = gen - self._last_improvement_gen
        if stale >= self.patience or gen >= self.n_max_gen:
            return 1.0
        return max(gen / self.n_max_gen, stale / self.patience)


class _PatchedTerminationFactory:
    """Drop-in replacement for ``DefaultMultiObjectiveTermination`` that
    swaps in stagnation-or-max-gen termination. Mirrors the kwargs surface
    ``runner.run_optimization`` calls with."""

    def __new__(cls, *, xtol=None, cvtol=None, ftol=None, period=None,
                n_skip=None, n_max_gen=200, **_kwargs):
        return _StagnationOrMaxGenTermination(
            n_max_gen=n_max_gen, patience=_STAGNATION_PATIENCE,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-gen", type=int, default=200)
    parser.add_argument("--pop", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", default="default", help="Config name (no .yaml)")
    parser.add_argument("--out", required=True, help="Result JSON path")
    parser.add_argument("--run-dir", required=True, help="Optimizer output dir")
    args = parser.parse_args()

    runner_mod.DefaultMultiObjectiveTermination = _PatchedTerminationFactory

    config = OptimizationConfig.load(ROOT / "configs" / f"{args.config}.yaml")
    config.algorithm.n_gen = args.n_gen
    config.algorithm.pop_size = args.pop
    config.algorithm.heuristic_ratio = 0.30
    config.algorithm.crossover_prob = 0.0
    config.algorithm.mutation_prob = 1.0
    config.algorithm.seed = args.seed
    config.n_workers = args.workers

    catalog = TrackCatalog.load(ROOT / "data" / "track_pieces_v2.yaml")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res = run_optimization(config, catalog, output_dir=run_dir, seed=args.seed)
    elapsed = time.time() - t0

    F = res.pop.get("F")
    G = res.pop.get("G")
    feas_mask = np.all(G <= 0, axis=1)
    n_feas = int(feas_mask.sum())
    total_inv = sum(config.inventory.values())
    if n_feas > 0:
        utils = -F[feas_mask, 0]
        best_pieces = int(round(float(utils.max()) * total_inv))
        best_util = float(utils.max())
    else:
        best_pieces = 0
        best_util = 0.0

    payload = {
        "config": args.config,
        "seed": args.seed,
        "n_gen_requested": args.n_gen,
        "pop_size": args.pop,
        "n_workers": args.workers,
        "n_feasible": n_feas,
        "n_pop": int(F.shape[0]),
        "best_util": best_util,
        "best_pieces": best_pieces,
        "total_inventory": total_inv,
        "runtime_s": elapsed,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
