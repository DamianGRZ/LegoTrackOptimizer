"""Replication harness: one config x many seeds, one subprocess per run.

Decision-grade comparisons need seeded replications summarized as
median +/- IQR (thesis protocol, chapter 4.5). Each run goes through
``main.py --seed`` in a fresh process, so artifacts and crash handling are
identical to single-run invocations. Output: ``outputs/<config>_s<seed>/``.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def resolve_config(value: str) -> Path:
    if "/" in value or value.endswith(".yaml"):
        return Path(value)
    return Path("configs") / f"{value}.yaml"


def parse_seeds(spec: str) -> list[int]:
    """``"1..10"`` -> [1, ..., 10]; ``"1,2,5"`` -> [1, 2, 5]."""
    if ".." in spec:
        lo, hi = spec.split("..", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def has_results(output_dir: Path) -> bool:
    return (output_dir / "fitness.csv").exists()


def run_seed(config_path: Path, seed: int, output_dir: Path) -> dict:
    cmd = [sys.executable, "-u", "main.py",
           "--config", str(config_path),
           "--seed", str(seed),
           "--output", str(output_dir),
           "--verbose"]
    log_path = output_dir.parent / f"{output_dir.name}_console.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        returncode = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return {"seed": seed, "returncode": returncode, "elapsed_sec": time.time() - t0}


def best_feasible_util(output_dir: Path) -> float:
    fitness = output_dir / "fitness.csv"
    constraints = output_dir / "constraints.csv"
    if not fitness.exists() or not constraints.exists():
        return float("nan")
    F = np.loadtxt(fitness, delimiter=",", skiprows=1, ndmin=2)
    G = np.loadtxt(constraints, delimiter=",", skiprows=1, ndmin=2)
    feasible = np.all(G <= 0, axis=1)
    if not feasible.any():
        return float("nan")
    return float(-F[feasible, 0].min())


def summarize(utils: list[float]) -> str:
    values = np.array([u for u in utils if not np.isnan(u)])
    if len(values) == 0:
        return "no successful runs"
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    line = f"median {med:.1%} | IQR [{q1:.1%}, {q3:.1%}] | n={len(values)}"
    if len(values) > 1 and np.ptp(values) < 1e-12:
        line += " | WARNING: zero inter-seed variance — investigate before using statistically"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one config across multiple seeds (median±IQR summary)"
    )
    parser.add_argument("--config", required=True,
                        help="Config name (configs/<name>.yaml) or path")
    parser.add_argument("--seeds", default="1..10",
                        help='Seed spec: "1..10" or "1,2,5" (default 1..10)')
    parser.add_argument("--force", action="store_true",
                        help="Re-run seeds whose output dir already has results")
    args = parser.parse_args()

    config_path = resolve_config(args.config)
    stem = config_path.stem
    seeds = parse_seeds(args.seeds)

    results = []
    for seed in seeds:
        output_dir = Path("outputs") / f"{stem}_s{seed}"
        if has_results(output_dir) and not args.force:
            print(f"=== {stem} seed={seed}: existing results, skipping "
                  f"(--force to re-run)", flush=True)
            results.append({"seed": seed, "returncode": 0, "elapsed_sec": 0.0})
            continue
        print(f"=== {stem} seed={seed} -> {output_dir}", flush=True)
        results.append(run_seed(config_path, seed, output_dir))

    print(f"\n{stem}: best feasible utilization per seed")
    utils = []
    for r in results:
        util = best_feasible_util(Path("outputs") / f"{stem}_s{r['seed']}")
        utils.append(util)
        shown = "n/a" if np.isnan(util) else f"{util:.1%}"
        status = "ok" if r["returncode"] == 0 else f"FAILED rc={r['returncode']}"
        print(f"  seed {r['seed']:>3}: {shown} ({r['elapsed_sec']:.0f}s, {status})")
    print(f"  {summarize(utils)}")


if __name__ == "__main__":
    main()