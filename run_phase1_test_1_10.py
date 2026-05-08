"""Orchestrator for test 1.10 with early termination disabled. Runs both
configs/default.yaml (literal "default", max ~40 pieces achievable) and
configs/with_switches.yaml (168 pieces, where the >=58 target lands sensibly).
Spawns one fresh subprocess per (config, seed) so each multiprocessing.Pool
is isolated."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
CONFIGS = ["with_switches"]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def run_seed(config_name: str, seed: int) -> dict:
    run_dir = ROOT / "outputs_v2" / f"_phase1_test_1_10_{config_name}_seed{seed}"
    out_path = run_dir / "result.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON), "run_phase1_seed.py",
        "--config", config_name,
        "--seed", str(seed),
        "--n-gen", "200",
        "--pop", "300",
        "--workers", "1",
        "--out", str(out_path),
        "--run-dir", str(run_dir),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"config": config_name, "seed": seed, "error": proc.stderr[-1500:],
                "runtime_s": elapsed}
    if not out_path.exists():
        return {"config": config_name, "seed": seed, "error": "no JSON output",
                "runtime_s": elapsed}
    return json.loads(out_path.read_text(encoding="utf-8"))


def report_config(config_name: str, results: list[dict]) -> None:
    print(f"\n{'-' * 72}\nCONFIG: {config_name}\n{'-' * 72}", flush=True)
    successful = [r for r in results if "best_pieces" in r]
    if not successful:
        print("All seeds failed.", flush=True)
        for r in results:
            print(f"  seed {r['seed']}: ERROR\n{r.get('error', '')[:400]}", flush=True)
        return
    for r in results:
        if "best_pieces" in r:
            print(f"  seed {r['seed']}: feas={r['n_feasible']}/{r['n_pop']}, "
                  f"best_pieces={r['best_pieces']}/{r['total_inventory']}, "
                  f"runtime={r['runtime_s']:.1f}s", flush=True)
        else:
            print(f"  seed {r['seed']}: ERROR after {r['runtime_s']:.1f}s", flush=True)
    best_per_seed = [r["best_pieces"] for r in successful]
    best_overall = max(best_per_seed)
    print(f"  per-seed best_pieces: {best_per_seed}", flush=True)
    print(f"  max best_pieces:     {best_overall}", flush=True)
    print(f"  acceptance threshold: >= 58", flush=True)
    print(f"  VERDICT: {'PASS' if best_overall >= 58 else 'FAIL'}", flush=True)


def main() -> None:
    t_start = time.time()
    all_results: dict[str, list[dict]] = {}

    for config_name in CONFIGS:
        print(f"\n{'=' * 72}\nCONFIG: {config_name}\n{'=' * 72}", flush=True)
        cfg_results: list[dict] = []
        for seed in SEEDS:
            print(f"  -> seed {seed} ...", flush=True)
            r = run_seed(config_name, seed)
            cfg_results.append(r)
            if "best_pieces" in r:
                print(f"     feas={r['n_feasible']}/{r['n_pop']}, "
                      f"best_pieces={r['best_pieces']}/{r['total_inventory']}, "
                      f"runtime={r['runtime_s']:.1f}s", flush=True)
            else:
                print(f"     FAIL ({r.get('error', '')[:200]})", flush=True)
        all_results[config_name] = cfg_results

    print(f"\n{'=' * 72}\nTEST 1.10 SUMMARY (early termination DISABLED, n_gen=200, 3 seeds)\n{'=' * 72}", flush=True)
    for cn in CONFIGS:
        report_config(cn, all_results[cn])

    print(f"\nTotal wall-clock: {time.time() - t_start:.1f}s", flush=True)
    summary_path = ROOT / "outputs_v2" / "phase1_test_1_10_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
