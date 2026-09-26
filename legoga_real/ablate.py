"""Ablacja ustawien: python -m legoga_real.ablate --stage 1 [--seeds 5] [--workers 10]

Etapy 0, 1, 2, 2_lite, 3a, 3a_lite maja warianty w kodzie; 3b, 4 i 5 zaleza od wynikow, wiec biora
je z pliku JSON (--variants): lista {"name", "settings", "pop", "gens"}.
Kazdy bieg laduje w outputs/legoga_real_ablation/<etap>/<wariant>/<zestaw>_<stol>/seed_<n>/
jako run.json, front.csv, layouts.json, log.txt i to, co zapisuje run(). Bieg z gotowym
run.json jest pomijany, wiec przerwana kampanie wznawia ta sama komenda.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import math
import multiprocessing
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pymoo
from pymoo.indicators.hv import HV

from .ga import DEFAULT_INVENTORY, physical_counts
from .run import run
from .settings import NEUTRAL_BLOCKS, Settings
from .table import TransformTable

ROOT = Path("outputs/legoga_real_ablation")
# etapy wyboru na ziarnach od 1; potwierdzenie (etap 5) na ziarnach, ktorych wybor nie widzial
SEEDS = {"5": list(range(11, 41))}
SEEDS_DEFAULT = list(range(1, 11))
N_SEEDS = {"0": 1, "1": 5, "2": 10, "2_lite": 5, "3a": 3, "3a_lite": 3, "3b": 10, "4": 10,
           "5": 30}
POP, GENS = 60, 40
EVALS_LONG = 19200
# etap 2: rodzice x potomkowie; wersja lite to rzadsza siatka z tym samym budzetem ocen
POP_OFF = {"2": ((20, 30, 60, 100, 150), (10, 15, 30, 60, 120, 200)),
           "2_lite": ((20, 60, 150), (30, 60, 200))}
# etap 3a: glebokosc tabeli x wycinek; lite tylko tam, gdzie zamiennik moze byc dluzszy
TABLE_WIN = {"3a": (range(1, 13), range(1, 12)),
             "3a_lite": ((7, 9, 11, 12), (3, 6, 10))}

REAL_KIT = {"S16": 90, "S24": 2, "R40": 140, "WL": 7, "WR": 3, "XX": 3, "DC": 1}
KITS = {
    "real": REAL_KIT,
    "default": DEFAULT_INVENTORY,
    "plain": {"S16": 120, "S24": 8, "R40": 80, "WL": 0, "WR": 0, "XX": 0, "DC": 0},
    "onesided": {"S16": 90, "S24": 2, "R40": 140, "WL": 0, "WR": 6, "XX": 1, "DC": 0},
}
TABLES = {"250x350": (250.0, 350.0), "250x250": (250.0, 250.0),
          "400x600": (400.0, 600.0), "500x500": (500.0, 500.0)}

# pole frontu bez skalowania: punkt odniesienia to zero klockow i zero tras
HV_RAW = HV(ref_point=np.array([0.0, 0.0]))

# ile razy operator probuje: te pola skalujemy razem jako jedno ustawienie
EFFORT = ("swap_tries", "branch_attempts", "branch_seqs", "xx_attempts", "xx_limit",
          "xx_scan", "dc_attempts", "dc_limit", "dc_scan", "cx_tries")
WEIGHTS_UNIFORM = (0.125,) * 8
WEIGHTS_ADD = (0.30, 0.32, 0.38, 0.07, 0.20, 0.04, 0.20, 0.04)
BLOCKS_3 = NEUTRAL_BLOCKS[:3]
BLOCKS_12 = NEUTRAL_BLOCKS + (("S24", "S24"), ("S16", "S24"), ("CL", "CR", "S24"),
                              ("S24", "CL", "CR"), ("S16", "S16", "S16"))

# etap 1: kazde ustawienie osobno wokol bazy
LEVELS = {
    "mut_prob": [0.3, 0.5, 0.7, 1.0],
    "n_ops": [1, 4],
    "cx_prob": [0.0, 0.3, 0.6, 1.0],
    "pressure": [3, 4],
    "dedupe": [False],
    "sig_with_size": [False],
    "forget_each_gen": [False],
    "per_bucket": [20, 100, 300, 1000, None],
    "accept": ["best"],
    "mutate_draws": [3, None],
    "oval_sizes": [(8, 10), tuple(range(4, 21, 2))],
    "figure_eight": [False],
    "warmup": [(2, 10), (5, 40)],
    "antipodal_min_chain": [2, 8],
    "cx_min_chain": [2, 6],
    "dc_refill": ["s16"],
    "cx_tol": [0.15],
}


def label(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, tuple):
        return "-".join(map(str, value))
    return str(value).lower()


def variants(stage: str, variants_file: str | None) -> dict:
    """Nazwa wariantu -> (Settings, pop, gens)."""
    base = Settings()
    out = {"base": (base, POP, GENS)}
    if stage == "0":
        return out
    if stage == "1":
        for field, values in LEVELS.items():
            for v in values:
                out[f"{field}_{label(v)}"] = (dataclasses.replace(base, **{field: v}), POP, GENS)
        for scale in (0.5, 2.0):
            scaled = {f: max(1, round(getattr(base, f) * scale)) for f in EFFORT}
            out[f"effort_{scale}"] = (dataclasses.replace(base, **scaled), POP, GENS)
        out["weights_uniform"] = (dataclasses.replace(base, weights=WEIGHTS_UNIFORM), POP, GENS)
        out["weights_add"] = (dataclasses.replace(base, weights=WEIGHTS_ADD), POP, GENS)
        out["blocks_3"] = (dataclasses.replace(base, neutral_blocks=BLOCKS_3), POP, GENS)
        out["blocks_12"] = (dataclasses.replace(base, neutral_blocks=BLOCKS_12), POP, GENS)
        return out
    if stage in ("2", "2_lite"):
        pops, offs = POP_OFF[stage]
        out = {}
        for pop in pops:
            for off in offs:
                gens = math.ceil((EVALS_LONG - pop) / off) + 1   # pokolenie 1 = start
                cfg = dataclasses.replace(base, n_offsprings=off)
                out[f"pop_{pop}_off_{off}"] = (cfg, pop, gens)
        return out
    if stage in ("3a", "3a_lite"):
        depths, wins = TABLE_WIN[stage]
        out = {}
        for depth in depths:
            for win in wins:
                cfg = dataclasses.replace(base, table_len=depth, max_win=win)
                out[f"table_{depth}_win_{win}"] = (cfg, POP, GENS)
        return out
    if not variants_file:
        raise SystemExit(f"etap {stage} wymaga --variants plik.json")
    out = {}
    for v in json.loads(Path(variants_file).read_text(encoding="utf-8")):
        cfg = dataclasses.replace(base, **v.get("settings", {}))
        out[v["name"]] = (cfg, v.get("pop", POP), v.get("gens", GENS))
    return out


def conditions(stage: str) -> list[tuple[str, str]]:
    """(zestaw, stol): prawdziwy zestaw na kazdym stole; w etapach 1 i 5 takze
    pozostale zestawy na prawdziwym stole."""
    out = [("real", t) for t in TABLES]
    if stage in ("1", "5"):
        out += [(k, "250x350") for k in KITS if k != "real"]
    return out


def table_key(cfg: Settings) -> tuple:
    return cfg.table_len, cfg.per_bucket


def table_path(key: tuple) -> Path:
    depth, per_bucket = key
    return ROOT / "tables" / f"table_{depth}_{label(per_bucket)}.pkl"


def build_table(key: tuple) -> None:
    path = table_path(key)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    depth, per_bucket = key
    t0 = time.time()
    TransformTable(max_len=depth, per_bucket=per_bucket).save(path)
    path.with_suffix(".json").write_text(json.dumps({"build_s": time.time() - t0}))


def build_tables(keys) -> None:
    """Kazda potrzebna tabela raz, przed pula biegow. Kazda w swiezym procesie, bo
    proces glowny nie oddalby systemowi pamieci po zbudowanej tabeli."""
    with multiprocessing.Pool(1, maxtasksperchild=1) as pool:
        pool.map(build_table, sorted(keys, key=str))


_TABLES: dict = {}


def load_table(cfg: Settings) -> TransformTable:
    path = table_path(table_key(cfg))
    if path not in _TABLES:
        _TABLES.clear()                      # jeden proces trzyma jedna tabele
        _TABLES[path] = TransformTable.load(path)
    return _TABLES[path]


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--", "legoga_real")   # tylko kod, ktory biegnie
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty)}


def run_cell(task: dict) -> str:
    """Jeden bieg; zwraca 'ok', 'skip' albo 'fail'. Wszystko idzie do task['out']."""
    out = Path(task["out"])
    if (out / "run.json").exists():
        return "skip"
    out.mkdir(parents=True, exist_ok=True)
    cfg, kit, size = task["cfg"], KITS[task["kit"]], TABLES[task["table"]]
    try:
        with (out / "log.txt").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            res, _, times = run(inventory=kit, pop=task["pop"], gens=task["gens"],
                                seed=task["seed"], out=out, max_size=size, verbose=False,
                                cfg=cfg, table=load_table(cfg), hv=HV_RAW,
                                render=task["render"])
    except Exception as exc:                 # jeden padniety bieg nie zatrzymuje kampanii
        (out / "error.txt").write_text(repr(exc), encoding="utf-8")
        return "fail"
    build_json = table_path(table_key(cfg)).with_suffix(".json")
    times["table_build"] = json.loads(build_json.read_text())["build_s"]
    save_results(out, task, res, times)
    return "ok"


def save_results(out: Path, task: dict, res, times: dict) -> None:
    layouts = [x[0] for x in np.atleast_2d(res.X)]
    F = np.atleast_2d(res.F)
    G = np.atleast_2d(res.G)
    feas = (G <= 0).all(axis=1)
    progress = res.algorithm.callback.data
    with (out / "front.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pieces", "routes", "overlap", "inventory_excess", "over_size",
                    "width", "height", "feasible", "counts"])
        for k, lay in enumerate(layouts):
            _, width, height = lay.fit(TABLES[task["table"]])
            w.writerow([int(-F[k, 0]), int(-F[k, 1]), *G[k].tolist(), round(width, 1),
                        round(height, 1), bool(feas[k]), json.dumps(physical_counts(lay))])
    dump = [{"types": lay.types, "match": [[*a, *b] for a, b in lay.match.items()]}
            for lay in layouts]
    (out / "layouts.json").write_text(json.dumps(dump), encoding="utf-8")
    # pokolenie 1 to populacja startowa, potomstwo powstaje w pokoleniach 2..gens
    offspring = task["cfg"].n_offsprings or task["pop"]
    planned = task["pop"] + offspring * (task["gens"] - 1)
    info = {
        "stage": task["stage"], "variant": task["name"], "seed": task["seed"],
        "kit": task["kit"], "inventory": KITS[task["kit"]],
        "table": task["table"], "max_size": TABLES[task["table"]],
        "pop": task["pop"], "gens": task["gens"],
        "settings": dataclasses.asdict(task["cfg"]),
        "git": git_info(), "python": platform.python_version(),
        "pymoo": pymoo.__version__, "numpy": np.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "times": {**times, "total": times["table"] + times["optimize"]},
        "n_eval": int(res.algorithm.evaluator.n_eval),
        "n_eval_planned": planned,
        "hv": float(HV_RAW(F[feas])) if feas.any() else 0.0,
        "hv_ref": HV_RAW.ref_point.tolist(),
        "n_front": len(layouts), "n_feas_final": int(progress["n_feas"][-1]),
        "best_pieces": float(-F[feas, 0].min()) if feas.any() else 0.0,
        "best_routes": float(-F[feas, 1].min()) if feas.any() else 0.0,
    }
    (out / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


def tasks_for(stage: str, n_seeds: int | None, variants_file: str | None,
              render: bool = False) -> list[dict]:
    seeds = SEEDS.get(stage, SEEDS_DEFAULT)
    n_seeds = n_seeds or N_SEEDS[stage]
    if n_seeds > len(seeds):
        raise SystemExit(f"etap {stage} ma liste {len(seeds)} ziaren, zadano {n_seeds}")
    out = []
    for name, (cfg, pop, gens) in variants(stage, variants_file).items():
        for kit, table in conditions(stage):
            for seed in seeds[:n_seeds]:
                path = ROOT / f"stage{stage}" / name / f"{kit}_{table}" / f"seed_{seed}"
                task = {"stage": stage, "name": name, "cfg": cfg, "pop": pop, "gens": gens,
                        "kit": kit, "table": table, "seed": seed, "out": str(path),
                        "render": render}
                out.append(task)
    return out


def run_pool(tasks: list[dict], workers: int, done: list) -> None:
    if not tasks:
        return
    with multiprocessing.Pool(workers) as pool:
        for status in pool.imap_unordered(run_cell, tasks):
            done.append(status)
            n = len(done)
            print(f"{n:5d} biegow: ok {done.count('ok')}, pominietych {done.count('skip')}, "
                  f"padnietych {done.count('fail')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["0", "1", "2", "2_lite", "3a", "3a_lite", "3b", "4", "5"])
    ap.add_argument("--seeds", type=int, help="ile pierwszych ziaren; domyslnie wg etapu")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--variants", help="plik JSON z wariantami (etapy 3b, 4, 5)")
    ap.add_argument("--render", action="store_true", help="rysunki ukladow w kazdym biegu")
    a = ap.parse_args()

    tasks = tasks_for(a.stage, a.seeds, a.variants, a.render)
    print(f"etap {a.stage}: {len(tasks)} biegow", flush=True)
    build_tables({table_key(t["cfg"]) for t in tasks})
    done: list = []
    # proces z tabela 12 wczytana z dysku zajmuje ok. 4 GB, wiec najwyzej cztery naraz
    heavy = [t for t in tasks if t["cfg"].table_len >= 12]
    light = [t for t in tasks if t["cfg"].table_len < 12]
    run_pool(light, a.workers, done)
    run_pool(heavy, min(a.workers, 4), done)


if __name__ == "__main__":
    main()
