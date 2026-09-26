"""Ocena ablacji: python -m legoga_real.score_ablation --stage 1 [--base base]

Zbiera run.json z katalogow biegow do outputs/legoga_real_ablation/runs.csv, liczy pole
frontu na wspolnej skali, porownuje kazdy wariant z baza para do pary (Wilcoxon, wielkosc
efektu A12, Fisher dla udzialu poprawnych), wypisuje tabele i rysuje wykresy do plots/.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import false_discovery_control, fisher_exact, wilcoxon  # noqa: E402

ROOT = Path("outputs/legoga_real_ablation")
BUDGETS = (1200, 2400, 4800, 9600, 19200)
BASE = {"2": "pop_60_off_60", "3a": "table_11_win_6", "3b": "table_11_win_6"}
LARGEST = "500x500"


# --- zbieranie ---------------------------------------------------------------

def progress_at(path: Path, budgets) -> dict:
    """Pole frontu przy pierwszym pokoleniu, ktore osiagnelo dany budzet ocen."""
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for b in budgets:
        hit = next((r for r in rows if float(r["n_eval"]) >= b), None)
        out[f"hv_at_{b}"] = float(hit["hv"]) if hit else float("nan")
    return out


def collect(root: Path) -> list[dict]:
    """Jeden wiersz na bieg, z run.json i progress.csv; pole jeszcze surowe."""
    rows = []
    for path in sorted(root.glob("stage*/*/*/seed_*/run.json")):
        info = json.loads(path.read_text(encoding="utf-8"))
        times = info["times"]
        row = {
            "stage": info["stage"], "variant": info["variant"], "kit": info["kit"],
            "table": info["table"], "seed": info["seed"],
            "pop": info["pop"], "gens": info["gens"],
            "n_eval": info["n_eval"], "n_eval_planned": info["n_eval_planned"],
            "hv_raw": info["hv"], "feasible": int(info["hv"] > 0),
            "best_pieces": info["best_pieces"], "best_routes": info["best_routes"],
            "n_feas_final": info["n_feas_final"],
            "pieces_ceiling": sum(info["inventory"].values()),
            "t_table_build": times["table_build"], "t_optimize": times["optimize"],
            "t_per_gen": times["optimize"] / info["gens"],
            "commit": info["git"]["commit"][:8], "dirty": int(info["git"]["dirty"]),
        }
        row.update(progress_at(path.with_name("tory_progress.csv"), BUDGETS))
        rows.append(row)
    return rows


def scale(rows: list[dict]) -> None:
    """Wspolna skala: sufit klockow to suma zestawu, sufit tras to najwiecej z etapu 0
    na najwiekszym stole. Pole = surowe / (sufit klockow * sufit tras)."""
    stage0 = [r["best_routes"] for r in rows if r["stage"] == "0" and r["table"] == LARGEST]
    routes_ceiling = max(stage0 or [r["best_routes"] for r in rows] or [1.0])
    for r in rows:
        denom = r["pieces_ceiling"] * routes_ceiling
        r["routes_ceiling"] = routes_ceiling
        r["hv"] = r["hv_raw"] / denom
        for b in BUDGETS:
            r[f"hv_at_{b}"] = r[f"hv_at_{b}"] / denom


def write_runs(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --- porownanie z baza -------------------------------------------------------

def a12(x, y) -> float:
    """Jak czesto wariant wygrywa z baza, para do pary; remis liczy sie za pol."""
    wins = sum((a > b) + 0.5 * (a == b) for a, b in zip(x, y))
    return wins / len(x)


def paired_p(x, y) -> float:
    d = np.asarray(x, float) - np.asarray(y, float)
    if not np.any(d):
        return 1.0
    return float(wilcoxon(d).pvalue)


def iqr(values) -> str:
    q1, q2, q3 = np.percentile(values, (25, 50, 75))
    return f"{q2:.3f} [{q1:.3f}, {q3:.3f}]"


def compare(rows: list[dict], stage: str, base: str, metric: str = "hv") -> list[dict]:
    """Kazdy wariant kontra baza na tym samym zestawie, stole i ziarnie."""
    by = defaultdict(dict)
    for r in rows:
        if r["stage"] == stage:
            by[(r["variant"], r["kit"], r["table"])][r["seed"]] = r
    out = []
    for (variant, kit, table), runs in sorted(by.items()):
        if variant == base:
            continue
        ref = by.get((base, kit, table), {})
        seeds = sorted(set(runs) & set(ref))
        if not seeds:
            continue
        x = [runs[s][metric] for s in seeds]
        y = [ref[s][metric] for s in seeds]
        fx, fy = sum(runs[s]["feasible"] for s in seeds), sum(ref[s]["feasible"] for s in seeds)
        n = len(seeds)
        out.append({
            "variant": variant, "kit": kit, "table": table, "n": n,
            "variant_med": iqr(x), "base_med": iqr(y),
            "a12": a12(x, y), "p": paired_p(x, y),
            "feas_variant": fx, "feas_base": fy,
            "p_fisher": float(fisher_exact([[fx, n - fx], [fy, n - fy]]).pvalue),
            "t_variant": statistics.median(runs[s]["t_optimize"] for s in seeds),
            "t_base": statistics.median(ref[s]["t_optimize"] for s in seeds),
        })
    if out:
        adj = false_discovery_control([c["p"] for c in out])
        for c, p in zip(out, adj):
            c["p_adj"] = float(p)
    return out


def print_table(cmp: list[dict], metric: str) -> None:
    head = (f"{'wariant':<28} {'zestaw':<8} {'stol':<8} {'n':>2} {metric + ' wariant':>24} "
            f"{metric + ' baza':>24} {'A12':>5} {'p':>6} {'p_adj':>6} {'popr':>7} {'p_F':>6} "
            f"{'s/bieg':>7} {'s baza':>7}")
    print(head)
    for c in cmp:
        print(f"{c['variant']:<28} {c['kit']:<8} {c['table']:<8} {c['n']:>2} "
              f"{c['variant_med']:>24} {c['base_med']:>24} {c['a12']:>5.2f} {c['p']:>6.3f} "
              f"{c['p_adj']:>6.3f} {c['feas_variant']:>3}/{c['feas_base']:<3} "
              f"{c['p_fisher']:>6.3f} {c['t_variant']:>7.0f} {c['t_base']:>7.0f}")


# --- wykresy -----------------------------------------------------------------

def setting_of(variant: str) -> str:
    """Nazwa ustawienia z nazwy wariantu: wszystko przed ostatnim podkresleniem."""
    return variant.rsplit("_", 1)[0] if "_" in variant else variant


def by_condition(rows, stage, kit="real"):
    tables = sorted({r["table"] for r in rows if r["stage"] == stage and r["kit"] == kit})
    return tables, [r for r in rows if r["stage"] == stage and r["kit"] == kit]


def plot_levels(rows, stage, base, metric, out_dir: Path) -> None:
    """Pudelka z ziaren dla kazdego poziomu ustawienia, linia bazy, panel na stol."""
    tables, sub = by_condition(rows, stage)
    settings = defaultdict(set)
    for r in sub:
        if r["variant"] != base:
            settings[setting_of(r["variant"])].add(r["variant"])
    for setting, variants in sorted(settings.items()):
        variants = sorted(variants)
        fig, axes = plt.subplots(1, len(tables), figsize=(4 * len(tables), 4), sharey=True)
        for ax, table in zip(np.atleast_1d(axes), tables):
            data = [[r[metric] for r in sub if r["variant"] == v and r["table"] == table]
                    for v in variants]
            base_vals = [r[metric] for r in sub if r["variant"] == base and r["table"] == table]
            ax.boxplot([d or [np.nan] for d in data],
                       tick_labels=[v[len(setting) + 1:] for v in variants])
            if base_vals:
                ax.axhline(statistics.median(base_vals), color="gray", ls="--", label="baza")
            ax.set_title(table)
            ax.tick_params(axis="x", rotation=45)
        np.atleast_1d(axes)[0].set_ylabel(metric)
        fig.suptitle(f"etap {stage}: {setting}")
        fig.tight_layout()
        fig.savefig(out_dir / f"stage{stage}_{metric}_{setting}.png", dpi=110)
        plt.close(fig)


def plot_quality_vs_time(rows, stage, base, out_dir: Path) -> None:
    """Mediana pola wzgledem mediany czasu, jeden punkt na wariant i stol."""
    tables, sub = by_condition(rows, stage)
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = defaultdict(list)
    for r in sub:
        groups[(r["variant"], r["table"])].append(r)
    for (variant, table), rs in groups.items():
        t = statistics.median(r["t_optimize"] for r in rs)
        h = statistics.median(r["hv"] for r in rs)
        ax.scatter(t, h, color="red" if variant == base else "tab:blue", s=12)
        ax.annotate(variant, (t, h), fontsize=5)
    ax.set_xlabel("czas optymalizacji [s]")
    ax.set_ylabel("hv")
    ax.set_title(f"etap {stage}: jakosc a czas")
    fig.tight_layout()
    fig.savefig(out_dir / f"stage{stage}_hv_vs_time.png", dpi=110)
    plt.close(fig)


def progress_series(root: Path, stage, variant, table, column) -> list[np.ndarray]:
    out = []
    for path in sorted(root.glob(f"stage{stage}/{variant}/real_{table}/seed_*/tory_progress.csv")):
        with path.open(encoding="utf-8") as fh:
            out.append(np.array([[float(r["n_eval"]), float(r[column])]
                                 for r in csv.DictReader(fh)]))
    return out


def median_band(series, grid):
    stack = np.array([np.interp(grid, s[:, 0], s[:, 1], left=np.nan) for s in series])
    return (np.nanpercentile(stack, 25, axis=0), np.nanmedian(stack, axis=0),
            np.nanpercentile(stack, 75, axis=0))


def plot_progress(root, rows, stage, variants, column, out_dir: Path, budgets=()) -> None:
    """Przebieg wybranych wariantow wzgledem liczby ocen: mediana z pasmem, panel na stol."""
    tables, _ = by_condition(rows, stage)
    fig, axes = plt.subplots(1, len(tables), figsize=(4 * len(tables), 4), sharey=True)
    for ax, table in zip(np.atleast_1d(axes), tables):
        for variant in variants:
            series = progress_series(root, stage, variant, table, column)
            if not series:
                continue
            grid = np.linspace(0, max(s[-1, 0] for s in series), 200)
            lo, med, hi = median_band(series, grid)
            ax.plot(grid, med, label=variant)
            ax.fill_between(grid, lo, hi, alpha=0.2)
        for b in budgets:
            ax.axvline(b, color="gray", lw=0.5)
        ax.set_title(table)
        ax.set_xlabel("oceny")
    np.atleast_1d(axes)[0].set_ylabel(column)
    np.atleast_1d(axes)[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / f"stage{stage}_progress_{column}.png", dpi=110)
    plt.close(fig)


def plot_fronts(root, stage, base, winner, table, seed, out_dir: Path) -> None:
    """Koncowe fronty bazy i zwyciezcy na jednym ziarnie."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for variant, color in ((base, "gray"), (winner, "tab:blue")):
        path = root / f"stage{stage}" / variant / f"real_{table}" / f"seed_{seed}" / "front.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            pts = [(int(r["pieces"]), int(r["routes"])) for r in csv.DictReader(fh)
                   if r["feasible"] == "True"]
        if pts:
            ax.scatter(*zip(*pts), color=color, label=variant)
    ax.set_xlabel("klocki")
    ax.set_ylabel("trasy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"stage{stage}_fronts_{winner}_{table}_seed{seed}.png", dpi=110)
    plt.close(fig)


# --- glowne ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--base", help="nazwa wariantu bazowego; domyslnie wg etapu")
    ap.add_argument("--metric", default="hv", help="hv albo hv_at_<budzet>")
    ap.add_argument("--progress", nargs="*", default=[],
                    help="warianty do wykresu przebiegu (best_pieces)")
    ap.add_argument("--fronts", help="zwyciezca do nalozenia frontow z baza")
    ap.add_argument("--table", default="250x350")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    base = a.base or BASE.get(a.stage, "base")

    rows = collect(ROOT)
    if not rows:
        raise SystemExit(f"brak run.json w {ROOT}")
    scale(rows)
    write_runs(rows, ROOT / "runs.csv")
    print(f"{len(rows)} biegow -> {ROOT / 'runs.csv'}; sufit tras {rows[0]['routes_ceiling']}")

    cmp = compare(rows, a.stage, base, a.metric)
    if not cmp:
        raise SystemExit(f"etap {a.stage}: brak par z baza '{base}'")
    print_table(cmp, a.metric)

    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    plot_levels(rows, a.stage, base, a.metric, plots)
    plot_levels(rows, a.stage, base, "t_optimize", plots)
    plot_quality_vs_time(rows, a.stage, base, plots)
    if a.progress:
        plot_progress(ROOT, rows, a.stage, a.progress, "best_pieces", plots)
    if a.stage == "2":
        variants = sorted({r["variant"] for r in rows if r["stage"] == "2"})
        plot_progress(ROOT, rows, "2", variants, "hv", plots, budgets=BUDGETS)
    if a.fronts:
        plot_fronts(ROOT, a.stage, base, a.fronts, a.table, a.seed, plots)
    print(f"Wykresy: {plots}")


if __name__ == "__main__":
    main()
