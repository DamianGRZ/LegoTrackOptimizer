"""Score the ablation campaign: hypervolume, cost and paired sign tests.

Every arm of a config is scored inside ONE ideal/nadir box built from the union
of that config's archives, so the numbers are comparable across arms. Raw HV is
never pooled across configs -- ``special_piece_weight`` differs between them, so
F[0] is not on a common scale.

Adding an arm widens the box and therefore moves every previously published
number: rescore the whole campaign in one pass rather than appending rows.
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import numpy as np
from pymoo.indicators.hv import HV
from scipy.stats import binomtest

from src.config import AlgorithmConfig
from src.normalization import HV_REF_POINT, has_extent, ideal_nadir

OUTPUT_ROOT = Path("outputs/ablation")
REFERENCE_ARM = "full"

# Arms carrying a non-default crowding metric are named "<arm>_<metric>" by the
# driver; the same field feeds both scripts so the convention cannot drift.
CROWDING_FUNCS = get_args(AlgorithmConfig.model_fields["crowding_func"].annotation)

# "- **Best feasible**: 55 pieces, util=76.4%, ..." -- absent when the run never
# reached feasibility, which is exactly how a failed cell is recognised.
BEST_FEASIBLE = re.compile(r"\*\*Best feasible\*\*: (\d+) pieces, util=([\d.]+)%")


@dataclass(frozen=True)
class Cell:
    """One (config, arm, seed) run and everything scored off its artifacts."""

    config: str
    arm: str
    seed: int
    front: np.ndarray  # feasible archive, shape (n, 2); empty when none found
    utilization: float  # nan when no feasible layout exists
    seconds: float

    @property
    def solved(self) -> bool:
        return len(self.front) > 0


def _read_front(run_dir: Path) -> np.ndarray:
    """The run-cumulative feasible front, or an empty (0, 2) array."""
    archive = run_dir / "pareto_archive.csv"
    if not archive.exists():
        return np.empty((0, 2))
    return np.loadtxt(archive, delimiter=",", skiprows=1, ndmin=2)


def _read_utilization(info: str) -> float:
    match = BEST_FEASIBLE.search(info)
    return float(match.group(2)) if match else float("nan")


def _read_seconds(run_dir: Path) -> float:
    """Wall-clock spent in generations, summed off ``convergence.csv``."""
    convergence = run_dir / "convergence.csv"
    if not convergence.exists():
        return float("nan")
    with convergence.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sum(float(row["gen_seconds"]) for row in rows)


def read_cells(root: Path) -> list[Cell]:
    """Every finished cell under ``root``.

    A cell without ``## Run Summary`` died before writing artifacts; scoring it
    as zero would penalise whichever arm crashes more, so it is skipped instead.
    """
    cells = []
    for info_path in sorted(root.glob("*/*/s*/run_info.md")):
        info = info_path.read_text(encoding="utf-8")
        if "## Run Summary" not in info:
            continue
        run_dir = info_path.parent
        config, arm = run_dir.parent.parent.name, run_dir.parent.name
        cells.append(Cell(
            config=config,
            arm=arm,
            seed=int(run_dir.name.lstrip("s")),
            front=_read_front(run_dir),
            utilization=_read_utilization(info),
            seconds=_read_seconds(run_dir),
        ))
    return cells


def score_config(cells: list[Cell]) -> dict[tuple[str, int], float]:
    """Hypervolume per (arm, seed) for one config, in a box shared by all arms.

    An arm that found nothing scores 0. A config whose pooled front has no
    extent on some axis has no 0-1 mapping at all, so every cell scores 0 --
    normalizing it anyway would produce a number about nothing.
    """
    solved = [cell for cell in cells if cell.solved]
    zeros = {(cell.arm, cell.seed): 0.0 for cell in cells}
    if not solved:
        return zeros

    ideal, nadir = ideal_nadir(np.vstack([cell.front for cell in solved]))
    if not has_extent(ideal, nadir):
        return zeros

    indicator = HV(ref_point=np.array(HV_REF_POINT), norm_ref_point=False,
                   zero_to_one=True, ideal=ideal, nadir=nadir)
    return {**zeros,
            **{(cell.arm, cell.seed): float(indicator(cell.front)) for cell in solved}}


def score_all(cells: list[Cell]) -> dict[tuple[str, str, int], float]:
    by_config: dict[str, list[Cell]] = {}
    for cell in cells:
        by_config.setdefault(cell.config, []).append(cell)
    return {(config, arm, seed): hv
            for config, group in by_config.items()
            for (arm, seed), hv in score_config(group).items()}


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def block_winners(hv: dict[tuple[str, str, int], float]) -> tuple[dict, dict]:
    """Per (config, seed) block, who took the highest HV — sole wins and tied tops.

    Arms can land on bit-identical HV, so awarding the block to one of them would
    make the ranking depend on iteration order. Ties are reported, never broken.
    """
    blocks: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for (config, arm, seed), value in hv.items():
        blocks.setdefault((config, seed), []).append((arm, value))

    arms = {arm for _, arm, _ in hv}
    sole = dict.fromkeys(arms, 0)
    tied = dict.fromkeys(arms, 0)
    for entries in blocks.values():
        best = max(value for _, value in entries)
        if best <= 0:
            continue
        winners = [arm for arm, value in entries if value == best]
        target = sole if len(winners) == 1 else tied
        for arm in winners:
            target[arm] += 1
    return sole, tied


def print_arm_summary(cells: list[Cell], hv: dict[tuple[str, str, int], float]) -> None:
    sole, tied = block_winners(hv)
    print(f"\n{'arm':<24}{'cells':>7}{'solved':>8}{'mean HV':>10}{'median HV':>11}"
          f"{'HV wins':>9}{'tied top':>10}{'med front':>11}{'med util':>10}"
          f"{'mean s/run':>12}")
    for arm in sorted({cell.arm for cell in cells}):
        arm_cells = [cell for cell in cells if cell.arm == arm]
        values = [hv[(cell.config, arm, cell.seed)] for cell in arm_cells]
        fronts = [float(len(cell.front)) for cell in arm_cells if cell.solved]
        utils = [cell.utilization for cell in arm_cells if cell.solved]
        seconds = [cell.seconds for cell in arm_cells if np.isfinite(cell.seconds)]
        print(f"{arm:<24}{len(arm_cells):>7}"
              f"{sum(cell.solved for cell in arm_cells):>8}"
              f"{np.mean(values):>10.3f}{_median(values):>11.3f}{sole.get(arm, 0):>9}"
              f"{tied.get(arm, 0):>10}"
              f"{_median(fronts):>11.0f}{_median(utils):>10.1f}{np.mean(seconds):>12.1f}")


def print_config_table(cells: list[Cell], hv: dict[tuple[str, str, int], float]) -> None:
    """Median HV over seeds, per config and arm. Ranking is by HV only."""
    arms = sorted({cell.arm for cell in cells})
    configs = sorted({cell.config for cell in cells})
    print("\nMedian HV over seeds (best arm per row marked *)\n")
    print(f"{'config':<30}" + "".join(f"{arm[:13]:>14}" for arm in arms))
    for config in configs:
        medians = {arm: _median([value for (cfg, cell_arm, _seed), value in hv.items()
                                 if cfg == config and cell_arm == arm])
                   for arm in arms}
        best = max(medians.values())
        cells_text = "".join(
            f"{medians[arm]:>13.3f}{'*' if medians[arm] == best and best > 0 else ' '}"
            for arm in arms)
        print(f"{config:<30}{cells_text}")


def _sign_test_row(hv: dict[tuple[str, str, int], float],
                   blocks: list[tuple[str, int]], better: str, worse: str) -> str:
    """One rendered row of a one-sided sign test of ``better`` against ``worse``.

    Only blocks in which both arms ran contribute.
    """
    deltas = np.array([hv[(config, better, seed)] - hv[(config, worse, seed)]
                       for config, seed in blocks
                       if (config, better, seed) in hv and (config, worse, seed) in hv])
    wins, losses = int((deltas > 0).sum()), int((deltas < 0).sum())
    ties = len(deltas) - wins - losses
    p = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue \
        if wins + losses else float("nan")
    mean_delta = deltas.mean() if len(deltas) else float("nan")
    return (f"{f'{better} > {worse}':<52}{wins:>6}{losses:>8}{ties:>6}"
            f"{mean_delta:>12.3f}{p:>12.2g}")


def _sign_test_header(title: str) -> None:
    print(f"\n{title}\n")
    print(f"{'comparison':<52}{'wins':>6}{'losses':>8}{'ties':>6}"
          f"{'mean delta':>12}{'p':>12}")


def print_sign_tests(cells: list[Cell], hv: dict[tuple[str, str, int], float],
                     reference: str) -> None:
    """One-sided sign test of ``reference`` against every other arm, over blocks.

    Blocking, not pairing: arms with different samplers consume the RNG stream
    differently, so a shared seed index is not a common random number.
    """
    blocks = sorted({(cell.config, cell.seed) for cell in cells if cell.arm == reference})
    _sign_test_header(f"Paired over {len(blocks)} (config, seed) blocks "
                      "-- one-sided sign test")
    for arm in sorted({cell.arm for cell in cells} - {reference}):
        print(_sign_test_row(hv, blocks, reference, arm))


def crowding_pairs(arms: set[str]) -> list[tuple[str, str]]:
    """``(cd arm, variant arm)`` pairs present in the archive."""
    return sorted((variant[:-(len(func) + 1)], variant)
                  for variant in arms for func in CROWDING_FUNCS
                  if variant.endswith(f"_{func}") and variant[:-(len(func) + 1)] in arms)


def print_crowding_comparisons(cells: list[Cell],
                               hv: dict[tuple[str, str, int], float]) -> None:
    """Sign test of every crowding-metric variant against its own cd arm.

    This is the paired reading the campaign is for: both arms of a pair differ
    in the survival metric alone, so a block where both ran isolates it.
    """
    pairs = crowding_pairs({cell.arm for cell in cells})
    if not pairs:
        return
    # Every block in the archive is offered; a pair contributes only those in
    # which both of its arms ran, so read the counts per row, not off a total.
    blocks = sorted({(cell.config, cell.seed) for cell in cells})
    _sign_test_header("Crowding metric, each variant against its own cd arm "
                      "-- one-sided sign test over the blocks both ran")
    for base, variant in pairs:
        print(_sign_test_row(hv, blocks, variant, base))


def write_scores(path: Path, cells: list[Cell],
                 hv: dict[tuple[str, str, int], float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "arm", "seed", "hv", "front_size",
                         "utilization_pct", "gen_seconds"])
        for cell in sorted(cells, key=lambda c: (c.config, c.arm, c.seed)):
            writer.writerow([cell.config, cell.arm, cell.seed,
                             f"{hv[(cell.config, cell.arm, cell.seed)]:.6f}",
                             len(cell.front), f"{cell.utilization:.1f}",
                             f"{cell.seconds:.1f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset to score (default: every arm found)")
    parser.add_argument("--reference", default=REFERENCE_ARM,
                        help=f"arm the sign tests compare against (default: {REFERENCE_ARM})")
    args = parser.parse_args()

    cells = read_cells(args.root)
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",")}
        cells = [cell for cell in cells if cell.arm in keep]
    if not cells:
        parser.error(f"no finished runs under {args.root}")

    hv = score_all(cells)
    configs = {cell.config for cell in cells}
    arms = {cell.arm for cell in cells}
    # Completeness is measured against the seeds actually present, so adding a
    # replication does not turn every pair into a false warning.
    n_seeds = len({cell.seed for cell in cells})
    print(f"{len(cells)} finished cells: {len(arms)} arms x {len(configs)} configs "
          f"x {n_seeds} seeds")
    incomplete = [(config, arm) for config in sorted(configs) for arm in sorted(arms)
                  if sum(c.config == config and c.arm == arm for c in cells) != n_seeds]
    if incomplete:
        print(f"WARNING: {len(incomplete)} (config, arm) pairs lack {n_seeds} seeds: "
              f"{incomplete[:6]}{' ...' if len(incomplete) > 6 else ''}")

    print_arm_summary(cells, hv)
    print_config_table(cells, hv)
    print_crowding_comparisons(cells, hv)
    if args.reference in arms:
        print_sign_tests(cells, hv, args.reference)

    scores = args.root / "scores.csv"
    write_scores(scores, cells, hv)
    print(f"\nPer-cell scores written to {scores}")


if __name__ == "__main__":
    main()
