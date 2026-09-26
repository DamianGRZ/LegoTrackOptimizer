"""Strojone ustawienia biegu. Domyslne wartosci = dotychczasowe zachowanie."""
from __future__ import annotations

from dataclasses import dataclass

NEUTRAL_BLOCKS = (("S16",), ("S24",), ("CL", "CR"), ("CR", "CL"),
                  ("S16", "S16"), ("CL", "CR", "S16"), ("S16", "CL", "CR"))


@dataclass(frozen=True)
class Settings:
    n_offsprings: int | None = None      # None = tyle co populacja
    mut_prob: float = 0.9
    n_ops: int = 2                       # mutacji na osobnika: losowo 1..n_ops
    cx_prob: float = 0.9
    pressure: int = 2                    # kandydatow w turnieju o rodzica
    dedupe: bool = True
    sig_with_size: bool = True
    forget_each_gen: bool = True

    table_len: int = 11
    per_bucket: int | None = 60
    max_win: int = 6                     # ile klockow wycina podmiana i krzyzowanie

    swap_tries: int = 15
    branch_attempts: int = 30
    branch_seqs: int = 6
    xx_attempts: int = 12
    xx_limit: int = 2
    xx_scan: int = 1000
    dc_attempts: int = 12
    dc_limit: int = 8
    dc_scan: int = 4000
    cx_tries: int = 20

    accept: str = "first"                # albo "best": najlepszy ze wszystkich prob
    mutate_draws: int | None = 8         # None: kazdy operator raz, w losowej kolejnosci

    oval_sizes: tuple[int, ...] = (4, 6, 8, 10, 12, 14)
    figure_eight: bool = True
    warmup: tuple[int, int] = (3, 25)

    weights: tuple[float, ...] = (0.30, 0.16, 0.19, 0.07, 0.10, 0.04, 0.10, 0.04)
    neutral_blocks: tuple[tuple[str, ...], ...] = NEUTRAL_BLOCKS
    antipodal_min_chain: int = 4
    cx_min_chain: int = 3
    dc_refill: str = "random"            # albo "s16": zawsze 3xS16
    cx_tol: float = 0.05
