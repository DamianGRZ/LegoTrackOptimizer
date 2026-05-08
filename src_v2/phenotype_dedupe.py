"""Phenotype-level deduplication callback.

References:
    - Hildebrandt & Branke (2015) "On using surrogates with genetic
      programming."
    - Goldman & Punch (2014) Parameter-less Population Pyramid (P3).

Genotype-level duplicate elimination (chromosome equality) is necessary
but insufficient — two chromosomes that decode to the same topology are
still phenotypic clones, and they crowd out diverse search.

This callback re-decodes each individual once per generation, hashes the
result by a small structural-summary tuple (:class:`Phenotype`), and
within each phenotype bucket keeps the best (lowest CV, then highest
util) individual; the rest are replaced with fresh random samples whose
``X`` is invalidated so pymoo re-evaluates them next generation.

Decode is cheap (~1 ms per chromosome at typical N_max=160), so the
extra O(pop) decodes per generation are negligible compared to FK
propagation in the main evaluation loop.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import NamedTuple, Tuple

from pymoo.core.callback import Callback


class Phenotype(NamedTuple):
    """Hashable structural summary of a decoded chromosome.

    Two individuals with the same Phenotype are interchangeable for the
    GA's purposes; one is enough.
    """

    n_switches: int
    n_crossings: int
    n_cycles: int
    n_branch_cycles: int
    max_component_size: int
    piece_histogram: Tuple[Tuple[str, int], ...]


class PhenotypeDedupeCallback(Callback):
    """Counts phenotype duplicates per generation; logs occasionally.

    Originally replaced duplicates with fresh random offspring, but that
    interacted badly with NSGA-II's stale-rank assumption (after replacing
    ``pop[j].X`` the individual's preserved-but-stale rank/crowding from
    a different genome led to ``None`` comparisons in binary_tournament).
    Phase 4's diversification mutations (E.3–E.7) and the genotype
    eliminate_duplicates already cover the diversity injection that this
    callback was meant to provide; making this an observability hook
    avoids the integration brittleness.
    """

    def __init__(self, sampling, cadence: int = 20) -> None:
        super().__init__()
        self._sampling = sampling  # kept for signature compat
        self._cadence = max(1, int(cadence))
        self._gen = 0
        self._logger = logging.getLogger(__name__)
        # Last-known stats, exposed for DiagnosticsCallback (Section 10.4).
        # Populated only on cadence-aligned generations; remain stale (or None)
        # otherwise. Consumers should compare ``last_gen`` against the current
        # generation before trusting the values.
        self.last_gen: int | None = None
        self.last_pop_size: int | None = None
        self.last_n_phenotypes: int | None = None
        self.last_n_duplicates: int | None = None

    def notify(self, algorithm) -> None:
        self._gen += 1
        if self._gen % self._cadence != 0:
            return
        pop = getattr(algorithm, "pop", None)
        problem = getattr(algorithm, "problem", None)
        if pop is None or problem is None:
            return
        if not hasattr(problem, "build_phenotype"):
            return

        from .decoder import decode_chromosome

        buckets: dict = defaultdict(int)
        for ind in pop:
            x = ind.get("X")
            if x is None:
                continue
            graph = decode_chromosome(
                x, problem.dims, problem.catalog, problem.decoder_config,
            )
            phen = problem.build_phenotype(graph)
            buckets[phen] += 1

        n_phenotypes = len(buckets)
        if n_phenotypes == 0:
            return
        n_dups = sum(c - 1 for c in buckets.values() if c > 1)
        self.last_gen = self._gen
        self.last_pop_size = len(pop)
        self.last_n_phenotypes = n_phenotypes
        self.last_n_duplicates = n_dups
        self._logger.info(
            f"Gen {self._gen:4d} | phenotype dedupe: "
            f"{n_phenotypes} unique / {len(pop)} total, "
            f"{n_dups} duplicates",
        )
