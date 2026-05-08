"""Diagnostic instrumentation for the V2 optimizer (Section 10.4 of docs/PLAN.md).

`DiagnosticsCallback` writes a per-generation CSV that captures feasibility
counts, CV percentiles, per-constraint mean and p90, the topology mix
(switch / crossing chromosome counts), repair-CV columns reserved for
Phase 1, dedupe rejection rate (when wired to a fresh PhenotypeDedupeCallback),
and the util / min-speed distribution restricted to feasibles.

Per Rule 1 (custom ``out`` keys do not round-trip under
``StarmapParallelization``), all statistics are computed in the main process
from ``algorithm.pop`` rather than via worker-set keys.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from pymoo.core.callback import Callback


class DiagnosticsCallback(Callback):
    """Per-generation CSV writer over ``algorithm.pop``.

    The first ``notify`` truncates ``{output_dir}/diagnostics.csv`` and writes
    the schema header. Subsequent calls append one row each.

    Args:
        output_dir: directory to receive ``diagnostics.csv``. Created if missing.
        n_constraints: total constraint count for the problem (``11 + T`` for
            V2's port-pair encoding, where ``T = catalog.n_pieces``).
        n_max: chromosome piece-slot region width (``dims.N_max``). Required
            for switch / crossing chromosome counting; pass 0 to disable.
        switch_piece_indices: catalog piece indices whose ``kind`` is "switch".
            Used by ``np.isin`` against the chromosome's piece-slot region.
        crossing_piece_indices: catalog piece indices for CROSS_90 /
            DOUBLE_CROSSOVER (``PieceClass.CROSSING_4PORT``).
        dedupe_callback: optional ``PhenotypeDedupeCallback``. If provided and
            its ``last_gen`` matches the current generation, ``dedupe_rejection_rate``
            is populated as ``last_n_duplicates / last_pop_size``. Otherwise
            blank — typically because dedupe ran on a different cadence.
    """

    def __init__(
        self,
        output_dir: str | Path,
        n_constraints: int,
        *,
        n_max: int = 0,
        switch_piece_indices: Optional[Sequence[int]] = None,
        crossing_piece_indices: Optional[Sequence[int]] = None,
        dedupe_callback: Optional[Callback] = None,
    ) -> None:
        super().__init__()
        self._output_path = Path(output_dir) / "diagnostics.csv"
        self._n_constraints = int(n_constraints)
        self._n_max = int(n_max)
        self._switch_indices = np.asarray(
            switch_piece_indices if switch_piece_indices is not None else [],
            dtype=np.int16,
        )
        self._crossing_indices = np.asarray(
            crossing_piece_indices if crossing_piece_indices is not None else [],
            dtype=np.int16,
        )
        self._dedupe_cb = dedupe_callback
        self._header_written = False

    @property
    def columns(self) -> list[str]:
        return [
            "gen", "n_feasible", "n_infeasible",
            "mean_cv", "median_cv", "p90_cv", "p99_cv",
            *(f"constraint_{i}_mean" for i in range(self._n_constraints)),
            *(f"constraint_{i}_p90" for i in range(self._n_constraints)),
            "n_chromosomes_with_switches", "n_chromosomes_with_crossings",
            "mean_pre_repair_cv", "mean_post_repair_cv",
            "dedupe_rejection_rate",
            "mean_util_feasible", "max_util_feasible",
            "mean_min_speed_feasible", "max_min_speed_feasible",
        ]

    def notify(self, algorithm) -> None:
        pop = getattr(algorithm, "pop", None)
        if pop is None:
            return
        F = pop.get("F")
        if F is None or F.shape[0] == 0:
            return
        G = pop.get("G")
        cv = pop.get("CV").flatten()

        feasible_mask = cv <= 0.0
        n_feasible = int(feasible_mask.sum())
        n_infeasible = int(F.shape[0] - n_feasible)
        gen = int(getattr(algorithm, "n_gen", 0))

        g_means = G.mean(axis=0)
        g_p90 = np.percentile(G, 90, axis=0)

        row: dict[str, object] = {
            "gen": gen,
            "n_feasible": n_feasible,
            "n_infeasible": n_infeasible,
            "mean_cv": float(cv.mean()),
            "median_cv": float(np.median(cv)),
            "p90_cv": float(np.percentile(cv, 90)),
            "p99_cv": float(np.percentile(cv, 99)),
        }
        row.update({f"constraint_{i}_mean": float(g_means[i]) for i in range(self._n_constraints)})
        row.update({f"constraint_{i}_p90": float(g_p90[i]) for i in range(self._n_constraints)})

        # Switch / crossing chromosome counts via vectorized membership.
        X = pop.get("X")
        if X is not None and self._n_max > 0:
            piece_slots = X[:, : self._n_max]
            row["n_chromosomes_with_switches"] = (
                int(np.isin(piece_slots, self._switch_indices).any(axis=1).sum())
                if self._switch_indices.size else 0
            )
            row["n_chromosomes_with_crossings"] = (
                int(np.isin(piece_slots, self._crossing_indices).any(axis=1).sum())
                if self._crossing_indices.size else 0
            )
        else:
            row["n_chromosomes_with_switches"] = ""
            row["n_chromosomes_with_crossings"] = ""

        # Pre/post-repair CV: V2's Lamarckian flow puts ``cv`` already at
        # post-repair value, but pre-repair CV requires re-evaluating the
        # un-repaired chromosomes — an extra full-pop eval per generation.
        # Phase 1's Baldwinian repair (Rule 24 revised) introduces a pheno
        # passthrough that splits the two; until then both stay blank rather
        # than misleadingly duplicate ``mean_cv`` into one column.
        row["mean_pre_repair_cv"] = ""
        row["mean_post_repair_cv"] = ""

        # Dedupe rejection rate from a paired PhenotypeDedupeCallback.
        # The dedupe callback computes its bucketing on a cadence (default 20).
        # We accept its stats only when ``last_gen == current gen`` — otherwise
        # the stats are stale and writing them would be misleading.
        if self._dedupe_cb is not None:
            last_gen = getattr(self._dedupe_cb, "last_gen", None)
            last_pop = getattr(self._dedupe_cb, "last_pop_size", None)
            last_dups = getattr(self._dedupe_cb, "last_n_duplicates", None)
            if last_gen == gen and last_pop:
                row["dedupe_rejection_rate"] = float(last_dups) / float(last_pop)
            else:
                row["dedupe_rejection_rate"] = ""
        else:
            row["dedupe_rejection_rate"] = ""

        if n_feasible > 0:
            util_feas = -F[feasible_mask, 0]
            speed_feas = -F[feasible_mask, 1]
            row["mean_util_feasible"] = float(util_feas.mean())
            row["max_util_feasible"] = float(util_feas.max())
            row["mean_min_speed_feasible"] = float(speed_feas.mean())
            row["max_min_speed_feasible"] = float(speed_feas.max())
        else:
            row["mean_util_feasible"] = ""
            row["max_util_feasible"] = ""
            row["mean_min_speed_feasible"] = ""
            row["max_min_speed_feasible"] = ""

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._header_written else "a"
        with self._output_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
