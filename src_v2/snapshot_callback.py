"""Per-generation snapshot writer (Phase 17.B, PLAN Section 10.5).

Captures best feasible and best infeasible layouts at evenly-spaced
generations, writing PNGs + chromosome .npy + JSON metadata to
``outputs_v2/{config}/snapshots/``. Restores V1's diagnostic visualization
with V2 improvements (gen-numbered filenames, ``.npy`` chromosome dump,
JSON sidecar).

Schedule formula (V1 stride): for ``n_gen=N``, snapshots fire at gens
``[N//10, 2·(N//10), ..., 10·(N//10)]``, clamped to ``N`` and deduped.
Avoids the gen-1 random-pop snapshot (low information) and produces 10
post-evolution checkpoints.

Pymoo callbacks live in the main process only (Rule 1), so heavy
matplotlib calls do not need to be pickle-safe and ``StarmapParallelization``
workers never see this object.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pymoo.core.callback import Callback

from .catalog import TrackCatalog
from .config import OptimizationConfig
from .decoder import decode_chromosome, port_graph_to_layout
from .visualization import plot_layout


def compute_snapshot_schedule(n_gen: int, n_snapshots: int = 10) -> list[int]:
    """V1's stride formula, clamped to ``n_gen`` and deduped.

    Returns sorted, unique generation numbers; length never exceeds
    ``n_snapshots``. For small ``n_gen``, the dedupe collapses the tail to
    ``n_gen``.
    """
    n = max(int(n_gen), 1)
    stride = max(1, n // n_snapshots)
    raw = (i * stride for i in range(1, n_snapshots + 1))
    return sorted({min(g, n) for g in raw})


class SnapshotCallback(Callback):
    """Writes ordered snapshots of best feasible / best infeasible layouts.

    Construction wipes prior ``snapshot_*.png`` and ``snapshot_*.npy`` from
    the output directory (V1 pattern) so a re-run cannot be confused with
    stale outputs. Rendering is wrapped in ``try/except`` (matplotlib has
    many failure modes; one bad chromosome must not kill subsequent
    snapshots).
    """

    def __init__(
        self,
        n_gen: int,
        output_dir: Path,
        problem,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        n_snapshots: int = 10,
    ) -> None:
        super().__init__()
        self.schedule = compute_snapshot_schedule(n_gen, n_snapshots)
        self._target_set = set(self.schedule)
        self._taken: set[int] = set()
        self.snapshot_dir = Path(output_dir) / "snapshots"
        self.problem = problem
        self.catalog = catalog
        self.config = config
        self._metadata: list[dict] = []
        self._n: int = 0
        self._logger = logging.getLogger(__name__)
        self._clean_dir()

    def _clean_dir(self) -> None:
        """Wipe prior ``snapshot_*.png`` and ``.npy`` (V1 pattern)."""
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("snapshot_*.png", "snapshot_*.npy"):
            for f in self.snapshot_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass

    def notify(self, algorithm) -> None:
        gen = int(getattr(algorithm, "n_gen", 0))
        if gen not in self._target_set or gen in self._taken:
            return

        pop = getattr(algorithm, "pop", None)
        if pop is None:
            return
        F = pop.get("F")
        X = pop.get("X")
        if F is None or X is None or F.shape[0] == 0:
            return
        G = pop.get("G")
        if G is None:
            G = np.zeros((F.shape[0], 1), dtype=float)

        finite = ~np.isinf(F).any(axis=1)
        feas_mask = np.all(G <= 0, axis=1) & finite
        infeas_mask = (~feas_mask) & finite

        self._taken.add(gen)
        self._n += 1
        meta = {"n": self._n, "gen": gen, "feasible": None, "infeasible": None}

        if feas_mask.any():
            idx = int(np.where(feas_mask)[0][np.argmin(F[feas_mask, 0])])
            meta["feasible"] = self._render(X[idx], F[idx], G[idx], gen, "feasible")

        if infeas_mask.any():
            idx = int(np.where(infeas_mask)[0][np.argmin(F[infeas_mask, 0])])
            meta["infeasible"] = self._render(X[idx], F[idx], G[idx], gen, "infeasible")

        self._metadata.append(meta)
        self._logger.info(
            f"Snapshot {self._n:02d} written at gen {gen} "
            f"(feasible={meta['feasible'] is not None}, "
            f"infeasible={meta['infeasible'] is not None})"
        )

    def _render(
        self,
        x: NDArray,
        f: NDArray,
        g: NDArray,
        gen: int,
        kind: str,
    ) -> Optional[dict]:
        """Render PNG + dump chromosome ``.npy``. Returns metadata dict on success.

        Wrapped in ``try/except`` (D3) so a single bad chromosome does not
        abort the whole snapshot pass.
        """
        try:
            graph = decode_chromosome(
                x, self.problem.dims, self.catalog, self.problem.decoder_config,
            )
            layout = port_graph_to_layout(graph, self.catalog)

            cv = float(np.sum(np.maximum(0, g))) if kind == "infeasible" else 0.0
            spec_by_id = self.catalog.spec.by_id
            n_switch_slots = sum(
                1 for pid in graph.slot_pieces.values()
                if pid in spec_by_id and spec_by_id[pid].kind == "switch"
            )
            n_switches = n_switch_slots // 2
            n_crossings = sum(
                1 for pid in graph.slot_pieces.values()
                if pid in spec_by_id and spec_by_id[pid].kind == "crossing"
            )

            title_parts = [
                f"Snapshot {self._n:02d} (gen {gen})",
                kind.capitalize(),
                f"{graph.n_slots} pcs",
                f"{-f[0]:.1%} util",
                f"{-f[1]:.2f} m/s",
                f"{n_switches} switches",
                f"{graph.n_cycles} cycle",
            ]
            if kind == "infeasible":
                title_parts.append(f"CV={cv:.2f}")
            title = " | ".join(title_parts)

            png_path = self.snapshot_dir / (
                f"snapshot_{self._n:02d}_gen{gen:03d}_{kind}.png"
            )
            chromo_path = self.snapshot_dir / (
                f"snapshot_{self._n:02d}_gen{gen:03d}_{kind}.npy"
            )

            fig = plot_layout(
                layout, self.catalog, self.config.boundary, title, save_path=png_path,
            )
            plt.close(fig)
            np.save(chromo_path, x)

            return {
                "util": float(-f[0]),
                "min_speed": float(-f[1]),
                "n_pieces": int(graph.n_slots),
                "cv": cv,
                "n_switches": n_switches,
                "n_crossings": n_crossings,
                "n_components": int(graph.n_components),
                "n_cycles": int(graph.n_cycles),
                "png_path": str(png_path.relative_to(self.snapshot_dir.parent)),
                "chromosome_path": str(chromo_path.relative_to(self.snapshot_dir.parent)),
            }
        except Exception as e:
            self._logger.warning(
                f"Could not render snapshot {self._n:02d} {kind} at gen {gen}: {e}"
            )
            return None

    def finalize(self) -> None:
        """Write ``snapshots_metadata.json``. Call once after ``minimize()`` returns."""
        meta_path = self.snapshot_dir / "snapshots_metadata.json"
        # OptimizationConfig has no name field; derive from the output dir
        # (run_v2.py writes to outputs_v2/<config_name>/, so .parent.name
        # is the canonical config label).
        config_name = self.snapshot_dir.parent.name or "unknown"
        payload = {
            "config_name": config_name,
            "n_gen": int(getattr(self.config.algorithm, "n_gen", 0)),
            "schedule": self.schedule,
            "snapshots": self._metadata,
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._logger.info(
            f"Snapshot metadata written: {meta_path} "
            f"({len(self._metadata)} snapshots)"
        )
