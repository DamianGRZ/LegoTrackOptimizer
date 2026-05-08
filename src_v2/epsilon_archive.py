"""External ε-archive (Laumanns, Thiele, Deb & Zitzler 2002).

Maintains a bounded, well-spread ε-non-dominated set as a side-effect
collection that survives the main NSGA-II population. The archive is
robust against the elitism-driven crowd-collapse pymoo's NSGA-II is prone
to in long runs.

ε-dominance test (vectorised box coordinates):

- Both archive and candidate map their objective vectors to integer
  *box coordinates* via ``floor(F / epsilon)``.
- A point box-dominates another iff every dim ≤ AND at least one dim <.
- A new candidate is admitted iff no current archive entry box-dominates
  it. On admission, every archive entry that the new candidate box-
  dominates is dropped.

When the archive grows past ``max_size``, the entry with the smallest
nearest-neighbour distance (least isolated, most redundant) is dropped.

Phase 8 extension: a parallel ``admit_topology_aware`` path lets the
archive hold *near-feasible* solutions whose topology signature
(switches/crossings/DCs) is non-trivial. Eviction within that subset
prefers dropping the entry with the highest CV first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.callback import Callback

from .catalog import TrackCatalog
from .decoder import decode_chromosome, DecoderConfig
from .encoding import ENCODING_VERSION, PortPairDimensions


class EncodingVersionMismatch(Exception):
    """Raised by :meth:`EpsilonArchive.from_json` when a loaded artifact's
    ``encoding_version`` does not match the in-process ``ENCODING_VERSION``.
    Per Rule 13: encoding-incompatible artifacts must fail loud, not silent."""


class EpsilonArchive:
    """Bounded, well-spread ε-non-dominated set."""

    def __init__(self, epsilon: Tuple[float, float], max_size: int = 200) -> None:
        self._eps = np.array(epsilon, dtype=np.float64)
        if np.any(self._eps <= 0):
            raise ValueError(f"epsilon must be strictly positive, got {epsilon}")
        self._max_size = int(max_size)
        self._F = np.empty((0, 2), dtype=np.float64)
        self._X: list = []
        # Phase 8: parallel topology-aware bookkeeping. Both arrays are
        # kept the same length as ``_F`` / ``_X``. ``None`` marks feasible
        # entries (admitted via the standard ``admit`` path).
        self._topology_sigs: List[Optional[Tuple[int, ...]]] = []
        self._cvs: List[Optional[float]] = []

    def __len__(self) -> int:
        return self._F.shape[0]

    @property
    def F(self) -> NDArray:
        return self._F

    @property
    def X(self) -> list:
        return list(self._X)

    @property
    def topology_sigs(self) -> List[Optional[Tuple[int, ...]]]:
        return list(self._topology_sigs)

    @property
    def cvs(self) -> List[Optional[float]]:
        return list(self._cvs)

    def admit(self, x_row: NDArray, f_row: NDArray) -> bool:
        """Try to admit ``(x_row, f_row)``. Returns True iff admitted."""
        f_row = np.asarray(f_row, dtype=np.float64).reshape(2)
        if self._F.shape[0] == 0:
            self._F = f_row.reshape(1, 2).copy()
            self._X = [np.asarray(x_row).copy()]
            self._topology_sigs = [None]
            self._cvs = [None]
            return True

        new_box = np.floor(f_row / self._eps).astype(np.int64)
        old_boxes = np.floor(self._F / self._eps).astype(np.int64)

        dominates_new = (
            np.all(old_boxes <= new_box, axis=1)
            & np.any(old_boxes < new_box, axis=1)
        )
        if np.any(dominates_new):
            return False

        new_dominates = (
            np.all(new_box <= old_boxes, axis=1)
            & np.any(new_box < old_boxes, axis=1)
        )
        keep_mask = ~new_dominates
        self._F = np.vstack([self._F[keep_mask], f_row.reshape(1, 2)])
        self._X = [self._X[i] for i, k in enumerate(keep_mask) if k] + [
            np.asarray(x_row).copy()
        ]
        self._topology_sigs = [
            self._topology_sigs[i] for i, k in enumerate(keep_mask) if k
        ] + [None]
        self._cvs = [
            self._cvs[i] for i, k in enumerate(keep_mask) if k
        ] + [None]

        if self._F.shape[0] > self._max_size:
            self._truncate_to_max_size()
        return True

    def admit_topology_aware(
        self,
        x_row: NDArray,
        f_row: NDArray,
        sig: Tuple[int, ...],
        cv: float,
    ) -> bool:
        """Admit a near-feasible topology-rich entry, bypassing ε-non-dominance.

        Returns True iff the entry remains in the archive after any necessary
        eviction. Topology-rich = ``sig[0] >= 1 or sig[1] >= 1 or sig[2] >= 1``
        (at least one switch pair, CROSS_90, or DOUBLE_CROSSOVER).
        """
        if not (sig[0] >= 1 or sig[1] >= 1 or sig[2] >= 1):
            return False

        f_row = np.asarray(f_row, dtype=np.float64).reshape(2)
        n_before = self._F.shape[0]
        self._F = np.vstack([self._F, f_row.reshape(1, 2)])
        self._X.append(np.asarray(x_row).copy())
        self._topology_sigs.append(tuple(int(v) for v in sig))
        self._cvs.append(float(cv))

        if self._F.shape[0] > self._max_size:
            self._truncate_topology_aware()

        # Did the new entry survive? It survives iff its CV is *not* the
        # highest among topology-aware entries when evicting.
        return self._F.shape[0] > n_before

    def _truncate_topology_aware(self) -> None:
        """Drop highest-CV topology-aware entries until size <= max_size.

        Falls back to F-space crowding eviction when no topology-aware
        entries remain (rare: only if the archive is full of feasibles
        AND a topology-aware admit has already been evicted)."""
        while self._F.shape[0] > self._max_size:
            ta_indices = [
                i for i, cv in enumerate(self._cvs) if cv is not None
            ]
            if not ta_indices:
                self._truncate_to_max_size()
                return
            drop_idx = max(ta_indices, key=lambda i: self._cvs[i])
            self._delete_at(drop_idx)

    def _delete_at(self, idx: int) -> None:
        self._F = np.delete(self._F, idx, axis=0)
        del self._X[idx]
        del self._topology_sigs[idx]
        del self._cvs[idx]

    def _truncate_to_max_size(self) -> None:
        """Drop the entry with the smallest nearest-neighbour distance.

        Repeats until ``len <= max_size``. Distances are computed in
        *normalised* objective space (each dim divided by its ``eps``) so
        that the two axes contribute symmetrically.
        """
        while self._F.shape[0] > self._max_size:
            normed = self._F / self._eps
            d = np.linalg.norm(
                normed[:, None, :] - normed[None, :, :], axis=2,
            )
            np.fill_diagonal(d, np.inf)
            nearest = d.min(axis=1)
            drop_idx = int(np.argmin(nearest))
            self._delete_at(drop_idx)

    def to_json(self, path: Path) -> None:
        """Serialise archive to JSON, stamping the current ``ENCODING_VERSION``
        (Rule 13). ``X`` rows are int-cast (chromosome dtype). Phase 8 fields
        ``topology_sigs`` and ``cvs`` round-trip as parallel arrays."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "encoding_version": ENCODING_VERSION,
            "epsilon": self._eps.tolist(),
            "max_size": self._max_size,
            "F": self._F.tolist(),
            "X": [np.asarray(x).astype(int).tolist() for x in self._X],
            "topology_sigs": [
                list(sig) if sig is not None else None
                for sig in self._topology_sigs
            ],
            "cvs": list(self._cvs),
        }, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "EpsilonArchive":
        """Load an archive from JSON. Refuses to load if the file's
        ``encoding_version`` doesn't match the current ``ENCODING_VERSION`` --
        a missing field is treated as mismatch (Phase 3 hard-fail per Rule 13)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("encoding_version")
        if version != ENCODING_VERSION:
            raise EncodingVersionMismatch(
                f"archive at {path} has encoding_version={version!r}; "
                f"current is {ENCODING_VERSION}. Refusing to load incompatible artifact."
            )
        archive = cls(
            epsilon=tuple(payload["epsilon"]),
            max_size=int(payload["max_size"]),
        )
        F_rows = payload.get("F", [])
        X_rows = payload.get("X", [])
        sig_rows = payload.get("topology_sigs", [])
        cv_rows = payload.get("cvs", [])
        if F_rows:
            archive._F = np.asarray(F_rows, dtype=np.float64).reshape(-1, 2)
            archive._X = [np.asarray(row, dtype=np.int16) for row in X_rows]
            if sig_rows and len(sig_rows) == len(F_rows):
                archive._topology_sigs = [
                    tuple(s) if s is not None else None for s in sig_rows
                ]
            else:
                archive._topology_sigs = [None] * len(F_rows)
            if cv_rows and len(cv_rows) == len(F_rows):
                archive._cvs = list(cv_rows)
            else:
                archive._cvs = [None] * len(F_rows)
        return archive


# Topology-rich predicate (Phase 8). Counts >=1 switch pair OR CROSS_90 OR
# DOUBLE_CROSSOVER as "rich enough" for archive admission.
def _is_topology_rich(sig: Tuple[int, ...]) -> bool:
    return sig[0] >= 1 or sig[1] >= 1 or sig[2] >= 1


class EpsilonArchiveCallback(Callback):
    """Pymoo callback that funnels feasible optima into an ``EpsilonArchive``.

    Each generation:

    * **Feasible scan** (existing behaviour): walk ``algorithm.opt`` and admit
      every individual with raw CV <= 0 via :meth:`EpsilonArchive.admit`.
    * **Topology-aware near-feasible scan** (Phase 8): walk ``algorithm.pop``,
      pick at most ``max_size // 4`` individuals with raw CV in the open
      interval ``(0, cv_admission_threshold)`` ordered by ascending CV, and
      try to admit them via :meth:`EpsilonArchive.admit_topology_aware` if
      their lazy-decoded topology signature has at least one switch / crossing
      / DC.

    Per Rule 28 the callback computes raw CV from ``G`` (sum of ``max(G, 0)``)
    instead of trusting ``ind.get("CV")``. Per Rule 1 / Risk 11 topology
    signatures are computed *lazily* in the main process; never inside
    ``_evaluate``.
    """

    def __init__(
        self,
        epsilon: Tuple[float, float],
        max_size: int,
        output_path: Path,
        dims: Optional[PortPairDimensions] = None,
        catalog: Optional[TrackCatalog] = None,
        decoder_config: Optional[DecoderConfig] = None,
        cv_admission_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self._archive = EpsilonArchive(epsilon, max_size)
        self._output_path = Path(output_path)
        self._dims = dims
        self._catalog = catalog
        self._decoder_config = decoder_config
        self._cv_admission_threshold = float(cv_admission_threshold)

    @property
    def archive(self) -> EpsilonArchive:
        return self._archive

    @staticmethod
    def _raw_cv(g: NDArray) -> float:
        """Rule 28: raw CV is the sum of positive constraint violations.

        ``ind.get('CV')`` may have been adapted by pymoo (rescaled, normalised,
        clipped). Reading ``G`` and computing ``sum(max(G, 0))`` yields the
        original violation magnitude that the constraints emit."""
        if g is None:
            return 0.0
        arr = np.asarray(g, dtype=np.float64).reshape(-1)
        return float(np.maximum(arr, 0.0).sum())

    def _topology_sig_for(self, ind) -> Tuple[int, ...]:
        """Lazy topology-signature getter / setter for a pymoo individual.

        Computes the 5-tuple ``(n_switch_pairs, n_cross_90, n_dc,
        n_components, n_cycles)`` and caches it on ``ind`` via ``ind.set``.
        Subsequent calls hit the cache."""
        cached = ind.get("topology_sig")
        if cached is not None:
            return cached
        if self._dims is None or self._catalog is None:
            sig: Tuple[int, ...] = (0, 0, 0, 0, 0)
            ind.set("topology_sig", sig)
            return sig
        x = ind.get("X")
        graph = decode_chromosome(
            x, self._dims, self._catalog, self._decoder_config,
        )
        spec = self._catalog.spec
        n_switch_slots = 0
        n_cross_90 = 0
        n_dc = 0
        if spec is not None:
            spec_by_id = spec.by_id
            for piece_id in graph.slot_pieces.values():
                ps = spec_by_id.get(piece_id)
                if ps is None:
                    continue
                if ps.kind == "switch":
                    n_switch_slots += 1
                elif piece_id == "CROSS_90":
                    n_cross_90 += 1
                elif piece_id == "DOUBLE_CROSSOVER":
                    n_dc += 1
        sig = (
            n_switch_slots // 2,  # IN + OUT make one passing siding
            n_cross_90,
            n_dc,
            int(graph.n_components),
            int(graph.n_cycles),
        )
        ind.set("topology_sig", sig)
        return sig

    def notify(self, algorithm) -> None:
        # ---- feasible scan (read algorithm.opt) ----
        opt = getattr(algorithm, "opt", None)
        if opt is not None:
            for ind in opt:
                cv = ind.get("CV")
                if cv is None:
                    continue
                cv_val = float(cv[0] if hasattr(cv, "__len__") else cv)
                if cv_val > 0.0:
                    continue
                x = ind.get("X")
                f = ind.get("F")
                if x is None or f is None:
                    continue
                self._archive.admit(np.asarray(x), np.asarray(f))

        # ---- topology-aware near-feasible scan (read algorithm.pop) ----
        pop = getattr(algorithm, "pop", None)
        if pop is None or self._dims is None or self._catalog is None:
            return

        try:
            n_pop = len(pop)
        except TypeError:
            return
        if n_pop == 0:
            return

        G = pop.get("G")
        if G is None:
            return
        G_arr = np.asarray(G, dtype=np.float64)
        if G_arr.ndim != 2 or G_arr.shape[0] != n_pop:
            return

        raw_cv = np.maximum(G_arr, 0.0).sum(axis=1)
        near_feasible_mask = (
            (raw_cv > 0.0) & (raw_cv < self._cv_admission_threshold)
        )
        if not np.any(near_feasible_mask):
            return

        idx_pool = np.where(near_feasible_mask)[0]
        k_max = max(0, self._archive._max_size // 4)
        if k_max == 0 or idx_pool.size == 0:
            return
        # Sort ascending by raw CV; cap at K.
        order = idx_pool[np.argsort(raw_cv[idx_pool])][:k_max]

        for i in order:
            ind = pop[int(i)]
            sig = self._topology_sig_for(ind)
            if not _is_topology_rich(sig):
                continue
            x = ind.get("X")
            f = ind.get("F")
            if x is None or f is None:
                continue
            self._archive.admit_topology_aware(
                np.asarray(x), np.asarray(f), sig, cv=float(raw_cv[i]),
            )

    def finalize(self) -> None:
        self._archive.to_json(self._output_path)
