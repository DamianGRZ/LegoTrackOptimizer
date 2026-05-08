"""Tests for Phase 8 -- topology-aware archive admission.

Phase 8 lets the e-archive hold *topologically rich* near-feasible
solutions (any switches/crossings/DCs) even when the strict feasibility
gate would reject them. This is decoupled from the bi-objective F-space
admission used for feasibles. The eviction policy when over-capacity
prefers dropping the highest-CV topology-rich entry first.

Per the plan's Rule 28: the callback must read **raw CV** (sum of G's
clipped at 0), not pymoo's adapted ``ind.get("CV")``. Per Risk 11/Rule 1:
topology signatures are computed lazily in the main process via
``ind.set("topology_sig", ...)`` -- never inside ``_evaluate`` (which
runs in a worker under StarmapParallelization).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.encoding import compute_port_pair_dimensions, create_empty_chromosome
from src_v2.epsilon_archive import EpsilonArchive, EpsilonArchiveCallback


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
WITH_SWITCHES_CFG = (
    Path(__file__).parent.parent / "configs" / "with_switches.yaml"
)


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    cfg = OptimizationConfig.load(WITH_SWITCHES_CFG)
    dims = compute_port_pair_dimensions(cfg.boundary, catalog, cfg.inventory)
    return catalog, cfg, dims


# ------------------------------------------------------------ 8.1 admit-topo
def test_8_admit_topology_aware_admits_topology_rich() -> None:
    """sig with >=1 switch_pairs / cross / DC -> admitted unconditionally."""
    arc = EpsilonArchive(epsilon=(0.005, 0.01), max_size=10)
    x = np.zeros(8, dtype=np.int16)
    f = np.array([-0.5, -1.2])
    sig = (1, 0, 0, 1, 1)
    assert arc.admit_topology_aware(x, f, sig, cv=0.3) is True
    assert len(arc) == 1


# ------------------------------------------------------------ 8.2 reject-empty
def test_8_admit_topology_aware_rejects_topology_empty() -> None:
    """sig with no switches/crossings/DC -> refused; archive unchanged."""
    arc = EpsilonArchive(epsilon=(0.005, 0.01), max_size=10)
    x = np.zeros(8, dtype=np.int16)
    f = np.array([-0.5, -1.2])
    sig = (0, 0, 0, 1, 1)
    assert arc.admit_topology_aware(x, f, sig, cv=0.3) is False
    assert len(arc) == 0


# ------------------------------------------------------------ 8.3 evict-on-overflow
def test_8_admit_topology_aware_drops_highest_cv_on_overflow() -> None:
    """Over-capacity: evict highest-CV topology-aware entry first."""
    arc = EpsilonArchive(epsilon=(0.005, 0.01), max_size=2)
    x1 = np.zeros(8, dtype=np.int16)
    arc.admit_topology_aware(x1, np.array([-0.1, -1.0]), (1, 0, 0, 1, 1), cv=0.1)
    arc.admit_topology_aware(x1, np.array([-0.2, -1.0]), (1, 0, 0, 1, 1), cv=0.5)
    assert len(arc) == 2
    # Adding a third with cv=0.3 -> evicts the cv=0.5 entry
    arc.admit_topology_aware(x1, np.array([-0.3, -1.0]), (1, 0, 0, 1, 1), cv=0.3)
    assert len(arc) == 2
    cvs = sorted(c for c in arc.cvs if c is not None)
    assert cvs == [0.1, 0.3]


# ------------------------------------------------------------ 8.4 feasible-vs-topo
def test_8_feasible_admit_marks_cv_none() -> None:
    """Standard ``admit`` (feasibles) records ``None`` CV alongside its sig."""
    arc = EpsilonArchive(epsilon=(0.005, 0.01), max_size=10)
    arc.admit(np.zeros(8, dtype=np.int16), np.array([-0.5, -1.2]))
    assert len(arc) == 1
    assert arc.cvs == [None]
    assert arc.topology_sigs == [None]


# ------------------------------------------------------------ 8.5 to_json roundtrip
def test_8_to_json_includes_phase8_fields(tmp_path) -> None:
    """JSON serialisation includes ``topology_sigs`` and ``cvs`` arrays."""
    arc = EpsilonArchive(epsilon=(0.005, 0.01), max_size=10)
    arc.admit_topology_aware(
        np.zeros(8, dtype=np.int16),
        np.array([-0.1, -1.0]),
        (1, 0, 0, 1, 1),
        cv=0.2,
    )
    out = tmp_path / "arc.json"
    arc.to_json(out)
    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "topology_sigs" in payload
    assert "cvs" in payload
    assert payload["topology_sigs"] == [[1, 0, 0, 1, 1]]
    assert payload["cvs"] == [0.2]


# ------------------------------------------------------------ 8.6 topology-sig shape
def test_8_topology_sig_for_returns_5tuple(setup) -> None:
    """``_topology_sig_for`` returns a 5-tuple
    ``(n_switch_pairs, n_cross_90, n_dc, n_components, n_cycles)``."""
    catalog, cfg, dims = setup
    cb = EpsilonArchiveCallback(
        epsilon=(0.005, 0.01),
        max_size=10,
        output_path=Path("/tmp/_phase8_test.json"),
        dims=dims,
        catalog=catalog,
        cv_admission_threshold=1.0,
    )
    x = create_empty_chromosome(dims)

    # Use a SimpleNamespace stub instead of a real pymoo Individual to avoid
    # cross-project pymoo state. We need .get/.set semantics.
    cache: dict = {}

    class _Ind:
        def get(self, k):
            if k == "X":
                return x
            return cache.get(k)

        def set(self, k, v):
            cache[k] = v

    ind = _Ind()
    sig = cb._topology_sig_for(ind)
    assert isinstance(sig, tuple)
    assert len(sig) == 5
    assert all(isinstance(v, int) for v in sig)


# ------------------------------------------------------------ 8.7 lazy caching
def test_8_topology_sig_for_caches_on_individual(setup) -> None:
    """Once ``_topology_sig_for`` runs, subsequent calls hit ``ind.get``."""
    catalog, cfg, dims = setup
    cb = EpsilonArchiveCallback(
        epsilon=(0.005, 0.01),
        max_size=10,
        output_path=Path("/tmp/_phase8_test.json"),
        dims=dims,
        catalog=catalog,
        cv_admission_threshold=1.0,
    )
    x = create_empty_chromosome(dims)
    cache: dict = {"X": x}
    n_get = [0]

    class _Ind:
        def get(self, k):
            if k != "X":
                n_get[0] += 1
            return cache.get(k)

        def set(self, k, v):
            cache[k] = v

    ind = _Ind()
    sig1 = cb._topology_sig_for(ind)
    n_after_first = n_get[0]
    sig2 = cb._topology_sig_for(ind)
    assert sig1 == sig2
    # The second call short-circuits: only one extra ``get("topology_sig")``.
    assert n_get[0] - n_after_first == 1


# ------------------------------------------------------------ 8.8 raw CV from G
def test_8_callback_uses_raw_cv_not_adapted_cv(setup) -> None:
    """Rule 28: callback computes raw CV by summing ``max(G, 0)``,
    *not* by reading ``ind.get('CV')``. Verified by populating ``G``
    only and asserting the callback still correctly identifies near-
    feasible individuals."""
    catalog, cfg, dims = setup
    cb = EpsilonArchiveCallback(
        epsilon=(0.005, 0.01),
        max_size=10,
        output_path=Path("/tmp/_phase8_test.json"),
        dims=dims,
        catalog=catalog,
        cv_admission_threshold=1.0,
    )
    # G with violations totalling 0.4 (raw CV); two non-violating slots = 0
    G = np.array([0.1, 0.0, 0.3, -0.5, 0.0])
    raw_cv = cb._raw_cv(G)
    assert raw_cv == pytest.approx(0.4)


# ------------------------------------------------------------ 8.9 cap K
def test_8_notify_caps_near_feasible_scan_at_max_size_div_4(setup) -> None:
    """When pop has > max_size//4 near-feasibles, the scan is bounded at
    ``max_size // 4`` to keep wall-clock overhead in check."""
    catalog, cfg, dims = setup
    cb = EpsilonArchiveCallback(
        epsilon=(0.005, 0.01),
        max_size=8,            # K = 2
        output_path=Path("/tmp/_phase8_test.json"),
        dims=dims,
        catalog=catalog,
        cv_admission_threshold=1.0,
    )
    x = create_empty_chromosome(dims)
    n_decodes = [0]

    class _StubInd:
        def __init__(self, cv: float, g: np.ndarray) -> None:
            self._cv = cv
            self._g = g
            self._cache: dict = {}

        def get(self, k):
            if k == "X":
                return x
            if k == "G":
                return self._g
            if k == "F":
                return np.array([-0.0, -0.0])
            if k == "CV":
                # CV from pymoo, NOT what we should rely on
                return np.array([self._cv])
            return self._cache.get(k)

        def set(self, k, v):
            if k == "topology_sig":
                n_decodes[0] += 1
            self._cache[k] = v

    # 10 near-feasible inds (raw CV 0.1..1.0); cap should be max_size//4 = 2.
    inds = []
    for i in range(10):
        g = np.array([0.05 + 0.005 * i])  # raw CV ~0.05..0.095
        inds.append(_StubInd(cv=0.1 + 0.1 * i, g=g))
    pop_stub = SimpleNamespace(__len__=lambda: 10)
    pop_array = np.array(inds, dtype=object)

    class _Pop:
        def __init__(self, arr): self._arr = arr
        def __len__(self): return len(self._arr)
        def __getitem__(self, idx): return self._arr[idx]
        def get(self, k):
            if k == "G":
                return np.vstack([ind._g for ind in self._arr])
            if k == "CV":
                return np.array([[ind._cv] for ind in self._arr])
            return None

    algo = SimpleNamespace(opt=[], pop=_Pop(pop_array))
    cb.notify(algo)
    # At most K = max_size // 4 = 2 individuals decoded for topology
    assert n_decodes[0] <= 2
