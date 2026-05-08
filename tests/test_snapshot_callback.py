"""Tests for ``src_v2.snapshot_callback`` (Phase 17.B, PLAN Section 10.5).

Covers S.1-S.13 from the §10.5 catalog, with V1's stride-formula schedule and
``_clean_dir``/``try-except`` patterns adopted (D1+D2+D3 deviations approved
by user).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src_v2.snapshot_callback import SnapshotCallback, compute_snapshot_schedule


# ---------------------------------------------------------------- S.1
def test_schedule_n_gen_200_v1_stride():
    assert compute_snapshot_schedule(200) == [
        20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    ]


# ---------------------------------------------------------------- S.2
def test_schedule_n_gen_5_dedupes_to_one_to_five():
    assert compute_snapshot_schedule(5) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------- S.3
def test_schedule_n_gen_10_full_range():
    assert compute_snapshot_schedule(10) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ---------------------------------------------------------------- S.4
def test_schedule_n_gen_1_single_snapshot():
    assert compute_snapshot_schedule(1) == [1]


# ---------------------------------------------------------------- helpers
class _RecordingSnapshotCallback(SnapshotCallback):
    """Subclass that records ``_render`` invocations instead of touching matplotlib."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_calls = []

    def _render(self, x, f, g, gen, kind):
        self.render_calls.append({
            "x_index": int(np.asarray(x).flatten()[0]),
            "f": np.asarray(f).copy(),
            "g": np.asarray(g).copy(),
            "gen": gen,
            "kind": kind,
        })
        return {"util": float(-f[0]), "kind": kind, "gen": gen}


def _stub_algorithm(F, G, X=None, n_gen=1):
    if X is None:
        X = np.arange(F.shape[0]).reshape(-1, 1)
    data = {"F": F, "G": G, "X": X}
    pop = SimpleNamespace(get=lambda key, _d=data: _d.get(key))
    return SimpleNamespace(pop=pop, n_gen=n_gen)


def _make_callback(tmp_path, n_gen=10, recording=True):
    cls = _RecordingSnapshotCallback if recording else SnapshotCallback
    return cls(
        n_gen=n_gen,
        output_dir=tmp_path,
        problem=SimpleNamespace(dims=None, decoder_config=None),
        catalog=SimpleNamespace(spec=SimpleNamespace(by_id={})),
        config=SimpleNamespace(
            algorithm=SimpleNamespace(n_gen=n_gen),
            boundary=None,
            name="test",
        ),
        n_snapshots=10,
    )


# ---------------------------------------------------------------- S.5
def test_fires_only_at_scheduled_gens(tmp_path):
    """For n_gen=10 schedule = [1..10] — every gen fires, exactly once each."""
    cb = _make_callback(tmp_path, n_gen=10)
    F = np.array([[-0.5, -0.886]])
    G = np.zeros((1, 11))

    for gen in range(1, 12):
        cb.notify(_stub_algorithm(F, G, n_gen=gen))

    # Schedule has 10 gens, only those fire; gen=11 is past schedule.
    assert len(cb.render_calls) == 10
    assert {r["gen"] for r in cb.render_calls} == set(range(1, 11))


def test_fires_only_at_scheduled_gens_sparse(tmp_path):
    """For n_gen=200 schedule = [20, 40, ..., 200] — only those fire."""
    cb = _make_callback(tmp_path, n_gen=200)
    F = np.array([[-0.5, -0.886]])
    G = np.zeros((1, 11))

    for gen in range(1, 201):
        cb.notify(_stub_algorithm(F, G, n_gen=gen))

    assert {r["gen"] for r in cb.render_calls} == {
        20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    }


# ---------------------------------------------------------------- S.6
def test_best_feasible_is_argmax_util(tmp_path):
    cb = _make_callback(tmp_path, n_gen=10)
    # 4 individuals, all feasible; util values 0.3, 0.5, 0.7, 0.4 -> best at idx 2
    F = np.array([[-0.3, -0.886], [-0.5, -0.886], [-0.7, -0.886], [-0.4, -0.886]])
    G = np.zeros((4, 11))
    X = np.arange(4).reshape(-1, 1)
    cb.notify(_stub_algorithm(F, G, X=X, n_gen=1))

    assert len(cb.render_calls) == 1
    assert cb.render_calls[0]["kind"] == "feasible"
    assert cb.render_calls[0]["x_index"] == 2


# ---------------------------------------------------------------- S.7
def test_best_infeasible_is_argmax_util(tmp_path):
    cb = _make_callback(tmp_path, n_gen=10)
    # 4 individuals, all infeasible; util 0.2, 0.6, 0.4, 0.5 -> best at idx 1
    F = np.array([[-0.2, -0.886], [-0.6, -0.886], [-0.4, -0.886], [-0.5, -0.886]])
    G = np.ones((4, 11))
    X = np.arange(4).reshape(-1, 1)
    cb.notify(_stub_algorithm(F, G, X=X, n_gen=1))

    assert len(cb.render_calls) == 1
    assert cb.render_calls[0]["kind"] == "infeasible"
    assert cb.render_calls[0]["x_index"] == 1


def test_mixed_pop_picks_best_in_each_subset(tmp_path):
    cb = _make_callback(tmp_path, n_gen=10)
    # idx 0,2 feasible (util .3,.7); idx 1,3 infeasible (util .5,.4)
    F = np.array([[-0.3, -0.886], [-0.5, -0.886], [-0.7, -0.886], [-0.4, -0.886]])
    G = np.zeros((4, 11))
    G[1, 0] = 0.5  # infeasible
    G[3, 0] = 0.5  # infeasible
    X = np.arange(4).reshape(-1, 1)
    cb.notify(_stub_algorithm(F, G, X=X, n_gen=1))

    by_kind = {r["kind"]: r["x_index"] for r in cb.render_calls}
    assert by_kind["feasible"] == 2  # max-util feas (0.7)
    assert by_kind["infeasible"] == 1  # max-util infeas (0.5)


# ---------------------------------------------------------------- S.8
def test_all_feasible_pop_only_feasible_png(tmp_path):
    cb = _make_callback(tmp_path, n_gen=10)
    F = np.array([[-0.5, -0.886], [-0.7, -0.886]])
    G = np.zeros((2, 11))
    cb.notify(_stub_algorithm(F, G, n_gen=1))

    assert [r["kind"] for r in cb.render_calls] == ["feasible"]
    assert cb._metadata[-1]["infeasible"] is None


# ---------------------------------------------------------------- S.9
def test_all_infeasible_pop_only_infeasible_png(tmp_path):
    cb = _make_callback(tmp_path, n_gen=10)
    F = np.array([[-0.5, -0.886], [-0.7, -0.886]])
    G = np.ones((2, 11))
    cb.notify(_stub_algorithm(F, G, n_gen=1))

    assert [r["kind"] for r in cb.render_calls] == ["infeasible"]
    assert cb._metadata[-1]["feasible"] is None


# ---------------------------------------------------------------- S.10
def test_sentinel_inf_population_no_crash(tmp_path):
    cb = _make_callback(tmp_path, n_gen=10)
    F = np.full((3, 2), np.inf)  # unevaluated sentinel
    G = np.zeros((3, 11))
    cb.notify(_stub_algorithm(F, G, n_gen=1))  # should not raise

    assert cb.render_calls == []


def test_empty_pop_no_crash(tmp_path):
    cb = _make_callback(tmp_path, n_gen=10)
    F = np.empty((0, 2))
    G = np.empty((0, 11))
    X = np.empty((0, 1), dtype=np.int16)
    cb.notify(_stub_algorithm(F, G, X=X, n_gen=1))

    assert cb.render_calls == []


# ---------------------------------------------------------------- S.13
def test_chromosome_npy_roundtrip(tmp_path):
    """np.save → np.load yields the same array (foundation for chromosome dumps)."""
    chromo = np.array([1, 2, 3, -1, -1, 4, 5], dtype=np.int16)
    save_path = tmp_path / "chromo.npy"
    np.save(save_path, chromo)
    loaded = np.load(save_path)
    assert np.array_equal(loaded, chromo)
    assert loaded.dtype == chromo.dtype


# ---------------------------------------------------------------- finalize / metadata schema
def test_finalize_writes_metadata_json(tmp_path):
    cb = _make_callback(tmp_path, n_gen=5)
    F = np.array([[-0.5, -0.886]])
    G = np.zeros((1, 11))

    for gen in (1, 2, 3, 4, 5):
        cb.notify(_stub_algorithm(F, G, n_gen=gen))
    cb.finalize()

    meta_path = tmp_path / "snapshots" / "snapshots_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())

    # config_name now derives from output_dir.name (the snapshot dir's parent)
    assert meta["config_name"] == tmp_path.name
    assert meta["n_gen"] == 5
    assert meta["schedule"] == [1, 2, 3, 4, 5]
    assert len(meta["snapshots"]) == 5
    # Per-snapshot schema keys
    snap = meta["snapshots"][0]
    assert {"n", "gen", "feasible", "infeasible"} <= set(snap)


# ---------------------------------------------------------------- D2: clean_dir wipes prior PNGs
def test_clean_dir_wipes_prior_png_and_npy(tmp_path):
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    stale_png = snap_dir / "snapshot_99_gen999_feasible.png"
    stale_npy = snap_dir / "snapshot_99_gen999_feasible.npy"
    stale_png.write_text("stale data")
    stale_npy.write_text("stale data")
    assert stale_png.exists() and stale_npy.exists()

    _make_callback(tmp_path, n_gen=10)  # construction calls _clean_dir

    assert not stale_png.exists()
    assert not stale_npy.exists()


# ---------------------------------------------------------------- D3: try/except in _render
def test_render_failure_logged_not_raised(tmp_path):
    """A broken catalog/decode should not kill the whole snapshot pass."""
    # Use NON-recording subclass so real _render runs, but with a stub problem
    # whose decode_chromosome will fail (problem.dims=None breaks decoder).
    cb = SnapshotCallback(
        n_gen=10,
        output_dir=tmp_path,
        problem=SimpleNamespace(dims=None, decoder_config=None),
        catalog=SimpleNamespace(spec=SimpleNamespace(by_id={})),
        config=SimpleNamespace(
            algorithm=SimpleNamespace(n_gen=10), boundary=None, name="test",
        ),
        n_snapshots=10,
    )
    F = np.array([[-0.5, -0.886]])
    G = np.zeros((1, 11))
    cb.notify(_stub_algorithm(F, G, n_gen=1))  # _render fails internally → caught

    # _metadata still records the slot but feasible/infeasible are None
    assert cb._metadata[-1]["feasible"] is None
