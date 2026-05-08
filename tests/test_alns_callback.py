"""Tests for the Phase 5 ALNS callback (operator-weight tuning)."""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from src_v2.alns_callback import ALNSCallback


class _StubMutation:
    """Minimal stub matching the slice of PortPairMutation that ALNS reads."""

    def __init__(self, op_names, weights):
        self.OP_WEIGHTS = OrderedDict(zip(op_names, weights))
        self._op_names = tuple(op_names)
        self._last_op_indices: tuple = ()
        self._cdf_rebuilt = 0

    def _build_cdf(self):
        self._cdf_rebuilt += 1


def _stub_algorithm(cv: np.ndarray):
    """Return a SimpleNamespace mimicking the slice of pymoo Algorithm
    that ``ALNSCallback.notify`` reads."""
    off = SimpleNamespace(get=lambda key: cv if key == "CV" else None)
    return SimpleNamespace(off=off)


def test_floor_respected_when_one_op_dominates():
    mut = _StubMutation(["a", "b", "c"], [0.4, 0.3, 0.3])
    cb = ALNSCallback(reweight_every=1, exploration_floor=0.05)
    cb.attach_to(mut)

    # Op 'a' wins every round, 'b' and 'c' produce CV=10 (penalty).
    last_ops = (0, 0, 0, 0)
    cv = np.array([0, 0, 0, 0])  # all feasible → all credited
    mut._last_op_indices = last_ops
    cb.notify(_stub_algorithm(cv))
    # 'b' and 'c' got 0 calls so they receive 0 credit, but the floor saves them.
    assert min(mut.OP_WEIGHTS.values()) >= 0.05


def test_weights_sum_to_one_after_reweight():
    mut = _StubMutation(["a", "b"], [0.5, 0.5])
    cb = ALNSCallback(reweight_every=1)
    cb.attach_to(mut)
    mut._last_op_indices = (0, 1, 0, 1)
    cb.notify(_stub_algorithm(np.array([0, 5, 0, 5])))
    s = sum(mut.OP_WEIGHTS.values())
    assert abs(s - 1.0) < 1e-9


def test_no_reweight_before_K_generations():
    mut = _StubMutation(["a", "b"], [0.5, 0.5])
    weights_before = dict(mut.OP_WEIGHTS)
    cb = ALNSCallback(reweight_every=10)
    cb.attach_to(mut)
    mut._last_op_indices = (0, 1)
    for _ in range(5):  # Less than K
        cb.notify(_stub_algorithm(np.array([0.0, 0.0])))
    assert dict(mut.OP_WEIGHTS) == weights_before


def test_reweight_at_kth_generation():
    mut = _StubMutation(["a", "b"], [0.5, 0.5])
    cb = ALNSCallback(reweight_every=3)
    cb.attach_to(mut)
    # 3 gens of credit-only-to-op-0.
    mut._last_op_indices = (0, 0)
    for _ in range(3):
        cb.notify(_stub_algorithm(np.array([0.0, 0.0])))
    # Op 'a' should be heavier than 'b' after reweight.
    assert mut.OP_WEIGHTS["a"] > mut.OP_WEIGHTS["b"]


def test_no_attach_no_op():
    """notify() with no mutation attached must not crash."""
    cb = ALNSCallback()
    cb.notify(_stub_algorithm(np.array([0.0])))


def test_off_none_no_op():
    mut = _StubMutation(["a"], [1.0])
    cb = ALNSCallback(reweight_every=1)
    cb.attach_to(mut)
    weights_before = dict(mut.OP_WEIGHTS)
    cb.notify(SimpleNamespace(off=None))
    assert dict(mut.OP_WEIGHTS) == weights_before
