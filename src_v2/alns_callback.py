"""Adaptive Large Neighborhood Search callback for operator weight tuning.

References:
    - Pisinger & Ropke (2007) "A general heuristic for vehicle routing problems".
    - Li, Fialho, Kwong & Zhang (2014) FRRMAB.

Per generation, reads ``mutation._last_op_indices`` (set during
``PortPairMutation._do``) and the offspring constraint-violation values
from the algorithm. Credits each operator with a per-individual reward
(feasibility = full credit, infeasibility = inverse-CV partial credit).

Every K generations, exponentially blends the accumulated rewards into
``mutation.OP_WEIGHTS`` (with a floor to keep operators from dying out)
and rebuilds the dispatch CDF.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

import numpy as np
from pymoo.core.callback import Callback


class ALNSCallback(Callback):
    """Adaptive operator-weight tuning for ``PortPairMutation``.

    Designed to be attached to a single mutation instance. Mutating
    ``mutation.OP_WEIGHTS`` shadows the class-level dict on the instance, so
    multi-config runs don't bleed between ALNS instances.
    """

    LAMBDA: float = 0.4
    EXPLORATION_FLOOR: float = 0.02
    K: int = 40

    def __init__(
        self,
        lambd: float = LAMBDA,
        exploration_floor: float = EXPLORATION_FLOOR,
        reweight_every: int = K,
    ) -> None:
        super().__init__()
        self._lambd = float(lambd)
        self._floor = float(exploration_floor)
        self._k = int(reweight_every)
        self._mutation = None
        self._reward_sum: Dict[str, float] = defaultdict(float)
        self._call_count: Dict[str, int] = defaultdict(int)
        self._gen = 0
        self._logger = logging.getLogger(__name__)

    def attach_to(self, mutation) -> None:
        """Bind to a ``PortPairMutation`` instance. Must be called before
        ``minimize()`` so the callback can read ``_last_op_indices`` and
        write ``OP_WEIGHTS`` on the right object."""
        self._mutation = mutation

    def notify(self, algorithm) -> None:
        if self._mutation is None:
            return
        last_ops = self._mutation._last_op_indices or ()
        offspring = getattr(algorithm, "off", None)
        if offspring is None or not last_ops:
            return
        cv = offspring.get("CV")
        if cv is None:
            return
        cv_flat = np.asarray(cv).flatten()
        op_names = self._mutation._op_names

        for op_idx, cv_val in zip(last_ops, cv_flat):
            if op_idx < 0 or op_idx >= len(op_names):
                continue
            name = op_names[op_idx]
            self._call_count[name] += 1
            self._reward_sum[name] += 1.0 if cv_val <= 0 else 1.0 / (1.0 + cv_val)

        self._gen += 1
        if self._gen % self._k == 0:
            self._reweight()

    def _reweight(self) -> None:
        weights_old = dict(self._mutation.OP_WEIGHTS)
        rewards = {
            name: self._reward_sum[name] / max(1, self._call_count[name])
            for name in weights_old
        }
        total = sum(rewards.values()) or 1.0
        norm_rewards = {n: r / total for n, r in rewards.items()}

        new_unfloored = {
            n: (1 - self._lambd) * weights_old[n] + self._lambd * norm_rewards[n]
            for n in weights_old
        }
        floored = {n: max(self._floor, v) for n, v in new_unfloored.items()}
        # Re-normalise so weights sum to 1 after the floor truncation.
        total_floored = sum(floored.values()) or 1.0
        weights_new = {n: v / total_floored for n, v in floored.items()}

        self._mutation.OP_WEIGHTS = weights_new
        self._mutation._build_cdf()

        top = sorted(weights_new.items(), key=lambda kv: kv[1], reverse=True)[:3]
        self._logger.info(
            f"ALNS reweight @ gen {self._gen}: "
            + ", ".join(f"{n}={v:.3f}" for n, v in top),
        )

        self._reward_sum.clear()
        self._call_count.clear()
