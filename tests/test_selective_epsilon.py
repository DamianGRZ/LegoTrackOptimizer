"""Selective epsilon: closure + boundary are epsilon-relaxable (SOFT); collisions
and per-type inventory are HARD (never relaxed). A closed self-crossing layout
must therefore stay infeasible even at full epsilon, while a near-closed clean
siding is still relaxed. Tests exercise the real LegoAdaptiveEpsilon._adapt_
constraint_handling and pymoo's own calc_cv (no reimplementation of CV)."""
import types

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual, calc_cv

from src.algorithm.runner import LegoAdaptiveEpsilon, SOFT_CONSTRAINT_COUNT


def _adapted_config(perc, n_ieq=12, n_soft=SOFT_CONSTRAINT_COUNT, max_cv=30.0,
                    hold_until=0.2, perc_eps_until=0.9):
    """The config LegoAdaptiveEpsilon writes at schedule position ``perc``."""
    h = LegoAdaptiveEpsilon(NSGA2(pop_size=10), n_ieq_constr=n_ieq,
                            hold_until=hold_until, perc_eps_until=perc_eps_until)
    h.max_cv = max_cv
    h.termination = types.SimpleNamespace(perc=perc)
    cfg = {}
    h._adapt_constraint_handling(cfg)
    return cfg


def _feas(G, cfg):
    full = {**Individual.default_config(), **cfg}
    cv = float(calc_cv(G=np.asarray(G, dtype=float), config=full))
    return bool(cv <= cfg["cv_eps"])


def _G(closure=(-1.0, -1.0, -1.0), boundary=0.0, collisions=0.0, inv=0.0):
    # n_ieq=12 layout: [closure_x, closure_y, closure_theta, boundary,
    #                   collisions, inv_0..inv_6]
    return list(closure) + [boundary, collisions] + [inv] * 7


def test_closed_self_crosser_infeasible_even_at_full_epsilon():
    cfg = _adapted_config(perc=0.0)  # hold phase, alpha=1, full epsilon
    G = _G(closure=(-1, -1, -1), boundary=0.02, collisions=0.4)
    assert not _feas(G, cfg)


def test_near_closed_clean_siding_stays_epsilon_feasible():
    cfg = _adapted_config(perc=0.0)
    G = _G(closure=(4.0, 4.0, 0.5), boundary=0.01, collisions=0.0)
    assert _feas(G, cfg)


def test_standalone_feasible_circle_is_feasible():
    cfg = _adapted_config(perc=0.0)
    G = _G(closure=(-1, -1, -1), boundary=-0.1, collisions=0.0)
    assert _feas(G, cfg)


def test_over_inventory_infeasible_even_at_full_epsilon():
    cfg = _adapted_config(perc=0.0)
    G = _G(closure=(-1, -1, -1), collisions=0.0, inv=0.3)
    assert not _feas(G, cfg)


def test_strict_phase_relaxes_nothing():
    cfg = _adapted_config(perc=0.95)  # past perc_eps_until -> alpha=0
    assert cfg["cv_eps"] == 0.0
    G = _G(closure=(4.0, 0.0, 0.0), collisions=0.0)
    assert not _feas(G, cfg)
