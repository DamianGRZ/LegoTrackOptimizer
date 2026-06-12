"""Run-cumulative feasible front exposed by ConvergenceMonitorCallback.

Selection discards mid-run trade-off solutions; the monitor's archive is the
only record of every feasible F ever seen, so it must be publicly readable
for the Pareto plot and persisted artifacts.
"""

from types import SimpleNamespace

import numpy as np
from pymoo.core.population import Population

from src.algorithm.monitoring import ConvergenceMonitorCallback


def _algo(F_rows, feas):
    n = len(F_rows)
    pop = Population.new("X", np.zeros((n, 4)))
    pop.set("F", np.array(F_rows, dtype=float))
    pop.set("CV", np.array([[0.0 if f else 1.0] for f in feas]))
    pop.set("G", np.full((n, 3), -1.0))
    return SimpleNamespace(pop=pop, n_gen=1,
                           evaluator=SimpleNamespace(n_eval=n))


def test_best_front_survives_population_turnover():
    mon = ConvergenceMonitorCallback()
    mon.notify(_algo([[-0.5, -1.0]], feas=[True]))   # small fast loop
    mon.notify(_algo([[-0.7, -0.9]], feas=[True]))   # bigger, slower gen2
    front = mon.best_front
    assert front.shape == (2, 2), "both mutually non-dominated points kept"


def test_best_front_empty_before_any_feasible():
    mon = ConvergenceMonitorCallback()
    mon.notify(_algo([[-0.5, -1.0]], feas=[False]))
    assert mon.best_front.shape == (0, 2)
