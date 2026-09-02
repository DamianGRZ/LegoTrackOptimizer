"""Objective-space normalization for post-processing.

The GA itself runs on raw ``F``: dominance compares each objective
separately and NSGA-II's crowding re-normalizes within a front, so objective
scale cannot bias selection. Scale bites in reporting instead — hypervolume,
Pareto axes and the compromise pick all need the objectives on one common
0-1 range.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pymoo.decomposition.asf import ASF
from pymoo.util.normalization import ZeroToOneNormalization


def hv_ref_point(n_obj: int) -> NDArray[np.float64]:
    """Hypervolume reference just outside the unit box, so every point of a
    normalized front contributes volume."""
    return np.full(int(n_obj), 1.1)


def ideal_nadir(F: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Approximate the ideal and nadir points as the per-objective min/max."""
    F = np.asarray(F, dtype=float)
    return F.min(axis=0), F.max(axis=0)


def has_extent(ideal: NDArray[np.float64], nadir: NDArray[np.float64]) -> bool:
    """True when every objective spans a non-zero, finite range.

    A zero-range axis has no 0-1 mapping. ``normalize`` keeps going by
    substituting a unit span, which is fine for a plot but would turn a
    hypervolume into a number about nothing — check this first wherever the
    result has to mean something.
    """
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)
    return bool(np.all(np.isfinite(ideal) & np.isfinite(nadir) & (nadir > ideal)))


def normalize(F: NDArray[np.float64], ideal: NDArray[np.float64],
              nadir: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map ``F`` onto the unit box spanned by ``ideal``..``nadir``.

    0 is the best value of an objective and 1 its worst, whichever sign the
    raw objective carries. Points outside that box (dominated ones beyond
    the nadir) stay outside 0-1 by design; clipping would place them on a
    front they do not reach. A zero-range objective is given a unit span so
    the mapping stays defined — see ``has_extent``.
    """
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)
    span_nadir = np.where(nadir > ideal, nadir, ideal + 1.0)
    return ZeroToOneNormalization(ideal, span_nadir).forward(np.asarray(F, dtype=float))


def compromise_index(nF: NDArray[np.float64],
                     weights: NDArray[np.float64] | None = None) -> int:
    """Index of the ASF compromise solution on a normalized front.

    ASF divides by the weights, so the reciprocal is what gets passed in.
    Equal weights ask for the most balanced trade-off available.
    """
    nF = np.asarray(nF, dtype=float)
    if weights is None:
        weights = np.full(nF.shape[1], 1.0 / nF.shape[1])
    return int(ASF().do(nF, 1.0 / np.asarray(weights, dtype=float)).argmin())


def balance_ranking(F: NDArray[np.float64]) -> NDArray[np.intp]:
    """Rows of raw ``F``, most balanced across the objectives first.

    The normalization spans ``F`` itself, so the order is a statement about this
    set only and cannot be compared with one made on another set. Rows carrying
    the +inf sentinel are dropped rather than collapsing the scale, so the
    result may be shorter than ``F``.
    """
    F = np.asarray(F, dtype=float)
    finite = np.flatnonzero(np.isfinite(F).all(axis=1))
    if len(finite) == 0:
        return np.empty(0, dtype=np.intp)
    ideal, nadir = ideal_nadir(F[finite])
    nF = normalize(F[finite], ideal, nadir)
    # ASF().do keeps a (1, 1) shape for a single row but returns (n,) beyond it,
    # so the values are flattened before ranking.
    asf = np.asarray(ASF().do(nF, np.full(nF.shape[1], nF.shape[1], dtype=float))).ravel()
    return finite[np.argsort(asf)]


def first_objective_ranking(F: NDArray[np.float64]) -> NDArray[np.intp]:
    """Rows of raw ``F`` ordered by the first objective alone, best (lowest, as
    stored) first — the champion is simply the strongest on the run's primary
    term. Rows carrying the +inf sentinel are dropped as in ``balance_ranking``.
    """
    F = np.asarray(F, dtype=float)
    finite = np.flatnonzero(np.isfinite(F).all(axis=1))
    return finite[np.argsort(F[finite, 0], kind="stable")]


# config.champion_selection literal -> ranking function. Both rankings order a
# set champion-first and share the +inf-skipping contract, so every consumer
# can hold either without knowing which.
CHAMPION_RANKINGS = {
    "first_objective": first_objective_ranking,
    "balanced": balance_ranking,
}


def champion_ranking(rule: str):
    """Ranking function behind ``config.champion_selection``."""
    return CHAMPION_RANKINGS[rule]
