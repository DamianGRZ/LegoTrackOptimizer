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

# Hypervolume reference just outside the unit box, so every point of a
# normalized front contributes volume.
HV_REF_POINT = (1.1, 1.1)


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
