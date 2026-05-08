"""SE(2) rigid-transform composition utilities for the port-graph decoder.

A pose is ``(x, y, theta)`` with theta in **radians** (counter-clockwise positive).
All functions are pure; no NumPy or external state.

These three primitives are sufficient to express port-graph FK propagation:

- :func:`pose_compose` applies a child pose expressed in a parent's frame.
  Used to compute a port's world pose from its slot pose plus the port's
  piece-local offset.
- :func:`pose_inverse` returns the inverse rigid transform.
  Used to back out a slot's pose from a known port-world pose.
- :func:`pose_diff` returns the residual between two poses.
  Used to compute closure error when a BFS revisits a slot via a different
  edge (cycle closure check).
"""

from __future__ import annotations

import math
from typing import Tuple

Pose = Tuple[float, float, float]
"""(x, y, theta_rad) tuple."""


def pose_compose(parent: Pose, child_in_parent: Pose) -> Pose:
    """Compose a child pose expressed in the parent's local frame.

    Returns the world pose corresponding to ``child_in_parent`` when the
    parent is at ``parent``. Standard 2D rigid-body composition::

        x_world = x_p + dx*cos(theta_p) - dy*sin(theta_p)
        y_world = y_p + dx*sin(theta_p) + dy*cos(theta_p)
        theta_world = theta_p + dtheta
    """
    x_p, y_p, theta_p = parent
    dx, dy, dtheta = child_in_parent
    cos_t = math.cos(theta_p)
    sin_t = math.sin(theta_p)
    return (
        x_p + dx * cos_t - dy * sin_t,
        y_p + dx * sin_t + dy * cos_t,
        theta_p + dtheta,
    )


def pose_inverse(p: Pose) -> Pose:
    """Return the inverse rigid transform of ``p``.

    Satisfies ``pose_compose(pose_inverse(p), p) == identity`` (within fp tol).
    Geometrically: if ``p`` takes the origin to a pose, ``pose_inverse(p)``
    takes that pose back to the origin.
    """
    x, y, theta = p
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return (
        -(x * cos_t + y * sin_t),
        -(-x * sin_t + y * cos_t),
        -theta,
    )


def pose_diff(a: Pose, b: Pose) -> Pose:
    """Return ``a`` expressed in ``b``'s frame — the residual transform.

    For closure checks: if ``a`` and ``b`` should coincide, the magnitude of
    ``pose_diff(a, b)``'s components is the closure error in (x, y, theta).

    Equivalent to ``pose_compose(pose_inverse(b), a)``.
    """
    return pose_compose(pose_inverse(b), a)


# Identity pose constant for tests and defaults.
IDENTITY: Pose = (0.0, 0.0, 0.0)
