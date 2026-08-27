"""Comprehensive physical evaluation of a track layout under a given train consist.

Produces a PhysicalEvaluation across five physical domains (geometry,
stability, kinematics, dynamics, energy) in a single O(n) pass per chromosome.
Pure function; no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..catalog import TrackCatalog
from ..geometry import Layout
from .physics import DEFAULT_TRAIN_CONFIG, TrainConfig, derailment_caps
from .scoring import SpeedProfile, compute_speed_profile


@dataclass(frozen=True)
class PhysicalEvaluation:
    """Full physical evaluation of a layout under a given train consist.

    Conventions:
        - Lengths in meters (catalog studs converted via stud_mm/1000).
        - Angles in radians.
        - Speeds in m/s, accelerations in m/s^2.
        - Energies in joules.
        - Per-segment arrays have length == n_pieces of the layout main path.
    """

    # ---- Geometry ----
    coupler_phi_per_segment: NDArray[np.float64]
    coupler_phi_per_switch: dict[int, float]
    max_coupler_phi: float

    # ---- Stability ----
    v_slide_per_segment: NDArray[np.float64]
    v_tip_per_segment: NDArray[np.float64]
    v_nadal_per_segment: NDArray[np.float64]
    v_eff_per_segment: NDArray[np.float64]
    binding_cap_per_segment: NDArray[np.str_]

    # ---- Kinematics ----
    speed_profile: SpeedProfile
    safety_factor_min: float
    safety_factor_mean: float

    # ---- Dynamics ----
    a_lat_per_segment: NDArray[np.float64]
    a_long_per_segment: NDArray[np.float64]
    grip_utilization_per_segment: NDArray[np.float64]
    coupler_force_lat_per_segment: NDArray[np.float64]

    # ---- Energy ----
    motor_work_per_lap: float
    rolling_dissipation_per_lap: float
    ke_roundtrip_per_lap: float

    # ---- Provenance ----
    train_config: TrainConfig
    safety_margin: float
    catalog_signature: str


def _compute_geometry(
    radii_m: NDArray[np.float64],
    coupler_offset: float,
) -> NDArray[np.float64]:
    """Per-segment coupler hinge angle: phi(R) = L/(2R). 0 on straights (R=inf)."""
    safe_R = np.where(np.isfinite(radii_m) & (radii_m > 0), radii_m, np.inf)
    return coupler_offset / (2.0 * safe_R)


def _compute_stability(
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.str_],
]:
    """Per-segment v_slide, v_tip, v_nadal, v_eff, binding-cap label.

    Reuses physics.derailment_caps for the three derailment formulas and adds
    the motor cap so the binding cap can be labelled per segment.
    """
    v_slide, v_tip, v_nadal = derailment_caps(train_config, radii_m)
    v_motor = np.full_like(radii_m, train_config.v_motor_max)

    caps = np.stack([v_slide, v_tip, v_nadal, v_motor], axis=0)  # shape (4, n)
    v_eff = np.min(caps, axis=0)
    binding_idx = np.argmin(caps, axis=0)
    labels = np.array(["slide", "tip", "nadal", "motor"])
    binding_cap = labels[binding_idx]

    return v_slide, v_tip, v_nadal, v_eff, binding_cap


def _compute_dynamics(
    speeds: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    arc_lengths_m: NDArray[np.float64],
    coupler_phi: NDArray[np.float64],
    train_config: TrainConfig,
    is_closed: bool,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Per-segment a_lat, a_long, grip_utilization, lateral coupler force.

    a_lat[i]   = v[i]^2 / R[i]                 (0 on straights, R=inf)
    a_long[i]  = (v[next]^2 - v[i]^2) / (2 * arc_length[i])
                 (closed: next wraps to 0; open: last a_long = 0)
    grip_util  = sqrt((a_lat/(mu*g))^2 + (a_long/cap)^2)   cap = max_accel or brake_decel
    F_coup_lat = m_trailing * a_long * sin(phi)
    """
    n = len(speeds)
    safe_R = np.where(np.isfinite(radii_m) & (radii_m > 0), radii_m, np.inf)
    a_lat = speeds ** 2 / safe_R
    a_lat = np.where(np.isfinite(a_lat), a_lat, 0.0)

    # a_long via finite differences with wrap-around for closed loops
    if n == 0:
        v_next = speeds
    elif is_closed:
        v_next = np.roll(speeds, -1)
    else:
        # Open path: last segment's a_long is 0 (no successor)
        v_next = np.concatenate([speeds[1:], [speeds[-1]]]) if n > 1 else speeds.copy()

    safe_arc = np.where(arc_lengths_m > 0, arc_lengths_m, 1.0)
    a_long = np.where(arc_lengths_m > 0,
                      (v_next ** 2 - speeds ** 2) / (2.0 * safe_arc),
                      0.0)

    a_lat_max = train_config.mu_design * train_config.g
    cap = np.where(a_long >= 0, train_config.max_accel, train_config.brake_decel)
    safe_cap = np.where(cap > 0, cap, 1.0)
    grip_util = np.sqrt(
        (a_lat / a_lat_max) ** 2 + (a_long / safe_cap) ** 2
    )
    grip_util = np.clip(grip_util, 0.0, 1.0 + 1e-9)

    F_coupler_lat = train_config.mass_trailing * a_long * np.sin(coupler_phi)

    return a_lat, a_long, grip_util, F_coupler_lat


def _compute_energy(
    speeds: NDArray[np.float64],
    arc_lengths_m: NDArray[np.float64],
    a_long: NDArray[np.float64],
    train_config: TrainConfig,
    is_closed: bool,
) -> tuple[float, float, float]:
    """Energy domain: motor work, rolling dissipation, KE round-trips per lap.

    motor_work       = sum of positive m_total * a_long * arc_length
    rolling_diss     = mu_roll * m_total * g * total_distance
    ke_roundtrip     = sum of (v_high^2 - v_low^2) * 0.5 * m_total at brake-respin pairs
                       (local speed minima sandwiched by higher values).
    """
    m = train_config.mass_total
    g = train_config.g
    mu_roll = train_config.mu_roll

    # Motor work: only positive longitudinal accel contributes
    pos_force = np.maximum(0.0, m * a_long)
    motor_work = float(np.sum(pos_force * arc_lengths_m))

    total_distance = float(np.sum(arc_lengths_m))
    rolling_diss = mu_roll * m * g * total_distance

    # KE roundtrip: local speed minima (v[i] < v[i-1] AND v[i] < v[i+1])
    n = len(speeds)
    ke_roundtrip = 0.0
    if n >= 3:
        if is_closed:
            v_prev = np.roll(speeds, 1)
            v_next = np.roll(speeds, -1)
        else:
            v_prev = np.concatenate([[speeds[0]], speeds[:-1]])
            v_next = np.concatenate([speeds[1:], [speeds[-1]]])
        is_local_min = (speeds < v_prev) & (speeds < v_next)
        v_high = np.maximum(v_prev, v_next)
        roundtrips = np.where(is_local_min,
                              0.5 * m * (v_high ** 2 - speeds ** 2),
                              0.0)
        ke_roundtrip = float(np.sum(roundtrips))

    return motor_work, rolling_diss, ke_roundtrip


def evaluate_layout(
    layout: Layout,
    catalog: TrackCatalog,
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    safety_margin: float = 0.95,
) -> PhysicalEvaluation:
    """Comprehensive physical evaluation. Pure function, no side effects."""
    n = layout.n_pieces

    if n == 0:
        return _empty_evaluation(train_config, safety_margin)

    stud_to_m = catalog.stud_mm / 1000.0
    radii_m = catalog.get_radii(layout.indices) / 1000.0  # mm -> m

    # ---- Geometry ----
    # Known gap: coupler_phi_per_switch is left empty until switch-aware
    # diverging-route radius lookup is implemented. Per-segment phi already
    # covers R40 curves, so max_coupler_phi is correct on switchless layouts.
    coupler_phi_per_segment = _compute_geometry(radii_m, train_config.coupler_offset)
    coupler_phi_per_switch: dict[int, float] = {}
    max_phi = float(np.max(coupler_phi_per_segment))

    # ---- Stability ----
    (v_slide_per_segment,
     v_tip_per_segment,
     v_nadal_per_segment,
     v_eff_per_segment,
     binding_cap_per_segment) = _compute_stability(radii_m, train_config)

    # ---- Kinematics ----
    speed_profile = compute_speed_profile(
        layout, catalog, train_config, safety_margin=safety_margin,
    )
    # safety_factor[i] = operating_speed[i] / v_derail_cap[i]. The relevant
    # cap is the derailment cap (slide/tip/nadal) — motor-bound segments are
    # not derailment-relevant, so their ratio is recorded against v_eff and
    # the metrics are computed over derail-bound segments only. On segments
    # with no positive cap (degenerate), ratio is 0.
    v_derail = np.minimum.reduce(
        [v_slide_per_segment, v_tip_per_segment, v_nadal_per_segment]
    )
    derail_mask = np.isfinite(v_derail) & (v_derail > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            derail_mask,
            speed_profile.speeds / np.where(derail_mask, v_derail, 1.0),
            0.0,
        )
    arc_lengths_m = catalog.get_arc_lengths(layout.indices) * stud_to_m
    if derail_mask.any():
        safety_factor_min = float(np.min(ratio[derail_mask]))
        weights = arc_lengths_m[derail_mask]
        total = float(np.sum(weights))
        if total > 0:
            safety_factor_mean = float(np.sum(ratio[derail_mask] * weights) / total)
        else:
            safety_factor_mean = safety_factor_min
    else:
        # No derail-bound segments (e.g., all-straight layout): ratio is undefined.
        # +inf encodes "no derailment risk anywhere" rather than "right at margin".
        safety_factor_min = float("inf")
        safety_factor_mean = float("inf")

    # ---- Dynamics ----
    # Tolerances match OptimizationConfig closure defaults (4.0 studs, 5.0 deg)
    # so a feasible loop is never treated as open track.
    is_closed = layout.is_closed(pos_tol=4.0, angle_tol=5.0)
    (a_lat_per_segment,
     a_long_per_segment,
     grip_utilization_per_segment,
     coupler_force_lat_per_segment) = _compute_dynamics(
        speed_profile.speeds, radii_m, arc_lengths_m,
        coupler_phi_per_segment, train_config, is_closed,
    )

    # ---- Energy ----
    motor_work_per_lap, rolling_dissipation_per_lap, ke_roundtrip_per_lap = (
        _compute_energy(
            speed_profile.speeds, arc_lengths_m, a_long_per_segment,
            train_config, is_closed,
        )
    )

    return PhysicalEvaluation(
        coupler_phi_per_segment=coupler_phi_per_segment,
        coupler_phi_per_switch=coupler_phi_per_switch,
        max_coupler_phi=max_phi,
        v_slide_per_segment=v_slide_per_segment,
        v_tip_per_segment=v_tip_per_segment,
        v_nadal_per_segment=v_nadal_per_segment,
        v_eff_per_segment=v_eff_per_segment,
        binding_cap_per_segment=binding_cap_per_segment,
        speed_profile=speed_profile,
        safety_factor_min=safety_factor_min,
        safety_factor_mean=safety_factor_mean,
        a_lat_per_segment=a_lat_per_segment,
        a_long_per_segment=a_long_per_segment,
        grip_utilization_per_segment=grip_utilization_per_segment,
        coupler_force_lat_per_segment=coupler_force_lat_per_segment,
        motor_work_per_lap=motor_work_per_lap,
        rolling_dissipation_per_lap=rolling_dissipation_per_lap,
        ke_roundtrip_per_lap=ke_roundtrip_per_lap,
        train_config=train_config,
        safety_margin=safety_margin,
        catalog_signature=_catalog_signature(catalog),
    )


def _empty_evaluation(train_config: TrainConfig, safety_margin: float) -> PhysicalEvaluation:
    """Empty evaluation for n==0 layouts."""
    empty = np.zeros(0, dtype=np.float64)
    return PhysicalEvaluation(
        coupler_phi_per_segment=empty,
        coupler_phi_per_switch={},
        max_coupler_phi=0.0,
        v_slide_per_segment=empty,
        v_tip_per_segment=empty,
        v_nadal_per_segment=empty,
        v_eff_per_segment=empty,
        binding_cap_per_segment=np.array([], dtype="<U5"),
        speed_profile=SpeedProfile(
            speeds=empty, avg_speed=0.0, lap_time=0.0,
            total_distance=0.0, max_speed=0.0, min_speed=0.0,
        ),
        safety_factor_min=1.0,
        safety_factor_mean=1.0,
        a_lat_per_segment=empty,
        a_long_per_segment=empty,
        grip_utilization_per_segment=empty,
        coupler_force_lat_per_segment=empty,
        motor_work_per_lap=0.0,
        rolling_dissipation_per_lap=0.0,
        ke_roundtrip_per_lap=0.0,
        train_config=train_config,
        safety_margin=safety_margin,
        catalog_signature="",
    )


def _catalog_signature(catalog: TrackCatalog) -> str:
    """Short, stable identifier for the catalog (n_pieces)."""
    return f"npieces={catalog.n_pieces}"
