"""Tests for ``problem._residual_uses_branch_tolerance`` — per-cycle tolerance."""

from __future__ import annotations

from src_v2.problem import _residual_uses_branch_tolerance
from src_v2.types import CycleResidual


def _residual(slot_a: int, slot_b: int) -> CycleResidual:
    return CycleResidual(slot_a=slot_a, slot_b=slot_b, dx=0.0, dy=0.0, dtheta=0.0)


def test_no_labels_means_main_tolerance():
    """Empty branch_labels → conservative: residual treated as main."""
    r = _residual(0, 1)
    assert _residual_uses_branch_tolerance(r, {}) is False


def test_pure_main_loop_uses_main_tolerance():
    """All slots labeled 'main' on cycle 0 → residual is main."""
    labels = {(i, "main"): 0 for i in range(8)}
    r = _residual(0, 1)
    assert _residual_uses_branch_tolerance(r, labels) is False


def test_diverging_cycle_uses_branch_tolerance():
    """Residual on a cycle that contains a switch's 'diverging' route."""
    labels = {
        # Main cycle: through routes on switches + main slots
        (1, "through"): 0,
        (6, "through"): 0,
        (0, "main"): 0,
        (2, "main"): 0,
        # Branch cycle: diverging routes on switches + branch slots
        (1, "diverging"): 1,
        (6, "diverging"): 1,
        (4, "main"): 1,
        (5, "main"): 1,
    }
    main_residual = _residual(0, 2)         # closes main cycle
    branch_residual = _residual(4, 5)        # closes branch cycle

    assert _residual_uses_branch_tolerance(main_residual, labels) is False
    assert _residual_uses_branch_tolerance(branch_residual, labels) is True


def test_residual_through_switches_uses_main_tolerance():
    """Residual whose endpoints are both switches — common cycle is BOTH the
    main and branch cycle. Conservative: uses main tolerance unless ONLY
    the branch cycle is shared."""
    labels = {
        (1, "through"): 0, (6, "through"): 0, (0, "main"): 0,
        (1, "diverging"): 1, (6, "diverging"): 1,
    }
    r = _residual(1, 6)
    # Switches 1 and 6 are both on cycles 0 (main) and 1 (branch).
    # Common = {0, 1}. Branch cycles = {1}. Intersection nonempty → branch tolerance.
    assert _residual_uses_branch_tolerance(r, labels) is True


def test_crossing_routes_use_main_tolerance():
    """Figure-8 via CROSS_90 has horizontal/vertical routes — neither is
    'diverging' so both cycles use main tolerance."""
    labels = {
        (1, "horizontal"): 0,
        (0, "main"): 0, (2, "main"): 0,
        (1, "vertical"): 1,
        (3, "main"): 1, (4, "main"): 1,
    }
    r_h = _residual(0, 2)
    r_v = _residual(3, 4)

    assert _residual_uses_branch_tolerance(r_h, labels) is False
    assert _residual_uses_branch_tolerance(r_v, labels) is False


def test_residual_with_unmapped_endpoint_uses_main():
    """Endpoint not in branch_labels → no common cycle → main tolerance."""
    labels = {(0, "main"): 0, (1, "main"): 0}
    r = _residual(99, 100)  # neither slot in labels
    assert _residual_uses_branch_tolerance(r, labels) is False
