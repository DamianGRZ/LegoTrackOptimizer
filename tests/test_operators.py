# tests/test_operators.py
"""Tests for partitioned chromosome genetic operators."""

import numpy as np
import pytest

from src.encoding import (
    GENES_PER_JUNCTION,
    PartitionedDimensions,
    create_empty_chromosome,
    set_junction,
)
from src.operators import _change_handedness
from src.templates import TEMPLATES


@pytest.fixture
def dims() -> PartitionedDimensions:
    """Minimal partitioned dimensions with 2 junction slots."""
    return PartitionedDimensions(
        n_main=10,
        max_junctions=2,
        total_straights=10,
        boundary_min_x=-100.0,
        boundary_max_x=100.0,
        boundary_min_y=-100.0,
        boundary_max_y=100.0,
    )


class TestChangeHandedness:
    def test_stays_within_template_bounds(self, dims):
        """_change_handedness must only produce values in [0, len(TEMPLATES) - 1]."""
        np.random.seed(42)
        x = create_empty_chromosome(dims)
        set_junction(x, dims, 0, active=1, position=0, handedness=0, n_straights=0)
        set_junction(x, dims, 1, active=1, position=0, handedness=0, n_straights=0)

        seen = set()
        for _ in range(1000):
            _change_handedness(x, dims)
            for slot in range(dims.max_junctions):
                base = dims.junc_start + slot * GENES_PER_JUNCTION
                seen.add(int(x[base + 2]))

        max_valid = len(TEMPLATES) - 1
        out_of_bounds = seen - set(range(len(TEMPLATES)))
        assert not out_of_bounds, (
            f"handedness values {out_of_bounds} exceed declared xu={max_valid}"
        )
