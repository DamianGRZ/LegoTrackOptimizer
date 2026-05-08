"""Tests for Phase 3 -- auto-centering decoder + ENCODING_VERSION (PLAN §10.2 3.1-3.4).

Phase 3 reframes the chromosome anchor: instead of an absolute world position
of slot 0, anchor genes (off_x, off_y) become a small +/-5% offset relative to
the boundary-centered layout. The decoder runs FK as before but adds an
``_auto_center_layout`` step that shifts the bbox center to the boundary
center plus the chromosome's anchor offset. Side effect: pre-Phase-3
e-archive JSONs become silently ambiguous, hence ENCODING_VERSION + a hard
``EncodingVersionMismatch`` failure on ``from_json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import DecoderConfig, decode_chromosome
from src_v2.encoding import (
    ENCODING_VERSION,
    compute_port_pair_dimensions,
    set_anchor,
)
from src_v2.epsilon_archive import (
    EncodingVersionMismatch,
    EpsilonArchive,
)
from tests.fixtures.hand_crafted_chromosomes import perfect_oval_16_R40


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(CONFIG_PATH)
    dims = compute_port_pair_dimensions(config.boundary, catalog, config.inventory)
    decoder_config = DecoderConfig(
        boundary_min_x=config.boundary.min_x,
        boundary_max_x=config.boundary.max_x,
        boundary_min_y=config.boundary.min_y,
        boundary_max_y=config.boundary.max_y,
    )
    return catalog, config, dims, decoder_config


def _bbox_center(graph) -> tuple[float, float]:
    xs = [p[0] for p in graph.slot_poses.values()]
    ys = [p[1] for p in graph.slot_poses.values()]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


# ---------------------------------------------------------------- 3.1
def test_3_1_decoded_layout_centers_on_boundary_center(setup) -> None:
    """A decoded 16-R40 oval with anchor=(0,0,0) lands its bbox center within
    1 stud of the boundary center."""
    catalog, config, dims, dec_cfg = setup
    x = perfect_oval_16_R40(catalog, dims)
    set_anchor(x, dims, 0, 0, 0)

    graph = decode_chromosome(x, dims, catalog, dec_cfg)

    bcx = (config.boundary.min_x + config.boundary.max_x) / 2.0
    bcy = (config.boundary.min_y + config.boundary.max_y) / 2.0
    cx, cy = _bbox_center(graph)
    assert abs(cx - bcx) < 1.0, f"bbox.cx={cx} not within 1 stud of {bcx}"
    assert abs(cy - bcy) < 1.0, f"bbox.cy={cy} not within 1 stud of {bcy}"


# ---------------------------------------------------------------- 3.2
def test_3_2_anchor_offset_applied_additively(setup) -> None:
    """Setting anchor offset = (+5, -3) shifts the bbox center by exactly that."""
    catalog, config, dims, dec_cfg = setup
    bcx = (config.boundary.min_x + config.boundary.max_x) / 2.0
    bcy = (config.boundary.min_y + config.boundary.max_y) / 2.0

    x_zero = perfect_oval_16_R40(catalog, dims)
    set_anchor(x_zero, dims, 0, 0, 0)
    g_zero = decode_chromosome(x_zero, dims, catalog, dec_cfg)
    cx0, cy0 = _bbox_center(g_zero)

    x_off = perfect_oval_16_R40(catalog, dims)
    set_anchor(x_off, dims, 5, -3, 0)
    g_off = decode_chromosome(x_off, dims, catalog, dec_cfg)
    cx1, cy1 = _bbox_center(g_off)

    assert abs((cx1 - cx0) - 5.0) < 1e-6
    assert abs((cy1 - cy0) - (-3.0)) < 1e-6
    # Sanity: zero-offset version is on the boundary center
    assert abs(cx0 - bcx) < 1.0
    assert abs(cy0 - bcy) < 1.0


# ---------------------------------------------------------------- 3.3
def test_3_3_from_json_rejects_pre_phase3_archive(tmp_path) -> None:
    """An archive JSON without ``encoding_version`` raises EncodingVersionMismatch."""
    legacy_path = tmp_path / "legacy_archive.json"
    legacy_path.write_text(json.dumps({
        "epsilon": [0.005, 0.01],
        "max_size": 10,
        "F": [[-0.5, -0.97]],
        "X": [[0] * 50],
    }), encoding="utf-8")
    with pytest.raises(EncodingVersionMismatch):
        EpsilonArchive.from_json(legacy_path)


# ---------------------------------------------------------------- 3.4
def test_3_4_from_json_round_trip_identity(tmp_path) -> None:
    """Round-trip: write archive with current ENCODING_VERSION, read back identically."""
    archive = EpsilonArchive(epsilon=(0.005, 0.01), max_size=20)
    x_row = np.array([1, 2, 3, -1, -1], dtype=np.int16)
    archive.admit(x_row, np.array([-0.5, -0.97]))

    out_path = tmp_path / "archive.json"
    archive.to_json(out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["encoding_version"] == ENCODING_VERSION

    restored = EpsilonArchive.from_json(out_path)
    assert restored.F.tolist() == archive.F.tolist()
    assert [r.tolist() for r in restored.X] == [r.tolist() for r in archive.X]
    assert restored._eps.tolist() == archive._eps.tolist()
    assert restored._max_size == archive._max_size
