# Phase 1: Catalog V2 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the track catalog from the current section-keyed YAML schema (`straights:`, `curves:`, `r40_switch_components:`, `crossings:` plus a flat `piece_index:` map) to the V2 port-centric schema (a single `pieces:` list where every piece declares `kind`, `manufacturer`, `ports: {A, B, ...}` with `PortDef(dx, dy, dtheta)`, and named `routes: {through: [A, B], diverging: [A, C]}`) while keeping every existing test green. The legacy surface of `TrackCatalog` (`_fk_table`, `get_fk`, `get_radii`, `get_speed_limits`, `_id_to_index`, etc.) keeps working through a backward-compat wrapper, so `problem.py`, `sampling.py`, `repair.py`, and the test suite require no changes.

**Architecture:** New module `src/catalog/specs.py` owns the Pydantic v2 domain models (`PortDef`, `TrackPieceSpec`, `CatalogMeta`, `TrackCatalogSpec`) with `ConfigDict(extra='forbid', frozen=True)`. New module `src/catalog/loader.py` owns a ruamel.yaml **round-trip** loader that attaches file+line to Pydantic errors and enforces schema versioning via `packaging.Version`. The existing `TrackCatalog` class in `src/catalog/catalog.py` is refactored to wrap a `TrackCatalogSpec` and expose the legacy `_fk_table`/`_speed_table`/`_radius_table` numpy arrays via `@cached_property`. The existing YAML file stays on disk during Phase 1; a parallel v2 YAML is added and `TrackCatalog.load()` auto-detects which format is present.

**Tech Stack:** Python 3.11+, Pydantic v2 (already pinned), ruamel.yaml (new dep), `packaging` (stdlib-adjacent), numpy, pytest.

**Companion docs:**
- Strategic roadmap: `docs/superpowers/plans/2026-04-20-modular-v2-adoption-roadmap.md` §Phase 1
- Implementation research: `docs/superpowers/plans/2026-04-20-batch-1-implementation-research.md` §Catalog
- V2 spec: `Modular9PartResearchV2/The catalog package a first-principles design grounded in primary geometric sources.md`

**Non-goals (explicitly deferred):**
- **Correcting LEGO R40 switch geometry.** Current YAML has switch diverge `(dx=31.0, dy=±6.2, dtheta=±22.5°)`. V2's 3-4-5 Pythagorean derivation gives `(32.71, ±12.96, ±22.5°)`. The translated v2 YAML **preserves the current values verbatim** to keep this a non-behavioral migration. Correcting the geometry is a separate task (belongs after Phase 5 decoder lands, because it changes every layout's closure).
- Adding manufacturer pieces beyond current LEGO set (Fx Bricks, BrickTracks, Trixbrix). Deferred — each needs its own port-C verification.
- Migrating the YAML file away from `.yaml` extension or restructuring directory. `data/track_pieces_v2.yaml` sits beside the legacy file; the legacy file is retained through Phase 1.
- Any change to `configs/*.yaml` or pymoo wiring. Catalog is a pure-domain package; downstream consumers are unchanged.

**Risk posture:** Non-behavioral refactor protected by the existing test suite. Every task ends with `/test` green and a commit, so the plan can be paused or reverted between tasks. The single semantic change is the YAML format; the FK numpy tables that drive all downstream math are bit-for-bit identical before and after the migration.

---

## File Structure After Phase 1

```
src/catalog/
├── __init__.py              (MODIFY — add specs exports)
├── catalog.py               (MODIFY — wrap TrackCatalogSpec, cached_property tables)
├── pieces.py                (unchanged — legacy TrackPiece kept for compat)
├── specs.py                 (NEW — Pydantic v2 domain models)
└── loader.py                (NEW — ruamel.yaml + Pydantic + file:line error UX)

data/
├── track_pieces.yaml        (unchanged — legacy format)
└── track_pieces_v2.yaml     (NEW — V2 port-centric format)

tests/
├── test_catalog.py          (unchanged — must still pass verbatim)
├── test_catalog_specs.py    (NEW — Pydantic model validators)
├── test_catalog_loader.py   (NEW — file:line error UX + schema version)
├── test_catalog_parity.py   (NEW — v1 vs v2 load produces identical _fk_table)
└── fixtures/
    ├── catalog_tiny.yaml    (NEW — minimal 2-piece catalog for unit tests)
    └── catalog_bad_*.yaml   (NEW — per-error-type broken catalogs)

requirements.txt             (MODIFY — add ruamel.yaml)

.importlinter                (MODIFY — add catalog layer contract)
```

---

## Task 0: Dependencies + branch

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add `ruamel.yaml` to dependencies**

```
# Data handling
pyyaml>=6.0            # legacy; retire after Phase 1
pydantic>=2.0.0
ruamel.yaml>=0.18.0    # NEW: round-trip YAML with line numbers
```

Edit `requirements.txt` adding the `ruamel.yaml>=0.18.0` line after the `pydantic>=2.0.0` line.

- [ ] **Step 2: Install the new dependency**

Run: `pip install ruamel.yaml>=0.18.0`
Expected: `Successfully installed ruamel.yaml-0.18.x ruamel.yaml.clib-0.2.x`

- [ ] **Step 3: Verify pydantic ≥ 2.0 and packaging are available**

Run: `python -c "import pydantic, packaging, ruamel.yaml; print(pydantic.VERSION, packaging.__version__)"`
Expected: pydantic ≥ 2.0, packaging ≥ 23.0 (stdlib-adjacent; installed with pip).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add ruamel.yaml for catalog v2 line-number-aware loader"
```

---

## Task 1: PortDef Pydantic model

**Files:**
- Create: `src/catalog/specs.py`
- Create: `tests/test_catalog_specs.py`

- [ ] **Step 1: Write failing test for valid PortDef construction and frozen-ness**

Create `tests/test_catalog_specs.py` with:

```python
"""Tests for Pydantic v2 catalog specs (V2 schema)."""

import math
import pytest
from pydantic import ValidationError

from src.catalog.specs import PortDef


class TestPortDef:
    def test_valid_construction(self):
        """PortDef accepts three floats: dx, dy, dtheta."""
        port = PortDef(dx=0.0, dy=0.0, dtheta=0.0)
        assert port.dx == 0.0
        assert port.dy == 0.0
        assert port.dtheta == 0.0

    def test_frozen(self):
        """PortDef is immutable."""
        port = PortDef(dx=1.0, dy=2.0, dtheta=math.pi)
        with pytest.raises(ValidationError):
            port.dx = 99.0

    def test_extra_field_rejected(self):
        """Unknown fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError) as exc:
            PortDef(dx=0.0, dy=0.0, dtheta=0.0, color="red")
        assert "extra_forbidden" in str(exc.value) or "Extra inputs" in str(exc.value)

    def test_missing_field_rejected(self):
        """Missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            PortDef(dx=0.0, dy=0.0)  # missing dtheta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_catalog_specs.py::TestPortDef -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.catalog.specs'`

- [ ] **Step 3: Create `src/catalog/specs.py` with PortDef**

```python
"""Pydantic v2 domain models for the track catalog (V2 schema)."""

from __future__ import annotations

import math
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_FROZEN = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PortDef(BaseModel):
    """SE(2) pose of a port relative to the piece-local origin (port A)."""

    model_config = _FROZEN

    dx: float = Field(description="forward offset in studs")
    dy: float = Field(description="left offset in studs (y = left)")
    dtheta: float = Field(description="heading delta in radians, CCW positive")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_catalog_specs.py::TestPortDef -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/catalog/specs.py tests/test_catalog_specs.py
git commit -m "feat(catalog): add PortDef Pydantic v2 model

V2 port-centric schema: every port is (dx, dy, dtheta) relative to port A.
Frozen + extra='forbid' for immutability and strict validation."
```

---

## Task 2: TrackPieceSpec model with validators

**Files:**
- Modify: `src/catalog/specs.py`
- Modify: `tests/test_catalog_specs.py`

- [ ] **Step 1: Write failing tests for TrackPieceSpec**

Append to `tests/test_catalog_specs.py`:

```python
class TestTrackPieceSpec:
    def _base_straight(self, **overrides):
        from src.catalog.specs import TrackPieceSpec
        payload = dict(
            piece_id="straight_16",
            kind="straight",
            manufacturer="lego",
            part_numbers=("53401",),
            length_studs=16.0,
            ports={
                "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 16.0, "dy": 0.0, "dtheta": 0.0},
            },
            routes={"main": ("A", "B")},
        )
        payload.update(overrides)
        return TrackPieceSpec.model_validate(payload)

    def test_valid_straight_constructs(self):
        spec = self._base_straight()
        assert spec.piece_id == "straight_16"
        assert spec.kind == "straight"
        assert spec.manufacturer == "lego"

    def test_port_a_must_be_at_origin(self):
        with pytest.raises(ValidationError) as exc:
            self._base_straight(ports={
                "A": {"dx": 1.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 16.0, "dy": 0.0, "dtheta": 0.0},
            })
        assert "Port 'A'" in str(exc.value)

    def test_port_a_must_exist(self):
        with pytest.raises(ValidationError) as exc:
            self._base_straight(ports={
                "X": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 16.0, "dy": 0.0, "dtheta": 0.0},
            })
        assert "Port 'A'" in str(exc.value)

    def test_route_references_real_port(self):
        with pytest.raises(ValidationError) as exc:
            self._base_straight(routes={"main": ("A", "Z")})
        assert "route" in str(exc.value).lower() or "undefined" in str(exc.value).lower()

    def test_curve_requires_radius(self):
        from src.catalog.specs import TrackPieceSpec
        with pytest.raises(ValidationError) as exc:
            TrackPieceSpec.model_validate(dict(
                piece_id="curve_missing",
                kind="curve",
                manufacturer="lego",
                ports={
                    "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                    "B": {"dx": 15.307, "dy": 3.045, "dtheta": 0.3927},
                },
                routes={"main": ("A", "B")},
                # radius_studs and sector_angle_rad deliberately missing
            ))
        assert "radius_studs" in str(exc.value) or "curve" in str(exc.value)

    def test_switch_requires_diverging_radius(self):
        from src.catalog.specs import TrackPieceSpec
        with pytest.raises(ValidationError):
            TrackPieceSpec.model_validate(dict(
                piece_id="switch_missing",
                kind="switch",
                manufacturer="lego",
                ports={
                    "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                    "B": {"dx": 32.0, "dy": 0.0, "dtheta": 0.0},
                    "C": {"dx": 31.0, "dy": 6.2, "dtheta": 0.3927},
                },
                routes={"through": ("A", "B"), "diverging": ("A", "C")},
                body_length_studs=32.0,
                # diverging_radius_studs deliberately missing
            ))

    def test_on_angle_lattice_for_lego_r40(self):
        """R40 curve at 22.5° is on the π/32 lattice (k=4)."""
        from src.catalog.specs import TrackPieceSpec
        spec = TrackPieceSpec.model_validate(dict(
            piece_id="r40_left",
            kind="curve",
            manufacturer="lego",
            radius_studs=40.0,
            sector_angle_rad=math.pi / 8,
            hand="left",
            ports={
                "A": {"dx": 0.0, "dy": 0.0, "dtheta": 0.0},
                "B": {"dx": 15.307, "dy": 3.045, "dtheta": math.pi / 8},
            },
            routes={"main": ("A", "B")},
        ))
        assert spec.on_angle_lattice is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_catalog_specs.py::TestTrackPieceSpec -v`
Expected: FAIL with `ImportError: cannot import name 'TrackPieceSpec'`

- [ ] **Step 3: Add TrackPieceSpec to `src/catalog/specs.py`**

Append to `src/catalog/specs.py`:

```python
ATOMIC_ANGLE_RAD = math.pi / 32  # 5.625°; see V2 catalog report §Atomic angle
LATTICE_TOLERANCE = 1e-6         # corrected from V2 spec's 1e-9 per research finding


class TrackPieceSpec(BaseModel):
    """A single track piece: kind, manufacturer, ports, and named routes."""

    model_config = _FROZEN

    piece_id: str = Field(min_length=1)
    kind: Literal["straight", "curve", "switch", "wye", "crossing"]
    manufacturer: Literal["lego", "4dbrix", "fxbricks", "bricktracks", "trixbrix"]
    part_numbers: tuple[str, ...] = ()

    # kind-conditional geometry
    length_studs: float | None = None
    radius_studs: float | None = None
    sector_angle_rad: float | None = None
    hand: Literal["left", "right"] | None = None
    body_length_studs: float | None = None
    diverging_radius_studs: float | None = None

    # topology
    ports: Mapping[str, PortDef]
    routes: Mapping[str, tuple[str, ...]]

    @field_validator("ports")
    @classmethod
    def _port_A_is_origin(cls, v: Mapping[str, PortDef]) -> Mapping[str, PortDef]:
        a = v.get("A")
        if a is None or (a.dx, a.dy, a.dtheta) != (0.0, 0.0, 0.0):
            raise ValueError(
                "Port 'A' must exist and be at (0, 0, 0); "
                "the piece-local frame is defined by port A."
            )
        return v

    @model_validator(mode="after")
    def _routes_reference_real_ports(self):
        for route_name, port_seq in self.routes.items():
            missing = [p for p in port_seq if p not in self.ports]
            if missing:
                raise ValueError(
                    f"Route '{route_name}' references undefined port(s) {missing}; "
                    f"known ports are {sorted(self.ports)}."
                )
        return self

    @model_validator(mode="after")
    def _kind_geometry_complete(self):
        required: dict[str, list[str]] = {
            "straight": ["length_studs"],
            "curve": ["radius_studs", "sector_angle_rad"],
            "switch": ["body_length_studs", "diverging_radius_studs"],
            "wye": [],
            "crossing": [],
        }
        missing = [f for f in required.get(self.kind, [])
                   if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"kind='{self.kind}' requires field(s) {missing}; "
                f"piece_id='{self.piece_id}' is missing them."
            )
        return self

    @property
    def on_angle_lattice(self) -> bool:
        """True iff every port.dtheta is an integer multiple of π/32 within tolerance."""
        for port in self.ports.values():
            ratio = port.dtheta / ATOMIC_ANGLE_RAD
            nearest = round(ratio)
            if abs(nearest * ATOMIC_ANGLE_RAD - port.dtheta) > LATTICE_TOLERANCE:
                return False
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_catalog_specs.py::TestTrackPieceSpec -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/catalog/specs.py tests/test_catalog_specs.py
git commit -m "feat(catalog): add TrackPieceSpec with port-A/route/kind validators

- field_validator enforces port A at origin
- model_validator(after) enforces routes reference real ports
- model_validator(after) enforces kind-conditional required fields
- on_angle_lattice uses 1e-6 tolerance (corrected from V2's 1e-9)"
```

---

## Task 3: CatalogMeta + TrackCatalogSpec top-level models

**Files:**
- Modify: `src/catalog/specs.py`
- Modify: `tests/test_catalog_specs.py`

- [ ] **Step 1: Write failing tests for TrackCatalogSpec**

Append to `tests/test_catalog_specs.py`:

```python
class TestTrackCatalogSpec:
    def _minimal(self, pieces_override=None):
        from src.catalog.specs import TrackCatalogSpec
        if pieces_override is None:
            pieces_override = [
                dict(piece_id="straight_16", kind="straight", manufacturer="lego",
                     length_studs=16.0,
                     ports={"A": {"dx": 0, "dy": 0, "dtheta": 0},
                            "B": {"dx": 16, "dy": 0, "dtheta": 0}},
                     routes={"main": ("A", "B")}),
            ]
        return TrackCatalogSpec.model_validate(dict(
            meta={"schema_version": "1.0.0"},
            pieces=pieces_override,
        ))

    def test_valid_minimal_catalog(self):
        cat = self._minimal()
        assert cat.n_types == 1
        assert cat.piece_ids == ("straight_16",)

    def test_duplicate_piece_id_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self._minimal(pieces_override=[
                dict(piece_id="dup", kind="straight", manufacturer="lego",
                     length_studs=16.0,
                     ports={"A": {"dx": 0, "dy": 0, "dtheta": 0},
                            "B": {"dx": 16, "dy": 0, "dtheta": 0}},
                     routes={"main": ("A", "B")}),
                dict(piece_id="dup", kind="straight", manufacturer="lego",
                     length_studs=24.0,
                     ports={"A": {"dx": 0, "dy": 0, "dtheta": 0},
                            "B": {"dx": 24, "dy": 0, "dtheta": 0}},
                     routes={"main": ("A", "B")}),
            ])
        assert "Duplicate" in str(exc.value) or "unique" in str(exc.value).lower()

    def test_by_id_lookup(self):
        cat = self._minimal()
        assert cat.by_id["straight_16"].length_studs == 16.0

    def test_by_kind_filter(self):
        cat = self._minimal()
        straights = cat.by_kind("straight")
        assert len(straights) == 1
        assert straights[0].piece_id == "straight_16"

    def test_by_manufacturer_filter(self):
        cat = self._minimal()
        assert len(cat.by_manufacturer("lego")) == 1
        assert len(cat.by_manufacturer("4dbrix")) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_catalog_specs.py::TestTrackCatalogSpec -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add CatalogMeta + TrackCatalogSpec**

Append to `src/catalog/specs.py`:

```python
class CatalogMeta(BaseModel):
    model_config = _FROZEN

    schema_version: str = Field(min_length=1, description="MAJOR.MINOR.PATCH")
    unit: Literal["stud"] = "stud"
    stud_mm: float = 8.0
    angle_unit: Literal["rad"] = "rad"
    atomic_angle_rad: float = ATOMIC_ANGLE_RAD


class TrackCatalogSpec(BaseModel):
    """Top-level catalog: meta block + tuple of pieces."""

    model_config = _FROZEN

    meta: CatalogMeta
    pieces: tuple[TrackPieceSpec, ...]

    @model_validator(mode="after")
    def _piece_ids_unique(self):
        ids = [p.piece_id for p in self.pieces]
        dups = sorted({x for x in ids if ids.count(x) > 1})
        if dups:
            raise ValueError(f"Duplicate piece_id(s): {dups}. "
                             f"Each TrackPieceSpec.piece_id must be unique.")
        return self

    @property
    def by_id(self) -> Mapping[str, TrackPieceSpec]:
        return {p.piece_id: p for p in self.pieces}

    def by_kind(self, kind: str) -> tuple[TrackPieceSpec, ...]:
        return tuple(p for p in self.pieces if p.kind == kind)

    def by_manufacturer(self, m: str) -> tuple[TrackPieceSpec, ...]:
        return tuple(p for p in self.pieces if p.manufacturer == m)

    @property
    def n_types(self) -> int:
        return len(self.pieces)

    @property
    def piece_ids(self) -> tuple[str, ...]:
        """Canonical stable ordering; chromosome uses this index mapping."""
        return tuple(p.piece_id for p in self.pieces)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_catalog_specs.py::TestTrackCatalogSpec -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/catalog/specs.py tests/test_catalog_specs.py
git commit -m "feat(catalog): add CatalogMeta + TrackCatalogSpec root model"
```

---

## Task 4: SchemaVersionError + check_schema_version

**Files:**
- Modify: `src/catalog/specs.py`
- Modify: `tests/test_catalog_specs.py`

- [ ] **Step 1: Write failing tests for schema version policy**

Append to `tests/test_catalog_specs.py`:

```python
class TestSchemaVersion:
    def test_same_version_accepted_silently(self, caplog):
        from src.catalog.specs import check_schema_version
        check_schema_version("1.0.0")  # no exception, no warning
        assert not any("schema" in r.message.lower() for r in caplog.records)

    def test_patch_mismatch_accepted(self, caplog):
        from src.catalog.specs import check_schema_version
        check_schema_version("1.0.5")  # patch newer — silent accept
        assert not any("schema" in r.message.lower() for r in caplog.records)

    def test_minor_newer_warns(self, caplog):
        import logging
        from src.catalog.specs import check_schema_version
        with caplog.at_level(logging.WARNING, logger="src.catalog.specs"):
            check_schema_version("1.1.0")
        assert any("MINOR" in r.message or "newer" in r.message for r in caplog.records)

    def test_major_mismatch_rejected(self):
        from src.catalog.specs import check_schema_version, SchemaVersionError
        with pytest.raises(SchemaVersionError) as exc:
            check_schema_version("2.0.0")
        assert "MAJOR" in str(exc.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_catalog_specs.py::TestSchemaVersion -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add schema version machinery**

Append to `src/catalog/specs.py`:

```python
import logging
from packaging.version import Version

log = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSION = Version("1.0.0")


class SchemaVersionError(RuntimeError):
    """Raised when catalog schema MAJOR version is incompatible with code."""


def check_schema_version(file_version_str: str, path: str = "<catalog>") -> None:
    """Accept same-or-older PATCH silently, warn on MINOR newer, reject MAJOR mismatch."""
    file_ver = Version(file_version_str)
    code_ver = SUPPORTED_SCHEMA_VERSION

    if file_ver.major != code_ver.major:
        raise SchemaVersionError(
            f"{path}: schema MAJOR version mismatch — "
            f"file is {file_ver}, code supports {code_ver}. "
            f"Run migration: python -m legotrack.catalog.migrations."
            f"v{file_ver.major}_to_v{code_ver.major}"
        )
    if file_ver.minor > code_ver.minor:
        log.warning(
            "%s: schema MINOR version %s is newer than supported %s — "
            "unknown additive fields will be rejected by extra='forbid'.",
            path, file_ver, code_ver,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_catalog_specs.py::TestSchemaVersion -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/catalog/specs.py tests/test_catalog_specs.py
git commit -m "feat(catalog): add schema version policy (accept-patch/warn-minor/reject-major)

SchemaVersionError(RuntimeError) distinguishes tooling mismatch from data errors.
Called by loader before model_validate, not from inside a Pydantic validator."
```

---

## Task 5: Port C derivation test (3-4-5 geometry)

**Files:**
- Create: `tests/test_catalog_geometry.py`

This test documents the derivation but does NOT alter current switch YAML values (see "Non-goals" in the header). It serves as a reference point for future geometry correction.

- [ ] **Step 1: Create the derivation test**

Create `tests/test_catalog_geometry.py`:

```python
"""Derivation tests: V2 spec's Pythagorean-triple switch geometry.

These tests verify the REFERENCE derivation from V2's catalog report; they do
NOT assert that the current YAML matches these values. The current YAML
preserves historical switch FK values (~31.0, ±6.2) which differ from the
strict V2 derivation (~32.71, ±12.96). Correcting the catalog is a separate
task (see roadmap Phase 5 follow-up).
"""

import math
import numpy as np


def compute_lego_r40_switch_port_c() -> tuple[float, float, float]:
    """
    Two-arc compound path for LEGO R40 switch's diverging leg.

    Arc 1: +36.87° (arctan(3/4)) at R=40, left turn (CCW)
    Arc 2: -14.37° at R=40, right turn (CW)
    Net heading change: 22.5° = π/8

    Returns (dx, dy, dtheta) in studs/radians.
    """
    R = 40.0
    theta1 = math.atan2(3, 4)        # 36.87°, cos=4/5, sin=3/5
    cos1, sin1 = math.cos(theta1), math.sin(theta1)

    C1 = np.array([0.0, R])           # arc 1 center
    rot1 = np.array([[cos1, -sin1], [sin1, cos1]])
    end_vec1 = rot1 @ np.array([0.0, -R])  # (24, -32) expected
    p1 = C1 + end_vec1                # (24, 8)
    heading1 = theta1

    theta2 = math.pi / 8 - theta1     # ~-14.37° (negative = right turn)
    right_perp = np.array([math.sin(heading1), -math.cos(heading1)])
    C2 = p1 + R * right_perp          # (48, -24) expected
    cos2, sin2 = math.cos(theta2), math.sin(theta2)
    rot2 = np.array([[cos2, -sin2], [sin2, cos2]])
    end_vec2 = rot2 @ (p1 - C2)
    p_C = C2 + end_vec2
    return float(p_C[0]), float(p_C[1]), math.pi / 8


class TestSwitchGeometryReference:
    def test_arc_1_chord_is_integer_stud(self):
        """Arc 1 sweep produces the 8·(3,4,5) integer-stud chord."""
        R = 40.0
        theta1 = math.atan2(3, 4)
        cos1, sin1 = math.cos(theta1), math.sin(theta1)
        rot1 = np.array([[cos1, -sin1], [sin1, cos1]])
        end_vec1 = rot1 @ np.array([0.0, -R])
        assert abs(end_vec1[0] - 24.0) < 1e-9
        assert abs(end_vec1[1] - (-32.0)) < 1e-9

    def test_arc_1_end_at_integer_stud(self):
        """End of arc 1 is at piece-local (24, 8)."""
        R = 40.0
        theta1 = math.atan2(3, 4)
        cos1, sin1 = math.cos(theta1), math.sin(theta1)
        p1 = np.array([0, R]) + np.array([[cos1, -sin1], [sin1, cos1]]) @ np.array([0, -R])
        assert abs(p1[0] - 24.0) < 1e-9
        assert abs(p1[1] - 8.0) < 1e-9

    def test_arc_2_center_at_integer_stud(self):
        """Arc 2 center is at piece-local (48, -24)."""
        theta1 = math.atan2(3, 4)
        right_perp = np.array([math.sin(theta1), -math.cos(theta1)])
        p1 = np.array([24.0, 8.0])
        C2 = p1 + 40.0 * right_perp
        assert abs(C2[0] - 48.0) < 1e-9
        assert abs(C2[1] - (-24.0)) < 1e-9

    def test_port_c_derivation_matches_v2_spec(self):
        """V2's derivation: port C ≈ (32.71, 12.96, π/8)."""
        dx, dy, dtheta = compute_lego_r40_switch_port_c()
        assert abs(dx - 32.71) < 0.01, f"dx={dx:.4f}, expected ~32.71"
        assert abs(dy - 12.96) < 0.01, f"dy={dy:.4f}, expected ~12.96"
        assert abs(dtheta - math.pi / 8) < 1e-9

    def test_current_yaml_values_differ_from_v2_derivation(self):
        """
        REGRESSION GUARD: the YAML we ship has (31.0, 6.2) not (32.71, 12.96).
        This is a KNOWN divergence; correcting it is a separate task.
        This test documents the gap so future changes trip the right alarm.
        """
        v2_dx, v2_dy, _ = compute_lego_r40_switch_port_c()
        current_dx, current_dy = 31.0, 6.2   # from data/track_pieces.yaml
        assert abs(v2_dx - current_dx) > 1.0, "If this starts failing, geometry was corrected"
        assert abs(v2_dy - current_dy) > 6.0, "If this starts failing, geometry was corrected"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_catalog_geometry.py -v`
Expected: 5 passed. (These are pure derivation tests with no code dependency.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_catalog_geometry.py
git commit -m "test(catalog): reference derivation for LEGO R40 switch port C geometry

Documents V2's 3-4-5 Pythagorean-triple derivation (C ≈ 32.71, 12.96).
Pins the known divergence from current YAML (31.0, 6.2) with a regression guard.
Geometry correction deferred — see roadmap Phase 5 follow-up."
```

---

## Task 6: Fixture YAML + basic loader scaffold

**Files:**
- Create: `tests/fixtures/catalog_tiny.yaml`
- Create: `src/catalog/loader.py`
- Create: `tests/test_catalog_loader.py`

- [ ] **Step 1: Create the minimal fixture**

Create `tests/fixtures/catalog_tiny.yaml`:

```yaml
meta:
  schema_version: "1.0.0"
  unit: stud
  stud_mm: 8.0
  angle_unit: rad
  atomic_angle_rad: 0.19634954

pieces:
  - piece_id: lego_straight_16
    kind: straight
    manufacturer: lego
    part_numbers: ["53401"]
    length_studs: 16.0
    ports:
      A: {dx: 0.0,  dy: 0.0, dtheta: 0.0}
      B: {dx: 16.0, dy: 0.0, dtheta: 0.0}
    routes:
      main: [A, B]

  - piece_id: lego_curve_R40_left
    kind: curve
    manufacturer: lego
    part_numbers: ["53400"]
    radius_studs: 40.0
    sector_angle_rad: 0.39269908
    hand: left
    ports:
      A: {dx: 0.0,      dy: 0.0,     dtheta: 0.0}
      B: {dx: 15.307,   dy: 3.045,   dtheta: 0.39269908}
    routes:
      main: [A, B]
```

- [ ] **Step 2: Write failing test for happy-path load**

Create `tests/test_catalog_loader.py`:

```python
"""Tests for the ruamel.yaml + Pydantic catalog loader."""

from pathlib import Path
import pytest

FIXTURE_TINY = Path(__file__).parent / "fixtures" / "catalog_tiny.yaml"


class TestHappyPath:
    def test_loads_minimal_catalog(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(FIXTURE_TINY)
        assert cat.n_types == 2
        assert "lego_straight_16" in cat.by_id
        assert "lego_curve_R40_left" in cat.by_id

    def test_returns_track_catalog_spec_instance(self):
        from src.catalog.loader import load_catalog_spec
        from src.catalog.specs import TrackCatalogSpec
        cat = load_catalog_spec(FIXTURE_TINY)
        assert isinstance(cat, TrackCatalogSpec)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_catalog_loader.py::TestHappyPath -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.catalog.loader'`.

- [ ] **Step 4: Create `src/catalog/loader.py` happy-path loader**

```python
"""ruamel.yaml + Pydantic v2 catalog loader with file+line error UX."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .specs import TrackCatalogSpec, check_schema_version

log = logging.getLogger(__name__)


class CatalogLoadError(ValueError):
    """Raised when a YAML catalog fails schema validation after the loader
    has attached file+line context to each error."""


def load_catalog_spec(path: str | Path) -> TrackCatalogSpec:
    """Load a V2 catalog from YAML and return a validated TrackCatalogSpec.

    Uses ruamel.yaml round-trip mode so CommentedMap/CommentedSeq preserve
    .lc (line/column) on each node. Line numbers are re-attached to any
    Pydantic ValidationError as a wrapped CatalogLoadError.
    """
    path = Path(path)
    yaml = YAML()                       # default typ='rt' — preserves .lc
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.load(fh)

    if not isinstance(raw, CommentedMap):
        raise CatalogLoadError(f"{path}: root must be a mapping, got {type(raw).__name__}")

    # Enforce schema version before Pydantic sees the data
    meta = raw.get("meta") or {}
    version_str = meta.get("schema_version") if isinstance(meta, CommentedMap) else None
    if not version_str:
        raise CatalogLoadError(f"{path}: missing meta.schema_version")
    check_schema_version(str(version_str), str(path))

    # Map piece list index → 1-based line number for error reporting
    piece_lines: dict[int, int] = {}
    pieces = raw.get("pieces") or []
    if isinstance(pieces, CommentedSeq):
        for i, node in enumerate(pieces):
            if isinstance(node, CommentedMap) and node.lc is not None:
                piece_lines[i] = node.lc.line + 1   # 0-based → 1-based

    try:
        return TrackCatalogSpec.model_validate(_strip_comments(raw))
    except ValidationError as exc:
        _raise_with_location(exc, path, piece_lines)


def _strip_comments(obj):
    """Recursively convert CommentedMap/CommentedSeq to plain dict/list."""
    if isinstance(obj, CommentedMap):
        return {k: _strip_comments(v) for k, v in obj.items()}
    if isinstance(obj, CommentedSeq):
        return [_strip_comments(v) for v in obj]
    return obj


def _raise_with_location(exc: ValidationError, path: Path,
                         piece_lines: dict[int, int]) -> None:
    messages = []
    for err in exc.errors():
        loc = err["loc"]
        msg = err["msg"]
        typ = err["type"]

        line_hint = ""
        if loc and loc[0] == "pieces" and len(loc) > 1 and isinstance(loc[1], int):
            line_no = piece_lines.get(loc[1], "?")
            line_hint = f"{path.name}:{line_no} "

        field_path = ".".join(str(s) for s in loc)
        messages.append(f"{line_hint}in {field_path}: {msg} [type={typ}]")

    raise CatalogLoadError("\n".join(messages)) from exc
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_catalog_loader.py::TestHappyPath -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/catalog/loader.py tests/test_catalog_loader.py tests/fixtures/catalog_tiny.yaml
git commit -m "feat(catalog): ruamel.yaml + Pydantic v2 loader (happy path)

YAML() round-trip mode preserves CommentedMap.lc for line-number error UX.
_strip_comments converts to plain dicts before model_validate."
```

---

## Task 7: Loader error UX — file:line attribution

**Files:**
- Create: `tests/fixtures/catalog_bad_extra_field.yaml`
- Create: `tests/fixtures/catalog_bad_missing_port_a.yaml`
- Create: `tests/fixtures/catalog_bad_duplicate_id.yaml`
- Create: `tests/fixtures/catalog_bad_route_undefined_port.yaml`
- Create: `tests/fixtures/catalog_bad_major_version.yaml`
- Modify: `tests/test_catalog_loader.py`

- [ ] **Step 1: Create the five broken-catalog fixtures**

`tests/fixtures/catalog_bad_extra_field.yaml`:
```yaml
meta:
  schema_version: "1.0.0"
pieces:
  - piece_id: bad_extra
    kind: straight
    manufacturer: lego
    length_studs: 16.0
    color: red                         # extra field — forbidden
    ports:
      A: {dx: 0.0, dy: 0.0, dtheta: 0.0}
      B: {dx: 16.0, dy: 0.0, dtheta: 0.0}
    routes:
      main: [A, B]
```

`tests/fixtures/catalog_bad_missing_port_a.yaml`:
```yaml
meta:
  schema_version: "1.0.0"
pieces:
  - piece_id: bad_no_A
    kind: straight
    manufacturer: lego
    length_studs: 16.0
    ports:
      X: {dx: 0.0, dy: 0.0, dtheta: 0.0}
      B: {dx: 16.0, dy: 0.0, dtheta: 0.0}
    routes:
      main: [X, B]
```

`tests/fixtures/catalog_bad_duplicate_id.yaml`:
```yaml
meta:
  schema_version: "1.0.0"
pieces:
  - piece_id: dup
    kind: straight
    manufacturer: lego
    length_studs: 16.0
    ports:
      A: {dx: 0.0, dy: 0.0, dtheta: 0.0}
      B: {dx: 16.0, dy: 0.0, dtheta: 0.0}
    routes: {main: [A, B]}
  - piece_id: dup
    kind: straight
    manufacturer: lego
    length_studs: 24.0
    ports:
      A: {dx: 0.0, dy: 0.0, dtheta: 0.0}
      B: {dx: 24.0, dy: 0.0, dtheta: 0.0}
    routes: {main: [A, B]}
```

`tests/fixtures/catalog_bad_route_undefined_port.yaml`:
```yaml
meta:
  schema_version: "1.0.0"
pieces:
  - piece_id: bad_route
    kind: straight
    manufacturer: lego
    length_studs: 16.0
    ports:
      A: {dx: 0.0, dy: 0.0, dtheta: 0.0}
      B: {dx: 16.0, dy: 0.0, dtheta: 0.0}
    routes:
      main: [A, Z]                       # Z not in ports
```

`tests/fixtures/catalog_bad_major_version.yaml`:
```yaml
meta:
  schema_version: "2.0.0"                # MAJOR bump — unsupported
pieces:
  - piece_id: whatever
    kind: straight
    manufacturer: lego
    length_studs: 16.0
    ports:
      A: {dx: 0.0, dy: 0.0, dtheta: 0.0}
      B: {dx: 16.0, dy: 0.0, dtheta: 0.0}
    routes: {main: [A, B]}
```

- [ ] **Step 2: Write failing tests for each error UX**

Append to `tests/test_catalog_loader.py`:

```python
FIXTURES = Path(__file__).parent / "fixtures"


class TestErrorUX:
    def test_extra_field_reports_file_line_field(self):
        from src.catalog.loader import load_catalog_spec, CatalogLoadError
        with pytest.raises(CatalogLoadError) as exc:
            load_catalog_spec(FIXTURES / "catalog_bad_extra_field.yaml")
        msg = str(exc.value)
        assert "catalog_bad_extra_field.yaml:" in msg
        assert "pieces.0" in msg
        assert "extra_forbidden" in msg or "Extra" in msg

    def test_missing_port_a(self):
        from src.catalog.loader import load_catalog_spec, CatalogLoadError
        with pytest.raises(CatalogLoadError) as exc:
            load_catalog_spec(FIXTURES / "catalog_bad_missing_port_a.yaml")
        assert "Port 'A'" in str(exc.value)
        assert "catalog_bad_missing_port_a.yaml:" in str(exc.value)

    def test_duplicate_piece_id(self):
        from src.catalog.loader import load_catalog_spec, CatalogLoadError
        with pytest.raises(CatalogLoadError) as exc:
            load_catalog_spec(FIXTURES / "catalog_bad_duplicate_id.yaml")
        assert "Duplicate" in str(exc.value)
        assert "dup" in str(exc.value)

    def test_route_undefined_port(self):
        from src.catalog.loader import load_catalog_spec, CatalogLoadError
        with pytest.raises(CatalogLoadError) as exc:
            load_catalog_spec(FIXTURES / "catalog_bad_route_undefined_port.yaml")
        assert "undefined" in str(exc.value).lower() or "Z" in str(exc.value)

    def test_major_version_rejected(self):
        from src.catalog.loader import load_catalog_spec
        from src.catalog.specs import SchemaVersionError
        with pytest.raises(SchemaVersionError):
            load_catalog_spec(FIXTURES / "catalog_bad_major_version.yaml")
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_catalog_loader.py::TestErrorUX -v`
Expected: all 5 pass (the loader from Task 6 already handles these; this task adds the test fixtures and confirms the contract).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/catalog_bad_*.yaml tests/test_catalog_loader.py
git commit -m "test(catalog): file:line error UX for 5 authoring mistakes

Covers: extra_forbidden, missing port A, duplicate piece_id,
route with undefined port, MAJOR version mismatch."
```

---

## Task 8: Translate YAML straights

**Files:**
- Create: `data/track_pieces_v2.yaml` (partial — straights section only)

- [ ] **Step 1: Create the v2 YAML skeleton with straights**

```yaml
# LEGO/4DBrix Track Piece Catalog — V2 port-centric schema
# Source: data/track_pieces.yaml (v1 SECTION_TYPES form); derived per V2 spec.
# Units: studs internal; degrees in v1 source converted to radians below.
# FK values preserved verbatim — no geometry corrections in Phase 1.

meta:
  schema_version: "1.0.0"
  unit: stud
  stud_mm: 8.0
  angle_unit: rad
  atomic_angle_rad: 0.19634954                 # π/32 = 5.625°

pieces:
  # =========================================================================
  # STRAIGHTS
  # =========================================================================

  - piece_id: STRAIGHT_16
    kind: straight
    manufacturer: 4dbrix
    part_numbers: ["2.04.065"]
    length_studs: 16.0
    ports:
      A: {dx: 0.0,  dy: 0.0, dtheta: 0.0}
      B: {dx: 16.0, dy: 0.0, dtheta: 0.0}
    routes:
      main: [A, B]

  - piece_id: STRAIGHT_24
    kind: straight
    manufacturer: 4dbrix
    part_numbers: ["2.04.066"]
    length_studs: 24.0
    ports:
      A: {dx: 0.0,  dy: 0.0, dtheta: 0.0}
      B: {dx: 24.0, dy: 0.0, dtheta: 0.0}
    routes:
      main: [A, B]
```

- [ ] **Step 2: Load-and-verify test**

Append to `tests/test_catalog_loader.py`:

```python
class TestV2YamlStraights:
    def test_straight_16_matches_legacy_fk(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        straight = cat.by_id["STRAIGHT_16"]
        assert straight.kind == "straight"
        assert straight.length_studs == 16.0
        # port B matches legacy FK (dx=16, dy=0, dtheta=0)
        assert straight.ports["B"].dx == 16.0
        assert straight.ports["B"].dy == 0.0
        assert straight.ports["B"].dtheta == 0.0
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_catalog_loader.py::TestV2YamlStraights -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add data/track_pieces_v2.yaml tests/test_catalog_loader.py
git commit -m "feat(catalog): start v2 YAML — straights STRAIGHT_16/24

FK values preserved verbatim from v1 track_pieces.yaml."
```

---

## Task 9: Translate YAML curves

**Files:**
- Modify: `data/track_pieces_v2.yaml`
- Modify: `tests/test_catalog_loader.py`

- [ ] **Step 1: Append curves section to `data/track_pieces_v2.yaml`**

```yaml
  # =========================================================================
  # CURVES (R40, 22.5°)
  # FK: dx = R*sin(θ), dy = R*(1-cos(θ)), dtheta = θ
  # θ = 22.5° = π/8 rad
  # =========================================================================

  - piece_id: R40_LEFT
    kind: curve
    manufacturer: 4dbrix
    part_numbers: ["2.04.069"]
    radius_studs: 40.0
    sector_angle_rad: 0.39269908                # π/8
    hand: left
    ports:
      A: {dx: 0.0,     dy: 0.0,    dtheta: 0.0}
      B: {dx: 15.307,  dy: 3.045,  dtheta: 0.39269908}
    routes:
      main: [A, B]

  - piece_id: R40_RIGHT
    kind: curve
    manufacturer: 4dbrix
    part_numbers: ["2.04.069"]
    radius_studs: 40.0
    sector_angle_rad: 0.39269908
    hand: right
    ports:
      A: {dx: 0.0,     dy:  0.0,    dtheta:  0.0}
      B: {dx: 15.307,  dy: -3.045,  dtheta: -0.39269908}
    routes:
      main: [A, B]
```

- [ ] **Step 2: Append test for curve FK parity**

Append to `tests/test_catalog_loader.py`:

```python
class TestV2YamlCurves:
    def test_r40_left_matches_legacy_fk(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        curve = cat.by_id["R40_LEFT"]
        assert curve.kind == "curve"
        assert curve.radius_studs == 40.0
        assert abs(curve.sector_angle_rad - 0.39269908) < 1e-7
        b = curve.ports["B"]
        assert abs(b.dx - 15.307) < 1e-3
        assert abs(b.dy - 3.045) < 1e-3
        assert abs(b.dtheta - 0.39269908) < 1e-7

    def test_r40_right_has_negative_dy_and_dtheta(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        curve = cat.by_id["R40_RIGHT"]
        b = curve.ports["B"]
        assert b.dy < 0
        assert b.dtheta < 0
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_catalog_loader.py::TestV2YamlCurves -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add data/track_pieces_v2.yaml tests/test_catalog_loader.py
git commit -m "feat(catalog): add R40_LEFT/R40_RIGHT curves to v2 YAML

sector_angle_rad = π/8 (22.5° converted to radians).
Port B FK preserved from v1 (dx=15.307, dy=±3.045, dtheta=±π/8)."
```

---

## Task 10: Translate YAML switches (preserving historical FK)

**Files:**
- Modify: `data/track_pieces_v2.yaml`
- Modify: `tests/test_catalog_loader.py`

- [ ] **Step 1: Append four switch pieces**

```yaml
  # =========================================================================
  # SWITCHES (R40, 22.5°)
  # Historical FK preserved — diverge route uses (dx=31.0, dy=±6.2, dtheta=±π/8).
  # V2's 3-4-5 derivation gives (32.71, ±12.96); see test_catalog_geometry.py.
  # =========================================================================

  - piece_id: R40_SWITCH_LEFT_IN
    kind: switch
    manufacturer: 4dbrix
    part_numbers: ["2.04.021"]
    body_length_studs: 32.0
    diverging_radius_studs: 40.0
    hand: left
    ports:
      A: {dx: 0.0,    dy: 0.0,    dtheta: 0.0}        # throat
      B: {dx: 32.0,   dy: 0.0,    dtheta: 0.0}        # straight through
      C: {dx: 31.0,   dy: 6.2,    dtheta: 0.39269908} # diverging (left)
    routes:
      through:   [A, B]
      diverging: [A, C]

  - piece_id: R40_SWITCH_LEFT_OUT
    kind: switch
    manufacturer: 4dbrix
    part_numbers: ["2.04.022"]
    body_length_studs: 32.0
    diverging_radius_studs: 40.0
    hand: left
    ports:
      A: {dx: 0.0,    dy: 0.0,    dtheta: 0.0}
      B: {dx: 32.0,   dy: 0.0,    dtheta: 0.0}
      C: {dx: 1.0,    dy: 6.2,    dtheta: 0.39269908}
    routes:
      through:   [A, B]
      diverging: [A, C]

  - piece_id: R40_SWITCH_RIGHT_IN
    kind: switch
    manufacturer: 4dbrix
    part_numbers: ["2.04.018"]
    body_length_studs: 32.0
    diverging_radius_studs: 40.0
    hand: right
    ports:
      A: {dx: 0.0,    dy:  0.0,    dtheta:  0.0}
      B: {dx: 32.0,   dy:  0.0,    dtheta:  0.0}
      C: {dx: 31.0,   dy: -6.2,    dtheta: -0.39269908}
    routes:
      through:   [A, B]
      diverging: [A, C]

  - piece_id: R40_SWITCH_RIGHT_OUT
    kind: switch
    manufacturer: 4dbrix
    part_numbers: ["2.04.019"]
    body_length_studs: 32.0
    diverging_radius_studs: 40.0
    hand: right
    ports:
      A: {dx: 0.0,    dy:  0.0,    dtheta:  0.0}
      B: {dx: 32.0,   dy:  0.0,    dtheta:  0.0}
      C: {dx: 1.0,    dy: -6.2,    dtheta: -0.39269908}
    routes:
      through:   [A, B]
      diverging: [A, C]
```

- [ ] **Step 2: Write parity tests for all four switches**

Append to `tests/test_catalog_loader.py`:

```python
class TestV2YamlSwitches:
    @pytest.mark.parametrize("piece_id,port_c_dx,port_c_dy,port_c_dtheta", [
        ("R40_SWITCH_LEFT_IN",   31.0,  6.2,  0.39269908),
        ("R40_SWITCH_LEFT_OUT",   1.0,  6.2,  0.39269908),
        ("R40_SWITCH_RIGHT_IN",  31.0, -6.2, -0.39269908),
        ("R40_SWITCH_RIGHT_OUT",  1.0, -6.2, -0.39269908),
    ])
    def test_switch_port_c_matches_legacy_fk(self, piece_id, port_c_dx, port_c_dy, port_c_dtheta):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        sw = cat.by_id[piece_id]
        assert sw.kind == "switch"
        assert sw.body_length_studs == 32.0
        c = sw.ports["C"]
        assert abs(c.dx - port_c_dx) < 1e-3
        assert abs(c.dy - port_c_dy) < 1e-3
        assert abs(c.dtheta - port_c_dtheta) < 1e-7
        assert set(sw.routes.keys()) == {"through", "diverging"}
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_catalog_loader.py::TestV2YamlSwitches -v`
Expected: 4 passed (parametrized).

- [ ] **Step 4: Commit**

```bash
git add data/track_pieces_v2.yaml tests/test_catalog_loader.py
git commit -m "feat(catalog): add 4 R40 switches to v2 YAML

Historical FK preserved; port C matches v1 diverge-route fk values.
V2's 3-4-5 Pythagorean derivation is documented but NOT adopted here
(non-behavioral migration)."
```

---

## Task 11: Translate YAML crossings

**Files:**
- Modify: `data/track_pieces_v2.yaml`
- Modify: `tests/test_catalog_loader.py`

- [ ] **Step 1: Append CROSS_90 and DOUBLE_CROSSOVER**

```yaml
  # =========================================================================
  # CROSSINGS
  # =========================================================================

  - piece_id: CROSS_90
    kind: crossing
    manufacturer: 4dbrix
    part_numbers: ["2.04.002"]
    length_studs: 16.0
    ports:
      A: {dx: 0.0,  dy:  0.0,  dtheta:  0.0}                # west (port 0 entry)
      B: {dx: 16.0, dy:  0.0,  dtheta:  0.0}                # east (port 1)
      C: {dx: 8.0,  dy: -8.0,  dtheta: -1.57079633}         # south (port 2), -π/2
      D: {dx: 8.0,  dy:  8.0,  dtheta:  1.57079633}         # north (port 3), +π/2
    routes:
      horizontal: [A, B]
      vertical:   [C, D]

  - piece_id: DOUBLE_CROSSOVER
    kind: crossing
    manufacturer: 4dbrix
    part_numbers: ["210.1"]
    length_studs: 48.0
    ports:
      A: {dx: 0.0,  dy:  0.0,  dtheta: 0.0}    # entry_track1
      B: {dx: 48.0, dy:  0.0,  dtheta: 0.0}    # exit_track1
      C: {dx: 0.0,  dy: 16.0,  dtheta: 0.0}    # entry_track2
      D: {dx: 48.0, dy: 16.0,  dtheta: 0.0}    # exit_track2
    routes:
      track1_through: [A, B]
      track2_through: [C, D]
      cross_1_to_2:   [A, D]
      cross_2_to_1:   [C, B]
```

- [ ] **Step 2: Append tests**

```python
class TestV2YamlCrossings:
    def test_cross_90_has_4_ports_and_2_routes(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        x = cat.by_id["CROSS_90"]
        assert x.kind == "crossing"
        assert set(x.ports.keys()) == {"A", "B", "C", "D"}
        assert set(x.routes.keys()) == {"horizontal", "vertical"}

    def test_double_crossover_has_4_routes(self):
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        x = cat.by_id["DOUBLE_CROSSOVER"]
        assert x.kind == "crossing"
        assert x.length_studs == 48.0
        assert set(x.routes.keys()) == {"track1_through", "track2_through",
                                         "cross_1_to_2", "cross_2_to_1"}

    def test_v2_catalog_has_all_10_pieces(self):
        """Every piece from v1 track_pieces.yaml appears in v2."""
        from src.catalog.loader import load_catalog_spec
        cat = load_catalog_spec(Path("data") / "track_pieces_v2.yaml")
        expected = {
            "STRAIGHT_16", "STRAIGHT_24",
            "R40_LEFT", "R40_RIGHT",
            "R40_SWITCH_LEFT_IN", "R40_SWITCH_LEFT_OUT",
            "R40_SWITCH_RIGHT_IN", "R40_SWITCH_RIGHT_OUT",
            "CROSS_90", "DOUBLE_CROSSOVER",
        }
        assert set(cat.by_id.keys()) == expected
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_catalog_loader.py::TestV2YamlCrossings -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add data/track_pieces_v2.yaml tests/test_catalog_loader.py
git commit -m "feat(catalog): complete v2 YAML with crossings (CROSS_90, DOUBLE_CROSSOVER)

All 10 legacy pieces now present in v2 YAML."
```

---

## Task 12: Backward-compat shim — V2Catalog → legacy numpy tables

**Files:**
- Modify: `src/catalog/catalog.py`
- Create: `tests/test_catalog_parity.py`

- [ ] **Step 1: Write parity tests FIRST (before touching catalog.py)**

Create `tests/test_catalog_parity.py`:

```python
"""Parity tests: v1 vs v2 catalogs produce bit-identical numpy tables."""

from pathlib import Path

import numpy as np
import pytest

from src.catalog import TrackCatalog


V1_PATH = Path("data") / "track_pieces.yaml"
V2_PATH = Path("data") / "track_pieces_v2.yaml"


@pytest.fixture
def v1_catalog():
    return TrackCatalog.load(V1_PATH)


@pytest.fixture
def v2_catalog():
    return TrackCatalog.load(V2_PATH)


class TestFKTableParity:
    def test_both_catalogs_have_same_piece_count(self, v1_catalog, v2_catalog):
        assert v1_catalog.n_pieces == v2_catalog.n_pieces

    def test_fk_table_shape_matches(self, v1_catalog, v2_catalog):
        assert v1_catalog.fk_table.shape == v2_catalog.fk_table.shape

    def test_radius_table_matches(self, v1_catalog, v2_catalog):
        np.testing.assert_array_almost_equal(
            v1_catalog.radius_table, v2_catalog.radius_table, decimal=3)

    def test_speed_table_matches(self, v1_catalog, v2_catalog):
        np.testing.assert_array_almost_equal(
            v1_catalog.speed_table, v2_catalog.speed_table, decimal=3)

    def test_id_to_index_matches(self, v1_catalog, v2_catalog):
        assert v1_catalog._id_to_index == v2_catalog._id_to_index

    @pytest.mark.parametrize("piece_id", [
        "STRAIGHT_16", "STRAIGHT_24",
        "R40_LEFT", "R40_RIGHT",
        "CROSS_90", "DOUBLE_CROSSOVER",
        "R40_SWITCH_LEFT_IN", "R40_SWITCH_LEFT_OUT",
        "R40_SWITCH_RIGHT_IN", "R40_SWITCH_RIGHT_OUT",
    ])
    def test_per_piece_fk_matches(self, v1_catalog, v2_catalog, piece_id):
        idx1 = v1_catalog._id_to_index[piece_id]
        idx2 = v2_catalog._id_to_index[piece_id]
        np.testing.assert_array_almost_equal(
            v1_catalog.fk_table[idx1], v2_catalog.fk_table[idx2], decimal=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_catalog_parity.py -v`
Expected: FAIL — `TrackCatalog.load(V2_PATH)` fails because `TrackCatalog` doesn't know how to read V2 YAML yet.

- [ ] **Step 3: Modify `src/catalog/catalog.py` to detect V2 and wrap**

Open `src/catalog/catalog.py`. At the top of the file, add imports:

```python
from functools import cached_property

from .loader import load_catalog_spec, CatalogLoadError
from .specs import TrackCatalogSpec
```

Locate the `TrackCatalog.load` classmethod (around lines 50–60) and replace it with:

```python
    @classmethod
    def load(cls, path: str | Path) -> "TrackCatalog":
        """Load catalog from YAML. Auto-detects v1 (section-keyed) vs v2 (port-centric)."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if _is_v2_schema(data):
            spec = load_catalog_spec(path)
            return cls._from_v2_spec(spec)

        # Legacy v1 path — preserved verbatim
        catalog = cls()
        catalog._parse_yaml(data)
        catalog._build_tables()
        return catalog

    @classmethod
    def _from_v2_spec(cls, spec: TrackCatalogSpec) -> "TrackCatalog":
        """Build a legacy-surface TrackCatalog from a V2 TrackCatalogSpec."""
        import math
        from .pieces import FKDeltas, Port, TrackPiece

        catalog = cls()
        catalog.stud_mm = spec.meta.stud_mm

        for idx, ps in enumerate(spec.pieces):
            # Pick the default main route; switches/crossings use the first declared route
            main_route_name = _default_route_name(ps)
            main_route = ps.routes[main_route_name]
            exit_port_name = main_route[-1]
            exit_port = ps.ports[exit_port_name]

            fk = FKDeltas(
                dx=exit_port.dx,
                dy=exit_port.dy,
                dtheta=math.degrees(exit_port.dtheta),   # legacy table stores DEGREES
            )
            ports = tuple(
                Port(
                    x=port.dx, y=port.dy,
                    heading=math.degrees(port.dtheta),
                    gender="M" if name != "A" else "F",  # heuristic; decoder ignores
                )
                for name, port in ps.ports.items()
            )

            piece = TrackPiece(
                id=ps.piece_id,
                name=ps.piece_id,   # no dedicated name in V2; reuse piece_id
                piece_type=ps.kind,
                fk=fk,
                ports=ports,
                index=idx,
                length=ps.length_studs or ps.body_length_studs or 16.0,
                radius=ps.radius_studs or ps.diverging_radius_studs,
                angle=math.degrees(ps.sector_angle_rad) if ps.sector_angle_rad else None,
                direction=ps.hand,
                radius_mm=(ps.radius_studs * spec.meta.stud_mm) if ps.radius_studs else None,
                speed_limit_ms=cls.DEFAULT_SPEED,
                routes_data=_build_legacy_routes(ps),
            )
            catalog._pieces[ps.piece_id] = piece
            catalog._index_to_piece[idx] = piece
            catalog._id_to_index[ps.piece_id] = idx
            catalog._max_index = max(catalog._max_index, idx)

        catalog._build_tables()
        return catalog


def _is_v2_schema(data: dict) -> bool:
    """V2 has top-level `pieces:` list and `meta.schema_version`; v1 has `straights:` etc."""
    if not isinstance(data, dict):
        return False
    return (
        "pieces" in data
        and "meta" in data
        and isinstance(data.get("meta"), dict)
        and "schema_version" in data["meta"]
    )


def _default_route_name(ps) -> str:
    """Pick the route whose exit port is the 'main FK' row for the legacy table."""
    # straights/curves use 'main'; switches default to 'through'; crossings to first route
    if "main" in ps.routes:
        return "main"
    if "through" in ps.routes:
        return "through"
    return next(iter(ps.routes))


def _build_legacy_routes(ps) -> list:
    """Reconstruct the legacy routes_data list the decoder reads for multi-port pieces."""
    import math
    out = []
    for i, (name, port_seq) in enumerate(ps.routes.items()):
        exit_port = ps.ports[port_seq[-1]]
        entry_port = ps.ports[port_seq[0]]
        out.append({
            "name": name,
            "entry_port": list(ps.ports).index(port_seq[0]),
            "exit_port": list(ps.ports).index(port_seq[-1]),
            "fk": {
                "dx": exit_port.dx,
                "dy": exit_port.dy,
                "dtheta": math.degrees(exit_port.dtheta),
            },
        })
    return out
```

- [ ] **Step 4: Run parity tests**

Run: `pytest tests/test_catalog_parity.py -v`
Expected: all passed (7 tests including the 10-piece parametrize).

- [ ] **Step 5: Run the existing test suite**

Run: `pytest tests/test_catalog.py -v`
Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/catalog/catalog.py tests/test_catalog_parity.py
git commit -m "feat(catalog): V2→legacy shim with auto-detection of v1/v2 YAML

TrackCatalog.load() detects schema by presence of top-level 'pieces' +
'meta.schema_version'. V2 specs are converted into the legacy TrackPiece
structure, preserving _fk_table/_radius_table/_speed_table bit-for-bit.
Angles converted degrees↔radians at the boundary."
```

---

## Task 13: Switch optimizer-facing load path to v2 YAML

**Files:**
- Modify: every config under `configs/*.yaml` that references `data/track_pieces.yaml`
- Grep first: confirm nothing hardcodes the v1 path in `src/`

- [ ] **Step 1: Find call sites of `track_pieces.yaml`**

Run: `grep -rn "track_pieces.yaml" configs/ src/ main.py`
Note every file; most hits will be in `configs/*.yaml` or a fallback default in `src/config.py` or `main.py`.

- [ ] **Step 2: Run the test suite as a baseline**

Run: `pytest tests/ -q`
Expected: all tests pass (using v1 YAML still).

- [ ] **Step 3: Update each config file to point at v2**

For each `configs/*.yaml` that has a `track_pieces: data/track_pieces.yaml` or similar, change the value to `data/track_pieces_v2.yaml`.

If no config file references a catalog path (because the default is baked into `main.py` or `src/config.py`), skip this step and address the default in step 4.

- [ ] **Step 4: If there's a default catalog path in Python, update it**

Grep: `grep -n "track_pieces" src/*.py main.py`. If `src/config.py` or `main.py` has a default like `"data/track_pieces.yaml"`, change it to `"data/track_pieces_v2.yaml"`.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -q`
Expected: all tests pass — decoder, geometry, problem, sampling, etc. are all catalog-consumers and must work against the V2-loaded catalog.

- [ ] **Step 6: Run optimizer smoke test**

Run: `python main.py --config configs/default.yaml --quick`
Expected: exits cleanly, produces `outputs/` with a `best_layout.png`.

- [ ] **Step 7: Run diagnostic**

Use the `/diag` skill. Expected: `closure_error < 4`, `feasible_count > 0`, layout is closed.

- [ ] **Step 8: Commit**

```bash
git add configs/ src/config.py main.py
git commit -m "chore(catalog): point default configs and CLI at track_pieces_v2.yaml

All legacy section-keyed consumers now exercise the V2 spec via the
backward-compat shim. Quick optimizer smoke test green."
```

---

## Task 14: Deprecation warning on v1 YAML

**Files:**
- Modify: `src/catalog/catalog.py`
- Modify: `tests/test_catalog.py` (append one test)

- [ ] **Step 1: Write a test asserting v1 YAML load emits DeprecationWarning**

Append to `tests/test_catalog.py`:

```python
class TestV1Deprecation:
    def test_loading_v1_yaml_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TrackCatalog.load("data/track_pieces.yaml")
            deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert deprecations, "loading v1 YAML should emit DeprecationWarning"
            assert "v1" in str(deprecations[0].message).lower() or \
                   "legacy" in str(deprecations[0].message).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_catalog.py::TestV1Deprecation -v`
Expected: FAIL — no warning emitted yet.

- [ ] **Step 3: Emit warning in the v1 load path**

In `src/catalog/catalog.py`, modify `load`:

```python
    @classmethod
    def load(cls, path: str | Path) -> "TrackCatalog":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if _is_v2_schema(data):
            spec = load_catalog_spec(path)
            return cls._from_v2_spec(spec)

        import warnings
        warnings.warn(
            f"Loading legacy v1 catalog format from {path}. "
            f"Migrate to V2 port-centric schema (see data/track_pieces_v2.yaml). "
            f"v1 support will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        catalog = cls()
        catalog._parse_yaml(data)
        catalog._build_tables()
        return catalog
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_catalog.py::TestV1Deprecation -v`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add src/catalog/catalog.py tests/test_catalog.py
git commit -m "chore(catalog): DeprecationWarning when loading v1 YAML format"
```

---

## Task 15: Update `src/catalog/__init__.py` exports

**Files:**
- Modify: `src/catalog/__init__.py`

- [ ] **Step 1: Inspect current exports**

Run: `cat src/catalog/__init__.py`

- [ ] **Step 2: Add V2 types to the public surface**

Replace the file with a version that re-exports both V1 and V2 surfaces:

```python
"""Track catalog package — V1 (legacy) and V2 (port-centric) surfaces."""

# V1 (legacy) — kept for backward compatibility during migration
from .catalog import TrackCatalog
from .pieces import FKDeltas, Port, TrackPiece

# V2 — new port-centric domain model + loader
from .specs import (
    PortDef,
    TrackPieceSpec,
    CatalogMeta,
    TrackCatalogSpec,
    SchemaVersionError,
    check_schema_version,
    ATOMIC_ANGLE_RAD,
    LATTICE_TOLERANCE,
    SUPPORTED_SCHEMA_VERSION,
)
from .loader import load_catalog_spec, CatalogLoadError


__all__ = [
    # V1
    "TrackCatalog", "FKDeltas", "Port", "TrackPiece",
    # V2
    "PortDef", "TrackPieceSpec", "CatalogMeta", "TrackCatalogSpec",
    "SchemaVersionError", "check_schema_version",
    "load_catalog_spec", "CatalogLoadError",
    "ATOMIC_ANGLE_RAD", "LATTICE_TOLERANCE", "SUPPORTED_SCHEMA_VERSION",
]
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/catalog/__init__.py
git commit -m "chore(catalog): export V2 types (PortDef, TrackCatalogSpec, loader) from package"
```

---

## Task 16: Import-linter layer contract

**Files:**
- Modify: `.importlinter`

- [ ] **Step 1: Inspect current contracts**

Run: `cat .importlinter`

- [ ] **Step 2: Add catalog-internal independence contract**

Append (or adjust) a contract that forbids `src.catalog` from importing `src.geometry`, `src.problem`, `src.decoder`, `src.operators`, etc.:

```ini
[importlinter:contract:catalog-foundation]
name = Catalog is a leaf domain package — no upward imports
type = forbidden
source_modules =
    src.catalog
forbidden_modules =
    src.geometry
    src.train
    src.decoder
    src.encoding
    src.problem
    src.operators
    src.sampling
    src.repair
    src.templates
    src.intersection
    src.visualization
    src.algorithm
```

- [ ] **Step 3: Run lint-imports**

Run: `lint-imports`
Expected: contract passes (catalog already imports only stdlib, numpy, pydantic, ruamel, packaging, yaml, and its own submodules).

- [ ] **Step 4: Commit**

```bash
git add .importlinter
git commit -m "build(lint): pin catalog as a leaf domain package (no upward imports)"
```

---

## Task 17: Full regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run full test suite verbose**

Run: `pytest tests/ -v`
Expected: all tests pass. Count of passed tests should be the baseline from Task 0 **plus the new tests from Tasks 1–14** (roughly +30).

- [ ] **Step 2: Run full optimizer with default config**

Invoke `/optimize` skill with default config (not `--quick`), or run: `python main.py --config configs/default.yaml`
Expected: completes successfully; best_layout.png produced.

- [ ] **Step 3: Run `/diag` on the outputs**

Invoke `/diag` skill. Expected: `closure_error ≤ 4 studs`, feasible solutions present, layout closed.

- [ ] **Step 4: Run `/optimize -c with_switches --quick`**

Verify switches-enabled config works against V2 catalog. Expected: completes; best layout has branches.

- [ ] **Step 5: Run `/optimize -c with_crossing --quick`**

Verify crossings config works against V2 catalog. Expected: completes; best layout has crossings.

- [ ] **Step 6: Commit (if any incidental doc or fixture tweaks)**

```bash
git add -A
git status
# If there are changes, commit:
git commit -m "test(catalog): Phase 1 regression sweep — all configs pass on V2 YAML"
```

If nothing changed, skip the commit.

---

## Task 18: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the catalog row in the Project Structure table**

In `CLAUDE.md` under "Core Code (`/src`)", update the `catalog/` row to:

```markdown
| `catalog/` | `TrackCatalog` (legacy shim) + `TrackCatalogSpec`/`TrackPieceSpec`/`PortDef` (V2 port-centric), `CatalogMeta`, `load_catalog_spec`, schema versioning |
```

- [ ] **Step 2: Add a note under "Adding Track Pieces"**

Replace the "Adding Track Pieces" block with guidance for V2:

```markdown
### Adding Track Pieces (V2 schema)
1. Add to `data/track_pieces_v2.yaml` under `pieces:` with `piece_id`, `kind`, `manufacturer`, `ports: {A, B, ...}`, `routes: {...}`, and kind-conditional fields (`radius_studs` + `sector_angle_rad` for curves; `body_length_studs` + `diverging_radius_studs` for switches).
2. Port A must be at `(0, 0, 0)`; other ports use `PortDef(dx, dy, dtheta)` with dtheta in radians.
3. Named routes declare traversal paths, e.g. `through: [A, B]`, `diverging: [A, C]`.
4. Schema is versioned: a MAJOR change requires a migration script.
5. Run `/test test_catalog_specs test_catalog_loader test_catalog_parity` to validate.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for V2 catalog schema"
```

---

# Self-Review

**Spec coverage:**

| V2 catalog spec section | Phase 1 task | Status |
|---|---|---|
| PortDef / TrackPieceSpec models | 1, 2 | ✅ |
| port-A-at-origin validator | 2 | ✅ |
| routes-reference-real-ports validator | 2 | ✅ |
| kind-conditional required fields | 2 | ✅ |
| on_angle_lattice property | 2 (with 1e-6 correction) | ✅ |
| CatalogMeta + TrackCatalogSpec | 3 | ✅ |
| duplicate piece_id rejection | 3 | ✅ |
| by_id / by_kind / by_manufacturer | 3 | ✅ |
| schema version policy | 4 | ✅ |
| 3-4-5 Pythagorean port C derivation | 5 | ✅ (documented, not adopted) |
| ruamel.yaml round-trip loader | 6 | ✅ |
| file:line error UX (6 error types) | 7 | ✅ (5 types; `missing required field` covered by Pydantic's own `missing` error shape in Task 2) |
| all 10 current pieces translated | 8, 9, 10, 11 | ✅ |
| backward-compat shim (legacy numpy tables) | 12 | ✅ |
| auto-detection of v1/v2 YAML format | 12 | ✅ |
| DeprecationWarning for v1 | 14 | ✅ |
| V2 types exported | 15 | ✅ |
| import-linter contract | 16 | ✅ |
| regression sweep | 17 | ✅ |
| CLAUDE.md update | 18 | ✅ |

**Deferred (per "Non-goals"):**
- V2's 3-4-5 switch geometry correction — tracked as a Phase 5 follow-up
- Retirement of `data/track_pieces.yaml` — will happen at the end of Batch 1 once Geometry and Train are also migrated
- Multi-manufacturer pieces (Fx Bricks, BrickTracks, Trixbrix) — out of scope for Phase 1

**Placeholder scan:** No "TBD", "TODO", "fill in details", or abstract guidance. Every step has exact code or exact commands.

**Type consistency:**
- `TrackPieceSpec`, `TrackCatalogSpec`, `PortDef`, `CatalogMeta` — used consistently
- `load_catalog_spec` — single name across loader and tests
- `CatalogLoadError`, `SchemaVersionError` — distinct types; loader raises the former, version check raises the latter
- Fixture paths consistent (`tests/fixtures/catalog_*.yaml`)
- dtheta stored as **radians** in V2, converted to **degrees** at the legacy boundary inside `_from_v2_spec`

**Commit count:** ~16 commits, matching the roadmap estimate of "~10 commits, medium risk."

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-20-phase-1-catalog.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints

Which approach?