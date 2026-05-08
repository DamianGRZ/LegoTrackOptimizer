# Designing a catalog package for LEGO track optimization

**The catalog package must model track pieces as immutable, port-centric geometry objects loaded from YAML and indexed for O(1) lookup.** This report synthesizes precise dimensional data for all 4DBrix/LEGO-compatible track pieces, confirms the Pythagorean-triple foundations of switch geometry, and recommends a concrete implementation stack: ruamel.yaml for YAML 1.2 parsing, Pydantic v2 frozen models for validation, and a module-level registry with derived indices. Every dimension below has been cross-referenced against L-Gauge.org, Fx Bricks engineering documentation, the MacFreek geometry analysis, the transponderings.blog comparative catalog (May 2024), and MattzoBricks Geometry Corner.

## The foundational numbers every other module depends on

All LEGO-compatible track shares a common dimensional grammar. **One stud equals exactly 8 mm.** Rail gauge is 5 studs (40 mm) measured center-to-center between rails; the running-surface gauge is 37.5 mm. Every track piece is **8 studs wide** (64 mm). The standard straight is 16 studs (128 mm) long. Parallel tracks are spaced 16 studs center-to-center, matching 32×32 baseplate edges. All radii are measured to the **track centerline** — so an R40 curve has its center 40 studs from the midpoint between rails, with the inner track edge at 36 studs and the outer edge at 44 studs.

Studs should be the canonical internal unit. BlueBrick, the primary LEGO layout tool, uses studs for all port coordinates. Integer and clean-rational values dominate: straight lengths are 4, 8, 16, 32 studs; radii are 40, 56, 72, 88, 104, 120, 148 studs. Floating-point studs work fine for computed positions. A single constant `STUD_MM = 8.0` handles conversion for display or export.

## Curve geometry across manufacturers reveals critical incompatibilities

The L-Gauge.org standard defines a family of increasing radii at 16-stud intervals. The standard arc angles per piece differ between the smaller and larger radii, and — critically — **4DBrix uses non-standard angles** at several radii that differ from every other manufacturer.

| Radius | L-Gauge / TrixBrix / Fx Bricks | 4DBrix angle | 4DBrix pcs/circle | Standard pcs/circle |
|--------|-------------------------------|-------------|-------------------|-------------------|
| **R40** | 22.5° | 22.5° | 16 | 16 |
| **R56** | 22.5° | **18.0°** | **20** | 16 |
| **R72** | 22.5° (Fx) / 11.25° (TrixBrix) | **15.0°** | **24** | 16 or 32 |
| **R88** | 11.25° | 11.25° | 32 | 32 |
| **R104** | 11.25° | **10.0°** | **36** | 32 |
| **R120** | 11.25° | **9.0°** | **40** | 32 |
| **R136** | 11.25° | — | — | 32 |
| **R148** | 5.625° | (crossovers only) | 64 | 64 |
| **R152** | 11.25° | — | — | 32 |

The pattern for standard angles is simple: R40/R56/R72 use 22.5° (4 pieces per quarter-turn, 16 per circle), while R88 through R152 use 11.25° (8 per quarter, 32 per circle). R148 is the outlier at 5.625° (16 per quarter, 64 per circle). The 4DBrix angles break this pattern — their R56 at 18° means 5 pieces per quarter-turn (20 per circle), R72 at 15° means 6 per quarter (24 per circle), and so on. These 4DBrix curves **cannot be mixed with same-radius curves from TrixBrix or Fx Bricks** within a single circle, even though they physically connect via identical rail clips. The YAML catalog must therefore distinguish pieces by manufacturer, not just by radius.

For any curve with radius R studs and arc angle θ degrees, port B's local coordinates are computed as:

```
x_B = R × sin(θ)
y_B = R × (1 − cos(θ))
heading_B = θ
```

Selected port B positions (left-turning curves, port A at origin):

| Piece | R | θ | x_B (studs) | y_B (studs) |
|-------|---|---|------------|------------|
| LEGO R40 | 40 | 22.5° | 15.307 | 3.045 |
| 4DBrix R56 | 56 | 18.0° | 17.308 | 2.741 |
| 4DBrix R72 | 72 | 15.0° | 18.634 | 2.456 |
| 4DBrix R88 | 88 | 11.25° | 17.168 | 1.690 |
| 4DBrix R104 | 104 | 10.0° | 18.060 | 1.582 |
| 4DBrix R120 | 120 | 9.0° | 18.772 | 1.478 |

## Two Pythagorean triples govern all switch geometry

The most elegant mathematical result in LEGO track geometry is that switch diverging angles derive from Pythagorean triples, which guarantee that S-bend configurations land on exact integer-stud grid coordinates.

**The 3-4-5 triple (×8) controls the standard LEGO R40 switch.** The switch body is 32 studs long. Its diverging route consists of two R40 arcs: a **36.87° outward arc** (where 36.87° = arctan(3/4)) followed by a **14.37° return arc**, yielding a net exit angle of **22.5°**. The underlying right triangle is (24, 32, 40) studs — exactly 8 × (3, 4, 5). Each half of the S-bend to reach a parallel siding advances 24 studs forward and 8 studs laterally. A complete S-bend (switch + one R40 curve + one S16 straight) covers 48 studs forward and 16 studs lateral offset, landing perfectly on the stud grid.

The diverging port C of a standard LEGO switch sits at approximately **(32.69, 12.96) studs** with heading **22.5°**, confirmed by BlueBrick library coordinates for the 9V switch part. This position results from the compound two-arc path within the 32-stud switch body.

**The 5-12-13 triple (×8) controls the Fx Bricks P40 switch.** Fx Bricks designed their P40 switch at 40 studs long with a diverging angle of **22.62° = arctan(5/12)**, not the superficially similar 22.5°. The underlying triangle is (40, 96, 104) studs = 8 × (5, 12, 13). Each S-bend half advances 40 studs forward and 8 studs laterally; the full S-bend covers 80 studs and 16 studs offset. The return curve uses the special **R64P** element at exactly 22.62° — not suitable for complete circles (16 × 22.62° = 361.9° ≠ 360°) but geometrically perfect for grid-aligned sidings. The combination S8 + R64P + S8 is exactly equivalent to the P40 diverging path.

The table of Pythagorean triples relevant to LEGO track S-bends (all producing 16-stud parallel offset):

| Triple | Forward (studs) | R (studs) | Offset per half | Full S-bend length | Angle θ | Used by |
|--------|----------------|-----------|----------------|-------------------|---------|---------|
| 3, 4, 5 | 24 | 40 | 8 | 48 | 36.87° | LEGO switch |
| 5, 12, 13 | 40 | 104 | 8 | 80 | 22.62° | Fx Bricks P40 |
| 7, 24, 25 | 56 | 200 | 8 | 112 | 16.26° | Theoretical |
| 9, 40, 41 | 72 | 328 | 8 | 144 | 12.68° | Theoretical |

## Port-centric representation is the proven data model

Every major track layout tool — XTrackCAD, BlueBrick, JMRI — converges on the same core abstraction: each track piece type defines **named ports** with local (x, y, heading) coordinates, and two pieces connect when ports have matching world positions and headings that differ by 180°.

The recommended coordinate convention for the Python optimizer is the standard mathematical frame: **x = forward** (along the reference track direction), **y = left** (perpendicular, positive leftward), **heading = CCW from x-axis** in degrees. This avoids BlueBrick's inverted-Y confusion and is natural for trigonometry. Port A of every piece sits at the origin (0, 0) with heading 180° (outward direction pointing backward); the through-route exit port aligns with the +x axis.

Port counts by piece type:

- **Straight**: 2 ports (A, B) — entry and exit along a line
- **Curve**: 2 ports (A, B) — entry and exit separated by arc angle
- **Turnout/Switch**: 3 ports (A = throat, B = through, C = diverging)
- **Wye**: 3 ports (A = throat, B = left diverge, C = right diverge)
- **Crossing**: 4 ports (A, B on one track; C, D on the crossing track)
- **Single crossover**: 4 ports (A, B on track 1; C, D on track 2)
- **Double crossover**: 4 ports with two independent crossing paths

For multi-route pieces (switches, crossovers), the catalog should define **named paths** as ordered port sequences — following XTrackCAD's proven approach. A left R40 switch defines `through: [A, B]` and `diverging: [A, C]`. This lets the optimizer's chromosome encoding directly reference path choices.

BlueBrick's open-source part libraries on GitHub (l-gauge/bluebrick-lib) contain XML files with precise stud-based port coordinates for every common LEGO-compatible track piece. These serve as validation data for computed catalog values.

## YAML schema: ruamel.yaml plus Pydantic v2 eliminates every footgun

The YAML loading stack must solve two problems: avoid YAML's notorious implicit type coercion, and validate the loaded data into immutable Python objects with computed derived fields.

**Use ruamel.yaml in safe mode** (`YAML(typ='safe')`). Unlike PyYAML (stuck on YAML 1.1), ruamel.yaml implements YAML 1.2, which eliminates the "Norway problem" (`NO` → `False`), drops the 22 boolean spellings (`yes`, `on`, `off`, `y`, `n`, etc.), requires explicit `0o` prefix for octals, and removes sexagesimal interpretation (`22:22` → `1342`). StrictYAML would also work but adds redundant validation overhead when Pydantic is already in the pipeline.

**Use Pydantic v2 with `ConfigDict(frozen=True, extra='forbid')`** for all domain models. Pydantic v2's Rust-based core validates types and constraints at construction time, producing clear error messages with field paths — essential for catalog authors debugging YAML files. The `extra='forbid'` setting catches key typos immediately. The `frozen=True` setting makes instances immutable and hashable, safe for concurrent access and usable as dict keys. Computed derived fields (arc length, port positions) use `@model_validator(mode='after')` with `object.__setattr__` to bypass the freeze during initialization.

The recommended YAML schema:

```yaml
# catalog/track_pieces.yaml
meta:
  schema_version: "1.0"
  unit: studs
  stud_mm: 8.0
  gauge: 5.0

pieces:
  - id: lego_straight_16
    type: straight
    manufacturer: lego
    length: 16.0
    ports:
      A: { x: 0.0, y: 0.0, heading: 180.0 }
      B: { x: 16.0, y: 0.0, heading: 0.0 }

  - id: lego_curve_r40_left
    type: curve
    manufacturer: lego
    radius: 40.0
    angle: 22.5
    direction: left
    ports:
      A: { x: 0.0, y: 0.0, heading: 180.0 }
      B: { x: 15.307, y: 3.045, heading: 22.5 }

  - id: lego_switch_r40_left
    type: turnout
    manufacturer: lego
    length: 32.0
    diverging_radius: 40.0
    diverging_angle: 22.5
    ports:
      A: { x: 0.0, y: 0.0, heading: 180.0 }
      B: { x: 32.0, y: 0.0, heading: 0.0 }
      C: { x: 32.693, y: 12.955, heading: 22.5 }
    paths:
      through: [A, B]
      diverging: [A, C]

  - id: 4dbrix_curve_r56_left
    type: curve
    manufacturer: 4dbrix
    radius: 56.0
    angle: 18.0
    direction: left
    ports:
      A: { x: 0.0, y: 0.0, heading: 180.0 }
      B: { x: 17.308, y: 2.741, heading: 18.0 }

  - id: crossing_90
    type: crossing
    manufacturer: lego
    crossing_angle: 90.0
    ports:
      A: { x: 0.0, y: -4.0, heading: 180.0 }
      B: { x: 0.0, y: 4.0, heading: 0.0 }
      C: { x: -4.0, y: 0.0, heading: 270.0 }
      D: { x: 4.0, y: 0.0, heading: 90.0 }
    paths:
      track1: [A, B]
      track2: [C, D]
```

## Immutable registry pattern with derived lookup indices

The catalog architecture follows the IFC type/occurrence split: the catalog holds immutable **type definitions** (loaded once at startup); the layout holds mutable **occurrences** referencing types by ID. The registry uses a module-level loader pattern — simpler than singleton classes, easily testable, and naturally immutable after initialization.

```python
# catalog/models.py
from __future__ import annotations
import math
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

class Port(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    x: float        # studs, in local frame
    y: float        # studs, positive = left
    heading: float  # degrees, CCW from x-axis

class TrackPiece(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    id: str
    type: str                          # straight, curve, turnout, crossing, wye
    manufacturer: str
    ports: dict[str, Port]             # A, B, C, D...
    paths: dict[str, list[str]] | None = None

    # Optional geometry parameters (present by type)
    length: float | None = None        # studs (straights, switches)
    radius: float | None = None        # studs (curves, switches)
    angle: float | None = None         # degrees (curves)
    direction: str | None = None       # left / right
    diverging_radius: float | None = None
    diverging_angle: float | None = None
    crossing_angle: float | None = None

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        allowed = {'straight', 'curve', 'turnout', 'crossing',
                   'wye', 'crossover', 'double_crossover'}
        if v not in allowed:
            raise ValueError(f'type must be one of {allowed}')
        return v

    @property
    def port_count(self) -> int:
        return len(self.ports)

    @property
    def arc_length(self) -> float | None:
        if self.radius and self.angle:
            return self.radius * math.radians(self.angle)
        return self.length
```

```python
# catalog/registry.py
from __future__ import annotations
from pathlib import Path
from ruamel.yaml import YAML
from .models import TrackPiece, Port

class TrackCatalog:
    """Immutable registry with O(1) derived indices."""
    __slots__ = ('_pieces', '_by_id', '_by_type',
                 '_by_radius', '_by_manufacturer')

    def __init__(self, pieces: tuple[TrackPiece, ...]):
        object.__setattr__(self, '_pieces', pieces)
        object.__setattr__(self, '_by_id',
            {p.id: p for p in pieces})
        by_type: dict[str, list] = {}
        by_radius: dict[float, list] = {}
        by_mfr: dict[str, list] = {}
        for p in pieces:
            by_type.setdefault(p.type, []).append(p)
            if p.radius is not None:
                by_radius.setdefault(p.radius, []).append(p)
            by_mfr.setdefault(p.manufacturer, []).append(p)
        object.__setattr__(self, '_by_type',
            {k: tuple(v) for k, v in by_type.items()})
        object.__setattr__(self, '_by_radius',
            {k: tuple(v) for k, v in by_radius.items()})
        object.__setattr__(self, '_by_manufacturer',
            {k: tuple(v) for k, v in by_mfr.items()})

    def __setattr__(self, *_):
        raise AttributeError("TrackCatalog is immutable")

    def __getitem__(self, piece_id: str) -> TrackPiece:
        return self._by_id[piece_id]

    def by_type(self, t: str) -> tuple[TrackPiece, ...]:
        return self._by_type.get(t, ())

    def by_radius(self, r: float) -> tuple[TrackPiece, ...]:
        return self._by_radius.get(r, ())

    @property
    def all_ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    @property
    def all_radii(self) -> frozenset[float]:
        return frozenset(self._by_radius)

    @property
    def chromosome_bounds(self) -> dict:
        """Derive GA chromosome dimension constants."""
        return {
            'piece_ids': tuple(self._by_id.keys()),
            'n_piece_types': len(self._pieces),
            'available_radii': sorted(self.all_radii),
            'n_curves': len(self.by_type('curve')),
            'n_turnouts': len(self.by_type('turnout')),
        }

# Module-level singleton
_catalog: TrackCatalog | None = None

def load_catalog(path: str | Path) -> TrackCatalog:
    global _catalog
    if _catalog is not None:
        return _catalog
    yaml = YAML(typ='safe')
    with open(path) as f:
        raw = yaml.load(f)
    pieces = []
    for p in raw['pieces']:
        p['ports'] = {k: Port(**v) for k, v in p['ports'].items()}
        pieces.append(TrackPiece.model_validate(p))
    _catalog = TrackCatalog(tuple(pieces))
    return _catalog

def get_catalog() -> TrackCatalog:
    if _catalog is None:
        raise RuntimeError("Call load_catalog() first")
    return _catalog
```

## Complete 4DBrix product catalog for the YAML seed file

The catalog YAML must include every piece the optimizer can place. Based on product listings from 4DBrix.com, OKBrickWorks, and cross-referencing with the transponderings.blog comparative analysis, the complete 4DBrix product inventory is:

**Straights:** Half straight (8 studs), quarter straight (4 studs). The full 16-stud straight is LEGO's own part — 4DBrix does not manufacture it.

**Standard curves** (all 8 studs wide): R56 at 18° (20/circle), R72 at 15° (24/circle), R88 at 11.25° (32/circle), R104 at 10° (36/circle), R120 at 9° (40/circle). R104 and R120 may be available only as 3D-printable STL files rather than pre-printed commercial products.

**R40 Modular Switch System:** The flagship product line uses modular components (split tracks, diverging tracks, couplings) that assemble into complete switches. Key configurations include the R40 Parallel Track Switch (32 studs, 16-stud spacing, 45° diverging arc), R40 Continuous Curve Switch (smooth R40 diverging route), R40 Single Crossover (48 studs long), R40 Double Crossover (48 studs long), R40 Triple Crossover, R40 Three-Way Switch, R40 Wye Switch (dual symmetric R40 diverges), R40 Curved Switch, and R40 Yard Ladder (8-stud inter-track spacing).

**R56 Switches:** R56 Continuous Curve Switch and R56 Parallel Track Switch — using R56 diverging geometry.

**Ultimate Railroader (R148):** R148 Single Crossover (96 studs long, 16-stud spacing) and R148 Double Crossover (96 studs long). The R148 radius was chosen to keep crossovers within 6 standard straight lengths (96 studs).

**Specialty:** 90° Cross Track (8×8 stud intersection), Bumper Track (track terminator), and modular switch components including special arc segments (#25: R40/14.4°, #26: R40/8.1°, #62: R56/2°, #78: R72/9.53°, #79: R56/13°).

## Conclusion

Three architectural decisions dominate the catalog design. First, **manufacturer-tagged pieces with per-piece angle data** — not assumed standard angles per radius — because 4DBrix's non-standard angles at R56, R72, R104, and R120 make manufacturer identity load-bearing for geometric correctness. Second, **port coordinates as the canonical geometry representation**, with radius/angle as supplementary parameters for human readability and validation — because ports are what the optimizer's placement algorithm directly consumes. Third, the **ruamel.yaml → Pydantic v2 → frozen registry** pipeline, which catches YAML gotchas at parse time, validates every field at construction time, freezes the result immutably for the entire process lifetime, and builds O(1) lookup indices once at startup. The chromosome module then simply calls `get_catalog().chromosome_bounds` to derive its dimension constants, with zero coupling to YAML or file I/O. Every downstream module — geometry, physics, decoder — receives the same frozen `TrackCatalog` instance, with the guarantee that no piece definition can change after load.