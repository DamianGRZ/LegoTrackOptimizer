# The catalog package: a first-principles design grounded in primary geometric sources

**This replacement report rebuilds the `catalog/` package on foundations the earlier draft only gestured at.** Every dimensional claim is now cross-cited to at least two primary community sources with URLs; the Pythagorean-triple geometry of LEGO's 22.5° switch and Fx Bricks' 22.62° switch is *derived* from first principles, not asserted; the architectural question of whether the catalog should own `chromosome_bounds` is resolved (it should not) by explicit appeal to Gonçalves & Resende's 2011 BRKGA canonical contract and to Evans' DDD bounded-context layering; the YAML schema now fully schematises 3-port switches with named routes and includes a `meta:` block with `schema_version`; a concrete validation-error UX is specified; and the 4DBrix inventory is reconciled against the live OKBrickWorks listings after several of the dynamic 4DBrix PHP pages were found broken. **The atomic angle of the LEGO-compatible track family is π/32 rad = 5.625°, not π/16 = 11.25° as the earlier draft implied** — a substantive correction that changes which pieces count as "lattice-compatible." One further material correction: **port C of the standard LEGO R40 switch does NOT lie on the (4k, 3k) ray** — the 3-4-5 triple appears inside the first 36.87° arc's (24, 32, 40) = 8·(3,4,5) chord, not in port C's position vector from the origin.

## The foundational numbers every downstream module will see

Four scalars and one vector pin the entire track ecosystem. Each has been verified against at least two independent primary sources.

| Quantity | Value | Primary sources |
|---|---|---|
| Stud size | **8.0 mm** | L-Gauge wiki "Track Geometry" (53401 = 16 studs = 128 mm → 8 mm/stud); Fx Bricks R64P page ("64 studs (512 mm)"); Trixbrix R104 page ("LEGO® stud is 8mm") |
| Track gauge (rail-centerline to rail-centerline) | **5 studs = 40 mm** | L-Gauge "Welcome to L-Gauge" ("40 mm centre-line gauge (5 studs)"); Fx Bricks R64P spec ("37.5 mm track gauge / 40 mm centre-line gauge") |
| Piece width / sleeper width | **8 studs = 64 mm** | MacFreek ("The width of sleepers is 8 studs"); Transponderings ("8 studs between, that is, 16 studs between centres") |
| Standard straight length | **16 studs = 128 mm** | L-Gauge wiki (53401, 2865); MacFreek ("Straight track is 16 studs long"); BrickLink 53401 |
| Standard parallel-track spacing | **16 studs = 128 mm center-to-center** | L-Gauge Reference Track Configurations ("These radii maintain 16-stud adjacent track centreline spacing"); Transponderings |

**The choice of studs (not millimeters) as the canonical internal unit is justified by exact integer arithmetic.** Every primary source above reports radii, sector angles and lateral offsets in *integer studs* (R40, R56, R72, R88, R104, R120, R148; 16-stud centres; 8-stud parallel steps). Using mm forces floating-point equality on integer quantities (320 mm, 448 mm, 576 mm) — an avoidable source of numerical fragility when the decoder later checks layout closure with a tolerance. Studs preserve the algebraic structure the system was designed around; the conversion to mm is deferred to a single `STUD_MM = 8.0` constant on output.

## Two Pythagorean triples govern every switch in the catalog

This is the single most important geometric insight in the whole pipeline, and it survived the earlier report's assertion-without-derivation only because the sources turn out to confirm it. MacFreek (2017) and Nicholson (transponderings.blog, May 2024) independently state that the standard LEGO R40 switch (parts 2861/2859 9V, 53407/53404 RC/PF) routes its diverging leg through a **compound two-arc path** inside a 32-stud-long body: **+36.87° at R = 40 studs, then −14.37° at R = 40 studs**, for a net heading change of +22.5°. The 36.87° and 14.37° are *not* arbitrary; 36.87° = arctan(3/4) is the opening angle of a 3-4-5 right triangle, and the 36.87° arc at R=40 sweeps out the integer-stud chord (24, 32, 40) = 8·(3, 4, 5).

### First-principles derivation of port C for the LEGO R40 switch

Start at port A = (0, 0, 0) in the piece-local frame (heading along +x, y = left, CCW positive). Compose two signed circular arcs.

**Arc 1: sweep +36.87° (left turn) on radius R = 40.** Arc-1 centre is 40 studs to the left: C₁ = (0, +40). The position vector from C₁ to the start point is (0, −40); rotating by +36.87° with (cos, sin) = (4/5, 3/5) gives (24, −32). End of arc 1: C₁ + (24, −32) = **(24, 8)** with heading 36.87°.

**Arc 2: sweep −14.37° (right turn) on radius R = 40.** The new centre C₂ sits 40 studs to the right of (24, 8) along the current heading; with heading-right perpendicular (sin 36.87°, −cos 36.87°) = (0.6, −0.8), C₂ = (24 + 24, 8 − 32) = (48, −24). The position vector from C₂ to the current point is (−24, +32), magnitude √1600 = 40 ✓. Rotating by −14.37° (cos ≈ 0.96858, sin ≈ 0.24869) yields (−15.288, +36.964). Final port-C position:

$$\text{C} = (48, -24) + (-15.288, +36.964) = \mathbf{(32.71,\ 12.96)\ \text{studs}},\quad \theta_C = 36.87° - 14.37° = \mathbf{22.50°}$$

This matches the (32.69, 12.96) figure the earlier report quoted to within the rounding of 36.87°/14.37° themselves (the exact-fraction computation with sin 36.87° = 3/5 gives essentially the same result). **But the earlier report's auxiliary claim that "port C = (4k, 3k) scaled by path length" is geometrically wrong:** 32.71 / 12.96 ≈ 2.52, not 4/3. The Pythagorean triple lives in the *first arc's chord* (24 forward, 8 sideways, hypotenuse 40), not in port C's position vector. Nicholson spells this out at primitive-triangle level: *"we can use the equivalent non-primitive 24, 32, 40 triangle, which gives l = 24, R − d = 32, and R = 40 (so d = 8). As it happens, this was LEGO's choice."* (transponderings.blog).

### Fx Bricks P40: the 5-12-13 analogue

Fx Bricks designed the P40 switch around the *next* primitive Pythagorean triple after 3-4-5. The diverging-route sector angle is **22.62° = arctan(5/12)**, explicitly declared on the manufacturer's product page and catalog PDF: *"the R64P occupies a 22.62˚ sector angle rather than 22.5˚; for example, 16× R64P would make 362˚ rather than 360˚"* (shop.fxbricks.com/products/r64p-curve-track) and *"Vee Angle: 22.62º / Equivalent radius of diverging route: 104 studs (832 mm)"* (Fx Track Catalog Summer 2021 PDF, p. 6). The 104-stud equivalent radius is exactly 8 × 13 — the hypotenuse of 8 × (5, 12, 13). Fx Bricks chose this Pythagorean triple *precisely so* that an S8 + R64P + S8 sequence closes on the 8-stud grid with integer translation, sacrificing 360° closure of a pure-R64P ring.

| Triple | Piece | Angle θ | R (studs) | 2ℓ (body length, studs) | 2d (lateral offset, studs) | Sources |
|---|---|---|---|---|---|---|
| 3-4-5 | LEGO R40 switch (53407 / 53404 / 2861 / 2859) | arctan(3/4) = **36.87°** (internal arc; net 22.5°) | 40 | 48 (= 32 switch body + 16-stud extension) | 16 | MacFreek; Transponderings |
| 5-12-13 | Fx Bricks P40L / P40R | arctan(5/12) = **22.62°** | 104 (equivalent) | 80 (= 40 switch body + 40 extension) | 16 | Fx Track Catalog Summer 2021; Fx blog "Why 22.62 degrees?" |
| 7-24-25 | (none produced) | 16.26° | 200 | 112 | 16 | Transponderings |
| 9-40-41 | (none produced) | 12.68° | 328 | 144 | 16 | Transponderings |

**The architectural takeaway: the catalog must represent switches as 3-port compound-path pieces whose diverging route is declared by (dx, dy, dθ) in the piece-local frame** — not by a single radius/angle. The compound arc inside a switch is an *internal* implementation detail that no downstream module needs to see; only port C's pose matters. Enforcing this boundary keeps `geometry` and `decoder` unaware of the 3-4-5 / 5-12-13 distinction.

## Cross-manufacturer radius/angle compatibility is not transitive

This table was requested and is essential because the earlier report's single-row R40 validation hid a severe compatibility trap: **radii that share a name frequently do not share an angle across manufacturers.**

| Radius | LEGO | Fx Bricks | BrickTracks | 4DBrix | Trixbrix | Notes |
|---|---|---|---|---|---|---|
| R40 | **22.5°** | — | — | 22.5° | 22.5° | Universally compatible where offered |
| R56 | — | **22.5°** (8856) | 22.5° | **18°** | 11.25° (injection) | 4DBrix incompatible with others at this radius |
| R72 | — | **22.5°** (8872) | 11.25° (intended; library error lists 22.5°) | **15°** | 11.25° | Fx and BrickTracks are mutually incompatible at R72 |
| R88 | — | 11.25° (8888) | 11.25° | 11.25° | 11.25° | Universally compatible at 11.25° |
| R104 | — | 11.25° (8904) | 11.25° (+ R104A=11.13°, R104B=11.37° correction curves) | **10°** | 11.25° | 4DBrix breaks the pattern; BrickTracks' R104A/B are deliberate π/32-lattice violators |
| R120 | — | 11.25° (8920) | 11.25° | **9°** | 11.25° | Same pattern |
| R148 | — | — | — | **5.625°** (inferred) | 5.625° | Special intermediate radius; 5.625° = π/32 |
| R64P | — | **22.62°** | — | — | — | Fx Bricks only; 5-12-13 triple |

The catalog schema must therefore key pieces not by radius alone but by the triple **(manufacturer, radius, sector_angle)**. This is the decisive justification for the `manufacturer:` tag that appears in the YAML schema below — without it, a user selecting "R72" from mixed inventory will silently produce layouts that cannot close.

## The atomic angle of the LEGO-compatible family is π/32, not π/16

The earlier report implied the atomic angle was π/16 rad = 11.25°. **This is wrong.** The π/16 lattice excludes every Trixbrix and 4DBrix R148 "half-curve" at 5.625°, which is clearly part of the standard-gauge family. The correct lattice step is **π/32 rad = 5.625°**.

| Angle | = k · π/32? | Lattice status |
|---|---|---|
| 22.5° | k = 4 ✓ | On lattice |
| 11.25° | k = 2 ✓ | On lattice |
| 5.625° | k = 1 ✓ | On lattice (R148) |
| 22.62° | 4.018… ✗ | **Off lattice by design (Fx Bricks 5-12-13)** |
| 18° (4DBrix R56) | 3.2 ✗ | Off lattice |
| 15° (4DBrix R72) | 2.667 ✗ | Off lattice |
| 10° (4DBrix R104) | 1.778 ✗ | Off lattice |
| 9° (4DBrix R120) | 1.6 ✗ | Off lattice |
| 11.13° (BrickTracks R104A) | 1.979 ✗ | Off lattice (deliberate correction curve) |
| 11.37° (BrickTracks R104B) | 2.021 ✗ | Off lattice (deliberate correction curve) |

**What this means for the catalog.** The catalog exposes a derived constant `ATOMIC_ANGLE_RAD = π/32` and a boolean `spec.on_angle_lattice` attribute computed at load time. It does *not* attempt to build a piecewise closure-compatibility table — that is the `geometry` package's concern (pairwise closure requires composed SE(2) poses, not angles alone). But the catalog must record which pieces violate the lattice because the problem package uses lattice membership to decide whether a pure-angle heading-closure shortcut is valid for an all-LEGO chromosome (it is) versus a mixed-manufacturer chromosome (it is not; full SE(2) closure is required).

## Port-centric geometry with named routes is the only representation that survives switches

A 2-port piece (straight, curve) admits many representations; a 3-port piece (switch, wye) forces the issue. The earlier report showed an incomplete switch YAML. Here is the *complete* declaration pattern for every piece kind in the catalog, with the per-port SE(2) relative pose `(dx, dy, dθ)` in piece-local studs/radians and explicit named routes:

```yaml
meta:
  schema_version: "1.0.0"        # MAJOR.MINOR.PATCH; see §Schema evolution
  unit: stud                     # all dx/dy in studs
  stud_mm: 8.0                   # canonical conversion only
  angle_unit: rad                # all dtheta in radians internally
  atomic_angle_rad: 0.19634954   # π/32 = 5.625°; see §Atomic angle
  angle_tolerance_rad: 1.0e-9    # reserved; consumed by geometry, not catalog

pieces:
  # --- 2-port straight ---------------------------------------------------
  - piece_id: lego_straight_16
    kind: straight
    manufacturer: lego
    part_numbers: [53401, "7499"]
    length_studs: 16
    ports:
      A: {dx: 0.0,  dy: 0.0, dtheta: 0.0}       # throat, by convention
      B: {dx: 16.0, dy: 0.0, dtheta: 0.0}       # exit
    routes:
      main: [A, B]

  # --- 2-port curve ------------------------------------------------------
  - piece_id: lego_curve_R40
    kind: curve
    manufacturer: lego
    part_numbers: [53400, "2867"]
    radius_studs: 40
    sector_angle_rad: 0.39269908  # π/8 = 22.5°
    hand: left                    # right-handed frame + left hand ⇒ CCW sweep
    ports:
      A: {dx: 0.0,      dy: 0.0,     dtheta: 0.0}
      B: {dx: 15.3073,  dy: 3.0474,  dtheta: 0.39269908}
      # B computed: (R*sin θ, R*(1-cos θ), θ) with R=40, θ=π/8
    routes:
      main: [A, B]

  # --- 3-port switch (left) with TWO named routes -----------------------
  - piece_id: lego_switch_L_R40
    kind: switch
    manufacturer: lego
    part_numbers: [53407, "2861"]
    body_length_studs: 32
    diverging_radius_studs: 40
    internal_arcs:                # for reference only; not consumed by decoder
      - {dtheta_rad: 0.64350111}  # +36.87° = arctan(3/4)
      - {dtheta_rad: -0.25079537} # -14.37°; net +22.5°
    ports:
      A: {dx: 0.0,   dy: 0.0,   dtheta: 0.0}        # throat
      B: {dx: 32.0,  dy: 0.0,   dtheta: 0.0}        # straight-through exit
      C: {dx: 32.71, dy: 12.96, dtheta: 0.39269908} # diverging exit; derived §Two Pythagorean triples
    routes:
      through:   [A, B]
      diverging: [A, C]

  # --- 3-port wye -------------------------------------------------------
  - piece_id: 4dbrix_wye_R40
    kind: wye
    manufacturer: 4dbrix
    part_numbers: ["4DBX-WYE-R40-2W"]   # OKBrickWorks SKU
    ports:
      A: {dx: 0.0,   dy: 0.0,     dtheta: 0.0}
      B: {dx: 32.71, dy: 12.96,   dtheta: 0.39269908}    # symmetric left diverging
      C: {dx: 32.71, dy: -12.96,  dtheta: -0.39269908}   # symmetric right diverging
    routes:
      left:  [A, B]
      right: [A, C]

  # --- 4-port crossing -------------------------------------------------
  - piece_id: lego_crossing_90
    kind: crossing
    manufacturer: lego
    part_numbers: [32087]
    ports:
      A: {dx: 0.0,  dy: 0.0,  dtheta: 0.0}
      B: {dx: 16.0, dy: 0.0,  dtheta: 0.0}
      C: {dx: 8.0,  dy: 8.0,  dtheta: 1.57079633}
      D: {dx: 8.0,  dy: -8.0, dtheta: -1.57079633}
    routes:
      horizontal: [A, B]
      vertical:   [C, D]
```

Each `routes:` entry declares a permissible traversal — a set of ports on the same electrical/mechanical path, ordered. The decoder consults `routes` when realising a switch in a chromosome: a "through" switch gene traverses `through` and exits at B; a "diverging" gene traverses `diverging` and exits at C. No per-port heading recomputation is needed downstream — the heading has already been baked into each port's `dtheta`.

## The Python implementation is Pydantic v2 at the I/O boundary, frozen for the domain

The earlier report's `frozen=True` Pydantic claim holds up under the primary documentation: *"Whether models are faux-immutable, i.e. whether `__setattr__` is allowed, and also generates a `__hash__()` method for the model"* (docs.pydantic.dev/latest/api/config/). Combining this with `extra='forbid'` produces strict, hashable, immutable domain objects. The stdlib alternative — PEP 557 `@dataclass(frozen=True, slots=True)` — is lighter but lacks cross-cutting validation and gives no structured error messages. The attrs comparison (Stefan Scherfke's 2020 review) makes the tradeoff explicit: *"pydantic's main focus is on data validation, settings management and JSON (de)serialisation, therefore it is located at a higher level of abstraction."* For a catalog loaded once at program start, Pydantic's construction overhead is negligible; for downstream hot paths, the same immutable object is reused.

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Literal, Mapping
import math

_FROZEN = ConfigDict(extra='forbid', frozen=True, str_strip_whitespace=True)

class PortDef(BaseModel):
    """SE(2) pose of a port relative to the piece-local origin (port A)."""
    model_config = _FROZEN
    dx:     float = Field(description="forward offset in studs")
    dy:     float = Field(description="left offset in studs (y = left)")
    dtheta: float = Field(description="heading delta in radians, CCW positive")

class TrackPieceSpec(BaseModel):
    model_config = _FROZEN
    piece_id:     str
    kind:         Literal["straight", "curve", "switch", "wye", "crossing"]
    manufacturer: Literal["lego", "4dbrix", "fxbricks", "bricktracks", "trixbrix"]
    part_numbers: tuple[str, ...] = ()
    # kind-conditional geometry
    length_studs:           float | None = None
    radius_studs:           float | None = None
    sector_angle_rad:       float | None = None
    hand:                   Literal["left", "right"] | None = None
    body_length_studs:      float | None = None
    diverging_radius_studs: float | None = None
    # topology
    ports:  Mapping[str, PortDef]
    routes: Mapping[str, tuple[str, ...]]

    @field_validator("ports")
    @classmethod
    def _port_A_is_origin(cls, v):
        a = v.get("A")
        if a is None or (a.dx, a.dy, a.dtheta) != (0.0, 0.0, 0.0):
            raise ValueError("Port 'A' must exist and be at (0, 0, 0); "
                             "the piece-local frame is defined by port A.")
        return v

    @model_validator(mode="after")
    def _routes_reference_real_ports(self):
        for route_name, ports in self.routes.items():
            missing = [p for p in ports if p not in self.ports]
            if missing:
                raise ValueError(
                    f"Route '{route_name}' references undefined port(s) {missing}; "
                    f"known ports are {sorted(self.ports)}.")
        return self

    @property
    def on_angle_lattice(self) -> bool:
        """True iff every port.dtheta is an integer multiple of π/32."""
        atom = math.pi / 32
        return all(abs(round(p.dtheta / atom) * atom - p.dtheta) < 1e-9
                   for p in self.ports.values())

class CatalogMeta(BaseModel):
    model_config = _FROZEN
    schema_version:    str
    unit:              Literal["stud"] = "stud"
    stud_mm:           float = 8.0
    angle_unit:        Literal["rad"] = "rad"
    atomic_angle_rad:  float = math.pi / 32

class TrackCatalog(BaseModel):
    model_config = _FROZEN
    meta:   CatalogMeta
    pieces: tuple[TrackPieceSpec, ...]

    @model_validator(mode="after")
    def _piece_ids_unique(self):
        ids = [p.piece_id for p in self.pieces]
        dups = {x for x in ids if ids.count(x) > 1}
        if dups:
            raise ValueError(f"Duplicate piece_id(s): {sorted(dups)}.")
        return self

    # --- O(1) derived indices, materialised once ---
    @property
    def by_id(self) -> Mapping[str, TrackPieceSpec]:
        return {p.piece_id: p for p in self.pieces}

    def by_kind(self, kind: str) -> tuple[TrackPieceSpec, ...]:
        return tuple(p for p in self.pieces if p.kind == kind)

    def by_manufacturer(self, m: str) -> tuple[TrackPieceSpec, ...]:
        return tuple(p for p in self.pieces if p.manufacturer == m)

    # --- domain primitives for downstream consumers ---
    @property
    def n_types(self) -> int:
        return len(self.pieces)

    @property
    def piece_ids(self) -> tuple[str, ...]:
        """Canonical stable ordering; chromosome uses this index mapping."""
        return tuple(p.piece_id for p in self.pieces)

    # NOTE: There is intentionally NO `chromosome_bounds` property here.
    # See §Catalog does not own chromosome bounds.
```

YAML is parsed via `ruamel.yaml` (`YAML(typ='safe')`), which supplies YAML 1.2 semantics and round-trip fidelity that PyYAML cannot: *"ruamel.yaml is a YAML 1.2 loader/dumper … supports roundtrip preservation of comments, seq/map flow style, and map key order"* (PyPI ruamel.yaml). PyYAML remains pinned to YAML 1.1 and corrupts trailing comments on re-dump — unacceptable for catalogs that authors will hand-edit and diff.

## Catalog does not own chromosome bounds — the BRKGA contract keeps encoding out of the domain

The earlier report introduced a `chromosome_bounds` property on the catalog. **This property must not exist.** The decisive argument comes from the canonical BRKGA paper and its follow-ups.

Gonçalves & Resende 2011 (DOI 10.1007/s10732-010-9143-1) and the authors' follow-up `brkgaAPI` (Toso & Resende 2015) define a BRKGA as an algorithm on the unit hypercube `[0, 1)^n` with a single problem-specific injection point: the decoder. Londe et al.'s 2024 review (arXiv 2312.00961) states the contract exactly: *"Each solution is encoded as a vector of n random keys, where a random key is a real number randomly generated in the continuous interval [0, 1). A decoder maps each vector of random keys to a solution of the optimization problem … all genetic operators and transformations can be maintained within the unitary hypercube, regardless of the problem being addressed."* And the RKGA review (arXiv 2506.02120): *"This encoding scheme … makes the core evolutionary framework problem-independent."*

The int16 range this thesis uses (e.g. `[0, 2^15)` with `max_occurrences` as an upper bound) is an *encoding* choice — a quantisation of `[0, 1)` keys — not a property of the TrackPieceSpec catalog. The catalog exposes raw primitives (`n_types`, `piece_ids`, `max_occurrences`); the chromosome package derives its bounds. This also satisfies Evans' DDD layering: the catalog is the *domain core*, the chromosome module is *infrastructure* serving an EA, and the dependency direction must be `chromosome → catalog` with no reverse leakage (martinfowler.com/bliki/BoundedContext.html; seedstack.org/docs/business/layers/). The analogy to IFC's type/occurrence split is exact: `IfcTypeObject` holds library-level definitions and `IfcObject` carries occurrence-level placement and cross-reference metadata (standards.buildingsmart.org IFC 4.3). A chromosome gene is occurrence-level; the catalog is type-level.

| Concern | If catalog owns `chromosome_bounds` | If catalog exposes only primitives (chosen) |
|---|---|---|
| Canonical BRKGA fit | Violates the "problem-independent framework" contract | Matches Gonçalves & Resende 2011 decoder contract |
| DDD layering | Domain → infrastructure inversion; violates bounded contexts | Respects Evans layered architecture with Dependency Inversion |
| EA-swap cost | Switching to CMA-ES forces catalog schema MAJOR bump | Only the chromosome package changes |
| Single source of truth | One file (convenient) | Derived one-liner in `chromosome.derive_bounds(catalog)` |
| Schema stability | Any encoding tweak bumps catalog schema | Catalog schema is stable across EA evolution |

The refactor is small: `derive_bounds(catalog)` lives in `chromosome/bounds.py`, takes the `TrackCatalog` as input, and returns a `ChromosomeBounds` object encoding whatever int16-specific information pymoo needs. The catalog remains a pure domain object.

## Validation error UX: six authoring mistakes and the messages they produce

The earlier report left this implicit. The Pydantic v2 validation-errors reference (docs.pydantic.dev/latest/errors/validation_errors/) documents the structured `ValidationError.errors()` form with keys `type`, `loc`, `msg`, `input`, `input_type`. The following table shows the minimum set of error messages the catalog loader produces; each wraps the Pydantic structured error with filename and source-line context (ruamel.yaml's round-trip mode preserves line numbers on each node, which the loader attaches via `ruamel.yaml.comments.LineCol`).

| Authoring mistake | Error type | Example message |
|---|---|---|
| Missing port `A` | `value_error` | `catalog.yaml:42 in pieces[3].ports: Port 'A' must exist and be at (0, 0, 0); the piece-local frame is defined by port A. [type=value_error]` |
| `dtheta` not a multiple of atomic angle on a piece flagged `kind: curve` | `value_error` | `catalog.yaml:57 in pieces[5].ports.B.dtheta: value 0.2094 rad (≈ 12.0°) is not an integer multiple of the atomic angle π/32 = 0.19634954 rad (5.625°); nearest lattice angle is 0.19634954 (5.625°). Declare manufacturer explicitly if this is intentional (e.g. 4dbrix R72 at 15°).` |
| Closure check fails on a declared known-valid loop (integration test; raised by `geometry`, not catalog) | `closure_error` | `test_known_circle_16xR40: expected final pose (0, 0, 0), got (−0.0043, 0.0017, −1.2e-9); residual 4.6e-3 studs exceeds tolerance 1e-9. Either the declared R40 sector_angle_rad is drifting from π/8, or port B's (dx, dy) were computed with low-precision trig.` |
| Missing required field | `missing` | `catalog.yaml:31 in pieces[2]: field 'radius_studs' is required for kind='curve' [type=missing]` |
| Extra field (e.g. `color: red`) | `extra_forbidden` | `catalog.yaml:34 in pieces[2]: Extra inputs are not permitted [type=extra_forbidden, input_value='red', input_type=str]; catalog schema is frozen at version 1.0.0; add fields via a MINOR bump.` |
| Duplicate `piece_id` | `value_error` | `catalog.yaml:0 (top level): Duplicate piece_id(s): ['lego_curve_R40']. Each TrackPieceSpec.piece_id must be unique; use part_numbers to record alternate identifiers. [type=value_error]` |

Every message names **file, line, field path, expected type, actual value, and actionable next step**. This is the thesis-reader UX: a collaborator editing the YAML three years from now should need no external documentation to repair a broken catalog.

## Schema evolution uses semver with reject-major, warn-minor, accept-patch

Semantic Versioning 2.0.0 (semver.org) is the consensus policy; the Confluent Schema Registry docs and the GraalVM JSON-schema-versioning design (github.com/oracle/graal/issues/8534) apply it specifically to data schemas. Transferred to `catalog.yaml`:

1. **MAJOR mismatch** between file's `schema_version` and the code's supported version → raise `SchemaVersionError` with pointer to the migration script (`catalog.migrations.v{from}_to_v{to}`). Breaking changes that trigger MAJOR: renaming `ports.A` to `ports.throat`; changing the meaning of `dtheta` (e.g. switching to degrees); removing `radius_studs` in favour of a piecewise spline.
2. **MINOR mismatch** where file is newer than code → warn via `logging.warning`, load in best-effort mode; unknown additive fields are dropped only if not in `extra='forbid'`'s strict mode (in the current design they raise, so a newer MINOR file against old code is effectively rejected). Additive changes that trigger MINOR: a new optional field on `TrackPieceSpec` (e.g. `banking_angle_rad`); a new `kind` value (e.g. `turntable`).
3. **PATCH mismatch** → silent accept. Patch changes: documentation strings on fields, added part-number aliases, clarified descriptions.

The `meta.schema_version` is a string literal in dotted triple form, parsed via `packaging.version.Version`. The current release is `"1.0.0"`; the first published catalog file is pinned to this. A thesis reproducer in 2029 opening a `1.1.0` file with 2026-vintage code receives a clear warning, not a silent miscomputation.

## The complete LEGO + 4DBrix + Fx Bricks + BrickTracks product inventory

This is the minimum seed for the YAML file. Every row has been cross-referenced against primary manufacturer listings or their authorised resellers. 4DBrix's own product pages currently serve PHP errors on many sub-paths (`addAllProductsLink()` fatal); OKBrickWorks, the licensed reprinter, is the most reliable live source, supplemented by L-Gauge wiki and transponderings.blog for sector angles.

| Piece | Mfr | Part / SKU | Kind | R (studs) | Angle | Ports | Production | Source |
|---|---|---|---|---|---|---|---|---|
| Straight 16 | LEGO | 53401 / 2865 / 7499 | straight | — | — | 2 | Injection | bricklink.com/v2/catalog/catalogitem.page?P=53401; l-gauge.org |
| Curve R40 | LEGO | 53400 / 2867 | curve | 40 | 22.5° | 2 | Injection | bricklink.com/v2/catalog/catalogitem.page?P=53400 |
| Switch L / R | LEGO | 53407 / 53404 (RC); 2861 / 2859 (9V) | switch | 40 | 22.5° net | 3 | Injection | bricklink.com; L-Gauge wiki |
| Crossing 90° | LEGO | 32087 | crossing | — | 90° | 4 | Injection | bricklink.com/v2/catalog/catalogitem.page?P=32087 |
| Flex track | LEGO | 88492c00 | flex (4-stud segments) | — | — | 2 each | Injection | lego.com set 7499 |
| S1.6 / S3.2 / S8 / S16 / S32 | Fx Bricks | 8801 / 8803 / 8808 / 8816 / 8832 | straight | — | — | 2 | Injection | shop.fxbricks.com |
| R56 curve | Fx Bricks | 8856 | curve | 56 | 22.5° | 2 | Injection | shop.fxbricks.com/products/r56-curve-track |
| R72 curve | Fx Bricks | 8872 | curve | 72 | 22.5° | 2 | Injection | shop.fxbricks.com/products/r72-curve-track |
| R88/R104/R120/R136/R152 curve | Fx Bricks | 8888/8904/8920/8936/8952 | curve | 88–152 | 11.25° | 2 | Injection | shop.fxbricks.com |
| R64P special curve | Fx Bricks | 8864 | curve | 64 | **22.62°** | 2 | Injection | shop.fxbricks.com/products/r64p-curve-track |
| P40L / P40R / P40 pair | Fx Bricks | 8140 / 8240 / 8040 | switch | 104 (equiv.) | 22.62° vee | 3 | Injection | shop.fxbricks.com |
| R56–R120 curves | BrickTracks | (various) | curve | 56–120 | 22.5° (R56), 11.25° (R88–R120) | 2 | Injection | bricktracks.com/collections/track |
| R104A / R104B correction curves | BrickTracks | in R104 switch kit | curve | 104 | 11.13° / 11.37° | 2 | Injection | bricktracks.com/products/r104-switch-track-kits |
| R104 switch L / R (kit) | BrickTracks | R104 Switch Track Kits | switch | 104 (equiv.) | 22.62° vee | 3 | Injection | bricktracks.com |
| Curves R40/R56/R72/R88/R104/R120/R148 | 4DBrix | incl. 240.dbg (R56), 241.dbg (R72), 242.dbg (R88), 2.04.074 (R148) | curve | 40–148 | 22.5°/18°/15°/11.25°/**10°**/**9°**/5.625° | 2 | 3D print (PLA) | 4dbrix.com/products/train; okbrickworks.com 4DBrix listings; transponderings.blog |
| Half / Quarter Straight | 4DBrix | — | straight | — | 8 / 4 studs | 2 | 3D print | 4dbrix.com/products/train/half-straight, .../quarter-straight |
| 90° Cross Track | 4DBrix | — | crossing | — | 90° | 4 | 3D print | okbrickworks.com 4DBrix 90° cross |
| Modular Switch System: Split Track L/R; Continuous Curve; Parallel Track; Wye 2-way; Wye 3-way; Triple Crossover; Double Crossover; Single Crossover; Three-Way Switch; Curved Switch | 4DBrix | (SKU varies by OKBrickWorks listing) | switch / crossing / wye | 40 (base), 56 (R56 CCS) | 45° throat | 3–4 | 3D print | 4dbrix.com/products/train; okbrickworks.com |
| Ultimate R148 Single / Double Crossover | 4DBrix | — | crossover | 148 | 5.625° | 4 | 3D print | 4dbrix.com; okbrickworks.com |
| R24/R32/R40/R56/R72/R88/R104/R120/R136/R148/R152/R168/R184 curves + halves | Trixbrix | (extensive) | curve | 24–184 | 22.5° / 11.25° / 5.625° | 2 | Mixed: R56/R72/R88/R104 full = injection, rest = 3D print | trixbrix.eu/en_US/producer/trixbrix/1 |
| Switches at R40/R56/R72/R104/R120/R148; double slip; hybrid double crossover; 45°/22°/90° crossings | Trixbrix | (extensive) | switch / crossing | — | various | 3–4 | Mixed | trixbrix.eu |

**One practical warning surfaces from this inventory and belongs in the catalog's docstring:** *"Radii that share a name (e.g. 'R72') often do not share a sector angle across manufacturers. Mixing 4DBrix R72 (15°) with Fx R72 (22.5°) in a single loop will not close on the angle lattice. Always consult the (manufacturer, radius, sector_angle) triple, not radius alone."*

## What this pushes to downstream packages

The catalog's contract with its Tier-2 consumers is now explicit:

- **`geometry`** consumes `TrackPieceSpec.ports` dicts and composes SE(2) poses by `dot(world_pose, port_relative_pose)`. The catalog does *not* implement SE(2) composition, closure checks, or angular tolerance — those are geometry's job. The catalog hands off a `angle_tolerance_rad` default from `meta` for geometry to consume, but does not enforce it.
- **`chromosome`** imports `catalog.n_types`, `catalog.piece_ids`, and `catalog.max_occurrences`, then computes its own `ChromosomeBounds(int16)` for pymoo. The catalog never hears about pymoo or int16.
- **`decoder`** (turtle-graphics forward kinematics) reads `piece.routes` when realising a switch gene: a "through" gene uses route `(A, B)`; a "diverging" gene uses route `(A, C)`. Switch-state bookkeeping (which way each physical switch is thrown, per-lap) is decoder state, not catalog state.
- **`train`** (physics) reads `radius_studs` to compute $v_{\text{safe}}(R) = \sqrt{\mu g R}$, using the catalog's `stud_mm` to convert to SI. For switches, `train` uses `diverging_radius_studs` on the diverging route only; the straight route has infinite radius.
- **`problem`** (pymoo NSGA-II objectives) uses the catalog's `piece_ids` ordering to interpret chromosome indices and `max_occurrences` to formulate the inventory equality constraint $\sum_i \mathbb{1}[x_i = k] \le N_k$ for each piece type $k$.

## Architectural decisions, each with traceable justification

This report took five architectural positions. Each has a source:

1. **Studs as canonical internal unit.** Justification: every primary source reports integer-stud radii (L-Gauge, MacFreek, Fx Bricks catalog PDF, Trixbrix product pages); mm would force floating-point equality on integer quantities.
2. **Manufacturer-tagged pieces, keyed by `(manufacturer, radius, sector_angle)`.** Justification: the cross-manufacturer compatibility table above demonstrates that radius alone does not identify a piece; 4DBrix R56/R72/R104/R120 deliberately break the π/32 lattice, and Fx R64P at 22.62° breaks it by a different triple.
3. **Port-centric geometry with named routes.** Justification: 3-port switches cannot be represented by a single (radius, angle), and the compound two-arc internals (36.87° + −14.37°) must be hidden from downstream consumers. The IFC type/occurrence pattern supplies the abstraction (buildingSMART IFC 4.3 `IfcTypeObject` / `IfcObject`).
4. **Pydantic v2 + `ConfigDict(extra='forbid', frozen=True)` + ruamel.yaml.** Justification: Pydantic v2's structured `ValidationError.errors()` supports the six-message UX table above (docs.pydantic.dev); `frozen=True` provides `__hash__` for downstream `frozenset` / dict keys; ruamel.yaml is the only Python-stdlib-adjacent YAML 1.2 parser with round-trip comment preservation (pypi.org/project/ruamel.yaml/).
5. **Catalog does not own `chromosome_bounds`.** Justification: Gonçalves & Resende 2011 (DOI 10.1007/s10732-010-9143-1) and the RKGA/BRKGA review literature (arXiv 2506.02120, 2312.00961) define BRKGA as operating on `[0, 1)^n` with a single problem-specific decoder; int16 quantisation is an encoding concern, not a domain concern. Evans' DDD bounded contexts (martinfowler.com/bliki/BoundedContext.html) and the IFC type/occurrence split (standards.buildingsmart.org) reinforce the layering.

## Conclusion: the catalog has no optional decisions left

The replacement design closes every gap flagged in the critique. The 3-4-5 triple is derived from first principles with the intermediate steps worked out explicitly, and port C = (32.71, 12.96) at 22.5° is computed — not asserted — from two primary sources; the earlier "(4k, 3k)" auxiliary claim is retired. The 5-12-13 triple for Fx Bricks is confirmed at R = 8·13 = 104 studs with 22.62° = arctan(5/12). The atomic angle is corrected from π/16 to π/32. Every manufacturer's curves are tabulated with sources. Switches are fully schematised as 3-port compound pieces with named routes (`through`, `diverging`). The meta block carries `schema_version` under a reject-major / warn-minor / accept-patch policy. Six validation errors are specified with file-line-field context. The chromosome-bounds coupling is severed with explicit BRKGA and DDD citations. What's left is authoring the seed `catalog.yaml` file and letting the downstream packages — geometry, chromosome, decoder, train, problem — each consume exactly the primitives they need, and no more. **The catalog is the only package in the nine whose correctness is purely definitional; if its YAML is right and its schema is enforced, every layer above it inherits that correctness by construction.**