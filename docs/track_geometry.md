# Track Geometry Specification

## Coordinate System

### Units and Frame

- **Distance**: Studs (1 stud = 8.0 mm)
- **Angle**: Degrees (positive = counterclockwise)
- **Origin**: Arbitrary; typically first piece entry port
- **Heading 0°**: +X direction

### L-Gauge Constants

| Parameter | Studs | mm |
|-----------|-------|-----|
| Track width | 8 | 64 |
| Rail gauge (centerline) | 5 | 40 |
| Standard straight length | 16 | 128 |
| Parallel track spacing | 16 | 128 |

---

## Arc Geometry Formulas

### Curve Exit Point

For curve with radius **R** (studs), arc angle **θ** (degrees), entering at origin:

**Left turn (θ > 0):**
```
exit_x = R × sin(θ)
exit_y = R × (1 - cos(θ))
exit_heading = +θ
```

**Right turn:** Negate y and heading.

### Arc and Chord Length

```
arc_length = R × θ × (π / 180)
chord_length = 2 × R × sin(θ / 2)
```

---

## Forward Kinematics

### State and Transformation

State vector: **(x, y, θ)** — position and heading in world coordinates.

For piece with local deltas (dx_local, dy_local, dθ):

```
x_new = x + dx_local × cos(θ) - dy_local × sin(θ)
y_new = y + dx_local × sin(θ) + dy_local × cos(θ)
θ_new = θ + dθ
```

### Complete FK Lookup Table

| Piece Type                                  | dx_local | dy_local | dθ | Ports |
|---------------------------------------------|----------|----------|-----|-------|
| **Straights**                               |
| Straight_16                                 | 16.0     | 0.0 | 0° | 2 |
| Straight_24                                 | 24.0     | 0.0 | 0° | 2 |
| **Curves (4DBrix angles)**                  |
| R40_Left                                    | 15.31    | 3.05 | +22.5° | 2 |
| R40_Right                                   | 15.31    | -3.05 | -22.5° | 2 |
| **Switches (through-route FK)**             |
| R40_Switch_Left_IN                          | 16.0     | 0.0 | 0° | 3 |
| R40_Switch_Left_OUT                         | 16.0     | 0.0 | 0° | 3 |
| R40_Switch_Right_IN                         | 16.0     | 0.0 | 0° | 3 |
| R40_Switch_Right_OUT                        | 16.0     | 0.0 | 0° | 3 |
| **Crossings**                               |
| Cross_90                                    | 16.0     | 0.0 | 0° | 4 |
| **Crossovers**                              |
| Double_crossover                            | 48.0     | 0.0 | 0° | 4 |

**Note:** Switch FK values are for the **through-route** only. Diverging routes require separate path computation.

---

## Port Alignment

### Port Definition

```
Port:
  position: (x, y) in piece-local coords
  heading: degrees, piece-local
  gender: MALE | FEMALE
```

### Connection Validity

Two ports connect when:
```
|pos_A - pos_B| < 0.5 studs
|heading_A - heading_B - 180°| < 2°
gender_A ≠ gender_B  (optional)
```

### Port to World Transform

```
world_x = piece_x + local_x × cos(θ) - local_y × sin(θ)
world_y = piece_y + local_x × sin(θ) + local_y × cos(θ)
world_heading = piece_θ + local_heading
```

---

## Complete Piece Catalog

### Straights (2-Port)

| Name                  | Length | Port 0 | Port 1         |
|-----------------------|--------|--------|----------------|
| Full Straight         | 16     | (0, 0, 0°, F) | (16, 0, 0°, M) |
| 24 Stud Long Straight | 24     | (0, 0, 0°, F) | (24, 0, 0°, M) |

### Curves (2-Port)

Entry port always at (0, 0, 0°). Exit port computed from arc geometry.

| Radius | Angle | Exit Position | Exit Heading |
|--------|-------|---------------|--------------|
| R40 Left | 22.5° | (15.31, 3.05) | +22.5° |

Right curves: negate y-coordinate and heading.

### Crossings (4-Port)

**90° Diamond Crossing:**

| Port | Position | Heading |
|------|----------|---------|
| East | (8, 0) | 0° |
| West | (-8, 0) | 180° |
| North | (0, 8) | 90° |
| South | (0, -8) | 270° |

---

## Closure Detection

### Closed Loop Test

```
position_error = √((x_final - x_start)² + (y_final - y_start)²)
angle_error = |θ_final - θ_start| mod 360°

is_closed = position_error < ε_pos AND
            (angle_error < ε_angle OR |360° - angle_error| < ε_angle)
```

**Tolerances:**
- Evolutionary: ε_pos = 1.0 stud, ε_angle = 5°
- Final validation: ε_pos = 0.5 stud, ε_angle = 2°

### Angular Budget

```
Σ θ_i = 360° × k    (k = ±1 for single loop)
```

**Closure piece counts (single radius loops):**

| Radius | Angle | Pieces for 360° |
|--------|-------|-----------------|
| R40 | 22.5° | 16 |
---

## Open Paths (Bumpers)

### Validity Rules

Every open port must connect or be terminated with a bumper piece.
Open path are not allowed.

### Mixed Topologies

Main loop can close while branches terminate at bumpers:
```
Main loop: Σθ = 360° (closed)
Branch: switch → pieces → bumper (open)
```

---

## Collision and Boundary

### AABB Collision Test

```
overlap = (A.max_x > B.min_x) AND (A.min_x < B.max_x) AND
          (A.max_y > B.min_y) AND (A.min_y < B.max_y)
```

Track width offset: ±4 studs from centerline.

### Boundary Constraint

```
violation = Σ max(0, boundary_edge - piece_edge)
```

## Physics Integration

### Speed Limits by Piece Type

```
v_slide = SF × √(μ × g × R_mm)
SF = 0.8, μ = 0.30, g = 9810 mm/s²
```

| Piece Type | R (mm) | v_max (m/s) |
|------------|--------|-------------|
| R40 curves | 320 | 0.97 |
| Straights | ∞ | 1.57 |
| Switches (through) | ∞ | 1.57 |
| Switches (diverge) | 320 | 0.97 |
| Crossings | ∞ | 1.57 |
| Double crossover (cross) | 320 | 0.97 |

**Note:** Switch diverging routes use R40 geometry (same radius as standard curves).

---

**Version**: 1.0
**Source**: Piece_geometry_3ways_exploration.md, Locomotive_dynamics.md
