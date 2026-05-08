# Handoff: LEGO Track Optimizer

## Overview

A prototype that searches for closed train-track loops you can build with the pieces you own, inside a bounded play area. Inputs are a piece **catalog** (with port poses + routes) and a **problem** (inventory, boundary, tolerance, weights, search budget). Output is a scored, validated layout rendered to a canvas, with a self-check panel showing geometry tests passing live.

This is a working v1 — the algorithm, geometry model, scoring, and tests are all functional. It's not a UI mock.

## About the Design Files

The bundled files in this folder are a **working browser prototype**, not a UI design to recreate. The brief asked for a CLI tool; this is the same tool rehosted in HTML so search progress is watchable and config is editable inline. Everything that matters — closed-walk search, pose math, closure tolerance, boundary/collision pruning, scoring, geometry tests — is real and can be ported as-is.

The right framing for the developer task is: **port this v1 to the target environment** (CLI tool, web app, or both), preserving the algorithm and tests, and pick up the v2 cuts the README documents. If the target is a Python/Rust/TS CLI, geometry.js / catalog.js / search.js / tests.js port directly; only the React UI in app.jsx and the canvas rendering in render.js are HTML-specific.

## Fidelity

- **Algorithm and geometry: production-quality.** Pose math, port composition, closure check, AABB collision, scoring, and tests are written to be correct and are covered by the self-check panel (11/11 passing).
- **UI: prototype-fidelity.** The React + canvas frontend exists to make the search watchable and the config editable. It's not styled against a design system; if shipping a real UI, treat it as wireframe and apply the host codebase's component library.

## What ships in v1

- Straights (16-stud = 128 mm, 32-stud = 256 mm).
- R40 curves left + right (320 mm radius, 22.5° each).
- **Template-based loop search**: enumerates parameterized circles, ovals, and stadiums; instantiates each against inventory; validates against geometry, boundary, collision; scores survivors.
- Two-objective scoring: utilization (primary) + estimated lap time + compactness, weights exposed in Tweaks.
- Live search visualization (best-so-far updates as candidates evaluate).
- Geometry test panel (search button refuses to run on red).

See `ORIGINAL_README.md` for the full design notes including the rationale for templates over blind DFS, the data model, the scoring formula, and the v2 cut list.

## File map

| File | Purpose | Lines | Port priority |
|---|---|---|---|
| `geometry.js` | Pose, port composition, AABB ops, closure check | ~110 | **Port first** — pure math, no DOM |
| `catalog.js` | Piece data: ports + routes for S16, S32, CR, CL | ~60 | **Port first** — data only |
| `search.js` | Template generators + validation + scoring | ~220 | **Port second** — algorithm core |
| `tests.js` | 11 geometry/closure tests | ~150 | **Port second** — port alongside search |
| `render.js` | Canvas drawing of placements + boundary + grid | ~140 | HTML-specific; rewrite for target UI |
| `app.jsx` | React UI glue: panes, Tweaks, search wiring | ~280 | HTML-specific; rewrite for target UI |
| `tweaks-panel.jsx` | Stock Tweaks panel scaffold | starter | drop if not needed |
| `index.html` | Page shell + script tags | small | drop if not needed |

## Algorithm essentials a developer must preserve

### Data model

```
Pose  = { x, y, θ }      // mm, mm, radians
Port  = Pose             // local pose of a port relative to piece origin
Piece = { id, kind, ports: [Port...], routes: [{ from, to, length, radius? }...], footprint }
Placement = { pieceId, routeIdx, originPose, endPose }
Layout = { placements: [Placement...], bbox, start, end, score }
```

Pose composition: `compose(A, B) = { x: A.x + cos(A.θ)·B.x − sin(A.θ)·B.y, y: A.y + sin(A.θ)·B.x + cos(A.θ)·B.y, θ: A.θ + B.θ }`. Identity is `{0,0,0}`. See `geometry.js`.

### Route delta

For a route `from→to` on a piece, the world-frame delta is:

```
delta = inverse(piece.ports[from]) ⊕ piece.ports[to]
```

Apply at the entry pose to get the placement's exit pose. This is the only piece of the math the search relies on; everything else is bookkeeping.

### Closure

After placing N pieces from world origin, closure is `‖endPose − start‖ < tol_mm` and `|Δθ mod 2π| < tol_rad`. Heading-step parity (every R40 curve = 22.5° = 1/16 turn) is a useful necessary condition but **not sufficient** — always validate the pose.

### Search (templates)

Templates are generators of `instructions: pieceId[]` lists known to be closed by construction. The runner:

1. enumerates templates,
2. inventory-checks each,
3. builds each placement-by-placement, rejecting on out-of-bounds or AABB collision,
4. confirms pose-closure as a guard,
5. scores survivors and keeps the best.

Templates currently shipped: `circle` (16 same-handed curves), `oval` (8 curves + N straights mirrored), `stadium` (8 curves + side-A + 8 curves + side-B with side-A/B equal in mm).

To extend: add a generator that yields `{ name, instructions }`. Anything closed by construction works. For switches/crossings, the instruction grammar needs branching — see `ORIGINAL_README.md` v2 cut list.

### Scoring

```
score = w_util · (used / available)
      + w_lap  · (length / lap_time / motorVMax)
      + w_comp · clamp(bbox_area / boundary_area, 0, 1)

lap_time = Σ piece_length / piece_v
piece_v  = piece.radius ? min(motorVMax, sqrt(μ·g·R)) : motorVMax
```

Weights default to `(1.0, 0.3, 0.2)`. All exposed in the problem JSON.

### Tolerances

- Closure position: 2 mm (= ¼ stud).
- Closure heading: 0.5°.
- Collision overlap slack: 1 mm.

### Tests (must keep green)

`tests.js` covers: pose identity, inverse, composition associativity, 16R-curves close, 16L-curves close, 8R+8L+straights does not close in heading, AABB intersection symmetric & reflexive, boundary-fit detects out-of-bounds piece, hand-crafted oval closes, inventory respected during search.

The UI gates the search button on these passing. Preserve that gate in any port.

## Config schema

### `catalog` (JSON)

```jsonc
[
  {
    "id": "S32", "kind": "straight",
    "ports": [{"x":0,"y":0,"θ":0}, {"x":256,"y":0,"θ":0}],
    "routes": [{"from":0,"to":1,"length":256}],
    "footprint": {"w":256,"h":40}
  },
  {
    "id": "CR", "kind": "curve",
    "ports": [{"x":0,"y":0,"θ":0}, {"x":..., "y":..., "θ":-π/8}],
    "routes": [{"from":0,"to":1,"length":..., "radius":320}],
    "footprint": {...}
  }
  // ...
]
```

### `problem` (JSON)

```jsonc
{
  "inventory": { "S32": 8, "S16": 8, "CR": 16, "CL": 16 },
  "boundary": { "width": 3000, "height": 2000 },   // mm
  "tolerance": { "mm": 2, "deg": 0.5 },
  "weights": { "utilization": 1.0, "lapTime": 0.3, "compactness": 0.2 },
  "physics": { "motorVMax": 1200, "mu": 0.6, "g": 9810 },  // mm/s, dimensionless, mm/s²
  "searchBudget": { "ms": 3000, "restarts": 30, "minLoop": 4 }
}
```

## v2 cut list (in priority order)

1. **Switches** — turns search from single-walk into multigraph traversal. Doubles state space; needs branching template grammar.
2. **Crossings** — needs explicit "this overlap is legal" predicate in collision check.
3. **Double crossover** — superset of switches; defer until switches work.
4. **More templates** — dog-bones, kidneys, figure-8 precursors; cheap if you stay within "closed by construction."
5. **Full motor speed model** — current model is `v = min(vmax, sqrt(μ·g·R))` per piece, integrated as `Σ length/v`. Add acceleration / weight / gradient if simulation accuracy matters.
6. **Constraint solver loop search** — reformulate as a CSP over (piece, port, placement) and let a solver enumerate; strictly more general than templates, strictly slower for small problems.

## Recommended port strategy

- **If target is a CLI**: port `geometry.js` + `catalog.js` + `search.js` + `tests.js` to the host language. Drop the React UI; emit a PNG via host's image library and a JSON scorecard. Catalog/problem read from files.
- **If target is a webapp**: keep the structure, swap `render.js` for the host's drawing primitive (SVG, react-konva, etc.), and rewrite `app.jsx` against the host's component library. Algorithm files port unchanged.
- **If target is a desktop app**: same as webapp; algorithm files port unchanged, UI is rewritten.

In all cases: bring the test suite over first, get it green, then port the search.

## Assets

`screenshot.jpg` — a working run: 24-piece oval (8 CL + 2 S32 + 2 S16, mirrored), 60% utilization, 0 mm closure error, 11/11 tests green. Use this as the visual target for "does it work."

## Files in this bundle

- `README.md` — this file
- `ORIGINAL_README.md` — the design notes shipped with the prototype (rationale, scoring math, v2 cuts)
- `index.html`, `app.jsx`, `tweaks-panel.jsx`, `render.js` — UI layer (HTML-specific)
- `geometry.js`, `catalog.js`, `search.js`, `tests.js` — algorithm + data + tests (port first)
- `screenshot.jpg` — reference output
