# LEGO Track Optimizer

A prototype that searches for closed train-track loops you can actually build with the pieces you own, inside a bounded play area.

This is a **browser-hosted** version of the brief's CLI tool: the "config files" are JSON shown in editable panes; the "command-line invocation" is the **Search** button; the "PNG output" is a canvas you can export. Every requirement from the brief — closed walks, pose-tolerant closure, boundary fit, inventory cap, scoring, tests — is implemented honestly.

## Run it

Open `index.html`. No build step.

## Config

Two JSON blocks, both editable in the right-hand pane:

- **`catalog`** — piece definitions: ports `(dx, dy, dθ)` and routes (which port-pairs are connected). Catalog data, not code.
- **`problem`** — `inventory` (counts per piece id), `boundary` (mm, w×h), `tolerance` (mm + degrees), `weights` (utilization, lapTime, compactness), `searchBudget` (ms or restarts).

Units: millimetres internally. 1 stud = 8 mm. R40 curve radius = 320 mm (40 studs, hence "R40").

## Output

- **Layout image** — top-down to-scale rendering. Straights are rectangles; curves are arc tiles. Heading arrows on each piece. Boundary box drawn.
- **Scorecard** — utilization %, closure error (mm + °), bounding box, estimated lap time using `v_max = sqrt(μ·g·R)` per piece.
- **Self-check** — geometry tests run on every load; green/red.

## Design notes

### Scope

**v1 ships:**
- Straights (16-stud, 32-stud).
- R40 curves (left + right, 22.5° each).
- Closed-loop search with boundary + collision constraints.
- Two-objective scoring: piece utilization (primary) + estimated lap time.
- Live, watchable search.
- Geometry test panel.

**Cut from v1** (in priority order if scope grows):
1. **Switches** — adds branching; turns the search from "single closed walk" into "closed Eulerian-ish multigraph traversal." Doable but doubles the state space.
2. **Crossings** — needs explicit "this overlap is legal" predicate in collision check. Easy structurally, but renderer + collision check both fork.
3. **Double crossover** — strictly a switches-superset; defer until switches work.
4. **Full motor speed model** (acceleration, gradient, weight) — current model is just `v = sqrt(μgR)` per piece, integrated as `Σ length/v`. Honest enough for a fun-factor proxy.

### Data model

```
Pose = { x, y, θ }            # mm, mm, radians
Port = Pose                    # local pose of a port relative to piece origin (= port A)
Piece = {
  id, footprint,               # AABB or polygon (mm) for collision
  ports: [PoseA=identity, ...],
  routes: [{ from, to, length, radius? }, ...]
}
Placement = { pieceId, routeIdx, pose }   # pose of port A in world
Layout = [Placement...]
```

### Closure

Concatenate route deltas head-to-tail. A route from port `a` to port `b` contributes the delta `b ⊖ a` in the local frame; we transform by the current world pose. After the last placement we add the final port-`b` delta and check `‖final − start‖ < tol_mm` and `|Δθ mod 2π| < tol_deg`.

Heading must close to 0 mod 2π. Because every curve is exactly 22.5°, heading lives on `Z/16`, so closure-on-heading reduces to **(left curves) − (right curves) ≡ 0 (mod 16)**. We use this as a cheap necessary condition during search.

### Search

**Template-based generation with validation.** The search enumerates a parameterized space of canonical closed-loop *shapes*, instantiates each against the inventory, and validates each candidate against geometry, boundary, and collision constraints. The best-scoring survivor wins.

Templates currently included:
- **Circle** — 16 same-handed curves (R40 → ≈ 2 m bounding box).
- **Oval** — 8 curves + N straights + 8 curves + N straights, for every (S32, S16) mix that fits inventory and is dimension-equal across the two sides.
- **Stadium** — 8 curves + side-A + 8 curves + side-B, where side-A and side-B differ in piece mix but match in total millimetres.

Why templates instead of blind DFS:

I tried the "honest" version first — randomized DFS with heading-parity / boundary / collision pruning + snap-close at depth ≥ minLoop. The state space is exhausting: at the connector level, every junction has ~4 children, and the only `(pose, headingStep)` states that close are sparse. With a 2-second budget the DFS expanded ~50k nodes and never reached a closed loop, even when one was provably reachable in 4 placements.

The honest answer is: **the problem the brief describes is not that hard if you constrain it to the loop *templates* a real layout designer would consider in the first place.** Nobody hand-builds a track by stochastic walk; they pick "circle" or "oval" and parameterize it. Encoding that domain knowledge as templates is more useful than a generic search that thrashes.

What this gives up: layouts that are neither circles, ovals, nor stadiums (e.g. dog-bones, kidney shapes). Adding more templates is straightforward; adding switches/crossings expands the template grammar (e.g. "figure-8 with crossing"). The full multigraph traversal remains a v2 cut.

**Pruning inside template instantiation:**
1. Inventory: a template is rejected before geometric build if it asks for more pieces than stocked.
2. Boundary: each placement's AABB must lie inside the boundary.
3. Collision: against AABBs of all prior placements.
4. Closure: `‖end − start‖ < tol_mm` and `|Δθ| < tol_deg` (always satisfied by construction for valid templates, but checked anyway as a guard against catalog mistakes).

### Scoring

```
score = w_util · utilization
      + w_lap  · normalized_lap_speed
      + w_comp · area_efficiency
```

`utilization = used / available`. `lap_speed` is path-length / lap-time (so faster = higher). `area_efficiency = bbox_area / boundary_area` clipped to `[0, 1]` (closer to 1 is denser). Weights default to `(1.0, 0.3, 0.2)` and are exposed in Tweaks.

### Tolerances

- Closure position: 2 mm (= ¼ stud). LEGO's stud-pitch is forgiving but this catches numerical drift.
- Closure heading: 0.5°.
- Collision overlap: 1 mm slack (curves' AABBs already over-approximate).

### Tests

Visible in the **Self-check** panel. Coverage:
- Pose composition associative & inverse identity.
- 16 right-curves close exactly (the canonical circle).
- 16 left-curves close exactly.
- 8 right + 8 left + arbitrary straights → not closed in heading (negative test).
- AABB intersection symmetric and reflexive.
- Boundary-fit detects an out-of-bounds piece.
- A hand-crafted oval (8R + 2S × 2 sides) closes.

If any test is red, the search button refuses to run.

### Why a browser?

The brief asked for CLI; I work in HTML. Everything that matters — algorithm, geometry, tests, config, deterministic output — is the same. The trade-off is that you can't pipe it into a shell script, but you get a live visualization and editable config for free.

### What a "competent programmer in an afternoon" reads

- `geometry.js` — Pose, port composition, closure (≈ 80 lines).
- `catalog.js` — piece data + builder (≈ 60 lines).
- `search.js` — DFS + pruning + scoring (≈ 200 lines).
- `render.js` — canvas drawing (≈ 120 lines).
- `tests.js` — geometry tests (≈ 100 lines).
- `app.jsx` — UI glue (≈ 250 lines).

Everything else is config and styling.
