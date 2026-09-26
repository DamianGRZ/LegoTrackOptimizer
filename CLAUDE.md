# legoga — LEGO track prototype

**Subject of this document:** the `legoga_real/` prototype. It is the active line of work
and the only system described here.

---

## The genome

**V1 used an integer vector** — one long chromosome with extra fixed-purpose fields
appended at the end. **The prototype does not.** An individual is one `Layout` object
(legoga_real/layout.py:32) built from three parts:

**1. A list of piece names.** `types`, a plain list of strings. The *index* is the piece
id — "the first piece" means `types[0]`, and every other structure refers to pieces by
that integer. The string itself is not used directly for anything: it is a key into
`ELEMENTS` (legoga_real/geometry.py:131), which is where ports, routes and shape actually
come from.

**2. A dictionary of joints.** `match`, mapping a `(piece, port)` tuple to another
`(piece, port)` tuple. **Every joint is stored twice** — `(x, px) -> (y, py)` *and*
`(y, py) -> (x, px)` — so a connection can be found from either piece without searching.
Written and removed only through `link` / `unlink` (legoga_real/layout.py:55, :60), which
maintain both directions and invalidate the cache.

**3. A cache.** `_cache`, holding the results of the current round of work — placement
(legoga_real/layout.py:108), reduced graph (:157), route count (:199), near pairs (:260),
overlap (:288) and the bounding rectangle per rotation (:362). Any edit clears it. Near
pairs are keyed by the distance asked for, so a different distance cannot be served
somebody else's answer.

Because the genome carries the joints themselves, "no open track" is **definitional**
rather than a penalty: a port either has a partner in `match` or the object is not a
valid layout.

pymoo sees this as `n_var=1`, `vtype=object` (legoga_real/ga.py:72) — a numpy object
array of `Layout` instances. That is why sampling, crossover and mutation are all custom;
no stock pymoo operator can touch this representation.

### The two questions asked of a layout

- `is_matched()` — legoga_real/layout.py:94 — are there any free ports? Topology only.
- `is_planar_closed()` — legoga_real/layout.py:142 — does it close *geometrically*, in
  the plane?

The first is guaranteed by construction; the second is what can genuinely fail.

---

## No repair operator

There is no repair stage, and none should be added. The transform table
(legoga_real/table.py:54) groups chains of pieces by their **net transform**, so two
chains from one bucket are interchangeable anywhere in any loop. Mutation and crossover
build only out of those equivalences, so they do not create invalid layouts in the first
place — there is nothing left to repair. As a last guard, `mutate` accepts a candidate
only if it is still matched and still closes (legoga_real/ops.py:493).

Operators that drop pieces leave holes in the numbering. Closing them is one shared step,
`Layout.compact()` (legoga_real/layout.py:72), called at the end of every such operator
(legoga_real/ops.py:193, :471, :689, :810) — never hand-written again.

## Pairing: what is a law and what is only this operator

**Switches must come in pairs.** This one is structural. A switch has three ports; adding
one alone leaves a port without a partner, which the representation forbids outright
(legoga_real/ops.py:366, :439).

**Crossings come in pairs only because of how the current operator works.** It is *not* a
law of geometry. `mut_add_crossing_pair` (legoga_real/ops.py:568) inserts crossings into
an existing closed loop and threads a **second, separate closed loop** through them — and
two distinct closed curves in the plane do cross an even number of times, so within that
construction parity holds.

A single crossing is perfectly legal in a different topology: a **figure eight** is *one*
closed curve that crosses itself once. `figure_eight` (legoga_real/ops.py:37) builds
exactly that: one `XX` and two mirrored lobes of 2 straights, 12 curves, 2 straights — 33
pieces, one route, 160 × 160 studs. It is one of the seeds (legoga_real/ga.py:100), so
single-crossing layouts are in the population from generation 0. The pair operator is
still the only way to *add* a crossing later.

---

## Placement and geometry

`place()` — legoga_real/layout.py:105 — lays the pieces out on the table. It walks the
joints breadth-first from piece 0 and produces one **transform per piece**: how far to
move it in x, how far in y, and by what angle. Where two paths meet the same piece, it
reports the mismatch; an angle mismatch adds `1e3`, so it can never hide inside a
positional tolerance. Closure is accepted below `POS_TOL = 0.30` studs
(legoga_real/layout.py:21).

`nodes()` — legoga_real/layout.py:148 — finds the graph vertices. It does not guess: it
reads the `is_node` boolean set on each element type by hand
(legoga_real/geometry.py:107).

**Every element** (legoga_real/geometry.py:102) carries: a name, its ports, its routes
(which port pairs a train can actually traverse), a set of sample points along its length
— generated by the arc and line helpers (legoga_real/geometry.py:114, :123) and used for
collision detection — and that node flag. The dataclass is **frozen**: an element
definition is read-only by design.

`geometry.py` also holds all geometric constants and precomputed trig tables
(legoga_real/geometry.py:65) — angles are discrete, so sines and cosines are computed
once. Three functions do the SE(2) work:

- `compose(t1, t2)` — legoga_real/geometry.py:70 — apply `t1`, then `t2` on top of it.
- `inverse(t)` — legoga_real/geometry.py:81 — undo a transform.
- `apply(t, p)` — legoga_real/geometry.py:88 — move a point by a transform.

**Pose** is `(x, y, a)` with `a` an integer mod 16; one unit = 22.5°. Rotation
composition is therefore **exact**, and only positions are floating point. Never
introduce a float angle.

| Name | What it is | Ports |
|------|-----------|-------|
| `S16` / `S24` | straights, 16 and 24 studs | 2 |
| `CL` / `CR` | R40 curve, 22.5°, left / right | 2 |
| `WL` / `WR` | left / right switch | 3 (node) |
| `XX` | 90° crossing, two independent axes | 4 (node) |
| `DC` | double crossover, 48 studs, axes 16 studs apart | 4 (node) |

Port 0 always sits at the origin facing direction 8; port directions are outward.

`DC` is registered at legoga_real/geometry.py:228. The two slants joining its axes are not
built from prototype pieces and are not modelled — only its ports and routes count
(legoga_real/geometry.py:40).

### Which way up is the layout?

A layout's angle follows from which piece happens to be number 0, and turning the whole
thing on the table is free. So size is measured in whichever of the 16 rotations sticks out
least.

- `extents()` — legoga_real/layout.py:355 — bounding rectangle for every rotation.
- `fit(max_size)` — legoga_real/layout.py:375 — picks one and returns it with the width and
  height. Ties go to the shorter long side, then the shorter short side, on rounded values,
  so float noise cannot decide.

Everything that asks about size goes through `fit`: the size constraint
(legoga_real/ga.py:76), the printed front (legoga_real/run.py:71), the duplicate signature
(legoga_real/ga.py:165) and `summary()` (legoga_real/layout.py:394). The renderer draws the
rotation the GA judged (legoga_real/render_with_v1.py:50).

### Is the layout graph cubic?

Only where crossings are absent. `reduce()` (legoga_real/layout.py:151) collapses chains
of 2-port pieces into edges, leaving switches and crossings as vertices. A switch has
degree 3 — genuinely cubic. A crossing has four ports, but its routes are `((0,1),(2,3))`
(legoga_real/geometry.py:219): **no route joins the two axes**, so a train can never turn
there. A crossing is therefore two edges sharing a location, not a branch point. Switches
are the only real branch vertices, and they are all degree 3.

---

## Transform table

`TransformTable(max_len)` — legoga_real/table.py:54. It walks outward from the empty
chain **breadth-first**: one piece in every direction, then two, up to `max_len`,
recording every chain it meets. The depth is a parameter; its optimal value is to be
established by the ablation, not fixed by hand.

Each chain is filed by its net transform. `_key` (legoga_real/table.py:41) turns that
transform into a bucket address: a grid cell half a stud on a side, plus the angle. A
bucket therefore holds chains of **different lengths and different pieces that all
produce the same displacement** — the equivalence that makes safe mutation possible. By
default 60 chains are kept per bucket; `per_bucket=None` (legoga_real/table.py:56) lifts
the limit and exists so the ablation can measure whether it matters.

Cell numbers are rounded, not floored (legoga_real/table.py:34). Track coordinates land
exactly on grid multiples, where flooring jumps a cell on float drift of order 1e-14.

- `lookup` — legoga_real/table.py:89 — chains realizing a given transform.
- `closure_len` — legoga_real/table.py:99 — admissible lower bound; `inf` means
  unreachable within `max_len`.
- `_flat` — legoga_real/table.py:121 — flattens all buckets into one list ready for
  random scanning.
- `connectors` — legoga_real/table.py:127 — meet-in-the-middle over that flat list,
  reaching `2 * max_len` without extra memory. Answers are remembered and handed out as
  copies, so a caller cannot corrupt the next one.
- `forget` — legoga_real/table.py:157 — drops those remembered answers. Mutation calls it
  once per generation (legoga_real/ga.py:138), so the random scan is drawn afresh instead
  of repeating one generation's luck for the whole run.

Built per problem instance (legoga_real/ga.py:69) with depth 11 by default
(legoga_real/ga.py:66), not cached globally.

---

## The GA

legoga_real/ga.py:65 — `TrackProblem(ElementwiseProblem)`, `n_obj=2`, `n_ieq_constr=3`.

- `F = [-used_pieces, -count_routes]` — maximize both, via negation.
- `G = [overlap, inventory_excess, over_size]`, the last one measured at the best rotation
  (legoga_real/ga.py:76).
- Failed placement returns `10**6` from `overlap()` (legoga_real/layout.py:298), so a
  degenerate layout can never score well.

**Initial population** (legoga_real/ga.py:91): ovals, a circle and a figure eight, each
then put through 3–25 mutations. Inventory is checked **after every single mutation**, not
once at the end — a candidate that would overrun the box of pieces is discarded and the
previous layout kept (legoga_real/ga.py:108).

**Mutation** (legoga_real/ga.py:137): with probability 0.9 an individual receives 1–2
mutations. Each draws from the eight operators below, weighted; `mutate`
(legoga_real/ops.py:477) makes one weighted draw per operator and returns the original if
every one fails.

| Operator | Line | Weight | What it does |
|----------|------|--------|--------------|
| `mut_swap_segment` | ops.py:198 | 0.30 | swap a chain fragment for an equivalent one |
| `mut_antipodal_insert` | ops.py:235 | 0.16 | insert the same neutral block at two points 180° apart, so displacement cancels |
| `mut_add_branch` | ops.py:366 | 0.19 | insert a switch pair, join the diverge legs with a table chain |
| `mut_remove_branch` | ops.py:439 | 0.07 | inverse; each switch becomes two S16 again |
| `mut_add_crossing_pair` | ops.py:568 | 0.10 | insert two crossings, run a second closed loop through them |
| `mut_remove_crossing` | ops.py:665 | 0.04 | drop a cross-axis loop and every crossing sitting on it |
| `mut_add_dbl_crossover` | ops.py:718 | 0.10 | insert a double crossover, close its second track with a table chain |
| `mut_remove_dbl_crossover` | ops.py:769 | 0.04 | inverse; the first track gets 48 studs of straight back |

Two of those need a word beyond the table. Removing a crossing follows the cross axes
right round the loop (`cross_loop`, legoga_real/ops.py:642) and turns **every** crossing it
meets back into an S16 — ports 0 and 1 sit exactly where a straight's do, so nothing else
moves. Removing a double crossover takes the whole component hanging off its second track,
including nodes that grew there, and refills the first track with 48 studs: three S16 or
two S24, drawn at random. A double crossover whose second track rejoins the main one is
skipped.

**Crossover** (legoga_real/ga.py:118) works the same way: two parents, one child
(legoga_real/ops.py:498) — a fragment is transplanted when both sides have the same net
transform — and the child is discarded in favour of the first parent if it overruns
inventory.

`TrackDuplicates` (legoga_real/ga.py:152) is a plain `DuplicateElimination`, not the
elementwise one: it builds a set of signatures and asks each individual once, instead of
comparing every pair. The signature (legoga_real/ga.py:164) is the physical piece counts,
the route count, the longer and shorter side at the best rotation, and whether overlap is
zero.

Before asserting any pymoo API, verify with context7 (`/anyoptimization/pymoo`). Same for
non-trivial Python/numpy semantics. Never answer from memory.

---

## Reading results out of a layout

- `count_routes()` — legoga_real/layout.py:192 — distinct closed routes a train can
  drive, honoring that a switch does not join its through road to its diverge leg and
  that a crossing is two independent paths.
- `overlap()` — legoga_real/layout.py:286 — centerline samples in a grid, counting pairs
  closer than track width, excluding pairs already near *along the track*
  (`NEAR_STUDS = 40`, legoga_real/layout.py:29 — derived from part geometry, not chosen).
- `fit()`, `counts()`, `summary()` — legoga_real/layout.py:375, :97, :393. `bbox()`
  (:340) is still there but nothing calls it any more.

---

## Run it

```bash
python -m legoga_real.run --pop 60 --gens 40 --seed 1 --out outputs/legoga
```

Defaults in `run()` — legoga_real/run.py:43 — `pop=60`, `gens=40`, `seed=1`,
`max_size=(500.0, 500.0)`, `out="outputs/legoga"`. `--out` is a directory; `run()`
creates it and writes `tory.png` (best six), `tory_1.png` … `tory_6.png`,
`tory_progress.csv`, `tory_pieces.png`, `tory_routes.png` into it.

No config file exists. Inventory is a dict in code (legoga_real/ga.py:37), overridable by
passing `inventory=` to `run()`.

### Modules

legoga_real/geometry.py — SE(2) math, constants, the eight element types.
legoga_real/layout.py — the `Layout` genome.
legoga_real/table.py — the equivalence table.
legoga_real/ops.py — seeds and closure-preserving operators.
legoga_real/ga.py — the pymoo layer.
legoga_real/run.py — entry point; `render=False` skips every picture and keeps the CSV.
legoga_real/progress.py — per-generation callback: feasible count, best of each objective,
hypervolume, evaluations, wall time, operator counters; CSV plus two plots.
legoga_real/settings.py — every tunable knob as one frozen `Settings`; defaults are today's
behaviour.
legoga_real/ablate.py — the parameter ablation: variants per stage, seeds × kits × tables,
a process pool, tables built once and loaded from disk, resume by skipping `run.json`.
legoga_real/score_ablation.py — the report: `runs.csv`, paired tests against the base,
plots. Reads run directories only.
legoga_real/render_with_v1.py — draws results with the V1 renderer.
legoga_real/viz.py — standalone drawing; unused, but the only renderer with no dependency
outside the package.

---

## Rules that must not be broken

- **A switch replaces two adjacent S16.** Its body is 32 studs — exactly two straights —
  so the substitution does not move the rest of the layout. That is what `s16_pairs`
  hunts for (legoga_real/ops.py:325).
- **R40 is one physical piece.** `CL` and `CR` are the same brick laid either way and
  share one `R40` pool (legoga_real/ga.py:48). Never budget them separately.
- **Inventory is a constraint, not a knob.** It mirrors the real box; never enlarge it to
  make a run succeed.
- **Geometry constants stay in `geometry.py`.** No dimensions inside operators.
- **Element definitions are read-only.** They are frozen dataclasses; build new ones
  rather than mutating.

## Measured geometry

The switch is where measurement beats derivation — legoga_real/geometry.py:35.

- Body 32 studs along the through road (`SWITCH_BODY`).
- Diverge port measured at `(32.75, ±13.0)`, exit heading 22.5° (`SWITCH_C_X/Y`).
- The diverge leg is R40 along its whole length: two 3-4-5 arcs, `+36.87°` then `−14.37°`
  (legoga_real/geometry.py:170). Never model it as arc-plus-straight.
- The 3-4-5 derivation lands at `(32.69, 12.96)`; the **measured port wins**, and that
  ~0.1-stud residue is why the table's `SNAP` is 0.15 (legoga_real/table.py:31).

Numbers come from data/track_pieces_v2.yaml (4DBrix 2.04.021/018).

---

## What the prototype borrows from the V1 system

The V1 optimizer still lives in the repository (`src/`, `tests/`, `configs/`, `main.py`)
and is **not** described by this document. The prototype touches it in exactly two
places, both output-only:

legoga_real/render_with_v1.py:14 — `src.config`, `src.encoding`,
`src.visualization.track_renderer`, to draw layouts in V1 style.
legoga_real/progress.py:11 — `src.visualization.objective_progress`, for progress plots.

Consequence: `legoga_real` cannot be extracted into a standalone repository while those
imports stand. `legoga_real/viz.py` is the fallback that would make it standalone.

`legoga_orig/` is an earlier snapshot of the same prototype, kept untracked for
reference; 83–100% identical code. The differences are the measured switch geometry, the
real inventory and the V1 renderer. Do not develop against it.

---

## Testing

**There are no automated tests for the prototype.** No file under `tests/` references
`legoga` — verified by grep. The existing suite covers the V1 system only.

Until tests exist, verify changes by:

1. `python -c "import legoga_real.run"` — catches import and syntax breakage.
2. A real run at the documented command; read the printed front — feasible rows must show
   sensible piece counts and zero collisions.
3. `Layout.summary()` (legoga_real/layout.py:393) for a machine-readable dump.
4. `Layout.min_clearance()` (legoga_real/layout.py:406) — an **independent** check on
   `overlap()`: below track width means a physical conflict `overlap()` should have
   caught. Different method, so it is evidence, not a restatement.

No assertion without evidence: never claim a fix works without running the command and
pasting the literal output.

---

## Known gaps and stale statements

Facts about the current code, not a wish list.

- `legoga_real` is **not** in the style gate. The gate is
  `python -m pycodestyle src tests main.py run_v1_all_configs.py run_ablation.py score_ablation.py`
  (`setup.cfg`, 99 chars). The prototype currently reports 4 findings.

### Dead code (verified in both packages)

Never called: `crossing_connectors` (ops.py:475), `_XCONN` (ops.py:506), `other_port`
(geometry.py:251), `__len__` / `cyclomatic` / `bbox` / `summary` / `min_clearance`
(layout.py:83, :242, :340, :393, :406), `stats` (table.py:112), and all of `viz.py`.
`save` / `load` (table.py:163, :167) are called by the ablation only.

`bbox` joined the list when `fit` took over every size question. Nothing else replaced it,
so a caller that genuinely wants the untouched orientation still has it.

Two are deliberate tools rather than debris: `min_clearance` (independent
verification) and `crossing_connectors` (an analytic chain family the table provably
cannot reach — it needs 21 elements). Re-grep before deleting anything.

---

## Conventions

**Code.** Early returns, small functions, no if/elif towers, no run anecdotes in
comments — state the rule, not the session that produced it. Docstrings give the contract
in a few lines. Comments explain *why* a constant holds its value; `geometry.py` is the
model.

**Files and git.** "Remove" or "untrack" never means deleting from disk — use
`git rm --cached` or `.gitignore`. Never `git init` without checking first. Never create
or switch branches, and never `git stash`, without explicit permission; a foreign stash
exists in this repository. Stage an explicit file list, never `git add -A`. Confirm
before `git reset --hard`, force pushes, or dropping files.

**Run artifacts.** Everything under `outputs/` (gitignored). No parallel output trees.
