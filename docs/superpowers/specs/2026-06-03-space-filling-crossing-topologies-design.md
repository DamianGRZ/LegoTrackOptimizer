# Design: Space-filling crossing seed topologies (raise utilization past the simple-loop ceiling)

**Date:** 2026-06-03
**Status:** Design — approved for spec; Phase 1 (C, D) ready to implement, Phase 2 (A′) geometry open
**Author:** Damian Grzesło (with Claude)

---

## Streszczenie (PL)

W konfiguracji `all_pieces` najlepszy **dopuszczalny** układ utyka na ~60% wykorzystania
zestawu, bo prosta zamknięta pętla po prostu nie zmieści więcej toru w pudełku 500×500
(to geometria, nie błąd optymalizatora). Elementy krzyżujące (CROSS_90, DOUBLE_CROSSOVER)
nigdy nie są używane (0/2), bo jedyne istniejące „krzyżujące" nasiona to małe ósemki
(16–42%), które przegrywają z owalem (~60%).

Plan: dodać **gęstsze nasiona (seed) wykorzystujące krzyżowania**, żeby GA miało
wysoko-wykorzystaniowe opcje do utrzymania. Pudełka, zestawu i promienia **R40 nie
zmieniamy** — projektujemy wyłącznie w obrębie tego, co jest.

- **Faza 1 (gotowe): C i D** — tylko nowe generatory nasion, bez zmian dekodera.
- **Faza 2 (odłożone): A′** — zagnieżdżone pierścienie łączone rozjazdami; wymaga
  rozszerzenia dekodera i **jednego konkretnego, budowalnego kształtu** (geometria
  zamknięcia przy jednym promieniu R40 jest punktem otwartym).

---

## 1. Problem & motivation (measured, not assumed)

For `configs/all_pieces.yaml` (box 500×500 studs, fixed; inventory 120 STRAIGHT_16,
80 R40_CURVE, 2 CROSS_90, 2 DOUBLE_CROSSOVER, 3+3 switches), the best **feasible**
utilization across every run tops out at ~58–63% (122–128 pieces).

- This is **geometry, not search failure.** A closed loop of N pieces is ≈16·N studs
  of track. The most box-efficient simple loop is a square: perimeter that fits a 500
  box = 2000 studs ≈ 125 pieces ≈ **~60%**. The GA already converges to this optimum.
- High-utilization solutions are infeasible on the **boundary constraint only**
  (decoded `outputs_v1/verify_all_pieces_3` best-infeasible 68.6% / 140 pcs: closure
  and collisions satisfied, `boundary` G[3]=+0.17). That loop is 2027 studs of track;
  the minimum bounding box for a closed curve of that length (a circle, D=L/π) is
  **645 studs > 500** — it cannot fit in any shape. Shrinking/recentering can't help.
- `CROSS_90` and `DOUBLE_CROSSOVER` are **0/2 used** in every run. Measured seed
  utilization (decode of each heuristic family): racetrack 51% (→ ~60% feasible),
  `fig8_DC` 42%, `figure8`/`fig8_cross` 16%, oval 30%. Every crossing-bearing seed is
  a small figure-8, strictly dominated by the racetrack on utilization (and no better
  on speed), so NSGA-II discards it within a few generations. Mutation cannot *create*
  a crossing, and nothing grows a crossing layout to competitive density.

**Conclusion:** to exceed ~60% in the fixed box we need *new* dense, crossing-using
closed-loop topologies that the GA will keep. See linked session memories
`project_boundary_util_ceiling`, `project_adaptive_epsilon_harmful`.

## 2. Goal & non-goals

**Goal:** add heuristic seed families (and, for Phase 2, the minimal decoder support)
that produce **dense, closed, inventory-legal, boundary-fitting** layouts using the
crossing pieces, so the high-utilization end of the Pareto front has options above the
racetrack's ~60%.

**Non-goals (hard constraints — do NOT touch):**
- Do **not** change the boundary box, the inventory, or the **R40 radius**. They are
  external constraints. (See memory `feedback_config_is_constraint_not_knob`.)
- No new piece types.
- No change to the objectives (`F[0]` utilization, `F[1]` slowest-route speed).

## 3. Key insight

With a **single fixed turning radius (R40 = 40 studs)**:
- Every 180° U-turn is 80 studs wide, so naive serpentines are sparse.
- A planar single closed loop maxes near the racetrack (~60%).
- Folding one loop denser requires self-crossings (CROSS_90) or parallel-track weaves
  (DOUBLE_CROSSOVER) — but inventory caps those at 2 + 2.
- A large density jump needs a *multi-ring* structure threaded into one circuit by
  switch crossovers — which is new structure, not just a new seed (Phase 2 / A′).

So: **C and D are cheap (seed-only, reuse existing multi-descriptor injection) but
modest (~60–75%); A′ is the real density win (~85–95%) but needs decoder work and an
unresolved single-radius closure geometry.**

## 4. Scope & phasing

- **Phase 1 — C, D (this spec, ready):** new seed generators in `src/sampling.py`
  only. No decoder/encoding changes. Reuse the existing multi-descriptor injection
  (`_inject_cross_junctions`, `_inject_double_crossovers` already loop over multiple
  descriptors — `construction.py:411`, `:484`).
- **Phase 2 — A′ (deferred):** decoder/encoding extension. Blocked on a concrete,
  buildable ring-threading-and-closure geometry under the single R40 radius. Documented
  in §7 as open; no code until the geometry is pinned by the domain expert.

## 5. Family C — stacked loops + DOUBLE_CROSSOVER

Generalizes the existing, validated `_gen_figure_eight_dbl_crossover`.

- **Shape:** 2–3 parallel oval loops (16 studs apart — the DC's track spacing), woven
  into one circuit by 1–2 DC pieces (inventory owns 2 DC → ≤3 loops). A single DC
  provides both crossing directions, merging two adjacent loops into one circuit.
- **Emits:** `(main_pieces, main_flips, None, None, dbl_crossover_descriptors)` —
  stacked-oval main loop + DC descriptor(s) with a valid 2-route cover
  (`both-cross` or `both-through`).
- **Decoder:** unchanged. `_inject_double_crossovers` commits each DC where the named
  slots are STRAIGHT_16 and the FK port origins match.
- **Scaling (no hardcoded counts):** loop count bounded by DC inventory and boundary
  height; loop length by STRAIGHT_16 inventory and boundary width. Emit only when the
  required pieces are available.

## 6. Family D — triple-lobe, 2× CROSS_90

Generalizes the existing, validated `_gen_figure_eight_cross` (1 cross descriptor).

- **Shape:** three lobes in a row; the middle lobe shares a perpendicular self-crossing
  with each neighbour → two CROSS_90 crossings.
- **Emits:** `(main_pieces, main_flips, None, cross_junction_descriptors, None)` with
  **two** descriptors naming the two crossing slot-pairs.
- **Decoder:** unchanged. `_inject_cross_junctions` already commits multiple crossings
  where slots are STRAIGHT_16 and cross perpendicular by FK.
- **Scaling:** lobe straight-count grows with inventory/boundary; emit only when ≥2
  CROSS_90 and the curves/straights are available.

## 7. Family A′ — concentric rings + switch crossovers (Phase 2, geometry OPEN)

**Intent:** nested oval rings joined ring-to-ring by 2-switch crossovers, threaded into
one circuit — the only path to ~85–95% in the fixed box.

**Architecture (the buildable part):** mirror the existing DC-figure-8 treatment, which
already exists end-to-end:
- A switch traversed on its *diverging route* as a through-piece (not a branching
  siding). Reuse the route-aware FK already used for DC
  (`_state_at_position_with_routes` / `get_fk_route`) via a `route_map`. No 2^N branch
  enumeration — a crossover is a deterministic through-path.
- Encoding: main-loop slots may carry a switch index + route (mirrors the DC route_map
  threaded through `_compute_path_fk`); inventory counts them as switches.
- Protect from mutation like DC figure-8s (`_protected_positions`), plus a dedicated
  `_grow_concentric_ring` operator that adds one ring + crossover when inventory/boundary
  allow, so the GA can climb ring count.
- Self-validate by decode (must close, all switches/crossings committed, fits boundary).

**OPEN — blocking issue:** how N nested rings thread into **one closed circuit and
return to start** under a single R40 radius. Three return styles were proposed
(spiral-and-shoot-out; weave out-and-back; describe-your-own) and **all rejected** as
not buildable. Phase 2 does **not** start until a concrete buildable threading (a
specific train path through 3–4 rings) is supplied by the domain expert. Until then,
A′ is documented intent only.

## 8. Validation & testing

Per the project's self-validate pattern (every seed decodes before use):
- For each new family, a generator unit test: decode each emitted variant and assert
  (a) layout closed (closure within tolerance), (b) the expected CROSS_90 / DC pieces
  actually **committed** (not dropped), (c) bounding box fits the boundary, (d) inventory
  legal. Mirror `tests/test_cross90_objective.py` and the DC tests.
- Integration: a full `all_pieces` GA run should show `CROSS_90` and/or
  `DOUBLE_CROSSOVER` usage > 0 in the best feasible solution, and best feasible
  utilization above the ~60% racetrack ceiling.
- Run the FULL suite after changes (no `-k` subset) per project policy.

## 9. Risks & open questions

- **C/D density may be only marginally above 60%** (~60–75%). They primarily prove the
  pipeline and get the crossing pieces used; the real win is A′. Accept as Phase 1.
- **A′ geometry is unresolved** (§7) and is the main value — Phase 2 is gated on it.
- Seed variants that fail to commit their crossing must release inventory and be
  dropped (existing contract); verify no silent half-committed layouts.

## 10. Out of scope

Boundary/inventory/radius changes; objective changes; new piece types; operator changes
beyond the A′ grow operator (Phase 2).
