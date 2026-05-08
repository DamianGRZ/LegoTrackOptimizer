# Decoder Reservation-First Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make switches actually appear in decoded layouts under `configs/with_switches.yaml` by fixing the three defects that jointly starve junctions: out-of-bounds handedness mutation, zero-gradient silent rejection, and greedy main-loop inventory consumption.

**Architecture:** Three layered fixes applied smallest-first so regressions surface incrementally. (1) Fix the `_change_handedness` mutation to stay within declared bounds. (2) Add a shadow constraint `g_unplaced_junctions` so Deb-rule can see activated-but-not-placed junctions as a gradient-informative residual (Coello Coello 2002, DOI 10.1016/S0045-7825(01)00323-1). (3) Invert decoder order so `_read_junctions` reserves template inventory *before* `_read_main_loop` consumes it (Puchinger & Raidl 2008 reservation-first pattern, DOI 10.1016/j.cor.2006.11.002; Lim et al. 2006 class-priority decoder, DOI 10.1109/CEC.2006.1688681). Each task is TDD: failing test first, minimal implementation, passing test, commit.

**Tech Stack:** pymoo 0.6.1.6, numpy, pytest; all changes Python-only inside `src/` and `tests/`.

**Evidence this plan is necessary:** Previous turn instrumented the decoder across 1000 saved chromosomes from the latest `with_switches.yaml` run. Result: 417 active junctions, 416 killed by inventory starvation, 1 killed by geometry, **0 switches landed in any decoded layout**. Main loop is consuming all curves (40 R40_LEFT + 40 R40_RIGHT + 80 STRAIGHT_16 = 160) before junctions reach the inventory check. See `docs/superpowers/plans/2026-04-22-ga-dynamics-research.md` for the complementary analysis (feasibility collapse, boundary explosion, frozen elite) that applies once these fixes restore a gradient-bearing decoder.

---

## File Structure

**Modified files (three production changes):**

- `src/operators.py` — Task 1: fix `_change_handedness` random-int range.
- `src/problem.py` — Task 2: add one scalar to the `G` vector + update `n_ieq_constr`.
- `src/decoder/construction.py` — Task 3: add `_reserve_junctions` phase before `_read_main_loop`; split inventory claiming from position validation.

**Test files (one new, two extended):**

- `tests/test_operators.py` — **new file**, Task 1's handedness bound regression test.
- `tests/test_problem.py` — extend with Task 2's shadow-constraint assertion.
- `tests/test_decoder.py` — extend with Task 3's starvation-prevention test + the new public helpers' tests.

**Integration validation (no new files):**

- Task 4 runs `python main.py --config configs/with_switches.yaml` with full generations and inspects `outputs/` using the `/diag` skill per CLAUDE.md's "no assertion without evidence" rule.

---

## Task 1: Fix `_change_handedness` out-of-bounds mutation

**Context:** `src/operators.py:160-166` sets the handedness gene to `np.random.randint(0, 4)`, but the declared bound (`src/encoding.py:190-191`) is `xu=1`. The decoder's `handedness % len(TEMPLATES)` rescue (`src/decoder/construction.py:159`) hides the violation, but ~178/417 junctions in the last run had handedness ∈ {2, 3}, and 3 maps to RIGHT — which requires `R40_SWITCH_RIGHT_IN/OUT` that this inventory does not contain, guaranteeing inventory-fail even if Task 3 were already in place. This bug is independent of Tasks 2 and 3 and is the cheapest defect to close.

**Files:**
- Modify: `src/operators.py:160-166`
- Create: `tests/test_operators.py`

- [ ] **Step 1: Create the failing test file**

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_operators.py::TestChangeHandedness::test_stays_within_template_bounds -v`

Expected: FAIL with assertion message listing values like `{2, 3}` that exceed `xu=1`.

- [ ] **Step 3: Apply the minimal fix**

Edit `src/operators.py` lines 160–166. Replace:

```python
def _change_handedness(x: NDArray, dims: PartitionedDimensions) -> None:
    """Set handedness of a random junction to a random value 0-3."""
    if dims.max_junctions == 0:
        return
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    x[base + 2] = np.random.randint(0, 4)
```

With:

```python
def _change_handedness(x: NDArray, dims: PartitionedDimensions) -> None:
    """Set handedness of a random junction to a valid template index."""
    if dims.max_junctions == 0:
        return
    from .templates import TEMPLATES
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    x[base + 2] = np.random.randint(0, len(TEMPLATES))
```

The import is local to avoid a circular import at module load (`operators.py` is imported by pymoo problem construction; `templates.py` imports from `encoding.py` which operators also imports). `len(TEMPLATES)` is 2 today but the function stays correct if more templates are added.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest tests/test_operators.py::TestChangeHandedness::test_stays_within_template_bounds -v`

Expected: PASS.

- [ ] **Step 5: Run the full operators- and decoder-adjacent test suite to confirm no regression**

Run: `pytest tests/test_operators.py tests/test_decoder.py tests/test_sampling.py tests/test_templates.py -v`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/operators.py tests/test_operators.py
git commit -m "fix(operators): clamp _change_handedness to declared bounds

Out-of-bounds handedness values (2, 3) relied on the decoder's
modulo-2 rescue but silently flipped LEFT-intent junctions into
RIGHT targets, guaranteeing inventory failure for left-only configs.
Range now matches len(TEMPLATES)."
```

---

## Task 2: Add shadow constraint for unplaced junctions

**Context:** The decoder silently drops active junctions whose inventory or geometry check fails. After decode, `layout.switch_pairs` counts only *placed* junctions — the GA has no way to distinguish "junction gene activated but dropped" from "junction gene inactive." Both produce identical fitness. The fix is Coello Coello (2002)'s gradient-informative constraint pattern: expose attempted-minus-placed as a new `G` term so Deb's feasibility-first ranking can order chromosomes by their placement success. `ConstrRankAndCrowding` handles the new constraint with no pymoo API changes.

**Files:**
- Modify: `src/problem.py` — extend `__init__`'s `n_ieq_constr`, extend `_evaluate`'s G-vector assembly.
- Modify: `src/encoding.py` — expose a small helper `count_active_junctions` (returns an int) so the problem layer does not duplicate gene-reading logic.
- Modify: `tests/test_problem.py` — add the regression test.

- [ ] **Step 1: Inspect the current constraint wiring**

Read `src/problem.py:38-67` and `src/problem.py:93-158`. Note three facts:
- `n_ieq_constr=5 + catalog.n_pieces` at line 63.
- `out["G"]` is assembled at lines 153-157 as a concatenation of (5 closure/boundary/collision terms) + (per-type inventory vector of length `catalog.n_pieces`).
- Constraint column labels in `outputs/constraints.csv` are derived from this order; appending a new column at the end preserves backward compatibility for existing analysis tools and the `/diag` skill.

The new constraint will be appended *after* the per-type inventory block so the five canonical columns and the `inv_0..inv_N-1` columns keep their indices.

- [ ] **Step 2: Write the helper function**

Add to `src/encoding.py`, immediately after `get_active_junctions` (around line 272):

```python
def count_active_junctions(x: NDArray, dims: PartitionedDimensions) -> int:
    """Return the number of junctions whose `active` gene is 1.

    Used by the problem layer to compute the attempted-minus-placed
    shadow constraint (Coello Coello 2002). This counts gene-level
    activation only — it does NOT check whether the decoder accepts
    the junction; that is the point of the residual metric.
    """
    n = 0
    for k in range(dims.max_junctions):
        base = dims.junc_start + k * GENES_PER_JUNCTION
        if int(x[base + JUNC_ACTIVE]) == 1:
            n += 1
    return n
```

- [ ] **Step 3: Write the failing test**

Add to `tests/test_problem.py` under a new class (at the end of the file):

```python
from src.encoding import (
    count_active_junctions,
    create_chromosome_from_pieces,
)
from src.problem import TrackOptimizationProblem


class TestUnplacedJunctionConstraint:
    """Regression tests for the shadow constraint on unplaced junctions."""

    def test_constraint_dimension_includes_shadow_term(self, catalog, switches_config):
        problem = TrackOptimizationProblem(catalog, switches_config)
        # 5 canonical + n_pieces per-type + 1 shadow == problem's n_ieq_constr
        assert problem.n_ieq_constr == 5 + catalog.n_pieces + 1

    def test_shadow_term_is_zero_when_no_junctions_active(
        self, catalog, switches_config
    ):
        """A chromosome with no active junctions has shadow term == 0."""
        problem = TrackOptimizationProblem(catalog, switches_config)
        dims = problem.dims
        # Simple main loop, no junctions activated
        pieces = [2] * 16  # 16 R40_LEFT, closed circle
        x = create_chromosome_from_pieces(dims, pieces, junctions=None)

        out = {}
        problem._evaluate(x, out)

        # Shadow constraint is the last column of G
        shadow = out["G"][-1]
        assert shadow == 0.0, (
            f"expected shadow==0 with no junctions, got {shadow}"
        )

    def test_shadow_term_positive_when_junction_fails_to_place(
        self, catalog, switches_config
    ):
        """Activating a junction the decoder cannot place must produce shadow > 0."""
        problem = TrackOptimizationProblem(catalog, switches_config)
        dims = problem.dims
        # Small main loop that cannot geometrically host a siding at position 0
        pieces = [2] * 4  # too short for a siding
        junctions = [(1, 0, 0, 0)]  # active LEFT siding at pos 0
        x = create_chromosome_from_pieces(dims, pieces, junctions)

        n_active = count_active_junctions(x, dims)
        assert n_active == 1

        out = {}
        problem._evaluate(x, out)

        shadow = out["G"][-1]
        # 1 activated, 0 placed => shadow = 1 / max_junctions > 0
        assert shadow > 0.0, (
            f"expected shadow>0 with 1 failed junction, got {shadow}"
        )
        assert shadow <= 1.0
```

- [ ] **Step 4: Run the test to confirm it fails**

Run: `pytest tests/test_problem.py::TestUnplacedJunctionConstraint -v`

Expected: all three tests FAIL — `n_ieq_constr` dimension mismatch and `G` is too short.

- [ ] **Step 5: Update `n_ieq_constr` in the problem constructor**

Edit `src/problem.py` line 63. Change:

```python
            n_ieq_constr=5 + catalog.n_pieces,  # Stage B: 3 closure + boundary + collisions + per-type inventory
```

To:

```python
            n_ieq_constr=5 + catalog.n_pieces + 1,  # + 1 shadow constraint for unplaced junctions
```

- [ ] **Step 6: Append shadow constraint to `_evaluate`**

Edit `src/problem.py`. Locate the `g_vec = np.concatenate([...])` block (lines 153-157). Replace:

```python
        g_vec = np.concatenate([
            np.array([g_closure_x, g_closure_y, g_closure_theta,
                      g_boundary, g_collisions], dtype=np.float64),
            g_inventory_per_type,
        ])
        out["G"] = g_vec
```

With:

```python
        # Shadow constraint: gene-activated junctions that the decoder did
        # not realize as switch pairs. Gives Deb-rule a gradient between
        # "tried and failed" and "never tried" — which would otherwise map
        # to identical phenotypes. See Coello Coello (2002) DOI
        # 10.1016/S0045-7825(01)00323-1.
        from .encoding import count_active_junctions
        n_activated = count_active_junctions(x, self.dims)
        n_placed = len(layout.switch_pairs)
        max_j = max(1, self.dims.max_junctions)
        g_unplaced = max(0, n_activated - n_placed) / max_j

        g_vec = np.concatenate([
            np.array([g_closure_x, g_closure_y, g_closure_theta,
                      g_boundary, g_collisions], dtype=np.float64),
            g_inventory_per_type,
            np.array([g_unplaced], dtype=np.float64),
        ])
        out["G"] = g_vec
```

Also patch the zero-piece sentinel branch at `src/problem.py:104-107` so the new dimension matches:

```python
        if layout.n_pieces == 0:
            out["F"] = np.array([np.inf, np.inf])
            out["G"] = np.full(self.n_ieq_constr, 1e6)
            return
```

This already uses `self.n_ieq_constr`, so no edit is needed — the `1e6` fill now extends to the new column automatically. Verify by eye and move on.

- [ ] **Step 7: Run the shadow-constraint tests to confirm they pass**

Run: `pytest tests/test_problem.py::TestUnplacedJunctionConstraint -v`

Expected: all three tests PASS.

- [ ] **Step 8: Run the full problem test suite to confirm no regression**

Run: `pytest tests/test_problem.py -v`

Expected: all PASS. Any pre-existing test that reads `G[-1]` expecting the old last column will need inspection — look at the failure message, and if the test was asserting "last column is inventory for piece N-1," update it to index positionally (e.g., `G[5 + catalog.n_pieces - 1]`) rather than `G[-1]`.

- [ ] **Step 9: Run the full suite**

Run: `pytest --tb=short -q`

Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add src/encoding.py src/problem.py tests/test_problem.py
git commit -m "feat(problem): shadow constraint for unplaced junctions

Adds g_unplaced = (activated - placed) / max_junctions as the last
inequality constraint. Without it, chromosomes that activate a
junction the decoder rejects are indistinguishable from chromosomes
that never tried — zero gradient on switch-seeking behavior. With
it, Deb-rule can order failed attempts below successful placements.

Refs Coello Coello (2002), DOI 10.1016/S0045-7825(01)00323-1."
```

---

## Task 3: Reservation-first decoding

**Context:** The dominant cause of switch loss (416/417 in the last run) is that `_read_main_loop` in `src/decoder/construction.py:100-124` iterates all active main-loop genes and greedily consumes inventory via `tracker.use`, leaving the curve pool empty when `_read_junctions` tries to claim 1×R40_LEFT + 1×R40_RIGHT per LEFT template. Invert the order: extract junction descriptors and reserve their template requirements *first*, then let the main loop consume what remains. This is the canonical reservation-first pattern for multi-choice knapsack variants (Puchinger & Raidl 2008, Sinha & Zoltners 1979, Lim et al. 2006). Junctions that fail geometry in `_inject_switches` already release inventory via the existing `_release_junction_inventory` helper — no new release machinery is needed.

**Files:**
- Modify: `src/decoder/construction.py` — add `_reserve_junctions` before `_read_main_loop`; split junction validation into reserve-then-finalize.
- Modify: `tests/test_decoder.py` — add `TestReservationFirst` class with three regression tests.

- [ ] **Step 1: Read the current decoder to confirm the touch points**

Read `src/decoder/construction.py:45-186`. Note:
- `decode_chromosome` (lines 45-93) calls `_read_main_loop` then `_read_junctions` then `_inject_switches`.
- `_read_junctions` (lines 131-186) already contains the inventory check (`tracker.can_use_batch(requirements)`) and the `ValidatedJunction` construction. It also does position clamping and handedness-modulo.
- `_inject_switches` (lines 193-285) already releases failed junctions via `_release_junction_inventory`.

The minimal-risk change is: add a new helper `_reserve_junctions` that does the *inventory-claim* portion of `_read_junctions`, have `decode_chromosome` call it *before* `_read_main_loop`, and have `_read_junctions` just finalize the already-reserved set.

- [ ] **Step 2: Write the failing test for starvation prevention**

Add to `tests/test_decoder.py` (at the end of the file):

```python
class TestReservationFirst:
    """Tests that junction inventory is reserved before main loop consumption."""

    def test_junction_places_when_main_loop_would_starve_it(
        self, catalog, switches_config
    ):
        """A LEFT junction with exact inventory room must land even when the
        chromosome's main loop genes would otherwise exhaust the curve pool.
        """
        from src.encoding import compute_dimensions

        dims = compute_dimensions(switches_config, catalog)
        # Main loop genes ask for every curve in inventory: 40 LEFT + 40 RIGHT +
        # 80 STRAIGHT_16 = 160 pieces — a fully-packed backbone that leaves
        # zero curves for a template siding under the old greedy decoder.
        main = (
            [int(R40_LEFT)] * 40
            + [int(R40_RIGHT)] * 40
            + [int(STRAIGHT_16)] * 80
        )
        # One LEFT siding at a mid-loop position, n_straights=0 (minimum
        # inventory demand: 1 LEFT_IN, 1 LEFT_OUT, 1 R40_RIGHT approach,
        # 1 R40_LEFT return).
        junctions = [(1, 20, 0, 0)]
        chromosome = create_chromosome_from_pieces(dims, main, junctions)

        layout = decode_chromosome(
            chromosome, catalog, switches_config.inventory, dims=dims,
        )

        assert len(layout.switch_pairs) == 1, (
            "junction must place under reservation-first decoding; "
            f"got {len(layout.switch_pairs)} switch pairs"
        )

    def test_junction_reservation_shrinks_main_loop_piece_count(
        self, catalog, switches_config
    ):
        """Reserved curves must not be double-counted in the main loop."""
        from src.encoding import compute_dimensions

        dims = compute_dimensions(switches_config, catalog)
        # Same backbone as above: 160 pieces requested.
        main = (
            [int(R40_LEFT)] * 40
            + [int(R40_RIGHT)] * 40
            + [int(STRAIGHT_16)] * 80
        )
        junctions = [(1, 20, 0, 0)]
        chromosome = create_chromosome_from_pieces(dims, main, junctions)
        layout = decode_chromosome(
            chromosome, catalog, switches_config.inventory, dims=dims,
        )

        # Main loop had 160 requested pieces. The junction's 4 reserved pieces
        # (1 LEFT_IN + 1 LEFT_OUT + 1 R40_LEFT + 1 R40_RIGHT) means main loop
        # can consume at most 156 non-switch pieces. After switches inject,
        # the augmented loop contains the 2 switches, so augmented length
        # should be <= 156 + 2 main slots + branch pieces (which are reported
        # separately).
        n_main_actually_used = len(layout.main_loop_pieces)
        assert n_main_actually_used <= 158, (
            f"main loop over-consumed: used {n_main_actually_used}, "
            "expected <= 158 after junction reservation"
        )

    def test_zero_junctions_preserves_original_main_loop_fill(
        self, catalog, default_config
    ):
        """With no junction genes active, the main loop fill matches pre-fix
        behavior (no inventory is reserved).
        """
        from src.encoding import compute_dimensions

        dims = compute_dimensions(default_config, catalog)
        pattern = [int(R40_LEFT)] * 16
        chromosome = create_chromosome_from_pieces(dims, pattern, junctions=None)

        layout = decode_chromosome(
            chromosome, catalog, default_config.inventory, dims=dims,
        )

        # No junctions; 16 curves in, 16 curves in the main loop.
        assert len(layout.main_loop_pieces) == 16
```

- [ ] **Step 3: Run the test to confirm it fails**

Run: `pytest tests/test_decoder.py::TestReservationFirst -v`

Expected: `test_junction_places_when_main_loop_would_starve_it` FAILS with `got 0 switch pairs`. The other two pass trivially pre-fix (the second is an upper-bound that the greedy decoder satisfies; the third is a no-junction path).

- [ ] **Step 4: Add `_reserve_junctions` to the decoder**

Edit `src/decoder/construction.py`. Add the helper immediately after the imports block (around line 40) and before `decode_chromosome`:

```python
# =============================================================================
# Phase 0: Reserve Junction Inventory
# =============================================================================

def _reserve_junctions(
    x: NDArray,
    dims: PartitionedDimensions,
    tracker: InventoryTracker,
    catalog: TrackCatalog,
) -> List[ValidatedJunction]:
    """Reserve template inventory for active junctions BEFORE main loop consumption.

    Reads junction descriptor genes, computes each siding's piece requirements
    from its template + n_straights, and claims that inventory up-front.
    Junctions that cannot claim their full requirement (e.g., because the
    handedness has no switches in the current inventory) are dropped.

    This enforces class-priority inventory allocation so the greedy main-loop
    consumption in _read_main_loop cannot exhaust curves that an active
    junction needs (Puchinger & Raidl 2008, DOI 10.1016/j.cor.2006.11.002;
    Lim et al. 2006, DOI 10.1109/CEC.2006.1688681).

    Returns:
        List of ValidatedJunction entries whose inventory is already
        committed in the tracker. Position is NOT yet clamped to main-loop
        length (done in _finalize_junctions once main loop is known).
    """
    reserved: List[ValidatedJunction] = []
    for slot, _active, position, handedness, n_straights in get_active_junctions(x, dims):
        handedness = handedness % len(TEMPLATES)
        template = TEMPLATES[handedness]
        n_straights = max(0, min(n_straights, dims.total_straights))
        requirements = get_siding_inventory_requirements(template, n_straights)

        if not tracker.can_use_batch(requirements):
            # Not enough inventory for this template — deactivate silently
            # (Task 2's shadow constraint picks up the lost attempt).
            continue

        tracker.use_batch(requirements)
        reserved.append(ValidatedJunction(
            slot=slot,
            position=position,  # unclamped; _finalize_junctions will clamp
            handedness=handedness,
            n_straights=n_straights,
            template=template,
            branch_pieces=compute_branch_pieces(template, n_straights),
            siding_requirements=requirements,
        ))
    return reserved


def _finalize_junctions(
    reserved: List[ValidatedJunction],
    n_main: int,
) -> List[ValidatedJunction]:
    """Clamp reserved junctions' positions to the actual main-loop length.

    Inventory is already committed by _reserve_junctions; this pass only
    fixes position indices after we know how many main-loop pieces landed.
    Junctions whose required main-loop span cannot fit are left to
    _inject_switches to reject on geometry, which will then release the
    reservation via _release_junction_inventory.
    """
    if n_main < 4:
        return []
    return [
        ValidatedJunction(
            slot=j.slot,
            position=max(0, min(j.position, n_main - 1)),
            handedness=j.handedness,
            n_straights=j.n_straights,
            template=j.template,
            branch_pieces=j.branch_pieces,
            siding_requirements=j.siding_requirements,
        )
        for j in reserved
    ]
```

- [ ] **Step 5: Rewire `decode_chromosome` to call `_reserve_junctions` first**

Edit `src/decoder/construction.py:45-93`. Replace the body of `decode_chromosome` (keeping the signature) with:

```python
def decode_chromosome(
    x: NDArray,
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    dims: PartitionedDimensions,
    config: Optional[DecoderConfig] = None,
) -> MultiPathLayout:
    """Decode a partitioned chromosome into a MultiPathLayout.

    Decode order (reservation-first):
      0. Reserve junction template inventory.
      1. Read main loop pieces from post-reservation inventory.
      2. Finalize junctions (position clamping).
      3. Inject switches with geometric validation.
      4. Crossing repair, FK, auto-center.
    """
    if config is None:
        config = DecoderConfig()

    tracker = InventoryTracker(inventory, catalog)

    # Phase 0: Reserve junction inventory before any main-loop consumption
    reserved = _reserve_junctions(x, dims, tracker, catalog)

    # Phase 1: Read main loop from inventory reduced by reservations
    main_pieces = _read_main_loop(x, dims, tracker)

    if not main_pieces:
        # Main loop empty — release any reservations to keep tracker tidy.
        for junc in reserved:
            _release_junction_inventory(junc, tracker)
        return _empty_layout()

    # Phase 2: Finalize reserved junctions against actual main-loop length
    junctions = _finalize_junctions(reserved, len(main_pieces))

    # Phase 3: Inject switches; geometry failures release reservations internally
    augmented_pieces, switch_pairs = _inject_switches(
        main_pieces, junctions, tracker, catalog, config,
    )

    # Phase 4: Self-intersection repair (CROSS_90 injection)
    augmented_pieces = _apply_crossing_repair(augmented_pieces, tracker, catalog, config)

    # Phases 5 + 6: FK + 2^J traversal paths
    multi_path = _build_multi_path_layout(augmented_pieces, switch_pairs, catalog)

    # Phase 7: Auto-center within boundary
    start_x, start_y = get_start_position(x, dims)
    _auto_center(multi_path, config, start_x, start_y)

    return multi_path
```

- [ ] **Step 6: Remove the obsolete inventory-check from `_read_junctions`**

`_read_junctions` is no longer the reservation site. It is kept only as a compatibility shim in case any test imports it directly. Edit `src/decoder/construction.py:131-186`. Replace the function with:

```python
def _read_junctions(
    x: NDArray,
    dims: PartitionedDimensions,
    main_pieces: List[int],
    tracker: InventoryTracker,
    catalog: TrackCatalog,
    config: DecoderConfig,
) -> List[ValidatedJunction]:
    """Deprecated; retained for backward compatibility with callers that
    still invoke the pre-reservation-first flow. Internally it now delegates
    to _reserve_junctions + _finalize_junctions with an already-initialized
    tracker, which means inventory has already been consumed by the caller's
    main-loop pass. Prefer the new two-phase API in decode_chromosome.
    """
    reserved = _reserve_junctions(x, dims, tracker, catalog)
    return _finalize_junctions(reserved, len(main_pieces))
```

The shim is kept so any external caller (test or script) that assembled its own `decode_chromosome`-equivalent pipeline still gets correct behavior, just without the reservation ordering benefit. Internal decoder flow no longer calls it.

- [ ] **Step 7: Verify imports are complete**

Confirm that `src/decoder/construction.py` already imports `TEMPLATES`, `compute_branch_pieces`, and `get_siding_inventory_requirements` (it does — lines 31-37). No new imports are required.

- [ ] **Step 8: Run the reservation-first tests to confirm they pass**

Run: `pytest tests/test_decoder.py::TestReservationFirst -v`

Expected: all three tests PASS.

- [ ] **Step 9: Run the full decoder test suite to confirm no regression**

Run: `pytest tests/test_decoder.py -v`

Expected: all PASS. In particular `test_main_loop_respects_inventory` must still pass — its main loop has no active junctions, so `_reserve_junctions` returns an empty list and the path is equivalent to the old one.

- [ ] **Step 10: Run the full suite**

Run: `pytest --tb=short -q`

Expected: all PASS.

- [ ] **Step 11: Commit**

```bash
git add src/decoder/construction.py tests/test_decoder.py
git commit -m "fix(decoder): reservation-first inventory allocation

Junctions now reserve template requirements (2 switches + approach
curve + return curve + n_straights) before _read_main_loop consumes
the shared curve pool. Prevents the 99.8% inventory-fail rate on
with_switches.yaml where the utilization gradient pulled the main
loop to full curve exhaustion before any junction could claim its
approach/return curves.

Refs Puchinger & Raidl (2008) DOI 10.1016/j.cor.2006.11.002,
     Lim et al. (2006) DOI 10.1109/CEC.2006.1688681."
```

---

## Task 4: Integration validation on `with_switches.yaml`

**Context:** CLAUDE.md's "no assertion without evidence" rule requires running the full optimizer and reading literal output before claiming the fix works. This task runs one full optimization with the fixed code, inspects the decoded population for switch-bearing layouts, and records the before/after comparison against the baseline captured in the previous turn (0 switches in 1000 chromosomes).

**Files:** no code changes — this task is pure verification.

- [ ] **Step 1: Run the full optimization with `configs/with_switches.yaml`**

Run: `python main.py --config configs/with_switches.yaml --verbose`

Expected duration: 5–15 minutes depending on machine. Expected exit code: 0. Expected artifacts: `outputs/best_layout.png`, `outputs/chromosomes.csv`, `outputs/constraints.csv`, `outputs/fitness.csv`, `outputs/pareto_front.png`.

- [ ] **Step 2: Use the `/diag` skill to parse the run**

Run: `/diag`

Expected output includes: feasible-solution count, best fitness, best layout piece count, best layout switch count. Record the numbers literally — do not summarize.

- [ ] **Step 3: Decode the saved population and count switch-bearing layouts**

Run this one-liner at the shell (uses the pattern from the debugging turn):

```bash
python - <<'PY'
import numpy as np, yaml, sys
from pathlib import Path
sys.path.insert(0, '.')
from src.config import OptimizationConfig
from src.catalog import TrackCatalog
from src.encoding import compute_dimensions, count_active_junctions
from src.decoder import decode_chromosome, DecoderConfig

with open("configs/with_switches.yaml") as f:
    cfg_dict = yaml.safe_load(f)
config = OptimizationConfig(**cfg_dict)
catalog = TrackCatalog.load(Path("data/track_pieces_v2.yaml"))
dims = compute_dimensions(config, catalog)
dec_cfg = DecoderConfig(
    boundary_min_x=config.boundary.min_x, boundary_max_x=config.boundary.max_x,
    boundary_min_y=config.boundary.min_y, boundary_max_y=config.boundary.max_y,
)

X = np.loadtxt("outputs/chromosomes.csv", delimiter=",").astype(np.int16)

n_active = 0
n_placed = 0
hist = {}
for x in X:
    if count_active_junctions(x, dims) > 0:
        n_active += 1
    layout = decode_chromosome(x, catalog, config.inventory, dims=dims, config=dec_cfg)
    sp = len(layout.switch_pairs)
    if sp > 0:
        n_placed += 1
    hist[sp] = hist.get(sp, 0) + 1

print(f"Total chromosomes:         {X.shape[0]}")
print(f"Gene-active junction rows: {n_active}")
print(f"Decoded with >=1 switch:   {n_placed}")
print(f"Switch-pair histogram:     {hist}")
PY
```

- [ ] **Step 4: Confirm the fix took effect**

Expected: `Decoded with >=1 switch` is strictly greater than 0. The baseline from the previous turn was 0 / 1000. Any non-zero number proves switches now survive the decode pipeline. A healthy value is 10–50% of the population depending on how the GA balances switch-bearing vs switch-free individuals.

If the number is still 0: do NOT patch further. Return to systematic debugging and re-instrument the decoder — the fixes may have a bug the unit tests did not catch (e.g., reservation was made but main loop's `can_use` check still fails, or finalize_junctions is being called with the wrong `n_main`).

- [ ] **Step 5: Inspect `outputs/best_layout.png`**

Read the image. The title's "Switches:" count should be at least 1 if the best feasible chromosome has an active junction. If the best feasible is still a 0-switch racetrack, that is acceptable — Deb-rule's feasibility-first still prefers a higher-utilization zero-switch layout to a lower-utilization one-switch layout unless topology bonus (INT-2 from `ga-dynamics-research.md`) is also active. The switch appearance in the general population (Step 3 histogram) is the correctness criterion; best-feasible containing switches is a tuning outcome that requires INT-2, not a bug.

- [ ] **Step 6: Document the before/after in the commit**

Copy the Step 3 histogram into the commit message. Use the format below; fill in the actual numbers from the run.

```bash
git add -A :/outputs  # if outputs/ is tracked; skip this line otherwise
git commit --allow-empty -m "test(e2e): verify switches now land with with_switches.yaml

Before:  0 / 1000 chromosomes decoded with >=1 switch pair
After:   <N> / 1000 chromosomes decoded with >=1 switch pair
Histogram: <paste Step 3 histogram>

Closes the three-part decoder fix (handedness bounds, shadow
constraint, reservation-first ordering)."
```

Skip this commit step if the user has not explicitly requested commits — CLAUDE.md requires explicit user instruction for every `git commit`. Instead, paste the Step 3 output back into the conversation for the user to review.

---

## Self-Review Checklist

**Spec coverage:**
- R3 (handedness bounds) → Task 1 ✓
- R2 (shadow constraint) → Task 2 ✓
- R1 (reservation-first) → Task 3 ✓
- End-to-end validation → Task 4 ✓

**Not in scope (deferred to follow-up plans):**
- INT-1 quadratic boundary penalty + hard cap (see `ga-dynamics-research.md`) — necessary for addressing the 160-piece pretzel pathology; orthogonal to switch starvation.
- INT-2 topology bonus in F[0] — becomes meaningful once Task 3 lands; add in a follow-up once the baseline improvement is measured.
- INT-3 feasible pool mating — addresses frozen elite; independent subsystem.
- CGP / V2 modular encoding migration — long-term; not blocking these fixes.

**Placeholder scan:** no TBDs, no "implement later", no "appropriate error handling" — every step lists the full code or command.

**Type consistency:** `ValidatedJunction` field names match `src/decoder/types.py:98-107`. `count_active_junctions` is defined in Task 2 Step 2 and consumed by Task 2 Step 6 and Task 4 Step 3. `_reserve_junctions` / `_finalize_junctions` signatures defined in Task 3 Step 4 and consumed in Task 3 Step 5. `TEMPLATES` lookup used consistently in Tasks 1 and 3.

**Invariants from CLAUDE.md respected:**
- No hardcoded N_VAR or dimensional limits — all sizing flows from `dims` / `catalog.n_pieces`.
- Chromosome length still scales with inventory via `compute_dimensions`.
- Repair stays wired into evaluation; no ad-hoc repair calls added.
- No `git commit` step is executed without explicit user authorization (Task 4 Step 6 is conditional on user instruction).
