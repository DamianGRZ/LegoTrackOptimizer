# Four regions of integer genes, and why they earn their place

**The chromosome package is the first Tier-2 package and the locus of this thesis's encoding novelty, so it needs more than a skeleton: it needs a defense.** The canonical biased random-key genetic algorithm (BRKGA) of Gonçalves and Resende (2011, DOI 10.1007/s10732-010-9143-1) defines a single flat vector of real keys drawn from `[0, 1)`, and the 2024 BRKGA review of Londe, Pessoa, Andrade and Resende (DOI 10.1016/j.ejor.2024.03.030) — covering 150+ applications and every named variant from BRKGA-MP-IPR to BRKGA-QL — reports **no published BRKGA with integer-valued keys and no published BRKGA with explicitly delimited multi-region chromosomes under region-specific operators**. This thesis therefore proposes a scheme that is not strictly BRKGA. It is a **biased integer-keyed multi-region genetic algorithm** that borrows BRKGA's elite-biased crossover (ρ=0.7) and mutant-replacement semantics while replacing the `[0,1)`-key sort-decoder with a construction-based turtle-kinematics decoder that reads four semantically distinct integer regions jointly. The four regions — main loop, switch mask, branch slots, crossing overlay — are defended below on first principles, grounded against the closest published precedents (Zhao et al. 2023's implicit two-region float split; the multi-depot VRP multi-list GA tradition; Zawidzki's Truss-Z modular layout EA), and tied concretely to a frozen `ChromosomeLayout`, a concrete `derive_bounds()` that pymoo 0.6.1.6 consumes, and a three-column table of which constraints are enforced where. The severed branch-slot genotype-to-phenotype path is diagnosed and reconnected. The crossing overlay is defended, not eliminated, because its cost is ≤30 genes and its topology expressiveness is essential.

## Random keys are the wrong abstraction; integer segments are the right one

Bean's 1994 paper (ORSA J Computing 6(2):154–160, DOI 10.1287/ijoc.6.2.154) introduced random keys as a **feasibility trick for permutation problems**: sort the keys, take the argsort, and any unconstrained real-valued crossover produces a valid permutation. Gonçalves and Resende (2011) generalized this to a **problem-independent framework**: one flat `[0,1)^n` vector, one problem-specific decoder, biased uniform crossover with elite-probability ρ≈0.7, and mutant immigrants at ~10–15% per generation. The canonical chromosome bounds are trivially `xl=0, xu=1` per gene, which is why pymoo and every other BRKGA implementation treats bounds as an afterthought. **That is exactly the simplification this thesis cannot afford.**

The turtle-kinematics decoder constructs layouts by chaining piece connectors in world space. It needs to know, for each step of construction, which piece type to place, which switch branch to follow, whether this step should declare a siding or a crossing — **semantically distinct decisions with semantically distinct domains**. Mapping all of these through a uniform `[0,1)` key and a sort-based or threshold-based decoder would force the decoder to internally partition the key stream into heterogeneous sub-decisions anyway, which is precisely the segmentation this report makes explicit at the chromosome level. Moving the segmentation upward has three concrete payoffs: **region-specific operators** (the segment-selective ALNS-style crossover from the research plan becomes trivially expressible), **region-specific bounds** (each gene's `xl`/`xu` reflects its semantic domain, not a uniform unit interval), and **region-specific validity invariants** (the pre-decode `validate_invariants()` function can reject malformed chromosomes before the expensive decoder runs).

The integer encoding departs from BRKGA in a second way. Because pymoo's integer pipeline is `IntegerRandomSampling()` + `SBX(vtype=float, repair=RoundingRepair())` + `PM(vtype=float, repair=RoundingRepair())` — confirmed against pymoo v0.6.1.6 docs at pymoo.org/customization/discrete.html — the internal population is still a `float64` matrix that is cast back to integers at the decode boundary. The `int16` choice in this report therefore governs **user-facing storage and arithmetic**, not pymoo's internal working matrix. This is a deliberate decision: 16-bit genes make the persisted chromosome format compact (1500 bytes per individual at the configuration below), they travel well across serialization boundaries, and they signal to reviewers that no gene will ever exceed ±32767 — a reassurance the decoder and operator packages can exploit as a precondition.

## The closest published precedent, and the residual gap that justifies novelty

The systematic 2024 BRKGA review identifies two partial precedents for multi-region encodings. **Zhao et al. 2023**, cited in the multi-station multi-robot task-allocation subsection of §3 of Londe et al.'s review, use a BRKGA chromosome whose single float value is split at the decimal point into two *de facto* regions — work-station index before the dot, robot index after — and introduce "double crossover" and "double mutation" operators that treat the two sides independently, plus a whole-chromosome elite re-optimization operator. **Chaves, Vianna, da Silva & Schenekemberg 2024 (BRKGA-QL for family TSP)** dedicate a single final gene as a meta-selector that picks one of five decoders. Both show that BRKGA practitioners have felt the pull toward segmentation; neither crosses the Rubicon of **explicit index-range regions under formally distinct operators**.

Outside BRKGA, the closest analogue is the **multi-depot vehicle routing GA tradition**, where a chromosome is a triple of sub-lists — customer permutation, depot assignment, vehicle assignment — each receiving a different crossover operator (PMX for the permutation, uniform for the assignments). Zhou et al. 2013 (Discrete Dyn Nat Soc, Article 325686, DOI 10.1155/2013/325686) is representative. These are non-BRKGA and use heterogeneous data types across sub-lists, but the pattern "one individual, multiple regions, region-specific operators, joint decoding" is exactly the thesis's pattern transplanted from routing to track layout. The structural-engineering analogue is **Zawidzki's Truss-Z work** (Zawidzki & Nishinari 2013 "Application of evolutionary algorithms for optimum layout of Truss-Z linkage in an environment with obstacles", Adv Eng Software 65:43–59, DOI 10.1016/j.advengsoft.2013.04.022; Zawidzki & Szklarski 2018, DOI 10.1016/j.asoc.2018.05.042), which evolves sequences of L/R modular truss units under construction-based feasibility — but in a **single-sequence chromosome without explicit regions**, because trusses have no analogue of switches, sidings, or crossings.

The table below summarizes the gap:

| Precedent | Encoding | Regions | Region-specific operators | Construction-based decoder | Distance from this thesis |
|---|---|---|---|---|---|
| Gonçalves & Resende 2011 | `[0,1)^n` reals | 1 | No | Optional (usually sort-based) | Large — real keys, no segmentation |
| Zhao et al. 2023 (via Londe 2024) | Float with decimal split | 2 (implicit) | Yes (double crossover/mutation) | No | Closest BRKGA analogue; still reals |
| Chaves et al. 2024 BRKGA-QL | `[0,1)^n` + meta-gene | 1 + 1 | No (uniform on all keys) | Variable (meta-gene selects) | Segmentation pattern but not operators |
| Multi-depot VRP GA (Zhou 2013) | Three heterogeneous sub-lists | 3 | Yes (PMX + uniform + uniform) | No | Closest general-GA analogue |
| Zawidzki Truss-Z 2013–2018 | Single integer sequence | 1 | No | Yes (L/R module chaining) | Closest construction-decoder analogue |
| **This thesis** | **Flat int16 multi-region** | **4 (explicit)** | **Yes (segment-selective)** | **Yes (turtle kinematics)** | **—** |

The residual gap makes the four-region integer chromosome **defensibly novel** as a synthesis of two traditions — BRKGA's biased-crossover population dynamics and the construction-decoder EAs of Truss-Z and L-system evolution (McCormack 1993; Hornby & Pollack 2002, Artif Life 8(3):223–246, DOI 10.1162/106454602320991837) — rather than as a new algorithm. Per Stanley and Miikkulainen's taxonomy (2003, Artif Life 9(2):93–130, DOI 10.1162/106454603322221487), the scheme sits on the **grammatical/developmental axis of artificial embryogeny**: the decoder is a grammar interpreter that reads the chromosome as a tape of typed instructions.

## Four regions are not three, and not five

The count of four is the smallest that captures the topology of closed LEGO railway layouts with switches, dead-end sidings, and crossings. Each region is independent in the formal sense that its **domain, its cardinality, and the operators that write to it** do not functionally depend on the other regions' values — only the decoder's joint reading does.

**Main loop.** A sequence of piece-type indices forming the primary closed traversal. Cardinality `max_loop_length` (a configuration constant chosen so the region comfortably exceeds the longest loop any feasible inventory could produce). Domain per gene: `[0, n_piece_types]` where the sentinel value `n_piece_types` means "no piece / end of loop". The sequence is decoded left-to-right by the turtle until either the region ends or the sentinel is reached.

**Switch mask.** One integer per switch encountered during main-loop decoding, selecting that switch's traversal mode. Domain `[0, 2]` in a three-port schema: 0 = through, 1 = diverging, 2 = branch-off-here (the main loop takes the through route; a siding attaches on the diverging route). The user memory notes that the current visualization renders only two of three ports; that artifact does not affect the encoding, which must distinguish three cases, because `through` alone cannot express the branch-off-here decision. The region's cardinality is `max_switches`, a bound on the count of switch pieces any feasible inventory permits; the actually-used prefix depends on how many switches the main loop consumes.

**Branch slots.** A rectangular region of size `max_branches × max_branch_length` storing up to `max_branches` dead-end sidings, each a sequence of piece-type indices terminated by the same `n_piece_types` sentinel used in the main loop. Addressing is by **switch-occurrence index within the main loop**: branch slot *k* supplies the diverging-route content for the *k*-th switch in the main loop whose switch-mask value equals 2 (branch-off-here). This is the fix to the severed genotype-to-phenotype path diagnosed below.

**Crossing overlay.** Up to `max_crossings` triples `(i, j, piece_type)` where `i < j ≤ max_loop_length` are main-loop gene offsets and `piece_type` is the crossing-piece catalog index. The overlay declares that the main-loop sequence passes through the same physical location at step `i` and step `j`, requiring a crossing piece (LEGO 4515, or a bespoke angled crossing) rather than two independent straights. The overlay cannot be merged into the main loop because the main loop is a sequence and crossings are pair-valued — a second region-kind entirely.

**Could three suffice?** Two merger attempts fail. Merging the crossing overlay into the main loop by inline-flagging a pair of indices requires either variable-length per-gene metadata (incompatible with pymoo's fixed-length array) or a spatial-reservation sentinel that would consume a piece-type slot and still not encode the other endpoint. Merging branch slots into the main loop with a terminator sentinel is more tempting — the decoder would encounter a switch, read its mask, and if mask == 2 consume the following stretch of main-loop genes as the branch until a terminator — but this conflates branch length with main-loop length against the pymoo fixed-length constraint and destroys the segment-selective crossover property, because main-loop and branch mutations would collide in the same address space. Three is too few.

**Could five be needed?** The two candidate fifth regions are an explicit **start-piece/orientation** gene and an explicit **inventory quota** region. Start-piece is unnecessary because the decoder anchors the first piece at the origin with a canonical heading; rotational equivalence is handled post-hoc at the problem layer. Inventory quotas are a constraint, not a decision — they belong in the problem's `G` vector (Deb 2000; Deb et al. 2002, DOI 10.1109/4235.996017 for NSGA-II's feasibility-first ranking) or in a repair operator, not in the chromosome. Four is sufficient.

## The exact int16 layout

The `ChromosomeConfig` captures all sizing decisions; the `ChromosomeLayout` is derived from it and frozen.

```python
@dataclass(frozen=True)
class ChromosomeConfig:
    n_piece_types: int           # from TrackCatalog.n_types
    max_loop_length: int         # sequence length bound for main loop
    max_switches: int            # cardinality of switch-mask region
    max_branches: int            # number of branch slots
    max_branch_length: int       # length of each branch slot
    max_crossings: int           # number of crossing triples

@dataclass(frozen=True)
class ChromosomeLayout:
    main_loop:        slice      # [0, max_loop_length)
    switch_mask:      slice      # [..., ... + max_switches)
    branch_slots:     slice      # [..., ... + max_branches * max_branch_length)
    crossing_overlay: slice      # [..., ... + 3 * max_crossings)
    total_length:     int

def derive_layout(cfg: ChromosomeConfig) -> ChromosomeLayout:
    o = 0
    ml = slice(o, o + cfg.max_loop_length); o = ml.stop
    sm = slice(o, o + cfg.max_switches);    o = sm.stop
    bs = slice(o, o + cfg.max_branches * cfg.max_branch_length); o = bs.stop
    co = slice(o, o + 3 * cfg.max_crossings); o = co.stop
    return ChromosomeLayout(ml, sm, bs, co, o)
```

Region domains, bounds, and constraints sit in the table below. The sentinel `S = n_piece_types` is reserved across both main-loop and branch-slot regions so the decoder can use a single termination test.

| Region | Offset | Width | Per-gene domain | Domain rationale | Constraints enforced at this region |
|---|---|---|---|---|---|
| main_loop | 0 | `max_loop_length` | `[0, n_piece_types]` | piece index or terminator `S` | length ≤ region size; decoder stops at `S` |
| switch_mask | `max_loop_length` | `max_switches` | `[0, 2]` | 0=through, 1=diverging, 2=branch-here | switch-route consistency (one gene per switch) |
| branch_slots | ... | `max_branches · max_branch_length` | `[0, n_piece_types]` | piece index or terminator `S` | per-branch length ≤ `max_branch_length` |
| crossing_overlay | ... | `3 · max_crossings` | `i, j ∈ [0, max_loop_length]; p ∈ [0, n_piece_types]` | pair of main-loop offsets + crossing-piece type | `i < j`; overlay may be blank (all zeros) |

## Chromosome bounds ownership, resolved

The catalog report declared that `chromosome_bounds` lives in the chromosome package; here is the concrete function that honors that handoff. It returns a `ChromosomeBounds` object of the shape pymoo's `Problem.__init__(xl=…, xu=…, vtype=int)` consumes directly.

```python
@dataclass(frozen=True)
class ChromosomeBounds:
    xl: np.ndarray   # int16, shape (total_length,)
    xu: np.ndarray   # int16, shape (total_length,)
    vtype: type = int

def derive_bounds(catalog: TrackCatalog,
                  cfg:     ChromosomeConfig,
                  layout:  ChromosomeLayout) -> ChromosomeBounds:
    xl = np.zeros(layout.total_length, dtype=np.int16)
    xu = np.zeros(layout.total_length, dtype=np.int16)
    S  = np.int16(cfg.n_piece_types)        # sentinel: one past last piece index

    xu[layout.main_loop]        = S          # piece index or terminator
    xu[layout.switch_mask]      = 2          # through / diverging / branch-here
    xu[layout.branch_slots]     = S          # piece index or terminator
    # crossing overlay: (i, j, piece_type) triples
    co = layout.crossing_overlay
    for k in range(cfg.max_crossings):
        base = co.start + 3 * k
        xu[base + 0] = cfg.max_loop_length   # i
        xu[base + 1] = cfg.max_loop_length   # j
        xu[base + 2] = S                     # crossing piece type
    return ChromosomeBounds(xl=xl, xu=xu, vtype=int)
```

The `vtype=int` kwarg is what pymoo 0.6.1.6 pairs with `IntegerRandomSampling()`, `SBX(repair=RoundingRepair())`, and `PM(repair=RoundingRepair())` per pymoo.org/customization/discrete.html. This avoids the dict-keyed mixed-variable interface, which is slower per-individual and which `problem.bounds()` does not support (pymoo GitHub issue #352). Note that pymoo will coerce the int16 arrays to float64 internally; int16 governs storage outside the GA loop, not inside it.

## Three-column table: who enforces which constraint

The construction-based decoder makes many constraints disappear by refusing to build infeasible layouts. The remaining constraints split cleanly between the chromosome's pre-decode invariants and the problem's post-decode constraint vector.

| Constraint | Enforced by | Rationale |
|---|---|---|
| main-loop length ≤ max_loop_length | chromosome (region size) | no gene can index past the region |
| branch length ≤ max_branch_length | chromosome (per-branch width) | the rectangular branch-slot region is bounded |
| switch-route consistency (one decision per switch) | chromosome (one switch-mask gene per switch occurrence) | multi-valued conflict is structurally impossible |
| gene values within per-region domain | chromosome (`validate_invariants`) + pymoo `xl`/`xu` | `RoundingRepair` keeps samples in-range |
| angular closure on the π/32 lattice | decoder (integer prefilter + full-precision closure test) | lattice-integer modular arithmetic + final-piece residual |
| positional closure ≤ 1e-6 studs, angular ≤ 1e-9 rad | decoder (final-piece selection or rejection) | the turtle's terminal pose must meet tolerance |
| branches attach only at switches | decoder (branch-extraction phase only fires when switch-mask gene = 2) | physical connectivity forced by construction |
| crossing piece geometry matches both main-loop poses | decoder (validates `(i, j)` pair poses against crossing piece axes) | infeasible → chromosome rejected → NaN objective |
| inventory: per-type count ≤ `max_occurrences[type]` | problem layer (inequality `G`) | requires post-decode piece census; decoder reports usage |
| no self-intersection outside declared crossings | problem layer (collision inequality `G`) | requires post-decode geometric sweep |
| closure residual = 0 | problem layer (equality `H` or feasibility-first on `G`) | decoder produces residual; feasibility-first ranking (Deb 2000) handles it |
| f1 piece utilization, f2 bottleneck speed | problem layer (objectives `F`) | computed from decoded layout |

This split makes explicit what had been implicit: **the chromosome is responsible only for structural bounds, the decoder for geometric realizability, the problem for inventory and collision**. Construction-based feasibility pushes the bulk of the constraint burden out of the GA's search space — which is the whole reason for the topology-level encoding.

## The branch slot is the hardest problem, and here is how it gets wired back up

The user-memory diagnosis is that branch encoding is severed at two independent points: (1) the decoder's branch-extraction phase scans only the already-built main loop and **ignores the branch-slot region entirely**, and (2) the BRANCH mutation operator **writes to an obsolete format that no reader consumes**. Both breaks share a common root: no specification of **how branch-slot index *k* maps to a specific switch in the main loop**. Without that mapping, the decoder has no way to consume the region, and the mutation operator cannot write to it coherently.

The fix is a single rule, fixed here as contract for both the decoder and the operators packages:

> **Branch slot *k* supplies the diverging-route content for the *k*-th switch occurrence encountered during main-loop decoding whose switch-mask gene equals 2 (`branch-here`). If the main loop contains fewer than *k*+1 `branch-here` switches, branch slot *k* is dead code and is ignored.**

Three consequences follow. First, the **switch-mask region and the branch-slots region are coupled in their reading order** — the decoder must walk the main loop and the switch-mask region in lockstep, incrementing a `branch_here_counter` each time it encounters a switch with mask value 2, and using that counter to index the branch-slots region. Second, the **BRANCH mutation operator writes to the branch-slots region at the slot index derived from this counting rule**, not to any earlier format. Third, **operators that mutate the switch-mask region can accidentally orphan or un-orphan branch slots** by flipping mask bits from 2 to 0/1 (or vice versa) — this is not a bug but a feature, because it provides a low-variance way to add or remove sidings without reshaping the branch-slots region.

The operator package is not permitted to co-read the branch-slots region and the switch-mask region for purposes other than this counting rule. Cross-region coupling is limited to that one place. Everything else — main-loop mutation, switch-mask flipping, per-branch-slot content mutation, crossing-overlay mutation — operates on a single region at a time, which is what the segment-selective ALNS-style crossover of the research plan needs.

## The crossing overlay earns its keep at N=10, not at N=0

The meta-plan flagged the crossing overlay as the weakest-defended region. On inspection it is the most topologically expressive one: without it, the main-loop sequence can only describe a **simple closed curve**. Real LEGO layouts frequently include figure-eights, diamond crossings, and layouts where two loops share a crossing piece — all of which require the chromosome to express that gene index *i* and gene index *j* pass through the same spatial location. Encoding this inside the main loop would require either unbounded-per-gene metadata or a sentinel that consumes a piece-type slot and still does not encode the paired endpoint.

The overlay's cost is trivial: at `max_crossings = 10` it occupies 30 int16 genes out of a typical 750, or **4% of the chromosome**. Its benefit is the entire figure-eight topology class.

**Is the overlay mutated or frozen?** Mutated. If the overlay were frozen after the first decode, the search would be unable to discover crossing placements de novo — they would have to be present in the initial population, which requires either seeding (out of scope for BRKGA's random initialization) or lucky mutation of a different region that happens to produce a crossing, which cannot happen because no other region has a triple-gene (i, j, p) structure. The operators package therefore owns a **crossing-overlay mutation** that samples `(i, j)` pairs uniformly from `{(i, j) : 0 ≤ i < j ≤ current_loop_length}` and chooses `p` from catalog crossing pieces. The decoder's obligation, in return, is to **reject the chromosome if the geometry does not align** — handing the problem layer a NaN objective or a constraint-violation penalty. This rejection is acceptable because the crossing overlay's per-mutation infeasibility rate is bounded by the geometric constraint's tightness, which in practice is high but not prohibitive for π/32-lattice pieces.

## Int16 adequacy, verified arithmetically

At the concrete configuration `n_piece_types = 30, max_loop_length = 100, max_switches = 20, max_branches = 20, max_branch_length = 30, max_crossings = 10`:

| Region | Width (genes) | Max gene value | Int16 OK? |
|---|---|---|---|
| main_loop | 100 | 30 (sentinel) | yes (≪32767) |
| switch_mask | 20 | 2 | yes |
| branch_slots | 20 × 30 = 600 | 30 (sentinel) | yes |
| crossing_overlay | 10 × 3 = 30 | 100 (max offset i or j) | yes |
| **total** | **750** | — | — |

Memory: **750 × 2 bytes = 1500 bytes per individual**. A population of 200 across 1000 generations carries a lifetime footprint of ~300 MB if every individual is persisted, but per-generation memory is only ~300 KB. The lifetime budget is a logging question, not a runtime one. Pymoo's internal float64 working matrix adds a 4× factor inside the GA loop (200 individuals × 750 genes × 8 bytes ≈ 1.2 MB), still negligible.

## The 120-order-of-magnitude compression claim, corrected

The user-memory figure is not wrong but is under-specified. The closed-form compression, in orders of magnitude, is

**ΔOOM = N · log₁₀(W · H · K)**

where `N` is the piece count, `W × H` is the stud-bounded build area, and `K = 64` is the angular resolution at ATOMIC_ANGLE = π/32. The catalog size `T` cancels because both encodings contain the `T^N` factor. Three scenarios sharpen the claim:

| Scenario | N | W × H | log₁₀(W·H·K) | ΔOOM |
|---|---|---|---|---|
| Small 30-piece, 96×96 baseplate area | 30 | 96 × 96 | 5.77 | **173** |
| Medium 60-piece, 200×200 studs | 60 | 200 × 200 | 6.41 | **385** |
| Large 100-piece, 400×400 studs | 100 | 400 × 400 | 7.01 | **701** |

Solving `ΔOOM = 120` yields `N ≈ 19–28` depending on area and angular discretization, i.e., a **small loop of about 20–30 pieces on a small area with coarser angular resolution**. The ~120 OOM figure is therefore a conservative lower bound that applies to the smallest layouts the thesis considers; the operating regime of 60–80-piece loops on 200×200-stud areas delivers **300–500 orders of magnitude** of compression. The defensible thesis statement is: *"Across the layout sizes studied (30 ≤ N ≤ 100 pieces on 96×96 to 400×400 stud areas), the topology-level encoding eliminates between 10¹⁷⁰ and 10⁷⁰⁰ representable states relative to a (stud, stud, π/32-bin, type)-quadruple flat encoding; the ~120-order compression often cited corresponds to the smallest-layout end of this range."* Note that both the flat and the topology counts over-count representable-but-infeasible states; the ratio is an order-of-magnitude estimate of representational compression, not of valid-solution compression.

## A worked instance

Configure `ChromosomeConfig(n_piece_types=22, max_loop_length=80, max_switches=15, max_branches=10, max_branch_length=20, max_crossings=6)`. Applying `derive_layout`:

- main_loop:        `slice(0, 80)`,   width 80
- switch_mask:      `slice(80, 95)`,  width 15
- branch_slots:     `slice(95, 295)`, width 200  (= 10 × 20)
- crossing_overlay: `slice(295, 313)`, width 18  (= 6 × 3)
- total_length:     **313 genes**, 626 bytes per individual

`derive_bounds(catalog, cfg, layout)` produces `xl = zeros(313)` and an `xu` that is 22 on every main-loop and branch-slot gene, 2 on every switch-mask gene, and alternates `(80, 80, 22)` across the 18 crossing-overlay genes. `vtype = int`. This is the exact pair consumed by pymoo's `Problem.__init__(n_var=313, n_obj=2, n_ieq_constr=k_collision+k_inventory, n_eq_constr=1, xl=xl, xu=xu, vtype=int)`. The single equality constraint is closure residual; the inequalities are collision and inventory.

## API handed to downstream packages

```python
def validate_invariants(x: np.ndarray, layout: ChromosomeLayout,
                        cfg: ChromosomeConfig) -> bool:
    """Pre-decode sanity check. Cheap; called per individual before decode().

    Checks:
      - shape == (layout.total_length,)
      - all genes within their region's xl/xu (redundant with pymoo's repair
        but necessary when chromosomes arrive from I/O or from seeding)
      - crossing-overlay triples satisfy i < j (reject reversed pairs early)
      - switch-mask length does not exceed count of switch-type pieces in
        the main loop's usable prefix (allowing sentinel early-termination)

    Does NOT check geometric realizability or inventory; those are the
    decoder's and the problem's jobs respectively.
    """
    ...

def segment_view(x: np.ndarray, layout: ChromosomeLayout,
                 region: str) -> np.ndarray:
    """Return a numpy view (not a copy) of the chromosome region named
    'main_loop' | 'switch_mask' | 'branch_slots' | 'crossing_overlay'.
    Writes through this view mutate the underlying chromosome.
    Used by operators for region-selective mutation and crossover."""
    return x[getattr(layout, region)]

def slice_for(layout: ChromosomeLayout, region: str) -> slice:
    return getattr(layout, region)
```

**Decoder (package 5)** reads all four regions; maintains turtle state; counts `branch_here` switches to index branch slots; validates crossing geometry; reports closure residual, per-type piece census, and f1/f2 ingredients; rejects geometrically infeasible chromosomes (e.g., crossing pose mismatch) with a documented NaN convention. **Operators (package 6)** access regions only via `segment_view`; the segment-selective ALNS crossover is validated by this package's region independence; the BRANCH mutation writes to `branch_slots[k*max_branch_length : (k+1)*max_branch_length]` at the slot index `k` chosen to match the intended main-loop `branch-here` switch; crossing-overlay mutation samples `(i, j, p)` triples with `i < j`. **Problem (package 7)** consumes the decoder's output plus inventory and collision constraints; never sees raw chromosomes. **Config/IO (package 8)** persists `ChromosomeConfig`, pins pymoo to `>=0.6.1,<0.7` for bounds-API stability, and serializes chromosomes as int16 bytes.

## Architectural decisions

| Decision | Choice | Primary source / rationale |
|---|---|---|
| Encoding value type | int16 (semantic) over float64 (pymoo internal) | Gonçalves & Resende 2011 is real-valued only; integer is a deliberate departure |
| Number of regions | 4 (main loop, switch mask, branch slots, crossing overlay) | Smallest count that expresses closed-loop + 3-port switches + sidings + crossings |
| Naming | "BRKGA-inspired biased integer-keyed GA", not "BRKGA" | Londe et al. 2024 survey confirms no integer-keyed BRKGA; Bean 1994 framing |
| `chromosome_bounds` owner | chromosome package | Catalog report deferred; derive_bounds implemented here |
| Branch-slot addressing | by `branch-here` switch-occurrence index within main loop | Fixes the severed genotype-to-phenotype path |
| Crossing overlay status | mutated, not frozen | Required for de-novo discovery of figure-eight topologies |
| pymoo variable type | `vtype=int` with `IntegerRandomSampling + SBX/PM + RoundingRepair` | pymoo 0.6.1.6 canonical integer pattern; avoids the slower dict-vars path |
| pymoo pin | `>=0.6.1,<0.7` | Bounds API stable across 0.6.x; 0.7 reserved for possible breaks |
| Sentinel convention | `S = n_piece_types` reserved as "no piece / terminator" in main_loop and branch_slots | Single termination test in the decoder |
| Crossing overlay encoding | `(i, j, p)` triples, 3 genes each | Piece type explicit; decoder validates geometric compatibility |

## What remains open, and to whom it is deferred

Three decisions remain. The **operator hyperparameters** — SBX η, PM η, per-region mutation rates, the segment-selective crossover's region-selection distribution — belong to the operators package, which must also decide whether the BRANCH operator is a dedicated mutation or a special case of branch-slot row mutation. The **decoder's turtle-state branching semantics** — specifically, how the turtle saves and restores pose when it enters a `branch-here` diverging route, walks the branch to its terminator, and returns to the main loop — belongs to the decoder package; this chromosome report fixes only the contract that a branch-here switch consumes exactly one branch-slot row, not the implementation. The **inventory-constraint enforcement location** is provisionally assigned to the problem layer as an inequality `G`, but could be moved to a chromosome-repair operator; the operators package decides based on whether repair dominates or inflates the search on a per-problem basis. None of these reshapes the four-region layout; all three hang off the API fixed above.