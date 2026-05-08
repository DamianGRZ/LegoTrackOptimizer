# Designing a multi-segment BRKGA chromosome package for train track optimization

**A flat int16 NumPy vector, partitioned into four semantically distinct regions via a frozen `SegmentMap` dataclass, provides a pymoo-compatible, memory-efficient chromosome encoding for LEGO train track layouts.** This design draws directly from BRKGA (Biased Random-Key Genetic Algorithm) literature, which has formalized multi-segment chromosomes combining sorting-based and threshold-based decoding across 250+ published applications. The four regions — main loop sequence, switch mask, branch slots, and crossing overlay — mirror the decoder's construction DAG, and the `SegmentMap` gives operators safe, zero-copy access to each region via Python 3.12 hashable slice objects.

---

## Random-key encoding eliminates infeasibility by construction

Bean (1994) introduced the random-key representation: each gene is a real number in **[0, 1)**, and `argsort(genes)` produces a valid permutation. Any standard crossover or mutation on real-valued vectors yields another real-valued vector, which always decodes to a valid permutation — no repair operators needed. This single insight, published in the *ORSA Journal on Computing* (DOI: 10.1287/ijoc.6.2.154), has accumulated ~2,500+ citations and spawned the entire BRKGA family.

Gonçalves & Resende (2011) extended this with **biased crossover**: one parent is always drawn from the elite set, and the offspring inherits each gene from the elite parent with probability **ρ > 0.5** (typically 0.7). This "double elitism" — elite copying plus biased inheritance — preserves good gene combinations while mutant injection maintains diversity (DOI: 10.1007/s10732-010-9143-1).

The BRKGA literature formally classifies three decoder types (Londe et al., 2025, *Handbook of Heuristics*, DOI: 10.1007/978-3-319-07153-4_30-2):

- **Permutation decoder** — `argsort(keys)` yields an ordering (for sequencing, routing, priority decisions)
- **Indicator decoder** — `keys >= threshold` yields binary flags (for open/close, on/off decisions)
- **Mixed decoder** — different chromosome segments use different strategies

The train track optimizer uses all three: sorting for main loop sequence and branch slots, thresholding for switch mask and crossing overlay, making it a textbook mixed-decoder BRKGA.

## Multi-segment chromosomes are well-documented in BRKGA applications

The BRKGA framework itself is **segment-agnostic** — it sees only a flat `vector<double>` of length `chromosome_size`. Segment boundaries are tracked entirely within the decoder. The BRKGA-MP-IPR framework (Andrade et al., 2021, DOI: 10.1016/j.ejor.2019.11.037) makes this explicit: the framework handles evolution mechanics while the user-implemented decoder interprets regions.

Published multi-segment examples include:

**Andrade et al. (2015), wireless backhaul network design** — the most explicitly documented case, using a **five-part chromosome**: sections 1–2 encode equipment deployment and parameterization (indicator-based), sections 3–4 establish evaluation order and equipment relations (permutation-based), and section 5 determines network tree depths (indicator-based). This directly parallels the train track chromosome's four regions (DOI: 10.1016/j.asoc.2015.04.016).

**Villicaña-Cervantes & Ibarra-Rojas (2022), COVID-19 lab location** — first genes are sorted for service point ordering (permutation), last genes are thresholded for service radius expansion (indicator). **Freitas et al. (2023), hub location routing** — chromosome divided by number of hubs, with sorted values determining routing order and first genes identifying hubs. **BRKGA-QL (Chaves et al., 2021)** — appends a gene at position `n+1` where `id = ⌈c[n+1] × k⌉` selects among `k` decoders, demonstrating that even decoder-selection can be gene-encoded.

The Londe et al. (2025) comprehensive review of 250+ BRKGA papers confirms: *"Sometimes, a problem has characteristics that make a pure indicator- or permutation-based representation inadequate. When this happens, the authors tend to mix those two, generally by partitioning the chromosome into two or more parts"* (DOI: 10.1016/j.ejor.2024.03.030).

## Sorting vs threshold decoding: choosing the right semantics per region

Each region in the train track chromosome uses the decoding strategy matched to its decision type:

**Region 1 — Main loop sequence (sorting-based).** The genes represent random keys for available track pieces. `np.argsort(keys)` yields a permutation that defines piece ordering within the closed main loop. Only relative ordering matters; absolute values are irrelevant. This is Bean's original application.

```python
main_keys = chromosome[smap.main_loop].astype(np.float32) / 32767.0
piece_order = np.argsort(main_keys)  # valid permutation, always feasible
```

**Region 2 — Switch mask (threshold-based).** Each gene corresponds to a potential switch position on the main loop. Values above the midpoint (**16383** for int16, equivalent to 0.5) activate that switch. Gene magnitude encodes "confidence" — 0.9 is a strong activation, 0.51 is marginal.

```python
switch_keys = chromosome[smap.switches]
active_switches = (switch_keys >= 16383).astype(np.int8)  # binary mask
```

**Region 3 — Branch slots (sorting-based).** Pre-allocated as `max_branches × max_branch_length` genes. For each active switch, the corresponding sub-block is decoded via argsort to determine the ordering of track pieces in that branch. Inactive switches' genes exist but are ignored by the decoder.

**Region 4 — Crossing overlay (threshold-based).** Binary flags determining where crossing pieces are placed over existing track. Decoded identically to the switch mask.

**Multi-class extension.** For categorical decisions (e.g., choosing among k track piece types), divide [0, 1) into k equal bins: `class_id = floor(gene * k)`. With int16, each class gets ~32768/k distinct levels — for k=4, that is ~8,192 levels per class, far more than sufficient.

## The int16-to-float conversion provides 4× memory savings with negligible precision loss

Standard BRKGA uses float64 genes in [0, 1). The int16 alternative maps **value / 32767 → [0.0, 1.0]**, providing **32,768 discrete levels** in 2 bytes versus 8 bytes for float64. No published paper specifically advocates int16 for random keys, but the engineering rationale is sound:

For **sorting**, only relative ordering matters. With 32,768 distinct values and typical chromosome lengths under 1,000, the probability of a tie between any two genes is ~1/32,768 ≈ 0.003%. For **thresholding**, the binary decision boundary at the midpoint (16383) is exact. The **4× memory reduction** matters for large populations: a population of 10,000 individuals with 500-gene chromosomes costs 40 MB in float64 but only 10 MB in int16.

pymoo stores decision variables as NumPy arrays in `Population.X` with shape `(pop_size, n_var)`. Crowding distance computation operates on the **objective space** (`F` matrix), not the decision space, so int16 in `X` does not affect survival selection. BRKGA's biased crossover copies genes directly — it works identically regardless of dtype since it never performs arithmetic on gene values, only selection. A custom sampler is required:

```python
from pymoo.core.sampling import Sampling

class Int16RandomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.random.randint(
            0, 32768, size=(n_samples, problem.n_var)
        ).astype(np.int16)
```

pymoo also supports injecting custom initial populations by passing a NumPy array or `Population` object as the `sampling` parameter to any algorithm constructor.

## Frozen dataclass SegmentMap with Python 3.12 hashable slices

**Slice objects became hashable in Python 3.12** (CPython PR #101264, documented in What's New in Python 3.12). This means a `@dataclass(frozen=True)` containing `slice` fields is automatically hashable — enabling use as dict keys for caching. The `SegmentMap` stores four `slice` objects, one per region, and provides both direct indexing (`chromosome[smap.main_loop]`) and named extraction.

```python
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

type GeneArray = npt.NDArray[np.int16]  # Python 3.12+ type alias

@dataclass(frozen=True, slots=True)
class SegmentMap:
    """Maps named chromosome regions to contiguous slices of a flat int16 array."""
    main_loop: slice   # sorting-based: piece ordering
    switches:  slice   # threshold-based: switch activation
    branches:  slice   # sorting-based: branch piece ordering
    crossings: slice   # threshold-based: crossing placement

    def __post_init__(self) -> None:
        regions = [self.main_loop, self.switches, self.branches, self.crossings]
        if regions[0].start != 0:
            raise ValueError(f"First region must start at 0, got {regions[0].start}")
        for i in range(1, len(regions)):
            if regions[i].start != regions[i - 1].stop:
                raise ValueError(
                    f"Non-contiguous at boundary {i}: "
                    f"{regions[i-1].stop} != {regions[i].start}"
                )
        for r in regions:
            if r.step is not None and r.step != 1:
                raise ValueError(f"Only step=1 slices allowed, got {r}")

    @property
    def total_length(self) -> int:
        return self.crossings.stop

    @property
    def region_names(self) -> tuple[str, ...]:
        return ("main_loop", "switches", "branches", "crossings")

    def extract(self, chromosome: GeneArray, region: str) -> GeneArray:
        """Return a VIEW of the named region from the chromosome."""
        return chromosome[getattr(self, region)]

    def region_length(self, region: str) -> int:
        s: slice = getattr(self, region)
        return s.stop - s.start

    @classmethod
    def from_lengths(cls, *, main_loop: int, switches: int,
                     branches: int, crossings: int) -> "SegmentMap":
        offset = 0
        def _next(length: int) -> slice:
            nonlocal offset
            s = slice(offset, offset + length)
            offset += length
            return s
        return cls(
            main_loop=_next(main_loop), switches=_next(switches),
            branches=_next(branches), crossings=_next(crossings),
        )
```

Validation runs in `__post_init__` without setting any attributes — it only raises exceptions on invalid input. Computed properties like `total_length` use `@property` to avoid the `object.__setattr__` workaround for frozen fields. The factory classmethod `from_lengths` computes contiguous slices from region sizes, eliminating manual offset arithmetic.

**NumPy view semantics are critical**: `chromosome[slice_obj]` always returns a **view** (zero-copy, shared memory), meaning in-place mutations by genetic operators propagate to the original chromosome. Advanced indexing (integer or boolean arrays) returns copies. Operators that need isolation must explicitly call `.copy()`.

## Chromosome dimensions computed dynamically from catalog

The total chromosome length follows a clear formula driven by the parts catalog:

```
total = main_loop_pieces + switch_positions + (max_branches × max_branch_length) + crossings
```

A `ChromosomeSpec` dataclass mediates between catalog properties and SegmentMap construction:

```python
@dataclass(frozen=True, slots=True)
class ChromosomeSpec:
    main_loop_length: int       # catalog.track_piece_count
    switches_length: int        # catalog.max_switch_positions
    branches_length: int        # max_branches × max_branch_length
    crossings_length: int       # catalog.max_crossings
    max_branches: int           # stored for decoder reshape
    max_branch_length: int      # stored for decoder reshape

    @property
    def total_length(self) -> int:
        return (self.main_loop_length + self.switches_length
                + self.branches_length + self.crossings_length)

    @classmethod
    def from_catalog(cls, catalog) -> "ChromosomeSpec":
        return cls(
            main_loop_length=catalog.track_piece_count,
            switches_length=catalog.max_switch_positions,
            branches_length=catalog.max_branches * catalog.max_branch_length,
            crossings_length=catalog.max_crossings,
            max_branches=catalog.max_branches,
            max_branch_length=catalog.max_branch_length,
        )

    def build_segment_map(self) -> SegmentMap:
        return SegmentMap.from_lengths(
            main_loop=self.main_loop_length, switches=self.switches_length,
            branches=self.branches_length, crossings=self.crossings_length,
        )
```

This eliminates magic numbers entirely. The `max_branches` and `max_branch_length` fields are preserved in `ChromosomeSpec` for the decoder, which must reshape the flat branch region into a 2D `(max_branches, max_branch_length)` view.

## Variable-length branches via pre-allocation and structured redundancy

Fixed-length chromosomes are a **fundamental BRKGA requirement**, not a workaround. The BRKGA-MP-IPR documentation states: *"All genes are of type float. Consequently, all chromosomes are fixed length arrays of float values"* — `chromosome_size` is set at construction and cannot change. pymoo imposes the same constraint: `Problem.n_var` is fixed, and the `Population.X` matrix has shape `(pop_size, n_var)`.

Pre-allocating `max_branches × max_branch_length` slots means inactive switches produce **neutral genes** — genes that undergo crossover and mutation but have no phenotypic effect. The evolutionary computation literature on neutral genes is nuanced. Igel and Toussaint (2003) showed that neutral mutations enable "movement inside a neutral set that could result in totally different exploration distributions, which might speedup convergence." The Hidden Genes GA (Abdelkhalik, 2013, *JOTA* 156:450–468) formalizes this: chromosomes have "effective" and "ineffective" segments, with tags controlling which genes participate in fitness evaluation.

Critically, the train track encoding's neutral genes are **structurally redundant** — they can become active if the switch mask changes. This is the beneficial form of redundancy, maintaining a "standing reserve" of genetic diversity. When a switch activates in a later generation, its branch genes already contain evolved values rather than random noise, because the biased crossover has been passing elite parent genes through them.

The decoder handles this cleanly: for each switch position, check the mask; if inactive, skip the corresponding branch sub-block entirely. The branch region is logically a 2D array `(max_branches, max_branch_length)` stored flat:

```python
branch_genes = chromosome[smap.branches]
branch_matrix = branch_genes.reshape(spec.max_branches, spec.max_branch_length)
for i, is_active in enumerate(active_switches):
    if is_active:
        branch_order = np.argsort(branch_matrix[i])
        # ... build branch from branch_order
```

## Region ordering mirrors decoder's construction DAG

The four regions are ordered to match the decoder's information flow, creating a strict DAG with no circular dependencies:

1. **Main loop** (Region 0) → decoded first; establishes the closed track loop and identifies potential switch positions
2. **Switch mask** (Region 1) → decoded second; reads main loop results to know which positions exist, then activates a subset
3. **Branch slots** (Region 2) → decoded third; reads active switches to know which sub-blocks matter, then orders pieces within each active branch
4. **Crossing overlay** (Region 3) → decoded last; reads the full layout (loop + branches) to determine valid crossing positions

This ordering matters primarily for **code clarity and maintainability**, not crossover linkage. BRKGA uses parameterized uniform crossover where each gene is independently inherited — adjacent genes are NOT more likely to travel together (unlike one-point or two-point crossover). However, region ordering does benefit **Implicit Path Relinking** in BRKGA-MP-IPR, where block-based interpolation between chromosomes perturbs contiguous genes together, making it beneficial when related genes are adjacent.

The decoder must be **deterministic** for a given chromosome. The BRKGA-MP-IPR guide warns: *"The decoder must produce the exact solution for the same chromosome. If the decoder cannot do it, we will see a substantial degradation in the BRKGA performance regarding convergence."*

## Initialization: mixed random and heuristic seeding

The BRKGA review confirms that heuristic seeding improves performance: *"Many authors attempt to introduce interesting structures into the initial population by injecting warm start solutions. This addition is shown to increase the performance and convergence of the method"* (Londe et al., 2025).

**Converting a known-good layout to a chromosome** requires inverting each decoder:

For **sorting regions** (main loop, branches): given a desired permutation `[3, 1, 2, 0]`, generate sorted random keys and assign them so `argsort` reproduces the permutation:

```python
def permutation_to_int16_keys(desired_perm: list[int]) -> GeneArray:
    n = len(desired_perm)
    keys = np.sort(np.random.randint(0, 32768, size=n))
    chromosome = np.empty(n, dtype=np.int16)
    for rank, position in enumerate(desired_perm):
        chromosome[position] = keys[rank]
    return chromosome
# argsort(chromosome) == desired_perm ✓
```

For **threshold regions** (switches, crossings): given a binary mask `[1, 0, 1, 1]`, assign values above or below the midpoint:

```python
def binary_mask_to_int16_keys(mask: list[int], midpoint: int = 16384) -> GeneArray:
    chromosome = np.empty(len(mask), dtype=np.int16)
    for i, bit in enumerate(mask):
        if bit:
            chromosome[i] = np.random.randint(midpoint, 32768)
        else:
            chromosome[i] = np.random.randint(0, midpoint)
    return chromosome
```

A typical strategy: **5–10% heuristic-seeded individuals** (converted from known good layouts) plus **90–95% uniform random**. The BRKGA-MP-IPR framework provides `set_initial_population()` for injection before `initialize()`, which fills remaining slots with random chromosomes. In pymoo, pass a `(n_individuals, n_var)` NumPy array or custom `Sampling` class as the `sampling` parameter.

## Conclusion

The `chromosome/` package design rests on three well-grounded foundations. First, **BRKGA's random-key encoding** guarantees feasibility for both ordering and binary decisions through sorting and thresholding — any crossover or mutation yields a valid chromosome. Second, **multi-segment chromosomes** are an established BRKGA pattern (five-part chromosomes appear in published work), with segment boundaries managed by the decoder rather than the framework. Third, **Python 3.12's hashable slices** enable a clean frozen dataclass design where `chromosome[smap.main_loop]` returns a zero-copy NumPy view suitable for in-place genetic operators.

The int16 representation is a novel engineering choice not found in existing BRKGA literature — all published work uses float64 in [0, 1). However, the **32,768 discrete levels** provide more than sufficient resolution for sorting (ties are vanishingly rare below ~1,000 genes) and thresholding (exact midpoint boundary), while delivering **4× memory savings** — meaningful for populations of thousands of individuals with hundreds of genes. The key architectural insight is that `ChromosomeSpec.from_catalog()` computes all dimensions dynamically, eliminating magic numbers and ensuring the chromosome adapts automatically when the track piece catalog changes.