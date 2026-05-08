# Implementation Summary

All five planned changes from `analysis.md` are implemented. Five files changed; two new files added; one V1 dead file dropped from the import surface.

## Files changed

| File | Lines | What changed |
|---|---|---|
| `src_v2/problem.py` | small | Loose-port constraint now reports raw count (was: normalized by total active ports) — strict "no loose ports" feasibility. |
| `src_v2/operators.py` | +~430 | 5 new heuristic emitters (dogbone, double-oval-with-DOUBLE_CROSSOVER, ladder yard, classification yard, tri-lobe-crossing); `_COMPLEX_HEURISTIC_EMITTERS` set + `complex_emitter_quota=0.30` honored in `_do`; 3 new mutation sub-ops (`_split_with_switch`, `_insert_crossover_bridge`, `_merge_components_via_cross`) with rebalanced `OP_WEIGHTS`. |
| `src_v2/structural_mutations.py` | +~290 | 3 new functions: `split_with_switch`, `insert_crossover_bridge`, `merge_components_via_cross`. |
| `src_v2/runner.py` | small | Imports `CanonicalGraphDuplicates` + `TopologyNichedSurvival`; swaps the bool `eliminate_duplicates` for the canonical-hash eliminator instance; replaces `ConstrRankAndCrowding` with `TopologyNichedSurvival`. |
| `src_v2/canonical_hash.py` | NEW (267 LOC) | BFS-relabel canonicalization producing a 16-byte signature invariant to anchor pose, slot permutation, edge ordering, edge endpoint ordering. Also exports `_read_topology` reused by niched survival. |
| `src_v2/niched_survival.py` | NEW (186 LOC) | `topology_signature()` returns `(n_components, n_cycles, n_switches, n_crosses, n_crossovers)`. `TopologyNichedSurvival` runs NSGA-II rank+crowding once, then round-robins survivors across signature buckets so a single oval-bucket can never dominate. |

`src_v2/sampling.py` (V1 dead code) was **not** copied into the new tree — effectively deleted from the active source. Remove it from your local mounted folder when convenient.

## Knob defaults

- `heuristic_ratio` — unchanged at 0.20 (per your instruction).
- `complex_emitter_quota` — new, defaults to 0.30. Override at the call site if needed:
  ```python
  PortPairSampling(dims, catalog, config, heuristic_ratio=0.20, complex_emitter_quota=0.50)
  ```
- `OP_WEIGHTS` — local ops 0.90 total, structural ops 0.10 total (split_switch 0.04, crossover_bridge 0.04, merge_cross 0.02). Auto-normalized.

## Behavior changes you should expect

1. **Population diversity rises immediately.** `eliminate_duplicates` was previously a no-op for anchor-shifted clones; now they collapse. Expect lower effective pop on early gens, then recovery as random/structural ops repopulate.
2. **Switch & crossing pieces appear earlier in best-feasible.** Both the `complex_emitter_quota` seeding and the structural mutations push the GA toward switch-bearing layouts even from oval ancestors.
3. **Pareto front widens along the topology axis.** Topology niching means a marginally-dominated tri-lobe-crossing layout still survives if it's the only one in its bucket.
4. **Loose-port-bearing layouts are rejected harder.** Any individual with ≥1 loose port now reports a positive constraint violation on G[5+T]; the adaptive-epsilon handler will tolerate them early in the run, then squeeze them out by the perc_eps_until cutoff.

## Smoke test

All six edited / new files parse with balanced quotes, parens, brackets, braces. No import-time `NameError` paths visible from the static grep — every reference to a new symbol resolves.

I haven't run the actual NSGA-II loop because there's no Python runtime in this sandbox. Recommended first run:

```bash
python -m src_v2.runner --config configs/with_crossing.yaml --gens 30 --pop 80 --log-every 5
```

If the seed pool is empty (heuristic emitters all reject the inventory), you'll see a `Heuristic seeding: 0 patterns` log line — that means the inventory thresholds in the new emitters need relaxation. Easy to tune.

## What I deferred

- I left the unused top-level `ConstrRankAndCrowding` import in `runner.py` (harmless, still referenced in `niched_survival.py`).
- The `_emit_classification_yard` emitter intentionally produces loose-port topologies; with the new strict loose-port constraint these will be infeasible until repair closes them. Kept as a structural seed for the niching operator. If feasibility pressure is too aggressive in your run, drop it from `_HEURISTIC_EMITTERS`.
- `DOUBLE_CROSSOVER` port mapping in the new emitters and `insert_crossover_bridge` assumes the V2 catalog uses port names "A"/"B" for track1 and "C"/"D" for track2. This matches the convention in every existing emitter and the `_V2_ROUTE_PHYSICS` table. If your YAML uses different port names for `DOUBLE_CROSSOVER` specifically, those two code paths need name fixes — let me know.

## Files to copy back to your local repo

```
src_v2/problem.py
src_v2/operators.py
src_v2/structural_mutations.py
src_v2/runner.py
src_v2/canonical_hash.py    (new)
src_v2/niched_survival.py   (new)
```

And delete `src_v2/sampling.py` locally.
