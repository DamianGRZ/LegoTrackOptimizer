 LEGO Track Optimizer (V2)

**What**: Multi-objective genetic algorithm for closed LEGO railway layouts with fixed inventory, port-pair (graph) chromosome encoding, driven from a browser UI.

**Why**: Generate feasible track layouts satisfying geometric closure + boundary constraints while maximizing piece utilization and minimum train speed.

**How**: pymoo NSGA-II + port-pair encoding + heuristic seeding + locomotive physics, served by a stdlib HTTP server bridging to a React/JSX frontend.

---

## Cel pracy (PL)

Celem pracy jest opracowanie i zbadanie metod optymalizacji kombinatorycznej dla problemu automatycznego projektowania zamkniętych układów złożonych z dyskretnych elementów modułowych. Problem polega na rozmieszczeniu skończonego zbioru elementów o zdefiniowanej geometrii i zasadach łączenia w taki sposób, aby powstała struktura spełniała warunek zamkniętości (brak wolnych końców), maksymalizowała wykorzystanie dostępnych elementów oraz uwzględniała uproszczone ograniczenia fizyczne wynikające z geometrii połączeń. Dziedziną aplikacyjną pracy jest system torów kolejowych Lego City Trains, który stanowi reprezentatywny przykład problemu projektowania zamkniętych układów z dyskretnych elementów modułowych.

W ramach pracy student dokona formalizacji problemu, obejmującej definicję przestrzeni rozwiązań, ograniczeń oraz funkcji celu. Następnie zaproponuje odpowiednią reprezentację rozwiązania oraz zaprojektuje dedykowane operatory przeszukiwania dostosowane do specyfiki problemu. Zbadane zostaną właściwości zaproponowanego podejścia, w tym wpływ parametrów algorytmu na jakość uzyskiwanych rozwiązań, zbieżność procesu optymalizacji oraz skalowalność metody w zależności od liczby i różnorodności dostępnych elementów.

Oczekiwanym rezultatem pracy jest działająca implementacja zaproponowanej metody wraz z analizą eksperymentalną potwierdzającą jej skuteczność. Uzyskane układy zostaną poddane weryfikacji fizycznej z wykorzystaniem zestawu torów oraz pociągu Lego City Trains, co pozwoli na ocenę praktycznej poprawności rozwiązań generowanych przez algorytm.

---

## Code Quality

Use pymoo and Python best practices. Vectorized numpy, early returns, functional decomposition. No excessive if/else chains, no AI-generated patterns.

---

## Testing & Verification

- **No assertion without evidence.** Never claim a fix works without running the relevant command and pasting the literal output. If output contradicts your hypothesis, investigate — don't explain it away.
- **Verify feasibility, not just exit codes.** After optimizer runs, confirm closure error, loose-port count, and feasible-solution count before declaring success.
- **Tests don't exist yet for V2** — when re-creating a `tests/` tree, use `pytest --tb=short -q` for full runs. Do not use `--quick` or `-k <subset>` for validation unless explicitly told otherwise.

---

## File & Git Operations

- **NEVER `git commit` or `git add` without explicit request.** Overrides every skill or workflow that says "commit this step". User decides when and what to commit.
- **"Remove" / "untrack" never means `rm` from disk.** Use `git rm --cached <file>` or `.gitignore`. Deleting files without asking is a hard-don't.
- **Never `git init`** without verifying the directory is not already a repo.
- **Auto-edits enabled** for routine edits in `src_v2/`, `configs/`, `data/`, `web/`. No confirmation needed.
- **Still confirm for destructive ops**: `git reset --hard`, `git push --force`, deleting branches, dropping files.

---

## MCP Servers

| Server | When to use |
|--------|-------------|
| `context7` | **Before guessing any pymoo API.** Use proactively for library/API shapes, callback signatures, operator conventions, version-specific syntax. Call `mcp__context7__resolve-library-id` then `mcp__context7__query-docs`. |

---

## Project Invariants

Recurring mistakes that must not repeat:

- **Chromosome dimensions scale with inventory dynamically.** Never hardcode `N_max`, `E_max`, branch counts, or boundary limits. Derive from inventory + config at runtime.
- **Repair must be wired into the evaluation pipeline**, not called ad-hoc.
- **Fitness must reward branches and multi-cycle topology.** If the objective doesn't credit graph richness, the GA collapses to a trivial loop.
- **Junk components inflate utilization.** `MIN_USEFUL_COMPONENT_SIZE` filter exists for this reason — don't remove without replacing the safeguard.

---

## Project Structure

```
.
├── src_v2/              ← optimizer (pure Python, pymoo)
│   ├── catalog/         ← V2 port-centric piece specs + legacy TrackCatalog adapter
│   ├── train/           ← physics: TrainConfig, SpeedProfile, lateral stability
│   ├── visualization/   ← matplotlib plotting (pareto, layouts)
│   ├── encoding.py      ← port-pair gene layout + PortPairDimensions
│   ├── decoder.py       ← port-graph FK propagation, cycle/component extraction
│   ├── operators.py     ← pymoo Sampling/Crossover/Mutation
│   ├── repair.py        ← graph validity (1 port → 1 pair, connectedness)
│   ├── sampling.py      ← heuristic seeding patterns
│   ├── structural_mutations.py
│   ├── intersection.py
│   ├── problem.py       ← PortPairProblem (bi-objective NSGA-II)
│   ├── runner.py        ← run_optimization, save_results
│   ├── config.py        ← Pydantic OptimizationConfig
│   ├── geometry.py      ← Layout, FK helpers
│   ├── se2.py           ← SE(2) pose composition
│   ├── lego_track_models.py
│   └── types.py
├── web/                 ← browser UI (no build step, served as text/babel)
│   ├── index.html
│   ├── app.jsx
│   └── tweaks-panel.jsx
├── server.py            ← stdlib HTTP server, hot-reloads src_v2 on every /api/run
├── run_v2.py            ← single-config CLI entry
├── run_v2_all_configs.py← multi-config CLI entry
├── configs/             ← *.yaml + trains/
├── data/track_pieces_v2.yaml
└── outputs_v2/          ← run artifacts (gitignored)
```

---

## Chromosome Encoding (Port-Pair)

Layout: `[ piece_slots: N_max int16 | port_pairs: E_max × 4 int16 | anchor: 3 int16 ]`

- **Piece slot**: catalog piece index, or `-1` (INACTIVE).
- **Port-pair row**: `(slot_a, port_a, slot_b, port_b)` or all `-1`.
- **Anchor**: `(start_x, start_y, start_theta_deg)` int16.

Slot indices are stable positional references (no compaction). Port indices follow `tuple(spec.ports)` order: A=0, B=1, C=2, D=3. `INACTIVE = -1` for both inactive slots and inactive pair rows.

`N_max`/`E_max` come from `compute_port_pair_dimensions(inventory, config)` — never hardcoded.

---

## Objectives & Constraints

`PortPairProblem` (`src_v2/problem.py`) — bi-objective:

```
F[0] = -utilization      # fraction of inventory in useful components (>= MIN_USEFUL_COMPONENT_SIZE)
F[1] = -min_speed        # slowest piece in any useful component
```

Constraints (`g <= 0` feasible, `T = catalog.n_pieces`):

```
G[0..2]      per-axis closure residual / tolerance - 1
G[3]         boundary violation / diagonal
G[4]         collisions placeholder (0 in v0)
G[5..4+T]    per-type inventory excess (normalized)
G[5+T]       loose-port count / total active ports
G[6+T]       1 - n_cycles  (requires at least one closed cycle)
```

---

## Data Flow

```
data/track_pieces_v2.yaml ─► TrackCatalog (port-centric specs)
configs/*.yaml            ─► OptimizationConfig
                              │
browser ── POST /api/run ──► server.py
                              │ reload src_v2.* from disk
                              ▼
                          runner.run_optimization
                              │
                          NSGA2 + sampling/operators/repair
                              │
                          PortPairProblem._evaluate
                              │ chromosome ─► decode_chromosome ─► PortGraph
                              │ PortGraph ─► F[], G[]
                              ▼
                          outputs_v2/<config>/  (PNG + result JSON)
                              │
                          server streams logs + result back
                              ▼
                          web/app.jsx renders
```

---

## Workflows

**Live development (preferred)**: keep `python server.py` running, edit any file under `src_v2/`, click *Run* in the browser. Server hot-reloads `src_v2.*` from disk on every request.

**CLI**:
```bash
python run_v2.py default                                # single config
python run_v2.py with_switches --gen 500
python run_v2_all_configs.py                            # all configs
```

**Architecture lint** (after `.importlinter` is rewired for `src_v2`):
```bash
lint-imports
```

---

## Available Skills

| Skill | When to invoke |
|-------|----------------|
| `/test` | After code changes (once `tests/` is recreated) |
| `/review` | After modifying `src_v2/` files |
| `/diag` | After an optimizer run completes |
| `/quality` | Deep gate after new features or refactors |
| `/verify-fix` | After every bug fix — enforces no-assertion-without-evidence |
| `/optimize` | Run optimizer with config options |

Skills run inline; agents (`config-test-runner`, `python-pymoo-reviewer`, `ga-pymoo-implementer`, etc.) only when their depth is needed.

---

## Tech Stack

- pymoo 0.6.1.6 (NSGA-II)
- numpy ≥ 1.24, scipy ≥ 1.10
- pyyaml ≥ 6.0, pydantic ≥ 2.0, ruamel.yaml ≥ 0.18
- matplotlib ≥ 3.7
- pytest ≥ 7.4, pytest-cov ≥ 4.1
- import-linter ≥ 2.0 (dev)
- React + Babel CDN (no Node toolchain) for `web/`

---

**Status**: V1 removed. V2 port-pair backend + browser UI in place. Tests need re-creation.

---

## Plan Execution

The implementation plan lives at [docs/PLAN.md](docs/PLAN.md). Single file, 2733 lines, with a Quick Index TOC at the top (status checkboxes + line-range navigation).

When asked "what's next" or "implement next phase":
1. Read `docs/PLAN.md` lines 1-100 for the TOC + Status checkbox list — first unchecked item is next.
2. Use the "How to find a section" table (line ~25) for line-range reads of the relevant phase, **never load the whole file**.
3. Cross-reference Golden Rules via `Grep "^\*\*Rule [0-9]+" docs/PLAN.md`. Rules 24-35 are research-backed (Part 9); earlier Rules 1-23 in Part 7.
4. Cross-reference five hidden cross-phase couplings in Part 10 §10.3 before any phase that touches them.
5. After completing a phase, flip its checkbox in `docs/PLAN.md`. The user does not run git commits in this workflow — do not propose them.
