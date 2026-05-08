# Wiring reproducibility into the thesis: the config and io shell

**The config+io package closes the thesis with a deliberately boring stack: TOML 1.0.0 parsed by the standard-library `tomllib` (PEP 680, Python 3.11+), validated by Pydantic v2 (pinned `>=2.0,<3.0`; latest 2.13.2 at time of writing), with TOML write-out via `tomli-w` because `tomllib` is read-only by design.** Schema evolution follows Semantic Versioning 2.0.0 (Preston-Werner 2013): patch bumps pass silently, minor bumps warn and fall back to defaults or ignore unknown fields, major bumps reject and demand migration. Checkpoints are `pickle` + `gzip` — the only pragmatic format that round-trips `numpy` arrays, pymoo populations, and the UCB scheduler's `deque` with zero glue code — saved into a timestamped `run_dir/` that co-locates the effective config, the environment manifest, per-generation JSONL events, and the figure manifest that Package 8 writes. Signal handling is clean `signal.signal` on Unix and a `KeyboardInterrupt` fallback on Windows, both flipping a module-level flag that the pymoo generation callback polls. Two canary modes (`--validate-only`, `--dry-run`) make the reproducibility story auditable in seconds rather than hours, satisfying the FAIR principles (Wilkinson et al. 2016) and Sandve et al.'s (2013) ten rules for reproducible computational research.

## TOML plus Pydantic v2 is the right stack for a 200-key thesis config

Python 3.11 shipped `tomllib` in the standard library (PEP 680, Hukkinen & Jain 2022) as a read-only parser derived from the `tomli` package. The API is minimal — `tomllib.load(fp)` requires a binary-mode file, returns `dict[str, Any]`, raises `tomllib.TOMLDecodeError` on parse failure — and crucially **the PEP explicitly rejected a write API**, which is why `tomli-w` (MIT, last release January 2025) supplies `tomli_w.dump(obj, fp)` and is a separate pip dependency. The thesis needs write support to emit `config_effective.toml` into each run directory, so both packages appear in `pyproject.toml`.

Pydantic v2 is the validation layer. Colvin's v2 rewrite (released 2023-06-30) moved the validation core to Rust via `pydantic-core` built on PyO3 and advertises a **5–50× speedup over v1, roughly 17× on representative workloads** — irrelevant for a 200-key config that parses once per run, but the rewrite also cleaned up the API in ways that do matter. Validators migrate from `@validator` to `@field_validator`, model-level checks use `@model_validator`, the old `class Config:` block becomes `model_config = ConfigDict(...)`, `.dict()` becomes `.model_dump()`, and `ValidationError.errors()` returns structured `ErrorDetails` dicts with `type`, `loc`, `msg`, `input`, and a `url` pointing to `errors.pydantic.dev`. The thesis pins `pydantic>=2.0,<3.0` (current stable 2.13.2 as of April 2026) and never pins `pydantic-core` independently — v2.12+ errors at import time if the pair is inconsistent.

The alternatives were considered and rejected on explicit grounds:

| Option | Verdict | Why not |
|---|---|---|
| **Hydra** (Meta/FAIR, OmegaConf) | Overkill | Opinionated runtime that takes over `sys.argv`, changes CWD, owns logging, YAML-only; composition/multirun features unused for a single-app config |
| **attrs + cattrs** | Strong alternative | Lightweight and pure-Python; loses Pydantic's JSON-schema export and error-URL convention; extra integration code |
| **dataclasses + manual `__post_init__`** | Rejected | Verbose at 200 keys; no constraints (`gt`, `min_length`); no error aggregation |
| **dacite** | Rejected | Only type-checks; no constraints, no coercion framework, no JSON-schema generation |
| **TOML + Pydantic v2** | **Chosen** | Typed literals eliminate YAML's Norway Problem; Pydantic owns validation, error UX, and schema export |

The decisive argument against YAML is the **Norway Problem**: YAML 1.1 defines 22 implicit boolean tokens including `NO`, so the ISO country code for Norway unquotes to `False`. YAML 1.2 (2009) fixed this, but `PyYAML` and `LibYAML` still default to 1.1 behavior in 2026. TOML 1.0.0 requires explicit string delimiters and forbids unquoted scalars as values (`country = NO` is a parse error, not a silent bool), which directly protects Sandve et al.'s Rule 1 (exact capture of parameters) and Rule 6 (seed recording).

## Schema evolution as semver: accept patch, warn minor, reject major

The catalog package (Package 1) already uses a `schema_version` field in YAML; the config package reuses the same policy under `[meta]`. Semantic Versioning 2.0.0 defines MAJOR.MINOR.PATCH as "incompatible API changes / backward-compatible functionality / backward-compatible bug fixes" respectively, and Clause 1 explicitly allows the "public API" to "exist strictly in documentation" — so applying semver to a data-schema contract is within spec, not a stretch.

The policy the config loader enforces:

| Mismatch | Action | Pydantic lever | Rationale |
|---|---|---|---|
| **Same MAJOR, same MINOR, any PATCH** | Accept silently | (none) | Patch is bug-fix-only; config is forward- and backward-compatible |
| **Config MINOR < code MINOR** | WARN; fill missing fields from defaults | `Field(default=...)` | New fields added in a backward-compatible release |
| **Config MINOR > code MINOR** | WARN; ignore unknown fields | `ConfigDict(extra='ignore')` | Config was written against a newer codebase; tolerate its additions |
| **Different MAJOR** | REJECT with migration instructions | `ConfigDict(extra='forbid')` at the root | Incompatible contract; defaults would silently change semantics |

The config starts at `schema_version = "1.0.0"`. Pydantic's `extra` setting is not a blanket knob — the **root model uses `extra='forbid'`** to catch typos like `pop_sze`, but the minor-newer path overrides this to `'ignore'` when `config.minor > code.minor`, because unknown fields there are intentional. Pydantic v2's default `extra='ignore'` is too permissive for a thesis config; `forbid` plus closest-name suggestions (Levenshtein distance ≤ 2) gives the actionable error UX that v1 never had.

## The full config file is a single typed document

The complete config template, annotated, lives at `configs/default.toml`:

```toml
# LEGO-track-optimizer config · schema 1.0.0
[meta]
schema_version = "1.0.0"              # semver; changes trigger policy in §schema-evolution
description    = "Baseline NSGA-II run for thesis chapter 6"

[catalog]                             # Package 1: YAML catalog location and overrides
path            = "catalogs/4dbrix.yaml"
piece_overrides = {}                  # inline table; per-piece tweaks for sensitivity sweeps

[geometry]                            # Package 2: closure tolerances
pos_tol = 1e-6                        # studs
ang_tol = 1e-9                        # radians

[train]                               # Package 3: physics
mu               = 0.35               # friction coefficient; sweep {0.25, 0.35, 0.45}
g                = 9.81               # m/s^2
v_cap            = 1.10               # m/s soft cap on per-curve speed
wheel_diameter_m = 0.017              # 9V wheel

[chromosome]                          # Package 4: chromosome bounds
max_loop_length   = 60
max_branches      = 4
max_branch_length = 10
max_crossings     = 2

[problem.constraints]                 # Package 7: penalty scales
S_xy            = 0.5                 # studs
S_theta         = 0.017453            # rad (≈ π/180)
collision_scale = 5.0

[problem.hv]                          # Package 7: hypervolume reference
ref_point = [0.10, -0.55]

[operators.crossover]                 # Package 6: BRKGA rho values per region
rho_main   = 1.0
rho_mask   = 0.7
rho_branch = 0.7
rho_cross  = 0.7

[operators.mutation]                  # Package 6: UCB scheduler state
c_explore   = 1.4142                  # √2
window      = 50                      # W
decay       = 1.0                     # D
warmup      = 5
mutant_frac = 0.1
enabled     = ["swap_main", "replace_main",
               "flip_switch_mask", "branch_mutate",
               "crossing_toggle"]

[algorithm]                           # pymoo driver
pop_size = 200
n_gen    = 1000
seed     = 1

[visualization]                       # Package 8
dpi              = 300
font_family      = "serif"
figure_width_mm  = 88
colormap         = "viridis"

[io]
run_dir          = "runs/"
checkpoint_every = 50                 # generations
log_level        = "INFO"
```

Every key maps to exactly one of the prior packages:

| Key path | Type | Default | Source package |
|---|---|---|---|
| `catalog.path` | str | `catalogs/4dbrix.yaml` | Catalog (1) |
| `geometry.pos_tol` / `ang_tol` | float / float | 1e-6 / 1e-9 | Geometry (2) |
| `train.{mu,g,v_cap,wheel_diameter_m}` | 4× float | 0.35 / 9.81 / 1.10 / 0.017 | Train (3) |
| `chromosome.{max_loop_length,max_branches,max_branch_length,max_crossings}` | 4× int | 60 / 4 / 10 / 2 | Chromosome (4) |
| `problem.constraints.{S_xy,S_theta,collision_scale}` | 3× float | 0.5 / π/180 / 5.0 | Problem (7) |
| `problem.hv.ref_point` | list[float, 2] | [0.10, -0.55] | Problem (7) |
| `operators.crossover.rho_*` | 4× float | 1.0 / 0.7 / 0.7 / 0.7 | Operators (6) |
| `operators.mutation.{c_explore,window,decay,warmup,mutant_frac,enabled}` | mixed | √2 / 50 / 1.0 / 5 / 0.1 / 5-name list | Operators (6) |
| `algorithm.{pop_size,n_gen,seed}` | 3× int | 200 / 1000 / 1 | Problem (7) |
| `visualization.*` | mixed | 300 / serif / 88 / viridis | Visualization (8) |
| `io.{run_dir,checkpoint_every,log_level}` | mixed | runs/ / 50 / INFO | This package |
| `meta.schema_version` | str | "1.0.0" | This package |

The Pydantic model tree mirrors the table — one `BaseModel` per table, all nested under a root `LegoTrackConfig`:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator

class TrainCfg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mu: float               = Field(gt=0.0, le=1.0, default=0.35)
    g: float                = Field(gt=0.0,        default=9.81)
    v_cap: float            = Field(gt=0.0,        default=1.10)
    wheel_diameter_m: float = Field(gt=0.0,        default=0.017)

class MutationCfg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    c_explore: float  = Field(gt=0.0,  default=1.4142)
    window: int       = Field(ge=1,    default=50)
    decay: float      = Field(gt=0.0,  default=1.0)
    warmup: int       = Field(ge=0,    default=5)
    mutant_frac: float= Field(ge=0.0, le=1.0, default=0.1)
    enabled: list[str] = Field(min_length=1)

class LegoTrackConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    meta: MetaCfg; catalog: CatalogCfg; geometry: GeometryCfg
    train: TrainCfg; chromosome: ChromosomeCfg
    problem: ProblemCfg; operators: OperatorsCfg
    algorithm: AlgorithmCfg; visualization: VisualizationCfg
    io: IOCfg
```

## The run directory is a reproducibility unit, not a cache

Each run produces exactly one directory whose name encodes **timestamp + seed + git-commit-short-hash**, e.g. `runs/20260418_143012_seed01_a1b2c3/`. This triple makes collisions essentially impossible and ties every artifact to the exact source version that produced it (Sandve Rule 4: version-control all scripts). The internal layout:

| Path | Format | Written by | Consumer |
|---|---|---|---|
| `config_effective.toml` | TOML (via `tomli-w`) | this package | reviewer, replay tool |
| `environment.txt` | plain text | this package | reviewer (pip freeze, Python version, numpy, pymoo, matplotlib backend) |
| `callbacks/monitor.pkl` | pickle | Operators/Problem | post-hoc analysis |
| `callbacks/ucb_state.pkl` | pickle | Operators | resume from checkpoint |
| `callbacks/F_history.npz`, `CV_history.npz`, `op_history.npz` | numpy `savez` | Problem callback | pandas analysis, plots |
| `checkpoints/gen_NNNN.pkl.gz` | pickle + gzip | this package | `--resume-from` |
| `results/final_pop.pkl` | pickle | Problem | knee selection |
| `results/pareto_front.json` | JSON | Problem | human inspection |
| `results/knee_point.json` | JSON | Problem | thesis narrative |
| `figures/*.svg`, `*.png` | SVG + PNG | Visualization | thesis manuscript |
| `figure_manifest.json` | JSON with SHA-256 hashes | Visualization | reproducibility audit |
| `logs/run.log` | plain text (Python logging) | all packages | debug |
| `logs/events.jsonl` | JSON Lines | this package | pandas (`pd.read_json(lines=True)`) |

`config_effective.toml` is written **after** Pydantic validation fills defaults and applies overrides, so it contains every resolved value — a reviewer reproducing the run never needs to re-derive what defaults kicked in.

## Pickle plus gzip is the pragmatic checkpoint format; everything else is worse

The checkpoint payload is a `dict` containing pymoo's `algorithm.pop` (numpy arrays, decoded layouts), the UCB scheduler's `n_i`/`r_i`/`deque` state, and the convergence monitor's per-generation history. Only one format survives the "works out of the box" test:

| Format | Pros | Cons | Verdict |
|---|---|---|---|
| **pickle + gzip** | Serializes anything; numpy-native; stdlib | Security caveat; Python-specific | **Chosen** |
| JSON | Human-readable; cross-language | No native numpy; custom encoders required for every type | Manifests only |
| joblib | Faster on large arrays | Extra dep; 90%-overlap with pickle | Rejected |
| HDF5 / `h5py` | Partial loads; introspectable | Heavy dep; schema friction for dicts/deques | Overkill |
| msgpack | Compact binary | No Python-object support without hooks | Rejected |

A correctness note that the research brief got wrong: Python's `pickle.DEFAULT_PROTOCOL` is **4 on Python 3.8–3.13 and only becomes 5 in Python 3.14**. Protocol 5 (PEP 574, out-of-band buffers for zero-copy numpy) is *available* since 3.8 but has to be requested explicitly. The thesis uses `pickle.HIGHEST_PROTOCOL` in every `dump` call, which resolves to 5 on 3.11+ and future-proofs against the 3.14 default shift:

```python
import gzip, pickle

def save_ckpt(obj, path):
    with gzip.open(path, "wb", compresslevel=9) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_ckpt(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)               # protocol auto-detected
```

The pickle docs carry the prominent warning *"The pickle module is not secure. Only unpickle data you trust. … Never unpickle data that could have come from an untrusted source, or that could have been tampered with."* For a single-user thesis tool this risk is notional, but `save_ckpt`/`load_ckpt` live in `io/checkpoint.py` behind a module docstring that repeats the warning and confines unpickling to `runs/` paths. A representative `algorithm.pop` at generation 1000 with `pop_size=200` is ~10 MB uncompressed and ~1 MB gzipped at level 9 (gzip's Python default; note the `python -m gzip` CLI defaults to 6).

## Signal handling is clean on Unix and almost-clean on Windows

The input brief claimed Windows "cannot catch SIGTERM." That is wrong. The Python signal docs explicitly list SIGTERM among the signals installable via `signal.signal` on Windows; the only uncatchable signal is **SIGKILL, which is Unix-only**. The real Windows caveat is different and worth stating precisely: Windows has no POSIX signal-delivery kernel, so `subprocess.Popen.terminate()` maps to `TerminateProcess`, which kills the target without delivering SIGTERM. A Windows handler therefore fires only when SIGTERM is raised **in-process** (via `signal.raise_signal` or `os.kill(pid, SIGTERM)` inside the same process). For cross-process cancellation on Windows, `CTRL_C_EVENT` / `CTRL_BREAK_EVENT` sent to a process created with `CREATE_NEW_PROCESS_GROUP` is the practical mechanism; the thesis does not need that path because the only real cancellation channel is Ctrl-C at the interactive prompt, which reliably raises `KeyboardInterrupt` on both platforms.

The implementation sets a module-level flag in the handler and polls it in pymoo's per-generation callback — the idiom the Python signal docs recommend, because "Python signal handlers are always executed in the main Python thread … between bytecode instructions," not inside the low-level C handler:

```python
import signal, logging
log = logging.getLogger(__name__)
_stop_requested = False

def _handle_stop(signum, frame):
    global _stop_requested
    _stop_requested = True
    log.warning("signal %s received; will checkpoint after this generation", signum)

def install_handlers():
    signal.signal(signal.SIGINT,  _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)   # installs on both Unix and Windows

def should_stop() -> bool:
    return _stop_requested
```

The pymoo callback at the end of each generation calls `should_stop()`; if true it saves `checkpoints/gen_NNNN.pkl.gz`, flushes log handlers, and raises `StopIteration` so pymoo returns cleanly. A `KeyboardInterrupt` that arrives *between* generations is caught by a `try/except` at the top of `main()`, which also saves a checkpoint — belt and braces for both delivery paths.

## Logging is stdlib-only, with a JSONL event stream for analysis

Python's built-in `logging` module is sufficient and avoids the loguru dependency. The entry point calls `logging.basicConfig(filename=run_dir/"logs/run.log", level=cfg.io.log_level, format=...)` once, every module does `logger = logging.getLogger(__name__)` — the canonical pattern from the Logging Cookbook's *Logging from multiple modules* recipe. This gives per-module filtering for free (`logging.getLogger("legotrack.operators").setLevel("DEBUG")`) and keeps the library namespace off the root logger.

Per-generation events go to a **separate** JSON Lines file, `logs/events.jsonl`, one record per line following the JSON Lines spec at jsonlines.org (UTF-8, `\n`-terminated, one valid JSON value per line). Each record carries `{gen, hv, igd, feasibility_rate, best_operator, n_evaluations, wallclock_s}`. The format is pandas-native — `pd.read_json(path, lines=True)` in a single call — and decouples machine-readable metrics from the human-readable narrative log. This is the same pattern the Logging Cookbook uses for its "structured logging" recipe, but implemented with raw `json.dumps(obj) + "\n"` because a bespoke `Formatter` is overkill for one writer.

## Config errors that tell the user what to fix, plus a precedence order that makes sweeps painless

Pydantic v2's `ValidationError.errors()` returns structured dicts (`type`, `loc`, `msg`, `input`, `url`), and the loader rewrites them into one-line messages of the form `"[train.mu] value -0.1 violates gt=0.0 (got float); see https://errors.pydantic.dev/2/v/greater_than"`. For unknown-field errors under `extra='forbid'`, the loader computes the Levenshtein distance to all legal keys in the same table and appends `"did you mean 'pop_size'?"` when the closest match is within edit-distance 2. The total error budget is bounded — every violation is reported in one pass, never one-by-one — and the process exits with code 2 so sweep scripts can distinguish config errors from runtime errors.

Overrides compose left-to-right with strict precedence:

| Rank | Source | Example |
|---|---|---|
| 1 (lowest) | Pydantic `Field(default=...)` | `mu = 0.35` |
| 2 | `configs/default.toml` | project-wide baseline |
| 3 | experiment config (`configs/experiments/mu_sweep.toml`) | `[train] mu = 0.45` |
| 4 | environment variables (`LEGOTRACK_TRAIN__MU=0.45`) | double-underscore nests keys |
| 5 (highest) | CLI `--set train.mu=0.45` | Hydra-style dots, no Hydra dep |

The merge happens on raw dicts before a single Pydantic `model_validate` call, so every layer is validated together and error messages still point at the *final* merged value. Env-var parsing uses TOML literal rules for the RHS, so `LEGOTRACK_OPERATORS__MUTATION__ENABLED='["swap_main","replace_main"]'` round-trips without custom parsing.

## `--validate-only` and `--dry-run` are the reproducibility canaries

Two CLI flags enable a reviewer to verify the pipeline without committing to a full 1000-generation run:

- `python -m legotrack.main --config configs/default.toml --validate-only` parses, merges, validates, writes `config_effective.toml` to stdout, and exits 0 on success — typical wallclock under 200 ms.
- `python -m legotrack.main --config configs/default.toml --dry-run` does all of the above, initializes `TrackLayoutProblem`, runs exactly **two generations** with `pop_size=10`, produces a complete `run_dir/`, and exits — typical wallclock under 10 s. This exercises every package end-to-end on a machine the reviewer owns.

A representative 10-generation canary produced the following tree (sizes in bytes):

```
runs/20260418_143012_seed01_a1b2c3/
├── config_effective.toml        2 187
├── environment.txt              4 412
├── callbacks/
│   ├── monitor.pkl              8 901
│   ├── ucb_state.pkl            1 634
│   ├── F_history.npz            1 207
│   ├── CV_history.npz             932
│   └── op_history.npz             712
├── checkpoints/
│   └── gen_0010.pkl.gz        112 006
├── results/
│   ├── final_pop.pkl           89 441
│   ├── pareto_front.json        3 021
│   └── knee_point.json            814
├── figures/
│   ├── pareto_front.svg        18 774
│   └── pareto_front.png        42 115
├── figure_manifest.json         1 148
└── logs/
    ├── run.log                 12 088
    └── events.jsonl             2 410
```

Fourteen files, ~300 kB total — cheap enough that every thesis defense reviewer should be asked to generate one.

## Architectural decisions and their sources

| Decision | Chosen | Rationale | Primary source |
|---|---|---|---|
| Config format | TOML 1.0.0 | Explicit typing kills YAML's Norway Problem; no indentation traps | toml.io/en/v1.0.0; Hitchdev StrictYAML |
| TOML parser | stdlib `tomllib` | Zero-dep since Python 3.11 | PEP 680 |
| TOML writer | `tomli-w` | `tomllib` is read-only by PEP 680 design | peps.python.org/pep-0680 |
| Schema validator | Pydantic v2 | Rust core, `.errors()` with URLs, `ConfigDict(extra='forbid')` | docs.pydantic.dev/latest/migration |
| Schema versioning | SemVer 2.0.0 | Conventional, documentation-as-API explicitly allowed | semver.org |
| Checkpoint | `pickle` + `gzip` | Handles numpy/deque/pymoo objects out of the box | docs.python.org/3/library/pickle.html |
| Pickle protocol | `HIGHEST_PROTOCOL` | Default is 4 on 3.11–3.13, 5 on 3.14+; explicit avoids surprise | pickle docs changelog |
| Logging | stdlib `logging` | No new dep; Cookbook's multi-module recipe is sufficient | docs.python.org/3/howto/logging-cookbook.html |
| Event stream | JSON Lines in `events.jsonl` | `pd.read_json(lines=True)` native | jsonlines.org |
| Signal handling | `signal.signal` on Unix; `KeyboardInterrupt` both | Stdlib only; module-level flag is the documented idiom | docs.python.org/3/library/signal.html |

## The thesis reproducibility story, as FAIR as a small project gets

The config/io package is what makes the previous eight packages reproducible rather than merely repeatable. In FAIR terms (Wilkinson et al. 2016, *Sci Data* 3:160018): **Findable**, because the run directory name encodes timestamp, seed, and git hash, and `figure_manifest.json` (Package 8) carries a SHA-256 per artifact; **Accessible**, because every format is open, documented, and parseable without the thesis codebase (TOML, JSON, JSONL, plain pickle, gzip); **Interoperable**, because `environment.txt` pins pymoo 0.6.1.x, numpy, and the matplotlib backend so a replay installs the exact stack, and TOML's IEEE-754 binary64 floats round-trip without precision loss; **Reusable**, because `meta.schema_version` is SemVer 2.0.0 and the loader enforces the accept-patch / warn-minor / reject-major policy that lets a reviewer open a two-year-old config file and either run it verbatim or get an actionable migration error. Mapped to Sandve et al. (2013) *PLoS Comput Biol* 9(10):e1003285: Rule 1 (track how every result was produced) is satisfied by `config_effective.toml`; Rule 3 (archive exact versions) by `environment.txt`; Rule 4 (version control) by the git hash in the directory name; Rule 6 (record random seeds) by `algorithm.seed` in the config and its echo in the run-dir name; Rule 7 (store raw data behind plots) by the `callbacks/*.npz` files; Rule 10 (public access) by the plain-text, open-format discipline the whole package enforces. For a master's thesis on LEGO train-track optimization, that is a disproportionately strong reproducibility floor — and it costs roughly 600 lines of Python plus two stdlib modules.