# Infrastructure packages for a LEGO track layout optimizer

**The `config/` and `io/` packages form the foundation of a pymoo-based LEGO train track optimizer, handling everything from layered TOML configuration with Pydantic v2 validation through checkpoint serialization, graceful SIGTERM shutdown, and multi-format result export.** These packages must be rock-solid because every other module depends on them — a checkpoint corruption or config mutation mid-run can invalidate hours of computation. The design draws on proven patterns from DEAP, NEAT-Python, and MLflow's local artifact storage, adapted for pymoo v0.6+'s ask-and-tell architecture. What follows is a deep technical reference covering all 11 design topics with concrete code patterns ready for implementation.

---

## TOML for flat config, YAML for nested catalogs

The format choice is straightforward: **TOML handles run parameters and hyperparameters** (≤2 levels of nesting), while **YAML handles deeply nested track catalog data** with piece definitions, connection specs, and variant trees.

Python 3.11+ includes `tomllib` in the standard library (PEP 680), providing zero-dependency read support via just two functions: `tomllib.load(fp)` and `tomllib.loads(s)`. Files must be opened in binary mode (`"rb"`). For writing TOML back (frozen config snapshots), `tomli-w` v1.2+ (`pip install tomli-w`) provides `tomli_w.dump()` and `tomli_w.dumps()`, also requiring binary mode. The write library respects dict ordering but does not preserve comments.

TOML has three critical limitations for this project. First, **no null type** — the absence of a key is the only way to represent "no value," which complicates layered config merging since you cannot explicitly unset an override. Second, **no mixed-type arrays** in TOML v1.0.0 (the version `tomllib` targets). Third, **deep nesting becomes verbose** — `[[parent.child.grandchild]]` section headers repeat full paths, making catalog-like data unwieldy compared to YAML's indentation-based nesting.

A GA hyperparameter config in TOML stays clean and explicit:

```toml
[ga]
algorithm = "NSGA2"
pop_size = 100
n_gen = 200
seed = 42
crossover_prob = 0.9
mutation_eta = 15.0

[physics]
min_radius_cm = 40.0
max_grade_pct = 3.0
```

The track catalog, by contrast, demands YAML's depth — each piece has connections with gender, angle, and port identifiers; variants with direction; and compatibility lists referencing other pieces. `ruamel.yaml` (v0.18+) is preferred over PyYAML for its **round-trip preservation** of comments, block/flow style, and key ordering, which matters when the catalog is human-edited.

---

## Pydantic v2 BaseSettings delivers typed, immutable, validated config

The `pydantic-settings` package (v2.13+, installed separately from `pydantic` core via `pip install pydantic-settings`) provides `BaseSettings` with built-in support for environment variables, TOML files, CLI arguments, and custom sources. The critical configuration:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid",            # Reject unknown keys immediately
        frozen=True,               # Immutable after construction
        validate_default=True,     # Validate even defaults
        env_prefix="LEGO_",        # LEGO_GA__POP_SIZE=200
        env_nested_delimiter="__", # Nested access via double underscore
    )
```

**`frozen=True` is the most important setting** — it prevents accidental mutation of shared config during multi-hour optimization runs. Any assignment attempt raises `ValidationError`. Sub-configurations use `BaseModel` (not `BaseSettings`) with their own `frozen=True`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class GAConfig(BaseModel):
    model_config = {"frozen": True}
    algorithm: Literal["NSGA2", "NSGA3", "MOEAD"] = "NSGA2"
    pop_size: int = Field(default=100, ge=10, le=10000)
    n_gen: int = Field(default=200, ge=1, le=50000)
    seed: int | None = Field(default=42)
    crossover_prob: float = Field(default=0.9, ge=0.0, le=1.0)
    mutation_eta: float = Field(default=15.0, gt=0.0)

class PhysicsConfig(BaseModel):
    model_config = {"frozen": True}
    min_radius_cm: float = Field(default=40.0, gt=0.0)
    max_grade_pct: float = Field(default=3.0, ge=0.0, le=45.0)
```

Key Pydantic v2 differences from v1: `model_config = ConfigDict(...)` replaces inner `class Config`; `@model_validator(mode="after")` replaces `@root_validator`; `.model_dump()` replaces `.dict()`; `frozen=True` replaces `allow_mutation=False`; and `BaseSettings` lives in `pydantic_settings` rather than `pydantic`.

For TOML serialization, `model_dump()` returns Python types including `Path` objects and `None` values that TOML cannot represent. A sanitizer must convert `Path` to `str` and omit `None` keys before passing to `tomli_w.dump()`.

---

## Layered configuration with provenance tracking

The priority order — **base TOML → experiment TOML → CLI args → env vars** (highest priority last) — maps directly to `pydantic-settings`' built-in `settings_customise_sources` classmethod. The `TomlConfigSettingsSource` ships natively in `pydantic-settings` and supports multiple TOML files where later files override earlier ones.

```python
from pydantic_settings import (
    BaseSettings, SettingsConfigDict, TomlConfigSettingsSource,
    CliSettingsSource, PydanticBaseSettingsSource,
)

class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid", frozen=True, validate_default=True,
        env_prefix="LEGO_", env_nested_delimiter="__",
    )
    
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings,
                                    env_settings, dotenv_settings,
                                    file_secret_settings):
        return (
            CliSettingsSource(settings_cls, cli_parse_args=True),
            env_settings,
            TomlConfigSettingsSource(settings_cls,
                                     toml_file=Path("experiment_config.toml")),
            TomlConfigSettingsSource(settings_cls,
                                     toml_file=Path("base_config.toml")),
        )
```

The **first item in the returned tuple has the highest priority**. `CliSettingsSource` auto-generates CLI arguments from model fields (e.g., `--ga.pop_size 200`), eliminating the need for separate argparse or Click configuration. For richer CLI UX, Click can parse options first and pass them as `init_settings` kwargs.

For **provenance logging** (critical for reproducibility), load each source independently and compare values to the final resolved config. Record which source provided each parameter in a `provenance.json` alongside the frozen config. A `deep_merge()` utility handles recursive dict merging when manually composing layers:

```python
def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

---

## Checkpoint serialization captures full optimizer state

For exact restart of an NSGA-II run, a checkpoint must capture **12 distinct pieces of state**: the current generation number (`algorithm.n_iter`), total evaluation count (`algorithm.evaluator.n_eval`), the full `Population` object (X, F, G, CV, feasibility flags), the Pareto-optimal set (`algorithm.opt`), offspring (`algorithm.off`), Python RNG state (`random.getstate()`), NumPy legacy RNG state (`np.random.get_state()`), pymoo's Generator state (`algorithm.random_state`), the frozen config snapshot, metrics history (HV/IGD per generation), any UCB selector state, and version metadata (git hash, library versions, timestamp).

**pymoo's official documentation recommends `dill` over standard `pickle`** because custom operators often contain lambda functions (e.g., in `ElementwiseDuplicateElimination`) that standard pickle cannot handle. The recommended serialization combines dill with gzip compression:

```python
import dill, gzip, random, time
import numpy as np

def save_checkpoint(filepath, algorithm, config=None):
    data = {
        "algorithm": algorithm,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "metadata": {
            "n_gen": algorithm.n_iter,
            "n_eval": algorithm.evaluator.n_eval,
            "pymoo_version": pymoo.__version__,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pickle_protocol": 5,
            "timestamp": time.time(),
            "config": config,
        },
    }
    with gzip.open(filepath, "wb", compresslevel=5) as f:
        dill.dump(data, f, protocol=5)
```

Pickle protocol 5 (Python 3.8+) enables efficient large-buffer handling via `PickleBuffer` for numpy arrays. **Checkpoint rotation** keeps the last K files (default K=3) and deletes older ones, preventing unbounded disk growth over long runs. The NEAT-Python `Checkpointer` pattern provides a proven reference: it uses gzip + `pickle.HIGHEST_PROTOCOL`, saves RNG state, and supports both generation-interval and time-interval triggers.

A critical pymoo gotcha: **`copy_algorithm=False` must be set** when using `minimize()` with checkpointing, because the default `copy_algorithm=True` deep-copies the algorithm internally, leaving the original object un-updated. The ask-and-tell loop avoids this issue entirely.

---

## Pickle versioning demands a hybrid archival strategy

Pickle stores objects by class module path reference. **Any class rename, module move, or attribute change breaks unpickling** — a real risk across pymoo minor versions. Three mitigations work together:

First, **store version metadata** (pymoo, Python, NumPy, dill versions) in every checkpoint and validate on load. Warn loudly on version mismatch; refuse to load across major version changes.

Second, implement a **hybrid persistence strategy**: dill+gzip checkpoints for fast intra-run restore (protocol 5, compact, includes full algorithm object state), and numpy+JSON archival for long-term storage (every N=50 generations). The archival format saves `Population.X` and `Population.F` as `.npz` arrays with a JSON sidecar containing generation count, evaluation count, problem dimensions, and version info. This format survives any library upgrade.

Third, **`cloudpickle` does not help** with cross-version compatibility despite its name. Its own documentation states: "Cloudpickle can only be used to send objects between the exact same version of Python" and "using cloudpickle for long-term object storage is not supported and strongly discouraged." MLflow's internal comments confirm micro-version releases have broken cloudpickle compatibility. Stick with `dill` for checkpoint persistence and numpy+JSON for archival.

For loading archival data into a new algorithm version, reconstruct the Population from arrays using `Population.new("X", X)` and `Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)`, then pass as initial sampling to a fresh algorithm. This is a warm start, not exact resume — it loses RNG state and generation count.

---

## Graceful shutdown via flag-based signal handling

The optimizer must checkpoint cleanly on SIGTERM (Docker/Kubernetes) and SIGINT (Ctrl+C) without corrupting state. **The key principle: set a flag in the signal handler, check it only between generations** — never during evaluation, where partial state would corrupt the checkpoint.

`threading.Event` is preferred over a bare `bool` for the shutdown flag because `Event.is_set()` is thread-safe and `Event.wait(timeout)` enables responsive polling. Signal handlers must be registered in the main thread (Python limitation) and should be minimal — just set the flag and return.

```python
import signal
import threading
import logging

logger = logging.getLogger(__name__)

class GracefulShutdown:
    """Context manager for signal-safe graceful shutdown."""
    
    def __init__(self):
        self._shutdown_event = threading.Event()
        self._signal_count = 0
        self._old_handlers = {}
    
    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._old_handlers[sig] = signal.signal(sig, self._handler)
        return self
    
    def __exit__(self, *exc):
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)
    
    def _handler(self, signum, frame):
        self._signal_count += 1
        name = signal.Signals(signum).name
        if self._signal_count == 1:
            logger.warning(f"Received {name} — finishing current generation "
                          f"then checkpointing and exiting.")
            self._shutdown_event.set()
        else:
            logger.warning(f"Received {name} again — forcing immediate exit.")
            raise SystemExit(1)
    
    @property
    def should_stop(self) -> bool:
        return self._shutdown_event.is_set()
```

The **double-signal pattern** is important: first SIGINT sets the graceful shutdown flag; second SIGINT forces immediate exit. This gives users an escape hatch if the checkpoint is taking too long. Docker/Kubernetes sends SIGTERM with a **30-second default grace period** before SIGKILL, so checkpoint operations should complete well within that window.

Integration with the ask-and-tell loop is natural — the flag check goes between `algorithm.tell()` and the next `algorithm.ask()`:

```python
with GracefulShutdown() as shutdown:
    while algorithm.has_next() and not shutdown.should_stop:
        pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop)
        algorithm.tell(infills=pop)
        if ckpt_mgr.should_checkpoint(algorithm) or shutdown.should_stop:
            ckpt_mgr.save(algorithm)
```

The `atexit` module provides a fallback for cleanup tasks that must run regardless of exit path, but it cannot perform complex operations reliably and should not be the primary checkpoint mechanism.

---

## Directory structure mirrors MLflow's local artifact layout

The run directory layout follows established ML experiment patterns while adding optimization-specific subdirectories:

```
runs/
  {run_id}_{timestamp}/
    config.toml           # Frozen resolved configuration
    metadata.json          # Versions, git hash, hardware, start time
    checkpoints/           # checkpoint_gen_000050.pkl.gz
    results/               # pareto_front.csv, all_layouts.json, final_population.npz
    figures/               # convergence.pdf, pareto_front.pdf, best_layouts/
    logs/                  # optimizer.log
```

**Run ID generation** uses UUID4's first 8 hex characters (e.g., `a3f7b2c1`) — short enough for human readability, random enough to prevent collisions without coordination. Sequential numbering requires scanning existing directories and breaks under parallel launches. The **timestamp format** `YYYYMMDD_HHMMSS` is filesystem-safe and sorts chronologically.

```python
import uuid
from datetime import datetime
from pathlib import Path

class RunManager:
    SUBDIRS = ("checkpoints", "results", "figures", "logs")
    
    def __init__(self, base_dir: Path = Path("runs")):
        self.base_dir = base_dir
    
    def create_run(self, run_id: str | None = None) -> Path:
        if run_id is None:
            run_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / f"{run_id}_{timestamp}"
        
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        
        run_dir.mkdir(parents=True)
        for sub in self.SUBDIRS:
            (run_dir / sub).mkdir()
        
        self._update_latest_symlink(run_dir)
        return run_dir
    
    def _update_latest_symlink(self, run_dir: Path):
        latest = self.base_dir / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.name, target_is_directory=True)
        except OSError:
            pass  # Symlinks may fail on Windows without dev mode
```

The `latest` → most recent run symlink enables scripts to always find `runs/latest/results/pareto_front.csv` without knowing the run ID. On Windows, symlink creation requires either Developer Mode or admin privileges (Python 3.8+), so the failure is caught silently.

---

## Result export spans CSV, JSON, and NumPy formats

Three export formats serve different consumers. **CSV** for the Pareto front uses columns `[f1, f2, ..., g1, ..., gN, cv, x1, ..., xM]` — objectives first, then constraints, then decision variables — compatible with R's `read.csv()`, Julia's `CSV.read()`, and Excel:

```python
import pandas as pd
import numpy as np

def export_pareto_csv(result, output_path: Path):
    n_obj = result.F.shape[1]
    n_var = result.X.shape[1]
    cols = [f"f{i+1}" for i in range(n_obj)]
    arrays = [result.F]
    
    if result.G is not None:
        n_con = result.G.shape[1]
        cols += [f"g{i+1}" for i in range(n_con)]
        arrays.append(result.G)
    
    cols += [f"x{i+1}" for i in range(n_var)]
    arrays.append(result.X)
    
    df = pd.DataFrame(np.column_stack(arrays), columns=cols)
    df.to_csv(output_path / "pareto_front.csv", index=False, float_format="%.10g")
```

**JSON** for decoded layouts requires a custom `NumpyEncoder` because `numpy.int64`, `numpy.float64`, and `ndarray` are not JSON-serializable:

```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)
```

**NumPy `.npz`** via `np.savez_compressed()` preserves exact floating-point values and is the fastest format for Python-to-Python data transfer. The metadata JSON records final **hypervolume** (computed via `pymoo.indicators.hv.HV(ref_point=ref_point)(result.F)`), **IGD** if a reference Pareto front exists, total evaluations, wall time, and the full frozen config.

---

## Logging balances structured JSON files with tqdm-safe console output

The logging architecture uses two file handlers and one console handler. The **optimizer.log** file uses JSON formatting via `python-json-logger` (v4.1+) for machine-parseable structured records. The console handler outputs human-readable key=value summaries that coexist with tqdm progress bars.

Per-generation log records follow a consistent schema: `gen=50 hv=0.423 igd=0.012 feasible=0.85 eval=10000 time=12.3s`. In the JSON file handler, these become structured fields enabling downstream analysis without parsing.

The **tqdm compatibility problem** — logging to stderr corrupts active progress bars — is solved via `tqdm.contrib.logging.logging_redirect_tqdm`, a context manager that reroutes log output through `tqdm.write()`:

```python
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

with logging_redirect_tqdm(loggers=[logger]):
    with tqdm(total=n_gen, desc="Optimization", unit="gen") as pbar:
        while algorithm.has_next():
            algorithm.next()
            pbar.set_postfix(hv=f"{hv:.4f}", feasible=f"{ratio:.0%}")
            pbar.update(1)
```

`RotatingFileHandler` prevents unbounded log growth: **10 MB max with 5 backups** for the optimizer log (50 MB total), **50 MB max with 3 backups** for the debug log (150 MB total). Both `maxBytes` and `backupCount` must be non-zero for rotation to activate. Log configuration is driven from the Pydantic config via `logging.config.dictConfig()`, using a `LoggingConfig` model that generates the dict config programmatically.

---

## Reproducibility requires capturing every source of randomness

Four categories of metadata guarantee reproducibility. **Random seeds**: a single `seed_all(seed)` function sets `random.seed()`, `np.random.seed()`, and passes the seed to pymoo via `minimize(..., seed=seed)` or `algorithm.setup(..., seed=seed)`. pymoo internally calls `np.random.default_rng(seed)` and stores the generator as `algorithm.random_state`.

**Code version**: `subprocess.check_output(["git", "rev-parse", "HEAD"])` captures the exact commit hash, with `git status --porcelain` detecting dirty working trees. Both are wrapped in `try/except` for non-git environments.

**Dependency versions**: `importlib.metadata.version("pymoo")` for individual packages, plus a full `pip freeze` snapshot saved as `requirements_frozen.txt`. This is more reliable than parsing `poetry.lock` because it reflects the actual installed state.

**Hardware info**: `platform.python_version()`, `platform.machine()`, `os.cpu_count()`, and optionally `psutil.virtual_memory().total` and `psutil.cpu_count(logical=False)` for physical core count. All four categories are combined into a `metadata.json` file at run start.

---

## pymoo ask-and-tell provides natural checkpoint boundaries

The ask-and-tell interface decouples evaluation from algorithm advancement, creating clean boundaries for checkpointing, shutdown checks, and metrics logging. The canonical loop:

```python
algorithm = NSGA2(pop_size=100)
algorithm.setup(problem, termination=("n_gen", 200), seed=42)

while algorithm.has_next():
    pop = algorithm.ask()                        # Get candidate solutions
    algorithm.evaluator.eval(problem, pop)       # Evaluate (counts evals)
    algorithm.tell(infills=pop)                  # Update algorithm state
    # ← Checkpoint, log, and shutdown check here
```

**After `tell()` completes**, the algorithm is in a fully consistent state — the population has been updated, non-dominated sorting is done, and `n_iter` has incremented. This is the only safe checkpoint point. Checkpointing during evaluation would capture partial fitness assignments, producing a corrupt restart.

For restoring, `dill.load()` returns the full algorithm object. If termination was already met, reset it: `algorithm.termination = MaximumGenerationTermination(new_n_gen)`. The `algorithm.has_next()` method then correctly resumes iteration.

One important generation-counting detail: **`n_gen` starts at 1 after `setup()`** (the initialization population), so after the first `next()` or `tell()`, `n_gen` is 2. Account for this offset when logging and naming checkpoint files.

The `Callback` class (`pymoo.core.callback.Callback`) provides an alternative integration point for `minimize()`-based runs. Its `notify(algorithm)` method fires after each generation with full algorithm access, enabling periodic checkpointing without manual loop control. However, the ask-and-tell loop offers more precise control and is recommended for production optimization runs with graceful shutdown requirements.

---

## Conclusion

The `config/` package reduces to three core components: Pydantic v2 `BaseModel` subclasses for each config section (`GAConfig`, `PhysicsConfig`, `DecoderConfig`, `OutputConfig`), a `BaseSettings` root class with `settings_customise_sources` controlling the four-layer priority, and a serialization module converting frozen configs to TOML via `tomli_w`. The `io/` package splits into four modules: `checkpoint.py` (dill+gzip save/load with rotation and hybrid numpy archival), `shutdown.py` (the `GracefulShutdown` context manager), `run_manager.py` (directory creation, symlink management), and `export.py` (CSV/JSON/npz output with `NumpyEncoder`).

Two design decisions merit emphasis. First, **`frozen=True` everywhere** — from Pydantic models to the config-snapshot-in-checkpoint pattern — prevents an entire class of subtle bugs where shared mutable state causes non-reproducible behavior. Second, the **hybrid checkpoint strategy** (fast dill+gzip for intra-run recovery, portable numpy+JSON for archival) elegantly resolves the fundamental tension between pickle's convenience and its versioning fragility. Together, these infrastructure packages ensure that every optimization run is restartable, reproducible, and auditable from a single `runs/{id}_{timestamp}/` directory.