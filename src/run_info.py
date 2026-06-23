"""Per-run provenance: write ``run_info.md`` into each optimizer output dir.

The file records, for every run:

1. **Code State** — HEAD commit + summary of uncommitted changes so a run can
   be tied back to exact source.
2. **Configuration** — which config YAML was loaded, and a verbatim copy of
   the file contents (configs with the same name can change between runs, so
   the on-disk text at run time is captured here).
3. **Run Summary** — feasibility counts, best feasible, best overall, piece
   usage. Appended *after* the optimizer finishes.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import compute_dimensions

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG = logging.getLogger(__name__)

RUN_INFO_FILENAME = "run_info.md"


# =============================================================================
# Git introspection
# =============================================================================

def _git(*args: str) -> str | None:
    """Run a git command in the repo root; return stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.rstrip("\n")


def _git_state_section() -> str:
    """Render the **Code State** markdown section."""
    head_line = _git("log", "-1", "--pretty=format:%h %s")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    diff_stat = _git("diff", "HEAD", "--stat")

    if head_line is None:
        return "## Code State\n\n_git not available — code state unknown._\n"

    lines = [
        "## Code State",
        "",
        f"- **Commit**: `{head_line}`",
        f"- **Branch**: `{branch or 'detached'}`",
        "",
    ]

    if status:
        lines += [
            "**Uncommitted changes** (`git status --porcelain`):",
            "",
            "```",
            status,
            "```",
            "",
        ]
        if diff_stat:
            lines += [
                "**Diff stat** (`git diff HEAD --stat`):",
                "",
                "```",
                diff_stat,
                "```",
                "",
            ]
    else:
        lines += ["_Working tree clean — no uncommitted changes._", ""]

    return "\n".join(lines)


# =============================================================================
# Config rendering
# =============================================================================

def _config_section(config_path: str | Path, config: OptimizationConfig,
                    quick_test: bool) -> str:
    """Render the **Configuration** section with the raw YAML file embedded.

    Configs with the same name can be edited between runs, so the on-disk
    file contents at run time are captured verbatim here (not the
    post-Pydantic ``model_dump``).
    """
    config_path = Path(config_path)
    try:
        raw_yaml = config_path.read_text(encoding="utf-8").rstrip()
    except OSError as exc:
        raw_yaml = f"# Could not read {config_path}: {exc}"

    overrides = []
    if quick_test:
        overrides.append(
            f"- **Quick test override**: n_gen={config.algorithm.n_gen}, "
            f"pop_size={config.algorithm.pop_size}"
        )

    lines = [
        "## Configuration",
        "",
        f"- **Config file**: `{config_path}`",
        f"- **Total inventory**: {config.total_inventory} pieces",
        *overrides,
        "",
        f"**Verbatim contents of `{config_path}` at run time:**",
        "",
        "```yaml",
        raw_yaml,
        "```",
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# Header writer
# =============================================================================

def write_run_info_header(
    output_dir: Path,
    config_path: str | Path,
    config: OptimizationConfig,
    *,
    quick_test: bool = False,
) -> Path:
    """Write the code-state + configuration sections to ``run_info.md``.

    Overwrites any previous file. Returns the path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RUN_INFO_FILENAME

    timestamp = datetime.now().isoformat(timespec="seconds")
    body = "\n".join([
        f"# Run Info — {timestamp}",
        "",
        _git_state_section(),
        _config_section(config_path, config, quick_test),
    ])
    path.write_text(body, encoding="utf-8")
    return path


# =============================================================================
# Run summary
# =============================================================================

def _piece_usage(layout, inventory: dict, catalog: TrackCatalog) -> list[str]:
    """Mirror ``runner.log_piece_usage`` as markdown bullet lines."""
    counts: dict[int, int] = {}
    main = getattr(layout, "main_loop_pieces", None)
    if main is not None:
        for p in main:
            if p >= 0:
                counts[p] = counts.get(p, 0) + 1
    else:
        for p in getattr(layout, "indices", []):
            if p >= 0:
                counts[p] = counts.get(p, 0) + 1

    for pair in getattr(layout, "switch_pairs", []) or []:
        for p in pair.branch_pieces:
            if p >= 0:
                counts[p] = counts.get(p, 0) + 1

    rows: list[str] = []
    for piece_id, capacity in sorted(inventory.items()):
        idx = catalog.id_to_index.get(piece_id)
        if idx is None:
            continue
        used = counts.get(idx, 0)
        rows.append(f"  - `{piece_id}`: {used}/{capacity}")
    return rows


def _format_individual(label: str, layout, util: float, speed: float,
                       cv: float | None) -> list[str]:
    line = (f"- **{label}**: {layout.n_pieces} pieces, util={util:.1%}, "
            f"speed={speed:.2f} m/s, switches={layout.n_switch_pairs}")
    if cv is not None:
        line += f", CV={cv:.2f}"
    return [line]


def append_run_summary(
    output_dir: Path,
    res,
    catalog: TrackCatalog,
    config: OptimizationConfig,
) -> None:
    """Append the **Run Summary** section based on the optimizer result.

    Mirrors the numbers logged at the end of :func:`run_optimization` and
    :func:`save_results` so the markdown file is a self-contained record.
    Silent on missing data — appends what it can.
    """
    output_dir = Path(output_dir)
    path = output_dir / RUN_INFO_FILENAME
    if not path.exists():
        return

    lines: list[str] = ["", "## Run Summary", ""]
    lines.append(f"- Generations: {config.algorithm.n_gen}")
    lines.append(f"- Population: {config.algorithm.pop_size}")

    pop = getattr(res, "pop", None)
    if pop is None:
        lines.append("- _No population available on result._")
        _append(path, lines)
        return

    F = pop.get("F")
    G = pop.get("G")
    X = pop.get("X")
    if F is None or X is None:
        lines.append("- _Result missing F/X arrays._")
        _append(path, lines)
        return

    feasible_mask = (np.all(G <= 0, axis=1) if G is not None
                     else np.ones(len(X), dtype=bool))
    n_feasible = int(np.sum(feasible_mask))
    lines.append(f"- Feasible solutions: {n_feasible}/{len(X)}")

    dims = compute_dimensions(config, catalog)

    best_feas_layout = None
    if n_feasible > 0:
        feas_idx = np.where(feasible_mask)[0]
        best_feas = int(feas_idx[int(np.argmin(F[feas_idx, 0]))])
        best_feas_layout = decode_chromosome(
            X[best_feas], catalog, config.inventory, dims=dims,
        )
        lines += _format_individual(
            "Best feasible", best_feas_layout,
            float(-F[best_feas, 0]), float(-F[best_feas, 1]), cv=None,
        )

    best_overall = int(np.argmin(F[:, 0]))
    overall_layout = decode_chromosome(
        X[best_overall], catalog, config.inventory, dims=dims,
    )
    overall_cv = (float(np.sum(np.maximum(0, G[best_overall])))
                  if G is not None else 0.0)
    overall_label = ("Best overall (feasible)" if feasible_mask[best_overall]
                     else "Best overall (infeasible)")
    lines += _format_individual(
        overall_label, overall_layout,
        float(-F[best_overall, 0]), float(-F[best_overall, 1]), cv=overall_cv,
    )

    usage_layout = best_feas_layout or overall_layout
    usage_label = "best feasible" if best_feas_layout is not None else "best overall"
    usage_rows = _piece_usage(usage_layout, config.inventory, catalog)
    if usage_rows:
        lines.append("")
        lines.append(f"**Piece usage** ({usage_label}):")
        lines.append("")
        lines += usage_rows

    _append(path, lines)


def _append(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")