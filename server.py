"""HTTP server wrapping the V2 port-pair optimizer for browser-driven runs.

Stdlib-only: ``http.server.ThreadingHTTPServer`` + ``BaseHTTPRequestHandler``.
No Flask, no FastAPI — keeps "easy to run" as a single ``python server.py``.

The server is **long-running with a hot backend**: on every ``/api/run`` it
clears every ``src_v2.*`` and ``src.*`` entry from ``sys.modules`` and
re-imports, so editing any file under ``src_v2/`` (operators, repair,
problem, structural_mutations, …) is picked up by the next click on the
browser's *Run* button — no server restart needed. Configs and the catalog
YAML are read from disk on each run too. Server stays up; only the backend
code rebinds.

Endpoints:

- ``GET /``                 — serves ``web/index.html``
- ``GET /<file>``           — serves any other file under ``web/``
- ``GET /api/configs``      — JSON list of available config names from ``configs/``
- ``POST /api/run``         — body ``{config, pop_size, n_gen, heuristic_ratio}``;
                              reloads ``src_v2`` from disk, then streams
                              ``{type: "log", line: ...}`` for each progress
                              line, ending with ``{type: "result", ...}`` or
                              ``{type: "error", error: ..., trace: ...}``.
- ``POST /api/reload``      — explicit backend reload (returns a list of
                              modules dropped). Useful for sanity-checking a
                              fresh edit before launching a long simulation.
- ``GET /outputs_v2/...``   — serves PNG outputs produced by the run

Run with:

    .venv/Scripts/python.exe server.py

Then open http://localhost:8000 in a browser. Leave it running across
edits.
"""

from __future__ import annotations

<<<<<<< Updated upstream
import os

# matplotlib's TkAgg default is not thread-safe and crashes the
# ThreadingHTTPServer worker with "Tcl_AsyncDelete: ... wrong thread" the
# moment any plotting runs off the main thread. Agg is headless and safe.
# Set BEFORE any matplotlib import (transitively pulled in via src_v2).
os.environ["MPLBACKEND"] = "Agg"

=======
>>>>>>> Stashed changes
import io
import json
import logging
import mimetypes
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


HOST = "127.0.0.1"
PORT = 8000

REPO_ROOT = Path(__file__).resolve().parent
WEB_DIR = REPO_ROOT / "web"
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "outputs_v2"


# =============================================================================
# Optimization runner with streaming log capture
# =============================================================================


class _StreamingLogHandler(logging.Handler):
    """Captures log records and pushes them through a callback per record."""

    def __init__(self, callback):
        super().__init__(level=logging.INFO)
        self._cb = callback
        self.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._cb(self.format(record))
        except Exception:
            pass


def reload_backend() -> list:
    """Drop every cached ``src_v2.*`` and ``src.*`` module from ``sys.modules``.

    The next ``import`` will then read the fresh source from disk. Returns the
    sorted list of module names that were dropped (useful for diagnostics).

    We deliberately also drop ``src.*`` because some integration tests / the
    legacy V1 runner share import roots; if either is mid-edit we want the
    fresh version, not a half-cached one.
    """
    drop_prefixes = ("src_v2.", "src_v2", "src.", "src")
    to_drop = [
        name for name in list(sys.modules)
        if any(name == p or name.startswith(p + ".") for p in drop_prefixes)
    ]
    for name in to_drop:
        del sys.modules[name]
    return sorted(to_drop)


def run_optimization_streaming(
    config_name: str,
    pop_size: int,
    n_gen: int,
    heuristic_ratio: float,
<<<<<<< Updated upstream
    crossover_prob: float,
    mutation_prob: float,
    n_workers: int,
=======
>>>>>>> Stashed changes
    on_log,
):
    """Run V2 optimization, emitting log lines via on_log(str).

    Reloads ``src_v2`` from disk before importing so that any backend edits
    made while the server was running are picked up by THIS run.

    Returns a summary dict suitable for JSON serialization.
    """
    dropped = reload_backend()
    on_log(f"Backend reloaded — {len(dropped)} modules refreshed")

    from src_v2.catalog import TrackCatalog
    from src_v2.config import OptimizationConfig
    from src_v2.decoder import decode_chromosome
    from src_v2.encoding import iter_active_slots
    from src_v2.problem import PortPairProblem
    from src_v2.runner import run_optimization, save_results
<<<<<<< Updated upstream
    from src_v2.visualization import port_graph_to_json
=======
>>>>>>> Stashed changes

    handler = _StreamingLogHandler(on_log)
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    try:
        catalog = TrackCatalog.load(REPO_ROOT / "data" / "track_pieces_v2.yaml")
        config = OptimizationConfig.load(CONFIG_DIR / f"{config_name}.yaml")
        config.algorithm.pop_size = int(pop_size)
        config.algorithm.n_gen = int(n_gen)
        config.algorithm.heuristic_ratio = float(heuristic_ratio)
<<<<<<< Updated upstream
        config.algorithm.crossover_prob = float(crossover_prob)
        config.algorithm.mutation_prob = float(mutation_prob)
        config.n_workers = int(n_workers)

        on_log(f"Config '{config_name}' loaded — pop={pop_size}, gen={n_gen}, "
               f"heuristic={heuristic_ratio:.0%}, "
               f"crossover={crossover_prob:.2f}, mutation={mutation_prob:.2f}, "
               f"workers={n_workers}")

        out_dir = OUTPUT_DIR / config_name
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        res = run_optimization(config, catalog, verbose=True, output_dir=out_dir)
        elapsed = time.time() - t0

=======

        on_log(f"Config '{config_name}' loaded — pop={pop_size}, gen={n_gen}, "
               f"heuristic={heuristic_ratio:.0%}")

        t0 = time.time()
        res = run_optimization(config, catalog, verbose=True)
        elapsed = time.time() - t0

        out_dir = OUTPUT_DIR / config_name
>>>>>>> Stashed changes
        save_results(res, out_dir, catalog, config)

        F = res.pop.get("F")
        G = res.pop.get("G")
        feasible = (
            np.all(G <= 0, axis=1) if G is not None else np.zeros(len(F), dtype=bool)
        )
        n_feasible = int(feasible.sum())

        summary = {
            "config": config_name,
            "elapsed_sec": elapsed,
            "n_feasible": n_feasible,
            "pop_size": int(pop_size),
<<<<<<< Updated upstream
            "total_inventory": int(config.total_inventory),
            "inventory": dict(config.inventory),
        }

        problem = PortPairProblem(catalog, config)
        layout_json: dict | None = None
        layout_infeasible_json: dict | None = None
        finite = ~np.isinf(F).any(axis=1)

        if n_feasible > 0:
            feas_idx = np.where(feasible)[0]
            best_idx = feas_idx[np.argmin(F[feas_idx, 0])]
=======
        }

        if n_feasible > 0:
            feas_idx = np.where(feasible)[0]
            best_idx = feas_idx[np.argmin(F[feas_idx, 0])]
            problem = PortPairProblem(catalog, config)
>>>>>>> Stashed changes
            graph = decode_chromosome(
                res.pop.get("X")[best_idx], problem.dims, catalog,
                problem.decoder_config,
            )
<<<<<<< Updated upstream
            counts: dict[str, int] = {}
=======
            counts = {}
>>>>>>> Stashed changes
            for _, idx in iter_active_slots(res.pop.get("X")[best_idx], problem.dims):
                pid = catalog.index_to_id.get(idx, f"?{idx}")
                counts[pid] = counts.get(pid, 0) + 1
            summary.update({
                "best_util": float(-F[best_idx, 0]),
                "best_min_speed": float(-F[best_idx, 1]),
                "n_pieces": int(graph.n_slots),
                "n_cycles": int(graph.n_cycles),
                "n_components": int(graph.n_components),
                "closure_pos": float(graph.max_closure_position),
                "closure_angle_deg": float(graph.max_closure_angle_deg),
                "piece_counts": counts,
            })
<<<<<<< Updated upstream
            layout_json = port_graph_to_json(graph, catalog, config)

        # Best-infeasible layout: highest utilization among infeasible+finite.
        infeas_finite = (~feasible) & finite
        if infeas_finite.any():
            inf_idx = np.where(infeas_finite)[0]
            best_inf_idx = inf_idx[np.argmin(F[inf_idx, 0])]
            graph_inf = decode_chromosome(
                res.pop.get("X")[best_inf_idx], problem.dims, catalog,
                problem.decoder_config,
            )
            layout_infeasible_json = port_graph_to_json(graph_inf, catalog, config)
=======
>>>>>>> Stashed changes

        return {
            "elapsed_sec": elapsed,
            "summary": summary,
<<<<<<< Updated upstream
            "layout": layout_json,
            "layout_infeasible": layout_infeasible_json,
=======
>>>>>>> Stashed changes
            "best_layout": (
                f"outputs_v2/{config_name}/best_layout.png"
                if (out_dir / "best_layout.png").exists() else None
            ),
            "best_infeasible": (
                f"outputs_v2/{config_name}/best_infeasible.png"
                if (out_dir / "best_infeasible.png").exists() else None
            ),
            "pareto_front": (
                f"outputs_v2/{config_name}/pareto_front.png"
                if (out_dir / "pareto_front.png").exists() else None
            ),
            "timestamp": int(time.time()),
        }
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(prev_level)


# =============================================================================
# HTTP handler
# =============================================================================


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[server] {self.address_string()} - {fmt % args}\n")

    # ---- routing ----

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._serve_static(WEB_DIR / "index.html")
            return
        if path == "/api/configs":
            self._send_json({"configs": self._list_configs()})
            return
<<<<<<< Updated upstream
        if path == "/api/catalog":
            self._handle_catalog()
            return
        if path == "/api/showcase":
            self._handle_showcase()
            return
        if path == "/api/topology-showcase":
            self._handle_topology_showcase()
            return
=======
>>>>>>> Stashed changes
        if path.startswith("/outputs_v2/"):
            rel = path.lstrip("/")
            self._serve_static(REPO_ROOT / unquote(rel))
            return
        # web/ static fallback
        rel = path.lstrip("/")
        candidate = WEB_DIR / unquote(rel)
        if candidate.is_file():
            self._serve_static(candidate)
            return
        self._send_text(404, "not found")

<<<<<<< Updated upstream
    def _handle_catalog(self) -> None:
        """Serialize the V2 catalog as render-ready JSON. Reloads on every
        request so edits to the YAML or to the serializer are picked up
        without restarting the server."""
        try:
            reload_backend()  # ensure fresh src_v2.* on edits
            from src_v2.catalog import TrackCatalog
            from src_v2.visualization import catalog_to_json
            catalog = TrackCatalog.load(REPO_ROOT / "data" / "track_pieces_v2.yaml")
            self._send_json(catalog_to_json(catalog))
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()})

    def _handle_showcase(self) -> None:
        """Return a fixed reference layout containing one of each piece kind,
        laid out in a grid. Used by the browser's "Showcase" view to eyeball
        every kind without running an optimization."""
        try:
            reload_backend()
            from src_v2.catalog import TrackCatalog
            from src_v2.visualization import build_showcase_layout
            catalog = TrackCatalog.load(REPO_ROOT / "data" / "track_pieces_v2.yaml")
            self._send_json(build_showcase_layout(catalog))
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()})

    def _handle_topology_showcase(self) -> None:
        """Return five hand-built layouts (circle, oval, siding, figure-8 via
        CROSS_90, double-crossover) proving the chromosome encoding can
        represent each topology end-to-end."""
        try:
            reload_backend()
            from src_v2.catalog import TrackCatalog
            from src_v2.config import OptimizationConfig
            from src_v2.visualization import build_topology_showcase
            catalog = TrackCatalog.load(REPO_ROOT / "data" / "track_pieces_v2.yaml")
            config = OptimizationConfig.load(REPO_ROOT / "configs" / "with_switches.yaml")
            self._send_json({"layouts": build_topology_showcase(catalog, config)})
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()})

=======
>>>>>>> Stashed changes
    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/run":
            self._handle_run()
            return
        if path == "/api/reload":
            self._handle_reload()
            return
        self._send_text(404, "not found")

    def _handle_reload(self) -> None:
        try:
            dropped = reload_backend()
            self._send_json({
                "ok": True,
                "n_dropped": len(dropped),
                "dropped": dropped,
            })
        except Exception as e:
            self._send_json({
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc(),
            })

    # ---- /api/run ----

    def _handle_run(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b"{}"
            params = json.loads(body or "{}")
        except Exception as e:
            self._send_text(400, f"bad request: {e}")
            return

        config_name = str(params.get("config", "default"))
        pop_size = int(params.get("pop_size", 500))
        n_gen = int(params.get("n_gen", 200))
        heuristic_ratio = float(params.get("heuristic_ratio", 0.30))
<<<<<<< Updated upstream
        crossover_prob = float(params.get("crossover_prob", 0.9))
        mutation_prob = float(params.get("mutation_prob", 0.1))
        n_workers = int(params.get("n_workers", 1))
=======
>>>>>>> Stashed changes

        # NDJSON streaming response — one JSON object per line
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        lock = threading.Lock()

        def emit_log(line: str) -> None:
            payload = json.dumps({"type": "log", "line": line}) + "\n"
            with lock:
                try:
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

        try:
            result = run_optimization_streaming(
<<<<<<< Updated upstream
                config_name, pop_size, n_gen, heuristic_ratio,
                crossover_prob, mutation_prob, n_workers, emit_log,
=======
                config_name, pop_size, n_gen, heuristic_ratio, emit_log,
>>>>>>> Stashed changes
            )
            payload = json.dumps({"type": "result", **result}) + "\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            payload = json.dumps({"type": "error", "error": str(e), "trace": tb}) + "\n"
            try:
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    # ---- helpers ----

    def _list_configs(self):
        if not CONFIG_DIR.is_dir():
            return []
        return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.is_file())

    def _send_json(self, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, code: int, msg: str) -> None:
        body = msg.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: Path) -> None:
        if not path.is_file():
            self._send_text(404, f"not found: {path.name}")
            return
        ctype, _ = mimetypes.guess_type(str(path))
        ctype = ctype or "application/octet-stream"
        # Force JSX/JS to be served as JS so the browser will run them
        if path.suffix == ".jsx":
            ctype = "text/babel"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"LEGO Track Optimizer (V2) — http://{HOST}:{PORT}")
    print("Open the URL above in a browser. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
        server.shutdown()


if __name__ == "__main__":
    main()
