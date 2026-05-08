# The visualization package, rendered honestly

**The visualization package delivers four canonical thesis figures — best-layout render, Pareto front with knee annotation, HV/IGD convergence, and speed heatmap with R\* overlay — plus a fifth diagnostic (operator-usage stacked area) kept as an appendix figure.** Rendering piece beds uses `matplotlib.patches.Rectangle` for straights, discretized `LineCollection` polylines for curves (because `matplotlib.patches.Arc` explicitly cannot be filled and cannot accept a `facecolor`), and compound `PathPatch` artists for switches; layers are distinguished by alpha and linestyle rather than hue. Colors follow the Okabe-Ito 2008 eight-tone categorical palette for piece types and `viridis` for continuous speed, the combination recommended by Crameri, Shephard & Heron (2020). The hypervolume reference point is pinned at **r = (+0.10, −0.55)** — ten percent of range beyond the nadir (0, −0.60) — which matches Ishibuchi et al.'s (2018) recommendation for a bi-objective problem with a roughly linear front. The package never runs the optimizer: it consumes pickled `Callback` data and `DecodedLayout` objects from disk under the `Agg` backend, because matplotlib is explicitly documented as not thread-safe.

## Five figures the thesis must produce

After auditing what each figure must prove, four should appear in the main text and one (operator usage) in the implementation appendix. The rule of thumb from Cleveland (1985) — one plot, one claim — drove the cut. A fifth figure helps only reviewers who care about the operator scheduler; the main argument of the thesis is carried by the other four.

| # | Figure | Primary claim it supports | Data source | Matplotlib primitives |
|---|---|---|---|---|
| 1 | Best layout from the Pareto front | "The optimizer produces closed, feasible, non-trivial layouts." | `DecodedLayout` of the selected knee individual | `Rectangle`, `LineCollection`, `PathPatch`, `Annotation` |
| 2 | Pareto front with non-dominated set | "Trade-off between piece utilization and bottleneck speed is real and explored." | `F_history[-1]`, `CV_history[-1]` from `ConvergenceMonitorCallback` | `scatter` (two styles), `axhline`/`axvline` for ideal/nadir, `annotate` for knee |
| 3 | Convergence of HV with feasibility overlay | "NSGA-II converges in HV and feasibility rate rises monotonically." | `monitor_data["hv"]`, `monitor_data["igd"]`, `monitor_data["n_feas"]` | `plot` on two `twinx()` axes, shaded envelopes from `fill_between` |
| 4 | Speed heatmap with R\* inset | "Tight curves are the binding constraint on f₂; R\* ≈ 44 studs is the phase boundary." | `DecodedLayout` placements + `train.v_bottleneck` | Discretized arcs colored via `LineCollection.set_array()`; inset axes for v(R) curve |
| 5 | Operator usage stacked area (appendix) | "UCB scheduler re-weights operators across phases." | `algorithm.callback.data["op_history"]` | `stackplot` |

Each figure is sized for a thesis column (88 mm ≈ 3.46 in wide, single column) or, for the layout render, full page width (180 mm ≈ 7.09 in), with 10-11 pt serif text matching the body font. SVG is the primary vector output; a 300-DPI PNG is co-emitted for preview purposes.

## Arc patches cannot be filled, so we discretize

The piece-to-patch mapping looks innocent but hits a documented matplotlib limitation. The canonical `matplotlib.patches.Arc` class states explicitly in its stable docs: *"The arc cannot be filled"* and *"Most Patch properties are supported as keyword arguments, with the exception of `fill` and `facecolor` because filling is not supported."* This is not an oversight — `Arc.draw()` is optimized to render only the segments inside the axes bounding box, and that optimization is incompatible with polygon fill. The consequence for us is that curves must be rendered as *discretized polylines* in a `LineCollection`, not as `Arc` patches.

The mapping we adopt:

| Piece type | Bed primitive | Rail primitive | Notes |
|---|---|---|---|
| Straight (S4/S8/S16) | `Rectangle` oriented by piece heading | Two parallel segments in a `LineCollection` | Gauge = 5 studs; rails at ±2.5 studs from centerline |
| Curve (R40–R148, L/R) | Discretized polygon (concave fan) via `Polygon` with ~24 samples for R40/22.5°, up to ~48 for tight radii | Two offset arcs as `LineCollection` at R±2.5 studs | **No** `Arc` patch; `facecolor` unsupported |
| Switch (LEGO 53407/53404, 4DBrix variants) | Compound `PathPatch` combining `MOVETO`+`LINETO` for the straight bed and `CURVE3`/`CURVE4` Bézier for the diverging route | Three rail polylines sharing the throat | Matplotlib's own `Path.arc()` uses eight cubic Béziers internally — we mirror that idiom |
| Crossing | Primary `Rectangle` + rotated `Rectangle` for the second axis | Two crossed `LineCollection` pairs | Z-order must place the second axis on top |

Layer distinction is carried by two channels, not by hue (hue is reserved for piece type): `main` layer renders with alpha=1.0 solid, `branch` with alpha=0.7 solid, `crossing` with alpha=0.7 dashed. This leaves the categorical palette free to encode the eight piece-type categories without collision.

`LineCollection` is not just a workaround — it is also the right performance choice. The matplotlib docs state: *"Matplotlib can efficiently draw multiple lines at once using a LineCollection… In order to efficiently plot many lines in a single set of axes, Matplotlib has the ability to add the lines all at once."* One `LineCollection` drawing all rails of a 200-piece layout is roughly an order of magnitude faster than 400 `plot()` calls and carries a single colormap array for per-segment speed coloring via `set_array()` — exactly what the speed heatmap requires.

## Okabe-Ito for categories, viridis for continuous; cite Crameri

Color is the most abused dimension in scientific figures. Crameri, Shephard & Heron (*Nat Commun* 11:5444, 2020) show that rainbow maps (`jet`, `hsv`) distort perceived magnitudes and fail catastrophically for readers with color-vision deficiency, which affects ~8% of men. Their prescription: *perceptually uniform, perceptually ordered, color-vision-deficiency-friendly* maps. Two categorical piece types collide into one perceived category under `jet`; they do not under Okabe-Ito.

**Categorical palette (piece type).** Okabe & Ito (2008, Color Universal Design, jfly.uni-koeln.de/color/) provide eight distinct hues designed to be distinguishable under protanopia, deuteranopia, and tritanopia: black `#000000`, orange `#E69F00`, sky blue `#56B4E9`, bluish green `#009E73`, yellow `#F0E442`, blue `#0072B2`, vermillion `#D55E00`, reddish purple `#CC79A7`. The catalog's eight piece-type buckets (straight, curve small, curve medium, curve large, switch-L, switch-R, crossing, special) map one-to-one; a ninth category would force us to drop a bucket or switch palettes.

**Continuous palette (speed, age, closure residual).** `viridis` (van der Walt & Smith, SciPy 2015, bids.github.io/colormap) and `cividis` are the perceptually uniform defaults; both are colorblind-safe and print-safe. For the speed heatmap we use `viridis` in [0.6, 1.10] m/s. Crameri's own `batlow` is an equally defensible alternative, though it requires the `cmcrameri` pip package.

## Publication-quality defaults

Thesis-quality defaults are less about any single `rcParams` key than about committing to a coherent stylesheet. We use SciencePlots (Garrett 2021, Zenodo DOI 10.5281/zenodo.4106649) as the base — `plt.style.use(['science', 'ieee'])` — then override only what the thesis template demands. A caveat from verification: matplotlib's `font.family` default is `sans-serif`, not `serif`; explicit override is required. Also `savefig.dpi` defaults to the string `'figure'`, meaning it inherits `figure.dpi` (default 100), so raster PNG previews need an explicit `dpi=300`.

| Setting | Value | Rationale / source |
|---|---|---|
| Backend (batch) | `Agg` | matplotlib FAQ: non-interactive, thread-safe-enough for file output |
| `figure.dpi` | 150 | Screen preview; vector output ignores this |
| `savefig.dpi` | 300 | Raster fallback (PNG); SVG/PDF are resolution-free |
| `savefig.format` | `svg` primary, `png` secondary | Native matplotlib vector outputs |
| `font.family` | `serif` | Matches thesis body; default would be `sans-serif` |
| `font.size` | 10–11 pt | Column-width readability after LaTeX scaling |
| `text.usetex` | `True` | Requires LaTeX + dvipng + Ghostscript per matplotlib docs |
| `svg.fonttype` | `'none'` | Smaller files, text editable in Inkscape; accept the font-availability caveat |
| Figure width | 88 mm (column) / 180 mm (full) | IEEE/Springer thesis columns |

Hunter's 2007 paper (*Comput Sci Eng* 9(3):90-95, DOI 10.1109/MCSE.2007.55) is the canonical matplotlib citation; every thesis using matplotlib cites it once.

## The Pareto front needs a knee, not just a scatter

A raw scatter of the final population buries the most interesting point. Bechikh, Ben Said & Ghédira (*Soft Computing* 15(9):1807-1823, 2011, and the 2015 survey in *Advances in Computers* vol. 98) argue that knee points — regions where a small improvement in one objective forces a large sacrifice in another — are the decision-theoretic "interesting" solutions when no explicit preference is stated. Our figure therefore:

- Plots dominated feasible solutions as small dots (alpha 0.3), non-dominated feasible solutions as filled circles (alpha 1.0), and infeasible individuals (any CV > 0) as light-gray crosses behind everything.
- Reorients axes to **positive** directions: x = piece utilization (= −f₁) in [0, 1], y = bottleneck speed (= −f₂) in [0.60, 1.10] m/s. Pymoo's internal F stores sign-flipped values for minimization; the visualization flips them back so the thesis reader sees "up and right is better."
- Draws the ideal point at (1.0, 1.10) m/s and nadir at (0.0, 0.60) m/s as dashed crosshair guides, following the EMO convention.
- Annotates the knee with pymoo's own `HighTradeoffPoints` (Deb & Gupta / Branke) from `pymoo.mcdm` — native to pymoo 0.6.1, not a custom implementation.

The empirical attainment surface across seeds (López-Ibáñez, Paquete & Stützle, 2010, DOI 10.1007/978-3-642-02538-9_9) is a natural extension. When ≥5 seeds are available, the median attainment surface plus 25–75% band conveys run variance cleanly. Watanabe's `empirical-attainment-func` (arXiv:2305.08852, 2023) is the Python implementation — however, the package README now redirects users to the OptunaHub port, which is the maintained path. For the thesis we treat EAS as optional: adequate seeds (20–30) are needed for it to be informative, and that requires a multi-run sweep that is itself future work.

## Convergence tells three stories at once

HV, IGD, and feasibility rate all move together on a healthy run, but they diverge diagnostically when something is wrong. HV rising while feasibility stalls at zero means the "Pareto front" is entirely infeasible — a catastrophic failure mode for a constrained problem. The figure therefore plots all three, carefully.

HV uses the reference point **r = (+0.10, −0.55)**. The derivation: nadir is (0, −0.60) (worst f₁ and worst f₂ for minimization); range is (1.00, 0.50); ten percent of range beyond the nadir gives (0.10, −0.55). Ishibuchi, Imada, Setoguchi & Nojima (*Evolutionary Computation* 26(3):411-440, 2018, DOI 10.1162/evco_a_00226) warn that the "slightly beyond the nadir" heuristic is only provably fair on roughly linear or triangular Pareto fronts; for inverted-triangular fronts the choice biases algorithm ranking. In our two-objective case with a front expected to be roughly linear, the heuristic is defensible, but the thesis must **report r explicitly** alongside every HV number so later comparisons are reproducible.

IGD goes on the right (`twinx()`) axis because its scale (often 0.01–0.5) is incompatible with HV's. The feasibility rate n_feas/N is plotted as a lightweight third trace on a 0-1 inset axis at the top of the figure, following Hansen et al.'s BBOB methodology of showing multiple performance dimensions on a single plot without overplotting. When ≥5 seeds are available, mean-with-shaded-std-envelope replaces the single trace.

## The speed heatmap is the thesis's central physics figure

This figure carries the train-physics report's narrative in one image. For each placed curve piece the decoder provides the radius; the train module provides `v_bottleneck(R, μ) = min(√(μ·g·R), v_cap)` with `v_cap = 1.10 m/s` and `μ = 0.35` nominal. The bed polyline for each curve is colored by its computed `v_safe`, using `LineCollection.set_array(v_values)` and `viridis` as the colormap mapped onto [0.60, 1.10]. Straight pieces are drawn in neutral gray — speed on a straight is trivially `v_cap` and coloring it would dilute the contrast that matters.

An inset axes in the lower-right corner shows the v_safe(R) curve itself as a function of radius. Two markers are drawn on the curve: a dashed vertical at **R\* ≈ 44 studs** (the friction-binding radius R*(0.35)), and a dashed horizontal at **v_cap = 1.10 m/s**. The R\* marker is the whole reason this figure exists: below R\*, tighter curves cost speed; above R\*, the physics saturates. A layout whose curves all sit above R\* extracts no f₂ gain from being further lattice-friendly, and a layout with curves far below R\* is paying an avoidable safety cost. The figure lets a reader see, in one glance, which regime the solution occupies.

## Operator usage as a stacked area reveals phase transitions

UCB-scheduled operator selection rarely stays uniform. Early in a run, the exploration operators (insertion mutation, sub-path reversal) typically win draws because their reward estimates are favorable on infeasible populations; once feasibility is achieved, exploitation operators (port-flip, radius-swap) dominate. Plotting `algorithm.callback.data["op_history"]` as a `stackplot` — generation on x, cumulative selection count on y, one band per operator in the Okabe-Ito palette — makes these phase transitions visible as kinks in the band widths. This is the diagnostic figure that justifies keeping the UCB scheduler instead of a uniform mix.

## Visualization runs from persisted artifacts, never from a live optimizer

The visualization package sits downstream of the optimizer and reads only from disk. This decoupling is not cosmetic. First, matplotlib is explicitly documented as not thread-safe — the stable FAQ states: *"You may be able to work on separate figures from separate threads. However, you must in that case use a non-interactive backend (typically Agg), because most GUI backends require being run from the main thread as well."* Second, plot regeneration during thesis writing must not require re-running a 30-minute NSGA-II sweep. Third, the reproducibility audit trail is cleaner when the plotting code is stateless with respect to optimizer internals.

| Artifact on disk | Format | Consumer figure |
|---|---|---|
| `run_dir/callbacks/monitor.pkl` | Pickled `ConvergenceMonitorCallback.data` dict | Convergence (HV/IGD/feasibility) |
| `run_dir/callbacks/op_history.npz` | NumPy array, shape `(n_gen, n_ops)` | Operator usage stacked area |
| `run_dir/final_pop.pkl` | Pickled list of `DecodedLayout` | Best layout render, speed heatmap |
| `run_dir/F_history.npz` + `CV_history.npz` | Per-generation F and CV arrays | Pareto front scatter |
| `catalog/pieces.yaml` | Piece catalog with colors | Color lookup for all layout figures |
| `config/physics.yaml` | μ, g, v_cap | v_safe computation for speed heatmap |

The `save_thesis_figures(run_dir, output_dir)` orchestrator walks this manifest, emits SVG+PNG for each figure, and writes a per-run `figure_manifest.json` recording which artifact hashes produced which figure — the audit trail the config/io package will eventually formalize.

## pymoo built-ins stay as a quick-look, not the production path

`pymoo.visualization` provides `Scatter`, `PCP`, `Heatmap`, `Petal`, `Radar`, `Radviz`, `StarCoordinate`, and `Video` classes (Blank & Deb, *IEEE Access* 8:89497-89509, 2020; pymoo.org/visualization). For a 2-objective problem the `Scatter` class produces a passable front plot in three lines and is useful during development. However, the thesis requires full control over axis orientation (positive-is-better), custom knee annotation, feasibility overlay, and SVG font embedding — none trivially exposed through pymoo's wrapper. The production path is therefore raw matplotlib; pymoo's `Scatter` appears only in scratch notebooks. Pymoo also provides `util.running_metric.RunningMetricAnimation` and `mcdm.high_tradeoff.HighTradeoffPoints`, both of which we **do** use — for termination diagnostics and knee detection respectively.

## API skeleton

```python
# visualization/render.py
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")                              # set before pyplot import
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import numpy as np

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

def render_layout(layout, ax=None, *, catalog, layer_style=None, annotate=True):
    """Render a DecodedLayout: beds + dual rails, colored by piece type."""
    ax = ax or plt.gca()
    for p in layout.placements:
        spec = catalog[p.piece_id]
        color = OKABE_ITO[spec.color_index]
        alpha = {"main": 1.0, "branch": 0.7, "crossing": 0.7}[p.layer]
        ls = "--" if p.layer == "crossing" else "-"
        _draw_bed(ax, p, spec, color, alpha)         # Rectangle | Polygon | PathPatch
        _draw_rails(ax, p, spec, color, alpha, ls)   # LineCollection at ± gauge/2
    if annotate:
        _annotate_aabb_and_closure(ax, layout)
    ax.set_aspect("equal"); return ax

def render_pareto_front(F_hist, CV_hist, ax=None, *, knee=True):
    """Scatter -f1 vs -f2 with non-dominated highlight + knee annotation."""
    from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
    F, CV = F_hist[-1], CV_hist[-1]
    feas = CV.flatten() <= 0
    ax = ax or plt.gca()
    ax.scatter(-F[~feas, 0], -F[~feas, 1], marker="x", c="lightgray", s=10)
    F_feas = F[feas]; nd = _non_dominated_mask(F_feas)
    ax.scatter(-F_feas[~nd, 0], -F_feas[~nd, 1], c="gray", s=6, alpha=0.3)
    ax.scatter(-F_feas[nd, 0],  -F_feas[nd, 1],  c=OKABE_ITO[5], s=25)
    if knee and nd.sum() >= 3:
        k = HighTradeoffPoints()(F_feas[nd])
        ax.annotate("knee", xy=(-F_feas[nd][k[0], 0], -F_feas[nd][k[0], 1]))
    ax.set_xlabel("piece utilization"); ax.set_ylabel(r"$v_{\mathrm{bot}}$ (m/s)")
    return ax

def render_convergence(monitor_data, ax=None):
    """HV (left), IGD (right twinx), feasibility rate as top inset."""
    gens = np.asarray(monitor_data["n_gen"])
    ax = ax or plt.gca()
    ax.plot(gens, monitor_data["hv"], color=OKABE_ITO[5], label="HV")
    ax2 = ax.twinx()
    ax2.plot(gens, monitor_data["igd"], color=OKABE_ITO[6], ls="--", label="IGD")
    ax.set_xlabel("generation"); ax.set_ylabel("hypervolume")
    ax2.set_ylabel("IGD"); return ax

def render_speed_heatmap(layout, physics, ax=None, *, catalog):
    """Color each curve bed by v_safe(R, mu); inset shows v(R) with R* marker."""
    segs, vals = [], []
    for p in layout.placements:
        spec = catalog[p.piece_id]
        if not spec.is_curve: continue
        poly = _discretize_curve(p, spec, n=32)
        v = physics.v_bottleneck(spec.radius_studs)
        segs.extend(zip(poly[:-1], poly[1:])); vals.extend([v] * (len(poly) - 1))
    lc = LineCollection(segs, array=np.asarray(vals), cmap="viridis",
                        norm=plt.Normalize(0.60, 1.10), linewidth=2.5)
    ax = ax or plt.gca(); ax.add_collection(lc); ax.autoscale_view()
    _add_v_of_R_inset(ax, physics)               # R* and v_cap markers
    plt.colorbar(lc, ax=ax, label=r"$v_{\mathrm{safe}}$ (m/s)")
    return ax

def render_operator_usage(op_history, op_names, ax=None):
    ax = ax or plt.gca()
    gens = np.arange(op_history.shape[0])
    ax.stackplot(gens, op_history.T, labels=op_names, colors=OKABE_ITO[:len(op_names)])
    ax.set_xlabel("generation"); ax.set_ylabel("cumulative selections"); ax.legend()
    return ax

def save_thesis_figures(run_dir, output_dir):
    """Orchestrator: load artifacts, render all five, emit SVG+PNG, write manifest."""
    ...
```

A `VisualizationCallback(Callback)` is offered as an optional in-loop helper that renders `render_pareto_front` every N generations under the `Agg` backend; by default it is off, and the post-hoc path is canonical.

## Worked example: the 16 × R40 closed circle

Feed `render_layout` a `DecodedLayout` containing sixteen R40 curves (each 22.5° = π/8 rad, because 16 × 22.5° = 360°), all on the `main` layer, no branches, no crossings, `closure_residual ≈ (0, 0, 0)`. The rendered figure shows sixteen curved bed polygons arranged in a circle centered at roughly (0, 0); each bed is a concave fan discretized to ~24 vertices, filled in the Okabe-Ito "curve-medium" color (bluish green `#009E73`). Two concentric rail `LineCollection` polylines orbit at radii 37.5 and 42.5 studs (R ± gauge/2 = 40 ± 2.5). The annotated AABB sits at approximately (−42.5, −42.5, +42.5, +42.5) studs; the closure-residual annotation reads `Δ = (0.0, 0.0, 0.0)`. Because all curves share R = 40 studs < R\* ≈ 44 studs, running `render_speed_heatmap` on the same layout paints every segment near `v_safe = √(0.35 · 9.81 · 40 · 0.008) ≈ 1.05 m/s` — below `v_cap` but only just, and the inset's R\* marker falls to the *right* of every placed curve, showing that this canonical layout is in the friction-binding regime.

## Architectural decisions

| Decision | Rationale | Primary source |
|---|---|---|
| `Agg` backend for batch rendering | Matplotlib not thread-safe; `Agg` is the documented non-interactive path | matplotlib FAQ, "Working with threads" |
| `LineCollection` + discretized arcs, not `Arc` patches | `Arc.draw` optimization precludes `fill` and `facecolor` | matplotlib.patches.Arc docs |
| Okabe-Ito categorical palette | Color-vision-deficiency safe for all three CVD types | Okabe & Ito 2008; Wong *Nat Methods* 2011 |
| `viridis` continuous palette | Perceptually uniform, colorblind-safe, matplotlib default | van der Walt & Smith 2015; Crameri et al. 2020 |
| HV reference point r = (+0.10, −0.55) | Ten percent of range beyond nadir; defensible for roughly linear 2-objective front | Ishibuchi et al. 2018, *Evol Comput* 26(3) |
| Custom matplotlib over pymoo built-ins | Full control of axis orientation, knee annotation, SVG font embedding | Blank & Deb 2020; pymoo.org/visualization |
| Post-hoc rendering from persisted artifacts | Decoupling, reproducibility, thread safety | this package's design |
| SciencePlots + IEEE style base | Consistent thesis typography with minimal override | Garrett 2021, Zenodo DOI 10.5281/zenodo.4106649 |
| Knee detection via `pymoo.mcdm.HighTradeoffPoints` | Native, based on Deb-Gupta / Branke; avoids re-implementation | Bechikh et al. 2011, 2015 for broader context |

## What this pushes to downstream packages

Two obligations land on config/io (package #9). First, the figure manifest — `figure_manifest.json` per run recording artifact hashes, figure dimensions, DPI, colormap name, and reference point — becomes part of the persistence schema. Second, user-configurable style overrides (column width, font family, colormap for continuous fields) belong in a validated config section, not in Python source. The visualization package itself declares read-only dependencies on the catalog (for piece-type color indices), the geometry module (for AABB computation), the train module (for `v_bottleneck`), the decoder (for `DecodedLayout`), and the operators/problem callbacks (for histories); it writes only to disk. This makes visualization the purest "leaf" package in the dependency DAG — nothing upstream imports from it.

## What remains open

Three items are deliberately deferred. The **empirical attainment surface** across 20–30 seeds requires a multi-run sweep infrastructure that does not yet exist; the plotting code is a half-day addition once seeds are available, using OptunaHub's maintained port of Watanabe's package (the standalone GitHub repo is now archived). **3D Pareto plots** are not applicable — the problem has exactly two objectives, and synthesizing a third (e.g., aesthetic complexity) would be a thesis-scope change. **Interactive plotly/bokeh renderings** for the thesis defense slide deck are unresolved: matplotlib SVGs embed cleanly in Beamer but do not support hover tooltips, which the defense Q&A might benefit from; the decision is deferred until the defense format is chosen. None of these blocks the main thesis figures, which are fully specified by the API skeleton above.