# Designing a train-physics package for LEGO curve-speed optimization

**The `train/` package owns the second optimization objective f2 = min over curves of min(v_safe(R), v_cap), and its correctness hinges on three numerical anchors that this research has now verified, corrected, or bounded.** The curve-speed formula v = √(μgR) is not from Kurhan et al. (2024) — that attribution is wrong — but is the unsuperelevated limit of the standard point-mass design equation e + f_s = V²/(gR) codified in AASHTO's *Green Book* (7th ed., Chapter 3, Eq. 3-9) and embodied in railway regulation at 49 CFR §213.57(b)/§213.329. The motor cap v_cap = 1.10 m/s survives independent verification from both Philo's bench measurements and his in-situ oval-track tests once a wheel-diameter error in the original plan (30 mm vs. the correct 17 mm for part 55423c01) is corrected. The friction coefficient μ is the dominant uncertainty, with defensible injection-molded ABS values spanning roughly 0.25–0.45 (*Tribology International*, 2017); this uncertainty propagates into a **transition radius R\* = v_cap²/(μg) that moves from 35 studs at μ=0.45 up to 62 studs at μ=0.25**, straddling the catalog's smallest usable curve and thereby making μ the one parameter that can actually flip the bottleneck-winning layout. The package ships a bottleneck objective (safety semantics, backed by FRA per-curve V_max regulation) while documenting the length-weighted harmonic-mean alternative (throughput semantics, Wardrop 1952) for completeness.

## The curve-speed formula is old engineering, miscited in the plan

The equation v_max = √(μgR) is the e=0 special case of the **point-mass lateral-equilibrium design equation** that appears in every highway and railway geometric-design standard. The most defensible primary citation for a thesis is AASHTO's *A Policy on Geometric Design of Highways and Streets* (7th ed., 2018), Chapter 3 "Elements of Design," Eq. 3-9: **e + f_s = V²/(127·R)** in SI units, derived by equating centripetal acceleration to the sum of the banking-component and side-friction-factor contributions of gravity. Setting e=0 (LEGO track has no superelevation) collapses this to V² = 127·R·f_s, and identifying f_s with the lateral kinetic-friction coefficient μ gives the thesis formula directly.

Two secondary citations strengthen the case. **Esveld, *Modern Railway Track* (2nd ed., MRT-Productions, 2001), §3.2–3.3** ("Curvature and superelevation in horizontal curves") derives the cant-and-cant-deficiency version in railway-engineering notation: v² = g·R·(h+h_d)/G, where h is actual cant and h_d is cant deficiency; the cant-deficiency term behaves algebraically as the friction allowance. **49 CFR §213.329** ("Curves; elevation and speed limitations") codifies this into the regulatory formula V_max = √((E_a+E_u)/(0.0007·D)), where the 0.0007 constant is v²=gR(e+f) re-expressed in US railway units (mph, inches, degree-of-curvature) — algebraically identical after unit conversion. The FRA regulation is noteworthy because it applies the speed limit **per curve**, not in aggregate, which is the regulatory precedent for the bottleneck formulation chosen below.

| Source | Role | Form | Why it matters for LEGO |
|---|---|---|---|
| AASHTO *Green Book* 7th ed. (2018), Ch. 3, Eq. 3-9 | Primary engineering standard | e + f_s = V²/(127·R), SI | Clean algebraic reduction to v=√(μgR) at e=0 |
| Esveld *Modern Railway Track* 2nd ed. (2001), §3.2–3.3 | Railway-engineering textbook | v² = g·R·(h+h_d)/G | Rail-framed treatment; introduces cant deficiency as the friction analog |
| 49 CFR §213.57(b), §213.329 (FRA) | Regulatory | V_max = √((E_a+E_u)/(0.0007·D)) | Establishes **per-curve** binding of the speed limit — justifies bottleneck objective |
| Rajamani *Vehicle Dynamics and Control* (2012), Ch. 2, 15 | Vehicle-dynamics textbook | a_y = V²/R; static rollover threshold | Equivalent lateral-force balance, but not labelled as a single equation |
| Kurhan et al. (2024) | **Previously cited in error** | Curve alignment / least-squares fitting | Does NOT contain v=√(μgR); remove from references |

**The Kurhan et al. (2024) attribution in research_plan.md is confirmed incorrect and must be replaced.** That paper is on computer modelling for curve realignment, not speed-equilibrium physics.

## μ is uncertain by a factor of two; sensitivity sweep is not optional

The derailment-governing friction is **lateral**: the passive (undriven) wheel flanges scrubbing against the rail gauge face. On modern Powered Up and Power Functions plastic rails that contact is **ABS-on-ABS**. Rubber traction tires on the driven axle are irrelevant to this limit — they govern longitudinal traction, which is not what caps curve speed. Older 9V metal rails substitute plastic-on-nickel-silver, which is a different (and typically lower) μ regime; this is out of scope here but worth flagging for users of legacy systems.

Measured ABS-on-ABS friction coefficients span a wide band because grade, surface finish, load, sliding speed, and humidity all matter. The relevant regimes and their primary sources:

| Regime | μ range | Source |
|---|---|---|
| Injection-molded ABS, low load, against steel | **0.30–0.45** | Nuntapichedkul et al., *Tribology International* (2017), and Amrishraj & Senthilvelan, *Materials Today: Proceedings* (2020) |
| Injection-molded ABS self-mated (low speed / low heating) | ~0.35–0.50 | Difallah et al., *Materials & Design* 34 (2012) 782–787 |
| Injection-molded ABS self-mated (high speed / thermal runaway) | 0.74–0.98 | Amrishraj & Senthilvelan (2020); excluded — not the LEGO regime |
| FDM 3D-printed ABS (relevant to 3D-printed TrixBrix/4DBrix pieces) | **0.11–0.23** | Nuntapichedkul et al. (2017); Zadorozhniy et al., *DOAJ* (2023) |
| Generic plastic-on-plastic textbook default | ~0.5 | Zeus Industrial "Friction and Wear of Polymers" (2005) |

The LEGO flange-on-rail contact operates at **grams-force normal load, <1 m/s sliding speed, dry, ambient temperature, smooth injection-molded finish**, which sits in the middle of the injection-molded band and well below the thermal-runaway regime. A defensible three-point sensitivity sweep for the thesis is therefore:

| Sweep point | μ | Regime represented |
|---|---|---|
| μ_low (conservative) | **0.25** | Worst reasonable injection-molded ABS; FDM pieces would force μ_low → 0.15 |
| μ_nominal | **0.35** | Midpoint of injection-molded ABS literature (consistent with research_plan.md's 0.3 within measurement scatter) |
| μ_high (optimistic) | **0.45** | Upper injection-molded range; bounded well below self-mated thermal runaway |

The thesis should use **μ_nominal = 0.35** for headline results — it is the median of published measurements rather than an unsourced round number — and always report the μ=0.25 and μ=0.45 corners alongside. If any third-party layouts use FDM 3D-printed track (common with 4DBrix retro-fitted switches), the low corner must drop to 0.15 with an accompanying note that the μ=0.15 Pareto front is a separate sensitivity scenario, not a combined envelope.

## v_cap = 1.10 m/s survives independent verification after a wheel-diameter correction

Philo's motor comparison page (philohome.com/motors/motorcomp.htm) reports the Powered Up Train Motor (part 88011) at 9 V delivering **1760 RPM no-load**, **2.9 N·cm stall torque** (thermistor-extrapolated), and at the loaded 0.88 N·cm / 31%-efficiency operating point, **1242 RPM drawing 0.41 A and 3.73 W**. The original research plan's claim is verified verbatim.

The RPM-to-linear-velocity chain required one correction. The 88011 motor is a **self-contained gearbox whose output shafts are the wheel axles**, so the 1242 RPM figure is already the wheel speed — no further gear reduction applies. The plan's 30 mm wheel diameter, however, is the diameter of **large steam-engine driver wheels (part 85489a/b)**, not of the 88011's own wheels. The 88011 ships with part **55423c01** (spoked train wheel with rubber friction band) whose tread diameter is **17.0 mm** (L-Gauge Standards Wiki). The correct calculation is:

```
v_cap = 1242 RPM × π × 0.017 m / 60 s = 1.105 m/s
```

This result cross-checks against Philo's **in-situ oval-track measurement** (philohome.com/ttrain/ttrain.htm), where a 3-car train at 5.38 V averaged 0.69 m/s. DC motor speed is linear in voltage, so scaling gives 0.69 × (9/5.38) ≈ **1.15 m/s** at 9 V — a 4% agreement with the bench calculation. A 5-car (heavier) train scaled identically gives **1.03 m/s**, bracketing 1.10 m/s on both sides.

| Method | Source | v_cap at 9 V | Notes |
|---|---|---|---|
| Bench RPM × wheel circumference | Philo motorcomp (88011) | **1.10 m/s** | 1242 RPM × π × 17 mm; 31% efficiency point |
| In-situ oval, 3-car scaled to 9 V | Philo ttrain | **1.15 m/s** | Voltage-linear scaling from 5.38 V measurement |
| In-situ oval, 5-car scaled to 9 V | Philo ttrain | **1.03 m/s** | Heavier load |
| PF Train Motor 88002 (sibling) | Philo motorcomp | 1.30 m/s | Upper bound; PU ≈ 15% slower than PF |

Sariel's motor database (motors.sariel.pl) republishes Philo's figures rather than reporting independent measurements, so it corroborates rather than independently verifies. Brick Experiment Channel does not publish on-rail speed measurements for the PU train motor. Eurobricks community observations qualitatively confirm that stock 9V operation "runs a bit fast," consistent with the 178 km/h scale speed (1.10 m/s × 45 for L-gauge) implied by this v_cap — realistic for an intercity train and on the high end for a toy, which matches how AFOLs actually operate their PU systems at reduced PWM duty cycles.

**Ship v_cap = 1.10 m/s as the default, exposed as configurable for users who run at non-nominal voltage or with significantly different rolling stock.**

## The transition radius R* is the single most important derived parameter

The curve at which friction replaces motor capacity as the binding constraint follows from equating v_safe(R) to v_cap:

```
R* = v_cap² / (μ·g)
```

For v_cap = 1.10 m/s and the sensitivity sweep above:

| μ | R* (meters) | R* (studs) | Catalog radii friction-bound | Catalog radii motor-bound |
|---|---|---|---|---|
| 0.15 (FDM worst case) | 0.823 | **103** | R40, R56, R72, R88, ~R104 | R120, R148 |
| 0.25 (sweep low) | 0.494 | **62** | R40, R56 | R72, R88, R104, R120, R148 |
| 0.35 (nominal) | 0.353 | **44** | R40 | R56, R72, R88, R104, R120, R148 |
| 0.45 (sweep high) | 0.274 | **34** | *(none)* | **All** |
| 0.60 (optimistic cap) | 0.206 | **26** | *(none, below minimum)* | All |

Two observations dominate the downstream design. First, at μ_nominal = 0.35, **only LEGO R40 curves are friction-bound**; every other catalog radius is capped at v_cap. This means the f2 objective is in practice a two-valued function on most of the Pareto front: either the layout contains an R40 (friction bottleneck ≈ 1.05 m/s at μ=0.35) or it does not (everything caps at 1.10 m/s). Second, **mu=0.25 vs μ=0.45 flips R56 between friction-bound (0.886 m/s at μ=0.25) and motor-bound (1.40 m/s at μ=0.45)**. Any layout whose smallest curve is R56 will therefore change its f2 rank across the sensitivity sweep. This is the single most important sensitivity result the thesis must report.

Worked numerical example, f2 at μ=0.35 for two representative closed loops:

| Layout | Smallest R | f2 value | Binding constraint |
|---|---|---|---|
| R40-only small oval | 40 studs | **0.970 m/s** (friction) | v_safe(40, 0.35) = √(0.35·9.80665·0.32) |
| R56-only medium oval | 56 studs | 1.100 m/s | v_cap (motor) |
| R40 mixed with R88 | 40 studs | **0.970 m/s** (friction at the R40) | Single R40 piece dominates |
| All R88+ | 88 studs | 1.100 m/s | v_cap |

The R40-only and the R40+R88 mixed layouts tie at f2 = 0.970 m/s — a direct consequence of the bottleneck being a minimum over curves. This is a feature, not a bug: the optimizer is incentivized to eliminate R40 entirely, or to trade R40 for R56+ on the critical path.

## Bottleneck over harmonic mean: the regulatory and physical case

The thesis uses **f2 = min_i v_safe(R_i, μ) capped by v_cap**, selecting over *curves only*. Straights have R = ∞ and therefore v_safe = ∞; they never participate in the minimum. Switches contribute only on the diverging branch, where R = diverging_radius_studs; the through branch is geometrically straight and does not contribute. Crossings (+, X) are straight-through at both axes and also do not contribute.

The alternative — Wardrop's space-mean speed — is well-defined and has high-quality primary sources: **Wardrop, *Proc. Inst. Civil Engineers* 1(3), Part II, 325–362 (1952), DOI 10.1680/ipeds.1952.11259**, introducing SMS as the harmonic mean of spot speeds, and **Edie, in *Proc. 2nd Int'l Symp. on Theory of Traffic Flow* (1963), pp. 139–154**, generalizing to arbitrary space-time regions. The length-weighted form v = ΣL_i / Σ(L_i/v_i) is the natural definition of average speed over a multi-segment journey and appears explicitly in Garber & Hoel, *Traffic and Highway Engineering* (5th ed., Cengage, 2015), Ch. 6. The FHWA *Travel Time Data Collection Handbook* (Turner et al., FHWA-PL-98-035, 1998), Ch. 1, Eq. 1-3 gives the operational form SMS = n·d/Σt_i and recommends SMS specifically "when reducing and analyzing travel time data."

The reason the thesis prefers bottleneck is that these two statistics answer different questions:

| Indicator | Formula | Answers question | Failure mode |
|---|---|---|---|
| Bottleneck (min) | min_i v_safe(R_i) | What is the safe operating envelope? | One tight curve derails the train — cannot be compensated for |
| Length-weighted harmonic mean | ΣL_i / Σ(L_i/v_i) | What is the average travel time per lap? | A fast straightaway masks a dangerous curve |
| Arithmetic mean | (1/n)·Σv_i | *(Neither; dimensionally inconsistent with travel time)* | Not used in traffic engineering |
| Time-mean speed | (1/n)·Σv_i sampled in time | Spot-speed distribution character | Wardrop: always ≥ SMS, biased upward |

The bottleneck is the correct *safety* indicator because derailment is a **local threshold phenomenon**: lateral force at one specific curve either does or does not exceed the friction limit, independent of conditions elsewhere. This logic is codified in railway regulation: **49 CFR §213.57(b) sets V_max per individual curve**, and **§213.9 re-classifies an entire route to the class of its weakest sub-segment**. Averaging would permit a fast straight to mathematically cancel a dangerously tight curve in the objective function — precisely the failure mode one is trying to avoid. The harmonic-mean SMS remains the right auxiliary indicator if the thesis later adds a travel-time objective, but it does not replace the bottleneck for derailment-risk optimization.

## The physics API: one scalar function, one batch kernel, one cache

The package exposes a deliberately small surface area. The scalar function is the source of truth; the batch kernel and the speed table are vectorized / cached forms of the same mathematics.

```python
# train/physics.py

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

STUD_M: float = 0.008                 # 1 stud = 8 mm
G_STD: float = 9.80665                # standard gravity, m/s^2
MU_DEFAULT: float = 0.35              # injection-molded ABS nominal
V_CAP_DEFAULT: float = 1.10           # Powered Up 88011 loaded, m/s

@dataclass(frozen=True)
class PhysicsParams:
    mu: float = MU_DEFAULT
    g: float = G_STD
    v_cap: float = V_CAP_DEFAULT
    stud_m: float = STUD_M

@dataclass(frozen=True)
class PhysicsResult:
    v_bottleneck: float              # f2, m/s
    binding_piece_index: int         # which curve sets the bottleneck (-1 if motor-bound)
    friction_bound: bool             # True if v_safe < v_cap at the bottleneck
    transition_radius_studs: float   # R* for the current mu
```

The scalar form carries the whole physics model:

```python
def v_safe_one(R_studs: float, params: PhysicsParams = PhysicsParams()) -> float:
    """Safe traversal speed on a curve of radius R_studs.

    Implements v = min(sqrt(mu * g * R_m), v_cap) with R_m = R_studs * stud_m.
    Straights should not call this function; the bottleneck skips them.

    Source: AASHTO Green Book 7th ed. (2018), Ch. 3, Eq. 3-9 at e=0;
            Esveld, Modern Railway Track 2nd ed. (2001), §3.2-3.3;
            49 CFR §213.329.
    """
    if R_studs <= 0.0 or not np.isfinite(R_studs):
        return 0.0                   # degenerate — caller must guard
    v_friction = np.sqrt(params.mu * params.g * R_studs * params.stud_m)
    return min(v_friction, params.v_cap)
```

The speed table pre-computes values for the catalog's finite set of radii at startup; an LRU cache handles the occasional off-catalog radius:

```python
class SpeedTable:
    """Pre-computed v_safe for catalog radii; cache for anything else."""
    def __init__(self, params: PhysicsParams, catalog_radii_studs: list[float]):
        self.params = params
        self._table = {R: v_safe_one(R, params) for R in catalog_radii_studs}
        self.R_star_studs = params.v_cap**2 / (params.mu * params.g * params.stud_m)

    @lru_cache(maxsize=None)
    def v_safe(self, R_studs: float) -> float:
        hit = self._table.get(R_studs)
        return hit if hit is not None else v_safe_one(R_studs, self.params)
```

The problem-layer consumer is the bottleneck evaluator over a placement list with resolved switch routes:

```python
def v_bottleneck(placements, speed_table: SpeedTable) -> PhysicsResult:
    """Evaluate f2 = min over curves of v_safe(R_i).

    `placements` is the decoder output: an iterable of PlacedPiece objects with
    `.curve_radius_studs` -> float or None (None for straights and for switches
    resolved to the through branch; the effective diverging radius for switches
    resolved to the diverging branch; the catalog radius_studs for curves).
    Crossings contribute None.
    """
    v_min, idx_min = speed_table.params.v_cap, -1
    for i, p in enumerate(placements):
        R = p.curve_radius_studs
        if R is None:
            continue                                    # straights / through switches / crossings
        v = speed_table.v_safe(R)
        if v < v_min:
            v_min, idx_min = v, i
    return PhysicsResult(
        v_bottleneck=v_min,
        binding_piece_index=idx_min,
        friction_bound=(idx_min >= 0 and v_min < speed_table.params.v_cap),
        transition_radius_studs=speed_table.R_star_studs,
    )
```

The batch kernel for NSGA-II population evaluation vectorizes over N layouts with up to S segments, using a sentinel (np.inf) for non-curves:

```python
def batch_v_bottleneck(R_matrix: np.ndarray, params: PhysicsParams) -> np.ndarray:
    """Shape (N, S) -> shape (N,).

    R_matrix[n, s] = radius in studs for curve s of layout n, or np.inf for
    straights / unused slots / through switches / crossings. The decoder
    populates this array after route selection.
    """
    R_m = R_matrix * params.stud_m
    v_fric = np.sqrt(params.mu * params.g * R_m)        # inf where R = inf
    v_safe = np.minimum(v_fric, params.v_cap)
    return v_safe.min(axis=1)                           # masked by the inf sentinel
```

| API member | Role | Consumer |
|---|---|---|
| `PhysicsParams` | Frozen immutable configuration (μ, g, v_cap, stud_m) | All |
| `v_safe_one(R_studs, params)` | Scalar source of truth | Tests, visualization tooltips |
| `SpeedTable` | Pre-computed cache for catalog radii | problem layer |
| `v_bottleneck(placements, table) -> PhysicsResult` | Single-layout f2 | problem.evaluate |
| `batch_v_bottleneck(R_matrix, params) -> (N,)` | Vectorized f2 across population | problem.evaluate when NSGA-II calls in bulk |
| `R_star_studs` attribute | Transition radius for current μ | visualization overlay, reports |

## Switch-route semantics are resolved by the decoder, consumed by train

The chromosome's switch-mask region encodes, for each switch, whether the main traversal route is **through** or **diverging**. The physics package never sees the raw bits. The decoder is the single point of switch-route resolution: it consumes the switch mask and produces a placement list in which each piece carries a single `curve_radius_studs` value — `None` for straights, the catalog `radius_studs` for curves, and for switches the **effective radius of the selected route** (None for through, `diverging_radius_studs` for diverging). This keeps the physics package route-oblivious and makes f2 a pure function of the placement list.

The consequence for f2 is a semantic one: on a layout where a switch is set to diverging with R = diverging_radius_studs = 40 studs, that switch contributes to the bottleneck. On the same layout with the switch set to through, it does not. The chromosome therefore influences f2 even without changing which pieces are placed, which is a feature — the optimizer learns to prefer through-routed switches when tight diverging radii would pin f2 low.

## Units are studs internally, meters at the physics boundary

Chromosome, catalog, and decoder work in studs throughout. The only place meters appear is inside `v_safe_one` and `batch_v_bottleneck`, where `R_m = R_studs * STUD_M`. Speed is always in m/s externally, matching v_cap's manufacturer-data units. This keeps the unit-conversion surface minimal and localized: a single multiplication at the friction calculation, with STUD_M = 0.008 as the sole conversion constant. No overflow or underflow concerns arise; the largest catalog radius of 148 studs gives R_m ≈ 1.184 m and v_friction ≈ 2.04 m/s at μ=0.35, well within double-precision range and always capped by v_cap downstream anyway.

## What the single-point quasi-static model deliberately does not capture

The f2 objective is quasi-static: at each curve the train is treated as a point mass in steady circular motion at the speed it would travel there if it were in isolation. Seven real effects are excluded by this choice, and the thesis must document each with a citation to the fuller model:

First, **train inertia at speed transitions** — the model assumes instantaneous speed changes at curve/straight boundaries, whereas a real train cannot decelerate infinitely fast. For a GA fitness function this is acceptable because the bottleneck is a worst-case indicator and adding deceleration constraints would only *tighten* the safe speed, never relax it. A thesis-rigorous extension would overlay a constant longitudinal deceleration limit (e.g., a_x ≈ 0.5 m/s² as a plausible hand-brake value) and solve for the pre-curve braking distance; see Iwnicki et al., *Handbook of Railway Vehicle Dynamics* (2nd ed., CRC Press, 2019), Ch. 2 §III "Concepts of Curving."

Second, **wheelbase and bogie effects on flange force** — a real train has leading and trailing axles at different radii through a curve, and the lateral force is concentrated on the outer flange of the leading axle. The single-point model absorbs this into the effective μ. For richer modelling, see Iwnicki Ch. 3 on wheel-rail contact mechanics. Per user directive, locomotive geometry is a fixed scope choice and not an optimization variable, so this simplification is deliberate.

Third, **wheel-rail contact mechanics** — the real contact is an elliptical Hertzian patch with longitudinal and lateral creep governed by Kalker's theory, not a dry-friction Coulomb contact. For LEGO's plastic-on-plastic interface this distinction is less important than for steel-on-steel (no creep-based curving guidance), but the richer model lives in Iwnicki Ch. 3 and Ayasse & Chollet's chapter therein.

Fourth, **flange climb** — the failure mode on very tight curves with low lateral friction is actually flange climb (the outer wheel rides up the rail gauge face) rather than pure lateral sliding. The climbing criterion is the Nadal formula, tan(γ) > (μ_lateral − μ_longitudinal)/(1 + μ_lateral·μ_longitudinal), with flange angle γ. For LEGO flanges (γ ≈ 60–70°) Nadal's criterion is satisfied with enormous margin at any realistic μ, so simple lateral-friction limits dominate in practice.

Fifth, **motor electromechanics at low speed** — Philo's 1242 RPM loaded operating point assumes steady-state. A full dynamic model (back-EMF, armature inductance, time-varying load) is in **Wick & Ramsdell, *Am. J. Phys.* 72(7), 863–874 (2004), DOI 10.1119/1.1637040** (*Note: this DOI supersedes the incorrect 10.1119/1.1703542 guess in research_plan.md*). That paper models a generic horizontal-track educational toy train, not LEGO specifically, and provides no μ value — but its methodology for extracting motor parameters from steady-state and transient measurements is the template if the thesis later adds a time-domain drivetrain model.

Sixth, **aerodynamic drag** — at 1.10 m/s with LEGO cross-sectional area ~10 cm², drag force is ≈ ½ρv²CA ≈ 10⁻⁵ N, completely negligible versus gram-scale normal forces. See Gillespie, *Fundamentals of Vehicle Dynamics* (SAE, 1992), Ch. 4 for the drag formulation if ever needed.

Seventh, **rolling resistance and wheel wear** — both affect power consumption and long-term reliability but not derailment safety. Out of scope.

## Numerical checks: worked f2 values at the catalog radii

Complete v_safe table at μ = 0.35 (nominal) and μ = 0.45 (upper sensitivity), with v_cap = 1.10 m/s:

| R (studs) | R (m) | v_friction @ μ=0.25 | v_safe @ μ=0.25 | v_friction @ μ=0.35 | v_safe @ μ=0.35 | v_friction @ μ=0.45 | v_safe @ μ=0.45 |
|---|---|---|---|---|---|---|---|
| 40 | 0.320 | 0.886 | **0.886 F** | 1.049 | **1.049 F** | 1.188 | 1.100 M |
| 56 | 0.448 | 1.048 | **1.048 F** | 1.241 | 1.100 M | 1.406 | 1.100 M |
| 72 | 0.576 | 1.188 | 1.100 M | 1.405 | 1.100 M | 1.593 | 1.100 M |
| 88 | 0.704 | 1.314 | 1.100 M | 1.554 | 1.100 M | 1.762 | 1.100 M |
| 104 | 0.832 | 1.429 | 1.100 M | 1.690 | 1.100 M | 1.916 | 1.100 M |
| 120 | 0.960 | 1.534 | 1.100 M | 1.815 | 1.100 M | 2.058 | 1.100 M |
| 148 | 1.184 | 1.704 | 1.100 M | 2.015 | 1.100 M | 2.285 | 1.100 M |

**F** = friction-bound (f2 is set by friction); **M** = motor-bound (f2 is capped at v_cap). The practical range of f2 values a layout can produce is approximately 0.89 m/s (R40 at μ=0.25) to 1.10 m/s (anything motor-bound). This 20% dynamic range is enough for NSGA-II to trace a meaningful Pareto front against f1 = piece utilization, but only if the catalog is allowed to include the tightest radii; a catalog restricted to R72+ renders f2 constant at v_cap under nominal μ, collapsing the optimization to single-objective.

## Architectural decisions for the train package

| Decision | Choice | Rationale |
|---|---|---|
| Curve-speed formula source | AASHTO Green Book Eq. 3-9 at e=0; Esveld §3.2–3.3; 49 CFR §213.329 | Replaces incorrect Kurhan 2024 citation |
| μ default | 0.35 (injection-molded ABS median) | Median of *Tribology International* (2017) and *Materials & Design* (2012) primary data |
| μ sensitivity sweep | {0.25, 0.35, 0.45} standard; {0.15, 0.35, 0.60} if FDM track is allowed | Covers the injection-molded scatter band; labeled 3D-printed scenario separate |
| v_cap default | 1.10 m/s (PU 88011 at 9 V nominal) | Cross-verified Philo bench + in-situ oval; corrected wheel diameter |
| f2 aggregator | Bottleneck (min over curves) | FRA §213.57(b) per-curve logic; safety semantics; local threshold phenomenon |
| Alternative documented | Length-weighted harmonic mean, Wardrop (1952) | Throughput semantics, for future travel-time objective |
| Speed table caching | Eager pre-compute for catalog radii + LRU for off-catalog | Catalog radii set is small (~7 values), 100% hit rate in steady state |
| Batch evaluation shape | (N, S) float32 with np.inf sentinel for non-curves | Masking-free min via infinity |
| Unit strategy | Studs internal, meters at physics boundary only | Single conversion point at v_safe_one |
| Quasi-static scope | Deliberate simplification | Point mass; no inertia, no flange climb, no bogie; documented limitations |
| Switch-route consumption | Resolved by decoder; train sees `curve_radius_studs` or None | Physics package is route-oblivious |

## What this pushes to the downstream packages

The **problem** package consumes `v_bottleneck(placements, SpeedTable) -> PhysicsResult` as its f2 evaluator and `batch_v_bottleneck(R_matrix, params)` for vectorized population fitness. f2 is the negative of the PhysicsResult's `v_bottleneck` if NSGA-II is formulated as minimization. The layout-closure equality constraint remains with **geometry** and is not re-asserted here.

The **decoder** stays physics-free per the architectural rule, but it is now the single owner of switch-route resolution: after reading the chromosome's switch mask, it emits placements whose `curve_radius_studs` is None (straight / through switch / crossing) or a radius (curve / diverging switch). The physics package's route-obliviousness is maintained by this contract.

The **visualization** package gains a speed-heatmap overlay that colors each piece by its v_safe value, and a transition-radius annotation R\* = v_cap²/(μg) to mark visually which curves are friction-bound versus motor-bound. Under the nominal sweep only R40 curves should be friction-bound, so the heatmap will be dichotomatic for most layouts — a useful diagnostic for spotting optimizer regressions where R40 pieces creep back into near-optimal solutions.

The **config / io** package must expose μ, μ-sensitivity sweep points, and v_cap as user-configurable parameters (with the defaults 0.35, {0.25, 0.35, 0.45}, and 1.10 m/s respectively) and must propagate STUD_M = 0.008 as a single system-wide constant. The μ parameter must be surfaced prominently in any results file because two Pareto fronts computed at different μ values are not comparable and must be clearly labeled.

## Known limitations, concisely

The physics model excludes train-inertia effects at segment boundaries (Iwnicki 2019, Ch. 2), wheel-rail contact mechanics and creep (Iwnicki Ch. 3), flange-climb via Nadal's criterion (Iwnicki Ch. 3), electromechanical motor dynamics (Wick & Ramsdell 2004), aerodynamic drag (Gillespie 1992, Ch. 4), rolling resistance, and wheel wear. The quasi-static point-mass assumption is a deliberate scope choice appropriate for a GA fitness function where the objective should be fast to evaluate, monotone in the parameters it actually depends on, and a conservative proxy for the physical phenomenon. Under the injection-molded ABS μ sweep and the PU 88011 v_cap, only R40 curves differ between bottleneck and v_cap, meaning the entire optimization's friction sensitivity lives at a single decision boundary — a property that makes the Pareto-front sensitivity analysis tractable with a three-point μ sweep and that justifies the modeling choice economically.