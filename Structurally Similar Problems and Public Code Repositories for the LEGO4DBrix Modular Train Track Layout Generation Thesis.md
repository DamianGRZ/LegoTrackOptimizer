# Structurally Similar Problems and Public Code Repositories for the LEGO/4DBrix Modular Train Track Layout Generation Thesis

This report maps the thesis problem — **closed-loop, multi-objective, BRKGA-style modular track layout generation on a discrete lattice with port-snapped pieces from a finite inventory** — to existing public code repositories across hobbyist, academic, and game-dev communities. Repos with runnable code are prioritized over papers without code; for each entry the structural similarity to the thesis, the algorithm used, license/maintenance status (when discoverable), and any technique that transfers to the thesis are noted.

A short headline first: **no published academic work and no open-source hobbyist tool was found that automatically *generates* a closed-loop, inventory-constrained, switch-bearing LEGO/4DBrix track plan.** All extant LEGO/L-Gauge tooling is interactive CAD (BlueBrick, 3D Train Studio, SCARM, Track Designer 2.0, nControl, Stud.io). The closest existing automatic generator is a brute-force enumerator for the trivial single-loop case on Duplo/IKEA tracks (mmm444/trains). The thesis therefore appears to genuinely occupy an empty niche; the most useful transferable code lives in adjacent domains (procedural racetrack generation, WFC with path/loop constraints, BRKGA combinatorial decoders, multi-floor dungeon generation, OpenTTD railway AIs, and the Closed-Loop Layout Problem in manufacturing).

---

## 1. LEGO / Model Railroad Track Layout Tools and Generators

### 1.1 mmm444/trains — IKEA LILLABO & LEGO DUPLO track generator (★★ closest match)
- **URL:** https://github.com/mmm444/trains
- **Language / License:** Go / WTFPL
- **Stars / Activity:** 4 stars, 1 fork, 2 commits, dormant
- **What it does:** Brute-force search that enumerates *all* closed-loop track layouts buildable from a fixed inventory of IKEA LILLABO and LEGO DUPLO pieces (straights, curves, switches optional). Outputs HTML and SVG. Built as a faster successor to John Graham-Cumming's 2010 blog post on the LILLABO problem.
- **Structural similarity:** Very high on three dimensions — (i) closed-loop (positional + angular closure), (ii) finite inventory of fixed-angle snap pieces, (iii) discrete lattice. **Does not** handle multiple loops, branches off switches, or multi-objective optimization. Single-loop only, brute-force, no BRKGA / NSGA-II.
- **Transferable insight:** Demonstrates the tolerance handling and the closure check on a coarser angular lattice (DUPLO uses 30°, LILLABO uses 45°). Branch-pruning ideas in `trains.go` are directly relevant to the thesis decoder's early-failure check.

### 1.2 BlueBrick (Lswbanban/BlueBrick) — Open-source LEGO layout CAD
- **URL:** https://github.com/Lswbanban/BlueBrick (main fork; original by Alban Nanty at https://bluebrick.lswproject.com/)
- **License:** GPL v3
- **Status:** Actively maintained by Alban Nanty; community libraries on GitHub at https://github.com/l-gauge/bluebrick-lib (l-gauge org) and https://github.com/NickAb/lego-community-open-source-projects index.
- **Structural similarity:** It is *manual* CAD, not a generator, but its data model encodes precisely the port-based snap geometry the thesis needs (connection points with angles, the π/32 atom, and 4DBrix/Track Designer file-format compatibility). Reading BlueBrick's parts library XML/INI is the fastest way to import a real, validated 4DBrix-aligned inventory into a Python decoder.
- **Transferable insight:** The "ConnectionTypeList.xml" and per-part connector definitions can be converted directly into the port-pose/angle table the turtle decoder consumes. MattzoBricks maintains the most complete community track libraries (TrixBrix, BrickTracks, ME Models): https://mattzobricks.com/lego-track-planning/bluebrick.

### 1.3 l-gauge/ldraw-lib & l-gauge/bluebrick-lib — community track libraries
- **URLs:** https://github.com/l-gauge/ldraw-lib , https://github.com/l-gauge/bluebrick-lib
- **License:** community licensed, open
- **Use:** Authoritative source for R40/R56/R72/R88/R104 curve geometry and switch geometry that match the thesis inventory. These two repos plus brickdimensions.com's "Schematic" libraries cover essentially every L-gauge track piece used by AFOLs.

### 1.4 Lego Track Designer 2.0 (Mathew D. Bates, 1998)
- **URL (binary):** https://archive.org/details/legotd2 (archived; closed source, runs under Win10)
- **Status:** Abandoned, closed-source. Predecessor of TrackDraw → BlueBrick. Useful only as a file-format reference; BlueBrick can read its files.

### 1.5 4DBrix nControl, 3D Train Studio, SCARM, AnyRail, XTrkCAD
- **4DBrix nControl:** Closed source, free, downloads currently broken on www.4dbrix.com. Provides 4DBrix-compatible track planning + automation but is *not* a generator.
- **3D Train Studio:** Commercial Windows tool used by MattzoBricks; integrates with Rocrail. Closed source.
- **SCARM:** Closed-source, free for layouts up to ~100 pieces; large Lego track-plan database at https://www.scarm.info/layouts/track_plans.php?tracks=Lego, all hand-designed.
- **XTrackCAD:** Open source (GPL), C, https://sourceforge.net/projects/xtrkcad-fork/. The most relevant *open-source HO-scale* analogue. Manual CAD, not a generator, but its turnout-geometry tables and "easement" code address the same compound-S-curve problem that LEGO R40 switches embody.
- **Traintastic** (https://traintastic.org/): Open-source model-railroad *control* (not layout planning), included for completeness of the ecosystem.

### 1.6 mattzobricks/SpikeShield-and-libraries / TrixBrix Bluebrick library
- **URL:** https://mattzobricks.com/lego-track-planning/bluebrick (track libraries downloadable; the parent MattzoBricks code is on GitHub at https://github.com/mattzobricks).
- **Status:** Actively maintained (TrixBrix R104 curved switches added recently). Best-in-class data on TrixBrix-extended geometries, which 4DBrix-style users care about.

### 1.7 Eurobricks community thread & blogs
- **Eurobricks "Simple track layout tool" (Apr 15 2025):** https://www.eurobricks.com/forum/forums/topic/202355-simple-track-layout-tool/ — confirms ongoing community demand for an automatic generator; user "cluening" enumerates layouts manually.
- **bricklayouts.com (https://app.bricklayouts.com/):** New (Aug 2025) web-based, closed-source, in beta — interactive only.
- **Monty's Trains (montystrains.net)** and **Bricks McGee LEGO track geometry primer** confirm explicitly that *"No, such software [an automatic LEGO track layout generator] does not exist."* This is documented in a 2018 blog post that still appears not to have been contradicted by any subsequent open release.
- **AIIrondev/Track-Generator** (https://github.com/AIIrondev/Track-Generator) — *not* a layout generator; it generates SPIKE Prime robot driving paths. Listed only because its name causes confusion in searches.

**Bottom line for category 1:** mmm444/trains is the only existing *automatic* code generator for LEGO/Duplo/IKEA train track layouts, and it covers only the single-loop case with no inventory-constrained multi-objective optimization. The thesis problem is, as the user suspected, an open niche.

---

## 2. Procedural Racetrack / Roller-Coaster Generators with Closed-Loop Constraints

### 2.1 LasseHenrich/racetrack-generation — Unity, closed arcade tracks (★ excellent match)
- **URL:** https://github.com/LasseHenrich/racetrack-generation
- **What it does:** Unity tool that procedurally generates **closed**, arcade-like, customizable race tracks with **explicit handling of crossings/intersections** (the "constrained-space + repulsive-curve" model). Exposes "manual" and "advanced" modes letting users override crossings and constraints.
- **Structural similarity:** Closed-loop ✓, multiple tracks/intersections ✓, crossings ✓. Continuous spline (not lattice / port-snapped) — but the *crossing geometry handling* is the closest open-source analogue of the thesis's switch/crossing topology problem.
- **Transferable insight:** The "constrained space + potentials" formulation is a clean way to encode "branches must rejoin" as an objective rather than a hard constraint — relevant if the thesis ever wants to soften angular closure into an objective.

### 2.2 juangallostra/procedural-tracks — Convex-hull spline racetrack
- **URL:** https://github.com/juangallostra/procedural-tracks
- **License:** open
- **Algorithm:** Random points → convex hull → midpoint displacement → spline interpolation (Gustavo Maciel's algorithm).
- **Companion blog:** https://bitesofcode.wordpress.com/2020/04/09/procedural-racetrack-generation/
- **Similarity:** Single closed loop, continuous (not modular). Useful as a baseline / contrast.

### 2.3 ArthurFDLR/race-track-generator — Fourier-descriptor + GAN
- **URL:** https://github.com/ArthurFDLR/race-track-generator
- **Algorithm:** GAN trained on Fourier descriptors of real F1 circuit boundaries; the Fourier basis *enforces closure by construction* (any finite-frequency sum is automatically a closed curve).
- **Transferable insight:** Closure-by-construction via a chosen basis is exactly the philosophy behind the thesis's turtle decoder. The Fourier-descriptor trick — design the parameterization so that *every* point in chromosome space decodes to a closed loop — is a clean alternative to the BRKGA random-key approach for problems where exact angular closure is hard to enforce.

### 2.4 Other racetrack repos (use as reference, lower priority)
- **bogdankharchenko/racetrack-generator** (https://github.com/bogdankharchenko/racetrack-generator) — JS port of opengameart's procgen track.
- **Drallig/ProceduralTrack** (https://github.com/Drallig/ProceduralTrack) — Unity, A* + Perlin terrain.
- **i-hudson.github.io/projects/2019-02-02-Race-track-Generator/** — Voronoi-based generator (interesting because the *adjacent-cell-walk* idea is structurally close to a port-graph traversal).
- **Zacros/procedural-tracks-dataset-generator** (https://github.com/Zacros/procedural-tracks-dataset-generator) — Formula-Student style dataset generator.

### 2.5 kevinburke/rct — Genetic algorithm for Roller Coaster Tycoon 2 tracks (★ very high BRKGA-adjacent value)
- **URL:** https://github.com/kevinburke/rct
- **Language:** Go
- **Algorithm:** Genetic algorithm with mutation, crossover, fitness function over a *fixed inventory of RCT2 track pieces* (each piece carries metadata: elevation change, turn direction). Fitness checks "is the track a complete loop and doesn't collide with itself."
- **Structural similarity:** **Highest of any GitHub repo to the thesis problem in spirit.** Modular pieces from a finite catalog ✓, closed-loop constraint ✓, GA-based search ✓, integer chromosome of pieces ✓. Differs in: 3D, no inventory cap, single-objective, naive GA (not BRKGA).
- **Transferable insight:** The piece-metadata model and the loop-closure fitness check translate almost directly. The repo's note that "tracks fail to load because collision detection is wrong" is a useful warning for the thesis — invariants that look right in 2D break under tolerance.

### 2.6 CMDRSpirit/RollercoasterDesigner — Unity coaster CAD
- **URL:** https://github.com/CMDRSpirit/RollercoasterDesigner — spline-based, NoLimits 2 import. Manual, not a generator, but the procedural track-mesh generation along a spline mirrors what 4DBrix users want for visualization.

### 2.7 pfroud/rollercoaster-designer / TommyTeaVee/rollercoaster-designer
- **URL:** https://github.com/pfroud/rollercoaster-designer (and fork TommyTeaVee/rollercoaster-designer)
- **Algorithm:** Interactive three.js coaster from modular pieces (`Piece` class with pre/post-correction metadata so pieces snap exactly). Per-piece "TrackConst" table is exactly the data structure the thesis needs.
- **Similarity:** Modular-piece + snap-correction model.

### 2.8 gorzen.github.io Procedural Roller Coaster (L-System + Blender)
- **URL:** https://gorzen.github.io/ressources/computer-graphics/report.html
- **Algorithm:** Stochastic L-system whose production rules are **designed so that the loop always closes** (the rule's end-point lies a fixed length ahead with a fixed orientation). The "END" symbol forces rotational continuity.
- **Transferable insight:** Designing grammar/decoder rules with built-in closure invariants is the exact strategy of the thesis decoder. This is a small but very direct conceptual analogue.

---

## 3. Closed-Loop Layout Problem (CLLP) in Manufacturing

The CLLP is the academic cousin of the thesis problem (place rectangular cells around a closed material-handling loop). Code repositories specifically for CLLP are scarce — most papers are MILP- or simulated-annealing-only with no public code:

- **xNok/OR_location_routing_problem_study** — https://github.com/xNok/OR_location_routing_problem_study — IPython notebooks with CPLEX-API MILP formulations of facility location and location-routing problems (siblings of CLLP). Useful as the most accessible code-bearing reference.
- Key papers without public code: ScienceDirect's *"Loop layout design problem in flexible manufacturing systems using genetic algorithms"* (https://www.sciencedirect.com/science/article/abs/pii/S0360835297001502), IEEE *"Design of a manufacturing facility layout with a closed loop conveyor with shortcuts using queueing theory and genetic algorithms"* (https://ieeexplore.ieee.org/document/6147910/). These motivate the problem and provide GA recipes that can be reimplemented in pymoo.

---

## 4. Multi-Floor / Multi-Level Floor-Plan and Dungeon Generation (the analogue requested)

### 4.1 vazgriz/Dungeon-Generator (and accompanying blog) — multi-floor TinyKeep extension (★ best multi-floor analogue)
- **URL:** https://vazgriz.com/119/procedurally-generated-dungeons/ (links to the GitHub repo from the post)
- **Algorithm:** Extends TinyKeep's 2D dungeon generator to 3D with explicit **stair handling** that resolves the same problem the thesis faces with switches: "the pathfinder must move from the start to the end of a staircase in one step, but later iterations must work around staircases that have already been placed." Solution: each node tracks its previous path so other paths can't cut through a staircase mid-construction.
- **Structural similarity to the multi-loop / multi-level aspect of the thesis:** Excellent. Multiple "floors" = multiple independent loops; stairs = switches connecting them; the bookkeeping required to keep one stair from being used twice is mathematically the same as ensuring branch tracks rejoin without crossing themselves.
- **Algorithm uses 3D Delaunay tetrahedralization + MST + corridor pathfinding.**

### 4.2 AmanSachan1/InterestingLevelGenerator — multi-layer Voronoi dungeon
- **URL:** https://github.com/AmanSachan1/InterestingLevelGenerator
- **Algorithm:** Stack of 2D Voronoi-graph maps connected by 3D walkways/paths between layers. Explicitly handles inter-layer connectivity (random "fromNode → toNode" edges across floors).
- **Similarity:** Multi-level connectivity is the analogue of multi-loop interconnection via switches.

### 4.3 tomnullpointer/multi-floor-dungeon-generator (itch.io, browser, source available)
- **URL:** https://tomnullpointer.itch.io/multi-floor-dungeon-generator
- **Algorithm:** Extension of the well-known donjon.bin.sh dungeon to multiple floors with spiral staircases connecting rooms across levels.
- **Similarity:** Direct multi-level analogue with explicit vertical connectivity.

### 4.4 shun126/DungeonGenerator — Unreal Engine 5 plugin
- **URL:** https://github.com/shun126/DungeonGenerator
- **License:** Open-source UE5 plugin, multi-floor, stair meshes, key/door/route progression via "MissionGraph."
- **Similarity:** Production-grade multi-floor generator with explicit stair connectivity and progression graph; the MissionGraph concept maps to "ensure each siding is reachable from the main loop."

### 4.5 PeterBennyFooda/ProceduralDungeon_UE5_Showcase
- **URL:** https://github.com/PeterBennyFooda/ProceduralDungeon_UE5_Showcase
- **Description:** Multi-floor dungeon system with parameters for "number of staircases per floor" and "minimum room count per floor." Direct mapping of "switches per loop" / "minimum sidings per layout."

### 4.6 watabou/one-page-dungeon (itch.io)
- **URL:** https://watabou.itch.io/one-page-dungeon (also pixijs/web demos linked from watabou.github.io). Closed source. Single-floor; the comments thread explicitly discusses adding multi-floor support and the difficulty of stair alignment between irregularly shaped levels — useful design-discussion reference.

### 4.7 orphu/mcdungeon — Minecraft procedural dungeons with stairwells
- **URL:** https://github.com/orphu/mcdungeon
- **Description:** Multi-level Minecraft dungeons; "places stairwells between levels, and a random entrance with a spiraling staircase." Highly configurable (# rooms, loops, traps). License: MIT.

### 4.8 Floor-plan generation (architectural; mostly single-floor but useful)
- **House-GAN** (https://github.com/ennauata/housegan) and **House-GAN++** (https://ennauata.github.io/houseganpp/page.html) — graph-constrained relational GAN for floorplan generation; bubble diagrams model adjacency the same way port-graphs model snap connectivity.
- **Graph2Plan** (https://github.com/HanHan55/Graph2plan) — GCN+CNN floorplan generation from layout graph + boundary; trained on RPLAN (80k floorplans).
- **HouseDiffusion** (https://github.com/aminshabani/house_diffusion) — diffusion model on RPLAN, vector floorplans.
- **FloorplanGAN** (https://github.com/luozn15/FloorplanGAN) — vector residential floor plans via differentiable rendering.
- **MSD / caspervanengelenburg/msd** (https://github.com/caspervanengelenburg/msd) — **Modified Swiss Dwellings, ECCV 2024**: the *first large-scale **multi-apartment / multi-floor** floor-plan dataset*. 5,300 plans, 18,900 apartments, includes vertical circulation. Datasets + benchmark code are open. Most relevant academic dataset for the multi-floor question.
- **mo7amed7assan1911/Floor_Plan_Generation_using_GNNs** (https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs) — graduation-project GNN floorplan generator with attention.
- **z-aqib/Floor-Plan-Generator-Using-AI** (https://github.com/z-aqib/Floor-Plan-Generator-Using-AI) — CSP-based floorplan generator (Python+Java); useful because CSP is precisely how port-snap constraints are formalized.
- **SebGr/fml-wright** (https://github.com/SebGr/fml-wright) — staged GAN floorplan generation (pix2pix / BiCycleGAN).
- **yuntaeJ/SkipNet-FloorPlanGen** — winning solution for the MSD CVAAD challenge.

### 4.9 MuNES — Multifloor Navigation Including Elevators and Stairs (Donghwi Jung et al.)
- **Paper:** https://arxiv.org/pdf/2402.04535
- **Code:** https://github.com/donghwijung/MuNES
- **Description:** Single multi-floor map for robot trajectory planning with A* over voxelized stairs+elevators; explicitly tackles inter-floor connectivity with TSP-related cost functions. Good code reference for the *vertical-connection* objective.

### 4.10 3D-IC / multi-die VLSI floorplanning (the EDA cousin)
- **IFTE-EDA/Corblivar** (https://github.com/IFTE-EDA/Corblivar) — Knechtel's 3D IC floorplanning suite & benchmarks (multi-die stacks, TSV planning, thermal). C++ open source. The *multiple stacked dies + through-silicon-via* abstraction is mathematically isomorphic to "multiple track loops + switches across them."
- **Open3DBench** (https://arxiv.org/html/2503.12946v1, code referenced in the paper) — open 3D-IC backend benchmark.
- See also Cuesta et al., *Thermal-Aware Floorplanner for 3D IC* (https://arxiv.org/pdf/2402.14627) — multi-objective evolutionary 3D floorplanning.

---

## 5. Modular / Tile-Based Generation with Snap Constraints (Wave Function Collapse and friends)

### 5.1 mxgmn/WaveFunctionCollapse — original WFC (★ foundational)
- **URL:** https://github.com/mxgmn/WaveFunctionCollapse
- **License:** MIT-style, very active community
- **Why it matters:** WFC's adjacency-constraint propagation is exactly the mechanism for handling port-snap rules ("R40-Curve's east port can only meet R40-Curve's west port or Switch-A's curved-leg port"). The Circuit/Knot tilesets in mxgmn's repo demonstrate non-Wang adjacency, which is what 4DBrix track snapping really is.

### 5.2 BorisTheBrave/DeBroglie — WFC with **path/loop/acyclic constraints** (★★ very high transferable value)
- **URL:** https://github.com/BorisTheBrave/DeBroglie (docs: https://boristhebrave.github.io/DeBroglie/articles/path_constraints.html)
- **License:** MIT
- **Key feature: `LoopConstraint`** — "ensures there are at least two independent paths between relevant tiles" (i.e., enforces a cycle exists). Plus `ConnectedConstraint`, `AcyclicConstraint`, `ParityConstraint`, `MaxConsecutiveConstraint`, full backtracking.
- **Structural similarity:** The thesis closure constraint can be re-expressed as DeBroglie's `LoopConstraint` on a tileset whose tiles are LEGO track pieces with port-coded edges. **This is the most directly applicable open-source tool.** Port-snap = WFC adjacency rule, closure = LoopConstraint, inventory cap = `CountConstraint`.
- **Companion:** Boris's commercial Unity asset **Tessera Pro** (built on DeBroglie) does the same in 3D with paint-on-cube connectors — very close to "snap-port" semantics.

### 5.3 BorisTheBrave/chiseled-random-paths — connected-path generation
- **URL:** https://github.com/BorisTheBrave/chiseled-random-paths
- **License:** MIT
- **Description:** Generates random tile-based paths between user-specified points by an iterative "chisel" method; companion blog: https://www.boristhebrave.com/2022/03/20/chiseled-paths-revisited/. Lighter-weight than full WFC; useful when the layout is mostly path-like (typical of single-loop track plans).

### 5.4 marian42/wavefunctioncollapse — infinite procedural city
- **URL:** https://github.com/marian42/wavefunctioncollapse (write-up: https://marian42.de/article/wfc/, follow-up: https://marian42.de/article/infinite-wfc/)
- **Similarity:** 3D WFC with backtracking; demonstrates handling of slot-port matching ("each of the 6 sides of a block has a connector ID") on an infinite grid. The chunked deterministic re-implementation in the follow-up article shows how to make WFC parallelizable, relevant for accelerating BRKGA decoding.

### 5.5 oddmax/unity-wave-function-collapse-3d, AlexeyBond/godot-constraint-solving, mxgmn/MarkovJunior
- **URLs:** https://github.com/oddmax/unity-wave-function-collapse-3d ; https://github.com/AlexeyBond/godot-constraint-solving ; https://github.com/mxgmn/MarkovJunior
- **Notes:** Strong tile/symmetry frameworks. AlexeyBond's Godot 4 generic constraint solver explicitly notes "global constraints, including path constraints" are *not* yet supported — emphasizes that the path/loop constraint is the hard part DeBroglie alone solves cleanly.
- **OmarAflak/wave_function_collapse** (https://github.com/OmarAflak/wave_function_collapse) — minimal Python WFC with explicit "Track" tileset example (literally tracks!). Useful as a learning-grade reference.
- **AustinHellerRepo/WaveFunctionCollapse** — generic node/state CSP solver with multiple algorithms; can do Sudoku, WFC, or proximity-graph layouts.

### 5.6 RaoulHeese/qwfc — Quantum WFC (curiosity)
- **URL:** https://github.com/RaoulHeese/qwfc — quantum-circuit-based WFC. Not practical for the thesis but interesting if quantum-annealing of the inventory-constrained closure becomes a future direction.

---

## 6. BRKGA / Random-Key GA with Construction Decoders

The encoding pattern is highly transferable; these repos contain the exact framework idioms (decoder + parameter file + main loop) the thesis can copy.

### 6.1 ceandrade BRKGA-MP-IPR family (★★ canonical reference implementations)
- **C++:** https://github.com/ceandrade/brkga_mp_ipr_cpp (BSD-like)
- **Python:** https://github.com/ceandrade/brkga_mp_ipr_python (PyPI: `brkga-mp-ipr`)
- **Julia:** https://github.com/ceandrade/BrkgaMpIpr.jl
- **Authors:** Andrade, Toso, Gonçalves, Resende — the field's canonical authors. Multi-Parent BRKGA with Implicit Path Relinking. Active.
- **Why it matters:** This is the framework most closely matching what the thesis describes (random-key chromosome + user-supplied decoder); Resende/Andrade's papers also explicitly discuss feasibility-by-construction decoders.

### 6.2 ceandrade/brkga_muti_depot_multi_tsp — k-IMDMTSP (★ specifically called out by the thesis)
- **URL:** https://github.com/ceandrade/brkga_muti_depot_multi_tsp
- **Paper:** Andrade, Miyazawa, Resende, GECCO'13 + Networks 2016 ("Heuristics for a Hub Location-Routing Problem")
- **Why it matters:** Direct code for the *k-Interconnected Multi-Depot Multi-TSP*, which the thesis already flags as a structural analogue (multiple interconnected loops). Same authors' Hub Location-Routing paper provides the heuristics.

### 6.3 ceandrade/brkga_combinatorial_auctions — BRKGA for combinatorial auctions
- **URL:** https://github.com/ceandrade/brkga_combinatorial_auctions
- **Reference:** A second canonical example showing how to write a BRKGA decoder for a combinatorial problem with capacity-like constraints (the inventory cap is structurally similar).

### 6.4 dasvision0212/3D-Bin-Packing-Problem-with-BRKGA — 2D/3D bin-packing tutorial
- **URL:** https://github.com/dasvision0212/3D-Bin-Packing-Problem-with-BRKGA
- **Why it matters:** Pedagogical NumPy implementation of BRKGA + a *placement-based decoder*, mirroring the thesis's turtle decoder. Bin-packing's BRKGA decoder turns a real-vector chromosome into a sequence of placements — exactly the turtle-graphics step pattern.

### 6.5 antoniochaves19/BRKGA-QL — BRKGA with Q-learning
- **URL:** https://github.com/antoniochaves19/BRKGA-QL — adaptive parameter control with reinforcement learning. Recent (CEC 2021).

### 6.6 K1m0sab3/A-Biased-Random-Key-Genetic-Algorithm-with-Variable-Mutants — VRPODTW
- **URL:** https://github.com/K1m0sab3/A-Biased-Random-Key-Genetic-Algorithm-with-Variable-Mutants-to-solve-a-Vehicle-Routing-Problem
- **Notes:** Routing-problem decoder pattern; relevant because the "feasibility-by-construction" idea (insert customers into routes only if feasible) maps cleanly to "place track pieces only if angle/inventory feasible."

### 6.7 pymoo (anyoptimization/pymoo)
- **URL:** https://github.com/anyoptimization/pymoo
- **Why:** This is the framework the thesis already uses for NSGA-II. The doc page https://pymoo.org/algorithms/moo/nsga2.html explicitly shows custom Sampling/Crossover/Mutation classes for non-standard chromosomes (e.g., `BinaryRandomSampling`, OX/ERX permutation operators), which is the right hook to plug in a BRKGA random-key sampler with the thesis's construction decoder.

### 6.8 jambrito/BRKGA, bosilveira/brkga-python — minimal Python BRKGA frameworks
- **URLs:** https://rdrr.io/github/jambrito/BRKGA/ ; https://github.com/bosilveira/brkga-python — small, readable templates.

---

## 7. Multi-loop / Branched / Subway / OpenTTD-style Network Generation

### 7.1 mkonstapel/choochoo — OpenTTD train network AI (★ very high topology match)
- **URL:** https://github.com/mkonstapel/choochoo
- **Language:** Squirrel (NoAI API)
- **Description:** Builds *branched* railway networks in OpenTTD using a custom A*-style pathfinder, separate "main / branch / cargo / road / station / train" builders. Free.
- **Similarity:** Exact analogue of "main loop + branches + sidings" — the AI's separation of `builder_main`, `builder_branch`, `builder_network`, `builder_track` mirrors the thesis's switch/branch decomposition.

### 7.2 trAIns OpenTTD AI (Luis Rios)
- **Project page:** https://www.luisrios.eti.br/public/en_us/research/trains/
- **Paper:** SBGames 2009 (https://www.researchgate.net/publication/232640358_trAIns_An_Artificial_Inteligence_for_OpenTTD)
- **Description:** Modular "double parts" (bends, diagonals, lines) snapped at base/next points — *literally a port-based modular track decoder*. A* over the part graph for railway construction. Code available via OpenTTD's online-content system.
- **Transferable insight:** The "double part" abstraction with `base point` and `next point` is *exactly* the port-snap abstraction in the thesis. The papers explicitly discuss how to keep junctions geometrically valid.

### 7.3 juliuste/transit-map — MIP metro-map generation
- **URL:** https://github.com/juliuste/transit-map — Mixed-Integer Programming over a transit network graph to produce schematic maps. Direct OR-style alternative to BRKGA for the layout of multi-line networks.

### 7.4 mitchellbusby/subway-map-generation, nathan-hellinga/subway-map-generator
- **URLs:** https://github.com/mitchellbusby/subway-map-generation , https://github.com/nathan-hellinga/subway-map-generator (Processing.py) — procedural subway-map generators with multiple lines + interchanges (= switches).

### 7.5 mymetro-dot-io/mymetro — AI-driven metrolib
- **URL:** https://github.com/mymetro-dot-io/mymetro — Python lib for metro/subway with clustering + graph route-finding; clean separation of "main line / branch / express / loop / transfer" mirrors the thesis topology.

### 7.6 Trainyard puzzle solvers
- **micoloth/OpenTrainyard** (https://github.com/micoloth/OpenTrainyard) — Bevy/Rust clone of the original Trainyard puzzle game. Not a solver but the data model of "tile = pair-of-port directions" is the cleanest minimal model of LEGO track abstraction.
- **rlerrr/trainyard** (https://github.com/rlerrr/trainyard) — React clone with full level set.
- **James-P-D/Traintracks** (https://github.com/James-P-D/Traintracks) — Python solver for The Times newspaper "Traintrack" puzzles using DFS over piece-orientation. The DFS-with-backtracking-over-piece-orientation is a very simple version of the thesis decoder's failure-recovery logic.

---

## 8. Self-Avoiding Polygons / Hamiltonian Cycle Generators (the geometric primitive)

### 8.1 cbracher69/Chains-on-a-2D-honeycomb-lattice
- **URL:** https://github.com/cbracher69/Chains-on-a-2D-honeycomb-lattice — CPU+CUDA enumerator of self-avoiding chains and **closed self-avoiding polygons** on the honeycomb (and embedded) lattice up to 42 segments. Gold-standard reference for counting and sampling closed loops on a regular lattice (the LEGO straight+curve subset projects onto a similar lattice).

### 8.2 Hamiltonian path/cycle generators
- **clisby.net/projects/hamiltonian_path/** — pirate-quality generator using the "backbite move" Markov chain (Oberdorf, Ferguson, Jacobsen, Kondev, *Phys Rev E 2006*); the algorithm samples (almost) uniformly random Hamiltonian paths and circuits on n×n grids. JS code on the page.
- **maros-o/hamiltonian-cycle** (https://github.com/maros-o/hamiltonian-cycle) — Prim-MST-based random Hamiltonian cycle on a 2D grid.
- **CheranMahalingam/Snake_Hamiltonian_Cycle_Solver** (https://github.com/CheranMahalingam/Snake_Hamiltonian_Cycle_Solver) — same approach, Prim's MST + grid-doubling.
- **oysterCrusher/hampath** (https://github.com/oysterCrusher/hampath) — JS library for Hamiltonian *paths* on grids.
- **Pascal Sommer's blog & code** — https://medium.com/@pascal.sommer.ch/generating-hamiltonian-cycles-in-rectangular-grid-graphs-316c94ecefe0 (linked GitHub) — the MST-outline trick for grid Hamiltonian cycles.
- **arxiv 2412.12655** (https://arxiv.org/html/2412.12655v1) — fast construction of self-avoiding polygons + closed-walk enumerator on the square lattice, source code published.

These are the cleanest open-source primitives for "generate a closed loop on a discrete lattice" — the *floor* of the thesis problem before inventory and angle constraints are imposed.

---

## 9. LEGO 3D Brick-Layout GA Repos (related but different problem)

These solve "how to build a *voxel object* out of LEGO bricks," not track layout, but the GA-with-domain-specific-mutations recipe is identical in spirit:

- **romanglo/2D-LEGO-GA** (https://github.com/romanglo/2D-LEGO-GA) — 2D LEGO-brick layout via GA with rectangle-region crossover.
- **aaronwalsman/ltron** (https://github.com/aaronwalsman/ltron) — "Break and Make" LEGO assembly RL environment (ECCV 2022).
- Lee, Kim, Kim, Moon GECCO 2015 — Optimal LEGO brick layout via GA (no public repo, but the method is documented).
- Petrovic, *Solving LEGO brick layout problem using Evolutionary Algorithms* (2001), and the Hungarian / Stellenbosch follow-ups — algorithmic reference for thickening / boundary mutations that the thesis can borrow.

---

## 10. Marble-Run / Modular Track Hobby Code (peripheral but instructive)

- **wofr06/marblerun** (https://github.com/wofr06/marblerun) — registers and visualizes Ravensburger GraviTrax marble runs from a *very compact text notation*; SVG export. Perl. The text notation captures piece+orientation+connection cleanly — useful inspiration for a thesis-level layout export format.
- **Nurgak/Endless-Marble-Run** (https://github.com/Nurgak/Endless-Marble-Run) — p5.js endless modular CUBORO-style marble run; collision-checking + piece-traversal logic.
- **jhpieper/marble-run** (https://github.com/jhpieper/marble-run) — OpenSCAD parametric magnetic marble run; demonstrates how a designer formalizes a piece "track family" in code.

---

## 11. Graph Drawing with Port Constraints (PCB / yFiles / ELK)

The "ports on track pieces" abstraction is identical to ports in graph drawing:

- **Eclipse ELK** (https://github.com/eclipse/elk) — production-grade Java graph layout with explicit port constraints, layered routing, multi-edge handling. EPL.
- **OGDF** (https://github.com/ogdf/ogdf) — Open Graph Drawing Framework; rich port/orthogonal routing. GPL.
- **PCB autorouters:** **FreeRouting** (https://github.com/freerouting/freerouting), **KiCad** (https://gitlab.com/kicad/code/kicad). Multi-net routing on a discrete grid with port pins is the EDA cousin of routing track segments between fixed switch ports. Open source.

These are not generators, but their *port-constrained edge-routing algorithms* are directly portable to the thesis's compound-S-curve switch-branch geometry handling.

---

## Summary Table — Top Picks Ranked by Direct Transfer Value

| Rank | Repo | Why | Algorithm |
|------|------|-----|-----------|
| 1 | **BorisTheBrave/DeBroglie** + **chiseled-random-paths** | Path/Loop/Acyclic constraints out-of-the-box; port-edge tile model | WFC + custom constraints |
| 2 | **ceandrade/brkga_mp_ipr_python** + **brkga_muti_depot_multi_tsp** | Canonical BRKGA framework; multi-depot multi-TSP is the explicit thesis-flagged analogue | BRKGA-MP-IPR |
| 3 | **kevinburke/rct** | Closest-in-spirit GA on closed-loop modular pieces from a fixed catalog | GA |
| 4 | **mmm444/trains** | Only existing automatic LEGO/Duplo track-loop generator | Brute force + pruning |
| 5 | **vazgriz dungeon (multi-floor) + AmanSachan1/InterestingLevelGenerator** | Best multi-floor connectivity analogue (multiple loops via switches ≈ multiple floors via stairs) | 3D Delaunay + A* (vazgriz) ; Voronoi multi-layer (AmanSachan1) |
| 6 | **mkonstapel/choochoo + trAIns AI for OpenTTD** | Modular branched railway construction; explicit "main / branch / network" decomposition + double-part port snapping | A*-style with module catalog |
| 7 | **LasseHenrich/racetrack-generation** | Closed arcade tracks with explicit crossings/intersections | Repulsive curves + constrained space |
| 8 | **Lswbanban/BlueBrick + l-gauge libraries** | Authoritative inventory geometry for 4DBrix-compatible pieces | Manual CAD (data source) |
| 9 | **caspervanengelenburg/msd + ennauata/housegan** | Closest *academic* multi-floor (MSD) and graph-constrained layout work with code | GAN, GNN, Transformer |
| 10 | **clisby.net Hamiltonian path generator + cbracher69/Chains-on-a-2D-honeycomb-lattice** | Cleanest "sample a closed loop on a regular lattice" primitives | Backbite Markov chain; brute enumeration |

---

## Caveats and Source-Quality Notes

- **No automatic LEGO/4DBrix track-layout generator was found** beyond mmm444/trains' single-loop Duplo/IKEA case. Multiple independent hobbyist sources (Monty's Trains 2018; Eurobricks 2025; Bricks McGee) explicitly state that no such tool exists. This is corroborating evidence, not proof — there could still be unindexed personal scripts on Brickshelf, Flickr, or Discord servers.
- **4DBrix's nControl downloads page returned a fatal PHP error** at the time of research (May 2026); the company appears largely inactive.
- BRKGA / pymoo references should be cross-checked against the latest pymoo 0.6.x docs; the API for custom samplers/operators changed between 0.5 and 0.6.
- DeBroglie's `LoopConstraint` enforces "at least two independent paths between relevant tiles," which is *not exactly* the same as a single closed Hamiltonian-style loop; it would need a custom constraint or a relevance-tile choice to enforce the "main loop is one connected cycle" invariant of the thesis.
- The k-IMDMTSP and Hub-Location-Routing repos by Andrade-Resende are **C++** and somewhat dated; reusing their decoders requires a port. Their *encoding ideas* (decoder maps real-key vector → feasible multi-tour structure) are what transfer, not the code verbatim.
- Some itch.io tools (watabou, marian42 city) are closed-source binaries; only the linked GitHub source for marian42 is reusable.

The combination most likely to accelerate thesis implementation is: **BRKGA-MP-IPR Python framework as the optimization backbone + a turtle-graphics decoder reading BlueBrick/l-gauge inventory XML + DeBroglie's LoopConstraint logic re-implemented in Python for closure enforcement during decoding + pymoo's NSGA-II as the multi-objective wrapper**.