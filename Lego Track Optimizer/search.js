/* search.js — Template-based loop generation with validation.
 *
 * Approach: instead of blind DFS, generate candidate closed loops from
 * combinatorial templates (n-curve circles, ovals with straight segments
 * between curve banks, figure-8-precursors, etc.), validate each against
 * the inventory + boundary + collision constraints, and score the survivors.
 *
 * This is HONEST about the v1 cut: we explore a parameterized space of
 * canonical loop shapes rather than the full multigraph traversal. The
 * geometry, closure, and physics models are unchanged; only the search
 * strategy is templatized. README.md documents this trade-off.
 */

(function (global) {
  const G = global.Geom;
  const C = global.Catalog;

  function routeDelta(piece, route) {
    return G.compose(G.inverse(piece.ports[route.from]), piece.ports[route.to]);
  }

  function placePiece(piece, route, entryPose) {
    const originPose = G.compose(entryPose, G.inverse(piece.ports[route.from]));
    const endPose = G.compose(entryPose, routeDelta(piece, route));
    const aabb = G.footprintAabb(piece, originPose);
    return { originPose, endPose, aabb };
  }

  function makeRng(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function lapTime(catalogById, placements, physics) {
    let t = 0, len = 0;
    for (const p of placements) {
      const piece = catalogById[p.pieceId];
      const route = piece.routes[p.routeIdx];
      let v;
      if (route.radius) {
        v = Math.min(physics.motorVMax, Math.sqrt(physics.mu * physics.g * route.radius));
      } else {
        v = physics.motorVMax;
      }
      t += route.length / v;
      len += route.length;
    }
    return { time: t, length: len };
  }

  function score(layout, problem, catalogById, totalAvailable) {
    const used = layout.placements.length;
    const utilization = used / totalAvailable;
    const { time: lap, length } = lapTime(catalogById, layout.placements, problem.physics);
    const lapSpeed = lap > 0 ? (length / lap) / problem.physics.motorVMax : 0;
    const bArea = problem.boundary.width * problem.boundary.height;
    const compact = bArea > 0 ? Math.min(1, G.aabbArea(layout.bbox) / bArea) : 0;
    const w = problem.weights;
    const total = w.utilization * utilization + w.lapTime * lapSpeed + w.compactness * compact;
    return { utilization, lapSpeed, compact, lapTime: lap, length, total };
  }

  /* Build a sequence of pieces from an "instruction list" of piece-ids and
   * verify it's a closed, in-bounds, non-self-intersecting layout. */
  function buildLayout(catalog, instructions, problem) {
    const catalogById = Object.fromEntries(catalog.map(p => [p.id, p]));
    const tolMm = problem.tolerance.mm;
    const tolRad = problem.tolerance.deg * Math.PI / 180;
    const boundary = {
      minX: -problem.boundary.width / 2, maxX: problem.boundary.width / 2,
      minY: -problem.boundary.height / 2, maxY: problem.boundary.height / 2,
    };
    const start = G.IDENTITY;
    let pose = start;
    const placements = [];
    const aabbs = [];
    let bbox = null;
    for (const id of instructions) {
      const piece = catalogById[id];
      if (!piece) return { ok: false, reason: `unknown piece ${id}` };
      const route = piece.routes[0]; // forward only in templates
      const placed = placePiece(piece, route, pose);
      if (!G.aabbInside(placed.aabb, boundary)) return { ok: false, reason: "out of bounds" };
      for (let i = 0; i < aabbs.length - 1; i++) {
        if (G.aabbIntersect(aabbs[i], placed.aabb, 1)) return { ok: false, reason: "collision" };
      }
      placements.push({ pieceId: id, routeIdx: 0, originPose: placed.originPose, endPose: placed.endPose });
      aabbs.push(placed.aabb);
      bbox = bbox ? G.aabbUnion(bbox, placed.aabb) : placed.aabb;
      pose = placed.endPose;
    }
    if (!G.isClosed(start, pose, tolMm, tolRad)) return { ok: false, reason: "not closed" };
    return { ok: true, layout: { placements, bbox, start, end: pose } };
  }

  /* Inventory check: does `instructions` fit within `inventory`? */
  function fitsInventory(instructions, inventory) {
    const need = {};
    for (const id of instructions) need[id] = (need[id] || 0) + 1;
    for (const [id, n] of Object.entries(need)) {
      if ((inventory[id] || 0) < n) return false;
    }
    return true;
  }

  /* TEMPLATES.
   *
   * Each template is a generator that yields an `instructions` list given an
   * inventory. They produce buildable shapes:
   *
   *   circle(n, dir):    n same-handed curves, n ∈ {16}
   *   oval(curveDir, banks=8, straightId, count):
   *                       banks curves + count straights, banks curves + count straights
   *   double-oval (longer straights)
   *   stadium with mixed straights
   */

  function* circleTemplates(catalog, inventory) {
    for (const piece of catalog) {
      if (piece.kind !== "curve") continue;
      const id = piece.id;
      if ((inventory[id] || 0) >= 16) {
        yield { name: `circle-${id}`, instructions: Array(16).fill(id) };
      }
    }
  }

  function* ovalTemplates(catalog, inventory) {
    const curves = catalog.filter(p => p.kind === "curve");
    const straights = catalog.filter(p => p.kind === "straight");
    // banks: 8 same-handed curves on each end → 180° each, 360° total
    for (const cv of curves) {
      if ((inventory[cv.id] || 0) < 16) continue;
      // straights on each side; sides must be same length to keep it parallel
      // Use combinations of S16 and S32 on each side, equal counts on both sides.
      for (const st of straights) {
        const stockS = inventory[st.id] || 0;
        const maxPerSide = Math.floor(stockS / 2);
        for (let k = 1; k <= maxPerSide; k++) {
          // 8 curves + k straights + 8 curves + k straights
          const instr = [
            ...Array(8).fill(cv.id),
            ...Array(k).fill(st.id),
            ...Array(8).fill(cv.id),
            ...Array(k).fill(st.id),
          ];
          yield { name: `oval-${cv.id}-${st.id}x${k}`, instructions: instr };
        }
      }
      // mixed straights: k S32 + j S16 each side
      const s32id = "S32", s16id = "S16";
      const s32n = inventory[s32id] || 0, s16n = inventory[s16id] || 0;
      for (let a = 0; a <= Math.floor(s32n / 2); a++) {
        for (let b = 0; b <= Math.floor(s16n / 2); b++) {
          if (a + b === 0) continue;
          const sideStraights = [...Array(a).fill(s32id), ...Array(b).fill(s16id)];
          if (sideStraights.length === 0) continue;
          const instr = [
            ...Array(8).fill(cv.id), ...sideStraights,
            ...Array(8).fill(cv.id), ...sideStraights,
          ];
          yield { name: `oval-mixed-${cv.id}-${a}S32-${b}S16`, instructions: instr };
        }
      }
    }
  }

  // Stadium with asymmetric straights: a, b on opposite sides isn't closed unless
  // a == b in length. We enforce equal mm-length per side.
  function* stadiumTemplates(catalog, inventory) {
    const curves = catalog.filter(p => p.kind === "curve");
    for (const cv of curves) {
      if ((inventory[cv.id] || 0) < 16) continue;
      const s32n = inventory["S32"] || 0;
      const s16n = inventory["S16"] || 0;
      // per side: a S32 + b S16 ≡ a' S32 + b' S16 in length (mm)
      // S32 = 256, S16 = 128. So 256a + 128b == 256a' + 128b' → 2a + b == 2a' + b'.
      const sets = [];
      for (let a = 0; a <= Math.min(s32n, 4); a++)
        for (let b = 0; b <= Math.min(s16n, 4); b++)
          if (a + b > 0) sets.push({ a, b, len: 2 * a + b });
      for (let i = 0; i < sets.length; i++) {
        for (let j = i; j < sets.length; j++) {
          const A = sets[i], B = sets[j];
          if (A.len !== B.len) continue;
          if (A.a + B.a > s32n || A.b + B.b > s16n) continue;
          const sideA = [...Array(A.a).fill("S32"), ...Array(A.b).fill("S16")];
          const sideB = [...Array(B.a).fill("S32"), ...Array(B.b).fill("S16")];
          if (sideA.length === 0 && sideB.length === 0) continue;
          const instr = [
            ...Array(8).fill(cv.id), ...sideA,
            ...Array(8).fill(cv.id), ...sideB,
          ];
          yield { name: `stadium-${cv.id}-A${A.a},${A.b}-B${B.a},${B.b}`, instructions: instr };
        }
      }
    }
  }

  // Generate all template candidates, dedupe by instruction string, and shuffle.
  function* allTemplates(catalog, inventory) {
    yield* circleTemplates(catalog, inventory);
    yield* ovalTemplates(catalog, inventory);
    yield* stadiumTemplates(catalog, inventory);
  }

  function search(catalog, problem, opts = {}) {
    const onProgress = opts.onProgress;
    const onRestart = opts.onRestart;
    const totalMs = problem.searchBudget.ms ?? 3000;
    const catalogById = Object.fromEntries(catalog.map(p => [p.id, p]));
    const totalAvailable = Object.values(problem.inventory).reduce((a, b) => a + b, 0);

    const t0 = performance.now();
    const candidates = [];
    for (const tmpl of allTemplates(catalog, problem.inventory)) {
      candidates.push(tmpl);
    }
    // Dedupe.
    const seen = new Set();
    const unique = [];
    for (const c of candidates) {
      const k = c.instructions.join(",");
      if (!seen.has(k)) { seen.add(k); unique.push(c); }
    }

    // Shuffle for visual variety across runs.
    const rng = makeRng(opts.seed ?? Math.floor(Math.random() * 1e9));
    for (let i = unique.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [unique[i], unique[j]] = [unique[j], unique[i]];
    }

    let best = null;
    let evaluated = 0;
    let i = 0;

    function step() {
      const batchEnd = Math.min(i + 8, unique.length);
      while (i < batchEnd) {
        const tmpl = unique[i++];
        evaluated++;
        if (!fitsInventory(tmpl.instructions, problem.inventory)) continue;
        const r = buildLayout(catalog, tmpl.instructions, problem);
        if (!r.ok) continue;
        const layout = r.layout;
        layout.score = score(layout, problem, catalogById, totalAvailable);
        layout.template = tmpl.name;
        if (!best || layout.score.total > best.score.total) {
          best = layout;
          if (onProgress) onProgress({ best, nodes: evaluated, attempt: i });
        }
      }
      if (onRestart) onRestart({ attempt: i, totalNodes: evaluated, best });
      if (i < unique.length && performance.now() - t0 < totalMs) {
        setTimeout(step, 0);
      } else {
        if (opts.onDone) opts.onDone({ best, nodes: evaluated, elapsed: performance.now() - t0, restarts: i });
      }
    }
    if (unique.length === 0) {
      if (opts.onDone) opts.onDone({ best: null, nodes: 0, elapsed: 0, restarts: 0 });
      return;
    }
    step();
  }

  // Single-attempt for tests: evaluate templates synchronously.
  function runOnce(catalog, problem, seed, deadlineMs) {
    const catalogById = Object.fromEntries(catalog.map(p => [p.id, p]));
    const totalAvailable = Object.values(problem.inventory).reduce((a, b) => a + b, 0);
    let best = null;
    let nodes = 0;
    const t0 = performance.now();
    for (const tmpl of allTemplates(catalog, problem.inventory)) {
      nodes++;
      if (performance.now() - t0 > deadlineMs) break;
      if (!fitsInventory(tmpl.instructions, problem.inventory)) continue;
      const r = buildLayout(catalog, tmpl.instructions, problem);
      if (!r.ok) continue;
      const layout = r.layout;
      layout.score = score(layout, problem, catalogById, totalAvailable);
      if (!best || layout.score.total > best.score.total) best = layout;
    }
    return { best, nodes, elapsed: performance.now() - t0 };
  }

  global.Search = { search, runOnce, routeDelta, placePiece, score, lapTime, buildLayout };
})(window);
