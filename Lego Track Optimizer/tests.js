/* tests.js — Geometry tests, surfaced in the Self-check panel.
 *
 * Each test returns { name, ok, detail }. Run on every load.
 */

(function (global) {
  const G = global.Geom;
  const C = global.Catalog;
  const S = global.Search;
  const TAU = G.TAU;
  const TOL = 1e-6;

  function approx(a, b, eps = TOL) { return Math.abs(a - b) < eps; }

  function test(name, fn) {
    try {
      const detail = fn() || "ok";
      return { name, ok: true, detail };
    } catch (e) {
      return { name, ok: false, detail: e.message || String(e) };
    }
  }

  function assert(cond, msg) {
    if (!cond) throw new Error(msg || "assertion failed");
  }

  function runAll() {
    const results = [];

    results.push(test("pose: identity is neutral", () => {
      const a = G.mkPose(123, -45, 0.7);
      assert(G.poseEqual(G.compose(a, G.IDENTITY), a), "a∘I ≠ a");
      assert(G.poseEqual(G.compose(G.IDENTITY, a), a), "I∘a ≠ a");
    }));

    results.push(test("pose: inverse cancels", () => {
      const a = G.mkPose(50, 30, 0.4);
      const i = G.inverse(a);
      assert(G.poseEqual(G.compose(a, i), G.IDENTITY, 1e-9), "a∘inv(a) ≠ I");
      assert(G.poseEqual(G.compose(i, a), G.IDENTITY, 1e-9), "inv(a)∘a ≠ I");
    }));

    results.push(test("pose: composition is associative", () => {
      const a = G.mkPose(10, 20, 0.3);
      const b = G.mkPose(-5, 40, -0.2);
      const c = G.mkPose(7, -3, 1.1);
      const lhs = G.compose(G.compose(a, b), c);
      const rhs = G.compose(a, G.compose(b, c));
      assert(G.poseEqual(lhs, rhs, 1e-9), "(a∘b)∘c ≠ a∘(b∘c)");
    }));

    results.push(test("circle: 16 right-curves close exactly", () => {
      const cr = C.DEFAULT_CATALOG.find(p => p.id === "CR");
      let pose = G.IDENTITY;
      for (let i = 0; i < 16; i++) {
        const placed = S.placePiece(cr, cr.routes[0], pose);
        pose = placed.endPose;
      }
      const d = G.poseDistance(G.IDENTITY, pose);
      const dt = G.headingDelta(G.IDENTITY, pose);
      assert(d < 1e-6, `position drift = ${d.toFixed(6)} mm`);
      assert(dt < 1e-9, `heading drift = ${dt.toFixed(9)} rad`);
      return `position drift ${d.toExponential(2)} mm`;
    }));

    results.push(test("circle: 16 left-curves close exactly", () => {
      const cl = C.DEFAULT_CATALOG.find(p => p.id === "CL");
      let pose = G.IDENTITY;
      for (let i = 0; i < 16; i++) {
        pose = S.placePiece(cl, cl.routes[0], pose).endPose;
      }
      assert(G.poseDistance(G.IDENTITY, pose) < 1e-6);
      assert(G.headingDelta(G.IDENTITY, pose) < 1e-9);
    }));

    results.push(test("heading: 8L + 8R + straights ≢ 0 (negative)", () => {
      // 8L + 8R cancel in heading, but the path never returns to origin.
      // Closure must check both heading AND position.
      const cl = C.DEFAULT_CATALOG.find(p => p.id === "CL");
      const cr = C.DEFAULT_CATALOG.find(p => p.id === "CR");
      let pose = G.IDENTITY;
      for (let i = 0; i < 8; i++) pose = S.placePiece(cl, cl.routes[0], pose).endPose;
      for (let i = 0; i < 8; i++) pose = S.placePiece(cr, cr.routes[0], pose).endPose;
      assert(G.headingDelta(G.IDENTITY, pose) < 1e-9, "heading should cancel");
      assert(G.poseDistance(G.IDENTITY, pose) > 100, "position should NOT be back");
    }));

    results.push(test("AABB: intersect symmetric & reflexive", () => {
      const a = { minX: 0, minY: 0, maxX: 10, maxY: 10 };
      const b = { minX: 5, minY: 5, maxX: 15, maxY: 15 };
      const c = { minX: 20, minY: 20, maxX: 30, maxY: 30 };
      assert(G.aabbIntersect(a, a), "self-intersect");
      assert(G.aabbIntersect(a, b) === G.aabbIntersect(b, a), "symmetric");
      assert(G.aabbIntersect(a, b), "a∩b expected");
      assert(!G.aabbIntersect(a, c), "a∩c expected disjoint");
    }));

    results.push(test("AABB: boundary detects out-of-bounds", () => {
      const inner = { minX: 0, minY: 0, maxX: 10, maxY: 10 };
      const outer = { minX: -1, minY: -1, maxX: 11, maxY: 11 };
      const tooBig = { minX: -2, minY: 0, maxX: 10, maxY: 10 };
      assert(G.aabbInside(inner, outer), "inside should pass");
      assert(!G.aabbInside(tooBig, outer), "out-of-bounds should fail");
    }));

    results.push(test("oval closes: 8 right curves + 2 straights × 2", () => {
      // Half-loop right, straight, half-loop right, straight.
      // 8 right curves = 180°. Add a straight, then another 8 right + straight.
      // Both straights must be the same length (and headed opposite).
      const cr = C.DEFAULT_CATALOG.find(p => p.id === "CR");
      const s32 = C.DEFAULT_CATALOG.find(p => p.id === "S32");
      let pose = G.IDENTITY;
      for (let i = 0; i < 8; i++) pose = S.placePiece(cr, cr.routes[0], pose).endPose;
      pose = S.placePiece(s32, s32.routes[0], pose).endPose;
      for (let i = 0; i < 8; i++) pose = S.placePiece(cr, cr.routes[0], pose).endPose;
      pose = S.placePiece(s32, s32.routes[0], pose).endPose;
      const d = G.poseDistance(G.IDENTITY, pose);
      const dt = G.headingDelta(G.IDENTITY, pose);
      assert(d < 1e-6, `oval drift = ${d}`);
      assert(dt < 1e-9, `oval heading drift = ${dt}`);
      return `closed exactly`;
    }));

    results.push(test("route delta = inv(from) ∘ to", () => {
      const cl = C.DEFAULT_CATALOG.find(p => p.id === "CL");
      const r = cl.routes[0];
      const expected = G.compose(G.inverse(cl.ports[r.from]), cl.ports[r.to]);
      // Apply at world origin.
      const placed = S.placePiece(cl, r, G.IDENTITY);
      assert(G.poseEqual(placed.endPose, expected, 1e-9), "delta mismatch");
    }));

    results.push(test("inventory respected during search", () => {
      // With only 4 right curves, no closed loop should exist (need 16 for a circle,
      // or specific oval combinations). Verify search finishes without crashing
      // and either returns null or a different shape using only inventory.
      const tinyProblem = {
        ...C.DEFAULT_PROBLEM,
        inventory: { CR: 4, S32: 0, S16: 0, CL: 0 },
        searchBudget: { ms: 200, restarts: 4, minLoop: 4 },
      };
      const r = S.runOnce(C.DEFAULT_CATALOG, tinyProblem, 1, 200);
      // A 4-curve loop is impossible (90° total). Best should be null.
      assert(r.best === null, "expected no closed loop with 4 right curves");
    }));

    return results;
  }

  global.Tests = { runAll };
})(window);
