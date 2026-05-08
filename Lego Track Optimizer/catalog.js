/* catalog.js — Piece catalog as data.
 *
 * Each piece has:
 *   id          unique string
 *   label       display name
 *   ports       array of local poses (port A is identity by convention)
 *   routes      array of {from, to, length_mm, radius_mm?, curveDir?}
 *   footprint   array of local corner points for AABB / drawing
 *   color       render hint
 *
 * Geometry constants come from the brief:
 *   - 1 stud = 8 mm
 *   - R40 curve: 22.5° turn, radius = 40 studs = 320 mm
 *   - Track tile width = 8 studs = 64 mm (so 4 studs each side of centerline)
 *   - 16-stud straight = 128 mm; 32-stud straight = 256 mm
 */

(function (global) {
  const STUD = 8;                 // mm per stud
  const TILE_W = 8 * STUD;        // 64 mm — track tile width
  const HALF_W = TILE_W / 2;
  const R40 = 40 * STUD;          // 320 mm — curve radius
  const CURVE_ANGLE = Math.PI / 8; // 22.5° in radians

  // Build a STRAIGHT piece of length L (mm), 2 ports, 1 route both directions.
  function makeStraight(id, label, lengthMm, color) {
    return {
      id, label, color,
      kind: "straight",
      ports: [
        { x: 0, y: 0, t: 0 },                       // A: at origin facing +x
        { x: lengthMm, y: 0, t: 0 },                // B: forward
      ],
      routes: [
        { from: 0, to: 1, length: lengthMm },
        { from: 1, to: 0, length: lengthMm },
      ],
      footprint: [
        { x: 0,        y: -HALF_W },
        { x: lengthMm, y: -HALF_W },
        { x: lengthMm, y:  HALF_W },
        { x: 0,        y:  HALF_W },
      ],
      length: lengthMm,
    };
  }

  // Build a CURVE piece: R40, 22.5°. dir = +1 (left, CCW) or -1 (right, CW).
  function makeCurve(id, label, dir, color) {
    // Centre of arc is at (0, dir * R40). Port A is at (0,0) facing +x.
    // Port B is the arc's other end: the train rotates by dir * CURVE_ANGLE
    // around the centre.
    const cx = 0, cy = dir * R40;
    const a0 = -dir * Math.PI / 2;          // angle from centre to port A
    const a1 = a0 + dir * CURVE_ANGLE;      // angle from centre to port B
    const bx = cx + R40 * Math.cos(a1);
    const by = cy + R40 * Math.sin(a1);
    const bt = dir * CURVE_ANGLE;
    const arcLen = R40 * CURVE_ANGLE;       // mm

    // Footprint: 4 corners of the curved tile (inner & outer arc endpoints).
    // We sample inner & outer arcs at start and end for an approximate quad.
    const rIn = R40 - HALF_W, rOut = R40 + HALF_W;
    const corners = [
      { x: cx + rIn  * Math.cos(a0), y: cy + rIn  * Math.sin(a0) },
      { x: cx + rOut * Math.cos(a0), y: cy + rOut * Math.sin(a0) },
      { x: cx + rOut * Math.cos(a1), y: cy + rOut * Math.sin(a1) },
      { x: cx + rIn  * Math.cos(a1), y: cy + rIn  * Math.sin(a1) },
    ];

    return {
      id, label, color,
      kind: "curve",
      dir,                                  // +1 = left, -1 = right
      ports: [
        { x: 0,  y: 0,  t: 0 },
        { x: bx, y: by, t: bt },
      ],
      routes: [
        { from: 0, to: 1, length: arcLen, radius: R40, curveDir: dir, headingStep: dir },
        { from: 1, to: 0, length: arcLen, radius: R40, curveDir: -dir, headingStep: -dir },
      ],
      footprint: corners,
      length: arcLen,
      headingStep: dir,                     // +1 / -1 in units of CURVE_ANGLE
    };
  }

  const DEFAULT_CATALOG = [
    makeStraight("S16", "Straight 16",  128, "#4a90e2"),
    makeStraight("S32", "Straight 32",  256, "#3a7fd2"),
    makeCurve("CL",    "Curve Left",   +1,  "#e2944a"),
    makeCurve("CR",    "Curve Right",  -1,  "#d97a35"),
  ];

  const DEFAULT_PROBLEM = {
    inventory: { S16: 4, S32: 4, CL: 16, CR: 16 },
    boundary: { width: 3000, height: 2000 },   // mm — 3m × 2m living-room rug
    tolerance: { mm: 2.0, deg: 0.5 },
    weights: { utilization: 1.0, lapTime: 0.3, compactness: 0.2 },
    physics: { mu: 0.6, g: 9810, motorVMax: 1200 }, // mm/s² and mm/s
    searchBudget: { ms: 5000, restarts: 60, minLoop: 4 },
  };

  global.Catalog = {
    STUD, TILE_W, HALF_W, R40, CURVE_ANGLE,
    DEFAULT_CATALOG, DEFAULT_PROBLEM,
    makeStraight, makeCurve,
  };
})(window);
