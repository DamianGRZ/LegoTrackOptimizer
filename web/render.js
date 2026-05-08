/* render.js — Top-down blueprint canvas renderer for V2 port-pair layouts.
 *
 * Inputs (driven by the Python backend):
 *   - layout: { stud_mm, boundary_mm:{...}, pieces:[{slot, piece_id, pose_studs:{x,y,theta}}], edges:[...] }
 *   - catalog: { stud_mm, pieces:[{id, kind, ports, routes, length_studs, radius_studs, sector_angle_rad, body_length_studs, diverging_radius_studs, hand}] }
 *
 * Drawing strategy: per-route, per-kind.
 *   straight:   one chord route (rect tile + rails + sleepers)
 *   curve:      one arc route (annular wedge + arc rails + sleepers)
 *   switch:     "through" route as chord, "diverging" route as arc
 *               (radius from spec.diverging_radius_studs, hand from spec.hand)
 *   crossing:   every route as a chord (straight tile between port positions);
 *               two routes overlap to form the X / +.
 *
 * Piece-local coordinates (studs) are converted to mm by stud_mm; the canvas
 * viewport then fits the boundary box into the canvas with a margin.
 */

(function (global) {
  const G = global.Geom;

  // Track tile half-width: 8 studs total (4 each side of centerline) — LEGO 9V.
  const HALF_W_STUDS = 4.0;

  const EPS_THETA = 1e-3;  // dtheta below this counts as straight

  const DEFAULT_THEME = {
    bg: "#0e1620",
    grid: "rgba(120, 160, 200, 0.07)",
    boundary: "rgba(180, 200, 230, 0.45)",
    straightFill: "#1f3852",
    straightDim: "#1a2738",
    curveFill: "#2a4567",
    curveDim: "#1f3149",
    pieceStroke: "rgba(180, 210, 240, 0.35)",
    sleeper: "rgba(220, 230, 245, 0.45)",
    rail: "rgba(220, 235, 255, 0.85)",
    centerline: "rgba(140, 200, 255, 0.55)",
    start: "#ffb47a",
    text: "#cfd9e6",
  };

  // Per-branch fill palette. Cycle 0 (typically the main loop) uses the
  // default theme color. Higher cycle ids (siding branches, figure-8 second
  // lobe, double-crossover second track) get successively distinct hues.
  const BRANCH_FILLS = [
    null,        // branch 0 → leave theme defaults
    "#7a3a1f",   // 1: warm orange
    "#1f5230",   // 2: forest green
    "#5b1f48",   // 3: plum
    "#3a3a1f",   // 4: olive
    "#1f4870",   // 5: deep teal
  ];

  function pickBranchTheme(theme, piece) {
    const labels = piece && piece.branch_labels;
    if (!labels) return theme;
    let bid = null;
    for (const v of Object.values(labels)) {
      if (v != null && v !== 0) { bid = v; break; }
    }
    if (bid == null) return theme;
    const fill = BRANCH_FILLS[bid % BRANCH_FILLS.length];
    if (fill == null) return theme;
    return Object.assign({}, theme, {
      straightFill: fill,
      curveFill: fill,
    });
  }

  // World→canvas transform: fit boundary into canvas with margin, preserve aspect.
  // boundaryMm: {min_x, max_x, min_y, max_y, width, height}
  function makeViewport(canvas, boundaryMm, marginPx) {
    const wPx = canvas.width;
    const hPx = canvas.height;
    const m = marginPx == null ? 28 : marginPx;
    const sx = (wPx - 2 * m) / Math.max(1, boundaryMm.width);
    const sy = (hPx - 2 * m) / Math.max(1, boundaryMm.height);
    const s = Math.min(sx, sy);
    const cx = (boundaryMm.min_x + boundaryMm.max_x) / 2;
    const cy = (boundaryMm.min_y + boundaryMm.max_y) / 2;
    const ox = wPx / 2 - cx * s;
    const oy = hPx / 2 + cy * s;   // +cy*s because we flip Y below
    return {
      scale: s,
      toX: (x) => ox + x * s,
      toY: (y) => oy - y * s,    // flip so +y is up
      toLen: (l) => l * s,
      cx, cy,
    };
  }

  function clear(ctx, w, h, bg) {
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
  }

  // 10-stud grid in mm.
  function drawGrid(ctx, vp, boundaryMm, stud_mm, theme) {
    const step = 10 * stud_mm;
    ctx.save();
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 0.5;

    const x0 = Math.floor(boundaryMm.min_x / step) * step;
    const y0 = Math.floor(boundaryMm.min_y / step) * step;
    for (let x = x0; x <= boundaryMm.max_x + 0.001; x += step) {
      ctx.beginPath();
      ctx.moveTo(vp.toX(x), vp.toY(boundaryMm.min_y));
      ctx.lineTo(vp.toX(x), vp.toY(boundaryMm.max_y));
      ctx.stroke();
    }
    for (let y = y0; y <= boundaryMm.max_y + 0.001; y += step) {
      ctx.beginPath();
      ctx.moveTo(vp.toX(boundaryMm.min_x), vp.toY(y));
      ctx.lineTo(vp.toX(boundaryMm.max_x), vp.toY(y));
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawBoundary(ctx, vp, boundaryMm, theme) {
    ctx.save();
    ctx.strokeStyle = theme.boundary;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(
      vp.toX(boundaryMm.min_x), vp.toY(boundaryMm.max_y),
      vp.toLen(boundaryMm.width), vp.toLen(boundaryMm.height),
    );
    ctx.restore();
  }

  // ----- per-kind drawing -----
  // Each function takes the canvas context already transformed into the
  // piece-local frame at port A (so port A is at (0,0), heading +x). All
  // dimensions are in studs at this point — the outer code applies stud_mm
  // and viewport scaling via ctx.scale(stud_mm * scale, -stud_mm * scale).

  function drawChordTile(ctx, entry, exit, drawScale, theme, dim) {
    // Rectangle along chord between two ports (studs, piece-local).
    // entry, exit: {dx, dy, dtheta} -- only positions used.
    const dx = exit.dx - entry.dx;
    const dy = exit.dy - entry.dy;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1e-6) return;
    const ang = Math.atan2(dy, dx);
    const HW = HALF_W_STUDS;

    ctx.save();
    ctx.translate(entry.dx, entry.dy);
    ctx.rotate(ang);

    ctx.fillStyle = dim ? theme.straightDim : theme.straightFill;
    ctx.strokeStyle = theme.pieceStroke;
    ctx.lineWidth = 0.6 / drawScale;
    ctx.fillRect(0, -HW, len, 2 * HW);
    ctx.strokeRect(0, -HW, len, 2 * HW);

    // sleepers
    ctx.strokeStyle = theme.sleeper;
    ctx.lineWidth = 0.4 / drawScale;
    const nSleep = Math.max(2, Math.round(len / 4));   // every ~4 studs
    for (let i = 1; i < nSleep; i++) {
      const x = (len * i) / nSleep;
      ctx.beginPath();
      ctx.moveTo(x, -HW + 0.5);
      ctx.lineTo(x,  HW - 0.5);
      ctx.stroke();
    }
    // rails (1 stud inset from edge)
    ctx.strokeStyle = theme.rail;
    ctx.lineWidth = 0.6 / drawScale;
    ctx.beginPath();
    ctx.moveTo(0, -HW + 1.5); ctx.lineTo(len, -HW + 1.5);
    ctx.moveTo(0,  HW - 1.5); ctx.lineTo(len,  HW - 1.5);
    ctx.stroke();

    ctx.restore();
  }

  // Compute the arc center given chord endpoints + signed radius + side.
  // Returns null if radius < chord/2 (geometrically impossible).
  // side: +1 for CCW (left), -1 for CW (right).
  function arcCenter(entry, exit, radius, side) {
    const cx = (entry.dx + exit.dx) / 2;
    const cy = (entry.dy + exit.dy) / 2;
    const dx = exit.dx - entry.dx;
    const dy = exit.dy - entry.dy;
    const chord = Math.sqrt(dx * dx + dy * dy);
    if (chord < 1e-6) return null;
    const half = chord / 2;
    if (Math.abs(radius) <= half) return null;
    const d = Math.sqrt(radius * radius - half * half);
    // perpendicular-left unit vector to chord direction
    const px = -dy / chord;
    const py =  dx / chord;
    return {
      x: cx + side * d * px,
      y: cy + side * d * py,
    };
  }

  function drawArcTile(ctx, entry, exit, radius, hand, drawScale, theme, dim) {
    const side = hand === "left" ? +1 : -1;
    const center = arcCenter(entry, exit, radius, side);
    if (center == null) {
      // Degenerate — fall back to straight chord.
      drawChordTile(ctx, entry, exit, drawScale, theme, dim);
      return;
    }
    const HW = HALF_W_STUDS;
    const rIn = radius - HW;
    const rOut = radius + HW;
    const a0 = Math.atan2(entry.dy - center.y, entry.dx - center.x);
    const a1 = Math.atan2(exit.dy  - center.y, exit.dx  - center.x);

    // Determine fill arc direction so it sweeps the SHORT way for ≤π routes.
    // canvas.arc anticlockwise param: true for CCW.
    const ccw = side > 0;

    ctx.save();
    ctx.fillStyle = dim ? theme.curveDim : theme.curveFill;
    ctx.strokeStyle = theme.pieceStroke;
    ctx.lineWidth = 0.6 / drawScale;

    ctx.beginPath();
    ctx.arc(center.x, center.y, rOut, a0, a1, !ccw);
    ctx.arc(center.x, center.y, rIn,  a1, a0,  ccw);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // rails
    ctx.strokeStyle = theme.rail;
    ctx.lineWidth = 0.6 / drawScale;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius - 1.5, a0, a1, !ccw);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius + 1.5, a0, a1, !ccw);
    ctx.stroke();

    // sleepers
    ctx.strokeStyle = theme.sleeper;
    ctx.lineWidth = 0.4 / drawScale;
    let span = a1 - a0;
    // Normalize span to the same direction as ccw
    if (ccw && span < 0) span += 2 * Math.PI;
    if (!ccw && span > 0) span -= 2 * Math.PI;
    const N = Math.max(2, Math.round(Math.abs(span) / (Math.PI / 16)));   // ~5.6° per sleeper
    for (let i = 1; i < N; i++) {
      const a = a0 + (span * i) / N;
      const x1 = center.x + (rIn  + 0.5) * Math.cos(a);
      const y1 = center.y + (rIn  + 0.5) * Math.sin(a);
      const x2 = center.x + (rOut - 0.5) * Math.cos(a);
      const y2 = center.y + (rOut - 0.5) * Math.sin(a);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }

    ctx.restore();
  }

  // For each route in the piece, decide chord (straight) vs arc (curved).
  //
  // Rules:
  //   - Crossings: every route is a chord. Port dthetas describe outward
  //     orientations and may differ by π even though the train path is
  //     straight (e.g. CROSS_90 vertical: C/D ports face ±π/2, but the
  //     route C→D is a straight line through the body).
  //   - Straights: chord trivially.
  //   - Anything else: classify by the dtheta delta between entry and exit
  //     ports. dtheta=0 ⇒ chord; otherwise arc with radius derived from
  //     chord + sweep angle (so endpoints + tangents match exactly, even
  //     for switches where the spec's diverging_radius is the LEGO arc
  //     radius and the C port sits beyond the arc on a straight stub).
  function planRoutes(spec) {
    const plans = [];
    const ports = spec.ports;
    const kind = spec.kind;

    for (const [routeName, portSeq] of Object.entries(spec.routes || {})) {
      if (!portSeq || portSeq.length < 2) continue;
      const entry = ports[portSeq[0]];
      const exit  = ports[portSeq[portSeq.length - 1]];
      if (!entry || !exit) continue;

      const dtheta_diff = exit.dtheta - entry.dtheta;
      const isStraightGeom =
        kind === "crossing" ||
        kind === "straight" ||
        Math.abs(dtheta_diff) < EPS_THETA;

      if (isStraightGeom) {
        plans.push({ type: "chord", entry, exit, kind, routeName });
        continue;
      }

      const cx = exit.dx - entry.dx;
      const cy = exit.dy - entry.dy;
      const chord = Math.sqrt(cx * cx + cy * cy);
      const sweep = Math.abs(dtheta_diff);
      let radius;
      if (chord < 1e-6 || sweep < EPS_THETA) {
        radius = spec.radius_studs || spec.diverging_radius_studs || 40;
      } else {
        radius = chord / (2 * Math.sin(sweep / 2));
      }
      const arcHand = dtheta_diff > 0 ? "left" : "right";
      plans.push({ type: "arc", entry, exit, radius, hand: arcHand, kind, routeName });
    }
    return plans;
  }

  // Draw a single piece at world pose (port A) given catalog spec.
  // worldA_mm: {x, y, t} where x,y in mm and t in radians.
  // flip:   0/1 — Y-mirror across the piece longitudinal axis (curves' L↔R turn).
  // rotate: 0/1 — 180° in-plane rotation around the piece body center
  //         (switches' IN↔OUT placement). Applied BEFORE flip so the two
  //         transformations compose with the same convention the decoder
  //         uses internally when computing port poses.
  function drawPiece(ctx, vp, spec, worldA_mm, stud_mm, theme, opts) {
    if (!spec) return;
    const dim = !!(opts && opts.dim);
    const flip = (opts && opts.flip) ? 1 : 0;
    const rotate = (opts && opts.rotate) ? 1 : 0;

    ctx.save();
    // Place into world: port A at (worldA.x, worldA.y), heading t.
    ctx.translate(vp.toX(worldA_mm.x), vp.toY(worldA_mm.y));
    ctx.rotate(-worldA_mm.t);   // canvas Y inverted; angle flips sign
    // Local coords below are in studs with +y up.
    const drawScale = vp.scale * stud_mm;
    ctx.scale(drawScale, -drawScale);

    if (rotate) {
      // 180° rotation around piece body center (L/2, 0). In a (+y up) frame
      // this is equivalent to: translate(L, 0); scale(-1, -1).
      const L = spec.body_length_studs != null ? spec.body_length_studs
              : spec.length_studs != null ? spec.length_studs
              : 0;
      ctx.translate(L, 0);
      ctx.scale(-1, -1);
    }
    if (flip) ctx.scale(1, -1);

    const plans = planRoutes(spec);
    for (const plan of plans) {
      if (plan.type === "chord") {
        drawChordTile(ctx, plan.entry, plan.exit, drawScale, theme, dim);
      } else {
        drawArcTile(ctx, plan.entry, plan.exit, plan.radius, plan.hand, drawScale, theme, dim);
      }
    }

    ctx.restore();
  }

  function drawStartMarker(ctx, vp, world_mm, theme) {
    ctx.save();
    ctx.translate(vp.toX(world_mm.x), vp.toY(world_mm.y));
    ctx.rotate(-(world_mm.t || 0));
    ctx.fillStyle = theme.start;
    ctx.beginPath();
    ctx.arc(0, 0, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.5)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = theme.start;
    ctx.beginPath();
    ctx.moveTo(8, -4);
    ctx.lineTo(16, 0);
    ctx.lineTo(8, 4);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function render(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const theme = Object.assign({}, DEFAULT_THEME, opts.theme || {});
    const layout = opts.layout;
    const catalog = opts.catalog;
    const showGrid = opts.showGrid !== false;
    const dim = !!opts.dimPieces;

    clear(ctx, w, h, theme.bg);

    if (!layout || !layout.boundary_mm) {
      drawEmpty(ctx, w, h, theme, "no layout");
      return null;
    }
    const stud_mm = layout.stud_mm || (catalog && catalog.stud_mm) || 8.0;
    const boundary = layout.boundary_mm;
    const vp = makeViewport(canvas, boundary, 28);

    if (showGrid) drawGrid(ctx, vp, boundary, stud_mm, theme);
    drawBoundary(ctx, vp, boundary, theme);

    if (!catalog || !catalog.pieces) {
      drawEmpty(ctx, w, h, theme, "no catalog");
      return vp;
    }
    const specById = Object.fromEntries(catalog.pieces.map(p => [p.id, p]));

    let firstWorld = null;
    for (const piece of (layout.pieces || [])) {
      const spec = specById[piece.piece_id];
      if (!spec) continue;
      const pose_mm = {
        x: piece.pose_studs.x * stud_mm,
        y: piece.pose_studs.y * stud_mm,
        t: piece.pose_studs.theta,
      };
      const pieceTheme = pickBranchTheme(theme, piece);
      drawPiece(ctx, vp, spec, pose_mm, stud_mm, pieceTheme, {
        dim,
        flip: piece.flip,
        rotate: piece.rotate,
      });
      if (firstWorld == null) firstWorld = pose_mm;
    }

    // Start marker only for non-showcase layouts (showcase has multiple
    // disconnected anchors, no single "start").
    if (firstWorld && !layout.is_showcase) {
      drawStartMarker(ctx, vp, firstWorld, theme);
    }

    if (Array.isArray(layout.labels)) {
      drawLabels(ctx, vp, layout.labels, stud_mm, theme);
    }

    return vp;
  }

  // Draw piece labels in screen-space (after all canvas transforms reset).
  function drawLabels(ctx, vp, labels, stud_mm, theme) {
    ctx.save();
    ctx.fillStyle = theme.text;
    ctx.font = "11px ui-monospace, 'SF Mono', Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const lbl of labels) {
      const wx = (lbl.x_studs || 0) * stud_mm;
      const wy = (lbl.y_studs || 0) * stud_mm;
      ctx.fillText(String(lbl.text || ""), vp.toX(wx), vp.toY(wy) + 4);
    }
    ctx.restore();
  }

  function drawEmpty(ctx, w, h, theme, msg) {
    ctx.fillStyle = theme.text;
    ctx.font = "12px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(msg, w / 2, h / 2);
  }

  function toPNG(canvas) { return canvas.toDataURL("image/png"); }

  global.Render = { render, toPNG, DEFAULT_THEME };
})(window);
