/* render.js — Top-down blueprint renderer.
 *
 * Draws:
 *   - Boundary rectangle
 *   - Optional grid (stud-pitch)
 *   - Each placement's footprint (filled by piece kind)
 *   - Centerline path (the actual rail)
 *   - Start marker, direction arrows
 */

(function (global) {
  const G = global.Geom;
  const C = global.Catalog;

  // World→canvas transform: fit boundary into canvas with margin, preserve aspect.
  function makeViewport(canvas, boundaryMm, marginPx = 24) {
    const wPx = canvas.width;
    const hPx = canvas.height;
    const sx = (wPx - 2 * marginPx) / boundaryMm.width;
    const sy = (hPx - 2 * marginPx) / boundaryMm.height;
    const s = Math.min(sx, sy);
    const ox = wPx / 2;
    const oy = hPx / 2;
    return {
      scale: s,
      toX: (x) => ox + x * s,
      toY: (y) => oy - y * s,    // flip Y so +y is up
      toLen: (l) => l * s,
    };
  }

  function clear(ctx, w, h, bg) {
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
  }

  function drawGrid(ctx, vp, boundaryMm, theme) {
    ctx.save();
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 0.5;
    const step = 80; // 10 studs
    const w = boundaryMm.width / 2;
    const h = boundaryMm.height / 2;
    for (let x = -w; x <= w + 0.001; x += step) {
      ctx.beginPath();
      ctx.moveTo(vp.toX(x), vp.toY(-h));
      ctx.lineTo(vp.toX(x), vp.toY(h));
      ctx.stroke();
    }
    for (let y = -h; y <= h + 0.001; y += step) {
      ctx.beginPath();
      ctx.moveTo(vp.toX(-w), vp.toY(y));
      ctx.lineTo(vp.toX(w), vp.toY(y));
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawBoundary(ctx, vp, boundaryMm, theme) {
    const w = boundaryMm.width / 2, h = boundaryMm.height / 2;
    ctx.save();
    ctx.strokeStyle = theme.boundary;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(vp.toX(-w), vp.toY(h), vp.toLen(boundaryMm.width), vp.toLen(boundaryMm.height));
    ctx.restore();
  }

  function drawPiece(ctx, vp, piece, originPose, theme, opts) {
    const { showLabels = false, dim = false } = opts || {};
    ctx.save();
    // Apply piece origin transform.
    ctx.translate(vp.toX(originPose.x), vp.toY(originPose.y));
    ctx.rotate(-originPose.t); // canvas Y is flipped; angle flips sign

    const drawScale = vp.scale;
    ctx.scale(drawScale, -drawScale); // local mm coords, with +y up

    if (piece.kind === "straight") {
      const L = piece.length;
      const HW = C.HALF_W;
      ctx.fillStyle = dim ? theme.straightDim : theme.straightFill;
      ctx.strokeStyle = theme.pieceStroke;
      ctx.lineWidth = 1 / drawScale;
      ctx.fillRect(0, -HW, L, 2 * HW);
      ctx.strokeRect(0, -HW, L, 2 * HW);
      // sleeper lines
      ctx.strokeStyle = theme.sleeper;
      ctx.lineWidth = 0.7 / drawScale;
      const sleepers = Math.max(2, Math.round(L / 32));
      for (let i = 1; i < sleepers; i++) {
        const x = (L * i) / sleepers;
        ctx.beginPath();
        ctx.moveTo(x, -HW + 4);
        ctx.lineTo(x, HW - 4);
        ctx.stroke();
      }
      // rails
      ctx.strokeStyle = theme.rail;
      ctx.lineWidth = 1 / drawScale;
      ctx.beginPath();
      ctx.moveTo(0, -HW + 12); ctx.lineTo(L, -HW + 12);
      ctx.moveTo(0,  HW - 12); ctx.lineTo(L,  HW - 12);
      ctx.stroke();
    } else if (piece.kind === "curve") {
      const dir = piece.dir;
      const cx = 0, cy = dir * C.R40;
      const a0 = -dir * Math.PI / 2;
      const a1 = a0 + dir * C.CURVE_ANGLE;
      const rIn = C.R40 - C.HALF_W, rOut = C.R40 + C.HALF_W;

      // tile (annular wedge)
      ctx.fillStyle = dim ? theme.curveDim : theme.curveFill;
      ctx.strokeStyle = theme.pieceStroke;
      ctx.lineWidth = 1 / drawScale;
      ctx.beginPath();
      const aLo = Math.min(a0, a1), aHi = Math.max(a0, a1);
      ctx.arc(cx, cy, rOut, aLo, aHi, false);
      ctx.arc(cx, cy, rIn, aHi, aLo, true);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // rails
      ctx.strokeStyle = theme.rail;
      ctx.lineWidth = 1 / drawScale;
      ctx.beginPath();
      ctx.arc(cx, cy, C.R40 - 12, aLo, aHi, false);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy, C.R40 + 12, aLo, aHi, false);
      ctx.stroke();

      // sleepers (5 across the arc)
      ctx.strokeStyle = theme.sleeper;
      ctx.lineWidth = 0.7 / drawScale;
      const N = 4;
      for (let i = 1; i < N; i++) {
        const a = a0 + (a1 - a0) * (i / N);
        const x1 = cx + (C.R40 - C.HALF_W + 4) * Math.cos(a);
        const y1 = cy + (C.R40 - C.HALF_W + 4) * Math.sin(a);
        const x2 = cx + (C.R40 + C.HALF_W - 4) * Math.cos(a);
        const y2 = cy + (C.R40 + C.HALF_W - 4) * Math.sin(a);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }
    }

    ctx.restore();
  }

  function drawStartMarker(ctx, vp, pose, theme) {
    ctx.save();
    ctx.translate(vp.toX(pose.x), vp.toY(pose.y));
    ctx.rotate(-pose.t);
    ctx.fillStyle = theme.start;
    ctx.beginPath();
    ctx.arc(0, 0, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = theme.bg;
    ctx.lineWidth = 2;
    ctx.stroke();
    // arrow
    ctx.fillStyle = theme.start;
    ctx.beginPath();
    ctx.moveTo(10, -5);
    ctx.lineTo(20, 0);
    ctx.lineTo(10, 5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // Sample world points along a route's centerline. Returns array of {x,y}.
  function sampleRoute(piece, route, originPose, n = 12) {
    const pts = [];
    if (piece.kind === "straight") {
      const A = piece.ports[route.from];
      const B = piece.ports[route.to];
      for (let i = 0; i <= n; i++) {
        const t = i / n;
        const lx = A.x + (B.x - A.x) * t;
        const ly = A.y + (B.y - A.y) * t;
        pts.push(G.applyPose(originPose, { x: lx, y: ly }));
      }
    } else if (piece.kind === "curve") {
      const dir = piece.dir;
      const cx = 0, cy = dir * C.R40;
      // start angle depends on which port we entered from.
      const aFrom = (route.from === 0 ? -dir * Math.PI / 2 : -dir * Math.PI / 2 + dir * C.CURVE_ANGLE);
      const aTo   = (route.to   === 0 ? -dir * Math.PI / 2 : -dir * Math.PI / 2 + dir * C.CURVE_ANGLE);
      for (let i = 0; i <= n; i++) {
        const t = i / n;
        const a = aFrom + (aTo - aFrom) * t;
        const lx = cx + C.R40 * Math.cos(a);
        const ly = cy + C.R40 * Math.sin(a);
        pts.push(G.applyPose(originPose, { x: lx, y: ly }));
      }
    }
    return pts;
  }

  function drawCenterline(ctx, vp, layout, catalogById, theme) {
    if (!layout || !layout.placements.length) return;
    ctx.save();
    ctx.strokeStyle = theme.centerline;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let first = true;
    for (const pl of layout.placements) {
      const piece = catalogById[pl.pieceId];
      const route = piece.routes[pl.routeIdx];
      const pts = sampleRoute(piece, route, pl.originPose, 16);
      for (const pt of pts) {
        const cx = vp.toX(pt.x), cy = vp.toY(pt.y);
        if (first) { ctx.moveTo(cx, cy); first = false; }
        else ctx.lineTo(cx, cy);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  function render(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    if (canvas._lastDpr !== dpr) {
      canvas._lastDpr = dpr;
    }
    const w = canvas.width, h = canvas.height;
    const theme = opts.theme;
    const boundary = opts.boundary;
    const layout = opts.layout;
    const catalog = opts.catalog;
    const showGrid = opts.showGrid !== false;
    const dim = !!opts.dimPieces;

    clear(ctx, w, h, theme.bg);
    const vp = makeViewport(canvas, boundary, 28);

    if (showGrid) drawGrid(ctx, vp, boundary, theme);
    drawBoundary(ctx, vp, boundary, theme);

    if (layout) {
      const catalogById = Object.fromEntries(catalog.map(p => [p.id, p]));
      for (const pl of layout.placements) {
        const piece = catalogById[pl.pieceId];
        drawPiece(ctx, vp, piece, pl.originPose, theme, { dim });
      }
      drawCenterline(ctx, vp, layout, catalogById, theme);
      if (layout.placements.length > 0) {
        drawStartMarker(ctx, vp, layout.start || { x: 0, y: 0, t: 0 }, theme);
      }
    }
    return vp;
  }

  // Export the canvas to a PNG data URL (for download).
  function toPNG(canvas) { return canvas.toDataURL("image/png"); }

  global.Render = { render, toPNG, sampleRoute };
})(window);
