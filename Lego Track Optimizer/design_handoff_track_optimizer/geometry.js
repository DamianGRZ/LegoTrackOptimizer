/* geometry.js — Pose math, port composition, closure checks.
 *
 * Pose: {x, y, t}  where (x,y) is in mm and t is heading in radians.
 * A "port" is a Pose in the piece's local frame.
 * A "route" through a piece carries the train from port `from` to port `to`,
 * and contributes a delta = compose(inverse(ports[from]), ports[to]).
 *
 * Conventions:
 *   - +x is "forward" along port A's heading.
 *   - Heading t increases counter-clockwise.
 *   - Composing world * local means: place `local` such that its origin sits at world.
 */

(function (global) {
  const TAU = Math.PI * 2;

  function mkPose(x, y, t) { return { x, y, t }; }
  const IDENTITY = mkPose(0, 0, 0);

  // Compose two poses: result = a * b  (apply b in a's frame).
  function compose(a, b) {
    const c = Math.cos(a.t), s = Math.sin(a.t);
    return {
      x: a.x + c * b.x - s * b.y,
      y: a.y + s * b.x + c * b.y,
      t: wrap(a.t + b.t),
    };
  }

  // Inverse pose: if compose(a, inv(a)) == identity.
  function inverse(a) {
    const c = Math.cos(-a.t), s = Math.sin(-a.t);
    return {
      x: c * (-a.x) - s * (-a.y),
      y: s * (-a.x) + c * (-a.y),
      t: wrap(-a.t),
    };
  }

  // Wrap angle into (-π, π].
  function wrap(t) {
    let r = t % TAU;
    if (r > Math.PI) r -= TAU;
    if (r <= -Math.PI) r += TAU;
    return r;
  }

  // Distance between two poses (positional).
  function poseDistance(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Heading difference, wrapped, absolute.
  function headingDelta(a, b) {
    return Math.abs(wrap(a.t - b.t));
  }

  // Closure check: does endPose match startPose within tolerance?
  function isClosed(startPose, endPose, tolMm, tolRad) {
    return poseDistance(startPose, endPose) <= tolMm
      && headingDelta(startPose, endPose) <= tolRad;
  }

  // Approximate equality used by tests.
  function poseEqual(a, b, eps = 1e-6) {
    return Math.abs(a.x - b.x) < eps
      && Math.abs(a.y - b.y) < eps
      && Math.abs(wrap(a.t - b.t)) < eps;
  }

  // Transform a local point by a world pose.
  function applyPose(pose, pt) {
    const c = Math.cos(pose.t), s = Math.sin(pose.t);
    return { x: pose.x + c * pt.x - s * pt.y, y: pose.y + s * pt.x + c * pt.y };
  }

  // AABB helpers. An aabb is {minX, minY, maxX, maxY}.
  function aabbFromPoints(pts) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    return { minX, minY, maxX, maxY };
  }
  function aabbIntersect(a, b, slack = 0) {
    return !(a.maxX - slack < b.minX || b.maxX - slack < a.minX
          || a.maxY - slack < b.minY || b.maxY - slack < a.minY);
  }
  function aabbInside(inner, outer) {
    return inner.minX >= outer.minX && inner.maxX <= outer.maxX
      && inner.minY >= outer.minY && inner.maxY <= outer.maxY;
  }
  function aabbUnion(a, b) {
    return {
      minX: Math.min(a.minX, b.minX), minY: Math.min(a.minY, b.minY),
      maxX: Math.max(a.maxX, b.maxX), maxY: Math.max(a.maxY, b.maxY),
    };
  }
  function aabbArea(a) {
    return Math.max(0, a.maxX - a.minX) * Math.max(0, a.maxY - a.minY);
  }

  // Transform a piece's local footprint (list of corner points) by a world pose,
  // then compute its AABB.
  function footprintAabb(piece, worldPose) {
    const pts = piece.footprint.map(p => applyPose(worldPose, p));
    return aabbFromPoints(pts);
  }

  global.Geom = {
    TAU, IDENTITY,
    mkPose, compose, inverse, wrap,
    poseDistance, headingDelta, isClosed, poseEqual,
    applyPose,
    aabbFromPoints, aabbIntersect, aabbInside, aabbUnion, aabbArea,
    footprintAabb,
  };
})(window);
