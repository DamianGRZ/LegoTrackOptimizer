/* geometry.js — SE(2) pose math, port composition, AABB helpers.
 *
 * Pose: {x, y, t}  where (x,y) is in mm and t is heading in radians.
 * A "port" is a Pose in the piece's local frame.
 *
 * Conventions:
 *   - +x is "forward" along port A's heading.
 *   - Heading t increases counter-clockwise.
 *   - compose(world, local) places `local` such that its origin sits at world.
 */

(function (global) {
  const TAU = Math.PI * 2;

  function mkPose(x, y, t) { return { x, y, t }; }
  const IDENTITY = mkPose(0, 0, 0);

  function compose(a, b) {
    const c = Math.cos(a.t), s = Math.sin(a.t);
    return {
      x: a.x + c * b.x - s * b.y,
      y: a.y + s * b.x + c * b.y,
      t: wrap(a.t + b.t),
    };
  }

  function inverse(a) {
    const c = Math.cos(-a.t), s = Math.sin(-a.t);
    return {
      x: c * (-a.x) - s * (-a.y),
      y: s * (-a.x) + c * (-a.y),
      t: wrap(-a.t),
    };
  }

  function wrap(t) {
    let r = t % TAU;
    if (r > Math.PI) r -= TAU;
    if (r <= -Math.PI) r += TAU;
    return r;
  }

  function poseDistance(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function headingDelta(a, b) {
    return Math.abs(wrap(a.t - b.t));
  }

  function applyPose(pose, pt) {
    const c = Math.cos(pose.t), s = Math.sin(pose.t);
    return { x: pose.x + c * pt.x - s * pt.y, y: pose.y + s * pt.x + c * pt.y };
  }

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
  function aabbUnion(a, b) {
    return {
      minX: Math.min(a.minX, b.minX), minY: Math.min(a.minY, b.minY),
      maxX: Math.max(a.maxX, b.maxX), maxY: Math.max(a.maxY, b.maxY),
    };
  }

  global.Geom = {
    TAU, IDENTITY,
    mkPose, compose, inverse, wrap,
    poseDistance, headingDelta,
    applyPose,
    aabbFromPoints, aabbUnion,
  };
})(window);
