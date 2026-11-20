// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// The tube of a certified zero-set surface; contract in zero_set_tube.hpp.
#include "zero_set_tube.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "phydrax_meshcore.h"
#include "rounding.hpp"

namespace phx::mc::remesh {
namespace {
using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
// Rounds of adapting the lengths at the corners of refused triangles.
constexpr int kTubeRounds = 6;

IntervalVector exact(const Point& p) { return {point(p[0]), point(p[1]), point(p[2])}; }

// Unit angle-weighted vertex normals along `orientation` (proposals: the
// certificate uses the resulting binary64 vectors as given).
std::vector<Point> vertex_directions(const std::vector<Point>& points,
                                     const std::vector<Triangle>& triangles, int orientation) {
  std::vector<Point> sums(points.size(), Point{0, 0, 0});
  for (const Triangle& f : triangles) {
    const Point& a = points[static_cast<size_t>(f[0])];
    const Point& b = points[static_cast<size_t>(f[1])];
    const Point& c = points[static_cast<size_t>(f[2])];
    const Point ab{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const Point ac{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    Point normal{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                 ab[0] * ac[1] - ab[1] * ac[0]};
    const double length =
        std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (!(length > 0)) continue;
    for (double& x : normal) x /= length;
    for (size_t k = 0; k < 3; ++k) {
      const Point& p = points[static_cast<size_t>(f[k])];
      const Point& q = points[static_cast<size_t>(f[(k + 1) % 3])];
      const Point& r = points[static_cast<size_t>(f[(k + 2) % 3])];
      const Point u{q[0] - p[0], q[1] - p[1], q[2] - p[2]};
      const Point v{r[0] - p[0], r[1] - p[1], r[2] - p[2]};
      const double cosine = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) /
                            std::sqrt((u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) *
                                      (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
      const double angle = std::acos(std::clamp(cosine, -1.0, 1.0));
      for (size_t d = 0; d < 3; ++d) sums[static_cast<size_t>(f[k])][d] += angle * normal[d];
    }
  }
  for (Point& s : sums) {
    const double length = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    for (double& x : s) x = length > 0 ? orientation * x / length : 0.0;
  }
  return sums;
}

// The per-triangle quantities of the certificate (zero_set_tube.hpp).
struct Prism {
  double g = 0;          // min over pieces of lower(G . w) over T x [-1, 1]
  double near = 0;       // the same over T x [-tau, tau], tau = R_T / g (>= g)
  double residual = 0;   // R_T >= |f| over T
  double reach = 0;      // r = max_k |w_k| (upper)
  double clearance = 0;  // q: cap clearance decided within q of each cap
  double lipschitz = 0;  // |grad f| over the cap boxes widened by q (upper)
  Point wide_lower{}, wide_upper{};  // hull of the cap boxes widened by q
};

// For each prism, the largest R_T' over the triangles T' whose boxes meet
// its wide box (S widened by r): every point m of M within r of a cap point
// of T lies on such a triangle. Triangles are bucketed on a uniform grid of
// their boxes; buckets are sorted (cell key, triangle) pairs.
std::vector<double> local_residuals(const std::vector<Point>& points,
                                    const std::vector<Triangle>& triangles,
                                    const std::vector<Prism>& prisms) {
  const size_t m = triangles.size();
  Point origin{kInfinity, kInfinity, kInfinity};
  double extent = 0;
  std::vector<std::array<Point, 2>> boxes(m);
  for (size_t t = 0; t < m; ++t) {
    Point& lower = boxes[t][0];
    Point& upper = boxes[t][1];
    for (int d = 0; d < 3; ++d) {
      lower[d] = kInfinity;
      upper[d] = -kInfinity;
      for (const int64_t v : triangles[t]) {
        lower[d] = std::min(lower[d], points[static_cast<size_t>(v)][d]);
        upper[d] = std::max(upper[d], points[static_cast<size_t>(v)][d]);
      }
      origin[d] = std::min(origin[d], lower[d]);
      extent = std::max(extent, prisms[t].wide_upper[d] - prisms[t].wide_lower[d]);
    }
  }
  // Cells about half a wide box across: a query spans a few cells per axis.
  const double cell = std::max(0.5 * extent, std::numeric_limits<double>::min());
  const auto index = [&](double x, int d) {
    return static_cast<int64_t>(std::floor((x - origin[d]) / cell));
  };
  const auto key = [](int64_t i, int64_t j, int64_t k) {
    return (static_cast<uint64_t>(i) << 42) | (static_cast<uint64_t>(j) << 21) |
           static_cast<uint64_t>(k);
  };
  std::vector<std::pair<uint64_t, int64_t>> buckets;
  for (size_t t = 0; t < m; ++t)
    for (int64_t i = index(boxes[t][0][0], 0); i <= index(boxes[t][1][0], 0); ++i)
      for (int64_t j = index(boxes[t][0][1], 1); j <= index(boxes[t][1][1], 1); ++j)
        for (int64_t k = index(boxes[t][0][2], 2); k <= index(boxes[t][1][2], 2); ++k)
          buckets.push_back({key(i, j, k), static_cast<int64_t>(t)});
  std::sort(buckets.begin(), buckets.end());
  std::vector<double> local(m, 0.0);
  for (size_t t = 0; t < m; ++t) {
    const Prism& prism = prisms[t];
    double largest = prism.residual;
    // Cells below the grid's origin hold no triangle box.
    const auto clamp = [](int64_t i) { return std::max<int64_t>(i, 0); };
    for (int64_t i = clamp(index(prism.wide_lower[0], 0)); i <= index(prism.wide_upper[0], 0); ++i)
      for (int64_t j = clamp(index(prism.wide_lower[1], 1)); j <= index(prism.wide_upper[1], 1);
           ++j)
        for (int64_t k = clamp(index(prism.wide_lower[2], 2));
             k <= index(prism.wide_upper[2], 2); ++k) {
          const uint64_t at = key(i, j, k);
          auto it = std::lower_bound(buckets.begin(), buckets.end(),
                                     std::pair<uint64_t, int64_t>{at, -1});
          for (; it != buckets.end() && it->first == at; ++it) {
            const size_t u = static_cast<size_t>(it->second);
            bool meets = true;
            for (int d = 0; d < 3; ++d)
              meets = meets && boxes[u][0][d] <= prism.wide_upper[d] &&
                      boxes[u][1][d] >= prism.wide_lower[d];
            if (meets) largest = std::max(largest, prisms[u].residual);
          }
        }
    local[t] = largest;
  }
  return local;
}

// How a prism failed: too short (the caps are not clear of the zero, so the
// corners' lengths double) or too long (a gradient sign, the orientation or
// an enclosure fails over the swept box, so they halve).
enum class Verdict : uint8_t { kAccepted, kShort, kLong };

// A piece of a prism: a sub-triangle (corners and their w as enclosures) and
// an interval of t.
struct Piece {
  std::array<IntervalVector, 3> corner, along;
  Interval t;
};

// The piece's eight children: four midpoint sub-triangles times two halves
// of t (every point of the piece's swept set lies in one of them).
std::array<Piece, 8> children_of(const Piece& piece) {
  const auto middle = [](const IntervalVector& a, const IntervalVector& b) {
    IntervalVector m{};
    for (size_t d = 0; d < 3; ++d) m[d] = multiply(add(a[d], b[d]), point(0.5));
    return m;
  };
  const auto& c = piece.corner;
  const auto& w = piece.along;
  const std::array<IntervalVector, 6> corners{c[0], c[1], c[2], middle(c[0], c[1]),
                                              middle(c[1], c[2]), middle(c[2], c[0])};
  const std::array<IntervalVector, 6> alongs{w[0], w[1], w[2], middle(w[0], w[1]),
                                             middle(w[1], w[2]), middle(w[2], w[0])};
  constexpr std::array<std::array<size_t, 3>, 4> kChildren{{{0, 3, 5}, {3, 1, 4}, {5, 4, 2}, {3, 4, 5}}};
  const double split = 0.5 * piece.t.lower + 0.5 * piece.t.upper;
  std::array<Piece, 8> out{};
  size_t next = 0;
  for (const auto& child : kChildren)
    for (const Interval half : {Interval{piece.t.lower, split}, Interval{split, piece.t.upper}}) {
      for (size_t k = 0; k < 3; ++k) {
        out[next].corner[k] = corners[child[k]];
        out[next].along[k] = alongs[child[k]];
      }
      out[next++].t = half;
    }
  return out;
}

// Levels of piece refinement for the monotonicity and orientation bounds:
// each splits a piece into its eight children.
constexpr int kPieceLevels = 2;
// Levels of sub-triangle refinement for the residual bound.
constexpr int kResidualLevels = 4;

// A lower bound of grad f . w(y) over the piece's swept set: the minimum
// over its corners' w of lower(G . w), G enclosing grad f over the outward
// box of the corners moved by t w (w(y) is a convex combination of the
// corners' w, every swept point in that box). Pieces whose bound falls below
// `target` are refined, each child holding part of the swept set, and the
// minimum over the children kept. -inf when an enclosure is not finite.
double swept_bound(const Piece& piece, const IntervalBoxProgram& gradient, double target,
                   int levels) {
  Point lower{kInfinity, kInfinity, kInfinity}, upper{-kInfinity, -kInfinity, -kInfinity};
  for (size_t k = 0; k < 3; ++k)
    for (size_t d = 0; d < 3; ++d) {
      const Interval at = add(piece.corner[k][d], multiply(piece.t, piece.along[k][d]));
      lower[d] = std::min(lower[d], at.lower);
      upper[d] = std::max(upper[d], at.upper);
    }
  IntervalVector field_gradient{};
  if (!gradient.evaluate(lower, upper, field_gradient.data()) || !finite(field_gradient[0]) ||
      !finite(field_gradient[1]) || !finite(field_gradient[2]))
    return -kInfinity;
  double g = kInfinity;
  for (size_t k = 0; k < 3; ++k) g = std::min(g, dot(field_gradient, piece.along[k]).lower);
  if (g >= target || levels == 0) return g;
  double lowest = kInfinity;
  for (const Piece& child : children_of(piece)) {
    lowest = std::min(lowest, swept_bound(child, gradient, target, levels - 1));
    if (!(lowest > 0)) return lowest;
  }
  return lowest;
}

// Whether det[dF/ds, dF/dr, dF/dt] has the orientation's sign over the
// piece: dF/ds = e1 + t (w1 - w0) and dF/dr = e2 + t (w2 - w0) with s, r the
// whole triangle's barycentric parameters, dF/dt = w(y) in the box hull of
// the piece's corners' w; refined over the piece's children.
bool oriented_over(const Piece& piece, const IntervalVector& e1, const IntervalVector& e2,
                   const IntervalVector& dw1, const IntervalVector& dw2, int orientation,
                   int levels) {
  IntervalVector ds{}, dr{}, hull{};
  for (size_t d = 0; d < 3; ++d) {
    ds[d] = add(e1[d], multiply(piece.t, dw1[d]));
    dr[d] = add(e2[d], multiply(piece.t, dw2[d]));
    hull[d] = {least(piece.along[0][d].lower, piece.along[1][d].lower, piece.along[2][d].lower),
               greatest(piece.along[0][d].upper, piece.along[1][d].upper,
                        piece.along[2][d].upper)};
  }
  const Interval determinant = dot(cross(ds, dr), hull);
  if (finite(determinant) && (orientation > 0 ? determinant.lower > 0 : determinant.upper < 0))
    return true;
  if (levels == 0 || !finite(determinant)) return false;
  for (const Piece& child : children_of(piece))
    if (!oriented_over(child, e1, e2, dw1, dw2, orientation, levels - 1)) return false;
  return true;
}

// An upper bound of |f| over the piece's (sub-)triangle (its t ignored): f
// over the outward box of its corners intersected with the mean-value forms
// anchored at each corner, f(p) + a G . (q - p) + b G . (r - p) over the
// simplex (G enclosing grad f over that box, f(p) over the corner's own
// enclosure, the edges as interval differences). Pieces above `target` are
// refined into their four sub-triangles, every point of the triangle lying
// in one, and the maximum kept. +inf when an enclosure is not finite.
double residual_bound(const Piece& piece, const IntervalBoxProgram& value,
                      const IntervalBoxProgram& gradient, double target, int levels) {
  Point lower{kInfinity, kInfinity, kInfinity}, upper{-kInfinity, -kInfinity, -kInfinity};
  for (size_t k = 0; k < 3; ++k)
    for (size_t d = 0; d < 3; ++d) {
      lower[d] = std::min(lower[d], piece.corner[k][d].lower);
      upper[d] = std::max(upper[d], piece.corner[k][d].upper);
    }
  Interval field{};
  IntervalVector field_gradient{};
  if (!value.evaluate(lower, upper, &field) || !finite(field) ||
      !gradient.evaluate(lower, upper, field_gradient.data()) || !finite(field_gradient[0]) ||
      !finite(field_gradient[1]) || !finite(field_gradient[2]))
    return kInfinity;
  for (size_t k = 0; k < 3; ++k) {
    const IntervalVector& p = piece.corner[k];
    const IntervalVector& q = piece.corner[(k + 1) % 3];
    const IntervalVector& r = piece.corner[(k + 2) % 3];
    Interval anchor{};
    if (!value.evaluate({p[0].lower, p[1].lower, p[2].lower}, {p[0].upper, p[1].upper, p[2].upper},
                        &anchor) ||
        !finite(anchor))
      continue;
    Interval along_q{0, 0}, along_r{0, 0};
    for (size_t d = 0; d < 3; ++d) {
      along_q = add(along_q, multiply(field_gradient[d], subtract(q[d], p[d])));
      along_r = add(along_r, multiply(field_gradient[d], subtract(r[d], p[d])));
    }
    const Interval form = add(anchor, {least(0.0, along_q.lower, along_r.lower),
                                       greatest(0.0, along_q.upper, along_r.upper)});
    if (!finite(form)) continue;
    field = {std::max(field.lower, form.lower), std::min(field.upper, form.upper)};
    if (field.lower > field.upper) return kInfinity;
  }
  const double bound = magnitude(field);
  if (bound <= target || levels == 0) return bound;
  double highest = 0;
  const std::array<Piece, 8> children = children_of(piece);
  for (size_t c = 0; c < children.size(); c += 2) {
    highest = std::max(highest, residual_bound(children[c], value, gradient, target, levels - 1));
    if (!(highest < kInfinity)) return highest;
  }
  return highest;
}

// Decides the prism of T and fills `prism`; the global rho is decided after.
Verdict decide(const Point& p0, const Point& p1, const Point& p2, const Point& w0,
               const Point& w1, const Point& w2, const IntervalBoxProgram& value,
               const IntervalBoxProgram& gradient, int orientation, double tolerance,
               Prism& prism) {
  const std::array<const Point*, 3> corner{&p0, &p1, &p2};
  const std::array<const Point*, 3> along{&w0, &w1, &w2};
  // The cap boxes: outward boxes of p_k + w_k and of p_k - w_k.
  std::array<Point, 2> cap_lower{}, cap_upper{};
  for (size_t side = 0; side < 2; ++side) {
    cap_lower[side] = {kInfinity, kInfinity, kInfinity};
    cap_upper[side] = {-kInfinity, -kInfinity, -kInfinity};
  }
  double reach = 0;
  for (size_t k = 0; k < 3; ++k) {
    for (int d = 0; d < 3; ++d)
      for (size_t side = 0; side < 2; ++side) {
        const double sign = side == 0 ? 1.0 : -1.0;
        const Interval end = add(point((*corner[k])[d]), point(sign * (*along[k])[d]));
        cap_lower[side][d] = std::min(cap_lower[side][d], end.lower);
        cap_upper[side][d] = std::max(cap_upper[side][d], end.upper);
      }
    reach = std::max(reach, norm_upper(exact(*along[k])));
  }
  // The monotonicity bound, refined toward half of its value at the
  // centroid (a point enclosure of grad f there against the corners' w).
  Point centroid{};
  for (int d = 0; d < 3; ++d) centroid[d] = (p0[d] + p1[d] + p2[d]) / 3;
  IntervalVector at_centroid{};
  if (!gradient.evaluate(centroid, centroid, at_centroid.data()) || !finite(at_centroid[0]) ||
      !finite(at_centroid[1]) || !finite(at_centroid[2]))
    return Verdict::kLong;
  double estimate = kInfinity;
  for (size_t k = 0; k < 3; ++k)
    estimate = std::min(estimate, dot(at_centroid, exact(*along[k])).lower);
  const Piece whole{{exact(p0), exact(p1), exact(p2)}, {exact(w0), exact(w1), exact(w2)}, {-1, 1}};
  const double g = swept_bound(whole, gradient, 0.5 * estimate, kPieceLevels);
  if (!(g > 0)) return Verdict::kLong;
  // R_T, refined toward the residual that keeps this prism's reverse bound
  // (R_T / g) |w| within the tolerance, and at most a quarter of g.
  const double residual = residual_bound(
      whole, value, gradient, std::min(0.25 * g, tolerance * g / reach), kResidualLevels);
  if (!(residual < kInfinity)) return Verdict::kLong;
  if (!(next_down(g - residual) > 0)) return Verdict::kShort;
  // The zero on each segment lies at |t| <= tau = R_T / g; over that thin
  // slab grad f . w is bounded again (a smaller box, so a larger bound
  // g_slab), and the zero lies at |t| <= R_T / g_slab (reverse).
  double g_slab = g;
  const double tau = next_up(residual / g);
  if (tau < 1) {
    const Piece slab{whole.corner, whole.along, {-tau, tau}};
    g_slab = std::max(g, swept_bound(slab, gradient, 0.9 * estimate, kPieceLevels));
  }
  // det[dF/ds, dF/dr, dF/dt] over T x [-1, 1] (oriented_over).
  const IntervalVector e1 = enclosed_difference(p1, p0), e2 = enclosed_difference(p2, p0);
  const IntervalVector dw1 = enclosed_difference(w1, w0), dw2 = enclosed_difference(w2, w0);
  if (!oriented_over(whole, e1, e2, dw1, dw2, orientation, kPieceLevels)) return Verdict::kLong;
  // Cap clearance is decided within q = r / 2 of each cap: |grad f| over
  // each cap box widened by q, and the wide box (both, hulled) for the
  // residual of the triangles a cap could reach.
  const double clearance = 0.5 * reach;
  Point lower{kInfinity, kInfinity, kInfinity}, upper{-kInfinity, -kInfinity, -kInfinity};
  double lipschitz = 0;
  for (size_t side = 0; side < 2; ++side) {
    Point low{}, high{};
    for (int d = 0; d < 3; ++d) {
      low[d] = next_down(cap_lower[side][d] - clearance);
      high[d] = next_up(cap_upper[side][d] + clearance);
      lower[d] = std::min(lower[d], low[d]);
      upper[d] = std::max(upper[d], high[d]);
    }
    IntervalVector near{};
    if (!gradient.evaluate(low, high, near.data()) || !finite(near[0]) || !finite(near[1]) ||
        !finite(near[2]))
      return Verdict::kLong;
    lipschitz = std::max(lipschitz, norm_upper(near));
  }
  prism = {g, g_slab, residual, reach, clearance, lipschitz, lower, upper};
  return std::isfinite(prism.lipschitz) && prism.lipschitz > 0 ? Verdict::kAccepted
                                                                : Verdict::kLong;
}
}  // namespace

int32_t certify_tube(const std::vector<Point>& points, const std::vector<Triangle>& triangles,
                     const IntervalBoxProgram& value, const IntervalBoxProgram& gradient,
                     int orientation, double length, double tolerance,
                     TubeCertificate& certificate) {
  if ((orientation != 1 && orientation != -1) || !(length > 0) || !std::isfinite(length) ||
      !(tolerance > 0) || !std::isfinite(tolerance))
    return PHX_MC_INVALID_ARGUMENT;
  if (triangles.empty() || !closed_manifold(triangles, points.size())) return PHX_MC_INVALID_INPUT;
  const std::vector<Point> directions = vertex_directions(points, triangles, orientation);
  std::vector<double> lengths(points.size(), length);
  std::vector<Prism> prisms(triangles.size());
  std::vector<Verdict> verdicts(triangles.size(), Verdict::kAccepted);
  certificate = TubeCertificate{};
  for (int round = 0; round < kTubeRounds; ++round) {
    certificate.rounds = round + 1;
    std::vector<Point> w(points.size());
    for (size_t v = 0; v < points.size(); ++v)
      for (size_t d = 0; d < 3; ++d) w[v][d] = lengths[v] * directions[v][d];
    int64_t failed = 0;
    for (size_t t = 0; t < triangles.size(); ++t) {
      const Triangle& f = triangles[t];
      const auto at = [](const std::vector<Point>& values, int64_t i) -> const Point& {
        return values[static_cast<size_t>(i)];
      };
      verdicts[t] = decide(at(points, f[0]), at(points, f[1]), at(points, f[2]), at(w, f[0]),
                           at(w, f[1]), at(w, f[2]), value, gradient, orientation, tolerance,
                           prisms[t]);
      if (verdicts[t] != Verdict::kAccepted) ++failed;
    }
    // rho needs each cap's clearance from the residual of the triangles it
    // could reach (local_residuals): decided only when no triangle is
    // refused, and a triangle whose cap is not clear of it is refused.
    double rho = kInfinity, reverse = 0;
    if (failed == 0) {
      const std::vector<double> nearby = local_residuals(points, triangles, prisms);
      for (size_t t = 0; t < triangles.size(); ++t) {
        const Prism& prism = prisms[t];
        const double margin = next_down(next_down(prism.g - prism.residual) - nearby[t]);
        if (!(margin > 0)) {
          verdicts[t] = Verdict::kShort;
          ++failed;
          continue;
        }
        rho = std::min(rho, std::min(prism.clearance, next_down(margin / prism.lipschitz)));
        reverse = std::max(reverse, next_up(next_up(prism.residual / prism.near) * prism.reach));
      }
    }
    certificate.refused = failed;
    if (failed == 0) {
      certificate.rho = rho;
      certificate.reverse = reverse;
      return PHX_MC_OK;
    }
    // Halving wins over doubling at a vertex shared by both kinds.
    std::vector<double> scale(points.size(), 1.0);
    for (size_t t = 0; t < triangles.size(); ++t)
      for (const int64_t v : triangles[t]) {
        double& s = scale[static_cast<size_t>(v)];
        if (verdicts[t] == Verdict::kLong) s = 0.5;
        else if (verdicts[t] == Verdict::kShort && s == 1.0) s = 2.0;
      }
    for (size_t v = 0; v < points.size(); ++v) lengths[v] *= scale[v];
  }
  return PHX_MC_UNRESOLVED;
}

bool box_within_reach(const MeshSource& mesh, const Point& lower, const Point& upper,
                      double reach) {
  Point center{};
  IntervalVector half{};
  double scale = 0;
  for (int d = 0; d < 3; ++d) {
    if (!std::isfinite(lower[d]) || !std::isfinite(upper[d]) || !(lower[d] <= upper[d]))
      return false;
    center[d] = 0.5 * lower[d] + 0.5 * upper[d];
    // Every box point is within max(c - lower, upper - c) of the computed
    // centre c in this coordinate.
    const Interval below = subtract(point(center[d]), point(lower[d]));
    const Interval above = subtract(point(upper[d]), point(center[d]));
    half[static_cast<size_t>(d)] = {0, std::max(below.upper, above.upper)};
    scale = std::max(scale, std::max(std::fabs(lower[d]), std::fabs(upper[d])));
  }
  const MeshSource::Nearest nearest = mesh.nearest(center);
  if (nearest.triangle < 0 || !std::isfinite(nearest.distance)) return false;
  // The computed closest point is a rounded convex combination of the
  // triangle's corners: within 16 u of the coordinate scale of a point of the
  // triangle; the computed distance to it within a relative 8 u.
  for (const double x : nearest.point) scale = std::max(scale, std::fabs(x));
  const double offset = next_up(nearest.distance * (1 + 0x1p-50) + 0x1p-49 * scale);
  const double bound = next_up(offset + norm_upper(half));
  return bound < reach;
}

}  // namespace phx::mc::remesh
