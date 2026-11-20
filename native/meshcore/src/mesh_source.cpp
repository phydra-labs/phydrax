// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Mesh source; contract in mesh_source.hpp.
#include "mesh_source.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc::remesh {
namespace {

Point sub(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Point& a) { return std::sqrt(dot(a, a)); }

// Closest point of triangle (a, b, c) to p by Voronoi-region classification
// (Ericson), with its barycentric coordinates.
Point closest_point(const Point& p, const Point& a, const Point& b, const Point& c,
                    std::array<double, 3>& weights) {
  const Point ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) {
    weights = {1, 0, 0};
    return a;
  }
  const Point bp = sub(p, b);
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) {
    weights = {0, 1, 0};
    return b;
  }
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const double v = d1 / (d1 - d3);
    weights = {1 - v, v, 0};
    return {a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]};
  }
  const Point cp = sub(p, c);
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) {
    weights = {0, 0, 1};
    return c;
  }
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const double w = d2 / (d2 - d6);
    weights = {1 - w, 0, w};
    return {a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]};
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    weights = {0, 1 - w, w};
    return {b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])};
  }
  const double denominator = 1 / (va + vb + vc);
  const double v = vb * denominator, w = vc * denominator;
  weights = {1 - v - w, v, w};
  return {a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w,
          a[2] + ab[2] * v + ac[2] * w};
}

}  // namespace

int32_t MeshSource::bind(std::vector<Point> points, std::vector<Triangle> triangles) {
  double scale = 0;
  for (const Point& p : points)
    for (const double x : p) {
      if (!std::isfinite(x)) return PHX_MC_NONFINITE_INPUT;
      scale = std::max(scale, std::abs(x));
    }
  std::vector<int64_t> twins;
  if (triangles.empty() || !closed_manifold(triangles, points.size(), &twins))
    return PHX_MC_INVALID_INPUT;
  points_ = std::move(points);
  triangles_ = std::move(triangles);
  scale_ = scale;
  const size_t n = points_.size(), m = triangles_.size();
  twins_ = std::move(twins);
  face_normals_.assign(m, Point{0, 0, 0});
  vertex_normals_.assign(n, Point{0, 0, 0});
  balls_.resize(m);
  double extents = 0;
  for (size_t t = 0; t < m; ++t) {
    const Triangle& f = triangles_[t];
    const Point& a = points_[static_cast<size_t>(f[0])];
    const Point& b = points_[static_cast<size_t>(f[1])];
    const Point& c = points_[static_cast<size_t>(f[2])];
    const Point normal = cross(sub(b, a), sub(c, a));
    const double length = norm(normal);
    if (!(length > 0)) return PHX_MC_INVALID_INPUT;
    face_normals_[t] = {normal[0] / length, normal[1] / length, normal[2] / length};
    // Angle-weighted vertex normals.
    for (size_t k = 0; k < 3; ++k) {
      const Point& p = points_[static_cast<size_t>(f[k])];
      const Point u = sub(points_[static_cast<size_t>(f[(k + 1) % 3])], p);
      const Point w = sub(points_[static_cast<size_t>(f[(k + 2) % 3])], p);
      const double angle = std::atan2(norm(cross(u, w)), dot(u, w));
      Point& sum = vertex_normals_[static_cast<size_t>(f[k])];
      for (int d = 0; d < 3; ++d) sum[d] += angle * face_normals_[t][d];
    }
    double extent = 0;
    for (int d = 0; d < 3; ++d)
      extent = std::max(extent, std::max(std::max(a[d], b[d]), c[d]) -
                                    std::min(std::min(a[d], b[d]), c[d]));
    extents += extent;
    const Point centroid{(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3,
                         (a[2] + b[2] + c[2]) / 3};
    const double radius =
        std::sqrt(std::max(std::max(dot(sub(a, centroid), sub(a, centroid)),
                                    dot(sub(b, centroid), sub(b, centroid))),
                           dot(sub(c, centroid), sub(c, centroid))));
    balls_[t] = {centroid[0], centroid[1], centroid[2], radius};
  }
  for (Point& normal : vertex_normals_) {
    const double length = norm(normal);
    if (length > 0)
      for (double& x : normal) x /= length;
  }
  // Normal curvature along each edge d = x_w - x_v, (n_w - n_v) . d / |d|^2
  // (positive where the surface bends away from its normal); by Euler's
  // theorem the largest and smallest over a vertex's ring estimate its two
  // principal curvatures. The value is the same (bitwise) in either
  // direction, so each edge is taken once, from its lower corner.
  vertex_curvatures_.assign(n, {-std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::infinity()});
  vertex_corners_.assign(n, -1);
  for (size_t c = 0; c < 3 * m; ++c) {
    const Triangle& f = triangles_[c / 3];
    const auto v = static_cast<size_t>(f[c % 3]), w = static_cast<size_t>(f[(c % 3 + 1) % 3]);
    vertex_corners_[v] = static_cast<int64_t>(c);
    if (twins_[c] < static_cast<int64_t>(c)) continue;
    const Point d = sub(points_[w], points_[v]);
    const double kappa = dot(sub(vertex_normals_[w], vertex_normals_[v]), d) / dot(d, d);
    for (const size_t u : {v, w}) {
      vertex_curvatures_[u][0] = std::max(vertex_curvatures_[u][0], kappa);
      vertex_curvatures_[u][1] = std::min(vertex_curvatures_[u][1], kappa);
    }
  }
  // Bucket grid of the triangle boxes, cells of twice the mean box extent: a
  // box meets one or two cells per axis, so the grid holds fewer entries,
  // and a query near the surface still finds its nearest triangle in its own
  // cell or the next ring.
  Point top{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()};
  origin_ = {-top[0], -top[1], -top[2]};
  for (const Point& p : points_)
    for (int d = 0; d < 3; ++d) {
      origin_[d] = std::min(origin_[d], p[d]);
      top[d] = std::max(top[d], p[d]);
    }
  cell_ = 2.0 * extents / static_cast<double>(m);
  for (int d = 0; d < 3; ++d) cell_ = std::max(cell_, (top[d] - origin_[d]) * 0x1p-10);
  if (!(cell_ > 0)) cell_ = 1;
  // The cells are a dense array over the bounding box: coarsened (doubled)
  // until it holds at most kCellsPerTriangle cells per triangle, so a long
  // thin or sparse surface cannot make it outgrow the triangles.
  constexpr double kCellsPerTriangle = 16;
  int64_t total = 1;
  for (;;) {
    double cells = 1;
    for (int d = 0; d < 3; ++d) cells *= std::floor((top[d] - origin_[d]) / cell_) + 1;
    if (cells <= kCellsPerTriangle * static_cast<double>(m)) break;
    cell_ *= 2;
  }
  for (int d = 0; d < 3; ++d) {
    shape_[static_cast<size_t>(d)] = static_cast<int64_t>(std::floor((top[d] - origin_[d]) / cell_)) + 1;
    total *= shape_[static_cast<size_t>(d)];
  }
  std::vector<std::array<int64_t, 6>> ranges(m);
  start_.assign(static_cast<size_t>(total) + 1, 0);
  for (size_t t = 0; t < m; ++t) {
    Point lower = points_[static_cast<size_t>(triangles_[t][0])], upper = lower;
    for (size_t k = 1; k < 3; ++k)
      for (int d = 0; d < 3; ++d) {
        lower[d] = std::min(lower[d], points_[static_cast<size_t>(triangles_[t][k])][d]);
        upper[d] = std::max(upper[d], points_[static_cast<size_t>(triangles_[t][k])][d]);
      }
    const auto low = cell_of(lower), high = cell_of(upper);
    ranges[t] = {low[0], low[1], low[2], high[0], high[1], high[2]};
    for (int64_t i = low[0]; i <= high[0]; ++i)
      for (int64_t j = low[1]; j <= high[1]; ++j)
        for (int64_t k = low[2]; k <= high[2]; ++k) ++start_[static_cast<size_t>(flat({i, j, k})) + 1];
  }
  for (int64_t c = 0; c < total; ++c) start_[static_cast<size_t>(c) + 1] += start_[static_cast<size_t>(c)];
  items_.resize(static_cast<size_t>(start_.back()));
  std::vector<int64_t> fill(start_.begin(), start_.end() - 1);
  for (size_t t = 0; t < m; ++t) {
    const auto& r = ranges[t];
    for (int64_t i = r[0]; i <= r[3]; ++i)
      for (int64_t j = r[1]; j <= r[4]; ++j)
        for (int64_t k = r[2]; k <= r[5]; ++k)
          items_[static_cast<size_t>(fill[static_cast<size_t>(flat({i, j, k}))]++)] =
              static_cast<int64_t>(t);
  }
  visited_.assign(m, 0);
  included_.assign(m, 0);
  tested_.assign(m, 0);
  vertex_stamps_.assign(n, 0);
  projected_.resize(n);
  stamp_ = 0;
  return PHX_MC_OK;
}

std::array<int64_t, 3> MeshSource::cell_of(const Point& p) const {
  std::array<int64_t, 3> c{};
  for (int d = 0; d < 3; ++d) {
    const double t = std::floor((p[d] - origin_[d]) / cell_);
    c[static_cast<size_t>(d)] = static_cast<int64_t>(
        std::clamp(t, 0.0, static_cast<double>(shape_[static_cast<size_t>(d)] - 1)));
  }
  return c;
}

MeshSource::Nearest MeshSource::nearest(const Point& position) const {
  Nearest best;
  best.distance = std::numeric_limits<double>::infinity();
  if (triangles_.empty()) return best;
  const auto center = cell_of(position);
  int64_t reach = 0;
  for (size_t d = 0; d < 3; ++d)
    reach = std::max(reach, std::max(center[d], shape_[d] - 1 - center[d]));
  ++stamp_;
  // The position's clearance inside its own cell (0 when it lies outside the
  // grid): a ball of that radius around it lies in the cell.
  double inner = std::numeric_limits<double>::infinity();
  for (size_t d = 0; d < 3; ++d) {
    const double low = origin_[d] + static_cast<double>(center[d]) * cell_;
    inner = std::min(inner, std::min(position[d] - low, low + cell_ - position[d]));
  }
  inner = std::max(inner, 0.0);
  // Squared distance from the position to a cell's box: a cell no closer
  // than the best distance found holds no closer point of any triangle.
  const auto gap2 = [&](int64_t i, int64_t j, int64_t k) {
    const std::array<int64_t, 3> index{i, j, k};
    double sum = 0;
    for (size_t d = 0; d < 3; ++d) {
      const double low = origin_[d] + static_cast<double>(index[d]) * cell_;
      const double below = low - position[d], above = position[d] - (low + cell_);
      const double gap = below > 0 ? below : (above > 0 ? above : 0.0);
      sum += gap * gap;
    }
    return sum;
  };
  for (int64_t r = 0; r <= reach; ++r) {
    // Every triangle within (r - 1) * cell + inner of the position meets a
    // cell within ring r - 1 of the position's (clamped) cell, all scanned.
    if (r > 0 && best.distance <= static_cast<double>(r - 1) * cell_ + inner) break;
    for (int64_t i = std::max<int64_t>(0, center[0] - r); i <= std::min(shape_[0] - 1, center[0] + r);
         ++i)
      for (int64_t j = std::max<int64_t>(0, center[1] - r);
           j <= std::min(shape_[1] - 1, center[1] + r); ++j) {
        const bool shell = std::abs(i - center[0]) == r || std::abs(j - center[1]) == r;
        for (int64_t k = std::max<int64_t>(0, center[2] - r);
             k <= std::min(shape_[2] - 1, center[2] + r); ++k) {
          if (!shell && std::abs(k - center[2]) != r) continue;
          if (gap2(i, j, k) >= best.distance * best.distance) continue;
          const int64_t cell = flat({i, j, k});
          for (int64_t s = start_[static_cast<size_t>(cell)];
               s < start_[static_cast<size_t>(cell) + 1]; ++s) {
            const int64_t t = items_[static_cast<size_t>(s)];
            if (visited_[static_cast<size_t>(t)] == stamp_) continue;
            visited_[static_cast<size_t>(t)] = stamp_;
            // The triangle lies in its bounding ball: skip it when the ball
            // is no closer than the best distance.
            const std::array<double, 4>& ball = balls_[static_cast<size_t>(t)];
            const Point offset = sub(position, {ball[0], ball[1], ball[2]});
            const double reach_ball = best.distance + ball[3];
            if (dot(offset, offset) >= reach_ball * reach_ball) continue;
            const Triangle& f = triangles_[static_cast<size_t>(t)];
            std::array<double, 3> weights{};
            const Point q = closest_point(position, points_[static_cast<size_t>(f[0])],
                                          points_[static_cast<size_t>(f[1])],
                                          points_[static_cast<size_t>(f[2])], weights);
            const double distance = norm(sub(position, q));
            if (distance < best.distance) best = {t, q, weights, distance};
          }
        }
      }
  }
  return best;
}

Point MeshSource::normal(const Nearest& at) const {
  const Triangle& f = triangles_[static_cast<size_t>(at.triangle)];
  Point n{0, 0, 0};
  for (size_t k = 0; k < 3; ++k)
    for (int d = 0; d < 3; ++d)
      n[d] += at.barycentric[k] * vertex_normals_[static_cast<size_t>(f[k])][d];
  const double length = norm(n);
  if (!(length > 0)) return face_normals_[static_cast<size_t>(at.triangle)];
  return {n[0] / length, n[1] / length, n[2] / length};
}

std::array<double, 2> MeshSource::curvatures(const Nearest& at) const {
  const Triangle& f = triangles_[static_cast<size_t>(at.triangle)];
  std::array<double, 2> kappa{0, 0};
  for (size_t k = 0; k < 3; ++k)
    for (size_t i = 0; i < 2; ++i)
      kappa[i] += at.barycentric[k] * vertex_curvatures_[static_cast<size_t>(f[k])][i];
  return kappa;
}

bool MeshSource::at_vertex(int64_t v, Nearest& out) const {
  if (v < 0 || static_cast<size_t>(v) >= vertex_corners_.size()) return false;
  const int64_t corner = vertex_corners_[static_cast<size_t>(v)];
  if (corner < 0 || !(norm(vertex_normals_[static_cast<size_t>(v)]) > 0)) return false;
  out.triangle = corner / 3;
  out.point = points_[static_cast<size_t>(v)];
  out.barycentric = {0, 0, 0};
  out.barycentric[static_cast<size_t>(corner % 3)] = 1;
  out.distance = 0;
  return true;
}

namespace {
constexpr double kUnit = 0x1p-53;  // unit roundoff

int exact_orient(const Planar& a, const Planar& b, const Planar& c) {
  return phx::mc::orient2d(a.data(), b.data(), c.data());
}

// Shewchuk's floating stage of orient2d(a, b, c), det[a - c, b - c], and a
// bound on the magnitude of its error. (Shewchuk's stage-A constant
// (3 + 16u)u is a sign threshold: it moves the final subtraction's u |det|
// to the other side. As a magnitude bound the constant is (4 + 32u)u.)
struct Orientation {
  double det;
  double error;
};
Orientation orientation(const Planar& a, const Planar& b, const Planar& c) {
  const double left = (a[0] - c[0]) * (b[1] - c[1]);
  const double right = (a[1] - c[1]) * (b[0] - c[0]);
  return {left - right, (4.0 + 32.0 * kUnit) * kUnit * (std::abs(left) + std::abs(right))};
}

// Whether an edge direction of `from` separates the two planar triangles by
// more than `margin` (far above the rounding of the projections compared).
bool separated(const std::array<Planar, 3>& from, const std::array<Planar, 3>& other,
               double margin) {
  for (size_t k = 0; k < 3; ++k) {
    const Planar& p = from[k];
    const Planar& r = from[(k + 1) % 3];
    const Planar axis{r[1] - p[1], p[0] - r[0]};
    const double width = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1]);
    if (!(width > 0)) continue;
    double from_low = std::numeric_limits<double>::infinity(), from_high = -from_low;
    double other_low = from_low, other_high = -from_low;
    for (size_t j = 0; j < 3; ++j) {
      const double s = axis[0] * from[j][0] + axis[1] * from[j][1];
      const double t = axis[0] * other[j][0] + axis[1] * other[j][1];
      from_low = std::min(from_low, s);
      from_high = std::max(from_high, s);
      other_low = std::min(other_low, t);
      other_high = std::max(other_high, t);
    }
    if (other_low > from_high + margin * width || other_high < from_low - margin * width)
      return true;
  }
  return false;
}

// The affine map sending planar corners p to spatial corners P,
// X(x) = P[0] + J (x - p[0]), with |J|_F, the planar conditioning
// |M|_F^2 / |det M| of M = [p1 - p0, p2 - p0] (outward, from the
// determinant's certified lower bound), and the relative error of the
// computed determinant (by which every computed column may be off). Nothing
// when the error bound does not keep the determinant at least twice its
// error away from 0, or a coefficient is not finite.
struct Affine {
  Planar origin;
  Point base;
  std::array<Point, 2> columns;
  double norm;
  double condition;
  double relative;
};
std::optional<Affine> affine(const std::array<Planar, 3>& p, const std::array<Point, 3>& space) {
  const double m00 = p[1][0] - p[0][0], m01 = p[2][0] - p[0][0];
  const double m10 = p[1][1] - p[0][1], m11 = p[2][1] - p[0][1];
  // det[p1 - p0, p2 - p0], with the rounding of the differences included.
  const Orientation determinant = orientation(p[1], p[2], p[0]);
  const double det = determinant.det;
  if (!(std::abs(det) > 2 * determinant.error)) return std::nullopt;
  const double lower = std::abs(det) - determinant.error;
  const Point first = sub(space[1], space[0]), second = sub(space[2], space[0]);
  Affine map{p[0], space[0], {}, 0, 0, 0};
  for (size_t d = 0; d < 3; ++d) {
    map.columns[0][d] = (first[d] * m11 - second[d] * m10) / det;
    map.columns[1][d] = (second[d] * m00 - first[d] * m01) / det;
  }
  map.norm = std::sqrt(dot(map.columns[0], map.columns[0]) + dot(map.columns[1], map.columns[1]));
  map.condition = (m00 * m00 + m01 * m01 + m10 * m10 + m11 * m11) / lower * (1 + 16 * kUnit);
  map.relative = determinant.error / lower * (1 + 16 * kUnit);
  if (!std::isfinite(map.norm) || !std::isfinite(map.condition) || !std::isfinite(map.relative))
    return std::nullopt;
  return map;
}
Point evaluate(const Affine& map, const Planar& x) {
  const double u = x[0] - map.origin[0], v = x[1] - map.origin[1];
  return {map.base[0] + map.columns[0][0] * u + map.columns[1][0] * v,
          map.base[1] + map.columns[0][1] * u + map.columns[1][1] * v,
          map.base[2] + map.columns[0][2] * u + map.columns[1][2] * v};
}
}  // namespace

double overlap_gap_bound(const std::array<Planar, 3>& own, const std::array<Point, 3>& own_space,
                         const std::array<Planar, 3>& other,
                         const std::array<Point, 3>& other_space) {
  const int own_side = exact_orient(own[0], own[1], own[2]);
  const int other_side = exact_orient(other[0], other[1], other[2]);
  if (own_side == 0) return std::numeric_limits<double>::infinity();
  // A degenerate `other` overlaps no area; its points lie in the closure of
  // its neighbors' overlaps.
  if (other_side == 0) return 0;
  const auto inside = [](const Planar& x, const std::array<Planar, 3>& corners, int side) {
    for (size_t k = 0; k < 3; ++k)
      if (exact_orient(corners[k], corners[(k + 1) % 3], x) * side < 0) return false;
    return true;
  };
  // The overlap's vertices, decided exactly before any map is formed: an
  // exactly empty overlap bounds nothing (0), whatever the maps' rounding.
  struct Vertex {
    Planar at;
    double offset;  // certified distance bound to the exact vertex
  };
  std::array<Vertex, 15> vertices{};
  size_t count = 0;
  double planar_scale = 0;
  for (size_t k = 0; k < 3; ++k)
    for (size_t d = 0; d < 2; ++d)
      planar_scale = std::max(planar_scale, std::max(std::abs(own[k][d]), std::abs(other[k][d])));
  for (size_t k = 0; k < 3; ++k) {
    if (inside(own[k], other, other_side)) vertices[count++] = {own[k], 0};
    if (inside(other[k], own, own_side)) vertices[count++] = {other[k], 0};
  }
  // Proper edge crossings (touching ones are corners inside the other): the
  // crossing's parameter along a segment is a / (a + b) for the two
  // orientations' magnitudes, enclosed from their error bounds (each
  // quotient and product rounded, 8u outward); of the two segments, the
  // tighter enclosure is kept. The point evaluated at the enclosure's middle
  // lies within half its width times the segment length of the exact
  // crossing, plus its own rounding (two operations per coordinate, at most
  // 16u planar_scale).
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j) {
      const Planar& p = own[i];
      const Planar& p_next = own[(i + 1) % 3];
      const Planar& q = other[j];
      const Planar& q_next = other[(j + 1) % 3];
      if (exact_orient(q, q_next, p) * exact_orient(q, q_next, p_next) >= 0 ||
          exact_orient(p, p_next, q) * exact_orient(p, p_next, q_next) >= 0)
        continue;
      Vertex crossing{{}, std::numeric_limits<double>::infinity()};
      for (const auto& [from, to, line, line_next] :
           {std::tuple{p, p_next, q, q_next}, std::tuple{q, q_next, p, p_next}}) {
        const Orientation a = orientation(line, line_next, from);
        const Orientation b = orientation(line, line_next, to);
        const double a_low = std::max(std::abs(a.det) - a.error, 0.0);
        const double a_high = std::abs(a.det) + a.error;
        const double b_low = std::max(std::abs(b.det) - b.error, 0.0);
        const double b_high = std::abs(b.det) + b.error;
        const double w_low = std::max(a_low / (a_low + b_high) * (1 - 8 * kUnit), 0.0);
        const double w_high = std::min(a_high / (a_high + b_low) * (1 + 8 * kUnit), 1.0);
        const double w = 0.5 * (w_low + w_high);
        const Planar x{from[0] + w * (to[0] - from[0]), from[1] + w * (to[1] - from[1])};
        const double length = std::sqrt((to[0] - from[0]) * (to[0] - from[0]) +
                                        (to[1] - from[1]) * (to[1] - from[1]));
        const double offset =
            0.5 * (w_high - w_low) * length * (1 + 8 * kUnit) + 16 * kUnit * planar_scale;
        if (offset < crossing.offset) crossing = {x, offset};
      }
      vertices[count++] = crossing;
    }
  if (count == 0) return 0;
  const std::optional<Affine> own_affine = affine(own, own_space);
  const std::optional<Affine> other_affine = affine(other, other_space);
  if (!own_affine || !other_affine) return std::numeric_limits<double>::infinity();
  const Affine& own_map = *own_affine;
  const Affine& other_map = *other_affine;
  // The rounding pad below is first order in u * condition; past 2^-10 its
  // neglected terms are no longer small, so the caller's height bound decides.
  if (kUnit * own_map.condition > 0x1p-10 || kUnit * other_map.condition > 0x1p-10)
    return std::numeric_limits<double>::infinity();
  double space_scale = 0;
  for (size_t k = 0; k < 3; ++k)
    for (size_t d = 0; d < 3; ++d)
      space_scale =
          std::max(space_scale, std::max(std::abs(own_space[k][d]), std::abs(other_space[k][d])));
  // First-order rounding of one map at a point x of its triangle, with
  // offset y = x - p0 (|y| <= the triangle's diameter <= 2 sqrt(2) times the
  // planar scale): y carries u relative; each computed column is
  // (first m11 - second m10) / det with first, second the spatial
  // differences (|.| <= 2 space_scale, u relative) and m the planar ones
  // (u relative): a numerator off by about 3u |first| |M|_F, so a column off
  // by 3u 2 space_scale |M|_F / |det| plus the determinant's relative error
  // times |J|; times |y| <= sqrt(2) |M|_F that is about 8.5u condition
  // space_scale plus relative |J| |y|. The products, sums and the base add
  // about 4u condition |J| planar_scale and 3u space_scale. Constants are
  // doubled (16, 8) and the relative term takes |y| <= 3 planar_scale.
  // Both maps' errors add; slope * offset carries a crossing's distance to
  // the exact vertex.
  const double slope = own_map.norm + other_map.norm;
  const double evaluation =
      16 * kUnit *
          (own_map.condition * (own_map.norm * planar_scale + space_scale) +
           other_map.condition * (other_map.norm * planar_scale + space_scale)) +
      8 * kUnit * space_scale +
      3 * (own_map.relative * own_map.norm + other_map.relative * other_map.norm) *
          planar_scale;
  double bound = 0;
  // A nonfinite value refuses (+inf), never drops out of the maximum.
  for (size_t k = 0; k < count; ++k) {
    const double value =
        norm(sub(evaluate(own_map, vertices[k].at), evaluate(other_map, vertices[k].at))) +
        slope * vertices[k].offset + evaluation;
    bound = std::isfinite(value) ? std::max(bound, value)
                                 : std::numeric_limits<double>::infinity();
  }
  return bound * (1 + 0x1p-40);
}

double MeshSource::triangle_bound(const Point& a, const Point& b, const Point& c,
                                  Projection projection) const {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  constexpr size_t kPatchLimit = 65536;
  if (triangles_.empty()) return kInfinity;
  const Point cross_ab_ac = cross(sub(b, a), sub(c, a));
  const double twice_area = norm(cross_ab_ac);
  if (!(twice_area > 0) || !std::isfinite(twice_area)) return kInfinity;
  const Nearest start = nearest({(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3,
                                 (a[2] + b[2] + c[2]) / 3});
  if (start.triangle < 0) return kInfinity;
  Point direction{};
  switch (projection) {
    case Projection::kOwnNormal:
      direction = {cross_ab_ac[0] / twice_area, cross_ab_ac[1] / twice_area,
                   cross_ab_ac[2] / twice_area};
      break;
    case Projection::kSourceNormal:
      direction = normal(start);
      break;
  }
  const Point helper = std::abs(direction[0]) < 0.6 ? Point{1, 0, 0} : Point{0, 1, 0};
  const Point across = cross(helper, direction);
  const double across_length = norm(across);
  const Point e1{across[0] / across_length, across[1] / across_length,
                 across[2] / across_length};
  const Point e2 = cross(direction, e1);
  double scale = scale_;
  for (const Point* p : {&a, &b, &c})
    for (const double x : *p) scale = std::max(scale, std::abs(x));
  // Projections are compared with `margin` (far above their rounding): a
  // patch triangle is excluded only when certainly apart from T's.
  const double margin = 0x1p-32 * scale;
  // T's projection, positively oriented (exactly), and T's heights along
  // `direction`.
  std::array<Point, 3> space{a, b, c};
  std::array<Planar, 3> corner{};
  std::array<double, 3> corner_height{};
  for (size_t k = 0; k < 3; ++k) {
    const Point d = sub(space[k], a);
    corner[k] = {dot(d, e1), dot(d, e2)};
    corner_height[k] = dot(d, direction);
  }
  bool in_domain = true;
  const auto admit = [&](const Planar& x) {
    in_domain = in_domain && phx::mc::coordinate_in_domain(x[0]) &&
                phx::mc::coordinate_in_domain(x[1]);
  };
  for (const Planar& x : corner) admit(x);
  if (!in_domain) return kInfinity;
  const int corner_side = exact_orient(corner[0], corner[1], corner[2]);
  if (corner_side == 0) return kInfinity;
  if (corner_side < 0) {
    std::swap(corner[1], corner[2]);
    std::swap(corner_height[1], corner_height[2]);
    std::swap(space[1], space[2]);
  }
  const double corner_low =
      std::min(std::min(corner_height[0], corner_height[1]), corner_height[2]);
  const double corner_high =
      std::max(std::max(corner_height[0], corner_height[1]), corner_height[2]);
  ++stamp_;
  // (u, v, height) of a source vertex in the frame, once per query.
  const auto lift = [&](int64_t v) -> const std::array<double, 3>& {
    const auto i = static_cast<size_t>(v);
    if (vertex_stamps_[i] != stamp_) {
      vertex_stamps_[i] = stamp_;
      const Point d = sub(points_[i], a);
      projected_[i] = {dot(d, e1), dot(d, e2), dot(d, direction)};
      admit({projected_[i][0], projected_[i][1]});
    }
    return projected_[i];
  };
  const auto planar = [&](int64_t t) {
    const Triangle& f = triangles_[static_cast<size_t>(t)];
    std::array<Planar, 3> q{};
    for (size_t k = 0; k < 3; ++k) {
      const auto& p = lift(f[k]);
      q[k] = {p[0], p[1]};
    }
    return q;
  };
  const auto meets = [&](int64_t t) {
    const auto q = planar(t);
    return !separated(corner, q, margin) && !separated(q, corner, margin);
  };
  // The patch: source triangles reached across edges whose projections come
  // within `margin` of T's. An edge on its boundary is held by a triangle
  // whose projection does not, so the boundary projects outside T's.
  patch_.clear();
  tested_[static_cast<size_t>(start.triangle)] = stamp_;
  if (!meets(start.triangle)) return kInfinity;
  included_[static_cast<size_t>(start.triangle)] = stamp_;
  patch_.push_back(start.triangle);
  for (size_t i = 0; i < patch_.size(); ++i)
    for (size_t k = 0; k < 3; ++k) {
      const int64_t u = twins_[3 * static_cast<size_t>(patch_[i]) + k] / 3;
      if (tested_[static_cast<size_t>(u)] == stamp_) continue;
      tested_[static_cast<size_t>(u)] = stamp_;
      if (!meets(u)) continue;
      if (patch_.size() == kPatchLimit) return kInfinity;
      included_[static_cast<size_t>(u)] = stamp_;
      patch_.push_back(u);
    }
  // Exact predicates need every planar coordinate in their domain.
  if (!in_domain) return kInfinity;
  // Each triangle maps affinely onto the triangle of its corners' planar
  // coordinates (exact binary64 values), so the patch is a piecewise-linear
  // map into the plane. Its boundary misses T's planar triangle, so the
  // winding number of the boundary is the same at every point of it (taken
  // at its first corner, with the exact orient2d and half-open crossings);
  // one means every point of T's triangle is the image of a patch point.
  const Planar& probe = corner[0];
  int64_t winding = 0;
  for (const int64_t t : patch_) {
    const Triangle& f = triangles_[static_cast<size_t>(t)];
    for (size_t k = 0; k < 3; ++k) {
      if (included_[static_cast<size_t>(twins_[3 * static_cast<size_t>(t) + k] / 3)] == stamp_)
        continue;
      const auto& p = lift(f[k]);
      const auto& q = lift(f[(k + 1) % 3]);
      const Planar from{p[0], p[1]}, to{q[0], q[1]};
      if (from[1] <= probe[1]) {
        if (to[1] > probe[1] && exact_orient(from, to, probe) > 0) ++winding;
      } else if (to[1] <= probe[1] && exact_orient(from, to, probe) < 0) {
        --winding;
      }
    }
  }
  if (winding != 1) return kInfinity;
  // A point x of T is X(y) for y in its planar triangle, and y is the image
  // Y(y) of a point of some patch triangle t: d(x, source) <= |X(y) - Y(y)|,
  // an affine vector's norm, convex over the overlap of the two planar
  // triangles, so it peaks at the overlap's vertices (overlap_gap_bound).
  // Through the frame, |X - Y| is also at most the gap between the heights,
  // each a convex combination of its corners' (computed along `direction`,
  // rounded within 64 units of `scale`; the projection's rounding reaches
  // about 48): that bounds overlaps too
  // ill-conditioned to evaluate.
  double bound = 0;
  for (const int64_t t : patch_) {
    const Triangle& f = triangles_[static_cast<size_t>(t)];
    double low = kInfinity, high = -kInfinity;
    for (size_t k = 0; k < 3; ++k) {
      low = std::min(low, lift(f[k])[2]);
      high = std::max(high, lift(f[k])[2]);
    }
    const double extreme =
        (std::max(high - corner_low, corner_high - low) + 64 * kUnit * scale) * (1 + 0x1p-40);
    if (extreme <= bound) continue;
    const std::array<Point, 3> source_space{points_[static_cast<size_t>(f[0])],
                                            points_[static_cast<size_t>(f[1])],
                                            points_[static_cast<size_t>(f[2])]};
    bound = std::max(bound, std::min(extreme, overlap_gap_bound(corner, space, planar(t),
                                                                source_space)));
  }
  return bound;
}

bool MeshSourceOracle::project(const Point& position, SurfacePoint& out) const {
  const MeshSource::Nearest at = source_.nearest(position);
  if (at.triangle < 0) return false;
  out.on = at.point;
  out.normal = source_.normal(at);
  const auto [largest, smallest] = source_.curvatures(at);
  out.curvature_first = largest;
  out.curvature_second = smallest;
  return true;
}

void MeshSourceOracle::project_all(const std::vector<Point>& positions,
                                   std::vector<SurfacePoint>& out,
                                   std::vector<uint8_t>& found) const {
  out.assign(positions.size(), SurfacePoint{});
  found.assign(positions.size(), 0);
  // The source's own vertices (a surface loaded from the source itself)
  // project onto themselves; anything else is searched.
  const std::vector<Point>& own = source_.points();
  const bool aligned = positions.size() == own.size();
  for (size_t i = 0; i < positions.size(); ++i) {
    MeshSource::Nearest at;
    if (aligned && positions[i] == own[i] && source_.at_vertex(static_cast<int64_t>(i), at)) {
      out[i].on = at.point;
      out[i].normal = source_.normal(at);
      const auto [largest, smallest] = source_.curvatures(at);
      out[i].curvature_first = largest;
      out[i].curvature_second = smallest;
      found[i] = 1;
      continue;
    }
    found[i] = project(positions[i], out[i]) ? 1 : 0;
  }
}

void MeshSourceOracle::distances(const std::vector<Point>& positions,
                                 std::vector<double>& out) const {
  out.resize(positions.size());
  for (size_t i = 0; i < positions.size(); ++i) out[i] = source_.nearest(positions[i]).distance;
}

namespace {
// Copies flat (vertices, triangles) arrays; false for counts or pointers that
// cannot describe a surface.
bool read_surface(const double* vertices, int64_t vertex_count, const int64_t* triangles,
                  int64_t triangle_count, std::vector<Point>& points,
                  std::vector<Triangle>& faces) {
  if (vertex_count < 3 || triangle_count < 1 || vertices == nullptr || triangles == nullptr ||
      !phx::mc::addressable(vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(triangle_count, 3, sizeof(int64_t)))
    return false;
  points.resize(static_cast<size_t>(vertex_count));
  for (size_t v = 0; v < points.size(); ++v)
    for (size_t d = 0; d < 3; ++d) points[v][d] = vertices[3 * v + d];
  faces.resize(static_cast<size_t>(triangle_count));
  for (size_t t = 0; t < faces.size(); ++t)
    for (size_t k = 0; k < 3; ++k) faces[t][k] = triangles[3 * t + k];
  return true;
}

bool valid_targets(double tolerance, double edge_length, double minimum_angle_degrees) {
  return tolerance > 0 && std::isfinite(tolerance) && edge_length > 0 &&
         std::isfinite(edge_length) && minimum_angle_degrees > 0 && minimum_angle_degrees < 60;
}

// Construction-only denoising of a mesh source. Noise shows in the normal
// component l = L(x) . n of the uniform Laplacian (its tangential part only
// reflects irregular vertex spacing): on a smooth surface l follows the
// curvature and varies slowly (|L(l)| is far below |l|), while vertex noise
// of size s gives |L(l)| of order s. When the mean |L(l)| exceeds 2% of the
// tolerance, Taubin smoothing (lambda 0.5, mu -0.53, which does not shrink)
// runs (h / 2e)^2 passes, at most 32 (a radius of about h / 2), and each
// vertex keeps within tolerance / 2 of its input position. Returns the
// smoothed vertices, or nothing when the source is smooth at this tolerance.
std::vector<Point> denoised(const std::vector<Point>& points,
                            const std::vector<Triangle>& triangles, double edge_length,
                            double tolerance) {
  const size_t n = points.size();
  // Ring neighbors: the heads of each vertex's outgoing edges.
  std::vector<int64_t> start(n + 1, 0);
  for (const Triangle& f : triangles)
    for (size_t k = 0; k < 3; ++k) {
      if (f[k] < 0 || static_cast<size_t>(f[k]) >= n) return {};
      ++start[static_cast<size_t>(f[k]) + 1];
    }
  for (size_t v = 0; v < n; ++v) start[v + 1] += start[v];
  std::vector<int64_t> ring(3 * triangles.size()), fill(start.begin(), start.end() - 1);
  for (const Triangle& f : triangles)
    for (size_t k = 0; k < 3; ++k) ring[static_cast<size_t>(fill[static_cast<size_t>(f[k])]++)] = f[(k + 1) % 3];
  double edges = 0;
  for (size_t v = 0; v < n; ++v)
    for (int64_t s = start[v]; s < start[v + 1]; ++s)
      edges += norm(sub(points[static_cast<size_t>(ring[static_cast<size_t>(s)])], points[v]));
  const double mean_edge = edges / static_cast<double>(ring.size());
  const auto laplacian = [&](const std::vector<Point>& x, std::vector<Point>& out) {
    out.assign(n, Point{0, 0, 0});
    for (size_t v = 0; v < n; ++v) {
      const int64_t count = start[v + 1] - start[v];
      if (count == 0) continue;
      Point sum{0, 0, 0};
      for (int64_t s = start[v]; s < start[v + 1]; ++s)
        for (size_t d = 0; d < 3; ++d) sum[d] += x[static_cast<size_t>(ring[static_cast<size_t>(s)])][d];
      for (size_t d = 0; d < 3; ++d) out[v][d] = sum[d] / static_cast<double>(count) - x[v][d];
    }
  };
  // Area-weighted vertex normals and the normal component of L(x).
  std::vector<Point> normals(n, Point{0, 0, 0}), first, step;
  for (const Triangle& f : triangles) {
    const Point& a = points[static_cast<size_t>(f[0])];
    const Point area = cross(sub(points[static_cast<size_t>(f[1])], a),
                             sub(points[static_cast<size_t>(f[2])], a));
    for (const int64_t v : f)
      for (size_t d = 0; d < 3; ++d) normals[static_cast<size_t>(v)][d] += area[d];
  }
  laplacian(points, first);
  std::vector<Point> along(n, Point{0, 0, 0});
  for (size_t v = 0; v < n; ++v) {
    const double length = norm(normals[v]);
    if (length > 0) along[v][0] = dot(first[v], normals[v]) / length;
  }
  laplacian(along, step);
  double rough = 0;
  for (const Point& p : step) rough += std::abs(p[0]);
  rough /= static_cast<double>(n);
  if (!(rough > 0.02 * tolerance) || !(mean_edge > 0)) return {};
  const double ratio = edge_length / (2 * mean_edge);
  const int passes = static_cast<int>(std::clamp(std::ceil(ratio * ratio), 1.0, 32.0));
  std::vector<Point> x = points;
  for (int pass = 0; pass < passes; ++pass)
    for (const double weight : {0.5, -0.53}) {
      laplacian(x, step);
      for (size_t v = 0; v < n; ++v)
        for (size_t d = 0; d < 3; ++d) x[v][d] += weight * step[v][d];
    }
  const double limit = tolerance / 2;
  for (size_t v = 0; v < n; ++v) {
    const Point moved = sub(x[v], points[v]);
    const double length = norm(moved);
    if (length > limit)
      for (size_t d = 0; d < 3; ++d) x[v][d] = points[v][d] + moved[d] * (limit / length);
  }
  return x;
}

// Passes after the first: local repairs for source triangles left farther
// than the tolerance from a certified surface (such as a noise outlier the
// surface smooths over), and angle relaxation after any repair.
constexpr int kRepairPasses = 3;

// Source -> surface: bounds every source triangle against the surface
// (closed, embedded and nondegenerate, as certified), projecting along the
// surface's normals, since a source triangle may be rougher than the surface.
// `largest` receives the largest bound, `refused` the triangles above the
// tolerance, and `flagged` (one flag per surface triangle) the surface
// triangles nearest the corners of each refused one.
int32_t reverse_bound(const MeshSource& source, const std::vector<Point>& points,
                      const std::vector<Triangle>& triangles, double tolerance, double& largest,
                      int64_t& refused, std::vector<uint8_t>& flagged) {
  MeshSource surface;
  if (surface.bind(points, triangles) != PHX_MC_OK) return PHX_MC_INTERNAL_ERROR;
  largest = 0;
  refused = 0;
  flagged.assign(triangles.size(), 0);
  for (const Triangle& f : source.triangles()) {
    const double bound = surface.triangle_bound(source.points()[static_cast<size_t>(f[0])],
                                                source.points()[static_cast<size_t>(f[1])],
                                                source.points()[static_cast<size_t>(f[2])],
                                                Projection::kSourceNormal);
    largest = std::max(largest, bound);
    if (bound <= tolerance) continue;
    ++refused;
    for (const int64_t v : f)
      flagged[static_cast<size_t>(
          surface.nearest(source.points()[static_cast<size_t>(v)]).triangle)] = 1;
  }
  return PHX_MC_OK;
}
}  // namespace

}  // namespace phx::mc::remesh

extern "C" PHX_MC_API int32_t phx_mc_mesh_remesh(const phx_mc_mesh_remesh_request* request) {
  if (request == nullptr || request->result == nullptr || request->counters == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  *request->result = nullptr;
  return phx::mc::guarded([&]() -> int32_t {
    using namespace phx::mc::remesh;
    std::vector<Point> points;
    std::vector<Triangle> triangles;
    if (!read_surface(request->vertices, request->vertex_count, request->triangles,
                      request->triangle_count, points, triangles) ||
        !valid_targets(request->tolerance, request->edge_length,
                       request->minimum_angle_degrees))
      return PHX_MC_INVALID_ARGUMENT;
    // Construction projects onto the denoised source when the input is
    // rough at this tolerance (or onto the input when smoothing leaves a
    // degenerate triangle); certification is against the input itself.
    const std::vector<Point> smoothed =
        denoised(points, triangles, request->edge_length, request->tolerance);
    MeshSource source;
    if (smoothed.empty() || source.bind(smoothed, triangles) != PHX_MC_OK) {
      const int32_t bound = source.bind(points, triangles);
      if (bound != PHX_MC_OK) return bound;
    }
    const MeshSourceOracle oracle(source);
    AdaptCounters adapted;
    // The source holds these very triangles, so its manifold check stands.
    const int32_t status = adapt_surface(points, triangles, oracle,
                                         AdaptTargets{request->edge_length, request->tolerance,
                                                      request->minimum_angle_degrees, false},
                                         &source.corner_twins(), adapted);
    if (status != PHX_MC_OK) return status;
    request->counters[0] = adapted.splits;
    request->counters[1] = adapted.collapses;
    request->counters[2] = adapted.flips;
    request->counters[3] = adapted.relocations;
    request->counters[4] = adapted.below_angle;
    request->counters[5] = adapted.above_tolerance;
    auto owned = std::make_unique<SurfaceResult>();
    owned->points = std::move(points);
    owned->triangles = std::move(triangles);
    *request->result = owned.release();
    return PHX_MC_OK;
  });
}

extern "C" PHX_MC_API int32_t phx_mc_mesh_certify(const phx_mc_mesh_certify_request* request) {
  if (request == nullptr || request->result == nullptr || request->distances == nullptr ||
      request->counters == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  *request->result = nullptr;
  return phx::mc::guarded([&]() -> int32_t {
    using namespace phx::mc::remesh;
    std::vector<Point> source_points, points;
    std::vector<Triangle> source_triangles, triangles;
    if (!read_surface(request->source_vertices, request->source_vertex_count,
                      request->source_triangles, request->source_triangle_count, source_points,
                      source_triangles) ||
        !read_surface(request->vertices, request->vertex_count, request->triangles,
                      request->triangle_count, points, triangles) ||
        !valid_targets(request->tolerance, request->edge_length,
                       request->minimum_angle_degrees))
      return PHX_MC_INVALID_ARGUMENT;
    MeshSource source;
    const int32_t bound = source.bind(std::move(source_points), std::move(source_triangles));
    if (bound != PHX_MC_OK) return bound;
    const MeshSourceOracle oracle(source);
    const MeshSourceCertifier certifier(source, request->tolerance);
    const AdaptTargets targets{request->edge_length, request->tolerance,
                               request->minimum_angle_degrees, false};
    std::fill(request->counters, request->counters + 6, 0);
    std::vector<uint8_t> flagged;
    // The last surface certified both ways, with its distances: relaxation
    // after it only restores angles (a target, not part of the certificate),
    // so when a later pass fails, this surface is returned.
    struct Certified {
      std::vector<Point> points;
      std::vector<Triangle> triangles;
      std::array<double, 2> distances;
    };
    std::optional<Certified> certified;
    const auto finish = [&](std::vector<Point>& out_points,
                            std::vector<Triangle>& out_triangles) -> int32_t {
      auto owned = std::make_unique<SurfaceResult>();
      owned->points = std::move(out_points);
      owned->triangles = std::move(out_triangles);
      *request->result = owned.release();
      return PHX_MC_OK;
    };
    // A failed pass after a certified surface: report that surface (its
    // distances, no source triangles refused); counters keep every pass.
    const auto fallback = [&](int32_t status) -> int32_t {
      if (!certified) return status;
      request->distances[0] = certified->distances[0];
      request->distances[1] = certified->distances[1];
      request->counters[4] = 0;
      return finish(certified->points, certified->triangles);
    };
    for (int pass = 0;; ++pass) {
      CertifyCounters counters;
      const int32_t status =
          certify_surface(points, triangles, oracle, targets, certifier, CarrierEvidence{},
                          nullptr, false, counters);
      if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
      request->distances[0] = counters.distance;
      request->distances[1] = std::numeric_limits<double>::infinity();
      request->counters[0] += counters.refused_fidelity;
      request->counters[1] += counters.refused_embedding;
      request->counters[2] += counters.repair_splits;
      request->counters[3] += counters.rounds;
      if (status != PHX_MC_OK) return fallback(status);
      int64_t refused = 0;
      const int32_t reverse = reverse_bound(source, points, triangles, request->tolerance,
                                            request->distances[1], refused, flagged);
      if (reverse != PHX_MC_OK) return reverse;
      request->counters[4] = refused;
      int64_t relocations = 0;
      if (refused == 0) {
        // Certified both ways. Splits that repaired surface refusals may
        // have left angles below the minimum: relax them and decide again,
        // unless no pass is left (the surface stays certified as it is).
        if (counters.repair_splits == 0 || pass == kRepairPasses) break;
        certified = Certified{points, triangles, {request->distances[0], request->distances[1]}};
        const int32_t relaxed = relax_surface(points, triangles, oracle, targets, relocations);
        if (relaxed != PHX_MC_OK) return relaxed;
        request->counters[5] += relocations;
        if (relocations == 0) break;
        continue;
      }
      if (pass == kRepairPasses) return fallback(PHX_MC_UNRESOLVED);
      // Split the surface triangles nearest the refused source triangles,
      // then restore angles (every triangle below the minimum angle); both
      // directions are decided again.
      int64_t splits = 0;
      int32_t repaired = repair_surface(points, triangles, oracle, targets, flagged, splits);
      if (repaired != PHX_MC_OK) return repaired;
      request->counters[2] += splits;
      if (splits == 0) return fallback(PHX_MC_UNRESOLVED);
      repaired = relax_surface(points, triangles, oracle, targets, relocations);
      if (repaired != PHX_MC_OK) return repaired;
      request->counters[5] += relocations;
    }
    return finish(points, triangles);
  });
}
