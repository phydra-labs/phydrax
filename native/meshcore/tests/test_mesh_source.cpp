// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// A triangle mesh as a remesh source: nearest points, the two-way distance
// certificate, and the remesh / certify entry points, each against
// brute-force references written independently here.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "check.hpp"
#include "mesh_source.hpp"
#include "phydrax_meshcore.h"

namespace {
using phx::mc::remesh::MeshSource;
using phx::mc::remesh::Point;
using phx::mc::remesh::Triangle;

Point sub(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Point lerp(const Point& a, const Point& b, double t) {
  return {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
}

// Distance from p to the closed segment ab.
double segment_distance(const Point& p, const Point& a, const Point& b) {
  const Point ab = sub(b, a);
  const double t = std::clamp(dot(sub(p, a), ab) / dot(ab, ab), 0.0, 1.0);
  const Point d = sub(p, lerp(a, b, t));
  return std::sqrt(dot(d, d));
}

// Distance from p to the closed triangle: the plane distance when the foot
// is inside (signed sub-areas agree), else the nearest edge.
double triangle_distance(const Point& p, const Point& a, const Point& b, const Point& c) {
  const Point normal = cross(sub(b, a), sub(c, a));
  const double height = dot(sub(p, a), normal) / std::sqrt(dot(normal, normal));
  const Point unit = {normal[0] / std::sqrt(dot(normal, normal)),
                      normal[1] / std::sqrt(dot(normal, normal)),
                      normal[2] / std::sqrt(dot(normal, normal))};
  const Point foot = {p[0] - height * unit[0], p[1] - height * unit[1], p[2] - height * unit[2]};
  const bool inside = dot(cross(sub(b, a), sub(foot, a)), normal) >= 0 &&
                      dot(cross(sub(c, b), sub(foot, b)), normal) >= 0 &&
                      dot(cross(sub(a, c), sub(foot, c)), normal) >= 0;
  if (inside) return std::abs(height);
  return std::min({segment_distance(p, a, b), segment_distance(p, b, c),
                   segment_distance(p, c, a)});
}

double mesh_distance(const Point& p, const std::vector<Point>& points,
                     const std::vector<Triangle>& triangles) {
  double best = std::numeric_limits<double>::infinity();
  for (const Triangle& f : triangles)
    best = std::min(best, triangle_distance(p, points[static_cast<size_t>(f[0])],
                                            points[static_cast<size_t>(f[1])],
                                            points[static_cast<size_t>(f[2])]));
  return best;
}

// Icosphere of radius `radius`, outward oriented.
void icosphere(int level, double radius, std::vector<Point>& points,
               std::vector<Triangle>& triangles) {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  points = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
            {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  triangles = {{0, 11, 5}, {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11},
               {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
               {3, 9, 4},  {3, 4, 2},  {3, 2, 6},  {3, 6, 8},  {3, 8, 9},
               {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},  {9, 8, 1}};
  const auto unit = [&](Point p) {
    const double length = std::sqrt(dot(p, p));
    return Point{radius * p[0] / length, radius * p[1] / length, radius * p[2] / length};
  };
  for (Point& p : points) p = unit(p);
  for (int round = 0; round < level; ++round) {
    std::map<std::pair<int64_t, int64_t>, int64_t> midpoints;
    const auto midpoint = [&](int64_t a, int64_t b) {
      const auto key = std::minmax(a, b);
      const auto found = midpoints.find(key);
      if (found != midpoints.end()) return found->second;
      points.push_back(unit(lerp(points[static_cast<size_t>(a)], points[static_cast<size_t>(b)], 0.5)));
      const auto id = static_cast<int64_t>(points.size()) - 1;
      midpoints.emplace(key, id);
      return id;
    };
    std::vector<Triangle> next;
    for (const Triangle& f : triangles) {
      const int64_t ab = midpoint(f[0], f[1]), bc = midpoint(f[1], f[2]), ca = midpoint(f[2], f[0]);
      next.push_back({f[0], ab, ca});
      next.push_back({f[1], bc, ab});
      next.push_back({f[2], ca, bc});
      next.push_back({ab, bc, ca});
    }
    triangles = std::move(next);
  }
}

// The nearest point agrees with brute force over every triangle.
void nearest_matches_brute_force() {
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  icosphere(3, 1.0, points, triangles);
  MeshSource source;
  PHX_CHECK(source.bind(points, triangles) == PHX_MC_OK);
  std::mt19937_64 generator(29);
  std::uniform_real_distribution<double> coordinate(-1.6, 1.6);
  for (int sample = 0; sample < 400; ++sample) {
    const Point p{coordinate(generator), coordinate(generator), coordinate(generator)};
    const MeshSource::Nearest at = source.nearest(p);
    const double expected = mesh_distance(p, points, triangles);
    PHX_CHECK(at.triangle >= 0);
    PHX_CHECK(std::abs(at.distance - expected) <= 1e-12);
    const Triangle& f = triangles[static_cast<size_t>(at.triangle)];
    Point rebuilt{0, 0, 0};
    for (size_t k = 0; k < 3; ++k)
      for (size_t d = 0; d < 3; ++d)
        rebuilt[d] += at.barycentric[k] * points[static_cast<size_t>(f[k])][d];
    PHX_CHECK(std::sqrt(dot(sub(rebuilt, at.point), sub(rebuilt, at.point))) <= 1e-12);
  }
}

// Four small spheres at the corners of a cube of side 1000: cells of the
// triangles' size over that box would number about 2^30 (the grid's per-axis
// cap); the grid coarsens to a bounded size instead, and nearest() stays
// exact.
void spread_source_binds_a_bounded_grid() {
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  for (const Point& center : {Point{0, 0, 0}, Point{1000, 0, 0}, Point{0, 1000, 0},
                              Point{0, 0, 1000}}) {
    std::vector<Point> sphere_points;
    std::vector<Triangle> sphere_triangles;
    icosphere(2, 1.0, sphere_points, sphere_triangles);
    const auto offset = static_cast<int64_t>(points.size());
    for (const Point& p : sphere_points)
      points.push_back({p[0] + center[0], p[1] + center[1], p[2] + center[2]});
    for (const Triangle& f : sphere_triangles)
      triangles.push_back({f[0] + offset, f[1] + offset, f[2] + offset});
  }
  MeshSource source;
  PHX_CHECK(source.bind(points, triangles) == PHX_MC_OK);
  std::mt19937_64 generator(31);
  std::uniform_real_distribution<double> coordinate(-1.5, 1.5);
  for (int sample = 0; sample < 100; ++sample) {
    const Point p{1000 * (sample % 2) + coordinate(generator), coordinate(generator),
                  1000 * (sample % 3 == 0) + coordinate(generator)};
    PHX_CHECK(std::abs(source.nearest(p).distance - mesh_distance(p, points, triangles)) <=
              1e-9);
  }
}

// Bound of triangle (a, b, c) and its largest sampled distance to the mesh.
std::pair<double, double> bound_and_sample(const MeshSource& source, const Point& a,
                                           const Point& b, const Point& c,
                                           const std::vector<Point>& points,
                                           const std::vector<Triangle>& triangles,
                                           phx::mc::remesh::Projection projection) {
  const double bound = source.triangle_bound(a, b, c, projection);
  double sampled = 0;
  constexpr int kSteps = 12;
  for (int i = 0; i <= kSteps; ++i)
    for (int j = 0; i + j <= kSteps; ++j) {
      const double u = static_cast<double>(i) / kSteps, v = static_cast<double>(j) / kSteps;
      const Point p{a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
                    a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]),
                    a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])};
      sampled = std::max(sampled, mesh_distance(p, points, triangles));
    }
  return {bound, sampled};
}

// Triangles near the source, tangent-ish, of several sizes and offsets: the
// bound is finite, never below a sampled distance, and within a small factor
// of it (an O(h^2) bound, not O(h)); a source triangle is bounded by its
// rounding margin.
void triangle_bound_is_sound_and_tight() {
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  icosphere(4, 1.0, points, triangles);
  MeshSource source;
  PHX_CHECK(source.bind(points, triangles) == PHX_MC_OK);
  std::mt19937_64 generator(71);
  std::normal_distribution<double> gaussian(0.0, 1.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int sample = 0; sample < 60; ++sample) {
    Point n{gaussian(generator), gaussian(generator), gaussian(generator)};
    const double length = std::sqrt(dot(n, n));
    n = {n[0] / length, n[1] / length, n[2] / length};
    const Point helper = std::abs(n[0]) < 0.9 ? Point{1, 0, 0} : Point{0, 1, 0};
    Point e1 = cross(n, helper);
    const double e1_length = std::sqrt(dot(e1, e1));
    e1 = {e1[0] / e1_length, e1[1] / e1_length, e1[2] / e1_length};
    const Point e2 = cross(n, e1);
    const double size = 0.02 + 0.2 * unit(generator);
    const double offset = 1.0 + 0.01 * (2 * unit(generator) - 1);
    std::array<Point, 3> corner{};
    for (size_t k = 0; k < 3; ++k) {
      const double angle = 2.0943951023931953 * static_cast<double>(k) + unit(generator);
      for (size_t d = 0; d < 3; ++d)
        corner[k][d] = offset * n[d] + size * (std::cos(angle) * e1[d] + std::sin(angle) * e2[d]);
    }
    for (const auto projection :
         {phx::mc::remesh::Projection::kOwnNormal, phx::mc::remesh::Projection::kSourceNormal}) {
      const auto [bound, sampled] = bound_and_sample(source, corner[0], corner[1], corner[2],
                                                     points, triangles, projection);
      PHX_CHECK(std::isfinite(bound));
      PHX_CHECK(bound >= sampled);
      PHX_CHECK(bound <= 1.5 * sampled + 1e-9);
    }
  }
  const Triangle& f = triangles[17];
  const double own = source.triangle_bound(
      points[static_cast<size_t>(f[0])], points[static_cast<size_t>(f[1])],
      points[static_cast<size_t>(f[2])], phx::mc::remesh::Projection::kOwnNormal);
  PHX_CHECK(own >= 0 && own <= 1e-9);
}

// No bound without coverage: a triangle standing across the source (its
// plane holds the normal), a degenerate triangle, and a triangle facing
// against the source projected along its own normal (the patch winds -1)
// are +inf. Projected along the source's normal, the reversed triangle is
// bounded, soundly.
void triangle_bound_refuses_without_coverage() {
  using phx::mc::remesh::Projection;
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  icosphere(3, 1.0, points, triangles);
  MeshSource source;
  PHX_CHECK(source.bind(points, triangles) == PHX_MC_OK);
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  PHX_CHECK(source.triangle_bound({0.0, 0.0, 0.9}, {0.05, 0.0, 1.1}, {-0.05, 0.0, 1.1},
                                  Projection::kOwnNormal) == kInfinity);
  PHX_CHECK(source.triangle_bound({0.0, 0.0, 1.0}, {0.1, 0.0, 1.0}, {0.2, 0.0, 1.0},
                                  Projection::kOwnNormal) == kInfinity);
  const Point a{0.05, 0.02, 1.0}, b{-0.04, 0.06, 0.995}, c{0.0, -0.06, 1.005};
  const bool forward = dot(cross(sub(b, a), sub(c, a)), a) > 0;
  const Point& second = forward ? c : b;
  const Point& third = forward ? b : c;  // (a, second, third) faces inward
  PHX_CHECK(source.triangle_bound(a, second, third, Projection::kOwnNormal) == kInfinity);
  const auto [bound, sampled] =
      bound_and_sample(source, a, second, third, points, triangles, Projection::kSourceNormal);
  PHX_CHECK(std::isfinite(bound) && bound >= sampled);
}

// A triangle hanging over the rim of a thin slab: half of it lies over no
// source at all (0.5 from the slab at its far corner) while the top face is
// 0.01 below its near half. The projected patch reaches the whole closed slab
// (top, sides, bottom), so its boundary is empty and the winding is 0: the
// triangle is refused. A coverage test that trusted the top face alone
// would state about 0.01 for points half a unit away.
void triangle_bound_refuses_a_partly_uncovered_triangle() {
  using phx::mc::remesh::Projection;
  const std::vector<Point> points{{-1, -1, -0.01}, {1, -1, -0.01}, {1, 1, -0.01}, {-1, 1, -0.01},
                                  {-1, -1, 0.01},  {1, -1, 0.01},  {1, 1, 0.01},  {-1, 1, 0.01}};
  const std::vector<Triangle> triangles{{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
                                        {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5},
                                        {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}};
  MeshSource source;
  PHX_CHECK(source.bind(points, triangles) == PHX_MC_OK);
  const Point a{0.5, -0.2, 0.02}, b{1.5, 0.0, 0.02}, c{0.5, 0.2, 0.02};
  for (const auto projection : {Projection::kOwnNormal, Projection::kSourceNormal}) {
    const auto [bound, sampled] = bound_and_sample(source, a, b, c, points, triangles, projection);
    PHX_CHECK(sampled > 0.4);
    PHX_CHECK(bound == std::numeric_limits<double>::infinity());
  }
}

// An overlap vertex where two edges cross at an angle near 2^-53: own's edge
// runs along y = x / 2 from (0, 0) to (1, 1/2); other's edge from
// q0 = (0.1, 0.05 - 2^-56) to q1 = (0.7, 0.35 + 2^-54), exactly 2^-56 below
// and 2^-54 above that line, so it crosses it at parameter 1/5 (x = 0.22),
// where other's height (1 at q0, 0 at q1) is 0.8; every other overlap vertex
// has a gap near 0. Floating orientations of these points are dominated by
// rounding, so a crossing placed from them can land anywhere on the edge;
// the bound must still reach the true 0.8.
void overlap_bound_encloses_a_near_parallel_crossing() {
  using phx::mc::remesh::Planar;
  const std::array<Planar, 3> own{Planar{0.0, 0.0}, Planar{1.0, 0.5}, Planar{0.0, 1.0}};
  const std::array<Point, 3> own_space{Point{0.0, 0.0, 0.0}, Point{1.0, 0.5, 0.0},
                                       Point{0.0, 1.0, 0.0}};
  const double low = 0.1, high = 0.7;
  const std::array<Planar, 3> other{Planar{low, low / 2 - 0x1p-56},
                                    Planar{high, high / 2 + 0x1p-54}, Planar{0.4, -1.0}};
  PHX_CHECK(other[0][1] < low / 2 && other[1][1] > high / 2);
  const std::array<Point, 3> other_space{Point{other[0][0], other[0][1], 1.0},
                                         Point{other[1][0], other[1][1], 0.0},
                                         Point{other[2][0], other[2][1], 0.0}};
  PHX_CHECK(phx::mc::remesh::overlap_gap_bound(own, own_space, other, other_space) >= 0.8);
}

// A rotated configuration found by a randomized search against exact
// rational clipping: the overlap's exact largest height gap is 0.467649...,
// while clipping with floating orientations and crossing parameters
// (s_x / (s_x - s_y)) gave 0.460594. The bound must reach the exact value.
void overlap_bound_covers_a_case_floating_clipping_missed() {
  using phx::mc::remesh::Planar;
  const std::array<Planar, 3> own{Planar{0x1.3d70a3d70a3d7p-2, 0x1.5c28f5c28f5c3p-3},
                                  Planar{-0x1.54d4016f73236p-1, 0x1.8e8a69d5e08f8p-2},
                                  Planar{0x1.73ead389c5704p-4, -0x1.9c8215ea546b1p-1}};
  const std::array<Planar, 3> other{Planar{-0x1.b3f79e1f14be0p-4, 0x1.0de23d9780fa3p-2},
                                    Planar{-0x1.71b0f267ee483p-2, 0x1.487414d076ef8p-2},
                                    Planar{-0x1.eb7079d8cbb4cp-4, 0x1.4aa186dbe8648p+0}};
  std::array<Point, 3> own_space{}, other_space{};
  for (size_t k = 0; k < 3; ++k) {
    own_space[k] = {own[k][0], own[k][1], 0.0};
    other_space[k] = {other[k][0], other[k][1], k == 0 ? 1.0 : 0.0};
  }
  PHX_CHECK(phx::mc::remesh::overlap_gap_bound(own, own_space, other, other_space) >=
            0.46764967849764);
}

// A planar triangle the exact orient2d proves nondegenerate (determinant
// e^2, e = 2^-52) whose binary64 determinant is 0 ((1 + e)(1 - e) rounds to
// 1): its affine map cannot be formed, and the bound must not drop it. Its
// third corner is 2 above the containing triangle at z = 0, so the exact
// gap reaches 2, in both roles.
void overlap_bound_refuses_an_uncertified_determinant() {
  using phx::mc::remesh::Planar;
  constexpr double e = 0x1p-52;
  const std::array<Planar, 3> thin{Planar{0.0, 0.0}, Planar{1.0, 1.0 - e},
                                   Planar{1.0 + e, 1.0}};
  const std::array<Point, 3> thin_space{Point{0.0, 0.0, 1.0}, Point{1.0, 1.0 - e, 1.0},
                                        Point{1.0 + e, 1.0, 2.0}};
  const std::array<Planar, 3> wide{Planar{-1.0, -1.0}, Planar{3.0, -1.0}, Planar{-1.0, 3.0}};
  const std::array<Point, 3> wide_space{Point{-1.0, -1.0, 0.0}, Point{3.0, -1.0, 0.0},
                                        Point{-1.0, 3.0, 0.0}};
  PHX_CHECK(phx::mc::remesh::overlap_gap_bound(wide, wide_space, thin, thin_space) >= 2.0);
  PHX_CHECK(phx::mc::remesh::overlap_gap_bound(thin, thin_space, wide, wide_space) >= 2.0);
}

// A sliver whose determinant is certified (2^-43, exact in binary64) but
// whose u * condition is about 5 * 2^-10: inside a wide triangle its map is
// refused (+inf, the first-order pad no longer bounds the rounding); moved
// exactly clear of the wide triangle the overlap is empty and bounds nothing.
void overlap_bound_refuses_an_ill_conditioned_map() {
  using phx::mc::remesh::Planar;
  constexpr double delta = 0x1p-44;
  const std::array<Planar, 3> sliver{Planar{-1.0, 0.0}, Planar{1.0, 0.0}, Planar{0.0, delta}};
  const std::array<Point, 3> sliver_space{Point{-1.0, 0.0, 0.0}, Point{1.0, 0.0, 0.0},
                                          Point{0.0, delta, 0.0}};
  const std::array<Planar, 3> wide{Planar{-3.0, -1.0}, Planar{3.0, -1.0}, Planar{0.0, 3.0}};
  const std::array<Point, 3> wide_space{Point{-3.0, -1.0, 0.0}, Point{3.0, -1.0, 0.0},
                                        Point{0.0, 3.0, 0.0}};
  PHX_CHECK(phx::mc::remesh::overlap_gap_bound(wide, wide_space, sliver, sliver_space) ==
            std::numeric_limits<double>::infinity());
  std::array<Planar, 3> clear = sliver;
  for (Planar& p : clear) p[0] += 10.0;
  PHX_CHECK(phx::mc::remesh::overlap_gap_bound(wide, wide_space, clear, sliver_space) == 0.0);
}

// A slab (|x|, |y| <= 1, |z| <= 0.01) whose top face holds such a sliver
// along y = 0, its corner at x = -1 lowered to z = 0.005. A flat triangle at
// z = 0.02 over |x| <= 0.5 crosses the sliver: its height gap (0.015) is
// the largest of the patch (the other top triangles stay within 0.01375 of
// it over its footprint), so the sliver is evaluated, its map is refused,
// and triangle_bound still returns the finite height-gap bound for it.
void triangle_bound_falls_back_to_the_height_gap() {
  using phx::mc::remesh::Projection;
  constexpr double t = 0.01, delta = 0x1p-44;
  const std::vector<Point> points{
      {-1, -1, t},  {1, -1, t},  {1, 1, t},    {-1, 1, t},   {-1, 0, t - 0.005}, {1, 0, t},
      {0, delta, t}, {-1, -1, -t}, {1, -1, -t}, {1, 1, -t}, {-1, 1, -t},       {-1, 0, -t},
      {1, 0, -t}};
  const std::vector<Triangle> triangles{
      {0, 1, 5},  {0, 5, 4},   {4, 5, 6},  {6, 5, 2},  {6, 2, 3},   {4, 6, 3},     // top
      {7, 12, 8}, {7, 11, 12}, {11, 9, 12}, {11, 10, 9},                          // bottom
      {0, 7, 8},  {0, 8, 1},                                                      // y = -1
      {1, 8, 12}, {1, 12, 5},  {5, 12, 9}, {5, 9, 2},                             // x = 1
      {2, 9, 10}, {2, 10, 3},                                                     // y = 1
      {3, 10, 11}, {3, 11, 4}, {4, 11, 7}, {4, 7, 0}};                            // x = -1
  MeshSource source;
  PHX_CHECK(source.bind(points, triangles) == PHX_MC_OK);
  const double bound = source.triangle_bound({-0.5, -0.3, 2 * t}, {0.5, -0.3, 2 * t},
                                             {0.0, 0.4, 2 * t}, Projection::kOwnNormal);
  PHX_CHECK(bound >= 1.5 * t && bound <= 1.501 * t);
}

// An open source or a nonfinite coordinate is refused at bind.
void bind_refuses_invalid_sources() {
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  icosphere(1, 1.0, points, triangles);
  MeshSource source;
  std::vector<Triangle> open = triangles;
  open.pop_back();
  PHX_CHECK(source.bind(points, open) == PHX_MC_INVALID_INPUT);
  std::vector<Point> nonfinite = points;
  nonfinite[3][1] = std::numeric_limits<double>::quiet_NaN();
  PHX_CHECK(source.bind(nonfinite, triangles) == PHX_MC_NONFINITE_INPUT);
}

struct Flat {
  std::vector<double> vertices;
  std::vector<int64_t> triangles;
};

Flat flatten(const std::vector<Point>& points, const std::vector<Triangle>& triangles) {
  Flat flat;
  for (const Point& p : points) flat.vertices.insert(flat.vertices.end(), p.begin(), p.end());
  for (const Triangle& f : triangles) flat.triangles.insert(flat.triangles.end(), f.begin(), f.end());
  return flat;
}

int32_t certify(const Flat& source, const Flat& surface, double tolerance, double edge,
                std::array<double, 2>& distances, std::array<int64_t, 6>& counters,
                Flat* certified) {
  void* handle = nullptr;
  const phx_mc_mesh_certify_request request{
      source.vertices.data(),  static_cast<int64_t>(source.vertices.size() / 3),
      source.triangles.data(), static_cast<int64_t>(source.triangles.size() / 3),
      surface.vertices.data(), static_cast<int64_t>(surface.vertices.size() / 3),
      surface.triangles.data(), static_cast<int64_t>(surface.triangles.size() / 3),
      tolerance,               edge,
      30.0,                    &handle,
      distances.data(),        counters.data()};
  const int32_t status = phx_mc_mesh_certify(&request);
  if (handle == nullptr) return status;
  std::array<int64_t, 2> sizes{};
  PHX_CHECK(phx_mc_surface_result_sizes(handle, sizes.data()) == PHX_MC_OK);
  certified->vertices.resize(static_cast<size_t>(3 * sizes[0]));
  certified->triangles.resize(static_cast<size_t>(3 * sizes[1]));
  PHX_CHECK(phx_mc_surface_result_arrays(handle, sizes[0], sizes[1], certified->vertices.data(),
                                         certified->triangles.data()) == PHX_MC_OK);
  phx_mc_surface_result_free(handle);
  return status;
}

// Remesh a polyhedral sphere through the C entry, certify both ways: the
// certified distances are within the tolerance and at least the sampled
// distances both ways. Against a source 2% larger the same surface is
// refused (UNRESOLVED, refusals counted, no result).
void remesh_then_certify_round_trip() {
  std::vector<Point> points;
  std::vector<Triangle> triangles;
  icosphere(4, 1.0, points, triangles);
  const Flat source = flatten(points, triangles);
  constexpr double kTolerance = 0.01, kEdge = 0.2;
  void* handle = nullptr;
  std::array<int64_t, 6> remesh_counters{};
  const phx_mc_mesh_remesh_request remesh{
      source.vertices.data(), static_cast<int64_t>(points.size()),
      source.triangles.data(), static_cast<int64_t>(triangles.size()),
      kTolerance, kEdge, 30.0, &handle, remesh_counters.data()};
  PHX_CHECK(phx_mc_mesh_remesh(&remesh) == PHX_MC_OK);
  std::array<int64_t, 2> sizes{};
  PHX_CHECK(phx_mc_surface_result_sizes(handle, sizes.data()) == PHX_MC_OK);
  Flat surface;
  surface.vertices.resize(static_cast<size_t>(3 * sizes[0]));
  surface.triangles.resize(static_cast<size_t>(3 * sizes[1]));
  PHX_CHECK(phx_mc_surface_result_arrays(handle, sizes[0], sizes[1], surface.vertices.data(),
                                         surface.triangles.data()) == PHX_MC_OK);
  phx_mc_surface_result_free(handle);
  PHX_CHECK(sizes[1] < static_cast<int64_t>(triangles.size()));

  std::array<double, 2> distances{};
  std::array<int64_t, 6> counters{};
  Flat certified;
  PHX_CHECK(certify(source, surface, kTolerance, kEdge, distances, counters, &certified) ==
            PHX_MC_OK);
  PHX_CHECK(distances[0] >= 0 && distances[0] <= kTolerance);
  PHX_CHECK(distances[1] >= 0 && distances[1] <= kTolerance);
  PHX_CHECK(counters[4] == 0);
  std::vector<Point> out_points(certified.vertices.size() / 3);
  for (size_t v = 0; v < out_points.size(); ++v)
    out_points[v] = {certified.vertices[3 * v], certified.vertices[3 * v + 1],
                     certified.vertices[3 * v + 2]};
  std::vector<Triangle> out_triangles(certified.triangles.size() / 3);
  for (size_t t = 0; t < out_triangles.size(); ++t)
    out_triangles[t] = {certified.triangles[3 * t], certified.triangles[3 * t + 1],
                        certified.triangles[3 * t + 2]};
  for (const Triangle& f : out_triangles)
    for (const double w : {0.0, 1.0 / 3.0, 0.5}) {
      const Point p = lerp(lerp(out_points[static_cast<size_t>(f[0])],
                                out_points[static_cast<size_t>(f[1])], w),
                           out_points[static_cast<size_t>(f[2])], w);
      PHX_CHECK(mesh_distance(p, points, triangles) <= distances[0]);
    }
  for (size_t v = 0; v < points.size(); v += 7)
    PHX_CHECK(mesh_distance(points[v], out_points, out_triangles) <= distances[1]);

  std::vector<Point> larger = points;
  for (Point& p : larger)
    for (double& x : p) x *= 1.02;
  Flat unused;
  PHX_CHECK(certify(flatten(larger, triangles), surface, kTolerance, kEdge, distances, counters,
                    &unused) == PHX_MC_UNRESOLVED);
  PHX_CHECK(counters[0] > 0);
  PHX_CHECK(unused.triangles.empty());
}

}  // namespace

int main() {
  nearest_matches_brute_force();
  spread_source_binds_a_bounded_grid();
  triangle_bound_is_sound_and_tight();
  triangle_bound_refuses_without_coverage();
  triangle_bound_refuses_a_partly_uncovered_triangle();
  overlap_bound_encloses_a_near_parallel_crossing();
  overlap_bound_covers_a_case_floating_clipping_missed();
  overlap_bound_refuses_an_uncertified_determinant();
  overlap_bound_refuses_an_ill_conditioned_map();
  triangle_bound_falls_back_to_the_height_gap();
  bind_refuses_invalid_sources();
  remesh_then_certify_round_trip();
  return phx::mc::test::finish("test_mesh_source");
}
