// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Source-agnostic surface remesher: the source-relative fold factor of the
// quality score and the quality pass's relocation candidates.
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
#include "dyadic.hpp"
#include "intersections.hpp"
#include "phydrax_meshcore.h"
#include "surface_remesh.hpp"

namespace {
using phx::mc::remesh::AdmissionKind;
using phx::mc::remesh::NormalCone;
using phx::mc::remesh::Point;
using phx::mc::remesh::RemeshCounters;
using phx::mc::remesh::RemeshTargets;
using phx::mc::remesh::Triangle;
using phx::mc::remesh::TriangleAdmission;
using phx::mc::remesh::VertexEvidence;

// Admits every triangle and reports one fixed source normal cone: isolates
// the score and topology rules.
class SourceCone final : public TriangleAdmission {
 public:
  explicit SourceCone(const NormalCone& source) : cone(source) {}
  AdmissionKind kind() const override { return AdmissionKind::kOffsetBand; }
  VertexEvidence evaluate(const Point&) const override { return {}; }
  bool admits(const Point&, const Point&, const Point&,
              const std::array<VertexEvidence, 3>&) const override {
    return true;
  }
  Point project(const Point& position) const override { return position; }
  NormalCone normal_cone(const Point&, const Point&, const Point&,
                         const std::array<VertexEvidence, 3>&) const override {
    return cone;
  }
  double normal_cone_floor(const Point&, const Point&, const Point&,
                           const std::array<VertexEvidence, 3>&) const override {
    return cone.half_angle;
  }
  NormalCone cone;
};

Point normal(const std::vector<Point>& points, const Triangle& t) {
  const Point& a = points[static_cast<size_t>(t[0])];
  const Point& b = points[static_cast<size_t>(t[1])];
  const Point& c = points[static_cast<size_t>(t[2])];
  const Point u{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const Point v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const Point n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
  const double length = std::hypot(n[0], n[1], n[2]);
  return {n[0] / length, n[1] / length, n[2] / length};
}

// Largest angle in degrees between the normals of two triangles sharing an edge.
double maximum_turn(const std::vector<Point>& points, const std::vector<Triangle>& triangles) {
  std::map<std::pair<int64_t, int64_t>, std::vector<size_t>> edges;
  for (size_t row = 0; row < triangles.size(); ++row)
    for (int k = 0; k < 3; ++k) {
      const int64_t a = triangles[row][k], b = triangles[row][(k + 1) % 3];
      edges[{std::min(a, b), std::max(a, b)}].push_back(row);
    }
  double turn = 0;
  for (const auto& [edge, rows] : edges) {
    if (rows.size() != 2) continue;
    const Point n = normal(points, triangles[rows[0]]);
    const Point m = normal(points, triangles[rows[1]]);
    const double cosine = std::clamp(n[0] * m[0] + n[1] * m[1] + n[2] * m[2], -1.0, 1.0);
    turn = std::max(turn, std::acos(cosine) * 180.0 / std::acos(-1.0));
  }
  return turn;
}

// A folded valence-3 cluster cut from a remeshed envelope (target edge 0.04):
// every triangle has angles of at least 24 degrees and edges far below the
// target, so shape and size alone score the fold almost full, and collapsing
// it lowers the minimum angle. Two of its faces turn 174 degrees against
// their neighbors, which bounds every tetrahedron on those edges to a 6 degree
// dihedral angle.
void folded_cluster() {
  const std::vector<Point> input{
      {-0x1.8e7701e5af140p-9, -0x1.f7b9cf6361500p-11, -0x1.1031c32d768a0p-8},
      {0x1.270d344f59120p-9, 0x1.270d344f59100p-9, -0x1.35bd147b25c00p-11},
      {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
      {0x1.baf29b33b86a0p-8, -0x1.4404d4584b900p-9, 0x1.3dd52d68d3700p-9},
      {0x1.0e87e202f2c00p-14, 0x1.bc491fddb6f00p-10, -0x1.840ec0e1ec400p-9},
      {0x1.83f73b385c000p-10, 0x1.4fcc567d938e0p-8, -0x1.7be911acdc360p-8},
      {-0x1.cb69ff4954dc0p-6, -0x1.05265f9162df0p-7, -0x1.1b1dd6b9350b0p-6},
      {-0x1.8135e4def9d08p-6, -0x1.54e9bb1ef2540p-6, -0x1.042e974eef240p-8},
      {-0x1.39429fc704098p-6, -0x1.12c763514b760p-7, -0x1.5c8b6f2321aa0p-7},
      {-0x1.8008df7210d00p-9, -0x1.7a12764aa9120p-7, 0x1.cbca84f7418c0p-9},
      {0x1.fc26941a76fd0p-7, -0x1.e52b78fa9b3c0p-8, 0x1.417a97e3f7720p-7},
      {-0x1.1d7c4f7511ba0p-6, 0x1.e48903f0a8c80p-8, -0x1.82ef6f6de8fc0p-6},
      {-0x1.02cc971ce7968p-6, 0x1.3b5b62af916e0p-6, -0x1.32fb942b02cf0p-5},
      {0x1.f723b646155a0p-8, 0x1.aeba5cae26d00p-6, -0x1.b8329b316eba0p-6},
      {0x1.36a0fc8fe79a6p-5, 0x1.515a4693e9c38p-5, -0x1.e8fbd4c3c3df0p-6},
      {-0x1.d48496e1d0a80p-8, -0x1.2c7cfbc8a5638p-5, 0x1.3c79162bbda98p-6},
      {0x1.2cb02a396e600p-6, -0x1.6c4d1b350d100p-6, 0x1.84a3b6ab0e698p-6},
      {0x1.b0fc53c7de8e6p-5, 0x1.39a5e6980abe8p-6, 0x1.7300274df2280p-10},
      {0x1.631adffac3098p-7, 0x1.22fa3ceb80450p-7, -0x1.bfe1b9ccccac0p-9},
      {0x1.45bb78b3dc2fep-5, -0x1.3a4cded54e340p-9, 0x1.0017690ce6720p-6},
      {-0x1.e93b24f5a4780p-9, 0x1.168bd3e1a3740p-7, -0x1.f70f03524e9e0p-7},
  };
  const std::vector<Triangle> triangles{
      {0, 1, 2},
      {0, 3, 1},
      {2, 1, 4},
      {0, 2, 4},
      {5, 4, 1},
      {0, 4, 5},
      {6, 7, 8},
      {3, 0, 9},
      {0, 8, 9},
      {7, 9, 8},
      {9, 10, 3},
      {6, 8, 11},
      {7, 15, 9},
      {9, 15, 16},
      {9, 16, 10},
      {5, 1, 18},
      {3, 18, 1},
      {18, 3, 10},
      {14, 13, 18},
      {14, 18, 17},
      {10, 16, 19},
      {18, 10, 19},
      {18, 19, 17},
      {0, 5, 20},
      {0, 20, 8},
      {11, 8, 20},
      {20, 5, 18},
      {13, 20, 18},
      {20, 13, 12},
      {20, 12, 11},
  };
  PHX_CHECK(maximum_turn(input, triangles) > 170);
  // A source that admits any normal (open cones) explains every turn: the
  // fold stays, as a genuine crease would.
  {
    const SourceCone admission(NormalCone{});
    std::vector<Point> points = input;
    std::vector<Triangle> remeshed = triangles;
    RemeshCounters counters;
    PHX_CHECK(phx::mc::remesh::remesh_surface(points, remeshed, admission, {0.04, 30},
                                             counters) == PHX_MC_OK);
    PHX_CHECK(maximum_turn(points, remeshed) > 170);
  }
  // A smooth source (normals within 30 degrees of the patch's mean normal)
  // admits turns up to 60 degrees: the 174 degree fold is unmet and removed.
  Point mean{0, 0, 0};
  for (const Triangle& t : triangles) {
    const Point& a = input[static_cast<size_t>(t[0])];
    const Point& b = input[static_cast<size_t>(t[1])];
    const Point& c = input[static_cast<size_t>(t[2])];
    const Point u{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const Point v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    mean[0] += u[1] * v[2] - u[2] * v[1];
    mean[1] += u[2] * v[0] - u[0] * v[2];
    mean[2] += u[0] * v[1] - u[1] * v[0];
  }
  const double norm = std::hypot(mean[0], mean[1], mean[2]);
  const double half = 30 * std::acos(-1.0) / 180;
  const SourceCone admission(NormalCone{{mean[0] / norm, mean[1] / norm, mean[2] / norm}, half});
  PHX_CHECK(phx::mc::remesh::cone_separation(admission.cone, admission.cone) == 2 * half);
  std::vector<Point> points = input;
  std::vector<Triangle> remeshed = triangles;
  RemeshCounters counters;
  PHX_CHECK(phx::mc::remesh::remesh_surface(points, remeshed, admission, {0.04, 30}, counters) ==
            PHX_MC_OK);
  PHX_CHECK(maximum_turn(points, remeshed) <= 60);
}

// Admits a triangle whose corners lie within `width` of the unit sphere and
// projects onto the sphere; open cones (no fold penalty).
class SphereBand final : public TriangleAdmission {
 public:
  explicit SphereBand(double band) : width(band) {}
  AdmissionKind kind() const override { return AdmissionKind::kOffsetBand; }
  VertexEvidence evaluate(const Point& p) const override {
    const double radius = std::hypot(p[0], p[1], p[2]);
    return {radius - 1, radius - 1};
  }
  bool admits(const Point&, const Point&, const Point&,
              const std::array<VertexEvidence, 3>& evidence) const override {
    for (const VertexEvidence& corner : evidence)
      if (!(std::abs(corner.lower) <= width)) return false;
    return true;
  }
  Point project(const Point& p) const override {
    const double radius = std::hypot(p[0], p[1], p[2]);
    return {p[0] / radius, p[1] / radius, p[2] / radius};
  }
  NormalCone normal_cone(const Point&, const Point&, const Point&,
                         const std::array<VertexEvidence, 3>&) const override {
    return NormalCone{};
  }
  double normal_cone_floor(const Point&, const Point&, const Point&,
                           const std::array<VertexEvidence, 3>&) const override {
    return NormalCone{}.half_angle;
  }
  double width;
};

double minimum_angle(const std::vector<Point>& points, const std::vector<Triangle>& triangles) {
  double result = 180;
  for (const Triangle& t : triangles)
    for (int k = 0; k < 3; ++k) {
      const Point& p = points[static_cast<size_t>(t[k])];
      const Point& a = points[static_cast<size_t>(t[(k + 1) % 3])];
      const Point& b = points[static_cast<size_t>(t[(k + 2) % 3])];
      const Point u{a[0] - p[0], a[1] - p[1], a[2] - p[2]};
      const Point v{b[0] - p[0], b[1] - p[1], b[2] - p[2]};
      const Point w{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0]};
      result = std::min(result, std::atan2(std::hypot(w[0], w[1], w[2]),
                                           u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) *
                                    180 / std::acos(-1.0));
    }
  return result;
}

// Octahedron subdivided `levels` times onto the unit sphere, each vertex
// moved radially and tangentially by a deterministic pseudo-random jitter.
void jittered_sphere(int levels, double jitter, uint64_t seed, std::vector<Point>& points,
                     std::vector<Triangle>& triangles) {
  points = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  triangles = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4},
               {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}};
  for (int level = 0; level < levels; ++level) {
    std::map<std::pair<int64_t, int64_t>, int64_t> middle;
    auto split = [&](int64_t a, int64_t b) {
      const auto key = std::minmax(a, b);
      const auto found = middle.find(key);
      if (found != middle.end()) return found->second;
      const Point& p = points[static_cast<size_t>(a)];
      const Point& q = points[static_cast<size_t>(b)];
      Point m{(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2};
      const double r = std::hypot(m[0], m[1], m[2]);
      points.push_back({m[0] / r, m[1] / r, m[2] / r});
      return middle[key] = static_cast<int64_t>(points.size()) - 1;
    };
    std::vector<Triangle> refined;
    for (const Triangle& t : triangles) {
      const int64_t ab = split(t[0], t[1]), bc = split(t[1], t[2]), ca = split(t[2], t[0]);
      refined.push_back({t[0], ab, ca});
      refined.push_back({ab, t[1], bc});
      refined.push_back({ca, bc, t[2]});
      refined.push_back({ab, bc, ca});
    }
    triangles = refined;
  }
  uint64_t state = seed;
  auto uniform = [&]() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<double>(state >> 11) * 0x1p-53 * 2 - 1;
  };
  for (Point& p : points)
    for (double& value : p) value += jitter * uniform();
}

// A jittered sphere inside a thin band, remeshed toward a 44 degree minimum
// angle. Every remaining shortfall is a remesher limit, not the source: moving
// a corner all the way to its star centroid either fails to lift the worst
// triangle or pushes another star triangle below it, while a partial move or
// a move toward the corner's equilateral apex succeeds (only the full moves
// left 2 to 10 triangles below the target on these seeds).
void sphere_band_quality() {
  for (uint64_t seed = 1; seed <= 6; ++seed) {
    std::vector<Point> points;
    std::vector<Triangle> triangles;
    jittered_sphere(3, 0.01, seed, points, triangles);
    const SphereBand admission(0.02);
    RemeshCounters counters;
    PHX_CHECK(phx::mc::remesh::remesh_surface(points, triangles, admission, {0.25, 44},
                                             counters) == PHX_MC_OK);
    PHX_CHECK(counters.unmet == 0);
    PHX_CHECK(minimum_angle(points, triangles) > 43.99);
    for (const Point& p : points) PHX_CHECK(std::abs(std::hypot(p[0], p[1], p[2]) - 1) <= 0.02);
  }
}

// Admits every triangle inside the ball of radius 1 + width and projects onto
// the unit sphere. Its cones are open, or (tight) hold exactly the
// triangle's own normal within `half` radians.
class BallBand final : public TriangleAdmission {
 public:
  BallBand(double band, bool tight_cones) : width(band), tight(tight_cones) {}
  AdmissionKind kind() const override { return AdmissionKind::kOffsetBand; }
  VertexEvidence evaluate(const Point& p) const override {
    const double radius = std::hypot(p[0], p[1], p[2]);
    return {radius, radius};
  }
  bool admits(const Point&, const Point&, const Point&,
              const std::array<VertexEvidence, 3>& evidence) const override {
    for (const VertexEvidence& corner : evidence)
      if (!(corner.upper <= 1 + width)) return false;
    return true;
  }
  Point project(const Point& p) const override {
    const double radius = std::hypot(p[0], p[1], p[2]);
    return {p[0] / radius, p[1] / radius, p[2] / radius};
  }
  NormalCone normal_cone(const Point& a, const Point& b, const Point& c,
                         const std::array<VertexEvidence, 3>&) const override {
    if (!tight) return NormalCone{};
    return NormalCone{normal({a, b, c}, {0, 1, 2}), half};
  }
  double normal_cone_floor(const Point&, const Point&, const Point&,
                           const std::array<VertexEvidence, 3>&) const override {
    return tight ? half : NormalCone{}.half_angle;
  }
  double width;
  bool tight;
  double half = 1.0e-3;
};

// Smallest solid wedge in degrees over the convex edges of a closed surface
// with outward normals (pi minus the turn where the neighbor's far corner lies
// below the triangle's plane).
double minimum_convex_wedge(const std::vector<Point>& points,
                            const std::vector<Triangle>& triangles) {
  std::map<std::pair<int64_t, int64_t>, size_t> directed;
  for (size_t row = 0; row < triangles.size(); ++row)
    for (int k = 0; k < 3; ++k) directed[{triangles[row][k], triangles[row][(k + 1) % 3]}] = row;
  double wedge = 180;
  for (const auto& [edge, row] : directed) {
    const auto other = directed.find({edge.second, edge.first});
    if (other == directed.end()) continue;
    const Triangle& t = triangles[row];
    const Triangle& s = triangles[other->second];
    int64_t far = s[0];
    for (int64_t v : s)
      if (v != edge.first && v != edge.second) far = v;
    const Point n = normal(points, t);
    const Point m = normal(points, s);
    const Point& p = points[static_cast<size_t>(edge.first)];
    const Point& q = points[static_cast<size_t>(far)];
    if (n[0] * (q[0] - p[0]) + n[1] * (q[1] - p[1]) + n[2] * (q[2] - p[2]) >= 0) continue;
    const double cosine = std::clamp(n[0] * m[0] + n[1] * m[1] + n[2] * m[2], -1.0, 1.0);
    wedge = std::min(wedge, 180 - std::acos(cosine) * 180.0 / std::acos(-1.0));
  }
  return wedge;
}

// An octahedron flattened to a knife-edge lens: its eight right isosceles
// faces (45 degree minimum angle, edges within the size target) score full
// on shape and size, and its rim closes a 1.6 degree convex solid wedge,
// which bounds every tetrahedron on a rim edge to a 1.6 degree dihedral
// angle. With open cones (the torus:1 envelope near its source's medial axis,
// where a partial relocation folded a valence-3 vertex into such a fin) the
// source-relative factor admits the turn, so the wedge alone must lower the
// score: the apexes move to their projections and every wedge opens to the
// octahedron's 109 degrees. Cones forcing the knife edge keep it.
void knife_edge_lens() {
  const double height = 0.01;
  const std::vector<Point> input{{1, 0, 0},  {0, 1, 0},       {-1, 0, 0},
                                 {0, -1, 0}, {0, 0, height}, {0, 0, -height}};
  const std::vector<Triangle> triangles{{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
                                        {1, 0, 5}, {2, 1, 5}, {3, 2, 5}, {0, 3, 5}};
  PHX_CHECK(minimum_convex_wedge(input, triangles) < 2);
  {
    const BallBand admission(0.01, false);
    std::vector<Point> points = input;
    std::vector<Triangle> remeshed = triangles;
    RemeshCounters counters;
    PHX_CHECK(phx::mc::remesh::remesh_surface(points, remeshed, admission, {1.2, 30},
                                             counters) == PHX_MC_OK);
    PHX_CHECK(counters.unmet == 0);
    PHX_CHECK(minimum_convex_wedge(points, remeshed) >= 30);
    for (const Point& p : points) PHX_CHECK(std::hypot(p[0], p[1], p[2]) <= 1.01);
  }
  const BallBand admission(0.01, true);
  std::vector<Point> points = input;
  std::vector<Triangle> remeshed = triangles;
  RemeshCounters counters;
  PHX_CHECK(phx::mc::remesh::remesh_surface(points, remeshed, admission, {1.2, 30}, counters) ==
            PHX_MC_OK);
  PHX_CHECK(counters.unmet == 0);
  PHX_CHECK(counters.relocations == 0 && counters.splits == 0 && counters.flips == 0 &&
            counters.collapses == 0);
  PHX_CHECK(minimum_convex_wedge(points, remeshed) < 2);
  const NormalCone up{{0, 0, 1}, 0.1}, side{{1, 0, 0}, 0.1};
  PHX_CHECK(phx::mc::remesh::forced_turn(up, up) == 0);
  PHX_CHECK(std::abs(phx::mc::remesh::forced_turn(up, side) - (std::acos(-1.0) / 2 - 0.2)) <
            1e-12);
  PHX_CHECK(phx::mc::remesh::forced_turn(NormalCone{}, side) == 0);
}

void exact_contact_parity() {
  using phx::mc::remesh::legal_triangle_contact;
  const std::array<Point, 3> first{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
  const Triangle first_ids{0, 1, 2};
  std::mt19937_64 generator(461);
  std::uniform_real_distribution<double> coordinate(-2.0, 2.0);
  for (int sample = 0; sample < 2000; ++sample) {
    std::array<Point, 3> second{{first[1], first[0],
                                 {coordinate(generator), coordinate(generator),
                                  sample % 4 == 0 ? 0.0 : coordinate(generator)}}};
    Triangle second_ids{1, 0, 3};
    if (sample % 17 == 0) second[0][0] += 0.25;  // inconsistent shared ID
    if (sample % 19 == 0) second_ids[2] = second_ids[1];  // repeated ID
    const double* a[3] = {first[0].data(), first[1].data(), first[2].data()};
    const double* b[3] = {second[0].data(), second[1].data(), second[2].data()};
    phx::mc::Contact contact;
    const auto status = phx::mc::intersect_triangles(a, b, first_ids.data(), second_ids.data(), contact);
    const bool expected = status == PHX_MC_OK &&
        (contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
         contact.kind == PHX_MC_SHARED_EDGE);
    PHX_CHECK(legal_triangle_contact(first, first_ids, second, second_ids) == expected);
  }
  for (int sample = 0; sample < 2000; ++sample) {
    std::array<Point, 3> second{{first[0],
                                 {coordinate(generator), coordinate(generator), coordinate(generator)},
                                 {coordinate(generator), coordinate(generator), coordinate(generator)}}};
    Triangle second_ids{0, 3, 4};
    if (sample % 5 == 0) second[2] = second[1];  // degenerate face
    if (sample % 7 == 0) second[1][2] = second[2][2] = 0;  // coplanar
    if (sample % 17 == 0) second[0][0] += 0.25;  // inconsistent shared ID
    const double* a[3] = {first[0].data(), first[1].data(), first[2].data()};
    const double* b[3] = {second[0].data(), second[1].data(), second[2].data()};
    phx::mc::Contact contact;
    const auto status = phx::mc::intersect_triangles(a, b, first_ids.data(), second_ids.data(), contact);
    const bool expected = status == PHX_MC_OK &&
        (contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
         contact.kind == PHX_MC_SHARED_EDGE);
    PHX_CHECK(legal_triangle_contact(first, first_ids, second, second_ids) == expected);
  }
}

// Exact squared free radius of binary64 corners (independent reference):
// non-obtuse, the circumradius |bc|^2 |ca|^2 |ab|^2 / (4 |ab x ac|^2); obtuse at
// corner o, the maximum over the other corners k of S_other^2 S_o /
// (S_o + S_other - S_k)^2 with S the squared sides. Returns numerator and
// denominator.
std::pair<phx::mc::Dyadic, phx::mc::Dyadic> exact_free_radius_squared(const Point& a,
                                                                     const Point& b,
                                                                     const Point& c) {
  using phx::mc::Dyadic;
  std::array<Dyadic, 3> ab, ac, bc;
  for (int axis = 0; axis < 3; ++axis) {
    ab[axis] = Dyadic(b[axis]) - Dyadic(a[axis]);
    ac[axis] = Dyadic(c[axis]) - Dyadic(a[axis]);
    bc[axis] = Dyadic(c[axis]) - Dyadic(b[axis]);
  }
  const auto dot = [](const std::array<Dyadic, 3>& u, const std::array<Dyadic, 3>& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  const std::array<Dyadic, 3> n{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                                ab[0] * ac[1] - ab[1] * ac[0]};
  const std::array<Dyadic, 3> s{dot(bc, bc), dot(ac, ac), dot(ab, ab)};
  int obtuse = -1;
  for (int k = 0; k < 3; ++k)
    if ((s[k] - s[(k + 1) % 3] - s[(k + 2) % 3]).sign() > 0) obtuse = k;
  if (obtuse < 0) return {s[0] * s[1] * s[2], Dyadic(4.0) * dot(n, n)};
  std::pair<Dyadic, Dyadic> best{Dyadic(), Dyadic(1.0)};
  for (int k = 0; k < 3; ++k) {
    if (k == obtuse) continue;
    const int other = 3 - k - obtuse;
    const Dyadic d = s[obtuse] + s[other] - s[k];
    const Dyadic numerator = s[other] * s[other] * s[obtuse], denominator = d * d;
    if ((numerator * best.second - best.first * denominator).sign() > 0)
      best = {numerator, denominator};
  }
  return best;
}

bool covers(double radius, const std::pair<phx::mc::Dyadic, phx::mc::Dyadic>& exact) {
  const phx::mc::Dyadic r(radius);
  return (r * r * exact.second - exact.first).sign() >= 0;
}

double envelope_roundoff(const Point& a, const Point& b, const Point& c) {
  double scale = 0;
  for (const Point* p : {&a, &b, &c})
    for (double v : *p) scale = std::max(scale, std::abs(v));
  return std::nextafter(std::max(256 * std::numeric_limits<double>::epsilon() * scale,
                                 256 * std::numeric_limits<double>::denorm_min()),
                        std::numeric_limits<double>::infinity());
}

// The free radius must bound the exact one for binary64 needles, whose
// floating cross product loses relative accuracy of about eps / theta. The
// first rows are the Windows audit's escapes (C8); the random needles and caps
// repeat its probe.
void certified_free_radius_is_exact() {
  using phx::mc::remesh::certified_free_radius;
  const std::array<std::array<Point, 3>, 5> escapes{{
      {{{-0x1.7156d5fa24df0p-1, 0x1.baf3c9d8ec664p-2, -0x1.856534f685164p-2},
        {-0x1.0af5efa4a7ba6p-1, 0x1.2176fc2305e7ap-1, -0x1.50629dffa805ap-1},
        {-0x1.0af600e997fdcp-1, 0x1.2177007bfa29ap-1, -0x1.5062a8640476ep-1}}},
      {{{0x1.8b3ce171aaa0ap-1, -0x1.10a8fb78985c0p-1, 0x1.522e4628d3c58p-1},
        {0x1.f830fb2ad305fp-1, -0x1.41fda84f50363p-1, 0x1.ab9e617334331p-2},
        {0x1.f830fb37251a1p-1, -0x1.41fda847cd213p-1, 0x1.ab9e6182d4cf9p-2}}},
      {{{0x1.83ec42ffac8c0p-5, -0x1.e174b90083ad4p-1, -0x1.f4b2c8b7ca1fcp-1},
        {-0x1.b63b05759e842p-2, -0x1.76298e9efd6cbp+0, -0x1.391e328c39dbbp+0},
        {-0x1.b638625b4e03ap-2, -0x1.762a5257200d5p+0, -0x1.391dd9a81f49fp+0}}},
      {{{0x1.8ed948f0188b0p-1, -0x1.ad390989186e4p-1, -0x1.f4c6549956f58p-3},
        {0x1.83b919c2a19adp-1, -0x1.bcbe2b1fdefa4p-1, -0x1.d01e2bdf83c57p-3},
        {0x1.83b90c5e43ec1p-1, -0x1.bcbe226dbefeep-1, -0x1.d01e320037e35p-3}}},
      {{{0x1.c0cdb5943415cp-2, -0x1.9646364cc6752p-1, 0x1.4ddf18f4f3a5ep-1},
        {0x1.6f9bb4f0f5d75p-2, -0x1.15443b8523851p-1, 0x1.6d5f7b85a308fp-1},
        {0x1.6f9bb4fba105bp-2, -0x1.15443842e5a03p-1, 0x1.6d5f6e340a1afp-1}}},
  }};
  for (const auto& [a, b, c] : escapes) {
    const double roundoff = envelope_roundoff(a, b, c);
    const double radius = certified_free_radius(a, b, c, roundoff);
    const auto exact = exact_free_radius_squared(a, b, c);
    PHX_CHECK(std::isfinite(radius));
    PHX_CHECK(covers(radius, exact));
    // Only the caller's absolute roundoff and directed rounding separate the
    // bound from the exact radius.
    PHX_CHECK(!covers(radius - 2 * roundoff, exact));
  }
  // An obtuse triangle takes the bisector bound (exactly 0.265625 here), not
  // its circumradius (1.0625); a negligible roundoff leaves directed rounding.
  {
    const Point a{0, 0, 0}, b{1, 0, 0}, c{0.5, 0.125, 0};
    const double radius = certified_free_radius(a, b, c, 0x1p-60);
    const auto exact = exact_free_radius_squared(a, b, c);
    PHX_CHECK(covers(radius, exact));
    double below = radius;
    for (int step = 0; step < 4; ++step)
      below = std::nextafter(below, -std::numeric_limits<double>::infinity());
    PHX_CHECK(!covers(below, exact));
    PHX_CHECK(radius < 0.5);
  }
  // An exactly zero cross product or a nonfinite corner has no free radius.
  PHX_CHECK(std::isinf(certified_free_radius({0, 0, 0}, {0.25, 0.5, 1}, {0.5, 1, 2}, 0x1p-60)));
  PHX_CHECK(std::isinf(certified_free_radius({0, 0, 0}, {1, 0, 0},
                                             {0, std::numeric_limits<double>::quiet_NaN(), 0},
                                             0x1p-60)));
  std::mt19937_64 generator(20261106);
  std::uniform_real_distribution<double> unit(-1.0, 1.0), exponent(0.0, 1.0);
  std::normal_distribution<double> normal;
  int escaped = 0, finite = 0;
  for (int sample = 0; sample < 20000; ++sample) {
    const Point apex{unit(generator), unit(generator), unit(generator)};
    Point direction{normal(generator), normal(generator), normal(generator)};
    const double direction_length = std::sqrt(direction[0] * direction[0] +
                                              direction[1] * direction[1] +
                                              direction[2] * direction[2]);
    for (double& v : direction) v /= direction_length;
    Point side{normal(generator), normal(generator), normal(generator)};
    const double along = side[0] * direction[0] + side[1] * direction[1] + side[2] * direction[2];
    for (int axis = 0; axis < 3; ++axis) side[axis] -= along * direction[axis];
    const double side_length = std::sqrt(side[0] * side[0] + side[1] * side[1] + side[2] * side[2]);
    for (double& v : side) v /= side_length;
    const double length = std::pow(10.0, -3.0 * exponent(generator));
    const double theta = std::pow(10.0, -9.0 + 8.0 * exponent(generator));
    Point b, c;
    for (int axis = 0; axis < 3; ++axis) {
      if (sample % 2 == 0) {
        b[axis] = apex[axis] + length * direction[axis] + 0.5 * length * theta * side[axis];
        c[axis] = apex[axis] + length * direction[axis] - 0.5 * length * theta * side[axis];
      } else {
        b[axis] = apex[axis] + length * direction[axis];
        c[axis] = apex[axis] + 0.5 * length * direction[axis] + length * theta * side[axis];
      }
    }
    const double radius = certified_free_radius(apex, b, c, envelope_roundoff(apex, b, c));
    if (!std::isfinite(radius)) continue;
    ++finite;
    if (!covers(radius, exact_free_radius_squared(apex, b, c))) ++escaped;
  }
  PHX_CHECK(finite == 20000);
  PHX_CHECK(escaped == 0);
}

}  // namespace

int main() {
  folded_cluster();
  sphere_band_quality();
  knife_edge_lens();
  exact_contact_parity();
  certified_free_radius_is_exact();
  return phx::mc::test::finish("test_surface_remesh");
}
