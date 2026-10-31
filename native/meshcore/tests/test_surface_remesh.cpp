// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Source-agnostic surface remesher: the fold factor of the quality score.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "surface_remesh.hpp"

namespace {
using phx::mc::remesh::AdmissionKind;
using phx::mc::remesh::Point;
using phx::mc::remesh::RemeshCounters;
using phx::mc::remesh::RemeshTargets;
using phx::mc::remesh::Triangle;
using phx::mc::remesh::TriangleAdmission;
using phx::mc::remesh::VertexEvidence;

// Admits every triangle: isolates the score and topology rules.
class AnyTriangle final : public TriangleAdmission {
 public:
  AdmissionKind kind() const override { return AdmissionKind::kOffsetBand; }
  VertexEvidence evaluate(const Point&) const override { return {}; }
  bool admits(const Point&, const Point&, const Point&,
              const std::array<VertexEvidence, 3>&) const override {
    return true;
  }
  Point project(const Point& position) const override { return position; }
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
  const AnyTriangle admission;
  // A bound near 180 degrees leaves the fold factor at one: the fold stays.
  {
    std::vector<Point> points = input;
    std::vector<Triangle> remeshed = triangles;
    RemeshCounters counters;
    PHX_CHECK(phx::mc::remesh::remesh_surface(points, remeshed, admission, {0.04, 30, 179.9},
                                             counters) == PHX_MC_OK);
    PHX_CHECK(maximum_turn(points, remeshed) > 170);
  }
  // The default 90 degree bound scores the fold as unmet and removes it.
  std::vector<Point> points = input;
  std::vector<Triangle> remeshed = triangles;
  RemeshCounters counters;
  PHX_CHECK(phx::mc::remesh::remesh_surface(points, remeshed, admission, {0.04, 30}, counters) ==
            PHX_MC_OK);
  PHX_CHECK(maximum_turn(points, remeshed) <= 90);
  // The bound is a full-score target strictly between 0 and 180 degrees.
  for (double bound : {0.0, 180.0, std::nan("")}) {
    std::vector<Point> rejected_points = input;
    std::vector<Triangle> rejected = triangles;
    PHX_CHECK(phx::mc::remesh::remesh_surface(rejected_points, rejected, admission,
                                             {0.04, 30, bound}, counters) ==
              PHX_MC_INVALID_ARGUMENT);
  }
}

}  // namespace

int main() {
  folded_cluster();
  return phx::mc::test::finish("test_surface_remesh");
}
