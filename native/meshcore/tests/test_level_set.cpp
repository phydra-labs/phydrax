//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Consecutive P1 edge-root insertions preserve constrained source strata,
// positive cells, parent provenance, and refusal without publishing output.
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "tet_mesh.hpp"

namespace {
using namespace phx::mc;

constexpr double kPoints[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
constexpr int32_t kCell[] = {0, 1, 2, 3};
constexpr int32_t kFaces[] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
constexpr int32_t kEdges[] = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
constexpr int32_t kFaceSources[] = {0, 1, 2, 3};
constexpr int32_t kEdgeSources[] = {0, 1, 2, 3, 4, 5};

void check_cells(const double* points, const int32_t* cells, int64_t count) {
  double determinant_sum = 0.0;
  for (int64_t row = 0; row < count; ++row) {
    const int32_t* v = cells + 4 * row;
    PHX_CHECK(orient3d(points + 3 * v[0], points + 3 * v[1],
                       points + 3 * v[2], points + 3 * v[3]) > 0);
    const double* corners[4] = {points + 3 * v[0], points + 3 * v[1],
                                points + 3 * v[2], points + 3 * v[3]};
    determinant_sum += 6.0 * geometry::signed_volume(corners);
  }
  // Every determinant of this midpoint subdivision is exactly dyadic.
  PHX_CHECK(determinant_sum == 1.0);
}

void test_consecutive_constrained_edge_insertions() {
  TetMesh mesh(BoundaryPolicy::kConforming, 7, 32);
  const int32_t region = 0;
  PHX_CHECK(mesh.build(4, kPoints, 1, kCell, &region, 4, kFaces, kFaceSources,
                       6, kEdges, kEdgeSources, nullptr) == PHX_MC_OK);
  for (int32_t endpoint = 1; endpoint <= 3; ++endpoint) {
    double root[3] = {};
    root[endpoint - 1] = 0.5;
    std::vector<int32_t> star;
    PHX_CHECK(mesh.vertex_star(0, star));
    int32_t origin = -1;
    for (int32_t cell : star) {
      if (!is_ghost(mesh.tet(cell)) && vertex_slot(mesh.tet(cell), endpoint) >= 0) {
        origin = cell;
        break;
      }
    }
    PHX_CHECK(origin >= 0);
    if (origin < 0) return;
    const Insertion prepared = mesh.prepare(root, origin, 0, endpoint);
    PHX_CHECK(prepared == Insertion::kOk);
    if (prepared != Insertion::kOk) return;
    const Insertion committed = mesh.commit(InsertKind::kSubsegment, 0.0);
    PHX_CHECK(committed == Insertion::kOk);
    if (committed != Insertion::kOk) return;
    PHX_CHECK(mesh.vertex_count() == 4 + endpoint);
    PHX_CHECK(mesh.complex().audit());
    PHX_CHECK(mesh.segment_source(0, 3 + endpoint) == endpoint - 1);
    PHX_CHECK(mesh.segment_source(3 + endpoint, endpoint) == endpoint - 1);
    std::vector<std::array<int32_t, 5>> cells;
    mesh.collect_cells(cells);
    std::vector<int32_t> connectivity;
    for (const auto& cell : cells) {
      connectivity.insert(connectivity.end(), cell.begin(), cell.begin() + 4);
      PHX_CHECK(cell[4] == region);
    }
    std::vector<double> points;
    for (int32_t vertex = 0; vertex < mesh.vertex_count(); ++vertex) {
      points.insert(points.end(), mesh.point(vertex), mesh.point(vertex) + 3);
    }
    check_cells(points.data(), connectivity.data(), static_cast<int64_t>(cells.size()));
  }
}

void test_level_set_midpoints_in_every_crossing_order() {
  std::array<int32_t, 3> order = {0, 1, 2};
  do {
    std::array<int32_t, 12> edges;
    for (std::size_t i = 0; i < order.size(); ++i) {
      std::copy_n(kEdges + 2 * order[i], 2, edges.data() + 2 * i);
    }
    std::copy_n(kEdges + 6, 6, edges.data() + 6);
    const double values[] = {-0.5, 0.5, 0.5, 0.5};
    std::array<int8_t, 6> protected_edges = {};
    std::array<double, 21> points;
    std::array<int32_t, 128> cells;
    std::array<int32_t, 32> parents;
    std::array<int32_t, 14> sources;
    std::array<double, 14> weights;
    std::array<int64_t, PHX_MC_LEVEL_SET_COUNTERS> counters = {};
    const phx_mc_level_set_split_3d_request request{
        4, kPoints, values, 1, kCell, 4, kFaces, 6, edges.data(), protected_edges.data(),
        7, 32, 100000, 32, points.data(), cells.data(), parents.data(), sources.data(),
        weights.data(), counters.data()};
    const int32_t status = phx_mc_level_set_split_3d(&request);
    PHX_CHECK(status == PHX_MC_OK);
    if (status != PHX_MC_OK) continue;
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_VERTICES] == 7);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_INSERTED] == 3);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_FAILED_EDGE] == -1);
    PHX_CHECK(std::equal(std::begin(kPoints), std::end(kPoints), points.begin()));
    for (int32_t root = 0; root < 3; ++root) {
      const int32_t vertex = 4 + root;
      PHX_CHECK(sources[2 * vertex] == 0);
      PHX_CHECK(sources[2 * vertex + 1] == order[root] + 1);
      PHX_CHECK(weights[2 * vertex] == 0.5 && weights[2 * vertex + 1] == 0.5);
      for (int axis = 0; axis < 3; ++axis) {
        PHX_CHECK(points[3 * vertex + axis] == (axis == order[root] ? 0.5 : 0.0));
      }
    }
    const int64_t count = counters[PHX_MC_LEVEL_SET_CELLS];
    check_cells(points.data(), cells.data(), count);
    for (int64_t row = 0; row < count; ++row) {
      PHX_CHECK(parents[row] == 0);
      bool negative = false, positive = false;
      for (int k = 0; k < 4; ++k) {
        const int32_t vertex = cells[4 * row + k];
        negative = negative || vertex == 0;
        positive = positive || (vertex > 0 && vertex < 4);
      }
      PHX_CHECK(negative != positive);
    }
    // A protected second crossing must refuse without publishing any output.
    protected_edges[1] = 1;
    points.fill(-7.0);
    cells.fill(-7);
    parents.fill(-7);
    sources.fill(-7);
    weights.fill(-7.0);
    PHX_CHECK(phx_mc_level_set_split_3d(&request) == PHX_MC_CONSTRAINT_INTERSECTION);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_INSERTED] == 1);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_FAILED_EDGE] == 1);
    PHX_CHECK(std::all_of(points.begin(), points.end(), [](double x) { return x == -7.0; }));
    PHX_CHECK(std::all_of(cells.begin(), cells.end(), [](int32_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(parents.begin(), parents.end(), [](int32_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(sources.begin(), sources.end(), [](int32_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(weights.begin(), weights.end(), [](double x) { return x == -7.0; }));
  } while (std::next_permutation(order.begin(), order.end()));
}
}  // namespace

int main() {
  test_consecutive_constrained_edge_insertions();
  test_level_set_midpoints_in_every_crossing_order();
  return phx::mc::test::finish("test_level_set");
}
