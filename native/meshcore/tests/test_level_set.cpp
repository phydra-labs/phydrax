//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact P1 edge-root insertion, source witnesses, positive children and protected refusal.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {
constexpr double points[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
constexpr int32_t cell[] = {0, 1, 2, 3};
constexpr int32_t faces[] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
constexpr int32_t edges[] = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};

void exact_edge_roots_and_refusal() {
  constexpr double values[] = {-0.5, 0.5, 0.5, 0.5};
  std::array<int8_t, 6> protected_edges{};
  std::array<double, 21> out_points{};
  std::array<int32_t, 16> out_cells{};
  std::array<int32_t, 4> out_parents{};
  std::array<int32_t, 14> out_sources{};
  std::array<double, 14> out_weights{};
  std::array<double, 7> out_deviations{};
  int64_t out_point_count = -1;
  int64_t out_cell_count = -1;
  int64_t failed_edge = -2;
  const phx_mc_level_set_split_3d_request request{
      .point_count = 4,
      .points = points,
      .values = values,
      .cell_count = 1,
      .cells = cell,
      .face_count = 4,
      .faces = faces,
      .edge_count = 6,
      .edges = edges,
      .protected_edges = protected_edges.data(),
      .output_mode = PHX_MC_LEVEL_SET_FULL_CARRIERS,
      .out_point_capacity = 7,
      .out_cell_capacity = 4,
      .out_points = out_points.data(),
      .out_cells = out_cells.data(),
      .out_parents = out_parents.data(),
      .out_sources = out_sources.data(),
      .out_weights = out_weights.data(),
      .out_deviations = out_deviations.data(),
      .out_point_count = &out_point_count,
      .out_cell_count = &out_cell_count,
      .failed_edge = &failed_edge,
  };
  PHX_CHECK(phx_mc_level_set_split_3d(&request) == PHX_MC_OK);
  PHX_CHECK(out_point_count == 7);
  PHX_CHECK(out_cell_count == 4);
  PHX_CHECK(failed_edge == -1);
  for (int root = 0; root < 3; ++root) {
    const int32_t vertex = 4 + root;
    PHX_CHECK(out_sources[2 * vertex] == 0);
    PHX_CHECK(out_sources[2 * vertex + 1] == root + 1);
    PHX_CHECK(out_weights[2 * vertex] == 0.5);
    PHX_CHECK(out_weights[2 * vertex + 1] == 0.5);
    PHX_CHECK(out_deviations[vertex] == 0.0);
    for (int axis = 0; axis < 3; ++axis) {
      PHX_CHECK(out_points[3 * vertex + axis] == (axis == root ? 0.5 : 0.0));
    }
  }
  double determinant_sum = 0.0;
  for (int64_t row = 0; row < out_cell_count; ++row) {
    const int32_t* v = out_cells.data() + 4 * row;
    PHX_CHECK(phx::mc::orient3d(out_points.data() + 3 * v[0], out_points.data() + 3 * v[1],
                                out_points.data() + 3 * v[2], out_points.data() + 3 * v[3]) > 0);
    determinant_sum += phx::mc::orient3d_exact(
                           out_points.data() + 3 * v[0], out_points.data() + 3 * v[1],
                           out_points.data() + 3 * v[2], out_points.data() + 3 * v[3])
                           .estimate();
    PHX_CHECK(out_parents[row] == 0);
  }
  PHX_CHECK_NEAR(determinant_sum, 1.0, 1.0e-14);

  protected_edges[1] = 1;
  out_point_count = -7;
  out_cell_count = -7;
  failed_edge = -1;
  PHX_CHECK(phx_mc_level_set_split_3d(&request) == PHX_MC_CONSTRAINT_INTERSECTION);
  PHX_CHECK(failed_edge == 1);
  PHX_CHECK(out_point_count == -7);
  PHX_CHECK(out_cell_count == -7);
}

void index_range_status() {
  phx_mc_level_set_split_3d_request request{};
  request.point_count = int64_t{1} << 31;
  PHX_CHECK(phx_mc_level_set_split_3d(&request) == PHX_MC_INDEX_RANGE_ERROR);
  request = {};
  request.cell_count = int64_t{1} << 31;
  PHX_CHECK(phx_mc_level_set_split_3d(&request) == PHX_MC_INDEX_RANGE_ERROR);
}

// Exact half-value roots and an exact vertex root must not depend on the
// caller's mode. A subnormal coordinate, or a subnormal value whose roots lie
// subnormally close to a vertex, is outside the exact coordinate domain.
void floating_point_hazards() {
  struct Split {
    int32_t status = -1;
    std::array<double, 21> points{};
    std::array<int32_t, 16> cells{};
    std::array<int32_t, 4> parents{};
    std::array<int32_t, 14> sources{};
    std::array<double, 14> weights{};
    std::array<double, 7> deviations{};
    int64_t point_count = -1;
    int64_t cell_count = -1;
    int64_t failed_edge = -2;

    bool operator==(const Split&) const = default;
  };
  const auto split = [](double apex_x, double first_value, double last_value) -> Split {
    const double source_points[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, apex_x, 0, 1};
    const double values[] = {first_value, 0.5, 0.5, last_value};
    constexpr int8_t protected_edges[] = {0, 0, 0, 0, 0, 0};
    Split result;
    const phx_mc_level_set_split_3d_request request{
        .point_count = 4,
        .points = source_points,
        .values = values,
        .cell_count = 1,
        .cells = cell,
        .face_count = 4,
        .faces = faces,
        .edge_count = 6,
        .edges = edges,
        .protected_edges = protected_edges,
        .output_mode = PHX_MC_LEVEL_SET_FULL_CARRIERS,
        .out_point_capacity = 7,
        .out_cell_capacity = 4,
        .out_points = result.points.data(),
        .out_cells = result.cells.data(),
        .out_parents = result.parents.data(),
        .out_sources = result.sources.data(),
        .out_weights = result.weights.data(),
        .out_deviations = result.deviations.data(),
        .out_point_count = &result.point_count,
        .out_cell_count = &result.cell_count,
        .failed_edge = &result.failed_edge,
    };
    result.status = phx_mc_level_set_split_3d(&request);
    return result;
  };
  const Split reference = split(0.0, -0.5, 0.5);
  PHX_CHECK(reference.status == PHX_MC_OK);
  PHX_CHECK(reference.point_count == 7);
  PHX_CHECK(reference.cell_count == 4);
  PHX_CHECK(reference.failed_edge == -1);
  for (int root = 0; root < 3; ++root) {
    for (int axis = 0; axis < 3; ++axis) {
      PHX_CHECK(reference.points[3 * (4 + root) + axis] == (axis == root ? 0.5 : 0.0));
    }
  }
  const Split vertex_root = split(0.0, -0.5, 0.0);
  PHX_CHECK(vertex_root.status == PHX_MC_OK);
  PHX_CHECK(vertex_root.point_count == 6);
  PHX_CHECK(vertex_root.cell_count == 3);
  PHX_CHECK(split(0x1p-1060, -0.5, 0.5).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(split(0.0, -0x1p-1060, 0.5).status == PHX_MC_INVALID_INPUT);
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    PHX_CHECK(split(0.0, -0.5, 0.5) == reference);
    PHX_CHECK(split(0.0, -0.5, 0.0) == vertex_root);
    PHX_CHECK(split(0x1p-1060, -0.5, 0.5).status == PHX_MC_INVALID_INPUT);
    PHX_CHECK(split(0.0, -0x1p-1060, 0.5).status == PHX_MC_INVALID_INPUT);
  });
}

}  // namespace

int main() {
  exact_edge_roots_and_refusal();
  index_range_status();
  floating_point_hazards();
  return phx::mc::test::finish("test_level_set");
}
