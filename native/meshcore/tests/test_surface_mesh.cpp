//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Chart-oriented surface edge insertion preserves reciprocal topology and source constraints.
#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"

namespace {
void split_shared_edge() {
  constexpr double charts[] = {0, 0, 1, 0, 1, 1, 0, 1};
  constexpr double points[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  constexpr double normals[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  constexpr int32_t triangles[] = {0, 1, 2, 0, 2, 3};
  constexpr int8_t constrained[] = {1, 0, 1, 1, 1, 0};
  constexpr double insert_chart[] = {0.5, 0.5};
  constexpr double insert_point[] = {0.5, 0.5, 0.0};
  constexpr double insert_normal[] = {0, 0, 0};
  constexpr double spacing[] = {0.0};
  constexpr int32_t hints[] = {-1};
  std::array<int32_t, 12> output_triangles{};
  std::array<int8_t, 12> output_constrained{};
  std::array<int32_t, 1> vertex_ids{};
  std::array<int32_t, 1> item_status{};
  int64_t output_count = -1;
  const phx_mc_surface_reconnect_request request{
      .vertex_count = 4,
      .charts = charts,
      .points = points,
      .normals = normals,
      .vertex_metrics = nullptr,
      .vertex_pole_ids = nullptr,
      .triangle_count = 2,
      .triangles = triangles,
      .constrained = constrained,
      .insert_count = 1,
      .insert_charts = insert_chart,
      .insert_points = insert_point,
      .insert_normals = insert_normal,
      .insert_metrics = nullptr,
      .insert_pole_ids = nullptr,
      .insert_spacing = spacing,
      .insert_hints = hints,
      .insert_edges = nullptr,
      .sweep = 0,
      .triangle_capacity = 4,
      .triangles_out = output_triangles.data(),
      .constrained_out = output_constrained.data(),
      .triangle_count_out = &output_count,
      .vertex_ids = vertex_ids.data(),
      .item_status = item_status.data(),
  };
  PHX_CHECK(phx_mc_surface_reconnect(&request) == PHX_MC_OK);
  PHX_CHECK(output_count == 4);
  PHX_CHECK(vertex_ids[0] == 4);
  PHX_CHECK(item_status[0] == PHX_MC_OK);
  std::map<std::pair<int32_t, int32_t>, int> edge_counts;
  for (int64_t cell = 0; cell < output_count; ++cell) {
    const int32_t* t = output_triangles.data() + 3 * cell;
    for (int k = 0; k < 3; ++k) {
      int32_t a = t[(k + 1) % 3];
      int32_t b = t[(k + 2) % 3];
      if (a > b) std::swap(a, b);
      ++edge_counts[{a, b}];
    }
  }
  for (const auto& [edge, count] : edge_counts) {
    const bool boundary = edge.first == 0 && edge.second == 1 ||
                          edge.first == 1 && edge.second == 2 ||
                          edge.first == 2 && edge.second == 3 ||
                          edge.first == 0 && edge.second == 3;
    PHX_CHECK(count == (boundary ? 1 : 2));
  }
}

void index_range_status() {
  phx_mc_surface_reconnect_request request{};
  request.vertex_count = int64_t{1} << 31;
  PHX_CHECK(phx_mc_surface_reconnect(&request) == PHX_MC_INDEX_RANGE_ERROR);
  request = {};
  request.triangle_count = int64_t{1} << 31;
  PHX_CHECK(phx_mc_surface_reconnect(&request) == PHX_MC_INDEX_RANGE_ERROR);
}

// The square's chart corners are exactly cocircular around the inserted center;
// a subnormal chart coordinate lies outside the exact coordinate domain.
void floating_point_hazards() {
  struct Reconnect {
    int32_t status = -1;
    std::array<int32_t, 12> triangles{};
    std::array<int8_t, 12> constrained{};
    std::array<int32_t, 1> vertex_ids{};
    std::array<int32_t, 1> item_status{};
    int64_t count = -1;
  };
  const auto reconnect = [](double corner_u) -> Reconnect {
    const double charts[] = {0, 0, 1, 0, 1, 1, corner_u, 1};
    constexpr double points[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    constexpr double normals[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr int32_t triangles[] = {0, 1, 2, 0, 2, 3};
    constexpr int8_t constrained[] = {1, 0, 1, 1, 1, 0};
    constexpr double insert_chart[] = {0.5, 0.5};
    constexpr double insert_point[] = {0.5, 0.5, 0.0};
    constexpr double insert_normal[] = {0, 0, 0};
    constexpr double spacing[] = {0.0};
    constexpr int32_t hints[] = {-1};
    Reconnect result;
    const phx_mc_surface_reconnect_request request{
        .vertex_count = 4,
        .charts = charts,
        .points = points,
        .normals = normals,
        .vertex_metrics = nullptr,
        .vertex_pole_ids = nullptr,
        .triangle_count = 2,
        .triangles = triangles,
        .constrained = constrained,
        .insert_count = 1,
        .insert_charts = insert_chart,
        .insert_points = insert_point,
        .insert_normals = insert_normal,
        .insert_metrics = nullptr,
        .insert_pole_ids = nullptr,
        .insert_spacing = spacing,
        .insert_hints = hints,
        .insert_edges = nullptr,
        .sweep = 0,
        .triangle_capacity = 4,
        .triangles_out = result.triangles.data(),
        .constrained_out = result.constrained.data(),
        .triangle_count_out = &result.count,
        .vertex_ids = result.vertex_ids.data(),
        .item_status = result.item_status.data(),
    };
    result.status = phx_mc_surface_reconnect(&request);
    return result;
  };
  const Reconnect reference = reconnect(0.0);
  PHX_CHECK(reference.status == PHX_MC_OK);
  PHX_CHECK(reference.count == 4);
  PHX_CHECK(reference.vertex_ids[0] == 4);
  PHX_CHECK(reference.item_status[0] == PHX_MC_OK);
  PHX_CHECK(reconnect(0x1p-1060).status == PHX_MC_RANGE_ERROR);
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    const Reconnect call = reconnect(0.0);
    PHX_CHECK(call.status == reference.status);
    PHX_CHECK(call.count == reference.count);
    PHX_CHECK(call.triangles == reference.triangles);
    PHX_CHECK(call.constrained == reference.constrained);
    PHX_CHECK(call.vertex_ids == reference.vertex_ids);
    PHX_CHECK(call.item_status == reference.item_status);
    PHX_CHECK(reconnect(0x1p-1060).status == PHX_MC_RANGE_ERROR);
  });
}

}  // namespace

int main() {
  split_shared_edge();
  index_range_status();
  floating_point_hazards();
  return phx::mc::test::finish("test_surface_mesh");
}
