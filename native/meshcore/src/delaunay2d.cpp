//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: 2D Delaunay and regular triangulations (incremental Bowyer-Watson
// with ghost triangles and symbolic perturbation, see triangulation2d.hpp).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "triangulation2d.hpp"

namespace phx::mc {

static int32_t triangulate_2d_impl(int64_t point_count, const double* points,
                                   const double* weights, bool weighted, bool canonical,
                                   bool retain_points, phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || !addressable(point_count, 2, sizeof(double)) ||
      !addressable(point_count, 1, sizeof(int32_t)) ||
      !addressable(point_count, 1, sizeof(double)) ||
      (point_count > 0 && points == nullptr) || (weighted && point_count > 0 && weights == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (point_count > kMaxMeshPoints) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const double* active_weights = weighted ? weights : nullptr;
  int32_t status = validate_points(points, point_count, 2, active_weights);
  if (status != PHX_MC_OK) {
    return status;
  }
  auto result = std::make_unique<phx_mc_mesh>();
  result->dimension = 2;
  result->input_point_count = point_count;
  if (point_count == 0) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  const auto [prepared, order] =
      deduplicated_insertion_order(points, point_count, 2, active_weights, result->vertex_map);
  if (prepared != PHX_MC_OK) return prepared;
  // Vertex k of the triangulation is input point order[k], stored in
  // insertion order so that walks and cavities touch nearby memory; its
  // input index orders the symbolic perturbation, so the triangulation is
  // that of the input points.
  const std::size_t count = order.size();
  std::vector<double> local_points(2 * count);
  std::vector<double> local_weights(weighted ? count : 0);
  for (std::size_t k = 0; k < count; ++k) {
    const double* point = points + 2 * static_cast<std::size_t>(order[k]);
    local_points[2 * k] = point[0];
    local_points[2 * k + 1] = point[1];
    if (weighted) local_weights[k] = weights[order[k]];
  }
  std::vector<int32_t> local_order(count);
  std::iota(local_order.begin(), local_order.end(), 0);
  Triangulation2D triangulation;
  triangulation.reset(local_points.data(), static_cast<int64_t>(count),
                      weighted ? local_weights.data() : nullptr);
  triangulation.identities = order.data();
  status = build_ordered_delaunay_2d(triangulation, local_order);
  if (status != PHX_MC_OK) {
    return status;
  }
  // In the unweighted triangulation no distinct representative is hidden.
  if (weighted) {
    std::vector<char> input_alive(static_cast<std::size_t>(point_count), 0);
    for (std::size_t k = 0; k < count; ++k) {
      input_alive[static_cast<std::size_t>(order[k])] = triangulation.vertex_triangle[k] >= 0;
    }
    for (int32_t& target : result->vertex_map) {
      if (target >= 0 && input_alive[static_cast<std::size_t>(target)] == 0) target = -1;
    }
  }
  if (retain_points) result->points.assign(points, points + 2 * point_count);
  triangulation.finite_cells(result->cells);  // input indices (identities)
  if (canonical) {
    const int32_t canonicalized = canonicalize_cells(*result, 1);
    if (canonicalized != PHX_MC_OK) return canonicalized;
  }
  *mesh = result.release();
  return PHX_MC_OK;
}

int32_t triangulate_2d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, bool canonical, phx_mc_mesh** mesh) {
  return triangulate_2d_impl(point_count, points, weights, weighted, canonical, true, mesh);
}

}  // namespace phx::mc

extern "C" {

int32_t phx_mc_delaunay_2d(int64_t point_count, const double* points, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::triangulate_2d(point_count, points, nullptr, false, true, mesh);
  });
}

int32_t phx_mc_delaunay_2d_into(
    int64_t point_count, const double* points, int64_t cell_capacity, int32_t* cells,
    int32_t* vertex_map, int64_t* cell_count) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!phx::mc::addressable(cell_capacity, 3, sizeof(int32_t)) || cell_count == nullptr ||
        (cell_capacity != 0 && cells == nullptr) || (point_count > 0 && vertex_map == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx_mc_mesh* raw = nullptr;
    const auto status =
        phx::mc::triangulate_2d_impl(point_count, points, nullptr, false, true, false, &raw);
    if (status != PHX_MC_OK) return status;
    const std::unique_ptr<phx_mc_mesh> result(raw);
    const auto count = result->cell_count();
    if (count > cell_capacity) return PHX_MC_CAPACITY_EXCEEDED;
    std::copy(result->cells.begin(), result->cells.end(), cells);
    std::copy(result->vertex_map.begin(), result->vertex_map.end(), vertex_map);
    *cell_count = count;
    return PHX_MC_OK;
  });
}

int32_t phx_mc_regular_2d(int64_t point_count, const double* points, const double* weights,
                          phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::triangulate_2d(point_count, points, weights, true, true, mesh);
  });
}

}  // extern "C"
