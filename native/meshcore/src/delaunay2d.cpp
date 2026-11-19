//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: 2D Delaunay and regular triangulations (incremental Bowyer-Watson
// with ghost triangles and symbolic perturbation, see triangulation2d.hpp).
#include <cstdint>
#include <memory>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "triangulation2d.hpp"

namespace phx::mc {

int32_t triangulate_2d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, bool canonical, phx_mc_mesh** mesh) {
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
  Triangulation2D triangulation;
  triangulation.reset(points, point_count, active_weights);
  status = build_delaunay_2d(triangulation, points, point_count, active_weights,
                             result->vertex_map);
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int32_t& target : result->vertex_map) {
    if (target >= 0 && triangulation.vertex_triangle[static_cast<std::size_t>(target)] < 0) {
      target = -1;
    }
  }
  result->points.assign(points, points + 2 * point_count);
  triangulation.finite_cells(result->cells);
  if (canonical) canonicalize_cells(*result);
  *mesh = result.release();
  return PHX_MC_OK;
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

int32_t phx_mc_regular_2d(int64_t point_count, const double* points, const double* weights,
                          phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::triangulate_2d(point_count, points, weights, true, true, mesh);
  });
}

}  // extern "C"
