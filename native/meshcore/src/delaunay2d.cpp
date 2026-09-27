//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: 2D Delaunay and regular triangulations (incremental Bowyer-Watson
// with ghost triangles and symbolic perturbation, see triangulation2d.hpp).
#include <cstdint>
#include <memory>
#include <vector>

#include "capi_guard.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "triangulation2d.hpp"

namespace {

int32_t triangulate_2d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, int64_t max_triangles, phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || point_count > phx::mc::kMaxMeshPoints ||
      (point_count > 0 && points == nullptr) ||
      (weighted && point_count > 0 && weights == nullptr) || max_triangles < 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  int32_t status = phx::mc::validate_points(points, point_count, 2, weights);
  if (status != PHX_MC_OK) {
    return status;
  }
  auto result = std::make_unique<phx_mc_mesh>();
  result->dimension = 2;
  result->input_point_count = point_count;
  if (point_count == 0) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  phx::mc::Triangulation2D triangulation;
  triangulation.reset(points, point_count, weights);
  status = phx::mc::build_delaunay_2d(triangulation, points, point_count, weights, max_triangles,
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
  phx::mc::canonicalize_cells(*result);
  *mesh = result.release();
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_delaunay_2d(int64_t point_count, const double* points, int64_t max_triangles,
                           phx_mc_mesh** mesh) {
  return phx::mc::guarded(
      [&] { return triangulate_2d(point_count, points, nullptr, false, max_triangles, mesh); });
}

int32_t phx_mc_regular_2d(int64_t point_count, const double* points, const double* weights,
                          int64_t max_triangles, phx_mc_mesh** mesh) {
  return phx::mc::guarded(
      [&] { return triangulate_2d(point_count, points, weights, true, max_triangles, mesh); });
}

}  // extern "C"
