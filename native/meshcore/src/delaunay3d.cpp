//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Incremental 3D Delaunay and regular triangulations (Bowyer-Watson).
//
// The triangulation of the convex hull is closed by ghost tetrahedra: every
// hull facet carries one tetrahedron whose remaining vertex is kInfinite.
// Every tetrahedron (v0, v1, v2, v3) is positively oriented: orient3d > 0 for
// finite ones, and for ghosts substituting a point strictly beyond the hull
// facet for the infinite vertex gives orient3d > 0.  n[k] is the neighbor
// across the facet opposite v[k]; the structure is always a closed
// pseudomanifold (every facet shared by exactly two tetrahedra).
//
// Conflicts are exact and index-ordered: a finite tetrahedron conflicts with p
// iff insphere_sos (power3d_sos for weighted points) > 0.  A ghost conflicts
// iff p lies strictly beyond its hull facet; when p is coplanar with the facet
// it conflicts iff its finite neighbor does (the neighbor's circumsphere, or
// orthosphere, restricted to the facet plane is the facet's circumcircle, so
// this is the limit of the perturbed predicate at the hull).  The triangulation
// is the regular triangulation of the symbolically perturbed lifting, hence
// the conflict region of p is connected and star-shaped from p, and the
// visibility walk is acyclic.
#include <cstdint>
#include <memory>
#include <vector>

#include "capi_guard.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "spatial_sort.hpp"
#include "triangulation3d.hpp"

namespace phx::mc {
namespace {



int32_t triangulate_3d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || !addressable(point_count, 3, sizeof(double)) ||
      !addressable(point_count, 1, sizeof(int32_t)) ||
      !addressable(point_count, 1, sizeof(double)) ||
      (point_count > 0 && (points == nullptr || (weighted && weights == nullptr)))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (point_count > kMaxMeshPoints) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const double* active_weights = weighted ? weights : nullptr;
  int32_t status = validate_points(points, point_count, 3, active_weights);
  if (status != PHX_MC_OK) {
    return status;
  }
  std::vector<int32_t> vertex_map;
  const std::vector<int32_t> representatives =
      deduplicate_points(points, point_count, 3, active_weights, vertex_map);
  const std::vector<int32_t> order = brio_hilbert_order(points, 3, representatives);

  std::vector<char> alive(static_cast<std::size_t>(point_count), 0);
  for (int32_t vertex : representatives) {
    alive[static_cast<std::size_t>(vertex)] = 1;
  }
  auto result = std::make_unique<phx_mc_mesh>();
  {
    Triangulation3D triangulation(points, active_weights, point_count);
    status = triangulation.build(order, alive);
    if (status != PHX_MC_OK) {
      return status;
    }
    triangulation.collect_cells(result->cells);
  }
  for (int32_t& target : vertex_map) {
    if (target >= 0 && alive[static_cast<std::size_t>(target)] == 0) {
      target = -1;
    }
  }
  result->dimension = 3;
  result->input_point_count = point_count;
  result->points.assign(points, points + 3 * point_count);
  result->vertex_map = std::move(vertex_map);
  canonicalize_cells(*result);
  *mesh = result.release();
  return PHX_MC_OK;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_delaunay_3d(int64_t point_count, const double* points, phx_mc_mesh** mesh) {
  return phx::mc::guarded(
      [&] { return phx::mc::triangulate_3d(point_count, points, nullptr, false, mesh); });
}

int32_t phx_mc_regular_3d(int64_t point_count, const double* points, const double* weights,
                          phx_mc_mesh** mesh) {
  return phx::mc::guarded(
      [&] { return phx::mc::triangulate_3d(point_count, points, weights, true, mesh); });
}

}  // extern "C"
