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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "spatial_sort.hpp"
#include "triangulation3d.hpp"

namespace phx::mc {

int32_t triangulate_3d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, bool canonical, phx_mc_mesh** mesh, int32_t parallelism) {
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
  const std::vector<int32_t> order =
      deduplicated_insertion_order(points, point_count, 3, active_weights, vertex_map);

  // Vertex k of the triangulation is input point order[k], stored in
  // insertion order so that walks and cavities touch nearby memory. Its
  // input index orders the symbolic perturbation and labels the output cells.
  const std::size_t count = order.size();
  std::vector<double> local_points(3 * count);
  std::vector<double> local_weights(weighted ? count : 0);
  for (std::size_t k = 0; k < count; ++k) {
    const double* point = points + 3 * static_cast<std::size_t>(order[k]);
    std::copy_n(point, 3, local_points.begin() + static_cast<std::ptrdiff_t>(3 * k));
    if (weighted) local_weights[k] = weights[order[k]];
  }
  std::vector<int32_t> local_order(count);
  std::iota(local_order.begin(), local_order.end(), 0);
  std::vector<char> alive(count, 1);
  auto result = std::make_unique<phx_mc_mesh>();
  {
    Triangulation3D triangulation(local_points.data(),
                                  weighted ? local_weights.data() : nullptr,
                                  static_cast<int64_t>(count));
    triangulation.set_identities(order.data());
    status = triangulation.build(local_order, alive, canonical ? parallelism : 1);
    if (status != PHX_MC_OK) {
      return status;
    }
    triangulation.collect_cells(result->cells);
  }
  std::vector<char> input_alive(static_cast<std::size_t>(point_count), 0);
  for (std::size_t k = 0; k < count; ++k) {
    input_alive[static_cast<std::size_t>(order[k])] = alive[k];
  }
  for (int32_t& target : vertex_map) {
    if (target >= 0 && input_alive[static_cast<std::size_t>(target)] == 0) {
      target = -1;
    }
  }
  result->dimension = 3;
  result->input_point_count = point_count;
  result->points.assign(points, points + 3 * point_count);
  result->vertex_map = std::move(vertex_map);
  if (canonical) {
    status = canonicalize_cells(*result, parallelism);
    if (status != PHX_MC_OK) return status;
  }
  *mesh = result.release();
  return PHX_MC_OK;
}

}  // namespace phx::mc

extern "C" {

int32_t phx_mc_delaunay_3d(int64_t point_count, const double* points, phx_mc_mesh** mesh,
                           int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    if (parallelism < 1) return PHX_MC_INVALID_ARGUMENT;
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::triangulate_3d(point_count, points, nullptr, false, true, mesh, parallelism);
  });
}

int32_t phx_mc_regular_3d(int64_t point_count, const double* points, const double* weights,
                          phx_mc_mesh** mesh, int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    if (parallelism < 1) return PHX_MC_INVALID_ARGUMENT;
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::triangulate_3d(point_count, points, weights, true, true, mesh, parallelism);
  });
}

int32_t phx_mc_triangulate(int32_t dimension, int64_t point_count, const double* points,
                           const double* weights, uint32_t options, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    if (mesh != nullptr) *mesh = nullptr;
    if ((dimension != 2 && dimension != 3) ||
        (options & ~PHX_MC_TRIANGULATION_CANONICAL) != 0U) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const bool canonical = (options & PHX_MC_TRIANGULATION_CANONICAL) != 0U;
    const bool weighted = weights != nullptr;
    return dimension == 2
               ? phx::mc::triangulate_2d(point_count, points, weights, weighted, canonical, mesh)
               : phx::mc::triangulate_3d(point_count, points, weights, weighted, canonical, mesh,
                                         1);
  });
}

}  // extern "C"
