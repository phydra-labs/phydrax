//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C entries of the BVH point queries: argument validation and dispatch.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "bvh_query.hpp"
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

struct phx_mc_affine_tet_locator {
  phx::mc::bvh::LocateTree nodes;
  phx::mc::bvh::LocateItems items;
  const double* vertices;
  const int32_t* cells;
  double tolerance;
};

namespace {

// Every index of `rows` x `width` entries lies in [0, bound).
bool indices_valid(const int32_t* indices, int64_t rows, int64_t width, int64_t bound) noexcept {
  const int64_t entries = rows * width;
  // As unsigned values, negative indices lie above the highest possible
  // nonnegative int32 index. Check one unsigned maximum per bounded block.
  const auto limit = static_cast<uint32_t>(std::min<int64_t>(
      bound, static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1));
  constexpr int64_t kBlock = 1024;
  for (int64_t first = 0; first < entries;) {
    const int64_t last = first + std::min(kBlock, entries - first);
    uint32_t largest = 0;
    for (; first < last; ++first) {
      largest = std::max(largest, static_cast<uint32_t>(indices[first]));
    }
    if (largest >= limit) return false;
  }
  return true;
}

bool query_points_valid(int64_t vertex_count, int64_t count, const double* points) {
  return vertex_count >= 0 && count >= 0 &&
         phx::mc::addressable(vertex_count, 3, sizeof(double)) &&
         phx::mc::addressable(count, 3, sizeof(double)) && (count == 0 || points != nullptr);
}

bool simplex_indices_valid(int64_t item_count, int64_t vertex_count, const double* vertices,
                           const int32_t* simplices, int64_t corners) {
  if (item_count == 0) return true;
  return vertices != nullptr && simplices != nullptr &&
         phx::mc::addressable(item_count, corners, sizeof(int32_t)) &&
         indices_valid(simplices, item_count, corners, vertex_count);
}

bool simplex_queries_valid(const phx_mc_bvh_tree_view* tree, int64_t vertex_count,
                           const double* vertices, const int32_t* simplices, int64_t corners,
                           int64_t count, const double* points) {
  return query_points_valid(vertex_count, count, points) && phx::mc::bvh::tree_valid(tree, 3) &&
         simplex_indices_valid(tree->item_count, vertex_count, vertices, simplices, corners);
}

}  // namespace

extern "C" {

int32_t phx_mc_bvh_query_nearest_triangles(
    const phx_mc_bvh_tree_view* tree, int64_t vertex_count, const double* vertices,
    const int32_t* faces, int64_t count, const double* points, int32_t k,
    int32_t* faces_out, double* distance_squared, double* closest, int8_t* features,
    int32_t parallelism) {
  if (parallelism < 1) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (k < 1 || !simplex_queries_valid(tree, vertex_count, vertices, faces, 3, count, points) ||
        !phx::mc::addressable(count, int64_t{k} * 3, sizeof(double)) ||
        (count > 0 && (faces_out == nullptr || distance_squared == nullptr ||
                       closest == nullptr || features == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    return phx::mc::bvh::triangle_nearest(
        *tree, {vertices, faces}, count, points, k, {faces_out, distance_squared, closest, features},
        parallelism);
  });
}

int32_t phx_mc_bvh_query_signed_distance(
    const phx_mc_bvh_tree_view* tree, int64_t vertex_count, const double* vertices,
    const int32_t* faces, const double* feature_normals, int64_t count,
    const double* points, double* distance, int32_t* faces_out, double* closest,
    int32_t parallelism) {
  if (parallelism < 1) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!simplex_queries_valid(tree, vertex_count, vertices, faces, 3, count, points) ||
        (tree->item_count > 0 &&
         (feature_normals == nullptr ||
          !phx::mc::addressable(tree->item_count, 3 * PHX_MC_BVH_FEATURE_COUNT, sizeof(double)))) ||
        (count > 0 && (distance == nullptr || faces_out == nullptr || closest == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    return phx::mc::bvh::triangle_signed_distance(
        *tree, {vertices, faces}, feature_normals, count, points, distance, faces_out, closest,
        parallelism);
  });
}

int32_t phx_mc_bvh_query_winding(
    const phx_mc_bvh_tree_view* tree, int64_t vertex_count, const double* vertices,
    const int32_t* faces, int32_t route, const int64_t* boundary_offsets,
    int64_t boundary_edge_count, const int32_t* boundary_edges, const double* dipoles,
    double opening_angle, int64_t count, const double* points, double* values,
    int32_t parallelism) {
  if (parallelism < 1) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!simplex_queries_valid(tree, vertex_count, vertices, faces, 3, count, points) ||
        (count > 0 && values == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::bvh::TriangleSoup soup{vertices, faces};
    switch (route) {
      case PHX_MC_WINDING_EXACT: {
        if (boundary_offsets == nullptr || boundary_edge_count < 0 ||
            !phx::mc::addressable(boundary_edge_count, 2, sizeof(int32_t)) ||
            (boundary_edge_count > 0 && boundary_edges == nullptr) || boundary_offsets[0] != 0 ||
            boundary_offsets[tree->node_count] != boundary_edge_count ||
            !indices_valid(boundary_edges, boundary_edge_count, 2, vertex_count)) {
          return PHX_MC_INVALID_ARGUMENT;
        }
        for (int64_t node = 0; node < tree->node_count; ++node) {
          if (boundary_offsets[node + 1] < boundary_offsets[node]) return PHX_MC_INVALID_ARGUMENT;
        }
        return phx::mc::bvh::triangle_winding_exact(
            *tree, soup, {boundary_offsets, boundary_edges}, count, points, values, parallelism);
      }
      case PHX_MC_WINDING_FAST_DIPOLE:
        if (dipoles == nullptr || !std::isfinite(opening_angle) || !(opening_angle > 0.0) ||
            !phx::mc::addressable(tree->node_count, 7, sizeof(double))) {
          return PHX_MC_INVALID_ARGUMENT;
        }
        return phx::mc::bvh::triangle_winding_dipole(
            *tree, soup, dipoles, opening_angle, count, points, values, parallelism);
      default:
        return PHX_MC_INVALID_ARGUMENT;
    }
  });
}

int32_t phx_mc_affine_tet_locator_prepare(
    const phx_mc_bvh_tree_view* tree, int64_t vertex_count, const double* vertices,
    const int32_t* cells, double tolerance, phx_mc_affine_tet_locator** locator) {
  if (locator == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!std::isfinite(tolerance) || tolerance < 0.0 ||
        !phx::mc::addressable(vertex_count, 3, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    auto nodes = phx::mc::bvh::prepare_locate_tree(tree, tolerance);
    if (!nodes.nodes || !simplex_indices_valid(tree->item_count, vertex_count, vertices, cells, 4)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *locator = new phx_mc_affine_tet_locator{
        std::move(nodes),
        {tree->item_count, tree->leaf_width, tree->leaf_items, tree->item_min, tree->item_max},
        vertices, cells, tolerance};
    return PHX_MC_OK;
  });
}

void phx_mc_affine_tet_locator_free(phx_mc_affine_tet_locator* locator) { delete locator; }

int32_t phx_mc_affine_tet_locator_query(
    const phx_mc_affine_tet_locator* locator, const uint8_t* cell_mask, int64_t count,
    const double* points, int32_t capacity, int32_t* cells_out, double* reference,
    double* residual, double* condition, int32_t* status, int32_t* containing,
    int32_t* candidates, double* candidate_reference, int32_t parallelism) {
  if (locator == nullptr || parallelism < 1) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (capacity < 1 || !phx::mc::addressable(count, 3, sizeof(double)) ||
        !phx::mc::addressable(count, int64_t{capacity} * 3, sizeof(double)) ||
        (count > 0 && (points == nullptr || cells_out == nullptr || reference == nullptr ||
                       residual == nullptr || condition == nullptr || status == nullptr ||
                       containing == nullptr ||
                       candidates == nullptr || candidate_reference == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    return phx::mc::bvh::tetrahedron_locate(
        locator->items, locator->nodes, locator->vertices, locator->cells, cell_mask,
        locator->tolerance, count, points, capacity,
        {cells_out, reference, residual, condition, status, containing, candidates,
         candidate_reference},
        parallelism);
  });
}

}  // extern "C"
