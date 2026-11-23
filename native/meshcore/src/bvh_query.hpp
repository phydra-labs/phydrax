//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// The single native owner of traversals over a caller-built packed BVH
// (phx_mc_bvh_tree_view): view validation shared with the pair stream, and the
// batched point queries (nearest faces with their closest feature, pseudonormal
// signed distance, hierarchical winding numbers, affine tetrahedron location).
// Kernels assume validated arguments and gradual round-to-nearest on entry;
// the C entries in bvh_query_capi.cpp establish both. Each query writes its
// original row, independent of worker count, and the kernel returns pool status.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "phydrax_meshcore.h"

namespace phx::mc::bvh {

// Whether the view is addressable, its reachable nodes form a rooted tree, and
// its leaf items index its item bounds.
bool tree_valid(const phx_mc_bvh_tree_view* tree, int32_t dimension);

struct TriangleSoup {
  const double* vertices;
  const int32_t* faces;
};

struct NearestOutput {
  int32_t* faces;
  double* distance_squared;
  double* closest;
  int8_t* features;
};

int32_t triangle_nearest(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup, int64_t count,
                         const double* points, int32_t k, const NearestOutput& output,
                         int32_t parallelism);

int32_t triangle_signed_distance(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                                 const double* feature_normals, int64_t count, const double* points,
                                 double* distance, int32_t* faces, double* closest,
                                 int32_t parallelism);

struct ClosingFans {
  const int64_t* offsets;
  const int32_t* edges;
};

int32_t triangle_winding_exact(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                               const ClosingFans& fans, int64_t count, const double* points,
                               double* values, int32_t parallelism);

int32_t triangle_winding_dipole(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                                const double* dipoles, double opening_angle, int64_t count,
                                const double* points, double* values, int32_t parallelism);

// The validated reachable nodes in preorder. A rejected node's escape
// skips its entire subtree without a per-query stack.
struct alignas(64) LocateNode {
  double lower[3];
  double upper[3];
  std::size_t escape;
  int64_t leaf;
};

struct LocateTree {
  std::unique_ptr<LocateNode[]> nodes;
  std::size_t size = 0;
};

LocateTree prepare_locate_tree(const phx_mc_bvh_tree_view* tree, double tolerance);

struct LocateItems {
  int64_t item_count;
  int64_t leaf_width;
  const int64_t* leaf_items;
  const double* item_min;
  const double* item_max;
};

struct LocateOutput {
  int32_t* cells;
  double* reference;
  double* residual;
  double* condition;
  int32_t* status;
  int32_t* containing;
  int32_t* candidates;
  double* candidate_reference;
};

int32_t tetrahedron_locate(const LocateItems& tree, const LocateTree& nodes,
                           const double* vertices, const int32_t* cells,
                           const uint8_t* cell_mask, double tolerance,
                           int64_t count, const double* points, int32_t capacity,
                           const LocateOutput& output, int32_t parallelism);

}  // namespace phx::mc::bvh
