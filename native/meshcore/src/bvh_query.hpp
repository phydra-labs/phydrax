//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// The single native owner of traversals over a caller-built packed BVH
// (phx_mc_bvh_tree_view): view validation shared with the pair stream, and the
// batched single-threaded point queries (nearest faces with their closest
// feature, pseudonormal signed distance, hierarchical winding numbers, affine
// tetrahedron location). Kernels assume validated arguments; the C entries in
// bvh_query_capi.cpp validate.
#pragma once

#include <cstdint>

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

void triangle_nearest(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup, int64_t count,
                      const double* points, int32_t k, const NearestOutput& output);

void triangle_signed_distance(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                              const double* feature_normals, int64_t count, const double* points,
                              double* distance, int32_t* faces, double* closest);

struct ClosingFans {
  const int64_t* offsets;
  const int32_t* edges;
};

void triangle_winding_exact(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                            const ClosingFans& fans, int64_t count, const double* points,
                            double* values);

void triangle_winding_dipole(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                             const double* dipoles, double opening_angle, int64_t count,
                             const double* points, double* values);

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

void tetrahedron_locate(const phx_mc_bvh_tree_view& tree, const double* vertices,
                        const int32_t* cells, const uint8_t* cell_mask, double tolerance,
                        int64_t count, const double* points, int32_t capacity,
                        const LocateOutput& output);

}  // namespace phx::mc::bvh
