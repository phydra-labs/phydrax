//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Internal (non-ABI) access to the exact polytope clipper of clip3d.cpp for
// kernels that need the clipped polytope itself rather than its moments.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "bounded_memory.hpp"
#include "expansion.hpp"
#include "filtered.hpp"

namespace phx::mc {

// A clipped convex polytope: compacted vertices (3 per vertex), face loops
// counterclockwise seen from outside as CSR (face_offsets, face_vertices) and
// face labels: -(1 + k) for the face of the base tetrahedron opposite its
// vertex k, and the halfspace index (>= 0) otherwise.  Faces are sorted by
// label; each loop starts at its smallest vertex index.  An empty (or
// zero-measure) intersection has no vertices and no faces.
struct ClippedPolytope {
  NativeVector<double> vertices;
  NativeVector<int32_t> face_offsets;
  NativeVector<int32_t> face_labels;
  NativeVector<int32_t> face_vertices;
  double volume = 0.0;
  double moment[3] = {0.0, 0.0, 0.0};

  [[nodiscard]] bool empty() const { return face_labels.empty(); }
};

// Reusable workspace clipping a positively oriented tetrahedron (4, 3) by
// halfspaces {x : n . x <= h}.  Every vertex classification is exact; the
// inputs follow the coordinate/weight domains of phx_mc_polyhedron_clip_moments.
// Returns PHX_MC_OK, PHX_MC_DEGENERATE_INPUT for a flat or negatively oriented
// tetrahedron, or the clipper's validation/capacity status.
struct ExactPowerCoordinates {
  const Approx* approximate;
  const Expansion* exact;
};

class TetrahedronClipper {
 public:
  TetrahedronClipper();
  ~TetrahedronClipper();
  TetrahedronClipper(const TetrahedronClipper&) = delete;
  TetrahedronClipper& operator=(const TetrahedronClipper&) = delete;

  int32_t clip(const double* tetrahedron, const double* normals, const double* offsets,
               int32_t plane_count, int32_t vertex_capacity, ClippedPolytope& out);

  // Source power planes are evaluated from the authored binary64 sites and
  // weights in exact arithmetic, never from rounded normal/offset coefficients.
  int32_t clip_power(const double* tetrahedron, const double* sites, const double* weights,
                     int32_t site, const int32_t* neighbors,
                     int32_t plane_count, int32_t vertex_capacity, ClippedPolytope& out,
                     const ExactPowerCoordinates* exact_coordinates = nullptr);

  // Exact sign at the implicit vertex of the last successful clip, never at
  // its rounded exported coordinates. Positive is inside, zero is incident.
  int vertex_base_side(int32_t vertex, int32_t face);
  int vertex_power_side(int32_t vertex, const double* site, double weight,
                        const double* neighbor, double neighbor_weight,
                        const ExactPowerCoordinates* exact_site = nullptr,
                        const ExactPowerCoordinates* exact_neighbor = nullptr);
  static int point_power_side(const double* point, const double* site, double weight,
                              const double* neighbor, double neighbor_weight,
                              const ExactPowerCoordinates* exact_site = nullptr,
                              const ExactPowerCoordinates* exact_neighbor = nullptr);

 private:
  struct Impl;
  int32_t finish_clip(int32_t vertex_capacity, ClippedPolytope& out);
  NativeUniquePtr<Impl> impl_;
};

}  // namespace phx::mc
