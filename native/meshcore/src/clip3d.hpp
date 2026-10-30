//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Internal access to the exact polytope clipper for callers that need the
// clipped topology or exact implicit-vertex classifications.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "expansion.hpp"
#include "filtered.hpp"

namespace phx::mc {

// A clipped convex polytope: compacted vertices (3 per vertex), face loops
// counterclockwise seen from outside as CSR (face_offsets, face_vertices) and
// labels: -(1 + k) for the base tetrahedron face opposite vertex k, and the
// halfspace index (>= 0) otherwise.
struct ClippedPolytope {
  std::vector<double> vertices;
  std::vector<int32_t> face_offsets;
  std::vector<int32_t> face_labels;
  std::vector<int32_t> face_vertices;
  double volume = 0.0;
  double moment[3] = {0.0, 0.0, 0.0};

  [[nodiscard]] bool empty() const { return face_labels.empty(); }
};

// Exact source coordinates supplied by a caller whose power sites are
// themselves constructions; otherwise authored binary64 sites are lifted.
struct ExactPowerCoordinates {
  const Approx* approximate;
  const Expansion* exact;
};

// Exact topology and source-plane witnesses of a reference tetrahedron clip.
struct ReferenceTetrahedronClip {
  std::vector<int32_t> supporting_planes;
  std::vector<int32_t> incident_offsets;
  std::vector<int32_t> incident_planes;
  std::vector<int32_t> face_offsets;
  std::vector<int32_t> face_planes;
  std::vector<int32_t> face_vertices;
};

class TetrahedronClipper {
 public:
  TetrahedronClipper();
  ~TetrahedronClipper();
  TetrahedronClipper(const TetrahedronClipper&) = delete;
  TetrahedronClipper& operator=(const TetrahedronClipper&) = delete;

  int32_t clip(const double* tetrahedron, const double* normals, const double* offsets,
               int32_t plane_count, ClippedPolytope& out);

  int32_t clip_power(const double* tetrahedron, const double* sites, const double* weights,
                     int32_t site, const int32_t* neighbors, int32_t plane_count,
                     ClippedPolytope& out,
                     const ExactPowerCoordinates* exact_coordinates = nullptr);

  // Exact sign at a clipped implicit vertex, never at its rounded carrier.
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
  int32_t finish_clip(ClippedPolytope& out);
  std::unique_ptr<Impl> impl_;
};

}  // namespace phx::mc
