//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Source-preserving interior collapse/relocation improvement with unmet-quality reporting.
#pragma once

#include <array>
#include <cstdint>

#include "tet_mesh.hpp"

namespace phx::mc {

// Radius-edge and dihedral targets contribute equally to a normalized
// satisfaction sum; the weaker component breaks ties. This deterministic score
// guides local search, but final per-criterion checks—not the score—set status.
// Determinant validity and source constraints remain hard gates. Each accepted
// edit is one generation-checked source-preserving batch.
struct TetQualityRequest {
  double radius_edge_bound;
  double minimum_dihedral_degrees;
  double minimum_relative_determinant;
};

std::int32_t improve_mesh(TetMesh& mesh, const TetQualityRequest& request,
                          std::int64_t& collapsed_vertices,
                          std::int64_t& relocated_vertices,
                          std::int64_t& unmet_cells);

namespace geometry {
// Caches only operands independent of one moving corner; candidate pivot and
// reduction order, exact orientation, and quality comparisons are recomputed.
struct RelocationMetricCache {
  std::int32_t moved_corner = -1;
  std::int32_t opposite_face = -1;
  std::array<std::array<double, 4>, 3> fixed_shape_rows{};
  std::array<std::array<double, 4>, 4> fixed_edge_squared{};
  std::array<std::array<double, 3>, 3> fixed_relative_vectors{};
  std::array<std::array<double, 4>, 4> fixed_directed_edge_lengths{};
  std::array<double, 3> opposite_face_unit_normal{};
  bool opposite_face_valid = false;
  bool prepared = false;
};

bool prepare_relocation_metric_cache(const double* const points[4],
                                     std::int32_t moved_corner,
                                     RelocationMetricCache& cache) noexcept;
bool relocation_radius_edge_ratio(
    const double* const points[4], const RelocationMetricCache& cache,
    double& ratio) noexcept;
double relocation_minimum_dihedral_degrees(
    const double* const points[4], const RelocationMetricCache& cache) noexcept;
double relocation_relative_determinant(
    const double* const points[4], const RelocationMetricCache& cache) noexcept;
double minimum_dihedral_degrees(const double* const points[4]) noexcept;
double relative_determinant(const double* const points[4]) noexcept;
}  // namespace geometry

}  // namespace phx::mc
