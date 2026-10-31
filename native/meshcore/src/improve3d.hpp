//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Source-preserving interior reconnection, insertion, collapse, and
// relocation improvement with unmet-quality reporting.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "refine3d.hpp"
#include "tet_mesh.hpp"

namespace phx::mc {

// Each cell scores the worse of its dihedral and radius-edge target fractions,
// floored to multiples of kTetScoreQuantum. Every edit (reconnection,
// collapse, relocation, interior vertex insertion) is accepted only if its
// affected cells' worst-first sorted scores strictly improve, with the
// determinant floor and source constraints as hard gates; the circumradius
// size target is a hard gate too, except for an edit that removes a
// degenerate cell (below the determinant floor or scoring 0), whose repair
// takes precedence. That one order makes the run terminate. Final checks of
// the radius-edge, dihedral, and determinant targets, not the score, set
// status, and unmet cells are reported individually.
inline constexpr double kTetScoreQuantum = 0x1p-20;

struct TetQualityRequest {
  double radius_edge_bound;
  double minimum_dihedral_degrees;
  double minimum_relative_determinant;
  // Circumradius size target per local vertex slot (0: unconstrained); set by
  // improve_mesh from the mesh's retained size field.
  const std::vector<double>* slot_sizes = nullptr;
};

// `vertex_sizes` (one per current vertex, snapshot order) replaces the mesh's
// retained size field when nonempty; otherwise the retained field applies.
std::int32_t improve_mesh(TetMesh& mesh, const TetQualityRequest& request,
                          std::span<const double> vertex_sizes,
                          std::int64_t& inserted_vertices,
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
bool relocation_tetrahedron_shape(const double* const points[4],
                                  const RelocationMetricCache& cache,
                                  TetrahedronShape& shape) noexcept;
double relocation_minimum_dihedral_degrees(
    const double* const points[4], const RelocationMetricCache& cache) noexcept;
double relocation_relative_determinant(
    const double* const points[4], const RelocationMetricCache& cache) noexcept;
double minimum_dihedral_degrees(const double* const points[4]) noexcept;
double relative_determinant(const double* const points[4]) noexcept;
}  // namespace geometry

}  // namespace phx::mc
