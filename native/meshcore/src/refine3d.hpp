//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained tetrahedral refinement with explicit finite unresolved reporting.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>

#include "tet_mesh.hpp"

namespace phx::mc {

std::int32_t refine_mesh(TetMesh& mesh, double radius_edge_bound,
                         std::span<const double> vertex_sizes,
                         std::int64_t& inserted_vertices,
                         std::int64_t& unmet_cells);

// Circumradius size target of a cell from its corners' targets: their mean,
// or +inf when any corner is unconstrained (target <= 0). Refinement and
// improvement share this rule.
inline double cell_size_target(const std::array<double, 4>& corner_sizes) noexcept {
  double scale = 0.0;
  for (double size : corner_sizes) {
    if (!(size > 0.0)) return std::numeric_limits<double>::infinity();
    scale = std::max(scale, size);
  }
  // Normalized by the largest target so the sum cannot overflow.
  double normalized_sum = 0.0;
  for (double size : corner_sizes) normalized_sum += size / scale;
  return std::min(scale, scale * (0.25 * normalized_sum));
}

namespace geometry {
struct TetrahedronShape {
  std::array<double, 3> circumcenter{};
  double circumradius = 0.0;
  double shortest_edge = 0.0;
  double radius_edge_ratio = 0.0;
};

bool tetrahedron_shape(const double* const points[4], TetrahedronShape& shape) noexcept;
}  // namespace geometry

}  // namespace phx::mc
