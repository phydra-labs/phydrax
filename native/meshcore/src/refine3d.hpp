//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained tetrahedral refinement with explicit finite unresolved reporting.
#pragma once

#include <cstdint>
#include <span>

#include "tet_mesh.hpp"

namespace phx::mc {

std::int32_t refine_mesh(TetMesh& mesh, double radius_edge_bound,
                         std::span<const double> vertex_sizes,
                         std::int64_t& inserted_vertices,
                         std::int64_t& unmet_cells);

namespace geometry {
struct TetrahedronShape {
  double circumradius = 0.0;
  double shortest_edge = 0.0;
  double radius_edge_ratio = 0.0;
};

bool tetrahedron_shape(const double* const points[4], TetrahedronShape& shape) noexcept;
}  // namespace geometry

}  // namespace phx::mc
