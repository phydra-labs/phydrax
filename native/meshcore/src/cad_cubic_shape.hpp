// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace phx::mc {

struct CadQuadraticInput;
struct CadCubicMesh;

struct CadCubicShape {
  std::array<std::int64_t, 2> unmet{};
  std::vector<std::int64_t> failing_cells;
  double minimum_scaled = std::numeric_limits<double>::infinity();
  double minimum_ratio = std::numeric_limits<double>::infinity();
};

// Numerical shape on the degree-six simplex lattice, independent of exact
// map validity and the source certificate. Empty cells means all cells.
CadCubicShape sample_cad_cubic_shape(const CadQuadraticInput& input,
                                    const CadCubicMesh& mesh,
                                    std::span<const std::int64_t> cells = {});

}  // namespace phx::mc
