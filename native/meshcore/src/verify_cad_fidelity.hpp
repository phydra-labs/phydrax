// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <cstdint>

#include "cad_geometry.hpp"

namespace phx::mc::cad {

struct FidelityTriangleResult {
  double upper;
  int32_t status;
  int32_t reason;
  int32_t subdivisions;
};

FidelityTriangleResult fidelity_triangle(const CadGeometry& geometry, int32_t face,
                                         const double* vertices, const double* uv,
                                         double tolerance, int32_t max_depth,
                                         IntervalEvaluationWorkspace& workspace);

}  // namespace phx::mc::cad
