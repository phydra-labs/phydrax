// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Rigorous binary64 screen in front of the exact quadratic tetrahedron
// validity proof (S2 design B.8 step 3).
#pragma once

#include <cstdint>

namespace phx::mc {

// True only if the exact path of phx_mc_quadratic_tetrahedron_validity decides
// the cell (30 node scalars, (3,4,10) derivative weights) VALID at its root;
// false means "not decided here", never "invalid". Requires finite nodes and
// round-to-nearest with gradual underflow (the entry's GradualNearestScope).
bool quadratic_validity_screen(const double* nodes, const std::int32_t* derivative_weights,
                               double relative_floor) noexcept;

}  // namespace phx::mc
