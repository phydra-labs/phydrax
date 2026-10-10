// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <cstdint>

namespace phx::mc {

// Construct from the original binary source line's primitive integer-direction
// lattice. The desired position selects its dominant physical coordinate; the
// returned coordinate is physical, not a binary affine fraction. Every accepted
// point is exactly representable and inside the current open source interval.
bool construct_exact_line_point(
    const double* origin, const double* endpoint, const double* first,
    const double* second, const double* desired, double* position,
    double* coordinate, bool (*spend)(void*, std::int64_t), void* context);

}  // namespace phx::mc
