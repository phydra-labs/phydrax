// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

namespace phx::mc {

// Construct from the source line's primitive integer-direction lattice. The
// desired position selects its dominant physical coordinate; the returned
// coordinate is physical, not a binary affine fraction. The point is exactly
// representable and inside the current open source interval.
bool construct_exact_line_point(const double* origin, const double* endpoint, const double* first,
                                const double* second, const double* desired, double* position,
                                double* coordinate);

}  // namespace phx::mc
