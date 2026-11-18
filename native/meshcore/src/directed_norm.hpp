// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Certified bounds of the Euclidean norm of a 3-vector from bounds of the
// magnitudes of its components: binary64 products, sums and one square root
// in a fixed order, each moved one value outward. No platform libm, so the
// bounds are the same doubles on every platform. Requires round-to-nearest
// with gradual underflow (GradualNearestScope of floating_point_mode.hpp).
#pragma once

#include <cmath>

#include "rounding.hpp"

namespace phx::mc::geometry {

// At most |v| for every v with |v[d]| >= least[d] >= 0.
inline double norm_lower(const double least[3]) noexcept {
  double sum = 0.0;
  for (int d = 0; d < 3; ++d) sum = next_down(sum + next_down(least[d] * least[d]));
  return sum > 0.0 ? next_down(std::sqrt(sum)) : 0.0;
}

// At least |v| for every v with |v[d]| <= most[d].
inline double norm_upper(const double most[3]) noexcept {
  double sum = 0.0;
  for (int d = 0; d < 3; ++d) sum = next_up(sum + next_up(most[d] * most[d]));
  return next_up(std::sqrt(sum));
}

}  // namespace phx::mc::geometry
