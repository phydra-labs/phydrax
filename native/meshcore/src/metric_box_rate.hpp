// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cmath>
#include <limits>

#include "deterministic_float.hpp"
#include "geometry3.hpp"

namespace phx::mc::metric {

// Integral of exp(s*u), u in [0,1], including the near-constant limit.
inline double integrated_exponential(double s) {
  if (std::abs(s) < 1e-4) return 1.0 + s * (0.5 + s * (1.0 / 6 + s / 24));
  return (deterministic::exp(s) - 1.0) / s;
}

// Prepare barycentric and scalar P1 gradients in the caller's coordinates.
// Only used for numerical proposals. Final interpolation uses exact carriers.
inline bool prepare_box_rate(const std::array<std::array<double, 3>, 4>& p,
                             const std::array<double, 4>& rate,
                             std::array<std::array<double, 3>, 4>& g,
                             std::array<double, 3>& gradient) {
  const auto a = geometry::difference(p[1], p[0]);
  const auto b = geometry::difference(p[2], p[0]);
  const auto c = geometry::difference(p[3], p[0]);
  g[1] = geometry::cross(b, c);
  g[2] = geometry::cross(c, a);
  g[3] = geometry::cross(a, b);
  const double det = geometry::dot(a.data(), g[1].data());
  const double scale = geometry::length(a) * geometry::length(b) * geometry::length(c);
  if (!(std::abs(det) > 64 * std::numeric_limits<double>::epsilon() * scale)) return false;
  for (int k = 0; k < 3; ++k) {
    for (int i = 1; i < 4; ++i) g[i][k] /= det;
    g[0][k] = -g[1][k] - g[2][k] - g[3][k];
    gradient[k] = (rate[1] - rate[0]) * g[1][k] + (rate[2] - rate[0]) * g[2][k] +
                  (rate[3] - rate[0]) * g[3][k];
  }
  return true;
}

}  // namespace phx::mc::metric
