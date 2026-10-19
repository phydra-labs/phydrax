// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Adjacent binary64 values directed toward the two infinities.
#pragma once

#include <cmath>
#include <limits>

namespace phx::mc::geometry {

inline double next_up(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity());
}

inline double next_down(double value) {
  return std::nextafter(value, -std::numeric_limits<double>::infinity());
}

}  // namespace phx::mc::geometry
