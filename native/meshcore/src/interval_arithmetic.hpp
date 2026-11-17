// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <cmath>
#include <limits>

#include "rounding.hpp"

namespace phx::mc::interval {

using geometry::next_down;
using geometry::next_up;

// IEEE 754-2019 minimum/maximum: NaN propagates and -0 orders below +0.
inline double ieee_minimum(double a, double b) noexcept {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  if (a < b) return a;
  if (b < a) return b;
  return std::signbit(a) ? a : b;
}

inline double ieee_maximum(double a, double b) noexcept {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  if (a > b) return a;
  if (b > a) return b;
  return std::signbit(a) ? b : a;
}

constexpr double kVeltkampSplitter = 134217729.0;
constexpr double kSplitMagnitudeLimit = 0x1p996;
constexpr double kProductMagnitudeFloor = 0x1p-969;

// Dekker TwoProduct in the exact window, or an exact zero factor.
inline bool product_is_exact(double first, double second, double product) noexcept {
  const double first_scaled = kVeltkampSplitter * first;
  const double first_high = first_scaled - (first_scaled - first);
  const double first_low = first - first_high;
  const double second_scaled = kVeltkampSplitter * second;
  const double second_high = second_scaled - (second_scaled - second);
  const double second_low = second - second_high;
  const double error = ((first_high * second_high - product) + first_high * second_low +
                        first_low * second_high) +
                       first_low * second_low;
  const bool window = std::isfinite(product) && std::fabs(first) < kSplitMagnitudeLimit &&
                      std::fabs(second) < kSplitMagnitudeLimit &&
                      std::fabs(product) >= kProductMagnitudeFloor;
  const bool zero_factor = (first == 0.0 && std::isfinite(second)) ||
                           (second == 0.0 && std::isfinite(first));
  return zero_factor || (window && error == 0.0);
}

// Set-based interval product; exact endpoints are not widened.
inline void interval_product(double a_lo, double a_hi, double b_lo, double b_hi, double& lower,
                             double& upper) noexcept {
  const double left[4] = {a_lo, a_lo, a_hi, a_hi};
  const double right[4] = {b_lo, b_hi, b_lo, b_hi};
  double low = std::numeric_limits<double>::infinity();
  double high = -std::numeric_limits<double>::infinity();
  for (int pair = 0; pair < 4; ++pair) {
    const double l = left[pair];
    const double r = right[pair];
    const bool zero_unbounded = (l == 0.0 && std::isinf(r)) || (std::isinf(l) && r == 0.0);
    const double product = zero_unbounded ? 0.0 : l * r;
    if (std::isnan(product)) {
      lower = -std::numeric_limits<double>::infinity();
      upper = std::numeric_limits<double>::infinity();
      return;
    }
    const bool exact = zero_unbounded || product_is_exact(l, r, product);
    low = ieee_minimum(low, exact ? product : next_down(product));
    high = ieee_maximum(high, exact ? product : next_up(product));
  }
  const bool first_nonnegative = a_lo >= 0.0;
  const bool first_nonpositive = a_hi <= 0.0;
  const bool second_nonnegative = b_lo >= 0.0;
  const bool second_nonpositive = b_hi <= 0.0;
  if ((first_nonnegative && second_nonnegative) || (first_nonpositive && second_nonpositive)) {
    low = ieee_maximum(low, 0.0);
  }
  if ((first_nonnegative && second_nonpositive) || (first_nonpositive && second_nonnegative)) {
    high = ieee_minimum(high, 0.0);
  }
  lower = low;
  upper = high;
}

inline bool sum_is_exact(double first, double second, double total) noexcept {
  const double partner = total - first;
  const double error = (first - (total - partner)) + (second - partner);
  return std::isfinite(total) && error == 0.0;
}

inline void round_out_inexact(double low, double high, bool low_exact, bool high_exact,
                              double& lower, double& upper) noexcept {
  const bool invalid = std::isnan(low) || std::isnan(high);
  const double widened_low = next_down(invalid ? -std::numeric_limits<double>::infinity() : low);
  const double widened_high = next_up(invalid ? std::numeric_limits<double>::infinity() : high);
  lower = low_exact && !invalid ? low : widened_low;
  upper = high_exact && !invalid ? high : widened_high;
}

inline void interval_sum(double a_lo, double a_hi, double second_lower, double second_upper,
                         bool subtract, double& lower, double& upper) noexcept {
  const double b_lo = subtract ? -second_upper : second_lower;
  const double b_hi = subtract ? -second_lower : second_upper;
  const double low = subtract ? a_lo - second_upper : a_lo + b_lo;
  const double high = subtract ? a_hi - second_lower : a_hi + b_hi;
  double out_low = 0.0;
  double out_high = 0.0;
  round_out_inexact(low, high, sum_is_exact(a_lo, b_lo, low), sum_is_exact(a_hi, b_hi, high),
                    out_low, out_high);
  if (!subtract && !std::isnan(low) && !std::isnan(high)) {
    if (a_lo >= 0.0 && b_lo >= 0.0) out_low = ieee_maximum(out_low, 0.0);
    if (a_hi <= 0.0 && b_hi <= 0.0) out_high = ieee_minimum(out_high, 0.0);
  }
  lower = out_low;
  upper = out_high;
}

}  // namespace phx::mc::interval
