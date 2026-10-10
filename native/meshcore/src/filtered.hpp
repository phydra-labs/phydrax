//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Dynamic forward-error filter for polynomial predicates without a published
// static error bound (weighted power tests, clipping classifications).  Each
// value carries a rigorous absolute bound on its distance from the exact real
// result of the same expression on the exact inputs.  A sign is certified when
// |value| > bound; otherwise callers re-evaluate the expression with exact
// arithmetic owned by the caller.
//
// Bound propagation for round-to-nearest binary64 with unit roundoff u = 2^-53:
//   fl(a + b):  |err| <= ea + eb + u/(1-u) |fl(a + b)|
//   fl(a * b):  |err| <= |a| eb + |b| ea + ea eb
//                         + u/(1-u) |fl(a * b)| + eta
// where eta = 2^-1074 covers gradual underflow of a product. Every operation
// on nonnegative bounds rounds outward independently. Multiplicative inflation
// cannot recover contributions already lost to subnormal rounding.
#pragma once

#include <cmath>
#include <limits>

namespace phx::mc {

struct Approx {
  double value = 0.0;
  double bound = 0.0;

  static Approx exact(double x) { return {x, 0.0}; }

  // Certified sign in {-1, 0, 1}, or 2 when the filter cannot decide.
  int certified_sign() const {
    if (!(std::isfinite(value) && std::isfinite(bound))) {
      return 2;
    }
    if (value > bound) {
      return 1;
    }
    if (-value > bound) {
      return -1;
    }
    if (value == 0.0 && bound == 0.0) {
      return 0;
    }
    return 2;
  }
};

namespace detail {
// An upward-rounded value of u/(1-u), not just u: the error is expressed in
// terms of the rounded result rather than the exact pre-rounding value.
inline constexpr double kRoundedResultError = 0x1.0000000000001p-53;
inline constexpr double kUnderflowEta = 0x1p-1074;

inline double add_upper(double a, double b) {
  if (a == 0.0) {
    return b;
  }
  if (b == 0.0) {
    return a;
  }
  return std::nextafter(a + b, std::numeric_limits<double>::infinity());
}

inline double multiply_upper(double a, double b) {
  return a == 0.0 || b == 0.0
             ? 0.0
             : std::nextafter(a * b, std::numeric_limits<double>::infinity());
}
}  // namespace detail

inline Approx operator+(Approx a, Approx b) {
  const double value = a.value + b.value;
  const double bound = detail::add_upper(
      detail::add_upper(a.bound, b.bound),
      detail::multiply_upper(detail::kRoundedResultError, std::fabs(value)));
  return {value, bound};
}

inline Approx operator-(Approx a, Approx b) {
  const double value = a.value - b.value;
  const double bound = detail::add_upper(
      detail::add_upper(a.bound, b.bound),
      detail::multiply_upper(detail::kRoundedResultError, std::fabs(value)));
  return {value, bound};
}

inline Approx operator-(Approx a) { return {-a.value, a.bound}; }

inline Approx operator*(Approx a, Approx b) {
  const double value = a.value * b.value;
  const double rounding = detail::add_upper(
      detail::multiply_upper(detail::kRoundedResultError, std::fabs(value)),
      detail::kUnderflowEta);
  if (a.bound == 0.0 && b.bound == 0.0) {
    return {value, rounding};
  }
  const double propagated = detail::add_upper(
      detail::add_upper(detail::multiply_upper(std::fabs(a.value), b.bound),
                        detail::multiply_upper(std::fabs(b.value), a.bound)),
      detail::multiply_upper(a.bound, b.bound));
  const double bound = detail::add_upper(propagated, rounding);
  return {value, bound};
}

}  // namespace phx::mc
