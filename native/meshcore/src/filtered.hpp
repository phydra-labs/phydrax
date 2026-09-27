//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Dynamic forward-error filter for polynomial predicates without a published
// static error bound (weighted power tests, clipping classifications).  Each
// value carries a rigorous absolute bound on its distance from the exact real
// result of the same expression on the exact inputs.  A sign is certified when
// |value| > bound; otherwise callers re-evaluate the expression with exact
// expansion arithmetic.
//
// Bound propagation for round-to-nearest binary64 with unit roundoff u = 2^-53:
//   fl(a + b):  |err| <= ea + eb + u |fl(a + b)|
//   fl(a * b):  |err| <= |a| eb + |b| ea + ea eb + u |fl(a * b)| + eta
// where eta = 2^-1074 covers gradual underflow of a product.  The bound itself
// is computed in floating point and inflated by (1 + 2^-50) per operation,
// which dominates the at most three roundings of the bound expression.
#pragma once

#include <cmath>

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
inline constexpr double kUnitRoundoff = 0x1p-53;
inline constexpr double kBoundInflation = 1.0 + 0x1p-50;
inline constexpr double kUnderflowEta = 0x1p-1074;
}  // namespace detail

inline Approx operator+(Approx a, Approx b) {
  const double value = a.value + b.value;
  const double bound =
      (a.bound + b.bound + detail::kUnitRoundoff * std::fabs(value)) * detail::kBoundInflation;
  return {value, bound};
}

inline Approx operator-(Approx a, Approx b) {
  const double value = a.value - b.value;
  const double bound =
      (a.bound + b.bound + detail::kUnitRoundoff * std::fabs(value)) * detail::kBoundInflation;
  return {value, bound};
}

inline Approx operator-(Approx a) { return {-a.value, a.bound}; }

inline Approx operator*(Approx a, Approx b) {
  const double value = a.value * b.value;
  if (a.bound == 0.0 && b.bound == 0.0) {
    return {value,
            (detail::kUnitRoundoff * std::fabs(value) + detail::kUnderflowEta) *
                detail::kBoundInflation};
  }
  const double propagated =
      std::fabs(a.value) * b.bound + std::fabs(b.value) * a.bound + a.bound * b.bound;
  const double bound =
      (propagated + detail::kUnitRoundoff * std::fabs(value) + detail::kUnderflowEta) *
      detail::kBoundInflation * detail::kBoundInflation;
  return {value, bound};
}

}  // namespace phx::mc
