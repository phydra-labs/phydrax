// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Scalar lifting, determinant expansion, and exact dyadic quotient rounding.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "dyadic.hpp"
#include "expansion.hpp"
#include "filtered.hpp"

namespace phx::mc::exact {

template <class T>
inline T lift(double value) {
  if constexpr (std::is_same_v<T, Approx>) {
    return Approx::exact(value);
  } else {
    return T(value);
  }
}

// Keep the first-row +, -, + expansion. Shewchuk's column/cofactor kernels
// and compensated binary64 determinants have different reduction orders.
template <class T>
inline T det3(const T& a0, const T& a1, const T& a2, const T& b0, const T& b1, const T& b2,
              const T& c0, const T& c1, const T& c2) {
  return a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0);
}

template <class T>
inline T det3(const T* a, const T* b, const T* c) {
  return det3(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}

template <class T>
inline T det3(const T (&rows)[3][4], int i, int j, int k) {
  return det3(rows[0][i], rows[0][j], rows[0][k], rows[1][i], rows[1][j], rows[1][k], rows[2][i],
              rows[2][j], rows[2][k]);
}
// Convert the existing exact expansion without rounding its components' sum.
inline Dyadic to_dyadic(const Expansion& value) {
  if (value.size() == 0) return Dyadic();
  if (value.size() == 1) return Dyadic(value.data()[0]);
  Dyadic result;
  for (int component = 0; component < value.size(); ++component) {
    result = result + Dyadic(value.data()[component]);
  }
  return result;
}

// Shewchuk's column form uses three additions of signed cofactors. The
// adaptive error bounds depend on this reduction, not the first-row expansion.
template <class T>
inline T det3_columns(const T* x, const T* y, const T* z) {
  return z[0] * (x[1] * y[2] - x[2] * y[1]) + z[1] * (x[2] * y[0] - x[0] * y[2]) +
         z[2] * (x[0] * y[1] - x[1] * y[0]);
}

// Correct a finite candidate with exact residuals; the residual sign decides
// which adjacent float lies on the quotient and an exact midpoint ties to even.
inline double nearest_quotient(const Dyadic& numerator, const Dyadic& denominator, double x) {
  if (denominator.sign() == 0) {
    throw std::invalid_argument("An exact quotient requires a nonzero denominator.");
  }
  Dyadic residual = numerator - Dyadic(x) * denominator;
  while (residual.sign() != 0) {
    const int direction = residual.sign() * denominator.sign();
    const double next = std::nextafter(
        x, direction > 0 ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity());
    if (!std::isfinite(next)) break;
    Dyadic next_residual = numerator - Dyadic(next) * denominator;
    if (next_residual.sign() == 0) return next;
    if (next_residual.sign() == residual.sign()) {
      x = next;
      residual = std::move(next_residual);
      continue;
    }
    const int closer = (residual + next_residual).sign() * residual.sign();
    if (closer > 0 || (closer == 0 && (std::bit_cast<std::uint64_t>(next) & 1u) == 0)) x = next;
    break;
  }
  return x;
}

// Equal rational values share the correctly rounded carrier. Values beyond
// binary64 saturate only this carrier, never a purported enclosing endpoint.
inline double rounded_quotient(const Dyadic& numerator, const Dyadic& denominator) {
  if (denominator.sign() == 0) {
    throw std::invalid_argument("An exact quotient requires a nonzero denominator.");
  }
  // Exact zero has one geometric identity, independent of denominator sign.
  // Do not canonicalize rounded zero for nonzero negative underflow.
  if (numerator.sign() == 0) return 0.0;
  std::int64_t exponent = 0;
  std::int64_t denominator_exponent = 0;
  const double fraction = numerator.fraction(exponent);
  const double denominator_fraction = denominator.fraction(denominator_exponent);
  const auto shift = std::clamp<std::int64_t>(
      exponent - denominator_exponent, std::numeric_limits<int>::min(),
      std::numeric_limits<int>::max());
  const double candidate = std::ldexp(fraction / denominator_fraction, static_cast<int>(shift));
  const double initial = std::isfinite(candidate)
                             ? candidate
                             : std::copysign(std::numeric_limits<double>::max(), candidate);
  return nearest_quotient(numerator, denominator, initial);
}

struct QuotientBounds final {
  double lower;
  double upper;
};

inline QuotientBounds directed_quotient_bounds(const Dyadic& numerator, const Dyadic& denominator) {
  const double carrier = rounded_quotient(numerator, denominator);
  const int side = (numerator - Dyadic(carrier) * denominator).sign() * denominator.sign();
  if (side < 0) {
    return {std::nextafter(carrier, -std::numeric_limits<double>::infinity()), carrier};
  }
  if (side > 0) {
    return {carrier, std::nextafter(carrier, std::numeric_limits<double>::infinity())};
  }
  return {carrier, carrier};
}

}  // namespace phx::mc::exact
