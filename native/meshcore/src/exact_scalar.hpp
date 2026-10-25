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

// Shewchuk's column form uses three additions of signed cofactors. The
// adaptive error bounds depend on this reduction, not the first-row expansion.
template <class T>
inline T det3_columns(const T* x, const T* y, const T* z) {
  return z[0] * (x[1] * y[2] - x[2] * y[1]) + z[1] * (x[2] * y[0] - x[0] * y[2]) +
         z[2] * (x[0] * y[1] - x[1] * y[0]);
}

// Scalar users own their admission. The optional callback admits each exact
// residual before it is formed; the counter preserves existing residual tariffs.
struct QuotientRoundingWork final {
  std::int64_t* residuals = nullptr;
  void* context = nullptr;
  void (*admit)(void*, std::int64_t) = nullptr;

  void charge(std::int64_t operations) const {
    if (admit != nullptr) admit(context, operations);
  }
  void residual() const {
    charge(3);  // Lift the candidate, multiply, and subtract.
    if (residuals != nullptr) ++*residuals;
  }
};

// Correct a finite candidate using exact residuals. The residual sum decides
// the midpoint, including ties to even; overflow retains a finite carrier.
inline double nearest_quotient(const Dyadic& numerator, const Dyadic& denominator,
                               double x, QuotientRoundingWork work = {}) {
  if (denominator.sign() == 0) {
    throw std::invalid_argument("An exact quotient requires a nonzero denominator.");
  }
  work.residual();
  Dyadic residual = numerator - Dyadic(x) * denominator;
  while (residual.sign() != 0) {
    const int direction = residual.sign() * denominator.sign();
    const double next = std::nextafter(
        x, direction > 0 ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity());
    if (!std::isfinite(next)) break;
    work.residual();
    Dyadic next_residual = numerator - Dyadic(next) * denominator;
    if (next_residual.sign() == 0) return next;
    if (next_residual.sign() == residual.sign()) {
      x = next;
      residual = std::move(next_residual);
      continue;
    }
    work.charge(1);
    const int closer = (residual + next_residual).sign() * residual.sign();
    if (closer > 0 || (closer == 0 && (std::bit_cast<std::uint64_t>(next) & 1u) == 0)) {
      x = next;
    }
    break;
  }
  return x;
}

// Equal rational values share the correctly rounded carrier. Values beyond
// binary64 saturate only this carrier, never a purported enclosing endpoint.
inline double rounded_quotient(const Dyadic& numerator, const Dyadic& denominator,
                               QuotientRoundingWork work = {}) {
  if (denominator.sign() == 0) {
    throw std::invalid_argument("An exact quotient requires a nonzero denominator.");
  }
  work.charge(6);
  std::int64_t exponent = 0, denominator_exponent = 0;
  const double fraction = numerator.fraction(exponent);
  const double denominator_fraction = denominator.fraction(denominator_exponent);
  const auto shift = std::clamp<std::int64_t>(
      exponent - denominator_exponent, std::numeric_limits<int>::min(),
      std::numeric_limits<int>::max());
  const double candidate = std::ldexp(fraction / denominator_fraction, static_cast<int>(shift));
  const double initial = std::isfinite(candidate)
                             ? candidate
                             : std::copysign(std::numeric_limits<double>::max(), candidate);
  return nearest_quotient(numerator, denominator, initial, work);
}

struct QuotientBounds final {
  double lower;
  double upper;
};

inline QuotientBounds directed_quotient_bounds(const Dyadic& numerator,
                                              const Dyadic& denominator,
                                              QuotientRoundingWork work = {}) {
  const double carrier = rounded_quotient(numerator, denominator, work);
  work.residual();
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
