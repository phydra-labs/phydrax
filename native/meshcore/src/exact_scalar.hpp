// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Scalar lifting and first-row determinant expansion shared by exact kernels.
#pragma once

#include <type_traits>

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

}  // namespace phx::mc::exact
