//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Shared helpers of the clipping kernels (clip2d.cpp, clip3d.cpp): generic
// evaluation of classification polynomials with the forward-error filter and
// exact expansions, and validation of the batched C ABI arguments.
//
// Input domains: coordinates and normal components satisfy
// coordinate_in_domain(); halfspace offsets h (of degree two, like n . x)
// satisfy weight_in_domain().  Every value entering a classification is then
// an integer multiple of 2^-172 (coordinates, normals) or 2^-292 (offsets), so
// every polynomial of the clipping kernels (at most six coordinate factors,
// or the equivalent weighted degree) is an integer multiple of 2^-1032 with
// magnitude below 2^740: expansion arithmetic neither underflows nor
// overflows and every sign is exact.
#pragma once

#include <cmath>
#include <cstdint>

#include "expansion.hpp"
#include "filtered.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc::clip {

template <class T>
T lift(double x);
template <>
inline Approx lift<Approx>(double x) {
  return Approx::exact(x);
}
template <>
inline Expansion lift<Expansion>(double x) {
  return Expansion(x);
}

template <class T>
T diff(double a, double b);
template <>
inline Approx diff<Approx>(double a, double b) {
  return Approx::exact(a) - Approx::exact(b);
}
template <>
inline Expansion diff<Expansion>(double a, double b) {
  return Expansion::difference(a, b);
}

template <class T>
T mul(double a, double b);
template <>
inline Approx mul<Approx>(double a, double b) {
  return Approx::exact(a) * Approx::exact(b);
}
template <>
inline Expansion mul<Expansion>(double a, double b) {
  return Expansion::product(a, b);
}

// Exact sign of a polynomial written once as a generic lambda
// `[&]<class T>() { ... }`: filtered evaluation first, expansions on doubt.
template <class F>
int exact_sign(const F& expression) {
  const int filtered = expression.template operator()<Approx>().certified_sign();
  if (filtered != 2) {
    return filtered;
  }
  return expression.template operator()<Expansion>().sign();
}

// Sign of a filtered value with an exact fallback producing an Expansion.
template <class F>
int exact_sign(const Approx& filtered, const F& exact) {
  const int sign = filtered.certified_sign();
  if (sign != 2) {
    return sign;
  }
  return exact().sign();
}

// Constructions (never decisions) use filtered values whose certified
// relative error is below this tolerance; otherwise the estimate of the exact
// expansion replaces them.
inline constexpr double kConstructionTolerance = 0x1p-44;

inline bool accurate(const Approx& value) {
  return value.bound <= kConstructionTolerance * std::fabs(value.value);
}

template <class F>
double construction_value(const Approx& filtered, const F& exact) {
  return accurate(filtered) ? filtered.value : exact().estimate();
}

// a b - c d with Kahan's fused multiply-add algorithm (error below 2 ulp).
inline double difference_of_products(double a, double b, double c, double d) {
  const double w = c * d;
  const double e = std::fma(-c, d, w);
  const double f = std::fma(a, b, -w);
  return f + e;
}

// Dot product accumulated with error-free transformations (Ogita-Rump-Oishi
// Dot2): as accurate as if computed in twice the working precision.
inline double dot3(const double* a, const double* b) {
  double sum = 0.0;
  double error = 0.0;
  for (int k = 0; k < 3; ++k) {
    double product, product_error, next, sum_error;
    two_product(a[k], b[k], product, product_error);
    two_sum(sum, product, next, sum_error);
    sum = next;
    error += product_error + sum_error;
  }
  return sum + error;
}

// Neumaier-compensated accumulation of measure contributions.
class CompensatedSum {
 public:
  void add(double value) {
    double next, error;
    two_sum(sum_, value, next, error);
    sum_ = next;
    compensation_ += error;
  }
  double value() const { return sum_ + compensation_; }

 private:
  double sum_ = 0.0;
  double compensation_ = 0.0;
};

// Sentinel returned by classifications of descriptor/plane combinations that
// the construction never produces.
inline constexpr int kInvalidSide = 2;

inline int32_t check_finite(const double* values, int64_t count) {
  for (int64_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return PHX_MC_NONFINITE_INPUT;
    }
  }
  return PHX_MC_OK;
}

inline int32_t check_coordinates(const double* values, int64_t count) {
  for (int64_t index = 0; index < count; ++index) {
    if (!coordinate_in_domain(values[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  return PHX_MC_OK;
}

inline int32_t check_offsets(const double* values, int64_t count) {
  for (int64_t index = 0; index < count; ++index) {
    if (!weight_in_domain(values[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  return PHX_MC_OK;
}

// Item validation of halfspaces {x : n . x <= h}: non-finite values, then the
// exact domain, then zero normals.
inline int32_t check_halfspaces(const double* normals, const double* offsets, int32_t planes,
                                int dimension) {
  const int64_t components = static_cast<int64_t>(planes) * dimension;
  int32_t status = check_finite(normals, components);
  if (status == PHX_MC_OK) {
    status = check_finite(offsets, planes);
  }
  if (status == PHX_MC_OK) {
    status = check_coordinates(normals, components);
  }
  if (status == PHX_MC_OK) {
    status = check_offsets(offsets, planes);
  }
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int32_t plane = 0; plane < planes; ++plane) {
    bool zero = true;
    for (int k = 0; k < dimension; ++k) {
      zero = zero && normals[static_cast<int64_t>(plane) * dimension + k] == 0.0;
    }
    if (zero) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}

inline bool counts_in_range(const int32_t* counts, int64_t count, int32_t capacity) {
  for (int64_t index = 0; index < count; ++index) {
    if (counts[index] < 0 || counts[index] > capacity) {
      return false;
    }
  }
  return true;
}

// Call-level validation of an axis-aligned box.
inline int32_t check_box(const double* lower, const double* upper, int dimension) {
  if (lower == nullptr || upper == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  int32_t status = check_finite(lower, dimension);
  if (status == PHX_MC_OK) {
    status = check_finite(upper, dimension);
  }
  if (status == PHX_MC_OK) {
    status = check_coordinates(lower, dimension);
  }
  if (status == PHX_MC_OK) {
    status = check_coordinates(upper, dimension);
  }
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int k = 0; k < dimension; ++k) {
    if (!(lower[k] < upper[k])) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }
  return PHX_MC_OK;
}

inline double clamp(double value, double lower, double upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

// Optional simplex partition of one clipped item: up to `capacity` simplices of
// (dimension + 1) points each, written row-major to `simplices`; `count`
// receives the number written.
struct SimplexOutput {
  int32_t capacity = 0;
  double* simplices = nullptr;
  int32_t count = 0;
};

}  // namespace phx::mc::clip
