//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary of the deterministic binary64 kernels. Every entry
// establishes round-to-nearest with gradual underflow for its duration and
// evaluates its rows independently, in order.
#include "capi_guard.hpp"
#include "deterministic_float.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

#include <cstddef>
#include <cstdint>

namespace {

using phx::mc::addressable;

template <class Kernel>
int32_t map_scalars(int64_t count, const double* values, double* results,
                    Kernel kernel) noexcept {
  if (!addressable(count, 1, sizeof(double)) ||
      (count > 0 && (values == nullptr || results == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t row = 0; row < count; ++row) results[row] = kernel(values[row]);
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_deterministic_float_self_test(int32_t* failed_check) {
  if (failed_check == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *failed_check = 0;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  *failed_check = static_cast<int32_t>(phx::mc::deterministic::self_test());
  return *failed_check == 0 ? PHX_MC_OK : PHX_MC_INTERNAL_ERROR;
}

int32_t phx_mc_deterministic_exp(int64_t count, const double* values, double* results) {
  return map_scalars(count, values, results, phx::mc::deterministic::exp);
}

int32_t phx_mc_deterministic_log(int64_t count, const double* values, double* results) {
  return map_scalars(count, values, results, phx::mc::deterministic::log);
}

int32_t phx_mc_deterministic_pow_two_thirds(int64_t count, const double* values,
                                            double* results) {
  return map_scalars(count, values, results, phx::mc::deterministic::pow_two_thirds);
}

int32_t phx_mc_deterministic_symmetric_eigen3(int64_t count, const double* matrices,
                                              double* values, double* vectors,
                                              int8_t* converged) {
  if (!addressable(count, 9, sizeof(double)) ||
      (count > 0 && (matrices == nullptr || values == nullptr || vectors == nullptr ||
                     converged == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t row = 0; row < count; ++row) {
    phx::mc::deterministic::SymmetricEigen3 eigen;
    const bool done = phx::mc::deterministic::symmetric_eigen3(matrices + 9 * row, eigen);
    converged[row] = done ? 1 : 0;
    for (std::size_t index = 0; index < 3; ++index) {
      values[3 * row + static_cast<int64_t>(index)] = done ? eigen.values[index] : 0.0;
    }
    for (std::size_t index = 0; index < 9; ++index) {
      vectors[9 * row + static_cast<int64_t>(index)] = done ? eigen.vectors[index] : 0.0;
    }
  }
  return PHX_MC_OK;
}

}  // extern "C"
