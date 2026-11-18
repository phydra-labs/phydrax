//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary of the portable angles (angle.hpp). Every entry
// establishes round-to-nearest with gradual underflow for its duration and
// evaluates its rows independently, in order.
#include "angle.hpp"
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

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

int32_t phx_mc_angle_sine(int64_t count, const double* values, double* results) {
  return map_scalars(count, values, results, phx::mc::angle::sine);
}

int32_t phx_mc_angle_cosine(int64_t count, const double* values, double* results) {
  return map_scalars(count, values, results, phx::mc::angle::cosine);
}

int32_t phx_mc_angle_arcsine(int64_t count, const double* values, double* results) {
  return map_scalars(count, values, results, phx::mc::angle::arcsine);
}

int32_t phx_mc_angle_arccosine(int64_t count, const double* values, double* results) {
  return map_scalars(count, values, results, phx::mc::angle::arccosine);
}

int32_t phx_mc_angle_between(int64_t count, const double* across, const double* along,
                             double* results) {
  if (!addressable(count, 1, sizeof(double)) ||
      (count > 0 && (across == nullptr || along == nullptr || results == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  for (int64_t row = 0; row < count; ++row) {
    results[row] = phx::mc::angle::between(across[row], along[row]);
  }
  return PHX_MC_OK;
}

}  // extern "C"
