// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "interval_arithmetic.hpp"
#include "parallel.hpp"
#include "wide_integer.hpp"
#include "phydrax_meshcore.h"

namespace {
bool overlaps(const double* first, std::size_t first_bytes,
              const double* second, std::size_t second_bytes) noexcept {
  const auto a = reinterpret_cast<std::uintptr_t>(first);
  const auto b = reinterpret_cast<std::uintptr_t>(second);
  return a <= b ? b - a < first_bytes : a - b < second_bytes;
}
}  // namespace

// Every exact common-degree control is a linear combination of the basis
// controls. Outward intervals for each factor, product, and sum therefore
// contain it; the Bernstein convex-hull property then bounds the image. The
// summation order is fixed independently for each cell, component, and control.
extern "C" int32_t phx_mc_interval_basis_bounds(
    int64_t count, int64_t basis_count, int64_t controls, int64_t components,
    const double* basis_lower, const double* basis_upper,
    const double* coefficient_lower, const double* coefficient_upper,
    double* result_lower, double* result_upper, double* inner_lower, double* inner_upper,
    int32_t workers) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    int64_t output_count = 0;
    if (count < 0 || basis_count < 1 || controls < 1 || components < 1 || workers < 1 ||
        !phx::mc::checked_multiply(count, components, output_count) ||
        !phx::mc::addressable(basis_count, controls, sizeof(double)) ||
        !phx::mc::addressable(1, output_count, sizeof(double)) ||
        !phx::mc::addressable(output_count, basis_count, sizeof(double)))
      return PHX_MC_INVALID_ARGUMENT;
    if (count == 0) return PHX_MC_OK;
    if (!basis_lower || !basis_upper || !coefficient_lower || !coefficient_upper ||
        !result_lower || !result_upper || !inner_lower || !inner_upper)
      return PHX_MC_INVALID_ARGUMENT;
    const auto basis_bytes = static_cast<std::size_t>(basis_count) *
                             static_cast<std::size_t>(controls) * sizeof(double);
    const auto output_bytes = static_cast<std::size_t>(output_count) * sizeof(double);
    const auto coefficient_bytes = static_cast<std::size_t>(output_count) *
                                   static_cast<std::size_t>(basis_count) * sizeof(double);
    const std::array<const double*, 4> inputs{
        basis_lower, basis_upper, coefficient_lower, coefficient_upper};
    const std::array<std::size_t, 4> input_bytes{
        basis_bytes, basis_bytes, coefficient_bytes, coefficient_bytes};
    const std::array<double*, 4> outputs{
        result_lower, result_upper, inner_lower, inner_upper};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      for (std::size_t j = 0; j < inputs.size(); ++j)
        if (overlaps(outputs[i], output_bytes, inputs[j], input_bytes[j]))
          return PHX_MC_INVALID_ARGUMENT;
      for (std::size_t j = 0; j < i; ++j)
        if (overlaps(outputs[i], output_bytes, outputs[j], output_bytes))
          return PHX_MC_INVALID_ARGUMENT;
    }
    constexpr double inf = std::numeric_limits<double>::infinity();
    return phx::mc::parallel_for(static_cast<std::size_t>(count),
        phx::mc::ParallelRequest{16, static_cast<uint32_t>(workers)},
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          const phx::mc::GradualNearestScope worker_mode;
          if (!worker_mode.established()) return PHX_MC_INTERNAL_ERROR;
          for (std::size_t cell = begin; cell < end; ++cell) {
            for (int64_t component = 0; component < components; ++component) {
              double lower = inf, upper = -inf, inner_lo = inf, inner_hi = -inf;
              for (int64_t control = 0; control < controls; ++control) {
                double sum_lower = 0.0, sum_upper = 0.0;
                for (int64_t basis = 0; basis < basis_count; ++basis) {
                  const auto bi = basis * controls + control;
                  const auto ci = (cell * basis_count + basis) * components + component;
                  const double bl = basis_lower[bi], bu = basis_upper[bi];
                  const double cl = coefficient_lower[ci], cu = coefficient_upper[ci];
                  if (!std::isfinite(bl) || !std::isfinite(bu) || bl > bu ||
                      !std::isfinite(cl) || !std::isfinite(cu) || cl > cu) {
                    sum_lower = -inf; sum_upper = inf; break;
                  }
                  if (cl == 0.0 && cu == 0.0) continue;
                  double pl, pu, sl, su;
                  phx::mc::interval::interval_product(bl, bu, cl, cu, pl, pu);
                  phx::mc::interval::interval_sum(sum_lower, sum_upper, pl, pu, false, sl, su);
                  sum_lower = sl; sum_upper = su;
                }
                if (!std::isfinite(sum_lower) || !std::isfinite(sum_upper)) {
                  lower = -inf; upper = inf; inner_lo = inf; inner_hi = -inf; break;
                }
                lower = std::min(lower, sum_lower);
                upper = std::max(upper, sum_upper);
                inner_lo = std::min(inner_lo, sum_upper);
                inner_hi = std::max(inner_hi, sum_lower);
              }
              const auto output = cell * components + component;
              result_lower[output] = lower;
              result_upper[output] = upper;
              inner_lower[output] = inner_lo;
              inner_upper[output] = inner_hi;
            }
          }
          return PHX_MC_OK;
        });
  });
}
