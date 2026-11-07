// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Behavioral tests for outward interval enclosures of simplex quadratic forms.
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <initializer_list>

#include "check.hpp"
#include "floating_point_mode.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"

namespace {

using Matrix2 = std::array<double, 4>;

bool same(double first, double second) {
  return std::bit_cast<std::uint64_t>(first) == std::bit_cast<std::uint64_t>(second);
}

std::int32_t bounds(const double* q_lower, const double* q_upper, const double* l_lower,
                    const double* l_upper, double* lower, double* upper, std::int64_t count = 1,
                    std::int32_t dimension = 1, std::int32_t components = 1) {
  return phx_mc_simplex_quadratic_bounds(count, dimension, components, q_lower, q_upper,
                                         l_lower, l_upper, lower, upper);
}

void exact_and_transformed_coefficients() {
  const Matrix2 q{{1.0, 2.0, 2.0, 4.0}};
  const Matrix2 identity{{1.0, 0.0, 0.0, 1.0}};
  double lower = -99.0, upper = 99.0;
  PHX_CHECK(bounds(q.data(), q.data(), identity.data(), identity.data(), &lower, &upper) ==
            PHX_MC_OK);
  PHX_CHECK(lower == 1.0 && upper == 4.0);

  // The exact transformed matrix for L=[[3/4,1/4],[1/4,3/4]] is
  // [[25/16,35/16],[35/16,49/16]]. Outward rounding encloses every coefficient.
  const Matrix2 l{{0.75, 0.25, 0.25, 0.75}};
  lower = 99.0;
  upper = -99.0;
  PHX_CHECK(bounds(q.data(), q.data(), l.data(), l.data(), &lower, &upper) == PHX_MC_OK);
  PHX_CHECK(lower <= 25.0 / 16.0);
  PHX_CHECK(upper >= 49.0 / 16.0);
  for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const double row0[2] = {0.75 * (1.0 - t) + 0.25 * t,
                            0.25 * (1.0 - t) + 0.75 * t};
    const double row1[2] = {0.5, 0.5};
    const double* rows[2] = {row0, row1};
    for (int i = 0; i < 2; ++i) {
      for (int j = i; j < 2; ++j) {
        double value = 0.0;
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            value += rows[i][a] * q[a * 2 + b] * rows[j][b];
          }
        }
        PHX_CHECK(lower <= value && value <= upper);
      }
    }
  }

  // Interval-valued coefficients and barycentric rows contain the sampled
  // exact symmetric quadratic and rows; every transformed coefficient remains
  // inside the published interval.
  const Matrix2 q_lower{{0.9, 1.8, 1.8, 3.8}};
  const Matrix2 q_upper{{1.1, 2.2, 2.2, 4.2}};
  const Matrix2 l_lower{{0.5, 0.2, 0.2, 0.5}};
  const Matrix2 l_upper{{0.8, 0.5, 0.5, 0.8}};
  lower = 99.0;
  upper = -99.0;
  PHX_CHECK(bounds(q_lower.data(), q_upper.data(), l_lower.data(), l_upper.data(),
                  &lower, &upper) == PHX_MC_OK);
  const double sample_l[2][2] = {{0.8, 0.2}, {0.5, 0.5}};
  for (int i = 0; i < 2; ++i) {
    for (int j = i; j < 2; ++j) {
      double value = 0.0;
      for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
          value += sample_l[i][a] * q[a * 2 + b] * sample_l[j][b];
        }
      }
      PHX_CHECK(lower <= value && value <= upper);
    }
  }

  // The identity tetrahedron transform retains all four diagonal coefficients
  // and the zero off-diagonals in its Bernstein hull.
  const std::array<double, 16> q3{{1.0, 0.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0}};
  std::array<double, 16> l3{};
  for (int i = 0; i < 4; ++i) l3[4 * i + i] = 1.0;
  double lower3 = -1.0, upper3 = -1.0;
  PHX_CHECK(bounds(q3.data(), q3.data(), l3.data(), l3.data(), &lower3, &upper3,
                  1, 3, 1) == PHX_MC_OK);
  PHX_CHECK(lower3 == 0.0 && upper3 == 1.0);
}

void unbounded_coefficients_and_determinism() {
  const double infinity = std::numeric_limits<double>::infinity();
  const Matrix2 q_lower{{-infinity, -infinity, -infinity, -infinity}};
  const Matrix2 q_upper{{infinity, infinity, infinity, infinity}};
  const Matrix2 identity{{1.0, 0.0, 0.0, 1.0}};
  double lower = 0.0, upper = 0.0;
  PHX_CHECK(bounds(q_lower.data(), q_upper.data(), identity.data(), identity.data(),
                  &lower, &upper) == PHX_MC_OK);
  PHX_CHECK(lower == -infinity && upper == infinity);

  const Matrix2 q{{1.0, 2.0, 2.0, 4.0}};
  const Matrix2 l{{0.75, 0.25, 0.25, 0.75}};
  double second_lower = 0.0, second_upper = 0.0;
  PHX_CHECK(bounds(q.data(), q.data(), l.data(), l.data(), &lower, &upper) == PHX_MC_OK);
  PHX_CHECK(bounds(q.data(), q.data(), l.data(), l.data(), &second_lower, &second_upper) ==
            PHX_MC_OK);
  PHX_CHECK(same(lower, second_lower) && same(upper, second_upper));
}

void invalid_inputs_leave_outputs_unchanged() {
  const Matrix2 identity{{1.0, 0.0, 0.0, 1.0}};
  Matrix2 q{{1.0, 2.0, 3.0, 4.0}};
  double lower = -123.0, upper = 456.0;
  PHX_CHECK(bounds(q.data(), q.data(), identity.data(), identity.data(), &lower, &upper) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(lower == -123.0 && upper == 456.0);

  q = {{1.0, 2.0, 2.0, 4.0}};
  Matrix2 q_reversed{{1.0, 2.0, 2.0, 4.0}};
  q_reversed[0] = 2.0;
  PHX_CHECK(bounds(q_reversed.data(), q.data(), identity.data(), identity.data(), &lower, &upper) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(lower == -123.0 && upper == 456.0);

  Matrix2 bad_rows{{0.75, 0.75, 0.0, 1.0}};
  PHX_CHECK(bounds(q.data(), q.data(), bad_rows.data(), bad_rows.data(), &lower, &upper) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(lower == -123.0 && upper == 456.0);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  Matrix2 nonfinite{{1.0, nan, nan, 4.0}};
  PHX_CHECK(bounds(nonfinite.data(), nonfinite.data(), identity.data(), identity.data(),
                  &lower, &upper) == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(lower == -123.0 && upper == 456.0);

  PHX_CHECK(phx_mc_simplex_quadratic_bounds(1, 0, 1, q.data(), q.data(), identity.data(),
                                             identity.data(), &lower, &upper) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_simplex_quadratic_bounds(1, 1, 1, q.data(), q.data(), identity.data(),
                                             identity.data(), nullptr, &upper) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_simplex_quadratic_bounds(std::numeric_limits<std::int64_t>::max(), 1, 1,
                                             q.data(), q.data(), identity.data(), identity.data(),
                                             &lower, &upper) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(lower == -123.0 && upper == 456.0);

  PHX_CHECK(phx_mc_simplex_quadratic_bounds(0, 1, 1, nullptr, nullptr, nullptr, nullptr,
                                             nullptr, nullptr) == PHX_MC_OK);
}

void caller_rounding_modes() {
  const Matrix2 q{{1.0, 2.0, 2.0, 4.0}};
  const Matrix2 l{{0.75, 0.25, 0.25, 0.75}};
  double expected_lower = 0.0, expected_upper = 0.0;
  PHX_CHECK(bounds(q.data(), q.data(), l.data(), l.data(), &expected_lower, &expected_upper) ==
            PHX_MC_OK);
  const auto initial_mode = phx::mc::floating_point_mode();
  const int initial_rounding = std::fegetround();
  const std::array<int, 3> rounding_modes{{FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}};
  for (int rounding : rounding_modes) {
    phx::mc::set_floating_point_mode(phx::mc::kGradualNearestMode);
    PHX_CHECK(std::fesetround(rounding) == 0);
    const auto installed = phx::mc::floating_point_mode();
    double lower = 0.0, upper = 0.0;
    PHX_CHECK(bounds(q.data(), q.data(), l.data(), l.data(), &lower, &upper) == PHX_MC_OK);
    PHX_CHECK(same(lower, expected_lower) && same(upper, expected_upper));
    PHX_CHECK(phx::mc::floating_point_mode() == installed);
    PHX_CHECK(std::fegetround() == rounding);
  }
  phx::mc::set_floating_point_mode(initial_mode);
  PHX_CHECK(std::fesetround(initial_rounding) == 0);
  PHX_CHECK(phx::mc::floating_point_mode() == initial_mode);
}

// Subnormal coefficients keep the round-to-nearest enclosure under flushing
// and directed caller modes, and a negative subnormal barycentric endpoint is
// rejected before any evaluation in every mode.
void subnormal_inputs_under_hazards() {
  const Matrix2 q{{0x1p-1060, -0x1p-1070, -0x1p-1070, 0x1p-1062}};
  const Matrix2 l{{0.75, 0.25, 0.25, 0.75}};
  double expected_lower = 99.0, expected_upper = -99.0;
  PHX_CHECK(bounds(q.data(), q.data(), l.data(), l.data(), &expected_lower, &expected_upper) ==
            PHX_MC_OK);
  // Exact transformed coefficients lie in [0x1p-1063, 0x1p-1061 + 0x1p-1063].
  PHX_CHECK(expected_lower <= 0x1p-1062 && expected_upper >= 0x1p-1061);
  const Matrix2 identity{{1.0, 0.0, 0.0, 1.0}};
  const Matrix2 negative_lower{{1.0, -0x1p-1074, 0.0, 1.0}};
  double lower = -123.0, upper = 456.0;
  PHX_CHECK(bounds(identity.data(), identity.data(), negative_lower.data(), identity.data(),
                   &lower, &upper) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(lower == -123.0 && upper == 456.0);
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    double hazard_lower = 0.0, hazard_upper = 0.0;
    PHX_CHECK(bounds(q.data(), q.data(), l.data(), l.data(), &hazard_lower, &hazard_upper) ==
              PHX_MC_OK);
    PHX_CHECK(same(hazard_lower, expected_lower) && same(hazard_upper, expected_upper));
    double rejected_lower = -123.0, rejected_upper = 456.0;
    PHX_CHECK(bounds(identity.data(), identity.data(), negative_lower.data(), identity.data(),
                     &rejected_lower, &rejected_upper) == PHX_MC_INVALID_INPUT);
    PHX_CHECK(rejected_lower == -123.0 && rejected_upper == 456.0);
  });
}

}  // namespace

int main() {
  exact_and_transformed_coefficients();
  unbounded_coefficients_and_determinism();
  invalid_inputs_leave_outputs_unchanged();
  caller_rounding_modes();
  subnormal_inputs_under_hazards();
  return phx::mc::test::finish("test_quadratic_bounds");
}
