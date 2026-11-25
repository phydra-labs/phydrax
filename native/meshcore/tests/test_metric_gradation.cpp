//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "check.hpp"
#include "phydrax_meshcore.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace {

constexpr double infinite = std::numeric_limits<double>::infinity();

void test_scalar() {
  constexpr std::array initial{1.0, 1.0, 8.0, 8.0, 4.0, 0.5};
  constexpr int64_t edges[] = {2, 0, 0, 2, 1, 2, 0, 2, 2, 3, 1, 4, 4, 3, 5, 4};
  constexpr double steps[] = {2.5, 3.0, 1.0, 2.0, 0.5, 1.0, 9.0, infinite};
  std::array<double, initial.size()> graded{};
  int64_t stats[4];
  double violation;
  PHX_CHECK(phx_mc_metric_grade_physical(initial.size(), 1, initial.data(), std::size(steps),
                                         edges, steps, nullptr, graded.data(), stats,
                                         &violation) == PHX_MC_OK);
  // Row 0 wins the tie with row 1, and its first-endpoint incidences precede
  // the reversed parallel edge. These choices matter to update counts.
  PHX_CHECK(stats[0] == 6 && stats[1] == 5 && stats[2] == 3 && stats[3] == 1);
  PHX_CHECK(violation == 0.0);

  auto fixed_point = initial;
  for (std::size_t sweep = 0; sweep < initial.size(); ++sweep) {
    const auto previous = fixed_point;
    for (std::size_t e = 0; e < std::size(steps); ++e) {
      const int64_t a = edges[2 * e];
      const int64_t b = edges[2 * e + 1];
      fixed_point[a] = std::min(fixed_point[a], previous[b] + steps[e]);
      fixed_point[b] = std::min(fixed_point[b], previous[a] + steps[e]);
    }
  }
  PHX_CHECK(graded == fixed_point);
  PHX_CHECK(phx_mc_metric_grade_physical(initial.size(), 1, fixed_point.data(), std::size(steps),
                                         edges, steps, nullptr, graded.data(), stats,
                                         &violation) == PHX_MC_OK);
  PHX_CHECK(stats[0] == 6 && stats[1] == 0 && stats[2] == 0 && graded == fixed_point);
}

void test_diagonal() {
  constexpr double initial[] = {16.0, 3.1, 4.0, 1.0, 3.1, 4.0, 1.0, 3.1, 4.0};
  constexpr int64_t edges[] = {1, 2, 0, 1};
  constexpr double steps[] = {0.25, 0.25};
  double graded[std::size(initial)];
  int64_t stats[4];
  double violation;
  const double bounds[] = {1.0, 16.0, 10.0};
  PHX_CHECK(phx_mc_metric_grade_physical(3, 3, initial, 2, edges, steps, bounds, graded,
                                         stats, &violation) == PHX_MC_OK);
  PHX_CHECK(stats[0] == 9 && stats[1] == 2 && stats[2] == 2 && stats[3] == 1);
  PHX_CHECK(violation <= 4 * std::numeric_limits<double>::epsilon());
  PHX_CHECK(graded[0] == initial[0] && graded[3] == 4.0 && graded[6] == 16.0 / 9.0);
  for (int row = 0; row < 3; ++row) {
    PHX_CHECK(graded[3 * row + 1] == initial[3 * row + 1]);
    PHX_CHECK(graded[3 * row + 2] == initial[3 * row + 2]);
  }
  const double too_narrow[] = {1.0, 16.0, 1.1};
  PHX_CHECK(phx_mc_metric_grade_physical(3, 3, initial, 2, edges, steps, too_narrow, graded,
                                         stats, &violation) == PHX_MC_OK);
  PHX_CHECK(stats[3] == 0);
}

void test_invalid() {
  const int64_t good_edge[] = {0, 1};
  const int64_t bad_edge[] = {0, 2};
  const double initial[] = {1.0, 2.0};
  const double nonfinite[] = {1.0, infinite};
  const double nonpositive[] = {1.0, 0.0};
  const double good_step[] = {0.0};
  const double bad_step[] = {-1.0};
  const double nan_step[] = {std::numeric_limits<double>::quiet_NaN()};
  const double bad_bounds[] = {2.0, 1.0, 1.0};
  double graded[2];
  int64_t stats[4];
  double violation;
  PHX_CHECK(phx_mc_metric_grade_physical(2, 1, initial, 1, bad_edge, good_step, nullptr, graded,
                                         stats, &violation) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_grade_physical(2, 1, initial, 1, good_edge, bad_step, nullptr, graded,
                                         stats, &violation) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_grade_physical(2, 1, initial, 1, good_edge, nan_step, nullptr, graded,
                                         stats, &violation) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_grade_physical(2, 1, nonpositive, 0, nullptr, nullptr, nullptr, graded,
                                         stats, &violation) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_grade_physical(2, 1, nonfinite, 0, nullptr, nullptr, nullptr, graded,
                                         stats, &violation) == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(phx_mc_metric_grade_physical(2, 1, nullptr, 0, nullptr, nullptr, nullptr, graded,
                                         stats, &violation) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_grade_physical(1, 2, initial, 0, nullptr, nullptr, bad_bounds, graded,
                                         stats, &violation) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_metric_grade_physical(-1, 3, nullptr, 0, nullptr, nullptr, bad_bounds, nullptr,
                                         stats, &violation) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_metric_grade_physical(0, 1, nullptr, 0, nullptr, nullptr, nullptr, nullptr,
                                         stats, &violation) == PHX_MC_OK);
  PHX_CHECK(stats[0] == 0 && stats[1] == 0 && stats[2] == 0 && stats[3] == 1 && violation == 0.0);
}

}  // namespace

int main() {
  test_scalar();
  test_diagonal();
  test_invalid();
  return phx::mc::test::finish("test_metric_gradation");
}
