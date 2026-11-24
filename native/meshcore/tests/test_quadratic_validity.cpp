// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Behavioral tests for exact continuous validity of quadratic tetrahedral maps.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <random>
#include <vector>

#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"
#include "quadratic_validity_screen.hpp"

namespace {

constexpr std::int32_t kValid = 0;
constexpr std::int32_t kInvalid = 1;
constexpr std::int32_t kUnresolved = 2;

struct Input final {
  std::array<double, 30> nodes{};
  std::array<std::int32_t, 120> derivative_weights{};
  int next_node = 0;

  void derivative(int column, int power, const std::array<double, 3>& value,
                  std::int32_t weight = 1) {
    const int node = next_node++;
    for (int row = 0; row < 3; ++row) nodes[3 * node + row] = value[row];
    derivative_weights[(column * 4 + power) * 10 + node] = weight;
  }
};

struct Result final {
  std::int32_t status = -1;
  std::int32_t decision = 77;
  double lower = -123.0;
  double upper = 456.0;
};

Result call(const Input& input, double floor = 0.0) {
  Result result;
  result.status = phx_mc_quadratic_tetrahedron_validity(
      1, input.nodes.data(), input.derivative_weights.data(), floor, &result.decision,
      &result.lower, &result.upper);
  // Verdicts only (the binary64 screen may decide): same status and verdict.
  std::int32_t verdict = 77;
  const std::int32_t status = phx_mc_quadratic_tetrahedron_validity(
      1, input.nodes.data(), input.derivative_weights.data(), floor, &verdict, nullptr, nullptr);
  PHX_CHECK(status == result.status && verdict == result.decision);
  return result;
}

Input identity() {
  Input input;
  input.derivative(0, 0, {1.0, 0.0, 0.0});
  input.derivative(1, 0, {0.0, 1.0, 0.0});
  input.derivative(2, 0, {0.0, 0.0, 1.0});
  return input;
}

void affine_and_degenerate_maps() {
  const Result valid = call(identity());
  PHX_CHECK(valid.status == PHX_MC_OK);
  PHX_CHECK(valid.decision == kValid);
  PHX_CHECK(valid.lower == 1.0 && valid.upper == 1.0);

  Input inverted;
  inverted.derivative(0, 0, {-1.0, 0.0, 0.0});
  inverted.derivative(1, 0, {0.0, 1.0, 0.0});
  inverted.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result negative = call(inverted);
  PHX_CHECK(negative.status == PHX_MC_OK);
  PHX_CHECK(negative.decision == kInvalid);
  PHX_CHECK(negative.lower == -1.0 && negative.upper == -1.0);

  Input singular;
  singular.derivative(0, 1, {1.0, 0.0, 0.0});
  singular.derivative(1, 0, {0.0, 1.0, 0.0});
  singular.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result zero_corner = call(singular);
  PHX_CHECK(zero_corner.status == PHX_MC_OK);
  PHX_CHECK(zero_corner.decision == kInvalid);
  PHX_CHECK(zero_corner.lower == 0.0 && zero_corner.upper == 1.0);
}

void polynomial_degree_and_range() {
  // det(J) = (1+x)(1+y), whose degree-two Bernstein range encloses [1, 2.25].
  Input quadratic;
  quadratic.derivative(0, 0, {1.0, 0.0, 0.0});
  quadratic.derivative(0, 1, {1.0, 0.0, 0.0});
  quadratic.derivative(1, 0, {0.0, 1.0, 0.0});
  quadratic.derivative(1, 2, {0.0, 1.0, 0.0});
  quadratic.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result degree_two = call(quadratic, 0.1);
  PHX_CHECK(degree_two.status == PHX_MC_OK);
  PHX_CHECK(degree_two.decision == kValid);
  PHX_CHECK(degree_two.lower <= 1.0 && degree_two.upper >= 2.25);

  // det(I + x 11^T) = 1 + 3x; cancellation leaves an actual linear polynomial.
  Input canceled_degree;
  canceled_degree.derivative(0, 0, {1.0, 0.0, 0.0});
  canceled_degree.derivative(1, 0, {0.0, 1.0, 0.0});
  canceled_degree.derivative(2, 0, {0.0, 0.0, 1.0});
  canceled_degree.derivative(0, 1, {1.0, 1.0, 1.0});
  canceled_degree.derivative(1, 1, {1.0, 1.0, 1.0});
  canceled_degree.derivative(2, 1, {1.0, 1.0, 1.0});
  const Result linear = call(canceled_degree);
  PHX_CHECK(linear.status == PHX_MC_OK);
  PHX_CHECK(linear.decision == kValid);
  PHX_CHECK(linear.lower == 1.0 && linear.upper == 4.0);
}

void relative_floor_and_unresolved_contacts() {
  // A conservative whole-cell Hadamard scale puts the exact corner on the
  // declared floor. It is not promoted to a valid proof.
  Input shear;
  shear.derivative(0, 0, {1.0, 0.0, 0.0});
  shear.derivative(1, 0, {0.0, 1.0, 0.0});
  shear.derivative(2, 0, {0.0, 0.0, 1.0});
  shear.derivative(1, 1, {1.0, 0.0, 0.0});
  shear.derivative(0, 2, {0.0, 1.0, 0.0});
  const Result floor_contact = call(shear, 0.5);
  PHX_CHECK(floor_contact.status == PHX_MC_OK);
  PHX_CHECK(floor_contact.decision == kUnresolved);
  PHX_CHECK(floor_contact.lower <= 0.5 && floor_contact.upper >= 1.0);

  // A smaller relative floor is strictly met but the root Bernstein hull is
  // too wide; continuous adaptive subdivision resolves the multi-axis shear.
  const Result refined_shear = call(shear, 0.25);
  PHX_CHECK(refined_shear.status == PHX_MC_OK);
  PHX_CHECK(refined_shear.decision == kValid);
  PHX_CHECK(refined_shear.lower <= 0.75 && refined_shear.upper >= 1.0);

  // (3x-1)^2 has an exact zero at the non-dyadic rational parameter 1/3.
  // Adaptive Bernstein refinement cannot represent that contact exactly and
  // must stop with a finite unresolved result at binary64 parameter resolution.
  Input non_dyadic_contact;
  non_dyadic_contact.derivative(0, 0, {-1.0, 0.0, 0.0});
  non_dyadic_contact.derivative(0, 1, {3.0, 0.0, 0.0});
  non_dyadic_contact.derivative(1, 0, {0.0, -1.0, 0.0});
  non_dyadic_contact.derivative(1, 1, {0.0, 3.0, 0.0});
  non_dyadic_contact.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result contact = call(non_dyadic_contact);
  PHX_CHECK(contact.status == PHX_MC_OK);
  PHX_CHECK(contact.decision == kUnresolved);
  PHX_CHECK(contact.lower <= 0.0 && contact.upper >= 0.0);

  // A very thin but strictly positive feature takes substantially deeper
  // subdivision than a shallow fixed schedule; the exact positive minimum is
  // still certified rather than refused by a work/depth quota.
  constexpr double center = 0x1.5555555555555p-2;
  constexpr double offset = 0x1p-500;
  Input thin_positive;
  thin_positive.derivative(0, 0, {-center, 0.0, 0.0});
  thin_positive.derivative(0, 1, {1.0, 0.0, 0.0});
  thin_positive.derivative(1, 0, {0.0, -center, 0.0});
  thin_positive.derivative(1, 1, {0.0, 1.0, 0.0});
  thin_positive.derivative(1, 0, {-offset, 0.0, 0.0});
  thin_positive.derivative(0, 0, {0.0, offset, 0.0});
  thin_positive.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result refined = call(thin_positive);
  PHX_CHECK(refined.status == PHX_MC_OK);
  PHX_CHECK(refined.decision == kValid);
  PHX_CHECK(refined.lower <= offset * offset && refined.upper >= offset * offset);
}

void extreme_values_and_input_validation() {
  const double tiny = std::numeric_limits<double>::denorm_min();
  Input subnormal;
  subnormal.derivative(0, 0, {tiny, 0.0, 0.0});
  subnormal.derivative(1, 0, {0.0, tiny, 0.0});
  subnormal.derivative(2, 0, {0.0, 0.0, tiny});
  const Result underflow = call(subnormal);
  PHX_CHECK(underflow.status == PHX_MC_OK);
  PHX_CHECK(underflow.decision == kValid);
  PHX_CHECK(underflow.lower == 0.0 && underflow.upper == tiny);

  Input nonfinite = identity();
  nonfinite.nodes[0] = std::numeric_limits<double>::infinity();
  const Result rejected = call(nonfinite);
  PHX_CHECK(rejected.status == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(rejected.decision == 77 && rejected.lower == -123.0 && rejected.upper == 456.0);

  Input identity_input = identity();
  Result invalid_floor;
  invalid_floor.status = phx_mc_quadratic_tetrahedron_validity(
      1, identity_input.nodes.data(), identity_input.derivative_weights.data(), 1.0,
      &invalid_floor.decision, &invalid_floor.lower, &invalid_floor.upper);
  PHX_CHECK(invalid_floor.status == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(invalid_floor.decision == 77 && invalid_floor.lower == -123.0 &&
            invalid_floor.upper == 456.0);

  const std::int32_t empty = phx_mc_quadratic_tetrahedron_validity(
      0, nullptr, nullptr, 0.0, nullptr, nullptr, nullptr);
  PHX_CHECK(empty == PHX_MC_OK);
  const std::int32_t overflow = phx_mc_quadratic_tetrahedron_validity(
      std::numeric_limits<std::int64_t>::max(), identity_input.nodes.data(),
      identity_input.derivative_weights.data(), 0.0, &invalid_floor.decision,
      &invalid_floor.lower, &invalid_floor.upper);
  PHX_CHECK(overflow == PHX_MC_INVALID_ARGUMENT);
}

bool same_result(const Result& first, const Result& second) {
  return first.status == second.status && first.decision == second.decision &&
         std::bit_cast<std::uint64_t>(first.lower) == std::bit_cast<std::uint64_t>(second.lower) &&
         std::bit_cast<std::uint64_t>(first.upper) == std::bit_cast<std::uint64_t>(second.upper);
}

void floating_point_hazards() {
  // det(J) = (1+x)(1+y) with floor 0.1, and the subnormal identity scaled by
  // the smallest subnormal, keep their round-to-nearest results in every mode.
  Input quadratic;
  quadratic.derivative(0, 0, {1.0, 0.0, 0.0});
  quadratic.derivative(0, 1, {1.0, 0.0, 0.0});
  quadratic.derivative(1, 0, {0.0, 1.0, 0.0});
  quadratic.derivative(1, 2, {0.0, 1.0, 0.0});
  quadratic.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result quadratic_reference = call(quadratic, 0.1);
  PHX_CHECK(quadratic_reference.status == PHX_MC_OK && quadratic_reference.decision == kValid);
  const double tiny = std::numeric_limits<double>::denorm_min();
  Input subnormal;
  subnormal.derivative(0, 0, {tiny, 0.0, 0.0});
  subnormal.derivative(1, 0, {0.0, tiny, 0.0});
  subnormal.derivative(2, 0, {0.0, 0.0, tiny});
  const Result subnormal_reference = call(subnormal);
  PHX_CHECK(subnormal_reference.status == PHX_MC_OK &&
            subnormal_reference.decision == kValid);
  PHX_CHECK(subnormal_reference.lower == 0.0 && subnormal_reference.upper == tiny);
  // A negative subnormal relative floor is rejected before any evaluation.
  const Result negative_floor = call(identity(), -0x1p-1074);
  PHX_CHECK(negative_floor.status == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(negative_floor.decision == 77 && negative_floor.lower == -123.0 &&
            negative_floor.upper == 456.0);
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    PHX_CHECK(same_result(call(quadratic, 0.1), quadratic_reference));
    PHX_CHECK(same_result(call(subnormal), subnormal_reference));
    PHX_CHECK(same_result(call(identity(), -0x1p-1074), negative_floor));
  });
}

// P2 Jacobian weights (corners, then edges 01 02 03 12 13 23) from the P2
// shape functions N_i = l_i (2 l_i - 1), N_ij = 4 l_i l_j.
std::array<std::int32_t, 120> p2_derivative_weights() {
  constexpr int corners[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  const int linear[4][4] = {{1, -1, -1, -1}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  std::array<std::int32_t, 120> values{};
  for (int c = 0; c < 3; ++c) {
    const auto gradient = [c](int i) { return i == 0 ? -1 : (i == c + 1 ? 1 : 0); };
    for (int p = 0; p < 4; ++p) {
      for (int i = 0; i < 4; ++i)
        values[(c * 4 + p) * 10 + i] = gradient(i) * (4 * linear[i][p] - (p == 0 ? 1 : 0));
      for (int e = 0; e < 6; ++e) {
        const int i = corners[e][0], j = corners[e][1];
        values[(c * 4 + p) * 10 + 4 + e] =
            4 * (gradient(i) * linear[j][p] + gradient(j) * linear[i][p]);
      }
    }
  }
  return values;
}

// Randomized P2 tets built to stress the screen's error bound: generic,
// nearly flat, scaled across the binary64 range (guards against underflow and
// overflow), large coordinate offsets (cancellation), and cells bisected onto
// the exact validity boundary. Verdicts without bounds (screen enabled) must
// equal verdicts with bounds (exact path only), and every screened cell must
// be exactly VALID. Catches a too-small error constant or a missed rounding
// (a screened cell on the boundary that is exactly INVALID or UNRESOLVED), and
// a screen that never fires (the generic family must be mostly screened).
void screen_agrees_with_exact_path() {
  const auto weights = p2_derivative_weights();
  constexpr int kFamilies = 6;
  const char* names[kFamilies] = {"generic", "nearly_flat", "scaled", "offset", "near_boundary",
                                  "bisected_boundary"};
  std::mt19937_64 random(0x5eed2026);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::uniform_real_distribution<double> fraction(0.0, 1.0);
  using Cell = std::array<double, 30>;
  const auto base = [&](double height) {
    Cell cell{};
    for (int i = 0; i < 4; ++i)
      for (int c = 0; c < 3; ++c) cell[3 * i + c] = unit(random);
    if (height > 0.0) {
      for (int i = 0; i < 3; ++i) cell[3 * i + 2] = 0.0;
      cell[11] = height;
    }
    const double* p = cell.data();
    const double ax = p[3] - p[0], ay = p[4] - p[1], az = p[5] - p[2];
    const double bx = p[6] - p[0], by = p[7] - p[1], bz = p[8] - p[2];
    const double cx = p[9] - p[0], cy = p[10] - p[1], cz = p[11] - p[2];
    if (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx) < 0.0) {
      for (int c = 0; c < 3; ++c) std::swap(cell[3 + c], cell[6 + c]);
    }
    constexpr int corners[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    for (int e = 0; e < 6; ++e)
      for (int c = 0; c < 3; ++c)
        cell[12 + 3 * e + c] = 0.5 * (cell[3 * corners[e][0] + c] + cell[3 * corners[e][1] + c]);
    return cell;
  };
  const auto displace = [&](Cell& cell, const Cell& direction, double scale) {
    for (int k = 12; k < 30; ++k) cell[k] += scale * direction[k];
  };
  const auto random_direction = [&] {
    Cell direction{};
    for (int k = 12; k < 30; ++k) direction[k] = unit(random);
    return direction;
  };
  const auto log_uniform = [&](double low, double high) {
    return std::exp2(std::log2(low) + fraction(random) * (std::log2(high) - std::log2(low)));
  };
  const auto exact_verdict = [&](const Cell& cell, double floor) {
    std::int32_t decision = 77;
    double lower = 0.0, upper = 0.0;
    PHX_CHECK(phx_mc_quadratic_tetrahedron_validity(1, cell.data(), weights.data(), floor,
                                                    &decision, &lower, &upper) == PHX_MC_OK);
    return decision;
  };

  constexpr double kFloors[3] = {0.0, 1e-3, 0.2};
  std::array<std::vector<double>, 3> nodes;
  std::array<std::vector<int>, 3> families;
  const auto add = [&](const Cell& cell, int family, int floor) {
    nodes[floor].insert(nodes[floor].end(), cell.begin(), cell.end());
    families[floor].push_back(family);
  };
  constexpr int kRandomCells = 120000;
  for (int item = 0; item < kRandomCells; ++item) {
    const int family = item % (kFamilies - 1);
    const int floor = (item / kFamilies) % 3;
    Cell cell;
    if (family == 0) {
      cell = base(0.0);
      displace(cell, random_direction(), log_uniform(1e-6, 0.5));
    } else if (family == 1) {
      const double height = log_uniform(1e-16, 1e-2);
      cell = base(height);
      displace(cell, random_direction(), height * log_uniform(1e-6, 0.5));
    } else if (family == 2) {
      cell = base(0.0);
      displace(cell, random_direction(), log_uniform(1e-6, 0.5));
      const double scale = std::ldexp(1.0, static_cast<int>(fraction(random) * 2060.0) - 1050);
      for (double& value : cell) value *= scale;
    } else if (family == 3) {
      cell = base(0.0);
      displace(cell, random_direction(), log_uniform(1e-6, 0.5));
      const std::array<double, 3> offset{unit(random) * log_uniform(1e3, 1e15),
                                         unit(random) * log_uniform(1e3, 1e15),
                                         unit(random) * log_uniform(1e3, 1e15)};
      for (int k = 0; k < 30; ++k) cell[k] += offset[k % 3];
    } else {
      cell = base(0.0);
      displace(cell, random_direction(), log_uniform(1e-3, 3.0));
    }
    add(cell, family, floor);
  }
  // Bisection of the displacement scale onto the boundary of exact validity.
  for (int item = 0; item < 600; ++item) {
    const int floor = item % 3;
    const Cell start = base(0.0);
    if (exact_verdict(start, kFloors[floor]) != kValid) continue;
    const Cell direction = random_direction();
    double valid_scale = 0.0, failed_scale = 4.0;
    Cell probe = start;
    displace(probe, direction, failed_scale);
    if (exact_verdict(probe, kFloors[floor]) == kValid) continue;
    for (int step = 0; step < 45; ++step) {
      const double middle = 0.5 * (valid_scale + failed_scale);
      probe = start;
      displace(probe, direction, middle);
      (exact_verdict(probe, kFloors[floor]) == kValid ? valid_scale : failed_scale) = middle;
    }
    for (double scale : {valid_scale, failed_scale}) {
      probe = start;
      displace(probe, direction, scale);
      add(probe, kFamilies - 1, floor);
    }
  }

  std::array<std::int64_t, kFamilies> cells{}, valid{}, screened{};
  const phx::mc::QuadraticDerivativeStencil stencil(weights.data());
  for (int floor = 0; floor < 3; ++floor) {
    const auto count = static_cast<std::int64_t>(families[floor].size());
    std::vector<std::int32_t> fast(families[floor].size(), -1), exact(families[floor].size(), -1);
    std::vector<double> lower(families[floor].size()), upper(families[floor].size());
    PHX_CHECK(phx_mc_quadratic_tetrahedron_validity(count, nodes[floor].data(), weights.data(),
                                                    kFloors[floor], fast.data(), nullptr,
                                                    nullptr) == PHX_MC_OK);
    PHX_CHECK(phx_mc_quadratic_tetrahedron_validity(count, nodes[floor].data(), weights.data(),
                                                    kFloors[floor], exact.data(), lower.data(),
                                                    upper.data()) == PHX_MC_OK);
    PHX_CHECK(fast == exact);
    for (std::size_t k = 0; k < families[floor].size(); ++k) {
      const int family = families[floor][k];
      ++cells[family];
      valid[family] += exact[k] == kValid;
      if (!phx::mc::quadratic_validity_screen(nodes[floor].data() + 30 * k, stencil,
                                              kFloors[floor]))
        continue;
      ++screened[family];
      PHX_CHECK(exact[k] == kValid);
    }
  }
  std::int64_t total = 0;
  for (int family = 0; family < kFamilies; ++family) {
    total += cells[family];
    std::printf("screen %s: cells=%lld exactly_valid=%lld screened=%lld exact_path=%lld\n",
                names[family], static_cast<long long>(cells[family]),
                static_cast<long long>(valid[family]), static_cast<long long>(screened[family]),
                static_cast<long long>(cells[family] - screened[family]));
  }
  PHX_CHECK(total >= 100000);
  PHX_CHECK(cells[kFamilies - 1] > 0);
  PHX_CHECK(2 * screened[0] > valid[0]);

  // Bounds are optional only as a pair.
  std::int32_t decision = 77;
  double bound = 0.0;
  PHX_CHECK(phx_mc_quadratic_tetrahedron_validity(1, nodes[0].data(), weights.data(), 0.0,
                                                  &decision, nullptr, &bound) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_quadratic_tetrahedron_validity(1, nodes[0].data(), weights.data(), 0.0,
                                                  &decision, &bound, nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(decision == 77 && bound == 0.0);
}

}  // namespace

int main() {
  affine_and_degenerate_maps();
  polynomial_degree_and_range();
  relative_floor_and_unresolved_contacts();
  floating_point_hazards();
  extreme_values_and_input_validation();
  screen_agrees_with_exact_path();
  return phx::mc::test::finish("test_quadratic_validity");
}
