// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Behavioral tests for exact continuous validity of quadratic tetrahedral maps.
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "check.hpp"
#include "phydrax_meshcore.h"

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

}  // namespace

int main() {
  affine_and_degenerate_maps();
  polynomial_degree_and_range();
  relative_floor_and_unresolved_contacts();
  extreme_values_and_input_validation();
  return phx::mc::test::finish("test_quadratic_validity");
}
