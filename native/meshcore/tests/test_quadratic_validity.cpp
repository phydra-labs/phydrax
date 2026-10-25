// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Exact numerical/status contract for the native P2 tetrahedron root proof.
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "bounded_memory.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr std::int64_t kUnlimited = std::numeric_limits<std::int64_t>::max();

struct Input final {
  std::array<double, 30> nodes{};
  std::array<std::int32_t, 120> derivative_weights{};
  int next_node = 0;

  void derivative(int column, int power, const std::array<double, 3>& values) {
    const int node = next_node++;
    for (int row = 0; row < 3; ++row) nodes[3 * node + row] = values[row];
    derivative_weights[(column * 4 + power) * 10 + node] = 1;
  }
};

struct Result final {
  std::int32_t status = -1;
  std::int32_t decision = 77;
  double lower = -123.0;
  double upper = 456.0;
  std::array<std::int64_t, 6> evidence{};
};

Result call(const Input& input, double floor = 0.0, std::int64_t maximum_nodes = 20,
            std::int64_t maximum_depth = 0, std::int64_t maximum_work = kUnlimited,
            std::int64_t maximum_scratch = kUnlimited) {
  Result result;
  result.status = phx_mc_quadratic_tetrahedron_validity(
      1, input.nodes.data(), input.derivative_weights.data(), floor, maximum_nodes,
      maximum_depth, maximum_work, maximum_scratch, &result.decision, &result.lower,
      &result.upper, result.evidence.data());
  return result;
}

Input identity() {
  Input input;
  input.derivative(0, 0, {1.0, 0.0, 0.0});
  input.derivative(1, 0, {0.0, 1.0, 0.0});
  input.derivative(2, 0, {0.0, 0.0, 1.0});
  return input;
}

void affine_and_inverted() {
  const Result valid = call(identity(), 0.0, 1);
  PHX_CHECK(valid.status == PHX_MC_OK);
  PHX_CHECK(valid.decision == 1);
  PHX_CHECK(valid.lower == 1.0 && valid.upper == 1.0);
  PHX_CHECK(valid.evidence[1] == 1 && valid.evidence[3] == 1);
  PHX_CHECK(valid.evidence[4] == -1 && valid.evidence[5] == 0);

  Input inverted;
  inverted.derivative(0, 0, {-1.0, 0.0, 0.0});
  inverted.derivative(1, 0, {0.0, 1.0, 0.0});
  inverted.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result invalid = call(inverted);
  PHX_CHECK(invalid.status == PHX_MC_OK);
  PHX_CHECK(invalid.decision == 2);
  PHX_CHECK(invalid.lower == -1.0 && invalid.upper == -1.0);

  Input singular;
  singular.derivative(0, 1, {1.0, 0.0, 0.0});
  singular.derivative(1, 0, {0.0, 1.0, 0.0});
  singular.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result zero_corner = call(singular);
  PHX_CHECK(zero_corner.status == PHX_MC_OK);
  PHX_CHECK(zero_corner.decision == 2);
  PHX_CHECK(zero_corner.lower == 0.0 && zero_corner.upper == 1.0);
}

void variable_determinants_and_actual_degree() {
  Input quadratic;
  quadratic.derivative(0, 0, {1.0, 0.0, 0.0});
  quadratic.derivative(0, 1, {1.0, 0.0, 0.0});
  quadratic.derivative(1, 0, {0.0, 1.0, 0.0});
  quadratic.derivative(1, 2, {0.0, 1.0, 0.0});
  quadratic.derivative(2, 0, {0.0, 0.0, 1.0});
  const Result degree_two = call(quadratic, 0.1, 10);
  PHX_CHECK(degree_two.status == PHX_MC_OK);
  PHX_CHECK(degree_two.decision == 1);
  PHX_CHECK(degree_two.lower == 1.0 && degree_two.upper == 2.5);
  const Result node_limited = call(quadratic, 0.1, 9);
  PHX_CHECK(node_limited.status == PHX_MC_OK);
  PHX_CHECK(node_limited.decision == 3);
  PHX_CHECK(std::isnan(node_limited.lower) && std::isnan(node_limited.upper));

  // det(I + x 11^T) = 1 + 3x: its degree is one although the six-term
  // determinant expansion contains products through degree three.
  Input canceled_degree;
  canceled_degree.derivative(0, 0, {1.0, 0.0, 0.0});
  canceled_degree.derivative(1, 0, {0.0, 1.0, 0.0});
  canceled_degree.derivative(2, 0, {0.0, 0.0, 1.0});
  canceled_degree.derivative(0, 1, {1.0, 1.0, 1.0});
  canceled_degree.derivative(1, 1, {1.0, 1.0, 1.0});
  canceled_degree.derivative(2, 1, {1.0, 1.0, 1.0});
  const Result actual_degree_one = call(canceled_degree, 0.0, 4);
  PHX_CHECK(actual_degree_one.status == PHX_MC_OK);
  PHX_CHECK(actual_degree_one.decision == 1);
  PHX_CHECK(actual_degree_one.lower == 1.0 && actual_degree_one.upper == 4.0);
  const Result canceled_node_limit = call(canceled_degree, 0.0, 3);
  PHX_CHECK(canceled_node_limit.status == PHX_MC_OK);
  PHX_CHECK(canceled_node_limit.decision == 3);
}

void strict_floor_and_depth() {
  // J=[[1,x,0],[y,1,0],[0,0,1]], det=1-xy. All four corners equal
  // the exact floor threshold (floor=1/2, squared Hadamard scale=4), while
  // the minimum scaled Bernstein control is 3 and cannot certify the floor.
  Input shear;
  shear.derivative(0, 0, {1.0, 0.0, 0.0});
  shear.derivative(1, 0, {0.0, 1.0, 0.0});
  shear.derivative(2, 0, {0.0, 0.0, 1.0});
  shear.derivative(1, 1, {1.0, 0.0, 0.0});
  shear.derivative(0, 2, {0.0, 1.0, 0.0});
  const Result depth_zero = call(shear, 0.5, 10, 0);
  PHX_CHECK(depth_zero.status == PHX_MC_OK);
  PHX_CHECK(depth_zero.decision == 4);
  PHX_CHECK(depth_zero.lower == 0.5 && depth_zero.upper == 1.0);
  const Result subdivide = call(shear, 0.5, 10, 1);
  PHX_CHECK(subdivide.status == PHX_MC_OK);
  PHX_CHECK(subdivide.decision == 0);
  PHX_CHECK(subdivide.lower == 0.5 && subdivide.upper == 1.0);
}

void directed_subnormal_and_saturation() {
  const double tiny = std::numeric_limits<double>::denorm_min();
  Input subnormal;
  subnormal.derivative(0, 0, {tiny, 0.0, 0.0});
  subnormal.derivative(1, 0, {0.0, tiny, 0.0});
  subnormal.derivative(2, 0, {0.0, 0.0, tiny});
  const Result underflowed = call(subnormal);
  PHX_CHECK(underflowed.status == PHX_MC_OK);
  PHX_CHECK(underflowed.decision == 1);
  PHX_CHECK(underflowed.lower == 0.0 && underflowed.upper == tiny);

  Input negative_subnormal;
  negative_subnormal.derivative(0, 0, {tiny, 0.0, 0.0});
  negative_subnormal.derivative(1, 0, {0.0, tiny, 0.0});
  negative_subnormal.derivative(2, 0, {0.0, 0.0, -tiny});
  const Result rounded_negative = call(negative_subnormal);
  PHX_CHECK(rounded_negative.status == PHX_MC_OK);
  PHX_CHECK(rounded_negative.decision == 2);
  PHX_CHECK(rounded_negative.lower == -tiny);
  PHX_CHECK(rounded_negative.upper == 0.0 && std::signbit(rounded_negative.upper));

  const double largest = std::numeric_limits<double>::max();
  Input huge;
  huge.derivative(0, 0, {largest, 0.0, 0.0});
  huge.derivative(1, 0, {0.0, largest, 0.0});
  huge.derivative(2, 0, {0.0, 0.0, largest});
  const Result overflowed = call(huge);
  PHX_CHECK(overflowed.status == PHX_MC_OK);
  PHX_CHECK(overflowed.decision == 1);
  PHX_CHECK(overflowed.lower == largest && std::isinf(overflowed.upper));
}

void validation_and_empty_batch() {
  Input nonfinite = identity();
  nonfinite.nodes[0] = std::numeric_limits<double>::infinity();
  const Result rejected = call(nonfinite);
  PHX_CHECK(rejected.status == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(rejected.decision == 77 && rejected.lower == -123.0 && rejected.upper == 456.0);
  PHX_CHECK(rejected.evidence[4] == 0 && rejected.evidence[5] == 1);

  Result overflow;
  overflow.status = phx_mc_quadratic_tetrahedron_validity(
      kUnlimited, nonfinite.nodes.data(), nonfinite.derivative_weights.data(), 0.0, 20, 0,
      kUnlimited, kUnlimited, &overflow.decision, &overflow.lower, &overflow.upper,
      overflow.evidence.data());
  PHX_CHECK(overflow.status == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(overflow.decision == 77 && overflow.lower == -123.0 && overflow.upper == 456.0);
  PHX_CHECK(overflow.evidence[5] == 1);

  const Result invalid_policy = call(identity(), 0.0, 0);
  PHX_CHECK(invalid_policy.status == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(invalid_policy.decision == 77 && invalid_policy.lower == -123.0 &&
            invalid_policy.upper == 456.0);
  PHX_CHECK(invalid_policy.evidence[5] == 1);

  std::array<std::int64_t, 6> empty_evidence{};
  const std::int32_t empty_status = phx_mc_quadratic_tetrahedron_validity(
      0, nullptr, nullptr, 0.0, 1, 0, 0, 0, nullptr, nullptr, nullptr,
      empty_evidence.data());
  PHX_CHECK(empty_status == PHX_MC_OK);
  PHX_CHECK(empty_evidence[0] == 0 && empty_evidence[1] == 0 && empty_evidence[2] == 0);
  PHX_CHECK(empty_evidence[3] == 0 && empty_evidence[4] == -1 && empty_evidence[5] == 0);
}

void exact_work_and_scratch_admission() {
  const Input input = identity();
  const Result baseline = call(input);
  PHX_CHECK(baseline.status == PHX_MC_OK);
  PHX_CHECK(baseline.evidence[0] > 0 && baseline.evidence[2] > 0);

  const Result exact_work = call(input, 0.0, 20, 0, baseline.evidence[0]);
  PHX_CHECK(exact_work.status == PHX_MC_OK);
  PHX_CHECK(exact_work.evidence[0] == baseline.evidence[0]);
  const Result short_work = call(input, 0.0, 20, 0, baseline.evidence[0] - 1);
  PHX_CHECK(short_work.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(short_work.decision == 77 && short_work.lower == -123.0 && short_work.upper == 456.0);

  const Result exact_scratch = call(input, 0.0, 20, 0, kUnlimited,
                                    baseline.evidence[2]);
  PHX_CHECK(exact_scratch.status == PHX_MC_OK);
  const Result short_scratch = call(input, 0.0, 20, 0, kUnlimited,
                                    baseline.evidence[2] - 1);
  PHX_CHECK(short_scratch.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(short_scratch.decision == 77 && short_scratch.lower == -123.0 &&
            short_scratch.upper == 456.0);
  PHX_CHECK(short_scratch.evidence[5] == 2);
}

void native_deadline_refusal() {
  const Input input = identity();
  phx::mc::NativeExecutionScope deadline(kUnlimited, kUnlimited, kUnlimited,
                                         std::numeric_limits<std::size_t>::max(), 0.0);
  const Result expired = call(input);
  PHX_CHECK(expired.status == PHX_MC_TIMEOUT);
  PHX_CHECK(expired.decision == 77 && expired.lower == -123.0 && expired.upper == 456.0);
  PHX_CHECK(expired.evidence[3] == 0 && expired.evidence[4] == -1);
  PHX_CHECK(expired.evidence[5] == 2);
}

}  // namespace

int main() {
  affine_and_inverted();
  variable_determinants_and_actual_degree();
  strict_floor_and_depth();
  directed_subnormal_and_saturation();
  validation_and_empty_batch();
  exact_work_and_scratch_admission();
  native_deadline_refusal();
  return phx::mc::test::finish("test_quadratic_validity");
}
