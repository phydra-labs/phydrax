// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Behavioral contract for outward interval-program execution and whole-program
// validation before writes. No execution-accounting policy is exercised here.
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <vector>

#include "check.hpp"
#include "elementary.hpp"
#include "floating_point_mode.hpp"
#include "interval_program.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr std::int64_t kGather = 1;
constexpr std::int64_t kAdd = 2;
constexpr std::int64_t kSubtract = 3;
constexpr std::int64_t kSubtractSame = 4;
constexpr std::int64_t kMultiply = 5;
constexpr std::int64_t kNegate = 6;
constexpr std::int64_t kReduceSum = 7;
constexpr std::int64_t kSine = 8;
constexpr std::int64_t kCosine = 9;
constexpr std::int64_t kSqrt = 10;
constexpr std::int64_t kReciprocal = 11;
constexpr std::int64_t kSquare = 12;
constexpr std::int64_t kCompare = 13;
constexpr std::int64_t kSelect = 14;
constexpr std::int64_t kMaximum = 15;
constexpr std::int64_t kMinimum = 16;
constexpr std::int64_t kExp = 17;
constexpr std::int64_t kLog = 18;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kLargest = std::numeric_limits<double>::max();
constexpr double kSmallest = std::numeric_limits<double>::denorm_min();
constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
constexpr double kPi = 3.141592653589793;
constexpr double kTwoPi = 2.0 * kPi;
constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();

std::uint64_t bits(double value) { return std::bit_cast<std::uint64_t>(value); }
double from_bits(std::uint64_t value) { return std::bit_cast<double>(value); }
bool same(double a, double b) { return bits(a) == bits(b); }

double step_down(double value) {
  if (std::isnan(value) || value == -kInfinity) return value;
  if (value == 0.0) return -kSmallest;
  return from_bits(value > 0.0 ? bits(value) - 1 : bits(value) + 1);
}

double step_up(double value) {
  if (std::isnan(value) || value == kInfinity) return value;
  if (value == 0.0) return kSmallest;
  return from_bits(value > 0.0 ? bits(value) + 1 : bits(value) - 1);
}

double steps_down(double value, int count) {
  for (int step = 0; step < count; ++step) value = step_down(value);
  return value;
}

double numpy_minimum(double first, double second) {
  if (std::isnan(first)) return first;
  if (std::isnan(second)) return second;
  if (first == second) return std::signbit(first) ? first : second;
  return first < second ? first : second;
}

double numpy_maximum(double first, double second) {
  if (std::isnan(first)) return first;
  if (std::isnan(second)) return second;
  if (first == second) return std::signbit(first) ? second : first;
  return first > second ? first : second;
}

double endpoint_product(double first, double second) {
  if ((first == 0.0 && std::isinf(second)) || (std::isinf(first) && second == 0.0)) return 0.0;
  return first * second;
}

struct Program {
  std::vector<std::int64_t> rows;
  std::vector<std::int64_t> indices;
  std::vector<double> parameters;
  std::vector<double> lower;
  std::vector<double> upper;

  std::int64_t count() const { return static_cast<std::int64_t>(rows.size() / 8); }
  void row(std::int64_t opcode, std::int64_t out, std::int64_t size, std::int64_t a,
           std::int64_t a_index, std::int64_t b, std::int64_t b_index, std::int64_t aux) {
    rows.insert(rows.end(), {opcode, out, size, a, a_index, b, b_index, aux});
  }
  std::int64_t values(std::initializer_list<std::array<double, 2>> entries) {
    const auto offset = static_cast<std::int64_t>(lower.size());
    for (const auto& entry : entries) {
      lower.push_back(entry[0]);
      upper.push_back(entry[1]);
    }
    return offset;
  }
  std::int64_t slots(std::int64_t size) {
    const auto offset = static_cast<std::int64_t>(lower.size());
    lower.insert(lower.end(), static_cast<std::size_t>(size), 12345.0);
    upper.insert(upper.end(), static_cast<std::size_t>(size), -12345.0);
    return offset;
  }
  std::int64_t index(std::initializer_list<std::int64_t> entries) {
    const auto offset = static_cast<std::int64_t>(indices.size());
    indices.insert(indices.end(), entries);
    return offset;
  }
  std::int64_t parameter(double value) {
    parameters.push_back(value);
    return static_cast<std::int64_t>(parameters.size()) - 1;
  }
};

std::int32_t execute(Program& program) {
  std::vector<double> prepared_lower = program.lower, prepared_upper = program.upper;
  phx::mc::PreparedIntervalProgram prepared;
  const auto bound = prepared.bind(
      program.rows.data(), program.count(), program.indices.data(),
      static_cast<std::int64_t>(program.indices.size()), program.parameters.data(),
      static_cast<std::int64_t>(program.parameters.size()), static_cast<std::int64_t>(program.lower.size()));
  const auto status = phx_mc_interval_program_execute(
      program.rows.data(), program.count(), program.indices.data(),
      static_cast<std::int64_t>(program.indices.size()), program.parameters.data(),
      static_cast<std::int64_t>(program.parameters.size()), program.lower.data(),
      program.upper.data(), static_cast<std::int64_t>(program.lower.size()));
  if (bound == PHX_MC_OK) {
    PHX_CHECK(prepared.execute(prepared_lower.data(), prepared_upper.data(),
                              static_cast<std::int64_t>(prepared_lower.size())) == status);
    for (size_t i = 0; i < prepared_lower.size(); ++i) {
      PHX_CHECK(same(prepared_lower[i], program.lower[i]));
      PHX_CHECK(same(prepared_upper[i], program.upper[i]));
    }
  } else {
    PHX_CHECK(bound == status);
  }
  return status;
}

void reference(const Program& program, std::vector<double>& lower, std::vector<double>& upper) {
  lower = program.lower;
  upper = program.upper;
  for (std::int64_t row_index = 0; row_index < program.count(); ++row_index) {
    const std::int64_t* fields = program.rows.data() + row_index * 8;
    const std::int64_t opcode = fields[0], out = fields[1], size = fields[2], a = fields[3];
    const std::int64_t ai = fields[4], b = fields[5], bi = fields[6], aux = fields[7];
    const auto at = [&](std::int64_t offset, std::int64_t index, std::int64_t element) {
      return static_cast<std::size_t>(offset + (index < 0 ? element :
          program.indices[static_cast<std::size_t>(index + element)]));
    };
    for (std::int64_t element = 0; element < size; ++element) {
      const std::size_t target = static_cast<std::size_t>(out + element);
      if (opcode == kReduceSum) {
        const std::int64_t base = a + element * aux;
        double low_sum = 0.0, high_sum = 0.0, scale = 0.0;
        for (std::int64_t term = 0; term < aux; ++term) {
          low_sum += lower[static_cast<std::size_t>(base + term)];
          high_sum += upper[static_cast<std::size_t>(base + term)];
        }
        for (std::int64_t term = 0; term < aux; ++term) {
          scale += numpy_maximum(std::fabs(lower[static_cast<std::size_t>(base + term)]),
                                 std::fabs(upper[static_cast<std::size_t>(base + term)]));
        }
        const double gamma = program.parameters[static_cast<std::size_t>(bi)];
        lower[target] = step_down(low_sum - gamma * scale);
        upper[target] = step_up(high_sum + gamma * scale);
        continue;
      }
      const std::size_t x = at(a, ai, element);
      const double al = lower[x], ah = upper[x];
      if (opcode == kGather) {
        lower[target] = al;
        upper[target] = ah;
      } else if (opcode == kNegate) {
        lower[target] = -ah;
        upper[target] = -al;
      } else if (opcode == kSine || opcode == kCosine) {
        const phx::mc::elementary::Enclosure interval =
            opcode == kSine ? phx::mc::elementary::sin(al, ah) : phx::mc::elementary::cos(al, ah);
        lower[target] = interval.lower;
        upper[target] = interval.upper;
      } else {
        const std::size_t y = at(b, bi, element);
        const double bl = lower[y], bh = upper[y];
        if (opcode == kAdd) {
          const double low = al + bl, high = ah + bh;
          lower[target] = std::isnan(low) ? -kInfinity : step_down(low);
          upper[target] = std::isnan(high) ? kInfinity : step_up(high);
        } else if (opcode == kMultiply) {
          const double ll = endpoint_product(al, bl), lu = endpoint_product(al, bh);
          const double ul = endpoint_product(ah, bl), uu = endpoint_product(ah, bh);
          if (std::isnan(ll) || std::isnan(lu) || std::isnan(ul) || std::isnan(uu)) {
            lower[target] = -kInfinity;
            upper[target] = kInfinity;
          } else {
            lower[target] = step_down(numpy_minimum(numpy_minimum(ll, lu), numpy_minimum(ul, uu)));
            upper[target] = step_up(numpy_maximum(numpy_maximum(ll, lu), numpy_maximum(ul, uu)));
          }
        } else {
          const double difference_lower = al - bh, difference_upper = ah - bl;
          double low = std::isnan(difference_lower) ? -kInfinity : step_down(difference_lower);
          double high = std::isnan(difference_upper) ? kInfinity : step_up(difference_upper);
          if (opcode == kSubtractSame && std::isfinite(al) && std::isfinite(ah)) low = high = 0.0;
          lower[target] = low;
          upper[target] = high;
        }
      }
    }
  }
}

bool identical(const std::vector<double>& first, const std::vector<double>& second,
               bool any_nan = false) {
  if (first.size() != second.size()) return false;
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (any_nan && std::isnan(first[index]) && std::isnan(second[index])) continue;
    if (!same(first[index], second[index])) return false;
  }
  return true;
}

bool unchanged(const Program& first, const Program& second) {
  return identical(first.lower, second.lower, true) && identical(first.upper, second.upper, true);
}

void check_reference(Program& program, bool any_nan = false) {
  std::vector<double> expected_lower, expected_upper;
  reference(program, expected_lower, expected_upper);
  PHX_CHECK(execute(program) == PHX_MC_OK);
  PHX_CHECK(identical(program.lower, expected_lower, any_nan));
  PHX_CHECK(identical(program.upper, expected_upper, any_nan));
}

Program base_program() {
  Program program;
  program.values({{1.0, 2.0}, {-1.0, 3.0}, {0.5, 0.5}, {2.0, 2.0},
                  {4.0, 5.0}, {-2.0, -1.0}});
  program.slots(4);
  program.values({{7.0, 7.0}, {8.0, 8.0}});
  program.parameter(kEpsilon);
  program.index({1, 0, 5, 3});
  program.row(kAdd, 6, 2, 0, -1, 2, 0, 0);
  program.row(kMultiply, 8, 2, 6, -1, 0, 2, 0);
  return program;
}

void arithmetic_and_reductions() {
  Program program;
  const std::int64_t a = program.values({{1.0, 2.0}, {kLargest, kLargest},
      {-kInfinity, -kInfinity}, {kSmallest, kSmallest}, {-0.0, -0.0}});
  const std::int64_t b = program.values({{0x1p-60, 0x1p-60}, {kLargest, kLargest},
      {-1.0, 1.0}, {kSmallest, kSmallest}, {-0.0, -0.0}});
  const std::int64_t out = program.slots(5);
  program.row(kAdd, out, 5, a, -1, b, -1, 0);
  const std::int64_t subtract = program.slots(2);
  program.row(kSubtract, subtract, 2, a, -1, b, -1, 0);
  const std::int64_t multiplied = program.slots(2);
  program.row(kMultiply, multiplied, 2, a, -1, b, -1, 0);
  const std::int64_t negated = program.slots(2);
  program.row(kNegate, negated, 2, a, -1, -1, -1, 0);
  const std::int64_t gathered = program.slots(3);
  program.row(kGather, gathered, 3, a, program.index({4, 0, 3}), -1, -1, 0);
  const std::int64_t terms = program.values({{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}});
  const std::int64_t reduced = program.slots(1);
  const double gamma = 3.0 * kEpsilon / (1.0 - 3.0 * kEpsilon);
  program.row(kReduceSum, reduced, 1, terms, -1, -1, program.parameter(gamma), 3);
  check_reference(program);
  PHX_CHECK(same(program.lower[out], 1.0 - 0x1p-53));
  PHX_CHECK(same(program.upper[out], 2.0 + 0x1p-51));
  PHX_CHECK(same(program.lower[out + 1], kLargest) && same(program.upper[out + 1], kInfinity));
  PHX_CHECK(same(program.lower[out + 2], -kInfinity) && same(program.upper[out + 2], -kLargest));
  PHX_CHECK(same(program.lower[out + 3], kSmallest) && same(program.upper[out + 3], 3.0 * kSmallest));
  PHX_CHECK(same(program.lower[subtract], steps_down(1.0 - kSmallest, 1)));
  PHX_CHECK(same(program.lower[multiplied], steps_down(0x1p-60, 1)));
  PHX_CHECK(same(program.lower[negated], -2.0) && same(program.upper[negated], -1.0));
  PHX_CHECK(same(program.lower[gathered], -0.0) && same(program.upper[gathered], -0.0));
  PHX_CHECK(program.lower[reduced] < 6.0 && program.upper[reduced] > 6.0);

  Program same_subtract;
  const std::int64_t input = same_subtract.values({{-1.0, 2.0}, {-kInfinity, 1.0}});
  const std::int64_t result = same_subtract.slots(2);
  same_subtract.row(kSubtractSame, result, 2, input, -1, input, -1, 0);
  check_reference(same_subtract);
  PHX_CHECK(same(same_subtract.lower[result], 0.0) && same(same_subtract.upper[result], 0.0));
  PHX_CHECK(same(same_subtract.lower[result + 1], -kInfinity));
  PHX_CHECK(same(same_subtract.upper[result + 1], kInfinity));
}

// An exact zero endpoint times an unbounded side contributes 0 (IEEE 1788),
// matching the host interval multiply: [0, 0] * [1, inf] stays a tight
// enclosure of 0 and [0, 1] * [1, inf] = [0, inf] up to outward rounding,
// instead of the whole line from 0 * inf = NaN.
void zero_times_unbounded_multiply() {
  Program program;
  const std::int64_t a = program.values({{0.0, 0.0}, {0.0, 1.0}, {-kInfinity, 0.0}, {kNaN, kNaN}});
  const std::int64_t b = program.values({{1.0, kInfinity}, {1.0, kInfinity}, {0.0, 0.0}, {0.0, kInfinity}});
  const std::int64_t out = program.slots(4);
  program.row(kMultiply, out, 4, a, -1, b, -1, 0);
  check_reference(program);
  PHX_CHECK(same(program.lower[out], -kSmallest) && same(program.upper[out], kSmallest));
  PHX_CHECK(same(program.lower[out + 1], -kSmallest) && same(program.upper[out + 1], kInfinity));
  PHX_CHECK(same(program.lower[out + 2], -kSmallest) && same(program.upper[out + 2], kSmallest));
  PHX_CHECK(same(program.lower[out + 3], -kInfinity) && same(program.upper[out + 3], kInfinity));
}

// The opcodes are the rigorous elementary kernels (test_elementary owns their
// numerical contract); here they must match them bit for bit inside programs.
void trigonometric_enclosures() {
  Program program;
  const double below_two_pi = step_down(kTwoPi);
  const std::int64_t input = program.values({{0.0, 0.0}, {-kInfinity, kInfinity},
      {kNaN, kNaN}, {1e300, 1e300}, {0.0, kTwoPi}, {0.0, below_two_pi},
      {0.1, 0.2}, {kInfinity, kInfinity}});
  const std::int64_t sine = program.slots(8);
  program.row(kSine, sine, 8, input, -1, -1, -1, 0);
  const std::int64_t cosine = program.slots(8);
  program.row(kCosine, cosine, 8, input, -1, -1, -1, 0);
  check_reference(program, true);
  PHX_CHECK(same(program.lower[sine], 0.0) && same(program.upper[sine], 0.0));
  PHX_CHECK(same(program.lower[cosine], 1.0) && same(program.upper[cosine], 1.0));
  for (std::int64_t element : {1, 4, 5, 7}) {
    PHX_CHECK(same(program.lower[sine + element], -1.0));
    PHX_CHECK(same(program.upper[sine + element], 1.0));
    PHX_CHECK(same(program.lower[cosine + element], -1.0));
    PHX_CHECK(same(program.upper[cosine + element], 1.0));
  }
  // A NaN endpoint gives the whole line.
  PHX_CHECK(program.lower[sine + 2] == -kInfinity && program.upper[sine + 2] == kInfinity);
  PHX_CHECK(program.lower[cosine + 2] == -kInfinity && program.upper[cosine + 2] == kInfinity);
  // A huge point keeps a tight enclosure from the exact argument reduction.
  PHX_CHECK(program.lower[sine + 3] > -1.0 && program.upper[sine + 3] < 1.0);
  PHX_CHECK(bits(program.upper[sine + 3]) - bits(program.lower[sine + 3]) <= 2 ||
            bits(program.lower[sine + 3]) - bits(program.upper[sine + 3]) <= 2);
  for (std::int64_t element = 0; element < 8; ++element) {
    const double lo = program.lower[input + element], hi = program.upper[input + element];
    if (!std::isfinite(lo) || !std::isfinite(hi)) continue;
    for (int sample = 0; sample <= 8; ++sample) {
      const double x = lo + (hi - lo) * (static_cast<double>(sample) / 8.0);
      const auto point_sine = phx::mc::elementary::sin(x, x);
      const auto point_cosine = phx::mc::elementary::cos(x, x);
      PHX_CHECK(program.lower[sine + element] <= point_sine.lower);
      PHX_CHECK(point_sine.upper <= program.upper[sine + element]);
      PHX_CHECK(program.lower[cosine + element] <= point_cosine.lower);
      PHX_CHECK(point_cosine.upper <= program.upper[cosine + element]);
    }
  }
}

void unary_interval_enclosures() {
  Program program;
  const std::int64_t input =
      program.values({{-1.0, 2.0}, {4.0, 9.0}, {-4.0, -1.0}, {1.0, 4.0}, {-1.0, 1.0}});
  const std::int64_t square = program.slots(5);
  program.row(kSquare, square, 5, input, -1, -1, -1, 0);
  const std::int64_t root = program.slots(5);
  program.row(kSqrt, root, 5, input, -1, -1, -1, 0);
  const std::int64_t reciprocal = program.slots(5);
  program.row(kReciprocal, reciprocal, 5, input, -1, -1, -1, 0);
  PHX_CHECK(execute(program) == PHX_MC_OK);
  PHX_CHECK(program.lower[square] == 0.0 && program.upper[square] >= 4.0);
  PHX_CHECK(program.lower[square + 1] <= 16.0 && program.upper[square + 1] >= 81.0);
  PHX_CHECK(program.lower[root] <= 0.0 && program.upper[root] >= std::sqrt(2.0));
  PHX_CHECK(program.lower[root + 2] == -kInfinity);
  PHX_CHECK(program.upper[root + 2] == kInfinity);
  PHX_CHECK(program.lower[reciprocal] == -kInfinity);
  PHX_CHECK(program.upper[reciprocal] == kInfinity);
  PHX_CHECK(program.lower[reciprocal + 1] <= 1.0 / 9.0);
  PHX_CHECK(program.upper[reciprocal + 1] >= 1.0 / 4.0);
}

// Shared vectors with tests/unit/geometry (host interpreter): comparisons
// decide only on certain endpoint order; select returns a decided branch's
// enclosure exactly and the hull of both when undecided, infinite endpoints
// included.
void compare_and_select() {
  Program program;
  const std::int64_t left = program.values(
      {{2.0, 3.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 1.0}, {-kInfinity, 0.0}, {-1.0, 1.0}});
  const std::int64_t right = program.values(
      {{0.0, 1.0}, {1.0, 2.0}, {0.5, 0.5}, {1.0, 1.0}, {0.0, kInfinity}, {0.0, 0.0}});
  const std::int64_t greater = program.slots(6);
  program.row(kCompare, greater, 6, left, -1, right, -1, 0);
  const std::int64_t at_least = program.slots(6);
  program.row(kCompare, at_least, 6, left, -1, right, -1, 1);
  const std::int64_t select = program.slots(6);
  program.row(kSelect, select, 6, greater, -1, left, -1, right - left);
  PHX_CHECK(execute(program) == PHX_MC_OK);
  const double expected_greater[6][2] = {{1, 1}, {0, 0}, {0, 1}, {0, 0}, {0, 0}, {0, 1}};
  const double expected_at_least[6][2] = {{1, 1}, {0, 1}, {0, 1}, {1, 1}, {0, 1}, {0, 1}};
  // case0 = left, case1 = right.
  const double expected_select[6][2] = {{0.0, 1.0}, {0.0, 1.0},       {0.0, 1.0},
                                        {1.0, 1.0}, {-kInfinity, 0.0}, {-1.0, 1.0}};
  for (int k = 0; k < 6; ++k) {
    PHX_CHECK(program.lower[greater + k] == expected_greater[k][0]);
    PHX_CHECK(program.upper[greater + k] == expected_greater[k][1]);
    PHX_CHECK(program.lower[at_least + k] == expected_at_least[k][0]);
    PHX_CHECK(program.upper[at_least + k] == expected_at_least[k][1]);
    PHX_CHECK(program.lower[select + k] == expected_select[k][0]);
    PHX_CHECK(program.upper[select + k] == expected_select[k][1]);
  }
  // Invalid strictness flags and case blocks reaching the output are refused.
  Program invalid;
  const std::int64_t value = invalid.values({{0.0, 1.0}, {1.0, 2.0}});
  const std::int64_t out = invalid.slots(1);
  invalid.row(kCompare, out, 1, value, -1, value + 1, -1, 2);
  PHX_CHECK(execute(invalid) == PHX_MC_INVALID_INPUT);
  Program reaching;
  const std::int64_t cases = reaching.values({{0.0, 1.0}, {1.0, 2.0}});
  const std::int64_t target = reaching.slots(2);
  reaching.row(kSelect, target, 2, cases, -1, cases, -1, 1);
  PHX_CHECK(execute(reaching) == PHX_MC_INVALID_INPUT);
}

// Shared vectors with tests/unit/geometry (host interpreter): endpointwise and
// exact, infinite endpoints included.
void maximum_and_minimum() {
  Program program;
  const std::int64_t left =
      program.values({{-1.0, 2.0}, {-kInfinity, 0.5}, {3.0, kInfinity}, {-2.0, -1.0}});
  const std::int64_t right =
      program.values({{0.0, 1.0}, {1e-12, 1e-12}, {-kInfinity, 4.0}, {-3.0, 5.0}});
  const std::int64_t high = program.slots(4);
  program.row(kMaximum, high, 4, left, -1, right, -1, 0);
  const std::int64_t low = program.slots(4);
  program.row(kMinimum, low, 4, left, -1, right, -1, 0);
  PHX_CHECK(execute(program) == PHX_MC_OK);
  const double expected_high[4][2] = {
      {0.0, 2.0}, {1e-12, 0.5}, {3.0, kInfinity}, {-2.0, 5.0}};
  const double expected_low[4][2] = {
      {-1.0, 1.0}, {-kInfinity, 1e-12}, {-kInfinity, 4.0}, {-3.0, -1.0}};
  for (int k = 0; k < 4; ++k) {
    PHX_CHECK(program.lower[high + k] == expected_high[k][0]);
    PHX_CHECK(program.upper[high + k] == expected_high[k][1]);
    PHX_CHECK(program.lower[low + k] == expected_low[k][0]);
    PHX_CHECK(program.upper[low + k] == expected_low[k][1]);
  }
}

// The exp and log opcodes are the rigorous elementary kernels: exp(0) = 1
// exactly, exp(-inf) = 0, and log sends a nonpositive endpoint to the matching
// infinity.
void exponential_and_logarithm() {
  Program program;
  const std::int64_t input = program.values(
      {{0.0, 1.0}, {-kInfinity, 0.0}, {-1.0, 2.0}, {1.0, kInfinity}, {-2.0, -1.0}});
  const std::int64_t exponential = program.slots(5);
  program.row(kExp, exponential, 5, input, -1, -1, -1, 0);
  const std::int64_t logarithm = program.slots(5);
  program.row(kLog, logarithm, 5, input, -1, -1, -1, 0);
  PHX_CHECK(execute(program) == PHX_MC_OK);
  const double lows[5] = {0.0, -kInfinity, -1.0, 1.0, -2.0};
  const double highs[5] = {1.0, 0.0, 2.0, kInfinity, -1.0};
  for (int k = 0; k < 5; ++k) {
    const auto exp_expected = phx::mc::elementary::exp(lows[k], highs[k]);
    const auto log_expected = phx::mc::elementary::log(lows[k], highs[k]);
    PHX_CHECK(same(program.lower[exponential + k], exp_expected.lower));
    PHX_CHECK(same(program.upper[exponential + k], exp_expected.upper));
    PHX_CHECK(same(program.lower[logarithm + k], log_expected.lower));
    PHX_CHECK(same(program.upper[logarithm + k], log_expected.upper));
  }
  PHX_CHECK(same(program.lower[exponential], 1.0));
  PHX_CHECK(same(program.lower[exponential + 1], 0.0) && same(program.upper[exponential + 1], 1.0));
  PHX_CHECK(same(program.lower[logarithm + 3], 0.0) && program.upper[logarithm + 3] == kInfinity);
  PHX_CHECK(program.lower[logarithm + 4] == -kInfinity && program.upper[logarithm + 4] == kInfinity);
}

void interval_capability_query() {
  std::uint64_t opcodes = 0;
  const std::uint64_t expected =
      (std::uint64_t{1} << kGather) | (std::uint64_t{1} << kAdd) |
      (std::uint64_t{1} << kSubtract) | (std::uint64_t{1} << kSubtractSame) |
      (std::uint64_t{1} << kMultiply) | (std::uint64_t{1} << kNegate) |
      (std::uint64_t{1} << kReduceSum) | (std::uint64_t{1} << kSine) |
      (std::uint64_t{1} << kCosine) | (std::uint64_t{1} << kSqrt) |
      (std::uint64_t{1} << kReciprocal) | (std::uint64_t{1} << kSquare) |
      (std::uint64_t{1} << kCompare) | (std::uint64_t{1} << kSelect) |
      (std::uint64_t{1} << kMaximum) | (std::uint64_t{1} << kMinimum) |
      (std::uint64_t{1} << kExp) | (std::uint64_t{1} << kLog);
  PHX_CHECK(phx_mc_interval_program_capabilities(&opcodes) == PHX_MC_OK);
  PHX_CHECK(opcodes == expected);
  PHX_CHECK(phx_mc_interval_program_capabilities(nullptr) == PHX_MC_INVALID_ARGUMENT);
}

Program random_program(std::mt19937_64& random) {
  Program program;
  std::uniform_real_distribution<double> finite(-3.0, 3.0);
  const std::int64_t input_count = 12;
  for (std::int64_t i = 0; i < input_count; ++i) {
    const double center = finite(random);
    const double width = std::fabs(finite(random)) * 0.125;
    program.lower.push_back(center - width);
    program.upper.push_back(center + width);
  }
  const std::int64_t indexed = program.index({3, 1, 8, 0, 5, 11});
  std::int64_t out = program.slots(6);
  program.row(kGather, out, 6, 0, indexed, -1, -1, 0);
  for (int row = 0; row < 5; ++row) {
    out = program.slots(6);
    const auto opcode = static_cast<std::int64_t>(kAdd + (random() % 4));
    if (opcode == kSubtractSame) {
      const std::int64_t source = input_count + 6 * row;
      program.row(opcode, out, 6, source, -1, source, -1, 0);
    } else {
      const std::int64_t a = input_count + 6 * row;
      const std::int64_t b = input_count + 6 * static_cast<std::int64_t>(random() % (row + 1));
      program.row(opcode, out, 6, a, -1, b, -1, 0);
    }
  }
  return program;
}

void deterministic_random_programs() {
  std::mt19937_64 random(11);
  for (int trial = 0; trial < 250; ++trial) {
    Program program = random_program(random);
    std::vector<double> expected_lower, expected_upper;
    reference(program, expected_lower, expected_upper);
    Program repeat = program;
    PHX_CHECK(execute(program) == PHX_MC_OK);
    PHX_CHECK(identical(program.lower, expected_lower));
    PHX_CHECK(identical(program.upper, expected_upper));
    PHX_CHECK(execute(repeat) == PHX_MC_OK);
    PHX_CHECK(identical(program.lower, repeat.lower));
    PHX_CHECK(identical(program.upper, repeat.upper));
  }
}

void validation_and_empty_programs() {
  Program valid = base_program();
  check_reference(valid);
  auto invalid_row = [&](std::array<std::int64_t, 8> fields) {
    Program invalid = base_program();
    invalid.rows.insert(invalid.rows.end(), fields.begin(), fields.end());
    const Program original = invalid;
    PHX_CHECK(execute(invalid) == PHX_MC_INVALID_INPUT);
    PHX_CHECK(unchanged(invalid, original));
  };
  invalid_row({0, 10, 1, 0, -1, -1, -1, 0});
  invalid_row({13, 10, 1, 0, -1, -1, -1, 0});
  invalid_row({kNegate, 11, 2, 0, -1, -1, -1, 0});
  invalid_row({kNegate, 10, -1, 0, -1, -1, -1, 0});
  invalid_row({kNegate, 9, 1, 0, -1, -1, -1, 0});
  invalid_row({kNegate, 10, 2, 9, -1, -1, -1, 0});
  invalid_row({kNegate, 10, 2, 0, 3, -1, -1, 0});
  invalid_row({kNegate, 10, 1, 0, -2, -1, -1, 0});
  invalid_row({kAdd, 10, 2, 0, -1, 9, -1, 0});
  invalid_row({kNegate, 10, 1, 0, -1, 0, -1, 0});
  invalid_row({kSine, 10, 1, 0, -1, -1, 0, 0});
  invalid_row({kReduceSum, 10, 1, 0, 0, -1, 0, 2});
  invalid_row({kReduceSum, 10, 2, 1, -1, -1, 0, 5});
  invalid_row({kReduceSum, 10, 2, 0, -1, -1, 0, kMax});
  Program bad_gamma = base_program();
  bad_gamma.parameters[0] = kNaN;
  bad_gamma.row(kReduceSum, 10, 1, 0, -1, -1, 0, 0);
  const Program original_bad_gamma = bad_gamma;
  PHX_CHECK(execute(bad_gamma) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(unchanged(bad_gamma, original_bad_gamma));
  Program invalid_interval = base_program();
  invalid_interval.rows[0] = kMax / 8;
  const Program original_interval = invalid_interval;
  PHX_CHECK(execute(invalid_interval) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(unchanged(invalid_interval, original_interval));

  Program constants = base_program();
  constants.rows.clear();
  const Program constant_copy = constants;
  PHX_CHECK(execute(constants) == PHX_MC_OK && unchanged(constants, constant_copy));
  Program zero_rows = base_program();
  zero_rows.rows = {kNegate, 6, 0, 6, -1, -1, -1, 0,
                    kAdd, 6, 0, 0, 4, 0, 4, 0,
                    kReduceSum, 6, 0, 6, -1, -1, 0, 7,
                    kSine, 12, 0, 12, -1, -1, -1, 0};
  const Program zero_copy = zero_rows;
  PHX_CHECK(execute(zero_rows) == PHX_MC_OK && unchanged(zero_rows, zero_copy));
  PHX_CHECK(phx_mc_interval_program_execute(nullptr, 0, nullptr, 0, nullptr, 0,
                                              nullptr, nullptr, 0) == PHX_MC_OK);

  Program args = base_program();
  const Program original = args;
  const auto* rows = args.rows.data();
  const auto* indices = args.indices.data();
  const auto* parameters = args.parameters.data();
  auto call = [&](const std::int64_t* r, std::int64_t rc, const std::int64_t* i,
                  std::int64_t ic, const double* p, std::int64_t pc, double* lo, double* hi,
                  std::int64_t extent) {
    return phx_mc_interval_program_execute(r, rc, i, ic, p, pc, lo, hi, extent);
  };
  const auto extent = static_cast<std::int64_t>(args.lower.size());
  PHX_CHECK(call(rows, -1, indices, 4, parameters, 1, args.lower.data(), args.upper.data(), extent) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(rows, args.count(), indices, -1, parameters, 1, args.lower.data(), args.upper.data(),
                 extent) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(rows, args.count(), indices, 4, parameters, -1, args.lower.data(), args.upper.data(),
                 extent) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(rows, args.count(), indices, 4, parameters, 1, args.lower.data(), args.upper.data(),
                 -1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(nullptr, args.count(), indices, 4, parameters, 1, args.lower.data(),
                 args.upper.data(), extent) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(rows, args.count(), indices, 4, parameters, 1, nullptr, args.upper.data(), extent) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(rows, args.count(), indices, 4, parameters, 1, args.lower.data(), nullptr, extent) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(rows, args.count(), indices, 4, parameters, 1, args.lower.data(), args.upper.data(),
                 9) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(call(rows, kMax / 8 + 1, indices, 4, parameters, 1, args.lower.data(),
                 args.upper.data(), extent) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(unchanged(args, original));
}

void prepared_ownership_and_refusal() {
  Program source;
  const auto values = source.values({{1.0, 2.0}, {3.0, 4.0}});
  const auto indices = source.index({1, 0});
  const auto gamma = source.parameter(0.25);
  const auto gathered = source.slots(2);
  source.row(kGather, gathered, 2, values, indices, -1, -1, 0);
  const auto reduced = source.slots(1);
  source.row(kReduceSum, reduced, 1, gathered, -1, -1, gamma, 2);
  Program reference_program = source;
  phx::mc::PreparedIntervalProgram prepared;
  const auto extent = static_cast<int64_t>(source.lower.size());
  PHX_CHECK(prepared.execute(source.lower.data(), source.upper.data(), extent) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(prepared.bind(source.rows.data(), source.count(), source.indices.data(),
                          source.indices.size(), source.parameters.data(), source.parameters.size(),
                          extent) == PHX_MC_OK);
  PHX_CHECK(prepared.execute(source.lower.data(), source.upper.data(), extent - 1) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(prepared.execute(nullptr, source.upper.data(), extent) == PHX_MC_INVALID_ARGUMENT);
  source.rows[0] = 999;
  source.indices[0] = 999;
  source.parameters[0] = -1.0;
  PHX_CHECK(execute(reference_program) == PHX_MC_OK);
  PHX_CHECK(prepared.execute(source.lower.data(), source.upper.data(), extent) == PHX_MC_OK);
  PHX_CHECK(identical(source.lower, reference_program.lower));
  PHX_CHECK(identical(source.upper, reference_program.upper));
  const Program before = source;
  PHX_CHECK(phx_mc_interval_program_execute(
                source.rows.data(), source.count(), source.indices.data(), source.indices.size(),
                source.parameters.data(), source.parameters.size(), source.lower.data(),
                source.upper.data(), extent) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(prepared.bind(source.rows.data(), source.count(), source.indices.data(),
                          source.indices.size(), source.parameters.data(), source.parameters.size(),
                          extent) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(prepared.execute(source.lower.data(), source.upper.data(), extent) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(unchanged(source, before));
}

void floating_point_modes() {
  Program subnormal;
  const std::int64_t source = subnormal.values({{kSmallest, 0x1p-1060}, {0x1p-1023, 0x1p-1022}});
  const std::int64_t result = subnormal.slots(4);
  subnormal.row(kAdd, result, 2, source, -1, source, -1, 0);
  subnormal.row(kMultiply, result + 2, 2, source, -1, source, -1, 0);
  Program nearest = subnormal;
  PHX_CHECK(execute(nearest) == PHX_MC_OK);
  PHX_CHECK(same(nearest.lower[result], kSmallest));
  PHX_CHECK(same(nearest.lower[result + 2], -kSmallest));
  PHX_CHECK(same(nearest.upper[result + 2], kSmallest));

  struct Hazard { bool flush; int rounding; };
  const std::array<Hazard, 5> hazards{{{true, FE_TONEAREST}, {false, FE_UPWARD},
      {false, FE_DOWNWARD}, {false, FE_TOWARDZERO}, {true, FE_DOWNWARD}}};
  const auto initial_mode = phx::mc::floating_point_mode();
  const int initial_rounding = std::fegetround();
  for (const Hazard& hazard : hazards) {
    if (hazard.flush && !phx::mc::kFlushControl) continue;
    phx::mc::set_floating_point_mode(hazard.flush ? phx::mc::kFlushingMode
                                                  : phx::mc::kGradualNearestMode);
    PHX_CHECK(std::fesetround(hazard.rounding) == 0);
    const auto installed = phx::mc::floating_point_mode();
    Program call = subnormal;
    PHX_CHECK(execute(call) == PHX_MC_OK);
    PHX_CHECK(identical(call.lower, nearest.lower));
    PHX_CHECK(identical(call.upper, nearest.upper));
    PHX_CHECK(phx::mc::floating_point_mode() == installed);
    PHX_CHECK(std::fegetround() == hazard.rounding);
  }
  phx::mc::set_floating_point_mode(initial_mode);
  PHX_CHECK(std::fesetround(initial_rounding) == 0);
  PHX_CHECK(phx::mc::floating_point_mode() == initial_mode);
}

}  // namespace

int main() {
  arithmetic_and_reductions();
  zero_times_unbounded_multiply();
  trigonometric_enclosures();
  unary_interval_enclosures();
  interval_capability_query();
  compare_and_select();
  maximum_and_minimum();
  exponential_and_logarithm();
  deterministic_random_programs();
  validation_and_empty_programs();
  prepared_ownership_and_refusal();
  floating_point_modes();
  return phx::mc::test::finish("test_interval_program");
}
