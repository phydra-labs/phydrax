// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Straight-line outward interval execution. A whole program is validated before
// the first workspace write; validated rows execute in source order.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "elementary.hpp"
#include "floating_point_mode.hpp"
#include "interval_arithmetic.hpp"
#include "interval_program.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"
#include "rounding.hpp"

namespace {

using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr std::int64_t kFields = 8;
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
// Three-valued comparison a > b (aux 0) or a >= b (aux 1) as an indicator
// interval: [1, 1] certainly true, [0, 0] certainly false, [0, 1] undecided.
constexpr std::int64_t kCompare = 13;
// select_n(p, case0, case1): case1 when p is certainly true (p.lower > 0),
// case0 when certainly false (p.upper <= 0), else the hull of both. p is
// operand a; case0 is the contiguous block at b and case1 the block at b + aux.
constexpr std::int64_t kSelect = 14;
// Elementwise maximum/minimum: monotone, so endpointwise and exact (NumPy NaN
// propagation, -0 below +0).
constexpr std::int64_t kMaximum = 15;
constexpr std::int64_t kMinimum = 16;
// exp and log: the rigorous elementary kernels (elementary.hpp) with the
// conventions of phx_mc_interval_elementary, as in the host interpreter.
constexpr std::int64_t kExp = 17;
constexpr std::int64_t kLog = 18;
// Branch semantics of non-smooth programs, as the host interpreter's
// comparisons, absolute value and value-identity decisions
// (phydrax/geometry/_interval_enclosure.py, _ValueIdentity).
// eq (aux 0) / ne (aux 1) as an indicator interval: eq is certainly true only
// when both operands are the same point, certainly false when they are disjoint.
constexpr std::int64_t kEqual = 19;
// |x| with the host's outward rounding of both endpoints.
constexpr std::int64_t kAbsolute = 20;
// Winner flags of an elementwise min (aux 0) / max (aux 1) of two values with
// distinct identities. Writes 2 * size flag pairs: [out, out + size) for
// operand a and [out + size, out + 2 size) for operand b. A flag pair is not
// an interval: lower = 1 where one operand strictly wins on the whole box
// (decided), upper = 1 where this operand is that winner.
constexpr std::int64_t kExtremumPair = 21;
// Winner flags of a min (aux 0) / max (aux 1) reduction: size outputs, b lanes
// each, operand element (k, l) at a + indices[a_index + k * b + l] (or
// a + k * b + l when a_index is -1). b_index names b representative lanes
// (lanes of equal identity share the first one), or is -1 when every lane is
// its own. Writes size * b flag pairs at out + k * b + l.
constexpr std::int64_t kExtremumReduce = 22;
// eq (aux 0) / ne (aux 1) of a min/max operand against its result: where the
// flag pair b is decided the result is the operand's winner flag (negated for
// ne) exactly, else the comparison interval a.
constexpr std::int64_t kDecide = 23;
// A sum of a reduction's attainment indicators counts at least one element:
// lower = maximum(lower, 1), upper unchanged.
constexpr std::int64_t kAttain = 24;
constexpr std::int64_t kDivideElementwise = 25;
constexpr std::int64_t kAtan = 26;
constexpr std::int64_t kPowerElementwise = 27;
constexpr std::int64_t kTan = 28;

constexpr std::uint64_t kIntervalOpcodeCapabilities =
    (std::uint64_t{1} << kGather) | (std::uint64_t{1} << kAdd) |
    (std::uint64_t{1} << kSubtract) | (std::uint64_t{1} << kSubtractSame) |
    (std::uint64_t{1} << kMultiply) | (std::uint64_t{1} << kNegate) |
    (std::uint64_t{1} << kReduceSum) | (std::uint64_t{1} << kSine) |
    (std::uint64_t{1} << kCosine) | (std::uint64_t{1} << kSqrt) |
    (std::uint64_t{1} << kReciprocal) | (std::uint64_t{1} << kSquare) |
    (std::uint64_t{1} << kCompare) | (std::uint64_t{1} << kSelect) |
    (std::uint64_t{1} << kMaximum) | (std::uint64_t{1} << kMinimum) |
    (std::uint64_t{1} << kExp) | (std::uint64_t{1} << kLog) |
    (std::uint64_t{1} << kEqual) | (std::uint64_t{1} << kAbsolute) |
    (std::uint64_t{1} << kExtremumPair) | (std::uint64_t{1} << kExtremumReduce) |
    (std::uint64_t{1} << kDecide) | (std::uint64_t{1} << kAttain) |
    (std::uint64_t{1} << kDivideElementwise) | (std::uint64_t{1} << kAtan) |
    (std::uint64_t{1} << kPowerElementwise) | (std::uint64_t{1} << kTan);

// Branch-evidence event rows: kind, test, size, a, a_index, boxes. Element k
// is the workspace entry a + (a_index < 0 ? k : indices[a_index + k]); where
// the test holds, the event clears the flags of box indices[boxes + k], or of
// every box when that entry or `boxes` is -1. A kink clears smooth; a jump
// clears smooth and continuous.
constexpr std::int64_t kEventFields = 6;
constexpr std::int64_t kKink = 0;
constexpr std::int64_t kJump = 1;
constexpr std::int64_t kMeetsZero = 0;       // !(lower > 0 || upper < 0)
constexpr std::int64_t kNotPositive = 1;     // !(lower > 0)
constexpr std::int64_t kNotNonnegative = 2;  // !(lower >= 0)
constexpr std::int64_t kUndecided = 3;       // lower != upper
constexpr std::int64_t kNonfinite = 4;       // !(isfinite(lower) && isfinite(upper))
constexpr std::int64_t kAlways = 5;

struct Row {
  std::int64_t opcode;
  std::int64_t out;
  std::int64_t size;
  std::int64_t a;
  std::int64_t a_index;
  std::int64_t b;
  std::int64_t b_index;
  std::int64_t aux;
};

Row row_at(const std::int64_t* instructions, std::int64_t row) noexcept {
  const std::int64_t* fields = instructions + row * kFields;
  return {fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7]};
}

// IEEE 754-2019 minimum/maximum: preserve the first NaN operand and order -0
// below +0, matched by the host interpreter's IEEE endpoint helpers.
double minimum(double first, double second) noexcept {
  if (std::isnan(first)) return first;
  if (std::isnan(second)) return second;
  if (first == second) return std::signbit(first) ? first : second;
  return first < second ? first : second;
}

double maximum(double first, double second) noexcept {
  if (std::isnan(first)) return first;
  if (std::isnan(second)) return second;
  if (first == second) return std::signbit(first) ? second : first;
  return first > second ? first : second;
}

double down(double value) noexcept { return next_down(value); }

double up(double value) noexcept { return next_up(value); }

// Endpoint product. Interval endpoints bound real numbers; an infinite
// endpoint marks an unbounded side, never a member, so an exact zero
// endpoint times it contributes 0 (IEEE 1788 set-based multiplication), as
// in the host interval multiply.
double product(double first, double second) noexcept {
  if ((first == 0.0 && std::isinf(second)) || (std::isinf(first) && second == 0.0)) return 0.0;
  return first * second;
}

// Every element offset + (index < 0 ? k : indices[index + k]), k < size,
// lies in [0, limit), with 0 <= offset <= limit.
bool operand_valid(const std::int64_t* indices, std::int64_t index_count, std::int64_t offset,
                   std::int64_t index, std::int64_t size, std::int64_t limit) noexcept {
  if (offset < 0 || offset > limit) return false;
  const std::int64_t room = limit - offset;
  if (index == -1) return size <= room;
  if (index < 0 || index > index_count || size > index_count - index) return false;
  for (std::int64_t k = 0; k < size; ++k) {
    const std::int64_t value = indices[index + k];
    if (value < 0 || value >= room) return false;
  }
  return true;
}

// Output entries of a row: 2 size flag pairs for kExtremumPair, size * b for
// kExtremumReduce, else size. Only called on rows with a valid extent.
std::int64_t row_extent(const Row& row) noexcept {
  if (row.opcode == kExtremumPair || row.opcode == kPowerElementwise) return 2 * row.size;
  if (row.opcode == kExtremumReduce) return row.size * row.b;
  return row.size;
}

// Representative lanes: rep[l] in [0, l] and every representative is its own.
bool representatives_valid(const std::int64_t* indices, std::int64_t index_count,
                           std::int64_t index, std::int64_t lanes) noexcept {
  if (index == -1) return true;
  if (index < 0 || index > index_count || lanes > index_count - index) return false;
  for (std::int64_t lane = 0; lane < lanes; ++lane) {
    const std::int64_t representative = indices[index + lane];
    if (representative < 0 || representative > lane) return false;
    if (indices[index + representative] != representative) return false;
  }
  return true;
}

bool row_valid(const Row& row, std::int64_t previous_end, const std::int64_t* indices,
               std::int64_t index_count, const double* parameters, std::int64_t parameter_count,
               std::int64_t workspace_size) noexcept {
  if (row.opcode < kGather || row.opcode > kTan) return false;
  if (row.size < 0 || row.out < previous_end || row.out > workspace_size) return false;
  const std::int64_t room = workspace_size - row.out;
  switch (row.opcode) {
    case kExtremumPair:
    case kPowerElementwise:
      if (row.size > room / 2) return false;
      break;
    case kExtremumReduce:
      if (row.b < 1 || row.size > room / row.b) return false;
      break;
    default:
      if (row.size > room) return false;
  }
  const auto operand = [&](std::int64_t offset, std::int64_t index) {
    return operand_valid(indices, index_count, offset, index, row.size, row.out);
  };
  switch (row.opcode) {
    case kReduceSum: {
      if ((row.a_index != -1 && row.a_index != -2) || row.b != -1 || row.aux < 0) return false;
      if (row.b_index < 0 || row.b_index >= parameter_count) return false;
      const double gamma = parameters[row.b_index];
      if (!std::isfinite(gamma) || !(gamma >= 0.0)) return false;
      // a + size * aux <= out without forming the product.
      if (row.a < 0 || row.a > row.out) return false;
      return row.aux == 0 || row.size <= (row.out - row.a) / row.aux;
    }
    case kAdd:
    case kSubtract:
    case kSubtractSame:
    case kMultiply:
    case kDivideElementwise:
    case kPowerElementwise:
    case kMaximum:
    case kMinimum:
      return row.aux == 0 && operand(row.a, row.a_index) && operand(row.b, row.b_index);
    case kCompare:
    case kEqual:
    case kExtremumPair:
    case kDecide:
      return (row.aux == 0 || row.aux == 1) && operand(row.a, row.a_index) &&
             operand(row.b, row.b_index);
    case kExtremumReduce:
      // size * b operand entries; the product fits since it fits the output room.
      return (row.aux == 0 || row.aux == 1) &&
             operand_valid(indices, index_count, row.a, row.a_index, row.size * row.b,
                           row.out) &&
             representatives_valid(indices, index_count, row.b_index, row.b);
    case kSelect:
      // Both case blocks are contiguous and precede the output.
      return row.b_index == -1 && row.aux >= 0 && row.b >= 0 && row.b <= row.out &&
             row.aux <= row.out - row.b && row.size <= row.out - row.b - row.aux &&
             operand(row.a, row.a_index);
    default:
      return row.b == -1 && row.b_index == -1 && row.aux == 0 && operand(row.a, row.a_index);
  }
}

// Whether every row is valid in order. No workspace values are touched here.
bool program_valid(const std::int64_t* instructions, std::int64_t instruction_count,
                   const std::int64_t* indices, std::int64_t index_count,
                   const double* parameters, std::int64_t parameter_count,
                   std::int64_t workspace_size) noexcept {
  std::int64_t previous_end = 0;
  for (std::int64_t row_index = 0; row_index < instruction_count; ++row_index) {
    const Row row = row_at(instructions, row_index);
    if (!row_valid(row, previous_end, indices, index_count, parameters, parameter_count,
                   workspace_size)) {
      return false;
    }
    previous_end = row.out + row_extent(row);
  }
  return true;
}

// Whether every event row reads valid workspace entries and names valid boxes.
bool events_valid(const std::int64_t* events, std::int64_t event_count,
                  const std::int64_t* indices, std::int64_t index_count,
                  std::int64_t workspace_size, std::int64_t box_count) noexcept {
  for (std::int64_t row = 0; row < event_count; ++row) {
    const std::int64_t* fields = events + row * kEventFields;
    const std::int64_t kind = fields[0], test = fields[1], size = fields[2];
    const std::int64_t a = fields[3], a_index = fields[4], boxes = fields[5];
    if ((kind != kKink && kind != kJump) || test < kMeetsZero || test > kAlways || size < 0)
      return false;
    if (!operand_valid(indices, index_count, a, a_index, size, workspace_size)) return false;
    if (boxes == -1) continue;
    if (boxes < 0 || boxes > index_count || size > index_count - boxes) return false;
    for (std::int64_t k = 0; k < size; ++k) {
      const std::int64_t box = indices[boxes + k];
      if (box < -1 || box >= box_count) return false;
    }
  }
  return true;
}

inline std::int64_t address(const std::int64_t* indices, std::int64_t offset,
                            std::int64_t index, std::int64_t element) noexcept {
  return offset + (index < 0 ? element : indices[index + element]);
}

// In-order REDUCE_SUM accumulators for one element: lower and upper sums and
// the scale sum of max(|lo|, |hi|), each initialized from +0, and whether every
// term's lower (upper) endpoint is nonnegative (nonpositive).
struct Reduction {
  double lower = 0.0;
  double upper = 0.0;
  double scale = 0.0;
  bool nonnegative = true;
  bool nonpositive = true;
};

inline void accumulate(const double* lower, const double* upper, std::int64_t first,
                       std::int64_t last, Reduction& sum) noexcept {
  double lower_sum = sum.lower;
  double upper_sum = sum.upper;
  double scale = sum.scale;
  bool nonnegative = sum.nonnegative;
  bool nonpositive = sum.nonpositive;
  for (std::int64_t term = first; term < last; ++term) {
    lower_sum += lower[term];
    upper_sum += upper[term];
  }
  for (std::int64_t term = first; term < last; ++term) {
    scale += maximum(std::fabs(lower[term]), std::fabs(upper[term]));
    nonnegative = nonnegative && lower[term] >= 0.0;
    nonpositive = nonpositive && upper[term] <= 0.0;
  }
  sum = {lower_sum, upper_sum, scale, nonnegative, nonpositive};
}

// A sum of nonnegative (nonpositive) terms keeps its sign through outward
// rounding, as the host interval sum's sign clamp: a sum of squares stays an
// admissible base for roots and real powers.
inline void store_reduction(const Reduction& sum, double gamma, double* lower, double* upper,
                            std::int64_t target) noexcept {
  const double low = sum.lower - gamma * sum.scale;
  const double high = sum.upper + gamma * sum.scale;
  if (std::isnan(low) || std::isnan(high)) {
    lower[target] = -std::numeric_limits<double>::infinity();
    upper[target] = std::numeric_limits<double>::infinity();
  } else {
    lower[target] = sum.nonnegative ? maximum(down(low), 0.0) : down(low);
    upper[target] = sum.nonpositive ? minimum(up(high), 0.0) : up(high);
  }
}

struct PowerInterval {
  double lower;
  double upper;
};

// Exact binary64 integrality, without a platform math-library decision.
bool integral_binary64(double value) noexcept {
  const auto bits = std::bit_cast<std::uint64_t>(value);
  const auto magnitude = bits & UINT64_C(0x7fffffffffffffff);
  const int exponent = static_cast<int>((magnitude >> 52) & UINT64_C(0x7ff));
  if (exponent == 0x7ff) return false;
  if (magnitude == 0) return true;
  const int shift = exponent - 1023;
  if (shift < 0) return false;
  if (shift >= 52) return true;
  return (magnitude & ((UINT64_C(1) << (52 - shift)) - 1)) == 0;
}

PowerInterval integer_power(PowerInterval base, double exponent) noexcept {
  if (exponent == 0.0) return {1.0, 1.0};
  if (exponent >= 0x1p53 || (static_cast<std::uint64_t>(exponent) & 1U) == 0) {
    const auto half = integer_power(base, exponent * 0.5);
    const bool crosses = half.lower <= 0.0 && half.upper >= 0.0;
    const double first = half.lower * half.lower, second = half.upper * half.upper;
    const double low = crosses ? 0.0 : minimum(first, second);
    const double high = maximum(first, second);
    if (std::isnan(low) || std::isnan(high)) return {0.0, std::numeric_limits<double>::infinity()};
    return {maximum(down(low), 0.0), up(high)};
  }
  const auto previous = integer_power(base, exponent - 1.0);
  PowerInterval result;
  phx::mc::interval::interval_product(previous.lower, previous.upper, base.lower, base.upper,
                                       result.lower, result.upper);
  return result;
}

PowerInterval divide_interval(double a_low, double a_high, double b_low, double b_high) noexcept {
  if (b_low <= 0.0 && b_high >= 0.0)
    return {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  const double rlow = 1.0 / b_high, rhigh = 1.0 / b_low;
  const auto exact = [](double numerator, double denominator, double quotient) {
    const double product = quotient * denominator;
    return std::isfinite(quotient) && denominator != 0.0 && product == numerator &&
           phx::mc::interval::product_is_exact(quotient, denominator, product);
  };
  double reciprocal_low, reciprocal_high;
  phx::mc::interval::round_out_inexact(
      rlow, rhigh, exact(1.0, b_high, rlow), exact(1.0, b_low, rhigh),
      reciprocal_low, reciprocal_high);
  PowerInterval result;
  phx::mc::interval::interval_product(a_low, a_high, reciprocal_low, reciprocal_high,
                                       result.lower, result.upper);
  const double quotient = a_low / b_low;
  if (a_low == a_high && b_low == b_high && exact(a_low, b_low, quotient))
    result.lower = result.upper = quotient;
  return result;
}

// Execute elements [begin, end) of a validated row.
void execute_row(const Row& row, std::int64_t begin, std::int64_t end,
                 const std::int64_t* indices, const double* parameters, double* lower,
                 double* upper, bool elementwise_arithmetic) noexcept {
  const std::int64_t out = row.out;
  switch (row.opcode) {
    case kGather:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t input = address(indices, row.a, row.a_index, k);
        lower[out + k] = lower[input];
        upper[out + k] = upper[input];
      }
      return;
    case kNegate:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t input = address(indices, row.a, row.a_index, k);
        const double low = lower[input];
        const double high = upper[input];
        lower[out + k] = elementwise_arithmetic ? down(-high) : -high;
        upper[out + k] = elementwise_arithmetic ? up(-low) : -low;
      }
      return;
    case kSelect:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t predicate = address(indices, row.a, row.a_index, k);
        const std::int64_t first = row.b + k;
        const std::int64_t second = row.b + row.aux + k;
        if (lower[predicate] > 0.0) {
          lower[out + k] = lower[second];
          upper[out + k] = upper[second];
        } else if (upper[predicate] <= 0.0) {
          lower[out + k] = lower[first];
          upper[out + k] = upper[first];
        } else {
          lower[out + k] = minimum(lower[first], lower[second]);
          upper[out + k] = maximum(upper[first], upper[second]);
        }
      }
      return;
    case kCompare:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t a = address(indices, row.a, row.a_index, k);
        const std::int64_t b = address(indices, row.b, row.b_index, k);
        const bool strict = row.aux == 0;
        const bool certainly_true = strict ? lower[a] > upper[b] : lower[a] >= upper[b];
        const bool certainly_false = strict ? upper[a] <= lower[b] : upper[a] < lower[b];
        lower[out + k] = certainly_true ? 1.0 : 0.0;
        upper[out + k] = certainly_false ? 0.0 : 1.0;
      }
      return;
    case kEqual:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t a = address(indices, row.a, row.a_index, k);
        const std::int64_t b = address(indices, row.b, row.b_index, k);
        const bool same_point =
            lower[a] == upper[a] && lower[b] == upper[b] && lower[a] == lower[b];
        const bool disjoint = upper[a] < lower[b] || upper[b] < lower[a];
        const bool certainly_true = row.aux == 0 ? same_point : disjoint;
        const bool certainly_false = row.aux == 0 ? disjoint : same_point;
        lower[out + k] = certainly_true ? 1.0 : 0.0;
        upper[out + k] = certainly_false ? 0.0 : 1.0;
      }
      return;
    case kAbsolute:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t input = address(indices, row.a, row.a_index, k);
        const double low = lower[input];
        const double high = upper[input];
        const bool crosses = low <= 0.0 && high >= 0.0;
        const double low_magnitude = std::fabs(low);
        const double high_magnitude = std::fabs(high);
        const double least = crosses ? 0.0 : minimum(low_magnitude, high_magnitude);
        const double most = maximum(low_magnitude, high_magnitude);
        if (std::isnan(least) || std::isnan(most)) {
          lower[out + k] = -std::numeric_limits<double>::infinity();
          upper[out + k] = std::numeric_limits<double>::infinity();
        } else {
          lower[out + k] = down(least);
          upper[out + k] = up(most);
        }
      }
      return;
    case kExtremumPair:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t a = address(indices, row.a, row.a_index, k);
        const std::int64_t b = address(indices, row.b, row.b_index, k);
        const bool first = row.aux == 0 ? upper[a] < lower[b] : lower[a] > upper[b];
        const bool second = row.aux == 0 ? upper[b] < lower[a] : lower[b] > upper[a];
        const double decided = first || second ? 1.0 : 0.0;
        lower[out + k] = decided;
        upper[out + k] = first ? 1.0 : 0.0;
        lower[out + row.size + k] = decided;
        upper[out + row.size + k] = second ? 1.0 : 0.0;
      }
      return;
    case kExtremumReduce: {
      const std::int64_t lanes = row.b;
      const bool minimum_reduction = row.aux == 0;
      const auto representative = [&](std::int64_t lane) {
        return row.b_index < 0 ? lane : indices[row.b_index + lane];
      };
      for (std::int64_t k = begin; k < end; ++k) {
        const auto entry = [&](std::int64_t lane) {
          return address(indices, row.a, row.a_index, k * lanes + lane);
        };
        // NumPy min/max over the representatives: the first NaN propagates.
        double bound = minimum_reduction ? upper[entry(0)] : lower[entry(0)];
        for (std::int64_t lane = 1; lane < lanes && !std::isnan(bound); ++lane) {
          if (representative(lane) != lane) continue;
          const double value = minimum_reduction ? upper[entry(lane)] : lower[entry(lane)];
          if (std::isnan(value) || (minimum_reduction ? value < bound : value > bound))
            bound = value;
        }
        const auto candidate = [&](std::int64_t lane) {
          return minimum_reduction ? lower[entry(lane)] <= bound : upper[entry(lane)] >= bound;
        };
        std::int64_t count = 0;
        for (std::int64_t lane = 0; lane < lanes; ++lane)
          if (representative(lane) == lane && candidate(lane)) ++count;
        const bool decided = count == 1;
        for (std::int64_t lane = 0; lane < lanes; ++lane) {
          lower[out + k * lanes + lane] = decided ? 1.0 : 0.0;
          upper[out + k * lanes + lane] = decided && candidate(representative(lane)) ? 1.0 : 0.0;
        }
      }
      return;
    }
    case kDecide:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t a = address(indices, row.a, row.a_index, k);
        const std::int64_t b = address(indices, row.b, row.b_index, k);
        if (lower[b] > 0.0) {
          const bool winner = upper[b] > 0.0;
          const double truth = (row.aux == 0 ? winner : !winner) ? 1.0 : 0.0;
          lower[out + k] = truth;
          upper[out + k] = truth;
        } else {
          lower[out + k] = lower[a];
          upper[out + k] = upper[a];
        }
      }
      return;
    case kAttain:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t input = address(indices, row.a, row.a_index, k);
        const double high = upper[input];
        lower[out + k] = maximum(lower[input], 1.0);
        upper[out + k] = high;
      }
      return;
    case kExp:
    case kLog:
    case kAtan:
    case kTan:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t input = address(indices, row.a, row.a_index, k);
        const phx::mc::elementary::Enclosure result =
            row.opcode == kExp ? phx::mc::elementary::exp(lower[input], upper[input])
            : row.opcode == kLog ? phx::mc::elementary::log(lower[input], upper[input])
            : row.opcode == kAtan ? phx::mc::elementary::atan(lower[input], upper[input])
                                  : phx::mc::elementary::tan(lower[input], upper[input]);
        lower[out + k] = result.lower;
        upper[out + k] = result.upper;
      }
      return;
    case kSquare:
    case kSqrt:
    case kReciprocal:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t input = address(indices, row.a, row.a_index, k);
        const double low = lower[input];
        const double high = upper[input];
        if (std::isnan(low) || std::isnan(high)) {
          lower[out + k] = -std::numeric_limits<double>::infinity();
          upper[out + k] = std::numeric_limits<double>::infinity();
        } else if (row.opcode == kSquare) {
          const bool crosses = low <= 0.0 && high >= 0.0;
          const double low_square = low * low;
          const double high_square = high * high;
          const double rounded_low = down(crosses ? 0.0 : minimum(low_square, high_square));
          lower[out + k] = (crosses || elementwise_arithmetic) ? maximum(rounded_low, 0.0) : rounded_low;
          upper[out + k] = up(maximum(low_square, high_square));
        } else if (row.opcode == kSqrt) {
          if (high < 0.0) {
            lower[out + k] = -std::numeric_limits<double>::infinity();
            upper[out + k] = std::numeric_limits<double>::infinity();
          } else {
            const double sqrt_low = std::sqrt(maximum(low, 0.0));
            const double sqrt_high = std::sqrt(high);
            lower[out + k] = down(sqrt_low);
            upper[out + k] = up(sqrt_high);
          }
        } else if (low <= 0.0 && high >= 0.0) {
          lower[out + k] = -std::numeric_limits<double>::infinity();
          upper[out + k] = std::numeric_limits<double>::infinity();
        } else {
          lower[out + k] = down(1.0 / high);
          upper[out + k] = up(1.0 / low);
        }
      }
      return;
    case kSine:
    case kCosine:
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t input = address(indices, row.a, row.a_index, k);
        const phx::mc::elementary::Enclosure result =
            row.opcode == kSine ? phx::mc::elementary::sin(lower[input], upper[input])
                                : phx::mc::elementary::cos(lower[input], upper[input]);
        lower[out + k] = result.lower;
        upper[out + k] = result.upper;
      }
      return;
    case kReduceSum: {
      const double gamma = parameters[row.b_index];
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t base = row.a + k * row.aux;
        Reduction sum;
        accumulate(lower, upper, base, base + row.aux, sum);
        store_reduction(sum, gamma, lower, upper, out + k);
      }
      return;
    }
    default:
      break;
  }
  for (std::int64_t k = begin; k < end; ++k) {
    const std::int64_t a = address(indices, row.a, row.a_index, k);
    const std::int64_t b = address(indices, row.b, row.b_index, k);
    const double a_low = lower[a];
    const double a_high = upper[a];
    const double b_low = lower[b];
    const double b_high = upper[b];
    double low = 0.0;
    double high = 0.0;
    if (row.opcode == kDivideElementwise || row.opcode == kPowerElementwise) {
      PowerInterval result;
      if (row.opcode == kDivideElementwise) {
        result = divide_interval(a_low, a_high, b_low, b_high);
      } else {
        const bool integer = b_low == b_high && integral_binary64(b_low);
        const bool invalid = !integer && a_low < 0.0;
        lower[out + row.size + k] = upper[out + row.size + k] = invalid ? -1.0 : 0.0;
        if (integer) {
          result = integer_power({a_low, a_high}, std::fabs(b_low));
          if (b_low < 0.0)
            result = divide_interval(1.0, 1.0, result.lower, result.upper);
        } else if (invalid) {
          result = {0.0, 0.0};
        } else {
          const auto logarithm = phx::mc::elementary::log(a_low, a_high);
          double product_low, product_high;
          phx::mc::interval::interval_product(b_low, b_high, logarithm.lower, logarithm.upper,
                                               product_low, product_high);
          const auto power = phx::mc::elementary::exp(product_low, product_high);
          result = {maximum(power.lower, 0.0), power.upper};
        }
      }
      lower[out + k] = result.lower;
      upper[out + k] = result.upper;
      continue;
    }
    if (elementwise_arithmetic &&
        (row.opcode == kAdd || row.opcode == kSubtract ||
         row.opcode == kSubtractSame || row.opcode == kMultiply)) {
      if (row.opcode == kMultiply) {
        phx::mc::interval::interval_product(a_low, a_high, b_low, b_high, low, high);
      } else {
        phx::mc::interval::interval_sum(a_low, a_high, b_low, b_high,
                                        row.opcode != kAdd, low, high);
        if (row.opcode == kSubtractSame && std::isfinite(a_low) && std::isfinite(a_high))
          low = high = 0.0;
      }
      lower[out + k] = low;
      upper[out + k] = high;
      continue;
    }
    switch (row.opcode) {
      case kAdd: {
        const double sum_lower = a_low + b_low;
        const double sum_upper = a_high + b_high;
        if (std::isnan(sum_lower) || std::isnan(sum_upper)) {
          low = -std::numeric_limits<double>::infinity();
          high = std::numeric_limits<double>::infinity();
        } else {
          // The host interval sum's sign clamp: nonnegative (nonpositive)
          // operands give a nonnegative (nonpositive) sum.
          low = a_low >= 0.0 && b_low >= 0.0 ? maximum(down(sum_lower), 0.0) : down(sum_lower);
          high = a_high <= 0.0 && b_high <= 0.0 ? minimum(up(sum_upper), 0.0) : up(sum_upper);
        }
        break;
      }
      case kSubtract:
      case kSubtractSame: {
        const double difference_lower = a_low - b_high;
        const double difference_upper = a_high - b_low;
        if (std::isnan(difference_lower) || std::isnan(difference_upper)) {
          low = -std::numeric_limits<double>::infinity();
          high = std::numeric_limits<double>::infinity();
        } else {
          low = down(difference_lower);
          high = up(difference_upper);
        }
        if (row.opcode == kSubtractSame && std::isfinite(a_low) && std::isfinite(a_high)) {
          low = high = 0.0;
        }
        break;
      }
      case kMaximum:
        low = maximum(a_low, b_low);
        high = maximum(a_high, b_high);
        break;
      case kMinimum:
        low = minimum(a_low, b_low);
        high = minimum(a_high, b_high);
        break;
      default: {  // kMultiply
        const double ll = product(a_low, b_low);
        const double lu = product(a_low, b_high);
        const double ul = product(a_high, b_low);
        const double uu = product(a_high, b_high);
        if (std::isnan(ll) || std::isnan(lu) || std::isnan(ul) || std::isnan(uu)) {
          low = -std::numeric_limits<double>::infinity();
          high = std::numeric_limits<double>::infinity();
        } else {
          low = down(minimum(minimum(ll, lu), minimum(ul, uu)));
          high = up(maximum(maximum(ll, lu), maximum(ul, uu)));
        }
        break;
      }
    }
    lower[out + k] = low;
    upper[out + k] = high;
  }
}
void execute_validated(const int64_t* instructions, int64_t instruction_count,
                       const int64_t* indices, const double* parameters,
                       double* lower, double* upper, bool elementwise_arithmetic = false) noexcept {
  for (int64_t row_index = 0; row_index < instruction_count; ++row_index) {
    const Row row = row_at(instructions, row_index);
    if (row.size == 0) continue;
    if (row.opcode == kReduceSum) {
      for (int64_t element = 0; element < row.size; ++element) {
        const int64_t base = row.a + element * row.aux;
        if (elementwise_arithmetic) {
          double low = 0.0, high = 0.0;
          int64_t first = 0;
          if (row.a_index == -1 && row.aux > 0) {
            low = lower[base];
            high = upper[base];
            first = 1;
          }
          for (int64_t term = first; term < row.aux; ++term)
            phx::mc::interval::interval_sum(low, high, lower[base + term],
                                            upper[base + term], false, low, high);
          lower[row.out + element] = low;
          upper[row.out + element] = high;
        } else {
          Reduction sum;
          accumulate(lower, upper, base, base + row.aux, sum);
          store_reduction(sum, parameters[row.b_index], lower, upper, row.out + element);
        }
      }
    } else {
      execute_row(row, 0, row.size, indices, parameters, lower, upper, elementwise_arithmetic);
    }
  }
}

bool event_holds(std::int64_t test, double low, double high) noexcept {
  switch (test) {
    case kMeetsZero:
      return !(low > 0.0 || high < 0.0);
    case kNotPositive:
      return !(low > 0.0);
    case kNotNonnegative:
      return !(low >= 0.0);
    case kUndecided:
      return low != high;
    case kNonfinite:
      return !(std::isfinite(low) && std::isfinite(high));
    default:  // kAlways
      return true;
  }
}

// Clear the flags of every box with a holding event over an executed workspace.
void record_events(const int64_t* events, int64_t event_count, const int64_t* indices,
                   const double* lower, const double* upper, int64_t box_count,
                   std::uint8_t* smooth, std::uint8_t* continuous) noexcept {
  for (int64_t box = 0; box < box_count; ++box) smooth[box] = continuous[box] = 1;
  for (int64_t row = 0; row < event_count; ++row) {
    const int64_t* fields = events + row * kEventFields;
    const bool jump = fields[0] == kJump;
    for (int64_t k = 0; k < fields[2]; ++k) {
      const int64_t entry = address(indices, fields[3], fields[4], k);
      if (!event_holds(fields[1], lower[entry], upper[entry])) continue;
      const int64_t box = fields[5] == -1 ? -1 : indices[fields[5] + k];
      const int64_t first = box == -1 ? 0 : box;
      const int64_t last = box == -1 ? box_count : box + 1;
      for (int64_t target = first; target < last; ++target) {
        smooth[target] = 0;
        if (jump) continuous[target] = 0;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Point executors of the same validated programs. They evaluate a batch of
// boxes at points, in binary64 with ordinary rounding: proposals, never a
// certificate. Layout is entry-major so that every row runs as a contiguous
// loop over the batch: entry e of box j is values[e * count + j] (float), and
// component c of entry e of box j is values[(e * components + c) * count + j]
// (jets). Min/max winner flags have no point meaning; their rows write zeros,
// and a decision row passes its comparison through, which at a point is
// already decided.

inline double float_power(double base, double exponent, double& invalid) noexcept {
  const bool integer = integral_binary64(exponent);
  invalid = !integer && base < 0.0 ? -1.0 : 0.0;
  return std::pow(base, exponent);
}

void execute_float(const int64_t* instructions, int64_t instruction_count,
                   const int64_t* indices, double* values, int64_t count) noexcept {
  const auto column = [&](int64_t entry) { return values + entry * count; };
  for (int64_t row_index = 0; row_index < instruction_count; ++row_index) {
    const Row row = row_at(instructions, row_index);
    for (int64_t k = 0; k < row.size; ++k) {
      double* out = column(row.out + k);
      switch (row.opcode) {
        case kReduceSum: {
          const int64_t base = row.a + k * row.aux;
          const double* first = column(base);
          for (int64_t j = 0; j < count; ++j) out[j] = row.aux > 0 ? first[j] : 0.0;
          for (int64_t term = 1; term < row.aux; ++term) {
            const double* next = column(base + term);
            for (int64_t j = 0; j < count; ++j) out[j] += next[j];
          }
          continue;
        }
        case kExtremumPair:
          for (int64_t j = 0; j < count; ++j) out[j] = 0.0;
          for (int64_t j = 0; j < count; ++j) column(row.out + row.size + k)[j] = 0.0;
          continue;
        case kExtremumReduce:
          for (int64_t lane = 0; lane < row.b; ++lane) {
            double* flags = column(row.out + k * row.b + lane);
            for (int64_t j = 0; j < count; ++j) flags[j] = 0.0;
          }
          continue;
        default:
          break;
      }
      const double* a = column(address(indices, row.a, row.a_index, k));
      switch (row.opcode) {
        case kGather:
        case kDecide:
          for (int64_t j = 0; j < count; ++j) out[j] = a[j];
          continue;
        case kNegate:
          for (int64_t j = 0; j < count; ++j) out[j] = -a[j];
          continue;
        case kAbsolute:
          for (int64_t j = 0; j < count; ++j) out[j] = std::fabs(a[j]);
          continue;
        case kSquare:
          for (int64_t j = 0; j < count; ++j) out[j] = a[j] * a[j];
          continue;
        case kSqrt:
          for (int64_t j = 0; j < count; ++j) out[j] = std::sqrt(a[j]);
          continue;
        case kReciprocal:
          for (int64_t j = 0; j < count; ++j) out[j] = 1.0 / a[j];
          continue;
        case kSine:
          for (int64_t j = 0; j < count; ++j) out[j] = std::sin(a[j]);
          continue;
        case kCosine:
          for (int64_t j = 0; j < count; ++j) out[j] = std::cos(a[j]);
          continue;
        case kExp:
          for (int64_t j = 0; j < count; ++j) out[j] = std::exp(a[j]);
          continue;
        case kLog:
          for (int64_t j = 0; j < count; ++j) out[j] = std::log(a[j]);
          continue;
        case kAtan:
          for (int64_t j = 0; j < count; ++j) out[j] = std::atan(a[j]);
          continue;
        case kTan:
          for (int64_t j = 0; j < count; ++j) out[j] = std::tan(a[j]);
          continue;
        case kAttain:
          for (int64_t j = 0; j < count; ++j) out[j] = maximum(a[j], 1.0);
          continue;
        case kSelect: {
          const double* first = column(row.b + k);
          const double* second = column(row.b + row.aux + k);
          for (int64_t j = 0; j < count; ++j) out[j] = a[j] > 0.0 ? second[j] : first[j];
          continue;
        }
        default:
          break;
      }
      const double* b = column(address(indices, row.b, row.b_index, k));
      switch (row.opcode) {
        case kAdd:
          for (int64_t j = 0; j < count; ++j) out[j] = a[j] + b[j];
          break;
        case kSubtract:
          for (int64_t j = 0; j < count; ++j) out[j] = a[j] - b[j];
          break;
        case kSubtractSame:
          for (int64_t j = 0; j < count; ++j) out[j] = std::isfinite(a[j]) ? 0.0 : a[j] - b[j];
          break;
        case kMultiply:
          for (int64_t j = 0; j < count; ++j) out[j] = a[j] * b[j];
          break;
        case kDivideElementwise:
          for (int64_t j = 0; j < count; ++j) out[j] = a[j] / b[j];
          break;
        case kMaximum:
          for (int64_t j = 0; j < count; ++j) out[j] = maximum(a[j], b[j]);
          break;
        case kMinimum:
          for (int64_t j = 0; j < count; ++j) out[j] = minimum(a[j], b[j]);
          break;
        case kCompare:
          for (int64_t j = 0; j < count; ++j)
            out[j] = (row.aux == 0 ? a[j] > b[j] : a[j] >= b[j]) ? 1.0 : 0.0;
          break;
        case kEqual:
          for (int64_t j = 0; j < count; ++j)
            out[j] = ((a[j] == b[j]) == (row.aux == 0)) ? 1.0 : 0.0;
          break;
        default: {  // kPowerElementwise
          double* invalid = column(row.out + row.size + k);
          for (int64_t j = 0; j < count; ++j) out[j] = float_power(a[j], b[j], invalid[j]);
          break;
        }
      }
    }
  }
}

// Second-order forward jets in up to three variables: value, gradient (n) and
// the upper triangle of the Hessian (n (n + 1) / 2), stored per component.
class JetLayout {
 public:
  explicit JetLayout(int64_t variables) noexcept
      : variables_(variables), components_(1 + variables + variables * (variables + 1) / 2) {}
  int64_t variables() const noexcept { return variables_; }
  int64_t components() const noexcept { return components_; }
  int64_t gradient(int64_t i) const noexcept { return 1 + i; }
  int64_t hessian(int64_t i, int64_t l) const noexcept {
    const int64_t low = i < l ? i : l;
    const int64_t high = i < l ? l : i;
    return 1 + variables_ + low * variables_ - low * (low - 1) / 2 + (high - low);
  }

 private:
  int64_t variables_;
  int64_t components_;
};

struct JetView {
  double* base;
  int64_t count;
  double* component(int64_t c) const noexcept { return base + c * count; }
};

// Per-thread derivative columns of one unary row, reused across rows.
struct JetScratch {
  std::vector<double> value, first, second;
  void resize(int64_t count) {
    const auto size = static_cast<std::size_t>(count);
    value.resize(size);
    first.resize(size);
    second.resize(size);
  }
};

JetScratch& jet_scratch() {
  thread_local JetScratch scratch;
  return scratch;
}

// f(a) with f' and f'' per point: g = f' g_a, H = f' H_a + f'' g_a g_a^T. Every
// component is one contiguous loop over the batch.
template <class Derivatives>
void jet_unary(const JetLayout& layout, const JetView& a, const JetView& out,
               Derivatives derivatives) {
  const int64_t n = layout.variables();
  const int64_t count = a.count;
  JetScratch& s = jet_scratch();
  s.resize(count);
  double* value = s.value.data();
  double* first = s.first.data();
  double* second = s.second.data();
  const double* a0 = a.component(0);
  for (int64_t j = 0; j < count; ++j) derivatives(a0[j], value[j], first[j], second[j]);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t l = i; l < n; ++l) {
      const int64_t h = layout.hessian(i, l);
      const double* ah = a.component(h);
      const double* gi = a.component(layout.gradient(i));
      const double* gl = a.component(layout.gradient(l));
      double* oh = out.component(h);
      for (int64_t j = 0; j < count; ++j) oh[j] = first[j] * ah[j] + second[j] * gi[j] * gl[j];
    }
  for (int64_t i = 0; i < n; ++i) {
    const double* gi = a.component(layout.gradient(i));
    double* og = out.component(layout.gradient(i));
    for (int64_t j = 0; j < count; ++j) og[j] = first[j] * gi[j];
  }
  double* o0 = out.component(0);
  for (int64_t j = 0; j < count; ++j) o0[j] = value[j];
}

// a * b: g = a g_b + b g_a, H = a H_b + b H_a + g_a g_b^T + g_b g_a^T.
void jet_multiply(const JetLayout& layout, const JetView& a, const JetView& b,
                  const JetView& out) noexcept {
  const int64_t n = layout.variables();
  const int64_t count = a.count;
  const double* a0 = a.component(0);
  const double* b0 = b.component(0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t l = i; l < n; ++l) {
      const int64_t h = layout.hessian(i, l);
      const double* ah = a.component(h);
      const double* bh = b.component(h);
      const double* agi = a.component(layout.gradient(i));
      const double* agl = a.component(layout.gradient(l));
      const double* bgi = b.component(layout.gradient(i));
      const double* bgl = b.component(layout.gradient(l));
      double* oh = out.component(h);
      for (int64_t j = 0; j < count; ++j)
        oh[j] = a0[j] * bh[j] + b0[j] * ah[j] + agi[j] * bgl[j] + bgi[j] * agl[j];
    }
  for (int64_t i = 0; i < n; ++i) {
    const double* agi = a.component(layout.gradient(i));
    const double* bgi = b.component(layout.gradient(i));
    double* og = out.component(layout.gradient(i));
    for (int64_t j = 0; j < count; ++j) og[j] = a0[j] * bgi[j] + b0[j] * agi[j];
  }
  double* o0 = out.component(0);
  for (int64_t j = 0; j < count; ++j) o0[j] = a0[j] * b0[j];
}

// a / b from f b = a: g_f = (g_a - f g_b) / b, H_f = (H_a - f H_b - g_f g_b^T - g_b g_f^T) / b.
void jet_divide(const JetLayout& layout, const JetView& a, const JetView& b,
                const JetView& out) noexcept {
  const int64_t n = layout.variables();
  const int64_t count = a.count;
  const double* a0 = a.component(0);
  const double* b0 = b.component(0);
  double* f = out.component(0);
  for (int64_t j = 0; j < count; ++j) f[j] = a0[j] / b0[j];
  for (int64_t i = 0; i < n; ++i) {
    const double* agi = a.component(layout.gradient(i));
    const double* bgi = b.component(layout.gradient(i));
    double* og = out.component(layout.gradient(i));
    for (int64_t j = 0; j < count; ++j) og[j] = (agi[j] - f[j] * bgi[j]) / b0[j];
  }
  for (int64_t i = 0; i < n; ++i)
    for (int64_t l = i; l < n; ++l) {
      const int64_t h = layout.hessian(i, l);
      const double* ah = a.component(h);
      const double* bh = b.component(h);
      const double* fgi = out.component(layout.gradient(i));
      const double* fgl = out.component(layout.gradient(l));
      const double* bgi = b.component(layout.gradient(i));
      const double* bgl = b.component(layout.gradient(l));
      double* oh = out.component(h);
      for (int64_t j = 0; j < count; ++j)
        oh[j] = (ah[j] - f[j] * bh[j] - fgi[j] * bgl[j] - bgi[j] * fgl[j]) / b0[j];
    }
}

// Per box, out takes every component of the first or the second operand.
template <class Choose>
void jet_choose(const JetLayout& layout, const JetView& first, const JetView& second,
                const JetView& out, Choose choose_second) noexcept {
  for (int64_t j = 0; j < first.count; ++j) {
    const JetView& source = choose_second(j) ? second : first;
    for (int64_t c = 0; c < layout.components(); ++c)
      out.component(c)[j] = source.component(c)[j];
  }
}

void jet_linear(const JetLayout& layout, const JetView& a, const JetView& b, double sign,
                const JetView& out) noexcept {
  for (int64_t c = 0; c < layout.components(); ++c) {
    const double* x = a.component(c);
    const double* y = b.component(c);
    double* z = out.component(c);
    for (int64_t j = 0; j < a.count; ++j) z[j] = x[j] + sign * y[j];
  }
}

void jet_constant(const JetLayout& layout, const JetView& out, const double* values) noexcept {
  for (int64_t j = 0; j < out.count; ++j) out.component(0)[j] = values[j];
  for (int64_t c = 1; c < layout.components(); ++c)
    for (int64_t j = 0; j < out.count; ++j) out.component(c)[j] = 0.0;
}

void execute_jet(const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
                 const JetLayout& layout, double* values, int64_t count,
                 std::vector<double>& scratch) {
  const int64_t width = layout.components() * count;
  const auto entry = [&](int64_t e) { return JetView{values + e * width, count}; };
  for (int64_t row_index = 0; row_index < instruction_count; ++row_index) {
    const Row row = row_at(instructions, row_index);
    for (int64_t k = 0; k < row.size; ++k) {
      const JetView out = entry(row.out + k);
      switch (row.opcode) {
        case kReduceSum: {
          const int64_t base = row.a + k * row.aux;
          for (int64_t c = 0; c < layout.components(); ++c) {
            double* z = out.component(c);
            for (int64_t j = 0; j < count; ++j)
              z[j] = row.aux > 0 ? entry(base).component(c)[j] : 0.0;
            for (int64_t term = 1; term < row.aux; ++term) {
              const double* x = entry(base + term).component(c);
              for (int64_t j = 0; j < count; ++j) z[j] += x[j];
            }
          }
          continue;
        }
        case kExtremumPair:
          scratch.assign(static_cast<std::size_t>(count), 0.0);
          jet_constant(layout, out, scratch.data());
          jet_constant(layout, entry(row.out + row.size + k), scratch.data());
          continue;
        case kExtremumReduce:
          scratch.assign(static_cast<std::size_t>(count), 0.0);
          for (int64_t lane = 0; lane < row.b; ++lane)
            jet_constant(layout, entry(row.out + k * row.b + lane), scratch.data());
          continue;
        default:
          break;
      }
      const JetView a = entry(address(indices, row.a, row.a_index, k));
      const double* av = a.component(0);
      switch (row.opcode) {
        case kGather:
        case kDecide:
          jet_choose(layout, a, a, out, [](int64_t) { return false; });
          continue;
        case kNegate:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = -x, d1 = -1.0, d2 = 0.0;
          });
          continue;
        case kAbsolute:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = std::fabs(x), d1 = std::signbit(x) ? -1.0 : 1.0, d2 = 0.0;
          });
          continue;
        case kSquare:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = x * x, d1 = 2.0 * x, d2 = 2.0;
          });
          continue;
        case kSqrt:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = std::sqrt(x), d1 = 0.5 / v, d2 = -0.25 / (v * x);
          });
          continue;
        case kReciprocal:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = 1.0 / x, d1 = -v * v, d2 = 2.0 * v * v * v;
          });
          continue;
        case kSine:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = std::sin(x), d1 = std::cos(x), d2 = -v;
          });
          continue;
        case kCosine:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = std::cos(x), d1 = -std::sin(x), d2 = -v;
          });
          continue;
        case kExp:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = std::exp(x), d1 = v, d2 = v;
          });
          continue;
        case kLog:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = std::log(x), d1 = 1.0 / x, d2 = -d1 * d1;
          });
          continue;
        case kAtan:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            const double denominator = 1.0 + x * x;
            v = std::atan(x), d1 = 1.0 / denominator, d2 = -2.0 * x * d1 * d1;
          });
          continue;
        case kTan:
          jet_unary(layout, a, out, [](double x, double& v, double& d1, double& d2) {
            v = std::tan(x), d1 = 1.0 + v * v, d2 = 2.0 * v * d1;
          });
          continue;
        case kAttain:
          scratch.resize(static_cast<std::size_t>(count));
          for (int64_t j = 0; j < count; ++j) scratch[static_cast<std::size_t>(j)] = maximum(av[j], 1.0);
          jet_constant(layout, out, scratch.data());
          continue;
        case kSelect: {
          const JetView first = entry(row.b + k);
          const JetView second = entry(row.b + row.aux + k);
          jet_choose(layout, first, second, out, [&](int64_t j) { return av[j] > 0.0; });
          continue;
        }
        default:
          break;
      }
      const JetView b = entry(address(indices, row.b, row.b_index, k));
      const double* bv = b.component(0);
      switch (row.opcode) {
        case kAdd:
          jet_linear(layout, a, b, 1.0, out);
          break;
        case kSubtract:
          jet_linear(layout, a, b, -1.0, out);
          break;
        case kSubtractSame:
          scratch.resize(static_cast<std::size_t>(count));
          for (int64_t j = 0; j < count; ++j)
            scratch[static_cast<std::size_t>(j)] = std::isfinite(av[j]) ? 0.0 : av[j] - bv[j];
          jet_constant(layout, out, scratch.data());
          break;
        case kMultiply:
          jet_multiply(layout, a, b, out);
          break;
        case kDivideElementwise:
          jet_divide(layout, a, b, out);
          break;
        case kMaximum:
          jet_choose(layout, a, b, out, [&](int64_t j) { return bv[j] > av[j]; });
          break;
        case kMinimum:
          jet_choose(layout, a, b, out, [&](int64_t j) { return bv[j] < av[j]; });
          break;
        case kCompare:
        case kEqual:
          scratch.resize(static_cast<std::size_t>(count));
          for (int64_t j = 0; j < count; ++j) {
            const bool truth = row.opcode == kCompare
                                   ? (row.aux == 0 ? av[j] > bv[j] : av[j] >= bv[j])
                                   : ((av[j] == bv[j]) == (row.aux == 0));
            scratch[static_cast<std::size_t>(j)] = truth ? 1.0 : 0.0;
          }
          jet_constant(layout, out, scratch.data());
          break;
        default: {  // kPowerElementwise: a^p for a constant integral p, else exp(b log a)
          const JetView invalid = entry(row.out + row.size + k);
          scratch.resize(static_cast<std::size_t>(count));
          bool constant_exponent = true;
          for (int64_t i = 0; i < layout.variables() && constant_exponent; ++i)
            for (int64_t j = 0; j < count; ++j)
              if (b.component(layout.gradient(i))[j] != 0.0) {
                constant_exponent = false;
                break;
              }
          for (int64_t j = 0; j < count; ++j) {
            double flag = 0.0;
            float_power(av[j], bv[j], flag);
            scratch[static_cast<std::size_t>(j)] = flag;
          }
          jet_constant(layout, invalid, scratch.data());
          if (constant_exponent) {
            // Exponent constant per box: f = a^p, f' = p a^(p-1), f'' = p (p-1) a^(p-2).
            const int64_t n = layout.variables();
            for (int64_t j = 0; j < count; ++j) {
              const double x = av[j];
              const double p = bv[j];
              const double value = std::pow(x, p);
              const double first = p == 0.0 ? 0.0 : p * std::pow(x, p - 1.0);
              const double second = (p == 0.0 || p == 1.0) ? 0.0 : p * (p - 1.0) * std::pow(x, p - 2.0);
              double gradient[3] = {0.0, 0.0, 0.0};
              for (int64_t i = 0; i < n; ++i) gradient[i] = a.component(layout.gradient(i))[j];
              for (int64_t i = 0; i < n; ++i)
                for (int64_t l = i; l < n; ++l) {
                  const int64_t h = layout.hessian(i, l);
                  out.component(h)[j] = first * a.component(h)[j] + second * gradient[i] * gradient[l];
                }
              for (int64_t i = 0; i < n; ++i) out.component(layout.gradient(i))[j] = first * gradient[i];
              out.component(0)[j] = value;
            }
          } else {
            // f = exp(b log a): the jets of log a, b log a, and exp of that.
            std::vector<double> log_jet(static_cast<std::size_t>(width));
            std::vector<double> product_jet(static_cast<std::size_t>(width));
            const JetView logarithm{log_jet.data(), count};
            const JetView product_view{product_jet.data(), count};
            jet_unary(layout, a, logarithm, [](double x, double& v, double& d1, double& d2) {
              v = std::log(x), d1 = 1.0 / x, d2 = -d1 * d1;
            });
            jet_multiply(layout, b, logarithm, product_view);
            jet_unary(layout, product_view, out, [](double x, double& v, double& d1, double& d2) {
              v = std::exp(x), d1 = v, d2 = v;
            });
          }
          break;
        }
      }
    }
  }
}

}  // namespace

int32_t phx::mc::PreparedIntervalProgram::bind(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count,
    int64_t workspace_size) {
  workspace_size_ = -1;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!phx::mc::addressable(instruction_count, kFields, sizeof(int64_t)) ||
        !phx::mc::addressable(index_count, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(parameter_count, 1, sizeof(double)) ||
        !phx::mc::addressable(workspace_size, 1, sizeof(double)) ||
        (instruction_count > 0 && instructions == nullptr) ||
        (index_count > 0 && indices == nullptr) ||
        (parameter_count > 0 && parameters == nullptr))
      return PHX_MC_INVALID_ARGUMENT;
    std::vector<int64_t> owned_instructions, owned_indices;
    std::vector<double> owned_parameters;
    if (instruction_count)
      owned_instructions.assign(instructions, instructions + instruction_count * kFields);
    if (index_count) owned_indices.assign(indices, indices + index_count);
    if (parameter_count) owned_parameters.assign(parameters, parameters + parameter_count);
    if (!program_valid(owned_instructions.data(), instruction_count, owned_indices.data(),
                       index_count, owned_parameters.data(), parameter_count, workspace_size))
      return PHX_MC_INVALID_INPUT;
    instructions_ = std::move(owned_instructions);
    indices_ = std::move(owned_indices);
    parameters_ = std::move(owned_parameters);
    workspace_size_ = workspace_size;
    return PHX_MC_OK;
  });
}

int32_t phx::mc::PreparedIntervalProgram::execute(
    double* lower, double* upper, int64_t workspace_size) const {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (workspace_size_ < 0 || workspace_size != workspace_size_ ||
        (workspace_size > 0 && (lower == nullptr || upper == nullptr)))
      return PHX_MC_INVALID_ARGUMENT;
    execute_validated(instructions_.data(), static_cast<int64_t>(instructions_.size() / kFields),
                      indices_.data(), parameters_.data(), lower, upper);
    return PHX_MC_OK;
  });
}

extern "C" {

int32_t phx_mc_interval_program_execute(const int64_t* instructions, int64_t instruction_count,
                                        const int64_t* indices, int64_t index_count,
                                        const double* parameters, int64_t parameter_count,
                                        double* lower, double* upper, int64_t workspace_size) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!phx::mc::addressable(instruction_count, kFields, sizeof(int64_t)) ||
        !phx::mc::addressable(index_count, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(parameter_count, 1, sizeof(double)) ||
        !phx::mc::addressable(workspace_size, 1, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if ((instruction_count > 0 && instructions == nullptr) ||
        (index_count > 0 && indices == nullptr) ||
        (parameter_count > 0 && parameters == nullptr) ||
        (workspace_size > 0 && (lower == nullptr || upper == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!program_valid(instructions, instruction_count, indices, index_count, parameters,
                       parameter_count, workspace_size)) {
      return PHX_MC_INVALID_INPUT;
    }
    execute_validated(instructions, instruction_count, indices, parameters, lower, upper);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_interval_program_batch(const int64_t* instructions, int64_t instruction_count,
                                      const int64_t* indices, int64_t index_count,
                                      const double* parameters, int64_t parameter_count,
                                      double* lower, double* upper, int64_t workspace_size,
                                      int64_t count, int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    if (parallelism < 1 ||
        !phx::mc::addressable(instruction_count, kFields, sizeof(int64_t)) ||
        !phx::mc::addressable(index_count, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(parameter_count, 1, sizeof(double)) ||
        !phx::mc::addressable(count, workspace_size, sizeof(double)) ||
        (instruction_count > 0 && instructions == nullptr) ||
        (index_count > 0 && indices == nullptr) ||
        (parameter_count > 0 && parameters == nullptr) ||
        (count > 0 && workspace_size > 0 && (lower == nullptr || upper == nullptr)))
      return PHX_MC_INVALID_ARGUMENT;
    if (!program_valid(instructions, instruction_count, indices, index_count, parameters,
                       parameter_count, workspace_size))
      return PHX_MC_INVALID_INPUT;
    constexpr std::uint64_t supported =
        (std::uint64_t{1} << kGather) | (std::uint64_t{1} << kAdd) |
        (std::uint64_t{1} << kSubtract) | (std::uint64_t{1} << kSubtractSame) |
        (std::uint64_t{1} << kMultiply) |
        (std::uint64_t{1} << kNegate) | (std::uint64_t{1} << kReduceSum) |
        (std::uint64_t{1} << kSine) | (std::uint64_t{1} << kCosine) |
        (std::uint64_t{1} << kSqrt) | (std::uint64_t{1} << kSquare) |
        (std::uint64_t{1} << kAbsolute) | (std::uint64_t{1} << kCompare) |
        (std::uint64_t{1} << kEqual) | (std::uint64_t{1} << kSelect) |
        (std::uint64_t{1} << kMaximum) | (std::uint64_t{1} << kMinimum) |
        (std::uint64_t{1} << kExtremumPair) | (std::uint64_t{1} << kExtremumReduce) |
        (std::uint64_t{1} << kDecide) | (std::uint64_t{1} << kAttain) |
        (std::uint64_t{1} << kExp) | (std::uint64_t{1} << kLog) |
        (std::uint64_t{1} << kDivideElementwise) | (std::uint64_t{1} << kAtan) |
        (std::uint64_t{1} << kTan) |
        (std::uint64_t{1} << kPowerElementwise);
    for (int64_t row = 0; row < instruction_count; ++row)
      if (!(supported & (std::uint64_t{1} << row_at(instructions, row).opcode)))
        return PHX_MC_INVALID_INPUT;
    if (count == 0 || workspace_size == 0) return PHX_MC_OK;
    return phx::mc::parallel_for(
        static_cast<std::size_t>(count), {64, static_cast<uint32_t>(parallelism)},
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          const phx::mc::GradualNearestScope mode;
          if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
          const auto width = static_cast<std::size_t>(workspace_size);
          for (std::size_t box = begin; box < end; ++box)
            execute_validated(instructions, instruction_count, indices, parameters,
                              lower + box * width, upper + box * width, true);
          return PHX_MC_OK;
        });
  });
}

int32_t phx_mc_interval_program_branches(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count, double* lower,
    double* upper, int64_t workspace_size, const int64_t* events, int64_t event_count,
    int64_t box_count, uint8_t* smooth, uint8_t* continuous) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (!phx::mc::addressable(instruction_count, kFields, sizeof(int64_t)) ||
        !phx::mc::addressable(index_count, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(parameter_count, 1, sizeof(double)) ||
        !phx::mc::addressable(workspace_size, 1, sizeof(double)) ||
        !phx::mc::addressable(event_count, kEventFields, sizeof(int64_t)) ||
        !phx::mc::addressable(box_count, 1, sizeof(uint8_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if ((instruction_count > 0 && instructions == nullptr) ||
        (index_count > 0 && indices == nullptr) ||
        (parameter_count > 0 && parameters == nullptr) ||
        (workspace_size > 0 && (lower == nullptr || upper == nullptr)) ||
        (event_count > 0 && events == nullptr) ||
        (box_count > 0 && (smooth == nullptr || continuous == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!program_valid(instructions, instruction_count, indices, index_count, parameters,
                       parameter_count, workspace_size) ||
        !events_valid(events, event_count, indices, index_count, workspace_size, box_count)) {
      return PHX_MC_INVALID_INPUT;
    }
    execute_validated(instructions, instruction_count, indices, parameters, lower, upper);
    record_events(events, event_count, indices, lower, upper, box_count, smooth, continuous);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_interval_program_capabilities(uint64_t* opcodes) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    if (opcodes == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *opcodes = kIntervalOpcodeCapabilities;
    return PHX_MC_OK;
  });
}

}  // extern "C"

namespace {

// Points per block of a point execution: the block's whole workspace (entries
// times jet components times points) stays in cache.
constexpr int64_t kPointBlock = 256;

struct PointProgram {
  const int64_t* instructions;
  int64_t instruction_count;
  const int64_t* indices;
  const double* seeds;  // workspace_size template: seed values at their entries
  int64_t workspace_size;
  int64_t input_dimension;
  const int64_t* outputs;
  int64_t output_count;
};

int32_t point_arguments_valid(const int64_t* instructions, int64_t instruction_count,
                              const int64_t* indices, int64_t index_count,
                              const double* parameters, int64_t parameter_count,
                              const double* seeds, int64_t workspace_size,
                              int64_t input_dimension, const double* points, int64_t count,
                              const int64_t* outputs, int64_t output_count,
                              const double* result, int64_t result_width) {
  if (!phx::mc::addressable(instruction_count, kFields, sizeof(int64_t)) ||
      !phx::mc::addressable(index_count, 1, sizeof(int64_t)) ||
      !phx::mc::addressable(parameter_count, 1, sizeof(double)) ||
      !phx::mc::addressable(workspace_size, 1, sizeof(double)) ||
      !phx::mc::addressable(count, input_dimension, sizeof(double)) ||
      !phx::mc::addressable(count, result_width, sizeof(double)) || input_dimension < 1 ||
      input_dimension > workspace_size || output_count < 0 ||
      (instruction_count > 0 && instructions == nullptr) ||
      (index_count > 0 && indices == nullptr) ||
      (parameter_count > 0 && parameters == nullptr) ||
      (workspace_size > 0 && seeds == nullptr) || (count > 0 && points == nullptr) ||
      (output_count > 0 && outputs == nullptr) ||
      (count > 0 && result_width > 0 && result == nullptr))
    return PHX_MC_INVALID_ARGUMENT;
  for (int64_t k = 0; k < output_count; ++k)
    if (outputs[k] < 0 || outputs[k] >= workspace_size) return PHX_MC_INVALID_INPUT;
  if (!program_valid(instructions, instruction_count, indices, index_count, parameters,
                     parameter_count, workspace_size))
    return PHX_MC_INVALID_INPUT;
  return PHX_MC_OK;
}

void evaluate_point_blocks(const PointProgram& program, const double* points, int64_t count,
                           double* result) {
  std::vector<double> values(static_cast<std::size_t>(program.workspace_size * kPointBlock));
  for (int64_t start = 0; start < count; start += kPointBlock) {
    const int64_t block = std::min(kPointBlock, count - start);
    for (int64_t entry = 0; entry < program.workspace_size; ++entry)
      std::fill_n(values.data() + entry * block, block, program.seeds[entry]);
    for (int64_t d = 0; d < program.input_dimension; ++d)
      for (int64_t j = 0; j < block; ++j)
        values[static_cast<std::size_t>(d * block + j)] =
            points[(start + j) * program.input_dimension + d];
    execute_float(program.instructions, program.instruction_count, program.indices,
                  values.data(), block);
    for (int64_t j = 0; j < block; ++j)
      for (int64_t k = 0; k < program.output_count; ++k)
        result[(start + j) * program.output_count + k] =
            values[static_cast<std::size_t>(program.outputs[k] * block + j)];
  }
}

void jet_point_blocks(const PointProgram& program, const JetLayout& layout, const double* points,
                      int64_t count, double* result) {
  const int64_t components = layout.components();
  std::vector<double> values(
      static_cast<std::size_t>(program.workspace_size * components * kPointBlock));
  std::vector<double> scratch;
  for (int64_t start = 0; start < count; start += kPointBlock) {
    const int64_t block = std::min(kPointBlock, count - start);
    const int64_t width = components * block;
    for (int64_t entry = 0; entry < program.workspace_size; ++entry) {
      double* base = values.data() + entry * width;
      std::fill_n(base, block, program.seeds[entry]);
      std::fill_n(base + block, width - block, 0.0);
    }
    for (int64_t d = 0; d < program.input_dimension; ++d) {
      double* base = values.data() + d * width;
      for (int64_t j = 0; j < block; ++j) base[j] = points[(start + j) * program.input_dimension + d];
      std::fill_n(base + layout.gradient(d) * block, block, 1.0);
    }
    execute_jet(program.instructions, program.instruction_count, program.indices, layout,
                values.data(), block, scratch);
    for (int64_t j = 0; j < block; ++j)
      for (int64_t k = 0; k < program.output_count; ++k)
        for (int64_t c = 0; c < components; ++c)
          result[((start + j) * program.output_count + k) * components + c] =
              values[static_cast<std::size_t>(program.outputs[k] * width + c * block + j)];
  }
}

}  // namespace

extern "C" {

int32_t phx_mc_program_evaluate(const int64_t* instructions, int64_t instruction_count,
                                const int64_t* indices, int64_t index_count,
                                const double* parameters, int64_t parameter_count,
                                const double* seeds, int64_t workspace_size,
                                int64_t input_dimension, const double* points, int64_t count,
                                const int64_t* outputs, int64_t output_count, double* result) {
  return phx::mc::guarded([&]() -> int32_t {
    const int32_t status = point_arguments_valid(
        instructions, instruction_count, indices, index_count, parameters, parameter_count,
        seeds, workspace_size, input_dimension, points, count, outputs, output_count, result,
        output_count);
    if (status != PHX_MC_OK) return status;
    const PointProgram program{instructions, instruction_count, indices,  seeds,
                               workspace_size, input_dimension,  outputs,  output_count};
    evaluate_point_blocks(program, points, count, result);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_program_jet(const int64_t* instructions, int64_t instruction_count,
                           const int64_t* indices, int64_t index_count, const double* parameters,
                           int64_t parameter_count, const double* seeds, int64_t workspace_size,
                           int64_t input_dimension, const double* points, int64_t count,
                           const int64_t* outputs, int64_t output_count, double* result) {
  return phx::mc::guarded([&]() -> int32_t {
    if (input_dimension < 1 || input_dimension > 3) return PHX_MC_INVALID_ARGUMENT;
    const JetLayout layout(input_dimension);
    if (!phx::mc::addressable(output_count, layout.components(), sizeof(double)))
      return PHX_MC_INVALID_ARGUMENT;
    const int32_t status = point_arguments_valid(
        instructions, instruction_count, indices, index_count, parameters, parameter_count,
        seeds, workspace_size, input_dimension, points, count, outputs, output_count, result,
        output_count * layout.components());
    if (status != PHX_MC_OK) return status;
    const PointProgram program{instructions, instruction_count, indices,  seeds,
                               workspace_size, input_dimension,  outputs,  output_count};
    jet_point_blocks(program, layout, points, count, result);
    return PHX_MC_OK;
  });
}

}  // extern "C"
