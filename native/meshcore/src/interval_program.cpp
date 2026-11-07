// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Straight-line outward interval execution. A whole program is validated before
// the first workspace write; validated rows execute in source order.
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "capi_guard.hpp"
#include "elementary.hpp"
#include "floating_point_mode.hpp"
#include "interval_program.hpp"
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
    (std::uint64_t{1} << kDecide) | (std::uint64_t{1} << kAttain);

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
  if (row.opcode == kExtremumPair) return 2 * row.size;
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
  if (row.opcode < kGather || row.opcode > kAttain) return false;
  if (row.size < 0 || row.out < previous_end || row.out > workspace_size) return false;
  const std::int64_t room = workspace_size - row.out;
  switch (row.opcode) {
    case kExtremumPair:
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
      if (row.a_index != -1 || row.b != -1 || row.aux < 0) return false;
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

// Execute elements [begin, end) of a validated row.
void execute_row(const Row& row, std::int64_t begin, std::int64_t end,
                 const std::int64_t* indices, const double* parameters, double* lower,
                 double* upper) noexcept {
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
        lower[out + k] = -high;
        upper[out + k] = -low;
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
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t input = address(indices, row.a, row.a_index, k);
        const phx::mc::elementary::Enclosure result =
            row.opcode == kExp ? phx::mc::elementary::exp(lower[input], upper[input])
                               : phx::mc::elementary::log(lower[input], upper[input]);
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
          lower[out + k] = crosses ? maximum(rounded_low, 0.0) : rounded_low;
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
                       double* lower, double* upper) noexcept {
  for (int64_t row_index = 0; row_index < instruction_count; ++row_index) {
    const Row row = row_at(instructions, row_index);
    if (row.size == 0) continue;
    if (row.opcode == kReduceSum) {
      for (int64_t element = 0; element < row.size; ++element) {
        const int64_t base = row.a + element * row.aux;
        Reduction sum;
        accumulate(lower, upper, base, base + row.aux, sum);
        store_reduction(sum, parameters[row.b_index], lower, upper, row.out + element);
      }
    } else {
      execute_row(row, 0, row.size, indices, parameters, lower, upper);
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
