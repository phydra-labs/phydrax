//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: straight-line interval programs lowered from the canonical host
// interval owner (phydrax.geometry._interval_native). Each instruction applies
// one owner formula element by element over caller workspaces; the arithmetic
// is that of the owner's NumPy formulas: round-to-nearest binary64 with one
// nextafter step per outward rounding, NumPy minimum/maximum (first NaN
// operand, -0 below +0), +0 for zero times infinity, in-order sums from +0 and
// libm sin/cos. The whole program is validated before the first workspace
// write, so an invalid program leaves the workspaces untouched. Callers price
// each evaluation with the canonical owner's tariff, identical for the
// interpreter; the kernel therefore charges no work and allocates nothing.
// Under an active execution scope it polls only the deadline and prior
// refusals (a zero-work checkpoint) at entry and after every bounded number
// of validated entries or executed element operations; a refusal during
// execution leaves the workspaces partially written, and callers discard them.
// Validation and execution run on the calling thread in one
// GradualNearestScope, which restores the caller's mode.
#include <cmath>
#include <cstdint>
#include <limits>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "rounding.hpp"

namespace {

using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr int64_t kFields = 8;
constexpr int64_t kGather = 1;
constexpr int64_t kAdd = 2;
constexpr int64_t kSubtract = 3;
constexpr int64_t kSubtractSame = 4;
constexpr int64_t kMultiply = 5;
constexpr int64_t kNegate = 6;
constexpr int64_t kReduceSum = 7;
constexpr int64_t kSine = 8;
constexpr int64_t kCosine = 9;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
constexpr double kPi = 3.141592653589793;  // binary64 pi, Python math.pi.
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kHalfPi = 0.5 * kPi;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kTranscendentalSteps = 4;

// Zero-work deadline checkpoints poll once per this many validated rows and
// index entries, or executed element operations (aux + 1 per REDUCE_SUM
// element, one otherwise).
constexpr int64_t kCheckpointInterval = 4096;

class Checkpoints final {
 public:
  // Records `operations` units and polls the active scope once the interval
  // is reached; throws ExecutionRefusal on a deadline or prior refusal.
  void advance(int64_t operations) {
    pending_ += operations;
    if (pending_ >= kCheckpointInterval) poll();
  }

  // Polls the active scope now and restarts the interval.
  void poll() {
    pending_ = 0;
    phx::mc::native_execution_checkpoint();
  }

 private:
  int64_t pending_ = 0;
};

struct Row {
  int64_t opcode, out, size, a, a_index, b, b_index, aux;
};

Row row_at(const int64_t* instructions, int64_t row) noexcept {
  const int64_t* f = instructions + row * kFields;
  return Row{f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]};
}

// NumPy minimum/maximum: the first NaN operand; equal operands order -0 below +0.
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

double down(double value, int steps = 1) noexcept {
  for (int step = 0; step < steps; ++step) value = next_down(value);
  return value;
}

double up(double value, int steps = 1) noexcept {
  for (int step = 0; step < steps; ++step) value = next_up(value);
  return value;
}

double product(double first, double second) noexcept {
  if ((first == 0.0 && std::isinf(second)) || (second == 0.0 && std::isinf(first))) return 0.0;
  return first * second;
}

// Whether phase + 2 k pi may lie in [lower, upper] for an integer k.
bool contains_phase(double lower, double upper, double phase) noexcept {
  const double slack = 8.0 * kEpsilon * maximum(1.0, std::fabs(lower));
  const double index = std::ceil((lower - slack - phase) / kTwoPi);
  return phase + index * kTwoPi <= upper + slack;
}

double sine(double value) noexcept { return std::isfinite(value) ? std::sin(value) : kNaN; }
double cosine(double value) noexcept { return std::isfinite(value) ? std::cos(value) : kNaN; }

template <class Function>
void trigonometric(double lower, double upper, Function function, double high_phase,
                   double low_phase, double& out_lower, double& out_upper) noexcept {
  const bool wide = !(upper - lower < kTwoPi) || !std::isfinite(lower) || !std::isfinite(upper);
  const double first = function(lower);
  const double last = function(upper);
  double low = minimum(first, last);
  double high = maximum(first, last);
  if (contains_phase(lower, upper, high_phase) || wide) high = 1.0;
  if (contains_phase(lower, upper, low_phase) || wide) low = -1.0;
  low = down(low, kTranscendentalSteps);
  high = up(high, kTranscendentalSteps);
  out_lower = maximum(low, -1.0);
  out_upper = minimum(high, 1.0);
}

// Every element offset + (index < 0 ? k : indices[index + k]), k < size, lies
// in [0, limit) with 0 <= offset <= limit; index is -1 or a range of indices.
bool operand_valid(const int64_t* indices, int64_t index_count, int64_t offset, int64_t index,
                   int64_t size, int64_t limit, Checkpoints& checkpoints) {
  if (offset < 0 || offset > limit) return false;
  const int64_t room = limit - offset;
  if (index == -1) return size <= room;
  if (index < 0 || index > index_count || size > index_count - index) return false;
  for (int64_t k = 0; k < size; ++k) {
    const int64_t value = indices[index + k];
    if (value < 0 || value >= room) return false;
    checkpoints.advance(1);
  }
  return true;
}

bool row_valid(const Row& row, int64_t previous_end, const int64_t* indices, int64_t index_count,
               const double* parameters, int64_t parameter_count, int64_t workspace_size,
               Checkpoints& checkpoints) {
  if (row.opcode < kGather || row.opcode > kCosine) return false;
  if (row.size < 0 || row.out < previous_end || row.out > workspace_size - row.size) return false;
  const auto operand = [&](int64_t offset, int64_t index) {
    return operand_valid(indices, index_count, offset, index, row.size, row.out, checkpoints);
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
      return row.aux == 0 && operand(row.a, row.a_index) && operand(row.b, row.b_index);
    default:
      return row.b == -1 && row.b_index == -1 && row.aux == 0 && operand(row.a, row.a_index);
  }
}

// Whether every row is valid in order.
bool program_valid(const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
                   int64_t index_count, const double* parameters, int64_t parameter_count,
                   int64_t workspace_size, Checkpoints& checkpoints) {
  int64_t previous_end = 0;
  for (int64_t r = 0; r < instruction_count; ++r) {
    const Row row = row_at(instructions, r);
    if (!row_valid(row, previous_end, indices, index_count, parameters, parameter_count,
                   workspace_size, checkpoints)) {
      return false;
    }
    previous_end = row.out + row.size;
    checkpoints.advance(1);
  }
  return true;
}

inline int64_t address(const int64_t* indices, int64_t offset, int64_t index,
                       int64_t element) noexcept {
  return offset + (index < 0 ? element : indices[index + element]);
}

// In-order REDUCE_SUM accumulators of one element: the lower and upper sums
// and the scale sum of max(|lo|, |hi|), each from +0.
struct Reduction {
  double lower = 0.0;
  double upper = 0.0;
  double scale = 0.0;
};

// Continues each accumulator over workspace terms [first, last) in order.
inline void accumulate(const double* lower, const double* upper, int64_t first, int64_t last,
                       Reduction& sum) noexcept {
  double lower_sum = sum.lower;
  double upper_sum = sum.upper;
  double scale = sum.scale;
  for (int64_t t = first; t < last; ++t) {
    lower_sum += lower[t];
    upper_sum += upper[t];
  }
  for (int64_t t = first; t < last; ++t) {
    scale += maximum(std::fabs(lower[t]), std::fabs(upper[t]));
  }
  sum = {lower_sum, upper_sum, scale};
}

inline void store_reduction(const Reduction& sum, double gamma, double* lower, double* upper,
                            int64_t target) noexcept {
  lower[target] = down(sum.lower - gamma * sum.scale);
  upper[target] = up(sum.upper + gamma * sum.scale);
}

// Executes elements [begin, end) of one validated row.
void execute_row(const Row& row, int64_t begin, int64_t end, const int64_t* indices,
                 const double* parameters, double* lower, double* upper) noexcept {
  const int64_t out = row.out;
  switch (row.opcode) {
    case kGather:
      for (int64_t k = begin; k < end; ++k) {
        const int64_t a = address(indices, row.a, row.a_index, k);
        lower[out + k] = lower[a];
        upper[out + k] = upper[a];
      }
      return;
    case kNegate:
      for (int64_t k = begin; k < end; ++k) {
        const int64_t a = address(indices, row.a, row.a_index, k);
        const double low = lower[a];
        const double high = upper[a];
        lower[out + k] = -high;
        upper[out + k] = -low;
      }
      return;
    case kSine:
    case kCosine: {
      const bool is_sine = row.opcode == kSine;
      for (int64_t k = begin; k < end; ++k) {
        const int64_t a = address(indices, row.a, row.a_index, k);
        if (is_sine) {
          trigonometric(lower[a], upper[a], sine, kHalfPi, -kHalfPi, lower[out + k],
                        upper[out + k]);
        } else {
          trigonometric(lower[a], upper[a], cosine, 0.0, kPi, lower[out + k], upper[out + k]);
        }
      }
      return;
    }
    case kReduceSum: {
      const double gamma = parameters[row.b_index];
      for (int64_t k = begin; k < end; ++k) {
        const int64_t base = row.a + k * row.aux;
        Reduction sum;
        accumulate(lower, upper, base, base + row.aux, sum);
        store_reduction(sum, gamma, lower, upper, out + k);
      }
      return;
    }
    default:
      break;
  }
  for (int64_t k = begin; k < end; ++k) {
    const int64_t a = address(indices, row.a, row.a_index, k);
    const int64_t b = address(indices, row.b, row.b_index, k);
    const double a_low = lower[a];
    const double a_high = upper[a];
    const double b_low = lower[b];
    const double b_high = upper[b];
    double low = 0.0;
    double high = 0.0;
    switch (row.opcode) {
      case kAdd:
        low = down(a_low + b_low);
        high = up(a_high + b_high);
        break;
      case kSubtract:
      case kSubtractSame:
        low = down(a_low - b_high);
        high = up(a_high - b_low);
        if (a_low == a_high && b_low == b_high && a_low == b_low && std::isfinite(a_low)) {
          low = high = 0.0;
        }
        if (row.opcode == kSubtractSame && std::isfinite(a_low) && std::isfinite(a_high)) {
          low = high = 0.0;
        }
        break;
      default: {  // kMultiply
        const double ll = product(a_low, b_low);
        const double lu = product(a_low, b_high);
        const double ul = product(a_high, b_low);
        const double uu = product(a_high, b_high);
        low = down(minimum(minimum(ll, lu), minimum(ul, uu)));
        high = up(maximum(maximum(ll, lu), maximum(ul, uu)));
        break;
      }
    }
    lower[out + k] = low;
    upper[out + k] = high;
  }
}

// Executes a validated nonempty REDUCE_SUM row whose element costs more than
// one checkpoint interval: each element's terms accumulate in order in chunks
// of at most one interval, polling between chunks, so the accumulators and
// their IEEE operation order match the unchunked formula.
void execute_long_reduction(const Row& row, const double* parameters, double* lower, double* upper,
                            Checkpoints& checkpoints) {
  const double gamma = parameters[row.b_index];
  for (int64_t k = 0; k < row.size; ++k) {
    const int64_t base = row.a + k * row.aux;
    Reduction sum;
    for (int64_t first = 0; first < row.aux; first += kCheckpointInterval) {
      const int64_t last =
          row.aux - first < kCheckpointInterval ? row.aux : first + kCheckpointInterval;
      accumulate(lower, upper, base + first, base + last, sum);
      checkpoints.advance(last - first);
    }
    store_reduction(sum, gamma, lower, upper, row.out + k);
    checkpoints.advance(1);
  }
}

}  // namespace

extern "C" {

int32_t phx_mc_interval_program_execute(const int64_t* instructions, int64_t instruction_count,
                                        const int64_t* indices, int64_t index_count,
                                        const double* parameters, int64_t parameter_count,
                                        double* lower, double* upper, int64_t workspace_size) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!phx::mc::addressable(instruction_count, kFields, sizeof(int64_t)) ||
        !phx::mc::addressable(index_count, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(parameter_count, 1, sizeof(double)) ||
        !phx::mc::addressable(workspace_size, 1, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if ((instruction_count > 0 && instructions == nullptr) ||
        (index_count > 0 && indices == nullptr) || (parameter_count > 0 && parameters == nullptr) ||
        (workspace_size > 0 && (lower == nullptr || upper == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    phx::mc::native_execution_checkpoint();
    Checkpoints checkpoints;
    if (!program_valid(instructions, instruction_count, indices, index_count, parameters,
                       parameter_count, workspace_size, checkpoints)) {
      return PHX_MC_INVALID_INPUT;
    }
    // Validated rows execute in element blocks of at most one checkpoint
    // interval of operations; the first write follows a fresh checkpoint and
    // the final partial block is polled before success is reported.
    checkpoints.poll();
    for (int64_t r = 0; r < instruction_count; ++r) {
      const Row row = row_at(instructions, r);
      // Zero-size rows execute nothing, whatever their reduction length. A
      // nonempty REDUCE_SUM row reads size * aux terms below out, so aux + 1
      // stays within the addressable workspace size.
      if (row.size == 0) continue;
      const int64_t cost = row.opcode == kReduceSum ? row.aux + 1 : 1;
      if (cost > kCheckpointInterval) {
        execute_long_reduction(row, parameters, lower, upper, checkpoints);
        continue;
      }
      const int64_t block = kCheckpointInterval / cost;
      for (int64_t begin = 0; begin < row.size; begin += block) {
        const int64_t end = row.size - begin < block ? row.size : begin + block;
        execute_row(row, begin, end, indices, parameters, lower, upper);
        checkpoints.advance((end - begin) * cost);
      }
    }
    checkpoints.poll();
    return PHX_MC_OK;
  });
}

}  // extern "C"
