// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Batched exact and filtered kernels of Bernstein and box-pair proofs.
//
// Interval kernels enclose each endpoint operation before extrema reduction.
// Exact kernels use checked wide integers and refuse, never clamp, values
// outside the selected word range. Disjoint fixed chunks are deterministic.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"
#include "rounding.hpp"
#include "wide_integer.hpp"

namespace {

using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr std::size_t kItemChunk = 16;
constexpr double kIntervalRounding = 4.0 * std::numeric_limits<double>::epsilon();
constexpr double kIntervalUnderflow = std::numeric_limits<double>::denorm_min();

phx::mc::ParallelRequest request(int32_t parallelism, std::size_t chunk = kItemChunk) {
  return phx::mc::ParallelRequest{chunk, static_cast<uint32_t>(parallelism)};
}

// IEEE 754-2019 minimum/maximum as NumPy computes them: NaN propagates and
// -0 orders below +0.
double ieee_minimum(double a, double b) noexcept {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  if (a < b) return a;
  if (b < a) return b;
  return std::signbit(a) ? a : b;
}

double ieee_maximum(double a, double b) noexcept {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  if (a > b) return a;
  if (b > a) return b;
  return std::signbit(a) ? b : a;
}

// ------------------------------------------------------- interval de Casteljau

// One reference step on `rows` rows of `width` entries, in place: row i becomes
// the enclosure of (1 - t) x_i + t x_{i+1} over the parameter endpoints.
void casteljau_step(double* lower, double* upper, std::size_t rows, std::size_t width, double t0,
                    double t1) noexcept {
  const std::size_t entries = (rows - 1) * width;
  if (t0 == 0.0 && t1 == 0.0) return;
  if (t0 == 1.0 && t1 == 1.0) {
    for (std::size_t index = 0; index < entries; ++index) {
      lower[index] = lower[index + width];
      upper[index] = upper[index + width];
    }
    return;
  }
  const bool single = t0 == t1;
  const double c0 = 1.0 - t0;
  const double c1 = 1.0 - t1;
  for (std::size_t index = 0; index < entries; ++index) {
    const double l0 = lower[index];
    const double l1 = lower[index + width];
    const double h0 = upper[index];
    const double h1 = upper[index + width];
    const double m0 = ieee_maximum(std::fabs(l0), std::fabs(h0));
    const double m1 = ieee_maximum(std::fabs(l1), std::fabs(h1));
    const double error0 = kIntervalRounding * (c0 * m0 + t0 * m1) + kIntervalUnderflow;
    double low = next_down(c0 * l0 + t0 * l1 - error0);
    double high = next_up(c0 * h0 + t0 * h1 + error0);
    if (!single) {
      const double error1 = kIntervalRounding * (c1 * m0 + t1 * m1) + kIntervalUnderflow;
      low = ieee_minimum(low, next_down(c1 * l0 + t1 * l1 - error1));
      high = ieee_maximum(high, next_up(c1 * h0 + t1 * h1 + error1));
    }
    lower[index] = low;
    upper[index] = high;
  }
}

void interval_restrict_item(std::size_t degree, std::size_t width, const double* lower,
                            const double* upper, const double* parameters, int32_t sides,
                            double* out_lower, double* out_upper, std::vector<double>& work_lower,
                            std::vector<double>& work_upper) {
  const std::size_t entries = (degree + 1) * width;
  std::copy(lower, lower + entries, out_lower);
  std::copy(upper, upper + entries, out_upper);
  if (sides & 1) {
    work_lower.assign(out_lower, out_lower + entries);
    work_upper.assign(out_upper, out_upper + entries);
    for (std::size_t step = 1; step <= degree; ++step) {
      casteljau_step(work_lower.data(), work_upper.data(), degree + 2 - step, width, parameters[0],
                     parameters[1]);
      std::copy(work_lower.begin(), work_lower.begin() + width, out_lower + step * width);
      std::copy(work_upper.begin(), work_upper.begin() + width, out_upper + step * width);
    }
  }
  if (sides & 2) {
    work_lower.assign(out_lower, out_lower + entries);
    work_upper.assign(out_upper, out_upper + entries);
    for (std::size_t step = 1; step <= degree; ++step) {
      const std::size_t rows = degree + 2 - step;
      casteljau_step(work_lower.data(), work_upper.data(), rows, width, parameters[2],
                     parameters[3]);
      const std::size_t last = (rows - 2) * width;
      std::copy(work_lower.begin() + last, work_lower.begin() + last + width,
                out_lower + (degree - step) * width);
      std::copy(work_upper.begin() + last, work_upper.begin() + last + width,
                out_upper + (degree - step) * width);
    }
  }
}

// ------------------------------------------------------ interval product

constexpr double kVeltkampSplitter = 134217729.0;
constexpr double kSplitMagnitudeLimit = 0x1p996;
constexpr double kProductMagnitudeFloor = 0x1p-969;

// Whether binary64 `product = first * second` is error free (Dekker
// TwoProduct inside its exact window, or an exact zero factor), in the host
// reference's operation order.
bool product_is_exact(double first, double second, double product) noexcept {
  const double first_scaled = kVeltkampSplitter * first;
  const double first_high = first_scaled - (first_scaled - first);
  const double first_low = first - first_high;
  const double second_scaled = kVeltkampSplitter * second;
  const double second_high = second_scaled - (second_scaled - second);
  const double second_low = second - second_high;
  const double error = ((first_high * second_high - product) + first_high * second_low +
                        first_low * second_high) +
                       first_low * second_low;
  const bool window = std::isfinite(product) && std::fabs(first) < kSplitMagnitudeLimit &&
                      std::fabs(second) < kSplitMagnitudeLimit &&
                      std::fabs(product) >= kProductMagnitudeFloor;
  const bool zero_factor = (first == 0.0 && std::isfinite(second)) ||
                           (second == 0.0 && std::isfinite(first));
  return zero_factor || (window && error == 0.0);
}

// Set-based interval product of [a_lo, a_hi] * [b_lo, b_hi]: a zero endpoint
// times an unbounded one contributes 0, endpoints round outward only when
// their binary64 product was inexact, and sign-determined results keep their sign.
void interval_product(double a_lo, double a_hi, double b_lo, double b_hi, double& lower,
                      double& upper) noexcept {
  const double left[4] = {a_lo, a_lo, a_hi, a_hi};
  const double right[4] = {b_lo, b_hi, b_lo, b_hi};
  double low = std::numeric_limits<double>::infinity();
  double high = -std::numeric_limits<double>::infinity();
  for (int pair = 0; pair < 4; ++pair) {
    const double l = left[pair];
    const double r = right[pair];
    const bool zero_unbounded = (l == 0.0 && std::isinf(r)) || (std::isinf(l) && r == 0.0);
    const double product = zero_unbounded ? 0.0 : l * r;
    if (std::isnan(product)) {
      lower = -std::numeric_limits<double>::infinity();
      upper = std::numeric_limits<double>::infinity();
      return;
    }
    const bool exact = zero_unbounded || product_is_exact(l, r, product);
    low = ieee_minimum(low, exact ? product : next_down(product));
    high = ieee_maximum(high, exact ? product : next_up(product));
  }
  const bool first_nonnegative = a_lo >= 0.0;
  const bool first_nonpositive = a_hi <= 0.0;
  const bool second_nonnegative = b_lo >= 0.0;
  const bool second_nonpositive = b_hi <= 0.0;
  if ((first_nonnegative && second_nonnegative) || (first_nonpositive && second_nonpositive)) {
    low = ieee_maximum(low, 0.0);
  }
  if ((first_nonnegative && second_nonpositive) || (first_nonpositive && second_nonnegative)) {
    high = ieee_minimum(high, 0.0);
  }
  lower = low;
  upper = high;
}

// Whether binary64 `total = first + second` is error free (TwoSum), as the host
// reference computes it.
bool sum_is_exact(double first, double second, double total) noexcept {
  const double partner = total - first;
  const double error = (first - (total - partner)) + (second - partner);
  return std::isfinite(total) && error == 0.0;
}

// Outward rounding of the inexact endpoints of [low, high]; a NaN endpoint
// makes the widened interval the whole line.
void round_out_inexact(double low, double high, bool low_exact, bool high_exact,
                       double& lower, double& upper) noexcept {
  const bool invalid = std::isnan(low) || std::isnan(high);
  const double widened_low = next_down(invalid ? -std::numeric_limits<double>::infinity() : low);
  const double widened_high = next_up(invalid ? std::numeric_limits<double>::infinity() : high);
  lower = low_exact && !invalid ? low : widened_low;
  upper = high_exact && !invalid ? high : widened_high;
}

// ------------------------------------------------------ exact combinatorics

// b (b - 1) ... (b - a + 1); zero when a > b.
bool falling(std::int64_t b, std::int64_t a, std::uint64_t& out) noexcept {
  out = 1;
  if (a > b) {
    out = 0;
    return true;
  }
  for (std::int64_t factor = b - a + 1; factor <= b; ++factor) {
    if (!phx::mc::checked_multiply(out, static_cast<std::uint64_t>(factor), out)) return false;
  }
  return true;
}

bool binomial(std::int64_t n, std::int64_t k, std::uint64_t& out) noexcept {
  if (k < 0 || k > n) {
    out = 0;
    return true;
  }
  k = std::min(k, n - k);
  std::uint64_t value = 1;
  for (std::int64_t step = 1; step <= k; ++step) {
    // value (n - k + step) / step is the integer C(n - k + step, step). Cancelling
    // g = gcd(value, step) leaves a divisor step / g coprime to value / g, so it
    // divides n - k + step and the remaining product is the exact result.
    const std::uint64_t factor = static_cast<std::uint64_t>(n - k + step);
    const std::uint64_t common = std::gcd(value, static_cast<std::uint64_t>(step));
    const std::uint64_t divisor = static_cast<std::uint64_t>(step) / common;
    if (!phx::mc::checked_multiply(value / common, factor / divisor, value)) return false;
  }
  out = value;
  return true;
}

bool lcm_u64(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
  if (a == 0 || b == 0) {
    out = std::max(a, b);
    return true;
  }
  return phx::mc::checked_multiply(a / std::gcd(a, b), b, out);
}

// Runs `body` with a value of the wide-integer word type of `limbs` limbs.
template <class Body>
int32_t with_word(int32_t limbs, Body&& body) {
  switch (limbs) {
    case 4:
      return body(phx::mc::WideInt<4>{});
    case 8:
      return body(phx::mc::WideInt<8>{});
    case 16:
      return body(phx::mc::WideInt<16>{});
    case 32:
      return body(phx::mc::WideInt<32>{});
    case 64:
      return body(phx::mc::WideInt<64>{});
    default:
      return PHX_MC_INVALID_ARGUMENT;
  }
}

bool word_limbs_valid(int32_t limbs) noexcept {
  return limbs == 4 || limbs == 8 || limbs == 16 || limbs == 32 || limbs == 64;
}

int64_t composition_count(int32_t total, int32_t parts);

// Compositions of `total` into `parts` nonnegative parts, first part descending
// (the order of `_compositions` in the host reference), flattened row-major.
std::vector<int32_t> compositions(int32_t total, int32_t parts) {
  const int64_t rows = composition_count(total, parts);
  if (rows < 0 || !phx::mc::addressable(rows, parts, sizeof(int32_t))) {
    throw std::length_error("composition table is not addressable");
  }
  std::vector<int32_t> out;
  out.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(parts));
  std::vector<int32_t> current(static_cast<std::size_t>(parts), 0);
  current[0] = total;
  while (true) {
    out.insert(out.end(), current.begin(), current.end());
    int32_t slot = parts - 2;
    while (slot >= 0 && current[static_cast<std::size_t>(slot)] == 0) --slot;
    if (slot < 0) break;
    const int32_t remaining = current.back();
    current.back() = 0;
    --current[static_cast<std::size_t>(slot)];
    current[static_cast<std::size_t>(slot + 1)] = remaining + 1;
  }
  return out;
}

int64_t composition_count(int32_t total, int32_t parts) {
  std::uint64_t value = 0;
  if (!binomial(static_cast<std::int64_t>(total) + parts - 1, parts - 1, value) ||
      value > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
    return -1;
  }
  return static_cast<int64_t>(value);
}

template <class Word>
void record_extremes(const std::vector<Word>& values, int64_t* extremes) noexcept {
  std::size_t least = 0;
  std::size_t greatest = 0;
  for (std::size_t index = 1; index < values.size(); ++index) {
    if (phx::mc::compare(values[index], values[least]) < 0) least = index;
    if (phx::mc::compare(values[index], values[greatest]) > 0) greatest = index;
  }
  extremes[0] = static_cast<int64_t>(least);
  extremes[1] = static_cast<int64_t>(greatest);
}

// ------------------------------------------------------- Bernstein controls

struct ControlsCall {
  int32_t domain;
  int32_t dimension;
  int32_t degree_width;
  const int32_t* degrees;
  const int64_t* term_offsets;
  const int32_t* exponents;
  const std::uint8_t* numerators;
  const int64_t* control_offsets;
  std::uint8_t* controls;
  int64_t* denominators;
  int64_t* extremes;
};

// Integer weight of every (control, term) over one item denominator.
class WeightTable final {
 public:
  // Returns PHX_MC_OK, PHX_MC_RANGE_ERROR or PHX_MC_INVALID_INPUT.
  int32_t build(const ControlsCall& call, int64_t item) {
    const int32_t* degree = call.degrees + item * call.degree_width;
    const int64_t begin = call.term_offsets[item];
    const int64_t end = call.term_offsets[item + 1];
    const int32_t dimension = call.dimension;
    terms_ = static_cast<std::size_t>(end - begin);
    for (int32_t axis = 0; axis < call.degree_width; ++axis) {
      if (degree[axis] < 0) return PHX_MC_INVALID_INPUT;
    }
    exponents_.assign(call.exponents + begin * dimension, call.exponents + end * dimension);
    for (const int32_t value : exponents_) {
      if (value < 0) return PHX_MC_INVALID_INPUT;
    }
    std::uint64_t denominator = 1;
    switch (call.domain) {
      case PHX_MC_BERNSTEIN_SIMPLEX: {
        const int32_t n = degree[0];
        int32_t tmax = 0;
        totals_.assign(terms_, 0);
        for (std::size_t term = 0; term < terms_; ++term) {
          int64_t total = 0;
          for (int32_t axis = 0; axis < dimension; ++axis) total += exponent(term, axis);
          if (total > n) return PHX_MC_INVALID_INPUT;
          totals_[term] = static_cast<int32_t>(total);
          tmax = std::max(tmax, totals_[term]);
        }
        if (!falling(n, tmax, denominator)) return PHX_MC_RANGE_ERROR;
        scales_.assign(terms_, 0);
        for (std::size_t term = 0; term < terms_; ++term) {
          if (!falling(n - totals_[term], tmax - totals_[term], scales_[term])) {
            return PHX_MC_RANGE_ERROR;
          }
        }
        rows_ = compositions(n, dimension + 1);
        parts_ = dimension + 1;
        break;
      }
      case PHX_MC_BERNSTEIN_BOX: {
        axis_lcm_.assign(static_cast<std::size_t>(dimension), 1);
        for (int32_t axis = 0; axis < dimension; ++axis) {
          for (std::size_t term = 0; term < terms_; ++term) {
            if (exponent(term, axis) > degree[axis]) return PHX_MC_INVALID_INPUT;
            std::uint64_t value = 0;
            if (!binomial(degree[axis], exponent(term, axis), value) ||
                !lcm_u64(axis_lcm_[static_cast<std::size_t>(axis)], value,
                         axis_lcm_[static_cast<std::size_t>(axis)])) {
              return PHX_MC_RANGE_ERROR;
            }
          }
          if (!phx::mc::checked_multiply(denominator, axis_lcm_[static_cast<std::size_t>(axis)],
                                         denominator)) {
            return PHX_MC_RANGE_ERROR;
          }
        }
        break;
      }
      case PHX_MC_BERNSTEIN_PRISM: {
        const int32_t triangle = degree[0];
        const int32_t line = degree[1];
        int32_t tmax = 0;
        std::uint64_t line_lcm = 1;
        totals_.assign(terms_, 0);
        for (std::size_t term = 0; term < terms_; ++term) {
          const int64_t total = int64_t{exponent(term, 0)} + exponent(term, 1);
          if (total > triangle || exponent(term, 2) > line) return PHX_MC_INVALID_INPUT;
          totals_[term] = static_cast<int32_t>(total);
          tmax = std::max(tmax, totals_[term]);
          std::uint64_t value = 0;
          if (!binomial(line, exponent(term, 2), value) || !lcm_u64(line_lcm, value, line_lcm)) {
            return PHX_MC_RANGE_ERROR;
          }
        }
        std::uint64_t triangle_denominator = 1;
        if (!falling(triangle, tmax, triangle_denominator) ||
            !phx::mc::checked_multiply(triangle_denominator, line_lcm, denominator)) {
          return PHX_MC_RANGE_ERROR;
        }
        scales_.assign(terms_, 0);
        for (std::size_t term = 0; term < terms_; ++term) {
          if (!falling(triangle - totals_[term], tmax - totals_[term], scales_[term])) {
            return PHX_MC_RANGE_ERROR;
          }
        }
        axis_lcm_.assign(1, line_lcm);
        rows_ = compositions(triangle, 3);
        parts_ = 3;
        break;
      }
      default:
        return PHX_MC_INVALID_ARGUMENT;
    }
    if (denominator > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
      return PHX_MC_RANGE_ERROR;
    }
    denominator_ = static_cast<int64_t>(denominator);
    return PHX_MC_OK;
  }

  int64_t denominator() const noexcept { return denominator_; }

  // Weight of `term` in control `control` (domain order); false on overflow.
  bool weight(const ControlsCall& call, int64_t item, std::size_t control, std::size_t term,
              std::uint64_t& out) const noexcept {
    const int32_t* degree = call.degrees + item * call.degree_width;
    switch (call.domain) {
      case PHX_MC_BERNSTEIN_SIMPLEX:
        return simplex_weight(control, term, out);
      case PHX_MC_BERNSTEIN_BOX: {
        out = 1;
        std::size_t remainder = control;
        for (int32_t axis = call.dimension - 1; axis >= 0; --axis) {
          const std::size_t extent = static_cast<std::size_t>(degree[axis]) + 1;
          const int32_t beta = static_cast<int32_t>(remainder % extent);
          remainder /= extent;
          std::uint64_t factor = 0;
          if (!line_weight(degree[axis], beta, exponent(term, axis),
                           axis_lcm_[static_cast<std::size_t>(axis)], factor)) {
            return false;
          }
          if (factor == 0) {
            out = 0;
            return true;
          }
          if (!phx::mc::checked_multiply(out, factor, out)) return false;
        }
        return true;
      }
      case PHX_MC_BERNSTEIN_PRISM: {
        const std::size_t line_extent = static_cast<std::size_t>(degree[1]) + 1;
        const int32_t line_beta = static_cast<int32_t>(control % line_extent);
        std::uint64_t triangle = 0;
        std::uint64_t line = 0;
        if (!simplex_weight(control / line_extent, term, triangle) ||
            !line_weight(degree[1], line_beta, exponent(term, 2), axis_lcm_[0], line)) {
          return false;
        }
        return phx::mc::checked_multiply(triangle, line, out);
      }
      default:
        return false;
    }
  }

 private:
  int32_t exponent(std::size_t term, int32_t axis) const noexcept {
    return exponents_[term * static_cast<std::size_t>(exponents_width()) +
                      static_cast<std::size_t>(axis)];
  }
  int32_t exponents_width() const noexcept {
    return terms_ == 0 ? 1 : static_cast<int32_t>(exponents_.size() / terms_);
  }

  // prod_p falling(beta_p, alpha_p) * scale(term) with alpha = (0, exponents...).
  bool simplex_weight(std::size_t row, std::size_t term, std::uint64_t& out) const noexcept {
    const int32_t* beta = rows_.data() + row * static_cast<std::size_t>(parts_);
    out = scales_[term];
    for (int32_t part = 1; part < parts_; ++part) {
      std::uint64_t factor = 0;
      if (!falling(beta[part], exponent(term, part - 1), factor)) return false;
      if (factor == 0) {
        out = 0;
        return true;
      }
      if (!phx::mc::checked_multiply(out, factor, out)) return false;
    }
    return true;
  }

  // C(beta, alpha) * lcm / C(n, alpha), zero when alpha > beta.
  static bool line_weight(int32_t n, int32_t beta, int32_t alpha, std::uint64_t lcm,
                          std::uint64_t& out) noexcept {
    if (alpha > beta) {
      out = 0;
      return true;
    }
    std::uint64_t numerator = 0;
    std::uint64_t denominator = 0;
    if (!binomial(beta, alpha, numerator) || !binomial(n, alpha, denominator)) return false;
    return phx::mc::checked_multiply(numerator, lcm / denominator, out);
  }

  std::size_t terms_ = 0;
  std::vector<int32_t> exponents_;
  std::vector<int32_t> totals_;
  std::vector<std::uint64_t> scales_;
  std::vector<std::uint64_t> axis_lcm_;
  std::vector<int32_t> rows_;
  int32_t parts_ = 1;
  int64_t denominator_ = 1;
};

int64_t domain_control_count(int32_t domain, int32_t dimension, const int32_t* degree) {
  switch (domain) {
    case PHX_MC_BERNSTEIN_SIMPLEX:
      return degree[0] < 0 ? -1 : composition_count(degree[0], dimension + 1);
    case PHX_MC_BERNSTEIN_BOX: {
      int64_t count = 1;
      for (int32_t axis = 0; axis < dimension; ++axis) {
        if (degree[axis] < 0 ||
            !phx::mc::checked_multiply(count, int64_t{degree[axis]} + 1, count)) {
          return -1;
        }
      }
      return count;
    }
    case PHX_MC_BERNSTEIN_PRISM: {
      if (degree[0] < 0 || degree[1] < 0) return -1;
      const int64_t triangle = composition_count(degree[0], 3);
      int64_t count = 0;
      if (triangle < 0 || !phx::mc::checked_multiply(triangle, int64_t{degree[1]} + 1, count)) {
        return -1;
      }
      return count;
    }
    default:
      return -1;
  }
}

// Tensor (box) controls axis by axis: pass i maps exponent a to control b with
// C(b, a) * (L_i / C(n_i, a)), L_i the lcm of C(n_i, a) over the item's
// exponents on axis i, so the item denominator is prod L_i as for the
// per-control weights and the work is sum_i (n_i + 1) * controls.
template <class Word>
int32_t box_controls(const ControlsCall& call, int64_t item, std::vector<Word>& values,
                     int64_t& denominator) {
  const int32_t dimension = call.dimension;
  const int32_t* degree = call.degrees + item * call.degree_width;
  const int64_t term_begin = call.term_offsets[item];
  const std::size_t terms = static_cast<std::size_t>(call.term_offsets[item + 1] - term_begin);
  std::vector<std::size_t> extent(static_cast<std::size_t>(dimension));
  std::vector<std::size_t> stride(static_cast<std::size_t>(dimension));
  std::size_t size = 1;
  for (int32_t axis = dimension - 1; axis >= 0; --axis) {
    if (degree[axis] < 0) return PHX_MC_INVALID_INPUT;
    extent[static_cast<std::size_t>(axis)] = static_cast<std::size_t>(degree[axis]) + 1;
    stride[static_cast<std::size_t>(axis)] = size;
    size *= extent[static_cast<std::size_t>(axis)];
  }
  values.assign(size, Word{});
  std::vector<std::vector<bool>> present(static_cast<std::size_t>(dimension));
  for (int32_t axis = 0; axis < dimension; ++axis) {
    present[static_cast<std::size_t>(axis)].assign(extent[static_cast<std::size_t>(axis)], false);
  }
  for (std::size_t term = 0; term < terms; ++term) {
    const int32_t* exponent = call.exponents + (static_cast<std::size_t>(term_begin) + term) *
                                                   static_cast<std::size_t>(dimension);
    std::size_t offset = 0;
    for (int32_t axis = 0; axis < dimension; ++axis) {
      if (exponent[axis] < 0 || exponent[axis] > degree[axis]) return PHX_MC_INVALID_INPUT;
      present[static_cast<std::size_t>(axis)][static_cast<std::size_t>(exponent[axis])] = true;
      offset += static_cast<std::size_t>(exponent[axis]) * stride[static_cast<std::size_t>(axis)];
    }
    const Word numerator = Word::load(
        call.numerators + (static_cast<std::size_t>(term_begin) + term) * Word::kBytes);
    if (!phx::mc::checked_add(values[offset], numerator, values[offset])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  std::uint64_t total_denominator = 1;
  std::vector<Word> next(size);
  for (int32_t axis = 0; axis < dimension; ++axis) {
    const std::size_t n = extent[static_cast<std::size_t>(axis)];
    const auto& used = present[static_cast<std::size_t>(axis)];
    std::uint64_t lcm = 1;
    for (std::size_t a = 0; a < n; ++a) {
      std::uint64_t value = 0;
      if (used[a] && (!binomial(static_cast<std::int64_t>(n - 1), static_cast<std::int64_t>(a), value) ||
                      !lcm_u64(lcm, value, lcm))) {
        return PHX_MC_RANGE_ERROR;
      }
    }
    if (!phx::mc::checked_multiply(total_denominator, lcm, total_denominator)) {
      return PHX_MC_RANGE_ERROR;
    }
    // weights[b * n + a] for used a <= b.
    if (!phx::mc::addressable(static_cast<int64_t>(n), static_cast<int64_t>(n), Word::kBytes)) {
      return PHX_MC_RANGE_ERROR;
    }
    std::vector<Word> weights(n * n);
    for (std::size_t b = 0; b < n; ++b) {
      for (std::size_t a = 0; a <= b; ++a) {
        if (!used[a]) continue;
        std::uint64_t numerator = 0;
        std::uint64_t binomial_n = 0;
        std::uint64_t weight = 0;
        if (!binomial(static_cast<std::int64_t>(b), static_cast<std::int64_t>(a), numerator) ||
            !binomial(static_cast<std::int64_t>(n - 1), static_cast<std::int64_t>(a), binomial_n) ||
            !phx::mc::checked_multiply(numerator, lcm / binomial_n, weight)) {
          return PHX_MC_RANGE_ERROR;
        }
        weights[b * n + a] = Word::from_uint64(weight);
      }
    }
    const std::size_t inner = stride[static_cast<std::size_t>(axis)];
    const std::size_t outer = size / (n * inner);
    for (std::size_t block = 0; block < outer; ++block) {
      for (std::size_t position = 0; position < inner; ++position) {
        const std::size_t base = block * n * inner + position;
        for (std::size_t b = 0; b < n; ++b) {
          Word total;
          for (std::size_t a = 0; a <= b; ++a) {
            const Word& coefficient = values[base + a * inner];
            if (!used[a] || coefficient.zero()) continue;
            Word product;
            if (!phx::mc::checked_multiply(coefficient, weights[b * n + a], product) ||
                !phx::mc::checked_add(total, product, total)) {
              return PHX_MC_RANGE_ERROR;
            }
          }
          next[base + b * inner] = total;
        }
      }
    }
    values.swap(next);
  }
  if (total_denominator > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
    return PHX_MC_RANGE_ERROR;
  }
  denominator = static_cast<int64_t>(total_denominator);
  return PHX_MC_OK;
}

template <class Word>
int32_t controls_item(const ControlsCall& call, int64_t item, WeightTable& table,
                      std::vector<Word>& values) {
  int64_t denominator = 1;
  if (call.domain == PHX_MC_BERNSTEIN_BOX) {
    const int32_t status = box_controls(call, item, values, denominator);
    if (status != PHX_MC_OK) return status;
  } else {
    const int32_t built = table.build(call, item);
    if (built != PHX_MC_OK) return built;
    denominator = table.denominator();
    const int64_t term_begin = call.term_offsets[item];
    const std::size_t terms = static_cast<std::size_t>(call.term_offsets[item + 1] - term_begin);
    const std::size_t controls =
        static_cast<std::size_t>(call.control_offsets[item + 1] - call.control_offsets[item]);
    values.assign(controls, Word{});
    std::vector<Word> numerators(terms);
    for (std::size_t term = 0; term < terms; ++term) {
      numerators[term] = Word::load(call.numerators +
                                    (static_cast<std::size_t>(term_begin) + term) * Word::kBytes);
    }
    for (std::size_t control = 0; control < controls; ++control) {
      Word total;
      for (std::size_t term = 0; term < terms; ++term) {
        if (numerators[term].zero()) continue;
        std::uint64_t weight = 0;
        if (!table.weight(call, item, control, term, weight)) return PHX_MC_RANGE_ERROR;
        if (weight == 0) continue;
        Word product;
        if (!phx::mc::checked_multiply(numerators[term], Word::from_uint64(weight), product) ||
            !phx::mc::checked_add(total, product, total)) {
          return PHX_MC_RANGE_ERROR;
        }
      }
      values[control] = total;
    }
  }
  std::uint8_t* out =
      call.controls + static_cast<std::size_t>(call.control_offsets[item]) * Word::kBytes;
  for (std::size_t control = 0; control < values.size(); ++control) {
    values[control].store(out + control * Word::kBytes);
  }
  call.denominators[item] = denominator;
  record_extremes(values, call.extremes + 2 * item);
  return PHX_MC_OK;
}

// --------------------------------------------------------- simplex blossoms

// Composition rows of every degree 0..n with the successor index tables:
// next[r][row * parts + p] is the row of composition(r - 1)[row] + e_p in degree r.
struct CompositionLadder {
  std::vector<std::vector<int32_t>> rows;
  std::vector<std::vector<int32_t>> next;

  CompositionLadder(int32_t degree, int32_t parts) {
    rows.resize(static_cast<std::size_t>(degree) + 1);
    next.resize(static_cast<std::size_t>(degree) + 1);
    std::vector<std::map<std::vector<int32_t>, int32_t>> rank(static_cast<std::size_t>(degree) + 1);
    for (int32_t r = 0; r <= degree; ++r) {
      rows[static_cast<std::size_t>(r)] = compositions(r, parts);
      const auto& level = rows[static_cast<std::size_t>(r)];
      const std::size_t count = level.size() / static_cast<std::size_t>(parts);
      for (std::size_t row = 0; row < count; ++row) {
        rank[static_cast<std::size_t>(r)].emplace(
            std::vector<int32_t>(level.begin() + static_cast<std::ptrdiff_t>(row * parts),
                                 level.begin() + static_cast<std::ptrdiff_t>((row + 1) * parts)),
            static_cast<int32_t>(row));
      }
      if (r == 0) continue;
      const auto& lower = rows[static_cast<std::size_t>(r) - 1];
      const std::size_t lower_count = lower.size() / static_cast<std::size_t>(parts);
      auto& table = next[static_cast<std::size_t>(r)];
      table.resize(lower_count * static_cast<std::size_t>(parts));
      for (std::size_t row = 0; row < lower_count; ++row) {
        std::vector<int32_t> key(lower.begin() + static_cast<std::ptrdiff_t>(row * parts),
                                 lower.begin() + static_cast<std::ptrdiff_t>((row + 1) * parts));
        for (int32_t part = 0; part < parts; ++part) {
          ++key[static_cast<std::size_t>(part)];
          table[row * static_cast<std::size_t>(parts) + static_cast<std::size_t>(part)] =
              rank[static_cast<std::size_t>(r)].at(key);
          --key[static_cast<std::size_t>(part)];
        }
      }
    }
    final_rank = std::move(rank[static_cast<std::size_t>(degree)]);
  }

  std::map<std::vector<int32_t>, int32_t> final_rank;
};

template <class Word>
class BlossomWalk final {
 public:
  BlossomWalk(const CompositionLadder& ladder, int32_t parts, const std::vector<Word>& corners,
              std::vector<Word>& values)
      : ladder_(ladder), parts_(parts), corners_(corners), values_(values),
        counts_(static_cast<std::size_t>(parts), 0) {}

  // Applies nondecreasing corner sequences depth first; false on overflow.
  bool walk(const std::vector<Word>& net, int32_t degree, int32_t last) {
    if (degree == 0) {
      values_[static_cast<std::size_t>(ladder_.final_rank.at(counts_))] = net[0];
      return true;
    }
    const auto& table = ladder_.next[static_cast<std::size_t>(degree)];
    const std::size_t rows = table.size() / static_cast<std::size_t>(parts_);
    std::vector<Word> following(rows);
    for (int32_t corner = last; corner < parts_; ++corner) {
      const Word* weights = corners_.data() + static_cast<std::size_t>(corner * parts_);
      for (std::size_t row = 0; row < rows; ++row) {
        Word total;
        for (int32_t part = 0; part < parts_; ++part) {
          if (weights[part].zero()) continue;
          Word product;
          if (!phx::mc::checked_multiply(
                  weights[part],
                  net[static_cast<std::size_t>(
                      table[row * static_cast<std::size_t>(parts_) + static_cast<std::size_t>(part)])],
                  product) ||
              !phx::mc::checked_add(total, product, total)) {
            return false;
          }
        }
        following[row] = total;
      }
      ++counts_[static_cast<std::size_t>(corner)];
      const bool ok = walk(following, degree - 1, corner);
      --counts_[static_cast<std::size_t>(corner)];
      if (!ok) return false;
    }
    return true;
  }

 private:
  const CompositionLadder& ladder_;
  int32_t parts_;
  const std::vector<Word>& corners_;
  std::vector<Word>& values_;
  std::vector<int32_t> counts_;
};

// ------------------------------------------------------------------ bisect

struct BisectLine {
  int32_t total;
  std::vector<int64_t> rows;  // ordered by the `second` index value 0..total
};

// --------------------------------------------------------- separation screen

// Outward enclosure of normal . x over the box [lower, upper].
void dot_enclosure(const double* normal, const double* lower, const double* upper,
                   int32_t dimension, double& low, double& high) noexcept {
  bool started = false;
  for (int32_t axis = 0; axis < dimension; ++axis) {
    const double n = normal[axis];
    if (n == 0.0) continue;
    const double first = n * lower[axis];
    const double second = n * upper[axis];
    const double product_low = next_down(std::min(first, second));
    const double product_high = next_up(std::max(first, second));
    if (!started) {
      low = product_low;
      high = product_high;
      started = true;
    } else {
      low = next_down(low + product_low);
      high = next_up(high + product_high);
    }
  }
  if (!started) {
    low = 0.0;
    high = 0.0;
  }
}

int8_t separation_decision(int32_t dimension, const double* normal, int64_t first_begin,
                           int64_t first_end, const double* first_lower, const double* first_upper,
                           int64_t second_begin, int64_t second_end, const double* second_lower,
                           const double* second_upper) noexcept {
  if (first_begin >= first_end || second_begin >= second_end) return PHX_MC_SEPARATION_UNDECIDED;
  double first_low_min = std::numeric_limits<double>::infinity();
  double first_high_min = std::numeric_limits<double>::infinity();
  for (int64_t point = first_begin; point < first_end; ++point) {
    double low = 0.0;
    double high = 0.0;
    dot_enclosure(normal, first_lower + point * dimension, first_upper + point * dimension,
                  dimension, low, high);
    if (std::isnan(low) || std::isnan(high)) return PHX_MC_SEPARATION_UNDECIDED;
    first_low_min = std::min(first_low_min, low);
    first_high_min = std::min(first_high_min, high);
  }
  double second_low_max = -std::numeric_limits<double>::infinity();
  double second_high_max = -std::numeric_limits<double>::infinity();
  for (int64_t point = second_begin; point < second_end; ++point) {
    double low = 0.0;
    double high = 0.0;
    dot_enclosure(normal, second_lower + point * dimension, second_upper + point * dimension,
                  dimension, low, high);
    if (std::isnan(low) || std::isnan(high)) return PHX_MC_SEPARATION_UNDECIDED;
    second_low_max = std::max(second_low_max, low);
    second_high_max = std::max(second_high_max, high);
  }
  if (second_high_max < first_low_min) return PHX_MC_SEPARATION_SEPARATED;
  if (second_low_max >= first_high_min) return PHX_MC_SEPARATION_NOT_SEPARATED;
  return PHX_MC_SEPARATION_UNDECIDED;
}

bool offsets_valid(int64_t count, const int64_t* offsets, int64_t& total) noexcept {
  if (offsets[0] != 0) return false;
  for (int64_t item = 0; item < count; ++item) {
    if (offsets[item + 1] < offsets[item]) return false;
  }
  total = offsets[count];
  return true;
}

}  // namespace

// ------------------------------------------------------------- BVH pair stream

struct phx_mc_bvh_pair_stream {
  phx_mc_bvh_tree_view first;
  phx_mc_bvh_tree_view second;
  int32_t dimension;
  bool include_touching;
  double absolute_tolerance;
  double relative_tolerance;
  std::size_t leaf_pair_block;
  std::size_t node_block;
  int32_t parallelism;

  int64_t required_capacity = 0;
  int32_t failure = PHX_MC_OK;
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> pending;
  bool active = false;
  std::vector<int64_t> nodes_first;
  std::vector<int64_t> nodes_second;
  std::vector<int64_t> leaves_first;
  std::vector<int64_t> leaves_second;
  std::size_t leaf_offset = 0;
  std::vector<int64_t> yield_first;
  std::vector<int64_t> yield_second;
  bool has_yield = false;
  std::vector<std::uint8_t> flags;

  bool overlap(const double* a_min, const double* a_max, const double* b_min,
               const double* b_max) const noexcept {
    for (int32_t axis = 0; axis < dimension; ++axis) {
      const double extent = ieee_minimum(a_max[axis], b_max[axis]) -
                            ieee_maximum(a_min[axis], b_min[axis]);
      if (!include_touching) {
        if (!(extent > 0.0)) return false;
        continue;
      }
      const double scale = ieee_maximum(
          1.0, ieee_maximum(std::fabs(a_min[axis]),
                            ieee_maximum(std::fabs(a_max[axis]),
                                         ieee_maximum(std::fabs(b_min[axis]),
                                                      std::fabs(b_max[axis])))));
      if (!(extent >= -(absolute_tolerance + relative_tolerance * scale))) return false;
    }
    return true;
  }

  bool node_overlap(int64_t a, int64_t b) const noexcept {
    return overlap(first.node_min + a * dimension, first.node_max + a * dimension,
                   second.node_min + b * dimension, second.node_max + b * dimension);
  }

  bool item_overlap(int64_t a, int64_t b) const noexcept {
    return overlap(first.item_min + a * dimension, first.item_max + a * dimension,
                   second.item_min + b * dimension, second.item_max + b * dimension);
  }

  void start() {
    if (node_overlap(0, 0)) pending.emplace_back(std::vector<int64_t>{0}, std::vector<int64_t>{0});
  }

  // Tests one leaf-pair chunk; returns whether it produced a block.
  bool test_chunk(std::size_t begin, std::size_t end) {
    const std::size_t width_a = static_cast<std::size_t>(first.leaf_width);
    const std::size_t width_b = static_cast<std::size_t>(second.leaf_width);
    const std::size_t per_leaf = width_a * width_b;
    const std::size_t candidates = (end - begin) * per_leaf;
    flags.assign(candidates, 0);
    const auto body = [&](std::size_t lo, std::size_t hi, phx::mc::ChunkContext&) {
      const phx::mc::GradualNearestScope mode;
      for (std::size_t slot = lo; slot < hi; ++slot) {
        const std::size_t pair = begin + slot / per_leaf;
        const std::size_t i = (slot % per_leaf) / width_b;
        const std::size_t j = slot % width_b;
        const int64_t a = first.leaf_items[leaves_first[pair] * first.leaf_width + static_cast<int64_t>(i)];
        const int64_t b =
            second.leaf_items[leaves_second[pair] * second.leaf_width + static_cast<int64_t>(j)];
        flags[slot] = static_cast<std::uint8_t>(a >= 0 && b >= 0 && item_overlap(a, b));
      }
    };
    constexpr std::size_t kParallelPairs = 1 << 14;
    const int32_t status = phx::mc::parallel_for(
        candidates, request(candidates >= kParallelPairs ? parallelism : 1, 4096), body);
    if (status != PHX_MC_OK) throw std::runtime_error("bvh pair chunk failed");
    yield_first.clear();
    yield_second.clear();
    for (std::size_t slot = 0; slot < candidates; ++slot) {
      if (!flags[slot]) continue;
      const std::size_t pair = begin + slot / per_leaf;
      const std::size_t i = (slot % per_leaf) / width_b;
      const std::size_t j = slot % width_b;
      yield_first.push_back(first.leaf_items[leaves_first[pair] * first.leaf_width + static_cast<int64_t>(i)]);
      yield_second.push_back(
          second.leaf_items[leaves_second[pair] * second.leaf_width + static_cast<int64_t>(j)]);
    }
    return !yield_first.empty();
  }

  void descend() {
    std::vector<int64_t> child_first;
    std::vector<int64_t> child_second;
    std::vector<int64_t> repeat_first;
    std::vector<int64_t> repeat_second_parents;
    std::vector<int64_t> repeated_second;
    std::vector<int64_t> second_children;
    for (std::size_t k = 0; k < nodes_first.size(); ++k) {
      const int64_t a = nodes_first[k];
      const int64_t b = nodes_second[k];
      const bool leaf_a = first.leaf[a] >= 0;
      const bool leaf_b = second.leaf[b] >= 0;
      const bool down_first = !leaf_a && (leaf_b || first.node_size[a] >= second.node_size[b]);
      const bool down_second = !(leaf_a && leaf_b) && !down_first;
      if (down_first) {
        child_first.push_back(first.left[a]);
        child_first.push_back(first.right[a]);
        repeated_second.push_back(b);
        repeated_second.push_back(b);
      } else if (down_second) {
        repeat_first.push_back(a);
        repeat_first.push_back(a);
        second_children.push_back(second.left[b]);
        second_children.push_back(second.right[b]);
      }
    }
    child_first.insert(child_first.end(), repeat_first.begin(), repeat_first.end());
    child_second = std::move(repeated_second);
    child_second.insert(child_second.end(), second_children.begin(), second_children.end());
    std::vector<int64_t> keep_first;
    std::vector<int64_t> keep_second;
    for (std::size_t k = 0; k < child_first.size(); ++k) {
      if (node_overlap(child_first[k], child_second[k])) {
        keep_first.push_back(child_first[k]);
        keep_second.push_back(child_second[k]);
      }
    }
    if (!keep_first.empty()) pending.emplace_back(std::move(keep_first), std::move(keep_second));
  }

  // Advances to the next block; false when the traversal is complete.
  bool advance() {
    const phx::mc::GradualNearestScope mode;
    while (true) {
      if (active) {
        if (leaf_offset < leaves_first.size()) {
          const std::size_t begin = leaf_offset;
          const std::size_t end = begin + std::min(leaves_first.size() - begin, leaf_pair_block);
          const bool produced = test_chunk(begin, end);
          leaf_offset = end;
          if (produced) return true;
          continue;
        }
        descend();
        active = false;
        continue;
      }
      if (pending.empty()) return false;
      auto block = std::move(pending.back());
      pending.pop_back();
      if (block.first.size() > node_block) {
        pending.emplace_back(
            std::vector<int64_t>(block.first.begin() + static_cast<std::ptrdiff_t>(node_block),
                                 block.first.end()),
            std::vector<int64_t>(block.second.begin() + static_cast<std::ptrdiff_t>(node_block),
                                 block.second.end()));
        block.first.resize(node_block);
        block.second.resize(node_block);
      }
      nodes_first = std::move(block.first);
      nodes_second = std::move(block.second);
      leaves_first.clear();
      leaves_second.clear();
      for (std::size_t k = 0; k < nodes_first.size(); ++k) {
        const int64_t leaf_a = first.leaf[nodes_first[k]];
        const int64_t leaf_b = second.leaf[nodes_second[k]];
        if (leaf_a >= 0 && leaf_b >= 0) {
          leaves_first.push_back(leaf_a);
          leaves_second.push_back(leaf_b);
        }
      }
      leaf_offset = 0;
      active = true;
    }
  }
};

namespace {

bool tree_valid(const phx_mc_bvh_tree_view* tree, int32_t dimension) {
  if (tree == nullptr || tree->node_count < 1 || tree->leaf_count < 0 || tree->leaf_width < 1 ||
      tree->item_count < 0 || tree->node_min == nullptr || tree->node_max == nullptr ||
      tree->node_size == nullptr || tree->left == nullptr || tree->right == nullptr ||
      tree->leaf == nullptr || (tree->leaf_count > 0 && tree->leaf_items == nullptr) ||
      (tree->item_count > 0 && (tree->item_min == nullptr || tree->item_max == nullptr)) ||
      !phx::mc::addressable(tree->node_count, dimension, sizeof(double)) ||
      !phx::mc::addressable(tree->leaf_count, tree->leaf_width, sizeof(int64_t)) ||
      !phx::mc::addressable(tree->item_count, dimension, sizeof(double))) {
    return false;
  }
  for (int64_t node = 0; node < tree->node_count; ++node) {
    const int64_t leaf = tree->leaf[node];
    if (leaf >= tree->leaf_count) return false;
    if (leaf < 0 && (tree->left[node] < 0 || tree->left[node] >= tree->node_count ||
                     tree->right[node] < 0 || tree->right[node] >= tree->node_count)) {
      return false;
    }
  }
  // A rooted tree has exactly one incoming edge per reachable non-root node.
  // Mark on discovery: this rejects cycles and shared children without recursion.
  std::vector<uint8_t> seen(static_cast<std::size_t>(tree->node_count), 0);
  std::vector<int64_t> pending{0};
  seen[0] = 1;
  while (!pending.empty()) {
    const int64_t node = pending.back();
    pending.pop_back();
    if (tree->leaf[node] >= 0) {
      if (tree->left[node] != -1 || tree->right[node] != -1) return false;
      continue;
    }
    for (const int64_t child : {tree->left[node], tree->right[node]}) {
      if (seen[static_cast<std::size_t>(child)]) return false;
      seen[static_cast<std::size_t>(child)] = 1;
      pending.push_back(child);
    }
  }
  const int64_t entries = tree->leaf_count * tree->leaf_width;
  for (int64_t entry = 0; entry < entries; ++entry) {
    if (tree->leaf_items[entry] >= tree->item_count) return false;
  }
  return true;
}

}  // namespace

extern "C" {

int32_t phx_mc_interval_bernstein_restrict(int64_t count, int32_t degree, int64_t width,
                                           const double* lower, const double* upper,
                                           const double* parameters, const int32_t* sides,
                                           double* out_lower, double* out_upper,
                                           int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    int64_t row_entries = 0;
    if (count < 0 || degree < 0 || width < 1 || parallelism < 1 ||
        !phx::mc::checked_multiply(int64_t{degree} + 1, width, row_entries) ||
        !phx::mc::addressable(1, row_entries, sizeof(double)) ||
        !phx::mc::addressable(count, row_entries, sizeof(double)) ||
        !phx::mc::addressable(count, 4, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (lower == nullptr || upper == nullptr || parameters == nullptr ||
                      sides == nullptr || out_lower == nullptr || out_upper == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const std::size_t entries = static_cast<std::size_t>(row_entries);
    return phx::mc::parallel_for(
        static_cast<std::size_t>(count), request(parallelism),
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          const phx::mc::GradualNearestScope mode;
          if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
          std::vector<double> work_lower;
          std::vector<double> work_upper;
          for (std::size_t item = begin; item < end; ++item) {
            interval_restrict_item(static_cast<std::size_t>(degree), static_cast<std::size_t>(width),
                                   lower + item * entries, upper + item * entries,
                                   parameters + 4 * item, sides[item], out_lower + item * entries,
                                   out_upper + item * entries, work_lower, work_upper);
          }
          return PHX_MC_OK;
        });
  });
}

int32_t phx_mc_interval_multiply(int64_t count, const double* first_lower,
                                 const double* first_upper, const double* second_lower,
                                 const double* second_upper, double* lower, double* upper) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || !phx::mc::addressable(count, 1, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (first_lower == nullptr || first_upper == nullptr ||
                      second_lower == nullptr || second_upper == nullptr || lower == nullptr ||
                      upper == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    for (int64_t index = 0; index < count; ++index) {
      interval_product(first_lower[index], first_upper[index], second_lower[index],
                       second_upper[index], lower[index], upper[index]);
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_interval_sum(int64_t count, const double* first_lower, const double* first_upper,
                            const double* second_lower, const double* second_upper,
                            int32_t subtract, double* lower, double* upper) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || (subtract != 0 && subtract != 1) ||
        !phx::mc::addressable(count, 1, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (first_lower == nullptr || first_upper == nullptr ||
                      second_lower == nullptr || second_upper == nullptr || lower == nullptr ||
                      upper == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    for (int64_t index = 0; index < count; ++index) {
      const double a_lo = first_lower[index];
      const double a_hi = first_upper[index];
      // Subtraction adds the negated, swapped second interval.
      const double b_lo = subtract ? -second_upper[index] : second_lower[index];
      const double b_hi = subtract ? -second_lower[index] : second_upper[index];
      const double low = subtract ? a_lo - second_upper[index] : a_lo + b_lo;
      const double high = subtract ? a_hi - second_lower[index] : a_hi + b_hi;
      double out_low = 0.0;
      double out_high = 0.0;
      round_out_inexact(low, high, sum_is_exact(a_lo, b_lo, low), sum_is_exact(a_hi, b_hi, high),
                        out_low, out_high);
      if (!subtract && !std::isnan(low) && !std::isnan(high)) {
        // A sum of sign-fixed operands keeps that sign across outward rounding.
        if (a_lo >= 0.0 && b_lo >= 0.0) out_low = ieee_maximum(out_low, 0.0);
        if (a_hi <= 0.0 && b_hi <= 0.0) out_high = ieee_minimum(out_high, 0.0);
      }
      lower[index] = out_low;
      upper[index] = out_high;
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_bernstein_controls(int64_t count, int32_t domain, int32_t dimension,
                                  const int32_t* degrees, const int64_t* term_offsets,
                                  const int32_t* exponents, const uint8_t* numerators,
                                  const int64_t* control_offsets, uint8_t* controls,
                                  int64_t* denominators, int64_t* extremes, int32_t* item_status,
                                  int32_t word_limbs, int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!word_limbs_valid(word_limbs)) return PHX_MC_INVALID_ARGUMENT;
    const int64_t word_bytes = int64_t{8} * word_limbs;
    int32_t degree_width = 0;
    switch (domain) {
      case PHX_MC_BERNSTEIN_SIMPLEX:
        degree_width = 1;
        break;
      case PHX_MC_BERNSTEIN_BOX:
        degree_width = dimension;
        break;
      case PHX_MC_BERNSTEIN_PRISM:
        if (dimension != 3) return PHX_MC_INVALID_ARGUMENT;
        degree_width = 2;
        break;
      default:
        return PHX_MC_INVALID_ARGUMENT;
    }
    if (count < 0 || dimension < 1 ||
        (domain == PHX_MC_BERNSTEIN_SIMPLEX && dimension == std::numeric_limits<int32_t>::max()) ||
        parallelism < 1 || term_offsets == nullptr || control_offsets == nullptr ||
        !phx::mc::addressable(count, degree_width, sizeof(int32_t)) ||
        !phx::mc::addressable(count, 2, sizeof(int64_t)) ||
        count == std::numeric_limits<int64_t>::max() ||
        !phx::mc::addressable(count + 1, 1, sizeof(int64_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (degrees == nullptr || denominators == nullptr || extremes == nullptr ||
                      item_status == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    int64_t terms = 0;
    int64_t total_controls = 0;
    if (!offsets_valid(count, term_offsets, terms) ||
        !offsets_valid(count, control_offsets, total_controls) ||
        !phx::mc::addressable(terms, dimension, sizeof(int32_t)) ||
        !phx::mc::addressable(terms, word_bytes, 1) ||
        !phx::mc::addressable(total_controls, word_bytes, 1) ||
        (terms > 0 && (exponents == nullptr || numerators == nullptr)) ||
        (total_controls > 0 && controls == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (int64_t item = 0; item < count; ++item) {
      if (domain_control_count(domain, dimension, degrees + item * degree_width) !=
          control_offsets[item + 1] - control_offsets[item]) {
        return PHX_MC_INVALID_ARGUMENT;
      }
    }
    const ControlsCall call{domain,       dimension,       degree_width, degrees,
                            term_offsets, exponents,       numerators,   control_offsets,
                            controls,     denominators,    extremes};
    return with_word(word_limbs, [&](auto tag) -> int32_t {
      using Word = decltype(tag);
      return phx::mc::parallel_for(
        static_cast<std::size_t>(count), request(parallelism),
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          WeightTable table;
          std::vector<Word> values;
          for (std::size_t item = begin; item < end; ++item) {
            item_status[item] = controls_item(call, static_cast<int64_t>(item), table, values);
          }
          return PHX_MC_OK;
        });
    });
  });
}

int32_t phx_mc_simplex_bernstein_blossoms(int64_t count, int32_t parts, const int32_t* degrees,
                                          const int64_t* control_offsets, const uint8_t* controls,
                                          const uint8_t* corners, uint8_t* values,
                                          int64_t* extremes, int32_t* item_status,
                                          int32_t word_limbs, int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!word_limbs_valid(word_limbs)) return PHX_MC_INVALID_ARGUMENT;
    const int64_t word_bytes = int64_t{8} * word_limbs;
    int64_t corner_bytes = 0;
    if (count < 0 || parts < 1 || parallelism < 1 || control_offsets == nullptr ||
        !phx::mc::checked_multiply(int64_t{parts} * parts, word_bytes, corner_bytes) ||
        !phx::mc::addressable(1, corner_bytes, 1) ||
        !phx::mc::addressable(count, corner_bytes, 1) ||
        !phx::mc::addressable(count, 2, sizeof(int64_t)) ||
        count == std::numeric_limits<int64_t>::max() ||
        !phx::mc::addressable(count + 1, 1, sizeof(int64_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (degrees == nullptr || corners == nullptr || extremes == nullptr ||
                      item_status == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    int64_t total_controls = 0;
    if (!offsets_valid(count, control_offsets, total_controls) ||
        !phx::mc::addressable(total_controls, word_bytes, 1) ||
        (total_controls > 0 && (controls == nullptr || values == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (int64_t item = 0; item < count; ++item) {
      if (degrees[item] < 0 ||
          composition_count(degrees[item], parts) != control_offsets[item + 1] - control_offsets[item]) {
        return PHX_MC_INVALID_ARGUMENT;
      }
    }
    const std::size_t corner_words = static_cast<std::size_t>(parts) * static_cast<std::size_t>(parts);
    return with_word(word_limbs, [&](auto tag) -> int32_t {
      using Word = decltype(tag);
      return phx::mc::parallel_for(
        static_cast<std::size_t>(count), request(parallelism),
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          std::map<int32_t, std::unique_ptr<CompositionLadder>> ladders;
          for (std::size_t item = begin; item < end; ++item) {
            const int32_t degree = degrees[item];
            auto& ladder = ladders[degree];
            if (!ladder) ladder = std::make_unique<CompositionLadder>(degree, parts);
            const std::size_t first = static_cast<std::size_t>(control_offsets[item]);
            const std::size_t size = static_cast<std::size_t>(control_offsets[item + 1]) - first;
            std::vector<Word> net(size);
            for (std::size_t row = 0; row < size; ++row) {
              net[row] = Word::load(controls + (first + row) * Word::kBytes);
            }
            std::vector<Word> weights(corner_words);
            for (std::size_t word = 0; word < corner_words; ++word) {
              weights[word] = Word::load(corners + (item * corner_words + word) * Word::kBytes);
            }
            std::vector<Word> result(size);
            BlossomWalk<Word> walk(*ladder, parts, weights, result);
            if (!walk.walk(net, degree, 0)) {
              item_status[item] = PHX_MC_RANGE_ERROR;
              continue;
            }
            for (std::size_t row = 0; row < size; ++row) {
              result[row].store(values + (first + row) * Word::kBytes);
            }
            record_extremes(result, extremes + 2 * item);
            item_status[item] = PHX_MC_OK;
          }
          return PHX_MC_OK;
        });
    });
  });
}

int32_t phx_mc_bernstein_bisect(int64_t count, int64_t control_count, int32_t slot_count,
                                int32_t components, const int32_t* indices, int32_t first,
                                int32_t second, const uint8_t* nets, uint8_t* left,
                                uint8_t* right, int32_t* exponent, int32_t* item_status,
                                int32_t word_limbs, int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!word_limbs_valid(word_limbs)) return PHX_MC_INVALID_ARGUMENT;
    const int64_t word_bytes = int64_t{8} * word_limbs;
    int64_t net_words_checked = 0;
    int64_t net_bytes = 0;
    if (count < 0 || control_count < 1 || slot_count < 2 || components < 1 || parallelism < 1 ||
        first < 0 || second < 0 || first >= slot_count || second >= slot_count ||
        first == second || indices == nullptr || exponent == nullptr ||
        !phx::mc::addressable(control_count, slot_count, sizeof(int32_t)) ||
        !phx::mc::checked_multiply(control_count, int64_t{components}, net_words_checked) ||
        !phx::mc::checked_multiply(net_words_checked, word_bytes, net_bytes) ||
        !phx::mc::addressable(1, net_bytes, 1) ||
        !phx::mc::addressable(count, net_bytes, 1)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (nets == nullptr || left == nullptr || right == nullptr ||
                      item_status == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    // Lines of controls sharing every slot except (first, second).
    std::map<std::vector<int32_t>, std::vector<int64_t>> grouped;
    for (int64_t row = 0; row < control_count; ++row) {
      std::vector<int32_t> rest;
      for (int32_t slot = 0; slot < slot_count; ++slot) {
        const int32_t value = indices[row * slot_count + slot];
        if (value < 0) return PHX_MC_INVALID_INPUT;
        if (slot != first && slot != second) rest.push_back(value);
      }
      grouped[rest].push_back(row);
    }
    std::vector<BisectLine> lines;
    int32_t maximum = 0;
    for (auto& [rest, rows] : grouped) {
      const int64_t wide_total = int64_t{indices[rows[0] * slot_count + first]} +
                                 indices[rows[0] * slot_count + second];
      if (wide_total > std::numeric_limits<int32_t>::max() ||
          static_cast<int64_t>(rows.size()) != wide_total + 1) return PHX_MC_INVALID_INPUT;
      const int32_t total = static_cast<int32_t>(wide_total);
      const int64_t size = wide_total + 1;
      if (!phx::mc::addressable(size * (size + 1) / 2, word_bytes, 1)) {
        return PHX_MC_INVALID_ARGUMENT;
      }
      BisectLine line{total, std::vector<int64_t>(rows.size(), -1)};
      for (const int64_t row : rows) {
        const int32_t along = indices[row * slot_count + second];
        if (int64_t{indices[row * slot_count + first]} + along != wide_total ||
            line.rows[static_cast<std::size_t>(along)] != -1) {
          return PHX_MC_INVALID_INPUT;
        }
        line.rows[static_cast<std::size_t>(along)] = row;
      }
      maximum = std::max(maximum, total);
      lines.push_back(std::move(line));
    }
    *exponent = maximum;
    const std::size_t row_words = static_cast<std::size_t>(components);
    const std::size_t net_words = static_cast<std::size_t>(net_words_checked);
    return with_word(word_limbs, [&](auto tag) -> int32_t {
      using Word = decltype(tag);
      return phx::mc::parallel_for(
        static_cast<std::size_t>(count), request(parallelism),
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          std::vector<Word> levels;
          for (std::size_t item = begin; item < end; ++item) {
            const std::uint8_t* net = nets + item * net_words * Word::kBytes;
            std::uint8_t* out_left = left + item * net_words * Word::kBytes;
            std::uint8_t* out_right = right + item * net_words * Word::kBytes;
            int32_t status = PHX_MC_OK;
            for (const BisectLine& line : lines) {
              const std::size_t size = static_cast<std::size_t>(line.total) + 1;
              for (std::size_t component = 0; component < row_words && status == PHX_MC_OK;
                   ++component) {
                // levels[j][r] (unhalved sums) stored row-major by level.
                levels.assign(size * (size + 1) / 2, Word{});
                const auto at = [size](std::size_t level, std::size_t row) {
                  return level * size - level * (level - 1) / 2 + row;
                };
                for (std::size_t row = 0; row < size; ++row) {
                  levels[at(0, row)] = Word::load(
                      net + (static_cast<std::size_t>(line.rows[row]) * row_words + component) *
                                Word::kBytes);
                }
                for (std::size_t level = 1; level < size && status == PHX_MC_OK; ++level) {
                  for (std::size_t row = 0; row + level < size; ++row) {
                    if (!phx::mc::checked_add(levels[at(level - 1, row)],
                                              levels[at(level - 1, row + 1)],
                                              levels[at(level, row)])) {
                      status = PHX_MC_RANGE_ERROR;
                      break;
                    }
                  }
                }
                for (std::size_t step = 0; step < size && status == PHX_MC_OK; ++step) {
                  Word kept;
                  Word moved;
                  const std::size_t complement = size - 1 - step;
                  if (!phx::mc::checked_shift_left(levels[at(step, 0)],
                                                   maximum - static_cast<int32_t>(step), kept) ||
                      !phx::mc::checked_shift_left(levels[at(complement, step)],
                                                   maximum - static_cast<int32_t>(complement),
                                                   moved)) {
                    status = PHX_MC_RANGE_ERROR;
                    break;
                  }
                  const std::size_t offset =
                      (static_cast<std::size_t>(line.rows[step]) * row_words + component) *
                      Word::kBytes;
                  kept.store(out_left + offset);
                  moved.store(out_right + offset);
                }
              }
              if (status != PHX_MC_OK) break;
            }
            item_status[item] = status;
          }
          return PHX_MC_OK;
        });
    });
  });
}

int32_t phx_mc_separation_screen(int64_t count, int32_t dimension, const int64_t* first_offsets,
                                 const double* first_lower, const double* first_upper,
                                 const int64_t* second_offsets, const double* second_lower,
                                 const double* second_upper, const double* normals,
                                 int8_t* decisions, int32_t parallelism) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || dimension < 1 || parallelism < 1 || first_offsets == nullptr ||
        second_offsets == nullptr || !phx::mc::addressable(count, dimension, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    int64_t first_points = 0;
    int64_t second_points = 0;
    if (!offsets_valid(count, first_offsets, first_points) ||
        !offsets_valid(count, second_offsets, second_points) ||
        !phx::mc::addressable(first_points, dimension, sizeof(double)) ||
        !phx::mc::addressable(second_points, dimension, sizeof(double)) ||
        (count > 0 && (normals == nullptr || decisions == nullptr)) ||
        (first_points > 0 && (first_lower == nullptr || first_upper == nullptr)) ||
        (second_points > 0 && (second_lower == nullptr || second_upper == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    return phx::mc::parallel_for(
        static_cast<std::size_t>(count), request(parallelism, 64),
        [&](std::size_t begin, std::size_t end, phx::mc::ChunkContext&) -> int32_t {
          const phx::mc::GradualNearestScope mode;
          if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
          for (std::size_t item = begin; item < end; ++item) {
            decisions[item] = separation_decision(
                dimension, normals + item * static_cast<std::size_t>(dimension), first_offsets[item],
                first_offsets[item + 1], first_lower, first_upper, second_offsets[item],
                second_offsets[item + 1], second_lower, second_upper);
          }
          return PHX_MC_OK;
        });
  });
}

int32_t phx_mc_bvh_pair_stream_create(const phx_mc_bvh_tree_view* first,
                                      const phx_mc_bvh_tree_view* second, int32_t dimension,
                                      int32_t include_touching, double absolute_tolerance,
                                      double relative_tolerance, int64_t item_block,
                                      int64_t node_block, int32_t parallelism,
                                      phx_mc_bvh_pair_stream** stream) {
  return phx::mc::guarded([&]() -> int32_t {
    if (stream == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *stream = nullptr;
    if (dimension < 1 || item_block < 1 || node_block < 1 || parallelism < 1 ||
        (include_touching != 0 && include_touching != 1) || !std::isfinite(absolute_tolerance) ||
        !std::isfinite(relative_tolerance) || absolute_tolerance < 0.0 ||
        relative_tolerance < 0.0 || !tree_valid(first, dimension) ||
        !tree_valid(second, dimension)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    int64_t per_leaf = 0;
    if (!phx::mc::checked_multiply(first->leaf_width, second->leaf_width, per_leaf) ||
        !phx::mc::addressable(1, per_leaf, sizeof(int64_t)) ||
        !phx::mc::addressable(node_block, 2, sizeof(int64_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    auto created = std::make_unique<phx_mc_bvh_pair_stream>();
    created->first = *first;
    created->second = *second;
    created->dimension = dimension;
    created->include_touching = include_touching != 0;
    created->absolute_tolerance = absolute_tolerance;
    created->relative_tolerance = relative_tolerance;
    created->leaf_pair_block = static_cast<std::size_t>(
        std::max<int64_t>(1, item_block / per_leaf));
    created->required_capacity = static_cast<int64_t>(created->leaf_pair_block) * per_leaf;
    if (!phx::mc::addressable(1, created->required_capacity, sizeof(int64_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    created->node_block = static_cast<std::size_t>(node_block);
    created->parallelism = parallelism;
    {
      const phx::mc::GradualNearestScope mode;
      if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
      created->start();
    }
    *stream = created.release();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_bvh_pair_stream_next(phx_mc_bvh_pair_stream* stream, int64_t pair_capacity,
                                    int64_t block_capacity, int64_t* first_items,
                                    int64_t* second_items, int64_t* block_ends, int64_t* blocks) {
  bool advancing = false;
  const int32_t status = phx::mc::guarded([&]() -> int32_t {
    if (stream == nullptr || blocks == nullptr || block_capacity < 1 || first_items == nullptr ||
        second_items == nullptr || block_ends == nullptr ||
        !phx::mc::addressable(pair_capacity, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(block_capacity, 1, sizeof(int64_t)) ||
        pair_capacity < stream->required_capacity) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (stream->failure != PHX_MC_OK) return stream->failure;
    advancing = true;
    *blocks = 0;
    int64_t used = 0;
    while (*blocks < block_capacity) {
      if (!stream->has_yield) {
        if (!stream->advance()) break;
        stream->has_yield = true;
      }
      const int64_t size = static_cast<int64_t>(stream->yield_first.size());
      if (size > pair_capacity - used) break;
      std::copy(stream->yield_first.begin(), stream->yield_first.end(), first_items + used);
      std::copy(stream->yield_second.begin(), stream->yield_second.end(), second_items + used);
      used += size;
      block_ends[(*blocks)++] = used;
      stream->has_yield = false;
    }
    return PHX_MC_OK;
  });
  // Traversal scratch may already have changed when an allocation throws.
  // Never allow a subsequent call to silently complete a truncated traversal.
  if (advancing && status != PHX_MC_OK) {
    stream->failure = status;
  }
  return status;
}

void phx_mc_bvh_pair_stream_free(phx_mc_bvh_pair_stream* stream) { delete stream; }

}  // extern "C"
