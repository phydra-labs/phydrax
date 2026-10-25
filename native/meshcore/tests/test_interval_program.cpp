// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Interval program execution against the scalar instruction contract restated
// independently here (bit-level adjacent steps, NumPy minimum/maximum), exact
// dyadic enclosure and nearest-rounding truth, hand-derived special values,
// fail-closed validation that leaves both workspaces untouched, zero-work
// execution-scope checkpoints, and callers left in flush-to-zero or directed
// modes.
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include "check.hpp"
#include "dyadic.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

namespace {

using phx::mc::Dyadic;

constexpr int64_t kGather = 1;
constexpr int64_t kAdd = 2;
constexpr int64_t kSubtract = 3;
constexpr int64_t kSubtractSame = 4;
constexpr int64_t kMultiply = 5;
constexpr int64_t kNegate = 6;
constexpr int64_t kReduceSum = 7;
constexpr int64_t kSine = 8;
constexpr int64_t kCosine = 9;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kLargest = std::numeric_limits<double>::max();
constexpr double kSmallest = std::numeric_limits<double>::denorm_min();
constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
constexpr double kPi = 3.141592653589793;
constexpr double kTwoPi = 2.0 * kPi;
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
// A deadline that a scope meets on entry and misses after the sleep.
constexpr double kDeadlineSeconds = 0.05;
constexpr auto kPastDeadline = std::chrono::milliseconds(80);

uint64_t bits(double value) { return std::bit_cast<uint64_t>(value); }
double from_bits(uint64_t value) { return std::bit_cast<double>(value); }
bool same(double a, double b) { return bits(a) == bits(b); }

// ------------------------------------------------------ scalar contract

// Adjacent binary64 values by encoding, independent of libm nextafter.
double step_down(double x) {
  if (std::isnan(x) || x == -kInfinity) return x;
  if (x == 0.0) return -kSmallest;
  return from_bits(x > 0.0 ? bits(x) - 1 : bits(x) + 1);
}

double step_up(double x) {
  if (std::isnan(x) || x == kInfinity) return x;
  if (x == 0.0) return kSmallest;
  return from_bits(x > 0.0 ? bits(x) + 1 : bits(x) - 1);
}

double steps_down(double x, int count) {
  for (int i = 0; i < count; ++i) x = step_down(x);
  return x;
}

double steps_up(double x, int count) {
  for (int i = 0; i < count; ++i) x = step_up(x);
  return x;
}

double numpy_minimum(double a, double b) {
  if (a != a) return a;
  if (b != b) return b;
  if (a == b) return std::copysign(1.0, a) < 0.0 ? a : b;
  return a < b ? a : b;
}

double numpy_maximum(double a, double b) {
  if (a != a) return a;
  if (b != b) return b;
  if (a == b) return std::copysign(1.0, a) < 0.0 ? b : a;
  return a > b ? a : b;
}

double endpoint_product(double a, double b) {
  if ((a == 0.0 && std::isinf(b)) || (b == 0.0 && std::isinf(a))) return 0.0;
  return a * b;
}

bool phase_inside(double lower, double upper, double phase) {
  const double slack = 8.0 * kEpsilon * numpy_maximum(1.0, std::fabs(lower));
  const double index = std::ceil((lower - slack - phase) / kTwoPi);
  return phase + index * kTwoPi <= upper + slack;
}

std::array<double, 2> trig_contract(double lower, double upper, bool sine) {
  const bool wide = !(upper - lower < kTwoPi) || !std::isfinite(lower) || !std::isfinite(upper);
  auto f = [sine](double x) {
    if (!std::isfinite(x)) return kNaN;
    return sine ? std::sin(x) : std::cos(x);
  };
  const double first = f(lower);
  const double last = f(upper);
  double low = numpy_minimum(first, last);
  double high = numpy_maximum(first, last);
  if (phase_inside(lower, upper, sine ? 0.5 * kPi : 0.0) || wide) high = 1.0;
  if (phase_inside(lower, upper, sine ? -0.5 * kPi : kPi) || wide) low = -1.0;
  return {numpy_maximum(steps_down(low, 4), -1.0), numpy_minimum(steps_up(high, 4), 1.0)};
}

struct Program {
  std::vector<int64_t> rows;
  std::vector<int64_t> indices;
  std::vector<double> parameters;
  std::vector<double> lower;
  std::vector<double> upper;

  int64_t count() const { return static_cast<int64_t>(rows.size() / 8); }
  void row(int64_t opcode, int64_t out, int64_t size, int64_t a, int64_t a_index, int64_t b,
           int64_t b_index, int64_t aux) {
    rows.insert(rows.end(), {opcode, out, size, a, a_index, b, b_index, aux});
  }
  // Appends interval values and returns their offset.
  int64_t values(std::initializer_list<std::array<double, 2>> entries) {
    const auto offset = static_cast<int64_t>(lower.size());
    for (const auto& entry : entries) {
      lower.push_back(entry[0]);
      upper.push_back(entry[1]);
    }
    return offset;
  }
  // Appends size output slots holding a recognizable sentinel.
  int64_t slots(int64_t size) {
    const auto offset = static_cast<int64_t>(lower.size());
    lower.insert(lower.end(), static_cast<std::size_t>(size), 12345.0);
    upper.insert(upper.end(), static_cast<std::size_t>(size), -12345.0);
    return offset;
  }
  int64_t index(std::initializer_list<int64_t> entries) {
    const auto offset = static_cast<int64_t>(indices.size());
    indices.insert(indices.end(), entries);
    return offset;
  }
  int64_t parameter(double value) {
    parameters.push_back(value);
    return static_cast<int64_t>(parameters.size()) - 1;
  }
};

int32_t execute(Program& p) {
  return phx_mc_interval_program_execute(
      p.rows.data(), p.count(), p.indices.data(), static_cast<int64_t>(p.indices.size()),
      p.parameters.data(), static_cast<int64_t>(p.parameters.size()), p.lower.data(),
      p.upper.data(), static_cast<int64_t>(p.lower.size()));
}

// Executes the documented row semantics element by element on copies.
void reference(const Program& p, std::vector<double>& low, std::vector<double>& high) {
  low = p.lower;
  high = p.upper;
  for (int64_t r = 0; r < p.count(); ++r) {
    const int64_t* f = p.rows.data() + r * 8;
    const int64_t opcode = f[0], out = f[1], size = f[2], a = f[3], ai = f[4], b = f[5], bi = f[6],
                  aux = f[7];
    auto at = [&](int64_t offset, int64_t index, int64_t k) {
      return static_cast<std::size_t>(
          offset + (index < 0 ? k : p.indices[static_cast<std::size_t>(index + k)]));
    };
    for (int64_t k = 0; k < size; ++k) {
      const auto target = static_cast<std::size_t>(out + k);
      if (opcode == kReduceSum) {
        const int64_t base = a + k * aux;
        double ls = 0.0, hs = 0.0, scale = 0.0;
        for (int64_t t = 0; t < aux; ++t) {
          ls += low[static_cast<std::size_t>(base + t)];
          hs += high[static_cast<std::size_t>(base + t)];
        }
        for (int64_t t = 0; t < aux; ++t) {
          scale += numpy_maximum(std::fabs(low[static_cast<std::size_t>(base + t)]),
                                 std::fabs(high[static_cast<std::size_t>(base + t)]));
        }
        const double g = p.parameters[static_cast<std::size_t>(bi)];
        low[target] = step_down(ls - g * scale);
        high[target] = step_up(hs + g * scale);
        continue;
      }
      const std::size_t x = at(a, ai, k);
      const double al = low[x], ah = high[x];
      if (opcode == kGather) {
        low[target] = al;
        high[target] = ah;
      } else if (opcode == kNegate) {
        low[target] = -ah;
        high[target] = -al;
      } else if (opcode == kSine || opcode == kCosine) {
        const auto value = trig_contract(al, ah, opcode == kSine);
        low[target] = value[0];
        high[target] = value[1];
      } else {
        const std::size_t y = at(b, bi, k);
        const double bl = low[y], bh = high[y];
        if (opcode == kAdd) {
          low[target] = step_down(al + bl);
          high[target] = step_up(ah + bh);
        } else if (opcode == kMultiply) {
          const double ll = endpoint_product(al, bl), lu = endpoint_product(al, bh);
          const double ul = endpoint_product(ah, bl), uu = endpoint_product(ah, bh);
          low[target] = step_down(numpy_minimum(numpy_minimum(ll, lu), numpy_minimum(ul, uu)));
          high[target] = step_up(numpy_maximum(numpy_maximum(ll, lu), numpy_maximum(ul, uu)));
        } else {
          double rl = step_down(al - bh), rh = step_up(ah - bl);
          if (al == ah && bl == bh && al == bl && std::isfinite(al)) rl = rh = 0.0;
          if (opcode == kSubtractSame && std::isfinite(al) && std::isfinite(ah)) rl = rh = 0.0;
          low[target] = rl;
          high[target] = rh;
        }
      }
    }
  }
}

// Bitwise equality; with `any_nan`, NaN payloads are not compared.
bool identical(const std::vector<double>& a, const std::vector<double>& b, bool any_nan = false) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (any_nan && std::isnan(a[i]) && std::isnan(b[i])) continue;
    if (!same(a[i], b[i])) return false;
  }
  return true;
}

void check_against_reference(Program& p, bool any_nan = false) {
  std::vector<double> low, high;
  reference(p, low, high);
  PHX_CHECK(execute(p) == PHX_MC_OK);
  PHX_CHECK(identical(p.lower, low, any_nan));
  PHX_CHECK(identical(p.upper, high, any_nan));
}

// ------------------------------------------------- hand-derived contract

void additive_cases() {
  Program p;
  const int64_t a = p.values({{1.0, 2.0},
                              {kLargest, kLargest},
                              {-kInfinity, -kInfinity},
                              {kSmallest, kSmallest},
                              {-0.0, -0.0}});
  const int64_t b = p.values({{0x1p-60, 0x1p-60},
                              {kLargest, kLargest},
                              {-1.0, 1.0},
                              {kSmallest, kSmallest},
                              {-0.0, -0.0}});
  const int64_t out = p.slots(5);
  p.row(kAdd, out, 5, a, -1, b, -1, 0);
  PHX_CHECK(execute(p) == PHX_MC_OK);
  PHX_CHECK(same(p.lower[out], 1.0 - 0x1p-53) && same(p.upper[out], 2.0 + 0x1p-51));
  // Overflow: the lower sum is +inf and steps to the largest finite value.
  PHX_CHECK(same(p.lower[out + 1], kLargest) && same(p.upper[out + 1], kInfinity));
  PHX_CHECK(same(p.lower[out + 2], -kInfinity) && same(p.upper[out + 2], -kLargest));
  // Gradual underflow: 2 d is exact and steps one subnormal outward.
  PHX_CHECK(same(p.lower[out + 3], kSmallest) && same(p.upper[out + 3], 3 * kSmallest));
  PHX_CHECK(same(p.lower[out + 4], -kSmallest) && same(p.upper[out + 4], kSmallest));
}

void subtraction_cases() {
  Program p;
  const int64_t a = p.values({{3.0, 3.0}, {kInfinity, kInfinity}, {1.0, 2.0}, {-0.0, -0.0}});
  const int64_t b = p.values({{3.0, 3.0}, {kInfinity, kInfinity}, {0.5, 0.5}, {0.0, 0.0}});
  const int64_t out = p.slots(4);
  p.row(kSubtract, out, 4, a, -1, b, -1, 0);
  const int64_t same_input = p.values({{-1.0, 2.0}, {-kInfinity, 1.0}, {-0.0, -0.0}, {kNaN, 1.0}});
  const int64_t same_out = p.slots(4);
  p.row(kSubtractSame, same_out, 4, same_input, -1, same_input, -1, 0);
  PHX_CHECK(execute(p) == PHX_MC_OK);
  // A proved zero is +0 for equal finite singletons.
  PHX_CHECK(same(p.lower[out], 0.0) && same(p.upper[out], 0.0));
  PHX_CHECK(std::isnan(p.lower[out + 1]) && std::isnan(p.upper[out + 1]));
  PHX_CHECK(same(p.lower[out + 2], step_down(0.5)) && same(p.upper[out + 2], step_up(1.5)));
  // -0 - +0 = -0 is not a singleton pair: the equal values -0 == +0 prove zero.
  PHX_CHECK(same(p.lower[out + 3], 0.0) && same(p.upper[out + 3], 0.0));
  PHX_CHECK(same(p.lower[same_out], 0.0) && same(p.upper[same_out], 0.0));
  PHX_CHECK(same(p.lower[same_out + 1], -kInfinity) && same(p.upper[same_out + 1], kInfinity));
  PHX_CHECK(same(p.lower[same_out + 2], 0.0) && same(p.upper[same_out + 2], 0.0));
  PHX_CHECK(std::isnan(p.lower[same_out + 3]) && std::isnan(p.upper[same_out + 3]));
}

void multiplication_cases() {
  const double nan_a = from_bits(0x7ff8000000000123ULL);
  const double nan_b = from_bits(0x7ff8000000000456ULL);
  Program p;
  const int64_t a = p.values({{0.0, 0.0}, {-2.0, 3.0}, {nan_a, nan_b}, {1.0, 2.0}, {-1.0, 1.0}});
  const int64_t b =
      p.values({{-kInfinity, kInfinity}, {4.0, 5.0}, {2.0, 3.0}, {nan_b, 3.0}, {0.0, 0.0}});
  const int64_t out = p.slots(5);
  p.row(kMultiply, out, 5, a, -1, b, -1, 0);
  PHX_CHECK(execute(p) == PHX_MC_OK);
  // Zero times an infinite endpoint is +0.
  PHX_CHECK(same(p.lower[out], -kSmallest) && same(p.upper[out], kSmallest));
  PHX_CHECK(same(p.lower[out + 1], step_down(-10.0)) && same(p.upper[out + 1], step_up(15.0)));
  // NumPy minimum/maximum select the first NaN product: ll, which carries a.
  PHX_CHECK(bits(p.lower[out + 2]) == bits(nan_a) && bits(p.upper[out + 2]) == bits(nan_a));
  PHX_CHECK(bits(p.lower[out + 3]) == bits(nan_b) && bits(p.upper[out + 3]) == bits(nan_b));
  PHX_CHECK(same(p.lower[out + 4], -kSmallest) && same(p.upper[out + 4], kSmallest));
}

void structural_cases() {
  const double nan_a = from_bits(0x7ff8000000000789ULL);
  Program p;
  const int64_t a = p.values({{0.0, 0.0}, {-1.0, kInfinity}, {nan_a, -0.0}, {-0.0, 0.0}});
  const int64_t negated = p.slots(4);
  p.row(kNegate, negated, 4, a, -1, -1, -1, 0);
  const int64_t gathered = p.slots(5);
  p.row(kGather, gathered, 5, a, p.index({3, 2, 2, 0, 1}), -1, -1, 0);
  PHX_CHECK(execute(p) == PHX_MC_OK);
  PHX_CHECK(same(p.lower[negated], -0.0) && same(p.upper[negated], -0.0));
  PHX_CHECK(same(p.lower[negated + 1], -kInfinity) && same(p.upper[negated + 1], 1.0));
  PHX_CHECK(same(p.lower[negated + 2], 0.0));
  PHX_CHECK(bits(p.upper[negated + 2]) == (bits(nan_a) ^ (1ULL << 63)));
  const std::array<std::size_t, 5> order{3, 2, 2, 0, 1};
  for (std::size_t k = 0; k < order.size(); ++k) {
    PHX_CHECK(same(p.lower[gathered + k], p.lower[a + order[k]]));
    PHX_CHECK(same(p.upper[gathered + k], p.upper[a + order[k]]));
  }
}

void reduction_cases() {
  Program p;
  const double gamma = 3.0 * kEpsilon / (1.0 - 3.0 * kEpsilon);
  const int64_t terms =
      p.values({{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {-0.0, -0.0}, {-0.0, -0.0}, {-0.0, -0.0}});
  const int64_t out = p.slots(2);
  p.row(kReduceSum, out, 2, terms, -1, -1, p.parameter(gamma), 3);
  const int64_t empty = p.slots(2);
  p.row(kReduceSum, empty, 2, terms, -1, -1, p.parameter(1.0), 0);
  PHX_CHECK(execute(p) == PHX_MC_OK);
  PHX_CHECK(same(p.lower[out], step_down(6.0 - gamma * 6.0)));
  PHX_CHECK(same(p.upper[out], step_up(6.0 + gamma * 6.0)));
  PHX_CHECK(Dyadic(p.lower[out]).sign() > 0 && (Dyadic(6.0) - Dyadic(p.lower[out])).sign() > 0);
  PHX_CHECK((Dyadic(p.upper[out]) - Dyadic(6.0)).sign() > 0);
  // Sums start at +0; adjacent values of zero are the least subnormals.
  PHX_CHECK(same(p.lower[out + 1], -kSmallest) && same(p.upper[out + 1], kSmallest));
  PHX_CHECK(same(p.lower[empty], -kSmallest) && same(p.upper[empty], kSmallest));
  PHX_CHECK(same(p.lower[empty + 1], -kSmallest) && same(p.upper[empty + 1], kSmallest));
}

void trigonometric_cases() {
  Program p;
  const double below_two_pi = step_down(kTwoPi);
  const int64_t a = p.values({{0.0, 0.0},
                              {-kInfinity, kInfinity},
                              {kNaN, kNaN},
                              {1e300, 1e300},
                              {0.0, kTwoPi},
                              {0.0, below_two_pi},
                              {0.1, 0.2},
                              {kInfinity, kInfinity}});
  const int64_t sine = p.slots(8);
  p.row(kSine, sine, 8, a, -1, -1, -1, 0);
  const int64_t cosine = p.slots(8);
  p.row(kCosine, cosine, 8, a, -1, -1, -1, 0);
  std::vector<double> low, high;
  reference(p, low, high);
  PHX_CHECK(execute(p) == PHX_MC_OK);
  PHX_CHECK(identical(p.lower, low) && identical(p.upper, high));
  // sin [0, 0]: no phase inside; four steps from the exact zero.
  PHX_CHECK(same(p.lower[sine], -4 * kSmallest) && same(p.upper[sine], 4 * kSmallest));
  // cos [0, 0]: phase 0 lies inside, pi does not.
  PHX_CHECK(same(p.lower[cosine], 1.0 - 0x1p-51) && same(p.upper[cosine], 1.0));
  for (int64_t k : {1, 2, 3, 4, 5, 7}) {
    PHX_CHECK(same(p.lower[sine + k], -1.0) && same(p.upper[sine + k], 1.0));
    PHX_CHECK(same(p.lower[cosine + k], -1.0) && same(p.upper[cosine + k], 1.0));
  }
  PHX_CHECK(same(p.lower[sine + 6], steps_down(std::sin(0.1), 4)));
  PHX_CHECK(same(p.upper[sine + 6], steps_up(std::sin(0.2), 4)));
  PHX_CHECK(same(p.lower[cosine + 6], steps_down(std::cos(0.2), 4)));
  PHX_CHECK(same(p.upper[cosine + 6], steps_up(std::cos(0.1), 4)));
}

// Sampled trig enclosure, including phase neighborhoods and large arguments.
void trigonometric_enclosure() {
  std::mt19937_64 random(29);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int trial = 0; trial < 400; ++trial) {
    const double scale = std::ldexp(1.0, static_cast<int>(random() % 80) - 10);
    double lower = (2.0 * unit(random) - 1.0) * scale;
    if (trial % 5 == 0) lower = (static_cast<double>(random() % 9) - 4.0) * 0.5 * kPi;
    const int magnitude = -static_cast<int>(random() % 40);
    const double width = trial % 7 == 0 ? 0.0 : unit(random) * std::ldexp(1.0, magnitude);
    const double upper = lower + width;
    Program p;
    const int64_t a = p.values({{lower, upper}});
    const int64_t out = p.slots(2);
    p.row(kSine, out, 1, a, -1, -1, -1, 0);
    p.row(kCosine, out + 1, 1, a, -1, -1, -1, 0);
    check_against_reference(p);
    for (int sample = 0; sample <= 16; ++sample) {
      const double x = sample == 16 ? upper : lower + (upper - lower) * (sample / 16.0);
      if (x < lower || x > upper) continue;
      PHX_CHECK(p.lower[out] <= std::sin(x) && std::sin(x) <= p.upper[out]);
      PHX_CHECK(p.lower[out + 1] <= std::cos(x) && std::cos(x) <= p.upper[out + 1]);
    }
    PHX_CHECK(p.lower[out] >= -1.0 && p.upper[out] <= 1.0);
    PHX_CHECK(p.lower[out + 1] >= -1.0 && p.upper[out + 1] <= 1.0);
  }
}

// -------------------------------------------------------- dyadic truth

int order(const Dyadic& a, const Dyadic& b) { return (a - b).sign(); }

// Whether value is a binary64 nearest to exact: no neighbor is strictly closer.
bool nearest(const Dyadic& exact, double value) {
  const auto distance = [&](double x) {
    const Dyadic d = Dyadic(x) - exact;
    return d.sign() < 0 ? -d : d;
  };
  const Dyadic own = distance(value);
  return order(own, distance(step_down(value))) <= 0 && order(own, distance(step_up(value))) <= 0;
}

double random_finite(std::mt19937_64& random) {
  std::uniform_real_distribution<double> unit(1.0, 2.0);
  const double value = std::ldexp(unit(random), static_cast<int>(random() % 121) - 60);
  return random() % 2 == 0 ? value : -value;
}

void dyadic_truth() {
  std::mt19937_64 random(7);
  for (int trial = 0; trial < 600; ++trial) {
    std::array<double, 4> v{};
    for (double& x : v) x = random_finite(random);
    if (v[0] > v[1]) std::swap(v[0], v[1]);
    if (v[2] > v[3]) std::swap(v[2], v[3]);
    Program p;
    const int64_t a = p.values({{v[0], v[1]}});
    const int64_t b = p.values({{v[2], v[3]}});
    const int64_t out = p.slots(4);
    p.row(kAdd, out, 1, a, -1, b, -1, 0);
    p.row(kSubtract, out + 1, 1, a, -1, b, -1, 0);
    p.row(kMultiply, out + 2, 1, a, -1, b, -1, 0);
    const int64_t gamma_index = p.parameter(2.0 * kEpsilon / (1.0 - 2.0 * kEpsilon));
    p.row(kReduceSum, out + 3, 1, a, -1, -1, gamma_index, 2);
    PHX_CHECK(execute(p) == PHX_MC_OK);
    const Dyadic al(v[0]), ah(v[1]), bl(v[2]), bh(v[3]);
    // ADD/SUB: one step outward from the nearest value of the exact endpoint sum.
    const std::array<std::array<Dyadic, 2>, 2> sums{{{al + bl, ah + bh}, {al - bh, ah - bl}}};
    for (int k = 0; k < 2; ++k) {
      const double low = p.lower[out + k], high = p.upper[out + k];
      PHX_CHECK(order(Dyadic(low), sums[k][0]) < 0 && order(sums[k][1], Dyadic(high)) < 0);
      PHX_CHECK(nearest(sums[k][0], step_up(low)) && nearest(sums[k][1], step_down(high)));
    }
    // MUL: the exact endpoint-product hull, rounded to nearest, one step out.
    std::array<Dyadic, 4> products{al * bl, al * bh, ah * bl, ah * bh};
    Dyadic least = products[0], most = products[0];
    for (const Dyadic& value : products) {
      if (order(value, least) < 0) least = value;
      if (order(value, most) > 0) most = value;
    }
    const double low = p.lower[out + 2], high = p.upper[out + 2];
    PHX_CHECK(order(Dyadic(low), least) < 0 && order(most, Dyadic(high)) < 0);
    PHX_CHECK(nearest(least, step_up(low)) && nearest(most, step_down(high)));
    // REDUCE_SUM over the two entries a, b encloses the exact endpoint sums.
    PHX_CHECK(order(Dyadic(p.lower[out + 3]), al + bl) < 0);
    PHX_CHECK(order(ah + bh, Dyadic(p.upper[out + 3])) < 0);
  }
}

// ------------------------------------------------------ random programs

double random_value(std::mt19937_64& random) {
  static const std::array<double, 18> specials{
      0.0,       -0.0,       kSmallest, -kSmallest, 1.0,    -1.0,      0.5, kLargest,  -kLargest,
      kInfinity, -kInfinity, kNaN,      1e300,      -1e300, 0.5 * kPi, kPi, 0x1p-1022, 3.0};
  if (random() % 3 == 0) return specials[random() % specials.size()];
  return random_finite(random);
}

Program random_program(std::mt19937_64& random) {
  Program p;
  const auto inputs = static_cast<int64_t>(1 + random() % 12);
  for (int64_t i = 0; i < inputs; ++i) {
    double lo = random_value(random), hi = random_value(random);
    if (random() % 4 != 0 && hi < lo) std::swap(lo, hi);
    p.lower.push_back(lo);
    p.upper.push_back(hi);
  }
  const auto rows = static_cast<int>(1 + random() % 24);
  for (int r = 0; r < rows; ++r) {
    // Template constants may sit between outputs.
    for (auto c = random() % 3; c > 0; --c) {
      p.lower.push_back(random_value(random));
      p.upper.push_back(random_value(random));
    }
    const auto out = static_cast<int64_t>(p.lower.size());
    const auto opcode = static_cast<int64_t>(1 + random() % 9);
    auto size = static_cast<int64_t>(random() % 6);
    if (opcode == kReduceSum) {
      auto aux = static_cast<int64_t>(random() % 8);
      if (size * aux > out) aux = size == 0 ? aux : out / size;
      const int64_t span = size * aux;
      const auto a = static_cast<int64_t>(random() % static_cast<uint64_t>(out - span + 1));
      const double gamma = aux * kEpsilon / (1.0 - aux * kEpsilon);
      p.row(opcode, out, size, a, -1, -1, p.parameter(random() % 5 == 0 ? 0.0 : gamma), aux);
    } else {
      auto operand = [&](int64_t& offset, int64_t& index) {
        if (size <= out && random() % 2 == 0) {
          offset = static_cast<int64_t>(random() % static_cast<uint64_t>(out - size + 1));
          index = -1;
          return;
        }
        offset = static_cast<int64_t>(random() % static_cast<uint64_t>(out));
        index = static_cast<int64_t>(p.indices.size());
        for (int64_t k = 0; k < size; ++k) {
          p.indices.push_back(static_cast<int64_t>(random() % static_cast<uint64_t>(out - offset)));
        }
      };
      int64_t a = 0, ai = -1, b = -1, bi = -1;
      operand(a, ai);
      if (opcode >= kAdd && opcode <= kMultiply) {
        if (opcode == kSubtractSame) {
          b = a;
          bi = ai;
        } else {
          operand(b, bi);
        }
      }
      p.row(opcode, out, size, a, ai, b, bi, 0);
    }
    for (int64_t k = 0; k < size; ++k) {
      p.lower.push_back(random_value(random));
      p.upper.push_back(random_value(random));
    }
  }
  return p;
}

void random_programs() {
  std::mt19937_64 random(11);
  for (int trial = 0; trial < 3000; ++trial) {
    Program p = random_program(random);
    Program again = p;
    check_against_reference(p, true);
    PHX_CHECK(execute(again) == PHX_MC_OK);
    PHX_CHECK(identical(again.lower, p.lower) && identical(again.upper, p.upper));
  }
}

// ------------------------------------------------------- refusal paths

bool unchanged(const Program& p, const Program& original) {
  return p.lower.size() == original.lower.size() &&
         std::memcmp(p.lower.data(), original.lower.data(), p.lower.size() * sizeof(double)) == 0 &&
         std::memcmp(p.upper.data(), original.upper.data(), p.upper.size() * sizeof(double)) == 0;
}

// A valid two-row base: rows write [6, 8) and [8, 10) of a 12-entry workspace
// whose entries 0..5 hold inputs and constants and 10..11 a trailing constant.
Program base_program() {
  Program p;
  p.values({{1.0, 2.0}, {-1.0, 3.0}, {0.5, 0.5}, {2.0, 2.0}, {4.0, 5.0}, {-2.0, -1.0}});
  p.slots(4);
  p.values({{7.0, 7.0}, {8.0, 8.0}});
  p.parameter(kEpsilon);
  p.index({1, 0, 5, 3});
  p.row(kAdd, 6, 2, 0, -1, 2, 0, 0);
  p.row(kMultiply, 8, 2, 6, -1, 0, 2, 0);
  return p;
}

void expect_invalid(Program p, int line) {
  const Program original = p;
  const int32_t status = execute(p);
  if (status != PHX_MC_INVALID_INPUT || !unchanged(p, original)) {
    phx::mc::test::record_failure(__FILE__, line, "invalid program refused untouched");
  }
}

void expect_valid(Program p, int line) {
  std::vector<double> low, high;
  reference(p, low, high);
  if (execute(p) != PHX_MC_OK || !identical(p.lower, low) || !identical(p.upper, high)) {
    phx::mc::test::record_failure(__FILE__, line, "boundary program executes");
  }
}

// Appends a final row after the valid base rows, which must not have written.
Program with_row(std::array<int64_t, 8> fields) {
  Program p = base_program();
  p.rows.insert(p.rows.end(), fields.begin(), fields.end());
  return p;
}

void invalid_programs() {
  expect_valid(base_program(), __LINE__);
  // Opcodes.
  expect_invalid(with_row({0, 10, 1, 0, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({10, 10, 1, 0, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({-1, 10, 1, 0, -1, -1, -1, 0}), __LINE__);
  // Output region: R fits, R + 1 past the workspace; negative or overflowing.
  expect_valid(with_row({kNegate, 10, 2, 0, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, 3, 0, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 11, 2, 0, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, -1, 0, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, kMax, 1, 0, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, kMax, 0, -1, -1, -1, 0}), __LINE__);
  // Out of order: overlapping or preceding the previous output.
  expect_invalid(with_row({kNegate, 9, 1, 0, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 0, 0, 0, -1, -1, -1, 0}), __LINE__);
  // Reads: the last element below out fits; one at or after out is refused,
  // including a read of the row's own output or of a later row's output.
  expect_valid(with_row({kNegate, 10, 2, 8, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, 2, 9, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, 2, 10, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, 1, -1, -1, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, 1, kMax, -1, -1, -1, 0}), __LINE__);
  {
    Program read_ahead = base_program();
    read_ahead.rows[3] = 8;  // ADD reads the MUL output written after it.
    expect_invalid(read_ahead, __LINE__);
  }
  // Index ranges and entries.
  expect_valid(with_row({kNegate, 10, 2, 0, 2, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, 2, 0, 3, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, 1, 0, -2, -1, -1, 0}), __LINE__);
  expect_invalid(with_row({kNegate, 10, 2, 0, kMax, -1, -1, 0}), __LINE__);
  expect_valid(with_row({kNegate, 10, 1, 5, 0, -1, -1, 0}), __LINE__);    // 5 + 1 < out
  expect_invalid(with_row({kNegate, 10, 1, 9, 0, -1, -1, 0}), __LINE__);  // 9 + 1 = out
  {
    Program bad = base_program();
    const int64_t own = bad.index({0, 0});
    bad.row(kNegate, 10, 2, 0, own, -1, -1, 0);
    const auto entry = static_cast<std::size_t>(own + 1);
    bad.indices[entry] = -1;
    expect_invalid(bad, __LINE__);
    bad.indices[entry] = kMax;
    expect_invalid(bad, __LINE__);
    bad.indices[entry] = 10;
    expect_invalid(bad, __LINE__);
    bad.indices[entry] = 9;
    expect_valid(bad, __LINE__);
  }
  // Binary second operand.
  expect_invalid(with_row({kAdd, 10, 2, 0, -1, 9, -1, 0}), __LINE__);
  expect_invalid(with_row({kAdd, 10, 2, 0, -1, 0, 3, 0}), __LINE__);
  expect_invalid(with_row({kAdd, 10, 2, 0, -1, 0, -2, 0}), __LINE__);
  expect_invalid(with_row({kAdd, 10, 2, 0, -1, 0, -1, 1}), __LINE__);
  // Noncanonical unused fields.
  expect_invalid(with_row({kNegate, 10, 1, 0, -1, 0, -1, 0}), __LINE__);
  expect_invalid(with_row({kSine, 10, 1, 0, -1, -1, 0, 0}), __LINE__);
  expect_invalid(with_row({kGather, 10, 1, 0, -1, -1, -1, 1}), __LINE__);
  expect_invalid(with_row({kReduceSum, 10, 1, 0, 0, -1, 0, 2}), __LINE__);
  expect_invalid(with_row({kReduceSum, 10, 1, 0, -1, 0, 0, 2}), __LINE__);
  // Reductions: terms end exactly at out, one more is refused; size * aux
  // overflow; parameter index range; gamma finite and nonnegative.
  expect_valid(with_row({kReduceSum, 10, 2, 0, -1, -1, 0, 5}), __LINE__);
  expect_invalid(with_row({kReduceSum, 10, 2, 1, -1, -1, 0, 5}), __LINE__);
  expect_invalid(with_row({kReduceSum, 10, 2, 0, -1, -1, 0, kMax}), __LINE__);
  expect_invalid(with_row({kReduceSum, 10, 2, 0, -1, -1, 0, -1}), __LINE__);
  expect_invalid(with_row({kReduceSum, 10, 2, -1, -1, -1, 0, 1}), __LINE__);
  expect_invalid(with_row({kReduceSum, 10, 1, 0, -1, -1, 1, 1}), __LINE__);
  expect_invalid(with_row({kReduceSum, 10, 1, 0, -1, -1, -1, 1}), __LINE__);
  for (double gamma : {kNaN, kInfinity, -kEpsilon, -kInfinity}) {
    Program bad = with_row({kReduceSum, 10, 1, 0, -1, -1, 0, 1});
    bad.parameters[0] = gamma;
    expect_invalid(bad, __LINE__);
  }
  {
    Program zero = with_row({kReduceSum, 10, 1, 0, -1, -1, 0, 1});
    zero.parameters[0] = -0.0;
    expect_valid(zero, __LINE__);
  }
  // Long reductions: elements of more than one checkpoint interval of terms
  // (including one interval plus one term and several intervals across two
  // elements) match the in-order reference bit for bit; zero-output rows
  // with extreme lengths read and write nothing and are valid.
  for (const auto& [size, aux] :
       std::array<std::array<int64_t, 2>, 3>{{{1, 4096}, {1, 4097}, {2, 3 * 4096 + 5}}}) {
    std::mt19937_64 random(static_cast<uint64_t>(size * aux));
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    Program p;
    const int64_t terms = p.slots(size * aux);
    for (int64_t t = 0; t < size * aux; ++t) {
      const double value = std::ldexp(unit(random), static_cast<int>(random() % 80) - 40);
      const auto entry = static_cast<std::size_t>(terms + t);
      p.lower[entry] = value;
      p.upper[entry] = value + std::ldexp(std::fabs(unit(random)), -30);
    }
    const int64_t gamma = p.parameter(kEpsilon);
    p.row(kNegate, p.slots(1), 1, terms, -1, -1, -1, 0);
    p.row(kReduceSum, p.slots(size), size, terms, -1, -1, gamma, aux);
    expect_valid(p, __LINE__);
  }
  for (int64_t aux : {int64_t{4096}, kMax - 1, kMax}) {
    expect_valid(with_row({kReduceSum, 10, 0, 0, -1, -1, 0, aux}), __LINE__);
    expect_valid(with_row({kReduceSum, 12, 0, 12, -1, -1, 0, aux}), __LINE__);
  }
}

void invalid_arguments() {
  Program p = base_program();
  const Program original = p;
  const int64_t rows = p.count();
  const auto indices = static_cast<int64_t>(p.indices.size());
  const auto parameters = static_cast<int64_t>(p.parameters.size());
  const auto size = static_cast<int64_t>(p.lower.size());
  auto call = [&](const int64_t* r, int64_t rc, const int64_t* i, int64_t ic, const double* g,
                  int64_t gc, double* lo, double* hi, int64_t w) {
    return phx_mc_interval_program_execute(r, rc, i, ic, g, gc, lo, hi, w);
  };
  const int64_t* r = p.rows.data();
  const int64_t* i = p.indices.data();
  const double* g = p.parameters.data();
  double* lo = p.lower.data();
  double* hi = p.upper.data();
  PHX_CHECK(call(r, -1, i, indices, g, parameters, lo, hi, size) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(r, rows, i, -1, g, parameters, lo, hi, size) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(r, rows, i, indices, g, -1, lo, hi, size) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(r, rows, i, indices, g, parameters, lo, hi, -1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(r, kMax / 4, i, indices, g, parameters, lo, hi, size) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(r, rows, i, indices, g, parameters, lo, hi, kMax) == PHX_MC_INVALID_ARGUMENT);
  constexpr int32_t kArgument = PHX_MC_INVALID_ARGUMENT;
  PHX_CHECK(call(nullptr, rows, i, indices, g, parameters, lo, hi, size) == kArgument);
  PHX_CHECK(call(r, rows, nullptr, indices, g, parameters, lo, hi, size) == kArgument);
  PHX_CHECK(call(r, rows, i, indices, nullptr, parameters, lo, hi, size) == kArgument);
  PHX_CHECK(call(r, rows, i, indices, g, parameters, nullptr, hi, size) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(r, rows, i, indices, g, parameters, lo, nullptr, size) == PHX_MC_INVALID_ARGUMENT);
  // A shorter declared workspace makes the same rows out of range.
  PHX_CHECK(call(r, rows, i, indices, g, parameters, lo, hi, 9) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(unchanged(p, original));
}

// Empty programs return their template unchanged: a constant output region
// executes no row, zero-size rows write nothing, and empty arrays may be null.
void empty_programs() {
  Program constant = base_program();
  constant.rows.clear();
  const Program original = constant;
  PHX_CHECK(execute(constant) == PHX_MC_OK && unchanged(constant, original));
  Program zero = base_program();
  zero.rows = {kNegate,    6, 0, 6, -1, -1, -1, 0, kAdd,  6,  0, 0,  4,  0,  4,  0,
               kReduceSum, 6, 0, 6, -1, -1, 0,  7, kSine, 12, 0, 12, -1, -1, -1, 0};
  const Program zero_original = zero;
  PHX_CHECK(execute(zero) == PHX_MC_OK && unchanged(zero, zero_original));
  PHX_CHECK(phx_mc_interval_program_execute(nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
                                            0) == PHX_MC_OK);
}

class Execution final {
 public:
  Execution(uint64_t work, uint64_t bytes, double seconds = INFINITY) {
    PHX_CHECK(phx_mc_execution_begin(work, 0, 0, bytes, seconds, nullptr, &scope_) == PHX_MC_OK);
  }
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;
  ~Execution() {
    if (scope_ != nullptr) finish(PHX_MC_OK);
  }
  void* scope() const { return scope_; }
  std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> finish(
      int32_t status, std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES>* memory_out = nullptr) {
    std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> counters{};
    std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory{};
    double seconds = 0.0;
    PHX_CHECK(phx_mc_execution_end(scope_, counters.data(), memory.data(), &seconds) == status);
    scope_ = nullptr;
    if (memory_out != nullptr) *memory_out = memory;
    return counters;
  }

 private:
  void* scope_ = nullptr;
};

// Under an active execution scope the kernel charges no work and allocates
// nothing: a scope without work or scratch allowance executes and its ledger
// is unchanged. It polls the deadline and recorded refusals before writing,
// so a missed deadline or an earlier refusal returns that status untouched,
// for valid and invalid programs alike.
void execution_scope() {
  Program valid = base_program();
  std::vector<double> low, high;
  reference(valid, low, high);
  {
    Execution execution(0, 0);
    Program p = valid;
    PHX_CHECK(execute(p) == PHX_MC_OK);
    PHX_CHECK(identical(p.lower, low) && identical(p.upper, high));
    std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory{};
    const auto counters = execution.finish(PHX_MC_OK, &memory);
    PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
    PHX_CHECK(counters[PHX_MC_EXECUTION_WORK_REFUSALS] == 0);
    PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] == 0);
    PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] == 0);
  }
  const Program invalid = with_row({kNegate, 10, 3, 0, -1, -1, -1, 0});
  for (const Program* input : std::array<const Program*, 2>{&valid, &invalid}) {
    {
      Execution execution(0, UINT64_MAX);
      PHX_CHECK(phx_mc_execution_charge(execution.scope(), 1, 0) == PHX_MC_CAPACITY_EXCEEDED);
      Program p = *input;
      PHX_CHECK(execute(p) == PHX_MC_CAPACITY_EXCEEDED);
      PHX_CHECK(unchanged(p, *input));
      execution.finish(PHX_MC_CAPACITY_EXCEEDED);
    }
    {
      Execution execution(UINT64_MAX, UINT64_MAX, kDeadlineSeconds);
      PHX_CHECK(phx_mc_execution_charge(execution.scope(), 0, 0) == PHX_MC_OK);
      std::this_thread::sleep_for(kPastDeadline);
      Program p = *input;
      PHX_CHECK(execute(p) == PHX_MC_TIMEOUT);
      PHX_CHECK(unchanged(p, *input));
      const auto counters = execution.finish(PHX_MC_TIMEOUT);
      PHX_CHECK(counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0);
    }
  }
}

// Callers in flush-to-zero or directed rounding receive the round-to-nearest
// gradual-underflow results and get their own mode back, refusals included.
void caller_floating_point_modes() {
  Program subnormal;
  const int64_t a = subnormal.values({{kSmallest, 0x1p-1060}, {0x1p-1023, 0x1p-1022}});
  const int64_t out = subnormal.slots(4);
  subnormal.row(kAdd, out, 2, a, -1, a, -1, 0);
  subnormal.row(kMultiply, out + 2, 2, a, -1, a, -1, 0);
  std::vector<Program> programs{subnormal};
  std::mt19937_64 random(3);
  for (int k = 0; k < 40; ++k) programs.push_back(random_program(random));
  std::vector<std::vector<double>> lows, highs;
  for (const Program& p : programs) {
    lows.emplace_back();
    highs.emplace_back();
    Program copy = p;
    PHX_CHECK(execute(copy) == PHX_MC_OK);
    lows.back() = copy.lower;
    highs.back() = copy.upper;
  }
  // The nearest-mode result is the gradual one: d + d = 2 d, not zero.
  PHX_CHECK(same(lows[0][static_cast<std::size_t>(out)], kSmallest));
  const Program refused = with_row({kNegate, 10, 3, 0, -1, -1, -1, 0});

  struct Hazard {
    bool flushing;
    int rounding;
  };
  const std::array<Hazard, 5> hazards{{{true, FE_TONEAREST},
                                       {false, FE_UPWARD},
                                       {false, FE_DOWNWARD},
                                       {false, FE_TOWARDZERO},
                                       {true, FE_DOWNWARD}}};
  const phx::mc::FloatingPointMode initial_mode = phx::mc::floating_point_mode();
  const int initial_rounding = std::fegetround();
  for (const Hazard& hazard : hazards) {
    if (hazard.flushing && !phx::mc::kFlushControl) continue;
    phx::mc::set_floating_point_mode(hazard.flushing ? phx::mc::kFlushingMode
                                                     : phx::mc::kGradualNearestMode);
    PHX_CHECK(std::fesetround(hazard.rounding) == 0);
    const phx::mc::FloatingPointMode installed = phx::mc::floating_point_mode();
    PHX_CHECK(!phx::mc::gradual_nearest_arithmetic());
    for (std::size_t k = 0; k < programs.size(); ++k) {
      Program copy = programs[k];
      PHX_CHECK(execute(copy) == PHX_MC_OK);
      PHX_CHECK(identical(copy.lower, lows[k]) && identical(copy.upper, highs[k]));
      PHX_CHECK(phx::mc::floating_point_mode() == installed &&
                std::fegetround() == hazard.rounding);
    }
    Program bad = refused;
    PHX_CHECK(execute(bad) == PHX_MC_INVALID_INPUT && unchanged(bad, refused));
    PHX_CHECK(phx::mc::floating_point_mode() == installed && std::fegetround() == hazard.rounding);
    PHX_CHECK(!phx::mc::gradual_nearest_arithmetic());
  }
  phx::mc::set_floating_point_mode(initial_mode);
  PHX_CHECK(std::fesetround(initial_rounding) == 0);
  PHX_CHECK(phx::mc::floating_point_mode() == initial_mode);
  PHX_CHECK(phx::mc::gradual_nearest_arithmetic());
}

}  // namespace

int main() {
  additive_cases();
  subtraction_cases();
  multiplication_cases();
  structural_cases();
  reduction_cases();
  trigonometric_cases();
  trigonometric_enclosure();
  dyadic_truth();
  random_programs();
  invalid_programs();
  invalid_arguments();
  empty_programs();
  execution_scope();
  caller_floating_point_modes();
  return phx::mc::test::finish("test_interval_program");
}
