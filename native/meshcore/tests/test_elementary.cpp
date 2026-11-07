// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Behavioral contract of the rigorous elementary enclosures: constant
// consistency, special values with exact expectations, extremum and pole
// decisions, identities between independently enclosed functions, point
// widths, C ABI validation, determinism across parallelism and independence
// from the caller's floating-point mode.
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "check.hpp"
#include "elementary.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

namespace {

namespace elementary = phx::mc::elementary;
using elementary::DoubleWord;
using elementary::Enclosure;
using elementary::Function;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kLargest = std::numeric_limits<double>::max();
constexpr double kSmallest = std::numeric_limits<double>::denorm_min();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kHalfPiRounded = 0x1.921fb54442d18p0;  // RN(pi/2) < pi/2

double up(double value) { return std::nextafter(value, kInfinity); }
double down(double value) { return std::nextafter(value, -kInfinity); }
std::uint64_t bits(double value) { return std::bit_cast<std::uint64_t>(value); }

bool same(double first, double second) { return bits(first) == bits(second); }

bool exactly(const Enclosure& value, double lower, double upper) {
  return same(value.lower, lower) && same(value.upper, upper);
}

bool contains(const Enclosure& value, double point) {
  return value.lower <= point && point <= value.upper;
}

bool overlaps(const Enclosure& first, const Enclosure& second) {
  return first.lower <= second.upper && second.lower <= first.upper;
}

// Width in binary64 steps of an enclosure that does not straddle zero.
std::uint64_t steps(const Enclosure& value) {
  if (value.lower >= 0.0) return bits(value.upper) - bits(value.lower);
  return bits(-value.lower) - bits(-value.upper);
}

double relative_gap(DoubleWord value, DoubleWord expected) {
  const DoubleWord difference = elementary::add(value, DoubleWord{-expected.high, -expected.low});
  return std::fabs(difference.high) / std::fabs(expected.high);
}

// Outward interval arithmetic for identities: every RN result moves one step.
Enclosure product(const Enclosure& a, const Enclosure& b) {
  const std::array<double, 4> values = {a.lower * b.lower, a.lower * b.upper, a.upper * b.lower,
                                        a.upper * b.upper};
  double low = values[0];
  double high = values[0];
  for (const double value : values) {
    low = std::fmin(low, value);
    high = std::fmax(high, value);
  }
  return {down(low), up(high)};
}

Enclosure sum(const Enclosure& a, const Enclosure& b) {
  return {down(a.lower + b.lower), up(a.upper + b.upper)};
}

Enclosure point(Function function, double x) { return elementary::evaluate(function, x, x); }

void constants_are_consistent() {
  using namespace elementary::constants;
  // DW pairs are normalized: high = RN(high + low).
  const auto normalized = [](DoubleWord value) { return value.high + value.low == value.high; };
  PHX_CHECK(normalized(kHalfPi));
  for (const DoubleWord& value : kInverseFactorial) PHX_CHECK(normalized(value));
  for (const DoubleWord& value : kInverseOdd) PHX_CHECK(normalized(value));
  for (const DoubleWord& value : kArctangent) PHX_CHECK(normalized(value));
  // Cody-Waite ln 2: a 42-bit head and strictly decreasing parts.
  PHX_CHECK((bits(kLn2[0]) & ((std::uint64_t{1} << 11) - 1)) == 0);
  PHX_CHECK(std::fabs(kLn2[1]) < 0x1p-42 && std::fabs(kLn2[2]) < 0x1p-96);
  // (2/pi) (pi/2) = 1 from the first two 2/pi words (truncation < 2^-128).
  const auto word = [](std::uint64_t value, int scale) {
    return DoubleWord{static_cast<double>(value >> 32) * std::ldexp(1.0, scale + 32),
                      static_cast<double>(value & 0xFFFFFFFFu) * std::ldexp(1.0, scale)};
  };
  const DoubleWord first = word(kTwoOverPi[0], -64);
  const DoubleWord second = word(kTwoOverPi[1], -128);
  DoubleWord two_over_pi = elementary::two_sum(first.high, first.low);
  two_over_pi = elementary::add(two_over_pi, second.high);
  two_over_pi = elementary::add(two_over_pi, second.low);
  PHX_CHECK(relative_gap(elementary::multiply(two_over_pi, kHalfPi), DoubleWord{1.0, 0.0}) <=
            0x1p-100);
  // Table recurrences: n (1/n!) = 1/(n-1)! and (2n+1) (1/(2n+1)) = 1.
  for (int n = 1; n < 30; ++n) {
    PHX_CHECK(relative_gap(elementary::multiply(kInverseFactorial[n], static_cast<double>(n)),
                           kInverseFactorial[n - 1]) <= 0x1p-102);
  }
  for (int n = 0; n < 20; ++n) {
    PHX_CHECK(relative_gap(elementary::multiply(kInverseOdd[n], 2.0 * n + 1.0),
                           DoubleWord{1.0, 0.0}) <= 0x1p-102);
  }
  // atan(1) = pi/4 and atan(j/16) - atan((j-1)/16) = atan(16 / (256 + j (j-1))),
  // the difference summed by its own alternating series (argument < 1/16).
  PHX_CHECK(kArctangent[16].high == 0.5 * kHalfPi.high && kArctangent[16].low == 0.5 * kHalfPi.low);
  for (int j = 1; j <= 16; ++j) {
    const DoubleWord d = elementary::divide({16.0, 0.0}, {256.0 + j * (j - 1.0), 0.0});
    const DoubleWord square = elementary::multiply(d, d);
    DoubleWord term = d;
    DoubleWord series = d;
    for (int n = 1; n < 20; ++n) {
      term = elementary::multiply(term, DoubleWord{-square.high, -square.low});
      series = elementary::add(series, elementary::multiply(term, kInverseOdd[n]));
    }
    const DoubleWord difference = elementary::add(
        kArctangent[j], DoubleWord{-kArctangent[j - 1].high, -kArctangent[j - 1].low});
    PHX_CHECK(relative_gap(difference, series) <= 0x1p-98);
  }
}

void special_values() {
  // exp
  PHX_CHECK(exactly(point(Function::kExp, 0.0), 1.0, 1.0));
  PHX_CHECK(exactly(point(Function::kExp, -0.0), 1.0, 1.0));
  PHX_CHECK(exactly(point(Function::kExp, 1.0), 0x1.5bf0a8b145769p+1, 0x1.5bf0a8b14576ap+1));
  PHX_CHECK(exactly(point(Function::kExp, 1000.0), kLargest, kInfinity));
  PHX_CHECK(exactly(point(Function::kExp, kInfinity), kLargest, kInfinity));
  PHX_CHECK(exactly(point(Function::kExp, -1000.0), 0.0, kSmallest));
  PHX_CHECK(exactly(point(Function::kExp, -kInfinity), 0.0, kSmallest));
  PHX_CHECK(exactly(elementary::exp(-kInfinity, 0.0), 0.0, 1.0));
  PHX_CHECK(exactly(elementary::exp(kNaN, 0.0), -kInfinity, kInfinity));
  // e^-745 = 0.57 2^-1074 lies strictly between 0 and the smallest subnormal.
  PHX_CHECK(exactly(point(Function::kExp, -745.0), 0.0, kSmallest));
  // log
  PHX_CHECK(exactly(point(Function::kLog, 1.0), 0.0, 0.0));
  PHX_CHECK(exactly(point(Function::kLog, 2.0), 0x1.62e42fefa39efp-1, 0x1.62e42fefa39f0p-1));
  PHX_CHECK(exactly(elementary::log(0.0, 1.0), -kInfinity, 0.0));
  PHX_CHECK(exactly(elementary::log(-2.0, -1.0), -kInfinity, kInfinity));
  PHX_CHECK(exactly(elementary::log(-0.0, -0.0), -kInfinity, kInfinity));
  PHX_CHECK(exactly(elementary::log(kNaN, 1.0), -kInfinity, kInfinity));
  PHX_CHECK(exactly(point(Function::kLog, kInfinity), kLargest, kInfinity));
  const Enclosure log_two = point(Function::kLog, 2.0);
  const Enclosure log_smallest = point(Function::kLog, kSmallest);
  PHX_CHECK(overlaps(log_smallest, product(log_two, Enclosure{-1074.0, -1074.0})));
  PHX_CHECK(steps(log_smallest) <= 2);
  // sin, cos, tan with signed zeros and tiny arguments
  PHX_CHECK(exactly(point(Function::kSin, 0.0), 0.0, 0.0));
  PHX_CHECK(exactly(point(Function::kSin, -0.0), -0.0, -0.0));
  PHX_CHECK(exactly(point(Function::kTan, -0.0), -0.0, -0.0));
  PHX_CHECK(exactly(point(Function::kCos, -0.0), 1.0, 1.0));
  PHX_CHECK(exactly(point(Function::kSin, 0x1p-30), down(0x1p-30), 0x1p-30));
  PHX_CHECK(exactly(point(Function::kTan, -kSmallest), down(-kSmallest), -kSmallest));
  PHX_CHECK(exactly(point(Function::kCos, 0x1p-40), down(1.0), 1.0));
  PHX_CHECK(exactly(point(Function::kSin, kHalfPiRounded), down(1.0), 1.0));
  // sin(10^22) = -0.85220084976718880177..., cos(10^22) = 0.52321478539513894549...
  const Enclosure sin_huge = point(Function::kSin, 1e22);
  const Enclosure cos_huge = point(Function::kCos, 1e22);
  PHX_CHECK(sin_huge.lower <= up(-0.8522008497671888017727) &&
            sin_huge.upper >= down(-0.8522008497671888017727) && steps(sin_huge) <= 2);
  PHX_CHECK(cos_huge.lower <= up(0.5232147853951389454975) &&
            cos_huge.upper >= down(0.5232147853951389454975) && steps(cos_huge) <= 2);
  // full-range rules
  for (const Function function : {Function::kSin, Function::kCos}) {
    PHX_CHECK(exactly(elementary::evaluate(function, kNaN, 1.0), -kInfinity, kInfinity));
    PHX_CHECK(exactly(elementary::evaluate(function, 0.0, kInfinity), -1.0, 1.0));
    PHX_CHECK(exactly(elementary::evaluate(function, 1.0, 0.0), -1.0, 1.0));
    PHX_CHECK(exactly(elementary::evaluate(function, 0.0, 7.5), -1.0, 1.0));
  }
  PHX_CHECK(exactly(elementary::tan(-kInfinity, 0.0), -kInfinity, kInfinity));
  PHX_CHECK(exactly(elementary::tan(kNaN, kNaN), -kInfinity, kInfinity));
  // atan, tanh
  PHX_CHECK(exactly(point(Function::kAtan, kInfinity), kHalfPiRounded, up(kHalfPiRounded)));
  PHX_CHECK(exactly(point(Function::kAtan, -kInfinity), down(-kHalfPiRounded), -kHalfPiRounded));
  PHX_CHECK(exactly(point(Function::kAtan, 1.0), 0x1.921fb54442d18p-1, 0x1.921fb54442d19p-1));
  PHX_CHECK(exactly(point(Function::kAtan, -0.0), -0.0, -0.0));
  PHX_CHECK(exactly(point(Function::kAtan, 1e-300), down(1e-300), 1e-300));
  PHX_CHECK(exactly(point(Function::kTanh, kInfinity), 1.0, 1.0));
  PHX_CHECK(exactly(elementary::tanh(-kInfinity, kInfinity), -1.0, 1.0));
  PHX_CHECK(exactly(point(Function::kTanh, 20.0), down(1.0), 1.0));
  PHX_CHECK(exactly(point(Function::kTanh, -25.0), -1.0, up(-1.0)));
  PHX_CHECK(exactly(point(Function::kTanh, 0.0), 0.0, 0.0));
  PHX_CHECK(exactly(elementary::tanh(kNaN, 1.0), -kInfinity, kInfinity));
  PHX_CHECK(exactly(elementary::atan(1.0, kNaN), -kInfinity, kInfinity));
  // Monotone extensions are endpointwise, so inverted boxes stay inverted.
  const Enclosure inverted = elementary::exp(1.0, 0.0);
  PHX_CHECK(inverted.lower > inverted.upper);
}

void extremum_and_pole_decisions() {
  // sin has its maximum at pi/2 in [1, 2] and none in [2, 3].
  const Enclosure around_maximum = elementary::sin(1.0, 2.0);
  PHX_CHECK(around_maximum.upper == 1.0 &&
            same(around_maximum.lower, point(Function::kSin, 1.0).lower));
  const Enclosure past_maximum = elementary::sin(2.0, 3.0);
  PHX_CHECK(past_maximum.upper < 1.0);
  PHX_CHECK(same(past_maximum.lower, point(Function::kSin, 3.0).lower));
  PHX_CHECK(exactly(elementary::sin(0.0, 2.0), 0.0, 1.0));
  PHX_CHECK(elementary::cos(-0.5, 0.5).upper == 1.0);
  PHX_CHECK(elementary::cos(3.0, 3.5).lower == -1.0);
  PHX_CHECK(elementary::cos(0.5, 3.0).lower > -1.0 && elementary::cos(0.5, 3.0).upper < 1.0);
  // RN(pi/2) lies below the pole, its successor above it.
  const Enclosure below_pole = elementary::tan(1.5, kHalfPiRounded);
  PHX_CHECK(std::isfinite(below_pole.upper) && below_pole.upper > 1.6e16);
  PHX_CHECK(exactly(elementary::tan(1.5, up(kHalfPiRounded)), -kInfinity, kInfinity));
  PHX_CHECK(exactly(elementary::tan(up(kHalfPiRounded), 3.0), point(Function::kTan, up(kHalfPiRounded)).lower,
                    point(Function::kTan, 3.0).upper));
  PHX_CHECK(exactly(elementary::tan(-1.0, 1.0), point(Function::kTan, -1.0).lower,
                    point(Function::kTan, 1.0).upper));
  // sin over [RN(pi/2), next(RN(pi/2))] contains the maximum, so its upper bound is 1.
  PHX_CHECK(elementary::sin(kHalfPiRounded, up(kHalfPiRounded)).upper == 1.0);
  // A box of width in (2 pi, 7] holds both extrema; one of width just under
  // 2 pi placed between extrema misses the maximum.
  PHX_CHECK(exactly(elementary::sin(0.0, 6.5), -1.0, 1.0));
  const Enclosure narrow = elementary::sin(1.6, 1.6 + 6.2);
  PHX_CHECK(narrow.lower == -1.0 && narrow.upper < 1.0);
  // Huge arguments: adjacent binary64 values 2^-12 apart near 2^40 give a
  // narrow enclosure; near 2^60 they are 256 apart, wider than 7: full range.
  const double huge = 0x1p40;
  const Enclosure tiny_box = elementary::sin(huge, up(huge));
  PHX_CHECK(tiny_box.upper - tiny_box.lower <= 0x1p-11);
  for (const double x : {huge, up(huge)}) {
    PHX_CHECK(tiny_box.lower <= point(Function::kSin, x).lower &&
              point(Function::kSin, x).upper <= tiny_box.upper);
  }
  PHX_CHECK(exactly(elementary::sin(0x1p60, up(0x1p60)), -1.0, 1.0));
}

void identities_and_widths() {
  std::mt19937_64 random(20261107);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const std::array<double, 9> scales = {1e-12, 1e-6, 0.01, 0.7, 1.0, 10.0, 700.0, 1e8, 1e300};
  std::uint64_t widest[7] = {};
  for (int sample = 0; sample < 20000; ++sample) {
    const double scale = scales[static_cast<std::size_t>(sample) % scales.size()];
    const double x = (2.0 * unit(random) - 1.0) * scale;
    const Enclosure sine = point(Function::kSin, x);
    const Enclosure cosine = point(Function::kCos, x);
    PHX_CHECK(contains(sum(product(sine, sine), product(cosine, cosine)), 1.0));
    PHX_CHECK(exactly(point(Function::kSin, -x), -sine.upper, -sine.lower));
    PHX_CHECK(exactly(point(Function::kCos, -x), cosine.lower, cosine.upper));
    const Enclosure tangent = elementary::tan(x, x);
    if (std::isfinite(tangent.lower)) PHX_CHECK(overlaps(product(tangent, cosine), sine));
    const Enclosure arctangent = point(Function::kAtan, x);
    PHX_CHECK(exactly(point(Function::kAtan, -x), -arctangent.upper, -arctangent.lower));
    const Enclosure hyperbolic = point(Function::kTanh, x);
    PHX_CHECK(exactly(point(Function::kTanh, -x), -hyperbolic.upper, -hyperbolic.lower));
    if (std::fabs(x) < 700.0) {
      const Enclosure exponential = point(Function::kExp, x);
      PHX_CHECK(contains(product(exponential, point(Function::kExp, -x)), 1.0));
      PHX_CHECK(contains(elementary::log(exponential.lower, exponential.upper), x));
      const Enclosure doubled = point(Function::kExp, 2.0 * x);
      const Enclosure ratio_numerator = {down(doubled.lower - 1.0), up(doubled.upper - 1.0)};
      const Enclosure ratio_denominator = {down(doubled.lower + 1.0), up(doubled.upper + 1.0)};
      const Enclosure reciprocal = {down(1.0 / ratio_denominator.upper),
                                    up(1.0 / ratio_denominator.lower)};
      PHX_CHECK(overlaps(product(ratio_numerator, reciprocal), hyperbolic));
      widest[0] = std::max(widest[0], steps(exponential));
    }
    if (std::fabs(x) < 1.5) PHX_CHECK(contains(elementary::atan(tangent.lower, tangent.upper), x));
    if (x != 0.0) {
      const Enclosure logarithm = point(Function::kLog, std::fabs(x));
      if (std::fabs(x) != 1.0) widest[1] = std::max(widest[1], steps(logarithm));
    }
    if (std::fabs(x) < 1e8) {
      widest[2] = std::max(widest[2], steps(sine));
      widest[3] = std::max(widest[3], steps(cosine));
      if (std::isfinite(tangent.lower)) widest[4] = std::max(widest[4], steps(tangent));
    }
    widest[5] = std::max(widest[5], steps(arctangent));
    widest[6] = std::max(widest[6], steps(hyperbolic));
  }
  for (const std::uint64_t width : widest) PHX_CHECK(width <= 4);
}

void interval_containment() {
  // Endpoint bounds of monotone functions are nested inside the box enclosure
  // and sampled interior points of periodic boxes are enclosed.
  std::mt19937_64 random(7);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int sample = 0; sample < 4000; ++sample) {
    const double a = (2.0 * unit(random) - 1.0) * 20.0;
    const double b = a + unit(random) * 8.0;
    for (const Function function : {Function::kSin, Function::kCos, Function::kTan}) {
      const Enclosure box = elementary::evaluate(function, a, b);
      for (int step = 0; step <= 16; ++step) {
        const double x = a + (b - a) * (step / 16.0);
        if (x > b) continue;
        const Enclosure value = point(function, x);
        PHX_CHECK(box.lower <= value.lower && value.upper <= box.upper);
      }
    }
  }
}

void c_abi_contract() {
  const std::vector<double> lower = {0.0, -1.0, 2.0};
  const std::vector<double> upper = {1.0, 1.0, 2.0};
  std::vector<double> out_lower(3, 7.0), out_upper(3, 7.0);
  PHX_CHECK(phx_mc_interval_elementary(PHX_MC_ELEMENTARY_SIN, 3, lower.data(), upper.data(),
                                       out_lower.data(), out_upper.data(), 1) == PHX_MC_OK);
  for (std::size_t item = 0; item < 3; ++item) {
    const Enclosure expected = elementary::sin(lower[item], upper[item]);
    PHX_CHECK(exactly(Enclosure{out_lower[item], out_upper[item]}, expected.lower, expected.upper));
  }
  const auto call = [&](int32_t function, int64_t count, const double* low, double* out,
                        int32_t parallelism) {
    return phx_mc_interval_elementary(function, count, low, upper.data(), out, out_upper.data(),
                                      parallelism);
  };
  PHX_CHECK(call(-1, 3, lower.data(), out_lower.data(), 1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(7, 3, lower.data(), out_lower.data(), 1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(PHX_MC_ELEMENTARY_EXP, -1, lower.data(), out_lower.data(), 1) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(PHX_MC_ELEMENTARY_EXP, 3, nullptr, out_lower.data(), 1) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(PHX_MC_ELEMENTARY_EXP, 3, lower.data(), nullptr, 1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(PHX_MC_ELEMENTARY_EXP, 3, lower.data(), out_lower.data(), 0) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(PHX_MC_ELEMENTARY_EXP, 0, nullptr, nullptr, 1) == PHX_MC_OK);
}

std::vector<double> mixed_inputs(std::size_t count) {
  std::mt19937_64 random(99);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::vector<double> values(count);
  for (std::size_t item = 0; item < count; ++item) {
    values[item] = unit(random) * std::ldexp(1.0, static_cast<int>(item % 80) - 40);
  }
  return values;
}

void deterministic_across_parallelism() {
  const std::vector<double> lower = mixed_inputs(5000);
  std::vector<double> upper(lower.size());
  for (std::size_t item = 0; item < lower.size(); ++item) upper[item] = lower[item] + 0.25;
  for (int32_t function = PHX_MC_ELEMENTARY_EXP; function <= PHX_MC_ELEMENTARY_TANH; ++function) {
    std::vector<double> serial_lower(lower.size()), serial_upper(lower.size());
    std::vector<double> parallel_lower(lower.size()), parallel_upper(lower.size());
    PHX_CHECK(phx_mc_interval_elementary(function, static_cast<int64_t>(lower.size()), lower.data(),
                                         upper.data(), serial_lower.data(), serial_upper.data(),
                                         1) == PHX_MC_OK);
    PHX_CHECK(phx_mc_interval_elementary(function, static_cast<int64_t>(lower.size()), lower.data(),
                                         upper.data(), parallel_lower.data(),
                                         parallel_upper.data(), 4) == PHX_MC_OK);
    for (std::size_t item = 0; item < lower.size(); ++item) {
      PHX_CHECK(same(serial_lower[item], parallel_lower[item]));
      PHX_CHECK(same(serial_upper[item], parallel_upper[item]));
    }
  }
}

void independent_of_caller_mode() {
  const std::vector<double> lower = {0x1p-1060, -745.0, 0.5, 1e22, 3.0, 0.7};
  const std::vector<double> upper = {0x1p-1050, -744.0, 2.0, 1e22, 3.5, 1.3};
  const std::size_t count = lower.size();
  std::vector<std::vector<double>> nearest(7, std::vector<double>(2 * count));
  for (int32_t function = 0; function < 7; ++function) {
    PHX_CHECK(phx_mc_interval_elementary(function, static_cast<int64_t>(count), lower.data(),
                                         upper.data(), nearest[function].data(),
                                         nearest[function].data() + count, 2) == PHX_MC_OK);
  }
  struct Hazard {
    bool flush;
    int rounding;
  };
  const std::array<Hazard, 4> hazards{
      {{true, FE_TONEAREST}, {false, FE_UPWARD}, {false, FE_TOWARDZERO}, {true, FE_DOWNWARD}}};
  const auto initial_mode = phx::mc::floating_point_mode();
  const int initial_rounding = std::fegetround();
  for (const Hazard& hazard : hazards) {
    if (hazard.flush && !phx::mc::kFlushControl) continue;
    phx::mc::set_floating_point_mode(hazard.flush ? phx::mc::kFlushingMode
                                                  : phx::mc::kGradualNearestMode);
    PHX_CHECK(std::fesetround(hazard.rounding) == 0);
    const auto installed = phx::mc::floating_point_mode();
    for (int32_t function = 0; function < 7; ++function) {
      std::vector<double> result(2 * count);
      PHX_CHECK(phx_mc_interval_elementary(function, static_cast<int64_t>(count), lower.data(),
                                           upper.data(), result.data(), result.data() + count,
                                           2) == PHX_MC_OK);
      for (std::size_t item = 0; item < 2 * count; ++item) {
        PHX_CHECK(same(result[item], nearest[function][item]));
      }
    }
    PHX_CHECK(phx::mc::floating_point_mode() == installed);
    PHX_CHECK(std::fegetround() == hazard.rounding);
  }
  phx::mc::set_floating_point_mode(initial_mode);
  PHX_CHECK(std::fesetround(initial_rounding) == 0);
}

}  // namespace

int main() {
  const phx::mc::GradualNearestScope mode;
  PHX_CHECK(mode.established());
  constants_are_consistent();
  special_values();
  extremum_and_pole_decisions();
  identities_and_widths();
  interval_containment();
  c_abi_contract();
  deterministic_across_parallelism();
  independent_of_caller_mode();
  return phx::mc::test::finish("test_elementary");
}
