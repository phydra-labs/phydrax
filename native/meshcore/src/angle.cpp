// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// The angle kernels of angle.hpp. u = 2^-53; a relative error of r u is at
// most r units in the last place.
//
// Arcsine. For 0 <= x <= 1/2 and z = x^2,
//   asin(x) = x (1 + z S(z)),  S(z) = sum_{k >= 1} c_k z^(k-1),
//   c_k = (2k)! / (4^k (k!)^2 (2k+1)) = 1/6, 3/40, 5/112, ...
// The coefficients are positive and decrease, so after 26 terms the tail is
// below c_27 z^26 / (1 - z) < 2^-62 for z <= 1/4. S is evaluated as
// E(z^2) + z O(z^2) with the even and the odd terms by Horner's rule: all
// terms are positive, so its relative rounding error is below 30 u, and that
// of x (z S) below 33 u. Since z S(z) = asin(x) / x - 1 <= pi/3 - 1 < 0.048,
// the correction carries an error below 1.6 u x; with the final sum the core
// is within 2.6 u of asin(x) relatively, under 3 ulp.
//
// For 1/2 < x <= 1, w = (1 - x) / 2 is exact (Sterbenz) and
//   asin(x) = pi/2 - 2 asin(sqrt(w)),  acos(x) = 2 asin(sqrt(w)),
// with sqrt(w) <= 1/2 rounded once (relative error u, which asin at most
// carries over as 1.2 u): t = asin(sqrt(w)) is within 4 u relatively. acos is
// 2 t: under 5 ulp. asin is pi/2 - 2 t with pi/2 as a high and a low part;
// 2 t <= pi/3 has an absolute error below 4.2 u, the two subtractions add at
// most u and half an ulp of a result that is at least pi/6 > 1/2, whose ulp
// is at least u: under 7 ulp. On the other pieces acos is pi/2 -+ asin(|x|)
// or pi - 2 t, of size at least pi/3 with an ulp of at least 2 u: under 3 ulp.
//
// Arctangent. For 0 <= u <= 1 the argument is moved to |v| <= 1/4 by
//   atan(u) = atan(c) + atan((u - c) / (1 + c u)),  c = 0, 1/2 or 1,
// with u - c exact; v carries a relative error below 2 u. Then
//   atan(v) = v - v^3 T(v^2),  T(q) = 1/3 - q/5 + q^2/7 - ...,
// an alternating series whose tail after 14 terms is below q^14 / 31 <
// 2^-60. v^2 T <= 1/48 bounds the correction, whose rounding error stays
// below u |v|; atan(v) is within 3.2 u |v|. Added to atan(c) (high and low
// part), the worst case is u just above 1/4, where the result 0.245 is the
// difference of 0.464 and 0.222: under 4 ulp. For t > 1, atan(t) = pi/2 -
// atan(1 / t), the quotient adding u: under 6 ulp of a result in [pi/4, pi/2].
// `between` is atan of the quotient for along > 0 and pi minus it for
// along < 0, a result in [pi/2, pi] with an ulp of at least 2 u: under 7 ulp.
//
// test_angle holds every kernel to 3 binary64 values from the interval
// kernel's enclosure of the true value on its sample stream; the largest
// distance observed from the correctly rounded value is 2.
#include "angle.hpp"

#include <cstddef>

#include "deterministic_float.hpp"

namespace phx::mc::angle {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kHalfPiLow = 0x1.1a62633145c07p-54;
constexpr double kPiLow = 0x1.1a62633145c07p-53;
constexpr double kQuarterPi = 0x1.921fb54442d18p-1;
constexpr double kQuarterPiLow = 0x1.1a62633145c07p-55;
constexpr double kArctangentHalf = 0x1.dac670561bb4fp-2;
constexpr double kArctangentHalfLow = 0x1.a2b7f222f65e2p-56;

// c_1, c_3, ..., c_25 and c_2, c_4, ..., c_26 of the arcsine series, each the
// binary64 value nearest the rational.
constexpr double kEven[13] = {
    0x1.5555555555555p-3, 0x1.6db6db6db6db7p-5, 0x1.6e8ba2e8ba2e9p-6, 0x1.c99999999999ap-7,
    0x1.3fde50d79435ep-7, 0x1.df3bd37a6f4dfp-8, 0x1.782dda12f684cp-8, 0x1.31683bdef7bdfp-8,
    0x1.fcaf8fb6db6dbp-9, 0x1.b026f57b13b14p-9, 0x1.750de64d7d05fp-9, 0x1.464c0950f7d47p-9,
    0x1.208d3570ae5a6p-9};
constexpr double kOdd[13] = {
    0x1.3333333333333p-4, 0x1.f1c71c71c71c7p-6, 0x1.1c4ec4ec4ec4fp-6, 0x1.7a87878787878p-7,
    0x1.12ef3cf3cf3cfp-7, 0x1.a6863d70a3d71p-8, 0x1.51ba308d3dcb1p-8, 0x1.15ee9d45d1746p-8,
    0x1.d3d2a8e0dd67dp-9, 0x1.90cb77f60c7cep-9, 0x1.5c5f56efaaaabp-9, 0x1.3275586c5f2f0p-9,
    0x1.1052bc5fa960ap-9};
// 1/3, -1/5, 1/7, ..., -1/29.
constexpr double kArctangent[14] = {
    0x1.5555555555555p-2,  -0x1.999999999999ap-3, 0x1.2492492492492p-3,  -0x1.c71c71c71c71cp-4,
    0x1.745d1745d1746p-4,  -0x1.3b13b13b13b14p-4, 0x1.1111111111111p-4,  -0x1.e1e1e1e1e1e1ep-5,
    0x1.af286bca1af28p-5,  -0x1.8618618618618p-5, 0x1.642c8590b2164p-5,  -0x1.47ae147ae147bp-5,
    0x1.2f684bda12f68p-5,  -0x1.1a7b9611a7b96p-5};

template <std::size_t Count>
double horner(const double (&coefficients)[Count], double z) noexcept {
  double total = coefficients[Count - 1];
  for (std::size_t index = Count - 1; index-- > 0;) total = total * z + coefficients[index];
  return total;
}

// asin(x) as x + x z S(z) for 0 <= x <= 1/2, with z = x^2 or its exact value.
double arcsine_core(double x, double z) noexcept {
  const double q = z * z;
  const double series = horner(kEven, q) + z * horner(kOdd, q);
  return x + x * (z * series);
}

// atan(u) for 0 <= u <= 1.
double arctangent_unit(double u) noexcept {
  double v = u;
  double base = 0.0;
  double base_low = 0.0;
  if (u >= 0.75) {
    v = (u - 1.0) / (1.0 + u);
    base = kQuarterPi;
    base_low = kQuarterPiLow;
  } else if (u >= 0.25) {
    v = (u - 0.5) / (1.0 + 0.5 * u);
    base = kArctangentHalf;
    base_low = kArctangentHalfLow;
  }
  const double q = v * v;
  return base + (v - (v * (q * horner(kArctangent, q)) - base_low));
}

// atan(t) for t >= 0, +infinity included.
double arctangent(double t) noexcept {
  if (t <= 1.0) return arctangent_unit(t);
  return kHalfPi - (arctangent_unit(1.0 / t) - kHalfPiLow);
}

}  // namespace

double arcsine(double x) noexcept {
  if (!(x >= -1.0 && x <= 1.0)) return kNaN;
  const double magnitude = x < 0.0 ? -x : x;
  if (magnitude <= 0.5) return x < 0.0 ? -arcsine_core(magnitude, x * x) : arcsine_core(x, x * x);
  const double w = 0.5 * (1.0 - magnitude);
  const double value = kHalfPi - (2.0 * arcsine_core(std::sqrt(w), w) - kHalfPiLow);
  return x < 0.0 ? -value : value;
}

double arccosine(double x) noexcept {
  if (!(x >= -1.0 && x <= 1.0)) return kNaN;
  const double magnitude = x < 0.0 ? -x : x;
  if (magnitude <= 0.5) {
    const double inner = arcsine_core(magnitude, x * x);
    return x < 0.0 ? kHalfPi + (inner + kHalfPiLow) : kHalfPi - (inner - kHalfPiLow);
  }
  const double w = 0.5 * (1.0 - magnitude);
  const double twice = 2.0 * arcsine_core(std::sqrt(w), w);
  return x < 0.0 ? kPi - (twice - kPiLow) : twice;
}

double between(double across, double along) noexcept {
  if (!(across >= 0.0) || !std::isfinite(across) || !std::isfinite(along)) return kNaN;
  if (along > 0.0) return arctangent(across / along);
  if (along < 0.0) return kPi - (arctangent(across / -along) - kPiLow);
  return across > 0.0 ? kHalfPi : 0.0;
}

}  // namespace phx::mc::angle
