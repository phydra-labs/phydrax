// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Binary64 angles with one result per input on every platform.
//
// A platform asin, acos, atan2 or sin is within a unit or two in the last
// place of the true value but is not the same double on every platform. The
// values here are produced by binary64 addition, subtraction,
// multiplication, division and square root in a written order (the kernels
// of angle.cpp) or by the elementary interval kernel (elementary.hpp), so
// each input has one result everywhere.
//
// They measure and propose: a quality angle, a turn, a threshold, a radius.
// None is correctly rounded and no proof may rest on them; a proof takes its
// bound from the interval kernel. They require round-to-nearest with gradual
// underflow and no contraction, which the build and the C entries of the
// meshers establish (GradualNearestScope of floating_point_mode.hpp).
#pragma once

#include <cmath>
#include <limits>

#include "elementary.hpp"

namespace phx::mc::angle {

// pi and pi / 2 rounded to nearest; both are below the true values.
inline constexpr double kPi = 0x1.921fb54442d18p+1;
inline constexpr double kHalfPi = 0x1.921fb54442d18p+0;

// asin(x) within 7 ulp and acos(x) within 5 ulp for x in [-1, 1]; NaN outside
// it and for NaN. asin keeps the sign of a zero; acos(1) is +0.
double arcsine(double x) noexcept;
double arccosine(double x) noexcept;

// The angle in [0, pi] of the direction (along, across) for across >= 0,
// atan2(across, along), within 7 ulp: between two vectors, with the length of
// their cross product and their dot product. The quadrant follows the sign
// of `along` exactly; (0, 0) gives 0 as atan2 does; a negative `across` and a
// value that is not finite give NaN.
double between(double across, double along) noexcept;

// atan2(ordinate, abscissa) in [-pi, pi], within 7 ulp: `between` of the
// magnitude of the ordinate, with the ordinate's sign. Both quadrant
// decisions are on the signs of the arguments exactly. Zeros follow IEEE
// 754: the result has the sign of the ordinate, also of a negative zero, and
// a zero ordinate gives a zero for an abscissa that is positive or +0 and pi
// for one that is negative or -0. A value that is not finite gives NaN.
double arctangent2(double ordinate, double abscissa) noexcept;

// sin(x) and cos(x) for finite x, NaN otherwise: the lower endpoint of the
// interval kernel's enclosure, at most the true value and at most four
// binary64 values below it. Several times the cost of the kernels above; for
// thresholds, radii and generated geometry.
inline double sine(double x) noexcept {
  if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
  return elementary::sin(x, x).lower;
}
inline double cosine(double x) noexcept {
  if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
  return elementary::cos(x, x).lower;
}

}  // namespace phx::mc::angle
