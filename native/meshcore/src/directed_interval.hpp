// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Outward-rounded interval arithmetic on binary64 endpoints and the certified
// bounds the surface admissions build from it: norms of enclosed vectors, the
// angle between two of them, and the endpoints of asin and atan from the
// elementary interval kernel. Every operation is a binary64 operation moved
// one value outward or a kernel without a platform libm, so each bound is
// the same double on every platform. Requires round-to-nearest with gradual
// underflow (GradualNearestScope of floating_point_mode.hpp).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "directed_norm.hpp"
#include "elementary.hpp"
#include "rounding.hpp"

namespace phx::mc::remesh {

struct Interval {
  double lower = 0;
  double upper = 0;
};

// pi and pi / 2 rounded up.
inline constexpr double kPiUpper = 0x1.921fb54442d19p+1;
inline constexpr double kHalfPiUpper = 0x1.921fb54442d19p+0;

inline bool finite(const Interval& value) {
  return std::isfinite(value.lower) && std::isfinite(value.upper) && value.lower <= value.upper;
}
inline Interval point(double value) { return {value, value}; }
// The first smallest (largest) of the values in order, the element std::min
// (std::max) of a list selects. Spelled out: the list overloads are
// out-of-line library calls on some toolchains, and these run per interval
// product.
template <class... Rest>
double least(double first, Rest... rest) {
  double result = first;
  ((result = rest < result ? rest : result), ...);
  return result;
}
template <class... Rest>
double greatest(double first, Rest... rest) {
  double result = first;
  ((result = result < rest ? rest : result), ...);
  return result;
}
inline Interval add(const Interval& a, const Interval& b) {
  return {geometry::next_down(a.lower + b.lower), geometry::next_up(a.upper + b.upper)};
}
inline Interval subtract(const Interval& a, const Interval& b) {
  return {geometry::next_down(a.lower - b.upper), geometry::next_up(a.upper - b.lower)};
}
inline Interval multiply(const Interval& a, const Interval& b) {
  const double p[4] = {a.lower * b.lower, a.lower * b.upper, a.upper * b.lower,
                       a.upper * b.upper};
  return {geometry::next_down(least(p[0], p[1], p[2], p[3])),
          geometry::next_up(greatest(p[0], p[1], p[2], p[3]))};
}
// Smallest magnitude in the interval (0 when it contains zero).
inline double mignitude(const Interval& value) {
  if (value.lower > 0) return value.lower;
  if (value.upper < 0) return -value.upper;
  return 0;
}
inline double magnitude(const Interval& value) {
  return std::max(std::fabs(value.lower), std::fabs(value.upper));
}

using IntervalVector = std::array<Interval, 3>;

// Certified lower bound of |v| over a componentwise enclosure.
inline double norm_lower(const IntervalVector& vector) {
  const double least_parts[3] = {mignitude(vector[0]), mignitude(vector[1]),
                                 mignitude(vector[2])};
  return geometry::norm_lower(least_parts);
}
// Certified upper bound of |v| over a componentwise enclosure.
inline double norm_upper(const IntervalVector& vector) {
  const double most[3] = {magnitude(vector[0]), magnitude(vector[1]), magnitude(vector[2])};
  return geometry::norm_upper(most);
}
// b - a for two binary64 points.
inline IntervalVector enclosed_difference(const std::array<double, 3>& b,
                                          const std::array<double, 3>& a) {
  return {subtract(point(b[0]), point(a[0])), subtract(point(b[1]), point(a[1])),
          subtract(point(b[2]), point(a[2]))};
}
inline IntervalVector cross(const IntervalVector& u, const IntervalVector& v) {
  return {subtract(multiply(u[1], v[2]), multiply(u[2], v[1])),
          subtract(multiply(u[2], v[0]), multiply(u[0], v[2])),
          subtract(multiply(u[0], v[1]), multiply(u[1], v[0]))};
}
inline Interval dot(const IntervalVector& u, const IntervalVector& v) {
  return add(add(multiply(u[0], v[0]), multiply(u[1], v[1])), multiply(u[2], v[2]));
}

// The endpoints of asin at a ratio in [0, 1] and of atan at a ratio >= 0.
inline double asin_upper(double ratio) { return elementary::asin(ratio, ratio).upper; }
inline double asin_lower(double ratio) { return elementary::asin(ratio, ratio).lower; }
inline double atan_upper(double ratio) { return elementary::atan(ratio, ratio).upper; }
inline double atan_lower(double ratio) { return elementary::atan(ratio, ratio).lower; }

// An upper bound of the angle in [0, pi] between every u in `first` and every
// v in `second`; kPiUpper when the enclosures do not bound it.
inline double angle_upper(const IntervalVector& first, const IntervalVector& second) {
  const IntervalVector normal = cross(first, second);
  const Interval along = dot(first, second);
  // Acute: atan(|u x v| / (u . v)) with the quotient at its largest.
  if (along.lower > 0) return atan_upper(geometry::next_up(norm_upper(normal) / along.lower));
  const double sine = norm_lower(normal);
  if (!(sine > 0)) return kPiUpper;
  // Obtuse: pi - atan(|u x v| / |u . v|) with the quotient at its smallest.
  if (along.upper < 0) {
    return geometry::next_up(kPiUpper - atan_lower(geometry::next_down(sine / -along.lower)));
  }
  // Either: pi / 2 + atan(-(u . v) / |u x v|) with the quotient at its largest.
  return geometry::next_up(kHalfPiUpper + atan_upper(geometry::next_up(-along.lower / sine)));
}

}  // namespace phx::mc::remesh
