// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

// V3's regular pole chart of a sphere face. The closed-form sphere
//   S(u, v) = o + r (cos v R(u) + sin v a),  R(u) = cos u e1 + sin u e2,
// is singular at the poles v = sigma pi/2 (sigma = +1 north, -1 south): the
// whole u-edge there is one point. With the colatitude rho = pi/2 - sigma v
// and the Cartesian pole chart (x, y) = rho (cos u, sin u), q = x^2 + y^2,
//   S(x, y) = o + r (g(q) (x e1 + y e2) + sigma h(q) a),
//   g(q) = sin(sqrt q) / sqrt q = sum_k (-q)^k / (2k+1)!,
//   h(q) = cos(sqrt q)          = sum_k (-q)^k / (2k)!,
// so S is analytic in (x, y) (a reparameterization of the same surface,
// regular at the pole). Other pole families are not enclosed here: a cone
// apex, a line revolution meeting its axis off the perpendicular and a horn
// or spindle torus are not smooth at the pole, and a B-spline with a
// collapsed edge has no certified Cartesian enclosure; V3 keeps the uv chart
// for them.

#include <cmath>
#include <limits>

#include "directed_interval.hpp"
#include "predicates.hpp"

namespace phx::mc::cad::pole {

namespace di = phx::mc::remesh;

// Truncation order of the g, h series and the largest q they accept.
inline constexpr int kSeriesTerms = 12;
inline constexpr double kMaximumSquaredRadius = 4.0;
inline constexpr double kCapLatitude = 0x1.3f004dc9db074p-1;

// pi/2 lies in [kHalfPiLower, di::kHalfPiUpper] (binary64 neighbors).
inline constexpr double kHalfPiLower = 0x1.921fb54442d18p+0;

inline di::Interval square(const di::Interval& x) {
  const double a = std::fabs(x.lower), b = std::fabs(x.upper);
  const double low = x.lower > 0.0 || x.upper < 0.0 ? std::min(a, b) : 0.0;
  const double high = std::max(a, b);
  return {low == 0.0 ? 0.0 : std::max(0.0, phx::mc::geometry::next_down(low * low)),
          phx::mc::geometry::next_up(high * high)};
}

inline di::Interval divide_by(const di::Interval& x, double c) {
  return {phx::mc::geometry::next_down(x.lower / c), phx::mc::geometry::next_up(x.upper / c)};
}

// The derivative of order `order` (0..2) in q of
//   sum_k (-q)^k / (2k + offset)!   (offset 1: g, offset 0: h)
// over q in [0, kMaximumSquaredRadius]: the terms k < kSeriesTerms in
// outward Horner form, plus the tail. Term k of the derivative has
// magnitude k!/(k-order)! q^(k-order) / (2k+offset)!, and the ratio of
// consecutive terms is (k+1)/(k+1-order) q / ((2k+offset+1)(2k+offset+2))
// <= (13/11) 4 / (25 26) < 1/2 for k >= 12, so the tail is at most twice
// its first term. False when q leaves the domain.
inline bool series(const di::Interval& q, int offset, int order, di::Interval& out) {
  if (!(q.lower >= 0.0) || !(q.upper <= kMaximumSquaredRadius)) return false;
  // inverse[k] encloses 1 / (2k + offset)!.
  di::Interval inverse[kSeriesTerms + 1];
  di::Interval value = di::point(1.0);
  int n = 0;
  for (int k = 0; k <= kSeriesTerms; ++k) {
    while (n < 2 * k + offset) value = divide_by(value, static_cast<double>(++n));
    inverse[k] = value;
  }
  const auto falling = [&](int k) {
    double f = 1.0;
    for (int j = 0; j < order; ++j) f *= static_cast<double>(k - j);  // exact small integers
    return f;
  };
  di::Interval sum = di::point(0.0);
  for (int k = kSeriesTerms - 1; k >= order; --k) {
    di::Interval coefficient = di::multiply(di::point(falling(k)), inverse[k]);
    if (k % 2 == 1) coefficient = {-coefficient.upper, -coefficient.lower};
    sum = di::add(di::multiply(sum, q), coefficient);
  }
  di::Interval power = di::point(1.0);
  for (int j = 0; j < kSeriesTerms - order; ++j) power = di::multiply(power, di::point(q.upper));
  const double first =
      di::multiply(di::multiply(di::point(falling(kSeriesTerms)), inverse[kSeriesTerms]), power)
          .upper;
  const double tail = phx::mc::geometry::next_up(2.0 * first);
  out = di::add(sum, di::Interval{-tail, tail});
  return di::finite(out);
}

// Pole frame: o, e1, e2, a (3 each) and r, as enclosures. `flat`: a disk
// center (the revolution of a line perpendicular to the axis, meeting it at
// the chart parameter v = pole_v): S(u, v) = o + (v - pole_v)(cos u e1 +
// sin u e2), so in the Cartesian chart (x, y) = rho (cos u, sin u), rho =
// sigma (v - pole_v) >= 0 on the face's side sigma, S(x, y) = o + sigma (x e1
// + y e2): affine, regular, zero Hessian (a unused, r = 1).
struct SphereFrame {
  di::Interval o[3], e1[3], e2[3], a[3], r;
  bool flat = false;
  double pole_v = 0.0;
};

// The chart parameter v of the pole on side sigma.
inline double pole_value(const SphereFrame& frame, int sigma) {
  return frame.flat ? frame.pole_v : sigma * kHalfPiLower;
}

// Terms 0 (S), 1 (S_xx), 2 (S_xy), 3 (S_yy) of the pole-chart map over the
// box [x] x [y], with sigma = +1 (north) or -1 (south). With W = x e1 + y e2:
//   S_xx = r ((2g' + 4x^2 g'') W + 4x g' e1 + sigma (2h' + 4x^2 h'') a)
//   S_xy = r (4xy g'' W + 2g' (x e2 + y e1) + sigma 4xy h'' a)
//   S_yy = r ((2g' + 4y^2 g'') W + 4y g' e2 + sigma (2h' + 4y^2 h'') a)
// `first`, `last` select the terms. False when q = x^2 + y^2 may exceed
// kMaximumSquaredRadius or an enclosure is not finite.
// `radius_squared`: an upper bound on x^2 + y^2 over the points of interest
// when they lie in a disk smaller than the box (the box's own bound applies
// otherwise).
inline bool jets(const SphereFrame& frame, int sigma, const di::Interval& x,
                 const di::Interval& y, int first, int last, di::Interval out[4][3],
                 double radius_squared = std::numeric_limits<double>::infinity()) {
  using di::add, di::multiply;
  const di::Interval xx = square(x), yy = square(y);
  // x^2 + y^2 >= 0: the outward sum's lower end may be clamped there.
  di::Interval q = add(xx, yy);
  q.lower = std::max(q.lower, 0.0);
  q.upper = std::min(q.upper, radius_squared);
  if (!(q.lower <= q.upper)) return false;
  const di::Interval s = di::point(static_cast<double>(sigma));
  if (frame.flat) {
    for (int i = 0; i < 3; ++i) {
      if (first == 0)
        out[0][i] = add(frame.o[i],
                        multiply(s, add(multiply(x, frame.e1[i]), multiply(y, frame.e2[i]))));
      for (int t = 1; t <= 3; ++t) out[t][i] = di::point(0.0);
      if (first == 0 && !di::finite(out[0][i])) return false;
    }
    return true;
  }
  di::Interval g, h;
  if (first == 0) {
    if (!series(q, 1, 0, g) || !series(q, 0, 0, h)) return false;
    for (int i = 0; i < 3; ++i) {
      const di::Interval w = add(multiply(x, frame.e1[i]), multiply(y, frame.e2[i]));
      out[0][i] = add(frame.o[i],
                      multiply(frame.r, add(multiply(g, w), multiply(s, multiply(h, frame.a[i])))));
      if (!di::finite(out[0][i])) return false;
    }
  }
  if (last < 1) return true;
  di::Interval g1, g2, h1, h2;
  if (!series(q, 1, 1, g1) || !series(q, 1, 2, g2) || !series(q, 0, 1, h1) ||
      !series(q, 0, 2, h2))
    return false;
  const di::Interval two = di::point(2.0), four = di::point(4.0);
  const di::Interval xy = multiply(x, y);
  const di::Interval gx = add(multiply(two, g1), multiply(four, multiply(xx, g2)));
  const di::Interval gy = add(multiply(two, g1), multiply(four, multiply(yy, g2)));
  const di::Interval hx = add(multiply(two, h1), multiply(four, multiply(xx, h2)));
  const di::Interval hy = add(multiply(two, h1), multiply(four, multiply(yy, h2)));
  const di::Interval gm = multiply(four, multiply(xy, g2)), hm = multiply(four, multiply(xy, h2));
  const di::Interval g1x = multiply(four, multiply(x, g1)), g1y = multiply(four, multiply(y, g1));
  for (int i = 0; i < 3; ++i) {
    const di::Interval w = add(multiply(x, frame.e1[i]), multiply(y, frame.e2[i]));
    const di::Interval sa = multiply(s, frame.a[i]);
    const di::Interval xx_term =
        add(add(multiply(gx, w), multiply(g1x, frame.e1[i])), multiply(hx, sa));
    const di::Interval xy_term =
        add(add(multiply(gm, w),
                multiply(multiply(two, g1), add(multiply(x, frame.e2[i]), multiply(y, frame.e1[i])))),
            multiply(hm, sa));
    const di::Interval yy_term =
        add(add(multiply(gy, w), multiply(g1y, frame.e2[i])), multiply(hy, sa));
    out[1][i] = multiply(frame.r, xx_term);
    out[2][i] = multiply(frame.r, xy_term);
    out[3][i] = multiply(frame.r, yy_term);
    for (int t = 1; t <= 3; ++t)
      if (!di::finite(out[t][i])) return false;
  }
  return true;
}

// Colatitude enclosure rho of a chart parameter v on side sigma: pi/2 -
// sigma v on a sphere (false unless sigma v <= kHalfPiLower), sigma (v -
// pole_v) at a disk center (false unless it is >= 0); rho >= 0 either way.
inline bool colatitude(const SphereFrame& frame, double v, int sigma, di::Interval& rho) {
  if (frame.flat) {
    rho = di::subtract(di::point(v), di::point(frame.pole_v));
    if (sigma < 0) rho = {-rho.upper, -rho.lower};
    if (!(sigma > 0 ? v >= frame.pole_v : v <= frame.pole_v)) return false;
    rho.lower = std::max(rho.lower, 0.0);
    return di::finite(rho);
  }
  const double w = sigma > 0 ? v : -v;
  if (!(w <= kHalfPiLower)) return false;
  rho = di::subtract(di::Interval{kHalfPiLower, di::kHalfPiUpper}, di::point(w));
  rho.lower = std::max(rho.lower, 0.0);
  return di::finite(rho);
}

}  // namespace phx::mc::cad::pole
