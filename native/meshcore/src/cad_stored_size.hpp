// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "expansion.hpp"

namespace phx::mc {

// The stored-coordinate size test of a CAD face triangle: R^2 <= (h^2 / 2)
// (1 + tau) on the binary64 positions assembly emits, decided exactly
// (phydrax/meshing/_sizing.py, CAD_CHART_SURFACE_SIZE, makes the same
// decision on the same coordinates).
//
// tau is the rounding allowance: a triangle within eps of the stored one
// (each vertex moved by at most eps in Euclidean norm) with R <= h/sqrt(2)
// passes. eps = sqrt(3) * 4u * M, M the largest stored coordinate magnitude
// of the triangle, u = 2^-53: per coordinate, the affine chart evaluation
// o + u e1 + v e2 (gamma_3 of its terms, each at most M) and the rounding of
// the constructed chart point it maps (one u M). Moving the vertices by
// eps changes each side length l_k by at most 2 eps and the cross product
// x = e0 x e1 by at most d = 2 eps (l0 + l1) + 4 eps^2, so R^2 =
// l0^2 l1^2 l2^2 / (4 |x|^2) changes by a factor at most
// prod (1 + s_k)^2 / (1 - d/|x|)^2 <= exp(y), s_k = 2 eps / l_k,
// y = 2 sum s_k + 4 d / |x| (with d / |x| <= 1/2), and exp(y) <= 1 + 2y for
// y <= 1. tau = 2y, raised by 16u for its own rounding (a sum and products
// of a dozen positive rounded terms, relative error below 16u); tau = 0 (no
// allowance) when y > 1 or the triangle is degenerate in floating point.
inline double stored_size_allowance(const double* p, const double* q, const double* r) {
  constexpr double u = std::numeric_limits<double>::epsilon() / 2.0;
  double magnitude = 0.0;
  for (int k = 0; k < 3; ++k)
    magnitude = std::max({magnitude, std::abs(p[k]), std::abs(q[k]), std::abs(r[k])});
  const double eps = std::sqrt(3.0) * 4.0 * u * magnitude;
  const double* vertices[3] = {p, q, r};
  double sides[3][3], lengths[3];
  for (int k = 0; k < 3; ++k) {
    const double* a = vertices[(k + 1) % 3];
    const double* b = vertices[(k + 2) % 3];
    for (int i = 0; i < 3; ++i) sides[k][i] = b[i] - a[i];
    lengths[k] = std::sqrt(sides[k][0] * sides[k][0] + sides[k][1] * sides[k][1] +
                           sides[k][2] * sides[k][2]);
  }
  const double cross[3] = {sides[0][1] * sides[1][2] - sides[0][2] * sides[1][1],
                           sides[0][2] * sides[1][0] - sides[0][0] * sides[1][2],
                           sides[0][0] * sides[1][1] - sides[0][1] * sides[1][0]};
  const double norm =
      std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
  if (!(norm > 0.0) || !(lengths[0] > 0.0) || !(lengths[1] > 0.0) || !(lengths[2] > 0.0))
    return 0.0;
  const double d = 2.0 * eps * (lengths[0] + lengths[1]) + 4.0 * eps * eps;
  if (!(d <= 0.5 * norm)) return 0.0;
  const double y = 4.0 * eps * (1.0 / lengths[0] + 1.0 / lengths[1] + 1.0 / lengths[2]) +
                   4.0 * d / norm;
  if (!(y <= 1.0)) return 0.0;
  return 2.0 * y * (1.0 + 16.0 * u);
}

// Whether R^2 > (h^2 / 2)(1 + tau) for the stored triangle pqr: in floating
// point when the gap clears the proved error bounds (the bound of
// triangle_circumradius_measures in _sizing.py: relative 64u (1 + |e0|^2
// |e1|^2 / |x|^2) on R^2, 16u on the limit), else exactly as
// l0^2 l1^2 l2^2 > 2 h^2 (1 + tau) |x|^2 on Expansion arithmetic. A
// degenerate triangle (x = 0 exactly) exceeds every limit.
inline bool stored_size_exceeds(const double* p, const double* q, const double* r, double h,
                                double tau) {
  constexpr double u = std::numeric_limits<double>::epsilon() / 2.0;
  const double* vertices[3] = {p, q, r};
  double sides[3][3], squares[3];
  for (int k = 0; k < 3; ++k) {
    const double* a = vertices[(k + 1) % 3];
    const double* b = vertices[(k + 2) % 3];
    for (int i = 0; i < 3; ++i) sides[k][i] = b[i] - a[i];
    squares[k] = sides[k][0] * sides[k][0] + sides[k][1] * sides[k][1] +
                 sides[k][2] * sides[k][2];
  }
  const double cross[3] = {sides[0][1] * sides[1][2] - sides[0][2] * sides[1][1],
                           sides[0][2] * sides[1][0] - sides[0][0] * sides[1][2],
                           sides[0][0] * sides[1][1] - sides[0][1] * sides[1][0]};
  const double denominator =
      4.0 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
  const double limit = 0.5 * h * h * (1.0 + tau);
  if (denominator > 0.0) {
    const double squared = squares[0] * squares[1] * squares[2] / denominator;
    const double relative = 64.0 * u * (1.0 + 4.0 * squares[0] * squares[1] / denominator);
    if (std::isfinite(relative) && relative <= 0.25) {
      const double margin = 2.0 * relative * squared + 16.0 * u * limit;
      const double gap = squared - limit;
      if (std::isfinite(margin) && std::abs(gap) > margin) return gap > 0.0;
    }
  }
  std::array<std::array<Expansion, 3>, 3> exact;
  for (int k = 0; k < 3; ++k) {
    const double* a = vertices[(k + 1) % 3];
    const double* b = vertices[(k + 2) % 3];
    for (int i = 0; i < 3; ++i) exact[k][i] = Expansion::difference(b[i], a[i]);
  }
  Expansion numerator(1.0);
  for (int k = 0; k < 3; ++k)
    numerator = numerator * (exact[k][0] * exact[k][0] + exact[k][1] * exact[k][1] +
                             exact[k][2] * exact[k][2]);
  const auto& e0 = exact[0];
  const auto& e1 = exact[1];
  const Expansion normal[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2],
                               e0[0] * e1[1] - e0[1] * e1[0]};
  const Expansion area =
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
  if (area.is_zero()) return true;
  // tau is a double, so 1 + tau is the exact two-term sum.
  const Expansion bound = Expansion::product(h, h) * Expansion::sum(1.0, tau) * 2.0;
  return (numerator - bound * area).sign() > 0;
}

}  // namespace phx::mc
