//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// The certified bounds of directed_interval.hpp against exact arithmetic.
// Norms: the squares of the bounds bracket the exact sum of squares. Angles:
// with the exact dot product d, the exact squared cross product c2 and the
// exact product n2 of the squared lengths of two binary64 vectors, a bound A
// of their angle is proved by an enclosure of sin(A) or cos(A) from the
// elementary kernel, whichever is well conditioned:
//   acute, at most pi/4:      sin(A)^2 n2 >= c2
//   acute, above pi/4:        cos(A) <= 0, or cos(A)^2 n2 <= d^2
//   obtuse, below 3 pi/4:     cos(A) <= 0 and cos(A)^2 n2 >= d^2
//   obtuse, at least 3 pi/4:  sin(A)^2 n2 <= c2
// The inputs come from a fixed integer generator and the bounds use no
// platform libm, so every platform checks the same numbers.
#include <array>
#include <cmath>
#include <cstdint>

#include "check.hpp"
#include "directed_interval.hpp"
#include "dyadic.hpp"
#include "floating_point_mode.hpp"

namespace {

using phx::mc::Dyadic;
using phx::mc::remesh::IntervalVector;
using Vector = std::array<double, 3>;

struct Generator {
  std::uint64_t state;
  // Uniform in [0, 1) with 53 bits.
  double unit() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) * 0x1p-53;
  }
  double symmetric() {
    const double value = unit();
    return 2.0 * value - 1.0;
  }
};

Dyadic square(const Dyadic& value) { return value * value; }
bool at_most(const Dyadic& a, const Dyadic& b) { return (b - a).sign() >= 0; }

void norms() {
  Generator generator{20261118};
  for (int index = 0; index < 20000; ++index) {
    // Magnitudes spread over 2^-300 .. 2^300, some of them zero.
    double parts[3];
    Dyadic exact;
    for (double& part : parts) {
      const double mantissa = generator.unit();
      const double exponent = generator.unit();
      const double choice = generator.unit();
      part = choice < 0.1 ? 0.0 : std::ldexp(mantissa, static_cast<int>(600.0 * exponent) - 300);
      exact = exact + square(Dyadic(part));
    }
    const double lower = phx::mc::geometry::norm_lower(parts);
    const double upper = phx::mc::geometry::norm_upper(parts);
    PHX_CHECK(lower >= 0.0 && lower <= upper);
    PHX_CHECK(at_most(square(Dyadic(lower)), exact));
    PHX_CHECK(at_most(exact, square(Dyadic(upper))));
    // The bounds stay close: a few units in the last place apart.
    if (lower > 0x1p-200) PHX_CHECK(upper <= lower * (1.0 + 0x1p-48));
  }
  // An enclosure that holds zero in a component contributes nothing below.
  const IntervalVector straddling = {phx::mc::remesh::Interval{-1.0, 2.0},
                                     phx::mc::remesh::Interval{3.0, 4.0},
                                     phx::mc::remesh::Interval{-5.0, -4.0}};
  PHX_CHECK(phx::mc::remesh::norm_lower(straddling) <= 5.0);
  PHX_CHECK(phx::mc::remesh::norm_lower(straddling) > 4.9999999);
  PHX_CHECK(phx::mc::remesh::norm_upper(straddling) >= std::sqrt(45.0));
  PHX_CHECK(phx::mc::remesh::norm_upper(straddling) < 6.7082040);
}

IntervalVector exactly(const Vector& v) {
  return {phx::mc::remesh::point(v[0]), phx::mc::remesh::point(v[1]),
          phx::mc::remesh::point(v[2])};
}

// Whether `bound` is proved to be at least the angle between u and v.
bool proves(double bound, const Vector& u, const Vector& v) {
  if (bound >= phx::mc::remesh::kPiUpper) return true;
  const Dyadic ux(u[0]), uy(u[1]), uz(u[2]), vx(v[0]), vy(v[1]), vz(v[2]);
  const Dyadic dot = (ux * vx + uy * vy) + uz * vz;
  const Dyadic cross2 = (square(uy * vz - uz * vy) + square(uz * vx - ux * vz)) +
                        square(ux * vy - uy * vx);
  const Dyadic lengths2 = ((square(ux) + square(uy)) + square(uz)) *
                          ((square(vx) + square(vy)) + square(vz));
  const Dyadic dot2 = square(dot);
  const bool shallow = at_most(cross2, dot2);  // within pi/4 of the line of u
  const phx::mc::elementary::Enclosure sine = phx::mc::elementary::sin(bound, bound);
  const phx::mc::elementary::Enclosure cosine = phx::mc::elementary::cos(bound, bound);
  if (dot.sign() >= 0) {
    if (shallow) {
      // Beyond pi/2 the bound holds every acute angle.
      return cosine.upper <= 0.0 ||
             (sine.lower > 0.0 && at_most(cross2, square(Dyadic(sine.lower)) * lengths2));
    }
    return cosine.upper <= 0.0 || at_most(square(Dyadic(cosine.upper)) * lengths2, dot2);
  }
  if (!shallow) {
    return cosine.upper <= 0.0 && at_most(dot2, square(Dyadic(cosine.upper)) * lengths2);
  }
  return cosine.upper <= 0.0 && sine.upper >= 0.0 &&
         at_most(square(Dyadic(sine.upper)) * lengths2, cross2);
}

void angles() {
  Generator generator{20261119};
  int acute = 0, obtuse = 0, narrow = 0;
  for (int index = 0; index < 20000; ++index) {
    Vector u, v;
    for (double& part : u) part = generator.symmetric();
    const int kind = index % 5;
    const double scale = std::ldexp(1.0, static_cast<int>(120.0 * generator.unit()) - 60);
    for (std::size_t d = 0; d < 3; ++d) {
      const double other = generator.symmetric();
      switch (kind) {
        case 0:  // unrelated directions
          v[d] = scale * other;
          break;
        case 1:  // nearly parallel
          v[d] = scale * (u[d] + 0x1p-20 * other);
          break;
        case 2:  // nearly opposite
          v[d] = -scale * (u[d] + 0x1p-20 * other);
          break;
        case 3:  // parallel up to rounding
          v[d] = scale * u[d];
          break;
        default:  // nearly orthogonal: completed below
          v[d] = other;
          break;
      }
    }
    if (kind == 4) {
      // Remove most of the component along u.
      const double along = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) /
                           (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
      for (std::size_t d = 0; d < 3; ++d) v[d] = scale * (v[d] - along * u[d]);
    }
    const double bound = phx::mc::remesh::angle_upper(exactly(u), exactly(v));
    PHX_CHECK(bound >= 0.0);
    PHX_CHECK(proves(bound, u, v));
    // Not wider than a rounding allowance around the angle itself; the
    // reference here is only a plausibility bound on the width.
    const Vector normal = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                           u[0] * v[1] - u[1] * v[0]};
    const double sine =
        std::sqrt((normal[0] * normal[0] + normal[1] * normal[1]) + normal[2] * normal[2]);
    const double cosine = (u[0] * v[0] + u[1] * v[1]) + u[2] * v[2];
    const double lengths = std::sqrt(((u[0] * u[0] + u[1] * u[1]) + u[2] * u[2]) *
                                     ((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]));
    if (cosine > 0.5 * lengths) {
      ++acute;
      // sin(bound) <= sin(angle) (1 + 1e-9) + 1e-12 for an angle below pi/3.
      PHX_CHECK(phx::mc::elementary::sin(bound, bound).lower * lengths <=
                sine * (1.0 + 1.0e-9) + 1.0e-12 * lengths);
      if (sine < 1.0e-4 * lengths) ++narrow;
    } else if (cosine < -0.5 * lengths) {
      ++obtuse;
      PHX_CHECK(phx::mc::elementary::sin(bound, bound).upper * lengths >=
                sine * (1.0 - 1.0e-9) - 1.0e-12 * lengths);
    }
  }
  PHX_CHECK(acute > 4000 && obtuse > 4000 && narrow > 4000);

  // Enclosures that do not bound the angle give the open bound.
  const IntervalVector axis = exactly({1.0, 0.0, 0.0});
  const IntervalVector around = {phx::mc::remesh::Interval{-2.0, -1.0},
                                 phx::mc::remesh::Interval{-1.0, 1.0},
                                 phx::mc::remesh::Interval{-1.0, 1.0}};
  PHX_CHECK(phx::mc::remesh::angle_upper(axis, around) >= phx::mc::remesh::kPiUpper);
  // A box in front of the axis: at most the angle of its far corner.
  const IntervalVector ahead = {phx::mc::remesh::Interval{1.0, 2.0},
                                phx::mc::remesh::Interval{-1.0, 1.0},
                                phx::mc::remesh::Interval{-1.0, 1.0}};
  const double corner = phx::mc::remesh::angle_upper(axis, ahead);
  PHX_CHECK(proves(corner, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}));
  PHX_CHECK(corner < 0.9553166181245094 + 1.0e-12);  // atan(sqrt(2))
}

}  // namespace

int main() {
  const phx::mc::GradualNearestScope mode;
  norms();
  angles();
  return phx::mc::test::finish("test_directed_interval");
}
