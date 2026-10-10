//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact geometric predicates.  Every function returns the exact sign of its
// polynomial evaluated on the given binary64 inputs, provided every coordinate
// satisfies coordinate_in_domain() and every weight satisfies
// weight_in_domain().  The domain guarantees that no exact intermediate product
// (degree <= 6 in coordinates) overflows or underflows.
//
// Sign conventions (all consistent with positively oriented simplices):
//   orient2d(a, b, c)            = sign det[b - a, c - a]        (> 0: counterclockwise)
//   orient3d(a, b, c, d)         = sign det[b - a, c - a, d - a] (> 0: right-handed)
//   incircle(a, b, c, d)         > 0 iff d lies inside the circle through a, b, c
//                                  when orient2d(a, b, c) > 0
//   insphere(a, b, c, d, e)      > 0 iff e lies inside the sphere through a..d
//                                  when orient3d(a, b, c, d) > 0
//   power2d/power3d              generalize incircle/insphere to weighted points
//                                  (lifted coordinate |p|^2 - w); > 0 iff the
//                                  query conflicts with the orthogonal circle/sphere.
//
// orient2d/orient3d are evaluated with Shewchuk's adaptive scheme: stage A
// static filter, stage B exact evaluation on the rounded differences with the
// published stage-B bound (and exactness when all difference tails vanish),
// then full expansion arithmetic.  incircle/insphere use the same stages.
// Weighted power tests use the dynamic forward-error filter then expansions.
//
// *_sos variants apply index-ordered Simulation of Simplicity:
//   orient*_sos: Edelsbrunner-Muecke coordinate perturbation
//     p_{i,j} + eps^(2^(d*r_i - j)) with r_i the rank of the point index; the
//     result is nonzero for distinct indices.
//   incircle_sos/insphere_sos/power*_sos: lifted-coordinate perturbation
//     lambda_i + eps^(g(i)) with g increasing in the point index; the result is
//     nonzero unless all points lie on one hyperplane (collinear / coplanar).
#pragma once

#include <cstdint>

#include "expansion.hpp"
#include "filtered.hpp"

namespace phx::mc {

inline constexpr double kCoordinateMinMagnitude = 0x1p-120;
inline constexpr double kCoordinateMaxMagnitude = 0x1p120;
inline constexpr double kWeightMinMagnitude = 0x1p-240;
inline constexpr double kWeightMaxMagnitude = 0x1p240;
inline constexpr int kCoordinateMinExponent = -120;
inline constexpr int kCoordinateMaxExponent = 120;

inline bool coordinate_in_domain(double x) {
  const double magnitude = x < 0.0 ? -x : x;
  return magnitude == 0.0 ||
         (magnitude >= kCoordinateMinMagnitude && magnitude <= kCoordinateMaxMagnitude);
}

inline bool weight_in_domain(double w) {
  const double magnitude = w < 0.0 ? -w : w;
  return magnitude == 0.0 ||
         (magnitude >= kWeightMinMagnitude && magnitude <= kWeightMaxMagnitude);
}

int orient2d(const double* a, const double* b, const double* c);
int orient3d(const double* a, const double* b, const double* c, const double* d);
int incircle(const double* a, const double* b, const double* c, const double* d);
int insphere(const double* a, const double* b, const double* c, const double* d, const double* e);
int insphere_expansion(const Expansion* a, const Expansion* b, const Expansion* c,
                       const Expansion* d, const Expansion* e);
int power2d(const double* a, const double* b, const double* c, const double* d, double wa,
            double wb, double wc, double wd);
int power3d(const double* a, const double* b, const double* c, const double* d, const double* e,
            double wa, double wb, double wc, double wd, double we);

int orient2d_sos(const double* a, const double* b, const double* c, std::int64_t ia,
                 std::int64_t ib, std::int64_t ic);
int orient3d_sos(const double* a, const double* b, const double* c, const double* d,
                 std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id);
int incircle_sos(const double* a, const double* b, const double* c, const double* d,
                 std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id);
int insphere_sos(const double* a, const double* b, const double* c, const double* d,
                 const double* e, std::int64_t ia, std::int64_t ib, std::int64_t ic,
                 std::int64_t id, std::int64_t ie);
int power2d_sos(const double* a, const double* b, const double* c, const double* d, double wa,
                double wb, double wc, double wd, std::int64_t ia, std::int64_t ib,
                std::int64_t ic, std::int64_t id);
int power3d_sos(const double* a, const double* b, const double* c, const double* d,
                const double* e, double wa, double wb, double wc, double wd, double we,
                std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id,
                std::int64_t ie);

// Exact values and filtered approximations of the orientation determinants,
// for constructions whose classification multiplies orientation values.
Expansion orient2d_exact(const double* a, const double* b, const double* c);
Expansion orient3d_exact(const double* a, const double* b, const double* c, const double* d);
Approx orient2d_approx(const double* a, const double* b, const double* c);
Approx orient3d_approx(const double* a, const double* b, const double* c, const double* d);

// Exact original-coordinate squared Hadamard comparison. The requested floor
// is supplied by the owning consumer; `score` is the normalized squared ratio.
int relative_orient3d_exact(const double* a, const double* b, const double* c, const double* d,
                            double relative_floor, double* score);

// Whether three 3D points are collinear: (b - a) x (c - a) vanishes iff the
// orientations of the three coordinate-plane projections vanish, each an exact
// orient2d.
inline bool collinear3d(const double* a, const double* b, const double* c) {
  for (int axis = 0; axis < 3; ++axis) {
    const int i = (axis + 1) % 3;
    const int j = (axis + 2) % 3;
    const double pa[2] = {a[i], a[j]};
    const double pb[2] = {b[i], b[j]};
    const double pc[2] = {c[i], c[j]};
    if (orient2d(pa, pb, pc) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace phx::mc
