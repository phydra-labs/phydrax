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

// The floating stage of orient3d (Shewchuk's stage A expression and error
// bound): `det` the negated orientation determinant, `errbound` its bound.
struct OrientationStage {
  double det;
  double errbound;
};
inline OrientationStage orientation_stage(const double* a, const double* b, const double* c,
                                          const double* d) noexcept {
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const double adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;
  const double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
  const double permanent = (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * std::fabs(adz) +
                           (std::fabs(cdxady) + std::fabs(adxcdy)) * std::fabs(bdz) +
                           (std::fabs(adxbdy) + std::fabs(bdxady)) * std::fabs(cdz);
  constexpr double kEpsilon = 0x1p-53;
  return {det, (7.0 + 56.0 * kEpsilon) * kEpsilon * permanent};
}

// orient3d through its first, floating stage inlined: the clear cases are
// decided here, so only near-degenerate tests reach the exact predicate.
inline int orient3d_filtered(const double* a, const double* b, const double* c,
                             const double* d) noexcept {
  const OrientationStage stage = orientation_stage(a, b, c, d);
  // orient3d is the negated Shewchuk determinant.
  if (stage.det > stage.errbound) return -1;
  if (-stage.det > stage.errbound) return 1;
  return orient3d(a, b, c, d);
}

// The floating stage of insphere: the sign it decides (Shewchuk's stage A
// expression against his bound with the permanent replaced by 25 L X Y Z, X,
// Y, Z the largest coordinate differences per axis and L the largest lift;
// every product in the permanent is at most the matching product of maxima,
// so it is at most 24 L X Y Z up to a relative 1e-15 and the bound stays
// valid), or 0 when the exact predicate must decide.
inline int insphere_stage(const double* a, const double* b, const double* c, const double* d,
                          const double* e) noexcept {
  const double aex = a[0] - e[0], bex = b[0] - e[0], cex = c[0] - e[0], dex = d[0] - e[0];
  const double aey = a[1] - e[1], bey = b[1] - e[1], cey = c[1] - e[1], dey = d[1] - e[1];
  const double aez = a[2] - e[2], bez = b[2] - e[2], cez = c[2] - e[2], dez = d[2] - e[2];
  const double aexbey = aex * bey, bexaey = bex * aey;
  const double ab = aexbey - bexaey;
  const double bexcey = bex * cey, cexbey = cex * bey;
  const double bc = bexcey - cexbey;
  const double cexdey = cex * dey, dexcey = dex * cey;
  const double cd = cexdey - dexcey;
  const double dexaey = dex * aey, aexdey = aex * dey;
  const double da = dexaey - aexdey;
  const double aexcey = aex * cey, cexaey = cex * aey;
  const double ac = aexcey - cexaey;
  const double bexdey = bex * dey, dexbey = dex * bey;
  const double bd = bexdey - dexbey;
  const double abc = aez * bc - bez * ac + cez * ab;
  const double bcd = bez * cd - cez * bd + dez * bc;
  const double cda = cez * da + dez * ac + aez * cd;
  const double dab = dez * ab + aez * bd + bez * da;
  const double alift = aex * aex + aey * aey + aez * aez;
  const double blift = bex * bex + bey * bey + bez * bez;
  const double clift = cex * cex + cey * cey + cez * cez;
  const double dlift = dex * dex + dey * dey + dez * dez;
  const double det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
  const double x = std::fmax(std::fmax(std::fabs(aex), std::fabs(bex)),
                             std::fmax(std::fabs(cex), std::fabs(dex)));
  const double y = std::fmax(std::fmax(std::fabs(aey), std::fabs(bey)),
                             std::fmax(std::fabs(cey), std::fabs(dey)));
  const double z = std::fmax(std::fmax(std::fabs(aez), std::fabs(bez)),
                             std::fmax(std::fabs(cez), std::fabs(dez)));
  const double lift = std::fmax(std::fmax(alift, blift), std::fmax(clift, dlift));
  const double permanent = 25.0 * lift * (x * y * z);
  constexpr double kEpsilon = 0x1p-53;
  const double errbound = (16.0 + 224.0 * kEpsilon) * kEpsilon * permanent;
  // insphere is the negated Shewchuk determinant.
  if (det > errbound) return -1;
  if (-det > errbound) return 1;
  return 0;
}

// insphere through its floating stage inlined (insphere_stage).
inline int insphere_filtered(const double* a, const double* b, const double* c, const double* d,
                             const double* e) noexcept {
  const int sign = insphere_stage(a, b, c, d, e);
  return sign != 0 ? sign : insphere(a, b, c, d, e);
}

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

// Exact signs on points with expansion coordinates (two per point for
// orient2d, three otherwise), with the conventions and the Simulation of
// Simplicity scheme of the binary64 predicates above; used for vertices whose
// exact position is a source construction rather than its binary64 carrier.
int orient2d_sign(const Expansion* a, const Expansion* b, const Expansion* c);
int orient3d_sign(const Expansion* a, const Expansion* b, const Expansion* c,
                  const Expansion* d);
int insphere_sign(const Expansion* a, const Expansion* b, const Expansion* c,
                  const Expansion* d, const Expansion* e);
int orient3d_sos_sign(const Expansion* a, const Expansion* b, const Expansion* c,
                      const Expansion* d, std::int64_t ia, std::int64_t ib, std::int64_t ic,
                      std::int64_t id);
int insphere_sos_sign(const Expansion* a, const Expansion* b, const Expansion* c,
                      const Expansion* d, const Expansion* e, std::int64_t ia,
                      std::int64_t ib, std::int64_t ic, std::int64_t id, std::int64_t ie);

}  // namespace phx::mc
