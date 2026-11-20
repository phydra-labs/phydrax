//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "predicates.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "wide_integer.hpp"

namespace phx::mc {
namespace {

// Shewchuk's error-bound constants; epsilon is the unit roundoff 2^-53.
constexpr double kEpsilon = 0x1p-53;
constexpr double kCcwErrBoundA = (3.0 + 16.0 * kEpsilon) * kEpsilon;
constexpr double kCcwErrBoundB = (2.0 + 12.0 * kEpsilon) * kEpsilon;
constexpr double kO3dErrBoundA = (7.0 + 56.0 * kEpsilon) * kEpsilon;
constexpr double kO3dErrBoundB = (3.0 + 28.0 * kEpsilon) * kEpsilon;
constexpr double kIccErrBoundA = (10.0 + 96.0 * kEpsilon) * kEpsilon;
constexpr double kIccErrBoundB = (4.0 + 48.0 * kEpsilon) * kEpsilon;
constexpr double kIccErrBoundBCompensated = (4.0 + 128.0 * kEpsilon) * kEpsilon;
constexpr double kIccErrBoundBCompensatedExact = 64.0 * kEpsilon * kEpsilon;
constexpr double kIccErrBoundC = (44.0 + 576.0 * kEpsilon) * kEpsilon * kEpsilon;
constexpr double kResultErrBound = (3.0 + 8.0 * kEpsilon) * kEpsilon;
constexpr double kIspErrBoundA = (16.0 + 224.0 * kEpsilon) * kEpsilon;
constexpr double kIspErrBoundB = (5.0 + 72.0 * kEpsilon) * kEpsilon;
constexpr double kIspErrBoundC = (71.0 + 1408.0 * kEpsilon) * kEpsilon * kEpsilon;
constexpr double kIspErrBoundBCompensated = (5.0 + 256.0 * kEpsilon) * kEpsilon;
constexpr double kIspErrBoundBCompensatedExact = 128.0 * kEpsilon * kEpsilon;

inline int sign_of(double value) { return (value > 0.0) - (value < 0.0); }

// Shewchuk's orient3d value det[a - d; b - d; c - d] on difference rows.
template <class T>
T orient3d_rows(const T* x, const T* y, const T* z) {
  return z[0] * (x[1] * y[2] - x[2] * y[1]) + z[1] * (x[2] * y[0] - x[0] * y[2]) +
         z[2] * (x[0] * y[1] - x[1] * y[0]);
}

// det[[x_k, y_k, lift_k]] for k = 0, 1, 2 (incircle/power2d value).
template <class T>
T lifted2_rows(const T* x, const T* y, const T* lift) {
  return lift[0] * (x[1] * y[2] - x[2] * y[1]) + lift[1] * (x[2] * y[0] - x[0] * y[2]) +
         lift[2] * (x[0] * y[1] - x[1] * y[0]);
}

// Shewchuk's insphere value det[[x_k, y_k, z_k, lift_k]] for k = 0..3.
template <class T>
T lifted3_rows(const T* x, const T* y, const T* z, const T* lift) {
  const T ab = x[0] * y[1] - x[1] * y[0];
  const T bc = x[1] * y[2] - x[2] * y[1];
  const T cd = x[2] * y[3] - x[3] * y[2];
  const T da = x[3] * y[0] - x[0] * y[3];
  const T ac = x[0] * y[2] - x[2] * y[0];
  const T bd = x[1] * y[3] - x[3] * y[1];
  const T abc = z[0] * bc - z[1] * ac + z[2] * ab;
  const T bcd = z[1] * cd - z[2] * bd + z[3] * bc;
  const T cda = z[2] * da + z[3] * ac + z[0] * cd;
  const T dab = z[3] * ab + z[0] * bd + z[1] * da;
  return (lift[3] * abc - lift[2] * dab) + (lift[1] * cda - lift[0] * bcd);
}

// ---------------------------------------------------------------- orient2d

[[gnu::noinline, gnu::cold]]
int orient2d_adapt(const double* a, const double* b, const double* c, double detsum) {
  const double acx = a[0] - c[0];
  const double bcx = b[0] - c[0];
  const double acy = a[1] - c[1];
  const double bcy = b[1] - c[1];
  double detleft, detlefttail, detright, detrighttail;
  two_product(acx, bcy, detleft, detlefttail);
  two_product(acy, bcx, detright, detrighttail);
  double b_components[4];
  two_two_diff(detleft, detlefttail, detright, detrighttail, b_components);
  const Expansion stage_b = Expansion::from_components(b_components, 4);
  const double det = stage_b.estimate();
  const double errbound = kCcwErrBoundB * detsum;
  if (det >= errbound || -det >= errbound) {
    return sign_of(det);
  }
  if (two_diff_tail(a[0], c[0], acx) == 0.0 && two_diff_tail(b[0], c[0], bcx) == 0.0 &&
      two_diff_tail(a[1], c[1], acy) == 0.0 && two_diff_tail(b[1], c[1], bcy) == 0.0) {
    return stage_b.sign();
  }
  const Expansion ex = Expansion::difference(a[0], c[0]);
  const Expansion ey = Expansion::difference(a[1], c[1]);
  const Expansion fx = Expansion::difference(b[0], c[0]);
  const Expansion fy = Expansion::difference(b[1], c[1]);
  return (ex * fy - ey * fx).sign();
}

// Stages B and C of orient3d_shewchuk (out of line: the expansions would
// otherwise enlarge the stack frame of every filtered call).
[[gnu::noinline, gnu::cold]] int orient3d_adapt(const double* a, const double* b,
                                                const double* c, const double* d,
                                                double permanent) {
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const double adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
  const Expansion rx[3] = {Expansion(adx), Expansion(bdx), Expansion(cdx)};
  const Expansion ry[3] = {Expansion(ady), Expansion(bdy), Expansion(cdy)};
  const Expansion rz[3] = {Expansion(adz), Expansion(bdz), Expansion(cdz)};
  const Expansion stage_b = orient3d_rows(rx, ry, rz);
  const double estimate = stage_b.estimate();
  const double errbound_b = kO3dErrBoundB * permanent;
  if (estimate >= errbound_b || -estimate >= errbound_b) {
    return sign_of(estimate);
  }
  const bool exact_differences =
      two_diff_tail(a[0], d[0], adx) == 0.0 && two_diff_tail(b[0], d[0], bdx) == 0.0 &&
      two_diff_tail(c[0], d[0], cdx) == 0.0 && two_diff_tail(a[1], d[1], ady) == 0.0 &&
      two_diff_tail(b[1], d[1], bdy) == 0.0 && two_diff_tail(c[1], d[1], cdy) == 0.0 &&
      two_diff_tail(a[2], d[2], adz) == 0.0 && two_diff_tail(b[2], d[2], bdz) == 0.0 &&
      two_diff_tail(c[2], d[2], cdz) == 0.0;
  if (exact_differences) {
    return stage_b.sign();
  }
  const Expansion fx[3] = {Expansion::difference(a[0], d[0]), Expansion::difference(b[0], d[0]),
                           Expansion::difference(c[0], d[0])};
  const Expansion fy[3] = {Expansion::difference(a[1], d[1]), Expansion::difference(b[1], d[1]),
                           Expansion::difference(c[1], d[1])};
  const Expansion fz[3] = {Expansion::difference(a[2], d[2]), Expansion::difference(b[2], d[2]),
                           Expansion::difference(c[2], d[2])};
  return orient3d_rows(fx, fy, fz).sign();
}

// ---------------------------------------------------------------- orient3d (Shewchuk sign)

int orient3d_shewchuk(const double* a, const double* b, const double* c, const double* d) {
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
  const double errbound_a = kO3dErrBoundA * permanent;
  if (det > errbound_a || -det > errbound_a) {
    return sign_of(det);
  }
  return orient3d_adapt(a, b, c, d, permanent);
}

// ---------------------------------------------------------------- incircle

// Exact p q - r s as a four-component expansion (zeros allowed).
void product_difference(double p, double q, double r, double s, double* h) {
  double pq1, pq0, rs1, rs0;
  two_product(p, q, pq1, pq0);
  two_product(r, s, rs1, rs0);
  two_two_diff(pq1, pq0, rs1, rs0, h);
}

// Exact (x^2 + y^2) m for the four-component minor m; `h` holds at most 32
// components.
int lifted_minor(double x, double y, const double* minor, double* h) {
  double xm[8], xxm[16], ym[8], yym[16];
  const int xlen = scale_expansion_zeroelim(4, minor, x, xm);
  const int xxlen = scale_expansion_zeroelim(xlen, xm, x, xxm);
  const int ylen = scale_expansion_zeroelim(4, minor, y, ym);
  const int yylen = scale_expansion_zeroelim(ylen, ym, y, yym);
  return fast_expansion_sum_zeroelim(xxlen, xxm, yylen, yym, h);
}

// Stage D: the exact sign on the exact coordinate differences.
[[gnu::noinline, gnu::cold]]
int incircle_expansion(const double* const* p) {
  const double* d = p[3];
  Expansion rx[3], ry[3], lift[3];
  for (int k = 0; k < 3; ++k) {
    rx[k] = Expansion::difference(p[k][0], d[0]);
    ry[k] = Expansion::difference(p[k][1], d[1]);
    lift[k] = rx[k] * rx[k] + ry[k] * ry[k];
  }
  return lifted2_rows(rx, ry, lift).sign();
}

// The double-double value hi + lo (an unevaluated sum).
struct DoubleDouble {
  double hi;
  double lo;
};

// x^2 + y^2 to within 3.02 u^2 (x^2 + y^2), u = 2^-53, with |lo| <= 2.01 u
// (x^2 + y^2).
inline DoubleDouble square_sum_dd(double x, double y) {
  double xx, xx_error, yy, yy_error, hi, tail;
  two_product(x, x, xx, xx_error);
  two_product(y, y, yy, yy_error);
  two_sum(xx, yy, hi, tail);
  return {hi, tail + (xx_error + yy_error)};
}

// p q - r s to within 3.02 u^2 Q, Q = |p q| + |r s|, with |lo| <= 2.01 u Q.
inline DoubleDouble product_difference_dd(double p, double q, double r, double s) {
  double pq, pq_error, rs, rs_error, hi, tail;
  two_product(p, q, pq, pq_error);
  two_product(r, s, rs, rs_error);
  two_diff(pq, rs, hi, tail);
  return {hi, tail + (pq_error - rs_error)};
}

// l m without the l.lo m.lo term.
inline DoubleDouble product_dd(DoubleDouble l, DoubleDouble m) {
  double hi, error;
  two_product(l.hi, m.hi, hi, error);
  return {hi, error + (l.hi * m.lo + l.lo * m.hi)};
}

// Stage B on a compensated estimate, ahead of Shewchuk's stage-B expansion:
// E, the double-double evaluation of the determinant D~ of the rounded
// differences, rounded once. With u = 2^-53, Pi the exact permanent of the
// rounded differences (sum of |monomials| of D~) and P >= Pi (1 - u)^7 the
// stage-A permanent: the lifts and minors carry 3.02u^2 relative error, the
// three lifted products 23.4u^2 L_k Q_k, their summation 29u^2 Pi and the
// final rounding u |E|, so |D~ - E| <= 52.4u^2 Pi + u |E|. When every
// difference is exact, D = D~ and |E| > fl(64u^2 P) >= 63.9u^2 P gives |D -
// E| < |E|. Otherwise each of the twelve degree-4 monomials of the exact D
// moves by at most (4u + 22.1u^2) of its rounded value, so |D - D~| <= (4u +
// 22.1u^2) Pi, |D - E| <= (4u + 103u^2) P + u |E|, and |E| > fl((4 + 128u)
// u P) >= (4u + 123.9u^2) P gives |D - E| < |E|. In-domain coordinates and
// their rounded differences are multiples of 2^-172, so every nonzero
// product here is at least 2^-688 in magnitude: none underflows. Returns the
// certified sign, or 0.
int incircle_compensated(const double* const* p, double permanent) {
  const double *a = p[0], *b = p[1], *c = p[2], *d = p[3];
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const DoubleDouble at =
      product_dd(square_sum_dd(adx, ady), product_difference_dd(bdx, cdy, cdx, bdy));
  const DoubleDouble bt =
      product_dd(square_sum_dd(bdx, bdy), product_difference_dd(cdx, ady, adx, cdy));
  const DoubleDouble ct =
      product_dd(square_sum_dd(cdx, cdy), product_difference_dd(adx, bdy, bdx, ady));
  double ab_hi, ab_tail, hi, tail;
  two_sum(at.hi, bt.hi, ab_hi, ab_tail);
  two_sum(ab_hi, ct.hi, hi, tail);
  const double estimate = hi + (((ab_tail + tail) + (at.lo + bt.lo)) + ct.lo);
  const bool exact_differences =
      two_diff_tail(a[0], d[0], adx) == 0.0 && two_diff_tail(a[1], d[1], ady) == 0.0 &&
      two_diff_tail(b[0], d[0], bdx) == 0.0 && two_diff_tail(b[1], d[1], bdy) == 0.0 &&
      two_diff_tail(c[0], d[0], cdx) == 0.0 && two_diff_tail(c[1], d[1], cdy) == 0.0;
  const double errbound =
      (exact_differences ? kIccErrBoundBCompensatedExact : kIccErrBoundBCompensated) * permanent;
  if (estimate > errbound) return 1;
  if (-estimate > errbound) return -1;
  return 0;
}

// Shewchuk's incircleadapt from stage B on. Stage B: the exact determinant
// of the rounded differences (on stack buffers), whose estimate decides
// against iccerrboundB and whose sign is exact when every difference is.
// Stage C: that estimate plus the first-order terms in the difference tails,
// against iccerrboundC. Stage D: the full expansion. Every sign is exact.
[[gnu::noinline, gnu::cold]]
IncircleDecision incircle_expansion_stages(const double* const* p, double permanent) {
  const double *a = p[0], *b = p[1], *c = p[2], *d = p[3];
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  double bc[4], ca[4], ab[4];
  product_difference(bdx, cdy, cdx, bdy, bc);
  product_difference(cdx, ady, adx, cdy, ca);
  product_difference(adx, bdy, bdx, ady, ab);
  double adet[32], bdet[32], cdet[32], abdet[64], fin[96];
  const int alen = lifted_minor(adx, ady, bc, adet);
  const int blen = lifted_minor(bdx, bdy, ca, bdet);
  const int clen = lifted_minor(cdx, cdy, ab, cdet);
  const int ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  const int finlen = fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin);
  double det = 0.0;
  for (int k = 0; k < finlen; ++k) det += fin[k];
  const double errbound_b = kIccErrBoundB * permanent;
  if (det >= errbound_b || -det >= errbound_b) {
    return {sign_of(det), IncircleStage::kBExpansion};
  }
  const double adxtail = two_diff_tail(a[0], d[0], adx);
  const double adytail = two_diff_tail(a[1], d[1], ady);
  const double bdxtail = two_diff_tail(b[0], d[0], bdx);
  const double bdytail = two_diff_tail(b[1], d[1], bdy);
  const double cdxtail = two_diff_tail(c[0], d[0], cdx);
  const double cdytail = two_diff_tail(c[1], d[1], cdy);
  if (adxtail == 0.0 && bdxtail == 0.0 && cdxtail == 0.0 && adytail == 0.0 &&
      bdytail == 0.0 && cdytail == 0.0) {
    return {sign_of(fin[finlen - 1]), IncircleStage::kBExpansion};
  }
  const double errbound_c = kIccErrBoundC * permanent + kResultErrBound * std::fabs(det);
  det += ((adx * adx + ady * ady) *
              ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail)) +
          2.0 * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx)) +
         ((bdx * bdx + bdy * bdy) *
              ((cdx * adytail + ady * cdxtail) - (cdy * adxtail + adx * cdytail)) +
          2.0 * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx)) +
         ((cdx * cdx + cdy * cdy) *
              ((adx * bdytail + bdy * adxtail) - (ady * bdxtail + bdx * adytail)) +
          2.0 * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx));
  if (det >= errbound_c || -det >= errbound_c) {
    return {sign_of(det), IncircleStage::kC};
  }
  return {incircle_expansion(p), IncircleStage::kD};
}

// The stages after stage A: the compensated stage B, then the expansions.
[[gnu::noinline]]
IncircleDecision incircle_adapt(const double* const* p, double permanent) {
  const int sign = incircle_compensated(p, permanent);
  if (sign != 0) return {sign, IncircleStage::kBCompensated};
  return incircle_expansion_stages(p, permanent);
}

// Stage A, then incircle_adapt; inlined into both public entry points.
[[gnu::always_inline]] inline IncircleDecision incircle_staged(const double* a, const double* b,
                                                               const double* c, const double* d) {
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double alift = adx * adx + ady * ady;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double blift = bdx * bdx + bdy * bdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;
  const double clift = cdx * cdx + cdy * cdy;
  const double det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);
  const double permanent = (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * alift +
                           (std::fabs(cdxady) + std::fabs(adxcdy)) * blift +
                           (std::fabs(adxbdy) + std::fabs(bdxady)) * clift;
  const double errbound = kIccErrBoundA * permanent;
  if (det > errbound || -det > errbound) {
    return {sign_of(det), IncircleStage::kA};
  }
  const double* p[4] = {a, b, c, d};
  return incircle_adapt(p, permanent);
}

// ---------------------------------------------------------------- insphere (Shewchuk sign)

// Exact integer evaluation. Binary64 values whose set bits all lie within
// `width` bits of one another are integers times a common power of two;
// the determinant's sign does not depend on that power, so it is evaluated
// on the integers: 128-bit minors, then products and sums of fixed width in
// 64-bit limbs (little endian, two's complement for signed values).

// total += value (subtract: total -= value) for the unsigned `count`-limb
// `value` and the N-limb `total`, modulo 2^(64 N).
template <int N>
void accumulate(std::uint64_t (&total)[N], const std::uint64_t* value, int count, bool subtract) {
  std::uint64_t carry = subtract ? 1 : 0;
  for (int i = 0; i < N; ++i) {
    std::uint64_t limb = i < count ? value[i] : 0;
    if (subtract) limb = ~limb;
    total[i] = phx::mc::add_with_carry(total[i], limb, carry);
  }
}

template <int N>
bool limbs_negative(const std::uint64_t (&value)[N]) {
  return (value[N - 1] >> 63) != 0;
}

// The unsigned product of the `left_count`- and `right_count`-limb values
// into `left_count + right_count` limbs.
void limbs_product(const std::uint64_t* left, int left_count, const std::uint64_t* right,
                   int right_count, std::uint64_t* product) {
  std::fill(product, product + left_count + right_count, std::uint64_t{0});
  for (int i = 0; i < left_count; ++i) {
    std::uint64_t carry = 0;
    for (int j = 0; j < right_count; ++j) {
      std::uint64_t high = 0;
      product[i + j] = phx::mc::multiply_accumulate(left[i], right[j], product[i + j], carry, high);
      carry = high;
    }
    product[i + right_count] = carry;
  }
}

// values[i] = integers[i] * 2^e for one e, with |integers[i]| < 2^width;
// false when the set bits of the values span more than `width` bits (or a
// value is not finite). Read from the binary64 fields: a nonzero value is
// m 2^shift, m an odd integer.
bool common_integers(const double* values, int count, int width, std::int64_t* integers) {
  constexpr std::uint64_t kFraction = (std::uint64_t{1} << 52) - 1;
  int lowest = std::numeric_limits<int>::max();
  int highest = std::numeric_limits<int>::min();
  std::uint64_t significands[16];
  int shifts[16];
  for (int i = 0; i < count; ++i) {
    const auto bits = std::bit_cast<std::uint64_t>(values[i]);
    const auto biased = static_cast<int>((bits >> 52) & 0x7FF);
    if (biased == 0x7FF) return false;
    std::uint64_t significand =
        biased == 0 ? bits & kFraction : (bits & kFraction) | (std::uint64_t{1} << 52);
    int shift = biased == 0 ? -1074 : biased - 1075;
    if (significand != 0) {
      const int zeros = std::countr_zero(significand);
      significand >>= zeros;
      shift += zeros;
      // The value lies below 2^highest; its lowest set bit is 2^lowest.
      lowest = std::min(lowest, shift);
      highest = std::max(highest, shift + static_cast<int>(std::bit_width(significand)));
    }
    significands[i] = significand;
    shifts[i] = shift;
  }
  if (highest == std::numeric_limits<int>::min()) {
    std::fill(integers, integers + count, std::int64_t{0});
    return true;
  }
  if (highest - lowest > width) return false;
  for (int i = 0; i < count; ++i) {
    const auto magnitude = significands[i] == 0
                               ? std::int64_t{0}
                               : static_cast<std::int64_t>(significands[i] << (shifts[i] - lowest));
    integers[i] = std::signbit(values[i]) ? -magnitude : magnitude;
  }
  return true;
}

// The 128-bit pieces of the integer insphere rows. Compilers with a native
// 128-bit integer evaluate them in it; elsewhere the wide-integer limb helpers
// give the same two's-complement bits.
#if defined(__SIZEOF_INT128__)
__extension__ using SignedWide = __int128;
__extension__ using UnsignedWide = unsigned __int128;

void store_wide(UnsignedWide wide, std::uint64_t (&value)[2]) {
  value[0] = static_cast<std::uint64_t>(wide);
  value[1] = static_cast<std::uint64_t>(wide >> 64);
}
#else
// value = a * b as a two's-complement 128-bit value, exact for |a|, |b| < 2^63.
void signed_product(std::int64_t a, std::int64_t b, std::uint64_t (&value)[2]) {
  const std::uint64_t ua = a < 0 ? ~static_cast<std::uint64_t>(a) + 1 : static_cast<std::uint64_t>(a);
  const std::uint64_t ub = b < 0 ? ~static_cast<std::uint64_t>(b) + 1 : static_cast<std::uint64_t>(b);
  value[0] = phx::mc::multiply_wide(ua, ub, value[1]);
  if ((a < 0) != (b < 0)) {
    std::uint64_t zero[2]{};
    accumulate(zero, value, 2, true);
    value[0] = zero[0];
    value[1] = zero[1];
  }
}
#endif

// value = a d - b c in two's-complement 128-bit limbs, exact for |a|..|d| < 2^62.
void signed_minor(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d,
                  std::uint64_t (&value)[2]) {
#if defined(__SIZEOF_INT128__)
  store_wide(static_cast<UnsignedWide>(static_cast<SignedWide>(a) * d -
                                       static_cast<SignedWide>(b) * c),
             value);
#else
  std::uint64_t right[2];
  signed_product(a, d, value);
  signed_product(b, c, right);
  accumulate(value, right, 2, true);
#endif
}

// value = x^2 + y^2 + z^2 as an unsigned 128-bit value, exact for |x|, |y|, |z| < 2^62.
void square_sum(std::int64_t x, std::int64_t y, std::int64_t z, std::uint64_t (&value)[2]) {
#if defined(__SIZEOF_INT128__)
  store_wide(static_cast<UnsignedWide>(static_cast<SignedWide>(x) * x +
                                       static_cast<SignedWide>(y) * y +
                                       static_cast<SignedWide>(z) * z),
             value);
#else
  value[0] = 0;
  value[1] = 0;
  for (const std::int64_t coordinate : {x, y, z}) {
    std::uint64_t square[2];
    signed_product(coordinate, coordinate, square);
    accumulate(value, square, 2, false);
  }
#endif
}

// The magnitude of a two's-complement 128-bit value; true when it is negative.
bool wide_magnitude(const std::uint64_t (&value)[2], std::uint64_t (&magnitude)[2]) {
  const bool negative = limbs_negative(value);
#if defined(__SIZEOF_INT128__)
  const UnsignedWide bits = (static_cast<UnsignedWide>(value[1]) << 64) | value[0];
  store_wide(negative ? ~bits + 1 : bits, magnitude);
#else
  magnitude[0] = 0;
  magnitude[1] = 0;
  accumulate(magnitude, value, 2, negative);
#endif
  return negative;
}

// Sign of lifted3_rows on integer rows below 2^62 in magnitude: minors below
// 2^125 and lifts below 2^126 in 128 bits, cofactors below 2^189 in three
// limbs, the determinant below 2^317 in five.
int insphere_integer_rows(const std::int64_t* x, const std::int64_t* y, const std::int64_t* z) {
  // minor(i, j) = x_i y_j - x_j y_i, below 2^125 in magnitude.
  const auto minor = [&](int i, int j, std::uint64_t (&value)[2]) {
    signed_minor(x[i], x[j], y[i], y[j], value);
  };
  std::uint64_t ab[2], bc[2], cd[2], da[2], ac[2], bd[2];
  minor(0, 1, ab);
  minor(1, 2, bc);
  minor(2, 3, cd);
  minor(3, 0, da);
  minor(0, 2, ac);
  minor(1, 3, bd);
  // total += z[k] m (minus: -=).
  const auto add_term = [&](std::uint64_t (&total)[3], int k, const std::uint64_t (&m)[2],
                            bool minus) {
    const std::uint64_t row =
        z[k] < 0 ? ~static_cast<std::uint64_t>(z[k]) + 1 : static_cast<std::uint64_t>(z[k]);
    std::uint64_t right[2];
    const bool negative = wide_magnitude(m, right);
    std::uint64_t product[3];
    limbs_product(&row, 1, right, 2, product);
    accumulate(total, product, 3, ((z[k] < 0) != negative) != minus);
  };
  // abc = z0 bc - z1 ac + z2 ab; bcd = z1 cd - z2 bd + z3 bc;
  // cda = z2 da + z3 ac + z0 cd; dab = z3 ab + z0 bd + z1 da.
  std::uint64_t abc[3]{}, bcd[3]{}, cda[3]{}, dab[3]{};
  add_term(abc, 0, bc, false);
  add_term(abc, 1, ac, true);
  add_term(abc, 2, ab, false);
  add_term(bcd, 1, cd, false);
  add_term(bcd, 2, bd, true);
  add_term(bcd, 3, bc, false);
  add_term(cda, 2, da, false);
  add_term(cda, 3, ac, false);
  add_term(cda, 0, cd, false);
  add_term(dab, 3, ab, false);
  add_term(dab, 0, bd, false);
  add_term(dab, 1, da, false);
  // det = (l3 abc - l2 dab) + (l1 cda - l0 bcd).
  std::uint64_t det[5]{};
  const auto add_lifted = [&](int k, const std::uint64_t (&cofactor)[3], bool minus) {
    // x^2 + y^2 + z^2 below 2^126.
    std::uint64_t left[2];
    square_sum(x[k], y[k], z[k], left);
    const bool negative = limbs_negative(cofactor);
    std::uint64_t size[3]{cofactor[0], cofactor[1], cofactor[2]};
    if (negative) {
      std::uint64_t zero[3]{};
      accumulate(zero, size, 3, true);
      std::copy_n(zero, 3, size);
    }
    std::uint64_t product[5];
    limbs_product(left, 2, size, 3, product);
    accumulate(det, product, 5, negative != minus);
  };
  add_lifted(3, abc, false);
  add_lifted(2, dab, true);
  add_lifted(1, cda, false);
  add_lifted(0, bcd, true);
  if (limbs_negative(det)) return -1;
  return std::any_of(det, det + 5, [](std::uint64_t limb) { return limb != 0; }) ? 1 : 0;
}

// The exact insphere sign by integers: on the coordinates (differences then
// below 2^62) or on exact differences; false when neither fits.
bool insphere_integer(const double* const* p, const double* x, const double* y,
                      const double* z, bool exact_differences, int& sign) {
  double coordinates[15];
  for (int k = 0; k < 5; ++k) {
    for (int axis = 0; axis < 3; ++axis) coordinates[3 * k + axis] = p[k][axis];
  }
  std::int64_t integers[15];
  std::int64_t ix[4], iy[4], iz[4];
  if (common_integers(coordinates, 15, 61, integers)) {
    for (int k = 0; k < 4; ++k) {
      ix[k] = integers[3 * k] - integers[12];
      iy[k] = integers[3 * k + 1] - integers[13];
      iz[k] = integers[3 * k + 2] - integers[14];
    }
  } else {
    if (!exact_differences) return false;
    double differences[12];
    for (int k = 0; k < 4; ++k) {
      differences[k] = x[k];
      differences[4 + k] = y[k];
      differences[8 + k] = z[k];
    }
    if (!common_integers(differences, 12, 62, integers)) return false;
    for (int k = 0; k < 4; ++k) {
      ix[k] = integers[k];
      iy[k] = integers[4 + k];
      iz[k] = integers[8 + k];
    }
  }
  sign = insphere_integer_rows(ix, iy, iz);
  return true;
}

InsphereDecision insphere_expansion(const double* const* p, const double* x, const double* y,
                                    const double* z, bool exact_differences, double permanent);

// x^2 + y^2 + z^2 as a double-double (squares exact, then two exact sums):
// the tail sums five terms of total at most 3.01u L (L = x^2 + y^2 + z^2) in
// four additions, so the value is within 12.1u^2 L, with |lo| <= 3.02u L and
// |hi| <= (1 + 3.01u) L.
inline DoubleDouble square_sum3_dd(double x, double y, double z) {
  double xx, xx_error, yy, yy_error, zz, zz_error, xy, xy_tail, hi, tail;
  two_product(x, x, xx, xx_error);
  two_product(y, y, yy, yy_error);
  two_product(z, z, zz, zz_error);
  two_sum(xx, yy, xy, xy_tail);
  two_sum(xy, zz, hi, tail);
  return {hi, (xy_tail + tail) + ((xx_error + yy_error) + zz_error)};
}

// z m for a double-double m.
inline DoubleDouble scale_dd(double z, DoubleDouble m) {
  double hi, error;
  two_product(z, m.hi, hi, error);
  return {hi, error + z * m.lo};
}

// u + v + w for double-doubles: exact sums of the leading parts, then the
// tails.
inline DoubleDouble sum3_dd(DoubleDouble u, DoubleDouble v, DoubleDouble w) {
  double uv, uv_tail, hi, tail;
  two_sum(u.hi, v.hi, uv, uv_tail);
  two_sum(uv, w.hi, hi, tail);
  return {hi, ((uv_tail + tail) + (u.lo + v.lo)) + w.lo};
}

inline DoubleDouble negate_dd(DoubleDouble m) { return {-m.hi, -m.lo}; }

// Compensated stage B of insphere, ahead of the integer and expansion
// stages: E, the double-double evaluation of the determinant D~ of the
// rounded differences x_k, y_k, z_k (rows a..d minus e) rounded once, as
// (dlift abc - clift dab) + (blift cda - alift bcd). u = 2^-53; Pi is the
// exact permanent of D~ (sum of the |monomials| of its expansion, each of
// degree 5: a lift square times z times a 2x2 product), and P the stage-A
// permanent, which evaluates Pi along paths of at most 12 roundings, each
// rounding down by at most (1 - u): Pi <= (1 + 12.01u) P. Error of each
// double-double value V against its exact value, in units of its own
// permanent R:
// - 2x2 minor (product_difference_dd): 3.02u^2 R; |lo| <= 2.01u R and
//   |hi| <= (1 + 2.01u) R.
// - z m (scale_dd): the inherited 3.02u^2 R, z m.lo rounds by 2.01u^2 R and
//   the tail sum (at most 1.0001u R + 2.01u R) by 3.02u^2 R: 8.05u^2 R, with
//   |lo| <= 3.02u R and |hi| <= (1 + 3.02u) R.
// - 3x3 minor T (sum3_dd of three scaled minors, R_T the sum of their R):
//   the two_sum tails total at most 2.01u R_T and the input tails 3.02u R_T;
//   the three tail additions round by (2.01 + 5.03 + 5.04)u^2 R_T and the
//   inputs carry 8.05u^2 R_T: 23.3u^2 R_T, with |lo| <= 5.04u R_T and |hi| <=
//   (1 + 5.03u) R_T.
// - lift L (square_sum3_dd): 12.1u^2 L, |lo| <= 3.02u L, |hi| <= (1 +
//   3.01u) L.
// - lifted term W = L T (product_dd, W its permanent L R_T): the inherited
//   L 23.3u^2 R_T + (1 + 23.3u^2) R_T 12.1u^2 L <= 35.4u^2 W; the dropped
//   L.lo T.lo <= 15.23u^2 W; L.hi T.lo (<= 5.06u W) and L.lo T.hi (<= 3.04u W)
//   round by 8.1u^2 W, their sum by 8.11u^2 W, and adding the product error
//   (<= 1.0001u W) by 9.12u^2 W: 76.0u^2 W, with |lo| <= 9.13u W and |hi| <=
//   (1 + 9.06u) W.
// - D~ (four lifted terms, Pi = sum of their W): the three two_sum tails
//   total at most 3.0001u Pi and the term tails 9.13u Pi; the six tail
//   additions round by (2.0 + 3.0 + 9.13 + 9.14 + 12.14)u^2 Pi <= 35.5u^2
//   Pi, and the terms carry 76.0u^2 Pi: |D~ - (hi + lo)| <= 111.5u^2 Pi, and
//   E = fl(hi + lo) adds u |E|.
// So |D~ - E| <= 111.5u^2 (1 + 12.01u) P + u |E| <= 111.6u^2 P + u |E|.
// When every difference is exact, D = D~ and |E| > fl(128u^2 P) = 128u^2 P
// gives |D - E| < |E|. Otherwise each true difference is its rounded value
// times (1 + d), |d| <= u / (1 - u), so each degree-5 monomial of the exact
// D moves by at most (5u + 15.1u^2) of its rounded value: |D - D~| <= (5u +
// 15.1u^2) Pi and |D - E| <= (5u + 186.7u^2) P + u |E|, while |E| >
// fl((5 + 256u)u P) >= (5u + 251u^2) P gives |D - E| < |E|. In-domain
// coordinates and their rounded differences are multiples of 2^-172, so
// every nonzero product here is a multiple of 2^-860: none underflows.
// Returns the certified sign (Shewchuk's: positive when e lies outside), or
// 0.
int insphere_compensated(const double* x, const double* y, const double* z,
                         bool exact_differences, double permanent) {
  const DoubleDouble ab = product_difference_dd(x[0], y[1], x[1], y[0]);
  const DoubleDouble bc = product_difference_dd(x[1], y[2], x[2], y[1]);
  const DoubleDouble cd = product_difference_dd(x[2], y[3], x[3], y[2]);
  const DoubleDouble da = product_difference_dd(x[3], y[0], x[0], y[3]);
  const DoubleDouble ac = product_difference_dd(x[0], y[2], x[2], y[0]);
  const DoubleDouble bd = product_difference_dd(x[1], y[3], x[3], y[1]);
  const DoubleDouble abc = sum3_dd(scale_dd(z[0], bc), scale_dd(-z[1], ac), scale_dd(z[2], ab));
  const DoubleDouble bcd = sum3_dd(scale_dd(z[1], cd), scale_dd(-z[2], bd), scale_dd(z[3], bc));
  const DoubleDouble cda = sum3_dd(scale_dd(z[2], da), scale_dd(z[3], ac), scale_dd(z[0], cd));
  const DoubleDouble dab = sum3_dd(scale_dd(z[3], ab), scale_dd(z[0], bd), scale_dd(z[1], da));
  const DoubleDouble w0 = product_dd(square_sum3_dd(x[3], y[3], z[3]), abc);
  const DoubleDouble w1 = product_dd(square_sum3_dd(x[2], y[2], z[2]), negate_dd(dab));
  const DoubleDouble w2 = product_dd(square_sum3_dd(x[1], y[1], z[1]), cda);
  const DoubleDouble w3 = product_dd(square_sum3_dd(x[0], y[0], z[0]), negate_dd(bcd));
  double s1, t1, s2, t2, hi, t3;
  two_sum(w0.hi, w1.hi, s1, t1);
  two_sum(s1, w2.hi, s2, t2);
  two_sum(s2, w3.hi, hi, t3);
  const double estimate = hi + (((t1 + t2) + t3) + ((w0.lo + w1.lo) + (w2.lo + w3.lo)));
  const double errbound =
      (exact_differences ? kIspErrBoundBCompensatedExact : kIspErrBoundBCompensated) * permanent;
  if (estimate > errbound) return 1;
  if (-estimate > errbound) return -1;
  return 0;
}

[[gnu::noinline, gnu::cold]]
InsphereDecision insphere_adapt(const double* const* p, double permanent) {
  const double* e = p[4];
  double x[4], y[4], z[4];
  bool exact_differences = true;
  for (int k = 0; k < 4; ++k) {
    x[k] = p[k][0] - e[0];
    y[k] = p[k][1] - e[1];
    z[k] = p[k][2] - e[2];
    exact_differences = exact_differences && two_diff_tail(p[k][0], e[0], x[k]) == 0.0 &&
                        two_diff_tail(p[k][1], e[1], y[k]) == 0.0 &&
                        two_diff_tail(p[k][2], e[2], z[k]) == 0.0;
  }
  if (const int sign = insphere_compensated(x, y, z, exact_differences, permanent); sign != 0) {
    return {sign, InsphereStage::kBCompensated};
  }
  int integer_sign = 0;
  if (insphere_integer(p, x, y, z, exact_differences, integer_sign)) {
    return {integer_sign, InsphereStage::kInteger};
  }
  return insphere_expansion(p, x, y, z, exact_differences, permanent);
}

// Component bounds of insphere's stage-B expansions (Shewchuk's
// insphereadapt buffers): a 2x2 minor is a product difference of 4
// components, a 3x3 minor the sum of three minors scaled by one coordinate
// (3 * 2 * 4), a lifted term that minor scaled twice by each of the three
// coordinates and summed (3 * 2 * 2 * 24), and the determinant the sum of
// four lifted terms.
constexpr int kInspherePairTerms = 4;
constexpr int kInsphereTripleTerms = 3 * 2 * kInspherePairTerms;
constexpr int kInsphereLiftedTerms = 3 * 2 * 2 * kInsphereTripleTerms;
constexpr int kInsphereStageBTerms = 4 * kInsphereLiftedTerms;
static_assert(kInsphereTripleTerms == 24 && kInsphereLiftedTerms == 288 &&
              kInsphereStageBTerms == 1152);

// Exact u p + v q + w r for the 2x2 minors p, q, r; `h` holds
// kInsphereTripleTerms components.
int insphere_triple(double u, const double* p, double v, const double* q, double w,
                    const double* r, double* h) {
  double up[2 * kInspherePairTerms], vq[2 * kInspherePairTerms], wr[2 * kInspherePairTerms];
  double sum[4 * kInspherePairTerms];
  const int uplen = scale_expansion_zeroelim(kInspherePairTerms, p, u, up);
  const int vqlen = scale_expansion_zeroelim(kInspherePairTerms, q, v, vq);
  const int wrlen = scale_expansion_zeroelim(kInspherePairTerms, r, w, wr);
  const int sumlen = fast_expansion_sum_zeroelim(uplen, up, vqlen, vq, sum);
  return fast_expansion_sum_zeroelim(sumlen, sum, wrlen, wr, h);
}

// Exact sign (x^2 + y^2 + z^2) m for the 3x3 minor m of `length` components
// (sign +-1, so sign x is exact); `h` holds kInsphereLiftedTerms components.
int insphere_lifted(double x, double y, double z, double sign, int length, const double* minor,
                    double* h) {
  double once[2 * kInsphereTripleTerms];
  double xx[4 * kInsphereTripleTerms], yy[4 * kInsphereTripleTerms], zz[4 * kInsphereTripleTerms];
  double xy[8 * kInsphereTripleTerms];
  int oncelen = scale_expansion_zeroelim(length, minor, x, once);
  const int xxlen = scale_expansion_zeroelim(oncelen, once, sign * x, xx);
  oncelen = scale_expansion_zeroelim(length, minor, y, once);
  const int yylen = scale_expansion_zeroelim(oncelen, once, sign * y, yy);
  oncelen = scale_expansion_zeroelim(length, minor, z, once);
  const int zzlen = scale_expansion_zeroelim(oncelen, once, sign * z, zz);
  const int xylen = fast_expansion_sum_zeroelim(xxlen, xx, yylen, yy, xy);
  return fast_expansion_sum_zeroelim(xylen, xy, zzlen, zz, h);
}

// The expansion stages of insphere_adapt (values whose bits span too much).
// Shewchuk's insphereadapt from stage B on, on stack buffers. Stage B: the
// exact determinant of the rounded differences, whose estimate decides
// against isperrboundB and whose sign is exact when every difference is.
// Stage C: that estimate plus the first-order terms in the difference
// tails, against isperrboundC. Stage D: the exact determinant of the exact
// differences. Every sign is exact.
[[gnu::noinline, gnu::cold]]
InsphereDecision insphere_expansion(const double* const* p, const double* x, const double* y,
                                    const double* z, bool exact_differences, double permanent) {
  const double* e = p[4];
  double ab[kInspherePairTerms], bc[kInspherePairTerms], cd[kInspherePairTerms];
  double da[kInspherePairTerms], ac[kInspherePairTerms], bd[kInspherePairTerms];
  product_difference(x[0], y[1], x[1], y[0], ab);
  product_difference(x[1], y[2], x[2], y[1], bc);
  product_difference(x[2], y[3], x[3], y[2], cd);
  product_difference(x[3], y[0], x[0], y[3], da);
  product_difference(x[0], y[2], x[2], y[0], ac);
  product_difference(x[1], y[3], x[3], y[1], bd);
  double minor[kInsphereTripleTerms];
  double terms[4][kInsphereLiftedTerms];
  int lengths[4];
  // (lift3 abc - lift2 dab) + (lift1 cda - lift0 bcd), as lifted3_rows.
  int length = insphere_triple(z[1], cd, -z[2], bd, z[3], bc, minor);
  lengths[0] = insphere_lifted(x[0], y[0], z[0], -1.0, length, minor, terms[0]);
  length = insphere_triple(z[2], da, z[3], ac, z[0], cd, minor);
  lengths[1] = insphere_lifted(x[1], y[1], z[1], 1.0, length, minor, terms[1]);
  length = insphere_triple(z[3], ab, z[0], bd, z[1], da, minor);
  lengths[2] = insphere_lifted(x[2], y[2], z[2], -1.0, length, minor, terms[2]);
  length = insphere_triple(z[0], bc, -z[1], ac, z[2], ab, minor);
  lengths[3] = insphere_lifted(x[3], y[3], z[3], 1.0, length, minor, terms[3]);
  double first[2 * kInsphereLiftedTerms], second[2 * kInsphereLiftedTerms];
  double total[kInsphereStageBTerms];
  const int firstlen =
      fast_expansion_sum_zeroelim(lengths[0], terms[0], lengths[1], terms[1], first);
  const int secondlen =
      fast_expansion_sum_zeroelim(lengths[2], terms[2], lengths[3], terms[3], second);
  const int totallen = fast_expansion_sum_zeroelim(firstlen, first, secondlen, second, total);
  double estimate = 0.0;
  for (int k = 0; k < totallen; ++k) estimate += total[k];
  const double errbound = kIspErrBoundB * permanent;
  if (estimate >= errbound || -estimate >= errbound) {
    return {sign_of(estimate), InsphereStage::kB};
  }
  if (exact_differences) {
    return {sign_of(total[totallen - 1]), InsphereStage::kBExact};
  }
  // Stage C: the stage-B estimate plus the first-order terms in the
  // difference tails, against isperrboundC (Shewchuk's insphereadapt; ab3
  // and so on are the leading components of the 2x2 minors).
  const double aex = x[0], bex = x[1], cex = x[2], dex = x[3];
  const double aey = y[0], bey = y[1], cey = y[2], dey = y[3];
  const double aez = z[0], bez = z[1], cez = z[2], dez = z[3];
  const double aextail = two_diff_tail(p[0][0], e[0], aex);
  const double aeytail = two_diff_tail(p[0][1], e[1], aey);
  const double aeztail = two_diff_tail(p[0][2], e[2], aez);
  const double bextail = two_diff_tail(p[1][0], e[0], bex);
  const double beytail = two_diff_tail(p[1][1], e[1], bey);
  const double beztail = two_diff_tail(p[1][2], e[2], bez);
  const double cextail = two_diff_tail(p[2][0], e[0], cex);
  const double ceytail = two_diff_tail(p[2][1], e[1], cey);
  const double ceztail = two_diff_tail(p[2][2], e[2], cez);
  const double dextail = two_diff_tail(p[3][0], e[0], dex);
  const double deytail = two_diff_tail(p[3][1], e[1], dey);
  const double deztail = two_diff_tail(p[3][2], e[2], dez);
  const double ab3 = ab[3], bc3 = bc[3], cd3 = cd[3], da3 = da[3], ac3 = ac[3], bd3 = bd[3];
  const double abeps = (aex * beytail + bey * aextail) - (aey * bextail + bex * aeytail);
  const double bceps = (bex * ceytail + cey * bextail) - (bey * cextail + cex * beytail);
  const double cdeps = (cex * deytail + dey * cextail) - (cey * dextail + dex * ceytail);
  const double daeps = (dex * aeytail + aey * dextail) - (dey * aextail + aex * deytail);
  const double aceps = (aex * ceytail + cey * aextail) - (aey * cextail + cex * aeytail);
  const double bdeps = (bex * deytail + dey * bextail) - (bey * dextail + dex * beytail);
  const double errbound_c = kIspErrBoundC * permanent + kResultErrBound * std::fabs(estimate);
  const double det =
      estimate +
      ((((bex * bex + bey * bey + bez * bez) *
             ((cez * daeps + dez * aceps + aez * cdeps) +
              (ceztail * da3 + deztail * ac3 + aeztail * cd3)) +
         (dex * dex + dey * dey + dez * dez) *
             ((aez * bceps - bez * aceps + cez * abeps) +
              (aeztail * bc3 - beztail * ac3 + ceztail * ab3))) -
        ((aex * aex + aey * aey + aez * aez) *
             ((bez * cdeps - cez * bdeps + dez * bceps) +
              (beztail * cd3 - ceztail * bd3 + deztail * bc3)) +
         (cex * cex + cey * cey + cez * cez) *
             ((dez * abeps + aez * bdeps + bez * daeps) +
              (deztail * ab3 + aeztail * bd3 + beztail * da3)))) +
       2.0 * (((bex * bextail + bey * beytail + bez * beztail) *
                   (cez * da3 + dez * ac3 + aez * cd3) +
               (dex * dextail + dey * deytail + dez * deztail) *
                   (aez * bc3 - bez * ac3 + cez * ab3)) -
              ((aex * aextail + aey * aeytail + aez * aeztail) *
                   (bez * cd3 - cez * bd3 + dez * bc3) +
               (cex * cextail + cey * ceytail + cez * ceztail) *
                   (dez * ab3 + aez * bd3 + bez * da3))));
  if (det >= errbound_c || -det >= errbound_c) {
    return {sign_of(det), InsphereStage::kC};
  }
  Expansion rx[4], ry[4], rz[4], lift[4];
  for (int k = 0; k < 4; ++k) {
    rx[k] = Expansion::difference(p[k][0], e[0]);
    ry[k] = Expansion::difference(p[k][1], e[1]);
    rz[k] = Expansion::difference(p[k][2], e[2]);
    lift[k] = rx[k] * rx[k] + ry[k] * ry[k] + rz[k] * rz[k];
  }
  return {lifted3_rows(rx, ry, rz, lift).sign(), InsphereStage::kD};
}

// Shewchuk's insphere (his sign: positive when e lies outside), with the
// deciding stage.
InsphereDecision insphere_shewchuk(const double* const* p) {
  const double* e = p[4];
  const double aex = p[0][0] - e[0], bex = p[1][0] - e[0], cex = p[2][0] - e[0], dex = p[3][0] - e[0];
  const double aey = p[0][1] - e[1], bey = p[1][1] - e[1], cey = p[2][1] - e[1], dey = p[3][1] - e[1];
  const double aez = p[0][2] - e[2], bez = p[1][2] - e[2], cez = p[2][2] - e[2], dez = p[3][2] - e[2];

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

  const double aezplus = std::fabs(aez), bezplus = std::fabs(bez);
  const double cezplus = std::fabs(cez), dezplus = std::fabs(dez);
  const double aexbeyplus = std::fabs(aexbey), bexaeyplus = std::fabs(bexaey);
  const double bexceyplus = std::fabs(bexcey), cexbeyplus = std::fabs(cexbey);
  const double cexdeyplus = std::fabs(cexdey), dexceyplus = std::fabs(dexcey);
  const double dexaeyplus = std::fabs(dexaey), aexdeyplus = std::fabs(aexdey);
  const double aexceyplus = std::fabs(aexcey), cexaeyplus = std::fabs(cexaey);
  const double bexdeyplus = std::fabs(bexdey), dexbeyplus = std::fabs(dexbey);
  const double permanent =
      ((cexdeyplus + dexceyplus) * bezplus + (dexbeyplus + bexdeyplus) * cezplus +
       (bexceyplus + cexbeyplus) * dezplus) *
          alift +
      ((dexaeyplus + aexdeyplus) * cezplus + (aexceyplus + cexaeyplus) * dezplus +
       (cexdeyplus + dexceyplus) * aezplus) *
          blift +
      ((aexbeyplus + bexaeyplus) * dezplus + (bexdeyplus + dexbeyplus) * aezplus +
       (dexaeyplus + aexdeyplus) * bezplus) *
          clift +
      ((bexceyplus + cexbeyplus) * aezplus + (cexaeyplus + aexceyplus) * bezplus +
       (aexbeyplus + bexaeyplus) * cezplus) *
          dlift;
  const double errbound = kIspErrBoundA * permanent;
  if (det > errbound || -det > errbound) {
    return {sign_of(det), InsphereStage::kA};
  }
  return insphere_adapt(p, permanent);
}

// ---------------------------------------------------------------- weighted power tests

template <class T, class Diff>
T power2d_value(const double* const* p, const double* w, Diff diff) {
  T x[3], y[3], lift[3];
  for (int k = 0; k < 3; ++k) {
    x[k] = diff(p[k][0], p[3][0]);
    y[k] = diff(p[k][1], p[3][1]);
    lift[k] = x[k] * x[k] + y[k] * y[k] - diff(w[k], w[3]);
  }
  return lifted2_rows(x, y, lift);
}

template <class T, class Diff>
T power3d_shewchuk_value(const double* const* p, const double* w, Diff diff) {
  T x[4], y[4], z[4], lift[4];
  for (int k = 0; k < 4; ++k) {
    x[k] = diff(p[k][0], p[4][0]);
    y[k] = diff(p[k][1], p[4][1]);
    z[k] = diff(p[k][2], p[4][2]);
    lift[k] = x[k] * x[k] + y[k] * y[k] + z[k] * z[k] - diff(w[k], w[4]);
  }
  return lifted3_rows(x, y, z, lift);
}

Approx approx_difference(double a, double b) { return Approx::exact(a) - Approx::exact(b); }
Expansion exact_difference(double a, double b) { return Expansion::difference(a, b); }

int power2d_impl(const double* const* p, const double* w) {
  const Approx filtered = power2d_value<Approx>(p, w, approx_difference);
  const int sign = filtered.certified_sign();
  if (sign != 2) {
    return sign;
  }
  return power2d_value<Expansion>(p, w, exact_difference).sign();
}

int power3d_impl(const double* const* p, const double* w) {
  const Approx filtered = power3d_shewchuk_value<Approx>(p, w, approx_difference);
  const int sign = filtered.certified_sign();
  if (sign != 2) {
    return -sign;
  }
  return -power3d_shewchuk_value<Expansion>(p, w, exact_difference).sign();
}

// ---------------------------------------------------------------- Simulation of Simplicity

// Exact determinant of the n x n submatrix of m selected by rows[0..n) and
// cols[0..n), by Laplace expansion along the first selected row.
Expansion determinant_exact(const double m[4][4], const int* rows, const int* cols, int n) {
  if (n == 1) {
    return Expansion(m[rows[0]][cols[0]]);
  }
  Expansion total;
  int minor_cols[4];
  for (int j = 0; j < n; ++j) {
    const double entry = m[rows[0]][cols[j]];
    if (entry == 0.0) {
      continue;
    }
    int count = 0;
    for (int k = 0; k < n; ++k) {
      if (k != j) {
        minor_cols[count++] = cols[k];
      }
    }
    const Expansion minor = determinant_exact(m, rows + 1, minor_cols, n - 1).scaled(entry);
    total = (j % 2 == 0) ? total + minor : total - minor;
  }
  return total;
}

struct SosMonomial {
  int count;
  int rank[3];
  int column[3];
};

// Perturbation monomials of the (d + 1) x (d + 1) orientation matrix with rows
// [p_r, 1] in rank order, sorted by decreasing significance, ending with the
// first full matching (whose coefficient is +-1 and therefore nonzero).
std::vector<SosMonomial> build_sos_monomials(int dimension) {
  struct Keyed {
    std::uint64_t key;
    SosMonomial monomial;
  };
  std::vector<Keyed> all;
  const int points = dimension + 1;
  // Enumerate injective partial maps rank -> column with 1..dimension pairs.
  const int total_masks = 1 << points;
  for (int mask = 1; mask < total_masks; ++mask) {
    int ranks[4];
    int count = 0;
    for (int r = 0; r < points; ++r) {
      if (mask & (1 << r)) {
        ranks[count++] = r;
      }
    }
    if (count > dimension) {
      continue;
    }
    std::array<int, 3> columns = {0, 1, 2};
    // Iterate over all ordered selections of `count` distinct columns.
    std::vector<int> selection(static_cast<std::size_t>(count), 0);
    const int combinations = [&] {
      int value = 1;
      for (int k = 0; k < count; ++k) {
        value *= dimension;
      }
      return value;
    }();
    for (int code = 0; code < combinations; ++code) {
      int remainder = code;
      bool distinct = true;
      int used = 0;
      for (int k = 0; k < count; ++k) {
        selection[static_cast<std::size_t>(k)] = columns[static_cast<std::size_t>(remainder % dimension)];
        remainder /= dimension;
        const int bit = 1 << selection[static_cast<std::size_t>(k)];
        if (used & bit) {
          distinct = false;
        }
        used |= bit;
      }
      if (!distinct) {
        continue;
      }
      Keyed keyed{};
      keyed.monomial.count = count;
      for (int k = 0; k < count; ++k) {
        const int r = ranks[k];
        const int j = selection[static_cast<std::size_t>(k)];
        keyed.monomial.rank[k] = r;
        keyed.monomial.column[k] = j;
        keyed.key |= std::uint64_t{1} << (dimension * (r + 1) - (j + 1));
      }
      all.push_back(keyed);
    }
  }
  std::sort(all.begin(), all.end(), [](const Keyed& left, const Keyed& right) { return left.key < right.key; });
  std::vector<SosMonomial> result;
  for (const Keyed& keyed : all) {
    result.push_back(keyed.monomial);
    if (keyed.monomial.count == dimension) {
      break;
    }
  }
  return result;
}

const std::vector<SosMonomial>& sos_monomials(int dimension) {
  static const std::vector<SosMonomial> two = build_sos_monomials(2);
  static const std::vector<SosMonomial> three = build_sos_monomials(3);
  return dimension == 2 ? two : three;
}

// Sorts point slots by index; returns the permutation parity (+1/-1).
int rank_order(const std::int64_t* ids, int count, int* order) {
  for (int k = 0; k < count; ++k) {
    order[k] = k;
  }
  int parity = 1;
  for (int i = 1; i < count; ++i) {
    for (int j = i; j > 0 && ids[order[j - 1]] > ids[order[j]]; --j) {
      std::swap(order[j - 1], order[j]);
      parity = -parity;
    }
  }
  return parity;
}

// Sign of det of the perturbed matrix with rows [p_rank, 1] in rank order.
[[gnu::noinline, gnu::cold]]
int orientation_sos_sign(const double* const* points, const std::int64_t* ids, int dimension,
                         int* parity_out) {
  int order[4];
  const int points_count = dimension + 1;
  *parity_out = rank_order(ids, points_count, order);
  double base[4][4];
  for (int r = 0; r < points_count; ++r) {
    for (int j = 0; j < dimension; ++j) {
      base[r][j] = points[order[r]][j];
    }
    base[r][dimension] = 1.0;
  }
  const int all_indices[4] = {0, 1, 2, 3};
  for (const SosMonomial& monomial : sos_monomials(dimension)) {
    double matrix[4][4];
    for (int r = 0; r < points_count; ++r) {
      for (int j = 0; j <= dimension; ++j) {
        matrix[r][j] = base[r][j];
      }
    }
    for (int k = 0; k < monomial.count; ++k) {
      const int r = monomial.rank[k];
      for (int j = 0; j <= dimension; ++j) {
        matrix[r][j] = j == monomial.column[k] ? 1.0 : 0.0;
      }
    }
    const int sign = determinant_exact(matrix, all_indices, all_indices, points_count).sign();
    if (sign != 0) {
      return sign;
    }
  }
  return 0;
}

// Lifted-coordinate SoS in 2D: cofactor of row t's lift is (-1)^t orient2d(others).
[[gnu::noinline, gnu::cold]]
int lifted_sos_2d(const double* const* p, const std::int64_t* ids) {
  int order[4];
  rank_order(ids, 4, order);
  for (int k = 0; k < 4; ++k) {
    const int t = order[k];
    const double* others[3];
    int count = 0;
    for (int s = 0; s < 4; ++s) {
      if (s != t) {
        others[count++] = p[s];
      }
    }
    const int cofactor = orient2d(others[0], others[1], others[2]);
    if (cofactor != 0) {
      return (t % 2 == 0) ? cofactor : -cofactor;
    }
  }
  return 0;
}

// Lifted-coordinate SoS in 3D: the result is -(-1)^t orient3d(others).
[[gnu::noinline, gnu::cold]]
int lifted_sos_3d(const double* const* p, const std::int64_t* ids) {
  int order[5];
  rank_order(ids, 5, order);
  for (int k = 0; k < 5; ++k) {
    const int t = order[k];
    const double* others[4];
    int count = 0;
    for (int s = 0; s < 5; ++s) {
      if (s != t) {
        others[count++] = p[s];
      }
    }
    const int cofactor = orient3d(others[0], others[1], others[2], others[3]);
    if (cofactor != 0) {
      return (t % 2 == 0) ? -cofactor : cofactor;
    }
  }
  return 0;
}

}  // namespace

// ---------------------------------------------------------------- public predicates

int orient2d(const double* a, const double* b, const double* c) {
  const double detleft = (a[0] - c[0]) * (b[1] - c[1]);
  const double detright = (a[1] - c[1]) * (b[0] - c[0]);
  const double det = detleft - detright;
  double detsum;
  if (detleft > 0.0) {
    if (detright <= 0.0) {
      return sign_of(det);
    }
    detsum = detleft + detright;
  } else if (detleft < 0.0) {
    if (detright >= 0.0) {
      return sign_of(det);
    }
    detsum = -detleft - detright;
  } else {
    return sign_of(det);
  }
  const double errbound = kCcwErrBoundA * detsum;
  if (det >= errbound || -det >= errbound) {
    return sign_of(det);
  }
  return orient2d_adapt(a, b, c, detsum);
}

int orient3d(const double* a, const double* b, const double* c, const double* d) {
  return -orient3d_shewchuk(a, b, c, d);
}

IncircleDecision incircle_decision(const double* a, const double* b, const double* c,
                                   const double* d) {
  return incircle_staged(a, b, c, d);
}

int incircle(const double* a, const double* b, const double* c, const double* d) {
  return incircle_staged(a, b, c, d).sign;
}

InsphereDecision insphere_decision(const double* a, const double* b, const double* c,
                                   const double* d, const double* e) {
  const double* p[5] = {a, b, c, d, e};
  const InsphereDecision decision = insphere_shewchuk(p);
  return {-decision.sign, decision.stage};
}

int insphere(const double* a, const double* b, const double* c, const double* d, const double* e) {
  const double* p[5] = {a, b, c, d, e};
  return -insphere_shewchuk(p).sign;
}

int power2d(const double* a, const double* b, const double* c, const double* d, double wa,
            double wb, double wc, double wd) {
  const double* p[4] = {a, b, c, d};
  const double w[4] = {wa, wb, wc, wd};
  return power2d_impl(p, w);
}

int power3d(const double* a, const double* b, const double* c, const double* d, const double* e,
            double wa, double wb, double wc, double wd, double we) {
  const double* p[5] = {a, b, c, d, e};
  const double w[5] = {wa, wb, wc, wd, we};
  return power3d_impl(p, w);
}

int orient2d_sos(const double* a, const double* b, const double* c, std::int64_t ia,
                 std::int64_t ib, std::int64_t ic) {
  const int base = orient2d(a, b, c);
  if (base != 0) {
    return base;
  }
  const double* p[3] = {a, b, c};
  const std::int64_t ids[3] = {ia, ib, ic};
  int parity = 1;
  const int sign = orientation_sos_sign(p, ids, 2, &parity);
  return parity * sign;
}

int orient3d_sos(const double* a, const double* b, const double* c, const double* d,
                 std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id) {
  const int base = orient3d(a, b, c, d);
  if (base != 0) {
    return base;
  }
  const double* p[4] = {a, b, c, d};
  const std::int64_t ids[4] = {ia, ib, ic, id};
  int parity = 1;
  const int sign = orientation_sos_sign(p, ids, 3, &parity);
  // orient3d(a, b, c, d) = -det[[a, 1]; [b, 1]; [c, 1]; [d, 1]].
  return -parity * sign;
}

int incircle_sos(const double* a, const double* b, const double* c, const double* d,
                 std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id) {
  const int base = incircle(a, b, c, d);
  if (base != 0) {
    return base;
  }
  const double* p[4] = {a, b, c, d};
  const std::int64_t ids[4] = {ia, ib, ic, id};
  return lifted_sos_2d(p, ids);
}

int insphere_sos(const double* a, const double* b, const double* c, const double* d,
                 const double* e, std::int64_t ia, std::int64_t ib, std::int64_t ic,
                 std::int64_t id, std::int64_t ie) {
  const int base = insphere(a, b, c, d, e);
  if (base != 0) {
    return base;
  }
  const double* p[5] = {a, b, c, d, e};
  const std::int64_t ids[5] = {ia, ib, ic, id, ie};
  return lifted_sos_3d(p, ids);
}

int power2d_sos(const double* a, const double* b, const double* c, const double* d, double wa,
                double wb, double wc, double wd, std::int64_t ia, std::int64_t ib,
                std::int64_t ic, std::int64_t id) {
  const int base = power2d(a, b, c, d, wa, wb, wc, wd);
  if (base != 0) {
    return base;
  }
  const double* p[4] = {a, b, c, d};
  const std::int64_t ids[4] = {ia, ib, ic, id};
  return lifted_sos_2d(p, ids);
}

int power3d_sos(const double* a, const double* b, const double* c, const double* d,
                const double* e, double wa, double wb, double wc, double wd, double we,
                std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id,
                std::int64_t ie) {
  const int base = power3d(a, b, c, d, e, wa, wb, wc, wd, we);
  if (base != 0) {
    return base;
  }
  const double* p[5] = {a, b, c, d, e};
  const std::int64_t ids[5] = {ia, ib, ic, id, ie};
  return lifted_sos_3d(p, ids);
}

Expansion orient2d_exact(const double* a, const double* b, const double* c) {
  const Expansion bx = Expansion::difference(b[0], a[0]);
  const Expansion by = Expansion::difference(b[1], a[1]);
  const Expansion cx = Expansion::difference(c[0], a[0]);
  const Expansion cy = Expansion::difference(c[1], a[1]);
  return bx * cy - by * cx;
}

Expansion orient3d_exact(const double* a, const double* b, const double* c, const double* d) {
  const Expansion x[3] = {Expansion::difference(a[0], d[0]), Expansion::difference(b[0], d[0]),
                          Expansion::difference(c[0], d[0])};
  const Expansion y[3] = {Expansion::difference(a[1], d[1]), Expansion::difference(b[1], d[1]),
                          Expansion::difference(c[1], d[1])};
  const Expansion z[3] = {Expansion::difference(a[2], d[2]), Expansion::difference(b[2], d[2]),
                          Expansion::difference(c[2], d[2])};
  return -orient3d_rows(x, y, z);
}

Approx orient2d_approx(const double* a, const double* b, const double* c) {
  const Approx bx = approx_difference(b[0], a[0]);
  const Approx by = approx_difference(b[1], a[1]);
  const Approx cx = approx_difference(c[0], a[0]);
  const Approx cy = approx_difference(c[1], a[1]);
  return bx * cy - by * cx;
}

Approx orient3d_approx(const double* a, const double* b, const double* c, const double* d) {
  const Approx x[3] = {approx_difference(a[0], d[0]), approx_difference(b[0], d[0]),
                       approx_difference(c[0], d[0])};
  const Approx y[3] = {approx_difference(a[1], d[1]), approx_difference(b[1], d[1]),
                       approx_difference(c[1], d[1])};
  const Approx z[3] = {approx_difference(a[2], d[2]), approx_difference(b[2], d[2]),
                       approx_difference(c[2], d[2])};
  return -orient3d_rows(x, y, z);
}

// ---------------------------------------------------------------- expansion coordinates

namespace {

// Determinant of the n x n submatrix of m (expansion entries) selected by
// rows[0..n) and cols[0..n), by Laplace expansion along the first row.
Expansion determinant_expansion(const Expansion m[4][4], const int* rows, const int* cols,
                                int n) {
  if (n == 1) return m[rows[0]][cols[0]];
  Expansion total;
  int minor_cols[4];
  for (int j = 0; j < n; ++j) {
    const Expansion& entry = m[rows[0]][cols[j]];
    if (entry.is_zero()) continue;
    int count = 0;
    for (int k = 0; k < n; ++k) {
      if (k != j) minor_cols[count++] = cols[k];
    }
    const Expansion minor = entry * determinant_expansion(m, rows + 1, minor_cols, n - 1);
    total = (j % 2 == 0) ? total + minor : total - minor;
  }
  return total;
}

// orientation_sos_sign with expansion coordinates: the same rank order and
// perturbation monomials, so mixed binary64/expansion evaluations simulate
// one global perturbation.
int orientation_sos_sign_expansion(const Expansion* const* points, const std::int64_t* ids,
                                   int dimension, int* parity_out) {
  int order[4];
  const int points_count = dimension + 1;
  *parity_out = rank_order(ids, points_count, order);
  const int all_indices[4] = {0, 1, 2, 3};
  for (const SosMonomial& monomial : sos_monomials(dimension)) {
    Expansion matrix[4][4];
    for (int r = 0; r < points_count; ++r) {
      for (int j = 0; j < dimension; ++j) matrix[r][j] = points[order[r]][j];
      matrix[r][dimension] = Expansion(1.0);
    }
    for (int k = 0; k < monomial.count; ++k) {
      const int r = monomial.rank[k];
      for (int j = 0; j <= dimension; ++j) {
        matrix[r][j] = j == monomial.column[k] ? Expansion(1.0) : Expansion();
      }
    }
    const int sign = determinant_expansion(matrix, all_indices, all_indices, points_count).sign();
    if (sign != 0) return sign;
  }
  return 0;
}

}  // namespace

int orient2d_sign(const Expansion* a, const Expansion* b, const Expansion* c) {
  return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).sign();
}

int orient3d_sign(const Expansion* a, const Expansion* b, const Expansion* c,
                  const Expansion* d) {
  const Expansion x[3] = {a[0] - d[0], b[0] - d[0], c[0] - d[0]};
  const Expansion y[3] = {a[1] - d[1], b[1] - d[1], c[1] - d[1]};
  const Expansion z[3] = {a[2] - d[2], b[2] - d[2], c[2] - d[2]};
  return -orient3d_rows(x, y, z).sign();
}

int insphere_sign(const Expansion* a, const Expansion* b, const Expansion* c,
                  const Expansion* d, const Expansion* e) {
  const Expansion* p[4] = {a, b, c, d};
  Expansion x[4], y[4], z[4], lift[4];
  for (int k = 0; k < 4; ++k) {
    x[k] = p[k][0] - e[0];
    y[k] = p[k][1] - e[1];
    z[k] = p[k][2] - e[2];
    lift[k] = x[k] * x[k] + y[k] * y[k] + z[k] * z[k];
  }
  return -lifted3_rows(x, y, z, lift).sign();
}

int orient3d_sos_sign(const Expansion* a, const Expansion* b, const Expansion* c,
                      const Expansion* d, std::int64_t ia, std::int64_t ib, std::int64_t ic,
                      std::int64_t id) {
  const int base = orient3d_sign(a, b, c, d);
  if (base != 0) return base;
  const Expansion* p[4] = {a, b, c, d};
  const std::int64_t ids[4] = {ia, ib, ic, id};
  int parity = 1;
  const int sign = orientation_sos_sign_expansion(p, ids, 3, &parity);
  return -parity * sign;
}

int insphere_sos_sign(const Expansion* a, const Expansion* b, const Expansion* c,
                      const Expansion* d, const Expansion* e, std::int64_t ia,
                      std::int64_t ib, std::int64_t ic, std::int64_t id, std::int64_t ie) {
  const int base = insphere_sign(a, b, c, d, e);
  if (base != 0) return base;
  const Expansion* p[5] = {a, b, c, d, e};
  const std::int64_t ids[5] = {ia, ib, ic, id, ie};
  int order[5];
  rank_order(ids, 5, order);
  for (int k = 0; k < 5; ++k) {
    const int t = order[k];
    const Expansion* others[4];
    int count = 0;
    for (int s = 0; s < 5; ++s) {
      if (s != t) others[count++] = p[s];
    }
    const int cofactor = orient3d_sign(others[0], others[1], others[2], others[3]);
    if (cofactor != 0) return (t % 2 == 0) ? -cofactor : cofactor;
  }
  return 0;
}

}  // namespace phx::mc
