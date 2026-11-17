//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Binary64 kernels with one result per input on every platform; see
// deterministic_float.hpp for the arithmetic they assume. Below, u = 2^-53 is
// the unit roundoff, fl(.) one rounded operation, and "ulp" the spacing of
// binary64 at the result, so that u |y| < ulp(y) <= 2 u |y| for normal y.
#include "deterministic_float.hpp"

#include <bit>
#include <cstdint>
#include <limits>

#include <cmath>

namespace phx::mc::deterministic {
namespace {

// ln 2 = kLn2High + kLn2Low + d with |d| <= 1.17e-26. kLn2High has 33
// significant bits, so k * kLn2High is exact for every |k| < 2^20.
constexpr double kLn2High = 0x1.62e42feep-1;
constexpr double kLn2Low = 0x1.a39ef35793c76p-33;
constexpr double kInverseLn2 = 0x1.71547652b82fep+0;
constexpr double kExpOverflow = 0x1.62e42fefa39efp+9;
constexpr double kExpUnderflow = -0x1.74910d52d3052p+9;
constexpr double kTwoThirds = 0x1.5555555555555p-1;
constexpr std::uint64_t kMantissaMask = 0x000fffffffffffffULL;
// The NaN a kernel produces for an input outside its domain.
constexpr std::uint64_t kNaNBits = 0x7ff8000000000000ULL;
// Mantissa field of fl(sqrt 2): significands above it are halved.
constexpr std::uint64_t kSqrt2Mantissa = 0x0006a09e667f3bcdULL;

// 1/n! for n = 2..13, each the correctly rounded quotient.
constexpr double kInverseFactorial[12] = {
    0x1.0000000000000p-1,  0x1.5555555555555p-3,  0x1.5555555555555p-5,
    0x1.1111111111111p-7,  0x1.6c16c16c16c17p-10, 0x1.a01a01a01a01ap-13,
    0x1.a01a01a01a01ap-16, 0x1.71de3a556c734p-19, 0x1.27e4fb7789f5cp-22,
    0x1.ae64567f544e4p-26, 0x1.1eed8eff8d898p-29, 0x1.6124613a86d09p-33};

// 1/(2k+1) for k = 1..10, each the correctly rounded quotient.
constexpr double kOddReciprocal[10] = {
    0x1.5555555555555p-2, 0x1.999999999999ap-3, 0x1.2492492492492p-3,
    0x1.c71c71c71c71cp-4, 0x1.745d1745d1746p-4, 0x1.3b13b13b13b14p-4,
    0x1.1111111111111p-4, 0x1.e1e1e1e1e1e1ep-5, 0x1.af286bca1af28p-5,
    0x1.8618618618618p-5};

constexpr int kJacobiSweepCap = 32;

// Results the self test expects: exp(1), log(10), 8^(2/3) and the eigenvalues
// of the matrix written there.
constexpr std::uint64_t kSelfTestExp = 0x4005bf0a8b14576aULL;
constexpr std::uint64_t kSelfTestLog = 0x40026bb1bbb55516ULL;
constexpr std::uint64_t kSelfTestPower = 0x400ffffffffffffeULL;
constexpr std::uint64_t kSelfTestEigen[3] = {0x3ffe1b5cb5435ca4ULL, 0x40031815d2babbe8ULL,
                                            0x4012ed1de951cae3ULL};
// One Horner step and one argument reduction of exp whose rounded results
// differ in the last bit when the product is fused into the sum.
constexpr std::uint64_t kSelfTestHorner = 0x3fe187fc8076a9b2ULL;
constexpr std::uint64_t kSelfTestReduction = 0x3ec00528c876d582ULL;

std::uint64_t bits_of(double value) noexcept { return std::bit_cast<std::uint64_t>(value); }

// 2^exponent for exponent in [-1022, 1023], from its bit pattern.
double power_of_two(int exponent) noexcept {
  return std::bit_cast<double>(static_cast<std::uint64_t>(exponent + 1023) << 52);
}

double canonical(double value) noexcept {
  return value == value ? value : std::bit_cast<double>(kNaNBits);
}

// y * 2^k for y in [1/2, 2) and k in [-1076, 1025]: exact when the product is
// normal, one rounding when it is subnormal, +infinity when it overflows.
double scaled(double y, int k) noexcept {
  if (k > 1023) return (y * power_of_two(1023)) * power_of_two(k - 1023);
  if (k < -1021) return (y * power_of_two(k + 54)) * 0x1p-54;
  return y * power_of_two(k);
}

double magnitude(double value) noexcept { return value < 0.0 ? -value : value; }

// floor(log2 |value|) of a finite nonzero value, subnormals included.
int binary_exponent(double value) noexcept {
  const std::uint64_t bits = bits_of(magnitude(value));
  if ((bits >> 52) != 0) return static_cast<int>(bits >> 52) - 1023;
  // A subnormal times 2^54 is normal and exact.
  return static_cast<int>(bits_of(magnitude(value) * 0x1p+54) >> 52) - 1023 - 54;
}

// value * 2^k for a finite value and |k| <= 1100, in steps of at most 2^1000.
// Every step is one correctly rounded product: exact while its result is
// normal, rounded when it is subnormal, infinite when it overflows.
double shifted(double value, int k) noexcept {
  while (k > 1000) {
    value = value * 0x1p+1000;
    k -= 1000;
  }
  while (k < -1000) {
    value = value * 0x1p-1000;
    k += 1000;
  }
  return value * power_of_two(k);
}

}  // namespace

// exp(x) = 2^k exp(r), k the integer nearest x / ln 2, r = x - k ln 2.
//
// Error budget, relative to exp(x), for kExpUnderflow < x <= kExpOverflow:
//   Reduction. |k| <= 1075. high = x - k kLn2High is exact: the product is
//   exact, and for k != 0 the operands are within a factor two of each other
//   (|x - k ln 2| < 0.3467 <= |k| ln 2 / 2, and kLn2High < ln 2), so the
//   subtraction is exact. low = fl(k kLn2Low) errs by at most u 1075 kLn2Low
//   = 2.3e-23, the split of ln 2 by at most 1075 * 1.17e-26 = 1.3e-23, and
//   r = fl(high - low) by at most u |r| <= 3.9e-17. The reduced argument is
//   therefore within 3.9e-17 of x - k ln 2, which changes exp by 0.35 u.
//   Approximation. exp(r) = 1 + r + r^2 q(r) with q the Taylor polynomial
//   sum_{n=2}^{13} r^(n-2) / n!; |r| <= 0.3467, so the remainder is at most
//   |r|^14 e^|r| / 14! <= 5.9e-18, 0.075 u of exp(r) >= 0.707.
//   Evaluation. Each Horner step rounds a product and a sum; the error made
//   at coefficient n is carried outward by |r|^(n-2), and every partial sum is
//   below e^|r| / n!, so q is off by at most 2 u e^|r| sum |r|^(n-2) / n!
//   <= 1.7 u, plus 0.6 u for the rounded coefficients: 2.3 u. Then
//   fl(r r) q errs by at most 0.12 * 2.3 u + 2 u * 0.069 <= 0.42 u, the sum
//   r + (.) rounds by at most 0.42 u, and 1 + (.) by u relative: together at
//   most u + 0.84 u / 0.707 = 2.19 u.
//   Scaling. Exact for a normal result; one more rounding, at most half an
//   ulp, for a subnormal one.
// Total below 0.35 u + 0.075 u + 2.19 u = 2.62 u < 3 ulp, plus the half ulp of
// a subnormal result: within 4 ulp.
double exp(double x) noexcept {
  if (x != x) return x;
  if (x > kExpOverflow) return std::numeric_limits<double>::infinity();
  if (x <= kExpUnderflow) return 0.0;
  const double quotient = x * kInverseLn2;
  const int k = static_cast<int>(quotient < 0.0 ? quotient - 0.5 : quotient + 0.5);
  const double count = static_cast<double>(k);
  const double high = x - count * kLn2High;
  const double low = count * kLn2Low;
  const double r = high - low;
  double q = kInverseFactorial[11];
  for (int index = 10; index >= 0; --index) q = q * r + kInverseFactorial[index];
  const double y = 1.0 + (r + (r * r) * q);
  return scaled(y, k);
}

// log(x) = e ln 2 + log(m), x = 2^e m with m in (sqrt(1/2), sqrt 2], and
// log(m) = 2 atanh(s) = 2 (s + s z P(z)), s = (m - 1) / (m + 1), z = s^2,
// P(z) = sum_{k=1}^{10} z^(k-1) / (2k + 1).
//
// Error budget, relative to log(x), for positive finite x:
//   Decomposition. e and m are exact (a subnormal x is first scaled by 2^54,
//   exactly); f = m - 1 is exact because m lies in [1/2, 2].
//   s. fl(2 + f) and the division each round once: s is off by at most
//   2.01 u |s|, |s| <= 0.1716. z = fl(s s) is off by at most 5.1 u z,
//   z <= 0.02945.
//   P. 1/3 <= P <= 0.3395. Rounded coefficients 0.34 u, Horner rounding
//   0.72 u, the error of z 0.04 u: at most 1.1 u absolute, 3.3 u relative.
//   Tail. s z P is off by at most (2.01 + 5.1 + 3.3 + 2) u = 12.5 u of itself,
//   and it is at most 0.011 |s|: 0.14 u |s|. The sum s + s z P rounds by at
//   most 1.02 u |s|. The omitted series terms are below |s| z^11 / 22 =
//   0.006 u |s|. So the sum is within 3.18 u |s| of atanh(s) >= |s|, and
//   doubling is exact: log(m) is within 3.18 u of itself.
//   e = 0. The result is that value: within 3.18 u, below 3.2 ulp.
//   e != 0. e kLn2High is exact, fl(e kLn2Low) and the split of ln 2 err by
//   at most 2.2e-7 u together, the inner sum rounds by at most 0.35 u, the
//   outer by u |log x|, and log(m) contributes 3.18 u * 0.3467 = 1.11 u.
//   |log x| >= ln 2 - 0.3467 = 0.3464, so the relative error is at most
//   (1.11 + 0.35) u / 0.3464 + u = 5.22 u.
// Total: within 6 ulp.
double log(double x) noexcept {
  if (x != x) return x;
  if (x < 0.0) return std::bit_cast<double>(kNaNBits);
  if (x == 0.0) return -std::numeric_limits<double>::infinity();
  if (x == std::numeric_limits<double>::infinity()) return x;
  int exponent = 0;
  std::uint64_t bits = bits_of(x);
  if ((bits >> 52) == 0) {
    bits = bits_of(x * 0x1p54);
    exponent = -54;
  }
  exponent += static_cast<int>(bits >> 52) - 1023;
  const std::uint64_t mantissa = bits & kMantissaMask;
  std::uint64_t significand_exponent = 1023;
  if (mantissa > kSqrt2Mantissa) {
    significand_exponent = 1022;
    exponent += 1;
  }
  const double m = std::bit_cast<double>(mantissa | (significand_exponent << 52));
  const double f = m - 1.0;
  const double s = f / (2.0 + f);
  const double z = s * s;
  double p = kOddReciprocal[9];
  for (int index = 8; index >= 0; --index) p = p * z + kOddReciprocal[index];
  const double log_m = 2.0 * (s + (s * z) * p);
  const double count = static_cast<double>(exponent);
  return count * kLn2High + (count * kLn2Low + log_m);
}

// x^(2/3) = exp(fl(log(x) kTwoThirds)).
//
// Error budget for x in [2^-128, 2^128], where |log x| <= 88.73: the
// logarithm is within 6 ulp, at most 12 u |log x|; the constant and the
// product each add u |log x|. The exponent is therefore within
// (2/3) 14 u * 88.73 = 9.2e-14 of (2/3) log x, which changes the result by
// that relative amount; exp adds 4 ulp, at most 8.9e-16. Total relative error
// below 9.3e-14 < 2^-43.
double pow_two_thirds(double x) noexcept {
  if (x != x) return x;
  if (x < 0.0) return std::bit_cast<double>(kNaNBits);
  if (x == 0.0) return 0.0;
  if (x == std::numeric_limits<double>::infinity()) return x;
  return exp(log(x) * kTwoThirds);
}

// Products of finite entries can overflow, and two overflowed terms of
// opposite sign give a NaN whose sign and payload depend on the instruction
// set; so can a NaN entry. Every NaN result is replaced by the canonical one.
double determinant3(const double* m) noexcept {
  const double first = m[0] * (m[4] * m[8] - m[5] * m[7]);
  const double second = m[1] * (m[3] * m[8] - m[5] * m[6]);
  const double third = m[2] * (m[3] * m[7] - m[4] * m[6]);
  return canonical((first - second) + third);
}

double quadratic_form3(const double* v, const double* m) noexcept {
  const double row0 = (m[0] * v[0] + m[1] * v[1]) + m[2] * v[2];
  const double row1 = (m[3] * v[0] + m[4] * v[1]) + m[5] * v[2];
  const double row2 = (m[6] * v[0] + m[7] * v[1]) + m[8] * v[2];
  return canonical((v[0] * row0 + v[1] * row1) + v[2] * row2);
}

bool symmetric_eigen3(const double* matrix, SymmetricEigen3& out) noexcept {
  // Upper triangle a[p][q], p < q; d holds the running diagonal. The diagonal
  // is accumulated as b plus the sweep's corrections z, so that a sweep adds
  // its small corrections to each other before they meet the large diagonal.
  double a01 = matrix[1];
  double a02 = matrix[2];
  double a12 = matrix[5];
  double d[3] = {matrix[0], matrix[4], matrix[8]};
  double largest = 0.0;
  for (const double value : {a01, a02, a12, d[0], d[1], d[2]}) {
    if (!(magnitude(value) <= std::numeric_limits<double>::max())) return false;
    if (magnitude(value) > largest) largest = magnitude(value);
  }
  // The sweeps run on the matrix times the power of two that brings its
  // largest entry into [1, 2): every entry is then at most 2 in magnitude,
  // the diagonal stays below 6 and no rotation can overflow. Scaling by a
  // power of two commutes with every operation below, so the result is the
  // one of the unscaled iteration wherever that neither overflows nor
  // underflows. An entry below 2^-1022 times the largest rounds to a
  // subnormal, a change below 2^-1074 of the largest entry.
  const int shift = largest == 0.0 ? 0 : -binary_exponent(largest);
  a01 = shifted(a01, shift);
  a02 = shifted(a02, shift);
  a12 = shifted(a12, shift);
  for (double& value : d) value = shifted(value, shift);
  double b[3] = {d[0], d[1], d[2]};
  double z[3] = {0.0, 0.0, 0.0};
  double v[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  // One rotation in the (p, q) plane annihilating `pivot`; `left` and `right`
  // are the two other off-diagonal entries, coupling the third index with p
  // and with q.
  const auto rotate = [&](int p, int q, int sweep, double threshold, double& pivot,
                          double& left, double& right) noexcept {
    const double g = 100.0 * magnitude(pivot);
    if (sweep > 3 && magnitude(d[p]) + g == magnitude(d[p]) &&
        magnitude(d[q]) + g == magnitude(d[q])) {
      pivot = 0.0;
      return;
    }
    if (!(magnitude(pivot) > threshold)) return;
    double h = d[q] - d[p];
    double t = 0.0;
    if (magnitude(h) + g == magnitude(h)) {
      t = pivot / h;
    } else {
      const double theta = 0.5 * h / pivot;
      t = 1.0 / (magnitude(theta) + std::sqrt(theta * theta + 1.0));
      if (theta < 0.0) t = -t;
    }
    const double c = 1.0 / std::sqrt(t * t + 1.0);
    const double s = t * c;
    const double tau = s / (1.0 + c);
    h = t * pivot;
    z[p] -= h;
    z[q] += h;
    d[p] -= h;
    d[q] += h;
    pivot = 0.0;
    const double old_left = left;
    const double old_right = right;
    left = old_left - s * (old_right + old_left * tau);
    right = old_right + s * (old_left - old_right * tau);
    for (int row = 0; row < 3; ++row) {
      const double old_p = v[row][p];
      const double old_q = v[row][q];
      v[row][p] = old_p - s * (old_q + old_p * tau);
      v[row][q] = old_q + s * (old_p - old_q * tau);
    }
  };

  for (int sweep = 1; sweep <= kJacobiSweepCap; ++sweep) {
    const double off = (magnitude(a01) + magnitude(a02)) + magnitude(a12);
    if (off == 0.0) {
      // Ascending order by three fixed comparisons; equal values keep their order.
      int order[3] = {0, 1, 2};
      const auto arrange = [&](int i, int j) noexcept {
        if (d[order[j]] < d[order[i]]) {
          const int swap = order[i];
          order[i] = order[j];
          order[j] = swap;
        }
      };
      arrange(0, 1);
      arrange(1, 2);
      arrange(0, 1);
      for (int column = 0; column < 3; ++column) {
        // An eigenvalue beyond the binary64 range has no result.
        const double value = shifted(d[order[column]], -shift);
        if (!(magnitude(value) <= std::numeric_limits<double>::max())) return false;
        out.values[static_cast<std::size_t>(column)] = value;
        for (int row = 0; row < 3; ++row) {
          out.vectors[static_cast<std::size_t>(3 * row + column)] = v[row][order[column]];
        }
      }
      return true;
    }
    const double threshold = sweep < 4 ? (0x1.999999999999ap-3 * off) / 9.0 : 0.0;
    // (0,1): the third index 2 couples as a02 (with p = 0) and a12 (with q = 1).
    rotate(0, 1, sweep, threshold, a01, a02, a12);
    // (0,2): the third index 1 couples as a01 (with p = 0) and a12 (with q = 2).
    rotate(0, 2, sweep, threshold, a02, a01, a12);
    // (1,2): the third index 0 couples as a01 (with p = 1) and a02 (with q = 2).
    rotate(1, 2, sweep, threshold, a12, a01, a02);
    for (int index = 0; index < 3; ++index) {
      b[index] += z[index];
      d[index] = b[index];
      z[index] = 0.0;
    }
  }
  return false;
}

std::array<double, 9> symmetric_function3(const SymmetricEigen3& eigen,
                                          const std::array<double, 3>& mapped) noexcept {
  std::array<double, 9> result{};
  const std::array<double, 9>& v = eigen.vectors;
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = row; column < 3; ++column) {
      const double value = canonical(((v[3 * row] * mapped[0]) * v[3 * column] +
                                      (v[3 * row + 1] * mapped[1]) * v[3 * column + 1]) +
                                     (v[3 * row + 2] * mapped[2]) * v[3 * column + 2]);
      result[3 * row + column] = value;
      result[3 * column + row] = value;
    }
  }
  return result;
}

SelfTest self_test() noexcept {
  // Operands are volatile so that the checks run with this thread's arithmetic
  // and are not folded at compile time.
  volatile double one = 1.0;
  volatile double half_ulp = 0x1p-53;
  volatile double above_half_ulp = 0x1.8p-53;
  volatile double above_one = 0x1.00000004p+0;
  volatile double below_one = 0x1.fffffff8p-1;
  volatile double smallest_normal = 0x1p-1022;
  volatile double half = 0.5;
  // Round to nearest, ties to even, in binary64: a wider intermediate keeps
  // 1 + 2^-53 apart from 1, and a directed mode moves one of the two sums.
  const double tie = one + half_ulp;
  const double up = one + above_half_ulp;
  if (tie != 1.0 || up != 0x1.0000000000001p+0) return SelfTest::kRounding;
  // No contraction: (1 + 2^-30)(1 - 2^-30) rounds to 1, and fused with the
  // subtraction it would leave -2^-60.
  const double product_difference = above_one * below_one - one;
  if (product_difference != 0.0) return SelfTest::kContraction;
  // Gradual underflow, no flush to zero and no denormals-are-zero.
  const double subnormal = smallest_normal * half;
  if (subnormal != 0x1p-1023 || subnormal + subnormal != 0x1p-1022) {
    return SelfTest::kUnderflow;
  }
  // The same in the two places where exp would change: a Horner step and the
  // second part of the argument reduction.
  volatile double horner_r = 0x1.25fd6058ff452p-2;
  volatile double reduction_x = -0x1.3c6bf3ad6479bp+9;
  volatile double reduction_count = -913.0;
  const double horner = kInverseFactorial[1] * horner_r + half;
  const double reduced =
      (reduction_x - reduction_count * kLn2High) - reduction_count * kLn2Low;
  if (bits_of(horner) != kSelfTestHorner || bits_of(reduced) != kSelfTestReduction) {
    return SelfTest::kContraction;
  }
  // One result of every kernel.
  volatile double exp_input = 1.0;
  volatile double log_input = 10.0;
  volatile double power_input = 8.0;
  if (bits_of(exp(exp_input)) != kSelfTestExp ||
      bits_of(log(log_input)) != kSelfTestLog ||
      bits_of(pow_two_thirds(power_input)) != kSelfTestPower) {
    return SelfTest::kKernelBits;
  }
  const double matrix[9] = {4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0};
  SymmetricEigen3 eigen;
  if (!symmetric_eigen3(matrix, eigen)) return SelfTest::kKernelBits;
  const bool eigenvalues = bits_of(eigen.values[0]) == kSelfTestEigen[0] &&
                           bits_of(eigen.values[1]) == kSelfTestEigen[1] &&
                           bits_of(eigen.values[2]) == kSelfTestEigen[2];
  return eigenvalues ? SelfTest::kPassed : SelfTest::kKernelBits;
}

}  // namespace phx::mc::deterministic
