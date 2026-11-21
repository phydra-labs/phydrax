//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Binary64 kernels with one result per input on every platform.
//
// Every value below is produced by IEEE 754 binary64 addition, subtraction,
// multiplication, division and square root (each correctly rounded), exact
// integer conversion and scaling by a power of two (one correctly rounded
// product: exact unless the result is subnormal), in the written order.
// Nothing calls libm or LAPACK, and no sum depends on a vector width
// or a thread count. The kernels assume round-to-nearest-even, gradual
// underflow, no contraction of a product and a sum into one operation and no
// extended intermediate precision; `self_test` refuses a build or a thread
// state in which one of them does not hold. Callers at a C boundary establish
// the rounding mode with GradualNearestScope.
//
// The kernels steer floating-point proposals. Their stated error bounds are
// proved in the comments of deterministic_float.cpp; no bound is "correctly
// rounded", and no exact decision may rest on them.
#pragma once

#include <array>
#include <cmath>

// Contraction is forbidden twice: by the build flags, which the build states
// to the source as PHX_MC_FP_CONTRACT_OFF, and by the pragma below for every
// translation unit that includes this header.
#if !defined(PHX_MC_FP_CONTRACT_OFF)
#error "deterministic_float.hpp requires the meshcore floating-point build flags."
#endif
#if defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ != 0
#error "deterministic_float.hpp requires binary64 evaluation without extended precision."
#endif
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace phx::mc::deterministic {

// e^x within 4 ulp. NaN gives NaN; above 0x1.62e42fefa39efp+9 the result is
// +infinity and at or below -0x1.74910d52d3052p+9 it is +0.
double exp(double x) noexcept;

// Natural logarithm within 6 ulp for positive finite x (subnormals included).
// log(+0) and log(-0) are -infinity, a negative x and NaN give NaN,
// +infinity gives +infinity.
double log(double x) noexcept;

// log(1 + x) within 7 ulp for finite x > -1. log1p(-1) is -infinity, x < -1
// and NaN give NaN, +infinity gives +infinity.
double log1p(double x) noexcept;

// x^(2/3) as exp((2/3) log x): relative error at most 2^-43 for x in
// [2^-128, 2^128]. +0 gives +0, +infinity gives +infinity, a negative x and
// NaN give NaN.
double pow_two_thirds(double x) noexcept;

// The cube root of a positive finite value by Newton's iteration from a power
// of two (frexp and ldexp are exact): eight steps from a start within a factor
// of two reach the last places, not correctly rounded. Other values are
// returned as given.
inline double cube_root(double value) noexcept {
  if (!(value > 0.0) || !std::isfinite(value)) return value;
  int exponent = 0;
  std::frexp(value, &exponent);
  double root = std::ldexp(1.0, exponent / 3);
  for (int step = 0; step < 8; ++step) root = (2.0 * root + value / (root * root)) / 3.0;
  return root;
}

// Determinant of a row-major 3x3 matrix by cofactors of its first row, and
// v^T M v for a row-major 3x3 matrix (M v by rows, then the dot product).
// Neither scales its operands: a product of finite entries beyond the
// binary64 range gives an infinite result, or NaN when such terms cancel. A
// NaN result, also from a NaN entry, is always 0x7ff8000000000000.
double determinant3(const double* matrix) noexcept;
double quadratic_form3(const double* vector, const double* matrix) noexcept;

// Eigenvalues in ascending order (equal values keep their Jacobi order) and
// orthonormal eigenvectors as the columns of a row-major 3x3 matrix.
struct SymmetricEigen3 {
  std::array<double, 3> values{};
  std::array<double, 9> vectors{};
};

// Cyclic Jacobi diagonalization of the symmetric matrix given by the upper
// triangle of a row-major 3x3 array, run on the matrix scaled by the power of
// two that brings its largest entry into [1, 2), so no finite input can
// overflow a rotation; the eigenvalues are scaled back, exactly unless one
// becomes subnormal. Rotations run over (0,1), (0,2), (1,2) in every
// sweep; an off-diagonal entry that cannot change either diagonal entry
// (adding 100 times its magnitude leaves both unchanged) is set to zero; the
// iteration ends when all three are exactly zero. False for a non-finite
// entry, for an eigenvalue beyond the binary64 range, or when 32 sweeps do
// not end the iteration; `out` is then unspecified.
bool symmetric_eigen3(const double* matrix, SymmetricEigen3& out) noexcept;

// V diag(mapped) V^T as a row-major 3x3 matrix. The upper triangle is summed
// over the eigenvalues in order and mirrored, so the result is exactly
// symmetric. Infinite mapped values give infinite entries, or
// 0x7ff8000000000000 where they cancel.
std::array<double, 9> symmetric_function3(const SymmetricEigen3& eigen,
                                          const std::array<double, 3>& mapped) noexcept;

// Upper Cholesky factor U of the symmetric matrix given by the upper triangle
// of a row-major 3x3 array, M = U^T U, as a row-major 3x3 array with zeros
// below the diagonal. False, with `factor` untouched, unless every pivot is
// positive and every entry of U finite.
bool cholesky3(const double* matrix, double* factor) noexcept;

// U v by rows, and the x with U x = b by back substitution, for an upper
// triangular row-major 3x3 U (the factor of cholesky3).
void multiply_upper3(const double* factor, const double* vector, double* product) noexcept;
void solve_upper3(const double* factor, const double* rhs, double* solution) noexcept;

// Minimum-norm least-squares solution of N x = b for a symmetric positive
// semidefinite N of order 1, 2 or 3 (the upper triangle of a row-major
// order x order array): the sum, in ascending eigenvalue order, of
// v (v . b) / lambda over the eigenpairs of the Jacobi iteration with
// lambda > 2^-40 lambda_max. Smaller eigenvalues are read as zero: of a
// matrix of scalar products they are rounding noise of a dependent direction.
// Returns the number of eigenpairs used; 0, with x = 0, when no eigenvalue is
// positive; -1, with x untouched, for a non-finite entry, an iteration that
// does not end or a non-finite solution.
int symmetric_pseudo_solve(int order, const double* matrix, const double* rhs,
                           double* solution) noexcept;

// Orthonormal basis of the span of `count` <= 3 vectors of R^3 (the rows of
// `vectors`) by modified Gram-Schmidt in the given order: a vector has its
// components along the earlier basis vectors removed one after the other and
// is divided by its length. A vector whose remainder has a squared length at
// most 2^-40 times its own, or whose squared length is zero or not finite,
// adds no basis vector. Returns the number of basis vectors, written as the
// leading rows of `basis` (3 values each).
int orthonormal_basis3(int count, const double* vectors, double* basis) noexcept;

// What the self test found: the first assumption that does not hold on this
// thread, or that a fixed kernel result does not have its expected bits.
enum class SelfTest : int {
  kPassed = 0,
  kRounding = 1,     // not round-to-nearest-even in binary64 (or extended precision)
  kContraction = 2,  // a product and a sum were fused
  kUnderflow = 3,    // subnormals are flushed or read as zero
  kKernelBits = 4,   // exp, log, x^(2/3) or the eigenvalues differ
};

SelfTest self_test() noexcept;

}  // namespace phx::mc::deterministic
