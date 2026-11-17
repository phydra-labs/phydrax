//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Binary64 kernels with one result per input on every platform.
//
// Every value below is produced by IEEE 754 binary64 addition, subtraction,
// multiplication, division and square root (each correctly rounded), exact
// integer conversion and exact scaling by a power of two, in the written
// order. Nothing calls libm or LAPACK, and no sum depends on a vector width
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

// x^(2/3) as exp((2/3) log x): relative error at most 2^-43 for x in
// [2^-128, 2^128]. +0 gives +0, +infinity gives +infinity, a negative x and
// NaN give NaN.
double pow_two_thirds(double x) noexcept;

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
// overflow a rotation. Rotations run over (0,1), (0,2), (1,2) in every
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
