//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Riemannian metric measurements of tetrahedral meshes on the deterministic
// binary64 kernels (deterministic_float.hpp): one result per input on every
// platform. A metric is a symmetric positive-definite 3x3 tensor stored as
// nine row-major values; only its upper triangle is read by the tensor
// functions, and every tensor written is exactly symmetric.
//
// These values measure and steer: edge lengths, the violation quanta of the
// length band, cell quality and the relative determinant order and screen
// floating-point proposals. Whether a mesh edit is valid is decided elsewhere
// by exact predicates. A NaN result is always the one bit pattern kNaN.
#pragma once

#include <cstdint>

namespace phx::mc::metric {

// The band of admissible metric edge lengths [1 / fl(sqrt 2), fl(sqrt 2)] and
// the quantum 2^-10 in which a violation |log(length / band edge)| is counted.
inline constexpr double kLowerLength = 0x1.6a09e667f3bccp-1;
inline constexpr double kUpperLength = 0x1.6a09e667f3bcdp+0;
inline constexpr double kViolationQuanta = 0x1p+10;

// log(M) = V diag(log lambda) V^T. False, with `logarithm` untouched, when the
// eigen iteration does not end or an eigenvalue is not positive.
bool logarithm(const double* metric, double* logarithm) noexcept;

// exp(L) = V diag(exp lambda) V^T for a symmetric L. False when the eigen
// iteration does not end or the result is not finite.
bool exponential(const double* logarithm, double* metric) noexcept;

// Log-Euclidean mean exp(sum_k w_k log M_k) of `count` >= 1 support tensors,
// summed in support order. A support whose tensors all compare equal returns
// its first tensor unchanged, once that tensor has a logarithm: a constant
// field is a fixed point of the mean. `logarithms` holds log M_k, or is null
// to have them evaluated here; supplied logarithms are taken as those of the
// tensors, which are then not examined again. False, with `metric`
// untouched, when a logarithm or the exponential fails.
bool interpolate(std::int64_t count, const double* weights, const double* metrics,
                 const double* logarithms, double* metric) noexcept;

// Length of the segment from p to q: the logarithmic mean
// (a - b) / log(a / b) of a = sqrt(d^T M_p d) and b = sqrt(d^T M_q d),
// d = q - p, and their arithmetic mean when |b / a - 1| <= 1e-6. NaN when a
// quadratic form is negative or not a number.
double length(const double* p, const double* q, const double* metric_p,
              const double* metric_q) noexcept;

// Violation quanta of a length: `above` is max(ceil(log(length / sqrt 2) *
// 2^10), 1) for a length above the band and 0 otherwise, `below` the same for
// log((1/sqrt 2) / length) below it. An infinite or zero length gives an
// infinite count. A NaN length was not measured and has no count: both are
// 0x7ff8000000000000.
void band_quanta(double length, double& above, double& below) noexcept;

// det(p1 - p0, p2 - p0, p3 - p0) of four corners (12 values) and that
// determinant's magnitude over the largest product of the three edge lengths
// at a corner; the ratio is 0 when either is not finite or the product is 0.
// The evaluated determinant differs from the exact one by at most 50 u times
// that product, so above a ratio of 2^-22 its sign is the exact orientation.
void relative_determinant(const double* corners, double& determinant,
                          double& relative) noexcept;

// Euclidean mean ratio of an orientation-checked tetrahedron.
double geometric_shape(const double* corners, double determinant) noexcept;

// Mean-ratio quality 12 (3 V_M)^(2/3) / sum l^2 of one tetrahedron: V_M is
// the Euclidean volume times exp(mean_k log det M_k / 2) (the determinant of
// the log-Euclidean barycentric metric) and l its six metric edge lengths in
// the order (0,1), (0,2), (0,3), (1,2), (1,3), (2,3). 0 unless V_M and the
// sum are both positive; NaN when both are infinite.
double tetra_quality(const double* corners, const double* metrics) noexcept;

// The same quality with log det M_k of the four corners supplied (the values
// log(determinant) the function above evaluates), for callers that keep them
// per vertex: the result is bit for bit that of tetra_quality.
double tetra_quality(const double* corners, const double* metrics,
                     const double* log_determinants) noexcept;

// The same score from the signed corner determinant and a sum of squared
// edges measured and accumulated in tetrahedron corner order.
double tetra_quality_from_measurements(double determinant, double squares,
                                       const double* log_determinants) noexcept;

// log det M as tetra_quality evaluates it, for a caller's per-vertex cache.
double log_determinant(const double* metric) noexcept;

// Whether a full row-major 3x3 tensor is admitted as the metric of a field.
// With lambda the eigenvalues of its symmetric part (M + M^T) / 2 from the
// Jacobi iteration and tol = 64 * 2^-52 * max |lambda|: `symmetric` when
// max |M - M^T| <= tol, and `positive_definite` when it is symmetric and
// min lambda > tol. Neither for a non-finite entry, an eigenvalue beyond the
// binary64 range or an iteration that does not end.
void tensor_properties(const double* tensor, bool& symmetric,
                       bool& positive_definite) noexcept;

// Defect of the congruence A^T M A = R of row-major 3x3 arrays, the form in
// which a metric M at the image of a linear map A agrees with the metric R at
// its source: `error` is the infinity norm (largest absolute row sum) of
// A^T M A - R and `scale` the larger of the infinity norms of A^T M A and R.
// M A and then A^T (M A) are formed by rows, each entry a sum of three
// products in column order. Both are 0x7ff8000000000000 when a row sum is
// not a number.
void congruence_defect(const double* linear, const double* metric, const double* reference,
                       double& error, double& scale) noexcept;

}  // namespace phx::mc::metric
