//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Small dense solves of the deterministic binary64 kernels; see
// deterministic_float.hpp. Every sum is written in the order it is evaluated.
#include "deterministic_float.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace phx::mc::deterministic {
namespace {

// Eigenvalues at most this fraction of the largest are read as zero, and a
// Gram-Schmidt remainder at most this fraction of its vector is dependent.
constexpr double kRankCutoff = 0x1p-40;

double magnitude(double value) noexcept { return value < 0.0 ? -value : value; }
bool finite(double value) noexcept {
  return magnitude(value) <= std::numeric_limits<double>::max();
}
bool positive_finite(double value) noexcept { return value > 0.0 && finite(value); }

double dot3(const double* a, const double* b) noexcept {
  return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}

}  // namespace

bool cholesky3(const double* matrix, double* factor) noexcept {
  if (!positive_finite(matrix[0])) return false;
  const double u00 = std::sqrt(matrix[0]);
  const double u01 = matrix[1] / u00;
  const double u02 = matrix[2] / u00;
  const double pivot1 = matrix[4] - u01 * u01;
  if (!positive_finite(pivot1)) return false;
  const double u11 = std::sqrt(pivot1);
  const double u12 = (matrix[5] - u01 * u02) / u11;
  const double pivot2 = (matrix[8] - u02 * u02) - u12 * u12;
  if (!positive_finite(pivot2)) return false;
  const double u22 = std::sqrt(pivot2);
  if (!(finite(u01) && finite(u02) && finite(u12))) return false;
  const double result[9] = {u00, u01, u02, 0.0, u11, u12, 0.0, 0.0, u22};
  for (std::size_t index = 0; index < 9; ++index) factor[index] = result[index];
  return true;
}

void multiply_upper3(const double* factor, const double* vector, double* product) noexcept {
  const double first = (factor[0] * vector[0] + factor[1] * vector[1]) + factor[2] * vector[2];
  const double second = factor[4] * vector[1] + factor[5] * vector[2];
  const double third = factor[8] * vector[2];
  product[0] = first;
  product[1] = second;
  product[2] = third;
}

void solve_upper3(const double* factor, const double* rhs, double* solution) noexcept {
  const double third = rhs[2] / factor[8];
  const double second = (rhs[1] - factor[5] * third) / factor[4];
  const double first = ((rhs[0] - factor[1] * second) - factor[2] * third) / factor[0];
  solution[0] = first;
  solution[1] = second;
  solution[2] = third;
}

int symmetric_pseudo_solve(int order, const double* matrix, const double* rhs,
                           double* solution) noexcept {
  // The matrix in the leading block of a 3x3 one. The rows and columns beyond
  // `order` are exactly zero, the iteration never rotates them, and their
  // eigenvalue 0 is never used.
  double full[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double right[3] = {0.0, 0.0, 0.0};
  for (int row = 0; row < order; ++row) {
    right[row] = rhs[row];
    if (!finite(right[row])) return -1;
    for (int column = row; column < order; ++column) {
      full[3 * row + column] = matrix[order * row + column];
    }
  }
  SymmetricEigen3 eigen;
  if (!symmetric_eigen3(full, eigen)) return -1;
  const double largest = eigen.values[2];
  double result[3] = {0.0, 0.0, 0.0};
  int used = 0;
  if (largest > 0.0) {
    const double cutoff = largest * kRankCutoff;
    for (std::size_t pair = 0; pair < 3; ++pair) {
      if (!(eigen.values[pair] > cutoff)) continue;
      const double vector[3] = {eigen.vectors[pair], eigen.vectors[3 + pair],
                                eigen.vectors[6 + pair]};
      const double coefficient = dot3(vector, right) / eigen.values[pair];
      for (std::size_t row = 0; row < 3; ++row) {
        const double term = coefficient * vector[row];
        result[row] = used == 0 ? term : result[row] + term;
      }
      ++used;
    }
  }
  for (int row = 0; row < order; ++row) {
    if (!finite(result[row])) return -1;
  }
  for (int row = 0; row < order; ++row) solution[row] = result[row];
  return used;
}

int orthonormal_basis3(int count, const double* vectors, double* basis) noexcept {
  int size = 0;
  for (int index = 0; index < count; ++index) {
    double remainder[3] = {vectors[3 * index], vectors[3 * index + 1], vectors[3 * index + 2]};
    const double own = dot3(remainder, remainder);
    if (!positive_finite(own)) continue;
    for (int earlier = 0; earlier < size; ++earlier) {
      const double* direction = basis + 3 * earlier;
      const double component = dot3(remainder, direction);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        remainder[axis] = remainder[axis] - component * direction[axis];
      }
    }
    const double left = dot3(remainder, remainder);
    if (!(left > own * kRankCutoff)) continue;
    const double length = std::sqrt(left);
    for (std::size_t axis = 0; axis < 3; ++axis) basis[3 * size + axis] = remainder[axis] / length;
    ++size;
  }
  return size;
}

}  // namespace phx::mc::deterministic
