//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Bit vectors of log1p and the small fixed-order solves of the deterministic
// binary64 kernels. The inputs are produced here: fixed edge cases, then sets
// from the integer generator (small integers scaled by powers of two, so
// they are the same binary64 values on every platform).
// deterministic_solve_vectors.inc holds each input with the output bits the
// kernels produced when the file was written; this test requires the same
// bits on every platform. `--emit` prints that file.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bit_vectors.hpp"
#include "check.hpp"
#include "deterministic_float.hpp"
#include "floating_point_mode.hpp"

namespace {

using phx::mc::test::append_bits;
using phx::mc::test::BitRow;
using phx::mc::test::bits_of;
using phx::mc::test::IntegerGenerator;
using phx::mc::test::kInfinityBits;
using phx::mc::test::kNaNBits;
using phx::mc::test::power_of_two;
using phx::mc::test::value_of;

#define PHX_DS_LOG1P(...) {"PHX_DS_LOG1P", {__VA_ARGS__}},
#define PHX_DS_PSEUDO(...) {"PHX_DS_PSEUDO", {__VA_ARGS__}},
#define PHX_DS_CHOLESKY(...) {"PHX_DS_CHOLESKY", {__VA_ARGS__}},
#define PHX_DS_BASIS(...) {"PHX_DS_BASIS", {__VA_ARGS__}},
const std::vector<BitRow> kRecorded = {
#include "deterministic_solve_vectors.inc"
};

std::vector<double> log1p_inputs() {
  std::vector<double> inputs = {
      0.0, -0.0, 1.0, -0.5, 0.5, 2.0, 0x1p+53, 0x1p+60, 0x1.fffffffffffffp+1023,
      // Around the three branch points and the ends of the domain.
      0x1.fffffffffffffp-55, 0x1p-54, 0x1.0000000000001p-54, -0x1.fffffffffffffp-55, -0x1p-54,
      -0x1.0000000000001p-54, -0x1.0000000000001p-2, -0x1p-2, -0x1.fffffffffffffp-3,
      0x1.7ffffffffffffp-2, 0x1.8p-2, 0x1.8000000000001p-2, -0x1.fffffffffffffp-1, -1.0,
      -0x1.0000000000001p+0, -2.0, 0x1p-1074, -0x1p-1074, value_of(kInfinityBits),
      -value_of(kInfinityBits), value_of(kNaNBits)};
  IntegerGenerator generator{0x3f84d5b5b5470917ULL};
  for (int exponent = -60; exponent <= 60; ++exponent) {
    for (int repeat = 0; repeat < 2; ++repeat) {
      inputs.push_back(generator.with_exponent(exponent, true));
    }
  }
  return inputs;
}

// A symmetric matrix as (m00, m01, m02, m11, m12, m22), a right-hand side and
// the order of the system in its leading block.
struct System {
  int order;
  std::array<double, 6> matrix;
  std::array<double, 3> rhs;
};

// The matrix of scalar products of `order` integer vectors times `scale`, with
// the right-hand side of the circumcenter (half its diagonal).
System gram_system(int order, const std::int64_t vectors[3][3], double scale) {
  System system{order, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  const std::size_t slot[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  for (int row = 0; row < order; ++row) {
    for (int column = row; column < order; ++column) {
      std::int64_t sum = 0;
      for (int axis = 0; axis < 3; ++axis) sum += vectors[row][axis] * vectors[column][axis];
      system.matrix[slot[row][column]] = static_cast<double>(sum) * scale;
      if (row == column) system.rhs[static_cast<std::size_t>(row)] = 0.5 * system.matrix[slot[row][column]];
    }
  }
  return system;
}

std::vector<System> pseudo_inputs() {
  std::vector<System> inputs = {
      {1, {4.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}},
      {1, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}},
      {2, {1.0, 0.0, 0.0, 1.0, 0.0, 0.0}, {3.0, -2.0, 0.0}},
      {2, {2.0, 1.0, 0.0, 2.0, 0.0, 0.0}, {1.0, 1.0, 0.0}},
      // Dependent: the second direction is twice the first.
      {2, {1.0, 2.0, 0.0, 4.0, 0.0, 0.0}, {0.5, 2.0, 0.0}},
      {3, {1.0, 0.0, 0.0, 1.0, 0.0, 1.0}, {1.0, 2.0, 3.0}},
      {3, {4.0, 1.0, 0.5, 3.0, 0.25, 2.0}, {1.0, -1.0, 0.5}},
      {3, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}},
      // Negative semidefinite: no positive eigenvalue.
      {3, {-1.0, 0.0, 0.0, -2.0, 0.0, 0.0}, {1.0, 1.0, 1.0}},
      // At the ends of the range.
      {3, {0x1p+1000, 0.0, 0.0, 0x1p+1000, 0.0, 0x1p+1000}, {0x1p+1000, 0.0, 0x1p+999}},
      {3, {0x1p-1000, 0.0, 0.0, 0x1p-1000, 0.0, 0x1p-1000}, {0x1p-1000, 0.0, 0x1p-1001}},
      {2, {0x1p-1000, 0.0, 0.0, 0x1p+1000, 0.0, 0.0}, {1.0, 1.0, 0.0}},
      // Refused: a value that is not finite, a solution beyond the range.
      {3, {value_of(kNaNBits), 0.0, 0.0, 1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}},
      {2, {1.0, 0.0, 0.0, 1.0, 0.0, 0.0}, {value_of(kInfinityBits), 1.0, 0.0}},
      {1, {0x1p-1000, 0.0, 0.0, 0.0, 0.0, 0.0}, {0x1p+1000, 0.0, 0.0}}};
  IntegerGenerator generator{0xb7e151628aed2a6bULL};
  for (int row = 0; row < 48; ++row) {
    const int order = 1 + row % 3;
    std::int64_t vectors[3][3];
    for (auto& vector : vectors) {
      for (std::int64_t& entry : vector) entry = generator.integer(-8, 8);
    }
    // Every fourth set is dependent: its last vector is a combination of the
    // others (or zero for a single vector).
    if (row % 4 == 3) {
      for (int axis = 0; axis < 3; ++axis) {
        vectors[order - 1][axis] =
            order == 1 ? 0 : (order == 2 ? 2 * vectors[0][axis] : vectors[0][axis] - vectors[1][axis]);
      }
    }
    inputs.push_back(
        gram_system(order, vectors, power_of_two(static_cast<int>(generator.integer(-8, 8)))));
  }
  return inputs;
}

std::vector<std::array<double, 6>> cholesky_inputs() {
  std::vector<std::array<double, 6>> inputs = {
      {1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
      {4.0, 0.0, 0.0, 1.0, 0.0, 0.25},
      {4.0, 1.0, 0.5, 3.0, 0.25, 2.0},
      {0x1p+1000, 0.0, 0.0, 0x1p+1000, 0.0, 0x1p+1000},
      {0x1p-1000, 0.0, 0.0, 0x1p-1000, 0.0, 0x1p-1000},
      // Refused: indefinite, singular, zero, negative and non-finite matrices.
      {1.0, 2.0, 0.0, 1.0, 0.0, 1.0},
      {1.0, 1.0, 0.0, 1.0, 0.0, 1.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
      {1.0, 0.0, 0.0, 1.0, 0.0, -1.0},
      {value_of(kNaNBits), 0.0, 0.0, 1.0, 0.0, 1.0},
      {1.0, 0.0, value_of(kInfinityBits), 1.0, 0.0, 1.0},
      {value_of(kInfinityBits), 0.0, 0.0, 1.0, 0.0, 1.0}};
  IntegerGenerator generator{0x71f1bd5d1a3c9f0bULL};
  for (int row = 0; row < 32; ++row) {
    std::int64_t a[9];
    for (std::int64_t& entry : a) entry = generator.integer(-4, 4);
    const double scale = power_of_two(static_cast<int>(generator.integer(-6, 6)));
    // A^T A plus the identity; every fourth matrix without it (it may be singular).
    const std::int64_t shift = row % 4 == 3 ? 0 : 1;
    std::array<double, 6> matrix{};
    std::size_t slot = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        std::int64_t sum = i == j ? shift : 0;
        for (int k = 0; k < 3; ++k) sum += a[3 * k + i] * a[3 * k + j];
        matrix[slot++] = static_cast<double>(sum) * scale;
      }
    }
    inputs.push_back(matrix);
  }
  return inputs;
}

struct VectorSet {
  int count;
  std::array<double, 9> vectors;
};

std::vector<VectorSet> basis_inputs() {
  std::vector<VectorSet> inputs = {
      {1, {3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      {2, {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0}},
      {3, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}},
      {3, {1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0}},
      // Dependent, zero and non-finite vectors add no direction.
      {2, {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0}},
      {3, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0}},
      {3, {0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0}},
      {2, {value_of(kNaNBits), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}},
      {2, {0x1p+600, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}},
      // Nearly dependent, on both sides of the cutoff 2^-40 of the squared length.
      {2, {1.0, 0.0, 0.0, 1.0, 0x1p-19, 0.0, 0.0, 0.0, 0.0}},
      {2, {1.0, 0.0, 0.0, 1.0, 0x1p-21, 0.0, 0.0, 0.0, 0.0}},
      {1, {0x1p-500, 0x1p-500, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      {0, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};
  IntegerGenerator generator{0x9216d5d98979fb1bULL};
  for (int row = 0; row < 32; ++row) {
    VectorSet set{1 + row % 3, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    const double scale = power_of_two(static_cast<int>(generator.integer(-10, 10)));
    for (int index = 0; index < 3 * set.count; ++index) {
      set.vectors[static_cast<std::size_t>(index)] =
          static_cast<double>(generator.integer(-16, 16)) * scale;
    }
    inputs.push_back(set);
  }
  return inputs;
}

std::vector<BitRow> evaluate() {
  namespace kernels = phx::mc::deterministic;
  std::vector<BitRow> rows;
  for (const double input : log1p_inputs()) {
    rows.push_back({"PHX_DS_LOG1P", {bits_of(input), bits_of(kernels::log1p(input))}});
  }
  for (const System& system : pseudo_inputs()) {
    // The upper triangle of the leading block as a row-major order x order array.
    const std::size_t slot[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    double matrix[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int row = 0; row < system.order; ++row) {
      for (int column = row; column < system.order; ++column) {
        matrix[system.order * row + column] = system.matrix[slot[row][column]];
      }
    }
    double solution[3] = {0.0, 0.0, 0.0};
    const int rank =
        kernels::symmetric_pseudo_solve(system.order, matrix, system.rhs.data(), solution);
    BitRow record{"PHX_DS_PSEUDO", {static_cast<std::uint64_t>(system.order)}};
    append_bits(record, system.matrix.data(), 6);
    append_bits(record, system.rhs.data(), 3);
    // -1 (no solution) is recorded as 255.
    record.fields.push_back(rank < 0 ? 255U : static_cast<std::uint64_t>(rank));
    append_bits(record, solution, 3);
    rows.push_back(record);
  }
  for (const std::array<double, 6>& a : cholesky_inputs()) {
    const double matrix[9] = {a[0], a[1], a[2], 0.0, a[3], a[4], 0.0, 0.0, a[5]};
    double factor[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const bool valid = kernels::cholesky3(matrix, factor);
    BitRow record{"PHX_DS_CHOLESKY", {}};
    append_bits(record, a.data(), 6);
    record.fields.push_back(valid ? 1U : 0U);
    const double upper[6] = {factor[0], factor[1], factor[2], factor[4], factor[5], factor[8]};
    append_bits(record, upper, 6);
    rows.push_back(record);
    if (valid) {
      // U v and its inverse on a fixed vector follow from the factor alone.
      const double vector[3] = {1.0, -2.0, 0.5};
      double product[3];
      double solved[3];
      kernels::multiply_upper3(factor, vector, product);
      kernels::solve_upper3(factor, vector, solved);
      BitRow action{"PHX_DS_CHOLESKY", {}};
      append_bits(action, a.data(), 6);
      action.fields.push_back(2U);
      append_bits(action, product, 3);
      append_bits(action, solved, 3);
      rows.push_back(action);
    }
  }
  for (const VectorSet& set : basis_inputs()) {
    double basis[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int size = kernels::orthonormal_basis3(set.count, set.vectors.data(), basis);
    BitRow record{"PHX_DS_BASIS", {static_cast<std::uint64_t>(set.count)}};
    append_bits(record, set.vectors.data(), 9);
    record.fields.push_back(static_cast<std::uint64_t>(size));
    append_bits(record, basis, 9);
    rows.push_back(record);
  }
  return rows;
}

void emit(const std::vector<BitRow>& rows) {
  std::printf(
      "// Bit vectors of log1p and the small solves of the deterministic kernels,\n"
      "// written by `test_deterministic_solve --emit`. One row per input, in the\n"
      "// order the test generates them. A value is its IEEE 754 binary64 bit\n"
      "// pattern; orders, counts, ranks and flags are integers.\n"
      "// PHX_DS_LOG1P(input, output)\n"
      "// PHX_DS_PSEUDO(order, m00, m01, m02, m11, m12, m22, three right-hand\n"
      "//               values, rank (255: no solution), three solution values)\n"
      "//   The system is the leading order x order block.\n"
      "// PHX_DS_CHOLESKY(m00, m01, m02, m11, m12, m22, 1 or 0, u00, u01, u02,\n"
      "//                 u11, u12, u22): whether M = U^T U has a factor, and it.\n"
      "// PHX_DS_CHOLESKY(the same matrix, 2, three values of U v, three values\n"
      "//                 of the x with U x = v) for v = (1, -2, 1/2).\n"
      "// PHX_DS_BASIS(count, nine vector values, size, nine basis values)\n"
      "//   Only the first `count` vectors and `size` basis vectors are data.\n");
  phx::mc::test::emit_rows(rows);
}

}  // namespace

int main(int argc, char** argv) {
  const phx::mc::GradualNearestScope mode;
  PHX_CHECK(mode.established());
  const std::vector<BitRow> rows = evaluate();
  if (argc > 1 && std::strcmp(argv[1], "--emit") == 0) {
    emit(rows);
    return 0;
  }
  phx::mc::test::compare_rows(rows, kRecorded);
  return phx::mc::test::finish("test_deterministic_solve");
}
