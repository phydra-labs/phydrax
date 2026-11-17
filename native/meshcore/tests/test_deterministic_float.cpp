//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Bit vectors of the deterministic binary64 kernels. The inputs are produced
// here: a fixed list of edge cases per kernel, then a stratified set from the
// integer generator below (every exponent the kernel accepts, a pseudo-random
// significand). deterministic_float_vectors.inc holds each input with the
// output bits the kernels produced when the file was written; this test
// requires the same bits on every platform. `--emit` prints that file.
#include <array>
#include <bit>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

enum Kernel : int { k_exp = 0, k_log = 1, k_pow_two_thirds = 2 };
constexpr const char* kKernelNames[3] = {"exp", "log", "pow_two_thirds"};

struct ScalarVector {
  int kernel;
  std::uint64_t input;
  std::uint64_t output;
};

struct EigenVector {
  std::array<std::uint64_t, 6> input;  // a00, a01, a02, a11, a12, a22
  int converged;
  std::array<std::uint64_t, 3> values;
  std::array<std::uint64_t, 9> vectors;
};

#define PHX_DF_SCALAR(kernel, input, output) {k_##kernel, input##ULL, output##ULL},
#define PHX_DF_EIGEN(...)
const std::vector<ScalarVector> kScalarVectors = {
#include "deterministic_float_vectors.inc"
};
#undef PHX_DF_SCALAR
#undef PHX_DF_EIGEN

#define PHX_DF_SCALAR(kernel, input, output)
#define PHX_DF_EIGEN(a00, a01, a02, a11, a12, a22, converged, l0, l1, l2, v0, v1, v2, v3, \
                     v4, v5, v6, v7, v8)                                                  \
  {{a00##ULL, a01##ULL, a02##ULL, a11##ULL, a12##ULL, a22##ULL},                          \
   converged,                                                                             \
   {l0##ULL, l1##ULL, l2##ULL},                                                           \
   {v0##ULL, v1##ULL, v2##ULL, v3##ULL, v4##ULL, v5##ULL, v6##ULL, v7##ULL, v8##ULL}},
const std::vector<EigenVector> kEigenVectors = {
#include "deterministic_float_vectors.inc"
};
#undef PHX_DF_SCALAR
#undef PHX_DF_EIGEN

std::uint64_t bits_of(double value) { return std::bit_cast<std::uint64_t>(value); }
double value_of(std::uint64_t bits) { return std::bit_cast<double>(bits); }

// Knuth's 64-bit linear congruential generator: integer arithmetic only.
struct Generator {
  std::uint64_t state;
  std::uint64_t next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state;
  }
  // A binary64 value with the given unbiased exponent, a pseudo-random
  // significand and, when `sign` is set, a pseudo-random sign.
  double with_exponent(int exponent, bool sign) {
    const std::uint64_t word = next();
    std::uint64_t bits = (word >> 12) | (static_cast<std::uint64_t>(exponent + 1023) << 52);
    if (sign && (word & 1U) != 0) bits |= 0x8000000000000000ULL;
    return value_of(bits);
  }
};

constexpr std::uint64_t kNaN = 0x7ff8000000000000ULL;
constexpr std::uint64_t kInfinity = 0x7ff0000000000000ULL;

std::vector<double> exp_inputs() {
  std::vector<double> inputs = {
      0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0x1p-1074, -0x1p-1074, 0x1p-54, -0x1p-54,
      // Around ln 2 / 2, where the reduction changes k.
      0x1.62e42fefa39eep-2, 0x1.62e42fefa39efp-2, 0x1.62e42fefa39f0p-2,
      -0x1.62e42fefa39eep-2, -0x1.62e42fefa39efp-2, -0x1.62e42fefa39f0p-2,
      // Around the overflow and underflow thresholds and the subnormal range.
      0x1.62e42fefa39eep+9, 0x1.62e42fefa39efp+9, 0x1.62e42fefa39f0p+9, 709.0, 710.0,
      -0x1.74910d52d3051p+9, -0x1.74910d52d3052p+9, -0x1.74910d52d3053p+9, -708.0,
      -0x1.6232bdd7abcd2p+9, -0x1.6232bdd7abcd3p+9, -720.0, -744.0, -745.0, -746.0,
      88.7, -88.7, 1.0e-8, -1.0e-8, value_of(kNaN), value_of(kInfinity),
      -value_of(kInfinity)};
  Generator generator{0x243f6a8885a308d3ULL};
  for (int exponent = -60; exponent <= 9; ++exponent) {
    for (int repeat = 0; repeat < 3; ++repeat) {
      inputs.push_back(generator.with_exponent(exponent, true));
    }
  }
  return inputs;
}

std::vector<double> log_inputs() {
  std::vector<double> inputs = {
      0.0, -0.0, 1.0, 0x1.fffffffffffffp-1, 0x1.0000000000001p+0, 2.0, 0.5, 10.0,
      // Around sqrt 2 and sqrt(1/2), where the significand is halved.
      0x1.6a09e667f3bccp+0, 0x1.6a09e667f3bcdp+0, 0x1.6a09e667f3bcep+0,
      0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bcep-1,
      0x1p-1074, 0x1.ffffffffffffep-1023, 0x1p-1022, 0x1.fffffffffffffp+1023,
      value_of(kInfinity), -1.0, -0x1p-1074, value_of(kNaN)};
  Generator generator{0x13198a2e03707344ULL};
  for (int exponent = -1022; exponent <= 1023; exponent += 7) {
    inputs.push_back(generator.with_exponent(exponent, false));
  }
  for (int exponent = -40; exponent <= 40; ++exponent) {
    inputs.push_back(generator.with_exponent(exponent, false));
  }
  // Subnormals: a pseudo-random significand below the smallest normal.
  for (int shift = 13; shift <= 63; shift += 5) {
    inputs.push_back(value_of(generator.next() >> shift));
  }
  return inputs;
}

std::vector<double> pow_two_thirds_inputs() {
  std::vector<double> inputs = {0.0,     -0.0,    1.0,     8.0,       27.0,
                                0x1p-128, 0x1p+128, 0x1p-1074, 0x1.fffffffffffffp+1023,
                                value_of(kInfinity), -1.0, value_of(kNaN)};
  Generator generator{0xa4093822299f31d0ULL};
  for (int exponent = -128; exponent <= 128; exponent += 2) {
    inputs.push_back(generator.with_exponent(exponent, false));
  }
  return inputs;
}

// Symmetric matrices as (a00, a01, a02, a11, a12, a22).
std::vector<std::array<double, 6>> eigen_inputs() {
  std::vector<std::array<double, 6>> inputs = {
      {1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0, 2.0, 0.0, 3.0},
      {3.0, 0.0, 0.0, 2.0, 0.0, 1.0},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
      {2.0, 1.0, 0.0, 2.0, 1.0, 2.0},
      {4.0, 1.0, 0.5, 3.0, 0.25, 2.0},
      // Nearly equal eigenvalues.
      {1.0, 0x1p-30, 0.0, 1.0, 0x1p-30, 1.0},
      {1.0, 0x1p-52, 0x1p-52, 0x1.0000000000001p+0, 0x1p-52, 1.0},
      // Strong anisotropy, and entries near the ends of the binary64 range.
      {0x1p+40, 1.0, 0x1p-20, 1.0, 0x1p-20, 0x1p-40},
      {0x1p-1000, 0x1p-1001, 0.0, 0x1p-1000, 0x1p-1002, 0x1p-1000},
      {0x1p+1000, 0x1p+999, 0x1p+998, 0x1p+1000, 0x1p+999, 0x1p+1000},
      {-2.0, 0.5, -0.25, -1.0, 0.125, -3.0},
      // Refused: non-finite entries.
      {value_of(kNaN), 0.0, 0.0, 1.0, 0.0, 1.0},
      {1.0, value_of(kInfinity), 0.0, 1.0, 0.0, 1.0}};
  Generator generator{0x082efa98ec4e6c89ULL};
  for (int exponent = -12; exponent <= 12; exponent += 2) {
    for (int repeat = 0; repeat < 3; ++repeat) {
      std::array<double, 6> matrix{};
      for (double& entry : matrix) entry = generator.with_exponent(exponent, true);
      inputs.push_back(matrix);
    }
  }
  return inputs;
}

struct Evaluated {
  std::vector<ScalarVector> scalars;
  std::vector<EigenVector> eigen;
};

std::vector<double> map_kernel(int kernel, const std::vector<double>& inputs) {
  std::vector<double> results(inputs.size());
  const int64_t count = static_cast<int64_t>(inputs.size());
  int32_t status = PHX_MC_INTERNAL_ERROR;
  if (kernel == k_exp) {
    status = phx_mc_deterministic_exp(count, inputs.data(), results.data());
  } else if (kernel == k_log) {
    status = phx_mc_deterministic_log(count, inputs.data(), results.data());
  } else {
    status = phx_mc_deterministic_pow_two_thirds(count, inputs.data(), results.data());
  }
  PHX_CHECK(status == PHX_MC_OK);
  return results;
}

std::vector<double> full_matrices(const std::vector<std::array<double, 6>>& inputs) {
  std::vector<double> matrices;
  matrices.reserve(9 * inputs.size());
  for (const std::array<double, 6>& a : inputs) {
    // Only the upper triangle is read: the lower one is left as zeros.
    const double full[9] = {a[0], a[1], a[2], 0.0, a[3], a[4], 0.0, 0.0, a[5]};
    matrices.insert(matrices.end(), full, full + 9);
  }
  return matrices;
}

Evaluated evaluate() {
  Evaluated evaluated;
  const std::array<std::vector<double>, 3> inputs = {exp_inputs(), log_inputs(),
                                                     pow_two_thirds_inputs()};
  for (int kernel = 0; kernel < 3; ++kernel) {
    const std::vector<double>& values = inputs[static_cast<std::size_t>(kernel)];
    const std::vector<double> results = map_kernel(kernel, values);
    for (std::size_t row = 0; row < values.size(); ++row) {
      evaluated.scalars.push_back({kernel, bits_of(values[row]), bits_of(results[row])});
      // A row does not depend on the rows evaluated with it.
      const std::vector<double> alone = map_kernel(kernel, {values[row]});
      PHX_CHECK(bits_of(alone[0]) == bits_of(results[row]));
    }
  }
  const std::vector<std::array<double, 6>> symmetric = eigen_inputs();
  const std::vector<double> matrices = full_matrices(symmetric);
  const std::size_t count = symmetric.size();
  std::vector<double> values(3 * count), vectors(9 * count);
  std::vector<int8_t> converged(count);
  PHX_CHECK(phx_mc_deterministic_symmetric_eigen3(static_cast<int64_t>(count), matrices.data(),
                                                  values.data(), vectors.data(),
                                                  converged.data()) == PHX_MC_OK);
  for (std::size_t row = 0; row < count; ++row) {
    EigenVector vector{};
    for (std::size_t k = 0; k < 6; ++k) vector.input[k] = bits_of(symmetric[row][k]);
    vector.converged = converged[row];
    for (std::size_t k = 0; k < 3; ++k) vector.values[k] = bits_of(values[3 * row + k]);
    for (std::size_t k = 0; k < 9; ++k) vector.vectors[k] = bits_of(vectors[9 * row + k]);
    evaluated.eigen.push_back(vector);
    std::array<double, 3> alone_values{};
    std::array<double, 9> alone_vectors{};
    int8_t alone_converged = -1;
    PHX_CHECK(phx_mc_deterministic_symmetric_eigen3(1, matrices.data() + 9 * row,
                                                    alone_values.data(), alone_vectors.data(),
                                                    &alone_converged) == PHX_MC_OK);
    PHX_CHECK(alone_converged == converged[row]);
    for (std::size_t k = 0; k < 3; ++k) {
      PHX_CHECK(bits_of(alone_values[k]) == vector.values[k]);
    }
    for (std::size_t k = 0; k < 9; ++k) {
      PHX_CHECK(bits_of(alone_vectors[k]) == vector.vectors[k]);
    }
  }
  return evaluated;
}

void emit(const Evaluated& evaluated) {
  std::printf("// Bit vectors of the deterministic binary64 kernels, written by\n"
              "// `test_deterministic_float --emit`. One row per input, in the order the\n"
              "// test generates them; every field is the IEEE 754 binary64 bit pattern.\n"
              "// PHX_DF_SCALAR(kernel, input, output)\n"
              "// PHX_DF_EIGEN(a00, a01, a02, a11, a12, a22, converged, three eigenvalues,\n"
              "//              nine row-major eigenvector entries)\n");
  for (const ScalarVector& row : evaluated.scalars) {
    std::printf("PHX_DF_SCALAR(%s, 0x%016" PRIx64 ", 0x%016" PRIx64 ")\n",
                kKernelNames[row.kernel], row.input, row.output);
  }
  for (const EigenVector& row : evaluated.eigen) {
    std::printf("PHX_DF_EIGEN(");
    for (const std::uint64_t value : row.input) std::printf("0x%016" PRIx64 ", ", value);
    std::printf("%d", row.converged);
    for (const std::uint64_t value : row.values) std::printf(", 0x%016" PRIx64, value);
    for (const std::uint64_t value : row.vectors) std::printf(", 0x%016" PRIx64, value);
    std::printf(")\n");
  }
}

void compare(const Evaluated& evaluated) {
  PHX_CHECK(evaluated.scalars.size() == kScalarVectors.size());
  PHX_CHECK(evaluated.eigen.size() == kEigenVectors.size());
  int reported = 0;
  for (std::size_t row = 0; row < evaluated.scalars.size() && row < kScalarVectors.size();
       ++row) {
    const ScalarVector& got = evaluated.scalars[row];
    const ScalarVector& expected = kScalarVectors[row];
    const bool same = got.kernel == expected.kernel && got.input == expected.input &&
                      got.output == expected.output;
    PHX_CHECK(same);
    if (!same && reported++ < 8) {
      std::fprintf(stderr,
                   "  %s row %zu: input 0x%016" PRIx64 " (recorded 0x%016" PRIx64
                   "), output 0x%016" PRIx64 " (recorded 0x%016" PRIx64 ")\n",
                   kKernelNames[got.kernel], row, got.input, expected.input, got.output,
                   expected.output);
    }
  }
  for (std::size_t row = 0; row < evaluated.eigen.size() && row < kEigenVectors.size(); ++row) {
    const EigenVector& got = evaluated.eigen[row];
    const EigenVector& expected = kEigenVectors[row];
    const bool same = got.input == expected.input && got.converged == expected.converged &&
                      got.values == expected.values && got.vectors == expected.vectors;
    PHX_CHECK(same);
    if (!same && reported++ < 8) {
      std::fprintf(stderr, "  eigen row %zu differs from its recorded bits\n", row);
    }
  }
}

void guards_arguments() {
  double value = 1.0;
  double result = 0.0;
  PHX_CHECK(phx_mc_deterministic_exp(0, nullptr, nullptr) == PHX_MC_OK);
  PHX_CHECK(phx_mc_deterministic_exp(-1, &value, &result) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_deterministic_exp(1, nullptr, &result) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_deterministic_log(1, &value, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_deterministic_pow_two_thirds(1, nullptr, nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_deterministic_symmetric_eigen3(1, nullptr, nullptr, nullptr, nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_deterministic_symmetric_eigen3(0, nullptr, nullptr, nullptr, nullptr) ==
            PHX_MC_OK);
}

}  // namespace

int main(int argc, char** argv) {
  const Evaluated evaluated = evaluate();
  if (argc > 1 && std::strcmp(argv[1], "--emit") == 0) {
    emit(evaluated);
    return 0;
  }
  int32_t failed_check = -1;
  PHX_CHECK(phx_mc_deterministic_float_self_test(&failed_check) == PHX_MC_OK);
  if (failed_check != 0) std::fprintf(stderr, "  self test failed check %d\n", failed_check);
  PHX_CHECK(phx_mc_deterministic_float_self_test(nullptr) == PHX_MC_INVALID_ARGUMENT);
  compare(evaluated);
  guards_arguments();
  return phx::mc::test::finish("test_deterministic_float");
}
