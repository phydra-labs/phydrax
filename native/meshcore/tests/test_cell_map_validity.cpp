// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// phx_mc_cell_map_validity against values of the rational per-cell proof
// (phydrax _certify_source_block): statuses, directed determinant endpoints
// and BLAKE2b-128 exact-map digests of P1 tetrahedra and a Q1 hexahedron.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Basis {
  std::vector<int64_t> starts{0};
  std::vector<int32_t> exponents;
  std::vector<int64_t> numerators;

  void add(std::initializer_list<std::array<int, 4>> terms) {
    for (const auto& term : terms) {
      exponents.insert(exponents.end(), {term[0], term[1], term[2]});
      numerators.push_back(term[3]);
    }
    starts.push_back(static_cast<int64_t>(numerators.size()));
  }
};

// P1 tetrahedron: 1 - x - y - z, x, y, z (terms in stored order).
Basis p1() {
  Basis basis;
  basis.add({{0, 0, 0, 1}, {0, 0, 1, -1}, {0, 1, 0, -1}, {1, 0, 0, -1}});
  basis.add({{1, 0, 0, 1}});
  basis.add({{0, 1, 0, 1}});
  basis.add({{0, 0, 1, 1}});
  return basis;
}

// Q1 hexahedron in phydrax's dof order.
Basis q1() {
  Basis basis;
  basis.add({{0, 0, 0, 1}, {0, 0, 1, -1}, {0, 1, 0, -1}, {0, 1, 1, 1},
             {1, 0, 0, -1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, -1}});
  basis.add({{1, 0, 0, 1}, {1, 0, 1, -1}, {1, 1, 0, -1}, {1, 1, 1, 1}});
  basis.add({{1, 1, 0, 1}, {1, 1, 1, -1}});
  basis.add({{0, 1, 0, 1}, {0, 1, 1, -1}, {1, 1, 0, -1}, {1, 1, 1, 1}});
  basis.add({{0, 0, 1, 1}, {0, 1, 1, -1}, {1, 0, 1, -1}, {1, 1, 1, 1}});
  basis.add({{1, 0, 1, 1}, {1, 1, 1, -1}});
  basis.add({{1, 1, 1, 1}});
  basis.add({{0, 1, 1, 1}, {1, 1, 1, -1}});
  return basis;
}

// Canonical dyadic bank entries s 2^e with odd s (zero is 0, 0).
void dyadic(const std::vector<double>& values, std::vector<int64_t>& significands,
            std::vector<int16_t>& exponents) {
  for (const double value : values) {
    if (value == 0.0) {
      significands.push_back(0);
      exponents.push_back(0);
      continue;
    }
    int binary = 0;
    const double fraction = std::frexp(value, &binary);
    auto significand = static_cast<int64_t>(std::ldexp(fraction, 53));
    int exponent = binary - 53;
    while (significand % 2 == 0) {
      significand /= 2;
      ++exponent;
    }
    significands.push_back(significand);
    exponents.push_back(static_cast<int16_t>(exponent));
  }
}

std::string hex(const uint8_t* bytes) {
  std::string out;
  char buffer[3];
  for (int index = 0; index < 16; ++index) {
    std::snprintf(buffer, sizeof(buffer), "%02x", bytes[index]);
    out += buffer;
  }
  return out;
}

struct Result {
  std::vector<int8_t> decided;
  std::vector<int32_t> status;
  std::vector<double> lower, upper;
  std::vector<uint8_t> digests;
  int32_t code = 0;
};

Result run(int32_t domain, const Basis& basis, const char* kind, const std::vector<double>& points,
           const std::vector<int64_t>& routes, int16_t sparse_row = -1) {
  std::vector<int64_t> significands;
  std::vector<int16_t> exponents;
  dyadic(points, significands, exponents);
  if (sparse_row >= 0) {
    significands[3 * static_cast<std::size_t>(sparse_row)] = 0;
    exponents[3 * static_cast<std::size_t>(sparse_row)] = -32768;
  }
  const int64_t dofs = static_cast<int64_t>(basis.starts.size()) - 1;
  const int64_t cells = static_cast<int64_t>(routes.size()) / dofs;
  Result result;
  result.decided.assign(cells, -1);
  result.status.assign(cells, -1);
  result.lower.assign(cells, 0.0);
  result.upper.assign(cells, 0.0);
  result.digests.assign(16 * cells, 0);
  result.code = phx_mc_cell_map_validity(
      domain, dofs, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
      kind, static_cast<int64_t>(std::string(kind).size()),
      static_cast<int64_t>(points.size() / 3), significands.data(), exponents.data(), -32768,
      cells, routes.data(), 0x1p-35, result.decided.data(), result.status.data(),
      result.lower.data(), result.upper.data(), result.digests.data());
  return result;
}

const std::vector<double> kCube{0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
                                0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};

void check_tetrahedra() {
  const Result result = run(0, p1(), "tetrahedron", kCube, {0, 1, 2, 4, 0, 2, 1, 4});
  PHX_CHECK(result.code == PHX_MC_OK);
  PHX_CHECK(result.decided[0] == 1 && result.decided[1] == 1);
  PHX_CHECK(result.status[0] == 0 && result.lower[0] == 1.0 && result.upper[0] == 1.0);
  PHX_CHECK(result.status[1] == 1 && result.lower[1] == -1.0 && result.upper[1] == -1.0);
  PHX_CHECK(hex(result.digests.data()) == "cc75244e7a32dac1decef45e390166f1");
  PHX_CHECK(hex(result.digests.data() + 16) == "2f61f073824f397fc22dc5de336fbb89");
  // A non-dyadic entry leaves the cell to the rational proof.
  const Result sparse = run(0, p1(), "tetrahedron", kCube, {0, 1, 2, 4}, 2);
  PHX_CHECK(sparse.code == PHX_MC_OK && sparse.decided[0] == 0);
}

void check_hexahedron() {
  // Box scaled by (1, 1/2, 1/4) with vertex (0, 1, 1) moved by (1/4, 1/8, 0).
  std::vector<double> points;
  for (int vertex = 0; vertex < 8; ++vertex) {
    points.push_back(kCube[3 * vertex] * 1.0);
    points.push_back(kCube[3 * vertex + 1] * 0.5);
    points.push_back(kCube[3 * vertex + 2] * 0.25);
  }
  points[3 * 6] += 0.25;
  points[3 * 6 + 1] += 0.125;
  const Result result = run(1, q1(), "hexahedron", points, {0, 1, 3, 2, 4, 5, 7, 6});
  PHX_CHECK(result.code == PHX_MC_OK && result.decided[0] == 1);
  PHX_CHECK(result.status[0] == 0);
  PHX_CHECK(result.lower[0] == 0.09375 && result.upper[0] == 0.15625);
  PHX_CHECK(hex(result.digests.data()) == "9172011596d56359320eb3c1f12fd62d");
  // Swapping two vertices inverts a corner.
  const Result inverted = run(1, q1(), "hexahedron", points, {1, 0, 3, 2, 5, 4, 7, 6});
  PHX_CHECK(inverted.code == PHX_MC_OK && inverted.decided[0] == 1 && inverted.status[0] == 1);
}

void check_arguments() {
  const Basis basis = p1();
  std::vector<int64_t> significands(12, 0);
  std::vector<int16_t> exponents(12, 0);
  const std::vector<int64_t> outside{0, 1, 2, 9};
  int8_t decided = 0;
  int32_t status = 0;
  double lower = 0.0, upper = 0.0;
  uint8_t digest[16] = {};
  PHX_CHECK(phx_mc_cell_map_validity(2, 4, basis.starts.data(), basis.exponents.data(),
                                     basis.numerators.data(), 0, "tetrahedron", 11, 4,
                                     significands.data(), exponents.data(), -32768, 1,
                                     outside.data(), 0.0, &decided, &status, &lower, &upper,
                                     digest) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_cell_map_validity(0, 4, basis.starts.data(), basis.exponents.data(),
                                     basis.numerators.data(), 0, "tetrahedron", 11, 4,
                                     significands.data(), exponents.data(), -32768, 1,
                                     outside.data(), 0.0, &decided, &status, &lower, &upper,
                                     digest) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_cell_map_validity(0, 4, basis.starts.data(), basis.exponents.data(),
                                     basis.numerators.data(), 0, "tet'", 4, 4,
                                     significands.data(), exponents.data(), -32768, 0,
                                     outside.data(), 0.0, &decided, &status, &lower, &upper,
                                     digest) == PHX_MC_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  check_tetrahedra();
  check_hexahedron();
  check_arguments();
  return phx::mc::test::finish("test_cell_map_validity");
}
