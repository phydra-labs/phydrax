// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// phx_mc_certify_cell_maps against values of the rational per-cell proof
// (phydrax _certify_source_block): statuses, directed determinant endpoints
// and BLAKE2b-128 exact-map digests of P1 tetrahedra and a Q1 hexahedron.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

// P1 triangle in 2-D: 1 - x - y, x, y (terms in stored order, z exponent 0).
Basis p1_2d() {
  Basis basis;
  basis.add({{0, 0, 0, 1}, {0, 1, 0, -1}, {1, 0, 0, -1}});
  basis.add({{1, 0, 0, 1}});
  basis.add({{0, 1, 0, 1}});
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
           const std::vector<int64_t>& routes, int16_t sparse_row = -1,
           int32_t parallelism = 1) {
  const int ambient = domain == 3 ? 2 : 3;
  std::vector<int64_t> significands;
  std::vector<int16_t> exponents;
  dyadic(points, significands, exponents);
  if (sparse_row >= 0) {
    significands[static_cast<std::size_t>(ambient) * static_cast<std::size_t>(sparse_row)] = 0;
    exponents[static_cast<std::size_t>(ambient) * static_cast<std::size_t>(sparse_row)] = -32768;
  }
  const int64_t dofs = static_cast<int64_t>(basis.starts.size()) - 1;
  const int64_t cells = static_cast<int64_t>(routes.size()) / dofs;
  Result result;
  result.decided.assign(cells, -1);
  result.status.assign(cells, -1);
  result.lower.assign(cells, 0.0);
  result.upper.assign(cells, 0.0);
  result.digests.assign(16 * cells, 0);
  result.code = phx_mc_certify_cell_maps(
      domain, dofs, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
      kind, static_cast<int64_t>(std::string(kind).size()),
      static_cast<int64_t>(points.size() / static_cast<std::size_t>(ambient)),
      significands.data(), exponents.data(), -32768,
      0, nullptr, nullptr, 0, nullptr,
      cells, routes.data(), 0x1p-35, result.decided.data(), result.status.data(),
      result.lower.data(), result.upper.data(), result.digests.data(), parallelism);
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

void check_wide_tetrahedra() {
  constexpr int16_t kSparse = std::numeric_limits<int16_t>::min();
  constexpr std::size_t kCellCount = 5;
  const Basis basis = p1();
  // Wide entries: + (1 + 2^-80) at y2 and - (1 + 2^-81) at y4.
  std::array<int64_t, 21> significands{
      0, 0, 0,  1, 1, 0,  1, 1, 0,  0, 0, 1,  1, -2, 0,  1, 0, 0,  0, 0, 0};
  std::array<int16_t, 21> exponents{};
  exponents[7] = exponents[13] = exponents[18] = kSparse;
  std::array<int16_t, 2> wide_exponents{-80, -81};
  std::array<int64_t, 3> wide_offsets{0, 2, 4};
  std::array<uint64_t, 4> wide_limbs{1, 1u << 16, 1, 1u << 17};
  const std::array<int64_t, 4 * kCellCount> routes{
      0, 1, 2, 3,  0, 5, 2, 3,  0, 2, 4, 3,  0, 5, 1, 3,  0, 5, 6, 3};
  std::array<int8_t, kCellCount> decided{};
  std::array<int32_t, kCellCount> status{};
  std::array<double, kCellCount> lower{}, upper{};
  std::array<uint8_t, 16 * kCellCount> digests{};
  const auto certify = [&](double floor) {
    return phx_mc_certify_cell_maps(
        0, 4, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
        "tetrahedron", 11, 7, significands.data(), exponents.data(), kSparse,
        2, wide_exponents.data(), wide_offsets.data(), 4, wide_limbs.data(), kCellCount,
        routes.data(), floor, decided.data(), status.data(), lower.data(), upper.data(),
        digests.data(), 1);
  };

  PHX_CHECK(certify(0.0) == PHX_MC_OK);
  PHX_CHECK((decided == std::array<int8_t, kCellCount>{1, 1, 1, 1, 0}));
  // Cancellation leaves the 80th bit, so the first cell is not flat.
  PHX_CHECK(status[0] == 0 && lower[0] == 0x1p-80 && upper[0] == 0x1p-80);
  PHX_CHECK(status[1] == 0 && lower[1] == 1.0 && upper[1] == std::nextafter(1.0, 2.0));
  PHX_CHECK(status[2] == 1 && lower[2] == std::nextafter(-2.0, -3.0) && upper[2] == -2.0);
  PHX_CHECK(status[3] == 0 && lower[3] == 1.0 && upper[3] == 1.0);
  PHX_CHECK(hex(digests.data()) == "0b1ffa4809d3483da45318ad07fc452a");
  PHX_CHECK(hex(digests.data() + 16) == "e55f52da8fded22fbadbb932298a66b1");
  PHX_CHECK(hex(digests.data() + 32) == "60239ced0323fdf5e4ea798250831d0f");
  PHX_CHECK(hex(digests.data() + 48) == "7f104a1bcbf59c5dd86df0a5dd5793ba");

  // The first determinant is positive but smaller than floor times the
  // product of column lengths. Rounding y2 to 1 instead makes it flat.
  PHX_CHECK(certify(0x1p-80) == PHX_MC_OK);
  PHX_CHECK(decided[0] == 1 && status[0] == 1 && lower[0] == 0x1p-80);
  exponents[7] = 0;
  PHX_CHECK(certify(0.0) == PHX_MC_OK);
  PHX_CHECK(decided[0] == 1 && status[0] == 1 && lower[0] == 0.0 && upper[0] == 0.0);
  exponents[7] = kSparse;

  // An invalid reference on the last cell must not write even earlier results.
  decided.fill(-1);
  status.fill(-1);
  lower.fill(-1.0);
  upper.fill(-1.0);
  digests.fill(0);
  for (int64_t bad : {int64_t{3}, int64_t{-3}, std::numeric_limits<int64_t>::min()}) {
    significands[18] = bad;
    PHX_CHECK(certify(0.0) == PHX_MC_INVALID_INPUT);
  }
  significands[18] = 0;
  wide_offsets[1] = 5;
  PHX_CHECK(certify(0.0) == PHX_MC_INVALID_INPUT);
  wide_offsets[1] = 2;
  wide_offsets[2] = 3;
  PHX_CHECK(certify(0.0) == PHX_MC_INVALID_INPUT);
  wide_offsets[2] = 4;
  wide_exponents[1] = kSparse;
  PHX_CHECK(certify(0.0) == PHX_MC_INVALID_INPUT);
  wide_exponents[1] = -81;
  wide_limbs[0] = 2;
  PHX_CHECK(certify(0.0) == PHX_MC_INVALID_INPUT);
  wide_limbs[0] = 1;
  wide_limbs[3] = 0;
  PHX_CHECK(certify(0.0) == PHX_MC_INVALID_INPUT);
  for (std::size_t cell = 0; cell < kCellCount; ++cell) {
    PHX_CHECK(decided[cell] == -1 && status[cell] == -1);
    PHX_CHECK(lower[cell] == -1.0 && upper[cell] == -1.0);
  }
  PHX_CHECK((digests == std::array<uint8_t, 16 * kCellCount>{}));
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
  PHX_CHECK(phx_mc_certify_cell_maps(
                4, 4, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
                "tetrahedron", 11, 4, significands.data(), exponents.data(), -32768,
                0, nullptr, nullptr, 0, nullptr, 1, outside.data(), 0.0, &decided, &status,
                &lower, &upper, digest, 1) == PHX_MC_INVALID_ARGUMENT);
  // 2-D domain refuses basis terms with non-zero z exponent.
  PHX_CHECK(phx_mc_certify_cell_maps(
                3, 4, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
                "triangle", 8, 4, significands.data(), exponents.data(), -32768,
                0, nullptr, nullptr, 0, nullptr, 1, outside.data(), 0.0, &decided, &status,
                &lower, &upper, digest, 1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_certify_cell_maps(
                0, 4, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
                "tetrahedron", 11, 4, significands.data(), exponents.data(), -32768,
                0, nullptr, nullptr, 0, nullptr, 1, outside.data(), 0.0, &decided, &status,
                &lower, &upper, digest, 1) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_certify_cell_maps(
                0, 4, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
                "tet'", 4, 4, significands.data(), exponents.data(), -32768,
                0, nullptr, nullptr, 0, nullptr, 0, outside.data(), 0.0, &decided, &status,
                &lower, &upper, digest, 1) == PHX_MC_INVALID_ARGUMENT);
  // parallelism must be positive.
  PHX_CHECK(phx_mc_certify_cell_maps(
                0, 4, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
                "tetrahedron", 11, 4, significands.data(), exponents.data(), -32768,
                0, nullptr, nullptr, 0, nullptr, 0, outside.data(), 0.0, &decided, &status,
                &lower, &upper, digest, 0) == PHX_MC_INVALID_ARGUMENT);
}

// Byte-identical results for every parallelism, and across repeated runs at
// eight workers, on many cells (several fixed chunks): jittered lattice
// tetrahedra on random corner quadruples (valid, inverted and flat) and
// jittered Q1 hexahedra.
bool same(const Result& a, const Result& b) {
  const auto bytes = [](const auto& left, const auto& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(left[0])) == 0;
  };
  return a.code == b.code && bytes(a.decided, b.decided) && bytes(a.status, b.status) &&
         bytes(a.lower, b.lower) && bytes(a.upper, b.upper) && bytes(a.digests, b.digests);
}

void check_parallelism() {
  constexpr int kSide = 6;
  const auto corner = [](int x, int y, int z) {
    return static_cast<int64_t>((z * (kSide + 1) + y) * (kSide + 1) + x);
  };
  uint64_t state = 0x9e3779b97f4a7c15u;
  const auto next = [&state](uint64_t bound) {
    state = state * 6364136223846793005u + 1442695040888963407u;
    return (state >> 33) % bound;
  };
  std::vector<double> points;
  for (int z = 0; z <= kSide; ++z) {
    for (int y = 0; y <= kSide; ++y) {
      for (int x = 0; x <= kSide; ++x) {
        for (const int value : {x, y, z}) {
          points.push_back(value + (static_cast<double>(next(13)) - 6.0) / 16.0);
        }
      }
    }
  }
  std::vector<int64_t> tetrahedra, hexahedra;
  for (int z = 0; z < kSide; ++z) {
    for (int y = 0; y < kSide; ++y) {
      for (int x = 0; x < kSide; ++x) {
        const std::array<int64_t, 8> cube{corner(x, y, z),         corner(x + 1, y, z),
                                          corner(x, y + 1, z),     corner(x + 1, y + 1, z),
                                          corner(x, y, z + 1),     corner(x + 1, y, z + 1),
                                          corner(x, y + 1, z + 1), corner(x + 1, y + 1, z + 1)};
        for (const int order : {0, 1, 3, 2, 4, 5, 7, 6}) hexahedra.push_back(cube[order]);
        for (int tet = 0; tet < 2; ++tet) {
          for (int vertex = 0; vertex < 4; ++vertex) tetrahedra.push_back(cube[next(8)]);
        }
      }
    }
  }
  const Result tet = run(0, p1(), "tetrahedron", points, tetrahedra, -1, 1);
  const Result hex = run(1, q1(), "hexahedron", points, hexahedra, -1, 1);
  PHX_CHECK(tet.code == PHX_MC_OK && hex.code == PHX_MC_OK);
  // The tetrahedra hold certified and invalid decided cells.
  int certified = 0, invalid = 0;
  for (std::size_t cell = 0; cell < tet.decided.size(); ++cell) {
    certified += tet.decided[cell] == 1 && tet.status[cell] == 0;
    invalid += tet.decided[cell] == 1 && tet.status[cell] == 1;
  }
  PHX_CHECK(certified > 0 && invalid > 0);
  for (const int32_t parallelism : {2, 3, 4, 8, 8, 8, 8}) {
    PHX_CHECK(same(tet, run(0, p1(), "tetrahedron", points, tetrahedra, -1, parallelism)));
    PHX_CHECK(same(hex, run(1, q1(), "hexahedron", points, hexahedra, -1, parallelism)));
  }
}

void check_triangles_2d() {
  const std::vector<double> square{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5};
  // Positive, inverted, and degenerate 2-D affine triangles.
  const Result result = run(3, p1_2d(), "triangle", square, {0, 1, 2, 0, 2, 1, 0, 3, 4});
  PHX_CHECK(result.code == PHX_MC_OK);
  PHX_CHECK(result.decided[0] == 1 && result.decided[1] == 1 && result.decided[2] == 1);
  PHX_CHECK(result.status[0] == 0 && result.lower[0] == 1.0 && result.upper[0] == 1.0);
  PHX_CHECK(result.status[1] == 1 && result.lower[1] == -1.0 && result.upper[1] == -1.0);
  PHX_CHECK(result.status[2] == 1 && result.lower[2] == 0.0 && result.upper[2] == 0.0);
  // Sparse (non-dyadic) entry defers the cell to the rational proof.
  const Result sparse = run(3, p1_2d(), "triangle", square, {0, 1, 2}, 2);
  PHX_CHECK(sparse.code == PHX_MC_OK && sparse.decided[0] == 0);

  // Wide dyadic entry with cancellation leaving 2^-80.
  constexpr int16_t kSparse = std::numeric_limits<int16_t>::min();
  const Basis basis = p1_2d();
  // p0=(0,0), p1=(1,1), p2=(1, 1 + 2^-80), p3=(1,0), p4=(0,1)
  std::array<int64_t, 10> significands{0, 0,  1, 1,  1, 1,  1, 0,  0, 1};
  std::array<int16_t, 10> exponents{};
  exponents[5] = kSparse;
  std::array<int16_t, 1> wide_exponents{-80};
  std::array<int64_t, 2> wide_offsets{0, 2};
  std::array<uint64_t, 2> wide_limbs{1, 1u << 16};
  const std::array<int64_t, 6> routes{0, 1, 2,  0, 3, 4};
  std::array<int8_t, 2> decided{};
  std::array<int32_t, 2> status{};
  std::array<double, 2> lower{}, upper{};
  std::array<uint8_t, 32> digests{};
  PHX_CHECK(phx_mc_certify_cell_maps(
                3, 3, basis.starts.data(), basis.exponents.data(), basis.numerators.data(), 0,
                "triangle", 8, 5, significands.data(), exponents.data(), kSparse,
                1, wide_exponents.data(), wide_offsets.data(), 2, wide_limbs.data(),
                2, routes.data(), 0.0, decided.data(), status.data(), lower.data(), upper.data(),
                digests.data(), 1) == PHX_MC_OK);
  PHX_CHECK(decided[0] == 1 && status[0] == 0 && lower[0] == 0x1p-80 && upper[0] == 0x1p-80);
  PHX_CHECK(decided[1] == 1 && status[1] == 0 && lower[1] == 1.0 && upper[1] == 1.0);
  // Parallelism invariance on a 2-D grid.
  std::vector<double> grid;
  std::vector<int64_t> tris;
  constexpr int kGrid = 8;
  for (int y = 0; y <= kGrid; ++y) {
    for (int x = 0; x <= kGrid; ++x) {
      grid.push_back(static_cast<double>(x) + (y % 2 ? 0.125 : 0.0));
      grid.push_back(static_cast<double>(y) - (x % 2 ? 0.0625 : 0.0));
    }
  }
  for (int y = 0; y < kGrid; ++y) {
    for (int x = 0; x < kGrid; ++x) {
      const int64_t a = y * (kGrid + 1) + x;
      const int64_t b = a + 1;
      const int64_t c = a + (kGrid + 1);
      const int64_t d = c + 1;
      tris.insert(tris.end(), {a, b, d, a, d, c});
    }
  }
  const Result serial = run(3, p1_2d(), "triangle", grid, tris, -1, 1);
  PHX_CHECK(serial.code == PHX_MC_OK);
  for (const int32_t parallelism : {2, 4, 8}) {
    PHX_CHECK(same(serial, run(3, p1_2d(), "triangle", grid, tris, -1, parallelism)));
  }
}

}  // namespace

int main() {
  check_tetrahedra();
  check_wide_tetrahedra();
  check_hexahedron();
  check_arguments();
  check_parallelism();
  check_triangles_2d();
  return phx::mc::test::finish("test_cell_map_validity");
}
