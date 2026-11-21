// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// phx_mc_polyhedral_star_validity and phx_mc_polyhedral_star_decomposition on
// the unit cube (values of the rational proofs worked by hand) and refusals.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

const std::vector<int64_t> kCube = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
                                    0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
// Outward loops (counterclockwise seen from outside).
const std::vector<std::vector<int64_t>> kFaces = {
    {0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};

struct Bank {
  std::vector<int64_t> offsets{0};
  std::vector<uint32_t> words;
  std::vector<int8_t> signs;

  void add(int64_t value) {
    const uint64_t magnitude = value < 0 ? 0 - static_cast<uint64_t>(value) : value;
    if (magnitude) words.push_back(static_cast<uint32_t>(magnitude));
    if (magnitude >> 32) words.push_back(static_cast<uint32_t>(magnitude >> 32));
    offsets.push_back(static_cast<int64_t>(words.size()));
    signs.push_back(value < 0 ? -1 : 1);
  }
};

struct Faces {
  std::vector<int64_t> cells{0};
  std::vector<int64_t> offsets{0};
  std::vector<int64_t> values;
};

Faces faces_of(const std::vector<std::vector<int64_t>>& faces) {
  Faces result;
  for (const auto& face : faces) {
    result.values.insert(result.values.end(), face.begin(), face.end());
    result.offsets.push_back(static_cast<int64_t>(result.values.size()));
  }
  result.cells.push_back(static_cast<int64_t>(faces.size()));
  return result;
}

struct Decomposition {
  int32_t code = -2;
  std::vector<int8_t> reversed;
  std::vector<int32_t> triangles;
};

Decomposition decompose(const std::vector<int64_t>& points,
                        const std::vector<std::vector<int64_t>>& faces) {
  Bank bank;
  for (const int64_t value : points) bank.add(value);
  bank.add(1);
  Bank face_bank;
  for (const auto& face : faces) {
    for (const int64_t vertex : face) {
      for (int axis = 0; axis < 3; ++axis) face_bank.add(points[3 * vertex + axis]);
    }
  }
  std::vector<int64_t> point_offsets = {0, static_cast<int64_t>(points.size() / 3)};
  const Faces table = faces_of(faces);
  int64_t triangles = 0;
  for (const auto& face : faces) triangles += std::max<int64_t>(face.size(), 2) - 2;
  std::vector<int64_t> triangle_offsets = {0, triangles};
  Decomposition result;
  result.reversed.assign(faces.size(), -1);
  result.triangles.assign(static_cast<std::size_t>(4 * triangles), -1);
  PHX_CHECK(phx_mc_polyhedral_star_decomposition(
                1, point_offsets.data(), bank.offsets.data(), bank.words.data(),
                bank.signs.data(), face_bank.offsets.data(), face_bank.words.data(),
                face_bank.signs.data(), table.cells.data(), table.offsets.data(),
                table.values.data(), triangle_offsets.data(), &result.code,
                result.reversed.data(), result.triangles.data()) == PHX_MC_OK);
  return result;
}

void test_cube_decomposition() {
  auto faces = kFaces;
  std::reverse(faces[1].begin(), faces[1].end());
  const Decomposition result = decompose(kCube, faces);
  PHX_CHECK(result.code == 0);
  PHX_CHECK(result.reversed == (std::vector<int8_t>{0, 1, 0, 0, 0, 0}));
  // Ear clipping of (0, 2, 3, 1): the first ear is (1, 0, 2), local (3, 0, 1).
  PHX_CHECK(result.triangles[0] == 0 && result.triangles[1] == 3 && result.triangles[2] == 0 &&
            result.triangles[3] == 1);
  PHX_CHECK(result.triangles.size() == 48);
}

void test_inward_cube_is_reoriented() {
  auto faces = kFaces;
  for (auto& face : faces) std::reverse(face.begin(), face.end());
  const Decomposition result = decompose(kCube, faces);
  PHX_CHECK(result.code == 0);
  PHX_CHECK(result.reversed == (std::vector<int8_t>(6, 1)));
}

void test_refusals() {
  auto open = kFaces;
  open.pop_back();
  PHX_CHECK(decompose(kCube, open).code == 3);
  auto repeated = kFaces;
  repeated[0] = {0, 2, 2, 1};
  PHX_CHECK(decompose(kCube, repeated).code == 2);
  // A vertex moved off its faces' planes.
  auto bent = kCube;
  for (auto& value : bent) value *= 4;
  bent[3 * 7 + 2] = 5;
  PHX_CHECK(decompose(bent, kFaces).code == 7);
}

void test_face_triangulation() {
  // An L-shaped hexagon on z = 0 (ear clipping order of the rational proof),
  // a loop with a repeated vertex and a collinear loop.
  const std::vector<int64_t> rows = {0, 0, 0, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0,
                                     0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
                                     0, 0, 0, 1, 1, 1, 2, 2, 2};
  Bank bank;
  for (const int64_t value : rows) bank.add(value);
  const std::vector<int64_t> offsets = {0, 6, 10, 13};
  std::vector<int32_t> codes(3, -2);
  std::vector<int32_t> triangles(3 * 7, -2);
  PHX_CHECK(phx_mc_face_triangulation(3, offsets.data(), bank.offsets.data(),
                                      bank.words.data(), bank.signs.data(), codes.data(),
                                      triangles.data()) == PHX_MC_OK);
  PHX_CHECK(codes[0] == 0 && codes[1] == 8 && codes[2] == 9);
  const std::vector<int32_t> expected = {0, 1, 2, 0, 2, 3, 5, 0, 3, 3, 4, 5};
  PHX_CHECK(std::equal(expected.begin(), expected.end(), triangles.begin()));
  PHX_CHECK(std::all_of(triangles.begin() + 12, triangles.end(),
                        [](int32_t value) { return value == -1; }));
}

struct Validity {
  int8_t decided = 0;
  int32_t status = -1;
  double lower = 0.0;
  double upper = 0.0;
};

Validity validity(int64_t common, bool inward) {
  Bank bank;
  for (const int64_t value : kCube) bank.add(value * common);
  bank.add(common);
  std::vector<int64_t> point_offsets = {0, 8};
  const Faces table = faces_of(kFaces);
  std::vector<int64_t> stars;
  for (std::size_t face = 0; face < kFaces.size(); ++face) {
    const auto& loop = kFaces[face];
    for (std::size_t index = 0; index < loop.size(); ++index) {
      const int64_t first = loop[index];
      const int64_t second = loop[(index + 1) % loop.size()];
      stars.insert(stars.end(), {static_cast<int64_t>(face), inward ? second : first,
                                 inward ? first : second});
    }
  }
  std::vector<int64_t> star_offsets = {0, static_cast<int64_t>(stars.size() / 3)};
  Validity result;
  PHX_CHECK(phx_mc_polyhedral_star_validity(
                1, point_offsets.data(), bank.offsets.data(), bank.words.data(),
                bank.signs.data(), table.cells.data(), table.offsets.data(), table.values.data(),
                star_offsets.data(), stars.data(), 1e-6, &result.decided, &result.status,
                &result.lower, &result.upper) == PHX_MC_OK);
  return result;
}

void test_cube_validity() {
  for (const int64_t common : {1, 3}) {
    const Validity outward = validity(common, false);
    PHX_CHECK(outward.decided == 1 && outward.status == 0);
    PHX_CHECK(outward.lower == 0.25 && outward.upper == 0.25);
    const Validity inward = validity(common, true);
    PHX_CHECK(inward.decided == 1 && inward.status == 1);
    PHX_CHECK(inward.lower == -0.25 && inward.upper == -0.25);
  }
}

}  // namespace

int main() {
  test_cube_decomposition();
  test_inward_cube_is_reoriented();
  test_refusals();
  test_face_triangulation();
  test_cube_validity();
  return phx::mc::test::finish("test_polyhedral_certificates");
}
