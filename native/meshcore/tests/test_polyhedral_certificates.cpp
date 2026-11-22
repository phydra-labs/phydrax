// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Native star validity/decomposition and declared contact of exact polyhedra.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  // Rows sized by the face sizes: a fan face has one triangle per vertex.
  int64_t capacity = 0;
  for (const auto& face : faces) capacity += static_cast<int64_t>(face.size());
  std::vector<int64_t> triangle_offsets = {0, capacity};
  Decomposition result;
  result.reversed.assign(faces.size(), -1);
  result.triangles.assign(static_cast<std::size_t>(4 * capacity), -1);
  int64_t count = -1;
  PHX_CHECK(phx_mc_polyhedral_star_decomposition(
                1, point_offsets.data(), bank.offsets.data(), bank.words.data(),
                bank.signs.data(), face_bank.offsets.data(), face_bank.words.data(),
                face_bank.signs.data(), table.cells.data(), table.offsets.data(),
                table.values.data(), triangle_offsets.data(), &result.code,
                result.reversed.data(), result.triangles.data(), &count, 1) == PHX_MC_OK);
  result.triangles.resize(static_cast<std::size_t>(4 * std::max<int64_t>(count, 0)));
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
}

void test_bent_cube_fans() {
  // A vertex moved off the top face's plane (its side faces stay planar):
  // the top face is the fan (apex -1, i, i + 1) from its exact vertex average.
  auto bent = kCube;
  for (auto& value : bent) value *= 4;
  bent[3 * 7 + 2] = 5;
  const Decomposition result = decompose(bent, kFaces);
  PHX_CHECK(result.code == 0);
  PHX_CHECK(result.triangles.size() == 4 * (4 + 5 * 2));
  PHX_CHECK(result.triangles[8] == 1 && result.triangles[9] == -1 && result.triangles[10] == 0 &&
            result.triangles[11] == 1);
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
                &result.lower, &result.upper, 1) == PHX_MC_OK);
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

// Many jittered cubes (several fixed chunks; every fifth inward, every
// seventh with a bent top face): decompositions and star verdicts are
// byte-identical for every parallelism and across repeated runs.
struct Batch {
  int32_t validity_code = 0, decomposition_code = 0;
  std::vector<int8_t> decided, reversed;
  std::vector<int32_t> status, codes, triangles;
  std::vector<double> lower, upper;
  std::vector<int64_t> counts;

  bool operator==(const Batch& other) const {
    const auto bytes = [](const auto& left, const auto& right) {
      return left.size() == right.size() &&
             std::memcmp(left.data(), right.data(), left.size() * sizeof(left[0])) == 0;
    };
    return validity_code == other.validity_code &&
           decomposition_code == other.decomposition_code && bytes(decided, other.decided) &&
           bytes(reversed, other.reversed) && bytes(status, other.status) &&
           bytes(codes, other.codes) && bytes(triangles, other.triangles) &&
           bytes(lower, other.lower) && bytes(upper, other.upper) && bytes(counts, other.counts);
  }
};

void test_parallelism() {
  constexpr int64_t kCells = 150;
  uint64_t state = 0x2545f4914f6cdd1du;
  const auto next = [&state](uint64_t bound) {
    state = state * 6364136223846793005u + 1442695040888963407u;
    return static_cast<int64_t>((state >> 33) % bound);
  };
  Bank bank, face_bank;
  std::vector<int64_t> point_offsets{0}, cell_faces{0}, face_offsets{0}, face_values, stars,
      star_offsets{0}, triangle_offsets{0};
  std::vector<int64_t> scales;
  for (int64_t cell = 0; cell < kCells; ++cell) {
    std::vector<int64_t> points;
    for (const int64_t value : kCube) points.push_back(8 * value + next(3) - 1);
    if (cell % 7 == 0) points[3 * 7 + 2] += 3;
    const int64_t common = 1 + 2 * next(2);
    for (const int64_t value : points) bank.add(value * common);
    scales.push_back(common);
    point_offsets.push_back(point_offsets.back() + 8);
    const bool inward = cell % 5 == 0;
    int64_t sizes = 0;
    for (std::size_t face = 0; face < kFaces.size(); ++face) {
      auto loop = kFaces[face];
      if (inward) std::reverse(loop.begin(), loop.end());
      for (std::size_t index = 0; index < loop.size(); ++index) {
        face_values.push_back(loop[index]);
        for (int axis = 0; axis < 3; ++axis) face_bank.add(points[3 * loop[index] + axis]);
        stars.insert(stars.end(), {static_cast<int64_t>(face), loop[index],
                                   loop[(index + 1) % loop.size()]});
      }
      face_offsets.push_back(static_cast<int64_t>(face_values.size()));
      sizes += static_cast<int64_t>(loop.size());
    }
    cell_faces.push_back(static_cast<int64_t>(face_offsets.size()) - 1);
    star_offsets.push_back(static_cast<int64_t>(stars.size() / 3));
    triangle_offsets.push_back(triangle_offsets.back() + sizes);
  }
  for (const int64_t common : scales) bank.add(common);
  const auto run = [&](int32_t parallelism) {
    Batch batch;
    batch.decided.assign(kCells, -1);
    batch.status.assign(kCells, -1);
    batch.lower.assign(kCells, 0.0);
    batch.upper.assign(kCells, 0.0);
    batch.codes.assign(kCells, -2);
    batch.counts.assign(kCells, -1);
    batch.reversed.assign(face_offsets.size() - 1, -1);
    batch.triangles.assign(static_cast<std::size_t>(4 * triangle_offsets.back()), -2);
    batch.validity_code = phx_mc_polyhedral_star_validity(
        kCells, point_offsets.data(), bank.offsets.data(), bank.words.data(), bank.signs.data(),
        cell_faces.data(), face_offsets.data(), face_values.data(), star_offsets.data(),
        stars.data(), 1e-6, batch.decided.data(), batch.status.data(), batch.lower.data(),
        batch.upper.data(), parallelism);
    batch.decomposition_code = phx_mc_polyhedral_star_decomposition(
        kCells, point_offsets.data(), bank.offsets.data(), bank.words.data(), bank.signs.data(),
        face_bank.offsets.data(), face_bank.words.data(), face_bank.signs.data(),
        cell_faces.data(), face_offsets.data(), face_values.data(), triangle_offsets.data(),
        batch.codes.data(), batch.reversed.data(), batch.triangles.data(), batch.counts.data(),
        parallelism);
    return batch;
  };
  const Batch serial = run(1);
  PHX_CHECK(serial.validity_code == PHX_MC_OK && serial.decomposition_code == PHX_MC_OK);
  int certified = 0, invalid = 0, decomposed = 0;
  for (int64_t cell = 0; cell < kCells; ++cell) {
    certified += serial.decided[cell] == 1 && serial.status[cell] == 0;
    invalid += serial.decided[cell] == 1 && serial.status[cell] == 1;
    decomposed += serial.codes[cell] == 0;
  }
  PHX_CHECK(certified > 0 && invalid > 0 && decomposed > 0);
  for (const int32_t parallelism : {2, 3, 4, 8, 8, 8, 8}) PHX_CHECK(run(parallelism) == serial);
  PHX_CHECK(run(0).validity_code == PHX_MC_INVALID_ARGUMENT);
}

void test_declared_contact_workers() {
  // Two exact tetrahedra share the z=0 face. The third occupies the second
  // tetrahedron's position but claims no shared mesh vertices with the first.
  constexpr std::array<double, 15> points{
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1};
  constexpr std::array<int64_t, 12> corners{0, 1, 2, 3, 0, 2, 1, 4, 0, 2, 1, 4};
  constexpr std::array<int64_t, 12> vertices{0, 1, 2, 3, 0, 2, 1, 4, 10, 11, 12, 13};
  constexpr std::array<int64_t, 4> cells{0, 1, 2, 3};
  constexpr std::array<int64_t, 4> offsets{0, 4, 8, 12};
  constexpr std::array<int64_t, 12> tetrahedron_faces{0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
  std::array<int64_t, 13> face_offsets{};
  std::array<int64_t, 36> face_corners{};
  for (std::size_t i = 0; i < face_offsets.size(); ++i) face_offsets[i] = 3 * i;
  for (std::size_t i = 0; i < face_corners.size(); ++i) {
    face_corners[i] = tetrahedron_faces[i % tetrahedron_faces.size()];
  }
  Bank point_bank, rows;
  for (std::size_t i = 0; i < points.size(); i += 3) {
    for (std::size_t k = 0; k < 3; ++k) point_bank.add(static_cast<int64_t>(points[i + k]));
    point_bank.add(1);
  }
  for (const auto corner : corners) {
    for (int k = 0; k < 3; ++k) rows.add(static_cast<int64_t>(points[3 * corner + k]));
  }
  for (std::size_t cell = 0; cell < 3; ++cell) rows.add(1);

  constexpr std::size_t kPairs = 513;
  std::array<int64_t, 2 * kPairs> pairs;
  std::array<int8_t, kPairs> expected;
  for (std::size_t pair = 0; pair < kPairs; ++pair) {
    const int64_t other = pair % 3 == 0 ? 2 : 1;
    const bool reverse = pair % 2 != 0;
    pairs[2 * pair] = reverse ? other : 0;
    pairs[2 * pair + 1] = reverse ? 0 : other;
    expected[pair] = other == 1 ? 1 : 0;
  }
  for (const int32_t workers : {1, 2, 3, 4, 8}) {
    std::array<int8_t, kPairs> declared;
    declared.fill(-1);
    PHX_CHECK(phx_mc_polyhedral_declared_contacts(
                  5, point_bank.offsets.data(), point_bank.words.data(), point_bank.signs.data(),
                  points.data(), points.data(), corners.data(), 3, cells.data(), offsets.data(),
                  corners.data(), vertices.data(), rows.offsets.data(), rows.words.data(),
                  rows.signs.data(), offsets.data(), face_offsets.data(), face_corners.data(),
                  kPairs, pairs.data(), declared.data(), workers) == PHX_MC_OK);
    PHX_CHECK(declared == expected);
  }
}

}  // namespace

int main() {
  test_cube_decomposition();
  test_inward_cube_is_reoriented();
  test_refusals();
  test_bent_cube_fans();
  test_face_triangulation();
  test_cube_validity();
  test_parallelism();
  test_declared_contact_workers();
  return phx::mc::test::finish("test_polyhedral_certificates");
}
