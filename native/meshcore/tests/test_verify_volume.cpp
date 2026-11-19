//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// phx_mc_verify_volume on the unit cube split into six tetrahedra (Kuhn
// triangulation), and on deliberately broken variants of it.
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Cube {
  std::vector<double> points;
  std::vector<int32_t> cells;
  std::vector<int32_t> surface;
};

double determinant(const std::vector<double>& p, const int32_t* v) {
  const double* a = p.data() + 3 * v[0];
  double m[3][3];
  for (int r = 0; r < 3; ++r) {
    const double* b = p.data() + 3 * v[r + 1];
    for (int c = 0; c < 3; ++c) m[r][c] = b[c] - a[c];
  }
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Kuhn triangulation: vertex x + 2y + 4z; one cell per axis permutation along
// the path 0 -> e_a -> e_a + e_b -> 7, oriented positively by an exact integer
// determinant. The surface is the set of faces of exactly one cell, oriented
// so that its normal points away from the cube's center.
Cube kuhn_cube() {
  Cube cube;
  for (int index = 0; index < 8; ++index) {
    cube.points.push_back(index & 1);
    cube.points.push_back((index >> 1) & 1);
    cube.points.push_back((index >> 2) & 1);
  }
  const int axes[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (const auto& order : axes) {
    std::array<int32_t, 4> cell{0, 1 << order[0], (1 << order[0]) | (1 << order[1]), 7};
    if (determinant(cube.points, cell.data()) < 0.0) std::swap(cell[2], cell[3]);
    cube.cells.insert(cube.cells.end(), cell.begin(), cell.end());
  }
  std::map<std::array<int32_t, 3>, int> counts;
  std::map<std::array<int32_t, 3>, std::array<int32_t, 3>> any_orientation;
  for (int cell = 0; cell < 6; ++cell) {
    const int32_t* v = cube.cells.data() + 4 * cell;
    for (int skip = 0; skip < 4; ++skip) {
      std::array<int32_t, 3> face{};
      int n = 0;
      for (int k = 0; k < 4; ++k) {
        if (k != skip) face[n++] = v[k];
      }
      std::array<int32_t, 3> key = face;
      std::sort(key.begin(), key.end());
      ++counts[key];
      any_orientation[key] = face;
    }
  }
  for (const auto& [key, count] : counts) {
    if (count != 1) continue;
    std::array<int32_t, 3> face = any_orientation[key];
    const double* a = cube.points.data() + 3 * face[0];
    const double* b = cube.points.data() + 3 * face[1];
    const double* c = cube.points.data() + 3 * face[2];
    const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const double normal[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                              u[0] * w[1] - u[1] * w[0]};
    const double toward_center = normal[0] * (0.5 - a[0]) + normal[1] * (0.5 - a[1]) +
                                 normal[2] * (0.5 - a[2]);
    if (toward_center > 0.0) std::swap(face[1], face[2]);
    cube.surface.insert(cube.surface.end(), face.begin(), face.end());
  }
  return cube;
}

phx_mc_volume_verdict verify(const Cube& cube, int32_t* status) {
  phx_mc_volume_verdict verdict{};
  *status = phx_mc_verify_volume(
      static_cast<int64_t>(cube.points.size() / 3), cube.points.data(),
      static_cast<int64_t>(cube.cells.size() / 4), cube.cells.data(),
      static_cast<int64_t>(cube.surface.size() / 3), cube.surface.data(), &verdict);
  return verdict;
}

}  // namespace

int main() {
  const Cube cube = kuhn_cube();
  PHX_CHECK(cube.surface.size() == 36);
  int32_t status = 0;

  const phx_mc_volume_verdict valid = verify(cube, &status);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(valid.certified == 1);
  PHX_CHECK(valid.boundary_faces == 12);
  PHX_CHECK(valid.nonpositive_cells == 0 && valid.nonmanifold_faces == 0);

  Cube flipped = cube;
  std::swap(flipped.cells[4 * 2 + 0], flipped.cells[4 * 2 + 1]);
  const phx_mc_volume_verdict inverted = verify(flipped, &status);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(inverted.certified == 0);
  PHX_CHECK(inverted.nonpositive_cells == 1);
  PHX_CHECK(inverted.first_nonpositive_cell == 2);

  Cube open_surface = cube;
  open_surface.surface.resize(open_surface.surface.size() - 3);
  const phx_mc_volume_verdict open = verify(open_surface, &status);
  PHX_CHECK(open.certified == 0);
  PHX_CHECK(open.unmatched_boundary_faces == 1);

  Cube duplicated = cube;
  duplicated.cells.insert(duplicated.cells.end(), cube.cells.begin(), cube.cells.begin() + 4);
  const phx_mc_volume_verdict overlap = verify(duplicated, &status);
  PHX_CHECK(overlap.certified == 0);
  PHX_CHECK(overlap.nonmanifold_faces > 0);

  Cube inward = cube;
  for (std::size_t k = 0; k < inward.surface.size(); k += 3) {
    std::swap(inward.surface[k + 1], inward.surface[k + 2]);
  }
  const phx_mc_volume_verdict reversed = verify(inward, &status);
  PHX_CHECK(reversed.certified == 0);
  PHX_CHECK(reversed.unmatched_boundary_faces == 12);
  PHX_CHECK(reversed.unmatched_surface_triangles == 12);

  Cube missing_cell = cube;
  missing_cell.cells.resize(missing_cell.cells.size() - 4);
  const phx_mc_volume_verdict gap = verify(missing_cell, &status);
  PHX_CHECK(gap.certified == 0);
  PHX_CHECK(gap.unmatched_boundary_faces > 0);
  PHX_CHECK(gap.unmatched_surface_triangles > 0);

  Cube out_of_range = cube;
  out_of_range.cells[0] = 8;
  verify(out_of_range, &status);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);

  PHX_CHECK(phx_mc_verify_volume(8, cube.points.data(), 6, cube.cells.data(), 12,
                                 cube.surface.data(), nullptr) == PHX_MC_INVALID_ARGUMENT);
  return phx::mc::test::finish("test_verify_volume");
}
