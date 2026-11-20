//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// phx_mc_verify_tetrahedra, phx_mc_verify_volume and phx_mc_tetrahedra_validity
// on unions of unit lattice cubes split into six tetrahedra each (Kuhn
// triangulation, conforming across the lattice), and on deliberately broken
// variants of them.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Mesh {
  std::vector<double> points;
  std::vector<int32_t> cells;
  std::vector<int32_t> regions;
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

// Lattice points (x, y, z) in [0, n)^3 scaled by `scale` and shifted by
// `shift`, appended to `mesh`; returns the index of the first one.
int32_t add_lattice(Mesh& mesh, int n, double scale, double shift) {
  const auto first = static_cast<int32_t>(mesh.points.size() / 3);
  for (int z = 0; z < n; ++z) {
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        mesh.points.push_back(scale * x + shift);
        mesh.points.push_back(scale * y + shift);
        mesh.points.push_back(scale * z + shift);
      }
    }
  }
  return first;
}

// The six positive Kuhn tetrahedra of the lattice cube with lower corner
// (x, y, z): one per axis permutation along 0 -> e_a -> e_a + e_b -> 7.
void add_cube(Mesh& mesh, int32_t base, int n, int x, int y, int z, int32_t region) {
  const auto id = [&](int corner) {
    return base + (x + (corner & 1)) + n * (y + ((corner >> 1) & 1)) +
           n * n * (z + ((corner >> 2) & 1));
  };
  const int axes[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (const auto& order : axes) {
    std::array<int32_t, 4> cell{id(0), id(1 << order[0]), id((1 << order[0]) | (1 << order[1])),
                                id(7)};
    if (determinant(mesh.points, cell.data()) < 0.0) std::swap(cell[2], cell[3]);
    mesh.cells.insert(mesh.cells.end(), cell.begin(), cell.end());
    mesh.regions.push_back(region);
  }
}

Mesh cube_union(const std::vector<std::array<int, 3>>& cubes, int n) {
  Mesh mesh;
  const int32_t base = add_lattice(mesh, n, 1.0, 0.0);
  for (const auto& cube : cubes) add_cube(mesh, base, n, cube[0], cube[1], cube[2], 0);
  return mesh;
}

// Outward boundary faces (faces of one cell) with the regions on their two
// sides, and interface faces (faces of two cells of different regions)
// oriented out of the first region's cell.
void declared_surface(const Mesh& mesh, std::vector<int32_t>& surface,
                      std::vector<int32_t>& sides) {
  static constexpr int kOutward[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  std::map<std::array<int32_t, 3>, std::vector<std::pair<std::array<int32_t, 3>, int32_t>>> faces;
  for (std::size_t cell = 0; cell < mesh.cells.size() / 4; ++cell) {
    const int32_t* v = mesh.cells.data() + 4 * cell;
    for (const auto& local : kOutward) {
      std::array<int32_t, 3> face{v[local[0]], v[local[1]], v[local[2]]};
      std::array<int32_t, 3> key = face;
      std::sort(key.begin(), key.end());
      faces[key].push_back({face, mesh.regions[cell]});
    }
  }
  for (const auto& [key, uses] : faces) {
    (void)key;
    if (uses.size() == 1) {
      surface.insert(surface.end(), uses[0].first.begin(), uses[0].first.end());
      sides.push_back(uses[0].second);
      sides.push_back(-1);
    } else if (uses[0].second != uses[1].second) {
      surface.insert(surface.end(), uses[0].first.begin(), uses[0].first.end());
      sides.push_back(uses[0].second);
      sides.push_back(uses[1].second);
    }
  }
}

phx_mc_tetrahedra_verdict embedding(const Mesh& mesh, int32_t* status) {
  phx_mc_tetrahedra_verdict verdict{};
  *status = phx_mc_verify_tetrahedra(static_cast<int64_t>(mesh.points.size() / 3),
                                     mesh.points.data(),
                                     static_cast<int64_t>(mesh.cells.size() / 4),
                                     mesh.cells.data(), &verdict);
  return verdict;
}

phx_mc_volume_verdict coverage(const Mesh& mesh, const std::vector<int32_t>& surface,
                               const std::vector<int32_t>& sides, int32_t* status) {
  phx_mc_volume_verdict verdict{};
  *status = phx_mc_verify_volume(
      static_cast<int64_t>(mesh.points.size() / 3), mesh.points.data(),
      static_cast<int64_t>(mesh.cells.size() / 4), mesh.cells.data(), mesh.regions.data(),
      static_cast<int64_t>(surface.size() / 3), surface.data(), sides.data(), &verdict);
  return verdict;
}

void check_embedding() {
  int32_t status = 0;
  const Mesh cube = cube_union({{0, 0, 0}}, 2);
  const phx_mc_tetrahedra_verdict valid = embedding(cube, &status);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(valid.status == PHX_MC_VERDICT_CERTIFIED);
  PHX_CHECK(valid.boundary_faces == 12 && valid.boundary_shells == 1);

  Mesh flipped = cube;
  std::swap(flipped.cells[4 * 2 + 0], flipped.cells[4 * 2 + 1]);
  const phx_mc_tetrahedra_verdict inverted = embedding(flipped, &status);
  PHX_CHECK(inverted.status == PHX_MC_VERDICT_VIOLATED);
  PHX_CHECK(inverted.nonpositive_cells == 1 && inverted.first_nonpositive_cell == 2);

  Mesh duplicated = cube;
  duplicated.cells.insert(duplicated.cells.end(), cube.cells.begin(), cube.cells.begin() + 4);
  const phx_mc_tetrahedra_verdict overlap = embedding(duplicated, &status);
  PHX_CHECK(overlap.status == PHX_MC_VERDICT_VIOLATED && overlap.nonmanifold_faces > 0);

  // A 3 x 3 x 3 block of cubes without its center: an outward shell around an
  // inward cavity shell, both with private vertices.
  std::vector<std::array<int, 3>> hollow_cubes;
  for (int z = 0; z < 3; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        if (x != 1 || y != 1 || z != 1) hollow_cubes.push_back({x, y, z});
      }
    }
  }
  const phx_mc_tetrahedra_verdict hollow = embedding(cube_union(hollow_cubes, 4), &status);
  PHX_CHECK(hollow.status == PHX_MC_VERDICT_CERTIFIED);
  PHX_CHECK(hollow.boundary_shells == 2 && hollow.misnested_shells == 0);

  // Two separate cubes: two outward shells, neither inside the other.
  const phx_mc_tetrahedra_verdict apart = embedding(cube_union({{0, 0, 0}, {2, 2, 2}}, 4), &status);
  PHX_CHECK(apart.status == PHX_MC_VERDICT_CERTIFIED && apart.boundary_shells == 2);

  // A small cube meshed inside a larger meshed cube: two outward shells, one
  // inside the other, so the inner solid is covered twice.
  Mesh nested;
  add_cube(nested, add_lattice(nested, 2, 4.0, -1.5), 2, 0, 0, 0, 0);
  add_cube(nested, add_lattice(nested, 2, 1.0, 0.0), 2, 0, 0, 0, 0);
  const phx_mc_tetrahedra_verdict twice = embedding(nested, &status);
  PHX_CHECK(twice.status == PHX_MC_VERDICT_VIOLATED);
  PHX_CHECK(twice.boundary_shells == 2 && twice.misnested_shells == 1);
  PHX_CHECK(twice.boundary_contacts == 0);

  // Two cubes overlapping in a corner: their boundaries cross.
  Mesh crossing;
  add_cube(crossing, add_lattice(crossing, 2, 1.0, 0.0), 2, 0, 0, 0, 0);
  add_cube(crossing, add_lattice(crossing, 2, 1.0, 0.5), 2, 0, 0, 0, 0);
  const phx_mc_tetrahedra_verdict crossed = embedding(crossing, &status);
  PHX_CHECK(crossed.status == PHX_MC_VERDICT_VIOLATED && crossed.boundary_contacts > 0);

  // Two cubes sharing one edge: four boundary triangles at that edge, outside
  // the theorem.
  const phx_mc_tetrahedra_verdict edge = embedding(cube_union({{0, 0, 0}, {1, 1, 0}}, 3), &status);
  PHX_CHECK(edge.status == PHX_MC_VERDICT_UNRESOLVED);
  PHX_CHECK(edge.nonmanifold_boundary_edges > 0 && edge.misnested_shells == 0);

  Mesh far = cube;
  far.points[0] = 0x1p200;
  const phx_mc_tetrahedra_verdict outside = embedding(far, &status);
  PHX_CHECK(outside.status == PHX_MC_VERDICT_UNRESOLVED && outside.outside_domain_points == 1);

  Mesh out_of_range = cube;
  out_of_range.cells[0] = 8;
  embedding(out_of_range, &status);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(phx_mc_verify_tetrahedra(8, cube.points.data(), 6, cube.cells.data(), nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
}

void check_coverage() {
  int32_t status = 0;
  Mesh pair;
  const int32_t base = add_lattice(pair, 3, 1.0, 0.0);
  add_cube(pair, base, 3, 0, 0, 0, 0);
  add_cube(pair, base, 3, 1, 0, 0, 1);
  std::vector<int32_t> surface;
  std::vector<int32_t> sides;
  declared_surface(pair, surface, sides);
  PHX_CHECK(surface.size() / 3 == 20 + 2);
  const phx_mc_volume_verdict valid = coverage(pair, surface, sides, &status);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(valid.certified == 1 && valid.boundary_faces == 20);

  Mesh relabeled = pair;
  relabeled.regions[3] = 1;
  const phx_mc_volume_verdict mislabeled = coverage(relabeled, surface, sides, &status);
  PHX_CHECK(mislabeled.certified == 0 && mislabeled.region_mismatches > 0);

  std::vector<int32_t> swapped = sides;
  std::swap(swapped[0], swapped[1]);
  const phx_mc_volume_verdict reversed = coverage(pair, surface, swapped, &status);
  PHX_CHECK(reversed.certified == 0 && reversed.unmatched_boundary_faces == 1);

  std::vector<int32_t> open(surface.begin(), surface.end() - 3);
  std::vector<int32_t> open_sides(sides.begin(), sides.end() - 2);
  const phx_mc_volume_verdict missing = coverage(pair, open, open_sides, &status);
  PHX_CHECK(missing.certified == 0);
  PHX_CHECK(missing.unmatched_boundary_faces + missing.region_mismatches == 1);

  Mesh flipped = pair;
  std::swap(flipped.cells[4 * 2 + 0], flipped.cells[4 * 2 + 1]);
  const phx_mc_volume_verdict inverted = coverage(flipped, surface, sides, &status);
  PHX_CHECK(inverted.certified == 0 && inverted.nonpositive_cells == 1);

  Mesh fewer = pair;
  fewer.cells.resize(fewer.cells.size() - 4);
  fewer.regions.pop_back();
  const phx_mc_volume_verdict gap = coverage(fewer, surface, sides, &status);
  PHX_CHECK(gap.certified == 0 && gap.unmatched_surface_triangles > 0);

  std::vector<int32_t> bad_sides = sides;
  bad_sides[1] = bad_sides[0];
  coverage(pair, surface, bad_sides, &status);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
}

// One unit tetrahedron and seventeen of side 2^-100 (72 boundary faces, the
// grid broad phase with cells of the tiny median extent): the unit cell's
// boxes span 2^100 cells, a quotient no int64 holds, so the grid must give
// way to the sweep. The first tiny cell sits on the unit cell's face z = 0,
// across it when `contact`, just below it otherwise; the rest lie apart at
// negative coordinates. Every coordinate is a binary64 in the predicate
// domain.
Mesh unit_cell_with_tiny_cells(bool contact) {
  constexpr double kSide = 0x1p-100;
  Mesh mesh;
  const auto add_cell = [&mesh](const std::array<double, 3>& base, double side) {
    const auto first = static_cast<int32_t>(mesh.points.size() / 3);
    mesh.points.insert(mesh.points.end(), {base[0], base[1], base[2], base[0] + side, base[1],
                                           base[2], base[0], base[1] + side, base[2], base[0],
                                           base[1], base[2] + side});
    std::array<int32_t, 4> cell{first, first + 1, first + 2, first + 3};
    if (determinant(mesh.points, cell.data()) < 0.0) std::swap(cell[2], cell[3]);
    mesh.cells.insert(mesh.cells.end(), cell.begin(), cell.end());
    mesh.regions.push_back(0);
  };
  add_cell({0.0, 0.0, 0.0}, 1.0);
  add_cell({0x1p-95, 0x1p-95, contact ? -0x1p-101 : -2.0 * kSide}, kSide);
  for (int k = 1; k <= 16; ++k) add_cell({-(k + 1) * 0x1p-95, -0x1p-95, -0x1p-95}, kSide);
  return mesh;
}

void check_grid_quotient_overflow() {
  int32_t status = 0;
  const phx_mc_tetrahedra_verdict apart = embedding(unit_cell_with_tiny_cells(false), &status);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(apart.status == PHX_MC_VERDICT_CERTIFIED);
  PHX_CHECK(apart.boundary_faces == 72 && apart.boundary_shells == 18);
  PHX_CHECK(apart.boundary_contacts == 0);
  const phx_mc_tetrahedra_verdict across = embedding(unit_cell_with_tiny_cells(true), &status);
  PHX_CHECK(status == PHX_MC_OK);
  PHX_CHECK(across.status == PHX_MC_VERDICT_VIOLATED);
  PHX_CHECK(across.boundary_contacts > 0);
  PHX_CHECK(across.first_contact_cell == 0 && across.second_contact_cell == 1);
}

void check_validity() {
  Mesh mesh = cube_union({{0, 0, 0}}, 2);
  // A sliver: three cube corners and a point just off their plane.
  mesh.points.insert(mesh.points.end(), {1.0, 1.0, 0x1p-30});
  mesh.cells.insert(mesh.cells.end(), {0, 1, 2, 8});
  // A flat cell.
  mesh.cells.insert(mesh.cells.end(), {0, 1, 2, 3});
  const int64_t count = static_cast<int64_t>(mesh.cells.size() / 4);
  std::vector<int8_t> valid(static_cast<std::size_t>(count));
  std::vector<double> lower(static_cast<std::size_t>(count));
  std::vector<double> upper(static_cast<std::size_t>(count));
  PHX_CHECK(phx_mc_tetrahedra_validity(9, mesh.points.data(), count, mesh.cells.data(), 0x1p-35,
                                       valid.data(), lower.data(), upper.data()) == PHX_MC_OK);
  for (int cell = 0; cell < 6; ++cell) {
    PHX_CHECK(valid[cell] == 1);
    PHX_CHECK(lower[cell] <= 1.0 && 1.0 <= upper[cell]);
  }
  // det = 2^-30 > 0 but the margin is negative under floor 2^-20.
  PHX_CHECK(phx_mc_tetrahedra_validity(9, mesh.points.data(), count, mesh.cells.data(), 0x1p-20,
                                       valid.data(), lower.data(), upper.data()) == PHX_MC_OK);
  PHX_CHECK(valid[6] == 0 && lower[6] <= 0x1p-30 && 0x1p-30 <= upper[6]);
  PHX_CHECK(valid[7] == 0 && lower[7] <= 0.0 && 0.0 <= upper[7]);
  PHX_CHECK(phx_mc_tetrahedra_validity(9, mesh.points.data(), count, mesh.cells.data(), 0x1p-35,
                                       valid.data(), lower.data(), upper.data()) == PHX_MC_OK);
  PHX_CHECK(valid[6] == 1);
  PHX_CHECK(phx_mc_tetrahedra_validity(9, mesh.points.data(), count, mesh.cells.data(), 1.0,
                                       valid.data(), lower.data(), upper.data()) ==
            PHX_MC_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  check_embedding();
  check_coverage();
  check_validity();
  check_grid_quotient_overflow();
  return phx::mc::test::finish("test_verify_volume");
}
