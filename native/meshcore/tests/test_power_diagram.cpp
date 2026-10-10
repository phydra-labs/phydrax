//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Restricted power cells on a two-region box [0, 2] x [0, 1] x [0, 1]: exact
// region measures, closed oriented cells, reciprocal interior faces, boundary
// conformity, interface splitting, and fail-closed degenerate/budget/domain
// refusals.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "clip3d.hpp"

namespace {

struct Domain {
  std::vector<double> points;
  std::vector<int32_t> tets;
  std::vector<int32_t> regions;
  std::vector<int32_t> facets;
};

double det3(const double* a, const double* b, const double* c, const double* d) {
  const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double w[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
  return u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
         u[2] * (v[0] * w[1] - v[1] * w[0]);
}

// Facet of a face on a box side: x = 0 -> 0, x = 2 -> 1, y = 0 -> 2, y = 1 -> 3,
// z = 0 -> 4, z = 1 -> 5, the interface x = 1 -> 6; -1 otherwise.
int32_t face_facet(const std::vector<double>& points, const std::array<int32_t, 3>& face) {
  const double values[4][2] = {{0.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 1.0}};
  for (int axis = 0; axis < 3; ++axis) {
    for (int side = 0; side < 2; ++side) {
      const double value = values[axis][side];
      bool on = true;
      for (const int32_t v : face) {
        on = on && points[3 * v + axis] == value;
      }
      if (on) {
        return 2 * axis + side;
      }
    }
  }
  bool interface = true;
  for (const int32_t v : face) {
    interface = interface && points[3 * v] == 1.0;
  }
  return interface ? 6 : -1;
}

// Two Kuhn-triangulated unit cubes; region 0 is x < 1 and region 1 is x > 1.
Domain two_cubes() {
  Domain domain;
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 2; ++y) {
      for (int z = 0; z < 2; ++z) {
        domain.points.insert(domain.points.end(), {double(x), double(y), double(z)});
      }
    }
  }
  const auto id = [](int x, int y, int z) { return 4 * x + 2 * y + z; };
  const int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int cube = 0; cube < 2; ++cube) {
    for (const auto& permutation : permutations) {
      int corner[3] = {0, 0, 0};
      std::array<int32_t, 4> tet{};
      tet[0] = id(cube, 0, 0);
      for (int step = 0; step < 3; ++step) {
        corner[permutation[step]] = 1;
        tet[step + 1] = id(cube + corner[0], corner[1], corner[2]);
      }
      if (det3(&domain.points[3 * tet[0]], &domain.points[3 * tet[1]], &domain.points[3 * tet[2]],
               &domain.points[3 * tet[3]]) < 0.0) {
        std::swap(tet[0], tet[1]);
      }
      domain.tets.insert(domain.tets.end(), tet.begin(), tet.end());
      domain.regions.push_back(cube);
      for (int k = 0; k < 4; ++k) {
        std::array<int32_t, 3> face{};
        int count = 0;
        for (int v = 0; v < 4; ++v) {
          if (v != k) {
            face[count++] = tet[v];
          }
        }
        domain.facets.push_back(face_facet(domain.points, face));
      }
    }
  }
  return domain;
}

struct Sites {
  std::vector<double> points;
  std::vector<int64_t> offsets;
  std::vector<int32_t> neighbors;
};

// Generic sites from a fixed linear congruential sequence, with their
// Delaunay adjacency from the library's exact triangulation.
Sites generic_sites(int count) {
  Sites sites;
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (int k = 0; k < count; ++k) {
    for (int axis = 0; axis < 3; ++axis) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      const double unit = static_cast<double>(state >> 11) * 0x1p-53;
      sites.points.push_back(axis == 0 ? 2.0 * unit : unit);
    }
  }
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(count, sites.points.data(), 100000, &mesh) == PHX_MC_OK);
  const int64_t cells = phx_mc_mesh_cell_count(mesh);
  std::vector<int32_t> tets(static_cast<std::size_t>(4 * cells));
  phx_mc_mesh_copy_cells(mesh, tets.data());
  phx_mc_mesh_free(mesh);
  std::vector<std::vector<int32_t>> adjacent(static_cast<std::size_t>(count));
  for (int64_t t = 0; t < cells; ++t) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        if (a != b) {
          adjacent[tets[4 * t + a]].push_back(tets[4 * t + b]);
        }
      }
    }
  }
  sites.offsets.push_back(0);
  for (auto& row : adjacent) {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    sites.neighbors.insert(sites.neighbors.end(), row.begin(), row.end());
    sites.offsets.push_back(static_cast<int64_t>(sites.neighbors.size()));
  }
  return sites;
}

struct Cells {
  int32_t status = -1;
  phx_mc_power_cells* handle = nullptr;
  std::vector<double> vertices;
  std::vector<int64_t> face_offsets;
  std::vector<int32_t> face_vertices, face_cells, face_facets, cell_sites, cell_regions;
  std::vector<double> cell_volumes, cell_moments, cell_second;
  std::vector<int32_t> piece_sites, piece_tets, piece_cells;
  std::vector<double> piece_volumes;
  int64_t failure[PHX_MC_POWER_CELLS_FAILURE] = {};
};

Cells collect(Cells cells);

Cells run(const Sites& sites, const Domain& domain, const std::vector<int8_t>& split,
          int64_t work_limit) {
  Cells cells;
  const int64_t count = static_cast<int64_t>(sites.points.size() / 3);
  const std::vector<double> weights(static_cast<std::size_t>(count), 0.0);
  cells.status = phx_mc_restricted_power_cells(
      count, sites.points.data(), weights.data(), sites.offsets.data(), sites.neighbors.data(),
      split.data(), static_cast<int64_t>(domain.points.size() / 3), domain.points.data(),
      static_cast<int64_t>(domain.regions.size()), domain.tets.data(), domain.regions.data(),
      domain.facets.data(), 100000, 1000000, work_limit, 0, &cells.handle);
  return collect(std::move(cells));
}

Cells collect(Cells cells) {
  if (cells.handle == nullptr) {
    return cells;
  }
  phx_mc_power_cells_failure(cells.handle, cells.failure);
  if (cells.status == PHX_MC_OK) {
    int64_t sizes[PHX_MC_POWER_CELLS_SIZES];
    phx_mc_power_cells_sizes(cells.handle, sizes);
    cells.vertices.resize(3 * sizes[0]);
    cells.face_offsets.resize(sizes[1] + 1);
    cells.face_vertices.resize(sizes[2]);
    cells.face_cells.resize(2 * sizes[1]);
    cells.face_facets.resize(sizes[1]);
    cells.cell_sites.resize(sizes[3]);
    cells.cell_regions.resize(sizes[3]);
    cells.cell_volumes.resize(sizes[3]);
    cells.cell_moments.resize(3 * sizes[3]);
    cells.cell_second.resize(sizes[3]);
    cells.piece_sites.resize(sizes[4]);
    cells.piece_tets.resize(sizes[4]);
    cells.piece_cells.resize(sizes[4]);
    cells.piece_volumes.resize(sizes[4]);
    phx_mc_power_cells_export(
        cells.handle, cells.vertices.data(), cells.face_offsets.data(), cells.face_vertices.data(),
        cells.face_cells.data(), cells.face_facets.data(), cells.cell_sites.data(),
        cells.cell_regions.data(), cells.cell_volumes.data(), cells.cell_moments.data(),
        cells.cell_second.data(), cells.piece_sites.data(), cells.piece_tets.data(),
        cells.piece_cells.data(), cells.piece_volumes.data());
  }
  phx_mc_power_cells_free(cells.handle);
  cells.handle = nullptr;
  return cells;
}

// Signed volume of every cell from its outward faces (divergence theorem).
std::vector<double> face_volumes(const Cells& cells) {
  std::vector<double> volumes(cells.cell_sites.size(), 0.0);
  for (std::size_t f = 0; f + 1 < cells.face_offsets.size(); ++f) {
    const int64_t begin = cells.face_offsets[f];
    const int64_t end = cells.face_offsets[f + 1];
    double term = 0.0;
    const double* a = &cells.vertices[3 * cells.face_vertices[begin]];
    for (int64_t e = begin + 1; e + 1 < end; ++e) {
      const double* b = &cells.vertices[3 * cells.face_vertices[e]];
      const double* c = &cells.vertices[3 * cells.face_vertices[e + 1]];
      const double origin[3] = {0.0, 0.0, 0.0};
      term += det3(origin, a, b, c) / 6.0;
    }
    volumes[cells.face_cells[2 * f]] += term;
    if (cells.face_cells[2 * f + 1] >= 0) {
      volumes[cells.face_cells[2 * f + 1]] -= term;
    }
  }
  return volumes;
}

void test_two_region_box() {
  const Domain domain = two_cubes();
  const Sites sites = generic_sites(40);
  const std::vector<int8_t> split(40, 0);
  const Cells cells = run(sites, domain, split, 1000000);
  PHX_CHECK(cells.status == PHX_MC_OK);
  if (cells.status != PHX_MC_OK) {
    return;
  }
  // Region measures and per-cell closure against the divergence volumes.
  double region[2] = {0.0, 0.0};
  const std::vector<double> volumes = face_volumes(cells);
  for (std::size_t c = 0; c < cells.cell_sites.size(); ++c) {
    region[cells.cell_regions[c]] += cells.cell_volumes[c];
    PHX_CHECK(cells.cell_volumes[c] > 0.0);
    PHX_CHECK_NEAR(volumes[c], cells.cell_volumes[c], 1e-12);
  }
  PHX_CHECK_NEAR(region[0], 1.0, 1e-12);
  PHX_CHECK_NEAR(region[1], 1.0, 1e-12);
  // Every cell is a closed oriented surface: its directed edges cancel.
  std::map<std::pair<int32_t, std::pair<int32_t, int32_t>>, int> edges;
  for (std::size_t f = 0; f + 1 < cells.face_offsets.size(); ++f) {
    const int64_t begin = cells.face_offsets[f];
    const int64_t size = cells.face_offsets[f + 1] - begin;
    PHX_CHECK(size >= 3);
    for (int64_t k = 0; k < size; ++k) {
      const int32_t a = cells.face_vertices[begin + k];
      const int32_t b = cells.face_vertices[begin + (k + 1) % size];
      ++edges[{cells.face_cells[2 * f], {a, b}}];
      if (cells.face_cells[2 * f + 1] >= 0) {
        ++edges[{cells.face_cells[2 * f + 1], {b, a}}];
      }
    }
  }
  for (const auto& [key, count] : edges) {
    const auto reverse = edges.find({key.first, {key.second.second, key.second.first}});
    PHX_CHECK(reverse != edges.end() && reverse->second == count);
  }
  // Boundary faces lie on their facet; interface faces separate one site.
  bool interface_seen = false;
  for (std::size_t f = 0; f + 1 < cells.face_offsets.size(); ++f) {
    const int32_t other = cells.face_cells[2 * f + 1];
    const int32_t facet = cells.face_facets[f];
    if (other < 0) {
      PHX_CHECK(facet == -1 - other && facet >= 0 && facet < 6);
      const int axis = facet / 2;
      const double value = facet % 2 == 0 ? 0.0 : (axis == 0 ? 2.0 : 1.0);
      for (int64_t e = cells.face_offsets[f]; e < cells.face_offsets[f + 1]; ++e) {
        PHX_CHECK(cells.vertices[3 * cells.face_vertices[e] + axis] == value);
      }
    } else if (facet == 6) {
      interface_seen = true;
      const int32_t owner = cells.face_cells[2 * f];
      PHX_CHECK(cells.cell_sites[owner] == cells.cell_sites[other]);
      PHX_CHECK(cells.cell_regions[owner] != cells.cell_regions[other]);
    } else {
      PHX_CHECK(facet == -1);
      PHX_CHECK(cells.cell_sites[cells.face_cells[2 * f]] != cells.cell_sites[other]);
    }
  }
  PHX_CHECK(interface_seen);
  // Splitting every site keeps each piece a cell with the same total measure.
  const Cells pieces = run(sites, domain, std::vector<int8_t>(40, 1), 1000000);
  PHX_CHECK(pieces.status == PHX_MC_OK);
  PHX_CHECK(pieces.cell_sites.size() == pieces.piece_sites.size());
  PHX_CHECK(pieces.cell_sites.size() > cells.cell_sites.size());
  double total = 0.0;
  for (const double value : pieces.cell_volumes) {
    total += value;
  }
  PHX_CHECK_NEAR(total, 2.0, 1e-12);
}

void test_refusals() {
  const Domain domain = two_cubes();
  const Sites sites = generic_sites(40);
  const std::vector<int8_t> split(40, 0);
  const Cells exhausted = run(sites, domain, split, 3);
  PHX_CHECK(exhausted.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(exhausted.failure[0] == 6);

  Domain open = domain;
  std::replace(open.facets.begin(), open.facets.end(), 0, -1);
  const Cells unconstrained = run(sites, open, split, 1000000);
  PHX_CHECK(unconstrained.status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(unconstrained.failure[0] == 7);

  // Eight-way cospherical vertices and bisectors coincident with carrier
  // faces must reconcile, rather than treating a legal diagram as malformed.
  Sites lattice;
  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 2; ++y) {
      for (int z = 0; z < 2; ++z) {
        lattice.points.insert(lattice.points.end(), {0.25 + 0.5 * x, 0.25 + 0.5 * y, 0.25 + 0.5 * z});
      }
    }
  }
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(16, lattice.points.data(), 100000, &mesh) == PHX_MC_OK);
  const int64_t count = phx_mc_mesh_cell_count(mesh);
  std::vector<int32_t> tets(static_cast<std::size_t>(4 * count));
  phx_mc_mesh_copy_cells(mesh, tets.data());
  phx_mc_mesh_free(mesh);
  std::vector<std::vector<int32_t>> adjacent(16);
  for (int64_t t = 0; t < count; ++t) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        if (a != b) {
          adjacent[tets[4 * t + a]].push_back(tets[4 * t + b]);
        }
      }
    }
  }
  lattice.offsets.push_back(0);
  for (auto& row : adjacent) {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    lattice.neighbors.insert(lattice.neighbors.end(), row.begin(), row.end());
    lattice.offsets.push_back(static_cast<int64_t>(lattice.neighbors.size()));
  }
  const Cells coincident = run(lattice, domain, std::vector<int8_t>(16, 0), 1000000);
  PHX_CHECK(coincident.status == PHX_MC_OK);
  double total = 0.0;
  for (double volume : coincident.cell_volumes) {
    total += volume;
  }
  PHX_CHECK_NEAR(total, 2.0, 1e-12);
}

// Shared Kuhn tetrahedra of arbitrary admitted voxels. Every boundary triangle
// has its own planar constraint token, independently of semantic facet groups.
Domain voxels(const std::vector<std::array<int, 3>>& cubes) {
  Domain out;
  std::map<std::array<int, 3>, int32_t> vertices;
  const int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (const auto& cube : cubes) {
    for (const auto& permutation : permutations) {
      std::array<int, 3> corner = cube;
      std::array<int32_t, 4> tet;
      for (int step = 0; step < 4; ++step) {
        auto [found, inserted] = vertices.try_emplace(corner, static_cast<int32_t>(vertices.size()));
        if (inserted) {
          for (int coordinate : corner) {
            out.points.push_back(coordinate);
          }
        }
        tet[step] = found->second;
        if (step < 3) {
          ++corner[permutation[step]];
        }
      }
      if (det3(&out.points[3 * tet[0]], &out.points[3 * tet[1]],
               &out.points[3 * tet[2]], &out.points[3 * tet[3]]) < 0) {
        std::swap(tet[0], tet[1]);
      }
      out.tets.insert(out.tets.end(), tet.begin(), tet.end());
      out.regions.push_back(0);
    }
  }
  std::map<std::array<int32_t, 3>, std::vector<int32_t>> faces;
  out.facets.assign(out.tets.size(), -1);
  for (int32_t tet = 0; tet < static_cast<int32_t>(out.regions.size()); ++tet) {
    for (int k = 0; k < 4; ++k) {
      std::array<int32_t, 3> face;
      int cursor = 0;
      for (int v = 0; v < 4; ++v) {
        if (v != k) {
          face[cursor++] = out.tets[4 * tet + v];
        }
      }
      std::sort(face.begin(), face.end());
      faces[face].push_back(4 * tet + k);
    }
  }
  int32_t token = 0;
  for (const auto& [face, slots] : faces) {
    if (slots.size() == 1) {
      out.facets[slots[0]] = token++;
    }
  }
  return out;
}

void test_nonconvex_hole_and_disconnected_components() {
  const Sites singleton{{0.25, 0.25, 0.25}, {0, 0}, {}};
  const Domain lshape = voxels({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
  const Cells lcell = run(singleton, lshape, {0}, 1000000);
  PHX_CHECK(lcell.status == PHX_MC_OK);
  PHX_CHECK(lcell.cell_sites.size() == 1);
  if (!lcell.cell_volumes.empty()) {
    PHX_CHECK_NEAR(lcell.cell_volumes[0], 3.0, 1e-13);
    PHX_CHECK_NEAR(face_volumes(lcell)[0], 3.0, 1e-13);
  }
  const Domain disconnected = voxels({{0, 0, 0}, {3, 0, 0}});
  const Cells components = run(singleton, disconnected, {0}, 1000000);
  PHX_CHECK(components.status == PHX_MC_OK);
  PHX_CHECK(components.cell_sites.size() == 2);
  for (double volume : components.cell_volumes) {
    PHX_CHECK_NEAR(volume, 1.0, 1e-13);
  }
  std::vector<std::array<int, 3>> ring;
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
      if (x != 1 || y != 1) {
        ring.push_back({x, y, 0});
      }
    }
  }
  const Domain hole = voxels(ring);
  const Cells decomposed = run(singleton, hole, {1}, 1000000);
  PHX_CHECK(decomposed.status == PHX_MC_OK);
  PHX_CHECK(decomposed.cell_sites.size() == hole.regions.size());
  double volume = 0.0;
  for (double measure : decomposed.cell_volumes) {
    volume += measure;
  }
  PHX_CHECK_NEAR(volume, 8.0, 1e-13);
}

void test_exact_image_source_and_refusals() {
  using namespace phx::mc;
  // The exact translated site differs from its binary64 projection. A rounded
  // proxy makes this power predicate zero; the authored expansion makes it -1.
  const double image0[2] = {0x1p-54, 1.0};
  Expansion exact[6] = {Expansion::from_components(image0, 2), Expansion(), Expansion(),
                        Expansion(1.0), Expansion(), Expansion()};
  Approx filtered[6] = {Approx::exact(0x1p-54) + Approx::exact(1.0),
                        Approx::exact(0.0), Approx::exact(0.0),
                        Approx::exact(1.0), Approx::exact(0.0), Approx::exact(0.0)};
  const ExactPowerCoordinates images[2] = {{filtered, exact}, {filtered + 3, exact + 3}};
  const double point[3] = {1.0, 0.0, 0.0};
  PHX_CHECK(TetrahedronClipper::point_power_side(point, nullptr, 0.0, nullptr, 0.0,
                                               images, images + 1) == -1);

  const Domain domain = two_cubes();
  const double original_sites[6] = {0.25, 0.25, 0.25, 0.75, 0.25, 0.25};
  const double original_weights[2] = {0.0, 0.0};
  const int32_t owners[1] = {1};
  const int64_t coordinate_offsets[4] = {0, 1, 2, 3};
  double components[3] = {0.75, 0.25, 0.25};
  const int64_t neighbors[2] = {0, 0};
  const int8_t split[1] = {0};
  const auto construct = [&](int64_t work_limit, int64_t max_pieces) {
    Cells cells;
    cells.status = phx_mc_restricted_power_cells_exact(
        2, original_sites, original_weights, 1, owners, coordinate_offsets,
        3, components, neighbors, nullptr, split,
        static_cast<int64_t>(domain.points.size() / 3), domain.points.data(),
        static_cast<int64_t>(domain.regions.size()), domain.tets.data(),
        domain.regions.data(), domain.facets.data(), max_pieces, 1000000,
        work_limit, 0, &cells.handle);
    return collect(std::move(cells));
  };
  const Cells cells = construct(1000000, 100000);
  PHX_CHECK(cells.status == PHX_MC_OK);
  PHX_CHECK(cells.cell_sites.size() == 2);
  const auto independent = face_volumes(cells);
  for (std::size_t cell = 0; cell < cells.cell_sites.size(); ++cell) {
    PHX_CHECK(cells.cell_sites[cell] == 0);  // image axis, not original owner 1
    PHX_CHECK_NEAR(cells.cell_volumes[cell], 1.0, 1e-13);
    PHX_CHECK_NEAR(independent[cell], 1.0, 1e-13);
  }
  PHX_CHECK(construct(1000000, 1).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(construct(1, 100000).status == PHX_MC_CAPACITY_EXCEEDED);
  components[0] = 0x1p-173;
  const Cells unsupported = construct(1000000, 100000);
  PHX_CHECK(unsupported.status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(unsupported.failure[0] == 10);
  PHX_CHECK(unsupported.failure[1] == 0);
  PHX_CHECK(unsupported.failure[2] == 0);
  PHX_CHECK(original_sites[3] == 0.75);
}

}  // namespace

int main() {
  test_two_region_box();
  test_refusals();
  test_nonconvex_hole_and_disconnected_components();
  test_exact_image_source_and_refusals();
  return phx::mc::test::finish("test_power_diagram");
}
