//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"

namespace {

using namespace phx::mc;

constexpr int64_t kUnlimited = std::numeric_limits<int64_t>::max();

double uniform(std::uint64_t& state) {
  state = splitmix64(state);
  return static_cast<double>(state >> 11) * 0x1p-53;
}

std::vector<double> random_points(int64_t count, std::uint64_t seed) {
  std::vector<double> points(static_cast<std::size_t>(3 * count));
  for (double& value : points) {
    value = uniform(seed);
  }
  return points;
}

std::vector<double> lattice(int size, double offset) {
  std::vector<double> points;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        points.insert(points.end(), {offset + i, offset + j, offset + k});
      }
    }
  }
  return points;
}

struct Result {
  int32_t status = -1;
  int64_t input_point_count = 0;
  std::vector<int32_t> cells;
  std::vector<int32_t> vertex_map;
};

// Runs the triangulation and checks the mesh handle contract (dimension, points
// copied unchanged, NULL handle on failure).
Result run(const std::vector<double>& points, const std::vector<double>* weights,
           int64_t max_tetrahedra = kUnlimited) {
  int sentinel = 0;
  phx_mc_mesh* mesh = reinterpret_cast<phx_mc_mesh*>(&sentinel);
  const int64_t count = static_cast<int64_t>(points.size() / 3);
  Result result;
  result.status = weights == nullptr
                      ? phx_mc_delaunay_3d(count, points.data(), max_tetrahedra, &mesh)
                      : phx_mc_regular_3d(count, points.data(), weights->data(),
                                          max_tetrahedra, &mesh);
  if (result.status != PHX_MC_OK) {
    PHX_CHECK(mesh == nullptr);
    return result;
  }
  PHX_CHECK(mesh != nullptr);
  PHX_CHECK(phx_mc_mesh_dimension(mesh) == 3);
  PHX_CHECK(phx_mc_mesh_point_count(mesh) == count);
  result.input_point_count = phx_mc_mesh_input_point_count(mesh);
  PHX_CHECK(result.input_point_count == count);
  std::vector<double> copied(points.size());
  phx_mc_mesh_copy_points(mesh, copied.data());
  PHX_CHECK(copied == points);
  result.cells.resize(static_cast<std::size_t>(4 * phx_mc_mesh_cell_count(mesh)));
  phx_mc_mesh_copy_cells(mesh, result.cells.data());
  result.vertex_map.resize(static_cast<std::size_t>(count));
  phx_mc_mesh_copy_vertex_map(mesh, result.vertex_map.data());
  phx_mc_mesh_free(mesh);
  return result;
}

const double* at(const std::vector<double>& points, int32_t index) {
  return points.data() + 3 * static_cast<int64_t>(index);
}

std::size_t cell_count(const Result& result) { return result.cells.size() / 4; }

const int32_t* cell(const Result& result, std::size_t c) { return result.cells.data() + 4 * c; }

// 6 * signed volume of a cell (exact for small integer coordinates).
double six_volume(const std::vector<double>& points, const int32_t* v) {
  const double* a = at(points, v[0]);
  double u[3][3];
  for (int r = 0; r < 3; ++r) {
    for (int axis = 0; axis < 3; ++axis) {
      u[r][axis] = at(points, v[r + 1])[axis] - a[axis];
    }
  }
  return u[0][0] * (u[1][1] * u[2][2] - u[1][2] * u[2][1]) -
         u[0][1] * (u[1][0] * u[2][2] - u[1][2] * u[2][0]) +
         u[0][2] * (u[1][0] * u[2][1] - u[1][1] * u[2][0]);
}

double six_volume_sum(const std::vector<double>& points, const Result& result) {
  double sum = 0.0;
  for (std::size_t c = 0; c < cell_count(result); ++c) {
    sum += six_volume(points, cell(result, c));
  }
  return sum;
}

struct Facet {
  std::array<int32_t, 3> key;
  int32_t cell;
  int32_t slot;
};

std::vector<Facet> sorted_facets(const Result& result) {
  std::vector<Facet> facets;
  facets.reserve(4 * cell_count(result));
  for (std::size_t c = 0; c < cell_count(result); ++c) {
    const int32_t* v = cell(result, c);
    for (int s = 0; s < 4; ++s) {
      std::array<int32_t, 3> key{};
      int count = 0;
      for (int r = 0; r < 4; ++r) {
        if (r != s) {
          key[static_cast<std::size_t>(count++)] = v[r];
        }
      }
      std::sort(key.begin(), key.end());
      facets.push_back({key, static_cast<int32_t>(c), s});
    }
  }
  std::sort(facets.begin(), facets.end(), [](const Facet& left, const Facet& right) {
    return left.key < right.key;
  });
  return facets;
}

bool in_conflict(const std::vector<double>& points, const std::vector<double>* weights,
                 const int32_t* v, int32_t q) {
  const double* p[5] = {at(points, v[0]), at(points, v[1]), at(points, v[2]), at(points, v[3]),
                        at(points, q)};
  if (weights == nullptr) {
    return insphere_sos(p[0], p[1], p[2], p[3], p[4], v[0], v[1], v[2], v[3], q) > 0;
  }
  const std::vector<double>& w = *weights;
  return power3d_sos(p[0], p[1], p[2], p[3], p[4], w[static_cast<std::size_t>(v[0])],
                     w[static_cast<std::size_t>(v[1])], w[static_cast<std::size_t>(v[2])],
                     w[static_cast<std::size_t>(v[3])], w[static_cast<std::size_t>(q)], v[0],
                     v[1], v[2], v[3], q) > 0;
}

// Structural validity of a triangulation of the convex hull of `points`:
// positively oriented cells, every facet shared by at most two cells, the
// symbolically perturbed Delaunay/regular property across every interior facet,
// boundary facets of a convex region containing every input point, Euler
// characteristic of a ball.  Returns the number of boundary facets.
std::size_t check_triangulation(const std::vector<double>& points,
                                const std::vector<double>* weights, const Result& result,
                                bool check_hull = true) {
  for (std::size_t c = 0; c < cell_count(result); ++c) {
    const int32_t* v = cell(result, c);
    PHX_CHECK(orient3d(at(points, v[0]), at(points, v[1]), at(points, v[2]), at(points, v[3])) >
              0);
  }
  const std::vector<Facet> facets = sorted_facets(result);
  std::size_t boundary = 0;
  std::size_t distinct_facets = 0;
  int failures = 0;
  std::vector<const Facet*> hull;
  for (std::size_t i = 0; i < facets.size();) {
    std::size_t j = i + 1;
    while (j < facets.size() && facets[j].key == facets[i].key) {
      ++j;
    }
    ++distinct_facets;
    if (j - i == 1) {
      ++boundary;
      hull.push_back(&facets[i]);
    } else if (j - i == 2) {
      const int32_t* first = cell(result, static_cast<std::size_t>(facets[i].cell));
      const int32_t* second = cell(result, static_cast<std::size_t>(facets[i + 1].cell));
      if (in_conflict(points, weights, first, second[facets[i + 1].slot]) ||
          in_conflict(points, weights, second, first[facets[i].slot])) {
        ++failures;
      }
    } else {
      ++failures;
    }
    i = j;
  }
  PHX_CHECK(failures == 0);
  if (check_hull) {
    int outside = 0;
    const int64_t count = static_cast<int64_t>(points.size() / 3);
    for (const Facet* facet : hull) {
      const int32_t* v = cell(result, static_cast<std::size_t>(facet->cell));
      for (int32_t q = 0; q < count; ++q) {
        const double* p[4] = {at(points, v[0]), at(points, v[1]), at(points, v[2]),
                              at(points, v[3])};
        p[facet->slot] = at(points, q);
        if (orient3d(p[0], p[1], p[2], p[3]) < 0) {
          ++outside;
        }
      }
    }
    PHX_CHECK(outside == 0);
  }
  std::vector<int32_t> vertices(result.cells);
  std::sort(vertices.begin(), vertices.end());
  vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
  std::vector<std::uint64_t> edges;
  for (std::size_t c = 0; c < cell_count(result); ++c) {
    const int32_t* v = cell(result, c);
    for (int r = 0; r < 4; ++r) {
      for (int s = r + 1; s < 4; ++s) {
        const auto low = static_cast<std::uint64_t>(std::min(v[r], v[s]));
        const auto high = static_cast<std::uint64_t>(std::max(v[r], v[s]));
        edges.push_back((low << 32) | high);
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  const int64_t euler = static_cast<int64_t>(vertices.size()) -
                        static_cast<int64_t>(edges.size()) +
                        static_cast<int64_t>(distinct_facets) -
                        static_cast<int64_t>(cell_count(result));
  PHX_CHECK(euler == 1);
  return boundary;
}

// Exact global emptiness: no vertex strictly inside any circumsphere
// (orthosphere), and the perturbed predicate places every other vertex
// strictly outside.  For regular triangulations every input point, including
// redundant ones, must be on or outside each orthosphere.
void check_empty_spheres(const std::vector<double>& points, const std::vector<double>* weights,
                         const Result& result) {
  const int32_t count = static_cast<int32_t>(points.size() / 3);
  int strictly_inside = 0;
  int perturbed_inside = 0;
  for (std::size_t c = 0; c < cell_count(result); ++c) {
    const int32_t* v = cell(result, c);
    const double* p[4] = {at(points, v[0]), at(points, v[1]), at(points, v[2]), at(points, v[3])};
    for (int32_t q = 0; q < count; ++q) {
      const int sign =
          weights == nullptr
              ? insphere(p[0], p[1], p[2], p[3], at(points, q))
              : power3d(p[0], p[1], p[2], p[3], at(points, q),
                        (*weights)[static_cast<std::size_t>(v[0])],
                        (*weights)[static_cast<std::size_t>(v[1])],
                        (*weights)[static_cast<std::size_t>(v[2])],
                        (*weights)[static_cast<std::size_t>(v[3])],
                        (*weights)[static_cast<std::size_t>(q)]);
      if (sign > 0) {
        ++strictly_inside;
      }
      const bool is_vertex = result.vertex_map[static_cast<std::size_t>(q)] == q;
      if (is_vertex && q != v[0] && q != v[1] && q != v[2] && q != v[3] &&
          in_conflict(points, weights, v, q)) {
        ++perturbed_inside;
      }
    }
  }
  PHX_CHECK(strictly_inside == 0);
  PHX_CHECK(perturbed_inside == 0);
}

// vertex_map[i] == i exactly for the points used by cells, and every other
// point maps to a used point or -1.
void check_vertex_map(const Result& result) {
  std::vector<char> used(result.vertex_map.size(), 0);
  for (int32_t vertex : result.cells) {
    used[static_cast<std::size_t>(vertex)] = 1;
  }
  for (std::size_t i = 0; i < used.size(); ++i) {
    const int32_t target = result.vertex_map[i];
    if (used[i] != 0) {
      PHX_CHECK(target == static_cast<int32_t>(i));
    } else {
      PHX_CHECK(target == -1 || (target >= 0 && used[static_cast<std::size_t>(target)] != 0 &&
                                 target != static_cast<int32_t>(i)));
    }
  }
}

std::vector<std::array<int32_t, 4>> sorted_cell_sets(const Result& result,
                                                     const std::vector<int32_t>& relabel) {
  std::vector<std::array<int32_t, 4>> sets;
  for (std::size_t c = 0; c < cell_count(result); ++c) {
    std::array<int32_t, 4> set{};
    for (int r = 0; r < 4; ++r) {
      set[static_cast<std::size_t>(r)] = relabel[static_cast<std::size_t>(cell(result, c)[r])];
    }
    std::sort(set.begin(), set.end());
    sets.push_back(set);
  }
  std::sort(sets.begin(), sets.end());
  return sets;
}

void test_random_points() {
  const std::vector<double> points = random_points(400, 11);
  const Result result = run(points, nullptr);
  PHX_CHECK(result.status == PHX_MC_OK);
  check_triangulation(points, nullptr, result);
  check_empty_spheres(points, nullptr, result);
  check_vertex_map(result);
  for (std::size_t i = 0; i < result.vertex_map.size(); ++i) {
    PHX_CHECK(result.vertex_map[i] == static_cast<int32_t>(i));
  }
  // Canonical form: each cell starts at its smallest vertex; cells sorted.
  for (std::size_t c = 0; c < cell_count(result); ++c) {
    const int32_t* v = cell(result, c);
    PHX_CHECK(v[0] < v[1] && v[0] < v[2] && v[0] < v[3]);
    if (c > 0) {
      PHX_CHECK(std::lexicographical_compare(v - 4, v, v, v + 4));
    }
  }
}

// Integer lattices are cospherical everywhere; ties are resolved symbolically.
void test_cubic_lattice() {
  for (const double offset : {0.0, 1024.0}) {
    const std::vector<double> points = lattice(6, offset);
    const Result result = run(points, nullptr);
    PHX_CHECK(result.status == PHX_MC_OK);
    check_triangulation(points, nullptr, result);
    check_empty_spheres(points, nullptr, result);
    check_vertex_map(result);
    PHX_CHECK(std::count(result.vertex_map.begin(), result.vertex_map.end(), -1) == 0);
    PHX_CHECK(six_volume_sum(points, result) == 750.0);
  }
  // Reversed input order changes the symbolic ranks but not validity.
  std::vector<double> reversed;
  const std::vector<double> points = lattice(5, 0.0);
  for (int64_t i = static_cast<int64_t>(points.size() / 3) - 1; i >= 0; --i) {
    reversed.insert(reversed.end(), points.begin() + 3 * i, points.begin() + 3 * i + 3);
  }
  const Result result = run(reversed, nullptr);
  PHX_CHECK(result.status == PHX_MC_OK);
  check_triangulation(reversed, nullptr, result);
  check_empty_spheres(reversed, nullptr, result);
  PHX_CHECK(six_volume_sum(reversed, result) == 384.0);
}

// Lattice coordinates moved by -1, 0 or +1 ulp: every decision is a
// near-tie that only exact arithmetic resolves consistently.
void test_perturbed_lattice() {
  std::vector<double> points = lattice(5, 1.0);
  std::uint64_t state = 99;
  for (double& value : points) {
    state = splitmix64(state);
    const int move = static_cast<int>(state % 3) - 1;
    if (move != 0) {
      value = std::nextafter(value, move > 0 ? 1e9 : -1e9);
    }
  }
  const Result result = run(points, nullptr);
  PHX_CHECK(result.status == PHX_MC_OK);
  check_triangulation(points, nullptr, result);
  check_empty_spheres(points, nullptr, result);
  PHX_CHECK_NEAR(six_volume_sum(points, result), 384.0, 1e-9);
}

// All 84 integer points of x^2 + y^2 + z^2 = 50: every five points are
// cospherical and many are coplanar.
void test_sphere_points() {
  std::vector<double> points;
  for (int x = -7; x <= 7; ++x) {
    for (int y = -7; y <= 7; ++y) {
      for (int z = -7; z <= 7; ++z) {
        if (x * x + y * y + z * z == 50) {
          points.insert(points.end(), {double(x), double(y), double(z)});
        }
      }
    }
  }
  const int32_t count = static_cast<int32_t>(points.size() / 3);
  PHX_CHECK(count == 84);
  const Result result = run(points, nullptr);
  PHX_CHECK(result.status == PHX_MC_OK);
  // Every point is extreme: the hull is a triangulated sphere with 2n - 4 facets.
  PHX_CHECK(check_triangulation(points, nullptr, result) ==
            static_cast<std::size_t>(2 * count - 4));
  check_empty_spheres(points, nullptr, result);
  check_vertex_map(result);
  PHX_CHECK(std::count(result.vertex_map.begin(), result.vertex_map.end(), -1) == 0);

  // The center lies inside every circumsphere: the triangulation is the cone
  // from the center over the hull.
  points.insert(points.end(), {0.0, 0.0, 0.0});
  const Result coned = run(points, nullptr);
  PHX_CHECK(coned.status == PHX_MC_OK);
  PHX_CHECK(check_triangulation(points, nullptr, coned) ==
            static_cast<std::size_t>(2 * count - 4));
  PHX_CHECK(cell_count(coned) == static_cast<std::size_t>(2 * count - 4));
  for (std::size_t c = 0; c < cell_count(coned); ++c) {
    PHX_CHECK(cell(coned, c)[0] == count || cell(coned, c)[1] == count ||
              cell(coned, c)[2] == count || cell(coned, c)[3] == count);
  }
  check_empty_spheres(points, nullptr, coned);
}

void test_degenerate_inputs() {
  std::vector<double> plane;
  std::vector<double> tilted;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      plane.insert(plane.end(), {double(i), double(j), 0.0});
      tilted.insert(tilted.end(), {double(i), double(j), double(2 * i - 3 * j + 1)});
    }
  }
  PHX_CHECK(run(plane, nullptr).status == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(run(tilted, nullptr).status == PHX_MC_DEGENERATE_INPUT);
  std::vector<double> line;
  for (int i = 0; i < 8; ++i) {
    line.insert(line.end(), {double(i), double(2 * i), 0.5 * i});
  }
  PHX_CHECK(run(line, nullptr).status == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(run({0, 0, 0, 1, 0, 0, 0, 1, 0}, nullptr).status == PHX_MC_DEGENERATE_INPUT);
  std::vector<double> same;
  for (int i = 0; i < 10; ++i) {
    same.insert(same.end(), {0.25, 0.5, 0.75});
  }
  PHX_CHECK(run(same, nullptr).status == PHX_MC_DEGENERATE_INPUT);
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(0, nullptr, kUnlimited, &mesh) == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(mesh == nullptr);
  std::vector<double> weights(plane.size() / 3, 0.0);
  PHX_CHECK(run(plane, &weights).status == PHX_MC_DEGENERATE_INPUT);

  // A coplanar grid plus one apex: base points lie on a hull facet plane.
  std::vector<double> pyramid(plane);
  pyramid.insert(pyramid.end(), {2.0, 2.0, 1.0});
  const Result result = run(pyramid, nullptr);
  PHX_CHECK(result.status == PHX_MC_OK);
  check_triangulation(pyramid, nullptr, result);
  check_empty_spheres(pyramid, nullptr, result);
  PHX_CHECK(std::count(result.vertex_map.begin(), result.vertex_map.end(), -1) == 0);
  PHX_CHECK(six_volume_sum(pyramid, result) == 32.0);
}

void test_duplicates() {
  std::vector<double> points = random_points(60, 5);
  std::copy_n(points.begin() + 3 * 40, 3, points.begin() + 3 * 3);
  points.insert(points.end(), points.begin() + 3 * 5, points.begin() + 3 * 5 + 3);
  points.insert(points.end(), points.begin() + 3 * 17, points.begin() + 3 * 17 + 3);
  const Result result = run(points, nullptr);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.vertex_map[3] == 3);
  PHX_CHECK(result.vertex_map[40] == 3);
  PHX_CHECK(result.vertex_map[5] == 5);
  PHX_CHECK(result.vertex_map[60] == 5);
  PHX_CHECK(result.vertex_map[17] == 17);
  PHX_CHECK(result.vertex_map[61] == 17);
  check_vertex_map(result);
  for (int32_t vertex : result.cells) {
    PHX_CHECK(vertex != 40 && vertex != 60 && vertex != 61);
  }
  check_triangulation(points, nullptr, result);
  check_empty_spheres(points, nullptr, result);
}

void test_determinism() {
  const std::vector<double> points = random_points(500, 21);
  const Result first = run(points, nullptr);
  const Result second = run(points, nullptr);
  PHX_CHECK(first.status == PHX_MC_OK && second.status == PHX_MC_OK);
  PHX_CHECK(first.cells == second.cells);
  PHX_CHECK(first.vertex_map == second.vertex_map);

  // Points in general position have a unique Delaunay triangulation.
  const int32_t count = 500;
  std::vector<int32_t> permutation(count);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::uint64_t state = 3;
  for (int32_t i = count - 1; i > 0; --i) {
    state = splitmix64(state);
    std::swap(permutation[static_cast<std::size_t>(i)],
              permutation[static_cast<std::size_t>(state % static_cast<std::uint64_t>(i + 1))]);
  }
  std::vector<double> shuffled(points.size());
  for (int32_t i = 0; i < count; ++i) {
    std::copy_n(at(points, permutation[static_cast<std::size_t>(i)]), 3,
                shuffled.begin() + 3 * i);
  }
  const Result permuted = run(shuffled, nullptr);
  PHX_CHECK(permuted.status == PHX_MC_OK);
  std::vector<int32_t> identity(count);
  std::iota(identity.begin(), identity.end(), 0);
  PHX_CHECK(sorted_cell_sets(permuted, permutation) == sorted_cell_sets(first, identity));

  const std::vector<double> grid = lattice(5, 0.0);
  PHX_CHECK(run(grid, nullptr).cells == run(grid, nullptr).cells);
}

void test_cube_volume() {
  std::vector<double> points;
  for (int corner = 0; corner < 8; ++corner) {
    points.insert(points.end(), {double(corner & 1), double((corner >> 1) & 1),
                                 double((corner >> 2) & 1)});
  }
  std::uint64_t state = 77;
  for (int i = 0; i < 300; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      points.push_back(0.05 + 0.9 * uniform(state));
    }
  }
  const Result result = run(points, nullptr);
  PHX_CHECK(result.status == PHX_MC_OK);
  check_triangulation(points, nullptr, result);
  PHX_CHECK_NEAR(six_volume_sum(points, result) / 6.0, 1.0, 1e-12);
  // Every hull facet lies on a cube face.
  const std::vector<Facet> facets = sorted_facets(result);
  for (std::size_t i = 0; i < facets.size(); ++i) {
    const bool shared = (i > 0 && facets[i - 1].key == facets[i].key) ||
                        (i + 1 < facets.size() && facets[i + 1].key == facets[i].key);
    if (shared) {
      continue;
    }
    bool on_face = false;
    for (int axis = 0; axis < 3; ++axis) {
      const double value = at(points, facets[i].key[0])[axis];
      on_face = on_face || ((value == 0.0 || value == 1.0) &&
                            at(points, facets[i].key[1])[axis] == value &&
                            at(points, facets[i].key[2])[axis] == value);
    }
    PHX_CHECK(on_face);
  }
}

void test_regular_zero_weights_match_delaunay() {
  for (const std::vector<double>& points : {lattice(5, 0.0), random_points(300, 8)}) {
    const std::vector<double> weights(points.size() / 3, 0.0);
    const Result delaunay = run(points, nullptr);
    const Result regular = run(points, &weights);
    PHX_CHECK(delaunay.status == PHX_MC_OK && regular.status == PHX_MC_OK);
    PHX_CHECK(delaunay.cells == regular.cells);
    PHX_CHECK(delaunay.vertex_map == regular.vertex_map);
  }
}

// Unit cube corners with zero weight lift onto the plane h(c) = 3/2 at the
// center c, whose own lift is |c|^2 - w = 3/4 - w: the center is redundant iff
// w < -3/4 and otherwise conflicts with every corner tetrahedron.
void test_regular_center_point() {
  std::vector<double> points;
  for (int corner = 0; corner < 8; ++corner) {
    points.insert(points.end(), {double(corner & 1), double((corner >> 1) & 1),
                                 double((corner >> 2) & 1)});
  }
  points.insert(points.end(), {0.5, 0.5, 0.5});
  std::vector<double> weights(9, 0.0);
  weights[8] = -1.0;
  const Result redundant = run(points, &weights);
  PHX_CHECK(redundant.status == PHX_MC_OK);
  PHX_CHECK(redundant.vertex_map[8] == -1);
  PHX_CHECK(six_volume_sum(points, redundant) == 6.0);
  check_triangulation(points, &weights, redundant);
  check_empty_spheres(points, &weights, redundant);
  check_vertex_map(redundant);

  weights[8] = -0.5;
  const Result coned = run(points, &weights);
  PHX_CHECK(coned.status == PHX_MC_OK);
  PHX_CHECK(coned.vertex_map[8] == 8);
  PHX_CHECK(cell_count(coned) == 12);
  for (std::size_t c = 0; c < cell_count(coned); ++c) {
    PHX_CHECK(cell(coned, c)[0] == 8 || cell(coned, c)[1] == 8 || cell(coned, c)[2] == 8 ||
              cell(coned, c)[3] == 8);
  }
  PHX_CHECK(six_volume_sum(points, coned) == 6.0);
  check_empty_spheres(points, &weights, coned);
}

// Heavy points swallow nearby light vertices, including ones inserted earlier
// (their whole star lies in a later cavity).
void test_regular_redundant_points() {
  std::vector<double> points = random_points(400, 31);
  std::vector<double> weights(400, 0.0);
  std::uint64_t state = 41;
  for (int i = 0; i < 400; ++i) {
    weights[static_cast<std::size_t>(i)] = (i % 7 == 0) ? 0.05 * uniform(state) : 0.0;
  }
  // Coincident weighted points: the heavier one wins, equal weights merge.
  std::copy_n(points.begin() + 3 * 10, 3, points.begin() + 3 * 20);
  weights[10] = 0.001;
  weights[20] = 0.002;
  std::copy_n(points.begin() + 3 * 30, 3, points.begin() + 3 * 31);
  weights[30] = 0.0;
  weights[31] = 0.0;
  const Result result = run(points, &weights);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.vertex_map[10] == -1);
  PHX_CHECK(result.vertex_map[31] == result.vertex_map[30]);
  PHX_CHECK(result.vertex_map[30] == 30 || result.vertex_map[30] == -1);
  check_vertex_map(result);
  check_triangulation(points, &weights, result);
  check_empty_spheres(points, &weights, result);
  const auto redundant = std::count(result.vertex_map.begin(), result.vertex_map.end(), -1);
  PHX_CHECK(redundant > 10);
  std::printf("regular_3d: %lld of 400 points redundant\n", static_cast<long long>(redundant));

  const Result again = run(points, &weights);
  PHX_CHECK(again.cells == result.cells && again.vertex_map == result.vertex_map);
}

void test_max_tetrahedra() {
  const std::vector<double> points = random_points(100, 13);
  const Result full = run(points, nullptr);
  PHX_CHECK(full.status == PHX_MC_OK);
  const auto count = static_cast<int64_t>(cell_count(full));
  const Result exact = run(points, nullptr, count);
  PHX_CHECK(exact.status == PHX_MC_OK);
  PHX_CHECK(exact.cells == full.cells);
  PHX_CHECK(run(points, nullptr, count - 1).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(run(points, nullptr, 0).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(run(points, nullptr, 1).status == PHX_MC_CAPACITY_EXCEEDED);
  const std::vector<double> weights(100, 0.0);
  PHX_CHECK(run(points, &weights, count - 1).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(run(points, &weights, count).status == PHX_MC_OK);
}

void test_status_codes() {
  const std::vector<double> points = random_points(10, 1);
  const std::vector<double> weights(10, 0.0);
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_delaunay_3d(10, points.data(), kUnlimited, nullptr) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_delaunay_3d(10, nullptr, kUnlimited, &mesh) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(mesh == nullptr);
  PHX_CHECK(phx_mc_delaunay_3d(-1, points.data(), kUnlimited, &mesh) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_delaunay_3d(10, points.data(), -1, &mesh) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_delaunay_3d(int64_t{1} << 31, points.data(), kUnlimited, &mesh) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_regular_3d(10, points.data(), nullptr, kUnlimited, &mesh) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_regular_3d(10, points.data(), weights.data(), -5, &mesh) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(mesh == nullptr);

  std::vector<double> bad(points);
  bad[7] = std::nan("");
  PHX_CHECK(run(bad, nullptr).status == PHX_MC_NONFINITE_INPUT);
  bad[7] = std::numeric_limits<double>::infinity();
  PHX_CHECK(run(bad, nullptr).status == PHX_MC_NONFINITE_INPUT);
  bad[7] = 1e300;
  PHX_CHECK(run(bad, nullptr).status == PHX_MC_RANGE_ERROR);
  bad[7] = 1e-300;
  PHX_CHECK(run(bad, nullptr).status == PHX_MC_RANGE_ERROR);

  std::vector<double> bad_weights(weights);
  bad_weights[4] = std::nan("");
  PHX_CHECK(run(points, &bad_weights).status == PHX_MC_NONFINITE_INPUT);
  bad_weights[4] = 1e-200;
  PHX_CHECK(run(points, &bad_weights).status == PHX_MC_RANGE_ERROR);
  bad_weights[4] = -0x1p241;
  PHX_CHECK(run(points, &bad_weights).status == PHX_MC_RANGE_ERROR);
}

bool all_coplanar(const std::vector<double>& points) {
  const int32_t count = static_cast<int32_t>(points.size() / 3);
  for (int32_t a = 0; a < count; ++a) {
    for (int32_t b = a + 1; b < count; ++b) {
      for (int32_t c = b + 1; c < count; ++c) {
        for (int32_t d = c + 1; d < count; ++d) {
          if (orient3d(at(points, a), at(points, b), at(points, c), at(points, d)) != 0) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// Random multisets of small integer lattices with quantized weights: exact
// duplicates, coplanar/collinear subsets, cospherical and equal-power ties
// everywhere, translated far from the origin in some rounds.
void test_degenerate_lattice_subsets() {
  int triangulated = 0;
  for (int round = 0; round < 400; ++round) {
    std::uint64_t state = splitmix64(1000 + static_cast<std::uint64_t>(round));
    const int side = 2 + static_cast<int>(state % 4);
    state = splitmix64(state);
    const int count = 4 + static_cast<int>(state % 40);
    const double offset = round % 3 == 0 ? 1048576.0 : 0.0;
    std::vector<double> points;
    std::vector<double> weights;
    for (int i = 0; i < count; ++i) {
      for (int axis = 0; axis < 3; ++axis) {
        state = splitmix64(state);
        points.push_back(offset + static_cast<double>(state % static_cast<std::uint64_t>(side)));
      }
      state = splitmix64(state);
      weights.push_back(0.25 * static_cast<double>(state % 5));
    }
    const bool flat = all_coplanar(points);
    for (const std::vector<double>* w : {static_cast<const std::vector<double>*>(nullptr),
                                         static_cast<const std::vector<double>*>(&weights)}) {
      const Result result = run(points, w);
      if (flat) {
        PHX_CHECK(result.status == PHX_MC_DEGENERATE_INPUT);
        continue;
      }
      PHX_CHECK(result.status == PHX_MC_OK);
      if (result.status != PHX_MC_OK) {
        continue;
      }
      ++triangulated;
      check_triangulation(points, w, result);
      check_empty_spheres(points, w, result);
      check_vertex_map(result);
      if (w == nullptr) {
        PHX_CHECK(std::count(result.vertex_map.begin(), result.vertex_map.end(), -1) == 0);
      }
    }
  }
  PHX_CHECK(triangulated > 700);
}

void test_performance() {
  const int64_t count = 100000;
  const std::vector<double> points = random_points(count, 2026);
  const auto start = std::chrono::steady_clock::now();
  const Result result = run(points, nullptr);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("delaunay_3d: %lld points -> %zu tetrahedra in %.3f s\n",
              static_cast<long long>(count), cell_count(result), seconds);
  PHX_CHECK(result.status == PHX_MC_OK);
  check_triangulation(points, nullptr, result);
  PHX_CHECK(std::count(result.vertex_map.begin(), result.vertex_map.end(), -1) == 0);
#ifdef NDEBUG
  PHX_CHECK(seconds < 5.0);
#endif

  std::vector<double> weights(static_cast<std::size_t>(count));
  std::uint64_t state = 5;
  for (double& w : weights) {
    w = 1e-4 * uniform(state);
  }
  const auto regular_start = std::chrono::steady_clock::now();
  const Result regular = run(points, &weights);
  const double regular_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - regular_start).count();
  std::printf("regular_3d: %lld points -> %zu tetrahedra, %lld redundant in %.3f s\n",
              static_cast<long long>(count), cell_count(regular),
              static_cast<long long>(
                  std::count(regular.vertex_map.begin(), regular.vertex_map.end(), -1)),
              regular_seconds);
  PHX_CHECK(regular.status == PHX_MC_OK);
  check_triangulation(points, &weights, regular, false);
  check_vertex_map(regular);
}

}  // namespace

int main() {
  test_random_points();
  test_cubic_lattice();
  test_perturbed_lattice();
  test_sphere_points();
  test_degenerate_inputs();
  test_degenerate_lattice_subsets();
  test_duplicates();
  test_determinism();
  test_cube_volume();
  test_regular_zero_weights_match_delaunay();
  test_regular_center_point();
  test_regular_redundant_points();
  test_max_tetrahedra();
  test_status_codes();
  test_performance();
  return phx::mc::test::finish("test_delaunay3d");
}
