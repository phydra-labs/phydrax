//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"
#include "triangulation2d.hpp"

namespace {

using namespace phx::mc;

double uniform(std::uint64_t& state) {
  state = splitmix64(state);
  return static_cast<double>(state >> 11) * 0x1p-53;
}

struct Result {
  int32_t status = -1;
  std::vector<double> points;
  std::vector<int32_t> cells;
  std::vector<int32_t> vertex_map;
};

Result collect(int32_t status, phx_mc_mesh* mesh) {
  Result result;
  result.status = status;
  if (mesh == nullptr) {
    return result;
  }
  PHX_CHECK(phx_mc_mesh_dimension(mesh) == 2);
  result.points.resize(static_cast<std::size_t>(2 * phx_mc_mesh_point_count(mesh)));
  result.cells.resize(static_cast<std::size_t>(3 * phx_mc_mesh_cell_count(mesh)));
  result.vertex_map.resize(static_cast<std::size_t>(phx_mc_mesh_input_point_count(mesh)));
  phx_mc_mesh_copy_points(mesh, result.points.data());
  phx_mc_mesh_copy_cells(mesh, result.cells.data());
  phx_mc_mesh_copy_vertex_map(mesh, result.vertex_map.data());
  std::vector<int32_t> segments(result.cells.size());
  phx_mc_mesh_copy_cell_constraints(mesh, segments.data());
  PHX_CHECK(std::all_of(segments.begin(), segments.end(), [](int32_t s) { return s == -1; }));
  phx_mc_mesh_free(mesh);
  return result;
}

Result delaunay(const std::vector<double>& points, int64_t max_triangles = 1 << 30) {
  phx_mc_mesh* mesh = nullptr;
  const int32_t status = phx_mc_delaunay_2d(static_cast<int64_t>(points.size() / 2), points.data(), max_triangles, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh);
  PHX_CHECK((status == PHX_MC_OK) == (mesh != nullptr));
  return collect(status, mesh);
}

Result regular(const std::vector<double>& points, const std::vector<double>& weights) {
  phx_mc_mesh* mesh = nullptr;
  const int32_t status = phx_mc_regular_2d(static_cast<int64_t>(points.size() / 2), points.data(), weights.data(), 1 << 30, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh);
  PHX_CHECK((status == PHX_MC_OK) == (mesh != nullptr));
  return collect(status, mesh);
}

const double* at(const std::vector<double>& points, int32_t v) { return points.data() + 2 * v; }

// Twice the area of the convex hull (Andrew's monotone chain, exact turns).
double hull_twice_area(const std::vector<double>& points) {
  std::vector<std::pair<double, double>> p;
  for (std::size_t k = 0; k < points.size() / 2; ++k) {
    p.emplace_back(points[2 * k], points[2 * k + 1]);
  }
  std::sort(p.begin(), p.end());
  p.erase(std::unique(p.begin(), p.end()), p.end());
  std::vector<std::pair<double, double>> hull(2 * p.size());
  std::size_t count = 0;
  auto turn = [](const std::pair<double, double>& a, const std::pair<double, double>& b,
                 const std::pair<double, double>& c) {
    const double pa[2] = {a.first, a.second}, pb[2] = {b.first, b.second},
                 pc[2] = {c.first, c.second};
    return orient2d(pa, pb, pc);
  };
  for (std::size_t i = 0; i < p.size(); ++i) {
    while (count >= 2 && turn(hull[count - 2], hull[count - 1], p[i]) <= 0) --count;
    hull[count++] = p[i];
  }
  for (std::size_t i = p.size() - 1, lower = count + 1; i-- > 0;) {
    while (count >= lower && turn(hull[count - 2], hull[count - 1], p[i]) <= 0) --count;
    hull[count++] = p[i];
  }
  double area = 0.0;
  for (std::size_t i = 0; i + 1 < count; ++i) {
    area += hull[i].first * hull[i + 1].second - hull[i + 1].first * hull[i].second;
  }
  return area;
}

// Orientation, canonical order, edge manifoldness, Euler relation, area.
void check_triangulation(const std::vector<double>& points, const Result& r) {
  const std::size_t count = r.cells.size() / 3;
  std::set<std::pair<int32_t, int32_t>> edges;
  std::set<int32_t> used;
  double twice_area = 0.0;
  for (std::size_t c = 0; c < count; ++c) {
    const int32_t* v = r.cells.data() + 3 * c;
    PHX_CHECK(orient2d(at(points, v[0]), at(points, v[1]), at(points, v[2])) > 0);
    PHX_CHECK(v[0] < v[1] && v[0] < v[2]);
    if (c > 0) {
      PHX_CHECK(std::lexicographical_compare(v - 3, v, v, v + 3));
    }
    for (int k = 0; k < 3; ++k) {
      PHX_CHECK(edges.insert({v[k], v[(k + 1) % 3]}).second);
      used.insert(v[k]);
    }
    const double* a = at(points, v[0]);
    const double* b = at(points, v[1]);
    const double* d = at(points, v[2]);
    twice_area += (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);
  }
  std::size_t boundary = 0;
  for (const auto& [a, b] : edges) {
    boundary += edges.count({b, a}) == 0 ? 1 : 0;
  }
  // T = 2 V - 2 - B for a triangulated convex region with B boundary edges.
  PHX_CHECK(count == 2 * used.size() - 2 - boundary);
  const double hull = hull_twice_area(points);
  PHX_CHECK_NEAR(twice_area, hull, 1e-12 * std::fabs(hull));
  for (std::size_t i = 0; i < r.vertex_map.size(); ++i) {
    const int32_t target = r.vertex_map[i];
    if (target >= 0) {
      PHX_CHECK(used.count(target) == 1);
      PHX_CHECK(at(points, target)[0] == points[2 * i] && at(points, target)[1] == points[2 * i + 1]);
    }
  }
}

// Every vertex lies outside or on every circumcircle (exact).
void check_empty_circles(const std::vector<double>& points, const Result& r) {
  std::set<int32_t> used;
  for (int32_t v : r.cells) used.insert(v);
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) {
    const int32_t* v = r.cells.data() + 3 * c;
    for (int32_t p : used) {
      PHX_CHECK(incircle(at(points, v[0]), at(points, v[1]), at(points, v[2]), at(points, p)) <= 0);
    }
  }
}

std::vector<double> random_points(std::size_t count, std::uint64_t seed) {
  std::vector<double> points(2 * count);
  for (double& x : points) x = uniform(seed);
  return points;
}

void test_random_empty_circle() {
  const std::vector<double> points = random_points(500, 7);
  const Result r = delaunay(points);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_triangulation(points, r);
  check_empty_circles(points, r);
  PHX_CHECK(r.points == points);
  for (std::size_t i = 0; i < r.vertex_map.size(); ++i) PHX_CHECK(r.vertex_map[i] == static_cast<int32_t>(i));
}

void test_lattice() {
  std::vector<double> points;
  for (int i = 0; i < 20; ++i)
    for (int j = 0; j < 20; ++j) {
      points.push_back(0.25 * i);
      points.push_back(0.25 * j);
    }
  const Result r = delaunay(points);
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.cells.size() / 3 == 2 * 19 * 19);
  check_triangulation(points, r);
  check_empty_circles(points, r);
  // Every cell is a half lattice square (area 1/32).
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) {
    const double* a = at(points, r.cells[3 * c]);
    const double* b = at(points, r.cells[3 * c + 1]);
    const double* d = at(points, r.cells[3 * c + 2]);
    PHX_CHECK((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]) == 0.0625);
  }
}

void test_cocircular() {
  // Twelve integer points on the circle of radius 5 plus the center.
  const std::vector<double> points = {5, 0, 4, 3, 3, 4, 0, 5, -3, 4, -4, 3, -5, 0,
                                      -4, -3, -3, -4, 0, -5, 3, -4, 4, -3};
  const Result r = delaunay(points);
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.cells.size() / 3 == 10);
  check_triangulation(points, r);
  check_empty_circles(points, r);
  std::vector<double> with_center = points;
  with_center.push_back(0.0);
  with_center.push_back(0.0);
  const Result c = delaunay(with_center);
  PHX_CHECK(c.cells.size() / 3 == 12);
  check_triangulation(with_center, c);
  check_empty_circles(with_center, c);
}

void test_degenerate_and_arguments() {
  const std::vector<double> collinear = {0, 0, 1, 1, 2, 2, 3, 3, 0.5, 0.5};
  PHX_CHECK(delaunay(collinear).status == PHX_MC_DEGENERATE_INPUT);
  const std::vector<double> repeated = {1, 2, 1, 2, 1, 2, 3, 4};
  PHX_CHECK(delaunay(repeated).status == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(delaunay({}).status == PHX_MC_DEGENERATE_INPUT);
  phx_mc_mesh* mesh = reinterpret_cast<phx_mc_mesh*>(&mesh);
  const double pts[6] = {0, 0, 1, 0, 0, 1};
  PHX_CHECK(phx_mc_delaunay_2d(-1, pts, 10, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(mesh == nullptr);
  PHX_CHECK(phx_mc_delaunay_2d(3, pts, 10, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_delaunay_2d(3, nullptr, 10, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_delaunay_2d(3, pts, -1, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_regular_2d(3, pts, nullptr, 10, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh) == PHX_MC_INVALID_ARGUMENT);
  const double nan_pts[6] = {0, 0, 1, std::nan(""), 0, 1};
  PHX_CHECK(phx_mc_delaunay_2d(3, nan_pts, 10, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh) == PHX_MC_NONFINITE_INPUT);
  const double big_pts[6] = {0, 0, 1e300, 0, 0, 1};
  PHX_CHECK(phx_mc_delaunay_2d(3, big_pts, 10, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh) == PHX_MC_RANGE_ERROR);
  const double weights[3] = {0, 1e-200, 0};
  PHX_CHECK(phx_mc_regular_2d(3, pts, weights, 10, INT64_MAX, INT64_MAX, INT64_MAX, nullptr, nullptr, &mesh) == PHX_MC_RANGE_ERROR);
  PHX_CHECK(mesh == nullptr);
}

void test_capacity() {
  const std::vector<double> points = random_points(200, 11);
  const Result full = delaunay(points);
  const int64_t count = static_cast<int64_t>(full.cells.size() / 3);
  PHX_CHECK(delaunay(points, count).status == PHX_MC_OK);
  PHX_CHECK(delaunay(points, count - 1).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(delaunay(points, 0).status == PHX_MC_CAPACITY_EXCEEDED);
}

void test_duplicates() {
  std::vector<double> points = random_points(50, 3);
  points.insert(points.end(), {points[10], points[11], points[0], points[1], points[10], points[11]});
  const Result r = delaunay(points);
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.vertex_map[50] == 5);
  PHX_CHECK(r.vertex_map[51] == 0);
  PHX_CHECK(r.vertex_map[52] == 5);
  for (int32_t v : r.cells) PHX_CHECK(v < 50);
  check_triangulation(points, r);
  check_empty_circles(points, r);
}

std::set<std::vector<int32_t>> relabeled(const Result& r, const std::vector<int32_t>& label) {
  std::set<std::vector<int32_t>> cells;
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) {
    std::vector<int32_t> v = {label[r.cells[3 * c]], label[r.cells[3 * c + 1]], label[r.cells[3 * c + 2]]};
    std::rotate(v.begin(), std::min_element(v.begin(), v.end()), v.end());
    cells.insert(v);
  }
  return cells;
}

void test_determinism() {
  const std::vector<double> points = random_points(300, 5);
  const Result first = delaunay(points);
  const Result second = delaunay(points);
  PHX_CHECK(first.cells == second.cells);
  // A permutation of generic points yields the same triangles.
  const std::size_t n = points.size() / 2;
  std::vector<int32_t> permutation(n);
  for (std::size_t i = 0; i < n; ++i) permutation[i] = static_cast<int32_t>((i * 7919) % n);
  std::vector<double> permuted(points.size());
  for (std::size_t i = 0; i < n; ++i) {
    permuted[2 * i] = points[2 * permutation[i]];
    permuted[2 * i + 1] = points[2 * permutation[i] + 1];
  }
  const Result third = delaunay(permuted);
  std::vector<int32_t> identity(n);
  for (std::size_t i = 0; i < n; ++i) identity[i] = static_cast<int32_t>(i);
  PHX_CHECK(relabeled(first, identity) == relabeled(third, permutation));
}

void test_regular_zero_weights_is_delaunay() {
  std::vector<double> points;
  for (int i = 0; i < 12; ++i)
    for (int j = 0; j < 12; ++j) {
      points.push_back(i);
      points.push_back(j);
    }
  const std::vector<double> random = random_points(100, 9);
  for (double x : random) points.push_back(11.0 * x);
  const std::vector<double> weights(points.size() / 2, 0.0);
  const Result d = delaunay(points);
  const Result w = regular(points, weights);
  PHX_CHECK(w.status == PHX_MC_OK);
  PHX_CHECK(d.cells == w.cells);
  PHX_CHECK(d.vertex_map == w.vertex_map);
}

void test_regular_redundant() {
  // Unit square corners with weight 1: the center with weight 0 is redundant,
  // with weight 1 it is a vertex.
  std::vector<double> points = {0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5};
  Result r = regular(points, {1, 1, 1, 1, 0});
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.vertex_map[4] == -1);
  PHX_CHECK(r.cells.size() / 3 == 2);
  r = regular(points, {1, 1, 1, 1, 1});
  PHX_CHECK(r.vertex_map[4] == 4);
  PHX_CHECK(r.cells.size() / 3 == 4);
  // Coincident lighter point -> -1; equal weight -> representative.
  points.insert(points.end(), {1, 1, 1, 1});
  r = regular(points, {1, 1, 1, 1, 1, 0.5, 1});
  PHX_CHECK(r.vertex_map[5] == -1);
  PHX_CHECK(r.vertex_map[6] == 2);
  // A vertex inserted first and swallowed later: heavy center inserted after a
  // light interior point.
  std::vector<double> swallow = {0, 0, 4, 0, 4, 4, 0, 4, 2.1, 1.9, 2, 2};
  r = regular(swallow, {0, 0, 0, 0, 0, 3});
  PHX_CHECK(r.vertex_map[4] == -1);
  PHX_CHECK(r.vertex_map[5] == 5);
  PHX_CHECK(r.cells.size() / 3 == 4);
}

// Regular triangulation property for random weights, checked exactly.
void test_regular_random() {
  std::uint64_t seed = 21;
  const std::size_t n = 400;
  std::vector<double> points(2 * n), weights(n);
  for (double& x : points) x = uniform(seed);
  for (double& w : weights) w = 0.004 * uniform(seed);
  const Result r = regular(points, weights);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_triangulation(points, r);
  int redundant = 0;
  std::set<int32_t> used(r.cells.begin(), r.cells.end());
  for (std::size_t i = 0; i < n; ++i) {
    redundant += r.vertex_map[i] == -1 ? 1 : 0;
    PHX_CHECK((r.vertex_map[i] == -1) == (used.count(static_cast<int32_t>(i)) == 0));
  }
  PHX_CHECK(redundant > 0);
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) {
    const int32_t* v = r.cells.data() + 3 * c;
    for (std::size_t p = 0; p < n; ++p) {
      // No point (vertex or redundant) lies strictly below the lifted facet.
      PHX_CHECK(power2d(at(points, v[0]), at(points, v[1]), at(points, v[2]),
                        at(points, static_cast<int32_t>(p)), weights[v[0]], weights[v[1]],
                        weights[v[2]], weights[p]) <= 0);
    }
  }
}

void test_performance() {
  const std::vector<double> points = random_points(100000, 1234);
  const auto start = std::chrono::steady_clock::now();
  const Result r = delaunay(points);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("delaunay_2d 100k random points: %.3f s, %zu triangles\n", seconds,
              r.cells.size() / 3);
  PHX_CHECK(r.status == PHX_MC_OK);
}

void test_native_budget_boundaries() {
  const std::vector<double> points = random_points(80, 19);
  const std::vector<double> original = points;
  uint64_t work[9] = {}, memory[6] = {};
  phx_mc_mesh* mesh = nullptr;
  auto run = [&](int64_t cavity, int64_t units, int64_t bytes) {
    return phx_mc_delaunay_2d(80, points.data(), INT64_MAX, cavity, units, bytes,
                              work, memory, &mesh);
  };
  PHX_CHECK(run(INT64_MAX, INT64_MAX, INT64_MAX) == PHX_MC_OK);
  const Result baseline = collect(PHX_MC_OK, mesh);
  const int64_t exact_work = static_cast<int64_t>(work[0]);
  const int64_t exact_cavity = static_cast<int64_t>(work[6]);
  const int64_t exact_bytes = static_cast<int64_t>(memory[2]);
  PHX_CHECK(work[1] > 0 && work[2] > 0 && work[3] == 0);
  PHX_CHECK(work[4] == 80 && work[5] == 0);
  PHX_CHECK(run(exact_cavity, exact_work, exact_bytes) == PHX_MC_OK);
  const Result bounded = collect(PHX_MC_OK, mesh);
  PHX_CHECK(bounded.cells == baseline.cells && bounded.vertex_map == baseline.vertex_map);
  PHX_CHECK(run(exact_cavity, exact_work - 1, exact_bytes) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(mesh == nullptr && work[7] == 1 && memory[1] == 0);
  PHX_CHECK(run(exact_cavity - 1, INT64_MAX, INT64_MAX) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(mesh == nullptr && work[8] == 1 && work[6] > static_cast<uint64_t>(exact_cavity - 1));
  PHX_CHECK(run(INT64_MAX, INT64_MAX, exact_bytes - 1) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(mesh == nullptr && memory[5] == 1 && memory[1] == 0);
  PHX_CHECK(run(0, INT64_MAX, INT64_MAX) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(run(INT64_MAX, 0, INT64_MAX) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(run(INT64_MAX, INT64_MAX, 0) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(run(-1, INT64_MAX, INT64_MAX) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(points == original);

  const std::vector<double> weights(80, 0.0);
  PHX_CHECK(phx_mc_regular_2d(80, points.data(), weights.data(), INT64_MAX,
                              INT64_MAX, INT64_MAX, INT64_MAX, work, memory, &mesh) == PHX_MC_OK);
  const Result weighted = collect(PHX_MC_OK, mesh);
  PHX_CHECK(weighted.cells == baseline.cells);
  PHX_CHECK(work[2] == 0 && work[3] > 0);

  // A nested pointset call cannot escape an ancestor byte envelope, and its
  // evidence must exclude the ancestor's retained buffers and historical peak.
  const auto ancestor = std::make_shared<BoundedMemoryResource>();
  {
    MemoryScope ancestor_scope(ancestor);
    NativeVector<uint8_t> retained(64);
    {
      NativeVector<uint8_t> historical(static_cast<std::size_t>(2 * exact_bytes));
    }
    const std::size_t retained_bytes = ancestor->live_bytes();
    const std::size_t limited_total = retained_bytes + static_cast<std::size_t>(exact_bytes - 1);
    PHX_CHECK(ancestor->set_limit(limited_total));
    PHX_CHECK(run(INT64_MAX, INT64_MAX, INT64_MAX) == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(mesh == nullptr && memory[5] == 1 && memory[1] == 0);
    PHX_CHECK(memory[0] == static_cast<uint64_t>(exact_bytes - 1));
    PHX_CHECK(memory[2] <= memory[0]);
    PHX_CHECK(ancestor->live_bytes() == retained_bytes);
    PHX_CHECK(ancestor->limit_bytes() == limited_total);
    PHX_CHECK(ancestor->set_limit(retained_bytes + static_cast<std::size_t>(exact_bytes)));
    PHX_CHECK(run(INT64_MAX, INT64_MAX, INT64_MAX) == PHX_MC_OK);
    PHX_CHECK(memory[2] == static_cast<uint64_t>(exact_bytes));
  }
  // Output survives the creating scope and retains the shared allocation owner.
  const Result nested = collect(PHX_MC_OK, mesh);
  PHX_CHECK(nested.cells == baseline.cells && nested.vertex_map == baseline.vertex_map);
  PHX_CHECK(ancestor->live_bytes() == 0);
}

void test_insertion_refusal_preserves_topology() {
  const double points[] = {0, 0, 2, 0, 0, 2, 0.5, 0.5};
  auto trial_owner = std::make_shared<BoundedMemoryResource>();
  PlanarBudget trial_budget(INT64_MAX, INT64_MAX, trial_owner);
  Triangulation2D trial(trial_budget);
  trial.reset(points, 4, nullptr);
  trial.initialize(0, 1, 2);
  trial.insert(3);
  const uint64_t complete_work = trial_budget.work_units;
  for (int refusal = 0; refusal < 4; ++refusal) {
    auto owner = std::make_shared<BoundedMemoryResource>();
    PlanarBudget budget(INT64_MAX, INT64_MAX, owner);
    Triangulation2D tri(budget);
    tri.reset(points, 4, nullptr);
    tri.initialize(0, 1, 2);
    const std::vector<Triangle2D> before(tri.triangles.begin(), tri.triangles.end());
    const std::vector<int32_t> vertices(tri.vertex_triangle.begin(), tri.vertex_triangle.end());
    const std::vector<int32_t> constraints(tri.constraints.begin(), tri.constraints.end());
    if (refusal == 0) budget.max_work = complete_work - 1;  // final commit precharge
    if (refusal == 1) budget.max_cavity_cells = 0;
    if (refusal == 2) PHX_CHECK(owner->set_limit(owner->live_bytes()));
    if (refusal == 3) tri.max_finite_cells = 1;
    bool refused = false;
    try {
      tri.insert(3);
    } catch (const std::bad_alloc&) {
      refused = true;
    }
    PHX_CHECK(refused);
    PHX_CHECK(tri.finite_count == 1 && tri.triangles.size() == before.size());
    PHX_CHECK(std::equal(vertices.begin(), vertices.end(), tri.vertex_triangle.begin()));
    PHX_CHECK(std::equal(constraints.begin(), constraints.end(), tri.constraints.begin()));
    PHX_CHECK(tri.free_slots.empty());
    for (std::size_t t = 0; t < before.size(); ++t) {
      PHX_CHECK(tri.live[t] == 1);
      for (int k = 0; k < 3; ++k) {
        PHX_CHECK(tri.triangles[t].v[k] == before[t].v[k]);
        PHX_CHECK(tri.triangles[t].n[k] == before[t].n[k]);
      }
    }
    budget.max_work = INT64_MAX;
    budget.max_cavity_cells = INT64_MAX;
    tri.max_finite_cells = INT64_MAX;
    PHX_CHECK(owner->set_limit(INT64_MAX));
    PHX_CHECK(tri.insert(3) == Triangulation2D::Insertion::kInserted);
    PHX_CHECK(tri.finite_count == 3 && tri.vertex_triangle[3] >= 0);
    for (std::size_t t = 0; t < tri.triangles.size(); ++t) {
      if (!tri.live[t]) continue;
      for (int k = 0; k < 3; ++k) {
        const int32_t u = tri.triangles[t].v[next3(k)];
        const int32_t w = tri.triangles[t].v[prev3(k)];
        const int32_t nb = tri.triangles[t].n[k];
        const int twin = tri.edge_index(nb, w, u);
        PHX_CHECK(twin >= 0);
        if (twin >= 0) PHX_CHECK(tri.triangles[nb].n[twin] == static_cast<int32_t>(t));
      }
    }
  }
}

}  // namespace

int main() {
  test_random_empty_circle();
  test_lattice();
  test_cocircular();
  test_degenerate_and_arguments();
  test_capacity();
  test_native_budget_boundaries();
  test_insertion_refusal_preserves_topology();
  test_duplicates();
  test_determinism();
  test_regular_zero_weights_is_delaunay();
  test_regular_redundant();
  test_regular_random();
  test_performance();
  return phx::mc::test::finish("test_delaunay2d");
}
