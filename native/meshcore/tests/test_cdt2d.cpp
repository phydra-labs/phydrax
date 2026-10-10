//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "bounded_memory.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"

namespace {

using namespace phx::mc;

constexpr double kInf = std::numeric_limits<double>::infinity();

double uniform(std::uint64_t& state) {
  state = splitmix64(state);
  return static_cast<double>(state >> 11) * 0x1p-53;
}

struct Input {
  std::vector<double> points;
  std::vector<int32_t> segments;
  std::vector<double> holes;
  int32_t keep_hull = 0;
  double min_angle = 0.0;
  double max_area = kInf;
  int64_t max_steiner = 1 << 20;
  int64_t max_triangles = 1 << 30;
  int64_t max_cavity_cells = std::numeric_limits<int64_t>::max();
  int64_t max_work = std::numeric_limits<int64_t>::max();
  int64_t max_scratch_bytes = std::numeric_limits<int64_t>::max();
};

struct Result {
  int32_t status = -1;
  std::vector<double> points;
  std::vector<int32_t> cells;
  std::vector<int32_t> segments;  // per cell edge
  std::vector<int32_t> vertex_map;
  std::array<uint64_t, 9> work{};
  std::array<uint64_t, 6> memory{};
};

Result run(const Input& in) {
  phx_mc_mesh* mesh = nullptr;
  Result r;
  r.status = phx_mc_constrained_delaunay_2d(
      static_cast<int64_t>(in.points.size() / 2), in.points.data(),
      static_cast<int64_t>(in.segments.size() / 2), in.segments.data(),
      static_cast<int64_t>(in.holes.size() / 2), in.holes.data(), in.keep_hull, in.min_angle,
      in.max_area, in.max_steiner, in.max_triangles, in.max_cavity_cells, in.max_work,
      in.max_scratch_bytes, r.work.data(), r.memory.data(), &mesh);
  const bool has_mesh = r.status == PHX_MC_OK || r.status == PHX_MC_REFINEMENT_LIMIT;
  PHX_CHECK(has_mesh == (mesh != nullptr));
  if (mesh == nullptr) {
    return r;
  }
  r.points.resize(static_cast<std::size_t>(2 * phx_mc_mesh_point_count(mesh)));
  r.cells.resize(static_cast<std::size_t>(3 * phx_mc_mesh_cell_count(mesh)));
  r.segments.resize(r.cells.size());
  r.vertex_map.resize(static_cast<std::size_t>(phx_mc_mesh_input_point_count(mesh)));
  phx_mc_mesh_copy_points(mesh, r.points.data());
  phx_mc_mesh_copy_cells(mesh, r.cells.data());
  phx_mc_mesh_copy_cell_constraints(mesh, r.segments.data());
  phx_mc_mesh_copy_vertex_map(mesh, r.vertex_map.data());
  phx_mc_mesh_free(mesh);
  return r;
}

const double* at(const Result& r, int32_t v) { return r.points.data() + 2 * v; }

double twice_area(const Result& r, std::size_t c) {
  const double* a = at(r, r.cells[3 * c]);
  const double* b = at(r, r.cells[3 * c + 1]);
  const double* d = at(r, r.cells[3 * c + 2]);
  return (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);
}

double total_area(const Result& r) {
  double area = 0.0;
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) area += 0.5 * twice_area(r, c);
  return area;
}

double min_angle_degrees(const Result& r, std::size_t c) {
  double smallest = 180.0;
  for (int k = 0; k < 3; ++k) {
    const double* o = at(r, r.cells[3 * c + k]);
    const double* p = at(r, r.cells[3 * c + (k + 1) % 3]);
    const double* q = at(r, r.cells[3 * c + (k + 2) % 3]);
    const double ux = p[0] - o[0], uy = p[1] - o[1], vx = q[0] - o[0], vy = q[1] - o[1];
    const double angle = std::atan2(std::fabs(ux * vy - uy * vx), ux * vx + uy * vy);
    smallest = std::min(smallest, angle * 180.0 / 3.14159265358979323846);
  }
  return smallest;
}

// Orientation, canonical order, manifold edges, consistent edge labels.
void check_mesh(const Result& r) {
  std::map<std::pair<int32_t, int32_t>, int32_t> edges;
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) {
    const int32_t* v = r.cells.data() + 3 * c;
    PHX_CHECK(orient2d(at(r, v[0]), at(r, v[1]), at(r, v[2])) > 0);
    PHX_CHECK(v[0] < v[1] && v[0] < v[2]);
    if (c > 0) PHX_CHECK(std::lexicographical_compare(v - 3, v, v, v + 3));
    for (int k = 0; k < 3; ++k) {
      const std::pair<int32_t, int32_t> edge{v[(k + 1) % 3], v[(k + 2) % 3]};
      PHX_CHECK(edges.emplace(edge, r.segments[3 * c + k]).second);
    }
  }
  for (const auto& [edge, label] : edges) {
    const auto twin = edges.find({edge.second, edge.first});
    if (twin != edges.end()) PHX_CHECK(twin->second == label);
  }
}

// Segment s is a chain of edges labeled s from vertex_map[a] to vertex_map[b].
bool segment_preserved(const Result& r, const Input& in, int32_t s) {
  std::map<int32_t, std::set<int32_t>> adjacency;
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) {
    for (int k = 0; k < 3; ++k) {
      if (r.segments[3 * c + k] == s) {
        const int32_t u = r.cells[3 * c + (k + 1) % 3], w = r.cells[3 * c + (k + 2) % 3];
        adjacency[u].insert(w);
        adjacency[w].insert(u);
      }
    }
  }
  const int32_t a = r.vertex_map[in.segments[2 * s]];
  const int32_t b = r.vertex_map[in.segments[2 * s + 1]];
  int32_t previous = -1, current = a;
  for (std::size_t steps = 0; steps <= adjacency.size(); ++steps) {
    if (current == b) return adjacency[a].size() == 1 && adjacency[b].size() == 1;
    const std::set<int32_t>& next = adjacency[current];
    if (current != a && next.size() != 2) return false;
    int32_t chosen = -1;
    for (int32_t v : next) if (v != previous) chosen = v;
    if (chosen < 0) return false;
    previous = current;
    current = chosen;
  }
  return false;
}

// Every free interior edge is locally Delaunay (exact).
void check_constrained_delaunay(const Result& r) {
  std::map<std::pair<int32_t, int32_t>, std::pair<std::size_t, int>> edges;
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c)
    for (int k = 0; k < 3; ++k)
      edges[{r.cells[3 * c + (k + 1) % 3], r.cells[3 * c + (k + 2) % 3]}] = {c, k};
  for (const auto& [edge, where] : edges) {
    const auto twin = edges.find({edge.second, edge.first});
    if (twin == edges.end() || r.segments[3 * where.first + where.second] != -1) continue;
    const int32_t* v = r.cells.data() + 3 * where.first;
    const int32_t x = r.cells[3 * twin->second.first + twin->second.second];
    PHX_CHECK(incircle(at(r, v[0]), at(r, v[1]), at(r, v[2]), at(r, x)) <= 0);
  }
}

void check_segments(const Result& r, const Input& in) {
  for (int32_t s = 0; s < static_cast<int32_t>(in.segments.size() / 2); ++s) {
    PHX_CHECK(segment_preserved(r, in, s));
  }
}

Input square_with_hole() {
  Input in;
  in.points = {0, 0, 4, 0, 4, 4, 0, 4, 1, 1, 3, 1, 3, 3, 1, 3, 0.5, 2.5, 3.5, 0.5};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4};
  in.holes = {2, 2};
  return in;
}

void test_segments_and_vertex_on_segment() {
  Input in;
  in.points = {0, 0, 2, 0, 1, 0, 0.2, 1, 0.6, -1, 3, 0.5};
  in.segments = {0, 1, 3, 4};
  in.keep_hull = 1;
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_CONSTRAINT_INTERSECTION);
  in.segments = {0, 1, 3, 2};
  const Result ok = run(in);
  PHX_CHECK(ok.status == PHX_MC_OK);
  check_mesh(ok);
  check_segments(ok, in);
  check_constrained_delaunay(ok);
  PHX_CHECK(ok.points == in.points);
  // Segment 0 passes through vertex 2: both pieces carry id 0.
  std::set<std::pair<int32_t, int32_t>> labeled;
  for (std::size_t c = 0; c < ok.cells.size() / 3; ++c)
    for (int k = 0; k < 3; ++k)
      if (ok.segments[3 * c + k] == 0) {
        const int32_t u = ok.cells[3 * c + (k + 1) % 3], w = ok.cells[3 * c + (k + 2) % 3];
        labeled.insert({std::min(u, w), std::max(u, w)});
      }
  PHX_CHECK((labeled == std::set<std::pair<int32_t, int32_t>>{{0, 2}, {1, 2}}));
  // Overlapping collinear segments and duplicate segments intersect.
  in.segments = {0, 1, 2, 1};
  PHX_CHECK(run(in).status == PHX_MC_CONSTRAINT_INTERSECTION);
  in.segments = {0, 3, 3, 0};
  PHX_CHECK(run(in).status == PHX_MC_CONSTRAINT_INTERSECTION);
}

void test_random_segments() {
  std::uint64_t seed = 99;
  Input in;
  const int n = 300;
  for (int i = 0; i < 2 * n; ++i) in.points.push_back(uniform(seed));
  // Greedily keep non-crossing random segments (exact proper-crossing test).
  auto crosses = [&](int a, int b, int c, int d) {
    if (a == c || a == d || b == c || b == d) return false;
    const double* p = in.points.data();
    return orient2d(p + 2 * a, p + 2 * b, p + 2 * c) * orient2d(p + 2 * a, p + 2 * b, p + 2 * d) < 0 &&
           orient2d(p + 2 * c, p + 2 * d, p + 2 * a) * orient2d(p + 2 * c, p + 2 * d, p + 2 * b) < 0;
  };
  for (int attempt = 0; attempt < 400; ++attempt) {
    const int a = static_cast<int>(uniform(seed) * n), b = static_cast<int>(uniform(seed) * n);
    if (a == b) continue;
    bool ok = true;
    for (std::size_t s = 0; s < in.segments.size() && ok; s += 2) {
      ok = !crosses(a, b, in.segments[s], in.segments[s + 1]) &&
           !((a == in.segments[s] && b == in.segments[s + 1]) ||
             (b == in.segments[s] && a == in.segments[s + 1]));
    }
    if (ok) {
      in.segments.push_back(a);
      in.segments.push_back(b);
    }
  }
  in.keep_hull = 1;
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_mesh(r);
  check_segments(r, in);
  check_constrained_delaunay(r);
  // Triangles cover the hull: T = 2 V - 2 - B with all points used.
  PHX_CHECK(r.cells.size() / 3 > static_cast<std::size_t>(n));
  const Result again = run(in);
  PHX_CHECK(again.cells == r.cells && again.segments == r.segments);
}

void test_lattice_segments() {
  // Cocircular lattice with diagonal segments through lattice points.
  Input in;
  for (int i = 0; i <= 8; ++i)
    for (int j = 0; j <= 8; ++j) {
      in.points.push_back(i);
      in.points.push_back(j);
    }
  auto id = [](int i, int j) { return 9 * i + j; };
  in.segments = {id(0, 0), id(8, 8), id(0, 8), id(3, 5), id(8, 0), id(5, 5), id(1, 0), id(5, 2)};
  in.keep_hull = 1;
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_mesh(r);
  check_segments(r, in);
  check_constrained_delaunay(r);
  PHX_CHECK_NEAR(total_area(r), 64.0, 1e-12);
  PHX_CHECK(r.cells.size() / 3 == 128);
}

void test_hole_carving() {
  Input in = square_with_hole();
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_mesh(r);
  check_segments(r, in);
  check_constrained_delaunay(r);
  PHX_CHECK_NEAR(total_area(r), 12.0, 1e-12);
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) {
    double cx = 0.0, cy = 0.0;
    for (int k = 0; k < 3; ++k) {
      cx += at(r, r.cells[3 * c + k])[0] / 3.0;
      cy += at(r, r.cells[3 * c + k])[1] / 3.0;
    }
    PHX_CHECK(!(cx > 1 && cx < 3 && cy > 1 && cy < 3));
  }
  // Without the hole seed the inner square is meshed; without segments and
  // without the hull flag everything is carved.
  in.holes.clear();
  PHX_CHECK_NEAR(total_area(run(in)), 16.0, 1e-12);
  in.segments.clear();
  const Result empty = run(in);
  PHX_CHECK(empty.status == PHX_MC_OK && empty.cells.empty());
  in.keep_hull = 1;
  const Result hull = run(in);
  PHX_CHECK_NEAR(total_area(hull), 16.0, 1e-12);
  PHX_CHECK(std::all_of(hull.segments.begin(), hull.segments.end(), [](int32_t s) { return s == -1; }));
}

void check_quality(const Result& r, double min_angle, double max_area) {
  for (std::size_t c = 0; c < r.cells.size() / 3; ++c) {
    PHX_CHECK(min_angle_degrees(r, c) >= min_angle - 1e-9);
    PHX_CHECK(0.5 * twice_area(r, c) <= max_area * (1 + 1e-12));
  }
}

void test_ruppert_square() {
  Input in;
  in.points = {0, 0, 1, 0, 1, 1, 0, 1};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 0};
  in.min_angle = 20.7;
  in.max_area = 0.01;
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_mesh(r);
  check_segments(r, in);
  check_constrained_delaunay(r);
  check_quality(r, 20.7, 0.01);
  PHX_CHECK_NEAR(total_area(r), 1.0, 1e-12);
  PHX_CHECK(r.points.size() / 2 > 4);
  for (int i = 0; i < 8; ++i) PHX_CHECK(r.points[i] == in.points[i]);
}

void test_ruppert_l_shape() {
  Input in;
  in.points = {0, 0, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0};
  in.min_angle = 20.7;
  in.max_area = 0.05;
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_mesh(r);
  check_segments(r, in);
  check_constrained_delaunay(r);
  check_quality(r, 20.7, 0.05);
  PHX_CHECK_NEAR(total_area(r), 3.0, 1e-12);
  // Angle-only refinement on the L-shape with a hole-free interior point.
  in.max_area = kInf;
  in.min_angle = 30.0;
  in.points.insert(in.points.end(), {0.3, 0.2});
  const Result angle_only = run(in);
  PHX_CHECK(angle_only.status == PHX_MC_OK);
  check_quality(angle_only, 30.0, kInf);
}

void test_ruppert_slanted_polygon() {
  // Non-axis-aligned boundary (inexact midpoints), convex hull and a hole.
  Input in;
  const int sides = 7;
  for (int k = 0; k < sides; ++k) {
    const double angle = 2.0 * 3.14159265358979323846 * k / sides + 0.1;
    in.points.push_back(3.0 * std::cos(angle));
    in.points.push_back(3.0 * std::sin(angle));
  }
  for (int k = 0; k < 3; ++k) {
    const double angle = 2.0 * 3.14159265358979323846 * k / 3 + 0.3;
    in.points.push_back(std::cos(angle));
    in.points.push_back(std::sin(angle));
  }
  for (int k = 0; k < sides; ++k) {
    in.segments.push_back(k);
    in.segments.push_back((k + 1) % sides);
  }
  for (int k = 0; k < 3; ++k) {
    in.segments.push_back(sides + k);
    in.segments.push_back(sides + (k + 1) % 3);
  }
  in.holes = {0.0, 0.0};
  in.min_angle = 25.0;
  in.max_area = 0.05;
  for (int keep_hull = 0; keep_hull < 2; ++keep_hull) {
    in.keep_hull = keep_hull;
    const Result r = run(in);
    PHX_CHECK(r.status == PHX_MC_OK);
    check_mesh(r);
    check_segments(r, in);
    check_constrained_delaunay(r);
    check_quality(r, 25.0, 0.05);
  }
}

void test_refinement_limit() {
  Input in;
  in.points = {0, 0, 1, 0, 1, 1, 0, 1};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 0};
  in.min_angle = 30.0;
  in.max_area = 0.001;
  in.max_steiner = 5;
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(r.points.size() / 2 == 4 + 5);
  check_mesh(r);
  check_segments(r, in);
  check_constrained_delaunay(r);
  PHX_CHECK_NEAR(total_area(r), 1.0, 1e-12);
  in.max_steiner = 0;
  const Result none = run(in);
  PHX_CHECK(none.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(none.cells.size() / 3 == 2);
  in.max_steiner = 1 << 20;
  in.max_triangles = 50;
  PHX_CHECK(run(in).status == PHX_MC_CAPACITY_EXCEEDED);
}

void test_arguments() {
  Input in = square_with_hole();
  Input bad = in;
  bad.min_angle = 60.0;
  PHX_CHECK(run(bad).status == PHX_MC_INVALID_ARGUMENT);
  bad = in;
  bad.min_angle = std::nan("");
  PHX_CHECK(run(bad).status == PHX_MC_INVALID_ARGUMENT);
  bad = in;
  bad.max_area = 0.0;
  PHX_CHECK(run(bad).status == PHX_MC_INVALID_ARGUMENT);
  bad = in;
  bad.max_steiner = -1;
  PHX_CHECK(run(bad).status == PHX_MC_INVALID_ARGUMENT);
  for (int64_t Input::* limit : {&Input::max_cavity_cells, &Input::max_work,
                                &Input::max_scratch_bytes}) {
    bad = in;
    bad.*limit = -1;
    PHX_CHECK(run(bad).status == PHX_MC_INVALID_ARGUMENT);
    bad.*limit = 0;
    PHX_CHECK(run(bad).status == PHX_MC_CAPACITY_EXCEEDED);
  }
  bad = in;
  bad.segments.push_back(0);
  bad.segments.push_back(10);
  PHX_CHECK(run(bad).status == PHX_MC_INVALID_INPUT);
  bad = in;
  bad.points.insert(bad.points.end(), {4, 4});
  bad.segments.push_back(2);
  bad.segments.push_back(10);
  PHX_CHECK(run(bad).status == PHX_MC_INVALID_INPUT);
  bad = in;
  bad.holes = {std::nan(""), 0};
  PHX_CHECK(run(bad).status == PHX_MC_NONFINITE_INPUT);
  bad = in;
  bad.points = {0, 0, 1, 1, 2, 2};
  bad.segments = {0, 2};
  bad.holes.clear();
  PHX_CHECK(run(bad).status == PHX_MC_DEGENERATE_INPUT);
  phx_mc_mesh* mesh = nullptr;
  PHX_CHECK(phx_mc_constrained_delaunay_2d(4, in.points.data(), 1, nullptr, 0, nullptr, 0, 0.0,
                                           kInf, 0, 100, INT64_MAX, INT64_MAX, INT64_MAX,
                                           nullptr, nullptr, &mesh) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_constrained_delaunay_2d(4, in.points.data(), 0, nullptr, 0, nullptr, 0, 0.0,
                                           kInf, 0, 100, INT64_MAX, INT64_MAX, INT64_MAX,
                                           nullptr, nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  // Duplicated points are merged; segments reference either copy.
  Input dup = in;
  dup.points.insert(dup.points.end(), {4, 0});
  dup.segments[2] = 10;  // segment 1 now starts at the duplicate of point 1
  const Result r = run(dup);
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.vertex_map[10] == 1);
  check_segments(r, dup);
  PHX_CHECK_NEAR(total_area(r), 12.0, 1e-12);
}

void test_native_limits_and_source_rollback() {
  Input in = square_with_hole();
  in.max_area = 0.75;
  const auto source_points = in.points;
  const auto source_segments = in.segments;
  const auto source_holes = in.holes;
  const Result baseline = run(in);
  PHX_CHECK(baseline.status == PHX_MC_OK);
  check_mesh(baseline);
  check_segments(baseline, in);
  check_constrained_delaunay(baseline);
  check_quality(baseline, 0.0, in.max_area);
  // Every possible work refusal, including pseudo-polygon planning, carving,
  // constrained cavity discovery and Lawson flips, publishes no partial mesh.
  for (uint64_t limit = 0; limit < baseline.work[0]; ++limit) {
    Input bounded = in;
    bounded.max_work = static_cast<int64_t>(limit);
    const Result refused = run(bounded);
    PHX_CHECK(refused.status == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(refused.work[0] <= limit);
    PHX_CHECK(refused.work[7] == 1);
    PHX_CHECK(refused.memory[1] == 0);
    PHX_CHECK(bounded.points == source_points);
    PHX_CHECK(bounded.segments == source_segments);
    PHX_CHECK(bounded.holes == source_holes);
  }
  Input bounded = in;
  bounded.max_work = static_cast<int64_t>(baseline.work[0]);
  bounded.max_cavity_cells = static_cast<int64_t>(baseline.work[6]);
  bounded.max_scratch_bytes = static_cast<int64_t>(baseline.memory[2]);
  const Result exact = run(bounded);
  PHX_CHECK(exact.status == PHX_MC_OK);
  PHX_CHECK(exact.points == baseline.points);
  PHX_CHECK(exact.cells == baseline.cells);
  PHX_CHECK(exact.segments == baseline.segments);
  PHX_CHECK(exact.vertex_map == baseline.vertex_map);
  // Nested callers keep the parent's single ledger and its hard envelope.
  // Per-call evidence must not repeat an unrelated historical pool peak.
  const auto owner = std::make_shared<BoundedMemoryResource>(
      static_cast<std::size_t>(4 * baseline.memory[2] + 1024));
  {
    MemoryScope scope(owner);
    {
      NativeVector<uint8_t> warm(static_cast<std::size_t>(3 * baseline.memory[2]), 1);
    }
    NativeVector<uint8_t> held(128, 7);
    const auto live_baseline = owner->live_bytes();
    const auto pool_peak = owner->peak_bytes();
    PHX_CHECK(owner->set_limit(live_baseline + baseline.memory[2]));
    const Result nested = run(in);
    PHX_CHECK(nested.status == PHX_MC_OK);
    PHX_CHECK(nested.cells == baseline.cells && nested.segments == baseline.segments);
    PHX_CHECK(nested.memory[0] == baseline.memory[2]);
    PHX_CHECK(nested.memory[2] == baseline.memory[2]);
    PHX_CHECK(nested.memory[2] < pool_peak);
    PHX_CHECK(owner->live_bytes() == live_baseline);
    PHX_CHECK(owner->limit_bytes() == live_baseline + baseline.memory[2]);
    PHX_CHECK(owner->set_limit(live_baseline + baseline.memory[2] - 1));
    const Result nested_refused = run(in);
    PHX_CHECK(nested_refused.status == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(nested_refused.memory[5] == 1);
    PHX_CHECK(nested_refused.memory[1] == 0);
    PHX_CHECK(owner->live_bytes() == live_baseline);
    PHX_CHECK(owner->limit_bytes() == live_baseline + baseline.memory[2] - 1);
    PHX_CHECK(std::all_of(held.begin(), held.end(), [](uint8_t value) { return value == 7; }));
  }
  bounded = in;
  bounded.max_scratch_bytes = static_cast<int64_t>(baseline.memory[2]) - 1;
  const Result memory_refused = run(bounded);
  PHX_CHECK(memory_refused.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(memory_refused.memory[5] == 1);
  PHX_CHECK(memory_refused.memory[2] <= uint64_t(bounded.max_scratch_bytes));
  PHX_CHECK(memory_refused.memory[1] == 0);
  bounded = in;
  bounded.max_cavity_cells = static_cast<int64_t>(baseline.work[6]) - 1;
  const Result cavity_refused = run(bounded);
  PHX_CHECK(cavity_refused.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(cavity_refused.work[8] == 1);
  PHX_CHECK(cavity_refused.memory[1] == 0);
  // A constraint-intersection refusal must not repair or mutate the PSLG.
  Input crossing;
  crossing.points = {0, 0, 2, 0, 2, 2, 0, 2};
  crossing.segments = {0, 2, 1, 3};
  crossing.keep_hull = 1;
  const auto crossing_points = crossing.points;
  const auto crossing_segments = crossing.segments;
  PHX_CHECK(run(crossing).status == PHX_MC_CONSTRAINT_INTERSECTION);
  PHX_CHECK(crossing.points == crossing_points);
  PHX_CHECK(crossing.segments == crossing_segments);
  crossing.segments.resize(2);
  const Result recovered = run(crossing);
  PHX_CHECK(recovered.status == PHX_MC_OK);
  check_mesh(recovered);
  check_segments(recovered, crossing);
  check_constrained_delaunay(recovered);
  const Result again = run(in);
  PHX_CHECK(again.points == baseline.points && again.cells == baseline.cells &&
            again.segments == baseline.segments && again.vertex_map == baseline.vertex_map);
}

}  // namespace

int main() {
  test_segments_and_vertex_on_segment();
  test_random_segments();
  test_lattice_segments();
  test_hole_carving();
  test_ruppert_square();
  test_ruppert_l_shape();
  test_ruppert_slanted_polygon();
  test_refinement_limit();
  test_arguments();
  test_native_limits_and_source_rollback();
  return phx::mc::test::finish("test_cdt2d");
}
