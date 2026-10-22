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
  double max_edge_length = kInf;
  double lattice_spacing = 0.0;
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
  std::vector<int32_t> witness_parents;
  std::vector<int32_t> witness_segments;
  std::vector<double> witness_ratios;
  std::array<uint64_t, 9> work{};
  std::array<uint64_t, 6> memory{};
};

Result run(const Input& in, bool request_witnesses = false) {
  phx_mc_mesh* mesh = nullptr;
  Result r;
  const int64_t witness_capacity = request_witnesses
      ? static_cast<int64_t>(in.points.size() / 2) + in.max_steiner : 0;
  if (request_witnesses) {
    // The extra row checks that native publication writes only returned rows.
    r.witness_parents.assign(static_cast<std::size_t>(2 * (witness_capacity + 1)), -17);
    r.witness_segments.assign(static_cast<std::size_t>(witness_capacity + 1), -19);
    r.witness_ratios.assign(static_cast<std::size_t>(witness_capacity + 1), -23.0);
  }
  const phx_mc_constrained_delaunay_2d_request request{
      .point_count = static_cast<int64_t>(in.points.size() / 2),
      .points = in.points.data(),
      .segment_count = static_cast<int64_t>(in.segments.size() / 2),
      .segments = in.segments.data(),
      .hole_count = static_cast<int64_t>(in.holes.size() / 2),
      .holes = in.holes.data(),
      .keep_convex_hull = in.keep_hull,
      .min_angle_degrees = in.min_angle,
      .max_area = in.max_area,
      .max_edge_length = in.max_edge_length,
      .lattice_spacing = in.lattice_spacing,
      .max_steiner = in.max_steiner,
      .max_triangles = in.max_triangles,
      .max_cavity_cells = in.max_cavity_cells,
      .max_work = in.max_work,
      .max_scratch_bytes = in.max_scratch_bytes,
      .work_evidence = r.work.data(),
      .memory_evidence = r.memory.data(),
      .witness_capacity = witness_capacity,
      .witness_parents = request_witnesses ? r.witness_parents.data() : nullptr,
      .witness_segments = request_witnesses ? r.witness_segments.data() : nullptr,
      .witness_ratios = request_witnesses ? r.witness_ratios.data() : nullptr,
      .mesh = &mesh,
  };
  r.status = phx_mc_constrained_delaunay_2d(&request);
  const bool has_mesh = r.status == PHX_MC_OK || r.status == PHX_MC_REFINEMENT_LIMIT;
  PHX_CHECK(has_mesh == (mesh != nullptr));
  if (mesh == nullptr) {
    PHX_CHECK(std::all_of(r.witness_parents.begin(), r.witness_parents.end(),
                         [](int32_t v) { return v == -17; }));
    PHX_CHECK(std::all_of(r.witness_segments.begin(), r.witness_segments.end(),
                         [](int32_t s) { return s == -19; }));
    PHX_CHECK(std::all_of(r.witness_ratios.begin(), r.witness_ratios.end(),
                         [](double r) { return r == -23.0; }));
    return r;
  }
  r.points.resize(static_cast<std::size_t>(2 * phx_mc_mesh_point_count(mesh)));
  if (request_witnesses) {
    const std::size_t count = r.points.size() / 2;
    PHX_CHECK(std::all_of(r.witness_parents.begin() + 2 * count, r.witness_parents.end(),
                         [](int32_t v) { return v == -17; }));
    PHX_CHECK(std::all_of(r.witness_segments.begin() + count, r.witness_segments.end(),
                         [](int32_t s) { return s == -19; }));
    PHX_CHECK(std::all_of(r.witness_ratios.begin() + count, r.witness_ratios.end(),
                         [](double r) { return r == -23.0; }));
    r.witness_parents.resize(2 * count);
    r.witness_segments.resize(count);
    r.witness_ratios.resize(count);
  }
  r.cells.resize(static_cast<std::size_t>(3 * phx_mc_mesh_cell_count(mesh)));
  r.segments.resize(r.cells.size());
  r.vertex_map.resize(static_cast<std::size_t>(phx_mc_mesh_input_point_count(mesh)));
  PHX_CHECK(phx_mc_mesh_copy_points(mesh, r.points.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_cells(mesh, r.cells.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_cell_constraints(mesh, r.segments.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_vertex_map(mesh, r.vertex_map.data()) == PHX_MC_OK);
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

// Circumcenters replace their gathered constrained cavity by a fan; the
// result must stay a valid, deterministic constrained Delaunay refinement on
// cocircular lattices with a reflex corner, a hole and dangling interior
// segments (cavities that must fall back to Lawson insertion).
void test_ruppert_cavity_fan() {
  Input in;
  in.points = {0, 0, 4, 0, 4, 2, 2, 2, 2, 4, 0, 4,          // L-shaped outer boundary
               0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5,      // square hole
               3, 0.5, 3, 1.5, 0.5, 3, 1.5, 3.5};           // two dangling segments
  in.segments = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 6, 7, 7, 8, 8, 9, 9, 6, 10, 11, 12, 13};
  in.holes = {1, 1};
  for (int i = 0; i <= 8; ++i) {
    for (int j = 0; j <= 8; ++j) {
      const double x = 0.25 + 0.25 * i, y = 2.25 + 0.25 * j;
      if (x < 2 && y < 4) in.points.insert(in.points.end(), {x, y});
    }
  }
  in.min_angle = 20.7;
  in.max_area = 0.02;
  in.max_edge_length = 0.3;
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_mesh(r);
  check_segments(r, in);
  check_constrained_delaunay(r);
  check_quality(r, 20.7, 0.02);
  PHX_CHECK_NEAR(total_area(r), 12.0 - 1.0, 1e-12);
  // Most Steiner points are circumcenters inserted without flips; flips only
  // legalize subsegment splits and fallback cavities.
  const uint64_t steiner = r.work[4] - static_cast<uint64_t>(in.points.size() / 2);
  PHX_CHECK(steiner > 50);
  PHX_CHECK(r.work[5] < steiner);
  const Result again = run(in);
  PHX_CHECK(again.points == r.points && again.cells == r.cells && again.segments == r.segments);
  PHX_CHECK(again.work == r.work);
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

// Unconstrained edges obey the binary64 length bound in one call; segment
// pieces keep their own splitting rules and a disabled bound repeats exactly.
void test_ruppert_max_edge_length() {
  Input in;
  in.points = {0, 0, 2, 0, 2, 2, 0, 2, 0.8, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.2, 0.2, 0.5, 1.7};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 8, 9};
  in.holes = {1.0, 1.0};
  in.min_angle = 25.0;
  const double h = 0.2;
  in.max_area = 0.25 * std::sqrt(3.0) * h * h;
  const Result area_only = run(in);
  PHX_CHECK(area_only.status == PHX_MC_OK);
  in.max_edge_length = kInf;
  const Result disabled = run(in);
  PHX_CHECK(disabled.status == area_only.status);
  PHX_CHECK(disabled.points == area_only.points);
  PHX_CHECK(disabled.cells == area_only.cells);
  PHX_CHECK(disabled.work == area_only.work);
  in.max_edge_length = h;
  const Result bounded = run(in, true);
  PHX_CHECK(bounded.status == PHX_MC_OK);
  check_mesh(bounded);
  check_segments(bounded, in);
  check_constrained_delaunay(bounded);
  check_quality(bounded, 25.0, in.max_area);
  PHX_CHECK_NEAR(total_area(bounded), 4.0 - 0.16, 1e-12);
  for (std::size_t c = 0; c < bounded.cells.size() / 3; ++c) {
    for (int k = 0; k < 3; ++k) {
      if (bounded.segments[3 * c + k] >= 0) continue;
      const double* a = at(bounded, bounded.cells[3 * c + (k + 1) % 3]);
      const double* b = at(bounded, bounded.cells[3 * c + (k + 2) % 3]);
      const double dx = b[0] - a[0];
      const double dy = b[1] - a[1];
      PHX_CHECK(std::sqrt(dx * dx + dy * dy) <= h);
    }
  }
  PHX_CHECK(bounded.points.size() > area_only.points.size());
  // The same request with an insufficient Steiner allowance is a valid mesh
  // with unmet quality, never an unbounded refinement.
  in.max_steiner = 3;
  const Result limited = run(in);
  PHX_CHECK(limited.status == PHX_MC_REFINEMENT_LIMIT);
  check_mesh(limited);
  check_segments(limited, in);
}

// Splits every polygon/embedded segment of `in` into ceil(length / spacing)
// equal pieces, as the planar route prepares its constraint complex.
Input split_constraints(const Input& in, double spacing) {
  Input out = in;
  out.segments.clear();
  for (std::size_t s = 0; s < in.segments.size() / 2; ++s) {
    const int32_t a = in.segments[2 * s];
    const int32_t b = in.segments[2 * s + 1];
    const double ax = in.points[2 * a], ay = in.points[2 * a + 1];
    const double dx = in.points[2 * b] - ax, dy = in.points[2 * b + 1] - ay;
    const int pieces = static_cast<int>(std::ceil(std::hypot(dx, dy) / spacing));
    int32_t previous = a;
    for (int k = 1; k <= pieces; ++k) {
      int32_t next = b;
      if (k < pieces) {
        next = static_cast<int32_t>(out.points.size() / 2);
        const double t = static_cast<double>(k) / pieces;
        out.points.insert(out.points.end(), {ax + t * dx, ay + t * dy});
      }
      out.segments.insert(out.segments.end(), {previous, next});
      previous = next;
    }
  }
  return out;
}

// Interior lattice seeding of the plate with a hole and an embedded segment
// (constraints split at the lattice spacing) meets the same exact targets as
// unseeded refinement with fewer cells and less work, repeats exactly, and
// keeps the disabled request bit-identical.
void test_lattice_seeding() {
  Input source;
  source.points = {0, 0, 2, 0, 2, 2, 0, 2, 0.8, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.2, 0.2,
                   0.5, 1.7};
  source.segments = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 8, 9};
  const double bound = 0.15;
  Input in = split_constraints(source, 0.9 * bound);
  in.holes = {1.0, 1.0};
  in.min_angle = 25.0;
  in.max_area = 0.25 * std::sqrt(3.0) * bound * bound;
  in.max_edge_length = bound;
  const Result plain = run(in, true);
  PHX_CHECK(plain.status == PHX_MC_OK);
  in.lattice_spacing = 0.0;
  const Result disabled = run(in, true);
  PHX_CHECK(disabled.points == plain.points && disabled.cells == plain.cells);
  PHX_CHECK(disabled.work == plain.work);
  in.lattice_spacing = 0.9 * bound;
  const Result seeded = run(in, true);
  PHX_CHECK(seeded.status == PHX_MC_OK);
  check_mesh(seeded);
  check_segments(seeded, in);
  check_constrained_delaunay(seeded);
  check_quality(seeded, 25.0, in.max_area);
  PHX_CHECK_NEAR(total_area(seeded), 4.0 - 0.16, 1e-12);
  for (std::size_t c = 0; c < seeded.cells.size() / 3; ++c) {
    for (int k = 0; k < 3; ++k) {
      if (seeded.segments[3 * c + k] >= 0) continue;
      const double* a = at(seeded, seeded.cells[3 * c + (k + 1) % 3]);
      const double* b = at(seeded, seeded.cells[3 * c + (k + 2) % 3]);
      const double dx = b[0] - a[0];
      const double dy = b[1] - a[1];
      PHX_CHECK(std::sqrt(dx * dx + dy * dy) <= bound);
    }
  }
  PHX_CHECK(seeded.cells.size() < plain.cells.size());
  PHX_CHECK(seeded.work[PHX_MC_PLANAR_WORK_UNITS] < plain.work[PHX_MC_PLANAR_WORK_UNITS]);
  // Accepted lattice sites are free Steiner points at exact lattice positions.
  const double rise = in.lattice_spacing * (0.5 * std::sqrt(3.0));
  std::size_t sites = 0;
  for (std::size_t v = in.points.size() / 2; v < seeded.points.size() / 2; ++v) {
    const double j = std::round(seeded.points[2 * v + 1] / rise);
    const double i = std::round(seeded.points[2 * v] / in.lattice_spacing - 0.5 * j);
    if (seeded.points[2 * v] == (i + 0.5 * j) * in.lattice_spacing &&
        seeded.points[2 * v + 1] == j * rise) {
      ++sites;
      PHX_CHECK(seeded.witness_segments[v] == -1);
      PHX_CHECK(seeded.witness_parents[2 * v] == -1 && seeded.witness_parents[2 * v + 1] == -1);
    }
  }
  // At least a quarter of all vertices are accepted lattice sites.
  PHX_CHECK(4 * sites > seeded.points.size() / 2);
  const Result again = run(in, true);
  PHX_CHECK(again.points == seeded.points && again.cells == seeded.cells);
  PHX_CHECK(again.segments == seeded.segments && again.work == seeded.work);
  // Seeds share the Steiner allowance: a valid mesh with unmet targets.
  Input limited = in;
  limited.max_steiner = 5;
  const Result short_of_steiner = run(limited);
  PHX_CHECK(short_of_steiner.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(short_of_steiner.points.size() / 2 <= in.points.size() / 2 + 5);
  check_mesh(short_of_steiner);
  check_segments(short_of_steiner, in);
  // Every work refusal of a seeded request publishes nothing.
  Input unit;
  unit.points = {0, 0, 1, 0, 1, 1, 0, 1};
  unit.segments = {0, 1, 1, 2, 2, 3, 3, 0};
  Input square = split_constraints(unit, 0.27);
  square.min_angle = 25.0;
  square.max_edge_length = 0.3;
  square.lattice_spacing = 0.27;
  const Result baseline = run(square);
  PHX_CHECK(baseline.status == PHX_MC_OK);
  check_mesh(baseline);
  check_constrained_delaunay(baseline);
  for (uint64_t limit = 0; limit < baseline.work[PHX_MC_PLANAR_WORK_UNITS]; ++limit) {
    Input bounded = square;
    bounded.max_work = static_cast<int64_t>(limit);
    const Result refused = run(bounded);
    PHX_CHECK(refused.status == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(refused.work[PHX_MC_PLANAR_WORK_UNITS] <= limit);
    PHX_CHECK(refused.memory[PHX_MC_MEMORY_LIVE_BYTES] == 0);
  }
  Input exact = square;
  exact.max_work = static_cast<int64_t>(baseline.work[PHX_MC_PLANAR_WORK_UNITS]);
  const Result admitted = run(exact);
  PHX_CHECK(admitted.status == PHX_MC_OK);
  PHX_CHECK(admitted.points == baseline.points && admitted.cells == baseline.cells);
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
  for (const double length : {0.0, -1.0, std::nan("")}) {
    bad = in;
    bad.max_edge_length = length;
    PHX_CHECK(run(bad).status == PHX_MC_INVALID_ARGUMENT);
  }
  for (const double spacing : {-1.0, kInf, std::nan("")}) {
    bad = in;
    bad.lattice_spacing = spacing;
    PHX_CHECK(run(bad).status == PHX_MC_INVALID_ARGUMENT);
  }
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
  const phx_mc_constrained_delaunay_2d_request missing_segments_request{
      .point_count = 4,
      .points = in.points.data(),
      .segment_count = 1,
      .segments = nullptr,
      .hole_count = 0,
      .holes = nullptr,
      .keep_convex_hull = 0,
      .min_angle_degrees = 0.0,
      .max_area = kInf,
      .max_edge_length = kInf,
      .max_steiner = 0,
      .max_triangles = 100,
      .max_cavity_cells = INT64_MAX,
      .max_work = INT64_MAX,
      .max_scratch_bytes = INT64_MAX,
      .work_evidence = nullptr,
      .memory_evidence = nullptr,
      .witness_capacity = 0,
      .witness_parents = nullptr,
      .witness_segments = nullptr,
      .witness_ratios = nullptr,
      .mesh = &mesh,
  };
  PHX_CHECK(phx_mc_constrained_delaunay_2d(&missing_segments_request) == PHX_MC_INVALID_ARGUMENT);
  const phx_mc_constrained_delaunay_2d_request missing_output_request{
      .point_count = 4,
      .points = in.points.data(),
      .segment_count = 0,
      .segments = nullptr,
      .hole_count = 0,
      .holes = nullptr,
      .keep_convex_hull = 0,
      .min_angle_degrees = 0.0,
      .max_area = kInf,
      .max_edge_length = kInf,
      .max_steiner = 0,
      .max_triangles = 100,
      .max_cavity_cells = INT64_MAX,
      .max_work = INT64_MAX,
      .max_scratch_bytes = INT64_MAX,
      .work_evidence = nullptr,
      .memory_evidence = nullptr,
      .witness_capacity = 0,
      .witness_parents = nullptr,
      .witness_segments = nullptr,
      .witness_ratios = nullptr,
      .mesh = nullptr,
  };
  PHX_CHECK(phx_mc_constrained_delaunay_2d(&missing_output_request) == PHX_MC_INVALID_ARGUMENT);
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
  for (uint64_t limit = 0; limit < baseline.work[PHX_MC_PLANAR_WORK_UNITS]; ++limit) {
    Input bounded = in;
    bounded.max_work = static_cast<int64_t>(limit);
    const Result refused = run(bounded);
    PHX_CHECK(refused.status == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(refused.work[PHX_MC_PLANAR_WORK_UNITS] <= limit);
    PHX_CHECK(refused.work[PHX_MC_PLANAR_WORK_REFUSALS] == 1);
    PHX_CHECK(refused.memory[PHX_MC_MEMORY_LIVE_BYTES] == 0);
    PHX_CHECK(bounded.points == source_points);
    PHX_CHECK(bounded.segments == source_segments);
    PHX_CHECK(bounded.holes == source_holes);
  }
  Input bounded = in;
  bounded.max_work = static_cast<int64_t>(baseline.work[PHX_MC_PLANAR_WORK_UNITS]);
  bounded.max_cavity_cells = static_cast<int64_t>(baseline.work[PHX_MC_PLANAR_PEAK_CAVITY_CELLS]);
  bounded.max_scratch_bytes = static_cast<int64_t>(baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]);
  const Result exact = run(bounded);
  PHX_CHECK(exact.status == PHX_MC_OK);
  PHX_CHECK(exact.points == baseline.points);
  PHX_CHECK(exact.cells == baseline.cells);
  PHX_CHECK(exact.segments == baseline.segments);
  PHX_CHECK(exact.vertex_map == baseline.vertex_map);
  // Nested callers keep the parent's single ledger and its hard envelope.
  // Per-call evidence must not repeat an unrelated historical pool peak.
  const auto owner = std::make_shared<BoundedMemoryResource>(
      static_cast<std::size_t>(4 * baseline.memory[PHX_MC_MEMORY_PEAK_BYTES] + 1024));
  {
    MemoryScope scope(owner);
    {
      NativeVector<uint8_t> warm(static_cast<std::size_t>(3 * baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]), 1);
    }
    NativeVector<uint8_t> held(128, 7);
    const auto live_baseline = owner->live_bytes();
    const auto pool_peak = owner->peak_bytes();
    PHX_CHECK(owner->set_limit(live_baseline + baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]));
    const Result nested = run(in);
    PHX_CHECK(nested.status == PHX_MC_OK);
    PHX_CHECK(nested.cells == baseline.cells && nested.segments == baseline.segments);
    PHX_CHECK(nested.memory[PHX_MC_MEMORY_LIMIT_BYTES] == baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]);
    PHX_CHECK(nested.memory[PHX_MC_MEMORY_PEAK_BYTES] == baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]);
    PHX_CHECK(nested.memory[PHX_MC_MEMORY_PEAK_BYTES] < pool_peak);
    PHX_CHECK(owner->live_bytes() == live_baseline);
    PHX_CHECK(owner->limit_bytes() == live_baseline + baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]);
    PHX_CHECK(owner->set_limit(live_baseline + baseline.memory[PHX_MC_MEMORY_PEAK_BYTES] - 1));
    const Result nested_refused = run(in);
    PHX_CHECK(nested_refused.status == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(nested_refused.memory[PHX_MC_MEMORY_REFUSALS] == 1);
    PHX_CHECK(nested_refused.memory[PHX_MC_MEMORY_LIVE_BYTES] == 0);
    PHX_CHECK(owner->live_bytes() == live_baseline);
    PHX_CHECK(owner->limit_bytes() == live_baseline + baseline.memory[PHX_MC_MEMORY_PEAK_BYTES] - 1);
    PHX_CHECK(std::all_of(held.begin(), held.end(), [](uint8_t value) { return value == 7; }));
  }
  bounded = in;
  bounded.max_scratch_bytes = static_cast<int64_t>(baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]) - 1;
  const Result memory_refused = run(bounded);
  PHX_CHECK(memory_refused.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(memory_refused.memory[PHX_MC_MEMORY_REFUSALS] == 1);
  PHX_CHECK(memory_refused.memory[PHX_MC_MEMORY_PEAK_BYTES] <= uint64_t(bounded.max_scratch_bytes));
  PHX_CHECK(memory_refused.memory[PHX_MC_MEMORY_LIVE_BYTES] == 0);
  bounded = in;
  bounded.max_cavity_cells = static_cast<int64_t>(baseline.work[PHX_MC_PLANAR_PEAK_CAVITY_CELLS]) - 1;
  const Result cavity_refused = run(bounded);
  PHX_CHECK(cavity_refused.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(cavity_refused.work[PHX_MC_PLANAR_CAVITY_REFUSALS] == 1);
  PHX_CHECK(cavity_refused.memory[PHX_MC_MEMORY_LIVE_BYTES] == 0);
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

void test_split_witness_dag() {
  Input in;
  in.points = {0, 0, 2, 1, 1, 3, -1, 2};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 0};
  in.max_area = 0.025;
  in.max_steiner = 1024;
  const Result r = run(in, true);
  PHX_CHECK(r.status == PHX_MC_OK);
  const Result without = run(in);
  PHX_CHECK(r.points == without.points && r.cells == without.cells &&
            r.segments == without.segments && r.vertex_map == without.vertex_map);
  const std::size_t input_count = in.points.size() / 2;
  std::size_t constrained = 0, free = 0, nested = 0, shells = 0;
  bool rounded_complement = false;
  for (std::size_t v = 0; v < r.points.size() / 2; ++v) {
    const int32_t a = r.witness_parents[2 * v];
    const int32_t b = r.witness_parents[2 * v + 1];
    const int32_t s = r.witness_segments[v];
    const double ratio = r.witness_ratios[v];
    if (v < input_count || s == -1) {
      PHX_CHECK(a == -1 && b == -1 && s == -1 && ratio == 0.0);
      if (v >= input_count) ++free;
      continue;
    }
    ++constrained;
    PHX_CHECK(a >= 0 && b >= 0 && a != b);
    PHX_CHECK(static_cast<std::size_t>(a) < v && static_cast<std::size_t>(b) < v);
    PHX_CHECK(s >= 0 && static_cast<std::size_t>(s) < in.segments.size() / 2);
    PHX_CHECK(ratio > 0.0 && ratio < 1.0);
    for (int32_t parent : {a, b}) {
      if (static_cast<std::size_t>(parent) < input_count) {
        PHX_CHECK(parent == in.segments[2 * s] || parent == in.segments[2 * s + 1]);
      } else {
        PHX_CHECK(r.witness_segments[static_cast<std::size_t>(parent)] == s);
        ++nested;
      }
    }
    const bool input_a = static_cast<std::size_t>(a) < input_count;
    const bool input_b = static_cast<std::size_t>(b) < input_count;
    if (input_a != input_b) {
      ++shells;
      // Both input corners have degree two, so the shell parent must be first.
      PHX_CHECK(input_a);
      const double* pa = at(r, a);
      const double* pb = at(r, b);
      const double length = std::hypot(pb[0] - pa[0], pb[1] - pa[1]);
      int exponent = 0;
      const double fraction = std::frexp(0.5 * length, &exponent);
      const double expected = std::ldexp(1.0, fraction < 0.75 ? exponent - 1 : exponent) / length;
      PHX_CHECK(ratio == expected);
      rounded_complement = rounded_complement || (1.0 - (1.0 - ratio) != ratio);
    } else {
      PHX_CHECK(ratio == 0.5);
    }
  }
  PHX_CHECK(constrained > 0 && free > 0 && nested > 0 && shells > 0);
  PHX_CHECK(rounded_complement);
}

void test_hull_split_witnesses() {
  Input in;
  in.points = {0, 0, 1, 0, 1, 1, 0, 1};
  in.keep_hull = 1;
  in.max_area = 0.025;
  in.max_steiner = 256;
  const Result r = run(in, true);
  PHX_CHECK(r.status == PHX_MC_OK);
  bool has_hull_split = false;
  for (std::size_t v = 4; v < r.points.size() / 2; ++v) {
    if (r.witness_segments[v] == -1) {
      PHX_CHECK(r.witness_parents[2 * v] == -1 &&
                r.witness_parents[2 * v + 1] == -1 && r.witness_ratios[v] == 0.0);
      continue;
    }
    has_hull_split = true;
    PHX_CHECK(r.witness_segments[v] == -2);
    PHX_CHECK(r.witness_parents[2 * v] >= 0 && r.witness_parents[2 * v + 1] >= 0);
    PHX_CHECK(static_cast<std::size_t>(r.witness_parents[2 * v]) < v &&
              static_cast<std::size_t>(r.witness_parents[2 * v + 1]) < v);
    PHX_CHECK(r.witness_ratios[v] == 0.5);
  }
  PHX_CHECK(has_hull_split);
}

void test_witness_capacity_and_atomic_publication() {
  Input in;
  in.points = {0, 0, 1, 0, 1, 1, 0, 1};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 0};
  in.max_area = 0.001;
  in.max_steiner = 12;
  const Result baseline = run(in, true);
  PHX_CHECK(baseline.status == PHX_MC_REFINEMENT_LIMIT);
  PHX_CHECK(baseline.witness_segments.size() == 16);
  Input refused = in;
  // One fewer unit reaches the last all-or-nothing publication charge.
  refused.max_work = static_cast<int64_t>(baseline.work[PHX_MC_PLANAR_WORK_UNITS]) - 1;
  PHX_CHECK(run(refused, true).status == PHX_MC_CAPACITY_EXCEEDED);
  refused = in;
  refused.max_scratch_bytes = static_cast<int64_t>(baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]) - 1;
  PHX_CHECK(run(refused, true).status == PHX_MC_CAPACITY_EXCEEDED);
  refused = in;
  refused.segments = {0, 2, 1, 3};
  PHX_CHECK(run(refused, true).status == PHX_MC_CONSTRAINT_INTERSECTION);
  std::array<int32_t, 32> parents;
  std::array<int32_t, 16> segments;
  std::array<double, 16> ratios;
  parents.fill(-17);
  segments.fill(-19);
  ratios.fill(-23.0);
  auto call = [&](int64_t capacity, int32_t* p, int32_t* s, double* r) {
    phx_mc_mesh* mesh = nullptr;
    const phx_mc_constrained_delaunay_2d_request request{
        .point_count = 4,
        .points = in.points.data(),
        .segment_count = 4,
        .segments = in.segments.data(),
        .hole_count = 0,
        .holes = nullptr,
        .keep_convex_hull = 0,
        .min_angle_degrees = 0.0,
        .max_area = in.max_area,
        .max_edge_length = in.max_edge_length,
        .max_steiner = in.max_steiner,
        .max_triangles = in.max_triangles,
        .max_cavity_cells = INT64_MAX,
        .max_work = INT64_MAX,
        .max_scratch_bytes = INT64_MAX,
        .work_evidence = nullptr,
        .memory_evidence = nullptr,
        .witness_capacity = capacity,
        .witness_parents = p,
        .witness_segments = s,
        .witness_ratios = r,
        .mesh = &mesh,
    };
    const int32_t status = phx_mc_constrained_delaunay_2d(&request);
    PHX_CHECK(mesh == nullptr);
    PHX_CHECK(std::all_of(parents.begin(), parents.end(), [](int32_t v) { return v == -17; }));
    PHX_CHECK(std::all_of(segments.begin(), segments.end(), [](int32_t v) { return v == -19; }));
    PHX_CHECK(std::all_of(ratios.begin(), ratios.end(), [](double v) { return v == -23.0; }));
    return status;
  };
  PHX_CHECK(call(-1, parents.data(), segments.data(), ratios.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(16, nullptr, segments.data(), ratios.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(16, parents.data(), nullptr, ratios.data()) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(16, parents.data(), segments.data(), nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(16, nullptr, nullptr, nullptr) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(INT64_MAX, parents.data(), segments.data(), ratios.data()) ==
            PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(call(0, parents.data(), segments.data(), ratios.data()) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(call(3, parents.data(), segments.data(), ratios.data()) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(call(15, parents.data(), segments.data(), ratios.data()) == PHX_MC_CAPACITY_EXCEEDED);
  in.max_steiner = INT64_MAX;
  PHX_CHECK(call(16, parents.data(), segments.data(), ratios.data()) == PHX_MC_CAPACITY_EXCEEDED);
}

std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> finish_execution(
    void* scope, int32_t expected) {
  std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> work{};
  std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory{};
  double elapsed = -1;
  PHX_CHECK(phx_mc_execution_end(scope, work.data(), memory.data(), &elapsed) == expected);
  PHX_CHECK(memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
  return work;
}

void test_cumulative_execution_work_and_local_prefixes() {
  Input in = square_with_hole();
  in.max_area = 0.75;
  const Result baseline = run(in);
  PHX_CHECK(baseline.status == PHX_MC_OK);
  const uint64_t required = baseline.work[PHX_MC_PLANAR_WORK_UNITS];
  PHX_CHECK(required > 1);
  const auto points = in.points;
  const auto segments = in.segments;

  // A second construction draws from the same root, not a renewed local cap.
  void* scope = nullptr;
  PHX_CHECK(phx_mc_execution_begin(2 * required - 1, 0, 1 << 20, 1 << 24,
                                   INFINITY, nullptr, &scope) == PHX_MC_OK);
  const Result first = run(in);
  const Result second = run(in);
  PHX_CHECK(first.status == PHX_MC_OK);
  PHX_CHECK(second.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(first.work == baseline.work);
  PHX_CHECK(second.work[PHX_MC_PLANAR_WORK_REFUSALS] == 1);
  const uint64_t earned = first.work[PHX_MC_PLANAR_WORK_UNITS] +
                          second.work[PHX_MC_PLANAR_WORK_UNITS];
  const auto global = finish_execution(scope, PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(earned <= 2 * required - 1);
  PHX_CHECK(global[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == earned);
  PHX_CHECK(global[PHX_MC_EXECUTION_EXTERNAL_WORK] == 0);
  PHX_CHECK(global[PHX_MC_EXECUTION_WORK_REFUSALS] == 1);
  PHX_CHECK(in.points == points && in.segments == segments);

  // A local refusal does not debit the globally available but unexecuted unit.
  in.max_work = static_cast<int64_t>(required - 1);
  PHX_CHECK(phx_mc_execution_begin(required, 0, 1 << 20, 1 << 24,
                                   INFINITY, nullptr, &scope) == PHX_MC_OK);
  const Result local = run(in);
  PHX_CHECK(local.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(local.work[PHX_MC_PLANAR_WORK_REFUSALS] == 1);
  const auto local_global = finish_execution(scope, PHX_MC_OK);
  PHX_CHECK(local_global[PHX_MC_EXECUTION_CONSTRUCTION_WORK] ==
            local.work[PHX_MC_PLANAR_WORK_UNITS]);
  PHX_CHECK(local_global[PHX_MC_EXECUTION_WORK_REFUSALS] == 0);
  PHX_CHECK(local_global[PHX_MC_EXECUTION_CONSTRUCTION_WORK] <= required - 1);
  PHX_CHECK(in.points == points && in.segments == segments);
}

void test_cdt_work_uses_reserved_native_target_once() {
  Input in = square_with_hole();
  in.max_area = 0.75;
  const Result baseline = run(in);
  PHX_CHECK(baseline.status == PHX_MC_OK);
  const uint64_t required = baseline.work[PHX_MC_PLANAR_WORK_UNITS];
  void* scope = nullptr;
  PHX_CHECK(phx_mc_execution_begin(required, 0, 1 << 20, 1 << 24,
                                   INFINITY, nullptr, &scope) == PHX_MC_OK);
  void* ticket = nullptr;
  PHX_CHECK(phx_mc_execution_batch_admit(scope, nullptr, required, 0,
                                        1 << 20, 1 << 24, &ticket) == PHX_MC_OK);
  std::array<uint64_t, 4> remaining{};
  double seconds = -1;
  PHX_CHECK(phx_mc_execution_remaining(scope, remaining.data(), &seconds) == PHX_MC_OK);
  PHX_CHECK(remaining[0] == 0);
  void* target = nullptr;
  phx_mc_execution_usage* usage = nullptr;
  PHX_CHECK(phx_mc_execution_target_begin(ticket, &target, &usage) == PHX_MC_OK);
  const Result result = run(in);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.work == baseline.work);
  PHX_CHECK(usage->native_work == required && usage->host_work == 0);
  PHX_CHECK(phx_mc_execution_target_end(target) == PHX_MC_OK);
  PHX_CHECK(phx_mc_execution_batch_settle(ticket, 0, 0, 0, 0) == PHX_MC_OK);
  const auto global = finish_execution(scope, PHX_MC_OK);
  PHX_CHECK(global[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == required);
  PHX_CHECK(global[PHX_MC_EXECUTION_EXTERNAL_WORK] == 0);
  PHX_CHECK(global[PHX_MC_EXECUTION_WORK_REFUSALS] == 0);
  PHX_CHECK(result.points == baseline.points && result.cells == baseline.cells);
}

void test_quality_cache_current_geometry_controls_and_refusal_boundaries() {
  Input defaults;
  defaults.points = {0, 0, 1, 0, 1, 1, 0, 1};
  defaults.segments = {0, 1, 1, 2, 2, 3, 3, 0};
  defaults.max_steiner = 256;
  const Result unrefined = run(defaults, true);
  PHX_CHECK(unrefined.status == PHX_MC_OK);
  PHX_CHECK(unrefined.cells.size() == 6);

  Input strict = defaults;
  strict.min_angle = 25.0;
  strict.max_area = 0.01;
  strict.max_edge_length = 0.2;
  const auto source_points = strict.points;
  const auto source_segments = strict.segments;
  const Result baseline = run(strict, true);
  PHX_CHECK(baseline.status == PHX_MC_OK);
  PHX_CHECK(baseline.points.size() > unrefined.points.size());
  check_mesh(baseline);
  check_segments(baseline, strict);
  check_constrained_delaunay(baseline);
  check_quality(baseline, strict.min_angle, strict.max_area);
  for (std::size_t cell = 0; cell < baseline.cells.size() / 3; ++cell) {
    for (int edge = 0; edge < 3; ++edge) {
      if (baseline.segments[3 * cell + edge] >= 0) continue;
      const int32_t a = baseline.cells[3 * cell + (edge + 1) % 3];
      const int32_t b = baseline.cells[3 * cell + (edge + 2) % 3];
      const double dx = baseline.points[2 * a] - baseline.points[2 * b];
      const double dy = baseline.points[2 * a + 1] - baseline.points[2 * b + 1];
      PHX_CHECK(std::sqrt(dx * dx + dy * dy) <= strict.max_edge_length);
    }
  }
  // A later request cannot inherit another request's controls or slot values.
  const Result default_again = run(defaults, true);
  PHX_CHECK(default_again.points == unrefined.points);
  PHX_CHECK(default_again.cells == unrefined.cells);
  PHX_CHECK(default_again.segments == unrefined.segments);
  PHX_CHECK(default_again.work == unrefined.work);

  Input exact = strict;
  exact.max_work = static_cast<int64_t>(baseline.work[PHX_MC_PLANAR_WORK_UNITS]);
  exact.max_scratch_bytes = static_cast<int64_t>(baseline.memory[PHX_MC_MEMORY_PEAK_BYTES]);
  const Result repeated = run(exact, true);
  PHX_CHECK(repeated.status == PHX_MC_OK);
  PHX_CHECK(repeated.points == baseline.points && repeated.cells == baseline.cells);
  PHX_CHECK(repeated.segments == baseline.segments && repeated.vertex_map == baseline.vertex_map);
  PHX_CHECK(repeated.witness_parents == baseline.witness_parents);
  PHX_CHECK(repeated.witness_segments == baseline.witness_segments);
  PHX_CHECK(repeated.witness_ratios == baseline.witness_ratios);
  PHX_CHECK(repeated.work == baseline.work);
  --exact.max_work;
  const Result work_refused = run(exact, true);
  PHX_CHECK(work_refused.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(work_refused.memory[PHX_MC_MEMORY_LIVE_BYTES] == 0);
  exact.max_work = strict.max_work;
  --exact.max_scratch_bytes;
  const Result memory_refused = run(exact, true);
  PHX_CHECK(memory_refused.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(memory_refused.memory[PHX_MC_MEMORY_LIVE_BYTES] == 0);
  PHX_CHECK(strict.points == source_points && strict.segments == source_segments);
  PHX_CHECK(exact.points == source_points && exact.segments == source_segments);
}

}  // namespace

int main() {
  test_segments_and_vertex_on_segment();
  test_random_segments();
  test_lattice_segments();
  test_hole_carving();
  test_ruppert_square();
  test_ruppert_cavity_fan();
  test_ruppert_l_shape();
  test_ruppert_slanted_polygon();
  test_refinement_limit();
  test_ruppert_max_edge_length();
  test_lattice_seeding();
  test_arguments();
  test_native_limits_and_source_rollback();
  test_split_witness_dag();
  test_hull_split_witnesses();
  test_witness_capacity_and_atomic_publication();
  test_cumulative_execution_work_and_local_prefixes();
  test_cdt_work_uses_reserved_native_target_once();
  test_quality_cache_current_geometry_controls_and_refusal_boundaries();
  return phx::mc::test::finish("test_cdt2d");
}
