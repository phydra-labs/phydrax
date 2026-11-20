//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "refine_queue.hpp"
#include "spatial_sort.hpp"
#include "triangulation2d.hpp"

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
};

struct Result {
  int32_t status = -1;
  std::vector<double> points;
  std::vector<int32_t> cells;
  std::vector<int32_t> segments;  // per cell edge
  std::vector<int32_t> vertex_map;
  std::vector<int32_t> constraint_sources;
  std::vector<int32_t> constraint_parents;
  std::vector<double> constraint_ratios;
};

Result run(const Input& in) {
  phx_mc_mesh* mesh = nullptr;
  Result r;
  r.status = phx_mc_constrained_delaunay_2d(
      static_cast<int64_t>(in.points.size() / 2), in.points.data(),
      static_cast<int64_t>(in.segments.size() / 2), in.segments.data(),
      static_cast<int64_t>(in.holes.size() / 2), in.holes.data(), in.keep_hull, in.min_angle,
      in.max_area, &mesh);
  const bool has_mesh = r.status == PHX_MC_OK || r.status == PHX_MC_UNRESOLVED;
  PHX_CHECK(has_mesh == (mesh != nullptr));
  if (mesh == nullptr) {
    return r;
  }
  const int64_t point_count = phx_mc_mesh_point_count(mesh);
  const int64_t cell_count = phx_mc_mesh_cell_count(mesh);
  const int64_t input_count = phx_mc_mesh_input_point_count(mesh);
  r.points.resize(static_cast<std::size_t>(2 * point_count));
  r.cells.resize(static_cast<std::size_t>(3 * cell_count));
  r.segments.resize(r.cells.size());
  r.vertex_map.resize(static_cast<std::size_t>(input_count));
  r.constraint_sources.resize(static_cast<std::size_t>(point_count));
  r.constraint_parents.resize(static_cast<std::size_t>(2 * point_count));
  r.constraint_ratios.resize(static_cast<std::size_t>(point_count));
  PHX_CHECK(phx_mc_mesh_copy_points(mesh, point_count, r.points.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_cells(mesh, cell_count, r.cells.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_cell_segments(mesh, cell_count, r.segments.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_vertex_map(mesh, input_count, r.vertex_map.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_constraint_sources(
                mesh, point_count, r.constraint_sources.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_constraint_parents(
                mesh, point_count, r.constraint_parents.data()) == PHX_MC_OK);
  PHX_CHECK(phx_mc_mesh_copy_constraint_ratios(
                mesh, point_count, r.constraint_ratios.data()) == PHX_MC_OK);
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

void test_ghost_layout() {
  // Begin with one triangle, then insert outside each side and on hull edges.
  const std::vector<double> points = {
      0, 0, 2, 0, 0, 2, 1, -1, 2, 2, -1, 1, 1, 0, 0, 1,
      3, -1, -1, 3, 3, 3};
  Triangulation2D tri;
  tri.reset(points.data(), static_cast<int64_t>(points.size() / 2), nullptr);
  tri.initialize(0, 1, 2);
  auto check_ghosts = [&]() {
    int ghosts = 0;
    for (std::size_t t = 0; t < tri.triangles.size(); ++t) {
      if (!tri.live[t] || !tri.is_ghost(static_cast<int32_t>(t))) continue;
      ++ghosts;
      const Triangle2D& ghost = tri.triangles[t];
      PHX_CHECK(ghost.v[0] >= 0 && ghost.v[1] >= 0 && ghost.v[2] == kInfiniteVertex);
      for (int k = 0; k < 3; ++k) {
        const int32_t neighbor = ghost.n[k];
        PHX_CHECK(neighbor >= 0 && static_cast<std::size_t>(neighbor) < tri.triangles.size());
        if (neighbor < 0 || static_cast<std::size_t>(neighbor) >= tri.triangles.size()) continue;
        PHX_CHECK(tri.live[static_cast<std::size_t>(neighbor)] != 0);
        const int opposite = tri.edge_index(neighbor, ghost.v[prev3(k)], ghost.v[next3(k)]);
        PHX_CHECK(opposite >= 0);
        if (opposite >= 0) PHX_CHECK(tri.triangles[neighbor].n[opposite] == static_cast<int32_t>(t));
        PHX_CHECK(tri.is_ghost(neighbor) == (k != 2));
      }
    }
    PHX_CHECK(ghosts >= 3);
  };
  check_ghosts();
  for (int32_t vertex = 3; vertex < tri.vertex_count(); ++vertex) {
    tri.insert(vertex);
    check_ghosts();
  }
}

void test_spatial_queue_order() {
  struct Entry { int priority; int id; };
  struct Worse {
    bool operator()(const Entry& a, const Entry& b) const {
      return a.priority != b.priority ? a.priority < b.priority : a.id > b.id;
    }
  };
  auto sequence = [](bool reverse) {
    SpatialRefinementQueue<Entry, Worse> queue(0.0, 0.0, 1.0, 1.0, 10000.0, 4);
    PHX_CHECK(queue.spatial());
    for (int i = 0; i < 4; ++i) {
      const int id = reverse ? 3 - i : i;
      queue.push({id % 2, id}, id < 2 ? 0.1 : 0.9, 0.1);
    }
    std::vector<int> result;
    while (!queue.empty()) {
      result.push_back(queue.top().id);
      queue.pop();
    }
    return result;
  };
  PHX_CHECK((sequence(false) == std::vector<int>{1, 0, 3, 2}));
  PHX_CHECK(sequence(false) == sequence(true));
  SpatialRefinementQueue<Entry, Worse> single(0.0, 0.0, 1.0, 1.0, 10.0, 4);
  PHX_CHECK(!single.spatial());
  single.push({2, 0});
  single.push({1, 1});
  Entry current{};
  PHX_CHECK(single.pop_current(current, [](const Entry& e) { return e.id != 0; }));
  PHX_CHECK(current.id == 1);
  PHX_CHECK(!single.pop_current(current, [](const Entry&) { return true; }));
}

void test_refinement_determinism() {
  Input in;
  // Generic-position points from a fixed seed; unlike a square, these avoid
  // index-dependent simulation-of-simplicity choices on cocircular vertices.
  in.points = {
      0.77395604855596334, 0.43887843975205232, 0.85859791991138246, 0.6973680290593639,
      0.094177347887649532, 0.97562235163675592, 0.76113970199035297, 0.78606430527695381,
      0.12811363267554587, 0.45038593789556713, 0.37079802423258124, 0.92676498884860181,
      0.64386512008066454, 0.82276161327082997, 0.44341419882733113, 0.22723872178477689,
      0.5545847870158348, 0.063817256104175324, 0.82763117199258207, 0.63166439912206485,
      0.75808774008537383, 0.35452596812986836, 0.97069802439490327, 0.89312112132219768,
      0.77838349707376187, 0.19463870785196757, 0.46672100372703418, 0.043803765787228777,
      0.15428949206754783, 0.68304895324245463, 0.74476215590781714, 0.96750973243421001,
      0.32582535813815194, 0.37045970603486889, 0.46955581127580792, 0.1894713590842857,
      0.12992150533547164, 0.47570492622593374, 0.22690934905088411, 0.66981399468251035};
  in.segments = {0, 1, 2, 3};
  in.keep_hull = 1;
  in.max_area = 0.0003;
  const Result a = run(in), again = run(in);
  PHX_CHECK(a.status == PHX_MC_OK && again.status == PHX_MC_OK);
  auto same_bytes = [](const auto& x, const auto& y) {
    return x.size() == y.size() &&
           (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0);
  };
  PHX_CHECK(same_bytes(a.points, again.points));
  PHX_CHECK(same_bytes(a.cells, again.cells));
  PHX_CHECK(same_bytes(a.segments, again.segments));
  PHX_CHECK(same_bytes(a.vertex_map, again.vertex_map));
  PHX_CHECK(same_bytes(a.constraint_sources, again.constraint_sources));
  PHX_CHECK(same_bytes(a.constraint_parents, again.constraint_parents));
  PHX_CHECK(same_bytes(a.constraint_ratios, again.constraint_ratios));

  Input permuted = in;
  for (int i = 0; i < 20; ++i) {
    permuted.points[2 * i] = in.points[2 * (19 - i)];
    permuted.points[2 * i + 1] = in.points[2 * (19 - i) + 1];
  }
  permuted.segments = {17, 16, 19, 18};
  const Result b = run(permuted);
  PHX_CHECK(b.status == PHX_MC_OK);
  using Point = std::array<double, 2>;
  using Triangle = std::array<Point, 3>;
  using Edge = std::array<Point, 2>;
  auto point = [](const double* p) { return Point{p[0], p[1]}; };
  auto triangles = [&](const Result& r) {
    std::set<Triangle> result;
    for (std::size_t i = 0; i < r.cells.size(); i += 3) {
      Triangle t{point(at(r, r.cells[i])), point(at(r, r.cells[i + 1])),
                 point(at(r, r.cells[i + 2]))};
      std::sort(t.begin(), t.end());
      result.insert(t);
    }
    return result;
  };
  auto labeled_edges = [&](const Result& r, const Input& input) {
    std::set<std::pair<Edge, Edge>> result;
    for (std::size_t i = 0; i < r.cells.size(); i += 3) {
      for (int k = 0; k < 3; ++k) {
        const int32_t source = r.segments[i + k];
        if (source < 0) continue;
        Edge edge{point(at(r, r.cells[i + (k + 1) % 3])),
                  point(at(r, r.cells[i + (k + 2) % 3]))};
        Edge original{point(input.points.data() + 2 * input.segments[2 * source]),
                      point(input.points.data() + 2 * input.segments[2 * source + 1])};
        std::sort(edge.begin(), edge.end());
        std::sort(original.begin(), original.end());
        result.emplace(edge, original);
      }
    }
    return result;
  };
  PHX_CHECK(triangles(a) == triangles(b));
  PHX_CHECK(labeled_edges(a, in) == labeled_edges(b, permuted));
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

void test_constraint_source_witnesses() {
  Input in;
  in.points = {0, 0, 1, 0, 1, 1, 0, 1};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 0};
  in.min_angle = 30.0;
  in.max_area = 0.001;
  const Result r = run(in);
  PHX_CHECK(r.status == PHX_MC_OK || r.status == PHX_MC_UNRESOLVED);
  check_mesh(r);
  check_segments(r, in);
  check_constrained_delaunay(r);
  PHX_CHECK_NEAR(total_area(r), 1.0, 1e-12);
  bool found_source_witness = false;
  for (std::size_t vertex = in.points.size() / 2; vertex < r.constraint_sources.size(); ++vertex) {
    const int32_t source = r.constraint_sources[vertex];
    if (source < 0) continue;
    PHX_CHECK(source < static_cast<int32_t>(in.segments.size() / 2));
    const int32_t parent0 = r.constraint_parents[2 * vertex];
    const int32_t parent1 = r.constraint_parents[2 * vertex + 1];
    PHX_CHECK(parent0 >= 0 && parent0 < static_cast<int32_t>(vertex));
    PHX_CHECK(parent1 >= 0 && parent1 < static_cast<int32_t>(vertex));
    PHX_CHECK(std::isfinite(r.constraint_ratios[vertex]));
    PHX_CHECK(r.constraint_ratios[vertex] > 0.0 && r.constraint_ratios[vertex] < 1.0);
    found_source_witness = true;
  }
  PHX_CHECK(found_source_witness);
}

void test_refinement_preserves_smooth_samples_and_constraint_junctions() {
  Input in;
  // The first two bottom segments meet at a nonacute sampled boundary point;
  // the interior constraint makes it a true three-ray junction in the second run.
  in.points = {0, 0, 0.25, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 4, 4, 0};
  in.min_angle = 20.0;
  in.max_area = 0.03;
  for (bool junction : {false, true}) {
    Input candidate = in;
    if (junction) candidate.segments.insert(candidate.segments.end(), {1, 5});
    const Result result = run(candidate);
    PHX_CHECK(result.status == PHX_MC_OK);
    check_mesh(result);
    check_segments(result, candidate);
    check_constrained_delaunay(result);
    check_quality(result, 20.0, 0.03);
    PHX_CHECK_NEAR(total_area(result), 1.0, 1e-12);
  }
}

void test_distinct_nearby_constraint_vertices_remain_distinct() {
  Input in;
  const double next = std::nextafter(0.25, 1.0);
  in.points = {0, 0, 0.25, 0, next, 0, 1, 0, 1, 1, 0, 1};
  in.segments = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0};
  const Result result = run(in);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.vertex_map[1] != result.vertex_map[2]);
  check_mesh(result);
  check_segments(result, in);
  check_constrained_delaunay(result);
  PHX_CHECK_NEAR(total_area(result), 1.0, 1e-12);
}

void test_acute_input_corner_retains_its_constraints() {
  Input in;
  in.points = {0, 0, 2, 0, 2, 0.3};
  in.segments = {0, 1, 1, 2, 2, 0};
  in.min_angle = 20.0;
  in.max_area = 0.05;
  const Result result = run(in);
  PHX_CHECK(result.status == PHX_MC_OK || result.status == PHX_MC_UNRESOLVED);
  check_mesh(result);
  check_segments(result, in);
  check_constrained_delaunay(result);
  PHX_CHECK_NEAR(total_area(result), 0.3, 1e-12);
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
                                           kInf, &mesh) == PHX_MC_INVALID_ARGUMENT);
  const int32_t one_segment[2] = {0, 1};
  constexpr int64_t unrepresentable_segment_count = (int64_t{1} << 31) + 1;
  PHX_CHECK(phx_mc_constrained_delaunay_2d(
                4, in.points.data(), unrepresentable_segment_count, one_segment, 0, nullptr, 0,
                0.0, kInf, &mesh) == PHX_MC_INDEX_RANGE_ERROR);
  PHX_CHECK(mesh == nullptr);
  PHX_CHECK(phx_mc_constrained_delaunay_2d(4, in.points.data(), 0, nullptr, 0, nullptr, 0, 0.0,
                                           kInf, nullptr) == PHX_MC_INVALID_ARGUMENT);
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

bool same_result(const Result& left, const Result& right) {
  return left.status == right.status && left.points == right.points &&
         left.cells == right.cells && left.segments == right.segments &&
         left.vertex_map == right.vertex_map &&
         left.constraint_sources == right.constraint_sources &&
         left.constraint_parents == right.constraint_parents &&
         left.constraint_ratios == right.constraint_ratios;
}

// Constrained triangulation and Ruppert refinement give their round-to-nearest,
// gradual-underflow result in every caller mode: exactly cocircular interior
// orbits, Steiner constructions (circumcenters, midpoints, concentric shells at
// an acute corner), and the rejection of a subnormal coordinate.
void test_floating_point_hazards() {
  std::vector<Input> inputs;
  Input l_shape;
  l_shape.points = {0, 0, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2};
  l_shape.segments = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0};
  l_shape.min_angle = 20.7;
  l_shape.max_area = 0.05;
  inputs.push_back(l_shape);
  Input acute;
  acute.points = {0, 0, 2, 0, 2, 0.3};
  acute.segments = {0, 1, 1, 2, 2, 0};
  acute.min_angle = 20.0;
  acute.max_area = 0.05;
  inputs.push_back(acute);
  Input orbits;
  orbits.points = {-2, -2, 2, -2, 2, 2, -2, 2};
  orbits.segments = {0, 1, 1, 2, 2, 3, 3, 0};
  std::uint64_t seed = 2026;
  for (int orbit = 0; orbit < 24; ++orbit) {
    const double x = 2.0 * uniform(seed) - 1.0;
    const double y = 2.0 * uniform(seed) - 1.0;
    orbits.points.insert(orbits.points.end(), {x, y, -y, x, -x, -y, y, -x});
  }
  inputs.push_back(orbits);
  orbits.min_angle = 25.0;
  inputs.push_back(orbits);
  std::vector<Result> references;
  for (const Input& in : inputs) {
    references.push_back(run(in));
    const Result& r = references.back();
    PHX_CHECK(r.status == PHX_MC_OK || r.status == PHX_MC_UNRESOLVED);
    check_mesh(r);
    check_segments(r, in);
    check_constrained_delaunay(r);
  }
  Input subnormal = square_with_hole();
  subnormal.points[16] = 0x1p-1060;
  PHX_CHECK(run(subnormal).status == PHX_MC_RANGE_ERROR);
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    for (std::size_t k = 0; k < inputs.size(); ++k) {
      PHX_CHECK(same_result(run(inputs[k]), references[k]));
    }
    PHX_CHECK(run(subnormal).status == PHX_MC_RANGE_ERROR);
  });
}

}  // namespace

int main() {
  test_segments_and_vertex_on_segment();
  test_random_segments();
  test_lattice_segments();
  test_hole_carving();
  test_ghost_layout();
  test_refinement_determinism();
  test_spatial_queue_order();
  test_ruppert_square();
  test_ruppert_l_shape();
  test_ruppert_slanted_polygon();
  test_constraint_source_witnesses();
  test_refinement_preserves_smooth_samples_and_constraint_junctions();
  test_distinct_nearby_constraint_vertices_remain_distinct();
  test_acute_input_corner_retains_its_constraints();
  test_arguments();
  test_floating_point_hazards();
  return phx::mc::test::finish("test_cdt2d");
}
