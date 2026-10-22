//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Consecutive P1 edge-root insertions preserve constrained source strata,
// positive cells, parent provenance, and refusal without publishing output.
// Level-set roots are the exact source-owned P1 roots with certified
// correctly rounded carriers, including generic non-representable roots.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "check.hpp"
#include "dyadic.hpp"
#include "phydrax_meshcore.h"
#include "tet_mesh.hpp"

namespace {
using namespace phx::mc;

constexpr double kPoints[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
constexpr int32_t kCell[] = {0, 1, 2, 3};
constexpr int32_t kFaces[] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
constexpr int32_t kEdges[] = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
constexpr int32_t kFaceSources[] = {0, 1, 2, 3};
constexpr int32_t kEdgeSources[] = {0, 1, 2, 3, 4, 5};

void check_cells(const double* points, const int32_t* cells, int64_t count) {
  double determinant_sum = 0.0;
  for (int64_t row = 0; row < count; ++row) {
    const int32_t* v = cells + 4 * row;
    PHX_CHECK(orient3d(points + 3 * v[0], points + 3 * v[1],
                       points + 3 * v[2], points + 3 * v[3]) > 0);
    const double* corners[4] = {points + 3 * v[0], points + 3 * v[1],
                                points + 3 * v[2], points + 3 * v[3]};
    determinant_sum += 6.0 * geometry::signed_volume(corners);
  }
  // Every determinant of this midpoint subdivision is exactly dyadic.
  PHX_CHECK(determinant_sum == 1.0);
}

void test_consecutive_constrained_edge_insertions() {
  TetMesh mesh(BoundaryPolicy::kConforming, 7, 32);
  const int32_t region = 0;
  PHX_CHECK(mesh.build(4, kPoints, 1, kCell, &region, 4, kFaces, kFaceSources,
                       6, kEdges, kEdgeSources, nullptr) == PHX_MC_OK);
  for (int32_t endpoint = 1; endpoint <= 3; ++endpoint) {
    double root[3] = {};
    root[endpoint - 1] = 0.5;
    std::vector<int32_t> star;
    PHX_CHECK(mesh.vertex_star(0, star));
    int32_t origin = -1;
    for (int32_t cell : star) {
      if (!is_ghost(mesh.tet(cell)) && vertex_slot(mesh.tet(cell), endpoint) >= 0) {
        origin = cell;
        break;
      }
    }
    PHX_CHECK(origin >= 0);
    if (origin < 0) return;
    const Insertion prepared = mesh.prepare(root, origin, 0, endpoint);
    PHX_CHECK(prepared == Insertion::kOk);
    if (prepared != Insertion::kOk) return;
    const Insertion committed = mesh.commit(InsertKind::kSubsegment, 0.0);
    PHX_CHECK(committed == Insertion::kOk);
    if (committed != Insertion::kOk) return;
    PHX_CHECK(mesh.vertex_count() == 4 + endpoint);
    PHX_CHECK(mesh.complex().audit());
    PHX_CHECK(mesh.segment_source(0, 3 + endpoint) == endpoint - 1);
    PHX_CHECK(mesh.segment_source(3 + endpoint, endpoint) == endpoint - 1);
    std::vector<std::array<int32_t, 5>> cells;
    mesh.collect_cells(cells);
    std::vector<int32_t> connectivity;
    for (const auto& cell : cells) {
      connectivity.insert(connectivity.end(), cell.begin(), cell.begin() + 4);
      PHX_CHECK(cell[4] == region);
    }
    std::vector<double> points;
    for (int32_t vertex = 0; vertex < mesh.vertex_count(); ++vertex) {
      points.insert(points.end(), mesh.point(vertex), mesh.point(vertex) + 3);
    }
    check_cells(points.data(), connectivity.data(), static_cast<int64_t>(cells.size()));
  }
}

void test_level_set_midpoints_in_every_crossing_order() {
  std::array<int32_t, 3> order = {0, 1, 2};
  do {
    std::array<int32_t, 12> edges;
    for (std::size_t i = 0; i < order.size(); ++i) {
      std::copy_n(kEdges + 2 * order[i], 2, edges.data() + 2 * i);
    }
    std::copy_n(kEdges + 6, 6, edges.data() + 6);
    const double values[] = {-0.5, 0.5, 0.5, 0.5};
    std::array<int8_t, 6> protected_edges = {};
    std::array<double, 21> points;
    std::array<int32_t, 128> cells;
    std::array<int32_t, 32> parents;
    std::array<int32_t, 14> sources;
    std::array<double, 14> weights;
    std::array<double, 7> deviations;
    std::array<int64_t, PHX_MC_LEVEL_SET_COUNTERS> counters = {};
    const phx_mc_level_set_split_3d_request request{
        4, kPoints, values, 1, kCell, 4, kFaces, 6, edges.data(), protected_edges.data(),
        7, 32, 100000, 32, PHX_MC_LEVEL_SET_FULL_CARRIERS, points.data(), cells.data(),
        parents.data(), sources.data(), weights.data(), deviations.data(), counters.data()};
    const int32_t status = phx_mc_level_set_split_3d(&request);
    PHX_CHECK(status == PHX_MC_OK);
    if (status != PHX_MC_OK) continue;
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_VERTICES] == 7);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_INSERTED] == 3);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_FAILED_EDGE] == -1);
    PHX_CHECK(std::equal(std::begin(kPoints), std::end(kPoints), points.begin()));
    for (int32_t root = 0; root < 3; ++root) {
      const int32_t vertex = 4 + root;
      PHX_CHECK(sources[2 * vertex] == 0);
      PHX_CHECK(sources[2 * vertex + 1] == order[root] + 1);
      PHX_CHECK(weights[2 * vertex] == 0.5 && weights[2 * vertex + 1] == 0.5);
      PHX_CHECK(deviations[vertex] == 0.0);
      for (int axis = 0; axis < 3; ++axis) {
        PHX_CHECK(points[3 * vertex + axis] == (axis == order[root] ? 0.5 : 0.0));
      }
    }
    const int64_t count = counters[PHX_MC_LEVEL_SET_CELLS];
    check_cells(points.data(), cells.data(), count);
    for (int64_t row = 0; row < count; ++row) {
      PHX_CHECK(parents[row] == 0);
      bool negative = false, positive = false;
      for (int k = 0; k < 4; ++k) {
        const int32_t vertex = cells[4 * row + k];
        negative = negative || vertex == 0;
        positive = positive || (vertex > 0 && vertex < 4);
      }
      PHX_CHECK(negative != positive);
    }
    // A protected second crossing must refuse without publishing any output.
    protected_edges[1] = 1;
    points.fill(-7.0);
    cells.fill(-7);
    parents.fill(-7);
    sources.fill(-7);
    weights.fill(-7.0);
    deviations.fill(-7.0);
    PHX_CHECK(phx_mc_level_set_split_3d(&request) == PHX_MC_CONSTRAINT_INTERSECTION);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_INSERTED] == 1);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_FAILED_EDGE] == 1);
    PHX_CHECK(std::all_of(points.begin(), points.end(), [](double x) { return x == -7.0; }));
    PHX_CHECK(std::all_of(cells.begin(), cells.end(), [](int32_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(parents.begin(), parents.end(), [](int32_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(sources.begin(), sources.end(), [](int32_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(weights.begin(), weights.end(), [](double x) { return x == -7.0; }));
    PHX_CHECK(std::all_of(deviations.begin(), deviations.end(), [](double x) { return x == -7.0; }));
  } while (std::next_permutation(order.begin(), order.end()));
}

struct Split {
  int32_t status = -1;
  std::vector<double> points;
  std::vector<int32_t> cells;
  std::vector<int32_t> parents;
  std::vector<int32_t> sources;
  std::vector<double> weights;
  std::vector<double> deviations;
  std::array<int64_t, PHX_MC_LEVEL_SET_COUNTERS> counters{};
};

Split split_one(const double* source_points, const int32_t* cell, const double* values) {
  Split out;
  out.points.assign(3 * 10, -7.0);
  out.cells.assign(4 * 70, -7);
  out.parents.assign(70, -7);
  out.sources.assign(2 * 10, -7);
  out.weights.assign(2 * 10, -7.0);
  out.deviations.assign(10, -7.0);
  const std::array<int8_t, 6> protected_edges = {};
  const phx_mc_level_set_split_3d_request request{
      4, source_points, values, 1, cell, 4, kFaces, 6, kEdges, protected_edges.data(),
      10, 70, 100000, 32, PHX_MC_LEVEL_SET_FULL_CARRIERS, out.points.data(), out.cells.data(),
      out.parents.data(), out.sources.data(), out.weights.data(), out.deviations.data(),
      out.counters.data()};
  out.status = phx_mc_level_set_split_3d(&request);
  return out;
}

Dyadic magnitude(const Dyadic& value) { return value.sign() < 0 ? -value : value; }

double approximate_quotient(const Dyadic& numerator, const Dyadic& denominator) {
  std::int64_t top = 0, bottom = 0;
  const double fraction = numerator.fraction(top) / denominator.fraction(bottom);
  return numerator.sign() == 0 ? 0.0 : std::ldexp(fraction, static_cast<int>(top - bottom));
}

// Every root row is the exact source-owned root: homogeneous weights are the
// swapped source magnitudes (the P1 value vanishes exactly), each carrier
// coordinate is the nearest binary64 value to the exact quotient, and the
// deviation bounds the carrier error and is zero exactly when it vanishes.
void check_exact_roots(const Split& out, const double* source_points, const double* values,
                       int64_t source_count = 4) {
  const int64_t count = out.counters[PHX_MC_LEVEL_SET_VERTICES];
  for (int64_t v = source_count; v < count; ++v) {
    const int32_t a = out.sources[2 * v], b = out.sources[2 * v + 1];
    const double wa = out.weights[2 * v], wb = out.weights[2 * v + 1];
    PHX_CHECK(wa == std::abs(values[b]) && wb == std::abs(values[a]));
    PHX_CHECK((Dyadic(values[a]) * Dyadic(wa) + Dyadic(values[b]) * Dyadic(wb)).sign() == 0);
    const Dyadic w = Dyadic(wa) + Dyadic(wb);
    double squared = 0.0;
    bool exact = true;
    for (int k = 0; k < 3; ++k) {
      const Dyadic numerator = Dyadic(wa) * Dyadic(source_points[3 * a + k]) +
                               Dyadic(wb) * Dyadic(source_points[3 * b + k]);
      const double x = out.points[3 * v + k];
      const Dyadic residual = magnitude(numerator - Dyadic(x) * w);
      for (double neighbour : {geometry::next_up(x), geometry::next_down(x)}) {
        PHX_CHECK((residual - magnitude(numerator - Dyadic(neighbour) * w)).sign() <= 0);
      }
      exact = exact && residual.sign() == 0;
      const double component = approximate_quotient(residual, w);
      squared += component * component;
    }
    PHX_CHECK((out.deviations[v] == 0.0) == exact);
    PHX_CHECK(out.deviations[v] >= std::sqrt(squared));
    PHX_CHECK(out.deviations[v] <= 1e-15);
  }
}

// A13: a generic certified tetrahedron whose binary64 roots are not exactly
// collinear with their source edges inserts its exact roots.
void test_generic_roots_use_exact_source_law() {
  constexpr double kGeneric[] = {0.1, 0.2, 0.3, 0.7, 0.5, 0.9, 0.2, 0.8, 0.1, 0.6, 0.1, 0.7};
  constexpr int32_t kPositive[] = {0, 2, 1, 3};
  const double symmetric[] = {1.0, -1.0, 1.0, 1.0};
  const double asymmetric[] = {2.0, -0.3, 3.0, 0.7};
  for (const double* values : {symmetric, asymmetric}) {
    const Split out = split_one(kGeneric, kPositive, values);
    PHX_CHECK(out.status == PHX_MC_OK);
    if (out.status != PHX_MC_OK) continue;
    PHX_CHECK(out.counters[PHX_MC_LEVEL_SET_VERTICES] == 7);
    PHX_CHECK(out.counters[PHX_MC_LEVEL_SET_INSERTED] == 3);
    PHX_CHECK(out.counters[PHX_MC_LEVEL_SET_FAILED_EDGE] == -1);
    PHX_CHECK(std::equal(std::begin(kGeneric), std::end(kGeneric), out.points.begin()));
    check_exact_roots(out, kGeneric, values);
    bool inexact = false;
    for (int64_t v = 4; v < 7; ++v) inexact = inexact || out.deviations[v] > 0.0;
    PHX_CHECK(inexact);
    const int64_t cells = out.counters[PHX_MC_LEVEL_SET_CELLS];
    PHX_CHECK(cells == 4);
    for (int64_t row = 0; row < cells; ++row) {
      const int32_t* v = out.cells.data() + 4 * row;
      PHX_CHECK(out.parents[row] == 0);
      PHX_CHECK(orient3d(out.points.data() + 3 * v[0], out.points.data() + 3 * v[1],
                         out.points.data() + 3 * v[2], out.points.data() + 3 * v[3]) > 0);
    }
  }
}

// Roots are exact for every finite value pair; a carrier that cannot
// represent the exact subdivision refuses without output.
void test_full_range_values_keep_exact_roots() {
  constexpr double kShifted[] = {1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2};
  constexpr double kTiny = 0x1p-1074;
  constexpr double kHuge = std::numeric_limits<double>::max();
  const double balanced[] = {-kTiny, 2 * kTiny, 3 * kTiny, kTiny};
  const double opposed[] = {-kHuge, kHuge, kHuge / 3, kHuge};
  for (const double* values : {balanced, opposed}) {
    const Split out = split_one(kShifted, kCell, values);
    PHX_CHECK(out.status == PHX_MC_OK);
    if (out.status != PHX_MC_OK) continue;
    PHX_CHECK(out.counters[PHX_MC_LEVEL_SET_INSERTED] == 3);
    check_exact_roots(out, kShifted, values);
  }
  const double absorbed[] = {-kTiny, kHuge, kHuge, kHuge};
  const double near_vertex[] = {-0x1p-590, 1.0, 1.0, 1.0};
  for (const double* values : {absorbed, near_vertex}) {
    const Split coincident = split_one(kShifted, kCell, values);
    PHX_CHECK(coincident.status == PHX_MC_DEGENERATE_INPUT);
    PHX_CHECK(coincident.counters[PHX_MC_LEVEL_SET_FAILED_EDGE] == 0);
    PHX_CHECK(coincident.counters[PHX_MC_LEVEL_SET_INSERTED] == 3);
    PHX_CHECK(std::all_of(coincident.points.begin(), coincident.points.end(),
                          [](double x) { return x == -7.0; }));
    PHX_CHECK(std::all_of(coincident.deviations.begin(), coincident.deviations.end(),
                          [](double x) { return x == -7.0; }));
    PHX_CHECK(std::all_of(coincident.cells.begin(), coincident.cells.end(),
                          [](int32_t x) { return x == -7; }));
  }
}

// An exact root halfway between two binary64 values rounds to the even one,
// on both sides of zero.
void test_tied_root_rounds_to_even() {
  constexpr double kTie = 1.0 + 0x1p-52;
  constexpr double kPositive[] = {1, 1, 1, 2, 1, 1, kTie, 2, 1, 1, 1, 2};
  // The mirror image x -> -x, positively oriented by swapping two vertices.
  constexpr double kNegative[] = {-1, 1, 1, -2, 1, 1, -kTie, 2, 1, -1, 1, 2};
  constexpr int32_t kMirrored[] = {0, 2, 1, 3};
  const double values[] = {-1.0, 1.0, 1.0, 1.0};
  for (const auto& [points, cell, even] :
       {std::tuple{kPositive, kCell, 1.0}, std::tuple{kNegative, kMirrored, -1.0}}) {
    const Split out = split_one(points, cell, values);
    PHX_CHECK(out.status == PHX_MC_OK);
    if (out.status != PHX_MC_OK) continue;
    check_exact_roots(out, points, values);
    // Edge (0, 2) is the second crossing: |S_x| = 1 + 2^-53 ties 1 and kTie.
    PHX_CHECK(out.sources[2 * 5] == 0 && out.sources[2 * 5 + 1] == 2);
    PHX_CHECK(std::bit_cast<uint64_t>(out.points[3 * 5]) == std::bit_cast<uint64_t>(even));
    PHX_CHECK(out.deviations[5] > 0.0);
  }
}

// A root coordinate below the normal range is decided exactly at the
// subnormal rounding boundary: nearer the smallest subnormal it leaves the
// coordinate domain (INVALID_INPUT); nearer zero its carrier coincides with
// the source vertex (DEGENERATE_INPUT). Neither publishes output.
void test_tiny_root_coordinates_at_the_subnormal_boundary() {
  constexpr double kTiny = 0x1p-1074;
  const double above_half[] = {-kTiny, 1.0, 1.0, 1.0};  // S_x just below kTiny
  const double below_half[] = {-kTiny, 3.0, 3.0, 3.0};  // S_x near kTiny / 3
  for (const auto& [values, status, inserted] :
       {std::tuple{above_half, PHX_MC_INVALID_INPUT, int64_t{0}},
        std::tuple{below_half, PHX_MC_DEGENERATE_INPUT, int64_t{3}}}) {
    const Split out = split_one(kPoints, kCell, values);
    PHX_CHECK(out.status == status);
    PHX_CHECK(out.counters[PHX_MC_LEVEL_SET_FAILED_EDGE] == 0);
    PHX_CHECK(out.counters[PHX_MC_LEVEL_SET_INSERTED] == inserted);
    PHX_CHECK(std::all_of(out.points.begin(), out.points.end(),
                          [](double x) { return x == -7.0; }));
  }
}

// Two cells share a crossing edge whose star has both of them, with two
// negative vertices: every root is exact, children stay positive on carriers
// and the shared source face is subdivided conformingly from both sides.
void test_shared_edge_star_with_two_negative_vertices() {
  constexpr double kPoints5[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
  constexpr int32_t kCells2[] = {0, 1, 2, 3, 1, 2, 3, 4};
  constexpr int32_t kFaces7[] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4};
  constexpr int32_t kEdges9[] = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3, 1, 4, 2, 4, 3, 4};
  const double values[] = {-1.0, -2.0, 1.0, 3.0, 0.5};
  Split out;
  out.points.assign(3 * 10, -7.0);
  out.cells.assign(4 * 64, -7);
  out.parents.assign(64, -7);
  out.sources.assign(2 * 10, -7);
  out.weights.assign(2 * 10, -7.0);
  out.deviations.assign(10, -7.0);
  const std::array<int8_t, 9> protected_edges = {};
  const phx_mc_level_set_split_3d_request request{
      5, kPoints5, values, 2, kCells2, 7, kFaces7, 9, kEdges9, protected_edges.data(),
      10, 64, 100000, 32, PHX_MC_LEVEL_SET_FULL_CARRIERS, out.points.data(), out.cells.data(),
      out.parents.data(), out.sources.data(), out.weights.data(), out.deviations.data(),
      out.counters.data()};
  out.status = phx_mc_level_set_split_3d(&request);
  PHX_CHECK(out.status == PHX_MC_OK);
  if (out.status != PHX_MC_OK) return;
  PHX_CHECK(out.counters[PHX_MC_LEVEL_SET_INSERTED] == 5);
  check_exact_roots(out, kPoints5, values, 5);
  const int64_t cells = out.counters[PHX_MC_LEVEL_SET_CELLS];
  std::vector<std::array<int32_t, 3>> faces;
  bool first_parent = false, second_parent = false;
  for (int64_t row = 0; row < cells; ++row) {
    const int32_t* v = out.cells.data() + 4 * row;
    first_parent = first_parent || out.parents[row] == 0;
    second_parent = second_parent || out.parents[row] == 1;
    PHX_CHECK(orient3d(out.points.data() + 3 * v[0], out.points.data() + 3 * v[1],
                       out.points.data() + 3 * v[2], out.points.data() + 3 * v[3]) > 0);
    for (int skip = 0; skip < 4; ++skip) {
      std::array<int32_t, 3> face{};
      for (int k = 0, j = 0; k < 4; ++k) {
        if (k != skip) face[static_cast<std::size_t>(j++)] = v[k];
      }
      std::sort(face.begin(), face.end());
      faces.push_back(face);
    }
  }
  PHX_CHECK(first_parent && second_parent);
  const auto on_shared_face = [&](int32_t vertex) {
    const int32_t a = out.sources[2 * vertex], b = out.sources[2 * vertex + 1];
    return a >= 1 && a <= 3 && b >= 1 && b <= 3;
  };
  for (const auto& face : faces) {
    const auto uses = std::count(faces.begin(), faces.end(), face);
    PHX_CHECK(uses == 1 || uses == 2);
    if (std::all_of(face.begin(), face.end(), on_shared_face)) PHX_CHECK(uses == 2);
  }
}

// Each source cell gains k + 1 children for a 1/k sign split and exactly 6
// for a 2/2 split, in every crossing order; those capacities suffice.
// Only crossing edges are permuted; noncrossing rows follow in fixed order.
void test_child_count_is_exact_in_every_crossing_order() {
  struct Case {
    std::array<double, 4> values;
    std::vector<int32_t> crossing;  // kEdges rows
    std::vector<int32_t> fixed;
    int64_t children;
  };
  const std::array<Case, 2> cases = {
      Case{{-1.0, 1.0, 1.0, 1.0}, {0, 1, 2}, {3, 4, 5}, 4},
      Case{{-1.0, -1.0, 1.0, 1.0}, {1, 2, 3, 4}, {0, 5}, 6},
  };
  for (const Case& split : cases) {
    std::vector<int32_t> order = split.crossing;
    do {
      std::vector<int32_t> rows = order;
      rows.insert(rows.end(), split.fixed.begin(), split.fixed.end());
      std::array<int32_t, 12> edges;
      for (std::size_t row = 0; row < rows.size(); ++row) {
        std::copy_n(kEdges + 2 * rows[row], 2, edges.data() + 2 * row);
      }
      std::array<int8_t, 6> protected_edges = {};
      std::array<double, 30> points;
      std::array<int32_t, 4 * 6> cells;
      std::array<int32_t, 6> parents;
      std::array<int32_t, 20> sources;
      std::array<double, 20> weights;
      std::array<double, 10> deviations;
      std::array<int64_t, PHX_MC_LEVEL_SET_COUNTERS> counters = {};
      const phx_mc_level_set_split_3d_request request{
          4, kPoints, split.values.data(), 1, kCell, 4, kFaces, 6, edges.data(),
          protected_edges.data(), 10, split.children, 100000, 32,
          PHX_MC_LEVEL_SET_FULL_CARRIERS, points.data(), cells.data(), parents.data(),
          sources.data(), weights.data(), deviations.data(), counters.data()};
      PHX_CHECK(phx_mc_level_set_split_3d(&request) == PHX_MC_OK);
      PHX_CHECK(counters[PHX_MC_LEVEL_SET_CELLS] == split.children);
    } while (std::next_permutation(order.begin(), order.end()));
  }
}

// Work exhaustion inside a crossing star walk or split is a typed capacity
// refusal with its failed edge, never an internal error.
void test_work_refusals_are_capacity() {
  const double values[] = {-0.5, 0.5, 0.5, 0.5};
  const int64_t input_work = 4 + 4 * 1 + 3 * 4 + 2 * 6;
  std::array<double, 21> points;
  std::array<int32_t, 128> cells;
  std::array<int32_t, 32> parents;
  std::array<int32_t, 14> sources;
  std::array<double, 14> weights;
  std::array<double, 7> deviations;
  const std::array<int8_t, 6> protected_edges = {};
  for (int64_t extra = 0; extra < 40; ++extra) {
    std::array<int64_t, PHX_MC_LEVEL_SET_COUNTERS> counters = {};
    const phx_mc_level_set_split_3d_request request{
        4, kPoints, values, 1, kCell, 4, kFaces, 6, kEdges, protected_edges.data(),
        7, 32, input_work + extra, 32, PHX_MC_LEVEL_SET_FULL_CARRIERS, points.data(),
        cells.data(), parents.data(), sources.data(), weights.data(), deviations.data(),
        counters.data()};
    const int32_t status = phx_mc_level_set_split_3d(&request);
    PHX_CHECK(status == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_WORK_UNITS] <= input_work + extra);
    PHX_CHECK(counters[PHX_MC_LEVEL_SET_FAILED_EDGE] >= 0);
  }
}

// TOPOLOGY_ONLY output: the caller owns the exact source positions, so the
// kernel publishes only cells, parents, sources and weights; the point and
// deviation buffers do not exist.
Split split_topology(const double* source_points, const int32_t* cell, const double* values,
                     const int8_t* protected_edges, int64_t work_limit = 100000) {
  Split out;
  out.cells.assign(4 * 70, -7);
  out.parents.assign(70, -7);
  out.sources.assign(2 * 10, -7);
  out.weights.assign(2 * 10, -7.0);
  const phx_mc_level_set_split_3d_request request{
      4, source_points, values, 1, cell, 4, kFaces, 6, kEdges, protected_edges,
      10, 70, work_limit, 32, PHX_MC_LEVEL_SET_TOPOLOGY_ONLY, nullptr, out.cells.data(),
      out.parents.data(), out.sources.data(), out.weights.data(), nullptr,
      out.counters.data()};
  out.status = phx_mc_level_set_split_3d(&request);
  return out;
}

// The topology-only subdivision is the full-carrier subdivision: identical
// cells, parents, sources, exact weights and sizes; only carrier
// certification work is absent.
void test_topology_only_publishes_the_exact_subdivision_without_carriers() {
  constexpr double kGeneric[] = {0.1, 0.2, 0.3, 0.7, 0.5, 0.9, 0.2, 0.8, 0.1, 0.6, 0.1, 0.7};
  constexpr int32_t kPositive[] = {0, 2, 1, 3};
  const double asymmetric[] = {2.0, -0.3, 3.0, 0.7};
  const double split_pair[] = {-1.0, -2.0, 1.0, 3.0};
  const std::array<int8_t, 6> protected_edges = {};
  for (const auto& [points, cell, values] :
       {std::tuple{kGeneric, kPositive, asymmetric}, std::tuple{kPoints, kCell, split_pair}}) {
    const Split full = split_one(points, cell, values);
    const Split topology = split_topology(points, cell, values, protected_edges.data());
    PHX_CHECK(full.status == PHX_MC_OK && topology.status == PHX_MC_OK);
    if (full.status != PHX_MC_OK || topology.status != PHX_MC_OK) continue;
    for (int slot : {PHX_MC_LEVEL_SET_VERTICES, PHX_MC_LEVEL_SET_CELLS, PHX_MC_LEVEL_SET_INSERTED,
                     PHX_MC_LEVEL_SET_ZERO_VERTICES, PHX_MC_LEVEL_SET_FAILED_EDGE}) {
      PHX_CHECK(topology.counters[slot] == full.counters[slot]);
    }
    PHX_CHECK(topology.counters[PHX_MC_LEVEL_SET_WORK_UNITS] <
              full.counters[PHX_MC_LEVEL_SET_WORK_UNITS]);
    PHX_CHECK(topology.cells == full.cells);
    PHX_CHECK(topology.parents == full.parents);
    PHX_CHECK(topology.sources == full.sources);
    PHX_CHECK(std::equal(topology.weights.begin(), topology.weights.end(), full.weights.begin(),
                         [](double x, double y) {
                           return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y);
                         }));
  }
}

// Carrier refusals belong to FULL_CARRIERS only: a rounded carrier that
// would coincide with a source vertex or leave the coordinate domain is the
// caller's to certify from its exact source, while topology, protection and
// work refusals are unchanged and publish nothing.
void test_topology_only_keeps_topology_protection_and_work_refusals() {
  constexpr double kShifted[] = {1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2};
  constexpr double kTiny = 0x1p-1074;
  const double near_vertex[] = {-0x1p-590, 1.0, 1.0, 1.0};
  const double below_domain[] = {-kTiny, 1.0, 1.0, 1.0};
  std::array<int8_t, 6> protected_edges = {};
  for (const auto& [points, values] :
       {std::tuple{kShifted, near_vertex}, std::tuple{kPoints, below_domain}}) {
    PHX_CHECK(split_one(points, kCell, values).status != PHX_MC_OK);
    const Split topology = split_topology(points, kCell, values, protected_edges.data());
    PHX_CHECK(topology.status == PHX_MC_OK);
    PHX_CHECK(topology.counters[PHX_MC_LEVEL_SET_INSERTED] == 3);
  }
  const double crossing[] = {-0.5, 0.5, 0.5, 0.5};
  protected_edges[1] = 1;
  const Split refused = split_topology(kPoints, kCell, crossing, protected_edges.data());
  PHX_CHECK(refused.status == PHX_MC_CONSTRAINT_INTERSECTION);
  PHX_CHECK(refused.counters[PHX_MC_LEVEL_SET_FAILED_EDGE] == 1);
  PHX_CHECK(std::all_of(refused.cells.begin(), refused.cells.end(),
                        [](int32_t x) { return x == -7; }));
  PHX_CHECK(std::all_of(refused.weights.begin(), refused.weights.end(),
                        [](double x) { return x == -7.0; }));
  protected_edges[1] = 0;
  const int64_t input_work = 4 + 4 * 1 + 3 * 4 + 2 * 6;
  for (int64_t extra = 0; extra < 40; ++extra) {
    const Split short_work =
        split_topology(kPoints, kCell, crossing, protected_edges.data(), input_work + extra);
    PHX_CHECK(short_work.status == PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(short_work.counters[PHX_MC_LEVEL_SET_WORK_UNITS] <= input_work + extra);
    PHX_CHECK(std::all_of(short_work.cells.begin(), short_work.cells.end(),
                          [](int32_t x) { return x == -7; }));
  }
}

// The output mode is closed and fixes the carrier buffers: any other mode,
// or buffers that do not match the mode, is INVALID_ARGUMENT before any work
// or counter publication.
void test_output_mode_and_buffers_are_validated_before_work() {
  const double values[] = {-0.5, 0.5, 0.5, 0.5};
  const std::array<int8_t, 6> protected_edges = {};
  std::array<double, 30> points;
  std::array<int32_t, 4 * 70> cells;
  std::array<int32_t, 70> parents;
  std::array<int32_t, 20> sources;
  std::array<double, 20> weights;
  std::array<double, 10> deviations;
  struct Selector {
    int32_t mode;
    double* points;
    double* deviations;
  };
  const std::array<Selector, 8> invalid = {
      Selector{2, points.data(), deviations.data()},
      Selector{2, nullptr, nullptr},
      Selector{-1, points.data(), deviations.data()},
      Selector{PHX_MC_LEVEL_SET_FULL_CARRIERS, nullptr, deviations.data()},
      Selector{PHX_MC_LEVEL_SET_FULL_CARRIERS, points.data(), nullptr},
      Selector{PHX_MC_LEVEL_SET_TOPOLOGY_ONLY, points.data(), nullptr},
      Selector{PHX_MC_LEVEL_SET_TOPOLOGY_ONLY, nullptr, deviations.data()},
      Selector{PHX_MC_LEVEL_SET_TOPOLOGY_ONLY, points.data(), deviations.data()},
  };
  for (const Selector& selector : invalid) {
    points.fill(-7.0);
    cells.fill(-7);
    parents.fill(-7);
    sources.fill(-7);
    weights.fill(-7.0);
    deviations.fill(-7.0);
    std::array<int64_t, PHX_MC_LEVEL_SET_COUNTERS> counters;
    counters.fill(-7);
    const phx_mc_level_set_split_3d_request request{
        4, kPoints, values, 1, kCell, 4, kFaces, 6, kEdges, protected_edges.data(),
        10, 70, 100000, 32, selector.mode, selector.points, cells.data(), parents.data(),
        sources.data(), weights.data(), selector.deviations, counters.data()};
    PHX_CHECK(phx_mc_level_set_split_3d(&request) == PHX_MC_INVALID_ARGUMENT);
    PHX_CHECK(std::all_of(counters.begin(), counters.end(), [](int64_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(points.begin(), points.end(), [](double x) { return x == -7.0; }));
    PHX_CHECK(std::all_of(cells.begin(), cells.end(), [](int32_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(sources.begin(), sources.end(), [](int32_t x) { return x == -7; }));
    PHX_CHECK(std::all_of(deviations.begin(), deviations.end(),
                          [](double x) { return x == -7.0; }));
  }
}
}  // namespace

int main() {
  test_consecutive_constrained_edge_insertions();
  test_level_set_midpoints_in_every_crossing_order();
  test_generic_roots_use_exact_source_law();
  test_full_range_values_keep_exact_roots();
  test_tied_root_rounds_to_even();
  test_tiny_root_coordinates_at_the_subnormal_boundary();
  test_shared_edge_star_with_two_negative_vertices();
  test_child_count_is_exact_in_every_crossing_order();
  test_work_refusals_are_capacity();
  test_topology_only_publishes_the_exact_subdivision_without_carriers();
  test_topology_only_keeps_topology_protection_and_work_refusals();
  test_output_mode_and_buffers_are_validated_before_work();
  return phx::mc::test::finish("test_level_set");
}
