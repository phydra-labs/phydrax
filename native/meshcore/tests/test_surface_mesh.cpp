//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Chart-embedded surface reconnection: exact chart validity, physical
// Delaunay flips, constrained edges, collapsed sides, refusals and budgets.
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {

struct Patch {
  std::vector<double> charts;
  std::vector<double> points;
  std::vector<int32_t> triangles;
  std::vector<int8_t> constrained;
  std::vector<double> normals;  // empty: no source normals
};

struct Outcome {
  int32_t status = -1;
  std::vector<int32_t> triangles;
  std::vector<int8_t> constrained;
  std::vector<int32_t> vertex_ids;
  std::vector<int32_t> item_status;
  std::array<int64_t, PHX_MC_SURFACE_COUNTERS> counters{};
  std::vector<double> charts;
  std::vector<double> points;
};

Outcome reconnect(const Patch& patch, const std::vector<double>& insert_charts,
                  const std::vector<double>& insert_points, double spacing, int32_t sweep,
                  int64_t max_triangles = 1024, int64_t work_limit = 1 << 20,
                  const std::vector<double>& insert_normals = {},
                  const std::vector<int32_t>& insert_edges = {}) {
  Outcome outcome;
  const int64_t count = static_cast<int64_t>(insert_charts.size() / 2);
  const std::vector<double> spacings(static_cast<std::size_t>(count), spacing);
  const std::vector<int32_t> hints(static_cast<std::size_t>(count), -1);
  const std::vector<double> normals =
      patch.normals.empty() ? std::vector<double>(patch.points.size(), 0.0) : patch.normals;
  const std::vector<double> new_normals =
      insert_normals.empty() ? std::vector<double>(insert_points.size(), 0.0) : insert_normals;
  outcome.triangles.assign(static_cast<std::size_t>(3 * max_triangles), -1);
  outcome.constrained.assign(static_cast<std::size_t>(3 * max_triangles), 0);
  outcome.vertex_ids.assign(static_cast<std::size_t>(count), -7);
  outcome.item_status.assign(static_cast<std::size_t>(count), -7);
  int64_t produced = 0;
  outcome.status = phx_mc_surface_reconnect(
      static_cast<int64_t>(patch.charts.size() / 2), patch.charts.data(), patch.points.data(),
      normals.data(), nullptr, nullptr, static_cast<int64_t>(patch.triangles.size() / 3), patch.triangles.data(),
      patch.constrained.data(), count, insert_charts.data(), insert_points.data(),
      new_normals.data(), nullptr, nullptr, spacings.data(), hints.data(),
      insert_edges.empty() ? nullptr : insert_edges.data(), sweep, max_triangles, work_limit,
      outcome.triangles.data(),
      outcome.constrained.data(), &produced, outcome.vertex_ids.data(),
      outcome.item_status.data(), outcome.counters.data());
  outcome.triangles.resize(static_cast<std::size_t>(3 * produced));
  outcome.constrained.resize(static_cast<std::size_t>(3 * produced));
  outcome.charts = patch.charts;
  outcome.points = patch.points;
  for (int64_t item = 0; item < count; ++item) {
    if (outcome.vertex_ids[item] >= 0) {
      outcome.charts.insert(outcome.charts.end(), insert_charts.begin() + 2 * item,
                            insert_charts.begin() + 2 * item + 2);
      outcome.points.insert(outcome.points.end(), insert_points.begin() + 3 * item,
                            insert_points.begin() + 3 * item + 3);
    }
  }
  return outcome;
}

// Unit square chart with its diagonal (0, 2); boundary edges constrained.
Patch square(bool constrain_diagonal = false) {
  Patch patch;
  patch.charts = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  patch.points = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
  patch.triangles = {0, 1, 2, 0, 2, 3};
  const int8_t diagonal = constrain_diagonal ? 1 : 0;
  // Triangle (0, 1, 2): edge opposite 0 is (1, 2), opposite 1 is (2, 0).
  patch.constrained = {1, diagonal, 1, 1, 1, diagonal};
  return patch;
}

bool chart_valid(const Outcome& outcome) {
  for (std::size_t cell = 0; cell < outcome.triangles.size() / 3; ++cell) {
    const double* a = outcome.charts.data() + 2 * outcome.triangles[3 * cell];
    const double* b = outcome.charts.data() + 2 * outcome.triangles[3 * cell + 1];
    const double* c = outcome.charts.data() + 2 * outcome.triangles[3 * cell + 2];
    if (phx::mc::orient2d(a, b, c) <= 0) {
      return false;
    }
  }
  return true;
}

bool has_edge(const Outcome& outcome, int32_t first, int32_t second) {
  for (std::size_t cell = 0; cell < outcome.triangles.size() / 3; ++cell) {
    for (int k = 0; k < 3; ++k) {
      const int32_t a = outcome.triangles[3 * cell + (k + 1) % 3];
      const int32_t b = outcome.triangles[3 * cell + (k + 2) % 3];
      if ((a == first && b == second) || (a == second && b == first)) {
        return true;
      }
    }
  }
  return false;
}

// In the plane the physical criterion is the Delaunay criterion: every
// unconstrained edge must be locally Delaunay (exact incircle).
bool planar_delaunay(const Outcome& outcome) {
  const std::size_t count = outcome.triangles.size() / 3;
  for (std::size_t cell = 0; cell < count; ++cell) {
    const int32_t* t = outcome.triangles.data() + 3 * cell;
    for (std::size_t other = 0; other < count; ++other) {
      const int32_t* u = outcome.triangles.data() + 3 * other;
      for (int k = 0; k < 3; ++k) {
        if (u[k] == t[0] || u[k] == t[1] || u[k] == t[2]) {
          continue;
        }
        int shared = 0;
        for (int j = 0; j < 3; ++j) {
          shared += (u[j] == t[0] || u[j] == t[1] || u[j] == t[2]) ? 1 : 0;
        }
        if (shared == 2 &&
            phx::mc::incircle(outcome.charts.data() + 2 * t[0], outcome.charts.data() + 2 * t[1],
                              outcome.charts.data() + 2 * t[2],
                              outcome.charts.data() + 2 * u[k]) > 0) {
          return false;
        }
      }
    }
  }
  return true;
}

void planar_insertion_is_delaunay() {
  std::vector<double> charts;
  std::vector<double> points;
  for (int row = 1; row < 6; ++row) {
    for (int column = 1; column < 6; ++column) {
      const double u = column / 6.0 + 0.013 * ((row * 7 + column * 3) % 5 - 2);
      const double v = row / 6.0 + 0.011 * ((row * 5 + column * 2) % 7 - 3);
      charts.insert(charts.end(), {u, v});
      points.insert(points.end(), {u, v, 0.0});
    }
  }
  const Outcome outcome = reconnect(square(), charts, points, 0.0, 1);
  PHX_CHECK(outcome.status == PHX_MC_OK);
  PHX_CHECK(outcome.counters[0] == 25);
  PHX_CHECK(outcome.counters[1] == 0);
  PHX_CHECK(outcome.triangles.size() / 3 == 2 + 2 * 25);
  PHX_CHECK(outcome.vertex_ids[0] == 4 && outcome.vertex_ids[24] == 28);
  PHX_CHECK(chart_valid(outcome));
  PHX_CHECK(planar_delaunay(outcome));
}

// A chart-Delaunay kite whose physical image (v stretched tenfold) needs the
// other diagonal: the physical criterion decides.
Patch kite(bool coincident) {
  Patch patch;
  patch.charts = {0.0, 0.0, 1.0, -0.4, 2.0, 0.0, 1.0, 0.4};
  patch.points = {0.0, 0.0, 0.0, 1.0, -4.0, 0.0, 2.0, 0.0, 0.0, 1.0, 4.0, 0.0};
  if (coincident) {
    // Vertices 1 and 3 share one physical point (a collapsed chart side).
    patch.points = {0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  }
  patch.triangles = {0, 1, 3, 1, 2, 3};
  patch.constrained = {0, 1, 1, 1, 0, 1};
  return patch;
}

void physical_criterion_selects_diagonal() {
  const Outcome flipped = reconnect(kite(false), {}, {}, 0.0, 1);
  PHX_CHECK(flipped.status == PHX_MC_OK);
  PHX_CHECK(flipped.counters[2] == 1);
  PHX_CHECK(has_edge(flipped, 0, 2) && !has_edge(flipped, 1, 3));
  PHX_CHECK(chart_valid(flipped));
  const Outcome pole = reconnect(kite(true), {}, {}, 0.0, 1);
  PHX_CHECK(pole.status == PHX_MC_OK);
  PHX_CHECK(pole.counters[2] == 0);
  PHX_CHECK(has_edge(pole, 1, 3));
}

void constrained_diagonal_is_preserved() {
  Patch patch = kite(false);
  patch.constrained = {1, 1, 1, 1, 1, 1};
  const Outcome outcome = reconnect(patch, {}, {}, 0.0, 1);
  PHX_CHECK(outcome.status == PHX_MC_OK);
  PHX_CHECK(outcome.counters[2] == 0);
  PHX_CHECK(has_edge(outcome, 1, 3));
  // A point on the constrained diagonal is refused without any change.
  const Outcome refused = reconnect(patch, {1.0, 0.0}, {1.0, 0.0, 0.0}, 0.0, 0);
  PHX_CHECK(refused.item_status[0] == PHX_MC_CONSTRAINT_INTERSECTION);
  PHX_CHECK(refused.vertex_ids[0] == -1);
  PHX_CHECK(refused.triangles == patch.triangles);
}

void edge_split_and_refusals() {
  const Outcome split = reconnect(square(), {0.5, 0.5}, {0.5, 0.5, 0.0}, 0.0, 0);
  PHX_CHECK(split.item_status[0] == PHX_MC_OK);
  PHX_CHECK(split.triangles.size() / 3 == 4);
  PHX_CHECK(chart_valid(split));
  const std::vector<double> charts = {2.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.25, 0.26};
  const std::vector<double> points = {2.0, 0.5, 0.0, 1.0, 0.5, 0.0,
                                      0.0, 0.0, 0.0, 0.25, 0.26, 0.0};
  const Outcome refused = reconnect(square(), charts, points, 0.0, 0);
  PHX_CHECK(refused.item_status[0] == PHX_MC_INVALID_INPUT);
  PHX_CHECK(refused.item_status[1] == PHX_MC_CONSTRAINT_INTERSECTION);
  PHX_CHECK(refused.item_status[2] == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(refused.item_status[3] == PHX_MC_OK);
  PHX_CHECK(refused.vertex_ids[3] == 4);
  const Outcome close = reconnect(square(), {0.3, 0.6}, {0.3, 0.6, 0.0}, 0.6, 0);
  PHX_CHECK(close.item_status[0] == PHX_MC_DEGENERATE_INPUT);
  // A point whose split would fold the physical surface is refused.
  const Outcome folded = reconnect(square(), {0.7, 0.2}, {3.0, -2.0, 0.0}, 0.0, 0);
  PHX_CHECK(folded.item_status[0] == PHX_MC_DEGENERATE_INPUT);
}

void budgets_refuse_without_invalidating() {
  const std::vector<double> charts = {0.3, 0.1, 0.6, 0.2};
  const std::vector<double> points = {0.3, 0.1, 0.0, 0.6, 0.2, 0.0};
  const Outcome capacity = reconnect(square(), charts, points, 0.0, 0, 4);
  PHX_CHECK(capacity.item_status[0] == PHX_MC_OK);
  PHX_CHECK(capacity.item_status[1] == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(capacity.triangles.size() / 3 == 4);
  const Outcome work = reconnect(square(), charts, points, 0.0, 1, 1024, 2);
  PHX_CHECK(work.status == PHX_MC_OK);
  PHX_CHECK(work.counters[5] == 1);
  PHX_CHECK(work.item_status[1] == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(work.triangles == square().triangles);
  PHX_CHECK(work.constrained == square().constrained);
  PHX_CHECK(work.counters[4] <= 2);
  PHX_CHECK(chart_valid(work));
}

void invalid_triangulations_are_refused() {
  Patch clockwise = square();
  clockwise.triangles = {0, 2, 1, 0, 3, 2};
  PHX_CHECK(reconnect(clockwise, {}, {}, 0.0, 1).status == PHX_MC_INVALID_INPUT);
  Patch open = square();
  open.constrained[0] = 0;
  PHX_CHECK(reconnect(open, {}, {}, 0.0, 1).status == PHX_MC_INVALID_INPUT);
  Patch mismatched = square();
  mismatched.constrained[1] = 1;
  PHX_CHECK(reconnect(mismatched, {}, {}, 0.0, 1).status == PHX_MC_INVALID_INPUT);
  Patch infinite = square();
  infinite.points[0] = INFINITY;
  PHX_CHECK(reconnect(infinite, {}, {}, 0.0, 1).status == PHX_MC_NONFINITE_INPUT);
}

// Half-cylinder chart (angle, height): the reconnected surface keeps every
// physical normal outward.
void curved_patch_keeps_orientation() {
  Patch patch;
  const double pi = 3.14159265358979323846;
  patch.charts = {0.0, 0.0, pi, 0.0, pi, 1.0, 0.0, 1.0};
  patch.points = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  patch.triangles = {0, 1, 2, 0, 2, 3};
  patch.constrained = {1, 0, 1, 1, 1, 0};
  // Outward source normals of the cylinder at the chart corners.
  patch.normals = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  std::vector<double> charts;
  std::vector<double> points;
  std::vector<double> normals;
  for (int row = 1; row < 4; ++row) {
    for (int column = 1; column < 8; ++column) {
      const double u = pi * column / 8.0;
      const double v = row / 4.0;
      charts.insert(charts.end(), {u, v});
      points.insert(points.end(), {std::cos(u), std::sin(u), v});
      normals.insert(normals.end(), {std::cos(u), std::sin(u), 0.0});
    }
  }
  const Outcome outcome = reconnect(patch, charts, points, 0.0, 1, 1024, 1 << 20, normals);
  PHX_CHECK(outcome.status == PHX_MC_OK);
  PHX_CHECK(chart_valid(outcome));
  PHX_CHECK(outcome.counters[0] + outcome.counters[1] == 21);
  int outward = 0;
  int inward = 0;
  for (std::size_t cell = 0; cell < outcome.triangles.size() / 3; ++cell) {
    const double* a = outcome.points.data() + 3 * outcome.triangles[3 * cell];
    const double* b = outcome.points.data() + 3 * outcome.triangles[3 * cell + 1];
    const double* c = outcome.points.data() + 3 * outcome.triangles[3 * cell + 2];
    const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                         e1[0] * e2[1] - e1[1] * e2[0]};
    const double centroid[2] = {(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0};
    (n[0] * centroid[0] + n[1] * centroid[1] < 0.0 ? inward : outward) += 1;
  }
  // (angle, height) is counterclockwise about the outward normal.
  PHX_CHECK(inward == 0);
  PHX_CHECK(outward == static_cast<int>(outcome.triangles.size() / 3));
}

// A point whose source normal opposes its split triangles is refused.
void opposed_source_normal_is_refused() {
  const Outcome aligned = reconnect(square(), {0.3, 0.2}, {0.3, 0.2, 0.0}, 0.0, 0, 1024,
                                    1 << 20, {0.0, 0.0, 1.0});
  PHX_CHECK(aligned.item_status[0] == PHX_MC_OK);
  const Outcome opposed = reconnect(square(), {0.3, 0.2}, {0.3, 0.2, 0.0}, 0.0, 0, 1024,
                                    1 << 20, {0.0, 0.0, -1.0});
  PHX_CHECK(opposed.item_status[0] == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(opposed.triangles.size() / 3 == 2);
  Patch folded_parent = square();
  folded_parent.normals = {0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
                           0.0, 0.0, -1.0, 0.0, 0.0, -1.0};
  const Outcome folded = reconnect(folded_parent, {0.3, 0.2}, {0.3, 0.2, 0.0}, 0.0, 0,
                                   1024, 1 << 20, {0.0, 0.0, -1.0});
  PHX_CHECK(folded.item_status[0] == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(folded.triangles == folded_parent.triangles);
}

// A rounded source midpoint lies strictly on one side of the old diagonal.
// Explicit intent replaces the entire pair, not a containing-cell sliver.
void explicit_paired_cavity() {
  const double x = std::nextafter(0.5, 1.0);
  const Outcome result = reconnect(square(), {x, 0.5}, {x, 0.5, 0.0}, 0.0, 0,
                                   1024, 1 << 20, {}, {0, 2});
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.item_status[0] == PHX_MC_OK);
  PHX_CHECK(result.vertex_ids[0] == 4);
  PHX_CHECK(result.triangles.size() == 12);
  PHX_CHECK(chart_valid(result));
  PHX_CHECK(!has_edge(result, 0, 2));
  int boundary_count = 0;
  for (std::size_t cell = 0; cell < 4; ++cell) {
    bool inserted = false;
    for (int k = 0; k < 3; ++k) {
      inserted |= result.triangles[3 * cell + k] == 4;
      if (result.constrained[3 * cell + k] == 0) continue;
      ++boundary_count;
      const int32_t a = result.triangles[3 * cell + (k + 1) % 3];
      const int32_t b = result.triangles[3 * cell + (k + 2) % 3];
      PHX_CHECK(a < 4 && b < 4 && (std::abs(a - b) == 1 || std::abs(a - b) == 3));
    }
    PHX_CHECK(inserted);
  }
  PHX_CHECK(boundary_count == 4);
  // A successful first split makes the repeated old-edge request stale.
  const Outcome batch = reconnect(square(), {x, 0.5, 0.6, 0.6},
                                  {x, 0.5, 0.0, 0.6, 0.6, 0.0}, 0.0, 0,
                                  1024, 1 << 20, {}, {0, 2, 2, 0});
  PHX_CHECK(batch.vertex_ids[0] == 4 && batch.vertex_ids[1] == -1);
  PHX_CHECK(batch.item_status[1] == PHX_MC_INVALID_INPUT);
  PHX_CHECK(batch.triangles == result.triangles);
  PHX_CHECK(batch.constrained == result.constrained);
}

void explicit_edge_refusals_preserve_state() {
  for (const std::vector<int32_t> edge :
       {std::vector<int32_t>{0, 2}, {1, 3}, {-1, 2}, {0, 4}, {2, 2}}) {
    const Patch patch = square(edge == std::vector<int32_t>{0, 2});
    const Outcome result = reconnect(patch, {0.5, 0.5}, {0.5, 0.5, 0.0}, 0.0, 0,
                                     1024, 1 << 20, {}, edge);
    PHX_CHECK(result.status == PHX_MC_OK);
    PHX_CHECK(result.vertex_ids[0] == -1);
    PHX_CHECK(result.item_status[0] ==
              (edge == std::vector<int32_t>{0, 2} ? PHX_MC_CONSTRAINT_INTERSECTION
                                                  : PHX_MC_INVALID_INPUT));
    PHX_CHECK(result.triangles == patch.triangles);
    PHX_CHECK(result.constrained == patch.constrained);
  }
  for (const std::vector<double> point :
       {std::vector<double>{1.1, 0.5}, {0.0, 0.0}}) {
    const Patch patch = square();
    const Outcome result = reconnect(patch, point, {point[0], point[1], 0.0}, 0.0, 0,
                                     1024, 1 << 20, {}, {0, 2});
    PHX_CHECK(result.item_status[0] == PHX_MC_DEGENERATE_INPUT);
    PHX_CHECK(result.triangles == patch.triangles);
    PHX_CHECK(result.constrained == patch.constrained);
  }
  const Patch patch = square();
  const Outcome opposed = reconnect(patch, {0.5, 0.5}, {0.5, 0.5, 0.0}, 0.0, 0,
                                    1024, 1 << 20, {0.0, 0.0, -1.0}, {0, 2});
  PHX_CHECK(opposed.item_status[0] == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(opposed.triangles == patch.triangles);
  PHX_CHECK(opposed.constrained == patch.constrained);
}

}  // namespace

int main() {
  planar_insertion_is_delaunay();
  physical_criterion_selects_diagonal();
  constrained_diagonal_is_preserved();
  edge_split_and_refusals();
  budgets_refuse_without_invalidating();
  invalid_triangulations_are_refused();
  curved_patch_keeps_orientation();
  opposed_source_normal_is_refused();
  explicit_paired_cavity();
  explicit_edge_refusals_preserve_state();
  return phx::mc::test::finish("test_surface_mesh");
}
