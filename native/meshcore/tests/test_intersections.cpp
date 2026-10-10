//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact contact classification: every contact class and degeneracy with
// hand-derived answers, identity-based adjacency, bounded constructions, ABI
// statuses, and invariance of the classification under input exchange and
// vertex relabeling on an integer lattice rich in degeneracies.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "intersections.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"

namespace {

using phx::mc::Contact;
using phx::mc::kFeatureInterior;

using Triangle = std::array<double, 9>;
using Segment = std::array<double, 6>;

const Triangle kBase = {0, 0, 0, 1, 0, 0, 0, 1, 0};

Contact triangles(const Triangle& a, const Triangle& b, const int64_t* a_ids = nullptr,
                  const int64_t* b_ids = nullptr, int32_t* status = nullptr) {
  const double* const ta[3] = {a.data(), a.data() + 3, a.data() + 6};
  const double* const tb[3] = {b.data(), b.data() + 3, b.data() + 6};
  Contact contact;
  const int32_t result = phx::mc::intersect_triangles(ta, tb, a_ids, b_ids, contact);
  if (status != nullptr) {
    *status = result;
  } else {
    PHX_CHECK(result == PHX_MC_OK);
  }
  return contact;
}

Contact segment_triangle(const Segment& s, const Triangle& t, const int64_t* s_ids = nullptr,
                         const int64_t* t_ids = nullptr, int32_t* status = nullptr) {
  const double* const segment[2] = {s.data(), s.data() + 3};
  const double* const triangle[3] = {t.data(), t.data() + 3, t.data() + 6};
  Contact contact;
  const int32_t result =
      phx::mc::intersect_segment_triangle(segment, triangle, s_ids, t_ids, contact);
  if (status != nullptr) {
    *status = result;
  } else {
    PHX_CHECK(result == PHX_MC_OK);
  }
  return contact;
}

bool has_point(const Contact& contact, double x, double y, double z, int8_t first,
               int8_t second, double tolerance = 0.0) {
  for (int k = 0; k < contact.count; ++k) {
    const phx::mc::ContactPoint& p = contact.points[k];
    const double slack = p.bound + tolerance;
    if (std::fabs(p.x[0] - x) <= slack && std::fabs(p.x[1] - y) <= slack &&
        std::fabs(p.x[2] - z) <= slack && p.first_feature == first &&
        p.second_feature == second) {
      return true;
    }
  }
  return false;
}

void test_transversal_classes() {
  Contact c = triangles(kBase, {0, 0, 1, 1, 0, 1, 0, 1, 1});
  PHX_CHECK(c.kind == PHX_MC_DISJOINT && c.count == 0);
  // Separated by 2^-100 only: exact, not a contact.
  c = triangles(kBase, {0.25, 0.25, 0x1p-100, 1, 1, 1, 0, 1, 1});
  PHX_CHECK(c.kind == PHX_MC_DISJOINT);

  c = triangles(kBase, {0, 0, 0, -1, 0, 1, 0, -1, 1});
  PHX_CHECK(c.kind == PHX_MC_SHARED_VERTEX && c.count == 1);
  PHX_CHECK(has_point(c, 0, 0, 0, 0, 0));
  PHX_CHECK(c.points[0].bound == 0.0);

  c = triangles(kBase, {0, 0, 0, 1, 0, 0, 0, 0, 1});
  PHX_CHECK(c.kind == PHX_MC_SHARED_EDGE && c.count == 2);
  PHX_CHECK(has_point(c, 0, 0, 0, 0, 0) && has_point(c, 1, 0, 0, 1, 1));

  // Vertical triangle x = 0.2 through the interior of the base.
  c = triangles(kBase, {0.2, -1, -1, 0.2, -1, 1, 0.2, 2, 0});
  PHX_CHECK(c.kind == PHX_MC_CROSSING && c.count == 2);
  PHX_CHECK(has_point(c, 0.2, 0, 0, 5, kFeatureInterior, 1e-16));
  PHX_CHECK(has_point(c, 0.2, 1.0 - 0.2, 0, 3, kFeatureInterior, 1e-16));
  PHX_CHECK(c.points[0].bound < 1e-14 && c.points[1].bound < 1e-14);

  // A vertex of the second triangle on the interior of the base.
  c = triangles(kBase, {0.25, 0.25, 0, 0, 0, 1, 1, 0, 1});
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && c.count == 1);
  PHX_CHECK(has_point(c, 0.25, 0.25, 0, kFeatureInterior, 0));

  // An edge of the second triangle lying in the base (a fin).
  c = triangles(kBase, {0.125, 0.125, 0, 0.5, 0.125, 0, 0.25, 0.125, 1});
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && c.count == 2);
  PHX_CHECK(has_point(c, 0.125, 0.125, 0, kFeatureInterior, 0));
  PHX_CHECK(has_point(c, 0.5, 0.125, 0, kFeatureInterior, 1));

  // A vertex on an edge (T-junction) from above.
  c = triangles(kBase, {0.5, 0, 0, 0, -1, 1, 1, -1, 1});
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && c.count == 1);
  PHX_CHECK(has_point(c, 0.5, 0, 0, 5, 0));

  // Edge through edge: the second triangle's vertical edge passes through the
  // base edge y = 0 and the triangle leaves the base plane along y < 0.
  c = triangles(kBase, {0.5, 0, 1, 0.5, 0, -1, 0.5, -2, 0});
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && c.count == 1);
  PHX_CHECK(has_point(c, 0.5, 0, 0, 5, 5));
}

void test_coplanar_classes() {
  Contact c = triangles(kBase, {2, 0, 0, 3, 0, 0, 2, 1, 0});
  PHX_CHECK(c.kind == PHX_MC_DISJOINT);

  c = triangles(kBase, {0, 0, 0, -1, 0, 0, 0, -1, 0});
  PHX_CHECK(c.kind == PHX_MC_SHARED_VERTEX && c.count == 1);

  c = triangles(kBase, {0, 0, 0, 1, 0, 0, 0, -1, 0});
  PHX_CHECK(c.kind == PHX_MC_SHARED_EDGE && c.count == 2);

  // Folded pair sharing an edge on the same side.
  c = triangles(kBase, {0, 0, 0, 1, 0, 0, 0.25, 0.25, 0});
  PHX_CHECK(c.kind == PHX_MC_COPLANAR_OVERLAP && c.count == 3);
  PHX_CHECK(has_point(c, 0.25, 0.25, 0, kFeatureInterior, 2));

  c = triangles(kBase, {0, 1, 0, 0, 0, 0, 1, 0, 0});
  PHX_CHECK(c.kind == PHX_MC_COINCIDENT && c.count == 3);

  // Partial collinear edge overlap, triangles on opposite sides.
  c = triangles(kBase, {0.5, 0, 0, 2, 0, 0, 1, -1, 0});
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && c.count == 2);
  PHX_CHECK(has_point(c, 0.5, 0, 0, 5, 0) && has_point(c, 1, 0, 0, 1, 5));

  // T-junction in the plane.
  c = triangles(kBase, {0.5, 0, 0, 1, -1, 0, 0, -1, 0});
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && c.count == 1);
  PHX_CHECK(has_point(c, 0.5, 0, 0, 5, 0));

  // Star of David: a hexagonal overlap, counterclockwise about +z.
  const Triangle up = {0, 0, 0, 3, 0, 0, 1.5, 3, 0};
  const Triangle down = {0, 2, 0, 1.5, -1, 0, 3, 2, 0};
  c = triangles(up, down);
  PHX_CHECK(c.kind == PHX_MC_COPLANAR_OVERLAP && c.count == 6);
  double area = 0.0;
  for (int k = 0; k < c.count; ++k) {
    const double* p = c.points[k].x;
    const double* q = c.points[(k + 1) % c.count].x;
    area += p[0] * q[1] - p[1] * q[0];
    PHX_CHECK(c.points[k].first_feature >= 3 && c.points[k].first_feature < 6);
    PHX_CHECK(c.points[k].second_feature >= 3 && c.points[k].second_feature < 6);
  }
  PHX_CHECK(area > 0.0);

  // Contained triangle: the polygon is the inner triangle itself.
  c = triangles({0.25, 0.25, 0, 0.5, 0.25, 0, 0.25, 0.5, 0}, kBase);
  PHX_CHECK(c.kind == PHX_MC_COPLANAR_OVERLAP && c.count == 3);
  PHX_CHECK(has_point(c, 0.25, 0.25, 0, 0, kFeatureInterior));

  // Coplanar in a tilted plane whose projection needs an exact axis choice.
  const Triangle tilted = {0, 0, 0, 1, 0, 1, 0, 1, 1};
  c = triangles(tilted, {0.25, 0.25, 0.5, 2, 0.25, 2.25, 0.25, 2, 2.25});
  PHX_CHECK(c.kind == PHX_MC_COPLANAR_OVERLAP && c.count == 3);
}

void test_identity_adjacency() {
  const Triangle b = {0, 0, 0, -1, 0, 1, 0, -1, 1};
  const int64_t a_ids[3] = {10, 11, 12};
  const int64_t shared_ids[3] = {10, 20, 21};
  const int64_t distinct_ids[3] = {30, 20, 21};
  Contact c = triangles(kBase, b, a_ids, shared_ids);
  PHX_CHECK(c.kind == PHX_MC_SHARED_VERTEX);
  // Coincident positions without shared identity are an illegal contact.
  c = triangles(kBase, b, a_ids, distinct_ids);
  PHX_CHECK(c.kind == PHX_MC_TOUCHING);
  int32_t status = PHX_MC_OK;
  const int64_t wrong_ids[3] = {11, 20, 21};  // id 11 names (1, 0, 0), not the origin
  c = triangles(kBase, b, a_ids, wrong_ids, &status);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  const int64_t repeated[3] = {10, 10, 12};
  c = triangles(kBase, b, repeated, shared_ids, &status);
  PHX_CHECK(status == PHX_MC_INVALID_INPUT);
  const Triangle flat = {0, 0, 0, 1, 1, 1, 2, 2, 2};
  c = triangles(kBase, flat, nullptr, nullptr, &status);
  PHX_CHECK(status == PHX_MC_DEGENERATE_INPUT);
}

void test_segment_classes() {
  Contact c = segment_triangle({0.25, 0.25, -1, 0.25, 0.25, 1}, kBase);
  PHX_CHECK(c.kind == PHX_MC_CROSSING && c.count == 1);
  PHX_CHECK(has_point(c, 0.25, 0.25, 0, kFeatureInterior, kFeatureInterior));
  c = segment_triangle({0.25, 0.25, 0, 0.25, 0.25, 1}, kBase);
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && has_point(c, 0.25, 0.25, 0, 0, kFeatureInterior));
  c = segment_triangle({0, 0, 1, 0, 0, 0}, kBase);
  PHX_CHECK(c.kind == PHX_MC_SHARED_VERTEX && has_point(c, 0, 0, 0, 1, 0));
  c = segment_triangle({1, 0, -1, 1, 0, 1}, kBase);
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && has_point(c, 1, 0, 0, kFeatureInterior, 1));
  c = segment_triangle({0.5, 0.5, -1, 0.5, 0.5, 1}, kBase);
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && has_point(c, 0.5, 0.5, 0, kFeatureInterior, 3));
  c = segment_triangle({0.75, 0.75, -1, 0.75, 0.75, 1}, kBase);
  PHX_CHECK(c.kind == PHX_MC_DISJOINT);
  // Coplanar: an edge, a partial edge, the interior, a vertex touch.
  c = segment_triangle({1, 0, 0, 0, 1, 0}, kBase);
  PHX_CHECK(c.kind == PHX_MC_SHARED_EDGE && c.count == 2);
  c = segment_triangle({0.5, 0, 0, 2, 0, 0}, kBase);
  PHX_CHECK(c.kind == PHX_MC_TOUCHING && c.count == 2);
  PHX_CHECK(has_point(c, 0.5, 0, 0, 0, 5) && has_point(c, 1, 0, 0, kFeatureInterior, 1));
  c = segment_triangle({-1, 0.25, 0, 2, 0.25, 0}, kBase);
  PHX_CHECK(c.kind == PHX_MC_COPLANAR_OVERLAP && c.count == 2);
  PHX_CHECK(has_point(c, 0, 0.25, 0, kFeatureInterior, 4, 1e-16));
  PHX_CHECK(has_point(c, 0.75, 0.25, 0, kFeatureInterior, 3, 1e-16));
  c = segment_triangle({1, 0, 0, 2, -1, 0}, kBase);
  PHX_CHECK(c.kind == PHX_MC_SHARED_VERTEX && c.count == 1);
  const int64_t segment_ids[2] = {7, 8};
  const int64_t triangle_ids[3] = {1, 2, 3};
  c = segment_triangle({1, 0, 0, 2, -1, 0}, kBase, segment_ids, triangle_ids);
  PHX_CHECK(c.kind == PHX_MC_TOUCHING);
  int32_t status = PHX_MC_OK;
  c = segment_triangle({1, 0, 0, 1, 0, 0}, kBase, nullptr, nullptr, &status);
  PHX_CHECK(status == PHX_MC_DEGENERATE_INPUT);
}

void test_point_locations() {
  const double* const t[3] = {kBase.data(), kBase.data() + 3, kBase.data() + 6};
  const double points[][3] = {{0.25, 0.25, 0}, {0.5, 0, 0}, {0, 1, 0},
                              {2, 2, 0},       {0.25, 0.25, 1}, {0.25, 0.25, -0x1p-110}};
  const int8_t expected_features[] = {kFeatureInterior, 5, 2, -1, -1, -1};
  const int expected_sides[] = {0, 0, 0, 0, 1, -1};
  for (int k = 0; k < 6; ++k) {
    int side = 9;
    PHX_CHECK(phx::mc::locate_on_triangle(points[k], t, side) == expected_features[k]);
    PHX_CHECK(side == expected_sides[k]);
  }
}

// Integer lattice triangles exercise shared vertices, collinear edges and
// coplanar configurations densely.  The class and point count must not depend
// on the order of the inputs or the labeling of their vertices, and the
// contact points of (A, B) and (B, A) must agree within their bounds.
Triangle relabel(const Triangle& t, int rotation, bool reverse) {
  Triangle result{};
  for (int k = 0; k < 3; ++k) {
    int source = (k + rotation) % 3;
    if (reverse) {
      source = (3 - k + rotation) % 3;
    }
    std::copy_n(t.data() + 3 * source, 3, result.data() + 3 * k);
  }
  return result;
}

bool same_points(const Contact& left, const Contact& right) {
  for (int i = 0; i < left.count; ++i) {
    bool found = false;
    for (int j = 0; j < right.count && !found; ++j) {
      const double slack = left.points[i].bound + right.points[j].bound;
      found = std::fabs(left.points[i].x[0] - right.points[j].x[0]) <= slack &&
              std::fabs(left.points[i].x[1] - right.points[j].x[1]) <= slack &&
              std::fabs(left.points[i].x[2] - right.points[j].x[2]) <= slack &&
              left.points[i].first_feature == right.points[j].second_feature &&
              left.points[i].second_feature == right.points[j].first_feature;
    }
    if (!found) {
      return false;
    }
  }
  return left.count == right.count;
}

void test_lattice_invariance() {
  std::uint64_t state = 2026;
  auto coordinate = [&]() {
    state = phx::mc::splitmix64(state);
    return static_cast<double>(static_cast<int>(state % 3)) - 1.0;
  };
  int classes[7] = {0, 0, 0, 0, 0, 0, 0};
  int mismatches = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    Triangle a{};
    Triangle b{};
    for (double& x : a) {
      x = coordinate();
    }
    for (double& x : b) {
      x = coordinate();
    }
    // Planar pairs are frequent: flatten both into z = 0 every fourth trial.
    if (trial % 4 == 0) {
      a[2] = a[5] = a[8] = b[2] = b[5] = b[8] = 0.0;
    }
    if (phx::mc::collinear3d(a.data(), a.data() + 3, a.data() + 6) ||
        phx::mc::collinear3d(b.data(), b.data() + 3, b.data() + 6)) {
      continue;
    }
    const Contact reference = triangles(a, b);
    ++classes[reference.kind];
    const Contact swapped = triangles(b, a);
    const bool swap_ok = swapped.kind == reference.kind && same_points(reference, swapped);
    bool relabel_ok = true;
    for (int rotation = 0; rotation < 3; ++rotation) {
      for (int reverse = 0; reverse < 2; ++reverse) {
        const Contact other =
            triangles(relabel(a, rotation, reverse != 0), relabel(b, 2 - rotation, reverse == 0));
        relabel_ok = relabel_ok && other.kind == reference.kind && other.count == reference.count;
      }
    }
    if (!swap_ok || !relabel_ok) {
      ++mismatches;
    }
  }
  PHX_CHECK(mismatches == 0);
  for (int kind = 0; kind < 7; ++kind) {
    PHX_CHECK(classes[kind] > 0);
  }
}

void test_abi_batches() {
  const Triangle second = {0.2, -1, -1, 0.2, -1, 1, 0.2, 2, 0};
  double first_batch[27];
  double second_batch[27];
  std::copy(kBase.begin(), kBase.end(), first_batch);
  std::copy(second.begin(), second.end(), second_batch);
  std::copy(kBase.begin(), kBase.end(), first_batch + 9);
  std::copy(kBase.begin(), kBase.end(), second_batch + 9);
  std::copy(kBase.begin(), kBase.end(), first_batch + 18);
  std::copy(kBase.begin(), kBase.end(), second_batch + 18);
  second_batch[18] = std::numeric_limits<double>::quiet_NaN();
  int8_t classes[3] = {9, 9, 9};
  int32_t counts[3] = {9, 9, 9};
  double points[3 * 6 * 3];
  double bounds[3 * 6];
  int8_t features[3 * 6 * 2];
  int32_t status[3] = {9, 9, 9};
  PHX_CHECK(phx_mc_triangle_intersections(3, first_batch, second_batch, nullptr, nullptr, classes,
                                          counts, points, bounds, features,
                                          status) == PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_OK && classes[0] == PHX_MC_CROSSING && counts[0] == 2);
  PHX_CHECK(status[1] == PHX_MC_OK && classes[1] == PHX_MC_COINCIDENT && counts[1] == 3);
  PHX_CHECK(status[2] == PHX_MC_NONFINITE_INPUT && classes[2] == -1 && counts[2] == 0);
  PHX_CHECK(features[2 * 2] == -1 && points[3 * 2] == 0.0);
  // Classification only.
  PHX_CHECK(phx_mc_triangle_intersections(2, first_batch, second_batch, nullptr, nullptr, classes,
                                          nullptr, nullptr, nullptr, nullptr,
                                          status) == PHX_MC_OK);
  // Partial construction outputs, one-sided ids and negative counts are refused.
  PHX_CHECK(phx_mc_triangle_intersections(1, first_batch, second_batch, nullptr, nullptr, classes,
                                          counts, nullptr, bounds, features,
                                          status) == PHX_MC_INVALID_ARGUMENT);
  const int64_t ids[3] = {0, 1, 2};
  PHX_CHECK(phx_mc_triangle_intersections(1, first_batch, second_batch, ids, nullptr, classes,
                                          nullptr, nullptr, nullptr, nullptr,
                                          status) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_triangle_intersections(-1, first_batch, second_batch, nullptr, nullptr,
                                          classes, nullptr, nullptr, nullptr, nullptr,
                                          status) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_triangle_intersections(int64_t{1} << 60, first_batch, second_batch, nullptr,
                                          nullptr, classes, nullptr, nullptr, nullptr, nullptr,
                                          status) == PHX_MC_INVALID_ARGUMENT);

  const double segment[6] = {0.25, 0.25, -1, 0.25, 0.25, 0x1p130};
  PHX_CHECK(phx_mc_segment_triangle_intersections(1, segment, kBase.data(), nullptr, nullptr,
                                                  classes, counts, points, bounds, features,
                                                  status) == PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_RANGE_ERROR && classes[0] == -1);

  const double point[3] = {0.5, 0.5, 0};
  int8_t sides[1] = {9};
  int8_t located[1] = {9};
  PHX_CHECK(phx_mc_point_triangle_locations(1, point, kBase.data(), sides, located, status) ==
            PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_OK && sides[0] == 0 && located[0] == 3);
}

}  // namespace

int main() {
  test_transversal_classes();
  test_coplanar_classes();
  test_identity_adjacency();
  test_segment_classes();
  test_point_locations();
  test_lattice_invariance();
  test_abi_batches();
  return phx::mc::test::finish("test_intersections");
}
