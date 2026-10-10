//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

#include "bounded_memory.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "spatial_sort.hpp"

namespace {

using phx::mc::splitmix64;

double uniform(std::uint64_t& state) {
  state = splitmix64(state);
  return static_cast<double>(state >> 11) * 0x1p-53;
}

bool near_relative(double actual, double expected, double relative) {
  return std::fabs(actual - expected) <= relative * std::max(std::fabs(expected), 1e-300);
}

// ------------------------------------------------------------------ 2D

struct Area {
  double area = -1.0;
  double moment[2] = {0.0, 0.0};
  int32_t status = -1;
};

Area polygons(const std::vector<double>& a, const std::vector<double>& b) {
  const int32_t na = static_cast<int32_t>(a.size() / 2);
  const int32_t nb = static_cast<int32_t>(b.size() / 2);
  Area result;
  const int32_t call = phx_mc_polygon_intersection_moments(
      1, na, a.data(), &na, nb, b.data(), &nb, &result.area, result.moment, &result.status);
  PHX_CHECK(call == PHX_MC_OK);
  return result;
}

void check_area(const Area& r, double area, double mx, double my, double tolerance = 1e-15) {
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK_NEAR(r.area, area, tolerance);
  PHX_CHECK_NEAR(r.moment[0], mx, tolerance);
  PHX_CHECK_NEAR(r.moment[1], my, tolerance);
}

const std::vector<double> kSquare = {0, 0, 1, 0, 1, 1, 0, 1};

void test_polygon_overlaps() {
  check_area(polygons(kSquare, kSquare), 1.0, 0.5, 0.5);
  check_area(polygons(kSquare, {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5}), 0.25, 0.1875, 0.1875);
  // Contained triangle: area 1/32, centroid (1/3, 1/3).
  const std::vector<double> small = {0.25, 0.25, 0.5, 0.25, 0.25, 0.5};
  check_area(polygons(kSquare, small), 1.0 / 32.0, 1.0 / 96.0, 1.0 / 96.0);
  check_area(polygons(small, kSquare), 1.0 / 32.0, 1.0 / 96.0, 1.0 / 96.0);
  // Square minus the corner triangle (1, 0.5), (1, 1), (0.5, 1).
  const double corner_moment = 0.125 * 2.5 / 3.0;
  check_area(polygons(kSquare, {0, 0, 1.5, 0, 0, 1.5}), 0.875, 0.5 - corner_moment,
             0.5 - corner_moment);
  // Rectangles sharing the lines y = 0 and y = 1.
  check_area(polygons({0, 0, 2, 0, 2, 1, 0, 1}, {1, 0, 3, 0, 3, 1, 1, 1}), 1.0, 1.5, 0.5);
  // Triangles sharing the line y = 0 with partial edge overlap: triangle
  // (1, 0), (2, 0), (1.5, 0.5).
  check_area(polygons({0, 0, 2, 0, 1, 1}, {1, 0, 3, 0, 2, 1}), 0.25, 0.25 * 1.5, 0.25 / 6.0);
}

void test_polygon_contacts() {
  check_area(polygons(kSquare, {1, 0, 2, 0, 2, 1, 1, 1}), 0.0, 0.0, 0.0);
  check_area(polygons(kSquare, {1, 1, 2, 1, 2, 2, 1, 2}), 0.0, 0.0, 0.0);
  check_area(polygons(kSquare, {0.5, 1, 1, 2, 0, 2}), 0.0, 0.0, 0.0);
  check_area(polygons(kSquare, {5, 5, 6, 5, 6, 6}), 0.0, 0.0, 0.0);
}

void test_polygon_normalization() {
  const std::vector<double> shifted = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5};
  // Clockwise, collinear midpoints, repeated consecutive vertices and a
  // repeated closing vertex.
  const std::vector<double> messy = {0, 0, 0, 0.5, 0, 1, 0, 1, 1,   1,
                                     1, 0.5, 1, 0, 0.5, 0, 0, 0, 0, 0};
  check_area(polygons(messy, shifted), 0.25, 0.1875, 0.1875);
  check_area(polygons(shifted, messy), 0.25, 0.1875, 0.1875);

  const double pi = 3.14159265358979323846;
  std::vector<double> star;
  for (int k = 0; k < 5; ++k) {
    const double angle = 2.0 * pi * static_cast<double>((2 * k) % 5) / 5.0;
    star.push_back(std::cos(angle));
    star.push_back(std::sin(angle));
  }
  PHX_CHECK(polygons(star, kSquare).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(polygons(kSquare, star).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(polygons({0, 0, 2, 1, 0, 2, 1, 1}, kSquare).status == PHX_MC_INVALID_INPUT);
  // Degenerate: too few vertices, all collinear, all identical.
  PHX_CHECK(polygons({0, 0, 1, 1}, kSquare).status == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(polygons({0, 0, 1, 1, 2, 2, 3, 3}, kSquare).status == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(polygons({1, 1, 1, 1, 1, 1}, kSquare).status == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(polygons({}, kSquare).status == PHX_MC_DEGENERATE_INPUT);
  const Area failed = polygons(kSquare, {0, 0, 1, 1, 2, 2});
  PHX_CHECK(failed.status == PHX_MC_DEGENERATE_INPUT && failed.area == 0.0);
  // Non-finite and out-of-domain coordinates.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PHX_CHECK(polygons({0, 0, nan, 0, 0, 1}, kSquare).status == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(polygons({0, 0, 1e300, 0, 0, 1}, kSquare).status == PHX_MC_RANGE_ERROR);
  PHX_CHECK(polygons({0, 0, 1e-300, 0, 0, 1}, kSquare).status == PHX_MC_RANGE_ERROR);
}

// Random convex polygons: symmetry, cyclic rotation and reversal of the
// input order change the measures by at most rounding.
void test_polygon_invariance() {
  std::uint64_t state = 11;
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<double> poly[2];
    for (auto& p : poly) {
      const int n = 3 + static_cast<int>(uniform(state) * 8.0);
      const double cx = uniform(state), cy = uniform(state), radius = 0.3 + uniform(state);
      std::vector<double> angles;
      for (int k = 0; k < n; ++k) {
        angles.push_back(uniform(state) * 6.283185307179586);
      }
      std::sort(angles.begin(), angles.end());
      for (const double angle : angles) {
        p.push_back(cx + radius * std::cos(angle));
        p.push_back(cy + radius * std::sin(angle));
      }
    }
    const Area base = polygons(poly[0], poly[1]);
    if (base.status != PHX_MC_OK) {
      // Nearly collinear random angles may produce a degenerate polygon.
      PHX_CHECK(base.status == PHX_MC_DEGENERATE_INPUT);
      continue;
    }
    const Area swapped = polygons(poly[1], poly[0]);
    std::vector<double> reversed;
    for (std::size_t k = poly[0].size(); k >= 2; k -= 2) {
      reversed.push_back(poly[0][k - 2]);
      reversed.push_back(poly[0][k - 1]);
    }
    std::vector<double> rotated(poly[1].begin() + 2, poly[1].end());
    rotated.push_back(poly[1][0]);
    rotated.push_back(poly[1][1]);
    const Area variant = polygons(reversed, rotated);
    for (const Area* other : {&swapped, &variant}) {
      PHX_CHECK(other->status == PHX_MC_OK);
      PHX_CHECK(std::fabs(other->area - base.area) <= 1e-15 * (1.0 + base.area));
      PHX_CHECK(std::fabs(other->moment[0] - base.moment[0]) <= 1e-14);
      PHX_CHECK(std::fabs(other->moment[1] - base.moment[1]) <= 1e-14);
    }
  }
}

void test_polygon_call_status() {
  double area = 0.0, moment[2] = {0.0, 0.0};
  int32_t status = 0;
  const int32_t three = 3, five = 5;
  const double tri[6] = {0, 0, 1, 0, 0, 1};
  PHX_CHECK(phx_mc_polygon_intersection_moments(-1, 3, tri, &three, 3, tri, &three, &area, moment,
                                                &status) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_polygon_intersection_moments(1, 3, tri, &five, 3, tri, &three, &area, moment,
                                                &status) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_polygon_intersection_moments(1, 3, tri, &three, 3, nullptr, &three, &area,
                                                moment, &status) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_polygon_intersection_moments(0, 3, nullptr, nullptr, 3, nullptr, nullptr,
                                                nullptr, nullptr, nullptr) == PHX_MC_OK);
  // Batched rows with padding.
  const double batch_a[16] = {0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 2, 0, 0, 2, 99, 99};
  const int32_t counts_a[2] = {4, 3};
  const int32_t counts_b[2] = {3, 3};
  double areas[2], moments[4];
  int32_t statuses[2];
  const int32_t too_many[2] = {3, 5};
  const double batch_b4[16] = {0, 0, 1, 0, 0, 1, 7, 7, 0, 0, 1, 0, 0, 1, 7, 7};
  PHX_CHECK(phx_mc_polygon_intersection_moments(2, 4, batch_a, counts_a, 4, batch_b4, too_many,
                                                areas, moments, statuses) == PHX_MC_INVALID_ARGUMENT);
  // Row 0: square with triangle; row 1: triangle inside the larger one.
  PHX_CHECK(phx_mc_polygon_intersection_moments(2, 4, batch_a, counts_a, 4, batch_b4, counts_b,
                                                areas, moments, statuses) == PHX_MC_OK);
  PHX_CHECK(statuses[0] == PHX_MC_OK && statuses[1] == PHX_MC_OK);
  PHX_CHECK_NEAR(areas[0], 0.5, 1e-15);
  PHX_CHECK_NEAR(areas[1], 0.5, 1e-15);
  PHX_CHECK_NEAR(moments[2], 0.5 / 3.0, 1e-15);
}

struct Cell2 {
  std::vector<double> vertices;
  std::vector<int32_t> labels;
  int32_t count = 0;
  double area = -1.0;
  double moment[2] = {0.0, 0.0};
  int32_t status = -1;
};

Cell2 box2(const double* lower, const double* upper, const std::vector<double>& normals,
           const std::vector<double>& offsets, int32_t capacity = 64) {
  Cell2 cell;
  cell.vertices.assign(2 * static_cast<std::size_t>(capacity), 0.0);
  cell.labels.assign(capacity, 0);
  const int32_t planes = static_cast<int32_t>(offsets.size());
  const int32_t call = phx_mc_clip_box_halfplanes(
      1, lower, upper, planes, normals.data(), offsets.data(), &planes, capacity,
      cell.vertices.data(), cell.labels.data(), &cell.count, &cell.area, cell.moment, &cell.status);
  PHX_CHECK(call == PHX_MC_OK);
  return cell;
}

// Every labeled edge lies on its line (explicit plane or box side).
void check_cell2_labels(const Cell2& cell, const double* lower, const double* upper,
                        const std::vector<double>& normals, const std::vector<double>& offsets) {
  for (int32_t k = 0; k < cell.count; ++k) {
    const int32_t label = cell.labels[k];
    for (const int32_t v : {k, (k + 1) % cell.count}) {
      const double* x = cell.vertices.data() + 2 * v;
      if (label >= 0) {
        const double residual = normals[2 * label] * x[0] + normals[2 * label + 1] * x[1] -
                                offsets[label];
        PHX_CHECK(std::fabs(residual) <= 1e-12);
      } else {
        const int axis = (-label - 1) / 2;
        const int side = (-label - 1) % 2;
        PHX_CHECK(std::fabs(x[axis] - (side != 0 ? upper[axis] : lower[axis])) <= 1e-12);
      }
    }
  }
}

void test_box_halfplanes() {
  const double lower[2] = {1.0, 2.0};
  const double upper[2] = {3.0, 5.0};
  const Cell2 box = box2(lower, upper, {}, {});
  PHX_CHECK(box.status == PHX_MC_OK && box.count == 4);
  PHX_CHECK_NEAR(box.area, 6.0, 1e-14);
  PHX_CHECK_NEAR(box.moment[0], 12.0, 1e-13);
  PHX_CHECK_NEAR(box.moment[1], 21.0, 1e-13);
  std::vector<int32_t> labels(box.labels.begin(), box.labels.begin() + 4);
  std::sort(labels.begin(), labels.end());
  PHX_CHECK((labels == std::vector<int32_t>{-4, -3, -2, -1}));
  check_cell2_labels(box, lower, upper, {}, {});

  // x <= 2 keeps the left half; the cut edge carries label 0.
  const std::vector<double> normals = {1.0, 0.0};
  const std::vector<double> offsets = {2.0};
  const Cell2 half = box2(lower, upper, normals, offsets);
  PHX_CHECK(half.status == PHX_MC_OK && half.count == 4);
  PHX_CHECK_NEAR(half.area, 3.0, 1e-14);
  PHX_CHECK_NEAR(half.moment[0], 4.5, 1e-13);
  PHX_CHECK(std::count(half.labels.begin(), half.labels.begin() + 4, 0) == 1);
  check_cell2_labels(half, lower, upper, normals, offsets);

  // Corner cut x + y <= 7.5 creates a pentagon; capacity 4 is exceeded.
  const std::vector<double> corner_n = {1.0, 1.0};
  const std::vector<double> corner_h = {7.5};
  const Cell2 pentagon = box2(lower, upper, corner_n, corner_h);
  PHX_CHECK(pentagon.status == PHX_MC_OK && pentagon.count == 5);
  PHX_CHECK_NEAR(pentagon.area, 6.0 - 0.125, 1e-14);
  const Cell2 overflow = box2(lower, upper, corner_n, corner_h, 4);
  PHX_CHECK(overflow.status == PHX_MC_CAPACITY_EXCEEDED && overflow.count == 0 &&
            overflow.area == 0.0);

  // Empty, zero-measure contact, zero normal, non-finite, range.
  const Cell2 empty = box2(lower, upper, {1.0, 0.0}, {0.0});
  PHX_CHECK(empty.status == PHX_MC_OK && empty.count == 0 && empty.area == 0.0);
  const Cell2 contact = box2(lower, upper, {1.0, 0.0}, {1.0});
  PHX_CHECK(contact.status == PHX_MC_OK && contact.count == 0 && contact.area == 0.0);
  PHX_CHECK(box2(lower, upper, {0.0, 0.0}, {1.0}).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(box2(lower, upper, {1.0, std::nan("")}, {1.0}).status == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(box2(lower, upper, {1.0, 0.0}, {1e80}).status == PHX_MC_RANGE_ERROR);
  PHX_CHECK(box2(lower, upper, {1.0, 0.0}, {1e50}).status == PHX_MC_OK);

  // Call status: inverted box, capacity, negative count.
  double v[8], a = 0, m[2];
  int32_t l[4], c = 0, s = 0, zero = 0;
  const double bad_upper[2] = {3.0, 2.0};
  PHX_CHECK(phx_mc_clip_box_halfplanes(1, lower, bad_upper, 0, nullptr, nullptr, &zero, 4, v, l,
                                       &c, &a, m, &s) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_clip_box_halfplanes(1, lower, upper, 0, nullptr, nullptr, &zero, 0, v, l, &c,
                                       &a, m, &s) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_clip_box_halfplanes(-1, lower, upper, 0, nullptr, nullptr, &zero, 4, v, l, &c,
                                       &a, m, &s) == PHX_MC_INVALID_ARGUMENT);
  const double inf_upper[2] = {INFINITY, 5.0};
  PHX_CHECK(phx_mc_clip_box_halfplanes(1, lower, inf_upper, 0, nullptr, nullptr, &zero, 4, v, l,
                                       &c, &a, m, &s) == PHX_MC_NONFINITE_INPUT);
}

// Bisector halfplanes {x : (g_j - g_i) . x <= (|g_j|^2 - |g_i|^2) / 2} of
// dyadic generators (exact offsets) partition the box.
std::vector<double> bisectors(const std::vector<double>& generators, int dimension, int i,
                              std::vector<double>& offsets) {
  std::vector<double> normals;
  offsets.clear();
  const int count = static_cast<int>(generators.size()) / dimension;
  double gi2 = 0.0;
  for (int k = 0; k < dimension; ++k) {
    gi2 += generators[dimension * i + k] * generators[dimension * i + k];
  }
  for (int j = 0; j < count; ++j) {
    if (j == i) {
      continue;
    }
    double gj2 = 0.0;
    for (int k = 0; k < dimension; ++k) {
      normals.push_back(generators[dimension * j + k] - generators[dimension * i + k]);
      gj2 += generators[dimension * j + k] * generators[dimension * j + k];
    }
    offsets.push_back(0.5 * (gj2 - gi2));
  }
  return normals;
}

std::vector<double> dyadic_generators(int count, int dimension, std::uint64_t seed) {
  std::vector<double> generators;
  while (static_cast<int>(generators.size()) < count * dimension) {
    std::vector<double> g;
    for (int k = 0; k < dimension; ++k) {
      g.push_back(std::floor(uniform(seed) * 63.0 + 1.0) / 64.0);
    }
    bool duplicate = false;
    for (std::size_t j = 0; j < generators.size(); j += dimension) {
      duplicate = duplicate || std::equal(g.begin(), g.end(), generators.begin() + j);
    }
    if (!duplicate) {
      generators.insert(generators.end(), g.begin(), g.end());
    }
  }
  return generators;
}

void test_box_halfplanes_voronoi() {
  const double lower[2] = {0.0, 0.0};
  const double upper[2] = {1.0, 1.0};
  for (int configuration = 0; configuration < 3; ++configuration) {
    std::vector<double> generators;
    if (configuration == 0) {
      generators = {0.25, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.5, 0.5};
    } else {
      generators = dyadic_generators(6 + 2 * configuration, 2, 100 + configuration);
    }
    const int count = static_cast<int>(generators.size() / 2);
    double total = 0.0, mx = 0.0, my = 0.0;
    for (int i = 0; i < count; ++i) {
      std::vector<double> offsets;
      const std::vector<double> normals = bisectors(generators, 2, i, offsets);
      const Cell2 cell = box2(lower, upper, normals, offsets);
      PHX_CHECK(cell.status == PHX_MC_OK && cell.count >= 3);
      PHX_CHECK(cell.area > 0.0);
      check_cell2_labels(cell, lower, upper, normals, offsets);
      total += cell.area;
      mx += cell.moment[0];
      my += cell.moment[1];
    }
    PHX_CHECK(near_relative(total, 1.0, 1e-12));
    PHX_CHECK(near_relative(mx, 0.5, 1e-12));
    PHX_CHECK(near_relative(my, 0.5, 1e-12));
  }
}

// ------------------------------------------------------------------ 3D

struct Volume {
  double volume = -1.0;
  double moment[3] = {0.0, 0.0, 0.0};
  int32_t status = -1;
};

Volume tetrahedra(const std::array<double, 12>& a, const std::array<double, 12>& b,
                  int32_t capacity = 48) {
  Volume result;
  const int32_t call = phx_mc_tetrahedron_intersection_moments(
      1, a.data(), b.data(), capacity, &result.volume, result.moment, &result.status);
  PHX_CHECK(call == PHX_MC_OK);
  return result;
}

Volume halfspaces(const std::vector<double>& normals, const std::vector<double>& offsets,
                  const std::array<double, 12>& tet, int32_t capacity = 64) {
  Volume result;
  const int32_t planes = static_cast<int32_t>(offsets.size());
  const int32_t call =
      phx_mc_polyhedron_clip_moments(1, planes, normals.data(), offsets.data(), &planes,
                                     tet.data(), capacity, &result.volume, result.moment,
                                     &result.status);
  PHX_CHECK(call == PHX_MC_OK);
  return result;
}

void check_volume(const Volume& r, double volume, double mx, double my, double mz,
                  double tolerance = 1e-15) {
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK_NEAR(r.volume, volume, tolerance);
  PHX_CHECK_NEAR(r.moment[0], mx, tolerance);
  PHX_CHECK_NEAR(r.moment[1], my, tolerance);
  PHX_CHECK_NEAR(r.moment[2], mz, tolerance);
}

std::array<double, 12> affine(const std::array<double, 12>& tet, double scale, double dx,
                              double dy, double dz) {
  std::array<double, 12> out{};
  for (int v = 0; v < 4; ++v) {
    out[3 * v] = scale * tet[3 * v] + dx;
    out[3 * v + 1] = scale * tet[3 * v + 1] + dy;
    out[3 * v + 2] = scale * tet[3 * v + 2] + dz;
  }
  return out;
}

const std::array<double, 12> kUnitTet = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};

void test_tetrahedron_overlaps() {
  check_volume(tetrahedra(kUnitTet, kUnitTet), 1.0 / 6.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0);
  const auto half = affine(kUnitTet, 0.5, 0, 0, 0);
  check_volume(tetrahedra(kUnitTet, half), 1.0 / 48.0, 1.0 / 384.0, 1.0 / 384.0, 1.0 / 384.0);
  check_volume(tetrahedra(half, kUnitTet), 1.0 / 48.0, 1.0 / 384.0, 1.0 / 384.0, 1.0 / 384.0);
  const auto shifted = affine(kUnitTet, 1.0, 0.25, 0.25, 0.25);
  const double m = 0.3125 / 384.0;
  check_volume(tetrahedra(kUnitTet, shifted), 1.0 / 384.0, m, m, m);
  // Coplanar faces y = 0 and z = 0 with overlap: legs 1/2 from (1/2, 0, 0).
  const auto along_x = affine(kUnitTet, 1.0, 0.5, 0, 0);
  check_volume(tetrahedra(kUnitTet, along_x), 1.0 / 48.0, 0.625 / 48.0, 0.125 / 48.0,
               0.125 / 48.0);
  // T intersected with its point reflection about (0.3, 0.3, 0.3):
  // {x, y, z <= 0.6, x + y + z >= 0.8} within T, volume 0.32 / 6.
  const auto reflected = affine(kUnitTet, -1.0, 0.6, 0.6, 0.6);
  const Volume octa = tetrahedra(kUnitTet, reflected);
  PHX_CHECK(octa.status == PHX_MC_OK);
  PHX_CHECK_NEAR(octa.volume, 0.32 / 6.0, 1e-15);
  const Volume via_planes = halfspaces({1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -1, -1},
                                       {0.6, 0.6, 0.6, -0.8}, kUnitTet);
  PHX_CHECK(via_planes.status == PHX_MC_OK);
  PHX_CHECK_NEAR(via_planes.volume, octa.volume, 1e-15);
  for (int k = 0; k < 3; ++k) {
    PHX_CHECK_NEAR(via_planes.moment[k], octa.moment[k], 1e-15);
    PHX_CHECK_NEAR(octa.moment[k], octa.volume * 0.3, 1e-15);  // symmetric about (0.3, 0.3, 0.3)
  }
  // Six vertices after the first clip exceed a capacity of five.
  const Volume overflow = tetrahedra(kUnitTet, reflected, 5);
  PHX_CHECK(overflow.status == PHX_MC_CAPACITY_EXCEEDED && overflow.volume == 0.0);
}

void test_tetrahedron_contacts() {
  // Shared face x = 0.
  check_volume(tetrahedra(kUnitTet, {0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1}), 0, 0, 0, 0);
  // Shared edge on the x axis.
  check_volume(tetrahedra(kUnitTet, {0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1}), 0, 0, 0, 0);
  // Shared vertex at the origin.
  check_volume(tetrahedra(kUnitTet, affine(kUnitTet, -1.0, 0, 0, 0)), 0, 0, 0, 0);
  // Vertex (1/4, 1/4, 1/2) of B on the face x + y + z = 1 of A.
  check_volume(tetrahedra(kUnitTet, affine(kUnitTet, 1.0, 0.25, 0.25, 0.5)), 0, 0, 0, 0);
  // Disjoint.
  check_volume(tetrahedra(kUnitTet, affine(kUnitTet, 1.0, 5, 5, 5)), 0, 0, 0, 0);
}

void test_tetrahedron_statuses() {
  const std::array<double, 12> flat = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};
  PHX_CHECK(tetrahedra(flat, kUnitTet).status == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(tetrahedra(kUnitTet, flat).status == PHX_MC_DEGENERATE_INPUT);
  auto nan_tet = kUnitTet;
  nan_tet[4] = std::nan("");
  PHX_CHECK(tetrahedra(kUnitTet, nan_tet).status == PHX_MC_NONFINITE_INPUT);
  auto huge = kUnitTet;
  huge[4] = 1e40;
  PHX_CHECK(tetrahedra(huge, kUnitTet).status == PHX_MC_RANGE_ERROR);
  double v = 0, m[3];
  int32_t s = 0;
  PHX_CHECK(phx_mc_tetrahedron_intersection_moments(1, kUnitTet.data(), kUnitTet.data(), 0, &v,
                                                    m, &s) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_tetrahedron_intersection_moments(1, kUnitTet.data(), nullptr, 48, &v, m,
                                                    &s) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_tetrahedron_intersection_moments(-2, kUnitTet.data(), kUnitTet.data(), 48, &v,
                                                    m, &s) == PHX_MC_INVALID_ARGUMENT);
}

std::array<double, 12> permuted(const std::array<double, 12>& tet, const std::array<int, 4>& p) {
  std::array<double, 12> out{};
  for (int v = 0; v < 4; ++v) {
    for (int k = 0; k < 3; ++k) {
      out[3 * v + k] = tet[3 * p[v] + k];
    }
  }
  return out;
}

// All vertex permutations (both orientations) of both tetrahedra and the
// argument swap give the same measures up to rounding: 1e-15 relative for
// well-sized overlaps of unit-scale tetrahedra (A: perturbed unit
// tetrahedron, B: perturbed reflection, polytopes with up to 8 faces);
// small overlaps carry the absolute rounding of unit-scale coordinates.
void test_tetrahedron_invariance() {
  std::uint64_t state = 5;
  int well_sized = 0;
  for (int trial = 0; trial < 40; ++trial) {
    std::array<double, 12> a{}, b{};
    for (int i = 0; i < 12; ++i) {
      a[i] = kUnitTet[i] + 0.4 * (uniform(state) - 0.5);
      b[i] = 0.6 - kUnitTet[i] + 0.4 * (uniform(state) - 0.5);
    }
    const Volume base = tetrahedra(a, b);
    PHX_CHECK(base.status == PHX_MC_OK && base.volume > 0.0);
    const bool strict = base.volume > 0.02;
    well_sized += strict;
    const double tolerance = strict ? 2e-15 * base.volume : 3e-17;
    std::array<int, 4> p = {0, 1, 2, 3};
    int permutation = 0;
    do {
      const auto pa = permuted(a, p);
      std::array<int, 4> q = {p[(permutation + 1) % 4], p[(permutation + 2) % 4],
                              p[(permutation + 3) % 4], p[permutation % 4]};
      const auto pb = permuted(b, q);
      for (const Volume& r : {tetrahedra(pa, pb), tetrahedra(pb, pa)}) {
        PHX_CHECK(r.status == PHX_MC_OK);
        PHX_CHECK(std::fabs(r.volume - base.volume) <= tolerance);
        for (int k = 0; k < 3; ++k) {
          PHX_CHECK(std::fabs(r.moment[k] - base.moment[k]) <=
                    tolerance * (1.0 + std::fabs(base.moment[k]) / base.volume));
        }
      }
      ++permutation;
    } while (std::next_permutation(p.begin(), p.end()));
  }
  PHX_CHECK(well_sized >= 30);
}

void test_tetrahedron_halfspaces() {
  // Plane x = 1/4 through the centroid: the part x >= 1/4 is the tetrahedron
  // scaled by 3/4 about (1, 0, 0).
  const double upper_volume = 27.0 / 384.0;
  const double upper_centroid[3] = {0.4375, 0.1875, 0.1875};
  const Volume right = halfspaces({-1, 0, 0}, {-0.25}, kUnitTet);
  check_volume(right, upper_volume, upper_volume * upper_centroid[0],
               upper_volume * upper_centroid[1], upper_volume * upper_centroid[2]);
  const Volume left = halfspaces({1, 0, 0}, {0.25}, kUnitTet);
  check_volume(left, 37.0 / 384.0, 1.0 / 24.0 - right.moment[0], 1.0 / 24.0 - right.moment[1],
               1.0 / 24.0 - right.moment[2]);
  // Either orientation of the tetrahedron.
  const auto flipped = permuted(kUnitTet, {1, 0, 2, 3});
  check_volume(halfspaces({-1, 0, 0}, {-0.25}, flipped), upper_volume, right.moment[0],
               right.moment[1], right.moment[2]);
  // No planes: the tetrahedron.  Redundant and duplicated planes.
  check_volume(halfspaces({}, {}, kUnitTet), 1.0 / 6.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0);
  check_volume(halfspaces({1, 1, 1, 2, 2, 2, -1, 0, 0}, {1.0, 2.0, 0.0}, kUnitTet), 1.0 / 6.0,
               1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0);
  // Opposite halfspaces through a face: zero measure.
  check_volume(halfspaces({1, 0, 0}, {0.0}, kUnitTet), 0, 0, 0, 0);
  // A slab 1/4 <= z <= 1/2 cut by the explicit planes in both orders.
  const double slab = (std::pow(0.75, 3) - std::pow(0.5, 3)) / 6.0;
  const Volume slab1 = halfspaces({0, 0, -1, 0, 0, 1}, {-0.25, 0.5}, kUnitTet);
  const Volume slab2 = halfspaces({0, 0, 1, 0, 0, -1}, {0.5, -0.25}, kUnitTet);
  PHX_CHECK_NEAR(slab1.volume, slab, 1e-15);
  PHX_CHECK_NEAR(slab2.volume, slab, 1e-15);
  // Three clips meeting at an interior point of the tetrahedron create
  // PLANES3 vertices; the corner cube [0, 0.2]^3 lies inside.
  check_volume(halfspaces({1, 0, 0, 0, 1, 0, 0, 0, 1}, {0.2, 0.2, 0.2}, kUnitTet), 0.008, 0.0008,
               0.0008, 0.0008, 1e-16);
  // Statuses.
  PHX_CHECK(halfspaces({0, 0, 0}, {1.0}, kUnitTet).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(halfspaces({1, 0, 0}, {std::nan("")}, kUnitTet).status == PHX_MC_NONFINITE_INPUT);
  PHX_CHECK(halfspaces({1, 0, 0}, {1e80}, kUnitTet).status == PHX_MC_RANGE_ERROR);
  PHX_CHECK(halfspaces({1e40, 0, 0}, {1.0}, kUnitTet).status == PHX_MC_RANGE_ERROR);
  const std::array<double, 12> flat = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};
  PHX_CHECK(halfspaces({1, 0, 0}, {0.5}, flat).status == PHX_MC_DEGENERATE_INPUT);
  // Cutting a corner leaves 6 vertices: capacity 5 is exceeded.
  PHX_CHECK(halfspaces({1, 0, 0}, {0.5}, kUnitTet, 5).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(halfspaces({1, 0, 0}, {0.5}, kUnitTet, 6).status == PHX_MC_OK);
  PHX_CHECK(halfspaces({}, {}, kUnitTet, 3).status == PHX_MC_CAPACITY_EXCEEDED);
}

struct Cell3 {
  std::vector<double> vertices;
  std::vector<int32_t> offsets;
  std::vector<int32_t> labels;
  std::vector<int32_t> loops;
  int32_t vertex_count = 0;
  int32_t face_count = 0;
  double volume = -1.0;
  double moment[3] = {0.0, 0.0, 0.0};
  int32_t status = -1;
};

Cell3 box3(const double* lower, const double* upper, const std::vector<double>& normals,
           const std::vector<double>& offsets, int32_t vertex_capacity = 128,
           int32_t face_capacity = 64, int32_t face_vertex_capacity = 512) {
  Cell3 cell;
  cell.vertices.assign(3 * static_cast<std::size_t>(vertex_capacity), 0.0);
  cell.offsets.assign(static_cast<std::size_t>(face_capacity) + 1, -1);
  cell.labels.assign(face_capacity, 0);
  cell.loops.assign(face_vertex_capacity, -1);
  const int32_t planes = static_cast<int32_t>(offsets.size());
  const int32_t call = phx_mc_clip_box_halfspaces(
      1, lower, upper, planes, normals.data(), offsets.data(), &planes, vertex_capacity,
      face_capacity, face_vertex_capacity, cell.vertices.data(), &cell.vertex_count,
      cell.offsets.data(), cell.labels.data(), cell.loops.data(), &cell.face_count, &cell.volume,
      cell.moment, &cell.status);
  PHX_CHECK(call == PHX_MC_OK);
  return cell;
}

// Structural checks of a nonempty box cell: sorted labels, loops starting at
// their smallest vertex, closed 2-manifold with Euler characteristic 2, every
// vertex on its labeled planes, outward (counterclockwise) loops.
void check_cell3(const Cell3& cell, const double* lower, const double* upper,
                 const std::vector<double>& normals, const std::vector<double>& offsets) {
  PHX_CHECK(cell.offsets[0] == 0);
  std::vector<std::array<int32_t, 2>> edges;
  std::vector<int> used(cell.vertex_count, 0);
  for (int32_t f = 0; f < cell.face_count; ++f) {
    if (f > 0) {
      PHX_CHECK(cell.labels[f - 1] < cell.labels[f]);
    }
    const int32_t begin = cell.offsets[f];
    const int32_t end = cell.offsets[f + 1];
    PHX_CHECK(end - begin >= 3);
    const int32_t label = cell.labels[f];
    double outward[3] = {0.0, 0.0, 0.0};
    if (label >= 0) {
      for (int k = 0; k < 3; ++k) {
        outward[k] = normals[3 * label + k];
      }
    } else {
      outward[(-label - 1) / 2] = (-label - 1) % 2 != 0 ? 1.0 : -1.0;
    }
    double newell[3] = {0.0, 0.0, 0.0};
    for (int32_t i = begin; i < end; ++i) {
      const int32_t v = cell.loops[i];
      const int32_t w = cell.loops[i + 1 < end ? i + 1 : begin];
      PHX_CHECK(v >= 0 && v < cell.vertex_count);
      PHX_CHECK(cell.loops[begin] <= v);
      used[v] = 1;
      edges.push_back({v, w});
      const double* x = cell.vertices.data() + 3 * v;
      const double* y = cell.vertices.data() + 3 * w;
      newell[0] += (x[1] - y[1]) * (x[2] + y[2]);
      newell[1] += (x[2] - y[2]) * (x[0] + y[0]);
      newell[2] += (x[0] - y[0]) * (x[1] + y[1]);
      if (label >= 0) {
        const double residual = normals[3 * label] * x[0] + normals[3 * label + 1] * x[1] +
                                normals[3 * label + 2] * x[2] - offsets[label];
        PHX_CHECK(std::fabs(residual) <= 1e-12);
      } else {
        const int axis = (-label - 1) / 2;
        const double bound = (-label - 1) % 2 != 0 ? upper[axis] : lower[axis];
        PHX_CHECK(std::fabs(x[axis] - bound) <= 1e-12);
      }
    }
    PHX_CHECK(newell[0] * outward[0] + newell[1] * outward[1] + newell[2] * outward[2] > 0.0);
  }
  // Every directed edge has its reverse exactly once.
  std::vector<std::array<int32_t, 2>> sorted = edges;
  std::sort(sorted.begin(), sorted.end());
  PHX_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
  for (const auto& e : edges) {
    PHX_CHECK(std::binary_search(sorted.begin(), sorted.end(), std::array<int32_t, 2>{e[1], e[0]}));
  }
  PHX_CHECK(std::count(used.begin(), used.end(), 1) == cell.vertex_count);
  const int64_t euler = static_cast<int64_t>(cell.vertex_count) -
                        static_cast<int64_t>(edges.size() / 2) + cell.face_count;
  PHX_CHECK(euler == 2);
}

void test_box_halfspaces() {
  const double lower[3] = {1.0, 2.0, -1.0};
  const double upper[3] = {2.0, 4.0, 3.0};
  const Cell3 box = box3(lower, upper, {}, {});
  PHX_CHECK(box.status == PHX_MC_OK && box.vertex_count == 8 && box.face_count == 6);
  PHX_CHECK((std::vector<int32_t>(box.labels.begin(), box.labels.begin() + 6) ==
             std::vector<int32_t>{-6, -5, -4, -3, -2, -1}));
  PHX_CHECK(box.offsets[6] == 24);
  PHX_CHECK_NEAR(box.volume, 8.0, 1e-14);
  PHX_CHECK_NEAR(box.moment[0], 12.0, 1e-13);
  PHX_CHECK_NEAR(box.moment[1], 24.0, 1e-13);
  PHX_CHECK_NEAR(box.moment[2], 8.0, 1e-13);
  check_cell3(box, lower, upper, {}, {});

  // Oblique cut x + y + z <= 6.5 through the box: all six sides survive.
  const std::vector<double> n = {1, 1, 1};
  const std::vector<double> h = {6.5};
  const Cell3 cut = box3(lower, upper, n, h);
  PHX_CHECK(cut.status == PHX_MC_OK && cut.face_count == 7 && cut.labels[6] == 0);
  check_cell3(cut, lower, upper, n, h);
  const Cell3 rest = box3(lower, upper, {-1, -1, -1}, {-6.5});
  PHX_CHECK(rest.status == PHX_MC_OK);
  check_cell3(rest, lower, upper, {-1, -1, -1}, {-6.5});
  PHX_CHECK(near_relative(cut.volume + rest.volume, 8.0, 1e-14));
  for (int k = 0; k < 3; ++k) {
    PHX_CHECK(near_relative(cut.moment[k] + rest.moment[k], box.moment[k], 1e-14));
  }

  // Capacities: vertices, faces, face vertices.
  PHX_CHECK(box3(lower, upper, {}, {}, 7).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(box3(lower, upper, {}, {}, 8, 5).status == PHX_MC_CAPACITY_EXCEEDED);
  const Cell3 tight = box3(lower, upper, {}, {}, 8, 6, 23);
  PHX_CHECK(tight.status == PHX_MC_CAPACITY_EXCEEDED && tight.face_count == 0 &&
            tight.vertex_count == 0 && tight.volume == 0.0);
  PHX_CHECK(box3(lower, upper, {}, {}, 8, 6, 24).status == PHX_MC_OK);
  // Cutting the corner (2, 4, 3) adds three vertices and removes one.
  PHX_CHECK(box3(lower, upper, {1, 1, 1}, {8.5}, 9).status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(box3(lower, upper, {1, 1, 1}, {8.5}, 10).status == PHX_MC_OK);

  // Empty and zero-measure results.
  const Cell3 empty = box3(lower, upper, {1, 0, 0}, {0.0});
  PHX_CHECK(empty.status == PHX_MC_OK && empty.vertex_count == 0 && empty.face_count == 0 &&
            empty.offsets[0] == 0 && empty.volume == 0.0);
  PHX_CHECK(box3(lower, upper, {1, 0, 0}, {1.0}).volume == 0.0);
  // Statuses.
  PHX_CHECK(box3(lower, upper, {0, 0, 0}, {1.0}).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(box3(lower, upper, {1, 0, INFINITY}, {1.0}).status == PHX_MC_NONFINITE_INPUT);
  const double flat_upper[3] = {2.0, 2.0, 3.0};
  double v[24], vol = 0, m[3];
  int32_t vc = 0, fo[7], fl[6], fv[24], fc = 0, s = 0, zero = 0;
  PHX_CHECK(phx_mc_clip_box_halfspaces(1, lower, flat_upper, 0, nullptr, nullptr, &zero, 8, 6, 24,
                                       v, &vc, fo, fl, fv, &fc, &vol, m,
                                       &s) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_clip_box_halfspaces(1, lower, upper, 0, nullptr, nullptr, &zero, 8, 0, 24, v,
                                       &vc, fo, fl, fv, &fc, &vol, m,
                                       &s) == PHX_MC_INVALID_ARGUMENT);
}

void test_box_halfspaces_voronoi() {
  const double lower[3] = {0.0, 0.0, 0.0};
  const double upper[3] = {1.0, 1.0, 1.0};
  for (int configuration = 0; configuration < 4; ++configuration) {
    std::vector<double> generators;
    if (configuration == 0) {
      // Lattice: cells are the eight cubes [g - 1/4, g + 1/4]; many planes
      // coincide and pass through existing vertices.
      for (int k = 0; k < 8; ++k) {
        generators.push_back((k & 1) != 0 ? 0.75 : 0.25);
        generators.push_back((k & 2) != 0 ? 0.75 : 0.25);
        generators.push_back((k & 4) != 0 ? 0.75 : 0.25);
      }
    } else if (configuration == 1) {
      // Lattice perturbed by 2^-26 (squares stay exact): nearly coincident
      // and nearly concurrent planes.
      std::uint64_t seed = 77;
      for (int k = 0; k < 8; ++k) {
        for (int axis = 0; axis < 3; ++axis) {
          const double base = ((k >> axis) & 1) != 0 ? 0.75 : 0.25;
          const double offset = std::floor(uniform(seed) * 5.0) - 2.0;
          generators.push_back(base + offset * 0x1p-26);
        }
      }
    } else {
      generators = dyadic_generators(5 + 2 * configuration, 3, 300 + configuration);
    }
    const int count = static_cast<int>(generators.size() / 3);
    double total = 0.0, moment[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < count; ++i) {
      std::vector<double> offsets;
      const std::vector<double> normals = bisectors(generators, 3, i, offsets);
      const Cell3 cell = box3(lower, upper, normals, offsets);
      PHX_CHECK(cell.status == PHX_MC_OK);
      PHX_CHECK(cell.volume > 0.0);
      check_cell3(cell, lower, upper, normals, offsets);
      if (configuration == 0) {
        PHX_CHECK_NEAR(cell.volume, 0.125, 1e-15);
        PHX_CHECK(cell.vertex_count == 8 && cell.face_count == 6);
        for (int k = 0; k < 3; ++k) {
          PHX_CHECK_NEAR(cell.moment[k], 0.125 * generators[3 * i + k], 1e-15);
        }
      }
      total += cell.volume;
      for (int k = 0; k < 3; ++k) {
        moment[k] += cell.moment[k];
      }
    }
    PHX_CHECK(near_relative(total, 1.0, 1e-12));
    for (const double mk : moment) {
      PHX_CHECK(near_relative(mk, 0.5, 1e-12));
    }
  }
}

// Planes one ulp apart are classified exactly: the slab between x + y <= h
// and x + y >= h' is empty for h' >= h and has positive volume for h' < h,
// also when the vertices on the first plane are PLANES3 vertices.
void test_ulp_planes() {
  const double lower[3] = {0.0, 0.0, 0.0};
  const double upper[3] = {1.0, 1.0, 1.0};
  const double h = 0.7;
  const double above = std::nextafter(h, 2.0);
  const double below = std::nextafter(h, 0.0);
  for (const bool with_tilt : {false, true}) {
    std::vector<double> normals = {1, 1, 0};
    std::vector<double> offsets = {h};
    if (with_tilt) {
      // x - z <= 0.3 first, so the edges on (x + y = h) include PLANES3 vertices.
      normals = {1, 0, -1, 1, 1, 0};
      offsets = {0.3, h};
    }
    normals.insert(normals.end(), {-1, -1, 0});
    for (const double second : {above, h, below}) {
      std::vector<double> o = offsets;
      o.push_back(-second);
      const Cell3 cell = box3(lower, upper, normals, o);
      PHX_CHECK(cell.status == PHX_MC_OK);
      if (second == below) {
        PHX_CHECK(cell.face_count > 0 && std::fabs(cell.volume) < 1e-15);
      } else {
        PHX_CHECK(cell.volume == 0.0 && cell.face_count == 0);
      }
    }
    // Same in 2D with (x + y <= h) and (x + y >= h').
    const double lower2[2] = {0.0, 0.0};
    const double upper2[2] = {1.0, 1.0};
    for (const double second : {above, h, below}) {
      const Cell2 cell = box2(lower2, upper2, {1, 1, -1, -1}, {h, -second});
      PHX_CHECK(cell.status == PHX_MC_OK);
      PHX_CHECK((cell.count > 0) == (second == below));
    }
  }
  // Tetrahedra B = A reflected across z = 0 and lifted by delta: a sliver of
  // thickness delta for delta = +1 ulp, face contact for 0, empty for -1 ulp.
  const double ulp = std::nextafter(1.0, 2.0) - 1.0;
  const std::array<double, 12> mirrored = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1};
  const Volume lifted = tetrahedra(kUnitTet, affine(mirrored, 1.0, 0, 0, ulp));
  PHX_CHECK(lifted.status == PHX_MC_OK && lifted.volume > 0.0 && lifted.volume < 1e-15);
  PHX_CHECK_NEAR(lifted.volume, 0.5 * ulp, 1e-30);
  const Volume touching = tetrahedra(kUnitTet, mirrored);
  PHX_CHECK(touching.status == PHX_MC_OK && touching.volume == 0.0);
  const Volume dropped = tetrahedra(kUnitTet, affine(mirrored, 1.0, 0, 0, -ulp));
  PHX_CHECK(dropped.status == PHX_MC_OK && dropped.volume == 0.0);
}

// Determinism: repeated calls produce bitwise identical outputs.
void test_determinism() {
  std::uint64_t state = 3;
  std::array<double, 12> a{}, b{};
  for (double& x : a) {
    x = uniform(state);
  }
  for (double& x : b) {
    x = 0.1 + uniform(state);
  }
  const Volume first = tetrahedra(a, b);
  const Volume second = tetrahedra(a, b);
  PHX_CHECK(first.volume == second.volume && first.moment[0] == second.moment[0] &&
            first.moment[1] == second.moment[1] && first.moment[2] == second.moment[2]);
}

// Simplex partitions reproduce the reported moments term by term, match the
// moments-only entry points bitwise, and report capacity refusal.
void test_intersection_simplices() {
  std::uint64_t state = 17;
  for (int trial = 0; trial < 50; ++trial) {
    std::array<double, 12> a{}, b{};
    for (double& x : a) {
      x = uniform(state);
    }
    for (double& x : b) {
      x = 0.2 + uniform(state);
    }
    std::vector<double> simplices(20 * 12);
    int32_t simplex_count = -1;
    double volume = -1.0;
    double moment[3] = {0.0, 0.0, 0.0};
    int32_t status = -1;
    PHX_CHECK(phx_mc_tetrahedron_intersection_simplices(1, a.data(), b.data(), 48, 20,
                                                        simplices.data(), &simplex_count, &volume,
                                                        moment, &status) == PHX_MC_OK);
    PHX_CHECK(status == PHX_MC_OK);
    const Volume reference = tetrahedra(a, b);
    PHX_CHECK(reference.volume == volume && reference.moment[0] == moment[0]);
    PHX_CHECK((volume == 0.0) == (simplex_count == 0));
    double total = 0.0;
    double centroid_sum[3] = {0.0, 0.0, 0.0};
    for (int32_t s = 0; s < simplex_count; ++s) {
      const double* p = simplices.data() + 12 * s;
      const double u[3] = {p[3] - p[0], p[4] - p[1], p[5] - p[2]};
      const double v[3] = {p[6] - p[0], p[7] - p[1], p[8] - p[2]};
      const double w[3] = {p[9] - p[0], p[10] - p[1], p[11] - p[2]};
      const double det = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
                         u[2] * (v[0] * w[1] - v[1] * w[0]);
      PHX_CHECK(det >= -1e-15);
      total += det / 6.0;
      for (int k = 0; k < 3; ++k) {
        centroid_sum[k] += det / 6.0 * 0.25 * (p[k] + p[3 + k] + p[6 + k] + p[9 + k]);
      }
    }
    PHX_CHECK_NEAR(total, volume, 1e-14);
    for (int k = 0; k < 3; ++k) {
      PHX_CHECK_NEAR(centroid_sum[k], moment[k], 1e-14);
    }
  }
  // The unit tetrahedron against its point reflection has eight faces.
  const auto reflected = affine(kUnitTet, -1.0, 0.6, 0.6, 0.6);
  std::vector<double> simplices(4 * 12);
  int32_t simplex_count = -1;
  double volume = -1.0;
  double moment[3] = {0.0, 0.0, 0.0};
  int32_t status = -1;
  PHX_CHECK(phx_mc_tetrahedron_intersection_simplices(1, kUnitTet.data(), reflected.data(), 48, 4,
                                                      simplices.data(), &simplex_count, &volume,
                                                      moment, &status) == PHX_MC_OK);
  PHX_CHECK(status == PHX_MC_CAPACITY_EXCEEDED && simplex_count == 0 && volume == 0.0);
  PHX_CHECK(phx_mc_tetrahedron_intersection_simplices(1, kUnitTet.data(), reflected.data(), 48, 0,
                                                      simplices.data(), &simplex_count, &volume,
                                                      moment, &status) ==
            PHX_MC_INVALID_ARGUMENT);

  const std::vector<double> shifted = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5};
  const int32_t four = 4;
  std::vector<double> triangles(8 * 6);
  int32_t triangle_count = -1;
  double area = -1.0;
  double area_moment[2] = {0.0, 0.0};
  PHX_CHECK(phx_mc_polygon_intersection_simplices(1, 4, kSquare.data(), &four, 4, shifted.data(),
                                                  &four, 8, triangles.data(), &triangle_count,
                                                  &area, area_moment, &status) == PHX_MC_OK);
  PHX_CHECK(status == PHX_MC_OK && triangle_count == 4);
  PHX_CHECK_NEAR(area, 0.25, 1e-15);
  double triangle_area = 0.0;
  for (int32_t s = 0; s < triangle_count; ++s) {
    const double* p = triangles.data() + 6 * s;
    const double cross = (p[2] - p[0]) * (p[5] - p[1]) - (p[4] - p[0]) * (p[3] - p[1]);
    PHX_CHECK(cross > 0.0);
    triangle_area += 0.5 * cross;
  }
  PHX_CHECK_NEAR(triangle_area, area, 1e-15);
  PHX_CHECK(phx_mc_polygon_intersection_simplices(1, 4, kSquare.data(), &four, 4, shifted.data(),
                                                  &four, 3, triangles.data(), &triangle_count,
                                                  &area, area_moment, &status) == PHX_MC_OK);
  PHX_CHECK(status == PHX_MC_CAPACITY_EXCEEDED && triangle_count == 0 && area == 0.0);
}

void test_exact_reference_halfplanes() {
  // A positive rational strip whose two x boundaries round to the SAME
  // binary64 value. The scientific intersection must not become edge contact.
  std::array<std::vector<std::uint32_t>, 18> magnitudes = {{
      {1}, {}, {}, {}, {1}, {}, {1}, {1}, {1},
      {0, 0, 1u << 17}, {}, {0xffffffffu, 0xffffffffu, 0xffffu},
      {0, 0, 1u << 17}, {}, {0, 0, 1u << 16}, {}, {4}, {1},
  }};
  const std::array<std::int8_t, 18> signs = {
      -1, 0, 0, 0, -1, 0, 1, 1, 1, -1, 0, -1, 1, 0, 1, 0, 1, 1};
  std::vector<std::uint32_t> words;
  std::array<std::int64_t, 19> offsets{};
  auto pack = [&]() {
    words.clear();
    offsets[0] = 0;
    for (std::size_t index = 0; index < magnitudes.size(); ++index) {
      words.insert(words.end(), magnitudes[index].begin(), magnitudes[index].end());
      offsets[index + 1] = static_cast<std::int64_t>(words.size());
    }
  };
  std::array<std::int32_t, 12> supports{};
  std::array<std::int32_t, 6> labels{};
  std::int32_t count = -1;
  std::int64_t work = -1;
  auto run = [&](std::int64_t memory_limit, std::int64_t work_limit) {
    pack();
    return phx_mc_clip_reference_triangle_exact(
        words.data(), static_cast<std::int64_t>(words.size()), offsets.data(),
        signs.data(), memory_limit, work_limit, 6, supports.data(), labels.data(),
        &count, &work);
  };
  PHX_CHECK(run(1 << 20, 10000) == PHX_MC_OK && count == 4);
  std::array<bool, 4> seen{};
  for (int vertex = 0; vertex < count; ++vertex) {
    const int first = std::min(supports[2 * vertex], supports[2 * vertex + 1]);
    const int second = std::max(supports[2 * vertex], supports[2 * vertex + 1]);
    if (first == 1 && second == 3) seen[0] = true;
    if (first == 1 && second == 4) seen[1] = true;
    if (first == 3 && second == 5) seen[2] = true;
    if (first == 4 && second == 5) seen[3] = true;
    PHX_CHECK(labels[vertex] == 1 || labels[vertex] == 3 ||
              labels[vertex] == 4 || labels[vertex] == 5);
  }
  PHX_CHECK(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));
  // Exact zero-width contact and disjointness are distinct from that strip.
  magnitudes[11] = {0, 0, 1u << 16};
  PHX_CHECK(run(1 << 20, 10000) == PHX_MC_OK && count == 0);
  magnitudes[11] = {1, 0, 1u << 16};
  PHX_CHECK(run(1 << 20, 10000) == PHX_MC_OK && count == 0);
  // Resource refusal publishes no partial polygon or fake work success.
  magnitudes[11] = {0xffffffffu, 0xffffffffu, 0xffffu};
  PHX_CHECK(run(16384, 10000) == PHX_MC_CAPACITY_EXCEEDED && count == 0 && work == 0);
  PHX_CHECK(run(1 << 20, 1) == PHX_MC_CAPACITY_EXCEEDED && count == 0 && work == 0);
  {
    phx::mc::MemoryBudgetWindow parent_memory(16);
    phx::mc::MemoryScope scope(parent_memory.owner());
    PHX_CHECK(run(1 << 20, 10000) == PHX_MC_CAPACITY_EXCEEDED && count == 0);
    PHX_CHECK(parent_memory.evidence()[5] > 0);
    PHX_CHECK(parent_memory.evidence()[1] == 0);
  }
}

#include "reference_tetrahedron_clip_cases.inc"

}  // namespace

int main() {
  test_polygon_overlaps();
  test_polygon_contacts();
  test_polygon_normalization();
  test_polygon_invariance();
  test_polygon_call_status();
  test_box_halfplanes();
  test_exact_reference_halfplanes();
  test_exact_reference_tetrahedron_and_corner_closure();
  test_box_halfplanes_voronoi();
  test_tetrahedron_overlaps();
  test_tetrahedron_contacts();
  test_tetrahedron_statuses();
  test_tetrahedron_invariance();
  test_tetrahedron_halfspaces();
  test_box_halfspaces();
  test_box_halfspaces_voronoi();
  test_ulp_planes();
  test_determinism();
  test_intersection_simplices();
  return phx::mc::test::finish("test_clip");
}
