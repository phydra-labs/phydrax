//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"
#include "wide_integer.hpp"

namespace {

using namespace phx::mc;

double uniform(std::uint64_t& state) {
  state = splitmix64(state);
  return static_cast<double>(state >> 11) * 0x1p-53;
}

// Sign of p q - r s, from exact 256-bit integer products.
int product_difference_sign(std::int64_t p, std::int64_t q, std::int64_t r, std::int64_t s) {
  Int256 left;
  Int256 right;
  Int256 difference;
  PHX_CHECK(checked_multiply(Int256::from_int64(p), Int256::from_int64(q), left));
  PHX_CHECK(checked_multiply(Int256::from_int64(r), Int256::from_int64(s), right));
  PHX_CHECK(checked_add(left, wrapping_negate(right), difference));
  return compare(difference, Int256{});
}

// Near-degenerate grid around the line y = x (Kettner et al. classroom
// example): a = (0.5 + i 2^-53, 0.5 + j 2^-53), b = (12, 12), c = (24, 24).
// The differences are inexact, so the full expansion stage is exercised; the
// reference is exact integer arithmetic on coordinates scaled by 2^53.
void test_orient2d_near_degenerate_grid() {
  const double b[2] = {12.0, 12.0};
  const double c[2] = {24.0, 24.0};
  const std::int64_t scale = std::int64_t{1} << 53;
  const std::int64_t bx = 12 * scale, by = 12 * scale, cx = 24 * scale, cy = 24 * scale;
  int positive = 0, negative = 0, zero = 0;
  for (int i = 0; i < 128; ++i) {
    for (int j = 0; j < 128; ++j) {
      const double a[2] = {0.5 + std::ldexp(static_cast<double>(i), -53),
                           0.5 + std::ldexp(static_cast<double>(j), -53)};
      const std::int64_t ax = scale / 2 + i, ay = scale / 2 + j;
      const int expected = product_difference_sign(bx - ax, cy - ay, by - ay, cx - ax);
      const int actual = orient2d(a, b, c);
      PHX_CHECK(actual == expected);
      PHX_CHECK(orient2d(b, a, c) == -expected);
      PHX_CHECK(orient2d(b, c, a) == expected);
      positive += expected > 0;
      negative += expected < 0;
      zero += expected == 0;
    }
  }
  PHX_CHECK(positive > 0 && negative > 0 && zero > 0);
}

// Four points on the plane x = y spanning 60 binades: the determinant is
// exactly zero while every difference is inexact.  Moving one point by a
// single ulp off the plane must produce the sign of the projected 2D
// orientation of the other three.
void test_orient3d_exact_zero_and_ulp() {
  std::uint64_t state = 7;
  for (int trial = 0; trial < 2000; ++trial) {
    double p[4][3];
    for (auto& point : p) {
      const double scale = std::ldexp(1.0, static_cast<int>(uniform(state) * 80.0) - 40);
      const double t = (uniform(state) - 0.5) * scale;
      point[0] = t;
      point[1] = t;
      point[2] = (uniform(state) - 0.5) * std::ldexp(1.0, static_cast<int>(uniform(state) * 80.0) - 40);
    }
    PHX_CHECK(orient3d(p[0], p[1], p[2], p[3]) == 0);
    const double a_yz[2] = {p[0][1], p[0][2]};
    const double b_yz[2] = {p[1][1], p[1][2]};
    const double c_yz[2] = {p[2][1], p[2][2]};
    const int projected = orient2d(a_yz, b_yz, c_yz);
    double moved[3] = {std::nextafter(p[3][0], INFINITY), p[3][1], p[3][2]};
    PHX_CHECK(orient3d(p[0], p[1], p[2], moved) == projected);
    PHX_CHECK(orient3d(p[1], p[0], p[2], moved) == -projected);
    moved[0] = std::nextafter(p[3][0], -INFINITY);
    PHX_CHECK(orient3d(p[0], p[1], p[2], moved) == -projected);
    PHX_CHECK(orient3d_exact(p[0], p[1], p[2], moved).sign() == -projected);
  }
}

void test_orientation_conventions() {
  const double o[3] = {0.0, 0.0, 0.0};
  const double x[3] = {1.0, 0.0, 0.0};
  const double y[3] = {0.0, 1.0, 0.0};
  const double z[3] = {0.0, 0.0, 1.0};
  PHX_CHECK(orient2d(o, x, y) == 1);
  PHX_CHECK(orient3d(o, x, y, z) == 1);
  PHX_CHECK(orient3d(o, y, x, z) == -1);
  const double inside2[2] = {0.25, 0.25};
  const double outside2[2] = {2.0, 2.0};
  PHX_CHECK(incircle(o, x, y, inside2) == 1);
  PHX_CHECK(incircle(o, x, y, outside2) == -1);
  const double inside3[3] = {0.2, 0.2, 0.2};
  const double outside3[3] = {3.0, 3.0, 3.0};
  PHX_CHECK(insphere(o, x, y, z, inside3) == 1);
  PHX_CHECK(insphere(o, x, y, z, outside3) == -1);
  PHX_CHECK(insphere(o, y, x, z, inside3) == -1);
  // Zero weights reduce the power tests to incircle/insphere.
  PHX_CHECK(power2d(o, x, y, inside2, 0.0, 0.0, 0.0, 0.0) == 1);
  PHX_CHECK(power3d(o, x, y, z, outside3, 0.0, 0.0, 0.0, 0.0, 0.0) == -1);
  // A heavy query point conflicts; a light one does not.
  PHX_CHECK(power2d(o, x, y, outside2, 0.0, 0.0, 0.0, 100.0) == 1);
  PHX_CHECK(power2d(o, x, y, inside2, 0.0, 0.0, 0.0, -100.0) == -1);
  PHX_CHECK(power3d(o, x, y, z, outside3, 0.0, 0.0, 0.0, 0.0, 100.0) == 1);
}

// Integer lattice points on a circle/sphere translated by a mixed-scale offset
// are exactly cocircular/cospherical; one-ulp radial moves decide the sign.
void test_incircle_insphere_exact_zero_and_ulp() {
  const double cx = 0x1p30, cy = 0x1p-10, cz = -0x1p20;
  const double a[2] = {cx + 5.0, cy};
  const double b[2] = {cx, cy + 5.0};
  const double c[2] = {cx - 3.0, cy + 4.0};
  const double d[2] = {cx + 4.0, cy - 3.0};
  PHX_CHECK(incircle(a, b, c, d) == 0);
  const double d_out[2] = {std::nextafter(d[0], INFINITY), d[1]};
  const double d_in[2] = {std::nextafter(d[0], -INFINITY), d[1]};
  PHX_CHECK(incircle(a, b, c, d_out) == -1);
  PHX_CHECK(incircle(a, b, c, d_in) == 1);
  PHX_CHECK(incircle(b, a, c, d_in) == -1);

  const double p0[3] = {cx + 3.0, cy, cz};
  const double p1[3] = {cx, cy + 3.0, cz};
  const double p2[3] = {cx, cy, cz + 3.0};
  const double p3[3] = {cx - 1.0, cy - 2.0, cz - 2.0};
  const double q[3] = {cx + 2.0, cy + 1.0, cz - 2.0};
  const int orientation = orient3d(p0, p1, p2, p3);
  PHX_CHECK(orientation != 0);
  PHX_CHECK(insphere(p0, p1, p2, p3, q) == 0);
  const double q_out[3] = {std::nextafter(q[0], INFINITY), q[1], q[2]};
  const double q_in[3] = {std::nextafter(q[0], -INFINITY), q[1], q[2]};
  PHX_CHECK(insphere(p0, p1, p2, p3, q_out) == -orientation);
  PHX_CHECK(insphere(p0, p1, p2, p3, q_in) == orientation);
  PHX_CHECK(power3d(p0, p1, p2, p3, q_in, 0.0, 0.0, 0.0, 0.0, 0.0) == orientation);
}

void test_sos_orientation() {
  // Collinear and coplanar configurations are resolved to a nonzero sign that
  // is antisymmetric under exchanging (point, id) pairs.
  const double a[3] = {0.0, 0.0, 0.0};
  const double b[3] = {1.0, 1.0, 0.0};
  const double c[3] = {3.0, 3.0, 0.0};
  const double d[3] = {-2.0, -2.0, 0.0};
  const int s = orient2d_sos(a, b, c, 4, 9, 2);
  PHX_CHECK(s != 0);
  PHX_CHECK(orient2d_sos(b, a, c, 9, 4, 2) == -s);
  PHX_CHECK(orient2d_sos(b, c, a, 9, 2, 4) == s);
  // Coincident points are also resolved.
  PHX_CHECK(orient2d_sos(a, a, a, 0, 1, 2) != 0);
  const int t = orient3d_sos(a, b, c, d, 3, 1, 4, 1 + 4);
  PHX_CHECK(t != 0);
  PHX_CHECK(orient3d_sos(b, a, c, d, 1, 3, 4, 5) == -t);
  PHX_CHECK(orient3d_sos(a, b, d, c, 3, 1, 5, 4) == -t);
  // Nondegenerate inputs are unaffected.
  const double e[3] = {0.0, 0.0, 1.0};
  const double f[3] = {1.0, 0.0, 0.0};
  PHX_CHECK(orient3d(a, f, b, e) == 1);
  PHX_CHECK(orient3d_sos(a, f, b, e, 3, 2, 1, 0) == 1);
}

// Cocircular convex quadrilateral: under a valid lifting perturbation exactly
// one diagonal is Delaunay, i.e. the two triangles of one triangulation both
// have their opposite vertex outside.
void test_sos_lifted_consistency() {
  const double p[4][2] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
  const std::int64_t ids[4][4] = {{0, 1, 2, 3}, {3, 2, 1, 0}, {5, 0, 7, 2}, {1, 8, 3, 6}};
  for (const auto& id : ids) {
    const int abc_d = incircle_sos(p[0], p[1], p[2], p[3], id[0], id[1], id[2], id[3]);
    const int acd_b = incircle_sos(p[0], p[2], p[3], p[1], id[0], id[2], id[3], id[1]);
    const int abd_c = incircle_sos(p[0], p[1], p[3], p[2], id[0], id[1], id[3], id[2]);
    const int bcd_a = incircle_sos(p[1], p[2], p[3], p[0], id[1], id[2], id[3], id[0]);
    PHX_CHECK(abc_d != 0);
    PHX_CHECK(abc_d == acd_b);
    PHX_CHECK(abd_c == bcd_a);
    PHX_CHECK(abd_c == -abc_d);
  }
  // Five cospherical points in convex position: the tetrahedra whose fifth
  // point is outside form a triangulation of the hull (half the total
  // unsigned volume of the five tetrahedra).
  const double q[5][3] = {{3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0},
                          {-1.0, -2.0, -2.0}, {2.0, -1.0, 2.0}};
  const std::int64_t qids[3][5] = {{0, 1, 2, 3, 4}, {4, 3, 2, 1, 0}, {9, 2, 7, 5, 1}};
  for (const auto& id : qids) {
    double selected = 0.0;
    double total = 0.0;
    int count = 0;
    for (int omit = 0; omit < 5; ++omit) {
      int v[4];
      int k = 0;
      for (int s = 0; s < 5; ++s) {
        if (s != omit) {
          v[k++] = s;
        }
      }
      if (orient3d(q[v[0]], q[v[1]], q[v[2]], q[v[3]]) < 0) {
        std::swap(v[0], v[1]);
      }
      const double* A = q[v[0]];
      const double* B = q[v[1]];
      const double* C = q[v[2]];
      const double* D = q[v[3]];
      const double volume =
          ((B[0] - A[0]) * ((C[1] - A[1]) * (D[2] - A[2]) - (C[2] - A[2]) * (D[1] - A[1])) -
           (B[1] - A[1]) * ((C[0] - A[0]) * (D[2] - A[2]) - (C[2] - A[2]) * (D[0] - A[0])) +
           (B[2] - A[2]) * ((C[0] - A[0]) * (D[1] - A[1]) - (C[1] - A[1]) * (D[0] - A[0]))) /
          6.0;
      PHX_CHECK(insphere(A, B, C, D, q[omit]) == 0);
      total += std::fabs(volume);
      const int side = insphere_sos(A, B, C, D, q[omit], id[v[0]], id[v[1]], id[v[2]],
                                    id[v[3]], id[omit]);
      PHX_CHECK(side != 0);
      if (side < 0) {
        selected += std::fabs(volume);
        ++count;
      }
    }
    PHX_CHECK(count == 2 || count == 3);
    PHX_CHECK_NEAR(selected, 0.5 * total, 1.0e-12 * total);
  }
}

void test_c_abi_batches() {
  const double a[4] = {0.0, 0.0, 0.0, 0.0};
  const double b[4] = {1.0, 0.0, 1.0, 0.0};
  const double c[4] = {0.0, 1.0, 2.0, 0.0};
  int8_t signs[2] = {9, 9};
  PHX_CHECK(phx_mc_orient2d(2, a, b, c, signs) == PHX_MC_OK);
  PHX_CHECK(signs[0] == 1 && signs[1] == 0);
  const double bad[4] = {0.0, NAN, 0.0, 0.0};
  PHX_CHECK(phx_mc_orient2d(2, a, bad, c, signs) == PHX_MC_NONFINITE_INPUT);
  const double huge[4] = {0x1p121, 0.0, 0.0, 0.0};
  PHX_CHECK(phx_mc_orient2d(2, a, huge, c, signs) == PHX_MC_RANGE_ERROR);
  const std::int64_t ids[6] = {0, 1, 2, 0, 0, 2};
  PHX_CHECK(phx_mc_orient2d_sos(2, a, b, c, ids, signs) == PHX_MC_INVALID_INPUT);
  int32_t low = 0, high = 0;
  phx_mc_exact_domain(&low, &high);
  PHX_CHECK(low == -120 && high == 120);
}

// Exactly degenerate configurations whose coordinate differences are inexact,
// so the adaptive predicates reach their exact stages: (a, b, -a, -b) is
// coplanar with the origin, quarter-turn images of a point are cocircular, and
// cyclic coordinate permutations and negations of a point are cospherical.
struct DegenerateBatches {
  static constexpr std::int64_t kCount = 256;
  std::vector<double> plane[4];
  std::vector<double> circle[4];
  std::vector<double> sphere[5];
  std::vector<std::int64_t> plane_ids;
  std::vector<std::int64_t> circle_ids;
  std::vector<std::int64_t> sphere_ids;
};

DegenerateBatches degenerate_batches() {
  DegenerateBatches batches;
  std::uint64_t state = 0xD1B54A32D192ED03ULL;
  for (std::int64_t item = 0; item < DegenerateBatches::kCount; ++item) {
    double a[3];
    double b[3];
    for (int axis = 0; axis < 3; ++axis) {
      a[axis] = 2.0 * uniform(state) - 1.0;
      b[axis] = 2.0 * uniform(state) - 1.0;
    }
    const double plane[4][3] = {{a[0], a[1], a[2]},
                                {b[0], b[1], b[2]},
                                {-a[0], -a[1], -a[2]},
                                {-b[0], -b[1], -b[2]}};
    const double circle[4][2] = {{a[0], a[1]}, {-a[1], a[0]}, {-a[0], -a[1]}, {a[1], -a[0]}};
    const double sphere[5][3] = {{b[0], b[1], b[2]},
                                 {b[1], b[2], b[0]},
                                 {b[2], b[0], b[1]},
                                 {-b[0], -b[1], -b[2]},
                                 {-b[1], -b[2], -b[0]}};
    for (int row = 0; row < 4; ++row) {
      batches.plane[row].insert(batches.plane[row].end(), plane[row], plane[row] + 3);
      batches.circle[row].insert(batches.circle[row].end(), circle[row], circle[row] + 2);
      batches.plane_ids.push_back(row);
      batches.circle_ids.push_back(row);
    }
    for (int row = 0; row < 5; ++row) {
      batches.sphere[row].insert(batches.sphere[row].end(), sphere[row], sphere[row] + 3);
      batches.sphere_ids.push_back(row);
    }
  }
  return batches;
}

struct DegenerateSigns {
  std::int32_t status[6] = {-1, -1, -1, -1, -1, -1};
  std::vector<std::int8_t> signs[6];
};

DegenerateSigns evaluate_degenerate(const DegenerateBatches& b) {
  constexpr std::int64_t n = DegenerateBatches::kCount;
  DegenerateSigns result;
  for (auto& signs : result.signs) signs.assign(n, 9);
  result.status[0] = phx_mc_orient3d(n, b.plane[0].data(), b.plane[1].data(), b.plane[2].data(),
                                     b.plane[3].data(), result.signs[0].data());
  result.status[1] = phx_mc_incircle(n, b.circle[0].data(), b.circle[1].data(),
                                     b.circle[2].data(), b.circle[3].data(),
                                     result.signs[1].data());
  result.status[2] = phx_mc_insphere(n, b.sphere[0].data(), b.sphere[1].data(),
                                     b.sphere[2].data(), b.sphere[3].data(), b.sphere[4].data(),
                                     result.signs[2].data());
  result.status[3] = phx_mc_orient3d_sos(n, b.plane[0].data(), b.plane[1].data(),
                                         b.plane[2].data(), b.plane[3].data(),
                                         b.plane_ids.data(), result.signs[3].data());
  result.status[4] = phx_mc_incircle_sos(n, b.circle[0].data(), b.circle[1].data(),
                                         b.circle[2].data(), b.circle[3].data(),
                                         b.circle_ids.data(), result.signs[4].data());
  result.status[5] = phx_mc_insphere_sos(n, b.sphere[0].data(), b.sphere[1].data(),
                                         b.sphere[2].data(), b.sphere[3].data(),
                                         b.sphere[4].data(), b.sphere_ids.data(),
                                         result.signs[5].data());
  return result;
}

// Statuses of one out-of-domain subnormal coordinate per predicate entry point.
struct SubnormalStatus {
  std::int32_t status[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
};

SubnormalStatus evaluate_subnormal() {
  const double s = 0x1p-1060;
  const double o2[2] = {0.0, 0.0}, x2[2] = {1.0, 0.0}, y2[2] = {0.0, 1.0}, t2[2] = {s, 0.0};
  const double o3[3] = {0.0, 0.0, 0.0}, x3[3] = {1.0, 0.0, 0.0}, y3[3] = {0.0, 1.0, 0.0},
               z3[3] = {0.0, 0.0, 1.0}, t3[3] = {0.0, 0.0, s};
  const std::int64_t ids3[3] = {0, 1, 2};
  const std::int64_t ids4[4] = {0, 1, 2, 3};
  const std::int64_t ids5[5] = {0, 1, 2, 3, 4};
  std::int8_t sign = 9;
  SubnormalStatus result;
  result.status[0] = phx_mc_orient2d(1, o2, x2, t2, &sign);
  result.status[1] = phx_mc_orient3d(1, o3, x3, y3, t3, &sign);
  result.status[2] = phx_mc_incircle(1, o2, x2, y2, t2, &sign);
  result.status[3] = phx_mc_insphere(1, o3, x3, y3, z3, t3, &sign);
  result.status[4] = phx_mc_orient2d_sos(1, o2, x2, t2, ids3, &sign);
  result.status[5] = phx_mc_orient3d_sos(1, o3, x3, y3, t3, ids4, &sign);
  result.status[6] = phx_mc_incircle_sos(1, o2, x2, y2, t2, ids4, &sign);
  result.status[7] = phx_mc_insphere_sos(1, o3, x3, y3, z3, t3, ids5, &sign);
  return result;
}

// Certified predicates give their round-to-nearest, gradual-underflow decisions
// in every caller mode: exact zeros stay zero under directed rounding, and a
// subnormal coordinate stays outside the exact domain under flush-to-zero.
void test_floating_point_hazards() {
  const DegenerateBatches batches = degenerate_batches();
  const DegenerateSigns reference = evaluate_degenerate(batches);
  for (int k = 0; k < 6; ++k) PHX_CHECK(reference.status[k] == PHX_MC_OK);
  for (int k = 0; k < 3; ++k) {
    for (const std::int8_t sign : reference.signs[k]) PHX_CHECK(sign == 0);
  }
  for (int k = 3; k < 6; ++k) {
    for (const std::int8_t sign : reference.signs[k]) PHX_CHECK(sign == 1 || sign == -1);
  }
  const SubnormalStatus subnormal = evaluate_subnormal();
  for (const std::int32_t status : subnormal.status) PHX_CHECK(status == PHX_MC_RANGE_ERROR);

  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    const DegenerateSigns hazard = evaluate_degenerate(batches);
    for (int k = 0; k < 6; ++k) {
      PHX_CHECK(hazard.status[k] == reference.status[k]);
      PHX_CHECK(hazard.signs[k] == reference.signs[k]);
    }
    const SubnormalStatus flushed = evaluate_subnormal();
    for (int k = 0; k < 8; ++k) PHX_CHECK(flushed.status[k] == subnormal.status[k]);
  });
}

// insphere (whose exact stage runs on integers when the coordinates or the
// differences span at most 61 or 62 bits, else on expansions) against the
// expansion-coordinate insphere_sign on random, rounded-cospherical, exactly
// cospherical, and mixed-scale (expansion fallback) inputs.
void test_insphere_exact_paths() {
  std::uint64_t seed = 77;
  int checked = 0, zeros = 0, mismatches = 0;
  const auto compare = [&](const double* const* p) {
    Expansion e[5][3];
    for (int k = 0; k < 5; ++k) {
      for (int axis = 0; axis < 3; ++axis) e[k][axis] = Expansion(p[k][axis]);
    }
    const int expected = insphere_sign(e[0], e[1], e[2], e[3], e[4]);
    const int actual = insphere(p[0], p[1], p[2], p[3], p[4]);
    mismatches += expected == actual ? 0 : 1;
    zeros += expected == 0 ? 1 : 0;
    ++checked;
  };
  // Points of the unit sphere, rounded (near-degenerate), at four scales.
  std::vector<double> sphere;
  for (int i = 0; i < 400; ++i) {
    double v[3];
    for (double& c : v) c = 2.0 * uniform(seed) - 1.0;
    const double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (double c : v) sphere.push_back(c / norm);
  }
  for (const double scale : {1.0, 0x1p-40, 0x1p40, 3.0e-7}) {
    for (int trial = 0; trial < 4000; ++trial) {
      double points[5][3];
      const double* p[5];
      for (int k = 0; k < 5; ++k) {
        const auto row = static_cast<std::size_t>(splitmix64(seed++) % 400);
        for (int axis = 0; axis < 3; ++axis) points[k][axis] = scale * sphere[3 * row + axis];
        p[k] = points[k];
      }
      compare(p);
    }
  }
  // Exactly cospherical integer points (radius 45), some with a sixth row
  // nudged by one unit: exact zeros and the smallest nonzero values.
  std::vector<double> lattice;
  for (int x = -45; x <= 45; ++x) {
    for (int y = -45; y <= 45; ++y) {
      const int z2 = 45 * 45 - x * x - y * y;
      const int z = static_cast<int>(std::lround(std::sqrt(std::max(z2, 0))));
      if (z2 >= 0 && z * z == z2) lattice.insert(lattice.end(), {double(x), double(y), double(z)});
    }
  }
  const std::size_t rows = lattice.size() / 3;
  for (int trial = 0; trial < 20000; ++trial) {
    double points[5][3];
    const double* p[5];
    for (int k = 0; k < 5; ++k) {
      const auto row = static_cast<std::size_t>(splitmix64(seed++) % rows);
      for (int axis = 0; axis < 3; ++axis) points[k][axis] = lattice[3 * row + axis];
      p[k] = points[k];
    }
    if (trial % 3 == 1) points[4][trial % 3] += 1.0;
    if (trial % 5 == 2) {
      for (auto& point : points) {
        for (double& c : point) c = c * 0x1p-30 + 0.75;  // inexact differences
      }
    }
    compare(p);
  }
  // Mixed scales: one coordinate tiny, so neither integer form fits.
  for (int trial = 0; trial < 20000; ++trial) {
    double points[5][3];
    const double* p[5];
    for (int k = 0; k < 5; ++k) {
      for (int axis = 0; axis < 3; ++axis) points[k][axis] = uniform(seed) - 0.5;
      p[k] = points[k];
    }
    points[trial % 5][trial % 3] = 0x1p-90 * (uniform(seed) - 0.5);
    // Make the fifth point cospherical up to rounding: mirror the first.
    points[4][0] = -points[0][0];
    points[4][1] = points[0][1];
    points[4][2] = points[0][2];
    points[3][0] = -points[1][0];
    points[3][1] = points[1][1];
    points[3][2] = points[1][2];
    compare(p);
  }
  PHX_CHECK(mismatches == 0);
  PHX_CHECK(zeros > 100);
  std::printf("insphere exact paths: %d checked, %d exact zeros, %d mismatches\n", checked, zeros,
              mismatches);
}

// insphere's expansion stages (stage B on stack expansions of the rounded
// differences, stage C with the first-order difference tails, stage D on
// the exact differences) decide the sign of the unfiltered exact expansion
// insphere_sign on near-cospherical inputs whose coordinates span too many
// bits for the integer stage (a z coordinate below 1e-9 with a full
// significand): rounded points of spheres about non-dyadic centers in the
// plane z = 0 (inexact differences), small patches of a unit sphere at its
// equator (five nearly coplanar and nearly cospherical surface points, as a
// surface sample's Delaunay triangulation tests them), each with the query
// moved by 1-4 ulps and by a relative 4e-15 to 4e-14 along its first axis
// (determinants just above the stage-A and stage-B bounds), and spheres
// through the origin queried at the origin (exact differences).
void test_insphere_expansion_stages_agree() {
  std::uint64_t seed = 4242;
  int checked = 0, zeros = 0, mismatches = 0;
  const auto compare = [&](const double (&points)[5][3]) {
    Expansion e[5][3];
    for (int k = 0; k < 5; ++k) {
      for (int axis = 0; axis < 3; ++axis) e[k][axis] = Expansion(points[k][axis]);
    }
    const int expected = insphere_sign(e[0], e[1], e[2], e[3], e[4]);
    const int actual = insphere(points[0], points[1], points[2], points[3], points[4]);
    mismatches += expected == actual ? 0 : 1;
    zeros += expected == 0 ? 1 : 0;
    ++checked;
  };
  const auto compare_moved = [&](const double (&points)[5][3]) {
    compare(points);
    double moved[5][3];
    for (int ulps = 1; ulps <= 4; ++ulps) {
      for (const double direction : {-1.0, 1.0}) {
        std::copy(&points[0][0], &points[0][0] + 15, &moved[0][0]);
        double& value = moved[4][ulps % 2];
        for (int step = 0; step < ulps; ++step) {
          value = std::nextafter(value, direction * std::numeric_limits<double>::infinity());
        }
        compare(moved);
      }
    }
    for (const double relative : {4e-15, -1e-14, 4e-14}) {
      std::copy(&points[0][0], &points[0][0] + 15, &moved[0][0]);
      moved[4][0] *= 1.0 + relative;
      compare(moved);
    }
  };
  // A unit vector near (x, y, 0) with a z component below 1e-9.
  const auto unit = [&](double x, double y, double* v) {
    v[0] = x;
    v[1] = y;
    v[2] = 1e-9 * (2.0 * uniform(seed) - 1.0);
    const double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int axis = 0; axis < 3; ++axis) v[axis] /= norm;
  };
  for (const double radius : {1.0, 1e-3, 37.3}) {
    for (const std::array<double, 2> center :
         {std::array<double, 2>{0.1, 1.0 / 3.0}, std::array<double, 2>{1e3 / 7.0, -2.0}}) {
      for (int trial = 0; trial < 1000; ++trial) {
        double points[5][3];
        for (auto& point : points) {
          double v[3];
          unit(2.0 * uniform(seed) - 1.0, 2.0 * uniform(seed) - 1.0, v);
          point[0] = center[0] + radius * v[0];
          point[1] = center[1] + radius * v[1];
          point[2] = radius * v[2];
        }
        compare_moved(points);
      }
    }
  }
  for (const double patch : {0.05, 0.002}) {
    for (int trial = 0; trial < 3000; ++trial) {
      const double angle = 6.283185307179586 * uniform(seed);
      double points[5][3];
      for (auto& point : points) {
        double v[3];
        unit(std::cos(angle) + patch * (2.0 * uniform(seed) - 1.0),
             std::sin(angle) + patch * (2.0 * uniform(seed) - 1.0), v);
        point[0] = 0.25 + v[0];
        point[1] = 0.25 + v[1];
        point[2] = v[2];
      }
      compare_moved(points);
    }
  }
  for (int trial = 0; trial < 20000; ++trial) {
    double points[5][3];
    for (int k = 0; k < 4; ++k) {
      // Points of the unit sphere about (1, 0, 0) within about 1e-3 of the
      // origin.
      double v[3];
      unit(-1.0, 1e-3 * (2.0 * uniform(seed) - 1.0), v);
      points[k][0] = 1.0 + v[0];
      points[k][1] = v[1];
      points[k][2] = v[2];
    }
    for (double& c : points[4]) c = 0.0;
    compare(points);
  }
  PHX_CHECK(checked >= 100000);
  PHX_CHECK(mismatches == 0);
  std::printf("insphere expansion stages: %d checked, %d exact zeros, %d mismatches\n", checked,
              zeros, mismatches);
}

// power3d_stage decides only what power3d decides, with its sign: rounded
// near-orthogonal weighted points (weights from one sphere, so the power
// determinant nearly vanishes) at several scales and weight magnitudes,
// exactly orthogonal integer configurations (exact zeros, which the stage
// must leave undecided) and those with one weight nudged by one unit.
void test_power3d_stage_agrees() {
  std::uint64_t seed = 91;
  int checked = 0, decided = 0, zeros = 0, mismatches = 0;
  const auto compare = [&](const double (&p)[5][3], const double (&w)[5]) {
    const int exact = power3d(p[0], p[1], p[2], p[3], p[4], w[0], w[1], w[2], w[3], w[4]);
    const int stage = power3d_stage(p[0], p[1], p[2], p[3], p[4], w[0], w[1], w[2], w[3], w[4]);
    mismatches += stage != 0 && stage != exact ? 1 : 0;
    decided += stage != 0 ? 1 : 0;
    zeros += exact == 0 ? 1 : 0;
    ++checked;
  };
  for (const double scale : {1.0, 0x1p-40, 0x1p40, 3.0e-7}) {
    for (const double spread : {0.0, 1.0, 1e6}) {
      for (int trial = 0; trial < 3000; ++trial) {
        double center[3], p[5][3], w[5];
        for (double& c : center) c = scale * (uniform(seed) - 0.5);
        const double radius = scale * (0.25 + uniform(seed));
        for (int k = 0; k < 5; ++k) {
          double squared = 0.0;
          for (int axis = 0; axis < 3; ++axis) {
            p[k][axis] = scale * (2.0 * uniform(seed) - 1.0);
            squared += (p[k][axis] - center[axis]) * (p[k][axis] - center[axis]);
          }
          // Orthogonal to the sphere (center, radius) up to rounding, with
          // a common offset `spread` scale^2 that leaves the power test
          // unchanged but enlarges the weights.
          w[k] = squared - radius * radius + spread * scale * scale;
        }
        compare(p, w);
      }
    }
  }
  // Integer points with w = |p|^2 - 45^2 are exactly orthogonal to the
  // sphere of radius 45 at the origin: the determinant is exactly zero.
  for (int trial = 0; trial < 20000; ++trial) {
    double p[5][3], w[5];
    for (int k = 0; k < 5; ++k) {
      double squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        p[k][axis] = static_cast<double>(static_cast<int>(splitmix64(seed++) % 121) - 60);
        squared += p[k][axis] * p[k][axis];
      }
      w[k] = squared - 45.0 * 45.0;
    }
    if (trial % 3 == 1) w[trial % 5] += trial % 2 == 0 ? 1.0 : -1.0;
    compare(p, w);
  }
  PHX_CHECK(mismatches == 0);
  PHX_CHECK(zeros > 1000);
  PHX_CHECK(decided > checked / 4);
  std::printf("power3d stage: %d checked, %d decided, %d exact zeros, %d mismatches\n", checked,
              decided, zeros, mismatches);
}

// The incircle determinant by full expansion arithmetic on the exact
// coordinate differences (D) or, with `exact` false, on their binary64
// roundings (D~), without any filter.
Expansion incircle_value(const double* a, const double* b, const double* c, const double* d,
                         bool exact) {
  const double* p[3] = {a, b, c};
  Expansion x[3], y[3], lift[3];
  for (int k = 0; k < 3; ++k) {
    x[k] = exact ? Expansion::difference(p[k][0], d[0]) : Expansion(p[k][0] - d[0]);
    y[k] = exact ? Expansion::difference(p[k][1], d[1]) : Expansion(p[k][1] - d[1]);
    lift[k] = x[k] * x[k] + y[k] * y[k];
  }
  return lift[0] * (x[1] * y[2] - x[2] * y[1]) + lift[1] * (x[2] * y[0] - x[0] * y[2]) +
         lift[2] * (x[0] * y[1] - x[1] * y[0]);
}

// The exact incircle sign: the reference for the stages.
int incircle_reference(const double* a, const double* b, const double* c, const double* d) {
  return incircle_value(a, b, c, d, true).sign();
}

// d moved by `ulps` units in the last place of one coordinate (negative
// counts move down).
void nudge(double (&d)[2], int axis, int ulps) {
  for (int step = 0; step < (ulps < 0 ? -ulps : ulps); ++step) {
    d[axis] = std::nextafter(d[axis], ulps < 0 ? -INFINITY : INFINITY);
  }
}

// The staged incircle (stage A, compensated and expansion stage B, C, D) and
// the inlined floating stage the triangulations call first decide only the
// sign of the unfiltered full expansion, on exactly cocircular integer and
// dyadic lattice points (exact differences), exactly cocircular octagonal
// images of a real point (inexact differences), regular n-gons for
// n = 3..400 at several radii and centers, those configurations with the
// query moved by 1-4 ulps or by a relative 1e-12, and random points.
void test_incircle_stages_agree() {
  std::uint64_t seed = 2027;
  int checked = 0, zeros = 0, mismatches = 0;
  int stage_count[5] = {0, 0, 0, 0, 0};
  const auto compare = [&](const double* a, const double* b, const double* c, const double* d) {
    const int exact = incircle_reference(a, b, c, d);
    const IncircleDecision decision = incircle_decision(a, b, c, d);
    const int filter = incircle_stage(a, b, c, d);
    mismatches += decision.sign != exact || (filter != 0 && filter != exact) ? 1 : 0;
    mismatches += (filter != 0) != (decision.stage == IncircleStage::kA) ? 1 : 0;
    ++stage_count[static_cast<int>(decision.stage)];
    zeros += exact == 0 ? 1 : 0;
    ++checked;
  };
  // The query moved by 1-4 ulps either way along either axis, and by a
  // relative 1e-12 inward and outward.
  const auto compare_perturbed = [&](const double* a, const double* b, const double* c,
                                     const double* d) {
    compare(a, b, c, d);
    for (int ulps = 1; ulps <= 4; ++ulps) {
      for (const int direction : {-1, 1}) {
        double moved[2] = {d[0], d[1]};
        nudge(moved, ulps % 2, direction * ulps);
        compare(a, b, c, moved);
      }
    }
    for (const double relative : {-1e-12, 1e-12}) {
      const double moved[2] = {d[0] * (1.0 + relative), d[1] * (1.0 - relative)};
      compare(a, b, c, moved);
    }
  };
  const auto pick = [&](std::size_t count) {
    return static_cast<std::size_t>(splitmix64(seed++) % count);
  };

  // The 48 lattice points on x^2 + y^2 = 5525 = 5^2 13 17, scaled by powers
  // of two and translated by dyadic (exactly cocircular) or decimal
  // (rounded) centers.
  std::vector<std::array<double, 2>> lattice;
  for (int x = -74; x <= 74; ++x) {
    for (int y = -74; y <= 74; ++y) {
      if (x * x + y * y == 5525) {
        lattice.push_back({static_cast<double>(x), static_cast<double>(y)});
      }
    }
  }
  PHX_CHECK(lattice.size() == 48);
  for (const double scale : {1.0, 0x1p-30, 0x1p20}) {
    for (const std::array<double, 2> center :
         {std::array<double, 2>{0.0, 0.0}, std::array<double, 2>{0x1p30, -0x1p-10},
          std::array<double, 2>{0.1, 1.0 / 3.0}, std::array<double, 2>{1e3 / 7.0, -2e3}}) {
      for (int trial = 0; trial < 1500; ++trial) {
        double q[4][2];
        for (auto& point : q) {
          const std::array<double, 2>& lattice_point = lattice[pick(lattice.size())];
          point[0] = center[0] + scale * lattice_point[0];
          point[1] = center[1] + scale * lattice_point[1];
        }
        compare_perturbed(q[0], q[1], q[2], q[3]);
      }
    }
  }
  // Octagonal images (+-u, +-v), (+-v, +-u) of a real point lie exactly on
  // the circle of radius |(u, v)| about the origin.
  for (int trial = 0; trial < 3000; ++trial) {
    const double u = 2.0 * uniform(seed) - 1.0, v = 2.0 * uniform(seed) - 1.0;
    const double images[8][2] = {{u, v}, {-v, u}, {-u, -v}, {v, -u},
                                 {v, u}, {-u, v}, {-v, -u}, {u, -v}};
    const double* q[4];
    for (auto& point : q) point = images[pick(8)];
    compare_perturbed(q[0], q[1], q[2], q[3]);
  }
  // Regular n-gons: consecutive vertices, quarter turns and random choices.
  constexpr double kPi = 3.14159265358979323846;
  for (int n = 3; n <= 400; ++n) {
    for (const std::array<double, 3> circle :
         {std::array<double, 3>{0.0, 0.0, 1.0}, std::array<double, 3>{0.5, 0.5, 0.3},
          std::array<double, 3>{-1e3, 2e3, 1e-2}, std::array<double, 3>{1e5, 1e5, 37.0}}) {
      std::vector<double> polygon(2 * static_cast<std::size_t>(n));
      for (int k = 0; k < n; ++k) {
        const double angle = 2.0 * kPi * k / n;
        polygon[2 * k] = circle[0] + circle[2] * std::cos(angle);
        polygon[2 * k + 1] = circle[1] + circle[2] * std::sin(angle);
      }
      const auto vertex = [&](std::size_t k) { return polygon.data() + 2 * (k % n); };
      for (int start = 0; start < 4; ++start) {
        const std::size_t first = pick(n);
        compare_perturbed(vertex(first), vertex(first + 1), vertex(first + 2), vertex(first + 3));
        compare_perturbed(vertex(first), vertex(first + n / 4), vertex(first + n / 2),
                          vertex(first + 3 * n / 4));
        compare_perturbed(vertex(pick(n)), vertex(pick(n)), vertex(pick(n)), vertex(pick(n)));
      }
    }
  }
  for (int trial = 0; trial < 20000; ++trial) {
    double q[4][2];
    for (auto& point : q) {
      point[0] = uniform(seed);
      point[1] = uniform(seed);
    }
    compare(q[0], q[1], q[2], q[3]);
  }
  PHX_CHECK(mismatches == 0);
  PHX_CHECK(zeros > 1000);
  for (const int count : stage_count) PHX_CHECK(count > 0);
  std::printf(
      "incircle stages: %d checked, decided A %d, B compensated %d, B expansion %d, C %d, D %d, "
      "%d exact zeros, %d mismatches\n",
      checked, stage_count[0], stage_count[1], stage_count[2], stage_count[3], stage_count[4],
      zeros, mismatches);
}

// ---------------------------------------------------------------- stage thresholds

constexpr double kUnit = 0x1p-53;

// The binary64 neighbors low < high in the positive interval [low, high] with
// `holds` false at low and true at high, by bisection of the ordered bit
// patterns; `holds` need not be monotone.
template <class Holds>
std::array<double, 2> straddle(double low, double high, const Holds& holds) {
  PHX_CHECK(0.0 < low && low < high && !holds(low) && holds(high));
  std::uint64_t below = std::bit_cast<std::uint64_t>(low);
  std::uint64_t above = std::bit_cast<std::uint64_t>(high);
  while (above - below > 1) {
    const std::uint64_t middle = below + (above - below) / 2;
    if (holds(std::bit_cast<double>(middle))) {
      above = middle;
    } else {
      below = middle;
    }
  }
  return {std::bit_cast<double>(below), std::bit_cast<double>(above)};
}

// Shewchuk's incircle bounds (stage B on the compensated estimate, or on
// 64u^2 P when every difference is exact; see predicates.cpp).
constexpr double kIncircleBoundA = (10.0 + 96.0 * kUnit) * kUnit;
constexpr double kIncircleBoundB = (4.0 + 128.0 * kUnit) * kUnit;
constexpr double kIncircleBoundBExact = 64.0 * kUnit * kUnit;
constexpr double kIncircleBoundC = (44.0 + 576.0 * kUnit) * kUnit * kUnit;
constexpr double kIncircleResultBound = (3.0 + 8.0 * kUnit) * kUnit;

// What the incircle stages decide on: Shewchuk's stage-A value and permanent
// P (his published expressions), estimates of the exact determinant D~ of the
// rounded differences and of the exact determinant D, and whether every
// difference is exact.
struct IncircleQuantities {
  double stage_a;
  double permanent;
  double rounded;
  double exact;
  bool exact_differences;
};

IncircleQuantities incircle_quantities(const double* a, const double* b, const double* c,
                                       const double* d) {
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;
  const double alift = adx * adx + ady * ady;
  const double blift = bdx * bdx + bdy * bdy;
  const double clift = cdx * cdx + cdy * cdy;
  bool exact_differences = true;
  for (const double* p : {a, b, c}) {
    for (int axis = 0; axis < 2; ++axis) {
      exact_differences =
          exact_differences && two_diff_tail(p[axis], d[axis], p[axis] - d[axis]) == 0.0;
    }
  }
  return {alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady),
          (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * alift +
              (std::fabs(cdxady) + std::fabs(adxcdy)) * blift +
              (std::fabs(adxbdy) + std::fabs(bdxady)) * clift,
          incircle_value(a, b, c, d, false).estimate(), incircle_value(a, b, c, d, true).estimate(),
          exact_differences};
}

bool incircle_stage_a_decides(const IncircleQuantities& q) {
  const double bound = kIncircleBoundA * q.permanent;
  return q.stage_a > bound || -q.stage_a > bound;
}
double incircle_ratio_b(const IncircleQuantities& q) {
  return std::fabs(q.rounded) / (kIncircleBoundB * q.permanent);
}
double incircle_ratio_b_exact(const IncircleQuantities& q) {
  return std::fabs(q.exact) / (kIncircleBoundBExact * q.permanent);
}
double incircle_ratio_c(const IncircleQuantities& q) {
  return std::fabs(q.exact) /
         (kIncircleBoundC * q.permanent + kIncircleResultBound * std::fabs(q.rounded));
}

// The stage the published bounds force to decide, or none inside a band in
// which the certified error of a stage's estimate leaves its decision open.
// Compensated stage B: |D~ - E| <= 52.4u^2 Pi + u |E| with Pi <= (1 + 7u) P,
// a relative 15u at the bound, so E clears (4 + 128u)u P when |D~| exceeds it
// by a relative 2^-40 and cannot when |D~| falls short by as much; then the
// expansion stage B, whose estimate of D~ is good to a few u, cannot clear
// (4 + 48u)u P either. With exact differences (D = D~) E clears 64u^2 P when
// |D| >= 1.83 * 64u^2 P (117.1 - 52.5 > 64) and cannot when |D| <= 0.17 *
// 64u^2 P (10.9 + 52.5 < 64), and stage B on the exact expansion decides.
// Stage C certifies |D - det_C| < errbound_C, so it decides when |D| >= 2.01
// errbound_C.
std::optional<IncircleStage> forced_incircle_stage(const IncircleQuantities& q) {
  if (incircle_stage_a_decides(q)) return IncircleStage::kA;
  if (q.exact_differences) {
    const double ratio = incircle_ratio_b_exact(q);
    if (ratio >= 1.83) return IncircleStage::kBCompensated;
    if (ratio <= 0.17) return IncircleStage::kBExpansion;
    return std::nullopt;
  }
  const double ratio = incircle_ratio_b(q);
  if (ratio >= 1.0 + 0x1p-40) return IncircleStage::kBCompensated;
  if (ratio > 1.0 - 0x1p-40) return std::nullopt;
  if (incircle_ratio_c(q) >= 2.01) return IncircleStage::kC;
  return std::nullopt;
}

// Each incircle stage just above and just below its error bound (inputs at
// the binary64 neighbors where |estimate| / bound crosses 1 or the edge of
// the band its estimate's certified error leaves open) decides with the
// expected stage and the exact sign. a = (65, s) beside the circle
// x^2 + y^2 = 65^2 through the lattice points b, c, d gives D = orient(b, c,
// d) s^2 (its lift exceeds the circle's by s^2), so the ratios move by a
// relative 2^-26 or less between neighboring s, rounding included. With
// d = (-65, 0) every difference is exact; with d = (-33, -56), s + 56 rounds.
// Stage C is placed on the circle through a, b, c and the origin with d a
// tiny multiple of (1, phi): the rounded differences are a, b, c, so D~ = 0
// and det_C is the first-order correction alone, which here matches D to a
// relative 2^-40 (its terms do not cancel and the tails are below 2^-50 of
// the differences); the bound is then sharp to 2^-30.
void test_incircle_stage_thresholds() {
  const double b[2] = {16.0, 63.0};
  const double c[2] = {-60.0, 25.0};
  const double exact_d[2] = {-65.0, 0.0};
  const double rounded_d[2] = {-33.0, -56.0};
  int checked = 0;
  const auto check = [&](const double* a, const double* p, const double* q, const double* d,
                         IncircleStage expected, bool forced) {
    const IncircleQuantities quantities = incircle_quantities(a, p, q, d);
    if (forced) PHX_CHECK(forced_incircle_stage(quantities) == expected);
    const IncircleDecision decision = incircle_decision(a, p, q, d);
    PHX_CHECK(decision.stage == expected);
    PHX_CHECK(decision.sign != 0 && decision.sign == incircle_reference(a, p, q, d));
    PHX_CHECK(incircle(a, p, q, d) == decision.sign);
    ++checked;
  };
  const auto beside = [&](const double* d) {
    return [d, &b, &c](double s) {
      const double a[2] = {65.0, s};
      return incircle_quantities(a, b, c, d);
    };
  };
  const auto at = [&](const double* d, double s, IncircleStage expected, bool exact) {
    const double a[2] = {65.0, s};
    PHX_CHECK(incircle_quantities(a, b, c, d).exact_differences == exact);
    check(a, b, c, d, expected, true);
  };
  // Stage A, against the bound on its own floating value.
  for (const double* d : {exact_d, rounded_d}) {
    const auto quantities = beside(d);
    const auto [below, above] = straddle(0x1p-30, 1.0, [&](double s) {
      return incircle_stage_a_decides(quantities(s));
    });
    at(d, above, IncircleStage::kA, d == exact_d);
    at(d, below, IncircleStage::kBCompensated, d == exact_d);
  }
  // Compensated stage B with rounded differences; stage C decides below.
  const auto rounded = beside(rounded_d);
  const double b_above = straddle(0x1p-30, 1.0, [&](double s) {
    return incircle_ratio_b(rounded(s)) >= 1.0 + 0x1p-40;
  })[1];
  const double b_below = straddle(0x1p-30, 1.0, [&](double s) {
    return incircle_ratio_b(rounded(s)) > 1.0 - 0x1p-40;
  })[0];
  at(rounded_d, b_above, IncircleStage::kBCompensated, false);
  at(rounded_d, b_below, IncircleStage::kC, false);
  // Compensated stage B with exact differences; the expansion decides below.
  const auto exact = beside(exact_d);
  const double exact_above =
      straddle(0x1p-60, 1.0, [&](double s) { return incircle_ratio_b_exact(exact(s)) >= 1.83; })[1];
  const double exact_below =
      straddle(0x1p-60, 1.0, [&](double s) { return incircle_ratio_b_exact(exact(s)) > 0.17; })[0];
  at(exact_d, exact_above, IncircleStage::kBCompensated, true);
  at(exact_d, exact_below, IncircleStage::kBExpansion, true);
  // Stage C; the full expansion decides below.
  const double ca[2] = {12.0, 4.0}, cb[2] = {10.0, 10.0}, cc[2] = {-2.0, 4.0};
  const auto origin_offset = [&](double t) {
    const double d[2] = {t, 0.6180339887498949 * t};
    return incircle_quantities(ca, cb, cc, d);
  };
  const double c_above = straddle(0x1p-110, 0x1p-60, [&](double t) {
    return incircle_ratio_c(origin_offset(t)) >= 1.0 + 0x1p-30;
  })[1];
  const double c_below = straddle(0x1p-110, 0x1p-60, [&](double t) {
    return incircle_ratio_c(origin_offset(t)) > 1.0 - 0x1p-30;
  })[0];
  for (const auto& [t, expected] : {std::pair{c_above, IncircleStage::kC},
                                    std::pair{c_below, IncircleStage::kD}}) {
    const IncircleQuantities q = origin_offset(t);
    PHX_CHECK(q.rounded == 0.0 && !q.exact_differences && !incircle_stage_a_decides(q));
    const double d[2] = {t, 0.6180339887498949 * t};
    check(ca, cb, cc, d, expected, false);
  }
  std::printf(
      "incircle thresholds: %d checked; A on its value; B %.17g | %.17g; B exact %.6g | %.6g; "
      "C %.17g | %.17g\n",
      checked, incircle_ratio_b(rounded(b_below)), incircle_ratio_b(rounded(b_above)),
      incircle_ratio_b_exact(exact(exact_below)), incircle_ratio_b_exact(exact(exact_above)),
      incircle_ratio_c(origin_offset(c_below)), incircle_ratio_c(origin_offset(c_above)));
}

// orient2d's det[a - c, b - c] from exact (or rounded) differences.
Expansion orient2d_value(const double* a, const double* b, const double* c, bool exact) {
  const auto difference = [exact](double p, double q) {
    return exact ? Expansion::difference(p, q) : Expansion(p - q);
  };
  return difference(a[0], c[0]) * difference(b[1], c[1]) -
         difference(a[1], c[1]) * difference(b[0], c[0]);
}

// Shewchuk's orient3d determinant det[a - d; b - d; c - d] (orient3d is its
// negated sign) from exact (or rounded) differences.
Expansion orient3d_value(const double* a, const double* b, const double* c, const double* d,
                         bool exact) {
  const double* p[3] = {a, b, c};
  Expansion x[3], y[3], z[3];
  for (int k = 0; k < 3; ++k) {
    x[k] = exact ? Expansion::difference(p[k][0], d[0]) : Expansion(p[k][0] - d[0]);
    y[k] = exact ? Expansion::difference(p[k][1], d[1]) : Expansion(p[k][1] - d[1]);
    z[k] = exact ? Expansion::difference(p[k][2], d[2]) : Expansion(p[k][2] - d[2]);
  }
  return z[0] * (x[1] * y[2] - x[2] * y[1]) + z[1] * (x[2] * y[0] - x[0] * y[2]) +
         z[2] * (x[0] * y[1] - x[1] * y[0]);
}

// Shewchuk's insphere determinant, the lifted determinant over a - e, ..,
// d - e (insphere is its negated sign), from exact (or rounded) differences
// by cofactor expansion along the lifts.
Expansion insphere_value(const double* const* p, bool exact) {
  const double* e = p[4];
  Expansion x[4], y[4], z[4], lift[4];
  for (int k = 0; k < 4; ++k) {
    x[k] = exact ? Expansion::difference(p[k][0], e[0]) : Expansion(p[k][0] - e[0]);
    y[k] = exact ? Expansion::difference(p[k][1], e[1]) : Expansion(p[k][1] - e[1]);
    z[k] = exact ? Expansion::difference(p[k][2], e[2]) : Expansion(p[k][2] - e[2]);
    lift[k] = x[k] * x[k] + y[k] * y[k] + z[k] * z[k];
  }
  const auto minor = [&](int i, int j, int k) {
    return x[i] * (y[j] * z[k] - y[k] * z[j]) - x[j] * (y[i] * z[k] - y[k] * z[i]) +
           x[k] * (y[i] * z[j] - y[j] * z[i]);
  };
  return lift[3] * minor(0, 1, 2) - lift[2] * minor(0, 1, 3) + lift[1] * minor(0, 2, 3) -
         lift[0] * minor(1, 2, 3);
}

// Shewchuk's insphere stage-A value and permanent (his published
// expressions, as insphere evaluates them).
std::array<double, 2> insphere_stage_a(const double* const* p) {
  const double* e = p[4];
  const double aex = p[0][0] - e[0], bex = p[1][0] - e[0], cex = p[2][0] - e[0];
  const double dex = p[3][0] - e[0];
  const double aey = p[0][1] - e[1], bey = p[1][1] - e[1], cey = p[2][1] - e[1];
  const double dey = p[3][1] - e[1];
  const double aez = p[0][2] - e[2], bez = p[1][2] - e[2], cez = p[2][2] - e[2];
  const double dez = p[3][2] - e[2];
  const double aexbey = aex * bey, bexaey = bex * aey, ab = aexbey - bexaey;
  const double bexcey = bex * cey, cexbey = cex * bey, bc = bexcey - cexbey;
  const double cexdey = cex * dey, dexcey = dex * cey, cd = cexdey - dexcey;
  const double dexaey = dex * aey, aexdey = aex * dey, da = dexaey - aexdey;
  const double aexcey = aex * cey, cexaey = cex * aey, ac = aexcey - cexaey;
  const double bexdey = bex * dey, dexbey = dex * bey, bd = bexdey - dexbey;
  const double abc = aez * bc - bez * ac + cez * ab;
  const double bcd = bez * cd - cez * bd + dez * bc;
  const double cda = cez * da + dez * ac + aez * cd;
  const double dab = dez * ab + aez * bd + bez * da;
  const double alift = aex * aex + aey * aey + aez * aez;
  const double blift = bex * bex + bey * bey + bez * bez;
  const double clift = cex * cex + cey * cey + cez * cez;
  const double dlift = dex * dex + dey * dey + dez * dez;
  const double det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
  const double az = std::fabs(aez), bz = std::fabs(bez), cz = std::fabs(cez), dz = std::fabs(dez);
  const double ab_plus = std::fabs(aexbey) + std::fabs(bexaey);
  const double bc_plus = std::fabs(bexcey) + std::fabs(cexbey);
  const double cd_plus = std::fabs(cexdey) + std::fabs(dexcey);
  const double da_plus = std::fabs(dexaey) + std::fabs(aexdey);
  const double ac_plus = std::fabs(aexcey) + std::fabs(cexaey);
  const double bd_plus = std::fabs(bexdey) + std::fabs(dexbey);
  const double permanent = (cd_plus * bz + bd_plus * cz + bc_plus * dz) * alift +
                           (da_plus * cz + ac_plus * dz + cd_plus * az) * blift +
                           (ab_plus * dz + bd_plus * az + da_plus * bz) * clift +
                           (bc_plus * az + ac_plus * bz + ab_plus * cz) * dlift;
  return {det, permanent};
}

// orient2d, orient3d and insphere at the binary64 neighbors where their
// stage A (Shewchuk's floating value against his bound, as evaluated) and
// stage B (the estimate of D~ against his stage-B bound, decided with a
// 2^-40 margin, as that estimate is good to a few u) cross 1 return the exact
// sign; the public floating stages (orientation_stage, insphere_stage) decide
// just above their own bounds and defer just below. These predicates do not
// report their deciding stage, and being linear in the varied coordinate,
// orient2d and orient3d move by up to a tenth of their bound between
// neighbors. insphere uses e = (45, s, 0) beside the sphere of radius 45
// through lattice points, where D = orient s^2 and its expansion stages run
// (s spans more bits than its integer stages take).
void test_orientation_stage_thresholds() {
  int checked = 0;
  // orient2d: a = (0.5, y) beside the line through b and c; a - c rounds.
  const double b2[2] = {12.0, 12.0}, c2[2] = {24.0, 24.0};
  const auto orient2d_check = [&](double y) {
    const double a[2] = {0.5, y};
    const int exact = orient2d_value(a, b2, c2, true).sign();
    PHX_CHECK(exact != 0 && orient2d(a, b2, c2) == exact && orient2d_filtered(a, b2, c2) == exact);
    ++checked;
  };
  const auto orient2d_ratios = [&](double y) {
    const double a[2] = {0.5, y};
    const double left = (a[0] - c2[0]) * (b2[1] - c2[1]);
    const double right = (a[1] - c2[1]) * (b2[0] - c2[0]);
    const double sum = std::fabs(left) + std::fabs(right);
    return std::array<double, 2>{
        std::fabs(left - right) / ((3.0 + 16.0 * kUnit) * kUnit * sum),
        std::fabs(orient2d_value(a, b2, c2, false).estimate()) /
            ((2.0 + 12.0 * kUnit) * kUnit * sum)};
  };
  for (const double y : straddle(0.5, 0.5 + 0x1p-30, [&](double value) {
         return orient2d_ratios(value)[0] >= 1.0;
       })) {
    orient2d_check(y);
  }
  orient2d_check(straddle(0.5, 0.5 + 0x1p-30, [&](double y) {
    return orient2d_ratios(y)[1] >= 1.0 + 0x1p-40;
  })[1]);
  orient2d_check(straddle(0.5, 0.5 + 0x1p-30, [&](double y) {
    return orient2d_ratios(y)[1] > 1.0 - 0x1p-40;
  })[0]);

  // orient3d: a = (0.5, 0.25, z) beside the plane z = (x + y) / 2 + 6 through
  // b, c, d; a - d rounds.
  const double b3[3] = {12.0, 0.0, 12.0}, c3[3] = {0.0, 12.0, 12.0}, d3[3] = {24.0, 24.0, 30.0};
  const auto orient3d_at = [&](double z, bool stage_decides) {
    const double a[3] = {0.5, 0.25, z};
    const int exact = -orient3d_value(a, b3, c3, d3, true).sign();
    const OrientationStage stage = orientation_stage(a, b3, c3, d3);
    const bool decided = stage.det > stage.errbound || -stage.det > stage.errbound;
    PHX_CHECK(exact != 0 && orient3d(a, b3, c3, d3) == exact &&
              orient3d_filtered(a, b3, c3, d3) == exact);
    PHX_CHECK(decided == stage_decides && (!decided || (stage.det > 0.0 ? -1 : 1) == exact));
    ++checked;
  };
  const auto orient3d_ratio_b = [&](double z) {
    const double a[3] = {0.5, 0.25, z};
    const double permanent =
        orientation_stage(a, b3, c3, d3).errbound / ((7.0 + 56.0 * kUnit) * kUnit);
    return std::fabs(orient3d_value(a, b3, c3, d3, false).estimate()) /
           ((3.0 + 28.0 * kUnit) * kUnit * permanent);
  };
  const auto [z_below, z_above] = straddle(6.375, 6.375 + 0x1p-20, [&](double z) {
    const double a[3] = {0.5, 0.25, z};
    const OrientationStage stage = orientation_stage(a, b3, c3, d3);
    return stage.det > stage.errbound || -stage.det > stage.errbound;
  });
  orient3d_at(z_below, false);
  orient3d_at(z_above, true);
  orient3d_at(straddle(6.375, 6.375 + 0x1p-20, [&](double z) {
    return orient3d_ratio_b(z) >= 1.0 + 0x1p-40;
  })[1], false);
  orient3d_at(straddle(6.375, 6.375 + 0x1p-20, [&](double z) {
    return orient3d_ratio_b(z) > 1.0 - 0x1p-40;
  })[0], false);

  // insphere: e = (45, s, 0) beside the sphere of radius 45; e's differences
  // with the y coordinates round.
  const double pa[3] = {36.0, 27.0, 0.0}, pb[3] = {0.0, 27.0, 36.0};
  const double pc[3] = {-20.0, 5.0, 40.0}, pd[3] = {5.0, -40.0, -20.0};
  const auto insphere_at = [&](double s) {
    const double e[3] = {45.0, s, 0.0};
    const double* p[5] = {pa, pb, pc, pd, e};
    const int exact = -insphere_value(p, true).sign();
    PHX_CHECK(exact != 0 && insphere(pa, pb, pc, pd, e) == exact);
    ++checked;
    return insphere_stage(pa, pb, pc, pd, e);
  };
  const auto insphere_ratios = [&](double s) {
    const double e[3] = {45.0, s, 0.0};
    const double* p[5] = {pa, pb, pc, pd, e};
    const auto [det, permanent] = insphere_stage_a(p);
    return std::array<double, 2>{
        std::fabs(det) / ((16.0 + 224.0 * kUnit) * kUnit * permanent),
        std::fabs(insphere_value(p, false).estimate()) /
            ((5.0 + 72.0 * kUnit) * kUnit * permanent)};
  };
  for (const double s : straddle(0x1p-30, 1.0, [&](double value) {
         return insphere_ratios(value)[0] > 1.0;
       })) {
    insphere_at(s);
  }
  const auto [public_below, public_above] = straddle(0x1p-30, 1.0, [&](double s) {
    const double e[3] = {45.0, s, 0.0};
    return insphere_stage(pa, pb, pc, pd, e) != 0;
  });
  PHX_CHECK(insphere_at(public_below) == 0);
  PHX_CHECK(insphere_at(public_above) != 0);
  const double insphere_b_above =
      straddle(0x1p-30, 1.0, [&](double s) { return insphere_ratios(s)[1] >= 1.0 + 0x1p-40; })[1];
  const double insphere_b_below =
      straddle(0x1p-30, 1.0, [&](double s) { return insphere_ratios(s)[1] > 1.0 - 0x1p-40; })[0];
  insphere_at(insphere_b_above);
  insphere_at(insphere_b_below);
  std::printf("orientation thresholds: %d checked; insphere B %.17g | %.17g\n", checked,
              insphere_ratios(insphere_b_below)[1], insphere_ratios(insphere_b_above)[1]);
}

}  // namespace

int main() {
  test_orient2d_near_degenerate_grid();
  test_orient3d_exact_zero_and_ulp();
  test_orientation_conventions();
  test_incircle_insphere_exact_zero_and_ulp();
  test_insphere_exact_paths();
  test_insphere_expansion_stages_agree();
  test_power3d_stage_agrees();
  test_incircle_stages_agree();
  test_incircle_stage_thresholds();
  test_orientation_stage_thresholds();
  test_sos_orientation();
  test_sos_lifted_consistency();
  test_c_abi_batches();
  test_floating_point_hazards();
  return phx::mc::test::finish("test_predicates");
}
