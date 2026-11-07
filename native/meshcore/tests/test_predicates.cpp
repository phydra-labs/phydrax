//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <cmath>
#include <cstdint>
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

}  // namespace

int main() {
  test_orient2d_near_degenerate_grid();
  test_orient3d_exact_zero_and_ulp();
  test_orientation_conventions();
  test_incircle_insphere_exact_zero_and_ulp();
  test_sos_orientation();
  test_sos_lifted_consistency();
  test_c_abi_batches();
  test_floating_point_hazards();
  return phx::mc::test::finish("test_predicates");
}
