// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cmath>

#include "check.hpp"
#include "clip3d.hpp"
#include "floating_point_mode.hpp"
#include "metric_layer_overlap.hpp"
#include "phydrax_meshcore.h"

namespace {

using phx::mc::Expansion;
using phx::mc::TetrahedronClipper;
using phx::mc::metric::layer_detail::CommonSigns;
using phx::mc::metric::layer_detail::Contact;
using phx::mc::metric::layer_detail::StencilPoint;
using phx::mc::metric::layer_detail::Tet;
using phx::mc::metric::layer_detail::certified_contact;
using phx::mc::metric::layer_detail::positive_overlap;
using Coordinates = std::array<double, 12>;

Tet references(const Coordinates& points) {
  Tet tet{{{points.data()}, {points.data() + 3}, {points.data() + 6}, {points.data() + 9}}};
  const int orient = phx::mc::orient3d(tet[0].rounded, tet[1].rounded, tet[2].rounded, tet[3].rounded);
  PHX_CHECK(orient != 0);
  if (orient < 0) std::swap(tet[1], tet[2]);
  return tet;
}

bool clip_overlap(TetrahedronClipper& clipper, const Coordinates& a, const Coordinates& b) {
  bool positive = false;
  PHX_CHECK(clipper.overlaps(a.data(), b.data(), positive) == PHX_MC_OK);
  return positive;
}

void check_pair(TetrahedronClipper& clipper, const Coordinates& a, const Coordinates& b,
                bool expected, Contact expected_certificate) {
  const Tet first = references(a), second = references(b);
  for (bool reverse : {false, true}) {
    const auto& p = reverse ? second : first;
    const auto& q = reverse ? first : second;
    const bool oracle = reverse ? clip_overlap(clipper, b, a) : clip_overlap(clipper, a, b);
    PHX_CHECK(oracle == expected);
    const auto filtered = certified_contact(p, q);
    PHX_CHECK(filtered == Contact::kEdges || (filtered == Contact::kPositive) == oracle);
    if (expected_certificate != Contact::kEdges) PHX_CHECK(filtered == expected_certificate);
    CommonSigns signs;
    PHX_CHECK(positive_overlap(p, q, signs) == oracle);
    PHX_CHECK(signs.same());
  }
}

void check_lifted(TetrahedronClipper& clipper, const Coordinates& a, const Coordinates& b,
                  double dx, bool native_positive, bool exact_positive, Contact certificate) {
  const Tet first = references(a);
  Tet second = references(b);
  StencilPoint point;
  point.position = {Expansion(b[0]) + Expansion(dx), Expansion(b[1]), Expansion(b[2])};
  point.displacement[0] = std::nextafter(b[0], b[0] + std::copysign(1.0, dx)) - b[0];
  point.exact = false;
  PHX_CHECK(std::abs(dx) < 0.5 * std::abs(point.displacement[0]));
  second[0].stencil = &point;

  // Translate the exact embedding so its one unrepresentable component
  // becomes representable. The fixture's other subtractions are exact.
  Coordinates lifted_a = a, lifted_b = b;
  for (int j = 0; j < 12; j += 3) {
    lifted_a[j] -= b[0];
    lifted_b[j] -= b[0];
  }
  lifted_b[0] = dx;
  PHX_CHECK(clip_overlap(clipper, a, b) == native_positive);
  const bool oracle = clip_overlap(clipper, lifted_a, lifted_b);
  PHX_CHECK(oracle == exact_positive);
  PHX_CHECK(certified_contact(first, second) == certificate);
  CommonSigns signs;
  PHX_CHECK(positive_overlap(first, second, signs) == oracle);
  PHX_CHECK(!signs.same() || native_positive == exact_positive);
}

}  // namespace

int main() {
  const phx::mc::GradualNearestScope mode;
  PHX_CHECK(mode.established());
  TetrahedronClipper clipper;

  const Coordinates unit{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  check_pair(clipper, unit, unit, true, Contact::kPositive);
  const Coordinates aligned{0.5, 0, 0, 1.5, 0, 0, 0.5, 1, 0, 0.5, 0, 1};
  // Positive overlap, but no input edge enters the other tet's strict interior.
  check_pair(clipper, unit, aligned, true, Contact::kEdges);

  const Coordinates left{1, 0, 0, 1, 2, 0, 1, 0, 2, 0, 0, 0};
  const Coordinates touching{1, 0.5, 0.5, 2, 0.5, 0.5, 2, 1, 0.5, 2, 0.5, 1};
  check_pair(clipper, left, touching, false, Contact::kEdges);
  Coordinates moved = touching;
  moved[0] = std::nextafter(1.0, 0.0);
  check_pair(clipper, left, moved, true, Contact::kEdges);
  moved[0] = std::nextafter(1.0, 2.0);
  check_pair(clipper, left, moved, false, Contact::kEdges);
  moved[0] = 1.125;
  check_pair(clipper, left, moved, false, Contact::kDisjoint);
  check_lifted(clipper, left, touching, -0x1p-55, false, true, Contact::kEdges);

  const Coordinates long_tet{-2, 0, 0, 2, 0, 0, -2, 0.5, 0, -2, 0, 0.5};
  Coordinates rotated = long_tet;
  for (int j = 0; j < 12; j += 3) {
    // z-axis rotation followed by a 5/4 scale and a small translation.
    rotated[j] = 0.75 * long_tet[j] - long_tet[j + 1];
    rotated[j + 1] = long_tet[j] + 0.75 * long_tet[j + 1];
    rotated[j + 2] = 1.25 * long_tet[j + 2] + 0.0625;
  }
  check_pair(clipper, long_tet, rotated, true, Contact::kPositive);
  check_lifted(clipper, long_tet, rotated, 0x1p-55, true, true, Contact::kPositive);
  const auto anisotropic = [](Coordinates points) {
    for (int j = 0; j < 12; j += 3) {
      points[j] *= 0x1p100;
      points[j + 1] *= 0x1p-100;
      points[j + 2] *= 0x1p-85;
    }
    return points;
  };
  check_pair(clipper, anisotropic(long_tet), anisotropic(rotated), true, Contact::kPositive);

  const Coordinates lower{-2, 0, 0, 2, 0, 0, 0, 0.5, -0.5, 0, -0.5, -0.5};
  Coordinates upper{0, -2, 0.0625, 0, 2, 0.0625, 0.5, 0, 0.5625, -0.5, 0, 0.5625};
  check_pair(clipper, lower, upper, false, Contact::kDisjoint);
  for (int j = 2; j < 12; j += 3) upper[j] -= 0.0625;
  check_pair(clipper, lower, upper, false, Contact::kEdges);

  return phx::mc::test::finish("test_metric_layer_overlap");
}
