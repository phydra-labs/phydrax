//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Outward bounds of exact tetrahedron circumcenters.
#include <array>
#include <cstdint>

#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"

namespace {
struct CenterBounds {
  int32_t status = -1;
  std::array<double, 6> bounds{};
  std::array<int32_t, 1> item_status{};

  bool operator==(const CenterBounds&) const = default;
};

CenterBounds center_bounds(const std::array<double, 12>& points) {
  constexpr int32_t tets[] = {0, 1, 2, 3};
  CenterBounds result;
  result.status = phx_mc_restricted_centers_3d(4, points.data(), 1, tets, result.bounds.data(),
                                               result.item_status.data());
  return result;
}

// The unit corner tetrahedron has circumcenter (1/2, 1/2, 1/2); the symmetric
// tetrahedron is centered exactly at the origin. Both give their
// round-to-nearest results in every caller mode; a subnormal coordinate lies
// outside the exact coordinate domain.
void circumcenter_bounds() {
  constexpr std::array<double, 12> corner{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr std::array<double, 12> symmetric{1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1};
  constexpr std::array<double, 12> subnormal{0, 0, 0, 1, 0, 0, 0, 1, 0, 0x1p-1060, 0, 1};
  const CenterBounds corner_reference = center_bounds(corner);
  PHX_CHECK(corner_reference.status == PHX_MC_OK);
  PHX_CHECK(corner_reference.item_status[0] == PHX_MC_OK);
  const CenterBounds symmetric_reference = center_bounds(symmetric);
  PHX_CHECK(symmetric_reference.status == PHX_MC_OK);
  PHX_CHECK(symmetric_reference.item_status[0] == PHX_MC_OK);
  for (int axis = 0; axis < 3; ++axis) {
    PHX_CHECK(corner_reference.bounds[axis] <= 0.5);
    PHX_CHECK(corner_reference.bounds[axis + 3] >= 0.5);
    PHX_CHECK(symmetric_reference.bounds[axis] <= 0.0);
    PHX_CHECK(symmetric_reference.bounds[axis + 3] >= 0.0);
  }
  PHX_CHECK(center_bounds(subnormal).status == PHX_MC_RANGE_ERROR);
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    PHX_CHECK(center_bounds(corner) == corner_reference);
    PHX_CHECK(center_bounds(symmetric) == symmetric_reference);
    PHX_CHECK(center_bounds(subnormal).status == PHX_MC_RANGE_ERROR);
  });
}
}  // namespace

int main() {
  circumcenter_bounds();
  return phx::mc::test::finish("test_restricted_delaunay");
}
