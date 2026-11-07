//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Restricted-Delaunay dual incidence, clipping, and outward ray enclosures.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "fp_hazards.hpp"
#include "phydrax_meshcore.h"

namespace {
void one_tetrahedron_dual() {
  constexpr double points[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr int32_t tets[] = {0, 1, 2, 3};
  constexpr double domain[] = {0, 0, 0, 1, 1, 1};
  std::array<int32_t, 12> facets{};
  std::array<int32_t, 8> cells{};
  std::array<double, 24> endpoints{};
  std::array<int32_t, 4> kinds{};
  std::array<int32_t, 4> statuses{};
  int64_t facet_count = -1;
  PHX_CHECK(phx_mc_restricted_dual_3d(
                4, points, 1, tets, domain, 4, facets.data(), cells.data(), endpoints.data(),
                kinds.data(), statuses.data(), &facet_count) == PHX_MC_OK);
  PHX_CHECK(facet_count == 4);
  PHX_CHECK(std::all_of(kinds.begin(), kinds.end(), [](int32_t kind) { return kind == 1; }));
  PHX_CHECK(std::all_of(statuses.begin(), statuses.end(), [](int32_t s) { return s == PHX_MC_OK; }));
  for (int64_t row = 0; row < facet_count; ++row) {
    PHX_CHECK(cells[2 * row] == 0);
    PHX_CHECK(cells[2 * row + 1] == -1);
    for (int axis = 0; axis < 3; ++axis) {
      PHX_CHECK(endpoints[6 * row + axis] >= 0.0);
      PHX_CHECK(endpoints[6 * row + axis] <= 1.0);
      PHX_CHECK(endpoints[6 * row + axis + 3] >= 0.0);
      PHX_CHECK(endpoints[6 * row + axis + 3] <= 1.0);
    }
  }

  std::array<double, 6> center_bounds{};
  std::array<int32_t, 1> center_status{};
  PHX_CHECK(phx_mc_restricted_centers_3d(4, points, 1, tets, center_bounds.data(),
                                         center_status.data()) == PHX_MC_OK);
  PHX_CHECK(center_status[0] == PHX_MC_OK);
  for (int axis = 0; axis < 3; ++axis) {
    PHX_CHECK(center_bounds[axis] <= 0.5);
    PHX_CHECK(center_bounds[axis + 3] >= 0.5);
  }

  std::array<double, 48> ray_bounds{};
  std::array<int32_t, 4> ray_kinds{};
  std::array<int32_t, 4> ray_status{};
  PHX_CHECK(phx_mc_restricted_rays_3d(
                4, points, 1, tets, center_bounds.data(), facet_count, facets.data(), cells.data(),
                domain, ray_bounds.data(), ray_kinds.data(), ray_status.data()) == PHX_MC_OK);
  PHX_CHECK(std::all_of(ray_kinds.begin(), ray_kinds.end(), [](int32_t kind) { return kind == 1; }));
  PHX_CHECK(std::all_of(ray_status.begin(), ray_status.end(),
                        [](int32_t s) { return s == PHX_MC_OK; }));
  for (double bound : ray_bounds) PHX_CHECK(std::isfinite(bound));
}

struct RestrictedDual {
  std::array<int32_t, 3> status{-1, -1, -1};
  std::array<int32_t, 12> facets{};
  std::array<int32_t, 8> cells{};
  std::array<double, 24> endpoints{};
  std::array<int32_t, 4> kinds{};
  std::array<int32_t, 4> statuses{};
  int64_t facet_count = -1;
  std::array<double, 6> center_bounds{};
  std::array<int32_t, 1> center_status{};
  std::array<double, 48> ray_bounds{};
  std::array<int32_t, 4> ray_kinds{};
  std::array<int32_t, 4> ray_status{};

  bool operator==(const RestrictedDual&) const = default;
};

RestrictedDual restricted_dual(const std::array<double, 12>& points,
                               const std::array<double, 6>& domain) {
  constexpr int32_t tets[] = {0, 1, 2, 3};
  RestrictedDual result;
  result.status[0] = phx_mc_restricted_dual_3d(
      4, points.data(), 1, tets, domain.data(), 4, result.facets.data(), result.cells.data(),
      result.endpoints.data(), result.kinds.data(), result.statuses.data(), &result.facet_count);
  result.status[1] = phx_mc_restricted_centers_3d(4, points.data(), 1, tets,
                                                  result.center_bounds.data(),
                                                  result.center_status.data());
  const int64_t ray_count = result.facet_count < 0 ? 0 : result.facet_count;
  result.status[2] = phx_mc_restricted_rays_3d(
      4, points.data(), 1, tets, result.center_bounds.data(), ray_count, result.facets.data(),
      result.cells.data(), domain.data(), result.ray_bounds.data(), result.ray_kinds.data(),
      result.ray_status.data());
  return result;
}

// The unit corner and the symmetric tetrahedron (exactly centered at the origin)
// give their round-to-nearest results in every caller mode; a subnormal
// coordinate lies outside the exact coordinate domain.
void floating_point_hazards() {
  constexpr std::array<double, 12> corner{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr std::array<double, 6> unit_box{0, 0, 0, 1, 1, 1};
  constexpr std::array<double, 12> symmetric{1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1};
  constexpr std::array<double, 6> centered_box{-1, -1, -1, 1, 1, 1};
  constexpr std::array<double, 12> subnormal{0, 0, 0, 1, 0, 0, 0, 1, 0, 0x1p-1060, 0, 1};
  const RestrictedDual corner_reference = restricted_dual(corner, unit_box);
  PHX_CHECK(corner_reference.status == (std::array<int32_t, 3>{PHX_MC_OK, PHX_MC_OK, PHX_MC_OK}));
  PHX_CHECK(corner_reference.facet_count == 4);
  PHX_CHECK(corner_reference.center_status[0] == PHX_MC_OK);
  const RestrictedDual symmetric_reference = restricted_dual(symmetric, centered_box);
  PHX_CHECK(symmetric_reference.status ==
            (std::array<int32_t, 3>{PHX_MC_OK, PHX_MC_OK, PHX_MC_OK}));
  PHX_CHECK(symmetric_reference.facet_count == 4);
  PHX_CHECK(symmetric_reference.center_status[0] == PHX_MC_OK);
  for (int axis = 0; axis < 3; ++axis) {
    PHX_CHECK(symmetric_reference.center_bounds[axis] <= 0.0);
    PHX_CHECK(symmetric_reference.center_bounds[axis + 3] >= 0.0);
  }
  const std::array<int32_t, 3> out_of_domain{PHX_MC_RANGE_ERROR, PHX_MC_RANGE_ERROR,
                                             PHX_MC_RANGE_ERROR};
  PHX_CHECK(restricted_dual(subnormal, unit_box).status == out_of_domain);
  phx::mc::test::under_floating_point_hazards([&](const phx::mc::test::FloatingPointHazard&) {
    PHX_CHECK(restricted_dual(corner, unit_box) == corner_reference);
    PHX_CHECK(restricted_dual(symmetric, centered_box) == symmetric_reference);
    PHX_CHECK(restricted_dual(subnormal, unit_box).status == out_of_domain);
  });
}
}  // namespace

int main() {
  one_tetrahedron_dual();
  floating_point_hazards();
  return phx::mc::test::finish("test_restricted_delaunay");
}
