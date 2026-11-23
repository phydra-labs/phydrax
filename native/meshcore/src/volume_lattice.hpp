//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace phx::mc {

// A single lattice coordinate convention for both direct wall preparation
// and the adaptive field's volume-owned seeding.
inline constexpr double kBccLatticeMargin = 0.98;
inline constexpr std::array<double, 3> kBccLatticeShift{0.31, 0.47, 0.13};
using BccRotation = std::array<std::array<double, 3>, 3>;

BccRotation bcc_lattice_rotation();
double bcc_lattice_spacing(double circumradius);

inline void bcc_lattice_point(const double* middle, const BccRotation& rotation,
                              double spacing, std::int64_t shift, std::int64_t i,
                              std::int64_t j, std::int64_t k, double* point) {
  const double half = 0.5 * static_cast<double>(shift);
  const double local[]{(static_cast<double>(i) + kBccLatticeShift[0] + half) * spacing,
                       (static_cast<double>(j) + kBccLatticeShift[1] + half) * spacing,
                       (static_cast<double>(k) + kBccLatticeShift[2] + half) * spacing};
  for (int axis = 0; axis < 3; ++axis) {
    point[axis] = middle[axis] + rotation[axis][0] * local[0] + rotation[axis][1] * local[1] +
                  rotation[axis][2] * local[2];
  }
}

// Add the crossing of a triangle by the +z ray from x + (eps, eps^2, 0),
// with eps infinitesimal. False denotes a coplanar projected hit.
bool add_volume_ray_crossing(const double* a, const double* b, const double* c,
                             const double* x, std::int64_t& winding);

// Generate packed xyz BCC site proposals in (shift, k, j, i) order. The
// input wall is a prepared, closed, oriented triangle chain. PLC recovery
// owns final support. On refusal (unrepresentable or over budget), sites
// is unchanged.
bool prepare_owned_volume_lattice(std::span<const double> points,
                                 std::span<const std::int32_t> triangles,
                                 double circumradius, std::vector<double>& sites);

}  // namespace phx::mc
