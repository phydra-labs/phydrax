//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Outward bounds of exact tetrahedron circumcenters.
#include "restricted_delaunay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "capi_guard.hpp"
#include "expansion.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "mesh.hpp"
#include "predicates.hpp"

namespace phx::mc {
namespace {
using namespace restricted_layout;
using ExactVector = std::array<Expansion, kDimensions>;

ExactVector exact_cross(const ExactVector& a, const ExactVector& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

Expansion exact_dot(const ExactVector& a, const ExactVector& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 2> expansion_bounds(const Expansion& value) {
  double lower = 0.0;
  double upper = 0.0;
  for (int index = 0; index < value.size(); ++index) {
    lower = geometry::next_down(lower + value.data()[index]);
    upper = geometry::next_up(upper + value.data()[index]);
  }
  return {lower, upper};
}

bool exact_center_bounds(const double* const* p, std::array<double, kCenterBounds>& bounds) {
  ExactVector u, v, w;
  for (int axis = 0; axis < kDimensions; ++axis) {
    u[axis] = Expansion::difference(p[1][axis], p[0][axis]);
    v[axis] = Expansion::difference(p[2][axis], p[0][axis]);
    w[axis] = Expansion::difference(p[3][axis], p[0][axis]);
  }
  const ExactVector vw = exact_cross(v, w);
  const ExactVector wu = exact_cross(w, u);
  const ExactVector uv = exact_cross(u, v);
  const Expansion denominator = exact_dot(u, vw).scaled(2.0);
  const auto d = expansion_bounds(denominator);
  if (!(d[0] > 0.0) || !std::isfinite(d[1])) return false;
  const Expansion uu = exact_dot(u, u);
  const Expansion vv = exact_dot(v, v);
  const Expansion ww = exact_dot(w, w);
  for (int axis = 0; axis < kDimensions; ++axis) {
    const Expansion numerator = uu * vw[axis] + vv * wu[axis] + ww * uv[axis] +
                                denominator.scaled(p[0][axis]);
    const auto n = expansion_bounds(numerator);
    const double quotients[4] = {n[0] / d[0], n[0] / d[1], n[1] / d[0], n[1] / d[1]};
    bounds[axis] = geometry::next_down(*std::min_element(quotients, quotients + 4));
    bounds[axis + kUpper] = geometry::next_up(*std::max_element(quotients, quotients + 4));
    if (!std::isfinite(bounds[axis]) || !std::isfinite(bounds[axis + kUpper])) return false;
  }
  return true;
}

int32_t validate_tets(int64_t point_count, const double* points, int64_t tet_count,
                      const int32_t* tets) {
  if (point_count > kMaxMeshPoints ||
      tet_count > std::numeric_limits<int32_t>::max()) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  if (point_count < kTetVertices || tet_count < 1 || points == nullptr || tets == nullptr ||
      !addressable(point_count, kDimensions, sizeof(double)) ||
      !addressable(tet_count, kTetVertices, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, kDimensions, nullptr);
  if (valid != PHX_MC_OK) return valid;
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + kTetVertices * cell;
    for (int vertex = 0; vertex < kTetVertices; ++vertex) {
      if (ids[vertex] < 0 || ids[vertex] >= point_count ||
          std::count(ids, ids + kTetVertices, ids[vertex]) != 1) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    if (orient3d(points + kDimensions * ids[0], points + kDimensions * ids[1],
                 points + kDimensions * ids[2], points + kDimensions * ids[3]) <= 0) {
      return PHX_MC_DEGENERATE_INPUT;
    }
  }
  return PHX_MC_OK;
}

int32_t construct_centers(int64_t point_count, const double* points, int64_t tet_count,
                          const int32_t* tets, double* bounds, int32_t* item_status) {
  if (item_status == nullptr || bounds == nullptr ||
      !addressable(tet_count, kCenterBounds, sizeof(double))) return PHX_MC_INVALID_ARGUMENT;
  const int32_t valid = validate_tets(point_count, points, tet_count, tets);
  if (valid != PHX_MC_OK) return valid;
  std::vector<double> staged(static_cast<std::size_t>(tet_count) * kCenterBounds);
  std::vector<int32_t> statuses(static_cast<std::size_t>(tet_count), PHX_MC_OK);
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + kTetVertices * cell;
    const double* p[kTetVertices] = {
        points + kDimensions * ids[0], points + kDimensions * ids[1],
        points + kDimensions * ids[2], points + kDimensions * ids[3]};
    std::array<double, kCenterBounds> value{};
    if (!exact_center_bounds(p, value)) {
      statuses[static_cast<std::size_t>(cell)] = PHX_MC_DEGENERATE_INPUT;
      std::fill(value.begin(), value.begin() + kUpper, -std::numeric_limits<double>::infinity());
      std::fill(value.begin() + kUpper, value.end(), std::numeric_limits<double>::infinity());
    }
    std::copy(value.begin(), value.end(), staged.begin() + kCenterBounds * cell);
  }
  std::copy(staged.begin(), staged.end(), bounds);
  std::copy(statuses.begin(), statuses.end(), item_status);
  return PHX_MC_OK;
}
}  // namespace
}  // namespace phx::mc
extern "C" PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::construct_centers(point_count, points, tet_count, tets,
                                      center_bounds, item_status);
  });
}
