//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Restricted-Delaunay numerical dual geometry and outward circumcenter/ray bounds.
#include "restricted_delaunay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "capi_guard.hpp"
#include "expansion.hpp"
#include "geometry3.hpp"
#include "mesh.hpp"
#include "predicates.hpp"

namespace phx::mc {
namespace {
using namespace restricted_layout;
using ExactVector = std::array<Expansion, kDimensions>;

struct HalfFacet {
  std::array<int32_t, kFacetVertices> vertices;
  int32_t cell;
  int32_t opposite;
};

struct DualFacet {
  std::array<int32_t, kFacetVertices> vertices{};
  std::array<int32_t, kIncidentCells> cells{{-1, -1}};
  std::array<double, kDualEndpoints> endpoints{};
  int32_t kind = kSegment;
  int32_t status = PHX_MC_OK;
};

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

bool finite3(const double* value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

// The center is the exact rational formula evaluated with the predicate owner's
// expansion arithmetic, then rounded once to its binary64 numerical view.
bool circumcenter(const double* a, const double* b, const double* c, const double* d,
                  double* center) {
  ExactVector u, v, w;
  for (int axis = 0; axis < kDimensions; ++axis) {
    u[axis] = Expansion::difference(b[axis], a[axis]);
    v[axis] = Expansion::difference(c[axis], a[axis]);
    w[axis] = Expansion::difference(d[axis], a[axis]);
  }
  const ExactVector vw = exact_cross(v, w);
  const ExactVector wu = exact_cross(w, u);
  const ExactVector uv = exact_cross(u, v);
  const Expansion determinant = exact_dot(u, vw);
  const double den = determinant.scaled(2.0).estimate();
  if (determinant.is_zero() || den == 0.0 || !std::isfinite(den)) return false;
  const Expansion uu = exact_dot(u, u);
  const Expansion vv = exact_dot(v, v);
  const Expansion ww = exact_dot(w, w);
  for (int axis = 0; axis < kDimensions; ++axis) {
    center[axis] = a[axis] + (uu * vw[axis] + vv * wu[axis] + ww * uv[axis]).estimate() / den;
  }
  return finite3(center);
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

bool valid_domain(const double* domain) {
  if (domain == nullptr) return false;
  for (int axis = 0; axis < kDimensions; ++axis) {
    if (!std::isfinite(domain[axis]) || !std::isfinite(domain[axis + kUpper]) ||
        !(domain[axis] < domain[axis + kUpper])) return false;
  }
  return true;
}

bool enclose_ray(const double* const* face, const double* opposite,
                 const double* center, const double* domain,
                 std::array<double, kRayEndpointBounds>& endpoints, int32_t& kind) {
  for (int axis = 0; axis < kDimensions; ++axis) {
    if (!std::isfinite(center[axis]) || !std::isfinite(center[axis + kUpper])) return false;
  }
  ExactVector u, v;
  for (int axis = 0; axis < kDimensions; ++axis) {
    u[axis] = Expansion::difference(face[1][axis], face[0][axis]);
    v[axis] = Expansion::difference(face[2][axis], face[0][axis]);
  }
  ExactVector direction = exact_cross(u, v);
  if (orient3d(face[0], face[1], face[2], opposite) > 0) {
    for (Expansion& value : direction) value = -value;
  }
  std::array<std::array<double, 2>, kDimensions> component_bounds;
  int exit_axis = -1;
  double exit_speed = 0.0;
  for (int axis = 0; axis < kDimensions; ++axis) {
    component_bounds[axis] = expansion_bounds(direction[axis]);
    const double low = component_bounds[axis][0];
    const double high = component_bounds[axis][1];
    if ((center[axis] > domain[axis + kUpper] && low >= 0.0) ||
        (center[axis + kUpper] < domain[axis] && high <= 0.0)) {
      std::copy(center, center + kCenterBounds, endpoints.begin());
      std::copy(center, center + kCenterBounds, endpoints.begin() + kRayEndLower);
      kind = kDisjoint;
      return true;
    }
    const double speed = low > 0.0 ? low : (high < 0.0 ? -high : 0.0);
    if (speed > exit_speed) {
      exit_axis = axis;
      exit_speed = speed;
    }
  }
  if (exit_axis < 0 || !std::isfinite(exit_speed)) return false;
  const bool positive = component_bounds[exit_axis][0] > 0.0;
  const double offset = geometry::next_up(
      positive ? domain[exit_axis + kUpper] - center[exit_axis]
               : center[exit_axis + kUpper] - domain[exit_axis]);
  const double travel = geometry::next_up(std::max(0.0, offset / exit_speed));
  if (!std::isfinite(travel)) return false;
  std::copy(center, center + kCenterBounds, endpoints.begin());
  for (int axis = 0; axis < kDimensions; ++axis) {
    endpoints[axis + kRayEndLower] = geometry::next_down(
        center[axis] + geometry::next_down(travel * component_bounds[axis][0]));
    endpoints[axis + kRayEndUpper] = geometry::next_up(
        center[axis + kUpper] + geometry::next_up(travel * component_bounds[axis][1]));
    if (!std::isfinite(endpoints[axis + kRayEndLower]) ||
        !std::isfinite(endpoints[axis + kRayEndUpper])) return false;
  }
  kind = kRay;
  return true;
}

bool clip_dual(const double* origin, const double* direction, const double* domain,
               bool ray, std::array<double, kDualEndpoints>& endpoints) {
  double low = 0.0;
  double high = ray ? std::numeric_limits<double>::infinity() : 1.0;
  for (int axis = 0; axis < kDimensions; ++axis) {
    if (direction[axis] == 0.0) {
      if (origin[axis] < domain[axis] || origin[axis] > domain[axis + kUpper]) return false;
      continue;
    }
    double first = (domain[axis] - origin[axis]) / direction[axis];
    double second = (domain[axis + kUpper] - origin[axis]) / direction[axis];
    if (first > second) std::swap(first, second);
    low = std::max(low, first);
    high = std::min(high, second);
    if (low > high) return false;
  }
  if (!std::isfinite(low) || !std::isfinite(high)) return false;
  for (int axis = 0; axis < kDimensions; ++axis) {
    endpoints[axis] = std::clamp(origin[axis] + low * direction[axis], domain[axis],
                                 domain[axis + kUpper]);
    endpoints[axis + kUpper] = std::clamp(origin[axis] + high * direction[axis], domain[axis],
                                          domain[axis + kUpper]);
  }
  return true;
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

int32_t construct_rays(int64_t point_count, const double* points, int64_t tet_count,
                       const int32_t* tets, const double* centers, int64_t ray_count,
                       const int32_t* facets, const int32_t* cells, const double* domain,
                       double* bounds, int32_t* kinds, int32_t* item_status) {
  if (ray_count < 0 || centers == nullptr || domain == nullptr ||
      !addressable(tet_count, kCenterBounds, sizeof(double)) ||
      !addressable(ray_count, kFacetVertices, sizeof(int32_t)) ||
      !addressable(ray_count, kRayEndpointBounds, sizeof(double)) ||
      (ray_count > 0 && (facets == nullptr || cells == nullptr || bounds == nullptr ||
                         kinds == nullptr || item_status == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_tets(point_count, points, tet_count, tets);
  if (valid != PHX_MC_OK) return valid;
  if (!valid_domain(domain)) return PHX_MC_INVALID_INPUT;
  std::vector<int32_t> opposite(static_cast<std::size_t>(ray_count), -1);
  for (int64_t row = 0; row < ray_count; ++row) {
    const int32_t cell = cells[kIncidentCells * row];
    if (cell < 0 || cell >= tet_count) return PHX_MC_INVALID_INPUT;
    const int32_t* tet = tets + kTetVertices * cell;
    const int32_t* face = facets + kFacetVertices * row;
    for (int vertex = 0; vertex < kFacetVertices; ++vertex) {
      if (face[vertex] < 0 || face[vertex] >= point_count ||
          std::count(face, face + kFacetVertices, face[vertex]) != 1) return PHX_MC_INVALID_INPUT;
    }
    int shared = 0;
    for (int vertex = 0; vertex < kTetVertices; ++vertex) {
      if (std::find(face, face + kFacetVertices, tet[vertex]) != face + kFacetVertices) {
        ++shared;
      } else {
        opposite[static_cast<std::size_t>(row)] = tet[vertex];
      }
    }
    if (shared != kFacetVertices) return PHX_MC_INVALID_INPUT;
    const double* center = centers + kCenterBounds * cells[kIncidentCells * row];
    for (int axis = 0; axis < kDimensions; ++axis) {
      if (std::isnan(center[axis]) || std::isnan(center[axis + kUpper]) ||
          center[axis] > center[axis + kUpper]) return PHX_MC_INVALID_INPUT;
    }
  }
  if (ray_count == 0) return PHX_MC_OK;
  std::vector<double> staged(static_cast<std::size_t>(ray_count) * kRayEndpointBounds);
  std::vector<int32_t> staged_kinds(static_cast<std::size_t>(ray_count), kRay);
  std::vector<int32_t> statuses(static_cast<std::size_t>(ray_count), PHX_MC_OK);
  for (int64_t row = 0; row < ray_count; ++row) {
    const int32_t* ids = facets + kFacetVertices * row;
    const double* face[kFacetVertices] = {points + kDimensions * ids[0],
                                          points + kDimensions * ids[1],
                                          points + kDimensions * ids[2]};
    std::array<double, kRayEndpointBounds> value{};
    if (!enclose_ray(face, points + kDimensions * opposite[static_cast<std::size_t>(row)],
                     centers + kCenterBounds * cells[kIncidentCells * row], domain, value,
                     staged_kinds[static_cast<std::size_t>(row)])) {
      statuses[static_cast<std::size_t>(row)] = PHX_MC_DEGENERATE_INPUT;
      for (int endpoint = 0; endpoint < 2; ++endpoint) {
        std::fill(value.begin() + kCenterBounds * endpoint,
                  value.begin() + kCenterBounds * endpoint + kDimensions,
                  -std::numeric_limits<double>::infinity());
        std::fill(value.begin() + kCenterBounds * endpoint + kDimensions,
                  value.begin() + kCenterBounds * (endpoint + 1),
                  std::numeric_limits<double>::infinity());
      }
    }
    std::copy(value.begin(), value.end(), staged.begin() + kRayEndpointBounds * row);
  }
  std::copy(staged.begin(), staged.end(), bounds);
  std::copy(staged_kinds.begin(), staged_kinds.end(), kinds);
  std::copy(statuses.begin(), statuses.end(), item_status);
  return PHX_MC_OK;
}

int32_t construct_duals(int64_t point_count, const double* points, int64_t tet_count,
                        const int32_t* tets, const double* domain, int64_t facet_capacity,
                        int32_t* facets, int32_t* cells, double* endpoints, int32_t* kinds,
                        int32_t* item_status, int64_t* facet_count) {
  if (facet_capacity < 0 || facet_count == nullptr || domain == nullptr ||
      !addressable(facet_capacity, kFacetVertices, sizeof(int32_t)) ||
      !addressable(facet_capacity, kIncidentCells, sizeof(int32_t)) ||
      !addressable(facet_capacity, kDualEndpoints, sizeof(double)) ||
      (facet_capacity > 0 && (facets == nullptr || cells == nullptr || endpoints == nullptr ||
                              kinds == nullptr || item_status == nullptr))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_tets(point_count, points, tet_count, tets);
  if (valid != PHX_MC_OK) return valid;
  if (!valid_domain(domain)) return PHX_MC_INVALID_INPUT;
  std::vector<std::array<double, kDimensions>> centers(static_cast<std::size_t>(tet_count));
  std::vector<bool> center_valid(static_cast<std::size_t>(tet_count), false);
  std::vector<HalfFacet> half_facets;
  half_facets.reserve(static_cast<std::size_t>(kTetVertices) * static_cast<std::size_t>(tet_count));
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + kTetVertices * cell;
    const double* p[kTetVertices] = {points + kDimensions * ids[0], points + kDimensions * ids[1],
                                     points + kDimensions * ids[2], points + kDimensions * ids[3]};
    center_valid[static_cast<std::size_t>(cell)] =
        circumcenter(p[0], p[1], p[2], p[3], centers[static_cast<std::size_t>(cell)].data());
    for (int opposite = 0; opposite < kTetVertices; ++opposite) {
      HalfFacet facet{{}, static_cast<int32_t>(cell), ids[opposite]};
      int slot = 0;
      for (int vertex = 0; vertex < kTetVertices; ++vertex) {
        if (vertex != opposite) facet.vertices[slot++] = ids[vertex];
      }
      std::sort(facet.vertices.begin(), facet.vertices.end());
      half_facets.push_back(facet);
    }
  }
  std::sort(half_facets.begin(), half_facets.end(), [](const HalfFacet& a, const HalfFacet& b) {
    return a.vertices == b.vertices ? a.cell < b.cell : a.vertices < b.vertices;
  });
  std::vector<DualFacet> duals;
  for (std::size_t first = 0; first < half_facets.size();) {
    std::size_t last = first + 1;
    while (last < half_facets.size() && half_facets[last].vertices == half_facets[first].vertices) ++last;
    if (last - first > 2) return PHX_MC_INVALID_INPUT;
    const HalfFacet& a = half_facets[first];
    const HalfFacet* b = last - first == 2 ? &half_facets[first + 1] : nullptr;
    const double* p = points + kDimensions * a.vertices[0];
    const double* q = points + kDimensions * a.vertices[1];
    const double* r = points + kDimensions * a.vertices[2];
    if (b != nullptr) {
      const int side_a = orient3d(p, q, r, points + kDimensions * a.opposite);
      const int side_b = orient3d(p, q, r, points + kDimensions * b->opposite);
      if (side_a == side_b) return PHX_MC_INVALID_INPUT;
      const int32_t* ids = tets + kTetVertices * a.cell;
      if (insphere(points + kDimensions * ids[0], points + kDimensions * ids[1],
                   points + kDimensions * ids[2], points + kDimensions * ids[3],
                   points + kDimensions * b->opposite) > 0) return PHX_MC_INVALID_INPUT;
    }
    DualFacet dual;
    dual.vertices = a.vertices;
    dual.cells = {a.cell, b == nullptr ? -1 : b->cell};
    dual.kind = b == nullptr ? kRay : kSegment;
    if (!center_valid[static_cast<std::size_t>(a.cell)] ||
        (b != nullptr && !center_valid[static_cast<std::size_t>(b->cell)])) {
      dual.status = PHX_MC_DEGENERATE_INPUT;
    } else {
      double direction[kDimensions];
      if (b != nullptr) {
        geometry::difference(centers[static_cast<std::size_t>(b->cell)].data(),
                             centers[static_cast<std::size_t>(a.cell)].data(), direction);
      } else {
        double u[kDimensions], v[kDimensions];
        geometry::difference(q, p, u);
        geometry::difference(r, p, v);
        geometry::cross(u, v, direction);
        if (orient3d(p, q, r, points + kDimensions * a.opposite) > 0) {
          for (double& value : direction) value = -value;
        }
      }
      if (!finite3(direction)) {
        dual.status = PHX_MC_DEGENERATE_INPUT;
      } else if (!clip_dual(centers[static_cast<std::size_t>(a.cell)].data(), direction,
                            domain, b == nullptr, dual.endpoints)) {
        dual.kind = kDisjoint;
      }
    }
    duals.push_back(dual);
    first = last;
  }
  if (duals.size() > static_cast<std::size_t>(facet_capacity)) return PHX_MC_CAPACITY_EXCEEDED;
  for (std::size_t row = 0; row < duals.size(); ++row) {
    const DualFacet& dual = duals[row];
    std::copy(dual.vertices.begin(), dual.vertices.end(), facets + kFacetVertices * row);
    std::copy(dual.cells.begin(), dual.cells.end(), cells + kIncidentCells * row);
    std::copy(dual.endpoints.begin(), dual.endpoints.end(), endpoints + kDualEndpoints * row);
    kinds[row] = dual.kind;
    item_status[row] = dual.status;
  }
  *facet_count = static_cast<int64_t>(duals.size());
  return PHX_MC_OK;
}
}  // namespace
}  // namespace phx::mc

extern "C" PHX_MC_API int32_t phx_mc_restricted_dual_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* domain, int64_t facet_capacity,
    int32_t* facets, int32_t* cells, double* endpoints, int32_t* kinds,
    int32_t* item_status, int64_t* facet_count) {
  return phx::mc::guarded([&] {
    return phx::mc::construct_duals(point_count, points, tet_count, tets, domain,
                                    facet_capacity, facets, cells, endpoints, kinds,
                                    item_status, facet_count);
  });
}

extern "C" PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::construct_centers(point_count, points, tet_count, tets,
                                      center_bounds, item_status);
  });
}

extern "C" PHX_MC_API int32_t phx_mc_restricted_rays_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* center_bounds, int64_t ray_count,
    const int32_t* facets, const int32_t* cells, const double* domain,
    double* endpoint_bounds, int32_t* kinds, int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::construct_rays(point_count, points, tet_count, tets, center_bounds,
                                    ray_count, facets, cells, domain, endpoint_bounds,
                                    kinds, item_status);
  });
}
