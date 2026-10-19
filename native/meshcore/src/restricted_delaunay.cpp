//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "restricted_delaunay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "capi_guard.hpp"
#include "counter_records.hpp"
#include "geometry3.hpp"
#include "mesh.hpp"
#include "parallel.hpp"
#include "predicates.hpp"
#include "tet_mesh.hpp"

namespace phx::mc {
namespace {

using namespace restricted_layout;
// One cell construction and one visit for each primal half-facet.
constexpr int64_t kCellPreparationWork = 1 + kTetVertices;

// Bound all expansion spills, not merely a typical filtered path: difference
// has <=2 components, cross <=16, dot(u,cross) <=192, squared norm <=24,
// denominator <=384 and each numerator <=3072. Summing every product-loop
// reserve and sum reserve (>24 components) gives 303 allocations / 714880
// payload bytes. Rays keep cross/negation inline and have one orient3d:
// both adaptive stages together require <=11 allocations / 5632 payload bytes.
// These cumulative bounds dominate live arena storage and include no work or
// source queries; orient3d charges only primitive-query telemetry.
// Center/ray expansion temporaries release within each sequential row. Only
// binary64 bounds and scalar statuses escape, so each fixed chunk reuses one
// row's cumulative bound; no expansion storage survives into the next row.
ParallelRequest bound_request(uint64_t allocations, uint64_t payload) {
  ParallelRequest request;
  request.parallelism = current_parallelism();
  request.lease_seconds = current_parallel_lease_seconds();
  request.arena_bytes_per_chunk = payload +
      allocations * native_parallel_allocation_bytes(0, alignof(double));
  return request;
}

struct RayBatch {
  int64_t point_count;
  const double* points;
  int64_t tet_count;
  const int32_t* tets;
  const double* centers;
  int64_t ray_count;
  const int32_t* facets;
  const int32_t* cells;
};

struct HalfFacet {
  std::array<int32_t, kFacetVertices> vertices;
  int32_t cell;
  int32_t opposite;
};

struct DualFacet {
  std::array<int32_t, kFacetVertices> vertices;
  std::array<int32_t, kIncidentCells> cells;
  std::array<double, kDualEndpoints> endpoints{};
  int32_t kind = kSegment;
  int32_t status = PHX_MC_OK;
};

using ExactVector = std::array<Expansion, kDimensions>;

ExactVector exact_cross(const ExactVector& a, const ExactVector& b) {
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
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

// The circumcenter is a rational construction in the exact represented input
// coordinates. Reuse expansion arithmetic for its polynomial numerator and
// denominator, then outward-round only the final division. A conditioning
// refusal leaves an unbounded interval rather than a guessed error radius.
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
  if (!(d[0] > 0.0) || !std::isfinite(d[1])) {
    return false;
  }
  const Expansion uu = exact_dot(u, u);
  const Expansion vv = exact_dot(v, v);
  const Expansion ww = exact_dot(w, w);
  for (int axis = 0; axis < kDimensions; ++axis) {
    const Expansion numerator = uu * vw[axis] + vv * wu[axis] + ww * uv[axis] +
                                denominator.scaled(p[0][axis]);
    const auto n = expansion_bounds(numerator);
    const double quotients[4] = {n[0] / d[0], n[0] / d[1],
                                n[1] / d[0], n[1] / d[1]};
    bounds[axis] = geometry::next_down(*std::min_element(quotients, quotients + 4));
    bounds[axis + kUpper] = geometry::next_up(*std::max_element(quotients, quotients + 4));
    if (!std::isfinite(bounds[axis]) || !std::isfinite(bounds[axis + kUpper])) {
      return false;
    }
  }
  return true;
}

int32_t construct_center_bounds(int64_t point_count, const double* points,
                                int64_t tet_count, const int32_t* tets,
                                double* bounds, int32_t* item_status) {
  if (point_count < kTetVertices || point_count > kMaxMeshPoints || tet_count < 1 ||
      points == nullptr || tets == nullptr || bounds == nullptr || item_status == nullptr ||
      !addressable(point_count, kDimensions, sizeof(double)) ||
      !addressable(tet_count, kTetVertices, sizeof(int32_t)) ||
      !addressable(tet_count, kCenterBounds, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, kDimensions, nullptr);
  if (valid != PHX_MC_OK) {
    return valid;
  }
  // Validate the complete batch before writing any output.
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + kTetVertices * cell;
    for (int vertex = 0; vertex < kTetVertices; ++vertex) {
      if (ids[vertex] < 0 || ids[vertex] >= point_count) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    if (orient3d(points + kDimensions * ids[0], points + kDimensions * ids[1],
                 points + kDimensions * ids[2], points + kDimensions * ids[3]) <= 0) {
      return PHX_MC_DEGENERATE_INPUT;
    }
  }
  return parallel_for(static_cast<std::size_t>(tet_count), bound_request(303, 714880),
      [&](std::size_t begin, std::size_t end, ChunkContext&) {
        for (std::size_t cell = begin; cell < end; ++cell) {
          const int32_t* ids = tets + kTetVertices * cell;
          const double* p[kTetVertices] = {
              points + kDimensions * ids[0], points + kDimensions * ids[1],
              points + kDimensions * ids[2], points + kDimensions * ids[3]};
          std::array<double, kCenterBounds> value{};
          const bool resolved = exact_center_bounds(p, value);
          item_status[cell] = resolved ? PHX_MC_OK : PHX_MC_DEGENERATE_INPUT;
          if (!resolved) {
            std::fill(value.begin(), value.begin() + kUpper, -std::numeric_limits<double>::infinity());
            std::fill(value.begin() + kUpper, value.end(), std::numeric_limits<double>::infinity());
          }
          std::copy(value.begin(), value.end(), bounds + kCenterBounds * cell);
        }
      });
}

bool enclose_ray(const double* const* face, const double* opposite,
                 const double* center, const double* domain,
                 std::array<double, kRayEndpointBounds>& endpoints, int32_t& kind) {
  for (int axis = 0; axis < kDimensions; ++axis) {
    if (!std::isfinite(center[axis]) || !std::isfinite(center[axis + kUpper])) {
      return false;
    }
  }
  ExactVector u, v;
  for (int axis = 0; axis < kDimensions; ++axis) {
    u[axis] = Expansion::difference(face[1][axis], face[0][axis]);
    v[axis] = Expansion::difference(face[2][axis], face[0][axis]);
  }
  ExactVector normal = exact_cross(u, v);
  if (orient3d(face[0], face[1], face[2], opposite) > 0) {
    for (Expansion& value : normal) {
      value = -value;
    }
  }
  std::array<std::array<double, 2>, kDimensions> direction;
  int exit_axis = -1;
  double exit_speed = 0.0;
  for (int axis = 0; axis < kDimensions; ++axis) {
    direction[axis] = expansion_bounds(normal[axis]);
    const double low = direction[axis][0];
    const double high = direction[axis][1];
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
  if (exit_axis < 0 || !std::isfinite(exit_speed)) {
    return false;
  }
  const bool positive = direction[exit_axis][0] > 0.0;
  const double offset = geometry::next_up(
      positive ? domain[exit_axis + kUpper] - center[exit_axis]
               : center[exit_axis + kUpper] - domain[exit_axis]);
  const double travel = geometry::next_up(std::max(0.0, offset / exit_speed));
  if (!std::isfinite(travel)) {
    return false;
  }
  std::copy(center, center + kCenterBounds, endpoints.begin());
  for (int axis = 0; axis < kDimensions; ++axis) {
    endpoints[axis + kRayEndLower] = geometry::next_down(
        center[axis] + geometry::next_down(travel * direction[axis][0]));
    endpoints[axis + kRayEndUpper] = geometry::next_up(
        center[axis + kUpper] + geometry::next_up(travel * direction[axis][1]));
    if (!std::isfinite(endpoints[axis + kRayEndLower]) || !std::isfinite(endpoints[axis + kRayEndUpper])) {
      return false;
    }
  }
  kind = kRay;
  return true;
}

int32_t prepare_ray_incidence(const RayBatch& in, std::vector<int32_t>& opposite) {
  for (int64_t row = 0; row < in.ray_count; ++row) {
    if (in.cells[row] < 0 || in.cells[row] >= in.tet_count) {
      return PHX_MC_INVALID_INPUT;
    }
    const int32_t* cell = in.tets + kTetVertices * in.cells[row];
    const int32_t* face = in.facets + kFacetVertices * row;
    for (int vertex = 0; vertex < kFacetVertices; ++vertex) {
      if (face[vertex] < 0 || face[vertex] >= in.point_count ||
          std::count(face, face + kFacetVertices, face[vertex]) != 1) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    int shared = 0;
    for (int vertex = 0; vertex < kTetVertices; ++vertex) {
      if (cell[vertex] < 0 || cell[vertex] >= in.point_count) {
        return PHX_MC_INVALID_INPUT;
      }
      if (std::find(face, face + kFacetVertices, cell[vertex]) != face + kFacetVertices) {
        ++shared;
      } else {
        opposite[static_cast<std::size_t>(row)] = cell[vertex];
      }
    }
    if (shared != kFacetVertices ||
        orient3d(in.points + kDimensions * cell[0], in.points + kDimensions * cell[1],
                 in.points + kDimensions * cell[2], in.points + kDimensions * cell[3]) <= 0) {
      return PHX_MC_INVALID_INPUT;
    }
    const double* center = in.centers + kCenterBounds * in.cells[row];
    for (int axis = 0; axis < kDimensions; ++axis) {
      if (std::isnan(center[axis]) || std::isnan(center[axis + kUpper]) ||
          center[axis] > center[axis + kUpper]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  return PHX_MC_OK;
}

int32_t publish_ray_bounds(const RayBatch& in, const std::vector<int32_t>& opposite,
                           const double* domain, double* bounds, int32_t* kinds,
                           int32_t* item_status) {
  return parallel_for(static_cast<std::size_t>(in.ray_count), bound_request(11, 5632),
      [&](std::size_t begin, std::size_t end, ChunkContext&) {
        for (std::size_t row = begin; row < end; ++row) {
          const int32_t* ids = in.facets + kFacetVertices * row;
          const double* face[kFacetVertices] = {
              in.points + kDimensions * ids[0], in.points + kDimensions * ids[1],
              in.points + kDimensions * ids[2]};
          std::array<double, kRayEndpointBounds> endpoints{};
          int32_t kind = kRay;
          const bool resolved = enclose_ray(
              face, in.points + kDimensions * opposite[row],
              in.centers + kCenterBounds * in.cells[row], domain, endpoints, kind);
          item_status[row] = resolved ? PHX_MC_OK : PHX_MC_DEGENERATE_INPUT;
          kinds[row] = kind;
          if (!resolved) {
            for (int endpoint = 0; endpoint < 2; ++endpoint) {
              std::fill(endpoints.begin() + kCenterBounds * endpoint,
                        endpoints.begin() + kCenterBounds * endpoint + kDimensions,
                        -std::numeric_limits<double>::infinity());
              std::fill(endpoints.begin() + kCenterBounds * endpoint + kDimensions,
                        endpoints.begin() + kCenterBounds * endpoint + kCenterBounds,
                        std::numeric_limits<double>::infinity());
            }
          }
          std::copy(endpoints.begin(), endpoints.end(), bounds + kRayEndpointBounds * row);
        }
      });
}

int32_t construct_ray_bounds(int64_t point_count, const double* points,
                             int64_t tet_count, const int32_t* tets,
                             const double* centers, int64_t ray_count,
                             const int32_t* facets, const int32_t* cells,
                             const double* domain, double* bounds,
                             int32_t* kinds, int32_t* item_status) {
  if (point_count < kTetVertices || point_count > kMaxMeshPoints || tet_count < 1 || ray_count < 0 ||
      points == nullptr || tets == nullptr || centers == nullptr || domain == nullptr ||
      (ray_count > 0 && (facets == nullptr || cells == nullptr || bounds == nullptr ||
                         kinds == nullptr || item_status == nullptr)) ||
      !addressable(point_count, kDimensions, sizeof(double)) ||
      !addressable(tet_count, kTetVertices, sizeof(int32_t)) ||
      !addressable(tet_count, kCenterBounds, sizeof(double)) ||
      !addressable(ray_count, kRayEndpointBounds, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, kDimensions, nullptr);
  if (valid != PHX_MC_OK) {
    return valid;
  }
  for (int axis = 0; axis < kDimensions; ++axis) {
    if (!std::isfinite(domain[axis]) || !std::isfinite(domain[axis + kUpper]) ||
        !(domain[axis] < domain[axis + kUpper])) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  std::vector<int32_t> opposite(static_cast<std::size_t>(ray_count), -1);
  const RayBatch input{point_count, points, tet_count, tets, centers, ray_count, facets, cells};
  const int32_t incidence = prepare_ray_incidence(input, opposite);
  if (incidence != PHX_MC_OK) return incidence;
  return publish_ray_bounds(input, opposite, domain, bounds, kinds, item_status);
}

// Numerical slab clipping. A negative result is only about this numerical
// dual; consumers cannot turn it into a certified source-domain exclusion.
bool clip_dual(const double* origin, const double* direction, const double* domain,
               bool ray, std::array<double, kDualEndpoints>& endpoints) {
  double low = 0.0;
  double high = ray ? std::numeric_limits<double>::infinity() : 1.0;
  for (int axis = 0; axis < kDimensions; ++axis) {
    if (direction[axis] == 0.0) {
      if (origin[axis] < domain[axis] || origin[axis] > domain[axis + kUpper]) {
        return false;
      }
      continue;
    }
    double first = (domain[axis] - origin[axis]) / direction[axis];
    double second = (domain[axis + kUpper] - origin[axis]) / direction[axis];
    if (first > second) {
      std::swap(first, second);
    }
    low = std::max(low, first);
    high = std::min(high, second);
    if (low > high) {
      return false;
    }
  }
  if (!std::isfinite(low) || !std::isfinite(high)) {
    return false;
  }
  for (int axis = 0; axis < kDimensions; ++axis) {
    endpoints[axis] = std::clamp(origin[axis] + low * direction[axis],
                               domain[axis], domain[axis + kUpper]);
    endpoints[axis + kUpper] = std::clamp(origin[axis] + high * direction[axis],
                                   domain[axis], domain[axis + kUpper]);
  }
  return true;
}

int32_t prepare_primal_facets(int64_t point_count, const double* points,
                              int64_t tet_count, const int32_t* tets,
                              std::vector<std::array<double, kDimensions>>& centers,
                              std::vector<bool>& center_valid,
                              std::vector<HalfFacet>& half_facets) {
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + cell * kTetVertices;
    for (int vertex = 0; vertex < kTetVertices; ++vertex) {
      if (ids[vertex] < 0 || ids[vertex] >= point_count) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    const double* p[kTetVertices] = {
        points + kDimensions * ids[0], points + kDimensions * ids[1],
        points + kDimensions * ids[2], points + kDimensions * ids[3]};
    if (orient3d(p[0], p[1], p[2], p[3]) <= 0) {
      return PHX_MC_DEGENERATE_INPUT;
    }
    center_valid[static_cast<std::size_t>(cell)] = geometry::tetrahedron_circumcenter(
        p[0], p[1], p[2], p[3], centers[static_cast<std::size_t>(cell)].data());
    for (int opposite = 0; opposite < kTetVertices; ++opposite) {
      HalfFacet facet{{}, static_cast<int32_t>(cell), ids[opposite]};
      int slot = 0;
      for (int vertex = 0; vertex < kTetVertices; ++vertex) {
        if (vertex != opposite) {
          facet.vertices[slot++] = ids[vertex];
        }
      }
      std::sort(facet.vertices.begin(), facet.vertices.end());
      half_facets.push_back(facet);
    }
  }
  std::sort(half_facets.begin(), half_facets.end(), [](const HalfFacet& a, const HalfFacet& b) {
    return a.vertices == b.vertices ? a.cell < b.cell : a.vertices < b.vertices;
  });
  return PHX_MC_OK;
}

void construct_facet_dual(const HalfFacet& a, const HalfFacet* b,
                           const double* p, const double* q, const double* r,
                           const double* points, const double* domain,
                           const std::vector<std::array<double, kDimensions>>& centers,
                           const std::vector<bool>& center_valid,
                           RestrictedDualCounters& counts, DualFacet& dual) {
  dual.kind = b == nullptr ? kRay : kSegment;
  ++(b == nullptr ? counts.hull_facets : counts.interior_facets);
  if (!center_valid[a.cell] || (b != nullptr && !center_valid[b->cell])) {
    dual.status = PHX_MC_DEGENERATE_INPUT;
    ++counts.degenerate_facets;
  } else {
    double direction[kDimensions];
    if (b != nullptr) {
      geometry::difference(centers[b->cell].data(), centers[a.cell].data(), direction);
    } else {
      double u[kDimensions], v[kDimensions];
      geometry::difference(q, p, u);
      geometry::difference(r, p, v);
      geometry::cross(u, v, direction);
      if (orient3d(p, q, r, points + kDimensions * a.opposite) > 0) {
        for (double& value : direction) {
          value = -value;
        }
      }
    }
    if (!geometry::finite3(direction)) {
      dual.status = PHX_MC_DEGENERATE_INPUT;
      ++counts.degenerate_facets;
    } else if (!clip_dual(centers[a.cell].data(), direction, domain,
                          b == nullptr, dual.endpoints)) {
      dual.kind = kDisjoint;
    }
  }
}

void publish_facet_duals(const std::vector<DualFacet>& duals, int32_t* facets,
                         int32_t* cells, double* endpoints, int32_t* kinds,
                         int32_t* item_status, int64_t* facet_count) {
  for (std::size_t row = 0; row < duals.size(); ++row) {
    const DualFacet& dual = duals[row];
    std::copy(dual.vertices.begin(), dual.vertices.end(), facets + kFacetVertices * row);
    std::copy(dual.cells.begin(), dual.cells.end(), cells + kIncidentCells * row);
    std::copy(dual.endpoints.begin(), dual.endpoints.end(), endpoints + kDualEndpoints * row);
    kinds[row] = dual.kind;
    item_status[row] = dual.status;
  }
  *facet_count = static_cast<int64_t>(duals.size());
}

int32_t construct_duals(int64_t point_count, const double* points, int64_t tet_count,
                        const int32_t* tets, const double* domain, int64_t max_facets,
                        int64_t work_limit, int32_t* facets, int32_t* cells,
                        double* endpoints, int32_t* kinds, int32_t* item_status,
                        int64_t* facet_count, int64_t* counters) {
  if (point_count < kTetVertices || point_count > kMaxMeshPoints || tet_count < 1 ||
      tet_count > std::numeric_limits<int32_t>::max() || max_facets < 1 ||
      work_limit < 0 || points == nullptr || tets == nullptr || domain == nullptr ||
      facets == nullptr || cells == nullptr || endpoints == nullptr || kinds == nullptr ||
      item_status == nullptr || facet_count == nullptr || counters == nullptr ||
      !addressable(point_count, kDimensions, sizeof(double)) ||
      !addressable(tet_count, kTetVertices, sizeof(int32_t)) ||
      !addressable(tet_count, kTetVertices, sizeof(HalfFacet)) ||
      !addressable(max_facets, kDualEndpoints, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, kDimensions, nullptr);
  if (valid != PHX_MC_OK) {
    return valid;
  }
  for (int axis = 0; axis < kDimensions; ++axis) {
    if (!std::isfinite(domain[axis]) || !std::isfinite(domain[axis + kUpper]) ||
        !(domain[axis] < domain[axis + kUpper])) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  // At least 2*T primal facets, at most 4*T. Refuse before work allocation.
  if (tet_count > work_limit / kCellPreparationWork || tet_count > max_facets / kIncidentCells) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  std::vector<std::array<double, kDimensions>> centers(static_cast<std::size_t>(tet_count));
  std::vector<bool> center_valid(static_cast<std::size_t>(tet_count), false);
  std::vector<HalfFacet> half_facets;
  half_facets.reserve(static_cast<std::size_t>(kTetVertices * tet_count));
  const int32_t prepared = prepare_primal_facets(
      point_count, points, tet_count, tets, centers, center_valid, half_facets);
  if (prepared != PHX_MC_OK) return prepared;
  std::vector<DualFacet> duals;
  duals.reserve(static_cast<std::size_t>(std::min(max_facets, kTetVertices * tet_count)));
  RestrictedDualCounters counts{kCellPreparationWork * tet_count, 0, 0, 0};
  CounterPublisher publisher{counts, counters, false};
  for (std::size_t first = 0; first < half_facets.size();) {
    std::size_t last = first + 1;
    while (last < half_facets.size() &&
           half_facets[last].vertices == half_facets[first].vertices) {
      ++last;
    }
    if (last - first > 2) {
      return PHX_MC_INVALID_INPUT;
    }
    if (static_cast<int64_t>(duals.size()) >= max_facets || counts.work_units >= work_limit) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    ++counts.work_units;
    const HalfFacet& a = half_facets[first];
    const HalfFacet* b = last - first == 2 ? &half_facets[first + 1] : nullptr;
    const double* p = points + kDimensions * a.vertices[0];
    const double* q = points + kDimensions * a.vertices[1];
    const double* r = points + kDimensions * a.vertices[2];
    if (b != nullptr) {
      if (orient3d(p, q, r, points + kDimensions * a.opposite) ==
          orient3d(p, q, r, points + kDimensions * b->opposite)) {
        return PHX_MC_INVALID_INPUT;
      }
      const int32_t* ids = tets + kTetVertices * a.cell;
      if (insphere(points + kDimensions * ids[0], points + kDimensions * ids[1], points + kDimensions * ids[2],
                   points + kDimensions * ids[3], points + kDimensions * b->opposite) > 0) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    DualFacet dual{a.vertices, {a.cell, b == nullptr ? -1 : b->cell}};
    construct_facet_dual(a, b, p, q, r, points, domain, centers, center_valid, counts, dual);
    duals.push_back(dual);
    first = last;
  }
  // Commit outputs only after every exact incidence and budget check passed.
  publish_facet_duals(duals, facets, cells, endpoints, kinds, item_status, facet_count);
  publisher.enabled = true;
  return PHX_MC_OK;
}

}  // namespace
}  // namespace phx::mc

extern "C" int32_t phx_mc_restricted_dual_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* domain, int64_t max_facets,
    int64_t work_limit, int32_t* facets, int32_t* cells, double* endpoints,
    int32_t* kinds, int32_t* item_status, int64_t* facet_count,
    int64_t* counters) {
  return phx::mc::guarded([&] {
    return phx::mc::construct_duals(point_count, points, tet_count, tets, domain,
                                    max_facets, work_limit, facets, cells, endpoints,
                                    kinds, item_status, facet_count, counters);
  });
}

extern "C" int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::construct_center_bounds(point_count, points, tet_count,
                                            tets, center_bounds, item_status);
  });
}

extern "C" int32_t phx_mc_restricted_rays_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* center_bounds, int64_t ray_count,
    const int32_t* facets, const int32_t* cells, const double* domain,
    double* endpoint_bounds, int32_t* kinds, int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::construct_ray_bounds(point_count, points, tet_count, tets,
                                         center_bounds, ray_count, facets, cells, domain,
                                         endpoint_bounds, kinds, item_status);
  });
}
