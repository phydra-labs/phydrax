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
#include "mesh.hpp"
#include "predicates.hpp"
#include "tet_mesh.hpp"

namespace phx::mc {
namespace {

struct HalfFacet {
  std::array<int32_t, 3> vertices;
  int32_t cell;
  int32_t opposite;
};

struct DualFacet {
  std::array<int32_t, 3> vertices;
  std::array<int32_t, 2> cells;
  std::array<double, 6> endpoints{};
  int32_t kind = 0;
  int32_t status = PHX_MC_OK;
};

using ExactVector = std::array<Expansion, 3>;

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
    lower = std::nextafter(lower + value.data()[index],
                          -std::numeric_limits<double>::infinity());
    upper = std::nextafter(upper + value.data()[index],
                          std::numeric_limits<double>::infinity());
  }
  return {lower, upper};
}

// The circumcenter is a rational construction in the exact represented input
// coordinates. Reuse expansion arithmetic for its polynomial numerator and
// denominator, then outward-round only the final division. A conditioning
// refusal leaves an unbounded interval rather than a guessed error radius.
bool exact_center_bounds(const double* const* p, std::array<double, 6>& bounds) {
  ExactVector u, v, w;
  for (int axis = 0; axis < 3; ++axis) {
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
  for (int axis = 0; axis < 3; ++axis) {
    const Expansion numerator = uu * vw[axis] + vv * wu[axis] + ww * uv[axis] +
                                denominator.scaled(p[0][axis]);
    const auto n = expansion_bounds(numerator);
    const double quotients[4] = {n[0] / d[0], n[0] / d[1],
                                n[1] / d[0], n[1] / d[1]};
    bounds[axis] = std::nextafter(*std::min_element(quotients, quotients + 4),
                                 -std::numeric_limits<double>::infinity());
    bounds[axis + 3] = std::nextafter(*std::max_element(quotients, quotients + 4),
                                     std::numeric_limits<double>::infinity());
    if (!std::isfinite(bounds[axis]) || !std::isfinite(bounds[axis + 3])) {
      return false;
    }
  }
  return true;
}

int32_t construct_center_bounds(int64_t point_count, const double* points,
                                int64_t tet_count, const int32_t* tets,
                                double* bounds, int32_t* item_status) {
  if (point_count < 4 || point_count > kMaxMeshPoints || tet_count < 1 ||
      points == nullptr || tets == nullptr || bounds == nullptr || item_status == nullptr ||
      !addressable(point_count, 3, sizeof(double)) ||
      !addressable(tet_count, 4, sizeof(int32_t)) ||
      !addressable(tet_count, 6, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, 3, nullptr);
  if (valid != PHX_MC_OK) {
    return valid;
  }
  // Validate the complete batch before writing any output.
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + 4 * cell;
    for (int vertex = 0; vertex < 4; ++vertex) {
      if (ids[vertex] < 0 || ids[vertex] >= point_count) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    if (orient3d(points + 3 * ids[0], points + 3 * ids[1],
                 points + 3 * ids[2], points + 3 * ids[3]) <= 0) {
      return PHX_MC_DEGENERATE_INPUT;
    }
  }
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + 4 * cell;
    const double* p[4] = {points + 3 * ids[0], points + 3 * ids[1],
                          points + 3 * ids[2], points + 3 * ids[3]};
    std::array<double, 6> value{};
    const bool resolved = exact_center_bounds(p, value);
    item_status[cell] = resolved ? PHX_MC_OK : PHX_MC_DEGENERATE_INPUT;
    if (!resolved) {
      std::fill(value.begin(), value.begin() + 3, -std::numeric_limits<double>::infinity());
      std::fill(value.begin() + 3, value.end(), std::numeric_limits<double>::infinity());
    }
    std::copy(value.begin(), value.end(), bounds + 6 * cell);
  }
  return PHX_MC_OK;
}

bool enclose_ray(const double* const* face, const double* opposite,
                 const double* center, const double* domain,
                 std::array<double, 12>& endpoints, int32_t& kind) {
  const double infinity = std::numeric_limits<double>::infinity();
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(center[axis]) || !std::isfinite(center[axis + 3])) {
      return false;
    }
  }
  ExactVector u, v;
  for (int axis = 0; axis < 3; ++axis) {
    u[axis] = Expansion::difference(face[1][axis], face[0][axis]);
    v[axis] = Expansion::difference(face[2][axis], face[0][axis]);
  }
  ExactVector normal = exact_cross(u, v);
  if (orient3d(face[0], face[1], face[2], opposite) > 0) {
    for (Expansion& value : normal) {
      value = -value;
    }
  }
  std::array<std::array<double, 2>, 3> direction;
  int exit_axis = -1;
  double exit_speed = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    direction[axis] = expansion_bounds(normal[axis]);
    const double low = direction[axis][0];
    const double high = direction[axis][1];
    if ((center[axis] > domain[axis + 3] && low >= 0.0) ||
        (center[axis + 3] < domain[axis] && high <= 0.0)) {
      std::copy(center, center + 6, endpoints.begin());
      std::copy(center, center + 6, endpoints.begin() + 6);
      kind = 2;
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
  const double offset = std::nextafter(
      positive ? domain[exit_axis + 3] - center[exit_axis]
               : center[exit_axis + 3] - domain[exit_axis], infinity);
  const double travel = std::nextafter(std::max(0.0, offset / exit_speed), infinity);
  if (!std::isfinite(travel)) {
    return false;
  }
  std::copy(center, center + 6, endpoints.begin());
  for (int axis = 0; axis < 3; ++axis) {
    endpoints[axis + 6] = std::nextafter(
        center[axis] + std::nextafter(travel * direction[axis][0], -infinity),
        -infinity);
    endpoints[axis + 9] = std::nextafter(
        center[axis + 3] + std::nextafter(travel * direction[axis][1], infinity),
        infinity);
    if (!std::isfinite(endpoints[axis + 6]) || !std::isfinite(endpoints[axis + 9])) {
      return false;
    }
  }
  kind = 1;
  return true;
}

int32_t construct_ray_bounds(int64_t point_count, const double* points,
                             int64_t tet_count, const int32_t* tets,
                             const double* centers, int64_t ray_count,
                             const int32_t* facets, const int32_t* cells,
                             const double* domain, double* bounds,
                             int32_t* kinds, int32_t* item_status) {
  if (point_count < 4 || point_count > kMaxMeshPoints || tet_count < 1 || ray_count < 0 ||
      points == nullptr || tets == nullptr || centers == nullptr || domain == nullptr ||
      (ray_count > 0 && (facets == nullptr || cells == nullptr || bounds == nullptr ||
                         kinds == nullptr || item_status == nullptr)) ||
      !addressable(point_count, 3, sizeof(double)) ||
      !addressable(tet_count, 4, sizeof(int32_t)) ||
      !addressable(tet_count, 6, sizeof(double)) ||
      !addressable(ray_count, 12, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, 3, nullptr);
  if (valid != PHX_MC_OK) {
    return valid;
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(domain[axis]) || !std::isfinite(domain[axis + 3]) ||
        !(domain[axis] < domain[axis + 3])) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  std::vector<int32_t> opposite(static_cast<std::size_t>(ray_count), -1);
  for (int64_t row = 0; row < ray_count; ++row) {
    if (cells[row] < 0 || cells[row] >= tet_count) {
      return PHX_MC_INVALID_INPUT;
    }
    const int32_t* cell = tets + 4 * cells[row];
    const int32_t* face = facets + 3 * row;
    for (int vertex = 0; vertex < 3; ++vertex) {
      if (face[vertex] < 0 || face[vertex] >= point_count ||
          std::count(face, face + 3, face[vertex]) != 1) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    int shared = 0;
    for (int vertex = 0; vertex < 4; ++vertex) {
      if (cell[vertex] < 0 || cell[vertex] >= point_count) {
        return PHX_MC_INVALID_INPUT;
      }
      if (std::find(face, face + 3, cell[vertex]) != face + 3) {
        ++shared;
      } else {
        opposite[static_cast<std::size_t>(row)] = cell[vertex];
      }
    }
    if (shared != 3 || orient3d(points + 3 * cell[0], points + 3 * cell[1],
                               points + 3 * cell[2], points + 3 * cell[3]) <= 0) {
      return PHX_MC_INVALID_INPUT;
    }
    const double* center = centers + 6 * cells[row];
    for (int axis = 0; axis < 3; ++axis) {
      if (std::isnan(center[axis]) || std::isnan(center[axis + 3]) ||
          center[axis] > center[axis + 3]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int64_t row = 0; row < ray_count; ++row) {
    const int32_t* ids = facets + 3 * row;
    const double* face[3] = {points + 3 * ids[0], points + 3 * ids[1], points + 3 * ids[2]};
    std::array<double, 12> endpoints{};
    int32_t kind = 1;
    const bool resolved = enclose_ray(
        face, points + 3 * opposite[static_cast<std::size_t>(row)],
        centers + 6 * cells[row], domain, endpoints, kind);
    item_status[row] = resolved ? PHX_MC_OK : PHX_MC_DEGENERATE_INPUT;
    kinds[row] = kind;
    if (!resolved) {
      for (int endpoint = 0; endpoint < 2; ++endpoint) {
        std::fill(endpoints.begin() + 6 * endpoint, endpoints.begin() + 6 * endpoint + 3,
                  -std::numeric_limits<double>::infinity());
        std::fill(endpoints.begin() + 6 * endpoint + 3, endpoints.begin() + 6 * endpoint + 6,
                  std::numeric_limits<double>::infinity());
      }
    }
    std::copy(endpoints.begin(), endpoints.end(), bounds + 12 * row);
  }
  return PHX_MC_OK;
}

// Numerical slab clipping. A negative result is only about this numerical
// dual; consumers cannot turn it into a certified source-domain exclusion.
bool clip_dual(const double* origin, const double* direction, const double* domain,
               bool ray, std::array<double, 6>& endpoints) {
  double low = 0.0;
  double high = ray ? std::numeric_limits<double>::infinity() : 1.0;
  for (int axis = 0; axis < 3; ++axis) {
    if (direction[axis] == 0.0) {
      if (origin[axis] < domain[axis] || origin[axis] > domain[axis + 3]) {
        return false;
      }
      continue;
    }
    double first = (domain[axis] - origin[axis]) / direction[axis];
    double second = (domain[axis + 3] - origin[axis]) / direction[axis];
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
  for (int axis = 0; axis < 3; ++axis) {
    endpoints[axis] = std::clamp(origin[axis] + low * direction[axis],
                               domain[axis], domain[axis + 3]);
    endpoints[axis + 3] = std::clamp(origin[axis] + high * direction[axis],
                                   domain[axis], domain[axis + 3]);
  }
  return true;
}

int32_t construct_duals(int64_t point_count, const double* points, int64_t tet_count,
                        const int32_t* tets, const double* domain, int64_t max_facets,
                        int64_t work_limit, int32_t* facets, int32_t* cells,
                        double* endpoints, int32_t* kinds, int32_t* item_status,
                        int64_t* facet_count, int64_t* counters) {
  if (point_count < 4 || point_count > kMaxMeshPoints || tet_count < 1 ||
      tet_count > std::numeric_limits<int32_t>::max() || max_facets < 1 ||
      work_limit < 0 || points == nullptr || tets == nullptr || domain == nullptr ||
      facets == nullptr || cells == nullptr || endpoints == nullptr || kinds == nullptr ||
      item_status == nullptr || facet_count == nullptr || counters == nullptr ||
      !addressable(point_count, 3, sizeof(double)) ||
      !addressable(tet_count, 4, sizeof(int32_t)) ||
      !addressable(tet_count, 4, sizeof(HalfFacet)) ||
      !addressable(max_facets, 6, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, 3, nullptr);
  if (valid != PHX_MC_OK) {
    return valid;
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(domain[axis]) || !std::isfinite(domain[axis + 3]) ||
        !(domain[axis] < domain[axis + 3])) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  // At least 2*T primal facets, at most 4*T. Refuse before work allocation.
  if (tet_count > work_limit / 5 || tet_count > max_facets / 2) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  std::vector<std::array<double, 3>> centers(static_cast<std::size_t>(tet_count));
  std::vector<bool> center_valid(static_cast<std::size_t>(tet_count), false);
  std::vector<HalfFacet> half_facets;
  half_facets.reserve(static_cast<std::size_t>(4 * tet_count));
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + cell * 4;
    for (int vertex = 0; vertex < 4; ++vertex) {
      if (ids[vertex] < 0 || ids[vertex] >= point_count) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    const double* p[4] = {points + 3 * ids[0], points + 3 * ids[1],
                          points + 3 * ids[2], points + 3 * ids[3]};
    if (orient3d(p[0], p[1], p[2], p[3]) <= 0) {
      return PHX_MC_DEGENERATE_INPUT;
    }
    center_valid[static_cast<std::size_t>(cell)] = geometry::tetrahedron_circumcenter(
        p[0], p[1], p[2], p[3], centers[static_cast<std::size_t>(cell)].data());
    for (int opposite = 0; opposite < 4; ++opposite) {
      HalfFacet facet{{}, static_cast<int32_t>(cell), ids[opposite]};
      int slot = 0;
      for (int vertex = 0; vertex < 4; ++vertex) {
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
  std::vector<DualFacet> duals;
  duals.reserve(static_cast<std::size_t>(std::min(max_facets, 4 * tet_count)));
  std::array<int64_t, 4> counts{5 * tet_count, 0, 0, 0};
  for (std::size_t first = 0; first < half_facets.size();) {
    std::size_t last = first + 1;
    while (last < half_facets.size() &&
           half_facets[last].vertices == half_facets[first].vertices) {
      ++last;
    }
    if (last - first > 2) {
      return PHX_MC_INVALID_INPUT;
    }
    if (static_cast<int64_t>(duals.size()) >= max_facets || counts[0] >= work_limit) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    ++counts[0];
    const HalfFacet& a = half_facets[first];
    const HalfFacet* b = last - first == 2 ? &half_facets[first + 1] : nullptr;
    const double* p = points + 3 * a.vertices[0];
    const double* q = points + 3 * a.vertices[1];
    const double* r = points + 3 * a.vertices[2];
    if (b != nullptr) {
      if (orient3d(p, q, r, points + 3 * a.opposite) ==
          orient3d(p, q, r, points + 3 * b->opposite)) {
        return PHX_MC_INVALID_INPUT;
      }
      const int32_t* ids = tets + 4 * a.cell;
      if (insphere(points + 3 * ids[0], points + 3 * ids[1], points + 3 * ids[2],
                   points + 3 * ids[3], points + 3 * b->opposite) > 0) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    DualFacet dual{a.vertices, {a.cell, b == nullptr ? -1 : b->cell}};
    dual.kind = b == nullptr ? 1 : 0;
    ++counts[b == nullptr ? 2 : 1];
    if (!center_valid[a.cell] || (b != nullptr && !center_valid[b->cell])) {
      dual.status = PHX_MC_DEGENERATE_INPUT;
      ++counts[3];
    } else {
      double direction[3];
      if (b != nullptr) {
        geometry::difference(centers[b->cell].data(), centers[a.cell].data(), direction);
      } else {
        double u[3], v[3];
        geometry::difference(q, p, u);
        geometry::difference(r, p, v);
        geometry::cross(u, v, direction);
        if (orient3d(p, q, r, points + 3 * a.opposite) > 0) {
          for (double& value : direction) {
            value = -value;
          }
        }
      }
      if (!geometry::finite3(direction)) {
        dual.status = PHX_MC_DEGENERATE_INPUT;
        ++counts[3];
      } else if (!clip_dual(centers[a.cell].data(), direction, domain,
                            b == nullptr, dual.endpoints)) {
        dual.kind = 2;
      }
    }
    duals.push_back(dual);
    first = last;
  }
  // Commit outputs only after every exact incidence and budget check passed.
  for (std::size_t row = 0; row < duals.size(); ++row) {
    const DualFacet& dual = duals[row];
    std::copy(dual.vertices.begin(), dual.vertices.end(), facets + 3 * row);
    std::copy(dual.cells.begin(), dual.cells.end(), cells + 2 * row);
    std::copy(dual.endpoints.begin(), dual.endpoints.end(), endpoints + 6 * row);
    kinds[row] = dual.kind;
    item_status[row] = dual.status;
  }
  *facet_count = static_cast<int64_t>(duals.size());
  std::copy(counts.begin(), counts.end(), counters);
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
