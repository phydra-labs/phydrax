// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Explicit sampled unsigned-distance wrapping of a triangle soup. Source
// triangles have a bounded barycentric covering cloud. Distance to that cloud
// is sampled on a conforming Freudenthal grid; no inside inference or repair.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "bounded_memory.hpp"
#include "capi_guard.hpp"
#include "counter_records.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "topology_keys.hpp"

namespace {
using V = std::array<double, 3>;
using F = std::array<int64_t, 3>;
using phx::mc::NativeMap;
using phx::mc::NativeVector;
using phx::mc::geometry::cross;
using phx::mc::geometry::difference;
using phx::mc::geometry::dot;
using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;
using phx::mc::geometry::point_in_domain;
using C = std::array<int64_t, 4>;
struct EnvelopeData : phx::mc::NativeAllocatedObject {
  NativeVector<V> surface_points, volume_points;
  NativeVector<F> triangles;
  NativeVector<C> cells;
};
V add(const V& a, const V& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V mul(const V& a, double t) { return {a[0] * t, a[1] * t, a[2] * t}; }
double length(const V& a) { return std::hypot(a[0], a[1], a[2]); }
V point(const double* p, int64_t i) { return {p[3 * i], p[3 * i + 1], p[3 * i + 2]}; }
double upper_sum(double a, double b) { return next_up(a + b); }
struct Work {
  int64_t used = 0, limit;
  bool spend() {
    if (used >= limit) return false;
    phx::mc::native_execution_charge(1);
    ++used;
    return true;
  }
};
bool distance(const V& p, const NativeVector<V>& cloud, Work& work, double& result) {
  // One represented nearest-cloud distance query; pair comparisons below are
  // primitive work, not additional source-geometry evaluations.
  phx::mc::native_execution_charge(0, 1);
  result = std::numeric_limits<double>::infinity();
  for (const V& q : cloud) {
    if (!work.spend()) return false;
    result = std::min(result, length(difference(p, q)));
  }
  return true;
}

struct Carrier {
  NativeVector<V> centers;
  NativeVector<C> cells;
  NativeVector<int8_t> grid_used;
  int64_t grid_vertices = 0;
  Carrier(int64_t total) : grid_used(static_cast<size_t>(total), 0) {}
};

template <class Location, class Root, class PointAt>
int32_t collect_cut_boundary(const C& tet, const NativeVector<double>& field, double offset,
                             const NativeVector<F>& skin, size_t skin_begin, Location& location,
                             Root& root, PointAt& point_at, std::array<F, 10>& boundary,
                             size_t& boundary_count) {
  for (size_t i = skin_begin; i < skin.size(); ++i)
    boundary[boundary_count++] = {-skin[i][0] - 1, -skin[i][1] - 1, -skin[i][2] - 1};
  for (int opposite = 0; opposite < 4; ++opposite) {
    std::array<int64_t, 3> face;
    int slot = 0;
    for (int i = 0; i < 4; ++i)
      if (i != opposite) face[slot++] = tet[i];
    if (field[face[0]] == offset && field[face[1]] == offset && field[face[2]] == offset) continue;
    const V a = location(face[0]), b = location(face[1]), c = location(face[2]),
            d = location(tet[opposite]);
    if (!point_in_domain(a) || !point_in_domain(b) || !point_in_domain(c) || !point_in_domain(d))
      return PHX_MC_RANGE_ERROR;
    if (phx::mc::orient3d(a.data(), b.data(), c.data(), d.data()) > 0) std::swap(face[1], face[2]);
    std::array<int64_t, 4> polygon;
    size_t n = 0;
    auto append = [&](int64_t id) {
      if (n == 0 || polygon[n - 1] != id) polygon[n++] = id;
    };
    for (int i = 0; i < 3; ++i) {
      const int64_t v = face[i], w = face[(i + 1) % 3];
      if (field[v] < offset)
        append(v);
      else if (field[v] == offset) {
        const int64_t r = root(v, v);
        if (r < 0) return PHX_MC_CAPACITY_EXCEEDED;
        append(-r - 1);
      }
      if ((field[v] < offset && field[w] > offset) || (field[v] > offset && field[w] < offset)) {
        const int64_t r = root(v, w);
        if (r < 0) return PHX_MC_CAPACITY_EXCEEDED;
        append(-r - 1);
      }
    }
    if (n > 1 && polygon[0] == polygon[n - 1]) --n;
    if (n < 3) continue;
    auto less = [&](int64_t u, int64_t v) {
      if ((u < 0) != (v < 0)) return u < 0;
      return u < 0 ? -u - 1 < -v - 1 : u < v;
    };
    size_t first = 0;
    for (size_t i = 1; i < n; ++i)
      if (less(polygon[i], polygon[first])) first = i;
    for (size_t i = 1; i + 1 < n; ++i) {
      const F triangle = {polygon[first], polygon[(first + i) % n], polygon[(first + i + 1) % n]};
      const V p = point_at(triangle[0]), q = point_at(triangle[1]), r = point_at(triangle[2]);
      if (!point_in_domain(p) || !point_in_domain(q) || !point_in_domain(r))
        return PHX_MC_RANGE_ERROR;
      if (!phx::mc::geometry::collinear(p, q, r, 0)) {
        if (boundary_count == boundary.size()) return PHX_MC_INTERNAL_ERROR;
        boundary[boundary_count++] = triangle;
      }
    }
  }
  return PHX_MC_OK;
}

template <class PointAt>
int32_t create_cut_center(const std::array<F, 10>& boundary, size_t boundary_count,
                          PointAt& point_at, int64_t total, const NativeVector<V>& roots,
                          Carrier& carrier, int64_t vertex_capacity, int64_t& center) {
  std::array<int64_t, 6> vertices;
  size_t count = 0;
  for (size_t i = 0; i < boundary_count; ++i)
    for (int64_t id : boundary[i]) {
      bool present = false;
      for (size_t j = 0; j < count; ++j) present |= vertices[j] == id;
      if (!present) {
        if (count == vertices.size()) return PHX_MC_INTERNAL_ERROR;
        vertices[count++] = id;
      }
    }
  if (count < 4) return PHX_MC_DEGENERATE_INPUT;
  const V anchor = point_at(vertices[0]);
  V displacement = {0, 0, 0};
  for (size_t i = 1; i < count; ++i)
    displacement = add(displacement, mul(difference(point_at(vertices[i]), anchor), 1.0 / count));
  center = total + static_cast<int64_t>(carrier.centers.size());
  if (static_cast<int64_t>(roots.size() + carrier.centers.size()) + carrier.grid_vertices >=
      vertex_capacity)
    return PHX_MC_CAPACITY_EXCEEDED;
  carrier.centers.push_back(add(anchor, displacement));
  return PHX_MC_OK;
}

// Negative ids name surface roots; nonnegative ids below total name grid
// nodes; ids above total name this carrier's cell centers.
// This symbolic numbering keeps root-prefix identity independent of when an
// interior vertex is first encountered during extraction.
template <class Location, class Root>
int32_t cut_cell(const C& tet, const NativeVector<double>& field, double offset, int64_t total,
                 const NativeVector<V>& roots, const NativeVector<F>& skin, size_t skin_begin,
                 Location& location, Root& root, Carrier& carrier, int64_t vertex_capacity,
                 int64_t cell_capacity, Work& work) {
  auto point_at = [&](int64_t id) -> V {
    if (id < 0) return roots[static_cast<size_t>(-id - 1)];
    if (id < total) return location(id);
    return carrier.centers[static_cast<size_t>(id - total)];
  };
  auto put = [&](C cell) -> int32_t {
    if (!work.spend() || static_cast<int64_t>(carrier.cells.size()) >= cell_capacity)
      return PHX_MC_CAPACITY_EXCEEDED;
    const V a = point_at(cell[0]), b = point_at(cell[1]), c = point_at(cell[2]),
            d = point_at(cell[3]);
    if (!point_in_domain(a) || !point_in_domain(b) || !point_in_domain(c) || !point_in_domain(d))
      return PHX_MC_RANGE_ERROR;
    if (phx::mc::orient3d(a.data(), b.data(), c.data(), d.data()) <= 0)
      return PHX_MC_CONSTRAINT_INTERSECTION;
    for (int64_t id : cell)
      if (id >= 0 && id < total && !carrier.grid_used[id]) {
        carrier.grid_used[id] = 1;
        ++carrier.grid_vertices;
      }
    if (static_cast<int64_t>(roots.size() + carrier.centers.size()) + carrier.grid_vertices >
        vertex_capacity)
      return PHX_MC_CAPACITY_EXCEEDED;
    carrier.cells.push_back(cell);
    return PHX_MC_OK;
  };
  int inside = 0;
  for (int64_t id : tet) inside += field[id] < offset;
  if (inside == 0) return PHX_MC_OK;
  if (inside == 4) {
    C cell = tet;
    const V a = location(cell[0]), b = location(cell[1]), c = location(cell[2]),
            d = location(cell[3]);
    if (!point_in_domain(a) || !point_in_domain(b) || !point_in_domain(c) || !point_in_domain(d))
      return PHX_MC_RANGE_ERROR;
    if (phx::mc::orient3d(a.data(), b.data(), c.data(), d.data()) < 0) std::swap(cell[1], cell[2]);
    return put(cell);
  }
  std::array<F, 10> boundary;
  size_t boundary_count = 0;
  int32_t status = collect_cut_boundary(tet, field, offset, skin, skin_begin, location, root,
                                        point_at, boundary, boundary_count);
  if (status != PHX_MC_OK) return status;
  int64_t center;
  status = create_cut_center(boundary, boundary_count, point_at, total, roots, carrier,
                             vertex_capacity, center);
  if (status != PHX_MC_OK) return status;
  for (size_t i = 0; i < boundary_count; ++i) {
    const int32_t status = put({center, boundary[i][0], boundary[i][1], boundary[i][2]});
    if (status != PHX_MC_OK) return status;
  }
  return PHX_MC_OK;
}
struct EnvelopeGrid {
  V lo, hi;
  std::array<int64_t, 3> shape;
  int64_t total = 1;
  double roundoff, diameter, arithmetic_error, error, covering_radius = 0;
  double spacing;
  int64_t index(int64_t x, int64_t y, int64_t z) const { return (x * shape[1] + y) * shape[2] + z; }
  V operator()(int64_t i) const {
    const int64_t z = i % shape[2];
    i /= shape[2];
    const int64_t y = i % shape[1], x = i / shape[1];
    return {lo[0] + x * spacing, lo[1] + y * spacing, lo[2] + z * spacing};
  }
};

int32_t validate_envelope_request(const phx_mc_surface_envelope_create_request& request) {
  if (!request.result) return PHX_MC_INVALID_ARGUMENT;
  *request.result = nullptr;
  if (!request.vertices || !request.faces || !request.counts || !request.bounds ||
      request.vertex_count < 1 || request.face_count < 1 || !std::isfinite(request.offset) ||
      !std::isfinite(request.spacing) || !std::isfinite(request.certificate_spacing) ||
      request.offset <= 0 || request.spacing <= 0 || request.certificate_spacing <= 0 ||
      request.sample_limit < 1 || request.work_limit < 1 || request.surface_vertex_capacity < 1 ||
      request.surface_triangle_capacity < 1 ||
      !phx::mc::addressable(request.vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.face_count, 3, sizeof(int64_t)) ||
      !phx::mc::addressable(request.surface_vertex_capacity, 3, sizeof(double)) ||
      !phx::mc::addressable(request.surface_triangle_capacity, 3, sizeof(int64_t)) ||
      request.volume_vertex_capacity < 1 || request.tetrahedron_capacity < 1 ||
      !phx::mc::addressable(request.volume_vertex_capacity, 3, sizeof(double)) ||
      !phx::mc::addressable(request.tetrahedron_capacity, 4, sizeof(int64_t)))
    return PHX_MC_INVALID_ARGUMENT;
  return PHX_MC_OK;
}

int32_t prepare_envelope_grid(const phx_mc_surface_envelope_create_request& request,
                              EnvelopeGrid& grid) {
  grid.lo = point(request.vertices, 0);
  grid.hi = grid.lo;
  for (int64_t i = 0; i < request.vertex_count; ++i)
    for (int d = 0; d < 3; ++d) {
      const double x = request.vertices[3 * i + d];
      if (!std::isfinite(x)) return PHX_MC_NONFINITE_INPUT;
      grid.lo[d] = std::min(grid.lo[d], x);
      grid.hi[d] = std::max(grid.hi[d], x);
    }
  for (int64_t i = 0; i < 3 * request.face_count; ++i)
    if (request.faces[i] < 0 || request.faces[i] >= request.vertex_count)
      return PHX_MC_INVALID_INPUT;
  double scale = request.offset + request.spacing;
  for (int d = 0; d < 3; ++d) scale = std::max({scale, std::abs(grid.lo[d]), std::abs(grid.hi[d])});
  // A nonzero absolute floor also covers underflow-rounding when the source
  // coordinates and the declared sampling scales are finite subnormals.
  grid.roundoff = next_up(std::max(256 * std::numeric_limits<double>::epsilon() * scale,
                                   256 * std::numeric_limits<double>::denorm_min()));
  if (!std::isfinite(grid.roundoff)) return PHX_MC_RANGE_ERROR;
  grid.diameter = next_up(next_up(std::sqrt(3.0)) * request.spacing);
  grid.arithmetic_error = next_up(4 * grid.roundoff);
  grid.error =
      upper_sum(upper_sum(grid.diameter, request.certificate_spacing), grid.arithmetic_error);
  if (!(request.offset > grid.error)) return PHX_MC_RANGE_ERROR;
  for (int d = 0; d < 3; ++d) {
    grid.lo[d] -= request.offset + 2 * request.spacing;
    grid.hi[d] += request.offset + 2 * request.spacing;
    const double n = std::ceil((grid.hi[d] - grid.lo[d]) / request.spacing) + 1;
    if (!std::isfinite(n) || n < 2 || n > static_cast<double>(request.sample_limit) ||
        n >= static_cast<double>(std::numeric_limits<int64_t>::max()))
      return PHX_MC_CAPACITY_EXCEEDED;
    grid.shape[d] = static_cast<int64_t>(n);
    if (grid.total > request.sample_limit / grid.shape[d]) return PHX_MC_CAPACITY_EXCEEDED;
    grid.total *= grid.shape[d];
  }
  if (!phx::mc::addressable(grid.total, 1, sizeof(double))) return PHX_MC_CAPACITY_EXCEEDED;
  return PHX_MC_OK;
}

int32_t build_covering_cloud(const phx_mc_surface_envelope_create_request& request,
                             EnvelopeGrid& grid, NativeVector<V>& cloud, Work& work,
                             phx::mc::SurfaceEnvelopeCounts& state) {
  for (int64_t f = 0; f < request.face_count; ++f) {
    const V a = point(request.vertices, request.faces[3 * f]),
            b = point(request.vertices, request.faces[3 * f + 1]),
            c = point(request.vertices, request.faces[3 * f + 2]);
    const double diameter =
        upper_sum(next_up(std::max({length(difference(a, b)), length(difference(a, c)),
                                    length(difference(b, c))})),
                  grid.roundoff);
    const double needed = std::max(1.0, std::ceil(diameter / request.certificate_spacing));
    if (!std::isfinite(needed) || needed >= static_cast<double>(request.sample_limit) ||
        needed >= static_cast<double>(std::numeric_limits<int64_t>::max()))
      return PHX_MC_CAPACITY_EXCEEDED;
    const int64_t n = static_cast<int64_t>(needed);
    const long double samples =
        (static_cast<long double>(n) + 1) * (static_cast<long double>(n) + 2) / 2;
    if (samples > request.sample_limit - grid.total - state.source_samples)
      return PHX_MC_CAPACITY_EXCEEDED;
    grid.covering_radius =
        std::max(grid.covering_radius, upper_sum(next_up(diameter / n), grid.roundoff));
    for (int64_t i = 0; i <= n; ++i)
      for (int64_t j = 0; j <= n - i; ++j) {
        if (!work.spend()) {
          state.work_units = work.used;
          return PHX_MC_CAPACITY_EXCEEDED;
        }
        cloud.push_back(add(a, add(mul(difference(b, a), static_cast<double>(i) / n),
                                   mul(difference(c, a), static_cast<double>(j) / n))));
        ++state.source_samples;
      }
  }
  grid.error =
      next_up(upper_sum(upper_sum(grid.diameter, grid.covering_radius), grid.arithmetic_error));
  if (!std::isfinite(grid.error) || !(request.offset > grid.error)) return PHX_MC_RANGE_ERROR;
  return PHX_MC_OK;
}

int32_t sample_envelope_field(const EnvelopeGrid& grid, const NativeVector<V>& cloud,
                              NativeVector<double>& field, Work& work,
                              phx::mc::SurfaceEnvelopeCounts& state) {
  for (int64_t i = 0; i < grid.total; ++i) {
    if (!distance(grid(i), cloud, work, field[i])) {
      state.work_units = work.used;
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (!std::isfinite(field[i])) return PHX_MC_RANGE_ERROR;
  }
  state.grid_samples = grid.total;
  return PHX_MC_OK;
}

int32_t extract_envelope_carrier(const phx_mc_surface_envelope_create_request& request,
                                 EnvelopeGrid& grid, const NativeVector<double>& field,
                                 EnvelopeData& data, Carrier& carrier,
                                 NativeMap<std::array<int64_t, 2>, int64_t>& roots, Work& work,
                                 phx::mc::SurfaceEnvelopeCounts& state) {
  auto& result = data.surface_points;
  auto& triangles = data.triangles;
  auto& location = grid;
  auto root = [&](int64_t a, int64_t b) -> int64_t {
    if (field[a] == request.offset)
      b = a;
    else if (field[b] == request.offset)
      a = b;
    const auto key = phx::mc::topology::undirected_edge_vertices(a, b);
    const auto found = roots.find(key);
    if (found != roots.end()) return found->second;
    if (static_cast<int64_t>(result.size()) >= request.surface_vertex_capacity ||
        static_cast<int64_t>(result.size() + carrier.centers.size()) + carrier.grid_vertices >=
            request.volume_vertex_capacity)
      return -1;
    const double t = a == b ? 0.0 : (request.offset - field[a]) / (field[b] - field[a]);
    const int64_t id = static_cast<int64_t>(result.size());
    result.push_back(add(location(a), mul(difference(location(b), location(a)), t)));
    roots.emplace(key, id);
    return id;
  };
  auto emit = [&](int64_t a, int64_t b, int64_t c, const V& outward) -> bool {
    if (a < 0 || b < 0 || c < 0) return false;
    if (a == b || b == c || a == c) return true;
    const V normal = cross(difference(result[b], result[a]), difference(result[c], result[a]));
    if (dot(normal, normal) == 0) return true;
    if (static_cast<int64_t>(triangles.size()) >= request.surface_triangle_capacity) return false;
    if (dot(normal, outward) < 0) std::swap(b, c);
    triangles.push_back({a, b, c});
    return true;
  };
  constexpr int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                      {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int64_t x = 0; x < grid.shape[0] - 1; ++x)
    for (int64_t y = 0; y < grid.shape[1] - 1; ++y)
      for (int64_t z = 0; z < grid.shape[2] - 1; ++z) {
        for (const auto& permutation : permutations) {
          if (!work.spend()) {
            state.work_units = work.used;
            return PHX_MC_CAPACITY_EXCEEDED;
          }
          std::array<int64_t, 3> p = {x, y, z};
          std::array<int64_t, 4> tet;
          tet[0] = grid.index(p[0], p[1], p[2]);
          for (int k = 0; k < 3; ++k) {
            ++p[permutation[k]];
            tet[k + 1] = grid.index(p[0], p[1], p[2]);
          }
          std::array<int64_t, 4> inside{}, outside{};
          int ni = 0, no = 0;
          for (int64_t id : tet) {
            if (field[id] < request.offset)
              inside[ni++] = id;
            else
              outside[no++] = id;
          }
          if (ni == 0) continue;
          const size_t skin_begin = triangles.size();
          if (ni == 4) {
            const int32_t status = cut_cell(
                tet, field, request.offset, grid.total, result, triangles, skin_begin, location,
                root, carrier, request.volume_vertex_capacity, request.tetrahedron_capacity, work);
            if (status != PHX_MC_OK) {
              state.work_units = work.used;
              return status;
            }
            continue;
          }
          const V outward = difference(location(outside[0]), location(inside[0]));
          bool ok;
          if (ni == 1)
            ok = emit(root(inside[0], outside[0]), root(inside[0], outside[1]),
                      root(inside[0], outside[2]), outward);
          else if (ni == 3)
            ok = emit(root(outside[0], inside[0]), root(outside[0], inside[1]),
                      root(outside[0], inside[2]), outward);
          else {
            const int64_t a = root(inside[0], outside[0]), b = root(inside[0], outside[1]),
                          c = root(inside[1], outside[0]), d = root(inside[1], outside[1]);
            ok = emit(a, b, d, outward) && emit(a, d, c, outward);
          }
          if (!ok) {
            state.work_units = work.used;
            return PHX_MC_CAPACITY_EXCEEDED;
          }
          const int32_t status = cut_cell(
              tet, field, request.offset, grid.total, result, triangles, skin_begin, location, root,
              carrier, request.volume_vertex_capacity, request.tetrahedron_capacity, work);
          if (status != PHX_MC_OK) {
            state.work_units = work.used;
            return status;
          }
        }
      }
  return PHX_MC_OK;
}

int32_t assemble_envelope_carrier(const phx_mc_surface_envelope_create_request& request,
                                  EnvelopeGrid& grid, EnvelopeData* data, Carrier& carrier) {
  auto& result = data->surface_points;
  auto& triangles = data->triangles;
  auto& location = grid;
  if (triangles.empty()) return PHX_MC_DEGENERATE_INPUT;
  // Preserve source boundary roots as the exact prefix of the volume
  // carrier. Root-only zero-dimensional internal contacts remain interior
  // carrier request.vertices; they never enter the surface deviation certificate.
  NativeVector<int64_t> remap(result.size(), -1), root_volume(result.size(), -1);
  NativeVector<int8_t> root_used(result.size(), 0);
  for (const F& t : triangles)
    for (int64_t id : t) remap[id] = 0;
  for (const C& cell : carrier.cells)
    for (int64_t id : cell)
      if (id < 0) root_used[-id - 1] = 1;
  int64_t kept = 0;
  for (size_t i = 0; i < result.size(); ++i)
    if (remap[i] >= 0) {
      root_volume[i] = remap[i] = kept++;
      data->volume_points.push_back(result[i]);
    }
  for (size_t i = 0; i < result.size(); ++i)
    if (remap[i] < 0 && root_used[i]) {
      root_volume[i] = static_cast<int64_t>(data->volume_points.size());
      data->volume_points.push_back(result[i]);
    }
  NativeVector<int64_t> grid_volume(static_cast<size_t>(grid.total), -1);
  for (int64_t i = 0; i < grid.total; ++i)
    if (carrier.grid_used[i]) {
      grid_volume[i] = static_cast<int64_t>(data->volume_points.size());
      data->volume_points.push_back(location(i));
    }
  const int64_t center_begin = static_cast<int64_t>(data->volume_points.size());
  data->volume_points.insert(data->volume_points.end(), carrier.centers.begin(),
                             carrier.centers.end());
  data->cells = std::move(carrier.cells);
  for (C& cell : data->cells)
    for (int64_t& id : cell)
      id = id < 0            ? root_volume[-id - 1]
           : id < grid.total ? grid_volume[id]
                             : center_begin + (id - grid.total);
  for (size_t i = 0; i < result.size(); ++i)
    if (remap[i] >= 0) result[remap[i]] = result[i];
  result.resize(static_cast<size_t>(kept));
  for (F& t : triangles)
    for (int64_t& id : t) id = remap[id];
  if (static_cast<int64_t>(data->volume_points.size()) > request.volume_vertex_capacity)
    return PHX_MC_CAPACITY_EXCEEDED;
  return PHX_MC_OK;
}

int32_t certify_envelope(const phx_mc_surface_envelope_create_request& request,
                         const EnvelopeGrid& grid, EnvelopeData* data, const NativeVector<V>& cloud,
                         Work& work, phx::mc::SurfaceEnvelopeCounts& state) {
  auto& result = data->surface_points;
  auto& triangles = data->triangles;
  auto* bounds = request.bounds;
  double source_upper = 0;
  for (const V& p : cloud) {
    double nearest;
    if (!distance(p, result, work, nearest)) {
      state.work_units = work.used;
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    source_upper = std::max(
        source_upper,
        upper_sum(upper_sum(next_up(nearest), grid.covering_radius), grid.arithmetic_error));
  }
  state.surface_vertices = static_cast<int64_t>(result.size());
  state.surface_triangles = static_cast<int64_t>(triangles.size());
  state.work_units = work.used;
  // Triangle covering + 1-Lipschitz cloud distance + P1 interpolation:
  // |d_source-I_h d_cloud| <= covering radius + grid-cell diameter.
  // I_h d_cloud=offset on every output triangle and <=error on the source.
  bounds[0] = next_up(source_upper);
  bounds[1] = upper_sum(request.offset, grid.error);
  bounds[2] = next_down(request.offset - grid.error);
  bounds[3] = grid.error;
  if (!std::isfinite(bounds[0]) || !std::isfinite(bounds[1]) || !std::isfinite(bounds[3]) ||
      !(bounds[2] > 0))
    return PHX_MC_RANGE_ERROR;
  state.volume_vertices = static_cast<int64_t>(data->volume_points.size());
  state.tetrahedra = static_cast<int64_t>(data->cells.size());
  return PHX_MC_OK;
}
}  // namespace

extern "C" PHX_MC_API int32_t
phx_mc_surface_envelope_create(const phx_mc_surface_envelope_create_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    int32_t status = validate_envelope_request(*request);
    if (status != PHX_MC_OK) return status;
    phx::mc::SurfaceEnvelopeCounts state;
    phx::mc::CounterPublisher publisher{state, request->counts};
    std::fill(request->bounds, request->bounds + 4, 0.0);
    auto data = phx::mc::make_native_unique<EnvelopeData>();
    EnvelopeGrid grid;
    grid.spacing = request->spacing;
    status = prepare_envelope_grid(*request, grid);
    if (status != PHX_MC_OK) return status;
    Work work{0, request->work_limit};
    NativeVector<V> cloud;
    status = build_covering_cloud(*request, grid, cloud, work, state);
    if (status != PHX_MC_OK) return status;
    NativeVector<double> field(static_cast<size_t>(grid.total));
    status = sample_envelope_field(grid, cloud, field, work, state);
    if (status != PHX_MC_OK) return status;
    Carrier carrier(grid.total);
    NativeMap<std::array<int64_t, 2>, int64_t> roots;
    status = extract_envelope_carrier(*request, grid, field, *data, carrier, roots, work, state);
    if (status != PHX_MC_OK) return status;
    status = assemble_envelope_carrier(*request, grid, data.get(), carrier);
    if (status != PHX_MC_OK) return status;
    status = certify_envelope(*request, grid, data.get(), cloud, work, state);
    if (status != PHX_MC_OK) return status;
    *request->result = data.release();
    return PHX_MC_OK;
  });
}

extern "C" PHX_MC_API int32_t phx_mc_surface_envelope_arrays(void* handle, double* surface_vertices,
                                                             int64_t* surface_triangles,
                                                             double* volume_vertices,
                                                             int64_t* tetrahedra) {
  if (!handle || !surface_vertices || !surface_triangles || !volume_vertices || !tetrahedra)
    return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const EnvelopeData*>(handle);
  for (size_t i = 0; i < data.surface_points.size(); ++i)
    std::copy(data.surface_points[i].begin(), data.surface_points[i].end(),
              surface_vertices + 3 * i);
  for (size_t i = 0; i < data.triangles.size(); ++i)
    std::copy(data.triangles[i].begin(), data.triangles[i].end(), surface_triangles + 3 * i);
  for (size_t i = 0; i < data.volume_points.size(); ++i)
    std::copy(data.volume_points[i].begin(), data.volume_points[i].end(), volume_vertices + 3 * i);
  for (size_t i = 0; i < data.cells.size(); ++i)
    std::copy(data.cells[i].begin(), data.cells[i].end(), tetrahedra + 4 * i);
  return PHX_MC_OK;
}
extern "C" PHX_MC_API void phx_mc_surface_envelope_free(void* handle) {
  phx::mc::destroy_native_object(static_cast<EnvelopeData*>(handle));
}
