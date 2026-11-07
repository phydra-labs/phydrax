// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Explicit sampled unsigned-distance wrapping of a triangle soup. Source
// triangles have a finite barycentric covering cloud. Distance to that cloud
// is sampled on a conforming Freudenthal grid; no inside inference or repair.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <utility>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "surface_remesh.hpp"
#include "intersections.hpp"
#include "topology_keys.hpp"

namespace {
using V = std::array<double, 3>;
using F = std::array<int64_t, 3>;
using std::map;
using std::vector;
using phx::mc::geometry::cross;
using phx::mc::geometry::difference;
using phx::mc::geometry::dot;
using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;
using phx::mc::geometry::point_in_domain;
using C = std::array<int64_t, 4>;
struct EnvelopeData {
  std::vector<V> surface_points, volume_points;
  std::vector<F> triangles;
  std::vector<C> cells;
  // Certification state retained for band-constrained remeshing: the source
  // covering cloud, its covering radius, the arithmetic error, the certified
  // band and the sampling spacing.
  std::vector<V> cloud;
  double covering_radius = 0, arithmetic_error = 0, roundoff = 0, spacing = 0, offset = 0;
  std::array<double, 4> bounds{};
};
V add(const V& a, const V& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V mul(const V& a, double t) { return {a[0] * t, a[1] * t, a[2] * t}; }
double length(const V& a) { return std::hypot(a[0], a[1], a[2]); }
V point(const double* p, int64_t i) { return {p[3 * i], p[3 * i + 1], p[3 * i + 2]}; }
double upper_sum(double a, double b) { return next_up(a + b); }
double distance(const V& p, const std::vector<V>& cloud) {
  double result = std::numeric_limits<double>::infinity();
  for (const V& q : cloud) result = std::min(result, length(difference(p, q)));
  return result;
}

struct Carrier {
  std::vector<C> cells;
  std::vector<int8_t> grid_used;
  Carrier(int64_t total) : grid_used(static_cast<size_t>(total), 0) {}
};

template <class Location, class Root, class PointAt>
int32_t collect_cut_boundary(const C& tet, const std::vector<double>& field, double offset,
                             const std::vector<F>& skin, size_t skin_begin, Location& location,
                             Root& root, PointAt& point_at, std::array<F, 10>& boundary,
                             size_t& boundary_count) {
  for (size_t i = skin_begin; i < skin.size(); ++i)
    boundary[boundary_count++] = {-skin[i][0] - 1, -skin[i][1] - 1, -skin[i][2] - 1};
  for (int opposite = 0; opposite < 4; ++opposite) {
    std::array<int64_t, 3> face;
    int slot = 0;
    for (int i = 0; i < 4; ++i)
      if (i != opposite) face[slot++] = tet[i];
    // separate_field_from_offset leaves no grid value equal to the offset.
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
      if (field[v] < offset) append(v);
      if ((field[v] < offset && field[w] > offset) || (field[v] > offset && field[w] < offset)) {
        const int64_t r = root(v, w);
        if (r < 0) return PHX_MC_INTERNAL_ERROR;
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
template <class Location, class Root>
int32_t cut_cell(const C& tet, const std::vector<double>& field, double offset,
                 const std::vector<V>& roots, const std::vector<F>& skin, size_t skin_begin,
                 Location& location, Root& root, Carrier& carrier) {
  auto point_at = [&](int64_t id) -> V {
    return id < 0 ? roots[static_cast<size_t>(-id - 1)] : location(id);
  };
  auto put = [&](C cell) -> int32_t {
    const V a = point_at(cell[0]), b = point_at(cell[1]), c = point_at(cell[2]),
            d = point_at(cell[3]);
    if (!point_in_domain(a) || !point_in_domain(b) || !point_in_domain(c) || !point_in_domain(d))
      return PHX_MC_RANGE_ERROR;
    if (phx::mc::orient3d(a.data(), b.data(), c.data(), d.data()) <= 0)
      return PHX_MC_CONSTRAINT_INTERSECTION;
    for (int64_t id : cell)
      if (id >= 0 && !carrier.grid_used[id]) {
        carrier.grid_used[id] = 1;
      }
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
  if (boundary_count == 0) return PHX_MC_DEGENERATE_INPUT;
  const auto less = [](int64_t first, int64_t second) {
    if ((first < 0) != (second < 0)) return first < 0;
    return first < 0 ? -first - 1 < -second - 1 : first < second;
  };
  struct BoundaryEdge {
    int64_t first = 0;
    int64_t second = 0;
    int32_t uses = 0;
    int32_t direction = 0;
    size_t first_face = 0;
  };
  std::array<int64_t, 6> vertices{};
  size_t vertex_count = 0;
  std::array<BoundaryEdge, 30> edges{};
  std::array<std::array<uint8_t, 10>, 10> adjacent{};
  size_t edge_count = 0;
  int64_t apex = boundary[0][0];
  for (size_t face = 0; face < boundary_count; ++face) {
    for (int corner = 0; corner < 3; ++corner) {
      const int64_t vertex = boundary[face][corner];
      bool present = false;
      for (size_t index = 0; index < vertex_count; ++index)
        present = present || vertices[index] == vertex;
      if (!present) {
        if (vertex_count == vertices.size()) return PHX_MC_INTERNAL_ERROR;
        vertices[vertex_count++] = vertex;
      }
      if (less(vertex, apex)) apex = vertex;
      const int64_t next = boundary[face][(corner + 1) % 3];
      if (vertex == next) return PHX_MC_CONSTRAINT_INTERSECTION;
      const int64_t first = std::min(vertex, next);
      const int64_t second = std::max(vertex, next);
      size_t edge = 0;
      for (; edge < edge_count; ++edge)
        if (edges[edge].first == first && edges[edge].second == second) break;
      if (edge == edge_count) {
        if (edge_count == edges.size()) return PHX_MC_INTERNAL_ERROR;
        edges[edge_count] = {first, second, 0, 0, face};
        ++edge_count;
      }
      BoundaryEdge& use = edges[edge];
      ++use.uses;
      use.direction += vertex < next ? 1 : -1;
      if (use.uses > 2) return PHX_MC_CONSTRAINT_INTERSECTION;
      if (use.uses == 2) {
        adjacent[use.first_face][face] = 1;
        adjacent[face][use.first_face] = 1;
      }
    }
  }
  if (vertex_count < 4 ||
      static_cast<int64_t>(vertex_count) - static_cast<int64_t>(edge_count) +
              static_cast<int64_t>(boundary_count) !=
          2) {
    return PHX_MC_CONSTRAINT_INTERSECTION;
  }
  for (size_t edge = 0; edge < edge_count; ++edge)
    if (edges[edge].uses != 2 || edges[edge].direction != 0)
      return PHX_MC_CONSTRAINT_INTERSECTION;
  std::array<uint8_t, 10> connected{};
  connected[0] = 1;
  for (size_t step = 0; step < boundary_count; ++step) {
    bool changed = false;
    for (size_t first = 0; first < boundary_count; ++first) {
      if (!connected[first]) continue;
      for (size_t second = 0; second < boundary_count; ++second) {
        if (adjacent[first][second] && !connected[second]) {
          connected[second] = 1;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  for (size_t face = 0; face < boundary_count; ++face)
    if (!connected[face]) return PHX_MC_CONSTRAINT_INTERSECTION;
  for (size_t first = 0; first < boundary_count; ++first) {
    const V a0 = point_at(boundary[first][0]);
    const V a1 = point_at(boundary[first][1]);
    const V a2 = point_at(boundary[first][2]);
    const double* first_points[3] = {a0.data(), a1.data(), a2.data()};
    const int64_t first_ids[3] = {
        boundary[first][0], boundary[first][1], boundary[first][2]};
    for (size_t second = first + 1; second < boundary_count; ++second) {
      const V b0 = point_at(boundary[second][0]);
      const V b1 = point_at(boundary[second][1]);
      const V b2 = point_at(boundary[second][2]);
      const double* second_points[3] = {b0.data(), b1.data(), b2.data()};
      const int64_t second_ids[3] = {
          boundary[second][0], boundary[second][1], boundary[second][2]};
      phx::mc::Contact contact;
      const int32_t contact_status = phx::mc::intersect_triangles(
          first_points, second_points, first_ids, second_ids, contact);
      if (contact_status != PHX_MC_OK) return contact_status;
      if (contact.kind != PHX_MC_DISJOINT &&
          contact.kind != PHX_MC_SHARED_VERTEX &&
          contact.kind != PHX_MC_SHARED_EDGE) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
    }
  }
  const V apex_point = point_at(apex);
  for (size_t face = 0; face < boundary_count; ++face) {
    const F& triangle = boundary[face];
    if (triangle[0] == apex || triangle[1] == apex || triangle[2] == apex) continue;
    const V a = point_at(triangle[0]);
    const V b = point_at(triangle[1]);
    const V c = point_at(triangle[2]);
    if (phx::mc::orient3d(apex_point.data(), a.data(), b.data(), c.data()) <= 0)
      return PHX_MC_CONSTRAINT_INTERSECTION;
    const int32_t status = put({apex, triangle[0], triangle[1], triangle[2]});
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
  if (request.result == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *request.result = nullptr;
  if (request.vertices == nullptr || request.faces == nullptr || request.bounds == nullptr ||
      request.vertex_count < 1 || request.face_count < 1 || !std::isfinite(request.offset) ||
      !std::isfinite(request.spacing) || !std::isfinite(request.certificate_spacing) ||
      request.offset <= 0 || request.spacing <= 0 || request.certificate_spacing <= 0 ||
      !phx::mc::addressable(request.vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.face_count, 3, sizeof(int64_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
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
    if (!std::isfinite(n) || n < 2 ||
        n >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
      return PHX_MC_UNRESOLVED;
    }
    grid.shape[d] = static_cast<int64_t>(n);
    if (grid.total > std::numeric_limits<int64_t>::max() / grid.shape[d]) {
      return PHX_MC_UNRESOLVED;
    }
    grid.total *= grid.shape[d];
  }
  if (!phx::mc::addressable(grid.total, 1, sizeof(double))) return PHX_MC_UNRESOLVED;
  return PHX_MC_OK;
}

int32_t build_covering_cloud(const phx_mc_surface_envelope_create_request& request,
                             EnvelopeGrid& grid, std::vector<V>& cloud) {
  for (int64_t f = 0; f < request.face_count; ++f) {
    const V a = point(request.vertices, request.faces[3 * f]),
            b = point(request.vertices, request.faces[3 * f + 1]),
            c = point(request.vertices, request.faces[3 * f + 2]);
    const double diameter =
        upper_sum(next_up(std::max({length(difference(a, b)), length(difference(a, c)),
                                    length(difference(b, c))})),
                  grid.roundoff);
    const double needed = std::max(1.0, std::ceil(diameter / request.certificate_spacing));
    if (!std::isfinite(needed) ||
        needed >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
      return PHX_MC_UNRESOLVED;
    }
    const int64_t n = static_cast<int64_t>(needed);
    grid.covering_radius =
        std::max(grid.covering_radius, upper_sum(next_up(diameter / n), grid.roundoff));
    for (int64_t i = 0; i <= n; ++i)
      for (int64_t j = 0; j <= n - i; ++j) {
        cloud.push_back(add(a, add(mul(difference(b, a), static_cast<double>(i) / n),
                                   mul(difference(c, a), static_cast<double>(j) / n))));
      }
  }
  grid.error =
      next_up(upper_sum(upper_sum(grid.diameter, grid.covering_radius), grid.arithmetic_error));
  if (!std::isfinite(grid.error) || !(request.offset > grid.error)) return PHX_MC_RANGE_ERROR;
  return PHX_MC_OK;
}

int32_t sample_envelope_field(const EnvelopeGrid& grid, const std::vector<V>& cloud,
                              std::vector<double>& field) {
  for (int64_t i = 0; i < grid.total; ++i) {
    field[static_cast<std::size_t>(i)] = distance(grid(i), cloud);
    if (!std::isfinite(field[static_cast<std::size_t>(i)])) return PHX_MC_RANGE_ERROR;
  }
  return PHX_MC_OK;
}

// Moves every grid value within `band` of the offset to the band edge on its
// own side of the offset. The cloud distance is 1-Lipschitz, so on a grid
// edge of length L the values differ by at most L (up to roundoff) and every
// root then lies at a parameter in [band / (L + 2 band), 1 - band / (L + 2
// band)]: the roots on the edges at one grid vertex never collapse onto it
// (a value rounding next to the offset used to put ~10 roots within
// roundoff of that vertex), and no value equals the offset, so every cut is
// generic. Each grid vertex keeps its side, so the surface's combinatorics
// are unchanged; the values move by at most `shift`, which joins the
// certified interpolation error. The band is spacing / 8, capped at half the
// offset's slack over that error, so it never turns a certifiable request
// into a refusal.
int32_t separate_field_from_offset(const phx_mc_surface_envelope_create_request& request,
                                   EnvelopeGrid& grid, std::vector<double>& field) {
  const double offset = request.offset;
  const double slack = next_down(offset - grid.error);
  const double band = std::min(request.spacing / 8, slack / 2);
  const double below = next_down(offset - band);
  const double above = next_up(offset + band);
  const double shift = next_up(std::max(offset - below, above - offset));
  for (double& value : field) value = value < offset ? std::min(value, below) : std::max(value, above);
  grid.error = upper_sum(grid.error, shift);
  if (!std::isfinite(grid.error) || !(offset > grid.error)) return PHX_MC_RANGE_ERROR;
  return PHX_MC_OK;
}

int32_t extract_envelope_carrier(const phx_mc_surface_envelope_create_request& request,
                                 EnvelopeGrid& grid, const std::vector<double>& field,
                                 EnvelopeData& data, Carrier& carrier) {
  auto& result = data.surface_points;
  auto& triangles = data.triangles;
  auto& location = grid;
  // One root per crossing grid edge; no grid value equals the offset, so a
  // root is never a grid vertex and the cut of every grid tetrahedron is a
  // triangle or a quadrilateral.
  std::map<std::array<int64_t, 2>, int64_t> roots;
  auto root = [&](int64_t a, int64_t b) -> int64_t {
    const auto key = phx::mc::topology::undirected_edge_vertices(a, b);
    const auto found = roots.find(key);
    if (found != roots.end()) return found->second;
    const double t = (request.offset - field[a]) / (field[b] - field[a]);
    const int64_t id = static_cast<int64_t>(result.size());
    result.push_back(add(location(a), mul(difference(location(b), location(a)), t)));
    roots.emplace(key, id);
    return id;
  };
  auto emit = [&](int64_t a, int64_t b, int64_t c, const V& outward) -> bool {
    // Roots of distinct grid edges are distinct.
    if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c) return false;
    const V normal = cross(difference(result[b], result[a]), difference(result[c], result[a]));
    if (dot(normal.data(), normal.data()) == 0) return true;
    if (dot(normal.data(), outward.data()) < 0) std::swap(b, c);
    triangles.push_back({a, b, c});
    return true;
  };
  auto emit_quad = [&](const std::array<int64_t, 4>& polygon, const V& outward) {
    size_t first = 0;
    for (size_t vertex = 1; vertex < polygon.size(); ++vertex)
      if (polygon[vertex] < polygon[first]) first = vertex;
    for (size_t offset = 1; offset + 1 < polygon.size(); ++offset) {
      if (!emit(polygon[first], polygon[(first + offset) % polygon.size()],
                polygon[(first + offset + 1) % polygon.size()], outward)) {
        return false;
      }
    }
    return true;
  };
  constexpr int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                      {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int64_t x = 0; x < grid.shape[0] - 1; ++x)
    for (int64_t y = 0; y < grid.shape[1] - 1; ++y)
      for (int64_t z = 0; z < grid.shape[2] - 1; ++z) {
        for (const auto& permutation : permutations) {
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
            const int32_t status = cut_cell(tet, field, request.offset, result,
                                            triangles, skin_begin, location, root, carrier);
            if (status != PHX_MC_OK) return status;
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
            ok = emit_quad({a, b, d, c}, outward);
          }
          if (!ok) return PHX_MC_INTERNAL_ERROR;
          const int32_t status = cut_cell(tet, field, request.offset, result,
                                          triangles, skin_begin, location, root, carrier);
          if (status != PHX_MC_OK) return status;
        }
      }
  return PHX_MC_OK;
}

int32_t assemble_envelope_carrier(EnvelopeGrid& grid, EnvelopeData* data, Carrier& carrier) {
  auto& result = data->surface_points;
  auto& triangles = data->triangles;
  auto& location = grid;
  if (triangles.empty()) return PHX_MC_DEGENERATE_INPUT;
  // Preserve source boundary roots as the exact prefix of the volume
  // carrier. Roots only on skipped zero-area skin triangles remain interior
  // carrier vertices; they never enter the surface deviation certificate.
  std::vector<int64_t> remap(result.size(), -1), root_volume(result.size(), -1);
  std::vector<int8_t> root_used(result.size(), 0);
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
  std::vector<int64_t> grid_volume(static_cast<size_t>(grid.total), -1);
  for (int64_t i = 0; i < grid.total; ++i)
    if (carrier.grid_used[i]) {
      grid_volume[i] = static_cast<int64_t>(data->volume_points.size());
      data->volume_points.push_back(location(i));
    }
  data->cells = std::move(carrier.cells);
  for (C& cell : data->cells)
    for (int64_t& id : cell)
      id = id < 0 ? root_volume[-id - 1] : grid_volume[id];
  for (size_t i = 0; i < result.size(); ++i)
    if (remap[i] >= 0) result[remap[i]] = result[i];
  result.resize(static_cast<size_t>(kept));
  for (F& t : triangles)
    for (int64_t& id : t) id = remap[id];
  return PHX_MC_OK;
}

int32_t certify_envelope(const phx_mc_surface_envelope_create_request& request,
                         const EnvelopeGrid& grid, EnvelopeData* data,
                         const std::vector<V>& cloud) {
  auto& result = data->surface_points;
  auto* bounds = request.bounds;
  double source_upper = 0;
  for (const V& p : cloud) {
    const double nearest = distance(p, result);
    source_upper = std::max(
        source_upper,
        upper_sum(upper_sum(next_up(nearest), grid.covering_radius), grid.arithmetic_error));
  }
  // Triangle covering + 1-Lipschitz cloud distance + P1 interpolation:
  // |d_source-I_h d_cloud| <= covering radius + grid-cell diameter, and the
  // extracted field f moved d_cloud by at most the separation shift, all in
  // grid.error. I_h f=offset on every output triangle and <=error on the source.
  bounds[0] = next_up(source_upper);
  bounds[1] = upper_sum(request.offset, grid.error);
  bounds[2] = next_down(request.offset - grid.error);
  bounds[3] = grid.error;
  if (!std::isfinite(bounds[0]) || !std::isfinite(bounds[1]) || !std::isfinite(bounds[3]) ||
      !(bounds[2] > 0)) return PHX_MC_RANGE_ERROR;
  return PHX_MC_OK;
}
}  // namespace

extern "C" PHX_MC_API int32_t
phx_mc_surface_envelope_create(const phx_mc_surface_envelope_create_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    int32_t status = validate_envelope_request(*request);
    if (status != PHX_MC_OK) return status;
    std::fill(request->bounds, request->bounds + 4, 0.0);
    auto data = std::make_unique<EnvelopeData>();
    EnvelopeGrid grid;
    grid.spacing = request->spacing;
    status = prepare_envelope_grid(*request, grid);
    if (status != PHX_MC_OK) return status;
    std::vector<V> cloud;
    status = build_covering_cloud(*request, grid, cloud);
    if (status != PHX_MC_OK) return status;
    std::vector<double> field(static_cast<std::size_t>(grid.total));
    status = sample_envelope_field(grid, cloud, field);
    if (status != PHX_MC_OK) return status;
    status = separate_field_from_offset(*request, grid, field);
    if (status != PHX_MC_OK) return status;
    Carrier carrier(grid.total);
    status = extract_envelope_carrier(*request, grid, field, *data, carrier);
    if (status != PHX_MC_OK) return status;
    status = assemble_envelope_carrier(grid, data.get(), carrier);
    if (status != PHX_MC_OK) return status;
    status = certify_envelope(*request, grid, data.get(), cloud);
    if (status != PHX_MC_OK) return status;
    data->cloud = std::move(cloud);
    data->covering_radius = grid.covering_radius;
    data->arithmetic_error = grid.arithmetic_error;
    data->roundoff = grid.roundoff;
    data->spacing = grid.spacing;
    data->offset = request->offset;
    std::copy(request->bounds, request->bounds + 4, data->bounds.begin());
    *request->result = data.release();
    return PHX_MC_OK;
  });
}

extern "C" PHX_MC_API int32_t phx_mc_surface_envelope_sizes(const void* handle,
                                                             int64_t* sizes) {
  if (handle == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const EnvelopeData*>(handle);
  sizes[0] = static_cast<int64_t>(data.surface_points.size());
  sizes[1] = static_cast<int64_t>(data.triangles.size());
  sizes[2] = static_cast<int64_t>(data.volume_points.size());
  sizes[3] = static_cast<int64_t>(data.cells.size());
  return PHX_MC_OK;
}

extern "C" PHX_MC_API int32_t phx_mc_surface_envelope_arrays(
    void* handle, int64_t surface_vertex_capacity, int64_t surface_triangle_capacity,
    int64_t volume_vertex_capacity, int64_t tetrahedron_capacity, double* surface_vertices,
    int64_t* surface_triangles, double* volume_vertices, int64_t* tetrahedra) {
  if (handle == nullptr || surface_vertex_capacity < 0 || surface_triangle_capacity < 0 ||
      volume_vertex_capacity < 0 || tetrahedron_capacity < 0 ||
      !phx::mc::addressable(surface_vertex_capacity, 3, sizeof(double)) ||
      !phx::mc::addressable(surface_triangle_capacity, 3, sizeof(int64_t)) ||
      !phx::mc::addressable(volume_vertex_capacity, 3, sizeof(double)) ||
      !phx::mc::addressable(tetrahedron_capacity, 4, sizeof(int64_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& data = *static_cast<const EnvelopeData*>(handle);
  if (surface_vertex_capacity < static_cast<int64_t>(data.surface_points.size()) ||
      surface_triangle_capacity < static_cast<int64_t>(data.triangles.size()) ||
      volume_vertex_capacity < static_cast<int64_t>(data.volume_points.size()) ||
      tetrahedron_capacity < static_cast<int64_t>(data.cells.size())) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  if ((!data.surface_points.empty() && surface_vertices == nullptr) ||
      (!data.triangles.empty() && surface_triangles == nullptr) ||
      (!data.volume_points.empty() && volume_vertices == nullptr) ||
      (!data.cells.empty() && tetrahedra == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (std::size_t i = 0; i < data.surface_points.size(); ++i)
    std::copy(data.surface_points[i].begin(), data.surface_points[i].end(),
              surface_vertices + 3 * i);
  for (std::size_t i = 0; i < data.triangles.size(); ++i)
    std::copy(data.triangles[i].begin(), data.triangles[i].end(), surface_triangles + 3 * i);
  for (std::size_t i = 0; i < data.volume_points.size(); ++i)
    std::copy(data.volume_points[i].begin(), data.volume_points[i].end(), volume_vertices + 3 * i);
  for (std::size_t i = 0; i < data.cells.size(); ++i)
    std::copy(data.cells[i].begin(), data.cells[i].end(), tetrahedra + 4 * i);
  return PHX_MC_OK;
}

extern "C" PHX_MC_API void phx_mc_surface_envelope_free(void* handle) {
  delete static_cast<EnvelopeData*>(handle);
}
// ---------------------------------------------------------------------------
// Band-constrained remeshing of the extracted envelope surface.
//
// The marching-tetrahedra surface certifies that every point of every output
// triangle lies in the band bounds[2] <= d_source <= bounds[1]. Its triangles
// are cuts of grid tetrahedra, often needles and caps. The source-agnostic
// remesher (surface_remesh.hpp) replaces them toward a target edge length h
// (twice the sampling spacing) and a 30 degree minimum angle under the
// kOffsetBand admission: vertex values are bounded through the covering cloud
// exactly as in the certificate (d_cloud - covering - arithmetic <= d_source
// <= d_cloud + arithmetic), and a triangle is admitted when its 1-Lipschitz
// enclosure [min L_i - R, max U_i + R] (R its certified free radius) lies in
// the band. The band bounds and the interpolation error are unchanged, and the
// source-to-surface bound is recomputed on the remeshed vertices.
namespace {

// Uniform bucket grid over a point set answering nearest-distance queries.
struct PointIndex {
  const std::vector<V>* points = nullptr;
  double cell = 1;
  V lo{};
  std::array<int64_t, 3> shape{1, 1, 1};
  std::vector<int64_t> start, items;

  int32_t build(const std::vector<V>& values, double cell_size) {
    points = &values;
    cell = cell_size;
    if (values.empty() || !(cell > 0) || !std::isfinite(cell)) return PHX_MC_INVALID_INPUT;
    lo = values[0];
    V hi = values[0];
    for (const V& p : values)
      for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], p[d]);
        hi[d] = std::max(hi[d], p[d]);
      }
    int64_t total = 1;
    for (int d = 0; d < 3; ++d) {
      const double n = std::floor((hi[d] - lo[d]) / cell) + 1;
      if (!std::isfinite(n) || n < 1 || n > 1 << 20) return PHX_MC_UNRESOLVED;
      shape[d] = static_cast<int64_t>(n);
      total *= shape[d];
      if (total > (int64_t{1} << 31)) return PHX_MC_UNRESOLVED;
    }
    start.assign(static_cast<size_t>(total + 1), 0);
    std::vector<int64_t> slot(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      slot[i] = flat(coordinates(values[i]));
      ++start[static_cast<size_t>(slot[i] + 1)];
    }
    for (int64_t i = 0; i < total; ++i) start[i + 1] += start[i];
    items.resize(values.size());
    std::vector<int64_t> fill(start.begin(), start.end() - 1);
    for (size_t i = 0; i < values.size(); ++i)
      items[static_cast<size_t>(fill[static_cast<size_t>(slot[i])]++)] = static_cast<int64_t>(i);
    return PHX_MC_OK;
  }
  std::array<int64_t, 3> coordinates(const V& p) const {
    std::array<int64_t, 3> c{};
    for (int d = 0; d < 3; ++d) {
      const double t = std::floor((p[d] - lo[d]) / cell);
      c[d] = static_cast<int64_t>(std::clamp(t, -1.0e15, 1.0e15));
    }
    return c;
  }
  int64_t flat(const std::array<int64_t, 3>& c) const {
    return (c[0] * shape[1] + c[1]) * shape[2] + c[2];
  }
  // Binary64 distance to the nearest indexed point (each length rounded once).
  double nearest(const V& x) const {
    const auto c = coordinates(x);
    int64_t reach = 0;
    for (int d = 0; d < 3; ++d)
      reach = std::max({reach, std::abs(c[d]), std::abs(c[d] - (shape[d] - 1))});
    double best = std::numeric_limits<double>::infinity();
    for (int64_t r = 0; r <= reach; ++r) {
      if (r > 0 && best <= static_cast<double>(r - 1) * cell) break;
      for (int64_t i = std::max<int64_t>(0, c[0] - r); i <= std::min(shape[0] - 1, c[0] + r); ++i)
        for (int64_t j = std::max<int64_t>(0, c[1] - r); j <= std::min(shape[1] - 1, c[1] + r);
             ++j) {
          const bool shell = std::abs(i - c[0]) == r || std::abs(j - c[1]) == r;
          for (int64_t k = std::max<int64_t>(0, c[2] - r);
               k <= std::min(shape[2] - 1, c[2] + r); ++k) {
            if (!shell && std::abs(k - c[2]) != r) continue;
            const int64_t cell_index = flat({i, j, k});
            for (int64_t s = start[static_cast<size_t>(cell_index)];
                 s < start[static_cast<size_t>(cell_index + 1)]; ++s)
              best = std::min(best, length(difference(x, (*points)[static_cast<size_t>(items[s])])));
          }
        }
    }
    return best;
  }
  // Nearest indexed point itself (for projection of candidate positions).
  int64_t nearest_index(const V& x) const {
    const auto c = coordinates(x);
    int64_t reach = 0;
    for (int d = 0; d < 3; ++d)
      reach = std::max({reach, std::abs(c[d]), std::abs(c[d] - (shape[d] - 1))});
    double best = std::numeric_limits<double>::infinity();
    int64_t found = -1;
    for (int64_t r = 0; r <= reach; ++r) {
      if (r > 0 && best <= static_cast<double>(r - 1) * cell) break;
      for (int64_t i = std::max<int64_t>(0, c[0] - r); i <= std::min(shape[0] - 1, c[0] + r); ++i)
        for (int64_t j = std::max<int64_t>(0, c[1] - r); j <= std::min(shape[1] - 1, c[1] + r);
             ++j) {
          const bool shell = std::abs(i - c[0]) == r || std::abs(j - c[1]) == r;
          for (int64_t k = std::max<int64_t>(0, c[2] - r);
               k <= std::min(shape[2] - 1, c[2] + r); ++k) {
            if (!shell && std::abs(k - c[2]) != r) continue;
            const int64_t cell_index = flat({i, j, k});
            for (int64_t s = start[static_cast<size_t>(cell_index)];
                 s < start[static_cast<size_t>(cell_index + 1)]; ++s) {
              const int64_t item = items[static_cast<size_t>(s)];
              const double value = length(difference(x, (*points)[static_cast<size_t>(item)]));
              if (value < best || (value == best && item < found)) {
                best = value;
                found = item;
              }
            }
          }
        }
    }
    return found;
  }
  // Calls visit(point) for every indexed point in the cells the box
  // [x - radius, x + radius] meets (a superset of the points within radius).
  template <class Visit>
  void for_each_near(const V& x, double radius, Visit&& visit) const {
    std::array<int64_t, 3> low{}, high{};
    for (int d = 0; d < 3; ++d) {
      V shifted = x;
      shifted[d] = x[d] - radius;
      low[d] = std::max<int64_t>(0, coordinates(shifted)[d]);
      shifted[d] = x[d] + radius;
      high[d] = std::min<int64_t>(shape[d] - 1, coordinates(shifted)[d]);
    }
    for (int64_t i = low[0]; i <= high[0]; ++i)
      for (int64_t j = low[1]; j <= high[1]; ++j)
        for (int64_t k = low[2]; k <= high[2]; ++k) {
          const int64_t cell_index = flat({i, j, k});
          for (int64_t s = start[static_cast<size_t>(cell_index)];
               s < start[static_cast<size_t>(cell_index + 1)]; ++s)
            visit((*points)[static_cast<size_t>(items[static_cast<size_t>(s)])]);
        }
  }
};

class EnvelopeBandAdmission final : public phx::mc::remesh::TriangleAdmission {
 public:
  EnvelopeBandAdmission(const EnvelopeData& envelope, const PointIndex& cloud_index)
      : data(envelope), cloud(cloud_index), lower(envelope.bounds[2]), upper(envelope.bounds[1]) {}

  phx::mc::remesh::AdmissionKind kind() const override {
    return phx::mc::remesh::AdmissionKind::kOffsetBand;
  }
  phx::mc::remesh::VertexEvidence evaluate(const V& position) const override {
    const double nearest = cloud.nearest(position);
    return {next_down(next_down(nearest - data.covering_radius) - data.arithmetic_error),
            next_up(nearest + data.arithmetic_error)};
  }
  bool admits(const V& a, const V& b, const V& c,
              const std::array<phx::mc::remesh::VertexEvidence, 3>& evidence) const override {
    const double radius = phx::mc::remesh::certified_free_radius(a, b, c, data.roundoff);
    if (!std::isfinite(radius)) return false;
    double high = 0, low = std::numeric_limits<double>::infinity();
    for (const auto& vertex : evidence) {
      high = std::max(high, vertex.upper);
      low = std::min(low, vertex.lower);
    }
    return next_up(high + radius) <= upper && next_down(low - radius) >= lower;
  }
  // Radial projection onto the offset sphere of the nearest cloud point.
  V project(const V& x) const override {
    const int64_t item = cloud.nearest_index(x);
    if (item < 0) return x;
    const V& p = data.cloud[static_cast<size_t>(item)];
    const V direction = difference(x, p);
    const double norm = length(direction);
    if (!(norm > 0) || !std::isfinite(norm)) return x;
    return add(p, mul(direction, data.offset / norm));
  }
  // Source normals of the band piece are the directions (x - q) / |x - q| from
  // a nearest source point q of a triangle point x. Every x lies within R of
  // a corner v (R = certified free radius); q lies within the covering radius
  // c of a cloud point p, and |x - q| <= |x - p*| for v's nearest cloud point
  // p* (on the source), which gives, with d = |v - p*|,
  //   |v - p|^2 - d^2 <= 2 R |p - p*| + 2 c (d + R) + c^2,  |v - p| <= d + 2R + c,
  // and (x - q) / |x - q| lies within asin((R + c) / |v - p|) of (v - p) / |v - p|.
  // The cone holds every such direction over the candidate cloud points of the
  // three corners (relative slack 1e-9 on the filters).
  phx::mc::remesh::NormalCone normal_cone(
      const V& a, const V& b, const V& c,
      const std::array<phx::mc::remesh::VertexEvidence, 3>&) const override {
    phx::mc::remesh::NormalCone cone;
    const double radius = phx::mc::remesh::certified_free_radius(a, b, c, data.roundoff);
    if (!std::isfinite(radius)) return cone;
    const double covering = data.covering_radius;
    const std::array<const V*, 3> corners{&a, &b, &c};
    std::array<V, 3> nearest{};
    V axis{0, 0, 0};
    for (int k = 0; k < 3; ++k) {
      const int64_t item = cloud.nearest_index(*corners[k]);
      if (item < 0) return cone;
      nearest[k] = data.cloud[static_cast<size_t>(item)];
      const V direction = difference(*corners[k], nearest[k]);
      const double norm = length(direction);
      if (!(norm > radius + covering)) return cone;
      axis = add(axis, mul(direction, 1 / norm));
    }
    const double axis_length = length(axis);
    if (!(axis_length > 0)) return cone;
    axis = mul(axis, 1 / axis_length);
    double half = 0;
    bool open = false;
    for (int k = 0; k < 3 && !open; ++k) {
      const V& v = *corners[k];
      const double d = length(difference(v, nearest[k]));
      const double reach = (d + 2 * radius + covering) * (1 + 1.0e-9);
      cloud.for_each_near(v, reach, [&](const V& p) {
        const V offset = difference(v, p);
        const double squared = dot(offset.data(), offset.data());
        if (squared > reach * reach) return;
        const double allowance = 2 * radius * length(difference(p, nearest[k])) +
                                 2 * covering * (d + radius) + covering * covering;
        if (squared - d * d > allowance + (allowance + d * d) * 1.0e-9) return;
        const double norm = std::sqrt(squared);
        if (!(norm > radius + covering)) {
          open = true;
          return;
        }
        const double between =
            std::atan2(length(cross(axis, offset)), dot(axis.data(), offset.data()));
        half = std::max(half, between + std::asin((radius + covering) / norm));
      });
    }
    if (open || !(half < cone.half_angle)) return cone;
    cone.axis = axis;
    cone.half_angle = half;
    return cone;
  }
  // Every cone holds the nearest cloud point p* of each corner as a candidate,
  // whose term is at least asin((R + c) / |v - p*|) >= asin((R + c) / U_v).
  double normal_cone_floor(
      const V& a, const V& b, const V& c,
      const std::array<phx::mc::remesh::VertexEvidence, 3>& evidence) const override {
    const double radius = phx::mc::remesh::certified_free_radius(a, b, c, data.roundoff);
    double nearest_upper = std::numeric_limits<double>::infinity();
    for (const auto& vertex : evidence) nearest_upper = std::min(nearest_upper, vertex.upper);
    const double margin = radius + data.covering_radius;
    if (!std::isfinite(radius) || !(nearest_upper > margin)) return phx::mc::remesh::kOpenCone;
    return std::asin(margin / nearest_upper);
  }

 private:
  const EnvelopeData& data;
  const PointIndex& cloud;
  double lower, upper;
};

int32_t remesh_envelope(EnvelopeData& data, phx::mc::remesh::RemeshCounters& counters) {
  const double h = 2 * data.spacing;
  const double lower = data.bounds[2], upper = data.bounds[1];
  if (!(h > 0) || !(lower > 0) || !(upper > lower)) return PHX_MC_INVALID_ARGUMENT;
  PointIndex cloud;
  int32_t status = cloud.build(data.cloud, std::max(h / 2, data.covering_radius));
  if (status != PHX_MC_OK) return status;
  const EnvelopeBandAdmission admission(data, cloud);
  std::vector<V> points = data.surface_points;
  std::vector<F> triangles = data.triangles;
  status = phx::mc::remesh::remesh_surface(points, triangles, admission, {h, 30.0}, counters);
  if (status != PHX_MC_OK) return status;
  // Source-to-surface: every source point lies within the covering radius of
  // a cloud point, which lies within its nearest-vertex distance of the
  // surface.
  PointIndex vertices;
  status = vertices.build(points, h);
  if (status != PHX_MC_OK) return status;
  double source_upper = 0;
  for (const V& p : data.cloud)
    source_upper = std::max(source_upper,
                            upper_sum(upper_sum(next_up(vertices.nearest(p)),
                                                data.covering_radius),
                                      data.arithmetic_error));
  if (!std::isfinite(source_upper)) return PHX_MC_RANGE_ERROR;
  data.bounds[0] = next_up(source_upper);
  data.surface_points = std::move(points);
  data.triangles = std::move(triangles);
  // The marching-tetrahedra carrier no longer bounds the remeshed surface.
  data.volume_points = data.surface_points;
  data.cells.clear();
  return PHX_MC_OK;
}
}  // namespace

extern "C" PHX_MC_API int32_t phx_mc_surface_envelope_remesh(void* handle, double* bounds,
                                                              int64_t* counters) {
  if (handle == nullptr || bounds == nullptr || counters == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    auto& data = *static_cast<EnvelopeData*>(handle);
    if (data.cloud.empty() || data.triangles.empty()) return PHX_MC_INVALID_ARGUMENT;
    phx::mc::remesh::RemeshCounters result;
    const int32_t status = remesh_envelope(data, result);
    if (status != PHX_MC_OK) return status;
    std::copy(data.bounds.begin(), data.bounds.end(), bounds);
    counters[0] = result.splits;
    counters[1] = result.collapses;
    counters[2] = result.flips;
    counters[3] = result.relocations;
    counters[4] = result.unmet;
    return PHX_MC_OK;
  });
}
