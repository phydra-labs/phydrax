//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact PLC validation, triangulation, and source-entity cavity recovery.
// Cavity broad phases follow current adjacency through inclusive AABB overlaps;
// exact segment/face or triangle/face contacts alone decide membership.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <new>
#include <set>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <map>
#include <iterator>
#include "geometry3.hpp"
#include "predicates.hpp"
#include "facet_coverage.hpp"
#include "triangulation3d.hpp"

#include "intersections.hpp"
#include "mesh.hpp"
#include "plc3d.hpp"
#include "topology_keys.hpp"

namespace phx::mc {
namespace {

using Triple = std::array<int32_t, 3>;

Triple sorted_triple(const int32_t* vertices) {
  Triple key{vertices[0], vertices[1], vertices[2]};
  std::sort(key.begin(), key.end());
  return key;
}

bool legal_contact(int8_t kind) {
  return kind == PHX_MC_DISJOINT || kind == PHX_MC_SHARED_VERTEX ||
         kind == PHX_MC_SHARED_EDGE;
}

std::size_t trace_index_from_environment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0' || *value == '-') {
    return std::numeric_limits<std::size_t>::max();
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(parsed);
}

std::size_t diagnostic_threshold_from_environment(const char* name,
                                                  std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return fallback;
  }
  return static_cast<std::size_t>(parsed);
}

struct Box {
  double lower[3];
  double upper[3];
};

bool overlap(const Box& a, const Box& b) {
  for (int axis = 0; axis < 3; ++axis) {
    if (a.upper[axis] < b.lower[axis] || b.upper[axis] < a.lower[axis]) return false;
  }
  return true;
}

int projection_axis(const double* a, const double* b, const double* c) {
  double u[3], v[3], normal[3];
  geometry::difference(b, a, u);
  geometry::difference(c, a, v);
  geometry::cross(u, v, normal);
  int axes[3] = {0, 1, 2};
  std::sort(axes, axes + 3, [&](int left, int right) {
    return std::fabs(normal[left]) > std::fabs(normal[right]);
  });
  for (int axis : axes) {
    const int x = (axis + 1) % 3;
    const int y = (axis + 2) % 3;
    const double pa[2] = {a[x], a[y]};
    const double pb[2] = {b[x], b[y]};
    const double pc[2] = {c[x], c[y]};
    if (orient2d(pa, pb, pc) != 0) return axis;
  }
  return -1;
}

bool segments_meet(const double* a, const double* b, const double* c, const double* d) {
  if (orient3d(a, b, c, d) != 0) return false;
  const bool c_on_line = geometry::collinear(a, b, c, 1);
  const bool d_on_line = geometry::collinear(a, b, d, 1);
  if (c_on_line && d_on_line) {
    int axis = 0;
    for (int k = 0; k < 3; ++k) {
      if (a[k] != b[k]) {
        axis = k;
        break;
      }
    }
    const double lo1 = std::min(a[axis], b[axis]);
    const double hi1 = std::max(a[axis], b[axis]);
    const double lo2 = std::min(c[axis], d[axis]);
    const double hi2 = std::max(c[axis], d[axis]);
    return !(hi1 < lo2 || hi2 < lo1);
  }
  const int axis = projection_axis(a, b, c_on_line ? d : c);
  if (axis < 0) return false;
  const int x = (axis + 1) % 3;
  const int y = (axis + 2) % 3;
  const double pa[2] = {a[x], a[y]};
  const double pb[2] = {b[x], b[y]};
  const double pc[2] = {c[x], c[y]};
  const double pd[2] = {d[x], d[y]};
  const int o1 = orient2d(pa, pb, pc);
  const int o2 = orient2d(pa, pb, pd);
  const int o3 = orient2d(pc, pd, pa);
  const int o4 = orient2d(pc, pd, pb);
  const auto within = [](const double* p, const double* q, const double* r) {
    return std::min(p[0], q[0]) <= r[0] && r[0] <= std::max(p[0], q[0]) &&
           std::min(p[1], q[1]) <= r[1] && r[1] <= std::max(p[1], q[1]);
  };
  if (o1 * o2 < 0 && o3 * o4 < 0) return true;
  return (o1 == 0 && within(pa, pb, pc)) || (o2 == 0 && within(pa, pb, pd)) ||
         (o3 == 0 && within(pc, pd, pa)) || (o4 == 0 && within(pc, pd, pb));
}

// p lies in the closed coordinate box of a and b (p collinear with ab).
bool between_on_line(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& p) {
  for (int axis = 0; axis < 3; ++axis) {
    const int low = (coordinate(p, axis) - coordinate(a, axis)).sign();
    const int high = (coordinate(b, axis) - coordinate(p, axis)).sign();
    if (low * high < 0) return false;
  }
  return true;
}

// segments_meet on exact (possibly witnessed) positions.
bool segments_meet(const ExactPointRef& a, const ExactPointRef& b, const ExactPointRef& c,
                   const ExactPointRef& d) {
  if (orient3d(a, b, c, d) != 0) return false;
  const bool c_on_line = collinear(a, b, c);
  const bool d_on_line = collinear(a, b, d);
  if (c_on_line && d_on_line) {
    return between_on_line(a, b, c) || between_on_line(a, b, d) || between_on_line(c, d, a);
  }
  const ExactPointRef& third = c_on_line ? d : c;
  int axis = 0;
  while (axis < 3 && orient2d_projected(a, b, third, axis) == 0) ++axis;
  if (axis == 3) return false;
  const int o1 = orient2d_projected(a, b, c, axis);
  const int o2 = orient2d_projected(a, b, d, axis);
  const int o3 = orient2d_projected(c, d, a, axis);
  const int o4 = orient2d_projected(c, d, b, axis);
  if (o1 * o2 < 0 && o3 * o4 < 0) return true;
  return (o1 == 0 && between_on_line(a, b, c)) || (o2 == 0 && between_on_line(a, b, d)) ||
         (o3 == 0 && between_on_line(c, d, a)) || (o4 == 0 && between_on_line(c, d, b));
}

int32_t triangulate_planar_loop(
    const double* points, int64_t point_count, const int32_t* loop,
    int64_t count, const std::vector<std::array<int32_t, 2>>& constraints,
    std::vector<Triple>& triangles) {
  triangles.clear();
  if (points == nullptr || loop == nullptr || count < 3 ||
      count > std::numeric_limits<int32_t>::max()) {
    return PHX_MC_INVALID_INPUT;
  }
  std::vector<int32_t> distinct(loop, loop + count);
  std::sort(distinct.begin(), distinct.end());
  if (std::adjacent_find(distinct.begin(), distinct.end()) != distinct.end()) {
    return PHX_MC_INVALID_INPUT;
  }
  std::vector<int32_t> local_vertices(loop, loop + count);
  std::unordered_map<int32_t, int32_t> local_rows;
  local_rows.reserve(local_vertices.size() + 2 * constraints.size());
  for (std::size_t index = 0; index < local_vertices.size(); ++index) {
    const int32_t vertex = local_vertices[index];
    if (vertex < 0 || vertex >= point_count) return PHX_MC_INVALID_INPUT;
    local_rows.emplace(vertex, static_cast<int32_t>(index));
  }
  for (const auto& segment : constraints) {
    for (int32_t vertex : segment) {
      if (vertex < 0 || vertex >= point_count) return PHX_MC_INVALID_INPUT;
      if (local_rows.contains(vertex)) continue;
      const int32_t local = static_cast<int32_t>(local_vertices.size());
      local_vertices.push_back(vertex);
      local_rows.emplace(vertex, local);
    }
  }
  int64_t third = 2;
  const auto point = [&](int32_t vertex) {
    return points + 3 * static_cast<std::size_t>(vertex);
  };
  while (third < count &&
         geometry::collinear(point(loop[0]), point(loop[1]), point(loop[third]), 1)) {
    ++third;
  }
  if (third == count) return PHX_MC_DEGENERATE_INPUT;
  const double* a = point(loop[0]);
  const double* b = point(loop[1]);
  const double* c = point(loop[third]);
  for (int32_t vertex : local_vertices) {
    if (orient3d(a, b, c, point(vertex)) != 0) return PHX_MC_INVALID_INPUT;
  }
  const int axis = projection_axis(a, b, c);
  if (axis < 0) return PHX_MC_DEGENERATE_INPUT;
  const int x = (axis + 1) % 3;
  const int y = (axis + 2) % 3;
  std::vector<double> planar(2 * local_vertices.size());
  for (std::size_t index = 0; index < local_vertices.size(); ++index) {
    planar[2 * index] = point(local_vertices[index])[x];
    planar[2 * index + 1] = point(local_vertices[index])[y];
  }
  int64_t lowest = 0;
  for (int64_t index = 1; index < count; ++index) {
    const double* candidate =
        planar.data() + 2 * local_rows.at(loop[index]);
    const double* current =
        planar.data() + 2 * local_rows.at(loop[lowest]);
    if (candidate[0] < current[0] ||
        (candidate[0] == current[0] && candidate[1] < current[1])) {
      lowest = index;
    }
  }
  const double* before =
      planar.data() + 2 * local_rows.at(loop[(lowest + count - 1) % count]);
  const double* center = planar.data() + 2 * local_rows.at(loop[lowest]);
  const double* after =
      planar.data() + 2 * local_rows.at(loop[(lowest + 1) % count]);
  const int sense = orient2d(before, center, after);
  if (sense == 0) return PHX_MC_DEGENERATE_INPUT;
  if (count == 3 && constraints.empty()) {
    triangles.push_back({loop[0], loop[1], loop[2]});
    return PHX_MC_OK;
  }
  std::vector<int32_t> segments;
  std::unordered_set<uint64_t> segment_keys;
  segments.reserve(2 * (static_cast<std::size_t>(count) + constraints.size()));
  segment_keys.reserve(static_cast<std::size_t>(count) + constraints.size());
  const auto add_segment = [&](int32_t first, int32_t second) {
    if (first == second) return false;
    const uint64_t key = topology::undirected_edge(first, second);
    if (segment_keys.insert(key).second) {
      segments.push_back(first);
      segments.push_back(second);
    }
    return true;
  };
  for (int64_t index = 0; index < count; ++index) {
    if (!add_segment(local_rows.at(loop[index]),
                     local_rows.at(loop[(index + 1) % count]))) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (const auto& segment : constraints) {
    if (!add_segment(local_rows.at(segment[0]), local_rows.at(segment[1]))) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  phx_mc_mesh* raw = nullptr;
  const int32_t status = phx_mc_constrained_delaunay_2d(
      static_cast<int64_t>(local_vertices.size()), planar.data(),
      static_cast<int64_t>(segments.size() / 2), segments.data(), 0, nullptr,
      0, 0.0, std::numeric_limits<double>::infinity(), &raw);
  const std::unique_ptr<phx_mc_mesh, void (*)(phx_mc_mesh*)> mesh(
      raw, phx_mc_mesh_free);
  if (status != PHX_MC_OK) return status;
  if (mesh == nullptr || mesh->dimension != 2 || mesh->cell_count() < count - 2) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  if (segments.size() == 2 * static_cast<std::size_t>(count) &&
      local_vertices.size() == static_cast<std::size_t>(count) &&
      mesh->cell_count() != count - 2) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  std::vector<int32_t> inverse(static_cast<std::size_t>(mesh->point_count()), -1);
  for (std::size_t index = 0; index < local_vertices.size(); ++index) {
    const int32_t vertex = mesh->vertex_map[index];
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= inverse.size() ||
        inverse[static_cast<std::size_t>(vertex)] >= 0) {
      return PHX_MC_INTERNAL_ERROR;
    }
    inverse[static_cast<std::size_t>(vertex)] = static_cast<int32_t>(index);
  }
  triangles.reserve(static_cast<std::size_t>(mesh->cell_count()));
  for (int64_t cell = 0; cell < mesh->cell_count(); ++cell) {
    Triple triangle{};
    for (int corner = 0; corner < 3; ++corner) {
      const int32_t local =
          mesh->cells[3 * static_cast<std::size_t>(cell) + corner];
      if (local < 0 || static_cast<std::size_t>(local) >= inverse.size() ||
          inverse[static_cast<std::size_t>(local)] < 0) {
        return PHX_MC_INTERNAL_ERROR;
      }
      triangle[static_cast<std::size_t>(corner)] =
          local_vertices[static_cast<std::size_t>(inverse[static_cast<std::size_t>(local)])];
    }
    if (sense < 0) std::swap(triangle[1], triangle[2]);
    triangles.push_back(triangle);
  }
  return PHX_MC_OK;
}

class SourcePreparer {
 public:
  SourcePreparer(const PlcInput& input, PlcRecovery& output) : input_(input), output_(output) {}

  int32_t run() {
    const int32_t shape = validate_shape();
    if (shape != PHX_MC_OK) return shape;
    const int32_t points_status = validate_points(input_.points, input_.point_count, 3, nullptr);
    if (points_status != PHX_MC_OK) return points_status;
    const int32_t duplicate = check_distinct();
    if (duplicate != PHX_MC_OK) return duplicate;
    const int32_t facets = check_facets();
    if (facets != PHX_MC_OK) return facets;
    const int32_t tolerances = check_tolerances();
    if (tolerances != PHX_MC_OK) return tolerances;
    const int32_t seeds = check_seeds();
    if (seeds != PHX_MC_OK) return seeds;
    const int32_t polygons = triangulate_polygons();
    if (polygons != PHX_MC_OK) return polygons;
    build_groups();
    const int32_t edges = build_edges();
    if (edges != PHX_MC_OK) return edges;
    const int32_t contacts = check_contacts();
    if (contacts != PHX_MC_OK) return contacts;
    const int32_t closure = check_closure();
    if (closure != PHX_MC_OK) return closure;
    publish_source();
    return PHX_MC_OK;
  }

 private:
  enum Kind : int8_t { kTriangle, kSegment, kFree };
  struct Item {
    Box box;
    Kind kind;
    int32_t index;
  };

  int32_t fail(int32_t status, int32_t reason, int32_t first_kind, int64_t first,
               int32_t second_kind = kPlcNoEntity, int64_t second = -1) {
    output_.failure = {reason, first_kind, first, second_kind, second};
    return status;
  }
  int32_t invalid_polygon(int64_t polygon) {
    return fail(PHX_MC_INVALID_INPUT, kPlcInvalidPolygon, kPlcPolygon, polygon);
  }
  const double* point(int32_t vertex) const {
    return input_.points + 3 * static_cast<std::size_t>(vertex);
  }
  Box box_of(const int32_t* vertices, int count) const {
    Box box{};
    for (int axis = 0; axis < 3; ++axis) {
      box.lower[axis] = box.upper[axis] = input_.points[3 * vertices[0] + axis];
      for (int r = 1; r < count; ++r) {
        const double coordinate = input_.points[3 * vertices[r] + axis];
        box.lower[axis] = std::min(box.lower[axis], coordinate);
        box.upper[axis] = std::max(box.upper[axis], coordinate);
      }
    }
    return box;
  }

  int32_t validate_shape() {
    if (input_.point_count < 3 || input_.polygon_count < 1 || input_.facet_count < 1 ||
        input_.segment_count < 0 || input_.seed_count < 0 || input_.points == nullptr ||
        input_.polygon_offsets == nullptr || input_.polygon_facets == nullptr ||
        input_.facet_regions == nullptr ||
        (input_.segment_count > 0 && input_.segments == nullptr) ||
        (input_.seed_count > 0 &&
         (input_.seeds == nullptr || input_.seed_regions == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (input_.policy != BoundaryPolicy::kFixed &&
        input_.policy != BoundaryPolicy::kConforming) return PHX_MC_INVALID_ARGUMENT;
    if (input_.point_count > kMaxMeshPoints ||
        input_.polygon_count > std::numeric_limits<int32_t>::max() ||
        input_.facet_count > std::numeric_limits<int32_t>::max() ||
        input_.segment_count > std::numeric_limits<int32_t>::max()) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    if (input_.polygon_offsets[0] != 0) return PHX_MC_INVALID_INPUT;
    int64_t previous = 0;
    for (int64_t polygon = 0; polygon < input_.polygon_count; ++polygon) {
      const int64_t next = input_.polygon_offsets[polygon + 1];
      if (next < previous || next - previous < 3 || next - previous > input_.point_count) {
        return fail(PHX_MC_INVALID_INPUT, kPlcInvalidPolygon, kPlcPolygon, polygon);
      }
      previous = next;
    }
    if (previous > 0 && input_.polygon_vertices == nullptr) return PHX_MC_INVALID_ARGUMENT;
    return PHX_MC_OK;
  }

  int32_t check_distinct() {
    std::vector<int32_t> vertex_map;
    const auto representatives = deduplicate_points(input_.points, input_.point_count, 3,
                                                     nullptr, vertex_map);
    if (representatives.size() == static_cast<std::size_t>(input_.point_count)) return PHX_MC_OK;
    for (std::size_t vertex = 0; vertex < vertex_map.size(); ++vertex) {
      if (vertex_map[vertex] >= 0 && vertex_map[vertex] != static_cast<int32_t>(vertex)) {
        const int32_t other = vertex_map[vertex];
        return fail(PHX_MC_INVALID_INPUT, kPlcDuplicatePoint, kPlcPoint,
                    std::min<int32_t>(other, static_cast<int32_t>(vertex)), kPlcPoint,
                    std::max<int32_t>(other, static_cast<int32_t>(vertex)));
      }
    }
    return PHX_MC_OK;
  }

  int32_t check_facets() {
    std::vector<int64_t> uses(static_cast<std::size_t>(input_.facet_count), 0);
    for (int64_t polygon = 0; polygon < input_.polygon_count; ++polygon) {
      const int32_t facet = input_.polygon_facets[polygon];
      if (facet < 0 || facet >= input_.facet_count) {
        return fail(PHX_MC_INVALID_INPUT, kPlcInvalidFacet, kPlcPolygon, polygon,
                    kPlcFacet, facet);
      }
      ++uses[static_cast<std::size_t>(facet)];
    }
    for (int64_t facet = 0; facet < input_.facet_count; ++facet) {
      const int32_t positive = input_.facet_regions[2 * facet];
      const int32_t negative = input_.facet_regions[2 * facet + 1];
      if (uses[static_cast<std::size_t>(facet)] == 0 || positive < -1 || negative < -1 ||
          (positive == -1 && negative == -1)) {
        return fail(PHX_MC_INVALID_INPUT, kPlcInvalidFacet, kPlcFacet, facet);
      }
    }
    return PHX_MC_OK;
  }

  int32_t check_tolerances() {
    const auto valid = [](double value) { return std::isfinite(value) && value >= 0.0; };
    for (int64_t row = 0; input_.facet_tolerances != nullptr && row < input_.facet_count; ++row) {
      if (!valid(input_.facet_tolerances[row])) {
        return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcFacet, row);
      }
    }
    for (int64_t row = 0; input_.segment_tolerances != nullptr && row < input_.segment_count; ++row) {
      if (!valid(input_.segment_tolerances[row])) {
        return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcEdge, row);
      }
    }
    return PHX_MC_OK;
  }

  int32_t check_seeds() {
    if (input_.seed_count == 0) return PHX_MC_OK;
    const int32_t status = validate_points(input_.seeds, input_.seed_count, 3, nullptr);
    if (status != PHX_MC_OK) return status;
    std::unordered_set<int32_t> regions;
    for (int64_t facet = 0; facet < input_.facet_count; ++facet) {
      for (int side = 0; side < 2; ++side) {
        const int32_t region = input_.facet_regions[2 * facet + side];
        if (region >= 0) regions.insert(region);
      }
    }
    for (int64_t seed = 0; seed < input_.seed_count; ++seed) {
      const int32_t region = input_.seed_regions[seed];
      if (region < -1 || (region >= 0 && regions.find(region) == regions.end())) {
        return fail(PHX_MC_INVALID_INPUT, kPlcInvalidSeed, kPlcSeed, seed,
                    kPlcRegion, region);
      }
    }
    return PHX_MC_OK;
  }

  int32_t triangulate_polygons() {
    for (int64_t polygon = 0; polygon < input_.polygon_count; ++polygon) {
      const int64_t begin = input_.polygon_offsets[polygon];
      const int64_t count = input_.polygon_offsets[polygon + 1] - begin;
      const int32_t* loop = input_.polygon_vertices + begin;
      std::vector<Triple> triangles;
      const int32_t status = triangulate_planar_loop(
          input_.points, input_.point_count, loop, count, {}, triangles);
      if (status == PHX_MC_INVALID_INPUT || status == PHX_MC_DEGENERATE_INPUT ||
          status == PHX_MC_CONSTRAINT_INTERSECTION) {
        return invalid_polygon(polygon);
      }
      if (status != PHX_MC_OK) return status;
      for (const Triple& triangle : triangles) {
        const int32_t added = add_triangle(triangle, polygon);
        if (added != PHX_MC_OK) return added;
      }
    }
    return PHX_MC_OK;
  }

  int32_t add_triangle(const Triple& face, int64_t polygon) {
    if (triangles_.size() >=
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    triangles_.push_back(face);
    triangle_polygon_.push_back(static_cast<int32_t>(polygon));
    triangle_facet_.push_back(input_.polygon_facets[polygon]);
    return PHX_MC_OK;
  }

  bool coplanar_and_opposed(int32_t first, int32_t second) const {
    const Triple& a = triangles_[static_cast<std::size_t>(first)];
    const Triple& b = triangles_[static_cast<std::size_t>(second)];
    int32_t apex = -1;
    for (int32_t vertex : b) {
      if (std::find(a.begin(), a.end(), vertex) == a.end()) apex = vertex;
    }
    if (apex < 0 || orient3d(point(a[0]), point(a[1]), point(a[2]), point(apex)) != 0) {
      return false;
    }
    for (int r = 0; r < 3; ++r) {
      for (int q = 0; q < 3; ++q) {
        if (a[static_cast<std::size_t>(r)] == b[static_cast<std::size_t>((q + 1) % 3)] &&
            a[static_cast<std::size_t>((r + 1) % 3)] == b[static_cast<std::size_t>(q)]) return true;
      }
    }
    return false;
  }

  void build_groups() {
    std::vector<std::pair<uint64_t, int32_t>> uses;
    uses.reserve(3 * triangles_.size());
    for (int32_t triangle = 0; triangle < static_cast<int32_t>(triangles_.size()); ++triangle) {
      const Triple& face = triangles_[static_cast<std::size_t>(triangle)];
      for (int k = 0; k < 3; ++k) {
        uses.push_back({topology::undirected_edge(face[k], face[(k + 1) % 3]), triangle});
      }
    }
    std::sort(uses.begin(), uses.end());
    std::vector<int32_t> parent(triangles_.size());
    std::iota(parent.begin(), parent.end(), 0);
    for (std::size_t begin = 0; begin < uses.size();) {
      std::size_t end = begin + 1;
      while (end < uses.size() && uses[end].first == uses[begin].first) ++end;
      if (end - begin == 2) {
        const int32_t a = uses[begin].second, b = uses[begin + 1].second;
        if (triangle_facet_[static_cast<std::size_t>(a)] == triangle_facet_[static_cast<std::size_t>(b)] &&
            coplanar_and_opposed(a, b)) {
          parent[static_cast<std::size_t>(topology::find_root(parent, a))] =
              topology::find_root(parent, b);
        }
      }
      begin = end;
    }
    std::vector<int32_t> group_of_root(triangles_.size(), -1);
    triangle_group_.resize(triangles_.size());
    int32_t group_count = 0;
    for (int32_t triangle = 0; triangle < static_cast<int32_t>(triangles_.size()); ++triangle) {
      const int32_t root = topology::find_root(parent, triangle);
      int32_t& group = group_of_root[static_cast<std::size_t>(root)];
      if (group < 0) group = group_count++;
      triangle_group_[static_cast<std::size_t>(triangle)] = group;
    }
    edge_uses_ = std::move(uses);
  }

  bool internal_edge(uint64_t key) const {
    const auto first = std::lower_bound(edge_uses_.begin(), edge_uses_.end(), key,
        [](const auto& use, uint64_t wanted) { return use.first < wanted; });
    const auto last = std::upper_bound(first, edge_uses_.end(), key,
        [](uint64_t wanted, const auto& use) { return wanted < use.first; });
    return last - first == 2 &&
           triangle_group_[static_cast<std::size_t>(first->second)] ==
               triangle_group_[static_cast<std::size_t>((first + 1)->second)];
  }

  int32_t add_edge(int32_t first, int32_t second) {
    if (edges_.size() / 2 >=
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) return -1;
    const int32_t id = static_cast<int32_t>(edges_.size() / 2);
    edges_.push_back(first);
    edges_.push_back(second);
    edge_ids_.emplace(topology::undirected_edge(first, second), id);
    return id;
  }

  int32_t build_edges() {
    for (int64_t segment = 0; segment < input_.segment_count; ++segment) {
      const int32_t a = input_.segments[2 * segment];
      const int32_t b = input_.segments[2 * segment + 1];
      if (a < 0 || b < 0 || a >= input_.point_count || b >= input_.point_count || a == b) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcEdge, segment);
      }
      const uint64_t key = topology::undirected_edge(a, b);
      const auto existing = edge_ids_.find(key);
      if (existing != edge_ids_.end()) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcEdge,
                    existing->second, kPlcEdge, segment);
      }
      if (add_edge(a, b) < 0) return PHX_MC_INDEX_RANGE_ERROR;
    }
    for (const Triple& face : triangles_) {
      for (int k = 0; k < 3; ++k) {
        const int32_t a = face[static_cast<std::size_t>(k)];
        const int32_t b = face[static_cast<std::size_t>((k + 1) % 3)];
        const uint64_t key = topology::undirected_edge(a, b);
        if (!internal_edge(key) && edge_ids_.find(key) == edge_ids_.end()) {
          if (add_edge(std::min(a, b), std::max(a, b)) < 0) {
            return PHX_MC_INDEX_RANGE_ERROR;
          }
        }
      }
    }
    return PHX_MC_OK;
  }

  int32_t check_contacts() {
    std::vector<uint8_t> referenced(static_cast<std::size_t>(input_.point_count), 0);
    for (const Triple& face : triangles_) {
      for (int32_t vertex : face) referenced[static_cast<std::size_t>(vertex)] = 1;
    }
    for (int64_t segment = 0; segment < input_.segment_count; ++segment) {
      referenced[static_cast<std::size_t>(input_.segments[2 * segment])] = 1;
      referenced[static_cast<std::size_t>(input_.segments[2 * segment + 1])] = 1;
    }
    std::vector<Item> items;
    items.reserve(triangles_.size() + static_cast<std::size_t>(input_.segment_count) +
                  static_cast<std::size_t>(input_.point_count));
    for (int32_t triangle = 0; triangle < static_cast<int32_t>(triangles_.size()); ++triangle) {
      const Triple& face = triangles_[static_cast<std::size_t>(triangle)];
      items.push_back({box_of(face.data(), 3), kTriangle, triangle});
    }
    for (int32_t segment = 0; segment < static_cast<int32_t>(input_.segment_count); ++segment) {
      items.push_back({box_of(input_.segments + 2 * segment, 2), kSegment, segment});
    }
    for (int32_t vertex = 0; vertex < input_.point_count; ++vertex) {
      if (referenced[static_cast<std::size_t>(vertex)] != 0) continue;
      const double* p = point(vertex);
      items.push_back({Box{{p[0], p[1], p[2]}, {p[0], p[1], p[2]}}, kFree, vertex});
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
      return a.box.lower[0] < b.box.lower[0] ||
          (a.box.lower[0] == b.box.lower[0] &&
           (a.kind < b.kind || (a.kind == b.kind && a.index < b.index)));
    });
    for (std::size_t i = 0; i < items.size(); ++i) {
      for (std::size_t j = i + 1; j < items.size(); ++j) {
        if (items[j].box.lower[0] > items[i].box.upper[0]) break;
        if (!overlap(items[i].box, items[j].box)) continue;
        if (legal_pair(items[i], items[j]) == PHX_MC_OK) continue;
        const auto entity = [&](const Item& item) -> std::pair<int32_t, int64_t> {
          if (item.kind == kTriangle) {
            return {kPlcPolygon, triangle_polygon_[static_cast<std::size_t>(item.index)]};
          }
          if (item.kind == kSegment) return {kPlcEdge, item.index};
          return {kPlcPoint, item.index};
        };
        const auto [first_kind, first] = entity(items[i]);
        const auto [second_kind, second] = entity(items[j]);
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting,
                    first_kind, first, second_kind, second);
      }
    }
    return PHX_MC_OK;
  }

  int32_t legal_pair(const Item& first, const Item& second) const {
    Contact contact;
    if (first.kind == kTriangle && second.kind == kTriangle) {
      const Triple& a = triangles_[static_cast<std::size_t>(first.index)];
      const Triple& b = triangles_[static_cast<std::size_t>(second.index)];
      const double* first_triangle[3] = {point(a[0]), point(a[1]), point(a[2])};
      const double* second_triangle[3] = {point(b[0]), point(b[1]), point(b[2])};
      const int64_t first_ids[3] = {a[0], a[1], a[2]};
      const int64_t second_ids[3] = {b[0], b[1], b[2]};
      const int32_t status = intersect_triangles(first_triangle, second_triangle,
                                                 first_ids, second_ids, contact);
      return status == PHX_MC_OK && legal_contact(contact.kind) ? PHX_MC_OK
                                                                 : PHX_MC_CONSTRAINT_INTERSECTION;
    }
    if (first.kind == kTriangle && second.kind == kSegment) {
      const Triple& face = triangles_[static_cast<std::size_t>(first.index)];
      const int32_t* segment = input_.segments + 2 * second.index;
      const double* segment_points[2] = {point(segment[0]), point(segment[1])};
      const double* triangle_points[3] = {point(face[0]), point(face[1]), point(face[2])};
      const int64_t segment_ids[2] = {segment[0], segment[1]};
      const int64_t triangle_ids[3] = {face[0], face[1], face[2]};
      const int32_t status = intersect_segment_triangle(segment_points, triangle_points,
                                                        segment_ids, triangle_ids, contact);
      return status == PHX_MC_OK && legal_contact(contact.kind) ? PHX_MC_OK
                                                                 : PHX_MC_CONSTRAINT_INTERSECTION;
    }
    if (first.kind == kTriangle && second.kind == kFree) {
      const Triple& face = triangles_[static_cast<std::size_t>(first.index)];
      const double* triangle_points[3] = {point(face[0]), point(face[1]), point(face[2])};
      int side = 0;
      return locate_on_triangle(point(second.index), triangle_points, side) == kFeatureNone
          ? PHX_MC_OK : PHX_MC_CONSTRAINT_INTERSECTION;
    }
    if (first.kind == kSegment && second.kind == kSegment) {
      const int32_t* a = input_.segments + 2 * first.index;
      const int32_t* b = input_.segments + 2 * second.index;
      const bool shares = a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1];
      if (!shares) {
        return segments_meet(point(a[0]), point(a[1]), point(b[0]), point(b[1]))
            ? PHX_MC_CONSTRAINT_INTERSECTION : PHX_MC_OK;
      }
      const int32_t common = (a[0] == b[0] || a[0] == b[1]) ? a[0] : a[1];
      const int32_t x = a[0] == common ? a[1] : a[0];
      const int32_t y = b[0] == common ? b[1] : b[0];
      const double* c = point(common);
      if (!geometry::collinear(c, point(x), point(y), 1)) return PHX_MC_OK;
      for (int axis = 0; axis < 3; ++axis) {
        const double dx = point(x)[axis] - c[axis];
        const double dy = point(y)[axis] - c[axis];
        if (dx != 0.0 || dy != 0.0) {
          return (dx > 0.0) != (dy > 0.0) ? PHX_MC_OK : PHX_MC_CONSTRAINT_INTERSECTION;
        }
      }
      return PHX_MC_CONSTRAINT_INTERSECTION;
    }
    if (first.kind == kSegment && second.kind == kFree) {
      const int32_t* segment = input_.segments + 2 * first.index;
      const double* p = point(second.index);
      return !(geometry::collinear(point(segment[0]), point(segment[1]), p, 1) &&
               segments_meet(point(segment[0]), point(segment[1]), p, p))
          ? PHX_MC_OK : PHX_MC_CONSTRAINT_INTERSECTION;
    }
    return PHX_MC_OK;
  }

  int32_t check_closure() {
    struct Entry { int32_t region; uint64_t key; int32_t coefficient; };
    std::vector<Entry> entries;
    for (std::size_t triangle = 0; triangle < triangles_.size(); ++triangle) {
      const int32_t facet = triangle_facet_[triangle];
      const int32_t positive = input_.facet_regions[2 * facet];
      const int32_t negative = input_.facet_regions[2 * facet + 1];
      if (positive == negative) continue;
      const Triple& face = triangles_[triangle];
      for (int k = 0; k < 3; ++k) {
        const int32_t a = face[static_cast<std::size_t>(k)];
        const int32_t b = face[static_cast<std::size_t>((k + 1) % 3)];
        const int32_t sense = a < b ? 1 : -1;
        const uint64_t key = topology::undirected_edge(a, b);
        if (positive >= 0) entries.push_back({positive, key, -sense});
        if (negative >= 0) entries.push_back({negative, key, sense});
      }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
      return a.region < b.region || (a.region == b.region && a.key < b.key);
    });
    for (std::size_t begin = 0; begin < entries.size();) {
      std::size_t end = begin;
      int64_t sum = 0;
      while (end < entries.size() && entries[end].region == entries[begin].region &&
             entries[end].key == entries[begin].key) {
        sum += entries[end].coefficient;
        ++end;
      }
      if (sum != 0) {
        const auto edge = edge_ids_.find(entries[begin].key);
        return fail(PHX_MC_INVALID_INPUT, kPlcOpenBoundary, kPlcRegion,
                    entries[begin].region, kPlcEdge,
                    edge == edge_ids_.end() ? -1 : edge->second);
      }
      begin = end;
    }
    return PHX_MC_OK;
  }

  void publish_source() {
    output_.points.assign(input_.points, input_.points + 3 * input_.point_count);
    output_.plc_edges = edges_;
    output_.vertex_dimension.assign(static_cast<std::size_t>(input_.point_count), 0);
    output_.protection.assign(static_cast<std::size_t>(input_.point_count), 0.0);
    output_.witnesses.assign(static_cast<std::size_t>(input_.point_count), PlcRecoveryWitness{});
    output_.input_triangles.clear();
    output_.input_triangles.reserve(3 * triangles_.size());
    for (const Triple& face : triangles_) {
      output_.input_triangles.insert(output_.input_triangles.end(), face.begin(), face.end());
    }
    output_.input_polygons = triangle_polygon_;
    output_.face_orientations.clear();
  }

  const PlcInput& input_;
  PlcRecovery& output_;
  std::vector<Triple> triangles_;
  std::vector<int32_t> triangle_polygon_;
  std::vector<int32_t> triangle_facet_;
  std::vector<int32_t> triangle_group_;
  std::vector<std::pair<uint64_t, int32_t>> edge_uses_;
  std::vector<int32_t> edges_;
  std::unordered_map<uint64_t, int32_t> edge_ids_;
};

}  // namespace

int32_t collect_edge_cavity(const TetrahedralComplex& complex, const double* points,
                            int32_t first, int32_t second, int32_t start,
                            EdgeCavityScratch& scratch, std::vector<int32_t>& cavity) {
  cavity.clear();
  if (!edge_cavity_candidates(complex, points, first, second, start, scratch)) {
    return PHX_MC_INTERNAL_ERROR;
  }
  for (int32_t id : scratch.candidates) {
    const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
    for (int slot = 0; slot < 4; ++slot) {
      int32_t vertices[3];
      oriented_facet(tet.v, slot, vertices);
      const double* segment[2] = {points + 3 * static_cast<std::size_t>(first),
                                  points + 3 * static_cast<std::size_t>(second)};
      const double* triangle[3] = {
          points + 3 * static_cast<std::size_t>(vertices[0]),
          points + 3 * static_cast<std::size_t>(vertices[1]),
          points + 3 * static_cast<std::size_t>(vertices[2])};
      const int64_t segment_ids[2] = {first, second};
      const int64_t triangle_ids[3] = {vertices[0], vertices[1], vertices[2]};
      Contact contact;
      const int32_t status = intersect_segment_triangle(segment, triangle, segment_ids,
                                                        triangle_ids, contact);
      if (status != PHX_MC_OK) return PHX_MC_INTERNAL_ERROR;
      if (!legal_contact(contact.kind)) {
        cavity.push_back(id);
        break;
      }
    }
  }
  return PHX_MC_OK;
}

int32_t prepare_plc_source(const PlcInput& input, PlcRecovery& result) {
  result = PlcRecovery{};
  SourcePreparer preparer(input, result);
  return preparer.run();
}

namespace {

constexpr int32_t kPlcVoidRegion = -3;
constexpr int32_t kPlcCornerCount = 8;

struct PlcFaceUse {
  int32_t tet;
  int32_t slot;
  Triple oriented;
};

struct PlcSourceFace {
  Triple oriented;
  int32_t facet;
  int32_t source_triangle;
};
struct PlcSourceFacePatch {
  std::vector<PlcSourceFace> faces;
  std::vector<PlcSourceFace> source_faces;
  std::vector<PlcSourceFace> recovery_faces;
  std::vector<Triple> replaced_source_faces;
  std::vector<int32_t> cavity;
  std::array<int32_t, 2> recovery_edge{-1, -1};
};
struct PlcPlanarPolygon {
  int32_t facet;
  int32_t source_triangle;
  int32_t axis;
  Triple plane_vertices;
  int orientation;
  std::vector<int32_t> source_triangles;
  std::vector<std::vector<int32_t>> boundary_loops;
};

struct PlcEdgePoint {
  double parameter;
  int32_t vertex;
};

class PlcRecoveryBuilder {
 public:
  PlcRecoveryBuilder(const PlcInput& input, const PlcRecovery& source)
      : input_(input), candidate_(source) {}


  int32_t run(PlcRecovery& output) {
    if (!initialize_points()) return status_;
    if (progress_enabled_) {
      std::fprintf(stderr,
                   "PLC_PROGRESS start policy=%d points=%zu edges=%zu triangles=%zu\n",
                   static_cast<int>(input_.policy), candidate_.points.size() / 3,
                   candidate_.plc_edges.size() / 2,
                   candidate_.input_triangles.size() / 3);
    }
    if (trace_enabled_) {
      std::fprintf(stderr,
                   "PLC_TRACE start policy=%d points=%zu edges=%zu triangles=%zu\n",
                   static_cast<int>(input_.policy), candidate_.points.size() / 3,
                   candidate_.plc_edges.size() / 2,
                   candidate_.input_triangles.size() / 3);
    }
    std::size_t restart_iteration = 0;
    for (;;) {
      ++restart_iteration;
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS restart_begin restart=%zu points=%zu\n",
                     restart_iteration, candidate_.points.size() / 3);
      }
      const bool reuse_incremental = incremental_triangulation_ready_;
      incremental_triangulation_ready_ = false;
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS phase=triangulation_begin restart=%zu mode=%s "
                     "points=%zu rebuilds=%zu insert_attempts=%zu "
                     "insert_accepted=%zu insert_refused=%zu\n",
                     restart_iteration, reuse_incremental ? "reuse" : "rebuild",
                     candidate_.points.size() / 3, progress_full_rebuilds_,
                     progress_incremental_attempts_, progress_incremental_accepted_,
                     progress_incremental_attempts_ - progress_incremental_accepted_);
      }
      if (trace_enabled_) {
        std::fprintf(stderr,
                     "PLC_TRACE triangulation_%s restart=%zu points=%zu\n",
                     reuse_incremental ? "reuse" : "begin", restart_iteration,
                     candidate_.points.size() / 3);
      }
      const auto triangulation_started = std::chrono::steady_clock::now();
      if (!reuse_incremental) {
        status_ = build_triangulation();
        if (status_ != PHX_MC_OK) return status_;
      }
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS phase=triangulation restart=%zu mode=%s "
                     "points=%zu cells=%lld seconds=%.3f rebuilds=%zu "
                     "insert_attempts=%zu insert_accepted=%zu insert_refused=%zu\n",
                     restart_iteration, reuse_incremental ? "reuse" : "rebuild",
                     candidate_.points.size() / 3,
                     static_cast<long long>(triangulation_->complex().finite_count),
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - triangulation_started)
                         .count(),
                     progress_full_rebuilds_, progress_incremental_attempts_,
                     progress_incremental_accepted_,
                     progress_incremental_attempts_ - progress_incremental_accepted_);
      }
      if (trace_enabled_) {
        std::fprintf(stderr,
                     "PLC_TRACE triangulation_ready restart=%zu reused=%d "
                     "finite_cells=%lld\n",
                     restart_iteration, reuse_incremental ? 1 : 0,
                     static_cast<long long>(triangulation_->complex().finite_count));
        std::fprintf(stderr, "PLC_TRACE source_edges_begin restart=%zu\n",
                     restart_iteration);
      }
      bool restart = false;
      const auto edge_recovery_started = std::chrono::steady_clock::now();
      status_ = recover_source_edges(restart);
      if (progress_enabled_) {
        std::fprintf(
            stderr,
            "PLC_PROGRESS phase=source_edges restart=%zu status=%d restart_required=%d "
            "points=%zu cells=%lld seconds=%.3f rebuilds=%zu insert_attempts=%zu "
            "insert_accepted=%zu insert_refused=%zu\n",
            restart_iteration, status_, restart ? 1 : 0,
            candidate_.points.size() / 3,
            static_cast<long long>(triangulation_->complex().finite_count),
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - edge_recovery_started)
                .count(),
            progress_full_rebuilds_, progress_incremental_attempts_,
            progress_incremental_accepted_,
            progress_incremental_attempts_ - progress_incremental_accepted_);
      }
      if (status_ != PHX_MC_OK) return status_;
      if (trace_enabled_) {
        std::fprintf(stderr,
                     "PLC_TRACE source_edges_done restart=%zu restart_required=%d "
                     "points=%zu finite_cells=%lld\n",
                     restart_iteration, restart ? 1 : 0,
                     candidate_.points.size() / 3,
                     static_cast<long long>(triangulation_->complex().finite_count));
      }
      if (restart) continue;
      if (trace_enabled_) {
        std::fprintf(stderr, "PLC_TRACE source_faces_begin restart=%zu\n",
                     restart_iteration);
      }
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS phase=source_faces_begin restart=%zu points=%zu\n",
                     restart_iteration, candidate_.points.size() / 3);
      }
      const auto face_recovery_started = std::chrono::steady_clock::now();
      status_ = recover_source_faces(restart);
      if (progress_enabled_) {
        std::fprintf(
            stderr,
            "PLC_PROGRESS phase=source_faces restart=%zu status=%d restart_required=%d "
            "points=%zu cells=%lld seconds=%.3f\n",
            restart_iteration, status_, restart ? 1 : 0,
            candidate_.points.size() / 3,
            static_cast<long long>(triangulation_->complex().finite_count),
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - face_recovery_started)
                .count());
      }
      if (status_ != PHX_MC_OK) return status_;
      if (trace_enabled_) {
        std::fprintf(stderr,
                     "PLC_TRACE source_faces_done restart=%zu restart_required=%d "
                     "points=%zu finite_cells=%lld\n",
                     restart_iteration, restart ? 1 : 0,
                     candidate_.points.size() / 3,
                     static_cast<long long>(triangulation_->complex().finite_count));
      }
      if (!restart) break;
    }
    status_ = suppress_witnessed_vertices();
    if (status_ != PHX_MC_OK) return status_;
    status_ = classify_regions();
    if (status_ != PHX_MC_OK) return status_;
    if (trace_enabled_) std::fprintf(stderr, "PLC_TRACE embedding_begin\n");
    status_ = check_embedded_entities();
    if (status_ != PHX_MC_OK) return status_;
    publish();
    if (status_ != PHX_MC_OK) return status_;
    output = std::move(candidate_);
    if (trace_enabled_) {
      std::fprintf(stderr,
                   "PLC_TRACE complete points=%zu tets=%zu faces=%zu segments=%zu\n",
                   output.points.size() / 3, output.tets.size() / 4,
                   output.faces.size() / 3, output.segments.size() / 2);
    }
    return PHX_MC_OK;
  }

  const PlcSourceRefusal& source_refusal() const { return candidate_.source_refusal; }
  const PlcFailure& failure() const { return failure_; }

  static int32_t run_wrap_fixture(const detail::PlcWrapCavityInput& input,
                                  detail::PlcWrapCavityResult& result) {
    result = detail::PlcWrapCavityResult{};
    if (input.points.size() % 3 != 0 ||
        input.protected_segments.size() % 2 != 0) {
      return PHX_MC_INVALID_INPUT;
    }
    const std::size_t point_count = input.points.size() / 3;
    const auto valid_vertex = [&](int32_t vertex) {
      return vertex >= 0 && static_cast<std::size_t>(vertex) < point_count;
    };
    if (!valid_vertex(input.recovery_edge[0]) ||
        !valid_vertex(input.recovery_edge[1])) {
      return PHX_MC_INVALID_INPUT;
    }
    for (int32_t vertex : input.vertices) {
      if (!valid_vertex(vertex)) return PHX_MC_INVALID_INPUT;
    }
    for (int32_t vertex : input.protected_segments) {
      if (!valid_vertex(vertex)) return PHX_MC_INVALID_INPUT;
    }
    for (const detail::PlcWrapCavityFace& face : input.faces) {
      for (int32_t vertex : face.vertices) {
        if (!valid_vertex(vertex)) return PHX_MC_INVALID_INPUT;
      }
    }
    PlcInput empty_input;
    PlcRecovery empty_source;
    PlcRecoveryBuilder builder(empty_input, empty_source);
    builder.work_points_.assign(input.points.begin(), input.points.end());
    builder.recovery_edge_ = input.recovery_edge;
    builder.protected_segments_.reserve(input.protected_segments.size() / 2);
    for (std::size_t row = 0; row < input.protected_segments.size(); row += 2) {
      builder.protected_segments_.push_back(
          {input.protected_segments[row], input.protected_segments[row + 1]});
    }
    EdgeRecoverySide side;
    side.vertices.assign(input.vertices.begin(), input.vertices.end());
    std::sort(side.vertices.begin(), side.vertices.end());
    side.vertices.erase(std::unique(side.vertices.begin(), side.vertices.end()),
                        side.vertices.end());
    for (const detail::PlcWrapCavityFace& face : input.faces) {
      builder.add_edge_recovery_face(
          side, face.vertices, face.obstacle, face.source, face.cavity_internal,
          face.owner_tet, face.neighbor_tet);
    }
    EdgeRecoveryContactCache contact_cache;
    EdgeRecoveryFailureDetail failure{};
    const int32_t status = builder.wrap_edge_recovery_side(
        side, result.complete, failure, input.require_edge, &contact_cache);
    result.empty_sphere_apex_missing = failure.empty_sphere_failure;
    result.has_facet_contact = failure.has_facet_contact;
    result.cavity_internal_contact = failure.cavity_internal;
    result.blocking_source = failure.source;
    result.blocking_owner = failure.obstacle_owner;
    result.blocking_neighbor = failure.obstacle_neighbor;
    result.tetrahedra = std::move(side.tets);
    return status;
  }

 private:
  int32_t fail(int32_t status, int32_t reason, int32_t first_kind, int64_t first,
               int32_t second_kind = kPlcNoEntity, int64_t second = -1) {
    failure_ = {reason, first_kind, first, second_kind, second};
    return status;
  }

  const double* point(int32_t vertex) const {
    return work_points_.data() + 3 * static_cast<std::size_t>(vertex);
  }

  // Predicate argument of a work vertex: exact witnessed position or carrier.
  ExactPointRef ref(int32_t vertex) const {
    return exact_points_.reference(work_points_.data(), vertex);
  }

  int orient(int32_t a, int32_t b, int32_t c, int32_t d) const {
    return orient3d(ref(a), ref(b), ref(c), ref(d));
  }

  // Coordinate plane onto which triangle abc projects nondegenerately, tried
  // by decreasing carrier normal component and decided on exact positions.
  int exact_projection_axis(int32_t a, int32_t b, int32_t c) const {
    double u[3], v[3], normal[3];
    geometry::difference(point(b), point(a), u);
    geometry::difference(point(c), point(a), v);
    geometry::cross(u, v, normal);
    int axes[3] = {0, 1, 2};
    std::sort(axes, axes + 3, [&](int left, int right) {
      return std::fabs(normal[left]) > std::fabs(normal[right]);
    });
    for (int axis : axes) {
      if (orient2d_projected(ref(a), ref(b), ref(c), axis) != 0) return axis;
    }
    return -1;
  }

  // Triangulates a simple planar loop listed with projected orientation
  // `orientation` by ear clipping on exact positions: an ear is a strictly
  // convex corner whose closed triangle holds no other loop vertex, so every
  // triangle is nondegenerate, keeps the loop orientation, and the triangles
  // tile the loop (a simple polygon with more than three vertices has two
  // such ears; collinear loop vertices are never ear tips).
  bool exact_loop_triangles(const std::vector<int32_t>& loop, int axis, int orientation,
                            std::vector<Triple>& triangles) const {
    triangles.clear();
    std::vector<int32_t> ring = loop;
    const auto turn = [&](int32_t a, int32_t b, int32_t c) {
      return orient2d_projected(ref(a), ref(b), ref(c), axis) * orientation;
    };
    while (ring.size() > 3) {
      bool clipped = false;
      const std::size_t count = ring.size();
      for (std::size_t at = 0; at < count && !clipped; ++at) {
        const int32_t previous = ring[(at + count - 1) % count];
        const int32_t tip = ring[at];
        const int32_t next = ring[(at + 1) % count];
        if (turn(previous, tip, next) <= 0) continue;
        const bool blocked = std::any_of(ring.begin(), ring.end(), [&](int32_t vertex) {
          return vertex != previous && vertex != tip && vertex != next &&
                 turn(previous, tip, vertex) >= 0 && turn(tip, next, vertex) >= 0 &&
                 turn(next, previous, vertex) >= 0;
        });
        if (blocked) continue;
        triangles.push_back({previous, tip, next});
        ring.erase(ring.begin() + static_cast<std::ptrdiff_t>(at));
        clipped = true;
      }
      if (!clipped) return false;
    }
    if (ring.size() != 3 || turn(ring[0], ring[1], ring[2]) <= 0) return false;
    triangles.push_back({ring[0], ring[1], ring[2]});
    return true;
  }

  int32_t output_vertex(int32_t vertex) const {
    const int32_t points = static_cast<int32_t>(candidate_.points.size() / 3);
    if (vertex < points) return point_output_[static_cast<std::size_t>(vertex)];
    const int32_t volume = vertex - points - corner_count_;
    if (volume < 0 || static_cast<std::size_t>(volume) >= volume_output_.size()) return -1;
    return volume_output_[static_cast<std::size_t>(volume)];
  }

  // Appends one recovery Steiner point after the scaffold corners. Volume
  // points are rebuilt from scratch with the triangulation, so they never
  // enter source protection or split spacing.
  int32_t add_volume_point(const double* coordinates, int32_t& vertex) {
    const std::size_t count = work_points_.size() / 3;
    if (count >= static_cast<std::size_t>(kMaxMeshPoints)) return PHX_MC_INDEX_RANGE_ERROR;
    if (!triangulation_->insert_point_slot(count, count)) return PHX_MC_INTERNAL_ERROR;
    work_points_.insert(work_points_.end(), coordinates, coordinates + 3);
    alive_.push_back(1);
    vertex_tet_.push_back(-1);
    triangulation_->rebind(work_points_.data(), static_cast<int64_t>(count + 1));
    ++volume_count_;
    vertex = static_cast<int32_t>(count);
    return PHX_MC_OK;
  }

  // The domain needs no corner scaffold when every triangle bounding the
  // void supports the whole source point set: the closed exterior boundary
  // is then the convex hull, whose faces are present in every triangulation
  // of the source points. Interfaces between two regions may be nonconvex.
  bool convex_boundary() const {
    bool exterior = false;
    for (std::size_t triangle = 0; triangle < candidate_.input_triangles.size() / 3; ++triangle) {
      const int32_t facet = input_.polygon_facets[candidate_.input_polygons[triangle]];
      if (input_.facet_regions[2 * facet] >= 0 && input_.facet_regions[2 * facet + 1] >= 0) {
        continue;
      }
      exterior = true;
      const int32_t* face = candidate_.input_triangles.data() + 3 * triangle;
      bool positive = false;
      bool negative = false;
      for (int32_t vertex = 0; vertex < source_point_count_; ++vertex) {
        const int side = orient(face[0], face[1], face[2], vertex);
        positive = positive || side > 0;
        negative = negative || side < 0;
        if (positive && negative) return false;
      }
    }
    return exterior;
  }

  // A polygon's triangulation diagonal is not a PLC constraint when the
  // authored polygon is planar; recover the represented polygon surface.
  void prepare_planar_polygons() {
    const std::size_t polygon_count = static_cast<std::size_t>(input_.polygon_count);
    const std::size_t triangle_count = candidate_.input_triangles.size() / 3;
    planar_polygon_index_.assign(polygon_count, -1);
    if (triangle_count < 2) return;

    struct EdgeUse {
      std::int32_t triangle;
      std::int32_t first;
      std::int32_t second;
    };
    std::vector<std::int32_t> facets(triangle_count);
    std::map<std::uint64_t, std::vector<EdgeUse>> edge_uses;
    for (std::size_t row = 0; row < triangle_count; ++row) {
      const std::int32_t polygon = candidate_.input_polygons[row];
      facets[row] = input_.polygon_facets[polygon];
      const std::int32_t* triangle = candidate_.input_triangles.data() + 3 * row;
      for (int side = 0; side < 3; ++side) {
        const std::int32_t first = triangle[side];
        const std::int32_t second = triangle[(side + 1) % 3];
        edge_uses[topology::undirected_edge(first, second)].push_back(
            {static_cast<std::int32_t>(row), first, second});
      }
    }

    std::vector<std::int32_t> parent(triangle_count);
    std::iota(parent.begin(), parent.end(), 0);
    const auto root = [&](std::int32_t value) {
      while (parent[static_cast<std::size_t>(value)] != value) {
        const std::int32_t grandparent =
            parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(value)])];
        parent[static_cast<std::size_t>(value)] = grandparent;
        value = grandparent;
      }
      return value;
    };
    const auto coplanar = [&](std::int32_t first, std::int32_t second) {
      const std::int32_t* a = candidate_.input_triangles.data() + 3 * first;
      const std::int32_t* b = candidate_.input_triangles.data() + 3 * second;
      return orient(a[0], a[1], a[2], b[0]) == 0 &&
             orient(a[0], a[1], a[2], b[1]) == 0 &&
             orient(a[0], a[1], a[2], b[2]) == 0;
    };
    for (const auto& [edge, uses] : edge_uses) {
      (void)edge;
      if (uses.size() != 2 || uses[0].first != uses[1].second ||
          uses[0].second != uses[1].first ||
          facets[static_cast<std::size_t>(uses[0].triangle)] !=
              facets[static_cast<std::size_t>(uses[1].triangle)] ||
          !coplanar(uses[0].triangle, uses[1].triangle)) {
        continue;
      }
      const std::int32_t first_root = root(uses[0].triangle);
      const std::int32_t second_root = root(uses[1].triangle);
      if (first_root != second_root) {
        parent[static_cast<std::size_t>(first_root)] = second_root;
      }
    }

    std::map<std::int32_t, std::vector<std::int32_t>> components;
    for (std::size_t row = 0; row < triangle_count; ++row) {
      components[root(static_cast<std::int32_t>(row))].push_back(
          static_cast<std::int32_t>(row));
    }
    for (const auto& [component_root, members] : components) {
      (void)component_root;
      if (members.size() < 2) continue;
      const std::int32_t source_triangle = members.front();
      const std::int32_t facet = facets[static_cast<std::size_t>(source_triangle)];
      const std::int32_t* reference =
          candidate_.input_triangles.data() + 3 * source_triangle;
      const int axis = projection_axis(point(reference[0]), point(reference[1]),
                                       point(reference[2]));
      if (axis < 0) continue;
      bool planar = true;
      for (std::int32_t triangle : members) {
        if (facets[static_cast<std::size_t>(triangle)] != facet ||
            !coplanar(source_triangle, triangle)) {
          planar = false;
          break;
        }
      }
      if (!planar) continue;

      using DirectedEdge = std::pair<std::int32_t, std::int32_t>;
      std::map<std::uint64_t, std::vector<DirectedEdge>> component_edges;
      for (std::int32_t triangle_row : members) {
        const std::int32_t* triangle =
            candidate_.input_triangles.data() + 3 * triangle_row;
        for (int side = 0; side < 3; ++side) {
          const std::int32_t first = triangle[side];
          const std::int32_t second = triangle[(side + 1) % 3];
          component_edges[topology::undirected_edge(first, second)].emplace_back(first, second);
        }
      }
      std::map<std::int32_t, std::int32_t> next_vertex;
      std::map<std::int32_t, std::int32_t> incoming;
      bool boundary_valid = true;
      for (const auto& [edge, uses] : component_edges) {
        (void)edge;
        if (uses.size() == 2 && uses[0].first == uses[1].second &&
            uses[0].second == uses[1].first) {
          continue;
        }
        if (uses.size() != 1 || !next_vertex.emplace(uses[0].first, uses[0].second).second) {
          boundary_valid = false;
          break;
        }
        ++incoming[uses[0].second];
      }
      if (!boundary_valid || next_vertex.size() < 3) continue;
      for (const auto& [vertex, next] : next_vertex) {
        (void)next;
        if (incoming[vertex] != 1) {
          boundary_valid = false;
          break;
        }
      }
      if (!boundary_valid) continue;

      std::set<std::int32_t> visited;
      std::vector<std::vector<std::int32_t>> loops;
      for (const auto& [start, unused] : next_vertex) {
        (void)unused;
        if (visited.contains(start)) continue;
        std::vector<std::int32_t> loop;
        std::int32_t vertex = start;
        for (;;) {
          if (!visited.insert(vertex).second) {
            if (vertex != start) boundary_valid = false;
            break;
          }
          loop.push_back(vertex);
          const auto next = next_vertex.find(vertex);
          if (next == next_vertex.end()) {
            boundary_valid = false;
            break;
          }
          vertex = next->second;
          if (vertex == start) break;
        }
        if (!boundary_valid || loop.size() < 3) {
          boundary_valid = false;
          break;
        }
        loops.push_back(std::move(loop));
      }
      if (!boundary_valid || visited.size() != next_vertex.size()) continue;

      const int x = (axis + 1) % 3, y = (axis + 2) % 3;
      const double p0[2] = {point(reference[0])[x], point(reference[0])[y]};
      const double p1[2] = {point(reference[1])[x], point(reference[1])[y]};
      const double p2[2] = {point(reference[2])[x], point(reference[2])[y]};
      const int orientation = orient2d(p0, p1, p2);
      if (orientation == 0) continue;
      std::set<std::int32_t> component_polygons;
      for (std::int32_t triangle : members) {
        component_polygons.insert(
            candidate_.input_polygons[static_cast<std::size_t>(triangle)]);
      }
      bool polygon_unassigned = true;
      for (std::int32_t polygon : component_polygons) {
        const std::int32_t existing = planar_polygon_index_[static_cast<std::size_t>(polygon)];
        if (existing >= 0) {
          polygon_unassigned = false;
          break;
        }
      }
      if (!polygon_unassigned) continue;
      PlcPlanarPolygon group{facet, source_triangle, axis,
                             {reference[0], reference[1], reference[2]}, orientation,
                             members, std::move(loops)};
      const std::int32_t group_index = static_cast<std::int32_t>(planar_polygons_.size());
      planar_polygons_.push_back(std::move(group));
      for (std::int32_t polygon : component_polygons) {
        planar_polygon_index_[static_cast<std::size_t>(polygon)] = group_index;
      }
    }
  }

  // Projected coordinates (scaled by `scale`) of a work vertex, exactly.
  void projected(int32_t vertex, int axis, double scale, Expansion* coordinates) const {
    coordinates[0] = coordinate(ref(vertex), (axis + 1) % 3).scaled(scale);
    coordinates[1] = coordinate(ref(vertex), (axis + 2) % 3).scaled(scale);
  }

  // Closed containment of the projected query q in the polygon's projected
  // boundary loops; loop vertices are scaled by `scale` to match q, so a
  // rational query (a face centroid times three) is decided exactly.
  bool point_in_planar_polygon(const PlcPlanarPolygon& polygon, const Expansion* q,
                               double scale) const {
    bool inside = false;
    for (const auto& loop : polygon.boundary_loops) {
      for (std::size_t i = 0, j = loop.size() - 1; i < loop.size(); j = i++) {
        Expansion a[2], b[2];
        projected(loop[i], polygon.axis, scale, a);
        projected(loop[j], polygon.axis, scale, b);
        const int side = orient2d_sign(a, b, q);
        if (side == 0 && (q[0] - a[0]).sign() * (b[0] - q[0]).sign() >= 0 &&
            (q[1] - a[1]).sign() * (b[1] - q[1]).sign() >= 0) {
          return true;
        }
        const bool a_above = (a[1] - q[1]).sign() > 0;
        const bool b_above = (b[1] - q[1]).sign() > 0;
        const int rising = (b[1] - a[1]).sign();
        if (a_above != b_above && ((rising > 0 && side > 0) || (rising < 0 && side < 0))) {
          inside = !inside;
        }
      }
    }
    return inside;
  }

  bool face_in_planar_polygon(const PlcPlanarPolygon& polygon, const Triple& face) const {
    for (int slot = 0; slot < 3; ++slot) {
      Expansion q[2];
      projected(face[slot], polygon.axis, 1.0, q);
      if (orient(polygon.plane_vertices[0], polygon.plane_vertices[1],
                 polygon.plane_vertices[2], face[slot]) != 0 ||
          !point_in_planar_polygon(polygon, q, 1.0)) {
        return false;
      }
    }
    const int axis = polygon.axis;
    // c lies on the closed segment ab, distinct from both endpoints.
    const auto strictly_on_segment = [&](int32_t a, int32_t b, int32_t c) {
      return orient2d_projected(ref(a), ref(b), ref(c), axis) == 0 &&
             between_on_line(ref(a), ref(b), ref(c)) && !same_point(ref(c), ref(a)) &&
             !same_point(ref(c), ref(b));
    };
    for (int face_side = 0; face_side < 3; ++face_side) {
      const int32_t first = face[face_side];
      const int32_t second = face[(face_side + 1) % 3];
      for (const auto& loop : polygon.boundary_loops) {
        for (std::size_t edge = 0; edge < loop.size(); ++edge) {
          const int32_t boundary_first = loop[edge];
          const int32_t boundary_second = loop[(edge + 1) % loop.size()];
          const int first_side =
              orient2d_projected(ref(first), ref(second), ref(boundary_first), axis);
          const int second_side =
              orient2d_projected(ref(first), ref(second), ref(boundary_second), axis);
          const int third_side =
              orient2d_projected(ref(boundary_first), ref(boundary_second), ref(first), axis);
          const int fourth_side =
              orient2d_projected(ref(boundary_first), ref(boundary_second), ref(second), axis);
          if (first_side * second_side < 0 && third_side * fourth_side < 0) return false;
          if (strictly_on_segment(first, second, boundary_first) ||
              strictly_on_segment(first, second, boundary_second)) {
            return false;
          }
        }
      }
    }
    // Three times the centroid against loops scaled by three.
    Expansion centroid[2];
    for (int32_t vertex : face) {
      Expansion q[2];
      projected(vertex, axis, 1.0, q);
      centroid[0] = centroid[0] + q[0];
      centroid[1] = centroid[1] + q[1];
    }
    return point_in_planar_polygon(polygon, centroid, 3.0);
  }

  int32_t recover_planar_polygons(
      const std::map<Triple, std::vector<PlcFaceUse>>& faces) {
    planar_source_faces_.clear();
    for (auto source = source_faces_.begin(); source != source_faces_.end();) {
      const std::int32_t polygon = candidate_.input_polygons[
          static_cast<std::size_t>(source->second.source_triangle)];
      if (planar_polygon_index_[static_cast<std::size_t>(polygon)] >= 0) {
        source = source_faces_.erase(source);
      } else {
        ++source;
      }
    }
    TetrahedralComplex& complex = triangulation_->complex();
    for (const PlcPlanarPolygon& polygon : planar_polygons_) {
      const int x = (polygon.axis + 1) % 3, y = (polygon.axis + 2) % 3;
      Expansion polygon_area;
      for (std::int32_t source_triangle : polygon.source_triangles) {
        const std::int32_t* triangle =
            candidate_.input_triangles.data() + 3 * source_triangle;
        for (int slot = 0; slot < 3; ++slot) {
          const double* first = point(triangle[slot]);
          const double* second = point(triangle[(slot + 1) % 3]);
          polygon_area = polygon_area + Expansion::product(first[x], second[y]) -
                        Expansion::product(first[y], second[x]);
        }
      }
      Expansion mesh_area;
      std::size_t covered = 0;
      for (const auto& [key, uses] : faces) {
        const PlcFaceUse* finite[2] = {nullptr, nullptr};
        int finite_count = 0;
        for (const PlcFaceUse& use : uses) {
          if (!is_ghost(complex.tets[static_cast<std::size_t>(use.tet)])) {
            if (finite_count < 2) finite[finite_count] = &use;
            ++finite_count;
          }
        }
        if (finite_count == 0) continue;
        if (!face_in_planar_polygon(polygon, key)) continue;
        if (finite_count > 2) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable,
                      kPlcFacet, polygon.facet);
        }
        if (finite_count == 2 &&
            FacetKey::of(finite[0]->oriented.data()) !=
                FacetKey::of(finite[1]->oriented.data()).reversed()) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable,
                      kPlcFacet, polygon.facet);
        }
        Triple oriented = finite[0]->oriented;
        if (orient2d_projected(ref(oriented[0]), ref(oriented[1]), ref(oriented[2]),
                               polygon.axis) != polygon.orientation) {
          std::swap(oriented[1], oriented[2]);
        }
        for (int slot = 0; slot < 3; ++slot) {
          Expansion first[2], second[2];
          projected(oriented[slot], polygon.axis, 1.0, first);
          projected(oriented[(slot + 1) % 3], polygon.axis, 1.0, second);
          mesh_area = mesh_area + first[0] * second[1] - first[1] * second[0];
        }
        const auto [unused, inserted] = planar_source_faces_.emplace(
            key, PlcSourceFace{oriented, polygon.facet, polygon.source_triangle});
        (void)unused;
        if (!inserted) {
          return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting,
                      kPlcFacet, polygon.facet);
        }
        complex.set_facet_constraint(finite[0]->tet, finite[0]->slot, polygon.facet);
        ++covered;
      }
      if (covered == 0 || (mesh_area - polygon_area).sign() != 0) {
        return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable,
                    kPlcFacet, polygon.facet);
      }
    }
    return PHX_MC_OK;
  }
  bool initialize_points() {
    const int64_t input_count = static_cast<int64_t>(candidate_.points.size() / 3);
    if (input_count < 3 || input_count > kMaxMeshPoints) {
      status_ = fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcPoint, input_count);
      return false;
    }
    source_point_count_ = static_cast<int32_t>(input_count);
    work_points_ = candidate_.points;
    for (std::size_t triangle = 0; triangle < candidate_.input_triangles.size() / 3; ++triangle) {
      const int32_t polygon = candidate_.input_polygons[triangle];
      const int32_t facet = input_.polygon_facets[polygon];
      const std::size_t offset = 3 * triangle;
      const Triple face{candidate_.input_triangles[offset],
                        candidate_.input_triangles[offset + 1],
                        candidate_.input_triangles[offset + 2]};
      const auto [unused, inserted] = source_faces_.emplace(
          sorted_triple(face.data()),
          PlcSourceFace{face, facet, static_cast<int32_t>(triangle)});
      (void)unused;
      if (!inserted) {
        status_ = fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting,
                       kPlcPolygon, polygon);
        return false;
      }
    }
    prepare_planar_polygons();
    corner_count_ = convex_boundary() ? 0 : kPlcCornerCount;
    if (input_count > kMaxMeshPoints - corner_count_) {
      status_ = fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcPoint, input_count);
      return false;
    }
    if (corner_count_ != 0) {
      double lower[3], upper[3];
      for (int axis = 0; axis < 3; ++axis) {
        lower[axis] = upper[axis] = work_points_[static_cast<std::size_t>(axis)];
        for (int32_t vertex = 1; vertex < source_point_count_; ++vertex) {
          const double value = work_points_[3 * static_cast<std::size_t>(vertex) + axis];
          lower[axis] = std::min(lower[axis], value);
          upper[axis] = std::max(upper[axis], value);
        }
      }
      double reach = 0.0;
      for (int axis = 0; axis < 3; ++axis) reach = std::max(reach, upper[axis] - lower[axis]);
      reach = std::exp2(std::ceil(std::log2(std::max(reach, 0x1p-60))) + 2.0);
      for (int32_t corner = 0; corner < corner_count_; ++corner) {
        for (int axis = 0; axis < 3; ++axis) {
          const double center = lower[axis] + 0.5 * (upper[axis] - lower[axis]);
          corner_points_.push_back((corner >> axis) & 1 ? center + reach : center - reach);
        }
      }
    }
    const int32_t point_status = validate_points(
        candidate_.points.data(), static_cast<int64_t>(candidate_.points.size() / 3), 3, nullptr);
    if (point_status != PHX_MC_OK) {
      status_ = fail(point_status, kPlcNonrepresentable, kPlcPoint, -1);
      return false;
    }
    if (corner_count_ != 0) {
      const int32_t corner_status = validate_points(corner_points_.data(), corner_count_, 3, nullptr);
      if (corner_status != PHX_MC_OK) {
        status_ = fail(corner_status, kPlcNonrepresentable, kPlcPoint, -1);
        return false;
      }
    }
    const std::size_t edge_count = candidate_.plc_edges.size() / 2;
    source_segment_rows_.assign(edge_count, -1);
    polygon_loop_edges_.clear();
    for (int64_t polygon = 0; polygon < input_.polygon_count; ++polygon) {
      const int64_t begin = input_.polygon_offsets[polygon];
      const int64_t end = input_.polygon_offsets[polygon + 1];
      for (int64_t vertex = begin; vertex < end; ++vertex) {
        const int64_t next = vertex + 1 == end ? begin : vertex + 1;
        polygon_loop_edges_.insert(topology::undirected_edge(
            input_.polygon_vertices[vertex], input_.polygon_vertices[next]));
      }
    }
    std::unordered_map<uint64_t, int32_t> source_segment_rows;
    source_segment_rows.reserve(static_cast<std::size_t>(input_.segment_count));
    for (int32_t row = 0; row < input_.segment_count; ++row) {
      const uint64_t key = topology::undirected_edge(input_.segments[2 * row],
                                                     input_.segments[2 * row + 1]);
      if (!source_segment_rows.emplace(key, row).second) {
        status_ = fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting,
                       kPlcEdge, row);
        return false;
      }
    }
    std::vector<std::pair<uint64_t, int32_t>> triangle_edges;
    triangle_edges.reserve(candidate_.input_triangles.size());
    for (std::size_t triangle = 0;
         triangle < candidate_.input_triangles.size() / 3; ++triangle) {
      const int32_t* face = candidate_.input_triangles.data() + 3 * triangle;
      for (int side = 0; side < 3; ++side) {
        triangle_edges.emplace_back(
            topology::undirected_edge(face[side], face[(side + 1) % 3]),
            static_cast<int32_t>(triangle));
      }
    }
    std::sort(triangle_edges.begin(), triangle_edges.end());
    edge_triangle_offsets_.clear();
    edge_triangle_offsets_.reserve(edge_count + 1);
    edge_triangle_rows_.clear();
    edge_triangle_offsets_.push_back(0);
    for (std::size_t edge = 0; edge < edge_count; ++edge) {
      const int32_t first = candidate_.plc_edges[2 * edge];
      const int32_t second = candidate_.plc_edges[2 * edge + 1];
      const uint64_t key = topology::undirected_edge(first, second);
      const auto source = source_segment_rows.find(key);
      if (source != source_segment_rows.end()) source_segment_rows_[edge] = source->second;
      const auto begin = std::lower_bound(
          triangle_edges.begin(), triangle_edges.end(), key,
          [](const auto& use, uint64_t wanted) { return use.first < wanted; });
      const auto end = std::upper_bound(
          begin, triangle_edges.end(), key,
          [](uint64_t wanted, const auto& use) { return wanted < use.first; });
      for (auto use = begin; use != end; ++use) edge_triangle_rows_.push_back(use->second);
      edge_triangle_offsets_.push_back(edge_triangle_rows_.size());
      if (source_segment_rows_[edge] < 0 && !polygon_loop_edges_.contains(key) &&
          begin == end) {
        status_ = fail(PHX_MC_INVALID_INPUT, kPlcNonrepresentable, kPlcEdge, edge);
        return false;
      }
    }
    edge_chains_.resize(edge_count);
    std::vector<int32_t> source_edges;
    for (std::size_t edge = 0; edge < edge_chains_.size(); ++edge) {
      if (!is_source_edge(edge)) continue;
      const int32_t first = candidate_.plc_edges[2 * edge];
      const int32_t second = candidate_.plc_edges[2 * edge + 1];
      edge_chains_[edge] = {{0.0, first}, {1.0, second}};
      source_edges.push_back(first);
      source_edges.push_back(second);
    }
    // Acute input vertices (two incident source edges at an angle below 90
    // degrees) are the apices of the concentric split shells.
    const std::vector<double> radii = protection_radii(
        candidate_.points.data(), source_point_count_, source_edges,
        candidate_.input_triangles);
    shell_apex_.assign(static_cast<std::size_t>(source_point_count_), 0);
    for (std::size_t vertex = 0; vertex < radii.size(); ++vertex) {
      shell_apex_[vertex] = radii[vertex] > 0.0 ? 1 : 0;
    }
    candidate_.witnesses.resize(static_cast<std::size_t>(source_point_count_));
    candidate_.vertex_dimension.resize(static_cast<std::size_t>(source_point_count_), 0);
    return true;
  }

  int32_t build_triangulation() {
    if (progress_enabled_) ++progress_full_rebuilds_;
    triangulation_.reset();
    work_points_ = candidate_.points;
    work_points_.insert(work_points_.end(), corner_points_.begin(), corner_points_.end());
    volume_count_ = 0;
    local_edits_ = false;
    const int64_t count = static_cast<int64_t>(work_points_.size() / 3);
    triangulation_ = std::make_unique<Triangulation3D>(work_points_.data(), nullptr, count);
    triangulation_->set_exact_points(&exact_points_);
    alive_.assign(static_cast<std::size_t>(count), 1);
    std::vector<int32_t> all(static_cast<std::size_t>(count));
    std::iota(all.begin(), all.end(), 0);
    const std::vector<int32_t> order = brio_hilbert_order(work_points_.data(), 3, all);
    const int32_t status = triangulation_->build(order, alive_);
    if (status != PHX_MC_OK) return fail(status, kPlcNonrepresentable, kPlcNoEntity, -1);
    TetrahedralComplex& complex = triangulation_->complex();
    complex.enable_labels();
    vertex_tet_.assign(static_cast<std::size_t>(count), -1);
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
      for (int32_t vertex : tet.v) vertex_tet_[static_cast<std::size_t>(vertex)] = static_cast<int32_t>(t);
    }
    star_marks_.assign(complex.tets.size(), 0U);
    return PHX_MC_OK;
  }

  int32_t insert_source_vertex_incrementally(int32_t vertex) {
    std::chrono::steady_clock::time_point incremental_started;
    if (progress_enabled_) {
      ++progress_incremental_attempts_;
      incremental_started = std::chrono::steady_clock::now();
    }
    incremental_triangulation_ready_ = false;
    if (!triangulation_ || vertex < 0 ||
        static_cast<std::size_t>(vertex) + 1 != candidate_.points.size() / 3) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const std::size_t old_source_points = static_cast<std::size_t>(vertex);
    const std::size_t old_work_points = work_points_.size() / 3;
    const std::size_t prefix = old_source_points + static_cast<std::size_t>(corner_count_);
    if (old_work_points != prefix + volume_count_ ||
        alive_.size() != old_work_points || vertex_tet_.size() != old_work_points) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const std::size_t new_work_points = old_work_points + 1;
    std::vector<double> next_work_points;
    next_work_points.reserve(3 * new_work_points);
    next_work_points.insert(next_work_points.end(), candidate_.points.begin(),
                            candidate_.points.end());
    next_work_points.insert(next_work_points.end(), corner_points_.begin(),
                            corner_points_.end());
    next_work_points.insert(
        next_work_points.end(),
        work_points_.begin() + static_cast<std::ptrdiff_t>(3 * prefix),
        work_points_.end());
    if (next_work_points.size() != 3 * new_work_points) {
      return PHX_MC_INTERNAL_ERROR;
    }
    std::vector<int32_t> next_vertex_tet(new_work_points, -1);
    alive_.reserve(new_work_points);
    if (!triangulation_->insert_point_slot(old_source_points, old_work_points)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    TetrahedralComplex& complex = triangulation_->complex();
    work_points_.swap(next_work_points);
    alive_.insert(
        alive_.begin() +
            static_cast<std::vector<char>::difference_type>(old_source_points),
        static_cast<char>(1));
    triangulation_->rebind(
        work_points_.data(), static_cast<int64_t>(new_work_points));
    if (progress_enabled_) {
      std::fprintf(stderr,
                   "PLC_PROGRESS incremental_insert_begin vertex=%d work_points=%zu\n",
                   vertex, old_work_points);
    }
    const int32_t insert_status = triangulation_->insert(vertex, alive_);
    if (insert_status != PHX_MC_OK) {
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS incremental_insert_result vertex=%d accepted=0 "
                     "status=%d seconds=%.3f\n",
                     vertex, insert_status,
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - incremental_started)
                         .count());
      }
      if (trace_enabled_) {
        std::fprintf(stderr,
                     "PLC_TRACE incremental_insert vertex=%d status=%d fallback=1\n",
                     vertex, insert_status);
      }
      // Checked insertion refuses, leaving the complex unchanged, when the
      // conflict cavity is not star-shaped from the vertex (the complex is no
      // longer Delaunay after local recovery edits) or crosses a constrained
      // facet; only these typed refusals rebuild at the next restart.
      if (insert_status == PHX_MC_UNRESOLVED ||
          insert_status == PHX_MC_CONSTRAINT_INTERSECTION) {
        return PHX_MC_OK;
      }
      return fail(insert_status, kPlcNonrepresentable, kPlcPoint, vertex);
    }
    bool inserted = false;
    for (std::size_t row = 0; row < complex.tets.size(); ++row) {
      const Tetrahedron& tet = complex.tets[row];
      if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
      for (int32_t point : tet.v) {
        next_vertex_tet[static_cast<std::size_t>(point)] = static_cast<int32_t>(row);
        inserted = inserted || point == vertex;
      }
    }
    if (!inserted) return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, vertex);
    vertex_tet_.swap(next_vertex_tet);
    star_marks_.resize(complex.tets.size(), 0U);
    star_.clear();
    incremental_triangulation_ready_ = true;
    if (progress_enabled_) {
      ++progress_incremental_accepted_;
      std::fprintf(
          stderr,
          "PLC_PROGRESS incremental_insert vertex=%d accepted=1 status=%d "
          "seconds=%.3f\n",
          vertex, PHX_MC_OK,
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - incremental_started)
              .count());
    }
    if (trace_enabled_) {
      std::fprintf(stderr,
                   "PLC_TRACE incremental_insert vertex=%d status=%d fallback=0\n",
                   vertex, PHX_MC_OK);
    }
    return PHX_MC_OK;
  }


  bool artificial_corner_vertex(int32_t vertex) const {
    const int32_t first_corner = static_cast<int32_t>(candidate_.points.size() / 3);
    return vertex >= first_corner && vertex < first_corner + corner_count_;
  }

  bool artificial_corner_tet(const Tetrahedron& tet) const {
    for (int32_t vertex : tet.v) {
      if (artificial_corner_vertex(vertex)) return true;
    }
    return false;
  }

  int32_t has_edge(int32_t first, int32_t second, bool& found) {
    found = false;
    const TetrahedralComplex& complex = triangulation_->complex();
    if (first < 0 || second < 0 ||
        static_cast<std::size_t>(first) >= vertex_tet_.size() ||
        static_cast<std::size_t>(second) >= vertex_tet_.size() ||
        star_marks_.size() != complex.tets.size()) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
    }
    int32_t start = vertex_tet_[static_cast<std::size_t>(first)];
    if (start >= 0 && complex.live(start) &&
        !triangulation_->valid_live_cell(start)) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
    }
    if (start < 0 || !complex.live(start) ||
        vertex_slot(complex.tets[static_cast<std::size_t>(start)], first) < 0) {
      start = triangulation_->locate(ref(first));
      if (start < 0) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
      }
      vertex_tet_[static_cast<std::size_t>(first)] = start;
    }
    if (!triangulation_->valid_live_cell(start) ||
        vertex_slot(complex.tets[static_cast<std::size_t>(start)], first) < 0) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
    }
    if (++star_stamp_ == 0) {
      std::fill(star_marks_.begin(), star_marks_.end(), 0U);
      star_stamp_ = 1;
    }
    star_.clear();
    star_.push_back(start);
    star_marks_[static_cast<std::size_t>(start)] = star_stamp_;
    for (std::size_t at = 0; at < star_.size(); ++at) {
      const int32_t id = star_[at];
      if (!triangulation_->valid_live_cell(id)) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
      }
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      if (!is_ghost(tet) && !artificial_corner_tet(tet) &&
          vertex_slot(tet, second) >= 0) {
        found = true;
        return PHX_MC_OK;
      }
      const int vertex = vertex_slot(tet, first);
      if (vertex < 0) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
      }
      for (int slot = 0; slot < 4; ++slot) {
        if (slot == vertex) continue;
        const int32_t next = tet.n[slot];
        if (!triangulation_->valid_live_cell(next)) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
        }
        if (star_marks_[static_cast<std::size_t>(next)] != star_stamp_) {
          star_marks_[static_cast<std::size_t>(next)] = star_stamp_;
          star_.push_back(next);
        }
      }
    }
    return PHX_MC_OK;
  }

  // Only canonical segment rows and original polygon-loop edges are 1D source
  // entities; a non-loop triangulation diagonal is facet-supported instead.
  bool is_source_edge(std::size_t edge) const {
    if (source_segment_rows_[edge] >= 0) return true;
    const uint64_t key = topology::undirected_edge(candidate_.plc_edges[2 * edge],
                                                   candidate_.plc_edges[2 * edge + 1]);
    return polygon_loop_edges_.contains(key);
  }
  double source_edge_tolerance(std::size_t edge) const {
    double bound = std::numeric_limits<double>::infinity();
    const int32_t segment = source_segment_rows_[edge];
    if (segment >= 0) {
      bound = input_.segment_tolerances == nullptr
          ? 0.0 : input_.segment_tolerances[static_cast<std::size_t>(segment)];
    }
    for (std::size_t use = edge_triangle_offsets_[edge];
         use < edge_triangle_offsets_[edge + 1]; ++use) {
      const std::int32_t triangle = edge_triangle_rows_[use];
      const std::int32_t polygon = candidate_.input_polygons[
          static_cast<std::size_t>(triangle)];
      const std::int32_t facet = input_.polygon_facets[polygon];
      const double tolerance = input_.facet_tolerances == nullptr
          ? 0.0 : input_.facet_tolerances[static_cast<std::size_t>(facet)];
      bound = std::min(bound, tolerance);
    }
    return std::isfinite(bound) ? bound : 0.0;
  }

  // Concentric-shell split rule (Ruppert; Shewchuk, "Constrained Delaunay
  // tetrahedralizations and provably good boundary recovery", IMR 2002): a
  // chain interval ending at an acute input vertex (a shell apex) is split at
  // the distance from that apex that is the power of two in [l/3, 2l/3],
  // every other interval at its midpoint. Repeated splits toward an apex stay
  // on its power-of-two spheres, so a vertex on a shell never encroaches the
  // apex interval of another segment through the same apex: for an angle
  // theta > 0, |w - m|^2 = r^2 (5/4 - cos theta) > r^2 / 4.
  double split_parameter(std::size_t edge, const PlcEdgePoint& first,
                         const PlcEdgePoint& second) const {
    const int32_t source_first = candidate_.plc_edges[2 * edge];
    const int32_t source_second = candidate_.plc_edges[2 * edge + 1];
    const double length =
        std::sqrt(geometry::squared_distance(point(source_first), point(source_second)));
    const double span = second.parameter - first.parameter;
    const bool first_apex =
        first.vertex == source_first && shell_apex_[static_cast<std::size_t>(source_first)] != 0;
    const bool second_apex = second.vertex == source_second &&
                             shell_apex_[static_cast<std::size_t>(source_second)] != 0;
    if (first_apex == second_apex || !(length > 0.0)) return first.parameter + 0.5 * span;
    const double shell = std::exp2(std::floor(std::log2(2.0 * span * length / 3.0)));
    return first_apex ? first.parameter + shell / length : second.parameter - shell / length;
  }

  // True when a vertex other than p and q lies in the closed diametral ball
  // of pq, (w - p) . (w - q) <= 0, decided exactly on witnessed positions.
  bool encroached(int32_t p, int32_t q) const {
    const double* a = point(p);
    const double* b = point(q);
    const double radius = 0.5 * std::sqrt(geometry::squared_distance(a, b));
    double lower[3];
    double upper[3];
    for (int axis = 0; axis < 3; ++axis) {
      // Carriers of witnessed vertices are within one rounding of their
      // exact positions; the margin keeps the box conservative.
      const double margin =
          radius + 0x1p-40 * (radius + std::fabs(a[axis]) + std::fabs(b[axis]));
      lower[axis] = std::min(a[axis], b[axis]) - margin;
      upper[axis] = std::max(a[axis], b[axis]) + margin;
    }
    Expansion pa[3];
    Expansion pb[3];
    for (int axis = 0; axis < 3; ++axis) {
      pa[axis] = coordinate(ref(p), axis);
      pb[axis] = coordinate(ref(q), axis);
    }
    const int32_t count = static_cast<int32_t>(work_points_.size() / 3);
    for (int32_t vertex = 0; vertex < count; ++vertex) {
      if (vertex == p || vertex == q || alive_[static_cast<std::size_t>(vertex)] == 0) continue;
      const double* w = point(vertex);
      if (w[0] < lower[0] || w[0] > upper[0] || w[1] < lower[1] || w[1] > upper[1] ||
          w[2] < lower[2] || w[2] > upper[2]) {
        continue;
      }
      Expansion dot;
      for (int axis = 0; axis < 3; ++axis) {
        const Expansion coordinate_w = coordinate(ref(vertex), axis);
        dot = dot + (coordinate_w - pa[axis]) * (coordinate_w - pb[axis]);
      }
      if (dot.sign() <= 0) return true;
    }
    return false;
  }

  // Splits chain interval `interval` of source edge `edge` at the exactly
  // witnessed point S = A + t (B - A) of its authored segment AB, with t from
  // split_parameter. S is a dyadic rational point; its carrier is the
  // correctly rounded S, and every predicate on the new vertex evaluates S.
  int32_t split_source_edge(std::size_t edge, std::size_t interval, bool& restart) {
    const PlcEdgePoint first = edge_chains_[edge][interval];
    const PlcEdgePoint second = edge_chains_[edge][interval + 1];
    const double parameter = split_parameter(edge, first, second);
    // The interval is exhausted only when binary64 has no parameter strictly
    // inside it, i.e. after about 50 halvings of the source segment.
    if (!(first.parameter < parameter && parameter < second.parameter)) {
      return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcEdge, edge);
    }
    if (candidate_.points.size() / 3 >= static_cast<std::size_t>(kMaxMeshPoints - corner_count_)) {
      return fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
    }
    const double* corners[2] = {
        candidate_.points.data() + 3 * static_cast<std::size_t>(candidate_.plc_edges[2 * edge]),
        candidate_.points.data() +
            3 * static_cast<std::size_t>(candidate_.plc_edges[2 * edge + 1])};
    Expansion position[3];
    segment_witness_position(corners[0], corners[1], parameter, position);
    double carrier[3];
    for (int axis = 0; axis < 3; ++axis) carrier[axis] = nearest_double(position[axis]);
    if (!geometry::point_in_domain(carrier)) {
      return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcEdge, edge);
    }
    const double deviation = outward_distance(carrier, position);
    // A source point can coincide with no other vertex of a valid PLC; equal
    // carriers of distinct exact points would be indistinguishable output.
    const ExactPointRef split{carrier, deviation == 0.0 ? nullptr : position};
    for (int32_t vertex = 0; vertex < static_cast<int32_t>(candidate_.points.size() / 3);
         ++vertex) {
      if (!geometry::same_point(carrier, point(vertex))) continue;
      if (same_point(split, ref(vertex))) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcEdge, edge,
                    kPlcPoint, vertex);
      }
      return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcEdge, edge, kPlcPoint, vertex);
    }
    const int32_t vertex = static_cast<int32_t>(candidate_.points.size() / 3);
    candidate_.points.insert(candidate_.points.end(), carrier, carrier + 3);
    // Raw PlcRecovery witnesses index the carrier-edge row; Python resolves
    // it by exact endpoints to the canonical source edge identity.
    PlcRecoveryWitness witness{};
    witness.stratum = SourceStratum::kSegment;
    witness.entity = static_cast<int32_t>(edge);
    witness.parameters[0] = parameter;
    witness.deviation = deviation;
    candidate_.witnesses.push_back(witness);
    candidate_.vertex_dimension.push_back(1);
    if (deviation != 0.0) exact_points_.assign(vertex, position);
    edge_chains_[edge].insert(
        edge_chains_[edge].begin() + static_cast<std::ptrdiff_t>(interval + 1),
        {parameter, vertex});
    const int32_t insertion_status = insert_source_vertex_incrementally(vertex);
    if (insertion_status != PHX_MC_OK) return insertion_status;
    restart = true;
    return PHX_MC_OK;
  }

  struct EdgeRecoveryContactCache {
    using Key = std::pair<Triple, Triple>;
    struct Result {
      int32_t status;
      bool legal;
    };
    struct KeyHash {
      std::size_t operator()(const Key& key) const noexcept {
        std::uint64_t hash = 14695981039346656037ULL;
        const auto append = [&hash](const Triple& triangle) {
          for (int32_t vertex : triangle) {
            hash ^= static_cast<std::uint32_t>(vertex);
            hash *= 1099511628211ULL;
          }
        };
        append(key.first);
        append(key.second);
        return static_cast<std::size_t>(hash);
      }
    };
    std::unordered_map<Key, Result, KeyHash> results;
    std::size_t hits = 0;
    std::size_t misses = 0;
  };
  // Recovers every chain interval of every source edge, in row order. A
  // missing interval is first recovered without new vertices (planar flips on
  // an incident facet plane, else one edge removal); otherwise an encroached
  // interval is split by split_source_edge and the scan restarts. An interval
  // whose closed diametral ball holds no other vertex is strongly Delaunay
  // and so present in every Delaunay tetrahedralization of the vertices: it
  // can be missing only after local recovery edits, and the restart rebuilds.
  //
  // Termination. Splits happen only on encroached intervals, at the
  // concentric-shell points of split_parameter; Shewchuk (IMR 2002, segment
  // recovery with protecting shells) shows that every such split vertex has
  // insertion radius at least a constant fraction of the local feature size
  // of the PLC at that vertex, so a packing argument bounds the number of
  // split vertices. Recovery edits never remove a chain interval (edge
  // removal skips them; a planar flip removes only an edge crossing the
  // target, which no source interval of a valid PLC does), so after a
  // rebuild every non-encroached interval stays present: rebuild-only
  // restarts cannot follow each other without a split in between.
  int32_t recover_source_edges(bool& restart) {
    restart = false;
    chain_intervals_.clear();
    for (const auto& chain : edge_chains_) {
      for (std::size_t interval = 0; interval + 1 < chain.size(); ++interval) {
        chain_intervals_.insert(
            topology::undirected_edge(chain[interval].vertex, chain[interval + 1].vertex));
      }
    }
    protected_segments_.clear();
    for (const auto& chain : edge_chains_) {
      for (std::size_t interval = 0; interval + 1 < chain.size(); ++interval) {
        const int32_t first = chain[interval].vertex;
        const int32_t second = chain[interval + 1].vertex;
        bool edge_present = false;
        const int32_t edge_status = has_edge(first, second, edge_present);
        if (edge_status != PHX_MC_OK) return edge_status;
        if (edge_present) protected_segments_.push_back({first, second});
      }
    }
    for (std::size_t edge = 0; edge < edge_chains_.size(); ++edge) {
      active_source_edge_ = edge;
      for (std::size_t interval = 0; interval + 1 < edge_chains_[edge].size(); ++interval) {
        const int32_t first = edge_chains_[edge][interval].vertex;
        const int32_t second = edge_chains_[edge][interval + 1].vertex;
        bool edge_present = false;
        const int32_t edge_status = has_edge(first, second, edge_present);
        if (edge_status != PHX_MC_OK) return edge_status;
        if (edge_present) continue;
        bool recovered = false;
        const int32_t local_status = try_planar_source_edge(edge, first, second, recovered);
        if (local_status != PHX_MC_OK) return local_status;
        if (recovered) {
          protected_segments_.push_back({first, second});
          continue;
        }
        if (progress_enabled_) {
          std::fprintf(stderr,
                       "PLC_PROGRESS edge_missing row=%zu interval=%zu endpoints=(%d,%d) "
                       "points=%zu local_edits=%d\n",
                       edge, interval, first, second, candidate_.points.size() / 3,
                       local_edits_ ? 1 : 0);
        }
        if (encroached(first, second)) return split_source_edge(edge, interval, restart);
        if (!local_edits_) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
        }
        incremental_triangulation_ready_ = false;
        restart = true;
        return PHX_MC_OK;
      }
    }
    return PHX_MC_OK;
  }

  // Fixed boundaries admit no split: the local vertex cavity, planar flips,
  // and the incident authored-facet transaction are the recovery stages.
  int32_t recover_fixed_source_edge(std::size_t edge, int32_t first, int32_t second) {
    EdgeCavityScratch scratch;
    std::vector<int32_t> cavity;
    const int32_t start = finite_vertex_cell(first);
    if (start < 0) return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
    const int32_t cavity_status = collect_edge_cavity(
        triangulation_->complex(), work_points_.data(), first, second, start, scratch, cavity);
    if (cavity_status != PHX_MC_OK) {
      return fail(cavity_status, kPlcNonrepresentable, kPlcEdge, edge);
    }
    EdgeRecoveryContactCache contact_cache;
    const PlcFailure prior_failure = failure_;
    const PlcSourceRefusal prior_source_refusal = candidate_.source_refusal;
    protected_segments_.push_back({first, second});
    const int32_t recovery_status =
        recover_vertex_source_edge(edge, first, second, cavity, contact_cache, true);
    if (recovery_status == PHX_MC_OK) {
      bool edge_present = false;
      const int32_t edge_status = has_edge(first, second, edge_present);
      if (edge_status != PHX_MC_OK) return edge_status;
      if (!edge_present) return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
      return PHX_MC_OK;
    }
    if (recovery_status != PHX_MC_CONSTRAINT_INTERSECTION) return recovery_status;
    const PlcFailure recovery_failure = failure_;
    const PlcSourceRefusal recovery_refusal = candidate_.source_refusal;
    failure_ = prior_failure;
    candidate_.source_refusal = prior_source_refusal;
    bool incident_facet_recovered = false;
    const int32_t facet_status =
        recover_incident_authored_triangle_facets(edge, first, second, incident_facet_recovered);
    if (facet_status != PHX_MC_OK) return facet_status;
    if (incident_facet_recovered || cavity_steiner_restart_required_) {
      failure_ = prior_failure;
      candidate_.source_refusal = prior_source_refusal;
      return PHX_MC_OK;
    }
    bool planar_recovered = false;
    const int32_t planar_status = try_planar_source_edge(edge, first, second, planar_recovered);
    if (planar_status != PHX_MC_OK || planar_recovered) return planar_status;
    failure_ = recovery_failure;
    candidate_.source_refusal = recovery_refusal;
    return recovery_status;
  }


  struct EdgeRecoveryFace {
    Triple vertices;
    bool open;
    bool obstacle;
    int32_t source;
    bool cavity_internal;
    int32_t owner_tet;
    int32_t neighbor_tet;
    Box box;
  };

  struct EdgeRecoveryFailureDetail {
    bool has_facet_contact = false;
    bool empty_sphere_failure = false;
    Triple front_face{};
    int32_t candidate_apex = -1;
    Triple candidate_face{};
    Triple obstacle_face{};
    int8_t contact_kind = -1;
    int32_t contact_count = 0;
    std::array<ContactPoint, kMaxContactPoints> contact_points{};
    int32_t source = kNoConstraint;
    bool cavity_internal = false;
    int32_t obstacle_owner = -1;
    int32_t obstacle_neighbor = -1;
  };

  struct EdgeRecoveryCell {
    int32_t id = -1;
    std::array<int32_t, 4> vertices{};
    std::array<int32_t, 4> constraints{};
    int32_t region = kNoRegion;
  };
  struct EdgeRecoverySide {
    std::vector<EdgeRecoveryFace> faces;
    std::map<Triple, int32_t> index;
    std::vector<int32_t> vertices;
    std::vector<std::array<int32_t, 4>> tets;
    std::vector<EdgeRecoveryCell> original_cells;
  };
  struct HullPatchPlan {
    std::set<Triple> replaceable_faces;
    std::vector<Triple> old_faces;
    std::vector<int32_t> ghost_cells;
  };

  struct EdgeRecoveryDiagnosticCounters {
    std::uint64_t dfs_states = 0;
    std::uint64_t choose_calls = 0;
    std::uint64_t open_faces_scanned = 0;
    std::uint64_t face_apex_pairs_scanned = 0;
    std::uint64_t backtracks = 0;
    std::uint64_t maximum_depth = 0;
    bool dumped = false;
  };

  struct EdgeRecoverySearchFrame {
    int32_t face = -1;
    std::vector<int32_t> apices;
    EdgeRecoveryFailureDetail failure{};
    bool failure_is_targetable = false;
    bool complete = false;
  };

  static Triple oriented_key(const int32_t* vertices) {
    const FacetKey key = FacetKey::of(vertices);
    return {key.a, key.b, key.c};
  }

  static Triple reversed_key(const int32_t* vertices) {
    const FacetKey key = FacetKey::of(vertices).reversed();
    return {key.a, key.b, key.c};
  }

  bool trace_contact_details() const {
    return trace_enabled_ && active_source_edge_ == trace_edge_filter_ &&
           active_cavity_growth_round_ == trace_growth_filter_;
  }

  Box edge_recovery_box(const int32_t* vertices, int32_t count) const {
    Box box{};
    for (int axis = 0; axis < 3; ++axis) {
      box.lower[axis] = box.upper[axis] = point(vertices[0])[axis];
      for (int32_t row = 1; row < count; ++row) {
        const double coordinate = point(vertices[row])[axis];
        box.lower[axis] = std::min(box.lower[axis], coordinate);
        box.upper[axis] = std::max(box.upper[axis], coordinate);
      }
    }
    return box;
  }

  void add_edge_recovery_face(EdgeRecoverySide& side, const Triple& vertices,
                              bool obstacle, std::int32_t source = kNoConstraint,
                              bool cavity_internal = false, int32_t owner_tet = -1,
                              int32_t neighbor_tet = -1) const {
    const int32_t index = static_cast<int32_t>(side.faces.size());
    side.index.emplace(oriented_key(vertices.data()), index);
    side.faces.push_back(
        {vertices, true, obstacle, source, cavity_internal, owner_tet, neighbor_tet,
         edge_recovery_box(vertices.data(), 3)});
  }

  void reset_edge_recovery_front(EdgeRecoverySide& side, std::size_t initial_faces) const {
    while (side.faces.size() > initial_faces) {
      side.index.erase(oriented_key(side.faces.back().vertices.data()));
      side.faces.pop_back();
    }
    for (EdgeRecoveryFace& face : side.faces) face.open = true;
    side.tets.clear();
  }

  void maybe_dump_edge_recovery_cavity(
      const EdgeRecoverySide& initial_side,
      EdgeRecoveryDiagnosticCounters& counters) const {
    if (edge_recovery_dump_written_ || counters.dumped ||
        counters.dfs_states < edge_recovery_dump_threshold_) {
      return;
    }
    counters.dumped = true;
    edge_recovery_dump_written_ = true;
    std::ofstream stream(edge_recovery_dump_path_, std::ios::binary | std::ios::trunc);
    const auto write = [&](const auto& value) {
      stream.write(reinterpret_cast<const char*>(&value),
                   static_cast<std::streamsize>(sizeof(value)));
    };
    const char magic[8] = {'P', 'H', 'X', 'W', 'R', 'A', 'P', '1'};
    stream.write(magic, sizeof(magic));
    const std::uint32_t version = 1;
    const std::int32_t policy = static_cast<std::int32_t>(input_.policy);
    const std::uint64_t edge = active_source_edge_;
    const std::uint64_t growth = active_cavity_growth_round_;
    const std::int32_t recovery_first = recovery_edge_[0];
    const std::int32_t recovery_second = recovery_edge_[1];
    const std::uint64_t vertex_count = initial_side.vertices.size();
    const std::uint64_t face_count = initial_side.faces.size();
    const std::uint64_t cell_count = initial_side.original_cells.size();
    const std::uint64_t segment_count = protected_segments_.size();
    const Box cavity_box = edge_recovery_box(
        initial_side.vertices.data(),
        static_cast<std::int32_t>(initial_side.vertices.size()));
    std::uint64_t source_face_count = 0;
    for (const auto& [key, source] : source_faces_) {
      (void)source;
      if (std::all_of(key.begin(), key.end(), [&](int32_t vertex) {
            return std::binary_search(initial_side.vertices.begin(),
                                      initial_side.vertices.end(), vertex);
          })) {
        ++source_face_count;
      }
    }
    std::uint64_t relevant_segment_count = 0;
    for (const auto& segment : protected_segments_) {
      if (overlap(cavity_box, edge_recovery_box(segment.data(), 2))) {
        ++relevant_segment_count;
      }
    }
    write(version);
    write(policy);
    write(edge);
    write(growth);
    write(recovery_first);
    write(recovery_second);
    write(vertex_count);
    write(face_count);
    write(cell_count);
    write(segment_count);
    write(relevant_segment_count);
    write(source_face_count);
    write(counters.dfs_states);
    write(counters.choose_calls);
    write(counters.open_faces_scanned);
    write(counters.face_apex_pairs_scanned);
    write(counters.backtracks);
    write(counters.maximum_depth);
    for (int32_t vertex : initial_side.vertices) {
      write(vertex);
      const double* coordinates = point(vertex);
      for (int axis = 0; axis < 3; ++axis) write(coordinates[axis]);
      const int8_t dimension =
          static_cast<std::size_t>(vertex) < candidate_.vertex_dimension.size()
              ? candidate_.vertex_dimension[static_cast<std::size_t>(vertex)]
              : static_cast<int8_t>(-1);
      const PlcRecoveryWitness witness =
          static_cast<std::size_t>(vertex) < candidate_.witnesses.size()
              ? candidate_.witnesses[static_cast<std::size_t>(vertex)]
              : PlcRecoveryWitness{};
      const int8_t stratum = static_cast<int8_t>(witness.stratum);
      write(dimension);
      write(stratum);
      write(witness.entity);
      for (double parameter : witness.parameters) write(parameter);
      write(witness.deviation);
    }
    for (const EdgeRecoveryFace& face : initial_side.faces) {
      for (int32_t vertex : face.vertices) write(vertex);
      const std::uint8_t open = face.open ? 1U : 0U;
      const std::uint8_t obstacle = face.obstacle ? 1U : 0U;
      const std::uint8_t cavity_internal = face.cavity_internal ? 1U : 0U;
      write(open);
      write(obstacle);
      write(cavity_internal);
      write(face.source);
      write(face.owner_tet);
      write(face.neighbor_tet);
      const auto source = source_faces_.find(sorted_triple(face.vertices.data()));
      const std::int32_t source_triangle =
          source == source_faces_.end() ? -1 : source->second.source_triangle;
      const std::int32_t source_facet =
          source == source_faces_.end() ? -1 : source->second.facet;
      write(source_triangle);
      write(source_facet);
    }
    for (const EdgeRecoveryCell& cell : initial_side.original_cells) {
      write(cell.id);
      for (int32_t vertex : cell.vertices) write(vertex);
      for (int32_t constraint : cell.constraints) write(constraint);
      write(cell.region);
    }
    for (const auto& segment : protected_segments_) {
      if (!overlap(cavity_box, edge_recovery_box(segment.data(), 2))) continue;
      write(segment[0]);
      write(segment[1]);
    }
    for (const auto& [key, source] : source_faces_) {
      if (!std::all_of(key.begin(), key.end(), [&](int32_t vertex) {
            return std::binary_search(initial_side.vertices.begin(),
                                      initial_side.vertices.end(), vertex);
          })) {
        continue;
      }
      for (int32_t vertex : source.oriented) write(vertex);
      write(source.facet);
      write(source.source_triangle);
      const std::int32_t polygon = candidate_.input_polygons[
          static_cast<std::size_t>(source.source_triangle)];
      const std::int32_t left_region =
          input_.facet_regions == nullptr ? kNoRegion
              : input_.facet_regions[2 * static_cast<std::size_t>(source.facet)];
      const std::int32_t right_region =
          input_.facet_regions == nullptr ? kNoRegion
              : input_.facet_regions[2 * static_cast<std::size_t>(source.facet) + 1];
      write(polygon);
      write(left_region);
      write(right_region);
    }
    if (!stream) {
      std::fprintf(stderr, "PLC_WRAP_DIAGNOSTIC dump_failed path=%s\n",
                   edge_recovery_dump_path_);
      return;
    }
    std::fprintf(stderr,
                 "PLC_WRAP_DIAGNOSTIC dump=%s edge=%llu growth=%llu "
                 "dfs_states=%llu choose_calls=%llu open_faces=%llu "
                 "face_apex_pairs=%llu backtracks=%llu depth=%llu "
                 "vertices=%llu faces=%llu cells=%llu segments=%llu "
                 "relevant_segments=%llu source_faces=%llu\n",
                 edge_recovery_dump_path_,
                 static_cast<unsigned long long>(edge),
                 static_cast<unsigned long long>(growth),
                 static_cast<unsigned long long>(counters.dfs_states),
                 static_cast<unsigned long long>(counters.choose_calls),
                 static_cast<unsigned long long>(counters.open_faces_scanned),
                 static_cast<unsigned long long>(counters.face_apex_pairs_scanned),
                 static_cast<unsigned long long>(counters.backtracks),
                 static_cast<unsigned long long>(counters.maximum_depth),
                 static_cast<unsigned long long>(vertex_count),
                 static_cast<unsigned long long>(face_count),
                 static_cast<unsigned long long>(cell_count),
                 static_cast<unsigned long long>(segment_count),
                 static_cast<unsigned long long>(relevant_segment_count),
                 static_cast<unsigned long long>(source_face_count));
  }

  int32_t edge_recovery_triangle_contact(
      const Triple& first, const Triple& second, bool& legal,
      Contact* detail = nullptr,
      EdgeRecoveryContactCache* contact_cache = nullptr) const {
    const EdgeRecoveryContactCache::Key key{first, second};
    if (contact_cache != nullptr && detail == nullptr) {
      const auto found = contact_cache->results.find(key);
      if (found != contact_cache->results.end()) {
        if (progress_enabled_) ++contact_cache->hits;
        legal = found->second.legal;
        return found->second.status;
      }
      if (progress_enabled_) ++contact_cache->misses;
    }
    const double* first_points[3] = {point(first[0]), point(first[1]), point(first[2])};
    const double* second_points[3] = {point(second[0]), point(second[1]), point(second[2])};
    const int64_t first_ids[3] = {first[0], first[1], first[2]};
    const int64_t second_ids[3] = {second[0], second[1], second[2]};
    Contact contact{};
    const int32_t status = intersect_triangles(first_points, second_points, first_ids,
                                               second_ids, contact);
    legal = status == PHX_MC_OK && legal_contact(contact.kind);
    if (detail != nullptr) *detail = contact;
    if (contact_cache != nullptr && detail == nullptr && status == PHX_MC_OK) {
      contact_cache->results.emplace(key, EdgeRecoveryContactCache::Result{status, legal});
    }
    return status;
  }

  int32_t edge_recovery_segment_contact(int32_t first, int32_t second,
                                        const Triple& face, bool& legal) const {
    const double* segment[2] = {point(first), point(second)};
    const double* triangle[3] = {point(face[0]), point(face[1]), point(face[2])};
    const int64_t segment_ids[2] = {first, second};
    const int64_t triangle_ids[3] = {face[0], face[1], face[2]};
    Contact contact;
    const int32_t status = intersect_segment_triangle(segment, triangle, segment_ids,
                                                      triangle_ids, contact);
    legal = status == PHX_MC_OK && legal_contact(contact.kind);
    return status;
  }

  int32_t edge_recovery_admissible(
      const EdgeRecoverySide& side, const Triple& face, int32_t apex,
      bool& admissible, int8_t& rejection_reason,
      EdgeRecoveryFailureDetail& first_facet_contact,
      EdgeRecoveryContactCache* contact_cache = nullptr) const {
    admissible = false;
    rejection_reason = 0;
    const std::array<Triple, 3> new_faces{
        Triple{face[1], apex, face[2]}, Triple{face[0], face[2], apex},
        Triple{face[0], apex, face[1]}};
    for (const Triple& next : new_faces) {
      const auto same = side.index.find(oriented_key(next.data()));
      if (same != side.index.end()) {
        if (!side.faces[static_cast<std::size_t>(same->second)].open) {
          rejection_reason = 1;
          return PHX_MC_OK;
        }
      } else if (side.index.count(reversed_key(next.data())) != 0) {
        rejection_reason = 2;
        return PHX_MC_OK;
      }
    }
    const int32_t tet[4] = {face[0], face[1], face[2], apex};
    const Box box = edge_recovery_box(tet, 4);
    const Triple own = sorted_triple(face.data());
    bool obstructed = false;
    bool trace_contact_recorded = false;
    for (const EdgeRecoveryFace& obstacle : side.faces) {
      if (!(obstacle.open || obstacle.obstacle) || !overlap(box, obstacle.box) ||
          sorted_triple(obstacle.vertices.data()) == own) {
        continue;
      }
      const Triple other_key = sorted_triple(obstacle.vertices.data());
      for (const Triple& next : new_faces) {
        if (other_key == sorted_triple(next.data())) continue;
        bool legal = false;
        Contact contact{};
        const int32_t status = edge_recovery_triangle_contact(
            next, obstacle.vertices, legal,
            trace_enabled_ ? &contact : nullptr, contact_cache);
        if (status != PHX_MC_OK) return status;
        if (legal) continue;
        if (trace_contact_details() && !trace_contact_recorded) {
          std::fprintf(stderr,
                       "PLC_TRACE apex_contact edge=%zu growth=%zu step=%zu "
                       "front=(%d,%d,%d) apex=%d candidate=(%d,%d,%d) "
                       "obstacle=(%d,%d,%d) kind=%d source=%d internal=%d "
                       "owner=%d neighbor=%d count=%d\n",
                       active_source_edge_, active_cavity_growth_round_,
                       side.tets.size(), face[0], face[1], face[2], apex,
                       next[0], next[1], next[2], obstacle.vertices[0],
                       obstacle.vertices[1], obstacle.vertices[2],
                       static_cast<int>(contact.kind), obstacle.source,
                       obstacle.cavity_internal ? 1 : 0, obstacle.owner_tet,
                       obstacle.neighbor_tet, contact.count);
          for (int32_t index = 0; index < contact.count; ++index) {
            const ContactPoint& contact_point =
                contact.points[static_cast<std::size_t>(index)];
            std::fprintf(stderr,
                         "PLC_TRACE apex_contact_point edge=%zu growth=%zu "
                         "step=%zu front=(%d,%d,%d) apex=%d index=%d "
                         "xyz=(%.17g,%.17g,%.17g) bound=%.17g features=(%d,%d)\n",
                         active_source_edge_, active_cavity_growth_round_,
                         side.tets.size(), face[0], face[1], face[2], apex, index,
                         contact_point.x[0], contact_point.x[1],
                         contact_point.x[2], contact_point.bound,
                         static_cast<int>(contact_point.first_feature),
                         static_cast<int>(contact_point.second_feature));
          }
          trace_contact_recorded = true;
        }
        obstructed = true;
        const bool targetable =
            obstacle.source == kNoConstraint && !obstacle.cavity_internal &&
            obstacle.owner_tet >= 0 && obstacle.neighbor_tet >= 0;
        const bool has_targetable_contact =
            first_facet_contact.has_facet_contact &&
            first_facet_contact.source == kNoConstraint &&
            !first_facet_contact.cavity_internal &&
            first_facet_contact.obstacle_owner >= 0 &&
            first_facet_contact.obstacle_neighbor >= 0;
        if (!first_facet_contact.has_facet_contact ||
            (targetable && !has_targetable_contact)) {
          first_facet_contact.has_facet_contact = true;
          first_facet_contact.candidate_face = next;
          first_facet_contact.front_face = face;
          first_facet_contact.candidate_apex = apex;
          first_facet_contact.obstacle_face = obstacle.vertices;
          first_facet_contact.contact_kind = contact.kind;
          if (trace_enabled_) {
            first_facet_contact.contact_count = contact.count;
            for (int32_t index = 0; index < contact.count; ++index) {
              first_facet_contact.contact_points[static_cast<std::size_t>(index)] =
                  contact.points[index];
            }
          }
          first_facet_contact.source = obstacle.source;
          first_facet_contact.cavity_internal = obstacle.cavity_internal;
          first_facet_contact.obstacle_owner = obstacle.owner_tet;
          first_facet_contact.obstacle_neighbor = obstacle.neighbor_tet;
        }
        // Once an exact illegal contact rejects this tetrahedron and a
        // targetable external witness is selected, remaining obstacle pairs
        // cannot change either the rejection or its deterministic witness.
        if (targetable || has_targetable_contact) {
          rejection_reason = 3;
          return PHX_MC_OK;
        }
      }
    }
    if (obstructed) {
      rejection_reason = 3;
      return PHX_MC_OK;
    }
    for (const auto& segment : protected_segments_) {
      if (!overlap(box, edge_recovery_box(segment.data(), 2))) continue;
      bool legal = false;
      int32_t status = edge_recovery_segment_contact(segment[0], segment[1], face, legal);
      if (status != PHX_MC_OK) return status;
      if (!legal) {
        rejection_reason = 4;
        return PHX_MC_OK;
      }
      for (const Triple& next : new_faces) {
        status = edge_recovery_segment_contact(segment[0], segment[1], next, legal);
        if (status != PHX_MC_OK) return status;
        if (!legal) {
          rejection_reason = 4;
          return PHX_MC_OK;
        }
      }
    }
    for (int32_t vertex : side.vertices) {
      if (vertex == face[0] || vertex == face[1] || vertex == face[2] || vertex == apex) continue;
      if (orient(face[0], face[1], face[2], vertex) >= 0 &&
          orient(face[1], apex, face[2], vertex) >= 0 &&
          orient(face[0], face[2], apex, vertex) >= 0 &&
          orient(face[0], apex, face[1], vertex) >= 0) {
        rejection_reason = 5;
        return PHX_MC_OK;
      }
    }
    admissible = true;
    return PHX_MC_OK;
  }

  static bool edge_recovery_contact_targetable(
      const EdgeRecoveryFailureDetail& contact) {
    return contact.has_facet_contact && contact.source == kNoConstraint &&
           !contact.cavity_internal && contact.obstacle_owner >= 0 &&
           contact.obstacle_neighbor >= 0;
  }

  void merge_edge_recovery_failure(
      EdgeRecoverySearchFrame& frame,
      const EdgeRecoveryFailureDetail& contact) const {
    const bool targetable = edge_recovery_contact_targetable(contact);
    frame.failure.empty_sphere_failure =
        frame.failure.empty_sphere_failure || contact.empty_sphere_failure;
    if (contact.has_facet_contact &&
        (!frame.failure.has_facet_contact ||
         (targetable && !frame.failure_is_targetable))) {
      const bool empty_sphere_failure = frame.failure.empty_sphere_failure;
      frame.failure = contact;
      frame.failure.empty_sphere_failure =
          frame.failure.empty_sphere_failure || empty_sphere_failure;
      frame.failure_is_targetable = targetable;
    }
  }
  bool edge_recovery_side_complete(const EdgeRecoverySide& side,
                                  bool require_edge) const {
    if (!require_edge) return true;
    std::set<int32_t> used_vertices;
    bool recovered_edge = false;
    for (const auto& tet : side.tets) {
      bool has_first = false;
      bool has_second = false;
      for (int32_t vertex : tet) {
        used_vertices.insert(vertex);
        has_first = has_first || vertex == recovery_edge_[0];
        has_second = has_second || vertex == recovery_edge_[1];
      }
      recovered_edge = recovered_edge || (has_first && has_second);
    }
    return recovered_edge &&
           std::includes(used_vertices.begin(), used_vertices.end(),
                         side.vertices.begin(), side.vertices.end());
  }
  int32_t prepare_edge_recovery_delaunay_front(
      const EdgeRecoverySide& side,
      std::map<Triple, std::vector<int32_t>>& front) const {
    front.clear();
    if (side.vertices.size() < 4) return PHX_MC_OK;
    const std::size_t point_count = work_points_.size() / 3;
    if (point_count == 0) return PHX_MC_INTERNAL_ERROR;
    for (int32_t vertex : side.vertices) {
      if (vertex < 0 || static_cast<std::size_t>(vertex) >= point_count) {
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    // Reuse the immutable global coordinate bank and its vertex IDs; only the
    // local tetrahedral complex and liveness scratch are owned here.
    Triangulation3D local(work_points_.data(), nullptr,
                          static_cast<int64_t>(point_count));
    std::vector<char> alive(point_count, 1);
    const int32_t build_status = local.build(side.vertices, alive);
    if (build_status == PHX_MC_DEGENERATE_INPUT) return PHX_MC_OK;
    if (build_status != PHX_MC_OK) return build_status;
    const TetrahedralComplex& complex = local.complex();
    for (std::size_t row = 0; row < complex.tets.size(); ++row) {
      const int32_t cell = static_cast<int32_t>(row);
      const Tetrahedron& tet = complex.tets[row];
      if (!complex.live(cell) || is_ghost(tet)) continue;
      for (int slot = 0; slot < 4; ++slot) {
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        front[oriented_key(face.data())].push_back(tet.v[slot]);
      }
    }
    for (auto& [face, apices] : front) {
      (void)face;
      std::sort(apices.begin(), apices.end());
      apices.erase(std::unique(apices.begin(), apices.end()), apices.end());
    }
    return PHX_MC_OK;
  }


  // Select a deterministic constrained-Delaunay front without DFS. The local
  // Delaunay complex is built once for the cavity vertex bank; each open face
  // only checks its indexed tetrahedron apices against the exact contact
  // contract (plus a directly forced source endpoint). Rejected fronts do not
  // hide another face; exhausted sides use exact cavity closure or source-face
  // splitting. This is not a global constrained-Delaunay output certificate.
  int32_t choose_edge_recovery_front(
      const EdgeRecoverySide& side,
      const std::map<Triple, std::vector<int32_t>>& delaunay_front,
      EdgeRecoverySearchFrame& frame,
      EdgeRecoveryContactCache* contact_cache,
      EdgeRecoveryDiagnosticCounters* diagnostic_counters,
      const EdgeRecoverySide* diagnostic_snapshot) const {
    if (diagnostic_counters != nullptr) ++diagnostic_counters->choose_calls;
    frame.face = -1;
    frame.apices.clear();
    frame.failure = EdgeRecoveryFailureDetail{};
    frame.failure_is_targetable = false;
    frame.complete = false;
    bool has_open_face = false;
    Triple first_open_face{-1, -1, -1};
    EdgeRecoveryFailureDetail blocked_contact{};
    bool has_blocked_contact = false;
    EdgeRecoveryFailureDetail targetable_contact{};
    bool has_targetable_contact = false;
    Triple blocked_front{-1, -1, -1};
    const auto test_candidate =
        [&](int32_t face_index, const Triple& face, int32_t apex,
            EdgeRecoveryFailureDetail& contact, bool& admissible) -> int32_t {
      std::int8_t rejection_reason = 0;
      const int32_t status = edge_recovery_admissible(
          side, face, apex, admissible, rejection_reason, contact,
          contact_cache);
      if (status != PHX_MC_OK) return status;
      if (diagnostic_counters != nullptr) {
        ++diagnostic_counters->dfs_states;
        ++diagnostic_counters->face_apex_pairs_scanned;
        if (diagnostic_snapshot != nullptr) {
          maybe_dump_edge_recovery_cavity(*diagnostic_snapshot,
                                          *diagnostic_counters);
        }
      }
      if (trace_contact_details()) {
        std::fprintf(stderr,
                     "PLC_TRACE apex_trial edge=%zu growth=%zu step=%zu "
                     "front=(%d,%d,%d) apex=%d admissible=%d rejection=%d\n",
                     active_source_edge_, active_cavity_growth_round_,
                     side.tets.size(), face[0], face[1], face[2], apex,
                     admissible ? 1 : 0, static_cast<int>(rejection_reason));
      }
      if (admissible) {
        frame.face = face_index;
        frame.apices.assign(1, apex);
        return PHX_MC_OK;
      }
      contact.front_face = face;
      contact.candidate_apex = apex;
      if (edge_recovery_contact_targetable(contact)) {
        if (!has_targetable_contact) targetable_contact = contact;
        has_targetable_contact = true;
      } else if (!has_blocked_contact) {
        blocked_contact = contact;
        blocked_front = face;
        has_blocked_contact = true;
      }
      return PHX_MC_OK;
    };

    // The local exact Delaunay complex is built once per cavity side. Candidate
    // apices are then indexed by oriented front face, so the wrap walk never
    // repeats circumsphere tests as its finite front grows.
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t face_index = 0; face_index < side.faces.size();
           ++face_index) {
        if (!side.faces[face_index].open) continue;
        if (diagnostic_counters != nullptr) {
          ++diagnostic_counters->open_faces_scanned;
        }
        has_open_face = true;
        const Triple face = side.faces[face_index].vertices;
        if (first_open_face[0] < 0) first_open_face = face;
        const bool has_first =
            std::find(face.begin(), face.end(), recovery_edge_[0]) != face.end();
        const bool has_second =
            std::find(face.begin(), face.end(), recovery_edge_[1]) != face.end();
        const bool target_front = has_first || has_second;
        if ((pass == 0) != target_front) continue;

        const auto found = delaunay_front.find(oriented_key(face.data()));
        const std::vector<int32_t>* apices =
            found == delaunay_front.end() ? nullptr : &found->second;
        int32_t forced_apex = -1;
        bool forced_apex_tested = false;
        if (has_first != has_second) {
          const int32_t target =
              has_first ? recovery_edge_[1] : recovery_edge_[0];
          if (std::binary_search(side.vertices.begin(), side.vertices.end(),
                                 target) &&
              orient(face[0], face[1], face[2], target) > 0) {
            forced_apex = target;
            EdgeRecoveryFailureDetail contact{};
            bool admissible = false;
            const int32_t status = test_candidate(
                static_cast<int32_t>(face_index), face, target, contact,
                admissible);
            if (status != PHX_MC_OK) return status;
            forced_apex_tested = true;
            if (admissible) return PHX_MC_OK;
          }
        }
        if (apices == nullptr) continue;
        for (int32_t apex : *apices) {
          if (apex == forced_apex && forced_apex_tested) continue;
          EdgeRecoveryFailureDetail contact{};
          bool admissible = false;
          const int32_t status = test_candidate(
              static_cast<int32_t>(face_index), face, apex, contact,
              admissible);
          if (status != PHX_MC_OK) return status;
          if (admissible) return PHX_MC_OK;
        }
      }
    }

    if (!has_open_face) {
      frame.complete = true;
      return PHX_MC_OK;
    }
    if (has_targetable_contact) {
      frame.failure = targetable_contact;
      frame.failure_is_targetable = true;
      return PHX_MC_OK;
    }
    frame.failure = blocked_contact;
    frame.failure.front_face =
        has_blocked_contact ? blocked_front : first_open_face;
    frame.failure.empty_sphere_failure = true;
    return PHX_MC_OK;
  }
  void apply_edge_recovery_choice(EdgeRecoverySide& side, int32_t face_index,
                                  int32_t apex) const {
    const Triple face = side.faces[static_cast<std::size_t>(face_index)].vertices;
    side.tets.push_back({face[0], face[1], face[2], apex});
    side.faces[static_cast<std::size_t>(face_index)].open = false;
    const std::array<Triple, 3> new_faces{
        Triple{face[1], apex, face[2]}, Triple{face[0], face[2], apex},
        Triple{face[0], apex, face[1]}};
    for (const Triple& next : new_faces) {
      const auto same = side.index.find(oriented_key(next.data()));
      if (same != side.index.end()) {
        side.faces[static_cast<std::size_t>(same->second)].open = false;
      } else {
        add_edge_recovery_face(side, Triple{next[0], next[2], next[1]}, false);
      }
    }
  }

  int32_t try_split_wrap_source_face(
      const EdgeRecoveryFailureDetail& failure,
      std::span<const PlcSourceFace> source_patch) {
    if (!failure.empty_sphere_failure ||
        input_.policy != BoundaryPolicy::kConforming) {
      return PHX_MC_OK;
    }
    // Each accepted location is in the central face core, has an exact source
    // witness, and is separated from every existing vertex/feature by a
    // positive fraction of the protected local feature size. Its packing balls
    // fit in this bounded face core, so the finite packing count—not a row or
    // retry quota—bounds successive face-interior Steiner insertions.
    // The dyadic candidate lattice is derived from this face's protected
    // feature size; exact source and spacing checks in the constructor preserve
    // its packing radius after binary64 rounding.
    std::set<Triple> attempted_faces;
    const auto split = [&](const PlcSourceFace& face) -> int32_t {
      const Triple key = sorted_triple(face.oriented.data());
      if (!attempted_faces.insert(key).second) return PHX_MC_OK;
      const double local_feature_lower =
          source_face_steiner_lfs_lower_bound(face.source_triangle);
      if (!(local_feature_lower > 0.0)) return PHX_MC_OK;
      const double spacing = local_feature_lower;
      double parameters[2] = {0.0, 0.0};
      double carrier[3] = {0.0, 0.0, 0.0};
      double deviation = 0.0;
      if (!find_source_face_steiner_location(
              face, local_feature_lower, spacing, parameters, carrier,
              deviation)) {
        return PHX_MC_OK;
      }
      const auto current = source_faces_.find(key);
      if (current == source_faces_.end() ||
          current->second.source_triangle != face.source_triangle ||
          current->second.facet != face.facet) {
        return PHX_MC_OK;
      }
      const PlcFailure saved_failure = failure_;
      const PlcSourceRefusal saved_refusal = candidate_.source_refusal;
      const std::size_t old_point_count = candidate_.points.size();
      bool restart = false;
      const int32_t status = split_source_face(key, current->second, restart);
      if (status == PHX_MC_OK && restart) {
        if (progress_enabled_) {
          ++progress_face_steiner_insertions_;
          std::fprintf(
              stderr,
              "PLC_PROGRESS face_steiner source_triangle=%d facet=%d "
              "vertex=%zu lfs=%.17g spacing=%.17g total=%zu\n",
              face.source_triangle, face.facet,
              candidate_.points.size() / 3 - 1, local_feature_lower,
              spacing, progress_face_steiner_insertions_);
        }
        cavity_steiner_restart_required_ = true;
        return PHX_MC_OK;
      }
      if (candidate_.points.size() != old_point_count) return status;
      failure_ = saved_failure;
      candidate_.source_refusal = saved_refusal;
      return status == PHX_MC_UNRESOLVED ? PHX_MC_OK : status;
    };
    const auto failed_face =
        source_faces_.find(sorted_triple(failure.front_face.data()));
    if (failed_face != source_faces_.end()) {
      const int32_t status = split(failed_face->second);
      if (status != PHX_MC_OK || cavity_steiner_restart_required_) return status;
    }
    for (const PlcSourceFace& face : source_patch) {
      const int32_t status = split(face);
      if (status != PHX_MC_OK || cavity_steiner_restart_required_) return status;
    }
    return PHX_MC_OK;
  }


  int32_t wrap_edge_recovery_side(
      EdgeRecoverySide& side, bool& complete,
      EdgeRecoveryFailureDetail& first_facet_contact, bool require_edge,
      EdgeRecoveryContactCache* contact_cache = nullptr,
      bool diagnostic_capture = false,
      std::span<const PlcSourceFace> source_patch = {},
      bool allow_source_face_steiner = true) {
    complete = false;
    first_facet_contact = EdgeRecoveryFailureDetail{};
    EdgeRecoveryDiagnosticCounters diagnostic_counters;
    EdgeRecoverySide initial_side;
    EdgeRecoveryDiagnosticCounters* counters =
        diagnostic_capture ? &diagnostic_counters : nullptr;
    if (diagnostic_capture) initial_side = side;
    const EdgeRecoverySide* snapshot =
        diagnostic_capture ? &initial_side : nullptr;
    std::map<Triple, std::vector<int32_t>> delaunay_front;
    const auto front_started = std::chrono::steady_clock::now();
    const int32_t front_status =
        prepare_edge_recovery_delaunay_front(side, delaunay_front);
    if (front_status != PHX_MC_OK) return front_status;
    if (progress_enabled_) {
      std::size_t candidate_tets = 0;
      for (const auto& [face, apices] : delaunay_front) {
        (void)face;
        candidate_tets += apices.size();
      }
      std::fprintf(stderr,
                   "PLC_PROGRESS local_delaunay_front vertices=%zu faces=%zu "
                   "tet_face_candidates=%zu seconds=%.3f\n",
                   side.vertices.size(), delaunay_front.size(), candidate_tets,
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - front_started)
                       .count());
    }
    EdgeRecoverySearchFrame frame;
    EdgeRecoverySearchFrame failures;
    // Each accepted tetrahedron consumes an open oriented face and cannot
    // repeat. With n cavity vertices there are at most choose(n, 4) accepted
    // tetrahedra, so this deterministic front walk terminates without DFS.
    for (;;) {
      if (counters != nullptr) {
        counters->maximum_depth =
            std::max<std::uint64_t>(counters->maximum_depth, 1);
      }
      const int32_t status = choose_edge_recovery_front(
          side, delaunay_front, frame, contact_cache, counters, snapshot);
      if (status != PHX_MC_OK) return status;
      merge_edge_recovery_failure(failures, frame.failure);
      if (frame.complete) {
        complete = edge_recovery_side_complete(side, require_edge);
        if (!complete) first_facet_contact = failures.failure;
        return PHX_MC_OK;
      }
      if (frame.face < 0 || frame.apices.empty()) {
        if (frame.failure.empty_sphere_failure &&
            allow_source_face_steiner) {
          const int32_t steiner_status =
              try_split_wrap_source_face(frame.failure, source_patch);
          if (steiner_status != PHX_MC_OK) return steiner_status;
          if (cavity_steiner_restart_required_) return PHX_MC_OK;
        }
        first_facet_contact = frame.failure.has_facet_contact
            ? frame.failure : failures.failure;
        return PHX_MC_OK;
      }
      const int32_t face_index = frame.face;
      const int32_t apex = frame.apices.front();
      if (trace_contact_details()) {
        const Triple& face = side.faces[static_cast<std::size_t>(face_index)].vertices;
        std::fprintf(stderr,
                     "PLC_TRACE wrap_choice edge=%zu growth=%zu step=%zu "
                     "front=(%d,%d,%d) apex=%d\n",
                     active_source_edge_, active_cavity_growth_round_,
                     side.tets.size(), face[0], face[1], face[2], apex);
      }
      apply_edge_recovery_choice(side, face_index, apex);
      if (counters != nullptr) ++counters->dfs_states;
    }
  }

  void prepare_edge_recovery_side(
      const std::vector<int32_t>& cavity, EdgeRecoverySide& side,
      std::map<Triple, int32_t>& labels, bool capture_cells) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    side = EdgeRecoverySide{};
    labels.clear();
    for (int32_t id : cavity) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      if (capture_cells) {
        EdgeRecoveryCell cell;
        cell.id = id;
        std::copy_n(tet.v, 4, cell.vertices.begin());
        for (int slot = 0; slot < 4; ++slot) {
          cell.constraints[static_cast<std::size_t>(slot)] =
              complex.constraint(id, slot);
        }
        cell.region = complex.region(id);
        side.original_cells.push_back(cell);
      }
      side.vertices.insert(side.vertices.end(), tet.v, tet.v + 4);
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t neighbor = tet.n[slot];
        const bool inside = std::binary_search(cavity.begin(), cavity.end(), neighbor);
        const int32_t constraint = complex.constraint(id, slot);
        if (inside && (constraint == kNoConstraint || neighbor < id)) continue;
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        add_edge_recovery_face(side, face, true, constraint, inside, id, neighbor);
        if (constraint != kNoConstraint) labels.emplace(sorted_triple(face.data()), constraint);
        if (inside) {
          add_edge_recovery_face(
              side, Triple{face[0], face[2], face[1]}, true, constraint, true, neighbor, id);
        }
      }
    }
    std::sort(side.vertices.begin(), side.vertices.end());
    side.vertices.erase(std::unique(side.vertices.begin(), side.vertices.end()), side.vertices.end());
  }

  bool grow_edge_recovery_cavity(std::vector<int32_t>& cavity) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    std::vector<int32_t> added;
    for (int32_t id : cavity) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t neighbor = tet.n[slot];
        if (!complex.live(neighbor) ||
            is_ghost(complex.tets[static_cast<std::size_t>(neighbor)]) ||
            complex.constraint(id, slot) != kNoConstraint ||
            std::binary_search(cavity.begin(), cavity.end(), neighbor)) {
          continue;
        }
        added.push_back(neighbor);
      }
    }
    if (added.empty()) return false;
    std::sort(added.begin(), added.end());
    added.erase(std::unique(added.begin(), added.end()), added.end());
    std::vector<int32_t> merged;
    merged.reserve(cavity.size() + added.size());
    std::set_union(cavity.begin(), cavity.end(), added.begin(), added.end(),
                   std::back_inserter(merged));
    cavity.swap(merged);
    return true;
  }

  // Close the exact unconstrained finite-cell component crossed by one rejected
  // candidate face, rather than rebuilding the wrap after each neighbor.
  int32_t grow_edge_recovery_cavity(
      const EdgeRecoverySide& side, std::vector<int32_t>& cavity,
      const EdgeRecoveryFailureDetail& rejection, bool& grew) {
    grew = false;
    const TetrahedralComplex& complex = triangulation_->complex();
    const int32_t owner = rejection.obstacle_owner;
    const int32_t neighbor = rejection.obstacle_neighbor;
    if (!rejection.has_facet_contact || rejection.source != kNoConstraint ||
        rejection.cavity_internal || owner < 0 || neighbor < 0 ||
        !complex.live(owner) || !complex.live(neighbor) ||
        is_ghost(complex.tets[static_cast<std::size_t>(owner)]) ||
        is_ghost(complex.tets[static_cast<std::size_t>(neighbor)]) ||
        !std::binary_search(cavity.begin(), cavity.end(), owner) ||
        std::binary_search(cavity.begin(), cavity.end(), neighbor)) {
      return PHX_MC_OK;
    }
    if (star_marks_.size() != complex.tets.size()) {
      star_marks_.resize(complex.tets.size(), 0U);
    }
    if (++star_stamp_ == 0) {
      std::fill(star_marks_.begin(), star_marks_.end(), 0U);
      star_stamp_ = 1;
    }
    star_.clear();
    for (int32_t id : cavity) {
      star_marks_[static_cast<std::size_t>(id)] = star_stamp_;
    }
    const auto enqueue = [&](int32_t id) {
      std::uint32_t& mark = star_marks_[static_cast<std::size_t>(id)];
      if (mark == star_stamp_) return;
      mark = star_stamp_;
      star_.push_back(id);
    };
    const Triple& candidate = rejection.candidate_face;
    for (const EdgeRecoveryFace& boundary : side.faces) {
      const int32_t face_owner = boundary.owner_tet;
      const int32_t face_neighbor = boundary.neighbor_tet;
      if (!boundary.obstacle || boundary.cavity_internal ||
          boundary.source != kNoConstraint || face_owner < 0 || face_neighbor < 0 ||
          !complex.live(face_owner) || !complex.live(face_neighbor) ||
          is_ghost(complex.tets[static_cast<std::size_t>(face_owner)]) ||
          is_ghost(complex.tets[static_cast<std::size_t>(face_neighbor)]) ||
          !std::binary_search(cavity.begin(), cavity.end(), face_owner) ||
          std::binary_search(cavity.begin(), cavity.end(), face_neighbor)) {
        continue;
      }
      bool legal = false;
      const int32_t status =
          edge_recovery_triangle_contact(candidate, boundary.vertices, legal);
      if (status != PHX_MC_OK) return status;
      if (!legal) enqueue(face_neighbor);
    }
    for (std::size_t at = 0; at < star_.size(); ++at) {
      const int32_t id = star_[at];
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t next = tet.n[slot];
        if (!complex.live(next)) return PHX_MC_INTERNAL_ERROR;
        if (complex.constraint(id, slot) != kNoConstraint ||
            is_ghost(complex.tets[static_cast<std::size_t>(next)]) ||
            star_marks_[static_cast<std::size_t>(next)] == star_stamp_) {
          continue;
        }
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        bool legal = false;
        const int32_t status =
            edge_recovery_triangle_contact(candidate, face, legal);
        if (status != PHX_MC_OK) return status;
        if (!legal) enqueue(next);
      }
    }
    if (star_.empty()) return PHX_MC_OK;
    std::sort(star_.begin(), star_.end());
    star_.erase(std::unique(star_.begin(), star_.end()), star_.end());
    std::vector<int32_t> merged;
    merged.reserve(cavity.size() + star_.size());
    std::set_union(cavity.begin(), cavity.end(), star_.begin(), star_.end(),
                   std::back_inserter(merged));
    cavity.swap(merged);
    grew = true;
    return PHX_MC_OK;
  }

  int32_t recover_vertex_source_edge(std::size_t edge, int32_t first,
                                     int32_t second,
                                     std::vector<int32_t>& cavity,
                                     EdgeRecoveryContactCache& contact_cache,
                                     bool allow_contact_closure) {
    if (cavity.empty()) {
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, edge);
    }
    std::sort(cavity.begin(), cavity.end());
    cavity.erase(std::unique(cavity.begin(), cavity.end()), cavity.end());
    recovery_edge_ = {first, second};
    const bool diagnostic_capture =
        allow_contact_closure && !edge_recovery_dump_written_ &&
        edge_recovery_dump_path_ != nullptr && *edge_recovery_dump_path_ != '\0' &&
        active_source_edge_ == edge_recovery_dump_edge_filter_;
    EdgeRecoverySide side;
    std::map<Triple, int32_t> labels;
    EdgeRecoveryFailureDetail first_facet_contact{};
    std::size_t cavity_growth_round = 0;
    for (;;) {
      active_cavity_growth_round_ = cavity_growth_round;
      first_facet_contact = EdgeRecoveryFailureDetail{};
      prepare_edge_recovery_side(cavity, side, labels, diagnostic_capture);
      const std::size_t initial_faces = side.faces.size();
      if (trace_enabled_) {
        std::fprintf(stderr,
                     "PLC_TRACE cavity_wrap_begin edge=%zu growth=%zu cells=%zu "
                     "faces=%zu vertices=%zu\n",
                     edge, cavity_growth_round, cavity.size(), side.faces.size(),
                     side.vertices.size());
      }
      bool complete = false;
      const int32_t wrap_status = wrap_edge_recovery_side(
          side, complete, first_facet_contact, true, &contact_cache,
          diagnostic_capture);
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS contact_cache edge=%zu growth=%zu hits=%zu "
                     "misses=%zu entries=%zu\n",
                     edge, cavity_growth_round, contact_cache.hits,
                     contact_cache.misses, contact_cache.results.size());
      }
      if (trace_enabled_) {
        std::fprintf(stderr,
                     "PLC_TRACE cavity_wrap_end edge=%zu growth=%zu status=%d "
                     "complete=%d tets=%zu obstacle=%d owner=%d neighbor=%d source=%d\n",
                     edge, cavity_growth_round, wrap_status, complete ? 1 : 0,
                     side.tets.size(), first_facet_contact.has_facet_contact ? 1 : 0,
                     first_facet_contact.obstacle_owner,
                     first_facet_contact.obstacle_neighbor,
                     first_facet_contact.source);
      }
      if (trace_enabled_ && first_facet_contact.has_facet_contact) {
        const Triple& candidate = first_facet_contact.candidate_face;
        const Triple& obstacle = first_facet_contact.obstacle_face;
        const Triple& front_face = first_facet_contact.front_face;
        const int32_t apex = first_facet_contact.candidate_apex;
        const double* c0 = point(candidate[0]);
        const double* c1 = point(candidate[1]);
        const double* c2 = point(candidate[2]);
        const double* o0 = point(obstacle[0]);
        const double* o1 = point(obstacle[1]);
        const double* o2 = point(obstacle[2]);
        std::fprintf(stderr,
                     "PLC_TRACE contact_geometry edge=%zu growth=%zu kind=%d "
                     "front_face=(%d,%d,%d) apex=%d candidate=(%d,%d,%d) "
                     "obstacle=(%d,%d,%d) "
                     "candidate_xyz=[(%.17g,%.17g,%.17g),(%.17g,%.17g,%.17g),"
                     "(%.17g,%.17g,%.17g)] "
                     "obstacle_xyz=[(%.17g,%.17g,%.17g),(%.17g,%.17g,%.17g),"
                     "(%.17g,%.17g,%.17g)] count=%d\n",
                     edge, cavity_growth_round,
                     static_cast<int>(first_facet_contact.contact_kind),
                     front_face[0], front_face[1], front_face[2], apex,
                     candidate[0], candidate[1], candidate[2], obstacle[0],
                     obstacle[1], obstacle[2], c0[0], c0[1], c0[2], c1[0],
                     c1[1], c1[2], c2[0], c2[1], c2[2], o0[0], o0[1], o0[2],
                     o1[0], o1[1], o1[2], o2[0], o2[1], o2[2],
                     first_facet_contact.contact_count);
        for (int32_t index = 0;
             index < first_facet_contact.contact_count; ++index) {
          const ContactPoint& contact =
              first_facet_contact.contact_points[static_cast<std::size_t>(index)];
          std::fprintf(stderr,
                       "PLC_TRACE contact_point edge=%zu growth=%zu point=%d "
                       "xyz=(%.17g,%.17g,%.17g) bound=%.17g features=(%d,%d)\n",
                       edge, cavity_growth_round, index, contact.x[0], contact.x[1],
                       contact.x[2], contact.bound,
                       static_cast<int>(contact.first_feature),
                       static_cast<int>(contact.second_feature));
        }
      }
      if (wrap_status != PHX_MC_OK) {
        recovery_edge_ = {-1, -1};
        return fail(wrap_status, kPlcNonrepresentable, kPlcEdge, edge);
      }
      if (complete) break;
      if (!allow_contact_closure) {
        recovery_edge_ = {-1, -1};
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, edge);
      }
      reset_edge_recovery_front(side, initial_faces);
      bool grew = false;
      const int32_t growth_status =
          grow_edge_recovery_cavity(side, cavity, first_facet_contact, grew);
      if (growth_status != PHX_MC_OK) {
        recovery_edge_ = {-1, -1};
        return fail(growth_status, kPlcNonrepresentable, kPlcEdge, edge);
      }
      if (trace_enabled_) {
        std::fprintf(stderr,
                     "PLC_TRACE cavity_growth edge=%zu round=%zu grew=%d cells=%zu "
                     "obstacle=%d owner=%d neighbor=%d source=%d\n",
                     edge, cavity_growth_round, grew ? 1 : 0, cavity.size(),
                     first_facet_contact.has_facet_contact ? 1 : 0,
                     first_facet_contact.obstacle_owner,
                     first_facet_contact.obstacle_neighbor,
                     first_facet_contact.source);
      }
      if (!grew) {
        recovery_edge_ = {-1, -1};
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, edge);
      }
      ++cavity_growth_round;
    }
    recovery_edge_ = {-1, -1};
    TetrahedralComplex& complex = triangulation_->complex();
    CavityEdit edit(complex);
    edit.begin();
    for (int32_t id : cavity) {
      if (edit.remove(id) != EditStatus::kOk) {
        edit.rollback();
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
      }
    }
    for (const auto& tet : side.tets) {
      int32_t constraints[4] = {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint};
      for (int slot = 0; slot < 4; ++slot) {
        int32_t face[3];
        oriented_facet(tet.data(), slot, face);
        const auto label = labels.find(sorted_triple(face));
        if (label != labels.end()) constraints[slot] = label->second;
      }
      edit.add(tet.data(), constraints, kNoRegion);
    }
    EditStatus edit_status = edit.validate([&](const int32_t* vertices) {
      return orient(vertices[0], vertices[1], vertices[2], vertices[3]) > 0;
    });
    if (edit_status == EditStatus::kOk) edit_status = edit.commit();
    local_edits_ = local_edits_ || edit_status == EditStatus::kOk;
    if (edit_status != EditStatus::kOk) {
      edit.rollback();
      if (edit_status == EditStatus::kSlotLimit) {
        return fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
      }
      if (edit_status == EditStatus::kInternal) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
      }
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedSegment, kPlcEdge, edge);
    }
    star_marks_.resize(complex.tets.size(), 0U);
    for (int32_t id : edit.created()) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      for (int32_t vertex : tet.v) {
        if (vertex >= 0) vertex_tet_[static_cast<std::size_t>(vertex)] = id;
      }
    }
    for (int32_t vertex : side.vertices) {
      const int32_t incident = vertex_tet_[static_cast<std::size_t>(vertex)];
      if (!complex.live(incident) ||
          vertex_slot(complex.tets[static_cast<std::size_t>(incident)], vertex) < 0) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, vertex);
      }
    }
    if (edit.created().empty() ||
        !triangulation_->set_hint(edit.created().front())) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
    }
    return PHX_MC_OK;
  }

  // Planar recovery of a source edge ab lying on the plane H of an incident
  // authored triangle. It applies when the mesh triangles in H already cover
  // ab, so ab is a missing edge of that planar triangulation. Every step
  // flips one in-plane edge crossing ab whose quadrilateral is strictly convex
  // and whose new diagonal does not cross ab; such an edge exists while ab is
  // missing (Hanke, Ottmann, and Schuierer, J.UCS 2(8), 1996). The crossing
  // count therefore decreases by exactly one per flip, so recovery ends after
  // the initial crossing count of flips. Each flip replaces the edge star on
  // both sides of H by a cone from an off-plane link vertex (the ghost vertex
  // on the hull side); no vertex is created and no other edge is removed.
  struct PlanarFrame {
    std::array<int32_t, 3> plane{};
    int axis = -1;
  };
  struct PlanarCrossing {
    int32_t left;
    int32_t right;
    int32_t before;
    int32_t after;
  };
  enum class PlanarWalk { kCovered, kUncovered, kVertexOnEdge };

  int planar_side(const PlanarFrame& frame, int32_t vertex) const {
    return orient(frame.plane[0], frame.plane[1], frame.plane[2], vertex);
  }

  int planar_orient(const PlanarFrame& frame, int32_t a, int32_t b, int32_t c) const {
    return orient2d_projected(ref(a), ref(b), ref(c), frame.axis);
  }

  bool in_plane(const PlanarFrame& frame, int32_t vertex) const {
    return vertex >= 0 && !artificial_corner_vertex(vertex) &&
           planar_side(frame, vertex) == 0;
  }

  // A live finite cell incident to `vertex`, refreshing the cached hint.
  int32_t finite_vertex_cell(int32_t vertex) {
    const TetrahedralComplex& complex = triangulation_->complex();
    const int32_t cached = vertex_tet_[static_cast<std::size_t>(vertex)];
    if (complex.live(cached) && !is_ghost(complex.tets[static_cast<std::size_t>(cached)]) &&
        vertex_slot(complex.tets[static_cast<std::size_t>(cached)], vertex) >= 0) {
      return cached;
    }
    std::vector<int32_t> cells;
    if (vertex_cells(vertex, cells) != PHX_MC_OK) return -1;
    for (int32_t cell : cells) {
      if (is_ghost(complex.tets[static_cast<std::size_t>(cell)])) continue;
      vertex_tet_[static_cast<std::size_t>(vertex)] = cell;
      return cell;
    }
    return -1;
  }

  int32_t vertex_cells(int32_t vertex, std::vector<int32_t>& cells) {
    cells.clear();
    const TetrahedralComplex& complex = triangulation_->complex();
    int32_t start = vertex_tet_[static_cast<std::size_t>(vertex)];
    if (start < 0 || !complex.live(start) ||
        vertex_slot(complex.tets[static_cast<std::size_t>(start)], vertex) < 0) {
      start = triangulation_->locate(ref(vertex));
      if (start < 0 || !complex.live(start) ||
          vertex_slot(complex.tets[static_cast<std::size_t>(start)], vertex) < 0) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, vertex);
      }
      vertex_tet_[static_cast<std::size_t>(vertex)] = start;
    }
    if (star_marks_.size() != complex.tets.size()) {
      star_marks_.resize(complex.tets.size(), 0U);
    }
    if (++star_stamp_ == 0) {
      std::fill(star_marks_.begin(), star_marks_.end(), 0U);
      star_stamp_ = 1;
    }
    cells.push_back(start);
    star_marks_[static_cast<std::size_t>(start)] = star_stamp_;
    for (std::size_t at = 0; at < cells.size(); ++at) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cells[at])];
      const int own = vertex_slot(tet, vertex);
      for (int slot = 0; slot < 4; ++slot) {
        if (slot == own) continue;
        const int32_t next = tet.n[slot];
        if (!complex.live(next)) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, vertex);
        }
        if (star_marks_[static_cast<std::size_t>(next)] == star_stamp_) continue;
        star_marks_[static_cast<std::size_t>(next)] = star_stamp_;
        cells.push_back(next);
      }
    }
    return PHX_MC_OK;
  }

  int32_t edge_cells(int32_t first, int32_t second, std::vector<int32_t>& cells) {
    std::vector<int32_t> around;
    const int32_t status = vertex_cells(first, around);
    if (status != PHX_MC_OK) return status;
    cells.clear();
    for (int32_t cell : around) {
      if (vertex_slot(triangulation_->complex().tets[static_cast<std::size_t>(cell)],
                      second) >= 0) {
        cells.push_back(cell);
      }
    }
    return PHX_MC_OK;
  }

  // Third vertices of the mesh triangles in H that contain edge (first, second).
  int32_t planar_apices(const PlanarFrame& frame, int32_t first, int32_t second,
                        std::vector<int32_t>& apices) {
    std::vector<int32_t> cells;
    const int32_t status = edge_cells(first, second, cells);
    if (status != PHX_MC_OK) return status;
    apices.clear();
    for (int32_t cell : cells) {
      for (int32_t vertex : triangulation_->complex().tets[static_cast<std::size_t>(cell)].v) {
        if (vertex == first || vertex == second || !in_plane(frame, vertex)) continue;
        if (std::find(apices.begin(), apices.end(), vertex) == apices.end()) {
          apices.push_back(vertex);
        }
      }
    }
    return PHX_MC_OK;
  }

  bool within_segment_box(int32_t a, int32_t b, int32_t vertex) const {
    for (int axis = 0; axis < 3; ++axis) {
      const double low = std::min(point(a)[axis], point(b)[axis]);
      const double high = std::max(point(a)[axis], point(b)[axis]);
      if (point(vertex)[axis] < low || point(vertex)[axis] > high) return false;
    }
    return true;
  }

  // Walks the in-plane triangles from a to b and records each crossed edge
  // with the triangle apices before and after it.
  int32_t planar_crossings(const PlanarFrame& frame, int32_t a, int32_t b,
                           std::vector<PlanarCrossing>& crossings, PlanarWalk& walk,
                           int32_t& blocking_vertex) {
    crossings.clear();
    walk = PlanarWalk::kUncovered;
    blocking_vertex = -1;
    bool present = false;
    int32_t status = has_edge(a, b, present);
    if (status != PHX_MC_OK) return status;
    if (present) {
      walk = PlanarWalk::kCovered;
      return PHX_MC_OK;
    }
    std::vector<int32_t> cells;
    status = vertex_cells(a, cells);
    if (status != PHX_MC_OK) return status;
    int32_t left = -1;
    int32_t right = -1;
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t cell : cells) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      for (int i = 0; i < 4 && left < 0; ++i) {
        for (int j = i + 1; j < 4 && left < 0; ++j) {
          const int32_t u = tet.v[i];
          const int32_t v = tet.v[j];
          if (u == a || v == a || !in_plane(frame, u) || !in_plane(frame, v)) continue;
          for (int32_t candidate : {u, v}) {
            if (planar_orient(frame, a, b, candidate) == 0 &&
                within_segment_box(a, b, candidate)) {
              walk = PlanarWalk::kVertexOnEdge;
              blocking_vertex = candidate;
              return PHX_MC_OK;
            }
          }
          const int turn = planar_orient(frame, a, u, v);
          if (turn == 0 || planar_orient(frame, a, u, b) != turn ||
              planar_orient(frame, a, b, v) != turn) {
            continue;
          }
          left = planar_orient(frame, a, b, u) > 0 ? u : v;
          right = left == u ? v : u;
        }
      }
      if (left >= 0) break;
    }
    if (left < 0) return PHX_MC_OK;
    int32_t before = a;
    std::vector<int32_t> apices;
    for (;;) {
      status = planar_apices(frame, left, right, apices);
      if (status != PHX_MC_OK) return status;
      if (apices.size() != 2 || (apices[0] != before && apices[1] != before)) {
        crossings.clear();
        return PHX_MC_OK;
      }
      const int32_t after = apices[0] == before ? apices[1] : apices[0];
      crossings.push_back({left, right, before, after});
      if (after == b) break;
      const int side = planar_orient(frame, a, b, after);
      if (side == 0) {
        crossings.clear();
        walk = PlanarWalk::kVertexOnEdge;
        blocking_vertex = after;
        return PHX_MC_OK;
      }
      if (side > 0) {
        before = left;
        left = after;
      } else {
        before = right;
        right = after;
      }
    }
    walk = PlanarWalk::kCovered;
    return PHX_MC_OK;
  }

  // Ghost fan of a hull flip: the two hull faces through (left, right) become
  // (head, after, before) and (after, tail, before) in the same cyclic order.
  bool planar_ghost_fan(const std::vector<int32_t>& fan, const PlanarCrossing& crossing,
                        std::vector<std::array<int32_t, 4>>& tets) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t cell : fan) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      const int ghost = vertex_slot(tet, kGhostVertex);
      if (ghost < 0 || vertex_slot(tet, crossing.before) < 0) continue;
      Triple face{};
      oriented_facet(tet.v, ghost, face.data());
      int start = 0;
      while (start < 3 && face[static_cast<std::size_t>(start)] != crossing.left) ++start;
      if (start == 3) return false;
      const bool forward =
          face[static_cast<std::size_t>((start + 1) % 3)] == crossing.right;
      const int32_t head = forward ? crossing.left : crossing.right;
      const int32_t tail = forward ? crossing.right : crossing.left;
      tets = {{head, crossing.after, crossing.before, kGhostVertex},
              {crossing.after, tail, crossing.before, kGhostVertex}};
      return true;
    }
    return false;
  }

  static double tetrahedron_shape(const double* a, const double* b, const double* c,
                                  const double* d) {
    const double* corners[4] = {a, b, c, d};
    double squared = 0.0;
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        for (int axis = 0; axis < 3; ++axis) {
          const double delta = corners[i][axis] - corners[j][axis];
          squared += delta * delta;
        }
      }
    }
    double u[3], v[3], w[3], normal[3];
    geometry::difference(b, a, u);
    geometry::difference(c, a, v);
    geometry::difference(d, a, w);
    geometry::cross(u, v, normal);
    const double volume = std::fabs(normal[0] * w[0] + normal[1] * w[1] + normal[2] * w[2]);
    return volume / (squared * std::sqrt(squared));
  }

  // Link edges of the cells around edge (p, q).
  bool link_edges(const std::vector<int32_t>& cells, int32_t p, int32_t q,
                  std::vector<std::array<int32_t, 2>>& links) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    links.clear();
    for (int32_t cell : cells) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      std::array<int32_t, 2> link{};
      int count = 0;
      for (int32_t vertex : tet.v) {
        if (vertex != p && vertex != q) link[static_cast<std::size_t>(count++)] = vertex;
      }
      if (count != 2) return false;
      links.push_back(link);
    }
    return true;
  }

  // Walks unused link edges from `from` to `to`.
  static bool link_chain(const std::vector<std::array<int32_t, 2>>& links,
                         std::vector<char>& used, int32_t from, int32_t to,
                         std::vector<int32_t>& chain) {
    chain.assign(1, from);
    while (chain.size() == 1 || chain.back() != to) {
      bool extended = false;
      for (std::size_t row = 0; row < links.size() && !extended; ++row) {
        if (used[row] != 0) continue;
        for (int end = 0; end < 2; ++end) {
          if (links[row][static_cast<std::size_t>(end)] != chain.back()) continue;
          used[row] = 1;
          chain.push_back(links[row][static_cast<std::size_t>(1 - end)]);
          extended = true;
          break;
        }
      }
      if (!extended) return false;
    }
    return true;
  }

  // Edge removal over one link chain c0, ..., cn closed by the new edge
  // (c0, cn): every triangle t of a chain triangulation contributes the cells
  // (p, t) and (q, t). The cubic dynamic program over chain triangulations
  // maximizes the worst normalized cell volume; only exactly positive cells
  // are admitted. A cone from one link vertex is one of the enumerated
  // triangulations.
  bool triangulate_link_chain(int32_t p, int32_t q, const std::vector<int32_t>& chain,
                              std::vector<std::array<int32_t, 4>>& tets) const {
    const std::size_t n = chain.size();
    if (n < 3) return false;
    constexpr double kInvalid = -1.0;
    const double kEmpty = std::numeric_limits<double>::infinity();
    for (int sign : {1, -1}) {
      std::vector<double> score(n * n, kInvalid);
      std::vector<std::size_t> split(n * n, 0);
      for (std::size_t i = 0; i + 1 < n; ++i) score[i * n + i + 1] = kEmpty;
      for (std::size_t length = 2; length < n; ++length) {
        for (std::size_t i = 0; i + length < n; ++i) {
          const std::size_t j = i + length;
          for (std::size_t m = i + 1; m < j; ++m) {
            const double inner = std::min(score[i * n + m], score[m * n + j]);
            if (!(inner > score[i * n + j])) continue;
            const double* ci = point(chain[i]);
            const double* cm = point(chain[m]);
            const double* cj = point(chain[j]);
            if (orient(p, chain[i], chain[m], chain[j]) != sign ||
                orient(q, chain[i], chain[m], chain[j]) != -sign) {
              continue;
            }
            const double shape = std::min(tetrahedron_shape(point(p), ci, cm, cj),
                                          tetrahedron_shape(point(q), ci, cm, cj));
            const double value = std::min(inner, shape);
            if (value > score[i * n + j]) {
              score[i * n + j] = value;
              split[i * n + j] = m;
            }
          }
        }
      }
      if (!(score[n - 1] > kInvalid)) continue;
      std::vector<std::pair<std::size_t, std::size_t>> pending{{0, n - 1}};
      while (!pending.empty()) {
        const auto [i, j] = pending.back();
        pending.pop_back();
        if (j == i + 1) continue;
        const std::size_t m = split[i * n + j];
        const int32_t ci = chain[i];
        const int32_t cm = chain[m];
        const int32_t cj = chain[j];
        tets.push_back(sign > 0 ? std::array<int32_t, 4>{p, ci, cm, cj}
                                : std::array<int32_t, 4>{p, ci, cj, cm});
        tets.push_back(sign > 0 ? std::array<int32_t, 4>{q, ci, cj, cm}
                                : std::array<int32_t, 4>{q, ci, cm, cj});
        pending.push_back({i, m});
        pending.push_back({m, j});
      }
      return true;
    }
    return false;
  }

  // Finite fan of an in-plane flip: its link chain runs from before to after.
  bool planar_fan_removal(const std::vector<int32_t>& fan, const PlanarCrossing& crossing,
                          std::vector<std::array<int32_t, 4>>& tets) const {
    std::vector<std::array<int32_t, 2>> links;
    if (!link_edges(fan, crossing.left, crossing.right, links)) return false;
    std::vector<char> used(links.size(), 0);
    std::vector<int32_t> chain;
    if (!link_chain(links, used, crossing.before, crossing.after, chain) ||
        chain.size() != links.size() + 1) {
      return false;
    }
    tets.clear();
    return triangulate_link_chain(crossing.left, crossing.right, chain, tets);
  }

  // A volume Steiner point must not touch an authored facet or segment, so
  // later source recovery never meets it.
  bool clear_of_sources(const double* candidate) const {
    const auto inside_box = [&](std::initializer_list<const double*> corners) {
      for (int axis = 0; axis < 3; ++axis) {
        double low = std::numeric_limits<double>::infinity();
        double high = -low;
        for (const double* corner : corners) {
          low = std::min(low, corner[axis]);
          high = std::max(high, corner[axis]);
        }
        if (candidate[axis] < low || candidate[axis] > high) return false;
      }
      return true;
    };
    for (std::size_t row = 0; row < candidate_.input_triangles.size() / 3; ++row) {
      const int32_t* face = candidate_.input_triangles.data() + 3 * row;
      const double* corners[3] = {point(face[0]), point(face[1]), point(face[2])};
      int side = 0;
      if (inside_box({corners[0], corners[1], corners[2]}) &&
          locate_on_triangle(candidate, corners, side) != kFeatureNone) {
        return false;
      }
    }
    for (int64_t row = 0; row < input_.segment_count; ++row) {
      const double* first = point(input_.segments[2 * row]);
      const double* second = point(input_.segments[2 * row + 1]);
      if (inside_box({first, second}) && geometry::collinear(first, second, candidate, 1)) {
        return false;
      }
    }
    return true;
  }

  // Finite fan without a Steiner-free removal. The fan is star-shaped from
  // the open flipped edge: every face opposite left or right is strictly
  // visible from its interior points, and both flipped in-plane faces lie in
  // H. Points c + t (g - c), with c the edge midpoint and g the mean off-plane
  // link vertex, therefore see every face for all small t > 0. The halving
  // ends at the first exactly visible point clear of every source entity, or
  // refuses once binary64 can no longer separate the candidate from c.
  bool planar_fan_steiner(const std::vector<int32_t>& fan, const PlanarCrossing& crossing,
                          std::vector<Triple>& faces, double* location) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    faces.clear();
    double centroid[3] = {0.0, 0.0, 0.0};
    std::vector<int32_t> links;
    bool oriented = false;
    for (int32_t cell : fan) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t vertex = tet.v[slot];
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        if (vertex == crossing.left || vertex == crossing.right) {
          faces.push_back(face);
          continue;
        }
        if (vertex == crossing.before || vertex == crossing.after) continue;
        if (std::find(links.begin(), links.end(), vertex) == links.end()) {
          links.push_back(vertex);
        }
        if (oriented || vertex_slot(tet, crossing.before) < 0) continue;
        int start = 0;
        while (start < 3 && face[static_cast<std::size_t>(start)] != crossing.left) ++start;
        if (start == 3) return false;
        const bool forward =
            face[static_cast<std::size_t>((start + 1) % 3)] == crossing.right;
        const int32_t head = forward ? crossing.left : crossing.right;
        const int32_t tail = forward ? crossing.right : crossing.left;
        faces.push_back({head, crossing.after, crossing.before});
        faces.push_back({crossing.after, tail, crossing.before});
        oriented = true;
      }
    }
    if (!oriented || links.empty()) return false;
    for (int32_t vertex : links) {
      for (int axis = 0; axis < 3; ++axis) {
        centroid[axis] += point(vertex)[axis] / static_cast<double>(links.size());
      }
    }
    double midpoint[3];
    for (int axis = 0; axis < 3; ++axis) {
      midpoint[axis] = 0.5 * (point(crossing.left)[axis] + point(crossing.right)[axis]);
    }
    double previous[3] = {centroid[0], centroid[1], centroid[2]};
    for (double fraction = 0.5;; fraction *= 0.5) {
      double candidate[3];
      for (int axis = 0; axis < 3; ++axis) {
        candidate[axis] = midpoint[axis] + fraction * (centroid[axis] - midpoint[axis]);
      }
      if (geometry::same_point(candidate, previous) ||
          geometry::same_point(candidate, midpoint)) {
        return false;
      }
      std::copy_n(candidate, 3, previous);
      const bool visible = std::all_of(faces.begin(), faces.end(), [&](const Triple& face) {
        return orient3d(ref(face[0]), ref(face[1]), ref(face[2]),
                        ExactPointRef{candidate, nullptr}) > 0;
      });
      if (visible && clear_of_sources(candidate)) {
        std::copy_n(candidate, 3, location);
        return true;
      }
    }
  }

  int32_t flip_planar_edge(const PlanarFrame& frame, const PlanarCrossing& crossing,
                           bool& applied) {
    applied = false;
    std::vector<int32_t> cells;
    const int32_t status = edge_cells(crossing.left, crossing.right, cells);
    if (status != PHX_MC_OK) return status;
    TetrahedralComplex& complex = triangulation_->complex();
    // Positive, negative, and ghost fans around the flipped edge.
    std::vector<int32_t> fans[3];
    for (int32_t cell : cells) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      int side = 0;
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t vertex = tet.v[slot];
        if (vertex == crossing.left || vertex == crossing.right) continue;
        // Faces through the flipped edge are interior to its star.
        if (complex.constraint(cell, slot) != kNoConstraint) return PHX_MC_OK;
        if (vertex == kGhostVertex) continue;
        const int vertex_side = planar_side(frame, vertex);
        if (vertex_side == 0) continue;
        if (side != 0 && side != vertex_side) return PHX_MC_OK;
        side = vertex_side;
      }
      if (is_ghost(tet)) {
        if (side != 0) return PHX_MC_OK;
        fans[2].push_back(cell);
      } else if (side == 0) {
        return PHX_MC_OK;
      } else {
        fans[side > 0 ? 0 : 1].push_back(cell);
      }
    }
    std::vector<std::array<int32_t, 4>> cones[3];
    std::vector<Triple> steiner_faces[3];
    double steiner_points[3][3] = {};
    bool steiner[3] = {false, false, false};
    int nonempty = 0;
    for (int fan = 0; fan < 3; ++fan) {
      if (fans[fan].empty()) continue;
      ++nonempty;
      bool realized = fan == 2 ? planar_ghost_fan(fans[fan], crossing, cones[fan])
                               : planar_fan_removal(fans[fan], crossing, cones[fan]);
      if (!realized && fan != 2) {
        realized = planar_fan_steiner(fans[fan], crossing, steiner_faces[fan],
                                      steiner_points[fan]);
        steiner[fan] = realized;
      }
      if (!realized) return PHX_MC_OK;
    }
    if (nonempty != 2) return PHX_MC_OK;
    for (int fan = 0; fan < 2; ++fan) {
      if (!steiner[fan]) continue;
      int32_t center = -1;
      const int32_t point_status = add_volume_point(steiner_points[fan], center);
      if (point_status != PHX_MC_OK) return point_status;
      for (const Triple& face : steiner_faces[fan]) {
        cones[fan].push_back({face[0], face[1], face[2], center});
      }
    }
    std::vector<std::array<int32_t, 4>> replacement;
    for (const auto& cone : cones) replacement.insert(replacement.end(), cone.begin(), cone.end());
    return replace_cells(cells, replacement, applied);
  }

  // Validated replacement of `cells` by `tets`; an inconsistent proposal is
  // rolled back and reported as not applied.
  int32_t replace_cells(const std::vector<int32_t>& cells,
                        const std::vector<std::array<int32_t, 4>>& tets, bool& applied) {
    applied = false;
    TetrahedralComplex& complex = triangulation_->complex();
    CavityEdit edit(complex);
    edit.begin();
    for (int32_t cell : cells) {
      if (edit.remove(cell) != EditStatus::kOk) {
        edit.rollback();
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    for (const auto& tet : tets) edit.add(tet.data(), nullptr, kNoRegion);
    EditStatus edit_status = edit.validate([&](const int32_t* vertices) {
      return orient(vertices[0], vertices[1], vertices[2], vertices[3]) > 0;
    });
    if (edit_status != EditStatus::kOk) {
      edit.rollback();
      return edit_status == EditStatus::kSlotLimit ? PHX_MC_INDEX_RANGE_ERROR : PHX_MC_OK;
    }
    edit_status = edit.commit();
    local_edits_ = local_edits_ || edit_status == EditStatus::kOk;
    if (edit_status != EditStatus::kOk) {
      return edit_status == EditStatus::kSlotLimit ? PHX_MC_INDEX_RANGE_ERROR
                                                   : PHX_MC_INTERNAL_ERROR;
    }
    star_marks_.resize(complex.tets.size(), 0U);
    int32_t hint = -1;
    for (int32_t id : edit.created()) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      if (is_ghost(tet)) continue;
      if (hint < 0) hint = id;
      for (int32_t vertex : tet.v) vertex_tet_[static_cast<std::size_t>(vertex)] = id;
    }
    if (hint < 0 || !triangulation_->set_hint(hint)) return PHX_MC_INTERNAL_ERROR;
    applied = true;
    return PHX_MC_OK;
  }

  // Removes edge (u, v) so that the missing edge (a, b) appears. Both lie on
  // the closed link ring of (u, v), which they split into two chains; each is
  // triangulated by edge removal and the new chain triangulations share the
  // fins (u, a, b) and (v, b, a). This is the 3-2 flip for a ring of three.
  int32_t remove_edge_toward(int32_t u, int32_t v, int32_t a, int32_t b, bool& applied) {
    applied = false;
    // A recovered source interval is never traded for another edge.
    if (chain_intervals_.contains(topology::undirected_edge(u, v))) return PHX_MC_OK;
    std::vector<int32_t> cells;
    const int32_t status = edge_cells(u, v, cells);
    if (status != PHX_MC_OK) return status;
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t cell : cells) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      if (is_ghost(tet)) return PHX_MC_OK;
      for (int slot = 0; slot < 4; ++slot) {
        if (tet.v[slot] != u && tet.v[slot] != v &&
            complex.constraint(cell, slot) != kNoConstraint) {
          return PHX_MC_OK;
        }
      }
    }
    std::vector<std::array<int32_t, 2>> links;
    if (!link_edges(cells, u, v, links)) return PHX_MC_OK;
    std::vector<char> used(links.size(), 0);
    std::vector<int32_t> first_chain;
    std::vector<int32_t> second_chain;
    if (!link_chain(links, used, a, b, first_chain) ||
        !link_chain(links, used, b, a, second_chain) ||
        first_chain.size() + second_chain.size() != links.size() + 2) {
      return PHX_MC_OK;
    }
    std::vector<std::array<int32_t, 4>> tets;
    if (!triangulate_link_chain(u, v, first_chain, tets) ||
        !triangulate_link_chain(u, v, second_chain, tets)) {
      return PHX_MC_OK;
    }
    return replace_cells(cells, tets, applied);
  }

  // Steiner-free recovery of a missing edge ab whose crossing cells all
  // share one edge (u, v) with a and b on its link: a single edge removal
  // creates ab. Two crossing cells sharing a face with apices a and b are
  // the 2-3 flip instance (ab crosses that face, so the flip is convex).
  int32_t recover_edge_by_removal(std::size_t edge, int32_t a, int32_t b,
                                  const std::vector<int32_t>& crossing, bool& recovered) {
    recovered = false;
    if (crossing.empty()) return PHX_MC_OK;
    const TetrahedralComplex& complex = triangulation_->complex();
    if (crossing.size() == 2) {
      const Tetrahedron& first = complex.tets[static_cast<std::size_t>(crossing[0])];
      const Tetrahedron& second = complex.tets[static_cast<std::size_t>(crossing[1])];
      const int first_apex =
          vertex_slot(first, a) >= 0 ? vertex_slot(first, a) : vertex_slot(first, b);
      const int32_t other = first_apex >= 0 && first.v[first_apex] == a ? b : a;
      const int second_apex = vertex_slot(second, other);
      if (first_apex >= 0 && second_apex >= 0 && first.n[first_apex] == crossing[1] &&
          !is_ghost(first) && !is_ghost(second) &&
          complex.constraint(crossing[0], first_apex) == kNoConstraint) {
        Triple face{};
        oriented_facet(first.v, first_apex, face.data());
        std::vector<std::array<int32_t, 4>> tets;
        for (int side = 0; side < 3; ++side) {
          tets.push_back({face[static_cast<std::size_t>(side)],
                          face[static_cast<std::size_t>((side + 1) % 3)],
                          first.v[first_apex], other});
          if (orient(tets.back()[0], tets.back()[1], tets.back()[2], tets.back()[3]) < 0) {
            std::swap(tets.back()[0], tets.back()[1]);
          }
        }
        const int32_t status = replace_cells(crossing, tets, recovered);
        if (status != PHX_MC_OK || recovered) return status;
      }
    }
    std::vector<std::array<int32_t, 2>> shared;
    const Tetrahedron& seed = complex.tets[static_cast<std::size_t>(crossing.front())];
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        const int32_t u = seed.v[i];
        const int32_t v = seed.v[j];
        if (u < 0 || v < 0 || u == a || u == b || v == a || v == b) continue;
        const bool common = std::all_of(crossing.begin(), crossing.end(), [&](int32_t cell) {
          const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
          return vertex_slot(tet, u) >= 0 && vertex_slot(tet, v) >= 0;
        });
        if (common) shared.push_back({u, v});
      }
    }
    for (const auto& [u, v] : shared) {
      const int32_t status = remove_edge_toward(u, v, a, b, recovered);
      if (status != PHX_MC_OK) {
        return fail(status, kPlcNonrepresentable, kPlcEdge, static_cast<int64_t>(edge));
      }
      if (recovered) break;
    }
    if (!recovered) return PHX_MC_OK;
    bool present = false;
    const int32_t status = has_edge(a, b, present);
    if (status != PHX_MC_OK) return status;
    if (!present) return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge,
                              static_cast<int64_t>(edge));
    return PHX_MC_OK;
  }

  int32_t try_edge_removal(std::size_t edge, int32_t a, int32_t b, bool& recovered) {
    recovered = false;
    const int32_t start = finite_vertex_cell(a);
    if (start < 0) return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, a);
    EdgeCavityScratch scratch;
    std::vector<int32_t> crossing;
    const int32_t status = collect_edge_cavity(
        triangulation_->complex(), work_points_.data(), a, b, start, scratch, crossing);
    if (status != PHX_MC_OK) return PHX_MC_OK;
    std::sort(crossing.begin(), crossing.end());
    crossing.erase(std::unique(crossing.begin(), crossing.end()), crossing.end());
    return recover_edge_by_removal(edge, a, b, crossing, recovered);
  }

  // Flip recovery of an unsplittable source edge: planar flips on a covered
  // incident facet plane, else one edge removal (or 2-3 flip) around ab. An
  // unrealizable flip leaves a valid mesh, so later recovery stages may
  // continue from it; every other failure is returned unchanged.
  int32_t try_planar_source_edge(std::size_t edge, int32_t a, int32_t b,
                                 bool& recovered) {
    const PlcFailure prior = failure_;
    int32_t status = recover_planar_source_edge(edge, a, b, recovered);
    if (status == PHX_MC_CONSTRAINT_INTERSECTION && failure_.reason == kPlcFixedSegment) {
      failure_ = prior;
      status = PHX_MC_OK;
    }
    if (status != PHX_MC_OK || recovered) return status;
    return try_edge_removal(edge, a, b, recovered);
  }

  // Exact Lawson test for crossing edge (left, right) with apices before and
  // after under the lexicographic lifting h(v) = (cross(b - a, v - a)^2,
  // |2 v - a - b|^2) of the projected plane: true when the lifted after
  // vertex lies strictly below the lifted plane of (left, right, before).
  bool lifted_nonconvex(const PlanarFrame& frame, int32_t a, int32_t b,
                        const PlanarCrossing& crossing) const {
    const int x = (frame.axis + 1) % 3;
    const int y = (frame.axis + 2) % 3;
    const Expansion ax = coordinate(ref(a), x);
    const Expansion ay = coordinate(ref(a), y);
    const Expansion bx = coordinate(ref(b), x);
    const Expansion by = coordinate(ref(b), y);
    const Expansion ex = bx - ax;
    const Expansion ey = by - ay;
    const auto primary = [&](int32_t vertex) {
      const Expansion cross = ex * (coordinate(ref(vertex), y) - ay) -
                              ey * (coordinate(ref(vertex), x) - ax);
      return cross * cross;
    };
    const auto secondary = [&](int32_t vertex) {
      const Expansion dx = coordinate(ref(vertex), x).scaled(2.0) - ax - bx;
      const Expansion dy = coordinate(ref(vertex), y).scaled(2.0) - ay - by;
      return dx * dx + dy * dy;
    };
    const int32_t rows[3] = {crossing.left, crossing.right, crossing.before};
    const Expansion zx = coordinate(ref(crossing.after), x);
    const Expansion zy = coordinate(ref(crossing.after), y);
    Expansion dx[3], dy[3];
    for (int row = 0; row < 3; ++row) {
      dx[row] = coordinate(ref(rows[row]), x) - zx;
      dy[row] = coordinate(ref(rows[row]), y) - zy;
    }
    const auto lifted = [&](const auto& height) {
      const Expansion hz = height(crossing.after);
      Expansion dh[3];
      for (int row = 0; row < 3; ++row) dh[row] = height(rows[row]) - hz;
      return dx[0] * (dy[1] * dh[2] - dh[1] * dy[2]) -
             dy[0] * (dx[1] * dh[2] - dh[1] * dx[2]) +
             dh[0] * (dx[1] * dy[2] - dy[1] * dx[2]);
    };
    int sign = lifted(primary).sign();
    if (sign == 0) sign = lifted(secondary).sign();
    return sign * planar_orient(frame, crossing.left, crossing.right, crossing.before) > 0;
  }

  // Flips the planar triangulation of `frame` until ab is a mesh edge.
  // `covered` is false when the mesh triangles in the plane do not cover ab;
  // failures name the source entity (kind, entity) being recovered.
  //
  // Termination: each flip is a Lawson flip for the strictly convex
  // (lexicographic) lifting of lifted_nonconvex, so the lifted piecewise
  // linear surface strictly decreases and no triangulation repeats. While ab
  // is missing, the primary lift is zero at a and b and positive at every
  // crossing, so its restriction to ab is not convex and some crossing edge
  // is locally nonconvex; strict convexity of the lift makes that edge's
  // quadrilateral strictly convex, hence flippable. Collinear vertices are
  // admitted. The final triangulation is locally convex along ab and so
  // contains it.
  int32_t flip_to_planar_edge(const PlanarFrame& frame, int32_t a, int32_t b,
                              int32_t kind, int64_t entity, bool& covered) {
    covered = false;
    std::vector<PlanarCrossing> crossings;
    PlanarWalk walk = PlanarWalk::kUncovered;
    int32_t blocking = -1;
    int32_t status = planar_crossings(frame, a, b, crossings, walk, blocking);
    if (status != PHX_MC_OK) return status;
    while (walk == PlanarWalk::kCovered && !crossings.empty()) {
      bool applied = false;
      for (const PlanarCrossing& crossing : crossings) {
        if (!lifted_nonconvex(frame, a, b, crossing)) continue;
        const int left_turn =
            planar_orient(frame, crossing.before, crossing.after, crossing.left);
        const int right_turn =
            planar_orient(frame, crossing.before, crossing.after, crossing.right);
        if (left_turn == 0 || right_turn == 0 || left_turn == right_turn) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kind, entity);
        }
        status = flip_planar_edge(frame, crossing, applied);
        if (status != PHX_MC_OK) return fail(status, kPlcNonrepresentable, kind, entity);
        if (applied) break;
      }
      if (!applied) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION,
                    kind == kPlcEdge ? kPlcFixedSegment : kPlcFixedFacet, kind, entity);
      }
      status = planar_crossings(frame, a, b, crossings, walk, blocking);
      if (status != PHX_MC_OK) return status;
    }
    if (walk == PlanarWalk::kVertexOnEdge) {
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kind, entity,
                  kPlcPoint, blocking);
    }
    if (walk != PlanarWalk::kCovered) return PHX_MC_OK;
    bool present = false;
    status = has_edge(a, b, present);
    if (status != PHX_MC_OK) return status;
    if (!present) return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kind, entity);
    covered = true;
    return PHX_MC_OK;
  }

  int32_t recover_planar_source_edge(std::size_t edge, int32_t a, int32_t b,
                                     bool& recovered) {
    recovered = false;
    for (std::size_t use = edge_triangle_offsets_[edge];
         use < edge_triangle_offsets_[edge + 1]; ++use) {
      const int32_t* face = candidate_.input_triangles.data() +
                            3 * static_cast<std::size_t>(edge_triangle_rows_[use]);
      PlanarFrame frame;
      frame.plane = {face[0], face[1], face[2]};
      frame.axis = projection_axis(point(face[0]), point(face[1]), point(face[2]));
      if (frame.axis < 0 || !in_plane(frame, a) || !in_plane(frame, b)) continue;
      const int32_t status = flip_to_planar_edge(frame, a, b, kPlcEdge,
                                                 static_cast<int64_t>(edge), recovered);
      if (status != PHX_MC_OK || recovered) return status;
    }
    return PHX_MC_OK;
  }

  // A missing source face whose current chain subdivision lies in a plane
  // covered by mesh triangles is recovered by planar edge flips alone: once
  // every child edge is a mesh edge of that planar triangulation, every
  // child triangle is a mesh face. The parent row is then replaced by its
  // children exactly as the cavity transaction does.
  int32_t recover_planar_source_face(const Triple& key, const PlcSourceFace& source,
                                     bool& recovered) {
    recovered = false;
    std::unordered_map<int32_t, std::vector<std::pair<std::size_t, std::size_t>>>
        chain_positions;
    for (std::size_t row = 0; row < edge_chains_.size(); ++row) {
      for (std::size_t position = 0; position < edge_chains_[row].size(); ++position) {
        const int32_t vertex = edge_chains_[row][position].vertex;
        if (std::find(source.oriented.begin(), source.oriented.end(), vertex) !=
            source.oriented.end()) {
          chain_positions[vertex].push_back({row, position});
        }
      }
    }
    std::vector<PlcSourceFace> children;
    if (current_source_face_triangles(source, chain_positions, children) != PHX_MC_OK) {
      return PHX_MC_OK;
    }
    PlanarFrame frame;
    frame.plane = source.oriented;
    frame.axis = projection_axis(point(source.oriented[0]), point(source.oriented[1]),
                                 point(source.oriented[2]));
    if (frame.axis < 0) return PHX_MC_OK;
    for (const PlcSourceFace& child : children) {
      for (int side = 0; side < 3; ++side) {
        bool covered = false;
        const int32_t status = flip_to_planar_edge(
            frame, child.oriented[static_cast<std::size_t>(side)],
            child.oriented[static_cast<std::size_t>((side + 1) % 3)], kPlcFacet,
            source.facet, covered);
        if (status != PHX_MC_OK || !covered) return status;
      }
    }
    std::vector<int32_t> cells;
    for (const PlcSourceFace& child : children) {
      const int32_t status = edge_cells(child.oriented[0], child.oriented[1], cells);
      if (status != PHX_MC_OK) return status;
      const bool present = std::any_of(cells.begin(), cells.end(), [&](int32_t cell) {
        return vertex_slot(triangulation_->complex().tets[static_cast<std::size_t>(cell)],
                           child.oriented[2]) >= 0;
      });
      if (!present) return PHX_MC_OK;
    }
    source_faces_.erase(key);
    for (const PlcSourceFace& child : children) {
      source_faces_.emplace(sorted_triple(child.oriented.data()), child);
    }
    recovered = true;
    return PHX_MC_OK;
  }

  // Recover an admitted fixed facet without moving its source vertices.
  // Contact paths cross faces overlapping the source triangle AABB; around a
  // vertex contact, each incident tetrahedron also overlaps that AABB.
  int32_t collect_fixed_facet_cavity(const PlcSourceFace& source,
                                     std::vector<int32_t>& cavity) {
    cavity.clear();
    const TetrahedralComplex& complex = triangulation_->complex();
    const Box source_box = edge_recovery_box(source.oriented.data(), 3);
    const double* source_points[3] = {
        point(source.oriented[0]), point(source.oriented[1]), point(source.oriented[2])};
    const int64_t source_ids[3] = {
        source.oriented[0], source.oriented[1], source.oriented[2]};
    const auto overlaps_source_box = [&](int32_t id) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      if (tet.v[0] == kDeadVertex) return false;
      if (!is_ghost(tet)) return overlap(source_box, edge_recovery_box(tet.v, 4));
      Triple boundary{};
      int count = 0;
      for (int32_t vertex : tet.v) {
        if (vertex != kGhostVertex) boundary[static_cast<std::size_t>(count++)] = vertex;
      }
      return count == 3 && overlap(source_box, edge_recovery_box(boundary.data(), 3));
    };
    const int32_t first = source.oriented[0];
    int32_t start = vertex_tet_[static_cast<std::size_t>(first)];
    if (start < 0 || !complex.live(start) ||
        vertex_slot(complex.tets[static_cast<std::size_t>(start)], first) < 0) {
      start = triangulation_->locate(ref(first));
      if (start < 0) return PHX_MC_INTERNAL_ERROR;
      vertex_tet_[static_cast<std::size_t>(first)] = start;
    }
    if (++star_stamp_ == 0) {
      std::fill(star_marks_.begin(), star_marks_.end(), 0U);
      star_stamp_ = 1;
    }
    star_.clear();
    const auto add_candidate = [&](int32_t id) {
      if (id < 0 || star_marks_[static_cast<std::size_t>(id)] == star_stamp_ ||
          !overlaps_source_box(id)) {
        return;
      }
      star_marks_[static_cast<std::size_t>(id)] = star_stamp_;
      star_.push_back(id);
    };
    add_candidate(start);
    if (star_.empty()) return PHX_MC_INTERNAL_ERROR;
    for (std::size_t at = 0; at < star_.size(); ++at) {
      const int32_t id = star_[at];
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      if (!is_ghost(tet)) {
        for (int slot = 0; slot < 4; ++slot) {
          Triple face{};
          oriented_facet(tet.v, slot, face.data());
          if (!overlap(source_box, edge_recovery_box(face.data(), 3))) continue;
          const double* face_points[3] = {
              point(face[0]), point(face[1]), point(face[2])};
          const int64_t face_ids[3] = {face[0], face[1], face[2]};
          Contact contact;
          const int32_t status =
              intersect_triangles(source_points, face_points, source_ids, face_ids, contact);
          if (status != PHX_MC_OK) return status;
          if (!legal_contact(contact.kind)) {
            cavity.push_back(id);
            break;
          }
        }
      }
      for (int slot = 0; slot < 4; ++slot) add_candidate(tet.n[slot]);
    }
    return PHX_MC_OK;
  }

  int source_facet_side(const Triple& source, const Triple& face,
                        const double* interior) const {
    const double* a = point(source[0]);
    const double* b = point(source[1]);
    const double* c = point(source[2]);
    const int sides[3] = {
        orient3d(a, b, c, point(face[0])),
        orient3d(a, b, c, point(face[1])),
        orient3d(a, b, c, point(face[2]))};
    bool positive = false;
    bool negative = false;
    for (int side : sides) {
      positive = positive || side > 0;
      negative = negative || side < 0;
    }
    if (positive && !negative) return 1;
    if (negative && !positive) return -1;
    const Expansion centroid_side =
        orient3d_exact(a, b, c, point(face[0])) +
        orient3d_exact(a, b, c, point(face[1])) +
        orient3d_exact(a, b, c, point(face[2]));
    if (centroid_side.sign() != 0) return centroid_side.sign();
    return orient3d(a, b, c, interior);
  }

  int source_tet_side(const Triple& source, const Tetrahedron& tet) const {
    const double* a = point(source[0]);
    const double* b = point(source[1]);
    const double* c = point(source[2]);
    const Expansion center_side =
        orient3d_exact(a, b, c, point(tet.v[0])) +
        orient3d_exact(a, b, c, point(tet.v[1])) +
        orient3d_exact(a, b, c, point(tet.v[2])) +
        orient3d_exact(a, b, c, point(tet.v[3]));
    return center_side.sign();
  }

  bool add_recovery_label(std::map<Triple, int32_t>& labels,
                          const Triple& face, int32_t label) const {
    const auto [found, inserted] = labels.emplace(sorted_triple(face.data()), label);
    return inserted || found->second == label;
  }

  bool prepare_fixed_facet_sides(
      const std::vector<int32_t>& cavity,
      const std::vector<PlcSourceFace>& source_patch,
      EdgeRecoverySide& positive, EdgeRecoverySide& negative,
      std::map<Triple, int32_t>& labels,
      bool& has_positive, bool& has_negative,
      const HullPatchPlan* hull_patch, bool capture_cells) const {
    if (source_patch.empty()) return false;
    const PlcSourceFace& source = source_patch.front();
    const TetrahedralComplex& complex = triangulation_->complex();
    positive = EdgeRecoverySide{};
    negative = EdgeRecoverySide{};
    labels.clear();
    has_positive = false;
    has_negative = false;
    for (int32_t id : cavity) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      if (capture_cells) {
        EdgeRecoveryCell cell;
        cell.id = id;
        std::copy_n(tet.v, 4, cell.vertices.begin());
        for (int slot = 0; slot < 4; ++slot) {
          cell.constraints[static_cast<std::size_t>(slot)] =
              complex.constraint(id, slot);
        }
        cell.region = complex.region(id);
        positive.original_cells.push_back(cell);
        negative.original_cells.push_back(cell);
      }
      positive.vertices.insert(positive.vertices.end(), tet.v, tet.v + 4);
      negative.vertices.insert(negative.vertices.end(), tet.v, tet.v + 4);
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t neighbor = tet.n[slot];
        const bool inside = std::binary_search(cavity.begin(), cavity.end(), neighbor);
        const int32_t constraint = complex.constraint(id, slot);
        if (inside && (constraint == kNoConstraint || neighbor < id)) continue;
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        if (inside) {
          const int this_side = source_tet_side(source.oriented, tet);
          const int other_side = source_tet_side(
              source.oriented, complex.tets[static_cast<std::size_t>(neighbor)]);
          if (this_side == 0 || this_side != other_side) return false;
          EdgeRecoverySide& side = this_side > 0 ? positive : negative;
          add_edge_recovery_face(side, face, true, constraint, true);
          add_edge_recovery_face(
              side, Triple{face[0], face[2], face[1]}, true, constraint, true);
          has_positive = has_positive || this_side > 0;
          has_negative = has_negative || this_side < 0;
          if (!add_recovery_label(labels, face, constraint)) return false;
          continue;
        }
        if (hull_patch != nullptr &&
            hull_patch->replaceable_faces.contains(sorted_triple(face.data()))) {
          continue;
        }
        const int side_sign = source_facet_side(
            source.oriented, face, point(tet.v[slot]));
        if (side_sign == 0) return false;
        EdgeRecoverySide& side = side_sign > 0 ? positive : negative;
        add_edge_recovery_face(side, face, true, constraint, false);
        has_positive = has_positive || side_sign > 0;
        has_negative = has_negative || side_sign < 0;
        if (constraint != kNoConstraint &&
            !add_recovery_label(labels, face, constraint)) return false;
      }
    }
    for (EdgeRecoverySide* side : {&positive, &negative}) {
      std::sort(side->vertices.begin(), side->vertices.end());
      side->vertices.erase(std::unique(side->vertices.begin(), side->vertices.end()),
                           side->vertices.end());
    }
    const auto add_source_face = [&](EdgeRecoverySide& side,
                                    const Triple& face, int32_t facet) {
      side.vertices.insert(side.vertices.end(), face.begin(), face.end());
      const auto existing = side.index.find(oriented_key(face.data()));
      if (existing != side.index.end()) {
        const EdgeRecoveryFace& current =
            side.faces[static_cast<std::size_t>(existing->second)];
        return current.source == kNoConstraint || current.source == facet;
      }
      add_edge_recovery_face(side, face, true, facet, false);
      return true;
    };
    for (const PlcSourceFace& source_face : source_patch) {
      if (!add_source_face(positive, source_face.oriented, source_face.facet) ||
          !add_source_face(
              negative,
              Triple{source_face.oriented[0], source_face.oriented[2],
                     source_face.oriented[1]},
              source_face.facet)) {
        return false;
      }
    }
    for (EdgeRecoverySide* side : {&positive, &negative}) {
      std::sort(side->vertices.begin(), side->vertices.end());
      side->vertices.erase(
          std::unique(side->vertices.begin(), side->vertices.end()),
          side->vertices.end());
    }
    return true;
  }
  bool recovery_tetrahedra_disjoint(const std::array<int32_t, 4>& first,
                                   const std::array<int32_t, 4>& second) const {
    for (int32_t vertex : first) {
      if (std::find(second.begin(), second.end(), vertex) != second.end()) continue;
      bool inside = true;
      for (int slot = 0; slot < 4; ++slot) {
        Triple face{};
        oriented_facet(second.data(), slot, face.data());
        inside = inside && orient(face[0], face[1], face[2], vertex) >= 0;
      }
      if (inside) return false;
    }
    for (int32_t vertex : second) {
      if (std::find(first.begin(), first.end(), vertex) != first.end()) continue;
      bool inside = true;
      for (int slot = 0; slot < 4; ++slot) {
        Triple face{};
        oriented_facet(first.data(), slot, face.data());
        inside = inside && orient(face[0], face[1], face[2], vertex) >= 0;
      }
      if (inside) return false;
    }
    for (int first_slot = 0; first_slot < 4; ++first_slot) {
      Triple first_face{};
      oriented_facet(first.data(), first_slot, first_face.data());
      for (int second_slot = 0; second_slot < 4; ++second_slot) {
        Triple second_face{};
        oriented_facet(second.data(), second_slot, second_face.data());
        if (sorted_triple(first_face.data()) == sorted_triple(second_face.data())) continue;
        bool legal = false;
        const int32_t status =
            edge_recovery_triangle_contact(first_face, second_face, legal);
        if (status != PHX_MC_OK || !legal) return false;
      }
    }
    return true;
  }
  // A source-vertex fan is a finite exact fallback for star-shaped cavity
  // sides. Acceptance requires the complete oriented boundary, exact old
  // volume, positive nonoverlapping cells, and unchanged contact with every
  // source facet and protected segment; it never substitutes a plausible
  // shell for the removed finite volume.
  bool try_cone_source_facet_side(
      const EdgeRecoverySide& side, const std::vector<int32_t>& cavity,
      const PlcSourceFace& source, int side_sign,
      std::span<const PlcSourceFace> source_patch,
      std::vector<std::array<int32_t, 4>>& result) const {
    const auto cone_started = std::chrono::steady_clock::now();
    if (progress_enabled_) {
      std::fprintf(
          stderr,
          "PLC_PROGRESS source_cone_begin edge=%zu side=%d vertices=%zu "
          "faces=%zu\n",
          active_source_edge_, side_sign, side.vertices.size(),
          side.faces.size());
    }
    result.clear();
    const auto trace_cone = [&](int32_t apex, const char* stage) {
      if (trace_contact_details()) {
        std::fprintf(stderr,
                     "PLC_TRACE source_cone edge=%zu side=%d apex=%d stage=%s\n",
                     active_source_edge_, side_sign, apex, stage);
      }
    };
    if (side_sign == 0 || side.faces.empty()) {
      trace_cone(-1, "empty-side");
      return false;
    }

    std::map<Triple, FacetKey> boundary;
    std::vector<int32_t> vertices = side.vertices;
    std::set<int32_t> patch_triangles;
    for (const PlcSourceFace& face : source_patch) {
      patch_triangles.insert(face.source_triangle);
    }
    for (const EdgeRecoveryFace& face : side.faces) {
      const Triple key = sorted_triple(face.vertices.data());
      if (!face.open || face.cavity_internal ||
          !boundary.emplace(key, FacetKey::of(face.vertices.data())).second) {
        trace_cone(-1, "invalid-boundary");
        return false;
      }
      const auto source_face = source_faces_.find(key);
      if (source_face != source_faces_.end()) {
        patch_triangles.insert(source_face->second.source_triangle);
      }
      vertices.insert(vertices.end(), face.vertices.begin(),
                      face.vertices.end());
    }
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

    Expansion old_volume;
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t cell : cavity) {
      if (!complex.live(cell) ||
          is_ghost(complex.tets[static_cast<std::size_t>(cell)])) {
        trace_cone(-1, "invalid-cavity-cell");
        return false;
      }
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      const int cell_side = source_tet_side(source.oriented, tet);
      if (cell_side == 0) {
        trace_cone(-1, "mixed-side-cell");
        return false;
      }
      if (cell_side == side_sign) {
        old_volume = old_volume + orient3d_exact(
            point(tet.v[0]), point(tet.v[1]), point(tet.v[2]), point(tet.v[3]));
      }
    }
    if (old_volume.sign() <= 0) {
      trace_cone(-1, "empty-old-volume");
      return false;
    }

    std::size_t apex_candidate_count = 0;
    for (int32_t apex : vertices) {
      if (apex < 0 ||
          static_cast<std::size_t>(apex) >= candidate_.points.size() / 3) {
        continue;
      }
      ++apex_candidate_count;
      std::vector<std::array<int32_t, 4>> proposed;
      proposed.reserve(boundary.size());
      bool valid = true;
      for (const EdgeRecoveryFace& face : side.faces) {
        if (std::find(face.vertices.begin(), face.vertices.end(), apex) !=
            face.vertices.end()) {
          continue;
        }
        const Triple& on_face = face.vertices;
        if (orient(on_face[0], on_face[1], on_face[2], apex) <= 0) {
          valid = false;
          break;
        }
        proposed.push_back({on_face[0], on_face[1], on_face[2], apex});
      }
      if (!valid || proposed.empty()) {
        trace_cone(apex, "non-star-boundary");
        continue;
      }

      std::set<std::array<int32_t, 4>> unique_tets;
      std::set<int32_t> used_vertices;
      std::map<Triple, std::vector<FacetKey>> face_uses;
      Expansion new_volume;
      for (const auto& tet : proposed) {
        auto sorted_tet = tet;
        std::sort(sorted_tet.begin(), sorted_tet.end());
        if (!unique_tets.insert(sorted_tet).second ||
            orient(tet[0], tet[1], tet[2], tet[3]) <= 0) {
          valid = false;
          break;
        }
        for (int32_t vertex : tet) {
          const int sign = orient(source.oriented[0], source.oriented[1], source.oriented[2], vertex);
          if (sign != 0 && sign != side_sign) {
            valid = false;
            break;
          }
          used_vertices.insert(vertex);
        }
        if (!valid) break;
        new_volume = new_volume + orient3d_exact(
            point(tet[0]), point(tet[1]), point(tet[2]), point(tet[3]));
        for (int slot = 0; slot < 4; ++slot) {
          Triple face{};
          oriented_facet(tet.data(), slot, face.data());
          face_uses[sorted_triple(face.data())].push_back(
              FacetKey::of(face.data()));
        }
      }
      if (!valid || !std::includes(used_vertices.begin(), used_vertices.end(),
                                   vertices.begin(), vertices.end()) ||
          (new_volume - old_volume).sign() != 0) {
        trace_cone(apex, "volume-or-vertex-set");
        continue;
      }
      for (const auto& [key, boundary_orientation] : boundary) {
        const auto uses = face_uses.find(key);
        if (uses == face_uses.end() || uses->second.size() != 1 ||
            !(uses->second.front() == boundary_orientation)) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        trace_cone(apex, "boundary-face-incidence");
        continue;
      }
      for (const auto& [key, uses] : face_uses) {
        if (boundary.contains(key)) continue;
        if (uses.size() != 2 || !(uses[0] == uses[1].reversed())) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        trace_cone(apex, "interior-face-incidence");
        continue;
      }

      const char* failed_contact = "unclassified-contact";
      for (std::size_t first = 0; first < proposed.size() && valid; ++first) {
        const auto& tet = proposed[first];
        const Box tet_box = edge_recovery_box(tet.data(), 4);
        for (int32_t vertex : vertices) {
          if (std::find(tet.begin(), tet.end(), vertex) != tet.end()) continue;
          bool inside = true;
          for (int slot = 0; slot < 4; ++slot) {
            Triple face{};
            oriented_facet(tet.data(), slot, face.data());
            inside = inside &&
                orient(face[0], face[1], face[2], vertex) >= 0;
          }
          if (inside) {
            failed_contact = "unrepresented-cavity-vertex";
            valid = false;
            break;
          }
        }
        for (std::size_t second = 0; second < first && valid; ++second) {
          if (!recovery_tetrahedra_disjoint(proposed[second], tet)) {
            failed_contact = "overlapping-cone-tetrahedra";
            valid = false;
            break;
          }
        }
        for (const EdgeRecoveryFace& obstacle : side.faces) {
          if (!valid) break;
          if (!overlap(tet_box, obstacle.box)) continue;
          for (int slot = 0; slot < 4; ++slot) {
            Triple face{};
            oriented_facet(tet.data(), slot, face.data());
            if (sorted_triple(face.data()) ==
                sorted_triple(obstacle.vertices.data())) {
              continue;
            }
            bool legal = false;
            Contact contact{};
            const int32_t status = edge_recovery_triangle_contact(
                face, obstacle.vertices, legal, &contact);
            if (status != PHX_MC_OK || !legal) {
              failed_contact = "cavity-face-contact";
              if (trace_contact_details()) {
                std::fprintf(
                    stderr,
                    "PLC_TRACE source_cone_contact kind=%d face=(%d,%d,%d) "
                    "obstacle=(%d,%d,%d) source=%d owner=%d neighbor=%d\n",
                    static_cast<int>(contact.kind), face[0], face[1], face[2],
                    obstacle.vertices[0], obstacle.vertices[1],
                    obstacle.vertices[2], obstacle.source, obstacle.owner_tet,
                    obstacle.neighbor_tet);
              }
              valid = false;
              break;
            }
          }
        }
        for (const auto& segment : protected_segments_) {
          if (!valid || !overlap(tet_box, edge_recovery_box(segment.data(), 2))) {
            continue;
          }
          for (int slot = 0; slot < 4; ++slot) {
            Triple face{};
            oriented_facet(tet.data(), slot, face.data());
            bool legal = false;
            const int32_t status = edge_recovery_segment_contact(
                segment[0], segment[1], face, legal);
            if (status != PHX_MC_OK || !legal) {
              failed_contact = "protected-segment-contact";
              if (trace_contact_details()) {
                std::fprintf(
                    stderr,
                    "PLC_TRACE source_cone_segment segment=(%d,%d) "
                    "face=(%d,%d,%d)\n",
                    segment[0], segment[1], face[0], face[1], face[2]);
              }
              valid = false;
              break;
            }
          }
        }
        for (std::size_t triangle = 0;
             triangle < candidate_.input_triangles.size() / 3 && valid;
             ++triangle) {
          if (patch_triangles.contains(static_cast<int32_t>(triangle))) {
            continue;
          }
          const int32_t* source_face =
              candidate_.input_triangles.data() + 3 * triangle;
          if (!overlap(tet_box, edge_recovery_box(source_face, 3))) continue;
          for (int slot = 0; slot < 4; ++slot) {
            Triple face{};
            oriented_facet(tet.data(), slot, face.data());
            bool legal = false;
            Contact contact{};
            const int32_t status = edge_recovery_triangle_contact(
                face, Triple{source_face[0], source_face[1], source_face[2]},
                legal, &contact);
            if (status != PHX_MC_OK || !legal) {
              failed_contact = "other-source-facet-contact";
              if (trace_contact_details()) {
                std::fprintf(
                    stderr,
                    "PLC_TRACE source_cone_facet triangle=%zu kind=%d "
                    "face=(%d,%d,%d) source=(%d,%d,%d)\n",
                    triangle, static_cast<int>(contact.kind), face[0],
                    face[1], face[2], source_face[0], source_face[1],
                    source_face[2]);
              }
              valid = false;
              break;
            }
          }
        }
      }
      if (valid) {
        trace_cone(apex, "accepted");
        if (progress_enabled_) {
          std::fprintf(
              stderr,
              "PLC_PROGRESS source_cone_end edge=%zu side=%d accepted=1 "
              "apex=%d tets=%zu seconds=%.3f\n",
              active_source_edge_, side_sign, apex, proposed.size(),
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - cone_started)
                  .count());
        }
        result = std::move(proposed);
        return true;
      }
      trace_cone(apex, failed_contact);
    }
    trace_cone(static_cast<int32_t>(apex_candidate_count), "no-apex-accepted");
    if (progress_enabled_) {
      std::fprintf(
          stderr,
          "PLC_PROGRESS source_cone_end edge=%zu side=%d accepted=0 "
          "candidates=%zu seconds=%.3f\n",
          active_source_edge_, side_sign, apex_candidate_count,
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - cone_started)
              .count());
    }
    return false;
  }

  // Local source-facet recovery may replace only an exact coplanar hull patch.
  // It preserves every authored triangle and relinks the ghost shell through
  // CavityEdit; it does not certify a global constrained-Delaunay result.
  bool compatible_hull_source_patch(
      const std::vector<PlcSourceFace>& source_patch) const {
    if (source_patch.empty()) return false;
    const PlcSourceFace& reference = source_patch.front();
    if (reference.facet < 0 || reference.facet >= input_.facet_count) return false;
    const int axis = projection_axis(point(reference.oriented[0]),
                                     point(reference.oriented[1]),
                                     point(reference.oriented[2]));
    if (axis < 0) return false;
    const int x = (axis + 1) % 3;
    const int y = (axis + 2) % 3;
    const double* a3 = point(reference.oriented[0]);
    const double* b3 = point(reference.oriented[1]);
    const double* c3 = point(reference.oriented[2]);
    const double a2[2] = {a3[x], a3[y]};
    const double b2[2] = {b3[x], b3[y]};
    const double c2[2] = {c3[x], c3[y]};
    const int orientation = orient2d(a2, b2, c2);
    if (orientation == 0) return false;
    const int32_t positive_region = input_.facet_regions[2 * reference.facet];
    const int32_t negative_region = input_.facet_regions[2 * reference.facet + 1];
    std::set<Triple> source_faces;
    for (const PlcSourceFace& source : source_patch) {
      if (source.facet < 0 || source.facet >= input_.facet_count ||
          input_.facet_regions[2 * source.facet] != positive_region ||
          input_.facet_regions[2 * source.facet + 1] != negative_region ||
          !source_faces.insert(sorted_triple(source.oriented.data())).second) {
        return false;
      }
      const double* p[3] = {
          point(source.oriented[0]), point(source.oriented[1]),
          point(source.oriented[2])};
      for (const double* vertex : p) {
        if (orient3d(a3, b3, c3, vertex) != 0) return false;
      }
      const double p0[2] = {p[0][x], p[0][y]};
      const double p1[2] = {p[1][x], p[1][y]};
      const double p2[2] = {p[2][x], p[2][y]};
      if (orient2d(p0, p1, p2) != orientation) return false;
    }
    return true;
  }

  // Rebuild the exact current 2D face triangulation from every carrier-chain
  // interval, using the same constrained-Delaunay service as source preparation.
  int32_t current_source_face_triangles(
      const PlcSourceFace& source,
      const std::unordered_map<
          int32_t, std::vector<std::pair<std::size_t, std::size_t>>>&
          chain_positions,
      std::vector<PlcSourceFace>& children) const {
    children.clear();
    std::vector<int32_t> boundary;
    std::set<uint64_t> boundary_intervals;
    for (int side = 0; side < 3; ++side) {
      const int32_t first = source.oriented[static_cast<std::size_t>(side)];
      const int32_t second =
          source.oriented[static_cast<std::size_t>((side + 1) % 3)];
      const auto positions = chain_positions.find(first);
      std::size_t chain_row = edge_chains_.size();
      std::size_t chain_first = 0;
      std::size_t chain_second = 0;
      if (positions != chain_positions.end()) {
        for (const auto& [row, position] : positions->second) {
          if (row >= edge_chains_.size()) return PHX_MC_INTERNAL_ERROR;
          const auto& chain = edge_chains_[row];
          const auto second_position = std::find_if(
              chain.begin(), chain.end(),
              [&](const PlcEdgePoint& value) { return value.vertex == second; });
          if (second_position == chain.end()) continue;
          if (chain_row != edge_chains_.size()) return PHX_MC_INTERNAL_ERROR;
          chain_row = row;
          chain_first = position;
          chain_second =
              static_cast<std::size_t>(second_position - chain.begin());
        }
      }
      if (chain_row == edge_chains_.size()) {
        boundary.push_back(first);
        continue;
      }
      const auto& chain = edge_chains_[chain_row];
      if (chain_first >= chain.size() || chain_second >= chain.size() ||
          chain_first == chain_second) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const std::ptrdiff_t step = chain_first < chain_second ? 1 : -1;
      for (std::ptrdiff_t position = static_cast<std::ptrdiff_t>(chain_first);
           position != static_cast<std::ptrdiff_t>(chain_second);
           position += step) {
        const int32_t vertex =
            chain[static_cast<std::size_t>(position)].vertex;
        const int32_t next =
            chain[static_cast<std::size_t>(position + step)].vertex;
        boundary.push_back(vertex);
        boundary_intervals.insert(topology::undirected_edge(vertex, next));
      }
    }
    if (boundary.size() < 3) return PHX_MC_CONSTRAINT_INTERSECTION;

    const int axis =
        exact_projection_axis(source.oriented[0], source.oriented[1], source.oriented[2]);
    if (axis < 0) return PHX_MC_CONSTRAINT_INTERSECTION;
    const ExactPointRef corners[3] = {ref(source.oriented[0]), ref(source.oriented[1]),
                                      ref(source.oriented[2])};
    const int orientation = orient2d_projected(corners[0], corners[1], corners[2], axis);
    if (orientation == 0) return PHX_MC_CONSTRAINT_INTERSECTION;
    const auto point_location = [&](int32_t vertex) {
      const ExactPointRef query = ref(vertex);
      if (orient3d(corners[0], corners[1], corners[2], query) != 0) return 0;
      bool on_boundary = false;
      for (int side = 0; side < 3; ++side) {
        const ExactPointRef& p = corners[side];
        const ExactPointRef& q = corners[(side + 1) % 3];
        const int side_sign = orient2d_projected(p, q, query, axis) * orientation;
        if (side_sign < 0) return 0;
        if (side_sign == 0 && between_on_line(p, q, query)) on_boundary = true;
      }
      return on_boundary ? 1 : 2;
    };
    std::vector<std::array<int32_t, 2>> internal_constraints;
    std::set<uint64_t> internal_keys;
    for (const auto& chain : edge_chains_) {
      for (std::size_t interval = 0; interval + 1 < chain.size(); ++interval) {
        const int32_t first = chain[interval].vertex;
        const int32_t second = chain[interval + 1].vertex;
        const uint64_t key = topology::undirected_edge(first, second);
        if (boundary_intervals.contains(key)) continue;
        const int first_location = point_location(first);
        const int second_location = point_location(second);
        if (first_location > 0 && second_location > 0) {
          if (internal_keys.insert(key).second) {
            internal_constraints.push_back({first, second});
          }
          continue;
        }
        if ((first_location == 2 && second_location == 0) ||
            (first_location == 0 && second_location == 2)) {
          return PHX_MC_CONSTRAINT_INTERSECTION;
        }
        if (first_location == 0 && second_location == 0 &&
            orient3d(corners[0], corners[1], corners[2], ref(first)) == 0 &&
            orient3d(corners[0], corners[1], corners[2], ref(second)) == 0) {
          for (int side = 0; side < 3; ++side) {
            if (segments_meet(ref(first), ref(second), corners[side],
                              corners[(side + 1) % 3])) {
              return PHX_MC_CONSTRAINT_INTERSECTION;
            }
          }
        }
      }
    }
    std::vector<Triple> triangles;
    const bool witnessed = std::any_of(boundary.begin(), boundary.end(), [&](int32_t vertex) {
      return ref(vertex).exact != nullptr;
    });
    if (witnessed) {
      // Binary64 carriers of witnessed vertices are off the facet plane, so
      // the planar loop is triangulated on exact positions.
      if (!internal_constraints.empty()) return PHX_MC_CONSTRAINT_INTERSECTION;
      if (!exact_loop_triangles(boundary, axis, orientation, triangles)) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
      children.reserve(triangles.size());
      for (const Triple& face : triangles) {
        children.push_back({face, source.facet, source.source_triangle});
      }
      return PHX_MC_OK;
    }
    const int32_t status = triangulate_planar_loop(
        work_points_.data(),
        static_cast<int64_t>(candidate_.points.size() / 3), boundary.data(),
        static_cast<int64_t>(boundary.size()), internal_constraints, triangles);
    if (status != PHX_MC_OK) {
      return status == PHX_MC_INTERNAL_ERROR ? status
                                             : PHX_MC_CONSTRAINT_INTERSECTION;
    }
    std::map<uint64_t, std::size_t> edge_uses;
    for (const Triple& triangle : triangles) {
      for (int side = 0; side < 3; ++side) {
        ++edge_uses[topology::undirected_edge(
            triangle[static_cast<std::size_t>(side)],
            triangle[static_cast<std::size_t>((side + 1) % 3)])];
      }
    }
    for (uint64_t interval : boundary_intervals) {
      const auto uses = edge_uses.find(interval);
      if (uses == edge_uses.end() || uses->second != 1) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
    }
    for (uint64_t interval : internal_keys) {
      const auto uses = edge_uses.find(interval);
      if (uses == edge_uses.end() || uses->second != 2) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
    }

    std::vector<std::array<int32_t, 4>> parent_faces{
        {source.oriented[0], source.oriented[1], source.oriented[2], 0}};
    std::vector<int32_t> parent_rows{0};
    std::vector<std::array<int32_t, 4>> child_faces;
    std::vector<int32_t> child_rows;
    child_faces.reserve(triangles.size());
    child_rows.reserve(triangles.size());
    for (std::size_t row = 0; row < triangles.size(); ++row) {
      const Triple& face = triangles[row];
      child_faces.push_back(
          {face[0], face[1], face[2], static_cast<int32_t>(row)});
      child_rows.push_back(static_cast<int32_t>(row));
    }
    std::vector<int32_t> contributors;
    const auto covered_by = [&](const std::array<int32_t, 4>& subject,
                                const std::vector<std::array<int32_t, 4>>& faces,
                                const std::vector<int32_t>& rows) {
      double coordinates[9];
      for (int corner = 0; corner < 3; ++corner) {
        std::copy_n(point(subject[corner]), 3, coordinates + 3 * corner);
      }
      contributors.clear();
      return exact_facet_coverage(coordinates, work_points_.data(), faces, rows,
                                  contributors) &&
             !contributors.empty();
    };
    if (!covered_by(parent_faces.front(), child_faces, child_rows)) {
      return PHX_MC_CONSTRAINT_INTERSECTION;
    }
    for (const auto& child : child_faces) {
      if (!covered_by(child, parent_faces, parent_rows)) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
    }
    children.reserve(triangles.size());
    for (const Triple& face : triangles) {
      children.push_back({face, source.facet, source.source_triangle});
    }
    return PHX_MC_OK;
  }
  // Start from authored parent faces, rebuild their present edge-chain
  // subdivision, then add only same-oriented source triangles covering the
  // exact old hull patch; the transaction below retains their source rows.
  int32_t prepare_current_source_face_patch(
      const std::vector<std::pair<int32_t, int32_t>>& seed_parents,
      PlcSourceFacePatch& patch) {
    patch = PlcSourceFacePatch{};
    if (seed_parents.empty()) return PHX_MC_CONSTRAINT_INTERSECTION;
    std::unordered_map<int32_t, std::size_t> facet_polygon_counts;
    facet_polygon_counts.reserve(static_cast<std::size_t>(input_.polygon_count));
    for (int64_t polygon = 0; polygon < input_.polygon_count; ++polygon) {
      ++facet_polygon_counts[input_.polygon_facets[polygon]];
    }
    std::map<std::pair<int32_t, int32_t>,
             std::vector<std::pair<Triple, PlcSourceFace>>> source_groups;
    std::set<std::pair<int32_t, int32_t>> eligible_source_groups;
    for (const auto& [key, source] : source_faces_) {
      if (source.source_triangle < 0 ||
          static_cast<std::size_t>(source.source_triangle) >=
              candidate_.input_triangles.size() / 3 ||
          source.facet < 0 || source.facet >= input_.facet_count) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const int32_t polygon =
          candidate_.input_polygons[static_cast<std::size_t>(source.source_triangle)];
      if (polygon < 0 || polygon >= input_.polygon_count ||
          input_.polygon_facets[polygon] != source.facet) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const std::pair<int32_t, int32_t> parent{
          source.source_triangle, source.facet};
      source_groups[parent].push_back({key, source});
      if (input_.polygon_offsets[polygon + 1] - input_.polygon_offsets[polygon] == 3 &&
          facet_polygon_counts[source.facet] == 1) {
        eligible_source_groups.insert(parent);
      }
    }
    std::set<std::pair<int32_t, int32_t>> selected_groups;
    for (const auto& parent : seed_parents) {
      if (!source_groups.contains(parent)) return PHX_MC_CONSTRAINT_INTERSECTION;
      selected_groups.insert(parent);
    }

    std::unordered_map<
        int32_t, std::vector<std::pair<std::size_t, std::size_t>>> chain_positions;
    for (std::size_t edge = 0; edge < edge_chains_.size(); ++edge) {
      const auto& chain = edge_chains_[edge];
      for (std::size_t position = 0; position < chain.size(); ++position) {
        chain_positions[chain[position].vertex].push_back({edge, position});
      }
    }
    std::set<Triple> current_face_keys;
    const TetrahedralComplex& complex = triangulation_->complex();
    for (std::size_t row = 0; row < complex.tets.size(); ++row) {
      const int32_t id = static_cast<int32_t>(row);
      const Tetrahedron& tet = complex.tets[row];
      if (!complex.live(id) || is_ghost(tet)) continue;
      for (int slot = 0; slot < 4; ++slot) {
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        current_face_keys.insert(sorted_triple(face.data()));
      }
    }

    const auto source_rows_cover_parent = [&](const auto& rows) -> int32_t {
      if (rows.empty()) return PHX_MC_CONSTRAINT_INTERSECTION;
      const PlcSourceFace& reference = rows.front().second;
      if (reference.source_triangle < 0 ||
          static_cast<std::size_t>(reference.source_triangle) >=
              candidate_.input_triangles.size() / 3 ||
          reference.facet < 0 || reference.facet >= input_.facet_count) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const int32_t* original =
          candidate_.input_triangles.data() + 3 * reference.source_triangle;
      std::array<int32_t, 4> parent_face{
          original[0], original[1], original[2], 0};
      std::vector<std::array<int32_t, 4>> current_faces;
      std::vector<int32_t> current_rows;
      current_faces.reserve(rows.size());
      current_rows.reserve(rows.size());
      for (std::size_t row = 0; row < rows.size(); ++row) {
        const PlcSourceFace& source = rows[row].second;
        if (source.source_triangle != reference.source_triangle ||
            source.facet != reference.facet) {
          return PHX_MC_INTERNAL_ERROR;
        }
        current_faces.push_back(
            {source.oriented[0], source.oriented[1], source.oriented[2],
             static_cast<int32_t>(row)});
        current_rows.push_back(static_cast<int32_t>(row));
      }
      const std::vector<std::array<int32_t, 4>> parent_faces{parent_face};
      const std::vector<int32_t> parent_rows{0};
      std::vector<int32_t> contributors;
      const auto covered_by = [&](const std::array<int32_t, 4>& subject,
                                  const std::vector<std::array<int32_t, 4>>& faces,
                                  const std::vector<int32_t>& rows) {
        double coordinates[9];
        for (int corner = 0; corner < 3; ++corner) {
          std::copy_n(point(subject[corner]), 3, coordinates + 3 * corner);
        }
        contributors.clear();
        return exact_facet_coverage(coordinates, work_points_.data(), faces, rows,
                                    contributors) &&
               !contributors.empty();
      };
      if (!covered_by(parent_face, current_faces, current_rows)) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
      for (const auto& face : current_faces) {
        if (!covered_by(face, parent_faces, parent_rows)) {
          return PHX_MC_CONSTRAINT_INTERSECTION;
        }
      }
      return PHX_MC_OK;
    };
    struct CurrentGroup {
      std::vector<PlcSourceFace> source_faces;
      std::vector<PlcSourceFace> patch_faces;
      std::vector<PlcSourceFace> recovery_faces;
      std::vector<Triple> source_keys;
    };
    const auto build_group = [&](const auto& rows, CurrentGroup& group) -> int32_t {
      group = CurrentGroup{};
      const int32_t parent_coverage = source_rows_cover_parent(rows);
      if (parent_coverage != PHX_MC_OK) return parent_coverage;
      std::set<Triple> all_keys;
      for (const auto& [source_key, source] : rows) {
        group.source_keys.push_back(source_key);
        std::vector<PlcSourceFace> children;
        const int32_t status =
            current_source_face_triangles(source, chain_positions, children);
        if (status != PHX_MC_OK) return status;
        for (const PlcSourceFace& child : children) {
          const Triple key = sorted_triple(child.oriented.data());
          if (!all_keys.insert(key).second) return PHX_MC_CONSTRAINT_INTERSECTION;
          const auto existing = source_faces_.find(key);
          if (existing != source_faces_.end() &&
              (existing->second.source_triangle != child.source_triangle ||
               existing->second.facet != child.facet)) {
            return PHX_MC_CONSTRAINT_INTERSECTION;
          }
          group.source_faces.push_back(child);
          group.patch_faces.push_back(child);
          if (!current_face_keys.contains(key)) {
            group.recovery_faces.push_back(child);
          }
        }
      }
      return PHX_MC_OK;
    };
    const auto add_group = [&](const CurrentGroup& group) -> int32_t {
      for (const Triple& key : group.source_keys) {
        if (std::find(patch.replaced_source_faces.begin(),
                      patch.replaced_source_faces.end(), key) !=
            patch.replaced_source_faces.end()) {
          return PHX_MC_INTERNAL_ERROR;
        }
      }
      patch.replaced_source_faces.insert(patch.replaced_source_faces.end(),
                                         group.source_keys.begin(),
                                         group.source_keys.end());
      std::set<Triple> patch_keys;
      for (const PlcSourceFace& face : patch.faces) {
        patch_keys.insert(sorted_triple(face.oriented.data()));
      }
      for (const PlcSourceFace& face : group.patch_faces) {
        const Triple key = sorted_triple(face.oriented.data());
        if (!patch_keys.insert(key).second) return PHX_MC_CONSTRAINT_INTERSECTION;
        patch.faces.push_back(face);
      }
      patch.source_faces.insert(patch.source_faces.end(),
                                group.source_faces.begin(),
                                group.source_faces.end());
      for (const PlcSourceFace& face : group.recovery_faces) {
        patch.recovery_faces.push_back(face);
        std::vector<int32_t> face_cavity;
        const int32_t status = collect_fixed_facet_cavity(face, face_cavity);
        if (status != PHX_MC_OK) return status;
        if (face_cavity.empty()) return PHX_MC_CONSTRAINT_INTERSECTION;
        patch.cavity.insert(patch.cavity.end(), face_cavity.begin(),
                            face_cavity.end());
      }
      return PHX_MC_OK;
    };

    for (const auto& parent : selected_groups) {
      CurrentGroup group;
      const int32_t status = build_group(source_groups.at(parent), group);
      if (status != PHX_MC_OK) return status;
      const int32_t add_status = add_group(group);
      if (add_status != PHX_MC_OK) return add_status;
    }
    if (patch.faces.empty() || patch.recovery_faces.empty()) {
      return PHX_MC_CONSTRAINT_INTERSECTION;
    }
    for (;;) {
      std::sort(patch.cavity.begin(), patch.cavity.end());
      patch.cavity.erase(std::unique(patch.cavity.begin(), patch.cavity.end()),
                         patch.cavity.end());
      HullPatchPlan plan;
      const int32_t plan_status =
          collect_coplanar_hull_patch_plan(patch.faces, patch.cavity, plan);
      if (plan_status != PHX_MC_OK) return plan_status;
      if (plan.old_faces.empty()) break;
      bool added_group = false;
      for (const auto& [parent, rows] : source_groups) {
        if (selected_groups.contains(parent) ||
            !eligible_source_groups.contains(parent)) continue;
        bool raw_group_overlap = false;
        for (const auto& [key, source] : rows) {
          (void)key;
          const Box source_box = edge_recovery_box(source.oriented.data(), 3);
          const double* source_points[3] = {
              point(source.oriented[0]), point(source.oriented[1]),
              point(source.oriented[2])};
          const int64_t source_ids[3] = {
              source.oriented[0], source.oriented[1], source.oriented[2]};
          for (const Triple& old_face : plan.old_faces) {
            if (!overlap(source_box, edge_recovery_box(old_face.data(), 3))) {
              continue;
            }
            const double* old_points[3] = {
                point(old_face[0]), point(old_face[1]), point(old_face[2])};
            const int64_t old_ids[3] = {
                old_face[0], old_face[1], old_face[2]};
            Contact contact{};
            const int32_t status = intersect_triangles(
                source_points, old_points, source_ids, old_ids, contact);
            if (status != PHX_MC_OK) return status;
            if (contact.kind == PHX_MC_COPLANAR_OVERLAP) {
              raw_group_overlap = true;
              break;
            }
          }
          if (raw_group_overlap) break;
        }
        if (!raw_group_overlap) continue;
        CurrentGroup group;
        const int32_t build_status = build_group(rows, group);
        if (build_status != PHX_MC_OK) return build_status;
        bool overlaps_patch = false;
        for (const PlcSourceFace& source : group.source_faces) {
          const Box source_box = edge_recovery_box(source.oriented.data(), 3);
          const double* source_points[3] = {
              point(source.oriented[0]), point(source.oriented[1]),
              point(source.oriented[2])};
          const int64_t source_ids[3] = {
              source.oriented[0], source.oriented[1], source.oriented[2]};
          for (const Triple& old_face : plan.old_faces) {
            if (!overlap(source_box, edge_recovery_box(old_face.data(), 3))) {
              continue;
            }
            const double* old_points[3] = {
                point(old_face[0]), point(old_face[1]), point(old_face[2])};
            const int64_t old_ids[3] = {
                old_face[0], old_face[1], old_face[2]};
            Contact contact{};
            const int32_t status = intersect_triangles(
                source_points, old_points, source_ids, old_ids, contact);
            if (status != PHX_MC_OK) return status;
            if (contact.kind == PHX_MC_COPLANAR_OVERLAP) {
              overlaps_patch = true;
              break;
            }
          }
        }
        if (!overlaps_patch) continue;
        std::vector<PlcSourceFace> combined = patch.faces;
        combined.insert(combined.end(), group.patch_faces.begin(),
                        group.patch_faces.end());
        if (!compatible_hull_source_patch(combined)) {
          return PHX_MC_CONSTRAINT_INTERSECTION;
        }
        const int32_t add_status = add_group(group);
        if (add_status != PHX_MC_OK) return add_status;
        selected_groups.insert(parent);
        added_group = true;
      }
      if (!added_group) break;
    }
    std::sort(patch.cavity.begin(), patch.cavity.end());
    patch.cavity.erase(std::unique(patch.cavity.begin(), patch.cavity.end()),
                       patch.cavity.end());
    std::sort(patch.replaced_source_faces.begin(),
              patch.replaced_source_faces.end());
    patch.replaced_source_faces.erase(
        std::unique(patch.replaced_source_faces.begin(),
                    patch.replaced_source_faces.end()),
        patch.replaced_source_faces.end());
    std::set<Triple> replaced_keys;
    for (const Triple& key : patch.replaced_source_faces) {
      if (source_faces_.find(key) == source_faces_.end() ||
          !replaced_keys.insert(key).second) {
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    std::set<Triple> source_child_keys;
    for (const PlcSourceFace& face : patch.source_faces) {
      const Triple key = sorted_triple(face.oriented.data());
      if (!source_child_keys.insert(key).second) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
      const auto existing = source_faces_.find(key);
      if (existing != source_faces_.end() && !replaced_keys.contains(key)) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
    }
    for (const PlcSourceFace& face : patch.faces) {
      if (!source_child_keys.contains(sorted_triple(face.oriented.data()))) {
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    for (const PlcSourceFace& face : patch.recovery_faces) {
      if (!source_child_keys.contains(sorted_triple(face.oriented.data()))) {
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    if (!compatible_hull_source_patch(patch.faces)) {
      return PHX_MC_CONSTRAINT_INTERSECTION;
    }
    return PHX_MC_OK;
  }
  int32_t collect_coplanar_hull_patch_plan(
      const std::vector<PlcSourceFace>& source_patch,
      const std::vector<int32_t>& cavity, HullPatchPlan& plan) const {
    plan = HullPatchPlan{};
    if (!compatible_hull_source_patch(source_patch)) {
      return PHX_MC_CONSTRAINT_INTERSECTION;
    }
    const TetrahedralComplex& complex = triangulation_->complex();
    const std::set<Triple> source_keys = [&]() {
      std::set<Triple> keys;
      for (const PlcSourceFace& source : source_patch) {
        keys.insert(sorted_triple(source.oriented.data()));
      }
      return keys;
    }();
    for (int32_t id : cavity) {
      if (!complex.live(id) || is_ghost(complex.tets[static_cast<std::size_t>(id)])) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t neighbor = tet.n[slot];
        if (!complex.live(neighbor)) return PHX_MC_INTERNAL_ERROR;
        if (std::binary_search(cavity.begin(), cavity.end(), neighbor) ||
            !is_ghost(complex.tets[static_cast<std::size_t>(neighbor)])) {
          continue;
        }
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        const Triple key = sorted_triple(face.data());
        if (source_keys.contains(key)) {
          if (complex.constraint(id, slot) != kNoConstraint) {
            return PHX_MC_CONSTRAINT_INTERSECTION;
          }
          plan.replaceable_faces.insert(key);
          plan.old_faces.push_back(face);
          plan.ghost_cells.push_back(neighbor);
          continue;
        }
        const Box face_box = edge_recovery_box(face.data(), 3);
        const double* face_points[3] = {
            point(face[0]), point(face[1]), point(face[2])};
        const int64_t face_ids[3] = {face[0], face[1], face[2]};
        bool overlaps_patch = false;
        for (const PlcSourceFace& source : source_patch) {
          if (!overlap(edge_recovery_box(source.oriented.data(), 3), face_box)) {
            continue;
          }
          const double* source_points[3] = {
              point(source.oriented[0]), point(source.oriented[1]),
              point(source.oriented[2])};
          const int64_t source_ids[3] = {
              source.oriented[0], source.oriented[1], source.oriented[2]};
          Contact contact{};
          const int32_t status = intersect_triangles(
              source_points, face_points, source_ids, face_ids, contact);
          if (status != PHX_MC_OK) return status;
          if (contact.kind == PHX_MC_COPLANAR_OVERLAP) {
            overlaps_patch = true;
            break;
          }
        }
        if (!overlaps_patch) continue;
        if (complex.constraint(id, slot) != kNoConstraint) {
          return PHX_MC_CONSTRAINT_INTERSECTION;
        }
        plan.replaceable_faces.insert(key);
        plan.old_faces.push_back(face);
        plan.ghost_cells.push_back(neighbor);
      }
    }
    std::sort(plan.old_faces.begin(), plan.old_faces.end(),
              [](const Triple& first, const Triple& second) {
                return sorted_triple(first.data()) < sorted_triple(second.data());
              });
    plan.old_faces.erase(
        std::unique(plan.old_faces.begin(), plan.old_faces.end(),
                    [](const Triple& first, const Triple& second) {
                      return sorted_triple(first.data()) ==
                             sorted_triple(second.data());
                    }),
        plan.old_faces.end());
    std::sort(plan.ghost_cells.begin(), plan.ghost_cells.end());
    plan.ghost_cells.erase(
        std::unique(plan.ghost_cells.begin(), plan.ghost_cells.end()),
        plan.ghost_cells.end());
    if (plan.old_faces.size() != plan.ghost_cells.size() ||
        plan.old_faces.size() != plan.replaceable_faces.size()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    return PHX_MC_OK;
  }

  int32_t prepare_coplanar_hull_patch(
      const std::vector<PlcSourceFace>& source_patch,
      const std::vector<PlcSourceFace>& recovery_patch,
      const std::vector<int32_t>& cavity,
      const std::vector<std::array<int32_t, 4>>& proposed,
      const HullPatchPlan& plan, bool& active,
      std::vector<int32_t>& ghost_cells,
      std::vector<std::array<int32_t, 4>>& ghost_shells) const {
    active = false;
    ghost_cells.clear();
    ghost_shells.clear();
    if (plan.old_faces.empty()) {
      return plan.ghost_cells.empty() && plan.replaceable_faces.empty()
          ? PHX_MC_OK : PHX_MC_INTERNAL_ERROR;
    }
    if (proposed.empty() || !compatible_hull_source_patch(source_patch)) {
      return PHX_MC_CONSTRAINT_INTERSECTION;
    }
    const PlcSourceFace& reference = source_patch.front();
    const TetrahedralComplex& complex = triangulation_->complex();
    struct ProposedFaceUse {
      std::size_t cell;
      int slot;
      Triple oriented;
    };
    std::map<Triple, std::vector<ProposedFaceUse>> proposed_faces;
    for (std::size_t cell = 0; cell < proposed.size(); ++cell) {
      for (int slot = 0; slot < 4; ++slot) {
        Triple face{};
        oriented_facet(proposed[cell].data(), slot, face.data());
        proposed_faces[sorted_triple(face.data())].push_back({cell, slot, face});
      }
    }
    std::map<Triple, int32_t> old_boundary_neighbors;
    for (int32_t id : cavity) {
      if (!complex.live(id) || is_ghost(complex.tets[static_cast<std::size_t>(id)])) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t neighbor = tet.n[slot];
        if (!complex.live(neighbor)) return PHX_MC_INTERNAL_ERROR;
        if (std::binary_search(cavity.begin(), cavity.end(), neighbor)) continue;
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        if (!old_boundary_neighbors.emplace(sorted_triple(face.data()), neighbor).second) {
          return PHX_MC_INTERNAL_ERROR;
        }
      }
    }
    const std::set<int32_t> candidate_ghosts(
        plan.ghost_cells.begin(), plan.ghost_cells.end());
    std::set<int32_t> removed_ghosts;
    std::vector<Triple> replaced_old_faces;
    for (const Triple& face : plan.old_faces) {
      const Triple key = sorted_triple(face.data());
      const auto old = old_boundary_neighbors.find(key);
      if (old == old_boundary_neighbors.end() ||
          !candidate_ghosts.contains(old->second) ||
          !complex.live(old->second) ||
          !is_ghost(complex.tets[static_cast<std::size_t>(old->second)])) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const auto proposed_use = proposed_faces.find(key);
      if (proposed_use != proposed_faces.end() &&
          proposed_use->second.size() == 1) {
        continue;
      }
      removed_ghosts.insert(old->second);
      replaced_old_faces.push_back(face);
    }

    std::set<Triple> source_keys;
    for (const PlcSourceFace& source : source_patch) {
      source_keys.insert(sorted_triple(source.oriented.data()));
    }
    for (const PlcSourceFace& source : recovery_patch) {
      const auto use = proposed_faces.find(sorted_triple(source.oriented.data()));
      if (use == proposed_faces.end() ||
          (use->second.size() != 1 && use->second.size() != 2)) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
      if (use->second.size() == 2 &&
          !(FacetKey::of(use->second[0].oriented.data()) ==
            FacetKey::of(use->second[1].oriented.data()).reversed())) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
    }
    if (replaced_old_faces.empty()) return PHX_MC_OK;
    std::vector<ProposedFaceUse> new_hull_faces;
    for (const auto& [key, uses] : proposed_faces) {
      if (uses.size() == 2) {
        if (!(FacetKey::of(uses[0].oriented.data()) ==
              FacetKey::of(uses[1].oriented.data()).reversed())) {
          return PHX_MC_CONSTRAINT_INTERSECTION;
        }
        continue;
      }
      if (uses.size() != 1) return PHX_MC_CONSTRAINT_INTERSECTION;
      const auto old = old_boundary_neighbors.find(key);
      if (old != old_boundary_neighbors.end()) {
        if (!removed_ghosts.contains(old->second)) continue;
        if (!source_keys.contains(key)) return PHX_MC_CONSTRAINT_INTERSECTION;
      } else {
        if (!source_keys.contains(key)) return PHX_MC_CONSTRAINT_INTERSECTION;
        for (int32_t vertex : uses.front().oriented) {
          if (orient(reference.oriented[0], reference.oriented[1], reference.oriented[2], vertex) != 0) {
            return PHX_MC_CONSTRAINT_INTERSECTION;
          }
        }
      }
      new_hull_faces.push_back(uses.front());
    }
    if (new_hull_faces.empty()) return PHX_MC_CONSTRAINT_INTERSECTION;

    std::vector<std::array<int32_t, 4>> old_primitives;
    std::vector<std::array<int32_t, 4>> new_primitives;
    std::vector<int32_t> old_rows(replaced_old_faces.size());
    std::vector<int32_t> new_rows(new_hull_faces.size());
    old_primitives.reserve(replaced_old_faces.size());
    new_primitives.reserve(new_hull_faces.size());
    for (std::size_t row = 0; row < replaced_old_faces.size(); ++row) {
      const Triple& face = replaced_old_faces[row];
      old_primitives.push_back(
          {face[0], face[1], face[2], static_cast<int32_t>(row)});
      old_rows[row] = static_cast<int32_t>(row);
    }
    for (std::size_t row = 0; row < new_hull_faces.size(); ++row) {
      const Triple& face = new_hull_faces[row].oriented;
      new_primitives.push_back(
          {face[0], face[1], face[2], static_cast<int32_t>(row)});
      new_rows[row] = static_cast<int32_t>(row);
    }
    std::vector<int32_t> contributors;
    const auto covered_by = [&](const std::array<int32_t, 4>& subject,
                                const std::vector<std::array<int32_t, 4>>& faces,
                                const std::vector<int32_t>& rows) {
      double coordinates[9];
      for (int corner = 0; corner < 3; ++corner) {
        std::copy_n(point(subject[corner]), 3, coordinates + 3 * corner);
      }
      contributors.clear();
      return exact_facet_coverage(coordinates, work_points_.data(), faces, rows,
                                  contributors) &&
             !contributors.empty();
    };
    for (const auto& face : old_primitives) {
      if (!covered_by(face, new_primitives, new_rows)) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
    }
    for (const auto& face : new_primitives) {
      if (!covered_by(face, old_primitives, old_rows)) {
        return PHX_MC_CONSTRAINT_INTERSECTION;
      }
    }
    ghost_cells.assign(removed_ghosts.begin(), removed_ghosts.end());
    if (ghost_cells.size() != replaced_old_faces.size()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    ghost_shells.reserve(new_hull_faces.size());
    for (const ProposedFaceUse& use : new_hull_faces) {
      std::array<int32_t, 4> shell = proposed[use.cell];
      shell[use.slot] = kGhostVertex;
      const int first = use.slot == 0 ? 1 : 0;
      const int second = use.slot <= 1 ? 2 : 1;
      std::swap(shell[first], shell[second]);
      int32_t ghost_face[3];
      oriented_facet(shell.data(), use.slot, ghost_face);
      if (!(FacetKey::of(ghost_face) ==
            FacetKey::of(use.oriented.data()).reversed())) {
        return PHX_MC_INTERNAL_ERROR;
      }
      ghost_shells.push_back(shell);
    }
    active = true;
    return PHX_MC_OK;
  }
  int32_t recover_source_facet_patch(
      PlcSourceFacePatch patch, bool diagnostic_capture = false,
      bool allow_source_face_steiner = true) {
    const std::vector<PlcSourceFace>& source_patch = patch.faces;
    const std::vector<PlcSourceFace>& recovery_patch = patch.recovery_faces;
    std::vector<int32_t>& cavity = patch.cavity;
    if (source_patch.empty()) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet, -1);
    }
    const PlcSourceFace& source = source_patch.front();
    if (!compatible_hull_source_patch(source_patch)) {
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                  kPlcFacet, source.facet);
    }
    if (recovery_patch.empty() || cavity.empty()) {
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                  kPlcFacet, source.facet);
    }
    std::sort(cavity.begin(), cavity.end());
    cavity.erase(std::unique(cavity.begin(), cavity.end()), cavity.end());
    recovery_edge_ = patch.recovery_edge[0] >= 0
        ? std::array<int32_t, 2>{patch.recovery_edge[0], patch.recovery_edge[1]}
        : std::array<int32_t, 2>{source.oriented[0], source.oriented[1]};
    EdgeRecoverySide positive;
    EdgeRecoverySide negative;
    std::map<Triple, int32_t> labels;
    EdgeRecoveryContactCache contact_cache;
    EdgeRecoveryFailureDetail positive_facet_contact{};
    EdgeRecoveryFailureDetail negative_facet_contact{};
    bool has_positive = false;
    bool has_negative = false;
    HullPatchPlan hull_plan;
    std::size_t patch_iteration = 0;
    for (;;) {
      ++patch_iteration;
      const auto iteration_started = std::chrono::steady_clock::now();
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS source_patch_begin edge=%zu iteration=%zu "
                     "cavity=%zu source_faces=%zu recovery_faces=%zu\n",
                     active_source_edge_, patch_iteration, cavity.size(),
                     source_patch.size(), recovery_patch.size());
      }
      positive_facet_contact = EdgeRecoveryFailureDetail{};
      negative_facet_contact = EdgeRecoveryFailureDetail{};
      const int32_t patch_status =
          collect_coplanar_hull_patch_plan(source_patch, cavity, hull_plan);
      if (patch_status != PHX_MC_OK) {
        recovery_edge_ = {-1, -1};
        if (patch_status == PHX_MC_CONSTRAINT_INTERSECTION) {
          return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                      kPlcFacet, source.facet);
        }
        return fail(patch_status, kPlcNonrepresentable, kPlcFacet, source.facet);
      }
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS source_patch_plan edge=%zu iteration=%zu "
                     "old_hull_faces=%zu ghost_cells=%zu seconds=%.3f\n",
                     active_source_edge_, patch_iteration,
                     hull_plan.old_faces.size(), hull_plan.ghost_cells.size(),
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - iteration_started)
                         .count());
      }
      if (!prepare_fixed_facet_sides(cavity, source_patch, positive, negative,
                                     labels, has_positive, has_negative,
                                     &hull_plan, diagnostic_capture)) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                    kPlcFacet, source.facet);
      }
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS source_patch_sides edge=%zu iteration=%zu "
                     "positive=%d faces=%zu vertices=%zu negative=%d faces=%zu "
                     "vertices=%zu\n",
                     active_source_edge_, patch_iteration, has_positive ? 1 : 0,
                     positive.faces.size(), positive.vertices.size(),
                     has_negative ? 1 : 0, negative.faces.size(),
                     negative.vertices.size());
      }
      std::vector<std::array<int32_t, 4>> positive_cone;
      std::vector<std::array<int32_t, 4>> negative_cone;
      const bool positive_coned =
          !has_positive || try_cone_source_facet_side(
                               positive, cavity, source, 1, source_patch,
                               positive_cone);
      const bool negative_coned =
          !has_negative || try_cone_source_facet_side(
                               negative, cavity, source, -1, source_patch,
                               negative_cone);
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS source_patch_cone edge=%zu iteration=%zu "
                     "positive=%d tets=%zu negative=%d tets=%zu seconds=%.3f\n",
                     active_source_edge_, patch_iteration,
                     positive_coned ? 1 : 0, positive_cone.size(),
                     negative_coned ? 1 : 0, negative_cone.size(),
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - iteration_started)
                         .count());
      }
      if (positive_coned && negative_coned) {
        positive.tets = std::move(positive_cone);
        negative.tets = std::move(negative_cone);
        break;
      }
      bool positive_complete = !has_positive;
      bool negative_complete = !has_negative;
      int32_t wrap_status = PHX_MC_OK;
      if (has_positive) {
        wrap_status = wrap_edge_recovery_side(
            positive, positive_complete, positive_facet_contact, false,
            &contact_cache, diagnostic_capture, source_patch,
            allow_source_face_steiner);
      }
      if (wrap_status == PHX_MC_OK && has_negative) {
        wrap_status = wrap_edge_recovery_side(
            negative, negative_complete, negative_facet_contact, false,
            &contact_cache, diagnostic_capture, source_patch,
            allow_source_face_steiner);
      }
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS source_patch_wrap edge=%zu iteration=%zu "
                     "status=%d positive=%d/%zu negative=%d/%zu "
                     "cache_hits=%zu cache_misses=%zu seconds=%.3f\n",
                     active_source_edge_, patch_iteration, wrap_status,
                     positive_complete ? 1 : 0, positive.tets.size(),
                     negative_complete ? 1 : 0, negative.tets.size(),
                     contact_cache.hits, contact_cache.misses,
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - iteration_started)
                         .count());
      }
      if (cavity_steiner_restart_required_) {
        recovery_edge_ = {-1, -1};
        return PHX_MC_OK;
      }
      if (wrap_status != PHX_MC_OK) {
        recovery_edge_ = {-1, -1};
        return fail(wrap_status, kPlcNonrepresentable, kPlcFacet, source.facet);
      }
      if (positive_complete && negative_complete) break;
      bool grew = false;
      int32_t growth_status = PHX_MC_OK;
      if (!positive_complete &&
          positive_facet_contact.has_facet_contact) {
        growth_status = grow_edge_recovery_cavity(
            positive, cavity, positive_facet_contact, grew);
      }
      if (growth_status == PHX_MC_OK && !grew && !negative_complete &&
          negative_facet_contact.has_facet_contact) {
        growth_status = grow_edge_recovery_cavity(
            negative, cavity, negative_facet_contact, grew);
      }
      if (growth_status != PHX_MC_OK) {
        recovery_edge_ = {-1, -1};
        return fail(growth_status, kPlcNonrepresentable, kPlcFacet,
                    source.facet);
      }
      if (progress_enabled_) {
        std::fprintf(stderr,
                     "PLC_PROGRESS source_patch_conflict_growth edge=%zu "
                     "iteration=%zu grew=%d cavity=%zu pos=(%d,%d,%d) "
                     "neg=(%d,%d,%d) seconds=%.3f\n",
                     active_source_edge_, patch_iteration, grew ? 1 : 0,
                     cavity.size(),
                     positive_facet_contact.has_facet_contact
                         ? positive_facet_contact.obstacle_owner : -1,
                     positive_facet_contact.has_facet_contact
                         ? positive_facet_contact.obstacle_neighbor : -1,
                     positive_facet_contact.has_facet_contact
                         ? positive_facet_contact.source : kNoConstraint,
                     negative_facet_contact.has_facet_contact
                         ? negative_facet_contact.obstacle_owner : -1,
                     negative_facet_contact.has_facet_contact
                         ? negative_facet_contact.obstacle_neighbor : -1,
                     negative_facet_contact.has_facet_contact
                         ? negative_facet_contact.source : kNoConstraint,
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - iteration_started)
                         .count());
      }
      if (!grew) {
        recovery_edge_ = {-1, -1};
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                    kPlcFacet, source.facet);
      }
    }
    recovery_edge_ = {-1, -1};

    std::vector<std::array<int32_t, 4>> proposed;
    proposed.reserve(positive.tets.size() + negative.tets.size());
    proposed.insert(proposed.end(), positive.tets.begin(), positive.tets.end());
    proposed.insert(proposed.end(), negative.tets.begin(), negative.tets.end());
    if (proposed.empty()) {
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                  kPlcFacet, source.facet);
    }
    std::set<int32_t> used_vertices;
    for (const auto& tet : proposed) {
      for (int32_t vertex : tet) used_vertices.insert(vertex);
    }
    for (int32_t vertex : positive.vertices) {
      if (!used_vertices.contains(vertex)) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                    kPlcFacet, source.facet);
      }
    }
    Expansion old_volume;
    Expansion new_volume;
    const TetrahedralComplex& before = triangulation_->complex();
    for (int32_t id : cavity) {
      const Tetrahedron& tet = before.tets[static_cast<std::size_t>(id)];
      old_volume = old_volume + orient3d_exact(
          point(tet.v[0]), point(tet.v[1]), point(tet.v[2]), point(tet.v[3]));
    }
    for (std::size_t first = 0; first < proposed.size(); ++first) {
      const auto& tet = proposed[first];
      const Expansion volume = orient3d_exact(
          point(tet[0]), point(tet[1]), point(tet[2]), point(tet[3]));
      if (volume.sign() <= 0) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                    kPlcFacet, source.facet);
      }
      new_volume = new_volume + volume;
      for (std::size_t second = 0; second < first; ++second) {
        if (!recovery_tetrahedra_disjoint(proposed[second], tet)) {
          return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                      kPlcFacet, source.facet);
        }
      }
    }
    if ((new_volume - old_volume).sign() != 0) {
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                  kPlcFacet, source.facet);
    }
    std::vector<int32_t> hull_ghost_cells;
    std::vector<std::array<int32_t, 4>> hull_ghost_shells;
    bool hull_patch_replaced = false;
    const int32_t hull_status = prepare_coplanar_hull_patch(
        source_patch, recovery_patch, cavity, proposed, hull_plan,
        hull_patch_replaced, hull_ghost_cells, hull_ghost_shells);
    if (hull_status != PHX_MC_OK) {
      if (hull_status == PHX_MC_CONSTRAINT_INTERSECTION) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                    kPlcFacet, source.facet);
      }
      return fail(hull_status, kPlcNonrepresentable, kPlcFacet, source.facet);
    }
    std::set<Triple> replaced_source_keys;
    std::map<Triple, PlcSourceFace> staged_source_faces;
    for (const Triple& key : patch.replaced_source_faces) {
      if (source_faces_.find(key) == source_faces_.end() ||
          !replaced_source_keys.insert(key).second) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet,
                    source.facet);
      }
    }
    for (const PlcSourceFace& face : patch.source_faces) {
      const Triple key = sorted_triple(face.oriented.data());
      const auto existing = source_faces_.find(key);
      if (existing != source_faces_.end() &&
          !replaced_source_keys.contains(key)) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                    kPlcFacet, face.facet);
      }
      if (!staged_source_faces.emplace(key, face).second) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                    kPlcFacet, face.facet);
      }
    }

    TetrahedralComplex& complex = triangulation_->complex();
    const std::size_t max_slots = star_marks_.max_size();
    if (hull_ghost_shells.size() > max_slots ||
        proposed.size() > max_slots - hull_ghost_shells.size()) {
      return fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcFacet,
                  source.facet);
    }
    const std::size_t additional_slots =
        proposed.size() + hull_ghost_shells.size();
    if (complex.tets.size() > max_slots - additional_slots) {
      return fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcFacet,
                  source.facet);
    }
    const std::size_t required_slots = complex.tets.size() + additional_slots;
    if (star_marks_.capacity() < required_slots) {
      star_marks_.reserve(required_slots);
    }

    std::vector<int32_t> edit_cavity = cavity;
    if (hull_patch_replaced) {
      edit_cavity.insert(edit_cavity.end(), hull_ghost_cells.begin(),
                         hull_ghost_cells.end());
      std::sort(edit_cavity.begin(), edit_cavity.end());
      edit_cavity.erase(std::unique(edit_cavity.begin(), edit_cavity.end()),
                        edit_cavity.end());
    }
    CavityEdit edit(complex);
    edit.begin();
    for (int32_t id : edit_cavity) {
      if (edit.remove(id) != EditStatus::kOk) {
        edit.rollback();
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet,
                    source.facet);
      }
    }
    for (const auto& tet : proposed) {
      int32_t constraints[4] = {
          kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint};
      for (int slot = 0; slot < 4; ++slot) {
        int32_t face[3];
        oriented_facet(tet.data(), slot, face);
        const auto label = labels.find(sorted_triple(face));
        if (label != labels.end()) constraints[slot] = label->second;
      }
      if (edit.add(tet.data(), constraints, kNoRegion) != EditStatus::kOk) {
        edit.rollback();
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet,
                    source.facet);
      }
    }
    for (const auto& ghost : hull_ghost_shells) {
      if (edit.add(ghost.data(), nullptr, kNoRegion) != EditStatus::kOk) {
        edit.rollback();
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet,
                    source.facet);
      }
    }
    EditStatus edit_status = edit.validate([&](const int32_t* vertices) {
      return orient(vertices[0], vertices[1], vertices[2], vertices[3]) > 0;
    });
    if (edit_status == EditStatus::kOk) edit_status = edit.commit();
    local_edits_ = local_edits_ || edit_status == EditStatus::kOk;
    if (edit_status != EditStatus::kOk) {
      edit.rollback();
      if (edit_status == EditStatus::kSlotLimit) {
        return fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcFacet,
                    source.facet);
      }
      if (edit_status == EditStatus::kInternal) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet,
                    source.facet);
      }
      return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcFixedFacet,
                  kPlcFacet, source.facet);
    }
    star_marks_.resize(complex.tets.size(), 0U);

    // The map owns its replacement nodes before commit; merge transfers them
    // without allocation after the finite/ghost transaction succeeds.
    for (const Triple& key : patch.replaced_source_faces) {
      source_faces_.erase(key);
    }
    source_faces_.merge(staged_source_faces);
    for (int32_t id : edit.created()) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      if (is_ghost(tet)) continue;
      for (int32_t vertex : tet.v) {
        if (vertex >= 0) vertex_tet_[static_cast<std::size_t>(vertex)] = id;
      }
    }
    if (edit.created().empty() ||
        !triangulation_->set_hint(edit.created().front())) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet,
                  source.facet);
    }
    return PHX_MC_OK;
  }
  int32_t recover_current_source_face(
      const PlcSourceFace& source, bool allow_source_face_steiner = true) {
    PlcSourceFacePatch patch;
    const std::vector<std::pair<int32_t, int32_t>> parents{
        {source.source_triangle, source.facet}};
    const int32_t patch_status =
        prepare_current_source_face_patch(parents, patch);
    if (patch_status != PHX_MC_OK) {
      return fail(patch_status,
                  patch_status == PHX_MC_CONSTRAINT_INTERSECTION
                      ? kPlcFixedFacet : kPlcNonrepresentable,
                  kPlcFacet, source.facet);
    }
    patch.recovery_edge = {source.oriented[0], source.oriented[1]};
    return recover_source_facet_patch(std::move(patch), false,
                                      allow_source_face_steiner);
  }
  // A standalone authored triangle can recover an incident source edge as
  // part of the same exact face transaction. Do not pin triangles introduced
  // only by triangulating a larger loop or a shared source-facet group.
  int32_t recover_incident_authored_triangle_facets(
      std::size_t edge, int32_t first, int32_t second, bool& recovered) {
    const auto incident_patch_started = std::chrono::steady_clock::now();
    recovered = false;
    if (edge + 1 >= edge_triangle_offsets_.size()) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
    }
    const std::size_t begin = edge_triangle_offsets_[edge];
    const std::size_t end = edge_triangle_offsets_[edge + 1];
    if (begin > end || end > edge_triangle_rows_.size()) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
    }
    const int32_t carrier_first = candidate_.plc_edges[2 * edge];
    const int32_t carrier_second = candidate_.plc_edges[2 * edge + 1];
    std::vector<std::pair<int32_t, int32_t>> seed_parents;
    seed_parents.reserve(end - begin);
    for (std::size_t use = begin; use < end; ++use) {
      const int32_t triangle = edge_triangle_rows_[use];
      if (triangle < 0 ||
          static_cast<std::size_t>(triangle) >= candidate_.input_triangles.size() / 3) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
      }
      const int32_t polygon =
          candidate_.input_polygons[static_cast<std::size_t>(triangle)];
      if (polygon < 0 || polygon >= input_.polygon_count) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
      }
      if (input_.polygon_offsets[polygon + 1] -
              input_.polygon_offsets[polygon] != 3) {
        continue;
      }
      const int32_t facet = input_.polygon_facets[polygon];
      bool shared_facet_group = false;
      for (int64_t other = 0; other < input_.polygon_count; ++other) {
        if (other != polygon && input_.polygon_facets[other] == facet) {
          shared_facet_group = true;
          break;
        }
      }
      if (shared_facet_group) continue;
      const int32_t* vertices =
          candidate_.input_triangles.data() + 3 * static_cast<std::size_t>(triangle);
      if (std::find(vertices, vertices + 3, carrier_first) == vertices + 3 ||
          std::find(vertices, vertices + 3, carrier_second) == vertices + 3) {
        continue;
      }
      seed_parents.push_back({triangle, facet});
    }
    std::sort(seed_parents.begin(), seed_parents.end());
    seed_parents.erase(std::unique(seed_parents.begin(), seed_parents.end()),
                       seed_parents.end());
    const std::size_t direct_seed_count = seed_parents.size();
    // A hull replacement must close over every coplanar authored face that
    // shares a real source edge with the seeded patch. Keep each face/facet
    // identity; this only widens the exact patch transaction and never turns a
    // triangulation-only diagonal into a source edge.
    if (!seed_parents.empty()) {
      const int32_t reference_triangle = seed_parents.front().first;
      const int32_t reference_facet = seed_parents.front().second;
      const int32_t* reference =
          candidate_.input_triangles.data() +
          3 * static_cast<std::size_t>(reference_triangle);
      const int axis = projection_axis(point(reference[0]), point(reference[1]),
                                       point(reference[2]));
      if (axis >= 0) {
        const int x = (axis + 1) % 3;
        const int y = (axis + 2) % 3;
        const double* a = point(reference[0]);
        const double* b = point(reference[1]);
        const double* c = point(reference[2]);
        const double reference_a2[2] = {a[x], a[y]};
        const double reference_b2[2] = {b[x], b[y]};
        const double reference_c2[2] = {c[x], c[y]};
        const int reference_orientation =
            orient2d(reference_a2, reference_b2, reference_c2);
        std::set<uint64_t> source_edge_keys;
        for (std::size_t row = 0; row < candidate_.plc_edges.size() / 2;
             ++row) {
          source_edge_keys.insert(topology::undirected_edge(
              candidate_.plc_edges[2 * row], candidate_.plc_edges[2 * row + 1]));
        }
        std::unordered_map<uint64_t, std::vector<int32_t>> adjacent_rows;
        adjacent_rows.reserve(source_edge_keys.size());
        for (std::size_t row = 0;
             row < candidate_.input_triangles.size() / 3; ++row) {
          const int32_t* triangle =
              candidate_.input_triangles.data() + 3 * row;
          for (int side = 0; side < 3; ++side) {
            const uint64_t edge_key = topology::undirected_edge(
                triangle[side], triangle[(side + 1) % 3]);
            if (source_edge_keys.contains(edge_key)) {
              adjacent_rows[edge_key].push_back(static_cast<int32_t>(row));
            }
          }
        }
        std::set<int32_t> selected_rows;
        std::vector<int32_t> pending;
        for (const auto& [triangle, facet] : seed_parents) {
          if (selected_rows.insert(triangle).second) pending.push_back(triangle);
          (void)facet;
        }
        for (std::size_t at = 0; at < pending.size(); ++at) {
          const int32_t* triangle =
              candidate_.input_triangles.data() +
              3 * static_cast<std::size_t>(pending[at]);
          for (int side = 0; side < 3; ++side) {
            const uint64_t edge_key = topology::undirected_edge(
                triangle[side], triangle[(side + 1) % 3]);
            if (!source_edge_keys.contains(edge_key)) continue;
            const auto adjacent = adjacent_rows.find(edge_key);
            if (adjacent == adjacent_rows.end()) continue;
            for (int32_t neighbor : adjacent->second) {
              if (selected_rows.contains(neighbor)) continue;
              if (static_cast<std::size_t>(neighbor) >=
                  candidate_.input_polygons.size()) {
                return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable,
                            kPlcEdge, edge);
              }
              const int32_t polygon =
                  candidate_.input_polygons[static_cast<std::size_t>(neighbor)];
              if (polygon < 0 || polygon >= input_.polygon_count) {
                return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable,
                            kPlcEdge, edge);
              }
              const int32_t facet = input_.polygon_facets[polygon];
              if (facet < 0 || facet >= input_.facet_count) {
                return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable,
                            kPlcEdge, edge);
              }
              if (input_.facet_regions[2 * facet] !=
                      input_.facet_regions[2 * reference_facet] ||
                  input_.facet_regions[2 * facet + 1] !=
                      input_.facet_regions[2 * reference_facet + 1]) {
                continue;
              }
              const int32_t* candidate_triangle =
                  candidate_.input_triangles.data() +
                  3 * static_cast<std::size_t>(neighbor);
              const double* p0 = point(candidate_triangle[0]);
              const double* p1 = point(candidate_triangle[1]);
              const double* p2 = point(candidate_triangle[2]);
              const double p0_2[2] = {p0[x], p0[y]};
              const double p1_2[2] = {p1[x], p1[y]};
              const double p2_2[2] = {p2[x], p2[y]};
              const int orientation = orient2d(p0_2, p1_2, p2_2);
              if (orientation != reference_orientation ||
                  orient3d(a, b, c, point(candidate_triangle[0])) != 0 ||
                  orient3d(a, b, c, point(candidate_triangle[1])) != 0 ||
                  orient3d(a, b, c, point(candidate_triangle[2])) != 0) {
                continue;
              }
              if (!selected_rows.insert(neighbor).second) continue;
              pending.push_back(neighbor);
              seed_parents.push_back({neighbor, facet});
            }
          }
        }
        std::sort(seed_parents.begin(), seed_parents.end());
        seed_parents.erase(
            std::unique(seed_parents.begin(), seed_parents.end()),
            seed_parents.end());
      }
    }
    if (progress_enabled_) {
      std::fprintf(
          stderr,
          "PLC_PROGRESS incident_patch_seeds edge=%zu direct=%zu expanded=%zu "
          "seconds=%.3f\n",
          edge, direct_seed_count, seed_parents.size(),
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - incident_patch_started)
              .count());
    }
    if (seed_parents.empty()) return PHX_MC_OK;

    const PlcFailure saved_failure = failure_;
    const PlcSourceRefusal saved_source_refusal = candidate_.source_refusal;
    PlcSourceFacePatch patch;
    const bool diagnostic_capture =
        !edge_recovery_dump_written_ && edge_recovery_dump_path_ != nullptr &&
        *edge_recovery_dump_path_ != '\0' &&
        active_source_edge_ == edge_recovery_dump_edge_filter_;
    const auto patch_prepare_started = std::chrono::steady_clock::now();
    if (progress_enabled_) {
      std::fprintf(stderr,
                   "PLC_PROGRESS incident_patch_prepare_begin edge=%zu seeds=%zu\n",
                   edge, seed_parents.size());
    }
    int32_t status = prepare_current_source_face_patch(seed_parents, patch);
    if (progress_enabled_) {
      std::fprintf(
          stderr,
          "PLC_PROGRESS incident_patch_prepare_end edge=%zu status=%d faces=%zu "
          "recovery_faces=%zu cavity=%zu seconds=%.3f\n",
          edge, status, patch.faces.size(), patch.recovery_faces.size(),
          patch.cavity.size(),
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - patch_prepare_started)
              .count());
    }
    if (status == PHX_MC_OK) {
      patch.recovery_edge = {first, second};
      const bool target_face_missing = std::any_of(
          patch.recovery_faces.begin(), patch.recovery_faces.end(),
          [&](const PlcSourceFace& face) {
            return std::find(face.oriented.begin(), face.oriented.end(), first) !=
                       face.oriented.end() &&
                   std::find(face.oriented.begin(), face.oriented.end(), second) !=
                       face.oriented.end();
          });
      if (!target_face_missing) {
        status = PHX_MC_CONSTRAINT_INTERSECTION;
      } else {
        status = recover_source_facet_patch(std::move(patch),
                                            diagnostic_capture, false);
      }
    }
    if (cavity_steiner_restart_required_) return PHX_MC_OK;
    if (status == PHX_MC_OK) {
      bool edge_present = false;
      const int32_t edge_status = has_edge(first, second, edge_present);
      if (edge_status != PHX_MC_OK) return edge_status;
      failure_ = saved_failure;
      candidate_.source_refusal = saved_source_refusal;
      if (!edge_present) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
      }
      recovered = true;
      return PHX_MC_OK;
    }
    if (status != PHX_MC_CONSTRAINT_INTERSECTION) return status;
    failure_ = saved_failure;
    candidate_.source_refusal = saved_source_refusal;

    for (const auto& parent : seed_parents) {
      const auto source = std::find_if(
          source_faces_.begin(), source_faces_.end(),
          [&](const auto& entry) {
            return entry.second.source_triangle == parent.first &&
                   entry.second.facet == parent.second;
          });
      if (source == source_faces_.end()) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
      }
      const int32_t facet_status = recover_current_source_face(
          source->second, input_.policy != BoundaryPolicy::kConforming);
      if (cavity_steiner_restart_required_) return PHX_MC_OK;
      if (facet_status == PHX_MC_OK) {
        bool edge_present = false;
        const int32_t edge_status = has_edge(first, second, edge_present);
        if (edge_status != PHX_MC_OK) return edge_status;
        failure_ = saved_failure;
        candidate_.source_refusal = saved_source_refusal;
        if (!edge_present) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
        }
        recovered = true;
        return PHX_MC_OK;
      }
      if (facet_status == PHX_MC_CONSTRAINT_INTERSECTION) {
        failure_ = saved_failure;
        candidate_.source_refusal = saved_source_refusal;
        continue;
      }
      return facet_status;
    }
    return PHX_MC_OK;
  }

  double source_facet_tolerance(int32_t facet) const {
    return input_.facet_tolerances == nullptr
        ? 0.0 : input_.facet_tolerances[static_cast<std::size_t>(facet)];
  }

  double exact_sqrt_lower(const Expansion& squared) const {
    if (squared.sign() <= 0) return 0.0;
    double value = std::sqrt(std::max(0.0, squared.estimate()));
    if (!std::isfinite(value)) return 0.0;
    while (value > 0.0 &&
           (Expansion::product(value, value) - squared).sign() > 0) {
      value = geometry::next_down(value);
    }
    return value;
  }

  double exact_sqrt_upper(const Expansion& squared) const {
    if (squared.sign() <= 0) return 0.0;
    double value = std::sqrt(std::max(0.0, squared.estimate()));
    if (!std::isfinite(value)) return 0.0;
    while ((Expansion::product(value, value) - squared).sign() < 0) {
      value = geometry::next_up(value);
    }
    return value;
  }

  bool exact_point_segment_distance_at_least(
      const double* query, const double* first, const double* second,
      double radius) const {
    std::array<Expansion, 3> direction, offset;
    for (int axis = 0; axis < 3; ++axis) {
      direction[static_cast<std::size_t>(axis)] =
          Expansion::difference(second[axis], first[axis]);
      offset[static_cast<std::size_t>(axis)] =
          Expansion::difference(query[axis], first[axis]);
    }
    const auto dot = [](const auto& a, const auto& b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    const Expansion length_squared = dot(direction, direction);
    const Expansion radius_squared = Expansion::product(radius, radius);
    const Expansion projection = dot(offset, direction);
    if (length_squared.sign() == 0 || projection.sign() <= 0) {
      return (dot(offset, offset) - radius_squared).sign() >= 0;
    }
    if ((projection - length_squared).sign() >= 0) {
      std::array<Expansion, 3> end_offset;
      for (int axis = 0; axis < 3; ++axis) {
        end_offset[static_cast<std::size_t>(axis)] =
            Expansion::difference(query[axis], second[axis]);
      }
      return (dot(end_offset, end_offset) - radius_squared).sign() >= 0;
    }
    const Expansion numerator =
        dot(offset, offset) * length_squared - projection * projection -
        radius_squared * length_squared;
    return numerator.sign() >= 0;
  }

  bool exact_point_triangle_distance_at_least(
      const double* query, const double* a, const double* b, const double* c,
      double radius) const {
    std::array<Expansion, 3> u, v, w;
    for (int axis = 0; axis < 3; ++axis) {
      u[static_cast<std::size_t>(axis)] = Expansion::difference(b[axis], a[axis]);
      v[static_cast<std::size_t>(axis)] = Expansion::difference(c[axis], a[axis]);
      w[static_cast<std::size_t>(axis)] = Expansion::difference(query[axis], a[axis]);
    }
    const auto dot = [](const auto& x, const auto& y) {
      return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    };
    const std::array<Expansion, 3> normal{
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]};
    const Expansion normal_squared = dot(normal, normal);
    const Expansion uu = dot(u, u), uv = dot(u, v), vv = dot(v, v);
    const Expansion wu = dot(w, u), wv = dot(w, v);
    const Expansion determinant = uu * vv - uv * uv;
    if (determinant.sign() <= 0 || normal_squared.sign() <= 0) return false;
    const Expansion first_numerator = wu * vv - wv * uv;
    const Expansion second_numerator = wv * uu - wu * uv;
    const Expansion third_numerator =
        determinant - first_numerator - second_numerator;
    if (first_numerator.sign() >= 0 && second_numerator.sign() >= 0 &&
        third_numerator.sign() >= 0) {
      const Expansion height = dot(normal, w);
      const Expansion radius_squared = Expansion::product(radius, radius);
      return (height * height - radius_squared * normal_squared).sign() >= 0;
    }
    return exact_point_segment_distance_at_least(query, a, b, radius) &&
           exact_point_segment_distance_at_least(query, b, c, radius) &&
           exact_point_segment_distance_at_least(query, c, a, radius);
  }

  bool point_box_distance_at_least(const double* query, const Box& box,
                                   double radius) const {
    for (int axis = 0; axis < 3; ++axis) {
      if (query[axis] < box.lower[axis] &&
          (Expansion::difference(box.lower[axis], query[axis]) -
           Expansion(radius)).sign() >= 0) {
        return true;
      }
      if (query[axis] > box.upper[axis] &&
          (Expansion::difference(query[axis], box.upper[axis]) -
           Expansion(radius)).sign() >= 0) {
        return true;
      }
    }
    return false;
  }

  // Exact area and an upper edge-length bound give a conservative local
  // feature scale from the source triangle's minimum altitude.
  double source_face_steiner_lfs_lower_bound(int32_t source_triangle) const {
    if (source_triangle < 0 ||
        static_cast<std::size_t>(source_triangle) >=
            candidate_.input_triangles.size() / 3) {
      return 0.0;
    }
    const int32_t* face =
        candidate_.input_triangles.data() + 3 * static_cast<std::size_t>(source_triangle);
    const double* a = point(face[0]);
    const double* b = point(face[1]);
    const double* c = point(face[2]);
    std::array<Expansion, 3> u, v;
    for (int axis = 0; axis < 3; ++axis) {
      u[static_cast<std::size_t>(axis)] = Expansion::difference(b[axis], a[axis]);
      v[static_cast<std::size_t>(axis)] = Expansion::difference(c[axis], a[axis]);
    }
    const std::array<Expansion, 3> normal{
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]};
    const Expansion area_squared =
        normal[0] * normal[0] + normal[1] * normal[1] +
        normal[2] * normal[2];
    const std::array<Expansion, 3> ab{
        Expansion::difference(b[0], a[0]), Expansion::difference(b[1], a[1]),
        Expansion::difference(b[2], a[2])};
    const std::array<Expansion, 3> bc{
        Expansion::difference(c[0], b[0]), Expansion::difference(c[1], b[1]),
        Expansion::difference(c[2], b[2])};
    const std::array<Expansion, 3> ca{
        Expansion::difference(a[0], c[0]), Expansion::difference(a[1], c[1]),
        Expansion::difference(a[2], c[2])};
    const auto squared_length = [](const auto& edge) {
      return edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2];
    };
    const double max_edge = std::max(
        {exact_sqrt_upper(squared_length(ab)),
         exact_sqrt_upper(squared_length(bc)),
         exact_sqrt_upper(squared_length(ca))});
    const double area_lower = exact_sqrt_lower(area_squared);
    if (!(max_edge > 0.0) || !(area_lower > 0.0)) return 0.0;
    const double min_altitude_lower =
        geometry::next_down(area_lower / (2.0 * max_edge));
    return geometry::next_down(min_altitude_lower / 16.0);
  }

  bool source_face_steiner_features_clear(
      const double* candidate, int32_t source_triangle, int32_t source_facet,
      double local_feature_lower, double point_spacing) const {
    (void)source_facet;
    const Expansion spacing_squared =
        Expansion::product(point_spacing, point_spacing);
    for (std::size_t vertex = 0; vertex < work_points_.size() / 3; ++vertex) {
      const double* existing = work_points_.data() + 3 * vertex;
      const Expansion dx = Expansion::difference(candidate[0], existing[0]);
      const Expansion dy = Expansion::difference(candidate[1], existing[1]);
      const Expansion dz = Expansion::difference(candidate[2], existing[2]);
      const Expansion distance_squared = dx * dx + dy * dy + dz * dz;
      if ((distance_squared - spacing_squared).sign() < 0) return false;
    }
    for (std::size_t triangle = 0;
         triangle < candidate_.input_triangles.size() / 3; ++triangle) {
      if (triangle == static_cast<std::size_t>(source_triangle)) continue;
      const int32_t* vertices =
          candidate_.input_triangles.data() + 3 * triangle;
      const Box box = edge_recovery_box(vertices, 3);
      if (point_box_distance_at_least(candidate, box, local_feature_lower)) {
        continue;
      }
      if (!exact_point_triangle_distance_at_least(
              candidate, point(vertices[0]), point(vertices[1]),
              point(vertices[2]), local_feature_lower)) {
        return false;
      }
    }
    for (std::size_t edge = 0; edge < edge_chains_.size(); ++edge) {
      const auto& chain = edge_chains_[edge];
      if (chain.size() < 2 || !is_source_edge(edge)) continue;
      for (std::size_t interval = 0; interval + 1 < chain.size(); ++interval) {
        const int32_t first = chain[interval].vertex;
        const int32_t second = chain[interval + 1].vertex;
        const int32_t segment[2] = {first, second};
        const Box box = edge_recovery_box(segment, 2);
        if (point_box_distance_at_least(candidate, box, local_feature_lower)) {
          continue;
        }
        if (!exact_point_segment_distance_at_least(
                candidate, point(first), point(second), local_feature_lower)) {
          return false;
        }
      }
    }
    (void)source_triangle;
    return true;
  }

  bool find_source_face_steiner_location(
      const PlcSourceFace& source, double local_feature_lower,
      double point_spacing, double* parameters, double* carrier,
      double& deviation) const {
    const int32_t* original =
        candidate_.input_triangles.data() +
        3 * static_cast<std::size_t>(source.source_triangle);
    const double* corners[3] = {
        point(original[0]), point(original[1]), point(original[2])};
    const double* child[3] = {
        point(source.oriented[0]), point(source.oriented[1]),
        point(source.oriented[2])};
    // h_lower=area_lower/(2*max_edge) is at most half the minimum altitude;
    // the barycentric core is at least h_min/4 from its own edges. Thus
    // lfs=h_lower/16 is a positive protected feature-size lower bound. The
    // affine map has sigma_min>=h_lower=16*lfs; denominator 128 gives ideal
    // sites at least lfs/8 apart, then exact rounded-distance checks require
    // the stronger lfs separation used by the packing argument.
    constexpr int64_t denominator = 128;
    const int64_t minimum = denominator / 4;
    for (int64_t first = minimum; first <= denominator / 2; ++first) {
      for (int64_t second = minimum;
           second <= denominator - first - minimum; ++second) {
        const double candidate_parameters[2] = {
            static_cast<double>(first) / denominator,
            static_cast<double>(second) / denominator};
        if (!source_parameters_valid(SourceStratum::kFacet,
                                     candidate_parameters)) {
          continue;
        }
        const double third = 1.0 - candidate_parameters[0] -
                             candidate_parameters[1];
        if (third < 0.25) continue;
        double candidate_deviation = 0.0;
        if (!source_carrier(corners, SourceStratum::kFacet,
                            candidate_parameters, carrier,
                            candidate_deviation) ||
            candidate_deviation > source_facet_tolerance(source.facet)) {
          continue;
        }
        double witness_parameters[2] = {0.0, 0.0};
        source_locator(corners, SourceStratum::kFacet, carrier,
                      witness_parameters, false);
        if (!source_parameters_valid(SourceStratum::kFacet,
                                     witness_parameters) ||
            witness_parameters[0] < 0.25 ||
            witness_parameters[1] < 0.25 ||
            1.0 - witness_parameters[0] - witness_parameters[1] < 0.25) {
          continue;
        }
        const double witness_deviation = source_deviation(
            corners, SourceStratum::kFacet, witness_parameters, carrier);
        if (witness_deviation < 0.0 ||
            witness_deviation > source_facet_tolerance(source.facet)) {
          continue;
        }
        int side = 0;
        if (locate_on_triangle(carrier, child, side) != kFeatureInterior ||
            !source_face_steiner_features_clear(
                carrier, source.source_triangle, source.facet,
                local_feature_lower, point_spacing)) {
          continue;
        }
        parameters[0] = witness_parameters[0];
        parameters[1] = witness_parameters[1];
        deviation = witness_deviation;
        return true;
      }
    }
    return false;
  }

  // Splits the missing subface `source`, a child of source triangle
  // T = ABC, at the exactly witnessed point S = A + l1 (B - A) + l2 (C - A).
  // (l1, l2) are the barycentric coordinates of the child's centroid,
  // rounded to dyadic rationals of increasing precision until S lies
  // exactly in the child's open interior; S keeps its exact position in the
  // exact vertex bank (its carrier is the rounding of S), so S is exactly
  // coplanar with every subface of T and later recovery and suppression
  // decide on S itself. The child becomes the fan from S over its boundary,
  // which includes the split points on its edges.
  int32_t split_source_face(const Triple& key, const PlcSourceFace& source, bool& restart) {
    restart = false;
    const auto refuse = [&]() {
      return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcFacet, source.facet);
    };
    const int32_t source_count = static_cast<int32_t>(candidate_.points.size() / 3);
    std::vector<int32_t> boundary;
    for (int side = 0; side < 3; ++side) {
      const int32_t first = source.oriented[static_cast<std::size_t>(side)];
      const int32_t second = source.oriented[static_cast<std::size_t>((side + 1) % 3)];
      boundary.push_back(first);
      std::vector<int32_t> interior;
      for (int32_t vertex = 0; vertex < source_count; ++vertex) {
        if (vertex == first || vertex == second) continue;
        if (!collinear(ref(first), ref(second), ref(vertex)) ||
            !between_on_line(ref(first), ref(second), ref(vertex))) {
          continue;
        }
        interior.push_back(vertex);
      }
      // Order along the edge by the exact projection onto it.
      std::sort(interior.begin(), interior.end(), [&](int32_t x, int32_t y) {
        Expansion along;
        for (int axis = 0; axis < 3; ++axis) {
          along = along + (coordinate(ref(y), axis) - coordinate(ref(x), axis)) *
                              (coordinate(ref(second), axis) - coordinate(ref(first), axis));
        }
        return along.sign() > 0;
      });
      boundary.insert(boundary.end(), interior.begin(), interior.end());
    }
    const int32_t* original = candidate_.input_triangles.data() + 3 * source.source_triangle;
    const double* corners[3] = {
        candidate_.points.data() + 3 * static_cast<std::size_t>(original[0]),
        candidate_.points.data() + 3 * static_cast<std::size_t>(original[1]),
        candidate_.points.data() + 3 * static_cast<std::size_t>(original[2])};
    // Barycentric coordinates of the child's carrier centroid in T, by the
    // normal equations of the affine map (only a target for the search).
    double centroid[3] = {0.0, 0.0, 0.0};
    for (int32_t vertex : source.oriented) {
      for (int axis = 0; axis < 3; ++axis) centroid[axis] += point(vertex)[axis] / 3.0;
    }
    double u[3], v[3], w[3];
    geometry::difference(corners[1], corners[0], u);
    geometry::difference(corners[2], corners[0], v);
    geometry::difference(centroid, corners[0], w);
    const double uu = geometry::dot(u, u), uv = geometry::dot(u, v), vv = geometry::dot(v, v);
    const double wu = geometry::dot(w, u), wv = geometry::dot(w, v);
    const double determinant = uu * vv - uv * uv;
    if (!(determinant > 0.0)) return refuse();
    const double target[2] = {(wu * vv - wv * uv) / determinant, (wv * uu - wu * uv) / determinant};
    const ExactPointRef child[3] = {ref(source.oriented[0]), ref(source.oriented[1]),
                                    ref(source.oriented[2])};
    double parameters[2];
    Expansion position[3];
    double carrier[3];
    bool found = false;
    for (int bits = 4; bits <= 52 && !found; ++bits) {
      const double scale = std::exp2(bits);
      parameters[0] = std::round(target[0] * scale) / scale;
      parameters[1] = std::round(target[1] * scale) / scale;
      if (!source_parameters_valid(SourceStratum::kFacet, parameters) ||
          !source_point(corners, SourceStratum::kFacet, parameters, position)) {
        continue;
      }
      for (int axis = 0; axis < 3; ++axis) carrier[axis] = nearest_double(position[axis]);
      if (!geometry::point_in_domain(carrier)) continue;
      int side = 0;
      found = locate_on_triangle(ExactPointRef{carrier, position}, child, side) ==
              kFeatureInterior;
    }
    if (!found) return refuse();
    const double deviation = outward_distance(carrier, position);
    // Distinct exact points with equal carriers would be indistinguishable.
    const ExactPointRef split{carrier, deviation == 0.0 ? nullptr : position};
    for (int32_t vertex = 0; vertex < source_count; ++vertex) {
      if (!geometry::same_point(carrier, point(vertex))) continue;
      if (same_point(split, ref(vertex))) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcFacet, source.facet,
                    kPlcPoint, vertex);
      }
      return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcFacet, source.facet, kPlcPoint,
                  vertex);
    }
    if (source_count >= kMaxMeshPoints - corner_count_) {
      return fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcFacet, source.facet);
    }
    const int32_t center = source_count;
    std::map<Triple, PlcSourceFace> staged_source_faces;
    for (std::size_t side = 0; side < boundary.size(); ++side) {
      const Triple child_face{boundary[side], boundary[(side + 1) % boundary.size()], center};
      const Triple child_key = sorted_triple(child_face.data());
      if (source_faces_.contains(child_key) ||
          !staged_source_faces
               .emplace(child_key, PlcSourceFace{child_face, source.facet, source.source_triangle})
               .second) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcFacet, source.facet);
      }
    }
    candidate_.points.insert(candidate_.points.end(), carrier, carrier + 3);
    PlcRecoveryWitness witness{};
    witness.stratum = SourceStratum::kFacet;
    witness.entity = source.source_triangle;
    witness.parameters[0] = parameters[0];
    witness.parameters[1] = parameters[1];
    witness.deviation = deviation;
    candidate_.witnesses.push_back(witness);
    candidate_.vertex_dimension.push_back(2);
    if (deviation != 0.0) exact_points_.assign(center, position);
    source_faces_.erase(key);
    source_faces_.merge(staged_source_faces);
    const int32_t insertion_status = insert_source_vertex_incrementally(center);
    if (insertion_status != PHX_MC_OK) return insertion_status;
    restart = true;
    return PHX_MC_OK;
  }

  // Recovers every subface: present ones are marked; a missing one is
  // recovered by in-plane flips when its facet plane is covered, else split
  // at an exactly witnessed interior point (split_source_face) and the scan
  // restarts. FIXED boundaries take the same path; their split points are
  // removed again by suppression.
  int32_t recover_source_faces(bool& restart) {
    restart = false;
    std::map<Triple, std::vector<PlcFaceUse>> faces;
    TetrahedralComplex& complex = triangulation_->complex();
    const auto mark_source_faces = [&]() -> int32_t {
      for (const auto& [key, source] : source_faces_) {
        const auto found = faces.find(key);
        if (found == faces.end() || found->second.size() != 2) {
          return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcFacet, source.facet);
        }
        const PlcFaceUse* finite[2] = {nullptr, nullptr};
        int finite_count = 0;
        for (const PlcFaceUse& use : found->second) {
          if (!is_ghost(complex.tets[static_cast<std::size_t>(use.tet)])) {
            if (finite_count < 2) finite[finite_count] = &use;
            ++finite_count;
          }
        }
        if (finite_count == 2) {
          const bool first_positive =
              FacetKey::of(finite[0]->oriented.data()) == FacetKey::of(source.oriented.data());
          const bool second_positive =
              FacetKey::of(finite[1]->oriented.data()) == FacetKey::of(source.oriented.data());
          if (first_positive == second_positive) {
            return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet, source.facet);
          }
        } else if (finite_count != 1 || corner_count_ != 0) {
          return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcFacet, source.facet);
        }
        complex.set_facet_constraint(finite[0]->tet, finite[0]->slot, source.facet);
      }
      return PHX_MC_OK;
    };

    for (;;) {
      faces.clear();
      for (std::size_t t = 0; t < complex.tets.size(); ++t) {
        const Tetrahedron& tet = complex.tets[t];
        if (tet.v[0] == kDeadVertex) continue;
        for (int slot = 0; slot < 4; ++slot) {
          Triple face{};
          oriented_facet(tet.v, slot, face.data());
          faces[sorted_triple(face.data())].push_back({static_cast<int32_t>(t), slot, face});
        }
      }
      const int32_t planar_status = recover_planar_polygons(faces);
      if (planar_status != PHX_MC_OK) return planar_status;
      bool source_face_recovered = false;
      for (const auto& [key, source] : source_faces_) {
        const auto found = faces.find(key);
        if (found != faces.end() && found->second.size() == 2) continue;
        const PlcFailure prior_failure = failure_;
        bool planar_recovered = false;
        const Triple planar_key = key;
        const PlcSourceFace planar_source = source;
        const int32_t planar_status =
            recover_planar_source_face(planar_key, planar_source, planar_recovered);
        if (planar_recovered) {
          source_face_recovered = true;
          break;
        }
        if (planar_status != PHX_MC_OK) {
          if (planar_status != PHX_MC_CONSTRAINT_INTERSECTION ||
              failure_.reason != kPlcFixedFacet) {
            return planar_status;
          }
          failure_ = prior_failure;
        }
        return split_source_face(planar_key, planar_source, restart);
      }
      if (source_face_recovered) continue;
      return mark_source_faces();
    }
  }

  // Steiner suppression (cf. Si, TetGen boundary recovery with -Y): once
  // all facets are recovered, every boundary split point is removed again
  // when the policy is FIXED, and otherwise every split point whose exact
  // position is not a binary64 point, so published boundary vertices lie
  // exactly on their sources. The recovered subfaces through such a point s
  // are retriangulated without s within their own planar polygons (as the
  // fans from a neighbour t when s contracts onto t, else by exact ear
  // clipping), so the two chain intervals at a segment point merge into one,
  // and every wedge of the star of s is refilled by remove_witnessed_vertex.
  // Each suppression removes one boundary vertex and adds at most one volume
  // point per wedge, so the pass is finite. A point whose fans or wedges
  // admit none of the exactly verified replacements is a typed
  // STEINER_SUPPRESSION refusal naming its source segment or facet.
  struct SuppressionFan {
    int32_t facet = -1;
    int32_t source_triangle = -1;
    bool planar = false;
    Triple reference{};
    std::vector<Triple> old_faces;
    std::vector<Triple> new_faces;
  };

  int32_t suppress_witnessed_vertices() {
    const int32_t count = static_cast<int32_t>(candidate_.points.size() / 3);
    suppressed_.assign(static_cast<std::size_t>(count), 0);
    for (int32_t vertex = source_point_count_; vertex < count; ++vertex) {
      if (input_.policy != BoundaryPolicy::kFixed && exact_points_.position(vertex) == nullptr) {
        continue;
      }
      const int32_t status = suppress_vertex(vertex);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  // Half-fans of the recovered subfaces through s, split at the chain
  // neighbours p and q, each retriangulated without s.
  bool suppression_fans(int32_t s, int32_t p, int32_t q, std::vector<SuppressionFan>& fans) const {
    struct Use {
      Triple oriented;
      int32_t source_triangle;
      bool planar;
    };
    std::map<int32_t, std::vector<Use>> by_facet;
    for (const auto* faces : {&source_faces_, &planar_source_faces_}) {
      const bool planar = faces == &planar_source_faces_;
      for (const auto& [key, source] : *faces) {
        if (std::find(key.begin(), key.end(), s) == key.end()) continue;
        by_facet[source.facet].push_back({source.oriented, source.source_triangle, planar});
      }
    }
    fans.clear();
    for (const auto& [facet, uses] : by_facet) {
      std::map<int32_t, std::size_t> successor;
      std::set<int32_t> targets;
      std::vector<Triple> rotated;
      for (const Use& use : uses) {
        Triple face = use.oriented;
        while (face[0] != s) std::rotate(face.begin(), face.begin() + 1, face.end());
        if (!successor.emplace(face[1], rotated.size()).second) return false;
        targets.insert(face[2]);
        rotated.push_back(face);
      }
      std::vector<int32_t> starts;
      for (const auto& [vertex, unused] : successor) {
        (void)unused;
        if (!targets.contains(vertex)) starts.push_back(vertex);
      }
      // A full disk of one facet around s is cut into its two half-fans at
      // the chain neighbours p and q of a segment point; around a facet
      // point (p < 0) it is one closed loop.
      const bool disk = starts.empty();
      if (disk) starts = p >= 0 ? std::vector<int32_t>{p, q} : std::vector<int32_t>{rotated[0][1]};
      std::vector<char> used(rotated.size(), 0);
      for (int32_t start : starts) {
        SuppressionFan fan;
        fan.facet = facet;
        fan.planar = true;
        std::vector<int32_t> loop{start};
        for (;;) {
          const auto next = successor.find(loop.back());
          if (next == successor.end() || used[next->second] != 0) break;
          used[next->second] = 1;
          const Triple& face = rotated[next->second];
          const Use& use = uses[next->second];
          // A half-fan lies in one source triangle (or one planar polygon).
          if (fan.old_faces.empty()) fan.source_triangle = use.source_triangle;
          fan.planar = fan.planar && use.planar;
          fan.old_faces.push_back(face);
          loop.push_back(face[2]);
          if (p >= 0 && (loop.back() == p || loop.back() == q)) break;
        }
        const int32_t end = loop.back();
        if (p < 0) {
          if (!disk || end != start) return false;
          loop.pop_back();
        } else if (fan.old_faces.empty() ||
                   !((start == p && end == q) || (start == q && end == p))) {
          return false;
        }
        if (fan.old_faces.empty()) return false;
        fan.reference = fan.old_faces.front();
        const int axis = exact_projection_axis(fan.reference[0], fan.reference[1],
                                               fan.reference[2]);
        if (axis < 0 || loop.size() < 3) return false;
        const int orientation = orient2d_projected(
            ref(fan.reference[0]), ref(fan.reference[1]), ref(fan.reference[2]), axis);
        if (!exact_loop_triangles(loop, axis, orientation, fan.new_faces)) return false;
        fans.push_back(std::move(fan));
      }
      if (std::find(used.begin(), used.end(), 0) != used.end()) return false;
    }
    return p < 0 ? fans.size() == 1 : !fans.empty();
  }

  // Makes the star of s star-shaped from t where 2-3 flips allow: a cell
  // (s, a, b, c) whose link face abc does not strictly face t, with abc
  // unconstrained and a finite neighbour (a, b, c, d), is replaced by the
  // three cells around the new edge sd when all three are exactly positive
  // (sd crosses the open triangle abc). Each flip replaces one link face of
  // s by three faces through d and keeps a valid complex; the pass is bounded
  // by four times the number of cells around s at entry.
  int32_t flip_star_toward(int32_t s, int32_t t) {
    TetrahedralComplex& complex = triangulation_->complex();
    std::vector<int32_t> cells;
    int32_t status = vertex_cells(s, cells);
    if (status != PHX_MC_OK) return status;
    std::size_t budget = 4 * cells.size();
    while (budget-- > 0) {
      bool flipped = false;
      for (int32_t cell : cells) {
        const Tetrahedron tet = complex.tets[static_cast<std::size_t>(cell)];
        if (is_ghost(tet) || vertex_slot(tet, t) >= 0) continue;
        const int own = vertex_slot(tet, s);
        std::array<int32_t, 4> collapsed{tet.v[0], tet.v[1], tet.v[2], tet.v[3]};
        collapsed[static_cast<std::size_t>(own)] = t;
        if (orient(collapsed[0], collapsed[1], collapsed[2], collapsed[3]) > 0) continue;
        if (complex.constraint(cell, own) != kNoConstraint) continue;
        const int32_t other = tet.n[own];
        if (!complex.live(other)) continue;
        const Tetrahedron neighbor = complex.tets[static_cast<std::size_t>(other)];
        if (is_ghost(neighbor)) continue;
        const int back = neighbor_slot(neighbor, cell);
        if (back < 0) continue;
        const int32_t d = neighbor.v[back];
        // Three cells: `tet` with one link vertex replaced by d each; the
        // face opposite d is tet's, the face opposite s the neighbour's.
        std::array<SuppressionCell, 3> created{};
        int count = 0;
        bool valid = true;
        for (int slot = 0; slot < 4 && valid; ++slot) {
          if (slot == own) continue;
          SuppressionCell next{{tet.v[0], tet.v[1], tet.v[2], tet.v[3]},
                               {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint}};
          next.v[static_cast<std::size_t>(slot)] = d;
          valid = orient(next.v[0], next.v[1], next.v[2], next.v[3]) > 0;
          next.constraints[static_cast<std::size_t>(slot)] = complex.constraint(cell, slot);
          next.constraints[static_cast<std::size_t>(own)] =
              complex.constraint(other, vertex_slot(neighbor, tet.v[slot]));
          created[static_cast<std::size_t>(count++)] = next;
        }
        if (!valid) continue;
        CavityEdit edit(complex);
        edit.begin();
        if (edit.remove(cell) != EditStatus::kOk || edit.remove(other) != EditStatus::kOk) continue;
        for (const SuppressionCell& next : created) {
          edit.add(next.v.data(), next.constraints.data(), kNoRegion);
        }
        EditStatus edit_status = edit.validate([&](const int32_t* vertices) {
          return orient(vertices[0], vertices[1], vertices[2], vertices[3]) > 0;
        });
        if (edit_status == EditStatus::kOk) edit_status = edit.commit();
        if (edit_status != EditStatus::kOk) {
          edit.rollback();
          if (edit_status == EditStatus::kSlotLimit) {
            return fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcPoint, s);
          }
          continue;
        }
        local_edits_ = true;
        star_marks_.resize(complex.tets.size(), 0U);
        for (int32_t id : edit.created()) {
          for (int32_t vertex : complex.tets[static_cast<std::size_t>(id)].v) {
            vertex_tet_[static_cast<std::size_t>(vertex)] = id;
          }
        }
        if (!triangulation_->set_hint(edit.created().front())) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, s);
        }
        flipped = true;
        break;
      }
      if (!flipped) return PHX_MC_OK;
      status = vertex_cells(s, cells);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  // A cell of a suppression proposal and the source facet marks of its faces.
  struct SuppressionCell {
    std::array<int32_t, 4> v;
    std::array<int32_t, 4> constraints;
  };

  // Steiner-free fill of a closed cavity by exact gift wrapping. Each
  // boundary cell carries `placeholder` in one slot: the face opposite it
  // is a boundary face and a cavity vertex x lies on its inner side exactly
  // when the cell with x substituted is positive. Sides of faces are keyed
  // by the face oriented as its cell on that side sees it, so a face may
  // bound the cavity from both sides (a slit along an open sheet). An open
  // side is closed by a cavity vertex on its inner side; each other face of
  // the new cell either closes an open side, or opens the far side of a new
  // face, and may neither repeat a closed side nor lie outside a boundary
  // face. The side with the fewest admissible apices is closed first,
  // apices in increasing order of the number of rivals inside their
  // circumsphere (the Delaunay choice first) after the `preferred` apex,
  // with bounded backtracking. A finished fill is a set of positive cells
  // with every face paired and the cavity boundary as boundary, hence the
  // 3-chain of the cavity: the cells tile it.
  bool wrap_cavity(const std::vector<SuppressionCell>& boundary, int32_t placeholder,
                   int32_t preferred, const std::vector<int32_t>& interior,
                   std::vector<SuppressionCell>& filled) const {
    struct Open {
      std::array<int32_t, 4> cell;
      int slot;
      int32_t constraint;
    };
    const auto side_of = [](const std::array<int32_t, 4>& cell, int slot) {
      int32_t face[3];
      oriented_facet(cell.data(), slot, face);
      return FacetKey::of(face);
    };
    const auto with = [](std::array<int32_t, 4> cell, int slot, int32_t vertex) {
      cell[static_cast<std::size_t>(slot)] = vertex;
      return cell;
    };
    const auto positive = [&](const std::array<int32_t, 4>& v) {
      return orient(v[0], v[1], v[2], v[3]) > 0;
    };
    std::map<FacetKey, Open> front;
    std::set<FacetKey> outside;
    std::set<int32_t> vertices;
    for (const SuppressionCell& cell : boundary) {
      const auto found = std::find(cell.v.begin(), cell.v.end(), placeholder);
      if (found == cell.v.end()) return false;
      const int slot = static_cast<int>(found - cell.v.begin());
      const FacetKey side = side_of(cell.v, slot);
      if (!front.emplace(side, Open{cell.v, slot, cell.constraints[static_cast<std::size_t>(slot)]})
               .second) {
        return false;
      }
      outside.insert(side.reversed());
      for (int k = 0; k < 4; ++k) {
        if (k != slot) vertices.insert(cell.v[static_cast<std::size_t>(k)]);
      }
    }
    // Interior candidates: volume points that may become apices.
    vertices.insert(interior.begin(), interior.end());
    std::set<FacetKey> closed;
    filled.clear();
    std::size_t budget = 4096;
    std::function<bool()> search = [&]() -> bool {
      if (front.empty()) return true;
      if (budget == 0) return false;
      --budget;
      FacetKey key{};
      std::vector<int32_t> candidates;
      bool chosen = false;
      for (const auto& [side, open] : front) {
        std::vector<int32_t> admissible;
        for (int32_t vertex : vertices) {
          if (vertex == side.a || vertex == side.b || vertex == side.c) continue;
          if (positive(with(open.cell, open.slot, vertex))) admissible.push_back(vertex);
        }
        if (admissible.empty()) return false;
        if (!chosen || admissible.size() < candidates.size()) {
          key = side;
          candidates = std::move(admissible);
          chosen = true;
        }
      }
      const Open open = front.at(key);
      std::vector<std::pair<int, int32_t>> order;
      for (int32_t vertex : candidates) {
        const std::array<int32_t, 4> tet = with(open.cell, open.slot, vertex);
        int inside = 0;
        for (int32_t rival : candidates) {
          if (rival == vertex) continue;
          inside += insphere_sos(ref(tet[0]), ref(tet[1]), ref(tet[2]), ref(tet[3]), ref(rival),
                                 tet[0], tet[1], tet[2], tet[3], rival) > 0 ? 1 : 0;
        }
        // The preferred apex (the contraction target) goes first.
        order.push_back({vertex == preferred ? -1 : inside, vertex});
      }
      std::sort(order.begin(), order.end());
      for (const auto& [unused, vertex] : order) {
        (void)unused;
        const std::array<int32_t, 4> tet = with(open.cell, open.slot, vertex);
        SuppressionCell cell{tet, {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint}};
        cell.constraints[static_cast<std::size_t>(open.slot)] = open.constraint;
        std::vector<FacetKey> matched;
        std::vector<FacetKey> sealed;
        std::vector<std::pair<FacetKey, Open>> opened;
        bool valid = true;
        for (int slot = 0; slot < 4 && valid; ++slot) {
          if (slot == open.slot) continue;
          const FacetKey side = side_of(tet, slot);
          const auto other = front.find(side);
          if (other != front.end()) {
            matched.push_back(side);
            cell.constraints[static_cast<std::size_t>(slot)] = other->second.constraint;
            continue;
          }
          if (closed.contains(side) || outside.contains(side)) {
            valid = false;
            break;
          }
          sealed.push_back(side);
          const FacetKey far = side.reversed();
          if (closed.contains(far) || front.contains(far)) continue;
          // The cell across this face lies on its other side.
          std::array<int32_t, 4> flipped = with(tet, slot, placeholder);
          std::swap(flipped[static_cast<std::size_t>((slot + 1) % 4)],
                    flipped[static_cast<std::size_t>((slot + 2) % 4)]);
          opened.push_back({far, Open{flipped, slot, kNoConstraint}});
        }
        if (!valid) continue;
        front.erase(key);
        closed.insert(key);
        std::vector<Open> saved;
        for (const FacetKey& side : matched) {
          saved.push_back(front.at(side));
          front.erase(side);
          closed.insert(side);
        }
        for (const FacetKey& side : sealed) closed.insert(side);
        for (const auto& [side, entry] : opened) front.emplace(side, entry);
        filled.push_back(cell);
        if (search()) return true;
        filled.pop_back();
        for (const auto& [side, entry] : opened) {
          (void)entry;
          front.erase(side);
        }
        for (const FacetKey& side : sealed) closed.erase(side);
        for (std::size_t row = 0; row < matched.size(); ++row) {
          closed.erase(matched[row]);
          front.emplace(matched[row], saved[row]);
        }
        closed.erase(key);
        front.emplace(key, open);
        if (budget == 0) return false;
      }
      return false;
    };
    return search();
  }

  // A cell equal to `cell` except that the face opposite `slot` is `face`,
  // seen from the cell with the same orientation as `cell` sees its own face
  // there when `same` holds, reversed otherwise.
  static std::array<int32_t, 4> transplant(const int32_t* cell, int slot, const Triple& face,
                                           bool same) {
    const Triple wanted = same ? face : Triple{face[0], face[2], face[1]};
    std::array<int32_t, 4> result{cell[0], cell[1], cell[2], cell[3]};
    int others[3];
    int count = 0;
    for (int k = 0; k < 4; ++k) {
      if (k != slot) others[count++] = k;
    }
    for (int k = 0; k < 3; ++k) {
      result[static_cast<std::size_t>(others[k])] = wanted[static_cast<std::size_t>(k)];
    }
    Triple seen{};
    oriented_facet(result.data(), slot, seen.data());
    if (!(FacetKey::of(seen.data()) == FacetKey::of(wanted.data()))) {
      std::swap(result[static_cast<std::size_t>(others[0])],
                result[static_cast<std::size_t>(others[1])]);
    }
    return result;
  }

  // Removes the witnessed vertex s from the complex. With t >= 0 every
  // recovered subface (s, a, b) becomes (t, a, b) (an edge contraction;
  // subfaces through t vanish), otherwise each half-fan through s is ear
  // clipped without s (suppression_fans with chain neighbours p, q, or the
  // full disk of a facet point when p < 0). The star of s splits into wedges
  // (cells joined across faces through s that are not subfaces). A wedge is
  // replaced by its contraction onto t when every finite cell stays
  // positive; a hull wedge of ghost cells over subfaces by ghost cells over
  // the new subfaces; any other finite wedge by a Steiner-free gift-wrapped
  // fill of its cavity (link faces and new subfaces), else by the cones from
  // one binary64 point y = s + l d (d the sum of the inner unit normals of
  // its subface planes, l halving) over that cavity boundary. Every finite
  // proposed cell is exactly positive on exact positions and the cavity edit
  // checks that the proposal's boundary is the link of s with its marks, so
  // the proposal tiles the star. Otherwise nothing changes.
  int32_t remove_witnessed_vertex(int32_t s, int32_t t, int32_t p, int32_t q, bool& removed) {
    removed = false;
    TetrahedralComplex& complex = triangulation_->complex();
    if (t >= 0) {
      const int32_t flip_status = flip_star_toward(s, t);
      if (flip_status != PHX_MC_OK) return flip_status;
    }
    std::vector<int32_t> cells;
    const int32_t cells_status = vertex_cells(s, cells);
    if (cells_status != PHX_MC_OK) return cells_status;
    struct NewFace {
      Triple oriented;
      int32_t facet;
      int32_t source_triangle;
      bool planar;
    };
    struct OldFace {
      Triple oriented;
      std::size_t group;
    };
    std::map<Triple, OldFace> old_faces;
    std::vector<std::vector<NewFace>> groups;
    if (t >= 0) {
      for (const auto* faces : {&source_faces_, &planar_source_faces_}) {
        const bool planar = faces == &planar_source_faces_;
        for (const auto& [key, face] : *faces) {
          if (std::find(key.begin(), key.end(), s) == key.end()) continue;
          old_faces.emplace(key, OldFace{face.oriented, groups.size()});
          groups.emplace_back();
          if (std::find(key.begin(), key.end(), t) != key.end()) continue;
          Triple mapped = face.oriented;
          std::replace(mapped.begin(), mapped.end(), s, t);
          groups.back().push_back({mapped, face.facet, face.source_triangle, planar});
        }
      }
    } else {
      std::vector<SuppressionFan> fans;
      if (!suppression_fans(s, p, q, fans)) return PHX_MC_OK;
      for (const SuppressionFan& fan : fans) {
        for (const Triple& face : fan.old_faces) {
          old_faces.emplace(sorted_triple(face.data()), OldFace{face, groups.size()});
        }
        groups.emplace_back();
        for (const Triple& face : fan.new_faces) {
          groups.back().push_back({face, fan.facet, fan.source_triangle, fan.planar});
        }
      }
    }

    std::unordered_map<int32_t, std::size_t> row_of;
    for (std::size_t row = 0; row < cells.size(); ++row) row_of.emplace(cells[row], row);
    std::vector<std::size_t> parent(cells.size());
    std::iota(parent.begin(), parent.end(), 0);
    const auto root = [&](std::size_t row) {
      while (parent[row] != row) row = parent[row] = parent[parent[row]];
      return row;
    };
    struct Contact {
      std::size_t row;
      int slot;
      const OldFace* face;
    };
    std::vector<Contact> contacts;
    for (std::size_t row = 0; row < cells.size(); ++row) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cells[row])];
      const int own = vertex_slot(tet, s);
      for (int slot = 0; slot < 4; ++slot) {
        if (slot == own) continue;
        Triple face{};
        oriented_facet(tet.v, slot, face.data());
        const auto old = old_faces.find(sorted_triple(face.data()));
        if (old != old_faces.end()) {
          contacts.push_back({row, slot, &old->second});
          continue;
        }
        const auto neighbor = row_of.find(tet.n[slot]);
        if (neighbor == row_of.end()) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, s);
        }
        parent[root(row)] = root(neighbor->second);
      }
    }
    std::map<std::size_t, std::vector<std::size_t>> wedges;
    for (std::size_t row = 0; row < cells.size(); ++row) wedges[root(row)].push_back(row);

    using Proposal = SuppressionCell;
    constexpr int32_t kPlaceholder = std::numeric_limits<int32_t>::min() / 2;
    const auto positive = [&](const std::array<int32_t, 4>& v, const ExactPointRef* center) {
      if (std::find(v.begin(), v.end(), kGhostVertex) != v.end()) return true;
      ExactPointRef corners[4];
      for (int slot = 0; slot < 4; ++slot) {
        corners[slot] = v[static_cast<std::size_t>(slot)] == kPlaceholder
            ? *center : ref(v[static_cast<std::size_t>(slot)]);
      }
      return orient3d(corners[0], corners[1], corners[2], corners[3]) > 0;
    };
    // Cells over the new subfaces of the groups met by a wedge, once per
    // group, oriented like the wedge cell that meets the group.
    const auto subface_cells = [&](const std::vector<std::size_t>& rows) {
      std::vector<Proposal> result;
      std::set<std::pair<std::size_t, bool>> emitted;
      for (const Contact& contact : contacts) {
        if (root(contact.row) != root(rows.front())) continue;
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cells[contact.row])];
        Triple seen{};
        oriented_facet(tet.v, contact.slot, seen.data());
        const bool same = FacetKey::of(seen.data()) == FacetKey::of(contact.face->oriented.data());
        // A wedge meets a group from both sides along an open sheet.
        if (!emitted.insert({contact.face->group, same}).second) continue;
        for (const NewFace& face : groups[contact.face->group]) {
          Proposal cell{transplant(tet.v, contact.slot, face.oriented, same),
                        {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint}};
          if (tet.v[contact.slot] != kGhostVertex) cell.v[static_cast<std::size_t>(contact.slot)] = kPlaceholder;
          cell.constraints[static_cast<std::size_t>(contact.slot)] = face.facet;
          result.push_back(cell);
        }
      }
      return result;
    };
    std::vector<Proposal> proposals;
    std::vector<std::array<double, 3>> centers;
    std::vector<int32_t> added;
    // Volume points appended for a replacement that is not committed die.
    const auto abandon = [&]() {
      for (int32_t vertex : added) alive_[static_cast<std::size_t>(vertex)] = 0;
      return PHX_MC_OK;
    };
    for (const auto& [unused, rows] : wedges) {
      (void)unused;
      bool ghost = false;
      std::vector<std::size_t> blocked;
      for (std::size_t row : rows) {
        ghost = ghost || is_ghost(complex.tets[static_cast<std::size_t>(cells[row])]);
        if (t < 0) blocked.push_back(row);
      }
      if (t >= 0) {
        std::vector<Proposal> collapse;
        bool star_shaped = true;
        for (std::size_t row : rows) {
          const int32_t cell = cells[row];
          const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
          if (vertex_slot(tet, t) >= 0) continue;
          Proposal proposal{};
          for (int slot = 0; slot < 4; ++slot) {
            proposal.v[static_cast<std::size_t>(slot)] = tet.v[slot] == s ? t : tet.v[slot];
            proposal.constraints[static_cast<std::size_t>(slot)] = complex.constraint(cell, slot);
          }
          if (!positive(proposal.v, nullptr)) {
            star_shaped = false;
            blocked.push_back(row);
          }
          collapse.push_back(proposal);
        }
        if (star_shaped) {
          proposals.insert(proposals.end(), collapse.begin(), collapse.end());
          continue;
        }
      }
      if (ghost) {
        // A hull wedge: ghost cells, each over one subface through s.
        std::size_t covered = 0;
        for (const Contact& contact : contacts) {
          const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cells[contact.row])];
          if (root(contact.row) == root(rows.front()) && tet.v[contact.slot] == kGhostVertex) {
            ++covered;
          }
        }
        if (covered != rows.size()) return abandon();
        const std::vector<Proposal> hull = subface_cells(rows);
        proposals.insert(proposals.end(), hull.begin(), hull.end());
        continue;
      }
      // The cavity boundary: link faces (s replaced by the placeholder) and
      // the new subfaces.
      std::vector<Proposal> cavity = subface_cells(rows);
      std::vector<std::array<double, 3>> normals;
      double reach = std::numeric_limits<double>::infinity();
      for (std::size_t row : rows) {
        const int32_t cell = cells[row];
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
        const int own = vertex_slot(tet, s);
        Proposal link{{tet.v[0], tet.v[1], tet.v[2], tet.v[3]},
                      {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint}};
        link.v[static_cast<std::size_t>(own)] = kPlaceholder;
        link.constraints[static_cast<std::size_t>(own)] = complex.constraint(cell, own);
        cavity.push_back(link);
        for (int32_t vertex : tet.v) {
          if (vertex != s) {
            reach = std::min(reach, std::sqrt(geometry::squared_distance(point(s), point(vertex))));
          }
        }
      }
      for (const Contact& contact : contacts) {
        if (root(contact.row) != root(rows.front())) continue;
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cells[contact.row])];
        Triple face{};
        oriented_facet(tet.v, contact.slot, face.data());
        double u[3], w[3], normal[3], offset[3];
        geometry::difference(point(face[1]), point(face[0]), u);
        geometry::difference(point(face[2]), point(face[0]), w);
        geometry::cross(u, w, normal);
        geometry::difference(point(tet.v[contact.slot]), point(s), offset);
        const double length = std::sqrt(geometry::dot(normal, normal));
        if (!(length > 0.0)) continue;
        const double sign = geometry::dot(normal, offset) < 0.0 ? -1.0 : 1.0;
        const std::array<double, 3> unit{sign * normal[0] / length, sign * normal[1] / length,
                                         sign * normal[2] / length};
        if (std::none_of(normals.begin(), normals.end(), [&](const auto& known) {
              return known[0] * unit[0] + known[1] * unit[1] + known[2] * unit[2] > 1.0 - 0x1p-30;
            })) {
          normals.push_back(unit);
        }
      }
      std::vector<Proposal> wrapped;
      if (wrap_cavity(cavity, kPlaceholder, t, {}, wrapped)) {
        proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
        continue;
      }
      // Steiner-assisted fill: interior points of the cells that block the
      // contraction (every cell without a target) join the apex candidates.
      std::vector<int32_t> extra;
      for (std::size_t row : blocked) {
        if (root(row) != root(rows.front()) || extra.size() == 8) continue;
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cells[row])];
        double centroid[3] = {0.0, 0.0, 0.0};
        for (int32_t vertex : tet.v) {
          for (int axis = 0; axis < 3; ++axis) centroid[axis] += 0.25 * point(vertex)[axis];
        }
        if (!geometry::point_in_domain(centroid)) continue;
        int32_t vertex = -1;
        const int32_t status = add_volume_point(centroid, vertex);
        if (status != PHX_MC_OK) return status;
        added.push_back(vertex);
        extra.push_back(vertex);
      }
      if (!extra.empty() && wrap_cavity(cavity, kPlaceholder, t, extra, wrapped)) {
        proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
        continue;
      }
      double direction[3] = {0.0, 0.0, 0.0};
      for (const auto& unit : normals) {
        for (int axis = 0; axis < 3; ++axis) direction[axis] += unit[static_cast<std::size_t>(axis)];
      }
      const double norm = std::sqrt(geometry::dot(direction, direction));
      if (normals.empty() || !(norm > 0.0) || !std::isfinite(reach) || !(reach > 0.0)) {
        return abandon();
      }
      bool found = false;
      double location[3];
      for (double step = 0.25 * reach; step > 0x1p-40 * reach && !found; step *= 0.5) {
        for (int axis = 0; axis < 3; ++axis) {
          location[axis] = point(s)[axis] + step * direction[axis] / norm;
        }
        if (!geometry::point_in_domain(location)) continue;
        const ExactPointRef center{location, nullptr};
        found = std::all_of(cavity.begin(), cavity.end(), [&](const Proposal& cone) {
          return positive(cone.v, &center);
        });
      }
      if (!found) return abandon();
      // Placeholder ids of later centers stay distinct until appended.
      for (Proposal& cone : cavity) {
        std::replace(cone.v.begin(), cone.v.end(), kPlaceholder,
                     kPlaceholder + 1 + static_cast<int32_t>(centers.size()));
      }
      centers.push_back({location[0], location[1], location[2]});
      proposals.insert(proposals.end(), cavity.begin(), cavity.end());
    }

    for (std::size_t row = 0; row < centers.size(); ++row) {
      int32_t vertex = -1;
      const int32_t status = add_volume_point(centers[row].data(), vertex);
      if (status != PHX_MC_OK) return status;
      added.push_back(vertex);
      for (Proposal& proposal : proposals) {
        std::replace(proposal.v.begin(), proposal.v.end(),
                     kPlaceholder + 1 + static_cast<int32_t>(row), vertex);
      }
    }
    // Subfaces through s are interior to the removed star; their marks move
    // to the proposal, which carries them on the new subfaces.
    std::vector<std::array<int32_t, 3>> released;
    for (int32_t cell : cells) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      for (int slot = 0; slot < 4; ++slot) {
        if (tet.v[slot] == s) continue;
        const int32_t constraint = complex.constraint(cell, slot);
        if (constraint == kNoConstraint) continue;
        released.push_back({cell, slot, constraint});
        complex.set_facet_constraint(cell, slot, kNoConstraint);
      }
    }
    CavityEdit edit(complex);
    edit.begin();
    for (int32_t cell : cells) {
      if (edit.remove(cell) != EditStatus::kOk) {
        for (const auto& [mark_cell, slot, constraint] : released) {
          complex.set_facet_constraint(mark_cell, slot, constraint);
        }
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, s);
      }
    }
    for (const Proposal& proposal : proposals) {
      edit.add(proposal.v.data(), proposal.constraints.data(), kNoRegion);
    }
    EditStatus edit_status = edit.validate([&](const int32_t* vertices) {
      return orient(vertices[0], vertices[1], vertices[2], vertices[3]) > 0;
    });
    if (edit_status == EditStatus::kOk) edit_status = edit.commit();
    if (edit_status != EditStatus::kOk) {
      edit.rollback();
      for (const auto& [mark_cell, slot, constraint] : released) {
        complex.set_facet_constraint(mark_cell, slot, constraint);
      }
      for (int32_t vertex : added) alive_[static_cast<std::size_t>(vertex)] = 0;
      return edit_status == EditStatus::kSlotLimit
          ? fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcPoint, s)
          : PHX_MC_OK;
    }
    local_edits_ = true;
    star_marks_.resize(complex.tets.size(), 0U);
    int32_t hint = -1;
    for (int32_t id : edit.created()) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
      if (is_ghost(tet)) continue;
      if (hint < 0) hint = id;
      for (int32_t vertex : tet.v) vertex_tet_[static_cast<std::size_t>(vertex)] = id;
    }
    if (hint < 0 || !triangulation_->set_hint(hint)) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, s);
    }
    for (const auto& [key, unused] : old_faces) {
      (void)unused;
      source_faces_.erase(key);
      planar_source_faces_.erase(key);
    }
    for (const std::vector<NewFace>& group : groups) {
      for (const NewFace& face : group) {
        auto& faces = face.planar ? planar_source_faces_ : source_faces_;
        faces.emplace(sorted_triple(face.oriented.data()),
                      PlcSourceFace{face.oriented, face.facet, face.source_triangle});
      }
    }
    alive_[static_cast<std::size_t>(s)] = 0;
    vertex_tet_[static_cast<std::size_t>(s)] = -1;
    suppressed_[static_cast<std::size_t>(s)] = 1;
    for (int32_t vertex : added) {
      const bool used = std::any_of(proposals.begin(), proposals.end(), [&](const Proposal& cell) {
        return std::find(cell.v.begin(), cell.v.end(), vertex) != cell.v.end();
      });
      if (!used) alive_[static_cast<std::size_t>(vertex)] = 0;
    }
    removed = true;
    return PHX_MC_OK;
  }

  // Suppresses one witnessed vertex: contraction onto a chain neighbour (a
  // segment point) or a disk neighbour (a facet point), else the ear-clipped
  // half-fans. A point that cannot be removed is a STEINER_SUPPRESSION
  // refusal naming its source segment or facet; under FIXED it is the
  // FIXED_SEGMENT / FIXED_FACET refusal of that entity (its boundary could
  // not be recovered unmodified).
  int32_t suppress_vertex(int32_t s) {
    const PlcRecoveryWitness& witness = candidate_.witnesses[static_cast<std::size_t>(s)];
    const bool segment = witness.stratum == SourceStratum::kSegment;
    const auto refuse = [&]() {
      const bool fixed = input_.policy == BoundaryPolicy::kFixed;
      const int32_t status = fixed ? PHX_MC_CONSTRAINT_INTERSECTION : PHX_MC_UNRESOLVED;
      if (segment) {
        return fail(status, fixed ? kPlcFixedSegment : kPlcSteinerSuppression, kPlcEdge,
                    static_cast<int64_t>(witness.entity), kPlcPoint, s);
      }
      const int32_t facet = input_.polygon_facets[
          candidate_.input_polygons[static_cast<std::size_t>(witness.entity)]];
      return fail(status, fixed ? kPlcFixedFacet : kPlcSteinerSuppression, kPlcFacet, facet,
                  kPlcPoint, s);
    };
    int32_t p = -1;
    int32_t q = -1;
    std::vector<int32_t> targets;
    std::size_t position = 0;
    if (segment) {
      auto& chain = edge_chains_[static_cast<std::size_t>(witness.entity)];
      position = 1;
      while (position + 1 < chain.size() && chain[position].vertex != s) ++position;
      if (position + 1 >= chain.size()) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, s);
      }
      p = chain[position - 1].vertex;
      q = chain[position + 1].vertex;
      targets = {p, q};
    } else {
      std::set<int32_t> ring;
      for (const auto* faces : {&source_faces_, &planar_source_faces_}) {
        for (const auto& [key, unused] : *faces) {
          (void)unused;
          if (std::find(key.begin(), key.end(), s) == key.end()) continue;
          for (int32_t vertex : key) {
            if (vertex != s) ring.insert(vertex);
          }
        }
      }
      targets.assign(ring.begin(), ring.end());
    }
    targets.push_back(-1);
    for (int32_t target : targets) {
      bool removed = false;
      const int32_t status = remove_witnessed_vertex(s, target, p, q, removed);
      if (status != PHX_MC_OK) return status;
      if (!removed) continue;
      if (segment) {
        auto& chain = edge_chains_[static_cast<std::size_t>(witness.entity)];
        chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(position));
      }
      return PHX_MC_OK;
    }
    return refuse();
  }

  int32_t seed_tet(const double* seed, int64_t row) {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
      int signs[4];
      bool inside = true;
      for (int slot = 0; slot < 4; ++slot) {
        signs[slot] = triangulation_->orient_with(tet, slot, seed);
        inside = inside && signs[slot] >= 0;
      }
      if (!inside) continue;
      for (int slot = 0; slot < 4; ++slot) {
        if (signs[slot] == 0 && complex.constraint(static_cast<int32_t>(t), slot) != kNoConstraint) {
          fail(PHX_MC_INVALID_INPUT, kPlcInvalidSeed, kPlcSeed, row);
          return -1;
        }
      }
      return static_cast<int32_t>(t);
    }
    fail(PHX_MC_INVALID_INPUT, kPlcInvalidSeed, kPlcSeed, row);
    return -1;
  }

  int32_t label_conflict(int32_t first, int32_t second, int32_t facet) {
    if (first == kPlcVoidRegion || second == kPlcVoidRegion) {
      return fail(PHX_MC_INVALID_INPUT, kPlcRegionLeak, kPlcRegion,
                  first == kPlcVoidRegion ? second : first, kPlcFacet, facet);
    }
    return fail(PHX_MC_INVALID_INPUT, kPlcInconsistentRegions,
                kPlcRegion, std::min(first, second), kPlcRegion, std::max(first, second));
  }

  int32_t classify_regions() {
    TetrahedralComplex& complex = triangulation_->complex();
    complex.clear_regions();
    std::vector<int32_t> queue;
    const auto label_of = [](int32_t region) { return region < 0 ? kPlcVoidRegion : region; };
    const auto assign = [&](int32_t tet, int32_t region, int32_t facet) -> int32_t {
      const int32_t current = complex.region(tet);
      if (current != kNoRegion) return current == region ? PHX_MC_OK : label_conflict(current, region, facet);
      complex.set_region(tet, region);
      queue.push_back(tet);
      return PHX_MC_OK;
    };
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t facet = complex.constraint(static_cast<int32_t>(t), slot);
        if (facet == kNoConstraint) continue;
        int32_t face[3];
        oriented_facet(tet.v, slot, face);
        const Triple key = sorted_triple(face);
        const PlcSourceFace* source = nullptr;
        const auto original_source = source_faces_.find(key);
        if (original_source != source_faces_.end()) {
          source = &original_source->second;
        } else {
          const auto planar_source = planar_source_faces_.find(key);
          if (planar_source != planar_source_faces_.end()) source = &planar_source->second;
        }
        if (source == nullptr) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet, facet);
        }
        const bool positive = FacetKey::of(face) == FacetKey::of(source->oriented.data());
        const int32_t region = input_.facet_regions[2 * facet + (positive ? 0 : 1)];
        const int32_t status = assign(static_cast<int32_t>(t), label_of(region), facet);
        if (status != PHX_MC_OK) return status;
      }
      for (int32_t vertex : tet.v) {
        if (artificial_corner_vertex(vertex)) {
          const int32_t status = assign(static_cast<int32_t>(t), kPlcVoidRegion, -1);
          if (status != PHX_MC_OK) return status;
          break;
        }
      }
    }
    for (int64_t seed = 0; seed < input_.seed_count; ++seed) {
      const int32_t tet = seed_tet(input_.seeds + 3 * seed, seed);
      if (tet < 0) return PHX_MC_INVALID_INPUT;
      const int32_t status = assign(tet, label_of(input_.seed_regions[seed]), -1);
      if (status != PHX_MC_OK) return status;
    }
    std::size_t head = 0;
    while (head < queue.size()) {
      const int32_t t = queue[head++];
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(t)];
      for (int slot = 0; slot < 4; ++slot) {
        if (complex.constraint(t, slot) != kNoConstraint) continue;
        const int32_t next = tet.n[slot];
        const Tetrahedron& neighbor = complex.tets[static_cast<std::size_t>(next)];
        if (neighbor.v[0] == kDeadVertex || is_ghost(neighbor)) continue;
        const int32_t label = complex.region(next);
        if (label == kNoRegion) {
          complex.set_region(next, complex.region(t));
          queue.push_back(next);
        } else if (label != complex.region(t)) {
          return label_conflict(label, complex.region(t), -1);
        }
      }
    }
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] != kDeadVertex && !is_ghost(tet) && complex.region(static_cast<int32_t>(t)) == kNoRegion) {
        return fail(PHX_MC_INVALID_INPUT, kPlcRegionLeak, kPlcRegion, -1);
      }
    }
    return PHX_MC_OK;
  }

  int32_t check_embedded_entities() {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (std::size_t edge = 0; edge < edge_chains_.size(); ++edge) {
      for (std::size_t interval = 0; interval + 1 < edge_chains_[edge].size(); ++interval) {
        const int32_t first = edge_chains_[edge][interval].vertex;
        const int32_t second = edge_chains_[edge][interval + 1].vertex;
        bool in_domain = false;
        for (std::size_t t = 0; t < complex.tets.size(); ++t) {
          const Tetrahedron& tet = complex.tets[t];
          if (tet.v[0] != kDeadVertex && !is_ghost(tet) &&
              complex.region(static_cast<int32_t>(t)) >= 0 &&
              vertex_slot(tet, first) >= 0 && vertex_slot(tet, second) >= 0) {
            in_domain = true;
            break;
          }
        }
        if (!in_domain) return fail(PHX_MC_INVALID_INPUT, kPlcOutsideDomain, kPlcEdge, edge);
      }
    }
    std::vector<uint8_t> referenced(static_cast<std::size_t>(source_point_count_), 0);
    for (int32_t vertex : candidate_.input_triangles) referenced[static_cast<std::size_t>(vertex)] = 1;
    for (int64_t row = 0; row < input_.segment_count; ++row) {
      referenced[static_cast<std::size_t>(input_.segments[2 * row])] = 1;
      referenced[static_cast<std::size_t>(input_.segments[2 * row + 1])] = 1;
    }
    for (int32_t vertex = 0; vertex < source_point_count_; ++vertex) {
      if (referenced[static_cast<std::size_t>(vertex)] != 0) continue;
      bool in_domain = false;
      for (std::size_t t = 0; t < complex.tets.size(); ++t) {
        const Tetrahedron& tet = complex.tets[t];
        if (tet.v[0] != kDeadVertex && !is_ghost(tet) && complex.region(static_cast<int32_t>(t)) >= 0 &&
            vertex_slot(tet, vertex) >= 0) {
          in_domain = true;
          break;
        }
      }
      if (!in_domain) return fail(PHX_MC_INVALID_INPUT, kPlcOutsideDomain, kPlcPoint, vertex);
    }
    return PHX_MC_OK;
  }

  void publish() {
    const TetrahedralComplex& complex = triangulation_->complex();
    candidate_.tets.clear();
    candidate_.tet_regions.clear();
    candidate_.faces.clear();
    candidate_.face_sources.clear();
    candidate_.face_orientations.clear();
    candidate_.segments.clear();
    candidate_.segment_sources.clear();
    candidate_.tets.reserve(static_cast<std::size_t>(complex.finite_count) * 4);
    candidate_.tet_regions.reserve(static_cast<std::size_t>(complex.finite_count));
    // Suppressed split points are dropped; volume Steiner points are
    // published only when a domain cell uses them, in ascending work-vertex
    // order after the source and boundary points.
    const std::size_t candidate_points = candidate_.points.size() / 3;
    point_output_.assign(candidate_points, -1);
    std::size_t published_points = 0;
    for (std::size_t vertex = 0; vertex < candidate_points; ++vertex) {
      if (vertex < suppressed_.size() && suppressed_[vertex] != 0) continue;
      point_output_[vertex] = static_cast<int32_t>(published_points++);
    }
    const int32_t first_volume = static_cast<int32_t>(candidate_points) + corner_count_;
    volume_output_.assign(volume_count_, -1);
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet) || complex.region(static_cast<int32_t>(t)) < 0) continue;
      for (int32_t vertex : tet.v) {
        if (vertex >= first_volume) volume_output_[static_cast<std::size_t>(vertex - first_volume)] = 0;
      }
    }
    std::vector<std::size_t> published_volume;
    for (std::size_t row = 0; row < volume_output_.size(); ++row) {
      if (volume_output_[row] < 0) continue;
      volume_output_[row] = static_cast<int32_t>(published_points + published_volume.size());
      published_volume.push_back(row);
    }
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex || is_ghost(tet) || complex.region(static_cast<int32_t>(t)) < 0) continue;
      int32_t output_tet[4];
      for (int slot = 0; slot < 4; ++slot) {
        output_tet[slot] = output_vertex(tet.v[slot]);
        if (output_tet[slot] < 0) {
          status_ = fail(PHX_MC_INVALID_INPUT, kPlcRegionLeak, kPlcRegion, complex.region(static_cast<int32_t>(t)));
          return;
        }
      }
      candidate_.tets.insert(candidate_.tets.end(), output_tet, output_tet + 4);
      candidate_.tet_regions.push_back(complex.region(static_cast<int32_t>(t)));
    }
    const auto publish_face = [&](const PlcSourceFace& source) {
      for (int32_t vertex : source.oriented) {
        candidate_.faces.push_back(point_output_[static_cast<std::size_t>(vertex)]);
      }
      candidate_.face_sources.push_back(source.facet);
      candidate_.face_orientations.push_back(1);
    };
    for (const auto& [key, source] : source_faces_) {
      (void)key;
      publish_face(source);
    }
    for (const auto& [key, source] : planar_source_faces_) {
      (void)key;
      publish_face(source);
    }
    std::size_t segment_count = 0;
    for (const auto& chain : edge_chains_)
      if (chain.size() > 1) segment_count += chain.size() - 1;
    candidate_.segments.reserve(2 * segment_count);
    candidate_.segment_sources.reserve(segment_count);
    for (std::size_t edge = 0; edge < edge_chains_.size(); ++edge) {
      const auto& chain = edge_chains_[edge];
      if (chain.size() < 2) continue;
      for (std::size_t interval = 0; interval + 1 < chain.size(); ++interval) {
        candidate_.segments.push_back(
            point_output_[static_cast<std::size_t>(chain[interval].vertex)]);
        candidate_.segments.push_back(
            point_output_[static_cast<std::size_t>(chain[interval + 1].vertex)]);
        candidate_.segment_sources.push_back(static_cast<int32_t>(edge));
      }
    }
    std::size_t kept = 0;
    for (std::size_t vertex = 0; vertex < candidate_points; ++vertex) {
      if (point_output_[vertex] < 0) continue;
      std::copy_n(candidate_.points.begin() + static_cast<std::ptrdiff_t>(3 * vertex), 3,
                  candidate_.points.begin() + static_cast<std::ptrdiff_t>(3 * kept));
      candidate_.witnesses[kept] = candidate_.witnesses[vertex];
      candidate_.vertex_dimension[kept] = candidate_.vertex_dimension[vertex];
      ++kept;
    }
    candidate_.points.resize(3 * kept);
    candidate_.witnesses.resize(kept);
    candidate_.vertex_dimension.resize(kept);
    candidate_.protection = protection_radii(
        candidate_.points.data(), static_cast<int64_t>(published_points),
        candidate_.segments, candidate_.input_triangles);
    for (std::size_t row : published_volume) {
      const double* coordinates =
          point(first_volume + static_cast<int32_t>(row));
      candidate_.points.insert(candidate_.points.end(), coordinates, coordinates + 3);
      candidate_.witnesses.push_back(PlcRecoveryWitness{});
      candidate_.vertex_dimension.push_back(3);
      candidate_.protection.push_back(0.0);
    }
    candidate_.failure = {};
  }

  const PlcInput& input_;
  PlcRecovery candidate_;
  int32_t status_ = PHX_MC_OK;
  int32_t source_point_count_ = 0;
  int32_t corner_count_ = 0;
  PlcFailure failure_{};
  std::vector<double> work_points_;
  std::vector<char> alive_;
  std::unique_ptr<Triangulation3D> triangulation_;
  bool incremental_triangulation_ready_ = false;
  std::size_t progress_full_rebuilds_ = 0;
  std::size_t progress_incremental_attempts_ = 0;
  std::size_t progress_incremental_accepted_ = 0;
  std::size_t progress_face_steiner_insertions_ = 0;
  std::vector<int32_t> vertex_tet_;
  std::vector<std::uint32_t> star_marks_;
  std::vector<int32_t> star_;
  std::uint32_t star_stamp_ = 0;
  std::map<Triple, PlcSourceFace> source_faces_;
  std::vector<PlcPlanarPolygon> planar_polygons_;
  std::vector<int32_t> planar_polygon_index_;
  std::map<Triple, PlcSourceFace> planar_source_faces_;
  std::vector<double> corner_points_;
  std::vector<std::vector<PlcEdgePoint>> edge_chains_;
  bool trace_enabled_ = std::getenv("PHYDRAX_PLC_TRACE") != nullptr;
  const std::size_t trace_edge_filter_ =
      trace_index_from_environment("PHYDRAX_PLC_TRACE_EDGE");
  const std::size_t trace_growth_filter_ =
      trace_index_from_environment("PHYDRAX_PLC_TRACE_GROWTH");
  bool progress_enabled_ = std::getenv("PHYDRAX_PLC_PROGRESS") != nullptr;
  const char* edge_recovery_dump_path_ =
      std::getenv("PHYDRAX_PLC_WRAP_DUMP");
  const std::size_t edge_recovery_dump_edge_filter_ =
      trace_index_from_environment("PHYDRAX_PLC_WRAP_DUMP_EDGE");
  const std::size_t edge_recovery_dump_threshold_ =
      diagnostic_threshold_from_environment("PHYDRAX_PLC_WRAP_DUMP_THRESHOLD",
                                            100000);
  std::size_t active_source_edge_ = std::numeric_limits<std::size_t>::max();
  mutable bool edge_recovery_dump_written_ = false;
  std::size_t active_cavity_growth_round_ =
      std::numeric_limits<std::size_t>::max();
  std::vector<int32_t> source_segment_rows_;
  std::unordered_set<uint64_t> polygon_loop_edges_;
  std::vector<std::size_t> edge_triangle_offsets_;
  std::vector<int32_t> edge_triangle_rows_;
  // Acute input vertices: apices of the concentric split shells.
  std::vector<char> shell_apex_;
  // Exact positions of witnessed split vertices, keyed by work vertex.
  ExactVertexBank exact_points_;
  // Current chain intervals, which recovery edits must never remove.
  std::unordered_set<uint64_t> chain_intervals_;
  std::vector<std::array<int32_t, 2>> protected_segments_;
  std::array<int32_t, 2> recovery_edge_{-1, -1};
  bool cavity_steiner_restart_required_ = false;
  std::size_t volume_count_ = 0;
  std::vector<int32_t> volume_output_;
  // True once a local recovery edit has left the complex non-Delaunay.
  bool local_edits_ = false;
  // Candidate points moved off the boundary by Steiner suppression.
  std::vector<char> suppressed_;
  // Published index of each candidate point (-1 when suppressed).
  std::vector<int32_t> point_output_;
};

}  // namespace

namespace detail {

int32_t recover_plc_wrap_cavity(const PlcWrapCavityInput& input,
                                PlcWrapCavityResult& result) {
  try {
    return PlcRecoveryBuilder::run_wrap_fixture(input, result);
  } catch (const std::bad_alloc&) {
    result = PlcWrapCavityResult{};
    return PHX_MC_ALLOCATION_FAILURE;
  }
}

}  // namespace detail

int32_t recover_plc(const PlcInput& input, PlcRecovery& result) {
  try {
    PlcRecovery prepared;
    const int32_t source_status = prepare_plc_source(input, prepared);
    if (source_status != PHX_MC_OK) {
      result.failure = prepared.failure;
      return source_status;
    }
    PlcRecoveryBuilder builder(input, prepared);
    const int32_t status = builder.run(result);
    if (status != PHX_MC_OK) {
      result.failure = builder.failure();
      result.source_refusal = builder.source_refusal();
    }
    return status;
  } catch (const std::bad_alloc&) {
    result.failure = {};
    return PHX_MC_ALLOCATION_FAILURE;
  }
}

}  // namespace phx::mc

