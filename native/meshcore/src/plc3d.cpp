//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact PLC validation, triangulation, and source-entity cavity recovery.
// Cavity broad phases follow current adjacency through inclusive AABB overlaps;
// exact segment/face or triangle/face contacts alone decide membership.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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

// The cells using one face: at most two in a valid complex; `count` keeps
// counting past two so callers can reject an invalid one.
struct PlcFaceUses {
  std::array<PlcFaceUse, 2> uses{};
  std::size_t count = 0;
  void push_back(const PlcFaceUse& use) {
    if (count < 2) uses[count] = use;
    ++count;
  }
  std::size_t size() const { return count; }
  const PlcFaceUse* begin() const { return uses.data(); }
  const PlcFaceUse* end() const { return uses.data() + std::min<std::size_t>(count, 2); }
};

struct TripleHash {
  std::size_t operator()(const Triple& key) const {
    const auto word = [](int32_t value) { return static_cast<std::uint64_t>(static_cast<uint32_t>(value)); };
    const std::uint64_t mixed = (word(key[0]) * 0x9E3779B97F4A7C15ULL) ^
                                (word(key[1]) * 0xC2B2AE3D27D4EB4FULL) ^
                                (word(key[2]) * 0x165667B19E3779F9ULL);
    return static_cast<std::size_t>(mixed ^ (mixed >> 29));
  }
};

// Faces of the current complex keyed by sorted vertex triple.
using PlcFaceMap = std::unordered_map<Triple, PlcFaceUses, TripleHash>;

struct PlcSourceFace {
  Triple oriented;
  int32_t facet;
  int32_t source_triangle;
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
    for (;;) {
      const bool reuse_incremental = incremental_triangulation_ready_;
      incremental_triangulation_ready_ = false;
      if (!reuse_incremental) {
        status_ = build_triangulation();
        if (status_ != PHX_MC_OK) return status_;
      }
      bool restart = false;
      status_ = recover_source_edges(restart);
      if (status_ != PHX_MC_OK) return status_;
      if (restart) continue;
      status_ = recover_source_faces(restart);
      if (status_ != PHX_MC_OK) return status_;
      if (!restart) break;
    }
    status_ = suppress_witnessed_vertices();
    if (status_ != PHX_MC_OK) return status_;
    status_ = classify_regions();
    if (status_ != PHX_MC_OK) return status_;
    status_ = check_embedded_entities();
    if (status_ != PHX_MC_OK) return status_;
    publish();
    if (status_ != PHX_MC_OK) return status_;
    output = std::move(candidate_);
    return PHX_MC_OK;
  }

  const PlcSourceRefusal& source_refusal() const { return candidate_.source_refusal; }
  const PlcFailure& failure() const { return failure_; }

  static int32_t run_cavity_fixture(const detail::SuppressionCavity& input,
                                    detail::SuppressionCavityFill& result) {
    result = detail::SuppressionCavityFill{};
    const std::size_t witnesses = input.witnessed.size();
    if (input.points.size() % 3 != 0 || input.witness_strata.size() != witnesses ||
        input.witness_corners.size() != 9 * witnesses ||
        input.witness_parameters.size() != 2 * witnesses ||
        input.walls.size() != input.wall_marks.size() || input.placeholder >= 0) {
      return PHX_MC_INVALID_INPUT;
    }
    const std::size_t point_count = input.points.size() / 3;
    const auto valid_vertex = [&](int32_t vertex) {
      return vertex >= 0 && static_cast<std::size_t>(vertex) < point_count;
    };
    PlcInput empty_input;
    PlcRecovery empty_source;
    PlcRecoveryBuilder builder(empty_input, empty_source);
    builder.work_points_.assign(input.points.begin(), input.points.end());
    for (std::size_t row = 0; row < witnesses; ++row) {
      if (!valid_vertex(input.witnessed[row])) return PHX_MC_INVALID_INPUT;
      const double* corners[3] = {input.witness_corners.data() + 9 * row,
                                  input.witness_corners.data() + 9 * row + 3,
                                  input.witness_corners.data() + 9 * row + 6};
      Expansion position[3];
      if (!source_point(corners, input.witness_strata[row],
                        input.witness_parameters.data() + 2 * row, position)) {
        return PHX_MC_INVALID_INPUT;
      }
      builder.exact_points_.assign(input.witnessed[row], position);
    }
    std::vector<SuppressionCell> boundary;
    boundary.reserve(input.walls.size());
    for (std::size_t row = 0; row < input.walls.size(); ++row) {
      if (std::count(input.walls[row].begin(), input.walls[row].end(), input.placeholder) != 1) {
        return PHX_MC_INVALID_INPUT;
      }
      for (int32_t vertex : input.walls[row]) {
        if (vertex != input.placeholder && !valid_vertex(vertex)) return PHX_MC_INVALID_INPUT;
      }
      boundary.push_back({input.walls[row], input.wall_marks[row]});
    }
    std::vector<SuppressionCell> filled;
    result.delaunay =
        builder.wrap_cavity(boundary, input.placeholder, {}, false, filled, result.nodes);
    result.filled = result.delaunay ||
        builder.wrap_cavity(boundary, input.placeholder, {}, true, filled, result.nodes);
    if (result.filled) {
      for (const SuppressionCell& cell : filled) result.cells.push_back(cell.v);
    }
    return PHX_MC_OK;
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
      const PlcFaceMap& faces) {
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
    const int32_t insert_status = triangulation_->insert(vertex, alive_);
    if (insert_status != PHX_MC_OK) {
      // Checked insertion refuses, leaving the complex unchanged, when the
      // conflict cavity is not star-shaped from the vertex (the complex is no
      // longer Delaunay after local recovery edits) or crosses a constrained
      // facet. The vertex then goes into a shrunk star-shaped cavity; only
      // when that is refused too does the next restart rebuild.
      if (insert_status != PHX_MC_UNRESOLVED &&
          insert_status != PHX_MC_CONSTRAINT_INTERSECTION) {
        return fail(insert_status, kPlcNonrepresentable, kPlcPoint, vertex);
      }
      bool applied = false;
      const int32_t star_status = insert_into_star_shaped_cavity(vertex, applied);
      if (star_status != PHX_MC_OK || !applied) return star_status;
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
    return PHX_MC_OK;
  }

  // Inserts `vertex` into a star-shaped part of its conflict cavity. The core
  // is the set of cells whose closure holds the vertex (one cell, the two
  // cells of a face, or the ring of an edge): it is star-shaped from the
  // vertex and keeps every one of its vertices on its boundary. The cavity
  // grows from the core through unconstrained faces into finite cells whose
  // circumsphere strictly holds the vertex (symbolically perturbed, as in
  // the Delaunay construction), then non-core cells are dropped until the
  // connected part around the core has every boundary face strictly
  // visible from the vertex and every cavity vertex on its boundary. Each
  // round drops a cell and the core always qualifies, so the loop ends and
  // the cone from the vertex over the boundary is a valid edit; it need not
  // be Delaunay, which local_edits_ records. A cavity whose cone breaks a
  // constrained face is refused (not applied) and left to the rebuild.
  int32_t insert_into_star_shaped_cavity(int32_t vertex, bool& applied) {
    applied = false;
    TetrahedralComplex& complex = triangulation_->complex();
    const auto cell = [&](int32_t id) -> const Tetrahedron& {
      return complex.tets[static_cast<std::size_t>(id)];
    };
    const ExactPointRef position = ref(vertex);
    const int32_t origin = triangulation_->locate(position);
    if (!triangulation_->valid_live_cell(origin) || is_ghost(cell(origin))) return PHX_MC_OK;
    int zero_slots[4];
    int zeros = 0;
    for (int slot = 0; slot < 4; ++slot) {
      if (triangulation_->orient_with(cell(origin), slot, position) == 0) {
        zero_slots[zeros++] = slot;
      }
    }
    std::vector<int32_t> core;
    if (zeros == 0) {
      core.push_back(origin);
    } else if (zeros == 1) {
      core = {origin, cell(origin).n[zero_slots[0]]};
    } else if (zeros == 2) {
      int32_t ends[2];
      int count = 0;
      for (int slot = 0; slot < 4; ++slot) {
        if (slot != zero_slots[0] && slot != zero_slots[1]) ends[count++] = cell(origin).v[slot];
      }
      const int32_t status = edge_cells(ends[0], ends[1], core);
      if (status != PHX_MC_OK) return status;
    } else {
      return PHX_MC_OK;
    }
    for (int32_t id : core) {
      if (!complex.live(id) || is_ghost(cell(id))) return PHX_MC_OK;
    }
    const std::unordered_set<int32_t> core_cells(core.begin(), core.end());
    std::unordered_set<int32_t> candidates(core.begin(), core.end());
    std::vector<int32_t> grown(core.begin(), core.end());
    for (std::size_t at = 0; at < grown.size(); ++at) {
      const Tetrahedron& current = cell(grown[at]);
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t next = current.n[slot];
        if (complex.constraint(grown[at], slot) != kNoConstraint || candidates.contains(next) ||
            !complex.live(next) || is_ghost(cell(next))) {
          continue;
        }
        const int32_t* v = cell(next).v;
        if (insphere_sos(ref(v[0]), ref(v[1]), ref(v[2]), ref(v[3]), position, v[0], v[1], v[2],
                         v[3], vertex) > 0) {
          candidates.insert(next);
          grown.push_back(next);
        }
      }
    }
    std::vector<int32_t> cavity;
    std::vector<BoundaryFacet> boundary;
    std::unordered_set<int32_t> reached;
    std::unordered_set<int32_t> kept;
    // Drops a non-core cell from the candidates; refuses (false) for the core.
    const auto drop = [&](int32_t id) {
      if (core_cells.contains(id)) return false;
      candidates.erase(id);
      return true;
    };
    for (;;) {
      // The connected part of the candidates around the core, through
      // unconstrained faces.
      cavity.assign(core.begin(), core.end());
      reached.clear();
      reached.insert(core.begin(), core.end());
      for (std::size_t at = 0; at < cavity.size(); ++at) {
        const int32_t id = cavity[at];
        for (int slot = 0; slot < 4; ++slot) {
          const int32_t next = cell(id).n[slot];
          if (complex.constraint(id, slot) == kNoConstraint && candidates.contains(next) &&
              reached.insert(next).second) {
            cavity.push_back(next);
          }
        }
      }
      // A constrained face inside the cavity would be split by the cone; a
      // boundary face must see the vertex strictly on its inner side.
      boundary.clear();
      bool dropped = false;
      for (std::size_t at = 0; at < cavity.size() && !dropped; ++at) {
        const int32_t id = cavity[at];
        for (int slot = 0; slot < 4 && !dropped; ++slot) {
          const int32_t next = cell(id).n[slot];
          if (reached.contains(next)) {
            if (complex.constraint(id, slot) == kNoConstraint) continue;
            if (!drop(id) && !drop(next)) return PHX_MC_OK;
            dropped = true;
            continue;
          }
          std::array<int32_t, 4> cone{};
          std::copy(cell(id).v, cell(id).v + 4, cone.begin());
          cone[static_cast<std::size_t>(slot)] = vertex;
          if (orient(cone[0], cone[1], cone[2], cone[3]) <= 0) {
            if (!drop(id)) return PHX_MC_OK;
            dropped = true;
            continue;
          }
          const int back = neighbor_slot(cell(next), id);
          if (back < 0) {
            return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, vertex);
          }
          boundary.push_back({id, slot, next, back});
        }
      }
      if (dropped) continue;
      // Every cavity vertex must stay on the cavity boundary.
      kept.clear();
      for (const BoundaryFacet& face : boundary) {
        for (int slot = 0; slot < 4; ++slot) {
          if (slot != face.slot) kept.insert(cell(face.tet).v[slot]);
        }
      }
      for (int32_t id : cavity) {
        for (int32_t corner : cell(id).v) {
          if (kept.contains(corner)) continue;
          if (!drop(id)) return PHX_MC_OK;
          dropped = true;
          break;
        }
      }
      if (!dropped) break;
    }
    CavityEdit edit(complex);
    edit.begin();
    for (int32_t id : cavity) {
      if (edit.remove(id) != EditStatus::kOk) {
        edit.rollback();
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, vertex);
      }
    }
    for (const BoundaryFacet& face : boundary) {
      std::array<int32_t, 4> cone{};
      std::copy(cell(face.tet).v, cell(face.tet).v + 4, cone.begin());
      cone[static_cast<std::size_t>(face.slot)] = vertex;
      edit.add(cone.data(), nullptr, kNoRegion);
    }
    EditStatus status = edit.validate(
        [&](const int32_t* v) { return orient(v[0], v[1], v[2], v[3]) > 0; });
    if (status == EditStatus::kOk) status = edit.commit();
    if (status != EditStatus::kOk) {
      edit.rollback();
      return status == EditStatus::kSlotLimit ? PHX_MC_INDEX_RANGE_ERROR : PHX_MC_OK;
    }
    local_edits_ = true;
    for (int32_t id : edit.created()) {
      if (!is_ghost(cell(id)) && triangulation_->set_hint(id)) break;
    }
    applied = true;
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

  // Recovers every chain interval of every source edge, in row order. A
  // missing interval is first recovered without new vertices (planar flips on
  // an incident facet plane, else one edge removal); otherwise an encroached
  // interval is split by split_source_edge, the pass goes on with the next
  // edge, and the next pass rescans. An interval
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
    for (std::size_t edge = 0; edge < edge_chains_.size(); ++edge) {
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
        if (recovered) continue;
        if (encroached(first, second)) {
          // Split, then go on with the next edge on the updated complex; a
          // refused incremental insertion leaves the rebuild to the restart.
          const int32_t split_status = split_source_edge(edge, interval, restart);
          if (split_status != PHX_MC_OK || !incremental_triangulation_ready_) return split_status;
          const int32_t middle = edge_chains_[edge][interval + 1].vertex;
          chain_intervals_.erase(topology::undirected_edge(first, second));
          chain_intervals_.insert(topology::undirected_edge(first, middle));
          chain_intervals_.insert(topology::undirected_edge(middle, second));
          break;
        }
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
    // Chain intervals whose carrier box misses the face's carrier box,
    // widened by four units in the last place of the largest coordinate
    // (carriers of exact positions are within half a unit), cannot meet the
    // face; the exact tests below run only on the others.
    double box_lower[3], box_upper[3];
    for (int axis = 0; axis < 3; ++axis) {
      box_lower[axis] = std::min({point(source.oriented[0])[axis], point(source.oriented[1])[axis],
                                  point(source.oriented[2])[axis]});
      box_upper[axis] = std::max({point(source.oriented[0])[axis], point(source.oriented[1])[axis],
                                  point(source.oriented[2])[axis]});
      const double margin = 0x1p-50 * std::max({std::fabs(box_lower[axis]),
                                                std::fabs(box_upper[axis]),
                                                std::numeric_limits<double>::min()});
      box_lower[axis] -= margin;
      box_upper[axis] += margin;
    }
    const auto misses_box = [&](int32_t first, int32_t second) {
      for (int axis = 0; axis < 3; ++axis) {
        if (std::max(point(first)[axis], point(second)[axis]) < box_lower[axis] ||
            std::min(point(first)[axis], point(second)[axis]) > box_upper[axis]) {
          return true;
        }
      }
      return false;
    };
    std::vector<std::array<int32_t, 2>> internal_constraints;
    std::set<uint64_t> internal_keys;
    for (const auto& chain : edge_chains_) {
      for (std::size_t interval = 0; interval + 1 < chain.size(); ++interval) {
        const int32_t first = chain[interval].vertex;
        const int32_t second = chain[interval + 1].vertex;
        const uint64_t key = topology::undirected_edge(first, second);
        if (boundary_intervals.contains(key) || misses_box(first, second)) continue;
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

  // Lawson flips toward the in-plane Delaunay triangulation of the children
  // of one source triangle. A child edge shared by two children (a, b, c)
  // and (b, a, d) that is not a chain interval is flipped to cd when d lies
  // strictly inside the circumcircle of abc, decided exactly on the source
  // plane: the sphere through a, b, c and the lifted point a + n (n the
  // normal of abc) meets the plane in that circumcircle. A strictly
  // non-Delaunay edge has a strictly convex quadrilateral, so the flip keeps
  // the children a triangulation of the source triangle, and each flip
  // strictly lowers the lifted triangulation, so the pass ends (Lawson).
  // Children that follow the in-plane Delaunay triangulation are the faces
  // the Delaunay tetrahedralization tends to contain, where a fan from a
  // split point keeps demanding edges that cross in-plane mesh edges.
  void legalize_source_children(int32_t source_triangle) {
    const auto inside = [&](int32_t a, int32_t b, int32_t c, int32_t d) {
      Expansion pa[3], pb[3], pc[3], pd[3], pe[3], u[3], v[3];
      for (int axis = 0; axis < 3; ++axis) {
        pa[axis] = coordinate(ref(a), axis);
        pb[axis] = coordinate(ref(b), axis);
        pc[axis] = coordinate(ref(c), axis);
        pd[axis] = coordinate(ref(d), axis);
        u[axis] = pb[axis] - pa[axis];
        v[axis] = pc[axis] - pa[axis];
      }
      // orient3d(a, b, c, a + n) = |n|^2 > 0, so insphere > 0 is inside.
      pe[0] = pa[0] + (u[1] * v[2] - u[2] * v[1]);
      pe[1] = pa[1] + (u[2] * v[0] - u[0] * v[2]);
      pe[2] = pa[2] + (u[0] * v[1] - u[1] * v[0]);
      return insphere_sign(pa, pb, pc, pe, pd) > 0;
    };
    for (;;) {
      // Directed child edge -> (child key, opposite vertex).
      std::map<std::pair<int32_t, int32_t>, std::pair<Triple, int32_t>> edges;
      for (const auto& [key, child] : source_faces_) {
        if (child.source_triangle != source_triangle) continue;
        for (int side = 0; side < 3; ++side) {
          edges.emplace(std::pair{child.oriented[static_cast<std::size_t>(side)],
                                  child.oriented[static_cast<std::size_t>((side + 1) % 3)]},
                        std::pair{key, child.oriented[static_cast<std::size_t>((side + 2) % 3)]});
        }
      }
      bool flipped = false;
      for (const auto& [edge, use] : edges) {
        const auto [a, b] = edge;
        if (a > b) continue;
        const auto other = edges.find({b, a});
        if (other == edges.end()) continue;
        if (chain_intervals_.contains(topology::undirected_edge(a, b))) continue;
        const int32_t c = use.second;
        const int32_t d = other->second.second;
        if (!inside(a, b, c, d)) continue;
        const PlcSourceFace template_face = source_faces_.at(use.first);
        source_faces_.erase(use.first);
        source_faces_.erase(other->second.first);
        const Triple first{a, d, c};
        const Triple second{d, b, c};
        source_faces_.emplace(sorted_triple(first.data()),
                              PlcSourceFace{first, template_face.facet, source_triangle});
        source_faces_.emplace(sorted_triple(second.data()),
                              PlcSourceFace{second, template_face.facet, source_triangle});
        flipped = true;
        break;
      }
      if (!flipped) return;
    }
  }

  // Splits the missing subface `source`, a child of source triangle
  // T = ABC, at the exactly witnessed point S = A + l1 (B - A) + l2 (C - A).
  // (l1, l2) are the barycentric coordinates of the child's centroid,
  // rounded to dyadic rationals of increasing precision until S lies
  // exactly in the child's central region (every barycentric coordinate of
  // S in the child at least 1/4), else, failing that at every precision, in
  // its open interior. A point merely inside could lie arbitrarily close to
  // an edge, and the fan triangle on that edge would be a needle whose own
  // interior no dyadic split point reaches; in the central region every fan
  // triangle keeps at least a quarter of the child's height over its edge.
  // S keeps its exact position in the exact vertex bank (its carrier is the
  // rounding of S), so S is exactly coplanar with every subface of T and
  // later recovery and suppression decide on S itself. The child becomes
  // the fan from S over its boundary, which includes the split points on its
  // edges.
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
    // The central region: the triangle of the points (2 X + Y + Z) / 4.
    Expansion central_exact[3][3];
    double central_carrier[3][3];
    ExactPointRef central[3];
    for (int corner = 0; corner < 3; ++corner) {
      for (int axis = 0; axis < 3; ++axis) {
        central_exact[corner][axis] =
            (coordinate(child[corner], axis).scaled(2.0) +
             coordinate(child[(corner + 1) % 3], axis) +
             coordinate(child[(corner + 2) % 3], axis))
                .scaled(0.25);
        central_carrier[corner][axis] = central_exact[corner][axis].estimate();
      }
      central[corner] = ExactPointRef{central_carrier[corner], central_exact[corner]};
    }
    double parameters[2];
    Expansion position[3];
    double carrier[3];
    bool found = false;
    for (int pass = 0; pass < 2 && !found; ++pass) {
      const ExactPointRef* region = pass == 0 ? central : child;
      for (int bits = 4; bits <= 64 && !found; ++bits) {
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
        found = locate_on_triangle(ExactPointRef{carrier, position}, region, side) ==
                kFeatureInterior;
      }
    }
    if (!found) return refuse();
    // A split point in the closed diametral ball of a chain interval on the
    // child's boundary encroaches that interval, which is split instead
    // (Shewchuk: encroached segments are split before subfacets), so facet
    // splits never pile up against a segment.
    const ExactPointRef candidate{carrier, position};
    for (std::size_t side = 0; side < boundary.size(); ++side) {
      const int32_t a = boundary[side];
      const int32_t b = boundary[(side + 1) % boundary.size()];
      if (!chain_intervals_.contains(topology::undirected_edge(a, b))) continue;
      Expansion dot;
      for (int axis = 0; axis < 3; ++axis) {
        dot = dot + (coordinate(candidate, axis) - coordinate(ref(a), axis)) *
                        (coordinate(candidate, axis) - coordinate(ref(b), axis));
      }
      if (dot.sign() > 0) continue;
      for (std::size_t row = 0; row < edge_chains_.size(); ++row) {
        const auto& chain = edge_chains_[row];
        for (std::size_t interval = 0; interval + 1 < chain.size(); ++interval) {
          const int32_t first = chain[interval].vertex;
          const int32_t second = chain[interval + 1].vertex;
          if ((first == a && second == b) || (first == b && second == a)) {
            return split_source_edge(row, interval, restart);
          }
        }
      }
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet, source.facet);
    }
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
    legalize_source_children(source.source_triangle);
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
    PlcFaceMap faces;
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
      faces.reserve(2 * complex.tets.size());
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
      // The subfaces missing at this pass are handled in turn on the current
      // complex: one still missing (checked locally once the complex has
      // changed) is recovered by planar flips, else split. Facet splits go
      // on with the next one; an interval split (an encroached boundary) or
      // a refused incremental insertion ends the pass.
      std::vector<std::pair<Triple, PlcSourceFace>> missing;
      for (const auto& [key, source] : source_faces_) {
        const auto found = faces.find(key);
        if (found == faces.end() || found->second.size() != 2) missing.push_back({key, source});
      }
      if (missing.empty()) return mark_source_faces();
      bool flipped = false;
      std::vector<int32_t> cells;
      for (const auto& [key, source] : missing) {
        if (!source_faces_.contains(key)) continue;
        if (restart || flipped) {
          const int32_t status = edge_cells(key[0], key[1], cells);
          if (status != PHX_MC_OK) return status;
          if (std::any_of(cells.begin(), cells.end(), [&](int32_t cell) {
                return vertex_slot(complex.tets[static_cast<std::size_t>(cell)], key[2]) >= 0;
              })) {
            continue;
          }
        }
        const PlcFailure prior_failure = failure_;
        bool planar_recovered = false;
        const int32_t planar_status = recover_planar_source_face(key, source, planar_recovered);
        if (planar_recovered) {
          flipped = true;
          continue;
        }
        if (planar_status != PHX_MC_OK) {
          if (planar_status != PHX_MC_CONSTRAINT_INTERSECTION ||
              failure_.reason != kPlcFixedFacet) {
            return planar_status;
          }
          failure_ = prior_failure;
        }
        const std::size_t points_before = candidate_.points.size();
        bool split = false;
        const int32_t split_status = split_source_face(key, source, split);
        if (split_status != PHX_MC_OK) return split_status;
        restart = restart || split;
        if (candidate_.points.size() != points_before &&
            (candidate_.witnesses.back().stratum == SourceStratum::kSegment ||
             !incremental_triangulation_ready_)) {
          return PHX_MC_OK;
        }
      }
      if (restart) return PHX_MC_OK;
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

  // Points are tried newest first (undoing later splits before the earlier
  // ones they refined), in passes: a point none of whose replacements
  // applies is retried after the others have been removed, and the pass that
  // removes nothing refuses its first remaining point. Each pass removes at
  // least one point, so there are at most as many passes as points.
  int32_t suppress_witnessed_vertices() {
    const int32_t count = static_cast<int32_t>(candidate_.points.size() / 3);
    suppressed_.assign(static_cast<std::size_t>(count), 0);
    std::vector<int32_t> pending;
    for (int32_t vertex = count - 1; vertex >= source_point_count_; --vertex) {
      if (input_.policy != BoundaryPolicy::kFixed && exact_points_.position(vertex) == nullptr) {
        continue;
      }
      pending.push_back(vertex);
    }
    while (!pending.empty()) {
      std::vector<int32_t> remaining;
      for (int32_t vertex : pending) {
        bool removed = false;
        const int32_t status = suppress_vertex(vertex, removed);
        if (status != PHX_MC_OK) return status;
        if (!removed) remaining.push_back(vertex);
      }
      if (remaining.size() == pending.size()) return refuse_suppression(remaining.front());
      pending = std::move(remaining);
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
  // (sd crosses the open triangle abc) and each of them either contains t
  // or has a link face strictly facing t. Termination: let N be the number
  // of finite cells around s, without t, whose link face does not strictly
  // face t. A flip removes one such cell, adds three that are not counted,
  // and leaves every other cell around s unchanged (the neighbour does not
  // contain s), so N drops by one per flip and the pass ends, when no flip
  // applies, after at most N flips.
  int32_t flip_star_toward(int32_t s, int32_t t) {
    TetrahedralComplex& complex = triangulation_->complex();
    std::vector<int32_t> cells;
    int32_t status = vertex_cells(s, cells);
    if (status != PHX_MC_OK) return status;
    for (;;) {
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
          if (valid && d != t) {
            std::array<int32_t, 4> toward = next.v;
            toward[static_cast<std::size_t>(own)] = t;
            valid = orient(toward[0], toward[1], toward[2], toward[3]) > 0;
          }
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
  }

  // A cell of a suppression proposal and the source facet marks of its faces.
  struct SuppressionCell {
    std::array<int32_t, 4> v;
    std::array<int32_t, 4> constraints;
  };

  // Steiner-free fill of a closed cavity. Each boundary cell carries
  // `placeholder` in one slot: the face opposite it is a boundary face (a
  // wall) and a cavity vertex x lies on its inner side exactly when the cell
  // with x substituted is positive. Sides of faces are keyed by the face
  // oriented as its cell on that side sees it, so a face may bound the
  // cavity from both sides (a slit along an open sheet). An open side is
  // closed by an apex: a vertex on its inner side that is visible through
  // it (the new cell's interior meets no wall). Each other face of the new
  // cell closes an open side or opens its far side, and may neither repeat a
  // closed side nor lie outside a wall. Apices are ordered by the pencil of
  // spheres through the side (the exact symbolically perturbed insphere
  // test orders them totally). A finished fill is a set of positive cells
  // with every face paired and the cavity boundary as boundary, hence the
  // 3-chain of the cavity: the cells tile it.
  //
  // Without `exhaustive`, the least open side takes its first apex, with no
  // backtracking: the gift-wrapping construction of the constrained
  // Delaunay tetrahedralization of the cavity (Shewchuk, "General-
  // dimensional constrained Delaunay and constrained regular triangulations
  // I", DCG 39, 2008). When that tetrahedralization exists (unique under the
  // perturbation) every step adds one of its cells in any side order, so a
  // failure means the cavity has none. Each step closes its side for good,
  // so the fill takes at most one step per cell, each costing
  // O(candidates * walls) predicates.
  //
  // With `exhaustive`, the search branches over every apex of the open side
  // with the fewest apices and backtracks, so it finds a Steiner-free fill
  // whenever one exists (every fill closes that side by one of its apices,
  // and every cell of a fill is visible). The state is the set of closed
  // sides, which determines the open ones; failed states are remembered and
  // never explored again, so each state is expanded at most once and the
  // search ends. Deciding whether a polyhedron has a Steiner-free
  // tetrahedralization is NP-complete (Ruppert and Seidel, "On the
  // difficulty of triangulating three-dimensional nonconvex polyhedra",
  // DCG 7, 1992), so no polynomial method is complete; the caller reaches
  // this search only after its polynomial fills have failed. `nodes`
  // receives the number of states expanded.
  bool wrap_cavity(const std::vector<SuppressionCell>& boundary, int32_t placeholder,
                   const std::vector<int32_t>& interior, bool exhaustive,
                   std::vector<SuppressionCell>& filled, std::size_t& nodes) const {
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
    std::vector<Triple> walls;
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
      walls.push_back({side.a, side.b, side.c});
      for (int k = 0; k < 4; ++k) {
        if (k != slot) vertices.insert(cell.v[static_cast<std::size_t>(k)]);
      }
    }
    // Interior candidates: volume points that may become apices.
    vertices.insert(interior.begin(), interior.end());
    // The interior of the positive cell `tet` misses the closed wall exactly
    // when a plane separates them; the facets of the Minkowski difference
    // of two polytopes are parallel to a face of either or to an edge of
    // each, so these planes are the cell's face planes, the wall's plane,
    // and the planes through a cell edge parallel to a wall edge.
    const auto separated = [&](const std::array<int32_t, 4>& tet, const Triple& wall) {
      for (int slot = 0; slot < 4; ++slot) {
        if (std::none_of(wall.begin(), wall.end(), [&](int32_t x) {
              return positive(with(tet, slot, x));
            })) {
          return true;
        }
      }
      bool above = false;
      bool below = false;
      for (int32_t x : tet) {
        const int side = orient(wall[0], wall[1], wall[2], x);
        above = above || side > 0;
        below = below || side < 0;
      }
      if (!(above && below)) return true;
      constexpr int kEdges[6][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2},
                                    {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 3, 0, 1}};
      for (const auto& edge : kEdges) {
        const ExactPointRef p = ref(tet[static_cast<std::size_t>(edge[0])]);
        const ExactPointRef q = ref(tet[static_cast<std::size_t>(edge[1])]);
        const ExactPointRef o1 = ref(tet[static_cast<std::size_t>(edge[2])]);
        const ExactPointRef o2 = ref(tet[static_cast<std::size_t>(edge[3])]);
        for (int k = 0; k < 3; ++k) {
          const int32_t r = wall[static_cast<std::size_t>(k)];
          const int32_t s = wall[static_cast<std::size_t>((k + 1) % 3)];
          const int32_t m = wall[static_cast<std::size_t>((k + 2) % 3)];
          // e = p + (s - r): the plane through p, q, e contains pq and is
          // parallel to rs.
          Expansion e[3];
          double carrier[3];
          for (int axis = 0; axis < 3; ++axis) {
            e[axis] = coordinate(p, axis) + coordinate(ref(s), axis) - coordinate(ref(r), axis);
            carrier[axis] = e[axis].estimate();
          }
          const ExactPointRef through{carrier, e};
          if (collinear(p, q, through)) continue;
          const int a1 = orient3d(p, q, through, o1);
          const int a2 = orient3d(p, q, through, o2);
          const int b1 = orient3d(p, q, through, ref(r));
          const int b2 = orient3d(p, q, through, ref(m));
          if ((a1 <= 0 && a2 <= 0 && b1 >= 0 && b2 >= 0) ||
              (a1 >= 0 && a2 >= 0 && b1 <= 0 && b2 <= 0)) {
            return true;
          }
        }
      }
      return false;
    };
    // Apices of an open side: the vertices on its inner side whose cell is
    // visible, in pencil order (number of rivals inside the sphere through
    // the side). Both depend only on the oriented side and are kept per fill.
    std::map<FacetKey, std::vector<int32_t>> apices;
    const auto apices_of = [&](const FacetKey& side, const Open& open) -> const std::vector<int32_t>& {
      const auto known = apices.find(side);
      if (known != apices.end()) return known->second;
      std::vector<int32_t> candidates;
      for (int32_t vertex : vertices) {
        if (vertex == side.a || vertex == side.b || vertex == side.c) continue;
        const std::array<int32_t, 4> tet = with(open.cell, open.slot, vertex);
        if (!positive(tet)) continue;
        if (std::all_of(walls.begin(), walls.end(),
                        [&](const Triple& wall) { return separated(tet, wall); })) {
          candidates.push_back(vertex);
        }
      }
      std::vector<std::pair<int, int32_t>> ranked;
      ranked.reserve(candidates.size());
      for (int32_t vertex : candidates) {
        const std::array<int32_t, 4> tet = with(open.cell, open.slot, vertex);
        int inside = 0;
        for (int32_t rival : candidates) {
          if (rival == vertex) continue;
          inside += insphere_sos(ref(tet[0]), ref(tet[1]), ref(tet[2]), ref(tet[3]), ref(rival),
                                 tet[0], tet[1], tet[2], tet[3], rival) > 0 ? 1 : 0;
        }
        ranked.push_back({inside, vertex});
      }
      std::sort(ranked.begin(), ranked.end());
      std::vector<int32_t> order;
      order.reserve(ranked.size());
      for (const auto& entry : ranked) order.push_back(entry.second);
      return apices.emplace(side, std::move(order)).first->second;
    };
    // Closing side `key` by `apex`: the new cell, the open sides its other
    // faces close, the sides they seal, and the far sides they open; false
    // when a face repeats a closed side or lies outside a wall.
    struct Step {
      SuppressionCell cell;
      std::vector<std::pair<FacetKey, Open>> matched;
      std::vector<FacetKey> sealed;
      std::vector<std::pair<FacetKey, Open>> opened;
    };
    std::set<FacetKey> closed;
    const auto plan = [&](const Open& open, int32_t apex, Step& step) {
      step = Step{};
      const std::array<int32_t, 4> tet = with(open.cell, open.slot, apex);
      step.cell = {tet, {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint}};
      step.cell.constraints[static_cast<std::size_t>(open.slot)] = open.constraint;
      for (int slot = 0; slot < 4; ++slot) {
        if (slot == open.slot) continue;
        const FacetKey side = side_of(tet, slot);
        const auto other = front.find(side);
        if (other != front.end()) {
          step.cell.constraints[static_cast<std::size_t>(slot)] = other->second.constraint;
          step.matched.push_back(*other);
          continue;
        }
        if (closed.contains(side) || outside.contains(side)) return false;
        step.sealed.push_back(side);
        const FacetKey far = side.reversed();
        if (closed.contains(far) || front.contains(far)) continue;
        // The cell across this face lies on its other side.
        std::array<int32_t, 4> flipped = with(tet, slot, placeholder);
        std::swap(flipped[static_cast<std::size_t>((slot + 1) % 4)],
                  flipped[static_cast<std::size_t>((slot + 2) % 4)]);
        step.opened.push_back({far, Open{flipped, slot, kNoConstraint}});
      }
      return true;
    };
    const auto apply = [&](const FacetKey& key, const Step& step) {
      front.erase(key);
      closed.insert(key);
      for (const auto& [side, entry] : step.matched) {
        (void)entry;
        front.erase(side);
        closed.insert(side);
      }
      for (const FacetKey& side : step.sealed) closed.insert(side);
      for (const auto& [side, entry] : step.opened) front.emplace(side, entry);
      filled.push_back(step.cell);
    };
    filled.clear();
    nodes = 0;
    if (!exhaustive) {
      while (!front.empty()) {
        const FacetKey key = front.begin()->first;
        const Open open = front.begin()->second;
        const std::vector<int32_t>& order = apices_of(key, open);
        Step step;
        if (order.empty() || !plan(open, order.front(), step)) return false;
        apply(key, step);
        ++nodes;
      }
      return true;
    }
    const auto undo = [&](const FacetKey& key, const Open& open, const Step& step) {
      filled.pop_back();
      for (const auto& [side, entry] : step.opened) {
        (void)entry;
        front.erase(side);
      }
      for (const FacetKey& side : step.sealed) closed.erase(side);
      for (const auto& [side, entry] : step.matched) {
        closed.erase(side);
        front.emplace(side, entry);
      }
      closed.erase(key);
      front.emplace(key, open);
    };
    std::set<std::vector<FacetKey>> failed;
    std::function<bool()> search = [&]() -> bool {
      if (front.empty()) return true;
      std::vector<FacetKey> state(closed.begin(), closed.end());
      if (failed.contains(state)) return false;
      ++nodes;
      FacetKey key{};
      Open open{};
      const std::vector<int32_t>* order = nullptr;
      for (const auto& [side, entry] : front) {
        const std::vector<int32_t>& candidates = apices_of(side, entry);
        if (order == nullptr || candidates.size() < order->size()) {
          key = side;
          open = entry;
          order = &candidates;
        }
        if (candidates.empty()) break;
      }
      for (int32_t apex : *order) {
        Step step;
        if (!plan(open, apex, step)) continue;
        apply(key, step);
        if (search()) return true;
        undo(key, open, step);
      }
      failed.insert(std::move(state));
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
  // the new subfaces; any other finite wedge by the first of, in order: the
  // constrained Delaunay fill of its cavity (wrap_cavity; walls are the link
  // faces and the new subfaces), the cones from one cavity vertex that sees
  // every wall not through it, the constrained Delaunay fill with centroid
  // points of the blocking cells, the cones from one binary64 point
  // y = s + l d (d the sum of the inner unit normals of its subface planes,
  // l halving) over the walls, and the exhaustive Steiner-free search of
  // wrap_cavity; the polynomial options come first, the search (counted in
  // PlcRecovery::suppression_search) only when they all fail. Every finite
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
      // The cavity walls: the new subfaces and the link faces (s replaced
      // by the placeholder) with their marks.
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
      // 1. The constrained Delaunay fill.
      std::vector<Proposal> wrapped;
      std::size_t nodes = 0;
      if (wrap_cavity(cavity, kPlaceholder, {}, false, wrapped, nodes)) {
        proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
        continue;
      }
      // 2. Cones from one cavity vertex v over every wall not through v: a
      // tiling when the cavity is strictly star-shaped from v. A cone face
      // through v that is a wall carries that wall's mark.
      std::set<int32_t> on_walls;
      for (const Proposal& wall : cavity) {
        for (int32_t vertex : wall.v) {
          if (vertex != kPlaceholder) on_walls.insert(vertex);
        }
      }
      const auto slot_of = [](const std::array<int32_t, 4>& v, int32_t vertex) {
        return static_cast<int>(std::find(v.begin(), v.end(), vertex) - v.begin());
      };
      bool coned = false;
      for (int32_t apex : on_walls) {
        std::vector<Proposal> cones;
        std::map<Triple, int32_t> through;
        bool visible = true;
        for (const Proposal& wall : cavity) {
          if (std::find(wall.v.begin(), wall.v.end(), apex) == wall.v.end()) continue;
          const int slot = slot_of(wall.v, kPlaceholder);
          Triple face{};
          oriented_facet(wall.v.data(), slot, face.data());
          through.emplace(sorted_triple(face.data()),
                          wall.constraints[static_cast<std::size_t>(slot)]);
        }
        for (const Proposal& wall : cavity) {
          if (std::find(wall.v.begin(), wall.v.end(), apex) != wall.v.end()) continue;
          Proposal cone = wall;
          std::replace(cone.v.begin(), cone.v.end(), kPlaceholder, apex);
          if (!positive(cone.v, nullptr)) {
            visible = false;
            break;
          }
          const int own = slot_of(cone.v, apex);
          for (int slot = 0; slot < 4; ++slot) {
            if (slot == own) continue;
            Triple face{};
            oriented_facet(cone.v.data(), slot, face.data());
            const auto wall = through.find(sorted_triple(face.data()));
            if (wall != through.end()) cone.constraints[static_cast<std::size_t>(slot)] = wall->second;
          }
          cones.push_back(cone);
        }
        if (!visible) continue;
        proposals.insert(proposals.end(), cones.begin(), cones.end());
        coned = true;
        break;
      }
      if (coned) continue;
      // 3. Steiner-assisted fill: interior points of the cells that block
      // the contraction (every cell without a target) join the apices.
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
      if (!extra.empty() && wrap_cavity(cavity, kPlaceholder, extra, false, wrapped, nodes)) {
        proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
        continue;
      }
      // 4. Cones from one binary64 point y = s + l d over the walls.
      double direction[3] = {0.0, 0.0, 0.0};
      for (const auto& unit : normals) {
        for (int axis = 0; axis < 3; ++axis) direction[axis] += unit[static_cast<std::size_t>(axis)];
      }
      const double norm = std::sqrt(geometry::dot(direction, direction));
      bool found = false;
      double location[3];
      if (!normals.empty() && norm > 0.0 && std::isfinite(reach) && reach > 0.0) {
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
      }
      if (found) {
        // Placeholder ids of later centers stay distinct until appended.
        for (Proposal& cone : cavity) {
          std::replace(cone.v.begin(), cone.v.end(), kPlaceholder,
                       kPlaceholder + 1 + static_cast<int32_t>(centers.size()));
        }
        centers.push_back({location[0], location[1], location[2]});
        proposals.insert(proposals.end(), cavity.begin(), cavity.end());
        continue;
      }
      // 5. The exact exhaustive Steiner-free search (wrap_cavity), reached
      // only after every polynomial option has failed.
      nodes = 0;
      const bool searched = wrap_cavity(cavity, kPlaceholder, {}, true, wrapped, nodes);
      PlcSuppressionSearch& evidence = candidate_.suppression_search;
      ++evidence.searches;
      evidence.nodes += static_cast<int64_t>(nodes);
      evidence.max_nodes = std::max(evidence.max_nodes, static_cast<int64_t>(nodes));
      evidence.max_wedge_cells =
          std::max(evidence.max_wedge_cells, static_cast<int64_t>(rows.size()));
      if (!searched) return abandon();
      ++evidence.fills;
      proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
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

  // A point that cannot be removed is a STEINER_SUPPRESSION refusal naming
  // its source segment or facet; under FIXED it is the FIXED_SEGMENT /
  // FIXED_FACET refusal of that entity (its boundary could not be recovered
  // unmodified).
  int32_t refuse_suppression(int32_t s) {
    const PlcRecoveryWitness& witness = candidate_.witnesses[static_cast<std::size_t>(s)];
    const bool fixed = input_.policy == BoundaryPolicy::kFixed;
    const int32_t status = fixed ? PHX_MC_CONSTRAINT_INTERSECTION : PHX_MC_UNRESOLVED;
    if (witness.stratum == SourceStratum::kSegment) {
      return fail(status, fixed ? kPlcFixedSegment : kPlcSteinerSuppression, kPlcEdge,
                  static_cast<int64_t>(witness.entity), kPlcPoint, s);
    }
    const int32_t facet = input_.polygon_facets[
        candidate_.input_polygons[static_cast<std::size_t>(witness.entity)]];
    return fail(status, fixed ? kPlcFixedFacet : kPlcSteinerSuppression, kPlcFacet, facet,
                kPlcPoint, s);
  }

  // Suppresses one witnessed vertex if it can: contraction onto a chain
  // neighbour (a segment point) or a disk neighbour (a facet point), else
  // the ear-clipped half-fans.
  int32_t suppress_vertex(int32_t s, bool& removed) {
    removed = false;
    const PlcRecoveryWitness& witness = candidate_.witnesses[static_cast<std::size_t>(s)];
    const bool segment = witness.stratum == SourceStratum::kSegment;
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
      const int32_t status = remove_witnessed_vertex(s, target, p, q, removed);
      if (status != PHX_MC_OK) return status;
      if (!removed) continue;
      if (segment) {
        auto& chain = edge_chains_[static_cast<std::size_t>(witness.entity)];
        chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(position));
      }
      return PHX_MC_OK;
    }
    return PHX_MC_OK;
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

int32_t fill_suppression_cavity(const SuppressionCavity& cavity, SuppressionCavityFill& fill) {
  try {
    return PlcRecoveryBuilder::run_cavity_fixture(cavity, fill);
  } catch (const std::bad_alloc&) {
    fill = SuppressionCavityFill{};
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

