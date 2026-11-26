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
#include <cstring>
#include <limits>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <span>
#include <map>
#include "geometry3.hpp"
#include "predicates.hpp"
#include "facet_coverage.hpp"
#include "triangulation3d.hpp"

#include "intersections.hpp"
#include "mesh.hpp"
#include "plc3d.hpp"
#include "topology_keys.hpp"
#include "tet_mesh.hpp"
#include "improve3d.hpp"
#include "deterministic_float.hpp"
#include "volume_lattice.hpp"

namespace phx::mc {
namespace {

using Triple = std::array<int32_t, 3>;

Triple sorted_triple(const int32_t* vertices) {
  int32_t a = vertices[0], b = vertices[1], c = vertices[2];
  if (b < a) std::swap(a, b);
  if (c < b) std::swap(b, c);
  if (b < a) std::swap(a, b);
  return {a, b, c};
}

// Hash of the vertex ids of a face or cell key.
template <std::size_t N>
struct IdTupleHash {
  std::size_t operator()(const std::array<int32_t, N>& ids) const {
    std::uint64_t mixed = 0x9E3779B97F4A7C15ULL;
    for (int32_t id : ids) {
      mixed = (mixed ^ static_cast<std::uint32_t>(id)) * 0xC2B2AE3D27D4EB4FULL;
      mixed ^= mixed >> 29;
    }
    return static_cast<std::size_t>(mixed);
  }
};

// 2^floor(log2 x) and 2^ceil(log2 x) of a finite x > 0, from its exact
// binary exponent. std::log2 is not correctly rounded in every platform's
// libm, so the floor or ceiling of its result could differ across platforms
// next to a power of two.
double power_of_two_floor(double x) { return std::ldexp(1.0, std::ilogb(x)); }
double power_of_two_ceil(double x) {
  const double below = power_of_two_floor(x);
  return below == x ? x : 2.0 * below;
}

bool legal_contact(int8_t kind) {
  return kind == PHX_MC_DISJOINT || kind == PHX_MC_SHARED_VERTEX ||
         kind == PHX_MC_SHARED_EDGE;
}

// The exact normal of a declared plane row (origin, first axis, second axis):
// the cross product of its axes.
std::array<Expansion, 3> declared_plane_normal(const double* plane) {
  const double* u = plane + 3;
  const double* v = plane + 6;
  return {Expansion::product(u[1], v[2]) - Expansion::product(u[2], v[1]),
          Expansion::product(u[2], v[0]) - Expansion::product(u[0], v[2]),
          Expansion::product(u[0], v[1]) - Expansion::product(u[1], v[0])};
}

// Whether the declared plane row `plane` with exact normal `normal` admits
// the binary64 carrier x, exactly: the plane meets the closed box of the
// reals that round to x (per axis, between the midpoints with the adjacent
// binary64 values), so x is a correct rounding of a point of the plane, or
// lies within `tolerance` of x. `distance` receives an upper bound of the
// distance from x to the plane.
bool declared_plane_admits(const double* plane, const std::array<Expansion, 3>& normal,
                           double tolerance, const double* x, double& distance,
                           const double* affine_metric) {
  const double* origin = plane;
  // Twice the signed offset n . (c - o) at the corners c of the box, from
  // the exact doubled bounds x + next(x) of each axis.
  std::array<Expansion, 3> lower;
  std::array<Expansion, 3> upper;
  for (int axis = 0; axis < 3; ++axis) {
    const Expansion doubled_origin = Expansion::sum(origin[axis], origin[axis]);
    lower[static_cast<std::size_t>(axis)] =
        Expansion::sum(x[axis], std::nextafter(x[axis], -std::numeric_limits<double>::infinity())) -
        doubled_origin;
    upper[static_cast<std::size_t>(axis)] =
        Expansion::sum(x[axis], std::nextafter(x[axis], std::numeric_limits<double>::infinity())) -
        doubled_origin;
  }
  bool below = false;
  bool above = false;
  for (int corner = 0; corner < 8; ++corner) {
    Expansion offset;
    for (int axis = 0; axis < 3; ++axis) {
      const std::size_t row = static_cast<std::size_t>(axis);
      offset = offset + normal[row] * ((corner >> axis) & 1 ? upper[row] : lower[row]);
    }
    below = below || offset.sign() <= 0;
    above = above || offset.sign() >= 0;
  }
  Expansion offset;
  for (int axis = 0; axis < 3; ++axis) {
    offset = offset + normal[static_cast<std::size_t>(axis)] *
                          Expansion::difference(x[axis], origin[axis]);
  }
  Expansion length = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
  Expansion metric_offset = offset;
  if (affine_metric != nullptr) {
    // A^{-T} n = cof(A) n / det(A). Keep the comparison exact in
    // source coordinates, without rounding a transformed plane.
    std::array<std::array<Expansion, 3>, 3> cofactors;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const int r = (i + 1) % 3, s = (i + 2) % 3;
        const int c = (j + 1) % 3, d = (j + 2) % 3;
        cofactors[i][j] = Expansion::product(affine_metric[3 * r + c],
                                              affine_metric[3 * s + d]) -
                          Expansion::product(affine_metric[3 * r + d],
                                              affine_metric[3 * s + c]);
      }
    }
    const Expansion determinant = cofactors[0][0] * Expansion(affine_metric[0]) +
                                  cofactors[0][1] * Expansion(affine_metric[1]) +
                                  cofactors[0][2] * Expansion(affine_metric[2]);
    std::array<Expansion, 3> world_normal;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        world_normal[i] = world_normal[i] + cofactors[i][j] * normal[j];
      }
    }
    length = world_normal[0] * world_normal[0] + world_normal[1] * world_normal[1] +
             world_normal[2] * world_normal[2];
    metric_offset = determinant * offset;
  }
  distance = std::nextafter(std::fabs(metric_offset.estimate()) / std::sqrt(length.estimate()) *
                                (1.0 + 0x1p-40),
                            std::numeric_limits<double>::infinity());
  if (below && above) return true;
  return (length * Expansion::product(tolerance, tolerance) -
          metric_offset * metric_offset).sign() >= 0;
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
  const auto point = [&](int32_t vertex) {
    return points + 3 * static_cast<std::size_t>(vertex);
  };
  if (count == 3 && constraints.empty()) {
    // A triangle is its own triangulation; the checks below reduce to these
    // (every vertex lies on its own plane, and the projected orientation at
    // the lowest corner is the triangle's, a cyclic rotation of it).
    if (loop[0] == loop[1] || loop[0] == loop[2] || loop[1] == loop[2]) {
      return PHX_MC_INVALID_INPUT;
    }
    for (int corner = 0; corner < 3; ++corner) {
      if (loop[corner] < 0 || loop[corner] >= point_count) return PHX_MC_INVALID_INPUT;
    }
    const double* a = point(loop[0]);
    const double* b = point(loop[1]);
    const double* c = point(loop[2]);
    if (geometry::collinear(a, b, c, 1)) return PHX_MC_DEGENERATE_INPUT;
    const int axis = projection_axis(a, b, c);
    if (axis < 0) return PHX_MC_DEGENERATE_INPUT;
    const int x = (axis + 1) % 3;
    const int y = (axis + 2) % 3;
    const double pa[2] = {a[x], a[y]};
    const double pb[2] = {b[x], b[y]};
    const double pc[2] = {c[x], c[y]};
    if (orient2d(pa, pb, pc) == 0) return PHX_MC_DEGENERATE_INPUT;
    triangles.push_back({loop[0], loop[1], loop[2]});
    return PHX_MC_OK;
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

// The preparer's edge index, read by recovery (PlcRecoveryBuilder): the
// (edge key, triangle) uses in ascending order, the start of each run of
// equal keys (and their end), the run of every triangle edge (3 * triangle
// + k: edge k -> k + 1), and per PLC edge row its run (-1: no triangle has
// its edge) and whether its edge is an edge of an input polygon loop.
struct SourceEdgeUses {
  std::span<const std::pair<uint64_t, int32_t>> uses;
  std::span<const uint32_t> runs;
  std::span<const int32_t> triangle_runs;
  std::span<const int32_t> row_runs;
  std::span<const char> loop_rows;
};

class SourcePreparer {
 public:
  // `defer_contacts`: check_contacts is left to the caller, which runs it
  // before acting on any later outcome unless it proves every contact legal
  // (recover_plc); a closure failure here still reports a contact failure
  // first, as the undeferred order does.
  SourcePreparer(const PlcInput& input, PlcRecovery& output, bool defer_contacts = false)
      : input_(input), output_(output), defer_contacts_(defer_contacts) {}

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
    const int32_t flat = check_flat_triangles();
    if (flat != PHX_MC_OK) return flat;
    const int32_t planes = check_planes();
    if (planes != PHX_MC_OK) return planes;
    build_groups();
    const int32_t edges = build_edges();
    if (edges != PHX_MC_OK) return edges;
    if (!defer_contacts_) {
      const int32_t contacts = check_contacts();
      if (contacts != PHX_MC_OK) return contacts;
    }
    const int32_t closure = check_closure();
    if (closure != PHX_MC_OK) {
      if (defer_contacts_) {
        const int32_t contacts = check_contacts();
        if (contacts != PHX_MC_OK) return contacts;
      }
      return closure;
    }
    publish_source();
    return PHX_MC_OK;
  }

  // The edge index for recovery, after a successful run; it stays valid
  // while this preparer lives.
  SourceEdgeUses edge_uses() {
    if (loop_rows_.size() != row_runs_.size()) index_loop_rows();
    return {edge_uses_, edge_runs_, triangle_edge_runs_, row_runs_, loop_rows_};
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
    const std::vector<std::array<int32_t, 2>> no_constraints;
    std::vector<Triple> triangles;
    triangles_.reserve(static_cast<std::size_t>(input_.polygon_count));
    triangle_polygon_.reserve(static_cast<std::size_t>(input_.polygon_count));
    triangle_facet_.reserve(static_cast<std::size_t>(input_.polygon_count));
    for (int64_t polygon = 0; polygon < input_.polygon_count; ++polygon) {
      const int64_t begin = input_.polygon_offsets[polygon];
      const int64_t count = input_.polygon_offsets[polygon + 1] - begin;
      const int32_t* loop = input_.polygon_vertices + begin;
      const int32_t status = triangulate_planar_loop(
          input_.points, input_.point_count, loop, count, no_constraints, triangles);
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

  // A fixed boundary is never edited, so each source triangle stays a face of
  // the cells resting on it. A cell's relative determinant (|det(E)| over the
  // largest product of the three edge lengths at a vertex, the measure the
  // kernels floor at kMetricDeterminantFloor = 2^-22) is at most, at a corner
  // v of that face, |e1| |e2| sin(theta_v) |e3| / (|e1| |e2| |e3|) =
  // sin(theta_v): at most the sine of the triangle's smallest angle,
  // 2A / (l_max l_mid). A triangle whose smallest-angle sine is below the
  // floor therefore admits no cell above it: every cell on it stays below
  // the floor whatever refinement and improvement do, and they would only
  // shuffle vertices around it. Such a triangle (flat to rounding, such as a
  // seam between two coincident rings) is refused before recovery
  // (UNRESOLVED, INVALID_FACET naming the source triangle and its authored
  // polygon), with its smallest-angle sine and the floor as the
  // deviation/bound evidence. The binary64 evaluation error of the sine is a
  // few u = 2^-53 relative to the longest edges, far below the floor, so the
  // decision is the measure's. Measured in the authored coordinates.
  int32_t check_flat_triangles() {
    if (input_.policy != BoundaryPolicy::kFixed) return PHX_MC_OK;
    for (std::size_t t = 0; t < triangles_.size(); ++t) {
      const Triple& face = triangles_[t];
      std::array<double, 3> squared{};
      for (int k = 0; k < 3; ++k) {
        const double* p = point(face[static_cast<std::size_t>(k)]);
        const double* q = point(face[static_cast<std::size_t>((k + 1) % 3)]);
        for (int axis = 0; axis < 3; ++axis) {
          squared[static_cast<std::size_t>(k)] += (q[axis] - p[axis]) * (q[axis] - p[axis]);
        }
      }
      std::sort(squared.begin(), squared.end());
      std::array<double, 3> e1{}, e2{};
      for (int axis = 0; axis < 3; ++axis) {
        e1[static_cast<std::size_t>(axis)] = point(face[1])[axis] - point(face[0])[axis];
        e2[static_cast<std::size_t>(axis)] = point(face[2])[axis] - point(face[0])[axis];
      }
      const double cx = e1[1] * e2[2] - e1[2] * e2[1];
      const double cy = e1[2] * e2[0] - e1[0] * e2[2];
      const double cz = e1[0] * e2[1] - e1[1] * e2[0];
      const double sine = std::sqrt((cx * cx + cy * cy + cz * cz) / (squared[2] * squared[1]));
      if (sine < kMetricDeterminantFloor) {
        output_.source_refusal = {sine, kMetricDeterminantFloor};
        return fail(PHX_MC_UNRESOLVED, kPlcInvalidFacet, kPlcFacet, static_cast<int64_t>(t),
                    kPlcPolygon, triangle_polygon_[t]);
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

  // Whether the two triangles share an edge traversed in opposite directions.
  bool opposed(int32_t first, int32_t second) const {
    const Triple& a = triangles_[static_cast<std::size_t>(first)];
    const Triple& b = triangles_[static_cast<std::size_t>(second)];
    for (int r = 0; r < 3; ++r) {
      for (int q = 0; q < 3; ++q) {
        if (a[static_cast<std::size_t>(r)] == b[static_cast<std::size_t>((q + 1) % 3)] &&
            a[static_cast<std::size_t>((r + 1) % 3)] == b[static_cast<std::size_t>(q)]) return true;
      }
    }
    return false;
  }

  bool coplanar(int32_t first, int32_t second) const {
    const Triple& a = triangles_[static_cast<std::size_t>(first)];
    const Triple& b = triangles_[static_cast<std::size_t>(second)];
    int32_t apex = -1;
    for (int32_t vertex : b) {
      if (std::find(a.begin(), a.end(), vertex) == a.end()) apex = vertex;
    }
    return apex >= 0 && orient3d(point(a[0]), point(a[1]), point(a[2]), point(apex)) == 0;
  }

  // A facet declares a plane when its row of facet_planes is not all NaN.
  // The row must then be finite with exactly independent axes and its
  // tolerance finite and nonnegative, else the facet is refused
  // (SOURCE_DEVIATION, FACET). Every vertex of the facet's polygons must be
  // admitted by the plane (declared_plane_admits), else the facet and the
  // first vertex refused, with that vertex's certified distance to the plane
  // and the tolerance as evidence.
  int32_t check_planes() {
    if (input_.facet_planes == nullptr) {
      return input_.facet_plane_tolerances == nullptr ? PHX_MC_OK : PHX_MC_INVALID_ARGUMENT;
    }
    const auto facets = static_cast<std::size_t>(input_.facet_count);
    declared_.assign(facets, 0);
    std::vector<std::array<Expansion, 3>> normals(facets);
    for (std::size_t facet = 0; facet < facets; ++facet) {
      const double* row = input_.facet_planes + 9 * facet;
      if (std::all_of(row, row + 9, [](double value) { return std::isnan(value); })) continue;
      const double tolerance =
          input_.facet_plane_tolerances == nullptr ? 0.0 : input_.facet_plane_tolerances[facet];
      std::array<Expansion, 3> normal = declared_plane_normal(row);
      if (!std::all_of(row, row + 9, [](double value) { return std::isfinite(value); }) ||
          !std::isfinite(tolerance) || tolerance < 0.0 ||
          std::all_of(normal.begin(), normal.end(),
                      [](const Expansion& component) { return component.is_zero(); })) {
        return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcFacet,
                    static_cast<int64_t>(facet));
      }
      declared_[facet] = 1;
      normals[facet] = std::move(normal);
    }
    for (int64_t polygon = 0; polygon < input_.polygon_count; ++polygon) {
      const auto facet = static_cast<std::size_t>(input_.polygon_facets[polygon]);
      if (declared_[facet] == 0) continue;
      const double tolerance =
          input_.facet_plane_tolerances == nullptr ? 0.0 : input_.facet_plane_tolerances[facet];
      for (int64_t corner = input_.polygon_offsets[polygon];
           corner < input_.polygon_offsets[polygon + 1]; ++corner) {
        const int32_t vertex = input_.polygon_vertices[corner];
        double distance = 0.0;
        if (!declared_plane_admits(input_.facet_planes + 9 * facet, normals[facet], tolerance,
                                   point(vertex), distance, input_.affine_metric)) {
          output_.source_refusal = {distance, tolerance};
          return fail(PHX_MC_INVALID_INPUT, kPlcSourceDeviation, kPlcFacet,
                      static_cast<int64_t>(facet), kPlcPoint, vertex);
        }
      }
    }
    return PHX_MC_OK;
  }

  bool declared(int32_t facet) const {
    return !declared_.empty() && declared_[static_cast<std::size_t>(facet)] != 0;
  }

  void build_groups() {
    // The uses sorted by (edge key, triangle): a stable counting sort on the
    // lower endpoint of the uses listed in triangle order, then each lower
    // endpoint's few uses by key, stably.
    std::size_t vertex_bound = 0;
    for (const Triple& face : triangles_) {
      for (const int32_t vertex : face) {
        vertex_bound = std::max(vertex_bound, static_cast<std::size_t>(vertex) + 1);
      }
    }
    std::vector<uint32_t> starts(vertex_bound + 1, 0U);
    for (const Triple& face : triangles_) {
      for (int k = 0; k < 3; ++k) {
        ++starts[static_cast<std::size_t>(std::min(face[k], face[(k + 1) % 3])) + 1];
      }
    }
    std::partial_sum(starts.begin(), starts.end(), starts.begin());
    std::vector<std::pair<uint64_t, int32_t>> uses(3 * triangles_.size());
    {
      std::vector<uint32_t> cursor(starts.begin(), starts.end() - 1);
      for (int32_t triangle = 0; triangle < static_cast<int32_t>(triangles_.size()); ++triangle) {
        const Triple& face = triangles_[static_cast<std::size_t>(triangle)];
        for (int k = 0; k < 3; ++k) {
          const uint64_t key = topology::undirected_edge(face[k], face[(k + 1) % 3]);
          uses[cursor[static_cast<std::size_t>(key >> 32)]++] = {key, triangle};
        }
      }
    }
    for (std::size_t low = 0; low < vertex_bound; ++low) {
      std::stable_sort(uses.begin() + starts[low], uses.begin() + starts[low + 1],
                       [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    std::vector<int32_t> parent(triangles_.size());
    std::iota(parent.begin(), parent.end(), 0);
    edge_runs_.clear();
    triangle_edge_runs_.assign(3 * triangles_.size(), -1);
    for (std::size_t begin = 0; begin < uses.size();) {
      std::size_t end = begin + 1;
      while (end < uses.size() && uses[end].first == uses[begin].first) ++end;
      const auto run = static_cast<int32_t>(edge_runs_.size());
      edge_runs_.push_back(static_cast<uint32_t>(begin));
      for (std::size_t use = begin; use < end; ++use) {
        const auto triangle = static_cast<std::size_t>(uses[use].second);
        const Triple& face = triangles_[triangle];
        for (int k = 0; k < 3; ++k) {
          int32_t& slot = triangle_edge_runs_[3 * triangle + static_cast<std::size_t>(k)];
          if (slot < 0 &&
              topology::undirected_edge(face[k], face[(k + 1) % 3]) == uses[begin].first) {
            slot = run;
            break;
          }
        }
      }
      if (end - begin == 2) {
        const int32_t a = uses[begin].second, b = uses[begin + 1].second;
        const int32_t facet = triangle_facet_[static_cast<std::size_t>(a)];
        // A declared facet groups across every shared edge; an undeclared
        // one only where the two triangles are exactly coplanar.
        if (facet == triangle_facet_[static_cast<std::size_t>(b)] && opposed(a, b) &&
            (declared(facet) || coplanar(a, b))) {
          parent[static_cast<std::size_t>(topology::find_root(parent, a))] =
              topology::find_root(parent, b);
        }
      }
      begin = end;
    }
    edge_runs_.push_back(static_cast<uint32_t>(uses.size()));
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

  // The run of edge_uses_ of key `key`, else -1.
  int32_t edge_run(uint64_t key) const {
    const auto found = std::lower_bound(edge_uses_.begin(), edge_uses_.end(), key,
        [](const auto& use, uint64_t wanted) { return use.first < wanted; });
    if (found == edge_uses_.end() || found->first != key) return -1;
    const auto row = static_cast<uint32_t>(found - edge_uses_.begin());
    return static_cast<int32_t>(
        std::upper_bound(edge_runs_.begin(), edge_runs_.end(), row) - edge_runs_.begin() - 1);
  }

  // An edge used by exactly two triangles of one group.
  bool internal_run(std::size_t run) const {
    const uint32_t begin = edge_runs_[run];
    return edge_runs_[run + 1] - begin == 2 &&
           triangle_group_[static_cast<std::size_t>(edge_uses_[begin].second)] ==
               triangle_group_[static_cast<std::size_t>(edge_uses_[begin + 1].second)];
  }

  // Appends PLC edge row (first, second) of run `run` (-1: no triangle has
  // the edge).
  int32_t add_edge(int32_t first, int32_t second, int32_t run) {
    if (edges_.size() / 2 >=
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) return -1;
    const int32_t id = static_cast<int32_t>(edges_.size() / 2);
    edges_.push_back(first);
    edges_.push_back(second);
    row_runs_.push_back(run);
    if (run >= 0) run_rows_[static_cast<std::size_t>(run)] = id;
    return id;
  }

  int32_t build_edges() {
    const std::size_t runs = edge_runs_.size() - 1;
    // Runs whose edge already has a row.
    std::vector<char> listed(runs, 0);
    run_rows_.assign(runs, -1);
    // Rows of the segments, to refuse a repeated one.
    std::unordered_map<uint64_t, int32_t> segment_rows;
    segment_rows.reserve(static_cast<std::size_t>(input_.segment_count));
    for (int64_t segment = 0; segment < input_.segment_count; ++segment) {
      const int32_t a = input_.segments[2 * segment];
      const int32_t b = input_.segments[2 * segment + 1];
      if (a < 0 || b < 0 || a >= input_.point_count || b >= input_.point_count || a == b) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcEdge, segment);
      }
      const uint64_t key = topology::undirected_edge(a, b);
      const auto existing = segment_rows.find(key);
      if (existing != segment_rows.end()) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcEdge,
                    existing->second, kPlcEdge, segment);
      }
      const int32_t run = edge_run(key);
      const int32_t row = add_edge(a, b, run);
      if (row < 0) return PHX_MC_INDEX_RANGE_ERROR;
      segment_rows.emplace(key, row);
      if (run >= 0) listed[static_cast<std::size_t>(run)] = 1;
    }
    for (std::size_t triangle = 0; triangle < triangles_.size(); ++triangle) {
      const Triple& face = triangles_[triangle];
      for (int k = 0; k < 3; ++k) {
        const auto run = static_cast<std::size_t>(
            triangle_edge_runs_[3 * triangle + static_cast<std::size_t>(k)]);
        if (internal_run(run) || listed[run] != 0) continue;
        listed[run] = 1;
        const int32_t a = face[static_cast<std::size_t>(k)];
        const int32_t b = face[static_cast<std::size_t>((k + 1) % 3)];
        if (add_edge(std::min(a, b), std::max(a, b), static_cast<int32_t>(run)) < 0) {
          return PHX_MC_INDEX_RANGE_ERROR;
        }
      }
    }
    // Internal edges of declared facets between triangles that are not
    // exactly coplanar follow as hidden trailing rows: no planar flip can
    // recover them, so recovery keeps them as edges, while published results
    // omit them (PlcRecovery::hidden_plc_edges).
    const std::size_t visible = edges_.size() / 2;
    for (std::size_t run = 0; run < runs; ++run) {
      const int32_t a = edge_uses_[edge_runs_[run]].second;
      const int32_t b = edge_uses_[edge_runs_[run + 1] - 1].second;
      if (!internal_run(run) || listed[run] != 0 ||
          !declared(triangle_facet_[static_cast<std::size_t>(a)]) || coplanar(a, b)) {
        continue;
      }
      const Triple& first = triangles_[static_cast<std::size_t>(a)];
      const Triple& second = triangles_[static_cast<std::size_t>(b)];
      std::vector<int32_t> ends;
      for (int32_t vertex : first) {
        if (std::find(second.begin(), second.end(), vertex) != second.end()) ends.push_back(vertex);
      }
      if (ends.size() != 2) return PHX_MC_INTERNAL_ERROR;
      if (add_edge(std::min(ends[0], ends[1]), std::max(ends[0], ends[1]),
                   static_cast<int32_t>(run)) < 0) {
        return PHX_MC_INDEX_RANGE_ERROR;
      }
    }
    hidden_edges_ = edges_.size() / 2 - visible;
    return PHX_MC_OK;
  }

  // loop_rows_: the rows whose edge is an edge of an input polygon loop. A
  // loop edge lies on a triangle of its polygon (by run) or on none (by key).
  void index_loop_rows() {
    std::vector<char> loop_runs(edge_runs_.size() - 1, 0);
    std::vector<uint64_t> loose_keys;
    std::size_t triangle = 0;
    for (int64_t polygon = 0; polygon < input_.polygon_count; ++polygon) {
      const int64_t begin = input_.polygon_offsets[polygon];
      const int64_t end = input_.polygon_offsets[polygon + 1];
      const std::size_t first = triangle;
      while (triangle < triangles_.size() && triangle_polygon_[triangle] == polygon) ++triangle;
      const int32_t* loop = input_.polygon_vertices + begin;
      // A triangle loop triangulated as itself: its edges are its runs.
      if (end - begin == 3 && triangle - first == 1 &&
          std::is_permutation(loop, loop + 3, triangles_[first].begin())) {
        for (std::size_t k = 0; k < 3; ++k) {
          loop_runs[static_cast<std::size_t>(triangle_edge_runs_[3 * first + k])] = 1;
        }
        continue;
      }
      for (int64_t vertex = begin; vertex < end; ++vertex) {
        const int64_t next = vertex + 1 == end ? begin : vertex + 1;
        const uint64_t key = topology::undirected_edge(input_.polygon_vertices[vertex],
                                                       input_.polygon_vertices[next]);
        const int32_t run = edge_run(key);
        if (run >= 0) {
          loop_runs[static_cast<std::size_t>(run)] = 1;
        } else {
          loose_keys.push_back(key);
        }
      }
    }
    std::sort(loose_keys.begin(), loose_keys.end());
    loop_rows_.resize(row_runs_.size());
    for (std::size_t row = 0; row < row_runs_.size(); ++row) {
      const int32_t run = row_runs_[row];
      loop_rows_[row] =
          run >= 0 ? loop_runs[static_cast<std::size_t>(run)]
                   : static_cast<char>(std::binary_search(
                         loose_keys.begin(), loose_keys.end(),
                         topology::undirected_edge(edges_[2 * row], edges_[2 * row + 1])));
    }
  }

 public:
  // Every pair of items (triangles, segments, and points no triangle or
  // segment uses) whose closed boxes overlap is tested exactly (legal_pair).
  // An illegal input reports its first illegal pair in the order of a sweep
  // along x (items sorted by lower x, then kind and index).
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
    // The first illegal pair (i, j), i < j, in sweep order.
    std::size_t first = items.size();
    std::size_t second = items.size();
    const auto test = [&](std::size_t i, std::size_t j) {
      if (i > first || (i == first && j >= second)) return;
      if (!overlap(items[i].box, items[j].box)) return;
      if (legal_pair(items[i], items[j]) == PHX_MC_OK) return;
      first = i;
      second = j;
    };
    if (!grid_pairs(items, test)) {
      for (std::size_t i = 0; i < items.size() && first == items.size(); ++i) {
        for (std::size_t j = i + 1; j < items.size() && first == items.size(); ++j) {
          if (items[j].box.lower[0] > items[i].box.upper[0]) break;
          test(i, j);
        }
      }
    }
    if (first == items.size()) return PHX_MC_OK;
    const auto entity = [&](const Item& item) -> std::pair<int32_t, int64_t> {
      if (item.kind == kTriangle) {
        return {kPlcPolygon, triangle_polygon_[static_cast<std::size_t>(item.index)]};
      }
      if (item.kind == kSegment) return {kPlcEdge, item.index};
      return {kPlcPoint, item.index};
    };
    const auto [first_kind, first_entity] = entity(items[first]);
    const auto [second_kind, second_entity] = entity(items[second]);
    return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting,
                first_kind, first_entity, second_kind, second_entity);
  }

 private:
  // Visits every pair (i, j), i < j, of `items` whose closed boxes overlap,
  // once, through a uniform grid of cells of the median item extent: two
  // overlapping boxes both meet the cell of the larger lower corner on each
  // axis, where the pair is visited. The cell index is monotone in the
  // coordinate, so that cell is the larger of the boxes' lower cells. False,
  // visiting nothing, when the boxes would fill more than eight cells per
  // item on average (the caller sweeps instead).
  template <class Visit>
  bool grid_pairs(const std::vector<Item>& items, const Visit& visit) const {
    constexpr int64_t kAxisCells = int64_t{1} << 20;
    if (items.size() < 64) return false;
    std::vector<double> extents;
    extents.reserve(items.size());
    double origin[3];
    std::copy_n(items.front().box.lower, 3, origin);
    for (const Item& item : items) {
      double extent = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        extent = std::max(extent, item.box.upper[axis] - item.box.lower[axis]);
        origin[axis] = std::min(origin[axis], item.box.lower[axis]);
      }
      extents.push_back(extent);
    }
    const auto middle = extents.begin() + static_cast<std::ptrdiff_t>(extents.size() / 2);
    std::nth_element(extents.begin(), middle, extents.end());
    const double size = *middle;
    if (!(size > 0.0) || !std::isfinite(size)) return false;
    // kAxisCells for a quotient not below it (or not finite), checked as a
    // double: a cast of an out-of-range quotient is undefined.
    const auto cell = [&](double value, int axis) {
      const double quotient = std::floor((value - origin[axis]) / size);
      return quotient < static_cast<double>(kAxisCells) ? static_cast<int64_t>(quotient)
                                                         : kAxisCells;
    };
    // Lower and upper cells of every box (all below kAxisCells).
    std::vector<std::array<int64_t, 6>> spans(items.size());
    const std::size_t limit = 8 * items.size();
    std::size_t total = 0;
    for (std::size_t row = 0; row < items.size(); ++row) {
      std::size_t cells = 1;
      for (int axis = 0; axis < 3; ++axis) {
        spans[row][axis] = cell(items[row].box.lower[axis], axis);
        spans[row][axis + 3] = cell(items[row].box.upper[axis], axis);
        if (spans[row][axis + 3] >= kAxisCells) return false;
        cells *= static_cast<std::size_t>(spans[row][axis + 3] - spans[row][axis] + 1);
      }
      total += cells;
      if (total > limit) return false;
    }
    const auto key_of = [](int64_t x, int64_t y, int64_t z) {
      return static_cast<uint64_t>(x << 40 | y << 20 | z);
    };
    // Registrations bucketed by cell (cells numbered in order of first
    // registration through an open-addressing table), rows ascending in
    // each bucket.
    constexpr uint64_t kEmpty = ~uint64_t{0};
    std::size_t capacity = 16;
    while (capacity < 2 * total) capacity <<= 1;
    const std::size_t mask = capacity - 1;
    std::vector<uint64_t> table(capacity, kEmpty);
    std::vector<uint32_t> table_cell(capacity);
    std::vector<uint64_t> cell_keys;
    std::vector<uint32_t> starts(1, 0U);
    std::vector<uint32_t> registered(total);
    std::size_t at = 0;
    for (std::size_t row = 0; row < items.size(); ++row) {
      const std::array<int64_t, 6>& span = spans[row];
      for (int64_t x = span[0]; x <= span[3]; ++x) {
        for (int64_t y = span[1]; y <= span[4]; ++y) {
          for (int64_t z = span[2]; z <= span[5]; ++z) {
            const uint64_t key = key_of(x, y, z);
            std::size_t slot = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
            while (table[slot] != kEmpty && table[slot] != key) slot = (slot + 1) & mask;
            if (table[slot] == kEmpty) {
              table[slot] = key;
              table_cell[slot] = static_cast<uint32_t>(cell_keys.size());
              cell_keys.push_back(key);
              starts.push_back(0U);
            }
            registered[at++] = table_cell[slot];
            ++starts[table_cell[slot] + 1];
          }
        }
      }
    }
    for (std::size_t bucket = 1; bucket < starts.size(); ++bucket) {
      starts[bucket] += starts[bucket - 1];
    }
    std::vector<uint32_t> cursor(starts.begin(), starts.end() - 1);
    std::vector<uint32_t> rows(total);
    at = 0;
    for (std::size_t row = 0; row < items.size(); ++row) {
      const std::array<int64_t, 6>& span = spans[row];
      const std::size_t cells = static_cast<std::size_t>((span[3] - span[0] + 1) *
                                                         (span[4] - span[1] + 1) *
                                                         (span[5] - span[2] + 1));
      for (std::size_t k = 0; k < cells; ++k) {
        rows[cursor[registered[at++]]++] = static_cast<uint32_t>(row);
      }
    }
    for (std::size_t bucket = 0; bucket < cell_keys.size(); ++bucket) {
      const uint32_t begin = starts[bucket];
      const uint32_t end = starts[bucket + 1];
      for (uint32_t a = begin; a < end; ++a) {
        const std::array<int64_t, 6>& first = spans[rows[a]];
        for (uint32_t b = a + 1; b < end; ++b) {
          const std::array<int64_t, 6>& second = spans[rows[b]];
          if (key_of(std::max(first[0], second[0]), std::max(first[1], second[1]),
                     std::max(first[2], second[2])) != cell_keys[bucket]) {
            continue;
          }
          visit(rows[a], rows[b]);
        }
      }
    }
    return true;
  }

  // Exact sufficient test that triangles a and b meet in at most their
  // shared vertices or edge (a legal contact): two triangles sharing an edge
  // meet in it when the fourth vertex is off the plane of the first; for a
  // shared vertex, or none, the other triangle's vertices that are not
  // shared lie strictly on one side of a triangle's plane, so the other
  // triangle meets that plane in its shared vertex at most.
  bool separated_triangles(const Triple& a, const Triple& b) const {
    int shared = 0;
    for (int32_t vertex : a) shared += std::find(b.begin(), b.end(), vertex) != b.end() ? 1 : 0;
    const auto apart = [&](const Triple& plane, const Triple& other) {
      int side = 0;
      for (int32_t vertex : other) {
        if (std::find(plane.begin(), plane.end(), vertex) != plane.end()) continue;
        const int sign =
            orient3d_filtered(point(plane[0]), point(plane[1]), point(plane[2]), point(vertex));
        if (sign == 0 || (side != 0 && sign != side)) return false;
        side = sign;
      }
      return side != 0;
    };
    if (shared == 2) return apart(a, b) || opposite_in_plane(a, b);
    return shared < 2 && (apart(a, b) || apart(b, a));
  }

  // Whether triangles a and b, sharing an edge, are coplanar with their
  // apexes strictly on opposite sides of it: they then meet in the edge.
  // A projection along an axis that keeps a nondegenerate keeps every side
  // in the plane.
  bool opposite_in_plane(const Triple& a, const Triple& b) const {
    int32_t apex_a = -1, apex_b = -1;
    for (int32_t vertex : a) {
      if (std::find(b.begin(), b.end(), vertex) == b.end()) apex_a = vertex;
    }
    for (int32_t vertex : b) {
      if (std::find(a.begin(), a.end(), vertex) == a.end()) apex_b = vertex;
    }
    if (apex_a < 0 || apex_b < 0 ||
        orient3d_filtered(point(a[0]), point(a[1]), point(a[2]), point(apex_b)) != 0) {
      return false;
    }
    const int axis = projection_axis(point(a[0]), point(a[1]), point(a[2]));
    if (axis < 0) return false;
    int32_t edge[2]{};
    for (int corner = 0, count = 0; corner < 3; ++corner) {
      if (a[static_cast<std::size_t>(corner)] != apex_a) {
        edge[count++] = a[static_cast<std::size_t>(corner)];
      }
    }
    const auto planar = [&](int32_t vertex, double* out) {
      out[0] = point(vertex)[(axis + 1) % 3];
      out[1] = point(vertex)[(axis + 2) % 3];
    };
    double p[2], q[2], first[2], second[2];
    planar(edge[0], p);
    planar(edge[1], q);
    planar(apex_a, first);
    planar(apex_b, second);
    const int side_a = orient2d(p, q, first);
    const int side_b = orient2d(p, q, second);
    return side_a != 0 && side_b != 0 && side_a != side_b;
  }

  int32_t legal_pair(const Item& first, const Item& second) const {
    // Sweep order sorts kinds only among equal lower x; the tests below take
    // the lower kind first.
    if (second.kind < first.kind) return legal_pair(second, first);
    Contact contact;
    if (first.kind == kTriangle && second.kind == kTriangle) {
      const Triple& a = triangles_[static_cast<std::size_t>(first.index)];
      const Triple& b = triangles_[static_cast<std::size_t>(second.index)];
      if (separated_triangles(a, b)) return PHX_MC_OK;
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

  // Every edge's uses by the triangles bounding a region sum to zero (each
  // use signed by its direction and the region's side); the first failing
  // (region, edge key) pair in that order is reported.
  int32_t check_closure() {
    bool open = false;
    int32_t open_region = 0;
    uint64_t open_key = 0;
    std::size_t open_run = 0;
    std::vector<std::pair<int32_t, int32_t>> sums;  // (region, coefficient) of one edge
    for (std::size_t run = 0; run + 1 < edge_runs_.size(); ++run) {
      const uint64_t key = edge_uses_[edge_runs_[run]].first;
      sums.clear();
      for (uint32_t use = edge_runs_[run]; use < edge_runs_[run + 1]; ++use) {
        const auto triangle = static_cast<std::size_t>(edge_uses_[use].second);
        if (use > edge_runs_[run] && edge_uses_[use - 1].second == edge_uses_[use].second) continue;
        const int32_t facet = triangle_facet_[triangle];
        const int32_t positive = input_.facet_regions[2 * facet];
        const int32_t negative = input_.facet_regions[2 * facet + 1];
        if (positive == negative) continue;
        const Triple& face = triangles_[triangle];
        for (int k = 0; k < 3; ++k) {
          if (triangle_edge_runs_[3 * triangle + static_cast<std::size_t>(k)] !=
              static_cast<int32_t>(run)) {
            continue;
          }
          const int32_t sense =
              face[static_cast<std::size_t>(k)] < face[static_cast<std::size_t>((k + 1) % 3)] ? 1
                                                                                             : -1;
          if (positive >= 0) sums.push_back({positive, -sense});
          if (negative >= 0) sums.push_back({negative, sense});
        }
      }
      std::sort(sums.begin(), sums.end());
      for (std::size_t begin = 0; begin < sums.size();) {
        std::size_t end = begin;
        int64_t sum = 0;
        for (; end < sums.size() && sums[end].first == sums[begin].first; ++end) {
          sum += sums[end].second;
        }
        const int32_t region = sums[begin].first;
        if (sum != 0 && (!open || region < open_region ||
                         (region == open_region && key < open_key))) {
          open = true;
          open_region = region;
          open_key = key;
          open_run = run;
        }
        begin = end;
      }
    }
    if (!open) return PHX_MC_OK;
    return fail(PHX_MC_INVALID_INPUT, kPlcOpenBoundary, kPlcRegion, open_region, kPlcEdge,
                run_rows_[open_run]);
  }

  void publish_source() {
    output_.points.assign(input_.points, input_.points + 3 * input_.point_count);
    output_.plc_edges = edges_;
    output_.hidden_plc_edges = static_cast<int64_t>(hidden_edges_);
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
  const bool defer_contacts_;
  std::vector<Triple> triangles_;
  std::vector<int32_t> triangle_polygon_;
  std::vector<int32_t> triangle_facet_;
  std::vector<int32_t> triangle_group_;
  std::vector<char> declared_;
  std::size_t hidden_edges_ = 0;
  std::vector<std::pair<uint64_t, int32_t>> edge_uses_;  // (edge key, triangle), sorted
  // Start of each run of equal keys in edge_uses_ (and its end), and the run
  // of every triangle edge (3 * triangle + k: edge k -> k + 1).
  std::vector<uint32_t> edge_runs_;
  std::vector<int32_t> triangle_edge_runs_;
  std::vector<int32_t> edges_;
  // Per PLC edge row its run (-1: none), per run its row (-1: none), and
  // per row whether it is an edge of an input polygon loop (edge_uses()).
  std::vector<int32_t> row_runs_;
  std::vector<int32_t> run_rows_;
  std::vector<char> loop_rows_;
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

namespace {

// Source preparation keeping the hidden internal edge rows recovery needs.
int32_t prepare_recovery_source(const PlcInput& input, PlcRecovery& result) {
  result = PlcRecovery{};
  SourcePreparer preparer(input, result);
  return preparer.run();
}

}  // namespace

int32_t prepare_plc_source(const PlcInput& input, PlcRecovery& result) {
  const int32_t status = prepare_recovery_source(input, result);
  if (status == PHX_MC_OK) {
    result.plc_edges.resize(result.plc_edges.size() -
                            2 * static_cast<std::size_t>(result.hidden_plc_edges));
    result.hidden_plc_edges = 0;
  }
  return status;
}

namespace {

constexpr int32_t kPlcVoidRegion = -3;
constexpr int32_t kPlcCornerCount = 8;

struct PlcSourceFace {
  Triple oriented;
  int32_t facet;
  int32_t source_triangle;
};
// One face of a planar polygon's cover: its vertices in the polygon's
// orientation, the finite cell and slot carrying its constraint, and its
// place in a scan of the complex (the lower of its two cells).
struct PlcPlanarFace {
  Triple oriented;
  int32_t owner;
  int slot;
  int32_t scan_cell;
  int scan_slot;
};

struct PlcPlanarPolygon {
  int32_t facet;
  int32_t source_triangle;
  int32_t axis;
  Triple plane_vertices;
  int orientation;
  std::vector<int32_t> source_triangles;
  std::vector<std::vector<int32_t>> boundary_loops;
  std::array<double, 2> lower;
  std::array<double, 2> upper;
  bool recovered = false;
  // Complete cover from a previous pass. Only candidate vertices are stored:
  // corner and volume slots can be renumbered when a source point is added.
  std::vector<Triple> cover;
  // Every input triangle whose polygon belongs to this group, ascending.
  std::vector<int32_t> group_triangles;
  // `cover` as last verified, in scan order. It stays current until a cell
  // through one of its faces is removed or the complex is rebuilt; verifying
  // a current cover again would reproduce it.
  std::vector<PlcPlanarFace> verified_cover;
  bool cover_current = false;
};

struct PlcEdgePoint {
  double parameter;
  int32_t vertex;
};

// Subfaces keyed by sorted vertex triple, indexed by vertex and by source
// triangle so that local queries never scan the table (both indices are
// built at their first query and kept current from then on). `unverified`
// holds every key not known to be a face of the current complex: each added
// key, plus the keys the owner marks when an edit removes a cell through
// them. Iteration is in key order; lookups go through an open-addressing
// index of the map's nodes (linear probing, backward-shift erase), whose
// iterators stay valid while their entries live.
class SourceFaceTable {
 public:
  using Map = std::map<Triple, PlcSourceFace>;

  SourceFaceTable() = default;
  SourceFaceTable(const SourceFaceTable&) = delete;
  SourceFaceTable& operator=(const SourceFaceTable&) = delete;

  Map::const_iterator begin() const { return faces_.begin(); }
  Map::const_iterator end() const { return faces_.end(); }
  bool contains(const Triple& key) const { return find(key) != faces_.end(); }
  Map::const_iterator find(const Triple& key) const {
    if (index_.empty()) return faces_.end();
    const std::size_t mask = index_.size() - 1;
    for (std::size_t at = hash(key) & mask;; at = (at + 1) & mask) {
      const Entry& entry = index_[at];
      if (entry.key[0] < 0) return faces_.end();
      if (entry.key == key) return entry.face;
    }
  }
  const PlcSourceFace& at(const Triple& key) const {
    const auto found = find(key);
    return found != faces_.end() ? found->second : faces_.at(key);
  }

  bool emplace(const Triple& key, const PlcSourceFace& face) {
    if (!add(key, face)) return false;
    unverified_.insert(key);
    fresh_.insert(key);
    return true;
  }
  // emplace for the initial faces, before any complex: track_complex marks
  // every face unverified when it starts tracking one.
  bool load(const Triple& key, const PlcSourceFace& face) { return add(key, face); }
  // Room for `count` faces in the key index.
  void reserve(std::size_t count) {
    std::size_t size = 16;
    while (size < 2 * count) size *= 2;
    if (size > index_.size()) rehash(size);
  }
  bool erase(const Triple& key) {
    const auto found = find(key);
    if (found == faces_.end()) return false;
    unplace(key);
    if (indexed_) {
      for (int32_t vertex : key) drop(by_vertex_, vertex, key);
      drop(by_triangle_, found->second.source_triangle, key);
    }
    unverified_.erase(key);
    fresh_.erase(key);
    faces_.erase(found);
    return true;
  }
  void clear() {
    faces_.clear();
    index_.clear();
    by_vertex_.clear();
    by_triangle_.clear();
    indexed_ = false;
    unverified_.clear();
    fresh_.clear();
  }

  // Keys through `vertex`, and keys of source triangle `triangle`, in key order.
  std::vector<Triple> keys_through(int32_t vertex) const {
    build_indices();
    return sorted(by_vertex_, vertex);
  }
  std::vector<Triple> keys_of_triangle(int32_t triangle) const {
    build_indices();
    return sorted(by_triangle_, triangle);
  }

  void mark_unverified(const Triple& key) {
    if (contains(key)) unverified_.insert(key);
  }
  void mark_all_unverified() {
    for (const auto& [key, face] : faces_) {
      (void)face;
      unverified_.insert(key);
    }
  }
  // Faces added (or put back) since their source triangle was last
  // legalized (legalize_source_children), whose edges it rechecks.
  bool fresh(const Triple& key) const { return fresh_.contains(key); }
  void mark_fresh(const Triple& key) {
    if (contains(key)) fresh_.insert(key);
  }
  void settle(const Triple& key) { fresh_.erase(key); }
  std::set<Triple>& unverified() { return unverified_; }
  std::size_t size() const { return faces_.size(); }

 private:
  struct Entry {
    Triple key{-1, -1, -1};  // key[0] < 0: empty
    Map::const_iterator face;
  };
  static std::size_t hash(const Triple& key) {
    std::uint64_t value = static_cast<std::uint32_t>(key[0]);
    value = value * 0x9E3779B97F4A7C15ULL + static_cast<std::uint32_t>(key[1]);
    value = value * 0x9E3779B97F4A7C15ULL + static_cast<std::uint32_t>(key[2]);
    value ^= value >> 29;
    return static_cast<std::size_t>((value * 0xBF58476D1CE4E5B9ULL) >> 16);
  }
  bool add(const Triple& key, const PlcSourceFace& face) {
    const auto [inserted, fresh] = faces_.emplace(key, face);
    if (!fresh) return false;
    if (2 * faces_.size() > index_.size()) {
      rehash(std::max<std::size_t>(16, 2 * index_.size()));
    }
    place({key, inserted});
    if (indexed_) add_to_indices(key, face.source_triangle);
    return true;
  }
  void rehash(std::size_t size) {
    std::vector<Entry> previous(size);
    previous.swap(index_);
    for (const Entry& entry : previous) {
      if (entry.key[0] >= 0) place(entry);
    }
  }
  void place(const Entry& entry) {
    const std::size_t mask = index_.size() - 1;
    std::size_t at = hash(entry.key) & mask;
    while (index_[at].key[0] >= 0) at = (at + 1) & mask;
    index_[at] = entry;
  }
  // Removes the present `key`, shifting back the later entries of its run
  // whose home slot does not lie cyclically after the hole.
  void unplace(const Triple& key) {
    const std::size_t mask = index_.size() - 1;
    std::size_t hole = hash(key) & mask;
    while (index_[hole].key != key) hole = (hole + 1) & mask;
    for (std::size_t next = (hole + 1) & mask; index_[next].key[0] >= 0;
         next = (next + 1) & mask) {
      const std::size_t home = hash(index_[next].key) & mask;
      if (((next - home) & mask) >= ((next - hole) & mask)) {
        index_[hole] = index_[next];
        hole = next;
      }
    }
    index_[hole] = Entry{};
  }
  // Keys by vertex or by source triangle (both nonnegative indices).
  using Index = std::vector<std::vector<Triple>>;
  void build_indices() const {
    if (indexed_) return;
    for (const auto& [key, face] : faces_) add_to_indices(key, face.source_triangle);
    indexed_ = true;
  }
  void add_to_indices(const Triple& key, int32_t triangle) const {
    for (int32_t vertex : key) entry(by_vertex_, vertex).push_back(key);
    entry(by_triangle_, triangle).push_back(key);
  }
  static std::vector<Triple>& entry(Index& index, int32_t at) {
    const auto row = static_cast<std::size_t>(at);
    if (row >= index.size()) index.resize(row + 1);
    return index[row];
  }
  static void drop(Index& index, int32_t at, const Triple& key) {
    std::vector<Triple>& keys = index[static_cast<std::size_t>(at)];
    *std::find(keys.begin(), keys.end(), key) = keys.back();
    keys.pop_back();
  }
  static std::vector<Triple> sorted(const Index& index, int32_t at) {
    const auto row = static_cast<std::size_t>(at);
    if (row >= index.size()) return {};
    std::vector<Triple> keys = index[row];
    std::sort(keys.begin(), keys.end());
    return keys;
  }

  Map faces_;
  std::vector<Entry> index_;  // a power of two entries, at most half full
  mutable Index by_vertex_;
  mutable Index by_triangle_;
  mutable bool indexed_ = false;
  std::set<Triple> unverified_;
  std::set<Triple> fresh_;
};

// Uniform hash grid of items with axis-aligned boxes (points or segments).
// A box query visits a superset of the items whose boxes meet it, so the
// exact decisions stay with the caller. Items spanning many cells are kept
// aside and visited by every query.
class BoxGrid {
 public:
  // Cells of edge (volume / count)^(1/3) over the box [lower, upper], or the
  // largest extent over count^(1/3) when the box is flat.
  void reset(const double* lower, const double* upper, std::size_t count) {
    cells_.clear();
    spanning_.clear();
    double volume = 1.0, extent = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      origin_[axis] = lower[axis];
      volume *= upper[axis] - lower[axis];
      extent = std::max(extent, upper[axis] - lower[axis]);
    }
    const double items = static_cast<double>(std::max<std::size_t>(count, 1));
    double size = deterministic::cube_root(volume / items);
    if (!(size > extent * 0x1p-20)) size = extent / deterministic::cube_root(items);
    inverse_ = size > 0.0 && std::isfinite(size) ? 1.0 / size : 1.0;
  }

  void add(int32_t item, const double* lower, const double* upper) {
    std::array<int64_t, 3> first{}, last{};
    if (span(lower, upper, first, last) > kMaxItemCells) {
      spanning_.push_back(item);
      return;
    }
    for (int64_t i = first[0]; i <= last[0]; ++i) {
      for (int64_t j = first[1]; j <= last[1]; ++j) {
        for (int64_t k = first[2]; k <= last[2]; ++k) cells_[key(i, j, k)].push_back(item);
      }
    }
  }

  // Visits items until `visit` returns true. Returns false, visiting
  // nothing, when the box covers more cells than the grid holds items (the
  // caller then scans every item).
  template <class Visit>
  bool query(const double* lower, const double* upper, const Visit& visit) const {
    std::array<int64_t, 3> first{}, last{};
    if (span(lower, upper, first, last) > static_cast<int64_t>(cells_.size()) + kMaxItemCells) {
      return false;
    }
    for (int32_t item : spanning_) {
      if (visit(item)) return true;
    }
    for (int64_t i = first[0]; i <= last[0]; ++i) {
      for (int64_t j = first[1]; j <= last[1]; ++j) {
        for (int64_t k = first[2]; k <= last[2]; ++k) {
          const auto found = cells_.find(key(i, j, k));
          if (found == cells_.end()) continue;
          for (int32_t item : found->second) {
            if (visit(item)) return true;
          }
        }
      }
    }
    return true;
  }

 private:
  static constexpr int64_t kReach = int64_t{1} << 20;
  static constexpr int64_t kMaxItemCells = 64;

  // Cell coordinate, clamped monotonically so far boxes share boundary cells.
  int64_t cell(double value, int axis) const {
    const double scaled = std::floor((value - origin_[axis]) * inverse_);
    if (!(scaled > static_cast<double>(-kReach))) return -kReach;
    if (!(scaled < static_cast<double>(kReach - 1))) return kReach - 1;
    return static_cast<int64_t>(scaled);
  }
  int64_t span(const double* lower, const double* upper, std::array<int64_t, 3>& first,
               std::array<int64_t, 3>& last) const {
    int64_t count = 1;
    for (int axis = 0; axis < 3; ++axis) {
      first[static_cast<std::size_t>(axis)] = cell(lower[axis], axis);
      last[static_cast<std::size_t>(axis)] = cell(upper[axis], axis);
      count *= last[static_cast<std::size_t>(axis)] - first[static_cast<std::size_t>(axis)] + 1;
    }
    return count;
  }
  static uint64_t key(int64_t i, int64_t j, int64_t k) {
    return (static_cast<uint64_t>(i + kReach) << 42) | (static_cast<uint64_t>(j + kReach) << 21) |
           static_cast<uint64_t>(k + kReach);
  }

  double origin_[3] = {0.0, 0.0, 0.0};
  double inverse_ = 1.0;
  std::unordered_map<uint64_t, std::vector<int32_t>> cells_;
  std::vector<int32_t> spanning_;
};

// Binary64 coordinates as a hash key (signed zeros identified, as by ==).
struct CarrierKey {
  std::array<uint64_t, 3> bits;
  explicit CarrierKey(const double* point) : bits{} {
    for (int axis = 0; axis < 3; ++axis) {
      const double value = point[axis] == 0.0 ? 0.0 : point[axis];
      std::memcpy(&bits[static_cast<std::size_t>(axis)], &value, sizeof(double));
    }
  }
  bool operator==(const CarrierKey&) const = default;
};
struct CarrierKeyHash {
  std::size_t operator()(const CarrierKey& key) const {
    const std::uint64_t mixed = (key.bits[0] * 0x9E3779B97F4A7C15ULL) ^
                                (key.bits[1] * 0xC2B2AE3D27D4EB4FULL) ^
                                (key.bits[2] * 0x165667B19E3779F9ULL);
    return static_cast<std::size_t>(mixed ^ (mixed >> 29));
  }
};

class PlcRecoveryBuilder {
 public:
  // `edges`: the preparer's edge index of `source` (SourcePreparer::edge_uses).
  PlcRecoveryBuilder(const PlcInput& input, const PlcRecovery& source,
                     const SourceEdgeUses& edges, bool allow_preseeded = true)
      : input_(input), candidate_(source), source_edges_(edges),
        visible_edges_(source.plc_edges.size() / 2 -
                       static_cast<std::size_t>(source.hidden_plc_edges)),
        allow_preseeded_(allow_preseeded) {
    candidate_.suppression_search.node_budget = input_.search_node_budget;
    candidate_.suppression_search.byte_budget = input_.search_byte_budget;
    if (input_.facet_planes == nullptr) return;
    plane_normals_.resize(static_cast<std::size_t>(input_.facet_count));
    for (std::size_t facet = 0; facet < plane_normals_.size(); ++facet) {
      const double* row = input_.facet_planes + 9 * facet;
      if (!std::isnan(row[0])) plane_normals_[facet] = declared_plane_normal(row);
    }
  }


  // `check_contacts`: the source contact check the preparation deferred
  // (empty: it ran there). It runs before the first recovery edit unless the
  // source conforms to the first triangulation (conforming_faces), which
  // proves every contact legal; contacts_settled() tells whether that point
  // was reached. A source that conforms to the Delaunay triangulation of its
  // own points is recovered by it, without the bounding corners; one that
  // can never conform is not checked again with them.
  int32_t run(PlcRecovery& output, const std::function<int32_t()>& check_contacts) {
    if (!initialize_points()) return status_;
    std::vector<ConformingFace> conforming;
    std::optional<Conformity> conformity;
    const int32_t seeded = try_preseeded_triangulation(conforming);
    if (seeded != PHX_MC_OK) return seeded;
    if (preseeded_count_ != 0) {
      conformity = preseed_conforms_ ? Conformity::kConforms : Conformity::kNever;
    } else if (!prepare_corner_scaffold()) {
      return status_;
    } else if (source_triangulation_ready_ && corner_count_ != 0) {
      const int32_t corners = corner_count_;
      corner_count_ = 0;
      conformity = conforming_faces(conforming);
      if (conformity == Conformity::kConforms) {
        corner_points_.clear();
      } else {
        corner_count_ = corners;
        if (conformity == Conformity::kRefused) conformity.reset();
      }
    }
    if (preseeded_count_ == 0) {
      status_ = build_triangulation();
      if (status_ != PHX_MC_OK) return status_;
    }
    if (!conformity) conformity = conforming_faces(conforming);
    const bool conforms = conformity == Conformity::kConforms;
    contacts_settled_ = true;
    if (!conforms && check_contacts) {
      status_ = check_contacts();
      if (status_ != PHX_MC_OK) {
        contact_failure_ = true;
        return status_;
      }
    }
    if (conforms) {
      // Every subface is a face of the complex and every chain row an edge
      // of one, so recovery would only label them (mark_source_faces).
      TetrahedralComplex& complex = triangulation_->complex();
      for (const ConformingFace& face : conforming) {
        complex.set_facet_constraint(face.cell, face.slot, face.facet);
      }
    } else {
      index_candidate_points();
      track_complex();
      incremental_triangulation_ready_ = true;
      for (;;) {
        const bool reuse_incremental = incremental_triangulation_ready_;
        incremental_triangulation_ready_ = false;
        if (!reuse_incremental) {
          status_ = build_triangulation();
          if (status_ != PHX_MC_OK) return status_;
          track_complex();
        }
        bool restart = false;
        status_ = recover_source_edges(restart);
        if (status_ != PHX_MC_OK) return status_;
        if (restart) continue;
        status_ = recover_source_faces(restart);
        if (status_ != PHX_MC_OK) return status_;
        if (!restart) break;
      }
    }
    status_ = suppress_witnessed_vertices();
    if (status_ != PHX_MC_OK) return status_;
    // Seeds on a face, edge or vertex read the cells around it (seed_tet).
    bool tracked = !conforms;
    if (!tracked && input_.seed_count > 0) {
      track_complex();
      tracked = true;
    }
    // A conforming source without seeds is classified from its subfaces;
    // any failure is reported by classify_regions, in its order.
    if (!conforms || input_.seed_count > 0 || !classify_conforming(conforming)) {
      status_ = classify_regions();
    }
    if (status_ != PHX_MC_OK) return status_;
    if (!conforms || !conforming_embedded(conforming)) {
      if (!tracked) track_complex();
      status_ = check_embedded_entities();
      if (status_ != PHX_MC_OK) return status_;
    }
    publish();
    if (status_ != PHX_MC_OK) return status_;
    output = std::move(candidate_);
    return PHX_MC_OK;
  }

  bool contacts_settled() const { return contacts_settled_; }
  // Whether run failed on the deferred contact check (the failure is the
  // preparation's).
  bool contact_failure() const { return contact_failure_; }
  // Whether recovery proceeded with preseeded volume sites on a non-conforming
  // boundary (a proposal path that falls back cleanly if recovery refuses).
  bool used_nonconforming_preseed() const { return preseeded_count_ != 0 && !preseed_conforms_; }

  const PlcSourceRefusal& source_refusal() const { return candidate_.source_refusal; }
  // The failure, with a hidden internal edge (never published) named by its
  // facet.
  PlcFailure failure() const {
    PlcFailure published = failure_;
    const auto name = [&](int32_t& kind, int64_t& entity) {
      if (kind != kPlcEdge || entity < 0 || static_cast<std::size_t>(entity) < visible_edges_ ||
          static_cast<std::size_t>(entity) + 1 >= edge_triangle_offsets_.size()) {
        return;
      }
      const auto triangle = static_cast<std::size_t>(
          edge_triangle_rows_[edge_triangle_offsets_[static_cast<std::size_t>(entity)]]);
      kind = kPlcFacet;
      entity = input_.polygon_facets[candidate_.input_polygons[triangle]];
    };
    name(published.first_kind, published.first);
    name(published.second_kind, published.second);
    return published;
  }
  const PlcSuppressionSearch& suppression_search() const {
    return candidate_.suppression_search;
  }

  static int32_t run_cavity_fixture(const detail::SuppressionCavity& input,
                                    detail::SuppressionCavityFill& result) {
    result = detail::SuppressionCavityFill{};
    const std::size_t witnesses = input.witnessed.size();
    if (input.points.size() % 3 != 0 || input.witness_strata.size() != witnesses ||
        input.witness_corners.size() != 9 * witnesses ||
        input.witness_parameters.size() != 2 * witnesses ||
        input.walls.size() != input.wall_marks.size() || input.placeholder >= 0 ||
        input.search_node_budget < 0 || input.search_byte_budget < 0 ||
        (!input.relocation.empty() && input.relocation.size() != 7)) {
      return PHX_MC_INVALID_INPUT;
    }
    const std::size_t point_count = input.points.size() / 3;
    const auto valid_vertex = [&](int32_t vertex) {
      return vertex >= 0 && static_cast<std::size_t>(vertex) < point_count;
    };
    PlcInput empty_input;
    PlcRecovery empty_source;
    PlcRecoveryBuilder builder(empty_input, empty_source, {});
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
      builder.exact_points_.assign(
          input.witnessed[row], position,
          builder.work_points_.data() + 3 * static_cast<std::size_t>(input.witnessed[row]));
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
    result.filled = result.delaunay;
    if (!result.delaunay) {
      SearchAccount account{input.search_node_budget, input.search_byte_budget};
      result.filled = builder.wrap_cavity(boundary, input.placeholder, {}, true, filled,
                                          result.nodes, nullptr, nullptr, &account);
      result.bytes = account.bytes;
      result.depth = account.depth;
      result.exceeded = account.exceeded;
    }
    if (result.filled) {
      for (const SuppressionCell& cell : filled) result.cells.push_back(cell.v);
    } else if (!input.relocation.empty() && !result.exceeded) {
      const double* ray = input.relocation.data();
      const std::array<double, 3> direction{ray[3], ray[4], ray[5]};
      const double norm = std::sqrt(geometry::dot(direction.data(), direction.data()));
      const auto apex = static_cast<int32_t>(point_count);
      bool centered = false;
      result.relocated = builder.relocate_by_peeling(boundary, input.placeholder, apex, ray,
                                                     direction, norm, ray[6], nullptr, filled,
                                                     result.center, centered);
      if (result.relocated) {
        for (const SuppressionCell& cell : filled) {
          if (std::find(cell.v.begin(), cell.v.end(), apex) == cell.v.end()) ++result.peeled;
          result.cells.push_back(cell.v);
        }
      }
    }
    return PHX_MC_OK;
  }

 private:
  struct ConformingFace;

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
    // A repeated vertex is the same point: the orientation is exactly zero.
    if (a == b || a == c || a == d || b == c || b == d || c == d) return 0;
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
    // No live cell holds a vertex at the appended slot.
    if (!triangulation_->insert_point_slot(count, count, {})) return PHX_MC_INTERNAL_ERROR;
    work_points_.insert(work_points_.end(), coordinates, coordinates + 3);
    alive_.push_back(1);
    vertex_tet_.push_back(-1);
    incidence_.emplace_back();
    touched_mark_.push_back(0);
    triangulation_->rebind(work_points_.data(), static_cast<int64_t>(count + 1));
    ++volume_count_;
    vertex = static_cast<int32_t>(count);
    return PHX_MC_OK;
  }

  // The domain needs no corner scaffold when every triangle bounding the
  // void supports the whole source point set: the closed exterior boundary
  // is then the convex hull, whose faces are present in every triangulation
  // of the source points. Interfaces between two regions may be nonconvex.
  //
  // A triangle T supports the points iff its plane is the plane of a facet
  // of their hull. The hull triangles of the symbolically perturbed Delaunay
  // triangulation lie in supporting planes and triangulate the hull
  // boundary, so a facet through a vertex a of T holds a hull triangle h
  // incident to a, or, when a is no hull vertex, a hull triangle containing
  // a, which is then the coplanar apex of the cell below h (a point inside
  // the circumcircle of h would conflict with that cell otherwise). A
  // nondegenerate such h coplanar with T proves T supporting; a triangle
  // without one is decided by the scan of all points, so the answer is the
  // scan's for every triangle.
  bool convex_boundary() {
    const bool built = build_source_triangulation();
    std::vector<std::size_t> exterior;
    for (std::size_t triangle = 0; triangle < candidate_.input_triangles.size() / 3; ++triangle) {
      const int32_t facet = input_.polygon_facets[candidate_.input_polygons[triangle]];
      if (input_.facet_regions[2 * facet] < 0 || input_.facet_regions[2 * facet + 1] < 0) {
        exterior.push_back(triangle);
      }
    }
    if (exterior.empty()) return false;
    std::vector<int64_t> offsets;
    std::vector<std::array<int32_t, 3>> hull;
    if (built) hull_triangles_by_vertex(offsets, hull);
    for (const std::size_t triangle : exterior) {
      const int32_t* face = candidate_.input_triangles.data() + 3 * triangle;
      bool supported = false;
      for (int corner = 0; corner < 3 && !supported && !offsets.empty(); ++corner) {
        const auto vertex = static_cast<std::size_t>(face[corner]);
        for (int64_t at = offsets[vertex]; at < offsets[vertex + 1] && !supported; ++at) {
          const std::array<int32_t, 3>& h = hull[static_cast<std::size_t>(at)];
          supported = orient(face[0], face[1], face[2], h[0]) == 0 &&
                      orient(face[0], face[1], face[2], h[1]) == 0 &&
                      orient(face[0], face[1], face[2], h[2]) == 0;
        }
      }
      if (!supported && !supports_source_points(face)) return false;
    }
    return true;
  }

  // Whether all source points lie on one closed side of the plane of face.
  bool supports_source_points(const int32_t* face) const {
    bool positive = false;
    bool negative = false;
    for (int32_t vertex = 0; vertex < source_point_count_; ++vertex) {
      const int side = orient(face[0], face[1], face[2], vertex);
      positive = positive || side > 0;
      negative = negative || side < 0;
      if (positive && negative) return false;
    }
    return true;
  }

  // The Delaunay triangulation of the source points alone, kept for the
  // first build_triangulation; false (and none kept) when the points span
  // no tetrahedron.
  bool build_source_triangulation() {
    work_points_ = candidate_.points;
    const auto count = static_cast<std::size_t>(source_point_count_);
    triangulation_ = std::make_unique<Triangulation3D>(work_points_.data(), nullptr,
                                                       source_point_count_);
    triangulation_->set_exact_points(&exact_points_);
    triangulation_->set_affine_metric(input_.affine_metric);
    alive_.assign(count, 1);
    std::vector<int32_t> all(count);
    std::iota(all.begin(), all.end(), 0);
    source_triangulation_ready_ =
        triangulation_->build(brio_hilbert_order(work_points_.data(), 3, all), alive_) ==
        PHX_MC_OK;
    if (!source_triangulation_ready_) triangulation_.reset();
    return source_triangulation_ready_;
  }

  // Make the work slots, oriented corner order, and Triangulation3D walk state
  // independent of the DT3 region allocator. Physical point IDs still name
  // source rows, followed by any scaffold corners and canonical lattice order.
  int32_t normalize_preseeded_cells(std::size_t vertex_count,
                                    std::span<const int32_t> vertices) {
    TetrahedralComplex& complex = triangulation_->complex();
    auto& tets = complex.tets;
    const std::size_t tet_slots = tets.size();
    const auto workers = static_cast<uint32_t>(std::max<int32_t>(1, input_.parallelism));
    constexpr int kLead[4][4]{{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
    const auto canonicalize_range = [&](std::size_t begin, std::size_t end) {
      for (std::size_t slot = begin; slot < end; ++slot) {
        Tetrahedron& tet = tets[slot];
        if (tet.v[0] == kDeadVertex) continue;
        if (!vertices.empty()) {
          for (int32_t& v : tet.v) {
            if (v >= 0) v = vertices[static_cast<std::size_t>(v)];
          }
        }
        const auto first = std::min_element(tet.v, tet.v + 4) - tet.v;
        const int* lead = kLead[first];
        const int32_t ordered[]{tet.v[lead[1]], tet.v[lead[2]], tet.v[lead[3]]};
        const auto second = std::min_element(ordered, ordered + 3) - ordered;
        const int permutation[]{lead[0], lead[1 + second], lead[1 + (second + 1) % 3],
                                lead[1 + (second + 2) % 3]};
        const Tetrahedron original = tet;
        for (int j = 0; j < 4; ++j) {
          tet.v[j] = original.v[permutation[j]];
          tet.n[j] = original.n[permutation[j]];
        }
      }
    };
    if (workers > 1 && tet_slots >= 8192) {
      const int32_t status = parallel_for(
          tet_slots, ParallelRequest{8192, workers},
          [&](std::size_t begin, std::size_t end, ChunkContext&) {
            canonicalize_range(begin, end);
          });
      if (status != PHX_MC_OK) return status;
    } else {
      canonicalize_range(0, tet_slots);
    }
    // Single-pass bucket sort by v[0] + 1 (bucket 0 holds all ghost cells;
    // buckets 1..vertex_count hold finite cells with small average degree).
    const std::size_t bucket_count = vertex_count + 1;
    std::vector<std::size_t> offsets(bucket_count + 1, 0);
    for (std::size_t slot = 0; slot < tet_slots; ++slot) {
      const Tetrahedron& tet = tets[slot];
      if (tet.v[0] == kDeadVertex) continue;
      ++offsets[static_cast<std::size_t>(tet.v[0] + 1) + 1];
    }
    for (std::size_t b = 0; b < bucket_count; ++b) offsets[b + 1] += offsets[b];
    const std::size_t total = offsets[bucket_count];
    struct CanonicalEntry {
      int32_t v1, v2, v3, slot;
    };
    std::vector<CanonicalEntry> entries(total);
    std::vector<std::size_t> cursor(offsets.begin(), offsets.end() - 1);
    for (std::size_t slot = 0; slot < tet_slots; ++slot) {
      const Tetrahedron& tet = tets[slot];
      if (tet.v[0] == kDeadVertex) continue;
      const auto bucket = static_cast<std::size_t>(tet.v[0] + 1);
      entries[cursor[bucket]++] = {tet.v[1], tet.v[2], tet.v[3], static_cast<int32_t>(slot)};
    }
    const auto less_entry = [](const CanonicalEntry& a, const CanonicalEntry& b) {
      if (a.v1 != b.v1) return a.v1 < b.v1;
      if (a.v2 != b.v2) return a.v2 < b.v2;
      return a.v3 < b.v3;
    };
    if (offsets[1] - offsets[0] > 1) {
      std::sort(entries.begin() + static_cast<std::ptrdiff_t>(offsets[0]),
                entries.begin() + static_cast<std::ptrdiff_t>(offsets[1]), less_entry);
    }
    const auto sort_buckets = [&](std::size_t begin, std::size_t end) {
      for (std::size_t b = begin; b < end; ++b) {
        const std::size_t first = offsets[b + 1];
        const std::size_t last = offsets[b + 2];
        if (last - first > 1) {
          std::sort(entries.begin() + static_cast<std::ptrdiff_t>(first),
                    entries.begin() + static_cast<std::ptrdiff_t>(last), less_entry);
        }
      }
    };
    if (workers > 1 && vertex_count >= 2048) {
      const int32_t status = parallel_for(
          vertex_count, ParallelRequest{2048, workers},
          [&](std::size_t begin, std::size_t end, ChunkContext&) { sort_buckets(begin, end); });
      if (status != PHX_MC_OK) return status;
    } else {
      sort_buckets(0, vertex_count);
    }
    std::vector<int32_t> old_to_new(tet_slots, -1);
    const auto map_range = [&](std::size_t begin, std::size_t end) {
      for (std::size_t row = begin; row < end; ++row) {
        old_to_new[static_cast<std::size_t>(entries[row].slot)] = static_cast<int32_t>(row);
      }
    };
    if (workers > 1 && total >= 8192) {
      const int32_t status = parallel_for(
          total, ParallelRequest{8192, workers},
          [&](std::size_t begin, std::size_t end, ChunkContext&) { map_range(begin, end); });
      if (status != PHX_MC_OK) return status;
    } else {
      map_range(0, total);
    }
    std::vector<Tetrahedron> canonical(total);
    const auto rebuild_range = [&](std::size_t begin, std::size_t end) {
      for (std::size_t row = begin; row < end; ++row) {
        Tetrahedron tet = tets[static_cast<std::size_t>(entries[row].slot)];
        tet.stamp = 0;
        for (int32_t& neighbor : tet.n) neighbor = old_to_new[static_cast<std::size_t>(neighbor)];
        canonical[row] = tet;
      }
    };
    if (workers > 1 && total >= 8192) {
      const int32_t status = parallel_for(
          total, ParallelRequest{8192, workers},
          [&](std::size_t begin, std::size_t end, ChunkContext&) { rebuild_range(begin, end); });
      if (status != PHX_MC_OK) return status;
    } else {
      rebuild_range(0, total);
    }
    const int32_t start = offsets[1] < total ? static_cast<int32_t>(offsets[1]) : -1;
    const int64_t finite_count = complex.finite_count;
    // Replace the Triangulation3D instance so walk_step_ and visit stamps
    // start from zero regardless of how many workers built the complex.
    auto fresh = std::make_unique<Triangulation3D>(
        work_points_.data(), nullptr, static_cast<int64_t>(vertex_count));
    fresh->set_exact_points(&exact_points_);
    fresh->set_affine_metric(input_.affine_metric);
    fresh->complex().tets.swap(canonical);
    fresh->complex().finite_count = finite_count;
    if (start >= 0) fresh->set_hint(start);
    triangulation_ = std::move(fresh);
    return PHX_MC_OK;
  }

  // The additional sites are proposals, not authored source points. Keep the
  // parallel preseeded triangulation directly when every fixed wall face
  // conforms, or with a corner scaffold when almost all wall faces conform so
  // local recovery only edits the remaining faces.
  int32_t try_preseeded_triangulation(std::vector<ConformingFace>& conforming) {
    if (!allow_preseeded_ || !(input_.uniform_radius > 0.0) ||
        input_.policy != BoundaryPolicy::kFixed || input_.affine_metric != nullptr ||
        input_.protected_facets != nullptr || input_.seed_count != 0 ||
        !planar_polygons_.empty()) {
      return PHX_MC_OK;
    }
    for (std::size_t row = 0; row < edge_chains_.size(); ++row) {
      if (edge_chains_[row].size() > 1 &&
          edge_triangle_offsets_[row] == edge_triangle_offsets_[row + 1]) {
        return PHX_MC_OK;
      }
    }
    const int32_t first = input_.facet_regions[0], second = input_.facet_regions[1];
    if (!((first == -1 && second >= 0) || (second == -1 && first >= 0))) return PHX_MC_OK;
    bool mixed_orientation = false;
    for (int64_t facet = 1; facet < input_.facet_count; ++facet) {
      const int32_t pos = input_.facet_regions[2 * facet];
      const int32_t neg = input_.facet_regions[2 * facet + 1];
      if (pos == first && neg == second) continue;
      if (pos == second && neg == first) {
        mixed_orientation = true;
        continue;
      }
      return PHX_MC_OK;
    }
    std::vector<char> wall_vertices(static_cast<std::size_t>(source_point_count_), 0);
    for (const int32_t vertex : candidate_.input_triangles) {
      wall_vertices[static_cast<std::size_t>(vertex)] = 1;
    }
    if (std::ranges::find(wall_vertices, 0) != wall_vertices.end()) return PHX_MC_OK;
    std::vector<int32_t> oriented_triangles;
    std::span<const int32_t> lattice_triangles = candidate_.input_triangles;
    if (mixed_orientation) {
      oriented_triangles = candidate_.input_triangles;
      const std::size_t triangle_count = oriented_triangles.size() / 3;
      for (std::size_t triangle = 0; triangle < triangle_count; ++triangle) {
        const int32_t facet = input_.polygon_facets[candidate_.input_polygons[triangle]];
        if (input_.facet_regions[2 * facet] != first) {
          std::swap(oriented_triangles[3 * triangle + 1], oriented_triangles[3 * triangle + 2]);
        }
      }
      lattice_triangles = oriented_triangles;
    }
    std::vector<double> sites;
    if (!prepare_owned_volume_lattice(candidate_.points, lattice_triangles,
                                      input_.uniform_radius, sites) || sites.empty()) {
      return PHX_MC_OK;
    }
    const auto count = candidate_.points.size() / 3 + sites.size() / 3;
    // Reusing the sized triangulation pays off before parallel DT3 turns on.
    constexpr std::size_t kMinimumPreseededPoints = 4096;
    if (count < kMinimumPreseededPoints || count > static_cast<std::size_t>(kMaxMeshPoints)) {
      return PHX_MC_OK;
    }
    work_points_ = candidate_.points;
    work_points_.insert(work_points_.end(), sites.begin(), sites.end());
    std::vector<int32_t> active(count);
    std::iota(active.begin(), active.end(), 0);
    const auto order = brio_hilbert_order(work_points_.data(), 3, active);
    std::vector<double> ordered_points(3 * count);
    for (std::size_t row = 0; row < count; ++row) {
      std::copy_n(work_points_.data() + 3 * static_cast<std::size_t>(order[row]),
                  3, ordered_points.data() + 3 * row);
    }
    alive_.assign(count, 1);
    triangulation_ =
        std::make_unique<Triangulation3D>(ordered_points.data(), nullptr, count);
    triangulation_->set_identities(order.data());
    const auto status = triangulation_->build(active, alive_, input_.parallelism);
    triangulation_->set_identities(nullptr);
    if (status != PHX_MC_OK) {
      triangulation_.reset();
      return fail(status, kPlcNonrepresentable, kPlcNoEntity, -1);
    }
    // Map vertices to physical IDs before checking conformity, so the canonical
    // cell sort runs only once (either directly on conformity, or after
    // extending by the corner scaffold when a few faces need local recovery).
    {
      auto& tets = triangulation_->complex().tets;
      const auto workers = static_cast<uint32_t>(std::max<int32_t>(1, input_.parallelism));
      const auto remap_range = [&](std::size_t begin, std::size_t end) {
        for (std::size_t slot = begin; slot < end; ++slot) {
          Tetrahedron& tet = tets[slot];
          if (tet.v[0] == kDeadVertex) continue;
          for (int32_t& v : tet.v) {
            if (v >= 0) v = order[static_cast<std::size_t>(v)];
          }
        }
      };
      if (workers > 1 && tets.size() >= 8192) {
        const int32_t mapped = parallel_for(
            tets.size(), ParallelRequest{8192, workers},
            [&](std::size_t begin, std::size_t end, ChunkContext&) { remap_range(begin, end); });
        if (mapped != PHX_MC_OK) {
          triangulation_.reset();
          return mapped;
        }
      } else {
        remap_range(0, tets.size());
      }
    }
    triangulation_->rebind(work_points_.data(), count);
    const Conformity conf = conforming_faces(conforming);
    if (conf == Conformity::kConforms) {
      const int32_t normalized = normalize_preseeded_cells(count, {});
      if (normalized != PHX_MC_OK) {
        triangulation_.reset();
        return normalized;
      }
      if (conforming_faces(conforming) != Conformity::kConforms) {
        triangulation_.reset();
        return PHX_MC_OK;
      }
      preseed_conforms_ = true;
      preseeded_count_ = sites.size() / 3;
      volume_count_ = preseeded_count_;
      triangulation_->complex().enable_labels();
      return PHX_MC_OK;
    }
    // Keep the preseeded triangulation when almost every wall face already
    // conforms (e.g., curved surface meshes with a handful of non-Delaunay
    // quads): extend by the corner scaffold, canonicalize once, and let local
    // recovery repair only the missing faces.
    if (conf != Conformity::kNever ||
        conforming.size() * 20 < source_faces_.size() * 19 ||
        count > static_cast<std::size_t>(kMaxMeshPoints - kPlcCornerCount)) {
      triangulation_.reset();
      return PHX_MC_OK;
    }
    if (build_corner_points(corner_points_) != PHX_MC_OK) {
      corner_points_.clear();
      triangulation_.reset();
      return PHX_MC_OK;
    }
    for (Tetrahedron& tet : triangulation_->complex().tets) {
      if (tet.v[0] == kDeadVertex) continue;
      for (int32_t& v : tet.v) {
        if (v >= source_point_count_) v += kPlcCornerCount;
      }
    }
    const std::size_t total = count + kPlcCornerCount;
    work_points_.clear();
    work_points_.reserve(3 * total);
    work_points_.insert(work_points_.end(), candidate_.points.begin(), candidate_.points.end());
    work_points_.insert(work_points_.end(), corner_points_.begin(), corner_points_.end());
    work_points_.insert(work_points_.end(), sites.begin(), sites.end());
    triangulation_->rebind(work_points_.data(), static_cast<int64_t>(total));
    triangulation_->reserve_vertices(static_cast<int64_t>(total));
    alive_.assign(total, 1);
    std::array<int32_t, kPlcCornerCount> corners{};
    for (int32_t k = 0; k < kPlcCornerCount; ++k) {
      corners[static_cast<std::size_t>(k)] = source_point_count_ + k;
    }
    if (triangulation_->extend(corners, alive_) != PHX_MC_OK) {
      corner_points_.clear();
      triangulation_.reset();
      return PHX_MC_OK;
    }
    const int32_t normalized = normalize_preseeded_cells(total, {});
    if (normalized != PHX_MC_OK) {
      corner_points_.clear();
      triangulation_.reset();
      return normalized;
    }
    corner_count_ = kPlcCornerCount;
    preseed_conforms_ = false;
    preseeded_count_ = sites.size() / 3;
    volume_count_ = preseeded_count_;
    triangulation_->complex().enable_labels();
    return PHX_MC_OK;
  }

  // Nondegenerate hull triangles of the source triangulation by vertex (CSR
  // over source points, each vertex's in cell order): those incident to it
  // and those whose plane holds it as the apex of the cell below.
  void hull_triangles_by_vertex(std::vector<int64_t>& offsets,
                                std::vector<std::array<int32_t, 3>>& hull) const {
    offsets.clear();
    hull.clear();
    const auto count = static_cast<std::size_t>(source_point_count_);
    const Triangulation3D& triangulation = *triangulation_;
    std::vector<std::pair<int32_t, std::array<int32_t, 3>>> uses;
    const TetrahedralComplex& complex = triangulation.complex();
    offsets.assign(count + 1, 0);
    for (std::size_t cell = 0; cell < complex.tets.size(); ++cell) {
      const Tetrahedron& ghost = complex.tets[cell];
      if (!complex.live(static_cast<int32_t>(cell))) continue;
      const int slot = vertex_slot(ghost, kGhostVertex);
      if (slot < 0) continue;
      std::array<int32_t, 3> h{};
      for (int k = 0, at = 0; k < 4; ++k) {
        if (k != slot) h[static_cast<std::size_t>(at++)] = ghost.v[k];
      }
      if (collinear(ref(h[0]), ref(h[1]), ref(h[2]))) continue;
      for (const int32_t vertex : h) uses.push_back({vertex, h});
      const Tetrahedron& below = complex.tets[static_cast<std::size_t>(ghost.n[slot])];
      const int32_t apex = below.v[neighbor_slot(below, static_cast<int32_t>(cell))];
      if (orient(h[0], h[1], h[2], apex) == 0) uses.push_back({apex, h});
    }
    for (const auto& use : uses) ++offsets[static_cast<std::size_t>(use.first) + 1];
    for (std::size_t vertex = 0; vertex < count; ++vertex) offsets[vertex + 1] += offsets[vertex];
    hull.resize(uses.size());
    std::vector<int64_t> cursor(offsets.begin(), offsets.end() - 1);
    for (const auto& [vertex, h] : uses) {
      hull[static_cast<std::size_t>(cursor[static_cast<std::size_t>(vertex)]++)] = h;
    }
  }

  // A polygon's triangulation diagonal is not a PLC constraint when the
  // authored polygon is planar; recover the represented polygon surface.
  void prepare_planar_polygons() {
    const std::size_t polygon_count = static_cast<std::size_t>(input_.polygon_count);
    const std::size_t triangle_count = candidate_.input_triangles.size() / 3;
    planar_polygon_index_.assign(polygon_count, -1);
    if (triangle_count < 2) return;

    std::vector<std::int32_t> facets(triangle_count);
    for (std::size_t row = 0; row < triangle_count; ++row) {
      facets[row] = input_.polygon_facets[candidate_.input_polygons[row]];
    }
    // The edge uses grouped by undirected edge in key order, each group in
    // triangle order (the runs of source_edges_), and the directed edge of
    // use `at` of run `run`: the side of its triangle in that run (the
    // second such side for a triangle's second use).
    const SourceEdgeUses& index = source_edges_;
    const auto directed = [&](std::size_t run, uint32_t at) {
      const auto triangle = static_cast<std::size_t>(index.uses[at].second);
      int skip = at > index.runs[run] && index.uses[at - 1].second == index.uses[at].second ? 1 : 0;
      const std::int32_t* face = candidate_.input_triangles.data() + 3 * triangle;
      for (int side = 0; side < 3; ++side) {
        if (index.triangle_runs[3 * triangle + static_cast<std::size_t>(side)] !=
                static_cast<std::int32_t>(run) ||
            skip-- > 0) {
          continue;
        }
        return std::pair{face[side], face[(side + 1) % 3]};
      }
      return std::pair{-1, -1};
    };

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
    for (std::size_t run = 0; run + 1 < index.runs.size(); ++run) {
      const uint32_t begin = index.runs[run];
      if (index.runs[run + 1] - begin != 2) continue;
      const std::int32_t first = index.uses[begin].second;
      const std::int32_t second = index.uses[begin + 1].second;
      const auto use = directed(run, begin);
      const auto other = directed(run, begin + 1);
      if (use.first != other.second || use.second != other.first ||
          facets[static_cast<std::size_t>(first)] != facets[static_cast<std::size_t>(second)] ||
          !coplanar(first, second)) {
        continue;
      }
      const std::int32_t first_root = root(first);
      const std::int32_t second_root = root(second);
      if (first_root != second_root) {
        parent[static_cast<std::size_t>(first_root)] = second_root;
      }
    }

    // Components by ascending root, members in ascending row (a counting
    // sort on the roots).
    std::vector<std::int32_t> roots(triangle_count);
    std::vector<std::size_t> starts(triangle_count + 1, 0);
    for (std::size_t row = 0; row < triangle_count; ++row) {
      roots[row] = root(static_cast<std::int32_t>(row));
      ++starts[static_cast<std::size_t>(roots[row]) + 1];
    }
    std::partial_sum(starts.begin(), starts.end(), starts.begin());
    std::vector<std::int32_t> component_rows(triangle_count);
    {
      std::vector<std::size_t> cursor(starts.begin(), starts.end() - 1);
      for (std::size_t row = 0; row < triangle_count; ++row) {
        component_rows[cursor[static_cast<std::size_t>(roots[row])]++] =
            static_cast<std::int32_t>(row);
      }
    }
    for (std::size_t component = 0; component < triangle_count; ++component) {
      const std::span<const std::int32_t> members(component_rows.data() + starts[component],
                                                  starts[component + 1] - starts[component]);
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
      std::array<double, 2> lower = {std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity()};
      std::array<double, 2> upper = {-std::numeric_limits<double>::infinity(),
                                     -std::numeric_limits<double>::infinity()};
      for (const auto& loop : loops) {
        for (int32_t vertex : loop) {
          const double* p = point(vertex);
          lower[0] = std::min(lower[0], p[x]);
          lower[1] = std::min(lower[1], p[y]);
          upper[0] = std::max(upper[0], p[x]);
          upper[1] = std::max(upper[1], p[y]);
        }
      }
      PlcPlanarPolygon group{facet, source_triangle, axis,
                             {reference[0], reference[1], reference[2]}, orientation,
                             std::vector<std::int32_t>(members.begin(), members.end()),
                             std::move(loops), lower, upper, false, {}, {}, {}, false};
      const std::int32_t group_index = static_cast<std::int32_t>(planar_polygons_.size());
      planar_polygons_.push_back(std::move(group));
      for (std::int32_t polygon : component_polygons) {
        planar_polygon_index_[static_cast<std::size_t>(polygon)] = group_index;
      }
    }
    for (std::size_t row = 0; row < triangle_count; ++row) {
      const std::int32_t group = planar_polygon_index_[static_cast<std::size_t>(
          candidate_.input_polygons[row])];
      if (group >= 0) {
        planar_polygons_[static_cast<std::size_t>(group)].group_triangles.push_back(
            static_cast<std::int32_t>(row));
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
    const int ray_axis = (polygon.axis + 2) % 3;
    // The leading component of a nonoverlapping expansion is within one unit
    // in its last place of the value, so a binary64 gap above this pad
    // decides the strict side of a carrier-exact loop vertex; zero is
    // undecided and falls through to the exact comparison.
    const double ray = q[1].is_zero() ? 0.0 : q[1].data()[q[1].size() - 1];
    const auto clear_side = [&](const ExactPointRef& vertex) {
      if (vertex.exact != nullptr) return 0;
      const double scaled = scale * vertex.carrier[ray_axis];
      const double gap = scaled - ray;
      const double pad = 0x1p-50 * (std::fabs(scaled) + std::fabs(ray)) +
                         std::numeric_limits<double>::min();
      return gap > pad ? 1 : (gap < -pad ? -1 : 0);
    };
    for (const auto& loop : polygon.boundary_loops) {
      for (std::size_t i = 0, j = loop.size() - 1; i < loop.size(); j = i++) {
        const int clear = clear_side(ref(loop[i]));
        if (clear != 0 && clear == clear_side(ref(loop[j]))) continue;
        const int y = (polygon.axis + 2) % 3;
        Expansion a[2], b[2];
        a[1] = coordinate(ref(loop[i]), y).scaled(scale);
        b[1] = coordinate(ref(loop[j]), y).scaled(scale);
        const int a_side = (a[1] - q[1]).sign();
        const int b_side = (b[1] - q[1]).sign();
        // A horizontal ray cannot meet an edge strictly above or below it.
        // Retain endpoints on the ray for the exact closed-boundary check.
        if ((a_side > 0 && b_side > 0) || (a_side < 0 && b_side < 0)) continue;
        const int x = (polygon.axis + 1) % 3;
        a[0] = coordinate(ref(loop[i]), x).scaled(scale);
        b[0] = coordinate(ref(loop[j]), x).scaled(scale);
        const int side = orient2d_sign(a, b, q);
        if (side == 0 && (q[0] - a[0]).sign() * (b[0] - q[0]).sign() >= 0 &&
            (q[1] - a[1]).sign() * (b[1] - q[1]).sign() >= 0) {
          return true;
        }
        const bool a_above = a_side > 0;
        const bool b_above = b_side > 0;
        const int rising = (b[1] - a[1]).sign();
        if (a_above != b_above && ((rising > 0 && side > 0) || (rising < 0 && side < 0))) {
          inside = !inside;
        }
      }
    }
    return inside;
  }

  // Closed membership of a work vertex in the polygon: 2 inside, 1 outside.
  uint8_t planar_vertex_membership(const PlcPlanarPolygon& polygon, int32_t work_vertex) const {
    // Boundary vertices are exact input coordinates. Only ordinary
    // vertices can be rejected using their carrier's projected box.
    const ExactPointRef vertex = ref(work_vertex);
    const int x = (polygon.axis + 1) % 3, y = (polygon.axis + 2) % 3;
    const double* p = vertex.carrier;
    const bool outside = vertex.exact == nullptr &&
        (p[x] < polygon.lower[0] || p[x] > polygon.upper[0] ||
         p[y] < polygon.lower[1] || p[y] > polygon.upper[1]);
    bool contained = false;
    if (!outside &&
        orient(polygon.plane_vertices[0], polygon.plane_vertices[1],
               polygon.plane_vertices[2], work_vertex) == 0) {
      Expansion q[2];
      projected(work_vertex, polygon.axis, 1.0, q);
      contained = point_in_planar_polygon(polygon, q, 1.0);
    }
    return contained ? 2 : 1;
  }

  bool face_in_planar_polygon(const PlcPlanarPolygon& polygon, const Triple& face,
                              std::vector<uint8_t>& vertex_membership) const {
    for (int slot = 0; slot < 3; ++slot) {
      uint8_t& membership = vertex_membership[static_cast<std::size_t>(face[slot])];
      if (membership == 0) membership = planar_vertex_membership(polygon, face[slot]);
      if (membership == 1) return false;
    }
    const int axis = polygon.axis;
    // c lies on the closed segment ab, distinct from both endpoints.
    const auto strictly_on_segment = [&](int32_t a, int32_t b, int32_t c) {
      return orient2d_projected(ref(a), ref(b), ref(c), axis) == 0 &&
             between_on_line(ref(a), ref(b), ref(c)) && !same_point(ref(c), ref(a)) &&
             !same_point(ref(c), ref(b));
    };
    const int x = (axis + 1) % 3, y = (axis + 2) % 3;
    for (int face_side = 0; face_side < 3; ++face_side) {
      const int32_t first = face[face_side];
      const int32_t second = face[(face_side + 1) % 3];
      const ExactPointRef side_first = ref(first), side_second = ref(second);
      const bool side_exact = side_first.exact == nullptr && side_second.exact == nullptr;
      const double side_lower[2] = {std::min(side_first.carrier[x], side_second.carrier[x]),
                                    std::min(side_first.carrier[y], side_second.carrier[y])};
      const double side_upper[2] = {std::max(side_first.carrier[x], side_second.carrier[x]),
                                    std::max(side_first.carrier[y], side_second.carrier[y])};
      for (const auto& loop : polygon.boundary_loops) {
        for (std::size_t edge = 0; edge < loop.size(); ++edge) {
          const int32_t boundary_first = loop[edge];
          const int32_t boundary_second = loop[(edge + 1) % loop.size()];
          if (side_exact) {
            // Carrier-exact segments with disjoint closed projected boxes
            // neither cross nor hold an endpoint of the other.
            const ExactPointRef edge_first = ref(boundary_first);
            const ExactPointRef edge_second = ref(boundary_second);
            if (edge_first.exact == nullptr && edge_second.exact == nullptr) {
              const double* p = edge_first.carrier;
              const double* r = edge_second.carrier;
              if (std::max(p[x], r[x]) < side_lower[0] || std::min(p[x], r[x]) > side_upper[0] ||
                  std::max(p[y], r[y]) < side_lower[1] || std::min(p[y], r[y]) > side_upper[1]) {
                continue;
              }
            }
          }
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

  // Faces of the current complex inside each planar polygon become its
  // planar subfaces. Each face is visited once, from the lower of its two
  // cells; its lower finite cell carries the constraint.
  int32_t recover_planar_polygons() {
    planar_source_faces_.clear();
    if (planar_polygons_.empty()) return PHX_MC_OK;
    // Face lookup below uses incidence; local recovery may have journaled
    // replacements since the preceding polygon pass.
    sync_complex();
    TetrahedralComplex& complex = triangulation_->complex();
    using PlanarFace = PlcPlanarFace;
    std::vector<PlanarFace> covered;
    std::vector<std::pair<int32_t, int>> scan;
    std::vector<Triple> replaced;
    // The triangulation and exact positions do not change during a polygon
    // scan, so each vertex's plane/closed-membership result is reusable.
    std::vector<uint8_t> vertex_membership(work_points_.size() / 3);
    for (PlcPlanarPolygon& polygon : planar_polygons_) {
      covered.clear();
      bool reuse = polygon.recovered && !polygon.cover.empty();
      const bool retained = reuse && polygon.cover_current;
      if (retained) covered = polygon.verified_cover;
      if (!retained) {
      std::fill(vertex_membership.begin(), vertex_membership.end(), 0);
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
      // A planar source need not already be a surface of the unconstrained
      // tetrahedralization (in particular an internal material interface).
      // Stage the alternate triangulation; retain its authored subfaces for
      // exact facet recovery until complete coverage has been established.
      if (reuse) {
        for (const Triple& key : polygon.cover) {
          int32_t cell = -1;
          int slot = -1;
          if (!face_cell(key, cell, slot) ||
              !face_in_planar_polygon(polygon, key, vertex_membership)) {
            reuse = false;
            break;
          }
          const int32_t other = complex.tets[static_cast<std::size_t>(cell)].n[slot];
          const int32_t scan_cell = std::min(cell, other);
          const int scan_slot = scan_cell == cell
              ? slot : neighbor_slot(complex.tets[static_cast<std::size_t>(other)], cell);
          const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(scan_cell)];
          const int32_t neighbor_cell = tet.n[scan_slot];
          const Tetrahedron& neighbor = complex.tets[static_cast<std::size_t>(neighbor_cell)];
          const int back = neighbor_slot(neighbor, scan_cell);
          if (is_ghost(tet) && is_ghost(neighbor)) {
            reuse = false;
            break;
          }
          const int32_t owner = is_ghost(tet) ? neighbor_cell : scan_cell;
          const int owner_slot = is_ghost(tet) ? back : scan_slot;
          Triple oriented{};
          oriented_facet(complex.tets[static_cast<std::size_t>(owner)].v, owner_slot,
                         oriented.data());
          if (!is_ghost(tet) && !is_ghost(neighbor)) {
            Triple twin{};
            oriented_facet(neighbor.v, back, twin.data());
            if (FacetKey::of(oriented.data()) != FacetKey::of(twin.data()).reversed()) {
              reuse = false;
              break;
            }
          }
          if (orient2d_projected(ref(oriented[0]), ref(oriented[1]), ref(oriented[2]),
                                 polygon.axis) != polygon.orientation) {
            std::swap(oriented[1], oriented[2]);
          }
          for (int corner = 0; corner < 3; ++corner) {
            Expansion first[2], second[2];
            projected(oriented[corner], polygon.axis, 1.0, first);
            projected(oriented[(corner + 1) % 3], polygon.axis, 1.0, second);
            mesh_area = mesh_area + first[0] * second[1] - first[1] * second[0];
          }
          covered.push_back({oriented, owner, owner_slot, scan_cell, scan_slot});
        }
        if (reuse && (mesh_area - polygon_area).sign() != 0) reuse = false;
        if (reuse) {
          // The current faces still exactly cover the polygon. In a conforming
          // tetrahedral complex no additional nondegenerate face can lie in
          // that cover. Restore the full scan's cell/slot order before staging
          // constraints, including its intersection-error ordering.
          std::sort(covered.begin(), covered.end(), [](const PlanarFace& a, const PlanarFace& b) {
            return std::pair{a.scan_cell, a.scan_slot} < std::pair{b.scan_cell, b.scan_slot};
          });
        } else {
          covered.clear();
          mesh_area = Expansion{};
        }
      }
      if (!reuse) {
        // A face inside the polygon has its three vertices in the closed
        // polygon, so only the faces of the cells around those vertices can
        // be. In (cell, slot) order, from the lower cell of each face, they
        // are met exactly as a scan of the whole complex meets them.
        scan.clear();
        for (std::size_t vertex = 0; vertex < incidence_.size(); ++vertex) {
          const std::vector<int32_t>& cells = incidence_[vertex];
          if (cells.empty()) continue;
          uint8_t& membership = vertex_membership[vertex];
          if (membership == 0) {
            membership = planar_vertex_membership(polygon, static_cast<int32_t>(vertex));
          }
          if (membership != 2) continue;
          for (int32_t cell : cells) {
            const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
            for (int slot = 0; slot < 4; ++slot) {
              if (tet.v[slot] != static_cast<int32_t>(vertex) && tet.n[slot] >= cell) {
                scan.emplace_back(cell, slot);
              }
            }
          }
        }
        std::sort(scan.begin(), scan.end());
        scan.erase(std::unique(scan.begin(), scan.end()), scan.end());
        for (const auto& [cell, slot] : scan) {
          {
            const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
            const int32_t other = tet.n[slot];
            const Tetrahedron& neighbor = complex.tets[static_cast<std::size_t>(other)];
            const int back = neighbor_slot(neighbor, cell);
            if (is_ghost(tet) && is_ghost(neighbor)) continue;
            const int32_t owner = is_ghost(tet) ? other : cell;
            const int owner_slot = is_ghost(tet) ? back : slot;
            Triple oriented{};
            oriented_facet(complex.tets[static_cast<std::size_t>(owner)].v, owner_slot,
                           oriented.data());
            const Triple key = sorted_triple(oriented.data());
            if (!face_in_planar_polygon(polygon, key, vertex_membership)) continue;
            if (!is_ghost(tet) && !is_ghost(neighbor)) {
              Triple twin{};
              oriented_facet(neighbor.v, back, twin.data());
              if (FacetKey::of(oriented.data()) != FacetKey::of(twin.data()).reversed()) {
                return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable,
                            kPlcFacet, polygon.facet);
              }
            }
            if (orient2d_projected(ref(oriented[0]), ref(oriented[1]), ref(oriented[2]),
                                   polygon.axis) != polygon.orientation) {
              std::swap(oriented[1], oriented[2]);
            }
            for (int corner = 0; corner < 3; ++corner) {
              Expansion first[2], second[2];
              projected(oriented[corner], polygon.axis, 1.0, first);
              projected(oriented[(corner + 1) % 3], polygon.axis, 1.0, second);
              mesh_area = mesh_area + first[0] * second[1] - first[1] * second[0];
            }
            covered.push_back({oriented, owner, owner_slot, cell, slot});
          }
        }
      }
      if (covered.empty() || (mesh_area - polygon_area).sign() != 0) {
        if (!polygon.recovered) continue;
        return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable,
                    kPlcFacet, polygon.facet);
      }
      }
      replaced.clear();
      const int32_t group = planar_polygon_index_[static_cast<std::size_t>(
          candidate_.input_polygons[static_cast<std::size_t>(polygon.source_triangle)])];
      // The subfaces of this group's triangles, in the table's key order.
      for (std::int32_t triangle :
           planar_polygons_[static_cast<std::size_t>(group)].group_triangles) {
        for (const Triple& key : source_faces_.keys_of_triangle(triangle)) {
          replaced.push_back(key);
        }
      }
      std::sort(replaced.begin(), replaced.end());
      for (const Triple& key : replaced) source_faces_.erase(key);
      for (const PlanarFace& face : covered) {
        if (!planar_source_faces_.emplace(
                sorted_triple(face.oriented.data()),
                PlcSourceFace{face.oriented, polygon.facet, polygon.source_triangle})) {
          return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting,
                      kPlcFacet, polygon.facet);
        }
        complex.set_facet_constraint(face.owner, face.slot, polygon.facet);
      }
      if (!reuse) {
        polygon.cover.clear();
        const int32_t candidate_count = static_cast<int32_t>(candidate_.points.size() / 3);
        for (const PlanarFace& face : covered) {
          const Triple key = sorted_triple(face.oriented.data());
          if (key[0] < 0 || key[2] >= candidate_count) {
            polygon.cover.clear();
            break;
          }
          polygon.cover.push_back(key);
        }
      }
      if (!retained) {
        for (const PlanarFace& face : polygon.verified_cover) {
          cover_polygon_.erase(sorted_triple(face.oriented.data()));
        }
        polygon.cover_current = !polygon.cover.empty();
        polygon.verified_cover.clear();
        if (polygon.cover_current) {
          polygon.verified_cover = covered;
          const auto index = static_cast<int32_t>(&polygon - planar_polygons_.data());
          for (const PlanarFace& face : covered) {
            cover_polygon_[sorted_triple(face.oriented.data())] = index;
          }
        }
      }
      polygon.recovered = true;
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
    source_faces_.reserve(candidate_.input_triangles.size() / 3);
    for (std::size_t triangle = 0; triangle < candidate_.input_triangles.size() / 3; ++triangle) {
      const int32_t polygon = candidate_.input_polygons[triangle];
      const int32_t facet = input_.polygon_facets[polygon];
      const std::size_t offset = 3 * triangle;
      const Triple face{candidate_.input_triangles[offset],
                        candidate_.input_triangles[offset + 1],
                        candidate_.input_triangles[offset + 2]};
      if (!source_faces_.load(sorted_triple(face.data()),
                                 PlcSourceFace{face, facet, static_cast<int32_t>(triangle)})) {
        status_ = fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting,
                       kPlcPolygon, polygon);
        return false;
      }
    }
    prepare_planar_polygons();
    const int32_t point_status = validate_points(
        candidate_.points.data(), static_cast<int64_t>(candidate_.points.size() / 3), 3, nullptr);
    if (point_status != PHX_MC_OK) {
      status_ = fail(point_status, kPlcNonrepresentable, kPlcPoint, -1);
      return false;
    }
    // The preparer lists the segments first, as given, then the rows its
    // edge index (source_edges_) runs through.
    const std::size_t edge_count = candidate_.plc_edges.size() / 2;
    source_segment_rows_.assign(edge_count, -1);
    for (int32_t row = 0; row < input_.segment_count; ++row) {
      source_segment_rows_[static_cast<std::size_t>(row)] = row;
    }
    edge_triangle_offsets_.clear();
    edge_triangle_offsets_.reserve(edge_count + 1);
    edge_triangle_rows_.clear();
    edge_triangle_rows_.reserve(source_edges_.uses.size());
    edge_triangle_offsets_.push_back(0);
    for (std::size_t edge = 0; edge < edge_count; ++edge) {
      const int32_t run = source_edges_.row_runs[edge];
      if (run >= 0) {
        const auto row = static_cast<std::size_t>(run);
        for (uint32_t use = source_edges_.runs[row]; use < source_edges_.runs[row + 1]; ++use) {
          edge_triangle_rows_.push_back(source_edges_.uses[use].second);
        }
      }
      edge_triangle_offsets_.push_back(edge_triangle_rows_.size());
      if (source_segment_rows_[edge] < 0 && source_edges_.loop_rows[edge] == 0 && run < 0) {
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
    source_protection_ = protection_radii(
        candidate_.points.data(), source_point_count_, source_edges,
        candidate_.input_triangles, input_.affine_metric);
    source_protection_edges_ = std::move(source_edges);
    shell_apex_.assign(static_cast<std::size_t>(source_point_count_), 0);
    for (std::size_t vertex = 0; vertex < source_protection_.size(); ++vertex) {
      shell_apex_[vertex] = source_protection_[vertex] > 0.0 ? 1 : 0;
    }
    candidate_.witnesses.resize(static_cast<std::size_t>(source_point_count_));
    candidate_.vertex_dimension.resize(static_cast<std::size_t>(source_point_count_), 0);
    return true;
  }

  int32_t build_corner_points(std::vector<double>& corners) const {
    corners.clear();
    corners.reserve(3 * static_cast<std::size_t>(kPlcCornerCount));
    double lower[3], upper[3];
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = upper[axis] = candidate_.points[static_cast<std::size_t>(axis)];
      for (int32_t vertex = 1; vertex < source_point_count_; ++vertex) {
        const double value = candidate_.points[3 * static_cast<std::size_t>(vertex) + axis];
        lower[axis] = std::min(lower[axis], value);
        upper[axis] = std::max(upper[axis], value);
      }
    }
    double reach = 0.0;
    for (int axis = 0; axis < 3; ++axis) reach = std::max(reach, upper[axis] - lower[axis]);
    reach = 4.0 * power_of_two_ceil(std::max(reach, 0x1p-60));
    for (int32_t corner = 0; corner < kPlcCornerCount; ++corner) {
      for (int axis = 0; axis < 3; ++axis) {
        const double center = lower[axis] + 0.5 * (upper[axis] - lower[axis]);
        corners.push_back((corner >> axis) & 1 ? center + reach : center - reach);
      }
    }
    return validate_points(corners.data(), kPlcCornerCount, 3, nullptr);
  }

  bool prepare_corner_scaffold() {
    corner_count_ = convex_boundary() ? 0 : kPlcCornerCount;
    if (source_point_count_ > kMaxMeshPoints - corner_count_) {
      status_ = fail(PHX_MC_INDEX_RANGE_ERROR, kPlcNonrepresentable, kPlcPoint,
                     source_point_count_);
      return false;
    }
    if (corner_count_ == 0) return true;
    const auto status = build_corner_points(corner_points_);
    if (status == PHX_MC_OK) return true;
    status_ = fail(status, kPlcNonrepresentable, kPlcPoint, -1);
    return false;
  }

  // Indexes the source points (carrier grid and exact carriers) and the
  // source edge rows (carrier-box grid, chain rows through each vertex, row
  // of each chain interval).
  void index_candidate_points() {
    const double* points = candidate_.points.data();
    double lower[3], upper[3];
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = upper[axis] = points[axis];
      for (int32_t vertex = 1; vertex < source_point_count_; ++vertex) {
        lower[axis] = std::min(lower[axis], points[3 * static_cast<std::size_t>(vertex) + axis]);
        upper[axis] = std::max(upper[axis], points[3 * static_cast<std::size_t>(vertex) + axis]);
      }
    }
    point_grid_.reset(lower, upper, static_cast<std::size_t>(source_point_count_));
    edge_grid_.reset(lower, upper, edge_chains_.size());
    carrier_vertex_.clear();
    carrier_vertex_.reserve(static_cast<std::size_t>(source_point_count_));
    vertex_rows_.clear();
    for (int32_t vertex = 0; vertex < source_point_count_; ++vertex) {
      register_candidate_point(vertex);
    }
    chain_intervals_.clear();
    for (std::size_t row = 0; row < edge_chains_.size(); ++row) {
      if (edge_chains_[row].size() < 2) continue;
      const int32_t first = edge_chains_[row].front().vertex;
      const int32_t second = edge_chains_[row].back().vertex;
      // Split points lie on the segment, their carriers within a rounding.
      double row_lower[3], row_upper[3];
      for (int axis = 0; axis < 3; ++axis) {
        const double a = points[3 * static_cast<std::size_t>(first) + axis];
        const double b = points[3 * static_cast<std::size_t>(second) + axis];
        const double margin =
            0x1p-40 * (std::fabs(a) + std::fabs(b)) + std::numeric_limits<double>::min();
        row_lower[axis] = std::min(a, b) - margin;
        row_upper[axis] = std::max(a, b) + margin;
      }
      edge_grid_.add(static_cast<int32_t>(row), row_lower, row_upper);
      chain_intervals_.emplace(topology::undirected_edge(first, second), row);
      vertex_rows_[static_cast<std::size_t>(first)].push_back(row);
      vertex_rows_[static_cast<std::size_t>(second)].push_back(row);
    }
  }

  // Indexes candidate point `vertex`, the last one appended.
  void register_candidate_point(int32_t vertex) {
    const double* carrier = candidate_.points.data() + 3 * static_cast<std::size_t>(vertex);
    point_grid_.add(vertex, carrier, carrier);
    carrier_vertex_.emplace(CarrierKey(carrier), vertex);
    vertex_rows_.emplace_back();
  }

  // The first candidate point with this carrier, or -1.
  int32_t carrier_vertex(const double* carrier) const {
    const auto found = carrier_vertex_.find(CarrierKey(carrier));
    return found == carrier_vertex_.end() ? -1 : found->second;
  }

  // The Delaunay triangulation of the candidate points and the corners. The
  // first one extends the source triangulation of convex_boundary by the
  // corners (the triangulation is unique under the symbolic perturbation,
  // so the insertion order does not matter).
  int32_t build_triangulation() {
    volume_count_ = 0;
    preseeded_count_ = 0;
    preseed_conforms_ = false;
    local_edits_ = false;
    if (source_triangulation_ready_) {
      source_triangulation_ready_ = false;
      work_points_.insert(work_points_.end(), corner_points_.begin(), corner_points_.end());
      const int64_t count = static_cast<int64_t>(work_points_.size() / 3);
      triangulation_->rebind(work_points_.data(), count);
      triangulation_->reserve_vertices(count);
      alive_.resize(static_cast<std::size_t>(count), 1);
      // The source triangulation is as built (convex_boundary only read it).
      std::vector<int32_t> corners(static_cast<std::size_t>(count - source_point_count_));
      std::iota(corners.begin(), corners.end(), source_point_count_);
      const int32_t status = triangulation_->extend(corners, alive_);
      if (status != PHX_MC_OK) return fail(status, kPlcNonrepresentable, kPlcNoEntity, -1);
    } else {
      triangulation_.reset();
      work_points_ = candidate_.points;
      work_points_.insert(work_points_.end(), corner_points_.begin(), corner_points_.end());
      const int64_t count = static_cast<int64_t>(work_points_.size() / 3);
      triangulation_ = std::make_unique<Triangulation3D>(work_points_.data(), nullptr, count);
      triangulation_->set_exact_points(&exact_points_);
      triangulation_->set_affine_metric(input_.affine_metric);
      alive_.assign(static_cast<std::size_t>(count), 1);
      std::vector<int32_t> all(static_cast<std::size_t>(count));
      std::iota(all.begin(), all.end(), 0);
      const std::vector<int32_t> order = brio_hilbert_order(work_points_.data(), 3, all);
      const int32_t status = triangulation_->build(order, alive_);
      if (status != PHX_MC_OK) return fail(status, kPlcNonrepresentable, kPlcNoEntity, -1);
    }
    triangulation_->complex().enable_labels();
    return PHX_MC_OK;
  }

  // Starts tracking the complex build_triangulation built for local edits:
  // incidence lists, vertex hints and the edit journal, with every subface
  // and chain row unverified.
  void track_complex() {
    const int64_t count = static_cast<int64_t>(work_points_.size() / 3);
    TetrahedralComplex& complex = triangulation_->complex();
    vertex_tet_.assign(static_cast<std::size_t>(count), -1);
    incidence_.assign(static_cast<std::size_t>(count), {});
    {
      std::vector<int32_t> degree(static_cast<std::size_t>(count), 0);
      for (const Tetrahedron& tet : complex.tets) {
        if (tet.v[0] == kDeadVertex) continue;
        for (int32_t vertex : tet.v) {
          if (vertex >= 0) ++degree[static_cast<std::size_t>(vertex)];
        }
      }
      for (std::size_t vertex = 0; vertex < incidence_.size(); ++vertex) {
        incidence_[vertex].reserve(static_cast<std::size_t>(degree[vertex]));
      }
    }
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      const Tetrahedron& tet = complex.tets[t];
      if (tet.v[0] == kDeadVertex) continue;
      for (int32_t vertex : tet.v) {
        if (vertex < 0) continue;
        incidence_[static_cast<std::size_t>(vertex)].push_back(static_cast<int32_t>(t));
        if (!is_ghost(tet)) vertex_tet_[static_cast<std::size_t>(vertex)] = static_cast<int32_t>(t);
      }
    }
    journal_.clear();
    complex.journal = &journal_;
    touched_.clear();
    touched_mark_.assign(static_cast<std::size_t>(count), 0);
    star_marks_.assign(complex.tets.size(), 0U);
    // Nothing about the new complex is known yet.
    source_faces_.mark_all_unverified();
    for (std::size_t row = 0; row < edge_chains_.size(); ++row) {
      if (edge_chains_[row].size() > 1) unverified_rows_.insert(row);
    }
    // Cell indices of the old complex name nothing in the new one.
    cover_polygon_.clear();
    for (PlcPlanarPolygon& polygon : planar_polygons_) {
      polygon.cover_current = false;
      polygon.verified_cover.clear();
    }
  }

  // Applies the journaled cell replacements to the incidence lists, and
  // marks the subfaces and chain rows through removed cells unverified.
  void sync_complex() {
    for (const CellEvent& event : journal_) {
      for (int32_t vertex : event.v) {
        if (vertex < 0) continue;
        std::vector<int32_t>& cells = incidence_[static_cast<std::size_t>(vertex)];
        if (event.created) {
          cells.push_back(event.slot);
        } else {
          const auto found = std::find(cells.begin(), cells.end(), event.slot);
          if (found != cells.end()) {
            *found = cells.back();
            cells.pop_back();
          }
        }
        touch(vertex);
      }
      if (event.created) continue;
      for (int slot = 0; slot < 4; ++slot) {
        int32_t face[3];
        oriented_facet(event.v.data(), slot, face);
        if (face[0] >= 0 && face[1] >= 0 && face[2] >= 0) {
          const Triple key = sorted_triple(face);
          source_faces_.mark_unverified(key);
          const auto cover = cover_polygon_.find(key);
          if (cover != cover_polygon_.end()) {
            planar_polygons_[static_cast<std::size_t>(cover->second)].cover_current = false;
          }
        }
      }
      for (int first = 0; first < 4; ++first) {
        for (int second = first + 1; second < 4; ++second) {
          const auto row = chain_intervals_.find(topology::undirected_edge(
              event.v[static_cast<std::size_t>(first)], event.v[static_cast<std::size_t>(second)]));
          if (row != chain_intervals_.end()) unverified_rows_.insert(row->second);
        }
      }
    }
    journal_.clear();
  }

  void touch(int32_t vertex) {
    char& mark = touched_mark_[static_cast<std::size_t>(vertex)];
    if (mark != 0) return;
    mark = 1;
    touched_.push_back(vertex);
  }

  void set_vertex_hint(int32_t vertex, int32_t cell) {
    vertex_tet_[static_cast<std::size_t>(vertex)] = cell;
    touch(vertex);
  }

  // The highest live finite cell through `vertex`, or -1.
  int32_t highest_finite_cell(int32_t vertex) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    int32_t highest = -1;
    for (int32_t cell : incidence_[static_cast<std::size_t>(vertex)]) {
      if (cell > highest && !is_ghost(complex.tets[static_cast<std::size_t>(cell)])) highest = cell;
    }
    return highest;
  }

  // A live cell (finite or ghost) with face {a, b, c}, and the slot of its
  // fourth vertex; false when the face is not in the complex.
  bool face_cell(const Triple& face, int32_t& cell, int& slot) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    for (int32_t candidate : incidence_[static_cast<std::size_t>(face[0])]) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(candidate)];
      if (vertex_slot(tet, face[1]) < 0 || vertex_slot(tet, face[2]) < 0) continue;
      cell = candidate;
      for (slot = 0; slot < 4; ++slot) {
        if (tet.v[slot] != face[0] && tet.v[slot] != face[1] && tet.v[slot] != face[2]) break;
      }
      return true;
    }
    return false;
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
    // The new source slot renumbers the corner and volume vertices, so only
    // the cells through them change.
    sync_complex();
    std::vector<int32_t> renumbered;
    for (std::size_t later = old_source_points; later < old_work_points; ++later) {
      renumbered.insert(renumbered.end(), incidence_[later].begin(), incidence_[later].end());
    }
    std::sort(renumbered.begin(), renumbered.end());
    renumbered.erase(std::unique(renumbered.begin(), renumbered.end()), renumbered.end());
    if (!triangulation_->insert_point_slot(old_source_points, old_work_points, renumbered)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    TetrahedralComplex& complex = triangulation_->complex();
    const auto slot = static_cast<std::ptrdiff_t>(old_source_points);
    work_points_.insert(work_points_.begin() + 3 * slot, candidate_.points.end() - 3,
                        candidate_.points.end());
    alive_.insert(alive_.begin() + slot, static_cast<char>(1));
    vertex_tet_.insert(vertex_tet_.begin() + slot, -1);
    incidence_.insert(incidence_.begin() + slot, std::vector<int32_t>{});
    touched_mark_.insert(touched_mark_.begin() + slot, static_cast<char>(0));
    for (int32_t& touched : touched_) {
      if (touched >= vertex) ++touched;
    }
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
    // As after a rebuild, each vertex's hint is its highest live finite cell.
    sync_complex();
    for (int32_t touched : touched_) {
      vertex_tet_[static_cast<std::size_t>(touched)] = highest_finite_cell(touched);
      touched_mark_[static_cast<std::size_t>(touched)] = 0;
    }
    touched_.clear();
    if (vertex_tet_[static_cast<std::size_t>(vertex)] < 0) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, vertex);
    }
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
        if (triangulation_->delaunay_insphere(
                ref(v[0]), ref(v[1]), ref(v[2]), ref(v[3]), position,
                v[0], v[1], v[2], v[3], vertex) > 0) {
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

  // A recovery or suppression Steiner point in the volume (add_volume_point).
  bool volume_vertex(int32_t vertex) const {
    return vertex >= static_cast<int32_t>(candidate_.points.size() / 3) + corner_count_;
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
      set_vertex_hint(first, start);
    }
    if (!triangulation_->valid_live_cell(start) ||
        vertex_slot(complex.tets[static_cast<std::size_t>(start)], first) < 0) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, first);
    }
    // With no unsynced edits the incidence list of each end is exactly its
    // star (every live cell through it), and the edge is a finite cell
    // through both: the shorter list is scanned.
    if (journal_.empty()) {
      const bool by_first = incidence_[static_cast<std::size_t>(first)].size() <=
                            incidence_[static_cast<std::size_t>(second)].size();
      const int32_t scanned = by_first ? first : second;
      const int32_t other = by_first ? second : first;
      for (const int32_t id : incidence_[static_cast<std::size_t>(scanned)]) {
        if (!triangulation_->valid_live_cell(id)) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, scanned);
        }
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
        if (vertex_slot(tet, scanned) < 0) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcPoint, scanned);
        }
        if (!is_ghost(tet) && !artificial_corner_tet(tet) && vertex_slot(tet, other) >= 0) {
          found = true;
          return PHX_MC_OK;
        }
      }
      return PHX_MC_OK;
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
    return source_segment_rows_[edge] >= 0 || source_edges_.loop_rows[edge] != 0;
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
    double squared = geometry::squared_distance(point(source_first), point(source_second));
    if (input_.affine_metric != nullptr) {
      squared = 0.0;
      for (int row = 0; row < 3; ++row) {
        double component = 0.0;
        for (int col = 0; col < 3; ++col) {
          component += input_.affine_metric[3 * row + col] *
              (point(source_second)[col] - point(source_first)[col]);
        }
        squared += component * component;
      }
    }
    const double length = std::sqrt(squared);
    const double span = second.parameter - first.parameter;
    const bool first_apex =
        first.vertex == source_first && shell_apex_[static_cast<std::size_t>(source_first)] != 0;
    const bool second_apex = second.vertex == source_second &&
                             shell_apex_[static_cast<std::size_t>(source_second)] != 0;
    if (first_apex == second_apex || !(length > 0.0)) return first.parameter + 0.5 * span;
    const double shell = power_of_two_floor(2.0 * span * length / 3.0);
    return first_apex ? first.parameter + shell / length : second.parameter - shell / length;
  }

  // True when a vertex other than p and q lies in the closed diametral ball
  // of pq, (w - p) . (w - q) <= 0, decided exactly on witnessed positions.
  bool encroached(int32_t p, int32_t q) const {
    if (input_.affine_metric != nullptr) {
      for (int32_t vertex = 0; vertex < static_cast<int32_t>(work_points_.size() / 3); ++vertex) {
        if (vertex == p || vertex == q || alive_[static_cast<std::size_t>(vertex)] == 0) continue;
        Expansion dot;
        for (int row = 0; row < 3; ++row) {
          Expansion wp, wq;
          for (int col = 0; col < 3; ++col) {
            const Expansion factor(input_.affine_metric[3 * row + col]);
            wp = wp + factor * (coordinate(ref(vertex), col) - coordinate(ref(p), col));
            wq = wq + factor * (coordinate(ref(vertex), col) - coordinate(ref(q), col));
          }
          dot = dot + wp * wq;
        }
        if (dot.sign() <= 0) return true;
      }
      return false;
    }
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
    return any_live_vertex_in_box(lower, upper, [&](int32_t vertex) {
      if (vertex == p || vertex == q) return false;
      Expansion dot;
      for (int axis = 0; axis < 3; ++axis) {
        const Expansion coordinate_w = coordinate(ref(vertex), axis);
        dot = dot + (coordinate_w - pa[axis]) * (coordinate_w - pb[axis]);
      }
      return dot.sign() <= 0;
    });
  }

  // Visits the candidate points (source and split vertices) whose carriers
  // may lie in the box [lower, upper], until `visit` returns true.
  template <class Visit>
  void visit_candidate_points(const double* lower, const double* upper,
                              const Visit& visit) const {
    if (point_grid_.query(lower, upper, visit)) return;
    const int32_t count = static_cast<int32_t>(candidate_.points.size() / 3);
    for (int32_t vertex = 0; vertex < count; ++vertex) {
      if (visit(vertex)) return;
    }
  }

  // True when `test` holds for a live work vertex whose carrier lies in the
  // closed box [lower, upper]: candidate points through the grid, then the
  // scaffold corners and volume points.
  template <class Test>
  bool any_live_vertex_in_box(const double* lower, const double* upper, const Test& test) const {
    bool found = false;
    const auto visit = [&](int32_t vertex) {
      if (alive_[static_cast<std::size_t>(vertex)] == 0) return false;
      const double* w = point(vertex);
      if (w[0] < lower[0] || w[0] > upper[0] || w[1] < lower[1] || w[1] > upper[1] ||
          w[2] < lower[2] || w[2] > upper[2]) {
        return false;
      }
      found = test(vertex);
      return found;
    };
    visit_candidate_points(lower, upper, visit);
    const int32_t count = static_cast<int32_t>(work_points_.size() / 3);
    for (int32_t vertex = static_cast<int32_t>(candidate_.points.size() / 3);
         !found && vertex < count; ++vertex) {
      visit(vertex);
    }
    return found;
  }

  // True when a vertex other than a, b, c lies in the closed diametral ball
  // of triangle abc (the smallest ball through a, b, c, centered on its
  // plane), decided exactly on witnessed positions: with u = b - a,
  // v = c - a, m = u x v and x = w - a, the center is a + o with
  // 2 |m|^2 o = |u|^2 (v x m) + |v|^2 (m x u), and w lies in the closed ball
  // iff |x - o|^2 <= |o|^2, i.e. |x|^2 |m|^2 - x . (2 |m|^2 o) <= 0. A
  // subface whose ball holds no other vertex is strongly Delaunay: it is a
  // face of every Delaunay tetrahedralization of the vertices.
  bool subface_encroached(int32_t a, int32_t b, int32_t c) const {
    if (input_.affine_metric != nullptr) {
      const auto transformed_difference = [&](int32_t vertex, Expansion* out) {
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            out[row] = out[row] + Expansion(input_.affine_metric[3 * row + col]) *
                (coordinate(ref(vertex), col) - coordinate(ref(a), col));
          }
        }
      };
      const auto dot = [](const Expansion* p, const Expansion* q) {
        return p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
      };
      Expansion u[3], v[3];
      transformed_difference(b, u);
      transformed_difference(c, v);
      const Expansion uu = dot(u, u), uv = dot(u, v), vv = dot(v, v);
      const Expansion determinant = uu * vv - uv * uv;
      if (determinant.sign() <= 0) return true;
      for (int32_t vertex = 0; vertex < static_cast<int32_t>(work_points_.size() / 3); ++vertex) {
        if (vertex == a || vertex == b || vertex == c ||
            alive_[static_cast<std::size_t>(vertex)] == 0) continue;
        Expansion x[3];
        transformed_difference(vertex, x);
        const Expansion xu = dot(x, u), xv = dot(x, v);
        if ((dot(x, x) * determinant - xu * (uu * vv - vv * uv) -
             xv * (uu * vv - uu * uv)).sign() <= 0) return true;
      }
      return false;
    }
    Expansion pa[3], u[3], v[3];
    for (int axis = 0; axis < 3; ++axis) {
      pa[axis] = coordinate(ref(a), axis);
      u[axis] = coordinate(ref(b), axis) - pa[axis];
      v[axis] = coordinate(ref(c), axis) - pa[axis];
    }
    const auto cross = [](const Expansion* p, const Expansion* q, Expansion* out) {
      out[0] = p[1] * q[2] - p[2] * q[1];
      out[1] = p[2] * q[0] - p[0] * q[2];
      out[2] = p[0] * q[1] - p[1] * q[0];
    };
    const auto dot = [](const Expansion* p, const Expansion* q) {
      return p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    };
    Expansion m[3], vm[3], mu[3], center[3];
    cross(u, v, m);
    cross(v, m, vm);
    cross(m, u, mu);
    const Expansion uu = dot(u, u), vv = dot(v, v), mm = dot(m, m);
    for (int axis = 0; axis < 3; ++axis) center[axis] = uu * vm[axis] + vv * mu[axis];
    // Carrier box of the ball, widened for the roundings of witnessed carriers.
    double carrier_center[3], lower[3], upper[3];
    const double scale = 2.0 * mm.estimate();
    if (!(scale > 0.0)) return true;
    for (int axis = 0; axis < 3; ++axis) {
      carrier_center[axis] = point(a)[axis] + center[axis].estimate() / scale;
    }
    const double radius = std::sqrt(geometry::squared_distance(carrier_center, point(a)));
    for (int axis = 0; axis < 3; ++axis) {
      const double margin =
          radius * (1.0 + 0x1p-20) + 0x1p-40 * (radius + std::fabs(carrier_center[axis]));
      lower[axis] = carrier_center[axis] - margin;
      upper[axis] = carrier_center[axis] + margin;
    }
    return any_live_vertex_in_box(lower, upper, [&](int32_t vertex) {
      if (vertex == a || vertex == b || vertex == c) return false;
      Expansion x[3];
      for (int axis = 0; axis < 3; ++axis) x[axis] = coordinate(ref(vertex), axis) - pa[axis];
      return (dot(x, x) * mm - dot(x, center)).sign() <= 0;
    });
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
    const double index_deviation = outward_distance(carrier, position);
    const double deviation = input_.affine_metric == nullptr
                                 ? index_deviation
                                 : outward_affine_distance(input_.affine_metric, carrier, position);
    // A source point can coincide with no other vertex of a valid PLC; equal
    // carriers of distinct exact points would be indistinguishable output.
    const ExactPointRef split{carrier, index_deviation == 0.0 ? nullptr : position};
    if (const int32_t existing = carrier_vertex(carrier); existing >= 0) {
      if (same_point(split, ref(existing))) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcEdge, edge,
                    kPlcPoint, existing);
      }
      return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcEdge, edge, kPlcPoint, existing);
    }
    const int32_t vertex = static_cast<int32_t>(candidate_.points.size() / 3);
    candidate_.points.insert(candidate_.points.end(), carrier, carrier + 3);
    register_candidate_point(vertex);
    // Raw PlcRecovery witnesses index the carrier-edge row; Python resolves
    // it by exact endpoints to the canonical source edge identity.
    PlcRecoveryWitness witness{};
    witness.stratum = SourceStratum::kSegment;
    witness.entity = static_cast<int32_t>(edge);
    witness.parameters[0] = parameter;
    witness.deviation = deviation;
    candidate_.witnesses.push_back(witness);
    candidate_.vertex_dimension.push_back(1);
    if (index_deviation != 0.0) exact_points_.assign(vertex, position, carrier);
    edge_chains_[edge].insert(
        edge_chains_[edge].begin() + static_cast<std::ptrdiff_t>(interval + 1),
        {parameter, vertex});
    chain_intervals_.erase(topology::undirected_edge(first.vertex, second.vertex));
    // The old interval is no chain interval any more: a subface edge along
    // it (a segment inside a facet) is legalized again.
    for (const Triple& key : source_faces_.keys_through(first.vertex)) {
      if (std::find(key.begin(), key.end(), second.vertex) != key.end()) {
        source_faces_.mark_fresh(key);
      }
    }
    chain_intervals_.emplace(topology::undirected_edge(first.vertex, vertex), edge);
    chain_intervals_.emplace(topology::undirected_edge(vertex, second.vertex), edge);
    vertex_rows_[static_cast<std::size_t>(vertex)].push_back(edge);
    unverified_rows_.insert(edge);
    const int32_t insertion_status = insert_source_vertex_incrementally(vertex);
    if (insertion_status != PHX_MC_OK) return insertion_status;
    restart = true;
    return PHX_MC_OK;
  }

  // Recovers every chain interval of every source edge, in row order. A
  // missing interval is first recovered by flips (try_edge_removal), then
  // by planar flips on an incident facet plane, then by an edge removal with
  // volume Steiner points (steiner_edge_removal), so its segment keeps its
  // vertices; otherwise an encroached interval is split by
  // split_source_edge, the pass goes on with the next edge, and the next
  // pass rescans. An interval
  // whose closed diametral ball holds no other vertex is strongly Delaunay
  // and so present in every Delaunay tetrahedralization of the vertices: it
  // can be missing only after local recovery edits, and the restart rebuilds.
  //
  // Termination. Splits happen only on encroached intervals, at the
  // concentric-shell points of split_parameter; Shewchuk (IMR 2002, segment
  // recovery with protecting shells) shows that every such split vertex has
  // insertion radius at least a constant fraction of the local feature size
  // of the PLC at that vertex, so a packing argument bounds the number of
  // split vertices. Recovery edits never remove a chain interval (flips and
  // edge removals skip them; a planar flip removes only an edge crossing the
  // target, which no source interval of a valid PLC does), so after a
  // rebuild every non-encroached interval stays present: rebuild-only
  // restarts cannot follow each other without a split in between.
  //
  // Only unverified rows are scanned: a row leaves the set once all its
  // intervals are present, and an edit removing a cell through one of them
  // (or a split of the row, or a rebuild) puts it back, so every skipped
  // interval is present.
  int32_t recover_source_edges(bool& restart) {
    restart = false;
    for (std::size_t next = 0;;) {
      sync_complex();
      const auto row = unverified_rows_.lower_bound(next);
      if (row == unverified_rows_.end()) return PHX_MC_OK;
      const std::size_t edge = *row;
      next = edge + 1;
      unverified_rows_.erase(row);
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
          break;
        }
        if (!local_edits_) {
          return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge, edge);
        }
        unverified_rows_.insert(edge);
        incremental_triangulation_ready_ = false;
        restart = true;
        return PHX_MC_OK;
      }
    }
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
      set_vertex_hint(vertex, cell);
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
      set_vertex_hint(vertex, start);
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

  double tetrahedron_shape(const double* a, const double* b, const double* c,
                           const double* d) const {
    if (input_.affine_metric != nullptr) {
      double transformed[4][3] = {};
      const double* corners[4] = {a, b, c, d};
      for (int i = 1; i < 4; ++i) {
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            transformed[i][row] += input_.affine_metric[3 * row + col] *
                (corners[i][col] - a[col]);
          }
        }
      }
      double squared = 0.0;
      for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
          squared += geometry::squared_distance(transformed[i], transformed[j]);
        }
      }
      double normal[3];
      geometry::cross(transformed[1], transformed[2], normal);
      const double volume = std::fabs(geometry::dot(normal, transformed[3]));
      return volume / (squared * std::sqrt(squared));
    }
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
  // rolled back and reported as not applied. `created` receives the ids of
  // the new cells.
  int32_t replace_cells(const std::vector<int32_t>& cells,
                        const std::vector<std::array<int32_t, 4>>& tets, bool& applied,
                        std::vector<int32_t>* created = nullptr) {
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
      for (int32_t vertex : tet.v) set_vertex_hint(vertex, id);
    }
    if (hint < 0 || !triangulation_->set_hint(hint)) return PHX_MC_INTERNAL_ERROR;
    if (created != nullptr) created->assign(edit.created().begin(), edit.created().end());
    applied = true;
    return PHX_MC_OK;
  }

  // Removes edge (u, v) so that the missing edge (a, b) appears. Both lie on
  // the closed link ring of (u, v), which they split into two chains; each is
  // triangulated by edge removal and the new chain triangulations share the
  // fins (u, a, b) and (v, b, a). This is the 3-2 flip for a ring of three.
  //
  // With `steiner`, a chain without such a triangulation is filled by cones
  // from one volume Steiner point p instead (cf. Si, TetGen -Y: a Steiner
  // point in the volume rather than on the segment). The star of (u, v) is
  // star-shaped from u, so the fins lie in it and cut it into two halves,
  // one per chain; the half of chain c0 = a, ..., cm = b is bounded by the
  // fins and the faces (u, ci, ci+1), (v, ci+1, ci) of its cells, and p is
  // a binary64 point on the segment from the midpoint of ab toward the
  // centroid of c1, ..., cm-1 (halving toward ab) that sees all of them
  // exactly and touches no source entity. The cones replace v and u by p in
  // each cell of the chain and add (u, a, b, p) and (v, a, b, p); the edit
  // validates the tiling exactly, and p stays as a volume Steiner point.
  int32_t remove_edge_toward(int32_t u, int32_t v, int32_t a, int32_t b, bool& applied,
                             bool steiner = false) {
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
    // Placeholder ids of the Steiner points of the two chains.
    constexpr int32_t kFirst = std::numeric_limits<int32_t>::min() / 2;
    std::vector<std::array<int32_t, 4>> tets;
    std::vector<std::array<double, 3>> centers;
    for (const std::vector<int32_t>* chain : {&first_chain, &second_chain}) {
      if (triangulate_link_chain(u, v, *chain, tets)) continue;
      std::array<double, 3> location{};
      const int32_t placeholder = kFirst + static_cast<int32_t>(centers.size());
      if (!steiner || !chain_cone(u, v, cells, *chain, placeholder, location, tets)) {
        return PHX_MC_OK;
      }
      centers.push_back(location);
    }
    std::vector<int32_t> added;
    for (std::size_t row = 0; row < centers.size(); ++row) {
      int32_t vertex = -1;
      const int32_t point_status = add_volume_point(centers[row].data(), vertex);
      if (point_status != PHX_MC_OK) return point_status;
      added.push_back(vertex);
      for (auto& tet : tets) {
        std::replace(tet.begin(), tet.end(), kFirst + static_cast<int32_t>(row), vertex);
      }
    }
    const int32_t replace_status = replace_cells(cells, tets, applied);
    if (!applied) {
      for (int32_t vertex : added) alive_[static_cast<std::size_t>(vertex)] = 0;
    }
    return replace_status;
  }

  // The cones of remove_edge_toward from a Steiner point over the half of
  // the star of (u, v) on chain c0 = a, ..., cm = b, appended to `tets`
  // with `placeholder` for the point, whose binary64 position goes to
  // `location`; false when no candidate point sees every face exactly.
  bool chain_cone(int32_t u, int32_t v, const std::vector<int32_t>& cells,
                  const std::vector<int32_t>& chain, int32_t placeholder,
                  std::array<double, 3>& location, std::vector<std::array<int32_t, 4>>& tets) {
    const std::size_t m = chain.size() - 1;
    if (m < 2) return false;
    const TetrahedralComplex& complex = triangulation_->complex();
    std::vector<std::array<int32_t, 4>> cones;
    for (std::size_t k = 0; k < m; ++k) {
      const auto cell = std::find_if(cells.begin(), cells.end(), [&](int32_t id) {
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(id)];
        return vertex_slot(tet, chain[k]) >= 0 && vertex_slot(tet, chain[k + 1]) >= 0;
      });
      if (cell == cells.end()) return false;
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(*cell)];
      std::array<int32_t, 4> cone{tet.v[0], tet.v[1], tet.v[2], tet.v[3]};
      cone[static_cast<std::size_t>(vertex_slot(tet, v))] = placeholder;
      cones.push_back(cone);
      cone = {tet.v[0], tet.v[1], tet.v[2], tet.v[3]};
      cone[static_cast<std::size_t>(vertex_slot(tet, u))] = placeholder;
      cones.push_back(cone);
    }
    const int32_t a = chain.front();
    const int32_t b = chain.back();
    double middle[3];
    double centroid[3] = {0.0, 0.0, 0.0};
    for (int axis = 0; axis < 3; ++axis) {
      middle[axis] = 0.5 * (point(a)[axis] + point(b)[axis]);
      for (std::size_t k = 1; k < m; ++k) {
        centroid[axis] += point(chain[k])[axis] / static_cast<double>(m - 1);
      }
    }
    for (double fraction = 0.5; fraction > 0x1p-30; fraction *= 0.5) {
      for (int axis = 0; axis < 3; ++axis) {
        location[static_cast<std::size_t>(axis)] =
            middle[axis] + fraction * (centroid[axis] - middle[axis]);
      }
      if (!geometry::point_in_domain(location.data())) continue;
      const ExactPointRef center{location.data(), nullptr};
      const auto sign = [&](const std::array<int32_t, 4>& tet) {
        ExactPointRef corners[4];
        for (int k = 0; k < 4; ++k) {
          corners[k] = tet[static_cast<std::size_t>(k)] == placeholder
              ? center : ref(tet[static_cast<std::size_t>(k)]);
        }
        return orient3d(corners[0], corners[1], corners[2], corners[3]);
      };
      if (!std::all_of(cones.begin(), cones.end(),
                       [&](const auto& tet) { return sign(tet) > 0; })) {
        continue;
      }
      std::array<int32_t, 4> fins[2] = {{u, a, b, placeholder}, {v, a, b, placeholder}};
      bool flat = false;
      for (auto& fin : fins) {
        const int side = sign(fin);
        flat = flat || side == 0;
        if (side < 0) std::swap(fin[1], fin[2]);
      }
      if (flat || !clear_of_sources(location.data())) continue;
      tets.insert(tets.end(), cones.begin(), cones.end());
      tets.insert(tets.end(), std::begin(fins), std::end(fins));
      return true;
    }
    return false;
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

  // Whether the open segment ab meets triangle `face` other than in a shared
  // vertex or edge (the narrow phase of collect_edge_cavity).
  bool segment_meets_face(int32_t a, int32_t b, const int32_t* face) const {
    const double* segment[2] = {point(a), point(b)};
    const int64_t segment_ids[2] = {a, b};
    const double* triangle[3] = {point(face[0]), point(face[1]), point(face[2])};
    const int64_t triangle_ids[3] = {face[0], face[1], face[2]};
    Contact contact;
    return intersect_segment_triangle(segment, triangle, segment_ids, triangle_ids, contact) !=
               PHX_MC_OK ||
           !legal_contact(contact.kind);
  }

  // The finite cells with a face the open segment ab meets (as
  // collect_edge_cavity decides), sorted; false when the star of a is not
  // available. ab leaves a through the face opposite a of a star cell it
  // meets, and two cells sharing a face it meets both hold it, so the cells
  // form one chain across met faces, walked from the star of a.
  bool crossing_cells(int32_t a, int32_t b, std::vector<int32_t>& crossing) {
    crossing.clear();
    if (vertex_cells(a, crossing_star_) != PHX_MC_OK) return false;
    const TetrahedralComplex& complex = triangulation_->complex();
    int32_t face[3];
    for (int32_t cell : crossing_star_) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      if (is_ghost(tet)) continue;
      oriented_facet(tet.v, vertex_slot(tet, a), face);
      if (segment_meets_face(a, b, face)) crossing.push_back(cell);
    }
    for (std::size_t at = 0; at < crossing.size(); ++at) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(crossing[at])];
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t next = tet.n[slot];
        if (!complex.live(next) || is_ghost(complex.tets[static_cast<std::size_t>(next)]) ||
            std::find(crossing.begin(), crossing.end(), next) != crossing.end()) {
          continue;
        }
        oriented_facet(tet.v, slot, face);
        if (segment_meets_face(a, b, face)) crossing.push_back(next);
      }
    }
    std::sort(crossing.begin(), crossing.end());
    return true;
  }

  // Whether the open segment ab meets the interior of the cell `v`, by the
  // narrow phase of collect_edge_cavity.
  bool segment_crosses_cell(int32_t a, int32_t b, const std::array<int32_t, 4>& v) const {
    for (int slot = 0; slot < 4; ++slot) {
      int32_t face[3];
      oriented_facet(v.data(), slot, face);
      if (segment_meets_face(a, b, face)) return true;
    }
    return false;
  }

  // Whether every cell of `cells` is finite with no constrained face, so a
  // flip replacing them moves no mark.
  bool free_cells(const std::vector<int32_t>& cells) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    return std::all_of(cells.begin(), cells.end(), [&](int32_t cell) {
      if (is_ghost(complex.tets[static_cast<std::size_t>(cell)])) return false;
      for (int slot = 0; slot < 4; ++slot) {
        if (complex.constraint(cell, slot) != kNoConstraint) return false;
      }
      return true;
    });
  }

  // A flip toward a missing edge ab: the cells it replaces, the cells
  // replacing them, and the number of cells ab would cross after it.
  struct EdgeFlip {
    std::vector<int32_t> cells;
    std::vector<std::array<int32_t, 4>> tets;
    std::size_t crossed = 0;
  };

  // The flips of the cells ab crosses (`crossing`, sorted), fewest crossed
  // cells after them first: the 2-3 flips of faces between two crossed
  // cells and the edge removals (n-to-2n-4 flips, the 3-2 flip for n = 3)
  // of their edges away from a and b, each with every new cell exactly
  // positive. Chain intervals and cells with a marked face or a ghost
  // vertex are never flipped.
  int32_t edge_flips(int32_t a, int32_t b, const std::vector<int32_t>& crossing,
                     std::vector<EdgeFlip>& flips) {
    flips.clear();
    const TetrahedralComplex& complex = triangulation_->complex();
    EdgeFlip flip;
    std::vector<std::array<int32_t, 2>> links;
    std::vector<int32_t> ring;
    const auto consider = [&]() {
      if (flip.tets.empty() || !free_cells(flip.cells)) return;
      flip.crossed = crossing.size();
      for (int32_t cell : flip.cells) {
        if (std::binary_search(crossing.begin(), crossing.end(), cell)) --flip.crossed;
      }
      for (const auto& tet : flip.tets) {
        if (orient(tet[0], tet[1], tet[2], tet[3]) <= 0) return;
        if (segment_crosses_cell(a, b, tet)) ++flip.crossed;
      }
      flips.push_back(flip);
    };
    std::set<uint64_t> tried;
    for (int32_t cell : crossing) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t other = tet.n[slot];
        if (other < cell || !std::binary_search(crossing.begin(), crossing.end(), other)) {
          continue;
        }
        const Tetrahedron& across = complex.tets[static_cast<std::size_t>(other)];
        const int far = neighbor_slot(across, cell);
        if (far < 0) continue;
        int32_t face[3];
        oriented_facet(tet.v, slot, face);
        flip.cells = {cell, other};
        flip.tets.clear();
        // The 2-3 flip is convex exactly when its three cells have one sign.
        int sign = 0;
        for (int side = 0; side < 3 && sign != 2; ++side) {
          flip.tets.push_back({face[side], face[(side + 1) % 3], tet.v[slot], across.v[far]});
          const auto& made = flip.tets.back();
          const int own = orient(made[0], made[1], made[2], made[3]);
          sign = own == 0 || (side > 0 && own != sign) ? 2 : own;
        }
        if (sign == 2) continue;
        if (sign < 0) {
          for (auto& made : flip.tets) std::swap(made[0], made[1]);
        }
        consider();
      }
      for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
          const int32_t u = tet.v[i];
          const int32_t v = tet.v[j];
          if (u == a || u == b || v == a || v == b) continue;
          const uint64_t key = topology::undirected_edge(u, v);
          if (!tried.insert(key).second || chain_intervals_.contains(key)) continue;
          const int32_t status = edge_cells(u, v, flip.cells);
          if (status != PHX_MC_OK) return status;
          flip.tets.clear();
          if (!free_cells(flip.cells) || !link_edges(flip.cells, u, v, links)) continue;
          std::vector<char> used(links.size(), 0);
          if (!link_chain(links, used, links.front()[0], links.front()[0], ring) ||
              ring.size() != links.size() + 1) {
            continue;
          }
          ring.pop_back();
          if (triangulate_link_chain(u, v, ring, flip.tets)) consider();
        }
      }
    }
    std::stable_sort(flips.begin(), flips.end(), [](const EdgeFlip& x, const EdgeFlip& y) {
      return x.crossed < y.crossed;
    });
    return PHX_MC_OK;
  }

  // Lowers the number of cells ab crosses (`crossing`, sorted) below
  // `bound` by at most `depth` flips of edge_flips, depth first in their
  // order; a flip that leads nowhere is undone. On success `lowered` is set
  // and `crossing` holds the cells ab crosses now, or `recovered` when a
  // single edge removal on the way made ab an edge; otherwise the complex
  // holds its cells as before.
  int32_t lower_crossing(std::size_t edge, int32_t a, int32_t b, std::vector<int32_t>& crossing,
                         std::size_t bound, int depth, bool& lowered, bool& recovered) {
    lowered = false;
    std::vector<EdgeFlip> flips;
    int32_t status = edge_flips(a, b, crossing, flips);
    if (status != PHX_MC_OK) return status;
    const TetrahedralComplex& complex = triangulation_->complex();
    std::vector<int32_t> next;
    std::vector<int32_t> created;
    std::vector<std::array<int32_t, 4>> removed;
    for (const EdgeFlip& flip : flips) {
      if (depth == 1 && flip.crossed >= bound) break;
      removed.clear();
      for (int32_t cell : flip.cells) {
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
        removed.push_back({tet.v[0], tet.v[1], tet.v[2], tet.v[3]});
      }
      bool applied = false;
      status = replace_cells(flip.cells, flip.tets, applied, &created);
      if (status != PHX_MC_OK) return status;
      if (!applied) continue;
      if (crossing_cells(a, b, next)) {
        if (next.size() < bound) {
          crossing = std::move(next);
          lowered = true;
          return PHX_MC_OK;
        }
        if (depth > 1) {
          status = recover_edge_by_removal(edge, a, b, next, recovered);
          if (status != PHX_MC_OK || recovered) return status;
          status = lower_crossing(edge, a, b, next, bound, depth - 1, lowered, recovered);
          if (status != PHX_MC_OK || recovered) return status;
          if (lowered) {
            crossing = std::move(next);
            return PHX_MC_OK;
          }
        }
      }
      const std::vector<int32_t> undone = created;
      status = replace_cells(undone, removed, applied);
      if (status != PHX_MC_OK) return status;
      if (!applied) {
        return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge,
                    static_cast<int64_t>(edge));
      }
    }
    return PHX_MC_OK;
  }

  // Flip recovery of a missing edge ab without new vertices (cf. Si,
  // TetGen's recoveredgebyflips and flipnm). Each round first tries the
  // single edge removal toward ab (recover_edge_by_removal), else lowers
  // the number of cells ab crosses by a short sequence of flips
  // (lower_crossing). Every flip is validated exactly (replace_cells), so
  // the complex stays a valid tetrahedralization holding every recovered
  // interval and mark. Termination: each round strictly lowers the number
  // of cells ab crosses, recounted on the complex, or ends the recovery,
  // and ab is an edge once it crosses none.
  int32_t try_edge_removal(std::size_t edge, int32_t a, int32_t b, bool& recovered) {
    constexpr int kFlipDepth = 3;
    recovered = false;
    std::vector<int32_t> crossing;
    if (!crossing_cells(a, b, crossing)) return PHX_MC_OK;
    while (!crossing.empty()) {
      int32_t status = recover_edge_by_removal(edge, a, b, crossing, recovered);
      if (status != PHX_MC_OK || recovered) return status;
      bool lowered = false;
      status = lower_crossing(edge, a, b, crossing, crossing.size(), kFlipDepth, lowered,
                              recovered);
      if (status != PHX_MC_OK || recovered || !lowered) return status;
    }
    bool present = false;
    const int32_t status = has_edge(a, b, present);
    if (status != PHX_MC_OK) return status;
    recovered = present;
    return PHX_MC_OK;
  }

  // Edge removal with volume Steiner points (remove_edge_toward with
  // `steiner`) of an edge of a cell ab crosses, away from a and b, with a and
  // b on its link; the first that applies makes ab an edge.
  int32_t steiner_edge_removal(std::size_t edge, int32_t a, int32_t b, bool& recovered) {
    recovered = false;
    std::vector<int32_t> crossing;
    if (!crossing_cells(a, b, crossing)) return PHX_MC_OK;
    const TetrahedralComplex& complex = triangulation_->complex();
    std::vector<std::array<int32_t, 2>> edges;
    for (int32_t cell : crossing) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
          const int32_t u = std::min(tet.v[i], tet.v[j]);
          const int32_t v = std::max(tet.v[i], tet.v[j]);
          if (u < 0 || u == a || u == b || v == a || v == b) continue;
          edges.push_back({u, v});
        }
      }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    for (const auto& [u, v] : edges) {
      const int32_t status = remove_edge_toward(u, v, a, b, recovered, true);
      if (status != PHX_MC_OK) {
        return fail(status, kPlcNonrepresentable, kPlcEdge, static_cast<int64_t>(edge));
      }
      if (recovered) break;
    }
    if (!recovered) return PHX_MC_OK;
    bool present = false;
    const int32_t status = has_edge(a, b, present);
    if (status != PHX_MC_OK) return status;
    if (!present) {
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcEdge,
                  static_cast<int64_t>(edge));
    }
    return PHX_MC_OK;
  }

  // Recovery of a missing source edge before any split: flips
  // (try_edge_removal), then planar flips on a covered incident facet plane,
  // then, under FIXED (where every boundary split point would be suppressed
  // again), an edge removal with volume Steiner points
  // (steiner_edge_removal). An unrealizable flip leaves a valid mesh, so
  // later recovery stages may continue from it; every other failure is
  // returned unchanged.
  int32_t try_planar_source_edge(std::size_t edge, int32_t a, int32_t b,
                                 bool& recovered) {
    int32_t status = try_edge_removal(edge, a, b, recovered);
    if (status != PHX_MC_OK || recovered) return status;
    const PlcFailure prior = failure_;
    status = recover_planar_source_edge(edge, a, b, recovered);
    if (status == PHX_MC_CONSTRAINT_INTERSECTION && failure_.reason == kPlcFixedSegment) {
      failure_ = prior;
      status = PHX_MC_OK;
    }
    if (status != PHX_MC_OK || recovered || input_.policy != BoundaryPolicy::kFixed) {
      return status;
    }
    return steiner_edge_removal(edge, a, b, recovered);
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
    std::vector<PlcSourceFace> children;
    if (current_source_face_triangles(source, children) != PHX_MC_OK) {
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
  int32_t current_source_face_triangles(const PlcSourceFace& source,
                                        std::vector<PlcSourceFace>& children) const {
    children.clear();
    std::vector<int32_t> boundary;
    std::set<uint64_t> boundary_intervals;
    for (int side = 0; side < 3; ++side) {
      const int32_t first = source.oriented[static_cast<std::size_t>(side)];
      const int32_t second =
          source.oriented[static_cast<std::size_t>((side + 1) % 3)];
      std::size_t chain_row = edge_chains_.size();
      std::size_t chain_first = 0;
      std::size_t chain_second = 0;
      for (std::size_t row : vertex_rows_[static_cast<std::size_t>(first)]) {
        const auto& chain = edge_chains_[row];
        const auto has = [&](int32_t vertex) {
          return [vertex](const PlcEdgePoint& value) { return value.vertex == vertex; };
        };
        const auto second_position = std::find_if(chain.begin(), chain.end(), has(second));
        if (second_position == chain.end()) continue;
        if (chain_row != edge_chains_.size()) return PHX_MC_INTERNAL_ERROR;
        chain_row = row;
        chain_first = static_cast<std::size_t>(
            std::find_if(chain.begin(), chain.end(), has(first)) - chain.begin());
        chain_second = static_cast<std::size_t>(second_position - chain.begin());
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
    // Rows whose carrier boxes may meet the face's box, in row order.
    std::vector<std::size_t> rows;
    if (!edge_grid_.query(box_lower, box_upper, [&](int32_t row) {
          rows.push_back(static_cast<std::size_t>(row));
          return false;
        })) {
      rows.resize(edge_chains_.size());
      std::iota(rows.begin(), rows.end(), std::size_t{0});
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (std::size_t row : rows) {
      const auto& chain = edge_chains_[row];
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
  // strictly inside the physical circumcircle of abc, decided exactly on
  // the physical source plane: the sphere through a, b, c and a + n (n the
  // physical normal of abc) meets the plane in that circumcircle. A strictly
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
      if (input_.affine_metric != nullptr) {
        Expansion transformed_b[3], transformed_c[3], transformed_d[3];
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            const Expansion factor(input_.affine_metric[3 * row + col]);
            transformed_b[row] = transformed_b[row] + factor * u[col];
            transformed_c[row] = transformed_c[row] + factor * v[col];
            transformed_d[row] = transformed_d[row] + factor * (pd[col] - pa[col]);
          }
        }
        for (int axis = 0; axis < 3; ++axis) {
          pa[axis] = Expansion();
          pb[axis] = u[axis] = transformed_b[axis];
          pc[axis] = v[axis] = transformed_c[axis];
          pd[axis] = transformed_d[axis];
        }
      }
      // orient3d(a, b, c, a + n) = |n|^2 > 0, so insphere > 0 is inside.
      pe[0] = pa[0] + (u[1] * v[2] - u[2] * v[1]);
      pe[1] = pa[1] + (u[2] * v[0] - u[0] * v[2]);
      pe[2] = pa[2] + (u[0] * v[1] - u[1] * v[0]);
      return insphere_sign(pa, pb, pc, pe, pd) > 0;
    };
    // Directed child edge -> (child key, opposite vertex), kept current
    // across flips. Only edges that can have become illegal are checked: the
    // edges of the children added since the last legalization of this
    // source triangle (fresh), then the four outer edges of each flipped
    // quadrilateral. Every other edge joins two children unchanged since
    // that legalization left it legal (a split that frees a chain interval
    // marks its children fresh, split_source_edge).
    std::map<std::pair<int32_t, int32_t>, std::pair<Triple, int32_t>> edges;
    std::set<std::pair<int32_t, int32_t>> work;
    const auto add_child = [&](const Triple& key, const Triple& oriented, bool check) {
      for (int side = 0; side < 3; ++side) {
        const int32_t p = oriented[static_cast<std::size_t>(side)];
        const int32_t q = oriented[static_cast<std::size_t>((side + 1) % 3)];
        edges[{p, q}] = {key, oriented[static_cast<std::size_t>((side + 2) % 3)]};
        if (check) work.insert({std::min(p, q), std::max(p, q)});
      }
    };
    for (const Triple& key : source_faces_.keys_of_triangle(source_triangle)) {
      add_child(key, source_faces_.at(key).oriented, source_faces_.fresh(key));
    }
    while (!work.empty()) {
      const auto [a, b] = *work.begin();
      work.erase(work.begin());
      const auto use = edges.find({a, b});
      const auto other = edges.find({b, a});
      if (use == edges.end() || other == edges.end()) continue;
      if (chain_intervals_.contains(topology::undirected_edge(a, b))) continue;
      const int32_t c = use->second.second;
      const int32_t d = other->second.second;
      ++candidate_.legalize_tests;
      if (!inside(a, b, c, d)) continue;
      const Triple use_key = use->second.first;
      const Triple other_key = other->second.first;
      const int32_t facet = source_faces_.at(use_key).facet;
      source_faces_.erase(use_key);
      source_faces_.erase(other_key);
      for (const auto& [p, q] : {std::pair{a, b}, std::pair{b, c}, std::pair{c, a},
                                 std::pair{b, a}, std::pair{a, d}, std::pair{d, b}}) {
        edges.erase({p, q});
      }
      const Triple first{a, d, c};
      const Triple second{d, b, c};
      source_faces_.emplace(sorted_triple(first.data()),
                            PlcSourceFace{first, facet, source_triangle});
      source_faces_.emplace(sorted_triple(second.data()),
                            PlcSourceFace{second, facet, source_triangle});
      add_child(sorted_triple(first.data()), first, true);
      add_child(sorted_triple(second.data()), second, true);
      work.erase({std::min(c, d), std::max(c, d)});
    }
    for (const Triple& key : source_faces_.keys_of_triangle(source_triangle)) {
      source_faces_.settle(key);
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
      // Points exactly on the edge have carriers within a rounding of its box.
      double lower[3], upper[3];
      for (int axis = 0; axis < 3; ++axis) {
        const double a = point(first)[axis], b = point(second)[axis];
        const double margin = 0x1p-40 * (std::fabs(a) + std::fabs(b)) +
                              std::numeric_limits<double>::min();
        lower[axis] = std::min(a, b) - margin;
        upper[axis] = std::max(a, b) + margin;
      }
      visit_candidate_points(lower, upper, [&](int32_t vertex) {
        if (vertex == first || vertex == second) return false;
        const double* w = point(vertex);
        if (w[0] < lower[0] || w[0] > upper[0] || w[1] < lower[1] || w[1] > upper[1] ||
            w[2] < lower[2] || w[2] > upper[2]) {
          return false;
        }
        if (collinear(ref(first), ref(second), ref(vertex)) &&
            between_on_line(ref(first), ref(second), ref(vertex))) {
          interior.push_back(vertex);
        }
        return false;
      });
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
    if (input_.affine_metric != nullptr) {
      double world_u[3] = {}, world_v[3] = {}, world_w[3] = {};
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          world_u[row] += input_.affine_metric[3 * row + col] * u[col];
          world_v[row] += input_.affine_metric[3 * row + col] * v[col];
          world_w[row] += input_.affine_metric[3 * row + col] * w[col];
        }
      }
      std::copy_n(world_u, 3, u);
      std::copy_n(world_v, 3, v);
      std::copy_n(world_w, 3, w);
    }
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
        const double scale = std::ldexp(1.0, bits);
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
      const auto row = chain_intervals_.find(topology::undirected_edge(a, b));
      if (row == chain_intervals_.end()) continue;
      Expansion dot;
      if (input_.affine_metric == nullptr) {
        for (int axis = 0; axis < 3; ++axis) {
          dot = dot + (coordinate(candidate, axis) - coordinate(ref(a), axis)) *
                          (coordinate(candidate, axis) - coordinate(ref(b), axis));
        }
      } else {
        for (int axis = 0; axis < 3; ++axis) {
          Expansion da, db;
          for (int col = 0; col < 3; ++col) {
            const Expansion factor(input_.affine_metric[3 * axis + col]);
            da = da + factor * (coordinate(candidate, col) - coordinate(ref(a), col));
            db = db + factor * (coordinate(candidate, col) - coordinate(ref(b), col));
          }
          dot = dot + da * db;
        }
      }
      if (dot.sign() > 0) continue;
      const auto& chain = edge_chains_[row->second];
      for (std::size_t interval = 0; interval + 1 < chain.size(); ++interval) {
        const int32_t first = chain[interval].vertex;
        const int32_t second = chain[interval + 1].vertex;
        if ((first == a && second == b) || (first == b && second == a)) {
          return split_source_edge(row->second, interval, restart);
        }
      }
      return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet, source.facet);
    }
    const double index_deviation = outward_distance(carrier, position);
    const double deviation = input_.affine_metric == nullptr
                                 ? index_deviation
                                 : outward_affine_distance(input_.affine_metric, carrier, position);
    // Distinct exact points with equal carriers would be indistinguishable.
    const ExactPointRef split{carrier, index_deviation == 0.0 ? nullptr : position};
    if (const int32_t existing = carrier_vertex(carrier); existing >= 0) {
      if (same_point(split, ref(existing))) {
        return fail(PHX_MC_CONSTRAINT_INTERSECTION, kPlcIntersecting, kPlcFacet, source.facet,
                    kPlcPoint, existing);
      }
      return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcFacet, source.facet, kPlcPoint,
                  existing);
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
    register_candidate_point(center);
    PlcRecoveryWitness witness{};
    witness.stratum = SourceStratum::kFacet;
    witness.entity = source.source_triangle;
    witness.parameters[0] = parameters[0];
    witness.parameters[1] = parameters[1];
    witness.deviation = deviation;
    candidate_.witnesses.push_back(witness);
    candidate_.vertex_dimension.push_back(2);
    if (index_deviation != 0.0) exact_points_.assign(center, position, carrier);
    source_faces_.erase(key);
    for (const auto& [child_key, child] : staged_source_faces) source_faces_.emplace(child_key, child);
    // The new source index still addresses an artificial corner in work_points_
    // until insertion shifts the corner and volume slots. Legalization uses
    // work-point predicates and must therefore run only after that shift.
    const int32_t insertion_status = insert_source_vertex_incrementally(center);
    if (insertion_status != PHX_MC_OK) return insertion_status;
    legalize_source_children(source.source_triangle);
    restart = true;
    return PHX_MC_OK;
  }

  // Recovers every subface: present ones are marked; a missing one is
  // recovered by in-plane flips when its facet plane is covered, else split
  // at an exactly witnessed interior point (split_source_face) and the scan
  // restarts. FIXED boundaries take the same path; their split points are
  // removed again by suppression. Each pass checks only the unverified
  // subfaces (added ones, and those through cells an edit removed): every
  // other subface is a face of the complex.
  int32_t recover_source_faces(bool& restart) {
    restart = false;
    TetrahedralComplex& complex = triangulation_->complex();
    // Constrains each subface on the lower-numbered finite cell through it.
    const auto mark_source_faces = [&]() -> int32_t {
      sync_complex();
      for (const auto& [key, source] : source_faces_) {
        int32_t cell = -1;
        int slot = -1;
        if (!face_cell(key, cell, slot)) {
          return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcFacet, source.facet);
        }
        const int32_t other = complex.tets[static_cast<std::size_t>(cell)].n[slot];
        const int32_t sides[2] = {std::min(cell, other), std::max(cell, other)};
        int32_t finite[2] = {-1, -1};
        int finite_count = 0;
        for (int32_t side : sides) {
          if (!is_ghost(complex.tets[static_cast<std::size_t>(side)])) finite[finite_count++] = side;
        }
        const auto face_slot = [&](int32_t side) {
          const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(side)];
          return side == cell ? slot : neighbor_slot(tet, cell);
        };
        if (finite_count == 2) {
          const auto positive = [&](int32_t side) {
            Triple oriented{};
            oriented_facet(complex.tets[static_cast<std::size_t>(side)].v, face_slot(side),
                           oriented.data());
            return FacetKey::of(oriented.data()) == FacetKey::of(source.oriented.data());
          };
          if (positive(finite[0]) == positive(finite[1])) {
            return fail(PHX_MC_INTERNAL_ERROR, kPlcNonrepresentable, kPlcFacet, source.facet);
          }
        } else if (finite_count != 1 || corner_count_ != 0) {
          return fail(PHX_MC_UNRESOLVED, kPlcNonrepresentable, kPlcFacet, source.facet);
        }
        complex.set_facet_constraint(finite[0], face_slot(finite[0]), source.facet);
      }
      return PHX_MC_OK;
    };

    for (;;) {
      const int32_t planar_status = recover_planar_polygons();
      if (planar_status != PHX_MC_OK) return planar_status;
      // The subfaces missing at this pass are handled in turn on the current
      // complex: one still missing (checked locally once the complex has
      // changed) is recovered by planar flips, else split. Facet splits go
      // on with the next one; an interval split (an encroached boundary) or
      // a refused incremental insertion ends the pass.
      sync_complex();
      std::vector<std::pair<Triple, PlcSourceFace>> missing;
      std::set<Triple>& unverified = source_faces_.unverified();
      for (auto key = unverified.begin(); key != unverified.end();) {
        int32_t cell = -1;
        int slot = -1;
        if (face_cell(*key, cell, slot)) {
          key = unverified.erase(key);
        } else {
          missing.push_back({*key, source_faces_.at(*key)});
          ++key;
        }
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
        // A strongly Delaunay subface is missing only because local recovery
        // edits (planar flips, star-shaped insertions) left the complex
        // non-Delaunay; it reappears in a rebuild, so it is not split. Such
        // a rebuild follows a point insertion at most once, so rebuild-only
        // restarts cannot repeat.
        if (local_edits_ && subface_rebuild_points_ != candidate_.points.size() &&
            !subface_encroached(key[0], key[1], key[2])) {
          subface_rebuild_points_ = candidate_.points.size();
          incremental_triangulation_ready_ = false;
          restart = true;
          return PHX_MC_OK;
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
  bool protected_witness(int32_t s) const {
    if (input_.protected_facets == nullptr) return false;
    const PlcRecoveryWitness& witness = candidate_.witnesses[static_cast<std::size_t>(s)];
    if (witness.stratum == SourceStratum::kFacet) {
      const auto triangle = static_cast<std::size_t>(witness.entity);
      const auto polygon = static_cast<std::size_t>(candidate_.input_polygons[triangle]);
      return input_.protected_facets[input_.polygon_facets[polygon]] != 0;
    }
    if (witness.stratum == SourceStratum::kSegment) {
      const auto edge = static_cast<std::size_t>(witness.entity);
      for (std::size_t use = edge_triangle_offsets_[edge]; use < edge_triangle_offsets_[edge + 1];
           ++use) {
        const auto triangle = static_cast<std::size_t>(edge_triangle_rows_[use]);
        const auto polygon = static_cast<std::size_t>(candidate_.input_polygons[triangle]);
        if (input_.protected_facets[input_.polygon_facets[polygon]] != 0) return true;
      }
    }
    return false;
  }


  // Whether split point s is suppressed under either policy: it lies on a
  // hidden internal edge (never published), or on a facet with a declared
  // plane that does not admit its carrier (declared_plane_admits: a vertex
  // of a declared facet must round a point of the plane or lie within the
  // facet's tolerance of it).
  bool suppression_required(int32_t s) const {
    if (protected_witness(s)) return true;
    const PlcRecoveryWitness& witness = candidate_.witnesses[static_cast<std::size_t>(s)];
    std::vector<int32_t> facets;
    if (witness.stratum == SourceStratum::kSegment) {
      const auto edge = static_cast<std::size_t>(witness.entity);
      if (edge >= visible_edges_) return true;
      for (std::size_t use = edge_triangle_offsets_[edge]; use < edge_triangle_offsets_[edge + 1];
           ++use) {
        facets.push_back(input_.polygon_facets[candidate_.input_polygons[static_cast<std::size_t>(
            edge_triangle_rows_[use])]]);
      }
    } else if (witness.stratum == SourceStratum::kFacet) {
      facets.push_back(input_.polygon_facets[
          candidate_.input_polygons[static_cast<std::size_t>(witness.entity)]]);
    }
    if (plane_normals_.empty()) return false;
    return std::any_of(facets.begin(), facets.end(), [&](int32_t facet) {
      const double* row = input_.facet_planes + 9 * static_cast<std::size_t>(facet);
      if (std::isnan(row[0])) return false;
      const double tolerance = input_.facet_plane_tolerances == nullptr
          ? 0.0 : input_.facet_plane_tolerances[facet];
      double distance = 0.0;
      return !declared_plane_admits(row, plane_normals_[static_cast<std::size_t>(facet)],
                                    tolerance, point(s), distance, input_.affine_metric);
    });
  }

  // Points are tried newest first (undoing later splits before the earlier
  // ones they refined), in passes: a point none of whose replacements
  // applies is retried after the others have been removed. A pass that
  // removes nothing is repeated once with grown wedge cavities (see
  // remove_witnessed_vertex), a grown pass that removes nothing once as a
  // pinched pass (relocation by peeling), a pinched pass that removes
  // nothing once as a pockets pass (grown cavities again, their open
  // pockets coned apart), and a pockets pass that removes nothing refuses
  // its first remaining point. Every other pass removes at least one point,
  // so there are at most four times as many passes as points. The grown,
  // pinched and pockets passes run only where every pass before them
  // failed, so recoveries that never need them are unchanged.
  enum class SuppressionPass { kOrdinary, kGrown, kPinched, kPockets };
  int32_t suppress_witnessed_vertices() {
    const int32_t count = static_cast<int32_t>(candidate_.points.size() / 3);
    suppressed_.assign(static_cast<std::size_t>(count), 0);
    std::vector<int32_t> pending;
    for (int32_t vertex = count - 1; vertex >= source_point_count_; --vertex) {
      if (input_.policy != BoundaryPolicy::kFixed && exact_points_.position(vertex) == nullptr &&
          !suppression_required(vertex)) {
        continue;
      }
      pending.push_back(vertex);
    }
    SuppressionPass pass = SuppressionPass::kOrdinary;
    while (!pending.empty()) {
      std::vector<int32_t> remaining;
      for (int32_t vertex : pending) {
        bool removed = false;
        const int32_t status = suppress_vertex(vertex, pass, removed);
        if (status != PHX_MC_OK) return status;
        if (!removed) remaining.push_back(vertex);
      }
      if (remaining.size() == pending.size()) {
        if (pass == SuppressionPass::kPockets) {
          return refuse_suppression(remaining.front(), inconclusive_.contains(remaining.front()));
        }
        pass = pass == SuppressionPass::kOrdinary ? SuppressionPass::kGrown
               : pass == SuppressionPass::kGrown  ? SuppressionPass::kPinched
                                                  : SuppressionPass::kPockets;
        continue;
      }
      pass = SuppressionPass::kOrdinary;
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
      for (const Triple& key : faces->keys_through(s)) {
        const PlcSourceFace& source = faces->at(key);
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
            set_vertex_hint(vertex, id);
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

  // Visibility facts of the fills of one suppressed point, shared by its
  // targets, wedges, layers and options (wrap_cavity). Keys are vertex ids,
  // whose exact positions are fixed while the point is suppressed, so the
  // facts hold whatever the complex becomes. A positive cell whose interior
  // meets a closed wall (`blocked`, keyed by its sorted vertices) is blocked
  // in every cavity bounded by that wall. A lineage is the sequence of
  // cavities of one kind (the wedge, or the wedge merged with layers of
  // cells) of one target and wedge; each fill registers its walls as the
  // lineage's next generation, `since` holding the generation from which
  // each current wall has been present without a gap. A cell found visible
  // at generation g (`visible`) missed every wall present then, so later it
  // is tested only against the walls present since after g: a repeated
  // cavity costs no wall test, and a layer tests only its new walls against
  // the cells its previous layer saw. Verdicts are those of a full scan.
  struct SuppressionMemo {
    using CellKey = std::array<int32_t, 4>;
    struct Lineage {
      uint32_t generation = 0;
      std::unordered_map<Triple, uint32_t, IdTupleHash<3>> since;
      std::unordered_map<CellKey, uint32_t, IdTupleHash<4>> visible;
      std::unordered_map<CellKey, Triple, IdTupleHash<4>>* blocked = nullptr;
    };
    std::unordered_map<CellKey, Triple, IdTupleHash<4>> blocked;
    std::map<std::array<int64_t, 3>, Lineage> lineages;

    Lineage* lineage(int64_t target, int64_t wedge, int64_t kind) {
      Lineage& found = lineages[{target, wedge, kind}];
      found.blocked = &blocked;
      return &found;
    }
  };

  // What a failed attempt of a kGrown or kPockets pass reports about its
  // growth (remove_witnessed_vertex): whether the region of its first
  // unfilled wedge can still grow, and the obstruction at s that remained:
  // that wedge (its ordinal; the wedges before it were filled) and the
  // sides with a vertex among the wedge's own cavity vertices that the
  // partial constrained Delaunay wrap of its merged cavity left open
  // (option 6), sorted.
  struct SuppressionGrowth {
    bool extended = false;
    std::size_t wedge = 0;
    std::vector<FacetKey> obstruction;
  };

  // The declared budget of the exhaustive search (PlcInput) and what one
  // search used: the states the recovery may still expand, the bytes one
  // search may hold, the bytes the last search held (at most, at its end),
  // its deepest state (cells placed), and whether it stopped over budget.
  struct SearchAccount {
    int64_t nodes_left = 0;
    int64_t byte_limit = 0;
    int64_t bytes = 0;
    int64_t depth = 0;
    bool exceeded = false;
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
  // so the fill takes at most one step per cell. Only the first apex of a
  // side is used, the candidate whose sphere through the side holds no
  // other: a linear scan for the pencil minimum, confirmed by one more
  // linear pass, costs O(candidates) insphere tests instead of the
  // O(candidates^2) of a full ranking, and its visibility O(candidates *
  // walls) separation tests, fewer with `lineage` (SuppressionMemo).
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
  // this search only after its polynomial fills have failed, and bounds it
  // by the declared budget in `account` (required with `exhaustive`): over
  // it the search fails with `account->exceeded` set. `nodes` receives the
  // number of states expanded. `center` selects the relocation peel (see
  // below).
  bool wrap_cavity(const std::vector<SuppressionCell>& boundary, int32_t placeholder,
                   const std::vector<int32_t>& interior, bool exhaustive,
                   std::vector<SuppressionCell>& filled, std::size_t& nodes,
                   std::vector<SuppressionCell>* remainder = nullptr,
                   SuppressionMemo::Lineage* lineage = nullptr,
                   SearchAccount* account = nullptr,
                   const ExactPointRef* center = nullptr) const {
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
    // Closed boxes around the exact positions (carriers widened by their
    // rigorous deviation bounds, rounded outward): a wall whose box misses
    // the box of a cell is separated from it, so the exact test below runs
    // only for walls near the cell. This changes no verdict.
    struct Box {
      std::array<double, 3> lo;
      std::array<double, 3> hi;
    };
    const auto box_of = [&](const int32_t* ids, int count) {
      Box box{};
      box.lo.fill(std::numeric_limits<double>::infinity());
      box.hi.fill(-std::numeric_limits<double>::infinity());
      for (int k = 0; k < count; ++k) {
        const ExactPointRef at = ref(ids[k]);
        for (int axis = 0; axis < 3; ++axis) {
          const double deviation = at.deviation == nullptr ? 0.0 : at.deviation[axis];
          const double lo = std::nextafter(at.carrier[axis] - deviation,
                                           -std::numeric_limits<double>::infinity());
          const double hi = std::nextafter(at.carrier[axis] + deviation,
                                           std::numeric_limits<double>::infinity());
          box.lo[static_cast<std::size_t>(axis)] = std::min(box.lo[static_cast<std::size_t>(axis)], lo);
          box.hi[static_cast<std::size_t>(axis)] = std::max(box.hi[static_cast<std::size_t>(axis)], hi);
        }
      }
      return box;
    };
    const auto disjoint = [](const Box& first, const Box& second) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (first.hi[axis] < second.lo[axis] || second.hi[axis] < first.lo[axis]) return true;
      }
      return false;
    };
    std::vector<std::pair<std::size_t, Box>> wall_boxes;
    wall_boxes.reserve(walls.size());
    for (std::size_t row = 0; row < walls.size(); ++row) {
      wall_boxes.push_back({row, box_of(walls[row].data(), 3)});
    }
    // The distinct closed walls (a slit is one wall seen from both sides),
    // registered as the lineage's next generation, newest first.
    std::vector<std::pair<uint32_t, std::size_t>> recent;
    if (lineage != nullptr) {
      const uint32_t generation = ++lineage->generation;
      std::unordered_map<Triple, uint32_t, IdTupleHash<3>> since;
      since.reserve(walls.size());
      for (std::size_t row = 0; row < walls.size(); ++row) {
        const Triple key = sorted_triple(walls[row].data());
        const auto known = lineage->since.find(key);
        const uint32_t first = known == lineage->since.end() ? generation : known->second;
        if (since.emplace(key, first).second) recent.push_back({first, row});
      }
      lineage->since = std::move(since);
      std::stable_sort(recent.begin(), recent.end(),
                       [](const auto& a, const auto& b) { return a.first > b.first; });
    }
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
    // Whether the interior of the positive cell `tet` misses every wall.
    // With a lineage, a cell blocked by a current wall is rejected without a
    // test and a cell visible at generation g is tested only against the
    // walls present since after g (SuppressionMemo); the verdict is that of
    // the full scan, which runs without one.
    const auto clear = [&](const std::array<int32_t, 4>& tet) {
      const Box box = box_of(tet.data(), 4);
      const auto meets = [&](std::size_t row) {
        return !disjoint(box, wall_boxes[row].second) && !separated(tet, walls[row]);
      };
      if (lineage == nullptr) {
        return std::none_of(wall_boxes.begin(), wall_boxes.end(),
                            [&](const auto& entry) { return meets(entry.first); });
      }
      SuppressionMemo::CellKey key = tet;
      std::sort(key.begin(), key.end());
      auto& blocked = *lineage->blocked;
      const auto blocker = blocked.find(key);
      if (blocker != blocked.end() && lineage->since.contains(blocker->second)) return false;
      const auto known = lineage->visible.find(key);
      const uint32_t seen = known == lineage->visible.end() ? 0U : known->second;
      for (const auto& [first, row] : recent) {
        if (first <= seen) break;
        if (meets(row)) {
          blocked[key] = sorted_triple(walls[row].data());
          if (known != lineage->visible.end()) lineage->visible.erase(known);
          return false;
        }
      }
      lineage->visible[key] = lineage->generation;
      return true;
    };
    // Apices of an open side: the vertices on its inner side whose cell is
    // visible, in pencil order (number of rivals inside the sphere through
    // the side). The perturbed insphere test is antisymmetric and orders
    // the inner vertices totally, so the first apex is the only visible
    // vertex whose sphere holds no visible rival. Without `exhaustive` only
    // that one is kept: the pencil minimum of the inner vertices is taken
    // by a linear scan and dropped while its cell is not visible, and the
    // apex found is confirmed against every vertex not dropped (a superset
    // of the visible ones), O(inner vertices) insphere tests per vertex
    // dropped plus one visibility test each, instead of a visibility test
    // for every inner vertex and O(candidates^2) insphere tests. Both depend
    // only on the oriented side and are kept per fill.
    std::map<FacetKey, std::vector<int32_t>> apices;
    // Counted bytes of the ranked apex lists (see the exhaustive search).
    int64_t apex_bytes = 0;
    const auto apices_of = [&](const FacetKey& side, const Open& open) -> const std::vector<int32_t>& {
      const auto known = apices.find(side);
      if (known != apices.end()) return known->second;
      std::vector<int32_t> inner;
      for (int32_t vertex : vertices) {
        if (vertex == side.a || vertex == side.b || vertex == side.c) continue;
        if (positive(with(open.cell, open.slot, vertex))) inner.push_back(vertex);
      }
      // Whether `rival` lies inside the sphere through the side and `apex`.
      const auto holds = [&](int32_t apex, int32_t rival) {
        const std::array<int32_t, 4> tet = with(open.cell, open.slot, apex);
        // The standalone suppression-cavity fixture has no triangulation;
        // it uses the original Euclidean predicate on its exact witnesses.
        if (triangulation_ == nullptr) {
          return insphere_sos(ref(tet[0]), ref(tet[1]), ref(tet[2]), ref(tet[3]), ref(rival),
                              tet[0], tet[1], tet[2], tet[3], rival) > 0;
        }
        return triangulation_->delaunay_insphere(
            ref(tet[0]), ref(tet[1]), ref(tet[2]), ref(tet[3]), ref(rival),
            tet[0], tet[1], tet[2], tet[3], rival) > 0;
      };
      if (!exhaustive && center == nullptr) {
        std::vector<int32_t> remaining = inner;
        while (!remaining.empty()) {
          std::size_t best = 0;
          for (std::size_t row = 1; row < remaining.size(); ++row) {
            if (holds(remaining[best], remaining[row])) best = row;
          }
          const int32_t first = remaining[best];
          if (!clear(with(open.cell, open.slot, first))) {
            remaining[best] = remaining.back();
            remaining.pop_back();
            continue;
          }
          if (std::none_of(remaining.begin(), remaining.end(), [&](int32_t rival) {
                return rival != first && holds(first, rival);
              })) {
            return apices.emplace(side, std::vector<int32_t>{first}).first->second;
          }
          break;
        }
        if (remaining.empty()) return apices.emplace(side, std::vector<int32_t>{}).first->second;
      }
      std::vector<int32_t> candidates;
      for (int32_t vertex : inner) {
        if (clear(with(open.cell, open.slot, vertex))) candidates.push_back(vertex);
      }
      std::vector<std::pair<int, int32_t>> ranked;
      ranked.reserve(candidates.size());
      for (int32_t vertex : candidates) {
        int inside = 0;
        for (int32_t rival : candidates) {
          if (rival != vertex && holds(vertex, rival)) ++inside;
        }
        ranked.push_back({inside, vertex});
      }
      std::sort(ranked.begin(), ranked.end());
      std::vector<int32_t> order;
      order.reserve(ranked.size());
      for (const auto& entry : ranked) order.push_back(entry.second);
      apex_bytes += 64 + 4 * static_cast<int64_t>(order.size());
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
      // With `remainder`, a side without an applicable apex stays open and
      // the wrap goes on with the others. The verdict is unchanged (false
      // once a side was stuck); `filled` then holds the partial fill and
      // `remainder` the open sides left (oriented walls with the
      // placeholder bounding the unfilled part, empty when the stuck sides
      // were closed from their other cells). Past a stuck side the
      // tetrahedralization the steps follow does not exist, so a cell is
      // also refused (its side stuck) when its interior meets a face the
      // wrap opened: the partial fill bounds the cells placed so far by the
      // walls and those faces, so its cells stay interior-disjoint inside
      // the cavity, as the cells of any fill completing it must be (positive
      // cells pairing every face cover each point once). Each step closes a
      // side for good or marks one stuck, so this ends after at most two
      // steps per oriented triangle of cavity vertices.
      std::set<FacetKey> stuck;
      std::set<FacetKey> walls_of_front;
      if (remainder != nullptr) {
        for (const auto& [side, open] : front) {
          (void)open;
          walls_of_front.insert(side);
        }
      }
      const auto overlaps = [&](const std::array<int32_t, 4>& tet) {
        const Box box = box_of(tet.data(), 4);
        for (const auto& [side, open] : front) {
          (void)open;
          if (walls_of_front.contains(side)) continue;
          const Triple face{side.a, side.b, side.c};
          if (!disjoint(box, box_of(face.data(), 3)) && !separated(tet, face)) return true;
        }
        return false;
      };
      // With `center` (and `remainder`), the peel of a relocation: only the
      // sides that the point `center` does not see (the cone from it over
      // the side is not exactly positive) are closed, the least such side
      // first, each by the first apex in pencil order whose cell is
      // visible, plans (repeats no closed side, lies inside every wall) and
      // meets no face the peel opened, preferring the first that opens only
      // sides `center` sees. The verdict is whether every side left open is
      // seen from `center`; `filled` holds the peeled cells and `remainder`
      // the open sides, whose cones from `center` complete the fill. Each
      // step closes its side for good, so the peel takes at most one step
      // per oriented triangle of cavity vertices.
      if (center != nullptr) {
        const auto sees = [&](const Open& open) {
          ExactPointRef corners[4];
          for (int k = 0; k < 4; ++k) {
            corners[k] = k == open.slot ? *center : ref(open.cell[static_cast<std::size_t>(k)]);
          }
          return orient3d(corners[0], corners[1], corners[2], corners[3]) > 0;
        };
        for (;;) {
          auto next = front.begin();
          while (next != front.end() && sees(next->second)) ++next;
          if (next == front.end()) break;
          const FacetKey key = next->first;
          const Open open = next->second;
          Step chosen;
          bool found = false;
          for (int32_t apex : apices_of(key, open)) {
            Step step;
            if (!plan(open, apex, step) || overlaps(step.cell.v)) continue;
            const bool seen = std::all_of(step.opened.begin(), step.opened.end(),
                                          [&](const auto& entry) { return sees(entry.second); });
            if (!found || seen) chosen = std::move(step);
            found = true;
            if (seen) break;
          }
          if (!found) return false;
          apply(key, chosen);
          ++nodes;
        }
        remainder->clear();
        for (const auto& [side, open] : front) {
          (void)side;
          SuppressionCell wall{open.cell, {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint}};
          wall.constraints[static_cast<std::size_t>(open.slot)] = open.constraint;
          remainder->push_back(wall);
        }
        return true;
      }
      for (;;) {
        auto next = front.begin();
        while (next != front.end() && stuck.contains(next->first)) ++next;
        if (next == front.end()) break;
        const FacetKey key = next->first;
        const Open open = next->second;
        const std::vector<int32_t>& order = apices_of(key, open);
        Step step;
        if (order.empty() || !plan(open, order.front(), step) ||
            (remainder != nullptr && overlaps(step.cell.v))) {
          if (remainder == nullptr) return false;
          stuck.insert(key);
          continue;
        }
        apply(key, step);
        ++nodes;
      }
      if (stuck.empty()) return true;
      remainder->clear();
      for (const auto& [side, open] : front) {
        (void)side;
        SuppressionCell wall{open.cell, {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint}};
        wall.constraints[static_cast<std::size_t>(open.slot)] = open.constraint;
        remainder->push_back(wall);
      }
      return false;
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
    if (account == nullptr) return false;
    // Failed states, kept compactly and exactly. Each step closes the four
    // faces of its cell (as the cell sees them), none closed before, so a
    // state's closed sides are the faces of the cells placed on the way to
    // it, four per cell. An expanded state is stored as a node: the cell
    // its step placed and its parent node, so the faces of the cells on its
    // chain to the root are its closed set. Its fingerprint is the sum, in
    // two 64-bit lanes, of a mix of each closed side, updated with each
    // step. Failed nodes are indexed by fingerprint in an open-addressing
    // table, and a match is confirmed exactly: the node's chain has as many
    // cells as the current state and every face of them is closed now
    // (both sets have four sides per cell, so containment is equality). So
    // the memo answers as a set of full copies of the closed sets would,
    // and the search expands the same states in the same order, in 40
    // bytes per expanded state and 4 per table slot instead of 12 per
    // closed side and a set node per state.
    //
    // Counted bytes (fixed-width, the same on every platform): the node
    // chunks allocated, the table slots, and 64 per ranked side plus 4 per
    // apex of the apex lists. A state is expanded only while the recovery
    // may still expand one (`nodes_left`) and the count, with the chunk or
    // table it would allocate, stays within `byte_limit`; otherwise the
    // search stops with `exceeded`.
    struct Node {
      std::array<std::uint64_t, 2> print;
      std::array<int32_t, 4> cell;
      uint32_t parent;
      uint32_t unused;
    };
    static_assert(sizeof(Node) == 40);
    constexpr uint32_t kRoot = std::numeric_limits<uint32_t>::max();
    constexpr std::size_t kChunk = 4096;
    constexpr int64_t kChunkBytes = static_cast<int64_t>(kChunk * sizeof(Node));
    std::vector<std::unique_ptr<Node[]>> chunks;
    std::size_t node_count = 0;
    std::vector<uint32_t> table;  // node id + 1, 0 when empty
    std::size_t stored = 0;
    const auto node_at = [&](uint32_t id) -> Node& { return chunks[id / kChunk][id % kChunk]; };
    const auto counted = [&]() {
      return kChunkBytes * static_cast<int64_t>(chunks.size()) +
             4 * static_cast<int64_t>(table.size()) + apex_bytes;
    };
    const auto mix = [](std::uint64_t x) {
      x += 0x9E3779B97F4A7C15ULL;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      return x ^ (x >> 31);
    };
    std::array<std::uint64_t, 2> print{0, 0};
    const auto stamp = [&](const std::array<int32_t, 4>& cell, bool add) {
      constexpr std::uint64_t kSeeds[2] = {0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL};
      for (int slot = 0; slot < 4; ++slot) {
        const FacetKey side = side_of(cell, slot);
        for (std::size_t lane = 0; lane < 2; ++lane) {
          std::uint64_t h = mix(kSeeds[lane] ^ static_cast<uint32_t>(side.a));
          h = mix(h ^ static_cast<uint32_t>(side.b));
          h = mix(h ^ static_cast<uint32_t>(side.c));
          print[lane] = add ? print[lane] + h : print[lane] - h;
        }
      }
    };
    const auto same_state = [&](uint32_t id) {
      std::size_t cells = 0;
      for (uint32_t at = id; node_at(at).parent != kRoot; at = node_at(at).parent) {
        if (++cells > filled.size()) return false;
        const Node& node = node_at(at);
        for (int slot = 0; slot < 4; ++slot) {
          if (!closed.contains(side_of(node.cell, slot))) return false;
        }
      }
      return cells == filled.size();
    };
    const auto known_failed = [&]() {
      if (table.empty()) return false;
      const std::size_t mask = table.size() - 1;
      for (std::size_t at = static_cast<std::size_t>(print[0]) & mask;; at = (at + 1) & mask) {
        const uint32_t entry = table[at];
        if (entry == 0) return false;
        if (node_at(entry - 1).print == print && same_state(entry - 1)) return true;
      }
    };
    const auto place = [&](std::vector<uint32_t>& slots, uint32_t id) {
      const std::size_t mask = slots.size() - 1;
      std::size_t at = static_cast<std::size_t>(node_at(id).print[0]) & mask;
      while (slots[at] != 0) at = (at + 1) & mask;
      slots[at] = id + 1;
    };
    const auto over = [&]() {
      account->exceeded = true;
      return false;
    };
    const auto remember_failed = [&](uint32_t id) {
      if (2 * (stored + 1) > table.size()) {
        const std::size_t size = table.empty() ? 64 : 2 * table.size();
        if (counted() + 4 * static_cast<int64_t>(size - table.size()) > account->byte_limit) {
          return over();
        }
        std::vector<uint32_t> grown(size, 0U);
        for (uint32_t entry : table) {
          if (entry != 0) place(grown, entry - 1);
        }
        table = std::move(grown);
      }
      place(table, id);
      ++stored;
      return false;
    };
    std::function<bool(uint32_t)> search = [&](uint32_t parent) -> bool {
      if (front.empty()) return true;
      if (known_failed()) return false;
      if (account->nodes_left <= 0 || counted() > account->byte_limit) return over();
      if (node_count % kChunk == 0) {
        if (counted() + kChunkBytes > account->byte_limit ||
            node_count >= static_cast<std::size_t>(kRoot)) {
          return over();
        }
        chunks.push_back(std::make_unique<Node[]>(kChunk));
      }
      const auto id = static_cast<uint32_t>(node_count++);
      node_at(id) = Node{print,
                         parent == kRoot ? std::array<int32_t, 4>{} : filled.back().v,
                         parent, 0U};
      ++nodes;
      --account->nodes_left;
      account->depth = std::max(account->depth, static_cast<int64_t>(filled.size()));
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
        stamp(step.cell.v, true);
        if (search(id)) return true;
        stamp(step.cell.v, false);
        undo(key, open, step);
        if (account->exceeded) return false;
      }
      return remember_failed(id);
    };
    account->exceeded = false;
    account->depth = 0;
    const bool found = search(kRoot);
    account->bytes = counted();
    return found;
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

  // Relocation by peeling (option 8 of remove_witnessed_vertex): the
  // suppressed point s moves into the volume, to a binary64 point y of the
  // ray `origin` + l `direction` / `norm` (l = reach / 4, reach / 8, ...,
  // above 2^-40 reach; the ray of option 4, or one toward a star cell), and
  // the cavity is filled by the cones from y over the sides y sees, after
  // the sides it does not see were closed by the peel of wrap_cavity
  // (`center`) with cavity vertices.
  //
  // Why the cavity needs it. s lies on its new subfaces' planes and
  // strictly inside every link face, so the kernel of the cavity near s is
  // the wedge between its subface planes cut by the link half-spaces; cones
  // from one point (option 4) need a binary64 point in it. A sliver cell
  // (s, a, b, v) over a subface (s, a, b), with v almost in the facet plane
  // (such as a volume Steiner point that recovery placed by halving toward
  // that plane), has a link face abv whose plane passes closer to s than
  // the spacing of binary64 points near s (in LIN's torus-1 input, 6e-18
  // from s, a tenth of an ulp of its coordinates): the kernel is pinched
  // into the thin wedge between that plane and the facet plane, and the
  // new subface under it admits no apex but v, so neither option 4 nor a
  // Steiner-free fill need exist. When every cell over a half-fan has the
  // apex v, their union is the cone from v over the new subfaces (s lies
  // on the segment), and peeling closes the sliver's link face by a cavity
  // vertex (that union), leaving beyond it the star of s whose subface
  // plane is turned by the sliver's tiny angle: its kernel near s is again
  // the dihedral wedge of the facets, which holds binary64 points.
  //
  // Correctness. Every cell is exactly positive (peeled cells by
  // wrap_cavity, cones because y sees their sides exactly), and the cells
  // pair every face with the walls as their boundary, so as a 3-chain
  // their sum has the walls as boundary and every point off the faces is
  // covered by as many cells as the walls wind around it, 1 inside the
  // cavity and 0 outside: they tile the cavity. The walls are kept with
  // their marks, so every subface stays on its source facet with its
  // exact vertices; only the volume point y is new. The caller's cavity
  // edit checks the pairing again before committing. Termination: each
  // peel step closes one side for good, so a peel takes at most one step
  // per oriented triangle of cavity vertices, and the ray has at most 40
  // points.
  //
  // On success `filled` holds the peeled cells followed by the cones with
  // `apex` in place of the placeholder, `location` is y and `centered`
  // whether any cone uses it (none when the peel closed every side).
  bool relocate_by_peeling(const std::vector<SuppressionCell>& cavity, int32_t placeholder,
                           int32_t apex, const double* origin,
                           const std::array<double, 3>& direction, double norm, double reach,
                           SuppressionMemo::Lineage* lineage, std::vector<SuppressionCell>& filled,
                           std::array<double, 3>& location, bool& centered) {
    centered = false;
    if (!(norm > 0.0) || !std::isfinite(reach) || !(reach > 0.0)) return false;
    PlcSuppressionSearch& evidence = candidate_.suppression_search;
    for (double step = 0.25 * reach; step > 0x1p-40 * reach; step *= 0.5) {
      for (int axis = 0; axis < 3; ++axis) {
        location[static_cast<std::size_t>(axis)] =
            origin[axis] + step * direction[static_cast<std::size_t>(axis)] / norm;
      }
      if (!geometry::point_in_domain(location.data())) continue;
      const ExactPointRef center{location.data(), nullptr};
      std::vector<SuppressionCell> rest;
      std::size_t nodes = 0;
      const bool peeled = wrap_cavity(cavity, placeholder, {}, false, filled, nodes, &rest,
                                      lineage, nullptr, &center);
      ++evidence.wraps;
      evidence.wrap_cells += static_cast<int64_t>(nodes);
      evidence.max_wrap_cells = std::max(evidence.max_wrap_cells, static_cast<int64_t>(nodes));
      if (!peeled) continue;
      for (SuppressionCell& cone : rest) {
        std::replace(cone.v.begin(), cone.v.end(), placeholder, apex);
      }
      centered = !rest.empty();
      filled.insert(filled.end(), rest.begin(), rest.end());
      return true;
    }
    return false;
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
  // every wall not through it and carries every wall through it as a cone
  // face, the constrained Delaunay fill with centroid
  // points of the blocking cells, the cones from one binary64 point
  // y = s + l d (d the sum of the inner unit normals of its subface planes,
  // l halving) over the walls, and the exhaustive Steiner-free search of
  // wrap_cavity; the search (counted in PlcRecovery::suppression_search)
  // runs only when the polynomial options fail. The star alone can have no
  // such fill: on nearly flat offset faces a sliver cell (four nearly
  // coplanar vertices, s on one diagonal) lies over a subface, some open
  // side (a new subface or a link face) is seen by no cavity vertex, so no
  // Steiner-free fill exists, and the kernel of the cavity is pinched to
  // the plane of the sliver, so no binary64 cone point exists.
  //
  // With `layers` > 0 (passes after one that removed no point), a finite
  // wedge that none of these fill is merged with `layers` layers of the
  // cells around it (grow_wedge: only across unconstrained faces, never
  // enclosing a source or boundary vertex; volume Steiner points it
  // encloses leave with s) and the merged cavity is filled by the same
  // options, the exhaustive search only for the first layer (fill
  // existence is NP-complete; a larger cavity only gives the polynomial
  // options more room). Then the wedge and the merged cavity get the
  // partial constrained Delaunay wrap (every side it can close) with cones
  // from one binary64 point over the sides it leaves open. `growth` reports
  // a failed attempt whose region can still grow, with its progress measure
  // (SuppressionGrowth). `memo` holds the
  // visibility facts of the point's earlier fills (SuppressionMemo), one
  // lineage per target, wedge and kind of cavity. Merging removes no
  // boundary vertex but s (only volume Steiner points besides, which no
  // witness or constraint involves) and moves no subface, so the
  // termination argument of the pass is unchanged; ordinary passes are as
  // without merging. With `pinched` (passes after a grown pass that
  // removed no point; layers 0) the exhaustive search is skipped and a
  // finite wedge none of the polynomial options fill is relocated with
  // peeling (option 8 of fill_wedge). With `pockets` (passes after a
  // pinched pass that removed no point; layers > 0) the exhaustive search
  // is skipped and option 6 goes on to cone the pockets it left open
  // apart (option 9). Every finite proposed cell is exactly
  // positive on exact positions and the cavity edit checks that the
  // proposal's boundary is the link of the removed cells with its marks, so
  // the proposal tiles them; the removal commits only when the proposal also
  // carries the chain interval (p, q) of a segment point and every new
  // subface with its mark. Otherwise nothing changes.
  int32_t remove_witnessed_vertex(int32_t s, int32_t t, int32_t p, int32_t q, int layers,
                                  bool pinched, bool pockets, SuppressionMemo& memo,
                                  bool& removed, SuppressionGrowth& growth) {
    removed = false;
    growth = SuppressionGrowth{};
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
        for (const Triple& key : faces->keys_through(s)) {
          const PlcSourceFace& face = faces->at(key);
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
    // The search and peeling routes of the wedges filled so far, recorded
    // when the removal commits.
    std::vector<PlcSuppressionRoute> routes;
    const auto stratum = static_cast<int64_t>(candidate_.witnesses[static_cast<std::size_t>(s)].stratum);
    const int64_t entity = candidate_.witnesses[static_cast<std::size_t>(s)].entity;
    const auto route = [&](int64_t kind, int64_t states, int64_t bytes, int64_t depth) {
      routes.push_back({s, stratum, entity, kind, states, bytes, depth});
    };
    // Volume points appended for a replacement that is not committed die.
    const auto abandon = [&]() {
      for (int32_t vertex : added) alive_[static_cast<std::size_t>(vertex)] = 0;
      return PHX_MC_OK;
    };
    // Fills one finite cavity (walls with the placeholder) by the first
    // option that applies; appends the fill to the proposals and sets
    // `done`, else leaves them unchanged. `blocked` are the star rows whose
    // centroids may serve as Steiner apices; `search` admits the exhaustive
    // Steiner-free search; `size` is the cell count reported as its
    // evidence. With `partial`, only options 1 and 6 run (the fallback after
    // every cavity of an attempt failed options 1 to 5) and `open_sides`
    // receives the sides its partial wrap left open. `lineage` carries
    // the visibility facts of earlier fills of this cavity's lineage.
    std::vector<Proposal> open_sides;
    const auto fill_wedge = [&](std::vector<Proposal> cavity,
                                const std::vector<std::size_t>& blocked,
                                const std::vector<std::array<double, 3>>& normals, double reach,
                                bool search, bool partial_only, std::size_t size,
                                SuppressionMemo::Lineage* lineage, bool& done) -> int32_t {
      done = false;
      // 1. The constrained Delaunay fill. Only for 6 (`partial_only`) does a
      // stuck wrap go on past its stuck sides, keeping the partial fill and
      // the open remainder; elsewhere it stops at its first stuck side.
      std::vector<Proposal> wrapped;
      std::vector<Proposal> remainder;
      std::size_t nodes = 0;
      const auto count_wrap = [&]() {
        PlcSuppressionSearch& evidence = candidate_.suppression_search;
        ++evidence.wraps;
        evidence.wrap_cells += static_cast<int64_t>(nodes);
        evidence.max_wrap_cells = std::max(evidence.max_wrap_cells, static_cast<int64_t>(nodes));
      };
      const bool wrap_ok = wrap_cavity(cavity, kPlaceholder, {}, false, wrapped, nodes,
                                       partial_only ? &remainder : nullptr, lineage);
      count_wrap();
      if (wrap_ok) {
        proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
        done = true;
        return PHX_MC_OK;
      }
      const std::vector<Proposal> partial = wrapped;
      // A binary64 point y = s + l d (l halving) with every cone over
      // `walls` exactly positive, and the cones, if one exists.
      double direction[3] = {0.0, 0.0, 0.0};
      for (const auto& unit : normals) {
        for (int axis = 0; axis < 3; ++axis) direction[axis] += unit[static_cast<std::size_t>(axis)];
      }
      const double norm = std::sqrt(geometry::dot(direction, direction));
      std::array<double, 3> index_direction{direction[0], direction[1], direction[2]};
      if (input_.affine_metric != nullptr) {
        TetMesh metric;
        if (metric.set_affine_metric({input_.affine_metric, 9}) != PHX_MC_OK) {
          return PHX_MC_INVALID_ARGUMENT;
        }
        index_direction = metric.metric_inverse_transform(index_direction);
      }
      const auto cone_from_point = [&](std::vector<Proposal> walls) {
        bool found = false;
        double location[3];
        if (!normals.empty() && norm > 0.0 && std::isfinite(reach) && reach > 0.0) {
          for (double step = 0.25 * reach; step > 0x1p-40 * reach && !found; step *= 0.5) {
            for (int axis = 0; axis < 3; ++axis) {
              location[axis] = point(s)[axis] + step * index_direction[axis] / norm;
            }
            if (!geometry::point_in_domain(location)) continue;
            const ExactPointRef center{location, nullptr};
            found = std::all_of(walls.begin(), walls.end(), [&](const Proposal& cone) {
              return positive(cone.v, &center);
            });
          }
        }
        if (!found) return false;
        // Placeholder ids of later centers stay distinct until appended.
        for (Proposal& cone : walls) {
          std::replace(cone.v.begin(), cone.v.end(), kPlaceholder,
                       kPlaceholder + 1 + static_cast<int32_t>(centers.size()));
        }
        centers.push_back({location[0], location[1], location[2]});
        proposals.insert(proposals.end(), walls.begin(), walls.end());
        return true;
      };
      const auto slot_of = [](const std::array<int32_t, 4>& v, int32_t vertex) {
        return static_cast<int>(std::find(v.begin(), v.end(), vertex) - v.begin());
      };
      // Cones from the cavity vertex `apex` over every wall of `walls` not
      // through it (option 2), into `cones`, when every cone is positive and
      // every wall through apex is a face of a cone with its orientation. A
      // cone face through apex that is a wall carries that wall's mark.
      const auto cones_from_vertex = [&](const std::vector<Proposal>& walls, int32_t apex,
                                         std::vector<Proposal>& cones) {
        cones.clear();
        std::map<Triple, int32_t> through;
        std::vector<FacetKey> sides;
        std::vector<FacetKey> faces;
        for (const Proposal& wall : walls) {
          if (std::find(wall.v.begin(), wall.v.end(), apex) == wall.v.end()) continue;
          const int slot = slot_of(wall.v, kPlaceholder);
          Triple face{};
          oriented_facet(wall.v.data(), slot, face.data());
          through.emplace(sorted_triple(face.data()),
                          wall.constraints[static_cast<std::size_t>(slot)]);
          sides.push_back(FacetKey::of(face.data()));
        }
        for (const Proposal& wall : walls) {
          if (std::find(wall.v.begin(), wall.v.end(), apex) != wall.v.end()) continue;
          Proposal cone = wall;
          std::replace(cone.v.begin(), cone.v.end(), kPlaceholder, apex);
          if (!positive(cone.v, nullptr)) return false;
          const int own = slot_of(cone.v, apex);
          for (int slot = 0; slot < 4; ++slot) {
            if (slot == own) continue;
            Triple face{};
            oriented_facet(cone.v.data(), slot, face.data());
            faces.push_back(FacetKey::of(face.data()));
            const auto known = through.find(sorted_triple(face.data()));
            if (known != through.end()) cone.constraints[static_cast<std::size_t>(slot)] = known->second;
          }
          cones.push_back(cone);
        }
        std::sort(faces.begin(), faces.end());
        return std::all_of(sides.begin(), sides.end(), [&](const FacetKey& side) {
          return std::binary_search(faces.begin(), faces.end(), side);
        });
      };
      // 9. (`pockets`: passes after a pinched pass that removed no point)
      // The sides 6 left open, split into pockets (sides joined across
      // shared edges), each filled on its own: cones from one of its
      // vertices (cones_from_vertex), else cones from the point y of 4,
      // else the peel of wrap_cavity from the binary64 centroid of its
      // vertices (the sides the centroid does not see closed by pocket
      // vertices, the rest coned from it, as in relocate_by_peeling). A
      // pocket's sides bound it as a closed surface, so positive cells
      // pairing every face tile it, and the partial fill with the pockets'
      // fills tiles the cavity. A grown cavity's partial wrap can close
      // every side around s and still leave pockets far from it, which y
      // does not see.
      const auto cone_pockets = [&](const std::vector<Proposal>& open) {
        std::vector<std::size_t> parent(open.size());
        std::iota(parent.begin(), parent.end(), 0);
        const auto find = [&](std::size_t row) {
          while (parent[row] != row) row = parent[row] = parent[parent[row]];
          return row;
        };
        std::map<std::pair<int32_t, int32_t>, std::size_t> edges;
        for (std::size_t row = 0; row < open.size(); ++row) {
          Triple face{};
          oriented_facet(open[row].v.data(), slot_of(open[row].v, kPlaceholder), face.data());
          for (int k = 0; k < 3; ++k) {
            const int32_t a = face[static_cast<std::size_t>(k)];
            const int32_t b = face[static_cast<std::size_t>((k + 1) % 3)];
            const auto [known, fresh] = edges.emplace(std::minmax(a, b), row);
            if (!fresh) parent[find(row)] = find(known->second);
          }
        }
        std::map<std::size_t, std::vector<Proposal>> pockets;
        for (std::size_t row = 0; row < open.size(); ++row) pockets[find(row)].push_back(open[row]);
        for (const auto& [unused, walls] : pockets) {
          (void)unused;
          std::set<int32_t> vertices;
          for (const Proposal& wall : walls) {
            for (int32_t vertex : wall.v) {
              if (vertex != kPlaceholder) vertices.insert(vertex);
            }
          }
          bool coned = false;
          for (int32_t apex : vertices) {
            std::vector<Proposal> cones;
            if (cones_from_vertex(walls, apex, cones)) {
              proposals.insert(proposals.end(), cones.begin(), cones.end());
              coned = true;
              break;
            }
          }
          if (!coned) coned = cone_from_point(walls);
          if (!coned) {
            double location[3] = {0.0, 0.0, 0.0};
            for (int32_t vertex : vertices) {
              for (int axis = 0; axis < 3; ++axis) location[axis] += point(vertex)[axis];
            }
            for (double& coordinate : location) coordinate /= static_cast<double>(vertices.size());
            if (geometry::point_in_domain(location)) {
              // The peel of wrap_cavity (relocate_by_peeling) from the
              // centroid: the sides it does not see are closed by pocket
              // vertices first, the rest are coned from it.
              const ExactPointRef center{location, nullptr};
              std::vector<Proposal> peeled;
              std::vector<Proposal> rest;
              std::size_t peel_nodes = 0;
              coned = wrap_cavity(walls, kPlaceholder, {}, false, peeled, peel_nodes, &rest,
                                  nullptr, nullptr, &center);
              PlcSuppressionSearch& evidence = candidate_.suppression_search;
              ++evidence.wraps;
              evidence.wrap_cells += static_cast<int64_t>(peel_nodes);
              evidence.max_wrap_cells =
                  std::max(evidence.max_wrap_cells, static_cast<int64_t>(peel_nodes));
              if (coned) {
                if (!rest.empty()) {
                  for (Proposal& cone : rest) {
                    std::replace(cone.v.begin(), cone.v.end(), kPlaceholder,
                                 kPlaceholder + 1 + static_cast<int32_t>(centers.size()));
                  }
                  centers.push_back({location[0], location[1], location[2]});
                }
                proposals.insert(proposals.end(), peeled.begin(), peeled.end());
                proposals.insert(proposals.end(), rest.begin(), rest.end());
              }
            }
          }
          if (!coned) return false;
        }
        return true;
      };
      // 6. (`partial_only`) The partial constrained Delaunay wrap of 1
      // (every side it could close) and cones from one binary64 point over
      // the sides it left open: positive cells pairing every face, with the
      // cavity walls as boundary, tile the cavity, so the remainder need
      // not be star-shaped as a whole, only seen from y side by side; with
      // `pockets`, then 9.
      if (partial_only) {
        open_sides = remainder;
        if (remainder.empty()) {
          if (partial.empty()) return PHX_MC_OK;
          proposals.insert(proposals.end(), partial.begin(), partial.end());
          done = true;
          return PHX_MC_OK;
        }
        const std::size_t before = proposals.size();
        const std::size_t placed = centers.size();
        proposals.insert(proposals.end(), partial.begin(), partial.end());
        done = cone_from_point(remainder);
        if (!done && pockets) done = cone_pockets(remainder);
        if (!done) {
          proposals.resize(before);
          centers.resize(placed);
        }
        return PHX_MC_OK;
      }
      // 2. Cones from one cavity vertex v over every wall not through v: a
      // tiling when the cavity is strictly star-shaped from v. As chains,
      // the cones' boundary is the sum of the walls, in which the two sides
      // of a sheet inside the cavity (a new subface the wedge meets from
      // both sides) cancel; so every wall through v must also be a face of
      // a cone, with its orientation, or the cones run across that subface
      // and it and its segment intervals are lost. On a cavity whose walls
      // bound it as a closed surface this holds whenever every cone is
      // positive (cones_from_vertex).
      std::set<int32_t> on_walls;
      for (const Proposal& wall : cavity) {
        for (int32_t vertex : wall.v) {
          if (vertex != kPlaceholder) on_walls.insert(vertex);
        }
      }
      for (int32_t apex : on_walls) {
        std::vector<Proposal> cones;
        if (!cones_from_vertex(cavity, apex, cones)) continue;
        proposals.insert(proposals.end(), cones.begin(), cones.end());
        done = true;
        return PHX_MC_OK;
      }
      // 3. Steiner-assisted fill: interior points of the cells that block
      // the contraction (every cell without a target) join the apices.
      std::vector<int32_t> extra;
      for (std::size_t row : blocked) {
        if (extra.size() == 8) break;
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
      nodes = 0;
      const bool extra_ok =
          !extra.empty() &&
          wrap_cavity(cavity, kPlaceholder, extra, false, wrapped, nodes, nullptr, lineage);
      if (!extra.empty()) count_wrap();
      if (extra_ok) {
        proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
        done = true;
        return PHX_MC_OK;
      }
      // 4. Cones from one binary64 point over the walls.
      if (cone_from_point(cavity)) {
        done = true;
        return PHX_MC_OK;
      }
      // 5. The exact exhaustive Steiner-free search (wrap_cavity), reached
      // only after every polynomial option has failed, and only for the
      // wedge itself and its first merged layer: Steiner-free fill existence
      // is NP-complete, while a larger merged cavity gives the polynomial
      // fills more room. A search over the declared budget is inconclusive:
      // the wedge stays unfilled here and the point's first such search is
      // kept as the evidence of a later peeling (or of its refusal).
      if (search) {
        nodes = 0;
        PlcSuppressionSearch& evidence = candidate_.suppression_search;
        SearchAccount account{evidence.node_budget - evidence.nodes, evidence.byte_budget};
        const bool searched = wrap_cavity(cavity, kPlaceholder, {}, true, wrapped, nodes,
                                          nullptr, lineage, &account);
        ++evidence.searches;
        evidence.nodes += static_cast<int64_t>(nodes);
        evidence.max_nodes = std::max(evidence.max_nodes, static_cast<int64_t>(nodes));
        evidence.max_wedge_cells = std::max(evidence.max_wedge_cells, static_cast<int64_t>(size));
        evidence.max_bytes = std::max(evidence.max_bytes, account.bytes);
        evidence.max_depth = std::max(evidence.max_depth, account.depth);
        if (account.exceeded) {
          ++evidence.inconclusive;
          if (evidence.exceeded == 0) {
            evidence.exceeded = 1;
            evidence.exceeded_nodes = static_cast<int64_t>(nodes);
            evidence.exceeded_bytes = account.bytes;
            evidence.exceeded_depth = account.depth;
          }
          inconclusive_.emplace(s, std::array<int64_t, 3>{static_cast<int64_t>(nodes),
                                                          account.bytes, account.depth});
        } else if (searched) {
          ++evidence.fills;
          route(kPlcRouteSearch, static_cast<int64_t>(nodes), account.bytes, account.depth);
          proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
          done = true;
          return PHX_MC_OK;
        }
      }
      // 8. (`pinched` passes, after every option above has failed in the
      // passes before) Relocation of s with peeling (relocate_by_peeling)
      // along the ray of 4, then along the rays from s toward the centroids
      // of the blocked star cells (the cells of 3), in their order. In a
      // thin wedge (a crease of a few degrees between its subface planes)
      // the points of the ray of 4 within reach can all leave a side that
      // neither they see nor a cavity vertex closes, while a point toward
      // the inside of one of its cells peels (R002: lobed-sphere-1 with 64
      // ulps, a 9.5 degree crease). Each ray has at most 40 points and
      // there are at most as many rays as star cells, so this ends; every
      // relocation is checked exactly as before.
      if (pinched && !normals.empty()) {
        const int32_t apex = kPlaceholder + 1 + static_cast<int32_t>(centers.size());
        std::array<double, 3> location{};
        bool centered = false;
        bool relocated = relocate_by_peeling(cavity, kPlaceholder, apex, point(s),
                                             index_direction, norm, reach, lineage, wrapped,
                                             location, centered);
        if (!relocated) {
          for (std::size_t row : blocked) {
            const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cells[row])];
            std::array<double, 3> toward{0.0, 0.0, 0.0};
            for (int32_t vertex : tet.v) {
              for (int axis = 0; axis < 3; ++axis) {
                toward[static_cast<std::size_t>(axis)] +=
                    0.25 * (point(vertex)[axis] - point(s)[axis]);
              }
            }
            const double length = std::sqrt(geometry::dot(toward.data(), toward.data()));
            if (!(length > 0.0)) continue;
            relocated = relocate_by_peeling(cavity, kPlaceholder, apex, point(s), toward,
                                            length, reach, lineage, wrapped, location, centered);
            if (relocated) break;
          }
        }
        if (relocated) {
          if (centered) centers.push_back(location);
          proposals.insert(proposals.end(), wrapped.begin(), wrapped.end());
          const auto first = inconclusive_.find(s);
          if (first == inconclusive_.end()) {
            route(kPlcRoutePeel, 0, 0, 0);
          } else {
            route(kPlcRoutePeelAfterInconclusive, first->second[0], first->second[1],
                  first->second[2]);
          }
          done = true;
        }
      }
      return PHX_MC_OK;
    };
    // Cells outside the star absorbed into a wedge's grown cavity.
    std::vector<int32_t> absorbed;
    std::unordered_set<int32_t> absorbed_set;
    const auto wall_of = [&](int32_t cell, int slot) {
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
      Proposal wall{{tet.v[0], tet.v[1], tet.v[2], tet.v[3]},
                    {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint}};
      wall.v[static_cast<std::size_t>(slot)] = kPlaceholder;
      wall.constraints[static_cast<std::size_t>(slot)] = complex.constraint(cell, slot);
      return wall;
    };
    // Cells around a vertex, for the enclosure test of merging.
    std::unordered_map<int32_t, std::vector<int32_t>> around;
    // The cavity of a wedge merged with `layers` layers of finite cells, each
    // layer the admissible cells across unconstrained faces of the previous
    // one (never across a subface, so the cavity stays on the same side of
    // every constraint), excluding the star and the cells merged by other
    // wedges. A cell is admissible unless it shares a subface with the
    // cavity or every cell around one of its source or boundary vertices
    // would be removed: that vertex would be enclosed, and the fill keeps
    // every such vertex but s. A volume Steiner point (placed by a planar
    // fan flip of recovery or by an earlier suppression; no witness or
    // constraint involves it) may be enclosed: the fill, built from the
    // walls, leaves it out and it is removed with s. Such points sit as
    // close as the halvings that placed them allowed to the subfaces and
    // segments of their own fans, which s may share, and a point that
    // stays a wall vertex of every merged cavity can pin a wedge with no
    // fill at any layer.
    // Admission only shrinks as the cavity grows, so a rejected cell stays
    // rejected. The walls are the new subfaces and the faces of the cavity
    // not shared by two of its cells. `extended` tells whether layer
    // `layers` admitted a cell; when it did not, the constraint-delimited
    // region reachable from the wedge is exhausted.
    const auto grow_wedge = [&](const std::vector<std::size_t>& rows, int layers,
                                std::vector<int32_t>& grown, std::vector<Proposal>& walls,
                                bool& extended) -> int32_t {
      grown.clear();
      walls.clear();
      extended = false;
      const std::size_t wedge = root(rows.front());
      std::unordered_set<int32_t> inside;
      const auto in_cavity = [&](int32_t cell) {
        if (inside.contains(cell)) return true;
        const auto row = row_of.find(cell);
        return row != row_of.end() && root(row->second) == wedge;
      };
      const auto removed_cell = [&](int32_t cell) {
        return row_of.contains(cell) || absorbed_set.contains(cell) || inside.contains(cell);
      };
      std::vector<int32_t> frontier;
      for (std::size_t row : rows) frontier.push_back(cells[row]);
      for (int layer = 0; layer < layers && !frontier.empty(); ++layer) {
        std::vector<int32_t> next;
        for (int32_t cell : frontier) {
          const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
          const bool star = row_of.contains(cell);
          for (int slot = 0; slot < 4; ++slot) {
            if (star && tet.v[slot] != s) continue;
            const int32_t neighbor = tet.n[slot];
            if (neighbor < 0 || complex.constraint(cell, slot) != kNoConstraint) continue;
            const Tetrahedron& candidate = complex.tets[static_cast<std::size_t>(neighbor)];
            if (removed_cell(neighbor) || is_ghost(candidate)) continue;
            bool admissible = true;
            for (int k = 0; k < 4 && admissible; ++k) {
              admissible = !in_cavity(candidate.n[k]) ||
                  complex.constraint(neighbor, k) == kNoConstraint;
            }
            for (int k = 0; k < 4 && admissible; ++k) {
              const int32_t vertex = candidate.v[k];
              if (volume_vertex(vertex)) continue;
              auto found = around.find(vertex);
              if (found == around.end()) {
                std::vector<int32_t> incident;
                const int32_t status = vertex_cells(vertex, incident);
                if (status != PHX_MC_OK) return status;
                found = around.emplace(vertex, std::move(incident)).first;
              }
              admissible = std::any_of(found->second.begin(), found->second.end(),
                                       [&](int32_t other) {
                                         return other != neighbor && !removed_cell(other);
                                       });
            }
            if (!admissible) continue;
            inside.insert(neighbor);
            grown.push_back(neighbor);
            next.push_back(neighbor);
          }
        }
        extended = layer + 1 == layers && !next.empty();
        frontier = std::move(next);
      }
      if (grown.empty()) return PHX_MC_OK;
      walls = subface_cells(rows);
      for (std::size_t row : rows) {
        const int32_t cell = cells[row];
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
        const int own = vertex_slot(tet, s);
        if (!inside.contains(tet.n[own])) walls.push_back(wall_of(cell, own));
      }
      for (int32_t cell : grown) {
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
        for (int slot = 0; slot < 4; ++slot) {
          if (!in_cavity(tet.n[slot])) walls.push_back(wall_of(cell, slot));
        }
      }
      return PHX_MC_OK;
    };
    int64_t ordinal = -1;
    for (const auto& [unused, rows] : wedges) {
      (void)unused;
      ++ordinal;
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
            double distance = geometry::squared_distance(point(s), point(vertex));
            if (input_.affine_metric != nullptr) {
              distance = 0.0;
              for (int axis = 0; axis < 3; ++axis) {
                double component = 0.0;
                for (int col = 0; col < 3; ++col) {
                  component += input_.affine_metric[3 * axis + col] *
                      (point(vertex)[col] - point(s)[col]);
                }
                distance += component * component;
              }
            }
            reach = std::min(reach, std::sqrt(distance));
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
        if (input_.affine_metric != nullptr) {
          double world_u[3] = {}, world_w[3] = {}, world_offset[3] = {};
          for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
              world_u[row] += input_.affine_metric[3 * row + col] * u[col];
              world_w[row] += input_.affine_metric[3 * row + col] * w[col];
              world_offset[row] += input_.affine_metric[3 * row + col] * offset[col];
            }
          }
          geometry::cross(world_u, world_w, normal);
          std::copy_n(world_offset, 3, offset);
        }
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
      bool done = false;
      SuppressionMemo::Lineage* own = memo.lineage(t, ordinal, 0);
      int32_t status = fill_wedge(cavity, blocked, normals, reach,
                                  layers <= 1 && !pinched && !pockets, false, rows.size(), own,
                                  done);
      if (status != PHX_MC_OK) return status;
      // 7. With layers > 0, the wedge merged with that many layers of cells
      // around it (grow_wedge), filled by options 1 to 5 (the exhaustive
      // search only for the first layer); then option 6 on the wedge and on
      // the merged cavity. Option 6 runs only in these passes, so the fills
      // of the ordinary passes are unchanged.
      std::vector<int32_t> grown;
      bool merged = false;
      if (!done && layers > 0) {
        std::vector<Proposal> walls;
        bool reached = false;
        status = grow_wedge(rows, layers, grown, walls, reached);
        if (status != PHX_MC_OK) return status;
        const std::size_t size = rows.size() + grown.size();
        SuppressionMemo::Lineage* wider = memo.lineage(t, ordinal, 1);
        if (!grown.empty()) {
          status = fill_wedge(walls, blocked, normals, reach, layers == 1 && !pockets, false, size,
                              wider, merged);
          if (status != PHX_MC_OK) return status;
        }
        if (!merged) {
          status = fill_wedge(cavity, blocked, normals, reach, false, true, rows.size(), own, done);
          if (status != PHX_MC_OK) return status;
        }
        if (!done && !merged && !grown.empty()) {
          status = fill_wedge(walls, blocked, normals, reach, false, true, size, wider, merged);
          if (status != PHX_MC_OK) return status;
        }
        done = done || merged;
        PlcSuppressionSearch& evidence = candidate_.suppression_search;
        if (merged) {
          ++evidence.merges;
          evidence.max_merge_layers = std::max(evidence.max_merge_layers,
                                               static_cast<int64_t>(layers));
        } else if (!done && !reached) {
          ++evidence.exhausted;
          evidence.max_exhausted_cells =
              std::max(evidence.max_exhausted_cells, static_cast<int64_t>(size));
        }
        // Unfilled with its region still growing: the merged cavity's
        // partial wrap was the last fill tried, so `open_sides` are its;
        // the obstruction is those with a vertex among the wedge's own
        // cavity vertices.
        if (!done && reached) {
          std::unordered_set<int32_t> own_vertices;
          for (const Proposal& wall : cavity) {
            for (int32_t vertex : wall.v) {
              if (vertex != kPlaceholder) own_vertices.insert(vertex);
            }
          }
          growth.extended = true;
          growth.wedge = static_cast<std::size_t>(ordinal);
          for (const Proposal& side : open_sides) {
            const int slot = static_cast<int>(
                std::find(side.v.begin(), side.v.end(), kPlaceholder) - side.v.begin());
            Triple face{};
            oriented_facet(side.v.data(), slot, face.data());
            if (std::any_of(face.begin(), face.end(),
                            [&](int32_t vertex) { return own_vertices.contains(vertex); })) {
              growth.obstruction.push_back(FacetKey::of(face.data()));
            }
          }
          std::sort(growth.obstruction.begin(), growth.obstruction.end());
        }
      }
      if (!done) return abandon();
      if (merged) {
        for (int32_t cell : grown) {
          absorbed.push_back(cell);
          absorbed_set.insert(cell);
        }
      }
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
    // The cavity edit checks the proposal against the link of the removed
    // cells, around which the subfaces and segment interval through s are
    // interior. So the removal also requires, whatever filled each wedge,
    // the chain interval (p, q) that replaces the two through a segment
    // point as an edge of the proposal (the segment passes through s, inside
    // the removed cells) and every new subface as a face with its mark
    // (CavityEdit::preserve). A proposal without them is not committed.
    if (p >= 0 && q >= 0 &&
        std::none_of(proposals.begin(), proposals.end(), [&](const Proposal& cell) {
          return std::find(cell.v.begin(), cell.v.end(), p) != cell.v.end() &&
                 std::find(cell.v.begin(), cell.v.end(), q) != cell.v.end();
        })) {
      return abandon();
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
    // Volume points of merged cells that no proposed cell uses are enclosed
    // (grow_wedge); they leave the complex with s. Every wall vertex is a
    // vertex of the proposal, which tiles the cavity with the walls as its
    // boundary, so these are exactly the interior vertices of the cavity.
    std::vector<int32_t> enclosed;
    if (!absorbed.empty()) {
      std::unordered_set<int32_t> seen;
      for (const Proposal& proposal : proposals) seen.insert(proposal.v.begin(), proposal.v.end());
      for (int32_t cell : absorbed) {
        for (int32_t vertex : complex.tets[static_cast<std::size_t>(cell)].v) {
          if (volume_vertex(vertex) && seen.insert(vertex).second) enclosed.push_back(vertex);
        }
      }
    }
    CavityEdit edit(complex);
    edit.begin();
    std::vector<int32_t> removed_cells = cells;
    removed_cells.insert(removed_cells.end(), absorbed.begin(), absorbed.end());
    for (int32_t cell : removed_cells) {
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
    for (const std::vector<NewFace>& group : groups) {
      for (const NewFace& face : group) {
        edit.preserve(face.oriented[0], face.oriented[1], face.oriented[2], face.facet);
      }
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
      for (int32_t vertex : tet.v) set_vertex_hint(vertex, id);
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
    set_vertex_hint(s, -1);
    suppressed_[static_cast<std::size_t>(s)] = 1;
    for (int32_t vertex : enclosed) {
      alive_[static_cast<std::size_t>(vertex)] = 0;
      set_vertex_hint(vertex, -1);
    }
    for (int32_t vertex : added) {
      const bool used = std::any_of(proposals.begin(), proposals.end(), [&](const Proposal& cell) {
        return std::find(cell.v.begin(), cell.v.end(), vertex) != cell.v.end();
      });
      if (!used) alive_[static_cast<std::size_t>(vertex)] = 0;
    }
    PlcSuppressionSearch& evidence = candidate_.suppression_search;
    const auto took = [&](int64_t kind) {
      return std::any_of(routes.begin(), routes.end(),
                         [&](const PlcSuppressionRoute& entry) { return entry.route == kind; });
    };
    if (took(kPlcRoutePeel) || took(kPlcRoutePeelAfterInconclusive)) ++evidence.peeled;
    if (took(kPlcRoutePeelAfterInconclusive)) ++evidence.peeled_after_inconclusive;
    candidate_.suppression_routes.insert(candidate_.suppression_routes.end(), routes.begin(),
                                         routes.end());
    removed = true;
    return PHX_MC_OK;
  }

  // A point that cannot be removed is a STEINER_SUPPRESSION refusal naming
  // its source segment or facet; under FIXED it is the FIXED_SEGMENT /
  // FIXED_FACET refusal of that entity (its boundary could not be recovered
  // unmodified). A point on a hidden internal edge names the edge's facet.
  // A point with an inconclusive (`over_budget`) exhaustive search is
  // always the STEINER_SUPPRESSION refusal (UNRESOLVED): no fill was ruled
  // out.
  int32_t refuse_suppression(int32_t s, bool over_budget) {
    const PlcRecoveryWitness& witness = candidate_.witnesses[static_cast<std::size_t>(s)];
    const bool fixed =
        !over_budget && (input_.policy == BoundaryPolicy::kFixed || protected_witness(s));
    const int32_t status = fixed ? PHX_MC_CONSTRAINT_INTERSECTION : PHX_MC_UNRESOLVED;
    int32_t triangle = witness.entity;
    if (witness.stratum == SourceStratum::kSegment) {
      const auto edge = static_cast<std::size_t>(witness.entity);
      if (edge < visible_edges_) {
        return fail(status, fixed ? kPlcFixedSegment : kPlcSteinerSuppression, kPlcEdge,
                    static_cast<int64_t>(witness.entity), kPlcPoint, s);
      }
      triangle = static_cast<int32_t>(edge_triangle_rows_[edge_triangle_offsets_[edge]]);
    }
    const int32_t facet = input_.polygon_facets[
        candidate_.input_polygons[static_cast<std::size_t>(triangle)]];
    return fail(status, fixed ? kPlcFixedFacet : kPlcSteinerSuppression, kPlcFacet, facet,
                kPlcPoint, s);
  }

  // Suppresses one witnessed vertex if it can: contraction onto a chain
  // neighbour (a segment point) or a disk neighbour (a facet point), else
  // the ear-clipped half-fans. In a kGrown pass (and a kPockets pass, with
  // option 9 and without the exhaustive search), the targets are retried
  // with wedge cavities merged with 1, 2, ... layers of cells. A target
  // leaves the rotation once its unfillable wedge has exhausted its region
  // (layer `layers` admitted no cell), or once a layer repeats an
  // obstruction of an earlier layer: a failed attempt reports its first
  // unfilled wedge and the sides with a vertex among that wedge's own
  // cavity vertices that the partial constrained Delaunay wrap of its
  // merged cavity left open (SuppressionGrowth), and growth stops at the
  // first such state the target has already met. Growth serves only to
  // let that wrap close the sides around s; an obstruction around s that
  // recurs is one growth has stopped changing, and the later passes take
  // over. (R002: torus-2 with one ulp grew one wedge for hours to tens of
  // thousands of cells, its open sides around s the same 13 from the
  // fourteenth layer on, the fill failing on sides far from s; the merges
  // that succeed, after up to seven layers, never met a state twice.) The
  // states are pairs of a wedge and a set of oriented triangles of the
  // complex's finitely many vertices, so they are finitely many and a
  // target meets each at most once: the guard is exact and needs no cap.
  // A kPinched pass tries the targets once more on the wedges themselves
  // with relocation by peeling in place of the exhaustive search
  // (remove_witnessed_vertex).
  //
  // Work bound, with no cap: let n be the cells of the region a target's
  // wedge grows into (its constraint-delimited region, at most the
  // complex), so its cavities have O(n) vertices and walls. The target
  // takes at most one layer per obstruction state and per BFS level of the
  // region, whichever ends first. A layer rebuilds its merged cavity
  // breadth first in O(n) cells and runs a fixed number of wraps (options
  // 1, 3 and 6 on the wedge and on the merged cavity): a wrap takes at most
  // two steps per oriented triangle of its vertices, O(n^3) (one per cell
  // of the fill, O(n^2), when the wrap completes; its partial fill keeps
  // its cells interior-disjoint), and a step O(n) apex extractions of O(n)
  // insphere tests, O(n) visibility tests of O(n) separation tests, and
  // one overlap test per open side, O(n^3); so the target's work is
  // polynomial in n per layer, O(n^7) predicates at worst. Visibility facts
  // persist across the layers, targets and options of the point
  // (SuppressionMemo): a repeated cavity tests no wall again and a grown
  // one tests its new walls only against the cells its previous layer
  // saw. Only the exhaustive search (option 5; Steiner-free fill existence
  // is NP-complete) is not polynomial, and it runs on the wedge and its
  // first layer only, never on the region beyond.
  int32_t suppress_vertex(int32_t s, SuppressionPass pass, bool& removed) {
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
        for (const Triple& key : faces->keys_through(s)) {
          for (int32_t vertex : key) {
            if (vertex != s) ring.insert(vertex);
          }
        }
      }
      targets.assign(ring.begin(), ring.end());
    }
    targets.push_back(-1);
    const auto committed = [&]() {
      if (segment) {
        auto& chain = edge_chains_[static_cast<std::size_t>(witness.entity)];
        chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(position));
        std::vector<std::size_t>& rows = vertex_rows_[static_cast<std::size_t>(s)];
        rows.erase(std::find(rows.begin(), rows.end(), static_cast<std::size_t>(witness.entity)));
      }
      return PHX_MC_OK;
    };
    SuppressionGrowth growth;
    SuppressionMemo memo;
    if (pass != SuppressionPass::kGrown && pass != SuppressionPass::kPockets) {
      for (int32_t target : targets) {
        const int32_t status = remove_witnessed_vertex(
            s, target, p, q, 0, pass == SuppressionPass::kPinched, false, memo, removed, growth);
        if (status != PHX_MC_OK) return status;
        if (removed) return committed();
      }
      return PHX_MC_OK;
    }
    std::vector<char> open(targets.size(), 1);
    std::vector<std::set<std::pair<std::size_t, std::vector<FacetKey>>>> met(targets.size());
    for (int layers = 1; std::find(open.begin(), open.end(), 1) != open.end(); ++layers) {
      for (std::size_t row = 0; row < targets.size(); ++row) {
        if (open[row] == 0) continue;
        const int32_t status =
            remove_witnessed_vertex(s, targets[row], p, q, layers, false,
                                    pass == SuppressionPass::kPockets, memo, removed, growth);
        if (status != PHX_MC_OK) return status;
        if (removed) return committed();
        if (!growth.extended ||
            !met[row].emplace(growth.wedge, std::move(growth.obstruction)).second) {
          open[row] = 0;
        }
      }
    }
    return PHX_MC_OK;
  }

  // The finite cell whose closure holds `seed`, found by the visibility walk
  // (Triangulation3D::locate); refused (INVALID_SEED) outside the finite
  // cells or on a marked face. The cells whose closures hold the seed are
  // that cell, the two cells of a face, the ring of an edge, or the star of
  // a vertex holding it, and every face of them through the seed is checked.
  int32_t seed_tet(const double* seed, int64_t row) {
    const TetrahedralComplex& complex = triangulation_->complex();
    const ExactPointRef position{seed, nullptr};
    const auto refuse = [&]() {
      fail(PHX_MC_INVALID_INPUT, kPlcInvalidSeed, kPlcSeed, row);
      return -1;
    };
    const int32_t found = triangulation_->locate(position);
    if (!triangulation_->valid_live_cell(found) ||
        is_ghost(complex.tets[static_cast<std::size_t>(found)])) {
      return refuse();
    }
    const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(found)];
    int zero_slots[4];
    int zeros = 0;
    for (int slot = 0; slot < 4; ++slot) {
      const int sign = triangulation_->orient_with(tet, slot, position);
      if (sign < 0) return refuse();
      if (sign == 0) zero_slots[zeros++] = slot;
    }
    std::vector<int32_t> cells{found};
    if (zeros == 1) {
      cells.push_back(tet.n[zero_slots[0]]);
    } else if (zeros == 2 || zeros == 3) {
      int32_t ends[2] = {-1, -1};
      // The 4 - zeros nonzero slots (at most two) are the edge or vertex ends.
      int count = 0;
      for (int slot = 0; slot < 4 && count < 2; ++slot) {
        if (std::find(zero_slots, zero_slots + zeros, slot) == zero_slots + zeros) {
          ends[count++] = tet.v[slot];
        }
      }
      const int32_t status =
          zeros == 2 ? edge_cells(ends[0], ends[1], cells) : vertex_cells(ends[0], cells);
      if (status != PHX_MC_OK) return -1;
    } else if (zeros == 4) {
      return refuse();
    }
    for (int32_t cell : cells) {
      const Tetrahedron& around = complex.tets[static_cast<std::size_t>(cell)];
      if (is_ghost(around)) continue;
      for (int slot = 0; slot < 4; ++slot) {
        if (complex.constraint(cell, slot) != kNoConstraint &&
            triangulation_->orient_with(around, slot, position) == 0) {
          return refuse();
        }
      }
    }
    return found;
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
    queue.reserve(static_cast<std::size_t>(std::max<int64_t>(complex.finite_count, 0)));
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
      if (tet.constrained != 0) {
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
      }
      if (corner_count_ != 0) {
        for (int32_t vertex : tet.v) {
          if (artificial_corner_vertex(vertex)) {
            const int32_t status = assign(static_cast<int32_t>(t), kPlcVoidRegion, -1);
            if (status != PHX_MC_OK) return status;
            break;
          }
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
        if (tet.constrained != 0 && complex.constraint(t, slot) != kNoConstraint) continue;
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

  // A subface of a conforming source: the lower-numbered finite cell through
  // it and its slot there (where mark_source_faces labels it), its facet,
  // its source triangle and the cells on both sides.
  struct ConformingFace {
    int32_t cell;
    int slot;
    int32_t facet;
    int32_t triangle;
    std::array<int32_t, 2> sides;
  };
  // conforming_faces' answer. kNever: a precondition fails or a subface is
  // no face of the complex, which no further points can change (a face of
  // the Delaunay triangulation of more points, under the same perturbation,
  // is one of the triangulation of the fewer points it joins).
  enum class Conformity : std::uint8_t { kConforms, kNever, kRefused };

  // classify_regions for a conforming source without seeds, read from its
  // subfaces instead of the source face table: the labels, when no two
  // meet and none is missing, are the ones classify_regions gives (each
  // region of cells the subfaces enclose takes the one label its seeds
  // agree on). False, with the labels to be redone, when classify_regions
  // must report a failure in its order.
  bool classify_conforming(const std::vector<ConformingFace>& faces) {
    TetrahedralComplex& complex = triangulation_->complex();
    complex.clear_regions();
    std::vector<int32_t> queue;
    queue.reserve(static_cast<std::size_t>(complex.finite_count));
    const auto assign = [&](int32_t tet, int32_t region) {
      const int32_t current = complex.region(tet);
      if (current != kNoRegion) return current == region;
      complex.set_region(tet, region);
      queue.push_back(tet);
      return true;
    };
    for (const ConformingFace& face : faces) {
      const int32_t* source =
          candidate_.input_triangles.data() + 3 * static_cast<std::size_t>(face.triangle);
      const FacetKey oriented = FacetKey::of(source);
      for (int side = 0; side < 2; ++side) {
        const int32_t cell = face.sides[static_cast<std::size_t>(side)];
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
        if (is_ghost(tet)) continue;
        int32_t facet[3];
        oriented_facet(tet.v, neighbor_slot(tet, face.sides[static_cast<std::size_t>(1 - side)]),
                       facet);
        const int32_t region =
            input_.facet_regions[2 * face.facet + (FacetKey::of(facet) == oriented ? 0 : 1)];
        if (!assign(cell, region < 0 ? kPlcVoidRegion : region)) return false;
      }
    }
    if (corner_count_ != 0) {
      for (std::size_t t = 0; t < complex.tets.size(); ++t) {
        const Tetrahedron& tet = complex.tets[t];
        if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
        if (std::any_of(tet.v, tet.v + 4, [&](int32_t vertex) {
              return artificial_corner_vertex(vertex);
            }) &&
            !assign(static_cast<int32_t>(t), kPlcVoidRegion)) {
          return false;
        }
      }
    }
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const int32_t t = queue[head];
      const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(t)];
      for (int slot = 0; slot < 4; ++slot) {
        if (complex.constraint(t, slot) != kNoConstraint) continue;
        const int32_t next = tet.n[slot];
        const Tetrahedron& neighbor = complex.tets[static_cast<std::size_t>(next)];
        if (neighbor.v[0] == kDeadVertex || is_ghost(neighbor)) continue;
        if (!assign(next, tet.region)) return false;
      }
    }
    return static_cast<int64_t>(queue.size()) == complex.finite_count;
  }

  // Whether the source conforms to the triangulation build_triangulation
  // built first: without planar polygons, exact positions or a metric, with
  // every finite cell exactly positive (so the complex triangulates its
  // points and two of its faces meet in a shared face or not at all), every
  // subface a face of it that mark_source_faces would label (into `faces`,
  // in cell order) and every chain row an edge of a subface. The subfaces,
  // chain rows and points then meet only in shared vertices and edges, so
  // every source contact is legal, and recovery has nothing to edit.
  Conformity conforming_faces(std::vector<ConformingFace>& faces) const {
    faces.clear();
    if (!planar_polygons_.empty() || planar_source_faces_.size() != 0 ||
        !exact_points_.empty() || input_.affine_metric != nullptr) {
      return Conformity::kNever;
    }
    for (std::size_t row = 0; row < edge_chains_.size(); ++row) {
      if (edge_chains_[row].size() > 1 &&
          edge_triangle_offsets_[row] == edge_triangle_offsets_[row + 1]) {
        return Conformity::kNever;
      }
    }
    // Without planar polygons the subfaces are the input triangles. They are
    // listed by lowest vertex: per source point (offsets), the other two
    // vertices of each, ascending, and its triangle.
    struct LowestRow {
      int32_t second;
      int32_t third;
      int32_t triangle;
    };
    const std::size_t triangles = candidate_.input_triangles.size() / 3;
    const auto points = static_cast<std::size_t>(source_point_count_);
    std::vector<uint32_t> offsets(points + 2, 0);
    for (std::size_t t = 0; t < triangles; ++t) {
      const Triple key = sorted_triple(candidate_.input_triangles.data() + 3 * t);
      ++offsets[static_cast<std::size_t>(key[0]) + 2];
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<LowestRow> lowest(triangles);
    for (std::size_t t = 0; t < triangles; ++t) {
      const Triple key = sorted_triple(candidate_.input_triangles.data() + 3 * t);
      lowest[offsets[static_cast<std::size_t>(key[0]) + 1]++] = {key[1], key[2],
                                                                  static_cast<int32_t>(t)};
    }
    // offsets[v] .. offsets[v + 1] are now the rows of v.
    const auto subface = [&](const int32_t* face) {
      const Triple key = sorted_triple(face);
      if (key[0] < 0 || key[0] >= source_point_count_) return -1;
      const auto vertex = static_cast<std::size_t>(key[0]);
      for (uint32_t at = offsets[vertex]; at < offsets[vertex + 1]; ++at) {
        if (lowest[at].second == key[1] && lowest[at].third == key[2]) return lowest[at].triangle;
      }
      return -1;
    };
    const TetrahedralComplex& complex = triangulation_->complex();
    const std::size_t tet_count = complex.tets.size();
    const auto scan_range = [&](std::size_t begin, std::size_t end,
                                std::vector<ConformingFace>& out) -> bool {
      for (std::size_t row = begin; row < end; ++row) {
        const auto cell = static_cast<int32_t>(row);
        if (!complex.live(cell)) continue;
        const Tetrahedron& tet = complex.tets[row];
        const bool ghost = is_ghost(tet);
        if (!ghost && orient3d_filtered(point(tet.v[0]), point(tet.v[1]), point(tet.v[2]),
                                        point(tet.v[3])) <= 0) {
          return false;
        }
        const int source_corners =
            static_cast<int>(tet.v[0] >= 0 && tet.v[0] < source_point_count_) +
            static_cast<int>(tet.v[1] >= 0 && tet.v[1] < source_point_count_) +
            static_cast<int>(tet.v[2] >= 0 && tet.v[2] < source_point_count_) +
            static_cast<int>(tet.v[3] >= 0 && tet.v[3] < source_point_count_);
        if (source_corners < 3) continue;
        for (int slot = 0; slot < 4; ++slot) {
          // Each face once, from its lower cell.
          const int32_t other = tet.n[slot];
          if (other < cell) continue;
          int32_t face[3];
          oriented_facet(tet.v, slot, face);
          const int32_t triangle = subface(face);
          if (triangle < 0) continue;
          const Tetrahedron& beyond = complex.tets[static_cast<std::size_t>(other)];
          const int back = neighbor_slot(beyond, cell);
          const int32_t* source =
              candidate_.input_triangles.data() + 3 * static_cast<std::size_t>(triangle);
          const FacetKey oriented = FacetKey::of(source);
          ConformingFace conforming{
              cell, slot,
              input_.polygon_facets[candidate_.input_polygons[static_cast<std::size_t>(triangle)]],
              triangle, {cell, other}};
          if (!ghost && !is_ghost(beyond)) {
            int32_t opposite[3];
            oriented_facet(beyond.v, back, opposite);
            if ((FacetKey::of(face) == oriented) == (FacetKey::of(opposite) == oriented)) {
              return false;
            }
          } else if (corner_count_ != 0) {
            return false;
          } else if (ghost) {
            conforming.cell = other;
            conforming.slot = back;
          }
          out.push_back(conforming);
        }
      }
      return true;
    };
    constexpr std::size_t kChunkSize = 8192;
    const auto workers = static_cast<uint32_t>(std::max<int32_t>(1, input_.parallelism));
    if (workers > 1 && tet_count >= kChunkSize) {
      const std::size_t chunk_count = (tet_count + kChunkSize - 1) / kChunkSize;
      std::vector<std::vector<ConformingFace>> chunks(chunk_count);
      std::vector<uint8_t> refused(chunk_count, 0U);
      const int32_t status = parallel_for(
          tet_count, ParallelRequest{kChunkSize, workers},
          [&](std::size_t begin, std::size_t end, ChunkContext& chunk) -> int32_t {
            const GradualNearestScope mode;
            if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
            if (!scan_range(begin, end, chunks[chunk.chunk_index])) {
              refused[chunk.chunk_index] = 1U;
            }
            return PHX_MC_OK;
          });
      if (status != PHX_MC_OK ||
          std::find(refused.begin(), refused.end(), uint8_t{1U}) != refused.end()) {
        return Conformity::kRefused;
      }
      std::size_t total = 0;
      for (const auto& bank : chunks) total += bank.size();
      faces.reserve(total);
      for (const auto& bank : chunks) {
        faces.insert(faces.end(), bank.begin(), bank.end());
      }
    } else {
      faces.reserve(triangles);
      if (!scan_range(0, tet_count, faces)) return Conformity::kRefused;
    }
    return faces.size() == source_faces_.size() ? Conformity::kConforms : Conformity::kNever;
  }

  // check_embedded_entities for a conforming source, true when it passes:
  // every chain row is an edge of a subface with a finite domain cell on one
  // side, and every source point is used by a triangle or segment. False
  // leaves the decision to check_embedded_entities.
  bool conforming_embedded(const std::vector<ConformingFace>& faces) const {
    const TetrahedralComplex& complex = triangulation_->complex();
    std::vector<char> referenced(static_cast<std::size_t>(source_point_count_), 0);
    for (int32_t vertex : candidate_.input_triangles) referenced[static_cast<std::size_t>(vertex)] = 1;
    for (int64_t row = 0; row < input_.segment_count; ++row) {
      referenced[static_cast<std::size_t>(input_.segments[2 * row])] = 1;
      referenced[static_cast<std::size_t>(input_.segments[2 * row + 1])] = 1;
    }
    if (std::find(referenced.begin(), referenced.end(), 0) != referenced.end()) return false;
    std::vector<char> in_domain(candidate_.input_triangles.size() / 3, 0);
    for (const ConformingFace& face : faces) {
      for (int32_t side : face.sides) {
        if (!is_ghost(complex.tets[static_cast<std::size_t>(side)]) && complex.region(side) >= 0) {
          in_domain[static_cast<std::size_t>(face.triangle)] = 1;
        }
      }
    }
    for (std::size_t row = 0; row < edge_chains_.size(); ++row) {
      if (edge_chains_[row].size() < 2) continue;
      const auto begin = edge_triangle_rows_.begin() +
                         static_cast<std::ptrdiff_t>(edge_triangle_offsets_[row]);
      const auto end = edge_triangle_rows_.begin() +
                       static_cast<std::ptrdiff_t>(edge_triangle_offsets_[row + 1]);
      if (std::none_of(begin, end, [&](int32_t triangle) {
            return in_domain[static_cast<std::size_t>(triangle)] != 0;
          })) {
        return false;
      }
    }
    return true;
  }

  int32_t check_embedded_entities() {
    const TetrahedralComplex& complex = triangulation_->complex();
    sync_complex();
    // A live finite domain cell through `vertex` (and `other`, when given).
    const auto in_domain = [&](int32_t vertex, int32_t other) {
      const std::vector<int32_t>& cells = incidence_[static_cast<std::size_t>(vertex)];
      return std::any_of(cells.begin(), cells.end(), [&](int32_t cell) {
        const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(cell)];
        return !is_ghost(tet) && complex.region(cell) >= 0 &&
               (other < 0 || vertex_slot(tet, other) >= 0);
      });
    };
    for (std::size_t edge = 0; edge < edge_chains_.size(); ++edge) {
      for (std::size_t interval = 0; interval + 1 < edge_chains_[edge].size(); ++interval) {
        if (!in_domain(edge_chains_[edge][interval].vertex, edge_chains_[edge][interval + 1].vertex)) {
          return fail(PHX_MC_INVALID_INPUT, kPlcOutsideDomain, kPlcEdge, edge);
        }
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
      if (!in_domain(vertex, -1)) return fail(PHX_MC_INVALID_INPUT, kPlcOutsideDomain, kPlcPoint, vertex);
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
    // Published row of each complex cell (-1: unpublished), for the adjacency.
    std::vector<int32_t> published_row(complex.tets.size(), -1);
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
      published_row[t] = static_cast<int32_t>(candidate_.tet_regions.size());
      candidate_.tets.insert(candidate_.tets.end(), output_tet, output_tet + 4);
      candidate_.tet_regions.push_back(complex.region(static_cast<int32_t>(t)));
    }
    candidate_.tet_neighbors.clear();
    candidate_.tet_neighbors.reserve(candidate_.tets.size());
    for (std::size_t t = 0; t < complex.tets.size(); ++t) {
      if (published_row[t] < 0) continue;
      for (const int32_t across : complex.tets[t].n) {
        candidate_.tet_neighbors.push_back(
            across < 0 ? -1 : published_row[static_cast<std::size_t>(across)]);
      }
    }
    // Recovery found every source face and edge in the Delaunay
    // triangulation of the source points (and the corners) as built.
    candidate_.delaunay = !local_edits_ && volume_count_ == preseeded_count_ &&
                          exact_points_.empty() &&
                          candidate_points == static_cast<std::size_t>(source_point_count_) &&
                          input_.affine_metric == nullptr;
    if (candidate_.delaunay && preseeded_count_ != 0 && !published_volume.empty()) {
      candidate_.preseeded_radius = input_.uniform_radius;
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
    // Hidden internal edges (the trailing rows past visible_edges_) are
    // neither PLC edges nor segments of the result; their split points were
    // all suppressed.
    candidate_.plc_edges.resize(2 * visible_edges_);
    candidate_.hidden_plc_edges = 0;
    std::size_t segment_count = 0;
    for (std::size_t edge = 0; edge < visible_edges_; ++edge) {
      if (edge_chains_[edge].size() > 1) segment_count += edge_chains_[edge].size() - 1;
    }
    candidate_.segments.reserve(2 * segment_count);
    candidate_.segment_sources.reserve(segment_count);
    for (std::size_t edge = 0; edge < visible_edges_; ++edge) {
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
    // The radii of the source points are those of initialize_points when
    // the published points and segments are the source ones.
    if (published_points == static_cast<std::size_t>(source_point_count_) &&
        candidate_.segments == source_protection_edges_) {
      candidate_.protection = std::move(source_protection_);
    } else {
      candidate_.protection = protection_radii(
          candidate_.points.data(), static_cast<int64_t>(published_points),
          candidate_.segments, candidate_.input_triangles, input_.affine_metric);
    }
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
  // The preparer's edge index (empty in run_cavity_fixture).
  SourceEdgeUses source_edges_;
  // PLC edge rows published (the rest are hidden internal edges), and the
  // exact normals of the declared facet planes (empty without any).
  std::size_t visible_edges_ = 0;
  std::vector<std::array<Expansion, 3>> plane_normals_;
  int32_t status_ = PHX_MC_OK;
  int32_t source_point_count_ = 0;
  int32_t corner_count_ = 0;
  PlcFailure failure_{};
  std::vector<double> work_points_;
  std::vector<char> alive_;
  std::unique_ptr<Triangulation3D> triangulation_;
  bool incremental_triangulation_ready_ = false;
  // triangulation_ holds the Delaunay triangulation of the source points
  // built by convex_boundary, not yet extended by build_triangulation.
  bool source_triangulation_ready_ = false;
  // Protection radii of the source points over the source edges (as
  // protection_radii receives them), for publish.
  std::vector<double> source_protection_;
  std::vector<int32_t> source_protection_edges_;
  std::vector<int32_t> vertex_tet_;
  // Live cells (finite and ghost) through each work vertex, kept current
  // from the complex's edit journal.
  std::vector<std::vector<int32_t>> incidence_;
  // Planar polygon holding each face of a current verified cover.
  std::map<Triple, int32_t> cover_polygon_;
  std::vector<CellEvent> journal_;
  // Vertices whose incidence or vertex_tet_ hint changed since vertex_tet_
  // was last reset to the highest live finite cell through each vertex.
  std::vector<int32_t> touched_;
  std::vector<char> touched_mark_;
  // Candidate points by carrier box and by exact carrier.
  BoxGrid point_grid_;
  std::unordered_map<CarrierKey, int32_t, CarrierKeyHash> carrier_vertex_;
  // Source edge rows by carrier box, and the rows whose chains hold each
  // candidate vertex (ascending).
  BoxGrid edge_grid_;
  std::vector<std::vector<std::size_t>> vertex_rows_;
  // Rows with a chain interval not known to be an edge of the complex.
  std::set<std::size_t> unverified_rows_;
  // The star of the first endpoint of crossing_cells, reused across calls.
  std::vector<int32_t> crossing_star_;
  std::vector<std::uint32_t> star_marks_;
  std::vector<int32_t> star_;
  std::uint32_t star_stamp_ = 0;
  SourceFaceTable source_faces_;
  std::vector<PlcPlanarPolygon> planar_polygons_;
  std::vector<int32_t> planar_polygon_index_;
  SourceFaceTable planar_source_faces_;
  std::vector<double> corner_points_;
  std::vector<std::vector<PlcEdgePoint>> edge_chains_;
  std::vector<int32_t> source_segment_rows_;
  std::vector<std::size_t> edge_triangle_offsets_;
  std::vector<int32_t> edge_triangle_rows_;
  // Acute input vertices: apices of the concentric split shells.
  std::vector<char> shell_apex_;
  // Exact positions of witnessed split vertices, keyed by work vertex.
  ExactVertexBank exact_points_;
  // The first inconclusive (over-budget) exhaustive search of each point:
  // states, bytes and depth when it stopped.
  std::map<int32_t, std::array<int64_t, 3>> inconclusive_;
  // Current chain intervals, which recovery edits must never remove, and
  // the row of each.
  std::unordered_map<uint64_t, std::size_t> chain_intervals_;
  std::size_t volume_count_ = 0;
  std::size_t preseeded_count_ = 0;
  bool allow_preseeded_ = true;
  bool preseed_conforms_ = false;
  std::vector<int32_t> volume_output_;
  // True once a local recovery edit has left the complex non-Delaunay.
  bool local_edits_ = false;
  // Whether run reached the deferred contact check, and whether it failed.
  bool contacts_settled_ = false;
  bool contact_failure_ = false;
  // Point count at the last rebuild requested for a strongly Delaunay subface.
  std::size_t subface_rebuild_points_ = 0;
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
    if (input.search_node_budget < 0 || input.search_byte_budget < 0 ||
        !(input.uniform_radius >= 0.0) || !std::isfinite(input.uniform_radius) ||
        input.parallelism < 1) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (input.affine_metric != nullptr) {
      TetMesh metric;
      const int32_t status = metric.set_affine_metric({input.affine_metric, 9});
      if (status != PHX_MC_OK) return status;
    }
    PlcRecovery prepared;
    SourcePreparer preparer(input, prepared, true);
    const int32_t source_status = preparer.run();
    if (source_status != PHX_MC_OK) {
      result.failure = prepared.failure;
      result.source_refusal = prepared.source_refusal;
      return source_status;
    }
    PlcRecoveryBuilder builder(input, prepared, preparer.edge_uses());
    int32_t status = builder.run(result, [&] { return preparer.check_contacts(); });
    if (status == PHX_MC_OK) return PHX_MC_OK;
    // An illegal source contact is reported before any later failure.
    const int32_t contacts = builder.contact_failure()    ? status
                             : builder.contacts_settled() ? PHX_MC_OK
                                                          : preparer.check_contacts();
    if (contacts != PHX_MC_OK) {
      result.failure = prepared.failure;
      result.source_refusal = prepared.source_refusal;
      return contacts;
    }
    if (builder.used_nonconforming_preseed()) {
      PlcRecoveryBuilder fallback(input, prepared, preparer.edge_uses(), false);
      status = fallback.run(result, {});
      if (status == PHX_MC_OK) return PHX_MC_OK;
      result.failure = fallback.failure();
      result.source_refusal = fallback.source_refusal();
      result.suppression_search = fallback.suppression_search();
      return status;
    }
    result.failure = builder.failure();
    result.source_refusal = builder.source_refusal();
    result.suppression_search = builder.suppression_search();
    return status;
  } catch (const std::bad_alloc&) {
    result.failure = {};
    return PHX_MC_ALLOCATION_FAILURE;
  }
}

}  // namespace phx::mc

