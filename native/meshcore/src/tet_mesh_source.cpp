//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact source-witness validation, ancestry coverage, and orientation authority.
#include "tet_mesh.hpp"
#include "dyadic.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "facet_coverage.hpp"
#include "geometry3.hpp"
#include "exact_scalar.hpp"
#include "predicates.hpp"

namespace phx::mc {
namespace {

bool same_bits(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

int exact_orientation(const Expansion* const points[4][3]) {
  const Expansion x0 = *points[1][0] - *points[0][0];
  const Expansion y0 = *points[1][1] - *points[0][1];
  const Expansion z0 = *points[1][2] - *points[0][2];
  const Expansion x1 = *points[2][0] - *points[0][0];
  const Expansion y1 = *points[2][1] - *points[0][1];
  const Expansion z1 = *points[2][2] - *points[0][2];
  const Expansion x2 = *points[3][0] - *points[0][0];
  const Expansion y2 = *points[3][1] - *points[0][1];
  const Expansion z2 = *points[3][2] - *points[0][2];
  return (x0 * (y1 * z2 - z1 * y2) - y0 * (x1 * z2 - z1 * x2) +
          z0 * (x1 * y2 - y1 * x2)).sign();
}

// These scratch records validate source witnesses only; edit ancestry is
// validated on the original change payload before it enters the mesh state.
TetEditVertex source_witness_record(std::uint64_t id, const double* carrier,
                                    const SourceWitness& witness) {
  TetEditVertex vertex{};
  vertex.id = id;
  std::copy_n(carrier, 3, vertex.carrier.begin());
  vertex.witness = witness;
  vertex.ancestry = {};
  return vertex;
}

// n = first axis x second axis of a declared plane (origin, first axis,
// second axis), exactly.
void plane_normal(const double* plane, Expansion* normal) {
  const double* u = plane + 3;
  const double* v = plane + 6;
  normal[0] = Expansion::product(u[1], v[2]) - Expansion::product(u[2], v[1]);
  normal[1] = Expansion::product(u[2], v[0]) - Expansion::product(u[0], v[2]);
  normal[2] = Expansion::product(u[0], v[1]) - Expansion::product(u[1], v[0]);
}

// n.(point - origin), exactly.
Expansion plane_offset(const double* plane, const Expansion* normal, const double* point) {
  return normal[0] * Expansion::difference(point[0], plane[0]) +
         normal[1] * Expansion::difference(point[1], plane[1]) +
         normal[2] * Expansion::difference(point[2], plane[2]);
}

// Sign of n.(a x b).
int triple_sign(const Expansion* n, const Expansion* a, const Expansion* b) {
  return (n[0] * (a[1] * b[2] - a[2] * b[1]) + n[1] * (a[2] * b[0] - a[0] * b[2]) +
          n[2] * (a[0] * b[1] - a[1] * b[0]))
      .sign();
}

// Orientation of the orthogonal projections of a, b, c onto a plane of
// normal n, seen from n: projection changes b - a and c - a only by multiples
// of n, which leave n.((b - a) x (c - a)) unchanged.
int projected_orientation(const Expansion* n, const Expansion* a, const Expansion* b,
                          const Expansion* c) {
  Expansion ab[3];
  Expansion ac[3];
  for (int axis = 0; axis < 3; ++axis) {
    ab[axis] = b[axis] - a[axis];
    ac[axis] = c[axis] - a[axis];
  }
  return triple_sign(n, ab, ac);
}

// Whether the projected closed segment pq meets the open interior of the
// positively oriented projected triangle t (separating-axis test over the
// triangle's edges and the segment's line, all exact).
bool segment_meets_open_triangle(const Expansion* n, const Expansion* p, const Expansion* q,
                                 const Expansion* const* t) {
  for (int edge = 0; edge < 3; ++edge) {
    const Expansion* a = t[edge];
    const Expansion* b = t[(edge + 1) % 3];
    if (projected_orientation(n, a, b, p) <= 0 && projected_orientation(n, a, b, q) <= 0) {
      return false;
    }
  }
  const int sides[3] = {projected_orientation(n, p, q, t[0]), projected_orientation(n, p, q, t[1]),
                        projected_orientation(n, p, q, t[2])};
  const bool nonnegative = sides[0] >= 0 && sides[1] >= 0 && sides[2] >= 0;
  const bool nonpositive = sides[0] <= 0 && sides[1] <= 0 && sides[2] <= 0;
  return !nonnegative && !nonpositive;
}

bool declared_plane_point(const double* plane, const double* parameters, Expansion* point) {
  if (!std::isfinite(parameters[0]) || !std::isfinite(parameters[1])) return false;
  for (int axis = 0; axis < 3; ++axis) {
    point[axis] = Expansion(plane[axis]) + Expansion(plane[3 + axis]).scaled(parameters[0]) +
                  Expansion(plane[6 + axis]).scaled(parameters[1]);
  }
  return true;
}

double plane_deviation(const double* plane, const Expansion* point, const double* position) {
  Expansion normal[3];
  plane_normal(plane, normal);
  return plane_offset(plane, normal, position).is_zero() ? 0.0 : outward_distance(position, point);
}
}  // namespace

double declared_plane_deviation(const double* plane, const double* parameters,
                                const double* position) {
  Expansion point[3];
  return declared_plane_point(plane, parameters, point) ? plane_deviation(plane, point, position)
                                                         : -1.0;
}

bool TetMesh::source_carrier_valid(const TetEditVertex& vertex) const {
  for (double coordinate : vertex.carrier) {
    if (!std::isfinite(coordinate) || !coordinate_in_domain(coordinate)) return false;
  }
  if (vertex.has_edge_parameter) {
    if (!std::isfinite(vertex.edge_parameter) ||
        !(vertex.edge_parameter > 0.0 && vertex.edge_parameter < 1.0) ||
        vertex.edge_endpoints[0] == vertex.edge_endpoints[1]) return false;
    const std::int32_t first = vertex_slot(vertex.edge_endpoints[0]);
    const std::int32_t second = vertex_slot(vertex.edge_endpoints[1]);
    if (!live_vertex(first) || !live_vertex(second)) return false;
    const Dyadic parameter(vertex.edge_parameter);
    const Dyadic one(1.0);
    for (int axis = 0; axis < 3; ++axis) {
      const Dyadic origin(point(first)[axis]);
      const Dyadic destination(point(second)[axis]);
      const Dyadic exact_coordinate =
          origin + (destination - origin) * parameter;
      const double rounded = exact::rounded_quotient(exact_coordinate, one);
      if (!same_bits(rounded, vertex.carrier[static_cast<std::size_t>(axis)])) return false;
    }
  }
  if (!vertex.ancestry.empty()) {
    Dyadic weight_sum(0.0);
    const Dyadic one(1.0);
    std::array<Dyadic, 3> coordinates{
        Dyadic(0.0), Dyadic(0.0), Dyadic(0.0)};
    for (std::size_t row = 0; row < vertex.ancestry.size(); ++row) {
      const TetVertexAncestry& ancestor = vertex.ancestry[row];
      if (!std::isfinite(ancestor.weight) || ancestor.weight < 0.0 ||
          ancestor.weight > 1.0) return false;
      for (std::size_t prior = 0; prior < row; ++prior) {
        if (vertex.ancestry[prior].source_vertex_id == ancestor.source_vertex_id) return false;
      }
      const std::int32_t source_slot = vertex_slot(ancestor.source_vertex_id);
      if (!live_vertex(source_slot)) return false;
      const Dyadic weight(ancestor.weight);
      weight_sum = weight_sum + weight;
      for (int axis = 0; axis < 3; ++axis) {
        coordinates[static_cast<std::size_t>(axis)] =
            coordinates[static_cast<std::size_t>(axis)] +
            Dyadic(point(source_slot)[axis]) * weight;
      }
    }
    if (!(weight_sum - one).is_zero()) return false;
    for (int axis = 0; axis < 3; ++axis) {
      const double rounded = exact::rounded_quotient(
          coordinates[static_cast<std::size_t>(axis)], one);
      if (!same_bits(rounded, vertex.carrier[static_cast<std::size_t>(axis)])) return false;
    }
  }
  const SourceWitness& witness = vertex.witness;
  if (witness.stratum == SourceStratum::kNone) {
    return witness.entity == -1 && witness.deviation == 0.0;
  }
  if (!std::isfinite(witness.deviation) || witness.deviation < 0.0) return false;
  // The exact source point S of the witness is authoritative and the carrier
  // must be its correctly rounded realization. The recorded deviation must
  // bound |carrier - S|; it may exceed the entity tolerance (which governs an
  // authored approximation) only when it is exactly the realization rounding
  // bound, i.e. the carrier is round(S) and nothing else was approximated. A
  // plane witness names the declared plane of its face row; any binary64
  // parameters address a point of it.
  double carrier[3];
  double deviation = 0.0;
  double tolerance = 0.0;
  if (witness.stratum == SourceStratum::kPlane) {
    if (!plane_carrier(witness.entity, witness.parameters, carrier, deviation)) return false;
    tolerance = source_face_tolerances_[static_cast<std::size_t>(witness.entity)];
  } else {
    const double* corners[3] = {nullptr, nullptr, nullptr};
    if (!source_parameters_valid(witness.stratum, witness.parameters) ||
        !source_corners(witness.stratum, witness.entity, corners) ||
        !source_carrier(corners, witness.stratum, witness.parameters, carrier, deviation)) {
      return false;
    }
    const std::size_t row = static_cast<std::size_t>(witness.entity);
    tolerance = witness.stratum == SourceStratum::kSegment ? source_segment_tolerances_[row]
                                                           : source_face_tolerances_[row];
  }
  if (witness.deviation < deviation ||
      (witness.deviation > tolerance && witness.deviation != deviation)) {
    return false;
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (!same_bits(carrier[axis], vertex.carrier[static_cast<std::size_t>(axis)])) return false;
  }
  return true;
}

bool TetMesh::source_corners(SourceStratum stratum, std::int32_t row,
                             const double** corners) const {
  if (row < 0) return false;
  const std::size_t entity = static_cast<std::size_t>(row);
  if (stratum == SourceStratum::kSegment) {
    if (entity >= source_segments_.size() / 2) return false;
    for (std::size_t k = 0; k < 2; ++k) {
      corners[k] = source_points_.data() + 3 * static_cast<std::size_t>(source_segments_[2 * entity + k]);
    }
    corners[2] = nullptr;
    return true;
  }
  if (stratum == SourceStratum::kFacet) {
    if (entity >= source_faces_.size() / 3) return false;
    for (std::size_t k = 0; k < 3; ++k) {
      corners[k] = source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * entity + k]);
    }
    return true;
  }
  return false;
}

bool TetMesh::witness_position(const SourceWitness& witness, const double* carrier,
                               Expansion* position) const {
  if (witness.stratum == SourceStratum::kNone) {
    for (int axis = 0; axis < 3; ++axis) position[axis] = Expansion(carrier[axis]);
    return true;
  }
  if (witness.stratum == SourceStratum::kPlane) {
    return plane_point(witness.entity, witness.parameters, position);
  }
  const double* corners[3] = {nullptr, nullptr, nullptr};
  return source_corners(witness.stratum, witness.entity, corners) &&
         source_point(corners, witness.stratum, witness.parameters, position);
}

bool TetMesh::exact_position(std::int32_t slot, Expansion* position) const {
  return live_vertex(slot) &&
         witness_position(witnesses_[static_cast<std::size_t>(slot)], point(slot), position);
}

const double* TetMesh::face_plane(std::int32_t row) const noexcept {
  if (row < 0 || static_cast<std::size_t>(row) >= source_face_plane_tolerances_.size()) {
    return nullptr;
  }
  const double* plane = source_face_planes_.data() + 9 * static_cast<std::size_t>(row);
  return std::isnan(plane[0]) ? nullptr : plane;
}

bool TetMesh::plane_point(std::int32_t row, const double* parameters, Expansion* point) const {
  const double* plane = face_plane(row);
  return plane != nullptr && declared_plane_point(plane, parameters, point);
}

bool TetMesh::plane_carrier(std::int32_t row, const double* parameters, double* carrier,
                            double& deviation) const {
  Expansion point[3];
  if (!plane_point(row, parameters, point)) return false;
  for (int axis = 0; axis < 3; ++axis) carrier[axis] = nearest_double(point[axis]);
  if (!geometry::point_in_domain(carrier)) return false;
  deviation = plane_deviation(face_plane(row), point, carrier);
  return true;
}

bool TetMesh::plane_admits(std::int32_t row, const double* carrier) const {
  const double* plane = face_plane(row);
  if (plane == nullptr) return false;
  Expansion normal[3];
  plane_normal(plane, normal);
  const Expansion offset = plane_offset(plane, normal, carrier);
  const int side = offset.sign();
  if (side == 0) return true;
  // The closed rounding box of coordinate k spans half the gap to each
  // neighbouring binary64 value. Doubled: the plane meets the box iff
  // 2 n.(c - o) - side * sum_k |n_k| gap_k does not keep the sign of n.(c - o),
  // gap_k the gap on the side that moves c toward the plane.
  Expansion reach = offset.scaled(2.0);
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  for (int axis = 0; axis < 3; ++axis) {
    const int direction = normal[axis].sign();
    if (direction == 0) continue;
    const double value = carrier[axis];
    const double gap = direction == side ? value - std::nextafter(value, -kInfinity)
                                         : std::nextafter(value, kInfinity) - value;
    const Expansion magnitude = direction < 0 ? -normal[axis] : normal[axis];
    reach = side > 0 ? reach - magnitude.scaled(gap) : reach + magnitude.scaled(gap);
  }
  if (reach.sign() * side <= 0) return true;
  const double tolerance = source_face_plane_tolerances_[static_cast<std::size_t>(row)];
  if (!(tolerance > 0.0)) return false;
  const Expansion squared_norm =
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
  return (offset * offset - squared_norm * Expansion::product(tolerance, tolerance)).sign() <= 0;
}

bool TetMesh::plane_record(std::int32_t row, const double* carrier,
                           TetPlaneRecord& record) const {
  const double* plane = face_plane(row);
  if (plane == nullptr) return false;
  Dyadic first[3];
  Dyadic second[3];
  Dyadic offset[3];
  for (int axis = 0; axis < 3; ++axis) {
    first[axis] = Dyadic(plane[3 + axis]);
    second[axis] = Dyadic(plane[6 + axis]);
    offset[axis] = Dyadic(carrier[axis]) - Dyadic(plane[axis]);
  }
  const auto dot = [](const Dyadic* a, const Dyadic* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const Dyadic uu = dot(first, first);
  const Dyadic uv = dot(first, second);
  const Dyadic vv = dot(second, second);
  const Dyadic ur = dot(first, offset);
  const Dyadic vr = dot(second, offset);
  // Gram determinant |u|^2 |v|^2 - (u.v)^2 = |u x v|^2, positive for a
  // declared plane.
  const Dyadic gram = uu * vv - uv * uv;
  const Dyadic normal[3] = {first[1] * second[2] - first[2] * second[1],
                            first[2] * second[0] - first[0] * second[2],
                            first[0] * second[1] - first[1] * second[0]};
  const Dyadic height = dot(normal, offset);
  record.face_row = row;
  record.parameters[0] = exact::rounded_quotient(ur * vv - vr * uv, gram);
  record.parameters[1] = exact::rounded_quotient(vr * uu - ur * uv, gram);
  record.deviation = 0.0;
  if (height.sign() != 0) {
    // |n.(c - o)| / |n| rounded up through an upward quotient and square root.
    const double squared = exact::directed_quotient_bounds(height * height, gram).upper;
    double distance = std::sqrt(squared);
    while ((Dyadic(distance) * Dyadic(distance) - Dyadic(squared)).sign() < 0) {
      distance = std::nextafter(distance, std::numeric_limits<double>::infinity());
    }
    record.deviation = distance;
  }
  return true;
}

void TetMesh::plane_records(std::vector<std::uint64_t>& vertex_ids,
                            std::vector<TetPlaneRecord>& records) const {
  vertex_ids.clear();
  records.clear();
  std::set<std::pair<std::uint64_t, std::int32_t>> incidences;
  for (const TetSnapshotFacet& facet : facets_) {
    const auto group = declared_groups_.find(facet.source);
    if (group == declared_groups_.end()) continue;
    for (std::uint64_t vertex : facet.vertices) incidences.emplace(vertex, group->second.rows.front());
  }
  for (const auto& [vertex, row] : incidences) {
    const std::int32_t slot = vertex_slot(vertex);
    TetPlaneRecord record;
    if (!live_vertex(slot) || !plane_record(row, point(slot), record)) continue;
    vertex_ids.push_back(vertex);
    records.push_back(record);
  }
}

int32_t TetMesh::install_declared_planes(std::span<const double> planes,
                                         std::span<const double> tolerances) {
  plane_refusal_.reset();
  declared_groups_.clear();
  const std::size_t face_count = source_faces_.size() / 3;
  if ((!planes.empty() && planes.size() != 9 * face_count) ||
      (!tolerances.empty() && tolerances.size() != face_count)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  source_face_planes_.assign(9 * face_count, std::numeric_limits<double>::quiet_NaN());
  source_face_plane_tolerances_.assign(face_count, 0.0);
  std::copy(planes.begin(), planes.end(), source_face_planes_.begin());
  std::copy(tolerances.begin(), tolerances.end(), source_face_plane_tolerances_.begin());
  std::unordered_map<std::int32_t, std::int32_t> first_rows;
  for (std::size_t row = 0; row < face_count; ++row) {
    const double* plane = source_face_planes_.data() + 9 * row;
    const double tolerance = source_face_plane_tolerances_[row];
    const auto undeclared = std::count_if(plane, plane + 9, [](double value) {
      return std::isnan(value);
    });
    if (!std::isfinite(tolerance) || tolerance < 0.0 || (undeclared != 0 && undeclared != 9) ||
        (undeclared == 0 && !std::all_of(plane, plane + 9, [](double value) {
          return std::isfinite(value);
        }))) {
      return PHX_MC_INVALID_INPUT;
    }
    if (undeclared == 0) {
      Expansion normal[3];
      plane_normal(plane, normal);
      if (normal[0].is_zero() && normal[1].is_zero() && normal[2].is_zero()) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    const auto [first, inserted] =
        first_rows.emplace(source_face_groups_[row], static_cast<std::int32_t>(row));
    if (inserted) continue;
    // Every row of a group declares the same plane, or none does.
    const double* group_plane = source_face_planes_.data() + 9 * static_cast<std::size_t>(first->second);
    for (int value = 0; value < 9; ++value) {
      if (std::isnan(plane[value]) != std::isnan(group_plane[value]) ||
          (!std::isnan(plane[value]) && !same_bits(plane[value], group_plane[value]))) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    if (!same_bits(tolerance, source_face_plane_tolerances_[static_cast<std::size_t>(first->second)])) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  using Edge = std::array<std::int32_t, 2>;
  std::map<std::int32_t, std::map<Edge, std::vector<Edge>>> group_edges;
  for (std::size_t row = 0; row < face_count; ++row) {
    const std::int32_t plane_row = static_cast<std::int32_t>(row);
    const double* plane = face_plane(plane_row);
    if (plane == nullptr) continue;
    DeclaredGroup& group = declared_groups_[source_face_groups_[row]];
    Expansion normal[3];
    plane_normal(plane, normal);
    Expansion corners[3][3];
    for (int corner = 0; corner < 3; ++corner) {
      const double* source = source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + static_cast<std::size_t>(corner)]);
      for (int axis = 0; axis < 3; ++axis) corners[corner][axis] = Expansion(source[axis]);
    }
    const int orientation = projected_orientation(normal, corners[0], corners[1], corners[2]);
    if (orientation == 0 || (group.orientation != 0 && orientation != group.orientation)) {
      return PHX_MC_INVALID_INPUT;
    }
    group.orientation = static_cast<std::int8_t>(orientation);
    group.rows.push_back(plane_row);
    auto& edges = group_edges[source_face_groups_[row]];
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const std::int32_t a = source_faces_[3 * row + corner];
      const std::int32_t b = source_faces_[3 * row + (corner + 1) % 3];
      edges[Edge{std::min(a, b), std::max(a, b)}].push_back(Edge{a, b});
    }
  }
  // The rows of a declared group triangulate their union as an oriented
  // manifold: an edge is used once (on the union's boundary) or twice in
  // opposite directions.
  for (const auto& [group, edges] : group_edges) {
    DeclaredGroup& declared = declared_groups_.at(group);
    for (const auto& [key, uses] : edges) {
      if (uses.size() == 1) {
        declared.boundary_edges.push_back(key);
      } else if (uses.size() != 2 || uses[0][0] != uses[1][1] || uses[0][1] != uses[1][0]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  return PHX_MC_OK;
}

bool TetMesh::declared_face_valid(const DeclaredGroup& group, const double* const* carriers,
                                  const SourceWitness* const* witness_records,
                                  std::int8_t orientation) const {
  // Every corner has an admitted record on the plane; the exact positions,
  // projected onto the plane, form a triangle of the facet's orientation that
  // the union of the group's rows covers: no edge of the union's boundary
  // meets its open interior, and its centroid lies in a row.
  const std::int32_t plane_row = group.rows.front();
  const double* plane = face_plane(plane_row);
  if (plane == nullptr) return false;
  Expansion corners[3][3];
  for (int corner = 0; corner < 3; ++corner) {
    if (!plane_admits(plane_row, carriers[corner]) ||
        !witness_position(*witness_records[corner], carriers[corner], corners[corner])) {
      return false;
    }
  }
  Expansion normal[3];
  plane_normal(plane, normal);
  const int side = projected_orientation(normal, corners[0], corners[1], corners[2]);
  if (side == 0 || side != orientation * group.orientation) return false;
  const Expansion* triangle[3] = {corners[0], side > 0 ? corners[1] : corners[2],
                                  side > 0 ? corners[2] : corners[1]};
  Expansion ends[2][3];
  for (const std::array<std::int32_t, 2>& edge : group.boundary_edges) {
    for (int end = 0; end < 2; ++end) {
      const double* source = source_points_.data() + 3 * static_cast<std::size_t>(edge[static_cast<std::size_t>(end)]);
      for (int axis = 0; axis < 3; ++axis) ends[end][axis] = Expansion(source[axis]);
    }
    if (segment_meets_open_triangle(normal, ends[0], ends[1], triangle)) return false;
  }
  // Three times the centroid, against each row's edges (p, q):
  // n.((q - p) x (3 centroid - 3 p)) has the sign of the centroid's side.
  Expansion sum[3];
  for (int axis = 0; axis < 3; ++axis) {
    sum[axis] = corners[0][axis] + corners[1][axis] + corners[2][axis];
  }
  for (std::int32_t row : group.rows) {
    const double* points[3];
    for (int corner = 0; corner < 3; ++corner) {
      points[corner] = source_points_.data() +
                       3 * static_cast<std::size_t>(source_faces_[3 * static_cast<std::size_t>(row) +
                                                                  static_cast<std::size_t>(corner)]);
    }
    if (group.orientation < 0) std::swap(points[1], points[2]);
    bool inside = true;
    for (int corner = 0; corner < 3 && inside; ++corner) {
      const double* p = points[corner];
      const double* q = points[(corner + 1) % 3];
      Expansion along[3];
      Expansion toward[3];
      for (int axis = 0; axis < 3; ++axis) {
        along[axis] = Expansion::difference(q[axis], p[axis]);
        toward[axis] = sum[axis] - Expansion(p[axis]).scaled(3.0);
      }
      inside = triple_sign(normal, along, toward) >= 0;
    }
    if (inside) return true;
  }
  return false;
}

bool TetMesh::source_face_valid(const TetEditFacet& facet,
                                const RelocationOverlay* overlay) const {
  if (facet.source < 0 || (facet.orientation != 1 && facet.orientation != -1)) return false;
  std::array<std::int32_t, 3> slots{};
  for (int k = 0; k < 3; ++k) {
    slots[k] = vertex_slot(facet.vertices[k]);
    if (!live_vertex(slots[k])) return false;
    for (int j = 0; j < k; ++j) if (slots[k] == slots[j]) return false;
  }
  const auto overlay_vertex = [&](std::int32_t slot) {
    return overlay == nullptr ? nullptr : overlay->find(vertex_id(slot));
  };
  const auto carrier_at = [&](std::int32_t slot) -> const double* {
    const TetEditVertex* changed = overlay_vertex(slot);
    return changed == nullptr ? point(slot) : changed->carrier.data();
  };
  const auto witness_at = [&](std::int32_t slot) -> const SourceWitness& {
    const TetEditVertex* changed = overlay_vertex(slot);
    return changed == nullptr
               ? witnesses_[static_cast<std::size_t>(slot)]
               : changed->witness;
  };
  if (const auto declared = declared_groups_.find(facet.source);
      declared != declared_groups_.end()) {
    const double* carriers[3];
    const SourceWitness* witness_records[3];
    for (int corner = 0; corner < 3; ++corner) {
      carriers[corner] = carrier_at(slots[corner]);
      witness_records[corner] = &witness_at(slots[corner]);
      if (!source_carrier_valid(source_witness_record(vertex_id(slots[corner]), carriers[corner],
                                                      *witness_records[corner]))) {
        return false;
      }
    }
    return declared_face_valid(declared->second, carriers, witness_records, facet.orientation);
  }

  std::vector<std::array<std::int32_t, 4>> primitives;
  std::vector<std::int32_t> rows;
  primitives.reserve(source_faces_.size() / 3);
  rows.reserve(source_faces_.size() / 3);
  for (std::size_t row = 0; row < source_faces_.size() / 3; ++row) {
    if (source_face_groups_[row] != facet.source) continue;
    // Coverage uses only the first three fields; retain the exact bank row in slot four.
    primitives.push_back({source_faces_[3 * row], source_faces_[3 * row + 1],
                          source_faces_[3 * row + 2], static_cast<std::int32_t>(row)});
    rows.push_back(static_cast<std::int32_t>(primitives.size() - 1));
  }
  if (rows.empty()) return false;

  double subject[9];
  bool exact_carriers = true;
  for (int vertex = 0; vertex < 3; ++vertex) {
    const std::int32_t slot = slots[vertex];
    const double* carrier = carrier_at(slot);
    std::copy_n(carrier, 3, subject + 3 * vertex);
    const SourceWitness& witness = witness_at(slot);
    if (!source_carrier_valid(source_witness_record(vertex_id(slot), carrier, witness))) return false;
    exact_carriers = exact_carriers && witness.deviation == 0.0;
  }
  std::vector<std::int32_t> contributors;
  if (exact_carriers && exact_facet_coverage(subject, source_points_.data(), primitives,
                                              rows, contributors) && !contributors.empty()) {
    // Coverage returns local primitive rows; slot four maps each to the source bank.
    // Unrelated triangles in one authored group can have different normals.
    for (std::int32_t contributor : contributors) {
      if (contributor < 0 || static_cast<std::size_t>(contributor) >= primitives.size()) {
        return false;
      }
      const std::int32_t source_row = primitives[static_cast<std::size_t>(contributor)][3];
      if (source_row < 0 || static_cast<std::size_t>(source_row) >= source_faces_.size() / 3) {
        return false;
      }
      const std::size_t row = static_cast<std::size_t>(source_row);
      const double* primitive[3] = {
          source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row]),
          source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + 1]),
          source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + 2])};
      bool oriented = false;
      for (int axis = 0; axis < 3; ++axis) {
        const int first = (axis + 1) % 3;
        const int second = (axis + 2) % 3;
        const double source_a[2] = {primitive[0][first], primitive[0][second]};
        const double source_b[2] = {primitive[1][first], primitive[1][second]};
        const double source_c[2] = {primitive[2][first], primitive[2][second]};
        const int source_orientation = orient2d(source_a, source_b, source_c);
        if (source_orientation == 0) continue;
        const double carrier_a[2] = {subject[first], subject[second]};
        const double carrier_b[2] = {subject[3 + first], subject[3 + second]};
        const double carrier_c[2] = {subject[6 + first], subject[6 + second]};
        const int carrier_orientation = orient2d(carrier_a, carrier_b, carrier_c);
        if (carrier_orientation == 0 ||
            carrier_orientation != source_orientation * facet.orientation) {
          return false;
        }
        oriented = true;
        break;
      }
      if (!oriented) return false;
    }
    return true;
  }

  // Exact source positions, not rounded carriers, authorize an off-plane
  // binary64 realization. A common convex primitive contains the complete
  // source triangle; multi-primitive coverage was proved above. A carrier that
  // is only the realization rounding of its exact source point approximates
  // nothing, so only an authored deviation is held to the row tolerance.
  Expansion exact_points[3][3];
  double point_deviations[3]{};
  for (int vertex = 0; vertex < 3; ++vertex) {
    const std::int32_t slot = slots[vertex];
    const SourceWitness& witness = witness_at(slot);
    const double* carrier = carrier_at(slot);
    if (witness.stratum != SourceStratum::kNone &&
        !source_carrier_valid(source_witness_record(vertex_id(slot), carrier, witness))) {
      return false;
    }
    if (!witness_position(witness, carrier, exact_points[vertex])) return false;
    if (witness.deviation != 0.0 &&
        witness.deviation != outward_distance(carrier, exact_points[vertex])) {
      point_deviations[vertex] = witness.deviation;
    }
  }

  for (std::size_t row = 0; row < source_faces_.size() / 3; ++row) {
    if (source_face_groups_[row] != facet.source) continue;
    if (point_deviations[0] > source_face_tolerances_[row] ||
        point_deviations[1] > source_face_tolerances_[row] ||
        point_deviations[2] > source_face_tolerances_[row]) continue;
    const double* primitive[3] = {
        source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row]),
        source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + 1]),
        source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + 2])};
    // Projected containment is valid only after exact 3D plane incidence.
    std::array<Expansion, 3> first_edge{};
    std::array<Expansion, 3> second_edge{};
    std::array<Expansion, 3> normal{};
    for (int axis = 0; axis < 3; ++axis) {
      first_edge[static_cast<std::size_t>(axis)] =
          Expansion::difference(primitive[1][axis], primitive[0][axis]);
      second_edge[static_cast<std::size_t>(axis)] =
          Expansion::difference(primitive[2][axis], primitive[0][axis]);
    }
    normal[0] = first_edge[1] * second_edge[2] -
                first_edge[2] * second_edge[1];
    normal[1] = first_edge[2] * second_edge[0] -
                first_edge[0] * second_edge[2];
    normal[2] = first_edge[0] * second_edge[1] -
                first_edge[1] * second_edge[0];
    bool exact_coplanar = true;
    for (int vertex = 0; vertex < 3 && exact_coplanar; ++vertex) {
      std::array<Expansion, 3> offset{};
      for (int axis = 0; axis < 3; ++axis) {
        offset[static_cast<std::size_t>(axis)] =
            exact_points[vertex][axis] - Expansion(primitive[0][axis]);
      }
      const Expansion distance =
          normal[0] * offset[0] + normal[1] * offset[1] +
          normal[2] * offset[2];
      exact_coplanar = distance.is_zero();
    }
    if (!exact_coplanar) continue;
    for (int axis = 0; axis < 3; ++axis) {
      const int first = (axis + 1) % 3;
      const int second = (axis + 2) % 3;
      const double source_a[2] = {primitive[0][first], primitive[0][second]};
      const double source_b[2] = {primitive[1][first], primitive[1][second]};
      const double source_c[2] = {primitive[2][first], primitive[2][second]};
      const int source_orientation = orient2d(source_a, source_b, source_c);
      if (source_orientation == 0) continue;
      Expansion source_projection[3][2];
      for (int corner = 0; corner < 3; ++corner) {
        source_projection[corner][0] = Expansion(primitive[corner][first]);
        source_projection[corner][1] = Expansion(primitive[corner][second]);
      }
      Expansion projected[3][2];
      for (int vertex = 0; vertex < 3; ++vertex) {
        projected[vertex][0] = exact_points[vertex][first];
        projected[vertex][1] = exact_points[vertex][second];
      }
      const auto orientation = [](const Expansion* a, const Expansion* b,
                                  const Expansion* c) {
        return ((b[0] - a[0]) * (c[1] - a[1]) -
                (b[1] - a[1]) * (c[0] - a[0])).sign();
      };
      bool inside = true;
      for (int vertex = 0; vertex < 3; ++vertex) {
        const int s0 = orientation(source_projection[0], source_projection[1],
                                   projected[vertex]);
        const int s1 = orientation(source_projection[1], source_projection[2],
                                   projected[vertex]);
        const int s2 = orientation(source_projection[2], source_projection[0],
                                   projected[vertex]);
        inside = inside && (source_orientation > 0 ? s0 >= 0 && s1 >= 0 && s2 >= 0
                                                   : s0 <= 0 && s1 <= 0 && s2 <= 0);
      }
      if (!inside) break;
      const int subject_orientation = orientation(projected[0], projected[1], projected[2]);
      if (subject_orientation != 0 &&
          subject_orientation == source_orientation * facet.orientation) {
        return true;
      }
      break;
    }
  }
  return false;
}

bool TetMesh::source_segment_valid(const TetEditSegment& segment,
                                   const RelocationOverlay* overlay) const {
  if (segment.source < 0 || segment.vertices[0] == segment.vertices[1]) return false;
  const std::int32_t first = vertex_slot(segment.vertices[0]);
  const std::int32_t second = vertex_slot(segment.vertices[1]);
  if (!live_vertex(first) || !live_vertex(second)) return false;
  const auto overlay_vertex = [&](std::int32_t slot) {
    return overlay == nullptr ? nullptr : overlay->find(vertex_id(slot));
  };
  const auto carrier_at = [&](std::int32_t slot) -> const double* {
    const TetEditVertex* changed = overlay_vertex(slot);
    return changed == nullptr ? point(slot) : changed->carrier.data();
  };
  const auto witness_at = [&](std::int32_t slot) -> const SourceWitness& {
    const TetEditVertex* changed = overlay_vertex(slot);
    return changed == nullptr
               ? witnesses_[static_cast<std::size_t>(slot)]
               : changed->witness;
  };
  const SourceWitness& first_witness = witness_at(first);
  const SourceWitness& second_witness = witness_at(second);
  for (std::size_t row = 0; row < source_segment_groups_.size(); ++row) {
    if (source_segment_groups_[row] != segment.source) continue;
    const double* corners[3] = {
        source_points_.data() + 3 * static_cast<std::size_t>(source_segments_[2 * row]),
        source_points_.data() + 3 * static_cast<std::size_t>(source_segments_[2 * row + 1]),
        nullptr};
    const auto parameter_for = [&](std::int32_t slot, const SourceWitness& witness,
                                   double& parameter) {
      if (witness.stratum == SourceStratum::kSegment &&
          witness.entity == static_cast<std::int32_t>(row)) {
        if (!source_carrier_valid(
                source_witness_record(vertex_id(slot), carrier_at(slot), witness))) return false;
        parameter = witness.parameters[0];
        return true;
      }
      if (witness.stratum != SourceStratum::kNone || witness.entity != -1 ||
          witness.deviation != 0.0) return false;
      double parameters[2]{};
      const double* current_carrier = carrier_at(slot);
      source_locator(corners, SourceStratum::kSegment, current_carrier,
                     parameters, false);
      double carrier[3];
      double deviation = 0.0;
      if (!source_carrier(corners, SourceStratum::kSegment, parameters, carrier, deviation) ||
          deviation != 0.0 || !same_bits(carrier[0], current_carrier[0]) ||
          !same_bits(carrier[1], current_carrier[1]) ||
          !same_bits(carrier[2], current_carrier[2])) {
        return false;
      }
      parameter = parameters[0];
      return true;
    };
    double first_parameter = 0.0;
    double second_parameter = 0.0;
    if (parameter_for(first, first_witness, first_parameter) &&
        parameter_for(second, second_witness, second_parameter) &&
        first_parameter != second_parameter) {
      return true;
    }
  }
  return false;
}

template <bool WithRelocation>
bool TetMesh::positive_impl(
    const std::int32_t* vertices, const RelocationOverlay* overlay) const {
  if (vertices == nullptr) return false;
  const double* carriers[4];
  const SourceWitness* witness_records[4];
  for (int vertex = 0; vertex < 4; ++vertex) {
    const std::int32_t slot = vertices[vertex];
    if (!live_vertex(slot)) return false;
    const TetEditVertex* changed = nullptr;
    if constexpr (WithRelocation) {
      if (overlay == nullptr) return false;
      changed = overlay->find(vertex_id(slot));
      carriers[vertex] = changed == nullptr ? point(slot) : changed->carrier.data();
    } else {
      carriers[vertex] = point(slot);
    }
    witness_records[vertex] =
        &witnesses_[static_cast<std::size_t>(slot)];
    if constexpr (WithRelocation) {
      if (changed != nullptr) witness_records[vertex] = &changed->witness;
    }
  }
  return carriers_positive(carriers, witness_records);
}

bool TetMesh::carriers_positive(const double* const* carriers,
                                const SourceWitness* const* witness_records) const {
  // orient3d uses static error filters and exact fallback. When every witness
  // is kNone, its binary64 carriers are also the exact source points.
  if (orient3d(carriers[0], carriers[1], carriers[2], carriers[3]) <= 0) return false;
  bool all_carrier_points = true;
  for (int vertex = 0; vertex < 4; ++vertex) {
    all_carrier_points =
        all_carrier_points && witness_records[vertex]->stratum == SourceStratum::kNone;
  }
  if (all_carrier_points) return true;

  Expansion exact_storage[4][3];
  const Expansion* exact_points[4][3];
  for (int vertex = 0; vertex < 4; ++vertex) {
    if (!witness_position(*witness_records[vertex], carriers[vertex], exact_storage[vertex])) {
      return false;
    }
    for (int axis = 0; axis < 3; ++axis) exact_points[vertex][axis] = &exact_storage[vertex][axis];
  }
  return exact_orientation(exact_points) > 0;
}

bool TetMesh::cone_positive(const std::int32_t* vertices, int apex,
                            const TetEditVertex& point) const {
  const double* carriers[4];
  const SourceWitness* witness_records[4];
  for (int vertex = 0; vertex < 4; ++vertex) {
    if (vertex == apex) {
      carriers[vertex] = point.carrier.data();
      witness_records[vertex] = &point.witness;
      continue;
    }
    const std::int32_t slot = vertices[vertex];
    if (!live_vertex(slot)) return false;
    carriers[vertex] = this->point(slot);
    witness_records[vertex] = &witnesses_[static_cast<std::size_t>(slot)];
  }
  return carriers_positive(carriers, witness_records);
}

bool TetMesh::positive(const std::int32_t* vertices) const {
  return positive_impl<false>(vertices, nullptr);
}

bool TetMesh::positive_with_relocation(
    const std::int32_t* vertices, const RelocationOverlay& overlay) const {
  return positive_impl<true>(vertices, &overlay);
}

}  // namespace phx::mc
