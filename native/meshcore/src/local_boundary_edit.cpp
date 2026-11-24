//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Star-local boundary edits through LocalIncidence: the collapse of a facet
// or segment vertex into a vertex on its carrier, and the swap of a boundary
// edge between two facets of one source group. Each is decided on its cavity
// (the removed cells, the hull ghosts across their hull faces and the cells
// across their other faces) in TetMesh::apply_edit_batch's refusal order, so
// verdict and committed state equal the batch's for the same single
// operation, without a state copy or a scan of the mesh.
#include "local_incidence.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cavity.hpp"
#include "dyadic.hpp"
#include "exact_scalar.hpp"
#include "tet_mesh.hpp"

namespace phx::mc::detail {
namespace {

using FaceIds = std::array<std::uint64_t, 3>;
using EdgeIds = std::array<std::uint64_t, 2>;
using SlotFace = std::array<std::int32_t, 3>;
using SlotTet = std::array<std::int32_t, 4>;

constexpr std::uint64_t kMaxId = std::numeric_limits<std::uint64_t>::max();

FaceIds sorted_ids(FaceIds face) {
  std::sort(face.begin(), face.end());
  return face;
}

EdgeIds sorted_ids(std::uint64_t first, std::uint64_t second) {
  return {std::min(first, second), std::max(first, second)};
}

SlotFace sorted_slots(const std::int32_t* face) {
  SlotFace key{face[0], face[1], face[2]};
  std::sort(key.begin(), key.end());
  return key;
}

std::uint64_t slot_edge(std::int32_t first, std::int32_t second) {
  const auto low = static_cast<std::uint32_t>(std::min(first, second));
  const auto high = static_cast<std::uint32_t>(std::max(first, second));
  return (static_cast<std::uint64_t>(low) << 32) | high;
}

bool periodic_operation(const TetEdit& operation) {
  return !operation.periodic_members.empty() || !operation.removed_periodic_orbit_ids.empty() ||
         !operation.replacement_periodic_orbits.empty() ||
         !operation.removed_quotient_cell_ids.empty() ||
         !operation.replacement_quotient_cells.empty() ||
         !operation.replacement_periodic_entity_maps.empty();
}

// The exact determinant of the carriers (TetMesh::apply_edit_batch's volume
// check of a volume-preserving edit).
Dyadic carrier_determinant(const TetMesh& mesh, const std::int32_t* vertices) {
  const double* origin = mesh.point(vertices[0]);
  std::array<std::array<Dyadic, 3>, 3> rows{};
  for (int row = 0; row < 3; ++row) {
    const double* corner = mesh.point(vertices[row + 1]);
    for (int axis = 0; axis < 3; ++axis) {
      rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(axis)] =
          Dyadic(corner[axis]) - Dyadic(origin[axis]);
    }
  }
  const auto& a = rows[0];
  const auto& b = rows[1];
  const auto& c = rows[2];
  return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
         a[2] * (b[0] * c[1] - b[1] * c[0]);
}

// The exact normal of a triangle of carriers.
std::array<Dyadic, 3> carrier_normal(const TetMesh& mesh, const std::array<std::int32_t, 3>& face) {
  const double* p = mesh.point(face[0]);
  const double* q = mesh.point(face[1]);
  const double* r = mesh.point(face[2]);
  std::array<Dyadic, 3> u{};
  std::array<Dyadic, 3> v{};
  for (int axis = 0; axis < 3; ++axis) {
    u[static_cast<std::size_t>(axis)] = Dyadic(q[axis]) - Dyadic(p[axis]);
    v[static_cast<std::size_t>(axis)] = Dyadic(r[axis]) - Dyadic(p[axis]);
  }
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
}

// The link condition of TetMesh::apply_edit_batch (full_link_condition) on
// `cells`, every live cell (hull ghosts included) holding r or s.
bool link_condition(const TetMesh& mesh, std::span<const std::int32_t> cells, std::int32_t r,
                    std::int32_t s) {
  std::set<SlotFace> r_faces, s_faces;
  std::set<std::uint64_t> r_edges, s_edges, edge_edges;
  std::set<std::int32_t> r_vertices, s_vertices, edge_vertices;
  const auto add_link = [](const Tetrahedron& tet, std::int32_t vertex,
                           std::set<std::int32_t>& vertices, std::set<std::uint64_t>& edges,
                           std::set<SlotFace>& faces) {
    std::array<std::int32_t, 3> rest{};
    std::size_t count = 0;
    for (std::int32_t other : tet.v) {
      if (other != vertex) rest[count++] = other;
    }
    for (std::size_t i = 0; i < count; ++i) {
      vertices.insert(rest[i]);
      for (std::size_t j = i + 1; j < count; ++j) edges.insert(slot_edge(rest[i], rest[j]));
    }
    if (count == 3) faces.insert(sorted_slots(rest.data()));
  };
  for (std::int32_t cell : cells) {
    const Tetrahedron& tet = mesh.tet(cell);
    const bool has_r = vertex_slot(tet, r) >= 0;
    const bool has_s = vertex_slot(tet, s) >= 0;
    if (has_r) add_link(tet, r, r_vertices, r_edges, r_faces);
    if (has_s) add_link(tet, s, s_vertices, s_edges, s_faces);
    if (has_r && has_s) {
      std::array<std::int32_t, 2> rest{};
      std::size_t count = 0;
      for (std::int32_t other : tet.v) {
        if (other != r && other != s) rest[count++] = other;
      }
      if (count != 2) return false;
      edge_vertices.insert(rest[0]);
      edge_vertices.insert(rest[1]);
      edge_edges.insert(slot_edge(rest[0], rest[1]));
    }
  }
  std::set<std::int32_t> common_vertices;
  std::set_intersection(r_vertices.begin(), r_vertices.end(), s_vertices.begin(),
                        s_vertices.end(),
                        std::inserter(common_vertices, common_vertices.end()));
  std::set<std::uint64_t> common_edges;
  std::set_intersection(r_edges.begin(), r_edges.end(), s_edges.begin(), s_edges.end(),
                        std::inserter(common_edges, common_edges.end()));
  bool common_face = false;
  for (const SlotFace& face : r_faces) common_face = common_face || s_faces.contains(face);
  return common_vertices == edge_vertices && common_edges == edge_edges && !common_face;
}

bool source_stencil_image(const SourceStencil& stencil, std::span<const double> source_points,
                          std::array<Dyadic, 3>& combination) {
  const std::size_t point_count = source_points.size() / 3;
  if (stencil.points.empty() || stencil.points.size() != stencil.weights.size() ||
      stencil.points.size() > point_count) {
    return false;
  }
  auto rows = stencil.points;
  std::vector<std::int32_t> sorted;
  if (!std::is_sorted(rows.begin(), rows.end())) {
    sorted.assign(rows.begin(), rows.end());
    std::sort(sorted.begin(), sorted.end());
    rows = sorted;
  }
  if (rows.front() < 0 || static_cast<std::size_t>(rows.back()) >= point_count ||
      std::adjacent_find(rows.begin(), rows.end()) != rows.end()) {
    return false;
  }
  Dyadic weight_sum;
  for (std::size_t k = 0; k < stencil.points.size(); ++k) {
    const double weight = stencil.weights[k];
    if (!std::isfinite(weight)) return false;
    if (weight < 0.0 && stencil.image == SourceStencilImage::kRoundedPlane) return false;
    if (weight == 0.0) continue;
    const Dyadic exact_weight(weight);
    weight_sum = weight_sum + exact_weight;
    const double* p = source_points.data() + 3 * static_cast<std::size_t>(stencil.points[k]);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      combination[axis] = combination[axis] + Dyadic(p[axis]) * exact_weight;
    }
  }
  return weight_sum.is_positive_one();
}

bool source_stencil_matches(const SourceStencil& stencil, std::span<const double> source_points,
                            const double* carrier) {
  if (stencil.image != SourceStencilImage::kExact) return false;
  std::array<Dyadic, 3> combination{};
  if (!source_stencil_image(stencil, source_points, combination)) return false;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(carrier[axis]) || !(combination[axis] - Dyadic(carrier[axis])).is_zero()) {
      return false;
    }
  }
  return true;
}

bool source_stencil_supported(const SourceStencil& stencil, std::span<const std::int32_t> entity) {
  for (std::size_t k = 0; k < stencil.points.size(); ++k) {
    if (stencil.weights[k] != 0.0 &&
        !std::binary_search(entity.begin(), entity.end(), stencil.points[k])) {
      return false;
    }
  }
  return true;
}

// Closed boxes conservatively screen both points: a binary64 bound that
// contains the exact image also contains its correctly rounded image.
struct PlanarSourcePoint {
  std::span<const double> points;
  std::array<int, 2> axes;
  std::array<Dyadic, 2> exact;
  std::array<Dyadic, 2> rounded;
  std::array<double, 2> carrier;

  const double* point(std::int32_t row) const {
    return points.data() + 3 * static_cast<std::size_t>(row);
  }

  template <std::size_t N>
  bool in_box(std::span<const std::int32_t, N> rows) const {
    for (std::size_t axis = 0; axis < 2; ++axis) {
      double low = point(rows.front())[axes[axis]];
      double high = low;
      for (std::int32_t row : rows.subspan(1)) {
        const double value = point(row)[axes[axis]];
        low = std::min(low, value);
        high = std::max(high, value);
      }
      if (carrier[axis] < low || carrier[axis] > high) return false;
    }
    return true;
  }

  int side(std::int32_t a, std::int32_t b, const std::array<Dyadic, 2>& p) const {
    const Dyadic x(point(a)[axes[0]]), y(point(a)[axes[1]]);
    return ((Dyadic(point(b)[axes[0]]) - x) * (p[1] - y) -
            (Dyadic(point(b)[axes[1]]) - y) * (p[0] - x)).sign();
  }

  bool in_triangle(std::span<const std::int32_t, 3> rows, const std::array<Dyadic, 2>& p,
                   int winding) const {
    for (std::size_t corner = 0; corner < 3; ++corner) {
      if (side(rows[corner], rows[(corner + 1) % 3], p) * winding < 0) return false;
    }
    return true;
  }
};

}  // namespace

// A boundary edit resolved against the index: the cavity and the rows of the
// banks it retires, with the contract findings apply_edit_batch reports at
// its region and constraint check.
struct LocalIncidence::BoundaryEdit {
  // Finite removed cell slots, sorted.
  std::vector<std::int32_t> removed;
  std::int32_t removed_vertex = -1;
  const TetMesh::RelocationOverlay* overlay = nullptr;
  std::int32_t moved_vertex = -1;
  // Bank rows of operation.removed_facet_ids and removed_segment_ids, in order.
  std::vector<std::size_t> facet_rows;
  std::vector<std::size_t> segment_rows;
  // A replacement facet or segment is not the contract's image, or its image
  // is already constrained (kRejectedTopology at the region check).
  bool image_refused = false;
  // A swap facet whose exact normal is orthogonal to the old facets'.
  bool degenerate_image = false;
  // The edit preserves volume (a ring reconnection): exact volume check.
  bool volume_preserving = false;
  // A split: the edge's ends and the inserted vertex (staged), whose
  // retired hull faces must be covered by their halves.
  std::int32_t split_first = -1;
  std::int32_t split_second = -1;
  std::int32_t inserted = -1;
};

bool LocalIncidence::current_star(const Star& star, std::int32_t vertex) const noexcept {
  return star.owner_ == this && star.vertex_ == vertex && star.revision_ == topology_revision_ &&
         vertex >= 0 && static_cast<std::size_t>(vertex) < vertex_degree_.size() &&
         static_cast<std::size_t>(vertex_degree_[static_cast<std::size_t>(vertex)]) ==
             star.cells_.size();
}

bool LocalIncidence::ensure_boundary_index(const TetMesh& mesh) {
  if (boundary_index_ready_) return true;
  BoundaryIndex index;
  index.facets.reserve(mesh.facets_.size());
  for (std::size_t row = 0; row < mesh.facets_.size(); ++row) {
    if (!index.facets.emplace(sorted_ids(mesh.facets_[row].vertices), row).second) return false;
  }
  index.segments.reserve(mesh.segments_.size());
  for (std::size_t row = 0; row < mesh.segments_.size(); ++row) {
    const auto& ends = mesh.segments_[row].vertices;
    if (!index.segments.emplace(sorted_ids(ends[0], ends[1]), row).second) return false;
  }
  // apply_edit_batch's collect_source_regions over the whole complex.
  std::map<std::pair<std::int32_t, int>, std::int32_t> source_regions;
  const TetrahedralComplex& complex = mesh.complex_;
  for (std::size_t row = 0; row < complex.tets.size() && index.regions_consistent; ++row) {
    const auto cell = static_cast<std::int32_t>(row);
    if (!complex.live(cell) || is_ghost(complex.tets[row])) continue;
    for (int face = 0; face < 4; ++face) {
      const std::int32_t source = complex.constraint(cell, face);
      if (source < 0) continue;
      std::int32_t corners[3];
      oriented_facet(complex.tets[row].v, face, corners);
      const auto found = index.facets.find(sorted_ids(
          {mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]), mesh.vertex_id(corners[2])}));
      if (found == index.facets.end()) continue;
      const TetSnapshotFacet& facet = mesh.facets_[found->second];
      if (facet.source != source) continue;
      std::int32_t slots[3];
      for (int corner = 0; corner < 3; ++corner) {
        slots[corner] = mesh.vertex_slot(facet.vertices[static_cast<std::size_t>(corner)]);
      }
      const int direction = FacetKey::of(corners) == FacetKey::of(slots) ? 1 : -1;
      const auto [entry, inserted] = source_regions.emplace(
          std::pair{source, direction * facet.orientation}, complex.region(cell));
      index.regions_consistent = inserted || entry->second == complex.region(cell);
      if (!index.regions_consistent) break;
    }
  }
  for (std::size_t row = 0; row < mesh.source_face_groups_.size(); ++row) {
    auto& points = index.face_group_points[mesh.source_face_groups_[row]];
    points.insert(points.end(), mesh.source_faces_.begin() + static_cast<std::ptrdiff_t>(3 * row),
                  mesh.source_faces_.begin() + static_cast<std::ptrdiff_t>(3 * row + 3));
  }
  for (std::size_t row = 0; row < mesh.source_segment_groups_.size(); ++row) {
    auto& points = index.segment_group_points[mesh.source_segment_groups_[row]];
    points.insert(points.end(),
                  mesh.source_segments_.begin() + static_cast<std::ptrdiff_t>(2 * row),
                  mesh.source_segments_.begin() + static_cast<std::ptrdiff_t>(2 * row + 2));
  }
  for (auto* groups : {&index.face_group_points, &index.segment_group_points}) {
    for (auto& [group, points] : *groups) {
      (void)group;
      std::sort(points.begin(), points.end());
      points.erase(std::unique(points.begin(), points.end()), points.end());
    }
  }
  boundary_index_ = std::move(index);
  boundary_index_ready_ = true;
  return true;
}

TetEditStatus LocalIncidence::boundary_facet(std::array<std::uint64_t, 3> vertices,
                                             const TetSnapshotFacet*& record) {
  record = nullptr;
  if (mesh_ == nullptr || !ensure_boundary_index(*mesh_)) return TetEditStatus::kStaleGeneration;
  const auto key = sorted_ids(vertices);
  const auto found = boundary_index_.facets.find(key);
  if (found == boundary_index_.facets.end()) return TetEditStatus::kApplied;
  if (found->second >= mesh_->facets_.size()) return TetEditStatus::kStaleGeneration;
  const TetSnapshotFacet& facet = mesh_->facets_[found->second];
  if (sorted_ids(facet.vertices) != key) return TetEditStatus::kStaleGeneration;
  record = &facet;
  return TetEditStatus::kApplied;
}

TetEditStatus LocalIncidence::boundary_segment(std::array<std::uint64_t, 2> vertices,
                                               const TetSnapshotSegment*& record) {
  record = nullptr;
  if (mesh_ == nullptr || !ensure_boundary_index(*mesh_)) return TetEditStatus::kStaleGeneration;
  const auto key = sorted_ids(vertices[0], vertices[1]);
  const auto found = boundary_index_.segments.find(key);
  if (found == boundary_index_.segments.end()) return TetEditStatus::kApplied;
  if (found->second >= mesh_->segments_.size()) return TetEditStatus::kStaleGeneration;
  const TetSnapshotSegment& segment = mesh_->segments_[found->second];
  if (sorted_ids(segment.vertices[0], segment.vertices[1]) != key) {
    return TetEditStatus::kStaleGeneration;
  }
  record = &segment;
  return TetEditStatus::kApplied;
}

bool LocalIncidence::rounded_plane_support(const TetMesh& mesh, std::int32_t group,
                                           const SourceStencil& stencil,
                                           const double* carrier) const {
  const auto declared = mesh.declared_groups_.find(group);
  if (declared == mesh.declared_groups_.end()) return false;
  const auto& plane_group = declared->second;
  const auto row = plane_group.rows.front();
  const double* plane = mesh.face_plane(row);
  if (plane == nullptr || mesh.source_face_plane_tolerances_[static_cast<std::size_t>(row)] != 0.0) {
    return false;
  }
  int normal = 0;
  while (normal < 3 && (plane[3 + normal] != 0.0 || plane[6 + normal] != 0.0)) ++normal;
  if (normal == 3 || carrier[normal] != plane[normal]) return false;
  const auto& group_points = boundary_index_.face_group_points.at(group);
  if (!std::ranges::all_of(group_points, [&](std::int32_t point) {
        return mesh.source_points_[3 * static_cast<std::size_t>(point) + normal] == plane[normal];
      })) {
    return false;
  }
  std::array<Dyadic, 3> image{};
  if (!source_stencil_image(stencil, mesh.source_points_, image) ||
      !source_stencil_supported(stencil, group_points)) return false;
  const Dyadic unit(1.0);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double rounded = exact::rounded_quotient(image[axis], unit);
    if (std::bit_cast<std::uint64_t>(rounded) != std::bit_cast<std::uint64_t>(carrier[axis])) {
      return false;
    }
  }
  const int u = (normal + 1) % 3;
  const int v = (normal + 2) % 3;
  const PlanarSourcePoint projected{
      mesh.source_points_, {u, v}, {std::move(image[u]), std::move(image[v])},
      {Dyadic(carrier[u]), Dyadic(carrier[v])}, {carrier[u], carrier[v]}};
  // Every point of the exact-to-binary64 displacement rounds numerically
  // to the same carrier. If it meets a box with binary64 endpoints, the
  // carrier is in the box. Reject any possible rim or source-segment contact,
  // including an authored segment internal to this face group.
  for (const auto& edge : plane_group.boundary_edges) {
    if (projected.in_box(std::span(edge))) return false;
  }
  for (std::size_t row = 0; row < mesh.source_segments_.size(); row += 2) {
    const double* a = projected.point(mesh.source_segments_[row]);
    const double* b = projected.point(mesh.source_segments_[row + 1]);
    bool overlaps = true;
    for (std::size_t axis = 0; axis < 3 && overlaps; ++axis) {
      const auto [low, high] = std::minmax(a[axis], b[axis]);
      overlaps = carrier[axis] >= low && carrier[axis] <= high;
    }
    if (overlaps) return false;
  }
  bool exact_inside = false;
  bool rounded_inside = false;
  for (std::int32_t face : plane_group.rows) {
    const std::span<const std::int32_t, 3> rows(
        mesh.source_faces_.data() + 3 * static_cast<std::size_t>(face), 3);
    if (!projected.in_box(rows)) continue;
    const double* c = projected.point(rows[2]);
    const int winding = projected.side(rows[0], rows[1], {Dyadic(c[u]), Dyadic(c[v])});
    if (winding == 0) continue;
    exact_inside = exact_inside || projected.in_triangle(rows, projected.exact, winding);
    rounded_inside = rounded_inside || projected.in_triangle(rows, projected.rounded, winding);
    if (exact_inside && rounded_inside) return true;
  }
  return false;
}

std::optional<TetEditStatus> LocalIncidence::collapse_boundary(
    TetMesh& mesh, const TetEdit& operation, const Star& removed_star,
    const Star* survivor_star, const SourceStencil* survivor_stencil) {
  mesh.clear_implicit_facet_refusal();
  try {
    if (mesh_ != &mesh || operation.kind != TetEditKind::kCollapse ||
        mesh.periodic_identity_.bound || !mesh.periodic_orbits_.empty() ||
        !mesh.quotient_cells_.empty() || mesh.implicit_source_ ||
        operation.collapse_pairs.size() != 1 || operation.removed_vertex_ids.size() != 1 ||
        operation.collapse_pairs.front().removed_vertex_id !=
            operation.removed_vertex_ids.front() ||
        !operation.split_edges.empty() || operation.changed_vertices.size() > 1 ||
        periodic_operation(operation)) {
      return TetEditStatus::kRejectedTopology;
    }
    const std::uint64_t r_id = operation.removed_vertex_ids.front();
    const std::uint64_t s_id = operation.collapse_pairs.front().survivor_vertex_id;
    const std::int32_t r = mesh.vertex_slot(r_id);
    const std::int32_t s = mesh.vertex_slot(s_id);
    const bool moved = !operation.changed_vertices.empty();
    if (!mesh.live_vertex(r) || !mesh.live_vertex(s) || r == s ||
        (moved && operation.changed_vertices.front().id != s_id) ||
        moved != (survivor_star != nullptr) || !current_star(removed_star, r) ||
        (moved && !current_star(*survivor_star, s)) || removed_star.empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    const std::int8_t r_dimension = mesh.vertex_dimension(r);
    if ((r_dimension != 1 && r_dimension != 2) || !ensure_boundary_index(mesh)) {
      return TetEditStatus::kRejectedTopology;
    }

    // r's constrained facets and segments, from its star.
    std::vector<std::size_t> r_facets;
    std::vector<std::size_t> r_segments;
    for (std::int32_t cell : removed_star) {
      const Tetrahedron& tet = mesh.tet(cell);
      for (int face = 0; face < 4; ++face) {
        if (tet.v[face] == r || mesh.complex_.constraint(cell, face) == kNoConstraint) continue;
        std::int32_t corners[3];
        oriented_facet(tet.v, face, corners);
        const auto found = boundary_index_.facets.find(sorted_ids(
            {mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]), mesh.vertex_id(corners[2])}));
        if (found == boundary_index_.facets.end()) return TetEditStatus::kRejectedTopology;
        if (std::find(r_facets.begin(), r_facets.end(), found->second) == r_facets.end()) {
          r_facets.push_back(found->second);
        }
      }
      for (std::int32_t other : tet.v) {
        if (other == r || r_dimension > 1 || mesh.vertex_dimension(other) > 1) continue;
        const auto found = boundary_index_.segments.find(sorted_ids(r_id, mesh.vertex_id(other)));
        if (found != boundary_index_.segments.end() &&
            std::find(r_segments.begin(), r_segments.end(), found->second) == r_segments.end()) {
          r_segments.push_back(found->second);
        }
      }
    }
    if (r_facets.empty() && r_segments.empty()) return TetEditStatus::kRejectedTopology;

    // The source rule: s lies on r's carrier, the source points common to the
    // groups of r's facets and segments (fewer than two pin r), and a segment
    // vertex collapses only along a constrained segment.
    std::vector<std::int32_t> entity;
    bool first_carrier = true;
    const auto narrow = [&](const std::vector<std::int32_t>& points) {
      if (first_carrier) {
        entity = points;
        first_carrier = false;
        return;
      }
      std::vector<std::int32_t> common;
      std::set_intersection(entity.begin(), entity.end(), points.begin(), points.end(),
                            std::back_inserter(common));
      entity.swap(common);
    };
    for (std::size_t row : r_facets) {
      const auto found = boundary_index_.face_group_points.find(mesh.facets_[row].source);
      if (found == boundary_index_.face_group_points.end()) return TetEditStatus::kRejectedTopology;
      narrow(found->second);
    }
    for (std::size_t row : r_segments) {
      const auto found = boundary_index_.segment_group_points.find(mesh.segments_[row].source);
      if (found == boundary_index_.segment_group_points.end()) {
        return TetEditStatus::kRejectedTopology;
      }
      narrow(found->second);
    }
    if (entity.size() < 2) return TetEditStatus::kRejectedTopology;
    if (!r_segments.empty() && !boundary_index_.segments.contains(sorted_ids(r_id, s_id))) {
      return TetEditStatus::kRejectedTopology;
    }
    const TetEditVertex* move = moved ? &operation.changed_vertices.front() : nullptr;
    const SourceWitness& s_witness = moved ? move->witness : mesh.witness(s);
    const double* s_carrier = moved ? move->carrier.data() : mesh.point(s);
    const auto on_entity = [&](std::int32_t point) {
      return std::binary_search(entity.begin(), entity.end(), point);
    };
    if (survivor_stencil != nullptr && s_witness.stratum != SourceStratum::kNone) {
      return TetEditStatus::kRejectedTopology;
    }
    const bool rounded = survivor_stencil != nullptr &&
                         survivor_stencil->image == SourceStencilImage::kRoundedPlane;
    std::vector<std::int32_t> s_star;
    if (rounded) {
      if (moved || r_facets.empty() || !r_segments.empty() ||
          r_dimension != 2 || mesh.vertex_dimension(s) != 2) {
        return std::nullopt;
      }
      const std::int32_t group = mesh.facets_[r_facets.front()].source;
      if (!std::ranges::all_of(r_facets, [&](std::size_t facet) {
            return mesh.facets_[facet].source == group;
          })) {
        return std::nullopt;
      }
      collect_star(s, s_star);
      bool on_group = false;
      for (std::int32_t slot : s_star) {
        const Tetrahedron& cell = mesh.tet(slot);
        for (int side = 0; side < 4; ++side) {
          if (cell.v[side] == s) continue;
          const std::int32_t label = mesh.complex_.constraint(slot, side);
          if (label != kNoConstraint && label != group) return std::nullopt;
          on_group = on_group || label == group;
        }
      }
      if (!on_group || !rounded_plane_support(mesh, group, *survivor_stencil, s_carrier)) {
        return std::nullopt;
      }
    }
    bool supported = true;
    const auto row = static_cast<std::size_t>(s_witness.entity);
    switch (s_witness.stratum) {
      case SourceStratum::kSegment: {
        if (s_witness.entity < 0 || row >= mesh.source_segment_groups_.size()) {
          return TetEditStatus::kRejectedTopology;
        }
        const double t = s_witness.parameters[0];
        supported = (t == 1.0 || on_entity(mesh.source_segments_[2 * row])) &&
                    (t == 0.0 || on_entity(mesh.source_segments_[2 * row + 1]));
        break;
      }
      case SourceStratum::kFacet: {
        if (s_witness.entity < 0 || row >= mesh.source_face_groups_.size()) {
          return TetEditStatus::kRejectedTopology;
        }
        const double u = s_witness.parameters[0];
        const double v = s_witness.parameters[1];
        const bool first_weight = !(Dyadic(1.0) - Dyadic(u) - Dyadic(v)).is_zero();
        supported = (!first_weight || on_entity(mesh.source_faces_[3 * row])) &&
                    (u == 0.0 || on_entity(mesh.source_faces_[3 * row + 1])) &&
                    (v == 0.0 || on_entity(mesh.source_faces_[3 * row + 2]));
        break;
      }
      case SourceStratum::kPlane: {
        if (s_witness.entity < 0 || row >= mesh.source_face_groups_.size()) {
          return TetEditStatus::kRejectedTopology;
        }
        const auto& points = boundary_index_.face_group_points[mesh.source_face_groups_[row]];
        supported = std::all_of(points.begin(), points.end(), on_entity);
        break;
      }
      case SourceStratum::kNone: {
        if (rounded) break;
        if (survivor_stencil == nullptr) {
          // A witness-free vertex is its own exact point: one of the carrier's.
          supported = std::any_of(entity.begin(), entity.end(), [&](std::int32_t point) {
            const double* p = mesh.source_points_.data() + 3 * static_cast<std::size_t>(point);
            return p[0] == s_carrier[0] && p[1] == s_carrier[1] && p[2] == s_carrier[2];
          });
          break;
        }
        supported = source_stencil_matches(*survivor_stencil, mesh.source_points_, s_carrier) &&
                    source_stencil_supported(*survivor_stencil, entity);
        break;
      }
    }
    if (!supported) return TetEditStatus::kRejectedTopology;

    // The cavity: r's star, and s's when s moves.
    BoundaryEdit edit;
    edit.removed_vertex = r;
    edit.removed.assign(removed_star.begin(), removed_star.end());
    if (moved) {
      s_star.assign(survivor_star->begin(), survivor_star->end());
    } else if (!rounded) {
      collect_star(s, s_star);
    }
    if (moved) edit.removed.insert(edit.removed.end(), s_star.begin(), s_star.end());
    std::sort(edit.removed.begin(), edit.removed.end());
    edit.removed.erase(std::unique(edit.removed.begin(), edit.removed.end()), edit.removed.end());
    std::vector<std::int32_t> requested;
    requested.reserve(operation.removed_cell_ids.size());
    for (std::uint64_t id : operation.removed_cell_ids) requested.push_back(mesh.cell_slot(id));
    std::sort(requested.begin(), requested.end());
    if (requested != edit.removed) return TetEditStatus::kRejectedTopology;

    // The retired facets and segments are r's.
    const auto resolve = [](std::span<const std::uint64_t> ids, const std::vector<std::size_t>& rows,
                            const auto& bank, std::vector<std::size_t>& resolved) {
      if (ids.size() != rows.size()) return false;
      for (std::uint64_t id : ids) {
        const auto found = std::find_if(rows.begin(), rows.end(),
                                        [&](std::size_t row) { return bank[row].id == id; });
        if (found == rows.end() ||
            std::find(resolved.begin(), resolved.end(), *found) != resolved.end()) {
          return false;
        }
        resolved.push_back(*found);
      }
      return true;
    };
    if (!resolve(operation.removed_facet_ids, r_facets, mesh.facets_, edit.facet_rows) ||
        !resolve(operation.removed_segment_ids, r_segments, mesh.segments_, edit.segment_rows)) {
      return TetEditStatus::kRejectedTopology;
    }

    // apply_edit_batch: the removed vertex, the survivor and the link.
    if (mesh.protected_vertex(r) || mesh.protection_radius(r) > 0.0) {
      return TetEditStatus::kRejectedProtected;
    }
    if (mesh.protected_vertex(s) || mesh.protection_radius(s) > 0.0) {
      return TetEditStatus::kRejectedProtected;
    }
    std::vector<std::int32_t> link_cells = edit.removed;
    if (!moved) link_cells.insert(link_cells.end(), s_star.begin(), s_star.end());
    const std::size_t finite_cells = link_cells.size();
    for (std::size_t index = 0; index < finite_cells; ++index) {
      const Tetrahedron& tet = mesh.tet(link_cells[index]);
      for (int face = 0; face < 4; ++face) {
        const std::int32_t neighbor = tet.n[face];
        if (!is_ghost(mesh.tet(neighbor))) continue;
        if (vertex_slot(mesh.tet(neighbor), r) >= 0 || vertex_slot(mesh.tet(neighbor), s) >= 0) {
          link_cells.push_back(neighbor);
        }
      }
    }
    std::sort(link_cells.begin(), link_cells.end());
    link_cells.erase(std::unique(link_cells.begin(), link_cells.end()), link_cells.end());
    if (!link_condition(mesh, link_cells, r, s)) return TetEditStatus::kRejectedTopology;

    // The moved survivor's record (validate_and_commit's changed vertex).
    const TetMesh::RelocationOverlay overlay{operation.changed_vertices};
    if (moved) {
      if (move->id == kMaxId) return TetEditStatus::kIndexOverflow;
      if (!move->ancestry.empty()) return TetEditStatus::kRejectedTopology;
      if (move->protected_vertex != mesh.protected_vertex(s) ||
          move->protection_radius != mesh.protection_radius(s) ||
          move->witness.stratum != mesh.witness(s).stratum) {
        return TetEditStatus::kRejectedProtected;
      }
      TetEditVertex checked = *move;
      const auto index = static_cast<std::size_t>(s);
      if (!checked.edge_witness_supplied && !checked.has_edge_parameter &&
          mesh.vertex_has_edge_parameter_[index] != 0) {
        checked.has_edge_parameter = true;
        checked.edge_endpoints = mesh.vertex_edge_endpoints_[index];
        checked.edge_parameter = mesh.vertex_edge_parameters_[index];
      }
      if (!std::isfinite(move->protection_radius) || move->protection_radius < 0.0 ||
          !mesh.source_carrier_valid(checked)) {
        return TetEditStatus::kInvalidGeometry;
      }
      if (move->protection_radius > 0.0 && !move->protected_vertex) {
        return TetEditStatus::kRejectedProtected;
      }
      // Only a vertex of the cavity can share the moved point: the replacement
      // cells, exactly positive, tile the cavity, whose closure holds no other
      // vertex.
      for (std::int32_t cell : edit.removed) {
        for (std::int32_t vertex : mesh.tet(cell).v) {
          if (vertex == s || vertex == r) continue;
          const double* p = mesh.point(vertex);
          if (p[0] == move->carrier[0] && p[1] == move->carrier[1] && p[2] == move->carrier[2]) {
            return TetEditStatus::kInvalidGeometry;
          }
        }
      }
      edit.overlay = &overlay;
      edit.moved_vertex = s;
    }

    // The images of r's facets and segments not through s.
    const auto image = [&](std::uint64_t id) { return id == r_id ? s_id : id; };
    std::vector<std::tuple<FaceIds, std::int32_t, std::int8_t>> facet_images;
    for (std::size_t row : r_facets) {
      const TetSnapshotFacet& facet = mesh.facets_[row];
      if (std::find(facet.vertices.begin(), facet.vertices.end(), s_id) != facet.vertices.end()) {
        continue;
      }
      const FaceIds key = sorted_ids(
          {image(facet.vertices[0]), image(facet.vertices[1]), image(facet.vertices[2])});
      edit.image_refused = edit.image_refused || boundary_index_.facets.contains(key);
      facet_images.emplace_back(key, facet.source, facet.orientation);
    }
    std::vector<std::pair<EdgeIds, std::int32_t>> segment_images;
    for (std::size_t row : r_segments) {
      const TetSnapshotSegment& segment = mesh.segments_[row];
      if (segment.vertices[0] == s_id || segment.vertices[1] == s_id) continue;
      const EdgeIds key = sorted_ids(image(segment.vertices[0]), image(segment.vertices[1]));
      edit.image_refused = edit.image_refused || boundary_index_.segments.contains(key);
      segment_images.emplace_back(key, segment.source);
    }
    if (operation.replacement_facets.size() != facet_images.size() ||
        operation.replacement_segments.size() != segment_images.size()) {
      edit.image_refused = true;
    }
    for (const TetEditFacet& facet : operation.replacement_facets) {
      const auto found = std::find(facet_images.begin(), facet_images.end(),
                                   std::tuple{sorted_ids(facet.vertices), facet.source,
                                              facet.orientation});
      if (found == facet_images.end()) {
        edit.image_refused = true;
      } else {
        facet_images.erase(found);
      }
    }
    for (const TetEditSegment& segment : operation.replacement_segments) {
      const auto found =
          std::find(segment_images.begin(), segment_images.end(),
                    std::pair{sorted_ids(segment.vertices[0], segment.vertices[1]),
                              segment.source});
      if (found == segment_images.end()) {
        edit.image_refused = true;
      } else {
        segment_images.erase(found);
      }
    }
    return commit_boundary_edit(mesh, operation, edit);
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

TetEditStatus LocalIncidence::swap_boundary_edge(TetMesh& mesh, const TetEdit& operation) {
  mesh.clear_implicit_facet_refusal();
  try {
    if (mesh_ != &mesh || operation.kind != TetEditKind::kRingReconnect ||
        mesh.periodic_identity_.bound || !mesh.periodic_orbits_.empty() ||
        !mesh.quotient_cells_.empty() || mesh.implicit_source_ ||
        !operation.collapse_pairs.empty() || !operation.removed_vertex_ids.empty() ||
        !operation.changed_vertices.empty() || !operation.split_edges.empty() ||
        operation.removed_facet_ids.size() != 2 || operation.replacement_facets.size() != 2 ||
        !operation.removed_segment_ids.empty() || !operation.replacement_segments.empty() ||
        operation.removed_cell_ids.empty() || periodic_operation(operation) ||
        !ensure_boundary_index(mesh)) {
      return TetEditStatus::kRejectedTopology;
    }
    BoundaryEdit edit;
    edit.volume_preserving = true;
    for (std::uint64_t id : operation.removed_cell_ids) {
      const std::int32_t slot = mesh.cell_slot(id);
      if (slot < 0 || !mesh.complex_.live(slot) || is_ghost(mesh.tet(slot))) {
        return TetEditStatus::kRejectedTopology;
      }
      edit.removed.push_back(slot);
    }
    std::sort(edit.removed.begin(), edit.removed.end());
    if (std::adjacent_find(edit.removed.begin(), edit.removed.end()) != edit.removed.end()) {
      return TetEditStatus::kRejectedTopology;
    }

    // abc and abd: the two retired facets, constrained faces of the cells.
    edit.facet_rows.assign(2, mesh.facets_.size());
    for (std::int32_t cell : edit.removed) {
      const Tetrahedron& tet = mesh.tet(cell);
      for (int face = 0; face < 4; ++face) {
        if (mesh.complex_.constraint(cell, face) == kNoConstraint) continue;
        std::int32_t corners[3];
        oriented_facet(tet.v, face, corners);
        const auto found = boundary_index_.facets.find(sorted_ids(
            {mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]), mesh.vertex_id(corners[2])}));
        if (found == boundary_index_.facets.end()) return TetEditStatus::kRejectedTopology;
        for (std::size_t which = 0; which < 2; ++which) {
          if (mesh.facets_[found->second].id == operation.removed_facet_ids[which]) {
            edit.facet_rows[which] = found->second;
          }
        }
      }
    }
    if (edit.facet_rows[0] == mesh.facets_.size() || edit.facet_rows[1] == mesh.facets_.size() ||
        edit.facet_rows[0] == edit.facet_rows[1]) {
      return TetEditStatus::kRejectedTopology;
    }
    const TetSnapshotFacet& first = mesh.facets_[edit.facet_rows[0]];
    const TetSnapshotFacet& second = mesh.facets_[edit.facet_rows[1]];
    if (first.source != second.source || first.orientation != second.orientation) {
      return TetEditStatus::kRejectedTopology;
    }
    std::vector<std::uint64_t> shared;
    for (std::uint64_t id : first.vertices) {
      if (std::find(second.vertices.begin(), second.vertices.end(), id) != second.vertices.end()) {
        shared.push_back(id);
      }
    }
    if (shared.size() != 2) return TetEditStatus::kRejectedTopology;
    const auto opposite = [&](const TetSnapshotFacet& facet) {
      for (std::uint64_t id : facet.vertices) {
        if (id != shared[0] && id != shared[1]) return id;
      }
      return kMaxId;
    };
    const std::int32_t a = mesh.vertex_slot(shared[0]);
    const std::int32_t b = mesh.vertex_slot(shared[1]);
    const std::int32_t c = mesh.vertex_slot(opposite(first));
    const std::int32_t d = mesh.vertex_slot(opposite(second));
    if (boundary_index_.segments.contains(sorted_ids(shared[0], shared[1])) ||
        !edge_star(c, d).empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    // The cells around ab: an open fan from c to d.
    const std::span<const std::int32_t> fan = edge_star(a, b);
    if (!std::equal(fan.begin(), fan.end(), edit.removed.begin(), edit.removed.end()) ||
        fan.size() < 2) {
      return TetEditStatus::kRejectedTopology;
    }
    std::map<std::int32_t, std::vector<std::int32_t>> ring;
    for (std::int32_t cell : fan) {
      std::array<std::int32_t, 2> rest{};
      std::size_t count = 0;
      for (std::int32_t vertex : mesh.tet(cell).v) {
        if (vertex == a || vertex == b) continue;
        if (count == 2 || vertex < 0) return TetEditStatus::kRejectedTopology;
        rest[count++] = vertex;
      }
      ring[rest[0]].push_back(rest[1]);
      ring[rest[1]].push_back(rest[0]);
    }
    if (ring[c].size() != 1 || ring[d].size() != 1) return TetEditStatus::kRejectedTopology;
    std::size_t walked = 1;
    for (std::int32_t previous = -1, current = c; current != d; ++walked) {
      const std::vector<std::int32_t>& next = ring[current];
      if (next.size() > 2 || walked > ring.size()) return TetEditStatus::kRejectedTopology;
      const std::int32_t following = next[0] != previous ? next[0] : next.size() == 2 ? next[1] : -1;
      if (following < 0) return TetEditStatus::kRejectedTopology;
      previous = current;
      current = following;
    }
    if (walked != ring.size()) return TetEditStatus::kRejectedTopology;

    // acd and bcd, each oriented like abc.
    std::array<std::int32_t, 3> reference{};
    for (std::size_t corner = 0; corner < 3; ++corner) {
      reference[corner] = mesh.vertex_slot(first.vertices[corner]);
    }
    const std::array<Dyadic, 3> normal = carrier_normal(mesh, reference);
    std::vector<FaceIds> expected;
    for (std::int32_t apex : {a, b}) {
      const std::array<Dyadic, 3> other = carrier_normal(mesh, {apex, c, d});
      const Dyadic alignment = other[0] * normal[0] + other[1] * normal[1] + other[2] * normal[2];
      edit.degenerate_image = edit.degenerate_image || alignment.is_zero();
      const FaceIds key = sorted_ids({mesh.vertex_id(apex), mesh.vertex_id(c), mesh.vertex_id(d)});
      edit.image_refused = edit.image_refused || boundary_index_.facets.contains(key);
      expected.push_back(key);
    }
    for (const TetEditFacet& facet : operation.replacement_facets) {
      const auto found = std::find(expected.begin(), expected.end(), sorted_ids(facet.vertices));
      if (found == expected.end() || facet.source != first.source ||
          facet.orientation != first.orientation) {
        edit.image_refused = true;
      } else {
        expected.erase(found);
      }
    }
    return commit_boundary_edit(mesh, operation, edit);
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

TetEditStatus LocalIncidence::split_boundary_edge(TetMesh& mesh, std::int32_t a, std::int32_t b,
                                                  const TetEditVertex& vertex,
                                                  std::vector<std::uint64_t>& created_cells) {
  mesh.clear_implicit_facet_refusal();
  created_cells.clear();
  const std::uint64_t generation = mesh.generation_;
  const std::size_t vertex_count = mesh.vertex_ids_.size();
  bool staged = false;
  // The staged vertex leaves with a refused edit.
  const auto unstage = [&]() noexcept {
    if (!staged || mesh.generation_ != generation) return;
    mesh.vertex_index_.forget(vertex.id);
    mesh.points_.resize(3 * vertex_count);
    mesh.vertex_ids_.resize(vertex_count);
    mesh.vertex_live_.resize(vertex_count);
    mesh.vertex_dimension_.resize(vertex_count);
    mesh.protection_radii_.resize(vertex_count);
    mesh.protected_vertices_.resize(vertex_count);
    mesh.witnesses_.resize(vertex_count);
    mesh.vertex_has_edge_parameter_.resize(vertex_count);
    mesh.vertex_edge_endpoints_.resize(vertex_count);
    mesh.vertex_edge_parameters_.resize(vertex_count);
    staged = false;
  };
  try {
    if (mesh_ != &mesh || mesh.periodic_identity_.bound || !mesh.periodic_orbits_.empty() ||
        !mesh.quotient_cells_.empty() || mesh.implicit_source_ || !mesh.live_vertex(a) ||
        !mesh.live_vertex(b) || a == b || !ensure_boundary_index(mesh)) {
      return TetEditStatus::kRejectedTopology;
    }
    const std::uint64_t a_id = mesh.vertex_id(a);
    const std::uint64_t b_id = mesh.vertex_id(b);
    if (boundary_index_.segments.contains(sorted_ids(a_id, b_id))) {
      return TetEditStatus::kRejectedTopology;
    }
    BoundaryEdit edit;
    const std::span<const std::int32_t> around = edge_star(a, b);
    edit.removed.assign(around.begin(), around.end());
    if (edit.removed.empty()) return TetEditStatus::kRejectedTopology;
    // abc and abd: the constrained faces through ab, both on the hull.
    for (std::int32_t cell : edit.removed) {
      const Tetrahedron& tet = mesh.tet(cell);
      for (int face = 0; face < 4; ++face) {
        if (tet.v[face] == a || tet.v[face] == b ||
            mesh.complex_.constraint(cell, face) == kNoConstraint) {
          continue;
        }
        if (!is_ghost(mesh.tet(tet.n[face]))) return TetEditStatus::kRejectedTopology;
        std::int32_t corners[3];
        oriented_facet(tet.v, face, corners);
        const auto found = boundary_index_.facets.find(sorted_ids(
            {mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]), mesh.vertex_id(corners[2])}));
        if (found == boundary_index_.facets.end()) return TetEditStatus::kRejectedTopology;
        edit.facet_rows.push_back(found->second);
      }
    }
    if (edit.facet_rows.size() != 2 || edit.facet_rows[0] == edit.facet_rows[1]) {
      return TetEditStatus::kRejectedTopology;
    }

    // validate_and_commit's new vertex, in its order.
    if (vertex.id == kMaxId) return TetEditStatus::kIndexOverflow;
    if (mesh.vertex_index_.used(vertex.id)) return TetEditStatus::kRejectedTopology;
    if (vertex_count >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      return TetEditStatus::kIndexOverflow;
    }
    if (!std::isfinite(vertex.protection_radius) || vertex.protection_radius < 0.0 ||
        !mesh.source_carrier_valid(vertex)) {
      return TetEditStatus::kInvalidGeometry;
    }
    if (vertex.protection_radius > 0.0 && !vertex.protected_vertex) {
      return TetEditStatus::kRejectedProtected;
    }
    // Only a vertex of the cavity can share the point: the exactly positive
    // halves tile the cavity, whose closure holds no other vertex.
    for (std::int32_t cell : edit.removed) {
      for (std::int32_t other : mesh.tet(cell).v) {
        const double* p = mesh.point(other);
        if (p[0] == vertex.carrier[0] && p[1] == vertex.carrier[1] && p[2] == vertex.carrier[2]) {
          return TetEditStatus::kInvalidGeometry;
        }
      }
    }
    const std::array<std::uint64_t, 2> edge{a_id, b_id};
    const TetEditStatus point_status =
        mesh.unconstrained_split_point_status(a, b, edge, vertex, edit.removed,
                                              &boundary_index_.segments);
    if (point_status != TetEditStatus::kApplied) return point_status;
    if (2 * edit.removed.size() > kMaxId - mesh.next_cell_id_ ||
        4 > kMaxId - mesh.next_facet_id_) {
      return TetEditStatus::kIndexOverflow;
    }

    // The kSplit edit: each cell's halves, then the facets' halves.
    TetEdit operation;
    operation.kind = TetEditKind::kSplit;
    operation.split_edges.push_back(edge);
    operation.changed_vertices.push_back(vertex);
    operation.removed_cell_ids.reserve(edit.removed.size());
    operation.replacement_cells.reserve(2 * edit.removed.size());
    std::uint64_t next_cell = mesh.next_cell_id_;
    // Fresh IDs follow logical parents, independent of cavity slot reuse.
    std::sort(edit.removed.begin(), edit.removed.end(), [&](std::int32_t p, std::int32_t q) {
      return mesh.cell_id(p) < mesh.cell_id(q);
    });
    for (std::int32_t cell : edit.removed) {
      const Tetrahedron& tet = mesh.tet(cell);
      const std::uint64_t parent = mesh.cell_id(cell);
      operation.removed_cell_ids.push_back(parent);
      const int slot_a = vertex_slot(tet, a);
      const int slot_b = vertex_slot(tet, b);
      for (const int replaced : {slot_b, slot_a}) {
        // The face opposite the kept end is the new face between the halves.
        const int kept = replaced == slot_b ? slot_a : slot_b;
        TetEditCell child;
        child.id = next_cell++;
        child.region = mesh.complex_.region(cell);
        child.ancestor_cell_ids = {parent};
        for (int corner = 0; corner < 4; ++corner) {
          const auto index = static_cast<std::size_t>(corner);
          child.vertices[index] = corner == replaced ? vertex.id : mesh.vertex_id(tet.v[corner]);
          child.facet_sources[index] =
              corner == kept ? kNoConstraint : mesh.complex_.constraint(cell, corner);
        }
        operation.replacement_cells.push_back(child);
      }
    }
    std::sort(edit.removed.begin(), edit.removed.end());
    std::sort(edit.facet_rows.begin(), edit.facet_rows.end(), [&](std::size_t p, std::size_t q) {
      return mesh.facets_[p].id < mesh.facets_[q].id;
    });
    std::uint64_t next_facet = mesh.next_facet_id_;
    for (std::size_t row : edit.facet_rows) {
      const TetSnapshotFacet& facet = mesh.facets_[row];
      operation.removed_facet_ids.push_back(facet.id);
      for (const std::uint64_t replaced : {b_id, a_id}) {
        TetEditFacet half{next_facet++, facet.vertices, facet.source, facet.orientation, false};
        std::replace(half.vertices.begin(), half.vertices.end(), replaced, vertex.id);
        operation.replacement_facets.push_back(half);
      }
    }

    // The vertex record, staged for the facet and cell checks.
    const auto slot = static_cast<std::int32_t>(vertex_count);
    // Geometric growth: an exact reserve would copy the banks every split.
    reserve_for(mesh.points_, mesh.points_.size() + 3);
    for (auto* bank : {&mesh.vertex_live_, &mesh.protected_vertices_,
                       &mesh.vertex_has_edge_parameter_}) {
      reserve_for(*bank, vertex_count + 1);
    }
    reserve_for(mesh.vertex_ids_, vertex_count + 1);
    reserve_for(mesh.vertex_dimension_, vertex_count + 1);
    reserve_for(mesh.protection_radii_, vertex_count + 1);
    reserve_for(mesh.witnesses_, vertex_count + 1);
    reserve_for(mesh.vertex_edge_endpoints_, vertex_count + 1);
    reserve_for(mesh.vertex_edge_parameters_, vertex_count + 1);
    staged = true;
    mesh.points_.insert(mesh.points_.end(), vertex.carrier.begin(), vertex.carrier.end());
    mesh.vertex_ids_.push_back(vertex.id);
    mesh.vertex_live_.push_back(1);
    mesh.vertex_dimension_.push_back(3);
    mesh.protection_radii_.push_back(vertex.protection_radius);
    mesh.protected_vertices_.push_back(
        vertex.protected_vertex || vertex.protection_radius > 0.0 ? 1 : 0);
    mesh.witnesses_.push_back(vertex.witness);
    mesh.vertex_has_edge_parameter_.push_back(vertex.has_edge_parameter ? 1U : 0U);
    mesh.vertex_edge_endpoints_.push_back(vertex.edge_endpoints);
    mesh.vertex_edge_parameters_.push_back(vertex.edge_parameter);
    mesh.vertex_index_.bind(vertex.id, slot);
    edit.split_first = a;
    edit.split_second = b;
    edit.inserted = slot;
    const TetEditStatus status = commit_boundary_edit(mesh, operation, edit);
    if (mesh.generation_ == generation) {
      unstage();
      return status;
    }
    if (!vertex.ancestry.empty()) mesh.vertex_ancestry_.emplace(vertex.id, vertex.ancestry);
    if (vertex.id >= mesh.next_vertex_id_) mesh.next_vertex_id_ = vertex.id + 1;
    for (const TetEditCell& cell : operation.replacement_cells) created_cells.push_back(cell.id);
    return status;
  } catch (const std::bad_alloc&) {
    unstage();
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    unstage();
    return TetEditStatus::kIndexOverflow;
  }
}

TetEditStatus LocalIncidence::relocate_boundary(TetMesh& mesh, const TetEditVertex& moved,
                                               const Star& star,
                                               const SourceStencil* moved_stencil) {
  mesh.clear_implicit_facet_refusal();
  try {
    // TetMesh::apply_relocation_edit's domain, without an implicit source.
    if (!mesh.periodic_orbits_.empty() || !mesh.quotient_cells_.empty() ||
        !mesh.periodic_entity_maps_.empty() || mesh.implicit_source_ ||
        !ensure_boundary_index(mesh)) {
      return TetEditStatus::kRejectedTopology;
    }
    if (mesh.generation_ == kMaxId) return TetEditStatus::kIndexOverflow;
    // probe_relocation_edit's record checks, in its order.
    if (moved.id == kMaxId) return TetEditStatus::kIndexOverflow;
    const std::int32_t slot = star.vertex_;
    if (!moved.ancestry.empty()) return TetEditStatus::kRejectedTopology;
    if (mesh.protected_vertex(slot) || mesh.protection_radius(slot) > 0.0 ||
        moved.protected_vertex != mesh.protected_vertex(slot) ||
        moved.protection_radius != mesh.protection_radius(slot) ||
        moved.witness.stratum != mesh.witness(slot).stratum) {
      return TetEditStatus::kRejectedProtected;
    }
    if (moved_stencil != nullptr && moved.witness.stratum != SourceStratum::kNone) {
      return TetEditStatus::kRejectedTopology;
    }
    TetEditVertex checked = moved;
    const auto index = static_cast<std::size_t>(slot);
    if (!checked.edge_witness_supplied && !checked.has_edge_parameter &&
        mesh.vertex_has_edge_parameter_[index] != 0) {
      checked.has_edge_parameter = true;
      checked.edge_endpoints = mesh.vertex_edge_endpoints_[index];
      checked.edge_parameter = mesh.vertex_edge_parameters_[index];
    }
    if (!std::isfinite(moved.protection_radius) || moved.protection_radius < 0.0 ||
        !mesh.source_carrier_valid(checked)) {
      return TetEditStatus::kInvalidGeometry;
    }
    if (moved.protection_radius > 0.0 && !moved.protected_vertex) {
      return TetEditStatus::kRejectedProtected;
    }
    // Only a star vertex can share the moved point: the exactly positive
    // moved star cells keep the point inside its link.
    for (std::int32_t cell : star) {
      for (std::int32_t other : mesh.tet(cell).v) {
        if (other == slot) continue;
        const double* p = mesh.point(other);
        if (p[0] == moved.carrier[0] && p[1] == moved.carrier[1] && p[2] == moved.carrier[2]) {
          return TetEditStatus::kInvalidGeometry;
        }
      }
    }

    // The facets and segments through the vertex, in bank order.
    const std::uint64_t id = moved.id;
    const auto dimension = mesh.vertex_dimension(slot);
    const bool rounded =
        moved_stencil != nullptr && moved_stencil->image == SourceStencilImage::kRoundedPlane;
    std::vector<std::size_t> facet_rows;
    std::vector<std::size_t> segment_rows;
    for (std::int32_t cell : star) {
      const Tetrahedron& tet = mesh.tet(cell);
      for (int face = 0; face < 4; ++face) {
        if (tet.v[face] == slot || mesh.complex_.constraint(cell, face) == kNoConstraint) continue;
        if (rounded && !is_ghost(mesh.tet(tet.n[face]))) {
          return TetEditStatus::kRejectedTopology;
        }
        std::int32_t corners[3];
        oriented_facet(tet.v, face, corners);
        const auto found = boundary_index_.facets.find(sorted_ids(
            {mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]), mesh.vertex_id(corners[2])}));
        if (found == boundary_index_.facets.end()) return TetEditStatus::kRejectedTopology;
        facet_rows.push_back(found->second);
      }
      for (std::int32_t other : tet.v) {
        if (other == slot || (!rounded && (dimension > 1 || mesh.vertex_dimension(other) > 1))) {
          continue;
        }
        const auto found = boundary_index_.segments.find(sorted_ids(id, mesh.vertex_id(other)));
        if (found == boundary_index_.segments.end()) continue;
        if (rounded) return TetEditStatus::kRejectedTopology;
        segment_rows.push_back(found->second);
      }
    }
    for (auto* rows : {&facet_rows, &segment_rows}) {
      std::sort(rows->begin(), rows->end());
      rows->erase(std::unique(rows->begin(), rows->end()), rows->end());
    }
    if (rounded) {
      if (dimension != 2 || facet_rows.empty()) {
        return TetEditStatus::kRejectedTopology;
      }
      const std::int32_t group = mesh.facets_[facet_rows.front()].source;
      if (!std::ranges::all_of(facet_rows, [&](std::size_t row) {
            return mesh.facets_[row].source == group;
          }) ||
          !rounded_plane_support(mesh, group, *moved_stencil, moved.carrier.data())) {
        return TetEditStatus::kRejectedTopology;
      }
    } else if (moved_stencil != nullptr &&
               ((facet_rows.empty() && segment_rows.empty()) ||
                !source_stencil_matches(*moved_stencil, mesh.source_points_,
                                        moved.carrier.data()))) {
      return TetEditStatus::kRejectedTopology;
    }
    const auto supports_stencil = [&](const auto& groups, std::int32_t source) {
      const auto group = groups.find(source);
      return group != groups.end() && source_stencil_supported(*moved_stencil, group->second);
    };
    const TetMesh::RelocationOverlay overlay{std::span<const TetEditVertex>(&moved, 1)};
    for (std::size_t row : facet_rows) {
      const TetSnapshotFacet& facet = mesh.facets_[row];
      if (!rounded && moved_stencil != nullptr &&
          !supports_stencil(boundary_index_.face_group_points, facet.source)) {
        return TetEditStatus::kRejectedTopology;
      }
      const TetEditFacet check{facet.id, facet.vertices, facet.source, facet.orientation, false};
      if (!mesh.source_face_valid(check, &overlay)) return mesh.source_face_refusal_status();
    }
    for (std::size_t row : segment_rows) {
      const TetSnapshotSegment& segment = mesh.segments_[row];
      if (moved_stencil != nullptr &&
          !supports_stencil(boundary_index_.segment_group_points, segment.source)) {
        return TetEditStatus::kRejectedTopology;
      }
      const TetEditSegment check{segment.id, segment.vertices, segment.source, false};
      if (!mesh.source_segment_valid(check, &overlay)) return TetEditStatus::kInvalidGeometry;
    }
    for (std::int32_t cell : star) {
      if (!mesh.positive_with_relocation(mesh.tet(cell).v, overlay)) {
        return TetEditStatus::kInvalidGeometry;
      }
    }

    // validate_and_commit's relocation writes; none allocates but the
    // ancestry erase, which frees.
    std::copy(moved.carrier.begin(), moved.carrier.end(), mesh.points_.begin() + 3 * index);
    mesh.witnesses_[index] = moved.witness;
    mesh.vertex_ancestry_.erase(id);
    if (moved.has_edge_parameter) {
      mesh.vertex_has_edge_parameter_[index] = 1;
      mesh.vertex_edge_endpoints_[index] = moved.edge_endpoints;
      mesh.vertex_edge_parameters_[index] = moved.edge_parameter;
    } else if (moved.edge_witness_supplied) {
      mesh.vertex_has_edge_parameter_[index] = 0;
      mesh.vertex_edge_endpoints_[index] = {};
      mesh.vertex_edge_parameters_[index] = 0.0;
    }
    ++mesh.generation_;
    mesh.unmet_records_.clear();
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  }
}

TetEditStatus LocalIncidence::commit_boundary_edit(TetMesh& mesh, const TetEdit& operation,
                                                   BoundaryEdit& edit) {
  const bool fixed = mesh.policy_ == BoundaryPolicy::kFixed;
  const auto removed_cell = [&](std::int32_t slot) {
    return std::binary_search(edit.removed.begin(), edit.removed.end(), slot);
  };
  const auto face_key = [&](const std::int32_t* corners) {
    return sorted_ids(
        {mesh.vertex_id(corners[0]), mesh.vertex_id(corners[1]), mesh.vertex_id(corners[2])});
  };
  const auto facet_slots = [&](const std::array<std::uint64_t, 3>& vertices,
                               std::array<std::int32_t, 3>& slots) {
    for (std::size_t corner = 0; corner < 3; ++corner) {
      slots[corner] = mesh.vertex_slot(vertices[corner]);
      if (!mesh.live_vertex(slots[corner])) return false;
    }
    return true;
  };

  // Retired facets: unprotected, every side a removed cell (or its ghost).
  std::set<SlotFace> retired;
  std::vector<std::pair<std::int32_t, int>> cleared_sides;
  std::vector<std::uint8_t> owned(edit.facet_rows.size(), 0);
  std::vector<std::uint8_t> foreign(edit.facet_rows.size(), 0);
  std::vector<SlotFace> retired_rows;
  for (std::size_t row : edit.facet_rows) {
    std::array<std::int32_t, 3> slots{};
    if (!facet_slots(mesh.facets_[row].vertices, slots)) return TetEditStatus::kRejectedTopology;
    retired_rows.push_back(sorted_slots(slots.data()));
    retired.insert(retired_rows.back());
  }
  for (std::int32_t cell : edit.removed) {
    const Tetrahedron& tet = mesh.tet(cell);
    for (int face = 0; face < 4; ++face) {
      std::int32_t corners[3];
      oriented_facet(tet.v, face, corners);
      const SlotFace key = sorted_slots(corners);
      const auto which = std::find(retired_rows.begin(), retired_rows.end(), key);
      if (which == retired_rows.end()) continue;
      const auto index = static_cast<std::size_t>(which - retired_rows.begin());
      owned[index] = 1;
      const std::int32_t neighbor = tet.n[face];
      if (!is_ghost(mesh.tet(neighbor)) && !removed_cell(neighbor)) foreign[index] = 1;
      cleared_sides.emplace_back(cell, face);
    }
  }
  for (std::size_t index = 0; index < edit.facet_rows.size(); ++index) {
    if (mesh.facets_[edit.facet_rows[index]].protected_facet || fixed) {
      return TetEditStatus::kRejectedProtected;
    }
    if (owned[index] == 0 || foreign[index] != 0) return TetEditStatus::kRejectedTopology;
  }

  // Replacement facets.
  std::unordered_set<std::uint64_t> new_ids;
  for (const TetEditFacet& facet : operation.replacement_facets) {
    if (facet.id == kMaxId || mesh.facet_index_.used(facet.id) || facet.protected_facet || fixed) {
      return facet.protected_facet || fixed ? TetEditStatus::kRejectedProtected
                                            : TetEditStatus::kRejectedTopology;
    }
    if (!mesh.source_face_valid(facet, edit.overlay)) return mesh.source_face_refusal_status();
  }
  for (const TetEditFacet& facet : operation.replacement_facets) {
    if (!new_ids.insert(facet.id).second) return TetEditStatus::kRejectedTopology;
  }
  if (edit.degenerate_image) return TetEditStatus::kInvalidGeometry;

  // Retired and replacement segments.
  for (std::size_t row : edit.segment_rows) {
    if (mesh.segments_[row].protected_segment || fixed) return TetEditStatus::kRejectedProtected;
  }
  new_ids.clear();
  for (const TetEditSegment& segment : operation.replacement_segments) {
    if (segment.id == kMaxId || mesh.segment_index_.used(segment.id) ||
        segment.protected_segment || fixed) {
      return segment.protected_segment || fixed ? TetEditStatus::kRejectedProtected
                                                : TetEditStatus::kRejectedTopology;
    }
    if (!mesh.source_segment_valid(segment, edit.overlay)) {
      return TetEditStatus::kInvalidGeometry;
    }
  }
  for (const TetEditSegment& segment : operation.replacement_segments) {
    if (!new_ids.insert(segment.id).second) return TetEditStatus::kRejectedTopology;
  }

  // A moved survivor keeps its retained facets and segments only on their
  // sources (in ID order, as apply_edit_batch revalidates them).
  if (edit.overlay != nullptr) {
    const std::int32_t s = edit.moved_vertex;
    std::vector<std::size_t> facet_rows;
    std::vector<std::size_t> segment_rows;
    for (std::int32_t cell : edit.removed) {
      const Tetrahedron& tet = mesh.tet(cell);
      if (vertex_slot(tet, s) < 0) continue;
      for (int face = 0; face < 4; ++face) {
        if (tet.v[face] == s || mesh.complex_.constraint(cell, face) == kNoConstraint) continue;
        std::int32_t corners[3];
        oriented_facet(tet.v, face, corners);
        const auto found = boundary_index_.facets.find(face_key(corners));
        if (found == boundary_index_.facets.end()) return TetEditStatus::kRejectedTopology;
        if (std::find(edit.facet_rows.begin(), edit.facet_rows.end(), found->second) ==
            edit.facet_rows.end()) {
          facet_rows.push_back(found->second);
        }
      }
      for (std::int32_t other : tet.v) {
        if (other == s || mesh.vertex_dimension(s) > 1 || mesh.vertex_dimension(other) > 1) {
          continue;
        }
        const auto found = boundary_index_.segments.find(
            sorted_ids(mesh.vertex_id(s), mesh.vertex_id(other)));
        if (found != boundary_index_.segments.end() &&
            std::find(edit.segment_rows.begin(), edit.segment_rows.end(), found->second) ==
                edit.segment_rows.end()) {
          segment_rows.push_back(found->second);
        }
      }
    }
    const auto by_id = [](const auto& bank) {
      return [&bank](std::size_t x, std::size_t y) { return bank[x].id < bank[y].id; };
    };
    std::sort(facet_rows.begin(), facet_rows.end(), by_id(mesh.facets_));
    facet_rows.erase(std::unique(facet_rows.begin(), facet_rows.end()), facet_rows.end());
    std::sort(segment_rows.begin(), segment_rows.end(), by_id(mesh.segments_));
    segment_rows.erase(std::unique(segment_rows.begin(), segment_rows.end()), segment_rows.end());
    for (std::size_t row : facet_rows) {
      const TetSnapshotFacet& facet = mesh.facets_[row];
      TetEditFacet check;
      check.id = facet.id;
      check.vertices = facet.vertices;
      check.source = facet.source;
      check.orientation = facet.orientation;
      if (!mesh.source_face_valid(check, edit.overlay)) return mesh.source_face_refusal_status();
    }
    for (std::size_t row : segment_rows) {
      const TetSnapshotSegment& segment = mesh.segments_[row];
      TetEditSegment check;
      check.id = segment.id;
      check.vertices = segment.vertices;
      check.source = segment.source;
      if (!mesh.source_segment_valid(check, edit.overlay)) return TetEditStatus::kInvalidGeometry;
    }
  }

  // Regions and constraints (apply_edit_batch's validate_regions_and_constraints).
  const std::size_t created = operation.replacement_cells.size();
  std::vector<std::int32_t> replacement_vertices(4 * created, -1);
  std::map<SlotFace, std::vector<std::pair<std::size_t, int>>> replacement_faces;
  for (std::size_t row = 0; row < created; ++row) {
    const TetEditCell& cell = operation.replacement_cells[row];
    for (std::size_t corner = 0; corner < 4; ++corner) {
      replacement_vertices[4 * row + corner] = mesh.vertex_slot(cell.vertices[corner]);
      if (replacement_vertices[4 * row + corner] < 0) return TetEditStatus::kRejectedTopology;
    }
    for (int face = 0; face < 4; ++face) {
      std::int32_t corners[3];
      oriented_facet(replacement_vertices.data() + 4 * row, face, corners);
      replacement_faces[sorted_slots(corners)].emplace_back(row, face);
    }
  }
  if (edit.image_refused || !boundary_index_.regions_consistent) {
    return TetEditStatus::kRejectedTopology;
  }
  for (std::int32_t cell : edit.removed) {
    const Tetrahedron& tet = mesh.tet(cell);
    for (int face = 0; face < 4; ++face) {
      const std::int32_t neighbor = tet.n[face];
      if (removed_cell(neighbor)) continue;
      std::int32_t corners[3];
      oriented_facet(tet.v, face, corners);
      const SlotFace key = sorted_slots(corners);
      if (is_ghost(mesh.tet(neighbor)) && retired.contains(key)) {
        if (edit.inserted < 0) continue;
        // validate_regions_and_constraints' split face: covered by its
        // halves, each a face of the replacement cells of this region with
        // its orientation.
        std::array<std::int32_t, 3> first_child{};
        std::array<std::int32_t, 3> second_child{};
        bool directed = false;
        for (int offset = 0; offset < 3 && !directed; ++offset) {
          const std::int32_t from = corners[offset];
          const std::int32_t to = corners[(offset + 1) % 3];
          if ((from != edit.split_first || to != edit.split_second) &&
              (from != edit.split_second || to != edit.split_first)) {
            continue;
          }
          const std::int32_t third = corners[(offset + 2) % 3];
          first_child = {from, edit.inserted, third};
          second_child = {edit.inserted, to, third};
          directed = true;
        }
        if (!directed ||
            !mesh.split_face_covered(corners, mesh.complex_.constraint(cell, face), first_child,
                                     second_child, edit.removed, &boundary_index_.segments)) {
          return TetEditStatus::kRejectedTopology;
        }
        for (const auto& child : {first_child, second_child}) {
          const auto halves = replacement_faces.find(sorted_slots(child.data()));
          bool found = false;
          if (halves != replacement_faces.end()) {
            for (const auto& [row, side] : halves->second) {
              if (operation.replacement_cells[row].region != mesh.complex_.region(cell)) continue;
              std::int32_t actual[3];
              oriented_facet(replacement_vertices.data() + 4 * row, side, actual);
              if (!(FacetKey::of(actual) == FacetKey::of(child.data()))) {
                return TetEditStatus::kRejectedTopology;
              }
              found = true;
            }
          }
          if (!found) return TetEditStatus::kRejectedTopology;
        }
        continue;
      }
      const auto found = replacement_faces.find(key);
      if (found == replacement_faces.end()) return TetEditStatus::kRejectedTopology;
      for (const auto& [row, side] : found->second) {
        (void)side;
        if (operation.replacement_cells[row].region != mesh.complex_.region(cell)) {
          return TetEditStatus::kRejectedTopology;
        }
      }
    }
  }
  // The region on each side of each source group, bound by the removed cells.
  struct FacetRecord {
    std::array<std::int32_t, 3> slots{};
    std::int32_t source = kNoConstraint;
    std::int8_t orientation = 1;
  };
  std::map<SlotFace, FacetRecord> new_facets;
  for (const TetEditFacet& facet : operation.replacement_facets) {
    FacetRecord record{{}, facet.source, facet.orientation};
    if (!facet_slots(facet.vertices, record.slots)) return TetEditStatus::kRejectedTopology;
    if (!new_facets.emplace(sorted_slots(record.slots.data()), record).second) {
      return TetEditStatus::kRejectedTopology;
    }
  }
  // The future facet on a face: a replacement facet, else a retained one.
  const auto future_facet = [&](const SlotFace& key, const std::int32_t* corners,
                                FacetRecord& record) {
    const auto made = new_facets.find(key);
    if (made != new_facets.end()) {
      record = made->second;
      return true;
    }
    if (retired.contains(key)) return false;
    const auto found = boundary_index_.facets.find(face_key(corners));
    if (found == boundary_index_.facets.end()) return false;
    const TetSnapshotFacet& facet = mesh.facets_[found->second];
    record.source = facet.source;
    record.orientation = facet.orientation;
    return facet_slots(facet.vertices, record.slots);
  };
  std::map<std::pair<std::int32_t, int>, std::int32_t> source_regions;
  for (std::int32_t cell : edit.removed) {
    const Tetrahedron& tet = mesh.tet(cell);
    for (int face = 0; face < 4; ++face) {
      const std::int32_t source = mesh.complex_.constraint(cell, face);
      if (source < 0) continue;
      std::int32_t corners[3];
      oriented_facet(tet.v, face, corners);
      const auto found = boundary_index_.facets.find(face_key(corners));
      if (found == boundary_index_.facets.end()) return TetEditStatus::kRejectedTopology;
      const TetSnapshotFacet& facet = mesh.facets_[found->second];
      if (facet.source != source) continue;
      std::array<std::int32_t, 3> slots{};
      if (!facet_slots(facet.vertices, slots)) return TetEditStatus::kRejectedTopology;
      const int direction = FacetKey::of(corners) == FacetKey::of(slots.data()) ? 1 : -1;
      const auto [entry, inserted] = source_regions.emplace(
          std::pair{source, direction * facet.orientation}, mesh.complex_.region(cell));
      if (!inserted && entry->second != mesh.complex_.region(cell)) {
        return TetEditStatus::kRejectedTopology;
      }
    }
  }
  for (std::size_t row = 0; row < created; ++row) {
    const TetEditCell& cell = operation.replacement_cells[row];
    for (int face = 0; face < 4; ++face) {
      const std::int32_t source = cell.facet_sources[static_cast<std::size_t>(face)];
      if (source < 0) continue;
      std::int32_t corners[3];
      oriented_facet(replacement_vertices.data() + 4 * row, face, corners);
      FacetRecord record;
      if (!future_facet(sorted_slots(corners), corners, record) || record.source != source) {
        return TetEditStatus::kRejectedTopology;
      }
      const int direction =
          (FacetKey::of(corners) == FacetKey::of(record.slots.data()) ? 1 : -1) *
          record.orientation;
      const auto expected = source_regions.find({source, direction});
      if (expected == source_regions.end() || expected->second != cell.region) {
        return TetEditStatus::kRejectedTopology;
      }
    }
  }
  for (const auto& [key, record] : new_facets) {
    const auto found = replacement_faces.find(key);
    if (found == replacement_faces.end()) return TetEditStatus::kRejectedTopology;
    for (const auto& [row, face] : found->second) {
      std::int32_t corners[3];
      oriented_facet(replacement_vertices.data() + 4 * row, face, corners);
      const int side =
          (FacetKey::of(corners) == FacetKey::of(record.slots.data()) ? 1 : -1) *
          record.orientation;
      const auto expected = source_regions.find({record.source, side});
      if (expected == source_regions.end() ||
          expected->second != operation.replacement_cells[row].region) {
        return TetEditStatus::kRejectedTopology;
      }
    }
  }

  // Replacement cells: identities, ancestry, corners and exact orientation.
  std::unordered_set<std::uint64_t> cell_ids;
  for (std::size_t row = 0; row < created; ++row) {
    const TetEditCell& cell = operation.replacement_cells[row];
    if (cell.id == kMaxId || mesh.cell_index_.used(cell.id) || !cell_ids.insert(cell.id).second) {
      return TetEditStatus::kRejectedTopology;
    }
    for (std::size_t ancestor = 0; ancestor < cell.ancestor_cell_ids.size(); ++ancestor) {
      const std::uint64_t id = cell.ancestor_cell_ids[ancestor];
      const auto prior = cell.ancestor_cell_ids.begin() + static_cast<std::ptrdiff_t>(ancestor);
      if (std::find(operation.removed_cell_ids.begin(), operation.removed_cell_ids.end(), id) ==
              operation.removed_cell_ids.end() ||
          std::find(cell.ancestor_cell_ids.begin(), prior, id) != prior) {
        return TetEditStatus::kRejectedTopology;
      }
    }
    const std::int32_t* corners = replacement_vertices.data() + 4 * row;
    for (int corner = 0; corner < 4; ++corner) {
      if (!mesh.live_vertex(corners[corner]) || corners[corner] == edit.removed_vertex) {
        return TetEditStatus::kRejectedTopology;
      }
      for (int other = 0; other < corner; ++other) {
        if (corners[corner] == corners[other]) return TetEditStatus::kRejectedTopology;
      }
      if (cell.facet_sources[static_cast<std::size_t>(corner)] < kNoConstraint) {
        return TetEditStatus::kRejectedTopology;
      }
    }
    const bool positive = edit.overlay == nullptr
                              ? mesh.positive(corners)
                              : mesh.positive_with_relocation(corners, *edit.overlay);
    if (!positive) return TetEditStatus::kInvalidGeometry;
  }
  if (edit.volume_preserving) {
    Dyadic volume;
    for (std::int32_t cell : edit.removed) volume = volume + carrier_determinant(mesh, mesh.tet(cell).v);
    for (std::size_t row = 0; row < created; ++row) {
      volume = volume - carrier_determinant(mesh, replacement_vertices.data() + 4 * row);
    }
    if (!volume.is_zero()) return TetEditStatus::kRejectedTopology;
  }

  // Face pairing over the cavity: each replacement face meets its reversed
  // twin (a replacement cell, or the kept cell across a cavity boundary face)
  // with the same label, or is a new hull face.
  struct Candidate {
    std::array<std::int32_t, 3> oriented{};
    std::int32_t source = kNoConstraint;
    std::uint8_t count = 1;
    std::int32_t region = kNoRegion;
    bool kept = false;
  };
  std::map<SlotFace, Candidate> candidates;
  std::map<SlotFace, std::int32_t> old_inner;
  std::vector<std::int32_t> ghosts;
  for (std::int32_t cell : edit.removed) {
    const Tetrahedron& tet = mesh.tet(cell);
    for (int face = 0; face < 4; ++face) {
      const std::int32_t neighbor = tet.n[face];
      std::int32_t corners[3];
      oriented_facet(tet.v, face, corners);
      const SlotFace key = sorted_slots(corners);
      const Tetrahedron& outside = mesh.tet(neighbor);
      if (removed_cell(neighbor) || is_ghost(outside)) {
        const std::int32_t source = mesh.complex_.constraint(cell, face);
        if (source != kNoConstraint) old_inner.emplace(key, source);
        if (is_ghost(outside)) ghosts.push_back(neighbor);
        continue;
      }
      const int back = neighbor_slot(outside, cell);
      if (back < 0) return TetEditStatus::kRejectedTopology;
      Candidate kept;
      oriented_facet(outside.v, back, kept.oriented.data());
      kept.source = mesh.complex_.constraint(neighbor, back);
      kept.region = mesh.complex_.region(neighbor);
      kept.kept = true;
      candidates.emplace(key, kept);
    }
  }
  std::set<SlotTet> unique_cells;
  for (std::size_t row = 0; row < created; ++row) {
    const TetEditCell& cell = operation.replacement_cells[row];
    const std::int32_t* corners = replacement_vertices.data() + 4 * row;
    SlotTet sorted{corners[0], corners[1], corners[2], corners[3]};
    std::sort(sorted.begin(), sorted.end());
    bool duplicate = !unique_cells.insert(sorted).second;
    collect_star(corners[0], star_scratch_);
    for (std::int32_t kept : star_scratch_) {
      if (removed_cell(kept)) continue;
      SlotTet other{};
      std::copy_n(mesh.tet(kept).v, 4, other.begin());
      std::sort(other.begin(), other.end());
      duplicate = duplicate || other == sorted;
    }
    if (duplicate) return TetEditStatus::kInvalidGeometry;
    for (int face = 0; face < 4; ++face) {
      std::int32_t corners_of_face[3];
      oriented_facet(corners, face, corners_of_face);
      const std::int32_t source = cell.facet_sources[static_cast<std::size_t>(face)];
      const auto [found, inserted] = candidates.emplace(
          sorted_slots(corners_of_face),
          Candidate{{corners_of_face[0], corners_of_face[1], corners_of_face[2]}, source, 1,
                    cell.region, false});
      if (inserted) continue;
      Candidate& previous = found->second;
      const bool reversed =
          FacetKey::of(previous.oriented.data()) == FacetKey::of(corners_of_face).reversed();
      if (previous.count != 1 || !reversed || previous.source != source ||
          (source == kNoConstraint && previous.region != cell.region)) {
        return TetEditStatus::kRejectedTopology;
      }
      previous.count = 2;
    }
  }
  // New hull faces must be labeled, and no kept cell may hold one.
  std::vector<TetMesh::LocalGhostCell> new_ghosts;
  for (const auto& [key, face] : candidates) {
    if (face.count != 1 || face.kept) continue;
    if (face.source == kNoConstraint) return TetEditStatus::kRejectedProtected;
  }
  std::map<SlotFace, std::int32_t> new_inner;
  for (const auto& [key, face] : candidates) {
    if (face.kept) {
      if (face.count != 2) return TetEditStatus::kRejectedTopology;
      continue;
    }
    if (face.source != kNoConstraint) new_inner.emplace(key, face.source);
    if (face.count != 1) continue;
    collect_star(key[0], star_scratch_);
    for (std::int32_t kept : star_scratch_) {
      if (!removed_cell(kept) && vertex_slot(mesh.tet(kept), key[1]) >= 0 &&
          vertex_slot(mesh.tet(kept), key[2]) >= 0) {
        // A face of a kept cell that is no cavity boundary face.
        bool holds = false;
        for (int side = 0; side < 4; ++side) {
          std::int32_t corners[3];
          oriented_facet(mesh.tet(kept).v, side, corners);
          holds = holds || sorted_slots(corners) == key;
        }
        if (holds) return TetEditStatus::kRejectedTopology;
      }
    }
    new_ghosts.push_back({{kGhostVertex, face.oriented[0], face.oriented[1], face.oriented[2]},
                          face.source});
  }
  std::map<SlotFace, std::int32_t> expected_inner;
  for (const auto& [key, source] : old_inner) {
    if (!retired.contains(key)) expected_inner.emplace(key, source);
  }
  for (const auto& [key, record] : new_facets) {
    if (!expected_inner.emplace(key, record.source).second) return TetEditStatus::kRejectedTopology;
  }
  if (new_inner != expected_inner) return TetEditStatus::kRejectedTopology;

  // Every future segment is an edge: the new ones, and the retained ones
  // through an edge of the removed cells.
  std::set<std::uint64_t> replacement_edges;
  for (std::size_t row = 0; row < created; ++row) {
    const std::int32_t* corners = replacement_vertices.data() + 4 * row;
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        replacement_edges.insert(slot_edge(corners[first], corners[second]));
      }
    }
  }
  const auto edge_survives = [&](std::int32_t first, std::int32_t second) {
    return first != edit.removed_vertex && second != edit.removed_vertex &&
           (replacement_edges.contains(slot_edge(first, second)) ||
            edge_cell_outside(mesh, first, second, edit.removed) >= 0);
  };
  for (const TetEditSegment& segment : operation.replacement_segments) {
    const std::int32_t first = mesh.vertex_slot(segment.vertices[0]);
    const std::int32_t second = mesh.vertex_slot(segment.vertices[1]);
    if (!mesh.live_vertex(first) || !mesh.live_vertex(second) || first == second ||
        !edge_survives(first, second)) {
      return TetEditStatus::kRejectedTopology;
    }
  }
  for (std::int32_t cell : edit.removed) {
    const Tetrahedron& tet = mesh.tet(cell);
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const std::int32_t a = tet.v[first];
        const std::int32_t b = tet.v[second];
        if (mesh.vertex_dimension(a) > 1 || mesh.vertex_dimension(b) > 1) continue;
        const auto found =
            boundary_index_.segments.find(sorted_ids(mesh.vertex_id(a), mesh.vertex_id(b)));
        if (found == boundary_index_.segments.end() ||
            std::find(edit.segment_rows.begin(), edit.segment_rows.end(), found->second) !=
                edit.segment_rows.end()) {
          continue;
        }
        if (!edge_survives(a, b)) return TetEditStatus::kRejectedTopology;
      }
    }
  }
  if (mesh.generation_ == kMaxId) return TetEditStatus::kIndexOverflow;

  // Commit: the cells, then the vertex records and the banks.
  std::sort(ghosts.begin(), ghosts.end());
  ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
  std::vector<std::int32_t> removed = edit.removed;
  removed.insert(removed.end(), ghosts.begin(), ghosts.end());
  std::vector<LocalCell> removed_cells;
  removed_cells.reserve(edit.removed.size());
  for (std::int32_t cell : edit.removed) {
    LocalCell vertices{};
    std::copy_n(mesh.tet(cell).v, 4, vertices.begin());
    removed_cells.push_back(vertices);
  }
  // The bank row changes: replacement records overwrite retired rows in
  // order; the rest are appended, and the remaining retired rows are filled
  // by the bank's last row.
  const std::size_t facet_overwrites =
      std::min(edit.facet_rows.size(), operation.replacement_facets.size());
  const std::size_t segment_overwrites =
      std::min(edit.segment_rows.size(), operation.replacement_segments.size());
  reserve_for(mesh.facets_,
              mesh.facets_.size() + operation.replacement_facets.size() - facet_overwrites);
  reserve_for(mesh.segments_,
              mesh.segments_.size() + operation.replacement_segments.size() - segment_overwrites);
  std::vector<FaceIds> retired_facet_keys;
  for (std::size_t row : edit.facet_rows) {
    retired_facet_keys.push_back(sorted_ids(mesh.facets_[row].vertices));
  }
  std::vector<EdgeIds> retired_segment_keys;
  for (std::size_t row : edit.segment_rows) {
    const auto& ends = mesh.segments_[row].vertices;
    retired_segment_keys.push_back(sorted_ids(ends[0], ends[1]));
  }
  std::vector<std::size_t> deleted_facets(edit.facet_rows.begin() +
                                              static_cast<std::ptrdiff_t>(facet_overwrites),
                                          edit.facet_rows.end());
  std::vector<std::size_t> deleted_segments(edit.segment_rows.begin() +
                                                static_cast<std::ptrdiff_t>(segment_overwrites),
                                            edit.segment_rows.end());
  std::sort(deleted_facets.rbegin(), deleted_facets.rend());
  std::sort(deleted_segments.rbegin(), deleted_segments.rend());
  // The caller's segment index changes in place: its retired nodes carry
  // the new keys, so it allocates nothing after the commit.
  std::vector<TetSegmentIndex::node_type> segment_nodes;
  segment_nodes.reserve(retired_segment_keys.size());

  const TetEditStatus committed = mesh.commit_local_cells(
      operation, replacement_vertices, removed, new_ghosts, cleared_sides, edit.overlay);
  if (committed != TetEditStatus::kApplied) return committed;

  if (edit.removed_vertex >= 0) {
    const std::int32_t r = edit.removed_vertex;
    const std::uint64_t id = mesh.vertex_id(r);
    mesh.vertex_live_[static_cast<std::size_t>(r)] = 0;
    mesh.vertex_index_.retire(id);
    mesh.vertex_ancestry_.erase(id);
    mesh.set_vertex_size(r, 0.0);
  }
  for (const TetEditVertex& vertex : operation.changed_vertices) {
    const auto index = static_cast<std::size_t>(mesh.vertex_slot(vertex.id));
    std::copy(vertex.carrier.begin(), vertex.carrier.end(), mesh.points_.begin() + 3 * index);
    mesh.witnesses_[index] = vertex.witness;
    mesh.vertex_ancestry_.erase(vertex.id);
    if (vertex.has_edge_parameter) {
      mesh.vertex_has_edge_parameter_[index] = 1;
      mesh.vertex_edge_endpoints_[index] = vertex.edge_endpoints;
      mesh.vertex_edge_parameters_[index] = vertex.edge_parameter;
    } else if (vertex.edge_witness_supplied) {
      mesh.vertex_has_edge_parameter_[index] = 0;
      mesh.vertex_edge_endpoints_[index] = {};
      mesh.vertex_edge_parameters_[index] = 0.0;
    }
  }
  if (segment_index_ != nullptr) {
    for (const EdgeIds& key : retired_segment_keys) {
      TetSegmentIndex::node_type node = segment_index_->extract(key);
      if (!node.empty()) segment_nodes.push_back(std::move(node));
    }
  }

  // Facet bank.
  bool index_current = true;
  const auto reindex = [&](auto& map, const auto& key, std::size_t row) {
    if (!index_current) return;
    try {
      map[key] = row;
    } catch (const std::bad_alloc&) {
      index_current = false;
    }
  };
  for (const FaceIds& key : retired_facet_keys) boundary_index_.facets.erase(key);
  for (const EdgeIds& key : retired_segment_keys) boundary_index_.segments.erase(key);
  for (std::size_t index = 0; index < operation.replacement_facets.size(); ++index) {
    const TetEditFacet& facet = operation.replacement_facets[index];
    TetSnapshotFacet record;
    record.id = facet.id;
    record.vertices = facet.vertices;
    record.source = facet.source;
    record.orientation = facet.orientation;
    record.protected_facet = false;
    std::size_t row = mesh.facets_.size();
    if (index < facet_overwrites) {
      row = edit.facet_rows[index];
      mesh.facets_[row] = record;
    } else {
      mesh.facets_.push_back(record);
    }
    reindex(boundary_index_.facets, sorted_ids(facet.vertices), row);
    mesh.facet_index_.mark_used(facet.id);
    if (facet.id >= mesh.next_facet_id_) mesh.next_facet_id_ = facet.id + 1;
    for (std::uint64_t id : facet.vertices) {
      std::int8_t& dimension = mesh.vertex_dimension_[static_cast<std::size_t>(mesh.vertex_slot(id))];
      dimension = std::min<std::int8_t>(dimension, 2);
    }
  }
  for (std::size_t row : deleted_facets) {
    const std::size_t last = mesh.facets_.size() - 1;
    if (row != last) {
      mesh.facets_[row] = mesh.facets_[last];
      reindex(boundary_index_.facets, sorted_ids(mesh.facets_[row].vertices), row);
    }
    mesh.facets_.pop_back();
  }

  // Segment bank.
  std::size_t reused = 0;
  const auto index_external = [&](const EdgeIds& key, std::size_t row) {
    if (segment_index_ == nullptr) return;
    const auto found = segment_index_->find(key);
    if (found != segment_index_->end()) {
      found->second = row;
    } else if (reused < segment_nodes.size()) {
      TetSegmentIndex::node_type& node = segment_nodes[reused++];
      node.key() = key;
      node.mapped() = row;
      segment_index_->insert(std::move(node));
    }
  };
  for (std::size_t index = 0; index < operation.replacement_segments.size(); ++index) {
    const TetEditSegment& segment = operation.replacement_segments[index];
    TetSnapshotSegment record;
    record.id = segment.id;
    record.vertices = segment.vertices;
    record.source = segment.source;
    record.protected_segment = false;
    std::size_t row = mesh.segments_.size();
    if (index < segment_overwrites) {
      row = edit.segment_rows[index];
      mesh.segments_[row] = record;
    } else {
      mesh.segments_.push_back(record);
    }
    const EdgeIds key = sorted_ids(segment.vertices[0], segment.vertices[1]);
    reindex(boundary_index_.segments, key, row);
    index_external(key, row);
    mesh.segment_index_.mark_used(segment.id);
    if (segment.id >= mesh.next_segment_id_) mesh.next_segment_id_ = segment.id + 1;
    for (std::uint64_t id : segment.vertices) {
      std::int8_t& dimension = mesh.vertex_dimension_[static_cast<std::size_t>(mesh.vertex_slot(id))];
      dimension = std::min<std::int8_t>(dimension, 1);
    }
  }
  for (std::size_t row : deleted_segments) {
    const std::size_t last = mesh.segments_.size() - 1;
    if (row != last) {
      mesh.segments_[row] = mesh.segments_[last];
      const auto& ends = mesh.segments_[row].vertices;
      const EdgeIds key = sorted_ids(ends[0], ends[1]);
      reindex(boundary_index_.segments, key, row);
      index_external(key, row);
    }
    mesh.segments_.pop_back();
  }
  // The bank indexes are a cache: rebuilt by the next boundary edit.
  if (!index_current) boundary_index_ready_ = false;
  ++mesh.generation_;
  mesh.unmet_records_.clear();

  for (std::size_t index = 0; index < edit.removed.size(); ++index) {
    if (!erase(edit.removed[index], removed_cells[index])) return TetEditStatus::kStaleGeneration;
  }
  for (const TetEditCell& cell : operation.replacement_cells) {
    const std::int32_t slot = mesh.cell_slot(cell.id);
    if (!mesh.complex_.live(slot) || is_ghost(mesh.tet(slot))) {
      return TetEditStatus::kStaleGeneration;
    }
    LocalCell vertices{};
    std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
    if (!insert_logical(mesh, slot, vertices)) return TetEditStatus::kStaleGeneration;
  }
  return TetEditStatus::kApplied;
}

}  // namespace phx::mc::detail
