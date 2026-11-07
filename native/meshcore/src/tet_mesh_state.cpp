//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Atomic construction of persistent tetrahedral state, including its closed
// ghost hull, source banks, labels, and stable logical identities.
#include "tet_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <vector>

#include "facet_coverage.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc {
namespace {

using SortedFace = std::array<std::int32_t, 3>;

SortedFace sorted_face(const std::int32_t* vertices) {
  SortedFace result{vertices[0], vertices[1], vertices[2]};
  std::sort(result.begin(), result.end());
  return result;
}

bool addressable_count(std::size_t count, std::size_t width) {
  return width == 0 || count <= std::numeric_limits<std::size_t>::max() / width;
}

struct FaceOccurrence {
  std::int32_t tet;
  std::int32_t slot;
  std::array<std::int32_t, 3> oriented;
  std::uint8_t count;
};

std::uint64_t edge_key(std::int32_t a, std::int32_t b) {
  const std::uint32_t low = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t high = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(low) << 32) | high;
}


}  // namespace

int32_t TetMesh::set_affine_metric(std::span<const double> linear) {
  if (linear.size() != 9 ||
      !std::all_of(linear.begin(), linear.end(), [](double x) { return std::isfinite(x); })) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::array<double, 9> next;
  std::copy(linear.begin(), linear.end(), next.begin());
  if (next == metric_) return PHX_MC_OK;
  if (!points_.empty()) return PHX_MC_INVALID_ARGUMENT;
  const long double a = next[0], b = next[1], c = next[2];
  const long double d = next[3], e = next[4], f = next[5];
  const long double g = next[6], h = next[7], i = next[8];
  const long double det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);
  if (det == 0 || !std::isfinite(det)) return PHX_MC_INVALID_ARGUMENT;
  const std::array<long double, 9> adj{
      e*i-f*h, c*h-b*i, b*f-c*e, f*g-d*i, a*i-c*g,
      c*d-a*f, d*h-e*g, b*g-a*h, a*e-b*d};
  std::array<double, 9> inverse;
  for (std::size_t k = 0; k < 9; ++k) {
    inverse[k] = static_cast<double>(adj[k] / det);
    if (!std::isfinite(inverse[k])) return PHX_MC_INVALID_ARGUMENT;
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) return PHX_MC_INVALID_ARGUMENT;
  metric_ = next;
  metric_inverse_ = inverse;
  metric_active_ = next != std::array<double, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1};
  metric_orientation_ = det < 0 ? -1 : 1;
  unmet_records_.clear();
  ++generation_;
  return PHX_MC_OK;
}

int32_t TetMesh::build(std::span<const double> points,
                       std::span<const std::int32_t> tetrahedra,
                       std::span<const std::int32_t> regions,
                       std::span<const std::int32_t> faces,
                       std::span<const std::int32_t> face_sources,
                       std::span<const std::int8_t> face_orientations,
                       std::span<const std::int32_t> segments,
                       std::span<const std::int32_t> segment_sources,
                       std::span<const double> protection_radii,
                       const SourceComplexView* source,
                       std::span<const std::int8_t> protected_vertices) {
  if (implicit_source_) return PHX_MC_INVALID_ARGUMENT;
  try {
    plane_refusal_.reset();
    TetMesh staged(policy_);
    staged.metric_ = metric_;
    staged.metric_inverse_ = metric_inverse_;
    staged.metric_active_ = metric_active_;
    staged.metric_orientation_ = metric_orientation_;
    staged.identity_bound_ = identity_bound_;
    staged.source_id_ = source_id_;
    staged.source_revision_ = source_revision_;
    staged.source_map_id_ = source_map_id_;
    staged.frame_id_ = frame_id_;
    staged.pending_implicit_source_ = pending_implicit_source_;
    const std::int32_t status = staged.build_state(
        points, tetrahedra, regions, faces, face_sources, face_orientations,
        segments, segment_sources, protection_radii, source, protected_vertices);
    if (status != PHX_MC_OK) {
      plane_refusal_ = staged.plane_refusal_;
      return status;
    }
    staged.generation_ = 0;
    staged.swap_state(*this);
    unmet_records_.clear();
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return PHX_MC_ALLOCATION_FAILURE;
  } catch (const std::length_error&) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
}

int32_t TetMesh::build_state(std::span<const double> points,
                             std::span<const std::int32_t> tetrahedra,
                             std::span<const std::int32_t> regions,
                             std::span<const std::int32_t> faces,
                             std::span<const std::int32_t> face_sources,
                             std::span<const std::int8_t> face_orientations,
                             std::span<const std::int32_t> segments,
                             std::span<const std::int32_t> segment_sources,
                             std::span<const double> protection_radii,
                             const SourceComplexView* source,
                             std::span<const std::int8_t> protected_vertices) {
  if (points.empty() || points.size() % 3 != 0 || tetrahedra.empty() ||
      tetrahedra.size() % 4 != 0 || faces.size() % 3 != 0 ||
      segments.size() % 2 != 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const std::size_t vertex_count = points.size() / 3;
  const std::size_t cell_count = tetrahedra.size() / 4;
  const std::size_t face_count = faces.size() / 3;
  const std::size_t segment_count = segments.size() / 2;
  if (vertex_count < 4 || vertex_count >= static_cast<std::size_t>(kMaxTetrahedronSlots) ||
      cell_count >= static_cast<std::size_t>(kMaxTetrahedronSlots) ||
      face_sources.size() != face_count || segment_sources.size() != segment_count ||
      (!face_orientations.empty() && face_orientations.size() != face_count) ||
      (!regions.empty() && regions.size() != cell_count) ||
      (!protection_radii.empty() && protection_radii.size() != vertex_count) ||
      (!protected_vertices.empty() && protected_vertices.size() != vertex_count)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!addressable_count(vertex_count, 3) || !addressable_count(cell_count, 4) ||
      !addressable_count(face_count, 3) || !addressable_count(segment_count, 2)) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const std::int32_t coordinate_status = validate_points(
      points.data(), static_cast<std::int64_t>(vertex_count), 3, nullptr);
  if (coordinate_status != PHX_MC_OK) return coordinate_status;

  points_.assign(points.begin(), points.end());
  used_vertex_ids_.reserve(vertex_count);
  vertex_ids_.resize(vertex_count);
  vertex_live_.assign(vertex_count, 1);
  vertex_dimension_.assign(vertex_count, -1);
  protection_radii_.assign(vertex_count, 0.0);
  if (!protection_radii.empty()) {
    std::copy(protection_radii.begin(), protection_radii.end(), protection_radii_.begin());
  }
  protected_vertices_.assign(vertex_count, 0);
  if (!protected_vertices.empty()) {
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      if (protected_vertices[vertex] != 0 && protected_vertices[vertex] != 1) {
        return PHX_MC_INVALID_INPUT;
      }
      protected_vertices_[vertex] = static_cast<std::uint8_t>(protected_vertices[vertex]);
    }
  }
  witnesses_.assign(vertex_count, SourceWitness{});
  vertex_has_edge_parameter_.assign(vertex_count, 0);
  vertex_edge_endpoints_.assign(vertex_count, {});
  vertex_edge_parameters_.assign(vertex_count, 0.0);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    vertex_ids_[vertex] = static_cast<std::uint64_t>(vertex);
    vertex_slots_.emplace(vertex_ids_[vertex], static_cast<std::int32_t>(vertex));
    used_vertex_ids_.insert(vertex_ids_[vertex]);
    if (!std::isfinite(protection_radii_[vertex]) || protection_radii_[vertex] < 0.0) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  next_vertex_id_ = static_cast<std::uint64_t>(vertex_count);

  if (source == nullptr) {
    source_points_.assign(points.begin(), points.end());
    source_faces_.assign(faces.begin(), faces.end());
    source_face_groups_.assign(face_sources.begin(), face_sources.end());
    source_face_ids_.assign(face_sources.begin(), face_sources.end());
    source_face_tolerances_.assign(face_count, 0.0);
    source_segments_.assign(segments.begin(), segments.end());
    source_segment_groups_.assign(segment_sources.begin(), segment_sources.end());
    source_segment_ids_.assign(segment_sources.begin(), segment_sources.end());
    source_segment_tolerances_.assign(segment_count, 0.0);
  } else {
    if (source->points.size() % 3 != 0 || source->faces.size() % 3 != 0 ||
        source->segments.size() % 2 != 0 ||
        source->face_group_rows.size() != source->faces.size() / 3 ||
        source->segment_group_rows.size() != source->segments.size() / 2 ||
        (!source->face_ids.empty() && source->face_ids.size() != source->faces.size() / 3) ||
        (!source->segment_ids.empty() && source->segment_ids.size() != source->segments.size() / 2) ||
        (!source->face_tolerances.empty() &&
         source->face_tolerances.size() != source->faces.size() / 3) ||
        (!source->segment_tolerances.empty() &&
         source->segment_tolerances.size() != source->segments.size() / 2) ||
        (!source->witnesses.empty() && source->witnesses.size() != vertex_count)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const std::int32_t source_status = validate_points(
        source->points.data(), static_cast<std::int64_t>(source->points.size() / 3), 3, nullptr);
    if (source_status != PHX_MC_OK) return source_status;
    source_points_.assign(source->points.begin(), source->points.end());
    source_faces_.assign(source->faces.begin(), source->faces.end());
    source_face_groups_.assign(source->face_group_rows.begin(), source->face_group_rows.end());
    if (source->face_ids.empty()) {
      source_face_ids_.assign(source->face_group_rows.begin(), source->face_group_rows.end());
    } else {
      source_face_ids_.assign(source->face_ids.begin(), source->face_ids.end());
    }
    source_face_tolerances_.assign(source->face_group_rows.size(), 0.0);
    if (!source->face_tolerances.empty()) {
      std::copy(source->face_tolerances.begin(), source->face_tolerances.end(),
                source_face_tolerances_.begin());
    }
    source_segments_.assign(source->segments.begin(), source->segments.end());
    source_segment_groups_.assign(source->segment_group_rows.begin(),
                                  source->segment_group_rows.end());
    if (source->segment_ids.empty()) {
      source_segment_ids_.assign(source->segment_group_rows.begin(), source->segment_group_rows.end());
    } else {
      source_segment_ids_.assign(source->segment_ids.begin(), source->segment_ids.end());
    }
    source_segment_tolerances_.assign(source->segment_group_rows.size(), 0.0);
    if (!source->segment_tolerances.empty()) {
      std::copy(source->segment_tolerances.begin(), source->segment_tolerances.end(),
                source_segment_tolerances_.begin());
    }
    if (!source->witnesses.empty()) {
      std::copy(source->witnesses.begin(), source->witnesses.end(), witnesses_.begin());
    }
    if ((!source->face_planes.empty() &&
         source->face_planes.size() != 9 * source->face_group_rows.size()) ||
        (!source->face_plane_tolerances.empty() &&
         source->face_plane_tolerances.size() != source->face_group_rows.size())) {
      return PHX_MC_INVALID_ARGUMENT;
    }
  }

  const std::size_t source_vertex_count = source_points_.size() / 3;
  for (std::int32_t vertex : source_faces_) {
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= source_vertex_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (std::int32_t vertex : source_segments_) {
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= source_vertex_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (double tolerance : source_face_tolerances_) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) return PHX_MC_INVALID_INPUT;
  }
  for (double tolerance : source_segment_tolerances_) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) return PHX_MC_INVALID_INPUT;
  }
  if (const std::int32_t declared =
          source == nullptr
              ? install_declared_planes({}, {})
              : install_declared_planes(source->face_planes, source->face_plane_tolerances);
      declared != PHX_MC_OK) {
    return declared;
  }
  for (std::size_t row = 0; row < face_count; ++row) {
    const std::int32_t source_group = face_sources[row];
    if (source_group < 0 ||
        (!face_orientations.empty() && face_orientations[row] != 1 &&
         face_orientations[row] != -1)) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (std::size_t row = 0; row < segment_count; ++row) {
    if (segment_sources[row] < 0 || segments[2 * row] == segments[2 * row + 1] ||
        segments[2 * row] < 0 || segments[2 * row + 1] < 0 ||
        static_cast<std::size_t>(segments[2 * row]) >= vertex_count ||
        static_cast<std::size_t>(segments[2 * row + 1]) >= vertex_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }

  complex_.enable_labels();
  complex_.tets.reserve(cell_count + face_count + 8);
  cell_ids_.reserve(cell_count + face_count + 8);
  for (std::size_t row = 0; row < cell_count; ++row) {
    std::int32_t vertices[4];
    std::copy_n(tetrahedra.data() + 4 * row, 4, vertices);
    for (int corner = 0; corner < 4; ++corner) {
      if (vertices[corner] < 0 || static_cast<std::size_t>(vertices[corner]) >= vertex_count) {
        return PHX_MC_INVALID_INPUT;
      }
      for (int other = 0; other < corner; ++other) {
        if (vertices[corner] == vertices[other]) return PHX_MC_INVALID_INPUT;
      }
      vertex_dimension_[static_cast<std::size_t>(vertices[corner])] = 3;
    }
    if (!positive(vertices)) return PHX_MC_INVALID_INPUT;
    const std::int32_t slot = complex_.allocate();
    if (slot < 0) return PHX_MC_INDEX_RANGE_ERROR;
    Tetrahedron tet{{vertices[0], vertices[1], vertices[2], vertices[3]},
                    {-1, -1, -1, -1}, 0, 0};
    complex_.tets[static_cast<std::size_t>(slot)] = tet;
    complex_.set_region(slot, regions.empty() ? kNoRegion : regions[row]);
    const std::uint64_t logical_id = static_cast<std::uint64_t>(row);
    cell_ids_.push_back(logical_id);
    cell_slots_.emplace(logical_id, slot);
    used_cell_ids_.insert(logical_id);
  }
  complex_.finite_count = static_cast<std::int64_t>(cell_count);
  next_cell_id_ = static_cast<std::uint64_t>(cell_count);

  std::map<SortedFace, FaceOccurrence> faces_seen;
  std::vector<FaceOccurrence> boundary_faces;
  for (std::size_t row = 0; row < cell_count; ++row) {
    const std::int32_t slot = static_cast<std::int32_t>(row);
    Tetrahedron& tet = complex_.tets[row];
    for (int face_slot = 0; face_slot < 4; ++face_slot) {
      std::int32_t oriented[3];
      oriented_facet(tet.v, face_slot, oriented);
      const SortedFace key = sorted_face(oriented);
      const auto [it, inserted] = faces_seen.emplace(
          key, FaceOccurrence{slot, face_slot,
                              {oriented[0], oriented[1], oriented[2]}, 1});
      if (!inserted) {
        FaceOccurrence& previous = it->second;
        if (previous.count != 1 ||
            !(FacetKey::of(previous.oriented.data()) ==
              FacetKey::of(oriented).reversed())) {
          return PHX_MC_INVALID_INPUT;
        }
        previous.count = 2;
        tet.n[face_slot] = previous.tet;
        complex_.tets[static_cast<std::size_t>(previous.tet)].n[previous.slot] = slot;
      }
    }
  }
  for (const auto& [key, occurrence] : faces_seen) {
    (void)key;
    if (occurrence.count == 1) boundary_faces.push_back(occurrence);
  }
  if (boundary_faces.empty()) return PHX_MC_INVALID_INPUT;

  std::map<SortedFace, std::size_t> face_rows;
  facets_.reserve(face_count);
  for (std::size_t row = 0; row < face_count; ++row) {
    const std::int32_t* vertices = faces.data() + 3 * row;
    for (int corner = 0; corner < 3; ++corner) {
      if (vertices[corner] < 0 || static_cast<std::size_t>(vertices[corner]) >= vertex_count) {
        return PHX_MC_INVALID_INPUT;
      }
      vertex_dimension_[static_cast<std::size_t>(vertices[corner])] =
          std::min<std::int8_t>(vertex_dimension_[static_cast<std::size_t>(vertices[corner])], 2);
      if (policy_ == BoundaryPolicy::kFixed) {
        protected_vertices_[static_cast<std::size_t>(vertices[corner])] = 1;
      }
    }
    const SortedFace key = sorted_face(vertices);
    if (!faces_seen.contains(key) || !face_rows.emplace(key, row).second) {
      return PHX_MC_INVALID_INPUT;
    }
    const std::int8_t orientation = face_orientations.empty() ? 1 : face_orientations[row];
    TetSnapshotFacet face;
    face.id = next_facet_id_++;
    for (int k = 0; k < 3; ++k) face.vertices[k] = static_cast<std::uint64_t>(vertices[k]);
    used_facet_ids_.insert(face.id);
    face.source = face_sources[row];
    face.orientation = orientation;
    face.protected_facet = policy_ == BoundaryPolicy::kFixed;
    facets_.push_back(face);
  }
  for (const FaceOccurrence& boundary : boundary_faces) {
    if (!face_rows.contains(sorted_face(boundary.oriented.data()))) return PHX_MC_INVALID_INPUT;
  }


  std::set<std::uint64_t> represented_edges;
  for (std::size_t row = 0; row < cell_count; ++row) {
    const Tetrahedron& tet = complex_.tets[row];
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        represented_edges.insert(edge_key(tet.v[a], tet.v[b]));
      }
    }
  }
  segments_.reserve(segment_count);
  for (std::size_t row = 0; row < segment_count; ++row) {
    const std::int32_t a = segments[2 * row];
    const std::int32_t b = segments[2 * row + 1];
    if (!represented_edges.contains(edge_key(a, b))) return PHX_MC_INVALID_INPUT;
    for (std::size_t previous = 0; previous < row; ++previous) {
      if (edge_key(a, b) == edge_key(segments[2 * previous], segments[2 * previous + 1])) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    TetSnapshotSegment segment;
    segment.id = next_segment_id_++;
    segment.vertices = {static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b)};
    segment.source = segment_sources[row];
    used_segment_ids_.insert(segment.id);
    segment.protected_segment = policy_ == BoundaryPolicy::kFixed;
    segments_.push_back(segment);
    vertex_dimension_[static_cast<std::size_t>(a)] =
        std::min<std::int8_t>(vertex_dimension_[static_cast<std::size_t>(a)], 1);
    vertex_dimension_[static_cast<std::size_t>(b)] =
        std::min<std::int8_t>(vertex_dimension_[static_cast<std::size_t>(b)], 1);
    if (policy_ == BoundaryPolicy::kFixed) {
      protected_vertices_[static_cast<std::size_t>(a)] = 1;
      protected_vertices_[static_cast<std::size_t>(b)] = 1;
    }
  }

  const std::size_t finite_count = complex_.tets.size();
  for (const FaceOccurrence& boundary : boundary_faces) {
    const std::int32_t ghost_slot = complex_.allocate();
    if (ghost_slot < 0) return PHX_MC_INDEX_RANGE_ERROR;
    Tetrahedron ghost{{kGhostVertex, boundary.oriented[0], boundary.oriented[1],
                       boundary.oriented[2]},
                      {-1, -1, -1, -1}, 0, 0};
    ghost.n[0] = boundary.tet;
    complex_.tets[static_cast<std::size_t>(ghost_slot)] = ghost;
    complex_.set_region(ghost_slot, kNoRegion);
    complex_.tets[static_cast<std::size_t>(boundary.tet)].n[boundary.slot] = ghost_slot;
  }
  cell_ids_.resize(complex_.tets.size(), std::numeric_limits<std::uint64_t>::max());

  std::map<FacetKey, std::pair<std::int32_t, std::int32_t>> ghost_faces;
  for (std::size_t slot = finite_count; slot < complex_.tets.size(); ++slot) {
    const Tetrahedron& ghost = complex_.tets[slot];
    for (int face_slot = 1; face_slot < 4; ++face_slot) {
      std::int32_t face[3];
      oriented_facet(ghost.v, face_slot, face);
      const FacetKey key = FacetKey::of(face);
      const auto reverse = ghost_faces.find(key.reversed());
      if (reverse != ghost_faces.end()) {
        const auto [other, other_slot] = reverse->second;
        complex_.tets[slot].n[face_slot] = other;
        complex_.tets[static_cast<std::size_t>(other)].n[other_slot] =
            static_cast<std::int32_t>(slot);
        ghost_faces.erase(reverse);
      } else if (!ghost_faces.emplace(key, std::pair{
                     static_cast<std::int32_t>(slot), static_cast<std::int32_t>(face_slot)}).second) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  if (!ghost_faces.empty()) return PHX_MC_INVALID_INPUT;
  for (std::size_t row = 0; row < face_count; ++row) {
    const std::int32_t* vertices = faces.data() + 3 * row;
    const SortedFace key = sorted_face(vertices);
    const FaceOccurrence& occurrence = faces_seen.at(key);
    complex_.set_facet_constraint(occurrence.tet, occurrence.slot, face_sources[row]);
  }
  for (const FaceOccurrence& boundary : boundary_faces) {
    const std::int32_t source_id = complex_.constraint(boundary.tet, boundary.slot);
    if (source_id == kNoConstraint) return PHX_MC_INVALID_INPUT;
  }
  if (pending_implicit_source_) {
    auto pending = std::move(pending_implicit_source_);
    const int32_t status = bind_implicit_source(std::move(pending));
    if (status != PHX_MC_OK) return status;
  }

  // Every boundary vertex of a constrained facet on a declared plane must be
  // admitted on it; the first that is not (facet order, then corner order)
  // refuses the declaration with its record as evidence.
  for (std::size_t row = 0; row < face_count; ++row) {
    if (implicit_source_ &&
        std::binary_search(implicit_source_->groups.begin(), implicit_source_->groups.end(),
                           facets_[row].source)) continue;
    const auto group = declared_groups_.find(facets_[row].source);
    if (group == declared_groups_.end()) continue;
    const std::int32_t plane_row = group->second.rows.front();
    for (std::uint64_t vertex : facets_[row].vertices) {
      const double* carrier = points_.data() + 3 * vertex;
      if (plane_admits(plane_row, carrier)) continue;
      TetPlaneRefusal refusal;
      refusal.vertex = static_cast<std::int64_t>(vertex);
      if (!plane_record(plane_row, carrier, refusal.record)) return PHX_MC_INTERNAL_ERROR;
      refusal.tolerance = source_face_plane_tolerances_[static_cast<std::size_t>(plane_row)];
      plane_refusal_ = refusal;
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (std::size_t row = 0; row < face_count; ++row) {
    const TetSnapshotFacet& face = facets_[row];
    TetEditFacet candidate;
    candidate.id = face.id;
    candidate.vertices = face.vertices;
    candidate.source = face.source;
    candidate.orientation = face.orientation;
    candidate.protected_facet = face.protected_facet;
    if (!source_face_valid(candidate)) return PHX_MC_INVALID_INPUT;
  }
  for (std::size_t row = 0; row < segment_count; ++row) {
    const TetSnapshotSegment& segment = segments_[row];
    TetEditSegment candidate;
    candidate.id = segment.id;
    candidate.vertices = segment.vertices;
    candidate.source = segment.source;
    candidate.protected_segment = segment.protected_segment;
    if (!source_segment_valid(candidate)) return PHX_MC_INVALID_INPUT;
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (vertex_dimension_[vertex] < 0) return PHX_MC_INVALID_INPUT;
    TetEditVertex candidate;
    candidate.id = vertex_ids_[vertex];
    std::copy_n(points_.data() + 3 * vertex, 3, candidate.carrier.begin());
    candidate.witness = witnesses_[vertex];
    candidate.has_edge_parameter = vertex_has_edge_parameter_[vertex] != 0;
    candidate.edge_endpoints = vertex_edge_endpoints_[vertex];
    candidate.edge_parameter = vertex_edge_parameters_[vertex];
    candidate.protection_radius = protection_radii_[vertex];
    candidate.protected_vertex = protected_vertices_[vertex] != 0;
    if (!source_carrier_valid(candidate)) return PHX_MC_INVALID_INPUT;
  }
  if (!audit_state()) return PHX_MC_INTERNAL_ERROR;
  return PHX_MC_OK;
}

namespace {
template <class IdAt>
int32_t install_size_bank(std::unordered_map<std::uint64_t, double>& target,
                          std::size_t count, std::span<const double> sizes, IdAt id_at) {
  if (sizes.empty()) return PHX_MC_OK;
  if (sizes.size() != count) return PHX_MC_INVALID_ARGUMENT;
  for (double size : sizes) {
    if (!std::isfinite(size) || size < 0.0) return PHX_MC_INVALID_INPUT;
  }
  try {
    std::unordered_map<std::uint64_t, double> installed;
    installed.reserve(sizes.size());
    for (std::size_t row = 0; row < sizes.size(); ++row) {
      if (sizes[row] > 0.0) installed.emplace(id_at(row), sizes[row]);
    }
    target.swap(installed);
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return PHX_MC_ALLOCATION_FAILURE;
  }
}
}  // namespace

int32_t TetMesh::install_vertex_sizes(const TetMeshSnapshot& current,
                                      std::span<const double> sizes) {
  return install_size_bank(vertex_sizes_, current.vertices.size(), sizes,
                           [&](std::size_t row) { return current.vertices[row].id; });
}

int32_t TetMesh::install_vertex_sizes(std::span<const std::uint64_t> vertex_ids,
                                      std::span<const double> sizes) {
  return install_size_bank(vertex_sizes_, vertex_ids.size(), sizes,
                           [&](std::size_t row) { return vertex_ids[row]; });
}

void TetMesh::copy_state_to(TetMesh& destination) const {
  destination.metric_ = metric_;
  destination.metric_inverse_ = metric_inverse_;
  destination.metric_active_ = metric_active_;
  destination.metric_orientation_ = metric_orientation_;
  destination.complex_ = complex_;
  destination.points_ = points_;
  destination.vertex_ids_ = vertex_ids_;
  destination.vertex_live_ = vertex_live_;
  destination.vertex_dimension_ = vertex_dimension_;
  destination.protection_radii_ = protection_radii_;
  destination.protected_vertices_ = protected_vertices_;
  destination.witnesses_ = witnesses_;
  destination.vertex_has_edge_parameter_ = vertex_has_edge_parameter_;
  destination.vertex_edge_endpoints_ = vertex_edge_endpoints_;
  destination.vertex_edge_parameters_ = vertex_edge_parameters_;
  destination.cell_ids_ = cell_ids_;
  destination.vertex_ancestry_ = vertex_ancestry_;
  destination.cell_ancestry_ = cell_ancestry_;
  destination.facets_ = facets_;
  destination.segments_ = segments_;
  destination.source_points_ = source_points_;
  destination.source_faces_ = source_faces_;
  destination.source_face_groups_ = source_face_groups_;
  destination.source_face_ids_ = source_face_ids_;
  destination.source_face_tolerances_ = source_face_tolerances_;
  destination.source_segments_ = source_segments_;
  destination.source_segment_groups_ = source_segment_groups_;
  destination.source_segment_ids_ = source_segment_ids_;
  destination.source_segment_tolerances_ = source_segment_tolerances_;
  destination.source_face_planes_ = source_face_planes_;
  destination.source_face_plane_tolerances_ = source_face_plane_tolerances_;
  destination.declared_groups_ = declared_groups_;
  destination.implicit_source_ = implicit_source_;
  destination.pending_implicit_source_ = pending_implicit_source_;
  destination.periodic_orbits_ = periodic_orbits_;
  destination.quotient_cells_ = quotient_cells_;
  destination.periodic_entity_maps_ = periodic_entity_maps_;
  destination.vertex_sizes_ = vertex_sizes_;
  destination.vertex_slots_ = vertex_slots_;
  destination.cell_slots_ = cell_slots_;
  destination.used_vertex_ids_ = used_vertex_ids_;
  destination.used_cell_ids_ = used_cell_ids_;
  destination.used_periodic_orbit_ids_ = used_periodic_orbit_ids_;
  destination.used_quotient_cell_ids_ = used_quotient_cell_ids_;
  destination.used_periodic_lifted_ids_ = used_periodic_lifted_ids_;
  destination.used_periodic_quotient_ids_ = used_periodic_quotient_ids_;
  destination.used_facet_ids_ = used_facet_ids_;
  destination.used_segment_ids_ = used_segment_ids_;
  destination.generation_ = generation_;
  destination.next_vertex_id_ = next_vertex_id_;
  destination.next_cell_id_ = next_cell_id_;
  destination.next_facet_id_ = next_facet_id_;
  destination.next_segment_id_ = next_segment_id_;
  destination.identity_bound_ = identity_bound_;
  destination.source_id_ = source_id_;
  destination.source_revision_ = source_revision_;
  destination.source_map_id_ = source_map_id_;
  destination.frame_id_ = frame_id_;
  destination.periodic_identity_ = periodic_identity_;
}

void TetMesh::swap_state(TetMesh& other) noexcept {
  using std::swap;
  swap(metric_, other.metric_);
  swap(metric_inverse_, other.metric_inverse_);
  swap(metric_active_, other.metric_active_);
  swap(metric_orientation_, other.metric_orientation_);
  swap(complex_, other.complex_);
  swap(points_, other.points_);
  swap(vertex_ids_, other.vertex_ids_);
  swap(vertex_live_, other.vertex_live_);
  swap(vertex_dimension_, other.vertex_dimension_);
  swap(protection_radii_, other.protection_radii_);
  swap(protected_vertices_, other.protected_vertices_);
  swap(witnesses_, other.witnesses_);
  swap(vertex_has_edge_parameter_, other.vertex_has_edge_parameter_);
  swap(vertex_edge_endpoints_, other.vertex_edge_endpoints_);
  swap(vertex_edge_parameters_, other.vertex_edge_parameters_);
  swap(cell_ids_, other.cell_ids_);
  swap(vertex_ancestry_, other.vertex_ancestry_);
  swap(cell_ancestry_, other.cell_ancestry_);
  swap(facets_, other.facets_);
  swap(segments_, other.segments_);
  swap(source_points_, other.source_points_);
  swap(source_faces_, other.source_faces_);
  swap(source_face_groups_, other.source_face_groups_);
  swap(source_face_ids_, other.source_face_ids_);
  swap(source_face_tolerances_, other.source_face_tolerances_);
  swap(source_segments_, other.source_segments_);
  swap(source_segment_groups_, other.source_segment_groups_);
  swap(source_segment_tolerances_, other.source_segment_tolerances_);
  swap(source_face_planes_, other.source_face_planes_);
  swap(source_face_plane_tolerances_, other.source_face_plane_tolerances_);
  swap(declared_groups_, other.declared_groups_);
  swap(implicit_source_, other.implicit_source_);
  swap(pending_implicit_source_, other.pending_implicit_source_);
  swap(source_segment_ids_, other.source_segment_ids_);
  swap(periodic_orbits_, other.periodic_orbits_);
  swap(quotient_cells_, other.quotient_cells_);
  swap(periodic_entity_maps_, other.periodic_entity_maps_);
  swap(vertex_sizes_, other.vertex_sizes_);
  swap(vertex_slots_, other.vertex_slots_);
  swap(cell_slots_, other.cell_slots_);
  swap(used_vertex_ids_, other.used_vertex_ids_);
  swap(used_cell_ids_, other.used_cell_ids_);
  swap(used_periodic_orbit_ids_, other.used_periodic_orbit_ids_);
  swap(used_quotient_cell_ids_, other.used_quotient_cell_ids_);
  swap(used_periodic_lifted_ids_, other.used_periodic_lifted_ids_);
  swap(used_periodic_quotient_ids_, other.used_periodic_quotient_ids_);
  swap(used_facet_ids_, other.used_facet_ids_);
  swap(used_segment_ids_, other.used_segment_ids_);
  swap(generation_, other.generation_);
  swap(next_vertex_id_, other.next_vertex_id_);
  swap(next_cell_id_, other.next_cell_id_);
  swap(next_facet_id_, other.next_facet_id_);
  swap(next_segment_id_, other.next_segment_id_);
  swap(identity_bound_, other.identity_bound_);
  swap(source_id_, other.source_id_);
  swap(source_revision_, other.source_revision_);
  swap(source_map_id_, other.source_map_id_);
  swap(frame_id_, other.frame_id_);
  swap(periodic_identity_, other.periodic_identity_);
}

bool TetMesh::audit_state() const {
  if (!complex_.audit() || points_.size() != 3 * vertex_ids_.size() ||
      vertex_live_.size() != vertex_ids_.size() ||
      vertex_dimension_.size() != vertex_ids_.size() ||
      protection_radii_.size() != vertex_ids_.size() ||
      protected_vertices_.size() != vertex_ids_.size() ||
      vertex_has_edge_parameter_.size() != vertex_ids_.size() ||
      vertex_edge_endpoints_.size() != vertex_ids_.size() ||
      vertex_edge_parameters_.size() != vertex_ids_.size() ||
      witnesses_.size() != vertex_ids_.size() || cell_ids_.size() != complex_.tets.size() ||
      source_faces_.size() % 3 != 0 || source_segments_.size() % 2 != 0 ||
      source_face_groups_.size() != source_faces_.size() / 3 ||
      source_face_ids_.size() != source_faces_.size() / 3 ||
      source_face_tolerances_.size() != source_faces_.size() / 3 ||
      source_segment_groups_.size() != source_segments_.size() / 2 ||
      source_segment_ids_.size() != source_segments_.size() / 2 ||
      source_segment_tolerances_.size() != source_segments_.size() / 2 ||
      source_face_planes_.size() != 3 * source_faces_.size() ||
      source_face_plane_tolerances_.size() != source_faces_.size() / 3 ||
      !std::all_of(vertex_ancestry_.begin(), vertex_ancestry_.end(), [&](const auto& item) {
        if (!used_vertex_ids_.contains(item.first) || item.second.empty()) return false;
        return std::all_of(item.second.begin(), item.second.end(), [&](const TetVertexAncestry& ancestor) {
          return used_vertex_ids_.contains(ancestor.source_vertex_id) &&
                 std::isfinite(ancestor.weight);
        });
      }) ||
      !std::all_of(cell_ancestry_.begin(), cell_ancestry_.end(), [&](const auto& item) {
        if (!used_cell_ids_.contains(item.first) || !item.second.ancestor_cell_ids ||
            item.second.edit_kind < 0 ||
            item.second.edit_kind > static_cast<std::int32_t>(TetEditKind::kRetriangulate)) {
          return false;
        }
        return std::all_of(item.second.ancestor_cell_ids->begin(),
                           item.second.ancestor_cell_ids->end(), [&](std::uint64_t ancestor) {
          return used_cell_ids_.contains(ancestor);
        });
      })) {
    return false;
  }
  std::size_t live_vertices = 0;
  for (std::size_t slot = 0; slot < vertex_ids_.size(); ++slot) {
    if (vertex_live_[slot] == 0) continue;
    ++live_vertices;
    const auto found = vertex_slots_.find(vertex_ids_[slot]);
    if (found == vertex_slots_.end() || found->second != static_cast<std::int32_t>(slot) ||
        !used_vertex_ids_.contains(vertex_ids_[slot])) return false;
  }
  std::size_t live_cells = 0;
  for (std::size_t slot = 0; slot < complex_.tets.size(); ++slot) {
    const std::int32_t local = static_cast<std::int32_t>(slot);
    if (!complex_.live(local)) continue;
    const Tetrahedron& cell = complex_.tets[slot];
    for (std::int32_t vertex : cell.v) {
      if (vertex != kGhostVertex && !live_vertex(vertex)) return false;
    }
    if (is_ghost(cell)) continue;
    ++live_cells;
    const auto found = cell_slots_.find(cell_ids_[slot]);
    if (found == cell_slots_.end() || found->second != local ||
        !used_cell_ids_.contains(cell_ids_[slot])) return false;
  }
  if (live_vertices != vertex_slots_.size() || live_cells != cell_slots_.size()) return false;
  for (const TetSnapshotFacet& facet : facets_) {
    if (!used_facet_ids_.contains(facet.id)) return false;
    for (std::uint64_t vertex : facet.vertices) if (vertex_slot(vertex) < 0) return false;
  }
  for (const TetSnapshotSegment& segment : segments_) {
    if (!used_segment_ids_.contains(segment.id)) return false;
    if (vertex_slot(segment.vertices[0]) < 0 || vertex_slot(segment.vertices[1]) < 0) return false;
  }
  return true;
}

}  // namespace phx::mc
