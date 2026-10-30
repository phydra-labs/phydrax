//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// One complete, source-aware stellar-refinement pass. Any remaining criterion
// violation is reported as UNRESOLVED rather than hidden behind a pass quota.
#include "refine3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "cavity.hpp"
#include "phydrax_meshcore.h"
#include "exact_scalar.hpp"

namespace phx::mc {
namespace {

bool tetrahedron_geometry(const TetMesh& mesh, const TetSnapshotCell& cell,
                          std::array<std::int32_t, 4>& slots,
                          geometry::TetrahedronShape& shape) {
  const double* points[4];
  for (int corner = 0; corner < 4; ++corner) {
    slots[corner] = mesh.vertex_slot(cell.vertices[corner]);
    if (!mesh.live_vertex(slots[corner])) return false;
    points[corner] = mesh.point(slots[corner]);
  }
  return geometry::tetrahedron_shape(points, shape);
}

bool mean_target_size(const std::array<std::uint64_t, 4>& vertices,
                      const std::unordered_map<std::uint64_t, double>& sizes,
                      double& target) {
  std::array<double, 4> values{};
  double scale = 0.0;
  for (std::size_t corner = 0; corner < vertices.size(); ++corner) {
    const auto found = sizes.find(vertices[corner]);
    if (found == sizes.end() || found->second == 0.0) return false;
    values[corner] = found->second;
    scale = std::max(scale, found->second);
  }
  double normalized_sum = 0.0;
  for (double value : values) normalized_sum += value / scale;
  target = std::min(scale, scale * (0.25 * normalized_sum));
  return true;
}

bool violates(const TetMesh& mesh, const TetSnapshotCell& cell,
              double radius_edge_bound,
              const std::unordered_map<std::uint64_t, double>& sizes) {
  std::array<std::int32_t, 4> slots{};
  geometry::TetrahedronShape shape;
  if (!tetrahedron_geometry(mesh, cell, slots, shape)) return true;
  if (shape.radius_edge_ratio > radius_edge_bound) return true;
  double target = 0.0;
  return mean_target_size(cell.vertices, sizes, target) &&
         shape.circumradius > target;
}

void collect_unmet(const TetMesh& mesh, const TetMeshSnapshot& snapshot,
                   double radius_edge_bound,
                   const std::unordered_map<std::uint64_t, double>& sizes,
                   std::vector<TetUnmetRecord>& records,
                   std::int64_t& unmet_cells) {
  records.clear();
  if (snapshot.cells.size() <= records.max_size() / 2) {
    records.reserve(2 * snapshot.cells.size());
  }
  unmet_cells = 0;
  for (const TetSnapshotCell& cell : snapshot.cells) {
    std::array<std::int32_t, 4> slots{};
    geometry::TetrahedronShape shape;
    bool cell_unmet = false;
    if (!tetrahedron_geometry(mesh, cell, slots, shape)) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_VALIDITY,
                         PHX_MC_TET_UNMET_NONREPRESENTABLE, 0.0});
      ++unmet_cells;
      continue;
    }
    if (shape.radius_edge_ratio > radius_edge_bound) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_RADIUS_EDGE,
                         PHX_MC_TET_UNMET_NO_IMPROVEMENT,
                         shape.radius_edge_ratio});
      cell_unmet = true;
    }
    double target = 0.0;
    if (mean_target_size(cell.vertices, sizes, target) &&
        shape.circumradius > target) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_SIZE,
                         PHX_MC_TET_UNMET_NO_IMPROVEMENT,
                         shape.circumradius / target});
      cell_unmet = true;
    }
    if (cell_unmet) ++unmet_cells;
  }
}

}  // namespace

std::int32_t refine_mesh(TetMesh& mesh, double radius_edge_bound,
                         std::span<const double> vertex_sizes,
                         std::int64_t& inserted_vertices,
                         std::int64_t& unmet_cells) {
  inserted_vertices = 0;
  unmet_cells = 0;
  if (!std::isfinite(radius_edge_bound) || radius_edge_bound <= 0.0) {
    return PHX_MC_INVALID_ARGUMENT;
  }

  TetMeshSnapshot initial;
  if (mesh.snapshot(initial) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  if (!vertex_sizes.empty() && vertex_sizes.size() != initial.vertices.size()) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::unordered_map<std::uint64_t, double> sizes;
  sizes.reserve(vertex_sizes.size());
  for (std::size_t row = 0; row < vertex_sizes.size(); ++row) {
    const double size = vertex_sizes[row];
    if (!std::isfinite(size) || size < 0.0) return PHX_MC_INVALID_INPUT;
    sizes.emplace(initial.vertices[row].id, size);
  }

  using EdgeKey = std::array<std::uint64_t, 2>;
  using FaceKey = std::array<std::uint64_t, 3>;
  const auto sorted_edge = [](std::uint64_t first, std::uint64_t second) {
    return first < second ? EdgeKey{first, second} : EdgeKey{second, first};
  };
  const auto sorted_face = [](FaceKey face) {
    std::sort(face.begin(), face.end());
    return face;
  };
  const auto same_point = [](const double* first, const double* second) {
    return first[0] == second[0] && first[1] == second[1] &&
           first[2] == second[2];
  };
  // Hard size fields are stored in regular-tetrahedron circumradius units.
  const double radius_to_edge = 4.0 / std::sqrt(6.0);
  const auto boundary_target = [&](const EdgeKey& edge, double& mean_size,
                                   double& edge_target) {
    const auto first = sizes.find(edge[0]);
    const auto second = sizes.find(edge[1]);
    if (first == sizes.end() || second == sizes.end() ||
        first->second <= 0.0 || second->second <= 0.0) {
      return false;
    }
    const double scale = std::max(first->second, second->second);
    mean_size =
        scale * 0.5 * (first->second / scale + second->second / scale);
    edge_target = mean_size * radius_to_edge;
    return edge_target > 0.0 && !std::isnan(edge_target);
  };
  const auto make_boundary_witness =
      [&](const EdgeKey& edge, const std::array<double, 3>& carrier,
          TetEditVertex& vertex) {
        const auto install = [&](SourceStratum stratum, std::size_t row,
                                 const double* const corners[3]) {
          if (row > static_cast<std::size_t>(
                        std::numeric_limits<std::int32_t>::max())) {
            return false;
          }
          double parameters[2]{};
          source_locator(corners, stratum, carrier.data(), parameters, false);
          if (!source_parameters_valid(stratum, parameters)) return false;
          double source_position[3]{};
          double deviation = 0.0;
          if (!source_carrier(corners, stratum, parameters, source_position,
                              deviation) ||
              deviation != 0.0 ||
              !same_point(source_position, carrier.data())) {
            return false;
          }
          vertex.witness.stratum = stratum;
          vertex.witness.entity = static_cast<std::int32_t>(row);
          vertex.witness.parameters[0] = parameters[0];
          vertex.witness.parameters[1] = parameters[1];
          vertex.witness.deviation = deviation;
          return true;
        };

        const TetSnapshotSegment* source_segment = nullptr;
        for (const TetSnapshotSegment& segment : mesh.segments_) {
          if (sorted_edge(segment.vertices[0], segment.vertices[1]) != edge) {
            continue;
          }
          if (segment.protected_segment ||
              (source_segment != nullptr &&
               source_segment->source != segment.source)) {
            return false;
          }
          source_segment = &segment;
        }
        if (source_segment != nullptr) {
          for (std::size_t row = 0;
               row < mesh.source_segment_groups_.size(); ++row) {
            if (mesh.source_segment_groups_[row] != source_segment->source) {
              continue;
            }
            const std::int32_t first =
                mesh.source_segments_[2 * row];
            const std::int32_t second =
                mesh.source_segments_[2 * row + 1];
            const double* corners[3]{
                mesh.source_points_.data() + 3 * static_cast<std::size_t>(first),
                mesh.source_points_.data() + 3 * static_cast<std::size_t>(second),
                nullptr};
            if (install(SourceStratum::kSegment, row, corners)) return true;
          }
          return false;
        }

        std::set<std::int32_t> source_groups;
        for (const TetSnapshotFacet& facet : mesh.facets_) {
          if (facet.source < 0) continue;
          bool has_first = false;
          bool has_second = false;
          for (std::uint64_t vertex_id : facet.vertices) {
            has_first = has_first || vertex_id == edge[0];
            has_second = has_second || vertex_id == edge[1];
          }
          if (has_first && has_second) source_groups.insert(facet.source);
        }
        for (std::int32_t group : source_groups) {
          for (std::size_t row = 0; row < mesh.source_face_groups_.size();
               ++row) {
            if (mesh.source_face_groups_[row] != group) continue;
            const std::int32_t first = mesh.source_faces_[3 * row];
            const std::int32_t second = mesh.source_faces_[3 * row + 1];
            const std::int32_t third = mesh.source_faces_[3 * row + 2];
            const double* corners[3]{
                mesh.source_points_.data() + 3 * static_cast<std::size_t>(first),
                mesh.source_points_.data() + 3 * static_cast<std::size_t>(second),
                mesh.source_points_.data() + 3 * static_cast<std::size_t>(third)};
            if (install(SourceStratum::kFacet, row, corners)) return true;
          }
        }
        return false;
      };
  const auto split_boundary_edge =
      [&](const EdgeKey& edge, std::uint64_t& inserted_id) {
        if (mesh.policy_ == BoundaryPolicy::kFixed) {
          return TetEditStatus::kRejectedProtected;
        }
        if (mesh.periodic_identity_.bound) {
          return TetEditStatus::kRejectedTopology;
        }
        const std::int32_t first_slot = mesh.vertex_slot(edge[0]);
        const std::int32_t second_slot = mesh.vertex_slot(edge[1]);
        if (!mesh.live_vertex(first_slot) || !mesh.live_vertex(second_slot)) {
          return TetEditStatus::kRejectedTopology;
        }
        if (mesh.next_vertex_id_ == std::numeric_limits<std::uint64_t>::max()) {
          return TetEditStatus::kIndexOverflow;
        }
        const double* first_point = mesh.point(first_slot);
        const double* second_point = mesh.point(second_slot);
        const Dyadic half(0.5);
        const Dyadic one(1.0);
        std::array<double, 3> midpoint{};
        for (int axis = 0; axis < 3; ++axis) {
          midpoint[static_cast<std::size_t>(axis)] = exact::rounded_quotient(
              (Dyadic(first_point[axis]) + Dyadic(second_point[axis])) * half,
              one);
        }
        int dominant = 0;
        for (int axis = 1; axis < 3; ++axis) {
          if (std::abs(second_point[axis] - first_point[axis]) >
              std::abs(second_point[dominant] - first_point[dominant])) {
            dominant = axis;
          }
        }
        const double low =
            std::min(first_point[dominant], second_point[dominant]);
        const double high =
            std::max(first_point[dominant], second_point[dominant]);
        if (!(low < midpoint[static_cast<std::size_t>(dominant)] &&
              midpoint[static_cast<std::size_t>(dominant)] < high)) {
          return TetEditStatus::kRejectedTopology;
        }

        TetEditVertex inserted;
        inserted.id = mesh.next_vertex_id_;
        inserted.carrier = midpoint;
        inserted.ancestry.push_back({edge[0], {}, 0.5});
        inserted.ancestry.push_back({edge[1], {}, 0.5});
        if (!make_boundary_witness(edge, midpoint, inserted)) {
          return TetEditStatus::kRejectedTopology;
        }

        struct ParentCell {
          std::uint64_t id;
          std::int32_t region;
          std::array<std::uint64_t, 4> vertices;
        };
        std::vector<ParentCell> parents;
        parents.reserve(static_cast<std::size_t>(mesh.complex_.finite_count));
        std::map<FaceKey, std::int32_t> source_by_child_face;
        const auto record_source =
            [&](FaceKey face, std::int32_t source) {
              const auto [found, inserted_face] =
                  source_by_child_face.emplace(sorted_face(face), source);
              return inserted_face || found->second == source;
            };
        for (std::size_t row = 0; row < mesh.complex_.tets.size(); ++row) {
          const std::int32_t cell = static_cast<std::int32_t>(row);
          if (!mesh.complex_.live(cell) || is_ghost(mesh.tet(cell))) continue;
          const Tetrahedron& tetrahedron = mesh.tet(cell);
          ParentCell parent{mesh.cell_id(cell), mesh.complex_.region(cell), {}};
          bool has_first = false;
          bool has_second = false;
          for (int corner = 0; corner < 4; ++corner) {
            parent.vertices[static_cast<std::size_t>(corner)] =
                mesh.vertex_id(tetrahedron.v[corner]);
            has_first = has_first || parent.vertices[corner] == edge[0];
            has_second = has_second || parent.vertices[corner] == edge[1];
          }
          if (!has_first || !has_second) continue;
          parents.push_back(parent);
          for (int face_slot = 0; face_slot < 4; ++face_slot) {
            const std::int32_t source =
                mesh.complex_.constraint(cell, face_slot);
            if (source < 0) continue;
            std::int32_t oriented[3];
            oriented_facet(tetrahedron.v, face_slot, oriented);
            FaceKey face{
                mesh.vertex_id(oriented[0]), mesh.vertex_id(oriented[1]),
                mesh.vertex_id(oriented[2])};
            bool face_has_first = false;
            bool face_has_second = false;
            std::uint64_t third = 0;
            for (std::uint64_t vertex_id : face) {
              face_has_first = face_has_first || vertex_id == edge[0];
              face_has_second = face_has_second || vertex_id == edge[1];
              if (vertex_id != edge[0] && vertex_id != edge[1]) {
                third = vertex_id;
              }
            }
            if (face_has_first && face_has_second) {
              if (!record_source({edge[0], inserted.id, third}, source) ||
                  !record_source({inserted.id, edge[1], third}, source)) {
                return TetEditStatus::kRejectedTopology;
              }
            } else if (!record_source(face, source)) {
              return TetEditStatus::kRejectedTopology;
            }
          }
        }
        if (parents.empty()) return TetEditStatus::kRejectedTopology;

        const std::uint64_t max_id =
            std::numeric_limits<std::uint64_t>::max();
        if (parents.size() >
                static_cast<std::size_t>(max_id / 2) ||
            2 * parents.size() >
                static_cast<std::size_t>(max_id - mesh.next_cell_id_)) {
          return TetEditStatus::kIndexOverflow;
        }
        const std::size_t replacement_cell_count = 2 * parents.size();
        TetEdit operation;
        operation.kind = TetEditKind::kSplit;
        operation.split_edges.push_back(edge);
        operation.changed_vertices.push_back(inserted);
        operation.removed_cell_ids.reserve(parents.size());
        operation.replacement_cells.reserve(replacement_cell_count);
        for (const ParentCell& parent : parents) {
          operation.removed_cell_ids.push_back(parent.id);
          std::array<std::uint64_t, 4> children[2]{
              parent.vertices, parent.vertices};
          for (std::uint64_t& vertex_id : children[0]) {
            if (vertex_id == edge[1]) vertex_id = inserted.id;
          }
          for (std::uint64_t& vertex_id : children[1]) {
            if (vertex_id == edge[0]) vertex_id = inserted.id;
          }
          for (const auto& child_vertices : children) {
            TetEditCell child;
            child.id = mesh.next_cell_id_ +
                       static_cast<std::uint64_t>(
                           operation.replacement_cells.size());
            child.vertices = child_vertices;
            child.region = parent.region;
            child.ancestor_cell_ids.push_back(parent.id);
            for (int face_slot = 0; face_slot < 4; ++face_slot) {
              FaceKey face{};
              int offset = 0;
              for (int corner = 0; corner < 4; ++corner) {
                if (corner == face_slot) continue;
                face[static_cast<std::size_t>(offset++)] =
                    child.vertices[static_cast<std::size_t>(corner)];
              }
              const auto source =
                  source_by_child_face.find(sorted_face(face));
              if (source != source_by_child_face.end()) {
                child.facet_sources[static_cast<std::size_t>(face_slot)] =
                    source->second;
              }
            }
            operation.replacement_cells.push_back(std::move(child));
          }
        }

        std::size_t split_facet_count = 0;
        for (const TetSnapshotFacet& facet : mesh.facets_) {
          if (sorted_edge(facet.vertices[0], facet.vertices[1]) == edge ||
              sorted_edge(facet.vertices[1], facet.vertices[2]) == edge ||
              sorted_edge(facet.vertices[2], facet.vertices[0]) == edge) {
            if (facet.protected_facet) {
              return TetEditStatus::kRejectedProtected;
            }
            ++split_facet_count;
          }
        }
        if (split_facet_count == 0) return TetEditStatus::kRejectedTopology;
        std::size_t split_segment_count = 0;
        for (const TetSnapshotSegment& segment : mesh.segments_) {
          if (sorted_edge(segment.vertices[0], segment.vertices[1]) == edge) {
            if (segment.protected_segment) {
              return TetEditStatus::kRejectedProtected;
            }
            ++split_segment_count;
          }
        }
        const std::size_t replacement_facet_count = 2 * split_facet_count;
        const std::size_t replacement_segment_count = 2 * split_segment_count;
        if (replacement_facet_count >
                static_cast<std::size_t>(max_id - mesh.next_facet_id_) ||
            replacement_segment_count >
                static_cast<std::size_t>(max_id - mesh.next_segment_id_)) {
          return TetEditStatus::kIndexOverflow;
        }
        operation.removed_facet_ids.reserve(split_facet_count);
        operation.replacement_facets.reserve(replacement_facet_count);
        std::uint64_t next_facet = mesh.next_facet_id_;
        for (const TetSnapshotFacet& facet : mesh.facets_) {
          bool split = false;
          for (int offset = 0; offset < 3; ++offset) {
            const std::uint64_t first = facet.vertices[offset];
            const std::uint64_t second = facet.vertices[(offset + 1) % 3];
            if (sorted_edge(first, second) != edge) continue;
            const std::uint64_t third = facet.vertices[(offset + 2) % 3];
            operation.removed_facet_ids.push_back(facet.id);
            TetEditFacet first_child;
            first_child.id = next_facet++;
            first_child.vertices = {first, inserted.id, third};
            first_child.source = facet.source;
            first_child.orientation = facet.orientation;
            TetEditFacet second_child;
            second_child.id = next_facet++;
            second_child.vertices = {inserted.id, second, third};
            second_child.source = facet.source;
            second_child.orientation = facet.orientation;
            operation.replacement_facets.push_back(std::move(first_child));
            operation.replacement_facets.push_back(std::move(second_child));
            split = true;
            break;
          }
          if (!split &&
              (sorted_edge(facet.vertices[0], facet.vertices[1]) == edge ||
               sorted_edge(facet.vertices[1], facet.vertices[2]) == edge ||
               sorted_edge(facet.vertices[2], facet.vertices[0]) == edge)) {
            return TetEditStatus::kRejectedTopology;
          }
        }
        operation.removed_segment_ids.reserve(split_segment_count);
        operation.replacement_segments.reserve(replacement_segment_count);
        std::uint64_t next_segment = mesh.next_segment_id_;
        for (const TetSnapshotSegment& segment : mesh.segments_) {
          if (sorted_edge(segment.vertices[0], segment.vertices[1]) != edge) {
            continue;
          }
          operation.removed_segment_ids.push_back(segment.id);
          TetEditSegment first_child;
          first_child.id = next_segment++;
          first_child.vertices = {segment.vertices[0], inserted.id};
          first_child.source = segment.source;
          TetEditSegment second_child;
          second_child.id = next_segment++;
          second_child.vertices = {inserted.id, segment.vertices[1]};
          second_child.source = segment.source;
          operation.replacement_segments.push_back(std::move(first_child));
          operation.replacement_segments.push_back(std::move(second_child));
        }
        if (inserted_vertices == std::numeric_limits<std::int64_t>::max()) {
          return TetEditStatus::kIndexOverflow;
        }
        inserted_id = inserted.id;
        TetEditBatch batch{mesh.generation_, mesh.source_revision_,
                           mesh.frame_id_, mesh.identity_bound_,
                           {std::move(operation)}, mesh.source_id_,
                           mesh.source_map_id_, mesh.periodic_identity_};
        std::vector<TetEditResult> results;
        const TetEditStatus status = mesh.apply_edit_batch(batch, results);
        if (status != TetEditStatus::kApplied) return status;
        if (results.size() != 1) return TetEditStatus::kRejectedTopology;
        return results.front().status;
      };
  std::set<EdgeKey> blocked_boundary_edges;
  if (!sizes.empty()) {
    for (;;) {
      std::set<EdgeKey> boundary_edges;
      for (const TetSnapshotFacet& facet : mesh.facets_) {
        boundary_edges.insert(sorted_edge(facet.vertices[0], facet.vertices[1]));
        boundary_edges.insert(sorted_edge(facet.vertices[1], facet.vertices[2]));
        boundary_edges.insert(sorted_edge(facet.vertices[2], facet.vertices[0]));
      }
      EdgeKey selected_edge{};
      double selected_size = 0.0;
      double selected_ratio = 1.0;
      bool found_candidate = false;
      for (const EdgeKey& edge : boundary_edges) {
        if (blocked_boundary_edges.contains(edge)) continue;
        double mean_size = 0.0;
        double edge_target = 0.0;
        if (!boundary_target(edge, mean_size, edge_target)) continue;
        const std::int32_t first_slot = mesh.vertex_slot(edge[0]);
        const std::int32_t second_slot = mesh.vertex_slot(edge[1]);
        if (!mesh.live_vertex(first_slot) || !mesh.live_vertex(second_slot)) {
          return PHX_MC_INTERNAL_ERROR;
        }
        const double* first = mesh.point(first_slot);
        const double* second = mesh.point(second_slot);
        const double length = std::hypot(
            std::hypot(second[0] - first[0], second[1] - first[1]),
            second[2] - first[2]);
        const double ratio = length / edge_target;
        if (ratio > selected_ratio) {
          selected_edge = edge;
          selected_size = mean_size;
          selected_ratio = ratio;
          found_candidate = true;
        }
      }
      if (!found_candidate) break;
      std::uint64_t inserted_id = 0;
      const TetEditStatus status =
          split_boundary_edge(selected_edge, inserted_id);
      if (status == TetEditStatus::kAllocationFailure) {
        return PHX_MC_ALLOCATION_FAILURE;
      }
      if (status == TetEditStatus::kIndexOverflow) {
        return PHX_MC_INDEX_RANGE_ERROR;
      }
      if (status == TetEditStatus::kStaleGeneration) return PHX_MC_UNRESOLVED;
      if (status == TetEditStatus::kApplied) {
        if (!sizes.emplace(inserted_id, selected_size).second) {
          return PHX_MC_INTERNAL_ERROR;
        }
        ++inserted_vertices;
        continue;
      }
      blocked_boundary_edges.insert(selected_edge);
    }
  }

  TetMeshSnapshot after_boundary_splits;
  if (mesh.snapshot(after_boundary_splits) != TetEditStatus::kApplied) {
    return PHX_MC_INTERNAL_ERROR;
  }
  const std::vector<std::uint64_t> candidate_cell_ids = [&] {
    std::vector<std::uint64_t> ids;
    ids.reserve(after_boundary_splits.cells.size());
    for (const TetSnapshotCell& cell : after_boundary_splits.cells) {
      ids.push_back(cell.id);
    }
    return ids;
  }();
  for (std::uint64_t cell_id : candidate_cell_ids) {
    const std::int32_t parent_slot = mesh.cell_slot(cell_id);
    if (parent_slot < 0) continue;
    TetSnapshotCell parent;
    parent.id = cell_id;
    parent.region = mesh.complex_.region(parent_slot);
    std::array<std::int32_t, 4> parent_slots{};
    for (int corner = 0; corner < 4; ++corner) {
      parent_slots[corner] = mesh.tet(parent_slot).v[corner];
      parent.vertices[corner] = mesh.vertex_id(parent_slots[corner]);
    }
    if (!violates(mesh, parent, radius_edge_bound, sizes)) continue;

    if (mesh.next_vertex_id_ == std::numeric_limits<std::uint64_t>::max() ||
        mesh.next_cell_id_ > std::numeric_limits<std::uint64_t>::max() - 4) {
      continue;
    }
    const std::uint64_t center_id = mesh.next_vertex_id_;
    double center_size = 0.0;
    const bool has_center_size = mean_target_size(parent.vertices, sizes, center_size);
    TetEditVertex center;
    center.id = center_id;
    std::array<Dyadic, 3> exact_center{};
    const Dyadic weight(0.25);
    const Dyadic one(1.0);
    center.ancestry.reserve(4);
    for (int corner = 0; corner < 4; ++corner) {
      center.ancestry.push_back({parent.vertices[corner], {}, 0.25});
      for (int axis = 0; axis < 3; ++axis) {
        exact_center[static_cast<std::size_t>(axis)] =
            exact_center[static_cast<std::size_t>(axis)] +
            Dyadic(mesh.point(parent_slots[corner])[axis]) * weight;
      }
    }
    for (int axis = 0; axis < 3; ++axis) {
      center.carrier[static_cast<std::size_t>(axis)] =
          exact::rounded_quotient(exact_center[static_cast<std::size_t>(axis)], one);
    }

    TetEdit operation;
    operation.kind = TetEditKind::kRefineCell;
    operation.removed_cell_ids.push_back(cell_id);
    operation.changed_vertices.push_back(center);
    operation.replacement_cells.reserve(4);
    for (int face = 0; face < 4; ++face) {
      std::int32_t oriented[3];
      oriented_facet(mesh.tet(parent_slot).v, face, oriented);
      TetEditCell child;
      child.id = mesh.next_cell_id_ + static_cast<std::uint64_t>(face);
      child.vertices = {mesh.vertex_id(oriented[0]), mesh.vertex_id(oriented[1]),
                        mesh.vertex_id(oriented[2]), center_id};
      child.facet_sources[3] = mesh.complex_.constraint(parent_slot, face);
      child.region = mesh.complex_.region(parent_slot);
      child.ancestor_cell_ids.push_back(cell_id);
      operation.replacement_cells.push_back(child);
    }

    TetEditBatch batch{mesh.generation_, mesh.source_revision_, mesh.frame_id_,
                       mesh.identity_bound_, {std::move(operation)},
                       mesh.source_id_, mesh.source_map_id_, mesh.periodic_identity_};
    std::vector<TetEditResult> result;
    const TetEditStatus status = mesh.apply_edit_batch(batch, result);
    if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
    if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
    if (status == TetEditStatus::kStaleGeneration) return PHX_MC_UNRESOLVED;
    if (status == TetEditStatus::kApplied && !result.empty() &&
        result.front().status == TetEditStatus::kApplied) {
      ++inserted_vertices;
      if (has_center_size) sizes.emplace(center_id, center_size);
    }
  }

  TetMeshSnapshot final_state;
  if (mesh.snapshot(final_state) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  std::vector<TetUnmetRecord> records;
  collect_unmet(mesh, final_state, radius_edge_bound, sizes, records, unmet_cells);
  std::map<std::uint64_t, double> boundary_unmet;
  for (const EdgeKey& edge : blocked_boundary_edges) {
    bool remains_constrained = false;
    for (const TetSnapshotFacet& facet : mesh.facets_) {
      if (sorted_edge(facet.vertices[0], facet.vertices[1]) == edge ||
          sorted_edge(facet.vertices[1], facet.vertices[2]) == edge ||
          sorted_edge(facet.vertices[2], facet.vertices[0]) == edge) {
        remains_constrained = true;
        break;
      }
    }
    if (!remains_constrained) continue;
    double mean_size = 0.0;
    double edge_target = 0.0;
    if (!boundary_target(edge, mean_size, edge_target)) continue;
    const std::int32_t first_slot = mesh.vertex_slot(edge[0]);
    const std::int32_t second_slot = mesh.vertex_slot(edge[1]);
    if (!mesh.live_vertex(first_slot) || !mesh.live_vertex(second_slot)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const double* first = mesh.point(first_slot);
    const double* second = mesh.point(second_slot);
    const double length = std::hypot(
        std::hypot(second[0] - first[0], second[1] - first[1]),
        second[2] - first[2]);
    const double ratio = length / edge_target;
    if (!(ratio > 1.0)) continue;
    bool incident = false;
    for (const TetSnapshotCell& cell : final_state.cells) {
      bool has_first = false;
      bool has_second = false;
      for (std::uint64_t vertex_id : cell.vertices) {
        has_first = has_first || vertex_id == edge[0];
        has_second = has_second || vertex_id == edge[1];
      }
      if (has_first && has_second) {
        auto& value = boundary_unmet[cell.id];
        value = std::max(value, ratio);
        incident = true;
      }
    }
    if (!incident) return PHX_MC_INTERNAL_ERROR;
  }
  for (const auto& [cell_id, ratio] : boundary_unmet) {
    const auto existing = std::find_if(
        records.begin(), records.end(), [&](const TetUnmetRecord& record) {
          return record.cell_id == cell_id &&
                 record.criterion == PHX_MC_TET_UNMET_SIZE;
        });
    if (existing == records.end()) {
      records.push_back({cell_id, PHX_MC_TET_UNMET_SIZE,
                         PHX_MC_TET_UNMET_NO_IMPROVEMENT, ratio});
    } else {
      existing->value = std::max(existing->value, ratio);
    }
  }
  if (!boundary_unmet.empty()) {
    std::set<std::uint64_t> unmet_cell_ids;
    for (const TetUnmetRecord& record : records) {
      unmet_cell_ids.insert(record.cell_id);
    }
    unmet_cells = static_cast<std::int64_t>(unmet_cell_ids.size());
  }
  mesh.set_unmet_records(records);
  return unmet_cells == 0 ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

}  // namespace phx::mc
