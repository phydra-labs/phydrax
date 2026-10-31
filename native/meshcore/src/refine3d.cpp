//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Source-aware Delaunay refinement by checked circumcenter cavity insertion.
// Any remaining criterion violation is reported per cell as UNRESOLVED.
#include "refine3d.hpp"
#include "improve3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cavity.hpp"
#include "local_incidence.hpp"
#include "predicates.hpp"
#include "phydrax_meshcore.h"
#include "exact_scalar.hpp"

namespace phx::mc {
double cell_size_target(const std::array<double, 4>& corner_sizes) noexcept {
  double scale = 0.0;
  for (double size : corner_sizes) {
    if (!(size > 0.0)) return std::numeric_limits<double>::infinity();
    scale = std::max(scale, size);
  }
  // Normalized by the largest target so the sum cannot overflow.
  double normalized_sum = 0.0;
  for (double size : corner_sizes) normalized_sum += size / scale;
  return std::min(scale, scale * (0.25 * normalized_sum));
}

namespace {

template <class Exit>
struct ScopeExit {
  Exit exit;
  ~ScopeExit() { exit(); }
};
template <class Exit>
ScopeExit(Exit) -> ScopeExit<Exit>;

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
  for (std::size_t corner = 0; corner < vertices.size(); ++corner) {
    const auto found = sizes.find(vertices[corner]);
    if (found == sizes.end()) return false;
    values[corner] = found->second;
  }
  target = cell_size_target(values);
  return std::isfinite(target);
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
                   const std::unordered_map<std::uint64_t, std::int32_t>& reasons,
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
    const auto reason = reasons.find(cell.id);
    const std::int32_t unmet_reason =
        reason == reasons.end() ? PHX_MC_TET_UNMET_NO_IMPROVEMENT : reason->second;
    if (shape.radius_edge_ratio > radius_edge_bound) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_RADIUS_EDGE, unmet_reason,
                         shape.radius_edge_ratio});
      cell_unmet = true;
    }
    double target = 0.0;
    if (mean_target_size(cell.vertices, sizes, target) &&
        shape.circumradius > target) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_SIZE, unmet_reason,
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
  const std::int32_t installed = mesh.install_vertex_sizes(initial, vertex_sizes);
  if (installed != PHX_MC_OK) return installed;
  std::unordered_map<std::uint64_t, double>& sizes = mesh.vertex_sizes_;

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
  // Local incidence and constrained-bank row indices: a boundary split or a
  // cone insertion touches only the cells, facets and segments around it.
  detail::LocalIncidence incidence;
  if (!incidence.initialize(mesh, initial)) return PHX_MC_INTERNAL_ERROR;
  initial = TetMeshSnapshot{};
  std::map<FaceKey, std::size_t> facet_rows_by_face;
  for (std::size_t row = 0; row < mesh.facets_.size(); ++row) {
    if (!facet_rows_by_face.emplace(sorted_face(mesh.facets_[row].vertices), row).second) {
      return PHX_MC_INTERNAL_ERROR;
    }
  }
  std::map<EdgeKey, std::size_t> segment_rows_by_edge;
  for (std::size_t row = 0; row < mesh.segments_.size(); ++row) {
    const TetSnapshotSegment& segment = mesh.segments_[row];
    if (!segment_rows_by_edge
             .emplace(sorted_edge(segment.vertices[0], segment.vertices[1]), row)
             .second) {
      return PHX_MC_INTERNAL_ERROR;
    }
  }
  // TetMesh::split_local_edge leaves the banks out of ID order.
  const auto restore_bank_order = [&]() noexcept {
    std::sort(mesh.facets_.begin(), mesh.facets_.end(),
              [](const TetSnapshotFacet& a, const TetSnapshotFacet& b) { return a.id < b.id; });
    std::sort(mesh.segments_.begin(), mesh.segments_.end(),
              [](const TetSnapshotSegment& a, const TetSnapshotSegment& b) {
                return a.id < b.id;
              });
  };
  const ScopeExit bank_order_guard{restore_bank_order};
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> source_face_rows;
  for (std::size_t row = 0; row < mesh.source_face_groups_.size(); ++row) {
    source_face_rows[mesh.source_face_groups_[row]].push_back(static_cast<std::int32_t>(row));
  }
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> source_segment_rows;
  for (std::size_t row = 0; row < mesh.source_segment_groups_.size(); ++row) {
    source_segment_rows[mesh.source_segment_groups_[row]].push_back(
        static_cast<std::int32_t>(row));
  }

  // Source witness of a boundary midpoint: the first source row of the split
  // segment's group (else of a split facet's group, by ascending group) whose
  // exact carrier is the midpoint. Rows witnessing an edge endpoint are tried
  // before the rest of their group.
  const auto make_boundary_witness =
      [&](const EdgeKey& edge, const std::array<double, 3>& carrier,
          std::span<const std::size_t> facet_rows, const TetSnapshotSegment* segment,
          TetEditVertex& vertex) {
        const auto install = [&](SourceStratum stratum, std::int32_t row) {
          const double* corners[3]{nullptr, nullptr, nullptr};
          const std::size_t entity = static_cast<std::size_t>(row);
          const int arity = stratum == SourceStratum::kSegment ? 2 : 3;
          const std::vector<std::int32_t>& bank = stratum == SourceStratum::kSegment
                                                       ? mesh.source_segments_
                                                       : mesh.source_faces_;
          for (int corner = 0; corner < arity; ++corner) {
            corners[corner] =
                mesh.source_points_.data() +
                3 * static_cast<std::size_t>(bank[arity * entity + static_cast<std::size_t>(corner)]);
          }
          double parameters[2]{};
          source_locator(corners, stratum, carrier.data(), parameters, false);
          if (!source_parameters_valid(stratum, parameters)) return false;
          double source_position[3]{};
          double deviation = 0.0;
          if (!source_carrier(corners, stratum, parameters, source_position, deviation) ||
              deviation != 0.0 || !same_point(source_position, carrier.data())) {
            return false;
          }
          vertex.witness.stratum = stratum;
          vertex.witness.entity = row;
          vertex.witness.parameters[0] = parameters[0];
          vertex.witness.parameters[1] = parameters[1];
          vertex.witness.deviation = deviation;
          return true;
        };
        const auto install_in_group = [&](SourceStratum stratum, std::int32_t group) {
          const std::vector<std::int32_t>& groups = stratum == SourceStratum::kSegment
                                                         ? mesh.source_segment_groups_
                                                         : mesh.source_face_groups_;
          for (std::uint64_t id : edge) {
            const SourceWitness& endpoint = mesh.witness(mesh.vertex_slot(id));
            if (endpoint.stratum == stratum && endpoint.entity >= 0 &&
                static_cast<std::size_t>(endpoint.entity) < groups.size() &&
                groups[static_cast<std::size_t>(endpoint.entity)] == group &&
                install(stratum, endpoint.entity)) {
              return true;
            }
          }
          const auto& rows =
              stratum == SourceStratum::kSegment ? source_segment_rows : source_face_rows;
          const auto found = rows.find(group);
          if (found == rows.end()) return false;
          return std::any_of(found->second.begin(), found->second.end(),
                             [&](std::int32_t row) { return install(stratum, row); });
        };
        if (segment != nullptr) {
          return !segment->protected_segment &&
                 install_in_group(SourceStratum::kSegment, segment->source);
        }
        std::set<std::int32_t> source_groups;
        for (std::size_t row : facet_rows) {
          if (mesh.facets_[row].source >= 0) source_groups.insert(mesh.facets_[row].source);
        }
        return std::any_of(source_groups.begin(), source_groups.end(), [&](std::int32_t group) {
          return install_in_group(SourceStratum::kFacet, group);
        });
      };
  std::vector<std::uint64_t> split_cells;
  std::vector<std::size_t> split_facet_rows;
  std::vector<std::size_t> split_segment_rows;
  std::vector<FaceKey> split_faces;
  const auto split_boundary_edge =
      [&](const EdgeKey& edge, std::uint64_t& inserted_id) {
        split_cells.clear();
        split_facet_rows.clear();
        split_segment_rows.clear();
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

        // Constrained faces through the edge, from its local star.
        for (std::int32_t cell : incidence.edge_star(first_slot, second_slot)) {
          const Tetrahedron& tetrahedron = mesh.tet(cell);
          for (int face = 0; face < 4; ++face) {
            if (tetrahedron.v[face] == first_slot || tetrahedron.v[face] == second_slot ||
                mesh.complex_.constraint(cell, face) == kNoConstraint) {
              continue;
            }
            FaceKey key{};
            int offset = 0;
            for (int corner = 0; corner < 4; ++corner) {
              if (corner != face) {
                key[static_cast<std::size_t>(offset++)] = mesh.vertex_id(tetrahedron.v[corner]);
              }
            }
            const auto row = facet_rows_by_face.find(sorted_face(key));
            if (row == facet_rows_by_face.end()) return TetEditStatus::kRejectedTopology;
            if (std::find(split_facet_rows.begin(), split_facet_rows.end(), row->second) ==
                split_facet_rows.end()) {
              split_facet_rows.push_back(row->second);
            }
          }
        }
        if (split_facet_rows.empty()) return TetEditStatus::kRejectedTopology;
        const auto segment = segment_rows_by_edge.find(edge);
        if (segment != segment_rows_by_edge.end()) split_segment_rows.push_back(segment->second);

        TetEditVertex inserted;
        inserted.id = mesh.next_vertex_id_;
        inserted.carrier = midpoint;
        inserted.ancestry.push_back({edge[0], {}, 0.5});
        inserted.ancestry.push_back({edge[1], {}, 0.5});
        if (!make_boundary_witness(
                edge, midpoint, split_facet_rows,
                segment == segment_rows_by_edge.end() ? nullptr
                                                      : &mesh.segments_[segment->second],
                inserted)) {
          return TetEditStatus::kRejectedTopology;
        }
        if (inserted_vertices == std::numeric_limits<std::int64_t>::max()) {
          return TetEditStatus::kIndexOverflow;
        }
        inserted_id = inserted.id;
        split_faces.clear();
        for (std::size_t row : split_facet_rows) {
          split_faces.push_back(sorted_face(mesh.facets_[row].vertices));
        }
        const TetEditStatus status =
            incidence.split_edge(mesh, first_slot, second_slot, inserted, split_facet_rows,
                                 split_segment_rows, split_cells);
        if (status != TetEditStatus::kApplied) return status;
        // Halves: the first overwrote each split row, the second was appended.
        const std::size_t appended_facets = mesh.facets_.size() - split_facet_rows.size();
        for (std::size_t index = 0; index < split_facet_rows.size(); ++index) {
          const std::size_t row = split_facet_rows[index];
          facet_rows_by_face.erase(split_faces[index]);
          if (!facet_rows_by_face.emplace(sorted_face(mesh.facets_[row].vertices), row).second ||
              !facet_rows_by_face
                   .emplace(sorted_face(mesh.facets_[appended_facets + index].vertices),
                            appended_facets + index)
                   .second) {
            return TetEditStatus::kStaleGeneration;
          }
        }
        if (!split_segment_rows.empty()) {
          const std::size_t row = split_segment_rows.front();
          const std::size_t appended = mesh.segments_.size() - 1;
          segment_rows_by_edge.erase(edge);
          const auto& first_half = mesh.segments_[row].vertices;
          const auto& second_half = mesh.segments_[appended].vertices;
          if (!segment_rows_by_edge.emplace(sorted_edge(first_half[0], first_half[1]), row)
                   .second ||
              !segment_rows_by_edge
                   .emplace(sorted_edge(second_half[0], second_half[1]), appended)
                   .second) {
            return TetEditStatus::kStaleGeneration;
          }
        }
        return TetEditStatus::kApplied;
      };
  std::set<EdgeKey> blocked_boundary_edges;
  if (!sizes.empty()) {
    // Longest boundary edge relative to its target first (ties: smallest
    // key). Splitting never moves a vertex, so a queued ratio stays exact
    // until its edge is split; split edges never reappear.
    std::set<std::pair<double, EdgeKey>> boundary_queue;
    const auto queue_boundary_edge = [&](std::uint64_t first_id, std::uint64_t second_id) {
      const EdgeKey edge = sorted_edge(first_id, second_id);
      double mean_size = 0.0;
      double edge_target = 0.0;
      if (!boundary_target(edge, mean_size, edge_target)) return true;
      const std::int32_t first_slot = mesh.vertex_slot(edge[0]);
      const std::int32_t second_slot = mesh.vertex_slot(edge[1]);
      if (!mesh.live_vertex(first_slot) || !mesh.live_vertex(second_slot)) return false;
      const double* first = mesh.point(first_slot);
      const double* second = mesh.point(second_slot);
      const double length = std::hypot(
          std::hypot(second[0] - first[0], second[1] - first[1]), second[2] - first[2]);
      const double ratio = length / edge_target;
      if (ratio > 1.0) boundary_queue.emplace(-ratio, edge);
      return true;
    };
    const auto queue_facet_edges = [&](const TetSnapshotFacet& facet) {
      return queue_boundary_edge(facet.vertices[0], facet.vertices[1]) &&
             queue_boundary_edge(facet.vertices[1], facet.vertices[2]) &&
             queue_boundary_edge(facet.vertices[2], facet.vertices[0]);
    };
    for (const TetSnapshotFacet& facet : mesh.facets_) {
      if (!queue_facet_edges(facet)) return PHX_MC_INTERNAL_ERROR;
    }
    while (!boundary_queue.empty()) {
      const EdgeKey selected_edge = boundary_queue.begin()->second;
      boundary_queue.erase(boundary_queue.begin());
      if (blocked_boundary_edges.contains(selected_edge)) continue;
      double selected_size = 0.0;
      double edge_target = 0.0;
      if (!boundary_target(selected_edge, selected_size, edge_target)) {
        return PHX_MC_INTERNAL_ERROR;
      }
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
        const std::size_t appended = mesh.facets_.size() - split_facet_rows.size();
        for (std::size_t index = 0; index < split_facet_rows.size(); ++index) {
          if (!queue_facet_edges(mesh.facets_[split_facet_rows[index]]) ||
              !queue_facet_edges(mesh.facets_[appended + index])) {
            return PHX_MC_INTERNAL_ERROR;
          }
        }
        continue;
      }
      blocked_boundary_edges.insert(selected_edge);
    }
  }

  // Interior Delaunay refinement. A violating cell (radius-edge ratio above the
  // bound, or circumradius above its size target) is removed by inserting its
  // circumcenter c: walk from the cell to c, grow the Bowyer-Watson cavity of
  // cells whose circumspheres strictly contain c without crossing constrained
  // faces, shrink it until it is star-shaped from c and keeps every vertex and
  // constrained segment on its boundary, then cone it to c through the
  // cavity-local TetMesh::insert_local_cone commit.
  // Encroachment follows Shewchuk: when c lies beyond a constrained face, or
  // inside the equatorial ball of a constrained face bounding its cavity, a
  // boundary that may be split is split at that face's longest edge instead of
  // inserting c. A fixed boundary keeps its faces: a c inside the domain is
  // still inserted (the cavity stops at the fixed faces and the cone is proved
  // valid), and a c beyond a fixed face leaves the cell unmet with reason
  // FIXED_BOUNDARY. Shewchuk's guarantee needs B > 2 and input dihedral angles
  // of at least 90 degrees; smaller input angles are not excluded but may
  // leave cells unmet.
  // Termination: a point x is inserted only at distance at least
  // g(x) = min(e0, c * h(x)) from every vertex present, where e0 is the
  // shortest edge entering this pass, h(x) the circumradius size target
  // interpolated at x (the target of the cell or edge being split; g = e0
  // where no target applies), and c = 1/2. Packing: refinement never removes
  // a vertex, so any two vertices of which at least one was inserted are at
  // least g_min = min(e0, c * h_min) > 0 apart, h_min the smallest positive
  // target. Balls of radius g_min / 2 around them are disjoint and lie in the
  // g_min / 2-neighborhood of the bounded domain, so at most
  // vol(domain + B(g_min / 2)) / vol(B(g_min / 2)) points are inserted. For a
  // Delaunay cavity the circumcenter's nearest vertex lies at the
  // circumradius, which exceeds h when the cell violates its size target and
  // exceeds B times the cell's shortest edge when it violates the
  // radius-edge bound, so the guard binds only near small input features;
  // there a cell is reported unmet with reason REFUSED.
  std::unordered_map<std::uint64_t, std::int32_t> unmet_reasons;
  double shortest_initial_edge = std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < mesh.complex_.tets.size(); ++row) {
    const std::int32_t cell = static_cast<std::int32_t>(row);
    if (!mesh.complex_.live(cell) || is_ghost(mesh.tet(cell))) continue;
    const Tetrahedron& tetrahedron = mesh.tet(cell);
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const double* a = mesh.point(tetrahedron.v[first]);
        const double* b = mesh.point(tetrahedron.v[second]);
        shortest_initial_edge = std::min(
            shortest_initial_edge,
            std::hypot(std::hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]));
      }
    }
  }
  if (!(shortest_initial_edge > 0.0) || !std::isfinite(shortest_initial_edge)) {
    return PHX_MC_INVALID_INPUT;
  }
  constexpr double kSizeGuardFraction = 0.5;
  const auto spacing_guard = [&](double target_size) {
    return target_size > 0.0
               ? std::min(shortest_initial_edge, kSizeGuardFraction * target_size)
               : shortest_initial_edge;
  };
  // Grid cells of the smallest guard; a query searches the cell rings that
  // cover its own guard radius.
  double separation = shortest_initial_edge;
  for (const auto& [vertex_id, size] : sizes) {
    (void)vertex_id;
    separation = std::min(separation, spacing_guard(size));
  }
  if (!(separation > 0.0) || !std::isfinite(separation)) return PHX_MC_INVALID_INPUT;

  struct GridKey {
    std::array<std::int64_t, 3> cell{};
    bool operator==(const GridKey&) const noexcept = default;
  };
  struct GridHash {
    std::size_t operator()(const GridKey& key) const noexcept {
      std::size_t seed = 0;
      for (std::int64_t value : key.cell) {
        seed ^= std::hash<std::int64_t>{}(value) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
  std::unordered_map<GridKey, std::vector<std::int32_t>, GridHash> grid;
  const auto grid_key = [&](const double* point, GridKey& key) {
    for (int axis = 0; axis < 3; ++axis) {
      const double scaled = std::floor(point[axis] / separation);
      if (!(std::abs(scaled) < 0x1p62)) return false;
      key.cell[static_cast<std::size_t>(axis)] = static_cast<std::int64_t>(scaled);
    }
    return true;
  };
  const auto grid_insert = [&](std::int32_t vertex) {
    GridKey key;
    if (!grid_key(mesh.point(vertex), key)) return false;
    grid[key].push_back(vertex);
    return true;
  };
  const auto separated = [&](const double* point, double target_size) {
    const double guard = spacing_guard(target_size);
    const auto clear_of = [&](const std::vector<std::int32_t>& vertices) {
      for (std::int32_t vertex : vertices) {
        const double* other = mesh.point(vertex);
        const double distance = std::hypot(
            std::hypot(point[0] - other[0], point[1] - other[1]), point[2] - other[2]);
        if (distance < guard) return false;
      }
      return true;
    };
    GridKey center;
    if (!grid_key(point, center)) return false;
    const double ring = std::ceil(guard / separation);
    const double span = 2.0 * ring + 1.0;
    if (span * span * span > static_cast<double>(grid.size())) {
      for (const auto& [key, vertices] : grid) {
        (void)key;
        if (!clear_of(vertices)) return false;
      }
      return true;
    }
    const std::int64_t cells = static_cast<std::int64_t>(ring);
    for (std::int64_t dx = -cells; dx <= cells; ++dx) {
      for (std::int64_t dy = -cells; dy <= cells; ++dy) {
        for (std::int64_t dz = -cells; dz <= cells; ++dz) {
          GridKey key{{center.cell[0] + dx, center.cell[1] + dy, center.cell[2] + dz}};
          const auto found = grid.find(key);
          if (found != grid.end() && !clear_of(found->second)) return false;
        }
      }
    }
    return true;
  };
  for (std::size_t vertex = 0; vertex < mesh.vertex_ids_.size(); ++vertex) {
    if (mesh.vertex_live_[vertex] != 0 &&
        !grid_insert(static_cast<std::int32_t>(vertex))) {
      return PHX_MC_UNRESOLVED;
    }
  }

  // Largest circumradius first; cells are immutable, so a key stays valid
  // until its cell dies.
  std::set<std::pair<double, std::uint64_t>> queue;
  const auto snapshot_cell = [&](std::int32_t slot, TetSnapshotCell& cell) {
    cell.id = mesh.cell_id(slot);
    cell.region = mesh.complex_.region(slot);
    for (int corner = 0; corner < 4; ++corner) {
      cell.vertices[static_cast<std::size_t>(corner)] = mesh.vertex_id(mesh.tet(slot).v[corner]);
    }
  };
  const auto enqueue = [&](std::uint64_t id) {
    const std::int32_t slot = mesh.cell_slot(id);
    if (slot < 0 || !mesh.complex_.live(slot) || is_ghost(mesh.tet(slot))) return;
    TetSnapshotCell cell;
    snapshot_cell(slot, cell);
    if (!violates(mesh, cell, radius_edge_bound, sizes)) return;
    std::array<std::int32_t, 4> slots{};
    geometry::TetrahedronShape shape;
    const double radius =
        tetrahedron_geometry(mesh, cell, slots, shape) ? shape.circumradius : 0.0;
    queue.emplace(-radius, id);
  };
  for (std::size_t row = 0; row < mesh.complex_.tets.size(); ++row) {
    const std::int32_t cell = static_cast<std::int32_t>(row);
    if (mesh.complex_.live(cell) && !is_ghost(mesh.tet(cell))) enqueue(mesh.cell_id(cell));
  }

  const bool boundary_splittable = mesh.policy_ != BoundaryPolicy::kFixed;
  const auto edit_error = [](TetEditStatus status) -> std::int32_t {
    if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
    if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
    if (status == TetEditStatus::kStaleGeneration) return PHX_MC_INTERNAL_ERROR;
    return PHX_MC_OK;
  };
  // Splits the longest edge of an encroached constrained face. Returns OK and
  // sets `split` when a vertex was inserted.
  const auto split_encroached_face = [&](std::int32_t cell, int face,
                                         bool& split) -> std::int32_t {
    split = false;
    std::int32_t facet[3];
    oriented_facet(mesh.tet(cell).v, face, facet);
    EdgeKey longest{};
    double longest_length = -1.0;
    std::array<double, 3> midpoint{};
    for (int corner = 0; corner < 3; ++corner) {
      const double* a = mesh.point(facet[corner]);
      const double* b = mesh.point(facet[(corner + 1) % 3]);
      const double length = std::hypot(std::hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]);
      if (length > longest_length) {
        longest_length = length;
        longest = sorted_edge(mesh.vertex_id(facet[corner]),
                              mesh.vertex_id(facet[(corner + 1) % 3]));
        for (int axis = 0; axis < 3; ++axis) {
          midpoint[static_cast<std::size_t>(axis)] = 0.5 * (a[axis] + b[axis]);
        }
      }
    }
    double mean_size = 0.0;
    double edge_target = 0.0;
    const bool has_size = boundary_target(longest, mean_size, edge_target);
    if (!separated(midpoint.data(), has_size ? mean_size : 0.0)) return PHX_MC_OK;
    std::uint64_t inserted_id = 0;
    const TetEditStatus status = split_boundary_edge(longest, inserted_id);
    const std::int32_t error = edit_error(status);
    if (error != PHX_MC_OK) return error;
    if (status != TetEditStatus::kApplied) return PHX_MC_OK;
    if (has_size) sizes.emplace(inserted_id, mean_size);
    if (!grid_insert(mesh.vertex_slot(inserted_id))) return PHX_MC_UNRESOLVED;
    for (std::uint64_t id : split_cells) enqueue(id);
    ++inserted_vertices;
    split = true;
    return PHX_MC_OK;
  };
  // Equatorial ball of a triangle: the smallest sphere through its corners.
  const auto in_equatorial_ball = [](const double* const corners[3], const double* point) {
    double ab[3], ac[3];
    for (int axis = 0; axis < 3; ++axis) {
      ab[axis] = corners[1][axis] - corners[0][axis];
      ac[axis] = corners[2][axis] - corners[0][axis];
    }
    const double normal[3]{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                           ab[0] * ac[1] - ab[1] * ac[0]};
    const double normal_squared =
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if (!(normal_squared > 0.0)) return false;
    const double ab_squared = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    const double ac_squared = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
    double offset[3];
    double radius_squared = 0.0;
    double distance_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const int next = (axis + 1) % 3;
      const int last = (axis + 2) % 3;
      offset[axis] = (ab_squared * (ac[next] * normal[last] - ac[last] * normal[next]) +
                      ac_squared * (normal[next] * ab[last] - normal[last] * ab[next])) /
                     (2.0 * normal_squared);
      radius_squared += offset[axis] * offset[axis];
      const double delta = point[axis] - (corners[0][axis] + offset[axis]);
      distance_squared += delta * delta;
    }
    return distance_squared < radius_squared;
  };

  TetInsertionCavity cavity;
  std::vector<std::uint64_t> created_cells;
  while (!queue.empty()) {
    const std::uint64_t cell_id = queue.begin()->second;
    queue.erase(queue.begin());
    const std::int32_t root_slot = mesh.cell_slot(cell_id);
    if (root_slot < 0 || !mesh.complex_.live(root_slot)) continue;
    TetSnapshotCell parent;
    snapshot_cell(root_slot, parent);
    if (!violates(mesh, parent, radius_edge_bound, sizes)) continue;
    std::array<std::int32_t, 4> parent_slots{};
    geometry::TetrahedronShape shape;
    if (!tetrahedron_geometry(mesh, parent, parent_slots, shape)) {
      unmet_reasons[cell_id] = PHX_MC_TET_UNMET_NONREPRESENTABLE;
      continue;
    }
    const double* center = shape.circumcenter.data();
    if (!std::all_of(shape.circumcenter.begin(), shape.circumcenter.end(),
                     [](double value) { return coordinate_in_domain(value); })) {
      unmet_reasons[cell_id] = PHX_MC_TET_UNMET_NONREPRESENTABLE;
      continue;
    }

    const TetCavityStatus located = mesh.insertion_cavity(root_slot, center, false, cavity);
    if (located == TetCavityStatus::kRefused) {
      unmet_reasons[cell_id] = PHX_MC_TET_UNMET_REFUSED;
      continue;
    }
    if (located == TetCavityStatus::kBlocked) {
      bool split = false;
      if (boundary_splittable) {
        const std::int32_t error =
            split_encroached_face(cavity.blocked_cell, cavity.blocked_face, split);
        if (error != PHX_MC_OK) return error;
      }
      if (split) {
        enqueue(cell_id);
      } else {
        unmet_reasons[cell_id] = boundary_splittable ? PHX_MC_TET_UNMET_REFUSED
                                                     : PHX_MC_TET_UNMET_FIXED_BOUNDARY;
      }
      continue;
    }
    double center_size = 0.0;
    const bool has_center_size = mean_target_size(parent.vertices, sizes, center_size);
    if (!separated(center, has_center_size ? center_size : 0.0)) {
      unmet_reasons[cell_id] = PHX_MC_TET_UNMET_REFUSED;
      continue;
    }

    if (boundary_splittable) {
      std::int32_t encroached_cell = -1;
      int encroached_face = -1;
      for (std::int32_t cell : cavity.cells) {
        for (int face = 0; face < 4 && encroached_cell < 0; ++face) {
          if (cavity.members.contains(mesh.tet(cell).n[face]) ||
              mesh.complex_.constraint(cell, face) == kNoConstraint) {
            continue;
          }
          std::int32_t facet[3];
          oriented_facet(mesh.tet(cell).v, face, facet);
          const double* corners[3]{mesh.point(facet[0]), mesh.point(facet[1]),
                                   mesh.point(facet[2])};
          if (in_equatorial_ball(corners, center)) {
            encroached_cell = cell;
            encroached_face = face;
          }
        }
        if (encroached_cell >= 0) break;
      }
      if (encroached_cell >= 0) {
        bool split = false;
        const std::int32_t error = split_encroached_face(encroached_cell, encroached_face, split);
        if (error != PHX_MC_OK) return error;
        if (split) {
          enqueue(cell_id);
        } else {
          unmet_reasons[cell_id] = PHX_MC_TET_UNMET_REFUSED;
        }
        continue;
      }
    }

    TetEditVertex inserted;
    inserted.id = mesh.next_vertex_id_;
    std::copy(shape.circumcenter.begin(), shape.circumcenter.end(), inserted.carrier.begin());
    const TetEditStatus status =
        incidence.insert_cone(mesh, inserted, cavity.cells, created_cells);
    const std::int32_t error = edit_error(status);
    if (error != PHX_MC_OK) return error;
    if (status != TetEditStatus::kApplied) {
      unmet_reasons[cell_id] = PHX_MC_TET_UNMET_REFUSED;
      continue;
    }
    if (has_center_size) sizes.emplace(inserted.id, center_size);
    if (!grid_insert(mesh.vertex_slot(inserted.id))) return PHX_MC_UNRESOLVED;
    ++inserted_vertices;
    for (std::uint64_t id : created_cells) enqueue(id);
  }

  restore_bank_order();
  TetMeshSnapshot final_state;
  if (mesh.snapshot(final_state) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  std::vector<TetUnmetRecord> records;
  collect_unmet(mesh, final_state, radius_edge_bound, sizes, unmet_reasons, records,
                unmet_cells);
  std::map<std::uint64_t, double> boundary_unmet;
  for (const EdgeKey& edge : blocked_boundary_edges) {
    const std::int32_t first_slot = mesh.vertex_slot(edge[0]);
    const std::int32_t second_slot = mesh.vertex_slot(edge[1]);
    if (!mesh.live_vertex(first_slot) || !mesh.live_vertex(second_slot)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const std::span<const std::int32_t> star = incidence.edge_star(first_slot, second_slot);
    const bool remains_constrained =
        std::any_of(star.begin(), star.end(), [&](std::int32_t cell) {
          for (int face = 0; face < 4; ++face) {
            const std::int32_t opposite = mesh.tet(cell).v[face];
            if (opposite != first_slot && opposite != second_slot &&
                mesh.complex_.constraint(cell, face) != kNoConstraint) {
              return true;
            }
          }
          return false;
        });
    if (!remains_constrained) continue;
    double mean_size = 0.0;
    double edge_target = 0.0;
    if (!boundary_target(edge, mean_size, edge_target)) continue;
    const double* first = mesh.point(first_slot);
    const double* second = mesh.point(second_slot);
    const double length = std::hypot(
        std::hypot(second[0] - first[0], second[1] - first[1]),
        second[2] - first[2]);
    const double ratio = length / edge_target;
    if (!(ratio > 1.0)) continue;
    for (std::int32_t cell : star) {
      auto& value = boundary_unmet[mesh.cell_id(cell)];
      value = std::max(value, ratio);
    }
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
