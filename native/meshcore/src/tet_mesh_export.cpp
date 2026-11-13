//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Canonical stable-ID snapshot and validated periodic quotient publication.
#include "tet_mesh.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "expansion.hpp"
#include "predicates.hpp"

namespace phx::mc {
namespace {

bool same_bits(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

bool exact_isometry(const std::array<double, 12>& affine) {
  for (double value : affine) {
    if (!coordinate_in_domain(value)) return false;
  }
  for (int row = 0; row < 3; ++row) {
    for (int other = row; other < 3; ++other) {
      Expansion dot;
      for (int column = 0; column < 3; ++column) {
        dot = dot + Expansion::product(affine[4 * row + column],
                                      affine[4 * other + column]);
      }
      const Expansion target(row == other ? 1.0 : 0.0);
      if (!(dot - target).is_zero()) return false;
    }
  }
  const Expansion a00(affine[0]), a01(affine[1]), a02(affine[2]);
  const Expansion a10(affine[4]), a11(affine[5]), a12(affine[6]);
  const Expansion a20(affine[8]), a21(affine[9]), a22(affine[10]);
  const Expansion determinant =
      a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) +
      a02 * (a10 * a21 - a11 * a20);
  return (determinant - Expansion(1.0)).is_zero() ||
         (determinant + Expansion(1.0)).is_zero();
}

bool affine_image(const std::array<double, 12>& affine, const double* point,
                  double* image) {
  for (int axis = 0; axis < 3; ++axis) {
    Expansion coordinate(affine[4 * axis + 3]);
    for (int component = 0; component < 3; ++component) {
      coordinate = coordinate + Expansion::product(affine[4 * axis + component],
                                                    point[component]);
    }
    image[axis] = nearest_double(coordinate);
    if (!std::isfinite(image[axis])) return false;
  }
  return true;
}

bool identity_transform(const std::array<double, 12>& affine) {
  static constexpr std::array<double, 12> identity{
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0};
  for (std::size_t i = 0; i < affine.size(); ++i) {
    if (!same_bits(affine[i], identity[i])) return false;
  }
  return true;
}

bool zero_winding(const std::array<std::int64_t, 3>& winding) {
  return winding[0] == 0 && winding[1] == 0 && winding[2] == 0;
}

}  // namespace

TetEditStatus TetMesh::snapshot(TetMeshSnapshot& output) const {
  return snapshot_internal(output, true);
}

TetEditStatus TetMesh::snapshot_internal(TetMeshSnapshot& output,
                                        bool include_cell_ancestry) const {
  try {
    TetMeshSnapshot staged;
    staged.generation = generation_;
    staged.identity_bound = identity_bound_;
    staged.source_id = source_id_;
    staged.source_revision = source_revision_;
    staged.source_map_id = source_map_id_;
    staged.frame_id = frame_id_;
    staged.vertices.reserve(vertex_ids_.size());
    staged.cells.reserve(static_cast<std::size_t>(complex_.finite_count));
    staged.constrained_facets = facets_;
    staged.constrained_segments = segments_;
    staged.source_points = source_points_;
    staged.source_faces = source_faces_;
    staged.source_face_groups = source_face_groups_;
    staged.source_face_ids = source_face_ids_;
    staged.source_face_tolerances = source_face_tolerances_;
    staged.source_segments = source_segments_;
    staged.source_segment_groups = source_segment_groups_;
    staged.source_segment_tolerances = source_segment_tolerances_;
    staged.source_segment_ids = source_segment_ids_;
    staged.source_face_planes = source_face_planes_;
    staged.source_face_plane_tolerances = source_face_plane_tolerances_;
    staged.periodic_orbits = periodic_orbits_;
    staged.quotient_cells = quotient_cells_;
    staged.periodic_identity = periodic_identity_;
    staged.periodic_entity_maps = periodic_entity_maps_;
    for (std::size_t slot = 0; slot < vertex_ids_.size(); ++slot) {
      if (!vertex_live_[slot]) continue;
      TetSnapshotVertex vertex;
      vertex.id = vertex_ids_[slot];
      std::copy_n(points_.data() + 3 * slot, 3, vertex.carrier.begin());
      vertex.witness = witnesses_[slot];
      vertex.stratum_dimension = vertex_dimension_[slot];
      vertex.protection_radius = protection_radii_[slot];
      vertex.protected_vertex = protected_vertices_[slot] != 0;
      vertex.has_edge_parameter = vertex_has_edge_parameter_[slot] != 0;
      vertex.edge_endpoints = vertex_edge_endpoints_[slot];
      vertex.edge_parameter = vertex_edge_parameters_[slot];
      const auto ancestry = vertex_ancestry_.find(vertex.id);
      if (ancestry != vertex_ancestry_.end()) vertex.ancestry = ancestry->second;
      staged.vertices.push_back(vertex);
    }
    for (std::size_t slot = 0; slot < complex_.tets.size(); ++slot) {
      const std::int32_t local_slot = static_cast<std::int32_t>(slot);
      if (!complex_.live(local_slot) || is_ghost(complex_.tets[slot])) continue;
      TetSnapshotCell cell;
      cell.id = cell_ids_[slot];
      cell.region = complex_.region(local_slot);
      if (include_cell_ancestry) {
        const auto ancestry = cell_ancestry_.find(cell.id);
        if (ancestry != cell_ancestry_.end()) {
          cell.ancestor_cell_ids = *ancestry->second.ancestor_cell_ids;
          cell.edit_kind = ancestry->second.edit_kind;
        }
      }
      for (int corner = 0; corner < 4; ++corner) {
        const std::int32_t vertex = complex_.tets[slot].v[corner];
        if (!live_vertex(vertex)) return TetEditStatus::kInvalidGeometry;
        cell.vertices[corner] = vertex_id(vertex);
      }
      staged.cells.push_back(std::move(cell));
    }
    using std::swap;
    swap(output, staged);
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

TetEditStatus TetMesh::set_periodic_state(
    std::span<const TetPeriodicOrbit> orbits,
    std::span<const TetPeriodicQuotientCell> cells,
    std::span<const TetPeriodicEntityOrbitMap> entity_maps,
    bool advance_generation) {
  clear_implicit_facet_refusal();
  if (implicit_source_ && (!orbits.empty() || !cells.empty() || !entity_maps.empty())) {
    return TetEditStatus::kRejectedTopology;
  }
  try {
    std::vector<TetPeriodicOrbit> staged_orbits(orbits.begin(), orbits.end());
    std::vector<TetPeriodicQuotientCell> staged_cells(cells.begin(), cells.end());
    std::vector<TetPeriodicEntityOrbitMap> staged_maps(entity_maps.begin(), entity_maps.end());
    std::unordered_set<std::uint64_t> orbit_ids;
    std::unordered_set<std::uint64_t> member_ids;
    std::unordered_set<std::uint64_t> lifted_cell_ids;
    std::unordered_set<std::uint64_t> quotient_ids;
    std::unordered_set<std::uint64_t> active_orbit_ids;
    std::unordered_set<std::uint64_t> active_quotient_ids;
    active_orbit_ids.reserve(periodic_orbits_.size());
    active_quotient_ids.reserve(quotient_cells_.size());
    for (const TetPeriodicOrbit& orbit : periodic_orbits_) active_orbit_ids.insert(orbit.id);
    for (const TetPeriodicQuotientCell& cell : quotient_cells_) active_quotient_ids.insert(cell.id);
    for (const TetPeriodicOrbit& orbit : staged_orbits) {
      if (orbit.id == std::numeric_limits<std::uint64_t>::max() ||
          !orbit_ids.insert(orbit.id).second || orbit.members.empty() ||
          vertex_slot(orbit.representative_vertex_id) < 0 ||
          (used_periodic_orbit_ids_.contains(orbit.id) &&
           !active_orbit_ids.contains(orbit.id))) {
        return TetEditStatus::kRejectedTopology;
      }
      std::size_t representative_count = 0;
      std::set<std::array<std::int64_t, 3>> windings;
      const std::int32_t representative_slot = vertex_slot(orbit.representative_vertex_id);
      const double* representative = point(representative_slot);
      for (const TetPeriodicVertexMember& member : orbit.members) {
        const std::int32_t member_slot = vertex_slot(member.vertex_id);
        if (member_slot < 0 || !member_ids.insert(member.vertex_id).second ||
            !windings.insert(member.winding).second ||
            !exact_isometry(member.representative_to_member)) {
          return TetEditStatus::kRejectedTopology;
        }
        double expected[3];
        if (!affine_image(member.representative_to_member, representative, expected)) {
          return TetEditStatus::kUncertainPredicate;
        }
        const double* actual = point(member_slot);
        for (int axis = 0; axis < 3; ++axis) {
          if (!same_bits(expected[axis], actual[axis])) {
            return TetEditStatus::kInvalidGeometry;
          }
        }
        if (member.vertex_id == orbit.representative_vertex_id) {
          ++representative_count;
          if (!identity_transform(member.representative_to_member) ||
              !zero_winding(member.winding)) return TetEditStatus::kRejectedTopology;
        }
      }
      if (representative_count != 1) return TetEditStatus::kRejectedTopology;
    }

    for (const TetPeriodicQuotientCell& cell : staged_cells) {
      if (cell.id == std::numeric_limits<std::uint64_t>::max() ||
          !quotient_ids.insert(cell.id).second || cell.lifted_cell_ids.empty() ||
          (used_quotient_cell_ids_.contains(cell.id) &&
           !active_quotient_ids.contains(cell.id))) {
        return TetEditStatus::kRejectedTopology;
      }
      for (std::uint64_t lifted : cell.lifted_cell_ids) {
        const std::int32_t slot = cell_slot(lifted);
        if (slot < 0 || !lifted_cell_ids.insert(lifted).second ||
            complex_.region(slot) != cell.region) return TetEditStatus::kRejectedTopology;
      }
      std::int32_t lifted_vertices[4];
      for (int corner = 0; corner < 4; ++corner) {
        const auto orbit = std::find_if(staged_orbits.begin(), staged_orbits.end(),
            [&](const TetPeriodicOrbit& candidate) {
              return candidate.representative_vertex_id == cell.representative_vertices[corner];
            });
        if (orbit == staged_orbits.end()) return TetEditStatus::kRejectedTopology;
        const auto member = std::find_if(orbit->members.begin(), orbit->members.end(),
            [&](const TetPeriodicVertexMember& candidate) {
              return candidate.winding == cell.winding[corner];
            });
        if (member == orbit->members.end()) return TetEditStatus::kRejectedTopology;
        lifted_vertices[corner] = vertex_slot(member->vertex_id);
        for (int prior = 0; prior < corner; ++prior) {
          if (lifted_vertices[prior] == lifted_vertices[corner]) {
            return TetEditStatus::kRejectedTopology;
          }
        }
      }
      if (!positive(lifted_vertices)) return TetEditStatus::kInvalidGeometry;
    }
    if (periodic_identity_.bound) {
      for (const std::string& id : periodic_identity_.ids) {
        if (id.empty()) return TetEditStatus::kRejectedTopology;
      }
      if (staged_maps.size() != 4) return TetEditStatus::kRejectedTopology;
    } else if (!staged_maps.empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    std::array<std::unordered_set<std::uint64_t>, 4> active_lifted;
    std::array<std::unordered_set<std::uint64_t>, 4> active_quotient;
    for (const TetPeriodicEntityOrbitMap& map : periodic_entity_maps_) {
      if (map.dimension < 0 || map.dimension > 3) return TetEditStatus::kRejectedTopology;
      const std::size_t dimension = static_cast<std::size_t>(map.dimension);
      active_lifted[dimension].insert(map.lifted_entity_ids.begin(), map.lifted_entity_ids.end());
      active_quotient[dimension].insert(map.quotient_entity_ids.begin(), map.quotient_entity_ids.end());
    }
    std::array<std::unordered_set<std::uint64_t>, 4> new_lifted;
    std::array<std::unordered_set<std::uint64_t>, 4> new_quotient;
    std::int32_t periodic_rank = -1;
    for (std::size_t dimension = 0; dimension < staged_maps.size(); ++dimension) {
      const TetPeriodicEntityOrbitMap& map = staged_maps[dimension];
      const std::size_t lifted_count = map.lifted_entity_ids.size();
      const std::size_t quotient_count = map.quotient_entity_ids.size();
      const std::size_t arity = dimension + 1;
      if (map.dimension != static_cast<std::int32_t>(dimension) ||
          map.arity != static_cast<std::int32_t>(arity) || map.rank < 0 || map.rank > 3 ||
          lifted_count == 0 || quotient_count == 0 || quotient_count >
              static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
          map.lifted_to_quotient.size() != lifted_count ||
          map.orientation_signs.size() != lifted_count ||
          (map.rank != 0 && lifted_count > std::numeric_limits<std::size_t>::max() /
                                                static_cast<std::size_t>(map.rank)) ||
          map.winding_shifts.size() != lifted_count * static_cast<std::size_t>(map.rank) ||
          lifted_count > std::numeric_limits<std::size_t>::max() / arity ||
          map.corner_permutations.size() != lifted_count * arity ||
          map.representative_lifted_rows.size() != quotient_count) {
        return TetEditStatus::kRejectedTopology;
      }
      if (periodic_rank < 0) periodic_rank = map.rank;
      if (map.rank != periodic_rank || (periodic_identity_.bound && map.rank == 0)) {
        return TetEditStatus::kRejectedTopology;
      }
      std::vector<std::uint8_t> quotient_seen(quotient_count, 0);
      std::vector<std::uint8_t> representative_seen(lifted_count, 0);
      for (std::size_t row = 0; row < quotient_count; ++row) {
        const std::uint64_t id = map.quotient_entity_ids[row];
        if (id == std::numeric_limits<std::uint64_t>::max() ||
            !new_quotient[dimension].insert(id).second ||
            (used_periodic_quotient_ids_[dimension].contains(id) &&
             !active_quotient[dimension].contains(id))) return TetEditStatus::kRejectedTopology;
        const std::int32_t representative = map.representative_lifted_rows[row];
        if (representative < 0 || static_cast<std::size_t>(representative) >= lifted_count ||
            representative_seen[static_cast<std::size_t>(representative)] != 0 ||
            map.lifted_to_quotient[static_cast<std::size_t>(representative)] !=
                static_cast<std::int32_t>(row)) return TetEditStatus::kRejectedTopology;
        representative_seen[static_cast<std::size_t>(representative)] = 1;
      }
      for (std::size_t row = 0; row < lifted_count; ++row) {
        const std::uint64_t id = map.lifted_entity_ids[row];
        const std::int32_t quotient_row = map.lifted_to_quotient[row];
        if (id == std::numeric_limits<std::uint64_t>::max() ||
            !new_lifted[dimension].insert(id).second || quotient_row < 0 ||
            static_cast<std::size_t>(quotient_row) >= quotient_count ||
            (map.orientation_signs[row] != 1 && map.orientation_signs[row] != -1) ||
            (used_periodic_lifted_ids_[dimension].contains(id) &&
             !active_lifted[dimension].contains(id))) return TetEditStatus::kRejectedTopology;
        quotient_seen[static_cast<std::size_t>(quotient_row)] = 1;
        if ((dimension == 0 && vertex_slot(id) < 0) ||
            (dimension == 3 && cell_slot(id) < 0)) return TetEditStatus::kRejectedTopology;
        std::array<std::uint8_t, 4> corners{};
        for (std::size_t corner = 0; corner < arity; ++corner) {
          const std::int32_t mapped = map.corner_permutations[row * arity + corner];
          if (mapped < 0 || static_cast<std::size_t>(mapped) >= arity ||
              corners[static_cast<std::size_t>(mapped)] != 0) return TetEditStatus::kRejectedTopology;
          corners[static_cast<std::size_t>(mapped)] = 1;
        }
      }
      if (std::any_of(quotient_seen.begin(), quotient_seen.end(),
                      [](std::uint8_t seen) { return seen == 0; })) {
        return TetEditStatus::kRejectedTopology;
      }
    }
    if (periodic_identity_.bound) {
      const std::size_t live_vertex_count = static_cast<std::size_t>(
          std::count(vertex_live_.begin(), vertex_live_.end(), std::int8_t{1}));
      const TetPeriodicEntityOrbitMap& vertex_map = staged_maps[0];
      const TetPeriodicEntityOrbitMap& cell_map = staged_maps[3];
      if (vertex_map.lifted_entity_ids.size() != live_vertex_count ||
          cell_map.lifted_entity_ids.size() !=
              static_cast<std::size_t>(complex_.finite_count) ||
          cell_map.quotient_entity_ids.size() != staged_cells.size()) {
        return TetEditStatus::kRejectedTopology;
      }
      std::unordered_map<std::uint64_t, std::size_t> quotient_rows;
      std::unordered_map<std::uint64_t, std::size_t> lifted_rows;
      quotient_rows.reserve(cell_map.quotient_entity_ids.size());
      lifted_rows.reserve(cell_map.lifted_entity_ids.size());
      std::vector<std::size_t> quotient_lifted_counts(cell_map.quotient_entity_ids.size(), 0);
      for (std::size_t row = 0; row < cell_map.quotient_entity_ids.size(); ++row) {
        if (!quotient_rows.emplace(cell_map.quotient_entity_ids[row], row).second) {
          return TetEditStatus::kRejectedTopology;
        }
      }
      for (std::size_t row = 0; row < cell_map.lifted_entity_ids.size(); ++row) {
        const std::size_t quotient_row = static_cast<std::size_t>(
            cell_map.lifted_to_quotient[row]);
        if (!lifted_rows.emplace(cell_map.lifted_entity_ids[row], quotient_row).second) {
          return TetEditStatus::kRejectedTopology;
        }
        ++quotient_lifted_counts[quotient_row];
      }
      for (const TetPeriodicQuotientCell& cell : staged_cells) {
        const auto quotient = quotient_rows.find(cell.id);
        const auto orbit = quotient_rows.find(cell.orbit_id);
        if (quotient == quotient_rows.end() || orbit == quotient_rows.end() ||
            quotient_lifted_counts[quotient->second] != cell.lifted_cell_ids.size()) {
          return TetEditStatus::kRejectedTopology;
        }
        for (std::uint64_t lifted : cell.lifted_cell_ids) {
          const auto row = lifted_rows.find(lifted);
          if (row == lifted_rows.end() || row->second != quotient->second) {
            return TetEditStatus::kRejectedTopology;
          }
        }
      }
    }

    if (advance_generation && generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return TetEditStatus::kIndexOverflow;
    }
    auto used_orbits = used_periodic_orbit_ids_;
    auto used_quotients = used_quotient_cell_ids_;
    auto used_lifted = used_periodic_lifted_ids_;
    auto used_periodic_quotients = used_periodic_quotient_ids_;
    used_orbits.reserve(used_orbits.size() + staged_orbits.size());
    used_quotients.reserve(used_quotients.size() + staged_cells.size());
    for (const TetPeriodicOrbit& orbit : staged_orbits) used_orbits.insert(orbit.id);
    for (const TetPeriodicQuotientCell& cell : staged_cells) used_quotients.insert(cell.id);
    for (std::size_t dimension = 0; dimension < staged_maps.size(); ++dimension) {
      for (std::uint64_t id : staged_maps[dimension].lifted_entity_ids) {
        used_lifted[dimension].insert(id);
      }
      for (std::uint64_t id : staged_maps[dimension].quotient_entity_ids) {
        used_periodic_quotients[dimension].insert(id);
      }
    }
    periodic_orbits_.swap(staged_orbits);
    quotient_cells_.swap(staged_cells);
    periodic_entity_maps_.swap(staged_maps);
    used_periodic_orbit_ids_.swap(used_orbits);
    used_quotient_cell_ids_.swap(used_quotients);
    used_periodic_lifted_ids_.swap(used_lifted);
    used_periodic_quotient_ids_.swap(used_periodic_quotients);
    if (advance_generation) ++generation_;
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

bool TetMesh::validate_periodic_closure(const TetEdit& operation) const {
  const bool has_periodic_payload = !operation.periodic_members.empty() ||
      !operation.removed_periodic_orbit_ids.empty() ||
      !operation.replacement_periodic_orbits.empty() ||
      !operation.removed_quotient_cell_ids.empty() ||
      !operation.replacement_quotient_cells.empty() ||
      !operation.replacement_periodic_entity_maps.empty();
  const bool topology_touched = !operation.changed_vertices.empty() ||
      !operation.removed_vertex_ids.empty() || !operation.split_edges.empty() ||
      !operation.removed_cell_ids.empty() || !operation.replacement_cells.empty();
  const bool relocation = operation.kind == TetEditKind::kRelocate;
  if (periodic_identity_.bound && topology_touched &&
      operation.replacement_periodic_entity_maps.size() != 4) return false;
  if (periodic_identity_.bound && topology_touched) {
    const TetPeriodicEntityOrbitMap& vertices = periodic_entity_maps_[0];
    for (const TetEditVertex& vertex : operation.changed_vertices) {
      if (vertex_slot(vertex.id) >= 0) continue;
      if (vertex.ancestry.empty()) return false;
      for (const TetVertexAncestry& ancestor : vertex.ancestry) {
        const auto lifted = std::find(vertices.lifted_entity_ids.begin(),
                                      vertices.lifted_entity_ids.end(),
                                      ancestor.source_vertex_id);
        if (lifted == vertices.lifted_entity_ids.end()) return false;
        const std::size_t row = static_cast<std::size_t>(lifted - vertices.lifted_entity_ids.begin());
        const TetPeriodicVertexMember* member = nullptr;
        for (const TetPeriodicOrbit& orbit : periodic_orbits_) {
          const auto found = std::find_if(orbit.members.begin(), orbit.members.end(),
              [&](const TetPeriodicVertexMember& candidate) {
                return candidate.vertex_id == ancestor.source_vertex_id;
              });
          if (found == orbit.members.end()) continue;
          if (member != nullptr) return false;
          member = &*found;
        }
        if (member != nullptr) {
          if (member->winding != ancestor.source_winding) return false;
        } else {
          for (std::size_t axis = 0; axis < 3; ++axis) {
            const std::int64_t winding = axis < static_cast<std::size_t>(vertices.rank)
                ? vertices.winding_shifts[row * static_cast<std::size_t>(vertices.rank) + axis]
                : 0;
            if (ancestor.source_winding[axis] != winding) return false;
          }
        }
      }
    }
  } else {
    for (const TetEditVertex& vertex : operation.changed_vertices) {
      for (const TetVertexAncestry& ancestor : vertex.ancestry) {
        if (ancestor.source_winding[0] != 0 || ancestor.source_winding[1] != 0 ||
            ancestor.source_winding[2] != 0) return false;
      }
    }
  }
  std::unordered_set<std::uint64_t> removed_cells(operation.removed_cell_ids.begin(),
                                                   operation.removed_cell_ids.end());
  std::unordered_set<std::uint64_t> affected_vertices;
  for (const TetEditVertex& vertex : operation.changed_vertices) affected_vertices.insert(vertex.id);
  for (std::uint64_t vertex : operation.removed_vertex_ids) affected_vertices.insert(vertex);
  for (const auto& edge : operation.split_edges) {
    affected_vertices.insert(edge[0]);
    affected_vertices.insert(edge[1]);
  }
  if (operation.kind == TetEditKind::kCollapse) {
    for (const TetCollapsePair& pair : operation.collapse_pairs) {
      affected_vertices.insert(pair.removed_vertex_id);
      affected_vertices.insert(pair.survivor_vertex_id);
    }
  }
  if (!relocation) {
    for (std::uint64_t id : operation.removed_cell_ids) {
      const std::int32_t slot = cell_slot(id);
      if (slot < 0) return false;
      for (std::int32_t vertex : tet(slot).v) affected_vertices.insert(vertex_id(vertex));
    }
   }
  std::unordered_set<std::uint64_t> replacement_cell_ids;
  for (const TetEditCell& cell : operation.replacement_cells) {
    replacement_cell_ids.insert(cell.id);
    for (std::uint64_t vertex : cell.vertices) affected_vertices.insert(vertex);
  }

  std::unordered_set<std::uint64_t> periodic_members;
  for (const TetPeriodicEditMember& member : operation.periodic_members) {
    if (!periodic_members.insert(member.vertex_id).second ||
        !exact_isometry(member.representative_to_member)) return false;
  }
  std::unordered_set<std::uint64_t> removed_orbits;
  for (std::uint64_t id : operation.removed_periodic_orbit_ids) {
    if (!removed_orbits.insert(id).second ||
        std::none_of(periodic_orbits_.begin(), periodic_orbits_.end(),
                     [&](const TetPeriodicOrbit& orbit) { return orbit.id == id; })) {
      return false;
    }
  }
  std::unordered_set<std::uint64_t> replacement_orbit_ids;
  for (const TetPeriodicOrbit& orbit : operation.replacement_periodic_orbits) {
    if (!replacement_orbit_ids.insert(orbit.id).second || orbit.members.empty()) return false;
  }

  const auto edit_member = [&](std::uint64_t id) -> const TetPeriodicEditMember* {
    const auto found = std::find_if(operation.periodic_members.begin(),
        operation.periodic_members.end(),
        [&](const TetPeriodicEditMember& member) { return member.vertex_id == id; });
    return found == operation.periodic_members.end() ? nullptr : &*found;
  };
  const auto changed_vertex = [&](std::uint64_t id) -> const TetEditVertex* {
    const auto found = std::find_if(operation.changed_vertices.begin(),
        operation.changed_vertices.end(),
        [&](const TetEditVertex& vertex) { return vertex.id == id; });
    return found == operation.changed_vertices.end() ? nullptr : &*found;
  };
  const auto position = [&](std::uint64_t id) -> const double* {
    if (const TetEditVertex* changed = changed_vertex(id)) return changed->carrier.data();
    const std::int32_t slot = vertex_slot(id);
    return live_vertex(slot) ? point(slot) : nullptr;
  };

  std::unordered_set<std::uint64_t> touched_vertex_orbits;
  for (const TetPeriodicOrbit& orbit : periodic_orbits_) {
    bool touched = false;
    std::size_t removed_count = 0;
    for (const TetPeriodicVertexMember& member : orbit.members) {
      touched = touched || affected_vertices.contains(member.vertex_id);
      removed_count += std::find(operation.removed_vertex_ids.begin(),
                                 operation.removed_vertex_ids.end(), member.vertex_id) !=
                       operation.removed_vertex_ids.end();
    }
    if (!touched) continue;
    touched_vertex_orbits.insert(orbit.id);
    if (!has_periodic_payload) return false;
    // A fundamental-domain lift edits the star of one copy of an orbit while the
    // other copies keep theirs; every copy still needs its exact edit member and
    // a surviving copy must remain the exact image of the representative.
    std::size_t surviving_count = 0;
    for (const TetPeriodicVertexMember& member : orbit.members) {
      const TetPeriodicEditMember* edit = edit_member(member.vertex_id);
      if (edit == nullptr || edit->winding != member.winding) return false;
      for (std::size_t component = 0; component < member.representative_to_member.size(); ++component) {
        if (!same_bits(edit->representative_to_member[component],
                       member.representative_to_member[component])) return false;
      }
      if (std::find(operation.removed_vertex_ids.begin(), operation.removed_vertex_ids.end(),
                    member.vertex_id) != operation.removed_vertex_ids.end()) continue;
      ++surviving_count;
      const double* representative = position(orbit.representative_vertex_id);
      const double* actual = position(member.vertex_id);
      if (representative == nullptr || actual == nullptr) return false;
      double expected[3];
      if (!affine_image(member.representative_to_member, representative, expected)) return false;
      for (int axis = 0; axis < 3; ++axis) {
        if (!same_bits(expected[axis], actual[axis])) return false;
      }
    }
    if (removed_count > 0) {
      if (!removed_orbits.contains(orbit.id)) return false;
      if (surviving_count > 0 && !replacement_orbit_ids.contains(orbit.id)) return false;
    }
  }

  for (const TetPeriodicEditMember& member : operation.periodic_members) {
    bool in_existing_orbit = false;
    for (const TetPeriodicOrbit& orbit : periodic_orbits_) {
      const auto current = std::find_if(orbit.members.begin(), orbit.members.end(),
          [&](const TetPeriodicVertexMember& candidate) {
            return candidate.vertex_id == member.vertex_id;
          });
      if (current != orbit.members.end()) {
        in_existing_orbit = true;
        break;
      }
    }
    if (in_existing_orbit) continue;
    if (changed_vertex(member.vertex_id) == nullptr) return false;
    bool described = false;
    for (const TetPeriodicOrbit& orbit : operation.replacement_periodic_orbits) {
      const auto replacement = std::find_if(orbit.members.begin(), orbit.members.end(),
          [&](const TetPeriodicVertexMember& candidate) {
            return candidate.vertex_id == member.vertex_id;
          });
      if (replacement == orbit.members.end()) continue;
      if (replacement->winding != member.winding) return false;
      for (std::size_t component = 0; component < member.representative_to_member.size(); ++component) {
        if (!same_bits(replacement->representative_to_member[component],
                       member.representative_to_member[component])) return false;
      }
      described = true;
    }
    if (!described) return false;
  }

  std::unordered_set<std::uint64_t> removed_quotients;
  for (std::uint64_t id : operation.removed_quotient_cell_ids) {
    if (!removed_quotients.insert(id).second ||
        std::none_of(quotient_cells_.begin(), quotient_cells_.end(),
                     [&](const TetPeriodicQuotientCell& cell) { return cell.id == id; })) {
      return false;
    }
  }
  std::unordered_set<std::uint64_t> touched_quotient_orbits;
  for (const TetPeriodicQuotientCell& cell : quotient_cells_) {
    const bool touched = !relocation && std::any_of(cell.lifted_cell_ids.begin(),
        cell.lifted_cell_ids.end(),
        [&](std::uint64_t lifted) { return removed_cells.contains(lifted); });
    if (!touched) continue;
    if (!removed_quotients.contains(cell.id)) return false;
    for (std::uint64_t lifted : cell.lifted_cell_ids) {
      if (!removed_cells.contains(lifted)) return false;
    }
    touched_quotient_orbits.insert(cell.orbit_id);
  }
  for (std::uint64_t orbit : touched_quotient_orbits) {
    for (const TetPeriodicQuotientCell& cell : quotient_cells_) {
      if (cell.orbit_id == orbit && !removed_quotients.contains(cell.id)) return false;
    }
  }
  if (!touched_quotient_orbits.empty()) {
    std::unordered_set<std::uint64_t> covered_replacement_cells;
    for (const TetPeriodicQuotientCell& cell : operation.replacement_quotient_cells) {
      if (cell.lifted_cell_ids.empty()) return false;
      for (std::uint64_t lifted : cell.lifted_cell_ids) {
        if (!replacement_cell_ids.contains(lifted) ||
            !covered_replacement_cells.insert(lifted).second) return false;
      }
    }
    for (std::uint64_t replacement : replacement_cell_ids) {
      if (!covered_replacement_cells.contains(replacement)) return false;
    }
  }

  if (!has_periodic_payload && !touched_vertex_orbits.empty()) return false;
  if (!has_periodic_payload && !touched_quotient_orbits.empty()) return false;
  return true;
}

TetMesh::PeriodicMemberIndex TetMesh::periodic_member_index() const {
  PeriodicMemberIndex index;
  for (std::size_t orbit = 0; orbit < periodic_orbits_.size(); ++orbit) {
    const std::vector<TetPeriodicVertexMember>& members = periodic_orbits_[orbit].members;
    for (std::size_t member = 0; member < members.size(); ++member) {
      index.emplace(members[member].vertex_id, std::pair{orbit, member});
    }
  }
  return index;
}

// A constrained facet of the lifted fundamental cover is a periodic seam when
// one nonzero lattice winding maps it onto another constrained cover facet:
// every vertex pair belongs to one orbit, the member windings differ by the
// same shift, and both members are the exact deck images of their orbit
// representative. The quotient identifies the two facets into one interior
// face, so the facet is not a physical boundary and its source group does not
// constrain the motion of a whole orbit. A deck image of the facet's own cell
// that bounds the partner from the same side would make the pair a periodic
// copy of one physical wall instead; that pair is refused.
bool TetMesh::periodic_seam_facet(const TetSnapshotFacet& facet,
                                  const PeriodicMemberIndex& members) const {
  using Winding = std::array<std::int64_t, 3>;
  const auto member_at = [&](std::uint64_t id) -> const TetPeriodicVertexMember* {
    const auto found = members.find(id);
    return found == members.end()
               ? nullptr
               : &periodic_orbits_[found->second.first].members[found->second.second];
  };
  const auto orbit_of = [&](std::uint64_t id) -> std::size_t {
    const auto found = members.find(id);
    return found == members.end() ? periodic_orbits_.size() : found->second.first;
  };
  const auto shift_between = [](const Winding& from, const Winding& to) {
    return Winding{to[0] - from[0], to[1] - from[1], to[2] - from[2]};
  };
  const auto deck_image = [&](std::size_t orbit, const TetPeriodicVertexMember& member) {
    const std::int32_t representative =
        vertex_slot(periodic_orbits_[orbit].representative_vertex_id);
    const std::int32_t slot = vertex_slot(member.vertex_id);
    if (!live_vertex(representative) || !live_vertex(slot)) return false;
    double expected[3];
    if (!affine_image(member.representative_to_member, point(representative), expected)) {
      return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!same_bits(expected[axis], point(slot)[axis])) return false;
    }
    return true;
  };
  // The apex of the only live cell incident to a cover facet, or -1.
  const auto single_apex = [&](const std::array<std::uint64_t, 3>& face) -> std::int32_t {
    std::int32_t slots[3];
    for (int k = 0; k < 3; ++k) {
      slots[k] = vertex_slot(face[static_cast<std::size_t>(k)]);
      if (!live_vertex(slots[k])) return -1;
    }
    std::int32_t apex = -1;
    for (std::size_t row = 0; row < complex_.tets.size(); ++row) {
      const std::int32_t cell = static_cast<std::int32_t>(row);
      if (!complex_.live(cell) || is_ghost(complex_.tets[row])) continue;
      const Tetrahedron& tet = complex_.tets[row];
      if (phx::mc::vertex_slot(tet, slots[0]) < 0 || phx::mc::vertex_slot(tet, slots[1]) < 0 ||
          phx::mc::vertex_slot(tet, slots[2]) < 0) continue;
      if (apex >= 0) return -1;
      for (std::int32_t vertex : tet.v) {
        if (vertex != slots[0] && vertex != slots[1] && vertex != slots[2]) apex = vertex;
      }
    }
    return apex;
  };

  std::array<std::size_t, 3> orbits{};
  std::array<const TetPeriodicVertexMember*, 3> own{};
  for (std::size_t k = 0; k < 3; ++k) {
    own[k] = member_at(facet.vertices[k]);
    if (own[k] == nullptr) return false;
    orbits[k] = orbit_of(facet.vertices[k]);
    for (std::size_t j = 0; j < k; ++j) {
      if (orbits[j] == orbits[k]) return false;
    }
  }
  for (const TetSnapshotFacet& partner : facets_) {
    if (partner.id == facet.id) continue;
    std::array<const TetPeriodicVertexMember*, 3> image{};
    bool paired = true;
    for (std::size_t k = 0; k < 3 && paired; ++k) {
      paired = false;
      for (std::uint64_t id : partner.vertices) {
        if (orbit_of(id) != orbits[k]) continue;
        image[k] = member_at(id);
        paired = true;
        break;
      }
    }
    if (!paired) continue;
    const Winding shift = shift_between(own[0]->winding, image[0]->winding);
    if (shift == Winding{} ||
        shift_between(own[1]->winding, image[1]->winding) != shift ||
        shift_between(own[2]->winding, image[2]->winding) != shift) {
      continue;
    }
    for (std::size_t k = 0; k < 3; ++k) {
      if (!deck_image(orbits[k], *own[k]) || !deck_image(orbits[k], *image[k])) return false;
    }
    const std::int32_t own_apex = single_apex(facet.vertices);
    const std::int32_t partner_apex = single_apex(partner.vertices);
    if (own_apex < 0 || partner_apex < 0) return false;
    const TetPeriodicVertexMember* own_apex_member = member_at(vertex_id(own_apex));
    const TetPeriodicVertexMember* partner_apex_member = member_at(vertex_id(partner_apex));
    if (own_apex_member != nullptr && partner_apex_member != nullptr &&
        orbit_of(vertex_id(own_apex)) == orbit_of(vertex_id(partner_apex)) &&
        shift_between(own_apex_member->winding, partner_apex_member->winding) == shift) {
      return false;
    }
    return true;
  }
  return false;
}

}  // namespace phx::mc
