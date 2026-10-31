//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary for persistent source-aware tetrahedral meshes.
#include "capi_guard.hpp"
#include "phydrax_meshcore.h"
#include "refine3d.hpp"
#include "improve3d.hpp"
#include "source_construction.hpp"
#include "tet_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <array>
#include <vector>

struct phx_mc_tet_mesh {
  explicit phx_mc_tet_mesh(phx::mc::BoundaryPolicy policy) : state(policy) {}
  phx::mc::TetMesh state;
};

namespace {

using phx::mc::addressable;
using phx::mc::guarded;
using phx::mc::SourceStratum;
using phx::mc::SourceWitness;
using phx::mc::TetEdit;
using phx::mc::TetEditBatch;
using phx::mc::TetEditCell;
using phx::mc::TetEditFacet;
using phx::mc::TetEditKind;
using phx::mc::TetEditSegment;
using phx::mc::TetEditStatus;
using phx::mc::TetEditVertex;
using phx::mc::TetMesh;
using phx::mc::TetMeshSnapshot;
using phx::mc::TetSnapshotCell;
using phx::mc::TetSnapshotFacet;
using phx::mc::TetSnapshotSegment;
using phx::mc::TetSnapshotVertex;
using phx::mc::TetPeriodicEntityOrbitMap;
using phx::mc::TetPeriodicIdentity;
using phx::mc::TetPeriodicOrbit;
using phx::mc::TetPeriodicQuotientCell;
using phx::mc::TetPeriodicVertexMember;

bool required_bank(const void* pointer, int64_t count) noexcept {
  return count == 0 || pointer != nullptr;
}

template <class T>
std::span<const T> input_span(const T* data, int64_t count) {
  return {data, static_cast<std::size_t>(count)};
}

template <class T>
void copy_bank(const std::vector<T>& source, T* destination) {
  if (!source.empty()) std::copy(source.begin(), source.end(), destination);
}

template <class T>
void assign_bank(std::vector<T>& destination, const T* source, int64_t count) {
  if (count > 0) destination.assign(source, source + count);
}

bool valid_utf8(const char* bytes, int64_t length) noexcept {
  if (length < 0 || (length > 0 && bytes == nullptr) ||
      !addressable(length, 1, sizeof(char))) return false;
  const auto* data = reinterpret_cast<const unsigned char*>(bytes);
  const std::size_t size = static_cast<std::size_t>(length);
  for (std::size_t i = 0; i < size;) {
    const unsigned char first = data[i++];
    if (first <= 0x7f) continue;
    if (first >= 0xc2 && first <= 0xdf) {
      if (i >= size || (data[i++] & 0xc0) != 0x80) return false;
      continue;
    }
    if (first >= 0xe0 && first <= 0xef) {
      if (i + 1 >= size) return false;
      const unsigned char second = data[i++];
      const unsigned char third = data[i++];
      if ((third & 0xc0) != 0x80 ||
          (first == 0xe0 ? second < 0xa0 || second > 0xbf
                         : first == 0xed ? second < 0x80 || second > 0x9f
                                         : (second & 0xc0) != 0x80)) {
        return false;
      }
      continue;
    }
    if (first >= 0xf0 && first <= 0xf4) {
      if (i + 2 >= size) return false;
      const unsigned char second = data[i++];
      const unsigned char third = data[i++];
      const unsigned char fourth = data[i++];
      if ((third & 0xc0) != 0x80 || (fourth & 0xc0) != 0x80 ||
          (first == 0xf0 ? second < 0x90 || second > 0xbf
                         : first == 0xf4 ? second < 0x80 || second > 0x8f
                                         : (second & 0xc0) != 0x80)) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

std::string_view identity_string(const char* bytes, int64_t length) noexcept {
  return length == 0 ? std::string_view{}
                     : std::string_view(bytes, static_cast<std::size_t>(length));
}

bool valid_identity_fields(int32_t bound, const char* source_id,
                           int64_t source_id_length, const char* source_revision,
                           int64_t source_revision_length, const char* source_map_id,
                           int64_t source_map_id_length, const char* frame_id,
                           int64_t frame_id_length) noexcept {
  if (bound != 0 && bound != 1) return false;
  if (bound == 0) {
    return source_id == nullptr && source_id_length == 0 &&
           source_revision == nullptr && source_revision_length == 0 &&
           source_map_id == nullptr && source_map_id_length == 0 &&
           frame_id == nullptr && frame_id_length == 0;
  }
  return source_revision_length > 0 && frame_id_length > 0 &&
         source_id_length >= 0 && source_map_id_length >= 0 &&
         valid_utf8(source_id, source_id_length) &&
         valid_utf8(source_revision, source_revision_length) &&
         valid_utf8(source_map_id, source_map_id_length) &&
         valid_utf8(frame_id, frame_id_length);
}

bool identity_matches(const TetMesh& mesh, int32_t bound,
                      const char* source_id, int64_t source_id_length,
                      const char* source_revision, int64_t source_revision_length,
                      const char* source_map_id, int64_t source_map_id_length,
                      const char* frame_id, int64_t frame_id_length) noexcept {
  if (bound != static_cast<int32_t>(mesh.identity_bound())) return false;
  if (bound == 0) return true;
  return identity_string(source_id, source_id_length) == mesh.source_id() &&
         identity_string(source_revision, source_revision_length) == mesh.source_revision() &&
         identity_string(source_map_id, source_map_id_length) == mesh.source_map_id() &&
         identity_string(frame_id, frame_id_length) == mesh.frame_id();
}

bool valid_range(int64_t offset, int64_t count, int64_t capacity) noexcept {
  return offset >= 0 && count >= 0 && capacity >= 0 && offset <= capacity &&
         count <= capacity - offset;
}

bool valid_periodic_identity_map_view(
    const phx_mc_tet_periodic_identity_map_view& view) noexcept {
  if (view.identity_bound != 0 && view.identity_bound != 1) return false;
  const char* identity_bytes[] = {
      view.source_mesh_id, view.source_topology_id, view.lifted_topology_id,
      view.periodic_topology_id, view.cell_entity_set_id, view.edge_entity_set_id};
  const int64_t identity_lengths[] = {
      view.source_mesh_id_length, view.source_topology_id_length,
      view.lifted_topology_id_length, view.periodic_topology_id_length,
      view.cell_entity_set_id_length, view.edge_entity_set_id_length};
  if (view.identity_bound == 0) {
    for (int field = 0; field < 6; ++field) {
      if (identity_bytes[field] != nullptr || identity_lengths[field] != 0) return false;
    }
    return view.entity_map_count == 0 && view.entity_maps == nullptr &&
           view.entity_lifted_count == 0 && view.entity_lifted_ids == nullptr &&
           view.entity_lifted_to_quotient == nullptr &&
           view.entity_orientation_signs == nullptr &&
           view.entity_quotient_count == 0 && view.entity_quotient_ids == nullptr &&
           view.entity_winding_count == 0 && view.entity_winding_shifts == nullptr &&
           view.entity_corner_count == 0 && view.entity_corner_permutations == nullptr &&
           view.entity_representative_count == 0 && view.entity_representative_rows == nullptr;
  }
  for (int field = 0; field < 6; ++field) {
    if (identity_lengths[field] <= 0 ||
        !valid_utf8(identity_bytes[field], identity_lengths[field])) return false;
  }
  const int64_t counts[] = {
      view.entity_map_count, view.entity_lifted_count, view.entity_quotient_count,
      view.entity_winding_count, view.entity_corner_count,
      view.entity_representative_count};
  const void* pointers[] = {
      view.entity_maps, view.entity_lifted_ids, view.entity_quotient_ids,
      view.entity_winding_shifts, view.entity_corner_permutations,
      view.entity_representative_rows};
  const std::size_t widths[] = {
      sizeof(phx_mc_tet_periodic_entity_orbit_map), sizeof(uint64_t), sizeof(uint64_t),
      sizeof(int64_t), sizeof(int32_t), sizeof(int32_t)};
  for (int bank = 0; bank < 6; ++bank) {
    if (counts[bank] < 0 || !addressable(counts[bank], 1, widths[bank]) ||
        !required_bank(pointers[bank], counts[bank])) return false;
  }
  if (!addressable(view.entity_lifted_count, 1, sizeof(int32_t)) ||
      !required_bank(view.entity_lifted_to_quotient, view.entity_lifted_count) ||
      !required_bank(view.entity_orientation_signs, view.entity_lifted_count)) return false;
  if (view.entity_map_count != 4) return false;
  int64_t lifted_offset = 0;
  int64_t quotient_offset = 0;
  int64_t winding_offset = 0;
  int64_t corner_offset = 0;
  int64_t representative_offset = 0;
  std::int32_t periodic_rank = -1;
  for (int32_t dimension = 0; dimension < 4; ++dimension) {
    const phx_mc_tet_periodic_entity_orbit_map& map = view.entity_maps[dimension];
    if (map.dimension != dimension || map.arity != dimension + 1 ||
        map.rank < 1 || map.rank > 3 || map.lifted_count <= 0 ||
        map.quotient_count <= 0 || map.lifted_offset != lifted_offset ||
        map.quotient_offset != quotient_offset || map.winding_offset != winding_offset ||
        map.corner_permutation_offset != corner_offset ||
        map.representative_offset != representative_offset ||
        !valid_range(map.lifted_offset, map.lifted_count, view.entity_lifted_count) ||
        !valid_range(map.quotient_offset, map.quotient_count, view.entity_quotient_count) ||
        !valid_range(map.representative_offset, map.quotient_count,
                     view.entity_representative_count) ||
        map.lifted_count > std::numeric_limits<int64_t>::max() / map.rank ||
        map.lifted_count > std::numeric_limits<int64_t>::max() / map.arity) return false;
    if (periodic_rank < 0) periodic_rank = map.rank;
    if (periodic_rank != map.rank) return false;
    const int64_t winding_count = map.lifted_count * map.rank;
    const int64_t corner_count = map.lifted_count * map.arity;
    if (!valid_range(map.winding_offset, winding_count, view.entity_winding_count) ||
        !valid_range(map.corner_permutation_offset, corner_count, view.entity_corner_count) ||
        lifted_offset > std::numeric_limits<int64_t>::max() - map.lifted_count ||
        quotient_offset > std::numeric_limits<int64_t>::max() - map.quotient_count ||
        winding_offset > std::numeric_limits<int64_t>::max() - winding_count ||
        corner_offset > std::numeric_limits<int64_t>::max() - corner_count ||
        representative_offset > std::numeric_limits<int64_t>::max() - map.quotient_count) {
      return false;
    }
    lifted_offset += map.lifted_count;
    quotient_offset += map.quotient_count;
    winding_offset += winding_count;
    corner_offset += corner_count;
    representative_offset += map.quotient_count;
  }
  return lifted_offset == view.entity_lifted_count &&
         quotient_offset == view.entity_quotient_count &&
         winding_offset == view.entity_winding_count &&
         corner_offset == view.entity_corner_count &&
         representative_offset == view.entity_representative_count;
}

bool valid_periodic_state_view(const phx_mc_tet_periodic_state_view& view) noexcept {
  if (!valid_periodic_identity_map_view(view.identity_map)) return false;
  const int64_t counts[] = {view.periodic_orbit_count, view.periodic_member_count,
                            view.periodic_quotient_cell_count,
                            view.periodic_quotient_lifted_count};
  const void* pointers[] = {view.periodic_orbits, view.periodic_members,
                            view.periodic_quotient_cells,
                            view.periodic_quotient_lifted_cell_ids};
  const std::size_t widths[] = {
      sizeof(phx_mc_tet_periodic_orbit), sizeof(phx_mc_tet_periodic_vertex_member),
      sizeof(phx_mc_tet_periodic_quotient_cell), sizeof(uint64_t)};
  for (int bank = 0; bank < 4; ++bank) {
    if (counts[bank] < 0 || !addressable(counts[bank], 1, widths[bank]) ||
        !required_bank(pointers[bank], counts[bank])) return false;
  }
  if (!view.identity_map.identity_bound &&
      (view.periodic_orbit_count != 0 || view.periodic_member_count != 0 ||
       view.periodic_quotient_cell_count != 0 || view.periodic_quotient_lifted_count != 0)) {
    return false;
  }
  for (int64_t row = 0; row < view.periodic_orbit_count; ++row) {
    const auto& orbit = view.periodic_orbits[row];
    if (!valid_range(orbit.member_offset, orbit.member_count, view.periodic_member_count)) {
      return false;
    }
  }
  for (int64_t row = 0; row < view.periodic_quotient_cell_count; ++row) {
    const auto& cell = view.periodic_quotient_cells[row];
    if (!valid_range(cell.lifted_cell_offset, cell.lifted_cell_count,
                     view.periodic_quotient_lifted_count)) return false;
  }
  return true;
}

bool valid_periodic_edit_view(const phx_mc_tet_periodic_edit_view& view) noexcept {
  const int64_t counts[] = {
      view.periodic_member_count, view.removed_periodic_orbit_count,
      view.replacement_periodic_orbit_count, view.replacement_periodic_member_count,
      view.removed_quotient_cell_count, view.replacement_quotient_cell_count,
      view.replacement_quotient_lifted_count};
  const void* pointers[] = {
      view.periodic_members, view.removed_periodic_orbit_ids,
      view.replacement_periodic_orbits, view.replacement_periodic_members,
      view.removed_quotient_cell_ids, view.replacement_quotient_cells,
      view.replacement_quotient_lifted_cell_ids};
  const std::size_t widths[] = {
      sizeof(phx_mc_tet_periodic_vertex_member), sizeof(uint64_t),
      sizeof(phx_mc_tet_periodic_orbit), sizeof(phx_mc_tet_periodic_vertex_member),
      sizeof(uint64_t), sizeof(phx_mc_tet_periodic_quotient_cell), sizeof(uint64_t)};
  for (int bank = 0; bank < 7; ++bank) {
    if (counts[bank] < 0 || !addressable(counts[bank], 1, widths[bank]) ||
        !required_bank(pointers[bank], counts[bank])) return false;
  }
  for (int64_t row = 0; row < view.replacement_periodic_orbit_count; ++row) {
    const auto& orbit = view.replacement_periodic_orbits[row];
    if (!valid_range(orbit.member_offset, orbit.member_count,
                     view.replacement_periodic_member_count)) return false;
  }
  for (int64_t row = 0; row < view.replacement_quotient_cell_count; ++row) {
    const auto& cell = view.replacement_quotient_cells[row];
    if (!valid_range(cell.lifted_cell_offset, cell.lifted_cell_count,
                     view.replacement_quotient_lifted_count)) return false;
  }
  return view.target_identity_map == nullptr ||
         valid_periodic_identity_map_view(*view.target_identity_map);
}

TetPeriodicIdentity periodic_identity_from_view(
    const phx_mc_tet_periodic_identity_map_view& view) {
  TetPeriodicIdentity identity;
  identity.bound = view.identity_bound != 0;
  if (!identity.bound) return identity;
  const char* bytes[] = {view.source_mesh_id, view.source_topology_id,
                         view.lifted_topology_id, view.periodic_topology_id,
                         view.cell_entity_set_id, view.edge_entity_set_id};
  const int64_t lengths[] = {view.source_mesh_id_length, view.source_topology_id_length,
                             view.lifted_topology_id_length, view.periodic_topology_id_length,
                             view.cell_entity_set_id_length, view.edge_entity_set_id_length};
  for (std::size_t field = 0; field < identity.ids.size(); ++field) {
    identity.ids[field].assign(bytes[field], static_cast<std::size_t>(lengths[field]));
  }
  return identity;
}

std::vector<TetPeriodicEntityOrbitMap> periodic_entity_maps_from_view(
    const phx_mc_tet_periodic_identity_map_view& view) {
  std::vector<TetPeriodicEntityOrbitMap> maps;
  maps.reserve(static_cast<std::size_t>(view.entity_map_count));
  for (int64_t row = 0; row < view.entity_map_count; ++row) {
    const phx_mc_tet_periodic_entity_orbit_map& descriptor = view.entity_maps[row];
    TetPeriodicEntityOrbitMap map;
    map.dimension = descriptor.dimension;
    map.arity = descriptor.arity;
    map.rank = descriptor.rank;
    const std::size_t lifted_offset = static_cast<std::size_t>(descriptor.lifted_offset);
    const std::size_t lifted_count = static_cast<std::size_t>(descriptor.lifted_count);
    const std::size_t quotient_offset = static_cast<std::size_t>(descriptor.quotient_offset);
    const std::size_t quotient_count = static_cast<std::size_t>(descriptor.quotient_count);
    const std::size_t winding_offset = static_cast<std::size_t>(descriptor.winding_offset);
    const std::size_t winding_count = lifted_count * static_cast<std::size_t>(descriptor.rank);
    const std::size_t corner_offset = static_cast<std::size_t>(descriptor.corner_permutation_offset);
    const std::size_t corner_count = lifted_count * static_cast<std::size_t>(descriptor.arity);
    const std::size_t representative_offset =
        static_cast<std::size_t>(descriptor.representative_offset);
    map.lifted_entity_ids.assign(view.entity_lifted_ids + lifted_offset,
                                 view.entity_lifted_ids + lifted_offset + lifted_count);
    map.quotient_entity_ids.assign(view.entity_quotient_ids + quotient_offset,
                                   view.entity_quotient_ids + quotient_offset + quotient_count);
    map.lifted_to_quotient.assign(view.entity_lifted_to_quotient + lifted_offset,
                                  view.entity_lifted_to_quotient + lifted_offset + lifted_count);
    map.orientation_signs.assign(view.entity_orientation_signs + lifted_offset,
                                 view.entity_orientation_signs + lifted_offset + lifted_count);
    map.winding_shifts.assign(view.entity_winding_shifts + winding_offset,
                              view.entity_winding_shifts + winding_offset + winding_count);
    map.corner_permutations.assign(view.entity_corner_permutations + corner_offset,
                                   view.entity_corner_permutations + corner_offset + corner_count);
    map.representative_lifted_rows.assign(
        view.entity_representative_rows + representative_offset,
        view.entity_representative_rows + representative_offset + quotient_count);
    maps.push_back(std::move(map));
  }
  return maps;
}

void periodic_state_from_view(const phx_mc_tet_periodic_state_view& view,
                              std::vector<TetPeriodicOrbit>& orbits,
                              std::vector<TetPeriodicQuotientCell>& cells,
                              std::vector<TetPeriodicEntityOrbitMap>& maps) {
  orbits.reserve(static_cast<std::size_t>(view.periodic_orbit_count));
  for (int64_t row = 0; row < view.periodic_orbit_count; ++row) {
    const phx_mc_tet_periodic_orbit& descriptor = view.periodic_orbits[row];
    TetPeriodicOrbit orbit;
    orbit.id = descriptor.id;
    orbit.representative_vertex_id = descriptor.representative_vertex_id;
    orbit.members.reserve(static_cast<std::size_t>(descriptor.member_count));
    for (int64_t member_row = 0; member_row < descriptor.member_count; ++member_row) {
      const phx_mc_tet_periodic_vertex_member& input =
          view.periodic_members[descriptor.member_offset + member_row];
      TetPeriodicVertexMember member;
      member.vertex_id = input.vertex_id;
      std::copy_n(input.winding, 3, member.winding.begin());
      std::copy_n(input.representative_to_member, 12,
                  member.representative_to_member.begin());
      orbit.members.push_back(member);
    }
    orbits.push_back(std::move(orbit));
  }
  cells.reserve(static_cast<std::size_t>(view.periodic_quotient_cell_count));
  for (int64_t row = 0; row < view.periodic_quotient_cell_count; ++row) {
    const phx_mc_tet_periodic_quotient_cell& input = view.periodic_quotient_cells[row];
    TetPeriodicQuotientCell cell;
    cell.orbit_id = input.orbit_id;
    cell.id = input.id;
    std::copy_n(input.representative_vertices, 4, cell.representative_vertices.begin());
    for (int corner = 0; corner < 4; ++corner) {
      std::copy_n(input.winding + 3 * corner, 3, cell.winding[corner].begin());
    }
    cell.region = input.region;
    cell.lifted_cell_ids.assign(
        view.periodic_quotient_lifted_cell_ids + input.lifted_cell_offset,
        view.periodic_quotient_lifted_cell_ids + input.lifted_cell_offset +
            input.lifted_cell_count);
    cells.push_back(std::move(cell));
  }
  maps = periodic_entity_maps_from_view(view.identity_map);
}

bool tet_snapshot_counts(const TetMeshSnapshot& snapshot,
                         std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES>& counts) noexcept {
  std::size_t members = 0;
  std::size_t quotient_lifted = 0;
  std::size_t entity_lifted = 0;
  std::size_t entity_quotient = 0;
  std::size_t entity_winding = 0;
  std::size_t entity_corners = 0;
  std::size_t entity_representatives = 0;
  std::size_t vertex_ancestry = 0;
  std::size_t cell_ancestors = 0;
  const auto add = [](std::size_t value, std::size_t& total) {
    if (value > std::numeric_limits<std::size_t>::max() - total) return false;
    total += value;
    return true;
  };
  for (const TetPeriodicOrbit& orbit : snapshot.periodic_orbits) {
    if (!add(orbit.members.size(), members)) return false;
  }
  for (const TetPeriodicQuotientCell& cell : snapshot.quotient_cells) {
    if (!add(cell.lifted_cell_ids.size(), quotient_lifted)) return false;
  }
  for (const TetPeriodicEntityOrbitMap& map : snapshot.periodic_entity_maps) {
    if (!add(map.lifted_entity_ids.size(), entity_lifted) ||
        !add(map.quotient_entity_ids.size(), entity_quotient) ||
        !add(map.winding_shifts.size(), entity_winding) ||
        !add(map.corner_permutations.size(), entity_corners) ||
        !add(map.representative_lifted_rows.size(), entity_representatives)) return false;
  }
  for (const TetSnapshotVertex& vertex : snapshot.vertices) {
    if (!add(vertex.ancestry.size(), vertex_ancestry)) return false;
  }
  for (const TetSnapshotCell& cell : snapshot.cells) {
    if (!add(cell.ancestor_cell_ids.size(), cell_ancestors)) return false;
  }
  const std::size_t values[PHX_MC_TET_MESH_SNAPSHOT_SIZES] = {
      snapshot.vertices.size(), snapshot.cells.size(), snapshot.constrained_facets.size(),
      snapshot.constrained_segments.size(), snapshot.source_points.size() / 3,
      snapshot.source_faces.size() / 3, snapshot.source_segments.size() / 2,
      snapshot.periodic_orbits.size(), members, snapshot.quotient_cells.size(),
      quotient_lifted, snapshot.periodic_entity_maps.size(), entity_lifted,
      entity_quotient, entity_winding, entity_corners, entity_representatives,
      vertex_ancestry, cell_ancestors};
  for (std::size_t index = 0; index < counts.size(); ++index) {
    if (values[index] > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    counts[index] = static_cast<int64_t>(values[index]);
  }
  return true;
}

bool valid_create_request(const phx_mc_tet_mesh_create_request& request) noexcept {
  if (request.result == nullptr || request.point_count < 0 || request.tetrahedron_count < 0 ||
      request.face_count < 0 || request.segment_count < 0 || request.source_point_count < 0 ||
      request.source_face_count < 0 || request.source_segment_count < 0 ||
      !valid_identity_fields(request.identity_bound, request.source_id,
                             request.source_id_length, request.source_revision,
                             request.source_revision_length, request.source_map_id,
                             request.source_map_id_length, request.frame_id,
                             request.frame_id_length) ||
      (request.periodic_state != nullptr &&
       !valid_periodic_state_view(*request.periodic_state))) {
    return false;
  }
  constexpr int64_t index_limit = std::numeric_limits<int32_t>::max();
  if (request.point_count >= index_limit || request.tetrahedron_count >= index_limit ||
      request.source_point_count >= index_limit || request.source_face_count >= index_limit ||
      request.source_segment_count >= index_limit) {
    return false;
  }
  if (!addressable(request.point_count, 3, sizeof(double)) ||
      !addressable(request.tetrahedron_count, 4, sizeof(int32_t)) ||
      !addressable(request.tetrahedron_count, 1, sizeof(int32_t)) ||
      !addressable(request.face_count, 3, sizeof(int32_t)) ||
      !addressable(request.face_count, 1, sizeof(int32_t)) ||
      !addressable(request.segment_count, 2, sizeof(int32_t)) ||
      !addressable(request.segment_count, 1, sizeof(int32_t)) ||
      !addressable(request.point_count, 1, sizeof(double)) ||
      !addressable(request.point_count, 1, sizeof(int8_t)) ||
      !addressable(request.point_count, 1, sizeof(uint64_t)) ||
      !addressable(request.tetrahedron_count, 1, sizeof(uint64_t)) ||
      !addressable(request.face_count, 1, sizeof(uint64_t)) ||
      !addressable(request.segment_count, 1, sizeof(uint64_t)) ||
      !addressable(request.source_point_count, 3, sizeof(double)) ||
      !addressable(request.source_face_count, 3, sizeof(int32_t)) ||
      !addressable(request.source_face_count, 1, sizeof(int64_t)) ||
      !addressable(request.source_face_count, 1, sizeof(int32_t)) ||
      !addressable(request.source_segment_count, 2, sizeof(int32_t)) ||
      !addressable(request.source_segment_count, 1, sizeof(int64_t)) ||
      !addressable(request.source_segment_count, 1, sizeof(int32_t))) {
    return false;
  }
  if (!required_bank(request.points, request.point_count) ||
      !required_bank(request.tetrahedra, request.tetrahedron_count) ||
      !required_bank(request.faces, request.face_count) ||
      !required_bank(request.face_sources, request.face_count) ||
      !required_bank(request.segments, request.segment_count) ||
      !required_bank(request.segment_sources, request.segment_count) ||
      (request.vertex_ids != nullptr && request.point_count == 0) ||
      (request.cell_ids != nullptr && request.tetrahedron_count == 0) ||
      (request.facet_ids != nullptr && request.face_count == 0) ||
      (request.segment_ids != nullptr && request.segment_count == 0) ||
      !required_bank(request.source_points, request.source_point_count) ||
      !required_bank(request.source_faces, request.source_face_count) ||
      !required_bank(request.source_face_groups, request.source_face_count) ||
      !required_bank(request.source_segments, request.source_segment_count) ||
      !required_bank(request.source_segment_groups, request.source_segment_count) ||
      (request.source_face_ids != nullptr && request.source_face_count == 0) ||
      (request.source_segment_ids != nullptr && request.source_segment_count == 0) ||
      (request.source_point_count > 0 && request.source_face_count > 0 &&
       request.source_face_ids == nullptr) ||
      (request.source_point_count > 0 && request.source_segment_count > 0 &&
       request.source_segment_ids == nullptr) ||
      (request.tetrahedron_regions != nullptr && request.tetrahedron_count == 0) ||
      (request.face_orientations != nullptr && request.face_count == 0) ||
      (request.protection_radii != nullptr && request.point_count == 0) ||
      (request.protected_vertices != nullptr && request.point_count == 0) ||
      (request.source_face_tolerances != nullptr && request.source_face_count == 0) ||
      (request.source_segment_tolerances != nullptr && request.source_segment_count == 0) ||
      (request.source_face_planes != nullptr && request.source_face_count == 0) ||
      (request.source_face_plane_tolerances != nullptr && request.source_face_planes == nullptr)) {
    return false;
  }
  if (request.protected_vertices != nullptr) {
    for (int64_t row = 0; row < request.point_count; ++row) {
      if (request.protected_vertices[row] != 0 && request.protected_vertices[row] != 1) return false;
    }
  }
  if (request.source_point_count == 0 &&
      (request.source_face_count != 0 || request.source_segment_count != 0)) {
    return false;
  }
  const bool witnesses = request.witness_strata != nullptr ||
                         request.witness_entities != nullptr ||
                         request.witness_parameters != nullptr;
  if (witnesses && (request.witness_strata == nullptr || request.witness_entities == nullptr ||
                    request.witness_parameters == nullptr)) {
    return false;
  }
  if (witnesses && !addressable(request.point_count, 2, sizeof(double))) return false;
  if (witnesses && request.point_count > 0 && request.witness_deviations != nullptr &&
      !addressable(request.point_count, 1, sizeof(double))) return false;
  if (!witnesses && request.witness_deviations != nullptr) return false;
  if ((request.boundary_policy != PHX_MC_TET_MESH_BOUNDARY_FIXED &&
       request.boundary_policy != PHX_MC_TET_MESH_BOUNDARY_CONFORMING) ||
      (request.tetrahedron_count > 0 && request.point_count == 0)) {
    return false;
  }
  return true;
}

bool valid_edit_request(const phx_mc_tet_mesh_edit_request& request) noexcept {
  const auto valid_count = [](int64_t count, int64_t width, std::size_t bytes,
                              const void* pointer) {
    return count >= 0 && addressable(count, width, bytes) && required_bank(pointer, count);
  };
  if (!valid_identity_fields(request.identity_bound, request.source_id,
                             request.source_id_length, request.source_revision,
                             request.source_revision_length, request.source_map_id,
                             request.source_map_id_length, request.frame_id,
                             request.frame_id_length)) return false;
  if (request.periodic != nullptr && !valid_periodic_edit_view(*request.periodic)) return false;
  if (request.kind < PHX_MC_TET_EDIT_SPLIT || request.kind > PHX_MC_TET_EDIT_RETRIANGULATE ||
      !valid_count(request.collapse_pair_count, 1, sizeof(uint64_t), request.collapse_removed_ids) ||
      !valid_count(request.collapse_pair_count, 1, sizeof(uint64_t), request.collapse_survivor_ids) ||
      !valid_count(request.split_edge_count, 2, sizeof(uint64_t), request.split_edges) ||
      !valid_count(request.removed_vertex_count, 1, sizeof(uint64_t), request.removed_vertex_ids) ||
      !valid_count(request.removed_cell_count, 1, sizeof(uint64_t), request.removed_cell_ids) ||
      !valid_count(request.changed_vertex_count, 1, sizeof(uint64_t), request.changed_vertex_ids) ||
      !addressable(request.changed_vertex_count, 3, sizeof(double)) ||
      !required_bank(request.changed_vertex_points, request.changed_vertex_count) ||
      !valid_count(request.replacement_cell_count, 4, sizeof(uint64_t), request.replacement_cell_vertices) ||
      !valid_count(request.replacement_cell_count, 1, sizeof(uint64_t), request.replacement_cell_ids) ||
      !valid_count(request.removed_facet_count, 1, sizeof(uint64_t), request.removed_facet_ids) ||
      !valid_count(request.replacement_facet_count, 1, sizeof(uint64_t), request.replacement_facet_ids) ||
      !valid_count(request.replacement_facet_count, 3, sizeof(uint64_t), request.replacement_facet_vertices) ||
      !valid_count(request.replacement_facet_count, 1, sizeof(int32_t), request.replacement_facet_sources) ||
      !valid_count(request.removed_segment_count, 1, sizeof(uint64_t), request.removed_segment_ids) ||
      !valid_count(request.replacement_segment_count, 1, sizeof(uint64_t), request.replacement_segment_ids) ||
      !valid_count(request.replacement_segment_count, 2, sizeof(uint64_t), request.replacement_segment_vertices) ||
      !valid_count(request.replacement_segment_count, 1, sizeof(int32_t), request.replacement_segment_sources)) {
    return false;
  }
  if (request.replacement_cell_facet_sources != nullptr &&
      !addressable(request.replacement_cell_count, 4, sizeof(int32_t))) return false;
  if (request.replacement_facet_orientations != nullptr &&
      !addressable(request.replacement_facet_count, 1, sizeof(int8_t))) return false;
  if (request.replacement_cell_count > 0 && request.replacement_regions != nullptr &&
      !addressable(request.replacement_cell_count, 1, sizeof(int32_t))) return false;
  if (request.replacement_facet_count > 0 && request.replacement_protected_facets != nullptr &&
      !addressable(request.replacement_facet_count, 1, sizeof(int8_t))) return false;
  if (request.replacement_segment_count > 0 && request.replacement_protected_segments != nullptr &&
      !addressable(request.replacement_segment_count, 1, sizeof(int8_t))) return false;
  if (request.changed_vertex_count > 0) {
    const bool has_witness = request.changed_witness_strata != nullptr ||
                             request.changed_witness_entities != nullptr ||
                             request.changed_witness_parameters != nullptr;
    if (has_witness && (request.changed_witness_strata == nullptr ||
                        request.changed_witness_entities == nullptr ||
                        request.changed_witness_parameters == nullptr)) return false;
    if (has_witness && !addressable(request.changed_vertex_count, 2, sizeof(double))) return false;
    if (!has_witness && request.changed_witness_deviations != nullptr) return false;
    if (request.changed_witness_deviations != nullptr &&
        !addressable(request.changed_vertex_count, 1, sizeof(double))) return false;
    if (request.changed_protection_radii != nullptr &&
        !addressable(request.changed_vertex_count, 1, sizeof(double))) return false;
    if (request.changed_protected_vertices != nullptr &&
        !addressable(request.changed_vertex_count, 1, sizeof(int8_t))) return false;
  }
  const bool edge_witness_fields = request.changed_edge_witness_present != nullptr ||
                                   request.changed_edge_witness_endpoints != nullptr ||
                                   request.changed_edge_parameters != nullptr;
  if (edge_witness_fields &&
      (request.changed_vertex_count == 0 || request.changed_edge_witness_present == nullptr ||
       request.changed_edge_witness_endpoints == nullptr ||
       request.changed_edge_parameters == nullptr ||
       !addressable(request.changed_vertex_count, 1, sizeof(int8_t)) ||
       !addressable(request.changed_vertex_count, 2, sizeof(uint64_t)) ||
       !addressable(request.changed_vertex_count, 1, sizeof(double)))) return false;
  if (edge_witness_fields) {
    for (int64_t row = 0; row < request.changed_vertex_count; ++row) {
      const int8_t present = request.changed_edge_witness_present[row];
      const uint64_t first = request.changed_edge_witness_endpoints[2 * row];
      const uint64_t second = request.changed_edge_witness_endpoints[2 * row + 1];
      const double parameter = request.changed_edge_parameters[row];
      if ((present != 0 && present != 1) ||
          (present == 0 && (first != 0 || second != 0 || parameter != 0.0)) ||
          (present == 1 && (first == second || !std::isfinite(parameter) ||
                            !(parameter > 0.0 && parameter < 1.0)))) return false;
    }
  }
  const bool vertex_ancestry_fields =
      request.changed_vertex_ancestry_offsets != nullptr ||
      request.changed_vertex_ancestor_ids != nullptr ||
      request.changed_vertex_ancestor_winding != nullptr ||
      request.changed_vertex_ancestry_weights != nullptr ||
      request.changed_vertex_ancestry_count != 0;
  if (request.changed_vertex_ancestry_count < 0 ||
      !addressable(request.changed_vertex_ancestry_count, 1, sizeof(uint64_t)) ||
      !addressable(request.changed_vertex_ancestry_count, 3, sizeof(int64_t)) ||
      !addressable(request.changed_vertex_ancestry_count, 1, sizeof(double))) return false;
  if (vertex_ancestry_fields) {
    if (request.changed_vertex_count == 0 ||
        request.changed_vertex_count == std::numeric_limits<int64_t>::max() ||
        request.changed_vertex_ancestry_offsets == nullptr ||
        !addressable(request.changed_vertex_count + 1, 1, sizeof(int64_t)) ||
        !required_bank(request.changed_vertex_ancestor_ids,
                       request.changed_vertex_ancestry_count) ||
        !required_bank(request.changed_vertex_ancestor_winding,
                       request.changed_vertex_ancestry_count) ||
        !required_bank(request.changed_vertex_ancestry_weights,
                       request.changed_vertex_ancestry_count) ||
        (request.changed_vertex_ancestry_count == 0 &&
         (request.changed_vertex_ancestor_ids != nullptr ||
          request.changed_vertex_ancestor_winding != nullptr ||
          request.changed_vertex_ancestry_weights != nullptr)) ||
        request.changed_vertex_ancestry_offsets[0] != 0 ||
        request.changed_vertex_ancestry_offsets[request.changed_vertex_count] !=
            request.changed_vertex_ancestry_count) return false;
    for (int64_t row = 0; row < request.changed_vertex_count; ++row) {
      const int64_t begin = request.changed_vertex_ancestry_offsets[row];
      const int64_t end = request.changed_vertex_ancestry_offsets[row + 1];
      if (begin < 0 || begin > end || end > request.changed_vertex_ancestry_count) return false;
    }
  }
  const bool cell_ancestry_fields =
      request.replacement_cell_ancestry_offsets != nullptr ||
      request.replacement_cell_ancestor_ids != nullptr ||
      request.replacement_cell_ancestry_count != 0;
  if (request.replacement_cell_ancestry_count < 0 ||
      !addressable(request.replacement_cell_ancestry_count, 1, sizeof(uint64_t))) return false;
  if (cell_ancestry_fields) {
    if (request.replacement_cell_count == 0 ||
        request.replacement_cell_count == std::numeric_limits<int64_t>::max() ||
        request.replacement_cell_ancestry_offsets == nullptr ||
        !addressable(request.replacement_cell_count + 1, 1, sizeof(int64_t)) ||
        !required_bank(request.replacement_cell_ancestor_ids,
                       request.replacement_cell_ancestry_count) ||
        (request.replacement_cell_ancestry_count == 0 &&
         request.replacement_cell_ancestor_ids != nullptr) ||
        request.replacement_cell_ancestry_offsets[0] != 0 ||
        request.replacement_cell_ancestry_offsets[request.replacement_cell_count] !=
            request.replacement_cell_ancestry_count) return false;
    for (int64_t row = 0; row < request.replacement_cell_count; ++row) {
      const int64_t begin = request.replacement_cell_ancestry_offsets[row];
      const int64_t end = request.replacement_cell_ancestry_offsets[row + 1];
      if (begin < 0 || begin > end || end > request.replacement_cell_ancestry_count) return false;
    }
  }
  if (request.kind == PHX_MC_TET_EDIT_COLLAPSE &&
      request.collapse_pair_count != request.removed_vertex_count) return false;
  if (request.kind != PHX_MC_TET_EDIT_COLLAPSE && request.collapse_pair_count != 0) return false;
  return true;
}

int32_t validate_snapshot_request(const phx_mc_tet_mesh_snapshot_request& request,
                                  const TetMeshSnapshot& snapshot) noexcept {
  std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> counts{};
  if (!tet_snapshot_counts(snapshot, counts)) return PHX_MC_INDEX_RANGE_ERROR;
  const int64_t capacities[PHX_MC_TET_MESH_SNAPSHOT_SIZES] = {
      request.vertex_capacity, request.cell_capacity, request.facet_capacity,
      request.segment_capacity, request.source_point_capacity, request.source_face_capacity,
      request.source_segment_capacity, request.periodic_orbit_capacity,
      request.periodic_member_capacity, request.periodic_quotient_cell_capacity,
      request.periodic_quotient_lifted_capacity, request.periodic_entity_map_capacity,
      request.periodic_entity_lifted_capacity, request.periodic_entity_quotient_capacity,
      request.periodic_entity_winding_capacity, request.periodic_entity_corner_capacity,
      request.periodic_entity_representative_capacity, request.vertex_ancestry_capacity,
      request.cell_ancestor_capacity};
  const int64_t widths[PHX_MC_TET_MESH_SNAPSHOT_SIZES] = {
      3, 4, 3, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const std::size_t item_bytes[PHX_MC_TET_MESH_SNAPSHOT_SIZES] = {
      sizeof(double), sizeof(std::uint64_t), sizeof(std::uint64_t), sizeof(std::uint64_t),
      sizeof(double), sizeof(std::int32_t), sizeof(std::int32_t),
      sizeof(phx_mc_tet_periodic_orbit), sizeof(phx_mc_tet_periodic_vertex_member),
      sizeof(phx_mc_tet_periodic_quotient_cell), sizeof(std::uint64_t),
      sizeof(phx_mc_tet_periodic_entity_orbit_map), sizeof(std::uint64_t), sizeof(std::uint64_t),
      sizeof(std::int64_t), sizeof(std::int32_t), sizeof(std::int32_t),
      sizeof(std::uint64_t), sizeof(std::uint64_t)};
  for (int bank = 0; bank < PHX_MC_TET_MESH_SNAPSHOT_SIZES; ++bank) {
    if (capacities[bank] < 0 ||
        !addressable(capacities[bank], widths[bank], item_bytes[bank])) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (capacities[bank] < counts[bank]) return PHX_MC_CAPACITY_EXCEEDED;
  }
  if (counts[0] == std::numeric_limits<int64_t>::max() ||
      counts[1] == std::numeric_limits<int64_t>::max()) return PHX_MC_INDEX_RANGE_ERROR;
  if (!addressable(request.vertex_capacity, 1, sizeof(int8_t)) ||
      !addressable(request.vertex_capacity, 2, sizeof(std::uint64_t)) ||
      !addressable(request.vertex_capacity, 1, sizeof(double)) ||
      !addressable(counts[0] + 1, 1, sizeof(int64_t)) ||
      !addressable(counts[1] + 1, 1, sizeof(int64_t)) ||
      !addressable(request.vertex_ancestry_capacity, 3, sizeof(int64_t)) ||
      !addressable(request.vertex_ancestry_capacity, 1, sizeof(double)) ||
      !addressable(request.cell_capacity, 1, sizeof(int32_t)) ||
      !addressable(request.periodic_entity_lifted_capacity, 1, sizeof(int32_t)) ||
      !addressable(request.periodic_entity_quotient_capacity, 1, sizeof(int32_t)) ||
      !addressable(request.periodic_entity_representative_capacity, 1, sizeof(int32_t)) ||
      !addressable(request.source_face_capacity, 9, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto required = [](const void* pointer, int64_t count) {
    return count == 0 || pointer != nullptr;
  };
  const auto check = [&](const void* pointer, std::size_t bank) {
    return required(pointer, counts[bank]);
  };
  if (!check(request.vertex_ids, 0) || !check(request.points, 0) ||
      !check(request.witness_strata, 0) || !check(request.witness_entities, 0) ||
      !check(request.witness_parameters, 0) || !check(request.witness_deviations, 0) ||
      !check(request.vertex_dimensions, 0) || !check(request.protection_radii, 0) ||
      !check(request.protected_vertices, 0) ||
      !check(request.vertex_edge_witness_present, 0) ||
      !check(request.vertex_edge_witness_endpoints, 0) ||
      !check(request.vertex_edge_parameters, 0) ||
      !check(request.cell_ids, 1) || !check(request.cell_vertices, 1) ||
      !check(request.cell_regions, 1) || !check(request.facet_ids, 2) ||
      !check(request.facet_vertices, 2) || !check(request.facet_sources, 2) ||
      !check(request.facet_orientations, 2) || !check(request.protected_facets, 2) ||
      !check(request.segment_ids, 3) || !check(request.segment_vertices, 3) ||
      !check(request.segment_sources, 3) || !check(request.protected_segments, 3) ||
      !check(request.source_points, 4) || !check(request.source_faces, 5) ||
      !check(request.source_face_groups, 5) || !check(request.source_face_ids, 5) ||
      !check(request.source_face_tolerances, 5) || !check(request.source_segments, 6) ||
      !check(request.source_segment_groups, 6) || !check(request.source_segment_ids, 6) ||
      !check(request.source_segment_tolerances, 6) ||
      !check(request.source_face_planes, 5) || !check(request.source_face_plane_tolerances, 5) ||
      !check(request.periodic_orbits, 7) || !check(request.periodic_members, 8) ||
      !check(request.periodic_quotient_cells, 9) ||
      !check(request.periodic_quotient_lifted_cell_ids, 10) ||
      !check(request.periodic_entity_maps, 11) ||
      !check(request.periodic_entity_lifted_ids, 12) ||
      !check(request.periodic_entity_lifted_to_quotient, 12) ||
      !check(request.periodic_entity_orientation_signs, 12) ||
      !check(request.periodic_entity_quotient_ids, 13) ||
      !check(request.periodic_entity_winding_shifts, 14) ||
      !check(request.periodic_entity_corner_permutations, 15) ||
      !check(request.periodic_entity_representative_rows, 16) ||
      !required(request.vertex_ancestry_offsets, counts[0] + 1) ||
      !check(request.vertex_ancestor_ids, 17) ||
      !check(request.vertex_ancestor_winding, 17) ||
      !check(request.vertex_ancestry_weights, 17) ||
      !required(request.cell_ancestry_offsets, counts[1] + 1) ||
      !check(request.cell_ancestor_ids, 18) ||
      !check(request.cell_ancestry_edit_kinds, 1) ||
      request.generation == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_tet_mesh_create(const phx_mc_tet_mesh_create_request* request) {
  if (request != nullptr && request->result != nullptr) *request->result = nullptr;
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return guarded([&]() -> int32_t {
    if (!valid_create_request(*request)) return PHX_MC_INVALID_ARGUMENT;
    const auto policy = static_cast<phx::mc::BoundaryPolicy>(request->boundary_policy);
    auto handle = std::make_unique<phx_mc_tet_mesh>(policy);
    handle->state.bind_source_identity(
        identity_string(request->source_id, request->source_id_length),
        identity_string(request->source_revision, request->source_revision_length),
        identity_string(request->source_map_id, request->source_map_id_length),
        identity_string(request->frame_id, request->frame_id_length),
        request->identity_bound != 0);

    std::vector<SourceWitness> witnesses;
    if (request->witness_strata != nullptr) {
      witnesses.resize(static_cast<std::size_t>(request->point_count));
      for (std::size_t row = 0; row < witnesses.size(); ++row) {
        SourceWitness& witness = witnesses[row];
        witness.stratum = static_cast<SourceStratum>(request->witness_strata[row]);
        witness.entity = request->witness_entities[row];
        witness.parameters[0] = request->witness_parameters[2 * row];
        witness.parameters[1] = request->witness_parameters[2 * row + 1];
        witness.deviation = request->witness_deviations == nullptr ? 0.0
                                                                  : request->witness_deviations[row];
        if (witness.stratum != SourceStratum::kNone &&
            witness.stratum != SourceStratum::kSegment &&
            witness.stratum != SourceStratum::kFacet &&
            witness.stratum != SourceStratum::kPlane) return PHX_MC_INVALID_INPUT;
      }
    }

    const bool explicit_source = request->source_point_count != 0;
    const std::span<const double> source_points = explicit_source
        ? input_span(request->source_points, request->source_point_count * 3)
        : input_span(request->points, request->point_count * 3);
    const std::span<const int32_t> source_faces = explicit_source
        ? input_span(request->source_faces, request->source_face_count * 3)
        : input_span(request->faces, request->face_count * 3);
    const std::span<const int32_t> source_face_groups = explicit_source
        ? input_span(request->source_face_groups, request->source_face_count)
        : input_span(request->face_sources, request->face_count);
    const std::span<const int64_t> source_face_ids =
        explicit_source && request->source_face_ids != nullptr
            ? input_span(request->source_face_ids, request->source_face_count)
            : std::span<const int64_t>{};
    const std::span<const double> source_face_tolerances = explicit_source &&
        request->source_face_tolerances != nullptr
        ? input_span(request->source_face_tolerances, request->source_face_count)
        : std::span<const double>{};
    const std::span<const int32_t> source_segments = explicit_source
        ? input_span(request->source_segments, request->source_segment_count * 2)
        : input_span(request->segments, request->segment_count * 2);
    const std::span<const int32_t> source_segment_groups = explicit_source
        ? input_span(request->source_segment_groups, request->source_segment_count)
        : input_span(request->segment_sources, request->segment_count);
    const std::span<const int64_t> source_segment_ids =
        explicit_source && request->source_segment_ids != nullptr
            ? input_span(request->source_segment_ids, request->source_segment_count)
            : std::span<const int64_t>{};
    const std::span<const double> source_segment_tolerances = explicit_source &&
        request->source_segment_tolerances != nullptr
        ? input_span(request->source_segment_tolerances, request->source_segment_count)
        : std::span<const double>{};
    const std::span<const double> source_face_planes =
        explicit_source && request->source_face_planes != nullptr
            ? input_span(request->source_face_planes, request->source_face_count * 9)
            : std::span<const double>{};
    const std::span<const double> source_face_plane_tolerances =
        explicit_source && request->source_face_plane_tolerances != nullptr
            ? input_span(request->source_face_plane_tolerances, request->source_face_count)
            : std::span<const double>{};

    if (!witnesses.empty()) {
      for (std::size_t row = 0; row < witnesses.size(); ++row) {
        SourceWitness& witness = witnesses[row];
        if (request->witness_deviations != nullptr || witness.stratum == SourceStratum::kNone) continue;
        const std::size_t entity = witness.entity < 0 ? source_face_groups.size()
                                                       : static_cast<std::size_t>(witness.entity);
        if (witness.stratum == SourceStratum::kSegment && entity < source_segments.size() / 2) {
          const double* corners[3] = {
              source_points.data() + 3 * static_cast<std::size_t>(source_segments[2 * entity]),
              source_points.data() + 3 * static_cast<std::size_t>(source_segments[2 * entity + 1]),
              nullptr};
          witness.deviation = phx::mc::source_deviation(
              corners, witness.stratum, witness.parameters,
              request->points + 3 * row);
        } else if (witness.stratum == SourceStratum::kFacet && entity < source_faces.size() / 3) {
          const double* corners[3] = {
              source_points.data() + 3 * static_cast<std::size_t>(source_faces[3 * entity]),
              source_points.data() + 3 * static_cast<std::size_t>(source_faces[3 * entity + 1]),
              source_points.data() + 3 * static_cast<std::size_t>(source_faces[3 * entity + 2])};
          witness.deviation = phx::mc::source_deviation(
              corners, witness.stratum, witness.parameters,
              request->points + 3 * row);
        } else if (witness.stratum == SourceStratum::kPlane && entity < source_faces.size() / 3 &&
                   !source_face_planes.empty()) {
          witness.deviation = phx::mc::declared_plane_deviation(
              source_face_planes.data() + 9 * entity, witness.parameters,
              request->points + 3 * row);
        }
      }
    }

    if (request->plane_refusal != nullptr) *request->plane_refusal = phx_mc_tet_plane_refusal{};
    const phx::mc::SourceComplexView source{
        source_points, source_faces, source_face_groups, source_face_ids,
        source_face_tolerances, source_segments, source_segment_groups,
        source_segment_ids, source_segment_tolerances, witnesses,
        source_face_planes, source_face_plane_tolerances};
    const int32_t status = handle->state.build(
        input_span(request->points, request->point_count * 3),
        input_span(request->tetrahedra, request->tetrahedron_count * 4),
        request->tetrahedron_regions == nullptr
            ? std::span<const int32_t>{}
            : input_span(request->tetrahedron_regions, request->tetrahedron_count),
        input_span(request->faces, request->face_count * 3),
        input_span(request->face_sources, request->face_count),
        request->face_orientations == nullptr
            ? std::span<const int8_t>{}
            : input_span(request->face_orientations, request->face_count),
        input_span(request->segments, request->segment_count * 2),
        input_span(request->segment_sources, request->segment_count),
        request->protection_radii == nullptr
            ? std::span<const double>{}
            : input_span(request->protection_radii, request->point_count),
        &source,
        request->protected_vertices == nullptr
            ? std::span<const int8_t>{}
            : input_span(request->protected_vertices, request->point_count));
    if (status != PHX_MC_OK) {
      const auto& refusal = handle->state.plane_refusal();
      if (request->plane_refusal != nullptr && refusal.has_value()) {
        phx_mc_tet_plane_refusal& out = *request->plane_refusal;
        out.refused = 1;
        out.face_row = refusal->record.face_row;
        out.vertex = refusal->vertex;
        out.parameters[0] = refusal->record.parameters[0];
        out.parameters[1] = refusal->record.parameters[1];
        out.deviation = refusal->record.deviation;
        out.tolerance = refusal->tolerance;
      }
      return status;
    }
    const int32_t id_status = handle->state.assign_initial_ids(
        request->vertex_ids == nullptr
            ? std::span<const uint64_t>{}
            : input_span(request->vertex_ids, request->point_count),
        request->cell_ids == nullptr
            ? std::span<const uint64_t>{}
            : input_span(request->cell_ids, request->tetrahedron_count),
        request->facet_ids == nullptr
            ? std::span<const uint64_t>{}
            : input_span(request->facet_ids, request->face_count),
        request->segment_ids == nullptr
            ? std::span<const uint64_t>{}
            : input_span(request->segment_ids, request->segment_count));
    if (id_status != PHX_MC_OK) return id_status;
    if (request->periodic_state != nullptr) {
      const phx_mc_tet_periodic_state_view& periodic = *request->periodic_state;
      const TetPeriodicIdentity identity = periodic_identity_from_view(periodic.identity_map);
      handle->state.bind_periodic_identity(identity);
      std::vector<TetPeriodicOrbit> orbits;
      std::vector<TetPeriodicQuotientCell> quotient_cells;
      std::vector<TetPeriodicEntityOrbitMap> entity_maps;
      periodic_state_from_view(periodic, orbits, quotient_cells, entity_maps);
      const TetEditStatus periodic_status = handle->state.set_periodic_state(
          orbits, quotient_cells, entity_maps, false);
      if (periodic_status == TetEditStatus::kAllocationFailure) {
        return PHX_MC_ALLOCATION_FAILURE;
      }
      if (periodic_status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
      if (periodic_status != TetEditStatus::kApplied) return PHX_MC_INVALID_INPUT;
    }
    *request->result = handle.release();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_snapshot_sizes(const phx_mc_tet_mesh* mesh, int64_t* sizes) {
  if (mesh == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return guarded([&]() -> int32_t {
    TetMeshSnapshot snapshot;
    const TetEditStatus status = mesh->state.snapshot(snapshot);
    if (status != TetEditStatus::kApplied) {
      return status == TetEditStatus::kAllocationFailure ? PHX_MC_ALLOCATION_FAILURE
                                                        : PHX_MC_INTERNAL_ERROR;
    }
    std::array<int64_t, PHX_MC_TET_MESH_SNAPSHOT_SIZES> snapshot_counts{};
    if (!tet_snapshot_counts(snapshot, snapshot_counts)) return PHX_MC_INDEX_RANGE_ERROR;
    std::copy(snapshot_counts.begin(), snapshot_counts.end(), sizes);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_identity_sizes(const phx_mc_tet_mesh* mesh,
                                       uint64_t* generation, int32_t* identity_bound,
                                       int64_t* source_id_length,
                                       int64_t* source_revision_length,
                                       int64_t* source_map_id_length,
                                       int64_t* frame_id_length) {
  if (mesh == nullptr || generation == nullptr || identity_bound == nullptr ||
      source_id_length == nullptr || source_revision_length == nullptr ||
      source_map_id_length == nullptr || frame_id_length == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const std::string* identity[] = {
      &mesh->state.source_id(), &mesh->state.source_revision(),
      &mesh->state.source_map_id(), &mesh->state.frame_id()};
  for (const std::string* value : identity) {
    if (value->size() > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
  }
  *generation = mesh->state.generation();
  *identity_bound = mesh->state.identity_bound() ? 1 : 0;
  *source_id_length = static_cast<int64_t>(identity[0]->size());
  *source_revision_length = static_cast<int64_t>(identity[1]->size());
  *source_map_id_length = static_cast<int64_t>(identity[2]->size());
  *frame_id_length = static_cast<int64_t>(identity[3]->size());
  return PHX_MC_OK;
}

int32_t phx_mc_tet_mesh_identity(const phx_mc_tet_mesh* mesh,
                                 int64_t source_id_capacity, char* source_id,
                                 int64_t source_revision_capacity, char* source_revision,
                                 int64_t source_map_id_capacity, char* source_map_id,
                                 int64_t frame_id_capacity, char* frame_id) {
  if (mesh == nullptr || source_id_capacity < 0 || source_revision_capacity < 0 ||
      source_map_id_capacity < 0 || frame_id_capacity < 0 ||
      !addressable(source_id_capacity, 1, sizeof(char)) ||
      !addressable(source_revision_capacity, 1, sizeof(char)) ||
      !addressable(source_map_id_capacity, 1, sizeof(char)) ||
      !addressable(frame_id_capacity, 1, sizeof(char))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const std::string* identity[] = {
      &mesh->state.source_id(), &mesh->state.source_revision(),
      &mesh->state.source_map_id(), &mesh->state.frame_id()};
  const int64_t capacities[] = {source_id_capacity, source_revision_capacity,
                                source_map_id_capacity, frame_id_capacity};
  char* buffers[] = {source_id, source_revision, source_map_id, frame_id};
  int64_t lengths[4];
  for (int field = 0; field < 4; ++field) {
    if (identity[field]->size() >
        static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    lengths[field] = static_cast<int64_t>(identity[field]->size());
    if (capacities[field] < lengths[field]) return PHX_MC_CAPACITY_EXCEEDED;
    if (!required_bank(buffers[field], lengths[field])) return PHX_MC_INVALID_ARGUMENT;
  }
  for (int field = 0; field < 4; ++field) {
    if (lengths[field] != 0) {
      std::copy(identity[field]->begin(), identity[field]->end(), buffers[field]);
    }
  }
  return PHX_MC_OK;
}

int32_t phx_mc_tet_mesh_periodic_identity_sizes(
    const phx_mc_tet_mesh* mesh, int32_t* identity_bound,
    int64_t* source_mesh_id_length, int64_t* source_topology_id_length,
    int64_t* lifted_topology_id_length, int64_t* periodic_topology_id_length,
    int64_t* cell_entity_set_id_length, int64_t* edge_entity_set_id_length) {
  if (mesh == nullptr || identity_bound == nullptr || source_mesh_id_length == nullptr ||
      source_topology_id_length == nullptr || lifted_topology_id_length == nullptr ||
      periodic_topology_id_length == nullptr || cell_entity_set_id_length == nullptr ||
      edge_entity_set_id_length == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const TetPeriodicIdentity& identity = mesh->state.periodic_identity();
  for (const std::string& value : identity.ids) {
    if (value.size() > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
  }
  *identity_bound = identity.bound ? 1 : 0;
  int64_t* lengths[] = {source_mesh_id_length, source_topology_id_length,
                        lifted_topology_id_length, periodic_topology_id_length,
                        cell_entity_set_id_length, edge_entity_set_id_length};
  for (std::size_t field = 0; field < identity.ids.size(); ++field) {
    *lengths[field] = static_cast<int64_t>(identity.ids[field].size());
  }
  return PHX_MC_OK;
}

int32_t phx_mc_tet_mesh_periodic_identity(
    const phx_mc_tet_mesh* mesh,
    int64_t source_mesh_id_capacity, char* source_mesh_id,
    int64_t source_topology_id_capacity, char* source_topology_id,
    int64_t lifted_topology_id_capacity, char* lifted_topology_id,
    int64_t periodic_topology_id_capacity, char* periodic_topology_id,
    int64_t cell_entity_set_id_capacity, char* cell_entity_set_id,
    int64_t edge_entity_set_id_capacity, char* edge_entity_set_id) {
  if (mesh == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const int64_t capacities[] = {
      source_mesh_id_capacity, source_topology_id_capacity, lifted_topology_id_capacity,
      periodic_topology_id_capacity, cell_entity_set_id_capacity, edge_entity_set_id_capacity};
  char* buffers[] = {source_mesh_id, source_topology_id, lifted_topology_id,
                     periodic_topology_id, cell_entity_set_id, edge_entity_set_id};
  for (int64_t capacity : capacities) {
    if (capacity < 0 || !addressable(capacity, 1, sizeof(char))) return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& identity = mesh->state.periodic_identity().ids;
  int64_t lengths[6];
  for (std::size_t field = 0; field < identity.size(); ++field) {
    if (identity[field].size() > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    lengths[field] = static_cast<int64_t>(identity[field].size());
    if (capacities[field] < lengths[field]) return PHX_MC_CAPACITY_EXCEEDED;
    if (!required_bank(buffers[field], lengths[field])) return PHX_MC_INVALID_ARGUMENT;
  }
  for (std::size_t field = 0; field < identity.size(); ++field) {
    if (lengths[field] != 0) {
      std::copy(identity[field].begin(), identity[field].end(), buffers[field]);
    }
  }
  return PHX_MC_OK;
}

int32_t phx_mc_tet_mesh_snapshot(const phx_mc_tet_mesh* mesh,
                                 const phx_mc_tet_mesh_snapshot_request* request) {
  if (mesh == nullptr || request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return guarded([&]() -> int32_t {
    TetMeshSnapshot snapshot;
    const TetEditStatus state_status = mesh->state.snapshot(snapshot);
    if (state_status != TetEditStatus::kApplied) {
      return state_status == TetEditStatus::kAllocationFailure ? PHX_MC_ALLOCATION_FAILURE
                                                              : PHX_MC_INTERNAL_ERROR;
    }
    const int32_t valid = validate_snapshot_request(*request, snapshot);
    if (valid != PHX_MC_OK) return valid;
    std::size_t vertex_ancestry_offset = 0;
    request->vertex_ancestry_offsets[0] = 0;
    for (std::size_t row = 0; row < snapshot.vertices.size(); ++row) {
      const TetSnapshotVertex& vertex = snapshot.vertices[row];
      request->vertex_ids[row] = vertex.id;
      std::copy(vertex.carrier.begin(), vertex.carrier.end(), request->points + 3 * row);
      request->witness_strata[row] = static_cast<int8_t>(vertex.witness.stratum);
      request->witness_entities[row] = vertex.witness.entity;
      request->witness_parameters[2 * row] = vertex.witness.parameters[0];
      request->witness_parameters[2 * row + 1] = vertex.witness.parameters[1];
      request->witness_deviations[row] = vertex.witness.deviation;
      request->vertex_dimensions[row] = vertex.stratum_dimension;
      request->protection_radii[row] = vertex.protection_radius;
      request->protected_vertices[row] = vertex.protected_vertex ? 1 : 0;
      request->vertex_edge_witness_present[row] = vertex.has_edge_parameter ? 1 : 0;
      std::copy(vertex.edge_endpoints.begin(), vertex.edge_endpoints.end(),
                request->vertex_edge_witness_endpoints + 2 * row);
      request->vertex_edge_parameters[row] = vertex.edge_parameter;
      for (const phx::mc::TetVertexAncestry& ancestry : vertex.ancestry) {
        request->vertex_ancestor_ids[vertex_ancestry_offset] = ancestry.source_vertex_id;
        std::copy(ancestry.source_winding.begin(), ancestry.source_winding.end(),
                  request->vertex_ancestor_winding + 3 * vertex_ancestry_offset);
        request->vertex_ancestry_weights[vertex_ancestry_offset] = ancestry.weight;
        ++vertex_ancestry_offset;
      }
      request->vertex_ancestry_offsets[row + 1] =
          static_cast<int64_t>(vertex_ancestry_offset);
    }
    std::size_t cell_ancestor_offset = 0;
    request->cell_ancestry_offsets[0] = 0;
    for (std::size_t row = 0; row < snapshot.cells.size(); ++row) {
      const TetSnapshotCell& cell = snapshot.cells[row];
      request->cell_ids[row] = cell.id;
      std::copy(cell.vertices.begin(), cell.vertices.end(), request->cell_vertices + 4 * row);
      request->cell_regions[row] = cell.region;
      if (!cell.ancestor_cell_ids.empty()) {
        copy_bank(cell.ancestor_cell_ids,
                  request->cell_ancestor_ids + cell_ancestor_offset);
      }
      cell_ancestor_offset += cell.ancestor_cell_ids.size();
      request->cell_ancestry_edit_kinds[row] = cell.edit_kind;
      request->cell_ancestry_offsets[row + 1] =
          static_cast<int64_t>(cell_ancestor_offset);
    }
    for (std::size_t row = 0; row < snapshot.constrained_facets.size(); ++row) {
      const TetSnapshotFacet& facet = snapshot.constrained_facets[row];
      request->facet_ids[row] = facet.id;
      std::copy(facet.vertices.begin(), facet.vertices.end(), request->facet_vertices + 3 * row);
      request->facet_sources[row] = facet.source;
      request->facet_orientations[row] = facet.orientation;
      request->protected_facets[row] = facet.protected_facet ? 1 : 0;
    }
    for (std::size_t row = 0; row < snapshot.constrained_segments.size(); ++row) {
      const TetSnapshotSegment& segment = snapshot.constrained_segments[row];
      request->segment_ids[row] = segment.id;
      std::copy(segment.vertices.begin(), segment.vertices.end(), request->segment_vertices + 2 * row);
      request->segment_sources[row] = segment.source;
      request->protected_segments[row] = segment.protected_segment ? 1 : 0;
    }
    copy_bank(snapshot.source_points, request->source_points);
    copy_bank(snapshot.source_faces, request->source_faces);
    copy_bank(snapshot.source_face_groups, request->source_face_groups);
    copy_bank(snapshot.source_face_ids, request->source_face_ids);
    copy_bank(snapshot.source_face_tolerances, request->source_face_tolerances);
    copy_bank(snapshot.source_segments, request->source_segments);
    copy_bank(snapshot.source_segment_groups, request->source_segment_groups);
    copy_bank(snapshot.source_segment_ids, request->source_segment_ids);
    copy_bank(snapshot.source_segment_tolerances, request->source_segment_tolerances);
    copy_bank(snapshot.source_face_planes, request->source_face_planes);
    copy_bank(snapshot.source_face_plane_tolerances, request->source_face_plane_tolerances);
    std::size_t member_offset = 0;
    for (std::size_t row = 0; row < snapshot.periodic_orbits.size(); ++row) {
      const TetPeriodicOrbit& orbit = snapshot.periodic_orbits[row];
      phx_mc_tet_periodic_orbit& output = request->periodic_orbits[row];
      output.id = orbit.id;
      output.representative_vertex_id = orbit.representative_vertex_id;
      output.member_offset = static_cast<int64_t>(member_offset);
      output.member_count = static_cast<int64_t>(orbit.members.size());
      for (const TetPeriodicVertexMember& member : orbit.members) {
        phx_mc_tet_periodic_vertex_member& target = request->periodic_members[member_offset++];
        target.vertex_id = member.vertex_id;
        std::copy(member.winding.begin(), member.winding.end(), target.winding);
        std::copy(member.representative_to_member.begin(),
                  member.representative_to_member.end(), target.representative_to_member);
      }
    }
    std::size_t quotient_lifted_offset = 0;
    for (std::size_t row = 0; row < snapshot.quotient_cells.size(); ++row) {
      const TetPeriodicQuotientCell& cell = snapshot.quotient_cells[row];
      phx_mc_tet_periodic_quotient_cell& target = request->periodic_quotient_cells[row];
      target.orbit_id = cell.orbit_id;
      target.id = cell.id;
      std::copy(cell.representative_vertices.begin(), cell.representative_vertices.end(),
                target.representative_vertices);
      for (int corner = 0; corner < 4; ++corner) {
        std::copy(cell.winding[corner].begin(), cell.winding[corner].end(),
                  target.winding + 3 * corner);
      }
      target.region = cell.region;
      target.lifted_cell_offset = static_cast<int64_t>(quotient_lifted_offset);
      target.lifted_cell_count = static_cast<int64_t>(cell.lifted_cell_ids.size());
      if (!cell.lifted_cell_ids.empty()) {
        std::copy(cell.lifted_cell_ids.begin(), cell.lifted_cell_ids.end(),
                  request->periodic_quotient_lifted_cell_ids + quotient_lifted_offset);
      }
      quotient_lifted_offset += cell.lifted_cell_ids.size();
    }
    std::size_t lifted_offset = 0;
    std::size_t quotient_offset = 0;
    std::size_t winding_offset = 0;
    std::size_t corner_offset = 0;
    std::size_t representative_offset = 0;
    for (std::size_t row = 0; row < snapshot.periodic_entity_maps.size(); ++row) {
      const TetPeriodicEntityOrbitMap& map = snapshot.periodic_entity_maps[row];
      phx_mc_tet_periodic_entity_orbit_map& target = request->periodic_entity_maps[row];
      target.dimension = map.dimension;
      target.arity = map.arity;
      target.rank = map.rank;
      target.lifted_offset = static_cast<int64_t>(lifted_offset);
      target.lifted_count = static_cast<int64_t>(map.lifted_entity_ids.size());
      target.quotient_offset = static_cast<int64_t>(quotient_offset);
      target.quotient_count = static_cast<int64_t>(map.quotient_entity_ids.size());
      target.winding_offset = static_cast<int64_t>(winding_offset);
      target.corner_permutation_offset = static_cast<int64_t>(corner_offset);
      target.representative_offset = static_cast<int64_t>(representative_offset);
      if (!map.lifted_entity_ids.empty()) {
        copy_bank(map.lifted_entity_ids, request->periodic_entity_lifted_ids + lifted_offset);
        copy_bank(map.lifted_to_quotient,
                  request->periodic_entity_lifted_to_quotient + lifted_offset);
        copy_bank(map.orientation_signs,
                  request->periodic_entity_orientation_signs + lifted_offset);
      }
      if (!map.quotient_entity_ids.empty()) {
        copy_bank(map.quotient_entity_ids,
                  request->periodic_entity_quotient_ids + quotient_offset);
        copy_bank(map.representative_lifted_rows,
                  request->periodic_entity_representative_rows + representative_offset);
      }
      if (!map.winding_shifts.empty()) {
        copy_bank(map.winding_shifts,
                  request->periodic_entity_winding_shifts + winding_offset);
      }
      if (!map.corner_permutations.empty()) {
        copy_bank(map.corner_permutations,
                  request->periodic_entity_corner_permutations + corner_offset);
      }
      lifted_offset += map.lifted_entity_ids.size();
      quotient_offset += map.quotient_entity_ids.size();
      winding_offset += map.winding_shifts.size();
      corner_offset += map.corner_permutations.size();
      representative_offset += map.representative_lifted_rows.size();
    }
    *request->generation = snapshot.generation;
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_apply_edit(phx_mc_tet_mesh* mesh,
                                   const phx_mc_tet_mesh_edit_request* request,
                                   int32_t* edit_status) {
  if (edit_status != nullptr) *edit_status = PHX_MC_TET_EDIT_REJECTED_TOPOLOGY;
  if (mesh == nullptr || request == nullptr || edit_status == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return guarded([&]() -> int32_t {
    if (!valid_edit_request(*request)) return PHX_MC_INVALID_ARGUMENT;
    TetEdit operation;
    switch (request->kind) {
      case PHX_MC_TET_EDIT_SPLIT: operation.kind = TetEditKind::kSplit; break;
      case PHX_MC_TET_EDIT_COLLAPSE: operation.kind = TetEditKind::kCollapse; break;
      case PHX_MC_TET_EDIT_FLIP23: operation.kind = TetEditKind::kFlip23; break;
      case PHX_MC_TET_EDIT_FLIP32: operation.kind = TetEditKind::kFlip32; break;
      case PHX_MC_TET_EDIT_RECONNECT: operation.kind = TetEditKind::kRingReconnect; break;
      case PHX_MC_TET_EDIT_RELOCATE: operation.kind = TetEditKind::kRelocate; break;
      case PHX_MC_TET_EDIT_REFINE_CELL: operation.kind = TetEditKind::kRefineCell; break;
      case PHX_MC_TET_EDIT_RETRIANGULATE: operation.kind = TetEditKind::kRetriangulate; break;
      default: return PHX_MC_INVALID_ARGUMENT;
    }
    operation.collapse_pairs.reserve(static_cast<std::size_t>(request->collapse_pair_count));
    for (int64_t row = 0; row < request->collapse_pair_count; ++row) {
      if (request->collapse_removed_ids[row] != request->removed_vertex_ids[row]) {
        return PHX_MC_INVALID_ARGUMENT;
      }
      operation.collapse_pairs.push_back(
          {request->collapse_removed_ids[row], request->collapse_survivor_ids[row]});
    }
    assign_bank(operation.removed_vertex_ids, request->removed_vertex_ids,
                request->removed_vertex_count);
    operation.split_edges.reserve(static_cast<std::size_t>(request->split_edge_count));
    for (int64_t row = 0; row < request->split_edge_count; ++row) {
      operation.split_edges.push_back({request->split_edges[2 * row], request->split_edges[2 * row + 1]});
    }
    assign_bank(operation.removed_cell_ids, request->removed_cell_ids,
                request->removed_cell_count);
    operation.changed_vertices.reserve(static_cast<std::size_t>(request->changed_vertex_count));
    const bool has_witness = request->changed_witness_strata != nullptr;
    for (int64_t row = 0; row < request->changed_vertex_count; ++row) {
      TetEditVertex vertex;
      vertex.id = request->changed_vertex_ids[row];
      std::copy_n(request->changed_vertex_points + 3 * row, 3, vertex.carrier.begin());
      if (has_witness) {
        vertex.witness.stratum = static_cast<SourceStratum>(request->changed_witness_strata[row]);
        vertex.witness.entity = request->changed_witness_entities[row];
        vertex.witness.parameters[0] = request->changed_witness_parameters[2 * row];
        vertex.witness.parameters[1] = request->changed_witness_parameters[2 * row + 1];
        vertex.witness.deviation = request->changed_witness_deviations == nullptr
                                       ? 0.0 : request->changed_witness_deviations[row];
        if (vertex.witness.stratum != SourceStratum::kNone &&
            vertex.witness.stratum != SourceStratum::kSegment &&
            vertex.witness.stratum != SourceStratum::kFacet &&
            vertex.witness.stratum != SourceStratum::kPlane) return PHX_MC_INVALID_INPUT;
      }
      vertex.protection_radius = request->changed_protection_radii == nullptr
                                     ? 0.0 : request->changed_protection_radii[row];
      vertex.protected_vertex = request->changed_protected_vertices != nullptr &&
                                request->changed_protected_vertices[row] != 0;
      vertex.edge_witness_supplied = request->changed_edge_witness_present != nullptr;
      if (request->changed_edge_witness_present != nullptr &&
          request->changed_edge_witness_present[row] != 0) {
        vertex.has_edge_parameter = true;
        std::copy_n(request->changed_edge_witness_endpoints + 2 * row, 2,
                    vertex.edge_endpoints.begin());
        vertex.edge_parameter = request->changed_edge_parameters[row];
      }
      if (request->changed_vertex_ancestry_offsets != nullptr) {
        const int64_t begin = request->changed_vertex_ancestry_offsets[row];
        const int64_t end = request->changed_vertex_ancestry_offsets[row + 1];
        vertex.ancestry.reserve(static_cast<std::size_t>(end - begin));
        for (int64_t entry = begin; entry < end; ++entry) {
          phx::mc::TetVertexAncestry ancestry;
          ancestry.source_vertex_id = request->changed_vertex_ancestor_ids[entry];
          std::copy_n(request->changed_vertex_ancestor_winding + 3 * entry, 3,
                      ancestry.source_winding.begin());
          ancestry.weight = request->changed_vertex_ancestry_weights[entry];
          vertex.ancestry.push_back(ancestry);
        }
      }
      operation.changed_vertices.push_back(vertex);
    }
    operation.replacement_cells.reserve(static_cast<std::size_t>(request->replacement_cell_count));
    for (int64_t row = 0; row < request->replacement_cell_count; ++row) {
      TetEditCell cell;
      cell.id = request->replacement_cell_ids[row];
      std::copy_n(request->replacement_cell_vertices + 4 * row, 4, cell.vertices.begin());
      if (request->replacement_cell_facet_sources != nullptr) {
        std::copy_n(request->replacement_cell_facet_sources + 4 * row, 4,
                    cell.facet_sources.begin());
      }
      cell.region = request->replacement_regions == nullptr ? phx::mc::kNoRegion
                                                            : request->replacement_regions[row];
      if (request->replacement_cell_ancestry_offsets != nullptr) {
        const int64_t begin = request->replacement_cell_ancestry_offsets[row];
        const int64_t end = request->replacement_cell_ancestry_offsets[row + 1];
        if (end > begin) {
          cell.ancestor_cell_ids.assign(request->replacement_cell_ancestor_ids + begin,
                                        request->replacement_cell_ancestor_ids + end);
        }
      }
      operation.replacement_cells.push_back(cell);
    }
    assign_bank(operation.removed_facet_ids, request->removed_facet_ids,
                request->removed_facet_count);
    operation.replacement_facets.reserve(static_cast<std::size_t>(request->replacement_facet_count));
    for (int64_t row = 0; row < request->replacement_facet_count; ++row) {
      TetEditFacet facet;
      facet.id = request->replacement_facet_ids[row];
      std::copy_n(request->replacement_facet_vertices + 3 * row, 3, facet.vertices.begin());
      facet.source = request->replacement_facet_sources[row];
      if (request->replacement_facet_orientations != nullptr) {
        facet.orientation = request->replacement_facet_orientations[row];
      }
      facet.protected_facet = request->replacement_protected_facets != nullptr &&
                              request->replacement_protected_facets[row] != 0;
      operation.replacement_facets.push_back(facet);
    }
    assign_bank(operation.removed_segment_ids, request->removed_segment_ids,
                request->removed_segment_count);
    operation.replacement_segments.reserve(static_cast<std::size_t>(request->replacement_segment_count));
    for (int64_t row = 0; row < request->replacement_segment_count; ++row) {
      TetEditSegment segment;
      segment.id = request->replacement_segment_ids[row];
      std::copy_n(request->replacement_segment_vertices + 2 * row, 2, segment.vertices.begin());
      segment.source = request->replacement_segment_sources[row];
      segment.protected_segment = request->replacement_protected_segments != nullptr &&
                                  request->replacement_protected_segments[row] != 0;
      operation.replacement_segments.push_back(segment);
    }
    std::optional<TetPeriodicIdentity> target_periodic_identity;
    if (request->periodic != nullptr) {
      const phx_mc_tet_periodic_edit_view& periodic = *request->periodic;
      operation.periodic_members.reserve(
          static_cast<std::size_t>(periodic.periodic_member_count));
      for (int64_t row = 0; row < periodic.periodic_member_count; ++row) {
        const phx_mc_tet_periodic_vertex_member& input = periodic.periodic_members[row];
        phx::mc::TetPeriodicEditMember member;
        member.vertex_id = input.vertex_id;
        std::copy_n(input.winding, 3, member.winding.begin());
        std::copy_n(input.representative_to_member, 12,
                    member.representative_to_member.begin());
        operation.periodic_members.push_back(member);
      }
      assign_bank(operation.removed_periodic_orbit_ids,
                  periodic.removed_periodic_orbit_ids,
                  periodic.removed_periodic_orbit_count);
      operation.replacement_periodic_orbits.reserve(
          static_cast<std::size_t>(periodic.replacement_periodic_orbit_count));
      for (int64_t row = 0; row < periodic.replacement_periodic_orbit_count; ++row) {
        const phx_mc_tet_periodic_orbit& input = periodic.replacement_periodic_orbits[row];
        TetPeriodicOrbit orbit;
        orbit.id = input.id;
        orbit.representative_vertex_id = input.representative_vertex_id;
        orbit.members.reserve(static_cast<std::size_t>(input.member_count));
        for (int64_t member_row = 0; member_row < input.member_count; ++member_row) {
          const phx_mc_tet_periodic_vertex_member& source_member =
              periodic.replacement_periodic_members[input.member_offset + member_row];
          TetPeriodicVertexMember member;
          member.vertex_id = source_member.vertex_id;
          std::copy_n(source_member.winding, 3, member.winding.begin());
          std::copy_n(source_member.representative_to_member, 12,
                      member.representative_to_member.begin());
          orbit.members.push_back(member);
        }
        operation.replacement_periodic_orbits.push_back(std::move(orbit));
      }
      assign_bank(operation.removed_quotient_cell_ids,
                  periodic.removed_quotient_cell_ids,
                  periodic.removed_quotient_cell_count);
      operation.replacement_quotient_cells.reserve(
          static_cast<std::size_t>(periodic.replacement_quotient_cell_count));
      for (int64_t row = 0; row < periodic.replacement_quotient_cell_count; ++row) {
        const phx_mc_tet_periodic_quotient_cell& input =
            periodic.replacement_quotient_cells[row];
        TetPeriodicQuotientCell cell;
        cell.orbit_id = input.orbit_id;
        cell.id = input.id;
        std::copy_n(input.representative_vertices, 4, cell.representative_vertices.begin());
        for (int corner = 0; corner < 4; ++corner) {
          std::copy_n(input.winding + 3 * corner, 3, cell.winding[corner].begin());
        }
        cell.region = input.region;
        if (input.lifted_cell_count > 0) {
          cell.lifted_cell_ids.assign(
              periodic.replacement_quotient_lifted_cell_ids + input.lifted_cell_offset,
              periodic.replacement_quotient_lifted_cell_ids + input.lifted_cell_offset +
                  input.lifted_cell_count);
        }
        operation.replacement_quotient_cells.push_back(std::move(cell));
      }
      if (periodic.target_identity_map != nullptr) {
        target_periodic_identity = periodic_identity_from_view(*periodic.target_identity_map);
        operation.replacement_periodic_entity_maps =
            periodic_entity_maps_from_view(*periodic.target_identity_map);
      }
    }
    TetEditBatch batch;
    batch.expected_generation = request->expected_generation;
    // The request names the target identity; staleness is the generation's.
    batch.periodic_identity = mesh->state.periodic_identity();
    batch.target_periodic_identity = std::move(target_periodic_identity);
    batch.identity_bound = request->identity_bound != 0;
    if (batch.identity_bound) {
      auto assign_identity = [](std::string& destination, const char* bytes,
                                int64_t length) {
        if (length > 0) destination.assign(bytes, static_cast<std::size_t>(length));
      };
      assign_identity(batch.source_id, request->source_id, request->source_id_length);
      assign_identity(batch.source_revision, request->source_revision,
                      request->source_revision_length);
      assign_identity(batch.source_map_id, request->source_map_id,
                      request->source_map_id_length);
      assign_identity(batch.frame_id, request->frame_id, request->frame_id_length);
    }
    batch.operations.push_back(std::move(operation));
    std::vector<phx::mc::TetEditResult> results;
    const TetEditStatus status = mesh->state.apply_edit_batch(batch, results);
    *edit_status = static_cast<int32_t>(status);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_refine(phx_mc_tet_mesh* mesh, uint64_t expected_generation,
                               int32_t identity_bound, const char* source_id,
                               int64_t source_id_length, const char* source_revision,
                               int64_t source_revision_length, const char* source_map_id,
                               int64_t source_map_id_length, const char* frame_id,
                               int64_t frame_id_length, double radius_edge_bound,
                               int64_t size_count, const double* sizes,
                               int32_t* edit_status, int32_t* run_status,
                               int64_t* inserted_count, int64_t* unmet_count) {
  if (mesh == nullptr || edit_status == nullptr || run_status == nullptr ||
      inserted_count == nullptr || unmet_count == nullptr || size_count < 0 ||
      !addressable(size_count, 1, sizeof(double)) || (size_count > 0 && sizes == nullptr) ||
      !valid_identity_fields(identity_bound, source_id, source_id_length,
                             source_revision, source_revision_length, source_map_id,
                             source_map_id_length, frame_id, frame_id_length)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *edit_status = PHX_MC_TET_EDIT_REJECTED_TOPOLOGY;
  *run_status = PHX_MC_UNRESOLVED;
  *inserted_count = 0;
  *unmet_count = 0;
  if (expected_generation != mesh->state.generation() ||
      !identity_matches(mesh->state, identity_bound, source_id, source_id_length,
                        source_revision, source_revision_length, source_map_id,
                        source_map_id_length, frame_id, frame_id_length)) {
    *edit_status = PHX_MC_TET_EDIT_STALE_GENERATION;
    return PHX_MC_OK;
  }
  return guarded([&]() -> int32_t {
    const std::int32_t status = phx::mc::refine_mesh(
        mesh->state, radius_edge_bound,
        input_span(sizes, size_count), *inserted_count, *unmet_count);
    *run_status = status;
    *edit_status = PHX_MC_TET_EDIT_APPLIED;
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_improve_quality(phx_mc_tet_mesh* mesh, uint64_t expected_generation,
                                int32_t identity_bound, const char* source_id,
                                int64_t source_id_length, const char* source_revision,
                                int64_t source_revision_length, const char* source_map_id,
                                int64_t source_map_id_length, const char* frame_id,
                                int64_t frame_id_length, double radius_edge_bound,
                                double minimum_dihedral_degrees,
                                double minimum_relative_determinant,
                                int64_t size_count, const double* sizes,
                                int32_t* edit_status, int32_t* run_status,
                                int64_t* inserted_vertices, int64_t* collapsed_vertices,
                                int64_t* relocated_vertices, int64_t* unmet_cells,
                                double* score_quantum) {
  if (mesh == nullptr || edit_status == nullptr || run_status == nullptr ||
      inserted_vertices == nullptr || collapsed_vertices == nullptr ||
      relocated_vertices == nullptr || unmet_cells == nullptr ||
      score_quantum == nullptr || size_count < 0 ||
      !addressable(size_count, 1, sizeof(double)) || (size_count > 0 && sizes == nullptr) ||
      !valid_identity_fields(identity_bound, source_id, source_id_length,
                             source_revision, source_revision_length, source_map_id,
                             source_map_id_length, frame_id, frame_id_length)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *edit_status = PHX_MC_TET_EDIT_REJECTED_TOPOLOGY;
  *run_status = PHX_MC_UNRESOLVED;
  *inserted_vertices = 0;
  *collapsed_vertices = 0;
  *relocated_vertices = 0;
  *unmet_cells = 0;
  *score_quantum = phx::mc::kTetScoreQuantum;
  if (expected_generation != mesh->state.generation() ||
      !identity_matches(mesh->state, identity_bound, source_id, source_id_length,
                        source_revision, source_revision_length, source_map_id,
                        source_map_id_length, frame_id, frame_id_length)) {
    *edit_status = PHX_MC_TET_EDIT_STALE_GENERATION;
    return PHX_MC_OK;
  }
  return guarded([&]() -> int32_t {
    const std::int32_t status = phx::mc::improve_mesh(
        mesh->state,
        phx::mc::TetQualityRequest{radius_edge_bound, minimum_dihedral_degrees,
                                   minimum_relative_determinant},
        input_span(sizes, size_count), *inserted_vertices, *collapsed_vertices,
        *relocated_vertices, *unmet_cells);
    *run_status = status;
    *edit_status = PHX_MC_TET_EDIT_APPLIED;
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_unmet_count(const phx_mc_tet_mesh* mesh, int64_t* count) {
  if (mesh == nullptr || count == nullptr) return PHX_MC_INVALID_ARGUMENT;
  if (mesh->state.unmet_records().size() >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  *count = static_cast<int64_t>(mesh->state.unmet_records().size());
  return PHX_MC_OK;
}

int32_t phx_mc_tet_mesh_unmet(const phx_mc_tet_mesh* mesh, int64_t capacity,
                              uint64_t* cell_ids, int32_t* criteria,
                              int32_t* reasons, double* values, uint64_t* apex_vertex_ids,
                              double* apex_angles, uint64_t* facet_ids,
                              double* facet_ratios) {
  if (mesh == nullptr || capacity < 0 ||
      !addressable(capacity, 1, sizeof(uint64_t)) ||
      !addressable(capacity, 1, sizeof(int32_t)) ||
      !addressable(capacity, 1, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const auto& records = mesh->state.unmet_records();
  if (records.size() > static_cast<std::size_t>(capacity)) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  const int64_t count = static_cast<int64_t>(records.size());
  if (!required_bank(cell_ids, count) || !required_bank(criteria, count) ||
      !required_bank(reasons, count) || !required_bank(values, count) ||
      !required_bank(apex_vertex_ids, count) || !required_bank(apex_angles, count) ||
      !required_bank(facet_ids, count) || !required_bank(facet_ratios, count)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (std::size_t row = 0; row < records.size(); ++row) {
    cell_ids[row] = records[row].cell_id;
    criteria[row] = records[row].criterion;
    reasons[row] = records[row].reason;
    values[row] = records[row].value;
    apex_vertex_ids[row] = records[row].apex_vertex;
    apex_angles[row] = records[row].apex_angle;
    facet_ids[row] = records[row].facet_id;
    facet_ratios[row] = records[row].facet_ratio;
  }
  return PHX_MC_OK;
}

int32_t phx_mc_tet_mesh_plane_record_count(const phx_mc_tet_mesh* mesh, int64_t* count) {
  if (mesh == nullptr || count == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return guarded([&]() -> int32_t {
    std::vector<uint64_t> vertex_ids;
    std::vector<phx::mc::TetPlaneRecord> records;
    mesh->state.plane_records(vertex_ids, records);
    *count = static_cast<int64_t>(records.size());
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_plane_records(const phx_mc_tet_mesh* mesh, int64_t capacity,
                                      uint64_t* vertex_ids, int32_t* face_rows,
                                      double* parameters, double* deviations) {
  if (mesh == nullptr || capacity < 0 || !addressable(capacity, 2, sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return guarded([&]() -> int32_t {
    std::vector<uint64_t> ids;
    std::vector<phx::mc::TetPlaneRecord> records;
    mesh->state.plane_records(ids, records);
    if (records.size() > static_cast<std::size_t>(capacity)) return PHX_MC_CAPACITY_EXCEEDED;
    const int64_t count = static_cast<int64_t>(records.size());
    if (!required_bank(vertex_ids, count) || !required_bank(face_rows, count) ||
        !required_bank(parameters, count) || !required_bank(deviations, count)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (std::size_t row = 0; row < records.size(); ++row) {
      vertex_ids[row] = ids[row];
      face_rows[row] = records[row].face_row;
      parameters[2 * row] = records[row].parameters[0];
      parameters[2 * row + 1] = records[row].parameters[1];
      deviations[row] = records[row].deviation;
    }
    return PHX_MC_OK;
  });
}

void phx_mc_tet_mesh_free(phx_mc_tet_mesh* mesh) { delete mesh; }

}  // extern "C"
