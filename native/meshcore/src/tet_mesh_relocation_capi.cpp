//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary for relocation probes and in-place relocation commits of
// persistent tetrahedral meshes.
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "tet_mesh.hpp"
#include "tet_mesh_handle.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

using phx::mc::addressable;
using phx::mc::guarded;
using phx::mc::TetEdit;
using phx::mc::TetEditKind;
using phx::mc::TetEditStatus;
using phx::mc::TetEditVertex;

// Offsets of `count` consecutive groups: zero first, nondecreasing.
bool group_offsets(const int64_t* offsets, int64_t count) noexcept {
  if (offsets == nullptr || offsets[0] != 0) return false;
  for (int64_t group = 0; group < count; ++group) {
    if (offsets[group + 1] < offsets[group]) return false;
  }
  return true;
}

// The relocation request of phx_mc_tet_mesh_apply_edit that moves `count`
// vertices without a source witness or protection, clears their edge
// parameters, supplies no ancestry and declares the given star cells.
TetEdit relocation_operation(const uint64_t* vertex_ids, const double* points, int64_t count,
                             const uint64_t* cell_ids, int64_t cell_count) {
  TetEdit operation;
  operation.kind = TetEditKind::kRelocate;
  operation.removed_cell_ids.assign(cell_ids, cell_ids + cell_count);
  operation.changed_vertices.reserve(static_cast<std::size_t>(count));
  for (int64_t row = 0; row < count; ++row) {
    TetEditVertex vertex;
    vertex.id = vertex_ids[row];
    std::copy_n(points + 3 * row, 3, vertex.carrier.begin());
    vertex.edge_witness_supplied = true;
    operation.changed_vertices.push_back(vertex);
  }
  return operation;
}

}  // namespace

extern "C" {

int32_t phx_mc_tet_mesh_probe_relocations(
    phx_mc_tet_mesh* mesh, uint64_t expected_generation, int32_t identity_bound,
    const char* source_id, int64_t source_id_length, const char* source_revision,
    int64_t source_revision_length, const char* source_map_id, int64_t source_map_id_length,
    const char* frame_id, int64_t frame_id_length, int64_t group_count,
    const int64_t* vertex_offsets, const uint64_t* vertex_ids, const double* points,
    const int64_t* cell_offsets, const uint64_t* cell_ids, int32_t* decided,
    int32_t* edit_statuses) {
  if (mesh == nullptr || group_count < 0 || !addressable(group_count, 1, sizeof(int64_t)) ||
      !addressable(group_count, 1, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t current = phx::mc::tet_mesh_current(
      *mesh, expected_generation, identity_bound, source_id, source_id_length,
      source_revision, source_revision_length, source_map_id, source_map_id_length, frame_id,
      frame_id_length);
  if (current == PHX_MC_INVALID_ARGUMENT || group_count == 0) return current;
  if (decided == nullptr || edit_statuses == nullptr ||
      !group_offsets(vertex_offsets, group_count) || !group_offsets(cell_offsets, group_count)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t vertex_count = vertex_offsets[group_count];
  const int64_t cell_count = cell_offsets[group_count];
  if (!addressable(vertex_count, 3, sizeof(double)) ||
      !addressable(vertex_count, 1, sizeof(uint64_t)) ||
      !addressable(cell_count, 1, sizeof(uint64_t)) ||
      (vertex_count > 0 && (vertex_ids == nullptr || points == nullptr)) ||
      (cell_count > 0 && cell_ids == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  std::fill_n(decided, static_cast<std::size_t>(group_count), 0);
  std::fill_n(edit_statuses, static_cast<std::size_t>(group_count),
              static_cast<int32_t>(PHX_MC_TET_EDIT_REJECTED_TOPOLOGY));
  if (current != PHX_MC_OK) return current;
  return guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const phx::mc::TetMesh& state = phx::mc::tet_mesh_state(*mesh);
    for (int64_t group = 0; group < group_count; ++group) {
      const int64_t first = vertex_offsets[group];
      const int64_t first_cell = cell_offsets[group];
      const TetEdit operation = relocation_operation(
          vertex_ids + first, points + 3 * first, vertex_offsets[group + 1] - first,
          cell_ids + first_cell, cell_offsets[group + 1] - first_cell);
      TetEditStatus status = TetEditStatus::kRejectedTopology;
      if (state.probe_relocation_edit(operation, status)) {
        decided[group] = 1;
        edit_statuses[group] = static_cast<int32_t>(status);
      }
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_tet_mesh_apply_relocation(
    phx_mc_tet_mesh* mesh, uint64_t expected_generation, int32_t identity_bound,
    const char* source_id, int64_t source_id_length, const char* source_revision,
    int64_t source_revision_length, const char* source_map_id, int64_t source_map_id_length,
    const char* frame_id, int64_t frame_id_length, int64_t vertex_count,
    const uint64_t* vertex_ids, const double* points, int64_t cell_count,
    const uint64_t* cell_ids, int32_t* decided, int32_t* edit_status) {
  if (mesh == nullptr || decided == nullptr || edit_status == nullptr || vertex_count < 0 ||
      cell_count < 0 || !addressable(vertex_count, 3, sizeof(double)) ||
      !addressable(vertex_count, 1, sizeof(uint64_t)) ||
      !addressable(cell_count, 1, sizeof(uint64_t)) ||
      (vertex_count > 0 && (vertex_ids == nullptr || points == nullptr)) ||
      (cell_count > 0 && cell_ids == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *decided = 0;
  *edit_status = static_cast<int32_t>(PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  const int32_t current = phx::mc::tet_mesh_current(
      *mesh, expected_generation, identity_bound, source_id, source_id_length,
      source_revision, source_revision_length, source_map_id, source_map_id_length, frame_id,
      frame_id_length);
  if (current != PHX_MC_OK) return current;
  return guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const TetEdit operation =
        relocation_operation(vertex_ids, points, vertex_count, cell_ids, cell_count);
    TetEditStatus status = TetEditStatus::kRejectedTopology;
    if (phx::mc::tet_mesh_state(*mesh).apply_relocation_edit(operation, status)) {
      *decided = 1;
      *edit_status = static_cast<int32_t>(status);
    }
    return PHX_MC_OK;
  });
}

}  // extern "C"
