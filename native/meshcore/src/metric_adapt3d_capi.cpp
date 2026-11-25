//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary of the metric adaptation loop (metric_adapt3d.hpp) on a
// persistent tetrahedral mesh: one run adapts the mesh in place and returns
// an owned result whose banks the caller reads by size.
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "metric_adapt3d.hpp"
#include "phydrax_meshcore.h"
#include "tet_mesh_handle.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

struct phx_mc_metric_adaptation {
  phx::mc::metric::AdaptOutput output;
};

namespace {

using phx::mc::addressable;
using phx::mc::copy_output_bank;
using phx::mc::guarded;
using phx::mc::output_bank_valid;
using phx::mc::metric::AdaptInput;

template <class T>
bool read_bank(const T* values, int64_t count, std::vector<T>& bank) {
  if (count < 0 || !addressable(count, 1, sizeof(T)) || (count > 0 && values == nullptr)) {
    return false;
  }
  if (count == 0) {
    bank.clear();
  } else {
    bank.assign(values, values + count);
  }
  return true;
}

bool read_request(const phx_mc_metric_adapt3d_request& request, AdaptInput& input) {
  const int64_t sources = request.source_count;
  const int64_t vertices = request.vertex_count;
  const int64_t cells = request.cell_count;
  if (!addressable(sources, 9, sizeof(double)) ||
      !addressable(vertices, 1, sizeof(uint64_t)) ||
      !addressable(cells, 1, sizeof(uint64_t)) ||
      !addressable(request.protected_edge_count, 2, sizeof(uint64_t)) ||
      request.stencil_offsets == nullptr || request.ancestry_offsets == nullptr) {
    return false;
  }
  if (!read_bank(request.source_points, 3 * sources, input.source_points) ||
      !read_bank(request.source_metrics, 9 * sources, input.source_metrics) ||
      !read_bank(request.source_logarithms, 9 * sources, input.source_logarithms) ||
      !read_bank(request.source_vertex_ids, sources, input.source_vertex_ids) ||
      request.source_cell_count < 0 || !addressable(request.source_cell_count, 4, sizeof(int64_t)) ||
      !read_bank(request.source_cells, 4 * request.source_cell_count, input.source_cells) ||
      !read_bank(request.vertex_ids, vertices, input.vertex_ids) ||
      !read_bank(request.stencil_offsets, vertices + 1, input.stencil_offsets) ||
      !read_bank(request.vertex_target_ids, vertices, input.vertex_target_ids) ||
      !read_bank(request.cell_ids, cells, input.cell_ids) ||
      !read_bank(request.cell_blocks, cells, input.cell_blocks) ||
      !read_bank(request.cell_target_ids, cells, input.cell_target_ids) ||
      !read_bank(request.ancestry_offsets, cells + 1, input.ancestry_offsets) ||
      !read_bank(request.ancestry_kinds, cells, input.ancestry_kinds) ||
      !read_bank(request.protected_edges, 2 * request.protected_edge_count,
                 input.protected_edges)) {
    return false;
  }
  const auto nondecreasing = [](const std::vector<int64_t>& offsets) {
    if (offsets.front() != 0) return false;
    for (std::size_t index = 1; index < offsets.size(); ++index) {
      if (offsets[index] < offsets[index - 1]) return false;
    }
    return true;
  };
  if (!nondecreasing(input.stencil_offsets) || !nondecreasing(input.ancestry_offsets)) {
    return false;
  }
  const int64_t terms = input.stencil_offsets.back();
  const int64_t ancestors = input.ancestry_offsets.back();
  if (!read_bank(request.stencil_rows, terms, input.stencil_rows) ||
      !read_bank(request.stencil_weights, terms, input.stencil_weights) ||
      !read_bank(request.ancestry_ids, ancestors, input.ancestry_ids)) {
    return false;
  }
  if (request.exact_source != nullptr) {
    input.exact_source = &phx::mc::metric::metric_carrier_source(*request.exact_source);
  }
  input.next_vertex_id = request.next_vertex_id;
  input.next_cell_id = request.next_cell_id;
  input.next_target_vertex = request.next_target_vertex;
  input.next_target_cell = request.next_target_cell;
  input.lower = request.lower;
  input.topology_operations = request.topology_operations != 0;
  input.relocation = request.relocation != 0;
  return true;
}

}  // namespace

extern "C" {

int32_t phx_mc_tet_mesh_metric_adapt(phx_mc_tet_mesh* mesh, uint64_t expected_generation,
                                     int32_t identity_bound, const char* source_id,
                                     int64_t source_id_length, const char* source_revision,
                                     int64_t source_revision_length, const char* source_map_id,
                                     int64_t source_map_id_length, const char* frame_id,
                                     int64_t frame_id_length,
                                     const phx_mc_metric_adapt3d_request* request,
                                     int32_t* edit_status, phx_mc_metric_adaptation** result) {
  if (mesh == nullptr || request == nullptr || edit_status == nullptr || result == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *result = nullptr;
  *edit_status = static_cast<int32_t>(PHX_MC_TET_EDIT_REJECTED_TOPOLOGY);
  if (!(request->lower > 0.0) || !(request->lower <= 1.0)) return PHX_MC_INVALID_ARGUMENT;
  const int32_t current = phx::mc::tet_mesh_current(
      *mesh, expected_generation, identity_bound, source_id, source_id_length, source_revision,
      source_revision_length, source_map_id, source_map_id_length, frame_id, frame_id_length);
  if (current != PHX_MC_OK) return current;
  return guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    AdaptInput input;
    if (!read_request(*request, input)) return PHX_MC_INVALID_ARGUMENT;
    auto owned = std::make_unique<phx_mc_metric_adaptation>();
    const phx::mc::TetEditStatus status =
        phx::mc::metric::adapt_metric(phx::mc::tet_mesh_state(*mesh), input, owned->output);
    *edit_status = static_cast<int32_t>(status);
    if (status == phx::mc::TetEditStatus::kApplied) *result = owned.release();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_metric_adaptation_sizes(const phx_mc_metric_adaptation* result, int64_t* sizes) {
  if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const phx::mc::metric::AdaptOutput& output = result->output;
  sizes[0] = static_cast<int64_t>(output.vertex_ids.size());
  sizes[1] = static_cast<int64_t>(output.stencil_rows.size());
  sizes[2] = static_cast<int64_t>(output.cell_ids.size());
  sizes[3] = static_cast<int64_t>(output.ancestry_ids.size());
  sizes[4] = static_cast<int64_t>(output.vertex_relation_sources.size());
  sizes[5] = static_cast<int64_t>(output.unrepresentable_edges.size() / 2);
  return PHX_MC_OK;
}

int32_t phx_mc_metric_adaptation_export(const phx_mc_metric_adaptation* result,
                                        const phx_mc_metric_adaptation_output* banks,
                                        int64_t counter_capacity) {
  if (counter_capacity < PHX_MC_METRIC_ADAPTATION_COUNTERS ||
      result == nullptr || banks == nullptr || banks->counters == nullptr ||
      banks->status == nullptr || banks->passes == nullptr || banks->stencil_offsets == nullptr ||
      banks->ancestry_offsets == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const phx::mc::metric::AdaptOutput& output = result->output;
  if (!output_bank_valid(output.vertex_ids, banks->vertex_ids) ||
      !output_bank_valid(output.vertex_target_ids, banks->vertex_target_ids) ||
      !output_bank_valid(output.stencil_rows, banks->stencil_rows) ||
      !output_bank_valid(output.stencil_weights, banks->stencil_weights) ||
      !output_bank_valid(output.metrics, banks->metrics) ||
      !output_bank_valid(output.cell_ids, banks->cell_ids) ||
      !output_bank_valid(output.cell_target_ids, banks->cell_target_ids) ||
      !output_bank_valid(output.cell_blocks, banks->cell_blocks) ||
      !output_bank_valid(output.ancestry_ids, banks->ancestry_ids) ||
      !output_bank_valid(output.ancestry_kinds, banks->ancestry_kinds) ||
      !output_bank_valid(output.cell_relocated, banks->cell_relocated) ||
      !output_bank_valid(output.vertex_relation_sources, banks->vertex_relation_sources) ||
      !output_bank_valid(output.vertex_relation_targets, banks->vertex_relation_targets) ||
      !output_bank_valid(output.vertex_relation_kinds, banks->vertex_relation_kinds) ||
      !output_bank_valid(output.unrepresentable_edges, banks->unrepresentable_edges)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return guarded([&]() -> int32_t {
    *banks->status = static_cast<int32_t>(output.status);
    *banks->passes = output.passes;
    const phx::mc::metric::AdaptCounters& counters = output.counters;
    const int64_t values[] = {counters.splits,           counters.collapses,
                              counters.flips23,          counters.flips32,
                              counters.ring_reconnections, counters.relocations,
                              counters.rejected_topology, counters.rejected_protected,
                              counters.rejected_uncertain, counters.rejected_invalid,
                              counters.retriangulations};
    static_assert(std::size(values) == PHX_MC_METRIC_ADAPTATION_COUNTERS);
    std::copy(std::begin(values), std::end(values), banks->counters);
    copy_output_bank(output.vertex_ids, banks->vertex_ids);
    copy_output_bank(output.vertex_target_ids, banks->vertex_target_ids);
    copy_output_bank(output.stencil_offsets, banks->stencil_offsets);
    copy_output_bank(output.stencil_rows, banks->stencil_rows);
    copy_output_bank(output.stencil_weights, banks->stencil_weights);
    copy_output_bank(output.metrics, banks->metrics);
    copy_output_bank(output.cell_ids, banks->cell_ids);
    copy_output_bank(output.cell_target_ids, banks->cell_target_ids);
    copy_output_bank(output.cell_blocks, banks->cell_blocks);
    copy_output_bank(output.ancestry_offsets, banks->ancestry_offsets);
    copy_output_bank(output.ancestry_ids, banks->ancestry_ids);
    copy_output_bank(output.ancestry_kinds, banks->ancestry_kinds);
    copy_output_bank(output.cell_relocated, banks->cell_relocated);
    copy_output_bank(output.vertex_relation_sources, banks->vertex_relation_sources);
    copy_output_bank(output.vertex_relation_targets, banks->vertex_relation_targets);
    copy_output_bank(output.vertex_relation_kinds, banks->vertex_relation_kinds);
    copy_output_bank(output.unrepresentable_edges, banks->unrepresentable_edges);
    return PHX_MC_OK;
  });
}

void phx_mc_metric_adaptation_free(phx_mc_metric_adaptation* result) { delete result; }

}  // extern "C"
