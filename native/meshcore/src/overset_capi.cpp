//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C entry of the overset assembly: argument validation, then overset.cpp.
#include <cmath>
#include <cstdint>
#include <limits>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "overset.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {

bool offsets_valid(const int64_t* offsets, int32_t parts) {
  if (offsets == nullptr || offsets[0] != 0) return false;
  for (int32_t part = 0; part < parts; ++part) {
    if (offsets[part + 1] < offsets[part]) return false;
  }
  return offsets[parts] <= std::numeric_limits<int32_t>::max();
}

// Every row of `width` entries holds `required` (or, when `optional`, all
// `width`) indices in its part's vertex range, then -1 padding.
bool rows_valid(const int32_t* rows, const int64_t* row_offsets, const int64_t* vertex_offsets,
                int32_t parts, int width, int required) {
  for (int32_t part = 0; part < parts; ++part) {
    const int64_t lo = vertex_offsets[part], hi = vertex_offsets[part + 1];
    for (int64_t row = row_offsets[part]; row < row_offsets[part + 1]; ++row) {
      const int32_t* r = rows + width * row;
      const int used = r[required] < 0 ? required : width;
      for (int k = 0; k < width; ++k) {
        if (k < used ? (r[k] < lo || r[k] >= hi) : r[k] != -1) return false;
      }
    }
  }
  return true;
}

int32_t validate(const phx_mc_overset_parts* in, const phx_mc_overset_result* out) {
  if (in == nullptr || out == nullptr || in->part_count < 1 || in->fringe_layers < 1 ||
      (in->donor_priority != 0 && in->donor_priority != 1) || out->stats == nullptr ||
      in->priorities == nullptr || in->name_ranks == nullptr ||
      !offsets_valid(in->vertex_offsets, in->part_count) ||
      !offsets_valid(in->cell_offsets, in->part_count) ||
      !offsets_valid(in->face_offsets, in->part_count)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int64_t vertices = in->vertex_offsets[in->part_count];
  const int64_t cells = in->cell_offsets[in->part_count];
  const int64_t faces = in->face_offsets[in->part_count];
  if ((vertices > 0 &&
       (in->vertices == nullptr || out->vertex_status == nullptr || out->fringe_layer == nullptr ||
        out->wall_contact == nullptr || out->protected_conflict == nullptr ||
        out->donor_status == nullptr || out->donor_cell == nullptr ||
        out->donor_reference == nullptr || out->donor_residual == nullptr ||
        out->donor_measure == nullptr || out->candidate_parts == nullptr)) ||
      (cells > 0 && (in->cells == nullptr || in->cell_ids == nullptr ||
                     in->cell_excluded == nullptr || out->cell_status == nullptr ||
                     out->ambiguous == nullptr)) ||
      (faces > 0 && (in->faces == nullptr || in->face_roles == nullptr)) ||
      !phx::mc::addressable(vertices, 3, sizeof(double)) ||
      !phx::mc::addressable(cells, 8, sizeof(int32_t)) ||
      !phx::mc::addressable(faces, 4, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t k = 0; k < 3 * vertices; ++k) {
    if (!std::isfinite(in->vertices[k])) return PHX_MC_NONFINITE_INPUT;
    if (!phx::mc::coordinate_in_domain(in->vertices[k])) return PHX_MC_RANGE_ERROR;
  }
  for (int64_t face = 0; face < faces; ++face) {
    if (in->face_roles[face] > 2) return PHX_MC_INVALID_INPUT;
  }
  if (!rows_valid(in->cells, in->cell_offsets, in->vertex_offsets, in->part_count, 8, 4) ||
      !rows_valid(in->faces, in->face_offsets, in->vertex_offsets, in->part_count, 4, 3)) {
    return PHX_MC_INVALID_INPUT;
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_overset_assemble(const phx_mc_overset_parts* parts,
                                const phx_mc_overset_result* result) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const int32_t status = validate(parts, result);
    if (status != PHX_MC_OK) return status;
    const phx::mc::overset::Input input{
        parts->part_count,   parts->vertex_offsets, parts->vertices,      parts->cell_offsets,
        parts->cells,        parts->cell_ids,       parts->cell_excluded, parts->face_offsets,
        parts->faces,        parts->face_roles,     parts->priorities,    parts->name_ranks,
        parts->fringe_layers, parts->donor_priority, parts->fringe_donors, parts->prove};
    const phx::mc::overset::Output output{
        result->vertex_status,   result->fringe_layer,   result->wall_contact,
        result->protected_conflict, result->cell_status, result->ambiguous,
        result->donor_status,    result->donor_cell,     result->donor_reference,
        result->donor_residual,  result->donor_measure,  result->candidate_parts,
        result->stats};
    phx::mc::overset::assemble(input, output);
    return PHX_MC_OK;
  });
}

}  // extern "C"
