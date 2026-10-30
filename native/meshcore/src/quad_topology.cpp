//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Barycentric dual quadrilaterals and canonical reference-cell quad templates.
#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

extern "C" PHX_MC_API int32_t phx_mc_triangle_dual_quads(
    int64_t triangle_count, int64_t vertex_count, const int64_t* nodes,
    int64_t quad_capacity, int64_t* quads) {
  return phx::mc::guarded([&]() -> int32_t {
    if (!phx::mc::addressable(triangle_count, 12, sizeof(int64_t)) ||
        vertex_count < 0 || quad_capacity < 0 ||
        (triangle_count != 0 && (nodes == nullptr || quads == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (quad_capacity < 3 * triangle_count) return PHX_MC_CAPACITY_EXCEEDED;
    // Validate the complete candidate before writing any output row.
    for (int64_t triangle = 0; triangle < triangle_count; ++triangle) {
      const int64_t* row = nodes + 7 * triangle;
      for (int node = 0; node < 7; ++node) {
        if (row[node] < 0 || row[node] >= vertex_count) return PHX_MC_INVALID_INPUT;
        for (int prior = 0; prior < node; ++prior) {
          if (row[node] == row[prior]) return PHX_MC_INVALID_INPUT;
        }
      }
    }

    // Node order: (v0,v1,v2,e01,e12,e20,face center).
    constexpr int template_nodes[3][4] = {
        {0, 3, 6, 5}, {1, 4, 6, 3}, {2, 5, 6, 4}};
    for (int64_t triangle = 0; triangle < triangle_count; ++triangle) {
      for (int quad = 0; quad < 3; ++quad) {
        for (int corner = 0; corner < 4; ++corner) {
          quads[12 * triangle + 4 * quad + corner] =
              nodes[7 * triangle + template_nodes[quad][corner]];
        }
      }
    }
    return PHX_MC_OK;
  });
}

namespace phx::mc {

// Canonical 2-D tensor refinement consumed by the mixed-family owner.
int32_t quad_template_counts(int32_t axial, int32_t index, int32_t* counts) {
  return guarded([&]() -> int32_t {
    if (counts == nullptr || axial < 0 || axial > 1 || index < 0 || index >= 4) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    counts[0] = 4;
    counts[1] = index == 0 ? 1 : index == 1 ? 4 : 2;
    counts[2] = 4;
    return PHX_MC_OK;
  });
}

int32_t quad_template(int32_t axial, int32_t index, int64_t capacity,
                      double* references) {
  return guarded([&]() -> int32_t {
    int32_t counts[3];
    const int32_t status = quad_template_counts(axial, index, counts);
    if (status != PHX_MC_OK || references == nullptr || capacity < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int64_t entries = int64_t{counts[1]} * counts[2] * 2;
    if (capacity < entries) return PHX_MC_CAPACITY_EXCEEDED;
    constexpr int corners[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    constexpr int masks[4] = {0, 3, 1, 2};
    const int mask = masks[index];
    const double sx = (mask & 1) != 0 ? 0.5 : 1.0;
    const double sy = (mask & 2) != 0 ? 0.5 : 1.0;
    int child = 0;
    for (int x = 0; x < ((mask & 1) != 0 ? 2 : 1); ++x) {
      for (int y = 0; y < ((mask & 2) != 0 ? 2 : 1); ++y) {
        for (int corner = 0; corner < 4; ++corner) {
          references[8 * child + 2 * corner] = (x + corners[corner][0]) * sx;
          references[8 * child + 2 * corner + 1] = (y + corners[corner][1]) * sy;
        }
        ++child;
      }
    }
    return PHX_MC_OK;
  });
}

}  // namespace phx::mc
