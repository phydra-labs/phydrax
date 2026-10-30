//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Barycentric dual hexahedra of positively oriented tetrahedra.
#include <array>
#include <cstdint>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

extern "C" PHX_MC_API int32_t phx_mc_tetrahedron_dual_hexes(
    int64_t tetrahedron_count, int64_t vertex_count, const int64_t* nodes,
    int64_t hex_capacity, int64_t* hexes) {
  return phx::mc::guarded([&]() -> int32_t {
    if (vertex_count < 0 || hex_capacity < 0 ||
        !phx::mc::addressable(tetrahedron_count, 15, sizeof(int64_t)) ||
        !phx::mc::addressable(tetrahedron_count, 32, sizeof(int64_t)) ||
        (tetrahedron_count > 0 && (nodes == nullptr || hexes == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (hex_capacity < 4 * tetrahedron_count) return PHX_MC_CAPACITY_EXCEEDED;
    for (int64_t tet = 0; tet < tetrahedron_count; ++tet) {
      const int64_t* row = nodes + 15 * tet;
      for (int node = 0; node < 15; ++node) {
        if (row[node] < 0 || row[node] >= vertex_count) return PHX_MC_INVALID_INPUT;
        for (int prior = 0; prior < node; ++prior) {
          if (row[node] == row[prior]) return PHX_MC_INVALID_INPUT;
        }
      }
    }

    // v0..v3, e01,e02,e03,e12,e13,e23, f012,f013,f023,f123, cell center.
    constexpr std::array<std::array<int, 8>, 4> template_nodes{{
        {{0, 4, 10, 5, 6, 11, 14, 12}},
        {{1, 4, 11, 8, 7, 10, 14, 13}},
        {{2, 5, 10, 7, 9, 12, 14, 13}},
        {{3, 6, 12, 9, 8, 11, 14, 13}},
    }};
    for (int64_t tet = 0; tet < tetrahedron_count; ++tet) {
      for (int hex = 0; hex < 4; ++hex) {
        for (int corner = 0; corner < 8; ++corner) {
          hexes[32 * tet + 8 * hex + corner] =
              nodes[15 * tet + template_nodes[static_cast<std::size_t>(hex)]
                                             [static_cast<std::size_t>(corner)]];
        }
      }
    }
    return PHX_MC_OK;
  });
}
