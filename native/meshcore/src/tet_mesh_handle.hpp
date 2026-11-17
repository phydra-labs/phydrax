//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Access to the opaque C handle of a persistent tetrahedral mesh for the
// translation units that add entry points to its C boundary. The handle and
// these functions are defined in tet_mesh_capi.cpp.
#pragma once

#include "phydrax_meshcore.h"
#include "tet_mesh.hpp"

#include <cstdint>

namespace phx::mc {

// The mesh a live handle owns.
TetMesh& tet_mesh_state(phx_mc_tet_mesh& mesh) noexcept;

// PHX_MC_OK when the identity fields are well formed and name the mesh's
// current generation and source identity; PHX_MC_INVALID_ARGUMENT for
// malformed identity fields and PHX_MC_STALE_GENERATION for any mismatch.
int32_t tet_mesh_current(const phx_mc_tet_mesh& mesh, uint64_t expected_generation,
                         int32_t identity_bound, const char* source_id,
                         int64_t source_id_length, const char* source_revision,
                         int64_t source_revision_length, const char* source_map_id,
                         int64_t source_map_id_length, const char* frame_id,
                         int64_t frame_id_length) noexcept;

}  // namespace phx::mc
