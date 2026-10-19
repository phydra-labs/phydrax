//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#pragma once

#include <cstdint>

#include "phydrax_meshcore.h"

namespace phx::mc {

enum class BoundaryPolicy : std::int32_t {
  kFixed = PHX_MC_TET_MESH_BOUNDARY_FIXED,
  kConforming = PHX_MC_TET_MESH_BOUNDARY_CONFORMING,
};

static_assert(PHX_MC_PLC3D_FIXED == PHX_MC_TET_MESH_BOUNDARY_FIXED);
static_assert(PHX_MC_PLC3D_CONFORMING == PHX_MC_TET_MESH_BOUNDARY_CONFORMING);

}  // namespace phx::mc
