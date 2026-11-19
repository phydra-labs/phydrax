//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Outward bounds of exact tetrahedron circumcenters and their C ABI.
#pragma once

#include <cstdint>

#include "phydrax_meshcore.h"

namespace phx::mc::restricted_layout {
inline constexpr int kDimensions = 3;
inline constexpr int kTetVertices = 4;
inline constexpr int kUpper = kDimensions;
inline constexpr int kCenterBounds = 2 * kDimensions;
}  // namespace phx::mc::restricted_layout

// Outward bounds for the exact circumcenters of positively oriented tets.
// center_bounds has shape (tet_count, 2, 3), with lower then upper per axis.
extern "C" PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status);
