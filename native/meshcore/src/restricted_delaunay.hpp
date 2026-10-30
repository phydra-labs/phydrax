//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Numerical restricted-Delaunay dual helpers and their C ABI.
#pragma once

#include <cstdint>

#include "phydrax_meshcore.h"

namespace phx::mc::restricted_layout {
inline constexpr int kDimensions = 3;
inline constexpr int kTetVertices = 4;
inline constexpr int kFacetVertices = 3;
inline constexpr int kIncidentCells = 2;
inline constexpr int kUpper = kDimensions;
inline constexpr int kCenterBounds = 2 * kDimensions;
inline constexpr int kDualEndpoints = 2 * kDimensions;
inline constexpr int kRayEndpointBounds = 2 * kCenterBounds;
inline constexpr int kRayEndLower = kCenterBounds;
inline constexpr int kRayEndUpper = kCenterBounds + kDimensions;
inline constexpr int32_t kSegment = 0;
inline constexpr int32_t kRay = 1;
inline constexpr int32_t kDisjoint = 2;
}  // namespace phx::mc::restricted_layout

// Canonical primal facets and their numerical Voronoi duals. Endpoints are
// floating-point geometry, not enclosures or certificates of source coverage.
// facet_capacity is the exact output row extent; 4*T rows always suffice.
extern "C" PHX_MC_API int32_t phx_mc_restricted_dual_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* domain, int64_t facet_capacity,
    int32_t* facets, int32_t* cells, double* endpoints, int32_t* kinds,
    int32_t* item_status, int64_t* facet_count);

// Outward bounds for the exact circumcenters of positively oriented tets.
// center_bounds has shape (tet_count, 2, 3), with lower then upper per axis.
extern "C" PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status);

// Outward bounds of hull dual rays for a finite parameter interval covering all
// ray points in the declared domain. Inputs use the center_bounds produced for
// the same triangulation; facets/cells have ray_count rows.
extern "C" PHX_MC_API int32_t phx_mc_restricted_rays_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* center_bounds, int64_t ray_count,
    const int32_t* facets, const int32_t* cells, const double* domain,
    double* endpoint_bounds, int32_t* kinds, int32_t* item_status);
