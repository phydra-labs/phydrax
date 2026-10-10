//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#pragma once

#include <cstdint>

#include "phydrax_meshcore.h"

// Canonical primal facets and their numerical Voronoi duals, recomputed from
// one complete native Delaunay snapshot. This is a geometry request workset,
// not a certificate of source intersection or restricted-domain coverage.
//
// The domain is lower[3], upper[3]. Outputs have max_facets rows: facets (F,3)
// sorted vertex ids; cells (F,2) finite row indices, second -1 for a hull face;
// endpoints (F,2,3), clipped to the domain; kinds (F,) 0 segment, 1 ray,
// 2 no domain intersection; item_status (F,) OK or DEGENERATE_INPUT when a
// numerical circumcenter/direction cannot be constructed. Failed rows have
// zero endpoints and must not be treated as absence of a source crossing.
// Numerical clipping and circumcenters are NOT outward-rounded enclosures.
// A consumer must establish their error bounds before certifying dual roots.
//
// Exact predicates validate positive tetrahedra, reciprocal facet incidence,
// and local weak Delaunay legality. Bounds/capacity refusal occurs before any
// output is changed. Work units: one per cell, primal half-facet and unique
// facet. counters (4): work, finite dual segments, hull rays, failed rows.
extern "C" PHX_MC_API int32_t phx_mc_restricted_dual_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* domain, int64_t max_facets,
    int64_t work_limit, int32_t* facets, int32_t* cells, double* endpoints,
    int32_t* kinds, int32_t* item_status, int64_t* facet_count,
    int64_t* counters);

// Outward bounds of exact circumcenters of positively oriented binary64
// tetrahedra. center_bounds (T,2,3): lower then upper; item_status (T,) OK or
// DEGENERATE_INPUT when a finite enclosing quotient could not be established.
// Failed rows are [-infinity,infinity], not guessed centers. Polynomial
// constructions reuse the exact expansion owner; final division is widened.
// Invalid input refuses the complete call before outputs change.
extern "C" PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status);

// Bounds of hull dual rays over a finite parameter interval containing every
// point of that ray in the declared box. Inputs centers (T,2,3) must be the
// enclosing output of restricted_centers_3d for this same triangulation;
// facets (H,3) and cells (H,) identify hull primal facets and their finite cell.
// Exact expansion cross products bound outward normal directions; travel is
// widened to guarantee exit from at least one domain slab. No numerical
// clipping is used. Outputs endpoint_bounds (H,2,2,3), kinds (H,) 1 ray-cover
// segment or 2 certified disjoint ray, status (H,) OK or DEGENERATE_INPUT
// (unbounded endpoints). Failed rows remain unresolved. roots outside the
// declared box may be conservatively included, never silently omitted.
extern "C" PHX_MC_API int32_t phx_mc_restricted_rays_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* center_bounds, int64_t ray_count,
    const int32_t* facets, const int32_t* cells, const double* domain,
    double* endpoint_bounds, int32_t* kinds, int32_t* item_status);
