//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact contact classification of triangles, segments and points in 3D, shared
// by PLC recovery, surface arrangements and embedding certification.
//
// Every decision is a sign of orient3d on input points or of orient2d (or a
// degree-four product of orient2d values) on a coordinate-plane projection with
// an exactly nonzero normal component, so classifications are exact on the
// exact domain.  Intersection sets are convex (empty, a point, a segment or a
// polygon of at most six vertices).  Each contact point is defined exactly by
// the features of both inputs that contain it; its coordinates are either an
// input vertex (bound zero) or a bounded construction x = p + t (q - p) whose
// parameter is a ratio of exact expansions, with a rigorous max-norm bound on
// the distance from the exact point.
//
// Feature codes of a contact point on a triangle: k in {0, 1, 2} is vertex k,
// 3 + k is the edge opposite vertex k, kFeatureInterior is the relative
// interior.  Segments use 0 and 1 for their endpoints and kFeatureInterior.
//
// Identity: with vertex ids, a vertex-vertex contact is shared adjacency only
// when both ids agree (coincident positions with distinct ids are an illegal
// touching contact, equal ids with distinct positions an invalid input);
// without ids, coincident positions are shared.
#pragma once

#include <cstdint>

#include "phydrax_meshcore.h"

namespace phx::mc {

inline constexpr int8_t kFeatureInterior = 6;
inline constexpr int8_t kFeatureNone = -1;
inline constexpr int kMaxContactPoints = 6;

struct ContactPoint {
  double x[3];
  double bound;  // max-norm distance of x from the exact contact point
  int8_t first_feature;
  int8_t second_feature;
};

// Contact of two inputs: a phx_mc_intersection_class and the intersection's
// vertices (a point, the two segment endpoints ordered along the intersection
// line, or the polygon counterclockwise about the first triangle's normal).
struct Contact {
  int8_t kind = PHX_MC_DISJOINT;
  int32_t count = 0;
  ContactPoint points[kMaxContactPoints];
};

// Inputs must be finite and in the exact domain; ids are three per triangle,
// two per segment, or null.  Returns PHX_MC_OK, PHX_MC_DEGENERATE_INPUT for a
// collinear triangle or zero-length segment, or PHX_MC_INVALID_INPUT for
// repeated ids within one input or equal ids at distinct positions.
int32_t intersect_triangles(const double* const first[3], const double* const second[3],
                            const int64_t* first_ids, const int64_t* second_ids,
                            Contact& contact);
int32_t intersect_segment_triangle(const double* const segment[2],
                                   const double* const triangle[3], const int64_t* segment_ids,
                                   const int64_t* triangle_ids, Contact& contact);

// Location of a point relative to a nondegenerate triangle: side receives
// orient3d(t0, t1, t2, x); returns the feature containing x, or kFeatureNone
// when x is off the closed triangle.
int8_t locate_on_triangle(const double* x, const double* const triangle[3], int& side);

}  // namespace phx::mc
