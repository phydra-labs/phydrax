//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: constrained Delaunay triangulation with Ruppert/Chew refinement.
//
// Pipeline: Delaunay triangulation of the input points (triangulation2d.hpp),
// segment recovery by exact walks and Anglada's pseudo-polygon
// retriangulation, optional hull constraints, carving by flood fill over
// unconstrained edges, then Delaunay refinement by Lawson insertion (splits
// plus flips that never cross constrained edges).
//
// Constraint ids stored on edges: input segment index (>= 0), kHullConstraint
// for convex hull boundary edges, -1 for free edges.  region[t] is 1 for
// triangles of the meshed domain and 0 elsewhere (carved or ghost); a domain
// boundary always runs along constrained edges, so flips and insertions that
// never cross constrained edges preserve the partition.
//
// A split point of a subsegment is a floating-point construction and need not
// lie exactly on the segment.  The split is performed with exact
// orientation tests: a point exactly on the open subsegment splits its two
// triangles into four; a point strictly on one side is inserted into the
// triangle on that side (or attached to the hull when that side is a ghost),
// the two half edges become the constrained subsegments and the old edge
// becomes a free edge whose triangles take the region of the far side.  The
// result is always a valid triangulation; configurations where no exact
// placement exists (degenerate neighborhoods) leave the subsegment unsplit and
// are reported through the final quality check.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <new>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "angle.hpp"
#include "cdt2d.hpp"
#include "capi_guard.hpp"
#include "expansion.hpp"
#include "filtered.hpp"
#include "floating_point_mode.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "triangulation2d.hpp"

namespace phx::mc {
namespace {


int32_t constrained_delaunay(int64_t point_count, const double* points, int64_t segment_count,
                             const int32_t* segments, int64_t hole_count, const double* holes,
                             int32_t keep_convex_hull, double min_angle_degrees, double max_area,
                             phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || !addressable(point_count, 2, sizeof(double)) ||
      !addressable(point_count, 1, sizeof(int32_t)) ||
      !addressable(segment_count, 2, sizeof(int32_t)) ||
      !addressable(hole_count, 2, sizeof(double)) ||
      (point_count > 0 && points == nullptr) ||
      (segment_count > 0 && segments == nullptr) || (hole_count > 0 && holes == nullptr) ||
      (keep_convex_hull != 0 && keep_convex_hull != 1) ||
      !(min_angle_degrees >= 0.0 && min_angle_degrees < 60.0) || !(max_area > 0.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  // Vertex and constraint identifiers are stored as int32 indices. Keep only
  // their actual representability bounds; this is not a work/entity quota.
  constexpr int64_t kMaxConstraintCount =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  if (point_count > kMaxMeshPoints || segment_count > kMaxConstraintCount) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  int32_t status = validate_points(points, point_count, 2, nullptr);
  if (status == PHX_MC_OK) {
    status = validate_points(holes, hole_count, 2, nullptr);
  }
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int64_t k = 0; k < 2 * segment_count; ++k) {
    if (segments[k] < 0 || segments[k] >= point_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  if (point_count == 0) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  auto result = std::make_unique<phx_mc_mesh>();
  result->dimension = 2;
  result->input_point_count = point_count;
  Triangulation2D triangulation;
  triangulation.reset(points, point_count, nullptr);
  status = build_delaunay_2d(triangulation, points, point_count, nullptr, result->vertex_map);
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int64_t s = 0; s < segment_count; ++s) {
    if (result->vertex_map[static_cast<std::size_t>(segments[2 * s])] ==
        result->vertex_map[static_cast<std::size_t>(segments[2 * s + 1])]) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  const bool refine = min_angle_degrees > 0.0 || std::isfinite(max_area);
  ConstrainedMesher mesher(triangulation, static_cast<int32_t>(point_count), min_angle_degrees,
                           max_area);
  try {
    for (int64_t s = 0; s < segment_count; ++s) {
      mesher.insert_segment(result->vertex_map[static_cast<std::size_t>(segments[2 * s])],
                            result->vertex_map[static_cast<std::size_t>(segments[2 * s + 1])],
                            static_cast<int32_t>(s));
    }
    if (keep_convex_hull != 0) {
      mesher.constrain_hull();
    }
    mesher.carve(keep_convex_hull != 0, hole_count, holes);
    status = refine ? mesher.refine() : PHX_MC_OK;
  } catch (const StatusError& error) {
    return error.status;
  }
  result->points = std::move(triangulation.coordinates);
  for (std::size_t t = 0; t < triangulation.triangles.size(); ++t) {
    if (triangulation.live[t] == 0 || triangulation.region[t] == 0 ||
        triangulation.is_ghost(static_cast<int32_t>(t))) {
      continue;
    }
    const Triangle2D& triangle = triangulation.triangles[t];
    result->cells.insert(result->cells.end(), triangle.v, triangle.v + 3);
    for (int k = 0; k < 3; ++k) {
      const int32_t id = triangulation.constraints[3 * t + static_cast<std::size_t>(k)];
      result->cell_segments.push_back(id >= 0 ? id : -1);
    }
  }
  mesher.publish_witnesses(*result);
  canonicalize_cells(*result);
  *mesh = result.release();
  return status;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_constrained_delaunay_2d(int64_t point_count, const double* points,
                                       int64_t segment_count, const int32_t* segments,
                                       int64_t hole_count, const double* holes,
                                       int32_t keep_convex_hull, double min_angle_degrees,
                                       double max_area, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::constrained_delaunay(point_count, points, segment_count, segments, hole_count,
                                         holes, keep_convex_hull, min_angle_degrees, max_area, mesh);
  });
}

}  // extern "C"
