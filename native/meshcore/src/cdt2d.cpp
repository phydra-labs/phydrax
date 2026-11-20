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
#include <numeric>
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
  // Cells and their segment rows in the order canonicalize_cells gives: each
  // kept triangle rotated to start at its smallest vertex (segment k stays
  // opposite vertex k), records sorted lexicographically by their vertices.
  // The vertex tuples are distinct, so that order is unique; it is built
  // here by a counting sort on the first vertex straight from the slots, then
  // each group ordered by rank (by radix when large), as sort_cell_records.
  const std::size_t vertex_total = static_cast<std::size_t>(triangulation.vertex_count());
  result->points = std::move(triangulation.coordinates);
  const std::size_t slots = triangulation.triangles.size();
  const auto kept = [&](std::size_t t) {
    return triangulation.live[t] != 0 && triangulation.region[t] != 0 &&
           !triangulation.is_ghost(static_cast<int32_t>(t));
  };
  const auto first_of = [](const Triangle2D& triangle) {
    int first = triangle.v[1] < triangle.v[0] ? 1 : 0;
    first = triangle.v[2] < triangle.v[first] ? 2 : first;
    return first;
  };
  std::vector<std::size_t> group_end(vertex_total + 1, 0);
  for (std::size_t t = 0; t < slots; ++t) {
    if (kept(t)) {
      const Triangle2D& triangle = triangulation.triangles[t];
      ++group_end[static_cast<std::size_t>(triangle.v[first_of(triangle)]) + 1];
    }
  }
  std::partial_sum(group_end.begin(), group_end.end(), group_end.begin());
  const std::size_t cell_count = group_end[vertex_total];
  const std::unique_ptr<int32_t[]> rows(new int32_t[6 * cell_count]);
  for (std::size_t t = 0; t < slots; ++t) {
    if (!kept(t)) continue;
    const Triangle2D& triangle = triangulation.triangles[t];
    const int first = first_of(triangle);
    int32_t* row = rows.get() + 6 * group_end[static_cast<std::size_t>(triangle.v[first])]++;
    for (int k = 0; k < 3; ++k) {
      const int r = (first + k) % 3;
      row[k] = triangle.v[r];
      const int32_t id = triangulation.constraints[3 * t + static_cast<std::size_t>(r)];
      row[3 + k] = id >= 0 ? id : -1;
    }
  }
  // group_end[v] is now the end of the group of first vertex v.
  int bits = 1;
  while ((std::uint64_t{1} << bits) <= static_cast<std::uint64_t>(vertex_total)) ++bits;
  std::unique_ptr<int32_t[]> scratch;
  std::vector<std::size_t> radix_offsets;
  for (std::size_t v = 0, begin = 0; v < vertex_total; begin = group_end[v++]) {
    const std::size_t size = group_end[v] - begin;
    if (size > 64) {
      if (!scratch) scratch.reset(new int32_t[6 * cell_count]);
      radix_sort_rows<6>(rows.get() + 6 * begin, scratch.get(), size, 1, 3, bits,
                         radix_offsets);
    } else if (size > 1) {
      rank_sort_group<6, 64>(rows.get() + 6 * begin, size, 3);
    }
  }
  result->cells.resize(3 * cell_count);
  result->cell_segments.resize(3 * cell_count);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    std::copy_n(rows.get() + 6 * cell, 3, result->cells.data() + 3 * cell);
    std::copy_n(rows.get() + 6 * cell + 3, 3, result->cell_segments.data() + 3 * cell);
  }
  mesher.publish_witnesses(*result);
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
