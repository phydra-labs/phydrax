// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <vector>

#include "triangulation2d.hpp"

namespace phx::mc {

// M = [[uu, uv], [uv, vv]] in the supplied chart coordinates.
struct CadMetric2D {
  double uu;
  double uv;
  double vv;
};
using CadMetric2DFunction = CadMetric2D (*)(const double uv[2], void* context);

struct CadRefine2DInput {
  const double* points = nullptr;
  int32_t point_count = 0;
  const int32_t* segments = nullptr;
  int32_t segment_count = 0;
  const double* holes = nullptr;
  int32_t hole_count = 0;
  bool keep_convex_hull = false;
  double min_angle_degrees = 25.0;
  int32_t point_cap = 0;
  CadMetric2DFunction metric = nullptr;
  void* metric_context = nullptr;
  const uint8_t* fixed_vertices = nullptr;
  int32_t (*admit_point)(void* context, const double point[2]) = nullptr;
  void* admit_context = nullptr;
  // Optional chart-to-physical position for angle-pass acceptance; return
  // nonzero on success and write three finite coordinates.
  int32_t (*surface_position)(void* context, const double point[2], double xyz[3]) = nullptr;
  void* surface_context = nullptr;
};

struct CadRefine2DStats {
  int64_t metric_evaluations = 0;
  int64_t insertions = 0;
  int64_t unresolved_size = 0;
  int64_t unresolved_angle = 0;
  int64_t encroachment_refusals = 0;
  int64_t nonstar_refusals = 0;
  int64_t outside_refusals = 0;
  int64_t proximity_refusals = 0;
  int64_t unresolved_triangles = 0;
  double invalid_metric_point[2] = {0.0, 0.0};
  int64_t excluded_refusals = 0;
  int64_t fixed_triangles_skipped = 0;
  int64_t relocations_tried = 0;
  int64_t relocations_accepted = 0;
  int64_t relocations_failed = 0;
};

enum class CadRefine2DRefusal : uint8_t {
  none, encroachment, outside, proximity, nonstar, excluded
};

struct CadRefine2DUnresolved {
  int32_t triangle = -1;
  bool bad_size = false;
  bool bad_angle = false;
  CadRefine2DRefusal refusal = CadRefine2DRefusal::none;
  double candidate[2] = {0.0, 0.0};
  int32_t segment = -1;
  bool relocation_failed = false;
};

struct CadRefine2DResult {
  Triangulation2D triangulation;
  std::vector<int32_t> vertex_map;
  CadRefine2DStats stats;
  std::vector<CadRefine2DUnresolved> unresolved;
};

// The caller owns the callback context and supplies a metric graded to boundary
// spacing (face stage, design B.2); boundary constraints are never split.
// With graded metrics and boundary lengths <= 1, no refusal was observed on the
// measured fixtures, but this is not a general no-refusal guarantee.
// A triangle is oversized when its metric circumradius exceeds sqrt(1/2) of
// the larger of one unit and the metric length (at its centroid) of each of
// its fixed segments, since no insertion can shorten a segment.
// On completed or point-cap exits, statistics count each unresolved triangle
// and each size or angle criterion it misses; refusal counts identify the rule.
// Refinement preserves exact orientation and constraints and terminates at the cap.
// Invalid metrics return PHX_MC_INVALID_INPUT and record the evaluation point.
// On normal and point-cap exits, unresolved lists failing live triangles in
// slot order with their last refusal, if any; a none refusal has no candidate.
// An encroachment segment is an index into the caller's input segment array.
// Encroachment and proximity refusals record that relocation was attempted
// and failed (encroachment: ray, offcenter, circumcenter; proximity: the
// circumcenter); each outcome is also counted in the relocation statistics.
// Triangles with three fixed input vertices are neither refined nor scanned;
// fixed_triangles_skipped counts them in the final mesh. The optional admission
// callback rejects candidate points before their metric is evaluated.
// The caller must ensure all other evaluated triangle centroids are in the
// metric callback's domain; candidate admission does not filter centroids.
// After a successful census, an angle pass moves free unconstrained vertices
// of triangles whose smallest metric angle is below halfway between
// min_angle_degrees and 60, worst star first. A move must improve the star's
// worst metric angle and keep every star triangle passing the refinement test;
// with surface_position it must also keep the star's smallest 3D angle from
// decreasing and its longest 3D edge from growing, otherwise it must keep the
// star locally metric-Delaunay. A failed or nonfinite position evaluation
// returns PHX_MC_INVALID_INPUT.
int32_t cad_refine2d(const CadRefine2DInput& input, CadRefine2DResult& result);

// Re-evaluate one live finite region triangle using the same centroid metric
// and criteria as refinement; the callback context remains caller-owned.
// Returns the metric status and sets the flags on success; it also records the
// callback evaluation and any invalid metric point in result.stats.
int32_t cad_refine2d_bad_triangle(const CadRefine2DInput& input,
                                  CadRefine2DResult& result, int32_t triangle,
                                  bool& bad_size, bool& bad_angle);

}  // namespace phx::mc
