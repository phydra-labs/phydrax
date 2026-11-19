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
};

struct CadRefine2DStats {
  int64_t metric_evaluations = 0;
  int64_t insertions = 0;
  int64_t encroachment_refusals = 0;
  int64_t nonstar_refusals = 0;
  int64_t outside_refusals = 0;
  int64_t proximity_refusals = 0;
  int64_t unresolved_triangles = 0;
  double invalid_metric_point[2] = {0.0, 0.0};
};

struct CadRefine2DResult {
  Triangulation2D triangulation;
  std::vector<int32_t> vertex_map;
  CadRefine2DStats stats;
};

// The caller owns the callback context. Boundary constraints are never split.
// The cavity uses the SPD metric at the candidate, not a Euclidean chart
// circle: general spline charts can remain strongly anisotropic after scaling.
// Invalid metrics return PHX_MC_INVALID_INPUT and record the evaluation point.
int32_t cad_refine2d(const CadRefine2DInput& input, CadRefine2DResult& result);

}  // namespace phx::mc
