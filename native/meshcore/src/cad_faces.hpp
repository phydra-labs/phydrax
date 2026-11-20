// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cad_curve_discretization.hpp"
#include "cad_geometry.hpp"
#include "cad_refine2d.hpp"

namespace phx::mc {

struct CadFaceVertex {
  std::array<double, 2> uv;
  int32_t dimension;
  int32_t entity;
  int32_t curve_index;
};

struct CadFaceEdge {
  std::array<int32_t, 2> vertices;
  int32_t use;
  int32_t curve;
  std::array<double, 2> parameters;
};
// Snapping uses a pre-existing adjacent curve sample at every use through the
// pole only when their exact offsets agree, within [spacing/2, spacing] and
// within the computed pole-zone width. The lower limit is the existing
// half-metric-unit boundary clearance; the upper preserves the original spacing.
// A ring edge or fan triangle may meet a trim/seam chord only at shared chart
// vertex IDs, or (for a triangle) along the identical chord edge. All other
// contacts, including non-shared touches and partial overlaps, remain free.
struct CadPoleSectors {
  int32_t use;
  int64_t fixed;
  int64_t free;
};


struct CadFaceMesh {
  std::vector<CadFaceVertex> vertices;
  std::vector<std::array<int32_t, 3>> triangles;
  std::vector<std::array<int32_t, 3>> slivers;
  std::vector<CadFaceEdge> boundary_edges;
  CadRefine2DStats stats;
  int64_t omitted_pole_points = 0;
  std::vector<CadPoleSectors> pole_sectors;
  std::vector<std::array<int32_t, 2>> protected_pole_edges;
  int64_t periodic_corner_gaps = 0;
  uint64_t max_periodic_corner_gap_ulps = 0;
  phx_mc_cad_face_failure failure{};
  std::vector<phx_mc_cad_face_failure> refinement_failures;
};

// Exact chart segment contact under the shared-vertex/chord rule above.
bool cad_face_segments_conflict(int32_t a, int32_t b, const std::array<double, 2>& auv,
                                const std::array<double, 2>& buv, int32_t c, int32_t d,
                                const std::array<double, 2>& cuv, const std::array<double, 2>& duv);

// Validates the shared curve discretization once before any face consumes it.
int32_t validate_cad_curve_discretization(const phx_mc_cad_model& model,
                                          const CadCurveDiscretization& curves);

// Meshes one native chart, preserving curve samples and collapsed-pole records.
int32_t cad_mesh_face(const cad::CadGeometry& geometry,
                      const CadCurveDiscretization& curves, int32_t face,
                      double size, double tolerance, double theta,
                      CadFaceMesh& result,
                      std::vector<std::array<double, 2>>* metric_points = nullptr);

}  // namespace phx::mc
