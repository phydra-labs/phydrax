// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cad_faces.hpp"

namespace phx::mc {

// Verifier input in chart-triangle order. Triangle vertex IDs refer to points
// in the assembled mesh, including repeated pole IDs in collapsed slivers.
struct CadChartEvidence {
  std::vector<std::array<int64_t, 3>> triangles;
  std::vector<int32_t> patches;
  std::vector<std::array<double, 6>> uv;
  std::vector<int64_t> chart_to_mesh;
  std::vector<std::array<int64_t, 3>> edge_chord;
  std::vector<int64_t> use_patch;
  std::vector<std::array<double, 2>> use_range;
  std::vector<int64_t> chord_use, chord_loop;
  std::vector<std::array<double, 4>> chord_uv;
  std::vector<std::array<double, 2>> chord_range;
};


struct CadSurfaceMesh {
  std::vector<std::array<double, 3>> points;
  std::vector<int8_t> vertex_dimensions;
  std::vector<int32_t> vertex_entities;
  std::vector<std::array<double, 2>> vertex_parameters;
  std::vector<std::array<int32_t, 3>> triangles;
  std::vector<int32_t> triangle_patches;
  std::vector<std::array<double, 6>> triangle_uv;
  std::vector<std::array<int32_t, 3>> triangle_edge_uses;
  std::vector<std::array<int32_t, 2>> facet_regions;
  std::vector<std::array<double, 6>> sliver_uv;
  std::vector<int32_t> sliver_patches;
  std::vector<std::array<int32_t, 2>> curve_edges;
  std::vector<int32_t> curve_edge_curves;
  std::vector<std::array<double, 2>> curve_edge_parameters;
  std::vector<CadRefine2DStats> face_stats;
  std::vector<int64_t> omitted_pole_points;
  // Rows are (face, pole use, protected sectors, free sectors).
  std::vector<std::array<int64_t, 4>> pole_sectors;
  std::vector<int64_t> periodic_corner_gaps;
  std::vector<uint64_t> max_periodic_corner_gap_ulps;
  CadChartEvidence chart;
  std::vector<phx_mc_cad_face_failure> face_failures;
};

int32_t cad_mesh_surface(const cad::CadGeometry& geometry, const CadCurveDiscretization& curves,
                         const double* sizes, const double* tolerances, double theta,
                         CadSurfaceMesh& result);

}  // namespace phx::mc
