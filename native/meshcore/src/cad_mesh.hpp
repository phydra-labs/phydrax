// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cad_faces.hpp"

namespace phx::mc {

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
  std::vector<int64_t> periodic_corner_gaps;
  std::vector<uint64_t> max_periodic_corner_gap_ulps;
  std::vector<phx_mc_cad_face_failure> face_failures;
};

int32_t cad_mesh_surface(const cad::CadGeometry& geometry, const CadCurveDiscretization& curves,
                         const double* sizes, const double* tolerances, double theta,
                         CadSurfaceMesh& result);

}  // namespace phx::mc
