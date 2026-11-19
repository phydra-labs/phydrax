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

struct CadFaceMesh {
  std::vector<CadFaceVertex> vertices;
  std::vector<std::array<int32_t, 3>> triangles;
  std::vector<std::array<int32_t, 3>> slivers;
  std::vector<CadFaceEdge> boundary_edges;
  CadRefine2DStats stats;
};

// Validates the shared curve discretization once before any face consumes it.
int32_t validate_cad_curve_discretization(const phx_mc_cad_model& model,
                                          const CadCurveDiscretization& curves);

// Meshes one native chart, preserving curve samples and collapsed-pole records.
int32_t cad_mesh_face(const cad::CadGeometry& geometry,
                      const CadCurveDiscretization& curves, int32_t face,
                      double size, double tolerance, double theta,
                      CadFaceMesh& result);

}  // namespace phx::mc
