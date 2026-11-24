// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Curving from stored curve parameters and triangle UV records. P2 has a
// native Bernstein validity and repair step; P3 only prepares the nodes.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cad_geometry.hpp"
#include "phydrax_meshcore.h"

namespace phx::mc {

// Views of a linear tet mesh whose boundary is the face stage's surface; the
// layout and preconditions are those of phx_mc_cad_quadratic_request.
struct CadQuadraticInput {
  int64_t point_count = 0;
  const double* points = nullptr;
  int64_t surface_point_count = 0;
  const int8_t* vertex_dimensions = nullptr;
  const int32_t* vertex_entities = nullptr;
  int64_t tetrahedron_count = 0;
  const int32_t* tetrahedra = nullptr;
  int64_t triangle_count = 0;
  const int32_t* triangles = nullptr;
  const int32_t* triangle_patches = nullptr;
  const double* triangle_uv = nullptr;
  int64_t curve_edge_count = 0;
  const int32_t* curve_edges = nullptr;
  const int32_t* curve_edge_curves = nullptr;
  const double* curve_edge_parameters = nullptr;
  double relative_floor = 0.0;
  double length_scale = 0.0;
  double sampled_minimum_scaled_jacobian = 0.0;
  double sampled_minimum_determinant_ratio = 0.0;
};

struct CadQuadraticMesh {
  std::vector<std::array<int32_t, 2>> edges;
  std::vector<std::array<double, 3>> nodes;
  std::vector<int8_t> edge_kinds;
  std::vector<int32_t> edge_entities;
  std::vector<std::array<int64_t, 6>> cell_edges;
  std::vector<int32_t> cell_decisions;
  // Decisions of the lifted nodes before any repair (initial_invalid and
  // initial_unresolved count these).
  std::vector<int32_t> initial_decisions;
  // Straightened curved boundary edges: (edge, cell whose failure caused it).
  std::vector<std::array<int64_t, 2>> straightened;
  phx_mc_cad_quadratic_stats stats{};
  std::array<std::int64_t, 2> unmet_sampled_shape{};
};

int32_t cad_quadratic_nodes(const cad::CadGeometry& geometry, const CadQuadraticInput& input,
                            CadQuadraticMesh& result);

struct CadCubicMesh {
  std::vector<std::array<double, 3>> nodes;
  // IDs in nodes (not the vertex bank): two per edge 01, 02, 03, 12, 13, 23
  // in local endpoint order, then one per face opposite corners 0, 1, 2, 3.
  std::vector<std::array<int64_t, 16>> cell_nodes;
  std::array<std::int64_t, 2> unmet_sampled_shape{};
};

int32_t cad_cubic_nodes(const cad::CadGeometry& geometry, const CadQuadraticInput& input,
                        CadCubicMesh& result);

}  // namespace phx::mc
