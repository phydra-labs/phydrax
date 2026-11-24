//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Hex-dominant meshing of a closed oriented triangle surface by a balanced
// octree core, a snapped projection layer and optional hexahedral boundary
// layers (see phx_mc_hex_dominant_build in phydrax_meshcore.h). Construction in
// floating point with exact predicates for inside/outside; certification after
// construction with local repair.
#pragma once

#include <cstdint>
#include <vector>

namespace phx::mc::hex_dominant {

inline constexpr int kStatCount = 28;

enum CellKind : uint8_t { kHexahedron = 0, kPolyhedron = 1, kPrism = 2 };
enum CellRole : uint8_t { kCore = 0, kProjection = 1, kLayer = 2 };
enum VertexKind : uint8_t { kInterior = 0, kPatch = 1, kFeature = 2, kCorner = 3, kLayerVertex = 4 };

struct Input {
  int64_t vertex_count;
  const double* vertices;
  int64_t triangle_count;
  const int32_t* triangles;
  const int32_t* triangle_facets;
  double core_size;
  double boundary_size;
  double feature_angle_degrees;
  int32_t layer_count;
  const double* layer_thicknesses;
  double layer_minimum_thickness_fraction;
  int32_t prove;
  uint32_t parallelism;
};

// The face-based mixed mesh: faces oriented out of their owner; boundary faces
// have neighbor -1 and the facet of their nearest input triangle. Hexahedra
// list their eight vertices in the reference order (bottom counterclockwise
// about the direction to the top); polyhedra list none.
struct Mesh {
  std::vector<double> points;
  std::vector<int64_t> face_offsets{0};
  std::vector<int32_t> face_vertices;
  std::vector<int32_t> face_owner;
  std::vector<int32_t> face_neighbor;
  std::vector<int32_t> face_facets;
  std::vector<uint8_t> cell_kinds;
  std::vector<uint8_t> cell_roles;
  std::vector<int64_t> cell_vertex_offsets{0};
  std::vector<int32_t> cell_vertices;
  std::vector<int32_t> vertex_triangles;
  std::vector<uint8_t> vertex_kinds;
  double stats[kStatCount] = {};
};

// Validated input. Returns PHX_MC_OK when the mesh is constructed (and, with
// prove, certified), PHX_MC_UNRESOLVED when a cell failed certification after
// repair (the mesh then holds the last construction), PHX_MC_INVALID_INPUT
// for a surface that is not a closed consistently oriented manifold enclosing
// positive volume, PHX_MC_DEGENERATE_INPUT for a zero-area triangle and
// PHX_MC_INDEX_RANGE_ERROR when the lattice or the mesh exceeds the native
// index ranges.
int32_t build(const Input& input, Mesh& mesh);

// Check the final face banks in their original construction order.
int64_t uncertified_cells(const Mesh& mesh);

}  // namespace phx::mc::hex_dominant
