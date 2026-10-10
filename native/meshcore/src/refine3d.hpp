//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained Delaunay refinement of a TetMesh (see refine3d.cpp).
#pragma once

#include <cstdint>

#include "tet_mesh.hpp"

namespace phx::mc {

struct RefineOptions {
  double radius_edge_bound = 2.0;
  int64_t max_insertions = 0;
};

// Refines until every tetrahedron meets the radius-edge and size criteria,
// the insertion budget is spent, or a limit stops it.  counters receives
// PHX_MC_TET_MESH_REFINE_COUNTERS values; mesh.unmet() the unmet cells and
// mesh.source_refusals() the ancestry-backed splits refused by their
// declared source deviation. Subfacet circumcenters must lie exactly on the
// represented plane and their prepared Delaunay patch must retire both sides
// of the requested facet. Otherwise refinement falls back to its longest
// exact edge split, then to the bounded carrier of that edge's source row,
// not an arbitrary interior star split that can shrink source angles.
int32_t refine_mesh(TetMesh& mesh, const RefineOptions& options, int64_t* counters);

}  // namespace phx::mc
