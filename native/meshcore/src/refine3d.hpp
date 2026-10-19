//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained Delaunay refinement of a TetMesh (see refine3d.cpp).
#pragma once
#include "counter_records.hpp"

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
// declared source deviation. Subfacet preparation tries the source-admitted
// circumcenter, then the represented-plane exact circumcenter, in that order.
// Both transactions must retire both sides of the queued facet. Otherwise its
// longest edge uses an exact split or the bounded carrier of that edge's source
// row; unrelated facet walks and arbitrary angle-shrinking star splits are refused.
int32_t refine_mesh(TetMesh& mesh, const RefineOptions& options, TetMeshRefineCounters& counters);

}  // namespace phx::mc
