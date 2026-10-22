//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Tetrahedral mesh improvement kernels on a TetMesh (see improve3d.cpp).
// Topology changes use one validated CavityEdit; relocation uses an exactly
// checked coordinate change. Original constrained strata and regions survive.
// Fixed constrained vertices and protecting-ball centers never move; conforming
// subdivision vertices may move exactly within their original source geometry.
// Refusal preserves coordinates and connectivity. require_improvement raises
// the replaced cells' smallest dihedral; tensor owners rank reconnect separately.
#pragma once
#include "counter_records.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

#include "bounded_memory.hpp"
#include "tet_mesh.hpp"

namespace phx::mc {

// 2-3 flip of the unconstrained interior face opposite v[slot] of t.
bool try_face_removal(TetMesh& mesh, int32_t t, int slot, bool require_improvement);

// Removes the unconstrained, non-segment interior edge (a, b): its ring of
// n tetrahedra (3 <= n <= 7) becomes 2n - 4 by the ring triangulation that
// maximizes the smallest dihedral angle (3-2 and 4-4 flips included).
// Construction repair instead maximizes exact determinant relative to the
// existing floating error enclosure, without changing the angle goal/margin.
// A weighted (regular) proposal is also refused when any child chart is at or
// below minimum_relative_determinant, the owning canonical-chart floor.
bool try_edge_removal(TetMesh& mesh, int32_t a, int32_t b, bool require_improvement,
                      std::optional<std::span<const double>> regular_weights = std::nullopt,
                      double radius_edge_bound = std::numeric_limits<double>::infinity(),
                      double max_weight_fraction = 0.25, bool construction_objective = false,
                      double minimum_relative_determinant = 0.0);

// Commits an externally ranked reconnection, without an isotropic objective.
// Vertex identities, protected segments, regions and oriented cavity boundary
// must survive; tensor metric owners rank proposals before this transaction.
EditStatus reconnect(TetMesh& mesh, std::span<const int32_t> removed,
                      std::span<const std::array<int32_t, 4>> proposed);

// Removes a connected unconstrained cluster of faces by coning its unchanged
// oriented boundary to an existing cavity vertex. The cavity remains bounded
// by the caller's explicit cell list and the mesh work budget.
bool try_multiface_removal(TetMesh& mesh, std::span<const int32_t> cavity,
                          int32_t apex, bool require_improvement);

// Splits the complete edge star at an exactly incident open-edge position.
// Returns distinct applied/refused/capacity/internal outcomes.
Insertion try_split_edge(TetMesh& mesh, int32_t a, int32_t b, const double* position,
                         double target_size, int32_t& inserted_vertex);

// Directional collapse. The full simplicial link condition, not just vertex
// neighbor counts, is required. Only an interior vertex may be retired.
bool try_collapse_edge(TetMesh& mesh, int32_t remove, int32_t keep,
                       bool require_improvement);

// Directional collapse of a curved-source surface vertex: the same volume
// link/orientation preparation, then the curved boundary transaction. With
// `inspection` the validated candidate is reported and nothing changes.
bool try_collapse_curved_edge(TetMesh& mesh, int32_t remove, int32_t keep,
                              bool require_improvement, const CurvedCollapseClaims* claims,
                              CurvedCollapseInspection* inspection);

// Certifies the protected star, source stratum and local quality before a
// coordinate/witness commit. Interior vertices and conforming facet/segment
// vertices may move within their source geometry; fixed vertices do not move.
bool try_relocate(TetMesh& mesh, int32_t vertex, const double* position, bool require_improvement);

// Prepares the union of sorted unique vertex stars, certifies positive cells,
// shape bounds and oriented source facets/segments, then admits all witnesses
// before the allocation-free coordinate commit. No intermediate move is published.
bool try_relocate_vertices(TetMesh& mesh, std::span<const int32_t> vertices,
                           const double* positions, double radius_edge_bound,
                           double minimum_dihedral_degrees);

// Removes an interior vertex by the valid collapse onto a neighbor that
// maximizes the smallest dihedral angle of the retriangulated star.
bool try_remove_vertex(TetMesh& mesh, int32_t vertex, bool require_improvement,
                       int32_t* accepted_neighbor);

struct ImproveOptions {
  double min_dihedral_degrees = 10.0;
  double minimum_relative_determinant = 0.0;
  int32_t max_passes = 8;
};

// Runs vertex-removal admission before ordered sliver-frontier reconnection;
// unmet evidence is assembled after the pass loop or its bounded refusal.
// Cells below the angle target, with uncertain construction, or at/below the
// canonical-chart determinant floor are candidates. A floor repair may insert
// one interior edge-star vertex whose children all exceed the floor.
// Accepted cavities enqueue their affected unmet children in the same pass,
// including angle failures. Coordinate proposals jointly solve positive
// dihedral-cosine violations for the owning cell's eligible vertices. A
// stationary residual may expand that same private proposal to its complete
// 3--7-cell edge ring (at most nine vertices); only full feasibility selects
// the expanded cavity. Complete incident unions retain single-star fallback.
// Trial state remains private; one exact source/region/protection-checked batch
// admission strictly raises the cavity minimum. Relocation never requeues an
// already visited ordinary cell within the requested pass.
// Complete local geometric negatives survive only edits outside their stars;
// resource/transaction refusals never certify a negative.
// counters receives PHX_MC_TET_MESH_IMPROVE_COUNTERS; mesh.unmet() the remaining cells.
int32_t improve_mesh(TetMesh& mesh, const ImproveOptions& options, TetMeshImproveCounters& counters);

struct ExudeOptions {
  double min_dihedral_degrees = 10.0;
  double max_weight_fraction = 0.1;  // weight / shortest incident edge squared
  double radius_edge_bound = 2.0;
  // Publishing cell-validity floor; no weighted reconnection creates a child
  // at or below it, and below-floor cells remain unmet VALIDITY records.
  double minimum_relative_determinant = 0.0;
  int32_t max_passes = 8;
};

// Bounded local sliver exudation. Only protected-stratum-free interior vertices
// receive weights; exact power predicates select regular bistellar and ring
// reconnections, which improve physical dihedral quality and recheck size.
// No globally regular connectivity is claimed. Output weights are scientific
// evidence, not coordinate displacement. Counters: trials, flips, weighted
// vertices, passes, remaining slivers, work. Unmet quality remains failure.
// Trials count every proposal decided, including those decided by reused
// weight-independent evaluation; work counts only evaluation performed.
int32_t exude_mesh(TetMesh& mesh, const ExudeOptions& options,
                   NativeVector<double>& weights, TetMeshExudeCounters& counters);

}  // namespace phx::mc
