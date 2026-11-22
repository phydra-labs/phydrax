//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Metric adaptation of a tetrahedral mesh on its live TetMesh state.
//
// The mesh is adapted toward unit metric edge length, the band
// [1/sqrt 2, sqrt 2], by edge splits, edge collapses, flips and vertex
// relocations. Every vertex is an exact dyadic combination of the vertices of
// the source P1 complex (its stencil, binary64 weights) and carries the
// log-Euclidean mean of the source tensors over its stencil. An edit is taken
// only when it strictly decreases the termination objective: the multiset of
// quantized long band violations, then of short ones (max-first, 2^-10
// quanta), then the cell quality; the objective takes finitely many values, so
// the loop ends without a pass cap. Validity is decided by the exact
// predicates of the TetMesh commits; kernel floats (metric_float.hpp) only
// measure, rank and screen.
//
// Commits are star-local through detail::LocalIncidence: interior splits,
// collapses, reconnections and relocations, boundary collapses (in place or
// with a moving survivor) and boundary edge swaps. A split on a constrained
// facet or segment goes through TetMesh::apply_edit_batch, which copies and
// audits the whole state: those splits are gathered into one disjoint batch
// per sweep, bisected on refusal; so does an interior collapse whose
// survivor moves, one at a time. A boundary vertex moves through
// TetMesh::apply_relocation_edit. Boundary relocations, survivor moves and
// deeper relocation searches are enabled in turn as the loop stalls.
#pragma once

#include <array>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "local_incidence.hpp"
#include "tet_mesh.hpp"

namespace phx::mc::metric {

// One term of a stencil: source point row and its binary64 weight.
struct StencilTerm {
  std::int32_t row = 0;
  double weight = 0.0;
};
using Stencil = std::vector<StencilTerm>;

// (1 - t) a + t b over ascending rows, zero terms dropped. False when a
// combined weight is not exactly a binary64 value (t is dyadic).
bool combine_stencils(const Stencil& a, const Stencil& b, double t, Stencil& combined);

// Lineage kinds of a target cell or vertex (EntityLineageKind values).
enum class AdaptLineage : std::int32_t {
  kPreserved = 0,
  kRefinedFrom = 1,
  kCoarsenedInto = 2,
  kCollapsedInto = 5,
  kSwappedFrom = 6,
  kRelocated = 7,
};

enum class AdaptStatus : std::int32_t {
  kComplete = 0,
  kStalled = 1,
  kNumericalStop = 2,
};

// The counters of one run, in TetraMetricEvidence order.
struct AdaptCounters {
  std::int64_t splits = 0;
  std::int64_t collapses = 0;
  std::int64_t flips23 = 0;
  std::int64_t flips32 = 0;
  std::int64_t ring_reconnections = 0;
  std::int64_t relocations = 0;
  std::int64_t rejected_topology = 0;
  std::int64_t rejected_protected = 0;
  std::int64_t rejected_uncertain = 0;
  std::int64_t rejected_invalid = 0;
};

// The exact P1 source and the starting state of the live mesh.
struct AdaptInput {
  // Source points (binary64, exact), their tensors and logarithms, rows in
  // source vertex order; source vertex IDs (target identity); the source
  // cells as four point rows each (relocation carriers are located in them).
  std::vector<double> source_points;
  std::vector<double> source_metrics;
  std::vector<double> source_logarithms;
  std::vector<std::uint64_t> source_vertex_ids;
  std::vector<std::int64_t> source_cells;
  // Per live mesh vertex (ascending native ID): its stencil (CSR over
  // source rows) and its target vertex ID.
  std::vector<std::uint64_t> vertex_ids;
  std::vector<std::int64_t> stencil_offsets;
  std::vector<std::int32_t> stencil_rows;
  std::vector<double> stencil_weights;
  std::vector<std::uint64_t> vertex_target_ids;
  // Per live mesh cell (ascending native ID): its block, target ID and
  // source-cell ancestry (CSR of source cell IDs) with its lineage kind.
  std::vector<std::uint64_t> cell_ids;
  std::vector<std::int32_t> cell_blocks;
  std::vector<std::uint64_t> cell_target_ids;
  std::vector<std::int64_t> ancestry_offsets;
  std::vector<std::uint64_t> ancestry_ids;
  std::vector<std::int32_t> ancestry_kinds;
  // Protected edges (pairs of native vertex IDs); never split or collapsed
  // and not measured.
  std::vector<std::uint64_t> protected_edges;
  // Fresh identifier allocators.
  std::uint64_t next_vertex_id = 0;
  std::uint64_t next_cell_id = 0;
  std::uint64_t next_target_vertex = 0;
  std::uint64_t next_target_cell = 0;
  // Completion: every measured edge length in [lower, sqrt 2].
  double lower = 0.0;
  bool topology_operations = true;
  bool relocation = true;
};

// The adapted state: per live vertex and cell, ascending native ID.
struct AdaptOutput {
  AdaptStatus status = AdaptStatus::kComplete;
  std::int64_t passes = 0;
  AdaptCounters counters;
  std::vector<std::uint64_t> vertex_ids;
  std::vector<std::uint64_t> vertex_target_ids;
  std::vector<std::int64_t> stencil_offsets;
  std::vector<std::int32_t> stencil_rows;
  std::vector<double> stencil_weights;
  std::vector<double> metrics;  // nine per vertex
  std::vector<std::uint64_t> cell_ids;
  std::vector<std::uint64_t> cell_target_ids;
  std::vector<std::int32_t> cell_blocks;
  std::vector<std::int64_t> ancestry_offsets;
  std::vector<std::uint64_t> ancestry_ids;
  std::vector<std::int32_t> ancestry_kinds;
  std::vector<std::int8_t> cell_relocated;
  // Vertex relations (source target ID, target ID, AdaptLineage) beyond the
  // preserved source vertices.
  std::vector<std::uint64_t> vertex_relation_sources;
  std::vector<std::uint64_t> vertex_relation_targets;
  std::vector<std::int32_t> vertex_relation_kinds;
  // Long edges with no exact split point (pairs of target vertex IDs).
  std::vector<std::uint64_t> unrepresentable_edges;
};

// Adapts `mesh` in place. The mesh must be unbound and nonperiodic, built from
// the source of `input`. Returns the TetEditStatus of a fatal commit failure
// (stale state, allocation, identifier range) or kApplied. May throw
// std::bad_alloc and std::invalid_argument for malformed input.
TetEditStatus adapt_metric(TetMesh& mesh, const AdaptInput& input, AdaptOutput& output);

}  // namespace phx::mc::metric
