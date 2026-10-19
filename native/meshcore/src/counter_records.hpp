//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#pragma once

#include <exception>

#include "phydrax_meshcore.h"

namespace phx::mc {

struct ExecutionCounters {
  uint64_t construction_work = 0;
  uint64_t source_queries = 0;
  uint64_t peak_cavity = 0;
  uint64_t work_refusals = 0;
  uint64_t query_refusals = 0;
  uint64_t cavity_refusals = 0;
  uint64_t external_work = 0;
  uint64_t external_queries = 0;
  uint64_t primitive_queries = 0;
  void publish(uint64_t* output) const noexcept {
    output[PHX_MC_EXECUTION_CONSTRUCTION_WORK] = construction_work;
    output[PHX_MC_EXECUTION_SOURCE_QUERIES] = source_queries;
    output[PHX_MC_EXECUTION_PEAK_CAVITY] = peak_cavity;
    output[PHX_MC_EXECUTION_WORK_REFUSALS] = work_refusals;
    output[PHX_MC_EXECUTION_QUERY_REFUSALS] = query_refusals;
    output[PHX_MC_EXECUTION_CAVITY_REFUSALS] = cavity_refusals;
    output[PHX_MC_EXECUTION_EXTERNAL_WORK] = external_work;
    output[PHX_MC_EXECUTION_EXTERNAL_QUERIES] = external_queries;
    output[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] = primitive_queries;
  }
  static ExecutionCounters from_slots(const uint64_t* input) noexcept {
    return {input[PHX_MC_EXECUTION_CONSTRUCTION_WORK], input[PHX_MC_EXECUTION_SOURCE_QUERIES], input[PHX_MC_EXECUTION_PEAK_CAVITY], input[PHX_MC_EXECUTION_WORK_REFUSALS], input[PHX_MC_EXECUTION_QUERY_REFUSALS], input[PHX_MC_EXECUTION_CAVITY_REFUSALS], input[PHX_MC_EXECUTION_EXTERNAL_WORK], input[PHX_MC_EXECUTION_EXTERNAL_QUERIES], input[PHX_MC_EXECUTION_PRIMITIVE_QUERIES]};
  }
};

struct ManagedMemory {
  uint64_t limit_bytes = 0;
  uint64_t live_bytes = 0;
  uint64_t peak_bytes = 0;
  uint64_t requested_bytes = 0;
  uint64_t allocations = 0;
  uint64_t refusals = 0;
  void publish(uint64_t* output) const noexcept {
    output[PHX_MC_MEMORY_LIMIT_BYTES] = limit_bytes;
    output[PHX_MC_MEMORY_LIVE_BYTES] = live_bytes;
    output[PHX_MC_MEMORY_PEAK_BYTES] = peak_bytes;
    output[PHX_MC_MEMORY_REQUESTED_BYTES] = requested_bytes;
    output[PHX_MC_MEMORY_ALLOCATIONS] = allocations;
    output[PHX_MC_MEMORY_REFUSALS] = refusals;
  }
  static ManagedMemory from_slots(const uint64_t* input) noexcept {
    return {input[PHX_MC_MEMORY_LIMIT_BYTES], input[PHX_MC_MEMORY_LIVE_BYTES], input[PHX_MC_MEMORY_PEAK_BYTES], input[PHX_MC_MEMORY_REQUESTED_BYTES], input[PHX_MC_MEMORY_ALLOCATIONS], input[PHX_MC_MEMORY_REFUSALS]};
  }
};

struct ExecutionMemory {
  uint64_t limit_bytes = 0;
  uint64_t live_bytes = 0;
  uint64_t peak_bytes = 0;
  uint64_t requested_bytes = 0;
  uint64_t allocations = 0;
  uint64_t refusals = 0;
  uint64_t external_live_upper_bytes = 0;
  uint64_t external_peak_upper_bytes = 0;
  void publish(uint64_t* output) const noexcept {
    output[PHX_MC_EXECUTION_MEMORY_LIMIT_BYTES] = limit_bytes;
    output[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] = live_bytes;
    output[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] = peak_bytes;
    output[PHX_MC_EXECUTION_MEMORY_REQUESTED_BYTES] = requested_bytes;
    output[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS] = allocations;
    output[PHX_MC_EXECUTION_MEMORY_REFUSALS] = refusals;
    output[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES] = external_live_upper_bytes;
    output[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES] = external_peak_upper_bytes;
  }
  static ExecutionMemory from_slots(const uint64_t* input) noexcept {
    return {input[PHX_MC_EXECUTION_MEMORY_LIMIT_BYTES], input[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES], input[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES], input[PHX_MC_EXECUTION_MEMORY_REQUESTED_BYTES], input[PHX_MC_EXECUTION_MEMORY_ALLOCATIONS], input[PHX_MC_EXECUTION_MEMORY_REFUSALS], input[PHX_MC_EXECUTION_MEMORY_EXTERNAL_LIVE_UPPER_BYTES], input[PHX_MC_EXECUTION_MEMORY_EXTERNAL_PEAK_UPPER_BYTES]};
  }
};

struct ExecutionBatchRefusal {
  uint64_t remaining_work = 0;
  uint64_t remaining_queries = 0;
  uint64_t remaining_cavity = 0;
  uint64_t remaining_bytes = 0;
  uint64_t consumed_work = 0;
  uint64_t consumed_queries = 0;
  uint64_t peak_cavity = 0;
  uint64_t peak_bytes = 0;
  void publish(uint64_t* output) const noexcept {
    output[PHX_MC_BATCH_REMAINING_WORK] = remaining_work;
    output[PHX_MC_BATCH_REMAINING_QUERIES] = remaining_queries;
    output[PHX_MC_BATCH_REMAINING_CAVITY] = remaining_cavity;
    output[PHX_MC_BATCH_REMAINING_BYTES] = remaining_bytes;
    output[PHX_MC_BATCH_CONSUMED_WORK] = consumed_work;
    output[PHX_MC_BATCH_CONSUMED_QUERIES] = consumed_queries;
    output[PHX_MC_BATCH_PEAK_CAVITY] = peak_cavity;
    output[PHX_MC_BATCH_PEAK_BYTES] = peak_bytes;
  }
};

struct PlanarCounters {
  uint64_t work_units = 0;
  uint64_t orient_requests = 0;
  uint64_t incircle_requests = 0;
  uint64_t power_requests = 0;
  uint64_t insertions = 0;
  uint64_t flips = 0;
  uint64_t peak_cavity_cells = 0;
  uint64_t work_refusals = 0;
  uint64_t cavity_refusals = 0;
  void publish(uint64_t* output) const noexcept {
    output[PHX_MC_PLANAR_WORK_UNITS] = work_units;
    output[PHX_MC_PLANAR_ORIENT_REQUESTS] = orient_requests;
    output[PHX_MC_PLANAR_INCIRCLE_REQUESTS] = incircle_requests;
    output[PHX_MC_PLANAR_POWER_REQUESTS] = power_requests;
    output[PHX_MC_PLANAR_INSERTIONS] = insertions;
    output[PHX_MC_PLANAR_FLIPS] = flips;
    output[PHX_MC_PLANAR_PEAK_CAVITY_CELLS] = peak_cavity_cells;
    output[PHX_MC_PLANAR_WORK_REFUSALS] = work_refusals;
    output[PHX_MC_PLANAR_CAVITY_REFUSALS] = cavity_refusals;
  }
  static PlanarCounters from_slots(const uint64_t* input) noexcept {
    return {input[PHX_MC_PLANAR_WORK_UNITS], input[PHX_MC_PLANAR_ORIENT_REQUESTS], input[PHX_MC_PLANAR_INCIRCLE_REQUESTS], input[PHX_MC_PLANAR_POWER_REQUESTS], input[PHX_MC_PLANAR_INSERTIONS], input[PHX_MC_PLANAR_FLIPS], input[PHX_MC_PLANAR_PEAK_CAVITY_CELLS], input[PHX_MC_PLANAR_WORK_REFUSALS], input[PHX_MC_PLANAR_CAVITY_REFUSALS]};
  }
};

struct GraphPartitionCounters {
  int64_t coarsening_levels = 0;
  int64_t coarsest_vertex_count = 0;
  int64_t matched_pairs = 0;
  int64_t bisection_trials = 0;
  int64_t refinement_passes = 0;
  int64_t moves_committed = 0;
  int64_t moves_rolled_back = 0;
  int64_t balance_moves = 0;
  int64_t nonempty_repairs = 0;
  int64_t adjacency_visits = 0;
  int64_t candidate_evaluations = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_GRAPH_PARTITION_COARSENING_LEVELS] = coarsening_levels;
    output[PHX_MC_GRAPH_PARTITION_COARSEST_VERTEX_COUNT] = coarsest_vertex_count;
    output[PHX_MC_GRAPH_PARTITION_MATCHED_PAIRS] = matched_pairs;
    output[PHX_MC_GRAPH_PARTITION_BISECTION_TRIALS] = bisection_trials;
    output[PHX_MC_GRAPH_PARTITION_REFINEMENT_PASSES] = refinement_passes;
    output[PHX_MC_GRAPH_PARTITION_MOVES_COMMITTED] = moves_committed;
    output[PHX_MC_GRAPH_PARTITION_MOVES_ROLLED_BACK] = moves_rolled_back;
    output[PHX_MC_GRAPH_PARTITION_BALANCE_MOVES] = balance_moves;
    output[PHX_MC_GRAPH_PARTITION_NONEMPTY_REPAIRS] = nonempty_repairs;
    output[PHX_MC_GRAPH_PARTITION_ADJACENCY_VISITS] = adjacency_visits;
    output[PHX_MC_GRAPH_PARTITION_CANDIDATE_EVALUATIONS] = candidate_evaluations;
  }
};

struct SurfaceCounters {
  int64_t inserted = 0;
  int64_t refused = 0;
  int64_t flips = 0;
  int64_t walk_steps = 0;
  int64_t work = 0;
  int64_t exhausted = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_SURFACE_INSERTED] = inserted;
    output[PHX_MC_SURFACE_REFUSED] = refused;
    output[PHX_MC_SURFACE_FLIPS] = flips;
    output[PHX_MC_SURFACE_WALK_STEPS] = walk_steps;
    output[PHX_MC_SURFACE_WORK] = work;
    output[PHX_MC_SURFACE_EXHAUSTED] = exhausted;
  }
};

struct TetMeshRefineCounters {
  int64_t circumcenter_insertions = 0;
  int64_t subfacet_splits = 0;
  int64_t subsegment_splits = 0;
  int64_t encroachment_deferrals = 0;
  int64_t refused_insertions = 0;
  int64_t queue_pops = 0;
  int64_t unmet_cells = 0;
  int64_t work_units = 0;
  int64_t largest_cavity = 0;
  int64_t vertices = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_TET_MESH_REFINE_CIRCUMCENTER_INSERTIONS] = circumcenter_insertions;
    output[PHX_MC_TET_MESH_REFINE_SUBFACET_SPLITS] = subfacet_splits;
    output[PHX_MC_TET_MESH_REFINE_SUBSEGMENT_SPLITS] = subsegment_splits;
    output[PHX_MC_TET_MESH_REFINE_ENCROACHMENT_DEFERRALS] = encroachment_deferrals;
    output[PHX_MC_TET_MESH_REFINE_REFUSED_INSERTIONS] = refused_insertions;
    output[PHX_MC_TET_MESH_REFINE_QUEUE_POPS] = queue_pops;
    output[PHX_MC_TET_MESH_REFINE_UNMET_CELLS] = unmet_cells;
    output[PHX_MC_TET_MESH_REFINE_WORK_UNITS] = work_units;
    output[PHX_MC_TET_MESH_REFINE_LARGEST_CAVITY] = largest_cavity;
    output[PHX_MC_TET_MESH_REFINE_VERTICES] = vertices;
  }
};

struct TetMeshImproveCounters {
  int64_t face_removals = 0;
  int64_t edge_removals = 0;
  int64_t relocations = 0;
  int64_t vertex_removals = 0;
  int64_t passes = 0;
  int64_t attempts = 0;
  int64_t remaining_slivers = 0;
  int64_t work_units = 0;
  int64_t vertex_insertions = 0;
  int64_t multiface_removals = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_TET_MESH_IMPROVE_FACE_REMOVALS] = face_removals;
    output[PHX_MC_TET_MESH_IMPROVE_EDGE_REMOVALS] = edge_removals;
    output[PHX_MC_TET_MESH_IMPROVE_RELOCATIONS] = relocations;
    output[PHX_MC_TET_MESH_IMPROVE_VERTEX_REMOVALS] = vertex_removals;
    output[PHX_MC_TET_MESH_IMPROVE_PASSES] = passes;
    output[PHX_MC_TET_MESH_IMPROVE_ATTEMPTS] = attempts;
    output[PHX_MC_TET_MESH_IMPROVE_REMAINING_SLIVERS] = remaining_slivers;
    output[PHX_MC_TET_MESH_IMPROVE_WORK_UNITS] = work_units;
    output[PHX_MC_TET_MESH_IMPROVE_VERTEX_INSERTIONS] = vertex_insertions;
    output[PHX_MC_TET_MESH_IMPROVE_MULTIFACE_REMOVALS] = multiface_removals;
  }
};

struct TetMeshExudeCounters {
  int64_t trials = 0;
  int64_t flips = 0;
  int64_t weighted_vertices = 0;
  int64_t passes = 0;
  int64_t remaining_slivers = 0;
  int64_t work_units = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_TET_MESH_EXUDE_TRIALS] = trials;
    output[PHX_MC_TET_MESH_EXUDE_FLIPS] = flips;
    output[PHX_MC_TET_MESH_EXUDE_WEIGHTED_VERTICES] = weighted_vertices;
    output[PHX_MC_TET_MESH_EXUDE_PASSES] = passes;
    output[PHX_MC_TET_MESH_EXUDE_REMAINING_SLIVERS] = remaining_slivers;
    output[PHX_MC_TET_MESH_EXUDE_WORK_UNITS] = work_units;
  }
};

struct PlcCounters {
  int64_t rounds = 0;
  int64_t input_triangles = 0;
  int64_t facet_groups = 0;
  int64_t plc_edges = 0;
  int64_t protected_vertices = 0;
  int64_t segment_steiner_points = 0;
  int64_t facet_steiner_points = 0;
  int64_t cavities = 0;
  int64_t largest_cavity = 0;
  int64_t gift_wrap_candidates = 0;
  int64_t contact_tests = 0;
  int64_t work_units = 0;
  int64_t peak_retained_bytes = 0;
  int64_t interior_steiner_points = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_PLC3D_ROUNDS] = rounds;
    output[PHX_MC_PLC3D_INPUT_TRIANGLES] = input_triangles;
    output[PHX_MC_PLC3D_FACET_GROUPS] = facet_groups;
    output[PHX_MC_PLC3D_PLC_EDGES] = plc_edges;
    output[PHX_MC_PLC3D_PROTECTED_VERTICES] = protected_vertices;
    output[PHX_MC_PLC3D_SEGMENT_STEINER_POINTS] = segment_steiner_points;
    output[PHX_MC_PLC3D_FACET_STEINER_POINTS] = facet_steiner_points;
    output[PHX_MC_PLC3D_CAVITIES] = cavities;
    output[PHX_MC_PLC3D_LARGEST_CAVITY] = largest_cavity;
    output[PHX_MC_PLC3D_GIFT_WRAP_CANDIDATES] = gift_wrap_candidates;
    output[PHX_MC_PLC3D_CONTACT_TESTS] = contact_tests;
    output[PHX_MC_PLC3D_WORK_UNITS] = work_units;
    output[PHX_MC_PLC3D_PEAK_RETAINED_BYTES] = peak_retained_bytes;
    output[PHX_MC_PLC3D_INTERIOR_STEINER_POINTS] = interior_steiner_points;
  }
};

struct PlcFailureRecord {
  int64_t reason = 0;
  int64_t first_kind = 0;
  int64_t first_id = 0;
  int64_t second_kind = 0;
  int64_t second_id = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_PLC3D_FAILURE_REASON] = reason;
    output[PHX_MC_PLC3D_FAILURE_FIRST_KIND] = first_kind;
    output[PHX_MC_PLC3D_FAILURE_FIRST_ID] = first_id;
    output[PHX_MC_PLC3D_FAILURE_SECOND_KIND] = second_kind;
    output[PHX_MC_PLC3D_FAILURE_SECOND_ID] = second_id;
  }
};

struct PowerCounters {
  int64_t clips = 0;
  int64_t pieces = 0;
  int64_t empty_clips = 0;
  int64_t walk_steps = 0;
  int64_t merged_links = 0;
  int64_t unmerged_fragment_groups = 0;
  int64_t removed_collinear_vertices = 0;
  int64_t cells = 0;
  int64_t faces = 0;
  int64_t vertices = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_POWER_CELLS_CLIPS] = clips;
    output[PHX_MC_POWER_CELLS_PIECES] = pieces;
    output[PHX_MC_POWER_CELLS_EMPTY_CLIPS] = empty_clips;
    output[PHX_MC_POWER_CELLS_WALK_STEPS] = walk_steps;
    output[PHX_MC_POWER_CELLS_MERGED_LINKS] = merged_links;
    output[PHX_MC_POWER_CELLS_UNMERGED_FRAGMENT_GROUPS] = unmerged_fragment_groups;
    output[PHX_MC_POWER_CELLS_REMOVED_COLLINEAR_VERTICES] = removed_collinear_vertices;
    output[PHX_MC_POWER_CELLS_CELLS] = cells;
    output[PHX_MC_POWER_CELLS_FACES] = faces;
    output[PHX_MC_POWER_CELLS_VERTICES] = vertices;
  }
};

struct PowerFailure {
  int64_t reason = 0;
  int64_t site = 0;
  int64_t tet = 0;
  int64_t clip_status = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_POWER_CELLS_FAILURE_REASON] = reason;
    output[PHX_MC_POWER_CELLS_FAILURE_SITE] = site;
    output[PHX_MC_POWER_CELLS_FAILURE_TET] = tet;
    output[PHX_MC_POWER_CELLS_FAILURE_CLIP_STATUS] = clip_status;
  }
};

struct ArrangementCounters {
  int64_t pair_count = 0;
  int64_t contact_pairs = 0;
  int64_t coplanar_pairs = 0;
  int64_t coincident_vertices = 0;
  int64_t edge_plane_vertices = 0;
  int64_t edge_edge_vertices = 0;
  int64_t triple_plane_vertices = 0;
  int64_t split_faces = 0;
  int64_t fragments = 0;
  int64_t filtered_predicates = 0;
  int64_t exact_predicates = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_ARRANGEMENT_PAIR_COUNT] = pair_count;
    output[PHX_MC_ARRANGEMENT_CONTACT_PAIRS] = contact_pairs;
    output[PHX_MC_ARRANGEMENT_COPLANAR_PAIRS] = coplanar_pairs;
    output[PHX_MC_ARRANGEMENT_COINCIDENT_VERTICES] = coincident_vertices;
    output[PHX_MC_ARRANGEMENT_EDGE_PLANE_VERTICES] = edge_plane_vertices;
    output[PHX_MC_ARRANGEMENT_EDGE_EDGE_VERTICES] = edge_edge_vertices;
    output[PHX_MC_ARRANGEMENT_TRIPLE_PLANE_VERTICES] = triple_plane_vertices;
    output[PHX_MC_ARRANGEMENT_SPLIT_FACES] = split_faces;
    output[PHX_MC_ARRANGEMENT_FRAGMENTS] = fragments;
    output[PHX_MC_ARRANGEMENT_FILTERED_PREDICATES] = filtered_predicates;
    output[PHX_MC_ARRANGEMENT_EXACT_PREDICATES] = exact_predicates;
  }
};

struct LevelSetCounters {
  int64_t vertices = 0;
  int64_t cells = 0;
  int64_t inserted = 0;
  int64_t zero_vertices = 0;
  int64_t work_units = 0;
  int64_t failed_edge = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_LEVEL_SET_VERTICES] = vertices;
    output[PHX_MC_LEVEL_SET_CELLS] = cells;
    output[PHX_MC_LEVEL_SET_INSERTED] = inserted;
    output[PHX_MC_LEVEL_SET_ZERO_VERTICES] = zero_vertices;
    output[PHX_MC_LEVEL_SET_WORK_UNITS] = work_units;
    output[PHX_MC_LEVEL_SET_FAILED_EDGE] = failed_edge;
  }
};

struct RestrictedDualCounters {
  int64_t work_units = 0;
  int64_t interior_facets = 0;
  int64_t hull_facets = 0;
  int64_t degenerate_facets = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_RESTRICTED_DUAL_WORK_UNITS] = work_units;
    output[PHX_MC_RESTRICTED_DUAL_INTERIOR_FACETS] = interior_facets;
    output[PHX_MC_RESTRICTED_DUAL_HULL_FACETS] = hull_facets;
    output[PHX_MC_RESTRICTED_DUAL_DEGENERATE_FACETS] = degenerate_facets;
  }
};

struct PeriodicEvidence {
  double rounds = 0;
  double margin = 0;
  double images = 0;
  double uncertified = 0;
  double required_margin = 0;
  double finite_cells = 0;
  double exact_evaluations = 0;
  double perturbed_decisions = 0;
  double exhausted_limit = 0;
  double duplicate_point = 0;
  void publish(double* output) const noexcept {
    output[PHX_MC_PERIODIC_ROUNDS] = rounds;
    output[PHX_MC_PERIODIC_MARGIN] = margin;
    output[PHX_MC_PERIODIC_IMAGES] = images;
    output[PHX_MC_PERIODIC_UNCERTIFIED] = uncertified;
    output[PHX_MC_PERIODIC_REQUIRED_MARGIN] = required_margin;
    output[PHX_MC_PERIODIC_FINITE_CELLS] = finite_cells;
    output[PHX_MC_PERIODIC_EXACT_EVALUATIONS] = exact_evaluations;
    output[PHX_MC_PERIODIC_PERTURBED_DECISIONS] = perturbed_decisions;
    output[PHX_MC_PERIODIC_EXHAUSTED_LIMIT] = exhausted_limit;
    output[PHX_MC_PERIODIC_DUPLICATE_POINT] = duplicate_point;
  }
};

struct TriangulationStatistics {
  int64_t vertex_ids = 0;
  int64_t live_vertices = 0;
  int64_t finite_cells = 0;
  int64_t ghost_cells = 0;
  int64_t cell_slots = 0;
  int64_t free_slots = 0;
  int64_t constrained_facets = 0;
  int64_t work = 0;
  int64_t committed = 0;
  int64_t refused = 0;
  int64_t largest_cavity = 0;
  int64_t retained_bytes = 0;
  int64_t peak_bytes = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_TRIANGULATION_3D_VERTEX_IDS] = vertex_ids;
    output[PHX_MC_TRIANGULATION_3D_LIVE_VERTICES] = live_vertices;
    output[PHX_MC_TRIANGULATION_3D_FINITE_CELLS] = finite_cells;
    output[PHX_MC_TRIANGULATION_3D_GHOST_CELLS] = ghost_cells;
    output[PHX_MC_TRIANGULATION_3D_CELL_SLOTS] = cell_slots;
    output[PHX_MC_TRIANGULATION_3D_FREE_SLOTS] = free_slots;
    output[PHX_MC_TRIANGULATION_3D_CONSTRAINED_FACETS] = constrained_facets;
    output[PHX_MC_TRIANGULATION_3D_WORK] = work;
    output[PHX_MC_TRIANGULATION_3D_COMMITTED] = committed;
    output[PHX_MC_TRIANGULATION_3D_REFUSED] = refused;
    output[PHX_MC_TRIANGULATION_3D_LARGEST_CAVITY] = largest_cavity;
    output[PHX_MC_TRIANGULATION_3D_RETAINED_BYTES] = retained_bytes;
    output[PHX_MC_TRIANGULATION_3D_PEAK_BYTES] = peak_bytes;
  }
};

struct SurfaceEnvelopeCounts {
  int64_t surface_vertices = 0;
  int64_t surface_triangles = 0;
  int64_t grid_samples = 0;
  int64_t work_units = 0;
  int64_t source_samples = 0;
  int64_t volume_vertices = 0;
  int64_t tetrahedra = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_SURFACE_ENVELOPE_SURFACE_VERTICES] = surface_vertices;
    output[PHX_MC_SURFACE_ENVELOPE_SURFACE_TRIANGLES] = surface_triangles;
    output[PHX_MC_SURFACE_ENVELOPE_GRID_SAMPLES] = grid_samples;
    output[PHX_MC_SURFACE_ENVELOPE_WORK_UNITS] = work_units;
    output[PHX_MC_SURFACE_ENVELOPE_SOURCE_SAMPLES] = source_samples;
    output[PHX_MC_SURFACE_ENVELOPE_VOLUME_VERTICES] = volume_vertices;
    output[PHX_MC_SURFACE_ENVELOPE_TETRAHEDRA] = tetrahedra;
  }
};

struct EdgeInsertionCounts {
  int64_t removed = 0;
  int64_t proposed = 0;
  int64_t kind = 0;
  int64_t generation = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_EDGE_INSERTION_REMOVED] = removed;
    output[PHX_MC_EDGE_INSERTION_PROPOSED] = proposed;
    output[PHX_MC_EDGE_INSERTION_KIND] = kind;
    output[PHX_MC_EDGE_INSERTION_GENERATION] = generation;
  }
};

struct ExecutionResources {
  uint64_t work = 0;
  uint64_t queries = 0;
  uint64_t cavity = 0;
  uint64_t bytes = 0;
  void publish(uint64_t* output) const noexcept {
    output[PHX_MC_RESOURCE_WORK] = work;
    output[PHX_MC_RESOURCE_QUERIES] = queries;
    output[PHX_MC_RESOURCE_CAVITY] = cavity;
    output[PHX_MC_RESOURCE_BYTES] = bytes;
  }
  static ExecutionResources from_slots(const uint64_t* input) noexcept {
    return {input[PHX_MC_RESOURCE_WORK], input[PHX_MC_RESOURCE_QUERIES], input[PHX_MC_RESOURCE_CAVITY], input[PHX_MC_RESOURCE_BYTES]};
  }
};

struct PlcSourceRefusal {
  double certified_deviation = 0;
  double declared_bound = 0;
  void publish(double* output) const noexcept {
    output[PHX_MC_PLC3D_SOURCE_REFUSAL_CERTIFIED_DEVIATION] = certified_deviation;
    output[PHX_MC_PLC3D_SOURCE_REFUSAL_DECLARED_BOUND] = declared_bound;
  }
};

struct TetMeshCounts {
  int64_t vertices = 0;
  int64_t tetrahedra = 0;
  int64_t faces = 0;
  int64_t segments = 0;
  int64_t unmet_records = 0;
  int64_t work_units = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_TET_MESH_COUNT_VERTICES] = vertices;
    output[PHX_MC_TET_MESH_COUNT_TETRAHEDRA] = tetrahedra;
    output[PHX_MC_TET_MESH_COUNT_FACES] = faces;
    output[PHX_MC_TET_MESH_COUNT_SEGMENTS] = segments;
    output[PHX_MC_TET_MESH_COUNT_UNMET_RECORDS] = unmet_records;
    output[PHX_MC_TET_MESH_COUNT_WORK_UNITS] = work_units;
  }
};

struct ArrangementClassificationVisits {
  int64_t components = 0;
  int64_t operands = 0;
  int64_t pair_scans = 0;
  int64_t matrix_entries = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_ARRANGEMENT_CLASSIFICATION_COMPONENTS] = components;
    output[PHX_MC_ARRANGEMENT_CLASSIFICATION_OPERANDS] = operands;
    output[PHX_MC_ARRANGEMENT_CLASSIFICATION_PAIR_SCANS] = pair_scans;
    output[PHX_MC_ARRANGEMENT_CLASSIFICATION_MATRIX_ENTRIES] = matrix_entries;
  }
};

struct PlcPhaseValues {
  int64_t validation = 0;
  int64_t preparation = 0;
  int64_t boundary_recovery = 0;
  int64_t classification = 0;
  int64_t publication = 0;
  void publish(int64_t* output) const noexcept {
    output[PHX_MC_PLC3D_PHASE_VALIDATION] = validation;
    output[PHX_MC_PLC3D_PHASE_PREPARATION] = preparation;
    output[PHX_MC_PLC3D_PHASE_BOUNDARY_RECOVERY] = boundary_recovery;
    output[PHX_MC_PLC3D_PHASE_CLASSIFICATION] = classification;
    output[PHX_MC_PLC3D_PHASE_PUBLICATION] = publication;
  }
  int64_t& for_phase(int32_t phase) noexcept {
    switch (phase) {
      case PHX_MC_PLC3D_PHASE_VALIDATION: return validation;
      case PHX_MC_PLC3D_PHASE_PREPARATION: return preparation;
      case PHX_MC_PLC3D_PHASE_BOUNDARY_RECOVERY: return boundary_recovery;
      case PHX_MC_PLC3D_PHASE_CLASSIFICATION: return classification;
      case PHX_MC_PLC3D_PHASE_PUBLICATION: return publication;
    }
    std::terminate();
  }
};

// Construct only when the old boundary admitted publication. Unwinding publishes
// the same named state as an ordinary return, without allocation or throwing.
template <class Record, class Scalar>
struct CounterPublisher {
  const Record& record;
  Scalar* output;
  bool enabled = true;
  ~CounterPublisher() noexcept {
    if (enabled && output != nullptr) record.publish(output);
  }
};

}  // namespace phx::mc
