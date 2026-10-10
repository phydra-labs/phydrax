//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Piecewise-linear complex (PLC) validation, protected boundary recovery and
// region classification in 3D.
//
// Input: points, oriented planar polygons grouped into facets (each facet
// declares the region on the positive side of its right-hand normal and on
// the negative side, -1 for void), explicit segments (internal curves) and
// optional region seeds.  Validation is exact: polygons are exactly planar
// and simple, distinct constraint entities meet only at shared vertices and
// edges (exact triangle/segment contacts), and the oriented facet chain of
// every region is closed (every region edge cancels).  Internal sheets are
// facets with the same region on both sides and are legal by incidence.
//
// Recovery (see plc3d.cpp): the exact Delaunay tetrahedralization of the
// vertices and eight enclosing corners; subdividable PLC edges are recovered
// by exactly representable on-segment splits, with numerical concentric-shell
// protection around acute vertices. When no exact split exists and the edge's
// declared source deviation (the smallest of its segment and incident facet
// tolerances) is positive, the split is the correctly rounded carrier of an
// exact source-line witness, accepted only when its certified deviation does
// not exceed that bound; facet Steiner points follow the same rule on their
// input triangle. Zero tolerance keeps the exact-only construction.
// Immutable edges are recovered instead by constrained cavity reconnection
// and permitted off-PLC interior insertion. An edge between coplanar facets
// that neither reconnects in its own cavity nor splits joins the facet
// recovery of that planar component.
// Missing subfacets are recovered per exactly planar component by
// exact-contact gift wrapping of the two cavity sides, enlarged first behind
// the faces that trap a side, and committed through one validated CavityEdit.
// A fixed boundary never receives an edge/facet Steiner point.
// Regions are flooded from both sides of every constrained facet (by its
// declared incidence), from the enclosing corners (void) and from seeds;
// any disagreement is refused with the conflicting regions as evidence.
#pragma once

#include <cstdint>
#include <limits>
#include <span>

#include "bounded_memory.hpp"
#include "source_construction.hpp"

namespace phx::mc {

enum class BoundaryPolicy : int32_t { kFixed = 0, kConforming = 1 };

// Why recovery refused (phx_mc_plc3d_failure reason codes).
enum PlcFailureReason : int32_t {
  kPlcNoFailure = 0,
  kPlcDuplicatePoint = 1,
  kPlcInvalidPolygon = 2,      // degenerate, nonplanar or non-simple polygon
  kPlcInvalidFacet = 3,        // unused facet or void on both sides
  kPlcIntersecting = 4,        // entities meet outside shared vertices/edges
  kPlcOpenBoundary = 5,        // a region's oriented facet chain is not closed
  kPlcInconsistentRegions = 6, // two region labels meet without a constrained facet
  kPlcRegionLeak = 7,          // a region reaches the unbounded void
  kPlcFixedSegment = 8,        // fixed boundary: a PLC edge needs splitting
  kPlcFixedFacet = 9,          // fixed boundary: a facet cavity has no fill
  kPlcNonrepresentable = 10,   // no exactly representable Steiner point (zero tolerance)
  kPlcVertexBudget = 11,
  kPlcTetrahedronBudget = 12,
  kPlcWorkBudget = 13,
  kPlcInvalidSeed = 14,        // a seed on a constrained facet or outside the hull
  kPlcOutsideDomain = 15,      // a segment or free point outside every region
  kPlcScratchByteBudget = 16,
  kPlcSourceDeviation = 17,    // the certified source deviation exceeds the declared bound
};

enum PlcEntityKind : int32_t {
  kPlcNoEntity = 0,
  kPlcPoint = 1,
  kPlcPolygon = 2,
  kPlcFacet = 3,
  kPlcEdge = 4,
  kPlcRegion = 5,
  kPlcSeed = 6,
};

enum PlcCounter : int32_t {
  kPlcRounds = 0,
  kPlcInputTriangles,
  kPlcFacetGroups,
  kPlcEdges,
  kPlcProtectedVertices,
  kPlcSegmentSteiner,
  kPlcFacetSteiner,
  kPlcCavities,
  kPlcLargestCavity,
  kPlcCandidates,
  kPlcContactTests,
  kPlcWork,
  kPlcPeakBytes,
  kPlcInteriorSteiner,
  kPlcCounterCount,
};

// Optional execution measurements; never part of work counters or identity.
enum PlcPhase : int32_t {
  kPlcValidationPhase,
  kPlcPreparationPhase,
  kPlcBoundaryRecoveryPhase,
  kPlcClassificationPhase,
  kPlcPublicationPhase,
  kPlcPhaseCount,
};

struct PlcInput {
  int64_t point_count = 0;
  const double* points = nullptr;  // point_count x 3
  int64_t polygon_count = 0;
  const int64_t* polygon_offsets = nullptr;   // polygon_count + 1
  const int32_t* polygon_vertices = nullptr;  // loop vertex ids
  const int32_t* polygon_facets = nullptr;    // facet of each polygon
  int64_t facet_count = 0;
  const int32_t* facet_regions = nullptr;  // facet_count x 2: positive, negative side
  int64_t segment_count = 0;
  const int32_t* segments = nullptr;  // segment_count x 2
  int64_t seed_count = 0;
  const double* seeds = nullptr;          // seed_count x 3
  const int32_t* seed_regions = nullptr;  // region or -1 (void)
  BoundaryPolicy policy = BoundaryPolicy::kConforming;
  // Declared nonnegative source deviation bounds (NULL: zero, exact-only).
  const double* facet_tolerances = nullptr;    // facet_count
  const double* segment_tolerances = nullptr;  // segment_count
  int64_t max_vertices = 0;    // output vertices, input included
  int64_t max_tetrahedra = 0;  // finite tetrahedra of the construction
  int64_t work_limit = 0;
  bool measure_phases = false;
  std::size_t max_scratch_bytes = std::numeric_limits<std::size_t>::max();
};

// Failure evidence: the reason and up to two entities (kind, id) involved.
struct PlcFailure {
  int32_t reason = kPlcNoFailure;
  int32_t first_kind = kPlcNoEntity;
  int64_t first = -1;
  int32_t second_kind = kPlcNoEntity;
  int64_t second = -1;
};

// Recovered complex: input points keep their indices, Steiner points follow.
// Domain tetrahedra only (region >= 0), positively oriented; every constrained
// subfacet oriented like its source facet; subsegments of every PLC edge.
struct PlcRecovery {
  explicit PlcRecovery(MemoryOwner owner = scratch_memory_owner())
      : memory_owner(std::move(owner)),
        points(NativeAllocator<double>(memory_owner)),
        tets(NativeAllocator<int32_t>(memory_owner)),
        tet_regions(NativeAllocator<int32_t>(memory_owner)),
        faces(NativeAllocator<int32_t>(memory_owner)),
        face_sources(NativeAllocator<int32_t>(memory_owner)),
        segments(NativeAllocator<int32_t>(memory_owner)),
        segment_sources(NativeAllocator<int32_t>(memory_owner)),
        protection(NativeAllocator<double>(memory_owner)),
        plc_edges(NativeAllocator<int32_t>(memory_owner)),
        vertex_dimension(NativeAllocator<int8_t>(memory_owner)),
        input_triangles(NativeAllocator<int32_t>(memory_owner)),
        input_polygons(NativeAllocator<int32_t>(memory_owner)),
        witnesses(NativeAllocator<SourceWitness>(memory_owner)) {}
  MemoryOwner memory_owner;
  NativeVector<double> points;
  NativeVector<int32_t> tets;
  NativeVector<int32_t> tet_regions;
  NativeVector<int32_t> faces;
  NativeVector<int32_t> face_sources;
  NativeVector<int32_t> segments;
  NativeVector<int32_t> segment_sources;
  NativeVector<double> protection;  // per output point, 0 when unprotected
  NativeVector<int32_t> plc_edges;  // explicit segments, then facet-group boundary edges
  NativeVector<int8_t> vertex_dimension;  // 0 input, 1 PLC edge, 2 facet, 3 interior Steiner
  NativeVector<int32_t> input_triangles;  // exact triangulation of the input polygons
  NativeVector<int32_t> input_polygons;   // polygon of each input triangle
  // Per output point: input vertices and interior Steiner points name no
  // source; an edge (facet) Steiner point names its PLC edge with t from the
  // edge's first endpoint (its input triangle with the weights of the second
  // and third corners) and carries zero deviation exactly when it lies on it.
  NativeVector<SourceWitness> witnesses;
  // A kPlcSourceDeviation refusal: certified deviation and declared bound.
  double source_refusal[2] = {0.0, 0.0};
  bool memory_measured = false;
  std::array<uint64_t, 6> memory_snapshot{};
  int64_t counters[kPlcCounterCount] = {};
  bool measurement_enabled = false;
  int64_t phase_nanoseconds[kPlcPhaseCount] = {};
  int64_t phase_invocations[kPlcPhaseCount] = {};
  PlcFailure failure;
};

// Returns PHX_MC_OK, PHX_MC_INVALID_INPUT, PHX_MC_CONSTRAINT_INTERSECTION or
// PHX_MC_CAPACITY_EXCEEDED with `failure` set, or a point-domain status.
int32_t recover_plc(const PlcInput& input, PlcRecovery& result);

// Source-only exact validation/triangulation/canonical edge compilation.
// It performs no feature protection, volume recovery or region classification.
int32_t prepare_plc_source(const PlcInput& input, PlcRecovery& result);

// ---- feature protection (feature_protection.cpp) ------------------------

// Numerical protecting-ball radius at every acute vertex, estimated from
// distances to other features. These radii schedule insertion; they are not
// certified distance enclosures and never replace exact contact decisions.
// Every angle/distance query is charged before evaluation. Returns an empty
// vector if work_limit is exhausted (a valid PLC has at least four vertices).
NativeVector<double> protection_radii(const double* points, int64_t point_count,
                                     std::span<const int32_t> edges,
                                     std::span<const int32_t> triangles, int64_t& work,
                                     int64_t work_limit);

// Target parameter in (0, 1) of the split point of subsegment (a, b): at an
// endpoint that is a protected apex the first split lands on its protecting
// sphere and later ones on concentric power-of-two shells; otherwise the
// midpoint.  `apex_a`/`apex_b` are the protecting radii of input endpoints
// (0 for Steiner endpoints).
double split_target(const double* a, const double* b, double apex_a, double apex_b,
                    double& tolerance);

// An exactly representable point strictly inside segment (a, b) and exactly
// collinear with it, at a dyadic parameter within `tolerance` of `target`;
// false when none exists.
bool exact_segment_point(const double* a, const double* b, double target, double tolerance,
                         double* point);

// An exactly coplanar point strictly inside triangle (a, b, c) at a dyadic
// barycentric combination; false when none exists.
bool exact_triangle_point(const double* a, const double* b, const double* c, double* point);

}  // namespace phx::mc
