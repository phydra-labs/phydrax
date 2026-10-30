//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Piecewise-linear complex validation, protected boundary recovery, region
// classification, and persistent-mesh source construction.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <span>
#include <vector>

#include "boundary_policy.hpp"
#include "source_construction.hpp"
#include "cavity.hpp"

namespace phx::mc {

enum PlcFailureReason : int32_t {
  kPlcNoFailure = 0,
  kPlcDuplicatePoint = 1,
  kPlcInvalidPolygon = 2,
  kPlcInvalidFacet = 3,
  kPlcIntersecting = 4,
  kPlcOpenBoundary = 5,
  kPlcInconsistentRegions = 6,
  kPlcRegionLeak = 7,
  kPlcFixedSegment = 8,
  kPlcFixedFacet = 9,
  kPlcNonrepresentable = 10,
  kPlcInvalidSeed = 14,
  kPlcOutsideDomain = 15,
  kPlcSourceDeviation = 17,
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

struct PlcInput {
  int64_t point_count = 0;
  const double* points = nullptr;
  int64_t polygon_count = 0;
  const int64_t* polygon_offsets = nullptr;
  const int32_t* polygon_vertices = nullptr;
  const int32_t* polygon_facets = nullptr;
  int64_t facet_count = 0;
  const int32_t* facet_regions = nullptr;
  int64_t segment_count = 0;
  const int32_t* segments = nullptr;
  int64_t seed_count = 0;
  const double* seeds = nullptr;
  const int32_t* seed_regions = nullptr;
  BoundaryPolicy policy = BoundaryPolicy::kConforming;
  const double* facet_tolerances = nullptr;
  const double* segment_tolerances = nullptr;
};

namespace detail {

struct PlcWrapCavityFace {
  std::array<int32_t, 3> vertices{};
  bool obstacle = true;
  int32_t source = -1;
  bool cavity_internal = false;
  int32_t owner_tet = -1;
  int32_t neighbor_tet = -1;
};

struct PlcWrapCavityInput {
  std::span<const double> points;
  std::span<const int32_t> vertices;
  std::span<const PlcWrapCavityFace> faces;
  std::span<const int32_t> protected_segments;
  std::array<int32_t, 2> recovery_edge{-1, -1};
  bool require_edge = false;
};

struct PlcWrapCavityResult {
  bool complete = false;
  bool empty_sphere_apex_missing = false;
  bool has_facet_contact = false;
  bool cavity_internal_contact = false;
  int32_t blocking_source = -1;
  int32_t blocking_owner = -1;
  int32_t blocking_neighbor = -1;
  std::vector<std::array<int32_t, 4>> tetrahedra;
};

// Replay the production source-cavity wrapper from an exact compact fixture.
int32_t recover_plc_wrap_cavity(const PlcWrapCavityInput& input,
                                PlcWrapCavityResult& result);

}  // namespace detail

struct PlcFailure {
  int32_t reason = kPlcNoFailure;
  int32_t first_kind = kPlcNoEntity;
  int64_t first = -1;
  int32_t second_kind = kPlcNoEntity;
  int64_t second = -1;
};

struct PlcSourceRefusal {
  double certified_deviation = 0.0;
  double declared_bound = 0.0;
};

// Recovered complex: input point ids are preserved and new vertices follow.
// Cells have positive orientation; constrained subfacets retain source facet
// orientation and subsegments retain their authored PLC edge.
struct PlcRecovery {
  std::vector<double> points;
  std::vector<int32_t> tets;
  std::vector<int32_t> tet_regions;
  std::vector<int32_t> faces;
  std::vector<int32_t> face_sources;
  std::vector<int8_t> face_orientations;  // +1: each subfacet follows its authored source row
  std::vector<int32_t> segments;
  std::vector<int32_t> segment_sources;
  std::vector<double> protection;
  std::vector<int32_t> plc_edges;
  std::vector<int8_t> vertex_dimension;
  std::vector<int32_t> input_triangles;
  std::vector<int32_t> input_polygons;
  // kSegment entities index this raw plc_edges bank; kFacet entities index
  // input_triangles. Persistent source banks use SourceWitness instead.
  std::vector<PlcRecoveryWitness> witnesses;
  PlcSourceRefusal source_refusal;
  PlcFailure failure;
};

// Borrowed exact-source banks used to seed persistent tetrahedral meshes.
// Owners must copy the spans before the producer's result is released.
struct SourceComplexView {
  std::span<const double> points;
  std::span<const int32_t> faces;
  std::span<const int32_t> face_group_rows;
  std::span<const int64_t> face_ids;
  std::span<const double> face_tolerances;
  std::span<const int32_t> segments;
  std::span<const int32_t> segment_group_rows;
  std::span<const int64_t> segment_ids;
  std::span<const double> segment_tolerances;
  std::span<const SourceWitness> witnesses;
};

// Reused scratch for adjacency traversal and its conservative candidates.
struct EdgeCavityScratch {
  std::vector<std::uint32_t> visited;
  std::vector<int32_t> frontier;
  std::vector<int32_t> candidates;
  std::uint32_t stamp = 0;

  void begin(std::size_t slot_count) {
    if (visited.size() < slot_count) visited.resize(slot_count, 0);
    frontier.clear();
    candidates.clear();
    if (stamp == std::numeric_limits<std::uint32_t>::max()) {
      std::fill(visited.begin(), visited.end(), 0);
      stamp = 1;
    } else {
      ++stamp;
    }
  }
  bool mark(int32_t slot) {
    std::uint32_t& value = visited[static_cast<std::size_t>(slot)];
    if (value == stamp) return false;
    value = stamp;
    return true;
  }
};

// Complete broad-phase candidates for a source edge. Starting at a live
// tetrahedron incident to `first`, the segment's closed AABB prunes adjacency
// traversal. In the validated tetrahedralization, cells touched by one segment
// form a face-connected chain in the overlap subgraph; closed box overlap keeps
// endpoint, edge, face, and coplanar contacts. The complex adjacency is updated
// by every cavity commit, so no stale side index or per-edge slot scan is used.
// Exact contact classification remains the caller's narrow phase.
inline bool edge_cavity_candidates(const TetrahedralComplex& complex,
                                   const double* points, int32_t first,
                                   int32_t second, int32_t start,
                                   EdgeCavityScratch& scratch) {
  if (points == nullptr || first < 0 || second < 0 || first == second ||
      !complex.live(start) || is_ghost(complex.tets[static_cast<std::size_t>(start)]) ||
      vertex_slot(complex.tets[static_cast<std::size_t>(start)], first) < 0) {
    return false;
  }
  scratch.begin(complex.tets.size());
  double lower[3], upper[3];
  for (int axis = 0; axis < 3; ++axis) {
    const double a = points[3 * static_cast<std::size_t>(first) + axis];
    const double b = points[3 * static_cast<std::size_t>(second) + axis];
    lower[axis] = std::min(a, b);
    upper[axis] = std::max(a, b);
  }
  const auto overlaps = [&](const Tetrahedron& tet) {
    if (is_ghost(tet)) return false;
    for (int axis = 0; axis < 3; ++axis) {
      double tet_lower = std::numeric_limits<double>::infinity();
      double tet_upper = -std::numeric_limits<double>::infinity();
      for (int k = 0; k < 4; ++k) {
        const double coordinate = points[3 * static_cast<std::size_t>(tet.v[k]) + axis];
        tet_lower = std::min(tet_lower, coordinate);
        tet_upper = std::max(tet_upper, coordinate);
      }
      if (upper[axis] < tet_lower || tet_upper < lower[axis]) return false;
    }
    return true;
  };
  if (!overlaps(complex.tets[static_cast<std::size_t>(start)])) return false;

  scratch.mark(start);
  scratch.frontier.push_back(start);
  while (!scratch.frontier.empty()) {
    const int32_t current = scratch.frontier.back();
    scratch.frontier.pop_back();
    const Tetrahedron& tet = complex.tets[static_cast<std::size_t>(current)];
    scratch.candidates.push_back(current);
    for (int slot = 0; slot < 4; ++slot) {
      const int32_t next = tet.n[slot];
      if (next < 0) continue;
      if (!complex.live(next)) return false;
      if (!scratch.mark(next)) continue;
      const Tetrahedron& neighbor = complex.tets[static_cast<std::size_t>(next)];
      if (!is_ghost(neighbor) && overlaps(neighbor)) scratch.frontier.push_back(next);
    }
  }
  std::sort(scratch.candidates.begin(), scratch.candidates.end());
  return true;
}

// Exact narrow phase over the complete adjacency-derived broad-phase set.
int32_t collect_edge_cavity(const TetrahedralComplex& complex, const double* points,
                            int32_t first, int32_t second, int32_t start,
                            EdgeCavityScratch& scratch, std::vector<int32_t>& cavity);

int32_t recover_plc(const PlcInput& input, PlcRecovery& result);
int32_t prepare_plc_source(const PlcInput& input, PlcRecovery& result);

std::vector<double> protection_radii(const double* points, int64_t point_count,
                                     std::span<const int32_t> edges,
                                     std::span<const int32_t> triangles);
double split_target(const double* a, const double* b, double apex_a, double apex_b,
                    double& tolerance);
bool exact_segment_point(const double* a, const double* b, double target, double tolerance,
                         double* point);
bool exact_triangle_point(const double* a, const double* b, const double* c, double* point);

}  // namespace phx::mc
