//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained tetrahedral mesh state shared by quality refinement
// (refine3d.cpp) and improvement (improve3d.cpp).
//
// The state is built from explicit host arrays: positively oriented domain
// tetrahedra with regions, constrained faces (every domain-boundary face and
// every region interface) with source ids, and protected subsegments with
// source ids.  The domain is closed into a pseudomanifold by ghost
// tetrahedra over its boundary faces (kGhostVertex apex); ghost facets
// through the ghost vertex are paired combinatorially around each boundary
// edge.  Ghosts are a closure device only: nothing walks into them and they
// change only in exact conforming boundary subdivision/coarsening transactions.
//
// Every topological change is one validated CavityEdit transaction: a
// refused change leaves the scientific mesh unchanged. Constrained facets
// subdivide by exact incidence or, under a positive declared source
// deviation, by bisecting a constrained edge at the correctly rounded
// carrier of an exact source witness (source_construction.hpp); removable
// Steiner strata coarsen only with exact source-patch coverage, oriented
// material-side and shell/link proofs. Source IDs, region labels and the
// original source complex remain immutable: every vertex carries a witness
// whose certified deviation (zero exactly on the source) bounds the distance
// of its constrained faces and segments from their source rows.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "bounded_memory.hpp"
#include "cavity.hpp"
#include "mesh.hpp"
#include "intersections.hpp"
#include "expansion.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "source_construction.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {

enum class BoundaryPolicy : int32_t {
  kFixed = PHX_MC_TET_MESH_BOUNDARY_FIXED,
  kConforming = PHX_MC_TET_MESH_BOUNDARY_CONFORMING,
};

enum class TetExecutionStage : std::size_t { kRefinement, kImprovement, kExudation };

// ------------------------------------------------------------------ geometry
// Inexact constructions and quality measures.  Decisions that change topology
// or claim exact incidence use the exact predicates; these values only rank
// candidates and decide admissible quality.
namespace geometry {

inline void difference(const double* a, const double* b, double* r) {
  r[0] = a[0] - b[0];
  r[1] = a[1] - b[1];
  r[2] = a[2] - b[2];
}
inline double dot(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline void cross(const double* a, const double* b, double* r) {
  r[0] = a[1] * b[2] - a[2] * b[1];
  r[1] = a[2] * b[0] - a[0] * b[2];
  r[2] = a[0] * b[1] - a[1] * b[0];
}
inline double squared_distance(const double* a, const double* b) {
  double d[3];
  difference(a, b, d);
  return dot(d, d);
}
inline bool finite3(const double* p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Circumcenter of a tetrahedron; false when degenerate or not finite.
inline bool tetrahedron_circumcenter(const double* a, const double* b, const double* c,
                                     const double* d, double* center) {
  native_execution_primitive_query();
  // Evaluate the existing closed-form construction in the predicate owner's
  // expansion arithmetic. A rounded scalar triple product can have the wrong
  // sign (or vanish) for an exactly positive, representable tetrahedron.
  Expansion u[3], v[3], w[3], vw[3], wu[3], uv[3];
  for (int i = 0; i < 3; ++i) {
    u[i] = Expansion::difference(b[i], a[i]);
    v[i] = Expansion::difference(c[i], a[i]);
    w[i] = Expansion::difference(d[i], a[i]);
  }
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3;
    const int k = (i + 2) % 3;
    vw[i] = v[j] * w[k] - v[k] * w[j];
    wu[i] = w[j] * u[k] - w[k] * u[j];
    uv[i] = u[j] * v[k] - u[k] * v[j];
  }
  const Expansion determinant = u[0] * vw[0] + u[1] * vw[1] + u[2] * vw[2];
  const double det = determinant.scaled(2.0).estimate();
  if (determinant.is_zero() || det == 0.0 || !std::isfinite(det)) {
    return false;
  }
  const Expansion uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  const Expansion vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const Expansion ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  for (int i = 0; i < 3; ++i) {
    center[i] = a[i] + (uu * vw[i] + vv * wu[i] + ww * uv[i]).estimate() / det;
  }
  return finite3(center);
}

// Circumcenter of a triangle in its plane; coordinates shared by all three
// vertices are copied exactly, so axis-aligned facets receive exactly
// coplanar centers.
inline bool triangle_circumcenter(const double* a, const double* b, const double* c,
                                  double* center) {
  native_execution_primitive_query();
  double u[3], v[3], n[3], vn[3], nu[3];
  difference(b, a, u);
  difference(c, a, v);
  cross(u, v, n);
  const double nn = dot(n, n);
  if (!(nn > 0.0) || !std::isfinite(nn)) {
    return false;
  }
  cross(v, n, vn);
  cross(n, u, nu);
  const double uu = dot(u, u);
  const double vv = dot(v, v);
  for (int i = 0; i < 3; ++i) {
    center[i] = a[i] + (uu * vn[i] + vv * nu[i]) / (2.0 * nn);
    if (a[i] == b[i] && b[i] == c[i]) {
      center[i] = a[i];
    }
  }
  return finite3(center);
}

inline constexpr double kDegreesPerRadian = 57.295779513082320876798154814105;

// Smallest and largest interior dihedral angle (degrees); an inverted or
// degenerate tetrahedron reports 0 and 180.
inline void dihedral_extremes(const double* const* p, double& minimum, double& maximum) {
  native_execution_primitive_query();
  static constexpr int kEdges[6][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2},
                                       {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 3, 0, 1}};
  minimum = 180.0;
  maximum = 0.0;
  for (const auto& edge : kEdges) {
    double e[3], f[3], g[3], n1[3], n2[3];
    difference(p[edge[1]], p[edge[0]], e);
    difference(p[edge[2]], p[edge[0]], f);
    difference(p[edge[3]], p[edge[0]], g);
    cross(e, f, n1);
    cross(e, g, n2);
    const double scale = std::sqrt(dot(n1, n1) * dot(n2, n2));
    double angle = 0.0;
    if (scale > 0.0 && std::isfinite(scale)) {
      double normal_cross[3];
      cross(n1, n2, normal_cross);
      angle = std::atan2(std::hypot(normal_cross[0], normal_cross[1], normal_cross[2]),
                         dot(n1, n2)) * kDegreesPerRadian;
    }
    minimum = std::min(minimum, angle);
    maximum = std::max(maximum, angle);
  }
}

inline double minimum_dihedral(const double* const* p) {
  double minimum = 0.0;
  double maximum = 0.0;
  dihedral_extremes(p, minimum, maximum);
  return minimum;
}

inline double shortest_edge(const double* const* p) {
  double shortest = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      shortest = std::min(shortest, squared_distance(p[i], p[j]));
    }
  }
  return std::sqrt(shortest);
}

inline double signed_volume(const double* const* p) {
  return orient3d_exact(p[0], p[1], p[2], p[3]).estimate() / 6.0;
}

}  // namespace geometry

// The single relative-determinant floor action of improvement, proposal
// assessment and the publishing CellValidityPolicy: the affine map is charted
// from the canonical first vertex (smallest vertex identity, orientation-
// preserving rotation) that publication retains as the cell's first column,
// and det^2 > floor^2 * prod |v_k - v_0|^2 is decided exactly. `identities[k]`
// is the distinct authoritative identity and `corners[k]` the position of cell
// vertex k: native owner indices (int32) or scientific global IDs (int64).
// Only the identity order selects the chart, so their exact ranks reuse the
// one canonical permutation. Sign and normalized squared `score` follow
// relative_orient3d_exact. Any other chart would be a different policy.
template <class Identity>
inline int canonical_relative_floor(const Identity* identities, const double* const* corners,
                                    double floor, double* score) {
  int32_t ranks[4];
  for (int k = 0; k < 4; ++k) {
    ranks[k] = 0;
    for (int other = 0; other < 4; ++other) ranks[k] += identities[other] < identities[k];
  }
  int perm[4];
  canonical_permutation(ranks, 4, perm);
  return relative_orient3d_exact(corners[perm[0]], corners[perm[1]], corners[perm[2]],
                                 corners[perm[3]], floor, score);
}

// Unordered vertex pair key.
inline std::uint64_t undirected_edge(int32_t a, int32_t b) {
  const int32_t low = std::min(a, b);
  const int32_t high = std::max(a, b);
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(low)) << 32) |
         static_cast<std::uint64_t>(static_cast<std::uint32_t>(high));
}
inline int32_t edge_low(std::uint64_t key) { return static_cast<int32_t>(key >> 32); }
inline int32_t edge_high(std::uint64_t key) {
  return static_cast<int32_t>(key & 0xFFFFFFFFULL);
}

// Sorted vertex triple of a face.
struct FaceKey {
  int32_t v[3];

  static FaceKey of(int32_t a, int32_t b, int32_t c) {
    FaceKey key{{a, b, c}};
    std::sort(key.v, key.v + 3);
    return key;
  }
  bool operator==(const FaceKey& other) const {
    return v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2];
  }
  bool operator<(const FaceKey& other) const {
    return std::lexicographical_compare(v, v + 3, other.v, other.v + 3);
  }
};

struct FaceKeyHash {
  std::size_t operator()(const FaceKey& key) const {
    std::uint64_t h = static_cast<std::uint32_t>(key.v[0]) * 0x9E3779B97F4A7C15ULL;
    h ^= (h >> 29) + static_cast<std::uint32_t>(key.v[1]) * 0xC2B2AE3D27D4EB4FULL;
    h ^= (h >> 31) + static_cast<std::uint32_t>(key.v[2]) * 0x165667B19E3779F9ULL;
    return static_cast<std::size_t>(h ^ (h >> 32));
  }
};

// Why an element was left unmet (PHX_MC_TET_MESH_REASON_*).
enum class Unmet : std::uint8_t {
  kNone = 255,
  kBudget = PHX_MC_TET_MESH_REASON_BUDGET,
  kFixedBoundary = PHX_MC_TET_MESH_REASON_FIXED_BOUNDARY,
  kProtected = PHX_MC_TET_MESH_REASON_PROTECTED,
  kNonrepresentable = PHX_MC_TET_MESH_REASON_NONREPRESENTABLE,
  kRefused = PHX_MC_TET_MESH_REASON_REFUSED,
  kNoImprovement = PHX_MC_TET_MESH_REASON_NO_IMPROVEMENT,
};

struct UnmetRecord {
  int32_t v[4];
  int32_t criterion;
  int32_t reason;
  double value;
};

// Declared original source complex: triangle and segment rows
// with scientific source ids and nonnegative deviation bounds (rows of one
// source share their bound; NULL bounds are zero), and per-carrier witnesses
// (NULL: none). Rows index the independent source-point bank. Without it the
// initial faces and segments are their own exact source complex.
struct SourceComplex {
  int64_t point_count = 0;
  const double* points = nullptr;           // independent original source bank
  int64_t face_count = 0;
  const int32_t* faces = nullptr;            // face_count x 3
  const int32_t* face_sources = nullptr;
  const double* face_tolerances = nullptr;
  int64_t segment_count = 0;
  const int32_t* segments = nullptr;         // segment_count x 2
  const int32_t* segment_sources = nullptr;
  const double* segment_tolerances = nullptr;
  const int8_t* witness_strata = nullptr;    // per point, SourceStratum
  const int32_t* witness_entities = nullptr;  // source triangle or segment row
  const double* witness_parameters = nullptr;  // per point x 2
};

// A refused ancestry-backed construction on represented edge (v[0], v[1]):
// the source row it would join and its certified deviation against the
// declared bound of that row.
struct SourceRefusal {
  int32_t v[2];
  SourceStratum stratum;
  int32_t entity;
  double deviation;
  double tolerance;
};

// What an inserted point splits.
enum class InsertKind : std::uint8_t { kInterior, kSubfacet, kSubsegment };

// Outcome of preparing or committing one insertion.
enum class Insertion : std::uint8_t {
  kOk,
  kBlocked,    // the walk reached a constrained face that may not be crossed
  kDuplicate,  // the point coincides with a vertex
  kRefused,    // the constrained cavity is not a valid star (would lose a
               // vertex or segment, or fails edit validation)
  kCapacity,   // work, vertex or cell limit
  kInternal,   // a commit contradicted its validated premise
  kDeviation,  // the certified source deviation exceeds the declared bound
};

class TetMesh {
 public:
  TetMesh(BoundaryPolicy policy, int64_t max_vertices, int64_t max_tetrahedra,
          std::size_t max_allocation_bytes = std::numeric_limits<std::size_t>::max())
      : TetMesh(policy, max_vertices, max_tetrahedra,
                select_memory_owner(max_allocation_bytes)) {}
  TetMesh(const TetMesh&) = delete;
  TetMesh& operator=(const TetMesh&) = delete;

  // ---------------------------------------------------------------- build
  int32_t build(int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
                const int32_t* regions, int64_t face_count, const int32_t* faces,
                const int32_t* face_sources, int64_t segment_count, const int32_t* segments,
                const int32_t* segment_sources, const double* protection,
                const SourceComplex* source = nullptr) {
    const MemoryScope memory_scope(memory_owner_);
    try {
      TetMesh staged(policy_, max_vertices_, max_tetrahedra_, memory_owner_);
      const int32_t status = staged.build_state(
          point_count, points, tet_count, tets, regions, face_count, faces, face_sources,
          segment_count, segments, segment_sources, protection, source);
      if (status != PHX_MC_OK) {
        return status;
      }
      if (!native_execution_spend(0)) return PHX_MC_CAPACITY_EXCEEDED;
      // Publish only after every input/state allocation and predicate succeeds.
      // Moving identically owned allocators does not allocate.
      complex_ = std::move(staged.complex_);
      points_ = std::move(staged.points_);
      original_points_ = std::move(staged.original_points_);
      original_faces_ = std::move(staged.original_faces_);
      original_segments_ = std::move(staged.original_segments_);
      face_tolerance_ = std::move(staged.face_tolerance_);
      segment_tolerance_ = std::move(staged.segment_tolerance_);
      face_rows_ = std::move(staged.face_rows_);
      segment_rows_ = std::move(staged.segment_rows_);
      witnesses_ = std::move(staged.witnesses_);
      source_refusals_.clear();
      dimension_ = std::move(staged.dimension_);
      protection_ = std::move(staged.protection_);
      sizes_ = std::move(staged.sizes_);
      vertex_tet_ = std::move(staged.vertex_tet_);
      vertex_stamp_ = std::move(staged.vertex_stamp_);
      segments_ = std::move(staged.segments_);
      records_ = std::move(staged.records_);
      slot_reason_ = std::move(staged.slot_reason_);
      unmet_.clear();
      cavity_.clear();
      stack_.clear();
      scratch_star_.clear();
      boundary_.clear();
      crossed_.clear();
      boundary_edges_.clear();
      new_faces_.clear();
      edit_.begin(EditLimits{});
      transaction_open_ = false;
      pending_ = kNoPending;
      return PHX_MC_OK;
    } catch (const std::bad_alloc&) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
  }

 private:
  int32_t build_state(int64_t point_count, const double* points, int64_t tet_count,
                      const int32_t* tets, const int32_t* regions, int64_t face_count,
                      const int32_t* faces, const int32_t* face_sources,
                      int64_t segment_count, const int32_t* segments,
                      const int32_t* segment_sources, const double* protection,
                      const SourceComplex* source) {
    const MemoryScope memory_scope(memory_owner_);
    const auto n = static_cast<std::size_t>(point_count);
    native_execution_charge(tet_count);
    points_.assign(points, points + 3 * n);
    dimension_.assign(n, -1);
    sizes_.assign(n, 0.0);
    vertex_tet_.assign(n, -1);
    vertex_stamp_.assign(n + 1, 0U);
    if (protection != nullptr) {
      protection_.assign(protection, protection + n);
    } else {
      protection_.assign(n, 0.0);
    }
    for (double radius : protection_) {
      native_execution_charge(0);
      if (!std::isfinite(radius) || radius < 0.0) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    int32_t status = build_cells(point_count, tet_count, tets, regions);
    if (status == PHX_MC_OK) {
      status = build_constraints(point_count, face_count, faces, face_sources);
    }
    if (status == PHX_MC_OK) {
      SourceComplex own;
      own.face_count = face_count;
      own.faces = faces;
      own.face_sources = face_sources;
      own.segment_count = segment_count;
      own.segments = segments;
      own.segment_sources = segment_sources;
      status = build_source(point_count, source != nullptr ? *source : own);
    }
    if (status == PHX_MC_OK) {
      status = build_segments(point_count, segment_count, segments, segment_sources);
    }
    if (status == PHX_MC_OK) {
      classify_vertices();
      status = complete_boundary_witnesses();
      if (status == PHX_MC_OK) {
        status = verify_ancestry(point_count, face_count, faces, face_sources, segment_count,
                                 segments, segment_sources);
      }
    }
    slot_reason_.assign(complex_.tets.size(), Unmet::kNone);
    records_.clear();
    records_.shrink_to_fit();
    return status;
  }

  // Source rows, their declared bounds and the natively certified witness
  // of every initial point (deviation recomputed, never trusted).
  int32_t build_source(int64_t point_count, const SourceComplex& source) {
    const MemoryScope memory_scope(memory_owner_);
    if (source.face_count < 0 || source.segment_count < 0 ||
        (source.face_count > 0 && (source.faces == nullptr || source.face_sources == nullptr)) ||
        (source.segment_count > 0 &&
         (source.segments == nullptr || source.segment_sources == nullptr)) ||
        ((source.witness_strata == nullptr) != (source.witness_entities == nullptr)) ||
        ((source.witness_strata == nullptr) != (source.witness_parameters == nullptr))) {
      return PHX_MC_INVALID_INPUT;
    }
    const int64_t source_count = source.points == nullptr ? point_count : source.point_count;
    if (source_count < 1 || source_count > kMaxMeshPoints) {
      return PHX_MC_INVALID_INPUT;
    }
    const auto valid_vertex = [&](int32_t v) { return v >= 0 && v < source_count; };
    const auto valid_bound = [](double value) { return std::isfinite(value) && value >= 0.0; };
    if (source.points == nullptr) {
      original_points_ = points_;
    } else {
      original_points_.assign(source.points, source.points + 3 * source_count);
      for (double value : original_points_) {
        native_execution_charge(1);
        if (!std::isfinite(value)) {
          return PHX_MC_INVALID_INPUT;
        }
      }
    }
    original_faces_.clear();
    original_segments_.clear();
    face_tolerance_.clear();
    segment_tolerance_.clear();
    face_rows_.clear();
    segment_rows_.clear();
    for (int64_t i = 0; i < source.face_count; ++i) {
      native_execution_charge(0);
      const int32_t* f = source.faces + 3 * i;
      const double bound = source.face_tolerances == nullptr ? 0.0 : source.face_tolerances[i];
      if (!valid_vertex(f[0]) || !valid_vertex(f[1]) || !valid_vertex(f[2]) ||
          source.face_sources[i] < 0 || !valid_bound(bound) ||
          collinear3d(original_point(f[0]), original_point(f[1]), original_point(f[2]))) {
        return PHX_MC_INVALID_INPUT;
      }
      original_faces_.push_back({f[0], f[1], f[2], source.face_sources[i]});
      face_tolerance_.push_back(bound);
      face_rows_.push_back({source.face_sources[i], static_cast<int32_t>(i)});
    }
    for (int64_t i = 0; i < source.segment_count; ++i) {
      native_execution_charge(0);
      const int32_t* s = source.segments + 2 * i;
      const double bound =
          source.segment_tolerances == nullptr ? 0.0 : source.segment_tolerances[i];
      if (!valid_vertex(s[0]) || !valid_vertex(s[1]) || s[0] == s[1] ||
          source.segment_sources[i] < 0 || !valid_bound(bound) ||
          std::equal(original_point(s[0]), original_point(s[0]) + 3, original_point(s[1]))) {
        return PHX_MC_INVALID_INPUT;
      }
      original_segments_.push_back({s[0], s[1], source.segment_sources[i]});
      segment_tolerance_.push_back(bound);
      segment_rows_.push_back({source.segment_sources[i], static_cast<int32_t>(i)});
    }
    std::sort(face_rows_.begin(), face_rows_.end());
    std::sort(segment_rows_.begin(), segment_rows_.end());
    // A source segment geometrically contained in a source triangle must obey
    // both declarations, including authored subsegments whose endpoints are
    // not the triangle's corner pair.
    for (std::size_t face_row = 0; face_row < original_faces_.size(); ++face_row) {
      const double* face_corners[3];
      row_corners(SourceStratum::kFacet, face_row, face_corners);
      for (std::size_t segment_row = 0; segment_row < original_segments_.size();
           ++segment_row) {
        // A zero segment bound cannot be tightened by any nonnegative facet
        // bound; no geometric containment query is needed for that row.
        if (segment_tolerance_[segment_row] == 0.0) continue;
        native_execution_charge(2);
        const auto& segment = original_segments_[segment_row];
        if (on_source_entity(face_corners, SourceStratum::kFacet,
                             original_point(segment[0])) &&
            on_source_entity(face_corners, SourceStratum::kFacet,
                             original_point(segment[1]))) {
          segment_tolerance_[segment_row] =
              std::min(segment_tolerance_[segment_row], face_tolerance_[face_row]);
        }
      }
    }
    witnesses_.assign(static_cast<std::size_t>(point_count), SourceWitness{});
    if (source.witness_strata == nullptr) {
      return PHX_MC_OK;
    }
    for (int64_t v = 0; v < point_count; ++v) {
      native_execution_charge(0);
      SourceWitness witness;
      witness.entity = source.witness_entities[v];
      witness.parameters[0] = source.witness_parameters[2 * v];
      witness.parameters[1] = source.witness_parameters[2 * v + 1];
      switch (source.witness_strata[v]) {
        case static_cast<int8_t>(SourceStratum::kNone):
          if (witness.entity != -1 || witness.parameters[0] != 0.0 ||
              witness.parameters[1] != 0.0) {
            return PHX_MC_INVALID_INPUT;
          }
          continue;
        case static_cast<int8_t>(SourceStratum::kSegment):
          witness.stratum = SourceStratum::kSegment;
          break;
        case static_cast<int8_t>(SourceStratum::kFacet):
          witness.stratum = SourceStratum::kFacet;
          break;
        default:
          return PHX_MC_INVALID_INPUT;
      }
      const std::size_t rows = witness.stratum == SourceStratum::kSegment
                                   ? original_segments_.size()
                                   : original_faces_.size();
      if (witness.entity < 0 || static_cast<std::size_t>(witness.entity) >= rows) {
        return PHX_MC_INVALID_INPUT;
      }
      const double* corners[3];
      row_corners(witness.stratum, static_cast<std::size_t>(witness.entity), corners);
      witness.deviation = source_deviation(corners, witness.stratum, witness.parameters,
                                           points_.data() + 3 * v);
      if (witness.deviation < 0.0 ||
          witness.deviation > row_tolerance(witness.stratum, witness.entity) ||
          (witness.deviation > 0.0 &&
           !rounded_carrier(witness.stratum, witness.entity, witness, points_.data() + 3 * v))) {
        return PHX_MC_INVALID_INPUT;
      }
      witnesses_[static_cast<std::size_t>(v)] = witness;
    }
    return PHX_MC_OK;
  }

  // Original exact boundary vertices also name a real source row. A missing
  // input witness is completed from its represented constrained support, never
  // interpreted as an unconstrained boundary authority.
  int32_t complete_boundary_witnesses() {
    for (int32_t v = 0; v < vertex_count(); ++v) {
      native_execution_charge(1);
      if (dimension(v) < 0 || dimension(v) >= 3 ||
          witness(v).stratum != SourceStratum::kNone) {
        continue;
      }
      SourceWitness completed;
      const Insertion admitted = relocation_witness(v, point(v), completed);
      if (admitted != Insertion::kOk) {
        return admitted == Insertion::kCapacity ? PHX_MC_CAPACITY_EXCEEDED
                                                : PHX_MC_INVALID_INPUT;
      }
      set_witness(v, completed);
    }
    return PHX_MC_OK;
  }

  // Every constrained face and segment of a carrier built over a declared
  // source complex lies on a source row of its own source, and only boundary
  // strata carry witnesses.
  int32_t verify_ancestry(int64_t point_count, int64_t face_count, const int32_t* faces,
                          const int32_t* face_sources, int64_t segment_count,
                          const int32_t* segments, const int32_t* segment_sources) const {
    for (int64_t v = 0; v < point_count; ++v) {
      native_execution_charge(0);
      const auto index = static_cast<std::size_t>(v);
      if ((witnesses_[index].stratum != SourceStratum::kNone) !=
          (dimension_[index] >= 0 && dimension_[index] < 3)) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    for (int64_t i = 0; i < face_count; ++i) {
      native_execution_charge(0);
      if (original_face_ancestor(faces + 3 * i, face_sources[i]) < 0) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    for (int64_t i = 0; i < segment_count; ++i) {
      native_execution_charge(0);
      if (original_segment_ancestor(segments[2 * i], segments[2 * i + 1], segment_sources[i]) < 0) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    for (int32_t t = 0; t < static_cast<int32_t>(complex_.tets.size()); ++t) {
      if (finite(t) && !positive(tet(t).v)) return PHX_MC_INVALID_INPUT;
    }
    return PHX_MC_OK;
  }

 public:
  // ------------------------------------------------------------- accessors
  BoundaryPolicy policy() const { return policy_; }
  int64_t vertex_count() const { return static_cast<int64_t>(dimension_.size()); }
  int64_t max_vertices() const { return max_vertices_; }
  int64_t max_tetrahedra() const { return max_tetrahedra_; }
  const MemoryOwner& memory_owner() const noexcept { return memory_owner_; }
  bool set_memory_limit(std::size_t bytes) noexcept { return memory_owner_->set_limit(bytes); }
  std::array<std::uint64_t, 6> memory_evidence() const noexcept {
    return memory_owner_->evidence();
  }
  TetrahedralComplex& complex() { return complex_; }
  const TetrahedralComplex& complex() const { return complex_; }
  const Tetrahedron& tet(int32_t t) const { return complex_.tets[static_cast<std::size_t>(t)]; }
  const double* point(int32_t v) const {
    return v == pending_ ? pending_point_ : points_.data() + 3 * static_cast<std::size_t>(v);
  }
  double* mutable_point(int32_t v) { return points_.data() + 3 * static_cast<std::size_t>(v); }
  int8_t dimension(int32_t v) const { return dimension_[static_cast<std::size_t>(v)]; }
  double protection(int32_t v) const { return protection_[static_cast<std::size_t>(v)]; }
  double size(int32_t v) const { return sizes_[static_cast<std::size_t>(v)]; }
  std::span<double> sizes() { return sizes_; }
  bool live_vertex(int32_t v) const {
    return v >= 0 && v < vertex_count() && dimension_[static_cast<std::size_t>(v)] >= 0;
  }
  // Broken only after a failed commit premise, never an allocation refusal.
  bool broken() const { return broken_ || transaction_open_; }
  void mark_broken() { broken_ = true; }
  int64_t work() const { return work_; }
  int64_t accepted_generation() const noexcept { return accepted_generation_; }
  void mark_accepted_state() noexcept { ++accepted_generation_; }
  void set_work_limit(int64_t budget) {
    work_exhausted_ = false;
    work_limit_ = work_ > std::numeric_limits<int64_t>::max() - budget
                      ? std::numeric_limits<int64_t>::max()
                      : work_ + budget;
  }
  int64_t work_ceiling() const noexcept { return work_limit_; }
  void restore_work_ceiling(int64_t ceiling) noexcept {
    work_limit_ = ceiling;
    work_exhausted_ = false;
  }
  bool spend(int64_t units) {
    if (work_exhausted_ || units < 0 || work_ > work_limit_ - units ||
        !native_execution_spend(units)) {
      work_exhausted_ = true;
      return false;
    }
    work_ += units;
    return true;
  }
  bool work_exhausted() const { return work_exhausted_; }
  std::size_t largest_cavity() const { return largest_cavity_; }
  NativeVector<UnmetRecord>& unmet() { return unmet_; }
  std::span<const UnmetRecord> unmet() const { return unmet_; }
  Unmet slot_reason(int32_t t) const { return slot_reason_[static_cast<std::size_t>(t)]; }
  void set_slot_reason(int32_t t, Unmet reason) {
    slot_reason_[static_cast<std::size_t>(t)] = reason;
  }
  void clear_slot_reasons() { std::fill(slot_reason_.begin(), slot_reason_.end(), Unmet::kNone); }

  // Segment source of an edge, or -1.
  int32_t segment_source(int32_t a, int32_t b) const {
    const auto found = segments_.find(undirected_edge(a, b));
    return found == segments_.end() ? -1 : found->second;
  }
  const NativeUnorderedMap<std::uint64_t, int32_t>& segments() const { return segments_; }

  // Source rows are ancestry identities, independent of source IDs that may
  // identify several triangles. Original coordinates never relocate.
  const double* original_point(int32_t vertex) const {
    return original_points_.data() + 3 * static_cast<std::size_t>(vertex);
  }
  void row_corners(SourceStratum stratum, std::size_t row, const double** corners) const {
    if (stratum == SourceStratum::kSegment) {
      const auto& segment = original_segments_[row];
      corners[0] = original_point(segment[0]);
      corners[1] = original_point(segment[1]);
      return;
    }
    const auto& face = original_faces_[row];
    for (int k = 0; k < 3; ++k) {
      corners[k] = original_point(face[static_cast<std::size_t>(k)]);
    }
  }
  double row_tolerance(SourceStratum stratum, int32_t row) const {
    return stratum == SourceStratum::kSegment ? segment_tolerance_[static_cast<std::size_t>(row)]
                                              : face_tolerance_[static_cast<std::size_t>(row)];
  }
  // Exact witness S membership in a source triangle. The binary carrier may
  // be off-plane, so ancestry is decided from the authored row and parameters.
  bool witness_on_facet(const SourceWitness& witness, std::size_t row) const {
    if (witness.stratum != SourceStratum::kSegment &&
        witness.stratum != SourceStratum::kFacet) {
      return false;
    }
    const double* witness_corners[3];
    row_corners(witness.stratum, static_cast<std::size_t>(witness.entity),
                witness_corners);
    Expansion source[3];
    if (!source_point(witness_corners, witness.stratum, witness.parameters, source)) {
      return false;
    }
    const double* triangle[3];
    row_corners(SourceStratum::kFacet, row, triangle);
    Expansion first[3], second[3], normal[3];
    for (int axis = 0; axis < 3; ++axis) {
      first[axis] = Expansion::difference(triangle[1][axis], triangle[0][axis]);
      second[axis] = Expansion::difference(triangle[2][axis], triangle[0][axis]);
    }
    normal[0] = first[1] * second[2] + (first[2] * second[1]).scaled(-1.0);
    normal[1] = first[2] * second[0] + (first[0] * second[2]).scaled(-1.0);
    normal[2] = first[0] * second[1] + (first[1] * second[0]).scaled(-1.0);
    Expansion plane;
    for (int axis = 0; axis < 3; ++axis) {
      plane = plane + normal[axis] * (source[axis] + Expansion(-triangle[0][axis]));
    }
    if (plane.sign() != 0) {
      return false;
    }
    int drop = 0;
    while (drop < 3 && normal[drop].sign() == 0) {
      ++drop;
    }
    if (drop == 3) {
      return false;
    }
    const int x = (drop + 1) % 3;
    const int y = (drop + 2) % 3;
    const int orientation = normal[drop].sign();
    for (int edge = 0; edge < 3; ++edge) {
      const double* a = triangle[edge];
      const double* b = triangle[(edge + 1) % 3];
      const Expansion turn =
          Expansion::difference(b[x], a[x]) * (source[y] + Expansion(-a[y])) +
          (Expansion::difference(b[y], a[y]) *
           (source[x] + Expansion(-a[x]))).scaled(-1.0);
      if (turn.sign() != 0 && turn.sign() != orientation) {
        return false;
      }
    }
    return true;
  }

  // Whether vertex v lies on a source row: exactly, or by its bounded witness
  // on that row (a bounded segment witness lies on every triangle row that
  // contains its whole segment row).
  bool on_source_row(int32_t v, SourceStratum stratum, std::size_t row) const {
    const SourceWitness& witness = witnesses_[static_cast<std::size_t>(v)];
    const double* corners[3];
    row_corners(stratum, row, corners);
    if (witness.deviation == 0.0) {
      return on_source_entity(corners, stratum, point(v));
    }
    if (witness.stratum == stratum &&
        static_cast<std::size_t>(witness.entity) == row) {
      return true;
    }
    if (stratum != SourceStratum::kFacet) {
      return false;
    }
    return witness.deviation <= face_tolerance_[row] && witness_on_facet(witness, row);
  }
  // First source row of `source` holding every support vertex, or -1.
  int32_t source_ancestor(SourceStratum stratum, int32_t source,
                          std::span<const int32_t> support) const {
    const MemoryScope memory_scope(memory_owner_);
    const auto& rows = stratum == SourceStratum::kSegment ? segment_rows_ : face_rows_;
    const std::pair<int32_t, int32_t> probe{source, std::numeric_limits<int32_t>::min()};
    for (auto row = std::lower_bound(rows.begin(), rows.end(), probe);
         row != rows.end() && row->first == source; ++row) {
      native_execution_charge(0);
      bool contained = true;
      for (int32_t v : support) {
        contained = contained && on_source_row(v, stratum, static_cast<std::size_t>(row->second));
      }
      if (contained) {
        return row->second;
      }
    }
    return -1;
  }
  int32_t original_face_ancestor(const int32_t* vertices, int32_t source) const {
    return source_ancestor(SourceStratum::kFacet, source, std::span<const int32_t>(vertices, 3));
  }
  int32_t original_segment_ancestor(int32_t a, int32_t b, int32_t source) const {
    const int32_t support[2] = {a, b};
    return source_ancestor(SourceStratum::kSegment, source, support);
  }
  int64_t source_facet_count() const { return static_cast<int64_t>(original_faces_.size()); }
  std::span<const std::array<int32_t, 4>> original_faces() const { return original_faces_; }
  std::span<const std::array<int32_t, 3>> original_segments() const { return original_segments_; }
  std::span<const double> face_tolerances() const { return face_tolerance_; }
  std::span<const double> segment_tolerances() const { return segment_tolerance_; }
  const SourceWitness& witness(int32_t v) const { return witnesses_[static_cast<std::size_t>(v)]; }
  std::span<const SourceRefusal> source_refusals() const { return source_refusals_; }
  void clear_source_refusals() { source_refusals_.clear(); }
  bool protect_vertices(int64_t count, const int32_t* vertices) {
    if (count < 0 || (count > 0 && vertices == nullptr) || !spend(count) || !spend(count)) {
      return false;
    }
    for (int64_t i = 0; i < count; ++i) {
      if (!live_vertex(vertices[i])) {
        return false;
      }
    }
    for (int64_t i = 0; i < count; ++i) {
      dimension_[static_cast<std::size_t>(vertices[i])] = 0;
    }
    if (count > 0) {
      mark_accepted_state();
    }
    return true;
  }
  // Maximum certified deviation of the live vertices from their source rows:
  // every constrained face and segment lies within it of its source row
  // (convexity of the source entities). Exact meshes report zero.
  double achieved_source_error_bound() const {
    double bound = 0.0;
    for (std::size_t v = 0; v < witnesses_.size(); ++v) {
      if (dimension_[v] >= 0) {
        bound = std::max(bound, witnesses_[v].deviation);
      }
    }
    return bound;
  }
  // Largest declared bound of any source row.
  double requested_source_tolerance() const {
    double bound = 0.0;
    for (double value : face_tolerance_) bound = std::max(bound, value);
    for (double value : segment_tolerance_) bound = std::max(bound, value);
    return bound;
  }

  // Runtime observation only: neither scientific arrays nor work counters carry
  // these values. Each slot records the latest actual native invocation.
  void measure_execution(bool enabled) {
    measure_execution_ = enabled;
    execution_seconds_.fill(0.0);
    execution_measured_.fill(0);
  }
  const std::array<double, 3>& execution_seconds() const { return execution_seconds_; }
  const std::array<int32_t, 3>& execution_measured() const { return execution_measured_; }

  // An explicit preference is tried first, then exact dyadics ranked by distance.
  // The default retains its original dyadic order. Every returned parameter is
  // the actual exactly represented affine construction, never a rounded snap.
  bool construct_edge_point(int32_t a, int32_t b, double* position, double* parameter = nullptr,
                            double preferred_fraction = 0.0) {
    const MemoryScope memory_scope(memory_owner_);
    if (!live_vertex(a) || !live_vertex(b) || a == b ||
        !std::isfinite(preferred_fraction) || preferred_fraction < 0.0 ||
        preferred_fraction >= 1.0) {
      return false;
    }
    const auto construct = [&](double fraction, bool general) {
      if (!spend(3)) {
        return false;
      }
      // Domain coordinates lie on a lattice with spacing at least 2^(min-52),
      // and an edge component is at most 2^(max+1). Smaller fractions cannot
      // produce a distinct domain point. Refusing that candidate also keeps
      // every scaled expansion component above binary64 product underflow.
      constexpr double minimum_fraction =
          (kCoordinateMinMagnitude / kCoordinateMaxMagnitude) * 0x1p-53;
      if (general && fraction < minimum_fraction) {
        return false;
      }
      bool exact = true;
      for (int k = 0; k < 3; ++k) {
        // A general preference must not round 1-f. Keep the original algebra
        // for the default dyadics, whose complementary fractions are exact.
        const Expansion value =
            general ? Expansion(point(a)[k]) +
                          Expansion::difference(point(b)[k], point(a)[k]).scaled(fraction)
                    : Expansion::product(point(a)[k], 1.0 - fraction) +
                          Expansion::product(point(b)[k], fraction);
        position[k] = value.estimate();
        exact = exact && (value - Expansion(position[k])).is_zero();
      }
      if (exact && coordinate_in_domain(position[0]) && coordinate_in_domain(position[1]) &&
          coordinate_in_domain(position[2]) && collinear3d(point(a), point(b), position) &&
          !std::equal(position, position + 3, point(a)) &&
          !std::equal(position, position + 3, point(b))) {
        if (parameter != nullptr) {
          *parameter = fraction;
        }
        return true;
      }
      return false;
    };
    if (preferred_fraction != 0.0) {
      if (construct(preferred_fraction, true)) {
        return true;
      }
      // The existing level-1..8 candidates are exactly [64,192]/256.
      // Walk the two nearest frontiers without allocating or sorting. Ties
      // retain the original order: coarser dyadics first, then lower numerator.
      int lower = std::clamp(static_cast<int>(preferred_fraction * 256.0), 63, 192);
      int upper = lower + 1;
      while (!work_exhausted() && (lower >= 64 || upper <= 192)) {
        bool take_lower = upper > 192;
        if (lower >= 64 && upper <= 192) {
          const double lower_distance = preferred_fraction - std::ldexp(lower, -8);
          const double upper_distance = std::ldexp(upper, -8) - preferred_fraction;
          take_lower = lower_distance < upper_distance ||
                       (lower_distance == upper_distance &&
                        (lower & -lower) >= (upper & -upper));
        }
        const double fraction = std::ldexp(take_lower ? lower-- : upper++, -8);
        if (fraction != preferred_fraction && construct(fraction, false)) {
          return true;
        }
      }
      return false;
    }
    for (int level = 1; level <= 8; ++level) {
      const int denominator = 1 << level;
      for (int numerator = 1; numerator < denominator; numerator += 2) {
        const double fraction = std::ldexp(static_cast<double>(numerator), -level);
        if (fraction < 0.25 || fraction > 0.75) {
          continue;
        }
        if (construct(fraction, false)) {
          return true;
        }
        if (work_exhausted()) {
          return false;
        }
      }
    }
    return false;
  }
  bool construct_facet_point(int32_t a, int32_t b, int32_t c, double* position) {
    const MemoryScope memory_scope(memory_owner_);
    if (!live_vertex(a) || !live_vertex(b) || !live_vertex(c)) {
      return false;
    }
    const double* triangle[3] = {point(a), point(b), point(c)};
    for (int level = 2; level <= 6; ++level) {
      const int denominator = 1 << level;
      for (int u = 1; u < denominator; ++u) {
        for (int v = 1; u + v < denominator; ++v) {
          const double x = std::ldexp(static_cast<double>(u), -level);
          const double y = std::ldexp(static_cast<double>(v), -level);
          if (x < 0.125 || y < 0.125 || x + y > 0.875) {
            continue;
          }
          if (!spend(4)) {
            return false;
          }
          for (int k = 0; k < 3; ++k) {
            position[k] = (Expansion::product(triangle[0][k], 1.0 - x - y) +
                           Expansion::product(triangle[1][k], x) +
                           Expansion::product(triangle[2][k], y)).estimate();
          }
          int side = 0;
          if (coordinate_in_domain(position[0]) && coordinate_in_domain(position[1]) &&
              coordinate_in_domain(position[2]) &&
              locate_on_triangle(position, triangle, side) == kFeatureInterior) {
            return true;
          }
        }
      }
    }
    return false;
  }

  bool finite(int32_t t) const { return !is_ghost(tet(t)); }
  void corners(int32_t t, const double** p) const {
    for (int k = 0; k < 4; ++k) {
      p[k] = point(tet(t).v[k]);
    }
  }
  void corners(const int32_t* v, const double** p) const {
    for (int k = 0; k < 4; ++k) {
      p[k] = point(v[k]);
    }
  }
  bool source_position(int32_t vertex, Expansion* coordinates) const {
    const SourceWitness& source = vertex == pending_ ? pending_source_ : witness(vertex);
    if (source.deviation > 0.0) {
      const double* corners[3];
      row_corners(source.stratum, static_cast<std::size_t>(source.entity), corners);
      if (!on_source_entity(corners, source.stratum, point(vertex))) {
        return source_point(corners, source.stratum, source.parameters, coordinates);
      }
    }
    for (int axis = 0; axis < 3; ++axis) coordinates[axis] = Expansion(point(vertex)[axis]);
    return true;
  }

  int source_orientation(const int32_t* v, int carrier_sign = 2) const {
    const MemoryScope memory_scope(memory_owner_);
    bool bounded = false;
    for (int k = 0; k < 4; ++k) {
      bounded = bounded || (v[k] == pending_ ? pending_source_.deviation
                                             : witness(v[k]).deviation) > 0.0;
    }
    if (!bounded) return carrier_sign == 2
        ? orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3])) : carrier_sign;
    native_execution_primitive_query();
    Expansion coordinates[4][3];
    for (int k = 0; k < 4; ++k) {
      if (!source_position(v[k], coordinates[k])) return 0;
    }
    Expansion u[3], w[3], z[3];
    for (int axis = 0; axis < 3; ++axis) {
      u[axis] = coordinates[1][axis] - coordinates[0][axis];
      w[axis] = coordinates[2][axis] - coordinates[0][axis];
      z[axis] = coordinates[3][axis] - coordinates[0][axis];
    }
    return (u[0] * (w[1] * z[2] - w[2] * z[1]) -
            u[1] * (w[0] * z[2] - w[2] * z[0]) +
            u[2] * (w[0] * z[1] - w[1] * z[0])).sign();
  }
  bool positive(const int32_t* v) const {
    const int carrier = orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3]));
    return carrier > 0 && source_orientation(v, carrier) > 0;
  }
  // orient3d of tetrahedron t with v[slot] replaced by p.
  int orient_with(int32_t t, int slot, const double* p) const {
    const MemoryScope memory_scope(memory_owner_);
    if (p == pending_point_ && pending_ != kNoPending) {
      int32_t vertices[4];
      std::copy_n(tet(t).v, 4, vertices);
      vertices[slot] = pending_;
      return source_orientation(vertices);
    }
    const double* q[4];
    for (int k = 0; k < 4; ++k) {
      q[k] = k == slot ? p : point(tet(t).v[k]);
    }
    return orient3d(q[0], q[1], q[2], q[3]);
  }
  double min_dihedral(int32_t t) const {
    const MemoryScope memory_scope(memory_owner_);
    const double* p[4];
    corners(t, p);
    return geometry::minimum_dihedral(p);
  }

  // ------------------------------------------------------------- topology
  // Live tetrahedra (ghosts included) incident to vertex a, in a
  // deterministic breadth-first order from its recorded incident tetrahedron.
  template <class Allocator>
  bool vertex_star(int32_t a, std::vector<int32_t, Allocator>& star) {
    const MemoryScope memory_scope(memory_owner_);
    star.clear();
    if (!live_vertex(a)) {
      return false;
    }
    const int32_t start = vertex_tet_[static_cast<std::size_t>(a)];
    if (start < 0 || !complex_.live(start) || vertex_slot(tet(start), a) < 0) {
      return false;
    }
    if (!spend(1)) return false;
    next_stamp();
    mark(start);
    star.push_back(start);
    for (std::size_t i = 0; i < star.size(); ++i) {
      native_execution_charge(0);
      const Tetrahedron& current = tet(star[i]);
      const int apex = vertex_slot(current, a);
      for (int k = 0; k < 4; ++k) {
        const int32_t next = current.n[k];
        if (k != apex && !marked(next)) {
          if (!spend(1)) return false;
          mark(next);
          star.push_back(next);
        }
      }
    }
    return true;
  }

  // A live tetrahedron containing edge (a, b), or -1.
  int32_t edge_tet(int32_t a, int32_t b) {
    const MemoryScope memory_scope(memory_owner_);
    if (!vertex_star(a, scratch_star_)) {
      return -1;
    }
    for (int32_t t : scratch_star_) {
      if (vertex_slot(tet(t), b) >= 0) {
        return t;
      }
    }
    return -1;
  }

  // A finite tetrahedron with face (a, b, c) and the slot opposite it.
  bool find_face(int32_t a, int32_t b, int32_t c, int32_t& t, int& slot) {
    const MemoryScope memory_scope(memory_owner_);
    if (!vertex_star(a, scratch_star_)) {
      return false;
    }
    for (int32_t s : scratch_star_) {
      const Tetrahedron& current = tet(s);
      if (is_ghost(current) || vertex_slot(current, b) < 0 || vertex_slot(current, c) < 0) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        if (current.v[k] != a && current.v[k] != b && current.v[k] != c) {
          t = s;
          slot = k;
          return true;
        }
      }
    }
    return false;
  }

  // Target size at p by clamped barycentric interpolation in finite
  // tetrahedron t (0 when no vertex carries a target).
  double interpolated_size(int32_t t, const double* p) const {
    const Tetrahedron& current = tet(t);
    double weights[4];
    double total = 0.0;
    for (int k = 0; k < 4; ++k) {
      const double* q[4];
      for (int j = 0; j < 4; ++j) {
        q[j] = j == k ? p : point(current.v[j]);
      }
      weights[k] = std::max(0.0, geometry::signed_volume(q));
      total += weights[k];
    }
    double value = 0.0;
    for (int k = 0; k < 4; ++k) {
      const double w = total > 0.0 ? weights[k] / total : 0.25;
      value += w * size(current.v[k]);
    }
    return std::isfinite(value) ? value : 0.0;
  }

  // Tetrahedra around edge (a, b) in rotation order: tets[i] is (a, b,
  // ring[i], ring[i + 1]) as an even permutation; ring may hold the ghost
  // vertex.  False when the edge does not exist.
  template <class TetAllocator, class RingAllocator>
  bool edge_ring(int32_t a, int32_t b, std::vector<int32_t, TetAllocator>& tets,
                 std::vector<int32_t, RingAllocator>& ring) {
    const MemoryScope memory_scope(memory_owner_);
    tets.clear();
    ring.clear();
    const int32_t start = edge_tet(a, b);
    if (start < 0) {
      return false;
    }
    int32_t t = start;
    int32_t entry = -1;
    for (;;) {
      const Tetrahedron& current = tet(t);
      int others[2];
      int count = 0;
      for (int k = 0; k < 4; ++k) {
        if (current.v[k] != a && current.v[k] != b) {
          others[count++] = k;
        }
      }
      int32_t first = current.v[others[0]];
      int32_t second = current.v[others[1]];
      const int32_t order[4] = {a, b, first, second};
      if (!even_permutation(current.v, order)) {
        std::swap(first, second);
      }
      if (entry >= 0 && first != entry) {
        return false;
      }
      tets.push_back(t);
      ring.push_back(first);
      // Next around the edge: across the facet (a, b, second).
      const int32_t next = current.n[vertex_slot(current, first)];
      if (next == start) {
        break;
      }
      if (tets.size() > complex_.tets.size()) {
        return false;
      }
      entry = second;
      t = next;
    }
    return spend(static_cast<int64_t>(tets.size()));
  }

  // ------------------------------------------------------------ insertion
  // Visibility walk from `start` toward p that never crosses a constrained
  // face: returns a tetrahedron whose closure contains p, or -1 with
  // blocked_tet/blocked_slot naming the constrained face that stopped it
  // (blocked_tet < 0 when the work limit stopped it).
  int32_t locate(const double* p, int32_t start, int32_t& blocked_tet, int& blocked_slot) {
    const MemoryScope memory_scope(memory_owner_);
    blocked_tet = -1;
    blocked_slot = -1;
    int32_t t = start;
    int32_t previous = -1;
    for (;;) {
      if (!spend(1)) {
        return -1;
      }
      const Tetrahedron& current = tet(t);
      const int first = static_cast<int>(splitmix64(walk_step_++) & 3U);
      int32_t next = -1;
      int crossing = -1;
      for (int i = 0; i < 4; ++i) {
        const int k = (first + i) & 3;
        if (current.n[k] == previous) {
          continue;
        }
        if (orient_with(t, k, p) < 0) {
          next = current.n[k];
          crossing = k;
          break;
        }
      }
      if (next < 0) {
        return t;
      }
      if (complex_.constraint(t, crossing) != kNoConstraint || is_ghost(tet(next))) {
        blocked_tet = t;
        blocked_slot = crossing;
        return -1;
      }
      previous = t;
      t = next;
    }
  }

  // One exact old-edge witness/source-interval admission, shared by both
  // fixed-star subdivision and expanded constrained conflict-cavity insertion.
  bool split_positions_admitted(int32_t a, int32_t b, const double* witness,
                                const double* position, int32_t& source,
                                bool& on_edge) const {
    if (!live_vertex(a) || !live_vertex(b) || a == b ||
        !geometry::finite3(witness) || !geometry::finite3(position) ||
        !coordinate_in_domain(witness[0]) || !coordinate_in_domain(witness[1]) ||
        !coordinate_in_domain(witness[2]) || !coordinate_in_domain(position[0]) ||
        !coordinate_in_domain(position[1]) || !coordinate_in_domain(position[2]) ||
        !collinear3d(point(a), point(b), witness)) {
      return false;
    }
    int axis = 0;
    for (int k = 1; k < 3; ++k) {
      if (std::abs(point(a)[k] - point(b)[k]) >
          std::abs(point(a)[axis] - point(b)[axis])) {
        axis = k;
      }
    }
    const double low = std::min(point(a)[axis], point(b)[axis]);
    const double high = std::max(point(a)[axis], point(b)[axis]);
    if (!(low < witness[axis] && witness[axis] < high)) {
      return false;
    }
    on_edge = witness == position || collinear3d(point(a), point(b), position);
    source = segment_source(a, b);
    return !(source >= 0 && !on_edge) &&
           !(on_edge && !(low < position[axis] && position[axis] < high));
  }

  // Locate and prepare the existing exact conflict-cavity insertion while
  // retaining the seed edge's scientific source stratum and protection balls.
  Insertion prepare_edge_insertion(int32_t a, int32_t b, const double* witness,
                                   const double* position, double source_fraction,
                                   std::size_t cell_limit,
                                   InsertKind& kind) try {
    const MemoryScope memory_scope(memory_owner_);
    int32_t source = -1;
    bool on_edge = false;
    SourceWitness construction;
    bool bounded = false;
    if (!split_positions_admitted(a, b, witness, position, source, on_edge)) {
      if (!(source_fraction > 0.0) || !std::equal(witness, witness + 3, position)) {
        return Insertion::kRefused;
      }
      double carrier[3];
      const Insertion status = construct_source_split(a, b, source_fraction, carrier, construction);
      if (status != Insertion::kOk) {
        return status;
      }
      if (!std::equal(carrier, carrier + 3, witness)) {
        return Insertion::kRefused;
      }
      bounded = true;
      source = segment_source(a, b);
      on_edge = true;
    }
    if (vertex_count() >= max_vertices_ || cell_limit == 0) {
      return Insertion::kCapacity;
    }
    NativeVector<int32_t> star;
    NativeVector<int32_t> ring;
    if (!edge_ring(a, b, star, ring)) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    int32_t seed = -1;
    kind = source >= 0 ? InsertKind::kSubsegment : InsertKind::kInterior;
    for (int32_t t : star) {
      if (is_ghost(tet(t))) {
        continue;
      }
      if (seed < 0) {
        seed = t;
      }
      for (int k = 0; k < 4; ++k) {
        if (tet(t).v[k] != a && tet(t).v[k] != b &&
            complex_.constraint(t, k) != kNoConstraint) {
          if (!spend(4)) {
            return Insertion::kCapacity;
          }
          if ((!bounded && orient_with(t, k, position) != 0) || policy_ == BoundaryPolicy::kFixed) {
            return Insertion::kRefused;
          }
          if (kind == InsertKind::kInterior) {
            kind = InsertKind::kSubfacet;
          }
        }
      }
    }
    if (seed < 0 || (kind == InsertKind::kSubsegment && policy_ == BoundaryPolicy::kFixed)) {
      return Insertion::kRefused;
    }
    int32_t blocked = -1;
    int blocked_slot = -1;
    const int32_t origin = bounded ? seed : locate(position, seed, blocked, blocked_slot);
    if (origin < 0) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    const Insertion status = prepare(
        position, origin, kind == InsertKind::kSubsegment ? a : -1,
        kind == InsertKind::kSubsegment ? b : -1, cell_limit, bounded ? &construction : nullptr);
    if (status != Insertion::kOk) {
      abandon();
      return bounded && status == Insertion::kRefused
                 ? prepare_bounded_edge_star(a, b, position, source, construction, cell_limit)
                 : status;
    }
    for (int32_t t : cavity_) {
      if (!spend(4)) {
        abandon();
        return Insertion::kCapacity;
      }
      for (int32_t vertex : tet(t).v) {
        if (vertex >= 0 && protection(vertex) > 0.0 &&
            geometry::squared_distance(point(vertex), position) <
                protection(vertex) * protection(vertex)) {
          abandon();
          return Insertion::kRefused;
        }
      }
    }
    return Insertion::kOk;
  } catch (const std::bad_alloc&) {
    abandon();
    return Insertion::kCapacity;
  }

  // Prepares the insertion of p (not yet a vertex) located in `origin`:
  // the conflict region, crossing only constrained faces coplanar with p when
  // the policy allows splits, shrunk until star-shaped from p.  The cavity,
  // its boundary and crossed faces are then available for inspection.
  Insertion prepare(const double* p, int32_t origin, int32_t split_a = -1,
                    int32_t split_b = -1,
                    std::size_t cell_limit = std::numeric_limits<std::size_t>::max(),
                    const SourceWitness* construction = nullptr) try {
    const MemoryScope memory_scope(memory_owner_);
    pending_edge_star_ = false;
    pending_ = static_cast<int32_t>(vertex_count());
    pending_source_ = construction == nullptr ? SourceWitness{} : *construction;
    pending_source_authored_ = construction != nullptr;
    std::copy_n(p, 3, pending_point_);
    split_a_ = split_a;
    split_b_ = split_b;
    preparation_cell_limit_ = cell_limit;
    if (coincident(origin, p)) {
      return Insertion::kDuplicate;
    }
    Insertion status = grow(origin);
    if (status == Insertion::kOk) {
      status = shrink(origin);
    }
    if (status == Insertion::kOk) {
      status = collect();
    }
    return status;
  }
  catch (const std::bad_alloc&) {
    pending_ = kNoPending;
    return Insertion::kCapacity;
  }

  std::span<const int32_t> cavity() const { return cavity_; }
  std::span<const BoundaryFacet> cavity_boundary() const { return boundary_; }

  // Commits the prepared insertion; the new vertex gets `size`.
  Insertion commit(InsertKind kind, double size) try {
    if (pending_edge_star_) {
      pending_edge_star_ = false;
      int32_t inserted = -1;
      return split_edge_star(split_a_, split_b_, pending_point_, size,
                             segment_source(split_a_, split_b_), true,
                             &pending_source_, inserted);
    }
    const MemoryScope memory_scope(memory_owner_);
    EditTransaction transaction(*this, crossed_);
    if (vertex_count() >= max_vertices_) {
      return Insertion::kCapacity;
    }
    const int32_t p = pending_;
    SourceWitness witness;
    const Insertion admitted = pending_witness(kind, witness);
    if (admitted != Insertion::kOk) {
      return admitted;
    }
    pending_source_ = witness;
    NativeUnorderedMap<std::uint64_t, int32_t> staged_segments{
        NativeAllocator<std::pair<const std::uint64_t, int32_t>>(memory_owner_)};
    if (kind == InsertKind::kSubsegment) {
      const int32_t source = segment_source(split_a_, split_b_);
      if (source < 0) {
        return Insertion::kRefused;
      }
      staged_segments.emplace(undirected_edge(split_a_, p), source);
      staged_segments.emplace(undirected_edge(p, split_b_), source);
      segments_.reserve(segments_.size() + 2);
    }
    // Every buffer the vertex needs is reserved before anything is written.
    reserve_for(points_, points_.size() + 3);
    reserve_for(dimension_, dimension_.size() + 1);
    reserve_for(witnesses_, witnesses_.size() + 1);
    reserve_for(protection_, protection_.size() + 1);
    reserve_for(sizes_, sizes_.size() + 1);
    reserve_for(vertex_tet_, vertex_tet_.size() + 1);
    reserve_for(vertex_stamp_, vertex_stamp_.size() + 1);
    reserve_for(slot_reason_, complex_.tets.size() + boundary_.size());
    edit_.begin(EditLimits{std::numeric_limits<std::size_t>::max(),
                           std::numeric_limits<std::size_t>::max(), max_tetrahedra_});
    for (int32_t t : cavity_) {
      if (edit_.remove(t) != EditStatus::kOk) {
        return Insertion::kInternal;
      }
    }
    transaction.retire_constraints();
    EditStatus status = EditStatus::kOk;
    for (const BoundaryFacet& face : boundary_) {
      Tetrahedron cone = tet(face.tet);
      cone.v[face.slot] = p;
      int32_t constraints[4];
      for (int j = 0; j < 4; ++j) {
        constraints[j] = j == face.slot ? complex_.constraint(face.tet, face.slot)
                                        : new_face_source(cone.v, face.slot, j);
      }
      status = edit_.add(cone.v, constraints, complex_.region(face.tet));
      if (status != EditStatus::kOk) {
        break;
      }
    }
    if (status == EditStatus::kOk) {
      status = edit_.validate([&](const int32_t* v) { return positive(v); });
    }
    if (status == EditStatus::kOk) {
      status = edit_.commit();
    }
    if (status != EditStatus::kOk) {
      edit_.rollback();
      if (status == EditStatus::kInternal) {
        broken_ = true;
        return Insertion::kInternal;
      }
      return status == EditStatus::kCellLimit || status == EditStatus::kSlotLimit ||
                     status == EditStatus::kBufferLimit
                 ? Insertion::kCapacity
                 : Insertion::kRefused;
    }
    transaction.release();
    points_.insert(points_.end(), pending_point_, pending_point_ + 3);
    const int8_t dim = kind == InsertKind::kSubsegment ? 1
                       : (kind == InsertKind::kSubfacet || !crossed_.empty()) ? 2
                                                                              : 3;
    dimension_.push_back(dim);
    witnesses_.push_back(witness);
    protection_.push_back(0.0);
    sizes_.push_back(size);
    vertex_tet_.push_back(-1);
    vertex_stamp_.push_back(0U);
    pending_ = kNoPending;
    if (kind == InsertKind::kSubsegment) {
      segments_.erase(undirected_edge(split_a_, split_b_));
      segments_.insert(staged_segments.extract(undirected_edge(split_a_, p)));
      segments_.insert(staged_segments.extract(undirected_edge(p, split_b_)));
    }
    largest_cavity_ = std::max(largest_cavity_, cavity_.size());
    after_commit();
    return Insertion::kOk;
  }
  catch (const std::bad_alloc&) {
    return Insertion::kCapacity;
  }

  // Abandons a prepared insertion (nothing was written).
  void abandon() {
    pending_ = kNoPending;
    pending_edge_star_ = false;
  }

  std::span<const int32_t> created() const { return edit_.created(); }

  // ------------------------------------------------------- general edits
  // Bisects the complete edge star, including its ghost closure. Unlike a
  // Delaunay insertion this operation cannot delete neighboring vertices.
  Insertion split_edge(int32_t a, int32_t b, const double* position, double size,
                       int32_t& inserted) {
    return split_edge_relocate(a, b, position, position, size, inserted);
  }

  // The same source witness admission used by the eventual cavity commit,
  // without publishing a point or changing topology.
  Insertion edge_construction_witness(int32_t a, int32_t b, const double* position,
                                      SourceWitness& witness) {
    NativeVector<int32_t> star, ring;
    if (!edge_ring(a, b, star, ring)) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    bool constrained = false;
    for (int32_t t : star) {
      if (!finite(t)) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        constrained = constrained ||
                      (tet(t).v[k] != a && tet(t).v[k] != b &&
                       complex_.constraint(t, k) != kNoConstraint);
      }
    }
    return star_witness(a, b, star, position, segment_source(a, b), constrained, witness);
  }

  // The witness is strictly inside the represented old edge. Only the final
  // position is staged: no intermediate split vertex is ever published.
  Insertion split_edge_relocate(int32_t a, int32_t b, const double* split_position,
                                const double* position, double size,
                                int32_t& inserted) {
    inserted = -1;
    int32_t source = -1;
    bool on_edge = false;
    if (!std::isfinite(size) || size < 0.0 ||
        !split_positions_admitted(a, b, split_position, position, source, on_edge)) {
      return Insertion::kRefused;
    }
    return split_edge_star(a, b, position, size, source, on_edge, nullptr, inserted);
  }

  // Replaces the complete star of an unprotected edge by cones from one new
  // vertex at `position`, which need not lie on the edge. Constrained or
  // interface faces containing the edge must contain the position exactly
  // (the shared surface-preservation proof), so no segment, constrained
  // patch or region changes; the exact orientation owner validates children.
  Insertion insert_edge_star_vertex(int32_t a, int32_t b, const double* position,
                                    double size, int32_t& inserted) {
    inserted = -1;
    if (!live_vertex(a) || !live_vertex(b) || a == b || segment_source(a, b) >= 0 ||
        !std::isfinite(size) || size < 0.0 || !geometry::finite3(position) ||
        !coordinate_in_domain(position[0]) || !coordinate_in_domain(position[1]) ||
        !coordinate_in_domain(position[2])) {
      return Insertion::kRefused;
    }
    return split_edge_star(a, b, position, size, -1, false, nullptr, inserted);
  }

  // Every source facet in the edge star must contain the authored S and admit
  // its RNE carrier error, not merely the row chosen to encode the witness.
  Insertion admit_split_facets(int32_t a, int32_t b, const SourceWitness& witness,
                               std::span<const int32_t> star) {
    for (int32_t t : star) {
      if (!finite(t)) continue;
      for (int k = 0; k < 4; ++k) {
        const int32_t source = complex_.constraint(t, k);
        if (tet(t).v[k] == a || tet(t).v[k] == b || source == kNoConstraint) continue;
        const int32_t support[3] = {a, b, star_apex(t, k, a, b)};
        const int32_t row = source_ancestor(SourceStratum::kFacet, source, support);
        if (row < 0 || !witness_on_facet(witness, static_cast<std::size_t>(row))) {
          return Insertion::kRefused;
        }
        const double tolerance = face_tolerance_[static_cast<std::size_t>(row)];
        if (witness.deviation > tolerance) {
          source_refusals_.push_back(
              {{a, b}, SourceStratum::kFacet, row, witness.deviation, tolerance});
          return Insertion::kDeviation;
        }
      }
    }
    return Insertion::kOk;
  }

  // Bounded constrained split of represented edge (a, b): the correctly
  // rounded carrier of the exact witness at `fraction` between the source
  // parameters of a and b on the source row holding the edge (its segment
  // row, or the triangle row of a constrained face through it), strictly
  // inside (a, b) along its dominant axis. kOk when that row's declared bound
  // admits the certified deviation; kDeviation (recorded) above it; kRefused
  // for an unconstrained or fixed edge, a missing row or no distinct carrier.
  Insertion construct_source_split(int32_t a, int32_t b, double fraction, double* position,
                                   SourceWitness& witness) try {
    const MemoryScope memory_scope(memory_owner_);
    if (!live_vertex(a) || !live_vertex(b) || a == b || policy_ == BoundaryPolicy::kFixed ||
        !(fraction > 0.0 && fraction < 1.0)) {
      return Insertion::kRefused;
    }
    NativeVector<int32_t> star;
    NativeVector<int32_t> ring;
    if (!edge_ring(a, b, star, ring)) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    int axis = 0;
    for (int k = 1; k < 3; ++k) {
      if (std::abs(point(a)[k] - point(b)[k]) > std::abs(point(a)[axis] - point(b)[axis])) {
        axis = k;
      }
    }
    const double low = std::min(point(a)[axis], point(b)[axis]);
    const double high = std::max(point(a)[axis], point(b)[axis]);
    const auto construct = [&](SourceStratum stratum, int32_t row) {
      if (!spend(16)) {
        return Insertion::kCapacity;
      }
      const double* corners[3];
      row_corners(stratum, static_cast<std::size_t>(row), corners);
      double first[2];
      double second[2];
      if (!row_parameters(a, stratum, row, corners, first) ||
          !row_parameters(b, stratum, row, corners, second)) {
        return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
      }
      witness = SourceWitness{stratum, row, {0.0, 0.0}, 0.0};
      for (int k = 0; k < 2; ++k) {
        witness.parameters[k] = first[k] + fraction * (second[k] - first[k]);
      }
      clamp_source_parameters(stratum, witness.parameters);
      if (!source_carrier(corners, stratum, witness.parameters, position, witness.deviation) ||
          !(low < position[axis] && position[axis] < high)) {
        return Insertion::kRefused;
      }
      const double tolerance = row_tolerance(stratum, row);
      if (witness.deviation > tolerance) {
        source_refusals_.push_back({{a, b}, stratum, row, witness.deviation, tolerance});
        return Insertion::kDeviation;
      }
      return admit_split_facets(a, b, witness, star);
    };
    const int32_t segment = segment_source(a, b);
    if (segment >= 0) {
      const int32_t support[2] = {a, b};
      const int32_t row = source_ancestor(SourceStratum::kSegment, segment, support);
      return row < 0 ? Insertion::kRefused : construct(SourceStratum::kSegment, row);
    }
    for (int32_t t : star) {
      if (!finite(t)) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        const int32_t source = complex_.constraint(t, k);
        if (tet(t).v[k] == a || tet(t).v[k] == b || source == kNoConstraint) {
          continue;
        }
        const int32_t support[3] = {a, b, star_apex(t, k, a, b)};
        const int32_t row = source_ancestor(SourceStratum::kFacet, source, support);
        if (row >= 0) {
          const Insertion constructed = construct(SourceStratum::kFacet, row);
          if (constructed != Insertion::kRefused) return constructed;
        }
      }
    }
    return Insertion::kRefused;
  } catch (const std::bad_alloc&) {
    return Insertion::kCapacity;
  }

  // Commits a bounded split constructed by construct_source_split: the star
  // of (a, b) is bisected at the carrier, whose witness is recertified here.
  // Children inherit every region and constraint mark; the exact orientation
  // owner validates them, so tolerance never crosses a stratum or region.
  Insertion prepare_source_facet(int32_t t, int slot, const double* center,
                                double* position, SourceWitness& witness) {
    int32_t face[3];
    oriented_facet(tet(t).v, slot, face);
    const int32_t row = original_face_ancestor(face, complex_.constraint(t, slot));
    if (row < 0 || !spend(32)) return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    const double* corners[3];
    row_corners(SourceStratum::kFacet, static_cast<std::size_t>(row), corners);
    witness = SourceWitness{SourceStratum::kFacet, row, {0.0, 0.0}, 0.0};
    source_locator(corners, SourceStratum::kFacet, center, witness.parameters, false);
    if (!source_carrier(corners, SourceStratum::kFacet, witness.parameters, position,
                        witness.deviation)) return Insertion::kRefused;
    const double tolerance = row_tolerance(SourceStratum::kFacet, row);
    if (witness.deviation > tolerance) {
      source_refusals_.push_back({{face[0], face[1]}, SourceStratum::kFacet, row,
                                  witness.deviation, tolerance});
      return Insertion::kDeviation;
    }
    pending_ = static_cast<int32_t>(vertex_count());
    pending_source_ = witness;
    pending_source_authored_ = true;
    std::copy_n(position, 3, pending_point_);
    int32_t blocked = -1;
    int blocked_slot = -1;
    const int32_t located = locate(pending_point_, t, blocked, blocked_slot);
    if (located < 0) {
      abandon();
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    return prepare(position, located, -1, -1, std::numeric_limits<std::size_t>::max(), &witness);
  }

  Insertion split_edge_bounded(int32_t a, int32_t b, const double* position,
                               const SourceWitness& witness, double size, int32_t& inserted) {
    inserted = -1;
    if (!live_vertex(a) || !live_vertex(b) || a == b || !std::isfinite(size) || size < 0.0 ||
        witness.stratum == SourceStratum::kNone || witness.entity < 0 ||
        static_cast<std::size_t>(witness.entity) >=
            (witness.stratum == SourceStratum::kSegment ? original_segments_.size()
                                                        : original_faces_.size()) ||
        !geometry::finite3(position)) {
      return Insertion::kRefused;
    }
    const double* corners[3];
    row_corners(witness.stratum, static_cast<std::size_t>(witness.entity), corners);
    const double deviation =
        source_deviation(corners, witness.stratum, witness.parameters, position);
    const int32_t segment = segment_source(a, b);
    const auto& identity = witness.stratum == SourceStratum::kSegment
                               ? original_segments_[static_cast<std::size_t>(witness.entity)][2]
                               : original_faces_[static_cast<std::size_t>(witness.entity)][3];
    if (deviation < 0.0 || deviation != witness.deviation ||
        (witness.stratum == SourceStratum::kSegment) != (segment >= 0) ||
        (segment >= 0 && identity != segment) ||
        (deviation > 0.0 && !rounded_carrier(witness.stratum, witness.entity, witness, position))) {
      return Insertion::kRefused;
    }
    if (deviation > row_tolerance(witness.stratum, witness.entity)) {
      return Insertion::kDeviation;
    }
    NativeVector<int32_t> star;
    NativeVector<int32_t> ring;
    if (!edge_ring(a, b, star, ring)) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    const Insertion admitted = admit_split_facets(a, b, witness, star);
    if (admitted != Insertion::kOk) return admitted;
    for (int32_t seed : star) {
      if (!finite(seed)) continue;
      const int32_t proposal = static_cast<int32_t>(vertex_count());
      Insertion result = prepare(position, seed, a, b,
          std::numeric_limits<std::size_t>::max(), &witness);
      if (result == Insertion::kOk) {
        result = commit(segment >= 0 ? InsertKind::kSubsegment : InsertKind::kSubfacet, size);
      }
      if (result == Insertion::kOk) {
        inserted = proposal;
        return result;
      }
      abandon();
      if (result != Insertion::kRefused) return result;
      break;
    }
    return split_edge_star(a, b, position, size, segment, true, &witness, inserted);
  }

  // Witness of constrained vertex v moved to `position`, which the caller has
  // proved to lie exactly on every represented plane (and, on a curve, the
  // represented line) of its star; interior vertices carry none.
  Insertion relocation_witness(int32_t v, const double* position, SourceWitness& witness) {
    witness = SourceWitness{};
    if (dimension(v) >= 3) {
      return Insertion::kOk;
    }
    if (dimension(v) <= 1) {
      for (const auto& [edge, source] : segments_) {
        if (edge_low(edge) == v || edge_high(edge) == v) {
          const int32_t support[2] = {edge_low(edge), edge_high(edge)};
          return derive_witness(SourceStratum::kSegment, source, support, position, witness);
        }
      }
      if (dimension(v) == 1) {
        return Insertion::kRefused;
      }
    }
    if (!vertex_star(v, scratch_star_)) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    for (int32_t t : scratch_star_) {
      if (!finite(t)) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        const int32_t source = complex_.constraint(t, k);
        if (tet(t).v[k] == v || source == kNoConstraint) {
          continue;
        }
        int32_t f[3];
        oriented_facet(tet(t).v, k, f);
        return derive_witness(SourceStratum::kFacet, source, f, position, witness);
      }
    }
    return Insertion::kRefused;
  }
  void set_witness(int32_t v, const SourceWitness& witness) {
    witnesses_[static_cast<std::size_t>(v)] = witness;
  }

 private:
  struct Crossed;

  // Both direct and inspected star routes retain the same protection, source,
  // surface and work admission before their one atomic cavity transaction.
  Insertion edge_star_preparation(int32_t a, int32_t b, const double* position,
                                 int32_t source, bool on_edge,
                                 const SourceWitness* bounded,
                                 NativeVector<int32_t>& star,
                                 NativeVector<Crossed>& retired,
                                 bool& constrained, SourceWitness& witness) {
    NativeVector<int32_t> ring;
    if (!edge_ring(a, b, star, ring)) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    constrained = source >= 0;
    for (int32_t t : star) {
      for (int32_t v : tet(t).v) {
        if (v >= 0 && protection(v) > 0.0 &&
            geometry::squared_distance(point(v), position) < protection(v) * protection(v)) {
          return Insertion::kRefused;
        }
      }
      for (int k = 0; k < 4; ++k) {
        if (tet(t).v[k] != a && tet(t).v[k] != b &&
            complex_.constraint(t, k) != kNoConstraint) {
          constrained = true;
          if (t < tet(t).n[k]) {
            retired.push_back({t, k, complex_.constraint(t, k)});
          }
        }
      }
    }
    if (constrained && policy_ == BoundaryPolicy::kFixed) {
      return Insertion::kRefused;
    }
    if (!on_edge && !split_surface_preserved(a, b, star, position)) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    if (vertex_count() >= max_vertices_ ||
        !spend(static_cast<int64_t>(8 * star.size()))) {
      return Insertion::kCapacity;
    }
    if (bounded != nullptr) {
      witness = *bounded;
      return Insertion::kOk;
    }
    return star_witness(a, b, star, position, source, constrained, witness);
  }

  // Inspect the existing bounded complete-star route when its Delaunay
  // conflict preparation refuses. No source witness or cell budget is relaxed.
  Insertion prepare_bounded_edge_star(int32_t a, int32_t b, const double* position,
                                     int32_t source, const SourceWitness& bounded,
                                     std::size_t cell_limit) {
    NativeVector<int32_t> star;
    NativeVector<Crossed> retired;
    bool constrained = false;
    SourceWitness witness;
    const Insertion admitted = edge_star_preparation(
        a, b, position, source, true, &bounded, star, retired, constrained, witness);
    if (admitted != Insertion::kOk) {
      return admitted;
    }
    std::size_t finite_cells = 0;
    for (int32_t t : star) {
      finite_cells += finite(t) ? 1 : 0;
    }
    if (finite_cells > cell_limit / 3) {
      return Insertion::kCapacity;
    }
    boundary_.clear();
    for (int32_t t : star) {
      const Tetrahedron& old = tet(t);
      for (int endpoint : {a, b}) {
        const int slot = vertex_slot(old, endpoint);
        const int32_t neighbor = old.n[slot];
        boundary_.push_back({t, slot, neighbor, neighbor_slot(tet(neighbor), t)});
      }
    }
    cavity_ = std::move(star);
    crossed_ = std::move(retired);
    pending_ = static_cast<int32_t>(vertex_count());
    pending_source_ = witness;
    pending_source_authored_ = true;
    split_a_ = a;
    split_b_ = b;
    std::copy_n(position, 3, pending_point_);
    pending_edge_star_ = true;
    return Insertion::kOk;
  }

  // `bounded` carries a recertified ancestry-backed witness (the surface is
  // then bounded by that witness, not preserved exactly); otherwise the new
  // vertex's witness is derived from its exact represented support.
  Insertion split_edge_star(int32_t a, int32_t b, const double* position, double size,
                            int32_t source, bool on_edge, const SourceWitness* bounded,
                            int32_t& inserted) try {
    const MemoryScope memory_scope(memory_owner_);
    NativeVector<int32_t> star;
    NativeVector<Crossed> retired;
    bool constrained = false;
    SourceWitness witness;
    const Insertion admitted = edge_star_preparation(
        a, b, position, source, on_edge, bounded, star, retired, constrained, witness);
    if (admitted != Insertion::kOk) {
      return admitted;
    }
    EditTransaction transaction(*this, retired);
    NativeUnorderedMap<std::uint64_t, int32_t> staged_segments{
        NativeAllocator<std::pair<const std::uint64_t, int32_t>>(memory_owner_)};
    const int32_t p = static_cast<int32_t>(vertex_count());
    if (source >= 0) {
      staged_segments.emplace(undirected_edge(a, p), source);
      staged_segments.emplace(undirected_edge(p, b), source);
      segments_.reserve(segments_.size() + 2);
    }
    reserve_for(points_, points_.size() + 3);
    reserve_for(dimension_, dimension_.size() + 1);
    reserve_for(witnesses_, witnesses_.size() + 1);
    reserve_for(protection_, protection_.size() + 1);
    reserve_for(sizes_, sizes_.size() + 1);
    reserve_for(vertex_tet_, vertex_tet_.size() + 1);
    reserve_for(vertex_stamp_, vertex_stamp_.size() + 1);
    reserve_for(slot_reason_, complex_.tets.size() + 2 * star.size());
    pending_ = p;
    pending_source_ = witness;
    std::copy_n(position, 3, pending_point_);
    edit_.begin(EditLimits{star.size(), 2 * star.size(), max_tetrahedra_});
    EditStatus status = EditStatus::kOk;
    for (int32_t t : star) {
      status = edit_.remove(t);
      if (status != EditStatus::kOk) {
        break;
      }
      const Tetrahedron& old = tet(t);
      for (int endpoint : {a, b}) {
        int32_t child[4] = {old.v[0], old.v[1], old.v[2], old.v[3]};
        const int slot = vertex_slot(old, endpoint);
        child[slot] = p;
        int32_t marks[4];
        for (int k = 0; k < 4; ++k) {
          marks[k] = old.v[k] == (endpoint == a ? b : a)
                         ? kNoConstraint
                         : complex_.constraint(t, k);
        }
        status = edit_.add(child, marks, complex_.region(t));
        if (status != EditStatus::kOk) {
          break;
        }
      }
      if (status != EditStatus::kOk) {
        break;
      }
    }
    transaction.retire_constraints();
    if (status == EditStatus::kOk) {
      status = edit_.validate([&](const int32_t* v) { return positive(v); });
    }
    if (status == EditStatus::kOk) {
      status = edit_.commit();
    }
    if (status != EditStatus::kOk) {
      edit_.rollback();
      broken_ = broken_ || status == EditStatus::kInternal;
      return status == EditStatus::kCellLimit || status == EditStatus::kSlotLimit ||
                     status == EditStatus::kBufferLimit
                 ? Insertion::kCapacity
                 : status == EditStatus::kInternal ? Insertion::kInternal : Insertion::kRefused;
    }
    transaction.release();
    points_.insert(points_.end(), pending_point_, pending_point_ + 3);
    dimension_.push_back(source >= 0 ? 1 : constrained ? 2 : 3);
    witnesses_.push_back(witness);
    protection_.push_back(0.0);
    sizes_.push_back(size);
    vertex_tet_.push_back(-1);
    vertex_stamp_.push_back(0U);
    if (source >= 0) {
      segments_.erase(undirected_edge(a, b));
      segments_.insert(staged_segments.extract(undirected_edge(a, p)));
      segments_.insert(staged_segments.extract(undirected_edge(p, b)));
    }
    pending_ = kNoPending;
    largest_cavity_ = std::max(largest_cavity_, star.size());
    after_commit();
    inserted = p;
    return Insertion::kOk;
  }
  catch (const std::bad_alloc&) {
    pending_ = kNoPending;
    return Insertion::kCapacity;
  }

  // Third vertex of the face of finite tetrahedron t opposite slot k that
  // contains edge (a, b).
  int32_t star_apex(int32_t t, int k, int32_t a, int32_t b) const {
    for (int r = 0; r < 4; ++r) {
      const int32_t v = tet(t).v[r];
      if (r != k && v != a && v != b) {
        return v;
      }
    }
    return -1;
  }

  // A cross-row locator must encode the SAME exact S, never the off-plane P.
  // Non-dyadic row coordinates cannot use binary64-parameter witnesses.
  bool exact_row_parameter(const Expansion& numerator, const Expansion& denominator,
                           double& parameter) {
    std::uint64_t low = 0;
    std::uint64_t high = std::bit_cast<std::uint64_t>(1.0);
    const int denominator_sign = denominator.sign();
    if (denominator_sign == 0) return false;
    while (low <= high) {
      if (!spend(1)) return false;
      const std::uint64_t middle = low + (high - low) / 2;
      const double candidate = std::bit_cast<double>(middle);
      const int comparison = (numerator - denominator.scaled(candidate)).sign() * denominator_sign;
      if (comparison == 0) {
        parameter = candidate;
        return true;
      }
      if (comparison > 0) low = middle + 1;
      else {
        if (middle == 0) break;
        high = middle - 1;
      }
    }
    return false;
  }

  bool row_parameters(int32_t v, SourceStratum stratum, int32_t row,
                      const double* const* corners, double* parameters) {
    const SourceWitness& own = witnesses_[static_cast<std::size_t>(v)];
    if (own.deviation == 0.0) {
      source_locator(corners, stratum, point(v), parameters);
      return true;
    }
    if (own.stratum == stratum && own.entity == row) {
      parameters[0] = own.parameters[0];
      parameters[1] = own.parameters[1];
      return true;
    }
    const double* own_corners[3];
    row_corners(own.stratum, static_cast<std::size_t>(own.entity), own_corners);
    Expansion source[3], u[3], r[3];
    if (!source_point(own_corners, own.stratum, own.parameters, source)) return false;
    for (int axis = 0; axis < 3; ++axis) {
      u[axis] = Expansion::difference(corners[1][axis], corners[0][axis]);
      r[axis] = source[axis] - Expansion(corners[0][axis]);
    }
    parameters[1] = 0.0;
    if (stratum == SourceStratum::kSegment) {
      for (int axis = 0; axis < 3; ++axis) {
        if (u[axis].sign() != 0) return exact_row_parameter(r[axis], u[axis], parameters[0]);
      }
      return false;
    }
    Expansion w[3];
    for (int axis = 0; axis < 3; ++axis) {
      w[axis] = Expansion::difference(corners[2][axis], corners[0][axis]);
    }
    for (int drop = 0; drop < 3; ++drop) {
      const int x = (drop + 1) % 3, y = (drop + 2) % 3;
      const Expansion area = u[x] * w[y] - u[y] * w[x];
      if (area.sign() != 0) {
        return exact_row_parameter(r[x] * w[y] - r[y] * w[x], area, parameters[0]) &&
               exact_row_parameter(u[x] * r[y] - u[y] * r[x], area, parameters[1]);
      }
    }
    return false;
  }

  // Whether the position is the correctly rounded carrier of the witness on
  // source row `row`: every bounded vertex is, so its exact source point is
  // the authoritative coordinate and the binary64 point its RNE carrier.
  bool rounded_carrier(SourceStratum stratum, int32_t row, const SourceWitness& witness,
                       const double* position) const {
    const double* corners[3];
    row_corners(stratum, static_cast<std::size_t>(row), corners);
    double carrier[3];
    double deviation = 0.0;
    return source_carrier(corners, stratum, witness.parameters, carrier, deviation) &&
           std::equal(carrier, carrier + 3, position);
  }

  // Witness of a new or moved constrained vertex lying exactly on its
  // represented support (two segment or three face vertices of one source).
  // Over an exact support the vertex is exactly on its source: the witness
  // names the source row of that source containing it exactly (zero
  // deviation; parameters locate it on the row). Otherwise the source row
  // holding the whole support names the witness, the position must be the
  // RNE carrier of the located witness, and that row's declared bound must
  // admit its certified deviation.
  Insertion derive_witness(SourceStratum stratum, int32_t source,
                           std::span<const int32_t> support, const double* position,
                           SourceWitness& witness) {
    witness = SourceWitness{stratum, -1, {0.0, 0.0}, 0.0};
    if (source < 0) {
      return Insertion::kRefused;
    }
    bool exact = true;
    for (int32_t v : support) {
      exact = exact && witnesses_[static_cast<std::size_t>(v)].deviation == 0.0;
    }
    if (!spend(static_cast<int64_t>(4 * support.size()))) {
      return Insertion::kCapacity;
    }
    const int32_t row =
        exact ? position_row(stratum, source, position) : source_ancestor(stratum, source, support);
    if (row < 0) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    const double* corners[3];
    row_corners(stratum, static_cast<std::size_t>(row), corners);
    witness.entity = row;
    source_locator(corners, stratum, position, witness.parameters);
    witness.deviation = source_deviation(corners, stratum, witness.parameters, position);
    if (witness.deviation < 0.0 || (exact && witness.deviation != 0.0) ||
        (witness.deviation > 0.0 && !rounded_carrier(stratum, row, witness, position))) {
      return Insertion::kRefused;
    }
    const double tolerance = row_tolerance(stratum, row);
    if (witness.deviation > tolerance) {
      source_refusals_.push_back({{support[0], support[1]}, stratum, row, witness.deviation,
                                  tolerance});
      return Insertion::kDeviation;
    }
    return Insertion::kOk;
  }

  // First source row of `source` containing the position exactly, or -1.
  int32_t position_row(SourceStratum stratum, int32_t source, const double* position) {
    const auto& rows = stratum == SourceStratum::kSegment ? segment_rows_ : face_rows_;
    const std::pair<int32_t, int32_t> probe{source, std::numeric_limits<int32_t>::min()};
    for (auto row = std::lower_bound(rows.begin(), rows.end(), probe);
         row != rows.end() && row->first == source; ++row) {
      if (!spend(4)) {
        return -1;
      }
      const double* corners[3];
      row_corners(stratum, static_cast<std::size_t>(row->second), corners);
      if (on_source_entity(corners, stratum, position)) {
        return row->second;
      }
    }
    return -1;
  }

  // Witness of the vertex splitting edge (a, b) with star `star`: its
  // segment, else a constrained face through the edge, else none.
  Insertion star_witness(int32_t a, int32_t b, std::span<const int32_t> star,
                         const double* position, int32_t source, bool constrained,
                         SourceWitness& witness) {
    witness = SourceWitness{};
    if (source >= 0) {
      const int32_t support[2] = {a, b};
      return derive_witness(SourceStratum::kSegment, source, support, position, witness);
    }
    if (!constrained) {
      return Insertion::kOk;
    }
    Insertion result = Insertion::kRefused;
    for (int32_t t : star) {
      if (!finite(t)) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        const int32_t face_source = complex_.constraint(t, k);
        if (tet(t).v[k] == a || tet(t).v[k] == b || face_source == kNoConstraint) {
          continue;
        }
        const int32_t support[3] = {a, b, star_apex(t, k, a, b)};
        result = derive_witness(SourceStratum::kFacet, face_source, support, position, witness);
        if (result != Insertion::kRefused) {
          return result;
        }
      }
    }
    return result;
  }

  // Witness of the pending Delaunay point: on its subsegment, or on a
  // crossed or cavity-boundary constrained face containing it (it lies
  // exactly in the plane of every crossed face), or none for an interior
  // point.
  Insertion pending_witness(InsertKind kind, SourceWitness& witness) {
    witness = SourceWitness{};
    if (pending_source_authored_) {
      witness = pending_source_;
      return Insertion::kOk;
    }
    const double* p = pending_point_;
    if (kind == InsertKind::kSubsegment) {
      const int32_t support[2] = {split_a_, split_b_};
      return derive_witness(SourceStratum::kSegment, segment_source(split_a_, split_b_),
                            support, p, witness);
    }
    if (kind != InsertKind::kSubfacet && crossed_.empty()) {
      return Insertion::kOk;
    }
    const auto attempt = [&](int32_t t, int32_t slot, int32_t source) {
      int32_t f[3];
      oriented_facet(tet(t).v, slot, f);
      if (source == kNoConstraint || f[0] < 0 || f[1] < 0 || f[2] < 0) {
        return Insertion::kRefused;
      }
      const double* triangle[3] = {point(f[0]), point(f[1]), point(f[2])};
      int side = 0;
      return locate_on_triangle(p, triangle, side) == kFeatureNone
                 ? Insertion::kRefused
                 : derive_witness(SourceStratum::kFacet, source, f, p, witness);
    };
    for (const Crossed& face : crossed_) {
      const Insertion result = attempt(face.tet, face.slot, face.source);
      if (result != Insertion::kRefused) {
        return result;
      }
    }
    for (const BoundaryFacet& face : boundary_) {
      const Insertion result =
          attempt(face.tet, face.slot, complex_.constraint(face.tet, face.slot));
      if (result != Insertion::kRefused) {
        return result;
      }
    }
    return Insertion::kRefused;
  }

 public:
  // Contract a removable boundary Steiner's complete closed star. The caller
  // proves the volume link condition; each exact planar source/material patch
  // proves its own link, oriented boundary-chain and area conservation. Actual
  // represented planes, not equal source labels alone, establish equivalence.
  EditStatus collapse_boundary_edge(int32_t remove, int32_t keep,
                                   std::span<const int32_t> star,
                                   std::span<const int32_t> keep_star) {
    const MemoryScope memory_scope(memory_owner_);
    if (policy_ != BoundaryPolicy::kConforming || !live_vertex(remove) ||
        !live_vertex(keep) || remove == keep || dimension(remove) < 1 ||
        dimension(remove) > 2 || dimension(keep) > dimension(remove) ||
        protection(remove) > 0.0) {
      return EditStatus::kConstraintViolation;
    }
    if (!spend(static_cast<int64_t>(8 * (star.size() + keep_star.size()) +
                                    segments_.size()))) {
      return EditStatus::kBufferLimit;
    }

    // A curve contraction replaces exactly two same-source collinear intervals
    // by their exact union. A junction, bend, source transition or existing
    // competing edge is not removable even if its source label happens to agree.
    int32_t curve_other = -1;
    int32_t curve_source = -1;
    int curve_degree = 0;
    bool curve_keep = false;
    for (const auto& [edge, source] : segments_) {
      const int32_t a = edge_low(edge), b = edge_high(edge);
      if (a != remove && b != remove) {
        continue;
      }
      if (dimension(remove) != 1) {
        return EditStatus::kConstraintViolation;
      }
      if (curve_degree > 0 && source != curve_source) {
        return EditStatus::kConstraintViolation;
      }
      curve_source = source;
      ++curve_degree;
      const int32_t other = a == remove ? b : a;
      if (other == keep) {
        curve_keep = true;
      } else {
        curve_other = other;
      }
    }
    NativeUnorderedMap<std::uint64_t, int32_t> staged_segments{
        NativeAllocator<std::pair<const std::uint64_t, int32_t>>(memory_owner_)};
    if (dimension(remove) == 1) {
      if (!spend(8)) {
        return EditStatus::kBufferLimit;
      }
      if (curve_degree != 2 || !curve_keep || curve_other < 0 ||
          segment_source(keep, curve_other) >= 0 ||
          !collinear3d(point(keep), point(curve_other), point(remove))) {
        return EditStatus::kConstraintViolation;
      }
      bool strict = false;
      for (int axis = 0; axis < 3; ++axis) {
        if (point(keep)[axis] != point(curve_other)[axis]) {
          strict = std::min(point(keep)[axis], point(curve_other)[axis]) < point(remove)[axis] &&
                   point(remove)[axis] < std::max(point(keep)[axis], point(curve_other)[axis]);
          break;
        }
      }
      if (!strict) {
        return EditStatus::kConstraintViolation;
      }
      staged_segments.emplace(undirected_edge(keep, curve_other), curve_source);
      segments_.reserve(segments_.size() + 1);
    }

    using Stratum = std::array<int32_t, 3>;  // exact source plane, oriented region pair
    using LinkSimplex = std::array<int32_t, 2>;
    struct SurfaceProof {
      Expansion area_delta;
      NativeSet<LinkSimplex> left, right, edge;
      NativeMap<std::uint64_t, int64_t> boundary_delta;
    };
    struct FaceProof {
      Stratum stratum;
      int32_t source;
      int axis;
      int reference_sign;
      int sides[2] = {0, 0};
    };
    struct SourcePlane {
      std::array<int32_t, 3> vertices;
      int32_t source;
      int axis;
      int reference_sign;
    };
    // These indices are local proof keys only; persistent source identities and
    // original triangle/segment ancestry storage are never rewritten. A face
    // spanning original triangles truthfully has no single triangle ancestor.
    NativeVector<SourcePlane> planes;
    NativeMap<Stratum, SurfaceProof> surfaces;
    NativeMap<FaceKey, FaceProof> faces;
    NativeVector<Crossed> retired;
    const auto projected_area = [&](const int32_t* f, int axis) {
      const int x = (axis + 1) % 3, y = (axis + 2) % 3;
      const double a[2] = {point(f[0])[x], point(f[0])[y]};
      const double b[2] = {point(f[1])[x], point(f[1])[y]};
      const double c[2] = {point(f[2])[x], point(f[2])[y]};
      return orient2d_exact(a, b, c);
    };
    const auto describe = [&](int32_t t, int slot, const int32_t* f,
                              Stratum& stratum, int& axis, int& reference_sign,
                              bool add_plane) {
      if (!spend(8)) {
        return false;
      }
      const int32_t source = complex_.constraint(t, slot);
      std::size_t plane = 0;
      for (; plane < planes.size(); ++plane) {
        if (!spend(4)) {
          return false;
        }
        const auto& represented = planes[plane];
        if (represented.source != source) {
          continue;
        }
        const auto& v = represented.vertices;
        if (orient3d(point(v[0]), point(v[1]), point(v[2]), point(f[0])) == 0 &&
            orient3d(point(v[0]), point(v[1]), point(v[2]), point(f[1])) == 0 &&
            orient3d(point(v[0]), point(v[1]), point(v[2]), point(f[2])) == 0) {
          break;
        }
      }
      if (plane == planes.size()) {
        if (!add_plane) {
          stratum = {-1, kNoRegion, kNoRegion};
          return true;  // unrelated keep-star patch cannot enter the intersection
        }
        reference_sign = 0;
        for (axis = 0; axis < 3; ++axis) {
          reference_sign = projected_area(f, axis).sign();
          if (reference_sign != 0) {
            break;
          }
        }
        if (reference_sign == 0) {
          return false;
        }
        planes.push_back({{f[0], f[1], f[2]}, source, axis, reference_sign});
      } else {
        axis = planes[plane].axis;
        reference_sign = planes[plane].reference_sign;
      }
      const int sign = projected_area(f, axis).sign();
      if (sign == 0) {
        return false;
      }
      const int32_t mine = complex_.region(t);
      const int32_t theirs = complex_.region(tet(t).n[slot]);
      const int32_t identity = static_cast<int32_t>(plane);
      stratum = sign == reference_sign ? Stratum{identity, mine, theirs}
                                      : Stratum{identity, theirs, mine};
      return true;
    };
    const auto add_link = [](const int32_t* f, int32_t a, int32_t b,
                             NativeSet<LinkSimplex>& link) {
      int32_t v[2];
      int n = 0;
      for (int i = 0; i < 3; ++i) {
        if (f[i] != a && f[i] != b) {
          v[n++] = f[i];
        }
      }
      for (int mask = 0; mask < (1 << n); ++mask) {
        LinkSimplex simplex{kDeadVertex, kDeadVertex};
        int count = 0;
        for (int i = 0; i < n; ++i) {
          if ((mask & (1 << i)) != 0) {
            simplex[static_cast<std::size_t>(count++)] = v[i];
          }
        }
        std::sort(simplex.begin(), simplex.end());
        link.insert(simplex);
      }
    };
    const auto add_boundary = [](const int32_t* f, int64_t coefficient,
                                 NativeMap<std::uint64_t, int64_t>& boundary) {
      for (int i = 0; i < 3; ++i) {
        const int32_t a = f[i], b = f[(i + 1) % 3];
        if (a != b) {
          boundary[undirected_edge(a, b)] += a < b ? coefficient : -coefficient;
        }
      }
    };
    for (int32_t t : star) {
      if (!finite(t)) {
        continue;
      }
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t source = complex_.constraint(t, slot);
        const int32_t other = tet(t).n[slot];
        if (tet(t).v[slot] == remove || source == kNoConstraint ||
            (finite(other) && other < t)) {
          continue;
        }
        int32_t f[3];
        oriented_facet(tet(t).v, slot, f);
        Stratum stratum;
        int axis, reference_sign;
        if (!describe(t, slot, f, stratum, axis, reference_sign, true) ||
            orient3d(point(f[0]), point(f[1]), point(f[2]), point(keep)) != 0) {
          return work_exhausted_ ? EditStatus::kBufferLimit : EditStatus::kConstraintViolation;
        }
        SurfaceProof& proof = surfaces[stratum];
        add_link(f, remove, kDeadVertex, proof.left);
        const bool edge_face = std::find(f, f + 3, keep) != f + 3;
        if (edge_face) {
          add_link(f, remove, keep, proof.edge);
        }
        const Expansion before = projected_area(f, axis);
        const int64_t orientation = before.sign() == reference_sign ? 1 : -1;
        add_boundary(f, orientation, proof.boundary_delta);
        for (int32_t& v : f) {
          if (v == remove) {
            v = keep;
          }
        }
        const Expansion after = projected_area(f, axis);
        if ((!edge_face && after.sign() != before.sign()) ||
            (edge_face && !after.is_zero())) {
          return EditStatus::kConstraintViolation;
        }
        if (!edge_face) {
          add_boundary(f, -orientation, proof.boundary_delta);
        }
        proof.area_delta = before.sign() == reference_sign
                               ? proof.area_delta + before - after
                               : proof.area_delta + after - before;
        if (!edge_face &&
            !faces.emplace(FaceKey::of(f[0], f[1], f[2]),
                           FaceProof{stratum, source, axis, reference_sign}).second) {
          return EditStatus::kConstraintViolation;
        }
        retired.push_back({t, slot, source});
      }
    }
    if (surfaces.empty()) {
      return EditStatus::kConstraintViolation;
    }
    for (int32_t t : keep_star) {
      if (!finite(t)) {
        continue;
      }
      for (int slot = 0; slot < 4; ++slot) {
        if (tet(t).v[slot] == keep || complex_.constraint(t, slot) == kNoConstraint ||
            (finite(tet(t).n[slot]) && tet(t).n[slot] < t)) {
          continue;
        }
        int32_t f[3];
        oriented_facet(tet(t).v, slot, f);
        Stratum stratum;
        int axis, reference_sign;
        if (!describe(t, slot, f, stratum, axis, reference_sign, false)) {
          return work_exhausted_ ? EditStatus::kBufferLimit : EditStatus::kConstraintViolation;
        }
        const auto found = surfaces.find(stratum);
        if (found != surfaces.end()) {
          add_link(f, keep, kDeadVertex, found->second.right);
        }
      }
    }
    for (const auto& [stratum, proof] : surfaces) {
      if (!proof.area_delta.is_zero()) {
        return EditStatus::kConstraintViolation;
      }
      NativeSet<LinkSimplex> common;
      std::set_intersection(proof.left.begin(), proof.left.end(),
                            proof.right.begin(), proof.right.end(),
                            std::inserter(common, common.end()));
      if (common != proof.edge) {
        return EditStatus::kConstraintViolation;
      }
      // Equality of oriented planar boundaries proves equality of their
      // represented patches, including coplanar source-ID transitions. Residual
      // edges may use different vertex subdivisions; compare exact collinear
      // interval chains, not vertex keys or approximate lengths.
      struct Line {
        int32_t a, b;
        int axis;
        NativeMap<double, int64_t> events;
      };
      NativeVector<Line> lines;
      for (const auto& [edge, coefficient] : proof.boundary_delta) {
        if (coefficient == 0) {
          continue;
        }
        const int32_t a = edge_low(edge), b = edge_high(edge);
        std::size_t line = 0;
        for (; line < lines.size(); ++line) {
          if (!spend(6)) {
            return EditStatus::kBufferLimit;
          }
          if (collinear3d(point(lines[line].a), point(lines[line].b), point(a)) &&
              collinear3d(point(lines[line].a), point(lines[line].b), point(b))) {
            break;
          }
        }
        if (line == lines.size()) {
          int axis = 0;
          while (axis < 3 && point(a)[axis] == point(b)[axis]) {
            ++axis;
          }
          if (axis == 3) {
            return EditStatus::kConstraintViolation;
          }
          lines.push_back(Line{a, b, axis, {}});
        }
        const int axis = lines[line].axis;
        lines[line].events[point(a)[axis]] += coefficient;
        lines[line].events[point(b)[axis]] -= coefficient;
      }
      for (const Line& line : lines) {
        for (const auto& [coordinate, coefficient] : line.events) {
          if (coefficient != 0) {
            return EditStatus::kConstraintViolation;
          }
        }
      }
    }

    // Every surviving constrained triangle must have exactly its original
    // two oriented material sides, including the unbounded ghost side.
    const auto check_side = [&](const int32_t* f, int32_t region, FaceProof& proof) {
      const int sign = projected_area(f, proof.axis).sign();
      if (sign == 0) {
        return false;
      }
      const int side = sign == proof.reference_sign ? 0 : 1;
      return region == proof.stratum[static_cast<std::size_t>(side + 1)] &&
             ++proof.sides[side] == 1;
    };
    EditTransaction transaction(*this, retired);
    reserve_for(slot_reason_, complex_.tets.size() + star.size());
    edit_.begin(EditLimits{star.size(), star.size(), max_tetrahedra_});
    bool curve_represented = dimension(remove) != 1;
    std::size_t finite_survivors = 0;
    for (int32_t t : star) {
      EditStatus status = edit_.remove(t);
      if (status != EditStatus::kOk) {
        return status;
      }
      const Tetrahedron& old = tet(t);
      const int remove_slot = vertex_slot(old, remove);
      if (remove_slot < 0) {
        return EditStatus::kNotLive;
      }
      // The shell is unchanged. Account for a source triangle whose mapped
      // second side belongs to an untouched shell neighbor.
      int32_t shell[3];
      oriented_facet(old.v, remove_slot, shell);
      const auto shell_proof = FaceKey::of(shell[0], shell[1], shell[2]);
      const auto found_shell = faces.find(shell_proof);
      if (found_shell != faces.end()) {
        const int32_t outside = old.n[remove_slot];
        const int back = neighbor_slot(tet(outside), t);
        if (back < 0) {
          return EditStatus::kNonManifold;
        }
        oriented_facet(tet(outside).v, back, shell);
        if (!check_side(shell, complex_.region(outside), found_shell->second)) {
          return EditStatus::kConstraintViolation;
        }
      }
      if (vertex_slot(old, keep) >= 0) {
        continue;
      }
      int32_t v[4] = {old.v[0], old.v[1], old.v[2], old.v[3]};
      v[remove_slot] = keep;
      int32_t marks[4] = {kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint};
      for (int slot = 0; slot < 4; ++slot) {
        int32_t f[3];
        oriented_facet(v, slot, f);
        const auto found = faces.find(FaceKey::of(f[0], f[1], f[2]));
        if (found != faces.end()) {
          if (!check_side(f, complex_.region(t), found->second)) {
            return EditStatus::kConstraintViolation;
          }
          marks[slot] = found->second.source;
        }
      }
      if (curve_other >= 0 && finite(t) &&
          std::find(v, v + 4, curve_other) != v + 4) {
        curve_represented = true;
      }
      status = edit_.add(v, marks, complex_.region(t));
      if (status != EditStatus::kOk) {
        return status;
      }
      finite_survivors += finite(t) ? 1 : 0;
    }
    if (finite_survivors == 0 || !curve_represented) {
      return EditStatus::kConstraintViolation;
    }
    for (const auto& [key, proof] : faces) {
      if (proof.sides[0] != 1 || proof.sides[1] != 1) {
        return EditStatus::kConstraintViolation;
      }
      edit_.preserve(key.v[0], key.v[1], key.v[2], proof.source);
    }
    if (!spend(static_cast<int64_t>(16 * star.size()))) {
      return EditStatus::kBufferLimit;
    }
    transaction.retire_constraints();
    EditStatus status = edit_.validate([&](const int32_t* v) { return positive(v); });
    if (status == EditStatus::kOk) {
      status = edit_.commit();
    }
    if (status != EditStatus::kOk) {
      broken_ = broken_ || status == EditStatus::kInternal;
      return status;
    }
    transaction.release();
    if (curve_other >= 0) {
      segments_.erase(undirected_edge(remove, keep));
      segments_.erase(undirected_edge(remove, curve_other));
      segments_.insert(staged_segments.extract(undirected_edge(keep, curve_other)));
    }
    largest_cavity_ = std::max(largest_cavity_, star.size());
    after_commit();
    return EditStatus::kOk;
  }

  class PreparedReplacement {
   public:
    PreparedReplacement() = default;
    PreparedReplacement(const PreparedReplacement&) = delete;
    PreparedReplacement& operator=(const PreparedReplacement&) = delete;
    PreparedReplacement(PreparedReplacement&& other) noexcept
        : owner_(other.owner_), generation_(other.generation_),
          revision_(other.revision_), status_(other.status_) { other.owner_ = nullptr; }
    PreparedReplacement& operator=(PreparedReplacement&& other) noexcept {
      if (this != &other) {
        discard();
        owner_ = other.owner_;
        other.owner_ = nullptr;
        generation_ = other.generation_;
        revision_ = other.revision_;
        status_ = other.status_;
      }
      return *this;
    }
    ~PreparedReplacement() { discard(); }
    EditStatus status() const noexcept { return status_; }
    EditStatus commit() {
      if (owner_ == nullptr || status_ != EditStatus::kOk ||
          owner_->accepted_generation_ != generation_ ||
          owner_->edit_.staging_revision() != revision_) {
        return EditStatus::kBoundaryMismatch;
      }
      TetMesh* owner = owner_;
      const EditStatus result = owner->commit_prepared_replacement();
      owner_ = nullptr;
      return result;
    }
   private:
    friend class TetMesh;
    explicit PreparedReplacement(EditStatus status) noexcept : status_(status) {}
    PreparedReplacement(TetMesh& owner, EditStatus status) noexcept
        : owner_(&owner), generation_(owner.accepted_generation_),
          revision_(owner.edit_.staging_revision()), status_(status) {}
    void discard() noexcept {
      if (owner_ != nullptr && owner_->edit_.staging_revision() == revision_) {
        owner_->edit_.rollback();
      }
      owner_ = nullptr;
    }
    TetMesh* owner_ = nullptr;
    int64_t generation_ = -1;
    uint64_t revision_ = 0;
    EditStatus status_ = EditStatus::kNotLive;
  };

  // The one existing edit_ retains the validated source/region/cavity proof.
  // The scoped token never copies a mesh, source bank or allocation arena.
  PreparedReplacement prepare_replacement(std::span<const int32_t> removed,
                     std::span<const std::array<int32_t, 4>> proposed) {
    const MemoryScope memory_scope(memory_owner_);
    EditTransaction transaction(*this, {});
    if (removed.empty() || proposed.empty()) return {};
    const int32_t region = complex_.region(removed.front());
    for (int32_t t : removed) {
      if (complex_.region(t) != region) return PreparedReplacement(EditStatus::kConstraintViolation);
    }
    reserve_for(slot_reason_, complex_.tets.size() + proposed.size());
    edit_.begin(EditLimits{std::numeric_limits<std::size_t>::max(),
                           std::numeric_limits<std::size_t>::max(), max_tetrahedra_});
    for (int32_t t : removed) {
      const EditStatus status = edit_.remove(t);
      if (status != EditStatus::kOk) return PreparedReplacement(status);
    }
    for (const auto& v : proposed) {
      const EditStatus status = edit_.add(v.data(), nullptr, region);
      if (status != EditStatus::kOk) return PreparedReplacement(status);
    }
    const EditStatus status = edit_.validate([&](const int32_t* v) { return positive(v); });
    if (status != EditStatus::kOk) {
      broken_ = broken_ || status == EditStatus::kInternal;
      return PreparedReplacement(status);
    }
    transaction.release();
    return PreparedReplacement(*this, status);
  }

  // Replaces `removed` by `proposed` tetrahedra (region of the first removed
  // tetrahedron, no new constraints) through one validated transaction.
  EditStatus replace(std::span<const int32_t> removed,
                     std::span<const std::array<int32_t, 4>> proposed) {
    auto prepared = prepare_replacement(removed, proposed);
    return prepared.status() == EditStatus::kOk ? prepared.commit() : prepared.status();
  }

  // Marks a vertex removed after an edit that deleted its star; a retired
  // vertex carries no source witness.
  void retire_vertex(int32_t v) {
    dimension_[static_cast<std::size_t>(v)] = -1;
    vertex_tet_[static_cast<std::size_t>(v)] = -1;
    witnesses_[static_cast<std::size_t>(v)] = SourceWitness{};
  }

  // ----------------------------------------------------------- evidence
  // Canonical faces: (a, b, c, source) per constrained finite face, boundary
  // faces oriented outward, interfaces outward from the lower region.
  template <class Allocator>
  void collect_faces(std::vector<std::array<int32_t, 4>, Allocator>& faces) const {
    const MemoryScope memory_scope(memory_owner_);
    faces.clear();
    for (std::size_t s = 0; s < complex_.tets.size(); ++s) {
      const auto t = static_cast<int32_t>(s);
      const Tetrahedron& current = tet(t);
      if (current.v[0] == kDeadVertex || is_ghost(current)) {
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        const int32_t source = complex_.constraint(t, k);
        if (source == kNoConstraint) {
          continue;
        }
        const int32_t other = current.n[k];
        const bool ghost = is_ghost(tet(other));
        const int32_t mine = complex_.region(t);
        const int32_t theirs = ghost ? kNoRegion : complex_.region(other);
        // Each face once: from its finite side, from the lower region, or
        // from the lower slot within one region.
        if (!ghost && (theirs < mine || (theirs == mine && other < t))) {
          continue;
        }
        int32_t facet[3];
        oriented_facet(current.v, k, facet);
        // oriented_facet faces into t; flip for the outward normal.
        std::swap(facet[1], facet[2]);
        if (!ghost && theirs == mine) {
          const FaceKey key = FaceKey::of(facet[0], facet[1], facet[2]);
          facet[0] = key.v[0];
          facet[1] = key.v[1];
          facet[2] = key.v[2];
        }
        canonicalize_cell(facet, nullptr, 3);
        faces.push_back({facet[0], facet[1], facet[2], source});
      }
    }
    std::sort(faces.begin(), faces.end());
  }

  // Canonical finite cells (even permutation from the smallest vertex),
  // lexicographically sorted, with regions.
  template <class Allocator>
  void collect_cells(std::vector<std::array<int32_t, 5>, Allocator>& cells) const {
    const MemoryScope memory_scope(memory_owner_);
    cells.clear();
    for (std::size_t s = 0; s < complex_.tets.size(); ++s) {
      const Tetrahedron& current = complex_.tets[s];
      if (current.v[0] == kDeadVertex || is_ghost(current)) {
        continue;
      }
      int32_t v[4] = {current.v[0], current.v[1], current.v[2], current.v[3]};
      canonicalize_cell(v, nullptr, 4);
      cells.push_back({v[0], v[1], v[2], v[3], complex_.region(static_cast<int32_t>(s))});
    }
    std::sort(cells.begin(), cells.end());
  }

  std::size_t retained_bytes() const noexcept { return memory_owner_->live_bytes(); }

 private:
  friend class MeasuredTetStage;
  EditStatus commit_prepared_replacement() {
    const MemoryScope memory_scope(memory_owner_);
    EditTransaction transaction(*this, {});
    const EditStatus status = edit_.commit();
    if (status != EditStatus::kOk) {
      broken_ = broken_ || status == EditStatus::kInternal;
      return status;
    }
    transaction.release();
    after_commit();
    return status;
  }
  static MemoryOwner select_memory_owner(std::size_t bytes) {
    MemoryOwner owner = scratch_memory_owner();
    if (owner && (bytes == std::numeric_limits<std::size_t>::max() ||
                  owner->limit_bytes() == bytes)) {
      return owner;
    }
    return std::make_shared<BoundedMemoryResource>(bytes);
  }
  TetMesh(BoundaryPolicy policy, int64_t max_vertices, int64_t max_tetrahedra,
          MemoryOwner owner)
      : memory_owner_(std::move(owner)),
        policy_(policy),
        max_vertices_(max_vertices),
        max_tetrahedra_(max_tetrahedra),
        complex_(kMaxTetrahedronSlots, memory_owner_),
        edit_(complex_) {}
  static constexpr int32_t kNoPending = std::numeric_limits<int32_t>::min();
  static constexpr std::uint8_t kInCavity = 1;

  struct Crossed {
    int32_t tet;
    int32_t slot;
    int32_t source;
  };

  // Entirely allocation-free, including restoration after a refused request
  // during validation/reservation while constraint marks are retired.
  class EditTransaction {
   public:
    EditTransaction(TetMesh& mesh, std::span<const Crossed> retired) noexcept
        : mesh_(mesh), retired_(retired) {}
    ~EditTransaction() {
      if (active_) {
        mesh_.edit_.rollback();
        if (marks_retired_) {
          for (const Crossed& face : retired_) {
            mesh_.complex_.set_facet_constraint(face.tet, face.slot, face.source);
          }
        }
        mesh_.transaction_open_ = false;
        mesh_.pending_ = kNoPending;
      }
    }
    void retire_constraints() noexcept {
      mesh_.transaction_open_ = true;
      marks_retired_ = true;
      for (const Crossed& face : retired_) {
        mesh_.complex_.set_facet_constraint(face.tet, face.slot, kNoConstraint);
      }
    }
    void release() noexcept {
      active_ = false;
      mesh_.transaction_open_ = false;
    }
    EditTransaction(const EditTransaction&) = delete;
    EditTransaction& operator=(const EditTransaction&) = delete;
   private:
    TetMesh& mesh_;
    std::span<const Crossed> retired_;
    bool active_ = true;
    bool marks_retired_ = false;
  };

  // For an off-edge split, each changed scientific/material patch must have
  // the same exact plane, orientation and outer link. A patch's old (a,b)
  // chain must cancel internally: otherwise moving the split point would move
  // its boundary, even when adjacent faces happen to share a source label.
  bool split_surface_preserved(int32_t a, int32_t b, std::span<const int32_t> star,
                               const double* position) {
    struct Plane {
      std::array<int32_t, 3> vertices;
      int32_t source;
      int axis;
      int sign;
    };
    struct Proof {
      Expansion area_delta;
      int64_t edge_delta = 0;
    };
    using Stratum = std::array<int32_t, 3>;  // represented plane, oriented regions
    NativeVector<Plane> planes;
    NativeMap<Stratum, Proof> patches;
    const auto area = [](const double* const* p, int axis) {
      const int x = (axis + 1) % 3, y = (axis + 2) % 3;
      const double u[2] = {p[0][x], p[0][y]};
      const double v[2] = {p[1][x], p[1][y]};
      const double w[2] = {p[2][x], p[2][y]};
      return orient2d_exact(u, v, w);
    };
    for (int32_t t : star) {
      if (!finite(t)) {
        continue;
      }
      for (int slot = 0; slot < 4; ++slot) {
        if (!spend(8)) {
          return false;
        }
        const int32_t other = tet(t).n[slot];
        const int32_t source = complex_.constraint(t, slot);
        const int32_t mine = complex_.region(t), theirs = complex_.region(other);
        if (tet(t).v[slot] == a || tet(t).v[slot] == b ||
            (finite(other) && other < t) ||
            (source == kNoConstraint && finite(other) && mine == theirs)) {
          continue;
        }
        int32_t f[3];
        oriented_facet(tet(t).v, slot, f);
        const double* p[3] = {point(f[0]), point(f[1]), point(f[2])};
        if (orient3d(p[0], p[1], p[2], position) != 0) {
          return false;
        }
        std::size_t plane = 0;
        for (; plane < planes.size(); ++plane) {
          if (!spend(4)) {
            return false;
          }
          const Plane& represented = planes[plane];
          const auto& v = represented.vertices;
          if (represented.source == source &&
              orient3d(point(v[0]), point(v[1]), point(v[2]), p[0]) == 0 &&
              orient3d(point(v[0]), point(v[1]), point(v[2]), p[1]) == 0 &&
              orient3d(point(v[0]), point(v[1]), point(v[2]), p[2]) == 0) {
            break;
          }
        }
        if (plane == planes.size()) {
          int axis = 0, sign = 0;
          for (; axis < 3; ++axis) {
            sign = area(p, axis).sign();
            if (sign != 0) {
              break;
            }
          }
          if (sign == 0) {
            return false;
          }
          planes.push_back({{f[0], f[1], f[2]}, source, axis, sign});
        }
        const Plane& represented = planes[plane];
        const Expansion before = area(p, represented.axis);
        const int sign = before.sign();
        if (sign == 0) {
          return false;
        }
        const int64_t orientation = sign == represented.sign ? 1 : -1;
        const Stratum stratum = orientation > 0
                                    ? Stratum{static_cast<int32_t>(plane), mine, theirs}
                                    : Stratum{static_cast<int32_t>(plane), theirs, mine};
        Proof& proof = patches[stratum];
        Expansion delta = before;
        for (int endpoint : {a, b}) {
          const double* child[3] = {p[0], p[1], p[2]};
          for (int i = 0; i < 3; ++i) {
            if (f[i] == endpoint) {
              child[i] = position;
            }
          }
          const Expansion after = area(child, represented.axis);
          if (after.sign() != sign) {
            return false;
          }
          delta = delta - after;
        }
        proof.area_delta = orientation > 0 ? proof.area_delta + delta
                                           : proof.area_delta - delta;
        for (int i = 0; i < 3; ++i) {
          if (f[i] == a && f[(i + 1) % 3] == b) {
            proof.edge_delta += orientation;
          } else if (f[i] == b && f[(i + 1) % 3] == a) {
            proof.edge_delta -= orientation;
          }
        }
      }
    }
    for (const auto& [stratum, proof] : patches) {
      if (proof.edge_delta != 0 || !proof.area_delta.is_zero()) {
        return false;
      }
    }
    // Child marks and regions inherit each old side unchanged. CavityEdit
    // validates the identical outer shell and all reciprocal finite/ghost
    // pairings; no source identity or triangle ancestry is synthesized here.
    return true;
  }

  // --------------------------------------------------------- build steps
  struct FacetRecord {
    FaceKey key;
    int32_t tet;
    int32_t slot;
    bool operator<(const FacetRecord& other) const {
      if (!(key == other.key)) {
        return key < other.key;
      }
      return tet != other.tet ? tet < other.tet : slot < other.slot;
    }
  };

  int32_t build_cells(int64_t point_count, int64_t tet_count, const int32_t* tets,
                      const int32_t* regions) {
    const MemoryScope memory_scope(memory_owner_);
    complex_.tets.clear();
    complex_.tets.reserve(static_cast<std::size_t>(2 * tet_count + 8));
    for (int64_t i = 0; i < tet_count; ++i) {
      native_execution_charge(0);
      const int32_t* v = tets + 4 * i;
      for (int k = 0; k < 4; ++k) {
        if (v[k] < 0 || v[k] >= point_count) {
          return PHX_MC_INVALID_INPUT;
        }
        for (int j = 0; j < k; ++j) {
          if (v[j] == v[k]) {
            return PHX_MC_INVALID_INPUT;
          }
        }
      }
      // Source witnesses are published after carrier topology is constructed.
      // verify_ancestry subsequently proves positivity on the authoritative S.
      if (regions[i] < 0 ||
          orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3])) <= 0) {
        return PHX_MC_INVALID_INPUT;
      }
      complex_.tets.push_back(Tetrahedron{{v[0], v[1], v[2], v[3]}, {-1, -1, -1, -1}, 0U, 0});
    }
    complex_.finite_count = tet_count;
    records_.clear();
    records_.reserve(static_cast<std::size_t>(4 * tet_count));
    for (int64_t i = 0; i < tet_count; ++i) {
      native_execution_charge(0);
      const int32_t* v = tets + 4 * i;
      for (int k = 0; k < 4; ++k) {
        int32_t facet[3];
        oriented_facet(v, k, facet);
        records_.push_back({FaceKey::of(facet[0], facet[1], facet[2]), static_cast<int32_t>(i), k});
      }
    }
    std::sort(records_.begin(), records_.end(), [](const FacetRecord& a, const FacetRecord& b) {
      native_execution_charge(0);
      return a < b;
    });
    NativeVector<int32_t> ghosts;
    for (std::size_t r = 0; r < records_.size();) {
      native_execution_charge(0);
      std::size_t end = r + 1;
      while (end < records_.size() && records_[end].key == records_[r].key) {
        ++end;
      }
      if (end - r > 2) {
        return PHX_MC_INVALID_INPUT;
      }
      const FacetRecord& first = records_[r];
      Tetrahedron& t = complex_.tets[static_cast<std::size_t>(first.tet)];
      if (end - r == 2) {
        const FacetRecord& second = records_[r + 1];
        Tetrahedron& u = complex_.tets[static_cast<std::size_t>(second.tet)];
        int32_t f[3];
        int32_t g[3];
        oriented_facet(t.v, first.slot, f);
        oriented_facet(u.v, second.slot, g);
        if (!(FacetKey::of(f) == FacetKey::of(g).reversed())) {
          return PHX_MC_INVALID_INPUT;
        }
        t.n[first.slot] = second.tet;
        u.n[second.slot] = first.tet;
      } else {
        Tetrahedron shell = t;
        shell.v[first.slot] = kGhostVertex;
        // Odd permutation of the facet: the outside becomes the positive side.
        const int a = first.slot == 0 ? 1 : 0;
        const int b = first.slot <= 1 ? 2 : 1;
        std::swap(shell.v[a], shell.v[b]);
        shell.n[0] = shell.n[1] = shell.n[2] = shell.n[3] = -1;
        shell.n[first.slot] = first.tet;
        const auto ghost = static_cast<int32_t>(complex_.tets.size());
        complex_.tets.push_back(shell);
        complex_.tets[static_cast<std::size_t>(first.tet)].n[first.slot] = ghost;
        ghosts.push_back(ghost);
      }
      r = end;
    }
    return link_ghosts(ghosts, point_count, regions);
  }

  // Pairs the ghost facets (G, p, q) and (G, q, p) around each boundary edge
  // in a deterministic order, then enables labels and assigns regions.
  int32_t link_ghosts(std::span<const int32_t> ghosts, int64_t point_count,
                      const int32_t* regions) {
    const MemoryScope memory_scope(memory_owner_);
    struct GhostFacet {
      std::uint64_t edge;
      int32_t forward;  // 1 when the facet reads (G, low, high)
      int32_t ghost;
      int32_t slot;
      bool operator<(const GhostFacet& other) const {
        if (edge != other.edge) {
          return edge < other.edge;
        }
        return forward != other.forward ? forward < other.forward : ghost < other.ghost;
      }
    };
    NativeVector<GhostFacet> entries;
    entries.reserve(3 * ghosts.size());
    for (int32_t g : ghosts) {
      const Tetrahedron& shell = tet(g);
      const int apex = vertex_slot(shell, kGhostVertex);
      for (int s = 0; s < 4; ++s) {
        if (s == apex) {
          continue;
        }
        int32_t facet[3];
        oriented_facet(shell.v, s, facet);
        const int at = static_cast<int>(std::find(facet, facet + 3, kGhostVertex) - facet);
        const int32_t p = facet[(at + 1) % 3];
        const int32_t q = facet[(at + 2) % 3];
        entries.push_back({undirected_edge(p, q), p < q ? 1 : 0, g, s});
      }
    }
    std::sort(entries.begin(), entries.end());
    for (std::size_t r = 0; r < entries.size();) {
      std::size_t middle = r;
      while (middle < entries.size() && entries[middle].edge == entries[r].edge &&
             entries[middle].forward == 0) {
        ++middle;
      }
      std::size_t end = middle;
      while (end < entries.size() && entries[end].edge == entries[r].edge) {
        ++end;
      }
      if (middle - r != end - middle) {
        return PHX_MC_INVALID_INPUT;
      }
      for (std::size_t i = 0; i < middle - r; ++i) {
        const GhostFacet& x = entries[r + i];
        const GhostFacet& y = entries[middle + i];
        complex_.tets[static_cast<std::size_t>(x.ghost)].n[x.slot] = y.ghost;
        complex_.tets[static_cast<std::size_t>(y.ghost)].n[y.slot] = x.ghost;
      }
      r = end;
    }
    complex_.enable_labels();
    for (int64_t i = 0; i < complex_.finite_count; ++i) {
      complex_.set_region(static_cast<int32_t>(i), regions[i]);
      for (int32_t v : complex_.tets[static_cast<std::size_t>(i)].v) {
        vertex_tet_[static_cast<std::size_t>(v)] = static_cast<int32_t>(i);
      }
    }
    (void)point_count;
    return complex_.audit() ? PHX_MC_OK : PHX_MC_INVALID_INPUT;
  }

  const FacetRecord* find_record(const FaceKey& key) const {
    const FacetRecord probe{key, std::numeric_limits<int32_t>::min(), 0};
    const auto found = std::lower_bound(records_.begin(), records_.end(), probe);
    return found != records_.end() && found->key == key ? &*found : nullptr;
  }

  int32_t build_constraints(int64_t point_count, int64_t face_count, const int32_t* faces,
                            const int32_t* sources) {
    const MemoryScope memory_scope(memory_owner_);
    for (int64_t i = 0; i < face_count; ++i) {
      native_execution_charge(0);
      const int32_t* f = faces + 3 * i;
      if (sources[i] < 0) {
        return PHX_MC_INVALID_INPUT;
      }
      for (int k = 0; k < 3; ++k) {
        if (f[k] < 0 || f[k] >= point_count) {
          return PHX_MC_INVALID_INPUT;
        }
      }
      const FacetRecord* record = find_record(FaceKey::of(f[0], f[1], f[2]));
      if (record == nullptr) {
        return PHX_MC_INVALID_INPUT;
      }
      const int32_t existing = complex_.constraint(record->tet, record->slot);
      if (existing == kNoConstraint) {
        complex_.set_facet_constraint(record->tet, record->slot, sources[i]);
      } else if (existing != sources[i]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    // The domain boundary and every region interface are constrained.
    for (int64_t i = 0; i < complex_.finite_count; ++i) {
      native_execution_charge(0);
      const auto t = static_cast<int32_t>(i);
      for (int k = 0; k < 4; ++k) {
        const int32_t other = tet(t).n[k];
        const bool open = is_ghost(tet(other)) || complex_.region(other) != complex_.region(t);
        if (open && complex_.constraint(t, k) == kNoConstraint) {
          return PHX_MC_INVALID_INPUT;
        }
      }
    }
    return PHX_MC_OK;
  }

  int32_t build_segments(int64_t point_count, int64_t segment_count, const int32_t* segments,
                         const int32_t* sources) {
    const MemoryScope memory_scope(memory_owner_);
    NativeVector<std::uint64_t> edges;
    edges.reserve(static_cast<std::size_t>(6 * complex_.finite_count));
    for (int64_t i = 0; i < complex_.finite_count; ++i) {
      native_execution_charge(0);
      const int32_t* v = complex_.tets[static_cast<std::size_t>(i)].v;
      for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
          edges.push_back(undirected_edge(v[a], v[b]));
        }
      }
    }
    std::sort(edges.begin(), edges.end(), [](uint64_t a, uint64_t b) {
      native_execution_charge(0);
      return a < b;
    });
    edges.erase(std::unique(edges.begin(), edges.end(), [](uint64_t a, uint64_t b) {
      native_execution_charge(0);
      return a == b;
    }), edges.end());
    segments_.clear();
    segments_.reserve(static_cast<std::size_t>(segment_count));
    for (int64_t i = 0; i < segment_count; ++i) {
      native_execution_charge(0);
      const int32_t a = segments[2 * i];
      const int32_t b = segments[2 * i + 1];
      if (a < 0 || b < 0 || a >= point_count || b >= point_count || a == b || sources[i] < 0) {
        return PHX_MC_INVALID_INPUT;
      }
      const std::uint64_t key = undirected_edge(a, b);
      if (!std::binary_search(edges.begin(), edges.end(), key)) {
        return PHX_MC_INVALID_INPUT;
      }
      const auto [found, inserted] = segments_.emplace(key, sources[i]);
      if (!inserted && found->second != sources[i]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    // A face edge that is not a segment lies inside one planar facet: exactly
    // two coplanar faces of one source.
    struct FaceEdge {
      std::uint64_t edge;
      int32_t source;
      int32_t apex;
      bool operator<(const FaceEdge& other) const {
        if (edge != other.edge) {
          return edge < other.edge;
        }
        return source != other.source ? source < other.source : apex < other.apex;
      }
    };
    NativeVector<FaceEdge> face_edges;
    for (const FacetRecord& record : records_) {
      const int32_t source = complex_.constraint(record.tet, record.slot);
      const int32_t other = tet(record.tet).n[record.slot];
      if (source == kNoConstraint || (!is_ghost(tet(other)) && other < record.tet)) {
        continue;
      }
      const int32_t* f = record.key.v;
      face_edges.push_back({undirected_edge(f[0], f[1]), source, f[2]});
      face_edges.push_back({undirected_edge(f[1], f[2]), source, f[0]});
      face_edges.push_back({undirected_edge(f[0], f[2]), source, f[1]});
    }
    std::sort(face_edges.begin(), face_edges.end());
    for (std::size_t r = 0; r < face_edges.size();) {
      std::size_t end = r + 1;
      while (end < face_edges.size() && face_edges[end].edge == face_edges[r].edge) {
        ++end;
      }
      const std::uint64_t key = face_edges[r].edge;
      if (segments_.count(key) == 0) {
        // Exactly planar, or folded only by a certified bounded witness of
        // one of its four vertices; source identity always agrees.
        const auto bounded = [&](int32_t v) {
          return witnesses_[static_cast<std::size_t>(v)].deviation > 0.0;
        };
        const bool paired = end - r == 2 && face_edges[r].source == face_edges[r + 1].source;
        const bool planar =
            paired && (orient3d(point(edge_low(key)), point(edge_high(key)),
                                point(face_edges[r].apex), point(face_edges[r + 1].apex)) == 0 ||
                       bounded(edge_low(key)) || bounded(edge_high(key)) ||
                       bounded(face_edges[r].apex) || bounded(face_edges[r + 1].apex));
        if (!planar) {
          return PHX_MC_INVALID_INPUT;
        }
      }
      r = end;
    }
    return PHX_MC_OK;
  }

  // Vertex dimension: 0 corner (segment degree other than 2, or a bend or
  // source change between its two segments), 1 on a segment, 2 on a
  // constrained face, 3 interior; -1 unused.
  void classify_vertices() {
    const MemoryScope memory_scope(memory_owner_);
    for (int64_t i = 0; i < complex_.finite_count; ++i) {
      native_execution_charge(0);
      for (int32_t v : complex_.tets[static_cast<std::size_t>(i)].v) {
        dimension_[static_cast<std::size_t>(v)] = 3;
      }
    }
    for (const FacetRecord& record : records_) {
      native_execution_charge(0);
      if (complex_.constraint(record.tet, record.slot) != kNoConstraint) {
        for (int32_t v : record.key.v) {
          dimension_[static_cast<std::size_t>(v)] = 2;
        }
      }
    }
    NativeVector<int32_t> degree(dimension_.size(), 0);
    NativeVector<int32_t> neighbor(2 * dimension_.size(), -1);
    NativeVector<int32_t> source(2 * dimension_.size(), -1);
    NativeVector<std::uint64_t> keys;
    keys.reserve(segments_.size());
    for (const auto& entry : segments_) {
      native_execution_charge(0);
      keys.push_back(entry.first);
    }
    std::sort(keys.begin(), keys.end());
    for (std::uint64_t key : keys) {
      native_execution_charge(0);
      const int32_t ends[2] = {edge_low(key), edge_high(key)};
      for (int e = 0; e < 2; ++e) {
        const auto v = static_cast<std::size_t>(ends[e]);
        if (degree[v] < 2) {
          neighbor[2 * v + static_cast<std::size_t>(degree[v])] = ends[1 - e];
          source[2 * v + static_cast<std::size_t>(degree[v])] = segments_.at(key);
        }
        ++degree[v];
      }
    }
    for (std::size_t v = 0; v < dimension_.size(); ++v) {
      native_execution_charge(0);
      if (degree[v] == 0) {
        continue;
      }
      // A bounded segment witness folds the represented curve within its
      // certified deviation; it does not make a corner.
      const SourceWitness& witness = witnesses_[v];
      const bool smooth = degree[v] == 2 && source[2 * v] == source[2 * v + 1] &&
                          ((witness.deviation > 0.0 && witness.stratum == SourceStratum::kSegment) ||
                           collinear3d(point(neighbor[2 * v]), point(static_cast<int32_t>(v)),
                                       point(neighbor[2 * v + 1])));
      dimension_[v] = smooth ? 1 : 0;
    }
  }

  // ------------------------------------------------------ insertion steps
  void next_stamp() {
    if (++stamp_ == 0) {
      for (Tetrahedron& t : complex_.tets) {
        t.stamp = 0U;
      }
      stamp_ = 1;
    }
  }
  bool marked(int32_t t) const {
    const Tetrahedron& current = tet(t);
    return current.stamp == stamp_ && current.state == kInCavity;
  }
  void mark(int32_t t) {
    Tetrahedron& current = complex_.tets[static_cast<std::size_t>(t)];
    current.stamp = stamp_;
    current.state = kInCavity;
  }
  void unmark(int32_t t) { complex_.tets[static_cast<std::size_t>(t)].state = 0; }

  bool coincident(int32_t t, const double* p) const {
    for (int32_t v : tet(t).v) {
      if (v >= 0) {
        const double* q = point(v);
        if (q[0] == p[0] && q[1] == p[1] && q[2] == p[2]) {
          return true;
        }
      }
    }
    return false;
  }

  bool conflicts(int32_t t, const double* p) const {
    const MemoryScope memory_scope(memory_owner_);
    const int32_t* v = tet(t).v;
    bool exact_source = p == pending_point_ && pending_source_.deviation > 0.0;
    for (int k = 0; k < 4; ++k) exact_source = exact_source || witness(v[k]).deviation > 0.0;
    if (!exact_source) return insphere(point(v[0]), point(v[1]), point(v[2]), point(v[3]), p) > 0;
    Expansion coordinates[5][3];
    for (int k = 0; k < 4; ++k) {
      if (!source_position(v[k], coordinates[k])) return false;
    }
    if (p == pending_point_) {
      if (!source_position(pending_, coordinates[4])) return false;
    } else {
      for (int axis = 0; axis < 3; ++axis) coordinates[4][axis] = Expansion(p[axis]);
    }
    return insphere_expansion(coordinates[0], coordinates[1], coordinates[2],
                              coordinates[3], coordinates[4]) > 0;
  }

  bool contains(int32_t t, const double* p) const {
    const MemoryScope memory_scope(memory_owner_);
    for (int k = 0; k < 4; ++k) {
      if (orient_with(t, k, p) < 0) {
        return false;
      }
    }
    return true;
  }

  Insertion grow(int32_t origin) {
    const MemoryScope memory_scope(memory_owner_);
    const double* p = pending_point_;
    const bool crossable = policy_ == BoundaryPolicy::kConforming;
    next_stamp();
    cavity_.clear();
    stack_.clear();
    std::size_t finite_count = is_ghost(tet(origin)) ? 0 : 1;
    if (finite_count > preparation_cell_limit_ || !native_execution_cavity(1)) {
      return Insertion::kCapacity;
    }
    mark(origin);
    cavity_.push_back(origin);
    stack_.push_back(origin);
    while (!stack_.empty()) {
      const int32_t t = stack_.back();
      stack_.pop_back();
      if (is_ghost(tet(t))) {
        continue;
      }
      if (!spend(4)) {
        return Insertion::kCapacity;
      }
      for (int k = 0; k < 4; ++k) {
        const int32_t u = tet(t).n[k];
        if (marked(u)) {
          continue;
        }
        bool take = false;
        if (complex_.constraint(t, k) != kNoConstraint) {
          if (crossable && orient_with(t, k, p) == 0 && pending_source_authored_) {
            int32_t face[3];
            oriented_facet(tet(t).v, k, face);
            const int32_t row = original_face_ancestor(face, complex_.constraint(t, k));
            if (row < 0 ||
                pending_source_.deviation > row_tolerance(SourceStratum::kFacet, row) ||
                (pending_source_.deviation > 0.0
                     ? !witness_on_facet(pending_source_, static_cast<std::size_t>(row))
                     : !on_source_entity(
                           std::array<const double*, 3>{
                               original_point(original_faces_[static_cast<std::size_t>(row)][0]),
                               original_point(original_faces_[static_cast<std::size_t>(row)][1]),
                               original_point(original_faces_[static_cast<std::size_t>(row)][2])}.data(),
                           SourceStratum::kFacet, p))) {
              continue;
            }
          }
          take = crossable && orient_with(t, k, p) == 0 && (is_ghost(tet(u)) || conflicts(u, p));
        } else {
          take = !is_ghost(tet(u)) && conflicts(u, p);
        }
        if (take) {
          if (!is_ghost(tet(u)) && ++finite_count > preparation_cell_limit_) {
            return Insertion::kCapacity;
          }
          if (!native_execution_cavity(cavity_.size() + 1)) return Insertion::kCapacity;
          mark(u);
          cavity_.push_back(u);
          stack_.push_back(u);
        }
      }
    }
    return Insertion::kOk;
  }

  // Removes cavity tetrahedra whose cone over a boundary facet would not be
  // positively oriented (and ghosts cut off from their finite facet) until
  // the cavity is star-shaped from p; the tetrahedra containing p must stay.
  Insertion shrink(int32_t origin) {
    const MemoryScope memory_scope(memory_owner_);
    const double* p = pending_point_;
    bool changed = true;
    while (changed) {
      changed = false;
      if (!spend(static_cast<int64_t>(cavity_.size()))) {
        return Insertion::kCapacity;
      }
      for (int32_t t : cavity_) {
        if (!marked(t)) {
          continue;
        }
        const Tetrahedron& current = tet(t);
        bool drop = false;
        if (is_ghost(current)) {
          drop = !marked(current.n[vertex_slot(current, kGhostVertex)]);
        } else {
          for (int k = 0; k < 4 && !drop; ++k) {
            drop = !marked(current.n[k]) && orient_with(t, k, p) <= 0;
          }
        }
        if (drop) {
          if (t == origin || (!is_ghost(current) && contains(t, p))) {
            return Insertion::kRefused;
          }
          unmark(t);
          changed = true;
        }
      }
    }
    std::size_t kept = 0;
    for (int32_t t : cavity_) {
      if (marked(t)) {
        cavity_[kept++] = t;
      }
    }
    cavity_.resize(kept);
    return Insertion::kOk;
  }

  // Boundary facets, crossed constrained faces and the new constrained faces
  // through p; refuses a cavity that would lose a vertex or a segment.
  Insertion collect() {
    const MemoryScope memory_scope(memory_owner_);
    const double* p = pending_point_;
    boundary_.clear();
    crossed_.clear();
    std::size_t finite_removed = 0;
    std::size_t finite_proposed = 0;
    for (int32_t t : cavity_) {
      const Tetrahedron& current = tet(t);
      if (!is_ghost(current)) {
        ++finite_removed;
      }
      for (int k = 0; k < 4; ++k) {
        const int32_t u = current.n[k];
        if (!marked(u)) {
          boundary_.push_back({t, k, u, neighbor_slot(tet(u), t)});
          Tetrahedron cone = current;
          cone.v[k] = pending_;
          if (!is_ghost(cone) && ++finite_proposed > preparation_cell_limit_) {
            return Insertion::kCapacity;
          }
        } else if (t < u && complex_.constraint(t, k) != kNoConstraint) {
          crossed_.push_back({t, k, complex_.constraint(t, k)});
        }
      }
    }
    if (finite_removed > preparation_cell_limit_ ||
        finite_proposed > preparation_cell_limit_ - finite_removed) {
      return Insertion::kCapacity;
    }
    // Every cavity vertex and segment must survive on the boundary.
    ++vertex_mark_;
    boundary_edges_.clear();
    for (const BoundaryFacet& face : boundary_) {
      const Tetrahedron& current = tet(face.tet);
      int32_t f[3];
      oriented_facet(current.v, face.slot, f);
      for (int r = 0; r < 3; ++r) {
        if (f[r] >= 0) {
          vertex_stamp_[static_cast<std::size_t>(f[r])] = vertex_mark_;
          if (f[(r + 1) % 3] >= 0) {
            boundary_edges_.push_back(undirected_edge(f[r], f[(r + 1) % 3]));
          }
        }
      }
    }
    std::sort(boundary_edges_.begin(), boundary_edges_.end());
    const std::uint64_t split = split_a_ >= 0 ? undirected_edge(split_a_, split_b_) : 0;
    for (int32_t t : cavity_) {
      const int32_t* v = tet(t).v;
      for (int a = 0; a < 4; ++a) {
        if (v[a] >= 0 && vertex_stamp_[static_cast<std::size_t>(v[a])] != vertex_mark_) {
          return Insertion::kRefused;
        }
        for (int b = a + 1; b < 4; ++b) {
          if (v[a] < 0 || v[b] < 0) {
            continue;
          }
          const std::uint64_t key = undirected_edge(v[a], v[b]);
          if ((split_a_ < 0 || key != split) && segments_.count(key) != 0 &&
              !std::binary_search(boundary_edges_.begin(), boundary_edges_.end(), key)) {
            return Insertion::kRefused;
          }
        }
      }
    }
    // Each edge of a crossed face not through p bounds the split region of
    // its source once; the face (p, edge) inherits that source.
    new_faces_.clear();
    for (const Crossed& face : crossed_) {
      int32_t f[3];
      oriented_facet(tet(face.tet).v, face.slot, f);
      for (int r = 0; r < 3; ++r) {
        const int32_t a = f[r];
        const int32_t b = f[(r + 1) % 3];
        if (!collinear3d(point(a), point(b), p)) {
          new_faces_.push_back({undirected_edge(a, b), face.source});
        }
      }
    }
    std::sort(new_faces_.begin(), new_faces_.end());
    std::size_t kept = 0;
    for (std::size_t r = 0; r < new_faces_.size();) {
      std::size_t end = r + 1;
      while (end < new_faces_.size() && new_faces_[end].first == new_faces_[r].first) {
        ++end;
      }
      if (end - r == 1) {
        new_faces_[kept++] = new_faces_[r];
      } else if (end - r > 2 || new_faces_[r].second != new_faces_[r + 1].second) {
        return Insertion::kRefused;
      }
      r = end;
    }
    new_faces_.resize(kept);
    return Insertion::kOk;
  }

  // Source of the facet of cone tetrahedron v (apex p at slot `apex`)
  // opposite v[j], or kNoConstraint.
  int32_t new_face_source(const int32_t* v, int apex, int j) const {
    int32_t edge[2];
    int count = 0;
    for (int r = 0; r < 4; ++r) {
      if (r != apex && r != j) {
        edge[count++] = v[r];
      }
    }
    if (edge[0] < 0 || edge[1] < 0) {
      return kNoConstraint;
    }
    const std::pair<std::uint64_t, int32_t> probe{undirected_edge(edge[0], edge[1]),
                                                  std::numeric_limits<int32_t>::min()};
    const auto found = std::lower_bound(new_faces_.begin(), new_faces_.end(), probe);
    return found != new_faces_.end() && found->first == probe.first ? found->second
                                                                    : kNoConstraint;
  }

  void after_commit() {
    const MemoryScope memory_scope(memory_owner_);
    mark_accepted_state();
    slot_reason_.resize(complex_.tets.size(), Unmet::kNone);
    for (int32_t id : edit_.created()) {
      slot_reason_[static_cast<std::size_t>(id)] = Unmet::kNone;
      const Tetrahedron& current = tet(id);
      for (int32_t v : current.v) {
        if (v >= 0) {
          int32_t& incident = vertex_tet_[static_cast<std::size_t>(v)];
          if (incident < 0 || !complex_.live(incident) || vertex_slot(tet(incident), v) < 0 ||
              (is_ghost(tet(incident)) && !is_ghost(current))) {
            incident = id;
          }
        }
      }
    }
  }

  static bool even_permutation(const int32_t* reference, const int32_t* order) {
    int position[4];
    for (int k = 0; k < 4; ++k) {
      position[k] = vertex_slot(Tetrahedron{{reference[0], reference[1], reference[2],
                                             reference[3]},
                                            {-1, -1, -1, -1}, 0U, 0},
                                order[k]);
    }
    int inversions = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        inversions += position[i] > position[j] ? 1 : 0;
      }
    }
    return inversions % 2 == 0;
  }

  MemoryOwner memory_owner_;
  BoundaryPolicy policy_;
  int64_t max_vertices_;
  int64_t max_tetrahedra_;
  TetrahedralComplex complex_;
  CavityEdit edit_;
  NativeVector<double> points_{NativeAllocator<double>(memory_owner_)};
  NativeVector<double> original_points_{NativeAllocator<double>(memory_owner_)};
  NativeVector<std::array<int32_t, 4>> original_faces_{
      NativeAllocator<std::array<int32_t, 4>>(memory_owner_)};
  NativeVector<std::array<int32_t, 3>> original_segments_{
      NativeAllocator<std::array<int32_t, 3>>(memory_owner_)};
  NativeVector<double> face_tolerance_{NativeAllocator<double>(memory_owner_)};
  NativeVector<double> segment_tolerance_{NativeAllocator<double>(memory_owner_)};
  // (source id, row), sorted: rows of one source.
  NativeVector<std::pair<int32_t, int32_t>> face_rows_{
      NativeAllocator<std::pair<int32_t, int32_t>>(memory_owner_)};
  NativeVector<std::pair<int32_t, int32_t>> segment_rows_{
      NativeAllocator<std::pair<int32_t, int32_t>>(memory_owner_)};
  NativeVector<SourceWitness> witnesses_{NativeAllocator<SourceWitness>(memory_owner_)};
  NativeVector<SourceRefusal> source_refusals_{NativeAllocator<SourceRefusal>(memory_owner_)};
  bool work_exhausted_ = false;
  NativeVector<int8_t> dimension_{NativeAllocator<int8_t>(memory_owner_)};
  NativeVector<double> protection_{NativeAllocator<double>(memory_owner_)};
  NativeVector<double> sizes_{NativeAllocator<double>(memory_owner_)};
  NativeVector<int32_t> vertex_tet_{NativeAllocator<int32_t>(memory_owner_)};
  NativeVector<std::uint32_t> vertex_stamp_{NativeAllocator<std::uint32_t>(memory_owner_)};
  std::uint32_t vertex_mark_ = 0;
  NativeUnorderedMap<std::uint64_t, int32_t> segments_{
      NativeAllocator<std::pair<const std::uint64_t, int32_t>>(memory_owner_)};
  NativeVector<FacetRecord> records_{NativeAllocator<FacetRecord>(memory_owner_)};
  NativeVector<Unmet> slot_reason_{NativeAllocator<Unmet>(memory_owner_)};
  NativeVector<UnmetRecord> unmet_{NativeAllocator<UnmetRecord>(memory_owner_)};
  std::uint32_t stamp_ = 0;
  std::uint64_t walk_step_ = 0;
  int64_t work_ = 0;
  int64_t work_limit_ = std::numeric_limits<int64_t>::max();
  std::size_t largest_cavity_ = 0;
  bool broken_ = false;
  bool transaction_open_ = false;
  int64_t accepted_generation_ = 0;
  bool measure_execution_ = false;
  std::array<double, 3> execution_seconds_{};
  std::array<int32_t, 3> execution_measured_{};
  int32_t pending_ = kNoPending;
  double pending_point_[3] = {0.0, 0.0, 0.0};
  SourceWitness pending_source_;
  bool pending_source_authored_ = false;
  bool pending_edge_star_ = false;
  int32_t split_a_ = -1;
  int32_t split_b_ = -1;
  std::size_t preparation_cell_limit_ = std::numeric_limits<std::size_t>::max();
  NativeVector<int32_t> cavity_{NativeAllocator<int32_t>(memory_owner_)};
  NativeVector<int32_t> stack_{NativeAllocator<int32_t>(memory_owner_)};
  NativeVector<int32_t> scratch_star_{NativeAllocator<int32_t>(memory_owner_)};
  NativeVector<BoundaryFacet> boundary_{NativeAllocator<BoundaryFacet>(memory_owner_)};
  NativeVector<Crossed> crossed_{NativeAllocator<Crossed>(memory_owner_)};
  NativeVector<std::uint64_t> boundary_edges_{NativeAllocator<std::uint64_t>(memory_owner_)};
  NativeVector<std::pair<std::uint64_t, int32_t>> new_faces_{
      NativeAllocator<std::pair<std::uint64_t, int32_t>>(memory_owner_)};
};

// A call-local allowance can only narrow the owning cumulative allowance.
// Restore the absolute ceiling, not a fresh allowance after consumed work.
class TetWorkBudgetWindow final {
 public:
  TetWorkBudgetWindow(TetMesh& mesh, int64_t allowance) noexcept
      : mesh_(mesh), previous_(mesh.work_ceiling()) {
    mesh_.set_work_limit(std::min(allowance, std::max<int64_t>(0, previous_ - mesh_.work())));
  }
  ~TetWorkBudgetWindow() { mesh_.restore_work_ceiling(previous_); }
  TetWorkBudgetWindow(const TetWorkBudgetWindow&) = delete;
  TetWorkBudgetWindow& operator=(const TetWorkBudgetWindow&) = delete;
 private:
  TetMesh& mesh_;
  int64_t previous_;
};

// The disabled path calls no clock. Destruction records failed invocations as
// well as success; times never participate in scientific identity or counters.
class MeasuredTetStage {
 public:
  MeasuredTetStage(TetMesh& mesh, TetExecutionStage stage)
      : mesh_(mesh), slot_(static_cast<std::size_t>(stage)), enabled_(mesh.measure_execution_) {
    if (enabled_) {
      start_ = std::chrono::steady_clock::now();
    }
  }
  ~MeasuredTetStage() {
    if (enabled_) {
      mesh_.execution_seconds_[slot_] =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
      mesh_.execution_measured_[slot_] = 1;
    }
  }
  MeasuredTetStage(const MeasuredTetStage&) = delete;
  MeasuredTetStage& operator=(const MeasuredTetStage&) = delete;
 private:
  TetMesh& mesh_;
  std::size_t slot_;
  bool enabled_;
  std::chrono::steady_clock::time_point start_{};
};

}  // namespace phx::mc

// Owned C ABI handle.
struct phx_mc_tet_mesh : phx::mc::NativeAllocatedObject {
  phx::mc::NativeUniquePtr<phx::mc::TetMesh> mesh;
};
