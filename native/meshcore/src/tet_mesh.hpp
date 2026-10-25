//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained tetrahedral mesh state shared by quality refinement
// (refine3d.cpp) and improvement (improve3d.cpp).
// TetMesh declarations and transaction tokens live here; state construction,
// source ancestry, insertion, collapse, refinement constructions, improvement
// transactions and export each have a tet_mesh_*.cpp implementation owner.
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
#include "boundary_policy.hpp"
#include "cavity.hpp"
#include "geometry3.hpp"
#include "topology_keys.hpp"
#include "mesh.hpp"
#include "intersections.hpp"
#include "expansion.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "source_construction.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {


enum class TetExecutionStage : std::size_t { kRefinement, kImprovement, kExudation };

// ------------------------------------------------------------------ geometry
// Inexact constructions and quality measures.  Decisions that change topology
// or claim exact incidence use the exact predicates; these values only rank
// candidates and decide admissible quality.
namespace geometry {

inline bool finite3(const double* p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Circumcenter of a tetrahedron; false when degenerate or not finite.
bool tetrahedron_circumcenter(const double* a, const double* b, const double* c,
                                     const double* d, double* center);

// Circumcenter of a triangle in its plane; coordinates shared by all three
// vertices are copied exactly, so axis-aligned facets receive exactly
// coplanar centers.
bool triangle_circumcenter(const double* a, const double* b, const double* c,
                                  double* center);

inline constexpr double kDegreesPerRadian = 57.295779513082320876798154814105;

// Smallest and largest interior dihedral angle (degrees); an inverted or
// degenerate tetrahedron reports 0 and 180.
void dihedral_extremes(const double* const* p, double& minimum, double& maximum);

double minimum_dihedral(const double* const* p);

double shortest_edge(const double* const* p);

double signed_volume(const double* const* p);

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

// Host-declared curved (smooth) face sources of an own source complex. A
// listed face source group is a smooth surface whose fidelity only its host
// owner can certify: `tolerances` bound every host-claimed deviation of its
// constrained faces from that surface and `digests` (32 bytes per source) bind
// the claims to the host source/profile/limits identity. `face_claims` gives
// the host-claimed certified deviation of every initial constrained face (NaN
// exactly for faces of undeclared sources). Native code retains claims bound
// to (accepted generation, exact face coordinates, source, digest); a claim is
// never native proof and every publication recertifies it from its source.
struct CurvedSourceDeclaration {
  int64_t source_count = 0;
  const int32_t* sources = nullptr;
  const double* tolerances = nullptr;
  const std::uint8_t* digests = nullptr;
  const double* face_claims = nullptr;
};

// Claims supplied for one curved boundary collapse: exactly the proposal's
// new constrained faces (vertex triples and their exact coordinates), each
// with its host-certified deviation, under the declared source and digest of
// the accepted state `generation` they were computed from.
struct CurvedCollapseClaims {
  int32_t source = -1;
  int64_t generation = -1;
  const std::uint8_t* digest = nullptr;
  int64_t count = 0;
  const int32_t* faces = nullptr;       // count x 3
  const double* coordinates = nullptr;  // count x 9
  const double* deviations = nullptr;   // count
};

// A validated, unpublished curved collapse candidate: its curved source,
// the accepted generation it was validated in and its new constrained faces,
// each oriented from the source stratum's first material side.
struct CurvedCollapseInspection {
  int32_t source = -1;
  int64_t generation = -1;
  NativeVector<std::array<int32_t, 3>> faces;
};

// One retained delegated host claim: a curved source row and its binding.
// It is host evidence for a curved authority, never a planar PLC proof.
// generation is the accepted state in which the row entered the source bank,
// not the predecessor generation used to authenticate the proposed edit.
struct CurvedRowClaim {
  int32_t row;
  int32_t source;
  double deviation;
  int64_t generation;
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

// Cumulative work estimates retain the existing admission granularity.
// Zero-unit checkpoints poll the enclosing execution allowance without charging.
struct TetWorkCost final {
  static constexpr int64_t kCheckpoint = 0;
  static constexpr int64_t kVisit = 1;
  static constexpr int64_t kEndpointPair = 2;
  static constexpr int64_t kCoordinates = 3;
  static constexpr int64_t kFacet = 4;
  static constexpr int64_t kCollinearityPair = 6;
  static constexpr int64_t kCell = 8;
  static constexpr int64_t kSourceConstruction = 16;
  static constexpr int64_t kCellValidation = 16;
  static constexpr int64_t kSourceFacetConstruction = 32;
};

class TetMesh {
 public:
  TetMesh(BoundaryPolicy policy, int64_t max_vertices, int64_t max_tetrahedra,
          std::size_t max_allocation_bytes = std::numeric_limits<std::size_t>::max());
  TetMesh(BoundaryPolicy policy, int64_t max_vertices, int64_t max_tetrahedra,
          MemoryOwner owner);
  TetMesh(const TetMesh&) = delete;
  TetMesh& operator=(const TetMesh&) = delete;

  // ---------------------------------------------------------------- build
  int32_t build(int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
                const int32_t* regions, int64_t face_count, const int32_t* faces,
                const int32_t* face_sources, int64_t segment_count, const int32_t* segments,
                const int32_t* segment_sources, const double* protection,
                const SourceComplex* source = nullptr,
                const CurvedSourceDeclaration* curved = nullptr);

 private:
  int32_t prepare_source_points(int64_t source_count, const SourceComplex& source);
  static bool source_vertex_admitted(int32_t vertex, int64_t count);
  static bool source_bound_admitted(double value);
  int32_t build_source_faces(int64_t source_count, const SourceComplex& source);
  int32_t build_source_segments(int64_t source_count, const SourceComplex& source);
  void build_source_face_index();
  void source_face_candidates(int32_t source, const double* lower, const double* upper,
                              NativeVector<int32_t>& rows) const;
  void tighten_source_segment_bounds();
  int32_t build_source_witnesses(int64_t point_count, const SourceComplex& source);

  int32_t build_state(int64_t point_count, const double* points, int64_t tet_count,
                      const int32_t* tets, const int32_t* regions, int64_t face_count,
                      const int32_t* faces, const int32_t* face_sources,
                      int64_t segment_count, const int32_t* segments,
                      const int32_t* segment_sources, const double* protection,
                      const SourceComplex* source, const CurvedSourceDeclaration* curved);

  // Retains the curved declaration on the own source complex (row i is input
  // face i): every face of a declared source carries a finite claim within
  // its tolerance and no other face carries one.
  int32_t declare_curved_sources(int64_t face_count, const int32_t* face_sources,
                                 const CurvedSourceDeclaration& curved);

  // Source rows, their declared bounds and the natively certified witness
  // of every initial point (deviation recomputed, never trusted).
  int32_t build_source(int64_t point_count, const SourceComplex& source);

  // Original exact boundary vertices also name a real source row. A missing
  // input witness is completed from its represented constrained support, never
  // interpreted as an unconstrained boundary authority.
  int32_t complete_boundary_witnesses();

  // Every constrained face is covered exactly by original primitive rows of
  // its explicit source; segments retain one primitive ancestor. Only boundary
  // strata carry per-vertex primitive witnesses.
  int32_t verify_ancestry(int64_t point_count, int64_t face_count, const int32_t* faces,
                          const int32_t* face_sources, int64_t segment_count,
                          const int32_t* segments, const int32_t* segment_sources) const;

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
  void mark_accepted_state() noexcept { cancel_prepared_edit(); ++accepted_generation_; }
  void set_work_limit(int64_t budget);
  int64_t work_ceiling() const noexcept { return work_limit_; }
  void restore_work_ceiling(int64_t ceiling) noexcept {
    work_limit_ = ceiling;
    work_exhausted_ = false;
  }
  bool spend(int64_t units);
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
    const auto found = segments_.find(topology::undirected_edge(a, b));
    return found == segments_.end() ? -1 : found->second;
  }
  const NativeUnorderedMap<std::uint64_t, int32_t>& segments() const { return segments_; }

  // Source rows are ancestry identities, independent of source IDs that may
  // identify several triangles. Original coordinates never relocate.
  const double* original_point(int32_t vertex) const {
    return original_points_.data() + 3 * static_cast<std::size_t>(vertex);
  }
  void row_corners(SourceStratum stratum, std::size_t row, const double** corners) const;
  double row_tolerance(SourceStratum stratum, int32_t row) const {
    return stratum == SourceStratum::kSegment ? segment_tolerance_[static_cast<std::size_t>(row)]
                                              : face_tolerance_[static_cast<std::size_t>(row)];
  }
  // Exact witness S membership in a source triangle. The binary carrier may
  // be off-plane, so ancestry is decided from the authored row and parameters.
  bool witness_on_facet(const SourceWitness& witness, std::size_t row) const;

  // Whether vertex v lies on a source row: exactly, or by its bounded witness
  // on that row (a bounded segment witness lies on every triangle row that
  // contains its whole segment row).
  bool on_source_row(int32_t v, SourceStratum stratum, std::size_t row) const;
  // One primitive containing all support vertices, or -1. This is a
  // primitive construction query, not the logical-facet ancestry contract.
  int32_t source_ancestor(SourceStratum stratum, int32_t source,
                          std::span<const int32_t> support) const;
  // Sorted positive-area contributors covering the complete facet exactly
  // once. Multi-primitive coverage is certified in the exact source plane;
  // bounded carriers require a single convex primitive.
  bool face_ancestry(const int32_t* vertices, int32_t source,
                     NativeVector<int32_t>& contributors) const;
  bool facet_witness_admitted(const int32_t* vertices, int32_t source,
                              const SourceWitness& witness) const;
  // Exact memo of the constrained-face ancestry CSR published by source
  // evidence. A lookup reuses it only when every face (vertices, source),
  // every face-vertex coordinate and every face-vertex witness equals the
  // stored inputs bit for bit (one visit charged per face); build() and any
  // differing input invalidate it. Storage lives in the mesh's managed pool
  // and is attempted only when that pool reports room; a refused store
  // leaves the memo empty and never changes the published evidence.
  bool cached_face_ancestry(std::span<const std::array<int32_t, 4>> faces,
                            NativeVector<int64_t>& offsets, NativeVector<int32_t>& rows) const;
  void store_face_ancestry(std::span<const std::array<int32_t, 4>> faces,
                           std::span<const int64_t> offsets,
                           std::span<const int32_t> rows) const;
  // Returns the memo's managed storage; build() and a memo miss call it. An
  // accepted change keeps the memo: a lookup decides reuse from the faces,
  // coordinates and witnesses themselves, and a no-op change must not drop it.
  void release_face_ancestry() const noexcept;
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
  // Declared curved tolerance of face source `source`, or NaN when the
  // source is not curved.
  double curved_tolerance(int32_t source) const;
  const std::uint8_t* curved_digest(int32_t source) const;
  // Every retained host claim (initial and committed curved rows), by row.
  std::span<const CurvedRowClaim> curved_claims() const { return curved_claims_; }
  void clear_source_refusals() { source_refusals_.clear(); }
  bool protect_vertices(int64_t count, const int32_t* vertices);
  // Maximum certified deviation of the live vertices from their source rows:
  // every constrained face and segment lies within it of its source row
  // (convexity of the source entities). Exact meshes report zero.
  double achieved_source_error_bound() const;
  // Largest declared bound of any source row.
  double requested_source_tolerance() const;

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
                            double preferred_fraction = 0.0);
  bool construct_facet_point(int32_t a, int32_t b, int32_t c, double* position);
  // Rounded carrier of an interior fraction of an edge without a segment,
  // strictly inside it along its dominant axis. It is an interior vertex
  // candidate only: constrained faces through the edge still require their
  // plane to contain the final position, and cavity orientation decides.
  bool construct_interior_edge_point(int32_t a, int32_t b, double fraction, double* position);

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
  bool source_position(int32_t vertex, Expansion* coordinates) const;

  int source_orientation(const int32_t* v, int carrier_sign = 2) const;
  bool positive(const int32_t* v) const {
    const int carrier = orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3]));
    return carrier > 0 && source_orientation(v, carrier) > 0;
  }
  // Whether existing vertex e lies strictly inside the circumsphere of finite
  // tetrahedron t: exact insphere on the binary64 carriers, or on the exact
  // source positions when any of the five vertices has a bounded witness
  // (false when such a position is unavailable).
  bool vertex_conflicts(int32_t t, int32_t e) const;
  // orient3d of tetrahedron t with v[slot] replaced by p.
  int orient_with(int32_t t, int slot, const double* p) const;
  double min_dihedral(int32_t t) const;

  // ------------------------------------------------------------- topology
  // Live tetrahedra (ghosts included) incident to vertex a, in a
  // deterministic breadth-first order from its recorded incident tetrahedron.
  template <class Allocator>
  bool vertex_star(int32_t a, std::vector<int32_t, Allocator>& star);

  // A live tetrahedron containing edge (a, b), or -1.  The breadth-first
  // star walk of a stops at the first cell (in vertex_star order) that
  // contains b, so only the visits up to it are made and charged.
  int32_t edge_tet(int32_t a, int32_t b);

  // A finite tetrahedron with face (a, b, c) and the slot opposite it: the
  // first such cell in vertex_star order of a, found by the same early-exit walk.
  bool find_face(int32_t a, int32_t b, int32_t c, int32_t& t, int& slot);

  // Target size at p by clamped barycentric interpolation in finite
  // tetrahedron t (0 when no vertex carries a target).
  double interpolated_size(int32_t t, const double* p) const;

  // Tetrahedra around edge (a, b) in rotation order: tets[i] is (a, b,
  // ring[i], ring[i + 1]) as an even permutation; ring may hold the ghost
  // vertex.  False when the edge does not exist.
  template <class TetAllocator, class RingAllocator>
  bool edge_ring(int32_t a, int32_t b, std::vector<int32_t, TetAllocator>& tets,
                 std::vector<int32_t, RingAllocator>& ring);

  // ------------------------------------------------------------ insertion
  // Visibility walk from `start` toward p that never crosses a constrained
  // face: returns a tetrahedron whose closure contains p, or -1 with
  // blocked_tet/blocked_slot naming the constrained face that stopped it
  // (blocked_tet < 0 when the work limit stopped it).
  int32_t locate(const double* p, int32_t start, int32_t& blocked_tet, int& blocked_slot);

  // One exact old-edge witness/source-interval admission, shared by both
  // fixed-star subdivision and expanded constrained conflict-cavity insertion.
  bool split_positions_admitted(int32_t a, int32_t b, const double* witness,
                                const double* position, int32_t& source,
                                bool& on_edge) const;

  // Locate and prepare the existing exact conflict-cavity insertion while
  // retaining the seed edge's scientific source stratum and protection balls.
  Insertion prepare_edge_insertion(int32_t a, int32_t b, const double* witness,
                                   const double* position, double source_fraction,
                                   std::size_t cell_limit,
                                   InsertKind& kind);

  // Prepares the insertion of p (not yet a vertex) located in `origin`:
  // the conflict region, crossing only constrained faces coplanar with p when
  // the policy allows splits, shrunk until star-shaped from p.  The cavity,
  // its boundary and crossed faces are then available for inspection.
  Insertion prepare(const double* p, int32_t origin, int32_t split_a = -1,
                    int32_t split_b = -1,
                    std::size_t cell_limit = std::numeric_limits<std::size_t>::max(),
                    const SourceWitness* construction = nullptr);

  std::span<const int32_t> cavity() const { return cavity_; }
  std::span<const BoundaryFacet> cavity_boundary() const { return boundary_; }

  // Whether tetrahedron t belongs to the pending conflict cavity: O(1) from
  // the preparation stamp (no other walk may run between prepare and use).
  bool in_prepared_cavity(int32_t t) const {
    return pending_ != kNoPending && !pending_edge_star_ && marked(t);
  }

  // Constrained faces interior to the pending conflict cavity (both sides
  // removed). Every face created by an insertion contains the new vertex,
  // so a committed insertion retires these vertex triples permanently.
  std::size_t crossed_count() const {
    return pending_ == kNoPending || pending_edge_star_ ? 0 : crossed_.size();
  }
  FaceKey crossed_face(std::size_t i) const;

  // Commits the prepared insertion; the new vertex gets `size`.
  Insertion commit(InsertKind kind, double size);
  Insertion prepare_projected_edge_insertion(
      int32_t a, int32_t b, const double* witness, const double* position,
      double fraction, std::size_t cell_limit, int32_t source, InsertKind& kind);
  std::span<const std::array<int32_t, 3>> projected_insertion_faces() const {
    return retained_edit_.binding_.faces;
  }
  Insertion authenticate_projected_insertion(const CurvedCollapseClaims& claims);

  // Abandons a prepared insertion (nothing was written).
  void abandon() {
    cancel_prepared_edit();
  }

  std::span<const int32_t> created() const { return edit_.created(); }

  // ------------------------------------------------------- general edits
  // Bisects the complete edge star, including its ghost closure. Unlike a
  // Delaunay insertion this operation cannot delete neighboring vertices.
  Insertion split_edge(int32_t a, int32_t b, const double* position, double size,
                       int32_t& inserted);

  // The same source witness admission used by the eventual cavity commit,
  // without publishing a point or changing topology.
  Insertion edge_construction_witness(int32_t a, int32_t b, const double* position,
                                      SourceWitness& witness);

  // The witness is strictly inside the represented old edge. Only the final
  // position is staged: no intermediate split vertex is ever published.
  Insertion split_edge_relocate(int32_t a, int32_t b, const double* split_position,
                                const double* position, double size,
                                int32_t& inserted);

  // Replaces the complete star of an unprotected edge by cones from one new
  // vertex at `position`, which need not lie on the edge. Constrained or
  // interface faces containing the edge must contain the position exactly
  // (the shared surface-preservation proof), so no segment, constrained
  // patch or region changes; the exact orientation owner validates children.
  Insertion insert_edge_star_vertex(int32_t a, int32_t b, const double* position,
                                    double size, int32_t& inserted);

  // Every source facet in the edge star must contain the authored S and admit
  // its RNE carrier error, not merely the row chosen to encode the witness.
  Insertion admit_split_facets(int32_t a, int32_t b, const SourceWitness& witness,
                               std::span<const int32_t> star);

  // Bounded constrained split of represented edge (a, b): the correctly
  // rounded carrier of the exact witness at `fraction` between the source
  // parameters of a and b on the source row holding the edge (its segment
  // row, or the triangle row of a constrained face through it), strictly
  // inside (a, b) along its dominant axis. kOk when that row's declared bound
  // admits the certified deviation; kDeviation (recorded) above it; kRefused
  // for an unconstrained or fixed edge, a missing row or no distinct carrier.
  Insertion construct_source_split(int32_t a, int32_t b, double fraction, double* position,
                                   SourceWitness& witness);

  // Commits a bounded split constructed by construct_source_split: the star
  // of (a, b) is bisected at the carrier, whose witness is recertified here.
  // Children inherit every region and constraint mark; the exact orientation
  // owner validates them, so tolerance never crosses a stratum or region.
  Insertion prepare_source_facet(int32_t t, int slot, const double* center,
                                double* position, SourceWitness& witness);

  Insertion split_edge_bounded(int32_t a, int32_t b, const double* position,
                               const SourceWitness& witness, double size, int32_t& inserted);

  // Witness of constrained vertex v moved to `position`, which the caller has
  // proved to lie exactly on every represented plane (and, on a curve, the
  // represented line) of its star; interior vertices carry none.
  Insertion relocation_witness(int32_t v, const double* position, SourceWitness& witness);
  void set_witness(int32_t v, const SourceWitness& witness) {
    witnesses_[static_cast<std::size_t>(v)] = witness;
  }

 private:
  Insertion admit_edge_construction(int32_t a, int32_t b, const double* witness, const double* position, double source_fraction, int32_t& source, bool& bounded, SourceWitness& construction);
  Insertion select_edge_insertion_seed(int32_t a, int32_t b, const double* position, int32_t source, bool bounded, std::span<const int32_t> star, InsertKind& kind, int32_t& seed);
  Insertion admit_cavity_protection(const double* position);

  struct Crossed {
    int32_t tet;
    int32_t slot;
    int32_t source;
  };

  // Both direct and inspected star routes retain the same protection, source,
  // surface and work admission before their one atomic cavity transaction.
  Insertion edge_star_preparation(int32_t a, int32_t b, const double* position,
                                 int32_t source, bool on_edge,
                                 const SourceWitness* bounded,
                                 NativeVector<int32_t>& star,
                                 NativeVector<Crossed>& retired,
                                 bool& constrained, SourceWitness& witness);

  // Inspect the existing bounded complete-star route when its Delaunay
  // conflict preparation refuses. No source witness or cell budget is relaxed.
  Insertion prepare_bounded_edge_star(int32_t a, int32_t b, const double* position,
                                     int32_t source, const SourceWitness& bounded,
                                     std::size_t cell_limit);

  // `bounded` carries a recertified ancestry-backed witness (the surface is
  // then bounded by that witness, not preserved exactly); otherwise the new
  // vertex's witness is derived from its exact represented support.
  Insertion split_edge_star(int32_t a, int32_t b, const double* position, double size,
                            int32_t source, bool on_edge, const SourceWitness* bounded,
                            int32_t& inserted);

  // Third vertex of the face of finite tetrahedron t opposite slot k that
  // contains edge (a, b).
  int32_t star_apex(int32_t t, int k, int32_t a, int32_t b) const;

  // A cross-row locator must encode the SAME exact S, never the off-plane P.
  // Non-dyadic row coordinates cannot use binary64-parameter witnesses.
  bool exact_row_parameter(const Expansion& numerator, const Expansion& denominator,
                           double& parameter);

  bool row_parameters(int32_t v, SourceStratum stratum, int32_t row,
                      const double* const* corners, double* parameters);

  // Whether the position is the correctly rounded carrier of the witness on
  // source row `row`: every bounded vertex is, so its exact source point is
  // the authoritative coordinate and the binary64 point its RNE carrier.
  bool rounded_carrier(SourceStratum stratum, int32_t row, const SourceWitness& witness,
                       const double* position) const;

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
                           SourceWitness& witness);
  // Bounded relocation must retain a convex primitive witness for every
  // incident facet, not merely the first source used to construct the point.
  Insertion admit_relocation_witness(int32_t v, const double* position,
                                     const SourceWitness& witness);

  // First source row of `source` containing the position exactly, or -1.
  int32_t position_row(SourceStratum stratum, int32_t source, const double* position);

  // Witness of the vertex splitting edge (a, b) with star `star`: its
  // segment, else a constrained face through the edge, else none.
  Insertion star_witness(int32_t a, int32_t b, std::span<const int32_t> star,
                         const double* position, int32_t source, bool constrained,
                         SourceWitness& witness);

  // Witness of the pending Delaunay point: on its subsegment, or on a
  // crossed or cavity-boundary constrained face containing it (it lies
  // exactly in the plane of every crossed face), or none for an interior
  // point.
  Insertion pending_witness(InsertKind kind, SourceWitness& witness);

 public:
  // Contract a removable boundary Steiner's complete closed star. The caller
  // proves the volume link condition; each exact planar source/material patch
  // proves its own link, oriented boundary-chain and area conservation. Actual
  // represented planes, not equal source labels alone, establish equivalence.
  EditStatus collapse_boundary_edge(int32_t remove, int32_t keep,
                                   std::span<const int32_t> star,
                                   std::span<const int32_t> keep_star);

  // Contract an unprotected vertex interior to one declared curved face
  // source (no incident segment) onto `keep`. The caller proves the volume
  // link condition and positive star cells. The patch keeps one material
  // stratum, its surface link condition and its exact oriented boundary
  // chain, and the staged cells validate. Inspection reports the validated
  // candidate and retains the existing edit owner with all accepted constraint
  // marks restored. A commit requires exactly the new constrained faces and
  // bitwise current coordinates in the supplied vertex order, the source, its
  // declared digest and current accepted generation, each deviation finite and
  // within tolerance. Native candidate orientation remains independently proved;
  // delegated row permutations retain their existing source-point identities.
  // Committed faces become delegated-claim
  // rows of the curved source.
  EditStatus collapse_curved_boundary_edge(int32_t remove, int32_t keep,
                                           std::span<const int32_t> star,
                                           std::span<const int32_t> keep_star,
                                           const CurvedCollapseClaims* claims,
                                           CurvedCollapseInspection* inspection);

 private:
  struct PreparedBinding {
    enum class Operation { kReplacement, kCollapse, kInsertion };
    Operation operation = Operation::kReplacement;
    int32_t first = -1, second = -1, source = -1;
    bool require_improvement = false;
    InsertKind kind = InsertKind::kInterior;
    std::array<double, 7> input{};
    std::array<double, 6> endpoints{};
    std::array<std::uint8_t, 32> digest{};
    double tolerance = 0.0;
    std::size_t cell_limit = 0, cavity_size = 0;
    NativeVector<std::array<int32_t, 3>> faces;
    NativeVector<Crossed> retired;
    NativeUnorderedMap<std::uint64_t, int32_t> segments;
    int32_t projected_source = -1;
    NativeVector<double> deviations;
  };

 public:
  class PreparedReplacement {
   public:
    PreparedReplacement() = default;
    PreparedReplacement(const PreparedReplacement&) = delete;
    PreparedReplacement& operator=(const PreparedReplacement&) = delete;
    PreparedReplacement(PreparedReplacement&& other) noexcept
        : owner_(other.owner_), generation_(other.generation_),
          revision_(other.revision_), status_(other.status_),
          binding_(std::move(other.binding_)) { other.owner_ = nullptr; }
    PreparedReplacement& operator=(PreparedReplacement&& other) noexcept {
      if (this != &other) {
        discard();
        owner_ = other.owner_;
        other.owner_ = nullptr;
        generation_ = other.generation_;
        revision_ = other.revision_;
        status_ = other.status_;
        binding_ = std::move(other.binding_);
      }
      return *this;
    }
    ~PreparedReplacement() { discard(); }
    EditStatus status() const noexcept { return status_; }
    EditStatus commit() {
      if (!current()) {
        return EditStatus::kBoundaryMismatch;
      }
      TetMesh* owner = owner_;
      const EditStatus result = owner->commit_prepared_replacement();
      owner_ = nullptr;
      return result;
    }
   private:
    bool current() const noexcept {
      return owner_ != nullptr && status_ == EditStatus::kOk &&
             owner_->accepted_generation_ == generation_ &&
             owner_->edit_.staging_revision() == revision_;
    }
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
    PreparedBinding binding_;
  };

  // The one existing edit_ retains the validated source/region/cavity proof.
  // The scoped token never copies a mesh, source bank or allocation arena.
  PreparedReplacement prepare_replacement(std::span<const int32_t> removed,
                     std::span<const std::array<int32_t, 4>> proposed);

  // There is one validated CavityEdit, owned either by a scoped replacement
  // token or this retained token, never a parallel prepared-edit cache.
  void cancel_prepared_edit() noexcept {
    const uint64_t revision = edit_.staging_revision();
    retained_edit_ = PreparedReplacement{};
    if (edit_.staging_revision() == revision) edit_.rollback();
    pending_ = kNoPending;
    pending_edge_star_ = false;
    pending_projected_source_ = -1;
  }
  bool reuse_curved_collapse(int32_t remove, int32_t keep, bool require_improvement,
                             const CurvedCollapseClaims& claims, bool& applied);
  bool retained_curved_collapse(int32_t remove, int32_t keep, int64_t generation) const noexcept {
    return retained_edit_.current() && accepted_generation_ == generation &&
        retained_edit_.binding_.operation == PreparedBinding::Operation::kCollapse &&
        retained_edit_.binding_.first == remove && retained_edit_.binding_.second == keep;
  }
  std::span<const int32_t> retained_removed() const { return edit_.removed(); }
  std::span<const Tetrahedron> retained_proposed() const { return edit_.proposal(); }
  void bind_curved_collapse(bool require_improvement) noexcept {
    retained_edit_.binding_.require_improvement = require_improvement;
  }
  Insertion retain_edge_insertion(int32_t a, int32_t b, const double* witness,
                                  const double* position, double fraction,
                                  std::size_t cell_limit, InsertKind kind);
  bool reuse_edge_insertion(int32_t a, int32_t b, const double* witness,
                            const double* position, double fraction,
                            std::size_t cell_limit, InsertKind& kind,
                            int32_t projected_source = -1) const noexcept;

  // Replaces `removed` by `proposed` tetrahedra (region of the first removed
  // tetrahedron, no new constraints) through one validated transaction.
  EditStatus replace(std::span<const int32_t> removed,
                     std::span<const std::array<int32_t, 4>> proposed);

  // Marks a vertex removed after an edit that deleted its star; a retired
  // vertex carries no source witness.
  void retire_vertex(int32_t v);

  // ----------------------------------------------------------- evidence
  // Canonical faces: (a, b, c, source) per constrained finite face, boundary
  // faces oriented outward, interfaces outward from the lower region.
  template <class Allocator>
  void collect_faces(std::vector<std::array<int32_t, 4>, Allocator>& faces) const;

  // Canonical finite cells (even permutation from the smallest vertex),
  // lexicographically sorted, with regions.
  template <class Allocator>
  void collect_cells(std::vector<std::array<int32_t, 5>, Allocator>& cells) const;

  std::size_t retained_bytes() const noexcept { return memory_owner_->live_bytes(); }

 private:
  friend class MeasuredTetStage;
  EditStatus commit_prepared_replacement();
  EditStatus commit_curved_collapse(PreparedReplacement& prepared,
                                    const CurvedCollapseClaims& claims,
                                    bool claims_validated);
  Insertion stage_insertion(InsertKind kind);
  Insertion commit_retained_insertion(InsertKind kind, double size);
  static MemoryOwner checked_memory_owner(MemoryOwner owner);
  static MemoryOwner select_memory_owner(std::size_t bytes);
  static constexpr int32_t kNoPending = std::numeric_limits<int32_t>::min();
  static constexpr std::uint8_t kInCavity = 1;
  // A cavity cell queued for re-examination by shrink (still in the cavity).
  static constexpr std::uint8_t kInCavityRecheck = 2;


  struct BoundaryCurve {
    int32_t other = -1;
    int32_t source = -1;
    int degree = 0;
    bool keep = false;
  };
  using BoundaryStratum = std::array<int32_t, 3>;
  using BoundaryLinkSimplex = std::array<int32_t, 2>;
  struct BoundarySurfaceProof {
    Expansion area_delta;
    NativeSet<BoundaryLinkSimplex> left, right, edge;
    NativeMap<std::uint64_t, int64_t> boundary_delta;
  };
  struct BoundaryFaceProof {
    BoundaryStratum stratum;
    int32_t source;
    int axis;
    int reference_sign;
    int sides[2] = {0, 0};
  };
  struct BoundarySourcePlane {
    std::array<int32_t, 3> vertices;
    int32_t source;
    int axis;
    int reference_sign;
  };
  struct BoundaryCollapseProof {
    NativeVector<BoundarySourcePlane> planes;
    NativeMap<BoundaryStratum, BoundarySurfaceProof> surfaces;
    NativeMap<FaceKey, BoundaryFaceProof> faces;
    NativeVector<Crossed> retired;
  };

  EditStatus collect_boundary_curve(int32_t remove, int32_t keep,
                                    std::span<const int32_t> star, BoundaryCurve& curve);
  EditStatus admit_boundary_curve(int32_t remove, int32_t keep, const BoundaryCurve& curve, NativeUnorderedMap<std::uint64_t, int32_t>& staged_segments);
  Expansion boundary_projected_area(const int32_t* f, int axis) const;
  bool describe_boundary_face(int32_t t, int slot, const int32_t* f, BoundaryStratum& stratum, int& axis, int& reference_sign, bool add_plane, BoundaryCollapseProof& proof);
  static void add_boundary_link(const int32_t* f, int32_t a, int32_t b, NativeSet<BoundaryLinkSimplex>& link);
  static void add_boundary_chain(const int32_t* f, int64_t coefficient, NativeMap<std::uint64_t, int64_t>& boundary);
  EditStatus collect_removed_boundary(int32_t remove, int32_t keep, std::span<const int32_t> star, BoundaryCollapseProof& proof_state);
  EditStatus collect_kept_boundary(int32_t keep, std::span<const int32_t> keep_star, BoundaryCollapseProof& proof_state);
  EditStatus certify_boundary_chain(const NativeMap<std::uint64_t, int64_t>& boundary_delta);
  EditStatus certify_boundary_surface(const BoundarySurfaceProof& proof);
  bool check_boundary_side(const int32_t* f, int32_t region, BoundaryFaceProof& proof) const;
  EditStatus stage_boundary_collapse(int32_t remove, int32_t keep, std::span<const int32_t> star, int32_t curve_other, BoundaryCollapseProof& proof_state);


  // Entirely allocation-free, including restoration after a refused request
  // during validation/reservation while constraint marks are retired.
  class EditTransaction {
   public:
    EditTransaction(TetMesh& mesh, std::span<const Crossed> retired) noexcept
        : mesh_(mesh), retired_(retired) {}
    ~EditTransaction() {
      if (active_) {
        mesh_.edit_.rollback();
        restore_constraints();
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
    void restore_constraints() noexcept {
      if (marks_retired_) {
        for (const Crossed& face : retired_) {
          mesh_.complex_.set_facet_constraint(face.tet, face.slot, face.source);
        }
        marks_retired_ = false;
      }
      mesh_.transaction_open_ = false;
    }
    void retain() noexcept {
      restore_constraints();
      active_ = false;
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
                               const double* position);

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
                      const int32_t* regions);

  // Pairs the ghost facets (G, p, q) and (G, q, p) around each boundary edge
  // in a deterministic order, then enables labels and assigns regions.
  int32_t link_ghosts(std::span<const int32_t> ghosts, int64_t point_count,
                      const int32_t* regions);

  const FacetRecord* find_record(const FaceKey& key) const {
    const FacetRecord probe{key, std::numeric_limits<int32_t>::min(), 0};
    const auto found = std::lower_bound(records_.begin(), records_.end(), probe);
    return found != records_.end() && found->key == key ? &*found : nullptr;
  }

  int32_t build_constraints(int64_t point_count, int64_t face_count, const int32_t* faces,
                            const int32_t* sources);

  int32_t build_segments(int64_t point_count, int64_t segment_count, const int32_t* segments,
                         const int32_t* sources);

  // Vertex dimension: 0 corner (segment degree other than 2, or a bend or
  // source change between its two segments), 1 on a segment, 2 on a
  // constrained face, 3 interior; -1 unused.
  void classify_vertices();

  // ------------------------------------------------------ insertion steps
  void next_stamp();
  bool marked(int32_t t) const {
    const Tetrahedron& current = tet(t);
    return current.stamp == stamp_ &&
           (current.state == kInCavity || current.state == kInCavityRecheck);
  }
  void mark(int32_t t) {
    Tetrahedron& current = complex_.tets[static_cast<std::size_t>(t)];
    current.stamp = stamp_;
    current.state = kInCavity;
  }
  void unmark(int32_t t) { complex_.tets[static_cast<std::size_t>(t)].state = 0; }

  // First cell of vertex_star(a) order containing b (and, when `face`, also
  // c, finite cells only), or -1; the walk stops at that cell.
  int32_t first_star_cell(int32_t a, int32_t b, int32_t c, bool face);

  bool coincident(int32_t t, const double* p) const;

  bool conflicts(int32_t t, const double* p) const;

  bool contains(int32_t t, const double* p) const;

  Insertion grow(int32_t origin);

  // Removes cavity tetrahedra whose cone over a boundary facet would not be
  // positively oriented (and ghosts cut off from their finite facet) until
  // the cavity is star-shaped from p; the tetrahedra containing p must stay.
  // One sweep examines every cell; afterwards only the cavity neighbors of
  // dropped cells are re-examined (worklist), one charged unit per check.
  Insertion shrink(int32_t origin);

  // Boundary facets, crossed constrained faces and the new constrained faces
  // through p; refuses a cavity that would lose a vertex or a segment.
  Insertion collect();

  // Source of the facet of cone tetrahedron v (apex p at slot `apex`)
  // opposite v[j], or kNoConstraint.
  int32_t new_face_source(const int32_t* v, int apex, int j) const;

  void after_commit();

  static bool even_permutation(const int32_t* reference, const int32_t* order);

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
  // Immutable original-row broad phase, partitioned by logical source.
  // Bounds are exact coordinate extrema; internal nodes enclose both children.
  struct SourceFaceNode {
    double lower[3], upper[3];
    std::size_t left = 0, right = 0;
    int32_t row = -1;
  };
  NativeVector<SourceFaceNode> source_face_nodes_{
      NativeAllocator<SourceFaceNode>(memory_owner_)};
  NativeVector<std::pair<int32_t, std::size_t>> source_face_roots_{
      NativeAllocator<std::pair<int32_t, std::size_t>>(memory_owner_)};
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
  struct CurvedSource {
    int32_t source;
    double tolerance;
    std::array<std::uint8_t, 32> digest;
  };
  // Sorted by source; empty without a curved declaration.
  NativeVector<CurvedSource> curved_sources_{NativeAllocator<CurvedSource>(memory_owner_)};
  NativeVector<CurvedRowClaim> curved_claims_{NativeAllocator<CurvedRowClaim>(memory_owner_)};
  // Committed claimed rows (source, row), sorted; the authored-row index
  // never contains them, so candidate queries scan them separately.
  NativeVector<std::pair<int32_t, int32_t>> claimed_rows_{
      NativeAllocator<std::pair<int32_t, int32_t>>(memory_owner_)};
  // Face-ancestry memo (see cached_face_ancestry); mutable cache state only.
  mutable NativeVector<std::array<int32_t, 4>> ancestry_faces_{
      NativeAllocator<std::array<int32_t, 4>>(memory_owner_)};
  mutable NativeVector<double> ancestry_points_{NativeAllocator<double>(memory_owner_)};
  mutable NativeVector<SourceWitness> ancestry_witnesses_{
      NativeAllocator<SourceWitness>(memory_owner_)};
  mutable NativeVector<int64_t> ancestry_offsets_{NativeAllocator<int64_t>(memory_owner_)};
  mutable NativeVector<int32_t> ancestry_rows_{NativeAllocator<int32_t>(memory_owner_)};
  mutable bool ancestry_cached_ = false;
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
  int32_t pending_projected_source_ = -1;
  int32_t split_a_ = -1;
  int32_t split_b_ = -1;
  std::size_t preparation_cell_limit_ = std::numeric_limits<std::size_t>::max();
  NativeVector<int32_t> cavity_{NativeAllocator<int32_t>(memory_owner_)};
  NativeVector<int32_t> stack_{NativeAllocator<int32_t>(memory_owner_)};
  NativeVector<int32_t> scratch_star_{NativeAllocator<int32_t>(memory_owner_)};
  // Discarded edge rings, collapse neighbors and source-facet contributors.
  NativeVector<int32_t> scratch_ring_{NativeAllocator<int32_t>(memory_owner_)};
  NativeVector<int32_t> facet_contributors_{NativeAllocator<int32_t>(memory_owner_)};
  NativeVector<BoundaryFacet> boundary_{NativeAllocator<BoundaryFacet>(memory_owner_)};
  NativeVector<Crossed> crossed_{NativeAllocator<Crossed>(memory_owner_)};
  NativeVector<std::uint64_t> boundary_edges_{NativeAllocator<std::uint64_t>(memory_owner_)};
  NativeVector<std::pair<std::uint64_t, int32_t>> new_faces_{
      NativeAllocator<std::pair<std::uint64_t, int32_t>>(memory_owner_)};
  // Declared last so its destructor rolls back before edit_ is destroyed.
  PreparedReplacement retained_edit_;
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
