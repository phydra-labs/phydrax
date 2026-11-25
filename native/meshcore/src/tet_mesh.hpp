//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Persistent source-aware tetrahedral state and generation-checked topology
// transactions. Local slots are reusable implementation indices; logical IDs
// are caller-visible and never inferred from coordinates.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <string_view>

#include "boundary_policy.hpp"
#include "cavity.hpp"
#include "facet_coverage.hpp"
#include "plc3d.hpp"
#include "source_construction.hpp"
#include "zero_set_admission.hpp"

struct phx_mc_tet_mesh_snapshot_request;

namespace phx::mc {
namespace detail {
class LocalIncidence;
}
namespace test {
struct TetMeshInteriorInsertion;
}
struct TetQualityRequest;
// A sorted pair of vertex IDs (an edge), hashed for the segment lookups of
// refinement and improvement (find, insert, erase; never iterated).
struct EdgeIdHash {
  std::size_t operator()(const std::array<std::uint64_t, 2>& edge) const noexcept {
    const std::uint64_t mixed = (edge[0] * 0x9E3779B97F4A7C15ULL) ^ (edge[1] + (edge[1] << 21));
    return static_cast<std::size_t>((mixed ^ (mixed >> 29)) * 0xBF58476D1CE4E5B9ULL);
  }
};
using TetSegmentIndex = std::unordered_map<std::array<std::uint64_t, 2>, std::size_t, EdgeIdHash>;

enum class TetEditKind : std::uint8_t {
  kSplit,
  kCollapse,
  kFlip23,
  kFlip32,
  kRingReconnect,
  kRelocate,
  kRefineCell,
  kRetriangulate,
};
enum class TetEditStatus : std::uint8_t {
  kApplied,
  kRejectedTopology,
  kRejectedProtected,
  kInvalidGeometry,
  kUncertainPredicate,
  kStaleGeneration,
  kAllocationFailure,
  kIndexOverflow,
};

struct TetImplicitFacetRefusal {
  std::uint64_t generation = 0;
  std::uint64_t facet_id = 0;
  std::int32_t facet_source = 0;
  std::int8_t orientation = 0;
  std::array<std::uint64_t, 3> vertex_ids{};
  std::array<std::array<double, 3>, 3> points{};
  std::int64_t transversality_refusals = 0;
  std::int64_t distance_refusals = 0;
  TetEditStatus status = TetEditStatus::kInvalidGeometry;
  std::optional<std::array<double, 3>> distance_barycentric;
};

// Exact dyadic interpolation from one source lifted vertex to a new target.
struct TetVertexAncestry {
  std::uint64_t source_vertex_id = 0;
  std::array<std::int64_t, 3> source_winding{};
  double weight = 0.0;
};

struct TetSnapshotVertex {
  std::uint64_t id = 0;
  std::array<double, 3> carrier{};
  SourceWitness witness{};
  std::int8_t stratum_dimension = 3;
  double protection_radius = 0.0;
  bool protected_vertex = false;
  bool has_edge_parameter = false;
  std::array<std::uint64_t, 2> edge_endpoints{};
  double edge_parameter = 0.0;
  std::vector<TetVertexAncestry> ancestry;
};

struct TetSnapshotCell {
  std::uint64_t id = 0;
  std::array<std::uint64_t, 4> vertices{};
  std::int32_t region = kNoRegion;
  std::vector<std::uint64_t> ancestor_cell_ids;
  std::int32_t edit_kind = -1;
};

struct TetSnapshotFacet {
  std::uint64_t id = 0;
  std::array<std::uint64_t, 3> vertices{};
  std::int32_t source = kNoConstraint;
  std::int8_t orientation = 1;
  bool protected_facet = false;
};

struct TetSnapshotSegment {
  std::uint64_t id = 0;
  std::array<std::uint64_t, 2> vertices{};
  std::int32_t source = kNoConstraint;
  bool protected_segment = false;
};

// Each member maps a lifted vertex to the orbit representative with an affine
// isometry and integer lattice winding. Quotient cells are exported separately
// from the positively oriented lifted tetrahedra.
struct TetPeriodicVertexMember {
  std::uint64_t vertex_id = 0;
  std::array<std::int64_t, 3> winding{};
  std::array<double, 12> representative_to_member{};
};

struct TetPeriodicOrbit {
  std::uint64_t id = 0;
  std::uint64_t representative_vertex_id = 0;
  std::vector<TetPeriodicVertexMember> members;
};

struct TetPeriodicQuotientCell {
  // Group key anchored to a dimension-3 quotient entity ID for atomic edits.
  std::uint64_t orbit_id = 0;
  // Stable dimension-3 quotient entity ID from the identity map.
  std::uint64_t id = 0;
  std::array<std::uint64_t, 4> representative_vertices{};
  std::array<std::array<std::int64_t, 3>, 4> winding{};
  std::int32_t region = kNoRegion;
  std::vector<std::uint64_t> lifted_cell_ids;
};
struct TetPeriodicIdentity {
  bool bound = false;
  std::array<std::string, 6> ids{};
  friend bool operator==(const TetPeriodicIdentity&, const TetPeriodicIdentity&) = default;
};

// Caller-owned global-identity maps for each simplex dimension. Winding and
// corner arrays are flattened row-major using rank and arity respectively.
struct TetPeriodicEntityOrbitMap {
  std::int32_t dimension = 0;
  std::int32_t arity = 1;
  std::int32_t rank = 0;
  std::vector<std::uint64_t> lifted_entity_ids;
  std::vector<std::uint64_t> quotient_entity_ids;
  std::vector<std::int32_t> lifted_to_quotient;
  std::vector<std::int32_t> orientation_signs;
  std::vector<std::int64_t> winding_shifts;
  std::vector<std::int32_t> corner_permutations;
  std::vector<std::int32_t> representative_lifted_rows;
};

// The interval programs own their instruction/constant banks and one-box
// scratch. Staged copies share this single-threaded owner within one edit.
struct TetImplicitSource {
  remesh::IntervalBoxProgram value;
  remesh::IntervalBoxProgram gradient;
  std::vector<std::int32_t> groups;
  std::vector<std::int8_t> group_orientations;
  double tolerance = 0.0;
  std::int32_t orientation = 0;
  std::string source_id;
  std::string source_revision;
  std::string binding_id;
  std::string frame_id;
  mutable std::int64_t transversality_refusals = 0;
  mutable std::int64_t distance_refusals = 0;
  mutable bool last_inconclusive = false;
  // Results of this source's facet admissions; a cache, not mesh state.
  mutable remesh::AdmissionMemo admissions;
};

struct TetMeshSnapshot {
  std::uint64_t generation = 0;
  bool identity_bound = false;
  std::string source_id;
  std::string source_revision;
  std::string frame_id;
  std::string source_map_id;
  std::vector<TetSnapshotVertex> vertices;
  std::vector<TetSnapshotCell> cells;
  std::vector<TetSnapshotFacet> constrained_facets;
  std::vector<TetSnapshotSegment> constrained_segments;
  std::vector<double> source_points;
  std::vector<std::int32_t> source_faces;
  std::vector<std::int32_t> source_face_groups;
  std::vector<std::int64_t> source_face_ids;
  std::vector<double> source_face_tolerances;
  std::vector<std::int32_t> source_segments;
  std::vector<std::int32_t> source_segment_groups;
  std::vector<std::int64_t> source_segment_ids;
  std::vector<double> source_segment_tolerances;
  // Declared exact facet planes per source face row (9 values: origin, first
  // and second axis; NaN rows undeclared) and their authored tolerances.
  std::vector<double> source_face_planes;
  std::vector<double> source_face_plane_tolerances;
  std::vector<TetPeriodicOrbit> periodic_orbits;
  std::vector<TetPeriodicQuotientCell> quotient_cells;
  TetPeriodicIdentity periodic_identity;
  std::vector<TetPeriodicEntityOrbitMap> periodic_entity_maps;
};

struct TetEditVertex {
  // Existing ID for relocation; a never-used ID for a newly inserted vertex.
  std::uint64_t id = 0;
  std::array<double, 3> carrier{};
  // The exact source construction remains separate from its rounded carrier.
  SourceWitness witness{};
  double protection_radius = 0.0;
  bool protected_vertex = false;
  // Exact dyadic parameter on this logical edge; independent of SourceWitness.
  bool has_edge_parameter = false;
  // Distinguish an omitted edge-witness update from explicit clearing.
  bool edge_witness_supplied = false;
  std::array<std::uint64_t, 2> edge_endpoints{};
  double edge_parameter = 0.0;
  std::vector<TetVertexAncestry> ancestry;
};

struct TetEditCell {
  std::uint64_t id = 0;
  std::array<std::uint64_t, 4> vertices{};  // positively oriented order
  std::array<std::int32_t, 4> facet_sources{
      kNoConstraint, kNoConstraint, kNoConstraint, kNoConstraint};
  std::int32_t region = kNoRegion;
  std::vector<std::uint64_t> ancestor_cell_ids;
};

struct TetCellAncestryRecord {
  std::int32_t edit_kind = -1;
  // Immutable immediate ancestors; siblings produced by one cavity share
  // their identical ordered bank without discarding any historical record.
  std::shared_ptr<const std::vector<std::uint64_t>> ancestor_cell_ids;
};

struct TetEditFacet {
  std::uint64_t id = 0;
  std::array<std::uint64_t, 3> vertices{};
  std::int32_t source = kNoConstraint;
  std::int8_t orientation = 1;
  bool protected_facet = false;
};

struct TetEditSegment {
  std::uint64_t id = 0;
  std::array<std::uint64_t, 2> vertices{};
  std::int32_t source = kNoConstraint;
  bool protected_segment = false;
};

struct TetPeriodicEditMember {
  std::uint64_t vertex_id = 0;
  std::array<std::int64_t, 3> winding{};
  std::array<double, 12> representative_to_member{};
};
struct TetCollapsePair {
  std::uint64_t removed_vertex_id = 0;
  std::uint64_t survivor_vertex_id = 0;
};

struct TetEdit {
  TetEditKind kind = TetEditKind::kRingReconnect;
  std::vector<TetCollapsePair> collapse_pairs;
  std::vector<std::array<std::uint64_t, 2>> split_edges;
  std::vector<std::uint64_t> removed_vertex_ids;
  std::vector<std::uint64_t> removed_cell_ids;
  std::vector<TetEditVertex> changed_vertices;
  std::vector<TetEditCell> replacement_cells;
  std::vector<std::uint64_t> removed_facet_ids;
  std::vector<TetEditFacet> replacement_facets;
  std::vector<std::uint64_t> removed_segment_ids;
  std::vector<TetEditSegment> replacement_segments;
  std::vector<TetPeriodicEditMember> periodic_members;
  std::vector<std::uint64_t> removed_periodic_orbit_ids;
  std::vector<TetPeriodicOrbit> replacement_periodic_orbits;
  std::vector<std::uint64_t> removed_quotient_cell_ids;
  std::vector<TetPeriodicQuotientCell> replacement_quotient_cells;
  std::vector<TetPeriodicEntityOrbitMap> replacement_periodic_entity_maps;
};

struct TetEditBatch {
  std::uint64_t expected_generation = 0;
  std::string source_revision;
  std::string frame_id;
  bool identity_bound = false;
  std::vector<TetEdit> operations;
  std::string source_id;
  std::string source_map_id;
  // The current periodic identity (staleness is guarded with the generation).
  TetPeriodicIdentity periodic_identity;
  // A periodic edit rebinds the identity of the state it produces; absent, the
  // identity is kept.
  std::optional<TetPeriodicIdentity> target_periodic_identity;
};

struct TetEditResult {
  TetEditStatus status = TetEditStatus::kRejectedTopology;
  std::vector<std::uint64_t> accepted_vertex_ids;
  std::vector<std::uint64_t> accepted_cell_ids;
};
// Unmet-record apex vertex ID of a record without an acute-feature apex.
inline constexpr std::uint64_t kNoUnmetApex = std::numeric_limits<std::uint64_t>::max();
// Unmet-record facet ID of a record without a forcing constrained facet.
inline constexpr std::uint64_t kNoUnmetFacet = std::numeric_limits<std::uint64_t>::max();
struct TetUnmetRecord {
  std::uint64_t cell_id = 0;
  std::int32_t criterion = 0;
  std::int32_t reason = 0;
  double value = 0.0;
  // INPUT_ANGLE evidence: the acute input feature's apex vertex ID and its
  // angle in degrees; kNoUnmetApex and NaN for every other reason.
  std::uint64_t apex_vertex = kNoUnmetApex;
  double apex_angle = std::numeric_limits<double>::quiet_NaN();
  // Improvement FIXED_BOUNDARY evidence: the ID of the cell's constrained
  // facet whose own triangle radius-edge ratio (`facet_ratio`) exceeds the
  // bound; kNoUnmetFacet and NaN for every other record.
  std::uint64_t facet_id = kNoUnmetFacet;
  double facet_ratio = std::numeric_limits<double>::quiet_NaN();
};

// A boundary vertex's record on the declared exact plane of source face row
// `face_row`: the correctly rounded parameters of the orthogonal projection
// of its carrier onto the plane and an outward bound of their distance.
struct TetPlaneRecord {
  std::int32_t face_row = -1;
  double parameters[2] = {0.0, 0.0};
  double deviation = 0.0;
};
// Typed refusal of a declared plane: input vertex row `vertex` of a
// constrained facet on it is neither in the plane's rounding-box reach nor
// within its declared tolerance (record.deviation > tolerance).
struct TetPlaneRefusal {
  std::int64_t vertex = -1;
  TetPlaneRecord record;
  double tolerance = 0.0;
};
// Deviation of `position` from the point origin + parameters[0] * first axis
// + parameters[1] * second axis of the declared plane `plane` (nine values):
// zero when `position` lies exactly on the plane, else an outward bound of
// their distance; -1 for non-finite parameters.
double declared_plane_deviation(const double* plane, const double* parameters,
                                const double* position);


// Star-shaped cavity of a candidate interior point for cone insertion.
enum class TetCavityStatus : std::uint8_t {
  kFound,    // `cells` (root first) is star-shaped from the point
  kBlocked,  // the walk to the point leaves through a constrained or hull face
  kRefused,  // the walk cycles, or no star-shaped measurable cavity remains
};
struct TetInsertionCavity {
  std::vector<std::int32_t> cells;
  std::unordered_set<std::int32_t> members;
  std::int32_t root = -1;
  std::int32_t blocked_cell = -1;  // kBlocked: the face `blocked_face` of this
  int blocked_face = -1;           // cell separates the point
};

// Tables of the replacement refusal probe (TetMesh::probe_replacement_edit),
// valid for one generation: whether that state passes every check that
// validate_and_commit applies to cells, facets and segments a replacement edit
// retains, and the derived source-side regions and facet indexes.
struct TetReplacementProbeCache {
  std::uint64_t generation = std::numeric_limits<std::uint64_t>::max();
  bool certified = false;
  std::map<std::pair<std::int32_t, int>, std::int32_t> source_regions;
  std::unordered_map<std::uint64_t, std::size_t> facet_row_by_id;
  std::map<std::array<std::int32_t, 3>, std::size_t> facet_row_by_face;
};

// Logical IDs of one entity kind: which were ever used and the slot of each
// live one. IDs handed out by the counters are dense and index a flat table;
// a caller-chosen ID far beyond every used one goes to a sparse map, so the
// table stays proportional to the number of used IDs.
class TetIdIndex {
 public:
  static constexpr std::int32_t kUnused = -1;
  static constexpr std::int32_t kRetired = -2;  // used, no longer live
  static constexpr std::int32_t kPending = -3;  // live, slot not yet assigned

  void clear() noexcept;
  void reserve(std::size_t count);
  // Replaces the contents by the IDs 0 .. count - 1, each live at its own slot.
  void assign_identity(std::size_t count);
  // The slot of a live ID, else -1.
  std::int32_t slot(std::uint64_t id) const noexcept {
    const std::int32_t value = state(id);
    return value >= 0 ? value : -1;
  }
  bool used(std::uint64_t id) const noexcept { return state(id) != kUnused; }
  // Records `id` as used and live at `slot` (a slot or kPending).
  void bind(std::uint64_t id, std::int32_t slot);
  // Records `id` as used and not live.
  void mark_used(std::uint64_t id);
  // Moves a live ID to `slot`.
  void set_slot(std::uint64_t id, std::int32_t slot) noexcept;
  // A live ID stays used but is no longer live.
  void retire(std::uint64_t id) noexcept;
  // Forgets `id` entirely (rollback of a staged binding).
  void forget(std::uint64_t id) noexcept;
  std::size_t live_count() const noexcept { return live_; }
  // Calls visit(ids, slots) on consecutive blocks (spans of at most
  // kVisitBlock) of the live IDs in ascending order and their slots; false
  // when a live ID has no slot yet (kPending), whose block is not visited.
  static constexpr std::size_t kVisitBlock = 512;
  template <class Visit>
  bool visit_live(Visit&& visit) const {
    std::array<std::uint64_t, kVisitBlock + 1> ids;
    std::array<std::int32_t, kVisitBlock + 1> slots;
    std::size_t count = 0;
    bool pending = false;
    // Branch-free compaction: the table is mostly retired IDs.
    const auto take = [&](std::uint64_t id, std::int32_t slot) {
      ids[count] = id;
      slots[count] = slot;
      count += slot >= 0 ? 1 : 0;
      pending |= slot == kPending;
      if (count == kVisitBlock) {
        if (pending) return false;
        visit(std::span<const std::uint64_t>(ids.data(), count),
              std::span<const std::int32_t>(slots.data(), count));
        count = 0;
      }
      return true;
    };
    for (std::size_t id = 0; id < dense_.size(); ++id) {
      if (!take(static_cast<std::uint64_t>(id), dense_[id])) return false;
    }
    // Sparse IDs all lie beyond the dense table (it stops growing once one exists).
    if (!sparse_.empty()) {
      std::vector<std::pair<std::uint64_t, std::int32_t>> sparse(sparse_.begin(), sparse_.end());
      std::sort(sparse.begin(), sparse.end());
      for (const auto& [id, slot] : sparse) {
        if (!take(id, slot)) return false;
      }
    }
    if (pending) return false;
    if (count != 0) {
      visit(std::span<const std::uint64_t>(ids.data(), count),
            std::span<const std::int32_t>(slots.data(), count));
    }
    return true;
  }

 private:
  std::int32_t state(std::uint64_t id) const noexcept {
    if (id < dense_.size()) return dense_[static_cast<std::size_t>(id)];
    if (sparse_.empty()) return kUnused;
    const auto found = sparse_.find(id);
    return found == sparse_.end() ? kUnused : found->second;
  }
  // Writes `value` for `id`, keeping the used and live counts.
  void write(std::uint64_t id, std::int32_t value);

  std::vector<std::int32_t> dense_;
  std::unordered_map<std::uint64_t, std::int32_t> sparse_;
  std::size_t used_ = 0;
  std::size_t live_ = 0;
};

// What a PLC recovery proved of the mesh it published (PlcRecovery), which
// TetMesh::build adopts rather than proves again; the resulting state is the
// one the arrays alone build.
struct RecoveredTopology {
  // Per cell and face slot: the cell across, -1 on the boundary (empty: the
  // adjacency is paired from the cells). Nonempty, the recovered segments
  // are also adopted as distinct cell edges (recovery published them as
  // intervals of its recovered edge chains).
  std::span<const std::int32_t> neighbors;
  // The mesh is its own source bank (no explicit source or witnesses): every
  // constrained face and segment is one of its own source rows, which covers
  // it alone, since recovery proved the faces pairwise legal (no other row
  // of a group overlaps one), so the coverage proofs are not repeated.
  bool self_covered = false;
  // The recovery published cells of a Delaunay triangulation of its points
  // (PlcRecovery::delaunay), so every unconstrained face is locally Delaunay
  // and, the points being the ones its exact predicates read, every cell is
  // positively oriented.
  bool delaunay = false;
  double preseeded_radius = 0.0;
};

class TetMesh {
 public:
  explicit TetMesh(BoundaryPolicy policy = BoundaryPolicy::kFixed);
  TetMesh(const TetMesh&) = delete;
  TetMesh& operator=(const TetMesh&) = delete;

  // `recovered`: the input is a PLC recovery's published mesh
  // (RecoveredTopology), adopted without repeating what recovery proved.
  int32_t build(std::span<const double> points,
                std::span<const std::int32_t> tetrahedra,
                std::span<const std::int32_t> regions,
                std::span<const std::int32_t> faces,
                std::span<const std::int32_t> face_sources,
                std::span<const std::int8_t> face_orientations,
                std::span<const std::int32_t> segments,
                std::span<const std::int32_t> segment_sources,
                std::span<const double> protection_radii,
                const SourceComplexView* source = nullptr,
                std::span<const std::int8_t> protected_vertices = {},
                const RecoveredTopology* recovered = nullptr);

  int32_t build(std::span<const double> points,
                std::span<const std::int32_t> tetrahedra,
                std::span<const std::int32_t> regions,
                std::span<const std::int32_t> faces,
                std::span<const std::int32_t> face_sources,
                std::span<const std::int32_t> segments,
                std::span<const std::int32_t> segment_sources) {
    return build(points, tetrahedra, regions, faces, face_sources, {}, segments,
                 segment_sources, {}, nullptr);
  }
  int32_t assign_initial_ids(std::span<const std::uint64_t> vertex_ids,
                             std::span<const std::uint64_t> cell_ids,
                             std::span<const std::uint64_t> facet_ids,
                             std::span<const std::uint64_t> segment_ids,
                             std::span<const double> initial_vertex_sizes = {});
  std::uint64_t generation() const noexcept { return generation_; }
  int32_t bind_implicit_source(std::shared_ptr<TetImplicitSource> source);
  int32_t prepare_implicit_source(std::shared_ptr<TetImplicitSource> source);
  const std::shared_ptr<TetImplicitSource>& implicit_source() const noexcept {
    return implicit_source_;
  }
  void clear_implicit_facet_refusal() const noexcept { implicit_facet_refusal_.reset(); }
  const std::optional<TetImplicitFacetRefusal>& implicit_facet_refusal() const noexcept {
    return implicit_facet_refusal_;
  }
  // Linear index-to-physical map; translation does not affect the metric.
  int32_t set_affine_metric(std::span<const double> linear);
  bool metric_active() const noexcept { return metric_active_; }
  const std::array<double, 9>& metric_matrix() const noexcept { return metric_; }
  double metric_source_deviation(const double* const* corners, SourceStratum stratum,
                                 const double* parameters, const double* position) const;
  double metric_declared_plane_deviation(const double* plane, const double* parameters,
                                         const double* position) const;
  bool source_carrier_metric(const double* const* corners, SourceStratum stratum,
                             const double* parameters, double* position,
                             double& deviation) const;
  int metric_orientation() const noexcept { return metric_orientation_; }
  int metric_insphere(const double* a, const double* b, const double* c,
                      const double* d, const double* e) const;
  std::array<double, 3> metric_transform(const std::array<double, 3>& v) const noexcept {
    return {metric_[0]*v[0] + metric_[1]*v[1] + metric_[2]*v[2],
            metric_[3]*v[0] + metric_[4]*v[1] + metric_[5]*v[2],
            metric_[6]*v[0] + metric_[7]*v[1] + metric_[8]*v[2]};
  }
  std::array<double, 3> metric_inverse_transform(const std::array<double, 3>& v) const noexcept {
    return {metric_inverse_[0]*v[0] + metric_inverse_[1]*v[1] + metric_inverse_[2]*v[2],
            metric_inverse_[3]*v[0] + metric_inverse_[4]*v[1] + metric_inverse_[5]*v[2],
            metric_inverse_[6]*v[0] + metric_inverse_[7]*v[1] + metric_inverse_[8]*v[2]};
  }
  double metric_dot(const std::array<double, 3>& a, const std::array<double, 3>& b) const noexcept {
    const auto x = metric_transform(a), y = metric_transform(b);
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }
  bool bind_source_identity(std::string_view source_id,
                            std::string_view source_revision,
                            std::string_view source_map_id,
                            std::string_view frame_id,
                            bool identity_bound);
  bool identity_bound() const noexcept { return identity_bound_; }
  const std::string& source_revision() const noexcept { return source_revision_; }
  const std::string& source_id() const noexcept { return source_id_; }
  const std::string& frame_id() const noexcept { return frame_id_; }
  const std::string& source_map_id() const noexcept { return source_map_id_; }
  BoundaryPolicy policy() const noexcept { return policy_; }
  const TetrahedralComplex& complex() const noexcept { return complex_; }
  // Reserve storage for `extra` more cells so the next edits that add cells
  // do not grow the cell arrays. No other effect.
  void reserve_cells(std::size_t extra) {
    complex_.reserve(extra, 0);
    cell_ids_.reserve(complex_.tets.size() + extra);
  }

  // Protect all current facets in the named source groups and their edges and
  // vertices. Returns false without mutation if a group is absent.
  bool protect_facet_sources(std::span<const std::int32_t> sources);
  TetEditStatus snapshot(TetMeshSnapshot& output) const;
  // The public snapshot's bank sizes (PHX_MC_TET_MESH_SNAPSHOT_SIZES values)
  // from the live counters; false when one is not representable.
  bool snapshot_sizes(std::int64_t* sizes) const;
  // Writes the public snapshot into the caller's banks, whose capacities hold
  // `sizes` (snapshot_sizes): vertices in slot order, cells, facets and
  // segments in ascending ID order with their corners as vertex rows, in one
  // pass over the live state. kInvalidGeometry when the state disagrees with
  // `sizes`.
  TetEditStatus export_snapshot(const phx_mc_tet_mesh_snapshot_request& request,
                                const std::int64_t* sizes) const;
  // Passing the full source ID bank binds a single whole-volume
  // retriangulation to the immutable original coordinates. Only that edit
  // may retain interpolation ancestry of a vertex it retires.
  TetEditStatus apply_edit_batch(const TetEditBatch& batch,
                                 std::vector<TetEditResult>& results,
                                 std::span<const std::uint64_t> original_vertex_ids = {});
  // Exact refusal probe of a single-operation flip, ring reconnection or collapse batch
  // against this unbound, nonperiodic state: true with `status` when
  // apply_edit_batch would refuse it with that status (mutating nothing but
  // the implicit-source refusal counters apply_edit_batch would advance);
  // false when the batch may apply, or the probe does not decide it.
  bool probe_cavity_edit(const TetEdit& operation, TetReplacementProbeCache& cache,
                         TetEditStatus& status) const;
  // The same probe for a single-operation relocation batch (no state tables:
  // a relocation is validated on its moved vertices and their stars).
  bool probe_relocation_edit(const TetEdit& operation, TetEditStatus& status) const;
  // Applies a single-operation relocation batch to this unbound, nonperiodic
  // state in place: true with the status apply_edit_batch would return for it
  // (kApplied after the moved vertices' records are written and the generation
  // advanced; a refusal mutating nothing but the implicit-source refusal
  // counters); false, mutating nothing, when the batch is outside the
  // relocation probe's domain and apply_edit_batch must decide it.
  bool apply_relocation_edit(const TetEdit& operation, TetEditStatus& status);

  // The retained circumradius size target of a vertex slot (0: none).
  double vertex_size(std::int32_t slot) const noexcept {
    return slot >= 0 && static_cast<std::size_t>(slot) < vertex_sizes_.size()
               ? vertex_sizes_[static_cast<std::size_t>(slot)]
               : 0.0;
  }
  // The retained size field by live vertex ID (vertices with a target only).
  std::unordered_map<std::uint64_t, double> vertex_sizes() const;
  const std::vector<TetUnmetRecord>& unmet_records() const noexcept {
    return unmet_records_;
  }
  void set_unmet_records(std::vector<TetUnmetRecord>& records) noexcept {
    unmet_records_.swap(records);
  }
  // Periodic quotient identity is independent of the lifted adjacency. The
  // supplied records are validated and copied atomically with the mesh state.
  TetEditStatus set_periodic_state(
      std::span<const TetPeriodicOrbit> orbits,
      std::span<const TetPeriodicQuotientCell> cells,
      std::span<const TetPeriodicEntityOrbitMap> entity_maps = {},
      bool advance_generation = true);
  bool bind_periodic_identity(const TetPeriodicIdentity& identity);
  const TetPeriodicIdentity& periodic_identity() const noexcept { return periodic_identity_; }

  // Number of vertex slots (live and dead).
  std::size_t vertex_slot_count() const noexcept { return vertex_ids_.size(); }
  // The hot accessors are defined here so the kernels inline them.
  bool live_vertex(std::int32_t slot) const noexcept {
    return slot >= 0 && static_cast<std::size_t>(slot) < vertex_live_.size() &&
           vertex_live_[static_cast<std::size_t>(slot)] != 0;
  }
  std::int32_t vertex_slot(std::uint64_t logical_id) const noexcept {
    return vertex_index_.slot(logical_id);
  }
  std::int32_t cell_slot(std::uint64_t logical_id) const noexcept {
    return cell_index_.slot(logical_id);
  }
  std::uint64_t vertex_id(std::int32_t slot) const noexcept {
    return slot >= 0 && static_cast<std::size_t>(slot) < vertex_ids_.size()
               ? vertex_ids_[static_cast<std::size_t>(slot)]
               : std::numeric_limits<std::uint64_t>::max();
  }
  std::uint64_t cell_id(std::int32_t slot) const noexcept {
    return slot >= 0 && static_cast<std::size_t>(slot) < cell_ids_.size()
               ? cell_ids_[static_cast<std::size_t>(slot)]
               : std::numeric_limits<std::uint64_t>::max();
  }
  const double* point(std::int32_t slot) const noexcept {
    return live_vertex(slot) ? points_.data() + 3 * static_cast<std::size_t>(slot) : nullptr;
  }
  // The carriers of every vertex slot, three per slot, unchecked: for loops
  // over the corners of live cells. Valid until the next vertex insertion.
  const double* carriers() const noexcept { return points_.data(); }
  const SourceWitness& witness(std::int32_t slot) const noexcept;
  std::int8_t vertex_dimension(std::int32_t slot) const noexcept;
  double protection_radius(std::int32_t slot) const noexcept;
  bool protected_vertex(std::int32_t slot) const noexcept;
  const Tetrahedron& tet(std::int32_t slot) const noexcept {
    static const Tetrahedron dead{{kDeadVertex, kDeadVertex, kDeadVertex, kDeadVertex},
                                  {-1, -1, -1, -1}, 0, 0};
    return complex_.live(slot) ? complex_.tets[static_cast<std::size_t>(slot)] : dead;
  }
  bool positive(const std::int32_t* vertices) const;
  // The exact position of a live vertex: the source point S of its witness
  // (authoritative; the carrier is its correctly rounded realization), else
  // its carrier. False for a dead slot or an unaddressable witness.
  bool exact_position(std::int32_t slot, Expansion* position) const;
  bool witness_position(const SourceWitness& witness, const double* carrier,
                        Expansion* position) const;
  // Pointers into the source point bank: the two corners of source segment
  // row `row` (corners[2] null) or the three of source face row `row`.
  bool source_corners(SourceStratum stratum, std::int32_t row, const double** corners) const;
  // The declared exact plane of source face row `row` (origin, first axis,
  // second axis: nine binary64 values), or null when its facet declares none.
  const double* face_plane(std::int32_t row) const noexcept;
  // The exact point origin + parameters[0] * first axis + parameters[1] *
  // second axis of the declared plane of source face row `row`.
  bool plane_point(std::int32_t row, const double* parameters, Expansion* point) const;
  // The carrier of that point (correctly rounded per coordinate) and its
  // deviation: zero when the carrier lies exactly on the plane, else an
  // outward bound of |carrier - point|.
  bool plane_carrier(std::int32_t row, const double* parameters, double* carrier,
                     double& deviation) const;
  // Admission of a boundary vertex record on the declared plane of source
  // face row `row`: the plane meets the closed rounding box of `carrier` (the
  // reals that round to it, so the carrier is a correct rounding of a point
  // of the plane), or the carrier lies within the plane's declared tolerance
  // of it. Both tests are exact.
  bool plane_admits(std::int32_t row, const double* carrier) const;
  // The record of `carrier` on the declared plane of row `row`: correctly
  // rounded parameters of its orthogonal projection onto the plane and an
  // outward bound of its distance to it. False when the row declares none.
  bool plane_record(std::int32_t row, const double* carrier, TetPlaneRecord& record) const;
  // One record per (live vertex, declared group) incidence of the constrained
  // facets, ordered by vertex ID then face row (the group's first row).
  void plane_records(std::vector<std::uint64_t>& vertex_ids,
                     std::vector<TetPlaneRecord>& records) const;
  // The declaration a boundary vertex refused in the last build, if any.
  const std::optional<TetPlaneRefusal>& plane_refusal() const noexcept { return plane_refusal_; }

  // Algorithmic entry points share the same validation and transaction path.
  TetEditStatus split_edge(std::uint64_t first, std::uint64_t second,
                           const TetEditVertex& inserted,
                           std::span<const std::uint64_t> removed_cells,
                           std::span<const TetEditCell> replacement_cells,
                           std::span<const TetEditFacet> replacement_facets,
                           std::span<const TetEditSegment> replacement_segments);
  TetEditStatus collapse_vertex(std::uint64_t remove, std::uint64_t keep,
                                std::span<const std::uint64_t> removed_cells,
                                std::span<const TetEditCell> replacement_cells,
                                std::span<const TetEditFacet> replacement_facets,
                                std::span<const TetEditSegment> replacement_segments);
  TetEditStatus reconnect(std::span<const std::uint64_t> removed_cells,
                          std::span<const TetEditCell> replacement_cells,
                          std::span<const TetEditFacet> replacement_facets = {},
                          std::span<const TetEditSegment> replacement_segments = {});
  TetEditStatus relocate(std::span<const TetEditVertex> moved_vertices,
                         std::span<const std::uint64_t> affected_cells);

  // Called by refinement: the cavity of the constrained boundary point
  // `vertex` (carrier and source witness) whose disk faces are `disk_sides`
  // (finite cell, face slot), cells on both sides of their facets, splitting
  // the segment `split_segment` (vertex slots; {-1, -1} for a facet point);
  // an embedded split segment has no disk faces and its finite edge star
  // `segment_star` (cell slots) seeds the cavity instead;
  // kFound or kRefused (see insert_local_boundary_point).
  TetCavityStatus boundary_insertion_cavity(
      const TetEditVertex& vertex, std::span<const std::pair<std::int32_t, int>> disk_sides,
      std::array<std::int32_t, 2> split_segment, std::span<const std::int32_t> segment_star,
      std::vector<std::int32_t>& cells, const TetSegmentIndex* segment_index) const;

 private:
  friend std::int32_t flip_to_constrained_delaunay(
      TetMesh&, detail::LocalIncidence&, std::span<const std::uint64_t>,
      const std::function<bool(std::uint64_t, std::uint64_t)>&, std::vector<std::uint64_t>&);
  friend std::int32_t refine_mesh(TetMesh&, double, std::span<const double>,
                                  std::int64_t&, std::int64_t&, std::uint32_t);
  friend std::int32_t improve_mesh(TetMesh&, const TetQualityRequest&,
                                   std::span<const double>, std::int64_t&,
                                   std::int64_t&, std::int64_t&, std::int64_t&);
  friend std::int32_t repair_boundary_quality(
      TetMesh&, const TetQualityRequest&, std::span<const double>,
      std::span<const std::uint64_t>, std::int64_t&, std::int64_t&,
      std::int64_t&, std::int64_t&, bool&);
  friend class detail::LocalIncidence;
  friend struct test::TetMeshInteriorInsertion;
  // Internal quality passes do not consume ancestry; public snapshot always
  // retains the complete ordered lineage.
  TetEditStatus snapshot_internal(TetMeshSnapshot& output,
                                  bool include_cell_ancestry) const;

  // Proposed carrier/witness records for read-only relocation validation.
  struct RelocationOverlay {
    std::span<const TetEditVertex> vertices;

    const TetEditVertex* find(std::uint64_t id) const noexcept {
      for (const TetEditVertex& vertex : vertices) {
        if (vertex.id == id) return &vertex;
      }
      return nullptr;
    }
  };
  // Called only after LocalIncidence verifies its maintained complete star;
  // public relocate still validates caller-provided cell IDs against topology.
  TetEditStatus relocate_with_complete_star(
      std::span<const TetEditVertex> moved_vertices,
      std::span<const std::int32_t> star_slots);
  TetEditStatus relocate_impl(
      std::span<const TetEditVertex> moved_vertices,
      std::span<const std::uint64_t> affected_cells,
      std::span<const std::int32_t> complete_star,
      bool star_verified);
  // Called only by LocalIncidence for an interior quality reconnection or
  // collapse; validation and commit touch only the cavity and its boundary.
  TetEditStatus replace_local_cavity(const TetEdit& operation,
                                    const TetSegmentIndex* segment_index);
  // A hull ghost cell a local boundary edit creates: {kGhostVertex, a, b, c}
  // over the new hull face (a, b, c) (the oriented face of the finite cell
  // below it), labeled with that face's source.
  struct LocalGhostCell {
    std::array<std::int32_t, 4> vertices{};
    std::int32_t source = kNoConstraint;
  };
  // The commit shared by the cavity-local edits: stages the removal of
  // `removed` (finite and hull ghost slots) and the creation of the
  // operation's replacement cells (corner slots `replacement_vertices`, four
  // per cell) then `ghosts`, clears the constraint labels of `cleared_sides`
  // (cell, face) for the proof, validates exact positivity of every finite
  // cell (moved vertices read from `overlay` when given) and commits, binding
  // the replacement cell IDs and ancestry and retiring
  // operation.removed_cell_ids. On refusal the complex, its labels and the
  // cell records are unchanged.
  TetEditStatus commit_local_cells(const TetEdit& operation,
                                   std::span<const std::int32_t> replacement_vertices,
                                   std::span<const std::int32_t> removed,
                                   std::span<const LocalGhostCell> ghosts,
                                   std::span<const std::pair<std::int32_t, int>> cleared_sides,
                                   const RelocationOverlay* overlay);
  // A split of the edge (a, b) whose removed cells are `cells` keeps its
  // constrained faces covered by their halves pairwise when (a, b) is no
  // constrained segment and lies in exactly two constrained faces of those
  // cells, both cavity boundary faces (their other side kept) of one
  // declared-plane group: that group, else kNoConstraint. `segment_index`
  // (segment vertex IDs -> segments_ row), when given, decides whether
  // (a, b) is a segment without a scan of the bank.
  std::int32_t declared_split_pair(std::int32_t a, std::int32_t b,
                                   std::span<const std::int32_t> cells,
                                   const TetSegmentIndex* segment_index = nullptr) const;
  // The halves (start, m, third) and (m, end, third) of a constrained face
  // of declared group `source` keep its projected orientation on the
  // group's plane, with m's exact position exactly on the plane. For both
  // faces of a declared_split_pair this proves their halves tile the two
  // faces exactly: the four fan triangles around m over the quadrilateral
  // (a, c, b, d) are positively oriented, so m lies in its kernel.
  bool declared_split_children(std::int32_t source, const std::int32_t* parent_face,
                               const std::array<std::int32_t, 3>& first_child,
                               const std::array<std::int32_t, 3>& second_child) const;
  // apply_edit_batch's coverage of a split constrained face `parent_face`
  // (source `source`) of the removed cells `cells` by its halves: exact
  // (exact_split_coverage), or by its declared pair.
  bool split_face_covered(const std::int32_t* parent_face, std::int32_t source,
                          const std::array<std::int32_t, 3>& first_child,
                          const std::array<std::int32_t, 3>& second_child,
                          std::span<const std::int32_t> cells,
                          const TetSegmentIndex* segment_index = nullptr) const;
  // apply_edit_batch's checks of the vertex `inserted` splitting the edge
  // (a, b), which is no constrained segment, of the removed cells `cells`:
  // kApplied, or the refusal.
  TetEditStatus unconstrained_split_point_status(std::int32_t a, std::int32_t b,
                                                 const std::array<std::uint64_t, 2>& edge,
                                                 const TetEditVertex& inserted,
                                                 std::span<const std::int32_t> cells,
                                                 const TetSegmentIndex* segment_index = nullptr) const;
  // Called by refinement and improvement: cones a star-shaped cavity to one
  // new interior vertex; created cell IDs are returned in cone order. The
  // cone buries exactly the cavity vertices `removed_vertices` (sorted
  // slots, each an unprotected interior vertex without a source witness),
  // which are removed; every other cavity vertex stays on its boundary.
  TetEditStatus insert_local_cone(const TetEditVertex& vertex,
                                  std::span<const std::int32_t> cavity,
                                  std::span<const std::int32_t> removed_vertices,
                                  std::vector<std::uint64_t>& created_cells,
                                  const TetSegmentIndex* segment_index);
  // Called by refinement: cones the cavity `cells` to a new interior vertex
  // at `point` (no source witness) with size target `size` (0: none). The
  // caller has checked that the cavity is one region with no constrained
  // face, constrained segment or vertex inside, and that every cone cell
  // over `boundary` (cavity cell and face slot whose neighbor stays, that
  // neighbor and its face slot) is positive on carriers; `vertices` are the
  // distinct vertices of the boundary faces. A cone over a source-witnessed
  // vertex is checked exactly here. New cells and the vertex take fresh IDs
  // and no lineage. Created cell slots are returned in `created`, one per
  // boundary face in order.
  TetEditStatus insert_interior_vertex(const double* point, double size,
                                       std::span<const std::int32_t> cells,
                                       std::span<const BoundaryFacet> boundary,
                                       std::span<const std::int32_t> vertices,
                                       std::vector<std::int32_t>& created);
  bool link_interior_cone(std::uint32_t local_count);
  // Called by refinement through LocalIncidence: inserts the source-witnessed
  // vertex `vertex` into a constrained facet, or into the constrained segment
  // of `segment_rows` (at most one row). `cavity` holds the finite and hull
  // ghost cells to replace around it and `facet_rows` the facets_ rows of the
  // constrained faces inside it (one disk per facet), which the fans of the
  // vertex over the disk boundaries replace: the fans overwrite those rows in
  // order and are appended beyond them; the segment row is overwritten by its
  // first half and the second appended. An embedded segment (on no
  // constrained face) is split with no facet rows. Created finite cell IDs
  // are returned.
  TetEditStatus insert_local_boundary_point(const TetEditVertex& vertex,
                                            std::span<const std::int32_t> cavity,
                                            std::span<const std::size_t> facet_rows,
                                            std::span<const std::size_t> segment_rows,
                                            std::vector<std::uint64_t>& created_cells,
                                            const TetSegmentIndex* segment_index);
  // Called by refinement and improvement: the star-shaped cavity of `point`
  // reached by walking from cell slot `start`.
  TetCavityStatus insertion_cavity(std::int32_t start, const double* point,
                                   bool keep_walk, TetInsertionCavity& cavity,
                                   const TetSegmentIndex* segment_index = nullptr) const;
  // Replaces the retained size field by `sizes` (one value per live vertex,
  // in slot order, the snapshot order; 0 leaves a vertex unconstrained)
  // unless `sizes` is empty, which keeps the retained field.
  int32_t install_vertex_sizes(std::span<const double> sizes);
  void set_vertex_size(std::int32_t slot, double size) {
    if (static_cast<std::size_t>(slot) >= vertex_sizes_.size()) {
      if (size == 0.0) return;
      vertex_sizes_.resize(static_cast<std::size_t>(slot) + 1, 0.0);
    }
    vertex_sizes_[static_cast<std::size_t>(slot)] = size;
  }
  // True if an edge of `removed_slots` absent from `kept_edges` (local edge
  // keys) is a constrained segment.
  bool removes_constrained_segment(std::span<const std::int32_t> removed_slots,
                                   const std::set<std::uint64_t>& kept_edges,
                                   const TetSegmentIndex* segment_index) const;
  TetEditStatus apply_one_edit(const TetEdit& operation, TetEditResult& result,
                               bool original_retessellation);
  TetEditStatus validate_and_commit(TetMesh& staged, const TetEdit& operation,
                                    TetEditResult& result, bool original_retessellation);
  void copy_state_to(TetMesh& destination) const;
  void swap_state(TetMesh& other) noexcept;
  int32_t build_state(std::span<const double> points,
                      std::span<const std::int32_t> tetrahedra,
                      std::span<const std::int32_t> regions,
                      std::span<const std::int32_t> faces,
                      std::span<const std::int32_t> face_sources,
                      std::span<const std::int8_t> face_orientations,
                      std::span<const std::int32_t> segments,
                      std::span<const std::int32_t> segment_sources,
                      std::span<const double> protection_radii,
                      const SourceComplexView* source,
                      std::span<const std::int8_t> protected_vertices,
                      const RecoveredTopology* recovered);
  // Whether every constrained face and segment is covered by its source
  // group (source_face_valid, source_segment_valid).
  bool constraints_covered() const;
  // A face group whose facet declares an exact plane: every row carries the
  // same plane, `rows` in bank order (the first is the group's plane row),
  // `boundary_edges` the source point pairs used by exactly one row (the
  // boundary of their union), `orientation` the common sign of the rows'
  // corners projected onto the plane (normal first axis x second axis).
  struct DeclaredGroup {
    std::vector<std::int32_t> rows;
    std::vector<std::array<std::int32_t, 2>> boundary_edges;
    std::int8_t orientation = 0;
  };
  // Installs the declared planes (empty: none) and derives declared_groups_;
  // INVALID_INPUT for a partially declared row, non-finite values, parallel
  // axes, a group mixing planes, or rows that do not form an oriented
  // manifold triangulation in projection.
  int32_t install_declared_planes(std::span<const double> planes,
                                  std::span<const double> tolerances);
  struct PreparedFacetGroup;
  // Subfacet validity on a declared group (see source_face_valid).
  bool declared_face_valid(const DeclaredGroup& group, const double* const* carriers,
                           const SourceWitness* const* witness_records,
                           std::int8_t orientation, const PreparedFacetGroup* prepared) const;
  struct PreparedFacetGroup {
    PreparedFacetGroup() = default;
    PreparedFacetGroup(const TetMesh& mesh, const DeclaredGroup& declared);
    // The indexes borrow this record's primitive and bounds banks.
    PreparedFacetGroup(const PreparedFacetGroup&) = delete;
    PreparedFacetGroup& operator=(const PreparedFacetGroup&) = delete;

    std::vector<std::array<std::int32_t, 4>> primitives;
    std::vector<std::int32_t> indices;
    std::vector<FacetCoverageBounds> bounds;
    FacetCoverageIndex coverage_index;
    std::vector<std::array<std::int32_t, 4>> boundary_primitives;
    std::vector<std::int32_t> boundary_indices;
    std::vector<FacetCoverageBounds> boundary_bounds;
    FacetCoverageIndex boundary_index;
    mutable std::vector<std::size_t> coverage_scratch;
    int declared_projection_axis = -1;
  };
  void prepare_declared_group(const DeclaredGroup& declared, PreparedFacetGroup& group) const;
  bool source_carrier_valid(const TetEditVertex& vertex) const;
  bool source_face_valid(const TetEditFacet& facet,
                         const RelocationOverlay* overlay = nullptr,
                         const PreparedFacetGroup* group = nullptr) const;
  TetEditStatus source_face_refusal_status() const noexcept {
    return implicit_source_ && implicit_source_->last_inconclusive
               ? TetEditStatus::kUncertainPredicate : TetEditStatus::kInvalidGeometry;
  }
  bool implicit_facets_embedded(
      std::span<const TetSnapshotFacet> facets, std::span<const std::int32_t> groups,
      const std::unordered_set<std::uint64_t>* changed_facets = nullptr,
      const std::unordered_set<std::uint64_t>* changed_vertices = nullptr,
      const RelocationOverlay* overlay = nullptr) const;
  bool source_segment_valid(const TetEditSegment& segment,
                            const RelocationOverlay* overlay = nullptr,
                            const std::vector<std::size_t>* group_rows = nullptr) const;
  template <bool WithRelocation>
  bool positive_impl(const std::int32_t* vertices,
                     const RelocationOverlay* overlay) const;
  bool positive_with_relocation(const std::int32_t* vertices,
                                const RelocationOverlay& overlay) const;
  // positive() on explicit carriers and witnesses (exact source positions).
  bool carriers_positive(const double* const* carriers,
                         const SourceWitness* const* witness_records) const;
  // positive() of `vertices` with corner `apex` replaced by the unstaged
  // vertex `point`.
  bool cone_positive(const std::int32_t* vertices, int apex, const TetEditVertex& point) const;
  bool validate_regions_and_constraints(
      const TetEdit& operation, std::span<const std::int32_t> removed_slots,
      std::span<const TetSnapshotFacet> candidate_facets) const;
  bool validate_periodic_closure(const TetEdit& operation) const;
  // Lifted vertex ID -> (orbit row, member row) in periodic_orbits_.
  using PeriodicMemberIndex =
      std::unordered_map<std::uint64_t, std::pair<std::size_t, std::size_t>>;
  PeriodicMemberIndex periodic_member_index() const;
  bool periodic_seam_facet(const TetSnapshotFacet& facet,
                           const PeriodicMemberIndex& members) const;
  bool audit_state() const;
  // Fills `cache` for this generation (see TetReplacementProbeCache).
  bool certify_replacement_probe(TetReplacementProbeCache& cache) const;

  BoundaryPolicy policy_;
  std::array<double, 9> metric_{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::array<double, 9> metric_inverse_{1, 0, 0, 0, 1, 0, 0, 0, 1};
  bool metric_active_ = false;
  int metric_orientation_ = 1;
  TetrahedralComplex complex_;
  CavityEdit edit_;
  std::vector<double> points_;
  std::vector<std::uint64_t> vertex_ids_;
  std::vector<std::uint8_t> vertex_live_;
  std::vector<std::int8_t> vertex_dimension_;
  std::vector<double> protection_radii_;
  std::vector<std::uint8_t> protected_vertices_;
  std::vector<SourceWitness> witnesses_;
  std::vector<std::uint8_t> vertex_has_edge_parameter_;
  std::vector<std::array<std::uint64_t, 2>> vertex_edge_endpoints_;
  std::vector<double> vertex_edge_parameters_;
  std::vector<std::uint64_t> cell_ids_;
  std::unordered_map<std::uint64_t, std::vector<TetVertexAncestry>> vertex_ancestry_;
  std::unordered_map<std::uint64_t, TetCellAncestryRecord> cell_ancestry_;
  std::vector<TetSnapshotFacet> facets_;
  std::vector<TetSnapshotSegment> segments_;
  std::vector<double> source_points_;
  std::vector<std::int32_t> source_faces_;
  std::vector<std::int32_t> source_face_groups_;
  std::vector<std::int64_t> source_face_ids_;
  std::vector<double> source_face_tolerances_;
  std::vector<std::int32_t> source_segments_;
  std::vector<std::int32_t> source_segment_groups_;
  std::vector<double> source_segment_tolerances_;
  std::vector<std::int64_t> source_segment_ids_;
  // Declared exact planes per source face row (nine values; NaN rows are
  // undeclared) and their declared tolerances, always one row per face row.
  std::vector<double> source_face_planes_;
  std::vector<double> source_face_plane_tolerances_;
  // Face group -> its declared plane data, derived from the immutable banks.
  std::unordered_map<std::int32_t, DeclaredGroup> declared_groups_;
  std::shared_ptr<TetImplicitSource> implicit_source_;
  std::shared_ptr<TetImplicitSource> pending_implicit_source_;
  std::optional<TetPlaneRefusal> plane_refusal_;
  mutable std::optional<TetImplicitFacetRefusal> implicit_facet_refusal_;
  std::vector<TetPeriodicOrbit> periodic_orbits_;
  std::vector<TetPeriodicQuotientCell> quotient_cells_;
  std::vector<TetPeriodicEntityOrbitMap> periodic_entity_maps_;
  std::vector<TetUnmetRecord> unmet_records_;
  // Hard circumradius size targets by vertex slot, installed by the `sizes` of
  // refinement/improvement and interpolated to the vertices they insert; 0
  // (or a slot beyond the bank) leaves a vertex unconstrained.
  std::vector<double> vertex_sizes_;
  TetIdIndex vertex_index_;
  TetIdIndex cell_index_;
  TetIdIndex facet_index_;
  TetIdIndex segment_index_;
  std::unordered_set<std::uint64_t> used_periodic_orbit_ids_;
  std::unordered_set<std::uint64_t> used_quotient_cell_ids_;
  std::array<std::unordered_set<std::uint64_t>, 4> used_periodic_lifted_ids_;
  std::array<std::unordered_set<std::uint64_t>, 4> used_periodic_quotient_ids_;
  std::uint64_t generation_ = 0;
  // The generation at which the cells were proven constrained Delaunay (a
  // Delaunay PLC recovery adopted, RecoveredTopology::delaunay); any edit
  // advances the generation past it. UINT64_MAX: never.
  std::uint64_t delaunay_generation_ = std::numeric_limits<std::uint64_t>::max();
  double initial_preseeded_radius_ = 0.0;
  std::uint64_t next_vertex_id_ = 0;
  std::uint64_t next_cell_id_ = 0;
  std::uint64_t next_facet_id_ = 0;
  std::uint64_t next_segment_id_ = 0;
  bool identity_bound_ = false;
  std::string source_revision_;
  std::string source_id_;
  std::string frame_id_;
  std::string source_map_id_;
  TetPeriodicIdentity periodic_identity_;
  // Scratch of insert_interior_vertex (not state: never copied or swapped).
  struct ConeFace {
    std::int32_t outside = -1;
    std::int32_t outside_face = -1;
    std::int32_t constraint = kNoConstraint;
    std::int32_t apex = -1;
    std::array<std::int32_t, 4> vertices{};
    // Indexed by the corner in kOrientedFacet[apex]. Initially dense
    // boundary-vertex indices; after preflight, the three neighboring cone rows.
    std::array<std::uint32_t, 3> locals{};
  };
  // An unmatched outer edge with its cone row and local facet corner.
  // For a cone too large for the dense owner table.
  struct ConeEdge {
    static constexpr std::uint64_t kEmpty = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t key = kEmpty;
    std::int32_t cell = -1;
    std::int32_t face = -1;
  };
  static constexpr std::uint32_t kDenseConeVertices = 512;
  std::vector<ConeFace> cone_faces_;
  std::vector<ConeEdge> cone_edges_;
  // Per vertex slot: cone_epoch_ << 32 | its local index in the current cone.
  std::vector<std::uint64_t> cone_vertex_tags_;
  // Directed boundary edge (from, to) of local indices, row-major over a
  // power-of-two side: cone_epoch_ << 32 | 4 cone row + facet corner.
  std::vector<std::uint64_t> cone_edge_owners_;
  std::uint32_t cone_epoch_ = 0;
};

}  // namespace phx::mc
