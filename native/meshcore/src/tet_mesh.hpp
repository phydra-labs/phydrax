//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Persistent source-aware tetrahedral state and generation-checked topology
// transactions. Local slots are reusable implementation indices; logical IDs
// are caller-visible and never inferred from coordinates.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
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
#include "plc3d.hpp"
#include "source_construction.hpp"
#include "zero_set_admission.hpp"

namespace phx::mc {
namespace detail {
class LocalIncidence;
}
struct TetQualityRequest;

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

class TetMesh {
 public:
  explicit TetMesh(BoundaryPolicy policy = BoundaryPolicy::kFixed);
  TetMesh(const TetMesh&) = delete;
  TetMesh& operator=(const TetMesh&) = delete;

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
                std::span<const std::int8_t> protected_vertices = {});

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

  // Protect all current facets in the named source groups and their edges and
  // vertices. Returns false without mutation if a group is absent.
  bool protect_facet_sources(std::span<const std::int32_t> sources);
  TetEditStatus snapshot(TetMeshSnapshot& output) const;
  TetEditStatus apply_edit_batch(const TetEditBatch& batch,
                                 std::vector<TetEditResult>& results);

  // Retained circumradius size targets by vertex ID (see vertex_sizes_).
  const std::unordered_map<std::uint64_t, double>& vertex_sizes() const noexcept {
    return vertex_sizes_;
  }
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

  bool live_vertex(std::int32_t slot) const noexcept;
  std::int32_t vertex_slot(std::uint64_t logical_id) const noexcept;
  std::int32_t cell_slot(std::uint64_t logical_id) const noexcept;
  std::uint64_t vertex_id(std::int32_t slot) const noexcept;
  std::uint64_t cell_id(std::int32_t slot) const noexcept;
  const double* point(std::int32_t slot) const noexcept;
  const SourceWitness& witness(std::int32_t slot) const noexcept;
  std::int8_t vertex_dimension(std::int32_t slot) const noexcept;
  double protection_radius(std::int32_t slot) const noexcept;
  bool protected_vertex(std::int32_t slot) const noexcept;
  const Tetrahedron& tet(std::int32_t slot) const noexcept;
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

 private:
  friend std::int32_t flip_to_constrained_delaunay(
      TetMesh&, detail::LocalIncidence&, std::span<const std::uint64_t>,
      const std::function<bool(std::uint64_t, std::uint64_t)>&, std::vector<std::uint64_t>&);
  friend std::int32_t refine_mesh(TetMesh&, double, std::span<const double>,
                                  std::int64_t&, std::int64_t&);
  friend std::int32_t improve_mesh(TetMesh&, const TetQualityRequest&,
                                   std::span<const double>, std::int64_t&,
                                   std::int64_t&, std::int64_t&, std::int64_t&);
  friend std::int32_t repair_boundary_quality(
      TetMesh&, const TetQualityRequest&, std::span<const double>,
      std::span<const std::uint64_t>, std::int64_t&, std::int64_t&,
      std::int64_t&, std::int64_t&, bool&);
  friend class detail::LocalIncidence;
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
  TetEditStatus replace_local_cavity(const TetEdit& operation);
  // Called by refinement and improvement: cones a star-shaped cavity to one
  // new interior vertex; created cell IDs are returned in cone order.
  TetEditStatus insert_local_cone(const TetEditVertex& vertex,
                                  std::span<const std::int32_t> cavity,
                                  std::vector<std::uint64_t>& created_cells);
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
                                            std::vector<std::uint64_t>& created_cells);
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
      std::vector<std::int32_t>& cells) const;
  // Called by refinement and improvement: the star-shaped cavity of `point`
  // reached by walking from cell slot `start`.
  TetCavityStatus insertion_cavity(std::int32_t start, const double* point,
                                   bool keep_walk, TetInsertionCavity& cavity) const;
  // Replaces the retained size field by `sizes` (one value per vertex of
  // `current`, in snapshot order; 0 leaves a vertex unconstrained) unless
  // `sizes` is empty, which keeps the retained field.
  int32_t install_vertex_sizes(const TetMeshSnapshot& current, std::span<const double> sizes);
  int32_t install_vertex_sizes(std::span<const std::uint64_t> vertex_ids,
                               std::span<const double> sizes);
  // True if an edge of `removed_slots` absent from `kept_edges` (local edge
  // keys) is a constrained segment.
  bool removes_constrained_segment(std::span<const std::int32_t> removed_slots,
                                   const std::set<std::uint64_t>& kept_edges) const;
  TetEditStatus apply_one_edit(const TetEdit& operation, TetEditResult& result);
  TetEditStatus validate_and_commit(TetMesh& staged, const TetEdit& operation,
                                    TetEditResult& result);
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
                      std::span<const std::int8_t> protected_vertices);
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
  // Subfacet validity on a declared group (see source_face_valid).
  bool declared_face_valid(const DeclaredGroup& group, const double* const* carriers,
                           const SourceWitness* const* witness_records,
                           std::int8_t orientation) const;
  bool source_carrier_valid(const TetEditVertex& vertex) const;
  bool source_face_valid(const TetEditFacet& facet,
                         const RelocationOverlay* overlay = nullptr) const;
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
                            const RelocationOverlay* overlay = nullptr) const;
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
  std::vector<TetPeriodicOrbit> periodic_orbits_;
  std::vector<TetPeriodicQuotientCell> quotient_cells_;
  std::vector<TetPeriodicEntityOrbitMap> periodic_entity_maps_;
  std::vector<TetUnmetRecord> unmet_records_;
  // Hard circumradius size targets by vertex ID, installed by the `sizes` of
  // refinement/improvement and interpolated to the vertices they insert; a
  // vertex without an entry is unconstrained.
  std::unordered_map<std::uint64_t, double> vertex_sizes_;
  std::unordered_map<std::uint64_t, std::int32_t> vertex_slots_;
  std::unordered_map<std::uint64_t, std::int32_t> cell_slots_;
  std::unordered_set<std::uint64_t> used_vertex_ids_;
  std::unordered_set<std::uint64_t> used_cell_ids_;
  std::unordered_set<std::uint64_t> used_facet_ids_;
  std::unordered_set<std::uint64_t> used_segment_ids_;
  std::unordered_set<std::uint64_t> used_periodic_orbit_ids_;
  std::unordered_set<std::uint64_t> used_quotient_cell_ids_;
  std::array<std::unordered_set<std::uint64_t>, 4> used_periodic_lifted_ids_;
  std::array<std::unordered_set<std::uint64_t>, 4> used_periodic_quotient_ids_;
  std::uint64_t generation_ = 0;
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
};

}  // namespace phx::mc
