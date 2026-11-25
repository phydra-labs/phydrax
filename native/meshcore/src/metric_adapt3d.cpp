//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "metric_adapt3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "cavity.hpp"
#include "deterministic_float.hpp"
#include "expansion.hpp"
#include "metric_carrier.hpp"
#include "metric_collapse.hpp"
#include "metric_relocation.hpp"
#include "metric_float.hpp"
#include "metric_goal.hpp"
#include "predicates.hpp"
#include "source_construction.hpp"

namespace phx::mc::metric {

bool combine_stencils(const Stencil& a, const Stencil& b, double t, Stencil& combined) {
  combined.clear();
  const double s = 1.0 - t;  // exact: t is dyadic in (0, 1)
  const auto scaled = [](double factor, double weight, double& product) {
    product = factor * weight;
    return std::fma(factor, weight, -product) == 0.0;
  };
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a.size() || j < b.size()) {
    StencilTerm term;
    if (j == b.size() || (i < a.size() && a[i].row < b[j].row)) {
      term.row = a[i].row;
      if (!scaled(s, a[i].weight, term.weight)) return false;
      ++i;
    } else if (i == a.size() || b[j].row < a[i].row) {
      term.row = b[j].row;
      if (!scaled(t, b[j].weight, term.weight)) return false;
      ++j;
    } else {
      double first = 0.0;
      double second = 0.0;
      if (!scaled(s, a[i].weight, first) || !scaled(t, b[j].weight, second)) return false;
      double sum = 0.0;
      double tail = 0.0;
      two_sum(first, second, sum, tail);
      if (tail != 0.0) return false;
      term.row = a[i].row;
      term.weight = sum;
      ++i;
      ++j;
    }
    if (term.weight != 0.0) combined.push_back(term);
  }
  return true;
}

namespace {

constexpr double kDeterminantFloor = 0x1p-22;
constexpr double kImprovement = 1e-6;
constexpr int kSplitBits = 5;
constexpr std::int32_t kNoBlock = -1;
constexpr std::int64_t kScalarShapeSweeps = 4;
constexpr std::int64_t kScalarTopologySweeps = 3;
constexpr std::int64_t kUntilShapeStall = std::numeric_limits<std::int64_t>::max();
constexpr double kShapeCageTarget = 0.61;
constexpr double kShapeCageFloor = 0.60;
constexpr std::size_t kShapeCageVertices = 40;
constexpr std::size_t kShapeCageCells = 256;
constexpr std::size_t kBandCageEdges = 16;
constexpr std::size_t kNoCageAxis = 3;
// Rings of in-band vertices around out-of-band spokes searched by stages 3
// and 4.
constexpr int kStageRegionRings = 2;

bool scalar_metric(const double* tensor) {
  return tensor[0] > 0.0 && std::isfinite(tensor[0]) &&
         tensor[0] == tensor[4] && tensor[0] == tensor[8] &&
         tensor[1] == 0.0 && tensor[2] == 0.0 && tensor[3] == 0.0 &&
         tensor[5] == 0.0 && tensor[6] == 0.0 && tensor[7] == 0.0;
}

double cage_distance_squared(const double* a, const double* b) {
  const double x = a[0] - b[0];
  const double y = a[1] - b[1];
  const double z = a[2] - b[2];
  return (x * x + y * y) + z * z;
}

// Euclidean mean ratio of an orientation-checked tetrahedron.
double geometric_shape(const double* corners, double determinant) {
  double squares = 0.0;
  for (std::size_t a = 0; a < 4; ++a) {
    for (std::size_t b = a + 1; b < 4; ++b) {
      squares += cage_distance_squared(corners + 3 * a, corners + 3 * b);
    }
  }
  const double volume_scale = deterministic::cube_root(0.5 * determinant);
  return 12.0 * volume_scale * volume_scale / squares;
}

constexpr std::size_t kRelocationCacheSlots = 4096;
constexpr std::size_t kRelocationCacheStarCells = 64;

// The exact stencil of a carrier: its dyadic weights as binary64 values.
Stencil carrier_stencil(const Carrier& carrier) {
  Stencil stencil;
  stencil.reserve(static_cast<std::size_t>(carrier.size));
  for (int term = 0; term < carrier.size; ++term) {
    const auto index = static_cast<std::size_t>(term);
    stencil.push_back({static_cast<std::int32_t>(carrier.rows[index]),
                       std::ldexp(static_cast<double>(carrier.weights[index]), -carrier.bits)});
  }
  return stencil;
}

constexpr std::uint64_t kExhaustedId = std::numeric_limits<std::uint64_t>::max();

bool fits_ids(std::uint64_t next, std::size_t count) {
  // Keep MAX as the exhausted cursor, never as a freshly assigned ID.
  return count <= kExhaustedId - next;
}

using detail::LocalIncidence;

// Max-first quantized band violations of a set of lengths.
struct Violations {
  std::vector<double> long_side;
  std::vector<double> short_side;
  bool measured = true;

  void clear() {
    long_side.clear();
    short_side.clear();
    measured = true;
  }
  void add(double length) {
    double above = 0.0;
    double below = 0.0;
    band_quanta(length, above, below);
    if (!std::isfinite(above) || !std::isfinite(below)) {
      measured = false;
      return;
    }
    if (above != 0.0) long_side.push_back(above);
    if (below != 0.0) short_side.push_back(below);
  }
  void finish() {
    std::sort(long_side.begin(), long_side.end(), std::greater<>());
    std::sort(short_side.begin(), short_side.end(), std::greater<>());
  }
};

// Lexicographic comparison of max-first sequences; a proper prefix is smaller.
int compare_descending(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t count = std::min(a.size(), b.size());
  for (std::size_t index = 0; index < count; ++index) {
    if (a[index] != b[index]) return a[index] < b[index] ? -1 : 1;
  }
  if (a.size() == b.size()) return 0;
  return a.size() < b.size() ? -1 : 1;
}

int compare_violations(const Violations& a, const Violations& b) {
  const int long_order = compare_descending(a.long_side, b.long_side);
  return long_order != 0 ? long_order : compare_descending(a.short_side, b.short_side);
}

bool objective_decreases(const Violations& removed, const Violations& added, bool gain) {
  const int order = compare_violations(added, removed);
  return order != 0 ? order < 0 : gain;
}

// A committed cavity increases the sum of these scores at a fixed
// reference, even when the cavity changes its cell count.
double shape_score(double total, std::size_t cells, double reference) {
  return total - static_cast<double>(cells) * reference;
}

std::uint64_t edge_key(std::int32_t a, std::int32_t b) {
  const auto low = static_cast<std::uint32_t>(std::min(a, b));
  const auto high = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(low) << 32) | high;
}

// An edge by its two vertex IDs, ascending (exact key; EdgeIdHash hashes it).
using IdEdge = std::array<std::uint64_t, 2>;
IdEdge id_edge(std::uint64_t a, std::uint64_t b) { return {std::min(a, b), std::max(a, b)}; }
using MeasuredEdge = std::pair<double, IdEdge>;

using IdFace = std::array<std::uint64_t, 3>;

// A first-order pull can vanish on a seam or be blocked by a nearly active
// spoke. A tangent to the short spoke's metric ellipsoid lengthens it to
// second order. The fixed metric only chooses goals; carrier search and the
// edit transaction still decide admissibility with the actual metric.
std::int64_t boundary_tangent_pulls(std::span<const double> points,
                                   std::span<const double> lengths,
                                   std::span<const std::int8_t> measured,
                                   const double* tensor, int rank, const double* basis,
                                   double* pulls) {
  const double zero[3]{};
  std::int64_t count = 0;
  for (std::size_t spoke = 0; spoke < lengths.size(); ++spoke) {
    const double short_length = lengths[spoke];
    if (measured[spoke] == 0 || short_length >= kLowerLength) continue;
    const double deficit = (kLowerLength - short_length) * (kLowerLength + short_length);
    const auto emit = [&](const double* direction) {
      const double norm = length(zero, direction, tensor, tensor);
      if (!(norm > 0.0 && std::isfinite(norm))) return;
      const double step = 2.0 * std::sqrt(deficit) / norm;
      for (double sign : {1.0, -1.0}) {
        for (int axis = 0; axis < 3; ++axis) {
          pulls[3 * count + axis] = (sign * step) * direction[axis];
        }
        ++count;
      }
    };
    if (rank == 1) {
      emit(basis);
    } else if (rank == 2) {
      const double* neighbor = points.data() + 3 * (spoke + 1);
      const double d[3]{neighbor[0] - points[0], neighbor[1] - points[1],
                        neighbor[2] - points[2]};
      double normal[2]{};
      for (int axis = 0; axis < 3; ++axis) {
        const double* row = tensor + 3 * axis;
        const double component = (row[0] * d[0] + row[1] * d[1]) + row[2] * d[2];
        normal[0] += component * basis[axis];
        normal[1] += component * basis[3 + axis];
      }
      const double scale = std::max(std::abs(normal[0]), std::abs(normal[1]));
      if (scale == 0.0) {
        emit(basis);
        emit(basis + 3);
      } else if (std::isfinite(scale)) {
        const double u = normal[0] / scale;
        const double v = normal[1] / scale;
        const double tangent[3]{v * basis[0] - u * basis[3],
                                v * basis[1] - u * basis[4],
                                v * basis[2] - u * basis[5]};
        emit(tangent);
      }
    }
  }
  return count;
}

Stencil carrier_stencil(const Carrier& carrier);

struct CellLineage {
  std::vector<std::uint64_t> ancestry;  // ascending source cell IDs
  AdaptLineage kind = AdaptLineage::kPreserved;
  bool relocated = false;
};

class Adapter {
 public:
  Adapter(TetMesh& mesh, const AdaptInput& input) : mesh_(mesh), input_(input) {}

  TetEditStatus run(AdaptOutput& output);

 private:
  // State setup and per-slot storage.
  void bind();
  void ensure_vertex(std::int32_t slot);
  // The phases that look for work, each at the vertices changed since it last
  // looked (every vertex of a created cell is changed for all of them).
  enum Phase : std::size_t {
    kSplitPhase = 0,
    kCollapsePhase = 1,
    kSwapPhase = 2,
    kRelocatePhase = 3,
  };
  void touch(std::int32_t vertex);
  void queue(Phase phase, std::int32_t vertex);
  // Queues a relocation stage step on an unchanged mesh. See stage_region_.
  void queue_stage_search();
  bool in_stage_region(std::int32_t vertex) const noexcept {
    const auto at = static_cast<std::size_t>(vertex);
    return vertex >= 0 && at < stage_region_.size() && stage_region_[at] != 0;
  }
  void dirty_edges(Phase phase, const std::function<bool(double)>& wanted,
                   std::vector<MeasuredEdge>& edges);
  void ensure_cell(std::int32_t slot);
  const double* metric(std::int32_t slot) const { return metrics_.data() + 9 * slot; }
  bool vertex_metric(const Stencil& stencil, double* tensor) const;
  double edge_length(std::int32_t a, std::int32_t b) const {
    // The graded log-mean is mathematically symmetric, not bit-symmetric.
    auto first_id = mesh_.vertex_id(a);
    auto second_id = mesh_.vertex_id(b);
    if (second_id < first_id) {
      std::swap(a, b);
      std::swap(first_id, second_id);
    }
    if (edge_measurements_.empty()) {
      return length(mesh_.point(a), mesh_.point(b), metric(a), metric(b));
    }
    const auto first = static_cast<std::uint32_t>(a);
    const auto second = static_cast<std::uint32_t>(b);
    const auto key = (static_cast<std::uint64_t>(first) << 32) | second;
    std::uint64_t hash = key ^ (key >> 33);
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    auto& entry = edge_measurements_[hash & (edge_measurements_.size() - 1)];
    const auto first_revision = carrier_revisions_[first];
    const auto second_revision = carrier_revisions_[second];
    // Persistent identities distinguish reused slots; only a carrier edit
    // changes the point or metric of a live endpoint.
    if (entry.first_id != first_id || entry.second_id != second_id ||
        entry.first_revision != first_revision || entry.second_revision != second_revision) {
      entry = {first_id, second_id, first_revision, second_revision,
               length(mesh_.point(a), mesh_.point(b), metric(a), metric(b))};
    }
    return entry.value;
  }
  IdEdge edge_ids(std::int32_t a, std::int32_t b) const {
    return id_edge(mesh_.vertex_id(a), mesh_.vertex_id(b));
  }
  bool protected_edge(std::int32_t a, std::int32_t b) const {
    return protected_edges_.contains(edge_ids(a, b));
  }

  // Measurement.
  bool complete(double lower) const;
  double shape_lower() const { return std::max(input_.lower, kLowerLength); }

  // Splits.
  std::int64_t split_sweep();
  // A two-facet hull shell can commit locally. Only a declared coplanar
  // pair can use coverage of the whole pair instead of exact edge coverage.
  enum class EdgeSupport { kFree, kBatch, kHull, kDeclaredHull };
  enum class SplitPointRule { kStencil, kExactEdge, kRoundedEdge };
  std::optional<EdgeSupport> edge_support(std::int32_t a, std::int32_t b,
                                          std::span<const std::int32_t> shell);
  bool split_parameter(std::int32_t a, std::int32_t b, SplitPointRule rule, Stencil& stencil,
                       double* point, double& parameter) const;
  bool sync_split_facets(std::int32_t slot, std::int32_t inserted);
  TetEditStatus split(std::int32_t a, std::int32_t b, bool& applied);
  // In a scalar field postpone inserting a midpoint with a short spoke.
  // Reopen unrestricted splitting at the first plateau, before exhaustion.
  bool separated_splits_ = false;
  // Constrained segment and non-hull splits use disjoint batches.
  struct PendingSplit {
    std::int32_t a = -1;
    std::int32_t b = -1;
    double parameter = 0.0;  // from a toward b
    std::array<double, 3> point{};
    std::array<double, 9> tensor{};
    Stencil stencil;
    std::vector<std::int32_t> shell;
  };
  TetEditStatus commit_pending(std::int64_t& applied);
  TetEditStatus apply_split_batch(std::span<const PendingSplit> splits, std::int64_t& applied);

  // Collapses.
  std::int64_t collapse_sweep();
  struct CollapsePlan {
    std::int32_t removed = -1;
    std::int32_t kept = -1;
    Violations added;
    bool boundary = false;  // the removed vertex lies on constrained facets
    bool moved = false;     // the survivor moves to `carrier` with tensor `metric`
    Carrier carrier;
    SourceWitness witness;
    std::array<double, 9> metric{};
  };
  bool may_collapse_in_place(std::int32_t removed, std::int32_t kept,
                             const LocalIncidence::Star& removed_star) const;
  bool plan_collapse(std::int32_t removed, std::int32_t kept, CollapsePlan& plan);
  // Commits a collapse star-locally where the cavity route applies: an
  // interior vertex into a fixed or moving free interior survivor
  // (LocalIncidence::replace_cavity) and a boundary vertex through
  // LocalIncidence::collapse_boundary. Other moving survivors and boundary
  // refusals of the local proof use a one-operation TetMesh::apply_edit_batch.
  TetEditStatus collapse(const CollapsePlan& plan, bool& applied);
  // The facet groups (labels of the constrained faces) through a vertex.
  std::vector<std::int32_t> facet_groups(std::int32_t vertex,
                                         std::span<const std::int32_t> star) const;
  // The vertex lineage of a committed collapse.
  void record_collapse(std::int32_t removed, std::int32_t kept);
  // The vertex's exact stencil and transported tensor after relocation.
  void assign_carrier(std::int32_t vertex, Stencil stencil, const double* tensor);
  std::optional<TetEditStatus> collapse_boundary(
      const CollapsePlan& plan, const TetEdit& edit,
      const LocalIncidence::Star& removed_star, const LocalIncidence::Star& kept_star);

  // A labeled face with no record, or a failed boundary index, is stale.
  bool facets_through(std::int32_t vertex, std::span<const std::int32_t> star,
                      std::vector<TetSnapshotFacet>& facets);
  bool segments_through(std::int32_t vertex, std::span<const std::int32_t> star,
                        std::vector<TetSnapshotSegment>& segments);

  // Swaps: 2-3 face flips and ring reconnections (3-2 included) of interior
  // cavities, committed star-locally.
  std::int64_t swap_sweep();
  std::int64_t shape_sweeps(std::int64_t maximum_sweeps = kUntilShapeStall);
  std::int64_t commit_swaps(std::vector<std::pair<std::uint64_t, int>>& faces,
                            std::vector<IdEdge>& edges, std::optional<double> shape_reference,
                            bool preserve_band = false);
  double quality(const std::array<std::int32_t, 4>& cell) const;
  // The quality of a live cell, kept until the cell or a corner changes.
  double cell_quality(std::int32_t slot);
  // Sets a vertex tensor and its cached log-determinant.
  void set_metric(std::int32_t vertex, const double* tensor);
  // Floor-oriented (float determinant above the floor) and its quality.
  bool oriented_quality(const std::array<std::int32_t, 4>& cell, double& value) const;
  TetEditStatus replace(TetEditKind kind, const std::vector<std::int32_t>& cavity,
                        const std::vector<std::array<std::int32_t, 4>>& cells, bool& applied);
  TetEditStatus try_ring(std::int32_t s, std::int32_t t, bool& applied,
                         std::optional<double> shape_reference, bool preserve_band);
  TetEditStatus try_flip23(std::int32_t slot, int face, bool& applied,
                            std::optional<double> shape_reference, bool preserve_band);
  // Swaps the boundary edge (s t), on the facets (s t c) and (s t d) of one
  // group, for (c d): `shell` becomes `cells` and the facets (s c d) and
  // (t c d), through LocalIncidence::swap_boundary_edge.
  TetEditStatus swap_boundary(const std::vector<std::int32_t>& shell,
                              const std::vector<std::array<std::int32_t, 4>>& cells,
                              std::int32_t s, std::int32_t t, std::int32_t c, std::int32_t d,
                              bool& applied);

  // Relocations of interior vertices to exact source carriers (in place).
  std::int64_t relocate_sweep(bool closed_band_shape = false);
  std::int64_t smooth_shape(std::int64_t maximum_sweeps);
  std::int64_t lift_shape_tail();
  std::int64_t lift_shape_cages();
  std::int64_t repair_band_cages();
  struct CageMover {
    std::int32_t vertex;
    CarrierChartView chart;
    unsigned axes;
    std::size_t row = 0;
  };
  struct ShapeCage {
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> slots;
    std::vector<CageMover> movers;
    std::vector<ShapeCagePoint> points;
    std::vector<std::array<std::size_t, 4>> cells;
    std::vector<ShapeCageEdge> edges;
    std::vector<std::array<double, 9>> frozen_tensors;
    std::size_t variables = 0;
    double minimum_distance = std::numeric_limits<double>::infinity();
    double minimum_shape = std::numeric_limits<double>::infinity();
    double sum_shape = 0.0;
  };
  double geometric_cell_shape(std::int32_t slot) const;
  std::optional<CageMover> cage_mover(std::int32_t vertex,
                                    const LocalIncidence::Star& star, bool band);
  bool prepare_shape_cage(std::int32_t seed, ShapeCage& cage, bool whole,
                          std::span<const std::int32_t> band_seeds, std::size_t fixed_axis);
  TetEditStatus commit_shape_cage(const ShapeCage& cage, std::int32_t seed,
                                 double seed_shape, std::span<const double> displacement,
                                 std::span<double> transported_scales, bool improve_mean,
                                 bool& applied);
  TetEditStatus relocate_shape_cage(std::int32_t seed, double seed_shape, bool& applied,
                                   bool improve_mean = false,
                                   std::span<const std::int32_t> band_seeds = {},
                                   std::size_t fixed_axis = kNoCageAxis);
  enum class ShapeGoal { kBand, kMean, kMinimum };
  TetEditStatus relocate(std::int32_t vertex, bool& applied, ShapeGoal goal);
  bool clearance_pull(std::int32_t vertex, std::int32_t cell, double* pull) const;
  bool wall_ring_pull(std::int32_t vertex, const LocalIncidence::Star& star, double* pull) const;
  struct RelocationAttempt {
    std::vector<std::uint64_t> cells;
    std::span<const CarrierSimplex> chart;
    std::uint64_t vertex_id = 0;
    std::uint64_t carrier_revision = 0;
    unsigned exhausted_modes = 0;
  };
  RelocationAttempt& bind_relocation_attempt(std::int32_t vertex,
                                             const LocalIncidence::Star& star,
                                             CarrierChartView chart);
  // Moves both interior ends of an out-of-band edge at once, symmetrically
  // along it toward unit length (halved down to 2^-40 of that step), each to
  // a source carrier near its goal, when the joint move decreases the
  // objective over both stars where single moves cannot; committed through
  // TetMesh::apply_relocation_edit.
  TetEditStatus relocate_pair(std::int32_t u, std::int32_t v, bool& applied);
  // The joint moves of the out-of-band edges at `vertices`.
  std::int64_t pair_sweep(std::span<const std::int32_t> vertices);
  void bind_rounded_sources(const TetMeshSnapshot& snapshot);
  enum class CarrierUse { kCollapse, kRelocation, kShapeCage, kBandCage };
  std::optional<CarrierChartView> carrier_chart(std::int32_t vertex,
                                                const LocalIncidence::Star& star,
                                                CarrierUse use = CarrierUse::kCollapse);
  bool source_seam_chart(const SourceWitness& witness, CarrierChartView chart) const;
  std::optional<SourceWitness> relocation_witness(std::int32_t vertex, const Carrier& carrier,
                                                  CarrierChartView chart = {}) const;
  std::optional<SourceWitness> seam_witness(const SourceWitness& old, const Carrier& carrier,
                                           CarrierChartView chart) const;

  // Lineage.
  void record_created_cell(std::int32_t slot, std::int32_t parent_block,
                           const CellLineage& parent, AdaptLineage kind);
  void retire_cell(std::int32_t slot);
  void count_refusal(TetEditStatus status);
  bool fatal(TetEditStatus status) const {
    return status == TetEditStatus::kStaleGeneration ||
           status == TetEditStatus::kAllocationFailure ||
           status == TetEditStatus::kIndexOverflow;
  }
  void write(AdaptOutput& output) const;

  TetMesh& mesh_;
  const AdaptInput& input_;
  LocalIncidence incidence_;
  AdaptCounters counters_;

  std::vector<Stencil> stencils_;          // per vertex slot
  std::vector<double> metrics_;            // nine per vertex slot
  std::vector<double> log_dets_;           // log det of each vertex tensor
  std::vector<std::uint64_t> vertex_targets_;
  std::vector<std::int32_t> cell_blocks_;  // per cell slot
  std::vector<std::uint64_t> cell_targets_;
  std::vector<CellLineage> cell_lineage_;
  std::vector<double> cell_quality_;       // NaN: not yet evaluated
  std::unordered_set<IdEdge, EdgeIdHash> protected_edges_;
  std::unordered_set<std::uint64_t> source_targets_;
  // Collapsed source vertices: target ID -> the target it collapsed into.
  std::unordered_map<std::uint64_t, std::uint64_t> collapsed_into_;
  std::unordered_set<std::uint64_t> relocated_sources_;
  // Per live target vertex: the collapsed source vertices that continue into it.
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> absorbed_;
  std::optional<CarrierSource> binary64_carriers_;
  const CarrierSource* carriers_ = nullptr;
  // Source triangles (rows into the source points) of each facet group, and
  // whether the mesh's source points are the adaptation's (boundary vertices
  // move on them only then).
  std::unordered_map<std::int32_t, CarrierChart> group_triangles_;
  std::unordered_map<std::uint64_t, CarrierChart> group_seams_;
  std::vector<std::int32_t> witness_face_groups_;
  std::vector<std::int32_t> witness_faces_;
  std::unordered_set<std::int32_t> declared_groups_;
  std::unordered_map<std::int32_t, std::size_t> exact_plane_axes_;
  // Source-only primitives use the source segment row, not a live edge ID.
  std::vector<std::array<std::int32_t, 2>> source_only_segments_;
  std::unordered_map<std::uint64_t, std::int32_t> source_only_segment_rows_;
  bool rounded_sources_ = false;
  bool source_banks_match_ = false;
  // Collapses may move their survivor: on once boundary relocation stalls.
  bool survivor_moves_ = false;
  // Boundary vertex relocations start once the interior stalls.
  bool boundary_moves_ = false;
  // Relocation stage: 0 out-of-band vertices, 1 deep, 2 deep and refined
  // (both band improvements only), 3 also in-band vertices near out-of-band
  // edges, 4 also pairs of ends of out-of-band edges, 5 terminal boundary
  // tangents; back to 0 after topology.
  int relocation_stage_ = 0;
  // A stage step leaves the mesh, the gates, and the split, collapse and
  // swap searches unchanged, so it queues only relocation. Stages 1, 2 and 5
  // can move only a vertex with an out-of-band spoke; stages 3 and 4 also
  // move in-band vertices, but only those within kStageRegionRings of such a
  // vertex (this mask, set at the step).
  std::vector<std::uint8_t> stage_region_;
  bool bounded_tensor_search_ = false;
  // Direct-mapped, bounded, and local to this adaptation call.
  std::vector<RelocationAttempt> relocation_attempts_;
  // Failed carrier searches depend on both endpoint states and the shell's
  // coverage rule. A move can make a previously refused edge splittable.
  struct SplitRefusal {
    std::uint64_t first_revision;
    std::uint64_t second_revision;
    EdgeSupport support;
    bool operator==(const SplitRefusal&) const noexcept = default;
  };
  std::vector<std::uint64_t> carrier_revisions_;
  struct EdgeMeasurement {
    std::uint64_t first_id = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t second_id = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t first_revision = 0;
    std::uint64_t second_revision = 0;
    double value = 0.0;
  };
  mutable std::vector<EdgeMeasurement> edge_measurements_;
  std::unordered_map<std::uint64_t, SplitRefusal> unrepresentable_;
  std::uint64_t next_vertex_id_ = 0;
  std::uint64_t next_cell_id_ = 0;
  std::uint64_t next_target_vertex_ = 0;
  std::uint64_t next_target_cell_ = 0;
  std::uint64_t next_facet_id_ = 0;
  std::uint64_t next_segment_id_ = 0;
  std::vector<PendingSplit> pending_;
  std::vector<char> occupied_;  // per cell slot: in the pending split batch
  std::array<std::vector<std::int32_t>, 4> dirty_;
  std::array<std::vector<char>, 4> dirty_flags_;
  TetEditStatus fatal_ = TetEditStatus::kApplied;
};

void Adapter::ensure_vertex(std::int32_t slot) {
  const auto size = static_cast<std::size_t>(slot) + 1;
  if (stencils_.size() < size) {
    stencils_.resize(size);
    metrics_.resize(9 * size, 0.0);
    log_dets_.resize(size, 0.0);
    carrier_revisions_.resize(size, 0);
    vertex_targets_.resize(size, std::numeric_limits<std::uint64_t>::max());
    for (auto& flags : dirty_flags_) flags.resize(size, 0);
  }
}

void Adapter::touch(std::int32_t vertex) {
  for (std::size_t phase = 0; phase < dirty_.size(); ++phase) queue(static_cast<Phase>(phase), vertex);
}

void Adapter::queue(Phase phase, std::int32_t vertex) {
  if (vertex < 0) return;
  ensure_vertex(vertex);
  const auto at = static_cast<std::size_t>(vertex);
  if (dirty_flags_[phase][at] != 0) return;
  dirty_flags_[phase][at] = 1;
  dirty_[phase].push_back(vertex);
}

void Adapter::queue_stage_search() {
  // An edge outside the relocation band or the completion band seeds the
  // step; its endpoints are the only vertices with an out-of-band spoke.
  const double lower = std::max(input_.lower, kLowerLength);
  stage_region_.assign(mesh_.vertex_slot_count(), 0);
  std::vector<std::int32_t> frontier;
  const auto mark = [&](std::int32_t vertex) {
    const auto at = static_cast<std::size_t>(vertex);
    if (!mesh_.live_vertex(vertex) || stage_region_[at] != 0) return;
    stage_region_[at] = 1;
    frontier.push_back(vertex);
  };
  const auto& tets = mesh_.complex().tets;
  for (std::size_t slot = 0; slot < tets.size(); ++slot) {
    const Tetrahedron& cell = tets[slot];
    if (cell.v[0] == kDeadVertex || is_ghost(cell)) continue;
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const std::int32_t a = cell.v[first];
        const std::int32_t b = cell.v[second];
        const double value = edge_length(a, b);
        if (!(value >= lower && value <= kUpperLength) && !protected_edge(a, b)) {
          mark(a);
          mark(b);
        }
      }
    }
  }
  if (relocation_stage_ == 3 || relocation_stage_ == 4) {
    std::vector<std::int32_t> star;
    for (int ring = 0; ring < kStageRegionRings; ++ring) {
      const std::vector<std::int32_t> seeds = std::move(frontier);
      frontier.clear();
      for (std::int32_t vertex : seeds) {
        incidence_.collect_star(vertex, star);
        for (std::int32_t slot : star) {
          for (std::int32_t other : mesh_.tet(slot).v) mark(other);
        }
      }
    }
  }
  for (std::size_t slot = 0; slot < stage_region_.size(); ++slot) {
    if (stage_region_[slot] != 0) queue(kRelocatePhase, static_cast<std::int32_t>(slot));
  }
}

void Adapter::dirty_edges(Phase phase, const std::function<bool(double)>& wanted,
                          std::vector<MeasuredEdge>& edges) {
  // The wanted measured edges at the vertices changed since this phase last
  // looked, each once; the phase's dirty list is consumed.
  edges.clear();
  const auto index = static_cast<std::size_t>(phase);
  std::vector<std::int32_t> star;
  for (std::int32_t vertex : dirty_[index]) {
    dirty_flags_[index][static_cast<std::size_t>(vertex)] = 0;
    if (!mesh_.live_vertex(vertex)) continue;
    incidence_.collect_star(vertex, star);
    for (std::int32_t slot : star) {
      const Tetrahedron& cell = mesh_.tet(slot);
      for (int corner = 0; corner < 4; ++corner) {
        const std::int32_t other = cell.v[corner];
        if (other == vertex || other < 0) continue;
        const double value = edge_length(vertex, other);
        if (wanted(value)) edges.emplace_back(value, edge_ids(vertex, other));
      }
    }
  }
  dirty_[index].clear();
  std::sort(edges.begin(), edges.end(), [](const auto& x, const auto& y) { return x.second < y.second; });
  edges.erase(std::unique(edges.begin(), edges.end(),
                          [](const auto& x, const auto& y) { return x.second == y.second; }),
              edges.end());
  std::erase_if(edges, [&](const auto& entry) {
    return protected_edges_.contains(entry.second);
  });
}

void Adapter::ensure_cell(std::int32_t slot) {
  const auto size = static_cast<std::size_t>(slot) + 1;
  if (cell_blocks_.size() < size) {
    cell_blocks_.resize(size, kNoBlock);
    cell_targets_.resize(size, std::numeric_limits<std::uint64_t>::max());
    cell_lineage_.resize(size);
    cell_quality_.resize(size, std::numeric_limits<double>::quiet_NaN());
  }
}

bool Adapter::vertex_metric(const Stencil& stencil, double* tensor) const {
  if (stencil.empty()) return false;
  thread_local std::vector<double> weights;
  thread_local std::vector<double> metrics;
  thread_local std::vector<double> logarithms;
  weights.clear();
  metrics.clear();
  logarithms.clear();
  for (const StencilTerm& term : stencil) {
    const auto row = static_cast<std::size_t>(term.row);
    weights.push_back(term.weight);
    metrics.insert(metrics.end(), input_.source_metrics.begin() + 9 * row,
                   input_.source_metrics.begin() + 9 * row + 9);
    logarithms.insert(logarithms.end(), input_.source_logarithms.begin() + 9 * row,
                      input_.source_logarithms.begin() + 9 * row + 9);
  }
  return interpolate(static_cast<std::int64_t>(stencil.size()), weights.data(), metrics.data(),
                     logarithms.data(), tensor);
}

void Adapter::bind() {
  const std::size_t sources = input_.source_vertex_ids.size();
  if (input_.source_points.size() != 3 * sources || input_.source_metrics.size() != 9 * sources ||
      input_.source_logarithms.size() != 9 * sources) {
    throw std::invalid_argument("The source banks disagree in size.");
  }
  for (std::uint64_t id : input_.source_vertex_ids) source_targets_.insert(id);
  TetMeshSnapshot snapshot;
  if (mesh_.snapshot(snapshot) != TetEditStatus::kApplied) {
    throw std::invalid_argument("The mesh has no snapshot.");
  }
  // Counts and strictly increasing IDs, followed by the live-slot checks
  // below, prove that every owner is bound exactly once before any edit.
  const auto covers = [](const std::vector<std::uint64_t>& ids, std::size_t live) {
    return ids.size() == live &&
           std::adjacent_find(ids.begin(), ids.end(), std::greater_equal<>{}) == ids.end();
  };
  if (!covers(input_.vertex_ids, snapshot.vertices.size()) ||
      !covers(input_.cell_ids, snapshot.cells.size())) {
    throw std::invalid_argument("Vertex and cell IDs must cover the live mesh in ascending order.");
  }
  const std::size_t vertices = input_.vertex_ids.size();
  if (input_.stencil_offsets.size() != vertices + 1 || input_.vertex_target_ids.size() != vertices) {
    throw std::invalid_argument("The vertex banks disagree in size.");
  }
  for (std::size_t index = 0; index < vertices; ++index) {
    const std::int32_t slot = mesh_.vertex_slot(input_.vertex_ids[index]);
    if (!mesh_.live_vertex(slot)) throw std::invalid_argument("A vertex ID is not live.");
    ensure_vertex(slot);
    Stencil& stencil = stencils_[static_cast<std::size_t>(slot)];
    for (std::int64_t term = input_.stencil_offsets[index];
         term < input_.stencil_offsets[index + 1]; ++term) {
      const std::int32_t row = input_.stencil_rows[static_cast<std::size_t>(term)];
      if (row < 0 || static_cast<std::size_t>(row) >= sources ||
          (!stencil.empty() && stencil.back().row >= row)) {
        throw std::invalid_argument("A stencil row is outside the source or out of order.");
      }
      stencil.push_back({row, input_.stencil_weights[static_cast<std::size_t>(term)]});
    }
    if (!vertex_metric(stencil, metrics_.data() + 9 * static_cast<std::size_t>(slot))) {
      throw std::invalid_argument("A vertex stencil has no metric.");
    }
    log_dets_[static_cast<std::size_t>(slot)] =
        log_determinant(metrics_.data() + 9 * static_cast<std::size_t>(slot));
    vertex_targets_[static_cast<std::size_t>(slot)] = input_.vertex_target_ids[index];
  }
  const std::size_t cells = input_.cell_ids.size();
  if (input_.cell_blocks.size() != cells || input_.cell_target_ids.size() != cells ||
      input_.ancestry_offsets.size() != cells + 1 ||
      input_.ancestry_kinds.size() != cells) {
    throw std::invalid_argument("The cell banks disagree in size.");
  }
  for (std::size_t index = 0; index < cells; ++index) {
    const std::int32_t slot = mesh_.cell_slot(input_.cell_ids[index]);
    if (!mesh_.complex().live(slot) || is_ghost(mesh_.tet(slot))) {
      throw std::invalid_argument("A cell ID is not live.");
    }
    ensure_cell(slot);
    const auto at = static_cast<std::size_t>(slot);
    cell_blocks_[at] = input_.cell_blocks[index];
    cell_targets_[at] = input_.cell_target_ids[index];
    CellLineage& lineage = cell_lineage_[at];
    lineage.kind = static_cast<AdaptLineage>(input_.ancestry_kinds[index]);
    for (std::int64_t row = input_.ancestry_offsets[index]; row < input_.ancestry_offsets[index + 1];
         ++row) {
      lineage.ancestry.push_back(input_.ancestry_ids[static_cast<std::size_t>(row)]);
    }
    std::sort(lineage.ancestry.begin(), lineage.ancestry.end());
  }
  if (input_.protected_edges.size() % 2 != 0) {
    throw std::invalid_argument("Protected edges are vertex ID pairs.");
  }
  for (std::size_t index = 0; index < input_.protected_edges.size(); index += 2) {
    protected_edges_.insert(id_edge(input_.protected_edges[index], input_.protected_edges[index + 1]));
  }
  next_vertex_id_ = input_.next_vertex_id;
  next_cell_id_ = input_.next_cell_id;
  next_target_vertex_ = input_.next_target_vertex;
  next_target_cell_ = input_.next_target_cell;
  for (const TetSnapshotFacet& facet : snapshot.constrained_facets) {
    next_facet_id_ = std::max(next_facet_id_, facet.id < kExhaustedId ? facet.id + 1 : facet.id);
  }
  for (const TetSnapshotSegment& segment : snapshot.constrained_segments) {
    next_segment_id_ =
        std::max(next_segment_id_, segment.id < kExhaustedId ? segment.id + 1 : segment.id);
  }
  if (!incidence_.initialize(mesh_)) throw std::invalid_argument("The mesh has no incidence.");
  source_banks_match_ = snapshot.source_points == input_.source_points;
  for (std::size_t row = 0; row < snapshot.source_face_groups.size(); ++row) {
    if (9 * row < snapshot.source_face_planes.size() &&
        !std::isnan(snapshot.source_face_planes[9 * row])) {
      declared_groups_.insert(snapshot.source_face_groups[row]);
    }
  }
  if (input_.exact_source != nullptr &&
      !input_.exact_source->matches_roundings(input_.source_points, input_.source_cells)) {
    throw std::invalid_argument("The authored source must match the adaptation point and cell rows.");
  }
  if (input_.relocation) {
    carriers_ = input_.exact_source;
    if (carriers_ == nullptr) {
      if (input_.source_cells.size() % 4 != 0) {
        throw std::invalid_argument("Source cells are four point rows each.");
      }
      // Binary64 source points share the denominator 2^shift.
      std::vector<Dyadic> numerators;
      numerators.reserve(input_.source_points.size());
      std::int64_t shift = 0;
      for (double value : input_.source_points) {
        numerators.emplace_back(value);
        if (!numerators.back().is_zero()) shift = std::max(shift, -numerators.back().exponent());
      }
      for (Dyadic& numerator : numerators) numerator = numerator.scaled_power(shift);
      binary64_carriers_.emplace(input_.source_points, std::move(numerators),
                                 Dyadic(1.0).scaled_power(shift), input_.source_cells);
      carriers_ = &*binary64_carriers_;
    }
    // Facet groups of the mesh's own source complex, when it is this source.
    const std::size_t faces = snapshot.source_faces.size() / 3;
    for (std::size_t row = 0; source_banks_match_ && row < faces; ++row) {
      CarrierSimplex triangle;
      triangle.size = 3;
      for (std::size_t corner = 0; corner < 3; ++corner) {
        triangle.rows[corner] = snapshot.source_faces[3 * row + corner];
      }
      // Boundary moves keep an exactly representable source image for the
      // verified collapse stencil and edges outside declared facet pairs.
      group_triangles_[snapshot.source_face_groups[row]].simplices.push_back(triangle);
    }
    for (auto& entry : group_triangles_) carriers_->prepare_chart(entry.second);
    bind_rounded_sources(snapshot);
    if (source_banks_match_ &&
        std::any_of(snapshot.vertices.begin(), snapshot.vertices.end(),
                    [](const TetSnapshotVertex& vertex) {
                      return vertex.witness.stratum == SourceStratum::kFacet;
                    })) {
      witness_face_groups_ = std::move(snapshot.source_face_groups);
      witness_faces_ = std::move(snapshot.source_faces);
    }
  }
}

void Adapter::bind_rounded_sources(const TetMeshSnapshot& snapshot) {
  rounded_sources_ = source_banks_match_ && input_.exact_source == nullptr &&
                     !mesh_.implicit_source() && !snapshot.periodic_identity.bound &&
                     snapshot.periodic_orbits.empty() && snapshot.periodic_entity_maps.empty();
  if (!rounded_sources_) return;
  // Rounded triangle coordinates do not identify the declared plane: an
  // oblique declaration can intersect every corner's binary64 rounding box.
  constexpr std::size_t no_axis = 3;
  for (std::size_t row = 0; row < snapshot.source_face_groups.size(); ++row) {
    const double* plane = snapshot.source_face_planes.data() + 9 * row;
    std::size_t axis = no_axis;
    if (snapshot.source_face_plane_tolerances[row] == 0.0 && std::isfinite(plane[0])) {
      axis = 0;
      while (axis < no_axis && (plane[3 + axis] != 0.0 || plane[6 + axis] != 0.0)) ++axis;
      for (std::size_t corner = 0; axis < no_axis && corner < 3; ++corner) {
        const auto point = static_cast<std::size_t>(snapshot.source_faces[3 * row + corner]);
        if (snapshot.source_points[3 * point + axis] != plane[axis]) axis = no_axis;
      }
    }
    const auto [entry, inserted] = exact_plane_axes_.try_emplace(snapshot.source_face_groups[row], axis);
    if (!inserted && entry->second != axis) entry->second = no_axis;
  }
  std::erase_if(exact_plane_axes_, [](const auto& entry) { return entry.second == no_axis; });
  if (exact_plane_axes_.empty()) return;
  std::unordered_set<std::int32_t> live_groups;
  for (const TetSnapshotSegment& segment : snapshot.constrained_segments) {
    live_groups.insert(segment.source);
  }
  const std::size_t count = snapshot.source_segment_groups.size();
  source_only_segments_.resize(count, {-1, -1});
  for (std::size_t row = 0; row < count; ++row) {
    if (live_groups.contains(snapshot.source_segment_groups[row]) ||
        snapshot.source_segment_tolerances[row] != 0.0) continue;
    const std::int32_t a = snapshot.source_segments[2 * row];
    const std::int32_t b = snapshot.source_segments[2 * row + 1];
    source_only_segments_[row] = {a, b};
    const auto [record, inserted] = source_only_segment_rows_.try_emplace(
        edge_key(a, b), static_cast<std::int32_t>(row));
    if (!inserted) record->second = -1;  // ambiguous primitive
  }
}

bool Adapter::complete(double lower) const {
  const auto& tets = mesh_.complex().tets;
  for (std::size_t slot = 0; slot < tets.size(); ++slot) {
    const Tetrahedron& cell = tets[slot];
    if (cell.v[0] == kDeadVertex || is_ghost(cell)) continue;
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const std::int32_t a = cell.v[first];
        const std::int32_t b = cell.v[second];
        const double value = edge_length(a, b);
        if (!(value >= lower && value <= kUpperLength) && !protected_edge(a, b)) {
          return false;
        }
      }
    }
  }
  return true;
}

void Adapter::count_refusal(TetEditStatus status) {
  switch (status) {
    case TetEditStatus::kUncertainPredicate:
      ++counters_.rejected_uncertain;
      break;
    case TetEditStatus::kRejectedProtected:
      ++counters_.rejected_protected;
      break;
    case TetEditStatus::kRejectedTopology:
      ++counters_.rejected_topology;
      break;
    default:
      ++counters_.rejected_invalid;
      break;
  }
}

void Adapter::record_created_cell(std::int32_t slot, std::int32_t parent_block,
                                  const CellLineage& parent, AdaptLineage kind) {
  ensure_cell(slot);
  cell_quality_[static_cast<std::size_t>(slot)] = std::numeric_limits<double>::quiet_NaN();
  for (std::int32_t vertex : mesh_.tet(slot).v) touch(vertex);
  const auto at = static_cast<std::size_t>(slot);
  cell_blocks_[at] = parent_block;
  cell_targets_[at] = next_target_cell_++;
  CellLineage& lineage = cell_lineage_[at];
  lineage.ancestry = parent.ancestry;
  lineage.kind = kind;
  lineage.relocated = false;
}

void Adapter::retire_cell(std::int32_t slot) {
  const auto at = static_cast<std::size_t>(slot);
  cell_blocks_[at] = kNoBlock;
  cell_targets_[at] = std::numeric_limits<std::uint64_t>::max();
  cell_lineage_[at] = CellLineage{};
}

// ---------------------------------------------------------------- splits ---

std::optional<Adapter::EdgeSupport> Adapter::edge_support(
    std::int32_t a, std::int32_t b, std::span<const std::int32_t> shell) {
  if (mesh_.vertex_dimension(a) <= 1 && mesh_.vertex_dimension(b) <= 1) {
    const TetSnapshotSegment* segment = nullptr;
    if (incidence_.boundary_segment({mesh_.vertex_id(a), mesh_.vertex_id(b)}, segment) !=
        TetEditStatus::kApplied) return std::nullopt;
    if (segment != nullptr) return EdgeSupport::kBatch;
  }
  int faces = 0;
  bool hull = true;
  std::int32_t group = kNoConstraint;
  bool one_group = true;
  for (std::int32_t slot : shell) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int face = 0; face < 4; ++face) {
      if (cell.v[face] == a || cell.v[face] == b) continue;
      const std::int32_t source = mesh_.complex().constraint(slot, face);
      if (source == kNoConstraint) continue;
      if (faces == 0) group = source;
      one_group = one_group && group == source;
      hull = hull && is_ghost(mesh_.tet(cell.n[face]));
      ++faces;
    }
  }
  if (faces == 0) return EdgeSupport::kFree;
  if (!hull || faces != 2 || mesh_.implicit_source()) return EdgeSupport::kBatch;
  return one_group && declared_groups_.contains(group) ? EdgeSupport::kDeclaredHull
                                                       : EdgeSupport::kHull;
}

bool Adapter::split_parameter(std::int32_t a, std::int32_t b, SplitPointRule rule,
                              Stencil& stencil, double* point, double& parameter) const {
  const bool exact_edge = rule == SplitPointRule::kExactEdge;
  const double* pa = mesh_.point(a);
  const double* pb = mesh_.point(b);
  const double middle = midpoint_parameter(pa, pb, metric(a), metric(b));
  if (!std::isfinite(middle)) return false;
  double tried = -1.0;
  for (int bits = kSplitBits; bits >= 1; --bits) {
    const double scale = std::ldexp(1.0, bits);
    const double numerator = std::clamp(std::nearbyint(middle * scale), 1.0, scale - 1.0);
    const double t = numerator / scale;
    if (t == tried) continue;
    tried = t;
    if (!combine_stencils(stencils_[static_cast<std::size_t>(a)],
                          stencils_[static_cast<std::size_t>(b)], t, stencil)) {
      continue;
    }
    // The carrier is the correctly rounded source stencil image. An exact
    // edge rule additionally checks representability; the rounded edge rule
    // checks agreement with the rounded carrier-to-carrier interpolation.
    bool admissible = true;
    for (int axis = 0; axis < 3 && admissible; ++axis) {
      const auto edge_coordinate = [&]() {
        return Expansion(pa[axis]) + Expansion::difference(pb[axis], pa[axis]).scaled(t);
      };
      if (input_.exact_source != nullptr) {
        point[axis] = input_.exact_source->stencil_coordinate(stencil, axis, exact_edge, admissible);
        if (exact_edge && admissible) {
          admissible = (edge_coordinate() - Expansion(point[axis])).is_zero();
        } else if (admissible && rule == SplitPointRule::kRoundedEdge) {
          admissible = nearest_double(edge_coordinate()) == point[axis];
        }
        continue;
      }
      Expansion image;
      for (const StencilTerm& term : stencil) {
        image = image + Expansion::product(
                            term.weight,
                            input_.source_points[3 * static_cast<std::size_t>(term.row) + axis]);
      }
      const double rounded = nearest_double(image);
      if (exact_edge) {
        admissible =
            (edge_coordinate() - image).is_zero() && (image - Expansion(rounded)).is_zero();
      } else if (rule == SplitPointRule::kRoundedEdge) {
        admissible = nearest_double(edge_coordinate()) == rounded;
      }
      point[axis] = rounded;
    }
    if (admissible) {
      parameter = t;
      return true;
    }
  }
  return false;
}

TetEditStatus Adapter::split(std::int32_t a, std::int32_t b, bool& applied) {
  applied = false;
  const std::span<const std::int32_t> shell_view = incidence_.edge_star(a, b);
  if (shell_view.empty()) return TetEditStatus::kApplied;
  std::vector<std::int32_t> shell(shell_view.begin(), shell_view.end());
  const auto by_id = [&](std::int32_t p, std::int32_t q) {
    return mesh_.cell_id(p) < mesh_.cell_id(q);
  };
  std::sort(shell.begin(), shell.end(), by_id);
  for (std::int32_t slot : shell) {
    if (static_cast<std::size_t>(slot) < occupied_.size() &&
        occupied_[static_cast<std::size_t>(slot)] != 0) {
      return TetEditStatus::kApplied;  // in this sweep's boundary batch
    }
  }
  const auto support = edge_support(a, b, shell);
  if (!support) return TetEditStatus::kStaleGeneration;
  const std::uint64_t key = edge_key(a, b);
  // The cache and its final evidence decode slot order, not operation order.
  const auto [first, second] = std::minmax(a, b);
  const SplitRefusal current{carrier_revisions_[static_cast<std::size_t>(first)],
                             carrier_revisions_[static_cast<std::size_t>(second)], *support};
  if (const auto failed = unrepresentable_.find(key); failed != unrepresentable_.end()) {
    if (failed->second == current) return TetEditStatus::kApplied;
    unrepresentable_.erase(failed);
  }

  Stencil stencil;
  std::array<double, 3> point{};
  double parameter = 0.0;
  bool exact_edge = *support != EdgeSupport::kFree;
  bool have_point = split_parameter(
      a, b, exact_edge ? SplitPointRule::kExactEdge : SplitPointRule::kStencil,
      stencil, point.data(), parameter);
  if (!have_point && *support == EdgeSupport::kDeclaredHull) {
    // Preserve exact edge witnesses where possible. Otherwise the kernel
    // proves coverage against the pair using the rounded source image.
    exact_edge = false;
    have_point = split_parameter(a, b, SplitPointRule::kStencil, stencil, point.data(), parameter);
  } else if (!have_point && *support == EdgeSupport::kHull) {
    const double* pa = mesh_.point(a);
    const double* pb = mesh_.point(b);
    const int fixed = (pa[0] == pb[0]) + (pa[1] == pb[1]) + (pa[2] == pb[2]);
    // Rounding the varying coordinate of an axis edge leaves the point
    // on its exact line. The local edit still proves both child facets.
    if (fixed == 2) {
      have_point =
          split_parameter(a, b, SplitPointRule::kRoundedEdge, stencil, point.data(), parameter);
    }
  }
  if (!have_point) {
    unrepresentable_.emplace(key, current);
    ++counters_.rejected_invalid;
    return TetEditStatus::kApplied;
  }
  std::array<double, 9> tensor{};
  if (!vertex_metric(stencil, tensor.data())) {
    ++counters_.rejected_invalid;
    return TetEditStatus::kApplied;
  }

  // Effects: the halves and ring medians, and the floor of every child.
  Violations removed;
  removed.add(edge_length(a, b));
  removed.finish();
  Violations added;
  std::vector<std::int32_t> others;
  for (std::int32_t slot : shell) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int corner = 0; corner < 4; ++corner) others.push_back(cell.v[corner]);
    for (std::int32_t end : {a, b}) {
      std::array<double, 12> corners{};
      for (int corner = 0; corner < 4; ++corner) {
        const double* source = cell.v[corner] == end ? point.data() : mesh_.point(cell.v[corner]);
        std::copy_n(source, 3, corners.data() + 3 * corner);
      }
      double determinant = 0.0;
      double relative = 0.0;
      relative_determinant(corners.data(), determinant, relative);
      if (!(determinant > 0.0) || !(relative > kDeterminantFloor)) return TetEditStatus::kApplied;
    }
  }
  std::sort(others.begin(), others.end());
  others.erase(std::unique(others.begin(), others.end()), others.end());
  for (std::int32_t other : others) {
    const double* fixed = mesh_.point(other);
    const double* fixed_tensor = metric(other);
    added.add(mesh_.vertex_id(other) < next_vertex_id_
                  ? length(fixed, point.data(), fixed_tensor, tensor.data())
                  : length(point.data(), fixed, tensor.data(), fixed_tensor));
  }
  if (!added.measured || !removed.measured) return TetEditStatus::kApplied;
  added.finish();
  if (!objective_decreases(removed, added, false)) return TetEditStatus::kApplied;
  if (separated_splits_ && !added.short_side.empty()) return TetEditStatus::kApplied;

  if (*support == EdgeSupport::kBatch) {
    for (std::int32_t slot : shell) {
      if (occupied_.size() <= static_cast<std::size_t>(slot)) {
        occupied_.resize(static_cast<std::size_t>(slot) + 1, 0);
      }
      occupied_[static_cast<std::size_t>(slot)] = 1;
    }
    pending_.push_back({a, b, parameter, point, tensor, std::move(stencil), shell});
    return TetEditStatus::kApplied;
  }
  if (!fits_ids(std::max(next_vertex_id_, next_target_vertex_), 1)) {
    return TetEditStatus::kIndexOverflow;
  }
  // A boundary split makes two children per parent. A free split cones every
  // exposed face, so count those faces if the target range cannot cover the
  // trivial upper bound. Both transactions check their native cell allocator.
  if (*support != EdgeSupport::kFree) {
    if (!fits_ids(next_target_cell_, 2 * shell.size())) return TetEditStatus::kIndexOverflow;
  } else if (!fits_ids(next_target_cell_, 4 * shell.size())) {
    std::size_t faces = 0;
    for (std::int32_t slot : shell) {
      for (std::int32_t neighbor : mesh_.tet(slot).n) {
        if (!mesh_.complex().live(neighbor) || is_ghost(mesh_.tet(neighbor)) ||
            !std::binary_search(shell.begin(), shell.end(), neighbor, by_id)) {
          ++faces;
        }
      }
    }
    if (!fits_ids(next_target_cell_, faces)) return TetEditStatus::kIndexOverflow;
  }

  // Parents by their two faces opposite the edge ends, before the commit.
  struct Parent {
    std::array<std::int32_t, 3> key;
    std::int32_t row;
  };
  std::vector<Parent> parents;
  std::vector<CellLineage> parent_lineage;
  std::vector<std::int32_t> parent_blocks;
  for (std::int32_t slot : shell) {
    const Tetrahedron& cell = mesh_.tet(slot);
    std::array<std::int32_t, 2> rest{};
    int count = 0;
    for (int corner = 0; corner < 4; ++corner) {
      if (cell.v[corner] != a && cell.v[corner] != b) {
        rest[static_cast<std::size_t>(count++)] = cell.v[corner];
      }
    }
    for (std::int32_t end : {a, b}) {
      std::array<std::int32_t, 3> key{end, rest[0], rest[1]};
      std::sort(key.begin(), key.end());
      parents.push_back({key, static_cast<std::int32_t>(parent_lineage.size())});
    }
    parent_lineage.push_back(cell_lineage_[static_cast<std::size_t>(slot)]);
    parent_blocks.push_back(cell_blocks_[static_cast<std::size_t>(slot)]);
  }

  TetEditVertex vertex;
  vertex.id = next_vertex_id_;
  vertex.carrier = point;
  const bool boundary = *support != EdgeSupport::kFree;
  if (boundary && exact_edge) {
    vertex.has_edge_parameter = true;
    vertex.edge_witness_supplied = true;
    vertex.edge_endpoints = {mesh_.vertex_id(a), mesh_.vertex_id(b)};
    vertex.edge_parameter = parameter;
    vertex.ancestry = {{vertex.edge_endpoints[0], {0, 0, 0}, 1.0 - parameter},
                       {vertex.edge_endpoints[1], {0, 0, 0}, parameter}};
  }
  std::vector<std::uint64_t> created;
  const TetEditStatus status =
      boundary ? incidence_.split_boundary_edge(mesh_, a, b, vertex, created)
               : incidence_.insert_cone(mesh_, vertex, shell, {}, created);
  if (status != TetEditStatus::kApplied) {
    if (fatal(status)) return status;
    count_refusal(status);
    return TetEditStatus::kApplied;
  }
  // Local splits draw cell IDs from the mesh's allocator.
  ++next_vertex_id_;
  for (std::uint64_t id : created) next_cell_id_ = std::max(next_cell_id_, id + 1);
  for (std::int32_t slot : shell) retire_cell(slot);
  const std::int32_t inserted = mesh_.vertex_slot(vertex.id);
  ensure_vertex(inserted);
  stencils_[static_cast<std::size_t>(inserted)] = std::move(stencil);
  set_metric(inserted, tensor.data());
  vertex_targets_[static_cast<std::size_t>(inserted)] = next_target_vertex_++;
  for (std::uint64_t id : created) {
    const std::int32_t slot = mesh_.cell_slot(id);
    if (boundary && !sync_split_facets(slot, inserted)) return TetEditStatus::kStaleGeneration;
    const Tetrahedron& cell = mesh_.tet(slot);
    std::array<std::int32_t, 3> key{};
    int count = 0;
    for (int corner = 0; corner < 4; ++corner) {
      if (cell.v[corner] != inserted) key[static_cast<std::size_t>(count++)] = cell.v[corner];
    }
    std::sort(key.begin(), key.end());
    const auto parent = std::find_if(parents.begin(), parents.end(),
                                     [&](const Parent& entry) { return entry.key == key; });
    if (parent == parents.end()) return TetEditStatus::kStaleGeneration;
    const auto row = static_cast<std::size_t>(parent->row);
    record_created_cell(slot, parent_blocks[row], parent_lineage[row], AdaptLineage::kRefinedFrom);
  }
  ++counters_.splits;
  applied = true;
  return TetEditStatus::kApplied;
}

bool Adapter::sync_split_facets(std::int32_t slot, std::int32_t inserted) {
  const Tetrahedron& cell = mesh_.tet(slot);
  for (int face = 0; face < 4; ++face) {
    if (cell.v[face] == inserted || mesh_.complex().constraint(slot, face) == kNoConstraint) {
      continue;
    }
    IdFace ids{};
    std::size_t count = 0;
    for (int corner = 0; corner < 4; ++corner) {
      if (corner != face) ids[count++] = mesh_.vertex_id(cell.v[corner]);
    }
    const TetSnapshotFacet* record = nullptr;
    if (incidence_.boundary_facet(ids, record) != TetEditStatus::kApplied || record == nullptr) {
      return false;
    }
    next_facet_id_ = std::max(next_facet_id_, record->id + 1);
  }
  return true;
}

TetEditStatus Adapter::apply_split_batch(std::span<const PendingSplit> splits,
                                         std::int64_t& applied) {
  // The constrained facets and segments of each split edge come from the
  // incidence index through the edge's shell.
  std::vector<TetSnapshotFacet> through;
  TetEdit op;
  op.kind = TetEditKind::kSplit;
  std::uint64_t vertex_id = next_vertex_id_;
  std::uint64_t cell_id = next_cell_id_;
  std::uint64_t facet_id = next_facet_id_;
  std::uint64_t segment_id = next_segment_id_;
  struct Child {
    std::uint64_t id;
    std::int32_t parent;  // shell slot
  };
  std::vector<Child> children;
  for (const PendingSplit& split : splits) {
    if (!fits_ids(vertex_id, 1) || !fits_ids(next_target_vertex_, op.changed_vertices.size() + 1) ||
        !fits_ids(cell_id, 2 * split.shell.size()) ||
        !fits_ids(next_target_cell_, op.replacement_cells.size() + 2 * split.shell.size())) {
      return TetEditStatus::kIndexOverflow;
    }
    const std::uint64_t ida = mesh_.vertex_id(split.a);
    const std::uint64_t idb = mesh_.vertex_id(split.b);
    const bool forward = ida < idb;
    const std::uint64_t low = forward ? ida : idb;
    const std::uint64_t high = forward ? idb : ida;
    const double t = forward ? split.parameter : 1.0 - split.parameter;
    const std::uint64_t middle = vertex_id++;
    op.split_edges.push_back({low, high});
    TetEditVertex vertex;
    vertex.id = middle;
    vertex.carrier = split.point;
    vertex.has_edge_parameter = true;
    vertex.edge_witness_supplied = true;
    vertex.edge_endpoints = {low, high};
    vertex.edge_parameter = t;
    vertex.ancestry.push_back({low, {0, 0, 0}, 1.0 - t});
    vertex.ancestry.push_back({high, {0, 0, 0}, t});
    op.changed_vertices.push_back(std::move(vertex));
    for (std::int32_t slot : split.shell) {
      const Tetrahedron& cell = mesh_.tet(slot);
      op.removed_cell_ids.push_back(mesh_.cell_id(slot));
      // Child 0 keeps a (b becomes the middle), child 1 keeps b.
      for (std::int32_t moved : {split.b, split.a}) {
        const std::int32_t kept = moved == split.b ? split.a : split.b;
        TetEditCell child;
        child.id = cell_id++;
        child.region = mesh_.complex().region(slot);
        child.ancestor_cell_ids.push_back(mesh_.cell_id(slot));
        for (int corner = 0; corner < 4; ++corner) {
          child.vertices[static_cast<std::size_t>(corner)] =
              cell.v[corner] == moved ? middle : mesh_.vertex_id(cell.v[corner]);
          // A face keeps its label (the halves of a constrained face keep its
          // source), except the new face between the children.
          child.facet_sources[static_cast<std::size_t>(corner)] =
              cell.v[corner] == kept ? kNoConstraint : mesh_.complex().constraint(slot, corner);
        }
        children.push_back({child.id, slot});
        op.replacement_cells.push_back(std::move(child));
      }
    }
    const IdEdge key = id_edge(ida, idb);
    if (!facets_through(split.a, split.shell, through)) return TetEditStatus::kStaleGeneration;
    for (const TetSnapshotFacet& facet : through) {
      if (std::find(facet.vertices.begin(), facet.vertices.end(), idb) == facet.vertices.end()) {
        continue;
      }
      op.removed_facet_ids.push_back(facet.id);
      // The facet's directed edge over {a, b} and its opposite corner.
      for (int side = 0; side < 3; ++side) {
        const std::uint64_t x = facet.vertices[static_cast<std::size_t>(side)];
        const std::uint64_t y = facet.vertices[static_cast<std::size_t>((side + 1) % 3)];
        if (id_edge(x, y) != key) continue;
        if (!fits_ids(facet_id, 2)) return TetEditStatus::kIndexOverflow;
        const std::uint64_t z = facet.vertices[static_cast<std::size_t>((side + 2) % 3)];
        for (const std::array<std::uint64_t, 3>& half :
             {std::array<std::uint64_t, 3>{x, middle, z}, std::array<std::uint64_t, 3>{middle, y, z}}) {
          TetEditFacet replacement;
          replacement.id = facet_id++;
          replacement.vertices = half;
          replacement.source = facet.source;
          replacement.orientation = facet.orientation;
          replacement.protected_facet = facet.protected_facet;
          op.replacement_facets.push_back(replacement);
        }
        break;
      }
    }
    const TetSnapshotSegment* segment = nullptr;
    const TetEditStatus lookup = incidence_.boundary_segment(key, segment);
    if (lookup != TetEditStatus::kApplied) return lookup;
    if (segment != nullptr) {
      if (!fits_ids(segment_id, 2)) return TetEditStatus::kIndexOverflow;
      const TetSnapshotSegment& old = *segment;
      op.removed_segment_ids.push_back(old.id);
      for (const std::array<std::uint64_t, 2>& half :
           {std::array<std::uint64_t, 2>{old.vertices[0], middle},
            std::array<std::uint64_t, 2>{middle, old.vertices[1]}}) {
        TetEditSegment replacement;
        replacement.id = segment_id++;
        replacement.vertices = half;
        replacement.source = old.source;
        replacement.protected_segment = old.protected_segment;
        op.replacement_segments.push_back(replacement);
      }
    }
  }
  std::sort(op.removed_cell_ids.begin(), op.removed_cell_ids.end());
  std::sort(op.removed_facet_ids.begin(), op.removed_facet_ids.end());
  std::sort(op.removed_segment_ids.begin(), op.removed_segment_ids.end());

  TetEditBatch batch;
  batch.expected_generation = mesh_.generation();
  batch.identity_bound = mesh_.identity_bound();
  batch.source_id = mesh_.source_id();
  batch.source_revision = mesh_.source_revision();
  batch.source_map_id = mesh_.source_map_id();
  batch.frame_id = mesh_.frame_id();
  batch.periodic_identity = mesh_.periodic_identity();
  batch.operations.push_back(std::move(op));
  // Lineage of the parents before the commit reuses their slots.
  std::vector<std::pair<std::int32_t, CellLineage>> parents;
  for (const PendingSplit& split : splits) {
    for (std::int32_t slot : split.shell) {
      parents.emplace_back(cell_blocks_[static_cast<std::size_t>(slot)],
                           cell_lineage_[static_cast<std::size_t>(slot)]);
    }
  }
  std::vector<TetEditResult> results;
  const TetEditStatus status = mesh_.apply_edit_batch(batch, results);
  if (status != TetEditStatus::kApplied) {
    if (fatal(status)) return status;
    if (splits.size() > 1) {
      const std::size_t half = splits.size() / 2;
      const TetEditStatus first = apply_split_batch(splits.first(half), applied);
      if (first != TetEditStatus::kApplied) return first;
      return apply_split_batch(splits.subspan(half), applied);
    }
    count_refusal(status);
    return TetEditStatus::kApplied;
  }
  next_vertex_id_ = vertex_id;
  next_cell_id_ = cell_id;
  next_facet_id_ = facet_id;
  next_segment_id_ = segment_id;
  if (!incidence_.initialize(mesh_)) return TetEditStatus::kStaleGeneration;
  std::size_t parent_row = 0;
  std::uint64_t middle = vertex_id - splits.size();
  for (const PendingSplit& split : splits) {
    for (std::int32_t slot : split.shell) retire_cell(slot);
  }
  for (const PendingSplit& split : splits) {
    const std::int32_t inserted = mesh_.vertex_slot(middle++);
    ensure_vertex(inserted);
    const auto at = static_cast<std::size_t>(inserted);
    stencils_[at] = split.stencil;
    set_metric(inserted, split.tensor.data());
    vertex_targets_[at] = next_target_vertex_++;
  }
  for (std::size_t row = 0; row < children.size(); ++row) {
    const auto& [block, lineage] = parents[parent_row + row / 2];
    record_created_cell(mesh_.cell_slot(children[row].id), block, lineage,
                        AdaptLineage::kRefinedFrom);
  }
  counters_.splits += static_cast<std::int64_t>(splits.size());
  applied += static_cast<std::int64_t>(splits.size());
  return TetEditStatus::kApplied;
}

TetEditStatus Adapter::commit_pending(std::int64_t& applied) {
  if (pending_.empty()) return TetEditStatus::kApplied;
  const TetEditStatus status = apply_split_batch(pending_, applied);
  pending_.clear();
  std::fill(occupied_.begin(), occupied_.end(), 0);
  return status;
}

std::int64_t Adapter::split_sweep() {
  std::vector<MeasuredEdge> long_edges;
  dirty_edges(kSplitPhase, [](double value) { return value > kUpperLength; }, long_edges);
  std::sort(long_edges.begin(), long_edges.end(),
            [](const auto& x, const auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
  std::int64_t applied = 0;
  for (const auto& [value, key] : long_edges) {
    const auto a = mesh_.vertex_slot(key[0]);
    const auto b = mesh_.vertex_slot(key[1]);
    if (!mesh_.live_vertex(a) || !mesh_.live_vertex(b)) continue;
    if (edge_length(a, b) != value) continue;  // a moved end: next sweep
    bool done = false;
    const TetEditStatus status = split(a, b, done);
    if (status != TetEditStatus::kApplied) {
      fatal_ = status;
      return applied;
    }
    applied += done ? 1 : 0;
  }
  const TetEditStatus committed = commit_pending(applied);
  if (committed != TetEditStatus::kApplied) fatal_ = committed;
  return applied;
}

// ------------------------------------------------------------- collapses ---

bool Adapter::may_collapse_in_place(std::int32_t removed, std::int32_t kept,
                                    const LocalIncidence::Star& removed_star) const {
  double longest_removed = 0.0;
  double longest_candidate = 0.0;
  for (std::int32_t slot : removed_star) {
    for (std::int32_t other : mesh_.tet(slot).v) {
      if (other == removed) continue;
      if (!protected_edge(removed, other)) {
        longest_removed = std::max(longest_removed, edge_length(removed, other));
      }
      if (other != kept && !protected_edge(kept, other)) {
        longest_candidate = std::max(longest_candidate, edge_length(kept, other));
      }
    }
  }
  double above_removed = 0.0;
  double above_candidate = 0.0;
  double below = 0.0;
  band_quanta(longest_removed, above_removed, below);
  band_quanta(longest_candidate, above_candidate, below);
  if (!(above_candidate > above_removed)) return true;

  // Retained survivor spokes cancel in the full band comparison. Confirm
  // that an excess quantum belongs to a new edge before refusing a collapse.
  const double threshold = std::max(longest_removed, kUpperLength);
  for (std::int32_t slot : removed_star) {
    for (std::int32_t other : mesh_.tet(slot).v) {
      if (other == removed || other == kept || protected_edge(kept, other)) continue;
      const double value = edge_length(kept, other);
      if (!(value > threshold)) continue;
      band_quanta(value, above_candidate, below);
      if (above_candidate > above_removed && incidence_.edge_valence(kept, other) == 0) {
        return false;
      }
    }
  }
  return true;
}

bool Adapter::plan_collapse(std::int32_t removed, std::int32_t kept, CollapsePlan& plan) {
  const int dimension = mesh_.vertex_dimension(removed);
  if (mesh_.protected_vertex(removed) || mesh_.protection_radius(removed) > 0.0 ||
      dimension < 1) {
    return false;
  }
  // A boundary vertex collapses onto a survivor on every facet group it lies
  // on (so it stays on each of their planes), along a constrained facet; the
  // batch commit refuses a protected survivor.
  if (dimension < 3 && (mesh_.protected_vertex(kept) || mesh_.protection_radius(kept) > 0.0)) {
    return false;
  }
  const LocalIncidence::Star removed_star = incidence_.vertex_star(removed);
  const bool in_place = may_collapse_in_place(removed, kept, removed_star);
  if (!in_place && !survivor_moves_) return false;
  const LocalIncidence::Star kept_star = incidence_.vertex_star(kept);
  if (removed_star.empty() || kept_star.empty()) return false;
  if (dimension < 3) {
    const std::vector<std::int32_t> removed_groups = facet_groups(removed, removed_star.cells());
    const std::vector<std::int32_t> kept_groups = facet_groups(kept, kept_star.cells());
    if (removed_groups.empty() || removed_groups.size() > 2 ||
        !std::includes(kept_groups.begin(), kept_groups.end(), removed_groups.begin(),
                       removed_groups.end())) {
      return false;
    }
    bool along_facet = false;
    for (std::int32_t slot : removed_star) {
      const Tetrahedron& cell = mesh_.tet(slot);
      if (vertex_slot(cell, kept) < 0) continue;
      for (int face = 0; face < 4 && !along_facet; ++face) {
        if (cell.v[face] == removed || cell.v[face] == kept) continue;
        along_facet = mesh_.complex().constraint(slot, face) != kNoConstraint;
      }
    }
    if (!along_facet) return false;
  }
  const std::int32_t region = mesh_.complex().region(removed_star[0]);
  const std::int32_t block = cell_blocks_[static_cast<std::size_t>(removed_star[0])];
  for (const LocalIncidence::Star* star : {&removed_star, &kept_star}) {
    for (std::int32_t slot : *star) {
      if (mesh_.complex().region(slot) != region ||
          cell_blocks_[static_cast<std::size_t>(slot)] != block) {
        return false;
      }
    }
  }
  // Links and the link condition.
  std::vector<std::int32_t> removed_link;
  std::vector<std::int32_t> kept_link;
  std::vector<std::int32_t> edge_link;
  for (std::int32_t slot : removed_star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    const bool shared = vertex_slot(cell, kept) >= 0;
    for (int corner = 0; corner < 4; ++corner) {
      const std::int32_t v = cell.v[corner];
      if (v == removed) continue;
      removed_link.push_back(v);
      if (shared && v != kept) edge_link.push_back(v);
    }
  }
  for (std::int32_t slot : kept_star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int corner = 0; corner < 4; ++corner) {
      if (cell.v[corner] != kept) kept_link.push_back(cell.v[corner]);
    }
  }
  for (std::vector<std::int32_t>* link : {&removed_link, &kept_link, &edge_link}) {
    std::sort(link->begin(), link->end());
    link->erase(std::unique(link->begin(), link->end()), link->end());
  }
  std::vector<std::int32_t> common;
  std::set_intersection(removed_link.begin(), removed_link.end(), kept_link.begin(),
                        kept_link.end(), std::back_inserter(common));
  if (common != edge_link) return false;

  // Star rows: 0 the survivor, 1 the removed vertex, then the union of links.
  std::vector<std::int32_t> rows{kept, removed};
  std::vector<std::int32_t> merged;
  std::set_union(removed_link.begin(), removed_link.end(), kept_link.begin(), kept_link.end(),
                 std::back_inserter(merged));
  for (std::int32_t v : merged) {
    if (v != kept && v != removed) rows.push_back(v);
  }
  const auto row_of = [&](std::int32_t v) {
    return static_cast<std::int64_t>(std::find(rows.begin(), rows.end(), v) - rows.begin());
  };
  std::vector<double> points(3 * rows.size());
  std::vector<double> tensors(9 * rows.size());
  for (std::size_t row = 0; row < rows.size(); ++row) {
    std::copy_n(mesh_.point(rows[row]), 3, points.data() + 3 * row);
    std::copy_n(metric(rows[row]), 9, tensors.data() + 9 * row);
  }
  // Before: every measured spoke of both ends (the edge once).
  std::vector<double> before;
  for (std::int32_t v : removed_link) {
    if (!protected_edge(removed, v)) before.push_back(edge_length(removed, v));
  }
  for (std::int32_t v : kept_link) {
    if (v != removed && !protected_edge(kept, v)) before.push_back(edge_length(kept, v));
  }
  std::vector<std::int64_t> spokes;
  std::vector<std::uint8_t> neighbor_first;
  for (std::size_t row = 2; row < rows.size(); ++row) {
    if (protected_edge(kept, rows[row])) continue;
    spokes.push_back(static_cast<std::int64_t>(row));
    neighbor_first.push_back(mesh_.vertex_id(rows[row]) < mesh_.vertex_id(kept));
  }
  std::vector<std::int64_t> cells;
  for (std::int32_t slot : removed_star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    if (vertex_slot(cell, kept) >= 0) continue;
    for (int corner = 0; corner < 4; ++corner) {
      cells.push_back(cell.v[corner] == removed ? 0 : row_of(cell.v[corner]));
    }
  }
  if (cells.empty()) return false;
  CollapseStar star;
  star.point_count = static_cast<std::int64_t>(rows.size());
  star.points = points.data();
  star.metrics = tensors.data();
  star.spoke_count = static_cast<std::int64_t>(spokes.size());
  star.spokes = spokes.data();
  star.neighbor_first = neighbor_first;
  star.before_count = static_cast<std::int64_t>(before.size());
  star.before_lengths = before.data();
  star.cell_count = static_cast<std::int64_t>(cells.size() / 4);
  star.cells = cells.data();
  star.determinant_floor = kDeterminantFloor;
  CollapseObjective objective(star);
  std::vector<double> lengths(spokes.size());
  double quality = 0.0;
  plan.removed = removed;
  plan.kept = kept;
  plan.boundary = dimension < 3;
  plan.moved = false;
  bool admitted = false;
  if (in_place && objective.evaluate(points.data(), tensors.data(), lengths.data(), quality) ==
                      PositionVerdict::kAdmitted) {
    admitted = true;
    plan.added.clear();
    for (double value : lengths) plan.added.add(value);
    plan.added.finish();
  }
  if (!survivor_moves_ || (admitted && !objective.long_violation()) ||
      mesh_.protected_vertex(kept) || mesh_.protection_radius(kept) > 0.0) {
    return admitted;
  }
  // In place the collapse is refused or leaves a long spoke: the survivor may
  // move on its carrier, replacing both stars. Interior edges always split;
  // a boundary survivor's spokes to boundary vertices may lie on its facets,
  // where splits are exact only, so none of those may be left long.
  const auto chart = carrier_chart(kept, kept_star);
  if (!chart) return admitted;
  const bool interior = chart->simplices.empty();
  for (std::int32_t slot : kept_star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    if (vertex_slot(cell, removed) >= 0) continue;
    for (int corner = 0; corner < 4; ++corner) cells.push_back(row_of(cell.v[corner]));
  }
  star.cell_count = static_cast<std::int64_t>(cells.size() / 4);
  star.cells = cells.data();
  std::vector<SurvivorMove> moves;
  std::vector<double> move_lengths;
  const SearchEnd end = survivor_search(*carriers_, input_.source_metrics.data(),
                                        input_.source_logarithms.data(), star, *chart, false,
                                        moves, move_lengths);
  std::size_t count = moves.size();
  if (count > 0 && (end == SearchEnd::kUnmeasured || end == SearchEnd::kOutsideDomain)) --count;
  for (std::size_t index = 0; index < count; ++index) {
    Violations candidate;
    bool blocked = false;
    for (std::size_t spoke = 0; spoke < spokes.size(); ++spoke) {
      const double value = move_lengths[index * spokes.size() + spoke];
      candidate.add(value);
      blocked = blocked || (!interior && value > kUpperLength &&
                            mesh_.vertex_dimension(rows[static_cast<std::size_t>(spokes[spoke])]) != 3);
    }
    candidate.finish();
    const bool in_band = candidate.long_side.empty();
    if (blocked) continue;
    const auto witness = relocation_witness(kept, moves[index].carrier);
    if (!witness) continue;
    if (admitted && compare_violations(candidate, plan.added) >= 0) continue;
    plan.added = std::move(candidate);
    plan.moved = true;
    plan.carrier = moves[index].carrier;
    plan.witness = *witness;
    plan.metric = moves[index].metric;
    admitted = true;
    if (in_band) break;
  }
  return admitted;
}

std::vector<std::int32_t> Adapter::facet_groups(std::int32_t vertex,
                                                std::span<const std::int32_t> star) const {
  std::vector<std::int32_t> groups;
  for (std::int32_t slot : star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int face = 0; face < 4; ++face) {
      if (cell.v[face] == vertex) continue;
      const std::int32_t label = mesh_.complex().constraint(slot, face);
      if (label != kNoConstraint) groups.push_back(label);
    }
  }
  std::sort(groups.begin(), groups.end());
  groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
  return groups;
}

void Adapter::record_collapse(std::int32_t removed, std::int32_t kept) {
  // Vertex lineage: a removed source vertex collapses into the survivor, and
  // every earlier collapse into the removed vertex now continues into it.
  const std::uint64_t removed_target = vertex_targets_[static_cast<std::size_t>(removed)];
  const std::uint64_t kept_target = vertex_targets_[static_cast<std::size_t>(kept)];
  std::vector<std::uint64_t> moved;
  if (const auto found = absorbed_.find(removed_target); found != absorbed_.end()) {
    moved = std::move(found->second);
    absorbed_.erase(found);
  }
  if (source_targets_.contains(removed_target)) moved.push_back(removed_target);
  for (std::uint64_t source : moved) collapsed_into_[source] = kept_target;
  std::vector<std::uint64_t>& into = absorbed_[kept_target];
  into.insert(into.end(), moved.begin(), moved.end());
  stencils_[static_cast<std::size_t>(removed)].clear();
  vertex_targets_[static_cast<std::size_t>(removed)] = std::numeric_limits<std::uint64_t>::max();
  ++counters_.collapses;
}

bool Adapter::facets_through(std::int32_t vertex, std::span<const std::int32_t> star,
                             std::vector<TetSnapshotFacet>& facets) {
  facets.clear();
  for (std::int32_t slot : star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int face = 0; face < 4; ++face) {
      if (cell.v[face] == vertex || mesh_.complex().constraint(slot, face) == kNoConstraint) continue;
      std::array<std::uint64_t, 3> ids{};
      int count = 0;
      for (int corner = 0; corner < 4; ++corner) {
        if (corner != face) ids[static_cast<std::size_t>(count++)] = mesh_.vertex_id(cell.v[corner]);
      }
      const TetSnapshotFacet* found = nullptr;
      if (incidence_.boundary_facet(ids, found) != TetEditStatus::kApplied || found == nullptr) {
        return false;
      }
      if (std::none_of(facets.begin(), facets.end(), [&](const TetSnapshotFacet& facet) {
            return facet.id == found->id;
          })) {
        facets.push_back(*found);
      }
    }
  }
  std::sort(facets.begin(), facets.end(),
            [](const TetSnapshotFacet& x, const TetSnapshotFacet& y) { return x.id < y.id; });
  return true;
}

bool Adapter::segments_through(std::int32_t vertex, std::span<const std::int32_t> star,
                               std::vector<TetSnapshotSegment>& segments) {
  segments.clear();
  if (mesh_.vertex_dimension(vertex) > 1) return true;
  const std::uint64_t id = mesh_.vertex_id(vertex);
  for (std::int32_t slot : star) {
    for (std::int32_t other : mesh_.tet(slot).v) {
      if (other == vertex || mesh_.vertex_dimension(other) > 1) continue;
      const TetSnapshotSegment* found = nullptr;
      if (incidence_.boundary_segment({id, mesh_.vertex_id(other)}, found) !=
          TetEditStatus::kApplied) return false;
      if (found == nullptr) continue;
      if (std::none_of(segments.begin(), segments.end(), [&](const TetSnapshotSegment& segment) {
            return segment.id == found->id;
          })) {
        segments.push_back(*found);
      }
    }
  }
  std::sort(segments.begin(), segments.end(),
            [](const TetSnapshotSegment& x, const TetSnapshotSegment& y) { return x.id < y.id; });
  return true;
}

std::optional<TetEditStatus> Adapter::collapse_boundary(
    const CollapsePlan& plan, const TetEdit& edit,
    const LocalIncidence::Star& removed_star, const LocalIncidence::Star& kept_star) {
  if (mesh_.implicit_source()) return std::nullopt;
  const LocalIncidence::Star* moved_star = plan.moved ? &kept_star : nullptr;
  if (mesh_.witness(plan.kept).stratum != SourceStratum::kNone) {
    return incidence_.collapse_boundary(mesh_, edit, removed_star, moved_star);
  }
  if (!source_banks_match_) return std::nullopt;
  const Stencil moved = plan.moved ? carrier_stencil(plan.carrier) : Stencil{};
  const Stencil& support = plan.moved ? moved : stencils_[static_cast<std::size_t>(plan.kept)];
  if (support.size() == 1 && support.front().weight == 1.0) {
    return incidence_.collapse_boundary(mesh_, edit, removed_star, moved_star);
  }
  bool exact = true;
  if (plan.moved && input_.exact_source == nullptr) {
    exact = plan.carrier.exact;
  } else {
    const double* point = plan.moved ? plan.carrier.point.data() : mesh_.point(plan.kept);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      Expansion image;
      for (const StencilTerm& term : support) {
        image = image + Expansion::product(
                            term.weight,
                            input_.source_points[3 * static_cast<std::size_t>(term.row) + axis]);
      }
      if (!(image - Expansion(point[axis])).is_zero()) {
        exact = false;
        break;
      }
    }
  }
  // Only the frozen binary64 source bank can prove the local rounded plane.
  if (!exact && (plan.moved || input_.exact_source != nullptr)) return std::nullopt;
  std::vector<std::int32_t> rows;
  std::vector<double> weights;
  rows.reserve(support.size());
  weights.reserve(support.size());
  for (const StencilTerm& term : support) {
    rows.push_back(term.row);
    weights.push_back(term.weight);
  }
  const detail::SourceStencil stencil{
      rows, weights, exact ? detail::SourceStencilImage::kExact
                           : detail::SourceStencilImage::kRoundedPlane};
  return incidence_.collapse_boundary(mesh_, edit, removed_star, moved_star, &stencil);
}

TetEditStatus Adapter::collapse(const CollapsePlan& plan, bool& applied) {
  applied = false;
  const std::int32_t removed = plan.removed;
  const std::int32_t kept = plan.kept;
  const std::uint64_t removed_id = mesh_.vertex_id(removed);
  const std::uint64_t kept_id = mesh_.vertex_id(kept);
  const LocalIncidence::Star removed_star = incidence_.vertex_star(removed);
  const LocalIncidence::Star kept_star = incidence_.vertex_star(kept);
  TetEdit edit;
  edit.kind = TetEditKind::kCollapse;
  edit.collapse_pairs.push_back({removed_id, kept_id});
  edit.removed_vertex_ids.push_back(removed_id);
  // The cavity: the removed vertex's star and, when the survivor moves, the
  // rest of the survivor's star. A replacement keeps its parent's corner
  // order and face labels, with the removed vertex imaged onto the survivor.
  std::vector<std::int32_t> cavity(removed_star.begin(), removed_star.end());
  if (plan.moved) {
    for (std::int32_t slot : kept_star) {
      if (vertex_slot(mesh_.tet(slot), removed) < 0) cavity.push_back(slot);
    }
  }
  // Face labels by their image under removed -> kept, from every cavity
  // cell: a cell holding both ends vanishes, and its labelled face through
  // the survivor becomes a face of the image of its neighbour across the
  // removed vertex.
  std::vector<std::pair<std::array<std::int32_t, 3>, std::int32_t>> labels;
  const auto image_key = [&](const Tetrahedron& cell, std::size_t skip) {
    std::array<std::int32_t, 3> key{};
    std::size_t count = 0;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      if (corner != skip) key[count++] = cell.v[corner] == removed ? kept : cell.v[corner];
    }
    std::sort(key.begin(), key.end());
    return key;
  };
  for (std::int32_t slot : cavity) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (std::size_t face = 0; face < 4; ++face) {
      const std::int32_t label = mesh_.complex().constraint(slot, static_cast<int>(face));
      if (label != kNoConstraint) labels.emplace_back(image_key(cell, face), label);
    }
  }
  std::vector<std::int32_t> parents;  // per replacement: its parent slot
  for (std::int32_t slot : cavity) {
    const Tetrahedron& cell = mesh_.tet(slot);
    edit.removed_cell_ids.push_back(mesh_.cell_id(slot));
    if (vertex_slot(cell, removed) >= 0 && vertex_slot(cell, kept) >= 0) continue;
    if (!fits_ids(std::max(next_cell_id_, next_target_cell_), edit.replacement_cells.size() + 1)) {
      return TetEditStatus::kIndexOverflow;
    }
    TetEditCell replacement;
    replacement.id = next_cell_id_ + edit.replacement_cells.size();
    replacement.region = mesh_.complex().region(slot);
    replacement.ancestor_cell_ids.push_back(mesh_.cell_id(slot));
    for (std::size_t corner = 0; corner < 4; ++corner) {
      const std::int32_t vertex = cell.v[corner];
      replacement.vertices[corner] = vertex == removed ? kept_id : mesh_.vertex_id(vertex);
      const std::array<std::int32_t, 3> key = image_key(cell, corner);
      for (const auto& [face, label] : labels) {
        if (face == key) replacement.facet_sources[corner] = label;
      }
    }
    parents.push_back(slot);
    edit.replacement_cells.push_back(std::move(replacement));
  }
  std::sort(edit.removed_cell_ids.begin(), edit.removed_cell_ids.end());
  if (plan.moved) {
    TetEditVertex survivor;
    survivor.id = kept_id;
    survivor.carrier = plan.carrier.point;
    survivor.witness = plan.witness;
    survivor.edge_witness_supplied = true;
    edit.changed_vertices.push_back(survivor);
  }
  // The removed vertex's facets and segments: those through the survivor
  // go, the others continue as their images.
  std::uint64_t facet_id = next_facet_id_;
  std::uint64_t segment_id = next_segment_id_;
  if (plan.boundary) {
    std::vector<TetSnapshotFacet> facets;
    std::vector<TetSnapshotSegment> segments;
    if (!facets_through(removed, removed_star.cells(), facets) ||
        !segments_through(removed, removed_star.cells(), segments)) {
      return TetEditStatus::kStaleGeneration;
    }
    const auto image = [&](std::uint64_t id) { return id == removed_id ? kept_id : id; };
    for (const TetSnapshotFacet& facet : facets) {
      edit.removed_facet_ids.push_back(facet.id);
      if (std::find(facet.vertices.begin(), facet.vertices.end(), kept_id) != facet.vertices.end()) {
        continue;
      }
      if (!fits_ids(facet_id, 1)) return TetEditStatus::kIndexOverflow;
      TetEditFacet replacement;
      replacement.id = facet_id++;
      replacement.vertices = {image(facet.vertices[0]), image(facet.vertices[1]),
                              image(facet.vertices[2])};
      replacement.source = facet.source;
      replacement.orientation = facet.orientation;
      replacement.protected_facet = facet.protected_facet;
      edit.replacement_facets.push_back(replacement);
    }
    for (const TetSnapshotSegment& segment : segments) {
      edit.removed_segment_ids.push_back(segment.id);
      if (segment.vertices[0] == kept_id || segment.vertices[1] == kept_id) continue;
      if (!fits_ids(segment_id, 1)) return TetEditStatus::kIndexOverflow;
      TetEditSegment replacement;
      replacement.id = segment_id++;
      replacement.vertices = {image(segment.vertices[0]), image(segment.vertices[1])};
      replacement.source = segment.source;
      replacement.protected_segment = segment.protected_segment;
      edit.replacement_segments.push_back(replacement);
    }
  }
  // Lineage of the parents before the commit reuses their slots.
  std::vector<CellLineage> parent_lineage;
  std::vector<std::int32_t> parent_blocks;
  for (std::int32_t slot : parents) {
    parent_lineage.push_back(cell_lineage_[static_cast<std::size_t>(slot)]);
    parent_blocks.push_back(cell_blocks_[static_cast<std::size_t>(slot)]);
  }
  const std::optional<TetEditStatus> local =
      plan.boundary ? collapse_boundary(plan, edit, removed_star, kept_star) : std::nullopt;
  const bool local_cavity =
      !plan.boundary &&
      (!plan.moved || (!mesh_.implicit_source() && mesh_.vertex_dimension(kept) == 3 &&
                       mesh_.witness(kept).stratum == SourceStratum::kNone));
  TetEditStatus status;
  if (local) {
    status = *local;
  } else if (!local_cavity) {
    TetEditBatch batch;
    batch.expected_generation = mesh_.generation();
    batch.identity_bound = mesh_.identity_bound();
    batch.source_id = mesh_.source_id();
    batch.source_revision = mesh_.source_revision();
    batch.source_map_id = mesh_.source_map_id();
    batch.frame_id = mesh_.frame_id();
    batch.periodic_identity = mesh_.periodic_identity();
    batch.operations.push_back(edit);
    std::vector<TetEditResult> results;
    status = mesh_.apply_edit_batch(batch, results);
    if (status == TetEditStatus::kApplied && !incidence_.initialize(mesh_)) {
      return TetEditStatus::kStaleGeneration;
    }
  } else {
    status = incidence_.replace_cavity(mesh_, edit, &removed_star,
                                      plan.moved ? &kept_star : nullptr);
  }
  if (status != TetEditStatus::kApplied) {
    if (fatal(status)) return status;
    count_refusal(status);
    return TetEditStatus::kApplied;
  }
  next_cell_id_ += edit.replacement_cells.size();
  next_facet_id_ = facet_id;
  next_segment_id_ = segment_id;
  for (std::int32_t slot : cavity) retire_cell(slot);
  for (std::size_t index = 0; index < edit.replacement_cells.size(); ++index) {
    const std::int32_t slot = mesh_.cell_slot(edit.replacement_cells[index].id);
    record_created_cell(slot, parent_blocks[index], parent_lineage[index],
                        AdaptLineage::kCoarsenedInto);
    cell_lineage_[static_cast<std::size_t>(slot)].relocated = plan.moved;
  }
  record_collapse(removed, kept);
  if (plan.moved) assign_carrier(kept, carrier_stencil(plan.carrier), plan.metric.data());
  applied = true;
  return TetEditStatus::kApplied;
}

std::int64_t Adapter::collapse_sweep() {
  // Shortest first at the changed vertices; a round's collapses change the
  // stars around their survivors, which the next round looks at.
  std::vector<MeasuredEdge> work;
  std::int64_t total = 0;
  for (;;) {
    dirty_edges(kCollapsePhase, [](double value) { return value < kLowerLength; }, work);
    if (work.empty()) break;
    std::sort(work.begin(), work.end());
    for (const auto& [value, key] : work) {
      const auto a = mesh_.vertex_slot(key[0]);
      const auto b = mesh_.vertex_slot(key[1]);
      if (!mesh_.live_vertex(a) || !mesh_.live_vertex(b)) continue;
      if (edge_length(a, b) != value || incidence_.edge_star(a, b).empty()) continue;
      CollapsePlan best;
      bool found = false;
      for (const auto& [removed, kept] : {std::pair{a, b}, std::pair{b, a}}) {
        CollapsePlan plan;
        const bool planned = plan_collapse(removed, kept, plan);
        if (!planned) continue;
        const int order = found ? compare_violations(plan.added, best.added) : -1;
        if (!found || order < 0 ||
            (order == 0 && vertex_targets_[static_cast<std::size_t>(removed)] >
                               vertex_targets_[static_cast<std::size_t>(best.removed)])) {
          best = std::move(plan);
          found = true;
        }
      }
      if (!found) continue;
      bool done = false;
      const TetEditStatus status = collapse(best, done);
      if (status != TetEditStatus::kApplied) {
        fatal_ = status;
        return total;
      }
      if (done) ++total;
    }
  }
  return total;
}

// ----------------------------------------------------------------- swaps ---

double Adapter::quality(const std::array<std::int32_t, 4>& cell) const {
  std::array<double, 12> corners{};
  std::array<double, 36> tensors{};
  std::array<double, 4> logs{};
  for (std::size_t corner = 0; corner < 4; ++corner) {
    std::copy_n(mesh_.point(cell[corner]), 3, corners.data() + 3 * corner);
    std::copy_n(metric(cell[corner]), 9, tensors.data() + 9 * corner);
    logs[corner] = log_dets_[static_cast<std::size_t>(cell[corner])];
  }
  return tetra_quality(corners.data(), tensors.data(), logs.data());
}

double Adapter::cell_quality(std::int32_t slot) {
  ensure_cell(slot);
  double& cached = cell_quality_[static_cast<std::size_t>(slot)];
  if (std::isnan(cached)) {
    const Tetrahedron& cell = mesh_.tet(slot);
    cached = quality({cell.v[0], cell.v[1], cell.v[2], cell.v[3]});
  }
  return cached;
}

void Adapter::set_metric(std::int32_t vertex, const double* tensor) {
  ensure_vertex(vertex);
  const auto at = static_cast<std::size_t>(vertex);
  std::copy_n(tensor, 9, metrics_.begin() + 9 * static_cast<std::ptrdiff_t>(at));
  log_dets_[at] = log_determinant(tensor);
}

bool Adapter::oriented_quality(const std::array<std::int32_t, 4>& cell, double& value) const {
  std::array<double, 12> corners{};
  for (std::size_t corner = 0; corner < 4; ++corner) {
    std::copy_n(mesh_.point(cell[corner]), 3, corners.data() + 3 * corner);
  }
  double determinant = 0.0;
  double relative = 0.0;
  relative_determinant(corners.data(), determinant, relative);
  if (!(determinant > 0.0) || !(relative > kDeterminantFloor)) return false;
  value = quality(cell);
  return value > 0.0 && std::isfinite(value);
}

TetEditStatus Adapter::replace(TetEditKind kind, const std::vector<std::int32_t>& cavity,
                               const std::vector<std::array<std::int32_t, 4>>& cells,
                               bool& applied) {
  applied = false;
  if (!fits_ids(std::max(next_cell_id_, next_target_cell_), cells.size())) {
    return TetEditStatus::kIndexOverflow;
  }
  // Labels of the cavity's outer faces, kept by the replacement faces on them.
  struct Label {
    std::array<std::int32_t, 3> key;
    std::int32_t value;
  };
  std::vector<Label> labels;
  for (std::int32_t slot : cavity) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int face = 0; face < 4; ++face) {
      const std::int32_t label = mesh_.complex().constraint(slot, face);
      if (label == kNoConstraint) continue;
      std::array<std::int32_t, 3> key{};
      int count = 0;
      for (int corner = 0; corner < 4; ++corner) {
        if (corner != face) key[static_cast<std::size_t>(count++)] = cell.v[corner];
      }
      std::sort(key.begin(), key.end());
      labels.push_back({key, label});
    }
  }
  TetEdit edit;
  edit.kind = kind;
  std::vector<std::uint64_t> ancestors;
  for (std::int32_t slot : cavity) ancestors.push_back(mesh_.cell_id(slot));
  std::sort(ancestors.begin(), ancestors.end());
  edit.removed_cell_ids = ancestors;
  const std::int32_t region = mesh_.complex().region(cavity.front());
  for (const std::array<std::int32_t, 4>& vertices : cells) {
    TetEditCell cell;
    cell.id = next_cell_id_ + edit.replacement_cells.size();
    cell.region = region;
    cell.ancestor_cell_ids = ancestors;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      cell.vertices[corner] = mesh_.vertex_id(vertices[corner]);
      std::array<std::int32_t, 3> key{};
      int count = 0;
      for (std::size_t other = 0; other < 4; ++other) {
        if (other != corner) key[static_cast<std::size_t>(count++)] = vertices[other];
      }
      std::sort(key.begin(), key.end());
      for (const Label& label : labels) {
        if (label.key == key) cell.facet_sources[corner] = label.value;
      }
    }
    edit.replacement_cells.push_back(std::move(cell));
  }
  CellLineage lineage;
  for (std::int32_t slot : cavity) {
    const CellLineage& parent = cell_lineage_[static_cast<std::size_t>(slot)];
    lineage.ancestry.insert(lineage.ancestry.end(), parent.ancestry.begin(), parent.ancestry.end());
  }
  std::sort(lineage.ancestry.begin(), lineage.ancestry.end());
  lineage.ancestry.erase(std::unique(lineage.ancestry.begin(), lineage.ancestry.end()),
                         lineage.ancestry.end());
  const std::int32_t block = cell_blocks_[static_cast<std::size_t>(cavity.front())];
  const TetEditStatus status = incidence_.replace_cavity(mesh_, edit, nullptr);
  if (status != TetEditStatus::kApplied) {
    if (fatal(status)) return status;
    count_refusal(status);
    return TetEditStatus::kApplied;
  }
  next_cell_id_ += edit.replacement_cells.size();
  for (std::int32_t slot : cavity) retire_cell(slot);
  for (const TetEditCell& cell : edit.replacement_cells) {
    record_created_cell(mesh_.cell_slot(cell.id), block, lineage, AdaptLineage::kSwappedFrom);
  }
  applied = true;
  return TetEditStatus::kApplied;
}

constexpr std::size_t kMaxRingCells = 7;

// The shared shell validation supplies either no constrained ends or two
// distinct ends with the same source label. An empty result refuses a fork,
// repeated vertex, or wrong closure before any tiling is proposed.
std::vector<std::int32_t> order_ring_link(
    std::span<const std::array<std::int32_t, 2>> links,
    std::span<const std::pair<std::int32_t, std::int32_t>> through) {
  const bool closed = through.empty();
  const std::size_t target = links.size() + (closed ? 0U : 1U);
  std::vector<std::int32_t> ring;
  ring.reserve(target);
  std::array<bool, kMaxRingCells> used{};
  if (closed) {
    ring.push_back(links.front()[0]);
    ring.push_back(links.front()[1]);
    used[0] = true;
  } else {
    ring.push_back(through.front().second);
  }
  while (ring.size() < target) {
    const auto last = ring.back();
    bool extended = false;
    for (std::size_t index = 0; index < links.size(); ++index) {
      const auto [first, second] = links[index];
      if (used[index] || (first != last && second != last)) continue;
      const auto next = first == last ? second : first;
      if (std::ranges::find(ring, next) != ring.end()) return {};
      ring.push_back(next);
      used[index] = true;
      extended = true;
      break;
    }
    if (!extended) return {};
  }
  if (closed) {
    for (std::size_t index = 0; index < links.size(); ++index) {
      if (used[index]) continue;
      const auto [first, second] = links[index];
      if ((first == ring.back() && second == ring.front()) ||
          (second == ring.back() && first == ring.front())) return ring;
      break;
    }
    return {};
  }
  if (ring.back() != through.back().second) return {};
  return ring;
}

TetEditStatus Adapter::try_ring(std::int32_t s, std::int32_t t, bool& applied,
                                 std::optional<double> shape_reference, bool preserve_band) {
  applied = false;
  if (protected_edge(s, t)) return TetEditStatus::kApplied;
  const std::span<const std::int32_t> view = incidence_.edge_star(s, t);
  std::vector<std::int32_t> shell(view.begin(), view.end());
  const std::size_t n = shell.size();
  if (n < 2 || n > kMaxRingCells) return TetEditStatus::kApplied;
  std::ranges::sort(shell, {}, [&](std::int32_t cell) { return mesh_.cell_id(cell); });
  const std::int32_t region = mesh_.complex().region(shell.front());
  const std::int32_t block = cell_blocks_[static_cast<std::size_t>(shell.front())];
  // The ring: the two other corners of each shell cell, chained. An interior
  // edge's ring is closed; a boundary edge on two coplanar facets (s t c) and
  // (s t d) of one group has an open ring from c to d, which the swap closes
  // with the new edge (c d) and its facets (s c d) and (t c d).
  std::vector<std::array<std::int32_t, 2>> links;
  std::vector<std::pair<std::int32_t, std::int32_t>> through;  // facet label, opposite vertex
  const bool shape = shape_reference.has_value();
  double before_quality = shape ? 0.0 : std::numeric_limits<double>::infinity();
  for (std::int32_t slot : shell) {
    const Tetrahedron& cell = mesh_.tet(slot);
    if (mesh_.complex().region(slot) != region ||
        cell_blocks_[static_cast<std::size_t>(slot)] != block) {
      return TetEditStatus::kApplied;
    }
    std::array<std::int32_t, 2> other{};
    std::array<int, 2> corners{};
    int count = 0;
    for (int corner = 0; corner < 4; ++corner) {
      if (cell.v[corner] == s || cell.v[corner] == t) continue;
      corners[static_cast<std::size_t>(count)] = corner;
      other[static_cast<std::size_t>(count++)] = cell.v[corner];
    }
    for (std::size_t side = 0; side < 2; ++side) {
      const std::int32_t label = mesh_.complex().constraint(slot, corners[side]);
      if (label != kNoConstraint) through.emplace_back(label, other[1 - side]);
    }
    links.push_back(other);
    const double quality = cell_quality(slot);
    before_quality = shape ? before_quality + quality : std::min(before_quality, quality);
  }
  const bool closed = through.empty();
  if (shape && !closed) return TetEditStatus::kApplied;
  if (!closed && (through.size() != 2 || through[0].first != through[1].first ||
                  through[0].second == through[1].second)) {
    return TetEditStatus::kApplied;
  }
  if (closed && n < 3) return TetEditStatus::kApplied;
  const auto ring = order_ring_link(links, through);
  if (ring.empty()) return TetEditStatus::kApplied;
  const std::size_t m = ring.size();

  const double removed_length = edge_length(s, t);
  Violations removed;
  if (shape) {
    if (!std::isfinite(removed_length)) return TetEditStatus::kApplied;
    before_quality = shape_score(before_quality, n, *shape_reference);
  } else {
    removed.add(removed_length);
    removed.finish();
  }
  // Every triangulation of the ring polygon: two cells per triangle.
  struct Option {
    std::vector<std::array<std::int32_t, 4>> cells;
    Violations added;
    double quality = 0.0;
  };
  std::optional<Option> best;
  std::vector<std::array<int, 3>> triangles;
  // The two cells of a ring triangle in one orientation are evaluated once:
  // memo[(i m + k) m + j][choice] holds state (0 unknown, 1 valid, 2 not) and
  // the two qualities.
  struct Evaluated {
    int state = 0;
    double first = 0.0;
    double second = 0.0;
  };
  std::vector<std::array<Evaluated, 2>> memo(m * m * m);
  std::vector<int> diagonal_state(m * m, 0);
  std::vector<double> diagonal_length(m * m, 0.0);
  const auto evaluate = [&]() {
    Option option;
    double quality = shape ? 0.0 : std::numeric_limits<double>::infinity();
    // The triangles share one winding, so one orientation serves them all; a
    // triangle that needs the other is folded and the option is no tiling.
    int winding = 0;
    for (const std::array<int, 3>& triangle : triangles) {
      const std::int32_t a = ring[static_cast<std::size_t>(triangle[0])];
      const std::int32_t b = ring[static_cast<std::size_t>(triangle[1])];
      const std::int32_t c = ring[static_cast<std::size_t>(triangle[2])];
      const std::size_t key =
          (static_cast<std::size_t>(triangle[0]) * m + static_cast<std::size_t>(triangle[1])) * m +
          static_cast<std::size_t>(triangle[2]);
      double first = 0.0;
      double second = 0.0;
      for (int choice : {1, -1}) {
        if (winding != 0 && choice != winding) continue;
        const std::array<std::int32_t, 4> up =
            choice > 0 ? std::array<std::int32_t, 4>{a, b, c, t} : std::array<std::int32_t, 4>{b, a, c, t};
        const std::array<std::int32_t, 4> down =
            choice > 0 ? std::array<std::int32_t, 4>{b, a, c, s} : std::array<std::int32_t, 4>{a, b, c, s};
        Evaluated& entry = memo[key][choice > 0 ? 0 : 1];
        if (entry.state == 0) {
          entry.state = oriented_quality(up, entry.first) && oriented_quality(down, entry.second) ? 1 : 2;
        }
        if (entry.state != 1) continue;
        first = entry.first;
        second = entry.second;
        winding = choice;
        option.cells.push_back(up);
        option.cells.push_back(down);
        break;
      }
      if (option.cells.size() != 2 * static_cast<std::size_t>(&triangle - triangles.data() + 1)) {
        return;
      }
      quality = shape ? (quality + first) + second : std::min({quality, first, second});
    }
    // The diagonals: ring index pairs of the triangles that are not ring
    // sides (an open ring's ends are joined by a new edge), each checked (new
    // and unprotected) and measured once per ring.
    std::vector<std::array<int, 2>> diagonals;
    for (const std::array<int, 3>& triangle : triangles) {
      for (int side = 0; side < 3; ++side) {
        const int i = triangle[static_cast<std::size_t>(side)];
        const int j = triangle[static_cast<std::size_t>((side + 1) % 3)];
        const int gap = std::abs(i - j);
        if (gap == 1 || (closed && gap == static_cast<int>(m) - 1)) continue;
        diagonals.push_back({std::min(i, j), std::max(i, j)});
      }
    }
    std::sort(diagonals.begin(), diagonals.end());
    diagonals.erase(std::unique(diagonals.begin(), diagonals.end()), diagonals.end());
    for (const auto& [i, j] : diagonals) {
      const std::size_t key = static_cast<std::size_t>(i) * m + static_cast<std::size_t>(j);
      if (diagonal_state[key] == 0) {
        const std::int32_t x = ring[static_cast<std::size_t>(i)];
        const std::int32_t y = ring[static_cast<std::size_t>(j)];
        if (!protected_edge(x, y) && incidence_.edge_star(x, y).empty()) {
          const double value = edge_length(x, y);
          diagonal_length[key] = value;
          const bool eligible = !(shape || preserve_band) ||
                                (value >= shape_lower() && value <= kUpperLength);
          diagonal_state[key] = eligible ? 1 : 2;
        } else {
          diagonal_state[key] = 2;
        }
      }
      if (diagonal_state[key] != 1) return;
      if (!shape) option.added.add(diagonal_length[key]);
    }
    if (!option.added.measured) return;
    if (!shape) option.added.finish();
    option.quality = shape ? shape_score(quality, option.cells.size(), *shape_reference) : quality;
    const int band_order = best && !shape ? compare_violations(option.added, best->added) : 0;
    if (!best || band_order < 0 || (band_order == 0 && option.quality > best->quality)) {
      best = std::move(option);
    }
  };
  // Triangulations of polygon [i, j] with base (i, j), recursively.
  const std::function<void(int, int, const std::function<void()>&)> triangulate =
      [&](int i, int j, const std::function<void()>& next) {
        if (j - i < 2) {
          next();
          return;
        }
        for (int k = i + 1; k < j; ++k) {
          triangles.push_back({i, k, j});
          triangulate(i, k, [&]() { triangulate(k, j, next); });
          triangles.pop_back();
        }
      };
  triangulate(0, static_cast<int>(m) - 1, evaluate);
  if (!best || !removed.measured) return TetEditStatus::kApplied;
  const bool gain = best->quality > before_quality + kImprovement;
  if (!(shape ? gain : objective_decreases(removed, best->added, gain))) {
    return TetEditStatus::kApplied;
  }
  if (!closed) {
    return swap_boundary(shell, best->cells, s, t, ring.front(), ring.back(), applied);
  }
  const TetEditKind kind = n == 3 ? TetEditKind::kFlip32 : TetEditKind::kRingReconnect;
  const TetEditStatus status = replace(kind, shell, best->cells, applied);
  if (applied) ++(n == 3 ? counters_.flips32 : counters_.ring_reconnections);
  return status;
}

TetEditStatus Adapter::try_flip23(std::int32_t slot, int face, bool& applied,
                                   std::optional<double> shape_reference, bool preserve_band) {
  applied = false;
  if (!mesh_.complex().live(slot) || mesh_.complex().constraint(slot, face) != kNoConstraint) {
    return TetEditStatus::kApplied;
  }
  const Tetrahedron& first = mesh_.tet(slot);
  const std::int32_t other = first.n[face];
  if (!mesh_.complex().live(other) || is_ghost(mesh_.tet(other))) return TetEditStatus::kApplied;
  const Tetrahedron& second = mesh_.tet(other);
  if (mesh_.complex().region(slot) != mesh_.complex().region(other) ||
      cell_blocks_[static_cast<std::size_t>(slot)] != cell_blocks_[static_cast<std::size_t>(other)]) {
    return TetEditStatus::kApplied;
  }
  const std::int32_t s = first.v[face];
  const int back = neighbor_slot(second, slot);
  if (back < 0) return TetEditStatus::kApplied;
  const std::int32_t t = second.v[back];
  if (protected_edge(s, t) || !incidence_.edge_star(s, t).empty()) return TetEditStatus::kApplied;
  // The face, oriented so that s is on its positive side.
  std::int32_t facet[3];
  oriented_facet(first.v, face, facet);
  const bool shape = shape_reference.has_value();
  const double before_quality =
      shape ? shape_score(cell_quality(slot) + cell_quality(other), 2, *shape_reference)
            : std::min(cell_quality(slot), cell_quality(other));
  std::vector<std::array<std::int32_t, 4>> cells;
  double after_quality = std::numeric_limits<double>::infinity();
  for (int choice : {1, -1}) {
    cells.clear();
    after_quality = shape ? 0.0 : std::numeric_limits<double>::infinity();
    for (int side = 0; side < 3; ++side) {
      std::array<std::int32_t, 4> cell{facet[side], facet[(side + 1) % 3], t, s};
      if (choice < 0) std::swap(cell[0], cell[1]);
      double value = 0.0;
      if (!oriented_quality(cell, value)) break;
      after_quality = shape ? after_quality + value : std::min(after_quality, value);
      cells.push_back(cell);
    }
    if (cells.size() == 3) break;
  }
  if (cells.size() != 3) return TetEditStatus::kApplied;
  const double added_length = edge_length(s, t);
  if (preserve_band && !(added_length >= shape_lower() && added_length <= kUpperLength)) {
    return TetEditStatus::kApplied;
  }
  if (shape) {
    if (!(added_length >= shape_lower() && added_length <= kUpperLength) ||
        !(shape_score(after_quality, cells.size(), *shape_reference) >
          before_quality + kImprovement)) {
      return TetEditStatus::kApplied;
    }
  } else {
    Violations removed;
    Violations added;
    added.add(added_length);
    if (!added.measured) return TetEditStatus::kApplied;
    added.finish();
    if (!objective_decreases(removed, added, after_quality > before_quality + kImprovement)) {
      return TetEditStatus::kApplied;
    }
  }
  const std::vector<std::int32_t> cavity{slot, other};
  const TetEditStatus status = replace(TetEditKind::kFlip23, cavity, cells, applied);
  if (applied) ++counters_.flips23;
  return status;
}

std::int64_t Adapter::swap_sweep() {
  // Candidates at the changed vertices: the faces and edges of poor cells and
  // every out-of-band edge.
  constexpr double kPoor = 0.4;
  std::vector<std::pair<std::uint64_t, int>> faces;
  std::vector<IdEdge> edges;
  std::vector<std::int32_t> star;
  for (std::int32_t vertex : dirty_[kSwapPhase]) {
    dirty_flags_[kSwapPhase][static_cast<std::size_t>(vertex)] = 0;
    if (!mesh_.live_vertex(vertex)) continue;
    incidence_.collect_star(vertex, star);
    for (std::int32_t slot : star) {
      const Tetrahedron& cell = mesh_.tet(slot);
      const bool poor = cell_quality(slot) < kPoor;
      for (int first = 0; first < 4; ++first) {
        if (poor) faces.emplace_back(mesh_.cell_id(slot), first);
        for (int second = first + 1; second < 4; ++second) {
          const std::int32_t a = cell.v[first];
          const std::int32_t b = cell.v[second];
          if (poor) {
            edges.push_back(edge_ids(a, b));
            continue;
          }
          if (a != vertex && b != vertex) continue;
          const double value = edge_length(a, b);
          if (!(value >= kLowerLength && value <= kUpperLength)) edges.push_back(edge_ids(a, b));
        }
      }
    }
  }
  dirty_[kSwapPhase].clear();
  return commit_swaps(faces, edges, std::nullopt);
}

std::int64_t Adapter::shape_sweeps(std::int64_t maximum_sweeps) {
  // The ordinary frontier contains only residual edges and poor cells.
  // A full closed-cavity sweep can improve their neighbors too. Fixing the
  // reference and vertex bank gives a finite improving topology search.
  const auto& cells = mesh_.complex().tets;
  std::size_t count = 0;
  double total = 0.0;
  for (std::size_t row = 0; row < cells.size(); ++row) {
    const auto slot = static_cast<std::int32_t>(row);
    if (!mesh_.complex().live(slot) || is_ghost(cells[row])) continue;
    total += cell_quality(slot);
    ++count;
  }
  if (count == 0 || !std::isfinite(total)) return 0;
  const double reference = total / static_cast<double>(count);
  std::vector<std::pair<std::uint64_t, int>> faces;
  std::vector<IdEdge> edges;
  std::int64_t sweeps = 0;
  for (; sweeps < maximum_sweeps; ++sweeps) {
    faces.clear();
    edges.clear();
    for (std::size_t row = 0; row < cells.size(); ++row) {
      const auto slot = static_cast<std::int32_t>(row);
      const Tetrahedron& cell = cells[row];
      if (!mesh_.complex().live(slot) || is_ghost(cell)) continue;
      const auto id = mesh_.cell_id(slot);
      for (int first = 0; first < 4; ++first) {
        const auto other = cell.n[first];
        if (mesh_.complex().live(other) && !is_ghost(mesh_.tet(other)) &&
            id < mesh_.cell_id(other) &&
            mesh_.complex().constraint(slot, first) == kNoConstraint) {
          faces.emplace_back(id, first);
        }
        for (int second = first + 1; second < 4; ++second) {
          edges.push_back(edge_ids(cell.v[first], cell.v[second]));
        }
      }
    }
    const auto applied = commit_swaps(faces, edges, reference);
    if (fatal_ != TetEditStatus::kApplied || applied == 0) return sweeps;
  }
  return sweeps;
}

std::int64_t Adapter::commit_swaps(std::vector<std::pair<std::uint64_t, int>>& faces,
                                   std::vector<IdEdge>& edges,
                                   std::optional<double> shape_reference, bool preserve_band) {
  std::sort(faces.begin(), faces.end());
  faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::int64_t applied = 0;
  for (const auto& [id, face] : faces) {
    const auto slot = mesh_.cell_slot(id);
    if (slot < 0) continue;
    bool done = false;
    const TetEditStatus status = try_flip23(slot, face, done, shape_reference, preserve_band);
    if (status != TetEditStatus::kApplied) {
      fatal_ = status;
      return applied;
    }
    applied += done ? 1 : 0;
  }
  for (const IdEdge& key : edges) {
    const auto a = mesh_.vertex_slot(key[0]);
    const auto b = mesh_.vertex_slot(key[1]);
    if (!mesh_.live_vertex(a) || !mesh_.live_vertex(b)) continue;
    bool done = false;
    const TetEditStatus status = try_ring(a, b, done, shape_reference, preserve_band);
    if (status != TetEditStatus::kApplied) {
      fatal_ = status;
      return applied;
    }
    applied += done ? 1 : 0;
  }
  return applied;
}

TetEditStatus Adapter::swap_boundary(const std::vector<std::int32_t>& shell,
                                     const std::vector<std::array<std::int32_t, 4>>& cells,
                                     std::int32_t s, std::int32_t t, std::int32_t c,
                                     std::int32_t d, bool& applied) {
  applied = false;
  const std::uint64_t sid = mesh_.vertex_id(s);
  const std::uint64_t tid = mesh_.vertex_id(t);
  const TetSnapshotFacet* first = nullptr;
  const TetSnapshotFacet* second = nullptr;
  if (incidence_.boundary_facet({sid, tid, mesh_.vertex_id(c)}, first) != TetEditStatus::kApplied ||
      incidence_.boundary_facet({sid, tid, mesh_.vertex_id(d)}, second) != TetEditStatus::kApplied ||
      first == nullptr || second == nullptr) {
    return TetEditStatus::kStaleGeneration;
  }
  const TetSnapshotFacet& reference = *first;
  const TetSnapshotFacet& other = *second;
  if (reference.protected_facet || other.protected_facet || reference.source != other.source ||
      reference.orientation != other.orientation) {
    ++counters_.rejected_protected;
    return TetEditStatus::kApplied;
  }
  // The new facets keep the old ones' oriented normal: the side of an
  // off-plane cell corner decides each one's vertex order exactly.
  const double* corners[3];
  for (std::size_t corner = 0; corner < 3; ++corner) {
    corners[corner] = mesh_.point(mesh_.vertex_slot(reference.vertices[corner]));
  }
  std::int32_t witness = -1;
  int side = 0;
  for (const std::array<std::int32_t, 4>& cell : cells) {
    for (std::int32_t vertex : cell) {
      if (side != 0) break;
      side = orient3d(corners[0], corners[1], corners[2], mesh_.point(vertex));
      if (side != 0) witness = vertex;
    }
  }
  if (side == 0) {
    ++counters_.rejected_invalid;
    return TetEditStatus::kApplied;
  }
  if (!fits_ids(std::max(next_cell_id_, next_target_cell_), cells.size()) ||
      !fits_ids(next_facet_id_, 2)) {
    return TetEditStatus::kIndexOverflow;
  }
  TetEdit edit;
  edit.kind = TetEditKind::kRingReconnect;
  edit.removed_facet_ids = {std::min(reference.id, other.id), std::max(reference.id, other.id)};
  std::array<std::array<std::int32_t, 3>, 2> images{};
  for (std::size_t apex = 0; apex < 2; ++apex) {
    std::array<std::int32_t, 3> image{apex == 0 ? s : t, c, d};
    const int turn = orient3d(mesh_.point(image[0]), mesh_.point(image[1]), mesh_.point(image[2]),
                              mesh_.point(witness));
    if (turn == 0) {
      ++counters_.rejected_invalid;
      return TetEditStatus::kApplied;
    }
    if (turn != side) std::swap(image[1], image[2]);
    images[apex] = image;
    TetEditFacet facet;
    facet.id = next_facet_id_ + apex;
    for (std::size_t corner = 0; corner < 3; ++corner) facet.vertices[corner] = mesh_.vertex_id(image[corner]);
    facet.source = reference.source;
    facet.orientation = reference.orientation;
    edit.replacement_facets.push_back(facet);
  }
  // Outer face labels of the shell, and the new facets' group.
  std::vector<std::pair<std::array<std::int32_t, 3>, std::int32_t>> labels;
  for (std::int32_t slot : shell) {
    const Tetrahedron& cell = mesh_.tet(slot);
    edit.removed_cell_ids.push_back(mesh_.cell_id(slot));
    for (int face = 0; face < 4; ++face) {
      const std::int32_t label = mesh_.complex().constraint(slot, face);
      if (label == kNoConstraint) continue;
      std::array<std::int32_t, 3> key{};
      int count = 0;
      for (int corner = 0; corner < 4; ++corner) {
        if (corner != face) key[static_cast<std::size_t>(count++)] = cell.v[corner];
      }
      std::sort(key.begin(), key.end());
      labels.emplace_back(key, label);
    }
  }
  for (std::array<std::int32_t, 3> image : images) {
    std::sort(image.begin(), image.end());
    labels.emplace_back(image, reference.source);
  }
  std::sort(edit.removed_cell_ids.begin(), edit.removed_cell_ids.end());
  const std::int32_t region = mesh_.complex().region(shell.front());
  for (const std::array<std::int32_t, 4>& vertices : cells) {
    TetEditCell cell;
    cell.id = next_cell_id_ + edit.replacement_cells.size();
    cell.region = region;
    cell.ancestor_cell_ids = edit.removed_cell_ids;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      cell.vertices[corner] = mesh_.vertex_id(vertices[corner]);
      std::array<std::int32_t, 3> key{};
      int count = 0;
      for (std::size_t other_corner = 0; other_corner < 4; ++other_corner) {
        if (other_corner != corner) key[static_cast<std::size_t>(count++)] = vertices[other_corner];
      }
      std::sort(key.begin(), key.end());
      for (const auto& [face, label] : labels) {
        if (face == key) cell.facet_sources[corner] = label;
      }
    }
    edit.replacement_cells.push_back(std::move(cell));
  }
  // The lineage the cells inherit, before the commit reuses the slots.
  CellLineage lineage;
  for (std::int32_t slot : shell) {
    const CellLineage& parent = cell_lineage_[static_cast<std::size_t>(slot)];
    lineage.ancestry.insert(lineage.ancestry.end(), parent.ancestry.begin(), parent.ancestry.end());
  }
  std::sort(lineage.ancestry.begin(), lineage.ancestry.end());
  lineage.ancestry.erase(std::unique(lineage.ancestry.begin(), lineage.ancestry.end()),
                         lineage.ancestry.end());
  const std::int32_t block = cell_blocks_[static_cast<std::size_t>(shell.front())];
  const TetEditStatus status = incidence_.swap_boundary_edge(mesh_, edit);
  if (status != TetEditStatus::kApplied) {
    if (fatal(status)) return status;
    count_refusal(status);
    return TetEditStatus::kApplied;
  }
  next_cell_id_ += edit.replacement_cells.size();
  next_facet_id_ += edit.replacement_facets.size();
  for (std::int32_t slot : shell) retire_cell(slot);
  for (const TetEditCell& cell : edit.replacement_cells) {
    record_created_cell(mesh_.cell_slot(cell.id), block, lineage, AdaptLineage::kSwappedFrom);
  }
  ++counters_.ring_reconnections;
  applied = true;
  return TetEditStatus::kApplied;
}

// ------------------------------------------------------------ relocation ---

CarrierChart prepare_seam(const CarrierSource& source, const CarrierChart& first,
                          const CarrierChart& second) {
  const auto sides = [](const CarrierChart& chart) {
    std::vector<std::array<std::int64_t, 2>> edges;
    edges.reserve(3 * chart.simplices.size());
    for (const CarrierSimplex& triangle : chart.simplices) {
      for (int corner = 0; corner < 3; ++corner) {
        const std::int64_t a = triangle.rows[static_cast<std::size_t>(corner)];
        const std::int64_t b = triangle.rows[static_cast<std::size_t>((corner + 1) % 3)];
        edges.push_back({std::min(a, b), std::max(a, b)});
      }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
  };
  const auto left = sides(first);
  const auto right = sides(second);
  CarrierChart seam;
  auto a = left.begin();
  auto b = right.begin();
  // Lexicographic source-edge intersection is also the exact search order.
  while (a != left.end() && b != right.end()) {
    if (*a < *b) {
      ++a;
    } else if (*b < *a) {
      ++b;
    } else {
      seam.simplices.push_back({{(*a)[0], (*a)[1], -1}, 2});
      ++a;
      ++b;
    }
  }
  source.prepare_chart(seam);
  return seam;
}

bool Adapter::source_seam_chart(const SourceWitness& witness, CarrierChartView chart) const {
  if (witness.stratum != SourceStratum::kSegment || witness.entity < 0 ||
      static_cast<std::size_t>(witness.entity) >= source_only_segments_.size()) return false;
  const auto& ends = source_only_segments_[static_cast<std::size_t>(witness.entity)];
  if (ends[0] < 0) return false;
  const auto found = source_only_segment_rows_.find(edge_key(ends[0], ends[1]));
  if (found == source_only_segment_rows_.end() || found->second != witness.entity) return false;
  const std::array<std::int64_t, 3> key{std::min(ends[0], ends[1]), std::max(ends[0], ends[1]), -1};
  const auto simplex = std::ranges::lower_bound(chart.simplices, key, {}, &CarrierSimplex::rows);
  return simplex != chart.simplices.end() && simplex->rows == key;
}

// An interior vertex has an empty view. A boundary vertex uses its one
// source group or the shared source edges of its two groups, not live facets.
std::optional<CarrierChartView> Adapter::carrier_chart(
    std::int32_t vertex, const LocalIncidence::Star& star, CarrierUse use) {
  const int dimension = mesh_.vertex_dimension(vertex);
  if (dimension == 3) return CarrierChartView{};
  const bool ridge = (use == CarrierUse::kShapeCage || use == CarrierUse::kBandCage) &&
                     dimension == 1;
  if ((!ridge && dimension != 2) || !source_banks_match_) return std::nullopt;
  bool exterior_rounding = use != CarrierUse::kCollapse && rounded_sources_;
  std::int32_t first_group = kNoConstraint;
  std::int32_t second_group = kNoConstraint;
  for (std::int32_t slot : star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int face = 0; face < 4; ++face) {
      if (cell.v[face] == vertex) continue;
      const std::int32_t group = mesh_.complex().constraint(slot, face);
      if (group == kNoConstraint) continue;
      exterior_rounding = exterior_rounding && is_ghost(mesh_.tet(cell.n[face]));
      if (group == first_group || group == second_group) continue;
      if (first_group == kNoConstraint) {
        first_group = group;
      } else if (second_group == kNoConstraint) {
        second_group = group;
      } else {
        return std::nullopt;
      }
    }
  }
  const auto first = group_triangles_.find(first_group);
  if (first == group_triangles_.end()) return std::nullopt;
  const SourceWitness& witness = mesh_.witness(vertex);
  if (second_group == kNoConstraint) {
    if (ridge) return std::nullopt;
    CarrierChartView chart = first->second.view();
    chart.allow_rounded_images = exterior_rounding && witness.stratum == SourceStratum::kNone &&
                                 exact_plane_axes_.contains(first_group);
    return chart;
  }
  const auto second = group_triangles_.find(second_group);
  if (second == group_triangles_.end()) return std::nullopt;

  // Group pairs refer to the immutable source bank even as the live star changes.
  const std::uint64_t pair = edge_key(first_group, second_group);
  auto cached = group_seams_.find(pair);
  if (cached == group_seams_.end()) {
    cached = group_seams_.emplace(pair, prepare_seam(*carriers_, first->second, second->second)).first;
  }
  const CarrierChart& seam = cached->second;
  if (seam.simplices.empty()) return std::nullopt;
  CarrierChartView chart = seam.view();
  bool cartesian = false;
  if (exterior_rounding) {
    const auto a = exact_plane_axes_.find(first_group);
    const auto b = exact_plane_axes_.find(second_group);
    cartesian = a != exact_plane_axes_.end() && b != exact_plane_axes_.end() &&
                a->second != b->second;
  }
  const bool witnessed = (ridge || cartesian) && source_seam_chart(witness, chart);
  // The shared chart of two exact exterior planes fixes both transverse
  // coordinates. On this derived seam, rounding the varying coordinate
  // retains exact source support even without an authored segment witness.
  const bool derived = use == CarrierUse::kBandCage && cartesian &&
                       witness.stratum == SourceStratum::kNone;
  if (ridge && !witnessed && !derived) return std::nullopt;
  chart.allow_rounded_images = cartesian && (witnessed || derived);
  return chart;
}

std::optional<SourceWitness> Adapter::seam_witness(const SourceWitness& old, const Carrier& carrier,
                                                   CarrierChartView chart) const {
  const auto bound = [&](std::int32_t row) -> SourceWitness {
    const auto& ends = source_only_segments_[static_cast<std::size_t>(row)];
    SourceWitness witness{SourceStratum::kSegment, row};
    for (int term = 0; term < carrier.size; ++term) {
      const auto slot = static_cast<std::size_t>(term);
      if (carrier.rows[slot] == ends[1]) {
        witness.parameters[0] = std::ldexp(static_cast<double>(carrier.weights[slot]), -carrier.bits);
      }
    }
    return witness;
  };
  const auto find = [&](std::int64_t a, std::int64_t b) -> std::optional<SourceWitness> {
    const auto found = source_only_segment_rows_.find(
        edge_key(static_cast<std::int32_t>(a), static_cast<std::int32_t>(b)));
    if (found == source_only_segment_rows_.end() || found->second < 0) return std::nullopt;
    return bound(found->second);
  };
  if (carrier.size == 2) return find(carrier.rows[0], carrier.rows[1]);
  if (carrier.size != 1) return std::nullopt;
  const auto point = carrier.rows[0];
  const auto& old_ends = source_only_segments_[static_cast<std::size_t>(old.entity)];
  if (point == old_ends[0] || point == old_ends[1]) return bound(old.entity);
  // At a source knot, keep the old primitive if possible; otherwise the
  // first source-only primitive in the same canonical chart owns the point.
  for (const CarrierSimplex& edge : chart.simplices) {
    if (edge.rows[0] != point && edge.rows[1] != point) continue;
    if (auto witness = find(edge.rows[0], edge.rows[1])) return witness;
  }
  return std::nullopt;
}

std::optional<SourceWitness> Adapter::relocation_witness(std::int32_t vertex,
                                                          const Carrier& carrier,
                                                          CarrierChartView chart) const {
  const SourceWitness& old = mesh_.witness(vertex);
  if (old.stratum == SourceStratum::kNone) return SourceWitness{};
  if (old.stratum == SourceStratum::kSegment && chart.allow_rounded_images) {
    return seam_witness(old, carrier, chart);
  }
  if (!carrier.exact || old.entity < 0) return std::nullopt;
  SourceWitness next{old.stratum, old.entity};
  const auto matches = [&](const Expansion* image) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (!(image[axis] - Expansion(carrier.point[axis])).is_zero()) return false;
    }
    return true;
  };
  Expansion image[3];
  if (old.stratum == SourceStratum::kPlane) {
    const double* plane = mesh_.face_plane(old.entity);
    if (plane == nullptr) return std::nullopt;
    const double zero[3]{};
    const double* const axes[3]{zero, plane + 3, plane + 6};
    double offset[3];
    for (std::size_t axis = 0; axis < 3; ++axis) offset[axis] = carrier.point[axis] - plane[axis];
    source_locator(axes, SourceStratum::kFacet, offset, next.parameters, false);
    if (std::isfinite(next.parameters[0]) && std::isfinite(next.parameters[1]) &&
        mesh_.plane_point(old.entity, next.parameters, image) && matches(image)) return next;
    return std::nullopt;
  }
  if (old.stratum == SourceStratum::kSegment) {
    const double* corners[3]{};
    if (!mesh_.source_corners(old.stratum, old.entity, corners)) return std::nullopt;
    source_locator(corners, old.stratum, carrier.point.data(), next.parameters, false);
    if (source_point(corners, old.stratum, next.parameters, image) && matches(image)) return next;
    return std::nullopt;
  }
  const auto old_row = static_cast<std::size_t>(old.entity);
  if (old_row >= witness_face_groups_.size()) return std::nullopt;
  const auto in_face = [&](std::size_t row) {
    std::array<std::int64_t, 3> weights{};
    for (int term = 0; term < carrier.size; ++term) {
      const auto slot = static_cast<std::size_t>(term);
      const auto first = witness_faces_.begin() + static_cast<std::ptrdiff_t>(3 * row);
      const auto found = std::find(first, first + 3, carrier.rows[slot]);
      if (found == first + 3) return false;
      weights[static_cast<std::size_t>(found - first)] = carrier.weights[slot];
    }
    next.entity = static_cast<std::int32_t>(row);
    next.parameters[0] = std::ldexp(static_cast<double>(weights[1]), -carrier.bits);
    next.parameters[1] = std::ldexp(static_cast<double>(weights[2]), -carrier.bits);
    if (!source_parameters_valid(SourceStratum::kFacet, next.parameters)) return false;
    if (input_.exact_source == nullptr) return true;
    const double* corners[3]{};
    return mesh_.source_corners(next.stratum, next.entity, corners) &&
           source_point(corners, next.stratum, next.parameters, image) && matches(image);
  };
  // The dyadic carrier already realizes these exact barycentric weights.
  // Prefer the existing source triangle, then any triangle in its group.
  if (in_face(old_row)) return next;
  for (std::size_t row = 0; row < witness_face_groups_.size(); ++row) {
    if (witness_face_groups_[row] == witness_face_groups_[old_row] && in_face(row)) return next;
  }
  return std::nullopt;
}

Adapter::RelocationAttempt& Adapter::bind_relocation_attempt(
    std::int32_t vertex, const LocalIncidence::Star& star, CarrierChartView chart) {
  if (relocation_attempts_.empty()) {
    relocation_attempts_.resize(std::min(mesh_.vertex_slot_count(), kRelocationCacheSlots));
  }
  RelocationAttempt& attempt =
      relocation_attempts_[static_cast<std::size_t>(vertex) % relocation_attempts_.size()];
  // In an unchanged cell set every changed point or tensor has a revision
  // newer than the prior maximum. Cell order also binds the spoke order.
  std::uint64_t revision = 0;
  for (std::int32_t slot : star) {
    for (std::int32_t row : mesh_.tet(slot).v) {
      revision = std::max(revision, carrier_revisions_[static_cast<std::size_t>(row)]);
    }
  }
  const std::uint64_t id = mesh_.vertex_id(vertex);
  if (attempt.vertex_id == id && attempt.carrier_revision == revision &&
      attempt.chart.data() == chart.simplices.data() &&
      attempt.chart.size() == chart.simplices.size() &&
      std::ranges::equal(star.cells(), attempt.cells, {},
                         [&](std::int32_t slot) { return mesh_.cell_id(slot); })) {
    return attempt;
  }
  attempt.vertex_id = id;
  attempt.carrier_revision = revision;
  attempt.chart = chart.simplices;
  attempt.exhausted_modes = 0;
  attempt.cells.resize(star.size());
  std::ranges::transform(star.cells(), attempt.cells.begin(),
                         [&](std::int32_t slot) { return mesh_.cell_id(slot); });
  return attempt;
}

bool Adapter::wall_ring_pull(std::int32_t vertex, const LocalIncidence::Star& star,
                             double* pull) const {
  const double* origin = mesh_.point(vertex);
  std::fill_n(pull, 3, 0.0);
  std::size_t faces = 0;
  for (const auto slot : star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int face = 0; face < 4; ++face) {
      if (cell.v[face] == vertex || mesh_.complex().constraint(slot, face) == kNoConstraint ||
          !is_ghost(mesh_.tet(cell.n[face]))) continue;
      ++faces;
      for (int corner = 0; corner < 4; ++corner) {
        if (corner == face || cell.v[corner] == vertex) continue;
        const double* p = mesh_.point(cell.v[corner]);
        for (int axis = 0; axis < 3; ++axis) pull[axis] += p[axis] - origin[axis];
      }
    }
  }
  if (faces < 3) return false;
  // Each neighbor of a closed planar fan appears in two exterior triangles.
  const double weight = 0.5 / static_cast<double>(faces);
  for (int axis = 0; axis < 3; ++axis) {
    pull[axis] *= weight;
    if (!std::isfinite(pull[axis])) return false;
  }
  return pull[0] != 0.0 || pull[1] != 0.0 || pull[2] != 0.0;
}

bool Adapter::clearance_pull(std::int32_t vertex, std::int32_t cell, double* pull) const {
  // A low-volume tetrahedron has its steepest signed height increase along
  // the opposite face normal. The search checks shape and every spoke again.
  std::array<const double*, 3> face{};
  std::size_t count = 0;
  for (std::int32_t v : mesh_.tet(cell).v) {
    if (v != vertex) face[count++] = mesh_.point(v);
  }
  const double* point = mesh_.point(vertex);
  double e[3], f[3];
  for (int axis = 0; axis < 3; ++axis) {
    e[axis] = face[1][axis] - face[0][axis];
    f[axis] = face[2][axis] - face[0][axis];
  }
  double n[3], normal2 = 0.0, separation = 0.0, base2 = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const int i = (axis + 1) % 3, j = (axis + 2) % 3;
    n[axis] = e[i] * f[j] - e[j] * f[i];
    normal2 += n[axis] * n[axis];
    separation += n[axis] * (point[axis] - face[0][axis]);
    base2 += e[axis] * e[axis] + f[axis] * f[axis];
  }
  if (!(normal2 > 0.0) || !std::isfinite(normal2) || separation == 0.0) return false;
  const double scale = std::copysign(0.25 * std::sqrt(base2 / normal2), separation);
  for (int axis = 0; axis < 3; ++axis) pull[axis] = scale * n[axis];
  return std::isfinite(scale);
}

TetEditStatus Adapter::relocate(std::int32_t vertex, bool& applied, ShapeGoal goal) {
  const bool closed_band_shape = goal != ShapeGoal::kBand;
  const bool mean_shape = goal == ShapeGoal::kMean;
  applied = false;
  if (!carriers_ || mesh_.protected_vertex(vertex) || mesh_.protection_radius(vertex) > 0.0) {
    return TetEditStatus::kApplied;
  }
  if (mean_shape && mesh_.vertex_dimension(vertex) != 3) return TetEditStatus::kApplied;
  const LocalIncidence::Star star = incidence_.vertex_star(vertex);
  if (star.empty()) return TetEditStatus::kApplied;
  const auto chart = carrier_chart(vertex, star, CarrierUse::kRelocation);
  if (!chart) return TetEditStatus::kApplied;
  const bool interior = chart->simplices.empty();
  const bool tangents = !closed_band_shape && relocation_stage_ == 5;
  if (tangents && interior) return TetEditStatus::kApplied;
  if (!interior && goal == ShapeGoal::kBand && !boundary_moves_) {
    return TetEditStatus::kApplied;
  }
  const bool deep_stage = !closed_band_shape && (relocation_stage_ == 1 || relocation_stage_ == 2);
  RelocationAttempt* attempt = nullptr;
  unsigned mode_bit = 0;
  if (deep_stage && !mesh_.implicit_source() && star.size() <= kRelocationCacheStarCells) {
    attempt = &bind_relocation_attempt(vertex, star, *chart);
    mode_bit = 1U << relocation_stage_;
    if ((attempt->exhausted_modes & mode_bit) != 0) return TetEditStatus::kApplied;
  }
  // Star rows: the vertex, then its link in first-seen order.
  std::vector<std::int32_t> rows{vertex};
  std::vector<std::int64_t> cells;
  double before_quality = mean_shape ? 0.0 : std::numeric_limits<double>::infinity();
  std::int32_t worst = -1;
  for (std::int32_t slot : star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    const double value = cell_quality(slot);
    if (mean_shape) {
      before_quality += value;
    } else if (value < before_quality) {
      before_quality = value;
      worst = slot;
    }
    for (int corner = 0; corner < 4; ++corner) {
      const auto row = static_cast<std::int64_t>(
          std::find(rows.begin(), rows.end(), cell.v[corner]) - rows.begin());
      if (row == static_cast<std::int64_t>(rows.size())) rows.push_back(cell.v[corner]);
      cells.push_back(row);
    }
  }
  const std::size_t spokes = rows.size() - 1;
  std::vector<double> points(3 * rows.size());
  std::vector<double> tensors(9 * rows.size());
  std::vector<std::int8_t> measured(spokes);
  std::vector<std::uint8_t> neighbor_first(spokes);
  std::vector<double> lengths(spokes);
  const double lower = closed_band_shape ? shape_lower() : kLowerLength;
  bool outside = false;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    std::copy_n(mesh_.point(rows[row]), 3, points.data() + 3 * row);
    std::copy_n(metric(rows[row]), 9, tensors.data() + 9 * row);
    if (row == 0) continue;
    measured[row - 1] = protected_edge(vertex, rows[row]) ? 0 : 1;
    if (closed_band_shape && measured[row - 1] == 0) return TetEditStatus::kApplied;
    neighbor_first[row - 1] = mesh_.vertex_id(rows[row]) < mesh_.vertex_id(vertex);
    lengths[row - 1] = edge_length(vertex, rows[row]);
    if (measured[row - 1] != 0 && !(lengths[row - 1] > 0.0 && std::isfinite(lengths[row - 1]))) {
      return TetEditStatus::kApplied;  // a spoke without a finite quantum is not relocated
    }
    outside = outside || (measured[row - 1] != 0 &&
                          !(lengths[row - 1] >= lower && lengths[row - 1] <= kUpperLength));
  }
  // Stages 3 and 4 can improve quality before the terminal band-only search,
  // in-band only near the out-of-band edges of their step.
  if (closed_band_shape ? outside
                        : !outside && (relocation_stage_ < 3 || tangents ||
                                       !in_stage_region(vertex))) {
    return TetEditStatus::kApplied;
  }
  const bool deep = outside && deep_stage;
  // A boundary vertex is pulled within the hull of its first carrier simplex.
  std::array<double, 9> basis{};
  std::array<double, 9> projector{};
  std::array<double, 3> origin{};
  int basis_count = 0;
  if (!interior) {
    const CarrierSimplex& hull = chart->simplices.front();
    if (hull.size != 2 && hull.size != 3) return TetEditStatus::kStaleGeneration;
    std::array<double, 9> corners{};
    const auto source_corner = [&](std::size_t index) {
      return input_.source_points.data() + 3 * static_cast<std::size_t>(hull.rows[index]);
    };
    std::copy_n(source_corner(0), 3, corners.data());
    std::copy_n(source_corner(1), 3, corners.data() + 3);
    if (hull.size == 3) std::copy_n(source_corner(2), 3, corners.data() + 6);
    basis_count = hull_frame(hull.size, corners.data(), basis.data(), projector.data());
    std::copy_n(corners.data(), 3, origin.data());
  }
  std::vector<double> pulls(3 * (4 * spokes + 8));
  std::int64_t pull_count = -1;
  const bool regular_wall = basis_count == 2 && chart->allow_rounded_images;
  const bool ring = !closed_band_shape && regular_wall && !tangents && !deep &&
                    wall_ring_pull(vertex, star, pulls.data());
  const bool clearance =
      goal == ShapeGoal::kMinimum && worst >= 0 && clearance_pull(vertex, worst, pulls.data());
  const std::int64_t prefix = ring || clearance ? 1 : 0;
  if (tangents) {
    pull_count = boundary_tangent_pulls(points, lengths, measured, metric(vertex), basis_count,
                                        basis.data(), pulls.data());
  } else {
    for (;;) {
      pull_count = relocation_pulls(
          static_cast<std::int64_t>(spokes), points.data(), metric(vertex), lengths.data(),
          measured.data(), deep, interior, basis_count, interior ? nullptr : basis.data(),
          interior ? nullptr : projector.data(), nullptr,
          static_cast<std::int64_t>(pulls.size() / 3) - prefix, pulls.data() + 3 * prefix);
      if (pull_count >= 0) {
        pull_count += prefix;
        break;
      }
      pulls.resize(2 * pulls.size());  // a deep search's tangents need more room
    }
  }
  if (pull_count == 0) return TetEditStatus::kApplied;
  RelocationMode mode;
  mode.deep = deep;
  mode.coarse_steps = bounded_tensor_search_;
  mode.refine = tangents || (deep && relocation_stage_ == 2);
  mode.band_only = tangents || deep;
  mode.shape_lower = closed_band_shape ? lower : 0.0;
  mode.minimum_shape = goal == ShapeGoal::kMinimum;
  mode.report_quality = false;
  mode.try_first_pull_once = ring;
  RelocationStar relocation_star;
  relocation_star.point_count = static_cast<std::int64_t>(rows.size());
  relocation_star.points = points.data();
  relocation_star.metrics = tensors.data();
  relocation_star.measured = measured.data();
  relocation_star.before_lengths = lengths.data();
  relocation_star.neighbor_first = neighbor_first;
  relocation_star.before_quality =
      mean_shape ? before_quality / static_cast<double>(star.size()) : before_quality;
  relocation_star.cell_count = static_cast<std::int64_t>(cells.size() / 4);
  relocation_star.cells = cells.data();
  relocation_star.determinant_floor = kDeterminantFloor;
  relocation_star.improvement = kImprovement;
  RelocationSearch search(*carriers_, input_.source_metrics.data(),
                          input_.source_logarithms.data(), relocation_star, pull_count,
                          pulls.data(), interior ? nullptr : origin.data(),
                          interior ? nullptr : projector.data(), *chart, mode);
  std::vector<RelocationCandidate> group;
  std::vector<double> candidate_lengths;
  bool saw_candidate = false;
  for (;;) {
    const RelocationEnd end = search.next(group, candidate_lengths);
    if (end != RelocationEnd::kGroup) {
      // Refusals arise in the native edit path, and are never cached.
      if (attempt != nullptr && !saw_candidate && end == RelocationEnd::kExhausted) {
        attempt->exhausted_modes |= mode_bit;
      }
      return TetEditStatus::kApplied;
    }
    saw_candidate = true;
    for (const RelocationCandidate& candidate : group) {
      TetEditVertex moved;
      moved.id = mesh_.vertex_id(vertex);
      moved.carrier = candidate.carrier.point;
      moved.edge_witness_supplied = true;
      const auto witness = relocation_witness(vertex, candidate.carrier, *chart);
      if (!witness) continue;
      moved.witness = *witness;
      TetEditStatus status = TetEditStatus::kRejectedTopology;
      const std::span<const TetEditVertex> records(&moved, 1);
      // Boundary stencil verification uses the mesh's binary64 source bank.
      // When the carrier uses an authored rational bank, the source facets
      // and segments are instead checked by the ordinary relocation route.
      if (interior || (!mesh_.implicit_source() &&
                      (moved.witness.stratum != SourceStratum::kNone ||
                       input_.exact_source != nullptr))) {
        status = incidence_.relocate(mesh_, records, star);
      } else if (!mesh_.implicit_source()) {
        const Carrier& carrier = candidate.carrier;
        std::array<std::int32_t, 4> source_rows{};
        std::array<double, 4> weights{};
        const auto count = static_cast<std::size_t>(carrier.size);
        for (std::size_t term = 0; term < count; ++term) {
          source_rows[term] = static_cast<std::int32_t>(carrier.rows[term]);
          weights[term] = std::ldexp(static_cast<double>(carrier.weights[term]), -carrier.bits);
        }
        const detail::SourceStencil stencil{
            std::span(source_rows).first(count), std::span(weights).first(count),
            carrier.exact ? detail::SourceStencilImage::kExact
                          : detail::SourceStencilImage::kRoundedPlane};
        status = incidence_.relocate(mesh_, records, star, &stencil);
      } else {
        // The star-local route does not admit an implicit-boundary mesh.
        TetEdit edit;
        edit.kind = TetEditKind::kRelocate;
        for (std::int32_t slot : star) edit.removed_cell_ids.push_back(mesh_.cell_id(slot));
        edit.changed_vertices.push_back(moved);
        if (!mesh_.apply_relocation_edit(edit, status)) status = TetEditStatus::kRejectedTopology;
      }
      if (status != TetEditStatus::kApplied) {
        if (fatal(status)) return status;
        count_refusal(status);
        continue;
      }
      assign_carrier(vertex, carrier_stencil(candidate.carrier), candidate.metric.data());
      for (std::int32_t row : rows) touch(row);
      for (std::int32_t slot : star) {
        cell_lineage_[static_cast<std::size_t>(slot)].relocated = true;
        cell_quality_[static_cast<std::size_t>(slot)] = std::numeric_limits<double>::quiet_NaN();
      }
      ++counters_.relocations;
      applied = true;
      return TetEditStatus::kApplied;
    }
  }
}

void Adapter::assign_carrier(std::int32_t vertex, Stencil stencil, const double* tensor) {
  stencils_[static_cast<std::size_t>(vertex)] = std::move(stencil);
  set_metric(vertex, tensor);
  carrier_revisions_[static_cast<std::size_t>(vertex)] = mesh_.generation();
  // A move ends the relocated record of everything that continues into the
  // vertex and starts its own.
  const std::uint64_t target = vertex_targets_[static_cast<std::size_t>(vertex)];
  if (const auto found = absorbed_.find(target); found != absorbed_.end()) {
    for (std::uint64_t source : found->second) relocated_sources_.erase(source);
  }
  if (source_targets_.contains(target)) relocated_sources_.insert(target);
}

std::int64_t Adapter::smooth_shape(std::int64_t maximum_sweeps) {
  // Seed the interior frontier once. Local commits requeue the changed
  // star; its mean shape must increase with all spokes still in band.
  // Protected and boundary vertices are never moved here.
  for (std::size_t row = 0; row < mesh_.vertex_slot_count(); ++row) {
    const auto vertex = static_cast<std::int32_t>(row);
    if (!mesh_.live_vertex(vertex) || mesh_.vertex_dimension(vertex) != 3 ||
        dirty_flags_[kRelocatePhase][row] != 0) {
      continue;
    }
    dirty_flags_[kRelocatePhase][row] = 1;
    dirty_[kRelocatePhase].push_back(vertex);
  }
  std::int64_t sweeps = 0;
  for (; sweeps < maximum_sweeps; ++sweeps) {
    const auto applied = relocate_sweep(true);
    if (fatal_ != TetEditStatus::kApplied || applied == 0) return sweeps;
  }
  return sweeps;
}

std::int64_t Adapter::lift_shape_tail() {
  constexpr double kSliverShape = 0.55;
  const auto& cells = mesh_.complex().tets;
  std::vector<std::pair<double, std::uint64_t>> ordered;
  std::vector<std::pair<std::uint64_t, int>> faces;
  std::vector<IdEdge> edges;
  std::int64_t sweeps = 0;
  for (; sweeps < 4; ++sweeps) {
    bool improved = false;
    ordered.clear();
    faces.clear();
    edges.clear();
    for (std::size_t row = 0; row < cells.size(); ++row) {
      const auto slot = static_cast<std::int32_t>(row);
      if (!mesh_.complex().live(slot) || is_ghost(cells[row])) continue;
      const double q = cell_quality(slot);
      if (q < kSliverShape) ordered.emplace_back(q, mesh_.cell_id(slot));
    }
    std::sort(ordered.begin(), ordered.end());
    for (const auto& entry : ordered) {
      const auto id = entry.second;
      const auto slot = mesh_.cell_slot(id);
      if (!(cell_quality(slot) < kSliverShape)) continue;
      const auto& cell = mesh_.tet(slot);
      for (std::int32_t vertex : cell.v) {
        bool applied = false;
        const auto status = relocate(vertex, applied, ShapeGoal::kMinimum);
        if (status != TetEditStatus::kApplied) {
          fatal_ = status;
          return sweeps;
        }
        improved = improved || applied;
      }
      if (!input_.topology_operations || !(cell_quality(slot) < kSliverShape)) continue;
      for (int a = 0; a < 4; ++a) {
        faces.emplace_back(id, a);
        for (int b = a + 1; b < 4; ++b) edges.push_back(edge_ids(cell.v[a], cell.v[b]));
      }
    }
    improved = commit_swaps(faces, edges, std::nullopt, true) > 0 || improved;
    if (fatal_ != TetEditStatus::kApplied) return sweeps;
    if (!improved) break;
  }
  return sweeps;
}

double Adapter::geometric_cell_shape(std::int32_t slot) const {
  std::array<double, 12> corners;
  for (std::size_t corner = 0; corner < 4; ++corner) {
    std::copy_n(mesh_.point(mesh_.tet(slot).v[corner]), 3, corners.data() + 3 * corner);
  }
  double determinant = 0.0;
  double relative = 0.0;
  relative_determinant(corners.data(), determinant, relative);
  return relative > kDeterminantFloor && determinant > 0.0
             ? geometric_shape(corners.data(), determinant)
             : 0.0;
}

std::optional<Adapter::CageMover> Adapter::cage_mover(
    std::int32_t vertex, const LocalIncidence::Star& star, bool band) {
  if (star.empty() || star.size() > kShapeCageCells || mesh_.protected_vertex(vertex) ||
      mesh_.protection_radius(vertex) > 0.0) {
    return std::nullopt;
  }
  for (std::int32_t slot : star) {
    for (std::int32_t other : mesh_.tet(slot).v) {
      if ((!band && !scalar_metric(metric(other))) ||
          (other != vertex && protected_edge(vertex, other))) {
        return std::nullopt;
      }
    }
  }
  const auto chart = carrier_chart(vertex, star, band ? CarrierUse::kBandCage : CarrierUse::kShapeCage);
  if (!chart) return std::nullopt;
  unsigned axes = 7;
  if (!chart->simplices.empty()) {
    axes = chart->simplices.front().projected_axes;
    // These masks come from the exact source numerators, not an angular
    // tolerance on live facets. Only a common Cartesian tangent is supported.
    if (axes == 0 || !std::ranges::all_of(chart->simplices, [&](const CarrierSimplex& simplex) {
          return simplex.projected_axes == axes;
        })) {
      return std::nullopt;
    }
  }
  return CageMover{vertex, *chart, axes};
}

bool Adapter::prepare_shape_cage(std::int32_t seed, ShapeCage& cage, bool whole,
                                std::span<const std::int32_t> band_seeds,
                                std::size_t fixed_axis) {
  const bool band = !band_seeds.empty();
  const auto vertex_id = [&](std::int32_t vertex) { return mesh_.vertex_id(vertex); };
  const auto cell_id = [&](std::int32_t slot) { return mesh_.cell_id(slot); };
  const auto seeds = band ? band_seeds : std::span<const std::int32_t>(mesh_.tet(seed).v, 4);
  std::vector<std::int32_t> candidates(seeds.begin(), seeds.end());
  const std::size_t seed_count = candidates.size();
  const bool sheet = fixed_axis != kNoCageAxis;
  const unsigned fixed_bit = sheet ? 1U << fixed_axis : 0U;
  const double fixed_coordinate = sheet ? mesh_.point(seeds.front())[fixed_axis] : 0.0;
  std::ranges::sort(candidates, {}, vertex_id);
  if (whole) {
    for (std::size_t row = 0; row < mesh_.vertex_slot_count(); ++row) {
      const auto vertex = static_cast<std::int32_t>(row);
      if (!mesh_.live_vertex(vertex)) continue;
      if (std::ranges::find(candidates, vertex) == candidates.end()) candidates.push_back(vertex);
      if (candidates.size() > kShapeCageVertices) return false;
    }
    std::ranges::sort(candidates, {}, vertex_id);
  } else {
    const auto in_seed = [&](std::int32_t vertex) {
      return std::ranges::find(seeds, vertex) != seeds.end();
    };
    std::vector<std::int32_t> star;
    for (std::int32_t vertex : seeds) {
      if (sheet) {
        const auto mover = cage_mover(vertex, incidence_.vertex_star(vertex), true);
        // Expand from face movers, not from a ridge into another sheet.
        if (!mover || mover->axes != (7U ^ fixed_bit) ||
            mesh_.point(vertex)[fixed_axis] != fixed_coordinate) continue;
      }
      incidence_.collect_star(vertex, star);
      if (star.size() > kShapeCageCells) return false;
      for (std::int32_t slot : star) {
        for (std::int32_t other : mesh_.tet(slot).v) {
          if (!in_seed(other)) candidates.push_back(other);
        }
      }
    }
    // Keep seed corners first. Other vertices follow persistent ID order.
    const auto first_neighbor = candidates.begin() + static_cast<std::ptrdiff_t>(seed_count);
    std::ranges::sort(first_neighbor, candidates.end(), {}, vertex_id);
    candidates.erase(std::unique(first_neighbor, candidates.end()), candidates.end());
  }
  std::vector<std::int32_t> united;
  for (std::int32_t vertex : candidates) {
    const auto local = incidence_.vertex_star(vertex);
    auto mover = cage_mover(vertex, local, band);
    if (!mover || (sheet && ((mover->axes & fixed_bit) != 0 ||
                             mesh_.point(vertex)[fixed_axis] != fixed_coordinate))) continue;
    united.clear();
    std::set_union(cage.slots.begin(), cage.slots.end(), local.begin(), local.end(),
                   std::back_inserter(united),
                   [&](std::int32_t a, std::int32_t b) { return cell_id(a) < cell_id(b); });
    if (united.size() > kShapeCageCells) continue;
    cage.slots.swap(united);
    cage.movers.push_back(*mover);
    if (cage.movers.size() == kShapeCageVertices) break;
  }
  if (cage.movers.size() < 2) return false;
  if (band) {
    if (!std::ranges::any_of(cage.movers, [&](const CageMover& mover) {
          return mover.vertex == seeds[0] || mover.vertex == seeds[1];
        })) return false;
  } else if (std::ranges::find(cage.slots, seed) == cage.slots.end()) {
    return false;
  }
  for (std::int32_t slot : cage.slots) {
    const auto& cell = mesh_.tet(slot);
    cage.rows.insert(cage.rows.end(), cell.v, cell.v + 4);
    const double quality = geometric_cell_shape(slot);
    if (!(quality > 0.0) || !std::isfinite(quality)) return false;
    if (!band) {
      cage.minimum_shape = std::min(cage.minimum_shape, quality);
      if (whole) cage.sum_shape += quality;
    }
  }
  std::ranges::sort(cage.rows, {}, vertex_id);
  cage.rows.erase(std::unique(cage.rows.begin(), cage.rows.end()), cage.rows.end());
  const auto row_of = [&](std::int32_t vertex) {
    return static_cast<std::size_t>(
        std::ranges::lower_bound(cage.rows, vertex_id(vertex), {}, vertex_id) - cage.rows.begin());
  };
  cage.points.resize(cage.rows.size());
  if (band) cage.frozen_tensors.resize(cage.rows.size());
  for (std::size_t row = 0; row < cage.rows.size(); ++row) {
    auto& point = cage.points[row];
    std::copy_n(mesh_.point(cage.rows[row]), 3, point.point.data());
    point.variables.fill(-1);
    if (band) std::copy_n(metric(cage.rows[row]), 9, cage.frozen_tensors[row].data());
  }
  for (auto& mover : cage.movers) {
    mover.row = row_of(mover.vertex);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if ((mover.axes & (1U << axis)) != 0) {
        cage.points[mover.row].variables[axis] = static_cast<std::int32_t>(cage.variables++);
      }
    }
  }
  const auto movable = [&](std::size_t row) {
    return std::ranges::any_of(cage.points[row].variables, [](auto index) { return index >= 0; });
  };
  std::vector<std::array<std::size_t, 2>> edges;
  cage.cells.reserve(cage.slots.size());
  for (std::int32_t slot : cage.slots) {
    const auto& cell = mesh_.tet(slot);
    const std::array corners{row_of(cell.v[0]), row_of(cell.v[1]),
                             row_of(cell.v[2]), row_of(cell.v[3])};
    cage.cells.push_back(corners);
    for (std::size_t a = 0; a < 4; ++a) {
      for (std::size_t b = a + 1; b < 4; ++b) {
        if (!movable(corners[a]) && !movable(corners[b])) continue;
        edges.push_back({std::min(corners[a], corners[b]), std::max(corners[a], corners[b])});
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  cage.edges.reserve(edges.size());
  for (const auto& edge : edges) {
    const auto [a, b] = edge;
    const double physical = std::sqrt(
        cage_distance_squared(cage.points[a].point.data(), cage.points[b].point.data()));
    // Both measurements use persistent endpoint order. In band mode the
    // original transported tensors are also the frozen model, hence scale 1.
    const double measured =
        band ? length(cage.points[a].point.data(), cage.points[b].point.data(),
                      cage.frozen_tensors[a].data(), cage.frozen_tensors[b].data())
             : edge_length(cage.rows[a], cage.rows[b]);
    if (!(physical > 0.0 && std::isfinite(physical)) ||
        (band ? !(measured > 0.0 && std::isfinite(measured))
              : !(measured >= shape_lower() && measured <= kUpperLength))) {
      return false;
    }
    const double scale = band ? 1.0 : measured / physical;
    if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    cage.minimum_distance = std::min(cage.minimum_distance, physical);
    cage.edges.push_back({edge, scale});
  }
  return !cage.edges.empty();
}

TetEditStatus Adapter::commit_shape_cage(
    const ShapeCage& cage, std::int32_t seed, double seed_shape,
    std::span<const double> displacement, std::span<double> transported_scales,
    bool improve_mean, bool& applied) {
  const bool band = !cage.frozen_tensors.empty();
  struct Move {
    std::array<double, 3> point;
    Stencil stencil;
    std::array<double, 9> metric;
  };
  std::vector<Move> moves;
  moves.reserve(cage.movers.size());
  std::vector<Move*> overlay(cage.rows.size(), nullptr);
  TetEdit edit;
  edit.kind = TetEditKind::kRelocate;
  edit.changed_vertices.reserve(cage.movers.size());
  for (const auto& mover : cage.movers) {
    const auto& origin = cage.points[mover.row];
    auto goal = origin.point;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto index = origin.variables[axis];
      if (index >= 0) goal[axis] += displacement[static_cast<std::size_t>(index)];
    }
    if (goal == origin.point) continue;
    const auto carriers = carriers_->carriers(goal.data(), origin.point.data(), mover.chart, -1,
                                              CarrierOrder::kFineFirst, 2);
    for (const auto& carrier : carriers) {
      bool on_chart = true;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (origin.variables[axis] < 0 && carrier.point[axis] != origin.point[axis]) {
          on_chart = false;
        }
      }
      if (!on_chart) continue;
      const auto witness = relocation_witness(mover.vertex, carrier, mover.chart);
      if (!witness) continue;
      std::array<double, 9> tensor;
      auto stencil = carrier_stencil(carrier);
      if (!vertex_metric(stencil, tensor.data()) || (!band && !scalar_metric(tensor.data()))) {
        continue;
      }
      moves.push_back({carrier.point, std::move(stencil), tensor});
      overlay[mover.row] = &moves.back();
      TetEditVertex changed;
      changed.id = mesh_.vertex_id(mover.vertex);
      changed.carrier = carrier.point;
      changed.witness = *witness;
      changed.edge_witness_supplied = true;
      edit.changed_vertices.push_back(changed);
      break;
    }
  }
  if (moves.empty()) return TetEditStatus::kApplied;
  const auto point = [&](std::size_t row) {
    return overlay[row] ? overlay[row]->point.data() : cage.points[row].point.data();
  };
  const auto tensor = [&](std::size_t row) {
    return overlay[row] ? overlay[row]->metric.data() : metric(cage.rows[row]);
  };
  const double lower = shape_lower();
  bool in_band = true;
  for (std::size_t index = 0; index < cage.edges.size(); ++index) {
    // Rows are already in the persistent endpoint order of edge_length().
    const auto [a, b] = cage.edges[index].rows;
    // Relinearize transport against the same frozen model on the actual
    // snapped edge, not on the continuous step or a Euclidean norm.
    const double baseline = band
        ? length(point(a), point(b), cage.frozen_tensors[a].data(), cage.frozen_tensors[b].data())
        : std::sqrt(cage_distance_squared(point(a), point(b)));
    const double value = length(point(a), point(b), tensor(a), tensor(b));
    if (!(baseline > 0.0) || !std::isfinite(baseline) || !std::isfinite(value) ||
        (band && !(value > 0.0))) {
      return TetEditStatus::kApplied;
    }
    transported_scales[index] = value / baseline;
    in_band = in_band && value >= lower && value <= kUpperLength;
  }
  if (!in_band) return TetEditStatus::kApplied;
  std::vector<std::int32_t> changed_cells;
  changed_cells.reserve(cage.cells.size());
  edit.removed_cell_ids.reserve(cage.cells.size());
  double sum_shape = 0.0;
  const double retained_shape = std::min(cage.minimum_shape, kShapeCageFloor);
  for (std::size_t i = 0; i < cage.cells.size(); ++i) {
    const auto& cell = cage.cells[i];
    const auto slot = cage.slots[i];
    std::array<double, 12> corners;
    bool changed = false;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      changed = changed || overlay[cell[corner]] != nullptr;
      std::copy_n(point(cell[corner]), 3, corners.data() + 3 * corner);
    }
    double determinant = 0.0;
    double relative = 0.0;
    relative_determinant(corners.data(), determinant, relative);
    if (!(determinant > 0.0 && relative > kDeterminantFloor) ||
        (band && !(std::isfinite(determinant) && std::isfinite(relative)))) {
      return TetEditStatus::kApplied;
    }
    if (!band) {
      const double quality = geometric_shape(corners.data(), determinant);
      const bool shape_gain = improve_mean
                                  ? quality >= retained_shape
                                  : quality > cage.minimum_shape + kImprovement &&
                                        (slot != seed || quality > seed_shape + kImprovement);
      if (!shape_gain) return TetEditStatus::kApplied;
      if (improve_mean) sum_shape += quality;
    }
    if (changed) {
      changed_cells.push_back(slot);
      edit.removed_cell_ids.push_back(mesh_.cell_id(slot));
    }
  }
  if (improve_mean && !(sum_shape > cage.sum_shape + kImprovement)) return TetEditStatus::kApplied;
  TetEditStatus status = TetEditStatus::kRejectedTopology;
  if (!mesh_.apply_relocation_edit(edit, status)) status = TetEditStatus::kRejectedTopology;
  if (status != TetEditStatus::kApplied) {
    if (fatal(status)) return status;
    count_refusal(status);
    return TetEditStatus::kApplied;
  }
  for (const auto& mover : cage.movers) {
    if (auto* move = overlay[mover.row]) {
      assign_carrier(mover.vertex, std::move(move->stencil), move->metric.data());
    }
  }
  for (auto slot : changed_cells) {
    for (std::int32_t vertex : mesh_.tet(slot).v) touch(vertex);
    cell_lineage_[static_cast<std::size_t>(slot)].relocated = true;
    cell_quality_[static_cast<std::size_t>(slot)] = std::numeric_limits<double>::quiet_NaN();
  }
  counters_.relocations += static_cast<std::int64_t>(moves.size());
  applied = true;
  return TetEditStatus::kApplied;
}

TetEditStatus Adapter::relocate_shape_cage(std::int32_t seed, double seed_shape, bool& applied,
                                          bool improve_mean,
                                          std::span<const std::int32_t> band_seeds,
                                          std::size_t fixed_axis) {
  applied = false;
  ShapeCage cage;
  if (!prepare_shape_cage(seed, cage, improve_mean, band_seeds, fixed_axis)) {
    return TetEditStatus::kApplied;
  }
  const bool band = !cage.frozen_tensors.empty();
  const auto budget = band && bounded_tensor_search_ ? ShapeCageBudget::kCoarseTensor
                                                      : ShapeCageBudget::kFull;
  std::vector<double> displacement(cage.variables);
  std::vector<double> scales(cage.edges.size());
  const double target = band ? 0.0
                            : improve_mean ? std::min(cage.minimum_shape, kShapeCageFloor) + 0.002
                                           : kShapeCageTarget;
  const int attempts = band || (!improve_mean && seed_shape < kShapeCageFloor - 0.05) ? 2 : 1;
  bool proposed = false;
  for (int attempt = 0; attempt < attempts; ++attempt) {
    // Retry within a wider trust region if the narrow model is obstructed.
    // The scalar minimum objective can also accept a smaller gain on retry.
    const double radius = (attempt == 0 ? 0.5 : 0.75) * cage.minimum_distance;
    const double goal = attempt == 0 || band ? target : 0.5 * (seed_shape + target);
    for (int correction = 0; correction < 5; ++correction) {
      if (!propose_shape_cage(cage.points, cage.cells, cage.edges, radius,
                              shape_lower(), goal, displacement, improve_mean,
                              cage.frozen_tensors, budget)) {
        break;
      }
      proposed = true;
      std::ranges::fill(scales, 0.0);
      const auto status =
          commit_shape_cage(cage, seed, seed_shape, displacement, scales, improve_mean, applied);
      if (status != TetEditStatus::kApplied || applied) return status;
      if (!std::ranges::all_of(scales, [](double scale) {
            return scale > 0.0 && std::isfinite(scale);
          })) break;
      // Correct the frozen source sizes with the measured transported
      // sizes. The next candidate must still meet the same exact band.
      for (std::size_t i = 0; i < scales.size(); ++i) {
        cage.edges[i].scale += 0.75 * (scales[i] - cage.edges[i].scale);
      }
    }
  }
  if (improve_mean && proposed) {
    // Rounding and transport can put the full step just below the retained
    // shape floor. A shorter joint move is subjected to the same exact gate.
    for (int backtrack = 0; backtrack < 3; ++backtrack) {
      for (double& coordinate : displacement) coordinate *= 0.75;
      const auto status =
          commit_shape_cage(cage, seed, seed_shape, displacement, scales, true, applied);
      if (status != TetEditStatus::kApplied || applied) return status;
    }
  }
  return TetEditStatus::kApplied;
}

std::int64_t Adapter::repair_band_cages() {
  const double lower = shape_lower();
  const auto short_length = [&](double value) { return value > 0.0 && value < lower; };
  // Keep just the worst distinct short edges, in persistent-ID tie order.
  std::vector<MeasuredEdge> ordered;
  ordered.reserve(kBandCageEdges);
  for (const Tetrahedron& cell : mesh_.complex().tets) {
    if (cell.v[0] == kDeadVertex || is_ghost(cell)) continue;
    for (int i = 0; i < 4; ++i) {
      const auto a = cell.v[i];
      for (int j = i + 1; j < 4; ++j) {
        const auto b = cell.v[j];
        const double value = edge_length(a, b);
        if (!short_length(value) || protected_edge(a, b)) continue;
        const MeasuredEdge edge{value, edge_ids(a, b)};
        const auto at = std::lower_bound(ordered.begin(), ordered.end(), edge);
        if (at != ordered.end() && *at == edge) continue;
        if (ordered.size() == kBandCageEdges) {
          if (at == ordered.end()) continue;
          const auto offset = at - ordered.begin();
          ordered.pop_back();
          ordered.insert(ordered.begin() + offset, edge);
        } else {
          ordered.insert(at, edge);
        }
      }
    }
  }

  std::int64_t applied = 0;
  for (const auto& entry : ordered) {
    const auto a = mesh_.vertex_slot(entry.second[0]);
    const auto b = mesh_.vertex_slot(entry.second[1]);
    if (!short_length(edge_length(a, b))) continue;

    std::size_t fixed_axis = kNoCageAxis;
    for (const auto vertex : {a, b}) {
      if (mesh_.vertex_dimension(vertex) != 2) continue;
      const auto mover = cage_mover(vertex, incidence_.vertex_star(vertex), true);
      if (!mover) continue;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (mover->axes == (7U ^ (1U << axis)) &&
            mesh_.point(a)[axis] == mesh_.point(b)[axis]) {
          fixed_axis = axis;
          break;
        }
      }
      if (fixed_axis != kNoCageAxis) break;
    }

    // Seed one rings from connected short edges. Keep the target endpoints
    // first so every accepted transaction actually closes the target.
    std::array<std::int32_t, 2 * kBandCageEdges> component{a, b};
    std::size_t count = 2;
    for (std::size_t next = 0; next < count; ++next) {
      const auto id = mesh_.vertex_id(component[next]);
      for (const auto& neighbor : ordered) {
        const auto& edge = neighbor.second;
        if (edge[0] != id && edge[1] != id) continue;
        const auto other = mesh_.vertex_slot(edge[edge[0] == id ? 1 : 0]);
        const auto known = std::span(component).first(count);
        if (std::ranges::find(known, other) != known.end()) continue;
        if (fixed_axis != kNoCageAxis &&
            mesh_.point(other)[fixed_axis] != mesh_.point(a)[fixed_axis]) continue;
        if (short_length(edge_length(component[next], other))) component[count++] = other;
      }
    }
    const auto seeds = std::span<const std::int32_t>(component).first(count);
    bool done = false;
    auto status = relocate_shape_cage(-1, 0.0, done, false, seeds, fixed_axis);
    if (status == TetEditStatus::kApplied && !done && fixed_axis != kNoCageAxis) {
      status = relocate_shape_cage(-1, 0.0, done, false, seeds);
    }
    if (status != TetEditStatus::kApplied) {
      fatal_ = status;
      return applied;
    }
    applied += done ? 1 : 0;
  }
  return applied;
}

std::int64_t Adapter::lift_shape_cages() {
  std::vector<std::pair<double, std::uint64_t>> ordered;
  const auto& complex = mesh_.complex();
  std::size_t cell_count = 0;
  std::optional<std::uint64_t> polish_seed;
  std::int64_t sweeps = 0;
  for (; sweeps < 3; ++sweeps) {
    ordered.clear();
    const double threshold = sweeps == 0 ? kShapeCageTarget : kShapeCageFloor;
    for (std::size_t row = 0; row < complex.tets.size(); ++row) {
      const auto slot = static_cast<std::int32_t>(row);
      const auto& cell = complex.tets[row];
      if (!complex.live(slot) || is_ghost(cell)) continue;
      if (sweeps == 0) ++cell_count;
      if (!std::ranges::all_of(cell.v, [&](std::int32_t vertex) {
            return scalar_metric(metric(vertex));
          })) continue;
      const double quality = geometric_cell_shape(slot);
      if (quality > 0.0 && quality < threshold) {
        ordered.emplace_back(quality, mesh_.cell_id(slot));
      }
    }
    std::sort(ordered.begin(), ordered.end());
    if (ordered.empty()) break;
    if (!polish_seed) polish_seed = ordered.front().second;
    bool applied = false;
    for (const auto& [prior, id] : ordered) {
      const auto slot = mesh_.cell_slot(id);
      const double quality = applied ? geometric_cell_shape(slot) : prior;
      if (!(quality > 0.0 && quality < threshold)) continue;
      bool done = false;
      const auto status = relocate_shape_cage(slot, quality, done);
      if (status != TetEditStatus::kApplied) {
        fatal_ = status;
        return sweeps;
      }
      applied = applied || done;
    }
    if (!applied) break;
  }
  if (sweeps > 0 && cell_count <= kShapeCageCells && polish_seed) {
    const auto slot = mesh_.cell_slot(*polish_seed);
    for (int pass = 0; pass < 3; ++pass) {
      bool polished = false;
      const auto status = relocate_shape_cage(slot, geometric_cell_shape(slot), polished, true);
      if (status != TetEditStatus::kApplied) {
        fatal_ = status;
        break;
      }
      if (!polished) break;
      ++sweeps;
    }
  }
  return sweeps;
}

std::int64_t Adapter::relocate_sweep(bool closed_band_shape) {
  std::vector<std::int32_t> vertices(dirty_[kRelocatePhase].begin(), dirty_[kRelocatePhase].end());
  for (std::int32_t vertex : vertices) dirty_flags_[kRelocatePhase][static_cast<std::size_t>(vertex)] = 0;
  dirty_[kRelocatePhase].clear();
  std::ranges::sort(vertices, {}, [&](std::int32_t vertex) { return mesh_.vertex_id(vertex); });
  std::int64_t applied = 0;
  for (std::int32_t vertex : vertices) {
    if (!mesh_.live_vertex(vertex)) continue;
    bool done = false;
    const auto goal = closed_band_shape ? ShapeGoal::kMean : ShapeGoal::kBand;
    const TetEditStatus status = relocate(vertex, done, goal);
    if (status != TetEditStatus::kApplied) {
      fatal_ = status;
      return applied;
    }
    applied += done ? 1 : 0;
  }
  if (!closed_band_shape && relocation_stage_ == 4) applied += pair_sweep(vertices);
  return applied;
}

TetEditStatus Adapter::relocate_pair(std::int32_t u, std::int32_t v, bool& applied) {
  applied = false;
  for (std::int32_t w : {u, v}) {
    if (mesh_.protected_vertex(w) || mesh_.protection_radius(w) > 0.0 ||
        mesh_.vertex_dimension(w) != 3) {
      return TetEditStatus::kApplied;
    }
  }
  // The joint star and its measured edges at either end.
  std::vector<std::int32_t> cavity;
  incidence_.collect_star(u, cavity);
  std::vector<std::int32_t> other;
  incidence_.collect_star(v, other);
  cavity.insert(cavity.end(), other.begin(), other.end());
  std::sort(cavity.begin(), cavity.end());
  cavity.erase(std::unique(cavity.begin(), cavity.end()), cavity.end());
  std::vector<std::array<std::int32_t, 2>> edges;
  double before_quality = std::numeric_limits<double>::infinity();
  for (std::int32_t slot : cavity) {
    before_quality = std::min(before_quality, cell_quality(slot));
    const Tetrahedron& cell = mesh_.tet(slot);
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        std::int32_t a = cell.v[first];
        std::int32_t b = cell.v[second];
        if (mesh_.vertex_id(b) < mesh_.vertex_id(a)) std::swap(a, b);
        if ((a == u || a == v || b == u || b == v) && !protected_edge(a, b)) edges.push_back({a, b});
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  Violations before;
  for (const auto& [a, b] : edges) before.add(edge_length(a, b));
  before.finish();
  const double current = edge_length(u, v);
  if (!before.measured || !(current > 0.0 && std::isfinite(current))) return TetEditStatus::kApplied;
  const double* pu = mesh_.point(u);
  const double* pv = mesh_.point(v);
  // Each end moves a fraction of uv outward (inward when negative): the
  // least move that brings uv to the band edge, then multiples of it up to
  // the move to unit length (the least-displacement joint moves first).
  const double band_edge = current < kLowerLength ? kLowerLength : kUpperLength;
  const double least = (band_edge / current - 1.0) / 2.0;
  const double unit = (1.0 / current - 1.0) / 2.0;
  for (const double factor : {1.0, 1.125, 1.25, 1.5, 2.0, 4.0, 16.0, 64.0}) {
    const double fraction = std::abs(least * factor) < std::abs(unit) ? least * factor : unit;
    std::array<double, 3> goal_u{};
    std::array<double, 3> goal_v{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double offset = fraction * (pv[axis] - pu[axis]);
      goal_u[axis] = pu[axis] - offset;
      goal_v[axis] = pv[axis] + offset;
    }
    // The carriers closest to each goal first: a cage admits only a narrow
    // window of joint moves.
    const std::vector<Carrier> moves_u =
        carriers_->carriers(goal_u.data(), pu, {}, -1, CarrierOrder::kFineFirst, 2);
    const std::vector<Carrier> moves_v =
        carriers_->carriers(goal_v.data(), pv, {}, -1, CarrierOrder::kFineFirst, 2);
    for (const Carrier& move_u : moves_u) {
      std::array<double, 9> tensor_u{};
      if (!vertex_metric(carrier_stencil(move_u), tensor_u.data())) continue;
      for (const Carrier& move_v : moves_v) {
        std::array<double, 9> tensor_v{};
        if (!vertex_metric(carrier_stencil(move_v), tensor_v.data())) continue;
        const auto point = [&](std::int32_t w) {
          return w == u ? move_u.point.data() : w == v ? move_v.point.data() : mesh_.point(w);
        };
        const auto tensor = [&](std::int32_t w) {
          return w == u ? tensor_u.data() : w == v ? tensor_v.data() : metric(w);
        };
        Violations after;
        for (const auto& [a, b] : edges) after.add(length(point(a), point(b), tensor(a), tensor(b)));
        if (!after.measured) continue;
        after.finish();
        double after_quality = std::numeric_limits<double>::infinity();
        bool valid = true;
        for (std::int32_t slot : cavity) {
          const Tetrahedron& cell = mesh_.tet(slot);
          std::array<double, 12> corners{};
          std::array<double, 36> tensors{};
          for (std::size_t corner = 0; corner < 4; ++corner) {
            std::copy_n(point(cell.v[corner]), 3, corners.data() + 3 * corner);
            std::copy_n(tensor(cell.v[corner]), 9, tensors.data() + 9 * corner);
          }
          double determinant = 0.0;
          double relative = 0.0;
          relative_determinant(corners.data(), determinant, relative);
          const double value = tetra_quality(corners.data(), tensors.data());
          if (!(determinant > 0.0) || !(relative > kDeterminantFloor) || !(value > 0.0) ||
              !std::isfinite(value)) {
            valid = false;
            break;
          }
          after_quality = std::min(after_quality, value);
        }
        if (!valid ||
            !objective_decreases(before, after, after_quality > before_quality + kImprovement)) {
          continue;
        }
        TetEdit edit;
        edit.kind = TetEditKind::kRelocate;
        for (std::int32_t slot : cavity) edit.removed_cell_ids.push_back(mesh_.cell_id(slot));
        std::sort(edit.removed_cell_ids.begin(), edit.removed_cell_ids.end());
        for (const auto& [w, move] : {std::pair{u, &move_u}, std::pair{v, &move_v}}) {
          TetEditVertex moved;
          moved.id = mesh_.vertex_id(w);
          moved.carrier = move->point;
          moved.witness = mesh_.witness(w);
          moved.edge_witness_supplied = true;
          edit.changed_vertices.push_back(moved);
        }
        TetEditStatus status = TetEditStatus::kRejectedTopology;
        if (!mesh_.apply_relocation_edit(edit, status)) status = TetEditStatus::kRejectedTopology;
        if (status != TetEditStatus::kApplied) {
          if (fatal(status)) return status;
          count_refusal(status);
          continue;
        }
        assign_carrier(u, carrier_stencil(move_u), tensor_u.data());
        assign_carrier(v, carrier_stencil(move_v), tensor_v.data());
        for (std::int32_t slot : cavity) {
          for (std::int32_t w : mesh_.tet(slot).v) touch(w);
          cell_lineage_[static_cast<std::size_t>(slot)].relocated = true;
          cell_quality_[static_cast<std::size_t>(slot)] = std::numeric_limits<double>::quiet_NaN();
        }
        counters_.relocations += 2;
        applied = true;
        return TetEditStatus::kApplied;
      }
    }
    if (fraction == unit) break;
  }
  return TetEditStatus::kApplied;
}

std::int64_t Adapter::pair_sweep(std::span<const std::int32_t> vertices) {
  // The out-of-band edges at the changed vertices, each once.
  std::vector<MeasuredEdge> edges;
  std::vector<std::int32_t> star;
  for (std::int32_t vertex : vertices) {
    if (!mesh_.live_vertex(vertex)) continue;
    incidence_.collect_star(vertex, star);
    for (std::int32_t slot : star) {
      for (std::int32_t other : mesh_.tet(slot).v) {
        if (other == vertex || protected_edge(vertex, other)) continue;
        const double value = edge_length(vertex, other);
        if (!(value >= kLowerLength && value <= kUpperLength)) {
          edges.emplace_back(value, edge_ids(vertex, other));
        }
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::int64_t applied = 0;
  for (const auto& [value, key] : edges) {
    const auto a = mesh_.vertex_slot(key[0]);
    const auto b = mesh_.vertex_slot(key[1]);
    if (!mesh_.live_vertex(a) || !mesh_.live_vertex(b) || edge_length(a, b) != value) continue;
    bool done = false;
    const TetEditStatus status = relocate_pair(a, b, done);
    if (status != TetEditStatus::kApplied) {
      fatal_ = status;
      return applied;
    }
    applied += done ? 1 : 0;
  }
  return applied;
}

// ----------------------------------------------------------------- run ---

void Adapter::write(AdaptOutput& output) const {
  output.counters = counters_;
  std::vector<std::pair<std::uint64_t, std::int32_t>> vertices;
  for (std::size_t slot = 0; slot < mesh_.vertex_slot_count(); ++slot) {
    const auto at = static_cast<std::int32_t>(slot);
    if (mesh_.live_vertex(at)) vertices.emplace_back(mesh_.vertex_id(at), at);
  }
  std::sort(vertices.begin(), vertices.end());
  output.stencil_offsets.assign(1, 0);
  for (const auto& [id, slot] : vertices) {
    const auto at = static_cast<std::size_t>(slot);
    output.vertex_ids.push_back(id);
    output.vertex_target_ids.push_back(vertex_targets_[at]);
    for (const StencilTerm& term : stencils_[at]) {
      output.stencil_rows.push_back(term.row);
      output.stencil_weights.push_back(term.weight);
    }
    output.stencil_offsets.push_back(static_cast<std::int64_t>(output.stencil_rows.size()));
    output.metrics.insert(output.metrics.end(), metrics_.begin() + 9 * at,
                          metrics_.begin() + 9 * at + 9);
  }
  std::vector<std::pair<std::uint64_t, std::int32_t>> cells;
  const auto& tets = mesh_.complex().tets;
  for (std::size_t slot = 0; slot < tets.size(); ++slot) {
    const auto at = static_cast<std::int32_t>(slot);
    if (mesh_.complex().live(at) && !is_ghost(tets[slot])) cells.emplace_back(mesh_.cell_id(at), at);
  }
  std::sort(cells.begin(), cells.end());
  output.ancestry_offsets.assign(1, 0);
  for (const auto& [id, slot] : cells) {
    const auto at = static_cast<std::size_t>(slot);
    output.cell_ids.push_back(id);
    output.cell_target_ids.push_back(cell_targets_[at]);
    output.cell_blocks.push_back(cell_blocks_[at]);
    const CellLineage& lineage = cell_lineage_[at];
    output.ancestry_ids.insert(output.ancestry_ids.end(), lineage.ancestry.begin(),
                               lineage.ancestry.end());
    output.ancestry_offsets.push_back(static_cast<std::int64_t>(output.ancestry_ids.size()));
    output.ancestry_kinds.push_back(static_cast<std::int32_t>(lineage.kind));
    output.cell_relocated.push_back(lineage.relocated ? 1 : 0);
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> collapsed(collapsed_into_.begin(),
                                                                 collapsed_into_.end());
  std::sort(collapsed.begin(), collapsed.end());
  for (const auto& [source, target] : collapsed) {
    output.vertex_relation_sources.push_back(source);
    output.vertex_relation_targets.push_back(target);
    output.vertex_relation_kinds.push_back(static_cast<std::int32_t>(AdaptLineage::kCollapsedInto));
  }
  std::vector<std::uint64_t> relocated(relocated_sources_.begin(), relocated_sources_.end());
  std::sort(relocated.begin(), relocated.end());
  for (std::uint64_t source : relocated) {
    const auto found = collapsed_into_.find(source);
    output.vertex_relation_sources.push_back(source);
    output.vertex_relation_targets.push_back(found == collapsed_into_.end() ? source : found->second);
    output.vertex_relation_kinds.push_back(static_cast<std::int32_t>(AdaptLineage::kRelocated));
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> unrepresentable;
  for (const auto& [key, failure] : unrepresentable_) {
    const auto a = static_cast<std::int32_t>(key >> 32);
    const auto b = static_cast<std::int32_t>(key & 0xffffffffU);
    if (!mesh_.live_vertex(a) || !mesh_.live_vertex(b)) continue;
    if (failure.first_revision != carrier_revisions_[static_cast<std::size_t>(a)] ||
        failure.second_revision != carrier_revisions_[static_cast<std::size_t>(b)] ||
        !(edge_length(a, b) > kUpperLength) || incidence_.edge_star(a, b).empty()) {
      continue;
    }
    const std::uint64_t x = vertex_targets_[static_cast<std::size_t>(a)];
    const std::uint64_t y = vertex_targets_[static_cast<std::size_t>(b)];
    unrepresentable.emplace_back(std::min(x, y), std::max(x, y));
  }
  std::sort(unrepresentable.begin(), unrepresentable.end());
  for (const auto& [x, y] : unrepresentable) {
    output.unrepresentable_edges.push_back(x);
    output.unrepresentable_edges.push_back(y);
  }
}

TetEditStatus Adapter::run(AdaptOutput& output) {
  bind();
  output = AdaptOutput{};
  // No split or collapse is eligible in this interval. The caller's
  // completion lower bound can be looser than the actual collapse trigger.
  const double initial_lower =
      input_.topology_operations ? std::max(input_.lower, kLowerLength) : input_.lower;
  if (complete(initial_lower)) {
    write(output);
    output.passes = 1;
    return TetEditStatus::kApplied;
  }
  bool tensor_field = false;
  if (input_.relocation && carriers_) {
    for (std::size_t i = 0; i < input_.source_metrics.size(); i += 9) {
      if (!scalar_metric(input_.source_metrics.data() + i)) {
        tensor_field = true;
        break;
      }
    }
  }
  bounded_tensor_search_ = tensor_field && !mesh_.implicit_source();
  separated_splits_ = input_.topology_operations && input_.relocation && carriers_ &&
                      !tensor_field && !mesh_.implicit_source();
  if (mesh_.vertex_slot_count() >= 512) {
    edge_measurements_.resize(16'384);
  }
  for (std::size_t slot = 0; slot < mesh_.vertex_slot_count(); ++slot) {
    if (mesh_.live_vertex(static_cast<std::int32_t>(slot))) touch(static_cast<std::int32_t>(slot));
  }
  AdaptStatus status = AdaptStatus::kComplete;
  std::int64_t passes = 0;
  bool shape_searched = false;
  bool band_searched = false;
  // A stall certifies exhaustion only for a generation that was fully
  // queued at stage zero with all requested search gates enabled.
  std::optional<std::uint64_t> exhaustion_generation;
  std::optional<std::uint64_t> cage_generation;
  for (;;) {
    ++passes;
    std::int64_t applied = 0;
    std::int64_t split = 0;
    if (input_.topology_operations) {
      split = split_sweep();
      if (fatal_ != TetEditStatus::kApplied) return fatal_;
      applied += split;
      applied += collapse_sweep();
      if (fatal_ != TetEditStatus::kApplied) return fatal_;
    }
    if (split != 0) continue;
    // Splits are exhausted at the changed vertices: the band may hold.
    const bool done = complete(input_.lower);
    if (done) {
      // Bound global polishing after net scalar refinement when the local
      // worst-cell repair will follow. Never cap an incomplete-band search.
      const bool scalar_refinement = input_.relocation && carriers_ && !tensor_field &&
                                    counters_.splits > counters_.collapses;
      // An edited mesh can reach the band before the shape search. This
      // pass leaves every retained edge unchanged and checks each new edge
      // against both the unit band and the caller's completion bound.
      if (input_.topology_operations && !shape_searched) {
        const auto shaped = shape_sweeps(scalar_refinement ? kScalarTopologySweeps
                                                           : kUntilShapeStall);
        if (fatal_ != TetEditStatus::kApplied) return fatal_;
        passes += shaped;
      }
      if (input_.relocation && carriers_) {
        passes += smooth_shape(scalar_refinement ? kScalarShapeSweeps : kUntilShapeStall);
        if (fatal_ != TetEditStatus::kApplied) return fatal_;
        passes += lift_shape_tail();
        if (fatal_ != TetEditStatus::kApplied) return fatal_;
        passes += lift_shape_cages();
        if (fatal_ != TetEditStatus::kApplied) return fatal_;
      }
      status = AdaptStatus::kComplete;
      break;
    }
    if (input_.topology_operations) {
      applied += swap_sweep();
      if (fatal_ != TetEditStatus::kApplied) return fatal_;
    }
    const std::int64_t topology = applied;
    if (input_.relocation) {
      applied += relocate_sweep();
      if (fatal_ != TetEditStatus::kApplied) return fatal_;
    }
    // A topology edit changes what shallow relocation can reach; relocation
    // progress alone continues at its depth until it stalls there.
    if (topology != 0) relocation_stage_ = 0;
    if (applied != 0) continue;
    // A short-edge cage can finish where another single-vertex sweep
    // cannot. Try it once per plateau before descending to finer steps.
    if (bounded_tensor_search_ && boundary_moves_ && relocation_stage_ == 0 &&
        cage_generation != mesh_.generation()) {
      cage_generation = mesh_.generation();
      const auto repaired = repair_band_cages();
      if (fatal_ != TetEditStatus::kApplied) return fatal_;
      if (repaired != 0) continue;
    }
    // Exhaust the cheap single-vertex interior search before enabling
    // boundary moves, moving survivors and deeper searches.
    bool stalled = false;
    bool stage_step = false;
    if (input_.topology_operations && separated_splits_) {
      separated_splits_ = false;
    } else if (input_.relocation && !boundary_moves_) {
      boundary_moves_ = true;
    } else if (input_.topology_operations && carriers_ && !survivor_moves_) {
      survivor_moves_ = true;
    } else if (input_.relocation && carriers_ && relocation_stage_ < 5) {
      ++relocation_stage_;
      stage_step = true;
    } else if (input_.relocation && carriers_ && exhaustion_generation != mesh_.generation()) {
      relocation_stage_ = 0;
    } else if (input_.topology_operations && !shape_searched) {
      shape_searched = true;
      const auto shaped = shape_sweeps();
      if (fatal_ != TetEditStatus::kApplied) return fatal_;
      stalled = shaped == 0;
      passes += shaped;
      // Topology changes reopen the ordinary search on the new generation.
      relocation_stage_ = 0;
    } else {
      stalled = true;
    }
    if (stalled) {
      const auto repaired = !band_searched && tensor_field ? repair_band_cages() : 0;
      if (fatal_ != TetEditStatus::kApplied) return fatal_;
      if (repaired != 0) {
        band_searched = true;
        ++passes;
      } else if (bounded_tensor_search_) {
        // Exhaustive search must get its own generation and refusal cache:
        // the coarse prefix has not certified a stall.
        bounded_tensor_search_ = false;
        relocation_attempts_.clear();
        band_searched = false;
        shape_searched = false;
      } else {
        status = AdaptStatus::kStalled;
        break;
      }
      // Requeue the ordinary search and let it certify global completion.
      relocation_stage_ = 0;
    }
    if (relocation_stage_ == 0 && boundary_moves_ && carriers_ &&
        (!input_.topology_operations || survivor_moves_)) {
      // Reached only when opening the last gate or restarting at stall.
      // A local topology reset never records a new full-sweep generation.
      exhaustion_generation = mesh_.generation();
    }
    if (stage_step) {
      queue_stage_search();
    } else {
      for (std::size_t slot = 0; slot < mesh_.vertex_slot_count(); ++slot) {
        if (mesh_.live_vertex(static_cast<std::int32_t>(slot))) touch(static_cast<std::int32_t>(slot));
      }
    }
  }
  write(output);
  output.status = status;
  output.passes = passes;
  return TetEditStatus::kApplied;
}

}  // namespace

TetEditStatus adapt_metric(TetMesh& mesh, const AdaptInput& input, AdaptOutput& output) {
  Adapter adapter(mesh, input);
  return adapter.run(output);
}

}  // namespace phx::mc::metric
