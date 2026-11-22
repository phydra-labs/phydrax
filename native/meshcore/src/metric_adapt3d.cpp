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
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "cavity.hpp"
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

std::uint64_t edge_key(std::int32_t a, std::int32_t b) {
  const auto low = static_cast<std::uint32_t>(std::min(a, b));
  const auto high = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(low) << 32) | high;
}

// An edge by its two vertex IDs, ascending (exact key; EdgeIdHash hashes it).
using IdEdge = std::array<std::uint64_t, 2>;
IdEdge id_edge(std::uint64_t a, std::uint64_t b) { return {std::min(a, b), std::max(a, b)}; }

using IdFace = std::array<std::uint64_t, 3>;

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
  void dirty_edges(Phase phase, const std::function<bool(double)>& wanted,
                   std::vector<std::pair<double, std::uint64_t>>& edges);
  void ensure_cell(std::int32_t slot);
  const double* metric(std::int32_t slot) const { return metrics_.data() + 9 * slot; }
  bool vertex_metric(const Stencil& stencil, double* tensor) const;
  double edge_length(std::int32_t a, std::int32_t b) const {
    return length(mesh_.point(a), mesh_.point(b), metric(a), metric(b));
  }
  bool protected_edge(std::int32_t a, std::int32_t b) const {
    return protected_edges_.contains(id_edge(mesh_.vertex_id(a), mesh_.vertex_id(b)));
  }

  // Measurement.
  // Measured edges above (long_side) or below the band, each once, with lengths.
  void out_of_band(bool long_side, std::vector<std::pair<double, std::uint64_t>>& edges) const;
  bool complete() const;

  // Splits.
  std::int64_t split_sweep();
  // A two-facet hull shell can commit locally. Only a declared coplanar
  // pair can use coverage of the whole pair instead of exact edge coverage.
  enum class EdgeSupport { kFree, kBatch, kHull, kDeclaredHull };
  std::optional<EdgeSupport> edge_support(std::int32_t a, std::int32_t b,
                                          std::span<const std::int32_t> shell);
  bool split_parameter(std::int32_t a, std::int32_t b, bool exact_edge, Stencil& stencil,
                       double* point, double& parameter) const;
  bool sync_split_facets(std::int32_t slot, std::int32_t inserted);
  TetEditStatus split(std::int32_t a, std::int32_t b, bool& applied);
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
  bool plan_collapse(std::int32_t removed, std::int32_t kept, CollapsePlan& plan);
  // Commits a collapse: in place of an interior vertex star-locally
  // (LocalIncidence::replace_cavity), of a boundary vertex through
  // LocalIncidence::collapse_boundary, and of an interior vertex into a
  // moving survivor through a one-operation TetMesh::apply_edit_batch.
  TetEditStatus collapse(const CollapsePlan& plan, bool& applied);
  // The facet groups (labels of the constrained faces) through a vertex.
  std::vector<std::int32_t> facet_groups(std::int32_t vertex,
                                         std::span<const std::int32_t> star) const;
  // The vertex lineage of a committed collapse.
  void record_collapse(std::int32_t removed, std::int32_t kept);
  // The vertex's exact stencil and tensor after it moves to `carrier`.
  void assign_carrier(std::int32_t vertex, const Carrier& carrier, const double* tensor);
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
  double quality(const std::array<std::int32_t, 4>& cell) const;
  // The quality of a live cell, kept until the cell or a corner changes.
  double cell_quality(std::int32_t slot);
  // Sets a vertex tensor and its cached log-determinant.
  void set_metric(std::int32_t vertex, const double* tensor);
  // Floor-oriented (float determinant above the floor) and its quality.
  bool oriented_quality(const std::array<std::int32_t, 4>& cell, double& value) const;
  TetEditStatus replace(TetEditKind kind, const std::vector<std::int32_t>& cavity,
                        const std::vector<std::array<std::int32_t, 4>>& cells, bool& applied);
  TetEditStatus try_ring(std::int32_t s, std::int32_t t, bool& applied);
  TetEditStatus try_flip23(std::int32_t slot, int face, bool& applied);
  // Swaps the boundary edge (s t), on the facets (s t c) and (s t d) of one
  // group, for (c d): `shell` becomes `cells` and the facets (s c d) and
  // (t c d), through LocalIncidence::swap_boundary_edge.
  TetEditStatus swap_boundary(const std::vector<std::int32_t>& shell,
                              const std::vector<std::array<std::int32_t, 4>>& cells,
                              std::int32_t s, std::int32_t t, std::int32_t c, std::int32_t d,
                              bool& applied);

  // Relocations of interior vertices to exact source carriers (in place).
  std::int64_t relocate_sweep();
  TetEditStatus relocate(std::int32_t vertex, bool& applied);
  // Moves both interior ends of an out-of-band edge at once, symmetrically
  // along it toward unit length (halved down to 2^-40 of that step), each to
  // a source carrier near its goal, when the joint move decreases the
  // objective over both stars where single moves cannot; committed through
  // TetMesh::apply_relocation_edit.
  TetEditStatus relocate_pair(std::int32_t u, std::int32_t v, bool& applied);
  // The joint moves of the out-of-band edges at `vertices`.
  std::int64_t pair_sweep(std::span<const std::int32_t> vertices);
  bool carrier_simplices(std::int32_t vertex, const LocalIncidence::Star& star, bool& interior,
                         std::vector<CarrierSimplex>& simplices);
  std::optional<SourceWitness> relocation_witness(std::int32_t vertex,
                                                  const Carrier& carrier) const;

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
  std::unique_ptr<CarrierSource> carriers_;
  // Source triangles (rows into the source points) of each facet group, and
  // whether the mesh's source points are the adaptation's (boundary vertices
  // move on them only then).
  std::unordered_map<std::int32_t, std::vector<CarrierSimplex>> group_triangles_;
  std::vector<std::int32_t> witness_face_groups_;
  std::vector<std::int32_t> witness_faces_;
  std::unordered_set<std::int32_t> declared_groups_;
  bool source_banks_match_ = false;
  // Collapses may move their survivor: on once boundary relocation stalls.
  bool survivor_moves_ = false;
  // Boundary vertex relocations start once the interior stalls.
  bool boundary_moves_ = false;
  // Relocation stage: 0 out-of-band vertices, 1 deep, 2 deep and refined
  // (both band improvements only), 3 every vertex, 4 also pairs of ends of
  // out-of-band edges; back to 0 after a topology edit.
  int relocation_stage_ = 0;
  // Failed carrier searches depend on both endpoint states and the shell's
  // coverage rule. A move can make a previously refused edge splittable.
  struct SplitRefusal {
    std::uint64_t first_revision;
    std::uint64_t second_revision;
    EdgeSupport support;
    bool operator==(const SplitRefusal&) const noexcept = default;
  };
  std::vector<std::uint64_t> carrier_revisions_;
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
  if (vertex < 0) return;
  ensure_vertex(vertex);
  const auto at = static_cast<std::size_t>(vertex);
  for (std::size_t phase = 0; phase < dirty_.size(); ++phase) {
    if (dirty_flags_[phase][at] != 0) continue;
    dirty_flags_[phase][at] = 1;
    dirty_[phase].push_back(vertex);
  }
}

void Adapter::dirty_edges(Phase phase, const std::function<bool(double)>& wanted,
                          std::vector<std::pair<double, std::uint64_t>>& edges) {
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
        if (wanted(value)) edges.emplace_back(value, edge_key(vertex, other));
      }
    }
  }
  dirty_[index].clear();
  std::sort(edges.begin(), edges.end(), [](const auto& x, const auto& y) { return x.second < y.second; });
  edges.erase(std::unique(edges.begin(), edges.end(),
                          [](const auto& x, const auto& y) { return x.second == y.second; }),
              edges.end());
  std::erase_if(edges, [&](const auto& entry) {
    return protected_edge(static_cast<std::int32_t>(entry.second >> 32),
                          static_cast<std::int32_t>(entry.second & 0xffffffffU));
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
    next_facet_id_ = std::max(next_facet_id_, facet.id + 1);
  }
  for (const TetSnapshotSegment& segment : snapshot.constrained_segments) {
    next_segment_id_ = std::max(next_segment_id_, segment.id + 1);
  }
  if (!incidence_.initialize(mesh_)) throw std::invalid_argument("The mesh has no incidence.");
  source_banks_match_ = snapshot.source_points == input_.source_points;
  for (std::size_t row = 0; row < snapshot.source_face_groups.size(); ++row) {
    if (9 * row < snapshot.source_face_planes.size() &&
        !std::isnan(snapshot.source_face_planes[9 * row])) {
      declared_groups_.insert(snapshot.source_face_groups[row]);
    }
  }
  if (input_.relocation) {
    if (input_.source_cells.size() % 4 != 0) {
      throw std::invalid_argument("Source cells are four point rows each.");
    }
    // Exact points as integers over one power of two (the carrier source's
    // representation): binary64 values share the denominator 2^shift.
    std::vector<Dyadic> numerators;
    numerators.reserve(input_.source_points.size());
    std::int64_t shift = 0;
    for (double value : input_.source_points) {
      numerators.emplace_back(value);
      if (!numerators.back().is_zero()) shift = std::max(shift, -numerators.back().exponent());
    }
    for (Dyadic& numerator : numerators) numerator = numerator.scaled_power(shift);
    carriers_ = std::make_unique<CarrierSource>(input_.source_points, std::move(numerators),
                                                Dyadic(1.0).scaled_power(shift), input_.source_cells);
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
      group_triangles_[snapshot.source_face_groups[row]].push_back(
          carriers_->prepare_simplex(triangle));
    }
    if (source_banks_match_ &&
        std::any_of(snapshot.vertices.begin(), snapshot.vertices.end(),
                    [](const TetSnapshotVertex& vertex) {
                      return vertex.witness.stratum == SourceStratum::kFacet;
                    })) {
      witness_face_groups_ = std::move(snapshot.source_face_groups);
      witness_faces_ = std::move(snapshot.source_faces);
    }
  }
  for (std::size_t slot = 0; slot < mesh_.vertex_slot_count(); ++slot) {
    if (mesh_.live_vertex(static_cast<std::int32_t>(slot))) touch(static_cast<std::int32_t>(slot));
  }
}

void Adapter::out_of_band(bool long_side, std::vector<std::pair<double, std::uint64_t>>& edges) const {
  // Every cell edge once per cell; the few out-of-band ones are deduplicated.
  edges.clear();
  const auto& tets = mesh_.complex().tets;
  for (std::size_t slot = 0; slot < tets.size(); ++slot) {
    const Tetrahedron& cell = tets[slot];
    if (cell.v[0] == kDeadVertex || is_ghost(cell)) continue;
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const std::int32_t a = cell.v[first];
        const std::int32_t b = cell.v[second];
        const double value = edge_length(a, b);
        if (long_side ? value > kUpperLength : value < kLowerLength) {
          edges.emplace_back(value, edge_key(a, b));
        }
      }
    }
  }
  std::sort(edges.begin(), edges.end(), [](const auto& x, const auto& y) { return x.second < y.second; });
  edges.erase(std::unique(edges.begin(), edges.end(),
                          [](const auto& x, const auto& y) { return x.second == y.second; }),
              edges.end());
  std::erase_if(edges, [&](const auto& entry) {
    return protected_edge(static_cast<std::int32_t>(entry.second >> 32),
                          static_cast<std::int32_t>(entry.second & 0xffffffffU));
  });
}

bool Adapter::complete() const {
  const auto& tets = mesh_.complex().tets;
  for (std::size_t slot = 0; slot < tets.size(); ++slot) {
    const Tetrahedron& cell = tets[slot];
    if (cell.v[0] == kDeadVertex || is_ghost(cell)) continue;
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const std::int32_t a = cell.v[first];
        const std::int32_t b = cell.v[second];
        const double value = edge_length(a, b);
        if (!(value >= input_.lower && value <= kUpperLength) && !protected_edge(a, b)) {
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

bool Adapter::split_parameter(std::int32_t a, std::int32_t b, bool exact_edge,
                              Stencil& stencil, double* point, double& parameter) const {
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
    // The carrier is the correctly rounded exact image of the stencil. On a
    // constrained facet or segment the point must also be the exact edge
    // point, and representable, so the children cover the facet exactly.
    bool admissible = true;
    for (int axis = 0; axis < 3 && admissible; ++axis) {
      Expansion image;
      for (const StencilTerm& term : stencil) {
        image = image + Expansion::product(
                            term.weight,
                            input_.source_points[3 * static_cast<std::size_t>(term.row) + axis]);
      }
      const double rounded = nearest_double(image);
      if (exact_edge) {
        const Expansion edge =
            Expansion(pa[axis]) + Expansion::difference(pb[axis], pa[axis]).scaled(t);
        admissible = (edge - image).is_zero() && (image - Expansion(rounded)).is_zero();
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
  const std::vector<std::int32_t> shell(shell_view.begin(), shell_view.end());
  for (std::int32_t slot : shell) {
    if (static_cast<std::size_t>(slot) < occupied_.size() &&
        occupied_[static_cast<std::size_t>(slot)] != 0) {
      return TetEditStatus::kApplied;  // in this sweep's boundary batch
    }
  }
  const auto support = edge_support(a, b, shell);
  if (!support) return TetEditStatus::kStaleGeneration;
  const std::uint64_t key = edge_key(a, b);
  const SplitRefusal current{carrier_revisions_[static_cast<std::size_t>(a)],
                             carrier_revisions_[static_cast<std::size_t>(b)], *support};
  if (const auto failed = unrepresentable_.find(key); failed != unrepresentable_.end()) {
    if (failed->second == current) return TetEditStatus::kApplied;
    unrepresentable_.erase(failed);
  }

  Stencil stencil;
  std::array<double, 3> point{};
  double parameter = 0.0;
  bool exact_edge = *support != EdgeSupport::kFree;
  bool have_point = split_parameter(a, b, exact_edge, stencil, point.data(), parameter);
  if (!have_point && *support == EdgeSupport::kDeclaredHull) {
    // Preserve exact edge witnesses where possible. Otherwise the kernel
    // proves coverage against the pair using the rounded source image.
    exact_edge = false;
    have_point = split_parameter(a, b, false, stencil, point.data(), parameter);
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
    added.add(length(point.data(), mesh_.point(other), tensor.data(), metric(other)));
  }
  if (!added.measured || !removed.measured) return TetEditStatus::kApplied;
  added.finish();
  if (!objective_decreases(removed, added, false)) return TetEditStatus::kApplied;

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
  std::vector<std::pair<double, std::uint64_t>> long_edges;
  dirty_edges(kSplitPhase, [](double value) { return value > kUpperLength; }, long_edges);
  std::sort(long_edges.begin(), long_edges.end(),
            [](const auto& x, const auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
  std::int64_t applied = 0;
  for (const auto& [value, key] : long_edges) {
    const auto a = static_cast<std::int32_t>(key >> 32);
    const auto b = static_cast<std::int32_t>(key & 0xffffffffU);
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
  bool in_place = true;
  // Screen: a survivor spoke longer than every spoke the removed vertex loses
  // makes the long violations grow unless the survivor already has a longer
  // one; such collapses are passed over without building the star.
  {
    double longest_removed = 0.0;
    double longest_new = 0.0;
    for (std::int32_t slot : removed_star) {
      const Tetrahedron& cell = mesh_.tet(slot);
      for (int corner = 0; corner < 4; ++corner) {
        const std::int32_t other = cell.v[corner];
        if (other == removed || protected_edge(removed, other)) continue;
        longest_removed = std::max(longest_removed, edge_length(removed, other));
        if (other != kept && !protected_edge(kept, other)) {
          longest_new = std::max(longest_new, edge_length(kept, other));
        }
      }
    }
    double above_removed = 0.0;
    double above_new = 0.0;
    double below = 0.0;
    band_quanta(longest_removed, above_removed, below);
    band_quanta(longest_new, above_new, below);
    if (above_new > above_removed) {
      // In place the collapse adds a long spoke; only a survivor move may do.
      if (!survivor_moves_) return false;
      in_place = false;
    }
  }
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
  for (std::size_t row = 2; row < rows.size(); ++row) {
    if (!protected_edge(kept, rows[row])) spokes.push_back(static_cast<std::int64_t>(row));
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
  bool interior = true;
  std::vector<CarrierSimplex> simplices;
  if (!carrier_simplices(kept, kept_star, interior, simplices)) return admitted;
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
                                        input_.source_logarithms.data(), star, simplices, false,
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
  // An affine image that had to be rounded is not in the exact-stencil
  // contract. A refusal for an exact image is authoritative.
  if (plan.moved) {
    if (!plan.carrier.exact) return std::nullopt;
  } else {
    const double* point = mesh_.point(plan.kept);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      Expansion image;
      for (const StencilTerm& term : support) {
        image = image + Expansion::product(
                            term.weight,
                            input_.source_points[3 * static_cast<std::size_t>(term.row) + axis]);
      }
      if (!(image - Expansion(point[axis])).is_zero()) return std::nullopt;
    }
  }
  std::vector<std::int32_t> rows;
  std::vector<double> weights;
  rows.reserve(support.size());
  weights.reserve(support.size());
  for (const StencilTerm& term : support) {
    rows.push_back(term.row);
    weights.push_back(term.weight);
  }
  const detail::SourceStencil stencil{rows, weights};
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
  TetEditStatus status;
  if (local) {
    status = *local;
  } else if (plan.boundary || plan.moved) {
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
    status = incidence_.replace_cavity(mesh_, edit, &removed_star);
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
  if (plan.moved) assign_carrier(kept, plan.carrier, plan.metric.data());
  applied = true;
  return TetEditStatus::kApplied;
}

std::int64_t Adapter::collapse_sweep() {
  // Shortest first at the changed vertices; a round's collapses change the
  // stars around their survivors, which the next round looks at.
  std::vector<std::pair<double, std::uint64_t>> work;
  std::int64_t total = 0;
  for (;;) {
    dirty_edges(kCollapsePhase, [](double value) { return value < kLowerLength; }, work);
    if (work.empty()) break;
    std::sort(work.begin(), work.end());
    for (const auto& [value, key] : work) {
      const auto a = static_cast<std::int32_t>(key >> 32);
      const auto b = static_cast<std::int32_t>(key & 0xffffffffU);
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

TetEditStatus Adapter::try_ring(std::int32_t s, std::int32_t t, bool& applied) {
  applied = false;
  if (protected_edge(s, t)) return TetEditStatus::kApplied;
  const std::span<const std::int32_t> view = incidence_.edge_star(s, t);
  const std::vector<std::int32_t> shell(view.begin(), view.end());
  const std::size_t n = shell.size();
  if (n < 2 || n > 7) return TetEditStatus::kApplied;
  const std::int32_t region = mesh_.complex().region(shell.front());
  const std::int32_t block = cell_blocks_[static_cast<std::size_t>(shell.front())];
  // The ring: the two other corners of each shell cell, chained. An interior
  // edge's ring is closed; a boundary edge on two coplanar facets (s t c) and
  // (s t d) of one group has an open ring from c to d, which the swap closes
  // with the new edge (c d) and its facets (s c d) and (t c d).
  std::vector<std::array<std::int32_t, 2>> links;
  std::vector<std::pair<std::int32_t, std::int32_t>> through;  // facet label, opposite vertex
  double before_quality = std::numeric_limits<double>::infinity();
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
    before_quality = std::min(before_quality, cell_quality(slot));
  }
  const bool closed = through.empty();
  if (!closed && (through.size() != 2 || through[0].first != through[1].first ||
                  through[0].second == through[1].second)) {
    return TetEditStatus::kApplied;
  }
  if (closed && n < 3) return TetEditStatus::kApplied;
  std::vector<std::int32_t> ring;
  std::vector<char> used(n, 0);
  if (closed) {
    ring = {links.front()[0], links.front()[1]};
    used[0] = 1;
  } else {
    ring = {through[0].second};
  }
  while (ring.size() < (closed ? n : n + 1)) {
    bool extended = false;
    for (std::size_t index = 0; index < n && !extended; ++index) {
      if (used[index] != 0) continue;
      for (int side = 0; side < 2; ++side) {
        if (links[index][static_cast<std::size_t>(side)] == ring.back()) {
          ring.push_back(links[index][static_cast<std::size_t>(1 - side)]);
          used[index] = 1;
          extended = true;
          break;
        }
      }
    }
    if (!extended) return TetEditStatus::kApplied;
  }
  if (closed) {
    // The last link joins the ring's ends.
    std::size_t last = 0;
    while (last < n && used[last] != 0) ++last;
    if (last == n) return TetEditStatus::kApplied;
    const auto& link = links[last];
    if (!((link[0] == ring.back() && link[1] == ring.front()) ||
          (link[1] == ring.back() && link[0] == ring.front()))) {
      return TetEditStatus::kApplied;
    }
  } else if (ring.back() != through[1].second) {
    return TetEditStatus::kApplied;
  }
  const std::size_t m = ring.size();
  if (std::set<std::int32_t>(ring.begin(), ring.end()).size() != m) return TetEditStatus::kApplied;

  Violations removed;
  removed.add(edge_length(s, t));
  removed.finish();
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
    double worst = std::numeric_limits<double>::infinity();
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
      worst = std::min({worst, first, second});
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
        const bool admitted = !protected_edge(x, y) && incidence_.edge_star(x, y).empty();
        diagonal_state[key] = admitted ? 1 : 2;
        if (admitted) diagonal_length[key] = edge_length(x, y);
      }
      if (diagonal_state[key] != 1) return;
      option.added.add(diagonal_length[key]);
    }
    if (!option.added.measured) return;
    option.added.finish();
    option.quality = worst;
    if (!best || compare_violations(option.added, best->added) < 0 ||
        (compare_violations(option.added, best->added) == 0 && option.quality > best->quality)) {
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
  if (!objective_decreases(removed, best->added, best->quality > before_quality + kImprovement)) {
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

TetEditStatus Adapter::try_flip23(std::int32_t slot, int face, bool& applied) {
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
  const double before_quality =
      std::min(cell_quality(slot), cell_quality(other));
  std::vector<std::array<std::int32_t, 4>> cells;
  double after_quality = std::numeric_limits<double>::infinity();
  for (int choice : {1, -1}) {
    cells.clear();
    after_quality = std::numeric_limits<double>::infinity();
    for (int side = 0; side < 3; ++side) {
      std::array<std::int32_t, 4> cell{facet[side], facet[(side + 1) % 3], t, s};
      if (choice < 0) std::swap(cell[0], cell[1]);
      double value = 0.0;
      if (!oriented_quality(cell, value)) break;
      after_quality = std::min(after_quality, value);
      cells.push_back(cell);
    }
    if (cells.size() == 3) break;
  }
  if (cells.size() != 3) return TetEditStatus::kApplied;
  Violations removed;
  removed.finish();
  Violations added;
  added.add(edge_length(s, t));
  if (!added.measured) return TetEditStatus::kApplied;
  added.finish();
  if (!objective_decreases(removed, added, after_quality > before_quality + kImprovement)) {
    return TetEditStatus::kApplied;
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
  std::vector<std::pair<std::int32_t, int>> faces;
  std::vector<std::uint64_t> edges;
  std::vector<std::int32_t> star;
  for (std::int32_t vertex : dirty_[kSwapPhase]) {
    dirty_flags_[kSwapPhase][static_cast<std::size_t>(vertex)] = 0;
    if (!mesh_.live_vertex(vertex)) continue;
    incidence_.collect_star(vertex, star);
    for (std::int32_t slot : star) {
      const Tetrahedron& cell = mesh_.tet(slot);
      const bool poor = cell_quality(slot) < kPoor;
      for (int first = 0; first < 4; ++first) {
        if (poor) faces.emplace_back(slot, first);
        for (int second = first + 1; second < 4; ++second) {
          const std::int32_t a = cell.v[first];
          const std::int32_t b = cell.v[second];
          if (poor) {
            edges.push_back(edge_key(a, b));
            continue;
          }
          if (a != vertex && b != vertex) continue;
          const double value = edge_length(a, b);
          if (!(value >= kLowerLength && value <= kUpperLength)) edges.push_back(edge_key(a, b));
        }
      }
    }
  }
  dirty_[kSwapPhase].clear();
  std::sort(faces.begin(), faces.end());
  faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::int64_t applied = 0;
  for (const auto& [slot, face] : faces) {
    bool done = false;
    const TetEditStatus status = try_flip23(slot, face, done);
    if (status != TetEditStatus::kApplied) {
      fatal_ = status;
      return applied;
    }
    applied += done ? 1 : 0;
  }
  for (std::uint64_t key : edges) {
    const auto a = static_cast<std::int32_t>(key >> 32);
    const auto b = static_cast<std::int32_t>(key & 0xffffffffU);
    if (!mesh_.live_vertex(a) || !mesh_.live_vertex(b)) continue;
    bool done = false;
    const TetEditStatus status = try_ring(a, b, done);
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

// The source simplices a boundary vertex moves in: the source triangles of
// its one facet group, the source edges two groups share, or none (a corner,
// or a vertex on a segment); empty with `interior` for an interior vertex.
bool Adapter::carrier_simplices(std::int32_t vertex, const LocalIncidence::Star& star,
                                bool& interior, std::vector<CarrierSimplex>& simplices) {
  simplices.clear();
  interior = mesh_.vertex_dimension(vertex) == 3;
  if (interior) return true;
  if (mesh_.vertex_dimension(vertex) != 2 || !source_banks_match_) return false;
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
  if (groups.size() == 1) {
    const auto found = group_triangles_.find(groups.front());
    if (found == group_triangles_.end()) return false;
    simplices = found->second;
    return !simplices.empty();
  }
  if (groups.size() != 2) return false;
  const auto first = group_triangles_.find(groups[0]);
  const auto second = group_triangles_.find(groups[1]);
  if (first == group_triangles_.end() || second == group_triangles_.end()) return false;
  const auto sides = [](const std::vector<CarrierSimplex>& triangles) {
    std::vector<std::array<std::int64_t, 2>> edges;
    for (const CarrierSimplex& triangle : triangles) {
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
  const auto left = sides(first->second);
  const auto right = sides(second->second);
  std::vector<std::array<std::int64_t, 2>> shared;
  std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                        std::back_inserter(shared));
  for (const auto& edge : shared) {
    CarrierSimplex simplex;
    simplex.rows = {edge[0], edge[1], -1};
    simplex.size = 2;
    simplices.push_back(carriers_->prepare_simplex(simplex));
  }
  return !simplices.empty();
}

std::optional<SourceWitness> Adapter::relocation_witness(std::int32_t vertex,
                                                          const Carrier& carrier) const {
  const SourceWitness& old = mesh_.witness(vertex);
  if (old.stratum == SourceStratum::kNone) return SourceWitness{};
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
    return source_parameters_valid(SourceStratum::kFacet, next.parameters);
  };
  // The dyadic carrier already realizes these exact barycentric weights.
  // Prefer the existing source triangle, then any triangle in its group.
  if (in_face(old_row)) return next;
  for (std::size_t row = 0; row < witness_face_groups_.size(); ++row) {
    if (witness_face_groups_[row] == witness_face_groups_[old_row] && in_face(row)) return next;
  }
  return std::nullopt;
}

TetEditStatus Adapter::relocate(std::int32_t vertex, bool& applied) {
  applied = false;
  if (!carriers_ || mesh_.protected_vertex(vertex) || mesh_.protection_radius(vertex) > 0.0) {
    return TetEditStatus::kApplied;
  }
  const LocalIncidence::Star star = incidence_.vertex_star(vertex);
  if (star.empty()) return TetEditStatus::kApplied;
  bool interior = true;
  std::vector<CarrierSimplex> simplices;
  if (!carrier_simplices(vertex, star, interior, simplices)) return TetEditStatus::kApplied;
  if (!interior && !boundary_moves_) return TetEditStatus::kApplied;
  // Star rows: the vertex, then its link in first-seen order.
  std::vector<std::int32_t> rows{vertex};
  std::vector<std::int64_t> cells;
  double before_quality = std::numeric_limits<double>::infinity();
  for (std::int32_t slot : star) {
    const Tetrahedron& cell = mesh_.tet(slot);
    before_quality = std::min(before_quality, cell_quality(slot));
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
  std::vector<double> lengths(spokes);
  bool outside = false;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    std::copy_n(mesh_.point(rows[row]), 3, points.data() + 3 * row);
    std::copy_n(metric(rows[row]), 9, tensors.data() + 9 * row);
    if (row == 0) continue;
    measured[row - 1] = protected_edge(vertex, rows[row]) ? 0 : 1;
    lengths[row - 1] = edge_length(vertex, rows[row]);
    if (measured[row - 1] != 0 && !(lengths[row - 1] > 0.0 && std::isfinite(lengths[row - 1]))) {
      return TetEditStatus::kApplied;  // a spoke without a finite quantum is not relocated
    }
    outside = outside || (measured[row - 1] != 0 &&
                          !(lengths[row - 1] >= kLowerLength && lengths[row - 1] <= kUpperLength));
  }
  // Until the last stage only a vertex with an out-of-band spoke is pulled:
  // in-band stars are left to the swaps, so relocations do not trickle
  // through the whole mesh.
  if (!outside && relocation_stage_ < 3) return TetEditStatus::kApplied;
  const bool deep = outside && (relocation_stage_ == 1 || relocation_stage_ == 2);
  // A boundary vertex is pulled within the hull of its first carrier simplex.
  std::array<double, 9> basis{};
  std::array<double, 9> projector{};
  std::array<double, 3> origin{};
  int basis_count = 0;
  if (!interior) {
    const CarrierSimplex& hull = simplices.front();
    std::array<double, 9> corners{};
    for (int corner = 0; corner < hull.size; ++corner) {
      std::copy_n(input_.source_points.data() +
                      3 * static_cast<std::size_t>(hull.rows[static_cast<std::size_t>(corner)]),
                  3, corners.data() + 3 * corner);
    }
    basis_count = hull_frame(hull.size, corners.data(), basis.data(), projector.data());
    std::copy_n(corners.data(), 3, origin.data());
  }
  std::vector<double> pulls(3 * (4 * spokes + 8));
  std::int64_t pull_count = -1;
  for (;;) {
    pull_count = relocation_pulls(
        static_cast<std::int64_t>(spokes), points.data(), metric(vertex), lengths.data(),
        measured.data(), deep, interior, basis_count, interior ? nullptr : basis.data(),
        interior ? nullptr : projector.data(), nullptr,
        static_cast<std::int64_t>(pulls.size() / 3), pulls.data());
    if (pull_count >= 0) break;
    pulls.resize(2 * pulls.size());  // a deep search's tangents need more room
  }
  if (pull_count == 0) return TetEditStatus::kApplied;
  RelocationMode mode;
  mode.deep = deep;
  mode.refine = deep && relocation_stage_ == 2;
  mode.band_only = deep;
  RelocationStar relocation_star;
  relocation_star.point_count = static_cast<std::int64_t>(rows.size());
  relocation_star.points = points.data();
  relocation_star.metrics = tensors.data();
  relocation_star.measured = measured.data();
  relocation_star.before_lengths = lengths.data();
  relocation_star.before_quality = before_quality;
  relocation_star.cell_count = static_cast<std::int64_t>(cells.size() / 4);
  relocation_star.cells = cells.data();
  relocation_star.determinant_floor = kDeterminantFloor;
  relocation_star.improvement = kImprovement;
  RelocationSearch search(*carriers_, input_.source_metrics.data(),
                          input_.source_logarithms.data(), relocation_star, pull_count,
                          pulls.data(), interior ? nullptr : origin.data(),
                          interior ? nullptr : projector.data(), std::move(simplices), mode);
  std::vector<RelocationCandidate> group;
  std::vector<double> candidate_lengths;
  for (;;) {
    if (search.next(group, candidate_lengths) != RelocationEnd::kGroup) return TetEditStatus::kApplied;
    for (const RelocationCandidate& candidate : group) {
      TetEditVertex moved;
      moved.id = mesh_.vertex_id(vertex);
      moved.carrier = candidate.carrier.point;
      moved.edge_witness_supplied = true;
      const auto witness = relocation_witness(vertex, candidate.carrier);
      if (!witness) continue;
      moved.witness = *witness;
      TetEditStatus status = TetEditStatus::kRejectedTopology;
      const std::span<const TetEditVertex> records(&moved, 1);
      if (interior || (!mesh_.implicit_source() && moved.witness.stratum != SourceStratum::kNone)) {
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
        const detail::SourceStencil stencil{std::span(source_rows).first(count),
                                             std::span(weights).first(count)};
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
      assign_carrier(vertex, candidate.carrier, candidate.metric.data());
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

// The exact stencil of a carrier: its dyadic weights as binary64 values.
Stencil carrier_stencil(const Carrier& carrier) {
  Stencil stencil;
  for (int term = 0; term < carrier.size; ++term) {
    const auto index = static_cast<std::size_t>(term);
    stencil.push_back({static_cast<std::int32_t>(carrier.rows[index]),
                       std::ldexp(static_cast<double>(carrier.weights[index]), -carrier.bits)});
  }
  return stencil;
}

void Adapter::assign_carrier(std::int32_t vertex, const Carrier& carrier, const double* tensor) {
  // The new exact stencil: the carrier's dyadic weights.
  stencils_[static_cast<std::size_t>(vertex)] = carrier_stencil(carrier);
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

std::int64_t Adapter::relocate_sweep() {
  std::vector<std::int32_t> vertices(dirty_[kRelocatePhase].begin(), dirty_[kRelocatePhase].end());
  for (std::int32_t vertex : vertices) dirty_flags_[kRelocatePhase][static_cast<std::size_t>(vertex)] = 0;
  dirty_[kRelocatePhase].clear();
  std::sort(vertices.begin(), vertices.end());
  std::int64_t applied = 0;
  for (std::int32_t vertex : vertices) {
    if (!mesh_.live_vertex(vertex)) continue;
    bool done = false;
    const TetEditStatus status = relocate(vertex, done);
    if (status != TetEditStatus::kApplied) {
      fatal_ = status;
      return applied;
    }
    applied += done ? 1 : 0;
  }
  if (relocation_stage_ == 4) applied += pair_sweep(vertices);
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
        const std::int32_t a = std::min(cell.v[first], cell.v[second]);
        const std::int32_t b = std::max(cell.v[first], cell.v[second]);
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
        assign_carrier(u, move_u, tensor_u.data());
        assign_carrier(v, move_v, tensor_v.data());
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
  std::vector<std::pair<double, std::uint64_t>> edges;
  std::vector<std::int32_t> star;
  for (std::int32_t vertex : vertices) {
    if (!mesh_.live_vertex(vertex)) continue;
    incidence_.collect_star(vertex, star);
    for (std::int32_t slot : star) {
      for (std::int32_t other : mesh_.tet(slot).v) {
        if (other == vertex || protected_edge(vertex, other)) continue;
        const double value = edge_length(vertex, other);
        if (!(value >= kLowerLength && value <= kUpperLength)) {
          edges.emplace_back(value, edge_key(vertex, other));
        }
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::int64_t applied = 0;
  for (const auto& [value, key] : edges) {
    const auto a = static_cast<std::int32_t>(key >> 32);
    const auto b = static_cast<std::int32_t>(key & 0xffffffffU);
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
  AdaptStatus status = AdaptStatus::kComplete;
  std::int64_t passes = 0;
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
    const bool done = complete();
    if (done) {
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
    // Exhaust the cheap single-vertex interior search before enabling
    // boundary moves, moving survivors and deeper searches.
    if (input_.relocation && !boundary_moves_) {
      boundary_moves_ = true;
    } else if (input_.topology_operations && carriers_ && !survivor_moves_) {
      survivor_moves_ = true;
    } else if (input_.relocation && carriers_ && relocation_stage_ < 4) {
      ++relocation_stage_;
    } else {
      status = AdaptStatus::kStalled;
      break;
    }
    for (std::size_t slot = 0; slot < mesh_.vertex_slot_count(); ++slot) {
      if (mesh_.live_vertex(static_cast<std::int32_t>(slot))) touch(static_cast<std::int32_t>(slot));
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
