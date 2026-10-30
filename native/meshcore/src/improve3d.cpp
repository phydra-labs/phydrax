//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Local constrained flips/reconnections, collapse, and relocation. Radius-edge
// and dihedral targets guide edits; determinant and source constraints are hard.
#include "improve3d.hpp"
#include "local_incidence.hpp"

#include <algorithm>
#include <bit>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <map>
#include <set>
#include <utility>
#include <unordered_set>
#include <vector>

#include "dyadic.hpp"
#include "exact_scalar.hpp"
#include "intersections.hpp"
#include "cavity.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "refine3d.hpp"
#include "predicates.hpp"

namespace phx::mc {
namespace detail {

LocalEdgeKey sorted_local_edge(std::int32_t first, std::int32_t second) noexcept {
  if (second < first) std::swap(first, second);
  return {first, second};
}

std::size_t LocalEdgeHash::operator()(const LocalEdgeKey& edge) const noexcept {
  const std::uint64_t first = static_cast<std::uint32_t>(edge.first);
  const std::uint64_t second = static_cast<std::uint32_t>(edge.second);
  return std::hash<std::uint64_t>{}((first << 32) | second);
}

std::size_t LocalPositionHash::operator()(const LocalPositionKey& point) const noexcept {
  std::size_t seed = 0;
  for (std::uint64_t bits : point.bits) {
    seed ^= std::hash<std::uint64_t>{}(bits) + 0x9e3779b9U + (seed << 6) +
            (seed >> 2);
  }
  return seed;
}

LocalPositionKey local_position_key(const double* point) noexcept {
  LocalPositionKey key;
  for (std::size_t axis = 0; axis < key.bits.size(); ++axis) {
    key.bits[axis] = point[axis] == 0.0
                         ? 0
                         : std::bit_cast<std::uint64_t>(point[axis]);
  }
  return key;
}

void LocalIncidence::reserve(std::size_t vertex_count, std::size_t edge_count) {
  vertex_cells_.reserve(vertex_count);
  edge_cells_.reserve(edge_count);
  vertex_positions_.reserve(vertex_count);
}

bool LocalIncidence::initialize(const TetMesh& mesh,
                                const TetMeshSnapshot& snapshot) {
  vertex_cells_.clear();
  edge_cells_.clear();
  vertex_positions_.clear();
  ++topology_revision_;
  if (snapshot.cells.size() > std::numeric_limits<std::size_t>::max() / 6) {
    return false;
  }
  reserve(snapshot.vertices.size(), 6 * snapshot.cells.size());
  for (const TetSnapshotVertex& vertex : snapshot.vertices) {
    const std::int32_t slot = mesh.vertex_slot(vertex.id);
    if (!mesh.live_vertex(slot) ||
        !insert_vertex_position(slot, mesh.point(slot))) {
      return false;
    }
  }
  for (const TetSnapshotCell& cell : snapshot.cells) {
    const std::int32_t slot = mesh.cell_slot(cell.id);
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return false;
    LocalCell vertices{};
    std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
    if (!insert_logical(mesh, slot, vertices)) return false;
  }
  return true;
}

bool LocalIncidence::insert_vertex_position(std::int32_t vertex,
                                            const double* point) {
  if (vertex < 0 || point == nullptr) return false;
  return vertex_positions_.emplace(local_position_key(point), vertex).second;
}

bool LocalIncidence::erase_vertex_position(
    std::int32_t vertex, const LocalPositionKey& old_key) noexcept {
  const auto found = vertex_positions_.find(old_key);
  if (found == vertex_positions_.end() || found->second != vertex) return false;
  vertex_positions_.erase(found);
  return true;
}

LocalIncidence::PositionStatus LocalIncidence::prepare_position_move(
    std::int32_t vertex, const double* old_point, const double* new_point,
    PositionMove& move) {
  if (vertex < 0 || old_point == nullptr || new_point == nullptr) {
    return PositionStatus::kIndexMismatch;
  }
  move = PositionMove{};
  move.vertex_slot = vertex;
  move.old_key = local_position_key(old_point);
  move.new_key = local_position_key(new_point);
  const auto current = vertex_positions_.find(move.old_key);
  if (current == vertex_positions_.end() || current->second != vertex) {
    return PositionStatus::kIndexMismatch;
  }
  if (move.old_key == move.new_key) {
    move.unchanged = true;
    return PositionStatus::kReady;
  }
  if (vertex_positions_.contains(move.new_key)) return PositionStatus::kDuplicate;

  PositionMap staged;
  staged.reserve(1);
  const auto [node, inserted] = staged.emplace(move.new_key, vertex);
  if (!inserted) return PositionStatus::kIndexMismatch;
  move.staged_node = staged.extract(node);
  return PositionStatus::kReady;
}

bool LocalIncidence::commit_position_move(PositionMove&& move) noexcept {
  if (move.unchanged) return true;
  const auto current = vertex_positions_.find(move.old_key);
  if (current == vertex_positions_.end() || current->second != move.vertex_slot) {
    return false;
  }
  auto old_node = vertex_positions_.extract(current);
  auto inserted = vertex_positions_.insert(std::move(move.staged_node));
  if (inserted.inserted) return true;
  vertex_positions_.insert(std::move(old_node));
  return false;
}

bool LocalIncidence::insert_logical(const TetMesh& mesh, std::int32_t cell,
                                    const LocalCell& vertices) {
  const std::uint64_t cell_id = mesh.cell_id(cell);
  for (std::int32_t vertex : vertices) {
    std::vector<std::int32_t>& cells = vertex_cells_[vertex];
    const auto position = std::lower_bound(
        cells.begin(), cells.end(), cell_id,
        [&](std::int32_t existing, std::uint64_t id) {
          return mesh.cell_id(existing) < id;
        });
    if (position != cells.end() && mesh.cell_id(*position) == cell_id) return false;
    cells.insert(position, cell);
  }
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      std::vector<std::int32_t>& cells =
          edge_cells_[sorted_local_edge(vertices[first], vertices[second])];
      if (std::find(cells.begin(), cells.end(), cell) != cells.end()) return false;
      cells.push_back(cell);
      std::sort(cells.begin(), cells.end());
    }
  }
  ++topology_revision_;
  return true;
}

bool LocalIncidence::erase_cell(std::vector<std::int32_t>& cells,
                                std::int32_t cell) {
  const auto found = std::find(cells.begin(), cells.end(), cell);
  if (found == cells.end()) return false;
  cells.erase(found);
  return true;
}

bool LocalIncidence::erase(std::int32_t cell, const LocalCell& vertices) {
  for (std::int32_t vertex : vertices) {
    const auto found = vertex_cells_.find(vertex);
    if (found == vertex_cells_.end() || !erase_cell(found->second, cell)) return false;
    if (found->second.empty()) vertex_cells_.erase(found);
  }
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      const auto found =
          edge_cells_.find(sorted_local_edge(vertices[first], vertices[second]));
      if (found == edge_cells_.end() || !erase_cell(found->second, cell)) return false;
      if (found->second.empty()) edge_cells_.erase(found);
    }
  }
  ++topology_revision_;
  return true;
}

LocalIncidence::Star LocalIncidence::vertex_star(std::int32_t vertex) const {
  std::vector<std::int32_t> result;
  const auto found = vertex_cells_.find(vertex);
  if (found != vertex_cells_.end()) result = found->second;
  return Star(this, vertex, topology_revision_, std::move(result));
}

std::size_t LocalIncidence::edge_valence(std::int32_t first,
                                         std::int32_t second) const {
  const auto found = edge_cells_.find(sorted_local_edge(first, second));
  return found == edge_cells_.end() ? 0 : found->second.size();
}

std::int32_t LocalIncidence::edge_cell_outside(
    const TetMesh& mesh, std::int32_t first, std::int32_t second,
    std::span<const std::int32_t> removed_slots) const {
  const auto found = edge_cells_.find(sorted_local_edge(first, second));
  if (found == edge_cells_.end()) return -1;
  std::int32_t result = -1;
  std::uint64_t result_id = std::numeric_limits<std::uint64_t>::max();
  for (std::int32_t cell : found->second) {
    if (!mesh.complex().live(cell) || is_ghost(mesh.tet(cell)) ||
        std::find(removed_slots.begin(), removed_slots.end(), cell) !=
            removed_slots.end()) {
      continue;
    }
    const std::uint64_t cell_id = mesh.cell_id(cell);
    if (cell_id < result_id) {
      result = cell;
      result_id = cell_id;
    }
  }
  return result;
}

TetEditStatus LocalIncidence::relocate(
    TetMesh& mesh, std::span<const TetEditVertex> moved_vertices,
    const Star& star) {
  if (mesh.periodic_identity_.bound || moved_vertices.empty()) {
    return TetEditStatus::kRejectedTopology;
  }
  for (const TetEditVertex& vertex : moved_vertices) {
    for (const TetVertexAncestry& ancestor : vertex.ancestry) {
      if (ancestor.source_winding[0] != 0 || ancestor.source_winding[1] != 0 ||
          ancestor.source_winding[2] != 0) {
        return TetEditStatus::kRejectedTopology;
      }
    }
  }
  for (const TetPeriodicOrbit& orbit : mesh.periodic_orbits_) {
    for (const TetPeriodicVertexMember& member : orbit.members) {
      if (std::any_of(moved_vertices.begin(), moved_vertices.end(),
                      [&](const TetEditVertex& vertex) {
                        return vertex.id == member.vertex_id;
                      })) {
        return TetEditStatus::kRejectedTopology;
      }
    }
  }
  if (moved_vertices.size() != 1) return TetEditStatus::kRejectedTopology;
  const TetEditVertex& moved = moved_vertices.front();
  const std::int32_t slot = mesh.vertex_slot(moved.id);
  if (slot < 0 || star.owner_ != this || star.vertex_ != slot ||
      star.revision_ != topology_revision_) {
    return TetEditStatus::kRejectedTopology;
  }
  const auto current_star = vertex_cells_.find(slot);
  if (current_star == vertex_cells_.end() ||
      current_star->second.size() != star.cells_.size()) {
    return TetEditStatus::kRejectedTopology;
  }
  std::uint64_t previous_cell_id = 0;
  bool has_previous_cell = false;
  for (std::int32_t cell : star.cells_) {
    if (!mesh.complex().live(cell) || is_ghost(mesh.tet(cell))) {
      return TetEditStatus::kRejectedTopology;
    }
    const std::uint64_t cell_id = mesh.cell_id(cell);
    if (has_previous_cell && cell_id <= previous_cell_id) {
      return TetEditStatus::kRejectedTopology;
    }
    const Tetrahedron& tet = mesh.tet(cell);
    bool contains_moved = false;
    for (std::int32_t vertex : tet.v) {
      contains_moved = contains_moved || vertex == slot;
    }
    if (!contains_moved) return TetEditStatus::kRejectedTopology;
    previous_cell_id = cell_id;
    has_previous_cell = true;
  }
  PositionMove point_move;
  const PositionStatus position_status = prepare_position_move(
      slot, mesh.point(slot), moved.carrier.data(), point_move);
  if (position_status == PositionStatus::kDuplicate) {
    return TetEditStatus::kInvalidGeometry;
  }
  if (position_status != PositionStatus::kReady) {
    return TetEditStatus::kStaleGeneration;
  }
  const TetEditStatus status =
      mesh.relocate_with_complete_star(moved_vertices, star.cells());
  if (status == TetEditStatus::kApplied && !commit_position_move(std::move(point_move))) {
    return TetEditStatus::kStaleGeneration;
  }
  return status;
}

}  // namespace detail

namespace {

struct CellMeasure {
  double minimum_angle = 0.0;
  double radius_edge_ratio = 0.0;
  double relative_determinant = 0.0;
};
struct PreparedRelocationCell {
  std::array<const double*, 4> fixed_points{};
  std::int32_t moved_corner = -1;
  geometry::RelocationMetricCache metric_cache;
};
using detail::LocalCell;
using detail::LocalEdgeKey;
using detail::LocalIncidence;
using detail::sorted_local_edge;
using FaceKey = std::array<std::int32_t, 3>;
using LogicalEdgeKey = std::pair<std::uint64_t, std::uint64_t>;
struct FlipQuality {
  std::int8_t validity_class = -1;
  double primary = -std::numeric_limits<double>::infinity();
};
using QualityVector = std::vector<FlipQuality>;
void sort_quality_vector(QualityVector& quality);


FlipQuality quality_from_measure(const CellMeasure& measure,
                                 const TetQualityRequest& request) {
  const double angle_score =
      request.minimum_dihedral_degrees == 0.0 ||
              measure.minimum_angle >= request.minimum_dihedral_degrees
          ? 1.0
          : measure.minimum_angle / request.minimum_dihedral_degrees;
  const double radius_score =
      measure.radius_edge_ratio <= request.radius_edge_bound
          ? 1.0
          : request.radius_edge_bound / measure.radius_edge_ratio;
  const double quality_score = std::min(angle_score, radius_score);
  const std::int8_t validity =
      measure.relative_determinant > request.minimum_relative_determinant ? 1 : 0;
  return {validity, quality_score};
}

bool measure_cell(const std::array<const double*, 4>& points, CellMeasure& result) {
  if (orient3d(points[0], points[1], points[2], points[3]) <= 0) return false;
  geometry::TetrahedronShape shape;
  if (!geometry::tetrahedron_shape(points.data(), shape)) return false;
  result.minimum_angle = geometry::minimum_dihedral_degrees(points.data());
  result.radius_edge_ratio = shape.radius_edge_ratio;
  result.relative_determinant = geometry::relative_determinant(points.data());
  return std::isfinite(result.minimum_angle) &&
         std::isfinite(result.radius_edge_ratio) &&
         std::isfinite(result.relative_determinant);
}
bool measure_relocated_cell(
    const std::array<const double*, 4>& points,
    const geometry::RelocationMetricCache& metric_cache, CellMeasure& result) {
  if (orient3d(points[0], points[1], points[2], points[3]) <= 0) return false;
  if (!geometry::relocation_radius_edge_ratio(
          points.data(), metric_cache, result.radius_edge_ratio)) {
    return false;
  }
  result.minimum_angle =
      geometry::relocation_minimum_dihedral_degrees(points.data(), metric_cache);
  result.relative_determinant =
      geometry::relocation_relative_determinant(points.data(), metric_cache);
  return std::isfinite(result.minimum_angle) &&
         std::isfinite(result.radius_edge_ratio) &&
         std::isfinite(result.relative_determinant);
}

bool cell_measure(const TetMesh& mesh, const Tetrahedron& tet, CellMeasure& result) {
  std::array<const double*, 4> points{};
  for (int corner = 0; corner < 4; ++corner) {
    points[corner] = mesh.point(tet.v[corner]);
    if (points[corner] == nullptr) return false;
  }
  return measure_cell(points, result);
}

enum class TraceOperation : std::size_t {
  kFlip23,
  kFlip32,
  kReconnect,
  kCollapse,
  kRelocate,
  kCount,
};

constexpr std::size_t kTraceOperationCount =
    static_cast<std::size_t>(TraceOperation::kCount);

struct TraceQualitySummary {
  std::size_t vertices = 0;
  std::size_t cells = 0;
  std::size_t unmet_radius_edge = 0;
  std::size_t unmet_dihedral = 0;
  std::size_t unmet_determinant = 0;
  double worst_quality = std::numeric_limits<double>::quiet_NaN();
  double second_worst_quality = std::numeric_limits<double>::quiet_NaN();
  double first_percentile_quality = std::numeric_limits<double>::quiet_NaN();
  double maximum_radius_edge = std::numeric_limits<double>::quiet_NaN();
  double minimum_dihedral = std::numeric_limits<double>::quiet_NaN();
  double minimum_determinant = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t quality_vector_hash = 0;
};

class ImproveTrace final {
 public:
  explicit ImproveTrace(const TetQualityRequest& request) noexcept
      : request_(request),
        enabled_(std::getenv("PHYDRAX_TET_IMPROVE_TRACE") != nullptr),
        started_(Clock::now()),
        last_report_(started_) {}

  void candidate(TraceOperation operation) noexcept {
    if (!enabled_) return;
    ++evaluated_[static_cast<std::size_t>(operation)];
  }

  void accepted(TraceOperation operation) noexcept {
    if (!enabled_) return;
    ++accepted_[static_cast<std::size_t>(operation)];
  }

  void boundary_pair() noexcept {
    if (!enabled_) return;
    ++boundary_pairs_total_;
    ++boundary_pairs_interval_;
  }

  bool enabled() const noexcept { return enabled_; }

  void report(const TetMesh& mesh, const char* phase, std::size_t pending,
              bool force = false) {
    if (!enabled_) return;
    const auto now = Clock::now();
    if (!force && now - last_report_ < report_interval_) return;
    TraceQualitySummary summary;
    bool quality_available = false;
    try {
      quality_available = summarize_quality(mesh, summary);
    } catch (const std::bad_alloc&) {
      quality_available = false;
    }
    const auto finished = Clock::now();
    const double pass_seconds =
        std::chrono::duration<double>(finished - last_report_).count();
    const double total_seconds =
        std::chrono::duration<double>(finished - started_).count();
    std::fprintf(
        stderr,
        "TET_IMPROVE_TRACE pass=%llu phase=%s pass_wall_s=%.3f "
        "total_s=%.3f pending=%zu boundary_pairs=%llu/%llu "
        "flip23=%llu/%llu flip32=%llu/%llu reconnect=%llu/%llu "
        "collapse=%llu/%llu relocate=%llu/%llu "
        "quality_available=%d vertices=%zu cells=%zu "
        "objective_valid=%d q_worst=%.9g q_second=%.9g q_p01=%.9g "
        "q_vector_hash=%016llx max_rer=%.9g min_dihedral_deg=%.9g "
        "min_relative_det=%.9g unmet_rer=%zu unmet_dihedral=%zu "
        "unmet_determinant=%zu\n",
        static_cast<unsigned long long>(pass_), phase, pass_seconds,
        total_seconds, pending,
        static_cast<unsigned long long>(boundary_pairs_interval_),
        static_cast<unsigned long long>(boundary_pairs_total_),
        static_cast<unsigned long long>(evaluated_[0]),
        static_cast<unsigned long long>(accepted_[0]),
        static_cast<unsigned long long>(evaluated_[1]),
        static_cast<unsigned long long>(accepted_[1]),
        static_cast<unsigned long long>(evaluated_[2]),
        static_cast<unsigned long long>(accepted_[2]),
        static_cast<unsigned long long>(evaluated_[3]),
        static_cast<unsigned long long>(accepted_[3]),
        static_cast<unsigned long long>(evaluated_[4]),
        static_cast<unsigned long long>(accepted_[4]),
        quality_available ? 1 : 0, summary.vertices, summary.cells,
        summary.unmet_determinant == 0 ? 1 : 0, summary.worst_quality,
        summary.second_worst_quality, summary.first_percentile_quality,
        static_cast<unsigned long long>(summary.quality_vector_hash),
        summary.maximum_radius_edge, summary.minimum_dihedral,
        summary.minimum_determinant, summary.unmet_radius_edge,
        summary.unmet_dihedral, summary.unmet_determinant);
    std::fflush(stderr);
    evaluated_.fill(0);
    accepted_.fill(0);
    boundary_pairs_interval_ = 0;
    last_report_ = finished;
    ++pass_;
  }

 private:
  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::seconds report_interval_{30};

  bool summarize_quality(const TetMesh& mesh, TraceQualitySummary& summary) {
    TetMeshSnapshot snapshot;
    if (mesh.snapshot(snapshot) != TetEditStatus::kApplied) return false;
    summary.vertices = snapshot.vertices.size();
    summary.cells = snapshot.cells.size();
    if (snapshot.cells.empty()) return false;
    quality_scores_.clear();
    quality_scores_.reserve(snapshot.cells.size());
    summary.maximum_radius_edge = 0.0;
    summary.minimum_dihedral = std::numeric_limits<double>::infinity();
    summary.minimum_determinant = std::numeric_limits<double>::infinity();
    for (const TetSnapshotCell& cell : snapshot.cells) {
      const std::int32_t slot = mesh.cell_slot(cell.id);
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return false;
      CellMeasure measure;
      if (!cell_measure(mesh, mesh.tet(slot), measure)) return false;
      summary.maximum_radius_edge =
          std::max(summary.maximum_radius_edge, measure.radius_edge_ratio);
      summary.minimum_dihedral =
          std::min(summary.minimum_dihedral, measure.minimum_angle);
      summary.minimum_determinant =
          std::min(summary.minimum_determinant, measure.relative_determinant);
      if (measure.radius_edge_ratio > request_.radius_edge_bound) {
        ++summary.unmet_radius_edge;
      }
      if (measure.minimum_angle < request_.minimum_dihedral_degrees) {
        ++summary.unmet_dihedral;
      }
      if (measure.relative_determinant <=
          request_.minimum_relative_determinant) {
        ++summary.unmet_determinant;
      }
      const double radius_score =
          measure.radius_edge_ratio <= request_.radius_edge_bound
              ? 1.0
              : request_.radius_edge_bound / measure.radius_edge_ratio;
      const double angle_score =
          request_.minimum_dihedral_degrees == 0.0 ||
                  measure.minimum_angle >= request_.minimum_dihedral_degrees
              ? 1.0
              : measure.minimum_angle / request_.minimum_dihedral_degrees;
      quality_scores_.push_back(std::min(radius_score, angle_score));
    }
    std::sort(quality_scores_.begin(), quality_scores_.end());
    const std::size_t first_percentile =
        (quality_scores_.size() - 1) / 100;
    summary.worst_quality = quality_scores_.front();
    summary.second_worst_quality =
        quality_scores_.size() > 1 ? quality_scores_[1] : quality_scores_.front();
    summary.first_percentile_quality = quality_scores_[first_percentile];
    std::uint64_t quality_hash = 14695981039346656037ULL;
    for (double score : quality_scores_) {
      quality_hash ^= std::bit_cast<std::uint64_t>(score);
      quality_hash *= 1099511628211ULL;
    }
    summary.quality_vector_hash = quality_hash;
    return true;
  }

  const TetQualityRequest& request_;
  bool enabled_ = false;
  Clock::time_point started_;
  Clock::time_point last_report_;
  // Diagnostic record index only; it never changes worklist scheduling.
  std::uint64_t pass_ = 0;
  std::uint64_t boundary_pairs_total_ = 0;
  std::uint64_t boundary_pairs_interval_ = 0;
  std::array<std::uint64_t, kTraceOperationCount> evaluated_{};
  std::array<std::uint64_t, kTraceOperationCount> accepted_{};
  std::vector<double> quality_scores_;
};



bool interior_quality(const TetMesh& mesh, std::span<const std::int32_t> star,
                      const TetQualityRequest& request, QualityVector& quality) {
  quality.clear();
  quality.reserve(star.size());
  for (std::int32_t slot : star) {
    CellMeasure measure;
    if (!cell_measure(mesh, mesh.tet(slot), measure)) return false;
    quality.push_back(quality_from_measure(measure, request));
  }
  sort_quality_vector(quality);
  return !quality.empty();
}

bool collapsed_quality(const TetMesh& mesh, std::span<const std::int32_t> star,
                       std::int32_t removed, std::int32_t survivor,
                       const TetQualityRequest& request,
                       QualityVector& quality,
                       std::vector<TetEditCell>& replacement_cells) {
  quality.clear();
  quality.reserve(star.size());
  replacement_cells.clear();
  for (std::int32_t slot : star) {
    const Tetrahedron& old = mesh.tet(slot);
    int removed_corner = -1;
    bool contains_survivor = false;
    for (int corner = 0; corner < 4; ++corner) {
      if (old.v[corner] == removed) removed_corner = corner;
      if (old.v[corner] == survivor) contains_survivor = true;
    }
    if (removed_corner < 0) return false;
    if (contains_survivor) continue;

    LocalCell local{};
    TetEditCell candidate;
    candidate.id = 0;
    candidate.region = mesh.complex().region(slot);
    candidate.ancestor_cell_ids.push_back(mesh.cell_id(slot));
    for (int corner = 0; corner < 4; ++corner) {
      const std::int32_t vertex = old.v[corner] == removed ? survivor : old.v[corner];
      local[corner] = vertex;
      candidate.vertices[corner] = mesh.vertex_id(vertex);
      const std::int32_t source = mesh.complex().constraint(slot, corner);
      if (corner == removed_corner) {
        candidate.facet_sources[corner] = source;
      } else if (source != kNoConstraint) {
        return false;
      }
    }
    if (!mesh.positive(local.data())) return false;
    std::array<const double*, 4> points{};
    for (int corner = 0; corner < 4; ++corner) {
      points[corner] = mesh.point(local[corner]);
      if (points[corner] == nullptr) return false;
    }
    CellMeasure measure;
    if (!measure_cell(points, measure)) return false;
    quality.push_back(quality_from_measure(measure, request));
    replacement_cells.push_back(candidate);
  }
  sort_quality_vector(quality);
  return !replacement_cells.empty() && quality.size() == replacement_cells.size();
}

bool prepare_relocation_plan(const TetMesh& mesh,
                             std::span<const std::int32_t> star,
                             std::int32_t moved,
                             std::vector<PreparedRelocationCell>& plan) {
  plan.clear();
  if (star.empty()) return false;
  plan.reserve(star.size());
  for (std::int32_t slot : star) {
    const Tetrahedron& tet = mesh.tet(slot);
    plan.emplace_back();
    PreparedRelocationCell& cell = plan.back();
    for (int corner = 0; corner < 4; ++corner) {
      if (tet.v[corner] == moved) {
        if (cell.moved_corner >= 0) return false;
        cell.moved_corner = corner;
      }
      cell.fixed_points[static_cast<std::size_t>(corner)] =
          mesh.point(tet.v[corner]);
      if (cell.fixed_points[static_cast<std::size_t>(corner)] == nullptr) {
        return false;
      }
    }
    if (cell.moved_corner < 0 ||
        !geometry::prepare_relocation_metric_cache(
            cell.fixed_points.data(), cell.moved_corner, cell.metric_cache)) {
      return false;
    }
  }
  return true;
}

bool relocated_quality(std::span<const PreparedRelocationCell> plan,
                       const double* candidate, const TetQualityRequest& request,
                       QualityVector& quality) {
  quality.clear();
  quality.reserve(plan.size());
  if (plan.empty()) return false;
  for (const PreparedRelocationCell& cell : plan) {
    std::array<const double*, 4> points = cell.fixed_points;
    points[static_cast<std::size_t>(cell.moved_corner)] = candidate;
    CellMeasure measure;
    if (!measure_relocated_cell(points, cell.metric_cache, measure)) return false;
    quality.push_back(quality_from_measure(measure, request));
  }
  sort_quality_vector(quality);
  return true;
}

FaceKey sorted_face_key(const std::int32_t* vertices) {
  FaceKey key{vertices[0], vertices[1], vertices[2]};
  std::sort(key.begin(), key.end());
  return key;
}

LogicalEdgeKey sorted_logical_edge(std::uint64_t first, std::uint64_t second) {
  if (second < first) std::swap(first, second);
  return {first, second};
}
bool make_positive_cell(const TetMesh& mesh, LocalCell& cell) {
  if (mesh.positive(cell.data())) return true;
  std::swap(cell[0], cell[1]);
  return mesh.positive(cell.data());
}

FlipQuality local_flip_quality(const TetMesh& mesh, const LocalCell& vertices,
                               const TetQualityRequest& request) {
  std::array<const double*, 4> points{};
  for (int corner = 0; corner < 4; ++corner) {
    if (!mesh.live_vertex(vertices[static_cast<std::size_t>(corner)])) return {};
    points[static_cast<std::size_t>(corner)] =
        mesh.point(vertices[static_cast<std::size_t>(corner)]);
  }
  CellMeasure measure;
  if (!measure_cell(points, measure) || measure.relative_determinant < 0.0) {
    return {};
  }
  return quality_from_measure(measure, request);
}


bool flip_quality_less(const FlipQuality& first, const FlipQuality& second) {
  return first.primary < second.primary;
}

void sort_quality_vector(QualityVector& quality) {
  std::sort(quality.begin(), quality.end(), flip_quality_less);
}

bool improves_quality_vector(std::span<const FlipQuality> before,
                             std::span<const FlipQuality> after) {
  if (before.empty() || after.empty() ||
      std::any_of(after.begin(), after.end(), [](const FlipQuality& quality) {
        return quality.validity_class != 1;
      })) {
    return false;
  }
  return std::lexicographical_compare(
      before.begin(), before.end(), after.begin(), after.end(),
      flip_quality_less);
}

bool cell_quality_meets_target(const FlipQuality& quality) noexcept {
  return quality.validity_class == 1 && quality.primary >= 1.0;
}
bool quality_vector_meets_target(std::span<const FlipQuality> quality) {
  return !quality.empty() &&
         std::all_of(quality.begin(), quality.end(), cell_quality_meets_target);
}

bool quality_vector_for_cells(const TetMesh& mesh,
                              std::span<const LocalCell> cells,
                              const TetQualityRequest& request,
                              QualityVector& quality) {
  quality.clear();
  quality.reserve(cells.size());
  for (const LocalCell& cell : cells) {
    const FlipQuality measure = local_flip_quality(mesh, cell, request);
    if (measure.validity_class < 0) return false;
    quality.push_back(measure);
  }
  sort_quality_vector(quality);
  return !quality.empty();
}

bool quality_vector_for_slots(const TetMesh& mesh,
                              std::span<const std::int32_t> cell_slots,
                              const TetQualityRequest& request,
                              QualityVector& quality) {
  quality.clear();
  quality.reserve(cell_slots.size());
  for (std::int32_t slot : cell_slots) {
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return false;
    LocalCell cell{};
    std::copy_n(mesh.tet(slot).v, 4, cell.begin());
    const FlipQuality measure = local_flip_quality(mesh, cell, request);
    if (measure.validity_class < 0) return false;
    quality.push_back(measure);
  }
  sort_quality_vector(quality);
  return !quality.empty();
}

struct EdgeRemovalScratch {
  std::vector<std::int32_t> removed_slots;
  std::vector<std::int32_t> ring;
  std::vector<LocalCell> replacement_cells;
  std::vector<std::size_t> quality_offsets;
  std::vector<std::int32_t> choices;
  std::vector<FlipQuality> best_quality;
  std::vector<FlipQuality> candidate_quality;
  std::vector<std::pair<std::size_t, std::size_t>> pending;
  std::set<LocalEdgeKey> ring_edges;
  std::array<FlipQuality, 2> triangle_quality{};
  QualityVector before_quality;
  QualityVector reconstructed_quality;
};


bool even_vertex_order(const int32_t* cell, const std::array<std::int32_t, 4>& order) {
  std::array<int, 4> positions{};
  for (int row = 0; row < 4; ++row) {
    int position = -1;
    for (int corner = 0; corner < 4; ++corner) {
      if (cell[corner] == order[static_cast<std::size_t>(row)]) {
        position = corner;
        break;
      }
    }
    positions[static_cast<std::size_t>(row)] = position;
    if (position < 0) return false;
  }
  int inversions = 0;
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      inversions += positions[static_cast<std::size_t>(first)] >
                    positions[static_cast<std::size_t>(second)] ? 1 : 0;
    }
  }
  return inversions % 2 == 0;
}

bool collect_edge_ring(const TetMesh& mesh, std::int32_t start,
                       std::int32_t first, std::int32_t second,
                       std::vector<std::int32_t>& cell_slots,
                       std::vector<std::int32_t>& ring) {
  cell_slots.clear();
  ring.clear();
  const TetrahedralComplex& complex = mesh.complex();
  if (first < 0 || second < 0 || first == second || !complex.live(start) ||
      is_ghost(mesh.tet(start)) || vertex_slot(mesh.tet(start), first) < 0 ||
      vertex_slot(mesh.tet(start), second) < 0) {
    return false;
  }
  const std::int32_t initial = start;
  std::int32_t entry = -1;
  for (;;) {
    if (!complex.live(start) || is_ghost(mesh.tet(start))) {
      return false;
    }
    const Tetrahedron& current = mesh.tet(start);
    std::int32_t other_slots[2] = {-1, -1};
    int other_count = 0;
    for (int slot = 0; slot < 4; ++slot) {
      if (current.v[slot] != first && current.v[slot] != second) {
        if (other_count == 2) {
          return false;
        }
        other_slots[other_count++] = slot;
      }
    }
    if (other_count != 2) {
      return false;
    }
    std::int32_t ring_first = current.v[other_slots[0]];
    std::int32_t ring_second = current.v[other_slots[1]];
    if (!even_vertex_order(current.v, {first, second, ring_first, ring_second})) {
      std::swap(ring_first, ring_second);
    }
    if (entry >= 0 && ring_first != entry) {
      return false;
    }
    const int first_slot = vertex_slot(current, ring_first);
    const int second_slot = vertex_slot(current, ring_second);
    const std::int32_t first_constraint =
        first_slot < 0 ? kNoConstraint : complex.constraint(start, first_slot);
    const std::int32_t second_constraint =
        second_slot < 0 ? kNoConstraint : complex.constraint(start, second_slot);
    if (first_slot < 0 || second_slot < 0 || first_constraint != kNoConstraint ||
        second_constraint != kNoConstraint) {
      return false;
    }
    cell_slots.push_back(start);
    ring.push_back(ring_first);
    const std::int32_t next = current.n[first_slot];
    if (next == initial) break;
    if (!complex.live(next) || is_ghost(mesh.tet(next)) ||
        std::find(cell_slots.begin(), cell_slots.end(), next) != cell_slots.end()) {
      return false;
    }
    entry = ring_second;
    start = next;
  }
  const bool closed_ring = ring.size() >= 3 && ring.size() == cell_slots.size();
  return closed_ring;
}

bool collect_boundary_neighbors(const TetMesh& mesh,
                                std::span<const std::int32_t> removed_slots,
                                std::vector<std::uint64_t>& neighbor_ids) {
  std::unordered_set<std::int32_t> removed(removed_slots.begin(), removed_slots.end());
  std::set<std::uint64_t> unique;
  for (std::int32_t slot : removed_slots) {
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return false;
    for (std::int32_t neighbor : mesh.tet(slot).n) {
      if (removed.contains(neighbor) || !mesh.complex().live(neighbor) ||
          is_ghost(mesh.tet(neighbor))) {
        continue;
      }
      unique.insert(mesh.cell_id(neighbor));
    }
  }
  neighbor_ids.assign(unique.begin(), unique.end());
  return true;
}

TetEditStatus prepare_quality_reconnection(
    const TetMesh& mesh, std::span<const std::int32_t> removed_slots,
    std::span<const LocalCell> local_cells, std::uint64_t first_cell_id,
    TetEditKind kind, TetEdit& operation,
    std::vector<std::uint64_t>& boundary_neighbors) {
  if (removed_slots.empty() || local_cells.empty() ||
      local_cells.size() > std::numeric_limits<std::uint64_t>::max() - first_cell_id) {
    return TetEditStatus::kIndexOverflow;
  }
  const TetrahedralComplex& complex = mesh.complex();
  std::unordered_set<std::int32_t> removed_set;
  removed_set.reserve(removed_slots.size());
  std::vector<std::uint64_t> removed_ids;
  removed_ids.reserve(removed_slots.size());
  std::map<FaceKey, std::int32_t> boundary_sources;
  std::int32_t region = kNoRegion;
  bool first_region = true;
  for (std::int32_t slot : removed_slots) {
    if (!complex.live(slot) || is_ghost(mesh.tet(slot)) || !removed_set.insert(slot).second) {
      return TetEditStatus::kRejectedTopology;
    }
    const std::int32_t current_region = complex.region(slot);
    if (first_region) {
      region = current_region;
      first_region = false;
    } else if (region != current_region) {
      return TetEditStatus::kRejectedTopology;
    }
    removed_ids.push_back(mesh.cell_id(slot));
  }
  std::sort(removed_ids.begin(), removed_ids.end());
  for (std::int32_t slot : removed_slots) {
    const Tetrahedron& tet = mesh.tet(slot);
    for (int facet = 0; facet < 4; ++facet) {
      const std::int32_t neighbor = tet.n[facet];
      const std::int32_t source = complex.constraint(slot, facet);
      if (removed_set.contains(neighbor)) {
        if (source != kNoConstraint) return TetEditStatus::kRejectedProtected;
        continue;
      }
      if (source == kNoConstraint) continue;
      std::int32_t face[3];
      oriented_facet(tet.v, facet, face);
      const auto [found, inserted] = boundary_sources.emplace(sorted_face_key(face), source);
      if (!inserted && found->second != source) return TetEditStatus::kRejectedProtected;
    }
  }
  operation = TetEdit{};
  operation.kind = kind;
  operation.removed_cell_ids = removed_ids;
  operation.replacement_cells.reserve(local_cells.size());
  for (std::size_t row = 0; row < local_cells.size(); ++row) {
    const LocalCell& vertices = local_cells[row];
    if (!mesh.positive(vertices.data())) return TetEditStatus::kInvalidGeometry;
    TetEditCell cell;
    cell.id = first_cell_id + static_cast<std::uint64_t>(row);
    cell.region = region;
    cell.ancestor_cell_ids = removed_ids;
    for (int corner = 0; corner < 4; ++corner) {
      cell.vertices[static_cast<std::size_t>(corner)] =
          mesh.vertex_id(vertices[static_cast<std::size_t>(corner)]);
      std::int32_t face[3];
      oriented_facet(vertices.data(), corner, face);
      const auto source = boundary_sources.find(sorted_face_key(face));
      if (source != boundary_sources.end()) {
        cell.facet_sources[static_cast<std::size_t>(corner)] = source->second;
      }
    }
    operation.replacement_cells.push_back(std::move(cell));
  }
  if (!collect_boundary_neighbors(mesh, removed_slots, boundary_neighbors)) {
    return TetEditStatus::kRejectedTopology;
  }
  return TetEditStatus::kApplied;
}
bool make_face_removal_proposal(
    const TetMesh& mesh, const LocalIncidence& incidence,
    std::int32_t cell_slot, int face_slot, const TetQualityRequest& request,
    const std::set<LogicalEdgeKey>& source_edges,
    std::array<std::int32_t, 2>& removed,
    std::array<LocalCell, 3>& replacement_cells,
    std::array<std::int32_t, 3>& blocked_edge,
    QualityVector& before_quality, QualityVector& after_quality) {
  blocked_edge = {-1, -1, -1};
  const TetrahedralComplex& complex = mesh.complex();
  if (!complex.live(cell_slot) || is_ghost(mesh.tet(cell_slot)) ||
      face_slot < 0 || face_slot > 3 ||
      complex.constraint(cell_slot, face_slot) != kNoConstraint) {
    return false;
  }
  const Tetrahedron& first = mesh.tet(cell_slot);
  const std::int32_t neighbor = first.n[face_slot];
  if (!complex.live(neighbor) || is_ghost(mesh.tet(neighbor))) {
    return false;
  }
  const int back = neighbor_slot(mesh.tet(neighbor), cell_slot);
  if (back < 0 || complex.constraint(neighbor, back) != kNoConstraint ||
      complex.region(cell_slot) != complex.region(neighbor)) {
    return false;
  }
  const Tetrahedron& second = mesh.tet(neighbor);
  const std::int32_t first_apex = first.v[face_slot];
  const std::int32_t second_apex = second.v[back];
  if (first_apex == second_apex ||
      source_edges.contains(sorted_logical_edge(mesh.vertex_id(first_apex),
                                                mesh.vertex_id(second_apex)))) {
    return false;
  }
  removed = {cell_slot, neighbor};
  const std::int32_t existing_edge =
      incidence.edge_cell_outside(mesh, first_apex, second_apex, removed);
  if (existing_edge >= 0) {
    blocked_edge = {existing_edge, first_apex, second_apex};
    return false;
  }

  std::size_t child = 0;
  for (int corner = 0; corner < 4; ++corner) {
    if (corner == back) continue;
    LocalCell cell{};
    std::copy_n(second.v, 4, cell.begin());
    cell[static_cast<std::size_t>(corner)] = first_apex;
    if (!make_positive_cell(mesh, cell)) {
      return false;
    }
    replacement_cells[child++] = cell;
  }
  if (child != replacement_cells.size()) return false;
  if (!quality_vector_for_slots(mesh, removed, request, before_quality) ||
      !quality_vector_for_cells(mesh, replacement_cells, request, after_quality)) {
    return false;
  }
  return improves_quality_vector(before_quality, after_quality);
}

TetEditStatus make_edge_removal_proposal(
    const TetMesh& mesh, const LocalIncidence& incidence, std::int32_t start,
    std::int32_t first, std::int32_t second, const TetQualityRequest& request,
    const std::set<LogicalEdgeKey>& source_edges, EdgeRemovalScratch& scratch) {
  auto& removed_slots = scratch.removed_slots;
  auto& ring = scratch.ring;
  auto& replacement_cells = scratch.replacement_cells;
  auto& quality_offsets = scratch.quality_offsets;
  auto& choices = scratch.choices;
  auto& best_quality = scratch.best_quality;
  auto& candidate_quality = scratch.candidate_quality;
  auto& pending = scratch.pending;
  auto& ring_edges = scratch.ring_edges;
  if (mesh.vertex_id(first) > mesh.vertex_id(second)) std::swap(first, second);
  if (source_edges.contains(sorted_logical_edge(mesh.vertex_id(first),
                                                mesh.vertex_id(second)))) {
    return TetEditStatus::kRejectedTopology;
  }
  removed_slots.clear();
  ring.clear();
  replacement_cells.clear();
  if (!collect_edge_ring(mesh, start, first, second, removed_slots, ring)) {
    return TetEditStatus::kRejectedTopology;
  }
  const std::size_t count = ring.size();
  if (count < 3) return TetEditStatus::kRejectedTopology;
  if (count > std::numeric_limits<std::size_t>::max() / count ||
      count - 1 >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      count - 2 > std::numeric_limits<std::size_t>::max() / 2) {
    return TetEditStatus::kIndexOverflow;
  }
  const std::size_t table_size = count * count;
  const std::size_t replacement_count = 2 * (count - 2);
  if (table_size > quality_offsets.max_size() ||
      table_size > choices.max_size() ||
      replacement_count > replacement_cells.max_size() ||
      replacement_count > candidate_quality.max_size() ||
      count > pending.max_size()) {
    return TetEditStatus::kIndexOverflow;
  }
  std::size_t quality_entry_count = 0;
  for (std::size_t length = 2; length < count; ++length) {
    if (length - 1 > std::numeric_limits<std::size_t>::max() / 2) {
      return TetEditStatus::kIndexOverflow;
    }
    const std::size_t interval_count = count - length;
    const std::size_t interval_quality_count = 2 * (length - 1);
    if (interval_count >
        (std::numeric_limits<std::size_t>::max() - quality_entry_count) /
            interval_quality_count) {
      return TetEditStatus::kIndexOverflow;
    }
    quality_entry_count += interval_count * interval_quality_count;
  }
  if (quality_entry_count > best_quality.max_size()) {
    return TetEditStatus::kIndexOverflow;
  }
  quality_offsets.assign(table_size, std::numeric_limits<std::size_t>::max());
  choices.assign(table_size, -1);
  for (std::size_t index = 0; index + 1 < count; ++index) {
    choices[index * count + index + 1] = 0;
  }
  std::size_t next_quality_offset = 0;
  for (std::size_t length = 2; length < count; ++length) {
    const std::size_t interval_quality_count = 2 * (length - 1);
    for (std::size_t i = 0; i + length < count; ++i) {
      const std::size_t current = i * count + i + length;
      quality_offsets[current] = next_quality_offset;
      next_quality_offset += interval_quality_count;
    }
  }
  if (next_quality_offset != quality_entry_count) {
    return TetEditStatus::kIndexOverflow;
  }
  best_quality.resize(quality_entry_count);
  candidate_quality.reserve(replacement_count);
  replacement_cells.reserve(replacement_count);
  pending.reserve(count - 2);
  ring_edges.clear();
  for (std::size_t index = 0; index < count; ++index) {
    ring_edges.insert(sorted_local_edge(ring[index], ring[(index + 1) % count]));
  }

  const auto triangle_cells = [&](std::size_t i, std::size_t k, std::size_t j,
                                  LocalCell& first_cell, LocalCell& second_cell,
                                  std::array<FlipQuality, 2>& triangle_quality) {
    const std::array<LocalEdgeKey, 3> sides{
        sorted_local_edge(ring[i], ring[k]),
        sorted_local_edge(ring[k], ring[j]),
        sorted_local_edge(ring[i], ring[j])};
    const bool exterior_conflict = std::any_of(
        sides.begin(), sides.end(), [&](const LocalEdgeKey& edge) {
          return !ring_edges.contains(edge) &&
                 incidence.edge_cell_outside(mesh, edge.first, edge.second,
                                             removed_slots) >= 0;
        });
    if (exterior_conflict) return false;
    first_cell = {ring[k], ring[i], ring[j], first};
    second_cell = {ring[i], ring[k], ring[j], second};
    if (!mesh.positive(first_cell.data()) || !mesh.positive(second_cell.data())) {
      return false;
    }
    triangle_quality = {local_flip_quality(mesh, first_cell, request),
                        local_flip_quality(mesh, second_cell, request)};
    if (triangle_quality[0].validity_class != 1 ||
        triangle_quality[1].validity_class != 1) {
      return false;
    }
    if (flip_quality_less(triangle_quality[1], triangle_quality[0])) {
      std::swap(triangle_quality[0], triangle_quality[1]);
    }
    return true;
  };

  const auto interval_span = [&](std::size_t i, std::size_t j) {
    const std::size_t length = j - i;
    if (length < 2) return std::span<const FlipQuality>{};
    const std::size_t offset = quality_offsets[i * count + j];
    return std::span<const FlipQuality>(
        best_quality.data() + offset, 2 * (length - 1));
  };

  for (std::size_t length = 2; length < count; ++length) {
    const std::size_t current_quality_count = 2 * (length - 1);
    for (std::size_t i = 0; i + length < count; ++i) {
      const std::size_t j = i + length;
      const std::size_t current = i * count + j;
      const std::size_t current_offset = quality_offsets[current];
      for (std::size_t k = i + 1; k < j; ++k) {
        const std::size_t left_index = i * count + k;
        const std::size_t right_index = k * count + j;
        if (choices[left_index] < 0 || choices[right_index] < 0) continue;
        const std::span<const FlipQuality> left = interval_span(i, k);
        const std::span<const FlipQuality> right = interval_span(k, j);
        LocalCell first_cell{}, second_cell{};
        std::array<FlipQuality, 2> triangle_quality{};
        if (!triangle_cells(i, k, j, first_cell, second_cell,
                            triangle_quality)) {
          continue;
        }
        candidate_quality.clear();
        std::size_t left_cursor = 0;
        std::size_t right_cursor = 0;
        std::size_t triangle_cursor = 0;
        while (candidate_quality.size() < current_quality_count) {
          const FlipQuality* selected = nullptr;
          int selected_source = -1;
          if (left_cursor < left.size()) {
            selected = &left[left_cursor];
            selected_source = 0;
          }
          if (right_cursor < right.size() &&
              (selected == nullptr ||
               flip_quality_less(right[right_cursor], *selected))) {
            selected = &right[right_cursor];
            selected_source = 1;
          }
          if (triangle_cursor < triangle_quality.size() &&
              (selected == nullptr ||
               flip_quality_less(triangle_quality[triangle_cursor], *selected))) {
            selected = &triangle_quality[triangle_cursor];
            selected_source = 2;
          }
          if (selected == nullptr) return TetEditStatus::kStaleGeneration;
          candidate_quality.push_back(*selected);
          switch (selected_source) {
            case 0:
              ++left_cursor;
              break;
            case 1:
              ++right_cursor;
              break;
            case 2:
              ++triangle_cursor;
              break;
            default:
              return TetEditStatus::kStaleGeneration;
          }
        }
        const std::span<const FlipQuality> current_best(
            best_quality.data() + current_offset, current_quality_count);
        if (choices[current] < 0 ||
            std::lexicographical_compare(
                current_best.begin(), current_best.end(),
                candidate_quality.begin(), candidate_quality.end(),
                flip_quality_less)) {
          std::copy(candidate_quality.begin(), candidate_quality.end(),
                    best_quality.begin() + current_offset);
          choices[current] = static_cast<std::int32_t>(k);
        }
      }
    }
  }
  if (choices[count - 1] < 0 ||
      !quality_vector_for_slots(mesh, removed_slots, request,
                                scratch.before_quality)) {
    return TetEditStatus::kRejectedTopology;
  }
  const std::span<const FlipQuality> after_quality(
      best_quality.data() + quality_offsets[count - 1], replacement_count);
  if (!improves_quality_vector(scratch.before_quality, after_quality)) {
    return TetEditStatus::kRejectedTopology;
  }

  replacement_cells.clear();
  pending.clear();
  pending.push_back({0, count - 1});
  while (!pending.empty()) {
    const auto [i, j] = pending.back();
    pending.pop_back();
    const std::int32_t split = choices[i * count + j];
    if (split <= 0 || static_cast<std::size_t>(split) >= j) {
      return TetEditStatus::kRejectedTopology;
    }
    LocalCell first_cell{}, second_cell{};
    std::array<FlipQuality, 2> triangle_quality{};
    if (!triangle_cells(i, static_cast<std::size_t>(split), j,
                        first_cell, second_cell, triangle_quality)) {
      return TetEditStatus::kRejectedTopology;
    }
    replacement_cells.push_back(first_cell);
    replacement_cells.push_back(second_cell);
    if (static_cast<std::size_t>(split) - i >= 2) {
      pending.push_back({i, static_cast<std::size_t>(split)});
    }
    if (j - static_cast<std::size_t>(split) >= 2) {
      pending.push_back({static_cast<std::size_t>(split), j});
    }
  }
  if (replacement_cells.size() != replacement_count ||
      !quality_vector_for_cells(mesh, replacement_cells, request,
                                scratch.reconstructed_quality) ||
      !improves_quality_vector(scratch.before_quality,
                              scratch.reconstructed_quality)) {
    return TetEditStatus::kRejectedTopology;
  }
  return TetEditStatus::kApplied;
}
bool make_fixed_boundary_star_retriangulation(
    const TetMesh& mesh, const TetMeshSnapshot& state,
    std::uint64_t next_vertex_id, std::uint64_t next_cell_id,
    TetEdit& operation, ImproveTrace& trace) {
  if (mesh.policy() != BoundaryPolicy::kFixed ||
      mesh.periodic_identity().bound || state.cells.empty()) return false;
  using FaceKey64 = std::array<std::uint64_t, 3>;
  using EdgeKey64 = std::array<std::uint64_t, 2>;
  struct BoundaryTriangle {
    std::array<std::int32_t, 3> slots{};
    std::array<std::uint64_t, 3> ids{};
    std::int32_t source = kNoConstraint;
  };
  const auto face_key = [](std::array<std::uint64_t, 3> vertices) {
    std::sort(vertices.begin(), vertices.end());
    return vertices;
  };
  const auto edge_key = [](std::uint64_t first, std::uint64_t second) {
    return first < second ? EdgeKey64{first, second} : EdgeKey64{second, first};
  };
  std::vector<BoundaryTriangle> boundary;
  boundary.reserve(state.constrained_facets.size());
  std::map<FaceKey64, std::int32_t> boundary_sources;
  std::set<std::uint64_t> boundary_vertices;
  std::set<EdgeKey64> boundary_edges;
  std::int32_t region = kNoRegion;
  bool first_cell = true;
  for (const TetSnapshotCell& cell : state.cells) {
    const std::int32_t slot = mesh.cell_slot(cell.id);
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return false;
    const std::int32_t cell_region = mesh.complex().region(slot);
    if (first_cell) {
      region = cell_region;
      first_cell = false;
    } else if (region != cell_region) {
      return false;
    }
    for (int face_slot = 0; face_slot < 4; ++face_slot) {
      const std::int32_t neighbor = mesh.tet(slot).n[face_slot];
      if (!mesh.complex().live(neighbor)) return false;
      const std::int32_t source = mesh.complex().constraint(slot, face_slot);
      if (!is_ghost(mesh.tet(neighbor))) {
        if (source != kNoConstraint) return false;
        continue;
      }
      if (source < 0) return false;
      BoundaryTriangle triangle;
      std::int32_t oriented[3];
      oriented_facet(mesh.tet(slot).v, face_slot, oriented);
      triangle.source = source;
      for (int corner = 0; corner < 3; ++corner) {
        triangle.slots[static_cast<std::size_t>(corner)] = oriented[corner];
        triangle.ids[static_cast<std::size_t>(corner)] = mesh.vertex_id(oriented[corner]);
        boundary_vertices.insert(triangle.ids[static_cast<std::size_t>(corner)]);
      }
      if (!boundary_sources.emplace(face_key(triangle.ids), source).second) return false;
      for (int first = 0; first < 3; ++first) {
        for (int second = first + 1; second < 3; ++second) {
          boundary_edges.insert(edge_key(
              triangle.ids[static_cast<std::size_t>(first)],
              triangle.ids[static_cast<std::size_t>(second)]));
        }
      }
      boundary.push_back(triangle);
    }
  }
  if (boundary.empty() || boundary_sources.size() != state.constrained_facets.size()) {
    return false;
  }
  for (const TetSnapshotFacet& facet : state.constrained_facets) {
    const FaceKey64 key = face_key(facet.vertices);
    const auto source = boundary_sources.find(key);
    if (source == boundary_sources.end() || source->second != facet.source) return false;
  }
  for (const TetSnapshotSegment& segment : state.constrained_segments) {
    if (!boundary_edges.contains(edge_key(segment.vertices[0], segment.vertices[1]))) {
      return false;
    }
  }

  if (boundary_vertices.empty() || !std::has_single_bit(boundary_vertices.size()) ||
      next_vertex_id == std::numeric_limits<std::uint64_t>::max() ||
      boundary.size() > std::numeric_limits<std::uint64_t>::max() - next_cell_id) {
    return false;
  }
  const double weight_value = 1.0 / static_cast<double>(boundary_vertices.size());
  const Dyadic weight(weight_value);
  const Dyadic one(1.0);
  std::array<Dyadic, 3> exact_center{};
  TetEditVertex center;
  center.id = next_vertex_id;
  center.ancestry.reserve(boundary_vertices.size());
  for (std::uint64_t id : boundary_vertices) {
    const std::int32_t slot = mesh.vertex_slot(id);
    if (!mesh.live_vertex(slot) ||
        mesh.witness(slot).stratum != SourceStratum::kNone) return false;
    center.ancestry.push_back({id, {}, weight_value});
    for (int axis = 0; axis < 3; ++axis) {
      exact_center[static_cast<std::size_t>(axis)] =
          exact_center[static_cast<std::size_t>(axis)] +
          Dyadic(mesh.point(slot)[axis]) * weight;
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    center.carrier[static_cast<std::size_t>(axis)] = exact::rounded_quotient(
        exact_center[static_cast<std::size_t>(axis)], one);
  }
  for (const BoundaryTriangle& triangle : boundary) {
    const double* a = mesh.point(triangle.slots[0]);
    const double* b = mesh.point(triangle.slots[1]);
    const double* c = mesh.point(triangle.slots[2]);
    if (orient3d(a, b, c, center.carrier.data()) <= 0) return false;
  }
  // Require a geometrically embedded PLC boundary; only its shared indexed
  // vertices and edges may intersect. This is exact, not a tolerance test.
  std::map<std::uint64_t, std::int64_t> intersection_ids;
  for (std::uint64_t id : boundary_vertices) {
    if (intersection_ids.size() >=
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) return false;
    intersection_ids.emplace(id, static_cast<std::int64_t>(intersection_ids.size()));
  }
  trace.report(mesh, "fixed_boundary_pair_scan", boundary.size(), true);
  const bool tracing = trace.enabled();
  std::uint64_t pair_checks = 0;
  for (std::size_t first = 0; first < boundary.size(); ++first) {
    const BoundaryTriangle& a = boundary[first];
    std::array<const double*, 3> points_a{};
    std::array<std::int64_t, 3> ids_a{};
    for (int corner = 0; corner < 3; ++corner) {
      points_a[static_cast<std::size_t>(corner)] =
          mesh.point(a.slots[static_cast<std::size_t>(corner)]);
      ids_a[static_cast<std::size_t>(corner)] =
          intersection_ids.at(a.ids[static_cast<std::size_t>(corner)]);
    }
    for (std::size_t second = first + 1; second < boundary.size(); ++second) {
      if (tracing) {
        trace.boundary_pair();
        ++pair_checks;
        if ((pair_checks & 0xffffU) == 0) {
          trace.report(mesh, "fixed_boundary_pair_scan", boundary.size());
        }
      }
      const BoundaryTriangle& b = boundary[second];
      std::array<const double*, 3> points_b{};
      std::array<std::int64_t, 3> ids_b{};
      for (int corner = 0; corner < 3; ++corner) {
        points_b[static_cast<std::size_t>(corner)] =
            mesh.point(b.slots[static_cast<std::size_t>(corner)]);
        ids_b[static_cast<std::size_t>(corner)] =
            intersection_ids.at(b.ids[static_cast<std::size_t>(corner)]);
      }
      Contact contact{};
      if (intersect_triangles(points_a.data(), points_b.data(), ids_a.data(), ids_b.data(),
                              contact) != PHX_MC_OK ||
          (contact.kind != PHX_MC_DISJOINT &&
           contact.kind != PHX_MC_SHARED_VERTEX &&
           contact.kind != PHX_MC_SHARED_EDGE)) return false;
    }
  }
  trace.report(mesh, "fixed_boundary_pair_complete", boundary.size(), true);


  operation = TetEdit{};
  operation.kind = TetEditKind::kRetriangulate;
  operation.removed_cell_ids.reserve(state.cells.size());
  for (const TetSnapshotCell& cell : state.cells) {
    operation.removed_cell_ids.push_back(cell.id);
  }
  std::sort(operation.removed_cell_ids.begin(), operation.removed_cell_ids.end());
  for (const TetSnapshotVertex& vertex : state.vertices) {
    if (boundary_vertices.contains(vertex.id)) continue;
    const std::int32_t slot = mesh.vertex_slot(vertex.id);
    if (!mesh.live_vertex(slot) || mesh.protected_vertex(slot) ||
        mesh.protection_radius(slot) > 0.0 || mesh.vertex_dimension(slot) != 3 ||
        mesh.witness(slot).stratum != SourceStratum::kNone) return false;
    operation.removed_vertex_ids.push_back(vertex.id);
  }
  std::sort(operation.removed_vertex_ids.begin(), operation.removed_vertex_ids.end());
  operation.changed_vertices.push_back(center);
  operation.replacement_cells.reserve(boundary.size());
  for (std::size_t row = 0; row < boundary.size(); ++row) {
    const BoundaryTriangle& triangle = boundary[row];
    TetEditCell child;
    child.id = next_cell_id + static_cast<std::uint64_t>(row);
    child.region = region;
    child.vertices = {triangle.ids[0], triangle.ids[1], triangle.ids[2], center.id};
    child.facet_sources[3] = triangle.source;
    // A fan cell spans the original cavity, not one exact parent simplex.
    operation.replacement_cells.push_back(std::move(child));
  }
  return true;
}
}  // namespace

std::int32_t improve_mesh(TetMesh& mesh, const TetQualityRequest& request,
                          std::int64_t& collapsed_vertices,
                          std::int64_t& relocated_vertices,
                          std::int64_t& unmet_cells) {
  collapsed_vertices = 0;
  relocated_vertices = 0;
  unmet_cells = 0;
  if (!std::isfinite(request.radius_edge_bound) ||
      request.radius_edge_bound <= 0.0 ||
      !std::isfinite(request.minimum_dihedral_degrees) ||
      request.minimum_dihedral_degrees < 0.0 ||
      request.minimum_dihedral_degrees >= 70.5 ||
      !std::isfinite(request.minimum_relative_determinant) ||
      request.minimum_relative_determinant < 0.0 ||
      request.minimum_relative_determinant >= 1.0) {
    return PHX_MC_INVALID_ARGUMENT;
  }

  ImproveTrace trace(request);
  TetMeshSnapshot initial;
  if (mesh.snapshot(initial) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  trace.report(mesh, "initial", initial.cells.size(), true);
  bool has_unmet_quality = false;
  QualityVector before_cavity_quality;
  before_cavity_quality.reserve(initial.cells.size());
  std::size_t initial_cell_index = 0;
  for (const TetSnapshotCell& cell : initial.cells) {
    const std::int32_t slot = mesh.cell_slot(cell.id);
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return PHX_MC_INTERNAL_ERROR;
    LocalCell vertices{};
    std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
    const FlipQuality quality = local_flip_quality(mesh, vertices, request);
    if (quality.validity_class < 0) return PHX_MC_INVALID_INPUT;
    before_cavity_quality.push_back(quality);
    has_unmet_quality = has_unmet_quality ||
                        quality.validity_class != 1 || quality.primary < 1.0;
    ++initial_cell_index;
    if ((initial_cell_index & 0xfffU) == 0) {
      trace.report(mesh, "initial_quality_scan",
                   initial.cells.size() - initial_cell_index);
    }
  }
  sort_quality_vector(before_cavity_quality);
  trace.report(mesh, "initial_quality_complete", initial.cells.size(), true);
  if (has_unmet_quality) {
    TetEdit operation;
    if (make_fixed_boundary_star_retriangulation(
            mesh, initial, mesh.next_vertex_id_, mesh.next_cell_id_, operation,
            trace)) {
      QualityVector after_cavity_quality;
      after_cavity_quality.reserve(operation.replacement_cells.size());
      for (const TetEditCell& cell : operation.replacement_cells) {
        std::array<const double*, 4> points{};
        for (int corner = 0; corner < 3; ++corner) {
          const std::int32_t slot = mesh.vertex_slot(
              cell.vertices[static_cast<std::size_t>(corner)]);
          if (!mesh.live_vertex(slot)) return PHX_MC_INTERNAL_ERROR;
          points[static_cast<std::size_t>(corner)] = mesh.point(slot);
        }
        points[3] = operation.changed_vertices.front().carrier.data();
        CellMeasure measure;
        if (!measure_cell(points, measure)) return PHX_MC_INVALID_INPUT;
        after_cavity_quality.push_back(quality_from_measure(measure, request));
      }
      sort_quality_vector(after_cavity_quality);
      if (improves_quality_vector(before_cavity_quality, after_cavity_quality)) {
        TetEditBatch batch;
        batch.expected_generation = mesh.generation();
        batch.source_revision = mesh.source_revision();
        batch.frame_id = mesh.frame_id();
        batch.identity_bound = mesh.identity_bound();
        batch.source_id = mesh.source_id();
        batch.source_map_id = mesh.source_map_id();
        batch.periodic_identity = mesh.periodic_identity();
        batch.operations.push_back(std::move(operation));
        std::vector<TetEditResult> results;
        const TetEditStatus status = mesh.apply_edit_batch(batch, results);
        if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
        if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
        if (status == TetEditStatus::kStaleGeneration) return PHX_MC_INTERNAL_ERROR;
        if (status == TetEditStatus::kApplied) {
          if (results.size() != 1 || results.front().status != TetEditStatus::kApplied) {
            return PHX_MC_INTERNAL_ERROR;
          }
          if (mesh.snapshot(initial) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
        }
      }
    }
  }
  trace.report(mesh, "topology_start", initial.cells.size(), true);
  std::vector<std::uint64_t> vertex_ids;
  vertex_ids.reserve(initial.vertices.size());
  for (const TetSnapshotVertex& vertex : initial.vertices) vertex_ids.push_back(vertex.id);
  std::sort(vertex_ids.begin(), vertex_ids.end());
  LocalIncidence incidence;
  if (initial.cells.size() >
      std::numeric_limits<std::size_t>::max() / 6) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  // The only vertex-creating retessellation runs before this index is built.
  // Later flips alter cells, collapse removes one point, and relocation moves one.
  if (!incidence.initialize(mesh, initial)) return PHX_MC_INTERNAL_ERROR;
  std::set<LogicalEdgeKey> source_edges;
  for (const TetSnapshotSegment& segment : initial.constrained_segments) {
    source_edges.insert(sorted_logical_edge(segment.vertices[0], segment.vertices[1]));
  }
  std::set<LogicalEdgeKey> rejected_high_edges;

  const auto apply_quality_flip = [&] (
      std::span<const std::int32_t> removed_slots,
      std::span<const LocalCell> replacement_cells, TetEditKind kind,
      std::vector<std::uint64_t>& affected_cells) {
    std::vector<LocalCell> removed_vertices;
    removed_vertices.reserve(removed_slots.size());
    for (std::int32_t slot : removed_slots) {
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) {
        return TetEditStatus::kStaleGeneration;
      }
      LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      removed_vertices.push_back(vertices);
    }
    TetEdit operation;
    std::vector<std::uint64_t> boundary_neighbors;
    const TetEditStatus prepared = prepare_quality_reconnection(
        mesh, removed_slots, replacement_cells, mesh.next_cell_id_, kind,
        operation, boundary_neighbors);
    if (prepared != TetEditStatus::kApplied) {
      return prepared;
    }
    // Reconnection edits preserve the vertex set indexed by LocalIncidence.
    if (!operation.changed_vertices.empty() || !operation.removed_vertex_ids.empty()) {
      return TetEditStatus::kStaleGeneration;
    }

    TetEditBatch batch;
    batch.expected_generation = mesh.generation();
    batch.source_revision = mesh.source_revision();
    batch.frame_id = mesh.frame_id();
    batch.identity_bound = mesh.identity_bound();
    batch.source_id = mesh.source_id();
    batch.source_map_id = mesh.source_map_id();
    batch.periodic_identity = mesh.periodic_identity();
    batch.operations.push_back(std::move(operation));
    std::vector<TetEditResult> results;
    const TetEditStatus status = mesh.apply_edit_batch(batch, results);
    if (status != TetEditStatus::kApplied) return status;
    if (results.size() != 1 || results.front().status != TetEditStatus::kApplied) {
      return TetEditStatus::kStaleGeneration;
    }
    std::vector<std::uint64_t> touched;
    const TetEdit& committed = batch.operations.front();
    for (std::size_t index = 0; index < removed_slots.size(); ++index) {
      if (!incidence.erase(removed_slots[index], removed_vertices[index])) {
        return TetEditStatus::kStaleGeneration;
      }
    }
    for (const TetEditCell& cell : committed.replacement_cells) {
      const std::int32_t slot = mesh.cell_slot(cell.id);
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) {
        return TetEditStatus::kStaleGeneration;
      }
      LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      if (!incidence.insert_logical(mesh, slot, vertices)) {
        return TetEditStatus::kStaleGeneration;
      }
    }
    touched.reserve(committed.replacement_cells.size() + boundary_neighbors.size());
    for (const TetEditCell& cell : committed.replacement_cells) touched.push_back(cell.id);
    touched.insert(touched.end(), boundary_neighbors.begin(), boundary_neighbors.end());
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    affected_cells.swap(touched);
    rejected_high_edges.clear();
    return TetEditStatus::kApplied;
  };

  const auto fatal_edit_error = [](TetEditStatus status) {
    if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
    if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
    if (status == TetEditStatus::kStaleGeneration) return PHX_MC_INTERNAL_ERROR;
    return PHX_MC_OK;
  };
  EdgeRemovalScratch edge_scratch;
  const auto try_edge_removal = [&](std::int32_t first, std::int32_t second,
                                   std::vector<std::uint64_t>& affected) {
    const std::span<const std::int32_t> no_removed;
    const std::int32_t start =
        incidence.edge_cell_outside(mesh, first, second, no_removed);
    if (start < 0) return TetEditStatus::kRejectedTopology;
    const TraceOperation trace_operation =
        incidence.edge_valence(first, second) == 3
            ? TraceOperation::kFlip32
            : TraceOperation::kReconnect;
    trace.candidate(trace_operation);
    const TetEditStatus proposal = make_edge_removal_proposal(
        mesh, incidence, start, first, second, request, source_edges,
        edge_scratch);
    if (proposal != TetEditStatus::kApplied) return proposal;
    const TetEditKind kind = edge_scratch.removed_slots.size() == 3
                                 ? TetEditKind::kFlip32
                                 : TetEditKind::kRingReconnect;
    const TetEditStatus status = apply_quality_flip(
        edge_scratch.removed_slots, edge_scratch.replacement_cells, kind,
        affected);
    if (status == TetEditStatus::kApplied) trace.accepted(trace_operation);
    return status;
  };



  // The ordered worklist contains only cells that currently violate a target.
  // Topology edits requeue only their changed star, filtered by the same test.
  std::set<std::uint64_t> pending_cells;
  QualityVector face_before_quality;
  QualityVector face_after_quality;
  face_before_quality.reserve(2);
  face_after_quality.reserve(3);
  const auto enqueue_violating_cell = [&](std::uint64_t id) {
    const std::int32_t slot = mesh.cell_slot(id);
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return true;
    LocalCell vertices{};
    std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
    const FlipQuality quality = local_flip_quality(mesh, vertices, request);
    if (quality.validity_class < 0) return false;
    if (!cell_quality_meets_target(quality)) pending_cells.insert(id);
    return true;
  };
  const auto enqueue_affected = [&](std::span<const std::uint64_t> ids) {
    for (std::uint64_t id : ids) {
      if (!enqueue_violating_cell(id)) return false;
    }
    return true;
  };
  for (const TetSnapshotCell& cell : initial.cells) {
    if (!enqueue_violating_cell(cell.id)) return PHX_MC_INVALID_INPUT;
  }
  while (!pending_cells.empty()) {
    const std::uint64_t logical_id = *pending_cells.begin();
    pending_cells.erase(pending_cells.begin());
    trace.report(mesh, "topology_worklist", pending_cells.size());
    const std::int32_t slot = mesh.cell_slot(logical_id);
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) continue;
    LocalCell current{};
    std::copy_n(mesh.tet(slot).v, 4, current.begin());
    const FlipQuality quality = local_flip_quality(mesh, current, request);
    if (quality.validity_class < 0) return PHX_MC_INTERNAL_ERROR;
    if (cell_quality_meets_target(quality)) continue;

    struct FaceOrder {
      std::array<std::uint64_t, 3> vertices{};
      int slot = 0;
    };
    std::array<FaceOrder, 4> faces{};
    for (int face_slot = 0; face_slot < 4; ++face_slot) {
      std::int32_t local_face[3];
      oriented_facet(mesh.tet(slot).v, face_slot, local_face);
      FaceOrder& face = faces[static_cast<std::size_t>(face_slot)];
      face.slot = face_slot;
      for (int corner = 0; corner < 3; ++corner) {
        face.vertices[static_cast<std::size_t>(corner)] =
            mesh.vertex_id(local_face[corner]);
      }
      std::sort(face.vertices.begin(), face.vertices.end());
    }
    std::sort(faces.begin(), faces.end(), [](const FaceOrder& first,
                                             const FaceOrder& second) {
      return first.vertices < second.vertices;
    });

    struct EdgeOrder {
      LogicalEdgeKey key{};
      std::int32_t first = -1;
      std::int32_t second = -1;
    };
    std::array<EdgeOrder, 6> edges{};
    std::size_t edge_count = 0;
    const Tetrahedron& current_tet = mesh.tet(slot);
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        EdgeOrder& edge = edges[edge_count++];
        edge.first = current_tet.v[first];
        edge.second = current_tet.v[second];
        if (mesh.vertex_id(edge.first) > mesh.vertex_id(edge.second)) {
          std::swap(edge.first, edge.second);
        }
        edge.key = sorted_logical_edge(mesh.vertex_id(edge.first),
                                       mesh.vertex_id(edge.second));
      }
    }
    std::sort(edges.begin(), edges.end(), [](const EdgeOrder& first,
                                             const EdgeOrder& second) {
      return first.key < second.key;
    });

    bool changed = false;
    // Larger closed rings can repair a cavity that no 2-3 flip can improve.
    // Try them first, then memoize failures until any topology edit changes
    // the local incidence and quality state.
    for (const EdgeOrder& edge : edges) {
      if (incidence.edge_valence(edge.first, edge.second) <= 4 ||
          rejected_high_edges.contains(edge.key)) {
        continue;
      }
      std::vector<std::uint64_t> affected;
      const TetEditStatus status =
          try_edge_removal(edge.first, edge.second, affected);
      const std::int32_t error = fatal_edit_error(status);
      if (error != PHX_MC_OK) return error;
      if (status == TetEditStatus::kApplied) {
        if (!enqueue_affected(affected)) return PHX_MC_INTERNAL_ERROR;
        changed = true;
        break;
      }
      rejected_high_edges.insert(edge.key);
    }
    if (changed) continue;

    for (const FaceOrder& face : faces) {
      std::array<std::int32_t, 2> removed{};
      std::array<LocalCell, 3> replacements{};
      std::array<std::int32_t, 3> blocked_edge{-1, -1, -1};
      trace.candidate(TraceOperation::kFlip23);
      if (!make_face_removal_proposal(
              mesh, incidence, slot, face.slot, request, source_edges,
              removed, replacements, blocked_edge, face_before_quality,
              face_after_quality)) {
        if (blocked_edge[0] >= 0) {
          const LogicalEdgeKey key = sorted_logical_edge(
              mesh.vertex_id(blocked_edge[1]), mesh.vertex_id(blocked_edge[2]));
          const std::size_t valence =
              incidence.edge_valence(blocked_edge[1], blocked_edge[2]);
          if (valence <= 4 || !rejected_high_edges.contains(key)) {
            std::vector<std::uint64_t> affected;
            const TetEditStatus status =
                try_edge_removal(blocked_edge[1], blocked_edge[2], affected);
            const std::int32_t error = fatal_edit_error(status);
            if (error != PHX_MC_OK) return error;
            if (status == TetEditStatus::kApplied) {
              if (!enqueue_affected(affected)) return PHX_MC_INTERNAL_ERROR;
              changed = true;
              break;
            }
            if (valence > 4) rejected_high_edges.insert(key);
          }
        }
        continue;
      }
      std::vector<std::uint64_t> affected;
      const TetEditStatus status = apply_quality_flip(
          removed, replacements, TetEditKind::kFlip23, affected);
      const std::int32_t error = fatal_edit_error(status);
      if (error != PHX_MC_OK) return error;
      if (status == TetEditStatus::kApplied) {
        trace.accepted(TraceOperation::kFlip23);
        if (!enqueue_affected(affected)) return PHX_MC_INTERNAL_ERROR;
        changed = true;
        break;
      }
    }
    if (changed) continue;

    for (const EdgeOrder& edge : edges) {
      if (incidence.edge_valence(edge.first, edge.second) > 4) continue;
      std::vector<std::uint64_t> affected;
      const TetEditStatus status =
          try_edge_removal(edge.first, edge.second, affected);
      const std::int32_t error = fatal_edit_error(status);
      if (error != PHX_MC_OK) return error;
      if (status == TetEditStatus::kApplied) {
        if (!enqueue_affected(affected)) return PHX_MC_INTERNAL_ERROR;
        break;
      }
    }
  }
  trace.report(mesh, "topology_complete", 0, true);


  // Revisit every changed star so a collapse or relocation cannot leave
  // earlier vertices with stale local quality.
  std::set<std::uint64_t> pending_vertices(vertex_ids.begin(), vertex_ids.end());
  std::vector<PreparedRelocationCell> relocation_plan;
  QualityVector before_quality;
  QualityVector after_quality;
  std::vector<TetEditCell> replacement_cells;
  while (!pending_vertices.empty()) {
    const auto first_vertex = pending_vertices.begin();
    const std::uint64_t vertex_id = *first_vertex;
    pending_vertices.erase(first_vertex);
    trace.report(mesh, "vertex_worklist", pending_vertices.size());
    const std::int32_t removed = mesh.vertex_slot(vertex_id);
    if (!mesh.live_vertex(removed) || mesh.vertex_dimension(removed) != 3 ||
        mesh.protected_vertex(removed) || mesh.protection_radius(removed) != 0.0 ||
        mesh.witness(removed).stratum != SourceStratum::kNone) {
      continue;
    }
    LocalIncidence::Star star = incidence.vertex_star(removed);
    if (!interior_quality(mesh, star.cells(), request, before_quality)) continue;
    if (quality_vector_meets_target(before_quality)) continue;

    std::unordered_set<std::int32_t> neighbor_set;
    for (std::int32_t cell : star) {
      for (std::int32_t vertex : mesh.tet(cell).v) {
        if (vertex != removed) neighbor_set.insert(vertex);
      }
    }
    std::vector<std::int32_t> neighbors(neighbor_set.begin(), neighbor_set.end());
    std::sort(neighbors.begin(), neighbors.end(), [&](std::int32_t a, std::int32_t b) {
      return mesh.vertex_id(a) < mesh.vertex_id(b);
    });

    bool changed = false;
    for (std::int32_t survivor : neighbors) {
      if (mesh.protected_vertex(survivor) || mesh.protection_radius(survivor) > 0.0 ||
          mesh.vertex_dimension(survivor) != 3) {
        continue;
      }
      trace.candidate(TraceOperation::kCollapse);
      if (!collapsed_quality(mesh, star.cells(), removed, survivor, request,
                             after_quality, replacement_cells) ||
          !improves_quality_vector(before_quality, after_quality) ||
          mesh.next_cell_id_ > std::numeric_limits<std::uint64_t>::max() -
                                   replacement_cells.size()) {
        continue;
      }
      for (std::size_t index = 0; index < replacement_cells.size(); ++index) {
        replacement_cells[index].id = mesh.next_cell_id_ + index;
      }
      std::vector<LocalCell> removed_vertices;
      removed_vertices.reserve(star.size());
      for (std::int32_t slot : star) {
        LocalCell vertices{};
        std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
        removed_vertices.push_back(vertices);
      }
      std::vector<std::uint64_t> removed_cell_ids;
      removed_cell_ids.reserve(star.size());
      for (std::int32_t slot : star) removed_cell_ids.push_back(mesh.cell_id(slot));
      const detail::LocalPositionKey removed_position =
          detail::local_position_key(mesh.point(removed));
      const TetEditStatus status = mesh.collapse_vertex(
          vertex_id, mesh.vertex_id(survivor), removed_cell_ids, replacement_cells, {}, {});
      if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
      if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
      if (status == TetEditStatus::kApplied) {
        trace.accepted(TraceOperation::kCollapse);
        for (std::size_t index = 0; index < star.size(); ++index) {
          if (!incidence.erase(star[index], removed_vertices[index])) {
            return PHX_MC_INTERNAL_ERROR;
          }
        }
        for (const TetEditCell& cell : replacement_cells) {
          const std::int32_t slot = mesh.cell_slot(cell.id);
          if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) {
            return PHX_MC_INTERNAL_ERROR;
          }
          LocalCell vertices{};
          std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
          if (!incidence.insert_logical(mesh, slot, vertices)) return PHX_MC_INTERNAL_ERROR;
        }
        if (!incidence.erase_vertex_position(removed, removed_position)) {
          return PHX_MC_INTERNAL_ERROR;
        }
        for (const LocalCell& vertices : removed_vertices) {
          for (std::int32_t vertex : vertices) {
            pending_vertices.insert(mesh.vertex_id(vertex));
          }
        }
        for (const TetEditCell& cell : replacement_cells) {
          pending_vertices.insert(cell.vertices.begin(), cell.vertices.end());
        }
        ++collapsed_vertices;
        changed = true;
        break;
      }
    }
    if (changed) continue;

    std::vector<double> target(3, 0.0);
    std::size_t target_count = 0;
    for (std::int32_t neighbor : neighbors) {
      const double* point = mesh.point(neighbor);
      for (int axis = 0; axis < 3; ++axis) target[axis] += point[axis];
      ++target_count;
    }
    if (target_count == 0) continue;
    for (double& value : target) value /= static_cast<double>(target_count);
    if (!prepare_relocation_plan(mesh, star.cells(), removed, relocation_plan)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const double* original = mesh.point(removed);
    double fraction = 1.0;
    while (fraction > 0.0) {
      double candidate[3];
      bool distinct = false;
      for (int axis = 0; axis < 3; ++axis) {
        candidate[axis] = original[axis] + fraction * (target[axis] - original[axis]);
        distinct = distinct || candidate[axis] != original[axis];
      }
      if (!distinct) break;
      trace.candidate(TraceOperation::kRelocate);
      if (relocated_quality(relocation_plan, candidate, request, after_quality) &&
          improves_quality_vector(before_quality, after_quality)) {
        TetEditVertex moved;
        moved.id = vertex_id;
        std::copy_n(candidate, 3, moved.carrier.begin());
        moved.witness = mesh.witness(removed);
        moved.protection_radius = mesh.protection_radius(removed);
        moved.protected_vertex = mesh.protected_vertex(removed);
        const TetEditStatus status =
            incidence.relocate(mesh, std::span(&moved, 1), star);
        if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
        if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
        if (status == TetEditStatus::kApplied) {
          trace.accepted(TraceOperation::kRelocate);
          for (std::int32_t slot : star) {
            for (std::int32_t vertex : mesh.tet(slot).v) {
              pending_vertices.insert(mesh.vertex_id(vertex));
            }
          }
          ++relocated_vertices;
          changed = true;
          break;
        }
      }
      const double next = fraction * 0.5;
      if (next == fraction) break;
      fraction = next;
    }
  }

  TetMeshSnapshot final_state;
  if (mesh.snapshot(final_state) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  trace.report(mesh, "complete", final_state.cells.size(), true);
  std::vector<TetUnmetRecord> records;
  if (final_state.cells.size() <= records.max_size() / 3) {
    records.reserve(3 * final_state.cells.size());
  }
  for (const TetSnapshotCell& cell : final_state.cells) {
    std::array<std::int32_t, 4> slots{};
    std::array<const double*, 4> points{};
    bool live = true;
    for (int corner = 0; corner < 4; ++corner) {
      slots[corner] = mesh.vertex_slot(cell.vertices[corner]);
      if (!mesh.live_vertex(slots[corner])) {
        live = false;
        break;
      }
      points[corner] = mesh.point(slots[corner]);
    }
    if (!live || orient3d(points[0], points[1], points[2], points[3]) <= 0) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_VALIDITY,
                         PHX_MC_TET_UNMET_NONREPRESENTABLE, 0.0});
      ++unmet_cells;
      continue;
    }
    bool cell_unmet = false;
    geometry::TetrahedronShape shape;
    if (!geometry::tetrahedron_shape(points.data(), shape) ||
        !std::isfinite(shape.radius_edge_ratio)) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_CONSTRUCTION,
                         PHX_MC_TET_UNMET_NONREPRESENTABLE,
                         shape.radius_edge_ratio});
      cell_unmet = true;
    } else if (shape.radius_edge_ratio > request.radius_edge_bound) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_RADIUS_EDGE,
                         PHX_MC_TET_UNMET_NO_IMPROVEMENT,
                         shape.radius_edge_ratio});
      cell_unmet = true;
    }
    const double angle = geometry::minimum_dihedral_degrees(points.data());
    if (!std::isfinite(angle)) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_CONSTRUCTION,
                         PHX_MC_TET_UNMET_NONREPRESENTABLE, angle});
      cell_unmet = true;
    } else if (angle < request.minimum_dihedral_degrees) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_DIHEDRAL,
                         PHX_MC_TET_UNMET_NO_IMPROVEMENT, angle});
      cell_unmet = true;
    }
    const double determinant = geometry::relative_determinant(points.data());
    if (!std::isfinite(determinant)) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_CONSTRUCTION,
                         PHX_MC_TET_UNMET_NONREPRESENTABLE, determinant});
      cell_unmet = true;
    } else if (determinant <= request.minimum_relative_determinant) {
      records.push_back({cell.id, PHX_MC_TET_UNMET_VALIDITY,
                         PHX_MC_TET_UNMET_NO_IMPROVEMENT,
                         determinant - request.minimum_relative_determinant});
      cell_unmet = true;
    }
    if (cell_unmet) ++unmet_cells;
  }
  mesh.set_unmet_records(records);
  return unmet_cells == 0 ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

}  // namespace phx::mc
