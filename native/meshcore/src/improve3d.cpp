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

std::span<const std::int32_t> LocalIncidence::edge_star(std::int32_t first,
                                                        std::int32_t second) const {
  const auto found = edge_cells_.find(sorted_local_edge(first, second));
  if (found == edge_cells_.end()) return {};
  return found->second;
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

TetEditStatus LocalIncidence::replace_cavity(TetMesh& mesh, const TetEdit& operation,
                                             const Star* collapsed_star) {
  try {
    const bool collapse = operation.kind == TetEditKind::kCollapse;
    if (collapse != (collapsed_star != nullptr)) return TetEditStatus::kRejectedTopology;
    std::vector<std::int32_t> removed_slots;
    std::vector<LocalCell> removed_cells;
    removed_slots.reserve(operation.removed_cell_ids.size());
    removed_cells.reserve(operation.removed_cell_ids.size());
    for (std::uint64_t id : operation.removed_cell_ids) {
      const std::int32_t slot = mesh.cell_slot(id);
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) {
        return TetEditStatus::kRejectedTopology;
      }
      removed_slots.push_back(slot);
      LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      removed_cells.push_back(vertices);
    }
    std::int32_t removed_vertex = -1;
    LocalPositionKey removed_position{};
    if (collapse) {
      if (operation.removed_vertex_ids.size() != 1) return TetEditStatus::kRejectedTopology;
      removed_vertex = mesh.vertex_slot(operation.removed_vertex_ids.front());
      const Star& star = *collapsed_star;
      if (removed_vertex < 0 || star.owner_ != this || star.vertex_ != removed_vertex ||
          star.revision_ != topology_revision_) {
        return TetEditStatus::kRejectedTopology;
      }
      std::vector<std::int32_t> removed_sorted = removed_slots;
      std::vector<std::int32_t> star_sorted = star.cells_;
      std::sort(removed_sorted.begin(), removed_sorted.end());
      std::sort(star_sorted.begin(), star_sorted.end());
      if (removed_sorted != star_sorted) return TetEditStatus::kRejectedTopology;
      removed_position = local_position_key(mesh.point(removed_vertex));
    }
    const TetEditStatus status = mesh.replace_local_cavity(operation);
    if (status != TetEditStatus::kApplied) return status;
    for (std::size_t index = 0; index < removed_slots.size(); ++index) {
      if (!erase(removed_slots[index], removed_cells[index])) {
        return TetEditStatus::kStaleGeneration;
      }
    }
    for (const TetEditCell& cell : operation.replacement_cells) {
      const std::int32_t slot = mesh.cell_slot(cell.id);
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) {
        return TetEditStatus::kStaleGeneration;
      }
      LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      if (!insert_logical(mesh, slot, vertices)) return TetEditStatus::kStaleGeneration;
    }
    if (collapse && !erase_vertex_position(removed_vertex, removed_position)) {
      return TetEditStatus::kStaleGeneration;
    }
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  }
}

TetEditStatus LocalIncidence::insert_cone(TetMesh& mesh, const TetEditVertex& vertex,
                                          std::span<const std::int32_t> cavity,
                                          std::vector<std::uint64_t>& created_cells) {
  try {
    std::vector<LocalCell> removed_cells;
    removed_cells.reserve(cavity.size());
    for (std::int32_t slot : cavity) {
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) {
        return TetEditStatus::kRejectedTopology;
      }
      LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      removed_cells.push_back(vertices);
    }
    if (vertex_positions_.contains(local_position_key(vertex.carrier.data()))) {
      return TetEditStatus::kInvalidGeometry;
    }
    const TetEditStatus status = mesh.insert_local_cone(vertex, cavity, created_cells);
    if (status != TetEditStatus::kApplied) return status;
    for (std::size_t index = 0; index < cavity.size(); ++index) {
      if (!erase(cavity[index], removed_cells[index])) return TetEditStatus::kStaleGeneration;
    }
    for (std::uint64_t id : created_cells) {
      const std::int32_t slot = mesh.cell_slot(id);
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) {
        return TetEditStatus::kStaleGeneration;
      }
      LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      if (!insert_logical(mesh, slot, vertices)) return TetEditStatus::kStaleGeneration;
    }
    if (!insert_vertex_position(mesh.vertex_slot(vertex.id), vertex.carrier.data())) {
      return TetEditStatus::kStaleGeneration;
    }
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  }
}

TetEditStatus LocalIncidence::insert_boundary_point(TetMesh& mesh, const TetEditVertex& vertex,
                                                    std::span<const std::int32_t> cavity,
                                                    std::span<const std::size_t> facet_rows,
                                                    std::span<const std::size_t> segment_rows,
                                                    std::vector<std::uint64_t>& created_cells) {
  try {
    std::vector<std::int32_t> removed_slots;
    std::vector<LocalCell> removed_cells;
    removed_slots.reserve(cavity.size());
    removed_cells.reserve(cavity.size());
    for (std::int32_t slot : cavity) {
      if (!mesh.complex().live(slot)) return TetEditStatus::kRejectedTopology;
      if (is_ghost(mesh.tet(slot))) continue;
      LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      removed_slots.push_back(slot);
      removed_cells.push_back(vertices);
    }
    if (vertex_positions_.contains(local_position_key(vertex.carrier.data()))) {
      return TetEditStatus::kInvalidGeometry;
    }
    const TetEditStatus status = mesh.insert_local_boundary_point(vertex, cavity, facet_rows,
                                                                  segment_rows, created_cells);
    if (status != TetEditStatus::kApplied) return status;
    for (std::size_t index = 0; index < removed_slots.size(); ++index) {
      if (!erase(removed_slots[index], removed_cells[index])) {
        return TetEditStatus::kStaleGeneration;
      }
    }
    for (std::uint64_t id : created_cells) {
      const std::int32_t slot = mesh.cell_slot(id);
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) {
        return TetEditStatus::kStaleGeneration;
      }
      LocalCell vertices{};
      std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
      if (!insert_logical(mesh, slot, vertices)) return TetEditStatus::kStaleGeneration;
    }
    if (!insert_vertex_position(mesh.vertex_slot(vertex.id), vertex.carrier.data())) {
      return TetEditStatus::kStaleGeneration;
    }
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  }
}

}  // namespace detail

namespace {

struct CellMeasure {
  double minimum_angle = 0.0;
  double radius_edge_ratio = 0.0;
  double relative_determinant = 0.0;
  double circumradius = std::numeric_limits<double>::infinity();
  double size_target = std::numeric_limits<double>::infinity();
};
struct PreparedRelocationCell {
  std::array<const double*, 4> fixed_points{};
  std::int32_t moved_corner = -1;
  double size_target = std::numeric_limits<double>::infinity();
  geometry::RelocationMetricCache metric_cache;
};
using detail::LocalCell;
using detail::LocalEdgeKey;
using detail::LocalIncidence;
using detail::sorted_local_edge;
using FaceKey = std::array<std::int32_t, 3>;
using LogicalEdgeKey = std::pair<std::uint64_t, std::uint64_t>;
// Validity classes of a measured cell (-1: not measurable).
constexpr std::int8_t kBelowDeterminantFloor = 0;
constexpr std::int8_t kAboveDeterminantFloor = 1;
struct FlipQuality {
  std::int8_t validity_class = -1;
  double primary = -std::numeric_limits<double>::infinity();
  // Circumradius over the cell's size target (0 without a target, +inf for
  // an unrepresentable circumsphere); above 1 the cell is beyond its size.
  double size_excess = 0.0;
};
using QualityVector = std::vector<FlipQuality>;
void sort_quality_vector(QualityVector& quality);

// The single cell score is the worse of the dihedral and radius-edge target
// fractions, 1 when both targets hold. It is resolved downward to multiples of
// kTetScoreQuantum = 2^-20 of the target (an exact power-of-two scaling and
// floor), far coarser than the score's floating evaluation error and far finer
// than any meaningful quality difference. The finite score alphabet is what
// bounds the number of accepted edits; without it a smoothing sequence can
// keep making strictly smaller improvements. Target attainment (score 1) and
// the final per-criterion residual records use the unresolved measures.
// The validity class separates cells below the determinant floor from the
// rest, and such a cell scores 0 whatever its angles and radius-edge ratio:
// the score orders every edit, and a flat cell can have target angles and
// ratio (a cocircular boundary quad split by its diagonal has a finite
// circumsphere), so its removal would otherwise not count as an improvement
// and the vertex phase would skip its star. The size excess is gated
// separately (admissible_size_excess).
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
  const double unresolved = std::min(angle_score, radius_score);
  const std::int8_t validity =
      !(measure.relative_determinant > request.minimum_relative_determinant)
          ? kBelowDeterminantFloor
          : kAboveDeterminantFloor;
  const double quality_score =
      validity == kBelowDeterminantFloor
          ? 0.0
          : std::floor(unresolved / kTetScoreQuantum) * kTetScoreQuantum;
  const double size_excess = std::isinf(measure.size_target)
                                 ? 0.0
                                 : measure.circumradius / measure.size_target;
  return {validity, quality_score, size_excess};
}

double vertex_size(const TetQualityRequest& request, std::int32_t slot) {
  if (request.slot_sizes == nullptr || slot < 0 ||
      static_cast<std::size_t>(slot) >= request.slot_sizes->size()) {
    return 0.0;
  }
  return (*request.slot_sizes)[static_cast<std::size_t>(slot)];
}
double slots_size_target(const TetQualityRequest& request, const std::int32_t* slots) {
  return cell_size_target({vertex_size(request, slots[0]), vertex_size(request, slots[1]),
                           vertex_size(request, slots[2]), vertex_size(request, slots[3])});
}

// Measures an exactly positive cell. A positive cell whose circumsphere is not
// representable in floating point (the circumcenter solve meets a zero pivot,
// as for a rounded cocircular sliver) has a radius-edge ratio above every
// representable bound, so it is measured as +inf (score 0) rather than refused;
// the exact orientation keeps its validity beyond doubt.
bool measure_cell(const std::array<const double*, 4>& points, double size_target,
                  CellMeasure& result) {
  if (orient3d(points[0], points[1], points[2], points[3]) <= 0) return false;
  geometry::TetrahedronShape shape;
  if (geometry::tetrahedron_shape(points.data(), shape)) {
    result.radius_edge_ratio = shape.radius_edge_ratio;
    result.circumradius = shape.circumradius;
  } else {
    result.radius_edge_ratio = std::numeric_limits<double>::infinity();
    result.circumradius = std::numeric_limits<double>::infinity();
  }
  result.size_target = size_target;
  result.minimum_angle = geometry::minimum_dihedral_degrees(points.data());
  result.relative_determinant = geometry::relative_determinant(points.data());
  return std::isfinite(result.minimum_angle) &&
         std::isfinite(result.relative_determinant);
}
bool measure_relocated_cell(const std::array<const double*, 4>& points,
                            const geometry::RelocationMetricCache& metric_cache,
                            double size_target, CellMeasure& result) {
  if (orient3d(points[0], points[1], points[2], points[3]) <= 0) return false;
  geometry::TetrahedronShape shape;
  if (geometry::relocation_tetrahedron_shape(points.data(), metric_cache, shape)) {
    result.radius_edge_ratio = shape.radius_edge_ratio;
    result.circumradius = shape.circumradius;
  } else {
    result.radius_edge_ratio = std::numeric_limits<double>::infinity();
    result.circumradius = std::numeric_limits<double>::infinity();
  }
  result.size_target = size_target;
  result.minimum_angle =
      geometry::relocation_minimum_dihedral_degrees(points.data(), metric_cache);
  result.relative_determinant =
      geometry::relocation_relative_determinant(points.data(), metric_cache);
  return std::isfinite(result.minimum_angle) &&
         std::isfinite(result.relative_determinant);
}

bool cell_measure(const TetMesh& mesh, const Tetrahedron& tet,
                  const TetQualityRequest& request, CellMeasure& result) {
  std::array<const double*, 4> points{};
  for (int corner = 0; corner < 4; ++corner) {
    points[corner] = mesh.point(tet.v[corner]);
    if (points[corner] == nullptr) return false;
  }
  return measure_cell(points, slots_size_target(request, tet.v), result);
}

bool interior_quality(const TetMesh& mesh, std::span<const std::int32_t> star,
                      const TetQualityRequest& request, QualityVector& quality) {
  quality.clear();
  quality.reserve(star.size());
  for (std::int32_t slot : star) {
    CellMeasure measure;
    if (!cell_measure(mesh, mesh.tet(slot), request, measure)) return false;
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
    if (!measure_cell(points, slots_size_target(request, local.data()), measure)) {
      return false;
    }
    quality.push_back(quality_from_measure(measure, request));
    replacement_cells.push_back(candidate);
  }
  sort_quality_vector(quality);
  return !replacement_cells.empty() && quality.size() == replacement_cells.size();
}

bool prepare_relocation_plan(const TetMesh& mesh,
                             std::span<const std::int32_t> star,
                             std::int32_t moved, const TetQualityRequest& request,
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
    // A relocated vertex keeps its own size target.
    cell.size_target = slots_size_target(request, tet.v);
    if (cell.moved_corner < 0 ||
        !geometry::prepare_relocation_metric_cache(
            cell.fixed_points.data(), cell.moved_corner, cell.metric_cache)) {
      return false;
    }
  }
  return true;
}

bool admissible_cell(const FlipQuality& quality, double size_excess_bound) noexcept;

// Scores the star with `candidate` in place of its moved vertex. Stops at the
// first cell that is not measurable, not admissible under `size_excess_bound`
// or scores below `score_floor`: no relocation creating such a cell can be
// accepted against a sorted vector that starts at `score_floor`.
bool relocated_quality(std::span<const PreparedRelocationCell> plan,
                       const double* candidate, const TetQualityRequest& request,
                       double score_floor, double size_excess_bound,
                       QualityVector& quality) {
  quality.clear();
  quality.reserve(plan.size());
  if (plan.empty()) return false;
  for (const PreparedRelocationCell& cell : plan) {
    std::array<const double*, 4> points = cell.fixed_points;
    points[static_cast<std::size_t>(cell.moved_corner)] = candidate;
    CellMeasure measure;
    if (!measure_relocated_cell(points, cell.metric_cache, cell.size_target, measure)) {
      return false;
    }
    const FlipQuality quality_of_cell = quality_from_measure(measure, request);
    if (quality_of_cell.primary < score_floor ||
        !admissible_cell(quality_of_cell, size_excess_bound)) {
      return false;
    }
    quality.push_back(quality_of_cell);
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
  if (!measure_cell(points, slots_size_target(request, vertices.data()), measure) ||
      measure.relative_determinant < 0.0) {
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

// Strict order on worst-first sorted score vectors: at the first unequal
// position the smaller score is worse, and a vector that ends first is padded
// with +inf. Equivalently, at the smallest score whose multiplicity differs,
// the multiset holding more copies is worse. Adding the same unchanged cells to
// both sides never changes that score or the multiplicity comparison, so an
// edit that improves its affected cells improves the sorted score vector of the
// whole mesh. std::lexicographical_compare is not this order: it ranks a proper
// prefix below its extension and would accept edits that only add cells.
bool quality_vector_less(std::span<const FlipQuality> worse,
                         std::span<const FlipQuality> better) {
  const std::size_t common = std::min(worse.size(), better.size());
  for (std::size_t index = 0; index < common; ++index) {
    if (flip_quality_less(worse[index], better[index])) return true;
    if (flip_quality_less(better[index], worse[index])) return false;
  }
  return worse.size() > better.size();
}

// Cells whose replacement takes precedence over the size field: below the
// determinant floor, or with score 0 (a ratio beyond 2^20 times its target,
// including unrepresentable circumspheres).
bool degenerate_cell(const FlipQuality& quality) noexcept {
  return quality.validity_class == kBelowDeterminantFloor || quality.primary <= 0.0;
}

// Largest size excess a cell created by an edit replacing `before` may have.
// The size field binds improvement relative to the cells an edit removes: an
// edit never creates a cell beyond its target where the removed cells were all
// within theirs (bound 1), and where refinement left cells beyond size (fixed
// boundary, refused or separation-guarded insertions) it may not exceed the
// worst excess it removes, so the mesh's largest excess never grows. An edit
// that removes a degenerate cell is not size-bound: repairing degeneracy takes
// precedence. The determinant floor binds every edit (admissible_cell).
double admissible_size_excess(std::span<const FlipQuality> before) {
  double bound = 1.0;
  for (const FlipQuality& quality : before) {
    if (degenerate_cell(quality)) return std::numeric_limits<double>::infinity();
    bound = std::max(bound, quality.size_excess);
  }
  return bound;
}
bool admissible_cell(const FlipQuality& quality, double size_excess_bound) noexcept {
  return quality.validity_class == kAboveDeterminantFloor &&
         quality.size_excess <= size_excess_bound;
}

// The single acceptance rule for every topology and vertex edit: all
// replacement cells are admissible under `size_excess_bound` and the affected
// cells' sorted score vector strictly increases in quality_vector_less.
bool improves_quality_vector(std::span<const FlipQuality> before,
                             std::span<const FlipQuality> after,
                             double size_excess_bound) {
  if (before.empty() || after.empty()) return false;
  if (std::any_of(after.begin(), after.end(), [&](const FlipQuality& quality) {
        return !admissible_cell(quality, size_excess_bound);
      })) {
    return false;
  }
  return quality_vector_less(before, after);
}
// Local edits: the gates admissible for the edit (above).
bool improves_quality_vector(std::span<const FlipQuality> before,
                             std::span<const FlipQuality> after) {
  return improves_quality_vector(before, after, admissible_size_excess(before));
}

// A relocation is continuous: an arbitrarily small move can carry a score
// across a grid line, so relocations accepted on any strict improvement could
// step through the score alphabet one quantum at a time while the geometry
// barely changes. A relocation must instead make sufficient progress toward
// the target: at the first position k where the star's worst-first sorted
// scores before and after differ, the after score must be larger and either
// meet the target (1) or close at least kRelocationGapFraction of the
// remaining gap 1 - s_k, rounded up to a whole quantum. Earlier positions are
// unchanged and position k rises, so the sorted vector strictly increases in
// quality_vector_less (the star's worst-first sorted score never drops) and an
// accepted relocation is accepted by the general rule too. For k = 0 (the
// usual case) the star's worst score s gains at least max(quantum,
// kRelocationGapFraction * (1 - s)): its gap shrinks geometrically, by the
// factor 1 - kRelocationGapFraction per accepted relocation, so it reaches
// the target after O(log(gap / quantum)) such relocations instead of up to
// gap / quantum.
constexpr double kRelocationGapFraction = 0x1p-3;
bool relocation_makes_progress(std::span<const FlipQuality> before,
                               std::span<const FlipQuality> after) {
  if (before.size() != after.size()) return false;
  for (std::size_t index = 0; index < before.size(); ++index) {
    const double score = before[index].primary;
    const double moved = after[index].primary;
    if (moved == score) continue;
    if (moved < score) return false;
    if (moved >= 1.0) return true;
    const double gain =
        std::ceil(kRelocationGapFraction * (1.0 - score) / kTetScoreQuantum) * kTetScoreQuantum;
    return moved >= score + gain;
  }
  return false;
}

// Improvement's targets are the score and the determinant floor; the size
// field constrains its edits but is not one of its targets.
bool cell_quality_meets_target(const FlipQuality& quality) noexcept {
  return quality.validity_class == kAboveDeterminantFloor && quality.primary >= 1.0;
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
  // The ring's own cells decide which replacement cells may be considered
  // (see admissible_size_excess).
  if (!quality_vector_for_slots(mesh, removed_slots, request, scratch.before_quality)) {
    return TetEditStatus::kRejectedTopology;
  }
  const double size_excess_bound = admissible_size_excess(scratch.before_quality);

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
    if (!admissible_cell(triangle_quality[0], size_excess_bound) ||
        !admissible_cell(triangle_quality[1], size_excess_bound)) {
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
        // Optimal sub-intervals compose because quality_vector_less is
        // invariant under adding the same cells to both sides.
        if (choices[current] < 0 ||
            quality_vector_less(current_best, candidate_quality)) {
          std::copy(candidate_quality.begin(), candidate_quality.end(),
                    best_quality.begin() + current_offset);
          choices[current] = static_cast<std::int32_t>(k);
        }
      }
    }
  }
  if (choices[count - 1] < 0) return TetEditStatus::kRejectedTopology;
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
    TetEdit& operation) {
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

// A cell on face F has circumradius at least F's circumradius r_F; the point
// at height r_F above F's circumcenter, on the side of `inside`, is the apex
// attaining that minimum. False for a floating-degenerate face.
bool face_minimum_radius_apex(const TetMesh& mesh, const std::int32_t facet[3],
                              const double* inside, std::array<double, 3>& apex) {
  const auto cross = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
    return std::array<double, 3>{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                                 u[0] * v[1] - u[1] * v[0]};
  };
  const auto dot = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  const double* a = mesh.point(facet[0]);
  std::array<double, 3> ab{}, ac{}, toward{};
  for (int axis = 0; axis < 3; ++axis) {
    ab[axis] = mesh.point(facet[1])[axis] - a[axis];
    ac[axis] = mesh.point(facet[2])[axis] - a[axis];
    toward[axis] = inside[axis] - a[axis];
  }
  const std::array<double, 3> normal = cross(ab, ac);
  const double normal_squared = dot(normal, normal);
  if (!(normal_squared > 0.0) || !std::isfinite(normal_squared)) return false;
  const std::array<double, 3> first = cross(normal, ab);
  const std::array<double, 3> second = cross(ac, normal);
  std::array<double, 3> offset{};
  for (int axis = 0; axis < 3; ++axis) {
    offset[axis] =
        (dot(ac, ac) * first[axis] + dot(ab, ab) * second[axis]) / (2.0 * normal_squared);
  }
  const double side = dot(normal, toward) > 0.0 ? 1.0 : -1.0;
  const double scale = side * std::sqrt(dot(offset, offset) / normal_squared);
  for (int axis = 0; axis < 3; ++axis) {
    apex[axis] = a[axis] + offset[axis] + scale * normal[axis];
  }
  return std::all_of(apex.begin(), apex.end(), [](double value) { return std::isfinite(value); });
}

// A point in the kernel of the region tiled by `cells` (membership
// `members`): every cone from it over a region boundary face is exactly
// positive. Compass search from the better of `seeds` with steps halving from
// `initial_step` maximizes the smallest relative determinant of the cones, so
// the point keeps clear of every boundary face plane. False when no seed or
// step reaches the kernel.
bool cavity_kernel_point(const TetMesh& mesh, std::span<const std::int32_t> cells,
                         const std::unordered_set<std::int32_t>& members,
                         std::span<const std::array<double, 3>> seeds, double initial_step,
                         std::array<double, 3>& point) {
  struct Cone {
    std::array<const double*, 4> points{};
    int apex = 0;
  };
  std::vector<Cone> cones;
  for (std::int32_t cell : cells) {
    const Tetrahedron& tetrahedron = mesh.tet(cell);
    for (int face = 0; face < 4; ++face) {
      if (members.contains(tetrahedron.n[face])) continue;
      Cone& cone = cones.emplace_back();
      cone.apex = face;
      for (int corner = 0; corner < 4; ++corner) {
        cone.points[static_cast<std::size_t>(corner)] = mesh.point(tetrahedron.v[corner]);
      }
    }
  }
  if (cones.empty() || seeds.empty() || !(initial_step > 0.0) || !std::isfinite(initial_step)) {
    return false;
  }
  const auto clearance = [&](const std::array<double, 3>& candidate) {
    double smallest = std::numeric_limits<double>::infinity();
    for (Cone& cone : cones) {
      std::array<const double*, 4> points = cone.points;
      points[static_cast<std::size_t>(cone.apex)] = candidate.data();
      if (orient3d(points[0], points[1], points[2], points[3]) <= 0) {
        return -std::numeric_limits<double>::infinity();
      }
      const double determinant = geometry::relative_determinant(points.data());
      if (!std::isfinite(determinant)) return -std::numeric_limits<double>::infinity();
      smallest = std::min(smallest, determinant);
    }
    return smallest;
  };
  point = seeds.front();
  double best = clearance(point);
  for (const std::array<double, 3>& seed : seeds.subspan(1)) {
    const double value = clearance(seed);
    if (value > best) {
      best = value;
      point = seed;
    }
  }
  // Each step length allows at most kMovesPerStep improving moves before it
  // halves, kHalvings times, so the search makes at most
  // 6 * kHalvings * (kMovesPerStep + 1) clearance evaluations.
  constexpr int kHalvings = 40;
  constexpr int kMovesPerStep = 16;
  double step = initial_step;
  for (int halving = 0; halving < kHalvings; ++halving, step *= 0.5) {
    for (int move = 0; move < kMovesPerStep; ++move) {
      bool moved = false;
      for (int direction = 0; direction < 6 && !moved; ++direction) {
        std::array<double, 3> candidate = point;
        candidate[static_cast<std::size_t>(direction / 2)] += direction % 2 == 0 ? step : -step;
        const double value = clearance(candidate);
        if (value > best) {
          best = value;
          point = candidate;
          moved = true;
        }
      }
      if (!moved) break;
    }
  }
  return best > 0.0;
}

// A vertex the vertex phase may collapse or relocate: live, interior, without
// protection, and not witnessed on a source stratum.
bool free_interior_vertex(const TetMesh& mesh, std::int32_t vertex) {
  return mesh.live_vertex(vertex) && mesh.vertex_dimension(vertex) == 3 &&
         !mesh.protected_vertex(vertex) && mesh.protection_radius(vertex) == 0.0 &&
         mesh.witness(vertex).stratum == SourceStratum::kNone;
}
}  // namespace

std::int32_t flip_to_constrained_delaunay(
    TetMesh& mesh, detail::LocalIncidence& incidence, std::span<const std::uint64_t> seeds,
    const std::function<bool(std::uint64_t, std::uint64_t)>& is_segment,
    std::vector<std::uint64_t>& created) {
  std::vector<std::pair<std::uint64_t, int>> pending;
  for (std::uint64_t id : seeds) {
    for (int face = 0; face < 4; ++face) pending.emplace_back(id, face);
  }
  const auto apply = [&](std::span<const std::int32_t> removed,
                         std::span<const LocalCell> replacements, TetEditKind kind) {
    TetEdit operation;
    std::vector<std::uint64_t> neighbors;
    TetEditStatus status = prepare_quality_reconnection(mesh, removed, replacements,
                                                        mesh.next_cell_id_, kind, operation,
                                                        neighbors);
    if (status == TetEditStatus::kApplied) status = incidence.replace_cavity(mesh, operation, nullptr);
    if (status != TetEditStatus::kApplied) return status;
    for (const TetEditCell& cell : operation.replacement_cells) {
      created.push_back(cell.id);
      for (int face = 0; face < 4; ++face) pending.emplace_back(cell.id, face);
    }
    return TetEditStatus::kApplied;
  };
  while (!pending.empty()) {
    const auto [cell_id, face] = pending.back();
    pending.pop_back();
    const std::int32_t slot = mesh.cell_slot(cell_id);
    if (slot < 0 || !mesh.complex().live(slot)) continue;
    const Tetrahedron& cell = mesh.tet(slot);
    const std::int32_t other = cell.n[face];
    if (!mesh.complex().live(other) || is_ghost(mesh.tet(other)) ||
        mesh.complex().constraint(slot, face) != kNoConstraint ||
        mesh.complex().region(other) != mesh.complex().region(slot)) {
      continue;
    }
    const int back = neighbor_slot(mesh.tet(other), slot);
    if (back < 0) return PHX_MC_INTERNAL_ERROR;
    const std::int32_t apex = mesh.tet(other).v[back];
    if (insphere(mesh.point(cell.v[0]), mesh.point(cell.v[1]), mesh.point(cell.v[2]),
                 mesh.point(cell.v[3]), mesh.point(apex)) <= 0) {
      continue;
    }
    // 2-3: the shared face is crossed by the edge between the apexes.
    std::array<LocalCell, 3> three{};
    bool convex = true;
    for (int corner = 0, row = 0; corner < 4; ++corner) {
      if (corner == face) continue;
      LocalCell replacement{};
      std::copy_n(cell.v, 4, replacement.begin());
      replacement[static_cast<std::size_t>(corner)] = apex;
      convex = convex && mesh.positive(replacement.data());
      three[static_cast<std::size_t>(row++)] = replacement;
    }
    TetEditStatus status = TetEditStatus::kRejectedTopology;
    if (convex) {
      const std::array<std::int32_t, 2> removed{slot, other};
      status = apply(removed, three, TetEditKind::kFlip23);
    } else {
      // 3-2: an edge of the shared face whose ring is this cell, its
      // neighbor and one cell through both apexes.
      for (int corner = 0; corner < 4 && status != TetEditStatus::kApplied; ++corner) {
        if (corner == face) continue;
        std::array<std::int32_t, 2> edge{};
        for (int other_corner = 0, count = 0; other_corner < 4; ++other_corner) {
          if (other_corner != face && other_corner != corner) edge[count++] = cell.v[other_corner];
        }
        if (is_segment(mesh.vertex_id(edge[0]), mesh.vertex_id(edge[1]))) continue;
        const std::span<const std::int32_t> ring = incidence.edge_star(edge[0], edge[1]);
        if (ring.size() != 3) continue;
        std::int32_t third = -1;
        for (std::int32_t member : ring) {
          if (member != slot && member != other) third = member;
        }
        if (third < 0 || vertex_slot(mesh.tet(third), apex) < 0 ||
            vertex_slot(mesh.tet(third), cell.v[face]) < 0) {
          continue;
        }
        std::array<LocalCell, 2> two{};
        bool positive = true;
        for (std::size_t side = 0; side < 2; ++side) {
          LocalCell replacement{};
          std::copy_n(cell.v, 4, replacement.begin());
          const int replaced = vertex_slot(cell, edge[side]);
          replacement[static_cast<std::size_t>(replaced)] = apex;
          positive = positive && mesh.positive(replacement.data());
          two[side] = replacement;
        }
        if (!positive) continue;
        const std::array<std::int32_t, 3> removed{slot, other, third};
        status = apply(removed, two, TetEditKind::kFlip32);
      }
    }
    if (status == TetEditStatus::kAllocationFailure) return PHX_MC_ALLOCATION_FAILURE;
    if (status == TetEditStatus::kIndexOverflow) return PHX_MC_INDEX_RANGE_ERROR;
    if (status == TetEditStatus::kStaleGeneration) return PHX_MC_INTERNAL_ERROR;
  }
  return PHX_MC_OK;
}

// Radius-edge ratio (circumradius over shortest edge) of a triangle:
// l_mid * l_max / (2 |(b - a) x (c - a)|); infinite when degenerate.
static double triangle_radius_edge_ratio(const double* a, const double* b,
                                         const double* c) noexcept {
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double bc[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
  const double normal[3] = {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                            ab[0] * ac[1] - ab[1] * ac[0]};
  const double twice_area = std::hypot(normal[0], normal[1], normal[2]);
  std::array<double, 3> lengths{std::hypot(ab[0], ab[1], ab[2]),
                                std::hypot(ac[0], ac[1], ac[2]),
                                std::hypot(bc[0], bc[1], bc[2])};
  std::sort(lengths.begin(), lengths.end());
  if (!(twice_area > 0.0)) return std::numeric_limits<double>::infinity();
  return lengths[1] * lengths[2] / (2.0 * twice_area);
}

std::int32_t improve_mesh(TetMesh& mesh, const TetQualityRequest& requested,
                          std::span<const double> vertex_sizes,
                          std::int64_t& inserted_vertices,
                          std::int64_t& collapsed_vertices,
                          std::int64_t& relocated_vertices,
                          std::int64_t& unmet_cells) {
  inserted_vertices = 0;
  collapsed_vertices = 0;
  relocated_vertices = 0;
  unmet_cells = 0;
  if (!std::isfinite(requested.radius_edge_bound) ||
      requested.radius_edge_bound <= 0.0 ||
      !std::isfinite(requested.minimum_dihedral_degrees) ||
      requested.minimum_dihedral_degrees < 0.0 ||
      requested.minimum_dihedral_degrees >= 70.5 ||
      !std::isfinite(requested.minimum_relative_determinant) ||
      requested.minimum_relative_determinant < 0.0 ||
      requested.minimum_relative_determinant >= 1.0) {
    return PHX_MC_INVALID_ARGUMENT;
  }

  TetMeshSnapshot initial;
  if (mesh.snapshot(initial) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  const std::int32_t installed = mesh.install_vertex_sizes(initial, vertex_sizes);
  if (installed != PHX_MC_OK) return installed;
  // Size targets by local vertex slot; slots are append-only during this run.
  std::vector<double> slot_sizes;
  const auto refresh_slot_sizes = [&]() {
    slot_sizes.assign(mesh.vertex_ids_.size(), 0.0);
    for (std::size_t slot = 0; slot < slot_sizes.size(); ++slot) {
      if (mesh.vertex_live_[slot] == 0) continue;
      const auto found = mesh.vertex_sizes_.find(mesh.vertex_ids_[slot]);
      if (found != mesh.vertex_sizes_.end()) slot_sizes[slot] = found->second;
    }
  };
  refresh_slot_sizes();
  TetQualityRequest request = requested;
  request.minimum_relative_determinant =
      std::max(requested.minimum_relative_determinant, kMetricDeterminantFloor);
  request.slot_sizes = &slot_sizes;
  bool has_unmet_quality = false;
  QualityVector before_cavity_quality;
  before_cavity_quality.reserve(initial.cells.size());
  for (const TetSnapshotCell& cell : initial.cells) {
    const std::int32_t slot = mesh.cell_slot(cell.id);
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return PHX_MC_INTERNAL_ERROR;
    LocalCell vertices{};
    std::copy_n(mesh.tet(slot).v, 4, vertices.begin());
    const FlipQuality quality = local_flip_quality(mesh, vertices, request);
    if (quality.validity_class < 0) return PHX_MC_INVALID_INPUT;
    before_cavity_quality.push_back(quality);
    has_unmet_quality = has_unmet_quality || !cell_quality_meets_target(quality);
  }
  sort_quality_vector(before_cavity_quality);
  if (has_unmet_quality) {
    TetEdit operation;
    if (make_fixed_boundary_star_retriangulation(
            mesh, initial, mesh.next_vertex_id_, mesh.next_cell_id_, operation)) {
      // The fan center's size target interpolates the boundary vertices'.
      double center_size = 0.0;
      for (const TetVertexAncestry& ancestor : operation.changed_vertices.front().ancestry) {
        const double size = vertex_size(request, mesh.vertex_slot(ancestor.source_vertex_id));
        if (!(size > 0.0)) {
          center_size = 0.0;
          break;
        }
        center_size += ancestor.weight * size;
      }
      QualityVector after_cavity_quality;
      after_cavity_quality.reserve(operation.replacement_cells.size());
      for (const TetEditCell& cell : operation.replacement_cells) {
        std::array<const double*, 4> points{};
        std::array<double, 4> sizes{};
        for (int corner = 0; corner < 3; ++corner) {
          const std::int32_t slot = mesh.vertex_slot(
              cell.vertices[static_cast<std::size_t>(corner)]);
          if (!mesh.live_vertex(slot)) return PHX_MC_INTERNAL_ERROR;
          points[static_cast<std::size_t>(corner)] = mesh.point(slot);
          sizes[static_cast<std::size_t>(corner)] = vertex_size(request, slot);
        }
        points[3] = operation.changed_vertices.front().carrier.data();
        sizes[3] = center_size;
        CellMeasure measure;
        if (!measure_cell(points, cell_size_target(sizes), measure)) return PHX_MC_INVALID_INPUT;
        after_cavity_quality.push_back(quality_from_measure(measure, request));
      }
      sort_quality_vector(after_cavity_quality);
      // This whole-mesh retessellation would discard every interior vertex,
      // so it may create no cell beyond its size target at all.
      if (improves_quality_vector(before_cavity_quality, after_cavity_quality, 1.0)) {
        TetEditBatch batch;
        batch.expected_generation = mesh.generation();
        batch.source_revision = mesh.source_revision();
        batch.frame_id = mesh.frame_id();
        batch.identity_bound = mesh.identity_bound();
        batch.source_id = mesh.source_id();
        batch.source_map_id = mesh.source_map_id();
        batch.periodic_identity = mesh.periodic_identity();
        const std::uint64_t center_id = operation.changed_vertices.front().id;
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
          if (center_size > 0.0) mesh.vertex_sizes_[center_id] = center_size;
          refresh_slot_sizes();
        }
      }
    }
  }
  LocalIncidence incidence;
  if (initial.cells.size() >
      std::numeric_limits<std::size_t>::max() / 6) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  // The fixed-boundary star retessellation runs before this index is built;
  // later edits maintain it (flips alter cells, insertion adds one point,
  // collapse removes one, and relocation moves one).
  if (!incidence.initialize(mesh, initial)) return PHX_MC_INTERNAL_ERROR;
  std::set<LogicalEdgeKey> source_edges;
  for (const TetSnapshotSegment& segment : initial.constrained_segments) {
    source_edges.insert(sorted_logical_edge(segment.vertices[0], segment.vertices[1]));
  }
  std::set<LogicalEdgeKey> rejected_high_edges;
  // Topology and vertex worklists. Each holds only entries whose own cell, or
  // star for a vertex, may violate a target, and an entry returns only after
  // an accepted edit changed that cell or star or was made while processing it.
  std::set<std::uint64_t> pending_cells;
  std::set<std::uint64_t> pending_vertices;

  const auto apply_quality_flip = [&] (
      std::span<const std::int32_t> removed_slots,
      std::span<const LocalCell> replacement_cells, TetEditKind kind,
      std::vector<std::uint64_t>& affected_cells) {
    TetEdit operation;
    std::vector<std::uint64_t> boundary_neighbors;
    const TetEditStatus prepared = prepare_quality_reconnection(
        mesh, removed_slots, replacement_cells, mesh.next_cell_id_, kind,
        operation, boundary_neighbors);
    if (prepared != TetEditStatus::kApplied) {
      return prepared;
    }
    const TetEditStatus status = incidence.replace_cavity(mesh, operation, nullptr);
    if (status != TetEditStatus::kApplied) return status;
    std::vector<std::uint64_t> touched;
    touched.reserve(operation.replacement_cells.size() + boundary_neighbors.size());
    for (const TetEditCell& cell : operation.replacement_cells) {
      touched.push_back(cell.id);
      pending_vertices.insert(cell.vertices.begin(), cell.vertices.end());
    }
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
    return status;
  };

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
  // A vertex edit changes its star cells and the face-neighbor candidates of
  // every cell on the star boundary.
  const auto enqueue_vertex_edit = [&](std::span<const std::int32_t> slots) {
    for (std::int32_t slot : slots) {
      if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return false;
      if (!enqueue_violating_cell(mesh.cell_id(slot))) return false;
      for (std::int32_t neighbor : mesh.tet(slot).n) {
        if (!mesh.complex().live(neighbor) || is_ghost(mesh.tet(neighbor))) continue;
        if (!enqueue_violating_cell(mesh.cell_id(neighbor))) return false;
      }
    }
    return true;
  };
  for (const TetSnapshotCell& cell : initial.cells) {
    if (!enqueue_violating_cell(cell.id)) return PHX_MC_INVALID_INPUT;
  }
  for (std::uint64_t id : pending_cells) {
    const Tetrahedron& tet = mesh.tet(mesh.cell_slot(id));
    for (std::int32_t vertex : tet.v) pending_vertices.insert(mesh.vertex_id(vertex));
  }

  // Sorted score vector of the cone from `candidate` over the boundary faces
  // of the star-shaped cavity `cells` (membership `members`); the new vertex's
  // size target interpolates at the candidate as the target of the cell
  // `slot` being improved (unconstrained if that cell is). False when a cone
  // cell is not exactly positive or not measurable.
  const auto measure_cone = [&](std::int32_t slot, const double* candidate,
                                std::span<const std::int32_t> cells,
                                const std::unordered_set<std::int32_t>& members,
                                QualityVector& quality) {
    const double target = slots_size_target(request, mesh.tet(slot).v);
    const double candidate_size = std::isfinite(target) ? target : 0.0;
    quality.clear();
    for (std::int32_t cell : cells) {
      const Tetrahedron& tetrahedron = mesh.tet(cell);
      for (int face = 0; face < 4; ++face) {
        if (members.contains(tetrahedron.n[face])) continue;
        std::array<const double*, 4> points{};
        std::array<double, 4> sizes{};
        for (int corner = 0; corner < 4; ++corner) {
          points[corner] = corner == face ? candidate : mesh.point(tetrahedron.v[corner]);
          sizes[corner] =
              corner == face ? candidate_size : vertex_size(request, tetrahedron.v[corner]);
        }
        CellMeasure measure;
        if (!measure_cell(points, cell_size_target(sizes), measure)) return false;
        quality.push_back(quality_from_measure(measure, request));
      }
    }
    sort_quality_vector(quality);
    return true;
  };

  // Cones the star-shaped cavity `cells` (membership `members`) to a new
  // interior vertex at `candidate` when the cone's sorted score vector
  // strictly improves on the cavity's, under the same rule as every other
  // edit. `slot` is the cell being improved.
  QualityVector insertion_before;
  QualityVector insertion_after;
  std::vector<std::uint64_t> inserted_cells;
  const auto cone_insertion = [&](std::int32_t slot, const double* candidate,
                                  std::span<const std::int32_t> cells,
                                  const std::unordered_set<std::int32_t>& members)
      -> TetEditStatus {
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(candidate[axis]) || !coordinate_in_domain(candidate[axis])) {
        return TetEditStatus::kInvalidGeometry;
      }
    }
    if (mesh.next_vertex_id_ == std::numeric_limits<std::uint64_t>::max() ||
        !quality_vector_for_slots(mesh, cells, request, insertion_before)) {
      return TetEditStatus::kRejectedTopology;
    }
    const double target = slots_size_target(request, mesh.tet(slot).v);
    const double candidate_size = std::isfinite(target) ? target : 0.0;
    if (!measure_cone(slot, candidate, cells, members, insertion_after) ||
        !improves_quality_vector(insertion_before, insertion_after)) {
      return TetEditStatus::kRejectedTopology;
    }
    TetEditVertex inserted;
    inserted.id = mesh.next_vertex_id_;
    std::copy_n(candidate, 3, inserted.carrier.begin());
    const TetEditStatus status =
        incidence.insert_cone(mesh, inserted, cells, inserted_cells);
    if (status != TetEditStatus::kApplied) return status;
    if (candidate_size > 0.0) mesh.vertex_sizes_[inserted.id] = candidate_size;
    slot_sizes.resize(mesh.vertex_ids_.size(), 0.0);
    slot_sizes[static_cast<std::size_t>(mesh.vertex_slot(inserted.id))] = candidate_size;
    std::vector<std::int32_t> created_slots;
    created_slots.reserve(inserted_cells.size());
    for (std::uint64_t id : inserted_cells) {
      const std::int32_t created = mesh.cell_slot(id);
      created_slots.push_back(created);
      for (std::int32_t vertex : mesh.tet(created).v) {
        pending_vertices.insert(mesh.vertex_id(vertex));
      }
    }
    if (!enqueue_vertex_edit(created_slots)) return TetEditStatus::kStaleGeneration;
    rejected_high_edges.clear();
    ++inserted_vertices;
    return TetEditStatus::kApplied;
  };

  // Insertion at `candidate` into its star-shaped cavity walked to from
  // `slot` (TetMesh::insertion_cavity).
  TetInsertionCavity insertion;
  const auto try_insertion = [&](std::int32_t slot,
                                 const double* candidate) -> TetEditStatus {
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(candidate[axis]) || !coordinate_in_domain(candidate[axis])) {
        return TetEditStatus::kInvalidGeometry;
      }
    }
    if (mesh.insertion_cavity(slot, candidate, true, insertion) != TetCavityStatus::kFound) {
      return TetEditStatus::kRejectedTopology;
    }
    return cone_insertion(slot, candidate, insertion.cells, insertion.members);
  };

  // Edge-ring insertion: replaces the closed ring of the unconstrained edge
  // (`first`, `second`) by the cone from a new vertex in the ring's kernel.
  // Every point of the open edge sees each ring face with the sign of its
  // ring cell, so the kernel is nonempty, and the cone removes the edge
  // without touching the ring's boundary (constrained faces included). This
  // is the edit that removes a flat cell whose four vertices lie on a fixed
  // boundary: its two constrained faces split a planar quad along one
  // diagonal, no flip can remove it (a 2-3 flip across either free face
  // inverts a cell, and the free diagonal's ring admits no positive
  // retriangulation without a new point), and no circumsphere insertion
  // cavity contains it (its circumsphere degenerates to the half-space
  // beyond the boundary).
  //
  // Vertex-star insertion cones instead the star of one of the cell's
  // vertices. For a vertex on a fixed boundary the vertex stays (its
  // constrained faces bound the star) while every interior edge from it is
  // replaced: a flat cell spanning two constrained faces at a folded rim,
  // whose single free edge's ring admits no better cone, is removed this
  // way. The commit refuses a star whose removal would drop a vertex, a
  // constrained face or segment, or mix regions (TetMesh::insert_local_cone).
  QualityVector cone_quality;
  QualityVector trial_quality;
  QualityVector cavity_before_quality;
  // Cones the star-shaped cavity `cells` (membership `members`) of the cell
  // `slot` to a new vertex in its kernel, found from `seeds` with the first
  // step `initial_step`.
  const auto try_kernel_cone = [&](std::int32_t slot, std::span<const std::int32_t> cells,
                                   const std::unordered_set<std::int32_t>& members,
                                   std::span<const std::array<double, 3>> seeds,
                                   double initial_step) -> TetEditStatus {
    std::array<double, 3> candidate{};
    if (!quality_vector_for_slots(mesh, cells, request, cavity_before_quality) ||
        !cavity_kernel_point(mesh, cells, members, seeds, initial_step, candidate) ||
        !measure_cone(slot, candidate.data(), cells, members, cone_quality)) {
      return TetEditStatus::kRejectedTopology;
    }
    // The kernel point only keeps clear of every cavity face; a cone from it
    // over a nearly flat cavity can be worse than the cells it replaces, or
    // exceed the size gate. A pattern search over the kernel then moves the
    // point only to a strictly better cone in the order the edit is judged
    // by: a cone whose cells all pass the gates (determinant floor, the size
    // excess the removed cells admit) beats one that does not, and otherwise
    // the larger sorted score vector wins. Its 26 directions are the nonzero
    // offsets of {-1, 0, 1}^3, so it also follows the diagonal ridges along
    // which two cone cells' scores tie, where axis moves alone stall. Steps
    // halve from the kernel search's first step down to 2^-20 of it, where
    // scores are resolved to their quantum anyway; each step length allows
    // at most kMovesPerStep moves, so the search makes at most
    // 26 * kHalvings * (kMovesPerStep + 1) cone measurements.
    const double size_excess_bound = admissible_size_excess(cavity_before_quality);
    const auto admissible_cone = [&](const QualityVector& cone) {
      return std::all_of(cone.begin(), cone.end(), [&](const FlipQuality& quality) {
        return admissible_cell(quality, size_excess_bound);
      });
    };
    bool candidate_admissible = admissible_cone(cone_quality);
    constexpr int kHalvings = 20;
    constexpr int kMovesPerStep = 16;
    double step = initial_step;
    for (int halving = 0; halving < kHalvings; ++halving, step *= 0.5) {
      for (int move = 0; move < kMovesPerStep; ++move) {
        bool moved = false;
        for (int direction = 0; direction < 27 && !moved; ++direction) {
          if (direction == 13) continue;  // the zero offset
          std::array<double, 3> trial = candidate;
          bool inside = true;
          for (int axis = 0, code = direction; axis < 3; ++axis, code /= 3) {
            trial[static_cast<std::size_t>(axis)] += static_cast<double>(code % 3 - 1) * step;
            inside = inside && coordinate_in_domain(trial[static_cast<std::size_t>(axis)]);
          }
          if (!inside || !measure_cone(slot, trial.data(), cells, members, trial_quality)) {
            continue;
          }
          const bool trial_admissible = admissible_cone(trial_quality);
          if (trial_admissible != candidate_admissible
                  ? !trial_admissible
                  : !quality_vector_less(cone_quality, trial_quality)) {
            continue;
          }
          candidate = trial;
          candidate_admissible = trial_admissible;
          cone_quality.swap(trial_quality);
          moved = true;
        }
        if (!moved) break;
      }
    }
    return cone_insertion(slot, candidate.data(), cells, members);
  };

  std::vector<std::int32_t> ring_cells;
  std::vector<std::int32_t> ring_vertices;
  std::unordered_set<std::int32_t> ring_members;
  const auto try_edge_ring_insertion = [&](std::int32_t slot, std::int32_t first,
                                           std::int32_t second) -> TetEditStatus {
    if (source_edges.contains(sorted_logical_edge(mesh.vertex_id(first),
                                                  mesh.vertex_id(second)))) {
      return TetEditStatus::kRejectedTopology;
    }
    const std::span<const std::int32_t> no_removed;
    const std::int32_t start = incidence.edge_cell_outside(mesh, first, second, no_removed);
    if (start < 0 || !collect_edge_ring(mesh, start, first, second, ring_cells, ring_vertices)) {
      return TetEditStatus::kRejectedTopology;
    }
    ring_members.clear();
    ring_members.insert(ring_cells.begin(), ring_cells.end());
    std::array<std::array<double, 3>, 2> seeds{};
    const double* first_point = mesh.point(first);
    const double* second_point = mesh.point(second);
    double edge_length_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      seeds[0][axis] = 0.5 * (first_point[axis] + second_point[axis]);
      double sum = first_point[axis] + second_point[axis];
      for (std::int32_t vertex : ring_vertices) sum += mesh.point(vertex)[axis];
      seeds[1][axis] = sum / static_cast<double>(ring_vertices.size() + 2);
      const double delta = second_point[axis] - first_point[axis];
      edge_length_squared += delta * delta;
    }
    return try_kernel_cone(slot, ring_cells, ring_members, seeds,
                           0.25 * std::sqrt(edge_length_squared));
  };

  std::vector<std::int32_t> star_cells;
  std::unordered_set<std::int32_t> star_members;
  const auto try_vertex_star_insertion = [&](std::int32_t slot,
                                             std::int32_t vertex) -> TetEditStatus {
    const LocalIncidence::Star star = incidence.vertex_star(vertex);
    star_cells.assign(star.cells().begin(), star.cells().end());
    if (star_cells.empty()) return TetEditStatus::kRejectedTopology;
    star_members.clear();
    star_members.insert(star_cells.begin(), star_cells.end());
    // Seeds: the centroid of the star's cell centroids and of the cell being
    // improved; the first step is a quarter of the longest edge at `vertex`.
    std::array<std::array<double, 3>, 2> seeds{};
    const double* center = mesh.point(vertex);
    double longest_squared = 0.0;
    for (std::int32_t cell : star_cells) {
      for (std::int32_t corner : mesh.tet(cell).v) {
        double length_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          const double coordinate = mesh.point(corner)[axis];
          seeds[0][axis] += 0.25 * coordinate / static_cast<double>(star_cells.size());
          const double delta = coordinate - center[axis];
          length_squared += delta * delta;
        }
        longest_squared = std::max(longest_squared, length_squared);
      }
    }
    for (std::int32_t corner : mesh.tet(slot).v) {
      for (int axis = 0; axis < 3; ++axis) seeds[1][axis] += 0.25 * mesh.point(corner)[axis];
    }
    return try_kernel_cone(slot, star_cells, star_members, seeds,
                           0.25 * std::sqrt(longest_squared));
  };

  const auto topology_step = [&](std::uint64_t logical_id) -> std::int32_t {
    const std::int32_t slot = mesh.cell_slot(logical_id);
    if (!mesh.complex().live(slot) || is_ghost(mesh.tet(slot))) return PHX_MC_OK;
    LocalCell current{};
    std::copy_n(mesh.tet(slot).v, 4, current.begin());
    const FlipQuality quality = local_flip_quality(mesh, current, request);
    if (quality.validity_class < 0) return PHX_MC_INTERNAL_ERROR;
    if (cell_quality_meets_target(quality)) return PHX_MC_OK;
    // An accepted edit made while processing this cell need not remove it
    // (an edge removal unblocking one of its faces, an insertion whose
    // cavity shrank around it); a surviving violating cell returns.
    const auto requeue_current = [&]() -> std::int32_t {
      return enqueue_violating_cell(logical_id) ? PHX_MC_OK : PHX_MC_INTERNAL_ERROR;
    };

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
    if (changed) return requeue_current();

    for (const FaceOrder& face : faces) {
      std::array<std::int32_t, 2> removed{};
      std::array<LocalCell, 3> replacements{};
      std::array<std::int32_t, 3> blocked_edge{-1, -1, -1};
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
        if (!enqueue_affected(affected)) return PHX_MC_INTERNAL_ERROR;
        changed = true;
        break;
      }
    }
    if (changed) return requeue_current();

    for (const EdgeOrder& edge : edges) {
      if (incidence.edge_valence(edge.first, edge.second) > 4) continue;
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
    }
    if (changed) return requeue_current();

    // Edge-ring insertion over this cell's unconstrained edges (in key
    // order), then vertex-star insertion over its vertices (in ID order). A
    // degenerate cell tries them before any point candidate: its circumcenter
    // is meaningless and no circumsphere cavity can contain a flat cell on a
    // fixed boundary (see try_edge_ring_insertion). Any other cell tries them
    // after the point candidates when none of its vertices is free, so the
    // vertex phase (collapse, relocation) can never change it.
    const auto ring_insertion = [&](bool& applied) -> std::int32_t {
      applied = false;
      for (const EdgeOrder& edge : edges) {
        const TetEditStatus status = try_edge_ring_insertion(slot, edge.first, edge.second);
        const std::int32_t error = fatal_edit_error(status);
        if (error != PHX_MC_OK) return error;
        if (status == TetEditStatus::kApplied) {
          applied = true;
          return requeue_current();
        }
      }
      std::array<std::int32_t, 4> star_order = current;
      std::sort(star_order.begin(), star_order.end(), [&](std::int32_t first, std::int32_t second) {
        return mesh.vertex_id(first) < mesh.vertex_id(second);
      });
      for (std::int32_t vertex : star_order) {
        const TetEditStatus status = try_vertex_star_insertion(slot, vertex);
        const std::int32_t error = fatal_edit_error(status);
        if (error != PHX_MC_OK) return error;
        if (status == TetEditStatus::kApplied) {
          applied = true;
          return requeue_current();
        }
      }
      return PHX_MC_OK;
    };
    const bool degenerate = degenerate_cell(quality);
    if (degenerate) {
      bool applied = false;
      const std::int32_t result = ring_insertion(applied);
      if (result != PHX_MC_OK || applied) return result;
    }

    // Otherwise insert a new interior vertex at the cell's circumcenter, then
    // at off-centers halfway and a quarter of the way from its centroid toward
    // the circumcenter (inside the domain when the circumcenter lies beyond a
    // fixed face), then above each of its constrained faces, then at the
    // centroid of its face-neighborhood (which lies off the plane of a flat
    // cell).
    std::array<std::array<double, 3>, 8> candidates{};
    std::size_t candidate_count = 0;
    {
      std::array<const double*, 4> points{};
      std::array<double, 3> cell_centroid{};
      for (int corner = 0; corner < 4; ++corner) {
        points[corner] = mesh.point(current[corner]);
        for (int axis = 0; axis < 3; ++axis) cell_centroid[axis] += 0.25 * points[corner][axis];
      }
      geometry::TetrahedronShape shape;
      if (geometry::tetrahedron_shape(points.data(), shape)) {
        candidates[candidate_count++] = shape.circumcenter;
        for (const double fraction : {0.5, 0.25}) {
          std::array<double, 3>& off_center = candidates[candidate_count++];
          for (int axis = 0; axis < 3; ++axis) {
            off_center[axis] = cell_centroid[axis] +
                               fraction * (shape.circumcenter[axis] - cell_centroid[axis]);
          }
        }
      }
      for (int face = 0; face < 4; ++face) {
        if (mesh.complex().constraint(slot, face) == kNoConstraint) continue;
        std::int32_t facet[3];
        oriented_facet(mesh.tet(slot).v, face, facet);
        if (face_minimum_radius_apex(mesh, facet, points[face], candidates[candidate_count])) {
          ++candidate_count;
        }
      }
      std::set<std::int32_t> neighborhood(current.begin(), current.end());
      for (std::int32_t neighbor : mesh.tet(slot).n) {
        if (!mesh.complex().live(neighbor) || is_ghost(mesh.tet(neighbor))) continue;
        neighborhood.insert(mesh.tet(neighbor).v, mesh.tet(neighbor).v + 4);
      }
      std::array<double, 3> centroid{};
      for (std::int32_t vertex : neighborhood) {
        for (int axis = 0; axis < 3; ++axis) centroid[axis] += mesh.point(vertex)[axis];
      }
      for (double& value : centroid) value /= static_cast<double>(neighborhood.size());
      candidates[candidate_count++] = centroid;
    }
    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
      const TetEditStatus status = try_insertion(slot, candidates[candidate].data());
      const std::int32_t error = fatal_edit_error(status);
      if (error != PHX_MC_OK) return error;
      if (status == TetEditStatus::kApplied) return requeue_current();
    }
    if (!degenerate &&
        std::none_of(current.begin(), current.end(), [&](std::int32_t vertex) {
          return free_interior_vertex(mesh, vertex);
        })) {
      bool applied = false;
      return ring_insertion(applied);
    }
    return PHX_MC_OK;
  };

  std::vector<PreparedRelocationCell> relocation_plan;
  QualityVector before_quality;
  QualityVector after_quality;
  std::vector<TetEditCell> replacement_cells;
  const auto vertex_step = [&](std::uint64_t vertex_id) -> std::int32_t {
    const std::int32_t removed = mesh.vertex_slot(vertex_id);
    if (!free_interior_vertex(mesh, removed)) return PHX_MC_OK;
    LocalIncidence::Star star = incidence.vertex_star(removed);
    if (!interior_quality(mesh, star.cells(), request, before_quality)) return PHX_MC_OK;
    if (quality_vector_meets_target(before_quality)) return PHX_MC_OK;

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

    for (std::int32_t survivor : neighbors) {
      if (mesh.protected_vertex(survivor) || mesh.protection_radius(survivor) > 0.0 ||
          mesh.vertex_dimension(survivor) != 3) {
        continue;
      }
      if (!collapsed_quality(mesh, star.cells(), removed, survivor, request,
                             after_quality, replacement_cells) ||
          !improves_quality_vector(before_quality, after_quality) ||
          mesh.next_cell_id_ > std::numeric_limits<std::uint64_t>::max() -
                                   replacement_cells.size()) {
        continue;
      }
      TetEdit operation;
      operation.kind = TetEditKind::kCollapse;
      operation.collapse_pairs.push_back({vertex_id, mesh.vertex_id(survivor)});
      operation.removed_vertex_ids.push_back(vertex_id);
      operation.removed_cell_ids.reserve(star.size());
      for (std::int32_t slot : star) operation.removed_cell_ids.push_back(mesh.cell_id(slot));
      operation.replacement_cells = replacement_cells;
      for (std::size_t index = 0; index < operation.replacement_cells.size(); ++index) {
        operation.replacement_cells[index].id = mesh.next_cell_id_ + index;
      }
      std::vector<std::int32_t> changed_slots;
      changed_slots.reserve(operation.replacement_cells.size());
      const TetEditStatus status = incidence.replace_cavity(mesh, operation, &star);
      const std::int32_t error = fatal_edit_error(status);
      if (error != PHX_MC_OK) return error;
      if (status == TetEditStatus::kApplied) {
        for (const TetEditCell& cell : operation.replacement_cells) {
          changed_slots.push_back(mesh.cell_slot(cell.id));
          pending_vertices.insert(cell.vertices.begin(), cell.vertices.end());
        }
        if (!enqueue_vertex_edit(changed_slots)) return PHX_MC_INTERNAL_ERROR;
        // The edit erased the removed vertex's size entry.
        slot_sizes[static_cast<std::size_t>(removed)] = 0.0;
        ++collapsed_vertices;
        return PHX_MC_OK;
      }
    }

    // Relocation targets: the neighbor centroid, then the minimum-radius apex
    // of every constrained face opposite this vertex in its star. Each is
    // searched by backtracking from the target toward the current position
    // over the fractions 1, 1/2, ..., kTetScoreQuantum, so a vertex costs at
    // most 21 star evaluations per target, and a candidate is accepted only if
    // it makes sufficient progress (relocation_makes_progress). A star whose
    // scores are all 1 cannot make any and is not searched.
    if (before_quality.front().primary >= 1.0) return PHX_MC_OK;
    const double size_excess_bound = admissible_size_excess(before_quality);
    std::vector<std::array<double, 3>> targets(1, std::array<double, 3>{});
    for (std::int32_t neighbor : neighbors) {
      const double* point = mesh.point(neighbor);
      for (int axis = 0; axis < 3; ++axis) targets.front()[axis] += point[axis];
    }
    if (neighbors.empty()) return PHX_MC_OK;
    for (double& value : targets.front()) value /= static_cast<double>(neighbors.size());
    for (std::int32_t cell : star) {
      const int corner = vertex_slot(mesh.tet(cell), removed);
      if (corner < 0 || mesh.complex().constraint(cell, corner) == kNoConstraint) continue;
      std::int32_t facet[3];
      oriented_facet(mesh.tet(cell).v, corner, facet);
      std::array<double, 3> apex{};
      if (face_minimum_radius_apex(mesh, facet, mesh.point(removed), apex)) {
        targets.push_back(apex);
      }
    }
    if (!prepare_relocation_plan(mesh, star.cells(), removed, request, relocation_plan)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const double* original = mesh.point(removed);
    for (const std::array<double, 3>& target : targets) {
      for (double fraction = 1.0; fraction >= kTetScoreQuantum; fraction *= 0.5) {
        double candidate[3];
        bool distinct = false;
        for (int axis = 0; axis < 3; ++axis) {
          candidate[axis] = original[axis] + fraction * (target[axis] - original[axis]);
          distinct = distinct || candidate[axis] != original[axis];
        }
        if (!distinct) break;
        if (relocated_quality(relocation_plan, candidate, request,
                              before_quality.front().primary, size_excess_bound,
                              after_quality) &&
            relocation_makes_progress(before_quality, after_quality)) {
          TetEditVertex moved;
          moved.id = vertex_id;
          std::copy_n(candidate, 3, moved.carrier.begin());
          moved.witness = mesh.witness(removed);
          moved.protection_radius = mesh.protection_radius(removed);
          moved.protected_vertex = mesh.protected_vertex(removed);
          const TetEditStatus status =
              incidence.relocate(mesh, std::span(&moved, 1), star);
          const std::int32_t error = fatal_edit_error(status);
          if (error != PHX_MC_OK) return error;
          if (status == TetEditStatus::kApplied) {
            // Every vertex of the star's cells sees a changed star (and a
            // changed neighbor centroid), so each is stepped again: a vertex
            // is last stepped on its final star, which an unmet
            // NO_IMPROVEMENT record relies on.
            for (std::int32_t slot : star) {
              for (std::int32_t vertex : mesh.tet(slot).v) {
                pending_vertices.insert(mesh.vertex_id(vertex));
              }
            }
            if (!enqueue_vertex_edit(star.cells())) return PHX_MC_INTERNAL_ERROR;
            ++relocated_vertices;
            return PHX_MC_OK;
          }
        }
      }
    }
    return PHX_MC_OK;
  };

  // Termination. Every accepted edit strictly increases the whole mesh's
  // worst-first sorted score vector in quality_vector_less: the edit improves
  // its affected cells in that order, the order is invariant under adding the
  // unchanged cells to both sides, and each cell's score is evaluated from its
  // stored vertex order and coordinates by the same arithmetic before and after
  // the edit. Scores lie in the finite alphabet s_1 < ... < s_K of
  // quality_from_measure (multiples of 2^-20 in [0, 1]). A mesh state's score
  // multiset is its count vector (c_1, ..., c_K) in N^K, c_k the number of
  // cells scoring s_k, and quality_vector_less is exactly the lexicographic
  // order on these count vectors with the smaller vector the better (at the
  // smallest score whose count differs, more copies is worse). N^K under the
  // lexicographic order is a well-order (order type omega^K), so it has no
  // infinite strictly decreasing sequence: the accepted edits form a finite
  // sequence even though insertion adds vertices and cells, collapse removes
  // them, and the vertex set is therefore not fixed. A pop without an
  // accepted edit removes its entry, and entries return only after an
  // accepted edit (finitely many, above); each phase therefore drains,
  // and the run ends when a topology phase and a vertex phase leave both
  // worklists empty. The admissibility gates (determinant floor, size excess
  // bound) only discard edits, so they leave this argument intact; the size
  // bound also keeps the largest size excess created by a non-degenerate
  // repair at most the largest one it removed. Edge-ring and point insertions
  // are accepted by the same rule; relocations by the stricter sufficient-
  // progress rule (relocation_makes_progress), which implies it. That rule
  // also bounds the relocation work beyond mere termination: an accepted
  // relocation closes at least kRelocationGapFraction of the gap to the target
  // of its star's first changed score, so the relocations of a vertex whose
  // surrounding does not otherwise change raise its star's worst score at most
  // ceil(log(2^20) / log(8/7)) = 104 times before it meets the target, instead
  // of up to 2^20 quantum-sized times. Each vertex step evaluates a bounded
  // number of candidates (its star's collapses and at most 21 per relocation
  // target). An accepted relocation re-queues every vertex of its star's
  // cells, since each of their stars changed, so every vertex is last stepped
  // on its final star (the evidence behind NO_IMPROVEMENT records); a rejected
  // step re-queues nothing, and the gap rule alone bounds the accepted moves.
  // No bound on the number of accepted topology edits is claimed; unmet
  // targets are reported per cell below.
  while (!pending_cells.empty() || !pending_vertices.empty()) {
    while (!pending_cells.empty()) {
      const std::uint64_t logical_id = *pending_cells.begin();
      pending_cells.erase(pending_cells.begin());
      const std::int32_t error = topology_step(logical_id);
      if (error != PHX_MC_OK) return error;
    }
    while (!pending_vertices.empty()) {
      const std::uint64_t vertex_id = *pending_vertices.begin();
      pending_vertices.erase(pending_vertices.begin());
      const std::int32_t error = vertex_step(vertex_id);
      if (error != PHX_MC_OK) return error;
    }
  }

  TetMeshSnapshot final_state;
  if (mesh.snapshot(final_state) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  // The records cover improvement's own targets. The size field is a
  // constraint on improvement's edits, not one of its targets: residual size
  // violations are refinement's records (and the route's size compliance).
  std::vector<TetUnmetRecord> records;
  if (final_state.cells.size() <= records.max_size() / 3) {
    records.reserve(3 * final_state.cells.size());
  }
  // Improvement never edits a constrained face. A tetrahedron's circumsphere
  // contains each face's circumcircle and its shortest edge is at most the
  // face's, so its radius-edge ratio is at least that of each of its faces:
  // a cell on a constrained face whose own triangle exceeds the bound cannot
  // meet it, and its residual is FIXED_BOUNDARY rather than NO_IMPROVEMENT,
  // recording that facet (the one with the largest ratio, the tightest lower
  // bound, when several force it).
  struct ForcingFacet {
    std::uint64_t id = kNoUnmetFacet;
    double ratio = std::numeric_limits<double>::quiet_NaN();
  };
  std::map<std::array<std::uint64_t, 3>, ForcingFacet> unmeetable_faces;
  for (const TetSnapshotFacet& facet : final_state.constrained_facets) {
    std::array<const double*, 3> corners{};
    for (int corner = 0; corner < 3; ++corner) {
      const std::int32_t slot = mesh.vertex_slot(facet.vertices[corner]);
      if (!mesh.live_vertex(slot)) return PHX_MC_INTERNAL_ERROR;
      corners[corner] = mesh.point(slot);
    }
    const double ratio = triangle_radius_edge_ratio(corners[0], corners[1], corners[2]);
    if (ratio > request.radius_edge_bound) {
      std::array<std::uint64_t, 3> key = facet.vertices;
      std::sort(key.begin(), key.end());
      unmeetable_faces.emplace(key, ForcingFacet{facet.id, ratio});
    }
  }
  const auto forcing_facet = [&](const TetSnapshotCell& cell) {
    ForcingFacet forcing;
    for (int omitted = 0; omitted < 4; ++omitted) {
      std::array<std::uint64_t, 3> key{};
      for (int corner = 0, next = 0; corner < 4; ++corner) {
        if (corner != omitted) key[next++] = cell.vertices[corner];
      }
      std::sort(key.begin(), key.end());
      const auto found = unmeetable_faces.find(key);
      if (found != unmeetable_faces.end() &&
          (forcing.id == kNoUnmetFacet || found->second.ratio > forcing.ratio)) {
        forcing = found->second;
      }
    }
    return forcing;
  };
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
    if (!geometry::tetrahedron_shape(points.data(), shape)) {
      // Exactly positive (checked above) with a circumsphere that floating
      // point cannot represent: the ratio exceeds every representable bound.
      records.push_back({cell.id, PHX_MC_TET_UNMET_RADIUS_EDGE,
                         PHX_MC_TET_UNMET_NONREPRESENTABLE,
                         std::numeric_limits<double>::infinity()});
      cell_unmet = true;
    } else if (shape.radius_edge_ratio > request.radius_edge_bound) {
      const ForcingFacet forcing = forcing_facet(cell);
      TetUnmetRecord record{cell.id, PHX_MC_TET_UNMET_RADIUS_EDGE,
                            static_cast<std::int32_t>(forcing.id == kNoUnmetFacet
                                                          ? PHX_MC_TET_UNMET_NO_IMPROVEMENT
                                                          : PHX_MC_TET_UNMET_FIXED_BOUNDARY),
                            shape.radius_edge_ratio};
      record.facet_id = forcing.id;
      record.facet_ratio = forcing.ratio;
      records.push_back(record);
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
