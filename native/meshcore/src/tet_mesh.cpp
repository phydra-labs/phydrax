//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Stable identity and persistent-state utilities for TetMesh.
#include "tet_mesh.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace phx::mc {

TetMesh::TetMesh(BoundaryPolicy policy) : policy_(policy), edit_(complex_) {}

bool TetMesh::protect_facet_sources(std::span<const std::int32_t> sources) {
  clear_implicit_facet_refusal();
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) return false;
  std::unordered_set<std::int32_t> requested(sources.begin(), sources.end());
  std::unordered_set<std::int32_t> found;
  std::set<std::array<std::uint64_t, 2>> edges;
  std::vector<std::size_t> facet_rows;
  std::vector<std::int32_t> vertices;
  for (std::size_t row = 0; row < facets_.size(); ++row) {
    const TetSnapshotFacet& facet = facets_[row];
    if (!requested.contains(facet.source)) continue;
    found.insert(facet.source);
    facet_rows.push_back(row);
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const std::int32_t slot = vertex_slot(facet.vertices[corner]);
      if (!live_vertex(slot)) return false;
      vertices.push_back(slot);
      std::array<std::uint64_t, 2> edge{
          facet.vertices[corner], facet.vertices[(corner + 1) % 3]};
      std::sort(edge.begin(), edge.end());
      edges.insert(edge);
    }
  }
  if (found != requested) return false;
  for (std::size_t row : facet_rows) facets_[row].protected_facet = true;
  for (std::int32_t vertex : vertices) protected_vertices_[static_cast<std::size_t>(vertex)] = 1;
  for (TetSnapshotSegment& segment : segments_) {
    auto edge = segment.vertices;
    std::sort(edge.begin(), edge.end());
    if (edges.contains(edge)) segment.protected_segment = true;
  }
  ++generation_;
  unmet_records_.clear();
  return true;
}

bool TetMesh::bind_source_identity(std::string_view source_id,
                                   std::string_view source_revision,
                                   std::string_view source_map_id,
                                   std::string_view frame_id,
                                   bool identity_bound) {
  clear_implicit_facet_refusal();
  const auto& implicit = implicit_source_ ? implicit_source_ : pending_implicit_source_;
  if (implicit && (!identity_bound || source_id != implicit->source_id ||
                   source_revision != implicit->source_revision ||
                   source_map_id != implicit->binding_id || frame_id != implicit->frame_id)) {
    return false;
  }
  identity_bound_ = identity_bound;
  auto copy = [](std::string& destination, std::string_view source) {
    if (source.empty()) destination.clear();
    else destination.assign(source.data(), source.size());
  };
  if (!identity_bound_) {
    copy(source_id_, {});
    copy(source_revision_, {});
    copy(source_map_id_, {});
    copy(frame_id_, {});
    return true;
  }
  copy(source_id_, source_id);
  copy(source_revision_, source_revision);
  copy(source_map_id_, source_map_id);
  copy(frame_id_, frame_id);
  return true;
}
bool TetMesh::bind_periodic_identity(const TetPeriodicIdentity& identity) {
  clear_implicit_facet_refusal();
  if (implicit_source_ && identity.bound) return false;
  periodic_identity_ = identity;
  return true;
}

int32_t TetMesh::assign_initial_ids(
    std::span<const std::uint64_t> vertex_ids,
    std::span<const std::uint64_t> cell_ids,
    std::span<const std::uint64_t> facet_ids,
    std::span<const std::uint64_t> segment_ids,
    std::span<const double> initial_vertex_sizes) {
  clear_implicit_facet_refusal();
  try {
    const std::size_t finite_cells = static_cast<std::size_t>(complex_.finite_count);
    if (generation_ != 0 ||
        (!vertex_ids.empty() && vertex_ids.size() != vertex_ids_.size()) ||
        (!cell_ids.empty() && cell_ids.size() != finite_cells) ||
        (!facet_ids.empty() && facet_ids.size() != facets_.size()) ||
        (!segment_ids.empty() && segment_ids.size() != segments_.size())) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    std::vector<std::uint64_t> staged_vertices = vertex_ids_;
    std::vector<std::uint64_t> staged_cells = cell_ids_;
    std::vector<TetSnapshotFacet> staged_facets = facets_;
    std::vector<TetSnapshotSegment> staged_segments = segments_;
    if (!vertex_ids.empty()) std::copy(vertex_ids.begin(), vertex_ids.end(), staged_vertices.begin());
    if (!cell_ids.empty()) std::copy(cell_ids.begin(), cell_ids.end(), staged_cells.begin());
    if (!facet_ids.empty()) {
      for (std::size_t row = 0; row < facet_ids.size(); ++row) staged_facets[row].id = facet_ids[row];
    }
    if (!segment_ids.empty()) {
      for (std::size_t row = 0; row < segment_ids.size(); ++row) staged_segments[row].id = segment_ids[row];
    }

    std::unordered_map<std::uint64_t, std::int32_t> staged_vertex_slots;
    std::unordered_map<std::uint64_t, std::int32_t> staged_cell_slots;
    std::unordered_set<std::uint64_t> staged_used_vertices;
    std::unordered_set<std::uint64_t> staged_used_cells;
    std::unordered_set<std::uint64_t> staged_used_facets;
    std::unordered_set<std::uint64_t> staged_used_segments;
    staged_vertex_slots.reserve(staged_vertices.size());
    staged_cell_slots.reserve(finite_cells);
    staged_used_vertices.reserve(staged_vertices.size());
    staged_used_cells.reserve(finite_cells);
    staged_used_facets.reserve(staged_facets.size());
    staged_used_segments.reserve(staged_segments.size());
    std::uint64_t next_vertex = 0;
    for (std::size_t slot = 0; slot < staged_vertices.size(); ++slot) {
      const std::uint64_t id = staged_vertices[slot];
      if (id == std::numeric_limits<std::uint64_t>::max() || vertex_live_[slot] == 0 ||
          !staged_vertex_slots.emplace(id, static_cast<std::int32_t>(slot)).second) {
        return PHX_MC_INVALID_INPUT;
      }
      staged_used_vertices.insert(id);
      next_vertex = std::max(next_vertex, id + 1);
    }
    std::uint64_t next_cell = 0;
    for (std::size_t row = 0; row < finite_cells; ++row) {
      const std::uint64_t id = staged_cells[row];
      if (id == std::numeric_limits<std::uint64_t>::max() ||
          !complex_.live(static_cast<std::int32_t>(row)) ||
          is_ghost(complex_.tets[row]) ||
          !staged_cell_slots.emplace(id, static_cast<std::int32_t>(row)).second) {
        return PHX_MC_INVALID_INPUT;
      }
      staged_used_cells.insert(id);
      next_cell = std::max(next_cell, id + 1);
    }
    std::uint64_t next_facet = 0;
    for (TetSnapshotFacet& facet : staged_facets) {
      if (facet.id == std::numeric_limits<std::uint64_t>::max() ||
          !staged_used_facets.insert(facet.id).second) return PHX_MC_INVALID_INPUT;
      for (std::uint64_t& vertex : facet.vertices) {
        const std::int32_t slot = vertex_slot(vertex);
        if (!live_vertex(slot)) return PHX_MC_INVALID_INPUT;
        vertex = staged_vertices[static_cast<std::size_t>(slot)];
      }
      next_facet = std::max(next_facet, facet.id + 1);
    }
    std::uint64_t next_segment = 0;
    for (TetSnapshotSegment& segment : staged_segments) {
      if (segment.id == std::numeric_limits<std::uint64_t>::max() ||
          !staged_used_segments.insert(segment.id).second) return PHX_MC_INVALID_INPUT;
      for (std::uint64_t& vertex : segment.vertices) {
        const std::int32_t slot = vertex_slot(vertex);
        if (!live_vertex(slot)) return PHX_MC_INVALID_INPUT;
        vertex = staged_vertices[static_cast<std::size_t>(slot)];
      }
      next_segment = std::max(next_segment, segment.id + 1);
    }
    std::vector<double> retained_sizes;
    if (initial_vertex_sizes.empty() && !vertex_sizes_.empty()) {
      retained_sizes.resize(vertex_ids_.size(), 0.0);
      for (std::size_t row = 0; row < vertex_ids_.size(); ++row) {
        const auto found = vertex_sizes_.find(vertex_ids_[row]);
        if (found != vertex_sizes_.end()) retained_sizes[row] = found->second;
      }
    }
    const int32_t size_status = install_vertex_sizes(
        staged_vertices, initial_vertex_sizes.empty() ? std::span<const double>(retained_sizes)
                                                       : initial_vertex_sizes);
    if (size_status != PHX_MC_OK) return size_status;
    vertex_ids_.swap(staged_vertices);
    cell_ids_.swap(staged_cells);
    facets_.swap(staged_facets);
    segments_.swap(staged_segments);
    vertex_slots_.swap(staged_vertex_slots);
    cell_slots_.swap(staged_cell_slots);
    used_vertex_ids_.swap(staged_used_vertices);
    used_cell_ids_.swap(staged_used_cells);
    used_facet_ids_.swap(staged_used_facets);
    used_segment_ids_.swap(staged_used_segments);
    next_vertex_id_ = next_vertex;
    next_cell_id_ = next_cell;
    next_facet_id_ = next_facet;
    next_segment_id_ = next_segment;
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return PHX_MC_ALLOCATION_FAILURE;
  } catch (const std::length_error&) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
}

bool TetMesh::live_vertex(std::int32_t slot) const noexcept {
  return slot >= 0 && static_cast<std::size_t>(slot) < vertex_live_.size() &&
         vertex_live_[static_cast<std::size_t>(slot)] != 0;
}

std::int32_t TetMesh::vertex_slot(std::uint64_t logical_id) const noexcept {
  const auto found = vertex_slots_.find(logical_id);
  return found == vertex_slots_.end() ? -1 : found->second;
}

std::int32_t TetMesh::cell_slot(std::uint64_t logical_id) const noexcept {
  const auto found = cell_slots_.find(logical_id);
  return found == cell_slots_.end() ? -1 : found->second;
}

std::uint64_t TetMesh::vertex_id(std::int32_t slot) const noexcept {
  return slot >= 0 && static_cast<std::size_t>(slot) < vertex_ids_.size()
             ? vertex_ids_[static_cast<std::size_t>(slot)]
             : std::numeric_limits<std::uint64_t>::max();
}

std::uint64_t TetMesh::cell_id(std::int32_t slot) const noexcept {
  return slot >= 0 && static_cast<std::size_t>(slot) < cell_ids_.size()
             ? cell_ids_[static_cast<std::size_t>(slot)]
             : std::numeric_limits<std::uint64_t>::max();
}

const double* TetMesh::point(std::int32_t slot) const noexcept {
  if (!live_vertex(slot)) return nullptr;
  return points_.data() + 3 * static_cast<std::size_t>(slot);
}

const SourceWitness& TetMesh::witness(std::int32_t slot) const noexcept {
  static const SourceWitness none{};
  return slot >= 0 && static_cast<std::size_t>(slot) < witnesses_.size()
             ? witnesses_[static_cast<std::size_t>(slot)]
             : none;
}

std::int8_t TetMesh::vertex_dimension(std::int32_t slot) const noexcept {
  return slot >= 0 && static_cast<std::size_t>(slot) < vertex_dimension_.size()
             ? vertex_dimension_[static_cast<std::size_t>(slot)]
             : -1;
}

double TetMesh::protection_radius(std::int32_t slot) const noexcept {
  return slot >= 0 && static_cast<std::size_t>(slot) < protection_radii_.size()
             ? protection_radii_[static_cast<std::size_t>(slot)]
             : std::numeric_limits<double>::infinity();
}

bool TetMesh::protected_vertex(std::int32_t slot) const noexcept {
  return slot >= 0 && static_cast<std::size_t>(slot) < protected_vertices_.size() &&
         protected_vertices_[static_cast<std::size_t>(slot)] != 0;
}

const Tetrahedron& TetMesh::tet(std::int32_t slot) const noexcept {
  static const Tetrahedron dead{{kDeadVertex, kDeadVertex, kDeadVertex, kDeadVertex},
                                {-1, -1, -1, -1}, 0, 0};
  return complex_.live(slot) ? complex_.tets[static_cast<std::size_t>(slot)] : dead;
}

}  // namespace phx::mc
