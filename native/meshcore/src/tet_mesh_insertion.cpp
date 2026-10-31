//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Generation-checked edit batches and source/region-preserving cavity commits.
#include "tet_mesh.hpp"
#include "triangulation3d.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dyadic.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "facet_coverage.hpp"
#include "improve3d.hpp"
#include "refine3d.hpp"

namespace phx::mc {
namespace {


using SortedFace = std::array<std::int32_t, 3>;
using SortedEdge = std::array<std::int32_t, 2>;
using SortedTet = std::array<std::int32_t, 4>;

SortedFace sorted_face(const std::int32_t* vertices) {
  SortedFace key{vertices[0], vertices[1], vertices[2]};
  std::sort(key.begin(), key.end());
  return key;
}

SortedEdge sorted_edge(std::int32_t a, std::int32_t b) {
  return {std::min(a, b), std::max(a, b)};
}

SortedTet sorted_tet(const std::int32_t* vertices) {
  SortedTet key{vertices[0], vertices[1], vertices[2], vertices[3]};
  std::sort(key.begin(), key.end());
  return key;
}

std::uint64_t edge_key(std::int32_t a, std::int32_t b) {
  const std::uint32_t low = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t high = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(low) << 32) | high;
}

bool same_point(const double* a, const double* b) {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}
Dyadic exact_oriented_determinant(const TetMesh& mesh,
                                  const std::int32_t* vertices) {
  std::array<Dyadic, 3> first{};
  std::array<Dyadic, 3> second{};
  std::array<Dyadic, 3> third{};
  const double* origin = mesh.point(vertices[0]);
  const double* a = mesh.point(vertices[1]);
  const double* b = mesh.point(vertices[2]);
  const double* c = mesh.point(vertices[3]);
  for (int axis = 0; axis < 3; ++axis) {
    first[static_cast<std::size_t>(axis)] = Dyadic(a[axis]) - Dyadic(origin[axis]);
    second[static_cast<std::size_t>(axis)] = Dyadic(b[axis]) - Dyadic(origin[axis]);
    third[static_cast<std::size_t>(axis)] = Dyadic(c[axis]) - Dyadic(origin[axis]);
  }
  return first[0] * (second[1] * third[2] - second[2] * third[1]) -
         first[1] * (second[0] * third[2] - second[2] * third[0]) +
         first[2] * (second[0] * third[1] - second[1] * third[0]);
}


TetEditStatus translate_edit_status(EditStatus status) {
  switch (status) {
    case EditStatus::kOk: return TetEditStatus::kApplied;
    case EditStatus::kSlotLimit: return TetEditStatus::kIndexOverflow;
    case EditStatus::kInverted: return TetEditStatus::kInvalidGeometry;
    case EditStatus::kConstraintViolation: return TetEditStatus::kRejectedProtected;
    case EditStatus::kNotLive:
    case EditStatus::kDuplicate:
    case EditStatus::kNonManifold:
    case EditStatus::kBoundaryMismatch:
    case EditStatus::kInternal: return TetEditStatus::kRejectedTopology;
  }
  return TetEditStatus::kRejectedTopology;
}

struct CandidateCell {
  std::array<std::int32_t, 4> vertices{};
  std::array<std::int32_t, 4> constraints{};
  std::int32_t region = kNoRegion;
  bool replacement = false;
};

struct CandidateFace {
  std::array<std::int32_t, 3> oriented{};
  std::int32_t source = kNoConstraint;
  std::uint8_t count = 1;
  std::int32_t region = kNoRegion;
};

bool has_id(std::span<const std::uint64_t> ids, std::uint64_t value) {
  return std::find(ids.begin(), ids.end(), value) != ids.end();
}

// True if the children (start, m, third) and (m, end, third) of the parent
// triangle (start, end, third) tile it exactly: on exact positions (witness
// source points, which the carriers only realize), m lies on the open edge
// (start, end) of the nondegenerate parent.
bool exact_split_coverage(const TetMesh& mesh, const std::int32_t* parent_face,
                          const std::array<std::int32_t, 3>& first_child,
                          const std::array<std::int32_t, 3>& second_child) {
  const std::int32_t start = first_child[0];
  const std::int32_t middle = first_child[1];
  const std::int32_t third = first_child[2];
  const std::int32_t end = second_child[1];
  if (second_child[0] != middle || second_child[2] != third) return false;
  std::array<std::int32_t, 3> parent{parent_face[0], parent_face[1], parent_face[2]};
  std::array<std::int32_t, 3> expected{start, end, third};
  std::sort(parent.begin(), parent.end());
  std::sort(expected.begin(), expected.end());
  if (parent != expected) return false;
  Expansion s[3], e[3], m[3], t[3];
  if (!mesh.exact_position(start, s) || !mesh.exact_position(end, e) ||
      !mesh.exact_position(middle, m) || !mesh.exact_position(third, t)) {
    return false;
  }
  Expansion edge[3], offset[3], apex[3], back[3];
  for (int axis = 0; axis < 3; ++axis) {
    edge[axis] = e[axis] - s[axis];
    offset[axis] = m[axis] - s[axis];
    apex[axis] = t[axis] - s[axis];
    back[axis] = m[axis] - e[axis];
  }
  const auto cross_zero = [](const Expansion* u, const Expansion* v) {
    return (u[1] * v[2] - u[2] * v[1]).is_zero() && (u[2] * v[0] - u[0] * v[2]).is_zero() &&
           (u[0] * v[1] - u[1] * v[0]).is_zero();
  };
  const auto dot = [](const Expansion* u, const Expansion* v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  return !cross_zero(edge, apex) && cross_zero(edge, offset) && dot(offset, edge).sign() > 0 &&
         dot(back, edge).sign() < 0;
}

bool full_link_condition(const TetMesh& mesh, std::int32_t remove,
                         std::int32_t keep) {
  std::set<SortedFace> remove_faces, keep_faces;
  std::set<SortedEdge> remove_edges, keep_edges, edge_edges;
  std::set<std::int32_t> remove_vertices, keep_vertices, edge_vertices;
  const auto add_vertex_link = [](const Tetrahedron& tet, std::int32_t vertex,
                                  std::set<std::int32_t>& vertices,
                                  std::set<SortedEdge>& edges,
                                  std::set<SortedFace>& faces) {
    std::array<std::int32_t, 3> remainder{};
    std::size_t count = 0;
    for (std::int32_t other : tet.v) {
      if (other != vertex) remainder[count++] = other;
    }
    for (std::size_t i = 0; i < count; ++i) {
      vertices.insert(remainder[i]);
      for (std::size_t j = i + 1; j < count; ++j) {
        edges.insert(sorted_edge(remainder[i], remainder[j]));
      }
    }
    if (count == 3) faces.insert(sorted_face(remainder.data()));
  };

  for (std::size_t slot = 0; slot < mesh.complex().tets.size(); ++slot) {
    const std::int32_t id = static_cast<std::int32_t>(slot);
    if (!mesh.complex().live(id)) continue;
    const Tetrahedron& tet = mesh.tet(id);
    const bool has_remove = vertex_slot(tet, remove) >= 0;
    const bool has_keep = vertex_slot(tet, keep) >= 0;
    if (!has_remove && !has_keep) continue;
    if (has_remove) {
      add_vertex_link(tet, remove, remove_vertices, remove_edges, remove_faces);
    }
    if (has_keep) {
      add_vertex_link(tet, keep, keep_vertices, keep_edges, keep_faces);
    }
    if (has_remove && has_keep) {
      std::array<std::int32_t, 2> remainder{};
      std::size_t count = 0;
      for (std::int32_t other : tet.v) {
        if (other != remove && other != keep) remainder[count++] = other;
      }
      if (count != 2) return false;
      edge_vertices.insert(remainder[0]);
      edge_vertices.insert(remainder[1]);
      edge_edges.insert(sorted_edge(remainder[0], remainder[1]));
    }
  }

  std::set<std::int32_t> common_vertices;
  std::set_intersection(remove_vertices.begin(), remove_vertices.end(),
                        keep_vertices.begin(), keep_vertices.end(),
                        std::inserter(common_vertices, common_vertices.end()));
  std::set<SortedEdge> common_edges;
  std::set_intersection(remove_edges.begin(), remove_edges.end(),
                        keep_edges.begin(), keep_edges.end(),
                        std::inserter(common_edges, common_edges.end()));
  std::set<SortedFace> common_faces;
  std::set_intersection(remove_faces.begin(), remove_faces.end(),
                        keep_faces.begin(), keep_faces.end(),
                        std::inserter(common_faces, common_faces.end()));
  return common_vertices == edge_vertices && common_edges == edge_edges &&
         common_faces.empty();
}

bool collect_source_regions(
    const TetMesh& mesh, std::span<const TetSnapshotFacet> facets,
    std::map<std::pair<std::int32_t, int>, std::int32_t>& regions) {
  using IndexedFacet = std::pair<FacetKey, const TetSnapshotFacet*>;
  std::vector<IndexedFacet> indexed_facets;
  indexed_facets.reserve(facets.size());
  for (const TetSnapshotFacet& facet : facets) {
    std::int32_t local[3];
    for (int corner = 0; corner < 3; ++corner) {
      local[corner] = mesh.vertex_slot(facet.vertices[corner]);
      if (local[corner] < 0) return false;
    }
    indexed_facets.emplace_back(FacetKey::of(local), &facet);
  }
  std::sort(indexed_facets.begin(), indexed_facets.end(),
            [](const IndexedFacet& first, const IndexedFacet& second) {
              return first.first < second.first;
            });
  const auto bind_region = [&](const FacetKey& key, int direction,
                               std::int32_t source, std::int32_t region) {
    auto found = std::lower_bound(
        indexed_facets.begin(), indexed_facets.end(), key,
        [](const IndexedFacet& facet, const FacetKey& requested) {
          return facet.first < requested;
        });
    for (; found != indexed_facets.end() && found->first == key; ++found) {
      const TetSnapshotFacet& facet = *found->second;
      if (facet.source != source) continue;
      const auto [entry, inserted] = regions.emplace(
          std::pair{source, direction * facet.orientation}, region);
      if (!inserted && entry->second != region) return false;
    }
    return true;
  };
  const TetrahedralComplex& complex = mesh.complex();
  for (std::size_t row = 0; row < complex.tets.size(); ++row) {
    const std::int32_t slot = static_cast<std::int32_t>(row);
    if (!complex.live(slot)) continue;
    const Tetrahedron& cell = mesh.tet(slot);
    if (is_ghost(cell)) continue;
    for (int face_slot = 0; face_slot < 4; ++face_slot) {
      const std::int32_t source = complex.constraint(slot, face_slot);
      if (source < 0) continue;
      std::int32_t face[3];
      oriented_facet(cell.v, face_slot, face);
      const FacetKey key = FacetKey::of(face);
      if (!bind_region(key, 1, source, complex.region(slot)) ||
          !bind_region(key.reversed(), -1, source, complex.region(slot))) return false;
    }
  }
  return true;
}

}  // namespace

TetEditStatus TetMesh::apply_edit_batch(const TetEditBatch& batch,
                                        std::vector<TetEditResult>& results) {
  try {
    if (batch.expected_generation != generation_ ||
        batch.identity_bound != identity_bound_ ||
        batch.source_id != source_id_ || batch.source_revision != source_revision_ ||
        batch.source_map_id != source_map_id_ || batch.frame_id != frame_id_ ||
        batch.periodic_identity != periodic_identity_) {
      std::vector<TetEditResult> stale(batch.operations.size());
      for (TetEditResult& result : stale) result.status = TetEditStatus::kStaleGeneration;
      results.swap(stale);
      return TetEditStatus::kStaleGeneration;
    }
    if (batch.operations.empty()) {
      results.clear();
      return TetEditStatus::kApplied;
    }
    // An edit keeps whether the state is identity-bound; a bound state is
    // rebound to the bound target identity of the state the edit produces.
    if (batch.target_periodic_identity.has_value() &&
        (batch.target_periodic_identity->bound != periodic_identity_.bound ||
         std::any_of(batch.target_periodic_identity->ids.begin(),
                     batch.target_periodic_identity->ids.end(),
                     [&](const std::string& id) {
                       return batch.target_periodic_identity->bound && id.empty();
                     }))) {
      results.assign(batch.operations.size(), TetEditResult{});
      return TetEditStatus::kRejectedTopology;
    }
    TetMesh staged(policy_);
    copy_state_to(staged);
    std::vector<TetEditResult> staged_results;
    staged_results.reserve(batch.operations.size());
    for (std::size_t index = 0; index < batch.operations.size(); ++index) {
      TetEditResult result;
      const TetEditStatus status = staged.apply_one_edit(batch.operations[index], result);
      result.status = status;
      if (status != TetEditStatus::kApplied) {
        staged_results.resize(batch.operations.size());
        for (std::size_t prior = 0; prior < index; ++prior) {
          staged_results[prior].status = TetEditStatus::kRejectedTopology;
        }
        staged_results[index].status = status;
        for (std::size_t later = index + 1; later < staged_results.size(); ++later) {
          staged_results[later].status = TetEditStatus::kRejectedTopology;
        }
        results.swap(staged_results);
        return status;
      }
      staged_results.push_back(std::move(result));
    }
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
      staged_results.assign(batch.operations.size(), TetEditResult{});
      for (TetEditResult& result : staged_results) result.status = TetEditStatus::kIndexOverflow;
      results.swap(staged_results);
      return TetEditStatus::kIndexOverflow;
    }
    if (!staged.audit_state()) {
      staged_results.assign(batch.operations.size(), TetEditResult{});
      for (TetEditResult& result : staged_results) result.status = TetEditStatus::kRejectedTopology;
      results.swap(staged_results);
      return TetEditStatus::kRejectedTopology;
    }
    staged.generation_ = generation_ + 1;
    if (batch.target_periodic_identity.has_value()) {
      staged.periodic_identity_ = *batch.target_periodic_identity;
    }
    swap_state(staged);
    unmet_records_.clear();
    results.swap(staged_results);
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

TetEditStatus TetMesh::apply_one_edit(const TetEdit& operation, TetEditResult& result) {
  if (!validate_periodic_closure(operation)) return TetEditStatus::kRejectedTopology;
  return validate_and_commit(*this, operation, result);
}

TetEditStatus TetMesh::split_edge(
    std::uint64_t first, std::uint64_t second, const TetEditVertex& inserted,
    std::span<const std::uint64_t> removed_cells,
    std::span<const TetEditCell> replacement_cells,
    std::span<const TetEditFacet> replacement_facets,
    std::span<const TetEditSegment> replacement_segments) {
  TetEdit operation;
  operation.kind = TetEditKind::kSplit;
  operation.split_edges.push_back({first, second});
  operation.changed_vertices.push_back(inserted);
  operation.removed_cell_ids.assign(removed_cells.begin(), removed_cells.end());
  operation.replacement_cells.assign(replacement_cells.begin(), replacement_cells.end());
  operation.replacement_facets.assign(replacement_facets.begin(), replacement_facets.end());
  operation.replacement_segments.assign(replacement_segments.begin(), replacement_segments.end());
  TetEditBatch batch{generation_, source_revision_, frame_id_, identity_bound_,
                     {std::move(operation)}, source_id_, source_map_id_, periodic_identity_,
                     std::nullopt};
  std::vector<TetEditResult> result;
  const TetEditStatus status = apply_edit_batch(batch, result);
  return status == TetEditStatus::kApplied ? result.front().status : status;
}


TetEditStatus TetMesh::reconnect(
    std::span<const std::uint64_t> removed_cells,
    std::span<const TetEditCell> replacement_cells,
    std::span<const TetEditFacet> replacement_facets,
    std::span<const TetEditSegment> replacement_segments) {
  TetEdit operation;
  operation.kind = TetEditKind::kRingReconnect;
  operation.removed_cell_ids.assign(removed_cells.begin(), removed_cells.end());
  operation.replacement_cells.assign(replacement_cells.begin(), replacement_cells.end());
  operation.replacement_facets.assign(replacement_facets.begin(), replacement_facets.end());
  operation.replacement_segments.assign(replacement_segments.begin(), replacement_segments.end());
  TetEditBatch batch{generation_, source_revision_, frame_id_, identity_bound_,
                     {std::move(operation)}, source_id_, source_map_id_, periodic_identity_,
                     std::nullopt};
  std::vector<TetEditResult> result;
  const TetEditStatus status = apply_edit_batch(batch, result);
  return status == TetEditStatus::kApplied ? result.front().status : status;
}

TetEditStatus TetMesh::relocate(
    std::span<const TetEditVertex> moved_vertices,
    std::span<const std::uint64_t> affected_cells) {
  return relocate_impl(moved_vertices, affected_cells, {}, false);
}

TetEditStatus TetMesh::relocate_with_complete_star(
    std::span<const TetEditVertex> moved_vertices,
    std::span<const std::int32_t> star_slots) {
  return relocate_impl(moved_vertices, {}, star_slots, true);
}

TetEditStatus TetMesh::relocate_impl(
    std::span<const TetEditVertex> moved_vertices,
    std::span<const std::uint64_t> affected_cells,
    std::span<const std::int32_t> complete_star, bool star_verified) {
  // This pure-relocation path is separate from atomic mixed/topology batches.
  try {
    const RelocationOverlay overlay{moved_vertices};
    if (periodic_identity_.bound || moved_vertices.empty()) {
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
    for (const TetPeriodicOrbit& orbit : periodic_orbits_) {
      for (const TetPeriodicVertexMember& member : orbit.members) {
        if (overlay.find(member.vertex_id) != nullptr) {
          return TetEditStatus::kRejectedTopology;
        }
      }
    }

    std::vector<std::int32_t> resolved_slots;
    std::span<const std::int32_t> affected_slots;
    if (star_verified) {
      if (!affected_cells.empty()) return TetEditStatus::kRejectedTopology;
      affected_slots = complete_star;
      std::uint64_t prior_id = 0;
      bool has_prior_id = false;
      for (std::int32_t slot : affected_slots) {
        if (slot < 0 || !complex_.live(slot) || is_ghost(tet(slot))) {
          return TetEditStatus::kRejectedTopology;
        }
        const std::uint64_t id = cell_id(slot);
        if (has_prior_id && id <= prior_id) {
          return TetEditStatus::kRejectedTopology;
        }
        prior_id = id;
        has_prior_id = true;
      }
    } else {
      if (!complete_star.empty()) return TetEditStatus::kRejectedTopology;
      resolved_slots.reserve(affected_cells.size());
      for (std::uint64_t id : affected_cells) {
        const std::int32_t slot = cell_slot(id);
        if (slot < 0 || !complex_.live(slot) || is_ghost(tet(slot))) {
          return TetEditStatus::kRejectedTopology;
        }
        resolved_slots.push_back(slot);
      }
      std::sort(resolved_slots.begin(), resolved_slots.end());
      if (std::adjacent_find(resolved_slots.begin(), resolved_slots.end()) !=
          resolved_slots.end()) {
        return TetEditStatus::kRejectedTopology;
      }
      affected_slots = resolved_slots;
    }

    struct Candidate {
      std::int32_t slot;
      const TetEditVertex* edit;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(moved_vertices.size());
    for (const TetEditVertex& vertex : moved_vertices) {
      if (vertex.id == std::numeric_limits<std::uint64_t>::max()) {
        return TetEditStatus::kIndexOverflow;
      }
      if (std::any_of(candidates.begin(), candidates.end(),
                      [&](const Candidate& prior) {
                        return prior.edit->id == vertex.id;
                      })) {
        return TetEditStatus::kRejectedTopology;
      }
      const std::int32_t slot = vertex_slot(vertex.id);
      if (slot < 0) return TetEditStatus::kRejectedTopology;
      if (!vertex.ancestry.empty()) return TetEditStatus::kRejectedTopology;
      if (protected_vertex(slot) || protection_radius(slot) > 0.0) {
        return TetEditStatus::kRejectedProtected;
      }
      if (vertex_dimension(slot) < 3 &&
          witness(slot).stratum == SourceStratum::kNone) {
        return TetEditStatus::kUncertainPredicate;
      }
      if (vertex.protected_vertex != protected_vertex(slot) ||
          vertex.protection_radius != protection_radius(slot)) {
        return TetEditStatus::kRejectedProtected;
      }
      if (vertex.witness.stratum != witness(slot).stratum) {
        return TetEditStatus::kRejectedProtected;
      }
      TetEditVertex checked_vertex = vertex;
      if (!checked_vertex.edge_witness_supplied &&
          !checked_vertex.has_edge_parameter &&
          vertex_has_edge_parameter_[static_cast<std::size_t>(slot)] != 0) {
        checked_vertex.has_edge_parameter = true;
        checked_vertex.edge_endpoints =
            vertex_edge_endpoints_[static_cast<std::size_t>(slot)];
        checked_vertex.edge_parameter =
            vertex_edge_parameters_[static_cast<std::size_t>(slot)];
      }
      if (!std::isfinite(vertex.protection_radius) ||
          vertex.protection_radius < 0.0 ||
          !source_carrier_valid(checked_vertex)) {
        return TetEditStatus::kInvalidGeometry;
      }
      if (vertex.protection_radius > 0.0 && !vertex.protected_vertex) {
        return TetEditStatus::kRejectedProtected;
      }
      if (!star_verified) {
        for (std::size_t other = 0; other < vertex_ids_.size(); ++other) {
          if (!vertex_live_[other] || vertex_ids_[other] == vertex.id) continue;
          if (same_point(points_.data() + 3 * other, vertex.carrier.data())) {
            return TetEditStatus::kInvalidGeometry;
          }
        }
      }
      candidates.push_back({slot, &vertex});
    }
    if (star_verified) {
      if (candidates.size() != 1) return TetEditStatus::kRejectedTopology;
      const Candidate& candidate = candidates.front();
      if (vertex_dimension(candidate.slot) != 3 ||
          witness(candidate.slot).stratum != SourceStratum::kNone ||
          candidate.edit->witness.stratum != SourceStratum::kNone) {
        return TetEditStatus::kRejectedTopology;
      }
    }

    std::vector<std::int32_t> moved_slots;
    moved_slots.reserve(candidates.size());
    for (const Candidate& candidate : candidates) {
      moved_slots.push_back(candidate.slot);
    }
    std::sort(moved_slots.begin(), moved_slots.end());
    if (std::adjacent_find(moved_slots.begin(), moved_slots.end()) !=
        moved_slots.end()) {
      return TetEditStatus::kRejectedTopology;
    }

    if (affected_slots.empty()) return TetEditStatus::kRejectedTopology;
    if (star_verified) {
      if (candidates.size() != 1) return TetEditStatus::kRejectedTopology;
      const std::int32_t moved = candidates.front().slot;
      for (std::int32_t cell : affected_slots) {
        const Tetrahedron& cell_tet = this->tet(cell);
        bool contains_moved = false;
        for (std::int32_t vertex : cell_tet.v) {
          contains_moved = contains_moved || vertex == moved;
        }
        if (!contains_moved) return TetEditStatus::kRejectedTopology;
      }
    } else {
      const auto moved_slot = [&](std::int32_t slot) {
        return std::binary_search(moved_slots.begin(), moved_slots.end(), slot);
      };
      for (std::size_t index = 0; index < complex_.tets.size(); ++index) {
        const std::int32_t cell = static_cast<std::int32_t>(index);
        if (!complex_.live(cell) || is_ghost(tet(cell))) continue;
        bool affected = false;
        for (std::int32_t vertex : tet(cell).v) {
          affected = affected || moved_slot(vertex);
        }
        if (affected &&
            !std::binary_search(affected_slots.begin(), affected_slots.end(), cell)) {
          return TetEditStatus::kRejectedTopology;
        }
      }
    }

    if (!star_verified) {
      for (const TetSnapshotFacet& facet : facets_) {
        if (std::none_of(facet.vertices.begin(), facet.vertices.end(),
                         [&](std::uint64_t id) { return overlay.find(id) != nullptr; })) {
          continue;
        }
        TetEditFacet check;
        check.id = facet.id;
        check.vertices = facet.vertices;
        check.source = facet.source;
        check.orientation = facet.orientation;
        if (!source_face_valid(check, &overlay)) {
          return TetEditStatus::kUncertainPredicate;
        }
      }
      for (const TetSnapshotSegment& segment : segments_) {
        if (std::none_of(segment.vertices.begin(), segment.vertices.end(),
                         [&](std::uint64_t id) { return overlay.find(id) != nullptr; })) {
          continue;
        }
        TetEditSegment check;
        check.id = segment.id;
        check.vertices = segment.vertices;
        check.source = segment.source;
        if (!source_segment_valid(check, &overlay)) {
          return TetEditStatus::kUncertainPredicate;
        }
      }
    } else {
      // Dimension three plus kNone witnesses excludes source facets/segments by
      // build_state's source-stratum invariant. The complete fixed link and
      // positive checks below keep the moved point inside its link polyhedron's
      // kernel, so its star union is unchanged and no disjoint source entity can
      // be crossed.
    }
    for (std::int32_t slot : affected_slots) {
      if (!positive_with_relocation(tet(slot).v, overlay)) {
        return TetEditStatus::kInvalidGeometry;
      }
    }
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return TetEditStatus::kIndexOverflow;
    }

    // The overlay has now proved every changed carrier/witness, incident cell,
    // and source entity. Topology and all ID/source banks are unchanged, so the
    // commit below only writes existing vertex records and cannot allocate.
    for (const Candidate& candidate : candidates) {
      const TetEditVertex& vertex = *candidate.edit;
      const std::size_t slot = static_cast<std::size_t>(candidate.slot);
      std::copy(vertex.carrier.begin(), vertex.carrier.end(),
                points_.begin() + 3 * slot);
      witnesses_[slot] = vertex.witness;
      vertex_ancestry_.erase(vertex.id);
      if (vertex.has_edge_parameter) {
        vertex_has_edge_parameter_[slot] = 1;
        vertex_edge_endpoints_[slot] = vertex.edge_endpoints;
        vertex_edge_parameters_[slot] = vertex.edge_parameter;
      } else if (vertex.edge_witness_supplied) {
        vertex_has_edge_parameter_[slot] = 0;
        vertex_edge_endpoints_[slot] = {};
        vertex_edge_parameters_[slot] = 0.0;
      }
    }
    ++generation_;
    unmet_records_.clear();
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

// Validity of an interior reconnection or collapse is local. CavityEdit matches
// every cavity boundary face, with its inherited constraint, to exactly one
// replacement face, pairs every other replacement face with its reversed twin,
// and checks exact positive orientation. A positive simplicial chain whose
// boundary is the cavity boundary covers the cavity exactly once, so the
// replacement tiles the cavity and conforms to the unchanged outside cells. A
// collapsed vertex has dimension three and no source stratum, so no source
// facet or segment contains it; it may not appear in a replacement cell, and
// any cell of its star outside the cavity would leave an unmatched boundary
// face containing it. Constrained interior faces must be reproduced, and an
// edge removed by the edit must not be a constrained segment. All allocation
// precedes the first topology write; staged ID records are rolled back if the
// commit cannot start.
TetEditStatus TetMesh::replace_local_cavity(const TetEdit& operation) {
  try {
    const bool collapse = operation.kind == TetEditKind::kCollapse;
    const bool reconnection = operation.kind == TetEditKind::kFlip23 ||
                              operation.kind == TetEditKind::kFlip32 ||
                              operation.kind == TetEditKind::kRingReconnect;
    if ((!collapse && !reconnection) || periodic_identity_.bound ||
        !periodic_orbits_.empty() || !quotient_cells_.empty() ||
        operation.removed_cell_ids.empty() || operation.replacement_cells.empty() ||
        !operation.changed_vertices.empty() || !operation.split_edges.empty() ||
        !operation.removed_facet_ids.empty() || !operation.replacement_facets.empty() ||
        !operation.removed_segment_ids.empty() ||
        !operation.replacement_segments.empty() || !operation.periodic_members.empty() ||
        !operation.removed_periodic_orbit_ids.empty() ||
        !operation.replacement_periodic_orbits.empty() ||
        !operation.removed_quotient_cell_ids.empty() ||
        !operation.replacement_quotient_cells.empty() ||
        !operation.replacement_periodic_entity_maps.empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    std::int32_t removed_vertex = -1;
    if (collapse) {
      if (operation.removed_vertex_ids.size() != 1 ||
          operation.collapse_pairs.size() != 1 ||
          operation.collapse_pairs.front().removed_vertex_id !=
              operation.removed_vertex_ids.front()) {
        return TetEditStatus::kRejectedTopology;
      }
      removed_vertex = vertex_slot(operation.removed_vertex_ids.front());
      const std::int32_t survivor =
          vertex_slot(operation.collapse_pairs.front().survivor_vertex_id);
      if (!live_vertex(removed_vertex) || !live_vertex(survivor) ||
          removed_vertex == survivor) {
        return TetEditStatus::kRejectedTopology;
      }
      if (protected_vertex(removed_vertex) || protection_radius(removed_vertex) > 0.0 ||
          protected_vertex(survivor) || protection_radius(survivor) > 0.0) {
        return TetEditStatus::kRejectedProtected;
      }
      if (vertex_dimension(removed_vertex) != 3 ||
          witness(removed_vertex).stratum != SourceStratum::kNone) {
        return TetEditStatus::kRejectedTopology;
      }
    } else if (!operation.removed_vertex_ids.empty() || !operation.collapse_pairs.empty()) {
      return TetEditStatus::kRejectedTopology;
    }

    std::vector<std::int32_t> removed_slots;
    removed_slots.reserve(operation.removed_cell_ids.size());
    std::int32_t region = kNoRegion;
    for (std::uint64_t id : operation.removed_cell_ids) {
      const std::int32_t slot = cell_slot(id);
      if (slot < 0 || !complex_.live(slot) || is_ghost(tet(slot))) {
        return TetEditStatus::kRejectedTopology;
      }
      if (removed_slots.empty()) {
        region = complex_.region(slot);
      } else if (complex_.region(slot) != region) {
        return TetEditStatus::kRejectedTopology;
      }
      removed_slots.push_back(slot);
    }
    std::vector<std::int32_t> sorted_removed = removed_slots;
    std::sort(sorted_removed.begin(), sorted_removed.end());
    if (std::adjacent_find(sorted_removed.begin(), sorted_removed.end()) !=
        sorted_removed.end()) {
      return TetEditStatus::kRejectedTopology;
    }

    const std::size_t created = operation.replacement_cells.size();
    std::vector<std::int32_t> replacement_vertices;
    replacement_vertices.reserve(4 * created);
    std::set<std::uint64_t> replacement_edges;
    for (std::size_t row = 0; row < created; ++row) {
      const TetEditCell& cell = operation.replacement_cells[row];
      if (cell.id == std::numeric_limits<std::uint64_t>::max() ||
          used_cell_ids_.contains(cell.id) || cell.region != region) {
        return TetEditStatus::kRejectedTopology;
      }
      for (std::size_t prior = 0; prior < row; ++prior) {
        if (operation.replacement_cells[prior].id == cell.id) {
          return TetEditStatus::kRejectedTopology;
        }
      }
      for (std::size_t ancestor = 0; ancestor < cell.ancestor_cell_ids.size(); ++ancestor) {
        const std::uint64_t id = cell.ancestor_cell_ids[ancestor];
        if (!has_id(operation.removed_cell_ids, id) ||
            has_id(std::span(cell.ancestor_cell_ids).first(ancestor), id)) {
          return TetEditStatus::kRejectedTopology;
        }
      }
      for (int corner = 0; corner < 4; ++corner) {
        const std::int32_t slot = vertex_slot(cell.vertices[static_cast<std::size_t>(corner)]);
        if (!live_vertex(slot) || slot == removed_vertex ||
            cell.facet_sources[static_cast<std::size_t>(corner)] < kNoConstraint) {
          return TetEditStatus::kRejectedTopology;
        }
        replacement_vertices.push_back(slot);
      }
      const std::int32_t* local = replacement_vertices.data() + 4 * row;
      for (int first = 0; first < 4; ++first) {
        for (int second = first + 1; second < 4; ++second) {
          replacement_edges.insert(edge_key(local[first], local[second]));
        }
      }
    }
    if (removes_constrained_segment(removed_slots, replacement_edges)) {
      return TetEditStatus::kRejectedProtected;
    }
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return TetEditStatus::kIndexOverflow;
    }

    edit_.begin();
    for (std::int32_t slot : removed_slots) {
      const EditStatus status = edit_.remove(slot);
      if (status != EditStatus::kOk) {
        edit_.rollback();
        return translate_edit_status(status);
      }
    }
    for (std::size_t row = 0; row < created; ++row) {
      const EditStatus status =
          edit_.add(replacement_vertices.data() + 4 * row,
                    operation.replacement_cells[row].facet_sources.data(), region);
      if (status != EditStatus::kOk) {
        edit_.rollback();
        return translate_edit_status(status);
      }
    }
    const EditStatus validated = edit_.validate(
        [this](const std::int32_t* vertices) { return positive(vertices); });
    if (validated != EditStatus::kOk) {
      edit_.rollback();
      return translate_edit_status(validated);
    }

    // CavityEdit::commit reuses removed slots, then free slots, then appends.
    const std::size_t extra = created > removed_slots.size() ? created - removed_slots.size() : 0;
    const std::size_t fresh = extra - std::min(extra, complex_.free_slots.size());
    const std::size_t cell_id_count = cell_ids_.size();
    // Rows below `staged` may hold staged ID records; replacement IDs were
    // unused, so erasing a row's keys removes only what this edit inserted.
    std::size_t staged = 0;
    const auto unstage = [&]() noexcept {
      for (std::size_t row = 0; row < staged; ++row) {
        const std::uint64_t id = operation.replacement_cells[row].id;
        used_cell_ids_.erase(id);
        cell_slots_.erase(id);
        cell_ancestry_.erase(id);
      }
      cell_ids_.resize(cell_id_count);
    };
    EditStatus committed = EditStatus::kInternal;
    try {
      if (collapse) used_vertex_ids_.insert(vertex_id(removed_vertex));
      cell_ids_.resize(complex_.tets.size() + fresh,
                       std::numeric_limits<std::uint64_t>::max());
      while (staged < created) {
        const TetEditCell& cell = operation.replacement_cells[staged++];
        used_cell_ids_.insert(cell.id);
        cell_slots_.emplace(cell.id, -1);
        cell_ancestry_.emplace(
            cell.id, TetCellAncestryRecord{static_cast<std::int32_t>(operation.kind),
                                           cell.ancestor_cell_ids});
      }
      committed = edit_.commit();
    } catch (...) {
      edit_.rollback();
      unstage();
      throw;
    }
    if (committed != EditStatus::kOk) {
      edit_.rollback();
      unstage();
      return translate_edit_status(committed);
    }

    for (std::uint64_t id : operation.removed_cell_ids) cell_slots_.erase(id);
    for (std::int32_t slot : removed_slots) {
      cell_ids_[static_cast<std::size_t>(slot)] = std::numeric_limits<std::uint64_t>::max();
    }
    const std::span<const std::int32_t> created_slots = edit_.created();
    for (std::size_t row = 0; row < created; ++row) {
      const std::uint64_t id = operation.replacement_cells[row].id;
      const std::int32_t slot = created_slots[row];
      cell_ids_[static_cast<std::size_t>(slot)] = id;
      cell_slots_.find(id)->second = slot;
      if (id >= next_cell_id_) next_cell_id_ = id + 1;
    }
    if (collapse) {
      const std::uint64_t id = vertex_id(removed_vertex);
      vertex_live_[static_cast<std::size_t>(removed_vertex)] = 0;
      vertex_slots_.erase(id);
      vertex_ancestry_.erase(id);
      vertex_sizes_.erase(id);
    }
    ++generation_;
    unmet_records_.clear();
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

bool TetMesh::removes_constrained_segment(std::span<const std::int32_t> removed_slots,
                                          const std::set<std::uint64_t>& kept_edges) const {
  // Segment endpoints carry dimension at most one, so only such edges need
  // the segment bank.
  for (std::int32_t slot : removed_slots) {
    const Tetrahedron& cell = tet(slot);
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        const std::int32_t a = cell.v[first];
        const std::int32_t b = cell.v[second];
        if (kept_edges.contains(edge_key(a, b)) || vertex_dimension(a) > 1 ||
            vertex_dimension(b) > 1) {
          continue;
        }
        const std::uint64_t id_a = vertex_id(a);
        const std::uint64_t id_b = vertex_id(b);
        if (std::any_of(segments_.begin(), segments_.end(),
                        [&](const TetSnapshotSegment& segment) {
                          return (segment.vertices[0] == id_a && segment.vertices[1] == id_b) ||
                                 (segment.vertices[0] == id_b && segment.vertices[1] == id_a);
                        })) {
          return true;
        }
      }
    }
  }
  return false;
}

// A star-shaped cavity coned to a new interior vertex. Every cone cell must be
// exactly positive; the cone then tiles the cavity (same chain argument as
// replace_local_cavity), the new point lies in the cavity interior and so
// coincides with no vertex, and every cavity vertex must remain on the
// cavity boundary. Constrained faces may bound the cavity but not lie inside it.
TetEditStatus TetMesh::insert_local_cone(const TetEditVertex& vertex,
                                         std::span<const std::int32_t> cavity,
                                         std::vector<std::uint64_t>& created_cells) {
  created_cells.clear();
  try {
    if (periodic_identity_.bound || !periodic_orbits_.empty() || !quotient_cells_.empty() ||
        cavity.empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    if (vertex.id == std::numeric_limits<std::uint64_t>::max() ||
        used_vertex_ids_.contains(vertex.id) || !vertex.ancestry.empty() ||
        vertex.witness.stratum != SourceStratum::kNone || vertex.protected_vertex ||
        vertex.protection_radius != 0.0 || vertex.has_edge_parameter) {
      return TetEditStatus::kRejectedTopology;
    }
    for (double coordinate : vertex.carrier) {
      if (!std::isfinite(coordinate) || !coordinate_in_domain(coordinate)) {
        return TetEditStatus::kInvalidGeometry;
      }
    }
    if (vertex_ids_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      return TetEditStatus::kIndexOverflow;
    }

    std::vector<std::int32_t> sorted_cavity(cavity.begin(), cavity.end());
    std::sort(sorted_cavity.begin(), sorted_cavity.end());
    if (std::adjacent_find(sorted_cavity.begin(), sorted_cavity.end()) != sorted_cavity.end()) {
      return TetEditStatus::kRejectedTopology;
    }
    const auto in_cavity = [&](std::int32_t slot) {
      return std::binary_search(sorted_cavity.begin(), sorted_cavity.end(), slot);
    };
    std::int32_t region = kNoRegion;
    std::vector<BoundaryFacet> boundary;
    boundary.reserve(2 * cavity.size() + 2);
    std::set<std::uint64_t> kept_edges;
    std::vector<std::uint64_t> ancestors;
    ancestors.reserve(cavity.size());
    for (std::int32_t slot : cavity) {
      if (!complex_.live(slot) || is_ghost(tet(slot))) return TetEditStatus::kRejectedTopology;
      if (slot == cavity.front()) {
        region = complex_.region(slot);
      } else if (complex_.region(slot) != region) {
        return TetEditStatus::kRejectedTopology;
      }
      ancestors.push_back(cell_id(slot));
      const Tetrahedron& cell = tet(slot);
      for (int face = 0; face < 4; ++face) {
        const std::int32_t outside = cell.n[face];
        if (in_cavity(outside)) {
          if (complex_.constraint(slot, face) != kNoConstraint) {
            return TetEditStatus::kRejectedProtected;
          }
          continue;
        }
        if (!complex_.live(outside)) return TetEditStatus::kRejectedTopology;
        const int back = neighbor_slot(tet(outside), slot);
        if (back < 0) return TetEditStatus::kRejectedTopology;
        boundary.push_back({slot, face, outside, back});
        std::int32_t facet[3];
        oriented_facet(cell.v, face, facet);
        for (int corner = 0; corner < 3; ++corner) {
          kept_edges.insert(edge_key(facet[corner], facet[(corner + 1) % 3]));
        }
      }
    }
    std::sort(ancestors.begin(), ancestors.end());
    if (removes_constrained_segment(cavity, kept_edges)) {
      return TetEditStatus::kRejectedProtected;
    }
    const std::int32_t center = static_cast<std::int32_t>(vertex_ids_.size());
    if (boundary.size() > std::numeric_limits<std::uint64_t>::max() - next_cell_id_ ||
        generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return TetEditStatus::kIndexOverflow;
    }

    // Stage the vertex record so exact orientation can address it; every
    // staged record is removed again unless the cone commits.
    const std::size_t vertex_count = vertex_ids_.size();
    const std::size_t cell_id_count = cell_ids_.size();
    std::size_t staged_cells = 0;
    bool vertex_staged = false;
    const auto unstage = [&]() noexcept {
      for (std::size_t row = 0; row < staged_cells; ++row) {
        const std::uint64_t id = next_cell_id_ + row;
        used_cell_ids_.erase(id);
        cell_slots_.erase(id);
        cell_ancestry_.erase(id);
      }
      cell_ids_.resize(cell_id_count);
      if (vertex_staged) {
        vertex_slots_.erase(vertex.id);
        used_vertex_ids_.erase(vertex.id);
      }
      points_.resize(3 * vertex_count);
      vertex_ids_.resize(vertex_count);
      vertex_live_.resize(vertex_count);
      vertex_dimension_.resize(vertex_count);
      protection_radii_.resize(vertex_count);
      protected_vertices_.resize(vertex_count);
      witnesses_.resize(vertex_count);
      vertex_has_edge_parameter_.resize(vertex_count);
      vertex_edge_endpoints_.resize(vertex_count);
      vertex_edge_parameters_.resize(vertex_count);
    };
    ConeInsertion inserted = ConeInsertion::kInternal;
    try {
      points_.insert(points_.end(), vertex.carrier.begin(), vertex.carrier.end());
      vertex_ids_.push_back(vertex.id);
      vertex_live_.push_back(1);
      vertex_dimension_.push_back(3);
      protection_radii_.push_back(0.0);
      protected_vertices_.push_back(0);
      witnesses_.push_back(vertex.witness);
      vertex_has_edge_parameter_.push_back(0);
      vertex_edge_endpoints_.push_back({});
      vertex_edge_parameters_.push_back(0.0);
      vertex_staged = true;
      vertex_slots_.emplace(vertex.id, center);
      used_vertex_ids_.insert(vertex.id);
      const std::size_t created = boundary.size();
      const std::size_t extra = created > cavity.size() ? created - cavity.size() : 0;
      const std::size_t fresh = extra - std::min(extra, complex_.free_slots.size());
      cell_ids_.resize(complex_.tets.size() + fresh, std::numeric_limits<std::uint64_t>::max());
      while (staged_cells < created) {
        const std::uint64_t id = next_cell_id_ + staged_cells++;
        if (used_cell_ids_.contains(id)) {
          --staged_cells;
          unstage();
          return TetEditStatus::kRejectedTopology;
        }
        used_cell_ids_.insert(id);
        cell_slots_.emplace(id, -1);
        cell_ancestry_.emplace(
            id, TetCellAncestryRecord{static_cast<std::int32_t>(TetEditKind::kRefineCell),
                                      ancestors});
      }
      created_cells.reserve(created);
      inserted = insert_checked_cone(
          complex_, edit_, [this](const std::int32_t* v) { return positive(v); }, center,
          cavity, boundary, false);
    } catch (...) {
      edit_.rollback();
      unstage();
      throw;
    }
    switch (inserted) {
      case ConeInsertion::kApplied: break;
      case ConeInsertion::kInvertedCell:
        unstage();
        return TetEditStatus::kInvalidGeometry;
      case ConeInsertion::kRemovesVertex:
        unstage();
        return TetEditStatus::kRejectedTopology;
      case ConeInsertion::kSlotLimit:
        unstage();
        return TetEditStatus::kIndexOverflow;
      case ConeInsertion::kInternal:
        // Only the cone star link can fail after the topology write; exact
        // positivity of every cone cell excludes it.
        return TetEditStatus::kStaleGeneration;
    }
    for (std::int32_t slot : cavity) {
      cell_slots_.erase(cell_id(slot));
      cell_ids_[static_cast<std::size_t>(slot)] = std::numeric_limits<std::uint64_t>::max();
    }
    const std::span<const std::int32_t> created_slots = edit_.created();
    for (std::size_t row = 0; row < created_slots.size(); ++row) {
      const std::uint64_t id = next_cell_id_ + row;
      const std::int32_t slot = created_slots[row];
      cell_ids_[static_cast<std::size_t>(slot)] = id;
      cell_slots_.find(id)->second = slot;
      created_cells.push_back(id);
    }
    next_cell_id_ += created_slots.size();
    if (vertex.id >= next_vertex_id_) next_vertex_id_ = vertex.id + 1;
    ++generation_;
    unmet_records_.clear();
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

// A constrained boundary point p (a source witness whose exact point S is
// authoritative) inserted into the cavity `cavity` of finite and hull ghost
// cells around it: on a facet (a facet-stratum witness, or a plane-stratum
// witness on the facet's declared plane), or on the constrained segment
// (a, b) of `segment_rows` (segment-stratum witness on its source).
// The constrained faces strictly inside the cavity (`facet_rows`) are disks
// of their facets, one around p in its facet, or one per facet through
// (a, b) on that facet's side of the segment; each is replaced by the fan of
// p over its boundary edges other than (a, b), and every cavity boundary face
// is coned to p. Cone faces through p and a disk boundary edge carry that
// edge's source on both sides; each fan triangle inherits the source,
// orientation and vertex order of the subfacet it replaces along that edge
// and must lie on its source with that orientation (source_face_valid, on
// exact positions), so the fans, closed around p, tile the disks exactly and
// the domain boundary does not move. The segment is replaced by its halves
// (a, p) and (p, b), which must lie on its source. CavityEdit proves that the
// cone conforms to the unchanged outside cells and that every finite cone
// cell is exactly positive; edges inside a disk and the split segment must
// leave the mesh, no other segment may, and every cavity vertex stays on the
// cavity boundary. Fan triangles overwrite the facet rows in order and the
// rest are appended; the first segment half overwrites the segment row and
// the second is appended.
TetEditStatus TetMesh::insert_local_boundary_point(const TetEditVertex& vertex,
                                                   std::span<const std::int32_t> cavity,
                                                   std::span<const std::size_t> facet_rows,
                                                   std::span<const std::size_t> segment_rows,
                                                   std::vector<std::uint64_t>& created_cells) {
  created_cells.clear();
  try {
    if (policy_ == BoundaryPolicy::kFixed) return TetEditStatus::kRejectedProtected;
    if (periodic_identity_.bound || !periodic_orbits_.empty() || !quotient_cells_.empty() ||
        cavity.empty() || facet_rows.empty() || segment_rows.size() > 1) {
      return TetEditStatus::kRejectedTopology;
    }
    const SourceStratum stratum =
        !segment_rows.empty() ? SourceStratum::kSegment
        : vertex.witness.stratum == SourceStratum::kPlane ? SourceStratum::kPlane
                                                          : SourceStratum::kFacet;
    if (vertex.id == std::numeric_limits<std::uint64_t>::max() ||
        used_vertex_ids_.contains(vertex.id) || vertex.protected_vertex ||
        vertex.protection_radius != 0.0 || vertex.has_edge_parameter ||
        !vertex.ancestry.empty() || vertex.witness.stratum != stratum ||
        !source_carrier_valid(vertex)) {
      return TetEditStatus::kRejectedTopology;
    }
    if (vertex_ids_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      return TetEditStatus::kIndexOverflow;
    }
    std::vector<std::int32_t> sorted_cavity(cavity.begin(), cavity.end());
    std::sort(sorted_cavity.begin(), sorted_cavity.end());
    if (std::adjacent_find(sorted_cavity.begin(), sorted_cavity.end()) != sorted_cavity.end()) {
      return TetEditStatus::kRejectedTopology;
    }
    const auto in_cavity = [&](std::int32_t slot) {
      return std::binary_search(sorted_cavity.begin(), sorted_cavity.end(), slot);
    };

    // The split segment.
    std::array<std::int32_t, 2> split{-1, -1};
    std::uint64_t split_key = 0;
    std::int32_t segment_source = kNoConstraint;
    if (!segment_rows.empty()) {
      const std::size_t row = segment_rows.front();
      if (row >= segments_.size()) return TetEditStatus::kRejectedTopology;
      const TetSnapshotSegment& segment = segments_[row];
      if (segment.protected_segment) return TetEditStatus::kRejectedProtected;
      split = {vertex_slot(segment.vertices[0]), vertex_slot(segment.vertices[1])};
      if (!live_vertex(split[0]) || !live_vertex(split[1])) return TetEditStatus::kRejectedTopology;
      split_key = edge_key(split[0], split[1]);
      segment_source = segment.source;
      const std::size_t entity = static_cast<std::size_t>(vertex.witness.entity);
      if (entity >= source_segment_groups_.size() ||
          source_segment_groups_[entity] != segment_source) {
        return TetEditStatus::kInvalidGeometry;
      }
    }

    // The disks: their faces, their interior edges and their directed boundaries.
    struct DiskEdge {
      std::int32_t from = -1;
      std::int32_t to = -1;
      std::int32_t source = kNoConstraint;
      std::int8_t orientation = 1;
    };
    std::map<SortedFace, std::int32_t> disk_faces;
    std::map<std::uint64_t, int> edge_uses;
    std::map<std::uint64_t, DiskEdge> directed;
    for (std::size_t row : facet_rows) {
      if (row >= facets_.size()) return TetEditStatus::kRejectedTopology;
      const TetSnapshotFacet& facet = facets_[row];
      if (facet.protected_facet) return TetEditStatus::kRejectedProtected;
      if (facet.source == kNoConstraint) return TetEditStatus::kRejectedTopology;
      std::int32_t slots[3];
      for (int corner = 0; corner < 3; ++corner) {
        slots[corner] = vertex_slot(facet.vertices[static_cast<std::size_t>(corner)]);
        if (!live_vertex(slots[corner])) return TetEditStatus::kRejectedTopology;
      }
      if (!disk_faces.emplace(sorted_face(slots), facet.source).second) {
        return TetEditStatus::kRejectedTopology;
      }
      for (int corner = 0; corner < 3; ++corner) {
        const std::int32_t from = slots[corner];
        const std::int32_t to = slots[(corner + 1) % 3];
        const std::uint64_t key = edge_key(from, to);
        ++edge_uses[key];
        directed[key] = {from, to, facet.source, facet.orientation};
      }
    }
    std::vector<DiskEdge> polygon;
    std::map<std::uint64_t, std::int32_t> polygon_sources;
    std::set<std::uint64_t> interior_edges;
    for (const auto& [key, uses] : edge_uses) {
      if (key == split_key && split[0] >= 0) continue;
      if (uses > 2) return TetEditStatus::kRejectedTopology;
      if (uses == 2) {
        interior_edges.insert(key);
      } else {
        polygon.push_back(directed[key]);
        polygon_sources.emplace(key, directed[key].source);
      }
    }
    if (split[0] >= 0 && !edge_uses.contains(split_key)) return TetEditStatus::kRejectedTopology;
    if (polygon.size() < facet_rows.size()) return TetEditStatus::kRejectedTopology;
    std::vector<BoundaryFacet> boundary;
    std::vector<std::int32_t> finite;
    std::vector<std::int32_t> ghosts;
    std::set<std::uint64_t> kept_edges;
    std::set<std::int32_t> kept_vertices;
    std::vector<std::pair<std::int32_t, int>> disk_sides;
    std::vector<std::uint64_t> ancestors;
    for (std::int32_t slot : sorted_cavity) {
      if (!complex_.live(slot)) return TetEditStatus::kRejectedTopology;
      const Tetrahedron& cell = tet(slot);
      (is_ghost(cell) ? ghosts : finite).push_back(slot);
      if (!is_ghost(cell)) ancestors.push_back(cell_id(slot));
      for (int face = 0; face < 4; ++face) {
        const std::int32_t outside = cell.n[face];
        if (in_cavity(outside)) {
          const std::int32_t constraint = complex_.constraint(slot, face);
          if (constraint != kNoConstraint) {
            std::int32_t corners[3];
            oriented_facet(cell.v, face, corners);
            const auto disk_face = disk_faces.find(sorted_face(corners));
            if (disk_face == disk_faces.end() || disk_face->second != constraint) {
              return TetEditStatus::kRejectedProtected;
            }
            disk_sides.emplace_back(slot, face);
          }
          continue;
        }
        if (!complex_.live(outside)) return TetEditStatus::kRejectedTopology;
        const int back = neighbor_slot(tet(outside), slot);
        if (back < 0) return TetEditStatus::kRejectedTopology;
        boundary.push_back({slot, face, outside, back});
        std::int32_t corners[3];
        oriented_facet(cell.v, face, corners);
        for (int corner = 0; corner < 3; ++corner) {
          if (corners[corner] >= 0) kept_vertices.insert(corners[corner]);
          if (corners[corner] >= 0 && corners[(corner + 1) % 3] >= 0) {
            kept_edges.insert(edge_key(corners[corner], corners[(corner + 1) % 3]));
          }
        }
      }
    }
    // Each disk face is seen from both of its cells.
    if (disk_sides.size() != 2 * facet_rows.size()) return TetEditStatus::kRejectedTopology;
    for (std::int32_t slot : sorted_cavity) {
      for (std::int32_t corner : tet(slot).v) {
        if (corner >= 0 && !kept_vertices.contains(corner)) return TetEditStatus::kRejectedTopology;
      }
    }
    for (std::uint64_t key : interior_edges) {
      if (kept_edges.contains(key)) return TetEditStatus::kRejectedTopology;
    }
    for (const auto& [key, edge_source] : polygon_sources) {
      if (!kept_edges.contains(key)) return TetEditStatus::kRejectedTopology;
    }
    if (split[0] >= 0 && kept_edges.contains(split_key)) return TetEditStatus::kRejectedTopology;
    // Only the split segment may leave the mesh.
    std::set<std::uint64_t> allowed_edges = kept_edges;
    if (split[0] >= 0) allowed_edges.insert(split_key);
    if (removes_constrained_segment(finite, allowed_edges)) return TetEditStatus::kRejectedProtected;
    std::sort(ancestors.begin(), ancestors.end());
    const std::uint64_t max_id = std::numeric_limits<std::uint64_t>::max();
    if (boundary.size() > max_id - next_cell_id_ || polygon.size() > max_id - next_facet_id_ ||
        next_segment_id_ > max_id - 2 || generation_ == max_id) {
      return TetEditStatus::kIndexOverflow;
    }

    const std::int32_t center = static_cast<std::int32_t>(vertex_ids_.size());
    const std::size_t vertex_count = vertex_ids_.size();
    const std::size_t cell_id_count = cell_ids_.size();
    const std::size_t facet_count = facets_.size();
    const std::size_t segment_count = segments_.size();
    std::size_t staged_cells = 0;
    std::size_t staged_facets = 0;
    std::size_t staged_segments = 0;
    bool vertex_staged = false;
    std::vector<std::tuple<std::int32_t, int, std::int32_t>> cleared_faces;
    const auto unstage = [&]() noexcept {
      for (const auto& [slot, face, constraint] : cleared_faces) {
        complex_.set_facet_constraint(slot, face, constraint);
      }
      cleared_faces.clear();
      for (std::size_t row = 0; row < staged_cells; ++row) {
        const std::uint64_t id = next_cell_id_ + row;
        used_cell_ids_.erase(id);
        cell_slots_.erase(id);
        cell_ancestry_.erase(id);
      }
      for (std::size_t row = 0; row < staged_facets; ++row) {
        used_facet_ids_.erase(next_facet_id_ + row);
      }
      for (std::size_t row = 0; row < staged_segments; ++row) {
        used_segment_ids_.erase(next_segment_id_ + row);
      }
      cell_ids_.resize(cell_id_count);
      facets_.resize(facet_count);
      segments_.resize(segment_count);
      if (vertex_staged) {
        vertex_slots_.erase(vertex.id);
        used_vertex_ids_.erase(vertex.id);
      }
      points_.resize(3 * vertex_count);
      vertex_ids_.resize(vertex_count);
      vertex_live_.resize(vertex_count);
      vertex_dimension_.resize(vertex_count);
      protection_radii_.resize(vertex_count);
      protected_vertices_.resize(vertex_count);
      witnesses_.resize(vertex_count);
      vertex_has_edge_parameter_.resize(vertex_count);
      vertex_edge_endpoints_.resize(vertex_count);
      vertex_edge_parameters_.resize(vertex_count);
    };
    const auto reject = [&](TetEditStatus status) {
      edit_.rollback();
      unstage();
      return status;
    };
    std::vector<TetSnapshotFacet> fan;
    fan.reserve(polygon.size());
    try {
      points_.insert(points_.end(), vertex.carrier.begin(), vertex.carrier.end());
      vertex_ids_.push_back(vertex.id);
      vertex_live_.push_back(1);
      vertex_dimension_.push_back(stratum == SourceStratum::kSegment ? 1 : 2);
      protection_radii_.push_back(0.0);
      protected_vertices_.push_back(0);
      witnesses_.push_back(vertex.witness);
      vertex_has_edge_parameter_.push_back(0);
      vertex_edge_endpoints_.push_back({});
      vertex_edge_parameters_.push_back(0.0);
      vertex_staged = true;
      vertex_slots_.emplace(vertex.id, center);
      used_vertex_ids_.insert(vertex.id);

      for (const DiskEdge& edge : polygon) {
        TetEditFacet record;
        record.vertices = {vertex_id(edge.from), vertex_id(edge.to), vertex.id};
        record.source = edge.source;
        record.orientation = edge.orientation;
        if (!source_face_valid(record)) return reject(TetEditStatus::kUncertainPredicate);
        TetSnapshotFacet facet;
        facet.vertices = record.vertices;
        facet.source = edge.source;
        facet.orientation = edge.orientation;
        fan.push_back(facet);
      }
      if (split[0] >= 0) {
        for (int half = 0; half < 2; ++half) {
          TetEditSegment record;
          record.vertices = {half == 0 ? vertex_id(split[0]) : vertex.id,
                             half == 0 ? vertex.id : vertex_id(split[1])};
          record.source = segment_source;
          if (!source_segment_valid(record)) return reject(TetEditStatus::kUncertainPredicate);
        }
      }

      edit_.begin();
      for (const std::vector<std::int32_t>* cells : {&finite, &ghosts}) {
        for (std::int32_t slot : *cells) {
          const EditStatus status = edit_.remove(slot);
          if (status != EditStatus::kOk) return reject(translate_edit_status(status));
        }
      }
      // Finite cone cells first: created()[row] for row < finite_cone
      // receives the cell ID next_cell_id_ + row.
      std::size_t finite_cone = 0;
      for (const bool ghost_pass : {false, true}) {
        for (const BoundaryFacet& face : boundary) {
          Tetrahedron cone = tet(face.tet);
          cone.v[face.slot] = center;
          if (is_ghost(cone) != ghost_pass) continue;
          std::int32_t constraints[4];
          for (int corner = 0; corner < 4; ++corner) {
            if (corner == face.slot) {
              constraints[corner] = complex_.constraint(face.tet, face.slot);
              continue;
            }
            std::int32_t pair[2];
            int count = 0;
            for (int other = 0; other < 4; ++other) {
              if (other != corner && other != face.slot) pair[count++] = cone.v[other];
            }
            const auto polygon_edge = pair[0] >= 0 && pair[1] >= 0
                                          ? polygon_sources.find(edge_key(pair[0], pair[1]))
                                          : polygon_sources.end();
            constraints[corner] =
                polygon_edge == polygon_sources.end() ? kNoConstraint : polygon_edge->second;
          }
          const EditStatus status = edit_.add(cone.v, constraints, complex_.region(face.tet));
          if (status != EditStatus::kOk) return reject(translate_edit_status(status));
          if (!ghost_pass) ++finite_cone;
        }
      }
      // Both sides of each disk face are listed and clearing one side clears
      // the other, so every constraint is captured before any is cleared.
      cleared_faces.reserve(disk_sides.size());
      for (const auto& [slot, face] : disk_sides) {
        cleared_faces.emplace_back(slot, face, complex_.constraint(slot, face));
      }
      for (const auto& [slot, face] : disk_sides) {
        complex_.set_facet_constraint(slot, face, kNoConstraint);
      }
      const EditStatus validated =
          edit_.validate([this](const std::int32_t* vertices) { return positive(vertices); });
      if (validated != EditStatus::kOk) return reject(translate_edit_status(validated));

      const std::size_t fresh =
          boundary.size() - std::min(boundary.size(), complex_.free_slots.size() + cavity.size());
      cell_ids_.resize(complex_.tets.size() + fresh, max_id);
      facets_.reserve(facet_count + fan.size() - facet_rows.size());
      segments_.reserve(segment_count + segment_rows.size());
      while (staged_facets < fan.size()) used_facet_ids_.insert(next_facet_id_ + staged_facets++);
      while (staged_segments < 2 * segment_rows.size()) {
        used_segment_ids_.insert(next_segment_id_ + staged_segments++);
      }
      created_cells.reserve(finite_cone);
      while (staged_cells < finite_cone) {
        const std::uint64_t id = next_cell_id_ + staged_cells++;
        if (used_cell_ids_.contains(id)) {
          --staged_cells;
          return reject(TetEditStatus::kRejectedTopology);
        }
        used_cell_ids_.insert(id);
        cell_slots_.emplace(id, -1);
        cell_ancestry_.emplace(
            id, TetCellAncestryRecord{static_cast<std::int32_t>(TetEditKind::kRefineCell),
                                      ancestors});
      }
      const EditStatus committed = edit_.commit();
      if (committed != EditStatus::kOk) return reject(translate_edit_status(committed));
    } catch (...) {
      edit_.rollback();
      unstage();
      throw;
    }

    for (std::int32_t slot : finite) cell_slots_.erase(cell_id(slot));
    for (std::int32_t slot : sorted_cavity) cell_ids_[static_cast<std::size_t>(slot)] = max_id;
    const std::span<const std::int32_t> created_slots = edit_.created();
    for (std::size_t row = 0; row < created_slots.size(); ++row) {
      const std::int32_t slot = created_slots[row];
      if (row >= staged_cells) {
        cell_ids_[static_cast<std::size_t>(slot)] = max_id;
        continue;
      }
      const std::uint64_t id = next_cell_id_ + row;
      cell_ids_[static_cast<std::size_t>(slot)] = id;
      cell_slots_.find(id)->second = slot;
      created_cells.push_back(id);
    }
    next_cell_id_ += staged_cells;
    for (std::size_t index = 0; index < fan.size(); ++index) {
      fan[index].id = next_facet_id_ + index;
      if (index < facet_rows.size()) {
        facets_[facet_rows[index]] = fan[index];
      } else {
        facets_.push_back(fan[index]);
      }
    }
    next_facet_id_ += fan.size();
    if (split[0] >= 0) {
      TetSnapshotSegment& first_half = segments_[segment_rows.front()];
      TetSnapshotSegment second_half = first_half;
      first_half.id = next_segment_id_;
      second_half.id = next_segment_id_ + 1;
      first_half.vertices = {vertex_id(split[0]), vertex.id};
      second_half.vertices = {vertex.id, vertex_id(split[1])};
      segments_.push_back(second_half);
      next_segment_id_ += 2;
    }
    if (vertex.id >= next_vertex_id_) next_vertex_id_ = vertex.id + 1;
    ++generation_;
    unmet_records_.clear();
    return TetEditStatus::kApplied;
  } catch (const std::bad_alloc&) {
    return TetEditStatus::kAllocationFailure;
  } catch (const std::length_error&) {
    return TetEditStatus::kIndexOverflow;
  }
}

// Visibility walk from `start` to `point` without crossing a constrained or
// hull face, Bowyer-Watson growth across unconstrained faces by exact
// insphere (seeded with the walked cells of the root's region when
// `keep_walk`, so the cavity can replace the start cell even when its
// circumsphere excludes `point`), then shrinking (dropping a cell and keeping
// the root's component) until every cone cell is exactly positive and
// measurable by the quality criteria (finite circumsphere, positive relative
// determinant, finite dihedral angles) and every cavity vertex and
// constrained segment stays on the cavity boundary. The root contains
// `point`, so its cone is positive unless `point` is too close to its faces,
// in which case the cavity is refused.
TetCavityStatus TetMesh::insertion_cavity(std::int32_t start, const double* point,
                                          bool keep_walk,
                                          TetInsertionCavity& cavity) const {
  cavity.cells.clear();
  cavity.members.clear();
  cavity.root = -1;
  cavity.blocked_cell = -1;
  cavity.blocked_face = -1;
  std::int32_t current = start;
  std::vector<std::int32_t> walk;
  std::unordered_set<std::int32_t>& visited = cavity.members;
  for (;;) {
    if (!visited.insert(current).second) return TetCavityStatus::kRefused;
    const Tetrahedron& cell = tet(current);
    int exit_face = -1;
    for (int face = 0; face < 4 && exit_face < 0; ++face) {
      std::int32_t facet[3];
      oriented_facet(cell.v, face, facet);
      if (orient3d(this->point(facet[0]), this->point(facet[1]), this->point(facet[2]),
                   point) < 0) {
        exit_face = face;
      }
    }
    if (exit_face < 0) break;
    const std::int32_t next = cell.n[exit_face];
    if (complex_.constraint(current, exit_face) != kNoConstraint || !complex_.live(next) ||
        is_ghost(tet(next))) {
      cavity.blocked_cell = current;
      cavity.blocked_face = exit_face;
      visited.clear();
      return TetCavityStatus::kBlocked;
    }
    walk.push_back(current);
    current = next;
  }

  const std::int32_t root = current;
  std::vector<std::int32_t>& cells = cavity.cells;
  std::unordered_set<std::int32_t>& members = cavity.members;
  cells.assign(1, root);
  members.clear();
  members.insert(root);
  std::vector<std::int32_t> stack(1, root);
  if (keep_walk) {
    for (std::int32_t cell : walk) {
      if (complex_.region(cell) != complex_.region(root)) continue;
      members.insert(cell);
      cells.push_back(cell);
      stack.push_back(cell);
    }
  }
  while (!stack.empty()) {
    const std::int32_t cell = stack.back();
    stack.pop_back();
    for (int face = 0; face < 4; ++face) {
      const std::int32_t next = tet(cell).n[face];
      if (complex_.constraint(cell, face) != kNoConstraint || !complex_.live(next) ||
          is_ghost(tet(next)) || members.contains(next) ||
          complex_.region(next) != complex_.region(root)) {
        continue;
      }
      const Tetrahedron& candidate = tet(next);
      if (insphere(this->point(candidate.v[0]), this->point(candidate.v[1]),
                   this->point(candidate.v[2]), this->point(candidate.v[3]), point) > 0) {
        members.insert(next);
        cells.push_back(next);
        stack.push_back(next);
      }
    }
  }

  for (;;) {
    std::int32_t drop = -1;
    std::set<std::uint64_t> kept_edges;
    std::set<std::int32_t> boundary_vertices;
    for (std::int32_t cell : cells) {
      const Tetrahedron& tetrahedron = tet(cell);
      for (int face = 0; face < 4 && drop < 0; ++face) {
        if (members.contains(tetrahedron.n[face])) continue;
        std::array<const double*, 4> cone{};
        for (int corner = 0; corner < 4; ++corner) {
          cone[static_cast<std::size_t>(corner)] =
              corner == face ? point : this->point(tetrahedron.v[corner]);
        }
        geometry::TetrahedronShape cone_shape;
        if (orient3d(cone[0], cone[1], cone[2], cone[3]) <= 0 ||
            !geometry::tetrahedron_shape(cone.data(), cone_shape) ||
            !(geometry::relative_determinant(cone.data()) > 0.0) ||
            !std::isfinite(geometry::minimum_dihedral_degrees(cone.data()))) {
          drop = cell;
          break;
        }
        std::int32_t facet[3];
        oriented_facet(tetrahedron.v, face, facet);
        for (int corner = 0; corner < 3; ++corner) {
          boundary_vertices.insert(facet[corner]);
          kept_edges.insert(edge_key(facet[corner], facet[(corner + 1) % 3]));
        }
      }
      if (drop >= 0) break;
    }
    if (drop < 0) {
      for (std::int32_t cell : cells) {
        if (cell == root) continue;
        const bool loses_vertex = std::any_of(
            tet(cell).v, tet(cell).v + 4,
            [&](std::int32_t vertex) { return !boundary_vertices.contains(vertex); });
        const std::int32_t single[1]{cell};
        if (loses_vertex || removes_constrained_segment(single, kept_edges)) {
          drop = cell;
          break;
        }
      }
    }
    if (drop < 0) break;
    if (drop == root) {
      cells.clear();
      members.clear();
      return TetCavityStatus::kRefused;
    }
    members.erase(drop);
    std::unordered_set<std::int32_t> reached{root};
    stack.assign(1, root);
    std::vector<std::int32_t> connected{root};
    while (!stack.empty()) {
      const std::int32_t cell = stack.back();
      stack.pop_back();
      for (std::int32_t next : tet(cell).n) {
        if (members.contains(next) && reached.insert(next).second) {
          connected.push_back(next);
          stack.push_back(next);
        }
      }
    }
    cells.swap(connected);
    members.swap(reached);
  }
  cavity.root = root;
  return TetCavityStatus::kFound;
}

// The cavity of a constrained boundary point: the cells on both sides of the
// disk faces (`disk_sides`, each a finite cell and its face slot; the other
// side is the neighbor across it, a hull ghost when the facet is on the hull),
// grown by Bowyer-Watson across unconstrained faces into finite cells whose
// circumspheres strictly contain the point's carrier, then made star-shaped:
// while a finite cone cell over the cavity boundary is not positive (the
// exact predicate of the commit, on the exact source positions) and
// measurable on carriers, a grown cell is dropped (keeping the component of
// the disk cells), or for a disk cell the cavity grows across that face into
// the finite cell beyond an unconstrained face, which is kept from then on;
// a cell whose drop keeps a vertex or constrained segment off the boundary
// is dropped likewise. Refused when a disk cell would have to go.
TetCavityStatus TetMesh::boundary_insertion_cavity(
    const TetEditVertex& vertex, std::span<const std::pair<std::int32_t, int>> disk_sides,
    std::array<std::int32_t, 2> split_segment, std::vector<std::int32_t>& cells) const {
  const double* point = vertex.carrier.data();
  cells.clear();
  std::unordered_set<std::int32_t> members;
  std::unordered_set<std::int32_t> disk_cells;
  for (const auto& [cell, face] : disk_sides) {
    if (!complex_.live(cell) || is_ghost(tet(cell)) || face < 0 || face > 3 ||
        complex_.constraint(cell, face) == kNoConstraint) {
      return TetCavityStatus::kRefused;
    }
    const std::int32_t other = tet(cell).n[face];
    if (!complex_.live(other)) return TetCavityStatus::kRefused;
    disk_cells.insert(cell);
    disk_cells.insert(other);
  }
  std::vector<std::int32_t> stack;
  for (std::int32_t cell : disk_cells) {
    members.insert(cell);
    if (!is_ghost(tet(cell))) stack.push_back(cell);
  }
  while (!stack.empty()) {
    const std::int32_t cell = stack.back();
    stack.pop_back();
    for (int face = 0; face < 4; ++face) {
      const std::int32_t next = tet(cell).n[face];
      if (complex_.constraint(cell, face) != kNoConstraint || !complex_.live(next) ||
          is_ghost(tet(next)) || members.contains(next)) {
        continue;
      }
      const Tetrahedron& candidate = tet(next);
      if (insphere(this->point(candidate.v[0]), this->point(candidate.v[1]),
                   this->point(candidate.v[2]), this->point(candidate.v[3]), point) > 0) {
        members.insert(next);
        stack.push_back(next);
      }
    }
  }

  for (;;) {
    std::int32_t drop = -1;
    int drop_face = -1;
    // The split segment leaves the mesh with the cavity.
    std::set<std::uint64_t> kept_edges;
    if (split_segment[0] >= 0) kept_edges.insert(edge_key(split_segment[0], split_segment[1]));
    std::set<std::int32_t> boundary_vertices;
    for (std::int32_t cell : members) {
      const Tetrahedron& tetrahedron = tet(cell);
      for (int face = 0; face < 4 && drop < 0; ++face) {
        if (members.contains(tetrahedron.n[face])) continue;
        std::int32_t facet[3];
        oriented_facet(tetrahedron.v, face, facet);
        for (int corner = 0; corner < 3; ++corner) {
          if (facet[corner] < 0) continue;
          boundary_vertices.insert(facet[corner]);
          if (facet[(corner + 1) % 3] >= 0) {
            kept_edges.insert(edge_key(facet[corner], facet[(corner + 1) % 3]));
          }
        }
        if (is_ghost(tetrahedron)) continue;
        std::array<const double*, 4> cone{};
        for (int corner = 0; corner < 4; ++corner) {
          cone[static_cast<std::size_t>(corner)] =
              corner == face ? point : this->point(tetrahedron.v[corner]);
        }
        geometry::TetrahedronShape cone_shape;
        if (!cone_positive(tetrahedron.v, face, vertex) ||
            !geometry::tetrahedron_shape(cone.data(), cone_shape) ||
            !(geometry::relative_determinant(cone.data()) > 0.0) ||
            !std::isfinite(geometry::minimum_dihedral_degrees(cone.data()))) {
          drop = cell;
          drop_face = face;
        }
      }
      if (drop >= 0) break;
    }
    if (drop < 0) {
      std::vector<std::int32_t> finite;
      for (std::int32_t cell : members) {
        if (is_ghost(tet(cell))) continue;
        finite.push_back(cell);
        const bool loses_vertex =
            std::any_of(tet(cell).v, tet(cell).v + 4, [&](std::int32_t vertex) {
              return vertex >= 0 && !boundary_vertices.contains(vertex);
            });
        const std::int32_t single[1]{cell};
        if (drop < 0 && (loses_vertex || removes_constrained_segment(single, kept_edges))) {
          drop = cell;
        }
      }
    }
    if (drop < 0) break;
    if (disk_cells.contains(drop)) {
      // A disk cell stays: the cavity grows across its failing face instead
      // (the grown cell stays too, so the cavity only grows and this ends).
      const std::int32_t next = drop_face >= 0 ? tet(drop).n[drop_face] : -1;
      if (next < 0 || !complex_.live(next) || is_ghost(tet(next)) ||
          complex_.constraint(drop, drop_face) != kNoConstraint) {
        cells.clear();
        return TetCavityStatus::kRefused;
      }
      members.insert(next);
      disk_cells.insert(next);
      continue;
    }
    members.erase(drop);
    std::unordered_set<std::int32_t> reached(disk_cells.begin(), disk_cells.end());
    stack.assign(disk_cells.begin(), disk_cells.end());
    while (!stack.empty()) {
      const std::int32_t cell = stack.back();
      stack.pop_back();
      for (std::int32_t next : tet(cell).n) {
        if (members.contains(next) && reached.insert(next).second) stack.push_back(next);
      }
    }
    members.swap(reached);
  }
  cells.assign(members.begin(), members.end());
  std::sort(cells.begin(), cells.end());
  return TetCavityStatus::kFound;
}

bool TetMesh::validate_regions_and_constraints(
    const TetEdit& operation, std::span<const std::int32_t> removed_slots,
    std::span<const TetSnapshotFacet> candidate_facets) const {
  std::unordered_set<std::int32_t> removed(removed_slots.begin(), removed_slots.end());
  const auto replacement_face = [&](const SortedFace& key, std::int32_t expected_region,
                                    bool check_region) {
    bool found = false;
    for (const TetEditCell& cell : operation.replacement_cells) {
      std::int32_t local[4];
      for (int corner = 0; corner < 4; ++corner) {
        local[corner] = vertex_slot(cell.vertices[corner]);
        if (local[corner] < 0) return false;
      }
      for (int slot = 0; slot < 4; ++slot) {
        std::int32_t face[3];
        oriented_facet(local, slot, face);
        if (sorted_face(face) != key) continue;
        if (check_region && cell.region != expected_region) return false;
        found = true;
      }
    }
    return found;
  };
  const auto replacement_oriented_face =
      [&](const std::array<std::int32_t, 3>& expected,
          std::int32_t expected_region) {
        bool found = false;
        for (const TetEditCell& cell : operation.replacement_cells) {
          if (cell.region != expected_region) continue;
          std::int32_t local[4];
          for (int corner = 0; corner < 4; ++corner) {
            local[corner] = vertex_slot(cell.vertices[corner]);
            if (!live_vertex(local[corner])) return false;
          }
          for (int slot = 0; slot < 4; ++slot) {
            std::int32_t actual[3];
            oriented_facet(local, slot, actual);
            if (sorted_face(actual) != sorted_face(expected.data())) continue;
            if (!(FacetKey::of(actual) == FacetKey::of(expected.data()))) {
              return false;
            }
            found = true;
          }
        }
        return found;
      };
  const auto replacement_boundary_face =
      [&](const Tetrahedron& parent, int face_slot, std::int32_t region) {
        std::int32_t face[3];
        oriented_facet(parent.v, face_slot, face);
        std::size_t split_index = operation.split_edges.size();
        std::int32_t edge_first = -1;
        std::int32_t edge_second = -1;
        for (std::size_t edge = 0; edge < operation.split_edges.size(); ++edge) {
          const std::int32_t first =
              vertex_slot(operation.split_edges[edge][0]);
          const std::int32_t second =
              vertex_slot(operation.split_edges[edge][1]);
          bool has_first = false;
          bool has_second = false;
          for (int corner = 0; corner < 3; ++corner) {
            has_first = has_first || face[corner] == first;
            has_second = has_second || face[corner] == second;
          }
          if (!has_first || !has_second) continue;
          if (split_index != operation.split_edges.size()) return false;
          split_index = edge;
          edge_first = first;
          edge_second = second;
        }
        if (split_index == operation.split_edges.size()) {
          return replacement_face(sorted_face(face), region, true);
        }
        if (split_index >= operation.changed_vertices.size()) return false;
        const std::int32_t inserted =
            vertex_slot(operation.changed_vertices[split_index].id);
        if (!live_vertex(inserted)) return false;
        std::array<std::int32_t, 3> first_child{};
        std::array<std::int32_t, 3> second_child{};
        bool directed_edge_found = false;
        for (int offset = 0; offset < 3; ++offset) {
          const std::int32_t first = face[offset];
          const std::int32_t second = face[(offset + 1) % 3];
          if (!((first == edge_first && second == edge_second) ||
                (first == edge_second && second == edge_first))) {
            continue;
          }
          const std::int32_t third = face[(offset + 2) % 3];
          first_child = {first, inserted, third};
          second_child = {inserted, second, third};
          directed_edge_found = true;
          break;
        }
        if (!directed_edge_found ||
            !exact_split_coverage(*this, face, first_child, second_child)) {
          return false;
        }
        return replacement_oriented_face(first_child, region) &&
               replacement_oriented_face(second_child, region);
      };


  for (std::int32_t slot : removed_slots) {
    if (!complex_.live(slot) || is_ghost(tet(slot))) return false;
    const Tetrahedron& cell = tet(slot);
    for (int face_slot = 0; face_slot < 4; ++face_slot) {
      const std::int32_t neighbor = cell.n[face_slot];
      if (removed.contains(neighbor)) continue;
      if (!replacement_boundary_face(cell, face_slot, complex_.region(slot))) {
        return false;
      }
    }
  }

  std::map<std::pair<std::int32_t, int>, std::int32_t> source_regions;
  // Retain source-side consistency checks even when this edit replaces no facets.
  if (!collect_source_regions(*this, facets_, source_regions)) return false;
  // Each constrained replacement side must retain its authored region label,
  // including both sides of interfaces not listed as replacement facets.
  std::map<SortedFace, const TetSnapshotFacet*> facet_by_face;
  for (const TetSnapshotFacet& facet : candidate_facets) {
    std::int32_t local[3];
    for (int corner = 0; corner < 3; ++corner) {
      local[corner] = vertex_slot(facet.vertices[corner]);
      if (!live_vertex(local[corner])) return false;
    }
    if (!facet_by_face.emplace(sorted_face(local), &facet).second) return false;
  }
  for (const TetEditCell& cell : operation.replacement_cells) {
    std::int32_t local[4];
    for (int corner = 0; corner < 4; ++corner) {
      local[corner] = vertex_slot(cell.vertices[corner]);
      if (!live_vertex(local[corner])) return false;
    }
    for (int face_slot = 0; face_slot < 4; ++face_slot) {
      const std::int32_t source = cell.facet_sources[face_slot];
      if (source < 0) continue;
      std::int32_t face[3];
      oriented_facet(local, face_slot, face);
      const auto record = facet_by_face.find(sorted_face(face));
      if (record == facet_by_face.end() || record->second->source != source) return false;
      std::int32_t facet_vertices[3];
      for (int corner = 0; corner < 3; ++corner) {
        facet_vertices[corner] = vertex_slot(record->second->vertices[corner]);
        if (!live_vertex(facet_vertices[corner])) return false;
      }
      const int direction =
          (FacetKey::of(face) == FacetKey::of(facet_vertices) ? 1 : -1) *
          record->second->orientation;
      const auto expected = source_regions.find({source, direction});
      if (expected == source_regions.end() || expected->second != cell.region) return false;
    }
  }

  for (const TetEditFacet& facet : operation.replacement_facets) {
    std::int32_t local_face[3];
    for (int corner = 0; corner < 3; ++corner) {
      local_face[corner] = vertex_slot(facet.vertices[corner]);
      if (local_face[corner] < 0) return false;
    }
    bool adjacent = false;
    for (const TetEditCell& cell : operation.replacement_cells) {
      std::int32_t local[4];
      for (int corner = 0; corner < 4; ++corner) {
        local[corner] = vertex_slot(cell.vertices[corner]);
        if (local[corner] < 0) return false;
      }
      for (int slot = 0; slot < 4; ++slot) {
        std::int32_t face[3];
        oriented_facet(local, slot, face);
        if (sorted_face(face) != sorted_face(local_face)) continue;
        adjacent = true;
        const int side = (FacetKey::of(face) == FacetKey::of(local_face) ? 1 : -1) *
                         facet.orientation;
        const auto found = source_regions.find({facet.source, side});
        if (found == source_regions.end() || found->second != cell.region) return false;
      }
    }
    if (!adjacent) return false;
  }
  return true;
}

// Keep the staged edit's candidate banks in one scope; they share validation and
// commit dependencies. Public rollback remains owned by apply_edit_batch's swap,
// rather than by a second mutable transaction representation.
TetEditStatus TetMesh::validate_and_commit(TetMesh& staged,
                                           const TetEdit& operation,
                                           TetEditResult& result) {
  TetMesh& mesh = staged;
  const bool relocation = operation.kind == TetEditKind::kRelocate;
  const bool refinement = operation.kind == TetEditKind::kRefineCell;
  const bool collapse = operation.kind == TetEditKind::kCollapse;
  const bool retriangulation = operation.kind == TetEditKind::kRetriangulate;
  const bool split = operation.kind == TetEditKind::kSplit;
  const bool volume_preserving =
      operation.kind == TetEditKind::kFlip23 ||
      operation.kind == TetEditKind::kFlip32 ||
      operation.kind == TetEditKind::kRingReconnect ||
      operation.kind == TetEditKind::kRetriangulate;
  if (relocation && operation.changed_vertices.empty()) {
    return TetEditStatus::kRejectedTopology;
  }
  if (operation.replacement_cells.empty() && !relocation) {
    return TetEditStatus::kRejectedTopology;
  }
  if (relocation && (!operation.replacement_cells.empty() ||
                     !operation.replacement_facets.empty() ||
                     !operation.replacement_segments.empty() ||
                     !operation.removed_vertex_ids.empty() ||
                     !operation.removed_facet_ids.empty() ||
                     !operation.removed_segment_ids.empty())) {
    return TetEditStatus::kRejectedTopology;
  }
  if (collapse && (operation.removed_vertex_ids.empty() ||
                   operation.collapse_pairs.size() != operation.removed_vertex_ids.size())) {
    return TetEditStatus::kRejectedTopology;
  }
  if (!collapse && !retriangulation &&
      (!operation.removed_vertex_ids.empty() || !operation.collapse_pairs.empty())) {
    return TetEditStatus::kRejectedTopology;
  }
  if (retriangulation &&
      (operation.changed_vertices.empty() || mesh.periodic_identity_.bound ||
       !operation.collapse_pairs.empty())) {
    return TetEditStatus::kRejectedTopology;
  }
  if (split && operation.changed_vertices.empty()) return TetEditStatus::kRejectedTopology;
  if (!split && !operation.split_edges.empty()) return TetEditStatus::kRejectedTopology;

  std::unordered_set<std::uint64_t> removed_cell_ids;
  std::vector<std::int32_t> removed_slots;
  removed_slots.reserve(operation.removed_cell_ids.size());
  for (std::uint64_t id : operation.removed_cell_ids) {
    const std::int32_t slot = mesh.cell_slot(id);
    if (slot < 0 || is_ghost(mesh.tet(slot)) || !removed_cell_ids.insert(id).second) {
      return TetEditStatus::kRejectedTopology;
    }
    removed_slots.push_back(slot);
  }
  if (removed_slots.empty() && !relocation) return TetEditStatus::kRejectedTopology;
  std::unordered_set<std::int32_t> removed_slot_set(removed_slots.begin(), removed_slots.end());
  Dyadic removed_volume;
  if (volume_preserving) {
    for (std::int32_t slot : removed_slots) {
      removed_volume = removed_volume +
                       exact_oriented_determinant(mesh, mesh.tet(slot).v);
    }
  }

  std::unordered_set<std::uint64_t> removed_vertex_ids;
  std::vector<std::int32_t> removed_vertex_slots;
  for (std::uint64_t id : operation.removed_vertex_ids) {
    const std::int32_t slot = mesh.vertex_slot(id);
    if (!mesh.live_vertex(slot)) return TetEditStatus::kRejectedTopology;
    if (mesh.protected_vertex(slot) || mesh.protection_radius(slot) > 0.0) {
      return TetEditStatus::kRejectedProtected;
    }
    if (!removed_vertex_ids.insert(id).second) return TetEditStatus::kRejectedTopology;
    removed_vertex_slots.push_back(slot);
    for (std::size_t t = 0; t < mesh.complex_.tets.size(); ++t) {
      const std::int32_t cell = static_cast<std::int32_t>(t);
      if (!mesh.complex_.live(cell) || is_ghost(mesh.tet(cell)) ||
          phx::mc::vertex_slot(mesh.tet(cell), slot) < 0) continue;
      if (!removed_slot_set.contains(cell)) return TetEditStatus::kRejectedTopology;
    }
  }
  if (retriangulation) {
    for (const TetEditVertex& vertex : operation.changed_vertices) {
      for (const TetVertexAncestry& ancestor : vertex.ancestry) {
        if (removed_vertex_ids.contains(ancestor.source_vertex_id)) {
          return TetEditStatus::kRejectedTopology;
        }
      }
    }
  }
  if (collapse) {
    std::unordered_set<std::uint64_t> paired_removed;
    std::unordered_set<std::uint64_t> paired_survivors;
    for (const TetCollapsePair& pair : operation.collapse_pairs) {
      if (!removed_vertex_ids.contains(pair.removed_vertex_id) ||
          removed_vertex_ids.contains(pair.survivor_vertex_id) ||
          !paired_removed.insert(pair.removed_vertex_id).second ||
          !paired_survivors.insert(pair.survivor_vertex_id).second) {
        return TetEditStatus::kRejectedTopology;
      }
      const std::int32_t remove = mesh.vertex_slot(pair.removed_vertex_id);
      const std::int32_t keep = mesh.vertex_slot(pair.survivor_vertex_id);
      if (!mesh.live_vertex(keep) || remove == keep) return TetEditStatus::kRejectedTopology;
      if (mesh.protected_vertex(keep) || mesh.protection_radius(keep) > 0.0) {
        return TetEditStatus::kRejectedProtected;
      }
      if (!full_link_condition(mesh, remove, keep)) return TetEditStatus::kRejectedTopology;
    }
  }


  // Relocations of lifted periodic orbits revalidate cover facets modulo the
  // lattice action: a seam facet stays valid when its deck partner does.
  const PeriodicMemberIndex seam_members =
      relocation ? mesh.periodic_member_index() : PeriodicMemberIndex{};
  // A boundary vertex without a source witness may move only as a whole orbit
  // member that lies on no constrained segment and only on periodic seams; in
  // the quotient it is interior.
  const auto periodic_seam_vertex = [&](std::uint64_t id) {
    if (!seam_members.contains(id)) return false;
    for (const TetSnapshotSegment& segment : mesh.segments_) {
      if (segment.vertices[0] == id || segment.vertices[1] == id) return false;
    }
    bool incident = false;
    for (const TetSnapshotFacet& facet : mesh.facets_) {
      if (std::find(facet.vertices.begin(), facet.vertices.end(), id) == facet.vertices.end()) {
        continue;
      }
      if (!mesh.periodic_seam_facet(facet, seam_members)) return false;
      incident = true;
    }
    return incident;
  };
  std::unordered_set<std::uint64_t> changed_vertex_ids;
  std::vector<std::int32_t> moved_vertex_slots;
  std::size_t new_count = 0;
  for (const TetEditVertex& vertex : operation.changed_vertices) {
    if (vertex.id == std::numeric_limits<std::uint64_t>::max()) {
      return TetEditStatus::kIndexOverflow;
    }
    if (!changed_vertex_ids.insert(vertex.id).second ||
        (mesh.used_vertex_ids_.contains(vertex.id) && mesh.vertex_slot(vertex.id) < 0)) {
      return TetEditStatus::kRejectedTopology;
    }
    const std::int32_t current_slot = mesh.vertex_slot(vertex.id);
    if (current_slot >= 0) {
      if (!vertex.ancestry.empty()) return TetEditStatus::kRejectedTopology;
      if (relocation == false && !split) return TetEditStatus::kRejectedTopology;
      if (mesh.protected_vertex(current_slot) || mesh.protection_radius(current_slot) > 0.0) {
        return TetEditStatus::kRejectedProtected;
      }
      if (mesh.vertex_dimension(current_slot) < 3 &&
          mesh.witness(current_slot).stratum == SourceStratum::kNone &&
          !periodic_seam_vertex(vertex.id)) {
        return TetEditStatus::kUncertainPredicate;
      }
      if (vertex.protected_vertex != mesh.protected_vertex(current_slot) ||
          vertex.protection_radius != mesh.protection_radius(current_slot)) {
        return TetEditStatus::kRejectedProtected;
      }
      if (vertex.witness.stratum != mesh.witness(current_slot).stratum) {
        return TetEditStatus::kRejectedProtected;
      }
      moved_vertex_slots.push_back(current_slot);
    } else {
      if (relocation || !mesh.used_vertex_ids_.insert(vertex.id).second) {
        return TetEditStatus::kRejectedTopology;
      }
      if (mesh.vertex_ids_.size() + new_count >=
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return TetEditStatus::kIndexOverflow;
      }
      ++new_count;
    }
    TetEditVertex checked_vertex = vertex;
    if (current_slot >= 0 && relocation && !checked_vertex.edge_witness_supplied &&
        !checked_vertex.has_edge_parameter &&
        mesh.vertex_has_edge_parameter_[static_cast<std::size_t>(current_slot)] != 0) {
      checked_vertex.has_edge_parameter = true;
      checked_vertex.edge_endpoints =
          mesh.vertex_edge_endpoints_[static_cast<std::size_t>(current_slot)];
      checked_vertex.edge_parameter =
          mesh.vertex_edge_parameters_[static_cast<std::size_t>(current_slot)];
    }
    if (!std::isfinite(vertex.protection_radius) || vertex.protection_radius < 0.0 ||
        !mesh.source_carrier_valid(checked_vertex)) {
      return TetEditStatus::kInvalidGeometry;
    }
    if (vertex.protection_radius > 0.0 && !vertex.protected_vertex) {
      return TetEditStatus::kRejectedProtected;
    }
    for (std::size_t other = 0; other < mesh.vertex_ids_.size(); ++other) {
      if (!mesh.vertex_live_[other] || mesh.vertex_ids_[other] == vertex.id) continue;
      if (same_point(mesh.points_.data() + 3 * other, vertex.carrier.data())) {
        return TetEditStatus::kInvalidGeometry;
      }
    }
  }
  if (split && (operation.split_edges.size() != new_count ||
                operation.changed_vertices.size() != new_count)) {
    return TetEditStatus::kRejectedTopology;
  }
  if ((refinement || retriangulation) &&
      (operation.changed_vertices.size() != new_count ||
       !operation.split_edges.empty())) {
    return TetEditStatus::kRejectedTopology;
  }
  if (!split && !refinement && !retriangulation && new_count != 0) {
    return TetEditStatus::kRejectedTopology;
  }

  if (split) {
    for (std::size_t index = 0; index < operation.split_edges.size(); ++index) {
      const std::int32_t a = mesh.vertex_slot(operation.split_edges[index][0]);
      const std::int32_t b = mesh.vertex_slot(operation.split_edges[index][1]);
      if (!mesh.live_vertex(a) || !mesh.live_vertex(b) || a == b) {
        return TetEditStatus::kRejectedTopology;
      }
      bool edge_exists = false;
      bool constrained_segment = false;
      std::int32_t segment_source = kNoConstraint;
      for (std::size_t t = 0; t < mesh.complex_.tets.size(); ++t) {
        const std::int32_t cell = static_cast<std::int32_t>(t);
        if (!mesh.complex_.live(cell) || is_ghost(mesh.tet(cell))) continue;
        const Tetrahedron& tet = mesh.tet(cell);
        if (phx::mc::vertex_slot(tet, a) >= 0 &&
            phx::mc::vertex_slot(tet, b) >= 0) edge_exists = true;
      }
      for (const TetSnapshotSegment& segment : mesh.segments_) {
        const std::int32_t segment_a = mesh.vertex_slot(segment.vertices[0]);
        const std::int32_t segment_b = mesh.vertex_slot(segment.vertices[1]);
        if (segment_a < 0 || segment_b < 0) return TetEditStatus::kRejectedTopology;
        if (sorted_edge(segment_a, segment_b) == sorted_edge(a, b)) {
          constrained_segment = true;
          segment_source = segment.source;
          break;
        }
      }
      if (!edge_exists) return TetEditStatus::kRejectedTopology;
      const TetEditVertex& inserted = operation.changed_vertices[index];
      if (inserted.has_edge_parameter &&
          inserted.edge_endpoints != operation.split_edges[index]) {
        return TetEditStatus::kInvalidGeometry;
      }
      if (constrained_segment) {
        const SourceWitness& witness = inserted.witness;
        if (mesh.policy_ == BoundaryPolicy::kFixed) return TetEditStatus::kRejectedProtected;
        if (witness.stratum != SourceStratum::kSegment || witness.entity < 0 ||
            static_cast<std::size_t>(witness.entity) >= mesh.source_segment_groups_.size() ||
            mesh.source_segment_groups_[static_cast<std::size_t>(witness.entity)] != segment_source) {
          return TetEditStatus::kInvalidGeometry;
        }
        const SourceWitness& wa = mesh.witness(a);
        const SourceWitness& wb = mesh.witness(b);
        if (wa.stratum == SourceStratum::kSegment && wb.stratum == SourceStratum::kSegment &&
            wa.entity == witness.entity && wb.entity == witness.entity) {
          const double low = std::min(wa.parameters[0], wb.parameters[0]);
          const double high = std::max(wa.parameters[0], wb.parameters[0]);
          if (!(low < witness.parameters[0] && witness.parameters[0] < high)) {
            return TetEditStatus::kInvalidGeometry;
          }
        }
      } else if (inserted.witness.stratum == SourceStratum::kNone &&
                 !inserted.has_edge_parameter) {
        const double* endpoints[3] = {mesh.point(a), mesh.point(b), nullptr};
        if (!on_source_entity(endpoints, SourceStratum::kSegment, inserted.carrier.data())) {
          return TetEditStatus::kInvalidGeometry;
        }
      }
      int dominant = 0;
      for (int axis = 1; axis < 3; ++axis) {
        if (std::abs(mesh.point(b)[axis] - mesh.point(a)[axis]) >
            std::abs(mesh.point(b)[dominant] - mesh.point(a)[dominant])) dominant = axis;
      }
      const double lo = std::min(mesh.point(a)[dominant], mesh.point(b)[dominant]);
      const double hi = std::max(mesh.point(a)[dominant], mesh.point(b)[dominant]);
      if (!(lo < inserted.carrier[dominant] && inserted.carrier[dominant] < hi)) {
        return TetEditStatus::kInvalidGeometry;
      }
    }
  }
  if (!relocation) {
    for (const TetEditVertex& vertex : operation.changed_vertices) {
      if (mesh.vertex_slot(vertex.id) >= 0) return TetEditStatus::kRejectedTopology;
      const std::int32_t slot = static_cast<std::int32_t>(mesh.vertex_ids_.size());
      mesh.points_.insert(mesh.points_.end(), vertex.carrier.begin(), vertex.carrier.end());
      mesh.vertex_ids_.push_back(vertex.id);
      mesh.vertex_live_.push_back(1);
      mesh.vertex_dimension_.push_back(3);
      mesh.protection_radii_.push_back(vertex.protection_radius);
      mesh.protected_vertices_.push_back(vertex.protected_vertex || vertex.protection_radius > 0.0);
      mesh.witnesses_.push_back(vertex.witness);
      mesh.vertex_has_edge_parameter_.push_back(vertex.has_edge_parameter ? 1U : 0U);
      mesh.vertex_edge_endpoints_.push_back(vertex.edge_endpoints);
      mesh.vertex_edge_parameters_.push_back(vertex.edge_parameter);
      mesh.vertex_slots_.emplace(vertex.id, slot);
      if (!vertex.ancestry.empty()) mesh.vertex_ancestry_.emplace(vertex.id, vertex.ancestry);
    }
  }

  if (relocation) {
    if (operation.changed_vertices.empty() || removed_slots.empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    for (std::int32_t moved : moved_vertex_slots) {
      for (std::size_t t = 0; t < mesh.complex_.tets.size(); ++t) {
        const std::int32_t cell = static_cast<std::int32_t>(t);
        if (!mesh.complex_.live(cell) || is_ghost(mesh.tet(cell)) ||
            phx::mc::vertex_slot(mesh.tet(cell), moved) < 0) continue;
        if (!removed_slot_set.contains(cell)) return TetEditStatus::kRejectedTopology;
      }
    }
  }

  std::unordered_set<std::uint64_t> removed_facets;
  std::vector<std::pair<std::int32_t, int>> cleared_facet_sides;
  for (std::uint64_t id : operation.removed_facet_ids) {
    if (!removed_facets.insert(id).second) return TetEditStatus::kRejectedTopology;
    const auto found = std::find_if(mesh.facets_.begin(), mesh.facets_.end(),
        [&](const TetSnapshotFacet& facet) { return facet.id == id; });
    if (found == mesh.facets_.end()) return TetEditStatus::kRejectedTopology;
    if (found->protected_facet || mesh.policy_ == BoundaryPolicy::kFixed) {
      return TetEditStatus::kRejectedProtected;
    }
    std::int32_t face[3];
    for (int k = 0; k < 3; ++k) {
      face[k] = mesh.vertex_slot(found->vertices[k]);
      if (face[k] < 0) return TetEditStatus::kRejectedTopology;
    }
    std::array<std::pair<std::int32_t, int>, 2> owners{{{-1, -1}, {-1, -1}}};
    std::size_t owner_count = 0;
    for (std::size_t t = 0; t < mesh.complex_.tets.size(); ++t) {
      const std::int32_t cell = static_cast<std::int32_t>(t);
      if (!mesh.complex_.live(cell) || is_ghost(mesh.tet(cell))) continue;
      for (int slot = 0; slot < 4; ++slot) {
        std::int32_t oriented[3];
        oriented_facet(mesh.tet(cell).v, slot, oriented);
        if (sorted_face(oriented) != sorted_face(face)) continue;
        if (owner_count == owners.size()) return TetEditStatus::kRejectedTopology;
        owners[owner_count++] = {cell, slot};
      }
    }
    if (owner_count == 0) return TetEditStatus::kRejectedTopology;
    for (std::size_t owner = 0; owner < owner_count; ++owner) {
      if (!removed_slot_set.contains(owners[owner].first)) return TetEditStatus::kRejectedTopology;
      cleared_facet_sides.push_back(owners[owner]);
    }
  }

  std::vector<TetSnapshotFacet> future_facets;
  if (!relocation) {
    future_facets.reserve(mesh.facets_.size() + operation.replacement_facets.size());
    for (const TetSnapshotFacet& facet : mesh.facets_) {
      if (!removed_facets.contains(facet.id)) future_facets.push_back(facet);
    }
    for (const TetEditFacet& facet : operation.replacement_facets) {
      if (facet.id == std::numeric_limits<std::uint64_t>::max() ||
          mesh.used_facet_ids_.contains(facet.id) || facet.protected_facet ||
          mesh.policy_ == BoundaryPolicy::kFixed) {
        return facet.protected_facet || mesh.policy_ == BoundaryPolicy::kFixed
                   ? TetEditStatus::kRejectedProtected : TetEditStatus::kRejectedTopology;
      }
      if (!mesh.source_face_valid(facet)) return TetEditStatus::kUncertainPredicate;
      TetSnapshotFacet record;
      record.id = facet.id;
      record.vertices = facet.vertices;
      record.source = facet.source;
      record.orientation = facet.orientation;
      record.protected_facet = false;
      future_facets.push_back(record);
    }
    std::sort(future_facets.begin(), future_facets.end(),
        [](const TetSnapshotFacet& a, const TetSnapshotFacet& b) { return a.id < b.id; });
    for (std::size_t i = 1; i < future_facets.size(); ++i) {
      if (future_facets[i - 1].id == future_facets[i].id) {
        return TetEditStatus::kRejectedTopology;
      }
    }
  }

  std::unordered_set<std::uint64_t> removed_segments;
  for (std::uint64_t id : operation.removed_segment_ids) {
    if (!removed_segments.insert(id).second) return TetEditStatus::kRejectedTopology;
    const auto found = std::find_if(mesh.segments_.begin(), mesh.segments_.end(),
        [&](const TetSnapshotSegment& segment) { return segment.id == id; });
    if (found == mesh.segments_.end()) return TetEditStatus::kRejectedTopology;
    if (found->protected_segment || mesh.policy_ == BoundaryPolicy::kFixed) {
      return TetEditStatus::kRejectedProtected;
    }
  }
  std::vector<TetSnapshotSegment> future_segments;
  if (!relocation) {
    future_segments.reserve(mesh.segments_.size() + operation.replacement_segments.size());
    for (const TetSnapshotSegment& segment : mesh.segments_) {
      if (!removed_segments.contains(segment.id)) future_segments.push_back(segment);
    }
    for (const TetEditSegment& segment : operation.replacement_segments) {
      if (segment.id == std::numeric_limits<std::uint64_t>::max() ||
          mesh.used_segment_ids_.contains(segment.id) || segment.protected_segment ||
          mesh.policy_ == BoundaryPolicy::kFixed) {
        return segment.protected_segment || mesh.policy_ == BoundaryPolicy::kFixed
                   ? TetEditStatus::kRejectedProtected : TetEditStatus::kRejectedTopology;
      }
      if (!mesh.source_segment_valid(segment)) return TetEditStatus::kUncertainPredicate;
      TetSnapshotSegment record;
      record.id = segment.id;
      record.vertices = segment.vertices;
      record.source = segment.source;
      record.protected_segment = false;
      future_segments.push_back(record);
    }
    std::sort(future_segments.begin(), future_segments.end(),
        [](const TetSnapshotSegment& a, const TetSnapshotSegment& b) { return a.id < b.id; });
    for (std::size_t i = 1; i < future_segments.size(); ++i) {
      if (future_segments[i - 1].id == future_segments[i].id) {
        return TetEditStatus::kRejectedTopology;
      }
    }
  }

  if (!relocation) {
    const bool boundary_valid =
        mesh.validate_regions_and_constraints(operation, removed_slots, future_facets);
    if (!boundary_valid) {
      return TetEditStatus::kRejectedTopology;
    }
  }
  for (const auto& [cell, slot] : cleared_facet_sides) {
    mesh.complex_.set_facet_constraint(cell, slot, kNoConstraint);
  }
  for (std::int32_t slot : removed_vertex_slots) {
    mesh.vertex_live_[static_cast<std::size_t>(slot)] = 0;
    mesh.vertex_slots_.erase(mesh.vertex_id(slot));
    mesh.vertex_ancestry_.erase(mesh.vertex_id(slot));
    mesh.vertex_sizes_.erase(mesh.vertex_id(slot));
  }
  if (relocation) {
    for (const TetEditVertex& vertex : operation.changed_vertices) {
      const std::int32_t slot = mesh.vertex_slot(vertex.id);
      std::copy(vertex.carrier.begin(), vertex.carrier.end(),
                mesh.points_.begin() + 3 * static_cast<std::size_t>(slot));
      mesh.witnesses_[static_cast<std::size_t>(slot)] = vertex.witness;
      // The relocation supplies no exact replacement stencil; do not retain stale weights.
      mesh.vertex_ancestry_.erase(vertex.id);
      if (vertex.has_edge_parameter) {
        mesh.vertex_has_edge_parameter_[static_cast<std::size_t>(slot)] = 1;
        mesh.vertex_edge_endpoints_[static_cast<std::size_t>(slot)] = vertex.edge_endpoints;
        mesh.vertex_edge_parameters_[static_cast<std::size_t>(slot)] = vertex.edge_parameter;
      } else if (vertex.edge_witness_supplied) {
        mesh.vertex_has_edge_parameter_[static_cast<std::size_t>(slot)] = 0;
        mesh.vertex_edge_endpoints_[static_cast<std::size_t>(slot)] = {};
        mesh.vertex_edge_parameters_[static_cast<std::size_t>(slot)] = 0.0;
      }
    }
    // A relocation cannot alter a source entity whose vertex IDs are untouched.
    // The source banks and bindings are immutable for this operation; the
    // unchanged entities retain their admitted exact-source evidence.
    for (const TetSnapshotFacet& facet : mesh.facets_) {
      const bool affected = std::any_of(
          facet.vertices.begin(), facet.vertices.end(),
          [&](std::uint64_t id) { return changed_vertex_ids.contains(id); });
      if (!affected) continue;
      TetEditFacet check;
      check.id = facet.id;
      check.vertices = facet.vertices;
      check.source = facet.source;
      check.orientation = facet.orientation;
      if (!mesh.periodic_seam_facet(facet, seam_members) && !mesh.source_face_valid(check)) {
        return TetEditStatus::kUncertainPredicate;
      }
    }
    for (const TetSnapshotSegment& segment : mesh.segments_) {
      const bool affected = std::any_of(
          segment.vertices.begin(), segment.vertices.end(),
          [&](std::uint64_t id) { return changed_vertex_ids.contains(id); });
      if (!affected) continue;
      TetEditSegment check;
      check.id = segment.id;
      check.vertices = segment.vertices;
      check.source = segment.source;
      if (!mesh.source_segment_valid(check)) return TetEditStatus::kUncertainPredicate;
    }
    for (std::int32_t slot : removed_slots) {
      const Tetrahedron& tet = mesh.tet(slot);
      if (!mesh.positive(tet.v)) return TetEditStatus::kInvalidGeometry;
    }
    if (!operation.removed_periodic_orbit_ids.empty() ||
        !operation.replacement_periodic_orbits.empty() ||
        !operation.removed_quotient_cell_ids.empty() ||
        !operation.replacement_quotient_cells.empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    if (mesh.periodic_identity_.bound) {
      const TetEditStatus periodic_status = mesh.set_periodic_state(
          mesh.periodic_orbits_, mesh.quotient_cells_,
          operation.replacement_periodic_entity_maps, false);
      if (periodic_status != TetEditStatus::kApplied) return periodic_status;
    } else if (!operation.replacement_periodic_entity_maps.empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    result.accepted_vertex_ids.reserve(operation.changed_vertices.size());
    for (const TetEditVertex& vertex : operation.changed_vertices) {
      result.accepted_vertex_ids.push_back(vertex.id);
    }
    return TetEditStatus::kApplied;
  }

  for (std::int32_t slot : removed_slots) {
    if (!mesh.complex_.live(slot) || is_ghost(mesh.tet(slot))) {
      return TetEditStatus::kRejectedTopology;
    }
  }
  for (std::size_t t = 0; t < mesh.complex_.tets.size(); ++t) {
    const std::int32_t cell = static_cast<std::int32_t>(t);
    if (!mesh.complex_.live(cell) || is_ghost(mesh.tet(cell))) continue;
    for (int corner = 0; corner < 4; ++corner) {
      if (removed_vertex_ids.contains(mesh.vertex_id(mesh.tet(cell).v[corner])) &&
          !removed_slot_set.contains(cell)) return TetEditStatus::kRejectedTopology;
    }
  }


  std::vector<CandidateCell> candidate_cells;
  candidate_cells.reserve(static_cast<std::size_t>(mesh.complex_.finite_count) -
                          removed_slots.size() + operation.replacement_cells.size());
  for (std::size_t t = 0; t < mesh.complex_.tets.size(); ++t) {
    const std::int32_t slot = static_cast<std::int32_t>(t);
    if (!mesh.complex_.live(slot) || is_ghost(mesh.tet(slot)) || removed_slot_set.contains(slot)) continue;
    CandidateCell cell;
    std::copy_n(mesh.tet(slot).v, 4, cell.vertices.begin());
    for (int facet = 0; facet < 4; ++facet) cell.constraints[facet] = mesh.complex_.constraint(slot, facet);
    cell.region = mesh.complex_.region(slot);
    candidate_cells.push_back(cell);
  }
  std::unordered_set<std::uint64_t> replacement_cell_ids;
  replacement_cell_ids.reserve(operation.replacement_cells.size());
  for (const TetEditCell& replacement : operation.replacement_cells) {
    if (replacement.id == std::numeric_limits<std::uint64_t>::max() ||
        mesh.used_cell_ids_.contains(replacement.id) ||
        !replacement_cell_ids.insert(replacement.id).second) {
      return TetEditStatus::kRejectedTopology;
    }
    if (mesh.periodic_identity_.bound && replacement.ancestor_cell_ids.empty()) {
      return TetEditStatus::kRejectedTopology;
    }
    for (std::size_t ancestor = 0; ancestor < replacement.ancestor_cell_ids.size(); ++ancestor) {
      const std::uint64_t id = replacement.ancestor_cell_ids[ancestor];
      if (!has_id(operation.removed_cell_ids, id) ||
          has_id(std::span(replacement.ancestor_cell_ids).first(ancestor), id)) {
        return TetEditStatus::kRejectedTopology;
      }
    }
    CandidateCell cell;
    for (int corner = 0; corner < 4; ++corner) {
      cell.vertices[corner] = mesh.vertex_slot(replacement.vertices[corner]);
      if (!mesh.live_vertex(cell.vertices[corner])) return TetEditStatus::kRejectedTopology;
      for (int other = 0; other < corner; ++other) {
        if (cell.vertices[corner] == cell.vertices[other]) return TetEditStatus::kRejectedTopology;
      }
      if (replacement.facet_sources[corner] < kNoConstraint) return TetEditStatus::kRejectedTopology;
    }
    cell.constraints = replacement.facet_sources;
    cell.region = replacement.region;
    cell.replacement = true;
    if (!mesh.positive(cell.vertices.data())) return TetEditStatus::kInvalidGeometry;
    candidate_cells.push_back(cell);
  }
  if (candidate_cells.empty()) return TetEditStatus::kRejectedTopology;
  if (volume_preserving) {
    Dyadic replacement_volume;
    for (const TetEditCell& replacement : operation.replacement_cells) {
      std::array<std::int32_t, 4> vertices{};
      for (int corner = 0; corner < 4; ++corner) {
        vertices[static_cast<std::size_t>(corner)] =
            mesh.vertex_slot(replacement.vertices[static_cast<std::size_t>(corner)]);
      }
      replacement_volume = replacement_volume +
                           exact_oriented_determinant(mesh, vertices.data());
    }
    if (!(removed_volume - replacement_volume).is_zero()) {
      return TetEditStatus::kRejectedTopology;
    }
  }

  std::set<SortedTet> unique_cells;
  std::map<SortedFace, CandidateFace> candidate_faces;
  std::set<std::uint64_t> candidate_edges;
  for (std::size_t row = 0; row < candidate_cells.size(); ++row) {
    CandidateCell& cell = candidate_cells[row];
    if (!mesh.positive(cell.vertices.data())) {
      return TetEditStatus::kInvalidGeometry;
    }
    if (!unique_cells.insert(sorted_tet(cell.vertices.data())).second) {
      return TetEditStatus::kInvalidGeometry;
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) candidate_edges.insert(edge_key(cell.vertices[a], cell.vertices[b]));
      std::int32_t oriented[3];
      oriented_facet(cell.vertices.data(), a, oriented);
      const SortedFace key = sorted_face(oriented);
      const auto [found, inserted] = candidate_faces.emplace(
          key, CandidateFace{{oriented[0], oriented[1], oriented[2]},
                             cell.constraints[a], 1, cell.region});
      if (!inserted) {
        CandidateFace& previous = found->second;
        const bool reversed =
            FacetKey::of(previous.oriented.data()) == FacetKey::of(oriented).reversed();
        if (previous.count != 1 || !reversed || previous.source != cell.constraints[a] ||
            (previous.source == kNoConstraint && previous.region != cell.region)) {
          return TetEditStatus::kRejectedTopology;
        }
        previous.count = 2;
      }
    }
  }

  std::map<SortedFace, std::int32_t> marked_faces;
  for (const auto& [key, face] : candidate_faces) {
    if (face.source != kNoConstraint) marked_faces.emplace(key, face.source);
    if (face.count == 1 && face.source == kNoConstraint) {
      return TetEditStatus::kRejectedProtected;
    }
  }
  std::map<SortedFace, std::int32_t> expected_marked_faces;
  for (const TetSnapshotFacet& facet : future_facets) {
    std::int32_t local[3];
    for (int corner = 0; corner < 3; ++corner) {
      local[corner] = mesh.vertex_slot(facet.vertices[corner]);
      if (!mesh.live_vertex(local[corner])) return TetEditStatus::kRejectedTopology;
    }
    if (!expected_marked_faces.emplace(sorted_face(local), facet.source).second) {
      return TetEditStatus::kRejectedTopology;
    }
  }
  if (marked_faces != expected_marked_faces) {
    return TetEditStatus::kRejectedTopology;
  }

  // Non-relocation edits cannot move retained vertices. Their source geometry was
  // admitted with the state; replacement records were checked above. Only the
  // changed incidence remains to validate here. Relocation rechecks moved geometry.

  for (const TetSnapshotSegment& segment : future_segments) {
    const std::int32_t a = mesh.vertex_slot(segment.vertices[0]);
    const std::int32_t b = mesh.vertex_slot(segment.vertices[1]);
    if (!mesh.live_vertex(a) || !mesh.live_vertex(b) || a == b ||
        !candidate_edges.contains(edge_key(a, b))) {
      return TetEditStatus::kRejectedTopology;
    }
  }


  std::size_t live_ghosts = 0;
  for (const Tetrahedron& tet : mesh.complex_.tets) {
    if (tet.v[0] != kDeadVertex && is_ghost(tet)) ++live_ghosts;
  }
  std::vector<std::int32_t> cavity_slots = removed_slots;
  cavity_slots.reserve(removed_slots.size() + live_ghosts);
  for (std::size_t slot = 0; slot < mesh.complex_.tets.size(); ++slot) {
    const std::int32_t id = static_cast<std::int32_t>(slot);
    if (mesh.complex_.live(id) && is_ghost(mesh.tet(id))) cavity_slots.push_back(id);
  }
  mesh.edit_.begin();
  for (std::int32_t slot : cavity_slots) {
    const EditStatus status = mesh.edit_.remove(slot);
    if (status != EditStatus::kOk) {
      return translate_edit_status(status);
    }
  }
  for (const CandidateCell& cell : candidate_cells) {
    if (!cell.replacement) continue;
    const EditStatus status = mesh.edit_.add(cell.vertices.data(), cell.constraints.data(), cell.region);
    if (status != EditStatus::kOk) {
      return translate_edit_status(status);
    }
  }
  for (const auto& [key, face] : candidate_faces) {
    (void)key;
    if (face.count != 1) continue;
    const std::int32_t ghost_vertices[4] = {
        kGhostVertex, face.oriented[0], face.oriented[1], face.oriented[2]};
    std::int32_t ghost_constraints[4] = {
        face.source, kNoConstraint, kNoConstraint, kNoConstraint};
    if (face.source == kNoConstraint) {
      return TetEditStatus::kRejectedProtected;
    }
    const EditStatus status = mesh.edit_.add(ghost_vertices, ghost_constraints, kNoRegion);
    if (status != EditStatus::kOk) {
      return translate_edit_status(status);
    }
  }
  const EditStatus validated = mesh.edit_.validate(
      [&mesh](const std::int32_t* vertices) { return mesh.positive(vertices); });
  if (validated != EditStatus::kOk) {
    return translate_edit_status(validated);
  }
  const EditStatus committed = mesh.edit_.commit();
  if (committed != EditStatus::kOk) {
    return translate_edit_status(committed);
  }

  for (std::uint64_t id : operation.removed_cell_ids) mesh.cell_slots_.erase(id);
  for (std::size_t i = 0; i < operation.replacement_cells.size(); ++i) {
    const std::int32_t slot = mesh.edit_.created()[i];
    if (mesh.cell_ids_.size() <= static_cast<std::size_t>(slot)) {
      mesh.cell_ids_.resize(static_cast<std::size_t>(slot) + 1,
                            std::numeric_limits<std::uint64_t>::max());
    }
    const std::uint64_t id = operation.replacement_cells[i].id;
    mesh.cell_ids_[static_cast<std::size_t>(slot)] = id;
    mesh.cell_slots_.emplace(id, slot);
    mesh.used_cell_ids_.insert(id);
    if (id >= mesh.next_cell_id_ && id != std::numeric_limits<std::uint64_t>::max()) {
      mesh.next_cell_id_ = id + 1;
    }
    result.accepted_cell_ids.push_back(id);
    mesh.cell_ancestry_.emplace(
        id, TetCellAncestryRecord{static_cast<std::int32_t>(operation.kind),
                                  operation.replacement_cells[i].ancestor_cell_ids});
  }
  for (std::size_t i = operation.replacement_cells.size();
       i < mesh.edit_.created().size(); ++i) {
    const std::int32_t slot = mesh.edit_.created()[i];
    if (mesh.cell_ids_.size() <= static_cast<std::size_t>(slot)) {
      mesh.cell_ids_.resize(static_cast<std::size_t>(slot) + 1,
                            std::numeric_limits<std::uint64_t>::max());
    }
    mesh.cell_ids_[static_cast<std::size_t>(slot)] = std::numeric_limits<std::uint64_t>::max();
  }
  for (std::int32_t slot : removed_slots) {
    if (std::find(mesh.edit_.created().begin(), mesh.edit_.created().end(), slot) ==
        mesh.edit_.created().end()) {
      if (mesh.cell_ids_.size() > static_cast<std::size_t>(slot)) {
        mesh.cell_ids_[static_cast<std::size_t>(slot)] = std::numeric_limits<std::uint64_t>::max();
      }
    }
  }
  for (std::int32_t slot : removed_vertex_slots) {
    mesh.used_vertex_ids_.insert(mesh.vertex_id(slot));
  }
  mesh.facets_.swap(future_facets);
  mesh.segments_.swap(future_segments);
  for (std::uint64_t id : operation.removed_facet_ids) mesh.used_facet_ids_.insert(id);
  for (const TetEditFacet& facet : operation.replacement_facets) {
    mesh.used_facet_ids_.insert(facet.id);
    if (facet.id >= mesh.next_facet_id_ && facet.id != std::numeric_limits<std::uint64_t>::max()) {
      mesh.next_facet_id_ = facet.id + 1;
    }
  }
  for (std::uint64_t id : operation.removed_segment_ids) mesh.used_segment_ids_.insert(id);
  for (const TetEditSegment& segment : operation.replacement_segments) {
    mesh.used_segment_ids_.insert(segment.id);
    if (segment.id >= mesh.next_segment_id_ && segment.id != std::numeric_limits<std::uint64_t>::max()) {
      mesh.next_segment_id_ = segment.id + 1;
    }
  }
  for (const TetEditVertex& vertex : operation.changed_vertices) {
    if (mesh.vertex_slot(vertex.id) >= 0) {
      result.accepted_vertex_ids.push_back(vertex.id);
      if (vertex.id >= mesh.next_vertex_id_) mesh.next_vertex_id_ = vertex.id + 1;
      const std::int32_t slot = mesh.vertex_slot(vertex.id);
      if (slot >= 0 && static_cast<std::size_t>(slot) >= mesh.protected_vertices_.size()) {
        return TetEditStatus::kRejectedTopology;
      }
    }
  }
  for (const TetEditFacet& facet : operation.replacement_facets) {
    for (std::uint64_t id : facet.vertices) {
      const std::int32_t slot = mesh.vertex_slot(id);
      if (slot >= 0) mesh.vertex_dimension_[static_cast<std::size_t>(slot)] =
                          std::min<std::int8_t>(mesh.vertex_dimension_[static_cast<std::size_t>(slot)], 2);
    }
  }
  for (const TetEditSegment& segment : operation.replacement_segments) {
    for (std::uint64_t id : segment.vertices) {
      const std::int32_t slot = mesh.vertex_slot(id);
      if (slot >= 0) mesh.vertex_dimension_[static_cast<std::size_t>(slot)] =
                          std::min<std::int8_t>(mesh.vertex_dimension_[static_cast<std::size_t>(slot)], 1);
    }
  }
  if (!operation.removed_periodic_orbit_ids.empty() ||
      !operation.replacement_periodic_orbits.empty() ||
      !operation.removed_quotient_cell_ids.empty() ||
      !operation.replacement_quotient_cells.empty() ||
      !operation.replacement_periodic_entity_maps.empty()) {
    std::vector<TetPeriodicOrbit> future_orbits;
    std::vector<TetPeriodicQuotientCell> future_quotients;
    future_orbits.reserve(mesh.periodic_orbits_.size() + operation.replacement_periodic_orbits.size());
    future_quotients.reserve(mesh.quotient_cells_.size() + operation.replacement_quotient_cells.size());
    for (const TetPeriodicOrbit& orbit : mesh.periodic_orbits_) {
      if (!has_id(operation.removed_periodic_orbit_ids, orbit.id)) future_orbits.push_back(orbit);
    }
    future_orbits.insert(future_orbits.end(), operation.replacement_periodic_orbits.begin(),
                         operation.replacement_periodic_orbits.end());
    for (const TetPeriodicQuotientCell& cell : mesh.quotient_cells_) {
      if (!has_id(operation.removed_quotient_cell_ids, cell.id)) future_quotients.push_back(cell);
    }
    future_quotients.insert(future_quotients.end(), operation.replacement_quotient_cells.begin(),
                            operation.replacement_quotient_cells.end());
    const TetEditStatus periodic_status = mesh.set_periodic_state(
        future_orbits, future_quotients,
        operation.replacement_periodic_entity_maps, false);
    if (periodic_status != TetEditStatus::kApplied) return periodic_status;
  }
  if (!mesh.audit_state()) {
    return TetEditStatus::kRejectedTopology;
  }
  return TetEditStatus::kApplied;
}

}  // namespace phx::mc
