//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact refusal probe of flip and ring-reconnection edits. validate_and_commit
// checks a replacement against the whole staged state; on a state that already
// passes every check of its retained part (certified once per generation), each
// check reduces to the cells around the cavity. The probe evaluates those local
// checks in validate_and_commit's order and reports the first refusal; an edit
// that passes them is left to apply_edit_batch, which commits it.
#include "tet_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dyadic.hpp"
#include "surface_remesh.hpp"

namespace phx::mc {
namespace {

using SortedFace = std::array<std::int32_t, 3>;
using SortedTet = std::array<std::int32_t, 4>;

SortedFace sorted_face(const std::int32_t* vertices) {
  SortedFace key{vertices[0], vertices[1], vertices[2]};
  std::sort(key.begin(), key.end());
  return key;
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

struct SortedFaceHash {
  std::size_t operator()(const SortedFace& face) const noexcept {
    std::uint64_t hash = 0x9E3779B97F4A7C15ULL;
    for (std::int32_t value : face) {
      hash ^= static_cast<std::uint32_t>(value) + 0x9E3779B97F4A7C15ULL + (hash << 6) + (hash >> 2);
    }
    return static_cast<std::size_t>(hash);
  }
};

bool has_id(std::span<const std::uint64_t> ids, std::uint64_t value) {
  return std::find(ids.begin(), ids.end(), value) != ids.end();
}

// Same link condition as the collapse check of validate_and_commit.
bool full_link_condition(const TetMesh& mesh, std::int32_t remove, std::int32_t keep) {
  using SortedEdge = std::array<std::int32_t, 2>;
  const auto sorted_edge = [](std::int32_t a, std::int32_t b) {
    return SortedEdge{std::min(a, b), std::max(a, b)};
  };
  std::set<SortedFace> remove_faces, keep_faces;
  std::set<SortedEdge> remove_edges, keep_edges, edge_edges;
  std::set<std::int32_t> remove_vertices, keep_vertices, edge_vertices;
  const auto add_vertex_link = [&](const Tetrahedron& tet, std::int32_t vertex,
                                   std::set<std::int32_t>& vertices,
                                   std::set<SortedEdge>& edges, std::set<SortedFace>& faces) {
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
    if (has_remove) add_vertex_link(tet, remove, remove_vertices, remove_edges, remove_faces);
    if (has_keep) add_vertex_link(tet, keep, keep_vertices, keep_edges, keep_faces);
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
  std::set_intersection(remove_vertices.begin(), remove_vertices.end(), keep_vertices.begin(),
                        keep_vertices.end(),
                        std::inserter(common_vertices, common_vertices.end()));
  std::set<SortedEdge> common_edges;
  std::set_intersection(remove_edges.begin(), remove_edges.end(), keep_edges.begin(),
                        keep_edges.end(), std::inserter(common_edges, common_edges.end()));
  std::set<SortedFace> common_faces;
  std::set_intersection(remove_faces.begin(), remove_faces.end(), keep_faces.begin(),
                        keep_faces.end(), std::inserter(common_faces, common_faces.end()));
  return common_vertices == edge_vertices && common_edges == edge_edges &&
         common_faces.empty();
}

// Same exact determinant as the volume check of validate_and_commit.
Dyadic exact_oriented_determinant(const TetMesh& mesh, const std::int32_t* vertices) {
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

// Same source-side region binding as validate_regions_and_constraints.
bool collect_source_regions(const TetMesh& mesh, std::span<const TetSnapshotFacet> facets,
                            std::map<std::pair<std::int32_t, int>, std::int32_t>& regions) {
  using IndexedFacet = std::pair<FacetKey, const TetSnapshotFacet*>;
  std::vector<IndexedFacet> indexed_facets;
  indexed_facets.reserve(facets.size());
  for (const TetSnapshotFacet& facet : facets) {
    std::int32_t local[3];
    for (int corner = 0; corner < 3; ++corner) {
      local[corner] = mesh.vertex_slot(facet.vertices[static_cast<std::size_t>(corner)]);
      if (local[corner] < 0) return false;
    }
    indexed_facets.emplace_back(FacetKey::of(local), &facet);
  }
  std::sort(indexed_facets.begin(), indexed_facets.end(),
            [](const IndexedFacet& first, const IndexedFacet& second) {
              return first.first < second.first;
            });
  const auto bind_region = [&](const FacetKey& key, int direction, std::int32_t source,
                               std::int32_t region) {
    auto found = std::lower_bound(
        indexed_facets.begin(), indexed_facets.end(), key,
        [](const IndexedFacet& facet, const FacetKey& requested) {
          return facet.first < requested;
        });
    for (; found != indexed_facets.end() && found->first == key; ++found) {
      const TetSnapshotFacet& facet = *found->second;
      if (facet.source != source) continue;
      const auto [entry, inserted] =
          regions.emplace(std::pair{source, direction * facet.orientation}, region);
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
          !bind_region(key.reversed(), -1, source, complex.region(slot))) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

// A certified state passes, for any replacement that retains a subset of its
// cells, every check validate_and_commit applies to the retained part alone:
// positive and distinct cells, faces shared by at most two cells, which are
// adjacent across them with opposite orientations, equal constraints and,
// unconstrained, equal regions; constrained boundary faces; constrained faces
// exactly the facets; facet and segment IDs unique and used; segments edges;
// and a consistent source-side region binding.
bool TetMesh::certify_replacement_probe(TetReplacementProbeCache& cache) const {
  if (cache.generation == generation_) return cache.certified;
  cache.generation = generation_;
  cache.certified = false;
  cache.source_regions.clear();
  cache.facet_row_by_id.clear();
  cache.facet_row_by_face.clear();
  if (periodic_identity_.bound || !periodic_orbits_.empty() || !quotient_cells_.empty() ||
      !periodic_entity_maps_.empty() || !complex_.audit()) {
    return false;
  }
  struct FaceRecord {
    SortedFace key;
    FacetKey oriented;
    std::int32_t cell;
    int slot;
  };
  std::vector<FaceRecord> faces;
  std::vector<SortedTet> cells;
  std::unordered_set<std::uint64_t> edges;
  faces.reserve(4 * static_cast<std::size_t>(complex_.finite_count));
  cells.reserve(static_cast<std::size_t>(complex_.finite_count));
  edges.reserve(2 * static_cast<std::size_t>(complex_.finite_count));
  for (std::size_t row = 0; row < complex_.tets.size(); ++row) {
    const std::int32_t slot = static_cast<std::int32_t>(row);
    if (!complex_.live(slot) || is_ghost(tet(slot))) continue;
    const Tetrahedron& cell = tet(slot);
    if (!positive(cell.v)) return false;
    cells.push_back(sorted_tet(cell.v));
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) edges.insert(edge_key(cell.v[a], cell.v[b]));
      std::int32_t face[3];
      oriented_facet(cell.v, a, face);
      faces.push_back({sorted_face(face), FacetKey::of(face), slot, a});
    }
  }
  std::sort(cells.begin(), cells.end());
  if (std::adjacent_find(cells.begin(), cells.end()) != cells.end()) return false;
  std::sort(faces.begin(), faces.end(), [](const FaceRecord& first, const FaceRecord& second) {
    return std::tie(first.key, first.cell, first.slot) <
           std::tie(second.key, second.cell, second.slot);
  });
  std::vector<std::pair<SortedFace, std::int32_t>> marked;
  for (std::size_t begin = 0; begin < faces.size();) {
    std::size_t end = begin + 1;
    while (end < faces.size() && faces[end].key == faces[begin].key) ++end;
    if (end - begin > 2) return false;
    const FaceRecord& first = faces[begin];
    const std::int32_t source = complex_.constraint(first.cell, first.slot);
    const std::int32_t neighbor = tet(first.cell).n[first.slot];
    if (end - begin == 2) {
      const FaceRecord& second = faces[begin + 1];
      if (!(first.oriented == second.oriented.reversed()) ||
          source != complex_.constraint(second.cell, second.slot) ||
          (source == kNoConstraint &&
           complex_.region(first.cell) != complex_.region(second.cell)) ||
          neighbor != second.cell) {
        return false;
      }
    } else if (source == kNoConstraint || !complex_.live(neighbor) ||
               !is_ghost(tet(neighbor))) {
      return false;
    }
    if (source != kNoConstraint) marked.emplace_back(first.key, source);
    begin = end;
  }
  std::vector<std::pair<SortedFace, std::int32_t>> expected;
  expected.reserve(facets_.size());
  for (std::size_t row = 0; row < facets_.size(); ++row) {
    const TetSnapshotFacet& facet = facets_[row];
    if (!used_facet_ids_.contains(facet.id) ||
        !cache.facet_row_by_id.emplace(facet.id, row).second) {
      return false;
    }
    std::int32_t local[3];
    for (int corner = 0; corner < 3; ++corner) {
      local[corner] = vertex_slot(facet.vertices[static_cast<std::size_t>(corner)]);
      if (!live_vertex(local[corner])) return false;
    }
    const SortedFace key = sorted_face(local);
    if (!cache.facet_row_by_face.emplace(key, row).second) return false;
    expected.emplace_back(key, facet.source);
  }
  std::sort(expected.begin(), expected.end());
  if (marked != expected) return false;
  std::unordered_set<std::uint64_t> segment_ids;
  for (const TetSnapshotSegment& segment : segments_) {
    const std::int32_t a = vertex_slot(segment.vertices[0]);
    const std::int32_t b = vertex_slot(segment.vertices[1]);
    if (!segment_ids.insert(segment.id).second || !used_segment_ids_.contains(segment.id) ||
        !live_vertex(a) || !live_vertex(b) ||
        a == b || !edges.contains(edge_key(a, b))) {
      return false;
    }
  }
  if (!collect_source_regions(*this, facets_, cache.source_regions)) {
    cache.source_regions.clear();
    return false;
  }
  cache.certified = true;
  return true;
}

bool TetMesh::probe_cavity_edit(const TetEdit& operation, TetReplacementProbeCache& cache,
                                TetEditStatus& status) const {
  const bool collapse = operation.kind == TetEditKind::kCollapse;
  if (!collapse && operation.kind != TetEditKind::kFlip23 &&
      operation.kind != TetEditKind::kFlip32 &&
      operation.kind != TetEditKind::kRingReconnect) {
    return false;
  }
  if ((!collapse && (!operation.changed_vertices.empty() ||
                     !operation.removed_vertex_ids.empty() ||
                     !operation.collapse_pairs.empty())) ||
      !operation.split_edges.empty() || !operation.periodic_members.empty() ||
      !operation.removed_periodic_orbit_ids.empty() ||
      !operation.replacement_periodic_orbits.empty() ||
      !operation.removed_quotient_cell_ids.empty() ||
      !operation.replacement_quotient_cells.empty() ||
      !operation.replacement_periodic_entity_maps.empty() ||
      generation_ == std::numeric_limits<std::uint64_t>::max() ||
      !certify_replacement_probe(cache)) {
    return false;
  }
  const TetImplicitSource* implicit = implicit_source_.get();
  const std::int64_t transversality = implicit == nullptr ? 0 : implicit->transversality_refusals;
  const std::int64_t distance = implicit == nullptr ? 0 : implicit->distance_refusals;
  const bool inconclusive = implicit != nullptr && implicit->last_inconclusive;
  // A refusal leaves the source counters as apply_edit_batch's own checks
  // would; an undecided edit restores them for apply_edit_batch to advance.
  const auto refuse = [&](TetEditStatus refusal) {
    status = refusal;
    return true;
  };
  const auto defer = [&]() {
    if (implicit != nullptr) {
      implicit->transversality_refusals = transversality;
      implicit->distance_refusals = distance;
      implicit->last_inconclusive = inconclusive;
    }
    return false;
  };
  constexpr TetEditStatus topology = TetEditStatus::kRejectedTopology;

  // validate_periodic_closure on a state without periodic records.
  for (const TetEditVertex& vertex : operation.changed_vertices) {
    for (const TetVertexAncestry& ancestor : vertex.ancestry) {
      if (ancestor.source_winding[0] != 0 || ancestor.source_winding[1] != 0 ||
          ancestor.source_winding[2] != 0) {
        return refuse(topology);
      }
    }
  }
  for (std::uint64_t id : operation.removed_cell_ids) {
    if (cell_slot(id) < 0) return refuse(topology);
  }
  if (operation.replacement_cells.empty()) return refuse(topology);
  if (collapse && (operation.removed_vertex_ids.empty() ||
                   operation.collapse_pairs.size() != operation.removed_vertex_ids.size())) {
    return refuse(topology);
  }
  std::unordered_set<std::uint64_t> removed_cell_ids;
  std::vector<std::int32_t> removed_slots;
  removed_slots.reserve(operation.removed_cell_ids.size());
  for (std::uint64_t id : operation.removed_cell_ids) {
    const std::int32_t slot = cell_slot(id);
    if (slot < 0 || is_ghost(tet(slot)) || !removed_cell_ids.insert(id).second) {
      return refuse(topology);
    }
    removed_slots.push_back(slot);
  }
  if (removed_slots.empty()) return refuse(topology);
  const std::unordered_set<std::int32_t> removed_slot_set(removed_slots.begin(),
                                                          removed_slots.end());
  Dyadic removed_volume;
  if (!collapse) {
    for (std::int32_t slot : removed_slots) {
      removed_volume = removed_volume + exact_oriented_determinant(*this, tet(slot).v);
    }
  }

  // Collapse: removed vertices with their whole stars, linked survivor pairs,
  // and moved survivors (the only changed vertices), whose stars are removed.
  std::unordered_set<std::uint64_t> removed_vertex_ids;
  std::vector<std::int32_t> removed_vertex_slots;
  const auto star_removed = [&](std::int32_t vertex) {
    for (std::size_t row = 0; row < complex_.tets.size(); ++row) {
      const std::int32_t cell = static_cast<std::int32_t>(row);
      if (!complex_.live(cell) || is_ghost(tet(cell)) ||
          phx::mc::vertex_slot(tet(cell), vertex) < 0) {
        continue;
      }
      if (!removed_slot_set.contains(cell)) return false;
    }
    return true;
  };
  for (std::uint64_t id : operation.removed_vertex_ids) {
    const std::int32_t slot = vertex_slot(id);
    if (!live_vertex(slot)) return refuse(topology);
    if (protected_vertex(slot) || protection_radius(slot) > 0.0) {
      return refuse(TetEditStatus::kRejectedProtected);
    }
    if (!removed_vertex_ids.insert(id).second) return refuse(topology);
    removed_vertex_slots.push_back(slot);
    if (!star_removed(slot)) return refuse(topology);
  }
  std::unordered_set<std::uint64_t> paired_survivors;
  if (collapse) {
    std::unordered_set<std::uint64_t> paired_removed;
    for (const TetCollapsePair& pair : operation.collapse_pairs) {
      if (!removed_vertex_ids.contains(pair.removed_vertex_id) ||
          removed_vertex_ids.contains(pair.survivor_vertex_id) ||
          !paired_removed.insert(pair.removed_vertex_id).second ||
          !paired_survivors.insert(pair.survivor_vertex_id).second) {
        return refuse(topology);
      }
      const std::int32_t remove = vertex_slot(pair.removed_vertex_id);
      const std::int32_t keep = vertex_slot(pair.survivor_vertex_id);
      if (!live_vertex(keep) || remove == keep) return refuse(topology);
      if (protected_vertex(keep) || protection_radius(keep) > 0.0) {
        return refuse(TetEditStatus::kRejectedProtected);
      }
      if (!full_link_condition(*this, remove, keep)) return refuse(topology);
    }
  }
  std::unordered_set<std::uint64_t> changed_vertex_ids;
  std::vector<std::int32_t> moved_vertex_slots;
  std::size_t new_count = 0;
  for (const TetEditVertex& vertex : operation.changed_vertices) {
    if (vertex.id == std::numeric_limits<std::uint64_t>::max()) {
      return refuse(TetEditStatus::kIndexOverflow);
    }
    if (!changed_vertex_ids.insert(vertex.id).second ||
        (used_vertex_ids_.contains(vertex.id) && vertex_slot(vertex.id) < 0)) {
      return refuse(topology);
    }
    const std::int32_t slot = vertex_slot(vertex.id);
    const bool moved_survivor = slot >= 0 && paired_survivors.contains(vertex.id);
    if (slot >= 0) {
      if (!vertex.ancestry.empty() || !moved_survivor) return refuse(topology);
      if (protected_vertex(slot) || protection_radius(slot) > 0.0 ||
          vertex.protected_vertex != protected_vertex(slot) ||
          vertex.protection_radius != protection_radius(slot) ||
          vertex.witness.stratum != witness(slot).stratum) {
        return refuse(TetEditStatus::kRejectedProtected);
      }
      moved_vertex_slots.push_back(slot);
    } else {
      // A new vertex is refused below (validate_and_commit stages it first).
      if (vertex_ids_.size() + new_count >=
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return refuse(TetEditStatus::kIndexOverflow);
      }
      ++new_count;
    }
    TetEditVertex checked_vertex = vertex;
    if (moved_survivor && !checked_vertex.edge_witness_supplied &&
        !checked_vertex.has_edge_parameter &&
        vertex_has_edge_parameter_[static_cast<std::size_t>(slot)] != 0) {
      checked_vertex.has_edge_parameter = true;
      checked_vertex.edge_endpoints = vertex_edge_endpoints_[static_cast<std::size_t>(slot)];
      checked_vertex.edge_parameter = vertex_edge_parameters_[static_cast<std::size_t>(slot)];
    }
    if (!std::isfinite(vertex.protection_radius) || vertex.protection_radius < 0.0 ||
        !source_carrier_valid(checked_vertex)) {
      return refuse(TetEditStatus::kInvalidGeometry);
    }
    if (vertex.protection_radius > 0.0 && !vertex.protected_vertex) {
      return refuse(TetEditStatus::kRejectedProtected);
    }
    for (std::size_t other = 0; other < vertex_ids_.size(); ++other) {
      if (!vertex_live_[other] || vertex_ids_[other] == vertex.id ||
          removed_vertex_ids.contains(vertex_ids_[other])) {
        continue;
      }
      const double* point = points_.data() + 3 * other;
      if (point[0] == vertex.carrier[0] && point[1] == vertex.carrier[1] &&
          point[2] == vertex.carrier[2]) {
        return refuse(TetEditStatus::kInvalidGeometry);
      }
    }
  }
  if (new_count != 0) return refuse(topology);
  for (std::int32_t moved : moved_vertex_slots) {
    if (!star_removed(moved)) return refuse(topology);
  }
  // Moved survivors carry their new carriers and witnesses in every later
  // check, as in the staged state.
  const RelocationOverlay overlay{operation.changed_vertices};
  const RelocationOverlay* moved = collapse ? &overlay : nullptr;
  const auto carrier = [&](std::int32_t slot) -> const double* {
    const TetEditVertex* changed = moved == nullptr ? nullptr : moved->find(vertex_id(slot));
    return changed == nullptr ? point(slot) : changed->carrier.data();
  };

  // Removed facets: each is owned only by removed cells. On a certified state
  // the finite owners of a face of a removed cell are that cell and its
  // finite neighbour across the face.
  std::unordered_set<std::uint64_t> removed_facets;
  std::set<SortedFace> retired;
  for (std::uint64_t id : operation.removed_facet_ids) {
    if (!removed_facets.insert(id).second) return refuse(topology);
    const auto row = cache.facet_row_by_id.find(id);
    if (row == cache.facet_row_by_id.end()) return refuse(topology);
    const TetSnapshotFacet& facet = facets_[row->second];
    if (facet.protected_facet || policy_ == BoundaryPolicy::kFixed) {
      return refuse(TetEditStatus::kRejectedProtected);
    }
    std::int32_t local[3];
    for (int corner = 0; corner < 3; ++corner) {
      local[corner] = vertex_slot(facet.vertices[static_cast<std::size_t>(corner)]);
      if (local[corner] < 0) return refuse(topology);
    }
    const SortedFace key = sorted_face(local);
    std::int32_t owner = -1;
    int owner_face = -1;
    for (std::int32_t slot : removed_slots) {
      for (int face_slot = 0; face_slot < 4 && owner < 0; ++face_slot) {
        std::int32_t face[3];
        oriented_facet(tet(slot).v, face_slot, face);
        if (sorted_face(face) == key) {
          owner = slot;
          owner_face = face_slot;
        }
      }
      if (owner >= 0) break;
    }
    if (owner < 0) return refuse(topology);
    const std::int32_t other = tet(owner).n[owner_face];
    if (!is_ghost(tet(other)) && !removed_slot_set.contains(other)) return refuse(topology);
    retired.insert(key);
  }

  // Replacement facets, in order, then the future facet IDs.
  for (const TetEditFacet& facet : operation.replacement_facets) {
    if (facet.id == std::numeric_limits<std::uint64_t>::max() ||
        used_facet_ids_.contains(facet.id) || facet.protected_facet ||
        policy_ == BoundaryPolicy::kFixed) {
      return refuse(facet.protected_facet || policy_ == BoundaryPolicy::kFixed
                        ? TetEditStatus::kRejectedProtected
                        : topology);
    }
    if (!source_face_valid(facet, moved)) return refuse(source_face_refusal_status());
  }
  std::unordered_set<std::uint64_t> replacement_facet_ids;
  for (const TetEditFacet& facet : operation.replacement_facets) {
    if (!replacement_facet_ids.insert(facet.id).second) return refuse(topology);
  }
  // Every replacement facet vertex is live (source_face_valid).
  std::vector<std::array<std::int32_t, 3>> replacement_facet_slots;
  replacement_facet_slots.reserve(operation.replacement_facets.size());
  for (const TetEditFacet& facet : operation.replacement_facets) {
    replacement_facet_slots.push_back({vertex_slot(facet.vertices[0]),
                                       vertex_slot(facet.vertices[1]),
                                       vertex_slot(facet.vertices[2])});
  }
  // Removed and replacement segments, then the future segment IDs.
  std::unordered_set<std::uint64_t> removed_segments;
  for (std::uint64_t id : operation.removed_segment_ids) {
    if (!removed_segments.insert(id).second) return refuse(topology);
    const auto found = std::find_if(segments_.begin(), segments_.end(),
        [&](const TetSnapshotSegment& segment) { return segment.id == id; });
    if (found == segments_.end()) return refuse(topology);
    if (found->protected_segment || policy_ == BoundaryPolicy::kFixed) {
      return refuse(TetEditStatus::kRejectedProtected);
    }
  }
  for (const TetEditSegment& segment : operation.replacement_segments) {
    if (segment.id == std::numeric_limits<std::uint64_t>::max() ||
        used_segment_ids_.contains(segment.id) || segment.protected_segment ||
        policy_ == BoundaryPolicy::kFixed) {
      return refuse(segment.protected_segment || policy_ == BoundaryPolicy::kFixed
                        ? TetEditStatus::kRejectedProtected
                        : topology);
    }
    if (!source_segment_valid(segment, moved)) return refuse(TetEditStatus::kInvalidGeometry);
  }
  std::unordered_set<std::uint64_t> replacement_segment_ids;
  for (const TetEditSegment& segment : operation.replacement_segments) {
    if (!replacement_segment_ids.insert(segment.id).second) return refuse(topology);
  }
  // Future facets and segments in ID order, as validate_and_commit keeps them.
  struct FutureFacet {
    std::uint64_t id;
    std::array<std::uint64_t, 3> vertices;
    std::int32_t source;
    std::int8_t orientation;
    bool replacement;
  };
  std::vector<FutureFacet> future_facets;
  future_facets.reserve(facets_.size() + operation.replacement_facets.size());
  for (const TetSnapshotFacet& facet : facets_) {
    if (!removed_facets.contains(facet.id)) {
      future_facets.push_back({facet.id, facet.vertices, facet.source, facet.orientation, false});
    }
  }
  for (const TetEditFacet& facet : operation.replacement_facets) {
    future_facets.push_back({facet.id, facet.vertices, facet.source, facet.orientation, true});
  }
  std::sort(future_facets.begin(), future_facets.end(),
            [](const FutureFacet& a, const FutureFacet& b) { return a.id < b.id; });
  std::vector<TetSnapshotSegment> future_segments;
  future_segments.reserve(segments_.size() + operation.replacement_segments.size());
  for (const TetSnapshotSegment& segment : segments_) {
    if (!removed_segments.contains(segment.id)) future_segments.push_back(segment);
  }
  for (const TetEditSegment& segment : operation.replacement_segments) {
    TetSnapshotSegment record;
    record.id = segment.id;
    record.vertices = segment.vertices;
    record.source = segment.source;
    future_segments.push_back(record);
  }
  std::sort(future_segments.begin(), future_segments.end(),
            [](const TetSnapshotSegment& a, const TetSnapshotSegment& b) { return a.id < b.id; });
  const auto through_moved = [&](std::span<const std::uint64_t> vertices) {
    return std::any_of(vertices.begin(), vertices.end(),
                       [&](std::uint64_t id) { return changed_vertex_ids.contains(id); });
  };
  if (collapse && !changed_vertex_ids.empty()) {
    for (const FutureFacet& facet : future_facets) {
      if (!through_moved(facet.vertices)) continue;
      TetEditFacet check;
      check.id = facet.id;
      check.vertices = facet.vertices;
      check.source = facet.source;
      check.orientation = facet.orientation;
      if (!source_face_valid(check, moved)) return refuse(source_face_refusal_status());
    }
    for (const TetSnapshotSegment& segment : future_segments) {
      if (!through_moved(segment.vertices)) continue;
      TetEditSegment check;
      check.id = segment.id;
      check.vertices = segment.vertices;
      check.source = segment.source;
      if (!source_segment_valid(check, moved)) return refuse(TetEditStatus::kInvalidGeometry);
    }
  }

  // implicit_facets_embedded: only pairs with a changed facet (a replacement
  // facet or one through a moved vertex) are checked.
  if (implicit_source_ && (!operation.replacement_facets.empty() || !changed_vertex_ids.empty())) {
    struct Box {
      std::array<std::int32_t, 3> slots;
      std::array<double, 3> lower, upper;
      bool implicit;
      bool changed;
    };
    const auto legal = [&](const Box& first, const Box& second) {
      std::array<remesh::Point, 3> a{}, b{};
      remesh::Triangle a_ids{}, b_ids{};
      for (std::size_t corner = 0; corner < 3; ++corner) {
        a_ids[corner] = first.slots[corner];
        b_ids[corner] = second.slots[corner];
        std::copy_n(carrier(first.slots[corner]), 3, a[corner].begin());
        std::copy_n(carrier(second.slots[corner]), 3, b[corner].begin());
      }
      return remesh::legal_triangle_contact(a, a_ids, b, b_ids);
    };
    // 1: legal, 0: illegal, -1: tied sort keys whose two orders disagree.
    const auto contact = [&](const Box& changed, const Box& other) -> int {
      if ((!changed.implicit && !other.implicit) || changed.upper[0] < other.lower[0] ||
          other.upper[0] < changed.lower[0] || changed.upper[1] < other.lower[1] ||
          other.upper[1] < changed.lower[1] || changed.upper[2] < other.lower[2] ||
          other.upper[2] < changed.lower[2]) {
        return 1;
      }
      if (changed.lower[0] < other.lower[0]) return legal(changed, other) ? 1 : 0;
      if (other.lower[0] < changed.lower[0]) return legal(other, changed) ? 1 : 0;
      const bool forward = legal(changed, other);
      return forward == legal(other, changed) ? (forward ? 1 : 0) : -1;
    };
    std::vector<Box> boxes;
    boxes.reserve(future_facets.size());
    for (const FutureFacet& facet : future_facets) {
      Box entry{};
      entry.implicit = std::binary_search(implicit_source_->groups.begin(),
                                          implicit_source_->groups.end(), facet.source);
      entry.changed = facet.replacement || through_moved(facet.vertices);
      entry.lower.fill(std::numeric_limits<double>::infinity());
      entry.upper.fill(-std::numeric_limits<double>::infinity());
      for (std::size_t corner = 0; corner < 3; ++corner) {
        const std::int32_t slot = vertex_slot(facet.vertices[corner]);
        if (!live_vertex(slot)) return refuse(TetEditStatus::kInvalidGeometry);
        entry.slots[corner] = slot;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          entry.lower[axis] = std::min(entry.lower[axis], carrier(slot)[axis]);
          entry.upper[axis] = std::max(entry.upper[axis], carrier(slot)[axis]);
        }
      }
      boxes.push_back(entry);
    }
    for (std::size_t row = 0; row < boxes.size(); ++row) {
      if (!boxes[row].changed) continue;
      for (std::size_t other = 0; other < boxes.size(); ++other) {
        if (other == row || (boxes[other].changed && other < row)) continue;
        const int result = contact(boxes[row], boxes[other]);
        if (result < 0) return defer();
        if (result == 0) return refuse(TetEditStatus::kInvalidGeometry);
      }
    }
  }

  // validate_regions_and_constraints.
  std::unordered_map<SortedFace, std::size_t, SortedFaceHash> replacement_facet_by_face;
  const auto regions_valid = [&]() {
    bool dead_replacement = false;
    std::unordered_map<SortedFace, std::vector<std::int32_t>, SortedFaceHash>
        replacement_face_regions;
    for (const TetEditCell& cell : operation.replacement_cells) {
      std::int32_t local[4];
      bool dead = false;
      for (int corner = 0; corner < 4; ++corner) {
        local[corner] = vertex_slot(cell.vertices[static_cast<std::size_t>(corner)]);
        dead = dead || local[corner] < 0;
      }
      if (dead) {
        dead_replacement = true;
        continue;
      }
      for (int face_slot = 0; face_slot < 4; ++face_slot) {
        std::int32_t face[3];
        oriented_facet(local, face_slot, face);
        replacement_face_regions[sorted_face(face)].push_back(cell.region);
      }
    }
    for (std::int32_t slot : removed_slots) {
      const Tetrahedron& cell = tet(slot);
      for (int face_slot = 0; face_slot < 4; ++face_slot) {
        const std::int32_t neighbor = cell.n[face_slot];
        if (removed_slot_set.contains(neighbor)) continue;
        std::int32_t face[3];
        oriented_facet(cell.v, face_slot, face);
        const SortedFace key = sorted_face(face);
        if (is_ghost(tet(neighbor)) && !retired.empty() && retired.contains(key)) continue;
        if (dead_replacement) return false;
        const auto found = replacement_face_regions.find(key);
        if (found == replacement_face_regions.end()) return false;
        for (std::int32_t region : found->second) {
          if (region != complex_.region(slot)) return false;
        }
      }
    }
    // Future facets: retained facets have distinct live faces (certified).
    for (std::size_t row = 0; row < operation.replacement_facets.size(); ++row) {
      const std::array<std::int32_t, 3>& local = replacement_facet_slots[row];
      for (std::int32_t slot : local) {
        if (!live_vertex(slot)) return false;
      }
      const SortedFace key = sorted_face(local.data());
      const auto existing = cache.facet_row_by_face.find(key);
      if ((existing != cache.facet_row_by_face.end() &&
           !removed_facets.contains(facets_[existing->second].id)) ||
          !replacement_facet_by_face.emplace(key, row).second) {
        return false;
      }
    }
    for (const TetEditCell& cell : operation.replacement_cells) {
      std::int32_t local[4];
      for (int corner = 0; corner < 4; ++corner) {
        local[corner] = vertex_slot(cell.vertices[static_cast<std::size_t>(corner)]);
        if (!live_vertex(local[corner])) return false;
      }
      for (int face_slot = 0; face_slot < 4; ++face_slot) {
        const std::int32_t source = cell.facet_sources[static_cast<std::size_t>(face_slot)];
        if (source < 0) continue;
        std::int32_t face[3];
        oriented_facet(local, face_slot, face);
        const SortedFace key = sorted_face(face);
        std::array<std::uint64_t, 3> record_vertices{};
        std::int32_t record_source = kNoConstraint;
        std::int8_t record_orientation = 1;
        if (const auto replacement = replacement_facet_by_face.find(key);
            replacement != replacement_facet_by_face.end()) {
          const TetEditFacet& facet = operation.replacement_facets[replacement->second];
          record_vertices = facet.vertices;
          record_source = facet.source;
          record_orientation = facet.orientation;
        } else if (const auto existing = cache.facet_row_by_face.find(key);
                   existing != cache.facet_row_by_face.end() &&
                   !removed_facets.contains(facets_[existing->second].id)) {
          const TetSnapshotFacet& facet = facets_[existing->second];
          record_vertices = facet.vertices;
          record_source = facet.source;
          record_orientation = facet.orientation;
        } else {
          return false;
        }
        if (record_source != source) return false;
        std::int32_t facet_vertices[3];
        for (int corner = 0; corner < 3; ++corner) {
          facet_vertices[corner] = vertex_slot(record_vertices[static_cast<std::size_t>(corner)]);
          if (!live_vertex(facet_vertices[corner])) return false;
        }
        const int direction =
            (FacetKey::of(face) == FacetKey::of(facet_vertices) ? 1 : -1) * record_orientation;
        const auto expected = cache.source_regions.find({source, direction});
        if (expected == cache.source_regions.end() || expected->second != cell.region) {
          return false;
        }
      }
    }
    for (std::size_t row = 0; row < operation.replacement_facets.size(); ++row) {
      const TetEditFacet& facet = operation.replacement_facets[row];
      const std::array<std::int32_t, 3>& local_face = replacement_facet_slots[row];
      bool adjacent = false;
      for (const TetEditCell& cell : operation.replacement_cells) {
        std::int32_t local[4];
        for (int corner = 0; corner < 4; ++corner) {
          local[corner] = vertex_slot(cell.vertices[static_cast<std::size_t>(corner)]);
          if (local[corner] < 0) return false;
        }
        for (int face_slot = 0; face_slot < 4; ++face_slot) {
          std::int32_t face[3];
          oriented_facet(local, face_slot, face);
          if (sorted_face(face) != sorted_face(local_face.data())) continue;
          adjacent = true;
          const int side =
              (FacetKey::of(face) == FacetKey::of(local_face.data()) ? 1 : -1) *
              facet.orientation;
          const auto found = cache.source_regions.find({facet.source, side});
          if (found == cache.source_regions.end() || found->second != cell.region) {
            return false;
          }
        }
      }
      if (!adjacent) return false;
    }
    return true;
  };
  if (!regions_valid()) return refuse(topology);

  // Replacement cells, in order, then the exact volume. The removed vertices
  // are no longer live here.
  const auto post_live = [&](std::int32_t slot) {
    return live_vertex(slot) && !removed_vertex_ids.contains(vertex_id(slot));
  };
  std::unordered_set<std::uint64_t> replacement_ids;
  std::vector<std::array<std::int32_t, 4>> replacement_slots;
  replacement_slots.reserve(operation.replacement_cells.size());
  for (const TetEditCell& cell : operation.replacement_cells) {
    if (cell.id == std::numeric_limits<std::uint64_t>::max() || used_cell_ids_.contains(cell.id) ||
        !replacement_ids.insert(cell.id).second) {
      return refuse(topology);
    }
    for (std::size_t ancestor = 0; ancestor < cell.ancestor_cell_ids.size(); ++ancestor) {
      const std::uint64_t id = cell.ancestor_cell_ids[ancestor];
      if (!has_id(operation.removed_cell_ids, id) ||
          has_id(std::span(cell.ancestor_cell_ids).first(ancestor), id)) {
        return refuse(topology);
      }
    }
    std::array<std::int32_t, 4> local{};
    for (int corner = 0; corner < 4; ++corner) {
      local[static_cast<std::size_t>(corner)] =
          vertex_slot(cell.vertices[static_cast<std::size_t>(corner)]);
      if (!post_live(local[static_cast<std::size_t>(corner)])) return refuse(topology);
      for (int other = 0; other < corner; ++other) {
        if (local[static_cast<std::size_t>(corner)] == local[static_cast<std::size_t>(other)]) {
          return refuse(topology);
        }
      }
      if (cell.facet_sources[static_cast<std::size_t>(corner)] < kNoConstraint) {
        return refuse(topology);
      }
    }
    if (!(moved == nullptr ? positive(local.data())
                           : positive_with_relocation(local.data(), *moved))) {
      return refuse(TetEditStatus::kInvalidGeometry);
    }
    replacement_slots.push_back(local);
  }
  if (!collapse) {
    Dyadic replacement_volume;
    for (const std::array<std::int32_t, 4>& local : replacement_slots) {
      replacement_volume = replacement_volume + exact_oriented_determinant(*this, local.data());
    }
    if (!(removed_volume - replacement_volume).is_zero()) return refuse(topology);
  }

  // The candidate cells: every retained cell through a vertex the edit touches
  // (in slot order), then the replacement cells. A face with a touched vertex
  // has all its candidate cells among them; every other face, cell and segment
  // is retained unchanged and passes on the certified state.
  std::vector<std::uint8_t> touched(vertex_ids_.size(), 0);
  const auto touch = [&](std::uint64_t id) {
    const std::int32_t slot = vertex_slot(id);
    if (slot >= 0) touched[static_cast<std::size_t>(slot)] = 1;
  };
  for (std::int32_t slot : removed_slots) {
    for (std::int32_t vertex : tet(slot).v) touched[static_cast<std::size_t>(vertex)] = 1;
  }
  for (const std::array<std::int32_t, 4>& local : replacement_slots) {
    for (std::int32_t vertex : local) touched[static_cast<std::size_t>(vertex)] = 1;
  }
  for (const std::array<std::int32_t, 3>& local : replacement_facet_slots) {
    for (std::int32_t vertex : local) touched[static_cast<std::size_t>(vertex)] = 1;
  }
  for (const TetEditSegment& segment : operation.replacement_segments) {
    for (std::uint64_t id : segment.vertices) touch(id);
  }
  for (const TetSnapshotSegment& segment : segments_) {
    if (removed_segments.contains(segment.id)) {
      for (std::uint64_t id : segment.vertices) touch(id);
    }
  }
  const auto touches = [&](std::int32_t vertex) {
    return touched[static_cast<std::size_t>(vertex)] != 0;
  };
  struct FaceEntry {
    FacetKey oriented;
    std::int32_t source;
    std::int32_t region;
    int count;
  };
  std::unordered_map<SortedFace, FaceEntry, SortedFaceHash> candidate_faces;
  std::set<SortedTet> candidate_cells;
  std::unordered_set<std::uint64_t> candidate_edges;
  for (std::size_t row = 0; row < complex_.tets.size(); ++row) {
    const std::int32_t slot = static_cast<std::int32_t>(row);
    if (!complex_.live(slot) || is_ghost(tet(slot)) || removed_slot_set.contains(slot)) continue;
    const Tetrahedron& cell = tet(slot);
    if (!touches(cell.v[0]) && !touches(cell.v[1]) && !touches(cell.v[2]) &&
        !touches(cell.v[3])) {
      continue;
    }
    candidate_cells.insert(sorted_tet(cell.v));
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) candidate_edges.insert(edge_key(cell.v[a], cell.v[b]));
      std::int32_t face[3];
      oriented_facet(cell.v, a, face);
      if (!touches(face[0]) && !touches(face[1]) && !touches(face[2])) continue;
      const auto [entry, inserted] = candidate_faces.emplace(
          sorted_face(face),
          FaceEntry{FacetKey::of(face), complex_.constraint(slot, a), complex_.region(slot), 1});
      if (!inserted) entry->second.count = 2;
    }
  }
  for (std::size_t row = 0; row < replacement_slots.size(); ++row) {
    const std::array<std::int32_t, 4>& local = replacement_slots[row];
    const TetEditCell& cell = operation.replacement_cells[row];
    if (!candidate_cells.insert(sorted_tet(local.data())).second) {
      return refuse(TetEditStatus::kInvalidGeometry);
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        candidate_edges.insert(
            edge_key(local[static_cast<std::size_t>(a)], local[static_cast<std::size_t>(b)]));
      }
      std::int32_t face[3];
      oriented_facet(local.data(), a, face);
      const std::int32_t source = cell.facet_sources[static_cast<std::size_t>(a)];
      const auto [entry, inserted] = candidate_faces.emplace(
          sorted_face(face), FaceEntry{FacetKey::of(face), source, cell.region, 1});
      if (inserted) continue;
      FaceEntry& previous = entry->second;
      if (previous.count != 1 || !(previous.oriented == FacetKey::of(face).reversed()) ||
          previous.source != source ||
          (previous.source == kNoConstraint && previous.region != cell.region)) {
        return refuse(topology);
      }
      previous.count = 2;
    }
  }
  for (const auto& [key, face] : candidate_faces) {
    (void)key;
    if (face.count == 1 && face.source == kNoConstraint) {
      return refuse(TetEditStatus::kRejectedProtected);
    }
  }
  // Constrained candidate faces against the future facets, on every face whose
  // marking or facet the edit may change.
  std::set<SortedFace> changed_faces(retired.begin(), retired.end());
  for (std::int32_t slot : removed_slots) {
    for (int a = 0; a < 4; ++a) {
      std::int32_t face[3];
      oriented_facet(tet(slot).v, a, face);
      changed_faces.insert(sorted_face(face));
    }
  }
  for (const std::array<std::int32_t, 4>& local : replacement_slots) {
    for (int a = 0; a < 4; ++a) {
      std::int32_t face[3];
      oriented_facet(local.data(), a, face);
      changed_faces.insert(sorted_face(face));
    }
  }
  for (const auto& [key, row] : replacement_facet_by_face) {
    (void)row;
    changed_faces.insert(key);
  }
  for (const SortedFace& key : changed_faces) {
    const auto candidate = candidate_faces.find(key);
    const bool is_marked =
        candidate != candidate_faces.end() && candidate->second.source != kNoConstraint;
    bool is_expected = false;
    std::int32_t expected_source = kNoConstraint;
    if (const auto replacement = replacement_facet_by_face.find(key);
        replacement != replacement_facet_by_face.end()) {
      is_expected = true;
      expected_source = operation.replacement_facets[replacement->second].source;
    } else if (const auto existing = cache.facet_row_by_face.find(key);
               existing != cache.facet_row_by_face.end() &&
               !removed_facets.contains(facets_[existing->second].id)) {
      is_expected = true;
      expected_source = facets_[existing->second].source;
    }
    if (is_marked != is_expected ||
        (is_marked && candidate->second.source != expected_source)) {
      return refuse(topology);
    }
  }
  for (const TetSnapshotSegment& segment : future_segments) {
    const std::int32_t a = vertex_slot(segment.vertices[0]);
    const std::int32_t b = vertex_slot(segment.vertices[1]);
    if (!post_live(a) || !post_live(b) || a == b) return refuse(topology);
    if (!touches(a) && !touches(b)) continue;
    if (!candidate_edges.contains(edge_key(a, b))) return refuse(topology);
  }
  return defer();
}

// validate_and_commit's relocation branch reads only the moved vertices, their
// stars and the facets and segments through them. It writes the new carriers
// and witnesses before the source checks; RelocationOverlay presents exactly
// those records to the same checks on the unchanged state.
bool TetMesh::probe_relocation_edit(const TetEdit& operation, TetEditStatus& status) const {
  if (operation.kind != TetEditKind::kRelocate || periodic_identity_.bound ||
      !periodic_orbits_.empty() || !quotient_cells_.empty() ||
      !periodic_entity_maps_.empty() || !operation.periodic_members.empty() ||
      !operation.removed_periodic_orbit_ids.empty() ||
      !operation.replacement_periodic_orbits.empty() ||
      !operation.removed_quotient_cell_ids.empty() ||
      !operation.replacement_quotient_cells.empty() ||
      !operation.replacement_periodic_entity_maps.empty() ||
      generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  const TetImplicitSource* implicit = implicit_source_.get();
  const std::int64_t transversality = implicit == nullptr ? 0 : implicit->transversality_refusals;
  const std::int64_t distance = implicit == nullptr ? 0 : implicit->distance_refusals;
  const bool inconclusive = implicit != nullptr && implicit->last_inconclusive;
  const auto refuse = [&](TetEditStatus refusal) {
    status = refusal;
    return true;
  };
  const auto defer = [&]() {
    if (implicit != nullptr) {
      implicit->transversality_refusals = transversality;
      implicit->distance_refusals = distance;
      implicit->last_inconclusive = inconclusive;
    }
    return false;
  };
  constexpr TetEditStatus topology = TetEditStatus::kRejectedTopology;

  // validate_periodic_closure on a state without periodic records.
  for (const TetEditVertex& vertex : operation.changed_vertices) {
    for (const TetVertexAncestry& ancestor : vertex.ancestry) {
      if (ancestor.source_winding[0] != 0 || ancestor.source_winding[1] != 0 ||
          ancestor.source_winding[2] != 0) {
        return refuse(topology);
      }
    }
  }
  if (operation.changed_vertices.empty() || !operation.replacement_cells.empty() ||
      !operation.replacement_facets.empty() || !operation.replacement_segments.empty() ||
      !operation.removed_vertex_ids.empty() || !operation.removed_facet_ids.empty() ||
      !operation.removed_segment_ids.empty() || !operation.collapse_pairs.empty() ||
      !operation.split_edges.empty()) {
    return refuse(topology);
  }
  std::unordered_set<std::uint64_t> removed_cell_ids;
  std::unordered_set<std::int32_t> removed_slot_set;
  std::vector<std::int32_t> removed_slots;
  for (std::uint64_t id : operation.removed_cell_ids) {
    const std::int32_t slot = cell_slot(id);
    if (slot < 0 || is_ghost(tet(slot)) || !removed_cell_ids.insert(id).second) {
      return refuse(topology);
    }
    removed_slots.push_back(slot);
    removed_slot_set.insert(slot);
  }
  std::unordered_set<std::uint64_t> changed_vertex_ids;
  std::vector<std::int32_t> moved_vertex_slots;
  for (const TetEditVertex& vertex : operation.changed_vertices) {
    if (vertex.id == std::numeric_limits<std::uint64_t>::max()) {
      return refuse(TetEditStatus::kIndexOverflow);
    }
    if (!changed_vertex_ids.insert(vertex.id).second ||
        (used_vertex_ids_.contains(vertex.id) && vertex_slot(vertex.id) < 0)) {
      return refuse(topology);
    }
    const std::int32_t slot = vertex_slot(vertex.id);
    if (slot < 0) return refuse(topology);
    if (!vertex.ancestry.empty()) return refuse(topology);
    if (protected_vertex(slot) || protection_radius(slot) > 0.0 ||
        vertex.protected_vertex != protected_vertex(slot) ||
        vertex.protection_radius != protection_radius(slot) ||
        vertex.witness.stratum != witness(slot).stratum) {
      return refuse(TetEditStatus::kRejectedProtected);
    }
    moved_vertex_slots.push_back(slot);
    TetEditVertex checked_vertex = vertex;
    const std::size_t index = static_cast<std::size_t>(slot);
    if (!checked_vertex.edge_witness_supplied && !checked_vertex.has_edge_parameter &&
        vertex_has_edge_parameter_[index] != 0) {
      checked_vertex.has_edge_parameter = true;
      checked_vertex.edge_endpoints = vertex_edge_endpoints_[index];
      checked_vertex.edge_parameter = vertex_edge_parameters_[index];
    }
    if (!std::isfinite(vertex.protection_radius) || vertex.protection_radius < 0.0 ||
        !source_carrier_valid(checked_vertex)) {
      return refuse(TetEditStatus::kInvalidGeometry);
    }
    if (vertex.protection_radius > 0.0 && !vertex.protected_vertex) {
      return refuse(TetEditStatus::kRejectedProtected);
    }
    for (std::size_t other = 0; other < vertex_ids_.size(); ++other) {
      if (!vertex_live_[other] || vertex_ids_[other] == vertex.id) continue;
      const double* point = points_.data() + 3 * other;
      if (point[0] == vertex.carrier[0] && point[1] == vertex.carrier[1] &&
          point[2] == vertex.carrier[2]) {
        return refuse(TetEditStatus::kInvalidGeometry);
      }
    }
  }
  if (removed_slots.empty()) return refuse(topology);
  for (std::int32_t moved : moved_vertex_slots) {
    for (std::size_t row = 0; row < complex_.tets.size(); ++row) {
      const std::int32_t cell = static_cast<std::int32_t>(row);
      if (!complex_.live(cell) || is_ghost(tet(cell)) ||
          phx::mc::vertex_slot(tet(cell), moved) < 0) {
        continue;
      }
      if (!removed_slot_set.contains(cell)) return refuse(topology);
    }
  }
  const RelocationOverlay overlay{operation.changed_vertices};
  const auto moved = [&](std::uint64_t id) { return changed_vertex_ids.contains(id); };
  for (const TetSnapshotFacet& facet : facets_) {
    if (std::none_of(facet.vertices.begin(), facet.vertices.end(), moved)) continue;
    TetEditFacet check;
    check.id = facet.id;
    check.vertices = facet.vertices;
    check.source = facet.source;
    check.orientation = facet.orientation;
    if (!source_face_valid(check, &overlay)) return refuse(source_face_refusal_status());
  }
  for (const TetSnapshotSegment& segment : segments_) {
    if (std::none_of(segment.vertices.begin(), segment.vertices.end(), moved)) continue;
    TetEditSegment check;
    check.id = segment.id;
    check.vertices = segment.vertices;
    check.source = segment.source;
    if (!source_segment_valid(check, &overlay)) return refuse(TetEditStatus::kInvalidGeometry);
  }
  if (implicit_source_ &&
      !implicit_facets_embedded(facets_, implicit_source_->groups, nullptr,
                                &changed_vertex_ids, &overlay)) {
    return refuse(TetEditStatus::kInvalidGeometry);
  }
  for (std::int32_t slot : removed_slots) {
    if (!positive_with_relocation(tet(slot).v, overlay)) {
      return refuse(TetEditStatus::kInvalidGeometry);
    }
  }
  return defer();
}

}  // namespace phx::mc
