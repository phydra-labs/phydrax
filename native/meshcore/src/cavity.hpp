//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Tetrahedral adjacency and validated cavity replacement. Every live finite
// tetrahedron is positively oriented; ghost tetrahedra close the unbounded
// side. Slot numbers are local reusable indices, not persistent identities.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "phydrax_meshcore.h"

namespace phx::mc {

inline constexpr int32_t kGhostVertex = -1;
inline constexpr int32_t kDeadVertex = -2;
inline constexpr int32_t kNoConstraint = -1;
inline constexpr int32_t kNoRegion = -1;
inline constexpr int64_t kMaxTetrahedronSlots = std::numeric_limits<int32_t>::max();

// One cell with the data every walk and cavity reads from it: vertices,
// neighbors, the visit stamp/state of its current owner, which faces carry a
// constraint label and its region (TetrahedralComplex keeps the labels'
// IDs). 48 bytes, so a cell rarely straddles cache lines.
struct alignas(16) Tetrahedron {
  int32_t v[4];
  int32_t n[4];
  std::uint32_t stamp = 0;
  std::uint8_t state = 0;
  // Bit k is set when face k carries a constraint label.
  std::uint8_t constrained = 0;
  int32_t region = kNoRegion;
};
static_assert(sizeof(Tetrahedron) == 48);

// Hint that `cell` is read soon (a walk or cavity about to visit it).
inline void prefetch_cell([[maybe_unused]] const Tetrahedron* cell) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(cell);
#endif
}

inline bool is_ghost(const Tetrahedron& tet) {
  return tet.v[0] == kGhostVertex || tet.v[1] == kGhostVertex ||
         tet.v[2] == kGhostVertex || tet.v[3] == kGhostVertex;
}

inline int vertex_slot(const Tetrahedron& tet, int32_t vertex) {
  for (int k = 0; k < 4; ++k) {
    if (tet.v[k] == vertex) return k;
  }
  return -1;
}

inline int neighbor_slot(const Tetrahedron& tet, int32_t neighbor) {
  for (int k = 0; k < 4; ++k) {
    if (tet.n[k] == neighbor) return k;
  }
  return -1;
}

// Facet opposite v[k], oriented so that the omitted vertex is on its positive side.
inline void oriented_facet(const int32_t* v, int k, int32_t* facet) {
  static constexpr int order[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  for (int r = 0; r < 3; ++r) facet[r] = v[order[k][r]];
}

struct FacetKey {
  int32_t a;
  int32_t b;
  int32_t c;

  static FacetKey of(const int32_t* facet) {
    int first = 0;
    for (int i = 1; i < 3; ++i) {
      if (facet[i] < facet[first]) first = i;
    }
    return {facet[first], facet[(first + 1) % 3], facet[(first + 2) % 3]};
  }
  FacetKey reversed() const { return {a, c, b}; }
  bool operator==(const FacetKey&) const = default;
  bool operator<(const FacetKey& other) const {
    return std::array<int32_t, 3>{a, b, c} <
           std::array<int32_t, 3>{other.a, other.b, other.c};
  }
};

template <class T, class Allocator>
void reserve_for(std::vector<T, Allocator>& values, std::size_t required) {
  if (values.capacity() >= required) return;
  const std::size_t capacity = values.capacity();
  const std::size_t grown = capacity > values.max_size() / 2 ? required : capacity * 2;
  values.reserve(std::max(required, grown));
}

// One cell replaced by a committed cavity edit: its slot and vertices, as
// removed (before the edit) or created (after it).
struct CellEvent {
  int32_t slot;
  bool created;
  std::array<int32_t, 4> v;
};

class TetrahedralComplex {
 public:
  std::vector<Tetrahedron> tets;
  std::vector<int32_t> free_slots;
  int64_t finite_count = 0;
  // Optional journal of committed cavity edits, in commit order: every
  // removed cell, then every created cell, of each edit. Bulk construction
  // and vertex renumbering are not journaled.
  std::vector<CellEvent>* journal = nullptr;

  bool live(int32_t t) const {
    return t >= 0 && static_cast<std::size_t>(t) < tets.size() &&
           tets[static_cast<std::size_t>(t)].v[0] != kDeadVertex;
  }
  int64_t live_count() const {
    return static_cast<int64_t>(tets.size() - free_slots.size());
  }
  bool labeled() const { return labeled_; }

  void enable_labels() {
    if (labeled_) return;
    reserve_for(constraints_, 4 * tets.size());
    constraints_.assign(4 * tets.size(), kNoConstraint);
    for (Tetrahedron& tet : tets) {
      tet.constrained = 0;
      tet.region = kNoRegion;
    }
    labeled_ = true;
  }
  // The cell's own flags decide an unconstrained face (every face of an
  // unlabeled complex), so only a constrained face reads the label bank,
  // whose entries are meaningful only where a flag is set.
  int32_t constraint(int32_t t, int k) const {
    return (tets[static_cast<std::size_t>(t)].constrained >> k) & 1U
               ? constraints_[4 * static_cast<std::size_t>(t) + k]
               : kNoConstraint;
  }
  int32_t region(int32_t t) const { return tets[static_cast<std::size_t>(t)].region; }
  void set_facet_constraint(int32_t t, int k, int32_t id) {
    const int32_t other = tets[static_cast<std::size_t>(t)].n[k];
    const int back = neighbor_slot(tets[static_cast<std::size_t>(other)], t);
    label(t, k, id);
    label(other, back, id);
  }
  // Labels face k of cell t on its side only. For a commit that labels both
  // sides of every face it creates, and keeps on each outer face the label
  // the cell beyond already holds.
  void set_cell_face_constraint(int32_t t, int k, int32_t id) { label(t, k, id); }
  void set_region(int32_t t, int32_t value) {
    tets[static_cast<std::size_t>(t)].region = value;
  }
  void clear_regions() {
    for (Tetrahedron& tet : tets) tet.region = kNoRegion;
  }

  int32_t allocate() {
    if (!free_slots.empty()) {
      const int32_t id = free_slots.back();
      free_slots.pop_back();
      return id;
    }
    if (tets.size() >= static_cast<std::size_t>(kMaxTetrahedronSlots)) return -1;
    reserve(1, 0);
    tets.push_back(Tetrahedron{{kDeadVertex, kDeadVertex, kDeadVertex, kDeadVertex},
                               {-1, -1, -1, -1}, 0, 0});
    if (labeled_) constraints_.insert(constraints_.end(), 4, kNoConstraint);
    return static_cast<int32_t>(tets.size() - 1);
  }
  void release(int32_t t) {
    reserve_for(free_slots, free_slots.size() + 1);
    Tetrahedron& dead = tets[static_cast<std::size_t>(t)];
    dead = Tetrahedron{{kDeadVertex, kDeadVertex, kDeadVertex, kDeadVertex},
                       {-1, -1, -1, -1}, 0, 0};
    free_slots.push_back(t);
  }
  void reserve(std::size_t fresh, std::size_t freed) {
    reserve_for(tets, tets.size() + fresh);
    reserve_for(free_slots, free_slots.size() + freed);
    if (labeled_) reserve_for(constraints_, constraints_.size() + 4 * fresh);
  }

  bool audit() const {
    int64_t finite = 0;
    std::vector<int32_t> free = free_slots;
    std::sort(free.begin(), free.end());
    if (std::adjacent_find(free.begin(), free.end()) != free.end()) return false;
    for (int32_t id : free) {
      if (id < 0 || static_cast<std::size_t>(id) >= tets.size() || live(id)) return false;
    }
    for (std::size_t t = 0; t < tets.size(); ++t) {
      const Tetrahedron& tet = tets[t];
      if (tet.v[0] == kDeadVertex) continue;
      finite += is_ghost(tet) ? 0 : 1;
      for (int k = 0; k < 4; ++k) {
        const int32_t other = tet.n[k];
        if (!live(other)) return false;
        const Tetrahedron& neighbor = tets[static_cast<std::size_t>(other)];
        const int back = neighbor_slot(neighbor, static_cast<int32_t>(t));
        if (back < 0) return false;
        int32_t facet[3], twin[3];
        oriented_facet(tet.v, k, facet);
        oriented_facet(neighbor.v, back, twin);
        if (!(FacetKey::of(facet) == FacetKey::of(twin).reversed()) ||
            constraint(static_cast<int32_t>(t), k) != constraint(other, back)) return false;
        if (((tet.constrained >> k) & 1U) != 0 &&
            (!labeled_ || constraints_[4 * t + static_cast<std::size_t>(k)] == kNoConstraint)) {
          return false;
        }
      }
    }
    return finite == finite_count;
  }

 private:
  void label(int32_t t, int k, int32_t id) {
    Tetrahedron& cell = tets[static_cast<std::size_t>(t)];
    const auto bit = static_cast<std::uint8_t>(1U << k);
    if (id == kNoConstraint) {
      cell.constrained = static_cast<std::uint8_t>(cell.constrained & ~bit);
      return;
    }
    cell.constrained = static_cast<std::uint8_t>(cell.constrained | bit);
    constraints_[4 * static_cast<std::size_t>(t) + k] = id;
  }

  bool labeled_ = false;
  std::vector<int32_t> constraints_;
};

enum class EditStatus : std::uint8_t {
  kOk,
  kSlotLimit,
  kNotLive,
  kDuplicate,
  kNonManifold,
  kBoundaryMismatch,
  kInverted,
  kConstraintViolation,
  kInternal,
};

inline int32_t edit_status_code(EditStatus status) {
  switch (status) {
    case EditStatus::kOk: return PHX_MC_OK;
    case EditStatus::kSlotLimit: return PHX_MC_INDEX_RANGE_ERROR;
    case EditStatus::kConstraintViolation: return PHX_MC_CONSTRAINT_INTERSECTION;
    case EditStatus::kNotLive:
    case EditStatus::kDuplicate:
    case EditStatus::kNonManifold:
    case EditStatus::kBoundaryMismatch:
    case EditStatus::kInverted: return PHX_MC_INVALID_INPUT;
    case EditStatus::kInternal: return PHX_MC_INTERNAL_ERROR;
  }
  return PHX_MC_INTERNAL_ERROR;
}

struct BoundaryFacet {
  int32_t tet;
  int32_t slot;
  int32_t outside;
  int32_t back;
};

class CavityEdit {
 public:
  explicit CavityEdit(TetrahedralComplex& complex) : complex_(complex) {}

  void begin() {
    ++staging_revision_;
    removed_.clear();
    boundary_.clear();
    created_.clear();
    created_constraints_.clear();
    created_regions_.clear();
    owners_.clear();
    preserved_.clear();
    staged_ = false;
    cone_center_ = kNoCone;
  }
  EditStatus remove(int32_t t) {
    ++staging_revision_;
    if (!complex_.live(t)) return EditStatus::kNotLive;
    removed_.push_back(t);
    return EditStatus::kOk;
  }
  EditStatus add(const int32_t* v, const int32_t* constraints, int32_t region) {
    ++staging_revision_;
    created_.push_back(Tetrahedron{{v[0], v[1], v[2], v[3]}, {-1, -1, -1, -1}, 0, 0});
    for (int k = 0; k < 4; ++k) {
      created_constraints_.push_back(constraints == nullptr ? kNoConstraint : constraints[k]);
    }
    created_regions_.push_back(region);
    return EditStatus::kOk;
  }
  void preserve(int32_t a, int32_t b, int32_t c, int32_t id) {
    const int32_t face[3] = {a, b, c};
    preserved_.push_back({FacetKey::of(face), id});
  }

  template <class Positive>
  EditStatus validate(const Positive& positive) {
    ++staging_revision_;
    staged_ = false;
    cone_center_ = kNoCone;
    EditStatus status = collect_boundary();
    if (status == EditStatus::kOk) status = link_proposal(positive);
    staged_ = status == EditStatus::kOk;
    return status;
  }

  EditStatus stage_cone(int32_t center, std::span<const int32_t> cavity,
                        std::span<const BoundaryFacet> boundary) {
    ++staging_revision_;
    staged_ = false;
    removed_.assign(cavity.begin(), cavity.end());
    boundary_.assign(boundary.begin(), boundary.end());
    created_.clear();
    created_constraints_.clear();
    created_regions_.clear();
    const bool labeled = complex_.labeled();
    for (const BoundaryFacet& face : boundary) {
      Tetrahedron tet = complex_.tets[static_cast<std::size_t>(face.tet)];
      tet.v[face.slot] = center;
      std::fill(tet.n, tet.n + 4, -1);
      tet.n[face.slot] = kExternal;
      tet.stamp = 0;
      tet.state = 0;
      tet.constrained = 0;
      tet.region = kNoRegion;
      created_.push_back(tet);
      if (labeled) {
        for (int k = 0; k < 4; ++k) {
          created_constraints_.push_back(k == face.slot
              ? complex_.constraint(face.tet, face.slot) : kNoConstraint);
        }
        created_regions_.push_back(complex_.region(face.tet));
      }
    }
    cone_center_ = center;
    staged_ = true;
    return EditStatus::kOk;
  }

  std::span<const int32_t> removed() const { return removed_; }
  std::span<const Tetrahedron> proposal() const { return created_; }
  int32_t proposal_constraint(std::size_t cell, int slot) const {
    return created_constraints_.empty() ? kNoConstraint : created_constraints_[4 * cell + slot];
  }

  // All allocations precede the first topology write. The no-throw mutation
  // block uses only the banks reserved here and staged edit storage. A cone
  // is linked on the staged cells first, so a boundary that is not a closed
  // manifold surface is refused (kNonManifold) with the complex unchanged.
  EditStatus commit() {
    ++staging_revision_;
    if (!staged_) return EditStatus::kBoundaryMismatch;
    if (cone_center_ != kNoCone && !link_cone()) {
      rollback();
      return EditStatus::kNonManifold;
    }
    int64_t removed_finite = 0, created_finite = 0;
    for (int32_t t : removed_) {
      removed_finite += is_ghost(complex_.tets[static_cast<std::size_t>(t)]) ? 0 : 1;
    }
    for (const Tetrahedron& tet : created_) created_finite += is_ghost(tet) ? 0 : 1;
    const std::size_t extra = created_.size() > removed_.size()
        ? created_.size() - removed_.size() : 0;
    const std::size_t fresh = extra - std::min(extra, complex_.free_slots.size());
    if (fresh > static_cast<std::size_t>(kMaxTetrahedronSlots) - complex_.tets.size()) {
      return EditStatus::kSlotLimit;
    }
    const std::size_t freed = removed_.size() > created_.size()
        ? removed_.size() - created_.size() : 0;
    complex_.reserve(fresh, freed);
    ids_.clear();
    ids_.reserve(created_.size() + 1);
    std::vector<CellEvent>* journal = complex_.journal;
    if (journal != nullptr) {
      reserve_for(*journal, journal->size() + removed_.size() + created_.size());
      for (int32_t t : removed_) {
        const int32_t* v = complex_.tets[static_cast<std::size_t>(t)].v;
        journal->push_back({t, false, {v[0], v[1], v[2], v[3]}});
      }
    }

    staged_ = false;
    apply();
    if (journal != nullptr) {
      for (int32_t id : ids_) {
        const int32_t* v = complex_.tets[static_cast<std::size_t>(id)].v;
        journal->push_back({id, true, {v[0], v[1], v[2], v[3]}});
      }
    }
    complex_.finite_count += created_finite - removed_finite;
    cone_center_ = kNoCone;
    return EditStatus::kOk;
  }
  void rollback() {
    ++staging_revision_;
    staged_ = false;
    cone_center_ = kNoCone;
  }
  std::span<const int32_t> created() const { return ids_; }
  uint64_t staging_revision() const noexcept { return staging_revision_; }

 private:
  static constexpr int32_t kNoCone = -3;
  static constexpr int32_t kExternal = std::numeric_limits<int32_t>::min();
  struct Preserved { FacetKey key; int32_t id; };
  struct FacetSlot { FacetKey key; int32_t owner; };
  struct EdgeRecord {
    std::uint64_t key;
    int32_t owner;
    bool operator<(const EdgeRecord& other) const { return key < other.key; }
  };

  EditStatus collect_boundary() {
    sorted_removed_.assign(removed_.begin(), removed_.end());
    std::sort(sorted_removed_.begin(), sorted_removed_.end());
    if (std::adjacent_find(sorted_removed_.begin(), sorted_removed_.end()) != sorted_removed_.end()) {
      return EditStatus::kDuplicate;
    }
    boundary_.clear();
    inner_constraints_.clear();
    for (int32_t t : removed_) {
      const Tetrahedron& tet = complex_.tets[static_cast<std::size_t>(t)];
      for (int k = 0; k < 4; ++k) {
        const int32_t other = tet.n[k];
        if (!complex_.live(other)) return EditStatus::kNonManifold;
        const int back = neighbor_slot(complex_.tets[static_cast<std::size_t>(other)], t);
        if (back < 0) return EditStatus::kNonManifold;
        if (!std::binary_search(sorted_removed_.begin(), sorted_removed_.end(), other)) {
          boundary_.push_back({t, k, other, back});
        } else if (complex_.constraint(t, k) != kNoConstraint) {
          int32_t face[3];
          oriented_facet(tet.v, k, face);
          inner_constraints_.push_back({FacetKey::of(face), complex_.constraint(t, k)});
        }
      }
    }
    return EditStatus::kOk;
  }

  int32_t find_facet(const FacetKey& key) const {
    const auto found = std::lower_bound(facets_.begin(), facets_.end(), key,
        [](const FacetSlot& slot, const FacetKey& wanted) { return slot.key < wanted; });
    return found != facets_.end() && found->key == key ? found->owner : -1;
  }
  int32_t& proposed_constraint(int32_t owner) {
    return created_constraints_[static_cast<std::size_t>(owner)];
  }
  static bool well_formed(const int32_t* v) {
    int ghosts = 0;
    for (int i = 0; i < 4; ++i) {
      if (v[i] < kGhostVertex) return false;
      ghosts += v[i] == kGhostVertex ? 1 : 0;
      for (int j = i + 1; j < 4; ++j) if (v[i] == v[j]) return false;
    }
    return ghosts <= 1;
  }

  template <class Positive>
  EditStatus link_proposal(const Positive& positive) {
    for (const Tetrahedron& tet : created_) {
      if (!well_formed(tet.v) || (!is_ghost(tet) && !positive(tet.v))) {
        return EditStatus::kInverted;
      }
    }
    facets_.clear();
    facets_.reserve(4 * created_.size());
    for (std::size_t i = 0; i < created_.size(); ++i) {
      for (int k = 0; k < 4; ++k) {
        int32_t face[3];
        oriented_facet(created_[i].v, k, face);
        facets_.push_back({FacetKey::of(face), static_cast<int32_t>(4 * i) + k});
      }
    }
    std::sort(facets_.begin(), facets_.end(),
        [](const FacetSlot& a, const FacetSlot& b) { return a.key < b.key; });
    if (std::adjacent_find(facets_.begin(), facets_.end(),
        [](const FacetSlot& a, const FacetSlot& b) { return a.key == b.key; }) != facets_.end()) {
      return EditStatus::kDuplicate;
    }
    owners_.clear();
    for (const BoundaryFacet& face : boundary_) {
      int32_t triple[3];
      oriented_facet(complex_.tets[static_cast<std::size_t>(face.tet)].v, face.slot, triple);
      const int32_t owner = find_facet(FacetKey::of(triple));
      if (owner < 0) return EditStatus::kBoundaryMismatch;
      Tetrahedron& tet = created_[static_cast<std::size_t>(owner / 4)];
      if (tet.n[owner % 4] != -1) return EditStatus::kBoundaryMismatch;
      const int32_t inherited = complex_.constraint(face.tet, face.slot);
      int32_t& proposed = proposed_constraint(owner);
      if (proposed != kNoConstraint && proposed != inherited) {
        return EditStatus::kConstraintViolation;
      }
      proposed = inherited;
      tet.n[owner % 4] = kExternal;
      owners_.push_back(owner);
    }
    for (std::size_t i = 0; i < created_.size(); ++i) {
      for (int k = 0; k < 4; ++k) {
        if (created_[i].n[k] != -1) continue;
        int32_t face[3];
        oriented_facet(created_[i].v, k, face);
        const int32_t twin = find_facet(FacetKey::of(face).reversed());
        if (twin < 0 || created_[static_cast<std::size_t>(twin / 4)].n[twin % 4] != -1) {
          return EditStatus::kBoundaryMismatch;
        }
        if (proposed_constraint(static_cast<int32_t>(4 * i) + k) != proposed_constraint(twin)) {
          return EditStatus::kConstraintViolation;
        }
        created_[i].n[k] = twin / 4;
        created_[static_cast<std::size_t>(twin / 4)].n[twin % 4] = static_cast<int32_t>(i);
      }
    }
    inner_constraints_.insert(inner_constraints_.end(), preserved_.begin(), preserved_.end());
    for (const Preserved& required : inner_constraints_) {
      int32_t owner = find_facet(required.key);
      if (owner < 0) owner = find_facet(required.key.reversed());
      if (owner < 0 || proposed_constraint(owner) != required.id) {
        return EditStatus::kConstraintViolation;
      }
    }
    return EditStatus::kOk;
  }

  // Cone cell i stands on boundary facet i, whose oriented triangle (the
  // facet opposite the center) runs along edge j of the cell, the edge
  // opposite facet corner j, as (corner j + 1, corner j + 2); the cell across
  // that edge runs along it the other way.
  static constexpr int kConeFacet[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  // Cones with more boundary vertices pair their edges by sorting.
  static constexpr std::uint32_t kDenseConeVertices = 512;

  // Links the staged cone cells across the edges of the cavity boundary
  // (their face opposite the center stays kExternal). Boundary vertices get
  // dense local numbers, so every directed edge owns one entry of a square
  // table: the first pass records each edge's owner (4 cell + corner), the
  // second gives each cell the owner of its reversed edge, without probing
  // or branching on which cell of a pair arrives first. The second pass
  // also proves the pairing a bijection (each entry still holds its own
  // edge; the reversed entry names an edge of this cone that is the
  // reverse), so the table is never cleared and a boundary that is not a
  // closed manifold surface is refused. A closed connected surface has
  // V - E + F <= 2, so at most count / 2 + 2 vertices.
  bool link_cone() {
    const std::size_t count = created_.size();
    if (count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max() / 4)) return false;
    if (++link_epoch_ == 0) {
      std::fill(vertex_tags_.begin(), vertex_tags_.end(), 0);
      link_epoch_ = 1;
    }
    cone_locals_.resize(3 * count);
    std::uint32_t vertices = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const int* facet = kConeFacet[boundary_[i].slot];
      for (int j = 0; j < 3; ++j) {
        // Vertices are at least kGhostVertex = -1.
        const auto index = static_cast<std::size_t>(created_[i].v[facet[j]] + 1);
        if (index >= vertex_tags_.size()) {
          vertex_tags_.resize(std::max(index + 1, 2 * vertex_tags_.size()), 0);
        }
        // A select, not a branch: first and repeated visits of a vertex
        // interleave unpredictably.
        std::uint64_t& tag = vertex_tags_[index];
        const bool fresh = (tag >> 32) != link_epoch_;
        tag = fresh ? (static_cast<std::uint64_t>(link_epoch_) << 32) | vertices : tag;
        vertices += fresh ? 1U : 0U;
        cone_locals_[3 * i + static_cast<std::size_t>(j)] = static_cast<std::uint32_t>(tag);
      }
    }
    if (vertices > count / 2 + 2) return false;
    return vertices <= kDenseConeVertices ? link_cone_dense(vertices) : link_cone_sorted();
  }
  std::uint32_t cone_corner(std::size_t owner, int offset) const {
    const std::size_t cell = owner >> 2;
    const int corner = static_cast<int>(owner & 3U) + offset;
    return cone_locals_[3 * cell + static_cast<std::size_t>(corner >= 3 ? corner - 3 : corner)];
  }
  bool link_cone_dense(std::uint32_t vertices) {
    std::size_t dim = 1;
    while (dim < vertices) dim <<= 1;
    if (edge_owners_.size() < dim * dim) edge_owners_.resize(dim * dim, -1);
    const std::size_t count = created_.size();
    for (std::size_t i = 0; i < count; ++i) {
      for (int j = 0; j < 3; ++j) {
        const std::size_t owner = 4 * i + static_cast<std::size_t>(j);
        edge_owners_[cone_corner(owner, 1) * dim + cone_corner(owner, 2)] =
            static_cast<int32_t>(owner);
      }
    }
    for (std::size_t i = 0; i < count; ++i) {
      const int* facet = kConeFacet[boundary_[i].slot];
      for (int j = 0; j < 3; ++j) {
        const std::size_t owner = 4 * i + static_cast<std::size_t>(j);
        const std::uint32_t from = cone_corner(owner, 1), to = cone_corner(owner, 2);
        const int32_t twin = edge_owners_[to * dim + from];
        if (edge_owners_[from * dim + to] != static_cast<int32_t>(owner) || twin < 0 ||
            static_cast<std::size_t>(twin) >= 4 * count || (twin & 3) == 3 ||
            cone_corner(static_cast<std::size_t>(twin), 1) != to ||
            cone_corner(static_cast<std::size_t>(twin), 2) != from) {
          return false;
        }
        created_[i].n[facet[j]] = twin >> 2;
      }
    }
    return true;
  }
  bool link_cone_sorted() {
    const std::size_t count = created_.size();
    const auto key = [](std::uint32_t from, std::uint32_t to) {
      return (static_cast<std::uint64_t>(from) << 32) | to;
    };
    edge_records_.clear();
    edge_records_.reserve(3 * count);
    for (std::size_t owner = 0; owner < 4 * count; ++owner) {
      if ((owner & 3U) == 3U) continue;
      edge_records_.push_back(
          {key(cone_corner(owner, 1), cone_corner(owner, 2)), static_cast<int32_t>(owner)});
    }
    std::sort(edge_records_.begin(), edge_records_.end());
    for (std::size_t r = 1; r < edge_records_.size(); ++r) {
      if (edge_records_[r].key == edge_records_[r - 1].key) return false;
    }
    for (const EdgeRecord& record : edge_records_) {
      const auto owner = static_cast<std::size_t>(record.owner);
      const EdgeRecord reversed{key(cone_corner(owner, 2), cone_corner(owner, 1)), 0};
      const auto twin = std::lower_bound(edge_records_.begin(), edge_records_.end(), reversed);
      if (twin == edge_records_.end() || twin->key != reversed.key) return false;
      created_[owner >> 2].n[kConeFacet[boundary_[owner >> 2].slot][owner & 3U]] =
          twin->owner >> 2;
    }
    return true;
  }
  void apply() {
    for (std::size_t i = 0; i < created_.size(); ++i) {
      ids_.push_back(i < removed_.size() ? removed_[i] : complex_.allocate());
    }
    for (std::size_t i = created_.size(); i < removed_.size(); ++i) complex_.release(removed_[i]);
    // Staged neighbors are created indices or kExternal (the outer face,
    // linked below); a sentinel slot past the created ones maps kExternal
    // to itself, so the remap selects rather than branches.
    const auto sentinel = static_cast<int32_t>(created_.size());
    ids_.push_back(kExternal);
    for (std::size_t i = 0; i < created_.size(); ++i) {
      Tetrahedron tet = created_[i];
      for (int k = 0; k < 4; ++k) {
        const int32_t staged = tet.n[k] == kExternal ? sentinel : tet.n[k];
        tet.n[k] = ids_[static_cast<std::size_t>(staged)];
      }
      complex_.tets[static_cast<std::size_t>(ids_[i])] = tet;
      if (complex_.labeled()) complex_.set_region(ids_[i], created_regions_[i]);
    }
    ids_.pop_back();
    const bool cone = cone_center_ != kNoCone;
    for (std::size_t f = 0; f < boundary_.size(); ++f) {
      const BoundaryFacet& face = boundary_[f];
      const int32_t owner = cone ? static_cast<int32_t>(4 * f) + face.slot : owners_[f];
      const int32_t id = ids_[static_cast<std::size_t>(owner / 4)];
      complex_.tets[static_cast<std::size_t>(id)].n[owner % 4] = face.outside;
      complex_.tets[static_cast<std::size_t>(face.outside)].n[face.back] = id;
    }
    if (complex_.labeled()) {
      for (std::size_t i = 0; i < created_.size(); ++i) {
        for (int k = 0; k < 4; ++k) {
          complex_.set_facet_constraint(ids_[i], k, created_constraints_[4 * i + k]);
        }
      }
    }
  }

  TetrahedralComplex& complex_;
  bool staged_ = false;
  std::uint64_t staging_revision_ = 0;
  int32_t cone_center_ = kNoCone;
  std::vector<int32_t> removed_, sorted_removed_;
  std::vector<BoundaryFacet> boundary_;
  std::vector<Tetrahedron> created_;
  std::vector<int32_t> created_constraints_, created_regions_, owners_, ids_;
  std::vector<Preserved> preserved_, inner_constraints_;
  std::vector<FacetSlot> facets_;
  std::vector<std::uint64_t> vertex_tags_;
  std::vector<std::uint32_t> cone_locals_;
  std::vector<int32_t> edge_owners_;
  std::vector<EdgeRecord> edge_records_;
  std::uint32_t link_epoch_ = 0;
};

// Quotient-cell transaction uses already-owned identity-aware cells and a
// caller-provided no-allocation publication action.
template <class Cell>
class QuotientCavityEdit final {
 public:
  explicit QuotientCavityEdit(std::vector<Cell>& cells) : cells_(cells) {}

  template <class Validate>
  EditStatus prepare(std::span<const int32_t> removed, std::span<const Cell> proposal,
                     int64_t live_count, const Validate& validate) {
    staged_ = false;
    removed_.assign(removed.begin(), removed.end());
    proposal_.assign(proposal.begin(), proposal.end());
    std::vector<int32_t> sorted(removed_.begin(), removed_.end());
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return EditStatus::kDuplicate;
    for (int32_t id : removed_) {
      if (id < 0 || static_cast<std::size_t>(id) >= cells_.size() || !cells_[id].live) {
        return EditStatus::kNotLive;
      }
    }
    if (live_count - static_cast<int64_t>(removed_.size()) +
            static_cast<int64_t>(proposal_.size()) < 0) return EditStatus::kBoundaryMismatch;
    if (!validate(removed_, proposal_)) return EditStatus::kBoundaryMismatch;
    ids_.clear();
    const std::size_t reused = std::min(removed_.size(), proposal_.size());
    for (std::size_t i = 0; i < reused; ++i) ids_.push_back(removed_[i]);
    const std::size_t fresh = proposal_.size() - reused;
    if (fresh > static_cast<std::size_t>(kMaxTetrahedronSlots) - cells_.size()) {
      return EditStatus::kSlotLimit;
    }
    for (std::size_t i = reused; i < proposal_.size(); ++i) {
      ids_.push_back(static_cast<int32_t>(cells_.size() + i - reused));
    }
    reserve_for(cells_, cells_.size() + fresh);
    staged_ = true;
    return EditStatus::kOk;
  }
  std::span<const int32_t> created() const { return ids_; }
  template <class Publish>
  EditStatus commit(const Publish& publish) {
    static_assert(std::is_nothrow_copy_assignable_v<Cell>);
    static_assert(std::is_nothrow_copy_constructible_v<Cell>);
    static_assert(std::is_nothrow_invocable_v<Publish>);
    if (!staged_) return EditStatus::kBoundaryMismatch;
    staged_ = false;
    for (int32_t id : removed_) cells_[id].live = false;
    for (std::size_t i = 0; i < proposal_.size(); ++i) {
      if (static_cast<std::size_t>(ids_[i]) == cells_.size()) cells_.push_back(proposal_[i]);
      else cells_[ids_[i]] = proposal_[i];
    }
    publish();
    return EditStatus::kOk;
  }

 private:
  std::vector<Cell>& cells_;
  std::vector<int32_t> removed_, ids_;
  std::vector<Cell> proposal_;
  bool staged_ = false;
};

}  // namespace phx::mc
