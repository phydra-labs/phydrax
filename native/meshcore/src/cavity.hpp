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

struct Tetrahedron {
  int32_t v[4];
  int32_t n[4];
  std::uint32_t stamp = 0;
  std::uint8_t state = 0;
};

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

class TetrahedralComplex {
 public:
  std::vector<Tetrahedron> tets;
  std::vector<int32_t> free_slots;
  int64_t finite_count = 0;

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
    reserve_for(regions_, tets.size());
    constraints_.assign(4 * tets.size(), kNoConstraint);
    regions_.assign(tets.size(), kNoRegion);
    labeled_ = true;
  }
  int32_t constraint(int32_t t, int k) const {
    return labeled_ ? constraints_[4 * static_cast<std::size_t>(t) + k] : kNoConstraint;
  }
  int32_t region(int32_t t) const {
    return labeled_ ? regions_[static_cast<std::size_t>(t)] : kNoRegion;
  }
  void set_facet_constraint(int32_t t, int k, int32_t id) {
    const int32_t other = tets[static_cast<std::size_t>(t)].n[k];
    const int back = neighbor_slot(tets[static_cast<std::size_t>(other)], t);
    constraints_[4 * static_cast<std::size_t>(t) + k] = id;
    constraints_[4 * static_cast<std::size_t>(other) + back] = id;
  }
  void set_region(int32_t t, int32_t value) {
    regions_[static_cast<std::size_t>(t)] = value;
  }
  void clear_regions() { std::fill(regions_.begin(), regions_.end(), kNoRegion); }

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
    if (labeled_) {
      constraints_.insert(constraints_.end(), 4, kNoConstraint);
      regions_.push_back(kNoRegion);
    }
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
    if (labeled_) {
      reserve_for(constraints_, constraints_.size() + 4 * fresh);
      reserve_for(regions_, regions_.size() + fresh);
    }
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
      }
    }
    return finite == finite_count;
  }

 private:
  bool labeled_ = false;
  std::vector<int32_t> constraints_;
  std::vector<int32_t> regions_;
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
      tet.n[face.slot] = face.outside;
      tet.stamp = 0;
      tet.state = 0;
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
  // block uses only the banks reserved here and staged edit storage.
  EditStatus commit() {
    ++staging_revision_;
    if (!staged_) return EditStatus::kBoundaryMismatch;
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
    ids_.reserve(created_.size());
    if (cone_center_ != kNoCone) reserve_star_table(created_.size());

    staged_ = false;
    const bool linked = apply(cone_center_ != kNoCone);
    complex_.finite_count += created_finite - removed_finite;
    cone_center_ = kNoCone;
    return linked ? EditStatus::kOk : EditStatus::kInternal;
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
  struct StarSlot { std::uint64_t key; int32_t tet; int32_t slot; std::uint32_t stamp; };

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

  void reserve_star_table(std::size_t count) {
    std::size_t capacity = 64;
    while (capacity < 6 * count) capacity <<= 1;
    if (star_table_.size() < capacity) {
      star_table_.assign(capacity, StarSlot{0, 0, 0, 0});
      star_stamp_ = 0;
    }
  }
  static std::uint64_t edge_key(int32_t a, int32_t b) {
    const int32_t low = std::min(a, b), high = std::max(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(low + 1)) << 32) |
           static_cast<std::uint32_t>(high + 1);
  }
  bool link_star(int32_t center) {
    ++star_stamp_;
    if (star_stamp_ == 0) {
      for (StarSlot& slot : star_table_) slot.stamp = 0;
      star_stamp_ = 1;
    }
    const std::size_t mask = star_table_.size() - 1;
    std::size_t pending = 0;
    for (int32_t id : ids_) {
      Tetrahedron& tet = complex_.tets[static_cast<std::size_t>(id)];
      const int apex = vertex_slot(tet, center);
      if (apex < 0) return false;
      for (int s = 0; s < 4; ++s) {
        if (s == apex) continue;
        int32_t pair[2];
        int count = 0;
        for (int r = 0; r < 4; ++r) if (r != s && r != apex) pair[count++] = tet.v[r];
        const std::uint64_t key = edge_key(pair[0], pair[1]);
        std::size_t h = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        for (;;) {
          StarSlot& entry = star_table_[h];
          if (entry.stamp != star_stamp_) {
            entry = {key, id, s, star_stamp_};
            ++pending;
            break;
          }
          if (entry.key == key) {
            if (entry.tet < 0) return false;
            tet.n[s] = entry.tet;
            complex_.tets[static_cast<std::size_t>(entry.tet)].n[entry.slot] = id;
            entry.tet = -1;
            --pending;
            break;
          }
          h = (h + 1) & mask;
        }
      }
    }
    return pending == 0;
  }
  bool apply(bool cone) {
    for (std::size_t i = 0; i < created_.size(); ++i) {
      ids_.push_back(i < removed_.size() ? removed_[i] : complex_.allocate());
    }
    for (std::size_t i = created_.size(); i < removed_.size(); ++i) complex_.release(removed_[i]);
    for (std::size_t i = 0; i < created_.size(); ++i) {
      Tetrahedron tet = created_[i];
      if (!cone) {
        for (int k = 0; k < 4; ++k) {
          if (tet.n[k] != kExternal) tet.n[k] = ids_[static_cast<std::size_t>(tet.n[k])];
        }
      }
      complex_.tets[static_cast<std::size_t>(ids_[i])] = tet;
      if (complex_.labeled()) complex_.set_region(ids_[i], created_regions_[i]);
    }
    for (std::size_t f = 0; f < boundary_.size(); ++f) {
      const BoundaryFacet& face = boundary_[f];
      const int32_t owner = cone ? static_cast<int32_t>(4 * f) + face.slot : owners_[f];
      const int32_t id = ids_[static_cast<std::size_t>(owner / 4)];
      complex_.tets[static_cast<std::size_t>(id)].n[owner % 4] = face.outside;
      complex_.tets[static_cast<std::size_t>(face.outside)].n[face.back] = id;
    }
    if (cone && !link_star(cone_center_)) return false;
    if (complex_.labeled()) {
      for (std::size_t i = 0; i < created_.size(); ++i) {
        for (int k = 0; k < 4; ++k) {
          complex_.set_facet_constraint(ids_[i], k, created_constraints_[4 * i + k]);
        }
      }
    }
    return true;
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
  std::vector<StarSlot> star_table_;
  std::uint32_t star_stamp_ = 0;
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
