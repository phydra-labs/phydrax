//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Tetrahedral complex storage and transactional cavity edits.
//
// TetrahedralComplex owns reusable tetrahedron slots, their adjacency and
// optional labels (a constraint reference per facet and a region per cell).
// Every live tetrahedron (v0, v1, v2, v3) is positively oriented; n[k] is the
// neighbor across the facet opposite v[k], and the complex is a closed
// pseudomanifold in which ghost tetrahedra carry kGhostVertex for the
// unbounded side.  Slots are local, reusable indices, not identities: a
// released slot is reused last-in first-out.
//
// A CavityEdit replaces a set of live tetrahedra (the cavity) by proposed
// tetrahedra filling exactly the same oriented boundary.  The edit is staged
// in bounded buffers and validated before anything is written: removed
// tetrahedra are live and distinct, the cavity boundary is reciprocally
// linked, every oriented boundary facet is filled by exactly one proposed
// facet with the same orientation, the remaining proposed facets pair up in
// opposite orientations, finite proposals are positively oriented, facet
// constraints of the boundary carry over and constrained facets inside the
// cavity (plus declared preserved facets) survive with their references.
// commit() first reserves every buffer it may need (the only step that can
// throw), then applies the edit without allocation; rollback() discards a
// staged edit.  A refused or failed edit therefore leaves the complex
// bit-identical.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "phydrax_meshcore.h"
#include "bounded_memory.hpp"

namespace phx::mc {

inline constexpr int32_t kGhostVertex = -1;
inline constexpr int32_t kDeadVertex = -2;
inline constexpr int32_t kNoConstraint = -1;
inline constexpr int32_t kNoRegion = -1;
inline constexpr int64_t kMaxTetrahedronSlots = std::numeric_limits<int32_t>::max();

struct Tetrahedron {
  int32_t v[4];
  int32_t n[4];
  // Scratch mark of the algorithm driving the complex (conflict caches).
  std::uint32_t stamp;
  std::uint8_t state;
};

inline bool is_ghost(const Tetrahedron& tet) {
  return tet.v[0] == kGhostVertex || tet.v[1] == kGhostVertex || tet.v[2] == kGhostVertex ||
         tet.v[3] == kGhostVertex;
}

inline int vertex_slot(const Tetrahedron& tet, int32_t vertex) {
  for (int k = 0; k < 4; ++k) {
    if (tet.v[k] == vertex) {
      return k;
    }
  }
  return -1;
}

inline int neighbor_slot(const Tetrahedron& tet, int32_t neighbor) {
  for (int k = 0; k < 4; ++k) {
    if (tet.n[k] == neighbor) {
      return k;
    }
  }
  return -1;
}

// Facet opposite v[k] as an oriented triple: (triple, v[k]) is an even
// permutation of (v0, v1, v2, v3), so v[k] lies on its positive side.
inline void oriented_facet(const int32_t* v, int k, int32_t* facet) {
  static constexpr int kOrder[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  for (int r = 0; r < 3; ++r) {
    facet[r] = v[kOrder[k][r]];
  }
}

// Orientation-preserving rotation of a triple to start at its smallest entry.
struct FacetKey {
  int32_t a;
  int32_t b;
  int32_t c;

  static FacetKey of(const int32_t* facet) {
    int first = 0;
    for (int r = 1; r < 3; ++r) {
      if (facet[r] < facet[first]) {
        first = r;
      }
    }
    return {facet[first], facet[(first + 1) % 3], facet[(first + 2) % 3]};
  }
  FacetKey reversed() const { return {a, c, b}; }
  bool operator==(const FacetKey& other) const {
    return a == other.a && b == other.b && c == other.c;
  }
  std::uint64_t hash() const {
    std::uint64_t h = static_cast<std::uint32_t>(a) * 0x9E3779B97F4A7C15ULL;
    h ^= (h >> 29) + static_cast<std::uint32_t>(b) * 0xC2B2AE3D27D4EB4FULL;
    h ^= (h >> 31) + static_cast<std::uint32_t>(c) * 0x165667B19E3779F9ULL;
    return h ^ (h >> 32);
  }
};

// Capacity grown geometrically so that repeated edits reserve amortized O(1).
template <class T, class Allocator>
void reserve_for(std::vector<T, Allocator>& values, std::size_t required) {
  if (values.capacity() < required) {
    values.reserve(std::max(required, 2 * values.capacity()));
  }
}

class TetrahedralComplex {
 private:
  MemoryOwner memory_owner_;
 public:
  explicit TetrahedralComplex(int64_t slot_limit = kMaxTetrahedronSlots,
                               MemoryOwner owner = scratch_memory_owner())
      : memory_owner_(std::move(owner)),
        tets(NativeAllocator<Tetrahedron>(memory_owner_)),
        free_slots(NativeAllocator<int32_t>(memory_owner_)),
        slot_limit_(std::min(slot_limit, kMaxTetrahedronSlots)),
        constraints_(NativeAllocator<int32_t>(memory_owner_)),
        regions_(NativeAllocator<int32_t>(memory_owner_)) {}

  NativeVector<Tetrahedron> tets;
  NativeVector<int32_t> free_slots;
  int64_t finite_count = 0;

  int64_t slot_limit() const { return slot_limit_; }
  const MemoryOwner& memory_owner() const noexcept { return memory_owner_; }
  bool live(int32_t t) const { return tets[static_cast<std::size_t>(t)].v[0] != kDeadVertex; }
  int64_t live_count() const {
    return static_cast<int64_t>(tets.size()) - static_cast<int64_t>(free_slots.size());
  }

  // Facet constraint references and cell regions, stored only once enabled.
  bool labeled() const { return labeled_; }
  void enable_labels() {
    MemoryScope memory(memory_owner_);
    if (!labeled_) {
      reserve_for(constraints_, 4 * tets.size());
      reserve_for(regions_, tets.size());
      constraints_.assign(4 * tets.size(), kNoConstraint);
      regions_.assign(tets.size(), kNoRegion);
      labeled_ = true;
    }
  }
  int32_t constraint(int32_t t, int k) const {
    return labeled_ ? constraints_[4 * static_cast<std::size_t>(t) + k] : kNoConstraint;
  }
  int32_t region(int32_t t) const {
    return labeled_ ? regions_[static_cast<std::size_t>(t)] : kNoRegion;
  }
  // Labels one facet on both of its sides.
  void set_facet_constraint(int32_t t, int k, int32_t id) {
    const int32_t other = tets[static_cast<std::size_t>(t)].n[k];
    const int back = neighbor_slot(tets[static_cast<std::size_t>(other)], t);
    constraints_[4 * static_cast<std::size_t>(t) + k] = id;
    constraints_[4 * static_cast<std::size_t>(other) + back] = id;
  }
  void set_region(int32_t t, int32_t region) { regions_[static_cast<std::size_t>(t)] = region; }
  void clear_regions() { std::fill(regions_.begin(), regions_.end(), kNoRegion); }

  // Slot for a new tetrahedron: the most recently released one, else a fresh
  // slot; -1 once the slot limit is reached.  Callers inside a commit reserve
  // first so that this never allocates there.
  int32_t allocate() {
    if (!free_slots.empty()) {
      const int32_t id = free_slots.back();
      free_slots.pop_back();
      return id;
    }
    if (static_cast<int64_t>(tets.size()) >= slot_limit_) {
      return -1;
    }
    reserve(1, 0);
    tets.push_back(Tetrahedron{});
    if (labeled_) {
      constraints_.insert(constraints_.end(), 4, kNoConstraint);
      regions_.push_back(kNoRegion);
    }
    return static_cast<int32_t>(tets.size() - 1);
  }

  void release(int32_t t) {
    reserve_for(free_slots, free_slots.size() + 1);
    Tetrahedron& dead = tets[static_cast<std::size_t>(t)];
    dead.v[0] = dead.v[1] = dead.v[2] = dead.v[3] = kDeadVertex;
    free_slots.push_back(t);
  }

  // Reserves room for `fresh` new slots and `freed` released slots.
  void reserve(std::size_t fresh, std::size_t freed) {
    reserve_for(tets, tets.size() + fresh);
    reserve_for(free_slots, free_slots.size() + freed);
    if (labeled_) {
      reserve_for(constraints_, constraints_.size() + 4 * fresh);
      reserve_for(regions_, regions_.size() + fresh);
    }
  }

  std::size_t retained_bytes() const {
    return tets.capacity() * sizeof(Tetrahedron) + free_slots.capacity() * sizeof(int32_t) +
           constraints_.capacity() * sizeof(int32_t) + regions_.capacity() * sizeof(int32_t);
  }

  // Structural audit: live neighbors, reciprocal links, shared facets with
  // opposite orientations, symmetric constraint references, the finite count.
  bool audit() const {
    int64_t finite = 0;
    for (std::size_t t = 0; t < tets.size(); ++t) {
      const Tetrahedron& tet = tets[t];
      if (tet.v[0] == kDeadVertex) {
        continue;
      }
      finite += is_ghost(tet) ? 0 : 1;
      for (int k = 0; k < 4; ++k) {
        const int32_t other = tet.n[k];
        if (other < 0 || static_cast<std::size_t>(other) >= tets.size() || !live(other)) {
          return false;
        }
        const Tetrahedron& neighbor = tets[static_cast<std::size_t>(other)];
        const int back = neighbor_slot(neighbor, static_cast<int32_t>(t));
        if (back < 0) {
          return false;
        }
        int32_t facet[3];
        int32_t twin[3];
        oriented_facet(tet.v, k, facet);
        oriented_facet(neighbor.v, back, twin);
        if (!(FacetKey::of(facet) == FacetKey::of(twin).reversed())) {
          return false;
        }
        if (constraint(static_cast<int32_t>(t), k) != constraint(other, back)) {
          return false;
        }
      }
    }
    return finite == finite_count;
  }

 private:
  int64_t slot_limit_;
  bool labeled_ = false;
  NativeVector<int32_t> constraints_;  // 4 per slot
  NativeVector<int32_t> regions_;      // 1 per slot
};

// Why a cavity edit was refused; the complex is unchanged in every case but
// kInternal, which reports a cone whose star linking failed after writing.
enum class EditStatus : std::uint8_t {
  kOk,
  kBufferLimit,          // more removed or proposed tetrahedra than the edit bound
  kCellLimit,            // the finite cell count would exceed its limit
  kSlotLimit,            // no slot index left
  kNotLive,              // a removed slot is dead or out of range
  kDuplicate,            // a slot removed twice, or a proposed facet repeated
  kNonManifold,          // the cavity boundary is not reciprocally linked
  kBoundaryMismatch,     // the proposal does not fill the cavity's oriented boundary
  kInverted,             // a proposal is degenerate or a finite one not positively oriented
  kConstraintViolation,  // a constrained or preserved facet would disappear or change
  kInternal,             // broken star-shapedness premise of a cone commit
};

inline int32_t edit_status_code(EditStatus status) {
  switch (status) {
    case EditStatus::kOk:
      return PHX_MC_OK;
    case EditStatus::kBufferLimit:
    case EditStatus::kCellLimit:
    case EditStatus::kSlotLimit:
      return PHX_MC_CAPACITY_EXCEEDED;
    case EditStatus::kConstraintViolation:
      return PHX_MC_CONSTRAINT_INTERSECTION;
    case EditStatus::kNotLive:
    case EditStatus::kDuplicate:
    case EditStatus::kNonManifold:
    case EditStatus::kBoundaryMismatch:
    case EditStatus::kInverted:
      return PHX_MC_INVALID_INPUT;
    case EditStatus::kInternal:
      return PHX_MC_INTERNAL_ERROR;
  }
  return PHX_MC_INTERNAL_ERROR;
}

struct EditLimits {
  std::size_t max_removed = std::numeric_limits<std::size_t>::max();
  std::size_t max_created = std::numeric_limits<std::size_t>::max();
  int64_t max_finite = std::numeric_limits<int64_t>::max();
};

// Facet of a cavity tetrahedron whose neighbor stays.
struct BoundaryFacet {
  int32_t tet;      // cavity tetrahedron
  int32_t slot;     // facet opposite tets[tet].v[slot]
  int32_t outside;  // remaining neighbor across the facet
  int32_t back;     // slot of `tet` in tets[outside].n
};

class CavityEdit {
 public:
  explicit CavityEdit(TetrahedralComplex& complex)
      : complex_(complex), memory_owner_(complex.memory_owner()),
        removed_(NativeAllocator<int32_t>(memory_owner_)),
        sorted_removed_(NativeAllocator<int32_t>(memory_owner_)),
        boundary_(NativeAllocator<BoundaryFacet>(memory_owner_)),
        created_(NativeAllocator<Tetrahedron>(memory_owner_)),
        created_constraints_(NativeAllocator<int32_t>(memory_owner_)),
        created_regions_(NativeAllocator<int32_t>(memory_owner_)),
        owners_(NativeAllocator<int32_t>(memory_owner_)),
        preserved_(NativeAllocator<Preserved>(memory_owner_)),
        inner_constraints_(NativeAllocator<Preserved>(memory_owner_)),
        ids_(NativeAllocator<int32_t>(memory_owner_)),
        facets_(NativeAllocator<FacetSlot>(memory_owner_)),
        star_table_(NativeAllocator<StarSlot>(memory_owner_)) {}

  // Starts a new staged edit (the previous one is discarded).
  void begin(const EditLimits& limits) {
    ++staging_revision_;
    limits_ = limits;
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
    if (removed_.size() >= limits_.max_removed ||
        !native_execution_cavity(removed_.size() + created_.size() + 1) ||
        !native_execution_spend(0)) {
      return EditStatus::kBufferLimit;
    }
    if (t < 0 || static_cast<std::size_t>(t) >= complex_.tets.size() || !complex_.live(t)) {
      return EditStatus::kNotLive;
    }
    removed_.push_back(t);
    return EditStatus::kOk;
  }

  // Proposes one tetrahedron; `constraints` (or null) are the references of
  // its facets inside the proposal, boundary facets inherit the cavity's.
  EditStatus add(const int32_t* v, const int32_t* constraints, int32_t region) {
    ++staging_revision_;
    if (created_.size() >= limits_.max_created ||
        !native_execution_cavity(removed_.size() + created_.size() + 1) ||
        !native_execution_spend(0)) {
      return EditStatus::kBufferLimit;
    }
    created_.push_back(Tetrahedron{{v[0], v[1], v[2], v[3]}, {-1, -1, -1, -1}, 0U, 0});
    for (int k = 0; k < 4; ++k) {
      created_constraints_.push_back(constraints == nullptr ? kNoConstraint : constraints[k]);
    }
    created_regions_.push_back(region);
    return EditStatus::kOk;
  }

  // Declares a facet (either orientation) that the proposal must contain with
  // the given constraint reference.
  void preserve(int32_t a, int32_t b, int32_t c, int32_t id) {
    ++staging_revision_;
    const int32_t facet[3] = {a, b, c};
    preserved_.push_back({FacetKey::of(facet), id});
  }

  // Validates a general proposal; positive(v) decides the orientation of a
  // finite proposed tetrahedron (ghosts are validated through their facets).
  template <class Positive>
  EditStatus validate(const Positive& positive) {
    ++staging_revision_;
    MemoryScope memory(memory_owner_);
    staged_ = false;
    cone_center_ = kNoCone;
    EditStatus status = collect_boundary();
    if (status == EditStatus::kOk) {
      status = link_proposal(positive);
    }
    staged_ = status == EditStatus::kOk;
    return status;
  }

  // Stages the cone from `center` over the boundary of a star-shaped cavity
  // (an unconstrained Delaunay conflict region): each boundary facet keeps its
  // outside neighbor, constraint and the region of the cavity tetrahedron it
  // bounded.  Star-shapedness is the caller's premise; commit() reports
  // kInternal if linking contradicts it.
  EditStatus stage_cone(int32_t center, std::span<const int32_t> cavity,
                        std::span<const BoundaryFacet> boundary) {
    ++staging_revision_;
    MemoryScope memory(memory_owner_);
    staged_ = false;
    if (cavity.size() > limits_.max_removed || boundary.size() > limits_.max_created) {
      return EditStatus::kBufferLimit;
    }
    removed_.assign(cavity.begin(), cavity.end());
    boundary_.assign(boundary.begin(), boundary.end());
    created_.clear();
    created_constraints_.clear();
    created_regions_.clear();
    const bool labeled = complex_.labeled();
    for (const BoundaryFacet& facet : boundary) {
      Tetrahedron tet = complex_.tets[static_cast<std::size_t>(facet.tet)];
      tet.v[facet.slot] = center;
      tet.n[0] = tet.n[1] = tet.n[2] = tet.n[3] = -1;
      tet.n[facet.slot] = facet.outside;
      tet.stamp = 0U;
      tet.state = 0;
      created_.push_back(tet);
      if (labeled) {
        for (int k = 0; k < 4; ++k) {
          created_constraints_.push_back(
              k == facet.slot ? complex_.constraint(facet.tet, facet.slot) : kNoConstraint);
        }
        created_regions_.push_back(complex_.region(facet.tet));
      }
    }
    cone_center_ = center;
    staged_ = true;
    return EditStatus::kOk;
  }

  std::span<const int32_t> removed() const { return removed_; }
  std::span<const Tetrahedron> proposal() const { return created_; }

  // Applies the staged edit.  Refusals (cell or slot limits) leave the complex
  // unchanged; std::bad_alloc can only escape before any change.  On success
  // created() holds the new slots in proposal order: removed slots are reused
  // in removal order, then released slots, then fresh ones.
  EditStatus commit() {
    ++staging_revision_;
    MemoryScope memory(memory_owner_);
    if (!staged_) {
      return EditStatus::kBoundaryMismatch;
    }
    int64_t removed_finite = 0;
    for (int32_t t : removed_) {
      removed_finite += is_ghost(complex_.tets[static_cast<std::size_t>(t)]) ? 0 : 1;
    }
    int64_t created_finite = 0;
    for (const Tetrahedron& tet : created_) {
      created_finite += is_ghost(tet) ? 0 : 1;
    }
    if (complex_.finite_count - removed_finite + created_finite > limits_.max_finite) {
      return EditStatus::kCellLimit;
    }
    const std::size_t extra =
        created_.size() > removed_.size() ? created_.size() - removed_.size() : 0;
    const std::size_t fresh = extra - std::min(extra, complex_.free_slots.size());
    if (static_cast<int64_t>(complex_.tets.size()) + static_cast<int64_t>(fresh) >
        complex_.slot_limit()) {
      return EditStatus::kSlotLimit;
    }
    if (!native_execution_spend(0)) {
      return EditStatus::kBufferLimit;
    }
    // Every allocation happens here, before the complex changes.
    const std::size_t freed =
        removed_.size() > created_.size() ? removed_.size() - created_.size() : 0;
    complex_.reserve(fresh, freed);
    ids_.clear();
    ids_.reserve(created_.size());
    const bool cone = cone_center_ != kNoCone;
    if (cone) {
      reserve_star_table(created_.size());
    }
    // Check the native deadline again after allocation, before the no-fail
    // mutation block. The already-admitted atomic commit is never interrupted.
    if (!native_execution_spend(0)) return EditStatus::kBufferLimit;
    staged_ = false;
    const bool linked = apply(cone);
    complex_.finite_count += created_finite - removed_finite;
    cone_center_ = kNoCone;
    return linked ? EditStatus::kOk : EditStatus::kInternal;
  }

  // Discards the staged edit; nothing has been written.
  void rollback() {
    ++staging_revision_;
    staged_ = false;
    cone_center_ = kNoCone;
  }

  std::span<const int32_t> created() const { return ids_; }
  uint64_t staging_revision() const noexcept { return staging_revision_; }

  std::size_t retained_bytes() const {
    return (removed_.capacity() + sorted_removed_.capacity() + created_constraints_.capacity() +
            created_regions_.capacity() + owners_.capacity() + ids_.capacity()) *
               sizeof(int32_t) +
           boundary_.capacity() * sizeof(BoundaryFacet) +
           created_.capacity() * sizeof(Tetrahedron) +
           (preserved_.capacity() + inner_constraints_.capacity()) * sizeof(Preserved) +
           facets_.capacity() * sizeof(FacetSlot) + star_table_.capacity() * sizeof(StarSlot);
  }

 private:
  static constexpr int32_t kNoCone = -3;
  static constexpr int32_t kExternal = std::numeric_limits<int32_t>::min();

  struct Preserved {
    FacetKey key;
    int32_t id;
  };

  // Open-addressing slot of the proposal's facet table.
  struct FacetSlot {
    FacetKey key;
    int32_t owner;  // 4 * proposal index + facet slot, -1 when empty
  };

  // Open-addressing slot of the cone's star table (facets through the center
  // keyed by their two other vertices).
  struct StarSlot {
    std::uint64_t key;
    int32_t tet;
    int32_t slot;
    std::uint32_t stamp;
  };

  EditStatus collect_boundary() {
    sorted_removed_.assign(removed_.begin(), removed_.end());
    std::sort(sorted_removed_.begin(), sorted_removed_.end());
    if (std::adjacent_find(sorted_removed_.begin(), sorted_removed_.end()) !=
        sorted_removed_.end()) {
      return EditStatus::kDuplicate;
    }
    boundary_.clear();
    inner_constraints_.clear();
    for (int32_t t : removed_) {
      const Tetrahedron& tet = complex_.tets[static_cast<std::size_t>(t)];
      for (int k = 0; k < 4; ++k) {
        const int32_t other = tet.n[k];
        if (other < 0 || static_cast<std::size_t>(other) >= complex_.tets.size() ||
            !complex_.live(other)) {
          return EditStatus::kNonManifold;
        }
        const int back = neighbor_slot(complex_.tets[static_cast<std::size_t>(other)], t);
        if (back < 0) {
          return EditStatus::kNonManifold;
        }
        if (!std::binary_search(sorted_removed_.begin(), sorted_removed_.end(), other)) {
          boundary_.push_back({t, k, other, back});
        } else if (complex_.constraint(t, k) != kNoConstraint) {
          int32_t facet[3];
          oriented_facet(tet.v, k, facet);
          inner_constraints_.push_back({FacetKey::of(facet), complex_.constraint(t, k)});
        }
      }
    }
    return EditStatus::kOk;
  }

  int32_t find_facet(const FacetKey& key) const {
    const std::size_t mask = facets_.size() - 1;
    std::size_t h = static_cast<std::size_t>(key.hash()) & mask;
    while (facets_[h].owner >= 0) {
      if (facets_[h].key == key) {
        return facets_[h].owner;
      }
      h = (h + 1) & mask;
    }
    return -1;
  }

  bool insert_facet(const FacetKey& key, int32_t owner) {
    const std::size_t mask = facets_.size() - 1;
    std::size_t h = static_cast<std::size_t>(key.hash()) & mask;
    while (facets_[h].owner >= 0) {
      if (facets_[h].key == key) {
        return false;
      }
      h = (h + 1) & mask;
    }
    facets_[h] = {key, owner};
    return true;
  }

  int32_t& proposed_constraint(int32_t owner) {
    return created_constraints_[static_cast<std::size_t>(owner)];
  }

  // Four distinct vertices, at most one of them the ghost vertex.
  static bool well_formed(const int32_t* v) {
    int ghosts = 0;
    for (int i = 0; i < 4; ++i) {
      if (v[i] < kGhostVertex) {
        return false;
      }
      ghosts += v[i] == kGhostVertex ? 1 : 0;
      for (int j = i + 1; j < 4; ++j) {
        if (v[i] == v[j]) {
          return false;
        }
      }
    }
    return ghosts <= 1;
  }

  // Links the proposal to the cavity boundary and to itself (see the header
  // comment for the invariants checked); owners_[f] is the proposal facet
  // filling boundary facet f.
  template <class Positive>
  EditStatus link_proposal(const Positive& positive) {
    for (const Tetrahedron& tet : created_) {
      if (!well_formed(tet.v) || (!is_ghost(tet) && !positive(tet.v))) {
        return EditStatus::kInverted;
      }
    }
    std::size_t capacity = 16;
    while (capacity < 8 * created_.size()) {
      capacity <<= 1;
    }
    facets_.assign(capacity, FacetSlot{{0, 0, 0}, -1});
    for (std::size_t i = 0; i < created_.size(); ++i) {
      for (int k = 0; k < 4; ++k) {
        int32_t facet[3];
        oriented_facet(created_[i].v, k, facet);
        if (!insert_facet(FacetKey::of(facet), static_cast<int32_t>(4 * i) + k)) {
          return EditStatus::kDuplicate;
        }
      }
    }
    // Oriented boundary facets, seen from inside the cavity, are filled with
    // the same orientation; they keep the outside link and the constraint.
    owners_.clear();
    for (const BoundaryFacet& facet : boundary_) {
      int32_t triple[3];
      oriented_facet(complex_.tets[static_cast<std::size_t>(facet.tet)].v, facet.slot, triple);
      const int32_t owner = find_facet(FacetKey::of(triple));
      if (owner < 0) {
        return EditStatus::kBoundaryMismatch;
      }
      Tetrahedron& tet = created_[static_cast<std::size_t>(owner / 4)];
      if (tet.n[owner % 4] != -1) {
        return EditStatus::kBoundaryMismatch;
      }
      const int32_t inherited = complex_.constraint(facet.tet, facet.slot);
      int32_t& proposed = proposed_constraint(owner);
      if (proposed != kNoConstraint && proposed != inherited) {
        return EditStatus::kConstraintViolation;
      }
      proposed = inherited;
      tet.n[owner % 4] = kExternal;
      owners_.push_back(owner);
    }
    // Every other proposed facet pairs with its reversed twin.
    for (std::size_t i = 0; i < created_.size(); ++i) {
      for (int k = 0; k < 4; ++k) {
        if (created_[i].n[k] != -1) {
          continue;
        }
        int32_t facet[3];
        oriented_facet(created_[i].v, k, facet);
        const int32_t twin = find_facet(FacetKey::of(facet).reversed());
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
    // Constrained facets inside the cavity and declared facets survive.
    inner_constraints_.insert(inner_constraints_.end(), preserved_.begin(), preserved_.end());
    for (const Preserved& required : inner_constraints_) {
      int32_t owner = find_facet(required.key);
      if (owner < 0) {
        owner = find_facet(required.key.reversed());
      }
      if (owner < 0 || proposed_constraint(owner) != required.id) {
        return EditStatus::kConstraintViolation;
      }
    }
    return EditStatus::kOk;
  }

  void reserve_star_table(std::size_t count) {
    std::size_t capacity = 64;
    while (capacity < 6 * count) {
      capacity <<= 1;
    }
    if (star_table_.size() < capacity) {
      star_table_.assign(capacity, StarSlot{0, 0, 0, 0U});
      star_stamp_ = 0;
    }
  }

  // Unordered vertex pair key; the ghost vertex maps to 0.
  static std::uint64_t edge_key(int32_t a, int32_t b) {
    const int32_t low = std::min(a, b);
    const int32_t high = std::max(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(low + 1)) << 32) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(high + 1));
  }

  // Links the facets through the cone center of the new tetrahedra: each is
  // keyed by its two other vertices and must occur exactly twice.
  bool link_star(int32_t center) {
    ++star_stamp_;
    if (star_stamp_ == 0) {
      for (StarSlot& slot : star_table_) {
        slot.stamp = 0U;
      }
      star_stamp_ = 1;
    }
    const std::size_t mask = star_table_.size() - 1;
    std::size_t pending = 0;
    for (int32_t id : ids_) {
      Tetrahedron& tet = complex_.tets[static_cast<std::size_t>(id)];
      const int apex = vertex_slot(tet, center);
      for (int s = 0; s < 4; ++s) {
        if (s == apex) {
          continue;
        }
        int32_t pair[2];
        int count = 0;
        for (int r = 0; r < 4; ++r) {
          if (r != s && r != apex) {
            pair[count++] = tet.v[r];
          }
        }
        const std::uint64_t key = edge_key(pair[0], pair[1]);
        std::size_t h = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        for (;;) {
          StarSlot& entry = star_table_[h];
          if (entry.stamp != star_stamp_) {
            entry = StarSlot{key, id, s, star_stamp_};
            ++pending;
            break;
          }
          if (entry.key == key) {
            if (entry.tet < 0) {
              return false;
            }
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

  // Writes the staged edit without allocating (commit() reserved every
  // buffer); returns false only when cone linking contradicts its premise.
  bool apply(bool cone) {
    for (std::size_t i = 0; i < created_.size(); ++i) {
      ids_.push_back(i < removed_.size() ? removed_[i] : complex_.allocate());
    }
    for (std::size_t i = created_.size(); i < removed_.size(); ++i) {
      complex_.release(removed_[i]);
    }
    for (std::size_t i = 0; i < created_.size(); ++i) {
      Tetrahedron tet = created_[i];
      if (!cone) {
        for (int k = 0; k < 4; ++k) {
          if (tet.n[k] != kExternal) {
            tet.n[k] = ids_[static_cast<std::size_t>(tet.n[k])];
          }
        }
      }
      complex_.tets[static_cast<std::size_t>(ids_[i])] = tet;
      if (complex_.labeled()) {
        complex_.set_region(ids_[i], created_regions_[i]);
      }
    }
    // Outside neighbors now point at the tetrahedra filling their facets.
    for (std::size_t f = 0; f < boundary_.size(); ++f) {
      const BoundaryFacet& facet = boundary_[f];
      const int32_t owner = cone ? static_cast<int32_t>(4 * f) + facet.slot : owners_[f];
      const int32_t id = ids_[static_cast<std::size_t>(owner / 4)];
      complex_.tets[static_cast<std::size_t>(id)].n[owner % 4] = facet.outside;
      complex_.tets[static_cast<std::size_t>(facet.outside)].n[facet.back] = id;
    }
    if (cone && !link_star(cone_center_)) {
      return false;
    }
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
  MemoryOwner memory_owner_;
  EditLimits limits_;
  bool staged_ = false;
  uint64_t staging_revision_ = 0;
  int32_t cone_center_ = kNoCone;
  NativeVector<int32_t> removed_;
  NativeVector<int32_t> sorted_removed_;
  NativeVector<BoundaryFacet> boundary_;
  NativeVector<Tetrahedron> created_;
  NativeVector<int32_t> created_constraints_;  // 4 per proposal
  NativeVector<int32_t> created_regions_;
  NativeVector<int32_t> owners_;
  NativeVector<Preserved> preserved_;
  NativeVector<Preserved> inner_constraints_;
  NativeVector<int32_t> ids_;
  NativeVector<FacetSlot> facets_;
  NativeVector<StarSlot> star_table_;
  std::uint32_t star_stamp_ = 0;
};

}  // namespace phx::mc
