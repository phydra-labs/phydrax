//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Incremental 3D Delaunay and regular triangulation state (Bowyer-Watson) on
// a TetrahedralComplex, shared by the point-set entry points and prepared
// incremental construction.
//
// The triangulation of the convex hull is closed by ghost tetrahedra: every
// hull facet carries one tetrahedron whose remaining vertex is kGhostVertex.
// Every tetrahedron (v0, v1, v2, v3) is positively oriented: orient3d > 0 for
// finite ones, and for ghosts substituting a point strictly beyond the hull
// facet for the ghost vertex gives orient3d > 0.
//
// Conflicts are exact and index-ordered: a finite tetrahedron conflicts with p
// iff insphere_sos (power3d_sos for weighted points) > 0.  A ghost conflicts
// iff p lies strictly beyond its hull facet; when p is coplanar with the facet
// it conflicts iff its finite neighbor does (the neighbor's circumsphere, or
// orthosphere, restricted to the facet plane is the facet's circumcircle, so
// this is the limit of the perturbed predicate at the hull).  The
// triangulation is the regular triangulation of the symbolically perturbed
// lifting, hence the conflict region of p is connected and star-shaped from
// p, the visibility walk is acyclic, and the result depends only on the
// point set and its indices, never on the insertion order.
//
// Constrained facets (constraint references of the complex) are never
// crossed by a conflict region.  With constraints present, the cone over the
// visible conflict region is committed through a fully validated CavityEdit
// (positive orientation, oriented boundary, surviving constraints); an
// insertion whose constrained cavity is not star-shaped from p, or would
// delete a vertex, is rolled back and reported as a constraint conflict.
// Every refusal (work, cavity, cell or slot limit) is decided before the
// complex changes.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>

#include "bounded_memory.hpp"

#include "cavity.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {

// Limits of one insertion; work counts tetrahedra visited by the location
// walk and the conflict search plus tetrahedra created, cumulatively against
// work_limit.
struct InsertionLimits {
  int64_t max_tetrahedra = std::numeric_limits<int64_t>::max();
  std::size_t max_cavity = std::numeric_limits<std::size_t>::max();
  int64_t work_limit = std::numeric_limits<int64_t>::max();
};

// Which limit refused the most recent insertion.
enum class Refusal : std::uint8_t { kNone, kWork, kCavity, kCells, kSlots };

class Triangulation3D {
 public:
  Triangulation3D(const double* points, const double* weights, int64_t vertex_count,
                  const InsertionLimits& limits,
                  int64_t slot_limit = kMaxTetrahedronSlots,
                  MemoryOwner owner = scratch_memory_owner())
      : points_(points, static_cast<std::size_t>(3 * vertex_count)),
        weights_(weights, weights == nullptr ? 0 : static_cast<std::size_t>(vertex_count)),
        limits_(limits),
        complex_(slot_limit, std::move(owner)),
        edit_(complex_),
        vertex_stamp_(static_cast<std::size_t>(vertex_count), 0U,
                      NativeAllocator<std::uint32_t>(complex_.memory_owner())),
        cavity_(NativeAllocator<int32_t>(complex_.memory_owner())),
        stack_(NativeAllocator<int32_t>(complex_.memory_owner())),
        boundary_(NativeAllocator<BoundaryFacet>(complex_.memory_owner())),
        removed_(NativeAllocator<int32_t>(complex_.memory_owner())) {}

  // Reserve stamp storage before an owner's coordinate buffer can relocate.
  // Binding the already-reserved coordinates is then allocation-free.
  void reserve_vertices(int64_t vertex_count) {
    MemoryScope memory(complex_.memory_owner());
    vertex_stamp_.resize(static_cast<std::size_t>(vertex_count), 0U);
  }
  void rebind(const double* points, int64_t vertex_count) noexcept {
    points_ = std::span<const double>(points, static_cast<std::size_t>(3 * vertex_count));
  }

  InsertionLimits& limits() { return limits_; }
  TetrahedralComplex& complex() { return complex_; }
  const TetrahedralComplex& complex() const { return complex_; }
  const MemoryOwner& memory_owner() const noexcept { return complex_.memory_owner(); }
  int64_t work() const { return work_; }
  Refusal refusal() const { return refusal_; }
  std::size_t largest_cavity() const { return largest_cavity_; }
  int64_t constrained_facets() const { return constrained_facets_; }
  void count_constrained_facet() { ++constrained_facets_; }

  // Inserts `order` (distinct points) incrementally.  alive[v] is cleared for
  // redundant or removed weighted vertices.
  int32_t build(std::span<const int32_t> order, std::span<char> alive) {
    MemoryScope memory(complex_.memory_owner());
    const std::size_t count = order.size();
    if (count < 4) {
      return PHX_MC_DEGENERATE_INPUT;
    }
    const int32_t a = order[0];
    int32_t b = order[1];
    std::size_t third = 2;
    while (third < count) {
      if (!spend(1)) return PHX_MC_CAPACITY_EXCEEDED;
      if (!collinear3d(point(a), point(b), point(order[third]))) break;
      ++third;
    }
    if (third == count) {
      return PHX_MC_DEGENERATE_INPUT;
    }
    int32_t c = order[third];
    std::size_t fourth = third + 1;
    int sign = 0;
    while (fourth < count) {
      if (!spend(1)) return PHX_MC_CAPACITY_EXCEEDED;
      sign = orient3d(point(a), point(b), point(c), point(order[fourth]));
      if (sign != 0) break;
      ++fourth;
    }
    if (fourth == count) {
      return PHX_MC_DEGENERATE_INPUT;
    }
    if (!spend(5)) return PHX_MC_CAPACITY_EXCEEDED;
    const int32_t d = order[fourth];
    if (sign < 0) {
      std::swap(b, c);
    }
    complex_.tets.reserve(7 * count + 16);
    int32_t status = initialize(a, b, c, d);
    for (std::size_t k = 2; k < count && status == PHX_MC_OK; ++k) {
      if (k != third && k != fourth) {
        status = insert(order[k], alive);
      }
    }
    return status;
  }

  // Finite cells in slot order.
  void collect_cells(NativeVector<int32_t>& cells) const {
    MemoryScope memory(complex_.memory_owner());
    cells.clear();
    cells.reserve(static_cast<std::size_t>(complex_.finite_count) * 4);
    for (const Tetrahedron& tet : complex_.tets) {
      if (tet.v[0] != kDeadVertex && !is_ghost(tet)) {
        cells.insert(cells.end(), tet.v, tet.v + 4);
      }
    }
  }

  // Visibility walk from the hint: returns a finite tetrahedron containing p
  // (closed) or a ghost whose hull facet has p strictly beyond it; -1 when the
  // work limit is reached first.
  int32_t locate(const double* p) {
    MemoryScope memory(complex_.memory_owner());
    int32_t t = hint_;
    {
      const Tetrahedron& start = tet(t);
      const int slot = vertex_slot(start, kGhostVertex);
      if (slot >= 0) {
        t = start.n[slot];
      }
    }
    int32_t previous = -1;
    for (;;) {
      if (!spend(1)) {
        return -1;
      }
      const Tetrahedron& current = tet(t);
      const int first = static_cast<int>(splitmix64(walk_step_++) & 3U);
      int32_t next = -1;
      for (int i = 0; i < 4; ++i) {
        const int k = (first + i) & 3;
        // p is strictly on this side of the facet just crossed.
        if (current.n[k] == previous) {
          continue;
        }
        if (orient_with(current, k, p) < 0) {
          next = current.n[k];
          break;
        }
      }
      if (next < 0) {
        return t;
      }
      previous = t;
      t = next;
      if (is_ghost(tet(t))) {
        return t;
      }
    }
  }

  // orient3d of `tet` with v[slot] replaced by p.
  int orient_with(const Tetrahedron& tetrahedron, int slot, const double* p) const {
    MemoryScope memory(complex_.memory_owner());
    const double* q[4];
    for (int k = 0; k < 4; ++k) {
      q[k] = k == slot ? p : point(tetrahedron.v[k]);
    }
    return orient3d(q[0], q[1], q[2], q[3]);
  }

  // Inserts vertex p (its coordinates bound).  With `duplicate`, a p equal to
  // an existing vertex v is reported through *duplicate = v and changes
  // nothing; without it, p must differ from every vertex.
  int32_t insert(int32_t p, std::span<char> alive, int32_t* duplicate = nullptr) {
    MemoryScope memory(complex_.memory_owner());
    refusal_ = Refusal::kNone;
    ++stamp_value_;
    const int32_t origin = locate(point(p));
    if (origin < 0) {
      refusal_ = Refusal::kWork;
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    if (duplicate != nullptr) {
      *duplicate = coincident_vertex(origin, point(p));
      if (*duplicate >= 0) {
        hint_ = origin;
        return PHX_MC_OK;
      }
    }
    if (!conflict(origin, p)) {
      // Only a weighted point lying above the lower hull can be conflict-free.
      if (weights_.empty()) {
        return PHX_MC_INTERNAL_ERROR;
      }
      alive[static_cast<std::size_t>(p)] = 0;
      hint_ = origin;
      return PHX_MC_OK;
    }
    int32_t status = grow_cavity(origin, p);
    if (status != PHX_MC_OK) {
      return status;
    }
    status = removed_vertices();
    if (status != PHX_MC_OK) {
      return status;
    }
    const EditLimits edit_limits{limits_.max_cavity, std::numeric_limits<std::size_t>::max(),
                                 limits_.max_tetrahedra};
    edit_.begin(edit_limits);
    EditStatus edit = constrained_facets_ == 0 ? edit_.stage_cone(p, cavity_, boundary_)
                                               : stage_constrained(p);
    if (edit == EditStatus::kOk) {
      edit = edit_.commit();
    }
    if (edit != EditStatus::kOk) {
      edit_.rollback();
      return refuse(edit);
    }
    for (int32_t vertex : removed_) {
      alive[static_cast<std::size_t>(vertex)] = 0;
    }
    largest_cavity_ = std::max(largest_cavity_, cavity_.size());
    hint_ = edit_.created().back();
    return PHX_MC_OK;
  }

  // Classification of p in a located tetrahedron: -1 outside the hull, else
  // the number of facets whose planes contain p (0 interior, 1 facet, 2 edge,
  // 3 vertex).
  int location(int32_t t, const double* p) const {
    const Tetrahedron& current = tet(t);
    if (is_ghost(current)) {
      return -1;
    }
    int zeros = 0;
    for (int k = 0; k < 4; ++k) {
      zeros += orient_with(current, k, p) == 0 ? 1 : 0;
    }
    return zeros;
  }

  const Tetrahedron& tet(int32_t t) const { return complex_.tets[static_cast<std::size_t>(t)]; }
  const double* point(int32_t vertex) const {
    return points_.data() + 3 * static_cast<int64_t>(vertex);
  }

  // The vertex of tetrahedron t at exactly the position p, or -1.
  int32_t coincident_vertex(int32_t t, const double* p) const {
    for (int32_t vertex : tet(t).v) {
      if (vertex >= 0) {
        const double* q = point(vertex);
        if (q[0] == p[0] && q[1] == p[1] && q[2] == p[2]) {
          return vertex;
        }
      }
    }
    return -1;
  }

  // Finds the facet with live vertices (a, b, c): t and the slot opposite it.
  // Searches the star of a, located by the walk.
  bool find_facet(int32_t a, int32_t b, int32_t c, int32_t& t, int& slot) {
    const int32_t start = locate(point(a));
    if (start < 0 || vertex_slot(tet(start), a) < 0) {
      return false;
    }
    ++stamp_value_;
    stack_.clear();
    mutable_tet(start).stamp = stamp_value_;
    stack_.push_back(start);
    while (!stack_.empty()) {
      const int32_t current = stack_.back();
      stack_.pop_back();
      const Tetrahedron& candidate = tet(current);
      if (vertex_slot(candidate, b) >= 0 && vertex_slot(candidate, c) >= 0) {
        for (int k = 0; k < 4; ++k) {
          if (candidate.v[k] != a && candidate.v[k] != b && candidate.v[k] != c) {
            t = current;
            slot = k;
            return true;
          }
        }
      }
      // Facets through a are those opposite the other vertices.
      const int apex = vertex_slot(candidate, a);
      for (int k = 0; k < 4; ++k) {
        const int32_t next = candidate.n[k];
        if (k != apex && tet(next).stamp != stamp_value_) {
          mutable_tet(next).stamp = stamp_value_;
          stack_.push_back(next);
        }
      }
    }
    return false;
  }

  // Traverse proposed labels first. No accepted region is written until every
  // allocation and predicate needed by the complete traversal has succeeded.
  void flood_region(int32_t start, int32_t label, std::span<int32_t> proposed) {
    MemoryScope memory(complex_.memory_owner());
    stack_.clear();
    stack_.push_back(start);
    proposed[static_cast<std::size_t>(start)] = label;
    while (!stack_.empty()) {
      const int32_t t = stack_.back();
      stack_.pop_back();
      for (int k = 0; k < 4; ++k) {
        const int32_t next = tet(t).n[k];
        if (complex_.constraint(t, k) == kNoConstraint && !is_ghost(tet(next)) &&
            proposed[static_cast<std::size_t>(next)] == kNoRegion) {
          stack_.push_back(next);
          proposed[static_cast<std::size_t>(next)] = label;
        }
      }
    }
  }

  std::size_t retained_bytes() const {
    return complex_.retained_bytes() + edit_.retained_bytes() +
           vertex_stamp_.capacity() * sizeof(std::uint32_t) +
           (cavity_.capacity() + stack_.capacity() + removed_.capacity()) * sizeof(int32_t) +
           boundary_.capacity() * sizeof(BoundaryFacet);
  }

 private:
  // Per-insertion conflict cache states, valid while Tetrahedron::stamp equals
  // the current insertion stamp.
  static constexpr std::uint8_t kNoConflict = 0;
  static constexpr std::uint8_t kConflict = 1;
  static constexpr std::uint8_t kInCavity = 2;

  Tetrahedron& mutable_tet(int32_t t) { return complex_.tets[static_cast<std::size_t>(t)]; }

  bool spend(int64_t units) {
    if (units < 0 || work_ > limits_.work_limit - units || !native_execution_spend(units)) {
      return false;
    }
    work_ += units;
    return true;
  }

  // Limits refuse the insertion; any other refusal of the cone contradicts
  // the unconstrained star-shapedness theorem, or is a constraint conflict.
  int32_t refuse(EditStatus edit) {
    switch (edit) {
      case EditStatus::kBufferLimit:
        refusal_ = Refusal::kCavity;
        return PHX_MC_CAPACITY_EXCEEDED;
      case EditStatus::kCellLimit:
        refusal_ = Refusal::kCells;
        return PHX_MC_CAPACITY_EXCEEDED;
      case EditStatus::kSlotLimit:
        refusal_ = Refusal::kSlots;
        return PHX_MC_CAPACITY_EXCEEDED;
      default:
        return constrained_facets_ != 0 ? PHX_MC_CONSTRAINT_INTERSECTION
                                        : PHX_MC_INTERNAL_ERROR;
    }
  }

  bool finite_conflict(const Tetrahedron& current, int32_t p) const {
    const int32_t* v = current.v;
    if (weights_.empty()) {
      return insphere_sos(point(v[0]), point(v[1]), point(v[2]), point(v[3]), point(p), v[0],
                          v[1], v[2], v[3], p) > 0;
    }
    return power3d_sos(point(v[0]), point(v[1]), point(v[2]), point(v[3]), point(p),
                       weights_[v[0]], weights_[v[1]], weights_[v[2]], weights_[v[3]],
                       weights_[p], v[0], v[1], v[2], v[3], p) > 0;
  }

  bool conflict(int32_t t, int32_t p) {
    Tetrahedron& current = mutable_tet(t);
    if (current.stamp == stamp_value_) {
      return current.state != kNoConflict;
    }
    const int slot = vertex_slot(current, kGhostVertex);
    bool result = false;
    if (slot < 0) {
      result = finite_conflict(current, p);
    } else {
      const int side = orient_with(current, slot, point(p));
      result = side > 0 || (side == 0 && conflict(current.n[slot], p));
    }
    // The recursion only marks tetrahedra; storage never moves during a
    // conflict search, so `current` stays valid.
    current.stamp = stamp_value_;
    current.state = result ? kConflict : kNoConflict;
    return result;
  }

  // Grows the conflict region of p from `origin` without crossing constrained
  // facets; fills cavity_ and boundary_.
  int32_t grow_cavity(int32_t origin, int32_t p) {
    cavity_.clear();
    boundary_.clear();
    stack_.clear();
    if (limits_.max_cavity == 0 || !native_execution_cavity(1)) {
      refusal_ = Refusal::kCavity;
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    mutable_tet(origin).state = kInCavity;
    cavity_.push_back(origin);
    stack_.push_back(origin);
    while (!stack_.empty()) {
      const int32_t t = stack_.back();
      stack_.pop_back();
      // Each cavity tetrahedron examines its four neighbors.
      if (!spend(4)) {
        refusal_ = Refusal::kWork;
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      for (int k = 0; k < 4; ++k) {
        const int32_t neighbor = tet(t).n[k];
        if (tet(neighbor).stamp == stamp_value_ && tet(neighbor).state == kInCavity) {
          continue;
        }
        if (complex_.constraint(t, k) == kNoConstraint && conflict(neighbor, p)) {
          if (cavity_.size() >= limits_.max_cavity ||
              !native_execution_cavity(cavity_.size() + 1)) {
            refusal_ = Refusal::kCavity;
            return PHX_MC_CAPACITY_EXCEEDED;
          }
          mutable_tet(neighbor).state = kInCavity;
          cavity_.push_back(neighbor);
          stack_.push_back(neighbor);
        } else {
          const int back = neighbor_slot(tet(neighbor), t);
          if (back < 0) {
            return PHX_MC_INTERNAL_ERROR;
          }
          boundary_.push_back({t, k, neighbor, back});
        }
      }
    }
    if (!spend(static_cast<int64_t>(boundary_.size()))) {
      refusal_ = Refusal::kWork;
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    return PHX_MC_OK;
  }

  // Vertices of the cavity absent from its boundary lose their whole star;
  // only weighted (redundant) vertices may, and never across constraints.
  int32_t removed_vertices() {
    for (const BoundaryFacet& face : boundary_) {
      const Tetrahedron& current = tet(face.tet);
      for (int s = 0; s < 4; ++s) {
        if (s != face.slot && current.v[s] >= 0) {
          vertex_stamp_[static_cast<std::size_t>(current.v[s])] = stamp_value_;
        }
      }
    }
    removed_.clear();
    for (int32_t t : cavity_) {
      const Tetrahedron& current = tet(t);
      for (int s = 0; s < 4; ++s) {
        const int32_t vertex = current.v[s];
        if (vertex >= 0 && vertex_stamp_[static_cast<std::size_t>(vertex)] != stamp_value_) {
          vertex_stamp_[static_cast<std::size_t>(vertex)] = stamp_value_;
          removed_.push_back(vertex);
        }
      }
    }
    if (!removed_.empty() && (weights_.empty() || constrained_facets_ != 0)) {
      return constrained_facets_ != 0 ? PHX_MC_CONSTRAINT_INTERSECTION : PHX_MC_INTERNAL_ERROR;
    }
    return PHX_MC_OK;
  }

  // The cone over a visibility-limited cavity, validated as a general edit.
  EditStatus stage_constrained(int32_t p) {
    for (int32_t t : cavity_) {
      const EditStatus status = edit_.remove(t);
      if (status != EditStatus::kOk) {
        return status;
      }
    }
    for (const BoundaryFacet& face : boundary_) {
      Tetrahedron cone = tet(face.tet);
      cone.v[face.slot] = p;
      const EditStatus status = edit_.add(cone.v, nullptr, complex_.region(face.tet));
      if (status != EditStatus::kOk) {
        return status;
      }
    }
    return edit_.validate([&](const int32_t* v) {
      return orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3])) > 0;
    });
  }

  int32_t initialize(int32_t a, int32_t b, int32_t c, int32_t d) {
    if (limits_.max_tetrahedra < 1) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    const int32_t root = complex_.allocate();
    if (root < 0) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    mutable_tet(root) = Tetrahedron{{a, b, c, d}, {-1, -1, -1, -1}, 0U, kNoConflict};
    int32_t ghosts[4];
    for (int k = 0; k < 4; ++k) {
      const int32_t ghost = complex_.allocate();
      if (ghost < 0) {
        return PHX_MC_CAPACITY_EXCEEDED;
      }
      Tetrahedron shell = tet(root);
      shell.v[k] = kGhostVertex;
      // Odd permutation of the facet: the outside becomes the positive side.
      const int first = k == 0 ? 1 : 0;
      const int second = k <= 1 ? 2 : 1;
      std::swap(shell.v[first], shell.v[second]);
      shell.n[0] = shell.n[1] = shell.n[2] = shell.n[3] = -1;
      shell.n[k] = root;
      mutable_tet(ghost) = shell;
      mutable_tet(root).n[k] = ghost;
      ghosts[k] = ghost;
    }
    // The ghosts pair up across the facets through the ghost vertex.
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        link_ghosts(ghosts[i], ghosts[j]);
      }
    }
    complex_.finite_count = 1;
    hint_ = root;
    return PHX_MC_OK;
  }

  // Links two ghosts sharing a facet through the ghost vertex.
  void link_ghosts(int32_t first, int32_t second) {
    Tetrahedron& x = mutable_tet(first);
    Tetrahedron& y = mutable_tet(second);
    for (int s = 0; s < 4; ++s) {
      if (x.v[s] == kGhostVertex || vertex_slot(y, x.v[s]) >= 0) {
        continue;
      }
      for (int r = 0; r < 4; ++r) {
        if (y.v[r] != kGhostVertex && vertex_slot(x, y.v[r]) < 0) {
          x.n[s] = second;
          y.n[r] = first;
        }
      }
    }
  }

  std::span<const double> points_;
  std::span<const double> weights_;
  InsertionLimits limits_;
  TetrahedralComplex complex_;
  CavityEdit edit_;
  NativeVector<std::uint32_t> vertex_stamp_;
  std::uint32_t stamp_value_ = 0;
  std::uint64_t walk_step_ = 0;
  int32_t hint_ = 0;
  int64_t work_ = 0;
  int64_t constrained_facets_ = 0;
  Refusal refusal_ = Refusal::kNone;
  std::size_t largest_cavity_ = 0;
  NativeVector<int32_t> cavity_;
  NativeVector<int32_t> stack_;
  NativeVector<BoundaryFacet> boundary_;
  NativeVector<int32_t> removed_;
};

}  // namespace phx::mc
