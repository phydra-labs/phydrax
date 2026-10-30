//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Incremental 3D Delaunay and regular triangulation on an owned tetrahedral
// complex. The closed ghost hull makes point location and cavity replacement
// use the same adjacency for point-set and PLC construction.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "cavity.hpp"
#include "geometry3.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {

class Triangulation3D {
 public:
  Triangulation3D(const double* points, const double* weights, int64_t vertex_count)
      : points_(points, static_cast<std::size_t>(3 * vertex_count)),
        weights_(weights, weights == nullptr ? 0 : static_cast<std::size_t>(vertex_count)),
        complex_(), edit_(complex_),
        vertex_stamp_(static_cast<std::size_t>(vertex_count), 0U) {}

  void reserve_vertices(int64_t vertex_count) {
    vertex_stamp_.resize(static_cast<std::size_t>(vertex_count), 0U);
  }
  void rebind(const double* points, int64_t vertex_count) noexcept {
    points_ = std::span<const double>(points, static_cast<std::size_t>(3 * vertex_count));
  }

  // Shift live tetrahedron references and per-point stamps for a point-bank
  // insertion; callers rebind the coordinate span before geometric use.
  bool insert_point_slot(std::size_t slot, std::size_t expected_count) {
    if (expected_count != vertex_stamp_.size() ||
        points_.size() / 3 != expected_count || !weights_.empty() ||
        slot > expected_count) {
      return false;
    }
    for (std::size_t row = 0; row < complex_.tets.size(); ++row) {
      if (!complex_.live(static_cast<int32_t>(row))) continue;
      for (int32_t vertex : complex_.tets[row].v) {
        if (vertex >= 0 && static_cast<std::size_t>(vertex) >= expected_count) {
          return false;
        }
      }
    }
    vertex_stamp_.reserve(expected_count + 1);
    vertex_stamp_.insert(
        vertex_stamp_.begin() +
            static_cast<std::vector<std::uint32_t>::difference_type>(slot),
        0U);
    for (std::size_t row = 0; row < complex_.tets.size(); ++row) {
      if (!complex_.live(static_cast<int32_t>(row))) continue;
      for (int32_t& vertex : complex_.tets[row].v) {
        if (vertex >= 0 && static_cast<std::size_t>(vertex) >= slot) ++vertex;
      }
    }
    return true;
  }

  TetrahedralComplex& complex() { return complex_; }
  const TetrahedralComplex& complex() const { return complex_; }
  bool set_hint(int32_t cell) noexcept {
    if (!valid_point_tet(cell)) return false;
    hint_ = cell;
    return true;
  }
  bool valid_live_cell(int32_t cell) const noexcept {
    return valid_point_tet(cell);
  }
  std::size_t largest_cavity() const { return largest_cavity_; }
  int64_t constrained_facets() const { return constrained_facets_; }
  void count_constrained_facet() { ++constrained_facets_; }

  int32_t build(std::span<const int32_t> order, std::span<char> alive) {
    const std::size_t count = order.size();
    if (count < 4) return PHX_MC_DEGENERATE_INPUT;
    const int32_t a = order[0];
    int32_t b = order[1];
    std::size_t third = 2;
    while (third < count && geometry::collinear(point(a), point(b), point(order[third]))) ++third;
    if (third == count) return PHX_MC_DEGENERATE_INPUT;
    int32_t c = order[third];
    std::size_t fourth = third + 1;
    int sign = 0;
    while (fourth < count) {
      sign = orient3d(point(a), point(b), point(c), point(order[fourth]));
      if (sign != 0) break;
      ++fourth;
    }
    if (fourth == count) return PHX_MC_DEGENERATE_INPUT;
    const int32_t d = order[fourth];
    if (sign < 0) std::swap(b, c);
    complex_.tets.reserve(7 * count + 16);
    const int32_t initialized = initialize(a, b, c, d);
    if (initialized != PHX_MC_OK) return initialized;
    for (std::size_t k = 2; k < count; ++k) {
      if (k == third || k == fourth) continue;
      const int32_t status = insert(order[k], alive);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  void collect_cells(std::vector<int32_t>& cells) const {
    cells.clear();
    cells.reserve(static_cast<std::size_t>(complex_.finite_count) * 4);
    for (const Tetrahedron& tet : complex_.tets) {
      if (tet.v[0] != kDeadVertex && !is_ghost(tet)) cells.insert(cells.end(), tet.v, tet.v + 4);
    }
  }

  int32_t locate(const double* p) {
    int32_t t = hint_;
    if (!complex_.live(t)) {
      t = -1;
      for (int32_t candidate = 0;
           candidate < static_cast<int32_t>(complex_.tets.size()); ++candidate) {
        if (!complex_.live(candidate)) continue;
        if (!valid_point_tet(candidate)) return -1;
        if (!is_ghost(tet(candidate))) {
          t = candidate;
          break;
        }
        if (t < 0) t = candidate;
      }
      if (t < 0) return -1;
      hint_ = t;
    }
    if (!valid_point_tet(t)) return -1;
    const int ghost_slot = vertex_slot(tet(t), kGhostVertex);
    if (ghost_slot >= 0) {
      const int32_t neighbor = tet(t).n[ghost_slot];
      if (!valid_point_tet(neighbor)) return -1;
      t = neighbor;
    }
    int32_t previous = -1;
    for (;;) {
      if (!valid_point_tet(t)) return -1;
      const Tetrahedron& current = tet(t);
      const int first = static_cast<int>(splitmix64(walk_step_++) & 3U);
      int32_t next = -1;
      for (int i = 0; i < 4; ++i) {
        const int k = (first + i) & 3;
        if (current.n[k] == previous) continue;
        if (orient_with(current, k, p) < 0) {
          next = current.n[k];
          if (!valid_point_tet(next)) return -1;
          break;
        }
      }
      if (next < 0) return t;
      previous = t;
      t = next;
      if (is_ghost(tet(t))) return t;
    }
  }

  int orient_with(const Tetrahedron& tetrahedron, int slot, const double* p) const {
    const double* q[4];
    for (int k = 0; k < 4; ++k) q[k] = k == slot ? p : point(tetrahedron.v[k]);
    return orient3d(q[0], q[1], q[2], q[3]);
  }

  int32_t insert(int32_t p, std::span<char> alive, int32_t* duplicate = nullptr) {
    next_stamp();
    const int32_t origin = locate(point(p));
    if (!valid_point_tet(origin)) return PHX_MC_INTERNAL_ERROR;
    if (duplicate != nullptr) {
      *duplicate = coincident_vertex(origin, point(p));
      if (*duplicate >= 0) {
        hint_ = origin;
        return PHX_MC_OK;
      }
    }
    bool topology_valid = true;
    const bool origin_conflict = conflict(origin, p, topology_valid);
    if (!topology_valid) return PHX_MC_INTERNAL_ERROR;
    if (!origin_conflict) {
      if (weights_.empty()) return PHX_MC_INTERNAL_ERROR;
      alive[static_cast<std::size_t>(p)] = 0;
      hint_ = origin;
      return PHX_MC_OK;
    }
    int32_t status = grow_cavity(origin, p);
    if (status != PHX_MC_OK) return status;
    status = removed_vertices();
    if (status != PHX_MC_OK) return status;
    EditStatus edit = EditStatus::kOk;
    if (constrained_facets_ == 0) {
      edit = edit_.stage_cone(p, cavity_, boundary_);
    } else {
      edit = stage_constrained(p);
    }
    if (edit == EditStatus::kOk) edit = edit_.commit();
    if (edit != EditStatus::kOk) {
      edit_.rollback();
      return refuse(edit);
    }
    for (int32_t vertex : removed_) alive[static_cast<std::size_t>(vertex)] = 0;
    largest_cavity_ = std::max(largest_cavity_, cavity_.size());
    hint_ = edit_.created().back();
    return PHX_MC_OK;
  }

  int location(int32_t t, const double* p) const {
    const Tetrahedron& current = tet(t);
    if (is_ghost(current)) return -1;
    int zeros = 0;
    for (int k = 0; k < 4; ++k) zeros += orient_with(current, k, p) == 0 ? 1 : 0;
    return zeros;
  }

  const Tetrahedron& tet(int32_t t) const { return complex_.tets[static_cast<std::size_t>(t)]; }
  const double* point(int32_t vertex) const {
    return points_.data() + 3 * static_cast<int64_t>(vertex);
  }
  int32_t coincident_vertex(int32_t t, const double* p) const {
    for (int32_t vertex : tet(t).v) {
      if (vertex >= 0 && geometry::same_point(point(vertex), p)) return vertex;
    }
    return -1;
  }

  bool find_facet(int32_t a, int32_t b, int32_t c, int32_t& t, int& slot) {
    const int32_t start = locate(point(a));
    if (start < 0 || vertex_slot(tet(start), a) < 0) return false;
    next_stamp();
    stack_.clear();
    mutable_tet(start).stamp = stamp_value_;
    stack_.push_back(start);
    while (!stack_.empty()) {
      const int32_t current_id = stack_.back();
      stack_.pop_back();
      const Tetrahedron& candidate = tet(current_id);
      if (vertex_slot(candidate, b) >= 0 && vertex_slot(candidate, c) >= 0) {
        for (int k = 0; k < 4; ++k) {
          if (candidate.v[k] != a && candidate.v[k] != b && candidate.v[k] != c) {
            t = current_id;
            slot = k;
            return true;
          }
        }
      }
      const int apex = vertex_slot(candidate, a);
      for (int k = 0; k < 4; ++k) {
        const int32_t next = candidate.n[k];
        if (k == apex) continue;
        if (!valid_point_tet(next)) return false;
        if (tet(next).stamp != stamp_value_) {
          mutable_tet(next).stamp = stamp_value_;
          stack_.push_back(next);
        }
      }
    }
    return false;
  }

  void flood_region(int32_t start, int32_t label, std::span<int32_t> proposed) {
    stack_.clear();
    stack_.push_back(start);
    proposed[static_cast<std::size_t>(start)] = label;
    while (!stack_.empty()) {
      const int32_t t = stack_.back();
      stack_.pop_back();
      for (int k = 0; k < 4; ++k) {
        const int32_t next = tet(t).n[k];
        if (complex_.constraint(t, k) == kNoConstraint && complex_.live(next) &&
            !is_ghost(tet(next)) && proposed[static_cast<std::size_t>(next)] == kNoRegion) {
          stack_.push_back(next);
          proposed[static_cast<std::size_t>(next)] = label;
        }
      }
    }
  }

 private:
  bool valid_point_tet(int32_t cell) const noexcept {
    if (!complex_.live(cell)) return false;
    int ghost_count = 0;
    for (int32_t vertex : tet(cell).v) {
      if (vertex < kGhostVertex) return false;
      if (vertex == kGhostVertex) {
        ++ghost_count;
        continue;
      }
      if (static_cast<std::size_t>(vertex) >= vertex_stamp_.size() ||
          static_cast<std::size_t>(vertex) >= points_.size() / 3) {
        return false;
      }
    }
    return ghost_count <= 1;
  }
  static constexpr std::uint8_t kNoConflict = 0;
  static constexpr std::uint8_t kConflict = 1;
  static constexpr std::uint8_t kInCavity = 2;

  Tetrahedron& mutable_tet(int32_t t) { return complex_.tets[static_cast<std::size_t>(t)]; }
  void next_stamp() {
    ++stamp_value_;
    if (stamp_value_ != 0) return;
    for (Tetrahedron& tetrahedron : complex_.tets) tetrahedron.stamp = 0;
    std::fill(vertex_stamp_.begin(), vertex_stamp_.end(), 0U);
    stamp_value_ = 1;
  }
  int32_t refuse(EditStatus edit) const {
    switch (edit) {
      case EditStatus::kSlotLimit: return PHX_MC_INDEX_RANGE_ERROR;
      case EditStatus::kConstraintViolation: return PHX_MC_CONSTRAINT_INTERSECTION;
      case EditStatus::kOk: return PHX_MC_OK;
      case EditStatus::kNotLive:
      case EditStatus::kDuplicate:
      case EditStatus::kNonManifold:
      case EditStatus::kBoundaryMismatch:
      case EditStatus::kInverted:
        return constrained_facets_ == 0 ? PHX_MC_INTERNAL_ERROR : PHX_MC_CONSTRAINT_INTERSECTION;
      case EditStatus::kInternal: return PHX_MC_INTERNAL_ERROR;
    }
    return PHX_MC_INTERNAL_ERROR;
  }
  bool finite_conflict(const Tetrahedron& current, int32_t p) const {
    const int32_t* v = current.v;
    if (weights_.empty()) {
      return insphere_sos(point(v[0]), point(v[1]), point(v[2]), point(v[3]), point(p),
                          v[0], v[1], v[2], v[3], p) > 0;
    }
    return power3d_sos(point(v[0]), point(v[1]), point(v[2]), point(v[3]), point(p),
                       weights_[v[0]], weights_[v[1]], weights_[v[2]], weights_[v[3]],
                       weights_[p], v[0], v[1], v[2], v[3], p) > 0;
  }
  bool conflict(int32_t t, int32_t p, bool& topology_valid) {
    if (!valid_point_tet(t)) {
      topology_valid = false;
      return false;
    }
    Tetrahedron& current = mutable_tet(t);
    if (current.stamp == stamp_value_) return current.state != kNoConflict;
    const int slot = vertex_slot(current, kGhostVertex);
    bool result = false;
    if (slot < 0) {
      result = finite_conflict(current, p);
    } else {
      const int side = orient_with(current, slot, point(p));
      if (side > 0) {
        result = true;
      } else if (side == 0) {
        result = conflict(current.n[slot], p, topology_valid);
      }
    }
    if (!topology_valid) return false;
    current.stamp = stamp_value_;
    current.state = result ? kConflict : kNoConflict;
    return result;
  }
  int32_t grow_cavity(int32_t origin, int32_t p) {
    if (!valid_point_tet(origin)) return PHX_MC_INTERNAL_ERROR;
    cavity_.clear();
    boundary_.clear();
    stack_.clear();
    mutable_tet(origin).state = kInCavity;
    cavity_.push_back(origin);
    stack_.push_back(origin);
    while (!stack_.empty()) {
      const int32_t t = stack_.back();
      stack_.pop_back();
      if (!valid_point_tet(t)) return PHX_MC_INTERNAL_ERROR;
      for (int k = 0; k < 4; ++k) {
        const int32_t neighbor = tet(t).n[k];
        if (!valid_point_tet(neighbor)) return PHX_MC_INTERNAL_ERROR;
        if (tet(neighbor).stamp == stamp_value_ &&
            tet(neighbor).state == kInCavity) {
          continue;
        }
        bool topology_valid = true;
        const bool unconstrained =
            complex_.constraint(t, k) == kNoConstraint;
        const bool neighbor_conflict =
            unconstrained && conflict(neighbor, p, topology_valid);
        if (!topology_valid) return PHX_MC_INTERNAL_ERROR;
        if (neighbor_conflict) {
          mutable_tet(neighbor).state = kInCavity;
          cavity_.push_back(neighbor);
          stack_.push_back(neighbor);
        } else {
          const int back = neighbor_slot(tet(neighbor), t);
          if (back < 0) return PHX_MC_INTERNAL_ERROR;
          boundary_.push_back({t, k, neighbor, back});
        }
      }
    }
    return PHX_MC_OK;
  }
  int32_t removed_vertices() {
    for (const BoundaryFacet& face : boundary_) {
      const Tetrahedron& current = tet(face.tet);
      for (int slot = 0; slot < 4; ++slot) {
        if (slot != face.slot && current.v[slot] >= 0) {
          vertex_stamp_[static_cast<std::size_t>(current.v[slot])] = stamp_value_;
        }
      }
    }
    removed_.clear();
    for (int32_t t : cavity_) {
      const Tetrahedron& current = tet(t);
      for (int slot = 0; slot < 4; ++slot) {
        const int32_t vertex = current.v[slot];
        if (vertex >= 0 && vertex_stamp_[static_cast<std::size_t>(vertex)] != stamp_value_) {
          vertex_stamp_[static_cast<std::size_t>(vertex)] = stamp_value_;
          removed_.push_back(vertex);
        }
      }
    }
    if (!removed_.empty()) {
      return weights_.empty() || constrained_facets_ != 0
          ? (constrained_facets_ != 0 ? PHX_MC_CONSTRAINT_INTERSECTION : PHX_MC_INTERNAL_ERROR)
          : PHX_MC_OK;
    }
    return PHX_MC_OK;
  }
  EditStatus stage_constrained(int32_t p) {
    edit_.begin();
    for (int32_t t : cavity_) {
      const EditStatus status = edit_.remove(t);
      if (status != EditStatus::kOk) return status;
    }
    for (const BoundaryFacet& face : boundary_) {
      Tetrahedron cone = tet(face.tet);
      cone.v[face.slot] = p;
      const EditStatus status = edit_.add(cone.v, nullptr, complex_.region(face.tet));
      if (status != EditStatus::kOk) return status;
    }
    return edit_.validate([&](const int32_t* v) {
      return orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3])) > 0;
    });
  }
  int32_t initialize(int32_t a, int32_t b, int32_t c, int32_t d) {
    const int32_t root = complex_.allocate();
    if (root < 0) return PHX_MC_INDEX_RANGE_ERROR;
    mutable_tet(root) = Tetrahedron{{a, b, c, d}, {-1, -1, -1, -1}, 0U, kNoConflict};
    int32_t ghosts[4];
    for (int k = 0; k < 4; ++k) {
      const int32_t ghost = complex_.allocate();
      if (ghost < 0) return PHX_MC_INDEX_RANGE_ERROR;
      Tetrahedron shell = tet(root);
      shell.v[k] = kGhostVertex;
      const int first = k == 0 ? 1 : 0;
      const int second = k <= 1 ? 2 : 1;
      std::swap(shell.v[first], shell.v[second]);
      std::fill(shell.n, shell.n + 4, -1);
      shell.n[k] = root;
      mutable_tet(ghost) = shell;
      mutable_tet(root).n[k] = ghost;
      ghosts[k] = ghost;
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) link_ghosts(ghosts[i], ghosts[j]);
    }
    complex_.finite_count = 1;
    hint_ = root;
    return PHX_MC_OK;
  }
  void link_ghosts(int32_t first, int32_t second) {
    Tetrahedron& a = mutable_tet(first);
    Tetrahedron& b = mutable_tet(second);
    for (int s = 0; s < 4; ++s) {
      if (a.v[s] == kGhostVertex || vertex_slot(b, a.v[s]) >= 0) continue;
      for (int r = 0; r < 4; ++r) {
        if (b.v[r] != kGhostVertex && vertex_slot(a, b.v[r]) < 0) {
          a.n[s] = second;
          b.n[r] = first;
        }
      }
    }
  }

  std::span<const double> points_;
  std::span<const double> weights_;
  TetrahedralComplex complex_;
  CavityEdit edit_;
  std::vector<std::uint32_t> vertex_stamp_;
  std::uint32_t stamp_value_ = 0;
  std::uint64_t walk_step_ = 0;
  int32_t hint_ = 0;
  int64_t constrained_facets_ = 0;
  std::size_t largest_cavity_ = 0;
  std::vector<int32_t> cavity_, stack_, removed_;
  std::vector<BoundaryFacet> boundary_;
};

}  // namespace phx::mc
