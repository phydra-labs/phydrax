//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Incremental 3D Delaunay and regular triangulation on an owned tetrahedral
// complex. The closed ghost hull makes point location and cavity replacement
// use the same adjacency for point-set and PLC construction.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <span>
#include <vector>

#include "cavity.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "parallel.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "prefetch.hpp"
#include "source_construction.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {

// Outcome of the shared checked cone insertion. Every refusal leaves the
// complex unchanged; callers choose their fallback (refinement reports an
// unmet cell, Delaunay/PLC construction rebuilds).
enum class ConeInsertion : std::uint8_t {
  kApplied,
  kInvertedCell,   // a finite cone cell is not exactly positively oriented
  kRemovesVertex,  // a cavity vertex is absent from the cavity boundary
  kSlotLimit,      // the slot bank cannot hold the cone
  kInternal,       // the cavity boundary is not a closed surface the cone can link
};

// The one incremental insertion commit used by Delaunay/PLC construction and
// tetrahedral refinement: cone `cavity` (cells to remove, with `boundary` its
// outward facets) to `center` after `positive` accepts each finite cone cell.
// `positive` is the caller's exact orientation model (symbolically perturbed
// for a Delaunay construction whose conflicts are perturbed, plain exact for a
// mesh). A positive cone whose boundary is the cavity boundary tiles the
// cavity, so `center` lies in its interior and every boundary facet is matched
// once. Unless `allow_removed_vertices` (regular triangulations may bury a
// point), every cavity vertex must remain on the cavity boundary.
template <class Positive>
ConeInsertion insert_checked_cone(TetrahedralComplex& complex, CavityEdit& edit,
                                  const Positive& positive, int32_t center,
                                  std::span<const int32_t> cavity,
                                  std::span<const BoundaryFacet> boundary,
                                  bool allow_removed_vertices) {
  for (const BoundaryFacet& face : boundary) {
    Tetrahedron cone = complex.tets[static_cast<std::size_t>(face.tet)];
    cone.v[face.slot] = center;
    if (is_ghost(cone)) continue;
    if (!positive(cone.v)) return ConeInsertion::kInvertedCell;
  }
  if (!allow_removed_vertices) {
    std::vector<int32_t> kept;
    kept.reserve(3 * boundary.size());
    for (const BoundaryFacet& face : boundary) {
      const Tetrahedron& cell = complex.tets[static_cast<std::size_t>(face.tet)];
      for (int slot = 0; slot < 4; ++slot) {
        if (slot != face.slot) kept.push_back(cell.v[slot]);
      }
    }
    std::sort(kept.begin(), kept.end());
    for (int32_t t : cavity) {
      for (int32_t vertex : complex.tets[static_cast<std::size_t>(t)].v) {
        if (vertex >= 0 && !std::binary_search(kept.begin(), kept.end(), vertex)) {
          return ConeInsertion::kRemovesVertex;
        }
      }
    }
  }
  EditStatus status = edit.stage_cone(center, cavity, boundary);
  if (status == EditStatus::kOk) status = edit.commit();
  switch (status) {
    case EditStatus::kOk: return ConeInsertion::kApplied;
    case EditStatus::kSlotLimit:
      edit.rollback();
      return ConeInsertion::kSlotLimit;
    case EditStatus::kNotLive:
    case EditStatus::kDuplicate:
    case EditStatus::kNonManifold:
    case EditStatus::kBoundaryMismatch:
    case EditStatus::kInverted:
    case EditStatus::kConstraintViolation:
    case EditStatus::kInternal: return ConeInsertion::kInternal;
  }
  return ConeInsertion::kInternal;
}

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
  // Exact positions of witnessed vertices; with a bank, every predicate tuple
  // that contains a witnessed vertex evaluates its exact position.
  void set_exact_points(const ExactVertexBank* bank) noexcept { exact_ = bank; }
  // Point-set construction stores the coordinates in insertion order, so a
  // walk and a cavity touch nearby memory; `identities[v]` is the input
  // index of vertex v, which orders the symbolic perturbation (null: v).
  void set_identities(const int32_t* identities) noexcept { identities_ = identities; }
  // Delaunay conflicts use physical distances but keep index-space geometry.
  void set_affine_metric(const double* linear) {
    metric_ = linear;
    if (linear == nullptr) {
      metric_orientation_ = 1;
      return;
    }
    const double origin[3] = {0.0, 0.0, 0.0};
    const double x[3] = {linear[0], linear[3], linear[6]};
    const double y[3] = {linear[1], linear[4], linear[7]};
    const double z[3] = {linear[2], linear[5], linear[8]};
    metric_orientation_ = orient3d(origin, x, y, z);
  }
  int delaunay_insphere(const ExactPointRef& a, const ExactPointRef& b,
                        const ExactPointRef& c, const ExactPointRef& d,
                        const ExactPointRef& e, int64_t ia, int64_t ib,
                        int64_t ic, int64_t id, int64_t ie) const {
    if (metric_ == nullptr) return insphere_sos(a, b, c, d, e, ia, ib, ic, id, ie);
    return affine_insphere_sos(metric_, metric_orientation_, a, b, c, d, e,
                               ia, ib, ic, id, ie);
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
  // insert_point_slot for a caller that knows every live cell holding a
  // vertex at or after `slot` (`cells`, each listed once): only those cells
  // are renumbered.
  bool insert_point_slot(std::size_t slot, std::size_t expected_count,
                         std::span<const int32_t> cells) {
    if (expected_count != vertex_stamp_.size() ||
        points_.size() / 3 != expected_count || !weights_.empty() ||
        slot > expected_count) {
      return false;
    }
    for (int32_t cell : cells) {
      if (!complex_.live(cell)) return false;
      for (int32_t vertex : complex_.tets[static_cast<std::size_t>(cell)].v) {
        if (vertex >= 0 && static_cast<std::size_t>(vertex) >= expected_count) return false;
      }
    }
    vertex_stamp_.insert(
        vertex_stamp_.begin() +
            static_cast<std::vector<std::uint32_t>::difference_type>(slot),
        0U);
    for (int32_t cell : cells) {
      for (int32_t& vertex : complex_.tets[static_cast<std::size_t>(cell)].v) {
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

  int32_t build(std::span<const int32_t> order, std::span<char> alive) {
    const std::size_t count = order.size();
    if (count < 4) return PHX_MC_DEGENERATE_INPUT;
    const int32_t a = order[0];
    int32_t b = order[1];
    std::size_t third = 2;
    while (third < count && collinear(ref(a), ref(b), ref(order[third]))) ++third;
    if (third == count) return PHX_MC_DEGENERATE_INPUT;
    int32_t c = order[third];
    std::size_t fourth = third + 1;
    int sign = 0;
    while (fourth < count) {
      sign = orient3d(ref(a), ref(b), ref(c), ref(order[fourth]));
      if (sign != 0) break;
      ++fourth;
    }
    if (fourth == count) return PHX_MC_DEGENERATE_INPUT;
    const int32_t d = order[fourth];
    if (sign < 0) std::swap(b, c);
    complex_.tets.reserve(7 * count + 16);
    const int32_t initialized = initialize(a, b, c, d);
    if (initialized != PHX_MC_OK) return initialized;
    // Only this loop edits the complex until build returns: it stays the
    // (regular) Delaunay triangulation of the inserted points, so the cells
    // it walks need no validation and an unweighted cavity buries no vertex.
    building_ = true;
    int32_t status = PHX_MC_OK;
    for (std::size_t k = 2; k < count && status == PHX_MC_OK; ++k) {
      if (k == third || k == fourth) continue;
      status = insert(order[k], alive);
    }
    building_ = false;
    return status;
  }
  // Inserts `points` in order into the triangulation build() returned, which
  // nothing has edited since (the caller's contract): it is still the
  // Delaunay triangulation of the inserted points, so as in build() the
  // cells need no validation.
  int32_t extend(std::span<const int32_t> points, std::span<char> alive) {
    building_ = true;
    int32_t status = PHX_MC_OK;
    for (std::size_t k = 0; k < points.size() && status == PHX_MC_OK; ++k) {
      status = insert(points[k], alive);
    }
    building_ = false;
    return status;
  }

  // build() with the points after the first four inserted by
  // insert_batches on `parallelism` workers when parallelism > 1 and the
  // point set holds at least kParallelPoints (below it the rounds cost more
  // than they save: measured even at 10^4 points, ahead from 3 * 10^4); the
  // same triangulation as build(), its cells canonicalized by the caller.
  int32_t build(std::span<const int32_t> order, std::span<char> alive, int32_t parallelism) {
    constexpr std::size_t kParallelPoints = 16384;
    const std::size_t count = order.size();
    if (parallelism <= 1 || count < kParallelPoints) return build(order, alive);
    const int32_t a = order[0];
    int32_t b = order[1];
    std::size_t third = 2;
    while (third < count && collinear(ref(a), ref(b), ref(order[third]))) ++third;
    if (third == count) return PHX_MC_DEGENERATE_INPUT;
    int32_t c = order[third];
    std::size_t fourth = third + 1;
    int sign = 0;
    while (fourth < count) {
      sign = orient3d(ref(a), ref(b), ref(c), ref(order[fourth]));
      if (sign != 0) break;
      ++fourth;
    }
    if (fourth == count) return PHX_MC_DEGENERATE_INPUT;
    const int32_t d = order[fourth];
    if (sign < 0) std::swap(b, c);
    // Rounds keep unused slots of their blocks free: room for them too.
    complex_.tets.reserve(9 * count + 16);
    const int32_t initialized = initialize(a, b, c, d);
    if (initialized != PHX_MC_OK) return initialized;
    std::vector<int32_t> rest;
    rest.reserve(count);
    for (std::size_t k = 2; k < count; ++k) {
      if (k != third && k != fourth) rest.push_back(order[k]);
    }
    building_ = true;
    const auto seed_end = static_cast<std::size_t>(std::max({a, b, c, d})) + 1;
    const int32_t status = insert_batches(rest, alive, seed_end, parallelism);
    building_ = false;
    return status;
  }

  // Emit the point IDs that order the symbolic perturbation, preserving
  // tetrahedron slot order and corner order.
  void collect_cells(std::vector<int32_t>& cells) const {
    cells.clear();
    cells.reserve(static_cast<std::size_t>(complex_.finite_count) * 4);
    for (const Tetrahedron& tet : complex_.tets) {
      if (tet.v[0] == kDeadVertex || is_ghost(tet)) continue;
      cells.insert(cells.end(),
                   {identity(tet.v[0]), identity(tet.v[1]), identity(tet.v[2]), identity(tet.v[3])});
    }
  }

  int32_t locate(const double* p) { return locate(ExactPointRef{p, nullptr}); }

  int32_t locate(const ExactPointRef& p) {
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
    if (!checked(t)) return -1;
    const int ghost_slot = vertex_slot(tet(t), kGhostVertex);
    if (ghost_slot >= 0) {
      const int32_t neighbor = tet(t).n[ghost_slot];
      if (!checked(neighbor)) return -1;
      t = neighbor;
    }
    int32_t previous = -1;
    for (;;) {
      if (!checked(t)) return -1;
      const Tetrahedron& current = tet(t);
      const int first = static_cast<int>(splitmix64(walk_step_++) & 3U);
      int32_t next = -1;
      for (int i = 0; i < 4; ++i) {
        const int k = (first + i) & 3;
        if (current.n[k] == previous) continue;
        if (orient_with(current, k, p) < 0) {
          next = current.n[k];
          if (!checked(next)) return -1;
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
    return orient_with(tetrahedron, slot, ExactPointRef{p, nullptr});
  }
  int orient_with(const Tetrahedron& tetrahedron, int slot, const ExactPointRef& p) const {
    if (p.exact == nullptr && carriers_exact()) {
      const double* q[4];
      for (int k = 0; k < 4; ++k) q[k] = k == slot ? p.carrier : point(tetrahedron.v[k]);
      return orient3d_filtered(q[0], q[1], q[2], q[3]);
    }
    ExactPointRef q[4];
    for (int k = 0; k < 4; ++k) q[k] = k == slot ? p : ref(tetrahedron.v[k]);
    if (carriers_only(q)) {
      return orient3d_filtered(q[0].carrier, q[1].carrier, q[2].carrier, q[3].carrier);
    }
    return orient3d(q[0], q[1], q[2], q[3]);
  }

  // Inserts point p by the cone over its conflict cavity, which grows only
  // through unconstrained faces. Refusals leave the complex unchanged:
  // CONSTRAINT_INTERSECTION when a constrained face has the cavity on both
  // sides (the cone would erase it); UNRESOLVED when the cavity is not
  // star-shaped from p or would bury a vertex of an unweighted triangulation
  // (the complex is no longer Delaunay after a caller's local edits).
  int32_t insert(int32_t p, std::span<char> alive, int32_t* duplicate = nullptr) {
    next_stamp();
    const int32_t origin = locate(ref(p));
    if (!checked(origin)) return PHX_MC_INTERNAL_ERROR;
    if (duplicate != nullptr) {
      *duplicate = coincident_vertex(origin, ref(p));
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
    if (constraint_inside_) return PHX_MC_CONSTRAINT_INTERSECTION;
    // Every vertex of an unweighted Delaunay cavity stays on its boundary:
    // a vertex strictly inside would have all its cells in conflict, so its
    // lifted point would lie above the lower hull of the lifted points with
    // p, but every point of the (perturbed) paraboloid is a lower-hull
    // vertex. Weighted points can be buried; after a caller's local edits
    // the complex need not be Delaunay, so the check stays.
    removed_.clear();
    if (!weights_.empty() || !building_) {
      collect_removed_vertices();
      if (weights_.empty() && !removed_.empty()) return PHX_MC_UNRESOLVED;
    }
    // While build() owns the complex, the cavity is the conflict region of p
    // in the triangulation of the perturbed lifting, star-shaped from p: a
    // facet whose plane holds p has both cells on one side of the conflict
    // test, so no cone cell is flat or inverted and the cone is committed
    // unchecked.
    ConeInsertion inserted = ConeInsertion::kApplied;
    if (building_) {
      switch (edit_.commit_cone(p, cavity_, boundary_)) {
        case EditStatus::kOk: break;
        case EditStatus::kSlotLimit: inserted = ConeInsertion::kSlotLimit; break;
        default: inserted = ConeInsertion::kInternal; break;
      }
    } else {
      inserted = insert_checked_cone(
          complex_, edit_,
          [this](const int32_t* v) {
            // A sign the floating stage decides is the unperturbed sign.
            if (carriers_exact()) {
              const OrientationStage stage =
                  orientation_stage(point(v[0]), point(v[1]), point(v[2]), point(v[3]));
              if (-stage.det > stage.errbound) return true;
              if (stage.det > stage.errbound) return false;
            }
            const ExactPointRef q[4]{ref(v[0]), ref(v[1]), ref(v[2]), ref(v[3])};
            if (carriers_only(q)) {
              const OrientationStage stage =
                  orientation_stage(q[0].carrier, q[1].carrier, q[2].carrier, q[3].carrier);
              if (-stage.det > stage.errbound) return true;
              if (stage.det > stage.errbound) return false;
            }
            return orient3d_sos(q[0], q[1], q[2], q[3], identity(v[0]), identity(v[1]),
                                identity(v[2]), identity(v[3])) > 0;
          },
          p, cavity_, boundary_, true);
    }
    switch (inserted) {
      case ConeInsertion::kApplied: break;
      case ConeInsertion::kInvertedCell:
      case ConeInsertion::kRemovesVertex: return PHX_MC_UNRESOLVED;
      case ConeInsertion::kSlotLimit: return PHX_MC_INDEX_RANGE_ERROR;
      case ConeInsertion::kInternal: return PHX_MC_INTERNAL_ERROR;
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
  ExactPointRef ref(int32_t vertex) const {
    return exact_ == nullptr ? ExactPointRef{point(vertex)}
                             : exact_->reference(points_.data(), vertex);
  }
  int32_t coincident_vertex(int32_t t, const ExactPointRef& p) const {
    for (int32_t vertex : tet(t).v) {
      if (vertex >= 0 && same_point(ref(vertex), p)) return vertex;
    }
    return -1;
  }

  bool find_facet(int32_t a, int32_t b, int32_t c, int32_t& t, int& slot) {
    const int32_t start = locate(ref(a));
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
  // Cell validation, skipped while build() owns the complex.
  bool checked(int32_t cell) const noexcept { return building_ || valid_point_tet(cell); }
  // Batched insertion (insert_batches): per worker, its search stamp, vertex
  // stamps (weighted), walk state and scratch.
  struct BatchWorker {
    std::uint32_t stamp = 0;
    std::vector<std::uint32_t> vertex_stamps;
    std::uint32_t vertex_stamp = 0;
    std::uint64_t walk_step = 0;
    std::vector<int32_t> stack;
    std::vector<std::array<int32_t, 4>> cone;
    // Open-addressed boundary edges of one cone (keys valid at edge_mark).
    std::vector<std::uint64_t> edge_keys;
    std::vector<std::uint32_t> edge_marks;
    std::vector<int32_t> edge_values;
    std::uint32_t edge_mark = 0;
  };
  // A point being inserted by a region: its conflict cavity, whether it is
  // hidden (weighted, conflicting with no cell), the vertices its cavity
  // buries, and the slots of its cone.
  struct BatchPoint {
    int32_t point = -1;
    int32_t status = PHX_MC_OK;
    bool hidden = false;
    std::vector<int32_t> cavity;
    std::vector<BoundaryFacet> boundary;
    std::vector<int32_t> removed;
    std::vector<int32_t> ids;
    int64_t finite_change = 0;
  };

  // conflict() on the region's own cells (only owned cells are tested, so
  // their stamps are the worker's to write), with the worker's stamp.
  bool batch_conflict(BatchWorker& worker, int32_t t, int32_t p) {
    Tetrahedron& current = mutable_tet(t);
    if (current.stamp == worker.stamp) return current.state != kNoConflict;
    const bool result = finite_conflict(current, p);
    current.stamp = worker.stamp;
    current.state = result ? kConflict : kNoConflict;
    return result;
  }

  // locate() from `start` with the worker's walk state, within the cells
  // `region` owns: -1 when the walk would leave them.
  int32_t batch_locate(BatchWorker& worker, int32_t start, const ExactPointRef& p,
                       int32_t region) {
    int32_t t = start;
    int32_t previous = -1;
    for (;;) {
      const Tetrahedron& current = tet(t);
      const int first = static_cast<int>(splitmix64(worker.walk_step++) & 3U);
      int32_t next = -1;
      for (int i = 0; i < 4; ++i) {
        const int k = (first + i) & 3;
        if (current.n[k] == previous) continue;
        if (orient_with(current, k, p) < 0) {
          next = current.n[k];
          break;
        }
      }
      if (next < 0) return t;
      if (!owned(next, region)) return -1;
      previous = t;
      t = next;
    }
  }

  // The conflict cavity of `entry.point` (grow_cavity of an unconstrained
  // complex, on the worker's visit table) and its buried vertices when
  // weighted, walked to from `start`; status kDeferred when the walk, the
  // cavity or a cell beyond it leaves the cells `region` owns.
  void batch_cavity(BatchWorker& worker, BatchPoint& entry, int32_t region,
                    int32_t start) {
    const int32_t p = entry.point;
    entry.status = kDeferred;
    entry.hidden = false;
    entry.cavity.clear();
    entry.boundary.clear();
    entry.removed.clear();
    worker.stamp = std::atomic_ref<std::uint32_t>(batch_stamp_).fetch_add(
                       1, std::memory_order_relaxed) + 1;
    if (start < 0 || !complex_.live(start) || !owned(start, region)) return;
    const int32_t origin = batch_locate(worker, start, ref(p), region);
    if (origin < 0) return;
    if (!batch_conflict(worker, origin, p)) {
      entry.status = weights_.empty() ? PHX_MC_INTERNAL_ERROR : PHX_MC_OK;
      entry.hidden = true;
      return;
    }
    // A neighbor of an owned cell shares three of the region's vertices, so
    // no other region stamps it.
    const auto in_cavity = [&](int32_t cell) {
      const Tetrahedron& current = tet(cell);
      return current.stamp == worker.stamp && current.state == kInCavity;
    };
    mutable_tet(origin).state = kInCavity;
    entry.cavity.push_back(origin);
    worker.stack.assign(1, origin);
    while (!worker.stack.empty()) {
      const int32_t t = worker.stack.back();
      worker.stack.pop_back();
      for (int k = 0; k < 4; ++k) {
        const int32_t neighbor = tet(t).n[k];
        if (in_cavity(neighbor)) continue;
        // Cavity and boundary cells are edited: both must be owned.
        if (!owned(neighbor, region)) return;
        if (batch_conflict(worker, neighbor, p)) {
          mutable_tet(neighbor).state = kInCavity;
          for (int32_t next : tet(neighbor).n) {
            if (static_cast<std::size_t>(next) < complex_.tets.size()) {
              prefetch(&complex_.tets[static_cast<std::size_t>(next)]);
            }
          }
          entry.cavity.push_back(neighbor);
          worker.stack.push_back(neighbor);
        } else {
          const int32_t* around = tet(neighbor).n;
          const int back = (around[1] == t ? 1 : 0) | (around[2] == t ? 2 : 0) |
                           (around[3] == t ? 3 : 0);
          if (around[back] != t) {
            entry.status = PHX_MC_INTERNAL_ERROR;
            return;
          }
          entry.boundary.push_back({t, k, neighbor, back});
        }
      }
    }
    entry.status = PHX_MC_OK;
    if (weights_.empty()) return;
    if (++worker.vertex_stamp == 0) {
      std::fill(worker.vertex_stamps.begin(), worker.vertex_stamps.end(), 0U);
      worker.vertex_stamp = 1;
    }
    for (const BoundaryFacet& face : entry.boundary) {
      const Tetrahedron& current = tet(face.tet);
      for (int slot = 0; slot < 4; ++slot) {
        if (slot != face.slot && current.v[slot] >= 0) {
          worker.vertex_stamps[static_cast<std::size_t>(current.v[slot])] = worker.vertex_stamp;
        }
      }
    }
    for (int32_t t : entry.cavity) {
      for (int32_t vertex : tet(t).v) {
        if (vertex >= 0 &&
            worker.vertex_stamps[static_cast<std::size_t>(vertex)] != worker.vertex_stamp) {
          worker.vertex_stamps[static_cast<std::size_t>(vertex)] = worker.vertex_stamp;
          entry.removed.push_back(vertex);
        }
      }
    }
  }

  // Writes the cone of `entry` into its slots `entry.ids` (its cavity's,
  // then the region's), linking the cone cells across the edges of the
  // cavity boundary and to the cells beyond it. All of them are cells the
  // region owns, so cones of concurrent regions never share a cell.
  void batch_commit(BatchWorker& worker, BatchPoint& entry) {
    const std::size_t count = entry.boundary.size();
    std::vector<Tetrahedron>& tets = complex_.tets;
    worker.cone.resize(count);
    int64_t created_finite = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const BoundaryFacet& face = entry.boundary[i];
      std::array<int32_t, 4>& cell = worker.cone[i];
      std::copy_n(tet(face.tet).v, 4, cell.begin());
      cell[static_cast<std::size_t>(face.slot)] = entry.point;
      created_finite += (cell[0] | cell[1] | cell[2] | cell[3]) >= 0 ? 1 : 0;
    }
    int64_t removed_finite = 0;
    for (int32_t t : entry.cavity) removed_finite += is_ghost(tet(t)) ? 0 : 1;
    entry.finite_change = created_finite - removed_finite;
    // Each boundary edge bounds two cone faces: the first one seen waits in
    // the worker's edge table for the second.
    std::size_t size = worker.edge_keys.size();
    if (size < 4 * 3 * count) {
      size = std::bit_ceil(4 * 3 * count);
      worker.edge_keys.assign(size, 0);
      worker.edge_marks.assign(size, 0U);
      worker.edge_values.assign(size, 0);
      worker.edge_mark = 0;
    }
    if (++worker.edge_mark == 0) {
      std::fill(worker.edge_marks.begin(), worker.edge_marks.end(), 0U);
      worker.edge_mark = 1;
    }
    const std::size_t mask = size - 1;
    for (std::size_t i = 0; i < count; ++i) {
      const BoundaryFacet& face = entry.boundary[i];
      const std::array<int32_t, 4>& cell = worker.cone[i];
      const int32_t id = entry.ids[i];
      Tetrahedron& created = tets[static_cast<std::size_t>(id)];
      created = Tetrahedron{{cell[0], cell[1], cell[2], cell[3]}, {-1, -1, -1, -1}, 0, 0, 0,
                            kNoRegion};
      created.n[face.slot] = face.outside;
      tets[static_cast<std::size_t>(face.outside)].n[face.back] = id;
      for (int k = 0; k < 4; ++k) {
        if (k == face.slot) continue;
        // The face opposite cell[k] is the center and the boundary edge
        // without cell[k].
        int32_t edge[2];
        int at = 0;
        for (int r = 0; r < 4; ++r) {
          if (r != k && r != face.slot) edge[at++] = cell[static_cast<std::size_t>(r)];
        }
        const std::uint64_t key =
            (std::uint64_t{static_cast<std::uint32_t>(std::min(edge[0], edge[1]))} << 32) |
            static_cast<std::uint32_t>(std::max(edge[0], edge[1]));
        std::size_t slot = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        while (worker.edge_marks[slot] == worker.edge_mark && worker.edge_keys[slot] != key) {
          slot = (slot + 1) & mask;
        }
        if (worker.edge_marks[slot] != worker.edge_mark) {
          worker.edge_marks[slot] = worker.edge_mark;
          worker.edge_keys[slot] = key;
          worker.edge_values[slot] = 4 * static_cast<int32_t>(i) + k;
          continue;
        }
        const int32_t other = worker.edge_values[slot];
        const int32_t twin = entry.ids[static_cast<std::size_t>(other / 4)];
        created.n[k] = twin;
        tets[static_cast<std::size_t>(twin)].n[other % 4] = id;
      }
    }
  }

  // A region of one pass: its points (in the caller's order), the
  // deferred ones, its slots (released cavity slots, then a block of fresh
  // ones), the walk start and the finite-cell change of its commits.
  struct RegionWork {
    int32_t region = -1;
    std::vector<int32_t> points;
    std::vector<int32_t> deferred;
    std::vector<int32_t> free;
    int32_t fresh = 0;
    int32_t fresh_end = 0;
    int32_t start = -1;
    int64_t finite_change = 0;
    int32_t status = PHX_MC_OK;
    // Whether points in the outer eighth of the region are deferred unseen.
    bool screen = false;
  };

  // Whether every vertex of `cell` lies in `region` (vertex_region_), so
  // only that region's worker edits it in a pass. An owned cell
  // keeps (pass epoch, region) in its region field, unused by a point-set
  // triangulation, so later tests read one field; only the owning region
  // writes it.
  bool owned(int32_t cell, int32_t region) {
    Tetrahedron& current = mutable_tet(cell);
    const int32_t tag = (owner_epoch_ << 16) | region;
    if (current.region == tag) return true;
    for (const int32_t vertex : current.v) {
      if (vertex < 0 || vertex_region_[static_cast<std::size_t>(vertex)] != region) {
        return false;
      }
    }
    current.region = tag;
    return true;
  }

  // Inserts the points of `work` one at a time as insert() does, but only
  // where the walk, the cavity and the cells beyond it are owned by the
  // region; any other point is deferred. A cell next to an owned cell shares
  // three of the region's vertices, so no other region edits it either: the
  // reads stay within cells no concurrent worker writes.
  void insert_region(BatchWorker& worker, RegionWork& work, std::span<char> alive) {
    BatchPoint entry;
    int32_t start = work.start;
    std::array<int32_t, 512> anchors;
    anchors.fill(-1);
    for (const int32_t p : work.points) {
      const int32_t anchor = anchors[vertex_fine_[static_cast<std::size_t>(p)]];
      if (anchor >= 0 && complex_.live(anchor) && owned(anchor, work.region)) start = anchor;
      entry.point = p;
      // On the first two passes a point in the outer eighth of its region
      // (by the 8^3 subdivision) is left to the next pass, whose shifted
      // grid puts it near a region centre: its cavity would likely leave.
      const std::uint16_t fine = vertex_fine_[static_cast<std::size_t>(p)];
      const auto border = [](unsigned index) { return index == 0 || index == 7; };
      if (work.screen &&
          (border(fine & 7U) || border((fine >> 3) & 7U) || border((fine >> 6) & 7U))) {
        work.deferred.push_back(p);
        continue;
      }
      batch_cavity(worker, entry, work.region, start);
      if (entry.status == kDeferred) {
        work.deferred.push_back(p);
        continue;
      }
      if (entry.status != PHX_MC_OK) {
        work.status = entry.status;
        return;
      }
      if (entry.hidden) {
        alive[static_cast<std::size_t>(p)] = 0;
        continue;
      }
      const std::size_t needed = entry.boundary.size();
      const std::size_t reused = std::min(entry.cavity.size(), needed);
      const std::size_t extra = needed - reused;
      if (extra > work.free.size() + static_cast<std::size_t>(work.fresh_end - work.fresh)) {
        work.deferred.push_back(p);
        continue;
      }
      entry.ids.assign(entry.cavity.begin(),
                       entry.cavity.begin() + static_cast<std::ptrdiff_t>(reused));
      while (entry.ids.size() < needed && !work.free.empty()) {
        entry.ids.push_back(work.free.back());
        work.free.pop_back();
      }
      while (entry.ids.size() < needed) entry.ids.push_back(work.fresh++);
      batch_commit(worker, entry);
      for (std::size_t i = needed; i < entry.cavity.size(); ++i) {
        complex_.tets[static_cast<std::size_t>(entry.cavity[i])] =
            Tetrahedron{{kDeadVertex, kDeadVertex, kDeadVertex, kDeadVertex},
                        {-1, -1, -1, -1}, 0, 0};
        work.free.push_back(entry.cavity[i]);
      }
      for (const int32_t vertex : entry.removed) alive[static_cast<std::size_t>(vertex)] = 0;
      work.finite_change += entry.finite_change;
      start = entry.ids.front();
      vertex_cell_[static_cast<std::size_t>(p)] = start;
      anchors[vertex_fine_[static_cast<std::size_t>(p)]] = start;
    }
  }

  // Inserts `points` (in order, into the triangulation build() owns) on
  // `parallelism` workers. After a sequential prefix, windows of points
  // (each at most as many as are inserted) are split by a grid of regions
  // that insert their points in parallel, handed out largest first, each only
  // where its walk, its cavity and every neighbour of the cavity lie in cells
  // whose vertices all lie in the region (insert_region), deferring the
  // others (on the first two passes, also every point in the outer eighth of
  // its region). Passes on the grid shifted by 1/2, 1/4 and 3/4 of a region
  // take the deferred points, and insert() the few left. Every insertion is
  // the conflict cavity of its point in the
  // current triangulation, so the result is the regular triangulation of the
  // perturbed lifting of all points, which is unique: the sequential build's
  // (as a set of cells; slots differ), for every parallelism. Uniqueness on
  // degenerate inputs rests on the perturbation ordered by identity(v), the
  // caller's fixed input index, which insertion order never changes.
  int32_t insert_batches(std::span<const int32_t> points, std::span<char> alive,
                         std::size_t active_vertices, int32_t parallelism) {
    // In a small triangulation every cavity reaches far: the first
    // kSequentialPoints points are inserted one at a time. Regions hold
    // about kRegionPoints points of a window.
    constexpr std::size_t kSequentialPoints = 4096;
    constexpr double kRegionPoints = 16384.0;
    vertex_cell_.assign(points_.size() / 3, -1);
    std::size_t next = 0;
    for (; next < points.size() && next < kSequentialPoints; ++next) {
      if (const int32_t status = insert(points[next], alive); status != PHX_MC_OK) return status;
      vertex_cell_[static_cast<std::size_t>(points[next])] = hint_;
      active_vertices = std::max(active_vertices, static_cast<std::size_t>(points[next]) + 1);
    }
    if (next == points.size()) return PHX_MC_OK;
    const std::size_t vertices = points_.size() / 3;
    double lower[3], upper[3];
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = std::numeric_limits<double>::infinity();
      upper[axis] = -std::numeric_limits<double>::infinity();
    }
    for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
      for (int axis = 0; axis < 3; ++axis) {
        lower[axis] = std::min(lower[axis], points_[3 * vertex + static_cast<std::size_t>(axis)]);
        upper[axis] = std::max(upper[axis], points_[3 * vertex + static_cast<std::size_t>(axis)]);
      }
    }
    const double extent[3] = {upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]};
    const auto workers = static_cast<std::size_t>(parallelism);
    workers_.resize(workers);
    vertex_region_.assign(vertices, 0);
    vertex_fine_.assign(vertices, 0);
    std::vector<int32_t> pending;
    std::vector<RegionWork> regions;
    std::vector<int32_t> deferred;
    while (next < points.size()) {
      const std::size_t count = std::min(points.size() - next, next);
      pending.assign(points.begin() + static_cast<std::ptrdiff_t>(next),
                     points.begin() + static_cast<std::ptrdiff_t>(next + count));
      next += count;
      // Future vertices have no live cell and cannot occur in a cavity
      // until their window is admitted. Seed and window IDs need not be consecutive.
      const int32_t last_vertex = *std::max_element(pending.begin(), pending.end());
      active_vertices = std::max(active_vertices, static_cast<std::size_t>(last_vertex) + 1);
      // At most 40 regions per axis (region numbers fit 16 bits).
      const auto grid = static_cast<int64_t>(std::clamp(
          std::round(std::cbrt(static_cast<double>(count) / kRegionPoints)), 2.0, 40.0));
      // Passes on the grid shifted by 0, 1/2, 1/4 and 3/4 of a region: a
      // point deferred near a region border lies inside a region of a later
      // pass.
      constexpr std::array<double, 4> kShifts{0.0, 0.5, 0.25, 0.75};
      for (std::size_t pass = 0; pass < kShifts.size() && !pending.empty(); ++pass) {
        if (++owner_epoch_ == (1 << 15)) {
          for (Tetrahedron& tetrahedron : complex_.tets) tetrahedron.region = kNoRegion;
          owner_epoch_ = 1;
        }
        // Grid coordinates of every vertex; regions are numbered by them.
        const double shift = kShifts[pass];
        const int64_t side = grid + (shift > 0.0 ? 1 : 0);
        // The shifted coordinate is nonnegative and bounded by grid + 3/4.
        // Truncating after exact scaling by eight supplies both floor(c)
        // and the low three bits of floor(8c).
        const auto bin = [&](const double* point, int axis) {
          const double unit = extent[axis] > 0.0
              ? (point[axis] - lower[axis]) / extent[axis] : 0.0;
          return static_cast<int64_t>(8.0 * (unit * static_cast<double>(grid) + shift));
        };
        const auto coarse = [side](int64_t fine) {
          return std::min(fine >> 3, side - 1);
        };
        const int32_t mapped = parallel_for(
            active_vertices, ParallelRequest{65536, static_cast<uint32_t>(workers)},
            [&](std::size_t first, std::size_t last, ChunkContext&) -> int32_t {
              for (std::size_t vertex = first; vertex < last; ++vertex) {
                const double* point = points_.data() + 3 * vertex;
                const int64_t x = bin(point, 0), y = bin(point, 1), z = bin(point, 2);
                vertex_region_[vertex] = static_cast<std::uint16_t>(
                    (coarse(z) * side + coarse(y)) * side + coarse(x));
                vertex_fine_[vertex] = static_cast<std::uint16_t>(
                    ((z & 7) * 8 + (y & 7)) * 8 + (x & 7));
              }
              return PHX_MC_OK;
            });
        if (mapped != PHX_MC_OK) return mapped;
        // A start cell per region: the recorded cell of its first vertex
        // that still holds that vertex and is owned by the region (no other
        // region edits it before this region's phase).
        std::vector<int32_t> region_start(static_cast<std::size_t>(side * side * side), -1);
        std::size_t unstarted = region_start.size();
        for (std::size_t vertex = 0; vertex < active_vertices && unstarted > 0; ++vertex) {
          const std::size_t region = vertex_region_[vertex];
          const int32_t cell = vertex_cell_[vertex];
          if (region_start[region] >= 0 || !complex_.live(cell) ||
              vertex_slot(tet(cell), static_cast<int32_t>(vertex)) < 0 ||
              !owned(cell, static_cast<int32_t>(region))) {
            continue;
          }
          region_start[region] = cell;
          --unstarted;
        }
        deferred.clear();
        {
          // Every region with pending points, each with its points in the
          // pending order. Regions run concurrently without colouring: a
          // region commits only cavities whose every neighbour it owns, so
          // its edits never touch a cell another region reads.
          std::vector<int32_t> region_index(static_cast<std::size_t>(side * side * side), -1);
          regions.clear();
          for (const int32_t p : pending) {
            const int32_t region = vertex_region_[static_cast<std::size_t>(p)];
            int32_t& at = region_index[static_cast<std::size_t>(region)];
            if (at < 0) {
              at = static_cast<int32_t>(regions.size());
              regions.emplace_back();
              regions.back().region = region;
            }
            regions[static_cast<std::size_t>(at)].points.push_back(p);
          }
          std::sort(regions.begin(), regions.end(), [](const RegionWork& a, const RegionWork& b) {
            return a.points.size() != b.points.size() ? a.points.size() > b.points.size()
                                                      : a.region < b.region;
          });
          // Slots: about eight cells per point, first from the free slots,
          // then a fresh block.
          std::size_t fresh = 0;
          for (RegionWork& work : regions) {
            const std::size_t block = 8 * work.points.size() + 64;
            const std::size_t taken = std::min(block, complex_.free_slots.size());
            work.free.assign(complex_.free_slots.end() - static_cast<std::ptrdiff_t>(taken),
                             complex_.free_slots.end());
            complex_.free_slots.resize(complex_.free_slots.size() - taken);
            work.fresh = static_cast<int32_t>(complex_.tets.size() + fresh);
            work.fresh_end = work.fresh + static_cast<int32_t>(block - taken);
            fresh += block - taken;
            work.start = region_start[static_cast<std::size_t>(work.region)];
            work.screen = pass < 2;
          }
          if (fresh > static_cast<std::size_t>(kMaxTetrahedronSlots) - complex_.tets.size()) {
            return PHX_MC_INDEX_RANGE_ERROR;
          }
          complex_.tets.resize(complex_.tets.size() + fresh,
                               Tetrahedron{{kDeadVertex, kDeadVertex, kDeadVertex, kDeadVertex},
                                           {-1, -1, -1, -1}, 0, 0});
          for (BatchWorker& worker : workers_) {
            if (worker.vertex_stamps.size() < vertex_stamp_.size()) {
              worker.vertex_stamps.resize(vertex_stamp_.size(), 0U);
            }
          }
          // Regions are handed out one at a time, largest first; a task takes
          // a free worker scratch for its region and returns it.
          std::vector<std::size_t> idle(workers);
          std::iota(idle.begin(), idle.end(), std::size_t{0});
          std::mutex idle_lock;
          // The searches take the stamps after the sequential ones.
          if (stamp_value_ > 0xC0000000U) {
            for (Tetrahedron& tetrahedron : complex_.tets) tetrahedron.stamp = 0;
            stamp_value_ = 0;
          }
          batch_stamp_ = stamp_value_;
          const int32_t status = parallel_for(
              regions.size(), ParallelRequest{1, static_cast<uint32_t>(workers)},
              [&](std::size_t first, std::size_t last, ChunkContext&) -> int32_t {
                const GradualNearestScope mode;
                std::size_t scratch = 0;
                {
                  const std::lock_guard guard(idle_lock);
                  scratch = idle.back();
                  idle.pop_back();
                }
                for (std::size_t at = first; at < last; ++at) {
                  insert_region(workers_[scratch], regions[at], alive);
                }
                const std::lock_guard guard(idle_lock);
                idle.push_back(scratch);
                return PHX_MC_OK;
              });
          stamp_value_ = batch_stamp_;
          if (status != PHX_MC_OK) return status;
          for (RegionWork& work : regions) {
            if (work.status != PHX_MC_OK) return work.status;
            complex_.finite_count += work.finite_change;
            // Released and borrowed slots precede this region's unused fresh
            // suffix. Append that suffix directly in descending slot order.
            for (int32_t slot = work.fresh_end; slot > work.fresh;) {
              complex_.free_slots.push_back(--slot);
            }
            std::sort(work.free.begin(), work.free.end(), std::greater<>());
            complex_.free_slots.insert(complex_.free_slots.end(), work.free.begin(),
                                       work.free.end());
            deferred.insert(deferred.end(), work.deferred.begin(), work.deferred.end());
          }
          if (!complex_.live(hint_)) {
            for (const RegionWork& work : regions) {
              if (complex_.live(work.start)) hint_ = work.start;
            }
          }
        }
        pending.swap(deferred);
      }
      // The points neither pass placed, one at a time.
      if (!complex_.live(hint_)) {
        hint_ = 0;
        while (!complex_.live(hint_)) ++hint_;
      }
      for (const int32_t p : pending) {
        if (const int32_t status = insert(p, alive); status != PHX_MC_OK) return status;
        vertex_cell_[static_cast<std::size_t>(p)] = hint_;
      }
    }
    // The ownership tags held in the (unlabeled) region field go.
    for (Tetrahedron& tetrahedron : complex_.tets) tetrahedron.region = kNoRegion;
    return PHX_MC_OK;
  }
  static constexpr std::uint8_t kNoConflict = 0;
  static constexpr std::uint8_t kConflict = 1;
  static constexpr std::uint8_t kInCavity = 2;
  // batch_cavity: the point is left to a later pass.
  static constexpr int32_t kDeferred = std::numeric_limits<int32_t>::min();

  Tetrahedron& mutable_tet(int32_t t) { return complex_.tets[static_cast<std::size_t>(t)]; }
  void next_stamp() {
    ++stamp_value_;
    if (stamp_value_ != 0) return;
    for (Tetrahedron& tetrahedron : complex_.tets) tetrahedron.stamp = 0;
    std::fill(vertex_stamp_.begin(), vertex_stamp_.end(), 0U);
    stamp_value_ = 1;
  }
  // Whether every reference is exact at its carrier, so the binary64
  // predicates decide them.
  template <std::size_t N>
  static bool carriers_only(const ExactPointRef (&refs)[N]) noexcept {
    for (const ExactPointRef& point : refs) {
      if (point.exact != nullptr) return false;
    }
    return true;
  }
  // Whether every vertex is exact at its carrier (no exact position banked).
  bool carriers_exact() const noexcept { return exact_ == nullptr || exact_->empty(); }
  bool finite_conflict(const Tetrahedron& current, int32_t p) const {
    const int32_t* v = current.v;
    if (weights_.empty()) {
      if (metric_ == nullptr) {
        // A sign the floating stage decides is the unperturbed sign.
        if (carriers_exact()) {
          const int sign =
              insphere_stage(point(v[0]), point(v[1]), point(v[2]), point(v[3]), point(p));
          if (sign != 0) return sign > 0;
        }
        const ExactPointRef q[5]{ref(v[0]), ref(v[1]), ref(v[2]), ref(v[3]), ref(p)};
        if (carriers_only(q)) {
          const int sign = insphere_stage(q[0].carrier, q[1].carrier, q[2].carrier,
                                          q[3].carrier, q[4].carrier);
          if (sign != 0) return sign > 0;
        }
        return insphere_sos(q[0], q[1], q[2], q[3], q[4], identity(v[0]), identity(v[1]),
                            identity(v[2]), identity(v[3]), identity(p)) > 0;
      }
      return delaunay_insphere(ref(v[0]), ref(v[1]), ref(v[2]), ref(v[3]), ref(p),
                               identity(v[0]), identity(v[1]), identity(v[2]),
                               identity(v[3]), identity(p)) > 0;
    }
    const double* q[5]{point(v[0]), point(v[1]), point(v[2]), point(v[3]), point(p)};
    const double w[5]{weights_[v[0]], weights_[v[1]], weights_[v[2]], weights_[v[3]], weights_[p]};
    // A sign the floating stage decides is the unperturbed sign.
    const int sign = power3d_stage(q[0], q[1], q[2], q[3], q[4], w[0], w[1], w[2], w[3], w[4]);
    if (sign != 0) return sign > 0;
    return power3d_sos(q[0], q[1], q[2], q[3], q[4], w[0], w[1], w[2], w[3], w[4],
                       identity(v[0]), identity(v[1]), identity(v[2]), identity(v[3]),
                       identity(p)) > 0;
  }
  // The index ordering the symbolic perturbation of vertex v.
  int32_t identity(int32_t v) const noexcept {
    return identities_ == nullptr ? v : identities_[static_cast<std::size_t>(v)];
  }
  bool conflict(int32_t t, int32_t p, bool& topology_valid) {
    if (!checked(t)) {
      topology_valid = false;
      return false;
    }
    Tetrahedron& current = mutable_tet(t);
    if (current.stamp == stamp_value_) return current.state != kNoConflict;
    // Live vertices are kGhostVertex (-1) or indices: a cell is finite iff
    // the bitwise or of its vertices is nonnegative.
    const int32_t* v = current.v;
    const bool finite = (v[0] | v[1] | v[2] | v[3]) >= 0;
    const int slot = finite ? -1 : vertex_slot(current, kGhostVertex);
    bool result = false;
    if (finite) {
      result = finite_conflict(current, p);
    } else {
      const int side = orient_with(current, slot, ref(p));
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
  // Grows the conflict cavity of p through unconstrained faces and records
  // in constraint_inside_ whether a constrained face has cavity cells on
  // both sides (reached around it).
  int32_t grow_cavity(int32_t origin, int32_t p) {
    if (!checked(origin)) return PHX_MC_INTERNAL_ERROR;
    cavity_.clear();
    boundary_.clear();
    stack_.clear();
    constraint_inside_ = false;
    mutable_tet(origin).state = kInCavity;
    cavity_.push_back(origin);
    stack_.push_back(origin);
    const auto in_cavity = [this](int32_t cell) {
      return tet(cell).stamp == stamp_value_ && tet(cell).state == kInCavity;
    };
    while (!stack_.empty()) {
      const int32_t t = stack_.back();
      stack_.pop_back();
      if (!checked(t)) return PHX_MC_INTERNAL_ERROR;
      for (int k = 0; k < 4; ++k) {
        const int32_t neighbor = tet(t).n[k];
        if (!checked(neighbor)) return PHX_MC_INTERNAL_ERROR;
        if (in_cavity(neighbor)) {
          if (complex_.constraint(t, k) != kNoConstraint) constraint_inside_ = true;
          continue;
        }
        bool topology_valid = true;
        const bool unconstrained =
            complex_.constraint(t, k) == kNoConstraint;
        const bool neighbor_conflict =
            unconstrained && conflict(neighbor, p, topology_valid);
        if (!topology_valid) return PHX_MC_INTERNAL_ERROR;
        if (neighbor_conflict) {
          Tetrahedron& joined = mutable_tet(neighbor);
          joined.state = kInCavity;
          // Its neighbors are read when it leaves the stack.
          for (int32_t next : joined.n) {
            if (static_cast<std::size_t>(next) < complex_.tets.size()) {
              prefetch(&complex_.tets[static_cast<std::size_t>(next)]);
            }
          }
          cavity_.push_back(neighbor);
          stack_.push_back(neighbor);
        } else {
          // The slot of t among the neighbor's neighbors, selected without
          // branches (a valid complex lists t once).
          const int32_t* around = tet(neighbor).n;
          const int back = (around[1] == t ? 1 : 0) | (around[2] == t ? 2 : 0) |
                           (around[3] == t ? 3 : 0);
          if (around[back] != t) return PHX_MC_INTERNAL_ERROR;
          boundary_.push_back({t, k, neighbor, back});
        }
      }
    }
    // A constrained boundary face whose outer cell joined the cavity later.
    for (const BoundaryFacet& face : boundary_) {
      if (in_cavity(face.outside)) constraint_inside_ = true;
    }
    return PHX_MC_OK;
  }
  // The cavity vertices absent from the cavity boundary (a regular
  // triangulation buries them).
  void collect_removed_vertices() {
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
  const ExactVertexBank* exact_ = nullptr;
  const int32_t* identities_ = nullptr;
  const double* metric_ = nullptr;
  int metric_orientation_ = 1;
  std::span<const double> weights_;
  TetrahedralComplex complex_;
  CavityEdit edit_;
  std::vector<std::uint32_t> vertex_stamp_;
  std::uint32_t stamp_value_ = 0;
  std::uint64_t walk_step_ = 0;
  int32_t hint_ = 0;
  bool constraint_inside_ = false;
  bool building_ = false;
  std::size_t largest_cavity_ = 0;
  std::vector<int32_t> cavity_, stack_, removed_;
  std::vector<BoundaryFacet> boundary_;

  std::vector<BatchWorker> workers_;
  // The region of every vertex in the current pass of insert_batches, and
  // its cell of the 8^3 subdivision of that region.
  std::vector<std::uint16_t> vertex_region_;
  std::vector<std::uint16_t> vertex_fine_;
  // The last cell stamp a region search took (atomically, so concurrent
  // searches never share a stamp).
  std::uint32_t batch_stamp_ = 0;
  // A cell created with each vertex (insert_batches; it may since be gone).
  std::vector<int32_t> vertex_cell_;
  // The pass of insert_batches whose ownership tags are current (owned()).
  int32_t owner_epoch_ = 0;
};

}  // namespace phx::mc
