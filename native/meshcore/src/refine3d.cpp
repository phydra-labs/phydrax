//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Constrained Delaunay refinement on a TetMesh.
//
// Work follows Shewchuk's priority order: encroached or oversized subsegments
// are split at exact shell/dyadic points, then subfacets at exact circumcenters
// that retire the queued source facet or exact longest-edge splits, then
// tetrahedra violating radius-edge or circumradius targets. Constrained edge
// sizes are checked independently, so large interior values cannot dilute a hard feature.
// A circumcenter that encroaches a subsegment/subfacet or lies beyond a constrained
// face, is not inserted; the encroached constraint is split instead and the
// tetrahedron is retried.  Exact split points lie exactly on their constraint.
// When none exists, the constrained edge is bisected at the carrier of an
// exact source witness whose certified deviation the source row's declared
// bound admits; otherwise (zero bound included) the constraint is left
// unsplit with NONREPRESENTABLE evidence and the refused deviation is
// recorded.  Points inside protecting balls are never inserted (PROTECTED
// evidence).  With a FIXED boundary no constraint is split and cells that
// would require it are reported.
//
// Queues are deterministic: FIFO for constraints, a max-heap keyed by
// (priority, sorted vertex tuple) for cells; stale entries are skipped by
// comparing the slot's vertex tuple.  Every insertion is one validated
// transaction, so the mesh is valid whenever refinement stops.
//
// Circumcenter insertion needs empty circumspheres: in a constrained Delaunay
// mesh no vertex visible from a cell lies inside its circumsphere, so a
// circumcenter is inserted at least the circumradius from every vertex it is
// connected to. A recovered carrier need not be constrained Delaunay, and
// Bowyer-Watson cavities refused or shrunk on it, like edge-star splits, add
// locally non-Delaunay faces; a circumcenter can then fall next to an existing
// vertex and the insertion radius collapses without end. Lawson flips restore
// the property on the carrier and on every cell an insertion creates: an
// unconstrained face whose opposite apex lies strictly inside the circumsphere
// of its cell (exact insphere, on exact source positions when bounded) is
// flipped 2-3, or 3-2 around a non-segment edge of it with three cells, or
// 4-4 around such an edge with four cells when both apexes are coplanar with
// it. Each flip replaces a locally non-Delaunay configuration by the other
// triangulation of the same vertices and strictly lowers the lifted
// triangulation, so flipping terminates. Constrained faces and segments never
// flip (immutable features are never crossed); a face no flip admits stays
// until a later change frees it. Flipped cells are queued and checked for
// encroachment like inserted ones. A circumcenter whose prepared cavity omits
// its cell, or connects it to a vertex strictly inside the cell's
// circumsphere (possible only beside such a face), is refused instead of
// inserted, so every circumcenter edge is at least the circumradius long.
//
// Work is not repeated where its answer is already determined: the
// circumcenter assessed for a cell is the one inserted; queued faces are not
// located unless the shape obstruction needs them; and a queued face whose
// vertex triple a committed insertion retired (its constrained faces with
// both sides in the cavity, or the faces around a split edge star) is
// skipped without a star walk, since every later face contains a later vertex.
#include "refine3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>

#include "bounded_memory.hpp"
#include "capi_guard.hpp"
#include "counter_records.hpp"
#include "filtered.hpp"
#include "geometry3.hpp"
#include "improve3d.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "tet_mesh.hpp"
#include "topology_keys.hpp"

namespace phx::mc {
namespace {


// Encroachment ranks inexact constructions: a point within rounding of a
// diametral or equatorial sphere does not encroach, so cospherical lattices
// do not trigger splits by rounding noise.
constexpr double kEncroachmentMargin = 1e-10;

bool encroaches_segment(const double* a, const double* b, const double* q) {
  double u[3], v[3], d[3];
  geometry::difference(a, q, u);
  geometry::difference(b, q, v);
  geometry::difference(b, a, d);
  return geometry::dot(u, v) < -kEncroachmentMargin * geometry::dot(d, d);
}

bool encroaches_face(const double* a, const double* b, const double* c, const double* q) {
  double center[3];
  if (!geometry::triangle_circumcenter(a, b, c, center)) {
    return false;
  }
  const double radius2 = geometry::squared_distance(a, center);
  return geometry::squared_distance(q, center) < (1.0 - kEncroachmentMargin) * radius2;
}


// Whether m lies exactly on the open segment (a, b).
bool exactly_inside_segment(const double* a, const double* b, const double* m) {
  if (!geometry::point_in_domain(m) || !geometry::collinear(a, b, m)) {
    return false;
  }
  int axis = 0;
  for (int i = 1; i < 3; ++i) {
    if (std::abs(b[i] - a[i]) > std::abs(b[axis] - a[axis])) {
      axis = i;
    }
  }
  const double low = std::min(a[axis], b[axis]);
  const double high = std::max(a[axis], b[axis]);
  return low < m[axis] && m[axis] < high;
}

struct TetEntry {
  double priority;
  int32_t slot;
  int32_t v[4];  // sorted
  // Max-heap order: larger priority first, then the smaller vertex tuple.
  bool operator<(const TetEntry& other) const {
    if (priority != other.priority) {
      return priority < other.priority;
    }
    return std::lexicographical_compare(other.v, other.v + 4, v, v + 4);
  }
};

struct SegmentEntry {
  std::uint64_t edge;
  bool forced;  // encroached by a rejected point rather than a vertex
};

struct FaceEntry {
  FaceKey key;
  bool forced;
};

// A face queued for the local Delaunay test: the cell, the vertex opposite
// the face in it, and the face triple (stale once either changed).
struct PendingFace {
  int32_t tet;
  int32_t apex;
  FaceKey face;
};

enum class Step : std::uint8_t { kContinue, kStop };

// At an acute protected source-facet corner theta, every descendant sector has
// angle <= theta. Its tetrahedron therefore has R/l_min >= 1/(2 sin(theta)).
// Comparing |u|^2|v|^2 with 4*b^2|u cross v|^2 is a division-free certificate;
// the acute condition is essential (obtuse corners can be subdivided).
template <class Number, class Difference, class Sign>
int source_corner_obstruction(const double* origin, const double* first,
                              const double* second, double bound, Difference difference,
                              Sign sign) {
  Number u[3], v[3];
  for (int k = 0; k < 3; ++k) {
    u[k] = difference(first[k], origin[k]);
    v[k] = difference(second[k], origin[k]);
  }
  const Number dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const int acute = sign(dot);
  if (acute <= 0) return 0;
  const Number uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  const Number vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const Number n[3] = {u[1] * v[2] - u[2] * v[1],
                       u[2] * v[0] - u[0] * v[2],
                       u[0] * v[1] - u[1] * v[0]};
  const Number nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
  const Number b = difference(bound, 0.0);
  const Number excess = uu * vv - difference(4.0, 0.0) * b * b * nn;
  const int obstruction = sign(excess);
  if (obstruction <= 0) return 0;
  return acute == 2 || obstruction == 2 ? 2 : 1;
}

bool certified_source_corner(const double* origin, const double* first,
                             const double* second, double bound) {
  const int filtered = source_corner_obstruction<Approx>(
      origin, first, second, bound,
      [](double a, double b) { return Approx::exact(a) - Approx::exact(b); },
      [](const Approx& value) { return value.certified_sign(); });
  if (filtered != 2) return filtered == 1;
  if (!coordinate_in_domain(bound)) return false;
  return source_corner_obstruction<Expansion>(
             origin, first, second, bound,
             [](double a, double b) { return Expansion::difference(a, b); },
             [](const Expansion& value) { return value.sign(); }) == 1;
}

class Refiner {
 public:
  Refiner(TetMesh& mesh, const RefineOptions& options) : mesh_(mesh), options_(options) {}

  int32_t run(TetMeshRefineCounters& counters) {
    const int64_t initial_work = mesh_.work();
    int32_t status = PHX_MC_OK;
    try {
      mesh_.unmet().clear();
      mesh_.clear_source_refusals();
      mesh_.clear_slot_reasons();
      prepare_shape_feasibility();
      if (restore_delaunay()) {
        seed_queues();
        status = mesh_.work_exhausted() ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
      } else {
        status = stop_status_;
      }
      while (status == PHX_MC_OK && (!segments_.empty() || !faces_.empty() || !tets_.empty())) {
        if (inserted_ >= options_.max_insertions) break;
        if (!mesh_.spend(1)) {
          status = PHX_MC_CAPACITY_EXCEEDED;
          break;
        }
        ++counters_.queue_pops;
        Step step = Step::kContinue;
        if (!segments_.empty()) {
          const SegmentEntry entry = segments_.front();
          segments_.pop_front();
          step = split_segment(entry);
        } else if (!faces_.empty()) {
          const FaceEntry entry = faces_.front();
          faces_.pop_front();
          step = split_face(entry);
        } else {
          const TetEntry entry = tets_.top();
          tets_.pop();
          step = refine_tet(entry);
        }
        if (step == Step::kStop) {
          status = stop_status_;
          break;
        }
      }
      record_unmet();
      if (mesh_.work_exhausted()) status = PHX_MC_CAPACITY_EXCEEDED;
      else if (status == PHX_MC_OK && !mesh_.unmet().empty()) status = PHX_MC_REFINEMENT_LIMIT;
    } catch (const ExecutionRefusal& refusal) {
      status = refusal.status;
    } catch (const std::bad_alloc&) {
      status = native_execution_status(PHX_MC_CAPACITY_EXCEEDED);
    }
    counters_.unmet_cells = static_cast<int64_t>(mesh_.unmet().size());
    counters_.work_units = mesh_.work() - initial_work;
    counters_.largest_cavity = static_cast<int64_t>(mesh_.largest_cavity());
    counters_.vertices = mesh_.vertex_count();
    counters = counters_;
    return native_execution_status(status);
  }

 private:
  bool conforming() const { return mesh_.policy() == BoundaryPolicy::kConforming; }

  void prepare_shape_feasibility() {
    NativeUnorderedSet<std::uint64_t> original_segments;
    for (const auto& edge : mesh_.original_segments()) {
      native_execution_charge(0);
      original_segments.insert(topology::undirected_edge(edge[0], edge[1]));
    }
    for (const auto& facet : mesh_.original_faces()) {
      native_execution_charge(0);
      for (int k = 0; k < 3; ++k) {
        const int32_t v = facet[static_cast<std::size_t>(k)];
        const int32_t a = facet[static_cast<std::size_t>((k + 1) % 3)];
        const int32_t b = facet[static_cast<std::size_t>((k + 2) % 3)];
        if (conforming() &&
            (original_segments.count(topology::undirected_edge(v, a)) == 0 ||
             original_segments.count(topology::undirected_edge(v, b)) == 0)) {
          continue;
        }
        if (!mesh_.spend(32)) return;
        if (certified_source_corner(mesh_.original_point(v), mesh_.original_point(a),
                                     mesh_.original_point(b), options_.radius_edge_bound)) {
          // The all-cell radius-edge goal is globally infeasible while this
          // exact angular stratum is retained. Continue independent size work,
          // but do not chase an impossible shape aim by ever-finer insertions.
          shape_obstructed_ = true;
          return;
        }
      }
    }
  }


  Unmet boundary_reason(int32_t t) const {
    for (int k = 0; k < 4; ++k) {
      if (mesh_.complex().constraint(t, k) != kNoConstraint) {
        const auto found = stuck_faces_.find(face_key(t, k));
        if (found != stuck_faces_.end()) {
          return found->second;
        }
        if (!conforming() && facet_size_ratio(t, k) > 1.0) {
          return Unmet::kFixedBoundary;
        }
      }
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        const int32_t a = mesh_.tet(t).v[i], b = mesh_.tet(t).v[j];
        const auto found = stuck_segments_.find(topology::undirected_edge(a, b));
        if (found != stuck_segments_.end()) {
          return found->second;
        }
        if (!conforming() && mesh_.segment_source(a, b) >= 0 && edge_size_ratio(a, b) > 1.0) {
          return Unmet::kFixedBoundary;
        }
      }
    }
    return Unmet::kBudget;
  }

  double edge_size_ratio(int32_t a, int32_t b) const {
    double target = std::numeric_limits<double>::infinity();
    for (int32_t v : {a, b}) {
      if (mesh_.size(v) > 0.0) {
        target = std::min(target, mesh_.size(v));
      }
    }
    return std::sqrt(geometry::squared_distance(mesh_.point(a), mesh_.point(b))) /
           (2.0 * target);
  }
  double facet_size_ratio(int32_t t, int slot) const {
    int32_t f[3];
    oriented_facet(mesh_.tet(t).v, slot, f);
    return std::max({edge_size_ratio(f[0], f[1]), edge_size_ratio(f[1], f[2]),
                     edge_size_ratio(f[2], f[0])});
  }

  // Whether finite tetrahedron t violates a criterion; priority > 1 ranks
  // it, ratio is its radius-edge ratio and size_ratio its circumradius over
  // the mean target size of its vertices (0 without targets). The computed
  // circumcenter is returned through `center` (valid when constructed) so the
  // insertion reuses it instead of constructing it again.
  struct Circumcenter {
    double point[3] = {0.0, 0.0, 0.0};
    bool valid = false;
  };
  bool assess(int32_t t, double& priority, double& ratio, double& size_ratio,
              Circumcenter* center = nullptr) const {
    const double* p[4];
    mesh_.corners(t, p);
    Circumcenter local;
    Circumcenter& built = center == nullptr ? local : *center;
    double* c = built.point;
    size_ratio = 0.0;
    built.valid = geometry::tetrahedron_circumcenter(p[0], p[1], p[2], p[3], c);
    if (!built.valid) {
      ratio = std::numeric_limits<double>::infinity();
      priority = ratio;
      return true;
    }
    const double radius = std::sqrt(geometry::squared_distance(c, p[0]));
    ratio = radius / geometry::shortest_edge(p);
    if (!std::isfinite(ratio)) {
      ratio = std::numeric_limits<double>::infinity();
      priority = ratio;
      return true;
    }
    double total = 0.0;
    int count = 0;
    for (int32_t v : mesh_.tet(t).v) {
      if (mesh_.size(v) > 0.0) {
        total += mesh_.size(v);
        ++count;
      }
    }
    if (count > 0) {
      size_ratio = radius / (total / count);
    }
    for (int k = 0; k < 4; ++k) {
      if (mesh_.complex().constraint(t, k) != kNoConstraint) {
        size_ratio = std::max(size_ratio, facet_size_ratio(t, k));
      }
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        const int32_t a = mesh_.tet(t).v[i];
        const int32_t b = mesh_.tet(t).v[j];
        if (mesh_.segment_source(a, b) >= 0) {
          size_ratio = std::max(size_ratio, edge_size_ratio(a, b));
        }
      }
    }
    priority = std::max(ratio / options_.radius_edge_bound, size_ratio);
    return ratio > options_.radius_edge_bound || size_ratio > 1.0;
  }

  void push_tet(int32_t t) {
    if (!mesh_.complex().live(t) || !mesh_.finite(t)) {
      return;
    }
    native_execution_charge(0);
    double priority = 0.0;
    double ratio = 0.0;
    double size_ratio = 0.0;
    if (!assess(t, priority, ratio, size_ratio)) {
      return;
    }
    if (ratio > options_.radius_edge_bound && shape_obstructed_) {
      if (size_ratio <= 1.0) {
        mesh_.set_slot_reason(t, Unmet::kProtected);
        return;
      }
      priority = size_ratio;
    }
    TetEntry entry{priority, t, {0, 0, 0, 0}};
    std::copy_n(mesh_.tet(t).v, 4, entry.v);
    std::sort(entry.v, entry.v + 4);
    tets_.push(entry);
  }

  bool current(const TetEntry& entry) const {
    if (!mesh_.complex().live(entry.slot) || !mesh_.finite(entry.slot)) {
      return false;
    }
    int32_t v[4];
    std::copy_n(mesh_.tet(entry.slot).v, 4, v);
    std::sort(v, v + 4);
    return std::equal(v, v + 4, entry.v);
  }

  bool protected_point(const double* p, const int32_t* vertices, int count) const {
    for (int k = 0; k < count; ++k) {
      const int32_t v = vertices[k];
      const double radius = v >= 0 ? mesh_.protection(v) : 0.0;
      if (radius > 0.0 && geometry::squared_distance(p, mesh_.point(v)) < radius * radius) {
        return true;
      }
    }
    return false;
  }

  bool cavity_protected(const double* p) const {
    for (int32_t t : mesh_.cavity()) {
      if (protected_point(p, mesh_.tet(t).v, 4)) {
        return true;
      }
    }
    return false;
  }

  bool segment_encroached(int32_t a, int32_t b) {
    if (!mesh_.edge_ring(a, b, ring_tets_, ring_)) {
      return false;
    }
    for (int32_t apex : ring_) {
      if (apex >= 0 &&
          encroaches_segment(mesh_.point(a), mesh_.point(b), mesh_.point(apex))) {
        return true;
      }
    }
    return false;
  }

  bool face_encroached_from(int32_t t, int slot) const {
    int32_t f[3];
    oriented_facet(mesh_.tet(t).v, slot, f);
    const int32_t apex = mesh_.tet(t).v[slot];
    return apex >= 0 && f[0] >= 0 && f[1] >= 0 && f[2] >= 0 &&
           encroaches_face(mesh_.point(f[0]), mesh_.point(f[1]), mesh_.point(f[2]),
                           mesh_.point(apex));
  }

  bool face_encroached(int32_t t, int slot) const {
    if (facet_size_ratio(t, slot) > 1.0) {
      return true;
    }
    const int32_t other = mesh_.tet(t).n[slot];
    return face_encroached_from(t, slot) ||
           face_encroached_from(other, neighbor_slot(mesh_.tet(other), t));
  }

  void push_segment(std::uint64_t edge, bool forced) {
    if (!conforming() || stuck_segments_.count(edge) != 0) {
      return;
    }
    const int32_t a = edge_low(edge);
    const int32_t b = edge_high(edge);
    if (edge_size_ratio(a, b) <= 1.0) {
      if (shape_obstructed_) {
        stuck_segments_.emplace(edge, Unmet::kProtected);
        return;
      }
    }
    segments_.push_back({edge, forced});
  }

  void push_face(const FaceKey& key, bool forced) {
    if (!conforming() || stuck_faces_.count(key) != 0) {
      return;
    }
    // The located face is only consulted for the shape-obstruction verdict;
    // without that obstruction the face is queued unconditionally, so the
    // star walk of find_face is not made.
    int32_t t = -1;
    int slot = -1;
    if (shape_obstructed_ && mesh_.find_face(key.v[0], key.v[1], key.v[2], t, slot) &&
        facet_size_ratio(t, slot) <= 1.0) {
      stuck_faces_.emplace(key, Unmet::kProtected);
      return;
    }
    faces_.push_back({key, forced});
  }

  void seed_queues() {
    native_execution_charge(0);
    if (conforming()) {
      NativeVector<std::uint64_t> keys;
      keys.reserve(mesh_.segments().size());
      for (const auto& entry : mesh_.segments()) {
        native_execution_charge(0);
        keys.push_back(entry.first);
      }
      std::sort(keys.begin(), keys.end(), [&](uint64_t a, uint64_t b) {
        native_execution_charge(0);
        return a < b;
      });
      for (std::uint64_t key : keys) {
        native_execution_charge(0);
        if (edge_size_ratio(edge_low(key), edge_high(key)) > 1.0 ||
            segment_encroached(edge_low(key), edge_high(key))) {
          push_segment(key, false);
        }
      }
      const auto slots = static_cast<int32_t>(mesh_.complex().tets.size());
      for (int32_t t = 0; t < slots; ++t) {
        native_execution_charge(0);
        if (!mesh_.complex().live(t) || !mesh_.finite(t)) {
          continue;
        }
        for (int k = 0; k < 4; ++k) {
          const int32_t other = mesh_.tet(t).n[k];
          if (mesh_.complex().constraint(t, k) == kNoConstraint ||
              (mesh_.finite(other) && other < t)) {
            continue;
          }
          if (face_encroached(t, k)) {
            push_face(face_key(t, k), false);
          }
        }
      }
    }
    const auto slots = static_cast<int32_t>(mesh_.complex().tets.size());
    for (int32_t t = 0; t < slots; ++t) {
      native_execution_charge(0);
      push_tet(t);
    }
  }

  FaceKey face_key(int32_t t, int slot) const {
    int32_t f[3];
    oriented_facet(mesh_.tet(t).v, slot, f);
    return FaceKey::of(f[0], f[1], f[2]);
  }

  // Queues the unconstrained faces of live finite cell t that have a finite
  // neighbor; a face whose neighbor is a smaller cell of the same group (every
  // cell when `everywhere`) is queued once, from that neighbor.
  void queue_faces(int32_t t, std::span<const int32_t> group, bool everywhere) {
    if (!mesh_.complex().live(t) || !mesh_.finite(t)) {
      return;
    }
    for (int k = 0; k < 4; ++k) {
      native_execution_charge(0);
      const int32_t u = mesh_.tet(t).n[k];
      if (mesh_.complex().constraint(t, k) != kNoConstraint || !mesh_.finite(u) ||
          (u < t && (everywhere || std::find(group.begin(), group.end(), u) != group.end()))) {
        continue;
      }
      pending_faces_.push_back({t, mesh_.tet(t).v[k], face_key(t, k)});
    }
  }

  bool halted(int32_t status) {
    stop_status_ = status;
    return false;
  }

  // Edge flips across the face of t opposite `slot`, whose apexes are d (in
  // t) and e (beyond): a non-segment edge (a, b) of the face whose ring is
  // exactly t, its neighbor and the cell of both apexes is removed 3-2 by the
  // unique two-cell ring triangulation; when (a, b) has four cells and d, e,
  // a, b are coplanar (the degenerate crossing of de through ab), the 4-4 flip
  // replaces it by the diagonal (d, e).
  bool flip_around_edge(int32_t t, int slot, int32_t d, int32_t e) {
    int32_t f[3];
    oriented_facet(mesh_.tet(t).v, slot, f);
    for (int r = 0; r < 3; ++r) {
      const int32_t a = f[r];
      const int32_t b = f[(r + 1) % 3];
      if (mesh_.segment_source(a, b) >= 0 || !mesh_.edge_ring(a, b, ring_tets_, ring_)) {
        continue;
      }
      if ((ring_.size() == 3 && try_edge_removal(mesh_, a, b, false)) ||
          (ring_.size() == 4 && flip_four_four(a, b, d, e))) {
        return true;
      }
    }
    return false;
  }

  // The 4-4 flip of the gathered ring of (a, b) (ring_tets_[i] is (a, b,
  // ring_[i], ring_[i + 1])): its unconstrained ring faces are kept and the
  // ring quadrilateral is split by the diagonal joining d and e, the same
  // cell convention as every ring triangulation (one validated replacement).
  bool flip_four_four(int32_t a, int32_t b, int32_t d, int32_t e) {
    int at_d = -1;
    int at_e = -1;
    for (int i = 0; i < 4; ++i) {
      const int32_t vertex = ring_[static_cast<std::size_t>(i)];
      const int32_t cell = ring_tets_[static_cast<std::size_t>(i)];
      if (vertex < 0 ||
          mesh_.complex().constraint(cell, vertex_slot(mesh_.tet(cell), vertex)) != kNoConstraint) {
        return false;
      }
      at_d = vertex == d ? i : at_d;
      at_e = vertex == e ? i : at_e;
    }
    if (at_d < 0 || at_e < 0 || (at_d - at_e + 4) % 4 != 2 ||
        orient3d(mesh_.point(a), mesh_.point(b), mesh_.point(d), mesh_.point(e)) != 0 ||
        !mesh_.spend(TetWorkCost::kCell)) {
      return false;
    }
    // Ring triangles (i < k < j) sharing the diagonal {0, 2} or {1, 3}.
    static constexpr int kTriangles[2][2][3] = {{{0, 1, 2}, {0, 2, 3}}, {{0, 1, 3}, {1, 2, 3}}};
    const auto& triangles = kTriangles[std::min(at_d, at_e)];
    std::array<std::array<int32_t, 4>, 4> proposed{};
    for (int s = 0; s < 2; ++s) {
      const int32_t ri = ring_[static_cast<std::size_t>(triangles[s][0])];
      const int32_t rk = ring_[static_cast<std::size_t>(triangles[s][1])];
      const int32_t rj = ring_[static_cast<std::size_t>(triangles[s][2])];
      proposed[static_cast<std::size_t>(2 * s)] = {rk, ri, rj, a};
      proposed[static_cast<std::size_t>(2 * s + 1)] = {ri, rk, rj, b};
    }
    return mesh_.replace(ring_tets_, proposed) == EditStatus::kOk;
  }

  // Lawson flips of the queued faces (see the file comment); every cell a
  // flip creates is appended to created_ and its faces are queued. False
  // (stop status set) when the work allowance or an internal check stopped it.
  bool legalize() {
    while (!pending_faces_.empty()) {
      const PendingFace entry = pending_faces_.back();
      pending_faces_.pop_back();
      const int32_t t = entry.tet;
      if (!mesh_.complex().live(t) || !mesh_.finite(t)) {
        continue;
      }
      const int slot = vertex_slot(mesh_.tet(t), entry.apex);
      if (slot < 0 || mesh_.complex().constraint(t, slot) != kNoConstraint ||
          !(face_key(t, slot) == entry.face)) {
        continue;
      }
      const int32_t u = mesh_.tet(t).n[slot];
      if (!mesh_.finite(u)) {
        continue;
      }
      if (!mesh_.spend(TetWorkCost::kVisit)) {
        return halted(PHX_MC_CAPACITY_EXCEEDED);
      }
      const int32_t far = mesh_.tet(u).v[neighbor_slot(mesh_.tet(u), t)];
      if (!mesh_.vertex_conflicts(t, far)) {
        continue;
      }
      if (!try_face_removal(mesh_, t, slot, false) &&
          !flip_around_edge(t, slot, entry.apex, far)) {
        if (mesh_.broken()) {
          return halted(PHX_MC_INTERNAL_ERROR);
        }
        if (mesh_.work_exhausted()) {
          return halted(PHX_MC_CAPACITY_EXCEEDED);
        }
        continue;
      }
      const std::span<const int32_t> flipped = mesh_.created();
      for (int32_t cell : flipped) {
        created_.push_back(cell);
        queue_faces(cell, flipped, false);
      }
    }
    return true;
  }

  // The carrier's faces are legalized before any queue is seeded.
  bool restore_delaunay() {
    const auto slots = static_cast<int32_t>(mesh_.complex().tets.size());
    for (int32_t t = 0; t < slots; ++t) {
      queue_faces(t, {}, true);
    }
    const bool restored = legalize();
    created_.clear();
    return restored;
  }

  // New cells, legalized first (flipped cells join them), enter the queue;
  // constraints of new cells encroached by their own vertices are queued
  // for splitting.
  Step after_insert() {
    ++inserted_;
    created_.assign(mesh_.created().begin(), mesh_.created().end());
    const std::size_t inserted_cells = created_.size();
    for (std::size_t i = 0; i < inserted_cells; ++i) {
      queue_faces(created_[i], std::span<const int32_t>(created_.data(), inserted_cells),
                  false);
    }
    if (!legalize()) {
      return Step::kStop;
    }
    if (created_.size() > inserted_cells) {
      // A slot freed by one flip can be reused by a later one.
      std::sort(created_.begin(), created_.end());
      created_.erase(std::unique(created_.begin(), created_.end()), created_.end());
    }
    for (int32_t t : created_) {
      if (!mesh_.complex().live(t) || !mesh_.finite(t)) {
        continue;
      }
      push_tet(t);
      if (!conforming()) {
        continue;
      }
      const int32_t* v = mesh_.tet(t).v;
      for (int k = 0; k < 4; ++k) {
        if (mesh_.complex().constraint(t, k) != kNoConstraint && face_encroached(t, k)) {
          push_face(face_key(t, k), false);
        }
      }
      for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
          if (mesh_.segment_source(v[a], v[b]) < 0) {
            continue;
          }
          for (int c = 0; c < 4; ++c) {
            if (edge_size_ratio(v[a], v[b]) > 1.0 ||
                (c != a && c != b &&
                 encroaches_segment(mesh_.point(v[a]), mesh_.point(v[b]), mesh_.point(v[c])))) {
              push_segment(topology::undirected_edge(v[a], v[b]), false);
              break;
            }
          }
        }
      }
    }
    return Step::kContinue;
  }

  Step stop(int32_t status) {
    stop_status_ = status;
    return Step::kStop;
  }

  // Stages the constrained faces interior to the prepared cavity; the commit
  // that follows removes both sides of each, retiring the vertex triple.
  void stage_retired_crossed() {
    retiring_.clear();
    const std::size_t count = mesh_.crossed_count();
    for (std::size_t i = 0; i < count; ++i) {
      retiring_.push_back(mesh_.crossed_face(i));
    }
  }

  // The staged faces after a successful commit.
  void retire_staged() {
    retired_faces_.insert(retiring_.begin(), retiring_.end());
    retiring_.clear();
  }

  // Maps a failed prepare/commit to a stop or to the given evidence.
  Step failed(Insertion result, int32_t t, Unmet* stuck) {
    mesh_.abandon();
    switch (result) {
      case Insertion::kCapacity:
        return stop(PHX_MC_CAPACITY_EXCEEDED);
      case Insertion::kInternal:
        return stop(PHX_MC_INTERNAL_ERROR);
      default: {
        // A carrier beyond its declared source deviation has no admissible
        // representation; the refused deviation is recorded by the mesh.
        const Unmet reason =
            result == Insertion::kDeviation ? Unmet::kNonrepresentable : Unmet::kRefused;
        ++counters_.refused_insertions;
        if (t >= 0) {
          mesh_.set_slot_reason(t, reason);
        }
        if (stuck != nullptr) {
          *stuck = reason;
        }
        return Step::kContinue;
      }
    }
  }

  // Constraints around the prepared cavity that p encroaches.
  void encroached_by(const double* p, bool include_faces) {
    hit_segments_.clear();
    hit_faces_.clear();
    for (int32_t t : mesh_.cavity()) {
      const int32_t* v = mesh_.tet(t).v;
      for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
          if (v[a] >= 0 && v[b] >= 0 && mesh_.segment_source(v[a], v[b]) >= 0 &&
              encroaches_segment(mesh_.point(v[a]), mesh_.point(v[b]), p)) {
            hit_segments_.push_back(topology::undirected_edge(v[a], v[b]));
          }
        }
      }
    }
    if (include_faces) {
      for (const BoundaryFacet& face : mesh_.cavity_boundary()) {
        if (mesh_.complex().constraint(face.tet, face.slot) == kNoConstraint) {
          continue;
        }
        int32_t f[3];
        oriented_facet(mesh_.tet(face.tet).v, face.slot, f);
        if (f[0] >= 0 && f[1] >= 0 && f[2] >= 0 &&
            encroaches_face(mesh_.point(f[0]), mesh_.point(f[1]), mesh_.point(f[2]), p)) {
          hit_faces_.push_back(FaceKey::of(f[0], f[1], f[2]));
        }
      }
    }
    std::sort(hit_segments_.begin(), hit_segments_.end());
    hit_segments_.erase(std::unique(hit_segments_.begin(), hit_segments_.end()),
                        hit_segments_.end());
    std::sort(hit_faces_.begin(), hit_faces_.end());
    hit_faces_.erase(std::unique(hit_faces_.begin(), hit_faces_.end()), hit_faces_.end());
  }

  // Queues the encroached constraints for splitting; returns the reason when
  // none of them can be split (kNone when at least one was queued).
  Unmet defer_to_constraints() {
    Unmet reason = Unmet::kNone;
    bool queued = false;
    for (std::uint64_t edge : hit_segments_) {
      const auto found = stuck_segments_.find(edge);
      if (found != stuck_segments_.end()) {
        reason = found->second;
        continue;
      }
      push_segment(edge, true);
      const auto refused = stuck_segments_.find(edge);
      if (refused == stuck_segments_.end()) {
        queued = true;
      } else {
        reason = refused->second;
      }
    }
    for (const FaceKey& key : hit_faces_) {
      const auto found = stuck_faces_.find(key);
      if (found != stuck_faces_.end()) {
        reason = found->second;
        continue;
      }
      push_face(key, true);
      const auto refused = stuck_faces_.find(key);
      if (refused == stuck_faces_.end()) {
        queued = true;
      } else {
        reason = refused->second;
      }
    }
    if (queued) {
      ++counters_.encroachment_deferrals;
      return Unmet::kNone;
    }
    return reason == Unmet::kNone ? Unmet::kRefused : reason;
  }

  Step refine_tet(const TetEntry& entry) {
    const int32_t t = entry.slot;
    double priority = 0.0;
    double ratio = 0.0;
    double size_ratio = 0.0;
    Circumcenter center;
    if (!current(entry) || mesh_.slot_reason(t) != Unmet::kNone ||
        !assess(t, priority, ratio, size_ratio, &center)) {
      return Step::kContinue;
    }
    if (ratio > options_.radius_edge_bound && size_ratio <= 1.0 && shape_obstructed_) {
      mesh_.set_slot_reason(t, Unmet::kProtected);
      return Step::kContinue;
    }
    // The circumcenter assess just constructed from the same corners.
    const double* c = center.point;
    if (!center.valid || !geometry::point_in_domain(c)) {
      mesh_.set_slot_reason(t, Unmet::kNonrepresentable);
      return Step::kContinue;
    }
    if (protected_point(c, mesh_.tet(t).v, 4)) {
      mesh_.set_slot_reason(t, Unmet::kProtected);
      return Step::kContinue;
    }
    int32_t blocked_tet = -1;
    int blocked_slot = -1;
    const int32_t located = mesh_.locate(c, t, blocked_tet, blocked_slot);
    if (located < 0) {
      if (blocked_tet < 0) {
        return stop(PHX_MC_CAPACITY_EXCEEDED);
      }
      return blocked(entry, blocked_tet, blocked_slot, c);
    }
    const Insertion prepared = mesh_.prepare(c, located);
    if (prepared != Insertion::kOk) {
      return failed(prepared, t, nullptr);
    }
    encroached_by(c, true);
    if (!hit_segments_.empty() || !hit_faces_.empty()) {
      mesh_.abandon();
      if (!conforming()) {
        mesh_.set_slot_reason(t, Unmet::kFixedBoundary);
        return Step::kContinue;
      }
      const Unmet reason = defer_to_constraints();
      if (reason == Unmet::kNone) {
        tets_.push(entry);
      } else {
        mesh_.set_slot_reason(t, reason);
      }
      return Step::kContinue;
    }
    if (cavity_protected(c)) {
      mesh_.abandon();
      mesh_.set_slot_reason(t, Unmet::kProtected);
      return Step::kContinue;
    }
    if (!empty_ball(t)) {
      mesh_.abandon();
      if (mesh_.work_exhausted()) {
        return stop(PHX_MC_CAPACITY_EXCEEDED);
      }
      ++counters_.refused_insertions;
      mesh_.set_slot_reason(t, Unmet::kRefused);
      return Step::kContinue;
    }
    const double size = mesh_.interpolated_size(located, c);
    stage_retired_crossed();
    const Insertion committed = mesh_.commit(InsertKind::kInterior, size);
    if (committed != Insertion::kOk) {
      return failed(committed, t, nullptr);
    }
    retire_staged();
    ++counters_.circumcenter_insertions;
    return after_insert();
  }

  // Delaunay refinement's separation premise for the circumcenter of t: t is
  // in the prepared cavity and no vertex the insertion connects to lies
  // strictly inside t's circumsphere, so every new edge is at least the
  // circumradius long. A constrained Delaunay mesh always satisfies it; next
  // to a face no flip admits it can fail, and the center would then land
  // beside an existing vertex. False also when the work allowance stops it.
  bool empty_ball(int32_t t) {
    if (!mesh_.in_prepared_cavity(t)) {
      return false;
    }
    ball_vertices_.clear();
    for (int32_t cell : mesh_.cavity()) {
      for (int32_t v : mesh_.tet(cell).v) {
        if (v >= 0) {
          ball_vertices_.push_back(v);
        }
      }
    }
    std::sort(ball_vertices_.begin(), ball_vertices_.end());
    ball_vertices_.erase(std::unique(ball_vertices_.begin(), ball_vertices_.end()),
                         ball_vertices_.end());
    for (int32_t v : ball_vertices_) {
      if (!mesh_.spend(TetWorkCost::kVisit) || mesh_.vertex_conflicts(t, v)) {
        return false;
      }
    }
    return true;
  }

  // The walk toward c stopped at constrained face (bt, bs): c lies beyond it.
  Step blocked(const TetEntry& entry, int32_t bt, int bs, const double* c) {
    const int32_t t = entry.slot;
    if (!conforming()) {
      mesh_.set_slot_reason(t, Unmet::kFixedBoundary);
      return Step::kContinue;
    }
    int32_t f[3];
    oriented_facet(mesh_.tet(bt).v, bs, f);
    hit_segments_.clear();
    hit_faces_.clear();
    if (encroaches_face(mesh_.point(f[0]), mesh_.point(f[1]), mesh_.point(f[2]), c)) {
      hit_faces_.push_back(FaceKey::of(f[0], f[1], f[2]));
    }
    const Unmet reason = defer_to_constraints();
    if (reason == Unmet::kNone) {
      tets_.push(entry);
    } else {
      mesh_.set_slot_reason(t, reason);
    }
    return Step::kContinue;
  }

  // Shell fraction from the lone corner endpoint of segment (a, b): a
  // power-of-two distance near half its length (concentric shells).
  double shell_fraction(int32_t a, int32_t b) const {
    const double length = std::sqrt(geometry::squared_distance(mesh_.point(a), mesh_.point(b)));
    double distance = std::ldexp(1.0, static_cast<int>(std::lround(std::log2(length / 2.0))));
    if (distance >= 2.0 * length / 3.0) {
      distance /= 2.0;
    } else if (distance <= length / 3.0) {
      distance *= 2.0;
    }
    return distance / length;
  }

  // Split fraction of segment (a, b) measured from a.
  double split_fraction(int32_t a, int32_t b) const {
    const bool corner_a = mesh_.dimension(a) == 0;
    if (corner_a == (mesh_.dimension(b) == 0)) {
      return 0.5;
    }
    const double fraction = shell_fraction(a, b);
    return corner_a ? fraction : 1.0 - fraction;
  }

  // Exact split point of segment (a, b) on the shell of a lone corner
  // endpoint, else the default exact dyadic; false unless it lies exactly
  // inside the segment.
  bool split_point(int32_t a, int32_t b, double* m) {
    const double* pa = mesh_.point(a);
    const double* pb = mesh_.point(b);
    const bool corner_a = mesh_.dimension(a) == 0;
    if (corner_a != (mesh_.dimension(b) == 0)) {
      const double* origin = corner_a ? pa : pb;
      const double* target = corner_a ? pb : pa;
      const double fraction = shell_fraction(a, b);
      for (int i = 0; i < 3; ++i) {
        m[i] = origin[i] + fraction * (target[i] - origin[i]);
        if (origin[i] == target[i]) {
          m[i] = origin[i];
        }
      }
      if (exactly_inside_segment(pa, pb, m)) {
        return true;
      }
    }
    return mesh_.construct_edge_point(a, b, m);
  }

  // Bisects constrained edge (a, b) at the ancestry-backed carrier of its
  // source row. Like exact subfacet splits (and unlike subsegment splits), a
  // facet carrier encroaching a subsegment of the edge star defers to it.
  // `stuck` receives the reason when the edge stays unsplit; `split` whether
  // the split was committed (its legalization may still stop the run).
  Step split_bounded(int32_t a, int32_t b, double fraction, double size, bool facet,
                     Unmet& stuck, bool& deferred, bool& split) {
    stuck = Unmet::kNone;
    deferred = false;
    split = false;
    double m[3];
    SourceWitness witness;
    const Insertion constructed = mesh_.construct_source_split(a, b, fraction, m, witness);
    if (constructed == Insertion::kCapacity) {
      return stop(PHX_MC_CAPACITY_EXCEEDED);
    }
    if (constructed != Insertion::kOk) {
      return failed(constructed, -1, &stuck);
    }
    if (!mesh_.edge_ring(a, b, ring_tets_, ring_)) {
      return stop(PHX_MC_INTERNAL_ERROR);
    }
    ring_.push_back(a);
    ring_.push_back(b);
    if (protected_point(m, ring_.data(), static_cast<int>(ring_.size()))) {
      stuck = Unmet::kProtected;
      return Step::kContinue;
    }
    hit_segments_.clear();
    hit_faces_.clear();
    for (int32_t t : facet ? std::span<const int32_t>(ring_tets_) : std::span<const int32_t>()) {
      if (!mesh_.finite(t)) {
        continue;
      }
      const int32_t* v = mesh_.tet(t).v;
      for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
          const std::uint64_t edge = topology::undirected_edge(v[i], v[j]);
          if (edge != topology::undirected_edge(a, b) && mesh_.segment_source(v[i], v[j]) >= 0 &&
              encroaches_segment(mesh_.point(v[i]), mesh_.point(v[j]), m)) {
            hit_segments_.push_back(edge);
          }
        }
      }
    }
    std::sort(hit_segments_.begin(), hit_segments_.end());
    hit_segments_.erase(std::unique(hit_segments_.begin(), hit_segments_.end()),
                        hit_segments_.end());
    if (!hit_segments_.empty()) {
      stuck = defer_to_constraints();
      deferred = stuck == Unmet::kNone;
      return Step::kContinue;
    }
    int32_t inserted = -1;
    const Insertion committed = mesh_.split_edge_bounded(a, b, m, witness, size, inserted);
    if (committed != Insertion::kOk) {
      return failed(committed, -1, &stuck);
    }
    split = true;
    return after_insert();
  }

  Step split_segment(const SegmentEntry& entry) {
    const int32_t a = edge_low(entry.edge);
    const int32_t b = edge_high(entry.edge);
    if (mesh_.segment_source(a, b) < 0 || stuck_segments_.count(entry.edge) != 0) {
      return Step::kContinue;
    }
    if (!mesh_.edge_ring(a, b, ring_tets_, ring_)) {
      return stop(PHX_MC_INTERNAL_ERROR);
    }
    bool encroached = entry.forced || edge_size_ratio(a, b) > 1.0;
    int32_t seed = -1;
    for (std::size_t i = 0; i < ring_.size(); ++i) {
      if (ring_[i] >= 0 && encroaches_segment(mesh_.point(a), mesh_.point(b), mesh_.point(ring_[i]))) {
        encroached = true;
      }
      if (seed < 0 && mesh_.finite(ring_tets_[i])) {
        seed = ring_tets_[i];
      }
    }
    if (!encroached || seed < 0) {
      return Step::kContinue;
    }
    double m[3];
    if (mesh_.witness(a).deviation > 0.0 || mesh_.witness(b).deviation > 0.0 ||
        !split_point(a, b, m)) {
      if (mesh_.work_exhausted()) {
        return stop(PHX_MC_CAPACITY_EXCEEDED);
      }
      Unmet stuck = Unmet::kNone;
      bool deferred = false;
      bool split = false;
      const Step step = split_bounded(a, b, split_fraction(a, b),
                                      0.5 * (mesh_.size(a) + mesh_.size(b)), false, stuck,
                                      deferred, split);
      if (stuck != Unmet::kNone) {
        stuck_segments_.emplace(entry.edge, stuck);
      } else if (split) {
        ++counters_.subsegment_splits;
      }
      return step;
    }
    ring_.push_back(a);
    ring_.push_back(b);
    if (protected_point(m, ring_.data(), static_cast<int>(ring_.size()))) {
      stuck_segments_.emplace(entry.edge, Unmet::kProtected);
      return Step::kContinue;
    }
    Insertion result = mesh_.prepare(m, seed, a, b);
    if (result == Insertion::kOk) {
      stage_retired_crossed();
      result = mesh_.commit(InsertKind::kSubsegment, 0.5 * (mesh_.size(a) + mesh_.size(b)));
    }
    if (result == Insertion::kRefused) {
      mesh_.abandon();
      // The refused commit wrote nothing, so ring_ is still the star of (a, b)
      // that the edge split replaces completely: every face (a, b, r) goes.
      retiring_.clear();
      for (std::size_t i = 0; i < ring_tets_.size(); ++i) {
        if (ring_[i] >= 0) {
          retiring_.push_back(FaceKey::of(a, b, ring_[i]));
        }
      }
      int32_t inserted = -1;
      result = mesh_.split_edge(a, b, m, 0.5 * (mesh_.size(a) + mesh_.size(b)), inserted);
    }
    if (result != Insertion::kOk) {
      Unmet reason = Unmet::kNone;
      const Step step = failed(result, -1, &reason);
      if (reason != Unmet::kNone) {
        stuck_segments_.emplace(entry.edge, reason);
      }
      return step;
    }
    retire_staged();
    ++counters_.subsegment_splits;
    return after_insert();
  }

  // Inserts the prepared split point of a subfacet unless it encroaches a
  // subsegment (queued instead) or a protecting ball.
  Step commit_face_split(const FaceEntry& entry, const double* p, double size) {
    encroached_by(p, false);
    if (!hit_segments_.empty()) {
      mesh_.abandon();
      const Unmet reason = defer_to_constraints();
      if (reason == Unmet::kNone) {
        faces_.push_back({entry.key, true});
      } else {
        stuck_faces_.emplace(entry.key, reason);
      }
      return Step::kContinue;
    }
    if (cavity_protected(p)) {
      mesh_.abandon();
      stuck_faces_.emplace(entry.key, Unmet::kProtected);
      return Step::kContinue;
    }
    stage_retired_crossed();
    const Insertion committed = mesh_.commit(InsertKind::kSubfacet, size);
    if (committed != Insertion::kOk) {
      Unmet reason = Unmet::kNone;
      const Step step = failed(committed, -1, &reason);
      if (reason != Unmet::kNone) {
        stuck_faces_.emplace(entry.key, reason);
      }
      return step;
    }
    retire_staged();
    ++counters_.subfacet_splits;
    return after_insert();
  }

  struct FaceSplitGeometry {
    int32_t cell;
    int slot;
    const double* x;
    const double* y;
    const double* z;
    double center[3];
    bool bounded_face;
    bool oversized;
    // The triangle circumcenter and its exact coplanarity, constructed once
    // for both center routes (they are mutually exclusive on `coplanar`).
    bool centered = false;
    bool coplanar = false;

    FaceSplitGeometry(TetMesh& mesh, int32_t t, int s, const int32_t* key, bool too_large)
        : cell(t), slot(s), x(mesh.point(key[0])), y(mesh.point(key[1])), z(mesh.point(key[2])),
          bounded_face(mesh.witness(key[0]).deviation > 0.0 ||
                       mesh.witness(key[1]).deviation > 0.0 ||
                       mesh.witness(key[2]).deviation > 0.0),
          oversized(too_large) {
      centered = !oversized && geometry::triangle_circumcenter(x, y, z, center) &&
                 geometry::point_in_domain(center);
      coplanar = centered && !bounded_face && orient3d(x, y, z, center) == 0;
    }
  };

  // Both sides of the queued facet lie in the prepared cavity, so the
  // insertion retires the facet itself (O(1) per side from the cavity stamp).
  bool retires_facet(const FaceSplitGeometry& state) const {
    return mesh_.in_prepared_cavity(state.cell) &&
           mesh_.in_prepared_cavity(mesh_.tet(state.cell).n[state.slot]);
  }

  std::optional<Step> prepare_source_face_center(const FaceEntry& entry,
                                                FaceSplitGeometry& state) {
    const int32_t t = state.cell;
    const int slot = state.slot;
    const double* c = state.center;
    if (state.centered && (state.bounded_face || !state.coplanar)) {
      double carrier[3];
      SourceWitness witness;
      const Insertion prepared = mesh_.prepare_source_facet(t, slot, c, carrier, witness);
      if (prepared == Insertion::kOk && retires_facet(state)) {
        return commit_face_split(entry, carrier, mesh_.interpolated_size(t, carrier));
      }
      mesh_.abandon();
      if (prepared == Insertion::kCapacity || prepared == Insertion::kInternal) {
        return failed(prepared, -1, nullptr);
      }
      if (prepared == Insertion::kDeviation) {
        stuck_faces_.emplace(entry.key, Unmet::kNonrepresentable);
        return Step::kContinue;
      }
    }
    return std::nullopt;
  }

  std::optional<Step> prepare_exact_face_center(const FaceEntry& entry,
                                               FaceSplitGeometry& state) {
    const int32_t t = state.cell;
    const double* c = state.center;
    if (state.coplanar) {
      int32_t blocked_tet = -1;
      int blocked_slot = -1;
      const int32_t located = mesh_.locate(c, t, blocked_tet, blocked_slot);
      if (located < 0 && blocked_tet < 0) {
        return stop(PHX_MC_CAPACITY_EXCEEDED);
      }
      if (located >= 0) {
        const Insertion prepared = mesh_.prepare(c, located);
        if (prepared == Insertion::kOk && retires_facet(state)) {
          // An obtuse facet's center can lie in another same-plane facet.
          // Accept its Delaunay patch only when it actually retires both sides
          // of the queued source facet, not just a remote triangle.
          return commit_face_split(entry, c, mesh_.interpolated_size(located, c));
        }
        mesh_.abandon();
        if (prepared == Insertion::kCapacity || prepared == Insertion::kInternal) {
          return failed(prepared, -1, nullptr);
        }
      }
    }
    return std::nullopt;
  }

  Step split_face_longest_edge(const FaceEntry& entry, int32_t t) {
    const int32_t* key = entry.key.v;
    // An arbitrary exact interior point is not a circumcenter: repeated star
    // subdivision can create ever smaller protected angles on a source plane.
    // If no exact circumcenter transaction retires this facet, bisect its
    // longest edge instead; do not refine an unrelated facet reached by a walk.
    int longest = 0;
    double best = -1.0;
    for (int r = 0; r < 3; ++r) {
      const double length = geometry::squared_distance(mesh_.point(key[r]), mesh_.point(key[(r + 1) % 3]));
      if (length > best) {
        best = length;
        longest = r;
      }
    }
    const int32_t u = key[longest];
    const int32_t w = key[(longest + 1) % 3];
    if (mesh_.segment_source(u, w) >= 0) {
      hit_segments_.assign(1, topology::undirected_edge(u, w));
      hit_faces_.clear();
      const Unmet reason = defer_to_constraints();
      if (reason == Unmet::kNone) {
        faces_.push_back({entry.key, true});
      } else {
        stuck_faces_.emplace(entry.key, reason);
      }
      return Step::kContinue;
    }
    double m[3];
    if (mesh_.witness(u).deviation > 0.0 || mesh_.witness(w).deviation > 0.0 ||
        !mesh_.construct_edge_point(u, w, m)) {
      if (mesh_.work_exhausted()) {
        return stop(PHX_MC_CAPACITY_EXCEEDED);
      }
      Unmet stuck = Unmet::kNone;
      bool deferred = false;
      bool split = false;
      const Step step = split_bounded(u, w, 0.5, 0.5 * (mesh_.size(u) + mesh_.size(w)), true,
                                      stuck, deferred, split);
      if (deferred) {
        faces_.push_back({entry.key, true});
      } else if (stuck != Unmet::kNone) {
        stuck_faces_.emplace(entry.key, stuck);
      } else if (split) {
        ++counters_.subfacet_splits;
      }
      return step;
    }
    const Insertion prepared = mesh_.prepare(m, t);
    if (prepared != Insertion::kOk) {
      Unmet reason = Unmet::kNone;
      const Step step = failed(prepared, -1, &reason);
      if (reason != Unmet::kNone) {
        stuck_faces_.emplace(entry.key, reason);
      }
      return step;
    }
    return commit_face_split(entry, m, 0.5 * (mesh_.size(u) + mesh_.size(w)));
  }

  Step split_face(const FaceEntry& entry) {
    const int32_t* key = entry.key.v;
    int32_t t = -1;
    int slot = -1;
    // A retired vertex triple can never be a face again (every later face
    // contains a later vertex), so find_face would walk the star of key[0]
    // only to fail; the retired set gives that same answer without the walk.
    if (stuck_faces_.count(entry.key) != 0 || retired_faces_.count(entry.key) != 0 ||
        !mesh_.find_face(key[0], key[1], key[2], t, slot) ||
        mesh_.complex().constraint(t, slot) == kNoConstraint) {
      return Step::kContinue;
    }
    if (!entry.forced && !face_encroached(t, slot)) {
      return Step::kContinue;
    }
    FaceSplitGeometry state(mesh_, t, slot, key, facet_size_ratio(t, slot) > 1.0);
    if (const auto step = prepare_source_face_center(entry, state)) return *step;
    if (const auto step = prepare_exact_face_center(entry, state)) return *step;
    return split_face_longest_edge(entry, t);
  }

  void record_unmet() {
    auto& unmet = mesh_.unmet();
    unmet.clear();
    const auto slots = static_cast<int32_t>(mesh_.complex().tets.size());
    for (int32_t t = 0; t < slots; ++t) {
      native_execution_charge(0);
      if (!mesh_.complex().live(t) || !mesh_.finite(t)) {
        continue;
      }
      double priority = 0.0;
      double ratio = 0.0;
      double size_ratio = 0.0;
      if (!assess(t, priority, ratio, size_ratio)) {
        continue;
      }
      UnmetRecord record{};
      std::copy_n(mesh_.tet(t).v, 4, record.v);
      canonicalize_cell(record.v, nullptr, 4);
      const bool shape = ratio > options_.radius_edge_bound;
      record.criterion =
          shape ? PHX_MC_TET_MESH_CRITERION_RADIUS_EDGE : PHX_MC_TET_MESH_CRITERION_SIZE;
      record.value = shape ? ratio : size_ratio;
      const Unmet reason = mesh_.slot_reason(t);
      record.reason =
          static_cast<int32_t>(reason == Unmet::kNone ? boundary_reason(t) : reason);
      unmet.push_back(record);
    }
    std::sort(unmet.begin(), unmet.end(), [&](const UnmetRecord& x, const UnmetRecord& y) {
      native_execution_charge(0);
      return std::lexicographical_compare(x.v, x.v + 4, y.v, y.v + 4);
    });
  }

  TetMesh& mesh_;
  RefineOptions options_;
  TetMeshRefineCounters counters_;
  int64_t inserted_ = 0;
  int32_t stop_status_ = PHX_MC_OK;
  NativeDeque<SegmentEntry> segments_;
  NativeDeque<FaceEntry> faces_;
  std::priority_queue<TetEntry, NativeVector<TetEntry>> tets_;
  NativeUnorderedMap<std::uint64_t, Unmet> stuck_segments_;
  NativeUnorderedMap<FaceKey, Unmet, FaceKeyHash> stuck_faces_;
  NativeVector<int32_t> ring_tets_;
  NativeVector<int32_t> ring_;
  NativeVector<int32_t> created_;
  NativeVector<PendingFace> pending_faces_;
  NativeVector<int32_t> ball_vertices_;
  NativeVector<std::uint64_t> hit_segments_;
  NativeVector<FaceKey> hit_faces_;
  // Constrained faces a pending commit retires (staged before the commit,
  // published only when it succeeds) and every face retired so far.
  NativeVector<FaceKey> retiring_;
  NativeUnorderedSet<FaceKey, FaceKeyHash> retired_faces_;
  bool shape_obstructed_ = false;
};

}  // namespace

int32_t refine_mesh(TetMesh& mesh, const RefineOptions& options, TetMeshRefineCounters& counters) {
  const MemoryScope memory_scope(mesh.memory_owner());
  Refiner refiner(mesh, options);
  return refiner.run(counters);
}

}  // namespace phx::mc

extern "C" {

int32_t phx_mc_tet_mesh_refine(phx_mc_tet_mesh* handle, const double* sizes,
                               double radius_edge_bound, int64_t max_insertions,
                               int64_t work_limit, int64_t* counters) {
  return phx::mc::guarded([&]() -> int32_t {
    if (handle == nullptr || handle->mesh == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::TetMesh& mesh = *handle->mesh;
    const phx::mc::MemoryScope memory_scope(mesh.memory_owner());
    phx::mc::MeasuredTetStage timing(mesh, phx::mc::TetExecutionStage::kRefinement);
    if (counters == nullptr || !(radius_edge_bound > 0.0) ||
        !std::isfinite(radius_edge_bound) || max_insertions < 0 || work_limit < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (mesh.broken()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    phx::mc::TetMeshRefineCounters state;
    phx::mc::CounterPublisher publisher{state, counters};
    phx::mc::TetWorkBudgetWindow budget(mesh, work_limit);
    if (sizes != nullptr) {
      const auto count = static_cast<std::size_t>(mesh.vertex_count());
      for (std::size_t v = 0; v < count; ++v) {
        phx::mc::native_execution_charge(0);
        if (!std::isfinite(sizes[v]) || sizes[v] < 0.0) {
          return PHX_MC_INVALID_INPUT;
        }
      }
      std::copy_n(sizes, count, mesh.sizes().begin());
    }
    phx::mc::RefineOptions options;
    options.radius_edge_bound = radius_edge_bound;
    options.max_insertions = max_insertions;
    const int32_t status = phx::mc::refine_mesh(mesh, options, state);
    return mesh.broken() ? PHX_MC_INTERNAL_ERROR : status;
  });
}

}  // extern "C"
