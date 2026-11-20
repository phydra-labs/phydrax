//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Shared 2D triangle data structure for Delaunay, regular and constrained
// Delaunay triangulations.
//
// Invariants:
//   * Every live triangle stores v[3] counterclockwise and n[3], the neighbor
//     across the edge opposite v[k] (edge k runs from v[k+1] to v[k+2]).
//   * The convex hull is closed by ghost triangles carrying the infinite vertex
//     kInfiniteVertex, always stored in v[2]; the finite edge v[0] -> v[1] of a
//     ghost is a hull edge seen from outside (the finite neighbor n[2] holds the
//     edge v[1] -> v[0]).  Every triangle therefore has three live neighbors.
//   * constraints[3 t + k] is the constraint id of edge k of slot t (-1 when
//     unconstrained); both triangles sharing an edge carry the same id.
//   * vertex_triangle[v] is a live triangle incident to v, or -1 when v is not
//     a vertex of the triangulation (duplicate, redundant, or removed).
// All topological decisions use exact predicates; ties in in-circle / power
// tests are broken by index-ordered symbolic perturbation with vertex ids.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "prefetch.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {

inline constexpr int32_t kInfiniteVertex = -1;

inline int next3(int k) { return k == 2 ? 0 : k + 1; }
inline int prev3(int k) { return k == 0 ? 2 : k - 1; }

struct Triangle2D {
  int32_t v[3];
  int32_t n[3];
};

class Triangulation2D {
 public:
  enum class Insertion { kInserted, kRedundant };

  std::vector<double> coordinates;      // 2 per vertex
  const double* weights = nullptr;      // per vertex (regular triangulations) or null
  // Input index of each vertex, which orders the symbolic perturbation, when
  // the coordinates are stored in another (insertion) order; null: the
  // vertex index itself.
  const int32_t* identities = nullptr;
  std::vector<Triangle2D> triangles;    // slots, live or free
  std::vector<int32_t> constraints;     // 3 per slot
  std::vector<uint8_t> region;          // per slot, owner-defined flag (CDT domain)
  std::vector<uint8_t> live;            // per slot
  std::vector<int32_t> free_slots;
  std::vector<int32_t> vertex_triangle;  // per vertex
  int64_t finite_count = 0;
  int32_t hint = -1;
  std::uint64_t walk_step = 0;

  const double* point(int32_t vertex) const {
    return coordinates.data() + 2 * static_cast<int64_t>(vertex);
  }
  int32_t vertex_count() const { return static_cast<int32_t>(coordinates.size() / 2); }
  int64_t identity(int32_t vertex) const {
    return identities == nullptr ? vertex : identities[static_cast<std::size_t>(vertex)];
  }
  bool is_ghost(int32_t t) const { return triangles[t].v[2] == kInfiniteVertex; }

  int index_of(int32_t t, int32_t vertex) const {
    const Triangle2D& tri = triangles[t];
    return tri.v[0] == vertex ? 0 : tri.v[1] == vertex ? 1 : tri.v[2] == vertex ? 2 : -1;
  }

  // Index k of the edge u -> w in triangle t (v[k+1] == u, v[k+2] == w), or -1.
  int edge_index(int32_t t, int32_t u, int32_t w) const {
    const Triangle2D& tri = triangles[t];
    for (int k = 0; k < 3; ++k) {
      if (tri.v[next3(k)] == u && tri.v[prev3(k)] == w) {
        return k;
      }
    }
    return -1;
  }

  int32_t allocate() {
    int32_t slot;
    if (!free_slots.empty()) {
      slot = free_slots.back();
      free_slots.pop_back();
    } else {
      if (triangles.size() >= static_cast<std::size_t>(kMaxMeshPoints)) {
        throw std::length_error("triangulation2d slot index exceeds int32 range");
      }
      slot = static_cast<int32_t>(triangles.size());
      triangles.push_back({{0, 0, 0}, {-1, -1, -1}});
      constraints.push_back(-1);
      constraints.push_back(-1);
      constraints.push_back(-1);
      region.push_back(0);
      live.push_back(0);
      marks_.push_back(0);
    }
    live[slot] = 1;
    return slot;
  }

  void release(int32_t t) {
    if (!is_ghost(t)) {
      --finite_count;
    }
    live[t] = 0;
    free_slots.push_back(t);
  }

  // Writes the vertices of slot t (live) with unset neighbors; constraints and
  // region are set by the caller when relevant.
  void set_triangle(int32_t t, int32_t a, int32_t b, int32_t c, int32_t ca = -1, int32_t cb = -1,
                    int32_t cc = -1) {
    Triangle2D& tri = triangles[t];
    tri.v[0] = a;
    tri.v[1] = b;
    tri.v[2] = c;
    tri.n[0] = tri.n[1] = tri.n[2] = -1;
    constraints[3 * static_cast<std::size_t>(t)] = ca;
    constraints[3 * static_cast<std::size_t>(t) + 1] = cb;
    constraints[3 * static_cast<std::size_t>(t) + 2] = cc;
    if (c != kInfiniteVertex) {
      ++finite_count;
    }
    for (int k = 0; k < 3; ++k) {
      if (tri.v[k] != kInfiniteVertex) {
        vertex_triangle[static_cast<std::size_t>(tri.v[k])] = t;
      }
    }
  }

  // Makes nb the neighbor of t across edge u -> w (both directions).
  void link_edge(int32_t t, int32_t u, int32_t w, int32_t nb) {
    const int k = edge_index(t, u, w);
    const int j = edge_index(nb, w, u);
    if (k < 0 || j < 0) {
      throw std::logic_error("triangulation2d: inconsistent adjacency");
    }
    triangles[t].n[k] = nb;
    triangles[nb].n[j] = t;
  }

  // Sets the constraint id of edge u -> w of t and of its twin.
  void set_edge_constraint(int32_t t, int32_t u, int32_t w, int32_t id) {
    const int k = edge_index(t, u, w);
    const int32_t nb = triangles[t].n[k];
    const int j = edge_index(nb, w, u);
    constraints[3 * static_cast<std::size_t>(t) + k] = id;
    constraints[3 * static_cast<std::size_t>(nb) + j] = id;
  }

  int32_t constraint(int32_t t, int k) const {
    return constraints[3 * static_cast<std::size_t>(t) + k];
  }

  // Resets storage for vertex_total vertices whose coordinates are copied.
  void reset(const double* points, int64_t vertex_total, const double* vertex_weights) {
    coordinates.assign(points, points + 2 * vertex_total);
    weights = vertex_weights;
    identities = nullptr;
    triangles.clear();
    constraints.clear();
    region.clear();
    live.clear();
    marks_.clear();
    free_slots.clear();
    vertex_triangle.assign(static_cast<std::size_t>(vertex_total), -1);
    vertex_marks_.assign(static_cast<std::size_t>(vertex_total) + 1, 0);
    vertex_slots_.assign(static_cast<std::size_t>(vertex_total) + 1, -1);
    finite_count = 0;
    hint = -1;
    walk_step = 0;
    epoch_ = 0;
    const std::size_t reserve = static_cast<std::size_t>(2 * vertex_total + 8);
    triangles.reserve(reserve);
    constraints.reserve(3 * reserve);
    region.reserve(reserve);
    live.reserve(reserve);
    marks_.reserve(reserve);
  }

  // Appends a vertex (constructed Steiner point); returns its index.
  int32_t add_vertex(const double* xy) {
    const int32_t index = vertex_count();
    coordinates.push_back(xy[0]);
    coordinates.push_back(xy[1]);
    vertex_triangle.push_back(-1);
    vertex_marks_.push_back(0);
    vertex_slots_.push_back(-1);
    return index;
  }

  // First triangle (a, b, c) with orient2d(a, b, c) > 0 and its three ghosts.
  void initialize(int32_t a, int32_t b, int32_t c) {
    const int32_t t = allocate();
    const int32_t g0 = allocate();
    const int32_t g1 = allocate();
    const int32_t g2 = allocate();
    set_triangle(t, a, b, c);
    set_triangle(g0, b, a, kInfiniteVertex);
    set_triangle(g1, c, b, kInfiniteVertex);
    set_triangle(g2, a, c, kInfiniteVertex);
    link_edge(t, a, b, g0);
    link_edge(t, b, c, g1);
    link_edge(t, c, a, g2);
    link_edge(g0, a, kInfiniteVertex, g2);
    link_edge(g1, b, kInfiniteVertex, g0);
    link_edge(g2, c, kInfiniteVertex, g1);
    hint = t;
  }

  // Exact conflict of point p (vertex id p, coordinates q) with live triangle t.
  [[gnu::always_inline]] bool conflict(int32_t t, const double* q, int32_t p) const {
    const Triangle2D& tri = triangles[t];
    if (tri.v[2] == kInfiniteVertex) {
      const int side = orient2d_filtered(point(tri.v[0]), point(tri.v[1]), q);
      if (side != 0) {
        return side > 0;
      }
      return finite_conflict(tri.n[2], q, p);
    }
    return finite_conflict(t, q, p);
  }

  [[gnu::always_inline]] bool finite_conflict(int32_t t, const double* q, int32_t p) const {
    if (weights != nullptr) return power_conflict(t, q, p);
    const Triangle2D& tri = triangles[t];
    // A sign the floating stage decides is the unperturbed sign.
    const int sign = incircle_stage(point(tri.v[0]), point(tri.v[1]), point(tri.v[2]), q);
    if (sign != 0) [[likely]] return sign > 0;
    return incircle_sos(point(tri.v[0]), point(tri.v[1]), point(tri.v[2]), q,
                        identity(tri.v[0]), identity(tri.v[1]), identity(tri.v[2]),
                        identity(p)) > 0;
  }

  bool power_conflict(int32_t t, const double* q, int32_t p) const {
    const Triangle2D& tri = triangles[t];
    return power2d_sos(point(tri.v[0]), point(tri.v[1]), point(tri.v[2]), q,
                       weights[tri.v[0]], weights[tri.v[1]], weights[tri.v[2]], weights[p],
                       identity(tri.v[0]), identity(tri.v[1]), identity(tri.v[2]),
                       identity(p)) > 0;
  }

  // Stochastic visibility walk (edge order from splitmix64 of a step counter).
  // Returns a finite triangle whose closure contains q, or the ghost triangle
  // over a hull edge that q lies strictly outside of.  When `blocked` is not
  // null the walk refuses to cross constrained edges: it then stores the
  // (triangle, edge) pair that would have been crossed and returns -1.
  int32_t walk(int32_t start, const double* q, std::pair<int32_t, int>* blocked = nullptr) {
    int32_t t = start;
    if (is_ghost(t)) {
      t = triangles[t].n[2];
    }
    int32_t previous = -1;
    for (;;) {
      const Triangle2D& tri = triangles[t];
      const int first = static_cast<int>(splitmix64(walk_step++) % 3U);
      const int crossed = exit_edge(tri, q, previous, first);
      if (crossed < 0) {
        return t;
      }
      if (blocked != nullptr && constraint(t, crossed) != -1) {
        *blocked = {t, crossed};
        return -1;
      }
      previous = t;
      t = tri.n[crossed];
      prefetch(&triangles[static_cast<std::size_t>(t)]);
      if (is_ghost(t)) {
        return t;
      }
    }
  }

  // The edge a walk step leaves finite triangle `tri` through: the first k
  // of first, first + 1, first + 2 (mod 3) whose neighbor is not `previous`
  // and with orient2d(v[k+1], v[k+2], q) < 0, else -1. The three
  // orientations are evaluated at once (each exactly as orient2d_filtered
  // does; exact predicates have no side effects), so the choice costs no
  // data-dependent branch.
  int exit_edge(const Triangle2D& tri, const double* q, int32_t previous, int first) const {
    const double* a = point(tri.v[0]);
    const double* b = point(tri.v[1]);
    const double* c = point(tri.v[2]);
    const double ax = a[0] - q[0], ay = a[1] - q[1];
    const double bx = b[0] - q[0], by = b[1] - q[1];
    const double cx = c[0] - q[0], cy = c[1] - q[1];
    // Edge k runs from v[k+1] to v[k+2]: (b, c), (c, a), (a, b).
    const double left[3] = {bx * cy, cx * ay, ax * by};
    const double right[3] = {by * cx, cy * ax, ay * bx};
    constexpr double kEpsilon = 0x1p-53;
    constexpr double kBound = (3.0 + 16.0 * kEpsilon) * kEpsilon;
    unsigned negative = 0;
    unsigned uncertain = 0;
    for (int k = 0; k < 3; ++k) {
      const double det = left[k] - right[k];
      const double errbound = kBound * (std::fabs(left[k]) + std::fabs(right[k]));
      negative |= static_cast<unsigned>(-det > errbound) << k;
      uncertain |= static_cast<unsigned>(!(det > errbound) && !(-det > errbound)) << k;
    }
    if (uncertain != 0) [[unlikely]] {
      for (int k = 0; k < 3; ++k) {
        if (((uncertain >> k) & 1U) != 0 &&
            orient2d(point(tri.v[next3(k)]), point(tri.v[prev3(k)]), q) < 0) {
          negative |= 1U << k;
        }
      }
    }
    negative &= static_cast<unsigned>(tri.n[0] != previous) |
                static_cast<unsigned>(tri.n[1] != previous) << 1 |
                static_cast<unsigned>(tri.n[2] != previous) << 2;
    // kExit[first][mask]: the first set bit of mask from `first` on.
    static constexpr int8_t kExit[3][8] = {{-1, 0, 1, 0, 2, 0, 1, 0},
                                           {-1, 0, 1, 1, 2, 2, 1, 1},
                                           {-1, 0, 1, 0, 2, 2, 2, 2}};
    return kExit[first][negative];
  }

  // Bowyer-Watson insertion of vertex p, distinct from every vertex
  // (unconstrained triangulations). The cone triangles reuse the cavity's
  // slots last-removed first, then free slots, then new ones (as releasing
  // the cavity and allocating would).
  Insertion insert(int32_t p) {
    const double* q = point(p);
    const int32_t start = walk(hint, q);
    // Unweighted, the walk's triangle conflicts: q lies in the closure of a
    // finite one (not at a vertex, so strictly inside its circumcircle) or
    // strictly outside a ghost's hull edge.
    if (weights != nullptr && !conflict(start, q, p)) {
      return Insertion::kRedundant;
    }
    epoch_ += 2;
    const std::uint64_t inside = epoch_;
    const std::uint64_t outside = epoch_ + 1;
    // Scratch arrays are sized ahead (a popped triangle adds at most three
    // stack entries and boundary edges) and addressed through locals.
    std::size_t top = 0, removed = 0, edges = 0;
    int32_t* stack = stack_.data();
    int32_t* cavity = cavity_.data();
    BoundaryEdge* boundary = boundary_.data();
    const auto reserve_scratch = [&]() {
      if (top + 4 <= stack_.size() && removed + 1 <= cavity_.size() &&
          edges + 3 <= boundary_.size()) [[likely]] {
        return;
      }
      stack_.resize(std::max(stack_.size(), 2 * (top + 4)));
      cavity_.resize(std::max(cavity_.size(), 2 * (removed + 1)));
      boundary_.resize(std::max(boundary_.size(), 2 * (edges + 3)));
      stack = stack_.data();
      cavity = cavity_.data();
      boundary = boundary_.data();
    };
    reserve_scratch();
    stack[top++] = start;
    marks_[start] = inside;
    int64_t finite_removed = 0;
    while (top != 0) {
      reserve_scratch();
      const int32_t t = stack[--top];
      cavity[removed++] = t;
      const Triangle2D& tri = triangles[t];
      finite_removed += tri.v[2] != kInfiniteVertex ? 1 : 0;
      for (int k = 0; k < 3; ++k) {
        const int32_t nb = tri.n[k];
        const std::uint64_t mark = marks_[nb];
        if (mark == inside) {
          continue;
        }
        if (mark != outside && conflict(nb, q, p)) {
          marks_[nb] = inside;
          // Its neighbors are read when it leaves the stack.
          for (int32_t next : triangles[nb].n) {
            if (static_cast<std::size_t>(next) < triangles.size()) {
              prefetch(&triangles[static_cast<std::size_t>(next)]);
              prefetch(&marks_[static_cast<std::size_t>(next)]);
            }
          }
          stack[top++] = nb;
        } else {
          marks_[nb] = outside;
          // The slot of nb facing t (two triangles share at most one edge).
          const int32_t* around = triangles[nb].n;
          const int back = around[0] == t ? 0 : around[1] == t ? 1 : 2;
          boundary[edges++] = {tri.v[next3(k)], tri.v[prev3(k)], nb, back};
        }
      }
    }
    if (weights != nullptr) {
      // Vertices strictly inside the cavity lose their whole star.
      for (std::size_t e = 0; e < edges; ++e) {
        vertex_marks_[static_cast<std::size_t>(boundary[e].u + 1)] = epoch_;
      }
      for (std::size_t c = 0; c < removed; ++c) {
        for (int32_t vertex : triangles[cavity[c]].v) {
          if (vertex != kInfiniteVertex &&
              vertex_marks_[static_cast<std::size_t>(vertex + 1)] != epoch_) {
            vertex_marks_[static_cast<std::size_t>(vertex + 1)] = epoch_;
            vertex_triangle[static_cast<std::size_t>(vertex)] = -1;
          }
        }
      }
    }
    finite_count -= finite_removed;
    const std::size_t reused = std::min(removed, edges);
    for (std::size_t c = 0; c < removed - reused; ++c) {
      live[static_cast<std::size_t>(cavity[c])] = 0;
      free_slots.push_back(cavity[c]);
    }
    if (created_.size() < edges) created_.resize(2 * edges);
    int32_t* created = created_.data();
    int32_t ghosts[2] = {-1, -1};
    int ghost_count = 0;
    for (std::size_t e = 0; e < edges; ++e) {
      const BoundaryEdge& edge = boundary[e];
      const int32_t t = e < reused ? cavity[removed - 1 - e] : allocate();
      created[e] = t;
      Triangle2D& tri = triangles[t];
      tri.v[0] = edge.u;
      tri.v[1] = edge.w;
      tri.v[2] = p;
      tri.n[2] = edge.outside;
      triangles[edge.outside].n[edge.back] = t;
      vertex_slots_[static_cast<std::size_t>(edge.u + 1)] = t;
      int32_t* constraint_row = constraints.data() + 3 * static_cast<std::size_t>(t);
      constraint_row[0] = constraint_row[1] = constraint_row[2] = -1;
      if (edge.u == kInfiniteVertex || edge.w == kInfiniteVertex) {
        ghosts[ghost_count++ & 1] = t;
      }
    }
    // New triangle (u, w, p) meets (w, x, p) across w - p. Each vertex's
    // triangle is the last new one around it.
    Triangle2D* slots = triangles.data();
    for (std::size_t e = 0; e < edges; ++e) {
      const int32_t t = created[e];
      Triangle2D& tri = slots[t];
      const int32_t other = vertex_slots_[static_cast<std::size_t>(tri.v[1] + 1)];
      tri.n[0] = other;
      slots[other].n[1] = t;
      const bool finite_u = tri.v[0] != kInfiniteVertex;
      const bool finite_w = tri.v[1] != kInfiniteVertex;
      if (finite_u) vertex_triangle[static_cast<std::size_t>(tri.v[0])] = t;
      if (finite_w) vertex_triangle[static_cast<std::size_t>(tri.v[1])] = t;
      if (finite_u && finite_w) {
        ++finite_count;
        hint = t;
      }
    }
    vertex_triangle[static_cast<std::size_t>(p)] = created[edges - 1];
    // Ghosts keep the infinite vertex in v[2].
    for (int g = 0; g < ghost_count; ++g) {
      Triangle2D& tri = slots[ghosts[g]];
      if (tri.v[0] == kInfiniteVertex) {
        tri = {{tri.v[1], tri.v[2], tri.v[0]}, {tri.n[1], tri.n[2], tri.n[0]}};
      } else {
        tri = {{tri.v[2], tri.v[0], tri.v[1]}, {tri.n[2], tri.n[0], tri.n[1]}};
      }
    }
    return Insertion::kInserted;
  }

  // Finite live triangles as a flat CCW array of vertex identities.
  void finite_cells(std::vector<int32_t>& cells) const {
    cells.resize(static_cast<std::size_t>(3 * finite_count));
    int32_t* out = cells.data();
    for (std::size_t t = 0; t < triangles.size(); ++t) {
      const Triangle2D& tri = triangles[t];
      if (live[t] != 0 && tri.v[2] != kInfiniteVertex) {
        for (int k = 0; k < 3; ++k) out[k] = static_cast<int32_t>(identity(tri.v[k]));
        out += 3;
      }
    }
  }

 private:
  struct BoundaryEdge {
    int32_t u;
    int32_t w;
    int32_t outside;
    int32_t back;  // slot of `outside` across the edge
  };

  std::vector<std::uint64_t> marks_;         // per slot, cavity epochs
  std::vector<std::uint64_t> vertex_marks_;  // per vertex + 1 (slot 0 = infinite vertex)
  std::vector<int32_t> vertex_slots_;        // per vertex + 1, scratch
  std::uint64_t epoch_ = 0;
  std::vector<int32_t> stack_;
  std::vector<int32_t> cavity_;
  std::vector<int32_t> created_;
  std::vector<BoundaryEdge> boundary_;
};

// Builds the (regular) Delaunay triangulation of the vertices `order` lists
// (distinct points, in insertion order) into `triangulation`, whose
// coordinates must already hold them. Returns DEGENERATE_INPUT when they are
// collinear.
inline int32_t build_ordered_delaunay_2d(Triangulation2D& triangulation,
                                         std::span<const int32_t> order) {
  if (order.size() < 3) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  const int32_t a = order[0];
  int32_t b = order[1];
  std::size_t third = 2;
  int side = 0;
  for (; third < order.size(); ++third) {
    side = orient2d(triangulation.point(a), triangulation.point(b),
                    triangulation.point(order[third]));
    if (side != 0) {
      break;
    }
  }
  if (third == order.size()) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  int32_t c = order[third];
  if (side < 0) {
    std::swap(b, c);
  }
  triangulation.initialize(a, b, c);
  for (std::size_t k = 2; k < order.size(); ++k) {
    if (k != third) {
      triangulation.insert(order[k]);
    }
  }
  return PHX_MC_OK;
}

// Deduplicates, orders and builds the (regular) Delaunay triangulation of the
// representatives into `triangulation` (whose coordinates must already hold
// the input points). Returns DEGENERATE_INPUT when the representatives are
// collinear.
inline int32_t build_delaunay_2d(Triangulation2D& triangulation, const double* points,
                                 int64_t point_count, const double* weights,
                                 std::vector<int32_t>& vertex_map) {
  const std::vector<int32_t> order =
      deduplicated_insertion_order(points, point_count, 2, weights, vertex_map);
  return build_ordered_delaunay_2d(triangulation, order);
}

}  // namespace phx::mc
