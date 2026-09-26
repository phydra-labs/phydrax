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

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
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
      slot = static_cast<int32_t>(triangles.size());
      triangles.push_back({{0, 0, 0}, {-1, -1, -1}});
      constraints.insert(constraints.end(), 3, -1);
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
  bool conflict(int32_t t, const double* q, int32_t p) const {
    const Triangle2D& tri = triangles[t];
    if (tri.v[2] == kInfiniteVertex) {
      const int side = orient2d(point(tri.v[0]), point(tri.v[1]), q);
      if (side != 0) {
        return side > 0;
      }
      return finite_conflict(tri.n[2], q, p);
    }
    return finite_conflict(t, q, p);
  }

  bool finite_conflict(int32_t t, const double* q, int32_t p) const {
    const Triangle2D& tri = triangles[t];
    if (weights != nullptr) {
      return power2d_sos(point(tri.v[0]), point(tri.v[1]), point(tri.v[2]), q,
                         weights[tri.v[0]], weights[tri.v[1]], weights[tri.v[2]], weights[p],
                         tri.v[0], tri.v[1], tri.v[2], p) > 0;
    }
    return incircle_sos(point(tri.v[0]), point(tri.v[1]), point(tri.v[2]), q, tri.v[0],
                        tri.v[1], tri.v[2], p) > 0;
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
      int32_t target = -1;
      int crossed = -1;
      for (int i = 0; i < 3; ++i) {
        const int k = (first + i) % 3;
        const int32_t nb = tri.n[k];
        if (nb == previous) {
          continue;
        }
        if (orient2d(point(tri.v[next3(k)]), point(tri.v[prev3(k)]), q) < 0) {
          target = nb;
          crossed = k;
          break;
        }
      }
      if (target < 0) {
        return t;
      }
      if (blocked != nullptr && constraint(t, crossed) != -1) {
        *blocked = {t, crossed};
        return -1;
      }
      previous = t;
      t = target;
      if (is_ghost(t)) {
        return t;
      }
    }
  }

  // Bowyer-Watson insertion of vertex p (unconstrained triangulations).
  Insertion insert(int32_t p) {
    const double* q = point(p);
    const int32_t start = walk(hint, q);
    if (!conflict(start, q, p)) {
      return Insertion::kRedundant;
    }
    epoch_ += 2;
    const std::uint64_t inside = epoch_;
    const std::uint64_t outside = epoch_ + 1;
    stack_.clear();
    cavity_.clear();
    boundary_.clear();
    stack_.push_back(start);
    marks_[start] = inside;
    while (!stack_.empty()) {
      const int32_t t = stack_.back();
      stack_.pop_back();
      cavity_.push_back(t);
      for (int k = 0; k < 3; ++k) {
        const int32_t nb = triangles[t].n[k];
        const std::uint64_t mark = marks_[nb];
        if (mark == inside) {
          continue;
        }
        if (mark != outside && conflict(nb, q, p)) {
          marks_[nb] = inside;
          stack_.push_back(nb);
        } else {
          marks_[nb] = outside;
          const Triangle2D& tri = triangles[t];
          boundary_.push_back({tri.v[next3(k)], tri.v[prev3(k)], nb});
        }
      }
    }
    if (weights != nullptr) {
      // Vertices strictly inside the cavity lose their whole star.
      for (const BoundaryEdge& edge : boundary_) {
        vertex_marks_[static_cast<std::size_t>(edge.u + 1)] = epoch_;
      }
      for (int32_t t : cavity_) {
        for (int32_t vertex : triangles[t].v) {
          if (vertex != kInfiniteVertex &&
              vertex_marks_[static_cast<std::size_t>(vertex + 1)] != epoch_) {
            vertex_marks_[static_cast<std::size_t>(vertex + 1)] = epoch_;
            vertex_triangle[static_cast<std::size_t>(vertex)] = -1;
          }
        }
      }
    }
    for (int32_t t : cavity_) {
      release(t);
    }
    created_.clear();
    for (const BoundaryEdge& edge : boundary_) {
      const int32_t t = allocate();
      created_.push_back(t);
      Triangle2D& tri = triangles[t];
      tri.v[0] = edge.u;
      tri.v[1] = edge.w;
      tri.v[2] = p;
      tri.n[2] = edge.outside;
      vertex_slots_[static_cast<std::size_t>(edge.u + 1)] = t;
      const int j = edge_index(edge.outside, edge.w, edge.u);
      triangles[edge.outside].n[j] = t;
    }
    // New triangle (u, w, p) meets (w, x, p) across w - p.
    for (int32_t t : created_) {
      const int32_t other = vertex_slots_[static_cast<std::size_t>(triangles[t].v[1] + 1)];
      triangles[t].n[0] = other;
      triangles[other].n[1] = t;
    }
    for (int32_t t : created_) {
      Triangle2D& tri = triangles[t];
      constraints[3 * static_cast<std::size_t>(t)] = -1;
      constraints[3 * static_cast<std::size_t>(t) + 1] = -1;
      constraints[3 * static_cast<std::size_t>(t) + 2] = -1;
      if (tri.v[0] == kInfiniteVertex) {
        tri = {{tri.v[1], tri.v[2], tri.v[0]}, {tri.n[1], tri.n[2], tri.n[0]}};
      } else if (tri.v[1] == kInfiniteVertex) {
        tri = {{tri.v[2], tri.v[0], tri.v[1]}, {tri.n[2], tri.n[0], tri.n[1]}};
      } else {
        ++finite_count;
        hint = t;
      }
      for (int32_t vertex : tri.v) {
        if (vertex != kInfiniteVertex) {
          vertex_triangle[static_cast<std::size_t>(vertex)] = t;
        }
      }
    }
    return Insertion::kInserted;
  }

  // Finite live triangles as a flat CCW vertex array.
  void finite_cells(std::vector<int32_t>& cells) const {
    cells.clear();
    cells.reserve(static_cast<std::size_t>(3 * finite_count));
    for (std::size_t t = 0; t < triangles.size(); ++t) {
      if (live[t] != 0 && triangles[t].v[2] != kInfiniteVertex) {
        cells.insert(cells.end(), triangles[t].v, triangles[t].v + 3);
      }
    }
  }

 private:
  struct BoundaryEdge {
    int32_t u;
    int32_t w;
    int32_t outside;
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

// Deduplicates, orders and builds the (regular) Delaunay triangulation of the
// representatives into `triangulation` (whose coordinates must already hold
// the input points).  Returns DEGENERATE_INPUT when the representatives are
// collinear and CAPACITY_EXCEEDED as soon as the finite triangle count
// exceeds max_triangles.
inline int32_t build_delaunay_2d(Triangulation2D& triangulation, const double* points,
                                 int64_t point_count, const double* weights,
                                 int64_t max_triangles, std::vector<int32_t>& vertex_map) {
  const std::vector<int32_t> representatives =
      deduplicate_points(points, point_count, 2, weights, vertex_map);
  if (representatives.size() < 3) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  const std::vector<int32_t> order = brio_hilbert_order(points, 2, representatives);
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
  if (triangulation.finite_count > max_triangles) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  for (std::size_t k = 2; k < order.size(); ++k) {
    if (k == third) {
      continue;
    }
    triangulation.insert(order[k]);
    if (triangulation.finite_count > max_triangles) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
  }
  return PHX_MC_OK;
}

}  // namespace phx::mc
