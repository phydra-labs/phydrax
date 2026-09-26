//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C ABI: constrained Delaunay triangulation with Ruppert/Chew refinement.
//
// Pipeline: Delaunay triangulation of the input points (triangulation2d.hpp),
// segment recovery by exact walks and Anglada's pseudo-polygon
// retriangulation, optional hull constraints, carving by flood fill over
// unconstrained edges, then Delaunay refinement by Lawson insertion (splits
// plus flips that never cross constrained edges).
//
// Constraint ids stored on edges: input segment index (>= 0), kHullConstraint
// for convex hull boundary edges, -1 for free edges.  region[t] is 1 for
// triangles of the meshed domain and 0 elsewhere (carved or ghost); a domain
// boundary always runs along constrained edges, so flips and insertions that
// never cross constrained edges preserve the partition.
//
// A split point of a subsegment is a floating-point construction and need not
// lie exactly on the segment.  The split is performed with exact
// orientation tests: a point exactly on the open subsegment splits its two
// triangles into four; a point strictly on one side is inserted into the
// triangle on that side (or attached to the hull when that side is a ghost),
// the two half edges become the constrained subsegments and the old edge
// becomes a free edge whose triangles take the region of the far side.  The
// result is always a valid triangulation; configurations where no exact
// placement exists (degenerate neighborhoods) leave the subsegment unsplit and
// are reported through the final quality check.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <new>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "expansion.hpp"
#include "filtered.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "triangulation2d.hpp"

namespace phx::mc {
namespace {

constexpr int32_t kFree = -1;
constexpr int32_t kHullConstraint = -2;

struct StatusError {
  int32_t status;
};

template <class T, class Difference>
T dot_value(const double* a, const double* b, const double* v, Difference difference) {
  const T ax = difference(a[0], v[0]);
  const T ay = difference(a[1], v[1]);
  const T bx = difference(b[0], v[0]);
  const T by = difference(b[1], v[1]);
  return ax * bx + ay * by;
}

// Exact sign of (a - v) . (b - v); negative iff v lies strictly inside the
// circle with diameter ab.
int dot_sign(const double* a, const double* b, const double* v) {
  const Approx filtered = dot_value<Approx>(
      a, b, v, [](double x, double y) { return Approx::exact(x) - Approx::exact(y); });
  const int sign = filtered.certified_sign();
  if (sign != 2) {
    return sign;
  }
  return dot_value<Expansion>(a, b, v, [](double x, double y) {
           return Expansion::difference(x, y);
         }).sign();
}

bool same_point(const double* a, const double* b) { return a[0] == b[0] && a[1] == b[1]; }

// For m collinear with a and b: whether m lies strictly between them.
bool strictly_between(const double* a, const double* b, const double* m) {
  const int axis = a[0] != b[0] ? 0 : 1;
  return (a[axis] < m[axis] && m[axis] < b[axis]) || (b[axis] < m[axis] && m[axis] < a[axis]);
}

// For u collinear with a and b: whether u lies on the ray from a through b.
bool same_direction(const double* a, const double* b, const double* u) {
  const int axis = a[0] != b[0] ? 0 : 1;
  return (b[axis] > a[axis]) == (u[axis] > a[axis]) && u[axis] != a[axis];
}

// Rounds a constructed coordinate into the exact domain; false when impossible.
bool snap_to_domain(double* xy) {
  for (int axis = 0; axis < 2; ++axis) {
    if (!std::isfinite(xy[axis]) || std::fabs(xy[axis]) > kCoordinateMaxMagnitude) {
      return false;
    }
    if (std::fabs(xy[axis]) < kCoordinateMinMagnitude) {
      xy[axis] = 0.0;
    }
  }
  return true;
}

struct BadTriangle {
  double badness;
  int32_t v[3];  // rotated to start at the smallest vertex
  int32_t slot;
};

struct BadOrder {
  bool operator()(const BadTriangle& left, const BadTriangle& right) const {
    if (left.badness != right.badness) {
      return left.badness < right.badness;
    }
    return std::lexicographical_compare(right.v, right.v + 3, left.v, left.v + 3);
  }
};

struct Quality {
  bool bad = false;
  double badness = 0.0;
};

class ConstrainedMesher {
 public:
  ConstrainedMesher(Triangulation2D& triangulation, int32_t input_count, double min_angle_degrees,
                    double max_area, int64_t max_steiner, int64_t max_triangles)
      : tri_(triangulation),
        input_count_(input_count),
        max_area_(max_area),
        max_steiner_(max_steiner),
        max_triangles_(max_triangles),
        angle_enabled_(min_angle_degrees > 0.0),
        sin_min_angle_(std::sin(min_angle_degrees * (3.14159265358979323846 / 180.0))) {}

  // Inserts the edge chain from a to b; vertices exactly on the open segment
  // split it and every piece carries id.
  void insert_segment(int32_t a, int32_t b, int32_t id) {
    while (a != b) {
      a = recover_from(a, b, id);
    }
  }

  void constrain_hull() {
    for (std::size_t t = 0; t < tri_.triangles.size(); ++t) {
      const int32_t slot = static_cast<int32_t>(t);
      if (tri_.live[t] != 0 && tri_.is_ghost(slot) && tri_.constraint(slot, 2) == kFree) {
        const Triangle2D& ghost = tri_.triangles[t];
        tri_.set_edge_constraint(slot, ghost.v[0], ghost.v[1], kHullConstraint);
      }
    }
  }

  void carve(bool keep_hull, int64_t hole_count, const double* holes) {
    std::vector<int32_t> queue;
    for (std::size_t t = 0; t < tri_.triangles.size(); ++t) {
      const bool ghost = tri_.live[t] != 0 && tri_.is_ghost(static_cast<int32_t>(t));
      tri_.region[t] = (tri_.live[t] != 0 && !ghost) ? 1 : 0;
      if (ghost && !keep_hull) {
        queue.push_back(static_cast<int32_t>(t));
      }
    }
    flood_outside(queue);
    for (int64_t h = 0; h < hole_count; ++h) {
      const int32_t start = tri_.vertex_triangle[static_cast<std::size_t>(any_vertex())];
      const int32_t located = tri_.walk(start, holes + 2 * h);
      if (!tri_.is_ghost(located) && tri_.region[static_cast<std::size_t>(located)] != 0) {
        tri_.region[static_cast<std::size_t>(located)] = 0;
        queue.assign(1, located);
        flood_outside(queue);
      }
    }
  }

  // Ruppert/Chew refinement; returns OK or REFINEMENT_LIMIT.
  int32_t refine() {
    count_segment_degrees();
    for (std::size_t t = 0; t < tri_.triangles.size(); ++t) {
      const int32_t slot = static_cast<int32_t>(t);
      if (tri_.live[t] != 0 && !tri_.is_ghost(slot) && tri_.region[t] != 0) {
        enqueue_triangle_checks(slot);
      }
    }
    bool limit_reached = false;
    while (!limit_reached) {
      if (!segment_queue_.empty()) {
        const auto [a, b] = segment_queue_.front();
        segment_queue_.pop_front();
        const auto [t, k] = find_edge(a, b);
        if (t < 0 || tri_.constraint(t, k) == kFree || !subsegment_encroached(t, k)) {
          continue;
        }
        if (steiner_count_ >= max_steiner_) {
          limit_reached = true;
          break;
        }
        split_subsegment(t, k);
        continue;
      }
      if (bad_queue_.empty()) {
        break;
      }
      const BadTriangle entry = bad_queue_.top();
      bad_queue_.pop();
      if (!matches(entry)) {
        continue;
      }
      if (steiner_count_ >= max_steiner_) {
        limit_reached = true;
        break;
      }
      process_bad_triangle(entry, limit_reached);
    }
    for (std::size_t t = 0; t < tri_.triangles.size(); ++t) {
      const int32_t slot = static_cast<int32_t>(t);
      if (tri_.live[t] != 0 && !tri_.is_ghost(slot) && tri_.region[t] != 0 &&
          quality(slot).bad) {
        return PHX_MC_REFINEMENT_LIMIT;
      }
    }
    return PHX_MC_OK;
  }

 private:
  Triangulation2D& tri_;
  int32_t input_count_;
  double max_area_;
  int64_t max_steiner_;
  int64_t max_triangles_;
  bool angle_enabled_;
  double sin_min_angle_;
  int64_t steiner_count_ = 0;
  std::vector<int32_t> segment_degree_;
  // Per Steiner vertex (index - input_count_): the input vertices ending the
  // constrained chain it was inserted on, {-1, -1} for circumcenters.
  std::vector<std::pair<int32_t, int32_t>> chain_ends_;
  std::deque<std::pair<int32_t, int32_t>> segment_queue_;
  std::priority_queue<BadTriangle, std::vector<BadTriangle>, BadOrder> bad_queue_;
  std::vector<int32_t> flip_stack_;
  std::vector<std::uint64_t> marks_;
  std::uint64_t epoch_ = 0;

  int32_t any_vertex() const {
    for (int32_t v = 0; v < tri_.vertex_count(); ++v) {
      if (tri_.vertex_triangle[static_cast<std::size_t>(v)] >= 0) {
        return v;
      }
    }
    throw std::logic_error("cdt2d: empty triangulation");
  }

  void flood_outside(std::vector<int32_t>& queue) {
    while (!queue.empty()) {
      const int32_t t = queue.back();
      queue.pop_back();
      for (int k = 0; k < 3; ++k) {
        const int32_t nb = tri_.triangles[static_cast<std::size_t>(t)].n[k];
        if (tri_.constraint(t, k) == kFree && tri_.region[static_cast<std::size_t>(nb)] != 0) {
          tri_.region[static_cast<std::size_t>(nb)] = 0;
          queue.push_back(nb);
        }
      }
    }
  }

  // Triangle containing the directed edge a -> b and the edge index, or {-1, -1}.
  std::pair<int32_t, int> find_edge(int32_t a, int32_t b) const {
    const int32_t start = tri_.vertex_triangle[static_cast<std::size_t>(a)];
    if (start < 0) {
      return {-1, -1};
    }
    int32_t t = start;
    do {
      const int i = tri_.index_of(t, a);
      const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
      if (triangle.v[next3(i)] == b) {
        return {t, prev3(i)};
      }
      t = triangle.n[next3(i)];
    } while (t != start);
    return {-1, -1};
  }

  void constrain_edge(int32_t t, int k, int32_t id) {
    if (tri_.constraint(t, k) != kFree) {
      throw StatusError{PHX_MC_CONSTRAINT_INTERSECTION};
    }
    const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
    tri_.set_edge_constraint(t, triangle.v[next3(k)], triangle.v[prev3(k)], id);
  }

  // Recovers the part of segment (a, b) starting at a; returns the vertex
  // where the recovered piece ends (b or a vertex on the open segment).
  int32_t recover_from(int32_t a, int32_t b, int32_t id) {
    const double* pa = tri_.point(a);
    const double* pb = tri_.point(b);
    const int32_t start = tri_.vertex_triangle[static_cast<std::size_t>(a)];
    int32_t t = start;
    do {
      const int i = tri_.index_of(t, a);
      const Triangle2D triangle = tri_.triangles[static_cast<std::size_t>(t)];
      const int32_t u = triangle.v[next3(i)];
      const int32_t w = triangle.v[prev3(i)];
      if (!tri_.is_ghost(t)) {
        if (u == b) {
          constrain_edge(t, prev3(i), id);
          return b;
        }
        if (w == b) {
          constrain_edge(t, next3(i), id);
          return b;
        }
        const int su = orient2d(pa, pb, tri_.point(u));
        if (su == 0 && same_direction(pa, pb, tri_.point(u))) {
          constrain_edge(t, prev3(i), id);
          return u;
        }
        const int sw = orient2d(pa, pb, tri_.point(w));
        if (sw == 0 && same_direction(pa, pb, tri_.point(w))) {
          constrain_edge(t, next3(i), id);
          return w;
        }
        if (su < 0 && sw > 0) {
          return cross_and_recover(t, a, b, u, w, id);
        }
      }
      t = triangle.n[next3(i)];
    } while (t != start);
    throw std::logic_error("cdt2d: segment direction not found around its endpoint");
  }

  int32_t cross_and_recover(int32_t first, int32_t a, int32_t b, int32_t right, int32_t left,
                            int32_t id) {
    const double* pa = tri_.point(a);
    const double* pb = tri_.point(b);
    std::vector<int32_t> crossed{first};
    std::vector<int32_t> left_chain{left};
    std::vector<int32_t> right_chain{right};
    int32_t current = first;
    int32_t end = -1;
    for (;;) {
      const int k = tri_.edge_index(current, right, left);
      if (tri_.constraint(current, k) != kFree) {
        throw StatusError{PHX_MC_CONSTRAINT_INTERSECTION};
      }
      const int32_t next = tri_.triangles[static_cast<std::size_t>(current)].n[k];
      if (tri_.is_ghost(next)) {
        throw std::logic_error("cdt2d: segment leaves the convex hull");
      }
      const int j = tri_.edge_index(next, left, right);
      const int32_t x = tri_.triangles[static_cast<std::size_t>(next)].v[j];
      crossed.push_back(next);
      if (x == b) {
        end = b;
        break;
      }
      const int side = orient2d(pa, pb, tri_.point(x));
      if (side == 0) {
        end = x;
        break;
      }
      if (side > 0) {
        left_chain.push_back(x);
        left = x;
      } else {
        right_chain.push_back(x);
        right = x;
      }
      current = next;
    }
    std::reverse(right_chain.begin(), right_chain.end());
    retriangulate(crossed, a, end, left_chain, right_chain);
    const auto [t, k] = find_edge(a, end);
    if (t < 0) {
      throw std::logic_error("cdt2d: recovered edge missing");
    }
    constrain_edge(t, k, id);
    return end;
  }

  // Replaces the triangles `region_triangles` (the union is the polygon
  // a -> end plus both pseudo-polygon chains) by constrained Delaunay
  // triangulations of the two pseudo-polygons (Anglada).
  void retriangulate(std::vector<int32_t> region_triangles, int32_t a, int32_t end,
                     const std::vector<int32_t>& left_chain,
                     const std::vector<int32_t>& right_chain) {
    struct Boundary {
      std::uint64_t key;
      int32_t outside;
      int32_t constraint;
    };
    auto key_of = [](int32_t u, int32_t w) {
      return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
             static_cast<std::uint32_t>(w);
    };
    std::sort(region_triangles.begin(), region_triangles.end());
    std::vector<Boundary> boundary;
    for (int32_t t : region_triangles) {
      const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
      for (int k = 0; k < 3; ++k) {
        if (!std::binary_search(region_triangles.begin(), region_triangles.end(),
                                triangle.n[k])) {
          boundary.push_back({key_of(triangle.v[next3(k)], triangle.v[prev3(k)]), triangle.n[k],
                              tri_.constraint(t, k)});
        }
      }
    }
    std::sort(boundary.begin(), boundary.end(),
              [](const Boundary& l, const Boundary& r) { return l.key < r.key; });
    const uint8_t region = tri_.region[static_cast<std::size_t>(region_triangles[0])];
    for (int32_t t : region_triangles) {
      tri_.release(t);
    }
    std::vector<int32_t> created;
    anglada(a, end, left_chain, region, created);
    anglada(end, a, right_chain, region, created);
    struct Half {
      std::uint64_t key;
      int32_t t;
    };
    std::vector<Half> halves;
    for (int32_t t : created) {
      const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
      for (int k = 0; k < 3; ++k) {
        halves.push_back({key_of(triangle.v[next3(k)], triangle.v[prev3(k)]), t});
      }
    }
    std::sort(halves.begin(), halves.end(),
              [](const Half& l, const Half& r) { return l.key < r.key; });
    for (int32_t t : created) {
      for (int k = 0; k < 3; ++k) {
        const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
        const int32_t u = triangle.v[next3(k)];
        const int32_t w = triangle.v[prev3(k)];
        const std::uint64_t twin = key_of(w, u);
        auto inner = std::lower_bound(halves.begin(), halves.end(), twin,
                                      [](const Half& h, std::uint64_t key) { return h.key < key; });
        if (inner != halves.end() && inner->key == twin) {
          tri_.link_edge(t, u, w, inner->t);
          continue;
        }
        const std::uint64_t own = key_of(u, w);
        auto outer =
            std::lower_bound(boundary.begin(), boundary.end(), own,
                             [](const Boundary& e, std::uint64_t key) { return e.key < key; });
        if (outer == boundary.end() || outer->key != own) {
          throw std::logic_error("cdt2d: retriangulation does not match its cavity");
        }
        tri_.link_edge(t, u, w, outer->outside);
        tri_.constraints[3 * static_cast<std::size_t>(t) + k] = outer->constraint;
      }
    }
  }

  // Constrained Delaunay triangulation of the polygon u, w, chain reversed
  // (chain listed from the u side to the w side, all left of u -> w).
  void anglada(int32_t u, int32_t w, const std::vector<int32_t>& chain, uint8_t region,
               std::vector<int32_t>& created) {
    struct Job {
      int32_t u, w;
      std::size_t lo, hi;
    };
    std::vector<Job> jobs{{u, w, 0, chain.size()}};
    while (!jobs.empty()) {
      const Job job = jobs.back();
      jobs.pop_back();
      if (job.lo >= job.hi) {
        continue;
      }
      std::size_t best = job.lo;
      for (std::size_t j = job.lo + 1; j < job.hi; ++j) {
        const int32_t c = chain[best];
        const int32_t p = chain[j];
        if (incircle_sos(tri_.point(job.u), tri_.point(job.w), tri_.point(c), tri_.point(p),
                         job.u, job.w, c, p) > 0) {
          best = j;
        }
      }
      const int32_t c = chain[best];
      const int32_t t = tri_.allocate();
      tri_.set_triangle(t, job.u, job.w, c);
      tri_.region[static_cast<std::size_t>(t)] = region;
      created.push_back(t);
      jobs.push_back({job.u, c, job.lo, best});
      jobs.push_back({c, job.w, best + 1, job.hi});
    }
  }

  // ------------------------------------------------------------ refinement

  void count_segment_degrees() {
    segment_degree_.assign(static_cast<std::size_t>(tri_.vertex_count()), 0);
    for (std::size_t t = 0; t < tri_.triangles.size(); ++t) {
      if (tri_.live[t] == 0) {
        continue;
      }
      const Triangle2D& triangle = tri_.triangles[t];
      for (int k = 0; k < 3; ++k) {
        const int32_t u = triangle.v[next3(k)];
        if (tri_.constraints[3 * t + k] != kFree && u != kInfiniteVertex) {
          ++segment_degree_[static_cast<std::size_t>(u)];
        }
      }
    }
  }

  bool shared_input_vertex(int32_t v) const {
    return v < input_count_ && segment_degree_[static_cast<std::size_t>(v)] >= 2;
  }

  Quality quality(int32_t t) const {
    const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
    const double* p[3] = {tri_.point(triangle.v[0]), tri_.point(triangle.v[1]),
                          tri_.point(triangle.v[2])};
    double squared[3];
    for (int k = 0; k < 3; ++k) {
      const double dx = p[prev3(k)][0] - p[next3(k)][0];
      const double dy = p[prev3(k)][1] - p[next3(k)][1];
      squared[k] = dx * dx + dy * dy;
    }
    const double twice_area = std::fabs((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) -
                                        (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]));
    Quality result;
    if (std::isfinite(max_area_)) {
      const double ratio = 0.5 * twice_area / max_area_;
      if (ratio > 1.0) {
        result.bad = true;
      }
      result.badness = ratio;
    }
    if (angle_enabled_) {
      int shortest = 0;
      for (int k = 1; k < 3; ++k) {
        if (squared[k] < squared[shortest]) {
          shortest = k;
        }
      }
      const int32_t apex = triangle.v[shortest];
      const bool input_angle = (apex < input_count_ && tri_.constraint(t, next3(shortest)) != kFree &&
                                tri_.constraint(t, prev3(shortest)) != kFree) ||
                               spans_small_input_angle(triangle.v[next3(shortest)],
                                                       triangle.v[prev3(shortest)]);
      if (!input_angle) {
        // sin(theta_min) / sin(smallest angle) via the two longer edges.
        const double ratio = twice_area > 0.0
                                 ? sin_min_angle_ * std::sqrt(squared[next3(shortest)]) *
                                       std::sqrt(squared[prev3(shortest)]) / twice_area
                                 : std::numeric_limits<double>::infinity();
        if (ratio > 1.0) {
          result.bad = true;
        }
        result.badness = std::max(result.badness, ratio);
      }
    }
    return result;
  }

  BadTriangle make_entry(int32_t t, double badness) const {
    const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
    BadTriangle entry{badness, {triangle.v[0], triangle.v[1], triangle.v[2]}, t};
    const int first = static_cast<int>(std::min_element(entry.v, entry.v + 3) - entry.v);
    std::rotate(entry.v, entry.v + first, entry.v + 3);
    return entry;
  }

  bool matches(const BadTriangle& entry) const {
    if (tri_.live[static_cast<std::size_t>(entry.slot)] == 0 || tri_.is_ghost(entry.slot)) {
      return false;
    }
    const BadTriangle current = make_entry(entry.slot, entry.badness);
    return std::equal(current.v, current.v + 3, entry.v);
  }

  bool encroaches(int32_t t, int k, const double* v) const {
    const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
    return dot_sign(tri_.point(triangle.v[next3(k)]), tri_.point(triangle.v[prev3(k)]), v) < 0;
  }

  // Encroachment of constrained edge k of t by the apex of either domain
  // triangle adjacent to it.
  bool subsegment_encroached(int32_t t, int k) const {
    const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
    if (!tri_.is_ghost(t) && tri_.region[static_cast<std::size_t>(t)] != 0 &&
        encroaches(t, k, tri_.point(triangle.v[k]))) {
      return true;
    }
    const int32_t o = triangle.n[k];
    const int j = tri_.edge_index(o, triangle.v[prev3(k)], triangle.v[next3(k)]);
    return !tri_.is_ghost(o) && tri_.region[static_cast<std::size_t>(o)] != 0 &&
           encroaches(o, j, tri_.point(tri_.triangles[static_cast<std::size_t>(o)].v[j]));
  }

  void enqueue_triangle_checks(int32_t t) {
    const Quality q = quality(t);
    if (q.bad) {
      bad_queue_.push(make_entry(t, q.badness));
    }
    const Triangle2D& triangle = tri_.triangles[static_cast<std::size_t>(t)];
    for (int k = 0; k < 3; ++k) {
      if (tri_.constraint(t, k) != kFree && encroaches(t, k, tri_.point(triangle.v[k]))) {
        segment_queue_.emplace_back(triangle.v[next3(k)], triangle.v[prev3(k)]);
      }
    }
  }

  void after_insertion(int32_t m) {
    ++steiner_count_;
    if (tri_.finite_count > max_triangles_) {
      throw StatusError{PHX_MC_CAPACITY_EXCEEDED};
    }
    const int32_t start = tri_.vertex_triangle[static_cast<std::size_t>(m)];
    int32_t t = start;
    do {
      if (!tri_.is_ghost(t) && tri_.region[static_cast<std::size_t>(t)] != 0) {
        enqueue_triangle_checks(t);
      }
      t = tri_.triangles[static_cast<std::size_t>(t)].n[next3(tri_.index_of(t, m))];
    } while (t != start);
  }

  int32_t new_vertex(const double* xy, std::pair<int32_t, int32_t> ends = {-1, -1}) {
    if (tri_.vertex_count() >= kMaxMeshPoints) {
      throw StatusError{PHX_MC_CAPACITY_EXCEEDED};
    }
    chain_ends_.push_back(ends);
    return tri_.add_vertex(xy);
  }

  std::pair<int32_t, int32_t> chain_ends_of(int32_t a, int32_t b) const {
    if (a >= input_count_) {
      return chain_ends_[static_cast<std::size_t>(a - input_count_)];
    }
    if (b >= input_count_) {
      return chain_ends_[static_cast<std::size_t>(b - input_count_)];
    }
    return {a, b};
  }

  // Shewchuk's rule for small input angles: the shortest edge u - w joins
  // Steiner vertices on two different constrained chains that share an input
  // vertex j and lie (nearly) equidistant from j, i.e. on one concentric shell.
  bool spans_small_input_angle(int32_t u, int32_t w) const {
    if (u < input_count_ || w < input_count_) {
      return false;
    }
    const auto eu = chain_ends_[static_cast<std::size_t>(u - input_count_)];
    const auto ew = chain_ends_[static_cast<std::size_t>(w - input_count_)];
    if (eu.first < 0 || ew.first < 0 ||
        std::minmax(eu.first, eu.second) == std::minmax(ew.first, ew.second)) {
      return false;
    }
    int32_t join = -1;
    if (eu.first == ew.first || eu.first == ew.second) {
      join = eu.first;
    } else if (eu.second == ew.first || eu.second == ew.second) {
      join = eu.second;
    }
    if (join < 0) {
      return false;
    }
    const double* pj = tri_.point(join);
    const double* pu = tri_.point(u);
    const double* pw = tri_.point(w);
    const double du = (pu[0] - pj[0]) * (pu[0] - pj[0]) + (pu[1] - pj[1]) * (pu[1] - pj[1]);
    const double dw = (pw[0] - pj[0]) * (pw[0] - pj[0]) + (pw[1] - pj[1]) * (pw[1] - pj[1]);
    return du < 1.001 * dw && du > 0.999 * dw;
  }

  // --------------------------------------------------- Lawson primitives

  // Splits t into three triangles around m (strictly inside t).
  void split_triangle(int32_t t, int32_t m) {
    const Triangle2D old = tri_.triangles[static_cast<std::size_t>(t)];
    const int32_t c0 = tri_.constraint(t, 0);
    const int32_t c1 = tri_.constraint(t, 1);
    const int32_t c2 = tri_.constraint(t, 2);
    const uint8_t region = tri_.region[static_cast<std::size_t>(t)];
    const int32_t a = old.v[0], b = old.v[1], c = old.v[2];
    tri_.release(t);
    const int32_t t0 = tri_.allocate();
    const int32_t t1 = tri_.allocate();
    const int32_t t2 = tri_.allocate();
    tri_.set_triangle(t0, b, c, m, kFree, kFree, c0);
    tri_.set_triangle(t1, c, a, m, kFree, kFree, c1);
    tri_.set_triangle(t2, a, b, m, kFree, kFree, c2);
    for (int32_t s : {t0, t1, t2}) {
      tri_.region[static_cast<std::size_t>(s)] = region;
    }
    tri_.link_edge(t0, b, c, old.n[0]);
    tri_.link_edge(t1, c, a, old.n[1]);
    tri_.link_edge(t2, a, b, old.n[2]);
    tri_.link_edge(t0, c, m, t1);
    tri_.link_edge(t1, a, m, t2);
    tri_.link_edge(t2, b, m, t0);
    flip_stack_.insert(flip_stack_.end(), {t0, t1, t2});
  }

  // Splits edge k of t (and its twin) at m, which lies exactly on the open
  // edge; the two halves carry `id`.
  void split_edge(int32_t t, int k, int32_t m, int32_t id) {
    const Triangle2D told = tri_.triangles[static_cast<std::size_t>(t)];
    const int32_t c = told.v[k], a = told.v[next3(k)], b = told.v[prev3(k)];
    const int32_t o = told.n[k];
    const int j = tri_.edge_index(o, b, a);
    const Triangle2D oold = tri_.triangles[static_cast<std::size_t>(o)];
    const int32_t d = oold.v[j];
    const int32_t n_bc = told.n[next3(k)], c_bc = tri_.constraint(t, next3(k));
    const int32_t n_ca = told.n[prev3(k)], c_ca = tri_.constraint(t, prev3(k));
    const int ja = tri_.edge_index(o, a, d);
    const int jd = tri_.edge_index(o, d, b);
    const int32_t n_ad = oold.n[ja], c_ad = tri_.constraint(o, ja);
    const int32_t n_db = oold.n[jd], c_db = tri_.constraint(o, jd);
    const uint8_t region_t = tri_.region[static_cast<std::size_t>(t)];
    const uint8_t region_o = tri_.region[static_cast<std::size_t>(o)];
    tri_.release(t);
    tri_.release(o);
    const int32_t t1 = tri_.allocate();
    const int32_t t2 = tri_.allocate();
    const int32_t o1 = tri_.allocate();
    const int32_t o2 = tri_.allocate();
    tri_.set_triangle(t1, c, a, m, id, kFree, c_ca);
    tri_.set_triangle(t2, c, m, b, id, c_bc, kFree);
    if (d != kInfiniteVertex) {
      tri_.set_triangle(o1, d, b, m, id, kFree, c_db);
      tri_.set_triangle(o2, d, m, a, id, c_ad, kFree);
    } else {
      tri_.set_triangle(o1, b, m, kInfiniteVertex, kFree, c_db, id);
      tri_.set_triangle(o2, m, a, kInfiniteVertex, c_ad, kFree, id);
    }
    tri_.region[static_cast<std::size_t>(t1)] = region_t;
    tri_.region[static_cast<std::size_t>(t2)] = region_t;
    tri_.region[static_cast<std::size_t>(o1)] = region_o;
    tri_.region[static_cast<std::size_t>(o2)] = region_o;
    tri_.link_edge(t1, c, a, n_ca);
    tri_.link_edge(t2, b, c, n_bc);
    tri_.link_edge(o1, d, b, n_db);
    tri_.link_edge(o2, a, d, n_ad);
    tri_.link_edge(t1, m, c, t2);
    tri_.link_edge(o1, m, d, o2);
    tri_.link_edge(t1, a, m, o2);
    tri_.link_edge(t2, m, b, o1);
    flip_stack_.insert(flip_stack_.end(), {t1, t2, o1, o2});
  }

  // m lies strictly outside the hull edge p -> q of ghost g: adds the finite
  // triangle (p, q, m) and replaces g by two ghosts.
  void attach_to_hull(int32_t g, int32_t p, int32_t q, int32_t m, int32_t id) {
    const Triangle2D old = tri_.triangles[static_cast<std::size_t>(g)];
    const int32_t inner = old.n[tri_.edge_index(g, p, q)];
    const int32_t ghost_q = old.n[tri_.edge_index(g, q, kInfiniteVertex)];
    const int32_t ghost_p = old.n[tri_.edge_index(g, kInfiniteVertex, p)];
    const uint8_t region = tri_.region[static_cast<std::size_t>(inner)];
    tri_.release(g);
    const int32_t f = tri_.allocate();
    const int32_t h1 = tri_.allocate();
    const int32_t h2 = tri_.allocate();
    tri_.set_triangle(f, p, q, m, id, id, kFree);
    tri_.set_triangle(h1, m, q, kInfiniteVertex, kFree, kFree, id);
    tri_.set_triangle(h2, p, m, kInfiniteVertex, kFree, kFree, id);
    tri_.region[static_cast<std::size_t>(f)] = region;
    tri_.region[static_cast<std::size_t>(h1)] = 0;
    tri_.region[static_cast<std::size_t>(h2)] = 0;
    tri_.link_edge(f, p, q, inner);
    tri_.link_edge(f, q, m, h1);
    tri_.link_edge(f, m, p, h2);
    tri_.link_edge(h1, q, kInfiniteVertex, ghost_q);
    tri_.link_edge(h1, kInfiniteVertex, m, h2);
    tri_.link_edge(h2, kInfiniteVertex, p, ghost_p);
    tri_.set_edge_constraint(f, p, q, kFree);
    flip_stack_.push_back(f);
  }

  // Lawson flips of free edges opposite m until every one is locally
  // Delaunay (a non-locally-Delaunay edge always has a strictly convex quad).
  void legalize(int32_t m) {
    while (!flip_stack_.empty()) {
      const int32_t t = flip_stack_.back();
      flip_stack_.pop_back();
      if (tri_.live[static_cast<std::size_t>(t)] == 0 || tri_.is_ghost(t)) {
        continue;
      }
      const int i = tri_.index_of(t, m);
      if (i < 0 || tri_.constraint(t, i) != kFree) {
        continue;
      }
      const Triangle2D told = tri_.triangles[static_cast<std::size_t>(t)];
      const int32_t o = told.n[i];
      if (tri_.is_ghost(o)) {
        continue;
      }
      const int32_t u = told.v[next3(i)];
      const int32_t w = told.v[prev3(i)];
      const int j = tri_.edge_index(o, w, u);
      const int32_t x = tri_.triangles[static_cast<std::size_t>(o)].v[j];
      if (incircle_sos(tri_.point(m), tri_.point(u), tri_.point(w), tri_.point(x), m, u, w, x) <=
          0) {
        continue;
      }
      const Triangle2D oold = tri_.triangles[static_cast<std::size_t>(o)];
      const int32_t n_wm = told.n[next3(i)], c_wm = tri_.constraint(t, next3(i));
      const int32_t n_mu = told.n[prev3(i)], c_mu = tri_.constraint(t, prev3(i));
      const int jw = tri_.edge_index(o, u, x);
      const int ju = tri_.edge_index(o, x, w);
      const int32_t n_ux = oold.n[jw], c_ux = tri_.constraint(o, jw);
      const int32_t n_xw = oold.n[ju], c_xw = tri_.constraint(o, ju);
      const uint8_t region = tri_.region[static_cast<std::size_t>(t)];
      tri_.release(t);
      tri_.release(o);
      const int32_t first = tri_.allocate();
      const int32_t second = tri_.allocate();
      tri_.set_triangle(first, m, u, x, c_ux, kFree, c_mu);
      tri_.set_triangle(second, m, x, w, c_xw, c_wm, kFree);
      tri_.region[static_cast<std::size_t>(first)] = region;
      tri_.region[static_cast<std::size_t>(second)] = region;
      tri_.link_edge(first, u, x, n_ux);
      tri_.link_edge(first, m, u, n_mu);
      tri_.link_edge(second, x, w, n_xw);
      tri_.link_edge(second, w, m, n_wm);
      tri_.link_edge(first, x, m, second);
      flip_stack_.push_back(first);
      flip_stack_.push_back(second);
    }
  }

  // ------------------------------------------------ refinement operations

  // Splits the constrained edge k of t; false when no exact placement exists.
  bool split_subsegment(int32_t t, int k) {
    const Triangle2D triangle = tri_.triangles[static_cast<std::size_t>(t)];
    const int32_t a = triangle.v[next3(k)];
    const int32_t b = triangle.v[prev3(k)];
    const int32_t id = tri_.constraint(t, k);
    const double* pa = tri_.point(a);
    const double* pb = tri_.point(b);
    double m[2];
    const bool shell_a = shared_input_vertex(a);
    const bool shell_b = shared_input_vertex(b);
    if (shell_a != shell_b) {
      // Concentric shells: the piece at the shared input vertex gets the power
      // of two length closest to half the subsegment.
      const double* po = shell_a ? pa : pb;
      const double* pq = shell_a ? pb : pa;
      const double dx = pq[0] - po[0];
      const double dy = pq[1] - po[1];
      const double length = std::hypot(dx, dy);
      int exponent = 0;
      const double fraction = std::frexp(0.5 * length, &exponent);
      const double piece = std::ldexp(1.0, fraction < 0.75 ? exponent - 1 : exponent);
      const double ratio = piece / length;
      m[0] = po[0] + ratio * dx;
      m[1] = po[1] + ratio * dy;
    } else {
      m[0] = 0.5 * (pa[0] + pb[0]);
      m[1] = 0.5 * (pa[1] + pb[1]);
    }
    if (!snap_to_domain(m)) {
      return false;
    }
    // The constructed point is rounded; among it and its neighbors within two
    // ulps per axis (rings of increasing radius, fixed order) prefer a point
    // exactly on the open subsegment, else one strictly inside the triangle
    // (or hull ghost) on its side.
    const int32_t across = triangle.n[k];
    auto placement = [&](const double* c) {
      if (same_point(c, pa) || same_point(c, pb)) {
        return 0;
      }
      const int side = orient2d(pa, pb, c);
      if (side == 0) {
        return strictly_between(pa, pb, c) ? 1 : 0;
      }
      const int32_t near_side = side > 0 ? t : across;
      if (tri_.is_ghost(near_side)) {
        return 2;
      }
      const int32_t p = side > 0 ? a : b;
      const int32_t q = side > 0 ? b : a;
      const int32_t apex =
          tri_.triangles[static_cast<std::size_t>(near_side)].v[tri_.edge_index(near_side, p, q)];
      return orient2d(tri_.point(q), tri_.point(apex), c) > 0 &&
                     orient2d(tri_.point(apex), tri_.point(p), c) > 0
                 ? 2
                 : 0;
    };
    auto step = [](double x, int ulps) {
      for (; ulps > 0; --ulps) x = std::nextafter(x, std::numeric_limits<double>::infinity());
      for (; ulps < 0; ++ulps) x = std::nextafter(x, -std::numeric_limits<double>::infinity());
      return x;
    };
    double chosen[2] = {0.0, 0.0};
    int kind = 0;
    for (int wanted = 1; wanted <= 2 && kind == 0; ++wanted) {
      for (int radius = 0; radius <= 2 && kind == 0; ++radius) {
        for (int di = -radius; di <= radius && kind == 0; ++di) {
          for (int dj = -radius; dj <= radius && kind == 0; ++dj) {
            if (std::max(std::abs(di), std::abs(dj)) != radius) {
              continue;
            }
            double c[2] = {step(m[0], di), step(m[1], dj)};
            if (snap_to_domain(c) && placement(c) == wanted) {
              chosen[0] = c[0];
              chosen[1] = c[1];
              kind = wanted;
            }
          }
        }
      }
    }
    if (kind == 0) {
      return false;
    }
    const std::pair<int32_t, int32_t> ends = chain_ends_of(a, b);
    const int side = orient2d(pa, pb, chosen);
    if (side == 0) {
      const int32_t v = new_vertex(chosen, ends);
      split_edge(t, k, v, id);
      legalize(v);
      after_insertion(v);
      return true;
    }
    // Orient the edge p -> q so that the point lies strictly to its left.
    const int32_t p = side > 0 ? a : b;
    const int32_t q = side > 0 ? b : a;
    const int32_t near_side = side > 0 ? t : across;
    const int32_t far_side = side > 0 ? across : t;
    if (tri_.is_ghost(near_side)) {
      const int32_t v = new_vertex(chosen, ends);
      attach_to_hull(near_side, p, q, v, id);
      legalize(v);
      after_insertion(v);
      return true;
    }
    const uint8_t far_region = tri_.region[static_cast<std::size_t>(far_side)];
    const int32_t v = new_vertex(chosen, ends);
    split_triangle(near_side, v);
    const int32_t sliver = find_edge(p, q).first;
    tri_.set_edge_constraint(sliver, p, q, kFree);
    tri_.set_edge_constraint(sliver, q, v, id);
    tri_.set_edge_constraint(sliver, v, p, id);
    tri_.region[static_cast<std::size_t>(sliver)] = far_region;
    legalize(v);
    after_insertion(v);
    return true;
  }

  // Splits the constrained edge a - b if it still exists; true on insertion.
  bool split_named_subsegment(int32_t a, int32_t b) {
    const auto [t, k] = find_edge(a, b);
    if (t < 0 || tri_.constraint(t, k) == kFree) {
      return false;
    }
    return split_subsegment(t, k);
  }

  void process_bad_triangle(const BadTriangle& entry, bool& limit_reached) {
    const int32_t t = entry.slot;
    const Triangle2D triangle = tri_.triangles[static_cast<std::size_t>(t)];
    const double* p0 = tri_.point(triangle.v[0]);
    const double* p1 = tri_.point(triangle.v[1]);
    const double* p2 = tri_.point(triangle.v[2]);
    const double bx = p1[0] - p0[0], by = p1[1] - p0[1];
    const double cx = p2[0] - p0[0], cy = p2[1] - p0[1];
    const double denominator = 2.0 * (bx * cy - by * cx);
    const double b2 = bx * bx + by * by;
    const double c2 = cx * cx + cy * cy;
    double center[2] = {p0[0] + (cy * b2 - by * c2) / denominator,
                        p0[1] + (bx * c2 - cx * b2) / denominator};
    if (!snap_to_domain(center)) {
      return;
    }
    std::vector<std::pair<int32_t, int32_t>> to_split;
    std::pair<int32_t, int> blocked{-1, -1};
    const int32_t located = tri_.walk(t, center, &blocked);
    if (located < 0) {
      const Triangle2D& b = tri_.triangles[static_cast<std::size_t>(blocked.first)];
      to_split.emplace_back(b.v[next3(blocked.second)], b.v[prev3(blocked.second)]);
    } else {
      if (tri_.is_ghost(located)) {
        return;
      }
      const Triangle2D& host = tri_.triangles[static_cast<std::size_t>(located)];
      int zero_count = 0;
      int zero_edge = -1;
      for (int k = 0; k < 3; ++k) {
        if (orient2d(tri_.point(host.v[next3(k)]), tri_.point(host.v[prev3(k)]), center) == 0) {
          ++zero_count;
          zero_edge = k;
        }
      }
      if (zero_count >= 2) {
        return;
      }
      if (zero_count == 1 && tri_.constraint(located, zero_edge) != kFree) {
        to_split.emplace_back(host.v[next3(zero_edge)], host.v[prev3(zero_edge)]);
      } else {
        collect_encroached_by(located, center, to_split);
      }
      if (to_split.empty()) {
        const int32_t v = new_vertex(center);
        if (zero_count == 1) {
          split_edge(located, zero_edge, v, kFree);
        } else {
          split_triangle(located, v);
        }
        legalize(v);
        after_insertion(v);
        return;
      }
    }
    std::sort(to_split.begin(), to_split.end());
    to_split.erase(std::unique(to_split.begin(), to_split.end()), to_split.end());
    bool progress = false;
    for (const auto& [a, b] : to_split) {
      if (steiner_count_ >= max_steiner_) {
        limit_reached = true;
        return;
      }
      progress = split_named_subsegment(a, b) || progress;
    }
    if (progress && matches(entry)) {
      bad_queue_.push(entry);
    }
  }

  // Constrained edges on the boundary of the constrained Delaunay cavity of
  // `center` (conflict region reachable from `start` without crossing
  // constraints) that `center` encroaches upon.
  void collect_encroached_by(int32_t start, const double* center,
                             std::vector<std::pair<int32_t, int32_t>>& out) {
    const int32_t id = tri_.vertex_count();
    if (marks_.size() < tri_.triangles.size()) {
      marks_.resize(tri_.triangles.size(), 0);
    }
    epoch_ += 2;
    const std::uint64_t inside = epoch_;
    const std::uint64_t outside = epoch_ + 1;
    std::vector<int32_t> stack{start};
    marks_[static_cast<std::size_t>(start)] = inside;
    while (!stack.empty()) {
      const int32_t t = stack.back();
      stack.pop_back();
      const Triangle2D triangle = tri_.triangles[static_cast<std::size_t>(t)];
      for (int k = 0; k < 3; ++k) {
        if (tri_.constraint(t, k) != kFree) {
          if (encroaches(t, k, center)) {
            out.emplace_back(triangle.v[next3(k)], triangle.v[prev3(k)]);
          }
          continue;
        }
        const int32_t nb = triangle.n[k];
        const std::uint64_t mark = marks_[static_cast<std::size_t>(nb)];
        if (mark == inside || mark == outside || tri_.is_ghost(nb)) {
          continue;
        }
        const Triangle2D& other = tri_.triangles[static_cast<std::size_t>(nb)];
        if (incircle_sos(tri_.point(other.v[0]), tri_.point(other.v[1]), tri_.point(other.v[2]),
                         center, other.v[0], other.v[1], other.v[2], id) > 0) {
          marks_[static_cast<std::size_t>(nb)] = inside;
          stack.push_back(nb);
        } else {
          marks_[static_cast<std::size_t>(nb)] = outside;
        }
      }
    }
  }
};

int32_t constrained_delaunay(int64_t point_count, const double* points, int64_t segment_count,
                             const int32_t* segments, int64_t hole_count, const double* holes,
                             int32_t keep_convex_hull, double min_angle_degrees, double max_area,
                             int64_t max_steiner, int64_t max_triangles, phx_mc_mesh** mesh) {
  if (mesh == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *mesh = nullptr;
  if (point_count < 0 || point_count > kMaxMeshPoints || (point_count > 0 && points == nullptr) ||
      !addressable(segment_count, 2, sizeof(int32_t)) ||
      (segment_count > 0 && segments == nullptr) || !addressable(hole_count, 2, sizeof(double)) ||
      (hole_count > 0 && holes == nullptr) || max_steiner < 0 || max_triangles < 0 ||
      !(min_angle_degrees >= 0.0 && min_angle_degrees < 60.0) || !(max_area > 0.0)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  int32_t status = validate_points(points, point_count, 2, nullptr);
  if (status == PHX_MC_OK) {
    status = validate_points(holes, hole_count, 2, nullptr);
  }
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int64_t k = 0; k < 2 * segment_count; ++k) {
    if (segments[k] < 0 || segments[k] >= point_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  if (point_count == 0) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  auto result = std::make_unique<phx_mc_mesh>();
  result->dimension = 2;
  result->input_point_count = point_count;
  Triangulation2D triangulation;
  triangulation.reset(points, point_count, nullptr);
  status = build_delaunay_2d(triangulation, points, point_count, nullptr, max_triangles,
                             result->vertex_map);
  if (status != PHX_MC_OK) {
    return status;
  }
  for (int64_t s = 0; s < segment_count; ++s) {
    if (result->vertex_map[static_cast<std::size_t>(segments[2 * s])] ==
        result->vertex_map[static_cast<std::size_t>(segments[2 * s + 1])]) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  const bool refine = min_angle_degrees > 0.0 || std::isfinite(max_area);
  ConstrainedMesher mesher(triangulation, static_cast<int32_t>(point_count), min_angle_degrees,
                           max_area, max_steiner, max_triangles);
  try {
    for (int64_t s = 0; s < segment_count; ++s) {
      mesher.insert_segment(result->vertex_map[static_cast<std::size_t>(segments[2 * s])],
                            result->vertex_map[static_cast<std::size_t>(segments[2 * s + 1])],
                            static_cast<int32_t>(s));
    }
    if (keep_convex_hull != 0) {
      mesher.constrain_hull();
    }
    mesher.carve(keep_convex_hull != 0, hole_count, holes);
    status = refine ? mesher.refine() : PHX_MC_OK;
  } catch (const StatusError& error) {
    return error.status;
  }
  result->points = triangulation.coordinates;
  for (std::size_t t = 0; t < triangulation.triangles.size(); ++t) {
    if (triangulation.live[t] == 0 || triangulation.region[t] == 0 ||
        triangulation.is_ghost(static_cast<int32_t>(t))) {
      continue;
    }
    const Triangle2D& triangle = triangulation.triangles[t];
    result->cells.insert(result->cells.end(), triangle.v, triangle.v + 3);
    for (int k = 0; k < 3; ++k) {
      const int32_t id = triangulation.constraints[3 * t + static_cast<std::size_t>(k)];
      result->cell_segments.push_back(id >= 0 ? id : -1);
    }
  }
  canonicalize_cells(*result);
  *mesh = result.release();
  return status;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_constrained_delaunay_2d(int64_t point_count, const double* points,
                                       int64_t segment_count, const int32_t* segments,
                                       int64_t hole_count, const double* holes,
                                       int32_t keep_convex_hull, double min_angle_degrees,
                                       double max_area, int64_t max_steiner,
                                       int64_t max_triangles, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&] {
    return phx::mc::constrained_delaunay(point_count, points, segment_count, segments, hole_count,
                                         holes, keep_convex_hull, min_angle_degrees, max_area,
                                         max_steiner, max_triangles, mesh);
  });
}

}  // extern "C"
