//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Chart-embedded surface triangulations with bounded physical-space
// reconnection.  A surface patch is triangulated in its chart (a planar
// parameter domain) and simultaneously carries the physical position of every
// vertex.  Ordinary decisions that keep it valid (point location, split
// legality, flip legality) use the exact orient2d predicate.  A terminal
// producer-owned rational edge split may carry a noncollinear rounded chart
// execution view; its explicit reciprocal edge and open componentwise bounds
// own the four-child topology, while the producer retains and validates the
// exact coordinate.  Quality decisions (which diagonal a quadrilateral should
// use) are physical: the Delaunay criterion is evaluated on the physical
// triangles, so metric distortion of the chart does not leak into the
// published surface mesh.  Physical quantities remain guarded by either the
// exact chart invariant or that explicit exact-restriction contract.
//
// Vertices may share physical coordinates (a chart side collapsing to a pole).
// A quadrilateral with two coincident corners has no meaningful physical
// diagonal, so it is never flipped.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory_resource>
#include <utility>
#include <vector>

#include "bounded_memory.hpp"
#include "capi_guard.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc {
namespace {

using Triangle = std::array<int32_t, 3>;
using Flags = std::array<int8_t, 3>;

// Physical flips need the opposite-angle sum to exceed pi by this cotangent
// margin, so cocircular quadrilaterals never oscillate.
constexpr double kFlipMargin = 1.0e-12;

struct Vector3 {
  double x;
  double y;
  double z;
};

Vector3 operator-(const Vector3& a, const Vector3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

double dot(const Vector3& a, const Vector3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vector3 cross(const Vector3& a, const Vector3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double metric_dot(const Vector3& a, const Vector3& b, const double* metric) {
  if (metric == nullptr) {
    return dot(a, b);
  }
  return a.x * (metric[0] * b.x + metric[1] * b.y + metric[2] * b.z) +
         a.y * (metric[3] * b.x + metric[4] * b.y + metric[5] * b.z) +
         a.z * (metric[6] * b.x + metric[7] * b.y + metric[8] * b.z);
}

double metric_area(const Vector3& a, const Vector3& b, const double* metric, double product) {
  if (metric == nullptr) {
    const Vector3 area = cross(a, b);
    return std::sqrt(dot(area, area));
  }
  return std::sqrt(metric_dot(a, a, metric) * metric_dot(b, b, metric) - product * product);
}

bool metric_valid(const double* metric) {
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(metric[k])) {
      return false;
    }
  }
  if (metric[1] != metric[3] || metric[2] != metric[6] || metric[5] != metric[7] ||
      !(metric[0] > 0.0)) {
    return false;
  }
  const double first = std::sqrt(metric[0]);
  const double second = metric[3] / first;
  const double third = metric[6] / first;
  const double diagonal = metric[4] - second * second;
  if (!(diagonal > 0.0)) {
    return false;
  }
  const double last = (metric[7] - third * second) / std::sqrt(diagonal);
  return metric[8] - third * third - last * last > 0.0;
}

struct Counters {
  int64_t inserted = 0;
  int64_t refused = 0;
  int64_t flips = 0;
  int64_t walk_steps = 0;
  int64_t work = 0;
  int64_t exhausted = 0;
};

class SurfaceTriangulation {
 public:
  SurfaceTriangulation(int64_t work_limit)
      : charts(scratch_memory_resource()), points(scratch_memory_resource()),
        normals(scratch_memory_resource()), metrics(scratch_memory_resource()),
        pole_ids(scratch_memory_resource()), cells(scratch_memory_resource()),
        neighbors(scratch_memory_resource()), constrained(scratch_memory_resource()),
        vertex_cells(scratch_memory_resource()),
        work_limit_(work_limit) {}

  std::pmr::vector<double> charts;
  std::pmr::vector<double> points;
  std::pmr::vector<double> normals;
  std::pmr::vector<double> metrics;
  const double* insertion_metric = nullptr;
  std::pmr::vector<int64_t> pole_ids;
  int64_t insertion_pole_id = -1;
  std::pmr::vector<Triangle> cells;
  std::pmr::vector<Triangle> neighbors;
  std::pmr::vector<Flags> constrained;
  std::pmr::vector<int32_t> vertex_cells;

  void remember_cell(int32_t cell) {
    if (!vertex_cells.empty()) {
      for (const int32_t vertex : cells[cell]) {
        vertex_cells[vertex] = cell;
      }
    }
  }

  // Walk only the owning endpoint's incident fan, in both directions on an
  // open boundary. No point-location or whole-mesh search infers edge intent.
  int32_t requested_edge(int32_t a, int32_t b, int32_t hint, int& edge) {
    if (hint >= 0 && static_cast<std::size_t>(hint) < cells.size()) {
      if (!charge()) return -2;
      edge = opposite_index(hint, a, b);
      if (edge >= 0) return hint;
    }
    const int32_t start = vertex_cells[a];
    if (start < 0) return -1;
    for (int direction = 1; direction <= 2; ++direction) {
      int32_t cell = start;
      int32_t previous = -1;
      do {
        if (!charge()) return -2;
        edge = opposite_index(cell, a, b);
        if (edge >= 0) return cell;
        int corner = 0;
        while (corner < 3 && cells[cell][corner] != a) ++corner;
        if (corner == 3) return -1;
        const int first = (corner + direction) % 3;
        const int second = (corner + 3 - direction) % 3;
        int32_t next = neighbors[cell][first];
        if (next == previous) next = neighbors[cell][second];
        previous = cell;
        cell = next;
      } while (cell >= 0 && cell != start);
      if (cell == start) break;
    }
    return -1;
  }
  Counters counters;

  const double* chart(int32_t vertex) const { return charts.data() + 2 * vertex; }

  Vector3 point(int32_t vertex) const {
    const double* p = points.data() + 3 * static_cast<std::size_t>(vertex);
    return {p[0], p[1], p[2]};
  }

  bool coincident(int32_t a, int32_t b) const {
    const double* p = points.data() + 3 * static_cast<std::size_t>(a);
    const double* q = points.data() + 3 * static_cast<std::size_t>(b);
    return p[0] == q[0] && p[1] == q[1] && p[2] == q[2];
  }

  Vector3 normal(int32_t a, int32_t b, int32_t c) const {
    const Vector3 origin = point(a);
    return cross(point(b) - origin, point(c) - origin);
  }

  // Charges one unit of work; false once the budget is exhausted.
  bool charge() {
    if (counters.work >= work_limit_ || !native_execution_spend(1)) {
      counters.exhausted = 1;
      return false;
    }
    ++counters.work;
    return true;
  }

  // Builds reciprocal adjacency and validates the chart embedding.
  int32_t prepare() {
    const std::size_t count = cells.size();
    // Account for validation and adjacency sorting before allocating its
    // working set. Work units are bounded comparison/side visits, not timing.
    int64_t levels = 0;
    for (std::size_t width = 1; width < 3 * count; width *= 2) {
      ++levels;
    }
    const int64_t preparation_work = static_cast<int64_t>(count) * (4 + 3 * levels);
    if (preparation_work > work_limit_ - counters.work ||
        !native_execution_spend(preparation_work)) {
      counters.exhausted = 1;
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    counters.work += preparation_work;
    neighbors.assign(count, Triangle{-1, -1, -1});
    if (!vertex_cells.empty()) {
      std::fill(vertex_cells.begin(), vertex_cells.end(), -1);
    }
    struct Side {
      int64_t key;
      int32_t cell;
      int32_t index;
      bool forward;
    };
    std::pmr::vector<Side> sides{scratch_memory_resource()};
    sides.reserve(3 * count);
    const int64_t width = static_cast<int64_t>(charts.size() / 2);
    for (std::size_t cell = 0; cell < count; ++cell) {
      const Triangle& t = cells[cell];
      remember_cell(static_cast<int32_t>(cell));
      if (orient2d(chart(t[0]), chart(t[1]), chart(t[2])) <= 0) {
        return PHX_MC_INVALID_INPUT;
      }
      for (int index = 0; index < 3; ++index) {
        const int32_t a = t[(index + 1) % 3];
        const int32_t b = t[(index + 2) % 3];
        const int64_t low = std::min(a, b);
        const int64_t high = std::max(a, b);
        sides.push_back({low * width + high, static_cast<int32_t>(cell), index, a < b});
      }
    }
    std::sort(sides.begin(), sides.end(), [](const Side& left, const Side& right) {
      return left.key != right.key ? left.key < right.key
                                   : (left.cell != right.cell ? left.cell < right.cell
                                                              : left.index < right.index);
    });
    for (std::size_t start = 0; start < sides.size();) {
      std::size_t end = start + 1;
      while (end < sides.size() && sides[end].key == sides[start].key) {
        ++end;
      }
      const Side& first = sides[start];
      const bool first_constrained = constrained[first.cell][first.index] != 0;
      if (end - start == 1) {
        if (!first_constrained) {
          return PHX_MC_INVALID_INPUT;  // an open chart boundary edge must be constrained
        }
      } else if (end - start == 2) {
        const Side& second = sides[start + 1];
        if (first.forward == second.forward ||
            first_constrained != (constrained[second.cell][second.index] != 0)) {
          return PHX_MC_INVALID_INPUT;
        }
        neighbors[first.cell][first.index] = second.cell;
        neighbors[second.cell][second.index] = first.cell;
      } else {
        return PHX_MC_INVALID_INPUT;
      }
      start = end;
    }
    return PHX_MC_OK;
  }

  void replace_neighbor(int32_t cell, int32_t from, int32_t to) {
    if (cell < 0) {
      return;
    }
    for (int index = 0; index < 3; ++index) {
      if (neighbors[cell][index] == from) {
        neighbors[cell][index] = to;
        return;
      }
    }
  }

  // Rotation of one cell so that position `index` comes first.
  struct Rotated {
    Triangle vertices;
    Triangle adjacent;
    Flags flags;
  };

  Rotated rotated(int32_t cell, int index) const {
    Rotated result{};
    for (int offset = 0; offset < 3; ++offset) {
      result.vertices[offset] = cells[cell][(index + offset) % 3];
      result.adjacent[offset] = neighbors[cell][(index + offset) % 3];
      result.flags[offset] = constrained[cell][(index + offset) % 3];
    }
    return result;
  }

  int opposite_index(int32_t cell, int32_t a, int32_t b) const {
    for (int index = 0; index < 3; ++index) {
      const int32_t first = cells[cell][(index + 1) % 3];
      const int32_t second = cells[cell][(index + 2) % 3];
      if ((first == a && second == b) || (first == b && second == a)) {
        return index;
      }
    }
    return -1;
  }

  // Physical Delaunay criterion and exact chart legality of flipping the edge
  // opposite `index` of `cell`.
  bool flip_wanted(int32_t cell, int index) const {
    if (constrained[cell][index] != 0 || neighbors[cell][index] < 0) {
      return false;
    }
    const Rotated t = rotated(cell, index);
    const int32_t other = t.adjacent[0];
    const int32_t c = t.vertices[0];
    const int32_t a = t.vertices[1];
    const int32_t b = t.vertices[2];
    const int back = opposite_index(other, a, b);
    if (back < 0) {
      return false;
    }
    const int32_t d = cells[other][back];
    const std::array<int32_t, 4> corners{a, b, c, d};
    for (int first = 0; first < 4; ++first) {
      for (int second = first + 1; second < 4; ++second) {
        if (coincident(corners[first], corners[second])) {
          return false;
        }
      }
    }
    const Vector3 pa = point(a);
    const Vector3 pb = point(b);
    const Vector3 pc = point(c);
    const Vector3 pd = point(d);
    const Vector3 ca = pa - pc;
    const Vector3 cb = pb - pc;
    const Vector3 da = pa - pd;
    const Vector3 db = pb - pd;
    std::array<double, 9> average{};
    const double* metric = nullptr;
    if (!metrics.empty()) {
      for (const int32_t vertex : corners) {
        for (int k = 0; k < 9; ++k) {
          average[k] += 0.25 * metrics[9 * static_cast<std::size_t>(vertex) + k];
        }
      }
      metric = average.data();
    }
    const double product_c = metric_dot(ca, cb, metric);
    const double product_d = metric_dot(da, db, metric);
    const double sine_c = metric_area(ca, cb, metric, product_c);
    const double sine_d = metric_area(da, db, metric, product_d);
    if (!(sine_c > 0.0) || !(sine_d > 0.0)) {
      return false;
    }
    const double cotangents = product_c / sine_c + product_d / sine_d;
    if (!(cotangents < -kFlipMargin)) {
      return false;
    }
    if (orient2d(chart(c), chart(a), chart(d)) <= 0 ||
        orient2d(chart(c), chart(d), chart(b)) <= 0) {
      return false;
    }
    const Vector3 old_first = normal(c, a, b);
    const Vector3 old_second = normal(d, b, a);
    const Vector3 new_first = normal(c, a, d);
    const Vector3 new_second = normal(c, d, b);
    const Vector3 na = vertex_normal(a);
    const Vector3 nb = vertex_normal(b);
    const Vector3 nc = vertex_normal(c);
    const Vector3 nd = vertex_normal(d);
    const bool first_consistent = valid(old_first, {nc, na, nb});
    const bool second_consistent = valid(old_second, {nd, nb, na});
    // Every supplied source normal remains authoritative even when both old
    // triangles are coarse or folded. A flip may repair them, but it cannot
    // publish another nondegenerate triangle on the wrong side of the source.
    if (!agrees(new_first, {nc, na, nd}) || !agrees(new_second, {nc, nd, nb})) {
      return false;
    }
    // Without source normals the old triangles are the only orientation witness.
    const auto kept = [&](const Vector3& old) {
      return dot(new_first, old) > 0.0 && dot(new_second, old) > 0.0;
    };
    if (bare({na, nb, nc, nd}) &&
        ((first_consistent && !kept(old_first)) || (second_consistent && !kept(old_second)))) {
      return false;
    }
    return true;
  }

  // Whether no source normal is supplied at any of the given vertices.
  static bool bare(std::initializer_list<Vector3> at_vertices) {
    for (const Vector3& n : at_vertices) {
      if (dot(n, n) > 0.0) {
        return false;
      }
    }
    return true;
  }

  // Replaces triangles (c, a, b) and (d, b, a) by (c, a, d) and (c, d, b).
  void flip(int32_t cell, int index) {
    native_execution_charge(0);
    const Rotated t = rotated(cell, index);
    const int32_t other = t.adjacent[0];
    const Rotated u = rotated(other, opposite_index(other, t.vertices[1], t.vertices[2]));
    const int32_t c = t.vertices[0];
    const int32_t a = t.vertices[1];
    const int32_t b = t.vertices[2];
    const int32_t d = u.vertices[0];
    cells[cell] = {c, a, d};
    neighbors[cell] = {u.adjacent[1], other, t.adjacent[2]};
    constrained[cell] = {u.flags[1], 0, t.flags[2]};
    cells[other] = {c, d, b};
    neighbors[other] = {u.adjacent[2], t.adjacent[1], cell};
    constrained[other] = {u.flags[2], t.flags[1], 0};
    remember_cell(cell);
    remember_cell(other);
    replace_neighbor(u.adjacent[1], other, cell);
    replace_neighbor(t.adjacent[1], cell, other);
    ++counters.flips;
  }

  // Lawson legalization of the edges on the stack (cell hint, endpoints).
  void legalize(std::pmr::vector<std::array<int32_t, 3>>& stack) {
    while (!stack.empty()) {
      const std::array<int32_t, 3> entry = stack.back();
      stack.pop_back();
      const int32_t cell = entry[0];
      const int index = opposite_index(cell, entry[1], entry[2]);
      if (index < 0) {
        continue;
      }
      if (!charge()) {
        stack.clear();
        return;
      }
      if (!flip_wanted(cell, index)) {
        continue;
      }
      const int32_t other = neighbors[cell][index];
      flip(cell, index);
      // The four outer edges of the flipped quadrilateral may now be illegal.
      for (const int32_t changed : {cell, other}) {
        for (int k = 0; k < 3; ++k) {
          if (neighbors[changed][k] != cell && neighbors[changed][k] != other) {
            stack.push_back(
                {changed, cells[changed][(k + 1) % 3], cells[changed][(k + 2) % 3]});
          }
        }
      }
    }
  }

  void sweep() {
    std::pmr::vector<std::array<int32_t, 3>> stack{scratch_memory_resource()};
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      for (int k = 0; k < 3; ++k) {
        const int32_t a = cells[cell][(k + 1) % 3];
        const int32_t b = cells[cell][(k + 2) % 3];
        if (a < b) {
          stack.push_back({static_cast<int32_t>(cell), a, b});
        }
      }
    }
    std::reverse(stack.begin(), stack.end());
    legalize(stack);
  }

  // Exact chart location: the containing cell and the number of its edges
  // through the point (0 interior, 1 edge, 2 vertex), the edge in `edge`.
  // Returns -1 outside the triangulated chart domain, -2 on exhausted work.
  int32_t locate(const double* p, int32_t start, int& on_edges, int& edge) {
    const int32_t count = static_cast<int32_t>(cells.size());
    int32_t cell = (start >= 0 && start < count) ? start : 0;
    // A visibility walk may cycle in a non-Delaunay chart triangulation; the
    // walk is bounded and then replaced by an exhaustive exact scan.
    for (int32_t step = 0; step <= count; ++step) {
      if (!charge()) {
        return -2;
      }
      ++counters.walk_steps;
      int32_t next = -1;
      int zeros = 0;
      edge = -1;
      for (int offset = 0; offset < 3; ++offset) {
        const int k = (offset + step) % 3;
        const int sign = orient2d(chart(cells[cell][(k + 1) % 3]), chart(cells[cell][(k + 2) % 3]), p);
        if (sign < 0) {
          next = neighbors[cell][k];
          if (next < 0) {
            return -1;
          }
          break;
        }
        if (sign == 0) {
          ++zeros;
          edge = k;
        }
      }
      if (next < 0) {
        on_edges = zeros;
        return cell;
      }
      cell = next;
    }
    for (int32_t candidate = 0; candidate < count; ++candidate) {
      if (!charge()) {
        return -2;
      }
      int zeros = 0;
      bool inside = true;
      for (int k = 0; k < 3 && inside; ++k) {
        const int sign = orient2d(chart(cells[candidate][(k + 1) % 3]),
                                  chart(cells[candidate][(k + 2) % 3]), p);
        if (sign < 0) {
          inside = false;
        } else if (sign == 0) {
          ++zeros;
          edge = k;
        }
      }
      if (inside) {
        on_edges = zeros;
        return candidate;
      }
    }
    return -1;
  }

  bool too_close(const Vector3& p, const std::array<int32_t, 4>& star, double spacing) const {
    for (const int32_t cell : star) {
      if (cell < 0) {
        continue;
      }
      for (const int32_t vertex : cells[cell]) {
        const Vector3 offset = point(vertex) - p;
        std::array<double, 9> average{};
        const double* metric = nullptr;
        if (insertion_metric != nullptr) {
          for (int k = 0; k < 9; ++k) {
            average[k] = 0.5 * (insertion_metric[k] + metrics[9 * static_cast<std::size_t>(vertex) + k]);
          }
          metric = average.data();
        }
        if (metric_dot(offset, offset, metric) < spacing * spacing) {
          return true;
        }
      }
    }
    return false;
  }

  bool aligned(const Vector3& parent, std::initializer_list<Vector3> children) const {
    for (const Vector3& child : children) {
      if (!(dot(child, parent) > 0.0)) {
        return false;
      }
    }
    return true;
  }

  // Oriented source normal at a vertex (zero when the source supplied none).
  Vector3 vertex_normal(int32_t vertex) const {
    const double* n = normals.data() + 3 * static_cast<std::size_t>(vertex);
    return {n[0], n[1], n[2]};
  }

  // A nondegenerate triangle normal agreeing with every supplied vertex normal.
  static bool agrees(const Vector3& triangle, std::initializer_list<Vector3> at_vertices) {
    if (!(dot(triangle, triangle) > 0.0)) {
      return false;
    }
    for (const Vector3& n : at_vertices) {
      if (dot(n, n) > 0.0 && !(dot(triangle, n) > 0.0)) {
        return false;
      }
    }
    return true;
  }

  static bool valid(const Vector3& triangle, std::initializer_list<Vector3> at_vertices) {
    return agrees(triangle, at_vertices);
  }

  // Only a declared source pole may retain a collapsed chart child. Exact
  // physical coincidence also has to hold; ordinary equal coordinates never
  // infer pole identity or excuse a degenerate split.
  bool declared_radial(const Rotated& t) const {
    const int32_t c = t.vertices[0];
    const int32_t a = t.vertices[1];
    const int32_t b = t.vertices[2];
    const bool first_collapsed = t.flags[2] != 0 && pole_ids[c] >= 0 &&
                                 pole_ids[c] == pole_ids[a] && coincident(c, a);
    const bool second_collapsed = t.flags[1] != 0 && pole_ids[c] >= 0 &&
                                  pole_ids[c] == pole_ids[b] && coincident(c, b);
    return first_collapsed != second_collapsed;
  }


  bool radial_children(const Rotated& t, const Vector3& first, const Vector3& second,
                       const Vector3& inserted_normal) const {
    const int32_t c = t.vertices[0];
    const int32_t a = t.vertices[1];
    const int32_t b = t.vertices[2];
    const bool first_collapsed = t.flags[2] != 0 && pole_ids[c] >= 0 &&
                                 pole_ids[c] == pole_ids[a] && coincident(c, a);
    const bool second_collapsed = t.flags[1] != 0 && pole_ids[c] >= 0 &&
                                  pole_ids[c] == pole_ids[b] && coincident(c, b);
    if (first_collapsed == second_collapsed) {
      return false;
    }
    const Vector3 nc = vertex_normal(c);
    const Vector3 na = vertex_normal(a);
    const Vector3 nb = vertex_normal(b);
    return (first_collapsed ? dot(first, first) == 0.0
                            : agrees(first, {nc, na, inserted_normal})) &&
           (second_collapsed ? dot(second, second) == 0.0
                             : agrees(second, {nc, inserted_normal, nb}));
  }


  int32_t append_vertex(const double* uv, const double* xyz, const double* n) {
    const int32_t vertex = static_cast<int32_t>(charts.size() / 2);
    charts.insert(charts.end(), uv, uv + 2);
    points.insert(points.end(), xyz, xyz + 3);
    normals.insert(normals.end(), n, n + 3);
    pole_ids.push_back(insertion_pole_id);
    if (!vertex_cells.empty()) vertex_cells.push_back(-1);
    if (insertion_metric != nullptr) {
      metrics.insert(metrics.end(), insertion_metric, insertion_metric + 9);
    }
    return vertex;
  }

  int32_t append_cell(const Triangle& vertices, const Triangle& adjacent, const Flags& flags) {
    cells.push_back(vertices);
    neighbors.push_back(adjacent);
    constrained.push_back(flags);
    remember_cell(static_cast<int32_t>(cells.size() - 1));
    return static_cast<int32_t>(cells.size() - 1);
  }

  // Splits `cell` at an interior point into (v0, v1, p), (v1, v2, p), (v2, v0, p).
  int32_t split_cell(int32_t cell, const double* uv, const double* xyz, const double* n,
                     std::pmr::vector<std::array<int32_t, 3>>& stack) {
    native_execution_charge(0);
    const Rotated t = rotated(cell, 0);
    const int32_t v0 = t.vertices[0];
    const int32_t v1 = t.vertices[1];
    const int32_t v2 = t.vertices[2];
    const int32_t p = append_vertex(uv, xyz, n);
    const int32_t second = static_cast<int32_t>(cells.size());
    const int32_t third = second + 1;
    cells[cell] = {v0, v1, p};
    neighbors[cell] = {second, third, t.adjacent[2]};
    constrained[cell] = {0, 0, t.flags[2]};
    append_cell({v1, v2, p}, {third, cell, t.adjacent[0]}, {0, 0, t.flags[0]});
    append_cell({v2, v0, p}, {cell, second, t.adjacent[1]}, {0, 0, t.flags[1]});
    remember_cell(cell);
    replace_neighbor(t.adjacent[0], cell, second);
    replace_neighbor(t.adjacent[1], cell, third);
    stack.push_back({cell, v0, v1});
    stack.push_back({second, v1, v2});
    stack.push_back({third, v2, v0});
    return p;
  }

  // Splits the unconstrained edge (a, b) of (c, a, b) and (d, b, a) at p.
  int32_t split_edge(int32_t cell, int index, const double* uv, const double* xyz,
                     const double* n, std::pmr::vector<std::array<int32_t, 3>>& stack) {
    native_execution_charge(0);
    const Rotated t = rotated(cell, index);
    const int32_t other = t.adjacent[0];
    const Rotated u = rotated(other, opposite_index(other, t.vertices[1], t.vertices[2]));
    const int32_t c = t.vertices[0];
    const int32_t a = t.vertices[1];
    const int32_t b = t.vertices[2];
    const int32_t d = u.vertices[0];
    const int32_t p = append_vertex(uv, xyz, n);
    const int32_t t2 = static_cast<int32_t>(cells.size());
    const int32_t u2 = t2 + 1;
    cells[cell] = {c, a, p};
    neighbors[cell] = {u2, t2, t.adjacent[2]};
    constrained[cell] = {0, 0, t.flags[2]};
    cells[other] = {d, b, p};
    neighbors[other] = {t2, u2, u.adjacent[2]};
    constrained[other] = {0, 0, u.flags[2]};
    append_cell({c, p, b}, {other, t.adjacent[1], cell}, {0, t.flags[1], 0});
    append_cell({d, p, a}, {cell, u.adjacent[1], other}, {0, u.flags[1], 0});
    remember_cell(cell);
    remember_cell(other);
    replace_neighbor(t.adjacent[1], cell, t2);
    replace_neighbor(u.adjacent[1], other, u2);
    stack.push_back({cell, c, a});
    stack.push_back({t2, b, c});
    stack.push_back({other, d, b});
    stack.push_back({u2, a, d});
    return p;
  }

  // Inserts one point; returns its vertex id or -1 with `status` set.
  int32_t insert(const double* uv, const double* xyz, const double* n, double spacing,
                 int32_t hint, const int32_t* requested, bool legalize_edges,
                 int64_t max_cells, int32_t& status) {
    status = PHX_MC_OK;
    if (static_cast<int64_t>(cells.size()) + 2 > max_cells) {
      status = PHX_MC_CAPACITY_EXCEEDED;
      return -1;
    }
    int on_edges = 0;
    int edge = -1;
    const bool explicit_edge = requested != nullptr && requested[0] != -1;
    const int32_t cell = explicit_edge
        ? requested_edge(requested[0], requested[1], hint, edge)
        : locate(uv, hint, on_edges, edge);
    if (explicit_edge) on_edges = 1;
    if (cell == -2) {
      status = PHX_MC_CAPACITY_EXCEEDED;
      return -1;
    }
    if (cell < 0) {
      status = PHX_MC_INVALID_INPUT;
      return -1;
    }
    if (on_edges >= 2) {
      status = PHX_MC_DEGENERATE_INPUT;
      return -1;
    }
    const Vector3 p{xyz[0], xyz[1], xyz[2]};
    const std::array<int32_t, 4> star{cell, neighbors[cell][0], neighbors[cell][1],
                                       neighbors[cell][2]};
    if (too_close(p, star, spacing)) {
      status = PHX_MC_DEGENERATE_INPUT;
      return -1;
    }
    std::pmr::vector<std::array<int32_t, 3>> stack{scratch_memory_resource()};
    if (!native_execution_cavity(on_edges == 1 ? 4 : 3)) {
      status = PHX_MC_CAPACITY_EXCEEDED;
      return -1;
    }
    int32_t vertex = -1;
    if (on_edges == 1) {
      if (constrained[cell][edge] != 0) {
        status = PHX_MC_CONSTRAINT_INTERSECTION;
        return -1;
      }
      const Rotated t = rotated(cell, edge);
      const int32_t other = t.adjacent[0];
      const int back = other >= 0 ? opposite_index(other, t.vertices[1], t.vertices[2]) : -1;
      if (back < 0 || neighbors[other][back] != cell ||
          cells[other][(back + 1) % 3] != t.vertices[2] ||
          cells[other][(back + 2) % 3] != t.vertices[1] ||
          constrained[other][back] != 0) {
        status = PHX_MC_INVALID_INPUT;
        return -1;
      }
      const Rotated u = rotated(other, back);
      const int32_t d = u.vertices[0];
      const int32_t c = t.vertices[0];
      const int32_t a = t.vertices[1];
      const int32_t b = t.vertices[2];
      const double* chart_a = chart(a);
      const double* chart_b = chart(b);
      const bool rounded_exact_restriction =
          explicit_edge && !legalize_edges &&
          uv[0] >= std::min(chart_a[0], chart_b[0]) &&
          uv[0] <= std::max(chart_a[0], chart_b[0]) &&
          uv[1] >= std::min(chart_a[1], chart_b[1]) &&
          uv[1] <= std::max(chart_a[1], chart_b[1]) &&
          (uv[0] != chart_a[0] || uv[1] != chart_a[1]) &&
          (uv[0] != chart_b[0] || uv[1] != chart_b[1]);
      if (!rounded_exact_restriction &&
          (orient2d(chart(c), chart(a), uv) <= 0 ||
           orient2d(chart(c), uv, chart(b)) <= 0 ||
           orient2d(chart(d), chart(b), uv) <= 0 ||
           orient2d(chart(d), uv, chart(a)) <= 0)) {
        status = PHX_MC_DEGENERATE_INPUT;
        return -1;
      }
      // Prove all four outer sides are conserved reciprocal boundary sides
      // before append_vertex or any topology mutation.
      for (int owner_index = 0; owner_index < 2; ++owner_index) {
        const int32_t owner_cell = owner_index == 0 ? cell : other;
        const Rotated& owner = owner_index == 0 ? t : u;
        for (int side = 1; side <= 2; ++side) {
          if (!charge()) {
            status = PHX_MC_CAPACITY_EXCEEDED;
            return -1;
          }
          const int32_t adjacent = owner.adjacent[side];
          if (adjacent < 0) {
            if (owner.flags[side] != 0) continue;
          } else if (adjacent != cell && adjacent != other) {
            const int32_t first = owner.vertices[(side + 1) % 3];
            const int32_t second = owner.vertices[(side + 2) % 3];
            const int reciprocal = opposite_index(adjacent, first, second);
            if (reciprocal >= 0 && neighbors[adjacent][reciprocal] == owner_cell &&
                cells[adjacent][(reciprocal + 1) % 3] == second &&
                cells[adjacent][(reciprocal + 2) % 3] == first &&
                constrained[adjacent][reciprocal] == owner.flags[side]) continue;
          }
          status = PHX_MC_INVALID_INPUT;
          return -1;
        }
      }
      const Vector3 pc = point(c);
      const Vector3 pd = point(d);
      // Sub-triangles (c, a, p), (c, p, b) and (d, b, p), (d, p, a).
      const Vector3 ca = cross(point(a) - pc, p - pc);
      const Vector3 pb = cross(p - pc, point(b) - pc);
      const Vector3 db = cross(point(b) - pd, p - pd);
      const Vector3 pa = cross(p - pd, point(a) - pd);
      const Vector3 parent_t = normal(c, a, b);
      const Vector3 parent_u = normal(d, b, a);
      const Vector3 np{n[0], n[1], n[2]};
      const Vector3 nc = vertex_normal(c);
      const Vector3 na = vertex_normal(a);
      const Vector3 nb = vertex_normal(b);
      const Vector3 nd = vertex_normal(d);
      // Children of a surface-consistent parent stay consistent and, without
      // source normals, on the parent's side. Children of a coarse or folded
      // parent still honor every supplied source normal; nondegeneracy alone
      // cannot establish physical orientation. A terminal exact restriction
      // on a declared radial pole complex is validated against its authoritative
      // chart and continuous source by the producer after this topology-only
      // split; its rounded execution view cannot own these sign decisions.
      const bool legal_t =
          valid(parent_t, {nc, na, nb})
              ? agrees(ca, {nc, na, np}) && agrees(pb, {nc, np, nb}) &&
                    (!bare({nc, na, nb, np}) || aligned(parent_t, {ca, pb}))
              : radial_children(t, ca, pb, np) ||
                    (agrees(ca, {nc, na, np}) && agrees(pb, {nc, np, nb}));
      const bool legal_u =
          valid(parent_u, {nd, nb, na})
              ? agrees(db, {nd, nb, np}) && agrees(pa, {nd, np, na}) &&
                    (!bare({nd, nb, na, np}) || aligned(parent_u, {db, pa}))
              : radial_children(u, db, pa, np) ||
                    (agrees(db, {nd, nb, np}) && agrees(pa, {nd, np, na}));
      const bool rounded_pole_restriction =
          rounded_exact_restriction && (declared_radial(t) || declared_radial(u));
      if (!rounded_pole_restriction && (!legal_t || !legal_u)) {
        status = PHX_MC_DEGENERATE_INPUT;
        return -1;
      }
      vertex = split_edge(cell, edge, uv, xyz, n, stack);
    } else {
      const Triangle& t = cells[cell];
      const Vector3 parent = normal(t[0], t[1], t[2]);
      const Vector3 first = cross(point(t[1]) - point(t[0]), p - point(t[0]));
      const Vector3 second = cross(point(t[2]) - point(t[1]), p - point(t[1]));
      const Vector3 third = cross(point(t[0]) - point(t[2]), p - point(t[2]));
      const Vector3 np{n[0], n[1], n[2]};
      const Vector3 n0 = vertex_normal(t[0]);
      const Vector3 n1 = vertex_normal(t[1]);
      const Vector3 n2 = vertex_normal(t[2]);
      const bool legal =
          valid(parent, {n0, n1, n2})
              ? agrees(first, {n0, n1, np}) && agrees(second, {n1, n2, np}) &&
                    agrees(third, {n2, n0, np}) &&
                    (!bare({n0, n1, n2, np}) || aligned(parent, {first, second, third}))
              : agrees(first, {n0, n1, np}) && agrees(second, {n1, n2, np}) &&
                    agrees(third, {n2, n0, np});
      if (!legal) {
        status = PHX_MC_DEGENERATE_INPUT;
        return -1;
      }
      vertex = split_cell(cell, uv, xyz, n, stack);
    }
    ++counters.inserted;
    // An owning exact restriction calls with `sweep == 0`: the rounded chart
    // point is only an execution representative, so neither local nor final
    // Lawson flips may change its authoritative rational complex. Ordinary
    // requested edges and generic insertions retain physical Delaunay
    // legalization.
    if (legalize_edges) {
      legalize(stack);
    }
    return vertex;
  }

 private:
  int64_t work_limit_;
};

bool chart_in_domain(const double* uv) {
  return std::isfinite(uv[0]) && std::isfinite(uv[1]) && coordinate_in_domain(uv[0]) &&
         coordinate_in_domain(uv[1]);
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_surface_reconnect(int64_t vertex_count, const double* charts, const double* points,
                                 const double* normals, const double* vertex_metrics,
                                 const int64_t* vertex_pole_ids,
                                 int64_t triangle_count,
                                 const int32_t* triangles, const int8_t* constrained,
                                 int64_t insert_count, const double* insert_charts,
                                 const double* insert_points, const double* insert_normals,
                                 const double* insert_metrics,
                                 const int64_t* insert_pole_ids,
                                 const double* insert_spacing, const int32_t* insert_hints,
                                 const int32_t* insert_edges,
                                 int32_t sweep, int64_t max_triangles, int64_t work_limit,
                                 int32_t* triangles_out, int8_t* constrained_out,
                                 int64_t* triangle_count_out, int32_t* vertex_ids,
                                 int32_t* item_status, int64_t* counters) {
  return phx::mc::guarded([&]() -> int32_t {
    using phx::mc::SurfaceTriangulation;
    if (vertex_count < 3 || vertex_count >= INT32_MAX || triangle_count < 1 ||
        insert_count < 0 || insert_count >= INT32_MAX || work_limit < 0 ||
        max_triangles < triangle_count || max_triangles >= INT32_MAX ||
        vertex_count + insert_count >= INT32_MAX ||
        !phx::mc::addressable(vertex_count, 3, sizeof(double)) ||
        !phx::mc::addressable(max_triangles, 3, sizeof(int32_t)) ||
        !phx::mc::addressable(insert_count, 3, sizeof(double)) || charts == nullptr ||
        (vertex_metrics == nullptr) != (insert_metrics == nullptr) ||
        (vertex_metrics != nullptr &&
         (!phx::mc::addressable(vertex_count + insert_count, 9, sizeof(double)))) ||
        !phx::mc::addressable(vertex_count + insert_count, 1, sizeof(int64_t)) ||
        points == nullptr || normals == nullptr || triangles == nullptr || constrained == nullptr ||
        triangles_out == nullptr || constrained_out == nullptr ||
        triangle_count_out == nullptr || counters == nullptr ||
        (insert_count > 0 &&
         (insert_charts == nullptr || insert_points == nullptr || insert_normals == nullptr ||
          insert_spacing == nullptr ||
          insert_hints == nullptr || vertex_ids == nullptr || item_status == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::native_execution_charge(0);
    SurfaceTriangulation surface(work_limit);
    const int64_t vertex_capacity = vertex_count + insert_count;
    const int64_t cell_capacity = std::min(max_triangles, triangle_count + 2 * insert_count);
    surface.charts.reserve(static_cast<std::size_t>(2 * vertex_capacity));
    surface.points.reserve(static_cast<std::size_t>(3 * vertex_capacity));
    surface.normals.reserve(static_cast<std::size_t>(3 * vertex_capacity));
    surface.pole_ids.reserve(static_cast<std::size_t>(vertex_capacity));
    if (insert_edges != nullptr &&
        std::any_of(insert_edges, insert_edges + 2 * insert_count,
                    [](int32_t vertex) { return vertex >= 0; })) {
      surface.vertex_cells.reserve(static_cast<std::size_t>(vertex_capacity));
      surface.vertex_cells.assign(static_cast<std::size_t>(vertex_count), -1);
    }
    if (vertex_pole_ids == nullptr) {
      surface.pole_ids.assign(static_cast<std::size_t>(vertex_count), -1);
    } else {
      for (int64_t vertex = 0; vertex < vertex_count; ++vertex) {
        if (vertex_pole_ids[vertex] < -1) {
          return PHX_MC_INVALID_INPUT;
        }
      }
      surface.pole_ids.assign(vertex_pole_ids, vertex_pole_ids + vertex_count);
    }
    surface.cells.reserve(static_cast<std::size_t>(cell_capacity));
    surface.neighbors.reserve(static_cast<std::size_t>(cell_capacity));
    surface.constrained.reserve(static_cast<std::size_t>(cell_capacity));
    surface.charts.assign(charts, charts + 2 * vertex_count);
    surface.points.assign(points, points + 3 * vertex_count);
    surface.normals.assign(normals, normals + 3 * vertex_count);
    if (vertex_metrics != nullptr) {
      for (int64_t vertex = 0; vertex < vertex_count; ++vertex) {
        if (!phx::mc::metric_valid(vertex_metrics + 9 * vertex)) {
          return PHX_MC_INVALID_INPUT;
        }
      }
      surface.metrics.reserve(static_cast<std::size_t>(9 * vertex_capacity));
      surface.metrics.assign(vertex_metrics, vertex_metrics + 9 * vertex_count);
    }
    for (int64_t vertex = 0; vertex < vertex_count; ++vertex) {
      if (!phx::mc::chart_in_domain(charts + 2 * vertex)) {
        return std::isfinite(charts[2 * vertex]) && std::isfinite(charts[2 * vertex + 1])
                   ? PHX_MC_RANGE_ERROR
                   : PHX_MC_NONFINITE_INPUT;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(points[3 * vertex + axis]) ||
            !std::isfinite(normals[3 * vertex + axis])) {
          return PHX_MC_NONFINITE_INPUT;
        }
      }
    }
    surface.cells.resize(static_cast<std::size_t>(triangle_count));
    surface.constrained.resize(static_cast<std::size_t>(triangle_count));
    for (int64_t cell = 0; cell < triangle_count; ++cell) {
      for (int k = 0; k < 3; ++k) {
        const int32_t vertex = triangles[3 * cell + k];
        if (vertex < 0 || vertex >= vertex_count) {
          return PHX_MC_INVALID_INPUT;
        }
        surface.cells[cell][k] = vertex;
        surface.constrained[cell][k] = constrained[3 * cell + k] != 0 ? 1 : 0;
      }
      const auto& t = surface.cells[cell];
      if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    const int32_t prepared = surface.prepare();
    if (prepared != PHX_MC_OK && prepared != PHX_MC_CAPACITY_EXCEEDED) {
      return prepared;
    }
    int32_t last = 0;
    for (int64_t item = 0; item < insert_count; ++item) {
      const double* uv = insert_charts + 2 * item;
      const double* xyz = insert_points + 3 * item;
      const double spacing = insert_spacing[item];
      int32_t status = PHX_MC_OK;
      int32_t vertex = -1;
      const double* n = insert_normals + 3 * item;
      const int32_t* requested = insert_edges == nullptr ? nullptr : insert_edges + 2 * item;
      surface.insertion_metric = insert_metrics == nullptr ? nullptr : insert_metrics + 9 * item;
      surface.insertion_pole_id = insert_pole_ids == nullptr ? -1 : insert_pole_ids[item];
      if (!std::isfinite(uv[0]) || !std::isfinite(uv[1]) || !std::isfinite(xyz[0]) ||
          !std::isfinite(xyz[1]) || !std::isfinite(xyz[2]) || !std::isfinite(n[0]) ||
          !std::isfinite(n[1]) || !std::isfinite(n[2]) || !std::isfinite(spacing) ||
          spacing < 0.0) {
        status = PHX_MC_NONFINITE_INPUT;
      } else if (surface.insertion_metric != nullptr &&
                 !phx::mc::metric_valid(surface.insertion_metric)) {
        status = PHX_MC_NONFINITE_INPUT;
      } else if (surface.insertion_pole_id < -1) {
        status = PHX_MC_INVALID_INPUT;
      } else if (!phx::mc::chart_in_domain(uv)) {
        status = PHX_MC_RANGE_ERROR;
      } else if (requested != nullptr &&
                 !((requested[0] == -1 && requested[1] == -1) ||
                   (requested[0] >= 0 && requested[1] >= 0 &&
                    requested[0] < vertex_count && requested[1] < vertex_count &&
                    requested[0] != requested[1]))) {
        status = PHX_MC_INVALID_INPUT;
      } else if (surface.counters.exhausted != 0) {
        status = PHX_MC_CAPACITY_EXCEEDED;
      } else {
        const int32_t hint = insert_hints[item] >= 0 ? insert_hints[item] : last;
        vertex = surface.insert(uv, xyz, insert_normals + 3 * item, spacing, hint, requested,
                                sweep != 0, max_triangles, status);
        if (vertex >= 0) {
          last = static_cast<int32_t>(surface.cells.size()) - 1;
        }
      }
      if (vertex < 0) {
        ++surface.counters.refused;
      }
      vertex_ids[item] = vertex;
      item_status[item] = status;
    }
    if (sweep != 0 && surface.counters.exhausted == 0) {
      surface.sweep();
    }
    phx::mc::native_execution_charge(0);
    const std::size_t produced = surface.cells.size();
    for (std::size_t cell = 0; cell < produced; ++cell) {
      for (int k = 0; k < 3; ++k) {
        triangles_out[3 * cell + k] = surface.cells[cell][k];
        constrained_out[3 * cell + k] = surface.constrained[cell][k];
      }
    }
    *triangle_count_out = static_cast<int64_t>(produced);
    const phx::mc::Counters& c = surface.counters;
    const int64_t values[PHX_MC_SURFACE_COUNTERS] = {c.inserted, c.refused,   c.flips,
                                                     c.walk_steps, c.work, c.exhausted};
    std::copy(values, values + PHX_MC_SURFACE_COUNTERS, counters);
    return PHX_MC_OK;
  });
}

}  // extern "C"
