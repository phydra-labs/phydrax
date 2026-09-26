//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact clipping of convex polygons (Sutherland-Hodgman with symbolic
// vertices): intersection moments of convex polygon pairs and axis-aligned
// boxes clipped by halfplanes.
//
// Every polygon vertex is an input point or the intersection of two input
// lines.  A line is point-defined (p, q) with side s(x) = orient2d(p, q, x)
// (inside: left, s >= 0) or explicit {x : n . x = h} with s(x) = h - n . x.
// Sides of a line-line vertex X with respect to a line Q are exact:
//   * first line point-defined, X = p1 + t (q1 - p1), t = s2(p1) / (s2(p1) - s2(q1)):
//       sQ(X) = (s2(p1) sQ(q1) - s2(q1) sQ(p1)) / (s2(p1) - s2(q1));
//   * both lines explicit, in coordinates y = x - o relative to an input point
//     o with g_i = s_i(o): the 3x3 block determinant gives
//       det[[n1, g1], [n2, g2], [nQ, gQ]] = det[n1; n2] sQ(X),
//     expanded along the last row with cofactors (C0, C1, C2 = det[n1; n2]):
//       C2 sQ(X) = C0 nQ0 + C1 nQ1 + C2 sQ(o), and y_k = -C_k / C2.
// Explicit-explicit vertices only arise in box clipping, where every clip
// line is explicit.  All decisions use the filter, then expansions.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "clip_common.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc {
namespace {

using clip::construction_value;
using clip::diff;
using clip::exact_sign;
using clip::kConstructionTolerance;
using clip::kInvalidSide;
using clip::lift;
using clip::mul;

struct Line {
  bool is_explicit = false;
  int p = -1;
  int q = -1;
  double n[2] = {0.0, 0.0};
  double h = 0.0;
  int label = 0;
};

// point >= 0: input point.  Otherwise the intersection of lines `first` and
// `second`, where `first` is point-defined unless both are explicit;
// denominator = sign(s2(p1) - s2(q1)) or sign(det[n1; n2]).
struct Vertex {
  int point = -1;
  int first = -1;
  int second = -1;
  int denominator = 0;
  double x[2] = {0.0, 0.0};
};

template <class T>
T line_value(const Line& line, const double* points, const double* x) {
  if (line.is_explicit) {
    return lift<T>(line.h) - (mul<T>(line.n[0], x[0]) + mul<T>(line.n[1], x[1]));
  }
  const double* p = points + 2 * line.p;
  const double* q = points + 2 * line.q;
  return diff<T>(q[0], p[0]) * diff<T>(x[1], p[1]) - diff<T>(q[1], p[1]) * diff<T>(x[0], p[0]);
}

template <class T>
void explicit_cofactors(const Line& first, const Line& second, const double* points,
                        const double* origin, T* c) {
  const T g1 = line_value<T>(first, points, origin);
  const T g2 = line_value<T>(second, points, origin);
  const T a0 = lift<T>(first.n[0]);
  const T a1 = lift<T>(first.n[1]);
  const T b0 = lift<T>(second.n[0]);
  const T b1 = lift<T>(second.n[1]);
  c[0] = a1 * g2 - g1 * b1;
  c[1] = g1 * b0 - a0 * g2;
  c[2] = a0 * b1 - a1 * b0;
}

class PolygonClipper {
 public:
  std::vector<double> points;   // (k, 2) input points
  std::vector<Line> lines;
  std::vector<Vertex> vertices;
  std::vector<int> loop;        // vertex ids, counterclockwise
  std::vector<int> edge_lines;  // supporting line of the edge loop[k] -> loop[k + 1]
  int origin = 0;               // input point: frame of explicit-explicit vertices
  int64_t capacity = std::numeric_limits<int64_t>::max();
  bool clamp_to_box = false;
  double lower[2] = {0.0, 0.0};
  double upper[2] = {0.0, 0.0};

  void reset() {
    points.clear();
    lines.clear();
    vertices.clear();
    loop.clear();
    edge_lines.clear();
    origin = 0;
    capacity = std::numeric_limits<int64_t>::max();
    clamp_to_box = false;
  }

  int add_point_vertex(int point) {
    Vertex vertex;
    vertex.point = point;
    vertex.x[0] = points[2 * point];
    vertex.x[1] = points[2 * point + 1];
    vertices.push_back(vertex);
    return static_cast<int>(vertices.size()) - 1;
  }

  // Keeps the part with s_line >= 0.  `empty` reports a zero-measure result.
  int32_t clip(int line_id, bool& empty) {
    empty = false;
    const Line& line = lines[line_id];
    const std::size_t n = loop.size();
    sides_.resize(n);
    bool positive = false;
    bool negative = false;
    int64_t kept = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const int side = classify(vertices[loop[i]], line);
      if (side == kInvalidSide) {
        return PHX_MC_INTERNAL_ERROR;
      }
      sides_[i] = static_cast<signed char>(side);
      positive = positive || side > 0;
      negative = negative || side < 0;
      kept += side >= 0;
    }
    if (!positive) {
      loop.clear();
      edge_lines.clear();
      empty = true;
      return PHX_MC_OK;
    }
    if (!negative) {
      return PHX_MC_OK;
    }
    int64_t crossings = 0;
    for (std::size_t i = 0; i < n; ++i) {
      crossings += sides_[i] * sides_[(i + 1) % n] < 0;
    }
    if (kept + crossings > capacity) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    next_loop_.clear();
    next_edge_lines_.clear();
    for (std::size_t i = 0; i < n; ++i) {
      const int si = sides_[i];
      const int sj = sides_[(i + 1) % n];
      if (si >= 0) {
        next_loop_.push_back(loop[i]);
        next_edge_lines_.push_back(si == 0 && sj < 0 ? line_id : edge_lines[i]);
      }
      if (si * sj < 0) {
        int id = -1;
        const int32_t status = intersection_vertex(edge_lines[i], line_id, id);
        if (status != PHX_MC_OK) {
          return status;
        }
        next_loop_.push_back(id);
        next_edge_lines_.push_back(si > 0 ? line_id : edge_lines[i]);
      }
    }
    loop.swap(next_loop_);
    edge_lines.swap(next_edge_lines_);
    return loop.size() >= 3 ? PHX_MC_OK : PHX_MC_INTERNAL_ERROR;
  }

  // Vertex average o of the loop: inside the polygon, so every fan term below
  // is nonnegative up to the rounding of o.
  void vertex_average(double* o) const {
    o[0] = 0.0;
    o[1] = 0.0;
    for (const int v : loop) {
      o[0] += vertices[v].x[0];
      o[1] += vertices[v].x[1];
    }
    o[0] /= static_cast<double>(loop.size());
    o[1] /= static_cast<double>(loop.size());
  }

  // Shoelace area and first moment (integral of x) relative to the vertex
  // average o (rounding errors scale with the polygon).
  void moments(double& area, double* first_moment) const {
    double o[2];
    vertex_average(o);
    clip::CompensatedSum twice;
    clip::CompensatedSum sx;
    clip::CompensatedSum sy;
    const std::size_t n = loop.size();
    for (std::size_t i = 0; i < n; ++i) {
      const double* a = vertices[loop[i]].x;
      const double* b = vertices[loop[(i + 1) % n]].x;
      const double ax = a[0] - o[0];
      const double ay = a[1] - o[1];
      const double bx = b[0] - o[0];
      const double by = b[1] - o[1];
      const double cross = clip::difference_of_products(ax, by, bx, ay);
      twice.add(cross);
      sx.add((ax + bx) * cross);
      sy.add((ay + by) * cross);
    }
    area = 0.5 * twice.value();
    first_moment[0] = sx.value() / 6.0 + o[0] * area;
    first_moment[1] = sy.value() / 6.0 + o[1] * area;
  }

  // The fan triangles (o, v_i, v_{i+1}) of moments(): a signed partition of the
  // polygon whose terms are exactly the shoelace terms of moments().
  int32_t fan_simplices(clip::SimplexOutput& out) const {
    const std::size_t n = loop.size();
    if (static_cast<int64_t>(n) > out.capacity) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    double o[2];
    vertex_average(o);
    for (std::size_t i = 0; i < n; ++i) {
      const double* a = vertices[loop[i]].x;
      const double* b = vertices[loop[(i + 1) % n]].x;
      double* row = out.simplices + 6 * i;
      row[0] = o[0];
      row[1] = o[1];
      row[2] = a[0];
      row[3] = a[1];
      row[4] = b[0];
      row[5] = b[1];
    }
    out.count = static_cast<int32_t>(n);
    return PHX_MC_OK;
  }

 private:
  int classify(const Vertex& vertex, const Line& line) const {
    const double* pts = points.data();
    if (vertex.point >= 0) {
      if (!line.is_explicit && (vertex.point == line.p || vertex.point == line.q)) {
        return 0;
      }
      const double* x = pts + 2 * vertex.point;
      return exact_sign([&]<class T>() { return line_value<T>(line, pts, x); });
    }
    const Line& first = lines[vertex.first];
    const Line& second = lines[vertex.second];
    if (!first.is_explicit) {
      const double* p1 = pts + 2 * first.p;
      const double* q1 = pts + 2 * first.q;
      return vertex.denominator * exact_sign([&]<class T>() {
               return line_value<T>(second, pts, p1) * line_value<T>(line, pts, q1) -
                      line_value<T>(second, pts, q1) * line_value<T>(line, pts, p1);
             });
    }
    if (!line.is_explicit) {
      return kInvalidSide;
    }
    const double* o = pts + 2 * origin;
    return vertex.denominator * exact_sign([&]<class T>() {
             T c[3];
             explicit_cofactors<T>(first, second, pts, o, c);
             return c[0] * lift<T>(line.n[0]) + c[1] * lift<T>(line.n[1]) +
                    c[2] * line_value<T>(line, pts, o);
           });
  }

  // Vertex where the edge on `edge_line` crosses `clip_line` strictly.  The
  // lines are distinct and not parallel because the edge endpoints lie
  // strictly on opposite sides of the clip line.
  int32_t intersection_vertex(int edge_line, int clip_line, int& id) {
    const Line& a = lines[edge_line];
    const Line& b = lines[clip_line];
    if (!a.is_explicit && !b.is_explicit) {
      int shared = -1;
      if (a.p == b.p || a.p == b.q) {
        shared = a.p;
      } else if (a.q == b.p || a.q == b.q) {
        shared = a.q;
      }
      if (shared >= 0) {
        id = add_point_vertex(shared);
        return PHX_MC_OK;
      }
    }
    Vertex vertex;
    vertex.first = edge_line;
    vertex.second = clip_line;
    if (a.is_explicit && !b.is_explicit) {
      std::swap(vertex.first, vertex.second);
    }
    const Line& first = lines[vertex.first];
    const Line& second = lines[vertex.second];
    const double* pts = points.data();
    if (!first.is_explicit) {
      const double* p1 = pts + 2 * first.p;
      const double* q1 = pts + 2 * first.q;
      vertex.denominator = exact_sign([&]<class T>() {
        return line_value<T>(second, pts, p1) - line_value<T>(second, pts, q1);
      });
      if (vertex.denominator == 0) {
        return PHX_MC_INTERNAL_ERROR;
      }
      // X lies on the closed segment [p1, q1] (an edge of an input polygon or
      // box), so both parameters are clamped to [0, 1].
      const double sp = construction_value(line_value<Approx>(second, pts, p1),
                                           [&] { return line_value<Expansion>(second, pts, p1); });
      const double sq = construction_value(line_value<Approx>(second, pts, q1),
                                           [&] { return line_value<Expansion>(second, pts, q1); });
      const double denominator = sp - sq;
      const double t = clip::clamp(sp / denominator, 0.0, 1.0);
      const double r = clip::clamp(-sq / denominator, 0.0, 1.0);
      for (int k = 0; k < 2; ++k) {
        vertex.x[k] = t <= 0.5 ? p1[k] + t * (q1[k] - p1[k]) : q1[k] + r * (p1[k] - q1[k]);
      }
    } else {
      const double* o = pts + 2 * origin;
      Approx c[3];
      explicit_cofactors<Approx>(first, second, pts, o, c);
      vertex.denominator = clip::exact_sign(c[2], [&] {
        return mul<Expansion>(first.n[0], second.n[1]) - mul<Expansion>(first.n[1], second.n[0]);
      });
      if (vertex.denominator == 0) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const double scale = std::max(std::fabs(c[0].value), std::fabs(c[1].value));
      double value[3] = {c[0].value, c[1].value, c[2].value};
      if (!(clip::accurate(c[2]) && c[0].bound <= kConstructionTolerance * scale &&
            c[1].bound <= kConstructionTolerance * scale)) {
        Expansion e[3];
        explicit_cofactors<Expansion>(first, second, pts, o, e);
        for (int k = 0; k < 3; ++k) {
          value[k] = e[k].estimate();
        }
      }
      for (int k = 0; k < 2; ++k) {
        vertex.x[k] = o[k] - value[k] / value[2];
      }
    }
    if (clamp_to_box) {
      for (int k = 0; k < 2; ++k) {
        vertex.x[k] = clip::clamp(vertex.x[k], lower[k], upper[k]);
      }
    }
    vertices.push_back(vertex);
    id = static_cast<int>(vertices.size()) - 1;
    return PHX_MC_OK;
  }

  std::vector<signed char> sides_;
  std::vector<int> next_loop_;
  std::vector<int> next_edge_lines_;
};

// Number of sign changes of the cyclic sequence of nonzero edge direction
// signs along `axis`; a convex polygon traversed once has at most two.
int direction_changes(const std::vector<double>& points, std::size_t count, int axis) {
  int first = 0;
  int last = 0;
  int changes = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double a = points[2 * i + axis];
    const double b = points[2 * ((i + 1) % count) + axis];
    const int sign = (b > a) - (b < a);
    if (sign == 0) {
      continue;
    }
    if (first == 0) {
      first = sign;
    } else if (sign != last) {
      ++changes;
    }
    last = sign;
  }
  if (first != 0 && last != first) {
    ++changes;
  }
  return changes;
}

struct PolygonWorkspace {
  std::vector<double> dedup;
  std::vector<int> turns;
  std::vector<double> first;
  std::vector<double> second;
  PolygonClipper clipper;
};

// Normalizes a convex polygon into a strictly convex counterclockwise vertex
// list (repeated consecutive and straight vertices removed).  Convexity: all
// turns of one sign or zero, no reversal at a zero turn, and at most two
// direction sign changes per axis (winding number one).
int32_t normalize_polygon(const double* vertices, int32_t count, PolygonWorkspace& work,
                          std::vector<double>& out) {
  std::vector<double>& p = work.dedup;
  p.clear();
  for (int32_t i = 0; i < count; ++i) {
    const double x = vertices[2 * i];
    const double y = vertices[2 * i + 1];
    if (!p.empty() && p[p.size() - 2] == x && p[p.size() - 1] == y) {
      continue;
    }
    p.push_back(x);
    p.push_back(y);
  }
  while (p.size() > 2 && p[0] == p[p.size() - 2] && p[1] == p[p.size() - 1]) {
    p.resize(p.size() - 2);
  }
  const std::size_t m = p.size() / 2;
  if (m < 3) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  work.turns.resize(m);
  bool positive = false;
  bool negative = false;
  for (std::size_t i = 0; i < m; ++i) {
    const double* prev = p.data() + 2 * ((i + m - 1) % m);
    const double* cur = p.data() + 2 * i;
    const double* next = p.data() + 2 * ((i + 1) % m);
    const int turn = orient2d(prev, cur, next);
    work.turns[i] = turn;
    positive = positive || turn > 0;
    negative = negative || turn < 0;
  }
  if (!positive && !negative) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  if (positive && negative) {
    return PHX_MC_INVALID_INPUT;
  }
  for (std::size_t i = 0; i < m; ++i) {
    if (work.turns[i] != 0) {
      continue;
    }
    const double* prev = p.data() + 2 * ((i + m - 1) % m);
    const double* cur = p.data() + 2 * i;
    const double* next = p.data() + 2 * ((i + 1) % m);
    const int dot = exact_sign([&]<class T>() {
      return diff<T>(cur[0], prev[0]) * diff<T>(next[0], cur[0]) +
             diff<T>(cur[1], prev[1]) * diff<T>(next[1], cur[1]);
    });
    if (dot < 0) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  if (direction_changes(p, m, 0) > 2 || direction_changes(p, m, 1) > 2) {
    return PHX_MC_INVALID_INPUT;
  }
  out.clear();
  for (std::size_t step = 0; step < m; ++step) {
    const std::size_t i = positive ? step : m - 1 - step;
    if (work.turns[i] != 0) {
      out.push_back(p[2 * i]);
      out.push_back(p[2 * i + 1]);
    }
  }
  return out.size() >= 6 ? PHX_MC_OK : PHX_MC_DEGENERATE_INPUT;
}

void add_point_line(PolygonClipper& clipper, int p, int q, int label) {
  Line line;
  line.p = p;
  line.q = q;
  line.label = label;
  clipper.lines.push_back(line);
}

// `simplices` (optional) receives the fan partition of a nonempty intersection.
int32_t intersect_polygons(const double* a, int32_t a_count, const double* b, int32_t b_count,
                           PolygonWorkspace& work, double& area, double* first_moment,
                           clip::SimplexOutput* simplices) {
  int32_t status = clip::check_finite(a, 2 * static_cast<int64_t>(a_count));
  if (status == PHX_MC_OK) {
    status = clip::check_finite(b, 2 * static_cast<int64_t>(b_count));
  }
  if (status == PHX_MC_OK) {
    status = clip::check_coordinates(a, 2 * static_cast<int64_t>(a_count));
  }
  if (status == PHX_MC_OK) {
    status = clip::check_coordinates(b, 2 * static_cast<int64_t>(b_count));
  }
  if (status == PHX_MC_OK) {
    status = normalize_polygon(a, a_count, work, work.first);
  }
  if (status == PHX_MC_OK) {
    status = normalize_polygon(b, b_count, work, work.second);
  }
  if (status != PHX_MC_OK) {
    return status;
  }
  PolygonClipper& clipper = work.clipper;
  clipper.reset();
  const int na = static_cast<int>(work.first.size() / 2);
  const int nb = static_cast<int>(work.second.size() / 2);
  clipper.points.insert(clipper.points.end(), work.first.begin(), work.first.end());
  clipper.points.insert(clipper.points.end(), work.second.begin(), work.second.end());
  for (int i = 0; i < na; ++i) {
    add_point_line(clipper, i, (i + 1) % na, i);
  }
  for (int j = 0; j < nb; ++j) {
    add_point_line(clipper, na + j, na + (j + 1) % nb, j);
  }
  for (int i = 0; i < na; ++i) {
    clipper.loop.push_back(clipper.add_point_vertex(i));
    clipper.edge_lines.push_back(i);
  }
  for (int j = 0; j < nb; ++j) {
    bool empty = false;
    status = clipper.clip(na + j, empty);
    if (status != PHX_MC_OK) {
      return status;
    }
    if (empty) {
      return PHX_MC_OK;
    }
  }
  clipper.moments(area, first_moment);
  return simplices == nullptr ? PHX_MC_OK : clipper.fan_simplices(*simplices);
}

int32_t clip_box(const double* lower, const double* upper, const double* normals,
                 const double* offsets, int32_t plane_count, int32_t vertex_capacity,
                 PolygonWorkspace& work, double* vertices, int32_t* edge_labels,
                 int32_t& vertex_count, double& area, double* first_moment) {
  int32_t status = clip::check_halfspaces(normals, offsets, plane_count, 2);
  if (status != PHX_MC_OK) {
    return status;
  }
  if (vertex_capacity < 4) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  PolygonClipper& clipper = work.clipper;
  clipper.reset();
  clipper.capacity = vertex_capacity;
  clipper.clamp_to_box = true;
  for (int k = 0; k < 2; ++k) {
    clipper.lower[k] = lower[k];
    clipper.upper[k] = upper[k];
  }
  // Counterclockwise corners (lower corner first) and the box sides between
  // them, labeled -(1 + 2 axis + side).
  const double corners[8] = {lower[0], lower[1], upper[0], lower[1],
                             upper[0], upper[1], lower[0], upper[1]};
  clipper.points.assign(corners, corners + 8);
  const int side_labels[4] = {-3, -2, -4, -1};
  for (int k = 0; k < 4; ++k) {
    add_point_line(clipper, k, (k + 1) % 4, side_labels[k]);
    clipper.loop.push_back(clipper.add_point_vertex(k));
    clipper.edge_lines.push_back(k);
  }
  for (int32_t plane = 0; plane < plane_count; ++plane) {
    Line line;
    line.is_explicit = true;
    line.n[0] = normals[2 * static_cast<int64_t>(plane)];
    line.n[1] = normals[2 * static_cast<int64_t>(plane) + 1];
    line.h = offsets[plane];
    line.label = plane;
    clipper.lines.push_back(line);
  }
  for (int32_t plane = 0; plane < plane_count; ++plane) {
    bool empty = false;
    status = clipper.clip(4 + plane, empty);
    if (status != PHX_MC_OK) {
      return status;
    }
    if (empty) {
      return PHX_MC_OK;
    }
  }
  const std::size_t n = clipper.loop.size();
  for (std::size_t k = 0; k < n; ++k) {
    const Vertex& vertex = clipper.vertices[clipper.loop[k]];
    vertices[2 * k] = vertex.x[0];
    vertices[2 * k + 1] = vertex.x[1];
    edge_labels[k] = clipper.lines[clipper.edge_lines[k]].label;
  }
  vertex_count = static_cast<int32_t>(n);
  clipper.moments(area, first_moment);
  return PHX_MC_OK;
}

// Batched polygon-pair intersections; `simplex_counts == nullptr` selects the
// moments-only variant.
int32_t polygon_batch(int64_t count, int32_t first_capacity, const double* first_vertices,
                      const int32_t* first_counts, int32_t second_capacity,
                      const double* second_vertices, const int32_t* second_counts,
                      int32_t simplex_capacity, double* simplices, int32_t* simplex_counts,
                      double* areas, double* first_moments, int32_t* item_status) {
  if (count < 0 || first_capacity < 0 || second_capacity < 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (count == 0) {
    return PHX_MC_OK;
  }
  if (first_counts == nullptr || second_counts == nullptr || areas == nullptr ||
      first_moments == nullptr || item_status == nullptr ||
      (first_capacity > 0 && first_vertices == nullptr) ||
      (second_capacity > 0 && second_vertices == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!clip::counts_in_range(first_counts, count, first_capacity) ||
      !clip::counts_in_range(second_counts, count, second_capacity)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  PolygonWorkspace work;
  for (int64_t item = 0; item < count; ++item) {
    double area = 0.0;
    double moment[2] = {0.0, 0.0};
    clip::SimplexOutput out{simplex_capacity,
                            simplices == nullptr ? nullptr : simplices + 6 * item * simplex_capacity,
                            0};
    int32_t status = intersect_polygons(
        first_vertices + 2 * item * first_capacity, first_counts[item],
        second_vertices + 2 * item * second_capacity, second_counts[item], work, area, moment,
        simplex_counts == nullptr ? nullptr : &out);
    if (status != PHX_MC_OK) {
      area = 0.0;
      moment[0] = moment[1] = 0.0;
      out.count = 0;
    }
    areas[item] = area;
    first_moments[2 * item] = moment[0];
    first_moments[2 * item + 1] = moment[1];
    if (simplex_counts != nullptr) {
      simplex_counts[item] = out.count;
    }
    item_status[item] = status;
  }
  return PHX_MC_OK;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_polygon_intersection_moments(int64_t count, int32_t first_capacity,
                                            const double* first_vertices,
                                            const int32_t* first_counts, int32_t second_capacity,
                                            const double* second_vertices,
                                            const int32_t* second_counts, double* areas,
                                            double* first_moments, int32_t* item_status) {
  try {
    return phx::mc::polygon_batch(count, first_capacity, first_vertices, first_counts,
                                  second_capacity, second_vertices, second_counts, 0, nullptr,
                                  nullptr, areas, first_moments, item_status);
  } catch (const std::bad_alloc&) {
    return PHX_MC_CAPACITY_EXCEEDED;
  } catch (...) {
    return PHX_MC_INTERNAL_ERROR;
  }
}

int32_t phx_mc_polygon_intersection_simplices(
    int64_t count, int32_t first_capacity, const double* first_vertices,
    const int32_t* first_counts, int32_t second_capacity, const double* second_vertices,
    const int32_t* second_counts, int32_t simplex_capacity, double* simplices,
    int32_t* simplex_counts, double* areas, double* first_moments, int32_t* item_status) {
  try {
    if (simplex_capacity < 1) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (simplices == nullptr || simplex_counts == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    return phx::mc::polygon_batch(count, first_capacity, first_vertices, first_counts,
                                  second_capacity, second_vertices, second_counts,
                                  simplex_capacity, simplices, simplex_counts, areas,
                                  first_moments, item_status);
  } catch (const std::bad_alloc&) {
    return PHX_MC_CAPACITY_EXCEEDED;
  } catch (...) {
    return PHX_MC_INTERNAL_ERROR;
  }
}

int32_t phx_mc_clip_box_halfplanes(int64_t count, const double* box_lower,
                                   const double* box_upper, int32_t plane_capacity,
                                   const double* normals, const double* offsets,
                                   const int32_t* plane_counts, int32_t vertex_capacity,
                                   double* vertices, int32_t* edge_labels,
                                   int32_t* vertex_counts, double* areas, double* first_moments,
                                   int32_t* item_status) {
  try {
    if (count < 0 || plane_capacity < 0 || vertex_capacity < 1) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int32_t box_status = phx::mc::clip::check_box(box_lower, box_upper, 2);
    if (box_status != PHX_MC_OK) {
      return box_status;
    }
    if (count == 0) {
      return PHX_MC_OK;
    }
    if (plane_counts == nullptr || vertices == nullptr || edge_labels == nullptr ||
        vertex_counts == nullptr || areas == nullptr || first_moments == nullptr ||
        item_status == nullptr ||
        (plane_capacity > 0 && (normals == nullptr || offsets == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!phx::mc::clip::counts_in_range(plane_counts, count, plane_capacity)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::PolygonWorkspace work;
    for (int64_t item = 0; item < count; ++item) {
      double area = 0.0;
      double moment[2] = {0.0, 0.0};
      int32_t vertex_count = 0;
      const int32_t status = phx::mc::clip_box(
          box_lower, box_upper, normals + 2 * item * plane_capacity,
          offsets + item * plane_capacity, plane_counts[item], vertex_capacity, work,
          vertices + 2 * item * vertex_capacity, edge_labels + item * vertex_capacity,
          vertex_count, area, moment);
      if (status != PHX_MC_OK) {
        vertex_count = 0;
        area = 0.0;
        moment[0] = moment[1] = 0.0;
      }
      vertex_counts[item] = vertex_count;
      areas[item] = area;
      first_moments[2 * item] = moment[0];
      first_moments[2 * item + 1] = moment[1];
      item_status[item] = status;
    }
    return PHX_MC_OK;
  } catch (const std::bad_alloc&) {
    return PHX_MC_CAPACITY_EXCEEDED;
  } catch (...) {
    return PHX_MC_INTERNAL_ERROR;
  }
}

}  // extern "C"
