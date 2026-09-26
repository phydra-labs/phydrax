//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact clipping of convex polyhedra by planes (face-list clipping after r3d
// with symbolic vertices): tetrahedron-tetrahedron and tetrahedron-halfspace
// intersection moments, and axis-aligned boxes clipped by halfspaces.
//
// A polytope is a list of faces, each a cyclic vertex loop (counterclockwise
// seen from outside) on one plane.  Planes are point-defined (a, b, c) with
// s(x) = orient3d(a, b, c, x) or explicit {x : n . x = h} with s(x) = h - n . x;
// the kept side is s >= 0.  Vertices are symbolic:
//   POINT(p)             input point;
//   EDGE_PLANE(u, v, P)  X = u + t (v - u), t = sP(u) / (sP(u) - sP(v)), for
//                        input points u, v;
//   PLANES3(F, H1, H2)   intersection of three planes, at most the first one
//                        point-defined, in coordinates y = x - o where o is an
//                        input point on F when F is point-defined.
// Exact sides with respect to a plane Q:
//   EDGE_PLANE: sQ(X) (sP(u) - sP(v)) = sP(u) sQ(v) - sP(v) sQ(u);
//   PLANES3 (Q explicit): rows (n_i, g_i) with g_i = s_i(o) (and (n_F, 0) for
//     point-defined F, n_F = (b - a) x (c - a)); the block determinant gives
//       det[[n1, g1], [n2, g2], [n3, g3], [nQ, gQ]] = det[n1; n2; n3] sQ(X),
//     expanded along the last row with cofactors (C0, C1, C2, C3 = det N):
//       C3 sQ(X) = C0 nQ0 + C1 nQ1 + C2 nQ2 + C3 sQ(o), and y_k = -C_k / C3.
// Degrees stay <= 6 in coordinates (see clip_common.hpp).
//
// A vertex created on an edge whose faces lie on planes f, g, cut by P:
//   f, g base planes          -> EDGE_PLANE(base edge endpoints, P);
//   base F, clip C:  tet-tet  -> EDGE_PLANE(B edge shared by C and P, F),
//                    explicit -> PLANES3(F, C, P);
//   clips C1, C2:    tet-tet  -> POINT(B vertex common to C1, C2, P),
//                    explicit -> PLANES3(C1, C2, P).
// Hence tet-tet only classifies POINT and EDGE_PLANE vertices and PLANES3
// vertices are only classified against explicit planes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "clip_common.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc {
namespace {

using clip::construction_value;
using clip::diff;
using clip::kConstructionTolerance;
using clip::kInvalidSide;
using clip::lift;
using clip::mul;

enum class Family { kTetTet, kTetHalfspaces, kBoxHalfspaces };

// tag: tetrahedron face -> opposite vertex (0..3) of its tetrahedron;
// box face -> 2 axis + side.  label: output face label.
struct Plane {
  bool is_explicit = false;
  int a = -1;
  int b = -1;
  int c = -1;
  double n[3] = {0.0, 0.0, 0.0};
  double h = 0.0;
  int tag = 0;
  int label = 0;
};

enum class Kind : std::uint8_t { kPoint, kEdgePlane, kPlanes3 };

// kPoint: a.  kEdgePlane: endpoints a, b and plane[0]; denominator =
// sign(s(a) - s(b)).  kPlanes3: origin point a and plane[0..2]; denominator =
// sign(C3); `cofactors`/`exact_cofactors` index four stored cofactors.
struct Vertex {
  Kind kind = Kind::kPoint;
  int a = -1;
  int b = -1;
  int plane[3] = {-1, -1, -1};
  int denominator = 0;
  int cofactors = -1;
  int exact_cofactors = -1;
  double x[3] = {0.0, 0.0, 0.0};
};

struct Face {
  int plane;
  int start;
  int size;
};

struct Cut {
  int lo;
  int hi;
  int plane;
  int vertex;
};

struct Segment {
  int from;
  int to;
};

// Positively oriented tetrahedron: the face opposite vertex k is spanned by
// kTetFace[k] with orient3d(face, v_k) > 0; its outside loop is (f0, f2, f1).
constexpr int kTetFace[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};

// Box corner index = x_bit + 2 y_bit + 4 z_bit; loop of face 2 axis + side,
// counterclockwise seen from outside.
constexpr int kBoxFace[6][4] = {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4},
                                {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};

constexpr int kMaxPoints = 8;

template <class T>
T orient_value(const double* a, const double* b, const double* c, const double* d) {
  if constexpr (std::is_same_v<T, Approx>) {
    return orient3d_approx(a, b, c, d);
  } else {
    return orient3d_exact(a, b, c, d);
  }
}

template <class T>
T det3(const T (&r)[3][4], int i, int j, int k) {
  return r[0][i] * (r[1][j] * r[2][k] - r[1][k] * r[2][j]) -
         r[0][j] * (r[1][i] * r[2][k] - r[1][k] * r[2][i]) +
         r[0][k] * (r[1][i] * r[2][j] - r[1][j] * r[2][i]);
}

// The two tetrahedron vertices outside {i, j}, ascending.
bool other_two(int i, int j, int& k, int& l) {
  if (i == j || i < 0 || j < 0 || i > 3 || j > 3) {
    return false;
  }
  int found[2] = {-1, -1};
  int count = 0;
  for (int v = 0; v < 4; ++v) {
    if (v != i && v != j) {
      found[count++] = v;
    }
  }
  k = found[0];
  l = found[1];
  return true;
}

class PolytopeClipper {
 public:
  Family family = Family::kTetTet;
  double points[kMaxPoints][3] = {};
  std::vector<Plane> planes;
  int base_planes = 0;
  int64_t capacity = std::numeric_limits<int64_t>::max();
  bool clamp_to_box = false;
  double lower[3] = {0.0, 0.0, 0.0};
  double upper[3] = {0.0, 0.0, 0.0};

  std::vector<Vertex> vertices;
  std::vector<int> live;
  std::vector<Face> faces;
  std::vector<int> loops;

  void reset(Family kind) {
    family = kind;
    planes.clear();
    base_planes = 0;
    capacity = std::numeric_limits<int64_t>::max();
    clamp_to_box = false;
    vertices.clear();
    live.clear();
    faces.clear();
    loops.clear();
    cofactors_.clear();
    exact_cofactors_.clear();
  }

  // Base polytope on points 0..3 (positively oriented).
  void begin_tetrahedron() {
    for (int k = 0; k < 4; ++k) {
      Plane plane;
      plane.a = kTetFace[k][0];
      plane.b = kTetFace[k][1];
      plane.c = kTetFace[k][2];
      plane.tag = k;
      plane.label = -1 - k;
      planes.push_back(plane);
    }
    base_planes = 4;
    for (int k = 0; k < 4; ++k) {
      add_point_vertex(k);
    }
    for (int k = 0; k < 4; ++k) {
      faces.push_back({k, static_cast<int>(loops.size()), 3});
      loops.push_back(kTetFace[k][0]);
      loops.push_back(kTetFace[k][2]);
      loops.push_back(kTetFace[k][1]);
    }
  }

  // Base polytope [lower, upper] on corner points 0..7.
  void begin_box(const double* box_lower, const double* box_upper) {
    clamp_to_box = true;
    for (int k = 0; k < 3; ++k) {
      lower[k] = box_lower[k];
      upper[k] = box_upper[k];
    }
    for (int corner = 0; corner < 8; ++corner) {
      for (int k = 0; k < 3; ++k) {
        points[corner][k] = ((corner >> k) & 1) != 0 ? box_upper[k] : box_lower[k];
      }
    }
    for (int id = 0; id < 6; ++id) {
      const int axis = id / 2;
      const int side = id % 2;
      Plane plane;
      plane.is_explicit = true;
      plane.n[axis] = side != 0 ? 1.0 : -1.0;
      plane.h = side != 0 ? box_upper[axis] : -box_lower[axis];
      plane.tag = id;
      plane.label = -(1 + id);
      planes.push_back(plane);
    }
    base_planes = 6;
    for (int corner = 0; corner < 8; ++corner) {
      add_point_vertex(corner);
    }
    for (int id = 0; id < 6; ++id) {
      faces.push_back({id, static_cast<int>(loops.size()), 4});
      loops.insert(loops.end(), kBoxFace[id], kBoxFace[id] + 4);
    }
  }

  void add_point_plane(int a, int b, int c, int tag) {
    Plane plane;
    plane.a = a;
    plane.b = b;
    plane.c = c;
    plane.tag = tag;
    plane.label = static_cast<int>(planes.size()) - base_planes;
    planes.push_back(plane);
  }

  void add_explicit_plane(const double* normal, double offset, int label) {
    Plane plane;
    plane.is_explicit = true;
    for (int k = 0; k < 3; ++k) {
      plane.n[k] = normal[k];
    }
    plane.h = offset;
    plane.label = label;
    planes.push_back(plane);
  }

  // Sizes the per-(plane, point) value caches once every plane is known and
  // enforces the vertex capacity on the base polytope.
  int32_t finish_setup(int64_t vertex_capacity) {
    capacity = vertex_capacity;
    const std::size_t size = planes.size() * kMaxPoints;
    approx_values_.resize(size);
    approx_ready_.assign(size, 0);
    if (exact_values_.size() < size) {
      exact_values_.resize(size);
    }
    exact_ready_.assign(size, 0);
    return static_cast<int64_t>(live.size()) > capacity ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_OK;
  }

  // Keeps the part with s_plane >= 0.  `empty` reports a zero-measure result.
  int32_t clip(int plane_id, bool& empty) {
    empty = false;
    if (side_.size() < vertices.size()) {
      side_.resize(vertices.size());
    }
    bool positive = false;
    bool negative = false;
    int64_t kept = 0;
    for (const int v : live) {
      const int side = classify(v, plane_id);
      if (side == kInvalidSide) {
        return PHX_MC_INTERNAL_ERROR;
      }
      side_[v] = static_cast<signed char>(side);
      positive = positive || side > 0;
      negative = negative || side < 0;
      kept += side >= 0;
    }
    if (!positive) {
      live.clear();
      faces.clear();
      loops.clear();
      empty = true;
      return PHX_MC_OK;
    }
    if (!negative) {
      return PHX_MC_OK;
    }

    // Every crossing edge appears once in each of its two faces.
    cuts_.clear();
    for (const Face& face : faces) {
      for (int i = 0; i < face.size; ++i) {
        const int a = loops[face.start + i];
        const int b = loops[face.start + (i + 1) % face.size];
        if (side_[a] * side_[b] < 0) {
          cuts_.push_back({std::min(a, b), std::max(a, b), face.plane, -1});
        }
      }
    }
    std::sort(cuts_.begin(), cuts_.end(), [](const Cut& x, const Cut& y) {
      if (x.lo != y.lo) {
        return x.lo < y.lo;
      }
      if (x.hi != y.hi) {
        return x.hi < y.hi;
      }
      return x.plane < y.plane;
    });
    if (cuts_.size() % 2 != 0) {
      return PHX_MC_INTERNAL_ERROR;
    }
    const int64_t created = static_cast<int64_t>(cuts_.size() / 2);
    if (kept + created > capacity) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    const int first_new = static_cast<int>(vertices.size());
    for (std::size_t k = 0; k < cuts_.size(); k += 2) {
      const Cut first = cuts_[k];
      const Cut second = cuts_[k + 1];
      if (first.lo != second.lo || first.hi != second.hi || first.plane == second.plane ||
          (k + 2 < cuts_.size() && cuts_[k + 2].lo == first.lo && cuts_[k + 2].hi == first.hi)) {
        return PHX_MC_INTERNAL_ERROR;
      }
      int id = -1;
      const int32_t status = make_vertex(first.plane, second.plane, plane_id, id);
      if (status != PHX_MC_OK) {
        return status;
      }
      cuts_[k].vertex = id;
      cuts_[k + 1].vertex = id;
    }

    // Sutherland-Hodgman on every face; the part of a clipped face loop that
    // left the kept side becomes the segment exit -> entry on the plane.
    next_faces_.clear();
    next_loops_.clear();
    segments_.clear();
    for (const Face& face : faces) {
      int nonnegative = 0;
      int first_kept = -1;
      for (int i = 0; i < face.size; ++i) {
        if (side_[loops[face.start + i]] >= 0) {
          if (first_kept < 0) {
            first_kept = i;
          }
          ++nonnegative;
        }
      }
      if (nonnegative == 0) {
        continue;
      }
      const int start = static_cast<int>(next_loops_.size());
      if (nonnegative == face.size) {
        next_loops_.insert(next_loops_.end(), loops.begin() + face.start,
                           loops.begin() + face.start + face.size);
        next_faces_.push_back({face.plane, start, face.size});
        continue;
      }
      int gap = -1;
      int gaps = 0;
      bool outside = false;
      for (int step = 0; step < face.size; ++step) {
        const int i = (first_kept + step) % face.size;
        const int cur = loops[face.start + i];
        const int nxt = loops[face.start + (i + 1) % face.size];
        const int sc = side_[cur];
        const int sn = side_[nxt];
        if (sc >= 0) {
          next_loops_.push_back(cur);
          outside = false;
        } else if (!outside) {
          outside = true;
          ++gaps;
          gap = static_cast<int>(next_loops_.size()) - 1 - start;
        }
        if (sc * sn < 0) {
          const int id = cut_vertex(cur, nxt);
          if (id < 0) {
            return PHX_MC_INTERNAL_ERROR;
          }
          next_loops_.push_back(id);
          outside = false;
        }
      }
      const int emitted = static_cast<int>(next_loops_.size()) - start;
      if (gaps != 1 || gap < 0) {
        return PHX_MC_INTERNAL_ERROR;
      }
      const int exit = next_loops_[start + gap];
      const int entry = next_loops_[start + (gap + 1) % emitted];
      if (exit != entry) {
        segments_.push_back({entry, exit});
      }
      if (emitted >= 3) {
        next_faces_.push_back({face.plane, start, emitted});
      } else {
        next_loops_.resize(start);
      }
    }

    // The reversed segments chain into the new face, counterclockwise from
    // outside because every edge is traversed opposite to its other face.
    if (segments_.size() < 3) {
      return PHX_MC_INTERNAL_ERROR;
    }
    std::sort(segments_.begin(), segments_.end(),
              [](const Segment& x, const Segment& y) { return x.from < y.from; });
    for (std::size_t k = 0; k + 1 < segments_.size(); ++k) {
      if (segments_[k].from == segments_[k + 1].from) {
        return PHX_MC_INTERNAL_ERROR;
      }
    }
    const int start = static_cast<int>(next_loops_.size());
    const int first_vertex = segments_[0].from;
    int current = first_vertex;
    for (std::size_t step = 0; step < segments_.size(); ++step) {
      const auto found = std::lower_bound(
          segments_.begin(), segments_.end(), current,
          [](const Segment& segment, int value) { return segment.from < value; });
      if (found == segments_.end() || found->from != current ||
          (step > 0 && current == first_vertex)) {
        return PHX_MC_INTERNAL_ERROR;
      }
      next_loops_.push_back(current);
      current = found->to;
    }
    if (current != first_vertex) {
      return PHX_MC_INTERNAL_ERROR;
    }
    next_faces_.push_back({plane_id, start, static_cast<int>(segments_.size())});

    next_live_.clear();
    for (const int v : live) {
      if (side_[v] >= 0) {
        next_live_.push_back(v);
      }
    }
    for (int v = first_new; v < static_cast<int>(vertices.size()); ++v) {
      next_live_.push_back(v);
    }
    // Every live vertex must be referenced by a face.
    stamp_.assign(vertices.size(), 0);
    std::size_t referenced = 0;
    for (const int v : next_loops_) {
      if (stamp_[v] == 0) {
        stamp_[v] = 1;
        ++referenced;
      }
    }
    if (referenced != next_live_.size()) {
      return PHX_MC_INTERNAL_ERROR;
    }
    live.swap(next_live_);
    faces.swap(next_faces_);
    loops.swap(next_loops_);
    return PHX_MC_OK;
  }

  // Vertex average o of the live vertices: inside the polytope, so every fan
  // term below is nonnegative up to the rounding of o.
  void vertex_average(double* o) const {
    for (int k = 0; k < 3; ++k) {
      o[k] = 0.0;
    }
    for (const int v : live) {
      for (int k = 0; k < 3; ++k) {
        o[k] += vertices[v].x[k];
      }
    }
    for (int k = 0; k < 3; ++k) {
      o[k] /= static_cast<double>(live.size());
    }
  }

  // Volume and first moment by fan tetrahedra from the vertex average o
  // (rounding errors scale with the polytope rather than with its distance to
  // the input points).
  void moments(double& volume, double* first_moment) const {
    double o[3];
    vertex_average(o);
    clip::CompensatedSum six;
    clip::CompensatedSum sum[3];
    for (const Face& face : faces) {
      const double* x0 = vertices[loops[face.start]].x;
      const double a[3] = {x0[0] - o[0], x0[1] - o[1], x0[2] - o[2]};
      for (int i = 1; i + 1 < face.size; ++i) {
        const double* x1 = vertices[loops[face.start + i]].x;
        const double* x2 = vertices[loops[face.start + i + 1]].x;
        const double b[3] = {x1[0] - o[0], x1[1] - o[1], x1[2] - o[2]};
        const double c[3] = {x2[0] - o[0], x2[1] - o[1], x2[2] - o[2]};
        const double cross[3] = {clip::difference_of_products(b[1], c[2], b[2], c[1]),
                                 clip::difference_of_products(b[2], c[0], b[0], c[2]),
                                 clip::difference_of_products(b[0], c[1], b[1], c[0])};
        const double det = clip::dot3(a, cross);
        six.add(det);
        for (int k = 0; k < 3; ++k) {
          sum[k].add(det * (a[k] + b[k] + c[k]));
        }
      }
    }
    volume = six.value() / 6.0;
    for (int k = 0; k < 3; ++k) {
      first_moment[k] = sum[k].value() / 24.0 + o[k] * volume;
    }
  }

  // The fan tetrahedra (o, x0, x_i, x_{i+1}) of moments(): a signed partition
  // of the polytope whose terms are exactly the fan terms of moments().
  int32_t cone_simplices(clip::SimplexOutput& out) const {
    int64_t needed = 0;
    for (const Face& face : faces) {
      needed += face.size - 2;
    }
    if (needed > out.capacity) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    double o[3];
    vertex_average(o);
    int32_t written = 0;
    for (const Face& face : faces) {
      const double* x0 = vertices[loops[face.start]].x;
      for (int i = 1; i + 1 < face.size; ++i) {
        const double* corners[4] = {o, x0, vertices[loops[face.start + i]].x,
                                    vertices[loops[face.start + i + 1]].x};
        double* row = out.simplices + 12 * static_cast<int64_t>(written);
        for (int corner = 0; corner < 4; ++corner) {
          for (int k = 0; k < 3; ++k) {
            row[3 * corner + k] = corners[corner][k];
          }
        }
        ++written;
      }
    }
    out.count = written;
    return PHX_MC_OK;
  }

 private:
  void add_point_vertex(int point) {
    Vertex vertex;
    vertex.a = point;
    for (int k = 0; k < 3; ++k) {
      vertex.x[k] = points[point][k];
    }
    vertices.push_back(vertex);
    live.push_back(static_cast<int>(vertices.size()) - 1);
  }

  template <class T>
  T plane_value(int plane_id, int point) const {
    const Plane& plane = planes[plane_id];
    const double* x = points[point];
    if (plane.is_explicit) {
      return lift<T>(plane.h) -
             (mul<T>(plane.n[0], x[0]) + mul<T>(plane.n[1], x[1]) + mul<T>(plane.n[2], x[2]));
    }
    if (point == plane.a || point == plane.b || point == plane.c) {
      return T();
    }
    return orient_value<T>(points[plane.a], points[plane.b], points[plane.c], x);
  }

  const Approx& approx_value(int plane_id, int point) {
    const std::size_t index = static_cast<std::size_t>(plane_id) * kMaxPoints + point;
    if (approx_ready_[index] == 0) {
      approx_values_[index] = plane_value<Approx>(plane_id, point);
      approx_ready_[index] = 1;
    }
    return approx_values_[index];
  }

  const Expansion& exact_value(int plane_id, int point) {
    const std::size_t index = static_cast<std::size_t>(plane_id) * kMaxPoints + point;
    if (exact_ready_[index] == 0) {
      exact_values_[index] = plane_value<Expansion>(plane_id, point);
      exact_ready_[index] = 1;
    }
    return exact_values_[index];
  }

  template <class T>
  void planes3_cofactors(const Vertex& vertex, T* c) const {
    T r[3][4];
    for (int i = 0; i < 3; ++i) {
      const Plane& plane = planes[vertex.plane[i]];
      if (plane.is_explicit) {
        for (int k = 0; k < 3; ++k) {
          r[i][k] = lift<T>(plane.n[k]);
        }
        r[i][3] = plane_value<T>(vertex.plane[i], vertex.a);
      } else {
        // The origin is plane.a, so the row is (n_F, 0).
        const double* pa = points[plane.a];
        const double* pb = points[plane.b];
        const double* pc = points[plane.c];
        const T e1[3] = {diff<T>(pb[0], pa[0]), diff<T>(pb[1], pa[1]), diff<T>(pb[2], pa[2])};
        const T e2[3] = {diff<T>(pc[0], pa[0]), diff<T>(pc[1], pa[1]), diff<T>(pc[2], pa[2])};
        r[i][0] = e1[1] * e2[2] - e1[2] * e2[1];
        r[i][1] = e1[2] * e2[0] - e1[0] * e2[2];
        r[i][2] = e1[0] * e2[1] - e1[1] * e2[0];
        r[i][3] = T();
      }
    }
    c[0] = -det3(r, 1, 2, 3);
    c[1] = det3(r, 0, 2, 3);
    c[2] = -det3(r, 0, 1, 3);
    c[3] = det3(r, 0, 1, 2);
  }

  int store_exact_cofactors(const Vertex& vertex) {
    Expansion e[4];
    planes3_cofactors<Expansion>(vertex, e);
    const int index = static_cast<int>(exact_cofactors_.size());
    for (Expansion& value : e) {
      exact_cofactors_.push_back(std::move(value));
    }
    return index;
  }

  int classify(int vertex_id, int plane_id) {
    Vertex& vertex = vertices[vertex_id];
    switch (vertex.kind) {
      case Kind::kPoint:
        return clip::exact_sign(approx_value(plane_id, vertex.a),
                                [&] { return exact_value(plane_id, vertex.a); });
      case Kind::kEdgePlane: {
        const int p = vertex.plane[0];
        const int u = vertex.a;
        const int w = vertex.b;
        const Approx filtered = approx_value(p, u) * approx_value(plane_id, w) -
                                approx_value(p, w) * approx_value(plane_id, u);
        return vertex.denominator * clip::exact_sign(filtered, [&] {
                 return exact_value(p, u) * exact_value(plane_id, w) -
                        exact_value(p, w) * exact_value(plane_id, u);
               });
      }
      case Kind::kPlanes3: {
        const Plane& plane = planes[plane_id];
        if (!plane.is_explicit) {
          return kInvalidSide;
        }
        const Approx* c = cofactors_.data() + vertex.cofactors;
        const Approx filtered = c[0] * Approx::exact(plane.n[0]) +
                                c[1] * Approx::exact(plane.n[1]) +
                                c[2] * Approx::exact(plane.n[2]) +
                                c[3] * approx_value(plane_id, vertex.a);
        return vertex.denominator * clip::exact_sign(filtered, [&] {
                 if (vertex.exact_cofactors < 0) {
                   vertex.exact_cofactors = store_exact_cofactors(vertex);
                 }
                 const Expansion* e = exact_cofactors_.data() + vertex.exact_cofactors;
                 return e[0].scaled(plane.n[0]) + e[1].scaled(plane.n[1]) +
                        e[2].scaled(plane.n[2]) + e[3] * exact_value(plane_id, vertex.a);
               });
      }
    }
    return kInvalidSide;
  }

  int cut_vertex(int a, int b) const {
    const int lo = std::min(a, b);
    const int hi = std::max(a, b);
    const auto found = std::lower_bound(cuts_.begin(), cuts_.end(), std::make_pair(lo, hi),
                                        [](const Cut& cut, const std::pair<int, int>& key) {
                                          return cut.lo != key.first ? cut.lo < key.first
                                                                     : cut.hi < key.second;
                                        });
    if (found == cuts_.end() || found->lo != lo || found->hi != hi) {
      return -1;
    }
    return found->vertex;
  }

  // New vertex on the edge between faces on planes f and g, crossed strictly
  // by plane p.
  int32_t make_vertex(int f, int g, int p, int& id) {
    if (f > g) {
      std::swap(f, g);
    }
    Vertex vertex;
    const bool f_base = f < base_planes;
    const bool g_base = g < base_planes;
    if (f_base && g_base) {
      int u = -1;
      int w = -1;
      if (family == Family::kBoxHalfspaces) {
        const int fa = planes[f].tag / 2;
        const int fs = planes[f].tag % 2;
        const int ga = planes[g].tag / 2;
        const int gs = planes[g].tag % 2;
        if (fa == ga) {
          return PHX_MC_INTERNAL_ERROR;
        }
        u = (fs << fa) | (gs << ga);
        w = u | (1 << (3 - fa - ga));
      } else if (!other_two(planes[f].tag, planes[g].tag, u, w)) {
        return PHX_MC_INTERNAL_ERROR;
      }
      vertex.kind = Kind::kEdgePlane;
      vertex.a = u;
      vertex.b = w;
      vertex.plane[0] = p;
    } else if (family == Family::kTetTet) {
      if (f_base) {
        int k = -1;
        int l = -1;
        if (!other_two(planes[g].tag, planes[p].tag, k, l)) {
          return PHX_MC_INTERNAL_ERROR;
        }
        vertex.kind = Kind::kEdgePlane;
        vertex.a = 4 + k;
        vertex.b = 4 + l;
        vertex.plane[0] = f;
      } else {
        const int i = planes[f].tag;
        const int j = planes[g].tag;
        const int k = planes[p].tag;
        if (i == j || i == k || j == k) {
          return PHX_MC_INTERNAL_ERROR;
        }
        vertex.kind = Kind::kPoint;
        vertex.a = 4 + (6 - i - j - k);
      }
    } else {
      vertex.kind = Kind::kPlanes3;
      vertex.plane[0] = f;
      vertex.plane[1] = g;
      vertex.plane[2] = p;
      vertex.a = planes[f].is_explicit ? 0 : planes[f].a;
    }

    switch (vertex.kind) {
      case Kind::kPoint:
        for (int k = 0; k < 3; ++k) {
          vertex.x[k] = points[vertex.a][k];
        }
        break;
      case Kind::kEdgePlane: {
        const int q = vertex.plane[0];
        const int u = vertex.a;
        const int w = vertex.b;
        vertex.denominator = clip::exact_sign(approx_value(q, u) - approx_value(q, w), [&] {
          return exact_value(q, u) - exact_value(q, w);
        });
        if (vertex.denominator == 0) {
          return PHX_MC_INTERNAL_ERROR;
        }
        // X lies on the closed segment [u, w] (an edge of a base polytope),
        // so both parameters are clamped to [0, 1].
        const double su =
            construction_value(approx_value(q, u), [&] { return exact_value(q, u); });
        const double sw =
            construction_value(approx_value(q, w), [&] { return exact_value(q, w); });
        const double denominator = su - sw;
        const double t = clip::clamp(su / denominator, 0.0, 1.0);
        const double r = clip::clamp(-sw / denominator, 0.0, 1.0);
        const double* pu = points[u];
        const double* pw = points[w];
        for (int k = 0; k < 3; ++k) {
          vertex.x[k] = t <= 0.5 ? pu[k] + t * (pw[k] - pu[k]) : pw[k] + r * (pu[k] - pw[k]);
        }
        break;
      }
      case Kind::kPlanes3: {
        Approx c[4];
        planes3_cofactors<Approx>(vertex, c);
        vertex.cofactors = static_cast<int>(cofactors_.size());
        cofactors_.insert(cofactors_.end(), c, c + 4);
        double value[4] = {c[0].value, c[1].value, c[2].value, c[3].value};
        const double scale =
            std::max({std::fabs(c[0].value), std::fabs(c[1].value), std::fabs(c[2].value)});
        const bool accurate = clip::accurate(c[3]) &&
                              c[0].bound <= kConstructionTolerance * scale &&
                              c[1].bound <= kConstructionTolerance * scale &&
                              c[2].bound <= kConstructionTolerance * scale;
        vertex.denominator = c[3].certified_sign();
        if (vertex.denominator == 2 || !accurate) {
          vertex.exact_cofactors = store_exact_cofactors(vertex);
          const Expansion* e = exact_cofactors_.data() + vertex.exact_cofactors;
          vertex.denominator = e[3].sign();
          for (int k = 0; k < 4; ++k) {
            value[k] = e[k].estimate();
          }
        }
        if (vertex.denominator == 0) {
          return PHX_MC_INTERNAL_ERROR;
        }
        const double* o = points[vertex.a];
        for (int k = 0; k < 3; ++k) {
          vertex.x[k] = o[k] - value[k] / value[3];
        }
        break;
      }
    }
    if (clamp_to_box) {
      for (int k = 0; k < 3; ++k) {
        vertex.x[k] = clip::clamp(vertex.x[k], lower[k], upper[k]);
      }
    }
    vertices.push_back(vertex);
    id = static_cast<int>(vertices.size()) - 1;
    return PHX_MC_OK;
  }

  std::vector<Approx> approx_values_;
  std::vector<unsigned char> approx_ready_;
  std::vector<Expansion> exact_values_;
  std::vector<unsigned char> exact_ready_;
  std::vector<Approx> cofactors_;
  std::vector<Expansion> exact_cofactors_;
  std::vector<signed char> side_;
  std::vector<Cut> cuts_;
  std::vector<Segment> segments_;
  std::vector<Face> next_faces_;
  std::vector<int> next_loops_;
  std::vector<int> next_live_;
  std::vector<unsigned char> stamp_;
};

// Loads a tetrahedron (4, 3) into points[offset..offset + 3], positively
// oriented (the first two vertices swapped when orient3d < 0).
bool load_tetrahedron(const double* tetrahedron, int offset, PolytopeClipper& clipper) {
  const int orientation =
      orient3d(tetrahedron, tetrahedron + 3, tetrahedron + 6, tetrahedron + 9);
  if (orientation == 0) {
    return false;
  }
  const int order[4] = {orientation > 0 ? 0 : 1, orientation > 0 ? 1 : 0, 2, 3};
  for (int k = 0; k < 4; ++k) {
    for (int axis = 0; axis < 3; ++axis) {
      clipper.points[offset + k][axis] = tetrahedron[3 * order[k] + axis];
    }
  }
  return true;
}

int32_t run_clips(PolytopeClipper& clipper, bool& empty) {
  empty = false;
  for (int plane = clipper.base_planes; plane < static_cast<int>(clipper.planes.size()); ++plane) {
    const int32_t status = clipper.clip(plane, empty);
    if (status != PHX_MC_OK || empty) {
      return status;
    }
  }
  return PHX_MC_OK;
}

// `simplices` (optional) receives the cone partition of a nonempty intersection.
int32_t intersect_tetrahedra(const double* first, const double* second, int32_t vertex_capacity,
                             PolytopeClipper& clipper, double& volume, double* first_moment,
                             clip::SimplexOutput* simplices) {
  int32_t status = clip::check_finite(first, 12);
  if (status == PHX_MC_OK) {
    status = clip::check_finite(second, 12);
  }
  if (status == PHX_MC_OK) {
    status = clip::check_coordinates(first, 12);
  }
  if (status == PHX_MC_OK) {
    status = clip::check_coordinates(second, 12);
  }
  if (status != PHX_MC_OK) {
    return status;
  }
  clipper.reset(Family::kTetTet);
  if (!load_tetrahedron(first, 0, clipper) || !load_tetrahedron(second, 4, clipper)) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  clipper.begin_tetrahedron();
  for (int k = 0; k < 4; ++k) {
    clipper.add_point_plane(4 + kTetFace[k][0], 4 + kTetFace[k][1], 4 + kTetFace[k][2], k);
  }
  status = clipper.finish_setup(vertex_capacity);
  bool empty = false;
  if (status == PHX_MC_OK) {
    status = run_clips(clipper, empty);
  }
  if (status == PHX_MC_OK && !empty) {
    clipper.moments(volume, first_moment);
    if (simplices != nullptr) {
      status = clipper.cone_simplices(*simplices);
    }
  }
  return status;
}

int32_t clip_tetrahedron(const double* normals, const double* offsets, int32_t plane_count,
                         const double* tetrahedron, int32_t vertex_capacity,
                         PolytopeClipper& clipper, double& volume, double* first_moment) {
  const int64_t components = 3 * static_cast<int64_t>(plane_count);
  int32_t status = clip::check_finite(tetrahedron, 12);
  if (status == PHX_MC_OK) {
    status = clip::check_finite(normals, components);
  }
  if (status == PHX_MC_OK) {
    status = clip::check_finite(offsets, plane_count);
  }
  if (status == PHX_MC_OK) {
    status = clip::check_coordinates(tetrahedron, 12);
  }
  if (status != PHX_MC_OK) {
    return status;
  }
  clipper.reset(Family::kTetHalfspaces);
  if (!load_tetrahedron(tetrahedron, 0, clipper)) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  status = clip::check_halfspaces(normals, offsets, plane_count, 3);
  if (status != PHX_MC_OK) {
    return status;
  }
  clipper.begin_tetrahedron();
  for (int32_t plane = 0; plane < plane_count; ++plane) {
    clipper.add_explicit_plane(normals + 3 * static_cast<int64_t>(plane), offsets[plane], plane);
  }
  status = clipper.finish_setup(vertex_capacity);
  bool empty = false;
  if (status == PHX_MC_OK) {
    status = run_clips(clipper, empty);
  }
  if (status == PHX_MC_OK && !empty) {
    clipper.moments(volume, first_moment);
  }
  return status;
}

struct BoxOutput {
  int32_t vertex_capacity;
  int32_t face_capacity;
  int32_t face_vertex_capacity;
  double* vertices;
  int32_t* face_offsets;
  int32_t* face_labels;
  int32_t* face_vertices;
  int32_t vertex_count;
  int32_t face_count;
};

struct BoxWorkspace {
  PolytopeClipper clipper;
  std::vector<int> sorted_live;
  std::vector<int> index;
  std::vector<int> order;
};

// Compacts the live vertices (ascending ids), sorts faces by label and
// rotates each loop to start at its smallest compacted vertex index.
int32_t write_box_output(BoxWorkspace& work, BoxOutput& out) {
  const PolytopeClipper& clipper = work.clipper;
  if (static_cast<int64_t>(clipper.live.size()) > out.vertex_capacity ||
      static_cast<int64_t>(clipper.faces.size()) > out.face_capacity ||
      static_cast<int64_t>(clipper.loops.size()) > out.face_vertex_capacity) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  work.index.assign(clipper.vertices.size(), -1);
  std::vector<int>& sorted_live = work.sorted_live;
  sorted_live.assign(clipper.live.begin(), clipper.live.end());
  std::sort(sorted_live.begin(), sorted_live.end());
  for (std::size_t k = 0; k < sorted_live.size(); ++k) {
    const int v = sorted_live[k];
    work.index[v] = static_cast<int>(k);
    for (int axis = 0; axis < 3; ++axis) {
      out.vertices[3 * k + axis] = clipper.vertices[v].x[axis];
    }
  }
  work.order.resize(clipper.faces.size());
  for (std::size_t f = 0; f < clipper.faces.size(); ++f) {
    work.order[f] = static_cast<int>(f);
  }
  std::sort(work.order.begin(), work.order.end(), [&](int x, int y) {
    return clipper.planes[clipper.faces[x].plane].label <
           clipper.planes[clipper.faces[y].plane].label;
  });
  int32_t offset = 0;
  out.face_offsets[0] = 0;
  for (std::size_t k = 0; k < work.order.size(); ++k) {
    const Face& face = clipper.faces[work.order[k]];
    int smallest = 0;
    for (int i = 1; i < face.size; ++i) {
      if (work.index[clipper.loops[face.start + i]] <
          work.index[clipper.loops[face.start + smallest]]) {
        smallest = i;
      }
    }
    for (int i = 0; i < face.size; ++i) {
      out.face_vertices[offset + i] =
          work.index[clipper.loops[face.start + (smallest + i) % face.size]];
    }
    offset += face.size;
    out.face_labels[k] = clipper.planes[face.plane].label;
    out.face_offsets[k + 1] = offset;
  }
  out.vertex_count = static_cast<int32_t>(sorted_live.size());
  out.face_count = static_cast<int32_t>(work.order.size());
  return PHX_MC_OK;
}

int32_t clip_box(const double* lower, const double* upper, const double* normals,
                 const double* offsets, int32_t plane_count, BoxWorkspace& work, BoxOutput& out,
                 double& volume, double* first_moment) {
  int32_t status = clip::check_halfspaces(normals, offsets, plane_count, 3);
  if (status != PHX_MC_OK) {
    return status;
  }
  PolytopeClipper& clipper = work.clipper;
  clipper.reset(Family::kBoxHalfspaces);
  clipper.begin_box(lower, upper);
  for (int32_t plane = 0; plane < plane_count; ++plane) {
    clipper.add_explicit_plane(normals + 3 * static_cast<int64_t>(plane), offsets[plane], plane);
  }
  status = clipper.finish_setup(out.vertex_capacity);
  bool empty = false;
  if (status == PHX_MC_OK) {
    status = run_clips(clipper, empty);
  }
  if (status != PHX_MC_OK || empty) {
    return status;
  }
  status = write_box_output(work, out);
  if (status == PHX_MC_OK) {
    clipper.moments(volume, first_moment);
  }
  return status;
}

// Batched tetrahedron-pair intersections; `simplex_counts == nullptr` selects
// the moments-only variant.
int32_t tetrahedron_batch(int64_t count, const double* first, const double* second,
                          int32_t vertex_capacity, int32_t simplex_capacity, double* simplices,
                          int32_t* simplex_counts, double* volumes, double* first_moments,
                          int32_t* item_status) {
  if (count < 0 || vertex_capacity < 1 || simplex_capacity < 0 ||
      !addressable(count, 12, sizeof(double)) ||
      !addressable(count, 12 * static_cast<int64_t>(simplex_capacity), sizeof(double))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (count == 0) {
    return PHX_MC_OK;
  }
  if (first == nullptr || second == nullptr || volumes == nullptr || first_moments == nullptr ||
      item_status == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  PolytopeClipper clipper;
  for (int64_t item = 0; item < count; ++item) {
    double volume = 0.0;
    double moment[3] = {0.0, 0.0, 0.0};
    clip::SimplexOutput out{
        simplex_capacity,
        simplices == nullptr ? nullptr : simplices + 12 * item * simplex_capacity, 0};
    const int32_t status =
        intersect_tetrahedra(first + 12 * item, second + 12 * item, vertex_capacity, clipper,
                             volume, moment, simplex_counts == nullptr ? nullptr : &out);
    if (status != PHX_MC_OK) {
      volume = 0.0;
      moment[0] = moment[1] = moment[2] = 0.0;
      out.count = 0;
    }
    volumes[item] = volume;
    for (int k = 0; k < 3; ++k) {
      first_moments[3 * item + k] = moment[k];
    }
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

int32_t phx_mc_tetrahedron_intersection_moments(int64_t count, const double* first,
                                                const double* second, int32_t vertex_capacity,
                                                double* volumes, double* first_moments,
                                                int32_t* item_status) {
  return phx::mc::guarded([&] {
    return phx::mc::tetrahedron_batch(count, first, second, vertex_capacity, 0, nullptr,
                                      nullptr, volumes, first_moments, item_status);
  });
}

int32_t phx_mc_tetrahedron_intersection_simplices(int64_t count, const double* first,
                                                  const double* second, int32_t vertex_capacity,
                                                  int32_t simplex_capacity, double* simplices,
                                                  int32_t* simplex_counts, double* volumes,
                                                  double* first_moments, int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    if (simplex_capacity < 1) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count > 0 && (simplices == nullptr || simplex_counts == nullptr)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    return phx::mc::tetrahedron_batch(count, first, second, vertex_capacity, simplex_capacity,
                                      simplices, simplex_counts, volumes, first_moments,
                                      item_status);
  });
}

int32_t phx_mc_polyhedron_clip_moments(int64_t count, int32_t plane_capacity,
                                       const double* normals, const double* offsets,
                                       const int32_t* plane_counts, const double* tetrahedra,
                                       int32_t vertex_capacity, double* volumes,
                                       double* first_moments, int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || plane_capacity < 0 || vertex_capacity < 1 ||
        !phx::mc::addressable(count, 3 * static_cast<int64_t>(plane_capacity), sizeof(double)) ||
        !phx::mc::addressable(count, 12, sizeof(double))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (count == 0) {
      return PHX_MC_OK;
    }
    if (plane_counts == nullptr || tetrahedra == nullptr || volumes == nullptr ||
        first_moments == nullptr || item_status == nullptr ||
        (plane_capacity > 0 && (normals == nullptr || offsets == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!phx::mc::clip::counts_in_range(plane_counts, count, plane_capacity)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::PolytopeClipper clipper;
    for (int64_t item = 0; item < count; ++item) {
      double volume = 0.0;
      double moment[3] = {0.0, 0.0, 0.0};
      const int32_t status = phx::mc::clip_tetrahedron(
          normals + 3 * item * plane_capacity, offsets + item * plane_capacity,
          plane_counts[item], tetrahedra + 12 * item, vertex_capacity, clipper, volume, moment);
      if (status != PHX_MC_OK) {
        volume = 0.0;
        moment[0] = moment[1] = moment[2] = 0.0;
      }
      volumes[item] = volume;
      for (int k = 0; k < 3; ++k) {
        first_moments[3 * item + k] = moment[k];
      }
      item_status[item] = status;
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_clip_box_halfspaces(int64_t count, const double* box_lower,
                                   const double* box_upper, int32_t plane_capacity,
                                   const double* normals, const double* offsets,
                                   const int32_t* plane_counts, int32_t vertex_capacity,
                                   int32_t face_capacity, int32_t face_vertex_capacity,
                                   double* vertices, int32_t* vertex_counts,
                                   int32_t* face_offsets, int32_t* face_labels,
                                   int32_t* face_vertices, int32_t* face_counts, double* volumes,
                                   double* first_moments, int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || plane_capacity < 0 || vertex_capacity < 1 || face_capacity < 1 ||
        face_vertex_capacity < 1 ||
        !phx::mc::addressable(count, 3 * static_cast<int64_t>(plane_capacity), sizeof(double)) ||
        !phx::mc::addressable(count, 3 * static_cast<int64_t>(vertex_capacity), sizeof(double)) ||
        !phx::mc::addressable(count, static_cast<int64_t>(face_capacity) + 1, sizeof(int32_t)) ||
        !phx::mc::addressable(count, face_vertex_capacity, sizeof(int32_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int32_t box_status = phx::mc::clip::check_box(box_lower, box_upper, 3);
    if (box_status != PHX_MC_OK) {
      return box_status;
    }
    if (count == 0) {
      return PHX_MC_OK;
    }
    if (plane_counts == nullptr || vertices == nullptr || vertex_counts == nullptr ||
        face_offsets == nullptr || face_labels == nullptr || face_vertices == nullptr ||
        face_counts == nullptr || volumes == nullptr || first_moments == nullptr ||
        item_status == nullptr ||
        (plane_capacity > 0 && (normals == nullptr || offsets == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    if (!phx::mc::clip::counts_in_range(plane_counts, count, plane_capacity)) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    phx::mc::BoxWorkspace work;
    for (int64_t item = 0; item < count; ++item) {
      phx::mc::BoxOutput out{vertex_capacity,
                             face_capacity,
                             face_vertex_capacity,
                             vertices + 3 * item * vertex_capacity,
                             face_offsets + item * (static_cast<int64_t>(face_capacity) + 1),
                             face_labels + item * face_capacity,
                             face_vertices + item * face_vertex_capacity,
                             0,
                             0};
      out.face_offsets[0] = 0;
      double volume = 0.0;
      double moment[3] = {0.0, 0.0, 0.0};
      const int32_t status =
          phx::mc::clip_box(box_lower, box_upper, normals + 3 * item * plane_capacity,
                            offsets + item * plane_capacity, plane_counts[item], work, out,
                            volume, moment);
      if (status != PHX_MC_OK) {
        out.vertex_count = 0;
        out.face_count = 0;
        out.face_offsets[0] = 0;
        volume = 0.0;
        moment[0] = moment[1] = moment[2] = 0.0;
      }
      vertex_counts[item] = out.vertex_count;
      face_counts[item] = out.face_count;
      volumes[item] = volume;
      for (int k = 0; k < 3; ++k) {
        first_moments[3 * item + k] = moment[k];
      }
      item_status[item] = status;
    }
    return PHX_MC_OK;
  });
}

}  // extern "C"
