//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact triangle/triangle, segment/triangle and point/triangle contact
// classification (see intersections.hpp).
//
// Transversal case.  With s_i = orient3d(B, a_i) and r_j = orient3d(A, b_j),
// A meets plane(B) in a point or segment I_A and B meets plane(A) in I_B, both
// on the line L = plane(A) n plane(B), directed by d = N_A x N_B where
// N_T = (t1 - t0) x (t2 - t0).  Every endpoint of I_A is X = (u, w) n plane(B)
// for vertices u, w of A with w strictly off plane(B) and u on the closed
// opposite side (X = u when s_u = 0); likewise Y = (u', w') n plane(A).  Then
//   sign(d . (Y - X)) = orient3d(u, w, u', w') * s_w * r_w'
// (both sides are continuous in the configuration, vanish exactly when the
// lines uw and u'w' meet on L, and agree on one instance), and when A has a
// lone vertex l strictly on one side with both others on the closed opposite
// side, I_A runs from X1 = (l + 1, l) to X2 = (l + 2, l) with
//   sign(d . (X2 - X1)) = -s_l,
// and symmetrically sign(d . (Y2 - Y1)) = +r_l for B (d reverses with the
// plane roles).  Each comparison is one orient3d on input points.
//
// Coplanar case.  Both triangles are projected along an axis whose normal
// component is exactly nonzero (an affine bijection of their plane, so every
// orient2d sign is exact) and the first input is clipped by the three closed
// halfplanes of the second (Sutherland-Hodgman with symbolic vertices: input
// vertices, or the crossing of a subject edge line with a clip line, whose
// side with respect to another clip line is a degree-four expansion sign).
// Clipping a strictly convex polygon by a closed halfplane yields a strictly
// convex polygon, a segment, a point or nothing, so no duplicate or straight
// vertices arise.
#include "intersections.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "capi_guard.hpp"
#include "clip_common.hpp"
#include "expansion.hpp"
#include "filtered.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc {
namespace {

constexpr double kUnitRoundoff = 0x1p-53;

bool same_point(const double* a, const double* b) {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

int8_t edge_feature(int k) { return static_cast<int8_t>(3 + k); }

bool is_vertex_feature(int8_t feature) { return feature >= 0 && feature < 3; }

// Feature of a point inside a closed triangle from its sides with respect to
// the three edge lines (edge k opposite vertex k).
int8_t feature_from_sides(const int sides[3]) {
  int zeros = 0;
  int zero = -1;
  int nonzero = -1;
  for (int k = 0; k < 3; ++k) {
    if (sides[k] == 0) {
      ++zeros;
      zero = k;
    } else {
      nonzero = k;
    }
  }
  switch (zeros) {
    case 0:
      return kFeatureInterior;
    case 1:
      return edge_feature(zero);
    default:
      return static_cast<int8_t>(nonzero);
  }
}

// Estimate of an exact expansion with a rigorous absolute error bound:
// recursive summation of n components errs by at most (n - 1) u sum |e_i| /
// (1 - (n - 1) u), dominated by 2 n u times the rounded magnitude sum.
Approx bounded_estimate(const Expansion& value) {
  const double* components = value.data();
  double sum = 0.0;
  double magnitude = 0.0;
  for (int k = 0; k < value.size(); ++k) {
    sum += components[k];
    magnitude += std::fabs(components[k]);
  }
  const double bound = value.size() <= 1 ? 0.0 : 2.0 * value.size() * kUnitRoundoff * magnitude;
  return {sum, bound};
}

// t = numerator / denominator with exact t in [0, 1].
Approx parameter(const Expansion& numerator, const Expansion& denominator) {
  const Approx n = bounded_estimate(numerator);
  const Approx d = bounded_estimate(denominator);
  const double lower = std::fabs(d.value) - d.bound;
  double t = n.value / d.value;
  if (!(lower > 0.0) || !std::isfinite(t)) {
    return {0.5, 0.5};
  }
  // |N / D - n / d| <= (|N - n| + |n / d| |D - d|) / |D|, plus the division.
  double bound = ((n.bound + std::fabs(t) * d.bound) / lower + kUnitRoundoff * std::fabs(t)) *
                 (1.0 + 0x1p-50);
  t = clip::clamp(t, 0.0, 1.0);
  return {t, std::min(bound, 1.0)};
}

ContactPoint exact_point(const double* x, int8_t first, int8_t second) {
  return {{x[0], x[1], x[2]}, 0.0, first, second};
}

ContactPoint constructed_point(const double* p, const double* q, const Approx& t, int8_t first,
                               int8_t second) {
  ContactPoint point{{0.0, 0.0, 0.0}, 0.0, first, second};
  for (int k = 0; k < 3; ++k) {
    const Approx x = Approx::exact(p[k]) + t * (Approx::exact(q[k]) - Approx::exact(p[k]));
    point.x[k] = x.value;
    point.bound = std::max(point.bound, x.bound);
  }
  return point;
}

// Shared adjacency of a vertex-vertex contact (see intersections.hpp).
bool shared(const ContactPoint& point, const int64_t* first_ids, const int64_t* second_ids) {
  if (!is_vertex_feature(point.first_feature) || !is_vertex_feature(point.second_feature)) {
    return false;
  }
  return first_ids == nullptr ||
         first_ids[point.first_feature] == second_ids[point.second_feature];
}

// Distinct ids within each input; equal ids across inputs name one position.
int32_t check_ids(const double* const first[], int first_count, const int64_t* first_ids,
                  const double* const second[], int second_count, const int64_t* second_ids) {
  if (first_ids == nullptr) {
    return PHX_MC_OK;
  }
  for (int i = 0; i < first_count; ++i) {
    for (int j = i + 1; j < first_count; ++j) {
      if (first_ids[i] == first_ids[j]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int i = 0; i < second_count; ++i) {
    for (int j = i + 1; j < second_count; ++j) {
      if (second_ids[i] == second_ids[j]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  for (int i = 0; i < first_count; ++i) {
    for (int j = 0; j < second_count; ++j) {
      if (first_ids[i] == second_ids[j] && !same_point(first[i], second[j])) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  return PHX_MC_OK;
}

// ------------------------------------------------------------ transversal case

// (u, w) n plane: w strictly off the plane, u on the closed opposite side.
struct PlanePoint {
  int u;
  int w;
};

// One triangle's intersection with the other triangle's plane.
struct PlaneSection {
  const double* const* v;
  const int* side;  // orient3d signs of v with respect to the other plane
  PlanePoint low;
  PlanePoint high;  // low and high are ordered along d
  bool single;
  int8_t interior;  // feature of the points strictly between low and high
};

PlaneSection section(const double* const v[3], const int side[3], int direction) {
  PlaneSection result{v, side, {0, 0}, {0, 0}, false, kFeatureInterior};
  for (int l = 0; l < 3; ++l) {
    const int next = (l + 1) % 3;
    const int last = (l + 2) % 3;
    if (side[l] != 0 && side[next] * side[l] <= 0 && side[last] * side[l] <= 0) {
      const PlanePoint first{next, l};
      const PlanePoint second{last, l};
      if (-side[l] * direction > 0) {
        result.low = first;
        result.high = second;
      } else {
        result.low = second;
        result.high = first;
      }
      if (side[next] == 0 && side[last] == 0) {
        result.interior = edge_feature(l);
      }
      return result;
    }
  }
  // No lone vertex: exactly one vertex touches the plane, the others lie
  // strictly on one side.
  for (int z = 0; z < 3; ++z) {
    if (side[z] == 0) {
      result.low = result.high = PlanePoint{z, (z + 1) % 3};
      result.single = true;
    }
  }
  return result;
}

int8_t feature_of(const PlaneSection& section, const PlanePoint& point) {
  if (section.side[point.u] == 0) {
    return static_cast<int8_t>(point.u);
  }
  return edge_feature(3 - point.u - point.w);
}

// sign(d . (Y - X)) for X on the first section and Y on the second.
int compare(const PlaneSection& first, const PlanePoint& x, const PlaneSection& second,
            const PlanePoint& y) {
  return orient3d(first.v[x.u], first.v[x.w], second.v[y.u], second.v[y.w]) * first.side[x.w] *
         second.side[y.w];
}

// A contact endpoint on L with its optional representations on each section.
struct LineEnd {
  const PlanePoint* first;
  const PlanePoint* second;
};

ContactPoint construct_on_plane(const PlaneSection& section, const PlanePoint& point,
                                const double* const plane[3], int8_t first, int8_t second) {
  const double* u = section.v[point.u];
  const double* w = section.v[point.w];
  const Expansion su = orient3d_exact(plane[0], plane[1], plane[2], u);
  const Expansion sw = orient3d_exact(plane[0], plane[1], plane[2], w);
  return constructed_point(u, w, parameter(su, su - sw), first, second);
}

ContactPoint line_point(const PlaneSection& a, const PlaneSection& b, const LineEnd& end,
                        const double* const first[3], const double* const second[3]) {
  const int8_t fa = end.first != nullptr ? feature_of(a, *end.first) : a.interior;
  const int8_t fb = end.second != nullptr ? feature_of(b, *end.second) : b.interior;
  if (is_vertex_feature(fa)) {
    return exact_point(first[fa], fa, fb);
  }
  if (is_vertex_feature(fb)) {
    return exact_point(second[fb], fa, fb);
  }
  if (end.first != nullptr) {
    return construct_on_plane(a, *end.first, second, fa, fb);
  }
  return construct_on_plane(b, *end.second, first, fa, fb);
}

void transversal(const double* const first[3], const double* const second[3], const int s[3],
                 const int r[3], const int64_t* first_ids, const int64_t* second_ids,
                 Contact& contact) {
  const PlaneSection a = section(first, s, 1);
  const PlaneSection b = section(second, r, -1);
  const int low_high = compare(a, a.low, b, b.high);
  const int high_low = compare(a, a.high, b, b.low);
  if (low_high < 0 || high_low > 0) {
    return;
  }
  LineEnd ends[2];
  int count = 1;
  if (low_high == 0) {
    ends[0] = {&a.low, &b.high};
  } else if (high_low == 0) {
    ends[0] = {&a.high, &b.low};
  } else if (a.single) {
    ends[0] = {&a.low, nullptr};
  } else if (b.single) {
    ends[0] = {nullptr, &b.low};
  } else {
    count = 2;
    const int low = compare(a, a.low, b, b.low);
    const int high = compare(a, a.high, b, b.high);
    ends[0] = low > 0 ? LineEnd{nullptr, &b.low}
                      : (low < 0 ? LineEnd{&a.low, nullptr} : LineEnd{&a.low, &b.low});
    ends[1] = high > 0 ? LineEnd{&a.high, nullptr}
                       : (high < 0 ? LineEnd{nullptr, &b.high} : LineEnd{&a.high, &b.high});
  }
  contact.count = count;
  for (int k = 0; k < count; ++k) {
    contact.points[k] = line_point(a, b, ends[k], first, second);
  }
  if (count == 2 && a.interior == kFeatureInterior && b.interior == kFeatureInterior) {
    contact.kind = PHX_MC_CROSSING;
  } else if (count == 1 && shared(contact.points[0], first_ids, second_ids)) {
    contact.kind = PHX_MC_SHARED_VERTEX;
  } else if (count == 2 && shared(contact.points[0], first_ids, second_ids) &&
             shared(contact.points[1], first_ids, second_ids)) {
    contact.kind = PHX_MC_SHARED_EDGE;
  } else {
    contact.kind = PHX_MC_TOUCHING;
  }
}

// -------------------------------------------------------------- coplanar case

// Axis whose exact normal component of the triangle is nonzero (the largest
// estimate among them); orient2d on the other two axes is then exact on the
// triangle's plane.
int projection_axis(const double* const t[3]) {
  int best = -1;
  double best_magnitude = -1.0;
  for (int axis = 0; axis < 3; ++axis) {
    const int i = (axis + 1) % 3;
    const int j = (axis + 2) % 3;
    const double p[2] = {t[0][i], t[0][j]};
    const double q[2] = {t[1][i], t[1][j]};
    const double r[2] = {t[2][i], t[2][j]};
    if (orient2d(p, q, r) != 0) {
      const double magnitude = std::fabs(orient2d_approx(p, q, r).value);
      if (magnitude > best_magnitude) {
        best = axis;
        best_magnitude = magnitude;
      }
    }
  }
  return best;
}

// Symbolic vertex of a clipped polygon.
struct ClipVertex {
  enum Kind : int8_t { kSubject, kClip, kCrossing };
  Kind kind;
  int8_t index;  // subject vertex, clip vertex, or crossing clip line
  int8_t line;   // crossing subject line
  int8_t label;  // outgoing edge line: subject line k, or 3 + clip line m
};

// Clips a subject triangle (three points, line k opposite vertex k) or segment
// (two points, line 0) by the closed halfplanes of a clip triangle in one
// exact projection.
class CoplanarClipper {
 public:
  CoplanarClipper(const double* const subject[], int subject_count, const double* const clip[3],
                  int axis)
      : subject_count_(subject_count) {
    const int i = (axis + 1) % 3;
    const int j = (axis + 2) % 3;
    for (int k = 0; k < subject_count; ++k) {
      subject_[k][0] = subject[k][i];
      subject_[k][1] = subject[k][j];
    }
    for (int k = 0; k < 3; ++k) {
      clip_[k][0] = clip[k][i];
      clip_[k][1] = clip[k][j];
    }
    clip_orientation_ = orient2d(clip_[0], clip_[1], clip_[2]);
    if (subject_count == 3) {
      subject_orientation_ = orient2d(subject_[0], subject_[1], subject_[2]);
    }
  }

  // Runs the clip; the result is in vertices()/count().
  void run() {
    count_ = subject_count_;
    for (int k = 0; k < subject_count_; ++k) {
      vertices_[k] = {ClipVertex::kSubject, static_cast<int8_t>(k), -1,
                      static_cast<int8_t>(subject_count_ == 3 ? (k + 2) % 3 : 0)};
    }
    for (int m = 0; m < 3 && count_ > 0; ++m) {
      clip_line(m);
    }
  }

  int count() const { return count_; }
  const ClipVertex& vertex(int k) const { return vertices_[k]; }

  // Exact side of a symbolic vertex with respect to clip line q (inside >= 0).
  int side(const ClipVertex& vertex, int q) const {
    switch (vertex.kind) {
      case ClipVertex::kSubject:
        return clip_orientation_ * orient2d(clip_[(q + 1) % 3], clip_[(q + 2) % 3],
                                            subject_[vertex.index]);
      case ClipVertex::kClip:
        return vertex.index == q ? 1 : 0;
      case ClipVertex::kCrossing:
        break;
    }
    if (vertex.index == q) {
      return 0;
    }
    const double* p = line_start(vertex.line);
    const double* r = line_end(vertex.line);
    const Expansion om_p = clip_value(vertex.index, p);
    const Expansion om_r = clip_value(vertex.index, r);
    const Expansion oq_p = clip_value(q, p);
    const Expansion oq_r = clip_value(q, r);
    return (om_p * oq_r - om_r * oq_p).sign() * (om_p - om_r).sign() * clip_orientation_;
  }

  // Sides of a subject-plane point with respect to the subject triangle edges.
  int subject_side(int k, const double* point) const {
    return subject_orientation_ * orient2d(subject_[(k + 1) % 3], subject_[(k + 2) % 3], point);
  }

  const double* projected_clip(int j) const { return clip_[j]; }

  // Crossing parameter along subject line `line` with clip line m.
  Approx crossing_parameter(int line, int m) const {
    const Expansion om_p = clip_value(m, line_start(line));
    const Expansion om_r = clip_value(m, line_end(line));
    return parameter(om_p, om_p - om_r);
  }

  int subject_line_start(int line) const { return subject_count_ == 3 ? (line + 1) % 3 : 0; }
  int subject_line_end(int line) const { return subject_count_ == 3 ? (line + 2) % 3 : 1; }

 private:
  const double* line_start(int line) const { return subject_[subject_line_start(line)]; }
  const double* line_end(int line) const { return subject_[subject_line_end(line)]; }

  Expansion clip_value(int m, const double* point) const {
    return orient2d_exact(clip_[(m + 1) % 3], clip_[(m + 2) % 3], point);
  }

  ClipVertex crossing(const ClipVertex& from, int m) const {
    if (from.label >= 3) {
      // Two clip lines meet at the clip vertex they share.
      return {ClipVertex::kClip, static_cast<int8_t>(3 - m - (from.label - 3)), -1, -1};
    }
    return {ClipVertex::kCrossing, static_cast<int8_t>(m), from.label, -1};
  }

  void clip_line(int m) {
    ClipVertex out[kMaxContactPoints + 2];
    int emitted = 0;
    int sides[kMaxContactPoints + 2];
    for (int k = 0; k < count_; ++k) {
      sides[k] = side(vertices_[k], m);
    }
    if (count_ <= 2) {
      const bool segment = count_ == 2;
      if (sides[0] >= 0) {
        out[emitted++] = vertices_[0];
      }
      if (segment && sides[0] * sides[1] < 0) {
        ClipVertex x = crossing(vertices_[0], m);
        x.label = vertices_[0].label;
        out[emitted++] = x;
      }
      if (segment && sides[1] >= 0) {
        out[emitted++] = vertices_[1];
      }
      for (int k = 0; k < emitted; ++k) {
        out[k].label = vertices_[0].label;
      }
    } else {
      for (int k = 0; k < count_; ++k) {
        const int next = (k + 1) % count_;
        const ClipVertex& current = vertices_[k];
        if (sides[k] > 0) {
          out[emitted++] = current;
          if (sides[next] < 0) {
            ClipVertex x = crossing(current, m);
            x.label = static_cast<int8_t>(3 + m);
            out[emitted++] = x;
          }
        } else if (sides[k] == 0) {
          ClipVertex kept = current;
          if (sides[next] < 0) {
            kept.label = static_cast<int8_t>(3 + m);
          }
          out[emitted++] = kept;
        } else if (sides[next] > 0) {
          ClipVertex x = crossing(current, m);
          x.label = current.label;
          out[emitted++] = x;
        }
      }
    }
    count_ = emitted;
    std::copy(out, out + emitted, vertices_);
  }

  int subject_count_;
  double subject_[3][2];
  double clip_[3][2];
  int clip_orientation_ = 0;
  int subject_orientation_ = 0;
  ClipVertex vertices_[kMaxContactPoints + 2];
  int count_ = 0;
};

// Contact points of a finished clip; `on_clip_line[k]` receives the bitmask of
// clip lines through point k.
void coplanar_points(const CoplanarClipper& clipper, const double* const subject[],
                     int subject_count, const double* const clip[3], Contact& contact,
                     int on_clip_line[]) {
  contact.count = clipper.count();
  for (int k = 0; k < clipper.count(); ++k) {
    const ClipVertex& vertex = clipper.vertex(k);
    int sides[3];
    on_clip_line[k] = 0;
    for (int q = 0; q < 3; ++q) {
      sides[q] = clipper.side(vertex, q);
      if (sides[q] == 0) {
        on_clip_line[k] |= 1 << q;
      }
    }
    const int8_t clip_feature = feature_from_sides(sides);
    switch (vertex.kind) {
      case ClipVertex::kSubject:
        contact.points[k] = exact_point(subject[vertex.index], vertex.index, clip_feature);
        break;
      case ClipVertex::kClip: {
        int8_t subject_feature = kFeatureInterior;
        if (subject_count == 3) {
          int subject_sides[3];
          for (int e = 0; e < 3; ++e) {
            subject_sides[e] = clipper.subject_side(e, clipper.projected_clip(vertex.index));
          }
          subject_feature = feature_from_sides(subject_sides);
        } else if (same_point(clip[vertex.index], subject[0])) {
          subject_feature = 0;
        } else if (same_point(clip[vertex.index], subject[1])) {
          subject_feature = 1;
        }
        contact.points[k] = exact_point(clip[vertex.index], subject_feature, vertex.index);
        break;
      }
      case ClipVertex::kCrossing: {
        const int8_t subject_feature =
            subject_count == 3 ? edge_feature(vertex.line) : kFeatureInterior;
        if (is_vertex_feature(clip_feature)) {
          contact.points[k] = exact_point(clip[clip_feature], subject_feature, clip_feature);
        } else {
          contact.points[k] = constructed_point(
              subject[clipper.subject_line_start(vertex.line)],
              subject[clipper.subject_line_end(vertex.line)],
              clipper.crossing_parameter(vertex.line, vertex.index), subject_feature,
              clip_feature);
        }
        break;
      }
    }
  }
}

void coplanar_triangles(const double* const first[3], const double* const second[3],
                        const int64_t* first_ids, const int64_t* second_ids, Contact& contact) {
  CoplanarClipper clipper(first, 3, second, projection_axis(first));
  clipper.run();
  int on_line[kMaxContactPoints];
  coplanar_points(clipper, first, 3, second, contact, on_line);
  switch (contact.count) {
    case 0:
      contact.kind = PHX_MC_DISJOINT;
      return;
    case 1:
      contact.kind = shared(contact.points[0], first_ids, second_ids) ? PHX_MC_SHARED_VERTEX
                                                                       : PHX_MC_TOUCHING;
      return;
    case 2:
      contact.kind = shared(contact.points[0], first_ids, second_ids) &&
                             shared(contact.points[1], first_ids, second_ids)
                         ? PHX_MC_SHARED_EDGE
                         : PHX_MC_TOUCHING;
      return;
    default:
      break;
  }
  bool coincident = contact.count == 3;
  for (int k = 0; k < contact.count && coincident; ++k) {
    coincident = is_vertex_feature(contact.points[k].first_feature) &&
                 is_vertex_feature(contact.points[k].second_feature);
  }
  contact.kind = coincident ? PHX_MC_COINCIDENT : PHX_MC_COPLANAR_OVERLAP;
}

void coplanar_segment(const double* const segment[2], const double* const triangle[3],
                      const int64_t* segment_ids, const int64_t* triangle_ids,
                      Contact& contact) {
  CoplanarClipper clipper(segment, 2, triangle, projection_axis(triangle));
  clipper.run();
  int on_line[kMaxContactPoints];
  coplanar_points(clipper, segment, 2, triangle, contact, on_line);
  switch (contact.count) {
    case 0:
      contact.kind = PHX_MC_DISJOINT;
      return;
    case 1:
      contact.kind = shared(contact.points[0], segment_ids, triangle_ids) ? PHX_MC_SHARED_VERTEX
                                                                           : PHX_MC_TOUCHING;
      return;
    default:
      break;
  }
  if (shared(contact.points[0], segment_ids, triangle_ids) &&
      shared(contact.points[1], segment_ids, triangle_ids)) {
    contact.kind = PHX_MC_SHARED_EDGE;
  } else if ((on_line[0] & on_line[1]) != 0) {
    contact.kind = PHX_MC_TOUCHING;
  } else {
    contact.kind = PHX_MC_COPLANAR_OVERLAP;
  }
}

void reset(Contact& contact) {
  contact.kind = PHX_MC_DISJOINT;
  contact.count = 0;
}

}  // namespace

int32_t intersect_triangles(const double* const first[3], const double* const second[3],
                            const int64_t* first_ids, const int64_t* second_ids,
                            Contact& contact) {
  reset(contact);
  if (collinear3d(first[0], first[1], first[2]) || collinear3d(second[0], second[1], second[2])) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  const int32_t status = check_ids(first, 3, first_ids, second, 3, second_ids);
  if (status != PHX_MC_OK) {
    return status;
  }
  int s[3];
  int r[3];
  for (int k = 0; k < 3; ++k) {
    s[k] = orient3d(second[0], second[1], second[2], first[k]);
  }
  if (s[0] == 0 && s[1] == 0 && s[2] == 0) {
    coplanar_triangles(first, second, first_ids, second_ids, contact);
    return PHX_MC_OK;
  }
  if ((s[0] > 0 && s[1] > 0 && s[2] > 0) || (s[0] < 0 && s[1] < 0 && s[2] < 0)) {
    return PHX_MC_OK;
  }
  for (int k = 0; k < 3; ++k) {
    r[k] = orient3d(first[0], first[1], first[2], second[k]);
  }
  if ((r[0] > 0 && r[1] > 0 && r[2] > 0) || (r[0] < 0 && r[1] < 0 && r[2] < 0)) {
    return PHX_MC_OK;
  }
  transversal(first, second, s, r, first_ids, second_ids, contact);
  return PHX_MC_OK;
}

int32_t intersect_segment_triangle(const double* const segment[2],
                                   const double* const triangle[3], const int64_t* segment_ids,
                                   const int64_t* triangle_ids, Contact& contact) {
  reset(contact);
  if (same_point(segment[0], segment[1]) || collinear3d(triangle[0], triangle[1], triangle[2])) {
    return PHX_MC_DEGENERATE_INPUT;
  }
  const int32_t status = check_ids(segment, 2, segment_ids, triangle, 3, triangle_ids);
  if (status != PHX_MC_OK) {
    return status;
  }
  const int s0 = orient3d(triangle[0], triangle[1], triangle[2], segment[0]);
  const int s1 = orient3d(triangle[0], triangle[1], triangle[2], segment[1]);
  if (s0 == 0 && s1 == 0) {
    coplanar_segment(segment, triangle, segment_ids, triangle_ids, contact);
    return PHX_MC_OK;
  }
  if (s0 * s1 > 0) {
    return PHX_MC_OK;
  }
  // The line is transversal: it meets the closed triangle iff its orientations
  // with respect to the three edges agree in sign.
  int sides[3];
  bool positive = false;
  bool negative = false;
  for (int k = 0; k < 3; ++k) {
    sides[k] = orient3d(segment[0], segment[1], triangle[(k + 1) % 3], triangle[(k + 2) % 3]);
    positive = positive || sides[k] > 0;
    negative = negative || sides[k] < 0;
  }
  if (positive && negative) {
    return PHX_MC_OK;
  }
  const int8_t triangle_feature = feature_from_sides(sides);
  const int8_t segment_feature = s0 == 0 ? 0 : (s1 == 0 ? 1 : kFeatureInterior);
  contact.count = 1;
  if (is_vertex_feature(segment_feature)) {
    contact.points[0] = exact_point(segment[segment_feature], segment_feature, triangle_feature);
  } else if (is_vertex_feature(triangle_feature)) {
    contact.points[0] = exact_point(triangle[triangle_feature], segment_feature, triangle_feature);
  } else {
    const Expansion e0 = orient3d_exact(triangle[0], triangle[1], triangle[2], segment[0]);
    const Expansion e1 = orient3d_exact(triangle[0], triangle[1], triangle[2], segment[1]);
    contact.points[0] = constructed_point(segment[0], segment[1], parameter(e0, e0 - e1),
                                          segment_feature, triangle_feature);
  }
  if (shared(contact.points[0], segment_ids, triangle_ids)) {
    contact.kind = PHX_MC_SHARED_VERTEX;
  } else if (segment_feature == kFeatureInterior && triangle_feature == kFeatureInterior) {
    contact.kind = PHX_MC_CROSSING;
  } else {
    contact.kind = PHX_MC_TOUCHING;
  }
  return PHX_MC_OK;
}

int8_t locate_on_triangle(const double* x, const double* const triangle[3], int& side) {
  side = orient3d(triangle[0], triangle[1], triangle[2], x);
  if (side != 0) {
    return kFeatureNone;
  }
  const int axis = projection_axis(triangle);
  const int i = (axis + 1) % 3;
  const int j = (axis + 2) % 3;
  const double t[3][2] = {{triangle[0][i], triangle[0][j]},
                          {triangle[1][i], triangle[1][j]},
                          {triangle[2][i], triangle[2][j]}};
  const double p[2] = {x[i], x[j]};
  const int orientation = orient2d(t[0], t[1], t[2]);
  int sides[3];
  for (int k = 0; k < 3; ++k) {
    sides[k] = orientation * orient2d(t[(k + 1) % 3], t[(k + 2) % 3], p);
    if (sides[k] < 0) {
      return kFeatureNone;
    }
  }
  return feature_from_sides(sides);
}

namespace {

// Item validation: finite values, then the exact domain.
int32_t check_values(const double* values, int64_t count) {
  const int32_t status = clip::check_finite(values, count);
  return status != PHX_MC_OK ? status : clip::check_coordinates(values, count);
}

// Optional construction outputs of a batched contact call.
struct ContactOutput {
  int32_t* point_counts;
  double* points;
  double* bounds;
  int8_t* features;
  int capacity;

  bool present() const { return point_counts != nullptr; }
  bool consistent() const {
    const bool all = points != nullptr && bounds != nullptr && features != nullptr;
    const bool none = points == nullptr && bounds == nullptr && features == nullptr;
    return present() ? all : none;
  }

  void write(int64_t item, const Contact& contact) const {
    if (!present()) {
      return;
    }
    point_counts[item] = contact.count;
    for (int k = 0; k < capacity; ++k) {
      const int64_t slot = item * capacity + k;
      const bool used = k < contact.count;
      for (int axis = 0; axis < 3; ++axis) {
        points[3 * slot + axis] = used ? contact.points[k].x[axis] : 0.0;
      }
      bounds[slot] = used ? contact.points[k].bound : 0.0;
      features[2 * slot] = used ? contact.points[k].first_feature : kFeatureNone;
      features[2 * slot + 1] = used ? contact.points[k].second_feature : kFeatureNone;
    }
  }
};

template <class Classify>
int32_t contact_batch(int64_t count, const double* first, int first_points, const double* second,
                      const int64_t* first_ids, const int64_t* second_ids, int8_t* classes,
                      const ContactOutput& output, int32_t* item_status,
                      const Classify& classify) {
  const int64_t first_width = 3 * first_points;
  if (count < 0 || !addressable(count, first_width, sizeof(double)) ||
      !addressable(count, 9, sizeof(double)) ||
      !addressable(count, 2 * output.capacity, sizeof(double)) || classes == nullptr ||
      item_status == nullptr || !output.consistent() || (first_ids == nullptr) != (second_ids == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (count > 0 && (first == nullptr || second == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  Contact contact;
  for (int64_t item = 0; item < count; ++item) {
    const double* a = first + item * first_width;
    const double* b = second + item * 9;
    int32_t status = check_values(a, first_width);
    if (status == PHX_MC_OK) {
      status = check_values(b, 9);
    }
    reset(contact);
    if (status == PHX_MC_OK) {
      const int64_t* a_ids = first_ids == nullptr ? nullptr : first_ids + item * first_points;
      const int64_t* b_ids = second_ids == nullptr ? nullptr : second_ids + item * 3;
      status = classify(a, b, a_ids, b_ids, contact);
    }
    if (status != PHX_MC_OK) {
      reset(contact);
    }
    item_status[item] = status;
    classes[item] = status == PHX_MC_OK ? contact.kind : static_cast<int8_t>(-1);
    output.write(item, contact);
  }
  return PHX_MC_OK;
}

}  // namespace
}  // namespace phx::mc

extern "C" {

int32_t phx_mc_triangle_intersections(int64_t count, const double* first, const double* second,
                                      const int64_t* first_ids, const int64_t* second_ids,
                                      int8_t* classes, int32_t* point_counts, double* points,
                                      double* point_bounds, int8_t* point_features,
                                      int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::ContactOutput output{point_counts, points, point_bounds, point_features,
                                        phx::mc::kMaxContactPoints};
    return phx::mc::contact_batch(
        count, first, 3, second, first_ids, second_ids, classes, output, item_status,
        [](const double* a, const double* b, const int64_t* a_ids, const int64_t* b_ids,
           phx::mc::Contact& contact) {
          const double* const ta[3] = {a, a + 3, a + 6};
          const double* const tb[3] = {b, b + 3, b + 6};
          return phx::mc::intersect_triangles(ta, tb, a_ids, b_ids, contact);
        });
  });
}

int32_t phx_mc_segment_triangle_intersections(int64_t count, const double* segments,
                                              const double* triangles,
                                              const int64_t* segment_ids,
                                              const int64_t* triangle_ids, int8_t* classes,
                                              int32_t* point_counts, double* points,
                                              double* point_bounds, int8_t* point_features,
                                              int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::ContactOutput output{point_counts, points, point_bounds, point_features, 2};
    return phx::mc::contact_batch(
        count, segments, 2, triangles, segment_ids, triangle_ids, classes, output, item_status,
        [](const double* s, const double* t, const int64_t* s_ids, const int64_t* t_ids,
           phx::mc::Contact& contact) {
          const double* const segment[2] = {s, s + 3};
          const double* const triangle[3] = {t, t + 3, t + 6};
          return phx::mc::intersect_segment_triangle(segment, triangle, s_ids, t_ids, contact);
        });
  });
}

int32_t phx_mc_point_triangle_locations(int64_t count, const double* points,
                                        const double* triangles, int8_t* sides,
                                        int8_t* features, int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    if (count < 0 || !phx::mc::addressable(count, 9, sizeof(double)) || sides == nullptr ||
        features == nullptr || item_status == nullptr ||
        (count > 0 && (points == nullptr || triangles == nullptr))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    for (int64_t item = 0; item < count; ++item) {
      const double* x = points + 3 * item;
      const double* t = triangles + 9 * item;
      int32_t status = phx::mc::check_values(x, 3);
      if (status == PHX_MC_OK) {
        status = phx::mc::check_values(t, 9);
      }
      const double* const triangle[3] = {t, t + 3, t + 6};
      if (status == PHX_MC_OK && phx::mc::collinear3d(t, t + 3, t + 6)) {
        status = PHX_MC_DEGENERATE_INPUT;
      }
      int side = 0;
      int8_t feature = phx::mc::kFeatureNone;
      if (status == PHX_MC_OK) {
        feature = phx::mc::locate_on_triangle(x, triangle, side);
      }
      item_status[item] = status;
      sides[item] = static_cast<int8_t>(side);
      features[item] = feature;
    }
    return PHX_MC_OK;
  });
}

}  // extern "C"
