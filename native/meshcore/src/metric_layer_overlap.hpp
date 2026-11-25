// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "cavity.hpp"
#include "clip_common.hpp"
#include "exact_scalar.hpp"
#include "geometry3.hpp"

namespace phx::mc::metric::layer_detail {

// A binary64 source coordinate is a multiple of 2^-172 in the native
// predicate domain. With <=53 dyadic weight bits the stencil image is a
// multiple of 2^-225, so even a cubic determinant evaluates by expansions
// without underflow. The signed one-ULP displacement encloses this image
// and the represented point; zero denotes an exactly represented axis.
struct StencilPoint {
  std::array<Expansion, 3> position;
  std::array<double, 3> displacement{};
  bool exact = true;
};

struct PointRef {
  const double* rounded;
  const StencilPoint* stencil = nullptr;
};
using Tet = std::array<PointRef, 4>;

class CommonSigns {
 public:
  // True while the predicates deciding the exact P1 classification are the
  // same on rounded native coordinates. An interval sign holds on both;
  // if any fallback differs, the caller classifies native geometry as well.
  bool same() const { return same_; }

  int orientation(PointRef a, PointRef b, PointRef c, PointRef d) {
    if (a.stencil == nullptr && b.stencil == nullptr && c.stencil == nullptr && d.stencil == nullptr) {
      return orient3d(a.rounded, b.rounded, c.rounded, d.rounded);
    }
    const int interval = determinant<Approx>(a, b, a, c, a, d).certified_sign();
    if (interval != 2) return interval;
    const int sign = determinant<Expansion>(a, b, a, c, a, d).sign();
    if (same_) same_ = sign == orient3d(a.rounded, b.rounded, c.rounded, d.rounded);
    return sign;
  }

  int edge_side(PointRef a, PointRef b, PointRef c, PointRef d, PointRef o, PointRef x) {
    const int interval = determinant<Approx>(a, b, c, d, o, x).certified_sign();
    if (interval != 2) return interval;
    const int sign = determinant<Expansion>(a, b, c, d, o, x).sign();
    if (same_ && (a.stencil || b.stencil || c.stencil || d.stencil || o.stencil || x.stencil)) {
      a.stencil = b.stencil = c.stencil = d.stencil = o.stencil = x.stencil = nullptr;
      same_ = sign == clip::exact_sign([&]<class T>() {
        return determinant<T>(a, b, c, d, o, x);
      });
    }
    return sign;
  }

 private:
  template <class T>
  static T difference(PointRef tail, PointRef head, int axis) {
    const double a = tail.rounded[axis];
    const double b = head.rounded[axis];
    if (tail.stencil == nullptr && head.stencil == nullptr) return clip::diff<T>(b, a);
    if constexpr (std::is_same_v<T, Approx>) {
      const double ea = tail.stencil == nullptr ? 0.0 : std::abs(tail.stencil->displacement[axis]);
      const double eb = head.stencil == nullptr ? 0.0 : std::abs(head.stencil->displacement[axis]);
      return Approx{b, eb} - Approx{a, ea};
    } else {
      return (head.stencil == nullptr ? Expansion(b) : head.stencil->position[axis]) -
             (tail.stencil == nullptr ? Expansion(a) : tail.stencil->position[axis]);
    }
  }

  template <class T>
  static T determinant(PointRef a, PointRef b, PointRef c, PointRef d, PointRef o, PointRef x) {
    const T u[3]{difference<T>(a, b, 0), difference<T>(a, b, 1), difference<T>(a, b, 2)};
    const T v[3]{difference<T>(c, d, 0), difference<T>(c, d, 1), difference<T>(c, d, 2)};
    const T w[3]{difference<T>(o, x, 0), difference<T>(o, x, 1), difference<T>(o, x, 2)};
    return exact::det3(u, v, w);
  }

  bool same_ = true;
};

enum class Contact { kDisjoint, kPositive, kEdges };

inline Contact face_contact(const Tet& first, const Tet& second, CommonSigns& signs) {
  unsigned strict = 15;
  unsigned closed = 15;
  for (const auto& face : kOrientedFacet) {
    unsigned positive = 0;
    unsigned on = 0;
    for (int j = 0; j < 4; ++j) {
      const int side = signs.orientation(first[face[0]], first[face[1]], first[face[2]], second[j]);
      if (side > 0) positive |= 1U << j;
      if (side == 0) on |= 1U << j;
    }
    if (positive == 0) return Contact::kDisjoint;
    strict &= positive;
    closed &= positive | on;
  }
  // The query tet is positive, so a strictly interior vertex or closed
  // containment gives a full-dimensional intersection.
  return strict != 0 || closed == 15 ? Contact::kPositive : Contact::kEdges;
}

using Edge = std::array<int, 2>;
inline constexpr Edge kEdges[]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

// A common error radius for (u x v) . w, where each vector is a difference
// of two pair vertices. It encloses the determinants of both embeddings
// around the determinant computed from the rounded coordinates.
inline double common_distance_error(const Tet& first, const Tet& second) {
  double lo[3] = {first[0].rounded[0], first[0].rounded[1], first[0].rounded[2]};
  double hi[3] = {lo[0], lo[1], lo[2]};
  double radius[3]{};
  for (const Tet* tet : {&first, &second}) {
    for (const auto& vertex : *tet) {
      for (int i = 0; i < 3; ++i) {
        const double x = vertex.rounded[i];
        if (!(std::abs(x) <= 0x1p199)) return 0.0;
        lo[i] = std::min(lo[i], x);
        hi[i] = std::max(hi[i], x);
      }
      if (vertex.stencil != nullptr) {
        for (int i = 0; i < 3; ++i) {
          const double r = std::abs(vertex.stencil->displacement[i]);
          if (!(r <= 0x1p199)) return 0.0;
          radius[i] = std::max(radius[i], r);
        }
      }
    }
  }
  double s[3], e[3];
  for (int i = 0; i < 3; ++i) {
    // s bounds both exact and floating differences, e accounts for the
    // two coordinate intervals and subtraction roundoff (u = 2^-53).
    const double r = 2.0 * radius[i];
    s[i] = std::max(((hi[i] - lo[i]) + r) * (1.0 + 0x1p-50), 0x1p-200);
    if (!(s[i] <= 0x1p200)) return 0.0;
    e[i] = r + 0x1p-52 * s[i];
  }
  // Each of the six monomials uses each axis once. Telescoping the
  // input perturbations costs <= 6 sum(e_i s_j s_k); evaluating the
  // cross/dot costs at most five roundings per monomial. The factor 16
  // covers both, including rounding in this bound. The scale caps keep
  // products finite, and the floor keeps the radius (> 2^-650) well above
  // any gradual-underflow error (even after multiplication by 2^200).
  return 16.0 * ((e[0] * s[1] + s[0] * e[1]) * s[2] + (s[0] * s[1]) * e[2]);
}

inline double projected_distance(const double* normal, const double* origin, const double* point) {
  double delta[3];
  geometry::difference(point, origin, delta);
  return geometry::dot(normal, delta);
}

inline Contact certified_faces(const Tet& first, const Tet& second, double error,
                               double (&distance)[4][4]) {
  unsigned strict = 15;
  bool centroid_inside = true;
  for (int i = 0; i < 4; ++i) {
    const auto& face = kOrientedFacet[i];
    const double* origin = first[face[0]].rounded;
    double u[3], v[3], normal[3];
    geometry::difference(first[face[1]].rounded, origin, u);
    geometry::difference(first[face[2]].rounded, origin, v);
    geometry::cross(u, v, normal);
    auto& row = distance[i];
    unsigned inside = 0;
    for (int j = 0; j < 4; ++j) {
      row[j] = projected_distance(normal, origin, second[j].rounded);
      if (row[j] > error) inside |= 1U << j;
    }
    if (row[0] < -error && row[1] < -error && row[2] < -error && row[3] < -error) {
      return Contact::kDisjoint;
    }
    strict &= inside;
    // A strictly interior vertex or centroid proves positive volume.
    // The latter also covers closed containment, using affine distances
    // without constructing the centroid. 8*error includes sum roundoff.
    centroid_inside = centroid_inside && ((row[0] + row[1]) + (row[2] + row[3]) > 8.0 * error);
  }
  return strict != 0 || centroid_inside ? Contact::kPositive : Contact::kEdges;
}

inline bool certified_crossing(const double (&distance)[4][4], double error) {
  const double margin = 2.0 * error;
  for (const auto& edge : kEdges) {
    double low = 0.0;
    double high = 1.0;
    for (const auto& face : distance) {
      const double a = face[edge[0]], b = face[edge[1]];
      if (a > margin && b > margin) continue;
      if (a <= margin && b <= margin) {
        high = low;
        break;
      }
      const double t = (margin - a) / (b - a);
      if (a <= margin) low = std::max(low, t);
      else high = std::min(high, t);
      if (low >= high) break;
    }
    if (!(low < high)) continue;
    const double t = 0.5 * (low + high);
    if (!(t > 0.0 && t < 1.0)) continue;
    // The clip above only proposes a witness. Prove that the same affine
    // parameter is strictly inside every face in both embeddings. The
    // extra error covers the floating interpolation of the cached values.
    bool inside = true;
    for (const auto& face : distance) {
      const double a = face[edge[0]], b = face[edge[1]];
      if (!(a + t * (b - a) > margin)) {
        inside = false;
        break;
      }
    }
    if (inside) return true;
  }
  return false;
}

inline bool certified_edge_separation(const Tet& first, const Tet& second, double error) {
  constexpr int count = sizeof(kEdges) / sizeof(Edge);
  double direction[count][3];
  for (int i = 0; i < count; ++i) {
    const auto& q = kEdges[i];
    geometry::difference(second[q[1]].rounded, second[q[0]].rounded, direction[i]);
  }
  const double margin = 4.0 * error;
  for (const auto& p : kEdges) {
    const double* origin = first[p[0]].rounded;
    double u[3], relative[4][3];
    geometry::difference(first[p[1]].rounded, origin, u);
    for (int j = 0; j < 4; ++j) geometry::difference(second[j].rounded, origin, relative[j]);
    for (int i = 0; i < count; ++i) {
      const auto& q = kEdges[i];
      double normal[3];
      geometry::cross(u, direction[i], normal);
      const double gap = geometry::dot(normal, relative[q[0]]);
      if (!(std::abs(gap) > margin)) continue;
      const double side = gap > 0.0 ? 1.0 : -1.0;
      // In either embedding the two vertices of each generating edge
      // project identically. The remaining vertices can only shrink the gap.
      double far = 0.0;
      for (int j = 0; j < 4; ++j) {
        if (j != p[0] && j != p[1]) {
          far = std::max(far, side * projected_distance(normal, origin, first[j].rounded));
        }
      }
      double near = std::abs(gap);
      if (near - far <= margin) continue;
      for (int j = 0; j < 4; ++j) {
        if (j != q[0] && j != q[1]) {
          near = std::min(near, side * geometry::dot(normal, relative[j]));
        }
      }
      if (near - far > margin) return true;
    }
  }
  return false;
}

// Both tetrahedra are positively oriented in both embeddings. An undecided
// filter must fall through to exact SAT, notably at contact and for overlaps
// with no edge that enters the other tet's interior.
inline Contact certified_contact(const Tet& first, const Tet& second) {
  const double error = common_distance_error(first, second);
  if (error == 0.0) return Contact::kEdges;
  double first_faces[4][4], second_faces[4][4];
  const Contact a = certified_faces(first, second, error, first_faces);
  if (a != Contact::kEdges) return a;
  const Contact b = certified_faces(second, first, error, second_faces);
  if (b != Contact::kEdges) return b;
  if (certified_crossing(first_faces, error) || certified_crossing(second_faces, error)) {
    return Contact::kPositive;
  }
  if (certified_edge_separation(first, second, error)) return Contact::kDisjoint;
  return Contact::kEdges;
}

inline int edge_support(const Tet& tet, Edge edge, PointRef a, PointRef b, PointRef c, PointRef d,
                        CommonSigns& signs) {
  int support = 0;
  for (int k = 0; k < 4; ++k) {
    if (k == edge[0] || k == edge[1]) continue;
    const int side = signs.edge_side(a, b, c, d, tet[edge[0]], tet[k]);
    if (side * support < 0) return 0;
    if (side != 0) support = side;
  }
  return support;
}

// The origin lies in the interior of the Minkowski difference precisely
// when no supporting plane separates it or puts it on the boundary. Its
// facets come from a tetrahedron face or two opposing supporting edges.
// In the edge case the axis is their cross product, computed solely by
// signs of exact 3x3 determinants. Parallel edges have zero axis and no
// nonzero support sign. A zero projected gap is only contact, not volume.
inline bool positive_overlap(const Tet& first, const Tet& second, CommonSigns& signs) {
  const Contact certified = certified_contact(first, second);
  if (certified != Contact::kEdges) return certified == Contact::kPositive;
  Contact contact = face_contact(first, second, signs);
  if (contact == Contact::kEdges) contact = face_contact(second, first, signs);
  if (contact != Contact::kEdges) return contact == Contact::kPositive;
  for (Edge p : kEdges) {
    const auto a = first[p[0]];
    const auto b = first[p[1]];
    for (Edge q : kEdges) {
      const auto c = second[q[0]];
      const auto d = second[q[1]];
      const int side = edge_support(first, p, a, b, c, d, signs);
      if (side == 0 || edge_support(second, q, a, b, c, d, signs) != -side) continue;
      if (side * signs.edge_side(a, b, c, d, a, c) <= 0) return false;
    }
  }
  return true;
}

}  // namespace phx::mc::metric::layer_detail
