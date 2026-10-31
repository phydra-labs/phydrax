// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Binary64 geometry primitives. Three-term sums retain left-to-right order;
// compensated and expansion kernels keep their own arithmetic.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "predicates.hpp"
#include "rounding.hpp"

namespace phx::mc::geometry {

inline double dot(const double* a, const double* b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline void cross(const double* a, const double* b, double* result) noexcept {
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}

inline void difference(const double* a, const double* b, double* result) noexcept {
  result[0] = a[0] - b[0];
  result[1] = a[1] - b[1];
  result[2] = a[2] - b[2];
}

inline double squared_distance(const double* a, const double* b) noexcept {
  double delta[3];
  difference(a, b, delta);
  return dot(delta, delta);
}

inline bool point_in_domain(const double* point) noexcept {
  return coordinate_in_domain(point[0]) && coordinate_in_domain(point[1]) &&
         coordinate_in_domain(point[2]);
}

template <int Dimension = 3>
inline bool same_point(const double* a, const double* b) noexcept {
  static_assert(Dimension == 2 || Dimension == 3);
  if constexpr (Dimension == 2) {
    return a[0] == b[0] && a[1] == b[1];
  } else {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
  }
}

inline std::array<double, 3> cross(const std::array<double, 3>& a,
                                  const std::array<double, 3>& b) noexcept {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

inline std::array<double, 3> difference(const std::array<double, 3>& a,
                                        const std::array<double, 3>& b) noexcept {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline bool point_in_domain(const std::array<double, 3>& point) noexcept {
  return point_in_domain(point.data());
}

// Preserve predicate request order: yz, zx, xy by default; callers with an
// xy-first sequence pass first_axis = 0.
inline bool collinear(const double* a, const double* b, const double* c, int first_axis = 1) {
  for (int offset = 0; offset < 3; ++offset) {
    const int axis = (first_axis + offset) % 3;
    const int next = (axis + 1) % 3;
    const double pa[2] = {a[axis], a[next]};
    const double pb[2] = {b[axis], b[next]};
    const double pc[2] = {c[axis], c[next]};
    if (orient2d(pa, pb, pc) != 0) return false;
  }
  return true;
}

inline bool collinear(const std::array<double, 3>& a, const std::array<double, 3>& b,
                      const std::array<double, 3>& c, int first_axis = 1) {
  return collinear(a.data(), b.data(), c.data(), first_axis);
}

struct Vector3 {
  double x;
  double y;
  double z;
};

inline Vector3 difference(const Vector3& a, const Vector3& b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vector3 operator-(const Vector3& a, const Vector3& b) noexcept {
  return difference(a, b);
}

inline double dot(const Vector3& a, const Vector3& b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vector3 cross(const Vector3& a, const Vector3& b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double point_segment_distance(const double* p, const double* a,
                                     const double* b) noexcept {
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double length = dot(ab, ab);
  const double t = length > 0.0 ? std::clamp(dot(ap, ab) / length, 0.0, 1.0) : 0.0;
  const double q[3] = {a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]};
  return std::sqrt(squared_distance(p, q));
}

// Closest point on a triangle by Voronoi-region classification (Ericson).
inline double point_triangle_distance(const double* p, const double* a, const double* b,
                                      const double* c) noexcept {
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double d1 = dot(ab, ap);
  const double d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return std::sqrt(squared_distance(p, a));
  const double bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return std::sqrt(squared_distance(p, b));
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return point_segment_distance(p, a, b);
  const double cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return std::sqrt(squared_distance(p, c));
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return point_segment_distance(p, a, c);
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    return point_segment_distance(p, b, c);
  }
  const double denominator = va + vb + vc;
  if (!(denominator > 0.0)) {
    return std::min({point_segment_distance(p, a, b), point_segment_distance(p, b, c),
                     point_segment_distance(p, c, a)});
  }
  const double v = vb / denominator;
  const double w = vc / denominator;
  const double q[3] = {a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w,
                       a[2] + ab[2] * v + ac[2] * w};
  return std::sqrt(squared_distance(p, q));
}

}  // namespace phx::mc::geometry
