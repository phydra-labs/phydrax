// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Binary64 geometry primitives. Three-term sums retain left-to-right order;
// compensated and expansion kernels keep their own arithmetic.
#pragma once

#include <array>

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

}  // namespace phx::mc::geometry
