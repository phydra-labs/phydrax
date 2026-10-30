//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Floating dihedral and normalized-volume measures used as improvement objectives.
#include "improve3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace phx::mc::geometry {
namespace {

struct Vector3 {
  double x;
  double y;
  double z;
};

Vector3 subtract(const double* first, const double* second) noexcept {
  return {first[0] - second[0], first[1] - second[1], first[2] - second[2]};
}

Vector3 cross(Vector3 first, Vector3 second) noexcept {
  return {first.y * second.z - first.z * second.y,
          first.z * second.x - first.x * second.z,
          first.x * second.y - first.y * second.x};
}

double dot(Vector3 first, Vector3 second) noexcept {
  return first.x * second.x + first.y * second.y + first.z * second.z;
}

double length(Vector3 vector) noexcept {
  return std::sqrt(dot(vector, vector));
}

}  // namespace

bool prepare_relocation_metric_cache(
    const double* const points[4], std::int32_t moved_corner,
    RelocationMetricCache& cache) noexcept {
  if (points == nullptr || moved_corner < 0 || moved_corner >= 4) return false;
  for (int corner = 0; corner < 4; ++corner) {
    if (points[corner] == nullptr) return false;
  }
  cache = RelocationMetricCache{};
  cache.moved_corner = moved_corner;

  for (int row = 0; row < 3; ++row) {
    if (moved_corner == 0 || moved_corner == row + 1) continue;
    double squared_length = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double delta = points[row + 1][axis] - points[0][axis];
      cache.fixed_shape_rows[static_cast<std::size_t>(row)]
                            [static_cast<std::size_t>(axis)] = 2.0 * delta;
      squared_length += delta * delta;
    }
    cache.fixed_shape_rows[static_cast<std::size_t>(row)][3] = squared_length;
  }
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      if (first == moved_corner || second == moved_corner) continue;
      double squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double delta = points[first][axis] - points[second][axis];
        squared += delta * delta;
      }
      cache.fixed_edge_squared[static_cast<std::size_t>(first)]
                              [static_cast<std::size_t>(second)] = squared;
    }
  }
  for (int vector = 0; vector < 3; ++vector) {
    if (moved_corner == 0 || moved_corner == vector + 1) continue;
    for (int axis = 0; axis < 3; ++axis) {
      cache.fixed_relative_vectors[static_cast<std::size_t>(vector)]
                                  [static_cast<std::size_t>(axis)] =
          points[vector + 1][axis] - points[0][axis];
    }
  }
  for (int origin = 0; origin < 4; ++origin) {
    if (origin == moved_corner) continue;
    for (int other = 0; other < 4; ++other) {
      if (other == origin || other == moved_corner) continue;
      cache.fixed_directed_edge_lengths[static_cast<std::size_t>(origin)]
                                       [static_cast<std::size_t>(other)] =
          length(subtract(points[other], points[origin]));
    }
  }

  constexpr int faces[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  for (int face = 0; face < 4; ++face) {
    const int* vertices = faces[face];
    if (vertices[0] == moved_corner || vertices[1] == moved_corner ||
        vertices[2] == moved_corner) {
      continue;
    }
    cache.opposite_face = face;
    Vector3 normal = cross(subtract(points[vertices[1]], points[vertices[0]]),
                           subtract(points[vertices[2]], points[vertices[0]]));
    const double magnitude = length(normal);
    if (magnitude > 0.0 && std::isfinite(magnitude)) {
      normal.x /= magnitude;
      normal.y /= magnitude;
      normal.z /= magnitude;
      cache.opposite_face_unit_normal = {normal.x, normal.y, normal.z};
      cache.opposite_face_valid = true;
    }
    break;
  }
  if (cache.opposite_face < 0) return false;
  cache.prepared = true;
  return true;
}


double relocation_minimum_dihedral_degrees(
    const double* const points[4], const RelocationMetricCache& cache) noexcept {
  if (!cache.prepared || points == nullptr || !cache.opposite_face_valid) return 0.0;
  constexpr int faces[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  std::array<Vector3, 4> normals{};
  for (int face = 0; face < 4; ++face) {
    if (face == cache.opposite_face) {
      normals[static_cast<std::size_t>(face)] = {
          cache.opposite_face_unit_normal[0], cache.opposite_face_unit_normal[1],
          cache.opposite_face_unit_normal[2]};
      continue;
    }
    const int* vertices = faces[face];
    normals[static_cast<std::size_t>(face)] =
        cross(subtract(points[vertices[1]], points[vertices[0]]),
              subtract(points[vertices[2]], points[vertices[0]]));
    const double magnitude = length(normals[static_cast<std::size_t>(face)]);
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) return 0.0;
    normals[static_cast<std::size_t>(face)].x /= magnitude;
    normals[static_cast<std::size_t>(face)].y /= magnitude;
    normals[static_cast<std::size_t>(face)].z /= magnitude;
  }

  double maximum_cosine = -1.0;
  constexpr double kRadiansToDegrees = 57.295779513082320876798154814105;
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      const double cosine = std::clamp(
          -dot(normals[static_cast<std::size_t>(first)],
               normals[static_cast<std::size_t>(second)]),
          -1.0, 1.0);
      maximum_cosine = std::max(maximum_cosine, cosine);
    }
  }
  const double minimum = std::acos(maximum_cosine) * kRadiansToDegrees;
  return std::isfinite(minimum) ? minimum : 0.0;
}

double relocation_relative_determinant(
    const double* const points[4], const RelocationMetricCache& cache) noexcept {
  if (!cache.prepared || points == nullptr) return 0.0;
  Vector3 vectors[3]{};
  for (int vector = 0; vector < 3; ++vector) {
    if (cache.moved_corner != 0 && cache.moved_corner != vector + 1) {
      vectors[vector] = {
          cache.fixed_relative_vectors[static_cast<std::size_t>(vector)][0],
          cache.fixed_relative_vectors[static_cast<std::size_t>(vector)][1],
          cache.fixed_relative_vectors[static_cast<std::size_t>(vector)][2]};
    } else {
      vectors[vector] = subtract(points[vector + 1], points[0]);
    }
  }
  const double determinant = std::abs(dot(vectors[0], cross(vectors[1], vectors[2])));
  double maximum_product = 0.0;
  for (int origin = 0; origin < 4; ++origin) {
    double product = 1.0;
    for (int other = 0; other < 4; ++other) {
      if (other == origin) continue;
      const double edge_length =
          origin != cache.moved_corner && other != cache.moved_corner
              ? cache.fixed_directed_edge_lengths[static_cast<std::size_t>(origin)]
                                                 [static_cast<std::size_t>(other)]
              : length(subtract(points[other], points[origin]));
      product *= edge_length;
    }
    maximum_product = std::max(maximum_product, product);
  }
  if (!(determinant > 0.0) || !(maximum_product > 0.0) ||
      !std::isfinite(determinant) || !std::isfinite(maximum_product)) {
    return 0.0;
  }
  return determinant / maximum_product;
}

double minimum_dihedral_degrees(const double* const points[4]) noexcept {
  constexpr int faces[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  std::array<Vector3, 4> normals{};
  for (int face = 0; face < 4; ++face) {
    const int* vertices = faces[face];
    normals[face] = cross(subtract(points[vertices[1]], points[vertices[0]]),
                          subtract(points[vertices[2]], points[vertices[0]]));
    const double magnitude = length(normals[face]);
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) return 0.0;
    normals[face].x /= magnitude;
    normals[face].y /= magnitude;
    normals[face].z /= magnitude;
  }

  double maximum_cosine = -1.0;
  constexpr double kRadiansToDegrees = 57.295779513082320876798154814105;
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      const double cosine = std::clamp(-dot(normals[first], normals[second]), -1.0, 1.0);
      maximum_cosine = std::max(maximum_cosine, cosine);
    }
  }
  // acos is monotone decreasing over the clamped domain; the extremal angle
  // therefore needs one transcendental evaluation instead of one per face pair.
  const double minimum = std::acos(maximum_cosine) * kRadiansToDegrees;
  return std::isfinite(minimum) ? minimum : 0.0;
}

double relative_determinant(const double* const points[4]) noexcept {
  const Vector3 a = subtract(points[1], points[0]);
  const Vector3 b = subtract(points[2], points[0]);
  const Vector3 c = subtract(points[3], points[0]);
  const double determinant = std::abs(dot(a, cross(b, c)));
  double maximum_product = 0.0;
  for (int origin = 0; origin < 4; ++origin) {
    double product = 1.0;
    for (int other = 0; other < 4; ++other) {
      if (other != origin) product *= length(subtract(points[other], points[origin]));
    }
    maximum_product = std::max(maximum_product, product);
  }
  if (!(determinant > 0.0) || !(maximum_product > 0.0) ||
      !std::isfinite(determinant) || !std::isfinite(maximum_product)) {
    return 0.0;
  }
  return determinant / maximum_product;
}

}  // namespace phx::mc::geometry
