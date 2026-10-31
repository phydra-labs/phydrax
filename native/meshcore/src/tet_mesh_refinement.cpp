//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Floating quality measurements used only to prioritize/refuse refinement.
#include "improve3d.hpp"
#include "refine3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace phx::mc::geometry {

namespace {

template <bool WithRelocationCache>
bool tetrahedron_shape_impl(const double* const points[4],
                            const RelocationMetricCache* cache,
                            TetrahedronShape& shape) noexcept {
  if constexpr (WithRelocationCache) {
    if (cache == nullptr || !cache->prepared) return false;
  }
  double matrix[3][4]{};
  for (int row = 0; row < 3; ++row) {
    if constexpr (WithRelocationCache) {
      if (cache->moved_corner != 0 && cache->moved_corner != row + 1) {
        std::copy(cache->fixed_shape_rows[static_cast<std::size_t>(row)].begin(),
                  cache->fixed_shape_rows[static_cast<std::size_t>(row)].end(),
                  matrix[row]);
        continue;
      }
    }
    double squared_length = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double delta = points[row + 1][axis] - points[0][axis];
      matrix[row][axis] = 2.0 * delta;
      squared_length += delta * delta;
    }
    matrix[row][3] = squared_length;
  }

  // Pivoting/elimination are always recomputed; only fixed initial rows are reused.
  for (int column = 0; column < 3; ++column) {
    int pivot = column;
    for (int row = column + 1; row < 3; ++row) {
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
    }
    if (matrix[pivot][column] == 0.0 || !std::isfinite(matrix[pivot][column])) return false;
    if (pivot != column) {
      for (int entry = column; entry < 4; ++entry) std::swap(matrix[column][entry], matrix[pivot][entry]);
    }
    const double diagonal = matrix[column][column];
    for (int entry = column; entry < 4; ++entry) matrix[column][entry] /= diagonal;
    for (int row = 0; row < 3; ++row) {
      if (row == column) continue;
      const double factor = matrix[row][column];
      for (int entry = column; entry < 4; ++entry) {
        matrix[row][entry] -= factor * matrix[column][entry];
      }
    }
  }

  double center[3];
  for (int axis = 0; axis < 3; ++axis) {
    center[axis] = points[0][axis] + matrix[axis][3];
    if (!std::isfinite(center[axis])) return false;
  }
  double radius_squared = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double delta = center[axis] - points[0][axis];
    radius_squared += delta * delta;
  }
  if (!(radius_squared > 0.0) || !std::isfinite(radius_squared)) return false;

  double shortest_squared = std::numeric_limits<double>::infinity();
  for (int first = 0; first < 4; ++first) {
    for (int second = first + 1; second < 4; ++second) {
      double squared = 0.0;
      if constexpr (WithRelocationCache) {
        if (first != cache->moved_corner && second != cache->moved_corner) {
          squared = cache->fixed_edge_squared[static_cast<std::size_t>(first)]
                                              [static_cast<std::size_t>(second)];
        } else {
          for (int axis = 0; axis < 3; ++axis) {
            const double delta = points[first][axis] - points[second][axis];
            squared += delta * delta;
          }
        }
      } else {
        for (int axis = 0; axis < 3; ++axis) {
          const double delta = points[first][axis] - points[second][axis];
          squared += delta * delta;
        }
      }
      shortest_squared = std::min(shortest_squared, squared);
    }
  }
  if (!(shortest_squared > 0.0) || !std::isfinite(shortest_squared)) return false;
  std::copy_n(center, 3, shape.circumcenter.begin());
  shape.circumradius = std::sqrt(radius_squared);
  shape.shortest_edge = std::sqrt(shortest_squared);
  shape.radius_edge_ratio = shape.circumradius / shape.shortest_edge;
  return std::isfinite(shape.circumradius) && std::isfinite(shape.shortest_edge) &&
         std::isfinite(shape.radius_edge_ratio);
}

}  // namespace

bool tetrahedron_shape(const double* const points[4], TetrahedronShape& shape) noexcept {
  return tetrahedron_shape_impl<false>(points, nullptr, shape);
}

bool relocation_tetrahedron_shape(const double* const points[4],
                                  const RelocationMetricCache& cache,
                                  TetrahedronShape& shape) noexcept {
  return tetrahedron_shape_impl<true>(points, &cache, shape);
}

}  // namespace phx::mc::geometry
