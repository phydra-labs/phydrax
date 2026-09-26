//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned triangulation result behind the opaque phx_mc_mesh handle, plus the
// shared deterministic preparation (validation, deduplication) and canonical
// cell ordering used by every triangulation entry point.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "phydrax_meshcore.h"
#include "predicates.hpp"

struct phx_mc_mesh {
  int32_t dimension = 0;
  int64_t input_point_count = 0;
  std::vector<double> points;          // point_count x dimension
  std::vector<int32_t> cells;          // cell_count x (dimension + 1)
  std::vector<int32_t> vertex_map;     // input_point_count
  std::vector<int32_t> cell_segments;  // cell_count x 3 (CDT) or empty

  int64_t point_count() const { return static_cast<int64_t>(points.size()) / dimension; }
  int64_t cell_count() const { return static_cast<int64_t>(cells.size()) / (dimension + 1); }
};

namespace phx::mc {

inline constexpr int64_t kMaxMeshPoints = 2147483646;

// Validates finiteness and the exact domain of coordinates (and weights).
inline int32_t validate_points(const double* points, int64_t count, int dimension,
                               const double* weights) {
  for (int64_t index = 0; index < count * dimension; ++index) {
    if (!std::isfinite(points[index])) {
      return PHX_MC_NONFINITE_INPUT;
    }
  }
  if (weights != nullptr) {
    for (int64_t index = 0; index < count; ++index) {
      if (!std::isfinite(weights[index])) {
        return PHX_MC_NONFINITE_INPUT;
      }
    }
  }
  for (int64_t index = 0; index < count * dimension; ++index) {
    if (!coordinate_in_domain(points[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  if (weights != nullptr) {
    for (int64_t index = 0; index < count; ++index) {
      if (!weight_in_domain(weights[index])) {
        return PHX_MC_RANGE_ERROR;
      }
    }
  }
  return PHX_MC_OK;
}

// Exact deduplication.  Identical points map to the smallest index; for
// weighted points the largest weight wins (ties: smallest index), strictly
// lighter coincident points map to -1 (redundant), equal-weight coincident
// points map to the representative.  Returns the representatives in ascending
// index order.
inline std::vector<int32_t> deduplicate_points(const double* points, int64_t count, int dimension,
                                               const double* weights,
                                               std::vector<int32_t>& vertex_map) {
  std::vector<int32_t> order(static_cast<std::size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  auto less = [&](int32_t left, int32_t right) {
    for (int axis = 0; axis < dimension; ++axis) {
      const double a = points[static_cast<int64_t>(left) * dimension + axis];
      const double b = points[static_cast<int64_t>(right) * dimension + axis];
      if (a != b) {
        return a < b;
      }
    }
    return left < right;
  };
  std::sort(order.begin(), order.end(), less);
  vertex_map.assign(static_cast<std::size_t>(count), -1);
  std::vector<int32_t> representatives;
  std::size_t start = 0;
  while (start < order.size()) {
    std::size_t stop = start + 1;
    auto same = [&](int32_t left, int32_t right) {
      for (int axis = 0; axis < dimension; ++axis) {
        if (points[static_cast<int64_t>(left) * dimension + axis] !=
            points[static_cast<int64_t>(right) * dimension + axis]) {
          return false;
        }
      }
      return true;
    };
    while (stop < order.size() && same(order[start], order[stop])) {
      ++stop;
    }
    int32_t winner = order[start];
    if (weights != nullptr) {
      for (std::size_t k = start + 1; k < stop; ++k) {
        const int32_t candidate = order[k];
        if (weights[candidate] > weights[winner] ||
            (weights[candidate] == weights[winner] && candidate < winner)) {
          winner = candidate;
        }
      }
    }
    for (std::size_t k = start; k < stop; ++k) {
      const int32_t member = order[k];
      if (weights == nullptr || weights[member] == weights[winner]) {
        vertex_map[static_cast<std::size_t>(member)] = winner;
      } else {
        vertex_map[static_cast<std::size_t>(member)] = -1;
      }
    }
    representatives.push_back(winner);
    start = stop;
  }
  std::sort(representatives.begin(), representatives.end());
  return representatives;
}

// Rotates every cell to start at its smallest vertex while preserving
// orientation, then sorts cells lexicographically (cell_segments follow their
// cells when present, with edge k opposite vertex k).
inline void canonicalize_cells(phx_mc_mesh& mesh) {
  const int width = mesh.dimension + 1;
  const int64_t count = mesh.cell_count();
  const bool has_segments = !mesh.cell_segments.empty();
  for (int64_t cell = 0; cell < count; ++cell) {
    int32_t* v = mesh.cells.data() + cell * width;
    int32_t* s = has_segments ? mesh.cell_segments.data() + cell * 3 : nullptr;
    if (width == 3) {
      const int first = static_cast<int>(std::min_element(v, v + 3) - v);
      std::rotate(v, v + first, v + 3);
      if (s != nullptr) {
        std::rotate(s, s + first, s + 3);
      }
    } else {
      // Even permutations of a tetrahedron: bring the minimum to slot 0 with a
      // double transposition, then rotate the remaining three.
      const int first = static_cast<int>(std::min_element(v, v + 4) - v);
      if (first != 0) {
        const int other = first == 1 ? 2 : 1;
        const int third = 6 - first - other;
        std::swap(v[0], v[first]);
        std::swap(v[other], v[third]);
      }
      const int second = 1 + static_cast<int>(std::min_element(v + 1, v + 4) - (v + 1));
      std::rotate(v + 1, v + second, v + 4);
    }
  }
  std::vector<int64_t> order(static_cast<std::size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int64_t left, int64_t right) {
    return std::lexicographical_compare(mesh.cells.begin() + left * width,
                                        mesh.cells.begin() + (left + 1) * width,
                                        mesh.cells.begin() + right * width,
                                        mesh.cells.begin() + (right + 1) * width);
  });
  std::vector<int32_t> sorted_cells(mesh.cells.size());
  std::vector<int32_t> sorted_segments(mesh.cell_segments.size());
  for (int64_t k = 0; k < count; ++k) {
    std::copy_n(mesh.cells.begin() + order[static_cast<std::size_t>(k)] * width, width,
                sorted_cells.begin() + k * width);
    if (has_segments) {
      std::copy_n(mesh.cell_segments.begin() + order[static_cast<std::size_t>(k)] * 3, 3,
                  sorted_segments.begin() + k * 3);
    }
  }
  mesh.cells.swap(sorted_cells);
  mesh.cell_segments.swap(sorted_segments);
}

}  // namespace phx::mc
