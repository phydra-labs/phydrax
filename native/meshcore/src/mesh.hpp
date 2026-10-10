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
#include "bounded_memory.hpp"

struct phx_mc_mesh : phx::mc::NativeAllocatedObject {
  int32_t dimension = 0;
  int64_t input_point_count = 0;
  phx::mc::NativeVector<double> points;            // point_count x dimension
  phx::mc::NativeVector<int32_t> cells;            // cell_count x (dimension + 1)
  phx::mc::NativeVector<int32_t> vertex_map;       // input_point_count
  phx::mc::NativeVector<int32_t> cell_constraints;  // cell_count x (dimension + 1), or empty
  phx::mc::NativeVector<int32_t> cell_regions;      // cell_count, or empty

  int64_t point_count() const { return static_cast<int64_t>(points.size()) / dimension; }
  int64_t cell_count() const { return static_cast<int64_t>(cells.size()) / (dimension + 1); }
};

namespace phx::mc {

inline constexpr int64_t kMaxMeshPoints = 2147483646;

// Validates finiteness and the exact domain of coordinates (and weights).
inline int32_t validate_points(const double* points, int64_t count, int dimension,
                               const double* weights, void (*charge)(void*) = nullptr,
                               void* context = nullptr) {
  for (int64_t index = 0; index < count * dimension; ++index) {
    charge_preparation_visit(charge, context);
    if (!std::isfinite(points[index])) {
      return PHX_MC_NONFINITE_INPUT;
    }
  }
  if (weights != nullptr) {
    for (int64_t index = 0; index < count; ++index) {
      charge_preparation_visit(charge, context);
      if (!std::isfinite(weights[index])) {
        return PHX_MC_NONFINITE_INPUT;
      }
    }
  }
  for (int64_t index = 0; index < count * dimension; ++index) {
    charge_preparation_visit(charge, context);
    if (!coordinate_in_domain(points[index])) {
      return PHX_MC_RANGE_ERROR;
    }
  }
  if (weights != nullptr) {
    for (int64_t index = 0; index < count; ++index) {
      charge_preparation_visit(charge, context);
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
template <class Allocator>
inline NativeVector<int32_t> deduplicate_points(
    const double* points, int64_t count, int dimension, const double* weights,
    std::vector<int32_t, Allocator>& vertex_map, void (*charge)(void*) = nullptr,
    void* context = nullptr) {
  NativeVector<int32_t> order(static_cast<std::size_t>(count));
  int32_t initial = 0;
  for (int32_t& vertex : order) {
    charge_preparation_visit(charge, context);
    vertex = initial++;
  }
  auto less = [&](int32_t left, int32_t right) {
    charge_preparation_visit(charge, context);
    for (int axis = 0; axis < dimension; ++axis) {
      charge_preparation_visit(charge, context);
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
  NativeVector<int32_t> representatives;
  std::size_t start = 0;
  while (start < order.size()) {
    charge_preparation_visit(charge, context);
    std::size_t stop = start + 1;
    auto same = [&](int32_t left, int32_t right) {
      charge_preparation_visit(charge, context);
      for (int axis = 0; axis < dimension; ++axis) {
        charge_preparation_visit(charge, context);
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
        charge_preparation_visit(charge, context);
        const int32_t candidate = order[k];
        if (weights[candidate] > weights[winner] ||
            (weights[candidate] == weights[winner] && candidate < winner)) {
          winner = candidate;
        }
      }
    }
    for (std::size_t k = start; k < stop; ++k) {
      charge_preparation_visit(charge, context);
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
  std::sort(representatives.begin(), representatives.end(), [&](int32_t left, int32_t right) {
    charge_preparation_visit(charge, context);
    return left < right;
  });
  return representatives;
}

// Orientation-preserving permutation of one cell that brings its smallest
// vertex first: a rotation of a triangle; for a tetrahedron a double
// transposition bringing the minimum to slot 0, then a rotation of the other
// three.  perm[k] receives the original position of new position k.
inline void canonical_permutation(const int32_t* v, int width, int* perm) {
  if (width == 3) {
    const int first = static_cast<int>(std::min_element(v, v + 3) - v);
    for (int k = 0; k < 3; ++k) {
      perm[k] = (first + k) % 3;
    }
    return;
  }
  int32_t w[4] = {v[0], v[1], v[2], v[3]};
  for (int k = 0; k < 4; ++k) {
    perm[k] = k;
  }
  const int first = static_cast<int>(std::min_element(w, w + 4) - w);
  if (first != 0) {
    const int other = first == 1 ? 2 : 1;
    const int third = 6 - first - other;
    std::swap(w[0], w[first]);
    std::swap(perm[0], perm[first]);
    std::swap(w[other], w[third]);
    std::swap(perm[other], perm[third]);
  }
  const int second = 1 + static_cast<int>(std::min_element(w + 1, w + 4) - (w + 1));
  std::rotate(perm + 1, perm + second, perm + 4);
}

// Applies canonical_permutation to one cell and its per-vertex data (the
// facet or edge opposite vertex k follows vertex k).
inline void canonicalize_cell(int32_t* v, int32_t* opposite, int width) {
  int perm[4];
  canonical_permutation(v, width, perm);
  int32_t vertices[4];
  int32_t labels[4];
  for (int k = 0; k < width; ++k) {
    vertices[k] = v[perm[k]];
    labels[k] = opposite == nullptr ? 0 : opposite[perm[k]];
  }
  std::copy_n(vertices, width, v);
  if (opposite != nullptr) {
    std::copy_n(labels, width, opposite);
  }
}

// Rotates every cell to start at its smallest vertex while preserving
// orientation, then sorts cells lexicographically; cell_constraints and
// cell_regions follow their cells when present.
inline void canonicalize_cells(phx_mc_mesh& mesh, void (*charge)(void*) = nullptr,
                                void* context = nullptr) {
  const int width = mesh.dimension + 1;
  const int64_t count = mesh.cell_count();
  const bool has_constraints = !mesh.cell_constraints.empty();
  const bool has_regions = !mesh.cell_regions.empty();
  for (int64_t cell = 0; cell < count; ++cell) {
    charge_preparation_visit(charge, context);
    canonicalize_cell(mesh.cells.data() + cell * width,
                      has_constraints ? mesh.cell_constraints.data() + cell * width : nullptr,
                      width);
  }
  NativeVector<int64_t> order(static_cast<std::size_t>(count));
  int64_t initial = 0;
  for (int64_t& cell : order) {
    charge_preparation_visit(charge, context);
    cell = initial++;
  }
  std::sort(order.begin(), order.end(), [&](int64_t left, int64_t right) {
    charge_preparation_visit(charge, context);
    return std::lexicographical_compare(mesh.cells.begin() + left * width,
                                        mesh.cells.begin() + (left + 1) * width,
                                        mesh.cells.begin() + right * width,
                                        mesh.cells.begin() + (right + 1) * width);
  });
  NativeVector<int32_t> sorted_cells(mesh.cells.size());
  NativeVector<int32_t> sorted_constraints(mesh.cell_constraints.size());
  NativeVector<int32_t> sorted_regions(mesh.cell_regions.size());
  for (int64_t k = 0; k < count; ++k) {
    charge_preparation_visit(charge, context);
    const int64_t source = order[static_cast<std::size_t>(k)];
    std::copy_n(mesh.cells.begin() + source * width, width, sorted_cells.begin() + k * width);
    if (has_constraints) {
      std::copy_n(mesh.cell_constraints.begin() + source * width, width,
                  sorted_constraints.begin() + k * width);
    }
    if (has_regions) {
      sorted_regions[static_cast<std::size_t>(k)] =
          mesh.cell_regions[static_cast<std::size_t>(source)];
    }
  }
  mesh.cells.swap(sorted_cells);
  mesh.cell_constraints.swap(sorted_constraints);
  mesh.cell_regions.swap(sorted_regions);
}

}  // namespace phx::mc
