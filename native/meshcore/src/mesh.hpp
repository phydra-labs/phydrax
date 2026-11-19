//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned triangulation result behind the opaque phx_mc_mesh handle, plus the
// shared deterministic preparation (validation, deduplication) and canonical
// cell ordering used by every triangulation entry point.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "spatial_sort.hpp"

struct phx_mc_mesh {
  int32_t dimension = 0;
  int64_t input_point_count = 0;
  std::vector<double> points;          // point_count x dimension
  std::vector<int32_t> cells;          // cell_count x (dimension + 1)
  std::vector<int32_t> vertex_map;     // input_point_count
  std::vector<int32_t> cell_segments;  // cell_count x 3 (CDT) or empty
  std::vector<int32_t> constraint_sources;  // input segment per constrained Steiner point
  std::vector<int32_t> constraint_parents;  // two endpoints of the split parent edge per point
  std::vector<double> constraint_ratios;    // split parameter from constraint_parents[2 * i]

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

// deduplicate_points and brio_hilbert_order of the representatives from one
// radix sort of Hilbert keys: identical points share their key, so every
// class of identical points lies in one run of equal keys (resolved there by
// the rules of deduplicate_points). Returns the insertion order; the
// bounding box, keys and rounds are those of the representatives, so the
// order equals brio_hilbert_order(deduplicate_points(...)).
inline std::vector<int32_t> deduplicated_insertion_order(const double* points, int64_t count,
                                                         int dimension, const double* weights,
                                                         std::vector<int32_t>& vertex_map) {
  vertex_map.assign(static_cast<std::size_t>(count), -1);
  std::vector<int32_t> all(static_cast<std::size_t>(count));
  std::iota(all.begin(), all.end(), 0);
  std::vector<SpatialKey> keys, kept;
  hilbert_keys(points, dimension, all, keys);
  radix_sort_keys(keys, kept, true, false);
  const auto coordinate = [&](int32_t point, int axis) {
    return points[static_cast<int64_t>(point) * dimension + axis];
  };
  const auto less = [&](const SpatialKey& left, const SpatialKey& right) {
    for (int axis = 0; axis < dimension; ++axis) {
      const double a = coordinate(left.index, axis), b = coordinate(right.index, axis);
      if (a != b) return a < b;
    }
    return left.index < right.index;
  };
  const auto same = [&](int32_t left, int32_t right) {
    for (int axis = 0; axis < dimension; ++axis) {
      if (coordinate(left, axis) != coordinate(right, axis)) return false;
    }
    return true;
  };
  kept.resize(keys.size());
  std::size_t written = 0;
  std::size_t start = 0;
  while (start < keys.size()) {
    std::size_t stop = start + 1;
    while (stop < keys.size() && keys[stop].key == keys[start].key) ++stop;
    if (stop == start + 1) {
      vertex_map[static_cast<std::size_t>(keys[start].index)] = keys[start].index;
      kept[written++] = keys[start];
      start = stop;
      continue;
    }
    // Distinct points of one key cell, and identical ones, in index order.
    std::sort(keys.begin() + static_cast<std::ptrdiff_t>(start),
              keys.begin() + static_cast<std::ptrdiff_t>(stop), less);
    const std::size_t run_written = written;
    std::size_t first = start;
    while (first < stop) {
      std::size_t last = first + 1;
      while (last < stop && same(keys[first].index, keys[last].index)) ++last;
      std::size_t winner = first;
      if (weights != nullptr) {
        for (std::size_t k = first + 1; k < last; ++k) {
          if (weights[keys[k].index] > weights[keys[winner].index]) winner = k;
        }
      }
      const int32_t representative = keys[winner].index;
      for (std::size_t k = first; k < last; ++k) {
        const int32_t member = keys[k].index;
        vertex_map[static_cast<std::size_t>(member)] =
            weights == nullptr || weights[member] == weights[representative] ? representative
                                                                             : -1;
      }
      kept[written++] = keys[winner];
      first = last;
    }
    std::sort(kept.begin() + static_cast<std::ptrdiff_t>(run_written),
              kept.begin() + static_cast<std::ptrdiff_t>(written),
              [](const SpatialKey& left, const SpatialKey& right) {
                return left.index < right.index;
              });
    start = stop;
  }
  kept.resize(written);
  const int rounds = brio_round_count(written);
  for (SpatialKey& entry : kept) entry.round = brio_round(entry.index, rounds);
  radix_sort_keys(kept, keys, false, true);
  std::vector<int32_t> order;
  order.reserve(kept.size());
  for (const SpatialKey& entry : kept) order.push_back(entry.index);
  return order;
}

// Sorts the cells (with their segment rows, as `Width` values per record)
// lexicographically by value; vertex tuples are distinct, so the order is
// that of the vertices. Counting sort by the first vertex (a point index),
// then a sort of each bucket (the cells whose smallest vertex it is).
template <std::size_t Width>
void sort_cell_records(phx_mc_mesh& mesh) {
  const int cell_width = mesh.dimension + 1;
  const std::size_t count = static_cast<std::size_t>(mesh.cell_count());
  const bool has_segments = Width > static_cast<std::size_t>(cell_width);
  std::vector<std::size_t> offsets(static_cast<std::size_t>(mesh.point_count()) + 1, 0);
  for (std::size_t cell = 0; cell < count; ++cell) {
    ++offsets[static_cast<std::size_t>(mesh.cells[cell * cell_width]) + 1];
  }
  for (std::size_t vertex = 1; vertex < offsets.size(); ++vertex) {
    offsets[vertex] += offsets[vertex - 1];
  }
  std::vector<std::array<int32_t, Width>> records(count);
  for (std::size_t cell = 0; cell < count; ++cell) {
    auto& record = records[offsets[static_cast<std::size_t>(mesh.cells[cell * cell_width])]++];
    std::copy_n(mesh.cells.begin() + static_cast<std::ptrdiff_t>(cell * cell_width), cell_width,
                record.begin());
    if (has_segments) {
      std::copy_n(mesh.cell_segments.begin() + static_cast<std::ptrdiff_t>(cell * 3), 3,
                  record.begin() + cell_width);
    }
  }
  // offsets[v] is now the end of bucket v, the start of bucket v + 1.
  std::size_t begin = 0;
  for (std::size_t vertex = 0; vertex + 1 < offsets.size(); ++vertex) {
    const std::size_t end = offsets[vertex];
    if (end - begin > 1) {
      std::sort(records.begin() + static_cast<std::ptrdiff_t>(begin),
                records.begin() + static_cast<std::ptrdiff_t>(end));
    }
    begin = end;
  }
  for (std::size_t cell = 0; cell < count; ++cell) {
    std::copy_n(records[cell].begin(), cell_width,
                mesh.cells.begin() + static_cast<std::ptrdiff_t>(cell * cell_width));
    if (has_segments) {
      std::copy_n(records[cell].begin() + cell_width, 3,
                  mesh.cell_segments.begin() + static_cast<std::ptrdiff_t>(cell * 3));
    }
  }
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
  if (width == 4) {
    sort_cell_records<4>(mesh);
  } else if (has_segments) {
    sort_cell_records<6>(mesh);
  } else {
    sort_cell_records<3>(mesh);
  }
}

// Point-set triangulations behind phx_mc_delaunay_2d/3d, phx_mc_regular_2d/3d
// and phx_mc_triangulate; `canonical` applies canonicalize_cells.
int32_t triangulate_2d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, bool canonical, phx_mc_mesh** mesh);
int32_t triangulate_3d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, bool canonical, phx_mc_mesh** mesh);

}  // namespace phx::mc
