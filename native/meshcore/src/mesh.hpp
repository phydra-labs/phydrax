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
#include <memory>
#include <numeric>
#include <span>
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
  // A stable counting sort of the representatives by round (below 62),
  // straight into the insertion order.
  const int rounds = brio_round_count(written);
  std::array<std::size_t, 64> starts{};
  for (std::size_t k = 0; k < written; ++k) {
    kept[k].round = brio_round(kept[k].index, rounds);
    ++starts[kept[k].round + 1];
  }
  std::partial_sum(starts.begin(), starts.end(), starts.begin());
  std::vector<int32_t> order(written);
  for (std::size_t k = 0; k < written; ++k) order[starts[kept[k].round]++] = kept[k].index;
  return order;
}

// Stable least-significant-digit radix sort of `count` rows of `Width`
// values (at `rows`; `scratch` holds as many) by their columns
// [first, last), first most significant, whose values lie in [0, 2^bits):
// digits of at most 11 bits, each pass a stable counting sort. Linear in
// the row count; `offsets` is working storage.
template <std::size_t Width>
void radix_sort_rows(int32_t* rows, int32_t* scratch, std::size_t count, std::size_t first,
                     std::size_t last, int bits, std::vector<std::size_t>& offsets) {
  int32_t* source = rows;
  int32_t* target = scratch;
  for (std::size_t column = last; column-- > first;) {
    for (int low = 0; low < bits; low += 11) {
      const int width = std::min(11, bits - low);
      const auto mask = (std::uint32_t{1} << width) - 1;
      const auto digit = [&](const int32_t* row) {
        return (static_cast<std::uint32_t>(row[column]) >> low) & mask;
      };
      offsets.assign((std::size_t{1} << width) + 1, 0);
      for (std::size_t index = 0; index < count; ++index) {
        ++offsets[digit(source + Width * index) + 1];
      }
      std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
      for (std::size_t index = 0; index < count; ++index) {
        const int32_t* row = source + Width * index;
        std::copy_n(row, Width, target + Width * offsets[digit(row)]++);
      }
      std::swap(source, target);
    }
  }
  if (source != rows) std::copy_n(source, Width * count, rows);
}

// Orders `count` (at most Capacity) rows of cells that share their first
// vertex by their remaining vertices (two or three, distinct as tuples):
// the rank of a row is the number of rows below it, counted without
// branches, since the order of a vertex's cells is unpredictable.
template <std::size_t Width, std::size_t Capacity>
void rank_sort_group(int32_t* rows, std::size_t count, std::size_t columns) {
  std::array<int32_t, Width * Capacity> copy;
  std::array<std::uint64_t, Capacity> high;
  std::array<std::uint32_t, Capacity> low;
  for (std::size_t index = 0; index < count; ++index) {
    const int32_t* row = rows + Width * index;
    std::copy_n(row, Width, copy.begin() + static_cast<std::ptrdiff_t>(Width * index));
    high[index] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row[1])) << 32) |
                  static_cast<std::uint32_t>(row[2]);
    low[index] = columns > 3 ? static_cast<std::uint32_t>(row[3]) : 0U;
  }
  for (std::size_t index = 0; index < count; ++index) {
    std::size_t rank = 0;
    for (std::size_t other = 0; other < count; ++other) {
      const auto below = static_cast<unsigned>(high[other] < high[index]);
      const auto tied = static_cast<unsigned>(high[other] == high[index]) &
                        static_cast<unsigned>(low[other] < low[index]);
      rank += below | tied;
    }
    std::copy_n(copy.begin() + static_cast<std::ptrdiff_t>(Width * index), Width,
                rows + Width * rank);
  }
}

// Sorts the tetrahedra lexicographically by their vertices (distinct
// tuples). A radix sort by the first vertex groups the cells by it (a few
// per vertex, at most its degree); each group is ordered by rank, or by
// radix sort when it is large, so the time is linear in the cell count.
inline void sort_tetrahedra(phx_mc_mesh& mesh) {
  constexpr std::size_t kWidth = 4;
  constexpr std::size_t kRankGroup = 64;
  const auto count = static_cast<std::size_t>(mesh.cell_count());
  // Vertices are point indices in [0, largest].
  int32_t largest = 1;
  for (int32_t vertex : mesh.cells) largest = std::max(largest, vertex);
  int bits = 1;
  while ((std::uint64_t{1} << bits) <= static_cast<std::uint64_t>(largest)) ++bits;
  int32_t* rows = mesh.cells.data();
  const std::unique_ptr<int32_t[]> scratch(new int32_t[kWidth * count]);
  std::vector<std::size_t> offsets;
  radix_sort_rows<kWidth>(rows, scratch.get(), count, 0, 1, bits, offsets);
  std::size_t begin = 0;
  while (begin < count) {
    std::size_t end = begin + 1;
    while (end < count && rows[kWidth * end] == rows[kWidth * begin]) ++end;
    const std::size_t size = end - begin;
    if (size > kRankGroup) {
      radix_sort_rows<kWidth>(rows + kWidth * begin, scratch.get(), size, 1, kWidth, bits,
                              offsets);
    } else if (size > 1) {
      rank_sort_group<kWidth, kRankGroup>(rows + kWidth * begin, size, kWidth);
    }
    begin = end;
  }
}

// Rotates every triangle to start at its smallest vertex while preserving
// orientation (cell_segments, edge k opposite vertex k, rotate with it),
// then sorts the triangles lexicographically: a counting sort of the cell
// ids by their first vertex, the few cells of each first vertex ordered by
// their other two vertices (distinct pairs), then one gather of the rows.
inline void canonicalize_triangles(phx_mc_mesh& mesh) {
  const auto count = static_cast<std::size_t>(mesh.cell_count());
  const bool has_segments = !mesh.cell_segments.empty();
  int32_t* cells = mesh.cells.data();
  int32_t* segments = mesh.cell_segments.data();
  int32_t largest = 0;
  for (std::size_t k = 0; k < 3 * count; ++k) largest = std::max(largest, cells[k]);
  // Group bounds fit 32 bits: there are at most 2 kMaxMeshPoints triangles.
  std::vector<std::uint32_t> offsets(static_cast<std::size_t>(largest) + 2, 0);
  // The rotation that brings the smallest of three distinct values first,
  // by selects on loaded values (its place in a cell is unpredictable).
  const auto rotate = [](int32_t* v, bool second, bool third) {
    const int32_t a = v[0], b = v[1], c = v[2];
    v[0] = second ? b : third ? c : a;
    v[1] = second ? c : third ? a : b;
    v[2] = second ? a : third ? b : c;
  };
  for (std::size_t cell = 0; cell < count; ++cell) {
    int32_t* v = cells + 3 * cell;
    const bool second = (v[1] < v[0]) & (v[1] < v[2]);
    const bool third = (v[2] < v[0]) & (v[2] < v[1]);
    rotate(v, second, third);
    if (has_segments) rotate(segments + 3 * cell, second, third);
    ++offsets[static_cast<std::size_t>(v[0]) + 1];
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  std::vector<std::uint32_t> order(count);
  for (std::size_t cell = 0; cell < count; ++cell) {
    order[offsets[static_cast<std::size_t>(cells[3 * cell])]++] = static_cast<std::uint32_t>(cell);
  }
  const auto rest = [cells](std::uint32_t cell) {
    const int32_t* v = cells + 3 * static_cast<std::size_t>(cell);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(v[1])) << 32) |
           static_cast<std::uint32_t>(v[2]);
  };
  constexpr std::size_t kInsertionGroup = 32;
  // offsets[v] now ends the group of first vertex v.
  std::size_t begin = 0;
  for (std::size_t vertex = 0; vertex <= static_cast<std::size_t>(largest); ++vertex) {
    const std::size_t end = offsets[vertex];
    if (end - begin > kInsertionGroup) {
      std::sort(order.begin() + static_cast<std::ptrdiff_t>(begin),
                order.begin() + static_cast<std::ptrdiff_t>(end),
                [&](std::uint32_t left, std::uint32_t right) { return rest(left) < rest(right); });
    } else {
      for (std::size_t at = begin + 1; at < end; ++at) {
        const std::uint32_t cell = order[at];
        const std::uint64_t key = rest(cell);
        std::size_t to = at;
        for (; to > begin && rest(order[to - 1]) > key; --to) order[to] = order[to - 1];
        order[to] = cell;
      }
    }
    begin = end;
  }
  const auto gather = [&](std::vector<int32_t>& rows) {
    std::vector<int32_t> sorted(rows.size());
    for (std::size_t at = 0; at < count; ++at) {
      std::copy_n(rows.data() + 3 * static_cast<std::size_t>(order[at]), 3, sorted.data() + 3 * at);
    }
    rows.swap(sorted);
  };
  gather(mesh.cells);
  if (has_segments) gather(mesh.cell_segments);
}

// Rotates every cell to start at its smallest vertex while preserving
// orientation, then sorts cells lexicographically (cell_segments follow their
// cells when present, with edge k opposite vertex k).
inline void canonicalize_cells(phx_mc_mesh& mesh) {
  if (mesh.dimension == 2) {
    canonicalize_triangles(mesh);
    return;
  }
  const int64_t count = mesh.cell_count();
  // The position of the smallest of `count` distinct values, by selects (its
  // place in a cell is unpredictable).
  const auto smallest = [](const int32_t* values, int size) {
    int position = 0;
    for (int k = 1; k < size; ++k) position = values[k] < values[position] ? k : position;
    return position;
  };
  for (int64_t cell = 0; cell < count; ++cell) {
    int32_t* v = mesh.cells.data() + cell * 4;
    // Even permutations of a tetrahedron: the double transposition that
    // brings the minimum to slot 0, then the rotation of the remaining
    // three that brings their minimum to slot 1.
    static constexpr int kLead[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
    const int* lead = kLead[smallest(v, 4)];
    const int32_t w[4] = {v[lead[0]], v[lead[1]], v[lead[2]], v[lead[3]]};
    const int second = smallest(w + 1, 3);
    v[0] = w[0];
    v[1] = w[1 + second];
    v[2] = w[1 + (second + 1) % 3];
    v[3] = w[1 + (second + 2) % 3];
  }
  sort_tetrahedra(mesh);
}

// Point-set triangulations behind phx_mc_delaunay_2d/3d, phx_mc_regular_2d/3d
// and phx_mc_triangulate; `canonical` applies canonicalize_cells.
int32_t triangulate_2d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, bool canonical, phx_mc_mesh** mesh);
// `parallelism` workers insert in rounds when `canonical` (the cell order of a
// non-canonical result is the sequential construction's, built on one).
int32_t triangulate_3d(int64_t point_count, const double* points, const double* weights,
                       bool weighted, bool canonical, phx_mc_mesh** mesh,
                       int32_t parallelism);

}  // namespace phx::mc
