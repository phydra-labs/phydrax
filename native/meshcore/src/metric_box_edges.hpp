// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace phx::mc::metric {

// Enumerate by lower endpoint and then higher endpoint. Each source vertex
// visits its incident tetrahedra, not a globally sorted bank of 6*C pairs.
// The caller owns the fixed site/cell capacity bound.
template <class Emit>
bool visit_box_edges(std::size_t vertices, std::span<const std::array<std::int32_t, 4>> cells,
                     Emit emit) {
  std::vector<std::uint32_t> offsets(vertices + 1, 0);
  for (const auto& c : cells) {
    for (int k = 0; k < 4; ++k) {
      if (c[k] < 0 || static_cast<std::size_t>(c[k]) >= vertices) return false;
      for (int j = 0; j < k; ++j) if (c[j] == c[k]) return false;
      ++offsets[static_cast<std::size_t>(c[k]) + 1];
    }
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  std::vector<std::uint32_t> incidence(offsets.back());
  {
    auto cursor = offsets;
    for (std::uint32_t row = 0; row < cells.size(); ++row) {
      for (auto v : cells[row]) incidence[cursor[v]++] = row;
    }
  }
  std::vector<std::int32_t> visited(vertices, -1), adjacent;
  for (std::int32_t a = 0; static_cast<std::size_t>(a) < vertices; ++a) {
    adjacent.clear();
    for (auto k = offsets[a]; k < offsets[a + 1]; ++k) {
      for (const auto b : cells[incidence[k]]) {
        if (a >= b || visited[b] == a) continue;
        visited[b] = a;
        adjacent.push_back(b);
      }
    }
    std::sort(adjacent.begin(), adjacent.end());
    for (auto b : adjacent) emit(std::array{a, b});
  }
  return true;
}

}  // namespace phx::mc::metric
