// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Deterministic topology keys and index union-find with explicit visit charging.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace phx::mc::topology {

template <class Index>
inline std::array<Index, 2> undirected_edge_vertices(Index a, Index b) {
  return {std::min(a, b), std::max(a, b)};
}

// Raw nonnegative vertex ids packed in lexicographic endpoint order.
inline std::uint64_t undirected_edge(std::int32_t a, std::int32_t b) {
  const auto vertices = undirected_edge_vertices(a, b);
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(vertices[0])) << 32) |
         static_cast<std::uint32_t>(vertices[1]);
}

// Cavity keys reserve zero for the ghost vertex (-1), unlike raw-id keys.
inline std::uint64_t ghost_undirected_edge(std::int32_t a, std::int32_t b) {
  const auto vertices = undirected_edge_vertices(a, b);
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(vertices[0] + 1)) << 32) |
         static_cast<std::uint32_t>(vertices[1] + 1);
}

// For a positive tetrahedron, orient3d(face[k], vertex[k]) is positive.
// The outward loop reverses the last two entries.
inline constexpr int kTetFaceOrder[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};

// Halve the path in the same read/write order as the source algorithms.
// The optional charge owns each non-root visit, not each parent assignment.
template <class Container, class Index, class Charge = std::nullptr_t>
inline Index find_root(Container& parent, Index index, Charge charge = nullptr) {
  while (parent[static_cast<std::size_t>(index)] != index) {
    if constexpr (!std::is_same_v<Charge, std::nullptr_t>) charge();
    parent[static_cast<std::size_t>(index)] =
        parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(index)])];
    index = parent[static_cast<std::size_t>(index)];
  }
  return index;
}

}  // namespace phx::mc::topology
