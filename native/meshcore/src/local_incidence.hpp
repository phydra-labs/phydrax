//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Per-improvement-call vertex/cell and exact carrier-key incidence.
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tet_mesh.hpp"

namespace phx::mc::detail {

using LocalCell = std::array<std::int32_t, 4>;
using LocalEdgeKey = std::pair<std::int32_t, std::int32_t>;

LocalEdgeKey sorted_local_edge(std::int32_t first, std::int32_t second) noexcept;

struct LocalEdgeHash {
  std::size_t operator()(const LocalEdgeKey& edge) const noexcept;
};

struct LocalPositionKey {
  std::array<std::uint64_t, 3> bits{};

  bool operator==(const LocalPositionKey&) const noexcept = default;
};

struct LocalPositionHash {
  std::size_t operator()(const LocalPositionKey& point) const noexcept;
};

LocalPositionKey local_position_key(const double* point) noexcept;

class LocalIncidence {
 public:
  using PositionMap = std::unordered_map<LocalPositionKey, std::int32_t,
                                          LocalPositionHash>;

  enum class PositionStatus : std::uint8_t {
    kReady,
    kDuplicate,
    kIndexMismatch,
  };

  struct PositionMove {
    LocalPositionKey old_key{};
    LocalPositionKey new_key{};
    std::int32_t vertex_slot = -1;
    bool unchanged = false;
    PositionMap::node_type staged_node;
  };

  class Star {
   public:
    std::size_t size() const noexcept { return cells_.size(); }
    std::int32_t operator[](std::size_t index) const noexcept {
      return cells_[index];
    }
    bool empty() const noexcept { return cells_.empty(); }
    std::span<const std::int32_t> cells() const noexcept { return cells_; }
    auto begin() const noexcept { return cells_.begin(); }
    auto end() const noexcept { return cells_.end(); }

   private:
    friend class LocalIncidence;
    Star(const LocalIncidence* owner, std::int32_t vertex,
         std::uint64_t revision, std::vector<std::int32_t>&& cells)
        : owner_(owner), vertex_(vertex), revision_(revision),
          cells_(std::move(cells)) {}

    const LocalIncidence* owner_ = nullptr;
    std::int32_t vertex_ = -1;
    std::uint64_t revision_ = 0;
    std::vector<std::int32_t> cells_;
  };

  bool initialize(const TetMesh& mesh, const TetMeshSnapshot& snapshot);
  void reserve(std::size_t vertex_count, std::size_t edge_count);
  bool insert_vertex_position(std::int32_t vertex, const double* point);
  bool erase_vertex_position(std::int32_t vertex,
                             const LocalPositionKey& old_key) noexcept;
  PositionStatus prepare_position_move(std::int32_t vertex,
                                       const double* old_point,
                                       const double* new_point,
                                       PositionMove& move);
  bool commit_position_move(PositionMove&& move) noexcept;

  bool insert_logical(const TetMesh& mesh, std::int32_t cell,
                      const LocalCell& vertices);
  bool erase(std::int32_t cell, const LocalCell& vertices);
  Star vertex_star(std::int32_t vertex) const;
  std::size_t edge_valence(std::int32_t first, std::int32_t second) const;
  // Finite cells containing the edge (sorted slots); empty if it is absent.
  std::span<const std::int32_t> edge_star(std::int32_t first, std::int32_t second) const;
  std::int32_t edge_cell_outside(
      const TetMesh& mesh, std::int32_t first, std::int32_t second,
      std::span<const std::int32_t> removed_slots) const;
  TetEditStatus relocate(TetMesh& mesh,
                         std::span<const TetEditVertex> moved_vertices,
                         const Star& star);
  // Commits an interior reconnection or collapse through the cavity-local
  // TetMesh route and updates this index. A collapse must pass this index's
  // current star of the removed vertex as the removed cells.
  TetEditStatus replace_cavity(TetMesh& mesh, const TetEdit& operation,
                               const Star* collapsed_star);
  // Cones a star-shaped cavity to one new interior vertex, removing the
  // buried vertices `removed_vertices` (TetMesh::insert_local_cone), through
  // the cavity-local TetMesh route and updates this index.
  TetEditStatus insert_cone(TetMesh& mesh, const TetEditVertex& vertex,
                            std::span<const std::int32_t> cavity,
                            std::span<const std::int32_t> removed_vertices,
                            std::vector<std::uint64_t>& created_cells);
  // Inserts a source-witnessed vertex into a constrained facet or segment
  // through TetMesh::insert_local_boundary_point (whose cavity and bank-row
  // contract applies) and updates this index.
  TetEditStatus insert_boundary_point(TetMesh& mesh, const TetEditVertex& vertex,
                                      std::span<const std::int32_t> cavity,
                                      std::span<const std::size_t> facet_rows,
                                      std::span<const std::size_t> segment_rows,
                                      std::vector<std::uint64_t>& created_cells);

 private:
  static bool erase_cell(std::vector<std::int32_t>& cells, std::int32_t cell);

  // Direct cell incidence, not a link walk, retains disconnected/nonmanifold stars.
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> vertex_cells_;
  std::unordered_map<LocalEdgeKey, std::vector<std::int32_t>, LocalEdgeHash>
      edge_cells_;
  PositionMap vertex_positions_;
  std::uint64_t topology_revision_ = 0;

};
}  // namespace phx::mc::detail
