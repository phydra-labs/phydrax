//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Vertex/cell incidence of a TetMesh for local edits: one incident cell and
// the finite-cell degree per vertex slot, stars gathered by walking face
// adjacency. A star the walk cannot reach whole (a pinched vertex) is found
// by a scan, so stars are always complete.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "tet_mesh.hpp"

namespace phx::mc::detail {

using LocalCell = std::array<std::int32_t, 4>;
using LocalEdgeKey = std::pair<std::int32_t, std::int32_t>;

LocalEdgeKey sorted_local_edge(std::int32_t first, std::int32_t second) noexcept;

class LocalIncidence {
 public:
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

  bool initialize(const TetMesh& mesh);
  // The caller owns and updates this index after each constrained segment split.
  void bind_segment_index(const TetSegmentIndex& index) noexcept { segment_index_ = &index; }

  bool insert_logical(const TetMesh& mesh, std::int32_t cell,
                      const LocalCell& vertices);
  bool erase(std::int32_t cell, const LocalCell& vertices);
  // Finite cells containing `vertex`, sorted by cell ID.
  Star vertex_star(std::int32_t vertex) const;
  std::size_t edge_valence(std::int32_t first, std::int32_t second) const;
  // Finite cells containing the edge (sorted slots); empty if it is absent.
  // The span stays valid until the next edge_star call.
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
  // The finite cells containing `vertex` into `cells` (unsorted).
  void collect_star(std::int32_t vertex, std::vector<std::int32_t>& cells) const;
  // The finite cells containing both endpoints into `cells` (sorted slots).
  void collect_edge(std::int32_t first, std::int32_t second,
                    std::vector<std::int32_t>& cells) const;

  const TetMesh* mesh_ = nullptr;
  // Per vertex slot: one incident cell (a hint, checked before use) and the
  // number of finite cells containing it.
  mutable std::vector<std::int32_t> vertex_cell_;
  std::vector<std::int32_t> vertex_degree_;
  mutable std::vector<std::uint32_t> visited_;
  mutable std::uint32_t visit_ = 0;
  mutable std::vector<std::int32_t> stack_;
  mutable std::vector<std::int32_t> star_scratch_;
  mutable std::vector<std::int32_t> edge_scratch_;
  const TetSegmentIndex* segment_index_ = nullptr;
  std::uint64_t topology_revision_ = 0;
};
}  // namespace phx::mc::detail
