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
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tet_mesh.hpp"

namespace phx::mc::detail {

using LocalCell = std::array<std::int32_t, 4>;
using LocalEdgeKey = std::pair<std::int32_t, std::int32_t>;

LocalEdgeKey sorted_local_edge(std::int32_t first, std::int32_t second) noexcept;

// Geometry proof for a caller's stencil of a witness-free boundary vertex.
enum class SourceStencilImage { kExact, kRoundedPlane };

// Distinct mesh source-point rows and finite binary64 weights summing to one.
// Nonzero weights form the support. Rounded-plane support is confined to
// stationary survivors; all other consumers require the exact image.
struct SourceStencil {
  std::span<const std::int32_t> points;
  std::span<const double> weights;
  SourceStencilImage image = SourceStencilImage::kExact;
};

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
  // The caller owns this index (segment vertex IDs -> segments_ row) and
  // updates it after each constrained segment split; collapse_boundary keeps
  // it current for the segments it retires, creates and moves.
  void bind_segment_index(TetSegmentIndex& index) noexcept { segment_index_ = &index; }

  bool insert_logical(const TetMesh& mesh, std::int32_t cell,
                      const LocalCell& vertices);
  bool erase(std::int32_t cell, const LocalCell& vertices);
  // The finite cells containing `vertex` into `cells` (unsorted).
  void collect_star(std::int32_t vertex, std::vector<std::int32_t>& cells) const;
  // Finite cells containing `vertex`, sorted by cell ID.
  Star vertex_star(std::int32_t vertex) const;
  std::size_t edge_valence(std::int32_t first, std::int32_t second) const;
  // The same count by walking around the edge from `seed`, a cell holding it.
  std::size_t edge_valence(std::int32_t first, std::int32_t second, std::int32_t seed) const;
  // Finite cells containing the edge (sorted slots); empty if it is absent.
  // The span stays valid until the next edge_star call.
  std::span<const std::int32_t> edge_star(std::int32_t first, std::int32_t second) const;
  // The finite cell of least ID holding the edge, not among `removed_slots`
  // (-1: none); `seed`, when a live cell holding the edge, starts a walk
  // around it.
  std::int32_t edge_cell_outside(
      const TetMesh& mesh, std::int32_t first, std::int32_t second,
      std::span<const std::int32_t> removed_slots, std::int32_t seed = -1) const;
  // Moves one vertex (moved_vertices holds its record) whose current star
  // is `star`. An interior vertex (dimension three, no witness) goes through
  // TetMesh::relocate_with_complete_star. A facet or segment vertex
  // (dimension two or one) is decided on its star: the record's checks, the
  // facets and segments through it revalidated on their sources with the
  // moved record (in bank order) and exact positivity of the star cells, in
  // TetMesh::apply_relocation_edit's order. Without a stencil the verdict
  // and state equal it; a periodic or implicit-bound mesh is refused.
  // An optional `moved_stencil` additionally certifies the new coordinates
  // of a witness-free facet or segment vertex. Its nonzero support must lie
  // in every incident facet and segment source group. An invalid or
  // inapplicable stencil is refused, not ignored. The stencil is not stored
  // and does not replace any relocation check.
  TetEditStatus relocate(TetMesh& mesh,
                         std::span<const TetEditVertex> moved_vertices,
                         const Star& star, const SourceStencil* moved_stencil = nullptr);
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
  // Collapses the boundary vertex r into s (a kCollapse `operation` with the
  // single pair (r, s)) through the cavity-local TetMesh route and updates
  // this index. r has dimension 2 or 1; s lies on r's carrier (the support of
  // its witness, after the move when s moves, is within the source points
  // common to the groups of r's constrained facets and segments), and a
  // segment vertex collapses only along the constrained segment rs.
  // `removed_star` is this index's current star of r and, exactly when s moves
  // (operation.changed_vertices is {s} with its new exact witness),
  // `survivor_star` its current star of s (otherwise null); the removed cells
  // are their union. The removed facets and segments are all of r's, the
  // replacement ones their images under r -> s (same source, orientation and
  // vertex order) except those through s. The verdict and the state equal
  // TetMesh::apply_edit_batch on the single-operation batch, except that a
  // periodic or implicit-bound mesh and an operation outside this contract
  // are refused (kRejectedTopology) and the facet and segment banks are not
  // kept in ID order (a removed row takes the bank's last row).
  // A witness-free survivor (after its move when s moves) is supported where
  // it is a source point of r's carrier or, with `survivor_stencil`, where
  // the verified stencil (SourceStencil) of its coordinates has its support
  // within the source points common to the groups of r's constrained facets
  // and segments. A stencil for a witnessed survivor or a failed exact check
  // is refused. A rounded stencil must prove its exact and represented points
  // lie inside the same axis-aligned source group, off its rim. A missing
  // result means no rounded-plane admission; no edit has been attempted.
  std::optional<TetEditStatus> collapse_boundary(
      TetMesh& mesh, const TetEdit& operation, const Star& removed_star,
      const Star* survivor_star, const SourceStencil* survivor_stencil = nullptr);
  // Swaps the boundary edge ab for cd (a kRingReconnect `operation`): ab is
  // no constrained segment and lies in exactly two constrained facets, abc
  // and abd (operation.removed_facet_ids), of one source group and
  // orientation; its cells (operation.removed_cell_ids) are an open fan from
  // c to d; cd is not an edge. The replacement cells fill the fan and the
  // replacement facets are acd and bcd with that group and orientation, each
  // ordered so its exact normal points the way abc's does (a zero alignment
  // is kInvalidGeometry). Verdict, state and refusals as collapse_boundary;
  // apply_edit_batch's exact volume check refuses the swap (kRejectedTopology)
  // unless a, b, c and d are exactly coplanar.
  TetEditStatus swap_boundary_edge(TetMesh& mesh, const TetEdit& operation);
  // Splits the boundary edge ab at the new vertex `vertex` (a never-used
  // ID): ab is no constrained segment and lies in exactly two constrained
  // faces, abc and abd, both on the hull. The kSplit edit replaces each cell
  // around ab by its halves (b, then a, replaced by the vertex; IDs from the
  // mesh's next cell ID, each with its parent as ancestor) and abc, abd by
  // (a, m, c), (m, b, c), (b, m, d), (m, a, d) (vertex order, source and
  // orientation kept; IDs from the next facet ID), whose finite cell IDs
  // are returned. Verdict and state equal TetMesh::apply_edit_batch on that
  // edit (whose coverage proof is exact, or the declared pair's of
  // TetMesh::declared_split_pair), except that a periodic or implicit-bound
  // mesh and a split outside this contract are refused (kRejectedTopology)
  // and the facet bank is not kept in ID order.
  TetEditStatus split_boundary_edge(TetMesh& mesh, std::int32_t a, std::int32_t b,
                                    const TetEditVertex& vertex,
                                    std::vector<std::uint64_t>& created_cells);

  // From the bound mesh's current boundary index. kApplied with a null
  // record means no constraint; a failed index is kStaleGeneration. Records
  // are read-only borrows, valid only until the next edit or initialize.
  TetEditStatus boundary_facet(std::array<std::uint64_t, 3> vertices,
                                const TetSnapshotFacet*& record);
  TetEditStatus boundary_segment(std::array<std::uint64_t, 2> vertices,
                                  const TetSnapshotSegment*& record);

 private:
  struct FaceIdHash {
    std::size_t operator()(const std::array<std::uint64_t, 3>& face) const noexcept {
      const std::uint64_t mixed = (face[0] * 0x9E3779B97F4A7C15ULL) ^
                                  (face[1] + (face[1] << 21)) ^ (face[2] * 0xC2B2AE3D27D4EB4FULL);
      return static_cast<std::size_t>((mixed ^ (mixed >> 29)) * 0xBF58476D1CE4E5B9ULL);
    }
  };
  // The constrained facets and segments by sorted vertex IDs (bank rows) and
  // the source points of every face and segment group (sorted point rows):
  // built on the first boundary edit, then kept current by the edits that
  // change the banks.
  struct BoundaryIndex {
    std::unordered_map<std::array<std::uint64_t, 3>, std::size_t, FaceIdHash> facets;
    TetSegmentIndex segments;
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> face_group_points;
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> segment_group_points;
    // Each side of each face group bounds one region (apply_edit_batch
    // refuses every cell edit otherwise); accepted edits keep it so.
    bool regions_consistent = true;
  };
  struct BoundaryEdit;

  // The finite cells containing both endpoints into `cells` (sorted slots).
  void collect_edge(std::int32_t first, std::int32_t second,
                    std::vector<std::int32_t>& cells) const;
  bool current_star(const Star& star, std::int32_t vertex) const noexcept;
  bool ensure_boundary_index(const TetMesh& mesh);
  bool rounded_plane_support(const TetMesh& mesh, std::int32_t group,
                             const SourceStencil& stencil, const double* carrier) const;
  // Validates and commits a boundary edit resolved by collapse_boundary or
  // swap_boundary_edge, in apply_edit_batch's refusal order, and updates
  // this index and the bank indexes.
  TetEditStatus commit_boundary_edit(TetMesh& mesh, const TetEdit& operation,
                                     BoundaryEdit& edit);
  // The facet or segment vertex branch of relocate.
  TetEditStatus relocate_boundary(TetMesh& mesh, const TetEditVertex& moved, const Star& star,
                                  const SourceStencil* moved_stencil);

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
  TetSegmentIndex* segment_index_ = nullptr;
  BoundaryIndex boundary_index_;
  bool boundary_index_ready_ = false;
  std::uint64_t topology_revision_ = 0;
};
}  // namespace phx::mc::detail
