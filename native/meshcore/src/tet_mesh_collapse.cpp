//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Explicit source-preserving collapse transactions.
#include "tet_mesh.hpp"

#include <utility>
#include <vector>

namespace phx::mc {

TetEditStatus TetMesh::collapse_vertex(
    std::uint64_t remove, std::uint64_t keep,
    std::span<const std::uint64_t> removed_cells,
    std::span<const TetEditCell> replacement_cells,
    std::span<const TetEditFacet> replacement_facets,
    std::span<const TetEditSegment> replacement_segments) {
  TetEdit operation;
  operation.kind = TetEditKind::kCollapse;
  operation.collapse_pairs.push_back({remove, keep});
  operation.removed_vertex_ids.push_back(remove);
  operation.removed_cell_ids.assign(removed_cells.begin(), removed_cells.end());
  operation.replacement_cells.assign(replacement_cells.begin(), replacement_cells.end());
  operation.replacement_facets.assign(replacement_facets.begin(), replacement_facets.end());
  operation.replacement_segments.assign(replacement_segments.begin(), replacement_segments.end());
  TetEditBatch batch{generation_, source_revision_, frame_id_, identity_bound_,
                     {std::move(operation)}, source_id_, source_map_id_, periodic_identity_};
  std::vector<TetEditResult> result;
  const TetEditStatus status = apply_edit_batch(batch, result);
  return status == TetEditStatus::kApplied ? result.front().status : status;
}

}  // namespace phx::mc
