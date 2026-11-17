//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// In-place commit of a relocation. apply_edit_batch stages every edit on a
// copy of the whole state, validates it there, audits the copy and swaps it
// in. For a relocation of an unbound, nonperiodic state that work reduces to
// the moved vertices: probe_relocation_edit evaluates every check of
// validate_and_commit's relocation branch on the unchanged state, in the same
// order, so a relocation it does not refuse is one validate_and_commit
// applies; the branch writes only the moved vertices' carriers, witnesses,
// edge parameters and ancestry; and audit_state reads none of the first three
// and holds after an ancestry row is erased. The commit below is those writes.
#include "tet_mesh.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace phx::mc {

bool TetMesh::apply_relocation_edit(const TetEdit& operation, TetEditStatus& status) {
  // probe_relocation_edit's domain. Outside it the probe returns false without
  // having validated anything, which is not an admission.
  if (operation.kind != TetEditKind::kRelocate || periodic_identity_.bound ||
      !periodic_orbits_.empty() || !quotient_cells_.empty() ||
      !periodic_entity_maps_.empty() || !operation.periodic_members.empty() ||
      !operation.removed_periodic_orbit_ids.empty() ||
      !operation.replacement_periodic_orbits.empty() ||
      !operation.removed_quotient_cell_ids.empty() ||
      !operation.replacement_quotient_cells.empty() ||
      !operation.replacement_periodic_entity_maps.empty() ||
      generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  if (probe_relocation_edit(operation, status)) return true;
  // validate_and_commit's relocation writes. None allocates, so the commit is
  // complete once it starts.
  for (const TetEditVertex& vertex : operation.changed_vertices) {
    const std::size_t slot = static_cast<std::size_t>(vertex_slot(vertex.id));
    std::copy(vertex.carrier.begin(), vertex.carrier.end(), points_.begin() + 3 * slot);
    witnesses_[slot] = vertex.witness;
    // The relocation supplies no exact replacement stencil; do not retain stale weights.
    vertex_ancestry_.erase(vertex.id);
    if (vertex.has_edge_parameter) {
      vertex_has_edge_parameter_[slot] = 1;
      vertex_edge_endpoints_[slot] = vertex.edge_endpoints;
      vertex_edge_parameters_[slot] = vertex.edge_parameter;
    } else if (vertex.edge_witness_supplied) {
      vertex_has_edge_parameter_[slot] = 0;
      vertex_edge_endpoints_[slot] = {};
      vertex_edge_parameters_[slot] = 0.0;
    }
  }
  ++generation_;
  unmet_records_.clear();
  status = TetEditStatus::kApplied;
  return true;
}

}  // namespace phx::mc
