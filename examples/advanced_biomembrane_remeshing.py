"""Transactional biomembrane remeshing through the multiregion surface owner."""

from __future__ import annotations

import numpy as np

from phydrax.applications.cellular_mechanics import BiomembranePlan
from phydrax.geometry.multiregion_surface import EdgeFlipProposal


def _cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int32,
    )
    return vertices, faces


def run() -> dict[str, object]:
    vertices, faces = _cube()
    membrane = BiomembranePlan(
        faces,
        vertex_ids=np.arange(100, 108),
        face_ids=np.arange(200, 212),
        bending_rigidity=0.0,
        local_area_modulus=0.3,
        species_diffusivity=(0.02,),
        species_ids=("membrane-lipid",),
    ).prepare(vertices)
    state = membrane.state(
        species_mass=(0.1 + 0.01 * np.arange(vertices.shape[0]))[:, None]
    )
    proposal = membrane.propose_remesh(state, EdgeFlipProposal((104, 106)))
    evidence = membrane.evaluate_remesh(
        proposal,
        maximum_relative_area_jump=1.0,
        maximum_relative_volume_jump=1.0,
        maximum_relative_energy_jump=1.0,
    )
    result = membrane.commit_remesh(proposal, evidence)
    before = float(np.sum(state.species_mass))
    after = float(np.sum(result.state.species_mass))
    sheet = proposal.surface_result.sheet_transfer
    face = proposal.surface_result.face_transfer
    return {
        "committed": result.committed,
        "surface_status": evidence.surface.status.name,
        "event_status": evidence.surface.records[0].status.name,
        "source_epoch": membrane.remesh_topology.epoch,
        "target_epoch": result.prepared.remesh_topology.epoch,
        "species_content_defect": after - before,
        "sheet_transfer_routes": None if sheet is None else sheet.route_count,
        "face_transfer_routes": None if face is None else face.route_count,
        "lineage_id": None if result.lineage is None else result.lineage.lineage_id,
        "derivative_available": evidence.derivative_available,
    }


if __name__ == "__main__":
    print(run())
