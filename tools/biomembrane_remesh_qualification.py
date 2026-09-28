"""Qualify biomembrane split/collapse/flip transactions and sparse conservation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from typing import Any

import jax
import numpy as np

from phydrax._meshcore import meshcore_available
from phydrax.applications.cellular_mechanics import BiomembranePlan
from phydrax.geometry.multiregion_surface import (
    EdgeCollapseProposal,
    EdgeFlipProposal,
    EdgeSplitProposal,
    seed_sphere,
)
from phydrax.qualification import QualificationRuntimeIdentity


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0)),
        dtype=np.float64,
    ) / np.sqrt(3.0)
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    return vertices, faces


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


def _run_event(
    vertices: np.ndarray,
    faces: np.ndarray,
    event: EdgeSplitProposal | EdgeCollapseProposal | EdgeFlipProposal,
    /,
) -> dict[str, Any]:
    membrane = BiomembranePlan(
        faces,
        bending_rigidity=1.0,
        local_area_modulus=np.linspace(0.2, 0.8, faces.shape[0]),
        species_diffusivity=(0.1, 0.05),
        species_ids=("lipid-a", "lipid-b"),
    ).prepare(vertices)
    mass = 0.01 + 0.001 * np.arange(2 * vertices.shape[0], dtype=np.float64).reshape(
        (-1, 2)
    )
    state = membrane.state(species_mass=mass)
    proposal = membrane.propose_remesh(state, event)
    evidence = membrane.evaluate_remesh(
        proposal,
        maximum_relative_area_jump=1.0,
        maximum_relative_volume_jump=1.0,
        maximum_relative_energy_jump=10.0,
    )
    result = membrane.commit_remesh(proposal, evidence)
    sheet = proposal.surface_result.sheet_transfer
    face = proposal.surface_result.face_transfer
    source_total = np.sum(state.species_mass, axis=0)
    target_total = np.sum(result.state.species_mass, axis=0)
    return {
        "kind": event.kind.name,
        "surface_status": evidence.surface.status.name,
        "event_status": evidence.surface.records[0].status.name,
        "committed": result.committed,
        "source_epoch": membrane.remesh_topology.epoch,
        "target_epoch": result.prepared.remesh_topology.epoch,
        "species_maximum_absolute_defect": float(
            np.max(np.abs(target_total - source_total))
        ),
        "relative_volume_jump": float(evidence.relative_volume_jump),
        "vertex_transfer_routes": None if sheet is None else sheet.route_count,
        "face_transfer_routes": None if face is None else face.route_count,
        "relative_area_jump": float(evidence.relative_area_jump),
        "dense_vertex_entries": None
        if sheet is None
        else sheet.source_size * sheet.target_size,
        "dense_face_entries": None
        if face is None
        else face.source_size * face.target_size,
        "derivative_available": evidence.derivative_available,
    }


def _rollback() -> dict[str, Any]:
    vertices, faces = _tetrahedron()
    membrane = BiomembranePlan(
        faces,
        bending_rigidity=1.0,
        species_diffusivity=(0.1,),
        species_ids=("lipid",),
    ).prepare(vertices)
    state = membrane.state(species_mass=np.full((4, 1), 0.1))
    proposal = membrane.propose_remesh(state, EdgeSplitProposal((0, 1)))
    evidence = membrane.evaluate_remesh(
        proposal,
        maximum_relative_area_jump=0.0,
        maximum_relative_volume_jump=0.0,
        maximum_relative_energy_jump=0.0,
    )
    result = membrane.commit_remesh(proposal, evidence)
    return {
        "committed": result.committed,
        "same_preparation_object": result.prepared is membrane,
        "same_state_object": result.state is state,
        "positions_bitwise_equal": bool(
            np.array_equal(result.state.positions, state.positions)
        ),
        "species_bitwise_equal": bool(
            np.array_equal(result.state.species_mass, state.species_mass)
        ),
    }


def _runtime() -> dict[str, Any]:
    identity = QualificationRuntimeIdentity(
        f"phydrax-{importlib.metadata.version('phydrax')}",
        f"python-{platform.python_version()}-jax-{jax.__version__}-numpy-{np.__version__}",
        jax.default_backend(),
        f"devices-{jax.device_count()}",
        "float64",
    )
    return {**dict(identity.to_record()), "meshcore": meshcore_available()}


def run() -> dict[str, Any]:
    tetra_vertices, tetra_faces = _tetrahedron()
    sphere = seed_sphere(1.0, subdivisions=0)
    cube_vertices, cube_faces = _cube()
    events = (
        _run_event(tetra_vertices, tetra_faces, EdgeSplitProposal((0, 1))),
        _run_event(sphere.positions, sphere.faces, EdgeCollapseProposal((0, 1))),
        _run_event(cube_vertices, cube_faces, EdgeFlipProposal((4, 6))),
    )
    rollback = _rollback()
    successful = all(
        row["surface_status"] == "COMMITTED"
        and row["event_status"] == "ACCEPTED"
        and row["committed"]
        and row["species_maximum_absolute_defect"] <= 1.0e-13
        and not row["derivative_available"]
        and row["vertex_transfer_routes"] < row["dense_vertex_entries"]
        and row["face_transfer_routes"] < row["dense_face_entries"]
        for row in events
    ) and (
        not rollback["committed"]
        and rollback["same_preparation_object"]
        and rollback["same_state_object"]
        and rollback["positions_bitwise_equal"]
        and rollback["species_bitwise_equal"]
    )
    return {
        "runtime": _runtime(),
        "events": events,
        "rollback": rollback,
        "successful": successful,
    }


def main() -> int:
    report = run()
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
