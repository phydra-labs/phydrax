# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Scientific state and source records shared by restart scenarios."""

from pathlib import Path
from typing import Any

import jax
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._restart_topology import (
    MeshingCheckpointState,
    TopologyRestartRelation,
)
from phydrax.meshing._audit import CellMeshAuditReport
from phydrax.units import METER


def build_repository(
    path: Path,
    *,
    maximum_chunk_bytes: int = 64,
    maximum_metadata_bytes: int = 64 * 1024,
) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.meshing-restart",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        path,
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=maximum_chunk_bytes,
            maximum_metadata_bytes=maximum_metadata_bytes,
        ),
    )


def checkpoint_state() -> MeshingCheckpointState:
    return MeshingCheckpointState(
        source_topology_id="source-topology",
        source_geometry_id="source-geometry",
        topology_id="accepted-topology",
        geometry_id="accepted-geometry",
        result_id="accepted-result",
        source_revision_id="source-revision",
        coordinate_contract_id="affine-triangle-coordinates",
        lineage_id="accepted-lineage",
        epoch_id="accepted-epoch",
        evidence_id="collective-evidence",
        transfer_id="accepted-transfer",
        request_id="meshing-request",
        mesh_epoch=7,
        array_roles=(
            ("vertex_ids", "entity-ids"),
            ("coordinates", "geometry"),
            ("cell_vertex_ids", "topology"),
            ("parent_ids", "lineage"),
            ("accepted_epochs", "epoch"),
            ("certificate_verdicts", "evidence"),
            ("material_inventory", "state"),
        ),
    )


def scientific_arrays() -> dict[str, np.ndarray]:
    # Nonconsecutive IDs make accidental reuse of old local slots observable.
    return {
        "vertex_ids": np.asarray([101, 509, 1003, 2401], dtype=np.int64),
        "coordinates": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64
        ),
        "cell_vertex_ids": np.asarray(
            [[101, 509, 1003], [509, 2401, 1003]], dtype=np.int64
        ),
        "parent_ids": np.asarray(
            [[101, 11], [509, 11], [1003, 23], [2401, 23]], dtype=np.int64
        ),
        "accepted_epochs": np.asarray([6, 7], dtype=np.int64),
        "certificate_verdicts": np.asarray([1, 1], dtype=np.uint8),
        "material_inventory": np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float64),
    }


def restart_relation() -> TopologyRestartRelation:
    return TopologyRestartRelation(
        canonical_fingerprint({"placement": "source"}),
        canonical_fingerprint({"placement": "destination"}),
        "bitwise",
    )


def native_source_closure() -> tuple[
    CellMesh, CellGeometrySpec, CellMeshAuditReport, dict[str, Any]
]:
    points = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        ),
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh.from_triangles(points, faces)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    audit = phx.meshing.audit_cell_mesh(mesh, geometry)
    source = phx.geometry.ImplicitBoundarySource(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="restart-sphere").compile(
            coordinate_unit=METER
        ),
        source_id="restart-sphere",
        spacing=0.2,
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("surface"),
        source=source,
        fidelity_tolerance=0.8,
    )
    association = phx.meshing.GeometryAssociation(
        phx.meshing.GeometryAssociationKind.IMPLICIT,
        source.source_id,
        source.source_revision,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
        tuple("implicit-zero-set" for _ in points),
        np.zeros((points.shape[0],), dtype=np.float64),
        exact=True,
    )
    return (
        mesh,
        geometry,
        audit,
        {
            "source": source,
            "certification_inputs": report.request,
            "report": report,
            "associations": (association,),
        },
    )


def native_source_state(
    closure: dict[str, Any],
    *,
    source_array_bindings: dict[str, str] | None = None,
    source_logical_arrays: dict[str, jax.Array] | None = None,
) -> MeshingCheckpointState:
    payload = checkpoint_state().to_payload()
    inputs = closure["certification_inputs"]
    payload.update(
        topology_id=inputs.topology_id,
        geometry_id=inputs.geometry_id,
        source_revision_id=inputs.source_revision or payload["source_revision_id"],
        source_closure=closure,
        source_owner_index=jax.process_index(),
        source_array_bindings=source_array_bindings,
        source_logical_arrays=source_logical_arrays,
    )
    return MeshingCheckpointState(**payload)


def native_source_arrays(mesh: CellMesh, mesh_epoch: int) -> dict[str, np.ndarray]:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    return {
        "vertex_ids": vertex_ids,
        "coordinates": np.asarray(mesh.coordinates),
        "cell_vertex_ids": vertex_ids[np.asarray(mesh.blocks[0].vertices)],
        "parent_ids": np.stack((vertex_ids, vertex_ids), axis=-1),
        "accepted_epochs": np.full(cell_ids.shape, mesh_epoch, dtype=np.int64),
        "certificate_verdicts": np.ones(cell_ids.shape, dtype=np.uint8),
        "material_inventory": np.arange(1, len(vertex_ids) + 1, dtype=np.float64),
    }
