from __future__ import annotations

import numpy as np
import pytest

from phydrax._identity import SemanticProvenance
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._result import CellMeshingResult


@pytest.fixture(scope="module")
def source_certified_tetrahedron() -> CellMeshingResult:
    mesh = CellMesh(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64),
        (
            CellBlock(
                "volume", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    domain = PiecewiseLinearDomain(
        np.asarray(mesh.coordinates),
        np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("solid",),
        source_id="certification-provenance-source",
    )
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("volume_plc"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=np.zeros(1, dtype=np.int64),
    )
    return certify_cell_mesh(
        mesh,
        SpatialCoordinateContract.si(),
        geometry=geometry,
        certification_inputs=inputs,
    )


def test_source_certified_result_reconstructs_its_scientific_identity(
    source_certified_tetrahedron: CellMeshingResult,
) -> None:
    restored = CellMeshingResult.from_stored_fields(source_certified_tetrahedron)
    assert restored.result_id == source_certified_tetrahedron.result_id
    assert (
        restored.provenance.semantic_id
        == source_certified_tetrahedron.provenance.semantic_id
    )
    assert restored.certification is not None
    restored.certification.require_passed()


@pytest.mark.parametrize(
    "field", ("mesh_id", "geometry_layout_id", "coordinate_contract", "audit")
)
def test_restored_result_refuses_inconsistent_certification_provenance(
    source_certified_tetrahedron: CellMeshingResult,
    field: str,
) -> None:
    accepted = source_certified_tetrahedron
    premises = {
        "kind": "native-cell-mesh-certification",
        "mesh_id": accepted.mesh.mesh_id,
        "geometry_layout_id": accepted.geometry.geometry_layout_id,
        "coordinate_contract": accepted.coordinate_contract.spatial_id,
        "audit": accepted.audit.report_id,
    }
    premises[field] = "different-scientific-premise"
    inconsistent = CellMeshingResult(
        accepted.mesh,
        accepted.geometry,
        accepted.coordinate_contract,
        accepted.audit,
        accepted.quality,
        accepted.compliance,
        accepted.trace,
        accepted.provider,
        accepted.runtime,
        accepted.derivative_mode,
        SemanticProvenance(premises),
        certification=accepted.certification,
    )
    with pytest.raises(ValueError, match="certification provenance"):
        CellMeshingResult.from_stored_fields(inconsistent)
