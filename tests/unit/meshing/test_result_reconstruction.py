from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint
from phydrax._identity import SemanticProvenance
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.meshing import _audit
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._quality import CellQualityReport
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


def _one_ulp_up(tree: Any, /) -> Any:
    def nudge(leaf: Any) -> Any:
        values = np.asarray(leaf)
        if not np.issubdtype(values.dtype, np.floating):
            return leaf
        return jnp.asarray(
            np.where(np.isfinite(values), np.nextafter(values, np.inf), values)
        )

    return jax.tree_util.tree_map(nudge, tree)


def test_restore_under_other_platform_quality_rounding_keeps_stored_evidence(
    source_certified_tetrahedron: CellMeshingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another platform's XLA quality samples differ by ulps (LIN vs MAC: 1-7).

    The sampled arrays are diagnostics: restore accepts with the same identities
    and returns the stored samples, not this platform's re-evaluation.
    """
    accepted = source_certified_tetrahedron
    evaluate = _audit.evaluate_cell_quality
    perturbed = []

    def other_platform(*args: Any, **kwargs: Any) -> Any:
        value = _one_ulp_up(evaluate(*args, **kwargs))
        perturbed.append(array_tree_fingerprint(value))
        return value

    monkeypatch.setattr(_audit, "evaluate_cell_quality", other_platform)
    restored = CellMeshingResult.from_stored_fields(accepted)
    assert perturbed
    stored_samples = array_tree_fingerprint(accepted.quality.evaluation)
    assert stored_samples not in perturbed
    assert restored.certification is not None and accepted.certification is not None
    assert (
        restored.result_id,
        restored.certification.report_id,
        restored.audit.report_id,
        restored.quality.report_id,
    ) == (
        accepted.result_id,
        accepted.certification.report_id,
        accepted.audit.report_id,
        accepted.quality.report_id,
    )
    for quality in (
        restored.quality,
        restored.audit.quality,
        restored.certification.audit.quality,
    ):
        assert array_tree_fingerprint(quality.evaluation) == stored_samples
        assert quality.minimum_mean_ratio == accepted.quality.minimum_mean_ratio


def test_restore_refuses_stored_quality_samples_of_other_cells(
    source_certified_tetrahedron: CellMeshingResult,
) -> None:
    """Kept diagnostics must still describe the recomputed cells exactly.

    The forged report is self-consistent and has the stored identity: only the
    binding of its samples to the recomputed cells refuses it.
    """

    def other_cells(node: Any) -> Any:
        if not isinstance(node, CellQualityReport):
            return node
        evaluation = jax.tree_util.tree_map(
            lambda leaf: (
                leaf + 1 if np.issubdtype(np.asarray(leaf).dtype, np.integer) else leaf
            ),
            node.evaluation,
        )
        return CellQualityReport(
            evaluation,
            mesh_id=node.mesh_id,
            gate_thresholds=node.gate_thresholds,
            measure_failure_ids=node.measure_failure_ids,
            mean_ratio_failure_ids=node.mean_ratio_failure_ids,
            aspect_ratio_failure_ids=node.aspect_ratio_failure_ids,
        )

    forged = jax.tree_util.tree_map(
        other_cells,
        source_certified_tetrahedron,
        is_leaf=lambda node: isinstance(node, CellQualityReport),
    )
    assert forged.quality.report_id == source_certified_tetrahedron.quality.report_id
    with pytest.raises(ValueError, match="names other cells"):
        CellMeshingResult.from_stored_fields(forged)


def test_restore_refuses_one_ulp_tampered_stored_geometry(
    source_certified_tetrahedron: CellMeshingResult,
) -> None:
    """Every stored geometry copy moved by one ulp no longer reproves.

    The mesh is unchanged and the move is inside the audit's 2^-46 corner
    binding, so only the exact geometry identity refuses it.
    """
    accepted = source_certified_tetrahedron
    tampered = np.asarray(accepted.geometry.coordinates).copy()
    tampered[1, 0] = np.nextafter(tampered[1, 0], np.inf)

    def swap(node: Any) -> Any:
        if isinstance(node, CellGeometrySpec):
            return eqx.tree_at(
                lambda geometry: geometry.coordinates, node, jnp.asarray(tampered)
            )
        return node

    forged = jax.tree_util.tree_map(
        swap, accepted, is_leaf=lambda node: isinstance(node, CellGeometrySpec)
    )
    assert np.array_equal(np.asarray(forged.geometry.coordinates), tampered)
    assert np.array_equal(
        np.asarray(forged.mesh.coordinates), np.asarray(accepted.mesh.coordinates)
    )
    with pytest.raises(ValueError, match="Stored certification request changed"):
        CellMeshingResult.from_stored_fields(forged)
