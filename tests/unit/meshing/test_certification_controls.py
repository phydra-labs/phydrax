"""Boundary values and prepared acceptance preserve consumer-visible identity."""

from dataclasses import replace

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.meshing import CellMeshAuditReport, MeshCertificationPreparedEvidence
from phydrax.meshing._sizing import size_statistic_quantile, SizeStatistic
from tests._support.meshing_fixtures import mesh_scope


@pytest.mark.parametrize(
    "angle",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(np.pi / 3.0, id="equilateral-limit"),
        pytest.param(np.inf, id="infinite"),
        pytest.param(np.nan, id="not-finite"),
    ],
)
def test_quality_target_refuses_outside_open_angle_interval(angle: float) -> None:
    with pytest.raises(ValueError, match="MeshQualityTarget.minimum_angle"):
        phx.meshing.MeshQualityTarget(minimum_angle=angle)


@pytest.mark.parametrize(
    "angle",
    [
        pytest.param(np.nextafter(0.0, 1.0), id="smallest-positive"),
        pytest.param(np.nextafter(np.pi / 3.0, 0.0), id="below-equilateral-limit"),
    ],
)
def test_quality_target_retains_valid_angle_without_rounding(angle: float) -> None:
    target = phx.meshing.MeshQualityTarget(minimum_angle=angle)
    assert target.minimum_angle == angle
    assert (
        target.target_id == phx.meshing.MeshQualityTarget(minimum_angle=angle).target_id
    )


@pytest.fixture
def prepared_acceptance() -> tuple[
    CellMesh, CellGeometrySpec, CellMeshAuditReport, MeshCertificationPreparedEvidence
]:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_tetrahedra(
        coordinates, np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    )
    geometry = CellGeometrySpec.affine(mesh)
    audit = phx.meshing.audit_cell_mesh(
        mesh,
        geometry,
        policy=phx.meshing.CellMeshAuditPolicy(
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
        ),
    )
    facets = np.asarray(
        phx.discretization.reference_cell_topology("tetrahedron").entities[2],
        dtype=np.int64,
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        coordinates,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("body",),
        source_id="prepared-acceptance-controls",
    )
    prepared = MeshCertificationPreparedEvidence(
        mesh,
        geometry,
        audit.validity,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
    )
    return mesh, geometry, audit, prepared


@pytest.mark.parametrize("variant", ["volume", "boundary"])
def test_prepared_acceptance_variants_preserve_report_identity(
    prepared_acceptance: tuple[
        CellMesh, CellGeometrySpec, CellMeshAuditReport, MeshCertificationPreparedEvidence
    ],
    variant: str,
) -> None:
    mesh, geometry, audit, proof = prepared_acceptance
    request = proof.request
    prepared = (
        phx.meshing.PreparedVolumeAcceptance(proof)
        if variant == "volume"
        else phx.meshing.PreparedBoundaryAcceptance(embedding=proof.embedding)
    )
    cold = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=request.domain,
        cell_regions=request.cell_regions,
    )
    reused = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=request.domain,
        cell_regions=request.cell_regions,
        prepared=prepared,
    )
    assert cold.passed and reused.passed
    assert reused.report_id == cold.report_id
    assert reused.embedding is proof.embedding


def test_whole_prepared_evidence_cannot_adopt_a_second_embedding(
    prepared_acceptance: tuple[
        CellMesh, CellGeometrySpec, CellMeshAuditReport, MeshCertificationPreparedEvidence
    ],
) -> None:
    _, _, _, proof = prepared_acceptance
    with pytest.raises(TypeError, match="embedding"):
        replace(phx.meshing.PreparedVolumeAcceptance(proof), embedding=proof.embedding)


def test_volume_specification_replace_recomputes_derived_identity(
    prepared_acceptance: tuple[
        CellMesh, CellGeometrySpec, CellMeshAuditReport, MeshCertificationPreparedEvidence
    ],
) -> None:
    mesh, _, _, _ = prepared_acceptance
    scope = mesh_scope(mesh, 3)
    target = phx.meshing.CellMeshingTarget(
        3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
    )
    specification = phx.meshing.VolumeMeshingSpec(
        target,
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.2),),
    )
    assert replace(specification).specification_id == specification.specification_id
    changed = replace(specification, deterministic=False)
    rebuilt = phx.meshing.VolumeMeshingSpec(
        target=target,
        boundary_scope=scope,
        fill_strategy=specification.fill_strategy,
        size_controls=specification.size_controls,
        deterministic=False,
    )
    assert changed.specification_id == rebuilt.specification_id
    assert changed.specification_id != specification.specification_id


def test_limits_replace_recomputes_derived_identity() -> None:
    limits = phx.geometry.MeshCertificateLimits()
    assert replace(limits).limits_id == limits.limits_id
    changed = replace(limits, maximum_candidate_pairs=17)
    assert changed.limits_id != limits.limits_id
    assert (
        changed.limits_id
        == phx.geometry.MeshCertificateLimits(maximum_candidate_pairs=17).limits_id
    )


@pytest.mark.parametrize(
    ("statistic", "quantile"),
    [
        pytest.param("p50", 0.5, id="median"),
        pytest.param("p95", 0.95, id="ninety-fifth-percentile"),
    ],
)
def test_size_statistic_quantile_retains_declared_percentile(
    statistic: SizeStatistic, quantile: float
) -> None:
    assert size_statistic_quantile(statistic) == quantile


@pytest.mark.parametrize(
    "statistics",
    [
        pytest.param((), id="empty"),
        pytest.param(("p50", "p50"), id="duplicate"),
    ],
)
def test_size_compliance_refuses_missing_or_duplicate_statistics(
    statistics: tuple[SizeStatistic, ...],
) -> None:
    with pytest.raises(ValueError, match=r"SizeCompliancePolicy\.target_statistics"):
        phx.meshing.SizeCompliancePolicy(target_statistics=statistics)


def test_size_compliance_refuses_unknown_statistic() -> None:
    with pytest.raises(ValueError, match=r"SizeCompliancePolicy\.target_statistics"):
        phx.meshing.SizeCompliancePolicy(target_statistics=("p99",))  # ty: ignore[invalid-argument-type]


def test_size_statistic_quantile_refuses_unknown_statistic() -> None:
    with pytest.raises(ValueError, match=r"SizeCompliancePolicy\.target_statistics"):
        size_statistic_quantile("p99")  # ty: ignore[invalid-argument-type]


def test_size_compliance_preserves_statistic_order_and_identity() -> None:
    ordered = phx.meshing.SizeCompliancePolicy(target_statistics=("p95", "p50"))
    assert ordered.target_statistics == ("p95", "p50")
    assert (
        ordered.policy_id
        == phx.meshing.SizeCompliancePolicy(target_statistics=("p95", "p50")).policy_id
    )
    assert ordered.policy_id != phx.meshing.SizeCompliancePolicy().policy_id
