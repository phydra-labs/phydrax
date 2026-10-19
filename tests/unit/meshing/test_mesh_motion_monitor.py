#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import numpy as np
import pytest

import phydrax as phx


meshing = phx.meshing
Decision = meshing.MeshMotionDecision
_POINTS = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)))
_CELLS = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32)

# Empirical `test_motion_retains_original_domain_certificate_and_refuses_boundary_escape`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
MOTION_COVERAGE_AREA_ATOL = 2e-14
# Empirical `test_motion_retains_original_domain_certificate_and_refuses_boundary_escape`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
MOTION_TRIANGLE_AREA_SUM_ATOL = 2e-14


def _certified(points: Any, cells: Any) -> Any:
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    return meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _shifted_center(source: Any, shift: Any) -> Any:
    coordinates = np.asarray(source.mesh.coordinates).copy()
    center = int(np.argmin(np.linalg.norm(coordinates - 0.5, axis=1)))
    coordinates[center, 0] += shift
    return coordinates, center


def _grid(count: Any) -> Any:
    axis = np.linspace(0.0, 1.0, count + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    index = np.arange(points.shape[0]).reshape(count + 1, count + 1)
    a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    return points, np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))


def test_mesh_motion_monitor_scenario_1() -> None:
    source = _certified(_POINTS, _CELLS)
    monitor = meshing.MeshMotionMonitor(source.mesh)

    decisions = [
        monitor.assess(_shifted_center(source, shift)[0], boundary_residual=0.0).decision
        for shift in (0.1, 0.3, 0.4, 0.45, 0.48)
    ]

    assert decisions == [
        Decision.ACCEPT_MOTION,
        Decision.ACCEPT_MOTION,
        Decision.RELOCATE,
        Decision.REMESH,
        Decision.REMESH,
    ]
    repeated = [
        monitor.assess(_shifted_center(source, 0.4)[0], boundary_residual=0.0)
        for _ in range(2)
    ]
    assert repeated[0].assessment_id == repeated[1].assessment_id
    assert repeated[0].minimum_jacobian_ratio < 0.3
    # A folded mesh is never remeshed directly: untangling must come first.
    folded = monitor.assess(_shifted_center(source, 0.6)[0], boundary_residual=0.0)
    assert folded.decision is Decision.RELOCATE
    assert "uncertified-cells" in folded.reasons
    assert folded.minimum_jacobian_ratio < 0.0
    source = _certified(_POINTS, _CELLS)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    coordinates = np.asarray(source.mesh.coordinates)

    residual = monitor.assess(coordinates, boundary_residual=1.0e-3)
    nonfinite = coordinates.copy()
    nonfinite[0, 0] = np.nan

    assert residual.decision is Decision.REJECT
    assert residual.reasons == ("boundary-residual",)
    rejected = monitor.assess(nonfinite, boundary_residual=0.0)
    assert rejected.decision is Decision.REJECT
    assert rejected.certificate is None
    source = _certified(_POINTS, _CELLS)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    folded, center = _shifted_center(source, 0.6)

    advance = meshing.advance_mesh_motion(monitor, source, folded, boundary_residual=0.0)

    assert advance.decision is Decision.RELOCATE
    assert advance.accepted
    assert [value.decision for value in advance.assessments] == [
        Decision.RELOCATE,
        Decision.ACCEPT_MOTION,
    ]
    # ty: ignore[unresolved-attribute]
    assert advance.relocation.untangling.succeeded
    # ty: ignore[unresolved-attribute]
    relocated = np.asarray(advance.relocation.coordinates)
    boundary = np.arange(relocated.shape[0]) != center
    np.testing.assert_array_equal(relocated[boundary], folded[boundary])
    np.testing.assert_allclose(relocated[center], (0.5, 0.5), atol=1.0e-5)
    # ty: ignore[unresolved-attribute]
    assert advance.result.audit.passed
    assert advance.transition is None
    points, cells = _grid(4)
    source = _certified(points, cells)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    squeezed = np.asarray(source.mesh.coordinates) * np.asarray((0.2, 1.0))

    advance = meshing.advance_mesh_motion(
        monitor, source, squeezed, boundary_residual=0.0
    )

    assert advance.decision is Decision.REMESH
    assert not advance.accepted
    assert advance.adaptation is None
    assert advance.assessments[0].decision in (Decision.RELOCATE, Decision.REMESH)
    assert advance.assessments[-1].decision is not Decision.ACCEPT_MOTION
    # ty: ignore[unresolved-attribute]
    assert advance.relocation.accepted
    metric = advance.remesh_metric
    # ty: ignore[unresolved-attribute]
    assert metric.scope.source_id == advance.result.mesh.mesh_id
    # The reference size h = 1/4 per vertex restores isotropic reference cells.
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        np.asarray(metric.values)[:, 0, 0],
        # ty: ignore[unresolved-attribute]
        np.asarray(metric.values)[:, 1, 1],
    )
    # ty: ignore[unresolved-attribute]
    assert np.all(np.asarray(metric.values)[:, 0, 0] > 1.0)


def test_advance_escalates_a_nonconverged_relocation_unless_explicitly_admitted() -> None:
    source = _certified(_POINTS, _CELLS)
    shifted, _ = _shifted_center(source, 0.4)

    def advance(accept: Any) -> Any:
        policy = meshing.MeshMotionMonitorPolicy(
            relocation_termination=phx.optim.OptimizationTermination(maximum_steps=1),
            accept_valid_nonconverged_relocation=accept,
        )
        monitor = meshing.MeshMotionMonitor(source.mesh, policy=policy)
        return meshing.advance_mesh_motion(
            monitor, source, shifted, boundary_residual=0.0
        )

    refused = advance(False)
    admitted = advance(True)

    assert refused.assessments[0].decision is Decision.RELOCATE
    assert refused.relocation.status is meshing.MeshOptimizationStatus.NONCONVERGED
    assert not refused.relocation.accepted
    # A failed relocation escalates: the certified, unrelocated proposal seeds the
    # remesh request and is never reassessed as relocated.
    assert refused.decision is Decision.REMESH and not refused.accepted
    assert len(refused.assessments) == 1
    np.testing.assert_array_equal(refused.result.mesh.coordinates, shifted)
    assert admitted.relocation.status is meshing.MeshOptimizationStatus.VALID_NONCONVERGED
    assert admitted.relocation.accepted and len(admitted.assessments) == 2
    np.testing.assert_array_equal(
        admitted.result.mesh.coordinates, admitted.relocation.coordinates
    )


def test_motion_retains_original_domain_certificate_and_refuses_boundary_escape() -> None:
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.geometry import PiecewiseLinearDomain
    from phydrax.meshing._certification_inputs import MeshCertificationInputs
    from phydrax.meshing._certification_schedule import MeshCertificationSchedule
    from phydrax.meshing._lineage import identity_lineage

    original = _certified(_POINTS, _CELLS)
    mesh = original.mesh
    geometry = CellGeometrySpec.affine(mesh)
    domain = PiecewiseLinearDomain(
        _POINTS[:4],
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64),
        np.tile(np.asarray([0, -1], dtype=np.int64), (4, 1)),
        ("original-fluid",),
        source_id="original-square-authority",
    )
    request = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("volume_plc"),
        domain=domain,
        cell_regions=np.zeros(4, dtype=np.int64),
    )
    source = meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=meshing.CellMeshAuditPolicy(
            watertight_boundary=meshing.CellMeshAuditDisposition.REJECT
        ),
        certification_inputs=request,
        lineage=identity_lineage(mesh, mesh),
    )
    monitor = meshing.MeshMotionMonitor(source.mesh)
    moved, _ = _shifted_center(source, 0.1)
    accepted = meshing.advance_mesh_motion(monitor, source, moved, boundary_residual=0.0)
    assert accepted.accepted
    result = accepted.result
    if (
        result is None
        or result.certification is None
        or result.certification.coverage is None
    ):
        raise AssertionError(
            "fixture.result.certification.coverage must exist after accepted motion; got missing certificate"
        )
    certificate = result.certification
    certificate.require_passed()
    np.testing.assert_allclose(
        np.asarray(
            result.certification.coverage.achieved_region_measures, dtype=np.float64
        ),
        [1.0],
        rtol=0.0,
        atol=MOTION_COVERAGE_AREA_ATOL,
    )
    corners = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    np.testing.assert_allclose(
        np.sum(0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])),
        1.0,
        rtol=0.0,
        atol=MOTION_TRIANGLE_AREA_SUM_ATOL,
    )
    with pytest.raises(meshing.MeshingFailure):
        meshing.advance_mesh_motion(
            monitor,
            source,
            np.asarray(source.mesh.coordinates) + [0.1, 0.0],
            boundary_residual=0.0,
        )


def test_periodic_motion_uses_declared_scope_without_renewing_parent_work() -> None:
    from phydrax._meshcore import (
        meshcore_available,
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionLimits,
        NativeExecutionScope,
    )
    from phydrax.discretization import PeriodicCell
    from tests._support.periodic_meshing import periodic_lattice_points

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    # Displaced coefficient maps must retain bitwise periodic trace agreement.
    # Dyadic authored sites avoid source-expression rounding at paired traces.
    native_source = meshing.NativePeriodicSource(
        meshing.PeriodicPointOrbits(periodic_lattice_points(4), PeriodicCell(np.eye(2))),
        "scope-motion-torus",
        "r1",
    )
    request_scope = meshing.MeshingScope(
        native_source.source_id,
        native_source.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        "regions",
        np.asarray((0,), dtype=np.int64),
    )
    # This ownership fixture admits the authored grid by its represented median,
    # rather than asking refinement to replace its dyadic periodic traces.
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(2, 2, meshing.CellFamilyPolicy(required=("triangle",))),
        request_scope,
        size_controls=(
            meshing.UniformSizeControl(request_scope, 0.25, maximum_size=0.375),
        ),
        size_compliance=meshing.SizeCompliancePolicy(target_statistics=("p50",)),
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
    )
    source = (
        meshing.NativeMeshingProvider(meshing.NativeMeshingOptions("periodic_delaunay"))
        .plan(
            native_source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    monitor = meshing.MeshMotionMonitor(source.mesh)
    transfer = meshing.PeriodicAssociationTransfer(native_source, specification)
    policy = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=specification.limits,
        association_transfer=transfer,
    )
    coordinates = np.asarray(source.mesh.coordinates).copy()
    source_identity = source.result_id
    lowered = specification.limits.native_limits()
    owner = NativeExecutionScope(lowered)
    with owner:
        accepted = meshing.advance_mesh_motion(
            monitor,
            source,
            coordinates,
            boundary_residual=0.0,
            adaptation_policy=policy,
        )
        assert accepted.accepted
        assert accepted.decision is meshing.MeshMotionDecision.ACCEPT_MOTION
        assert accepted.result is not None
        np.testing.assert_array_equal(accepted.result.mesh.coordinates, coordinates)
        assert len(accepted.result.associations) == len(source.associations)
        targets = {
            entity_set.entity_set_id: entity_set
            for entity_set in (
                accepted.result.mesh.entity_set(dimension)
                for dimension in range(accepted.result.mesh.topological_dimension + 1)
            )
        }
        for association in accepted.result.associations:
            association.validate_target(targets[association.target_entity_set_id])
            assert association.source_id == native_source.source_id
            assert association.source_revision == native_source.source_revision
            assert association.complete and association.exact
        assert owner.remaining().remaining_work_units < lowered.maximum_work_units
        assert owner.limits is lowered
    assert owner.completed_evidence().status is MeshcoreStatus.OK

    exhausted = NativeExecutionScope(
        NativeExecutionLimits(
            0,
            lowered.maximum_geometry_queries,
            lowered.maximum_cavity_cells,
            lowered.maximum_managed_bytes,
            lowered.maximum_seconds,
        )
    )
    with pytest.raises(MeshcoreError) as refusal, exhausted:
        meshing.advance_mesh_motion(
            monitor,
            source,
            coordinates,
            boundary_residual=0.0,
            adaptation_policy=policy,
        )
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert exhausted.completed_evidence().status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert source.result_id == source_identity
    np.testing.assert_array_equal(source.mesh.coordinates, coordinates)
