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
        or not isinstance(result.certification.coverage, phx.geometry.DomainCoverageCertificate)
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


@pytest.fixture(scope="module")
def implicit_motion_generation() -> tuple[Any, Any]:
    from phydrax._meshcore import meshcore_available
    from phydrax.units import METER
    from tests._support.implicit_meshing import IMPLICIT_DOMAIN, implicit_specification

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(np.asarray(1.2 * IMPLICIT_DOMAIN))
    native_source = meshing.NativeImplicitSource(geometry, grid, "implicit", "state")
    scope = implicit_specification().boundary_scope
    specification = meshing.VolumeMeshingSpec(
        meshing.CellMeshingTarget(
            3, 3, meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            meshing.UniformSizeControl(
                scope, 0.2, strength=meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    return native_source, specification


@pytest.fixture(scope="module")
def implicit_motion_source(
    implicit_motion_generation: tuple[Any, Any],
) -> tuple[Any, Any, Any]:
    native_source, specification = implicit_motion_generation
    source = (
        meshing.NativeMeshingProvider(
            meshing.NativeMeshingOptions("implicit_restricted_delaunay")
        )
        .plan(
            native_source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    assert source.certification is not None
    (association,) = source.associations
    query = source.certification.request.source
    if isinstance(query, phx.geometry.ImplicitProjectionBoundarySource):
        profile = query.profile
    else:
        assert isinstance(query, phx.geometry.AnalyticImplicitProfile)
        profile = query
    faces = int(np.asarray(association.target_global_ids).size)
    return source, profile, faces


def test_implicit_motion_revalidates_on_the_caller_adaptation_request(
    implicit_motion_source: tuple[Any, Any, Any],
) -> None:
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionScope,
    )

    source, profile, faces = implicit_motion_source
    (association,) = source.associations
    # The transfer's own allowance admits every face; only the caller's request
    # limits may refuse the revalidation queries.
    transfer = meshing.ImplicitAssociationTransfer(profile, maximum_queries=faces)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    coordinates = np.asarray(source.mesh.coordinates).copy()
    policy = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.NATIVE_BISECTION, association_transfer=transfer
    )
    lowered = policy.limits.native_limits()
    owner = NativeExecutionScope(lowered)
    with owner:
        accepted = meshing.advance_mesh_motion(
            monitor, source, coordinates, boundary_residual=0.0, adaptation_policy=policy
        )
        assert accepted.decision is Decision.ACCEPT_MOTION
        assert accepted.result is not None
        (carried,) = accepted.result.associations
        assert accepted.execution_evidence is not None
        record = accepted.execution_evidence
        record.require_valid()
        assert accepted.result.execution_evidence is record
        assert record.owner_id == accepted.advance_id
        assert source.execution_evidence is not None
        assert record.history_evidence == (source.execution_evidence,)
        assert record.preparation_evidence is None
        assert record.consumer_evidence is None
        assert int(np.asarray(record.source_preparation_work_units)) == 0
        assert int(np.asarray(record.source_preparation_geometry_queries)) == 0
        assert int(np.asarray(record.total_work_units)) == int(
            np.asarray(record.work_counters.construction_work)
        )
        assert int(np.asarray(record.total_geometry_queries)) == int(
            np.asarray(record.work_counters.source_queries)
        )
        # Current preparation is measured here, not replayed generation work.
        assert int(np.asarray(record.total_work_units)) > 0
        assert int(np.asarray(record.total_geometry_queries)) >= faces
        np.testing.assert_array_equal(
            np.sort(np.asarray(carried.target_global_ids)),
            np.sort(np.asarray(association.target_global_ids)),
        )
        assert carried.parent_association_id == association.association_id
        # One source-field evaluation per boundary face is charged to the caller.
        consumed = (
            lowered.maximum_geometry_queries
            - owner.remaining().remaining_geometry_queries
        )
        assert consumed >= faces
        assert int(np.asarray(record.total_geometry_queries)) == consumed
        assert source.execution_evidence is record.history_evidence[0]
    assert owner.completed_evidence().status is MeshcoreStatus.OK

    narrow = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=meshing.MeshingLimits(maximum_geometry_queries=faces - 1),
        association_transfer=transfer,
    )
    with pytest.raises(MeshcoreError) as refusal:
        meshing.advance_mesh_motion(
            monitor, source, coordinates, boundary_residual=0.0, adaptation_policy=narrow
        )
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_implicit_motion_moves_interior_and_rejects_boundary_residual_atomically(
    implicit_motion_source: tuple[Any, Any, Any],
) -> None:
    source, profile, faces = implicit_motion_source
    monitor = meshing.MeshMotionMonitor(source.mesh)
    coordinates = np.asarray(source.mesh.coordinates).copy()
    original = coordinates.copy()
    source_identity = source.result_id
    source_receipt = source.execution_evidence
    boundary = np.asarray(source.mesh.boundary_masks[0], dtype=np.bool_)
    interior = np.flatnonzero(~boundary)
    assert interior.size > 0
    coordinates[interior[0], 0] += 0.001 * monitor.length_scale
    policy = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        association_transfer=meshing.ImplicitAssociationTransfer(
            profile, maximum_queries=faces
        ),
    )
    moved = meshing.advance_mesh_motion(
        monitor, source, coordinates, boundary_residual=0.0, adaptation_policy=policy
    )
    assert moved.accepted and moved.decision is Decision.ACCEPT_MOTION
    assert moved.result is not None and moved.result.certification is not None
    moved.result.certification.require_passed()
    np.testing.assert_array_equal(moved.result.mesh.coordinates, coordinates)
    assert moved.execution_evidence is not None
    assert moved.execution_evidence.history_evidence == (source_receipt,)
    assert source_receipt is not None
    with pytest.raises(ValueError, match="another request owner"):
        meshing.MeshMotionAdvance(
            moved.decision,
            moved.assessments,
            result=moved.result,
            execution_evidence=source_receipt,
        )

    refused = meshing.advance_mesh_motion(
        monitor,
        source,
        coordinates,
        boundary_residual=1.0e-3,
        adaptation_policy=policy,
    )
    assert not refused.accepted and refused.decision is Decision.REJECT
    assert refused.result is None
    assert refused.execution_evidence is not None
    refused.execution_evidence.require_valid()
    assert refused.execution_evidence.history_evidence == (source_receipt,)
    assert source.result_id == source_identity
    assert source.execution_evidence is source_receipt
    np.testing.assert_array_equal(source.mesh.coordinates, original)


@pytest.mark.parametrize("resource", ("work", "memory", "clock"))
def test_implicit_motion_refuses_original_request_resources_without_mutation(
    implicit_motion_source: tuple[Any, Any, Any], resource: str
) -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionScope
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    source, profile, faces = implicit_motion_source
    coordinates = np.asarray(source.mesh.coordinates).copy()
    source_identity = source.result_id
    source_receipt = source.execution_evidence
    monitor = meshing.MeshMotionMonitor(source.mesh)
    limits = {
        "work": meshing.MeshingLimits(maximum_work_units=1),
        "memory": meshing.MeshingLimits(maximum_scratch_bytes=1),
        "clock": meshing.MeshingLimits(maximum_wall_seconds=1.0e-12),
    }[resource]
    policy = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=limits,
        association_transfer=meshing.ImplicitAssociationTransfer(
            profile, maximum_queries=faces
        ),
    )
    owner = NativeExecutionScope(limits.native_limits())
    error_type = CoordinateEnclosureResourceError if resource == "work" else MeshcoreError
    with pytest.raises(error_type) as refusal, owner:
        meshing.advance_mesh_motion(
            monitor,
            source,
            coordinates,
            boundary_residual=0.0,
            adaptation_policy=policy,
        )
    if resource == "work":
        error = refusal.value
        assert isinstance(error, CoordinateEnclosureResourceError)
        assert error.resource == "coefficient_work"
        assert error.limit == 1
        assert error.requested == 150
        assert error.completed == 0
        assert not error.admission
        assert error.phase_evidence is not None
        assert error.phase_evidence.work_units == 0
        # The coordinate preflight refuses before a native charge is attempted.
        assert owner.completed_evidence().status is MeshcoreStatus.OK
        assert owner.completed_evidence().work_counters.work_refusals == 0
    else:
        expected = (
            MeshcoreStatus.TIMEOUT
            if resource == "clock"
            else MeshcoreStatus.CAPACITY_EXCEEDED
        )
        assert isinstance(refusal.value, MeshcoreError)
        assert refusal.value.status is expected
        assert owner.completed_evidence().status is expected
    assert source.result_id == source_identity
    assert source.execution_evidence is source_receipt
    np.testing.assert_array_equal(source.mesh.coordinates, coordinates)


def test_implicit_motion_refuses_another_source_profile_without_mutation(
    implicit_motion_source: tuple[Any, Any, Any],
) -> None:
    source, profile, faces = implicit_motion_source
    coordinates = np.asarray(source.mesh.coordinates).copy()
    source_identity = source.result_id
    source_receipt = source.execution_evidence
    other = phx.geometry.AnalyticImplicitProfile(
        profile.geometry,
        source.coordinate_contract,
        source_id="another-implicit-motion-source",
        source_revision=profile.source_revision,
    )
    policy = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        association_transfer=meshing.ImplicitAssociationTransfer(
            other, maximum_queries=faces
        ),
    )
    with pytest.raises(ValueError, match="profile zero set"):
        meshing.advance_mesh_motion(
            meshing.MeshMotionMonitor(source.mesh),
            source,
            coordinates,
            boundary_residual=0.0,
            adaptation_policy=policy,
        )
    assert source.result_id == source_identity
    assert source.execution_evidence is source_receipt
    np.testing.assert_array_equal(source.mesh.coordinates, coordinates)


def test_policy_bearing_unassociated_motion_has_current_not_source_preparation() -> None:
    from phydrax._meshcore import meshcore_available
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    source = _certified(_POINTS, _CELLS)
    original_coordinates = np.asarray(source.mesh.coordinates).copy()
    assert source.execution_evidence is None
    monitor = meshing.MeshMotionMonitor(source.mesh)
    coordinates, _ = _shifted_center(source, 0.1)
    standalone = meshing.advance_mesh_motion(
        monitor, source, coordinates, boundary_residual=0.0
    )
    assert standalone.execution_evidence is None
    policy = meshing.MeshAdaptationPolicy(meshing.MeshAdaptationRoute.NATIVE_BISECTION)
    owned = meshing.advance_mesh_motion(
        monitor, source, coordinates, boundary_residual=0.0, adaptation_policy=policy
    )
    assert owned.accepted and standalone.accepted
    assert owned.execution_evidence is not None
    owned.execution_evidence.require_valid()
    assert owned.execution_evidence.history_evidence == ()
    assert owned.execution_evidence.preparation_evidence is None
    assert int(np.asarray(owned.execution_evidence.total_work_units)) > 0
    assert owned.assessments[0].assessment_id == standalone.assessments[0].assessment_id
    assert owned.result is not None and standalone.result is not None
    assert owned.result.result_id == standalone.result.result_id
    assert owned.advance_id == standalone.advance_id
    assert owned.result.execution_evidence is owned.execution_evidence
    narrow = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=meshing.MeshingLimits(maximum_work_units=1),
    )
    with pytest.raises(CoordinateEnclosureResourceError) as refusal:
        meshing.advance_mesh_motion(
            monitor, source, coordinates, boundary_residual=0.0, adaptation_policy=narrow
        )
    assert refusal.value.resource == "coefficient_work"
    assert refusal.value.limit == 1
    assert refusal.value.requested == 10
    assert refusal.value.completed == 0
    assert not refusal.value.admission
    assert refusal.value.phase_evidence is not None
    assert refusal.value.phase_evidence.work_units == 0
    assert source.execution_evidence is None
    np.testing.assert_array_equal(source.mesh.coordinates, original_coordinates)


def test_implicit_boundary_motion_preparation_uses_actual_request_and_refuses_missing_transfer() -> None:
    from phydrax._meshcore import meshcore_available
    from phydrax.meshing._motion import _carried, _motion_request
    from phydrax.units import METER

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    # Retain the established implicit boundary/tag fixture's source and controls.
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="design-sphere"
    ).compile(coordinate_unit=METER)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(9) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(np.asarray(((-1.4,) * 3, (1.4,) * 3), dtype=np.float64))
    native_source = meshing.NativeImplicitSource(
        geometry, grid, "design-sphere", "motion-boundary-r1"
    )
    boundary_scope = meshing.MeshingScope(
        native_source.source_id,
        native_source.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        "design-sphere-boundary",
        np.asarray((0,), dtype=np.int64),
    )
    limits = meshing.MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=60_000,
        maximum_faces=20_000,
        maximum_cells=20_000,
        maximum_connectivity_entries=240_000,
        maximum_data_bytes=8 * 1024**3,
        maximum_scratch_bytes=8 * 1024**3,
        maximum_work_units=320_000,
        maximum_cavity_cells=20_000,
        maximum_geometry_queries=1_280_000,
        maximum_wall_seconds=120.0,
    )
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(2, 3, meshing.CellFamilyPolicy(required=("triangle",))),
        boundary_scope,
        size_controls=(
            meshing.UniformSizeControl(
                boundary_scope, 2.8 / 8, strength=meshing.SizeControlStrength.SOFT
            ),
        ),
        limits=limits,
    )
    source = (
        meshing.NativeMeshingProvider(
            meshing.NativeMeshingOptions(
                "implicit_surface",
                implicit_policy=phx.geometry.ImplicitSurfacePolicy(
                    projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
                    maximum_intersection_pairs=640_000,
                ),
            )
        )
        .plan(
            native_source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute(geometry.state)
    )
    assert source.boundary is not None and source.certification is not None
    query = source.certification.request.source
    profile = (
        query.profile
        if isinstance(query, phx.geometry.ImplicitProjectionBoundarySource)
        else query
    )
    assert isinstance(profile, phx.geometry.AnalyticImplicitProfile)
    transfer = meshing.ImplicitAssociationTransfer(
        profile, maximum_queries=limits.maximum_geometry_queries
    )
    coordinates = np.asarray(source.mesh.coordinates).copy()
    source_identity = source.result_id
    moved = source.mesh.with_coordinates(
        coordinates, numeric_version="implicit-boundary-motion"
    )
    with _motion_request(limits) as (scope, budget):
        with budget.bind_ticket(scope.ticket_view()):
            carried = _carried(
                source,
                moved,
                transfer,
                limits=limits,
                scope=scope,
                budget=budget,
            )
    assert carried.boundary is not None and carried.certification is not None
    carried.certification.require_passed()
    assert carried.boundary.mesh.mesh_id == carried.mesh.mesh_id
    assert carried.boundary.metadata.source_id == source.boundary.metadata.source_id
    assert (
        carried.boundary.metadata.source_revision
        == source.boundary.metadata.source_revision
    )
    assert carried.boundary.metadata.cell_tags == source.boundary.metadata.cell_tags
    assert scope.completed_evidence().work_counters.construction_work > 0
    assert scope.completed_evidence().work_counters.source_queries > 0
    with _motion_request(limits) as (scope, budget):
        with budget.bind_ticket(scope.ticket_view()):
            with pytest.raises(ValueError, match="boundary model"):
                _carried(
                    source,
                    moved,
                    None,
                    limits=limits,
                    scope=scope,
                    budget=budget,
                )
    # Surface carrying is a shared preparation primitive, not a silent expansion
    # of the volume monitor's equal-dimension standalone contract.
    with pytest.raises(ValueError, match="equal topology and ambient dimensions"):
        meshing.MeshMotionMonitor(source.mesh)
    assert source.result_id == source_identity
    np.testing.assert_array_equal(source.mesh.coordinates, coordinates)


def test_repeated_motion_keeps_live_workflow_predecessor_as_history_only() -> None:
    from phydrax._meshcore import meshcore_available, NativeExecutionScope

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    source = _certified(_POINTS, _CELLS)
    policy = meshing.MeshAdaptationPolicy(meshing.MeshAdaptationRoute.NATIVE_BISECTION)
    with NativeExecutionScope(policy.limits.native_limits()) as owner:
        first = meshing.advance_mesh_motion(
            meshing.MeshMotionMonitor(source.mesh),
            source,
            _shifted_center(source, 0.1)[0],
            boundary_residual=0.0,
            adaptation_policy=policy,
        )
        assert first.result is not None and first.execution_evidence is not None
        second = meshing.advance_mesh_motion(
            meshing.MeshMotionMonitor(first.result.mesh),
            first.result,
            np.asarray(first.result.mesh.coordinates),
            boundary_residual=0.0,
            adaptation_policy=policy,
        )
        assert second.result is not None and second.execution_evidence is not None
        assert second.accepted
        assert second.execution_evidence.history_evidence == (first.execution_evidence,)
        assert second.execution_evidence.preparation_evidence is None
        assert int(np.asarray(second.execution_evidence.source_preparation_work_units)) == 0
        assert int(np.asarray(second.execution_evidence.total_work_units)) == int(
            np.asarray(second.execution_evidence.work_counters.construction_work)
        )
    actual = owner.completed_evidence().work_counters.construction_work
    assert actual >= int(np.asarray(first.execution_evidence.total_work_units)) + int(
        np.asarray(second.execution_evidence.total_work_units)
    )


def test_implicit_generation_then_motion_keeps_live_parent_spend_once(
    implicit_motion_generation: tuple[Any, Any],
) -> None:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.meshing._native_preparation import native_preparation_ledger

    native_source, specification = implicit_motion_generation
    with NativeExecutionScope(specification.limits.native_limits()) as owner:
        source = (
            meshing.NativeMeshingProvider(
                meshing.NativeMeshingOptions("implicit_restricted_delaunay")
            )
            .plan(
                native_source,
                specification,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
            )
            .execute()
        )
        assert source.execution_evidence is not None
        history = source.execution_evidence
        # Generation has settled into the still-live actual workflow request.
        ledger = native_preparation_ledger(owner)
        assert ledger.receipt_is_accounted(owner, history)
        before = owner.remaining()
        assert source.certification is not None
        query = source.certification.request.source
        profile = (
            query.profile
            if isinstance(query, phx.geometry.ImplicitProjectionBoundarySource)
            else query
        )
        assert isinstance(profile, phx.geometry.AnalyticImplicitProfile)
        policy = meshing.MeshAdaptationPolicy(
            meshing.MeshAdaptationRoute.NATIVE_BISECTION,
            limits=specification.limits,
            association_transfer=meshing.ImplicitAssociationTransfer(
                profile,
                maximum_queries=specification.limits.maximum_geometry_queries,
            ),
        )
        advance = meshing.advance_mesh_motion(
            meshing.MeshMotionMonitor(source.mesh),
            source,
            np.asarray(source.mesh.coordinates),
            boundary_residual=0.0,
            adaptation_policy=policy,
        )
        assert advance.accepted and advance.execution_evidence is not None
        record = advance.execution_evidence
        assert record.history_evidence == (history,)
        assert record.preparation_evidence is None
        assert int(np.asarray(record.source_preparation_work_units)) == 0
        assert int(np.asarray(record.source_preparation_geometry_queries)) == 0
        after = owner.remaining()
        # The parent spends current motion exactly once, plus the existing
        # ledger's one-unit retention of its newly completed request identity.
        assert before.remaining_work_units - after.remaining_work_units == (
            int(np.asarray(record.total_work_units)) + 1
        )
        assert before.remaining_geometry_queries - after.remaining_geometry_queries == int(
            np.asarray(record.total_geometry_queries)
        )
        assert after.remaining_seconds < before.remaining_seconds
        assert source.execution_evidence is history


def test_policy_bearing_relocation_keeps_original_assessments_and_current_receipt() -> None:
    from phydrax._meshcore import meshcore_available

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    source = _certified(_POINTS, _CELLS)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    folded, _ = _shifted_center(source, 0.6)
    policy = meshing.MeshAdaptationPolicy(meshing.MeshAdaptationRoute.NATIVE_BISECTION)
    owned = meshing.advance_mesh_motion(
        monitor, source, folded, boundary_residual=0.0, adaptation_policy=policy
    )
    assert owned.accepted and owned.decision is Decision.RELOCATE
    assert [value.decision for value in owned.assessments] == [
        Decision.RELOCATE,
        Decision.ACCEPT_MOTION,
    ]
    assert owned.result is not None and owned.execution_evidence is not None
    owned.execution_evidence.require_valid()
    assert owned.result.execution_evidence is owned.execution_evidence
    assert owned.execution_evidence.history_evidence == ()
    assert int(np.asarray(owned.execution_evidence.total_work_units)) > 0
    assert owned.relocation is not None and owned.relocation.accepted
    np.testing.assert_array_equal(
        owned.result.mesh.coordinates, owned.relocation.coordinates
    )


def test_relocation_preparation_and_certification_spend_the_actual_motion_ledger() -> None:
    from phydrax._meshcore import meshcore_available, MeshcoreError, MeshcoreStatus
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing._motion import _motion_request, _relocate

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    source = _certified(_POINTS, _CELLS)
    monitor = meshing.MeshMotionMonitor(source.mesh)
    coordinates, _ = _shifted_center(source, 0.6)
    original = np.asarray(source.mesh.coordinates).copy()
    moved = monitor.moved_mesh(coordinates)
    fixed = np.asarray(source.mesh.boundary_masks[0], dtype=np.bool_)
    limits = meshing.MeshingLimits()
    initial_audit = meshing.audit_cell_mesh(moved, CellGeometrySpec.affine(moved))
    assert not initial_audit.passed
    with _motion_request(limits) as (scope, budget):
        with budget.bind_ticket(scope.ticket_view()):
            before = budget.work_units
            relocated = _relocate(
                monitor,
                source,
                moved,
                fixed,
                "motion-owned-relocation",
                budget=budget,
            )
            assert relocated.accepted and relocated.result is not None
            assert relocated.untangling is not None and relocated.untangling.succeeded
            assert budget.work_units > before
            charged = budget.work_units - before
            coordinate_required = budget.required_work_units
    assert scope.completed_evidence().status is MeshcoreStatus.OK
    spent = scope.completed_evidence().work_counters.construction_work
    required = max(spent, coordinate_required)
    assert spent >= charged and required > 1
    assert relocated.status is meshing.MeshOptimizationStatus.OPTIMIZED
    assert relocated.result is not None
    assert meshing.audit_cell_mesh(
        relocated.result.mesh, relocated.result.geometry
    ).passed
    np.testing.assert_array_equal(relocated.coordinates[fixed], coordinates[fixed])

    # R covers the complete owning request: ended native construction includes
    # genuine work outside the coordinate ledger. Its dry-admission requirement
    # is also retained; neither source is substituted for the other.
    # Every other limit, numerical control and input is fixed.
    exact = meshing.MeshingLimits(maximum_work_units=required)
    with _motion_request(exact) as (scope, budget):
        with budget.bind_ticket(scope.ticket_view()):
            replay = _relocate(
                monitor,
                source,
                moved,
                fixed,
                "motion-owned-relocation",
                budget=budget,
            )
            assert budget.required_work_units == coordinate_required
    assert scope.completed_evidence().status is MeshcoreStatus.OK
    assert scope.completed_evidence().work_counters.construction_work == spent
    assert replay.status is relocated.status
    assert replay.optimization_id == relocated.optimization_id
    assert replay.result is not None
    assert meshing.audit_cell_mesh(replay.result.mesh, replay.result.geometry).passed
    np.testing.assert_array_equal(replay.coordinates, relocated.coordinates)
    np.testing.assert_array_equal(replay.coordinates[fixed], coordinates[fixed])

    # R-1 must refuse the same actual preparation/publication, atomically.
    # Dropping the optimizer's budget would incorrectly use host-derived quotas.
    narrow = meshing.MeshingLimits(maximum_work_units=required - 1)
    with pytest.raises((CoordinateEnclosureResourceError, MeshcoreError)) as refusal:
        with _motion_request(narrow) as (scope, budget):
            with budget.bind_ticket(scope.ticket_view()):
                _relocate(
                    monitor,
                    source,
                    moved,
                    fixed,
                    "motion-owned-relocation",
                    budget=budget,
                )
    if isinstance(refusal.value, CoordinateEnclosureResourceError):
        assert refusal.value.resource == "coefficient_work"
        assert refusal.value.limit == required - 1
        assert refusal.value.requested > refusal.value.limit
        assert refusal.value.completed <= refusal.value.limit
        assert refusal.value.phase_evidence is not None
        assert not refusal.value.admission
        # A preflight refusal need not manufacture a native refusal counter.
        assert scope.completed_evidence().status is MeshcoreStatus.OK
        assert scope.completed_evidence().work_counters.work_refusals == 0
    else:
        assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
        assert scope.completed_evidence().status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert source.execution_evidence is None
    np.testing.assert_array_equal(source.mesh.coordinates, original)
