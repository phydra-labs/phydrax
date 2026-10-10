from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()


def _grid(count: Any = 9) -> Any:
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))


def _implicit_case(
    *,
    size_strength: Any,
    limits: Any = None,
    periodic_constraints: Any = (),
    fidelity: Any = None,
) -> Any:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.75, feature_id="sphere").compile()
    scope = M.MeshingScope(
        "sphere",
        "r1",
        M.MeshingEntityKind.GEOMETRY,
        2,
        "sphere-boundary",
        np.asarray((0,), dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(M.UniformSizeControl(scope, 0.3, strength=size_strength),),
        protected_features=(
            ()
            if fidelity is None
            else (
                M.ProtectedFeature(
                    scope, M.FeatureKind.SURFACE, maximum_deviation=fidelity
                ),
            )
        ),
        periodic_constraints=periodic_constraints,
        limits=limits,
    )
    return geometry, scope, specification


def _stage(result: Any, kind: Any) -> Any:
    (stage,) = (value for value in result.trace.stages if value.stage is kind)
    return stage


def _implicit_plan(geometry: Any, specification: Any) -> Any:
    options = M.NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=500_000,
        ),
    )
    return M.NativeMeshingProvider(options).plan(
        M.NativeImplicitSource(geometry, _grid(), "sphere", "r1"),
        specification,
        coordinate_contract=_SI,
    )


def test_native_implicit_route_publishes_closed_audited_surface_with_gradients() -> None:
    geometry, _, specification = _implicit_case(size_strength=M.SizeControlStrength.SOFT)
    plan = _implicit_plan(geometry, specification)
    result = plan.execute()
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))

    def vertex_sum(radius: Any) -> Any:
        state = geometry.state.replace_at(radius_index, radius)
        return jnp.sum(plan.prepared.realize(state).proposed_vertices)

    derivative = jax.grad(vertex_sum)(jnp.asarray(0.75))

    assert result.audit.passed
    assert result.associations[0].complete
    assert "wall_seconds" in result.runtime.enforced_limits
    assert "cavity_cells" in result.runtime.unenforced_limits
    assert result.derivative_mode is M.MeshingDerivativeMode.FIXED_ROUTE_PIECEWISE
    assert jnp.isfinite(derivative)
    assert derivative != 0.0


def test_native_implicit_route_certifies_two_sided_sphere_fidelity() -> None:
    geometry, _, specification = _implicit_case(
        size_strength=M.SizeControlStrength.SOFT, fidelity=0.2
    )
    result = _implicit_plan(geometry, specification).execute()
    fidelity = result.certification.fidelity
    points = np.asarray(result.mesh.coordinates)
    faces = np.asarray(result.mesh.blocks[0].vertices)
    # Independent oracle: the deepest point of a flat face inscribed in the
    # sphere is at least the distance of its centroid from the sphere.
    centroid_gap = np.max(
        np.abs(np.linalg.norm(np.mean(points[faces], axis=1), axis=1) - 0.75)
    )

    assert result.certification.passed
    assert _stage(result, M.MeshingStageKind.CERTIFICATION).status == "passed"
    assert fidelity.mesh_to_source_semantics == "certified"
    assert fidelity.source_to_mesh_semantics == "certified"
    assert centroid_gap <= fidelity.mesh_to_source_upper <= 0.2
    assert fidelity.source_to_mesh_upper <= 0.2
    assert result.trace.binding.source_revision == "r1"


def test_native_implicit_route_refuses_unmet_hard_fidelity_at_certification() -> None:
    geometry, _, specification = _implicit_case(
        size_strength=M.SizeControlStrength.SOFT, fidelity=1.0e-4
    )
    with pytest.raises(M.MeshingFailure) as failure:
        _implicit_plan(geometry, specification).execute()
    evidence = failure.value.evidence
    assert failure.value.category is M.MeshingFailureCategory.AUDIT_FAILED
    assert evidence.stage == "certification"
    assert dict(evidence.requested)["mesh_to_source_deviation"] == 1.0e-4
    assert dict(evidence.achieved)["mesh_to_source_deviation"] > 1.0e-4


def _adaptive_plan(geometry: Any, specification: Any, policy: Any) -> Any:
    options = M.NativeMeshingOptions("implicit_surface", implicit_policy=policy)
    return M.NativeMeshingProvider(options).plan(
        M.NativeImplicitSource(geometry, _grid(), "sphere", "r1"),
        specification,
        coordinate_contract=_SI,
    )


def test_native_adaptive_implicit_route_publishes_certified_surface() -> None:
    geometry, _, specification = _implicit_case(
        size_strength=M.SizeControlStrength.SOFT, fidelity=0.2
    )
    assert isinstance(
        M.NativeMeshingOptions("implicit_surface").implicit_policy,
        phx.geometry.implicit.AdaptiveImplicitSurfacePolicy,
    )
    plan = _adaptive_plan(
        geometry,
        specification,
        phx.geometry.implicit.AdaptiveImplicitSurfacePolicy(maximum_level=6),
    )
    result = plan.execute()
    points = np.asarray(result.mesh.coordinates)

    assert result.certification.passed
    assert result.derivative_mode is M.MeshingDerivativeMode.NONDIFFERENTIABLE
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 0.75, atol=0.2)
    with pytest.raises(ValueError, match="design states"):
        plan.execute(geometry.state)


def test_native_adaptive_implicit_route_refuses_sampled_discovery_under_hard_fidelity() -> (
    None
):
    geometry, _, specification = _implicit_case(
        size_strength=M.SizeControlStrength.SOFT, fidelity=0.2
    )
    plan = _adaptive_plan(
        geometry,
        specification,
        phx.geometry.implicit.AdaptiveImplicitSurfacePolicy(
            enclosure="sampled", maximum_level=6
        ),
    )
    with pytest.raises(M.MeshingFailure) as failure:
        plan.execute()
    evidence = failure.value.evidence
    assert failure.value.category is M.MeshingFailureCategory.AUDIT_FAILED
    assert evidence.stage == "certification"
    assert dict(evidence.achieved)["accuracy_certified"] == 0.0


def test_native_implicit_route_refuses_unenforced_and_exhausted_requests() -> None:
    geometry, scope, _ = _implicit_case(size_strength=M.SizeControlStrength.SOFT)
    periodic = M.PeriodicConstraint(scope, scope, np.eye(4))
    _, _, periodic_specification = _implicit_case(
        size_strength=M.SizeControlStrength.SOFT, periodic_constraints=(periodic,)
    )
    with pytest.raises(M.MeshingFailure) as unsupported:
        _implicit_plan(geometry, periodic_specification)
    assert unsupported.value.category is M.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    assert "periodic constraints" in str(unsupported.value)

    _, _, limited = _implicit_case(
        size_strength=M.SizeControlStrength.SOFT,
        limits=M.MeshingLimits(maximum_vertices=3),
    )
    with pytest.raises(M.MeshingFailure) as resource:
        _implicit_plan(geometry, limited)
    assert resource.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(resource.value.evidence.requested)["maximum_vertices"] == 3.0

    _, _, hard = _implicit_case(size_strength=M.SizeControlStrength.HARD)
    with pytest.raises(M.MeshingFailure) as compliance:
        _implicit_plan(geometry, hard).execute()
    assert compliance.value.category is M.MeshingFailureCategory.COMPLIANCE_FAILED
    assert compliance.value.evidence.achieved

    _, _, timed = _implicit_case(
        size_strength=M.SizeControlStrength.SOFT,
        limits=M.MeshingLimits(maximum_wall_seconds=1.0e-6),
    )
    with pytest.raises(M.MeshingFailure) as timeout:
        _implicit_plan(geometry, timed).execute()
    assert timeout.value.category is M.MeshingFailureCategory.TIMED_OUT


def test_native_implicit_route_volume_gradient_matches_finite_differences() -> None:
    geometry, _, specification = _implicit_case(size_strength=M.SizeControlStrength.SOFT)
    plan = _implicit_plan(geometry, specification)
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))

    def enclosed_volume(radius: Any) -> Any:
        result = plan.execute(geometry.state.replace_at(radius_index, radius))
        faces = result.geometry.geometry_dofs[0]
        corners = result.geometry.coordinates[faces]
        return (
            jnp.sum(
                jnp.sum(corners[:, 0] * jnp.cross(corners[:, 1], corners[:, 2]), axis=-1)
            )
            / 6.0
        )

    radius = jnp.asarray(0.75)
    step = 1.0e-4
    derivative = jax.grad(enclosed_volume)(radius)
    central = (enclosed_volume(radius + step) - enclosed_volume(radius - step)) / (
        2.0 * step
    )
    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(derivative, central, rtol=1.0e-5)


def _plate(
    *,
    minimum_angle: Any = np.radians(25.0),
    target_size: float = 0.2,
    limits: Any = None,
) -> Any:
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(
            [
                [0, 0],
                [2, 0],
                [2, 2],
                [0, 2],
                [0.8, 0.8],
                [0.8, 1.2],
                [1.2, 1.2],
                [1.2, 0.8],
            ],
            dtype=np.float64,
        ),
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        feature_id="plate",
    )
    crack = phx.geometry.SegmentMesh(
        jnp.asarray([[0.2, 0.2], [0.5, 1.7]]), jnp.asarray([[0, 1]]), source_id="crack"
    )
    source = M.NativePlanarSource(region, "r1", embedded=crack)
    scope = M.MeshingScope(
        "plate", "r1", M.MeshingEntityKind.GEOMETRY, 2, "plate-region", np.asarray([0])
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            M.UniformSizeControl(
                scope,
                target_size,
                maximum_size=1.25 * target_size,
                strength=M.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=M.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        quality_target=M.MeshQualityTarget(minimum_angle=minimum_angle),
        limits=limits,
    )
    return source, specification


def _planar_provider() -> Any:
    return M.NativeMeshingProvider(M.NativeMeshingOptions("planar_constrained_delaunay"))


def test_native_planar_route_meshes_holes_embedded_segments_and_quality() -> None:
    source, specification = _plate()
    result = (
        _planar_provider().plan(source, specification, coordinate_contract=_SI).execute()
    )
    mesh = result.mesh
    points = np.asarray(mesh.coordinates)
    triangles = np.asarray(mesh.blocks[0].vertices)
    corners = points[triangles]
    areas = 0.5 * (
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
    )
    # Independent oracle: the plate minus the square hole.
    np.testing.assert_allclose(np.sum(areas), 4.0 - 0.16, rtol=0.0, atol=1.0e-12)
    assert np.all(areas > 0.0)
    assert result.quality.minimum_angle >= np.radians(25.0)
    edge_association = result.associations[1]
    embedded = np.asarray(
        [value == "r1:edge:8" for value in edge_association.source_entity_ids],
        dtype=np.bool_,
    )
    edges = np.asarray(mesh.connectivity.edges)
    rows = np.flatnonzero(
        np.isin(
            np.asarray(mesh.entity_set(1).entity_ids),
            np.asarray(edge_association.target_global_ids)[embedded],
        )
    )
    lengths = np.linalg.norm(points[edges[rows, 1]] - points[edges[rows, 0]], axis=1)
    np.testing.assert_allclose(np.sum(lengths), np.hypot(0.3, 1.5), rtol=1.0e-12)
    residuals = np.asarray(edge_association.residuals)
    assert not edge_association.exact
    assert np.all(residuals[~embedded] == 0.0)
    assert 0.0 < np.max(residuals[embedded]) < 1.0e-15
    assert np.all(np.abs(np.asarray(edge_association.orientations)) == 1)
    all_edges = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    assert np.max(all_edges) <= 0.25
    assert [patch.name for patch in result.patches] == ["loop:0", "loop:1"]
    assert [label.name for label in result.labels] == ["embedded"]
    coverage = result.certification.coverage
    assert result.certification.passed
    assert coverage.status == "certified"
    np.testing.assert_allclose(coverage.achieved_region_measures, (3.84,), rtol=1e-12)


def _square_hole_distance(points: Any) -> Any:
    """Exact distance to the boundary of the hole [0.8, 1.2]^2."""

    offset = np.abs(points - 1.0) - 0.2
    outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
    return np.where(outside > 0.0, outside, -np.max(offset, axis=1))


def test_native_planar_route_grades_patch_size_into_region_size() -> None:
    source, _ = _plate()
    scope = M.MeshingScope(
        "plate", "r1", M.MeshingEntityKind.GEOMETRY, 2, "plate-region", np.asarray([0])
    )
    hole = M.MeshingScope(
        "plate", "r1", M.MeshingEntityKind.GEOMETRY, 1, "plate-edges", np.arange(4, 8)
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            M.UniformSizeControl(
                scope, 0.3, maximum_growth_rate=1.5, strength=M.SizeControlStrength.HARD
            ),
            M.UniformSizeControl(hole, 0.05, strength=M.SizeControlStrength.HARD),
        ),
        size_compliance=M.SizeCompliancePolicy(
            relative_tolerance=0.6, target_statistics=("p50",)
        ),
        quality_target=M.MeshQualityTarget(minimum_angle=np.radians(20.0)),
    )
    result = (
        _planar_provider().plan(source, specification, coordinate_contract=_SI).execute()
    )
    points = np.asarray(result.mesh.coordinates)
    edges = np.asarray(result.mesh.connectivity.edges)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    middle = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    distance = _square_hole_distance(middle)
    # Independent size oracle: 0.05 on the hole, growing at rate 1.5.
    allowed = np.minimum(0.3, 0.05 + 0.5 * np.abs(distance))
    on_hole = np.abs(distance) <= 1.0e-12

    assert np.all(lengths <= allowed + 1.0e-12)
    assert np.max(lengths[on_hole]) <= 0.05 + 1.0e-12
    assert np.median(lengths[np.abs(distance) > 0.5]) > 0.1
    assert result.certification.passed


def test_native_planar_route_refuses_refinement_beyond_scratch_budget() -> None:
    source, specification = _plate(
        target_size=0.001,
        limits=M.MeshingLimits(maximum_scratch_bytes=1_000_000),
    )
    with pytest.raises(M.MeshingFailure) as failure:
        _planar_provider().plan(source, specification, coordinate_contract=_SI).execute()
    evidence = failure.value.evidence
    assert failure.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert evidence.stage == "surface_meshing"
    assert dict(evidence.requested)["maximum_scratch_bytes"] == 1_000_000
    assert dict(evidence.achieved)["scratch_bytes"] > 1_000_000


def test_native_publication_refuses_mesh_missing_its_declared_domain() -> None:
    from phydrax.geometry._mesh_certificates import PiecewiseLinearDomain
    from phydrax.meshing.providers._native_publication import (
        NativeCertificationRequest,
        publish_native_result,
    )

    source, specification = _plate()
    # A valid unit-square mesh published against the 2 x 2 plate outline.
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        numeric_version="r1",
    )
    domain = PiecewiseLinearDomain(
        np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64),
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]]),
        np.tile(np.asarray([[0, -1]]), (4, 1)),
        ("r1:region:0",),
        source_id="plate",
    )
    with pytest.raises(M.MeshingFailure) as failure:
        publish_native_result(
            mesh,
            _SI,
            M.MeshingComplianceReport(specification.specification_id),
            (
                M.MeshingStageReport(
                    M.MeshingStageKind.SURFACE_MESHING, M.MeshingStageStatus.PASSED
                ),
            ),
            M.NativeMeshingProvider.info(),
            {"kind": "corrupted-publication"},
            NativeCertificationRequest(
                M.MeshCertificationSchedule("volume_plc"),
                source.source_id,
                "r1",
                specification.limits,
                domain=domain,
                cell_regions=np.zeros((2,), dtype=np.int64),
            ),
            audit_policy=M.CellMeshAuditPolicy(
                watertight_boundary=M.CellMeshAuditDisposition.REJECT
            ),
            derivative_mode=M.MeshingDerivativeMode.NONDIFFERENTIABLE,
            enforced_limits=(),
            unenforced_limits=(),
        )
    evidence = failure.value.evidence
    assert failure.value.category is M.MeshingFailureCategory.AUDIT_FAILED
    assert evidence.stage == "certification"
    assert dict(evidence.requested)["region_measure[r1:region:0]"] == 4.0
    assert dict(evidence.achieved)["region_measure[r1:region:0]"] == 1.0


def test_native_planar_route_reports_unmet_quality_with_failure_evidence() -> None:
    source, specification = _plate(
        minimum_angle=np.radians(45.0),
        target_size=1.0,
        limits=M.MeshingLimits(maximum_vertices=30),
    )
    with pytest.raises(M.MeshingFailure) as failure:
        _planar_provider().plan(source, specification, coordinate_contract=_SI).execute()
    evidence = failure.value.evidence
    assert failure.value.category is M.MeshingFailureCategory.COMPLIANCE_FAILED
    assert evidence.stage == "specification_compliance"
    requested = dict(evidence.requested)
    achieved = dict(evidence.achieved)
    assert requested["minimum_angle"] == np.radians(45.0)
    assert achieved["minimum_angle"] < requested["minimum_angle"]


def test_native_provider_refuses_foreign_scope_and_route_kinds() -> None:
    source, specification = _plate()
    stale = M.NativePlanarSource(source.region, "r2", embedded=source.embedded)
    with pytest.raises(M.MeshingFailure) as scope:
        _planar_provider().validate(stale, specification)
    assert scope.value.category is M.MeshingFailureCategory.SCOPE_RESOLUTION_FAILED
    curve_provider = M.NativeMeshingProvider(M.NativeMeshingOptions("curve_arc_length"))
    with pytest.raises(TypeError, match="NativeCurveSource"):
        curve_provider.validate(source, specification)
    with pytest.raises(ValueError):
        M.NativeMeshingOptions(
            "planar_constrained_delaunay", curve_schedule=M.NativeCurveSchedule()
        )


def _curve_spec(scope: Any, junctions: Any, **arguments: Any) -> Any:
    return M.CurveMeshingSpec(
        M.CellMeshingTarget(1, 2, M.CellFamilyPolicy(required=("interval",))),
        scope,
        junctions=junctions,
        **arguments,
    )


def _curve_plan(curves: Any, specification: Any) -> Any:
    return M.NativeMeshingProvider(M.NativeMeshingOptions("curve_arc_length")).plan(
        M.NativeCurveSource(curves, "r1"), specification, coordinate_contract=_SI
    )


def test_native_curve_route_closes_circle_charts_within_certified_fidelity() -> None:
    atlas = (
        phx.geometry.Circle((0.0, 0.0), 1.0, feature_id="circle").compile().boundary_atlas
    )
    scope = M.MeshingScope(
        atlas.source_id, "r1", M.MeshingEntityKind.GEOMETRY, 1, "arcs", np.arange(4)
    )
    feature = M.ProtectedFeature(scope, M.FeatureKind.CURVE, maximum_deviation=1.0e-3)
    specification = _curve_spec(
        scope,
        tuple(
            M.CurveJunction(f"j{i}", ((i, "end"), ((i + 1) % 4, "start")))
            for i in range(4)
        ),
        size_controls=(
            M.UniformSizeControl(scope, 0.3, strength=M.SizeControlStrength.SOFT),
        ),
        protected_features=(feature,),
    )
    result = _curve_plan(atlas, specification).execute()
    points = np.asarray(result.mesh.coordinates)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    # A closed loop has one interval per vertex and all vertices on the circle.
    assert cells.shape[0] == points.shape[0]
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1.0e-12)
    chords = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    sagitta = 1.0 - np.sqrt(1.0 - 0.25 * chords**2)
    assert np.max(sagitta) <= 1.0e-3
    assert np.max(chords) <= 0.3
    assert sorted(label.name for label in result.labels) == ["j0", "j1", "j2", "j3"]
    association = result.associations[0]
    assert association.association_kind is M.GeometryAssociationKind.CURVE
    # Independent oracle: the exact sagitta of each chord is the true two-sided
    # deviation of its arc; every certified bound must dominate it.
    bounds = np.asarray(association.residuals)
    assert np.all(np.sort(bounds) >= np.sort(sagitta))
    achieved = dict(result.compliance.achieved)
    key = f"protected:{feature.feature_id}"
    assert achieved[f"{key}:chord_deviation_certified"] == 1.0
    assert np.max(sagitta) <= achieved[f"{key}:chord_deviation_upper"] <= 1.0e-3
    assert achieved[f"{key}:chord_deviation_upper"] <= 1.5 * np.max(sagitta)
    assert result.certification.passed


@pytest.mark.parametrize(
    ("size", "intervals"),
    ((1.0, 8), (0.75, 12), (0.5, 16)),
    ids=("symmetric-half", "thirds-neighbor", "symmetric-quarter"),
)
def test_native_curve_route_places_symmetric_density_targets_at_table_knots(
    size: float,
    intervals: int,
) -> None:
    atlas = (
        phx.geometry.Circle((0.0, 0.0), 1.0, feature_id="symmetric-circle")
        .compile()
        .boundary_atlas
    )
    scope = M.MeshingScope(
        atlas.source_id,
        "r1",
        M.MeshingEntityKind.GEOMETRY,
        1,
        "arcs",
        np.arange(4),
    )
    specification = _curve_spec(
        scope,
        tuple(
            M.CurveJunction(f"j{i}", ((i, "end"), ((i + 1) % 4, "start")))
            for i in range(4)
        ),
        size_controls=(
            M.UniformSizeControl(scope, size, strength=M.SizeControlStrength.SOFT),
        ),
    )
    result = _curve_plan(atlas, specification).execute()
    points = np.asarray(result.mesh.coordinates)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    assert cells.shape == (intervals, 2)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
    angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    np.testing.assert_allclose(
        np.sort(angles), np.arange(intervals) * 2.0 * np.pi / intervals, atol=1e-12
    )
    assert result.certification.passed


def test_native_sphere_meridian_density_targets_follow_exact_source() -> None:
    contract = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
    sphere = phx.geometry.brep_sphere(
        1.0,
        coordinate_contract=contract,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    atlas = phx.geometry.MeshingDomain.from_brep(sphere).curve_atlas
    scope = M.MeshingScope(
        atlas.source_id,
        sphere.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        1,
        "arcs",
        np.arange(atlas.num_charts, dtype=np.int64),
    )
    specification = M.CurveMeshingSpec(
        M.CellMeshingTarget(1, 3, M.CellFamilyPolicy(required=("interval",))),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, 1.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    provider = M.NativeMeshingProvider(M.NativeMeshingOptions("curve_arc_length"))
    result = provider.plan(
        M.NativeCurveSource(atlas, sphere.source_revision),
        specification,
        coordinate_contract=contract,
    ).execute()
    points = np.asarray(result.mesh.coordinates)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    assert cells.shape == (4, 2)
    assert points.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
    chords = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    np.testing.assert_allclose(chords, 2.0 * np.sin(np.pi / 8.0), atol=1e-12)
    certification = result.certification
    if certification is None:
        pytest.fail("The native meridian consumer must publish its certification.")
    assert certification.passed


@pytest.mark.parametrize(
    ("size", "intervals"),
    ((1.0, 6), (0.75, 8)),
    ids=("sixths-across-source-nodes", "eighths-neighbor"),
)
def test_native_intersection_curve_density_targets_follow_both_sphere_sources(
    size: float,
    intervals: int,
) -> None:
    from phydrax.geometry._meshing_domain import _PhysicalCurveMap

    first = phx.geometry.SurfaceRegion(
        phx.geometry.SpherePatch(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1.0
        ),
        ((0.0, -np.pi / 2.0), (2.0 * np.pi, np.pi / 2.0)),
    )
    second = phx.geometry.SurfaceRegion(
        phx.geometry.SpherePatch(
            (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1.0
        ),
        ((0.0, -np.pi / 2.0), (2.0 * np.pi, np.pi / 2.0)),
    )
    intersection = phx.geometry.intersect_surface_regions(first, second)
    assert intersection.complete
    (curve,) = intersection.curves
    assert curve.fully_certified and curve.closed
    atlas = phx.geometry.BoundaryAtlas(
        _PhysicalCurveMap(
            (curve,),
            np.asarray(((0.0, curve.num_charts),), dtype=np.float64),
        ),
        source_entity_ids=jnp.asarray((0,), dtype=jnp.int32),
        source_id=curve.branch_id,
    )
    scope = M.MeshingScope(
        atlas.source_id,
        curve.branch_id,
        M.MeshingEntityKind.GEOMETRY,
        1,
        "intersection-branch",
        np.asarray((0,), dtype=np.int64),
    )
    specification = M.CurveMeshingSpec(
        M.CellMeshingTarget(1, 3, M.CellFamilyPolicy(required=("interval",))),
        scope,
        junctions=(M.CurveJunction("closed-source-branch", ((0, "start"), (0, "end"))),),
        size_controls=(
            M.UniformSizeControl(scope, size, strength=M.SizeControlStrength.SOFT),
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("curve_arc_length"))
        .plan(
            M.NativeCurveSource(atlas, curve.branch_id),
            specification,
            coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        )
        .execute()
    )
    points = np.asarray(result.mesh.coordinates)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    assert cells.shape == (intervals, 2)
    assert points.shape == (intervals, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(
        np.linalg.norm(points - np.asarray((0.0, 1.0, 0.0)), axis=1),
        1.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(points[:, 1], 0.5, atol=1e-10)
    chords = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    np.testing.assert_allclose(
        chords, 2.0 * np.sqrt(0.75) * np.sin(np.pi / intervals), atol=1e-10
    )
    (association,) = result.associations
    assert association.source_id == curve.branch_id
    assert association.source_revision == curve.branch_id
    assert association.source_entity_ids == (f"{curve.branch_id}:curve:0",) * intervals
    assert np.all(np.asarray(association.resolved))
    source_parameters = np.asarray(association.parameters)[:, 0] * curve.num_charts
    source_points = curve.evaluate(jnp.asarray(source_parameters))
    assert np.all(np.isfinite(np.asarray(source_points.parameter_bound)))
    assert np.max(np.asarray(source_points.gap)) <= 1e-10
    certification = result.certification
    if certification is None:
        pytest.fail("The native intersection consumer must publish its certification.")
    assert certification.passed


def _tee() -> tuple[Any, Any]:
    segments = phx.geometry.SegmentMesh(
        jnp.asarray([[0, 0], [1, 0], [0, 1], [-1, 0]], dtype=jnp.float64),
        jnp.asarray([[0, 1], [0, 2], [0, 3]]),
        source_id="tee",
    )
    scope = M.MeshingScope(
        "tee", "r1", M.MeshingEntityKind.GEOMETRY, 1, "tee-edges", np.arange(3)
    )
    return segments, scope


def test_native_curve_route_shares_declared_network_junction() -> None:
    segments, scope = _tee()
    specification = _curve_spec(
        scope,
        (M.CurveJunction("hub", ((0, "start"), (1, "start"), (2, "start"))),),
        size_controls=(
            M.UniformSizeControl(scope, 0.25, strength=M.SizeControlStrength.SOFT),
        ),
    )
    result = _curve_plan(segments, specification).execute()
    cells = np.asarray(result.mesh.blocks[0].vertices)
    points = np.asarray(result.mesh.coordinates)
    counts = np.bincount(cells.reshape(-1), minlength=points.shape[0])
    assert cells.shape[0] == 12
    assert np.count_nonzero(counts == 3) == 1
    np.testing.assert_array_equal(points[np.argmax(counts)], (0.0, 0.0))
    (hub,) = result.labels
    assert hub.name == "hub"
    assert result.certification.passed
    assert _stage(result, M.MeshingStageKind.CERTIFICATION).status == "passed"


def test_native_curve_route_never_joins_undeclared_or_distant_endpoints() -> None:
    segments, scope = _tee()
    free = _curve_spec(
        scope,
        (),
        size_controls=(
            M.UniformSizeControl(scope, 0.5, strength=M.SizeControlStrength.SOFT),
        ),
    )
    with pytest.raises(M.MeshingFailure) as coincident:
        _curve_plan(segments, free).execute()
    assert coincident.value.category is M.MeshingFailureCategory.AUDIT_FAILED

    distant = _curve_spec(
        scope,
        (M.CurveJunction("gap", ((0, "end"), (1, "end"))),),
        size_controls=(
            M.UniformSizeControl(scope, 0.5, strength=M.SizeControlStrength.SOFT),
        ),
    )
    with pytest.raises(M.MeshingFailure) as illegal:
        _curve_plan(segments, distant)
    assert illegal.value.category is M.MeshingFailureCategory.INVALID_SOURCE
    assert dict(illegal.value.evidence.achieved)["junction_gap"] > 1.0


def test_native_curve_route_refuses_fidelity_beyond_work_budget() -> None:
    atlas = (
        phx.geometry.Circle((0.0, 0.0), 1.0, feature_id="circle").compile().boundary_atlas
    )
    scope = M.MeshingScope(
        atlas.source_id, "r1", M.MeshingEntityKind.GEOMETRY, 1, "arcs", np.arange(4)
    )
    specification = _curve_spec(
        scope,
        (),
        size_controls=(
            M.UniformSizeControl(scope, 0.5, strength=M.SizeControlStrength.SOFT),
        ),
        protected_features=(
            M.ProtectedFeature(scope, M.FeatureKind.CURVE, maximum_deviation=1.0e-9),
        ),
        limits=M.MeshingLimits(maximum_work_units=64),
    )
    with pytest.raises(M.MeshingFailure) as failure:
        _curve_plan(atlas, specification).execute()
    assert failure.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.value.evidence.requested)["maximum_work_units"] == 64.0


def test_native_periodic_provider_publishes_a_closed_degree_one_torus() -> None:
    orbits = M.PeriodicPointOrbits(
        np.asarray([[0.3, 0.6]], dtype=np.float64),
        phx.discretization.PeriodicCell(np.eye(2, dtype=np.float64)),
    )
    source = M.NativePeriodicSource(orbits, "torus", "r1")
    scope = M.MeshingScope(
        "torus",
        "r1",
        M.MeshingEntityKind.GEOMETRY,
        2,
        "torus-region",
        np.asarray([0], dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            M.UniformSizeControl(scope, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    provider = M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay"))
    result = provider.plan(source, specification, coordinate_contract=_SI).execute()
    periodic = result.mesh.periodic_topology
    assert periodic is not None
    corners = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    signed_areas = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 2.0

    # The fundamental square is covered once by positive lifted triangles.
    assert np.all(signed_areas > 0.0)
    np.testing.assert_allclose(np.sum(signed_areas), 1.0, rtol=1.0e-12)
    assert periodic.euler_characteristic == 0
    assert len(set(periodic.entity_keys(1))) == 3
    assert result.audit.passed
    assert result.certification is not None
    assert result.certification.passed
    assert result.trace.binding is not None
    assert result.trace.binding.source_revision == "r1"


def test_native_polyhedral_provider_covers_cube_for_finite_volume_consumers() -> None:
    vertices = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    complex_ = M.PiecewiseLinearComplex(
        vertices,
        loops,
        np.zeros(6, dtype=np.int32),
        np.asarray([[-1, 0]], dtype=np.int32),
        ("solid",),
    )
    source = M.NativePlcSource(complex_, "cube", "r1")
    scope = M.MeshingScope(
        "cube",
        "r1",
        M.MeshingEntityKind.GEOMETRY,
        2,
        "cube-facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("polyhedron",))),
        scope,
        M.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            M.UniformSizeControl(scope, 4.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    options = M.NativeMeshingOptions(
        "plc_restricted_power",
        polyhedral_schedule=M.NativePolyhedralSchedule(maximum_refinement_steps=0),
    )
    result = (
        M.NativeMeshingProvider(options)
        .plan(source, specification, coordinate_contract=_SI)
        .execute()
    )
    geometry = phx.discretization.prepare_polyhedral_finite_volume_geometry(result.mesh)
    volumes = np.asarray(geometry.cell_volumes)
    exterior = np.asarray(geometry.neighbor_cells) < 0

    assert np.all(volumes > 0.0)
    np.testing.assert_allclose(np.sum(volumes), 1.0, rtol=1.0e-12)
    np.testing.assert_allclose(
        np.sum(np.asarray(geometry.face_measures)[exterior]), 6.0, rtol=1.0e-12
    )
    assert all(block.cell_kind == "polyhedron" for block in result.mesh.blocks)
    assert {zone.name for zone in result.zones} == {"solid"}
    assert result.audit.passed
    assert result.certification is not None
    assert result.certification.passed


def test_native_route_selector_refuses_external_engine_vocabulary() -> None:
    with pytest.raises(ValueError):
        M.NativeMeshingOptions("tetgen")  # ty: ignore[invalid-argument-type]


def test_native_provider_constructor_requires_typed_options() -> None:
    with pytest.raises(TypeError, match="NativeMeshingOptions"):
        M.NativeMeshingProvider("native")  # ty: ignore[invalid-argument-type]


def test_native_planar_source_refuses_an_implicit_geometry_owner() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.75, feature_id="sphere").compile()
    with pytest.raises(TypeError, match="PlanarMeshRegion"):
        M.NativePlanarSource(geometry, "r1")  # ty: ignore[invalid-argument-type]


def test_native_plan_requires_an_explicit_physical_coordinate_contract() -> None:
    source, specification = _plate()
    with pytest.raises(TypeError):
        M.NativeMeshingProvider(
            M.NativeMeshingOptions("planar_constrained_delaunay")
        ).plan(source, specification)  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    "fraction",
    (0.0, -0.1, 1.01, np.nan, np.inf),
    ids=("zero", "negative", "above-one", "nan", "infinite"),
)
def test_native_hex_dominant_schedule_refuses_invalid_core_fraction(
    fraction: float,
) -> None:
    with pytest.raises(ValueError):
        M.NativeMeshingOptions("plc_hex_dominant", hex_core_fraction=fraction)


def test_native_planar_route_refuses_hex_dominant_scheduling() -> None:
    with pytest.raises(ValueError):
        M.NativeMeshingOptions("planar_constrained_delaunay", hex_core_fraction=0.5)


def test_native_source_revision_refuses_numeric_display_identity() -> None:
    region = phx.geometry.PlanarMeshRegion(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        [[0, 1, 2]],
        feature_id="triangle",
    )
    with pytest.raises(TypeError):
        M.NativePlanarSource(region, 1)  # ty: ignore[invalid-argument-type]


def test_native_dual_quad_provider_certifies_complete_square_cover() -> None:
    source = M.NativePlanarSource(
        phx.geometry.PlanarMeshRegion(
            np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
            ),
            [[0, 1, 2, 3]],
            feature_id="square",
        ),
        "r1",
    )
    scope = M.MeshingScope(
        "square",
        "r1",
        M.MeshingEntityKind.GEOMETRY,
        2,
        "square-region",
        np.asarray([0], dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("quadrilateral",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            M.UniformSizeControl(scope, 1.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("planar_dual_quad"))
        .plan(source, specification, coordinate_contract=_SI)
        .execute()
    )
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    corners = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    following = np.roll(corners, -1, axis=1)
    areas = (
        np.sum(
            corners[:, :, 0] * following[:, :, 1] - corners[:, :, 1] * following[:, :, 0],
            axis=1,
        )
        / 2.0
    )
    assert np.all(areas > 0.0)
    np.testing.assert_allclose(np.sum(areas), 1.0, rtol=1e-12)
    assert result.audit.passed
    assert result.compliance.passed
    certification = result.certification
    assert certification is not None and certification.passed
    coverage = certification.coverage
    assert coverage is not None
    assert all(value is not None for value in coverage.achieved_region_measures)
    np.testing.assert_allclose(
        np.asarray(coverage.achieved_region_measures, dtype=np.float64),
        (1.0,),
        rtol=1e-12,
    )


def test_native_phase_recorder_preserves_scientific_result_identity() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    plc = M.PiecewiseLinearComplex(
        points,
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray([[-1, 0]], dtype=np.int64), (4, 1)),
        ("solid",),
    )
    source = M.NativePlcSource(plc, "tetrahedron", "r1")
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "tetrahedron-facets",
        np.arange(4, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 4.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    records: list[M.NativeMeshingPhaseMeasurement] = []
    plan = M.NativeMeshingProvider(M.NativeMeshingOptions("plc_tetrahedral")).plan(
        source,
        specification,
        coordinate_contract=_SI,
        record_phase=records.append,
    )
    ordinary = plan.execute()
    measured = plan.execute(record_phase=records.append)

    assert measured.result_id == ordinary.result_id
    np.testing.assert_array_equal(measured.mesh.coordinates, ordinary.mesh.coordinates)
    np.testing.assert_array_equal(
        measured.mesh.blocks[0].vertices, ordinary.mesh.blocks[0].vertices
    )
    certification = measured.certification
    assert certification is not None and certification.passed
    coverage = certification.coverage
    assert coverage is not None
    assert all(value is not None for value in coverage.achieved_region_measures)
    np.testing.assert_allclose(
        np.asarray(coverage.achieved_region_measures, dtype=np.float64),
        (1.0 / 6.0,),
        rtol=1e-12,
    )
    assert {
        "source_preparation",
        "native_validation",
        "native_preparation",
        "boundary_recovery",
        "region_classification",
        "native_publication",
        "audit",
        "certification",
        "publication",
    } <= {record.phase for record in records}
    assert all(
        record.elapsed_seconds >= 0.0 and record.invocations > 0 for record in records
    )


def test_native_publication_refuses_full_retained_data_beyond_small_carrier() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    facets = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    plc = M.PiecewiseLinearComplex(
        points,
        facets,
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray(((-1, 0),), dtype=np.int64), (4, 1)),
        ("solid",),
    )
    source = M.NativePlcSource(plc, "retained-data-tetrahedron", "r1")
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "tetrahedron-facets",
        np.arange(4, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 4.0, strength=M.SizeControlStrength.SOFT),
        ),
        limits=M.MeshingLimits(maximum_data_bytes=2000),
    )
    plan = M.NativeMeshingProvider(M.NativeMeshingOptions("plc_tetrahedral")).plan(
        source,
        specification,
        coordinate_contract=_SI,
    )
    with pytest.raises(M.MeshingFailure) as refusal:
        plan.execute()
    evidence = refusal.value.evidence
    assert evidence.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert evidence.stage == M.MeshingStageKind.CANONICALIZATION.value
    assert dict(evidence.requested)["maximum_data_bytes"] == 2000
    achieved = dict(evidence.achieved)
    assert achieved["retained_data_bytes"] > 2000
    assert achieved["native_scope:work_units"] > 0
