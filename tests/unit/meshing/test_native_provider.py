#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import brep_box
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.geometry.simplicial import SegmentMesh
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    CurveMeshingSpec,
    FeatureKind,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingScope,
    MeshingStageKind,
    MeshQualityTarget,
    NativeMeshingOptions,
    NativeMeshingPlan,
    NativeMeshingProvider,
    PatchControl,
    ProtectedFeature,
    RegionControl,
    RegionRole,
    RegionSeed,
    SizeControlStrength,
    SurfaceMeshingSpec,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native import _PreparedDirectBRepCadVolume
from phydrax.meshing.providers._native_implicit import _NativeImplicitBoundaryQuery
from phydrax.meshing.providers._native_sources import (
    NativeCurveSource,
    NativeImplicitSource,
)


pytestmark = pytest.mark.meshcore


def _curve_request() -> tuple[NativeCurveSource, CurveMeshingSpec]:
    segments = SegmentMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        source_id="unified-native-provider-curve",
    )
    source = NativeCurveSource(
        segments,
        "revision-1",
        curve_entity_ids=np.asarray((1,), dtype=np.int64),
        vertex_entity_ids=np.asarray((1, 2), dtype=np.int64),
    )
    scope = MeshingScope.whole(source, 1)
    specification = CurveMeshingSpec(
        CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",))),
        scope,
        size_controls=(UniformSizeControl(scope, 0.5),),
        maximum_deviation=0.01,
    )
    return source, specification


def test_native_provider_plans_and_accepts_a_curve_mesh() -> None:
    source, specification = _curve_request()
    contract = SpatialCoordinateContract.si()
    provider = NativeMeshingProvider()

    first = provider.plan(source, specification, coordinate_contract=contract)
    second = provider.plan(source, specification, coordinate_contract=contract)

    assert isinstance(first, NativeMeshingPlan)
    assert first.route == "curve_arc_length"
    assert first.plan_id == second.plan_id
    result = first.execute(prove=True)
    assert result.certification is not None and result.certification.passed
    assert result.coordinate_contract.spatial_id == contract.spatial_id
    assert result.provider.name == "phydrax-native"
    assert result.runtime.actual_version.startswith("phydrax-meshcore ")
    assert any(
        stage.stage is MeshingStageKind.CURVE_MESHING for stage in result.trace.stages
    )


def test_native_provider_uses_adaptive_route_for_analytic_surface_input() -> None:
    contract = SpatialCoordinateContract.si()
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="unified-native-provider-implicit"
    ).compile()
    source = AnalyticImplicitProfile(
        geometry,
        contract,
        source_id="unified-native-provider-implicit",
    )
    scope = MeshingScope.whole(source, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(
                scope,
                0.5,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        maximum_deviation=0.25,
    )

    plan = NativeMeshingProvider().plan(source, specification)

    assert plan.route == "implicit_surface"
    assert plan.coordinate_contract.spatial_id == contract.spatial_id


def test_native_provider_keeps_analytic_volume_algorithms_distinct() -> None:
    contract = SpatialCoordinateContract.si()
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="unified-native-provider-volume"
    ).compile()
    source = AnalyticImplicitProfile(
        geometry,
        contract,
        source_id="unified-native-provider-volume",
    )
    scope = MeshingScope.whole(source, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(
                scope,
                0.5,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        maximum_deviation=0.25,
    )

    restricted = NativeMeshingProvider().plan(source, specification)
    adaptive = NativeMeshingProvider(
        NativeMeshingOptions("implicit_adaptive_tetrahedral")
    ).plan(source, specification)

    assert restricted.route == "implicit_restricted_delaunay"
    assert adaptive.route == "implicit_adaptive_tetrahedral"
    assert restricted.plan_id != adaptive.plan_id


def test_native_provider_builds_a_direct_brep_volume_boundary_internally() -> None:
    contract = SpatialCoordinateContract.si()
    source = brep_box(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((1.0, 1.0, 1.0), dtype=np.float64),
        coordinate_contract=contract,
        source_id="unified-native-provider-cad-box",
    )
    boundary = MeshingScope.whole(source, 2)
    missing_tolerance = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.5),),
    )
    with pytest.raises(MeshingFailure) as failure:
        NativeMeshingProvider().plan(source, missing_tolerance)
    assert failure.value.category is MeshingFailureCategory.UNSUPPORTED_COMBINATION
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.5),),
        maximum_deviation=0.05,
    )

    plan = NativeMeshingProvider().plan(source, specification)

    assert plan.route == "cad_tetrahedral"
    result = plan.execute(prove=True)
    assert result.certification is not None and result.certification.passed
    assert result.coordinate_contract.spatial_id == contract.spatial_id


def test_native_cad_volume_geometry_is_independent_of_source_id() -> None:
    contract = SpatialCoordinateContract.si()
    meshes = []
    for source_id in ("cad-volume-cylinder", "renamed-cad-cylinder"):
        source = phx.geometry.brep.brep_cylinder(
            0.5,
            1.0,
            coordinate_contract=contract,
            source_id=source_id,
        )
        boundary = MeshingScope.whole(source, 2)
        specification = VolumeMeshingSpec(
            CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
            boundary,
            VolumeFillStrategy.SIMPLEX,
            size_controls=(UniformSizeControl(boundary, 0.3),),
            maximum_deviation=0.05,
        )
        options = NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
            ),
        )
        result = (
            NativeMeshingProvider(options).plan(source, specification).execute(prove=True)
        )
        assert result.certification is not None and result.certification.passed
        meshes.append(result.mesh)

    np.testing.assert_array_equal(meshes[0].coordinates, meshes[1].coordinates)
    assert len(meshes[0].blocks) == len(meshes[1].blocks)
    for first, second in zip(meshes[0].blocks, meshes[1].blocks, strict=True):
        np.testing.assert_array_equal(first.vertices, second.vertices)


def test_native_cad_volume_soft_shape_strength_softens_boundary_angle() -> None:
    contract = SpatialCoordinateContract.si()
    source = brep_box(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((1.0, 1.0, 1.0), dtype=np.float64),
        coordinate_contract=contract,
        source_id="native-cad-soft-shape-box",
    )
    boundary = MeshingScope.whole(source, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.5),),
        maximum_deviation=0.05,
    )

    def plan(strength: SizeControlStrength) -> NativeMeshingPlan:
        schedule = NativeVolumeSchedule(shape_strength=strength)
        options = NativeMeshingOptions("cad_tetrahedral", volume_schedule=schedule)
        return NativeMeshingProvider(options).plan(source, specification)

    def boundary_target(candidate: NativeMeshingPlan) -> MeshQualityTarget:
        prepared = candidate.prepared
        assert isinstance(prepared, _PreparedDirectBRepCadVolume)
        target = prepared.boundary_specification.quality_target
        assert target is not None
        return target

    hard_target = boundary_target(plan(SizeControlStrength.HARD))
    soft_plan = plan(SizeControlStrength.SOFT)
    soft_target = boundary_target(soft_plan)
    assert hard_target.hard and not soft_target.hard
    assert soft_target.minimum_angle == hard_target.minimum_angle
    result = soft_plan.execute(prove=True)
    assert result.certification is not None and result.certification.passed
    assert result.compliance.issues == ()
    assert "soft_shape_target_unmet_count" in dict(result.compliance.achieved)


def test_native_provider_preserves_an_explicit_incompatible_route() -> None:
    source, specification = _curve_request()
    provider = NativeMeshingProvider(NativeMeshingOptions("planar_constrained_delaunay"))

    with pytest.raises(MeshingFailure) as failure:
        provider.plan(
            source,
            specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )

    assert failure.value.category is MeshingFailureCategory.UNSUPPORTED_COMBINATION


def test_native_provider_accepts_bounded_implicit_volume_region_and_patch() -> None:
    contract = SpatialCoordinateContract.si()
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="unified-native-provider-bounded-volume"
    ).compile()
    source_id, source_revision = "unified-native-provider-bounded-volume", "r1"
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08, root_tolerance=1.0e-8
    )
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=np.asarray(((-1.4, -1.4, -1.4), (1.4, 1.4, 1.4)), dtype=np.float64),
        policy=policy,
        source_id=source_id,
    )
    # The sphere distance field is singular at its centre; the exact profile
    # proves that singular set lies away from the zero-set cover.
    profile = AnalyticImplicitProfile(
        geometry, contract, source_id=source_id, source_revision=source_revision
    )
    source = NativeImplicitSource(
        geometry,
        source_id,
        source_revision,
        discovery=discovery,
        authoring_profile=profile,
    )
    boundary_scope = MeshingScope.whole(source, 2)
    region_scope = MeshingScope.whole(source, 3)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(
                boundary_scope,
                0.7,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        region_seeds=(
            RegionSeed(
                np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
                "interior",
                "fluid",
                RegionRole.FLUID,
            ),
        ),
        region_controls=(
            RegionControl(region_scope, "interior", "fluid", RegionRole.FLUID),
        ),
        patch_controls=(PatchControl("wall", boundary_scope, ("interior",)),),
        protected_features=(
            ProtectedFeature(
                boundary_scope,
                FeatureKind.SURFACE,
                maximum_deviation=0.8,
            ),
        ),
        maximum_deviation=0.8,
    )
    result = (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "implicit_adaptive_tetrahedral",
                implicit_policy=policy,
                volume_schedule=NativeVolumeSchedule(
                    radius_edge_bound=8.0,
                    minimum_dihedral_degrees=0.0,
                ),
            )
        )
        .plan(source, specification, coordinate_contract=contract)
        .execute(prove=True)
    )

    assert result.certification is not None and result.certification.passed
    assert result.certification.inputs.domain is discovery
    fidelity_source = result.certification.inputs.source
    assert isinstance(fidelity_source, _NativeImplicitBoundaryQuery)
    assert (fidelity_source.source_id, fidelity_source.source_revision) == (
        source_id,
        source_revision,
    )
    assert fidelity_source.adaptive_source.surface is discovery
    assert len(result.zones) == 1
    assert result.zones[0].name == "interior"
    assert result.zones[0].material_id == "fluid"
    assert result.zones[0].region_role is RegionRole.FLUID
    assert len(result.patches) == 1
    assert result.patches[0].name == "wall"
    assert result.patches[0].adjacent_zone_ids == (result.zones[0].zone_id,)
    region_key = source.source_entity_key_for_scope(3, int(region_scope.entity_ids[0]))
    assert result.patches[0].source_adjacent_region_ids == (region_key[0], None)
    assert result.patches[0].source_adjacent_region_occurrence_paths == (
        region_key[1],
        None,
    )


@pytest.mark.parametrize("route", ["planar", "sphere-surface", "sphere-volume"])
def test_native_plan_candidate_proves_the_constructed_mesh(route: str) -> None:
    if route == "planar":
        from tests.unit.meshing.test_native_curve_planar_routes import _planar_request

        source, specification, options = _planar_request()
        plan = NativeMeshingProvider(options).plan(
            source, specification, coordinate_contract=SpatialCoordinateContract.si()
        )
    else:
        contract = SpatialCoordinateContract.si()
        geometry = phx.geometry.Sphere(
            (0.0, 0.0, 0.0), 0.75, feature_id=f"candidate-{route}"
        ).compile()
        profile = AnalyticImplicitProfile(
            geometry, contract, source_id=f"candidate-{route}"
        )
        scope = MeshingScope.whole(profile, 2)
        if route == "sphere-surface":
            surface = SurfaceMeshingSpec(
                CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
                scope,
                size_controls=(
                    UniformSizeControl(scope, 0.5, strength=SizeControlStrength.SOFT),
                ),
                maximum_deviation=0.25,
            )
            plan = NativeMeshingProvider().plan(profile, surface)
        else:
            volume = VolumeMeshingSpec(
                CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
                scope,
                VolumeFillStrategy.SIMPLEX,
                size_controls=(
                    UniformSizeControl(scope, 0.5, strength=SizeControlStrength.SOFT),
                ),
                maximum_deviation=0.25,
            )
            plan = NativeMeshingProvider().plan(profile, volume)

    candidate = plan.execute()
    assert isinstance(candidate, phx.meshing.CellMeshingCandidate)
    assert candidate.certified_steps
    assert not hasattr(candidate, "certification")
    assert not hasattr(candidate, "audit")
    assert not hasattr(candidate, "compliance")
    accepted = plan.execute(prove=True)
    np.testing.assert_array_equal(candidate.mesh.coordinates, accepted.mesh.coordinates)
    assert (
        np.asarray(candidate.mesh.coordinates).tobytes()
        == np.asarray(accepted.mesh.coordinates).tobytes()
    )
    for left, right in zip(candidate.mesh.blocks, accepted.mesh.blocks, strict=True):
        assert np.asarray(left.vertices).tobytes() == np.asarray(right.vertices).tobytes()
    proved = candidate.prove()
    assert proved.result_id == accepted.result_id
    assert proved.mesh.mesh_id == accepted.mesh.mesh_id
    assert proved.certification is not None
    assert accepted.certification is not None
    assert proved.certification.status == accepted.certification.status
    assert proved.certification.report_id == accepted.certification.report_id


def test_deferred_certification_resolves_only_the_original_mesh() -> None:
    from dataclasses import replace

    from phydrax._meshcore import meshcore_identity
    from phydrax.discretization import CellGeometrySpec, CellMesh
    from phydrax.meshing._canonical import _accept_native_construction, _NATIVE_PROVIDER
    from phydrax.meshing._native_construction import NativeMeshConstruction

    class Deferred:
        def __init__(self, construction: NativeMeshConstruction) -> None:
            self.construction = construction

        def resolve(self, /) -> NativeMeshConstruction:
            return self.construction

    source, specification = _curve_request()
    plan = NativeMeshingProvider().plan(
        source, specification, coordinate_contract=SpatialCoordinateContract.si()
    )
    construction = plan.prepared.execute(steering="screened")
    version = meshcore_identity()
    pending = replace(
        construction,
        deferred_certification=Deferred(construction),
        screens={"error": 0.5},
    )
    candidate = phx.meshing.CellMeshingCandidate(
        pending, specification, _NATIVE_PROVIDER, version, plan.plan_id, plan.route
    )
    assert candidate.screens == {"error": 0.5}
    with pytest.raises(ValueError, match="pending certification"):
        _accept_native_construction(
            pending, specification, _NATIVE_PROVIDER, version, plan.plan_id, plan.route
        )
    assert candidate.prove().result_id == plan.execute(prove=True).result_id

    mesh = CellMesh(
        np.zeros_like(np.asarray(construction.mesh.coordinates)),
        construction.mesh.blocks,
        vertex_global_ids=construction.mesh.vertex_global_ids,
    )
    changed = replace(construction, mesh=mesh, geometry=CellGeometrySpec.affine(mesh))
    candidate = phx.meshing.CellMeshingCandidate(
        replace(pending, deferred_certification=Deferred(changed)),
        specification,
        _NATIVE_PROVIDER,
        version,
        plan.plan_id,
        plan.route,
    )
    with pytest.raises(ValueError, match="changed the constructed mesh"):
        candidate.prove()

    replaced_candidate = phx.meshing.CellMeshingCandidate(
        replace(pending, mesh=mesh, geometry=CellGeometrySpec.affine(mesh)),
        specification,
        _NATIVE_PROVIDER,
        version,
        plan.plan_id,
        plan.route,
    )
    with pytest.raises(ValueError, match="changed the constructed mesh"):
        replaced_candidate.prove()


def test_pending_candidate_is_refused_by_accepted_result_consumer() -> None:
    from tools.meshing_qualification._runtime import accepted_result_record

    source, specification = _curve_request()
    candidate = (
        NativeMeshingProvider()
        .plan(source, specification, coordinate_contract=SpatialCoordinateContract.si())
        .execute()
    )
    with pytest.raises(TypeError, match="accepted CellMeshingResult"):
        accepted_result_record(candidate)  # ty: ignore[invalid-argument-type]


def test_pending_owner_local_execution_is_refused() -> None:
    import jax

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup

    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "candidate-refusal",
            (process,),
            ((process, device.id),),
            mesh_axes=(("candidate", 1),),
        ),
        (device,),
    )
    source, specification = _curve_request()
    plan = NativeMeshingProvider().plan(
        source, specification, coordinate_contract=SpatialCoordinateContract.si()
    )
    with pytest.raises(ValueError, match="requires prove=True"):
        plan.execute(execution_group=group)  # ty: ignore[invalid-argument-type]


def test_broken_candidate_proof_reports_the_mesh_failure() -> None:
    from dataclasses import replace

    from phydrax._meshcore import meshcore_identity
    from phydrax.discretization import CellGeometrySpec, CellMesh
    from phydrax.meshing._canonical import _NATIVE_PROVIDER

    source, specification = _curve_request()
    specification = CurveMeshingSpec(
        specification.target,
        specification.scope,
        size_controls=specification.size_controls,
    )
    plan = NativeMeshingProvider().plan(
        source, specification, coordinate_contract=SpatialCoordinateContract.si()
    )
    construction = plan.prepared.execute(steering="screened")
    coordinates = np.zeros_like(np.asarray(construction.mesh.coordinates))
    broken_mesh = CellMesh(
        coordinates,
        construction.mesh.blocks,
        vertex_global_ids=construction.mesh.vertex_global_ids,
    )
    broken = replace(
        construction,
        mesh=broken_mesh,
        geometry=CellGeometrySpec.affine(broken_mesh),
    )
    candidate = phx.meshing.CellMeshingCandidate(
        broken,
        specification,
        _NATIVE_PROVIDER,
        meshcore_identity(),
        plan.plan_id,
        plan.route,
    )
    with pytest.raises(MeshingFailure) as failure:
        candidate.prove()
    assert failure.value.category is MeshingFailureCategory.AUDIT_FAILED
    assert failure.value.stage == "mesh_certification"
    assert "cell_validity" in str(failure.value)
    broken_cells = set(np.asarray(broken_mesh.blocks[0].global_ids).tolist())
    assert broken_cells <= set(failure.value.entity_ids)
