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
    result = first.execute()
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
    result = plan.execute()
    assert result.certification is not None and result.certification.passed
    assert result.coordinate_contract.spatial_id == contract.spatial_id


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
        .execute()
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
