"""Adaptive implicit source publication and cumulative seed-query allowances."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingLimits,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._implicit_volume import (
    classify_adaptive_volume_seeds,
    prepare_adaptive_implicit_volume,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.units import METER
from tests._support.implicit_meshing import (
    execution_scope as execution_scope,
    IMPLICIT_DOMAIN,
    implicit_specification,
    NATIVE,
    source_budget as source_budget,
)


# Unchanged empirical volume bound from the general enclosed-source fixture,
# whose provider uses minimum surface level 3 and maximum level 9.
ADAPTIVE_VOLUME_RELATIVE_ERROR_BOUND = 0.1


@NATIVE
def test_adaptive_seed_queries_reserve_the_remaining_source_allowance(
    execution_scope: NativeExecutionScope, source_budget: CoordinateEnclosureBudget
) -> None:
    geometry = phx.geometry.Sphere((0.013, -0.021, 0.007), 0.75).compile(
        coordinate_unit=METER
    )
    policy = phx.geometry.AdaptiveImplicitSurfacePolicy(
        maximum_level=7,
        flatness_tolerance=0.02,
    )
    schedule = NativeVolumeSchedule()
    contract = phx.SpatialCoordinateContract.si()
    original = implicit_specification()
    prepared = prepare_adaptive_implicit_volume(
        geometry,
        1.4 * IMPLICIT_DOMAIN,
        original,
        schedule,
        policy,
        "state",
        contract,
        scope=execution_scope,
    )
    specification = VolumeMeshingSpec(
        original.target,
        original.boundary_scope,
        original.fill_strategy,
        size_controls=original.size_controls,
        region_seeds=(
            phx.meshing.RegionSeed(
                np.asarray((0.013, -0.021, 0.007), dtype=np.float64),
                "body",
                "material",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        limits=MeshingLimits(maximum_geometry_queries=prepared.geometry_queries + 1),
    )
    with pytest.raises(MeshingFailure) as failure:
        classify_adaptive_volume_seeds(
            prepared.surface,
            specification,
            prepared.geometry_queries,
            budget=source_budget,
        )
    assert failure.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.value.evidence.achieved) == {
        "geometry_queries": prepared.geometry_queries,
        "requested_seed_queries": 1,
    }
    admitted = VolumeMeshingSpec(
        original.target,
        original.boundary_scope,
        original.fill_strategy,
        size_controls=original.size_controls,
        region_seeds=specification.region_seeds,
        limits=MeshingLimits(maximum_geometry_queries=prepared.geometry_queries + 2),
    )
    assert (
        classify_adaptive_volume_seeds(
            prepared.surface,
            admitted,
            prepared.geometry_queries,
            budget=source_budget,
        )
        == prepared.geometry_queries + 1
    )


@NATIVE
@pytest.mark.parametrize("source_kind", ("non-distance", "internal-tangency"))
def test_general_enclosed_scalar_source_publishes_and_rejects_stale_state(
    source_kind: str,
    tmp_path: Path,
) -> None:
    from phydrax.geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitBoundarySource,
    )
    from phydrax.meshing._implicit_volume import (
        execute_adaptive_implicit_volume,
        PreparedAdaptiveImplicitVolume,
    )

    if source_kind == "non-distance":
        primitive = phx.geometry.Ellipsoid((0.0123, 0.0037, -0.0102), (0.35, 0.28, 0.23))
        geometry = (primitive | primitive).compile(coordinate_unit=METER)
        expected_volume = 4.0 * np.pi / 3.0 * 0.35 * 0.28 * 0.23
        components = 1
    else:
        geometry = (
            phx.geometry.Sphere((-0.37, 0.013, -0.017), 0.19)
            | phx.geometry.Sphere((0.36, -0.019, 0.021), 0.14)
        ).compile(coordinate_unit=METER)
        expected_volume = 4.0 * np.pi / 3.0 * (0.19**3 + 0.14**3)
        components = 2
    meshing = phx.meshing
    domain = jnp.asarray(((-0.8, -0.6, -0.6), (0.8, 0.6, 0.6)), dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(domain)
    source = meshing.NativeImplicitSource(
        geometry, grid, "general-source", "scientific-revision"
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        "zero-set",
        np.asarray((0,), dtype=np.int64),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, 0.2, strength=SizeControlStrength.SOFT),
        ),
    )
    provider = meshing.NativeMeshingProvider(
        meshing.NativeMeshingOptions(
            "implicit_adaptive_tetrahedral",
            implicit_policy=phx.geometry.AdaptiveImplicitSurfacePolicy(
                initial_level=2,
                minimum_surface_level=3,
                maximum_level=9,
            ),
            volume_schedule=NativeVolumeSchedule(
                refinement_rounds=1,
                improvement_passes=1,
                metric_optimization_passes=1,
            ),
        )
    )
    contract = phx.SpatialCoordinateContract.si()
    plan = provider.plan(source, specification, coordinate_contract=contract)
    result = plan.execute()
    assert isinstance(plan.prepared, PreparedAdaptiveImplicitVolume)
    prepared = plan.prepared
    assert prepared.surface.mesh is not None
    assert prepared.surface.mesh.topology.num_face_components == components
    assert prepared.surface.evidence.unresolved_count == 0
    certificate = result.certification
    assert certificate is not None and certificate.passed
    assert certificate.coverage is not None and certificate.coverage.status == "certified"
    assert certificate.fidelity is not None and certificate.fidelity.status == "certified"
    corners = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    volume = np.sum(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
    assert (
        abs(volume - expected_volume)
        < ADAPTIVE_VOLUME_RELATIVE_ERROR_BOUND * expected_volume
    )
    if source_kind == "non-distance":
        boundary = AdaptiveImplicitBoundarySource(
            geometry, prepared.surface, source.source_revision
        )
        distance = boundary.boundary_distance(
            np.asarray(((0.4123, 0.0037, -0.0102),), dtype=np.float64)
        )
        assert distance.lower[0] <= 0.05 <= distance.upper[0]
    path = tmp_path / "general-source.zip"
    receipt = write_meshing_source_closure(path, prepared)
    restored = read_meshing_source_closure(path, expected_content_id=receipt.content_id)
    assert isinstance(restored, PreparedAdaptiveImplicitVolume)
    restored_boundary = AdaptiveImplicitBoundarySource(
        restored.geometry, restored.surface, restored.source_revision
    )
    point, exact_distance = (
        ((0.4123, 0.0037, -0.0102), 0.05)
        if source_kind == "non-distance"
        else ((0.5, -0.019, 0.021), 0.0)
    )
    distance = restored_boundary.boundary_distance(np.asarray((point,), dtype=np.float64))
    assert distance.lower[0] <= exact_distance <= distance.upper[0]
    changed = phx.geometry.Ellipsoid((0.02, 0.03, 0.04), (0.35, 0.28, 0.23)).compile(
        coordinate_unit=METER
    )
    refused_scope = NativeExecutionScope(specification.limits.native_limits())
    with pytest.raises(ValueError, match="stale"):
        execute_adaptive_implicit_volume(
            changed,
            specification,
            prepared,
            contract,
            provider.info(),
            plan.plan_id,
            scope=refused_scope,
            workspace=NativeHostStorageWorkspace(refused_scope),
        )
    assert result.certification is certificate
