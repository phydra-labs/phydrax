"""Typed scientific fixtures shared by native provider admission and execution tests."""

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import PreparedTensorGrid
from phydrax.geometry import (
    AdaptiveImplicitSurfacePolicy,
    BoundaryAtlas,
    CompiledGeometry,
    SegmentMesh,
)
from phydrax.meshing import (
    CellMeshingResult,
    CurveJunction,
    CurveMeshingSpec,
    MeshingLimits,
    MeshingScope,
    MeshingStageKind,
    MeshingStageReport,
    NativeMeshingPlan,
    NativeMeshingProvider,
    NativePlanarSource,
    PeriodicConstraint,
    ProtectedFeature,
    SizeControlStrength,
    SurfaceMeshingSpec,
    UniformSizeControl,
)
from phydrax.units import METER


M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()


def grid(count: int = 9) -> PreparedTensorGrid:
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))


def implicit_case(
    *,
    size_strength: SizeControlStrength,
    limits: MeshingLimits | None = None,
    periodic_constraints: tuple[PeriodicConstraint, ...] = (),
    fidelity: float | None = None,
) -> tuple[CompiledGeometry, MeshingScope, SurfaceMeshingSpec]:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.75, feature_id="sphere").compile(
        coordinate_unit=METER
    )
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


def stage_report(result: CellMeshingResult, kind: MeshingStageKind) -> MeshingStageReport:
    (stage,) = (value for value in result.trace.stages if value.stage is kind)
    return stage


def implicit_plan(
    geometry: CompiledGeometry, specification: SurfaceMeshingSpec
) -> NativeMeshingPlan:
    options = M.NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=500_000,
        ),
    )
    return M.NativeMeshingProvider(options).plan(
        M.NativeImplicitSource(geometry, grid(), "sphere", "r1"),
        specification,
        coordinate_contract=_SI,
    )


def adaptive_plan(
    geometry: CompiledGeometry,
    specification: SurfaceMeshingSpec,
    policy: AdaptiveImplicitSurfacePolicy,
) -> NativeMeshingPlan:
    options = M.NativeMeshingOptions("implicit_surface", implicit_policy=policy)
    return M.NativeMeshingProvider(options).plan(
        M.NativeImplicitSource(geometry, grid(), "sphere", "r1"),
        specification,
        coordinate_contract=_SI,
    )


def plate(
    *,
    minimum_angle: float | np.float64 = np.radians(25.0),
    target_size: float = 0.2,
    limits: MeshingLimits | None = None,
) -> tuple[NativePlanarSource, SurfaceMeshingSpec]:
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


def planar_provider() -> NativeMeshingProvider:
    return M.NativeMeshingProvider(M.NativeMeshingOptions("planar_constrained_delaunay"))


def curve_spec(
    scope: MeshingScope,
    junctions: tuple[CurveJunction, ...],
    *,
    size_controls: tuple[UniformSizeControl, ...] = (),
    protected_features: tuple[ProtectedFeature, ...] = (),
    limits: MeshingLimits | None = None,
) -> CurveMeshingSpec:
    return M.CurveMeshingSpec(
        M.CellMeshingTarget(1, 2, M.CellFamilyPolicy(required=("interval",))),
        scope,
        junctions=junctions,
        size_controls=size_controls,
        protected_features=protected_features,
        limits=limits,
    )


def curve_plan(
    curves: BoundaryAtlas | SegmentMesh, specification: CurveMeshingSpec
) -> NativeMeshingPlan:
    return M.NativeMeshingProvider(M.NativeMeshingOptions("curve_arc_length")).plan(
        M.NativeCurveSource(curves, "r1"), specification, coordinate_contract=_SI
    )


def tee() -> tuple[SegmentMesh, MeshingScope]:
    segments = phx.geometry.SegmentMesh(
        jnp.asarray([[0, 0], [1, 0], [0, 1], [-1, 0]], dtype=jnp.float64),
        jnp.asarray([[0, 1], [0, 2], [0, 3]]),
        source_id="tee",
    )
    scope = M.MeshingScope(
        "tee", "r1", M.MeshingEntityKind.GEOMETRY, 1, "tee-edges", np.arange(3)
    )
    return segments, scope
