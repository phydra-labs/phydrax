"""Native provider boundaries retain concrete source, schedule and result types."""

from typing import assert_type

import numpy as np
from jax.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray

from phydrax import SpatialCoordinateContract
from phydrax._meshcore import NativeExecutionBudget
from phydrax.discretization import CellMesh, PreparedTensorGrid
from phydrax.geometry import (
    BoundaryAtlas,
    CompiledGeometry,
    MeshingDomain,
    PlanarMeshRegion,
    SegmentMesh,
    SourceBoundaryQuery,
)
from phydrax.meshing import (
    BlockInterfaceControl,
    CellMeshingResult,
    MeshCertificationReport,
    NativeCurveSource,
    NativeImplicitSource,
    NativeMeshingOptions,
    NativeMeshingPlan,
    NativeMeshingProvider,
    NativeMeshingRoute,
    NativePeriodicSource,
    NativePlanarSource,
    NativePlcSource,
    NativePolyhedralSchedule,
    NativePolyhedralSource,
    NativeStructuredSchedule,
    NativeStructuredSource,
    NativeSurfaceSource,
    NativeSweepSource,
    providers,
    ProviderSupportReport,
    SurfaceMeshingSpec,
    SweepControl,
    TransfiniteBlock,
    VolumeMeshingSpec,
)


def host_allocator_boundary(
    execution: NativeExecutionBudget,
    dtype: DTypeLike,
) -> None:
    assert_type(execution.allocate_host_array((3, 4), np.float64), NDArray[np.float64])
    assert_type(execution.allocate_host_array((6, 2), np.int32), NDArray[np.int32])
    assert_type(
        execution.allocate_host_array((8,), np.dtype(np.int64)), NDArray[np.int64]
    )
    assert_type(execution.allocate_host_array((1,), dtype), NDArray[np.generic])


def planar_boundary(
    region: PlanarMeshRegion,
    specification: SurfaceMeshingSpec,
    coordinates: SpatialCoordinateContract,
) -> None:
    source = NativePlanarSource(region, "r1")
    assert_type(source, NativePlanarSource)
    assert_type(source.region, PlanarMeshRegion)
    assert_type(source.source_revision, str)
    options = NativeMeshingOptions("planar_constrained_delaunay")
    assert_type(options.route, NativeMeshingRoute)
    provider = NativeMeshingProvider(options)
    assert_type(provider.validate(source, specification), ProviderSupportReport)
    plan = provider.plan(source, specification, coordinate_contract=coordinates)
    assert_type(plan, NativeMeshingPlan)
    result = plan.execute()
    assert_type(result, CellMeshingResult)
    assert_type(result.mesh, CellMesh)
    assert_type(result.certification, MeshCertificationReport | None)
    assert_type(providers.NativePlanarSource(region, "r1"), NativePlanarSource)
    provider.plan(source, specification)  # ty: ignore[missing-argument]
    provider.plan(source, specification, coordinate_contract="m")  # ty: ignore[invalid-argument-type]


def polyhedral_boundary(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    coordinates: SpatialCoordinateContract,
) -> None:
    schedule = NativePolyhedralSchedule(maximum_refinement_steps=2)
    assert_type(schedule.maximum_refinement_steps, int)
    options = NativeMeshingOptions("plc_restricted_power", polyhedral_schedule=schedule)
    assert_type(options.polyhedral_schedule, NativePolyhedralSchedule | None)
    plan = providers.NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=coordinates
    )
    assert_type(plan.execute(), CellMeshingResult)
    NativePolyhedralSchedule(maximum_refinement_steps="two")  # ty: ignore[invalid-argument-type]


def periodic_boundary(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec,
    coordinates: SpatialCoordinateContract,
) -> None:
    plan = NativeMeshingProvider(NativeMeshingOptions("periodic_delaunay")).plan(
        source, specification, coordinate_contract=coordinates
    )
    assert_type(plan.execute(), CellMeshingResult)


def deliberate_invalid_boundaries(geometry: CompiledGeometry) -> None:
    NativeMeshingOptions("tetgen")  # ty: ignore[invalid-argument-type]
    NativeMeshingProvider("native")  # ty: ignore[invalid-argument-type]
    NativePlanarSource(geometry, "r1")  # ty: ignore[invalid-argument-type]


def structured_sources(
    blocks: tuple[TransfiniteBlock, ...],
    interfaces: tuple[BlockInterfaceControl, ...],
    profile: CellMesh,
    sweep: SweepControl,
    fidelity: SourceBoundaryQuery,
) -> None:
    schedule = NativeStructuredSchedule(optimization_steps=2)
    assert_type(schedule.optimization_steps, int)
    options = NativeMeshingOptions("structured_transfinite", structured_schedule=schedule)
    assert_type(options.structured_schedule, NativeStructuredSchedule | None)
    structured = NativeStructuredSource(
        blocks,
        interfaces,
        fidelity.source_id,
        fidelity.source_revision,
        fidelity_source=fidelity,
        maximum_deviation=0.01,
    )
    assert_type(structured, NativeStructuredSource)
    assert_type(structured.blocks, tuple[TransfiniteBlock, ...])
    assert_type(structured.interfaces, tuple[BlockInterfaceControl, ...])
    assert_type(structured.fidelity_source, SourceBoundaryQuery)
    swept = NativeSweepSource(
        profile,
        sweep,
        fidelity.source_id,
        fidelity.source_revision,
        fidelity_source=fidelity,
        maximum_deviation=0.01,
    )
    assert_type(swept, NativeSweepSource)
    assert_type(swept.profile, CellMesh)
    assert_type(swept.control, SweepControl)


def native_geometry_bindings(
    curves: BoundaryAtlas | SegmentMesh,
    geometry: CompiledGeometry,
    grid: PreparedTensorGrid,
    domain: MeshingDomain,
) -> None:
    curve = providers.NativeCurveSource(curves, "r1")
    assert_type(curve, NativeCurveSource)
    assert_type(curve.curves, BoundaryAtlas | SegmentMesh)
    implicit = providers.NativeImplicitSource(geometry, grid, "implicit", "r1")
    assert_type(implicit, NativeImplicitSource)
    assert_type(implicit.geometry, CompiledGeometry)
    assert_type(implicit.grid, PreparedTensorGrid)
    surface = providers.NativeSurfaceSource(domain)
    assert_type(surface, NativeSurfaceSource)
    assert_type(surface.domain, MeshingDomain)


def explicit_power_sites(
    plc: NativePlcSource,
    sites: ArrayLike,
    weights: ArrayLike,
    volume: VolumeMeshingSpec,
    coordinates: SpatialCoordinateContract,
) -> None:
    source = providers.NativePolyhedralSource(
        plc.complex,
        plc.source_id,
        plc.source_revision,
        sites=sites,
        weights=weights,
    )
    assert_type(source, NativePolyhedralSource)
    assert_type(
        NativeMeshingProvider(NativeMeshingOptions("plc_restricted_power"))
        .plan(source, volume, coordinate_contract=coordinates)
        .execute(),
        CellMeshingResult,
    )


def hybrid_hex_schedule() -> None:
    options = NativeMeshingOptions("plc_hex_dominant", hex_core_fraction=0.75)
    assert_type(options.hex_core_fraction, float | None)
