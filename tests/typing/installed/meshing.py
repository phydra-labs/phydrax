"""Installed consumers use the canonical generation, adaptation and overset APIs."""

from collections.abc import Mapping
from typing import assert_type

import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from phydrax import SpatialCoordinateContract
from phydrax.discretization import (
    CellMesh,
    PreparedFieldReconstruction,
    PreparedUnstructuredConservativeRemap,
)
from phydrax.discretization.fem import FiniteElementDiscretization
from phydrax.geometry import (
    CompartmentMeshingSource,
    MappedReferenceDomain,
    PiecewiseLinearDomain,
    PlanarMeshRegion,
    PreparedCommonRefinement,
    SourceBoundaryQuery,
)
from phydrax.geometry.multiregion_surface import LabelFieldVolumeBinding
from phydrax.meshing import (
    adapt_device_tetra_metric,
    BackgroundMetricControl,
    BlockInterfaceControl,
    BoundaryLayerMesh,
    CellMeshingResult,
    DecisionBudget,
    DeviceGenerationCandidates,
    DeviceGenerationLayout,
    DeviceGenerationUpdate,
    DeviceTetraMetricLayout,
    DeviceTetraMetricState,
    DeviceTetraMetricUpdate,
    evaluate_device_generation_candidates,
    execute_device_generation_round,
    execute_mesh_adaptation,
    LevelSetEvidence,
    LevelSetMeshAdaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    MeshAssembly,
    MeshPart,
    metric_simplex_quality,
    MixedAdaptationHierarchy,
    MixedLayerColumns,
    NativeHexGridRoute,
    NativeHexGridSchedule,
    NativeLayerCoreSource,
    NativeMappedHexSource,
    NativeMeshingOptions,
    NativeMeshingPhase,
    NativeMeshingPhaseMeasurement,
    NativeMeshingPhaseRecorder,
    NativeMeshingPlan,
    NativeMeshingProvider,
    NativePeriodicSource,
    NativePlanarSource,
    NativePlcSource,
    NativePolyhedralSchedule,
    NativePolyhedralSource,
    NativeStructuredSchedule,
    NativeStructuredSource,
    NativeSurfaceEnvelopeSource,
    NativeSurfaceSchedule,
    NativeSurfaceSource,
    NativeSweepSource,
    OversetConnectivity,
    OversetPartSpec,
    OversetRegistration,
    PolyhedralAdaptationOperation,
    PolyhedralMeshAdaptation,
    prepare_distributed_surface_generation,
    prepare_mesh_adaptation,
    prepare_overset_connectivity,
    prepare_overset_conservative_remap,
    prepare_overset_field_transfer,
    PreparedDeviceGeneration,
    PreparedDistributedSurfaceGeneration,
    PreparedMeshAdaptation,
    PreparedOversetFieldTransfer,
    providers,
    RegionMeshingEvidence,
    SurfaceMeshingSpec,
    SweepControl,
    TransfiniteBlock,
    VolumeMeshingSpec,
)


def native_generation(
    region: PlanarMeshRegion,
    surface: SurfaceMeshingSpec,
    plc: NativePlcSource,
    volume: VolumeMeshingSpec,
    periodic: NativePeriodicSource,
    coordinates: SpatialCoordinateContract,
) -> None:
    source = providers.NativePlanarSource(region, "r1")
    assert_type(source, NativePlanarSource)
    plan = NativeMeshingProvider(
        NativeMeshingOptions("planar_constrained_delaunay")
    ).plan(source, surface, coordinate_contract=coordinates)
    assert_type(plan, NativeMeshingPlan)
    result = plan.execute()
    assert_type(result, CellMeshingResult)
    assert_type(result.mesh, CellMesh)
    polyhedral = NativeMeshingProvider(
        NativeMeshingOptions(
            "plc_restricted_power",
            polyhedral_schedule=NativePolyhedralSchedule(maximum_refinement_steps=2),
        )
    ).plan(plc, volume, coordinate_contract=coordinates)
    assert_type(polyhedral.execute(), CellMeshingResult)
    periodic_plan = NativeMeshingProvider(NativeMeshingOptions("periodic_delaunay")).plan(
        periodic, surface, coordinate_contract=coordinates
    )
    assert_type(periodic_plan.execute(), CellMeshingResult)
    NativeMeshingOptions("tetgen")  # ty: ignore[invalid-argument-type]
    NativeMeshingProvider("native")  # ty: ignore[invalid-argument-type]
    providers.NativePlanarSource(plc, "r1")  # ty: ignore[invalid-argument-type]
    plan.execute("r2")  # ty: ignore[invalid-argument-type]


def native_adaptation(source: CellMeshingResult, request: LevelSetMeshAdaptation) -> None:
    assert_type(request.previous, LevelSetEvidence | None)
    assert_type(request.values, Array)
    prepared = prepare_mesh_adaptation(
        source,
        request,
        policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_LEVEL_SET),
    )
    assert_type(prepared, PreparedMeshAdaptation)
    adapted = execute_mesh_adaptation(prepared)
    assert_type(adapted, MeshAdaptationResult)
    assert_type(adapted.target, CellMeshingResult)
    prepare_mesh_adaptation(source, request)  # ty: ignore[missing-argument]


def overset_transfer(
    assembly: MeshAssembly,
    parts: tuple[OversetPartSpec, ...],
    discretizations: Mapping[
        str, FiniteElementDiscretization | PreparedFieldReconstruction
    ],
    values: Mapping[str, ArrayLike],
    source: MeshPart,
    target: MeshPart,
) -> None:
    connectivity = prepare_overset_connectivity(assembly, parts)
    assert_type(connectivity, OversetConnectivity)
    registration = connectivity.registration()
    assert_type(registration, OversetRegistration)
    assert_type(registration.prepare(), OversetConnectivity)
    transfer = prepare_overset_field_transfer(connectivity, discretizations)
    assert_type(transfer, PreparedOversetFieldTransfer)
    assert_type(transfer.apply(values), dict[str, Array])
    assert_type(transfer.transpose(values), dict[str, Array])
    conservative = prepare_overset_conservative_remap(source, target)
    assert_type(conservative, PreparedUnstructuredConservativeRemap)
    prepare_overset_connectivity(assembly, "parts")  # ty: ignore[invalid-argument-type]


def decision_budget() -> None:
    budget = DecisionBudget(
        tolerance=0.01,
        maximum_wall_seconds=1.0,
        maximum_memory_bytes=4096,
        maximum_dofs=100,
        maximum_condition=10.0,
    )
    assert_type(budget, DecisionBudget)
    assert_type(budget.maximum_dofs, int)
    DecisionBudget(tolerance=0.01)  # ty: ignore[missing-argument]


def supported_source_bindings(
    blocks: tuple[TransfiniteBlock, ...],
    interfaces: tuple[BlockInterfaceControl, ...],
    fidelity: SourceBoundaryQuery,
    domain: PiecewiseLinearDomain,
    profile: CellMesh,
    sweep: SweepControl,
    layers: NativeLayerCoreSource,
    envelope: NativeSurfaceEnvelopeSource,
    surface: SurfaceMeshingSpec,
    volume: VolumeMeshingSpec,
    coordinates: SpatialCoordinateContract,
) -> None:
    structured = providers.NativeStructuredSource(
        blocks,
        interfaces,
        fidelity.source_id,
        fidelity.source_revision,
        fidelity_source=fidelity,
        maximum_deviation=0.01,
    )
    assert_type(structured, NativeStructuredSource)
    assert_type(structured.fidelity_source, SourceBoundaryQuery)
    assert_type(structured.domain, PiecewiseLinearDomain | None)
    assert_type(structured.block_regions, tuple[tuple[str, str], ...])
    schedule = NativeStructuredSchedule(optimization_steps=2)
    assert_type(schedule, NativeStructuredSchedule)
    assert_type(
        NativeMeshingProvider(
            NativeMeshingOptions("structured_transfinite", structured_schedule=schedule)
        )
        .plan(structured, surface, coordinate_contract=coordinates)
        .execute(),
        CellMeshingResult,
    )
    swept = providers.NativeSweepSource(
        profile,
        sweep,
        fidelity.source_id,
        fidelity.source_revision,
        fidelity_source=fidelity,
        maximum_deviation=0.01,
        domain=domain,
    )
    assert_type(swept, NativeSweepSource)
    assert_type(swept.domain, PiecewiseLinearDomain | MappedReferenceDomain | None)
    assert_type(
        NativeMeshingProvider(NativeMeshingOptions("sweep"))
        .plan(swept, volume, coordinate_contract=coordinates)
        .execute(),
        CellMeshingResult,
    )
    assert_type(layers.layers, BoundaryLayerMesh)
    assert_type(
        NativeMeshingProvider(NativeMeshingOptions("layer_core"))
        .plan(layers, volume, coordinate_contract=coordinates)
        .execute(),
        CellMeshingResult,
    )
    assert_type(envelope.plc_source, NativePlcSource)
    assert_type(
        NativeMeshingProvider(NativeMeshingOptions("surface_envelope_tetrahedral"))
        .plan(envelope, volume, coordinate_contract=coordinates)
        .execute(),
        CellMeshingResult,
    )


def mixed_layer_adaptation(
    source: CellMeshingResult,
    cells: ArrayLike,
    columns: ArrayLike,
    intervals: ArrayLike,
) -> None:
    hierarchy = MixedAdaptationHierarchy()
    layer_columns = MixedLayerColumns(cells, columns, intervals)
    assert_type(hierarchy, MixedAdaptationHierarchy)
    assert_type(layer_columns, MixedLayerColumns)
    request = MarkedMeshAdaptation(
        cells, hierarchy=hierarchy, layer_columns=layer_columns
    )
    assert_type(request, MarkedMeshAdaptation)
    assert_type(request.layer_columns, MixedLayerColumns | None)
    prepared = prepare_mesh_adaptation(
        source,
        request,
        policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_MIXED),
    )
    assert_type(prepared, PreparedMeshAdaptation)
    assert_type(execute_mesh_adaptation(prepared), MeshAdaptationResult)


def weighted_polyhedral_source(
    source: NativePolyhedralSource,
    volume: VolumeMeshingSpec,
    coordinates: SpatialCoordinateContract,
) -> None:
    plan = NativeMeshingProvider(NativeMeshingOptions("plc_restricted_power")).plan(
        source, volume, coordinate_contract=coordinates
    )
    assert_type(plan, NativeMeshingPlan)
    assert_type(plan.execute(), CellMeshingResult)


def bounded_device_worksets(
    prepared: PreparedDeviceGeneration,
    metric_layout: DeviceTetraMetricLayout,
    metric_state: DeviceTetraMetricState,
) -> None:
    candidates = evaluate_device_generation_candidates(prepared)
    assert_type(candidates, DeviceGenerationCandidates)
    update = execute_device_generation_round(prepared, candidates=candidates)
    assert_type(update, DeviceGenerationUpdate)
    assert_type(update.prepared, PreparedDeviceGeneration)
    metric_update = adapt_device_tetra_metric(metric_layout, metric_state)
    assert_type(metric_update, DeviceTetraMetricUpdate)
    assert_type(metric_update.state, DeviceTetraMetricState)


def image_material_sources(
    compartments: CompartmentMeshingSource,
    reconstructed: LabelFieldVolumeBinding,
    volume: VolumeMeshingSpec,
) -> None:
    provider = NativeMeshingProvider(NativeMeshingOptions("image_material_tetrahedral"))
    assert_type(compartments.coordinate_contract, SpatialCoordinateContract)
    occupied_plan = provider.plan(
        compartments, volume, coordinate_contract=compartments.coordinate_contract
    )
    assert_type(occupied_plan, NativeMeshingPlan)
    occupied = occupied_plan.execute()
    assert_type(occupied, CellMeshingResult)
    assert_type(occupied.region_evidence, RegionMeshingEvidence | None)
    assert_type(reconstructed.coordinate_contract, SpatialCoordinateContract)
    reconstructed_plan = provider.plan(
        reconstructed,
        volume,
        coordinate_contract=reconstructed.coordinate_contract,
    )
    assert_type(reconstructed_plan.execute(), CellMeshingResult)


def explicit_polyhedral_edit(source: CellMeshingResult, planes: ArrayLike) -> None:
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.PLANE_SPLIT, planes=planes
    )
    assert_type(request, PolyhedralMeshAdaptation)
    assert_type(request.operation, PolyhedralAdaptationOperation)
    assert_type(request.planes, Array)
    assert_type(request.request_id, str)
    prepared = prepare_mesh_adaptation(
        source,
        request,
        policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_POLYHEDRAL),
    )
    assert_type(prepared, PreparedMeshAdaptation)
    result = execute_mesh_adaptation(prepared)
    assert_type(result, MeshAdaptationResult)
    assert_type(result.target, CellMeshingResult)
    assert_type(result.common_refinement, PreparedCommonRefinement | None)


def metric_simplex_contract(
    metrics: ArrayLike, coordinates: ArrayLike, cells: ArrayLike
) -> None:
    assert_type(metric_simplex_quality(metrics, coordinates, cells), Array)


def execution_phase_annotations(
    plan: NativeMeshingPlan,
    record_phase: NativeMeshingPhaseRecorder,
    measurement: NativeMeshingPhaseMeasurement,
) -> None:
    assert_type(measurement.phase, NativeMeshingPhase)
    assert_type(measurement.elapsed_seconds, float)
    assert_type(measurement.work_units, int | None)
    assert_type(measurement.invocations, int)
    assert_type(plan.execute(record_phase=record_phase), CellMeshingResult)


def physical_background_metric(
    surface: SurfaceMeshingSpec,
    control: BackgroundMetricControl,
) -> None:
    request = SurfaceMeshingSpec(
        surface.target,
        surface.scope,
        planar_embedding=surface.planar_embedding,
        size_controls=surface.size_controls,
        background_metric=control,
    )
    assert_type(request.background_metric, BackgroundMetricControl | None)


def mapped_grid_annotations(
    reference: NativePlcSource,
    domain: MappedReferenceDomain,
    specification: VolumeMeshingSpec,
    coordinates: SpatialCoordinateContract,
) -> None:
    source = NativeMappedHexSource(reference, domain)
    assert_type(source.domain, MappedReferenceDomain)
    schedule = NativeHexGridSchedule("balanced_grid")
    assert_type(schedule.route, NativeHexGridRoute)
    options = NativeMeshingOptions("mapped_balanced_grid_hex", grid_schedule=schedule)
    assert_type(options.grid_schedule, NativeHexGridSchedule | None)
    plan = NativeMeshingProvider(options).plan(
        source,
        specification,
        coordinate_contract=coordinates,
    )
    assert_type(plan.execute(), CellMeshingResult)


def partitioned_initial_generation(
    source: NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    schedule: NativeSurfaceSchedule,
    layout: DeviceGenerationLayout,
    patch_owners: NDArray[np.int32],
    coordinates: SpatialCoordinateContract,
) -> None:
    partition = prepare_distributed_surface_generation(
        source,
        specification,
        schedule,
        layout,
        patch_owners,
        maximum_metadata_bytes=1_048_576,
    )
    assert_type(partition, PreparedDistributedSurfaceGeneration)
    plan = NativeMeshingProvider(
        NativeMeshingOptions("parametric_surface", surface_schedule=schedule),
    ).plan(
        source,
        specification,
        coordinate_contract=coordinates,
        initial_partition=partition,
    )
    assert_type(plan, NativeMeshingPlan)
    assert_type(plan.execute(), CellMeshingResult)
