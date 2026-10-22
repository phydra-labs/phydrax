#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native Phydrax meshing provider: typed admission, one prepared route, audit.

`NativeMeshingProvider` admits a revision-bound native source and a physical
meshing specification for the route selected by `NativeMeshingOptions`, and
prepares exactly that route. `NativeMeshingPlan.execute` runs the prepared
route (it never switches routes after a failure) and returns a
`CellMeshingResult` only after the audit, the route certification, and
specification compliance pass.
Routes are added by extending `NativeMeshingRoute` and the central dispatch.
"""

from __future__ import annotations

import os
import threading
import weakref
from contextvars import ContextVar
from typing import assert_never, final, TYPE_CHECKING, TypedDict

import equinox as eqx
import jax
import numpy as np

import phydrax.geometry.design._schema as schema_owner
from phydrax.execution import ResourceRequest
from phydrax.meshing._contracts import (
    CurveMeshingSpec,
    MeshingCapability,
    MeshingExecutionMode,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceDescriptor,
    MeshingSourceKind,
    MeshingSpecification,
    ProviderSupportReport,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._coordinate_operands import CoordinateOperandCollector
from ...geometry._compartments import CompartmentMeshingSource
from ...geometry.brep._carrier_pieces import admit_carrier_fingerprint, carrier_payload
from ...geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    ImplicitSurfacePlan,
    ImplicitSurfacePolicy,
)
from ...geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from ...typing import checked
from .. import NativeHexGridSchedule, NativePolyhedralSchedule, NativeVolumeSchedule
from .._distributed_generation import PreparedDistributedSurfaceGeneration
from .._distributed_planar_generation import PreparedDistributedPlanarGeneration
from .._implicit_volume import (
    adaptive_implicit_volume_support_issues,
    execute_adaptive_implicit_volume,
    execute_restricted_implicit_volume,
    implicit_volume_support_issues,
    prepare_adaptive_implicit_volume,
    prepare_restricted_implicit_volume,
    PreparedAdaptiveImplicitVolume,
    PreparedImplicitVolume,
)
from .._measurements import (
    measure_phase,
    NativeExecutionImports,
    NativeExecutionRecord,
    NativeMeshingPhaseRecorder,
)
from .._native_preparation import (
    native_preparation_ledger,
    NativePreparationOverlap,
    stage_native_completed_source_issue,
    stage_native_source_publication,
)
from .._result import CellMeshingResult
from .._session import MeshingExecutionPolicy
from .._surface_envelope import (
    envelope_volume_support_issues,
    execute_surface_envelope_route,
    PreparedSurfaceEnvelopeVolume,
)
from .._trace import MeshingStageKind
from .._volume_generation import (
    native_volume_execution_budget,
    require_native_preparation_allowance,
)
from . import _native_periodic as periodic_owner, _native_sources as sources_owner
from ._implicit import (
    execute_adaptive_implicit_route,
    execute_implicit_route,
    implicit_support_issues,
    prepare_adaptive_implicit_route,
    prepare_implicit_route,
)
from ._native_cad_volume import (
    cad_volume_support_issues,
    execute_cad_volume_route,
    NativeCadVolumeSource,
    PreparedCadVolume,
)
from ._native_compartment import (
    compartment_support_issues,
    execute_compartment_route,
    PreparedCompartmentVolume,
)
from ._native_curve import (
    curve_preparation_upper,
    curve_support_issues,
    execute_curve_route,
    PreparedCurveNetwork,
)
from ._native_dual import (
    dual_hex_support_issues,
    dual_quad_support_issues,
    execute_grid_hex_route,
    execute_hex_dominant_route,
    execute_hex_route,
    execute_mapped_grid_hex_route,
    execute_quad_route,
    hex_dominant_support_issues,
    mapped_grid_support_issues,
    PreparedDualHex,
    PreparedDualQuad,
    PreparedGridHex,
    PreparedHexDominant,
    PreparedMappedGridHex,
)
from ._native_layer import (
    execute_layer_route,
    layer_core_support_issues,
    PreparedLayerCore,
)
from ._native_options import NativeMeshingOptions
from ._native_planar import (
    execute_planar_route,
    planar_support_issues,
)
from ._native_planar_distributed import execute_distributed_planar_route
from ._native_planar_domain import PreparedPlanarDomain
from ._native_polyhedral import (
    execute_polyhedral_route,
    polyhedral_support_issues,
    PreparedPolyhedralVolume,
)
from ._native_publication import bind_native_execution_result
from ._native_route_metadata import specification_scope
from ._native_schedules import (
    NativeCurveSchedule,
    NativeStructuredSchedule,
    NativeSurfaceSchedule,
)
from ._native_source_type import NativeMeshingSource
from ._native_structured import (
    execute_structured_route,
    execute_sweep_route,
    PreparedStructured,
    PreparedSweep,
    structured_support_issues,
)
from ._native_surface import (
    execute_parametric_surface_route,
    parametric_surface_support_issues,
    PreparedParametricSurface,
)
from ._native_volume import (
    execute_volume_route,
    PreparedPlcVolume,
    volume_support_issues,
)


if TYPE_CHECKING:
    from ._native_session import NativeMeshingSession


type _PreparedRoute = (
    AdaptiveImplicitSurface
    | ImplicitSurfacePlan
    | PreparedCurveNetwork
    | PreparedPlanarDomain
    | PreparedParametricSurface
    | PreparedPlcVolume
    | PreparedDistributedPlanarGeneration
    | PreparedCadVolume
    | periodic_owner.PreparedPeriodicDomain
    | PreparedPolyhedralVolume
    | PreparedLayerCore
    | PreparedSurfaceEnvelopeVolume
    | PreparedStructured
    | PreparedSweep
    | PreparedCompartmentVolume
    | PreparedDualHex
    | PreparedDualQuad
    | PreparedHexDominant
    | PreparedGridHex
    | PreparedMappedGridHex
    | PreparedImplicitVolume
    | PreparedAdaptiveImplicitVolume
)

_root_work_admission: ContextVar[tuple[object, int] | None] = ContextVar(
    "native_plan_root_work_admission", default=None
)


def _prepared_id(prepared: _PreparedRoute, /) -> str:
    match prepared:
        case ImplicitSurfacePlan():
            return prepared.plan_id
        case AdaptiveImplicitSurface():
            return prepared.evidence.evidence_id
        case (
            PreparedCurveNetwork()
            | PreparedPlanarDomain()
            | PreparedParametricSurface()
            | PreparedPlcVolume()
            | PreparedDistributedPlanarGeneration()
            | PreparedCadVolume()
            | periodic_owner.PreparedPeriodicDomain()
            | PreparedPolyhedralVolume()
            | PreparedLayerCore()
            | PreparedSurfaceEnvelopeVolume()
            | PreparedStructured()
            | PreparedSweep()
            | PreparedCompartmentVolume()
            | PreparedDualHex()
            | PreparedDualQuad()
            | PreparedHexDominant()
            | PreparedGridHex()
            | PreparedMappedGridHex()
            | PreparedImplicitVolume()
            | PreparedAdaptiveImplicitVolume()
        ):
            return prepared.prepared_id
        case _:
            assert_never(prepared)


@final
class _PlanarRequest(StrictModule):
    source: sources_owner.NativePlanarSource
    specification: SurfaceMeshingSpec


@final
class _ImplicitRequest(StrictModule):
    source: sources_owner.NativeImplicitSource
    specification: SurfaceMeshingSpec
    policy: AdaptiveImplicitSurfacePolicy | ImplicitSurfacePolicy


@final
class _CurveRequest(StrictModule):
    source: sources_owner.NativeCurveSource
    specification: CurveMeshingSpec
    schedule: NativeCurveSchedule


@final
class _SurfaceRequest(StrictModule):
    source: sources_owner.NativeSurfaceSource
    specification: SurfaceMeshingSpec
    schedule: NativeSurfaceSchedule


@final
class _CadVolumeRequest(StrictModule):
    source: NativeCadVolumeSource
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule


@final
class _PlcRequest(StrictModule):
    source: sources_owner.NativePlcSource
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule


@final
class _PolyhedralRequest(StrictModule):
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource
    specification: VolumeMeshingSpec
    schedule: NativePolyhedralSchedule


@final
class _PeriodicRequest(StrictModule):
    source: periodic_owner.NativePeriodicSource
    specification: SurfaceMeshingSpec | VolumeMeshingSpec


@final
class _LayerRequest(StrictModule):
    source: sources_owner.NativeLayerCoreSource
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule


@final
class _EnvelopeRequest(StrictModule):
    source: sources_owner.NativeSurfaceEnvelopeSource
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule


@final
class _StructuredRequest(StrictModule):
    source: sources_owner.NativeStructuredSource
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    schedule: NativeStructuredSchedule


@final
class _SweepRequest(StrictModule):
    source: sources_owner.NativeSweepSource
    specification: VolumeMeshingSpec


@final
class _CompartmentRequest(StrictModule):
    source: CompartmentMeshingSource | LabelFieldVolumeBinding
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule


@final
class _MappedHexRequest(StrictModule):
    source: sources_owner.NativeMappedHexSource
    specification: VolumeMeshingSpec
    grid: NativeHexGridSchedule


@final
class _ImplicitVolumeRequest(StrictModule):
    source: sources_owner.NativeImplicitSource
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule


@final
class _AdaptiveImplicitVolumeRequest(StrictModule):
    source: sources_owner.NativeImplicitSource
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    policy: AdaptiveImplicitSurfacePolicy


@final
class _GridHexRequest(StrictModule):
    source: sources_owner.NativePlcSource
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    grid: NativeHexGridSchedule


@final
class _HexDominantRequest(StrictModule):
    source: sources_owner.NativePlcSource
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    core_fraction: float


type _BoundRequest = (
    _PlanarRequest
    | _ImplicitRequest
    | _CurveRequest
    | _SurfaceRequest
    | _PlcRequest
    | _CadVolumeRequest
    | _PolyhedralRequest
    | _PeriodicRequest
    | _LayerRequest
    | _EnvelopeRequest
    | _StructuredRequest
    | _SweepRequest
    | _CompartmentRequest
    | _MappedHexRequest
    | _ImplicitVolumeRequest
    | _AdaptiveImplicitVolumeRequest
    | _GridHexRequest
    | _HexDominantRequest
)


def _required_schedule[T](value: T | None, field: str, /) -> T:
    """Return the resolved schedule required by a selected route."""
    if value is None:
        raise ValueError(f"{field} must resolve for the selected native route; got None")
    return value


# Keep exhaustive route/source/specification binding explicit, including ordered schedule validation.
def _bind_request(
    options: NativeMeshingOptions,
    source: NativeMeshingSource,
    specification: MeshingSpecification,
    /,
) -> _BoundRequest:
    """Bind scientific input kinds and resolved schedules once for a native route."""
    route = options.route
    match (route, source, specification):
        case (
            "planar_constrained_delaunay" | "planar_dual_quad",
            sources_owner.NativePlanarSource(),
            SurfaceMeshingSpec(),
        ):
            return _PlanarRequest(source, specification)
        case (
            "implicit_surface",
            sources_owner.NativeImplicitSource(),
            SurfaceMeshingSpec(),
        ):
            return _ImplicitRequest(
                source,
                specification,
                _required_schedule(
                    options.implicit_policy, "NativeMeshingOptions.implicit_policy"
                ),
            )
        case ("curve_arc_length", sources_owner.NativeCurveSource(), CurveMeshingSpec()):
            return _CurveRequest(
                source,
                specification,
                _required_schedule(
                    options.curve_schedule, "NativeMeshingOptions.curve_schedule"
                ),
            )
        case (
            "parametric_surface",
            sources_owner.NativeSurfaceSource(),
            SurfaceMeshingSpec(),
        ):
            return _SurfaceRequest(
                source,
                specification,
                _required_schedule(
                    options.surface_schedule, "NativeMeshingOptions.surface_schedule"
                ),
            )
        case (
            "plc_restricted_power",
            sources_owner.NativePlcSource() | sources_owner.NativePolyhedralSource(),
            VolumeMeshingSpec(),
        ):
            return _PolyhedralRequest(
                source,
                specification,
                _required_schedule(
                    options.polyhedral_schedule,
                    "NativeMeshingOptions.polyhedral_schedule",
                ),
            )
        case ("cad_tetrahedral", NativeCadVolumeSource(), VolumeMeshingSpec()):
            return _CadVolumeRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
            )
        case (
            "periodic_delaunay",
            periodic_owner.NativePeriodicSource(),
            SurfaceMeshingSpec() | VolumeMeshingSpec(),
        ):
            return _PeriodicRequest(source, specification)
        case ("layer_core", sources_owner.NativeLayerCoreSource(), VolumeMeshingSpec()):
            return _LayerRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
            )
        case (
            "surface_envelope_tetrahedral",
            sources_owner.NativeSurfaceEnvelopeSource(),
            VolumeMeshingSpec(),
        ):
            return _EnvelopeRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
            )
        case (
            "structured_transfinite",
            sources_owner.NativeStructuredSource(),
            SurfaceMeshingSpec() | VolumeMeshingSpec(),
        ):
            return _StructuredRequest(
                source,
                specification,
                _required_schedule(
                    options.structured_schedule,
                    "NativeMeshingOptions.structured_schedule",
                ),
            )
        case ("sweep", sources_owner.NativeSweepSource(), VolumeMeshingSpec()):
            return _SweepRequest(source, specification)
        case (
            "image_material_tetrahedral",
            CompartmentMeshingSource() | LabelFieldVolumeBinding(),
            VolumeMeshingSpec(),
        ):
            return _CompartmentRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
            )
        case (
            "mapped_balanced_grid_hex" | "mapped_frame_grid_hex",
            sources_owner.NativeMappedHexSource(),
            VolumeMeshingSpec(),
        ):
            return _MappedHexRequest(
                source,
                specification,
                _required_schedule(
                    options.grid_schedule, "NativeMeshingOptions.grid_schedule"
                ),
            )
        case (
            "implicit_restricted_delaunay",
            sources_owner.NativeImplicitSource(),
            VolumeMeshingSpec(),
        ):
            if (
                source.geometry.ambient_dimension != 3
                or len(source.grid.structured_axes) != 3
            ):
                raise ValueError(
                    "NativeMeshingSource.grid must enclose three physical dimensions; got an incompatible implicit enclosure"
                )
            return _ImplicitVolumeRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
            )
        case (
            "implicit_adaptive_tetrahedral",
            sources_owner.NativeImplicitSource(),
            VolumeMeshingSpec(),
        ):
            if (
                source.geometry.ambient_dimension != 3
                or len(source.grid.structured_axes) != 3
            ):
                raise ValueError(
                    "NativeMeshingSource.grid must enclose three physical dimensions; got an incompatible implicit enclosure"
                )
            match options.implicit_policy:
                case AdaptiveImplicitSurfacePolicy() as policy:
                    pass
                case _:
                    raise ValueError(
                        "NativeMeshingOptions.implicit_policy must use adaptive discovery for this route; got an incompatible policy"
                    )
            return _AdaptiveImplicitVolumeRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
                policy,
            )
        case (
            "plc_tetrahedral" | "plc_dual_hex",
            sources_owner.NativePlcSource(),
            VolumeMeshingSpec(),
        ):
            return _PlcRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
            )
        case ("plc_hex_dominant", sources_owner.NativePlcSource(), VolumeMeshingSpec()):
            return _HexDominantRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
                _required_schedule(
                    options.hex_core_fraction, "NativeMeshingOptions.hex_core_fraction"
                ),
            )
        case (
            "plc_balanced_grid_hex" | "plc_frame_grid_hex",
            sources_owner.NativePlcSource(),
            VolumeMeshingSpec(),
        ):
            return _GridHexRequest(
                source,
                specification,
                _required_schedule(
                    options.volume_schedule, "NativeMeshingOptions.volume_schedule"
                ),
                _required_schedule(
                    options.grid_schedule, "NativeMeshingOptions.grid_schedule"
                ),
            )
        case _:
            raise TypeError(
                "NativeMeshingProvider.request must match the selected route source and specification kinds; "
                f"got {type(source).__name__}, {type(specification).__name__} for {route!r}"
            )


def _execute_implicit_plan(
    source: sources_owner.NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    prepared: AdaptiveImplicitSurface | ImplicitSurfacePlan,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    state: schema_owner.DesignState | None,
    /,
    *,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Keep fixed-route design realization separate from adaptive discovery."""
    match prepared:
        case AdaptiveImplicitSurface():
            return execute_adaptive_implicit_route(
                source,
                specification,
                prepared,
                coordinate_contract,
                provider,
                plan_id,
                scope=scope,
                workspace=workspace,
                record_phase=record_phase,
            )
        case ImplicitSurfacePlan():
            return execute_implicit_route(
                source,
                specification,
                prepared,
                coordinate_contract,
                provider,
                plan_id,
                state,
                scope=scope,
                workspace=workspace,
                record_phase=record_phase,
            )
        case _:
            assert_never(prepared)


def _prepared_execution_charge(prepared: _PreparedRoute, /) -> tuple[int, int, float]:
    """Carry source-owned preparation charges; no inferred instruction counts."""
    match prepared:
        case PreparedAdaptiveImplicitVolume():
            return (
                prepared.source_work_units,
                prepared.geometry_queries,
                prepared.preparation_seconds,
            )
        case PreparedImplicitVolume():
            return 0, prepared.geometry_query_charge, 0.0
        case PreparedLayerCore():
            return prepared.source_work_units, 0, 0.0
        case AdaptiveImplicitSurface() | ImplicitSurfacePlan():
            raise ValueError(
                "NativeMeshingPlan.preparation_evidence must bind implicit discovery "
                "to its measured native preparation receipt; got missing receipt"
            )
        case _:
            return 0, 0, 0.0


@final
class _PreparedRequest(StrictModule):
    prepared: _PreparedRoute
    specification: CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec


def _implicit_enclosure(source: sources_owner.NativeImplicitSource, /) -> np.ndarray:
    axes = jax.device_get(
        tuple(axis.point_coordinates for axis in source.grid.structured_axes)
    )
    host_axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    return np.asarray(
        [(np.min(axis), np.max(axis)) for axis in host_axes], dtype=np.float64
    ).T


# This exhaustive dispatch owns route-specific preparation under the same live allowance.
def _prepare_bound_request(
    bound: _BoundRequest,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    scope: NativeExecutionScope,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
    initial_partition: PreparedDistributedSurfaceGeneration
    | PreparedDistributedPlanarGeneration
    | None,
    source_preparation_accounted: bool,
) -> _PreparedRequest:
    """Prepare the carrier under the bound schedules and live native allowance."""
    match (options.route, bound):
        case (
            "planar_constrained_delaunay",
            _PlanarRequest(source=source, specification=specification),
        ):
            if isinstance(initial_partition, PreparedDistributedPlanarGeneration):
                initial_partition.require_binding(source, specification)
                return _PreparedRequest(initial_partition, specification)
            if initial_partition is not None:
                raise ValueError(
                    "NativeMeshingProvider.initial_partition must be a prepared planar generation for this route; got another source kind"
                )
            prepared = PreparedPlanarDomain(source, specification, lattice=True)
            return _PreparedRequest(prepared, specification)
        case (
            "implicit_surface",
            _ImplicitRequest(source=source, specification=specification, policy=policy),
        ):
            match policy:
                case AdaptiveImplicitSurfacePolicy():
                    prepared = prepare_adaptive_implicit_route(
                        source, specification, policy, scope=scope
                    )
                case ImplicitSurfacePolicy():
                    prepared = prepare_implicit_route(
                        source, specification, policy, scope=scope
                    )
                case _:
                    assert_never(policy)
            return _PreparedRequest(prepared, specification)
        case (
            "curve_arc_length",
            _CurveRequest(source=source, specification=specification, schedule=schedule),
        ):
            work_upper, queries_upper = curve_preparation_upper(specification)
            with scope.batch(work_upper, queries_upper=queries_upper) as ticket:
                prepared = PreparedCurveNetwork(
                    source, specification, schedule, ticket=ticket
                )
            return _PreparedRequest(prepared, specification)
        case (
            "parametric_surface",
            _SurfaceRequest(
                source=source, specification=specification, schedule=schedule
            ),
        ):
            if initial_partition is not None and not isinstance(
                initial_partition, PreparedDistributedSurfaceGeneration
            ):
                raise ValueError(
                    "NativeMeshingProvider.initial_partition must be a prepared surface generation for this route; got another source kind"
                )
            prepared = PreparedParametricSurface(
                source,
                specification,
                schedule,
                scope=scope,
                initial_partition=initial_partition,
            )
            return _PreparedRequest(prepared, specification)
        case (
            "plc_tetrahedral",
            _PlcRequest(source=source, specification=specification, schedule=schedule),
        ):
            prepared = PreparedPlcVolume(source, specification, schedule)
            return _PreparedRequest(prepared, specification)
        case (
            "cad_tetrahedral",
            _CadVolumeRequest(
                source=source, specification=specification, schedule=schedule
            ),
        ):
            prepared = PreparedCadVolume(source, specification, schedule)
            return _PreparedRequest(prepared, specification)
        case (
            "periodic_delaunay",
            _PeriodicRequest(source=source, specification=specification),
        ):
            prepared = periodic_owner.PreparedPeriodicDomain(source, specification)
            return _PreparedRequest(prepared, specification)
        case (
            "plc_restricted_power",
            _PolyhedralRequest(
                source=source, specification=specification, schedule=schedule
            ),
        ):
            prepared = PreparedPolyhedralVolume(source, specification, schedule)
            return _PreparedRequest(prepared, specification)
        case (
            "layer_core",
            _LayerRequest(source=source, specification=specification, schedule=schedule),
        ):
            prepared = PreparedLayerCore(source, specification, schedule, scope=scope)
            return _PreparedRequest(prepared, specification)
        case (
            "surface_envelope_tetrahedral",
            _EnvelopeRequest(
                source=source, specification=specification, schedule=schedule
            ),
        ):
            if not source_preparation_accounted and (
                not native_preparation_ledger(scope).source_is_active(
                    scope, source.envelope.envelope_id
                )
            ):
                raise ValueError(
                    "NativeMeshingProvider.envelope.preparation must bind ended "
                    "execution evidence or its live source; got no active source preparation"
                )
            prepared = PreparedSurfaceEnvelopeVolume(source, specification, schedule)
            return _PreparedRequest(prepared, specification)
        case (
            "structured_transfinite",
            _StructuredRequest(
                source=source, specification=specification, schedule=schedule
            ),
        ):
            prepared = PreparedStructured(
                source.blocks,
                source.interfaces,
                specification,
                source.binding_id,
                optimization_steps=schedule.optimization_steps,
            )
            return _PreparedRequest(prepared, specification)
        case ("sweep", _SweepRequest(source=source, specification=specification)):
            prepared = PreparedSweep(
                source.profile,
                source.control,
                specification,
                source.binding_id,
                profile_geometry=source.profile_geometry,
                scope=scope,
            )
            return _PreparedRequest(prepared, specification)
        case (
            "image_material_tetrahedral",
            _CompartmentRequest(
                source=source, specification=specification, schedule=schedule
            ),
        ):
            prepared = PreparedCompartmentVolume(source, specification, schedule)
            return _PreparedRequest(prepared, specification)
        case (
            "planar_dual_quad",
            _PlanarRequest(source=source, specification=specification),
        ):
            prepared = PreparedDualQuad(source, specification)
            return _PreparedRequest(prepared, specification)
        case (
            "plc_dual_hex",
            _PlcRequest(source=source, specification=specification, schedule=schedule),
        ):
            prepared = PreparedDualHex(source, specification, schedule)
            return _PreparedRequest(prepared, specification)
        case (
            "plc_hex_dominant",
            _HexDominantRequest(
                source=source,
                specification=specification,
                schedule=schedule,
                core_fraction=core_fraction,
            ),
        ):
            prepared = PreparedHexDominant(
                source, specification, schedule, core_fraction=core_fraction
            )
            return _PreparedRequest(prepared, specification)
        case (
            "plc_balanced_grid_hex" | "plc_frame_grid_hex",
            _GridHexRequest(
                source=source, specification=specification, schedule=schedule, grid=grid
            ),
        ):
            prepared = PreparedGridHex(source, specification, schedule, grid)
            return _PreparedRequest(prepared, specification)
        case (
            "mapped_balanced_grid_hex" | "mapped_frame_grid_hex",
            _MappedHexRequest(source=source, specification=specification, grid=grid),
        ):
            prepared = PreparedMappedGridHex(
                source, specification, grid, scope=scope, record_phase=record_phase
            )
            return _PreparedRequest(prepared, specification)
        case (
            "implicit_restricted_delaunay",
            _ImplicitVolumeRequest(
                source=source, specification=specification, schedule=schedule
            ),
        ):
            domain = _implicit_enclosure(source)
            prepared = prepare_restricted_implicit_volume(
                source.geometry,
                domain,
                specification,
                schedule,
                coordinate_contract,
                scope=scope,
            )
            return _PreparedRequest(prepared, specification)
        case (
            "implicit_adaptive_tetrahedral",
            _AdaptiveImplicitVolumeRequest(
                source=source,
                specification=specification,
                schedule=schedule,
                policy=policy,
            ),
        ):
            domain = _implicit_enclosure(source)
            prepared = prepare_adaptive_implicit_volume(
                source.geometry,
                domain,
                specification,
                schedule,
                policy,
                source.source_revision,
                coordinate_contract,
                scope=scope,
            )
            return _PreparedRequest(prepared, specification)
        case _:
            raise TypeError(
                "NativeMeshingProvider.route must bind a supported request; got inconsistent route state"
            )


class _ExecutionKeywords(TypedDict):
    scope: NativeExecutionScope
    workspace: NativeHostStorageWorkspace
    record_phase: NativeMeshingPhaseRecorder | None


@final
class NativeMeshingPlan(StrictModule, NonTrainableState):
    """One admitted request prepared on one native route."""

    source: NativeMeshingSource
    specification: CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec
    options: NativeMeshingOptions
    support: ProviderSupportReport
    coordinate_contract: SpatialCoordinateContract
    prepared: _PreparedRoute
    plan_id: str = eqx.field(static=True)
    preparation_evidence: NativeExecutionRecord | None

    @checked
    def __init__(
        self,
        source: NativeMeshingSource,
        specification: CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec,
        options: NativeMeshingOptions,
        support: ProviderSupportReport,
        coordinate_contract: SpatialCoordinateContract,
        prepared: _PreparedRoute,
        /,
        *,
        preparation_evidence: NativeExecutionRecord | None = None,
    ) -> None:
        bound = _bind_request(options, source, specification)
        if (
            support.specification_id != specification.specification_id
            or support.provider_id != NativeMeshingProvider.info().provider_id
        ):
            raise ValueError(
                "NativeMeshingPlan.support must admit the specification and provider; "
                f"got report {support.report_id!r}"
            )
        support.require_supported()
        if (
            isinstance(specification, SurfaceMeshingSpec)
            and specification.background_metric is not None
            and specification.background_metric.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError(
                "NativeMeshingPlan.background_metric.coordinate_contract must match "
                "the meshing coordinate frame; got a different spatial_id"
            )
        match prepared:
            case PreparedGridHex() | PreparedMappedGridHex():
                match bound:
                    case _GridHexRequest(grid=grid) | _MappedHexRequest(grid=grid):
                        schedule_matches = (
                            prepared.grid_schedule.schedule_id == grid.schedule_id
                        )
                    case _:
                        schedule_matches = False
                if not schedule_matches:
                    raise ValueError(
                        "NativeMeshingPlan.prepared.grid_schedule must match the selected route; "
                        f"got {prepared.grid_schedule.schedule_id!r}"
                    )
            case _:
                pass
        self.source = source
        self.specification = specification
        self.options = options
        self.support = support
        self.coordinate_contract = coordinate_contract
        self.prepared = prepared
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-meshing-plan",
                "route": options.route,
                "options": options.options_id,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "specification": specification.specification_id,
                "support": support.report_id,
                "coordinate_contract": coordinate_contract.spatial_id,
                "prepared": _prepared_id(prepared),
            }
        )
        if preparation_evidence is not None:
            preparation_evidence.require_valid()
            if preparation_evidence.owner_id != self.plan_id:
                raise ValueError(
                    "NativeMeshingPlan.preparation_evidence.owner_id must bind this plan; "
                    f"got {preparation_evidence.owner_id!r}"
                )
        self.preparation_evidence = preparation_evidence

    def _registration_definition(self) -> tuple[object, ...]:
        """Complete inspected science, excluding only the later ended receipt."""
        return (
            self.source,
            self.specification,
            self.options,
            self.support,
            self.coordinate_contract,
            self.prepared,
        )

    @checked
    def with_preparation_evidence(
        self, record: NativeExecutionRecord, /
    ) -> NativeMeshingPlan:
        """Attach a validated receipt through canonical shallow root reconstruction."""
        record.require_valid()
        if record.owner_id != self.plan_id:
            raise ValueError(
                "NativeMeshingPlan.preparation_evidence.owner_id must bind this plan"
            )
        target = (jax.tree_util.GetAttrKey("preparation_evidence"),)
        children, structure = jax.tree_util.tree_flatten_with_path(
            self, is_leaf=lambda value: value is not self
        )
        if sum(path == target for path, _ in children) != 1:
            raise RuntimeError(
                "NativeMeshingPlan.preparation_evidence must own one root slot"
            )
        retained = jax.tree_util.tree_unflatten(
            structure,
            [record if path == target else value for path, value in children],
        )
        _rebind_issued_native_plan(self, retained)
        return retained

    @checked
    def execute(
        self,
        state: schema_owner.DesignState | None = None,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        coordinate_operands: CoordinateOperandCollector | None = None,
    ) -> CellMeshingResult:
        """Execute through a native session using the default execution policy."""
        with NativeMeshingProvider(self.options).open_session() as session:
            return session.execute(
                self,
                state,
                record_phase=record_phase,
                coordinate_operands=coordinate_operands,
            )

    @checked
    def _execute_with_root_work_admission(
        self,
        maximum_work_units: int,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        coordinate_operands: CoordinateOperandCollector | None = None,
    ) -> CellMeshingResult:
        """Replay this issued plan with only its mechanical root work gate lowered."""
        if (
            isinstance(maximum_work_units, bool)
            or not 0 < maximum_work_units <= self.specification.limits.maximum_work_units
            or _root_work_admission.get() is not None
        ):
            raise ValueError("Root work admission must be positive, non-reentrant and no larger than the original allowance.")
        token = _root_work_admission.set((self, maximum_work_units))
        try:
            return self.execute(
                record_phase=record_phase, coordinate_operands=coordinate_operands
            )
        finally:
            _root_work_admission.reset(token)

    @checked
    def _execute_in_session(
        self,
        state: schema_owner.DesignState | None = None,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        coordinate_operands: CoordinateOperandCollector | None = None,
    ) -> CellMeshingResult:
        """Execute and publish under the request's cumulative native allowance."""
        preparation = self.preparation_evidence
        if preparation is None:
            work, queries, seconds = _prepared_execution_charge(self.prepared)
        else:
            totals = jax.device_get(
                (
                    preparation.total_work_units,
                    preparation.total_geometry_queries,
                    preparation.total_elapsed_seconds,
                )
            )
            work = int(np.asarray(totals[0]))
            queries = int(np.asarray(totals[1]))
            seconds = float(np.asarray(totals[2]))
        admission = _root_work_admission.get()
        root_work = admission[1] if admission is not None and admission[0] is self else None
        with native_volume_execution_budget(
            self.specification.limits, root_maximum_work_units=root_work
        ) as scope:
            if scope.parent is None:
                scope.observe_managed_admission()
            with scope.host_workspace() as storage:
                storage.retain_owner(self)
                ledger = native_preparation_ledger(scope)
                if preparation is None:
                    scope.import_preparation(
                        work=work, geometry_queries=queries, elapsed_seconds=seconds
                    )
                    overlap = NativePreparationOverlap(
                        work=work,
                        geometry_queries=queries,
                        external_work=work,
                        external_geometry_queries=queries,
                        imported_elapsed_seconds=seconds,
                    )
                else:
                    ledger.import_receipt(scope, preparation)
                    if isinstance(self.prepared, PreparedLayerCore):
                        ledger.remember_live(scope, self.prepared)
                    overlap = ledger.overlap(scope, preparation)
                if coordinate_operands is not None:
                    coordinate_operands.begin(scope, self.plan_id, overlap)
                started = scope.operation_started
                result = self._execute_bound(
                    state,
                    record_phase=record_phase,
                    scope=scope,
                    workspace=storage,
                    coordinate_operands=coordinate_operands,
                )
                if scope.parent is not None:
                    stage_native_source_publication(scope, result)
                issue = (
                    stage_native_completed_source_issue(
                        scope,
                        result,
                        owner_id=(
                            preparation.owner_id
                            if preparation is not None
                            else result.execution_evidence.owner_id
                            if result.execution_evidence is not None
                            else result.result_id
                        ),
                        overlap=overlap,
                    )
                    if scope.parent is None
                    else None
                )
        evidence = scope.evidence
        if evidence is None:
            raise RuntimeError(
                "NativeMeshingPlan.execution.evidence must contain ended scope evidence; got None"
            )
        bound = bind_native_execution_result(
            result,
            overlap.normalize(evidence),
            self.specification.limits,
            started,
            source_work_units=work,
            source_geometry_queries=queries,
            preparation_seconds=seconds,
            preparation_evidence=preparation,
        )
        if issue is not None:
            issue.complete(bound)
        if coordinate_operands is not None:
            if bound.execution_evidence is None:
                raise RuntimeError(
                    "NativeMeshingPlan must bind its operand earning receipt."
                )
            coordinate_operands.complete(bound.execution_evidence)
        parent = scope.parent
        if parent is not None:
            if bound.execution_evidence is None:
                raise RuntimeError(
                    "NativeMeshingPlan must retain its ended execution receipt."
                )
            native_preparation_ledger(parent).remember_receipt(
                parent, bound.execution_evidence
            )
            stage_native_source_publication(parent, bound)
        return bound

    # Keep the exhaustive route/request/carrier match fail-closed before invoking route owners.
    def _execute_bound(
        self,
        state: schema_owner.DesignState | None = None,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        scope: NativeExecutionScope,
        workspace: NativeHostStorageWorkspace,
        coordinate_operands: CoordinateOperandCollector | None = None,
    ) -> CellMeshingResult:
        """Execute one typed source/request/prepared binding on its selected route.

        Design state is admitted only by fixed-lattice implicit discovery. Phase
        measurements remain transient and do not enter the scientific plan ID.
        """

        provider = NativeMeshingProvider.info()
        source = self.source
        specification = self.specification
        prepared = self.prepared
        route = self.options.route
        if state is not None and not isinstance(prepared, ImplicitSurfacePlan):
            raise ValueError(
                "NativeMeshingPlan.state must use fixed-lattice implicit discovery; "
                f"got a design state for {route!r}"
            )
        bound = _bind_request(self.options, source, specification)
        context = (self.coordinate_contract, provider, self.plan_id)
        execution: _ExecutionKeywords = {
            "scope": scope,
            "workspace": workspace,
            "record_phase": record_phase,
        }
        match (route, bound, prepared):
            case (
                "planar_constrained_delaunay",
                _PlanarRequest(source=source, specification=specification),
                PreparedDistributedPlanarGeneration() as prepared,
            ):
                return execute_distributed_planar_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "planar_constrained_delaunay",
                _PlanarRequest(source=source, specification=specification),
                (PreparedPlanarDomain()) as prepared,
            ):
                return execute_planar_route(
                    source,
                    specification,
                    prepared,
                    *context,
                    coordinate_operands=coordinate_operands,
                    **execution,
                )
            case (
                "implicit_surface",
                _ImplicitRequest(source=source, specification=specification),
                (AdaptiveImplicitSurface() | ImplicitSurfacePlan()) as prepared,
            ):
                return _execute_implicit_plan(
                    source, specification, prepared, *context, state, **execution
                )
            case (
                "curve_arc_length",
                _CurveRequest(source=source, specification=specification),
                (PreparedCurveNetwork()) as prepared,
            ):
                return execute_curve_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "parametric_surface",
                _SurfaceRequest(source=source, specification=specification),
                (PreparedParametricSurface()) as prepared,
            ):
                return execute_parametric_surface_route(
                    source,
                    specification,
                    prepared,
                    *context,
                    coordinate_operands=coordinate_operands,
                    **execution,
                )
            case (
                "plc_tetrahedral",
                _PlcRequest(source=source, specification=specification),
                (PreparedPlcVolume()) as prepared,
            ):
                return execute_volume_route(
                    source,
                    specification,
                    prepared,
                    *context,
                    coordinate_operands=coordinate_operands,
                    **execution,
                )
            case (
                "cad_tetrahedral",
                _CadVolumeRequest(source=source, specification=specification),
                PreparedCadVolume() as prepared,
            ):
                return execute_cad_volume_route(
                    source,
                    specification,
                    prepared,
                    *context,
                    coordinate_operands=coordinate_operands,
                    **execution,
                )
            case (
                "periodic_delaunay",
                _PeriodicRequest(source=source, specification=specification),
                (periodic_owner.PreparedPeriodicDomain()) as prepared,
            ):
                return periodic_owner.execute_periodic_route(
                    source,
                    specification,
                    prepared,
                    *context,
                    scope=scope,
                    record_phase=record_phase,
                )
            case (
                "plc_restricted_power",
                _PolyhedralRequest(source=source, specification=specification),
                (PreparedPolyhedralVolume()) as prepared,
            ):
                return execute_polyhedral_route(
                    source,
                    specification,
                    prepared,
                    *context,
                    coordinate_operands=coordinate_operands,
                    **execution,
                )
            case (
                "layer_core",
                _LayerRequest(source=source, specification=specification),
                (PreparedLayerCore()) as prepared,
            ):
                return execute_layer_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "surface_envelope_tetrahedral",
                _EnvelopeRequest(source=source, specification=specification),
                (PreparedSurfaceEnvelopeVolume()) as prepared,
            ):
                return execute_surface_envelope_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "structured_transfinite",
                _StructuredRequest(source=source, specification=specification),
                (PreparedStructured()) as prepared,
            ):
                return execute_structured_route(
                    prepared,
                    specification,
                    *context,
                    source.source_id,
                    source.source_revision,
                    source.binding_id,
                    fidelity_source=source.fidelity_source,
                    fidelity_tolerance=source.maximum_deviation,
                    domain=source.domain,
                    block_regions=source.block_regions,
                    **execution,
                )
            case (
                "sweep",
                _SweepRequest(source=source, specification=specification),
                (PreparedSweep()) as prepared,
            ):
                return execute_sweep_route(
                    prepared,
                    specification,
                    *context,
                    source.source_id,
                    source.source_revision,
                    source.binding_id,
                    fidelity_source=source.fidelity_source,
                    fidelity_tolerance=source.maximum_deviation,
                    domain=source.domain,
                    root_correspondence=source.root_correspondence,
                    block_regions=source.block_regions,
                    **execution,
                )
            case (
                "image_material_tetrahedral",
                _CompartmentRequest(source=source, specification=specification),
                (PreparedCompartmentVolume()) as prepared,
            ):
                return execute_compartment_route(
                    source,
                    specification,
                    prepared,
                    *context,
                    options=self.options,
                    **execution,
                )
            case (
                "planar_dual_quad",
                _PlanarRequest(source=source, specification=specification),
                (PreparedDualQuad()) as prepared,
            ):
                return execute_quad_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "plc_dual_hex",
                _PlcRequest(source=source, specification=specification),
                (PreparedDualHex()) as prepared,
            ):
                return execute_hex_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "plc_hex_dominant",
                _HexDominantRequest(source=source, specification=specification),
                (PreparedHexDominant()) as prepared,
            ):
                return execute_hex_dominant_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "plc_balanced_grid_hex" | "plc_frame_grid_hex",
                _GridHexRequest(source=source, specification=specification),
                (PreparedGridHex()) as prepared,
            ):
                return execute_grid_hex_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "mapped_balanced_grid_hex" | "mapped_frame_grid_hex",
                _MappedHexRequest(source=source, specification=specification),
                (PreparedMappedGridHex()) as prepared,
            ):
                return execute_mapped_grid_hex_route(
                    source, specification, prepared, *context, **execution
                )
            case (
                "implicit_restricted_delaunay",
                _ImplicitVolumeRequest(source=source, specification=specification),
                (PreparedImplicitVolume()) as prepared,
            ):
                return execute_restricted_implicit_volume(
                    source.geometry, specification, prepared, *context, **execution
                )
            case (
                "implicit_adaptive_tetrahedral",
                _AdaptiveImplicitVolumeRequest(
                    source=source, specification=specification
                ),
                (PreparedAdaptiveImplicitVolume()) as prepared,
            ):
                return execute_adaptive_implicit_volume(
                    source.geometry, specification, prepared, *context, **execution
                )
            case _:
                raise TypeError(
                    "NativeMeshingPlan.prepared must match the bound route; "
                    f"got {type(prepared).__name__} for {route!r}"
                )


@final
class NativeMeshingProvider:
    """Phydrax-owned meshing on one explicitly selected native route."""

    @checked
    def __init__(self, options: NativeMeshingOptions, /) -> None:
        self.options = options

    @checked
    def open_session(
        self,
        policy: MeshingExecutionPolicy | None = None,
        /,
        *,
        resources: ResourceRequest | None = None,
    ) -> NativeMeshingSession:
        """Open a native session with an optional outer CPU admission ceiling."""
        # Lazy: the session imports NativeMeshingPlan for checked plan admission.
        from ._native_session import NativeMeshingSession

        execution = MeshingExecutionPolicy() if policy is None else policy
        return NativeMeshingSession(execution, resources=resources)

    @staticmethod
    def info() -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "phydrax-native-meshing",
            "current",
            "Proprietary",
            operations=(
                MeshingOperation.MESH_CURVE,
                MeshingOperation.MESH_SURFACE,
                MeshingOperation.MESH_VOLUME,
            ),
            source_kinds=(
                MeshingSourceKind.CURVE,
                MeshingSourceKind.IMPLICIT,
                MeshingSourceKind.PIECEWISE_LINEAR,
                MeshingSourceKind.SURFACE,
                MeshingSourceKind.BREP,
                MeshingSourceKind.CELL_MESH,
                MeshingSourceKind.TENSOR_GRID,
                MeshingSourceKind.IMAGE,
                MeshingSourceKind.MAPPED_REFERENCE,
            ),
            capabilities=(
                MeshingCapability.DETERMINISTIC,
                MeshingCapability.IMPLICIT_CONFORMING,
                MeshingCapability.MULTI_MATERIAL,
                MeshingCapability.PERIODIC,
                MeshingCapability.MIXED_CELLS,
                MeshingCapability.POLYHEDRAL,
                MeshingCapability.HIGH_ORDER_GEOMETRY,
                MeshingCapability.PARALLEL,
            ),
            cell_kinds=(
                "interval",
                "triangle",
                "tetrahedron",
                "polyhedron",
                "prism",
                "pyramid",
                "hexahedron",
                "quadrilateral",
            ),
            dimensions=(1, 2, 3),
            execution_modes=(MeshingExecutionMode.IN_PROCESS,),
        )

    @checked
    def inspect_source(self, source: NativeMeshingSource, /) -> MeshingSourceDescriptor:
        match source:
            case sources_owner.NativePlanarSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.PIECEWISE_LINEAR,
                    2,
                    2,
                    closed=True,
                )
            case sources_owner.NativeImplicitSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.IMPLICIT,
                    3
                    if self.options.route
                    in ("implicit_restricted_delaunay", "implicit_adaptive_tetrahedral")
                    else 2,
                    source.geometry.ambient_dimension,
                    closed=True,
                )
            case sources_owner.NativeCurveSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.CURVE,
                    1,
                    source.ambient_dimension,
                    closed=False,
                )
            case sources_owner.NativeSurfaceSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.SURFACE,
                    2,
                    3,
                    closed=False,
                )
            case NativeCadVolumeSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.BREP,
                    3,
                    3,
                    closed=True,
                )
            case sources_owner.NativePlcSource() | sources_owner.NativePolyhedralSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.PIECEWISE_LINEAR,
                    3,
                    3,
                    closed=True,
                )
            case periodic_owner.NativePeriodicSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.CELL_MESH,
                    source.ambient_dimension,
                    source.ambient_dimension,
                    closed=True,
                )
            case sources_owner.NativeLayerCoreSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.CELL_MESH,
                    3,
                    3,
                    closed=True,
                )
            case sources_owner.NativeSurfaceEnvelopeSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.PIECEWISE_LINEAR,
                    3,
                    3,
                    closed=True,
                )
            case sources_owner.NativeStructuredSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.TENSOR_GRID,
                    len(source.blocks[0].intervals),
                    source.fidelity_source.ambient_dimension,
                    closed=len(source.blocks[0].intervals) == 3,
                )
            case sources_owner.NativeSweepSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.CELL_MESH,
                    3,
                    source.fidelity_source.ambient_dimension,
                    closed=True,
                )
            case CompartmentMeshingSource() | LabelFieldVolumeBinding():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.IMAGE,
                    3,
                    3,
                    closed=True,
                )
            case sources_owner.NativeMappedHexSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.MAPPED_REFERENCE,
                    3,
                    source.domain.ambient_dimension,
                    closed=True,
                )
            case _:
                raise TypeError(
                    "NativeMeshingProvider.source must be a native source binding; "
                    f"got {type(source).__name__}"
                )

    @checked
    def validate(
        self,
        source: NativeMeshingSource,
        specification: MeshingSpecification,
        /,
    ) -> ProviderSupportReport:
        """Admit one source and request on the selected route before any work."""

        bound = _bind_request(self.options, source, specification)
        scope = specification_scope(bound.specification)
        if (scope.source_id, scope.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "NativeMeshingProvider.scope must bind the supplied source revision; "
                f"got {(scope.source_id, scope.source_revision)!r}",
                provider_code="preflight",
            )
        match (self.options.route, bound):
            case (
                "planar_constrained_delaunay",
                _PlanarRequest(source=source, specification=specification),
            ):
                unsupported = planar_support_issues(*(source, specification))
            case (
                "implicit_surface",
                _ImplicitRequest(source=source, specification=specification),
            ):
                unsupported = implicit_support_issues(*(source, specification))
            case (
                "curve_arc_length",
                _CurveRequest(source=source, specification=specification),
            ):
                unsupported = curve_support_issues(*(source, specification))
            case (
                "parametric_surface",
                _SurfaceRequest(source=source, specification=specification),
            ):
                unsupported = parametric_surface_support_issues(*(source, specification))
            case (
                "plc_tetrahedral",
                _PlcRequest(source=source, specification=specification),
            ):
                unsupported = volume_support_issues(*(source, specification))
            case (
                "cad_tetrahedral",
                _CadVolumeRequest(source=source, specification=specification),
            ):
                unsupported = cad_volume_support_issues(source, specification)
            case (
                "planar_dual_quad",
                _PlanarRequest(source=source, specification=specification),
            ):
                unsupported = dual_quad_support_issues(*(source, specification))
            case (
                "plc_dual_hex",
                _PlcRequest(source=source, specification=specification),
            ):
                unsupported = dual_hex_support_issues(*(source, specification))
            case (
                "plc_hex_dominant",
                _HexDominantRequest(source=source, specification=specification),
            ):
                unsupported = hex_dominant_support_issues(*(source, specification))
            case (
                "plc_balanced_grid_hex" | "plc_frame_grid_hex",
                _GridHexRequest(source=source, specification=specification),
            ):
                unsupported = dual_hex_support_issues(*(source, specification))
            case (
                "mapped_balanced_grid_hex" | "mapped_frame_grid_hex",
                _MappedHexRequest(source=source, specification=specification),
            ):
                unsupported = mapped_grid_support_issues(*(source, specification))
            case (
                "implicit_restricted_delaunay",
                _ImplicitVolumeRequest(source=source, specification=specification),
            ):
                (_, volume) = source, specification
                unsupported = list(implicit_volume_support_issues(volume))
            case (
                "implicit_adaptive_tetrahedral",
                _AdaptiveImplicitVolumeRequest(
                    source=source, specification=specification
                ),
            ):
                (_, volume) = source, specification
                unsupported = list(adaptive_implicit_volume_support_issues(volume))
            case (
                "periodic_delaunay",
                _PeriodicRequest(source=source, specification=specification),
            ):
                unsupported = [
                    issue.detail
                    for issue in periodic_owner.periodic_support_issues(
                        source, specification
                    )
                ]
            case (
                "plc_restricted_power",
                _PolyhedralRequest(source=source, specification=specification),
            ):
                unsupported = polyhedral_support_issues(*(source, specification))
            case (
                "layer_core",
                _LayerRequest(source=source, specification=specification),
            ):
                unsupported = layer_core_support_issues(*(source, specification))
            case (
                "surface_envelope_tetrahedral",
                _EnvelopeRequest(source=source, specification=specification),
            ):
                unsupported = envelope_volume_support_issues(*(source, specification))
            case (
                "structured_transfinite",
                _StructuredRequest(source=source, specification=specification),
            ):
                (structured_source, request) = source, specification
                unsupported = structured_support_issues(request)
                if (
                    isinstance(request, VolumeMeshingSpec)
                    and structured_source.domain is None
                ):
                    unsupported.append(
                        "a declared represented volume approximation domain"
                    )
            case ("sweep", _SweepRequest(source=source, specification=specification)):
                (sweep_source, request) = source, specification
                unsupported = structured_support_issues(request, sweep=True)
                if sweep_source.domain is None:
                    unsupported.append(
                        "a declared represented volume approximation domain"
                    )
            case (
                "image_material_tetrahedral",
                _CompartmentRequest(source=source, specification=specification),
            ):
                unsupported = [
                    issue.detail
                    for issue in compartment_support_issues(source, specification)
                ]
            case _:
                raise TypeError(
                    "NativeMeshingProvider.route must bind a supported request; got inconsistent route state"
                )
        if (
            isinstance(specification, SurfaceMeshingSpec)
            and specification.background_metric is not None
        ):
            match self.options.route:
                case "parametric_surface":
                    pass
                case _:
                    unsupported.append(
                        "physical background metric sizing on this native route"
                    )
        if isinstance(specification, SurfaceMeshingSpec) and any(
            control.scope.entity_dimension == 3
            for control in specification.region_controls
        ):
            match self.options.route:
                case "parametric_surface":
                    pass
                case _:
                    unsupported.append(
                        "authoritative source-volume incidence for surface boundary controls"
                    )
        return ProviderSupportReport(
            self.info(),
            self.inspect_source(source),
            specification,
            unsupported=tuple(unsupported),
        )

    @checked
    def plan(
        self,
        source: NativeMeshingSource,
        specification: CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        initial_partition: PreparedDistributedSurfaceGeneration
        | PreparedDistributedPlanarGeneration
        | None = None,
        source_preparation_evidence: NativeExecutionRecord | None = None,
    ) -> NativeMeshingPlan:
        """Prepare in the cumulative request scope and retain its execution evidence."""
        receipt = source_preparation_evidence
        if isinstance(initial_partition, PreparedDistributedPlanarGeneration):
            partition_receipt = initial_partition.preparation_evidence
            partition_receipt.require_valid()
            if receipt is not None:
                receipt.require_valid()
                if not np.array_equal(
                    np.asarray(receipt.receipt_id),
                    np.asarray(partition_receipt.receipt_id),
                ):
                    raise ValueError(
                        "NativeMeshingProvider.source_preparation_evidence must bind the prepared initial partition; got a different receipt"
                    )
            receipt = partition_receipt
        if receipt is None and isinstance(
            source, sources_owner.NativeSurfaceEnvelopeSource
        ):
            receipt = source.envelope.execution_evidence
        if receipt is not None:
            receipt.require_valid()
            expected_owner = (
                source.envelope.envelope_id
                if isinstance(source, sources_owner.NativeSurfaceEnvelopeSource)
                else source.source_id
            )
            if receipt.owner_id != expected_owner:
                raise ValueError(
                    "NativeMeshingProvider.source_preparation_evidence.owner_id must "
                    f"bind the source; got {receipt.owner_id!r}"
                )
            require_native_preparation_allowance(specification.limits, receipt)
        with native_volume_execution_budget(
            specification.limits, stage=MeshingStageKind.SOURCE_INSPECTION
        ) as scope:
            if scope.parent is None:
                scope.observe_managed_admission()
            with scope.host_workspace() as storage:
                history: tuple[NativeExecutionRecord, ...] = ()
                if isinstance(source, NativeCadVolumeSource):
                    completed_surface = source.surface.execution_evidence
                    ledger = native_preparation_ledger(scope)
                    if completed_surface is not None and not ledger.receipt_is_accounted(
                        scope, completed_surface
                    ):
                        ledger.record_history(scope, completed_surface)
                        history = (completed_surface,)
                if receipt is not None:
                    native_preparation_ledger(scope).import_receipt(scope, receipt)
                plan = self._plan_bound(
                    source,
                    specification,
                    coordinate_contract=coordinate_contract,
                    record_phase=record_phase,
                    initial_partition=initial_partition,
                    source_preparation_accounted=receipt is not None,
                    scope=scope,
                    workspace=storage,
                )
                imports = NativeExecutionImports(scope, preparation=receipt)
        if scope.evidence is None:
            raise RuntimeError(
                "NativeMeshingProvider.preparation.evidence must contain ended scope evidence; got None"
            )
        preparation = NativeExecutionRecord(
            imports.evidence(),
            preparation_evidence=receipt,
            history_evidence=history,
            owner_id=plan.plan_id,
            imports=imports,
        )
        retained_plan = plan.with_preparation_evidence(preparation)
        retained_preparation = retained_plan.preparation_evidence
        if retained_preparation is None:
            raise RuntimeError(
                "NativeMeshingPlan.preparation_evidence must retain ended execution evidence; got None"
            )
        parent = scope.parent
        if parent is not None:
            ledger = native_preparation_ledger(parent)
            ledger.remember_receipt(parent, retained_preparation)
            if isinstance(retained_plan.prepared, PreparedLayerCore):
                ledger.remember_live(parent, retained_plan.prepared)
        return retained_plan

    def _plan_bound(
        self,
        source: NativeMeshingSource,
        specification: MeshingSpecification,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract,
        scope: NativeExecutionScope,
        workspace: NativeHostStorageWorkspace,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        initial_partition: PreparedDistributedSurfaceGeneration
        | PreparedDistributedPlanarGeneration
        | None = None,
        source_preparation_accounted: bool = False,
    ) -> NativeMeshingPlan:
        """Prepare while borrowing the caller's cumulative live allowance."""
        workspace.retain_owner(source)
        workspace.retain_owner(specification)

        if initial_partition is not None:
            match self.options.route, initial_partition:
                case "parametric_surface", PreparedDistributedSurfaceGeneration():
                    pass
                case "planar_constrained_delaunay", PreparedDistributedPlanarGeneration():
                    pass
                case _:
                    raise ValueError(
                        "NativeMeshingProvider.initial_partition must match the source "
                        f"kind and prepared route; got {self.options.route!r}"
                    )
        with measure_phase(record_phase, "source_preparation"):
            support = self.validate(source, specification)
            support.require_supported()
            if (
                isinstance(specification, SurfaceMeshingSpec)
                and specification.background_metric is not None
                and specification.background_metric.coordinate_contract.spatial_id
                != coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "NativeMeshingProvider.background_metric.coordinate_contract must "
                    "match the meshing coordinate frame; got a different spatial_id"
                )
            options = self.options
            bound = _bind_request(options, source, specification)
            preparation = _prepare_bound_request(
                bound,
                options,
                coordinate_contract,
                scope,
                record_phase=record_phase,
                initial_partition=initial_partition,
                source_preparation_accounted=source_preparation_accounted,
            )
            plan = NativeMeshingPlan(
                source,
                preparation.specification,
                options,
                support,
                coordinate_contract,
                preparation.prepared,
            )
        if isinstance(source, NativeCadVolumeSource):
            with scope.ticket_view() as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=plan.specification.limits.maximum_work_units,
                    maximum_memory_bytes=plan.specification.limits.maximum_scratch_bytes,
                )
                _remember_issued_native_plan(plan, budget)
        return plan


_issued_native_plan_lock = threading.RLock()
_issued_native_plans: dict[
    int,
    tuple[
        weakref.ReferenceType[NativeMeshingPlan],
        str,
        tuple[weakref.ReferenceType[object], ...],
        tuple[weakref.ReferenceType[object], ...],
        int,
    ],
] = {}


def _native_plan_constructors(plan: NativeMeshingPlan) -> tuple[object, ...]:
    return (
        NativeMeshingProvider.plan,
        NativeMeshingProvider._plan_bound,
        NativeMeshingProvider.validate,
        NativeMeshingProvider.info,
        NativeMeshingPlan.__init__,
        NativeMeshingPlan.with_preparation_evidence,
        NativeMeshingPlan._registration_definition,
        _prepare_bound_request,
        _bind_request,
        *(type(value).__init__ for value in plan._registration_definition()),
    )


def _native_plan_registration_binding(
    plan: NativeMeshingPlan, budget: CoordinateEnclosureBudget
) -> str:
    with budget.temporary_scope():
        definition = plan._registration_definition()
        admit_carrier_fingerprint(definition, budget)
        payloads = []
        for value in definition:
            if not isinstance(value, StrictModule):
                raise TypeError(
                    "Native CAD plan registration requires its complete concrete science owners"
                )
            payloads.append(carrier_payload(value))
        return canonical_fingerprint(
            {
                "plan": plan.plan_id,
                "definition": tuple(payloads),
            }
        )


def _forget_issued_native_plan(
    identifier: int, reference: weakref.ReferenceType[NativeMeshingPlan]
) -> None:
    with _issued_native_plan_lock:
        entry = _issued_native_plans.get(identifier)
        if entry is not None and entry[0] is reference:
            del _issued_native_plans[identifier]


def _remember_issued_native_plan(
    plan: NativeMeshingPlan, budget: CoordinateEnclosureBudget
) -> None:
    """Issue only at the real producer's completed source-inspection boundary."""
    binding = _native_plan_registration_binding(plan, budget)
    constructors = _native_plan_constructors(plan)
    definition = plan._registration_definition()
    budget.reserve(1, 1024 + 128 * (len(constructors) + len(definition)))
    identifier = id(plan)
    reference = weakref.ref(
        plan, lambda reference: _forget_issued_native_plan(identifier, reference)
    )
    entry = (
        reference,
        binding,
        tuple(weakref.ref(value) for value in constructors),
        tuple(weakref.ref(value) for value in definition),
        os.getpid(),
    )
    with _issued_native_plan_lock:
        _issued_native_plans[identifier] = entry


def _rebind_issued_native_plan(
    previous: NativeMeshingPlan, retained: NativeMeshingPlan
) -> None:
    """Move the paid full-plan issue only across its validated receipt-only update."""
    definition = retained._registration_definition()
    with _issued_native_plan_lock:
        entry = _issued_native_plans.get(id(previous))
        if entry is None or entry[0]() is not previous or entry[4] != os.getpid():
            return
        if (
            previous.plan_id != retained.plan_id
            or len(entry[3]) != len(definition)
            or any(
                old() is not current
                for old, current in zip(entry[3], definition, strict=True)
            )
        ):
            raise ValueError(
                "Native plan receipt replacement must preserve its inspected owners"
            )
        identifier = id(retained)
        reference = weakref.ref(
            retained, lambda reference: _forget_issued_native_plan(identifier, reference)
        )
        del _issued_native_plans[id(previous)]
        _issued_native_plans[identifier] = (
            reference,
            entry[1],
            entry[2],
            entry[3],
            entry[4],
        )


def _require_issued_native_plan(
    plan: NativeMeshingPlan, budget: CoordinateEnclosureBudget
) -> None:
    """Authenticate all current prepared/source bytes before warm registration."""
    with _issued_native_plan_lock:
        entry = _issued_native_plans.get(id(plan))
    constructors = _native_plan_constructors(plan)
    definition = plan._registration_definition()
    if (
        entry is None
        or entry[0]() is not plan
        or entry[4] != os.getpid()
        or len(entry[2]) != len(constructors)
        or any(
            old() is not current
            for old, current in zip(entry[2], constructors, strict=True)
        )
        or len(entry[3]) != len(definition)
        or any(
            old() is not current
            for old, current in zip(entry[3], definition, strict=True)
        )
        or entry[1] != _native_plan_registration_binding(plan, budget)
    ):
        raise ValueError(
            "native.plan.registration must retain its actual producer-issued full inspected plan"
        )


__all__ = ["NativeMeshingPlan", "NativeMeshingProvider"]
