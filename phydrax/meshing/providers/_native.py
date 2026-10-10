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

from contextlib import nullcontext
from time import monotonic
from typing import assert_never, final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import (
    current_native_execution_budget,
    current_native_host_workspace,
    NativeExecutionBudget,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import DesignState
from ...geometry._compartments import CompartmentMeshingSource
from ...geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    ImplicitSurfacePlan,
    ImplicitSurfacePolicy,
)
from ...geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from ...typing import checked
from .._contracts import (
    CurveMeshingSpec,
    MeshingCapability,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceDescriptor,
    MeshingSourceKind,
    MeshingSpecification,
    ProviderSupportReport,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from .._distributed_generation import PreparedDistributedSurfaceGeneration
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
    NativeExecutionRecord,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from .._result import CellMeshingResult
from .._scope import MeshingScope
from .._surface_envelope import (
    envelope_volume_support_issues,
    execute_surface_envelope_route,
    PreparedSurfaceEnvelopeVolume,
)
from .._trace import MeshingStageKind
from .._volume_generation import (
    _import_native_preparation,
    _native_volume_operation_started,
    _native_volume_source_is_active,
    _remember_native_live_preparation,
    _remember_native_preparation,
    _require_native_preparation_allowance,
    native_volume_checkpoint,
    native_volume_execution_budget,
)
from ._implicit import (
    execute_adaptive_implicit_route,
    execute_implicit_route,
    implicit_support_issues,
    prepare_adaptive_implicit_route,
    prepare_implicit_route,
)
from ._native_compartment import (
    compartment_support_issues,
    execute_compartment_route,
    PreparedCompartmentVolume,
)
from ._native_curve import (
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
from ._native_periodic import (
    execute_periodic_route,
    NativePeriodicSource,
    periodic_support_issues,
    PreparedPeriodicDomain,
)
from ._native_planar import (
    execute_planar_route,
    planar_support_issues,
    PreparedPlanarDomain,
)
from ._native_polyhedral import (
    execute_polyhedral_route,
    polyhedral_support_issues,
    PreparedPolyhedralVolume,
)
from ._native_publication import bind_native_execution_result
from ._native_sources import (
    NativeCurveSource,
    NativeImplicitSource,
    NativeLayerCoreSource,
    NativeMappedHexSource,
    NativeMeshingSource,
    NativePlanarSource,
    NativePlcSource,
    NativePolyhedralSource,
    NativeStructuredSource,
    NativeSurfaceEnvelopeSource,
    NativeSurfaceSource,
    NativeSweepSource,
)
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


type _PreparedRoute = (
    AdaptiveImplicitSurface
    | ImplicitSurfacePlan
    | PreparedCurveNetwork
    | PreparedPlanarDomain
    | PreparedParametricSurface
    | PreparedPlcVolume
    | PreparedPeriodicDomain
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
            | PreparedPeriodicDomain()
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


def _scope(specification: MeshingSpecification, /) -> MeshingScope:
    match specification:
        case CurveMeshingSpec() | SurfaceMeshingSpec():
            return specification.scope
        case VolumeMeshingSpec():
            return specification.boundary_scope
        case _:
            raise TypeError(
                "Native meshing admits CurveMeshingSpec, SurfaceMeshingSpec or "
                "VolumeMeshingSpec requests."
            )


def _planar_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativePlanarSource, SurfaceMeshingSpec]:
    if not isinstance(source, NativePlanarSource):
        raise TypeError("The planar route requires a NativePlanarSource.")
    if not isinstance(specification, SurfaceMeshingSpec):
        raise TypeError("The planar route requires a SurfaceMeshingSpec request.")
    return source, specification


def _implicit_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeImplicitSource, SurfaceMeshingSpec]:
    if not isinstance(source, NativeImplicitSource):
        raise TypeError("The implicit route requires a NativeImplicitSource.")
    if not isinstance(specification, SurfaceMeshingSpec):
        raise TypeError("The implicit route requires a SurfaceMeshingSpec request.")
    return source, specification


def _curve_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeCurveSource, CurveMeshingSpec]:
    if not isinstance(source, NativeCurveSource):
        raise TypeError("The curve route requires a NativeCurveSource.")
    if not isinstance(specification, CurveMeshingSpec):
        raise TypeError("The curve route requires a CurveMeshingSpec request.")
    return source, specification


def _surface_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeSurfaceSource, SurfaceMeshingSpec]:
    if not isinstance(source, NativeSurfaceSource):
        raise TypeError("The parametric surface route requires a NativeSurfaceSource.")
    if not isinstance(specification, SurfaceMeshingSpec):
        raise TypeError("The parametric surface route requires a SurfaceMeshingSpec.")
    return source, specification


def _plc_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativePlcSource, VolumeMeshingSpec]:
    if not isinstance(source, NativePlcSource):
        raise TypeError("The PLC tetrahedral route requires a NativePlcSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("The PLC tetrahedral route requires a VolumeMeshingSpec request.")
    return source, specification


def _polyhedral_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativePlcSource | NativePolyhedralSource, VolumeMeshingSpec]:
    if not isinstance(source, (NativePlcSource, NativePolyhedralSource)):
        raise TypeError("The restricted power route requires a PLC or power-site source.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("The restricted power route requires VolumeMeshingSpec.")
    return source, specification


def _periodic_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativePeriodicSource, SurfaceMeshingSpec | VolumeMeshingSpec]:
    if not isinstance(source, NativePeriodicSource):
        raise TypeError("The periodic route requires NativePeriodicSource.")
    if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
        raise TypeError("The periodic route requires a surface or volume specification.")
    return source, specification


def _layer_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeLayerCoreSource, VolumeMeshingSpec]:
    if not isinstance(source, NativeLayerCoreSource):
        raise TypeError("The layer core route requires NativeLayerCoreSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("The layer core route requires VolumeMeshingSpec.")
    return source, specification


def _envelope_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeSurfaceEnvelopeSource, VolumeMeshingSpec]:
    if not isinstance(source, NativeSurfaceEnvelopeSource):
        raise TypeError(
            "The surface envelope route requires NativeSurfaceEnvelopeSource."
        )
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("The surface envelope route requires VolumeMeshingSpec.")
    return source, specification


def _structured_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeStructuredSource, SurfaceMeshingSpec | VolumeMeshingSpec]:
    if not isinstance(source, NativeStructuredSource):
        raise TypeError("The structured route requires NativeStructuredSource.")
    if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
        raise TypeError(
            "The structured route requires a surface or volume specification."
        )
    return source, specification


def _sweep_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeSweepSource, VolumeMeshingSpec]:
    if not isinstance(source, NativeSweepSource):
        raise TypeError("The sweep route requires NativeSweepSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("The sweep route requires VolumeMeshingSpec.")
    return source, specification


def _compartment_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[CompartmentMeshingSource | LabelFieldVolumeBinding, VolumeMeshingSpec]:
    if not isinstance(source, (CompartmentMeshingSource, LabelFieldVolumeBinding)):
        raise TypeError(
            "The image material route requires an authoritative image source."
        )
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("The image material route requires VolumeMeshingSpec.")
    return source, specification


def _mapped_hex_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeMappedHexSource, VolumeMeshingSpec]:
    if not isinstance(source, NativeMappedHexSource):
        raise TypeError("The mapped hex-grid route requires NativeMappedHexSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("The mapped hex-grid route requires VolumeMeshingSpec.")
    return source, specification


def _implicit_volume_request(
    source: NativeMeshingSource, specification: MeshingSpecification, /
) -> tuple[NativeImplicitSource, VolumeMeshingSpec]:
    if not isinstance(source, NativeImplicitSource):
        raise TypeError(
            "The restricted implicit volume route requires NativeImplicitSource."
        )
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError(
            "The restricted implicit volume route requires VolumeMeshingSpec."
        )
    if source.geometry.ambient_dimension != 3 or len(source.grid.structured_axes) != 3:
        raise ValueError(
            "Restricted implicit volumes require a three-dimensional physical enclosure."
        )
    return source, specification


def _execute_implicit_plan(
    source: NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    prepared: AdaptiveImplicitSurface | ImplicitSurfacePlan,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    state: DesignState | None,
    /,
    *,
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
                record_phase=record_phase,
            )
        case _:
            assert_never(prepared)


def _require_route_types(
    options: NativeMeshingOptions,
    source: NativeMeshingSource,
    specification: MeshingSpecification,
    /,
) -> None:
    """Refuse a source or request kind that the selected route does not own."""

    match options.route:
        case "planar_constrained_delaunay" | "planar_dual_quad":
            _planar_request(source, specification)
        case "implicit_surface":
            _implicit_request(source, specification)
        case "implicit_restricted_delaunay" | "implicit_adaptive_tetrahedral":
            _implicit_volume_request(source, specification)
        case "curve_arc_length":
            _curve_request(source, specification)
        case "parametric_surface":
            _surface_request(source, specification)
        case (
            "plc_tetrahedral"
            | "plc_dual_hex"
            | "plc_hex_dominant"
            | "plc_balanced_grid_hex"
            | "plc_frame_grid_hex"
        ):
            _plc_request(source, specification)
        case "plc_restricted_power":
            _polyhedral_request(source, specification)
        case "periodic_delaunay":
            _periodic_request(source, specification)
        case "layer_core":
            _layer_request(source, specification)
        case "surface_envelope_tetrahedral":
            _envelope_request(source, specification)
        case "structured_transfinite":
            _structured_request(source, specification)
        case "sweep":
            _sweep_request(source, specification)
        case "image_material_tetrahedral":
            _compartment_request(source, specification)
        case "mapped_balanced_grid_hex" | "mapped_frame_grid_hex":
            _mapped_hex_request(source, specification)
        case _:
            assert_never(options.route)


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
        case AdaptiveImplicitSurface():
            evidence = prepared.evidence
            return (
                evidence.root_solves,
                (
                    evidence.box_evaluations
                    + evidence.point_evaluations
                    + evidence.extraction_evaluations
                ),
                0.0,
            )
        case _:
            return 0, 0, 0.0


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
        _require_route_types(options, source, specification)
        if (
            support.specification_id != specification.specification_id
            or support.provider_id != NativeMeshingProvider.info().provider_id
        ):
            raise ValueError("The support report does not admit this request.")
        support.require_supported()
        if (
            isinstance(specification, SurfaceMeshingSpec)
            and specification.background_metric is not None
            and specification.background_metric.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Background metric and meshing source must use one physical coordinate frame."
            )
        if isinstance(prepared, (PreparedGridHex, PreparedMappedGridHex)):
            if (
                options.grid_schedule is None
                or prepared.grid_schedule.schedule_id != options.grid_schedule.schedule_id
            ):
                raise ValueError(
                    "The prepared grid schedule must match the selected route."
                )
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
            if type(preparation_evidence) is not NativeExecutionRecord:
                raise TypeError(
                    "Preparation evidence must be its exact owning native record."
                )
            preparation_evidence.require_valid()
            if preparation_evidence.owner_id != self.plan_id:
                raise ValueError(
                    "Preparation evidence binds another native meshing plan."
                )
        self.preparation_evidence = preparation_evidence

    def execute(
        self,
        state: DesignState | None = None,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
    ) -> CellMeshingResult:
        """Execute and publish under the original complete native allowance."""
        active = current_native_execution_budget()
        if active is not None:
            native_volume_checkpoint(
                self.specification.limits,
                MeshingStageKind.VOLUME_FILL,
                execution_budget=active,
            )
            workspace = current_native_host_workspace()
            with (
                active.host_workspace() if workspace is None else nullcontext(workspace)
            ) as storage:
                storage.retain_owner(self)
                preparation = self.preparation_evidence
                if preparation is not None:
                    _import_native_preparation(active, storage, preparation)
                    return self._execute_with_receipt(
                        state,
                        record_phase=record_phase,
                        borrow_active=False,
                    )
                return self._execute_bound(
                    state,
                    record_phase=record_phase,
                    execution_budget=active,
                    operation_started=_native_volume_operation_started(),
                )
        return self._execute_with_receipt(state, record_phase=record_phase)

    def _execute_with_receipt(
        self,
        state: DesignState | None,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None,
        borrow_active: bool = True,
    ) -> CellMeshingResult:
        preparation = self.preparation_evidence
        if preparation is None:
            work, queries, seconds = _prepared_execution_charge(self.prepared)
        else:
            work = int(np.asarray(preparation.total_work_units))
            queries = int(np.asarray(preparation.total_geometry_queries))
            seconds = float(np.asarray(preparation.total_elapsed_seconds))
        started = monotonic() - seconds
        parent_started = _native_volume_operation_started()
        if parent_started is not None:
            started = min(started, parent_started)
        with native_volume_execution_budget(
            self.specification.limits,
            source_work_units=work,
            source_geometry_queries=queries,
            operation_started=started,
            borrow_active=borrow_active,
        ) as budget:
            workspace = current_native_host_workspace()
            if workspace is None:
                raise RuntimeError("Native execution lost its owning host workspace.")
            workspace.retain_owner(self)
            if preparation is not None:
                phase: NativeExecutionRecord | None = preparation
                while phase is not None:
                    _remember_native_preparation(
                        budget, workspace, phase, import_work=False
                    )
                    phase = phase.preparation_evidence
                if isinstance(self.prepared, PreparedLayerCore):
                    _remember_native_live_preparation(budget, workspace, self.prepared)
            result = self._execute_bound(
                state,
                record_phase=record_phase,
                execution_budget=budget,
                operation_started=started,
            )
        evidence = budget.evidence
        if evidence is None:
            raise RuntimeError(
                "Native execution completed without original-scope evidence."
            )
        return bind_native_execution_result(
            result,
            evidence,
            self.specification.limits,
            started,
            source_work_units=work,
            source_geometry_queries=queries,
            preparation_seconds=seconds,
            preparation_evidence=preparation,
        )

    def _execute_bound(
        self,
        state: DesignState | None = None,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        execution_budget: NativeExecutionBudget,
        operation_started: float | None,
    ) -> CellMeshingResult:
        """Run the prepared route and publish its audited, compliant result.

        ``state`` selects the design state of the implicit route; the planar and
        curve routes mesh their fixed source geometry and refuse a state.

        This exhaustive boundary deliberately keeps each source/request/prepared
        type guard visible. Numerical construction and acceptance stages live in
        the route owners; an untyped registry would conceal invalid combinations.

        ``record_phase`` receives only actual execution measurements. The channel
        is not retained in this plan or its result and never enters fingerprints.
        """

        if record_phase is not None and not callable(record_phase):
            raise TypeError("record_phase must be a callable phase recorder or None.")
        provider = NativeMeshingProvider.info()
        source = self.source
        specification = self.specification
        prepared = self.prepared
        route = self.options.route
        if state is not None and not isinstance(prepared, ImplicitSurfacePlan):
            raise ValueError(
                f"The {route} route meshes fixed source geometry; only the "
                "fixed-lattice implicit discovery realizes design states."
            )
        match route:
            case "planar_constrained_delaunay":
                if not (
                    isinstance(source, NativePlanarSource)
                    and isinstance(specification, SurfaceMeshingSpec)
                    and isinstance(prepared, PreparedPlanarDomain)
                ):
                    raise TypeError("Planar plan state is inconsistent with its route.")
                return execute_planar_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "implicit_surface":
                if not (
                    isinstance(source, NativeImplicitSource)
                    and isinstance(specification, SurfaceMeshingSpec)
                    and isinstance(
                        prepared, (AdaptiveImplicitSurface, ImplicitSurfacePlan)
                    )
                ):
                    raise TypeError("Implicit plan state is inconsistent with its route.")
                return _execute_implicit_plan(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    state,
                    record_phase=record_phase,
                )
            case "curve_arc_length":
                if not (
                    isinstance(source, NativeCurveSource)
                    and isinstance(specification, CurveMeshingSpec)
                    and isinstance(prepared, PreparedCurveNetwork)
                ):
                    raise TypeError("Curve plan state is inconsistent with its route.")
                return execute_curve_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "parametric_surface":
                if not (
                    isinstance(source, NativeSurfaceSource)
                    and isinstance(specification, SurfaceMeshingSpec)
                    and isinstance(prepared, PreparedParametricSurface)
                ):
                    raise TypeError("Surface plan state is inconsistent with its route.")
                return execute_parametric_surface_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "plc_tetrahedral":
                if not (
                    isinstance(source, NativePlcSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedPlcVolume)
                ):
                    raise TypeError("Volume plan state is inconsistent with its route.")
                return execute_volume_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "periodic_delaunay":
                if not (
                    isinstance(source, NativePeriodicSource)
                    and isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
                    and isinstance(prepared, PreparedPeriodicDomain)
                ):
                    raise TypeError("Periodic plan state is inconsistent with its route.")
                return execute_periodic_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "plc_restricted_power":
                if not (
                    isinstance(source, (NativePlcSource, NativePolyhedralSource))
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedPolyhedralVolume)
                ):
                    raise TypeError(
                        "Polyhedral plan state is inconsistent with its route."
                    )
                return execute_polyhedral_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "layer_core":
                if not (
                    isinstance(source, NativeLayerCoreSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedLayerCore)
                ):
                    raise TypeError(
                        "Layer core plan state is inconsistent with its route."
                    )
                return execute_layer_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "surface_envelope_tetrahedral":
                if not (
                    isinstance(source, NativeSurfaceEnvelopeSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedSurfaceEnvelopeVolume)
                ):
                    raise TypeError("Envelope plan state is inconsistent with its route.")
                return execute_surface_envelope_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "structured_transfinite":
                if not (
                    isinstance(source, NativeStructuredSource)
                    and isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
                    and isinstance(prepared, PreparedStructured)
                ):
                    raise TypeError(
                        "Structured plan state is inconsistent with its route."
                    )
                return execute_structured_route(
                    prepared,
                    specification,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    source.source_id,
                    source.source_revision,
                    source.binding_id,
                    fidelity_source=source.fidelity_source,
                    fidelity_tolerance=source.maximum_deviation,
                    domain=source.domain,
                    block_regions=source.block_regions,
                    record_phase=record_phase,
                )
            case "sweep":
                if not (
                    isinstance(source, NativeSweepSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedSweep)
                ):
                    raise TypeError("Sweep plan state is inconsistent with its route.")
                return execute_sweep_route(
                    prepared,
                    specification,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    source.source_id,
                    source.source_revision,
                    source.binding_id,
                    fidelity_source=source.fidelity_source,
                    fidelity_tolerance=source.maximum_deviation,
                    domain=source.domain,
                    root_correspondence=source.root_correspondence,
                    block_regions=source.block_regions,
                    record_phase=record_phase,
                )
            case "image_material_tetrahedral":
                if not (
                    isinstance(
                        source, (CompartmentMeshingSource, LabelFieldVolumeBinding)
                    )
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedCompartmentVolume)
                ):
                    raise TypeError("Material plan state is inconsistent with its route.")
                return execute_compartment_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                    execution_budget=execution_budget,
                    operation_started=operation_started,
                )
            case "planar_dual_quad":
                if not (
                    isinstance(source, NativePlanarSource)
                    and isinstance(specification, SurfaceMeshingSpec)
                    and isinstance(prepared, PreparedDualQuad)
                ):
                    raise TypeError("Quad plan state is inconsistent with its route.")
                return execute_quad_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "plc_dual_hex":
                if not (
                    isinstance(source, NativePlcSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedDualHex)
                ):
                    raise TypeError("Hex plan state is inconsistent with its route.")
                return execute_hex_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "plc_hex_dominant":
                if not (
                    isinstance(source, NativePlcSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedHexDominant)
                ):
                    raise TypeError("Hybrid plan state is inconsistent with its route.")
                return execute_hex_dominant_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "plc_balanced_grid_hex" | "plc_frame_grid_hex":
                if not (
                    isinstance(source, NativePlcSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedGridHex)
                ):
                    raise TypeError("Grid plan state is inconsistent with its route.")
                return execute_grid_hex_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "mapped_balanced_grid_hex" | "mapped_frame_grid_hex":
                if not (
                    isinstance(source, NativeMappedHexSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedMappedGridHex)
                ):
                    raise TypeError(
                        "Mapped grid plan state is inconsistent with its route."
                    )
                return execute_mapped_grid_hex_route(
                    source,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                )
            case "implicit_restricted_delaunay":
                if not (
                    isinstance(source, NativeImplicitSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedImplicitVolume)
                ):
                    raise TypeError(
                        "Restricted implicit plan state is inconsistent with its route."
                    )
                return execute_restricted_implicit_volume(
                    source.geometry,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                    execution_budget=execution_budget,
                    operation_started=operation_started,
                )
            case "implicit_adaptive_tetrahedral":
                if not (
                    isinstance(source, NativeImplicitSource)
                    and isinstance(specification, VolumeMeshingSpec)
                    and isinstance(prepared, PreparedAdaptiveImplicitVolume)
                ):
                    raise TypeError(
                        "Adaptive implicit plan state is inconsistent with its route."
                    )
                return execute_adaptive_implicit_volume(
                    source.geometry,
                    specification,
                    prepared,
                    self.coordinate_contract,
                    provider,
                    self.plan_id,
                    record_phase=record_phase,
                    execution_budget=execution_budget,
                    operation_started=operation_started,
                )
            case _:
                assert_never(route)


@final
class NativeMeshingProvider:
    """Phydrax-owned meshing on one explicitly selected native route."""

    @checked
    def __init__(self, options: NativeMeshingOptions, /) -> None:
        self.options = options

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

    def inspect_source(self, source: NativeMeshingSource, /) -> MeshingSourceDescriptor:
        match source:
            case NativePlanarSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.PIECEWISE_LINEAR,
                    2,
                    2,
                    closed=True,
                )
            case NativeImplicitSource():
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
            case NativeCurveSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.CURVE,
                    1,
                    source.ambient_dimension,
                    closed=False,
                )
            case NativeSurfaceSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.SURFACE,
                    2,
                    3,
                    closed=False,
                )
            case NativePlcSource() | NativePolyhedralSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.PIECEWISE_LINEAR,
                    3,
                    3,
                    closed=True,
                )
            case NativePeriodicSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.CELL_MESH,
                    source.ambient_dimension,
                    source.ambient_dimension,
                    closed=True,
                )
            case NativeLayerCoreSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.CELL_MESH,
                    3,
                    3,
                    closed=True,
                )
            case NativeSurfaceEnvelopeSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.PIECEWISE_LINEAR,
                    3,
                    3,
                    closed=True,
                )
            case NativeStructuredSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.TENSOR_GRID,
                    len(source.blocks[0].intervals),
                    source.fidelity_source.ambient_dimension,
                    closed=len(source.blocks[0].intervals) == 3,
                )
            case NativeSweepSource():
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
            case NativeMappedHexSource():
                return MeshingSourceDescriptor(
                    source.source_id,
                    source.source_revision,
                    MeshingSourceKind.MAPPED_REFERENCE,
                    3,
                    source.domain.ambient_dimension,
                    closed=True,
                )
            case _:
                raise TypeError("source must be a native meshing source binding.")

    def validate(
        self,
        source: NativeMeshingSource,
        specification: MeshingSpecification,
        /,
    ) -> ProviderSupportReport:
        """Admit one source and request on the selected route before any work."""

        _require_route_types(self.options, source, specification)
        scope = _scope(specification)
        if (scope.source_id, scope.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "The meshing scope does not bind the supplied source revision.",
                provider_code="preflight",
            )
        match self.options.route:
            case "planar_constrained_delaunay":
                unsupported = planar_support_issues(
                    *_planar_request(source, specification)
                )
            case "implicit_surface":
                unsupported = implicit_support_issues(
                    *_implicit_request(source, specification)
                )
            case "curve_arc_length":
                unsupported = curve_support_issues(*_curve_request(source, specification))
            case "parametric_surface":
                unsupported = parametric_surface_support_issues(
                    *_surface_request(source, specification)
                )
            case "plc_tetrahedral":
                unsupported = volume_support_issues(*_plc_request(source, specification))
            case "planar_dual_quad":
                unsupported = dual_quad_support_issues(
                    *_planar_request(source, specification)
                )
            case "plc_dual_hex":
                unsupported = dual_hex_support_issues(
                    *_plc_request(source, specification)
                )
            case "plc_hex_dominant":
                unsupported = hex_dominant_support_issues(
                    *_plc_request(source, specification)
                )
            case "plc_balanced_grid_hex" | "plc_frame_grid_hex":
                unsupported = dual_hex_support_issues(
                    *_plc_request(source, specification)
                )
            case "mapped_balanced_grid_hex" | "mapped_frame_grid_hex":
                unsupported = mapped_grid_support_issues(
                    *_mapped_hex_request(source, specification)
                )
            case "implicit_restricted_delaunay":
                _, volume = _implicit_volume_request(source, specification)
                unsupported = list(implicit_volume_support_issues(volume))
            case "implicit_adaptive_tetrahedral":
                _, volume = _implicit_volume_request(source, specification)
                unsupported = list(adaptive_implicit_volume_support_issues(volume))
            case "periodic_delaunay":
                unsupported = periodic_support_issues(
                    *_periodic_request(source, specification)
                )
            case "plc_restricted_power":
                unsupported = polyhedral_support_issues(
                    *_polyhedral_request(source, specification)
                )
            case "layer_core":
                unsupported = layer_core_support_issues(
                    *_layer_request(source, specification)
                )
            case "surface_envelope_tetrahedral":
                unsupported = envelope_volume_support_issues(
                    *_envelope_request(source, specification)
                )
            case "structured_transfinite":
                structured_source, request = _structured_request(source, specification)
                unsupported = structured_support_issues(request)
                if (
                    isinstance(request, VolumeMeshingSpec)
                    and structured_source.domain is None
                ):
                    unsupported.append(
                        "a declared represented volume approximation domain"
                    )
            case "sweep":
                sweep_source, request = _sweep_request(source, specification)
                unsupported = structured_support_issues(request, sweep=True)
                if sweep_source.domain is None:
                    unsupported.append(
                        "a declared represented volume approximation domain"
                    )
            case "image_material_tetrahedral":
                unsupported = compartment_support_issues(
                    *_compartment_request(source, specification)
                )
            case _:
                assert_never(self.options.route)
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
        specification: MeshingSpecification,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        initial_partition: PreparedDistributedSurfaceGeneration | None = None,
        source_preparation_evidence: NativeExecutionRecord | None = None,
    ) -> NativeMeshingPlan:
        """Prepare under the original root and retain an ended standalone receipt."""
        if not isinstance(
            specification, (CurveMeshingSpec, SurfaceMeshingSpec, VolumeMeshingSpec)
        ):
            raise TypeError(
                "Native meshing admits CurveMeshingSpec, SurfaceMeshingSpec or "
                "VolumeMeshingSpec requests."
            )
        receipt = source_preparation_evidence
        if receipt is None and isinstance(source, NativeSurfaceEnvelopeSource):
            receipt = source.envelope.execution_evidence
        if receipt is not None:
            receipt.require_valid()
            expected_owner = (
                source.envelope.envelope_id
                if isinstance(source, NativeSurfaceEnvelopeSource)
                else source.source_id
            )
            if receipt.owner_id != expected_owner:
                raise ValueError(
                    "Source preparation receipt binds another original source owner."
                )
            _require_native_preparation_allowance(specification.limits, receipt)
        active = current_native_execution_budget()
        workspace = current_native_host_workspace()
        scope = (
            active.host_workspace()
            if active is not None and workspace is None
            else nullcontext(workspace)
        )
        with scope as storage:
            if active is not None and receipt is not None:
                if storage is None:
                    raise RuntimeError(
                        "Source receipt import lost its owning host workspace."
                    )
                _import_native_preparation(active, storage, receipt)
            work = 0 if receipt is None else int(np.asarray(receipt.total_work_units))
            queries = (
                0 if receipt is None else int(np.asarray(receipt.total_geometry_queries))
            )
            seconds = (
                0.0
                if receipt is None
                else float(np.asarray(receipt.total_elapsed_seconds))
            )
            started = monotonic() - seconds
            parent_started = _native_volume_operation_started()
            if parent_started is not None:
                started = min(started, parent_started)
            with native_volume_execution_budget(
                specification.limits,
                source_work_units=work,
                source_geometry_queries=queries,
                operation_started=started,
                stage=MeshingStageKind.SOURCE_INSPECTION,
                borrow_active=False,
            ) as budget:
                plan = self._plan_bound(
                    source,
                    specification,
                    coordinate_contract=coordinate_contract,
                    record_phase=record_phase,
                    initial_partition=initial_partition,
                    source_preparation_accounted=receipt is not None,
                )
            if budget.evidence is None:
                raise RuntimeError(
                    "Native preparation completed without ended scope evidence."
                )
            preparation = NativeExecutionRecord(
                budget.evidence,
                preparation_evidence=receipt,
                owner_id=plan.plan_id if receipt is None else receipt.owner_id,
            )
            retained_plan = eqx.tree_at(
                lambda value: value.preparation_evidence,
                plan,
                preparation,
                is_leaf=lambda value: value is None,
            )
            if active is not None:
                if storage is None or retained_plan.preparation_evidence is None:
                    raise RuntimeError(
                        "Native preparation lost its parent receipt owner."
                    )
                _remember_native_preparation(
                    active,
                    storage,
                    retained_plan.preparation_evidence,
                    import_work=False,
                )
                if isinstance(retained_plan.prepared, PreparedLayerCore):
                    _remember_native_live_preparation(
                        active, storage, retained_plan.prepared
                    )
            return retained_plan

    def _plan_bound(
        self,
        source: NativeMeshingSource,
        specification: MeshingSpecification,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        initial_partition: PreparedDistributedSurfaceGeneration | None = None,
        source_preparation_accounted: bool = False,
    ) -> NativeMeshingPlan:
        """Prepare while borrowing the caller's live original allowance."""
        workspace = current_native_host_workspace()
        if workspace is None:
            raise RuntimeError("Native preparation requires its owning host workspace.")
        workspace.retain_owner(source)
        workspace.retain_owner(specification)

        if initial_partition is not None:
            match self.options.route:
                case "parametric_surface":
                    pass
                case _:
                    raise ValueError(
                        "Initial patch partitioning belongs to the parametric surface route."
                    )
        preparation_started = phase_started(record_phase)
        support = self.validate(source, specification)
        support.require_supported()
        if (
            isinstance(specification, SurfaceMeshingSpec)
            and specification.background_metric is not None
            and specification.background_metric.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Background metric and meshing source must use one physical coordinate frame."
            )
        options = self.options
        match options.route:
            case "planar_constrained_delaunay":
                planar_source, surface = _planar_request(source, specification)
                prepared = PreparedPlanarDomain(planar_source, surface)
                request = surface
            case "implicit_surface":
                implicit_source, surface = _implicit_request(source, specification)
                policy = options.implicit_policy
                if type(policy) is AdaptiveImplicitSurfacePolicy:
                    prepared = prepare_adaptive_implicit_route(
                        implicit_source, surface, policy
                    )
                elif type(policy) is ImplicitSurfacePolicy:
                    prepared = prepare_implicit_route(implicit_source, surface, policy)
                else:
                    raise ValueError("The implicit route requires its resolved policy.")
                request = surface
            case "curve_arc_length":
                if options.curve_schedule is None:
                    raise ValueError("The curve route requires its resolved schedule.")
                curve_source, curves = _curve_request(source, specification)
                prepared = PreparedCurveNetwork(
                    curve_source, curves, options.curve_schedule
                )
                request = curves
            case "parametric_surface":
                if options.surface_schedule is None:
                    raise ValueError("The surface route requires its resolved schedule.")
                surface_source, surface = _surface_request(source, specification)
                prepared = PreparedParametricSurface(
                    surface_source,
                    surface,
                    options.surface_schedule,
                    initial_partition=initial_partition,
                )
                request = surface
            case "plc_tetrahedral":
                if options.volume_schedule is None:
                    raise ValueError("The PLC tetrahedral route requires its schedule.")
                plc_source, volume = _plc_request(source, specification)
                prepared = PreparedPlcVolume(plc_source, volume, options.volume_schedule)
                request = volume
            case "periodic_delaunay":
                periodic_source, request = _periodic_request(source, specification)
                prepared = PreparedPeriodicDomain(periodic_source, request)
            case "plc_restricted_power":
                if options.polyhedral_schedule is None:
                    raise ValueError("The restricted power route requires its schedule.")
                polyhedral_source, volume = _polyhedral_request(source, specification)
                prepared = PreparedPolyhedralVolume(
                    polyhedral_source, volume, options.polyhedral_schedule
                )
                request = volume
            case "layer_core":
                if options.volume_schedule is None:
                    raise ValueError("The layer core route requires its schedule.")
                layer_source, volume = _layer_request(source, specification)
                prepared = PreparedLayerCore(
                    layer_source, volume, options.volume_schedule
                )
                request = volume
            case "surface_envelope_tetrahedral":
                if options.volume_schedule is None:
                    raise ValueError("The surface envelope route requires its schedule.")
                envelope_source, volume = _envelope_request(source, specification)
                execution = current_native_execution_budget()
                if execution is None:
                    raise RuntimeError(
                        "Envelope preparation lost its original native allowance."
                    )
                if (
                    not source_preparation_accounted
                    and not _native_volume_source_is_active(
                        execution,
                        envelope_source.envelope.envelope_id,
                    )
                ):
                    raise ValueError(
                        "Envelope preparation requires its actual ended source receipt "
                        "or its original active source owner."
                    )
                prepared = PreparedSurfaceEnvelopeVolume(
                    envelope_source, volume, options.volume_schedule
                )
                request = volume
            case "structured_transfinite":
                if options.structured_schedule is None:
                    raise ValueError("The structured route requires its schedule.")
                structured_source, request = _structured_request(source, specification)
                prepared = PreparedStructured(
                    structured_source.blocks,
                    structured_source.interfaces,
                    request,
                    structured_source.binding_id,
                    optimization_steps=options.structured_schedule.optimization_steps,
                )
            case "sweep":
                sweep_source, volume = _sweep_request(source, specification)
                prepared = PreparedSweep(
                    sweep_source.profile,
                    sweep_source.control,
                    volume,
                    sweep_source.binding_id,
                    profile_geometry=sweep_source.profile_geometry,
                )
                request = volume
            case "image_material_tetrahedral":
                if options.volume_schedule is None:
                    raise ValueError("The image material route requires its schedule.")
                material_source, volume = _compartment_request(source, specification)
                prepared = PreparedCompartmentVolume(
                    material_source, volume, options.volume_schedule
                )
                request = volume
            case "planar_dual_quad":
                planar_source, surface = _planar_request(source, specification)
                prepared = PreparedDualQuad(planar_source, surface)
                request = surface
            case "plc_dual_hex":
                if options.volume_schedule is None:
                    raise ValueError("The dual hex route requires its volume schedule.")
                plc_source, volume = _plc_request(source, specification)
                prepared = PreparedDualHex(plc_source, volume, options.volume_schedule)
                request = volume
            case "plc_hex_dominant":
                if options.volume_schedule is None or options.hex_core_fraction is None:
                    raise ValueError("The hybrid route requires its resolved schedule.")
                plc_source, volume = _plc_request(source, specification)
                prepared = PreparedHexDominant(
                    plc_source,
                    volume,
                    options.volume_schedule,
                    core_fraction=options.hex_core_fraction,
                )
                request = volume
            case "plc_balanced_grid_hex" | "plc_frame_grid_hex":
                if options.volume_schedule is None or options.grid_schedule is None:
                    raise ValueError(
                        "The hex-grid route requires its resolved schedules."
                    )
                plc_source, volume = _plc_request(source, specification)
                prepared = PreparedGridHex(
                    plc_source, volume, options.volume_schedule, options.grid_schedule
                )
                request = volume
            case "mapped_balanced_grid_hex" | "mapped_frame_grid_hex":
                if options.grid_schedule is None:
                    raise ValueError(
                        "The mapped hex-grid route requires its grid schedule."
                    )
                mapped_source, volume = _mapped_hex_request(source, specification)
                prepared = PreparedMappedGridHex(
                    mapped_source,
                    volume,
                    options.grid_schedule,
                    record_phase=record_phase,
                )
                request = volume
            case "implicit_restricted_delaunay":
                if options.volume_schedule is None:
                    raise ValueError(
                        "The restricted implicit volume route requires its schedule."
                    )
                implicit_source, volume = _implicit_volume_request(source, specification)
                axes = tuple(
                    np.asarray(axis.point_coordinates, dtype=np.float64)
                    for axis in implicit_source.grid.structured_axes
                )
                domain = np.asarray(
                    [(np.min(values), np.max(values)) for values in axes],
                    dtype=np.float64,
                ).T
                prepared = prepare_restricted_implicit_volume(
                    implicit_source.geometry,
                    domain,
                    volume,
                    options.volume_schedule,
                    coordinate_contract,
                )
                request = volume
            case "implicit_adaptive_tetrahedral":
                if (
                    options.volume_schedule is None
                    or type(options.implicit_policy) is not AdaptiveImplicitSurfacePolicy
                ):
                    raise ValueError(
                        "Adaptive implicit volumes require discovery and volume schedules."
                    )
                implicit_source, volume = _implicit_volume_request(source, specification)
                axes = tuple(
                    np.asarray(axis.point_coordinates, dtype=np.float64)
                    for axis in implicit_source.grid.structured_axes
                )
                domain = np.asarray(
                    [(np.min(axis), np.max(axis)) for axis in axes], dtype=np.float64
                ).T
                prepared = prepare_adaptive_implicit_volume(
                    implicit_source.geometry,
                    domain,
                    volume,
                    options.volume_schedule,
                    options.implicit_policy,
                    implicit_source.source_revision,
                    coordinate_contract,
                )
                request = volume
            case _:
                assert_never(options.route)
        plan = NativeMeshingPlan(
            source, request, options, support, coordinate_contract, prepared
        )
        record_elapsed(record_phase, "source_preparation", preparation_started)
        return plan


__all__ = ["NativeMeshingPlan", "NativeMeshingProvider"]
