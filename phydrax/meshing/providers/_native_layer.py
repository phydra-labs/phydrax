#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native immutable layer/core admission and canonical provider execution."""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import current_native_execution_budget, current_native_host_workspace
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._meshing_domain import MeshingDomain, MeshingDomainBoundarySource
from .._contracts import MeshingProviderInfo, VolumeFillStrategy, VolumeMeshingSpec
from .._controls import FeatureKind, ProtectedFeature, RegionControl
from .._layer_core import _core_request, _prepare_identity, execute_layer_core_route
from .._layer_core_controls import layer_control_issues
from .._layer_core_periodic import (
    core_periodic_constraint_evidence,
    prepare_core_periodic,
)
from .._layer_core_resources import LayerCoreSourceWork, reserve_layer_storage
from .._measurements import NativeExecutionRecord, NativeMeshingPhaseRecorder
from .._result import CellMeshingResult
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import UniformSizeControl
from .._volume_generation import (
    _import_native_preparation,
    _native_live_preparation_is_active,
    _native_volume_operation_started,
    _remember_native_live_preparation,
    _require_native_preparation_allowance,
    native_volume_execution_budget,
    NativeVolumeSchedule,
)
from ._native_sources import NativeLayerCoreSource


def _lower_core_specification(
    source: NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> VolumeMeshingSpec:
    """Lower original physical face controls through the declared ancestry map."""
    boundary = source.source_boundary_scope
    core_regions = tuple(
        RegionControl(
            MeshingScope(
                control.scope.source_id,
                control.scope.source_revision,
                control.scope.entity_kind,
                3,
                f"{source.complex.complex_id}:regions",
                np.asarray(
                    [source.complex.region_ids.index(control.region_name)], dtype=np.int64
                ),
            ),
            control.region_name,
            control.material_id,
            control.role,
            meshing_enabled=control.meshing_enabled,
        )
        for control in specification.region_controls
        if control.region_name in source.complex.region_ids
    )
    if boundary is None:
        return VolumeMeshingSpec(
            specification.target,
            specification.boundary_scope,
            specification.fill_strategy,
            size_controls=specification.size_controls,
            protected_features=specification.protected_features,
            region_controls=core_regions,
            region_seeds=specification.region_seeds,
            hole_seeds=specification.hole_seeds,
            size_combination=specification.size_combination,
            size_compliance=specification.size_compliance,
            limits=specification.limits,
            deterministic=specification.deterministic,
        )
    domain = source.source_domain
    retained = source.layers.source_domain
    mapping = source.core_facet_source_ids
    if domain is None or retained is None or mapping is None:
        raise ValueError(
            "Original layer controls require their retained domain and exact core facet ancestry."
        )
    if (
        domain.domain_id != retained.domain_id
        or boundary.scope_id != specification.boundary_scope.scope_id
    ):
        raise ValueError(
            "The layer/control/domain and original boundary request must bind the exact same scientific source."
        )
    represented = np.unique(
        np.concatenate(
            (
                mapping[mapping >= 0],
                np.asarray(source.layers.control.wall_scope.entity_ids, dtype=np.int64),
            )
        )
    )
    if not np.array_equal(represented, np.asarray(boundary.entity_ids, dtype=np.int64)):
        raise ValueError(
            "The original wall and remaining core facets must cover every requested original boundary face."
        )
    original = specification.size_controls[0]
    if (
        not isinstance(original, UniformSizeControl)
        or len(specification.size_controls) != 1
    ):
        raise ValueError(
            "A layer core requires one whole-original-boundary uniform size request."
        )
    complex_ = source.complex
    scope = MeshingScope(
        boundary.source_id,
        boundary.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        f"{complex_.complex_id}:facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    size = UniformSizeControl(
        scope,
        original.target_size,
        minimum_size=original.minimum_size,
        maximum_size=original.maximum_size,
        maximum_growth_rate=original.maximum_growth_rate,
        strength=original.strength,
        priority=original.priority,
    )
    features: list[ProtectedFeature] = []
    for feature in specification.protected_features:
        if feature.scope.entity_dimension != 2:
            raise ValueError(
                "Original lower-dimensional features require explicit vertex/edge ancestry."
            )
        selected = np.flatnonzero(np.isin(mapping, np.asarray(feature.scope.entity_ids)))
        if selected.size:
            features.append(
                ProtectedFeature(
                    MeshingScope(
                        scope.source_id,
                        scope.source_revision,
                        scope.entity_kind,
                        2,
                        scope.entity_set_id,
                        selected,
                    ),
                    feature.feature_kind,
                    maximum_deviation=feature.maximum_deviation,
                    hard=feature.hard,
                )
            )
    # The retained physical wall request has already produced the layers; it
    # must not grow another layer on the generated cap.
    return VolumeMeshingSpec(
        specification.target,
        scope,
        specification.fill_strategy,
        size_controls=(size,),
        protected_features=tuple(features),
        region_controls=core_regions,
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        deterministic=specification.deterministic,
    )


def layer_core_support_issues(
    source: NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    """Requests not enforced by the fixed-cap layer/core route remain explicit."""
    issues: list[str] = []
    target = specification.target
    if (
        target.topological_dimension,
        target.ambient_dimension,
        target.geometry_order,
    ) != (3, 3, 1):
        issues.append("affine mixed volume cells in three dimensions before curving")
    kinds = {block.cell_kind for block in source.layers.mesh.blocks} | {"tetrahedron"}
    policy = target.cell_families
    requested = set((*policy.required, *policy.preferred, *policy.allowed_transitions))
    if (
        not set(policy.required) <= kinds
        or not kinds <= requested
        or (len(kinds) > 1 and not policy.allow_mixed)
    ):
        issues.append("the requested exact layer/core cell-family policy")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        issues.append("a native simplex core fill strategy")
    if source.complex.boundary != "fixed":
        issues.append("a fixed immutable PLC core boundary")
    boundary = specification.boundary_scope
    if source.source_boundary_scope is None:
        if boundary.entity_dimension != 2 or not np.array_equal(
            np.asarray(boundary.entity_ids, dtype=np.int64),
            np.arange(source.complex.facet_count, dtype=np.int64),
        ):
            issues.append("a complete core PLC facet boundary scope")
        if specification.layer_controls:
            issues.append(
                "original boundary/domain/fidelity facts and exact core facet ancestry for physical layer controls"
            )
    elif boundary.scope_id != source.source_boundary_scope.scope_id:
        issues.append("the exact retained original boundary scope")
    controls = specification.size_controls
    if len(controls) != 1 or not isinstance(controls[0], UniformSizeControl):
        issues.append("one uniform core size control")
    elif controls[0].scope.scope_id != boundary.scope_id:
        issues.append("a whole-core size-control scope")
    issues.extend(layer_control_issues(source, specification))
    if (
        specification.periodic_constraints
        and source.layers.mesh.periodic_topology is None
    ):
        issues.append("source-authored periodic layers and core seam anatomy")
    if any(
        control.control_id != source.layers.control_id
        for control in specification.layer_controls
    ):
        issues.append(
            "layer controls different from the supplied accepted layer realization"
        )
    counts = {
        0: source.complex.vertices.shape[0],
        1: source.complex.segments.shape[0],
        2: source.complex.facet_count,
    }
    for feature in specification.protected_features:
        dimension = feature.scope.entity_dimension
        if source.source_boundary_scope is not None:
            if dimension != 2:
                issues.append(
                    "explicit original vertex/edge ancestry for lower-dimensional protected features"
                )
            elif np.setdiff1d(
                np.asarray(feature.scope.entity_ids), np.asarray(boundary.entity_ids)
            ).size:
                issues.append(
                    "protected source faces contained in the original boundary scope"
                )
            elif not isinstance(source.source_domain, MeshingDomain) or not isinstance(
                source.fidelity_source, MeshingDomainBoundarySource
            ):
                whole = tuple(
                    protected.maximum_deviation
                    for protected in specification.protected_features
                    if protected.feature_kind is FeatureKind.SURFACE
                    and protected.scope.scope_id == boundary.scope_id
                )
                size = specification.size_controls[0]
                if isinstance(
                    size, UniformSizeControl
                ) and feature.maximum_deviation < min(whole, default=size.target_size):
                    issues.append(
                        "a nominal original parametric face query for scoped continuous fidelity"
                    )
                if feature.feature_kind is FeatureKind.MATERIAL_INTERFACE:
                    issues.append(
                        "a nominal original parametric interface query for scoped continuous fidelity"
                    )
        elif dimension not in counts or np.any(
            np.asarray(feature.scope.entity_ids) >= counts[dimension]
        ):
            issues.append(
                "protected features outside supplied PLC vertices, segments and facets"
            )
    if any(
        seed.region_name not in source.complex.region_ids
        for seed in specification.region_seeds
    ):
        issues.append("region seeds naming an absent authoritative material region")
    return issues


def _mapped_reference_request(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
) -> VolumeMeshingSpec:
    """Lower physical placement through the exact source-wide Jacobian bound."""
    from .._hex_generation import mapped_source_lipschitz_bound

    reference, domain = source.reference_source, source.mapped_domain
    if reference is None or domain is None:
        raise ValueError(
            "Mapped layer/core placement requires its independently declared reference source and roots."
        )
    if specification.region_seeds or specification.hole_seeds:
        raise ValueError(
            "Physical seed coordinates cannot be reinterpreted as authored reference seed coordinates."
        )
    physical_core = _core_request(
        _lower_core_specification(source, specification), source.layers.control
    )
    original = physical_core.size_controls[0]
    if not isinstance(original, UniformSizeControl):
        raise TypeError(
            "Mapped layer/core placement requires the original uniform size control."
        )
    bound = mapped_source_lipschitz_bound(domain.reference_mesh, domain.source_geometry)
    scope = MeshingScope(
        reference.source_id,
        reference.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        f"{reference.complex.complex_id}:facets",
        np.arange(reference.complex.facet_count, dtype=np.int64),
    )
    size = UniformSizeControl(
        scope,
        original.target_size / bound,
        minimum_size=None
        if original.minimum_size is None
        else original.minimum_size / bound,
        maximum_size=None
        if original.maximum_size is None
        else original.maximum_size / bound,
        maximum_growth_rate=original.maximum_growth_rate,
        strength=original.strength,
        priority=original.priority,
    )
    regions = tuple(
        RegionControl(
            MeshingScope(
                reference.source_id,
                reference.source_revision,
                MeshingEntityKind.GEOMETRY,
                3,
                f"{reference.complex.complex_id}:regions",
                np.asarray(
                    [reference.region_ids.index(control.region_name)], dtype=np.int64
                ),
            ),
            control.region_name,
            control.material_id,
            control.role,
            meshing_enabled=control.meshing_enabled,
        )
        for control in specification.region_controls
        if control.region_name in reference.complex.region_ids
    )
    return VolumeMeshingSpec(
        specification.target,
        scope,
        specification.fill_strategy,
        size_controls=(size,),
        region_controls=regions,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        deterministic=specification.deterministic,
    )


@final
class PreparedLayerCore(StrictModule, NonTrainableState):
    """Validated exact identity handoff and the immutable numerical schedule."""

    schedule: NativeVolumeSchedule
    core_specification: VolumeMeshingSpec
    reference_specification: VolumeMeshingSpec | None
    reference_prepared: PreparedLayerCore | None
    source_work_units: int = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        if not isinstance(source, NativeLayerCoreSource):
            raise TypeError("source must be NativeLayerCoreSource.")
        if not isinstance(specification, VolumeMeshingSpec) or not isinstance(
            schedule, NativeVolumeSchedule
        ):
            raise TypeError(
                "A layer core requires VolumeMeshingSpec and NativeVolumeSchedule."
            )
        issues = layer_core_support_issues(source, specification)
        if issues:
            raise ValueError(
                "Unsupported native layer/core request: " + "; ".join(issues)
            )
        work = LayerCoreSourceWork(specification.limits.maximum_work_units)
        mapping, _, polygons, points = _prepare_identity(
            source.layers,
            source.complex,
            source.vertex_layer_ids,
            source.cap_polygon_ids,
            work=work,
        )
        prepare_core_periodic(source, mapping, polygons, points, work=work)
        core_periodic_constraint_evidence(source, specification, work=work)
        core_specification = _lower_core_specification(source, specification)
        reserve_layer_storage(
            source.layers,
            source.complex,
            _core_request(core_specification, source.layers.control),
            source.vertex_layer_ids,
        )
        reference_specification = None
        reference_prepared = None
        if source.reference_source is not None:
            from .._layer_core_mapped import (
                require_reference_root_sheets,
                validate_layer_reference_columns,
            )

            if source.mapped_domain is None:
                raise ValueError(
                    "Mapped layer/core source lost its original source roots."
                )
            validate_layer_reference_columns(
                source.layers, source.reference_source.layers, source.mapped_domain, work
            )
            require_reference_root_sheets(
                source.reference_source, source.mapped_domain, work
            )
            reference_specification = _mapped_reference_request(source, specification)
            reference_prepared = PreparedLayerCore(
                source.reference_source, reference_specification, schedule
            )
            work.work_units += reference_prepared.source_work_units
            work.charge(0)
        self.schedule = schedule
        self.core_specification = core_specification
        self.reference_specification = reference_specification
        self.reference_prepared = reference_prepared
        self.source_work_units = work.work_units
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-layer-core",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
                "core_specification": core_specification.specification_id,
                "source_work_units": work.work_units,
                "reference_specification": None
                if reference_specification is None
                else reference_specification.specification_id,
                "reference_prepared": None
                if reference_prepared is None
                else reference_prepared.prepared_id,
            }
        )


def execute_layer_route(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedLayerCore,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    preparation_evidence: NativeExecutionRecord | None = None,
) -> CellMeshingResult:
    """Execute only the exact source/specification pair admitted by the plan."""
    if (
        prepared.source_binding_id != source.binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "The prepared layer/core route binds another source or specification."
        )
    active = current_native_execution_budget()
    if active is None or preparation_evidence is not None:
        from time import monotonic

        from ._native_publication import bind_native_execution_result

        if type(preparation_evidence) is not NativeExecutionRecord:
            raise ValueError(
                "Standalone prepared layer execution requires its actual preparation receipt."
            )
        preparation_evidence.require_valid()
        if preparation_evidence.owner_id != plan_id:
            raise ValueError(
                "Layer preparation receipt binds another native meshing plan."
            )
        _require_native_preparation_allowance(specification.limits, preparation_evidence)
        work = int(np.asarray(preparation_evidence.total_work_units))
        queries = int(np.asarray(preparation_evidence.total_geometry_queries))
        seconds = float(np.asarray(preparation_evidence.total_elapsed_seconds))
        started = monotonic() - seconds
        parent_started = _native_volume_operation_started()
        if parent_started is not None:
            started = min(started, parent_started)
        if active is not None:
            storage = current_native_host_workspace()
            if storage is None:
                raise RuntimeError(
                    "Layer preparation import lost its original host workspace."
                )
            _import_native_preparation(active, storage, preparation_evidence)
        with native_volume_execution_budget(
            specification.limits,
            source_work_units=work,
            source_geometry_queries=queries,
            operation_started=started,
            borrow_active=False,
        ) as budget:
            storage = current_native_host_workspace()
            if storage is None:
                raise RuntimeError(
                    "Standalone layer execution lost its original host workspace."
                )
            storage.retain_owner((source, specification, prepared, preparation_evidence))
            _remember_native_live_preparation(budget, storage, prepared)
            result = execute_layer_route(
                source,
                specification,
                prepared,
                coordinate_contract,
                provider,
                plan_id,
                record_phase=record_phase,
            )
        if budget.evidence is None:
            raise RuntimeError(
                "Standalone layer execution lost its ended original-scope receipt."
            )
        return bind_native_execution_result(
            result,
            budget.evidence,
            specification.limits,
            started,
            preparation_evidence=preparation_evidence,
        )
    if not _native_live_preparation_is_active(prepared):
        raise ValueError(
            "Scoped layer execution requires its exact live prepared owner or its actual preparation receipt."
        )
    if source.reference_source is not None:
        from .._layer_core_mapped import execute_mapped_layer_core_route

        return execute_mapped_layer_core_route(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            record_phase=record_phase,
        )
    return execute_layer_core_route(
        source.layers,
        source.complex,
        specification,
        prepared.schedule,
        coordinate_contract,
        provider,
        plan_id,
        vertex_layer_ids=source.vertex_layer_ids,
        cap_polygon_ids=source.cap_polygon_ids,
        layer_regions=source.layer_regions,
        source_id=source.source_id,
        source_revision=source.source_revision,
        generation_specification=prepared.core_specification,
        original_source=source,
        source_preparation_work_units=prepared.source_work_units,
        prepared_owner=prepared,
        record_phase=record_phase,
    )


__all__ = ["PreparedLayerCore", "layer_core_support_issues", "execute_layer_route"]
