#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admit immutable layer/core requests and execute the native provider route."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...geometry._domain_boundary import MeshingDomainBoundarySource
from ...geometry._meshing_domain import MeshingDomain
from ...typing import checked
from .._contracts import MeshingProviderInfo, VolumeFillStrategy, VolumeMeshingSpec
from .._controls import FeatureKind, ProtectedFeature, RegionControl
from .._hex_generation import mapped_source_lipschitz_bound
from .._layer_core import core_request, execute_layer_core_route
from .._layer_core_controls import layer_control_issues
from .._layer_core_mapped import (
    execute_mapped_layer_core_route,
    require_reference_root_sheets,
    validate_layer_reference_columns,
)
from .._layer_core_periodic import (
    core_periodic_constraint_evidence,
    prepare_core_periodic,
)
from .._layer_core_resources import (
    layer_source_work_upper,
    LayerCoreSourceWork,
    reserve_layer_storage,
)
from .._layer_core_topology import prepare_identity
from .._measurements import NativeExecutionRecord, NativeMeshingPhaseRecorder
from .._native_preparation import native_preparation_ledger
from .._result import CellMeshingResult
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import UniformSizeControl
from .._volume_generation import (
    NativeVolumeSchedule,
    require_native_preparation_allowance,
)
from . import _native_sources as sources_owner


def _lower_core_specification(
    source: sources_owner.NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> VolumeMeshingSpec:
    """Lower source face controls onto derived core facet scopes."""
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
        return replace(
            specification,
            region_controls=core_regions,
            patch_controls=(),
            layer_controls=(),
            periodic_constraints=(),
        )
    domain = source.source_domain
    retained = source.layers.source_domain
    mapping = source.core_facet_source_ids
    if domain is None or retained is None or mapping is None:
        raise ValueError(
            "NativeLayerCoreSource.source_domain must be retained with layers.source_domain and core_facet_source_ids for source-face controls; got incomplete domain ancestry."
        )
    if domain.domain_id != retained.domain_id:
        raise ValueError(
            "NativeLayerCoreSource.source_domain.domain_id must match "
            f"layers.source_domain.domain_id; got {domain.domain_id!r}, "
            f"expected {retained.domain_id!r}."
        )
    if boundary.scope_id != specification.boundary_scope.scope_id:
        raise ValueError(
            "VolumeMeshingSpec.boundary_scope.scope_id must match "
            f"NativeLayerCoreSource.source_boundary_scope.scope_id; got "
            f"{specification.boundary_scope.scope_id!r}, expected {boundary.scope_id!r}."
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
            "NativeLayerCoreSource.core_facet_source_ids must cover VolumeMeshingSpec.boundary_scope.entity_ids together with layers.control.wall_scope.entity_ids; got missing or additional source faces."
        )
    original = specification.size_controls[0]
    if (
        not isinstance(original, UniformSizeControl)
        or len(specification.size_controls) != 1
    ):
        raise ValueError(
            "VolumeMeshingSpec.size_controls must contain one whole-boundary UniformSizeControl; got a different size request."
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
                "VolumeMeshingSpec.protected_features.scope.entity_dimension must equal 2 without vertex/edge ancestry; got a lower-dimensional feature."
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
    return replace(
        specification,
        boundary_scope=scope,
        size_controls=(size,),
        protected_features=tuple(features),
        region_controls=core_regions,
        patch_controls=(),
        layer_controls=(),
        periodic_constraints=(),
    )


def _original_face_fidelity_issues(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    feature: ProtectedFeature,
    boundary: MeshingScope,
    /,
) -> list[str]:
    """Refuse scoped fidelity when retained faces lack nominal parametric queries."""
    match (source.source_domain, source.fidelity_source):
        case (MeshingDomain(), MeshingDomainBoundarySource()):
            return []
        case _:
            issues: list[str] = []
            whole = tuple(
                protected.maximum_deviation
                for protected in specification.protected_features
                if protected.feature_kind is FeatureKind.SURFACE
                and protected.scope.scope_id == boundary.scope_id
            )
            size = specification.size_controls[0]
            match size:
                case UniformSizeControl():
                    if feature.maximum_deviation < min(whole, default=size.target_size):
                        issues.append(
                            "a nominal original parametric face query for scoped continuous fidelity"
                        )
                case _:
                    pass
            if feature.feature_kind is FeatureKind.MATERIAL_INTERFACE:
                issues.append(
                    "a nominal original parametric interface query for scoped continuous fidelity"
                )
            return issues


def _layer_core_protected_feature_issues(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    boundary: MeshingScope,
    /,
) -> list[str]:
    """Check feature ancestry and fidelity in the declared request order."""
    issues: list[str] = []
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
            else:
                issues.extend(
                    _original_face_fidelity_issues(
                        source, specification, feature, boundary
                    )
                )
        elif dimension not in counts or np.any(
            np.asarray(feature.scope.entity_ids) >= counts[dimension]
        ):
            issues.append(
                "protected features outside supplied PLC vertices, segments and facets"
            )
    return issues


def _layer_core_boundary_control_issues(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    boundary: MeshingScope,
    /,
) -> list[str]:
    """Require retained ancestry for every boundary facet and one whole-boundary size control."""
    issues: list[str] = []
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
    match controls:
        case (UniformSizeControl() as size,):
            if size.scope.scope_id != boundary.scope_id:
                issues.append("a whole-core size-control scope")
        case _:
            issues.append("one uniform core size control")
    return issues


# Retain target, boundary, control, feature, and region issue order at admission.
@checked
def layer_core_support_issues(
    source: sources_owner.NativeLayerCoreSource, specification: VolumeMeshingSpec, /
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
    issues.extend(_layer_core_boundary_control_issues(source, specification, boundary))
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
    issues.extend(_layer_core_protected_feature_issues(source, specification, boundary))
    if any(
        seed.region_name not in source.complex.region_ids
        for seed in specification.region_seeds
    ):
        issues.append("region seeds naming an absent authoritative material region")
    return issues


def _mapped_reference_request(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> VolumeMeshingSpec:
    """Lower physical placement through the source-wide Jacobian bound."""

    reference, domain = source.reference_source, source.mapped_domain
    if reference is None or domain is None:
        raise ValueError(
            "NativeLayerCoreSource.reference_source must be supplied with mapped_domain for mapped placement; got incomplete reference roots."
        )
    if specification.region_seeds or specification.hole_seeds:
        raise ValueError(
            "VolumeMeshingSpec.region_seeds must be empty together with hole_seeds for mapped reference placement; got physical seed coordinates."
        )
    physical_core = core_request(
        _lower_core_specification(source, specification), source.layers.control
    )
    original = physical_core.size_controls[0]
    if not isinstance(original, UniformSizeControl):
        raise TypeError(
            "VolumeMeshingSpec.size_controls must contain UniformSizeControl for mapped placement; got a different size control."
        )
    bound = mapped_source_lipschitz_bound(
        domain.reference_mesh,
        domain.source_geometry,
        budget=budget,
        scope=scope,
    )
    boundary_scope = MeshingScope(
        reference.source_id,
        reference.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        f"{reference.complex.complex_id}:facets",
        np.arange(reference.complex.facet_count, dtype=np.int64),
    )
    size = UniformSizeControl(
        boundary_scope,
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
    return replace(
        specification,
        boundary_scope=boundary_scope,
        size_controls=(size,),
        region_controls=regions,
        protected_features=(),
        patch_controls=(),
        region_seeds=(),
        hole_seeds=(),
        layer_controls=(),
        periodic_constraints=(),
    )


@final
class PreparedLayerCore(StrictModule, NonTrainableState):
    """Retain validated layer/core identity and its numerical schedule."""

    schedule: NativeVolumeSchedule
    core_specification: VolumeMeshingSpec
    reference_specification: VolumeMeshingSpec | None
    reference_prepared: PreparedLayerCore | None
    source_work_units: int = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
        *,
        scope: NativeExecutionScope,
    ) -> None:
        with ExitStack() as resources:
            issues = layer_core_support_issues(source, specification)
            if issues:
                raise ValueError(
                    "PreparedLayerCore.specification must be supported by the layer/core "
                    f"route; got {'; '.join(issues)}."
                )
            limits = specification.limits
            allowance = scope.remaining()
            upper = (
                min(limits.maximum_work_units, allowance.remaining_work_units)
                if source.reference_source is not None
                or source.layers.mesh.periodic_topology is not None
                else layer_source_work_upper(source, specification)
            )
            ticket = resources.enter_context(
                scope.batch(
                    upper,
                    bytes_upper=min(
                        limits.maximum_scratch_bytes, allowance.remaining_managed_bytes
                    ),
                    queries_upper=min(
                        limits.maximum_geometry_queries,
                        allowance.remaining_geometry_queries,
                    ),
                )
            )
            work = LayerCoreSourceWork(limits.maximum_work_units, ticket=ticket)
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            resources.enter_context(budget.activate())
            mapping, _, polygons, points = prepare_identity(
                source.layers,
                source.complex,
                source.vertex_layer_ids,
                source.cap_polygon_ids,
                work=work,
            )
            prepare_core_periodic(
                source, mapping, polygons, points, work=work, budget=budget
            )
            core_periodic_constraint_evidence(
                source, specification, work=work, budget=budget
            )
            core_specification = _lower_core_specification(source, specification)
            reserve_layer_storage(
                source.layers,
                source.complex,
                core_request(core_specification, source.layers.control),
                source.vertex_layer_ids,
            )
            reference_specification = None
            reference_prepared = None
            if source.reference_source is not None:
                if source.mapped_domain is None:
                    raise ValueError(
                        "NativeLayerCoreSource.mapped_domain must be supplied with reference_source; got None."
                    )
                validate_layer_reference_columns(
                    source.layers,
                    source.reference_source.layers,
                    source.mapped_domain,
                    work,
                    budget=budget,
                )
                require_reference_root_sheets(
                    source.reference_source, source.mapped_domain, work
                )
                reference_specification = _mapped_reference_request(
                    source, specification, budget=budget, scope=scope
                )
                resources.close()
                del budget
                reference_prepared = PreparedLayerCore(
                    source.reference_source,
                    reference_specification,
                    schedule,
                    scope=scope,
                )
                work.work_units += reference_prepared.source_work_units
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


@checked
def execute_layer_route(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedLayerCore,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    preparation_evidence: NativeExecutionRecord | None = None,
) -> CellMeshingResult:
    """Execute the source/specification pair admitted by the plan."""
    if prepared.source_binding_id != source.binding_id:
        raise ValueError(
            "PreparedLayerCore.source_binding_id must match NativeLayerCoreSource.binding_id; "
            f"got {prepared.source_binding_id!r}, expected {source.binding_id!r}."
        )
    if prepared.specification_id != specification.specification_id:
        raise ValueError(
            "PreparedLayerCore.specification_id must match VolumeMeshingSpec.specification_id; "
            f"got {prepared.specification_id!r}, expected {specification.specification_id!r}."
        )
    if preparation_evidence is not None:
        preparation_evidence.require_valid()
        if preparation_evidence.owner_id != plan_id:
            raise ValueError(
                "execute_layer_route.preparation_evidence.owner_id must match plan_id; "
                f"got {preparation_evidence.owner_id!r}, expected {plan_id!r}."
            )
        require_native_preparation_allowance(specification.limits, preparation_evidence)
        workspace.retain_owner((source, specification, prepared, preparation_evidence))
        ledger = native_preparation_ledger(scope)
        ledger.import_receipt(scope, preparation_evidence)
        ledger.remember_live(scope, prepared)
    if not native_preparation_ledger(scope).live_contains(prepared):
        raise ValueError(
            "execute_layer_route.prepared must be live in the scope or have ended "
            f"preparation evidence; got an unregistered owner {prepared.prepared_id!r}."
        )
    if source.reference_source is not None:
        return execute_mapped_layer_core_route(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            record_phase=record_phase,
            scope=scope,
            workspace=workspace,
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
        scope=scope,
        workspace=workspace,
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
