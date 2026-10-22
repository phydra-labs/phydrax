#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""prepared conservative finite-volume remap preparation.

certified reference tilings of represented coordinate maps use certified finer-cell
measures directly. Bounded surface chart deformation uses represented native UV
coverage with certified old physical contents and represented new physical areas.
Corner geometry uses the geometric common-refinement owner. The routes retain
distinct scientific evidence.
"""

from __future__ import annotations

from typing import assert_never, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
from phydrax.discretization._sphere_chart_records import PreparedSphereChartDeformation
from phydrax.discretization._surface_chart_records import PreparedSurfaceChartDeformation

from ... import (
    _meshcore as imported_meshcore,
    _resource_ticket as imported_resource_ticket,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._strict import StrictModule
from ...geometry import _supermesh as imported_geometry__supermesh
from ...geometry._supermesh import CommonRefinementPolicy
from ...meshing import (
    _adaptation as imported_meshing__adaptation,
    _measurements as imported_meshing__measurements,
    _tetra_metric as imported_meshing__tetra_metric,
    _volume_generation as imported_meshing__volume_generation,
)
from ...meshing._adaptation import MeshAdaptationResult
from ...meshing._measurements import NativeExecutionRecord
from ...typing import checked
from .. import (
    _cell_geometry as imported_cell_geometry,
    _coordinate_budget as imported_coordinate_budget,
    _nested_reference as imported_nested_reference,
)
from .._cell_geometry import cell_geometry_preparation_query_upper, CellGeometrySpec
from .._cell_geometry_transition_types import CellGeometryTransition
from .._cell_geometry_validity import cell_geometry_id
from .._coordinate_budget import CoordinateEnclosureBudget
from .._nested_reference_types import reference_pair_record
from ..fem import (
    _sphere_chart_transfer as imported_fem__sphere_chart_transfer,
    _surface_chart_transfer as imported_fem__surface_chart_transfer,
)
from ._coordinate_source import has_power_source
from ._remap_evidence import (
    MappedNestedRemapEvidence,
    MappedSurfaceChartRemapEvidence,
    PreparedUnstructuredConservativeRemap,
    RemapPreparationFailure,
)
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_remap import coverage_ledger, UnstructuredConservativeRemapPlan


@final
class _RemapRequest(StrictModule):
    """Endpoints and scientific inputs shared by the remap preparation phases."""

    source: UnstructuredFiniteVolumeDiscretization
    target: UnstructuredFiniteVolumeDiscretization
    provenance: str = eqx.field(static=True)
    policy: CommonRefinementPolicy
    source_geometry: CellGeometrySpec | None
    target_geometry: CellGeometrySpec | None
    parent_cells: ArrayLike | None
    parent_reference_vertices: ArrayLike | None
    parent_reference_homogeneous: ArrayLike | None
    geometry_transition: CellGeometryTransition | None
    surface_chart_deformation: (
        PreparedSurfaceChartDeformation | PreparedSphereChartDeformation | None
    )


@checked
def prepare_unstructured_conservative_remap(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    /,
    *,
    provenance: str,
    policy: imported_geometry__supermesh.CommonRefinementPolicy | None = None,
    source_geometry: CellGeometrySpec | None = None,
    target_geometry: CellGeometrySpec | None = None,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    parent_reference_homogeneous: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    surface_chart_deformation: PreparedSurfaceChartDeformation
    | PreparedSphereChartDeformation
    | None = None,
    adaptation: imported_meshing__adaptation.MeshAdaptationResult | None = None,
    source_workspace: NativeHostStorageWorkspace | None = None,
    scope: imported_meshcore.NativeExecutionScope | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedUnstructuredConservativeRemap:
    """Prepare remap contents under the policy and any preceding adaptation.

    certified mapped requests can run without meshcore. Native requests lower their
    domain limits once. A supplied scope, settled coordinate budget and persistent
    workspace keep a consumer on its explicit producer's cumulative allowance.
    Batch tickets are refused as wrong object kinds at the checked boundary.
    """

    policy_ = (
        imported_geometry__supermesh.CommonRefinementPolicy()
        if policy is None
        else policy
    )
    if source_workspace is not None and (scope is None or budget is None):
        raise ValueError(
            "prepare_unstructured_conservative_remap.source_workspace must have a scope and coordinate budget; got a missing execution owner."
        )
    if scope is not None and budget is None:
        raise ValueError(
            "prepare_unstructured_conservative_remap.budget must accompany an explicit scope; got None."
        )

    previous: imported_meshing__measurements.NativeExecutionRecord | None = None
    owner_id: str | None = None
    if adaptation is not None:
        previous, owner_id = _remap_predecessor(source, target, adaptation)
        source_geometry = (
            adaptation.source.geometry if source_geometry is None else source_geometry
        )
        target_geometry = (
            adaptation.target.geometry if target_geometry is None else target_geometry
        )
        geometry_transition = (
            adaptation.geometry_transition
            if geometry_transition is None
            else geometry_transition
        )

    request = _RemapRequest(
        source,
        target,
        provenance,
        policy_,
        source_geometry,
        target_geometry,
        parent_cells,
        parent_reference_vertices,
        parent_reference_homogeneous,
        geometry_transition,
        surface_chart_deformation,
    )
    exact_request = (
        surface_chart_deformation is not None
        or parent_cells is not None
        or parent_reference_vertices is not None
        or parent_reference_homogeneous is not None
        or geometry_transition is not None
        or any(
            geometry is not None
            and any(element.cell_kind == "polyhedron" for element in geometry.elements)
            and has_power_source(geometry)
            for geometry in (source_geometry, target_geometry)
        )
    )
    if adaptation is None and scope is None and exact_request:
        if budget is not None:
            return _prepare_unstructured_conservative_remap_state(
                request, scope=None, budget=budget
            )
        ticket = imported_resource_ticket.HostBatchTicket(
            policy_.maximum_exact_work,
            queries_upper=_remap_query_upper(
                source_geometry, target_geometry, maximum_work=policy_.maximum_exact_work
            ),
            bytes_upper=policy_.maximum_memory_bytes,
        )
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=policy_.maximum_exact_work,
            maximum_memory_bytes=policy_.maximum_memory_bytes,
        )
        with ticket, budget.activate():
            return _prepare_unstructured_conservative_remap_state(
                request, scope=None, budget=budget
            )

    if scope is not None:
        if budget is None:
            raise RuntimeError(
                "prepare_unstructured_conservative_remap.budget must accompany the native scope; got None."
            )
        if not budget.ticket.settled and not budget.ticket.native_accounting:
            raise ValueError(
                "prepare_unstructured_conservative_remap.budget must charge the active native request; got a live nonnative ticket."
            )
        if adaptation is not None and budget.ticket.settled:
            if owner_id is None:
                raise RuntimeError(
                    "prepare_unstructured_conservative_remap.owner_id must identify the adaptation source; got None."
                )
            with (
                imported_meshing__volume_generation.native_volume_source_execution_budget(
                    adaptation.policy.limits, previous, owner_id, scope=scope
                )
            ):
                return _prepare_remap_phase(
                    request, scope=scope, budget=budget, source_workspace=source_workspace
                )
        return _prepare_remap_phase(
            request, scope=scope, budget=budget, source_workspace=source_workspace
        )

    if budget is not None:
        raise ValueError(
            "prepare_unstructured_conservative_remap.scope must accompany a native coordinate budget; got None."
        )
    return _prepare_native_remap(request, adaptation, previous, owner_id)


def _prepare_native_remap(
    request: _RemapRequest,
    adaptation: MeshAdaptationResult | None,
    previous: NativeExecutionRecord | None,
    owner_id: str | None,
    /,
) -> PreparedUnstructuredConservativeRemap:
    """Execute an owned native scope and publish its ended remap evidence."""
    source, target, policy_ = request.source, request.target, request.policy
    bootstrap = imported_resource_ticket.HostBatchTicket(0)
    bootstrap.settle()
    budget = CoordinateEnclosureBudget(
        bootstrap,
        maximum_work_units=policy_.maximum_exact_work,
        maximum_memory_bytes=policy_.maximum_memory_bytes,
    )
    initial_work = budget.work_units
    native = imported_meshcore.NativeExecutionScope(
        policy_.native_limits()
        if adaptation is None
        else adaptation.policy.limits.native_limits()
    )
    if adaptation is None:
        with (
            native,
            budget.retained_storage(lambda size: native.batch(0, bytes_upper=size)),
        ):
            return _prepare_remap_phase(
                request, scope=native, budget=budget, source_workspace=None
            )
    if owner_id is None:
        raise RuntimeError(
            "prepare_unstructured_conservative_remap.owner_id must identify the adaptation source; got None."
        )
    prepared: PreparedUnstructuredConservativeRemap | None = None
    reason: str | None = None
    try:
        with budget.measurement(), native, native.host_workspace() as workspace:
            workspace.retain_owner((adaptation.source, adaptation.target, previous))
            with (
                budget.retained_storage(lambda size: native.batch(0, bytes_upper=size)),
                imported_meshing__volume_generation.native_volume_source_execution_budget(
                    adaptation.policy.limits, previous, owner_id, scope=native
                ),
            ):
                prepared = _prepare_remap_phase(
                    request, scope=native, budget=budget, source_workspace=workspace
                )
    except imported_coordinate_budget.CoordinateEnclosureResourceError as error:
        reason = str(error)
    except imported_meshcore.MeshcoreError as error:
        match error.status:
            case (
                imported_meshcore.MeshcoreStatus.CAPACITY_EXCEEDED
                | imported_meshcore.MeshcoreStatus.TIMEOUT
            ):
                reason = str(error)
            case _:
                raise
    except _leaf_phydrax_meshing__failures.MeshingFailure as error:
        match error.category:
            case (
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
                | _leaf_phydrax_meshing__failures.MeshingFailureCategory.TIMED_OUT
            ):
                reason = str(error)
            case _:
                raise
    if native.evidence is None:
        raise RuntimeError(
            "prepare_unstructured_conservative_remap.execution_evidence must describe the ended scope; got None."
        )
    record = imported_meshing__measurements.NativeExecutionRecord(
        native.evidence, owner_id=owner_id, preparation_evidence=previous
    )
    if reason is not None:
        failure = RemapPreparationFailure(
            source_discretization_id=source.prepared_id,
            target_discretization_id=target.prepared_id,
            execution_evidence=record,
            logical_exact_work_units=jnp.asarray(
                budget.work_units - initial_work, dtype=jnp.uint64
            ),
            host_storage_upper_bytes=jnp.asarray(
                budget.peak_bytes_upper, dtype=jnp.uint64
            ),
            reason=reason,
        )
        return PreparedUnstructuredConservativeRemap(
            failure,
            None,
            status=imported_geometry__supermesh.CommonRefinementStatus.RESOURCE_LIMIT,
            reason=reason,
            execution_evidence=record,
        )
    if prepared is None:
        raise RuntimeError(
            "prepare_unstructured_conservative_remap.result must contain a remap or resource refusal; got neither."
        )
    return bind_remap_execution(prepared, record)


def _prepare_remap_phase(
    request: _RemapRequest,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    source_workspace: NativeHostStorageWorkspace | None,
) -> PreparedUnstructuredConservativeRemap:
    source, target, policy = request.source, request.target, request.policy
    source_geometry, target_geometry = request.source_geometry, request.target_geometry
    surface_chart_deformation = request.surface_chart_deformation
    if source_workspace is not None:
        source_workspace.retain_owner(
            (source, target, source_geometry, target_geometry, surface_chart_deformation)
        )
    with (
        budget.bind_ticket(scope.ticket_view())
        if budget.ticket.settled
        else budget.activate(),
        budget.measurement(),
        budget.bound_stage(policy.maximum_exact_work, policy.maximum_memory_bytes),
        budget.ticket.native_target(),
    ):
        prepared = _prepare_unstructured_conservative_remap_state(
            request, scope=scope, budget=budget
        )
    if source_workspace is not None:
        source_workspace.retain_owner(prepared)
    return prepared


def _remap_query_upper(
    source_geometry: CellGeometrySpec | None,
    target_geometry: CellGeometrySpec | None,
    /,
    *,
    maximum_work: int,
) -> int:
    # A positive-query preparation is a counted visit; include the fail-fast
    # prefix without equating copied source vertices to coordinate work.
    return (maximum_work + 1) * sum(
        cell_geometry_preparation_query_upper(geometry)
        for geometry in (source_geometry, target_geometry)
        if geometry is not None
    )


def _remap_predecessor(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    adaptation: MeshAdaptationResult,
    /,
) -> tuple[NativeExecutionRecord, str]:
    """Validate the remap endpoints and return their ended producer execution record."""

    if (source.mesh.mesh_id, target.mesh.mesh_id) != (
        adaptation.source.mesh.mesh_id,
        adaptation.target.mesh.mesh_id,
    ):
        raise ValueError(
            "prepare_unstructured_conservative_remap.adaptation must bind both FV endpoint meshes; got different mesh IDs."
        )
    if (
        not isinstance(
            adaptation.evidence, imported_meshing__tetra_metric.MetricRemeshingEvidence
        )
        or adaptation.evidence.execution_evidence is None
    ):
        raise ValueError(
            "prepare_unstructured_conservative_remap.adaptation must carry an ended native execution record; got missing execution evidence."
        )
    previous = adaptation.evidence.execution_evidence
    previous.require_valid()
    owner_id = previous.owner_id
    if owner_id is None:
        raise ValueError(
            "prepare_unstructured_conservative_remap.adaptation must retain its source owner ID; got None."
        )
    if int(np.asarray(previous.status)) != 0:
        raise ValueError(
            "prepare_unstructured_conservative_remap.adaptation must have a successful native status; got a failed predecessor."
        )
    return previous, owner_id


@checked
def bind_remap_execution(
    prepared: PreparedUnstructuredConservativeRemap,
    record: NativeExecutionRecord,
    /,
) -> PreparedUnstructuredConservativeRemap:
    """Bind the ended execution record while retaining the geometric route."""
    match prepared.route:
        case MappedSurfaceChartRemapEvidence() | RemapPreparationFailure() as route:
            bound_route = eqx.tree_at(
                lambda value: value.execution_evidence,
                route,
                record,
                is_leaf=lambda value: value is None,
            )
        case (
            imported_geometry__supermesh.PreparedCommonRefinement()
            | MappedNestedRemapEvidence()
        ) as route:
            bound_route = route
        case invalid:
            assert_never(invalid)
    return PreparedUnstructuredConservativeRemap(
        bound_route,
        prepared.plan,
        status=prepared.status,
        reason=prepared.reason,
        execution_evidence=record,
    )


def _prepare_unstructured_conservative_remap_state(
    request: _RemapRequest,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    scope: NativeExecutionScope | None,
) -> PreparedUnstructuredConservativeRemap:
    """Dispatch chart, nested-reference, and common-refinement remap requests."""
    source, target, provenance = request.source, request.target, request.provenance
    policy_ = request.policy
    source_geometry, target_geometry = request.source_geometry, request.target_geometry
    parent_cells = request.parent_cells
    parent_reference_vertices = request.parent_reference_vertices
    parent_reference_homogeneous = request.parent_reference_homogeneous
    geometry_transition = request.geometry_transition
    if not isinstance(provenance, str):
        raise TypeError(
            "prepare_unstructured_conservative_remap.provenance must be a string; got a non-string value."
        )
    if not provenance:
        raise ValueError(
            "prepare_unstructured_conservative_remap.provenance must be non-empty; got an empty string."
        )
    chart = _bound_remap_chart(request)
    if chart is not None:
        if (
            parent_cells is not None
            or parent_reference_vertices is not None
            or parent_reference_homogeneous is not None
        ):
            raise ValueError(
                "prepare_unstructured_conservative_remap.parent_cells must be absent for chart deformation; got explicit nested-reference inputs."
            )
        return _prepare_mapped_surface_chart_remap_state(
            source,
            target,
            source_geometry,
            target_geometry,
            chart,
            provenance=provenance,
            policy=policy_,
            budget=budget,
        )
    # Only power sources own the certified polyhedral common refinement; PLC sources
    # take the mapped route through their certified source coordinates.
    exact_power = any(
        geometry is not None
        and any(element.cell_kind == "polyhedron" for element in geometry.elements)
        and has_power_source(geometry)
        for geometry in (source_geometry, target_geometry)
    )
    if exact_power:
        for discretization, geometry in (
            (source, source_geometry),
            (target, target_geometry),
        ):
            if (
                geometry is None
                or discretization.cell_geometry is None
                or cell_geometry_id(discretization.cell_geometry)
                != cell_geometry_id(geometry)
            ):
                raise ValueError(
                    "prepare_unstructured_conservative_remap.cell_geometry must bind the power-source FV endpoint measures; got a missing or different coordinate map."
                )
    witnesses = (
        parent_cells is not None
        or parent_reference_vertices is not None
        or parent_reference_homogeneous is not None
        or geometry_transition is not None
    )
    if witnesses and not exact_power:
        if source_geometry is None or target_geometry is None:
            raise TypeError(
                "prepare_unstructured_conservative_remap.cell_geometry must specify both nested endpoint maps; got a missing source or target geometry."
            )
        return _prepare_mapped_nested_remap(
            source,
            target,
            source_geometry,
            target_geometry,
            provenance=provenance,
            policy=policy_,
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            parent_reference_homogeneous=parent_reference_homogeneous,
            geometry_transition=geometry_transition,
            budget=budget,
        )
    _require_common_refinement_geometry(request, exact_power, budget)
    if exact_power:
        refinement = imported_geometry__supermesh.prepare_common_refinement(
            source.mesh,
            target.mesh,
            policy=policy_,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            budget=budget,
            scope=scope,
        )
    else:
        if scope is None:
            raise RuntimeError(
                "prepare_unstructured_conservative_remap.scope must bind the native remap; got None."
            )

        refinement = imported_geometry__supermesh.prepare_common_refinement_active(
            source.mesh, target.mesh, policy=policy_, scope=scope, budget=budget
        )
    return _publish_common_refinement(request, refinement)


def _bound_remap_chart(
    request: _RemapRequest, /
) -> PreparedSurfaceChartDeformation | PreparedSphereChartDeformation | None:
    """Validate the transition binding before selecting its material chart."""
    chart = request.surface_chart_deformation
    transition = request.geometry_transition
    if transition is not None:
        carried = transition.chart_deformation
        if carried is not None:
            if (
                transition.source_topology_id != carried.source_topology_id
                or transition.target_topology_id != carried.target_topology_id
                or transition.source_geometry_id != carried.source_geometry_id
                or transition.target_geometry_id != carried.target_geometry_id
                or cell_geometry_id(transition.geometry) != carried.target_geometry_id
            ):
                raise ValueError(
                    "prepare_unstructured_conservative_remap.geometry_transition must bind its carried chart deformation; got different topology or geometry IDs."
                )
            if chart is not None and chart.deformation_id != carried.deformation_id:
                raise ValueError(
                    "prepare_unstructured_conservative_remap.surface_chart_deformation must match the transition chart; got different deformation IDs."
                )
            match carried:
                case (
                    PreparedSurfaceChartDeformation() | PreparedSphereChartDeformation()
                ) as supported_chart:
                    chart = supported_chart
                case _:
                    raise TypeError(
                        "prepare_unstructured_conservative_remap.geometry_transition.chart_deformation must support surface or sphere FV remap; "
                        f"got {type(carried).__name__}."
                    )
    return chart


def _require_common_refinement_geometry(
    request: _RemapRequest,
    exact_power: bool,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Validate affine coordinate banks before geometric common refinement."""
    source, target = request.source, request.target
    source_geometry, target_geometry = request.source_geometry, request.target_geometry
    mapped = source_geometry is not None or target_geometry is not None
    if mapped and not exact_power:
        for discretization, geometry in (
            (source, source_geometry),
            (target, target_geometry),
        ):
            if geometry is None:
                raise TypeError(
                    "prepare_unstructured_conservative_remap.cell_geometry must specify both nonnested endpoint maps; got a missing source or target geometry."
                )
            affine = CellGeometrySpec.affine(discretization.mesh, budget=budget)
            if cell_geometry_id(geometry) != cell_geometry_id(
                affine
            ) or array_tree_fingerprint(geometry) != array_tree_fingerprint(affine):
                raise ValueError(
                    "prepare_unstructured_conservative_remap.cell_geometry must match the affine coordinate banks for nonnested remap; got a nonaffine or different map."
                )
            prepared_geometry = discretization.cell_geometry
            if prepared_geometry is not None and (
                cell_geometry_id(prepared_geometry) != cell_geometry_id(geometry)
                or array_tree_fingerprint(prepared_geometry)
                != array_tree_fingerprint(geometry)
            ):
                raise ValueError(
                    "prepare_unstructured_conservative_remap.cell_geometry must match the prepared affine coordinate banks; got different geometry data."
                )
    elif not exact_power and (
        source.cell_geometry is not None or target.cell_geometry is not None
    ):
        raise ValueError(
            "prepare_unstructured_conservative_remap.cell_geometry must accompany prepared mapped endpoints; got a missing coordinate map."
        )


def _publish_common_refinement(
    request: _RemapRequest,
    refinement: imported_geometry__supermesh.PreparedCommonRefinement,
    /,
) -> PreparedUnstructuredConservativeRemap:
    """Check coverage evidence and assemble the conservative remap plan."""
    source, target, policy_ = request.source, request.target, request.policy
    provenance = request.provenance
    if not refinement.succeeded:
        return PreparedUnstructuredConservativeRemap(
            refinement,
            None,
            status=refinement.status,
            reason=f"{refinement.status.name}: {refinement.evidence.reason}",
        )
    require_complete = (
        policy_.coverage is imported_geometry__supermesh.CommonRefinementCoverage.COMPLETE
    )
    if require_complete:
        coverage = coverage_ledger(
            np.asarray(refinement.target_cells),
            np.asarray(refinement.source_cells),
            np.asarray(refinement.volumes, dtype=np.float64),
            np.asarray(source.cell_volumes),
            np.asarray(target.cell_volumes),
            policy_.coverage_tolerance,
            intersection_error_bounds=None
            if refinement.volume_error_bounds is None
            else np.asarray(refinement.volume_error_bounds),
            source_error_bounds=np.asarray(source.cell_volume_error_bounds),
            target_error_bounds=np.asarray(target.cell_volume_error_bounds),
        )
        if np.any(coverage.target_defect > coverage.target_limit) or np.any(
            coverage.source_defect > coverage.source_limit
        ):
            return PreparedUnstructuredConservativeRemap(
                refinement,
                None,
                status=imported_geometry__supermesh.CommonRefinementStatus.DOUBLE_COVERAGE,
                reason="certified overlap measures exceed finite-volume cell volumes",
            )
        if np.any(coverage.target_defect < -coverage.target_limit) or np.any(
            coverage.source_defect < -coverage.source_limit
        ):
            return PreparedUnstructuredConservativeRemap(
                refinement,
                None,
                status=imported_geometry__supermesh.CommonRefinementStatus.COVERAGE_GAP,
                reason="certified overlap measures leave finite-volume cell volume uncovered",
            )
    plan = UnstructuredConservativeRemapPlan(
        source,
        target,
        refinement.target_offsets,
        refinement.source_cells,
        refinement.volumes,
        method="common-refinement",
        provenance=provenance,
        tolerance=policy_.coverage_tolerance,
        require_complete=require_complete,
        intersection_error_bounds=refinement.volume_error_bounds,
    )
    return PreparedUnstructuredConservativeRemap(
        refinement,
        plan,
        status=imported_geometry__supermesh.CommonRefinementStatus.SUCCESS,
        reason="certified complete common refinement"
        if require_complete
        else f"certified {policy_.coverage.value} common refinement",
    )


def _prepare_mapped_nested_remap(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    source_geometry: imported_cell_geometry.CellGeometrySpec,
    target_geometry: imported_cell_geometry.CellGeometrySpec,
    /,
    *,
    provenance: str,
    budget: CoordinateEnclosureBudget,
    policy: CommonRefinementPolicy,
    parent_cells: ArrayLike | None,
    parent_reference_vertices: ArrayLike | None,
    parent_reference_homogeneous: ArrayLike | None,
    geometry_transition: CellGeometryTransition | None,
) -> PreparedUnstructuredConservativeRemap:

    for discretization, geometry in (
        (source, source_geometry),
        (target, target_geometry),
    ):
        if discretization.cell_geometry is None or cell_geometry_id(
            discretization.cell_geometry
        ) != cell_geometry_id(geometry):
            raise ValueError(
                "prepare_unstructured_conservative_remap.cell_geometry must match the prepared FV measures; got a different geometry ID."
            )
        measures, errors, _ = mapped_measure_owner.certified_cell_measures(
            discretization.mesh, geometry, budget=budget
        )
        if not np.array_equal(
            np.asarray(discretization.cell_volumes), measures
        ) or not np.array_equal(
            np.asarray(discretization.cell_volume_error_bounds), errors
        ):
            raise ValueError(
                "prepare_unstructured_conservative_remap.cell_volumes must match the certified mapped measures and errors; got inconsistent prepared values."
            )
    pairs = imported_nested_reference.nested_reference_pairs(
        source.mesh,
        source_geometry,
        target.mesh,
        target_geometry,
        parent_cells=parent_cells,
        parent_reference_vertices=parent_reference_vertices,
        parent_reference_homogeneous=parent_reference_homogeneous,
        geometry_transition=geometry_transition,
        budget=budget,
    )
    geometry_error = mapped_measure_owner.certify_nested_geometry_pairs(
        source.mesh,
        source_geometry,
        target.mesh,
        target_geometry,
        pairs,
        budget=budget,
    )
    ordered = sorted(pairs, key=lambda pair: (pair.target_cell, pair.source_cell))
    indices = np.asarray([pair.source_cell for pair in ordered], dtype=np.int32)
    routes = np.asarray([pair.target_cell for pair in ordered], dtype=np.int32)
    measures = np.asarray(
        [
            source.cell_volumes[pair.source_cell]
            if pair.fine_is_source
            else target.cell_volumes[pair.target_cell]
            for pair in ordered
        ],
        dtype=np.float64,
    )
    errors = np.asarray(
        [
            source.cell_volume_error_bounds[pair.source_cell]
            if pair.fine_is_source
            else target.cell_volume_error_bounds[pair.target_cell]
            for pair in ordered
        ],
        dtype=np.float64,
    )
    offsets = np.concatenate(
        ([0], np.cumsum(np.bincount(routes, minlength=target.cell_count)))
    ).astype(np.int32)
    witness_id = canonical_fingerprint(
        {
            "kind": "mapped-nested-fv-reference-tiling",
            "source_topology": source.mesh.topology_id,
            "target_topology": target.mesh.topology_id,
            "source_geometry": cell_geometry_id(source_geometry),
            "target_geometry": cell_geometry_id(target_geometry),
            # Dyadic pairs keep their binary64 identity; retained exact maps
            # additionally bind their rational rows.
            "pairs": [
                (
                    p.source_cell,
                    p.target_cell,
                    p.fine_is_source,
                    array_tree_fingerprint(p.matrix),
                    array_tree_fingerprint(p.offset),
                    *(() if p.exact_matrix is None else (reference_pair_record(p)[3:],)),
                )
                for p in ordered
            ],
        }
    )
    plan = UnstructuredConservativeRemapPlan(
        source,
        target,
        offsets,
        indices,
        measures,
        method="mapped-nested",
        provenance=provenance,
        tolerance=policy.coverage_tolerance,
        require_complete=True,
        intersection_error_bounds=errors,
    )
    evidence = MappedNestedRemapEvidence(
        source_topology_id=source.mesh.topology_id,
        target_topology_id=target.mesh.topology_id,
        source_geometry_id=cell_geometry_id(source_geometry),
        target_geometry_id=cell_geometry_id(target_geometry),
        reference_witness_id=witness_id,
        geometry_error_bound=geometry_error,
        measure_exact=source.cell_volume_exact and target.cell_volume_exact,
        reason="exact reference tiling with certified mapped finer-cell overlap measures",
        report=plan.report,
    )
    return PreparedUnstructuredConservativeRemap(
        evidence,
        plan,
        status=imported_geometry__supermesh.CommonRefinementStatus.SUCCESS,
        reason=evidence.reason,
    )


def _prepare_mapped_surface_chart_remap_state(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    source_geometry: CellGeometrySpec | None,
    target_geometry: CellGeometrySpec | None,
    deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation,
    /,
    *,
    provenance: str,
    budget: CoordinateEnclosureBudget,
    policy: CommonRefinementPolicy,
) -> PreparedUnstructuredConservativeRemap:

    if source_geometry is None or target_geometry is None:
        raise TypeError(
            "prepare_unstructured_conservative_remap.cell_geometry must specify both chart endpoint maps; got a missing source or target geometry."
        )
    sphere_validation = None
    match deformation:
        case PreparedSphereChartDeformation():
            sphere_validation = deformation.require_bound(
                source.mesh, source_geometry, target.mesh, target_geometry, budget=budget
            )
        case PreparedSurfaceChartDeformation():
            deformation.require_bound(
                source.mesh, source_geometry, target.mesh, target_geometry
            )
        case _:
            assert_never(deformation)
    ledger = budget
    initial_work = ledger.work_units
    with (
        ledger.activate(),
        ledger.measurement(),
        ledger.bound_stage(policy.maximum_exact_work, policy.maximum_memory_bytes),
    ):
        match deformation:
            case PreparedSphereChartDeformation():
                piece_count = len(deformation.pieces)
            case PreparedSurfaceChartDeformation():
                piece_count = 0
                for occurrence in deformation.occurrences:
                    ledger.reserve(1)
                    piece_count += len(occurrence.pieces)
            case _:
                assert_never(deformation)
        if piece_count > min(
            policy.maximum_candidate_pairs, policy.maximum_accepted_pairs
        ):
            raise ValueError(
                "prepare_unstructured_conservative_remap.material_pairs must fit the candidate and accepted pair limits; got excess material pieces."
            )
        # CPython scalar/list slots and retained/working numeric content/CSR
        # arrays. This is not a compiler/device/native-managed memory claim.
        ledger.reserve(
            0, 1024 + 512 * piece_count + 256 * (source.cell_count + target.cell_count)
        )
        match deformation:
            case PreparedSphereChartDeformation():
                contents = imported_fem__sphere_chart_transfer.prepare_sphere_chart_finite_volume_contents(
                    source,
                    target,
                    deformation,
                    source_geometry=source_geometry,
                    target_geometry=target_geometry,
                    budget=budget,
                    validation=sphere_validation,
                    maximum_work=min(
                        policy.maximum_exact_work,
                        ledger.maximum_work_units - ledger.work_units,
                    ),
                )
            case PreparedSurfaceChartDeformation():
                contents = imported_fem__surface_chart_transfer.prepare_surface_chart_finite_volume_contents(
                    source,
                    target,
                    deformation,
                    source_geometry=source_geometry,
                    target_geometry=target_geometry,
                    budget=budget,
                    maximum_work=min(
                        policy.maximum_exact_work,
                        ledger.maximum_work_units - ledger.work_units,
                    ),
                )
            case _:
                assert_never(deformation)
        rows = np.asarray(contents.source_rows, dtype=np.int32)
        targets = np.asarray(contents.target_rows, dtype=np.int32)
        order = np.lexsort((rows, targets))
        offsets = np.concatenate(
            ([0], np.cumsum(np.bincount(targets, minlength=target.cell_count)))
        ).astype(np.int32)
        plan = UnstructuredConservativeRemapPlan(
            source,
            target,
            offsets,
            rows[order],
            np.asarray(contents.source_contents)[order],
            method="mapped-surface-chart",
            provenance=provenance,
            tolerance=policy.coverage_tolerance,
            require_complete=True,
            intersection_error_bounds=np.asarray(contents.source_content_errors)[order],
            surface_chart_deformation=deformation,
            surface_chart_contents=contents,
            budget=budget,
        )
        evidence = MappedSurfaceChartRemapEvidence(
            deformation=deformation,
            physical_contents=contents,
            report=plan.report,
            execution_evidence=None,
            logical_exact_work_units=jnp.asarray(
                ledger.work_units - initial_work, dtype=jnp.uint64
            ),
            host_storage_upper_bytes=jnp.asarray(
                ledger.peak_bytes_upper, dtype=jnp.uint64
            ),
            reason="complete material-reference coverage with certified old physical contents and actual new physical areas",
        )

        return PreparedUnstructuredConservativeRemap(
            evidence,
            plan,
            status=imported_geometry__supermesh.CommonRefinementStatus.SUCCESS,
            reason=evidence.reason,
        )


__all__ = [
    "prepare_unstructured_conservative_remap",
]
