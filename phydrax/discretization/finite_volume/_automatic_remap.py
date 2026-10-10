#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical conservative finite-volume remap preparation.

Exact reference tilings of represented coordinate maps use certified finer-cell
measures directly. Bounded surface chart deformation uses actual native UV
coverage with certified old physical contents and actual new physical areas.
Corner geometry uses the geometric common-refinement owner. The routes retain
distinct scientific evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._cell_geometry import CellGeometrySpec
from .._cell_geometry_validity import cell_geometry_id
from .._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from .._sphere_chart_deformation import PreparedSphereChartDeformation
from .._surface_chart_deformation import PreparedSurfaceChartDeformation
from ._remap_evidence import (
    MappedNestedRemapEvidence,
    MappedSurfaceChartRemapEvidence,
    PreparedUnstructuredConservativeRemap,
    RemapPreparationFailure,
)
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_remap import _coverage_ledger, UnstructuredConservativeRemapPlan


if TYPE_CHECKING:
    from ..._meshcore import NativeExecutionBudget, NativeHostStorageWorkspace
    from ...geometry._supermesh import CommonRefinementPolicy
    from ...meshing._adaptation import MeshAdaptationResult
    from ...meshing._measurements import NativeExecutionRecord
    from .._cell_geometry_transfer import CellGeometryTransition


def prepare_unstructured_conservative_remap(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    /,
    *,
    provenance: str,
    policy: CommonRefinementPolicy | None = None,
    source_geometry: CellGeometrySpec | None = None,
    target_geometry: CellGeometrySpec | None = None,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    surface_chart_deformation: PreparedSurfaceChartDeformation
    | PreparedSphereChartDeformation
    | None = None,
    adaptation: MeshAdaptationResult | None = None,
    source_workspace: NativeHostStorageWorkspace | None = None,
) -> PreparedUnstructuredConservativeRemap:
    """Prepare actual remap contents within the preceding adaptation allowance.

    Standalone remaps retain their authored common-refinement policy. A supplied
    adaptation binds the actual source/target meshes and its ended native record;
    work, queries and elapsed time are never refilled for the consumer phase.
    """
    from ..._meshcore import (
        current_native_execution_budget,
        MeshcoreError,
        MeshcoreStatus,
    )
    from ...geometry._supermesh import CommonRefinementPolicy, CommonRefinementStatus
    from ...meshing._adaptation import MeshAdaptationResult
    from ...meshing._contracts import MeshingFailure, MeshingFailureCategory
    from ...meshing._measurements import NativeExecutionRecord
    from ...meshing._tetra_metric import MetricRemeshingEvidence
    from ...meshing._volume_generation import native_volume_source_execution_budget
    from .._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    if adaptation is None:
        return _prepare_unstructured_conservative_remap_state(
            source,
            target,
            provenance=provenance,
            policy=policy,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            geometry_transition=geometry_transition,
            surface_chart_deformation=surface_chart_deformation,
        )
    if not isinstance(adaptation, MeshAdaptationResult):
        raise TypeError("adaptation must be the actual MeshAdaptationResult or None.")
    if not isinstance(source, UnstructuredFiniteVolumeDiscretization) or not isinstance(
        target, UnstructuredFiniteVolumeDiscretization
    ):
        raise TypeError("Remap endpoints must be actual unstructured FV discretizations.")
    if (source.mesh.mesh_id, target.mesh.mesh_id) != (
        adaptation.source.mesh.mesh_id,
        adaptation.target.mesh.mesh_id,
    ):
        raise ValueError("The preceding adaptation does not bind these actual FV meshes.")
    if (
        not isinstance(adaptation.evidence, MetricRemeshingEvidence)
        or adaptation.evidence.execution_evidence is None
    ):
        raise ValueError(
            "The preceding remesh lacks its actual ended native execution record."
        )
    previous = adaptation.evidence.execution_evidence
    previous.require_valid()
    owner_id = previous.owner_id
    if owner_id is None:
        raise ValueError("The preceding remesh lacks its actual source owner identity.")
    if int(np.asarray(previous.status)) != 0:
        raise ValueError(
            "A failed preceding native mesh scope cannot prepare a consumer remap."
        )
    policy_ = CommonRefinementPolicy() if policy is None else policy
    if not isinstance(policy_, CommonRefinementPolicy):
        raise TypeError("policy must be a CommonRefinementPolicy.")
    limits = adaptation.policy.limits
    remaining_work = limits.maximum_work_units - int(
        np.asarray(previous.total_work_units)
    )
    remaining_queries = limits.maximum_geometry_queries - int(
        np.asarray(previous.total_geometry_queries)
    )
    remaining_seconds = limits.maximum_wall_seconds - float(
        np.asarray(previous.total_elapsed_seconds)
    )
    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        ledger = CoordinateEnclosureBudget(
            max(0, min(remaining_work, policy_.maximum_exact_work)),
            min(limits.maximum_scratch_bytes, policy_.maximum_memory_bytes),
        )
    initial_work = ledger.work_units

    def refused(
        record: NativeExecutionRecord, reason: str
    ) -> PreparedUnstructuredConservativeRemap:
        failure = RemapPreparationFailure(
            source_discretization_id=source.prepared_id,
            target_discretization_id=target.prepared_id,
            execution_evidence=record,
            logical_exact_work_units=jnp.asarray(
                ledger.work_units - initial_work, dtype=jnp.uint64
            ),
            host_storage_upper_bytes=jnp.asarray(
                ledger.peak_bytes_upper, dtype=jnp.uint64
            ),
            reason=reason,
        )
        return PreparedUnstructuredConservativeRemap(
            None,
            None,
            status=CommonRefinementStatus.RESOURCE_LIMIT,
            reason=reason,
            preparation_failure=failure,
            execution_evidence=record,
        )

    if remaining_work <= 0 or remaining_queries < 0 or remaining_seconds <= 0:
        return refused(
            previous,
            "The actual preceding mesh scope exhausted the original consumer allowance.",
        )
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
    active = current_native_execution_budget()
    ancestor = active
    accounted = False
    while ancestor is not None:
        if id(previous) in ancestor._imported_preparation_receipts:
            accounted = True
            break
        ancestor = ancestor._parent
    if source_workspace is not None and source_workspace is ledger._host_workspace:
        raise ValueError(
            "FV source owners require a persistent reservation separate from algebra scratch."
        )
    if (
        active is not None
        and _COORDINATE_BUDGET.get() is not None
        and source_workspace is not None
        and accounted
    ):
        workspace = source_workspace
        workspace.retain_owner(
            (source, target, source_geometry, target_geometry, surface_chart_deformation)
        )
        allowance = active.remaining()
        debt = ledger.work_units - ledger.native_charged_work_units
        with ledger.bound_stage(
            max(
                0, min(allowance.remaining_work_units - debt, policy_.maximum_exact_work)
            ),
            min(
                allowance.remaining_scratch_bytes
                + ledger.retained_basis_bytes
                + ledger.temporary_bytes_upper,
                policy_.maximum_memory_bytes,
            ),
        ):
            try:
                prepared = _prepare_unstructured_conservative_remap_state(
                    source,
                    target,
                    provenance=provenance,
                    policy=policy_,
                    source_geometry=source_geometry,
                    target_geometry=target_geometry,
                    parent_cells=parent_cells,
                    parent_reference_vertices=parent_reference_vertices,
                    geometry_transition=geometry_transition,
                    surface_chart_deformation=surface_chart_deformation,
                )
                workspace.retain_owner(prepared)
            except BaseException as error:
                try:
                    ledger.charge_native_work(
                        ledger.work_units - ledger.native_charged_work_units
                    )
                except BaseException as debit_error:
                    error.add_note(
                        f"Actual coordinate work debit also failed: {debit_error!r}"
                    )
                raise
            else:
                ledger.charge_native_work(
                    ledger.work_units - ledger.native_charged_work_units
                )
            return prepared
    prepared: PreparedUnstructuredConservativeRemap | None = None
    reason: str | None = None
    native: NativeExecutionBudget | None = None
    try:
        with (
            native_volume_source_execution_budget(limits, previous, owner_id) as native,
            native.host_workspace() as source_workspace,
            ledger.activate(),
        ):
            source_workspace.retain_owner(
                (
                    source,
                    target,
                    adaptation.source,
                    adaptation.target,
                    source_geometry,
                    target_geometry,
                    surface_chart_deformation,
                    previous,
                )
            )
            allowance = native.remaining()
            debt = ledger.work_units - ledger.native_charged_work_units
            with ledger.bound_stage(
                max(
                    0,
                    min(
                        allowance.remaining_work_units - debt, policy_.maximum_exact_work
                    ),
                ),
                min(
                    allowance.remaining_scratch_bytes
                    + ledger.retained_basis_bytes
                    + ledger.temporary_bytes_upper,
                    policy_.maximum_memory_bytes,
                ),
            ):
                try:
                    prepared = _prepare_unstructured_conservative_remap_state(
                        source,
                        target,
                        provenance=provenance,
                        policy=policy_,
                        source_geometry=source_geometry,
                        target_geometry=target_geometry,
                        parent_cells=parent_cells,
                        parent_reference_vertices=parent_reference_vertices,
                        geometry_transition=geometry_transition,
                        surface_chart_deformation=surface_chart_deformation,
                    )
                    source_workspace.retain_owner(prepared)
                except BaseException as error:
                    try:
                        ledger.charge_native_work(
                            ledger.work_units - ledger.native_charged_work_units
                        )
                    except BaseException as debit_error:
                        error.add_note(
                            f"Actual coordinate work debit also failed: {debit_error!r}"
                        )
                    raise
                else:
                    ledger.charge_native_work(
                        ledger.work_units - ledger.native_charged_work_units
                    )
    except CoordinateEnclosureResourceError as error:
        if native is None:
            raise
        reason = str(error)
    except MeshcoreError as error:
        if native is None:
            raise
        if error.status not in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT):
            raise
        reason = str(error)
    except MeshingFailure as error:
        if native is None or error.category not in (
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            MeshingFailureCategory.TIMED_OUT,
        ):
            raise
        reason = str(error)
    if native is None or native.evidence is None:
        raise RuntimeError(
            "A native FV preparation lacks its actual ended execution record."
        )
    record = NativeExecutionRecord(
        native.evidence, owner_id=previous.owner_id, preparation_evidence=previous
    )
    if reason is not None:
        return refused(record, reason)
    if prepared is None:
        raise RuntimeError(
            "A native FV scope ended without a remap or an actual refusal."
        )
    return _bind_remap_execution(prepared, record)


def _bind_remap_execution(
    prepared: PreparedUnstructuredConservativeRemap,
    record: NativeExecutionRecord,
    /,
) -> PreparedUnstructuredConservativeRemap:
    """Bind the actual immutable owner only after its whole execution ends."""
    chart_evidence = prepared.surface_chart_evidence
    if chart_evidence is not None:
        chart_evidence = eqx.tree_at(
            lambda value: value.execution_evidence,
            chart_evidence,
            record,
            is_leaf=lambda value: value is None,
        )
    failure = prepared.preparation_failure
    if failure is not None:
        failure = eqx.tree_at(
            lambda value: value.execution_evidence,
            failure,
            record,
            is_leaf=lambda value: value is None,
        )
    return PreparedUnstructuredConservativeRemap(
        prepared.refinement,
        prepared.plan,
        status=prepared.status,
        reason=prepared.reason,
        nested_evidence=prepared.nested_evidence,
        surface_chart_evidence=chart_evidence,
        preparation_failure=failure,
        execution_evidence=record,
    )


def _prepare_unstructured_conservative_remap_state(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    /,
    *,
    provenance: str,
    policy: CommonRefinementPolicy | None = None,
    source_geometry: CellGeometrySpec | None = None,
    target_geometry: CellGeometrySpec | None = None,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    surface_chart_deformation: PreparedSurfaceChartDeformation
    | PreparedSphereChartDeformation
    | None = None,
) -> PreparedUnstructuredConservativeRemap:
    """Prepare the conservative remap of ``source`` cells onto ``target`` cells.

    With actual ``source_geometry`` and ``target_geometry``, both prepared FV
    discretizations must have been prepared using those ``cell_geometry`` specs.
    ``geometry_transition`` supplies bound refinement/coarsening witnesses;
    explicit ``parent_cells`` and ``parent_reference_vertices`` may instead
    supply refinement witnesses. Invalid or stale witnesses raise before any
    plan is published. This route always proves complete reference coverage.

    ``surface_chart_deformation`` supplies an actual bounded chart core directly;
    alternatively ``geometry_transition.chart_deformation`` carries that core.
    The canonical cell-average/extensive route is conservative density: old
    physical content is preserved while actual new physical areas determine
    target averages. It does not claim constants or density bounds preserved.

    The discretizations' cell meshes are intersected by
    :func:`phydrax.geometry.prepare_common_refinement` under ``policy`` (the
    default policy when ``None``).  ``COMPLETE`` coverage yields a plan requiring
    complete coverage with the policy's relative ``coverage_tolerance``; other
    coverage modes yield a plan that reports, but does not require, coverage.
    Common-refinement geometric/resource failures return a failed status without
    a plan; malformed scientific inputs and nested witnesses raise.
    """
    from ...geometry._supermesh import (
        CommonRefinementCoverage,
        CommonRefinementPolicy,
        CommonRefinementStatus,
        prepare_common_refinement,
    )

    if not isinstance(source, UnstructuredFiniteVolumeDiscretization) or not isinstance(
        target, UnstructuredFiniteVolumeDiscretization
    ):
        raise TypeError("Remap source and target must be unstructured FV geometry.")
    if not isinstance(provenance, str):
        raise TypeError("provenance must be a string.")
    if not provenance:
        raise ValueError("provenance must be non-empty.")
    policy_ = CommonRefinementPolicy() if policy is None else policy
    if not isinstance(policy_, CommonRefinementPolicy):
        raise TypeError("policy must be a CommonRefinementPolicy.")
    chart = surface_chart_deformation
    if geometry_transition is not None:
        from .._cell_geometry_transfer import CellGeometryTransition

        if not isinstance(geometry_transition, CellGeometryTransition):
            raise TypeError("geometry_transition must be a CellGeometryTransition.")
        carried = geometry_transition.chart_deformation
        if carried is not None:
            if (
                geometry_transition.source_topology_id != carried.source_topology_id
                or geometry_transition.target_topology_id != carried.target_topology_id
                or geometry_transition.source_geometry_id != carried.source_geometry_id
                or geometry_transition.target_geometry_id != carried.target_geometry_id
                or cell_geometry_id(geometry_transition.geometry)
                != carried.target_geometry_id
            ):
                raise ValueError(
                    "Chart geometry transition is stale for its actual deformation."
                )
            if chart is not None and chart.deformation_id != carried.deformation_id:
                raise ValueError(
                    "Explicit and carried surface chart deformations disagree."
                )
            chart = carried
    if chart is not None:
        if parent_cells is not None or parent_reference_vertices is not None:
            raise ValueError(
                "Bounded chart deformation is not an exact nested reference transfer."
            )
        return _prepare_mapped_surface_chart_remap_state(
            source,
            target,
            source_geometry,
            target_geometry,
            chart,
            provenance=provenance,
            policy=policy_,
        )
    mapped = source_geometry is not None or target_geometry is not None
    # Only power sources own the exact polyhedral common refinement; PLC sources
    # take the mapped route through their exact source coordinates.
    exact_power = any(
        geometry is not None
        and any(element.cell_kind == "polyhedron" for element in geometry.elements)
        and isinstance(
            geometry.exact_source,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        )
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
                    "Exact power remap requires the actual source and target geometry used to prepare FV measures."
                )
    witnesses = (
        parent_cells is not None
        or parent_reference_vertices is not None
        or geometry_transition is not None
    )
    if witnesses and not exact_power:
        if not isinstance(source_geometry, CellGeometrySpec) or not isinstance(
            target_geometry, CellGeometrySpec
        ):
            raise TypeError(
                "Mapped nested remap requires actual source and target CellGeometrySpec."
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
            geometry_transition=geometry_transition,
        )
    if mapped and not exact_power:
        for discretization, geometry in (
            (source, source_geometry),
            (target, target_geometry),
        ):
            if not isinstance(geometry, CellGeometrySpec):
                raise TypeError(
                    "Nonnested FV remap requires both actual coordinate geometries."
                )
            affine = CellGeometrySpec.affine(discretization.mesh)
            if cell_geometry_id(geometry) != cell_geometry_id(
                affine
            ) or array_tree_fingerprint(geometry) != array_tree_fingerprint(affine):
                raise ValueError(
                    "Nonnested FV remap requires complete canonical affine geometry; curved maps need their own witnesses."
                )
            prepared_geometry = discretization.cell_geometry
            if prepared_geometry is not None and (
                cell_geometry_id(prepared_geometry) != cell_geometry_id(geometry)
                or array_tree_fingerprint(prepared_geometry)
                != array_tree_fingerprint(geometry)
            ):
                raise ValueError(
                    "Supplied affine geometry differs from the actual prepared FV coordinate banks."
                )
    elif not exact_power and (
        source.cell_geometry is not None or target.cell_geometry is not None
    ):
        raise ValueError(
            "Mapped FV remap requires actual coordinate geometry and nested witnesses."
        )
    refinement = prepare_common_refinement(
        source.mesh,
        target.mesh,
        policy=policy_,
        source_geometry=source_geometry if exact_power else None,
        target_geometry=target_geometry if exact_power else None,
    )
    if not refinement.succeeded:
        return PreparedUnstructuredConservativeRemap(
            refinement,
            None,
            status=refinement.status,
            reason=f"{refinement.status.name}: {refinement.evidence.reason}",
        )
    require_complete = policy_.coverage is CommonRefinementCoverage.COMPLETE
    if require_complete:
        target_defect, source_defect, target_limit, source_limit = _coverage_ledger(
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
        if np.any(target_defect > target_limit) or np.any(source_defect > source_limit):
            return PreparedUnstructuredConservativeRemap(
                refinement,
                None,
                status=CommonRefinementStatus.DOUBLE_COVERAGE,
                reason="certified overlap measures exceed finite-volume cell volumes",
            )
        if np.any(target_defect < -target_limit) or np.any(source_defect < -source_limit):
            return PreparedUnstructuredConservativeRemap(
                refinement,
                None,
                status=CommonRefinementStatus.COVERAGE_GAP,
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
        status=CommonRefinementStatus.SUCCESS,
        reason="certified complete common refinement"
        if require_complete
        else f"certified {policy_.coverage.value} common refinement",
    )


def _prepare_mapped_nested_remap(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    /,
    *,
    provenance: str,
    policy: CommonRefinementPolicy,
    parent_cells: ArrayLike | None,
    parent_reference_vertices: ArrayLike | None,
    geometry_transition: CellGeometryTransition | None,
) -> PreparedUnstructuredConservativeRemap:
    from ...geometry._supermesh import CommonRefinementStatus
    from .._cell_geometry import CellGeometrySpec
    from .._cell_geometry_transfer import (
        _certified_cell_measures,
        _certify_nested_geometry_pairs,
    )
    from .._nested_reference import _nested_reference_pairs

    if not isinstance(source_geometry, CellGeometrySpec) or not isinstance(
        target_geometry, CellGeometrySpec
    ):
        raise TypeError(
            "Mapped nested remap requires actual source and target CellGeometrySpec."
        )
    for discretization, geometry in (
        (source, source_geometry),
        (target, target_geometry),
    ):
        if discretization.cell_geometry is None or cell_geometry_id(
            discretization.cell_geometry
        ) != cell_geometry_id(geometry):
            raise ValueError(
                "Mapped FV geometry is not bound to the prepared cell volumes."
            )
        measures, errors, _ = _certified_cell_measures(discretization.mesh, geometry)
        if not np.array_equal(
            np.asarray(discretization.cell_volumes), measures
        ) or not np.array_equal(
            np.asarray(discretization.cell_volume_error_bounds), errors
        ):
            raise ValueError("Prepared FV mapped measures are stale or inconsistent.")
    pairs = _nested_reference_pairs(
        source.mesh,
        source_geometry,
        target.mesh,
        target_geometry,
        parent_cells=parent_cells,
        parent_reference_vertices=parent_reference_vertices,
        geometry_transition=geometry_transition,
    )
    geometry_error = _certify_nested_geometry_pairs(
        source.mesh,
        source_geometry,
        target.mesh,
        target_geometry,
        pairs,
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
            "pairs": [
                (
                    p.source_cell,
                    p.target_cell,
                    p.fine_is_source,
                    array_tree_fingerprint(p.matrix),
                    array_tree_fingerprint(p.offset),
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
        None,
        plan,
        status=CommonRefinementStatus.SUCCESS,
        reason=evidence.reason,
        nested_evidence=evidence,
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
    policy: CommonRefinementPolicy,
) -> PreparedUnstructuredConservativeRemap:
    from ...geometry._supermesh import CommonRefinementStatus
    from .._coordinate_enclosure import _COORDINATE_BUDGET, CoordinateEnclosureBudget
    from ..fem._sphere_chart_transfer import prepare_sphere_chart_finite_volume_contents
    from ..fem._surface_chart_transfer import prepare_surface_chart_finite_volume_contents

    if not isinstance(
        deformation, (PreparedSurfaceChartDeformation, PreparedSphereChartDeformation)
    ):
        raise TypeError("An actual prepared surface chart deformation is required.")
    if not isinstance(source_geometry, CellGeometrySpec) or not isinstance(
        target_geometry, CellGeometrySpec
    ):
        raise TypeError(
            "Surface chart remap needs actual source and target coordinate geometry."
        )
    deformation.require_bound(source.mesh, source_geometry, target.mesh, target_geometry)
    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        ledger = CoordinateEnclosureBudget(
            policy.maximum_exact_work, policy.maximum_memory_bytes
        )
    initial_work = ledger.work_units
    with (
        ledger.activate(),
        ledger.bound_stage(policy.maximum_exact_work, policy.maximum_memory_bytes),
    ):
        if isinstance(deformation, PreparedSphereChartDeformation):
            piece_count = len(deformation.pieces)
        else:
            piece_count = 0
            for occurrence in deformation.occurrences:
                ledger.reserve(1)
                piece_count += len(occurrence.pieces)
        if piece_count > min(
            policy.maximum_candidate_pairs, policy.maximum_accepted_pairs
        ):
            raise ValueError("Material overlaps exceed the original pair allowance.")
        # CPython scalar/list slots and retained/working numeric content/CSR
        # arrays. This is not a compiler/device/native-managed memory claim.
        ledger.reserve(
            0, 1024 + 512 * piece_count + 256 * (source.cell_count + target.cell_count)
        )
        if isinstance(deformation, PreparedSphereChartDeformation):
            contents = prepare_sphere_chart_finite_volume_contents(
                source,
                target,
                deformation,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                maximum_work=min(
                    policy.maximum_exact_work,
                    ledger.maximum_work_units - ledger.work_units,
                ),
            )
        else:
            contents = prepare_surface_chart_finite_volume_contents(
                source,
                target,
                deformation,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                maximum_work=min(
                    policy.maximum_exact_work,
                    ledger.maximum_work_units - ledger.work_units,
                ),
            )
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
        ledger.charge_native_work(ledger.work_units - ledger.native_charged_work_units)
        return PreparedUnstructuredConservativeRemap(
            None,
            plan,
            status=CommonRefinementStatus.SUCCESS,
            reason=evidence.reason,
            surface_chart_evidence=evidence,
        )


__all__ = [
    "prepare_unstructured_conservative_remap",
]
