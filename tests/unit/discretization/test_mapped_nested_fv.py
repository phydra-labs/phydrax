#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellGeometryTransition,
    CellMesh,
    FiniteVolumePrecisionPolicy,
    lower_static_unstructured_stage_metrics,
    RestrictedCellGeometryElement,
    TopologyEpoch,
    transition_nested_cell_geometry,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._mapped_geometry_measure import certified_cell_measures
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._surface_chart_records import PreparedSurfaceChartDeformation
from phydrax.discretization.fem import FiniteElementSpec
from phydrax.discretization.fem._surface_chart_transfer import (
    PreparedSurfaceChartFiniteVolumeContents,
)
from phydrax.discretization.finite_volume import (
    FiniteVolumeStageMetrics,
    MappedNestedRemapEvidence,
    MappedSurfaceChartRemapEvidence,
    prepare_unstructured_conservative_remap,
    PreparedUnstructuredConservativeRemap,
    UnstructuredConservativeRemapPlan,
    UnstructuredFiniteVolumeDiscretization,
)
from phydrax.geometry import CommonRefinementPolicy
from phydrax.lifecycle import CompositionEntry
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh, MixedAdaptationOutcome
from phydrax.meshing._topology_edit import assemble_topology_edit
from phydrax.solver import FiniteVolumeConservativeContentState
from phydrax.solver._finite_volume_topology_events import (
    FiniteVolumeTopologyArtifacts,
    FiniteVolumeTopologyEventJournal,
    FiniteVolumeTopologyEventRequest,
    FiniteVolumeTopologyEventScheduler,
    FiniteVolumeTopologyEventTransactionResult,
    TopologyEventKind,
    TopologyEventStatus,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mapped_cells import (
    curved_fv_volume,
    curved_surface_chart_fv,
    fv_volume_map,
)


# Qualification thresholds retained from the pre-remediation curved FV tests:
# 2e-14 for polynomial measures and chart contents, 3e-14 for refinement
# content/duality, and 4e-14 for many-to-one coarsening. Reference totals integrate
# det(D map) = 1 + 0.1*y over each declared reference cell. Values are unchanged.
MAPPED_MEASURE_ATOL = 2.0e-14
REFINEMENT_ATOL = 3.0e-14
COARSENING_ATOL = 4.0e-14


def _require_plan(
    remap: PreparedUnstructuredConservativeRemap,
) -> UnstructuredConservativeRemapPlan:
    if remap.plan is None:
        raise ValueError(
            f"remap.plan must contain a successful plan; got {remap.status}: {remap.reason}"
        )
    return remap.plan


def _require_chart_evidence(
    remap: PreparedUnstructuredConservativeRemap,
) -> MappedSurfaceChartRemapEvidence:
    evidence = remap.evidence
    match evidence:
        case MappedSurfaceChartRemapEvidence():
            return evidence
        case _:
            raise TypeError(
                "remap.evidence must contain mapped surface-chart evidence; got another evidence kind."
            )


def _require_chart_contents(
    plan: UnstructuredConservativeRemapPlan,
) -> PreparedSurfaceChartFiniteVolumeContents:
    contents = plan.surface_chart_contents
    match contents:
        case PreparedSurfaceChartFiniteVolumeContents():
            return contents
        case _:
            raise TypeError(
                "remap.surface_chart_contents must contain mapped surface-chart contents; got another contents kind."
            )


def _prepare(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> UnstructuredFiniteVolumeDiscretization:
    return UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare(
        cell_geometry=geometry
    )


def _refine(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> tuple[CellMesh, CellGeometryTransition, MixedAdaptationOutcome]:
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=ids)
    with coordinate_reference_budget() as budget:
        target, _, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version="fine", budget=budget
        )
    transition = transition_nested_cell_geometry(
        mesh,
        geometry,
        target,
        CellGeometrySpec.affine(target),
        refinement=outcome.edit.refinement,
    )
    return target, transition, outcome


class _NestedFVCase(NamedTuple):
    geometry: CellGeometrySpec
    source: UnstructuredFiniteVolumeDiscretization
    fine_mesh: CellMesh
    transition: CellGeometryTransition
    outcome: MixedAdaptationOutcome
    fine: UnstructuredFiniteVolumeDiscretization
    remap: PreparedUnstructuredConservativeRemap


@pytest.fixture(
    scope="module",
    params=(
        ("hexahedron", 1.05),
        ("prism", 31 / 60),
        ("pyramid", 0.35),
        ("tetrahedron", 41 / 240),
        ("connected", 1.4),
    ),
    ids=("hexahedron", "prism", "pyramid", "tetrahedron", "connected"),
)
def nested_fv_case(request: pytest.FixtureRequest) -> _NestedFVCase:
    kind, total = request.param
    policy = CommonRefinementPolicy()
    mesh, geometry = curved_fv_volume(kind)
    source = _prepare(mesh, geometry)
    np.testing.assert_allclose(
        np.sum(source.cell_volumes), total, rtol=0, atol=MAPPED_MEASURE_ATOL
    )
    fine_mesh, transition, outcome = _refine(mesh, geometry)
    fine = _prepare(fine_mesh, transition.geometry)
    remap = prepare_unstructured_conservative_remap(
        source,
        fine,
        provenance="curved-refine",
        policy=policy,
        source_geometry=geometry,
        target_geometry=transition.geometry,
        geometry_transition=transition,
    )
    return _NestedFVCase(geometry, source, fine_mesh, transition, outcome, fine, remap)


def test_curved_nested_refinement_inventory_epoch_and_transpose(
    nested_fv_case: _NestedFVCase,
) -> None:
    _geometry, source, fine_mesh, transition, _outcome, fine, remap = nested_fv_case
    policy = CommonRefinementPolicy()
    assert isinstance(remap.route, MappedNestedRemapEvidence)
    values = jnp.arange(source.cell_count, dtype=jnp.float64)[:, None] + 2
    content = source.cell_volumes[:, None] * values
    transferred = _require_plan(remap).apply(values)
    transferred_content = _require_plan(remap).apply_content(content)
    source_epoch = TopologyEpoch(0, source.geometry_id, source.topology_id, "serial")
    fine_epoch = TopologyEpoch(1, fine.geometry_id, fine.topology_id, "serial")
    epoch_transfer = _require_plan(remap).epoch_transition(
        source.cell_space, fine.cell_space, source_epoch, fine_epoch
    )
    with pytest.raises(ValueError, match="geometry"):
        _require_plan(remap).epoch_transition(
            source.cell_space,
            fine.cell_space,
            source_epoch,
            TopologyEpoch(1, "stale-geometry", fine.topology_id, "serial"),
        )
    source_entry = CompositionEntry(
        values,
        entry_id="fv-state",
        role="physical-state",
        owner_id="finite-volume",
        structure_id=source_epoch.epoch_id,
        revision_id="accepted",
        semantics_id="cell-average",
    )
    target_entry = CompositionEntry(
        transferred,
        entry_id="fv-state",
        role="physical-state",
        owner_id="finite-volume",
        structure_id=fine_epoch.epoch_id,
        revision_id="refined",
        semantics_id="cell-average",
    )
    transport = epoch_transfer.composition_transport(source_entry, target_entry)
    assert bool(transport.successful)
    np.testing.assert_allclose(
        np.asarray(transport.target_content),
        np.asarray(transport.source_content),
        rtol=0,
        atol=REFINEMENT_ATOL,
    )
    corrupt_entry = CompositionEntry(
        transferred + 1,
        entry_id="fv-state",
        role="physical-state",
        owner_id="finite-volume",
        structure_id=fine_epoch.epoch_id,
        revision_id="wrong",
        semantics_id="cell-average",
    )
    assert not bool(
        epoch_transfer.composition_transport(source_entry, corrupt_entry).successful
    )
    np.testing.assert_allclose(
        transferred_content,
        fine.cell_volumes[:, None] * transferred,
        rtol=0,
        atol=REFINEMENT_ATOL,
    )
    np.testing.assert_allclose(
        np.sum(transferred_content), np.sum(content), rtol=0, atol=REFINEMENT_ATOL
    )
    probe = jnp.sin(jnp.arange(fine.cell_count, dtype=jnp.float64))[:, None]
    np.testing.assert_allclose(
        jnp.vdot(transferred, probe),
        jnp.vdot(values, epoch_transfer.pullback(probe)),
        rtol=0,
        atol=REFINEMENT_ATOL,
    )
    ticket = HostBatchTicket(
        policy.maximum_exact_work,
        queries_upper=(policy.maximum_exact_work + 1)
        * cell_geometry_preparation_query_upper(transition.geometry),
        bytes_upper=policy.maximum_memory_bytes,
    )
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=policy.maximum_exact_work,
        maximum_memory_bytes=policy.maximum_memory_bytes,
    )
    with ticket, budget.activate():
        measured, errors, _ = certified_cell_measures(
            fine_mesh, transition.geometry, budget=budget
        )
    np.testing.assert_array_equal(fine.cell_volumes, measured)
    np.testing.assert_array_equal(fine.cell_volume_error_bounds, errors)
    np.testing.assert_array_equal(_require_plan(remap).target_volumes, measured)


def test_curved_nested_coarsening_inventory_and_transpose(
    nested_fv_case: _NestedFVCase,
) -> None:
    _geometry, source, fine_mesh, transition, outcome, fine, remap = nested_fv_case
    values = jnp.arange(source.cell_count, dtype=jnp.float64)[:, None] + 2
    content = source.cell_volumes[:, None] * values
    transferred_content = _require_plan(remap).apply_content(content)
    ids = np.concatenate([np.asarray(block.global_ids) for block in fine_mesh.blocks])
    coarsened = adapt_mixed_mesh(
        fine_mesh,
        refine_cell_ids=np.asarray([], dtype=np.int64),
        coarsen_cell_ids=ids,
        hierarchy=outcome.hierarchy,
    )
    with coordinate_reference_budget() as budget:
        restored_mesh, _, _ = assemble_topology_edit(
            fine_mesh, coarsened.edit, numeric_version="restored", budget=budget
        )
    restoration = transition_nested_cell_geometry(
        fine_mesh,
        transition.geometry,
        restored_mesh,
        CellGeometrySpec.affine(restored_mesh),
        refinement=coarsened.edit.refinement,
        coarsening=coarsened.edit.coarsening,
    )
    restored = _prepare(restored_mesh, restoration.geometry)
    reverse = prepare_unstructured_conservative_remap(
        fine,
        restored,
        provenance="curved-coarsen",
        source_geometry=transition.geometry,
        target_geometry=restoration.geometry,
        geometry_transition=restoration,
    )
    reverse_epoch = _require_plan(reverse).epoch_transition(
        fine.cell_space,
        restored.cell_space,
        TopologyEpoch(1, fine.geometry_id, fine.topology_id, "serial"),
        TopologyEpoch(2, restored.geometry_id, restored.topology_id, "serial"),
    )
    # Nonconstant fine inventory exercises true many-to-one weighted aggregation.
    fine_values = jnp.arange(fine.cell_count, dtype=jnp.float64)[:, None] / 7 + 1
    fine_content = fine_values * fine.cell_volumes[:, None]
    coarse_content = _require_plan(reverse).apply_content(fine_content)
    np.testing.assert_allclose(
        coarse_content,
        _require_plan(reverse).apply(fine_values) * restored.cell_volumes[:, None],
        rtol=0,
        atol=COARSENING_ATOL,
    )
    np.testing.assert_allclose(
        np.sum(coarse_content), np.sum(fine_content), rtol=0, atol=COARSENING_ATOL
    )
    coarse_probe = jnp.arange(restored.cell_count, dtype=jnp.float64)[:, None] + 0.7
    np.testing.assert_allclose(
        jnp.vdot(_require_plan(reverse).apply(fine_values), coarse_probe),
        jnp.vdot(fine_values, reverse_epoch.pullback(coarse_probe)),
        rtol=0,
        atol=COARSENING_ATOL,
    )
    source_rows = {
        int(cell): row for row, cell in enumerate(np.asarray(source.cell_global_ids))
    }
    restored_rows = np.asarray(
        [source_rows[int(cell)] for cell in np.asarray(restored.cell_global_ids)]
    )
    np.testing.assert_allclose(
        _require_plan(reverse).apply_content(transferred_content),
        content[restored_rows],
        rtol=0,
        atol=COARSENING_ATOL,
    )
    np.testing.assert_array_equal(
        restored.cell_volumes, source.cell_volumes[restored_rows]
    )


def _foreign_topology(transition: CellGeometryTransition) -> CellGeometryTransition:
    return replace(transition, source_topology_id="foreign")


def _missing_target_cell(transition: CellGeometryTransition) -> CellGeometryTransition:
    return eqx.tree_at(
        lambda x: (x.target_cell_ids, x.parent_cell_ids, x.parent_reference_vertices),
        transition,
        (
            transition.target_cell_ids[:-1],
            transition.parent_cell_ids[:-1],
            transition.parent_reference_vertices[:-1],
        ),
    )


@pytest.mark.parametrize(
    ("mutation", "field"),
    (
        (_foreign_topology, r"nested_reference\.geometry_transition"),
        (_missing_target_cell, r"nested_reference\.records"),
    ),
    ids=("foreign-topology", "missing-target-cell"),
)
def test_nested_remap_rejects_invalid_reference_transition(
    mutation: Callable[[CellGeometryTransition], CellGeometryTransition],
    field: str,
) -> None:
    mesh, geometry = curved_fv_volume("hexahedron")
    source = _prepare(mesh, geometry)
    fine_mesh, transition, _ = _refine(mesh, geometry)
    fine = _prepare(fine_mesh, transition.geometry)
    with pytest.raises(ValueError, match=field):
        prepare_unstructured_conservative_remap(
            source,
            fine,
            provenance="reject",
            source_geometry=geometry,
            target_geometry=transition.geometry,
            geometry_transition=mutation(transition),
        )


def test_nested_remap_rejects_different_coordinate_bank() -> None:
    mesh, geometry = curved_fv_volume("hexahedron")
    source = _prepare(mesh, geometry)
    fine_mesh, transition, _ = _refine(mesh, geometry)
    fine = _prepare(fine_mesh, transition.geometry)
    stale = CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        np.asarray(geometry.coordinates) + 0.01,
    )
    with pytest.raises(
        ValueError, match=r"prepare_unstructured_conservative_remap\.cell_geometry"
    ):
        prepare_unstructured_conservative_remap(
            source,
            fine,
            provenance="stale",
            source_geometry=stale,
            target_geometry=transition.geometry,
            geometry_transition=transition,
        )


def _represented_two_child_hex(
    translation: Sequence[float],
) -> tuple[
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumeDiscretization,
    PreparedUnstructuredConservativeRemap,
]:
    """Return an affine reference partition whose children retain the root map."""

    source_mesh, source_geometry = curved_fv_volume("hexahedron")
    shift = np.asarray(translation, dtype=np.float64)
    source_mesh = CellMesh(
        np.asarray(source_mesh.coordinates) + shift, source_mesh.blocks
    )
    source_geometry = CellGeometrySpec(
        {"volume": source_geometry.elements[0]},
        {"volume": source_geometry.geometry_dofs[0]},
        np.asarray(source_geometry.coordinates) + shift,
    )
    reference = np.asarray(reference_cell_topology("hexahedron").vertices)
    root = source_geometry.elements[0]
    if not isinstance(root, (FiniteElementSpec, RestrictedCellGeometryElement)):
        raise TypeError(
            f"fixture.root must be a FiniteElementSpec or RestrictedCellGeometryElement; got {type(root).__name__}"
        )
    matrix = np.diag([1.0, 0.5, 1.0])
    corners = [reference @ matrix.T + [0, offset, 0] for offset in (0.0, 0.5)]
    points, lookup, blocks, elements, routes = [], {}, [], {}, {}
    for child, vertices in enumerate(corners):
        indices = []
        for point in vertices:
            key = tuple(point)
            if key not in lookup:
                lookup[key] = len(points)
                points.append(point)
            indices.append(lookup[key])
        name = f"child{child}"
        blocks.append(
            CellBlock(
                name,
                "hexahedron",
                np.asarray(indices)[None],
                global_ids=np.asarray([100 + child]),
            )
        )
        elements[name] = RestrictedCellGeometryElement(
            root, "hexahedron", matrix, np.asarray([0, 0.5 * child, 0])
        )
        routes[name] = source_geometry.geometry_dofs[0]
    target_mesh = CellMesh(
        fv_volume_map(np.asarray(points, dtype=np.float64)) + shift, blocks
    )
    target_geometry = CellGeometrySpec(elements, routes, source_geometry.coordinates)
    source, target = (
        _prepare(source_mesh, source_geometry),
        _prepare(target_mesh, target_geometry),
    )
    remap = prepare_unstructured_conservative_remap(
        source,
        target,
        provenance="represented-two-child-reference-partition",
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        parent_cells=np.zeros(2, dtype=np.int32),
        parent_reference_vertices=np.asarray(corners),
    )
    return source, target, remap


def test_mapped_transaction_rolls_back_different_remap_geometry() -> None:

    source, target, remap = _represented_two_child_hex([0, 0, 0])
    _, _, stale_remap = _represented_two_child_hex([0.2, 0, 0])
    initial = TopologyEpoch(0, source.geometry_id, source.topology_id, "serial")
    successor = TopologyEpoch(1, target.geometry_id, target.topology_id, "serial")
    source_artifacts = FiniteVolumeTopologyArtifacts(initial, source.prepared_id)
    target_artifacts = FiniteVolumeTopologyArtifacts(successor, target.prepared_id)
    source_metrics = lower_static_unstructured_stage_metrics(
        source, topology_epoch_id=initial.epoch_id
    )
    target_metrics = lower_static_unstructured_stage_metrics(
        target, topology_epoch_id=successor.epoch_id
    )

    def content(
        values: Array,
        geometry: UnstructuredFiniteVolumeDiscretization,
        metrics: FiniteVolumeStageMetrics,
        epoch: TopologyEpoch,
        time: float,
    ) -> FiniteVolumeConservativeContentState:
        return FiniteVolumeConservativeContentState(
            values * geometry.cell_volumes[:, None],
            geometry.cell_volumes,
            jnp.ones(geometry.cell_count, dtype=jnp.bool_),
            time,
            topology_epoch_id=epoch.epoch_id,
            geometry_family_id=metrics.geometry_family_id,
            geometry_layout_id=metrics.geometry_layout_id,
            geometry_version=metrics.geometry_version,
            evidence_policy_id=metrics.evidence.policy_id,
            evidence_version=metrics.evidence.evidence_version,
            precision=FiniteVolumePrecisionPolicy("float64"),
        )

    accepted = content(jnp.asarray([[3.0]]), source, source_metrics, initial, 0.0)
    journal = FiniteVolumeTopologyEventJournal.allocate(
        initial, source_artifacts, capacity=2, time=0.0
    )

    def transact(
        prepared_remap: PreparedUnstructuredConservativeRemap,
    ) -> FiniteVolumeTopologyEventTransactionResult:
        scheduler = FiniteVolumeTopologyEventScheduler(journal)
        scheduler.submit(
            FiniteVolumeTopologyEventRequest(
                TopologyEventKind.REMESH, initial.epoch_id, "mapped-reference-partition"
            ),
            1,
            0.1,
        )
        return scheduler.transact(
            source_geometry=source,
            target_geometry=target,
            source_content=accepted,
            candidate_epoch=successor,
            candidate_artifacts=target_artifacts,
            remap=prepared_remap,
            coverage_tolerance=0.0,
            transfer=lambda state, plan: content(
                plan.apply(state.cell_average()), target, target_metrics, successor, 0.1
            ),
        )

    committed = transact(remap)
    assert committed.committed
    assert committed.receipt is not None
    assert committed.receipt.published
    np.testing.assert_allclose(
        committed.content_state.volume_integral(), 3.15, rtol=0, atol=MAPPED_MEASURE_ATOL
    )
    assert committed.journal.current_epoch_id == successor.epoch_id
    rejected = transact(stale_remap)
    assert (
        not rejected.committed
        and rejected.failure is TopologyEventStatus.FAILED_STALE_EPOCH
    )
    assert rejected.content_state is accepted
    assert rejected.journal.current_epoch_id == initial.epoch_id
    assert rejected.journal.epoch_table == journal.epoch_table
    assert rejected.journal.artifact_table == journal.artifact_table


class _ChartFVCase(NamedTuple):
    source: UnstructuredFiniteVolumeDiscretization
    target: UnstructuredFiniteVolumeDiscretization
    core: PreparedSurfaceChartDeformation
    carrier: CellGeometryTransition
    remap: PreparedUnstructuredConservativeRemap


@pytest.fixture(scope="module")
def chart_fv_case() -> _ChartFVCase:
    source, target, core, carrier = curved_surface_chart_fv()
    remap = prepare_unstructured_conservative_remap(
        source,
        target,
        provenance="actual-sphere-chart-density",
        source_geometry=core.source_geometry,
        target_geometry=core.target_geometry,
        geometry_transition=carrier,
    )
    return _ChartFVCase(source, target, core, carrier, remap)


def test_curved_chart_fv_density_content_and_transpose(
    chart_fv_case: _ChartFVCase,
) -> None:
    source, target, _core, _carrier, remap = chart_fv_case
    assert isinstance(remap.route, MappedSurfaceChartRemapEvidence)
    assert _require_chart_evidence(remap).passed
    assert np.sum(source.cell_volumes) < np.sum(target.cell_volumes)
    assert np.max(_require_plan(remap).apply(jnp.ones((2, 1)))) < 0.99
    density = jnp.asarray([[2.0], [3.0]])
    source_content = density * source.cell_volumes[:, None]
    target_density = _require_plan(remap).apply(density)
    target_content = _require_plan(remap).apply_content(source_content)
    np.testing.assert_allclose(
        target_content,
        target_density * target.cell_volumes[:, None],
        rtol=0,
        atol=MAPPED_MEASURE_ATOL,
    )
    content_bound = jnp.vdot(
        _require_chart_contents(_require_plan(remap)).source_content_defect_bounds,
        jnp.abs(density[:, 0]),
    )
    assert abs(np.sum(target_content) - np.sum(source_content)) <= content_bound
    np.testing.assert_array_equal(_require_plan(remap).source_cell_global_ids, [107, 13])
    np.testing.assert_array_equal(_require_plan(remap).target_cell_global_ids, [59, 211])
    transition = _require_plan(remap).epoch_transition(
        source.cell_space,
        target.cell_space,
        TopologyEpoch(0, source.geometry_id, source.topology_id, "serial"),
        TopologyEpoch(1, target.geometry_id, target.topology_id, "serial"),
    )
    probe = jnp.asarray([[0.7], [-0.3]])
    np.testing.assert_allclose(
        jnp.vdot(target_density, probe),
        jnp.vdot(density, transition.pullback(probe)),
        rtol=0,
        atol=MAPPED_MEASURE_ATOL,
    )
    np.testing.assert_array_equal(
        _require_plan(remap).source_volumes, source.cell_volumes
    )
    np.testing.assert_array_equal(
        _require_plan(remap).target_volumes, target.cell_volumes
    )
    assert transition.transfer.properties.conservative
    assert not transition.transfer.properties.constant_preserving
    assert bool(transition.apply(density).successful)
    with pytest.raises(ValueError, match="density bounds"):
        _require_plan(remap).apply_bounded(jnp.full((2, 1), 0.5))


def test_curved_chart_fv_transaction_commit_and_stale_rollback(
    chart_fv_case: _ChartFVCase,
) -> None:
    source, target, _core, _carrier, remap = chart_fv_case
    density = jnp.asarray([[2.0], [3.0]])
    target_content = _require_plan(remap).apply_content(
        density * source.cell_volumes[:, None]
    )
    transition = _require_plan(remap).epoch_transition(
        source.cell_space,
        target.cell_space,
        TopologyEpoch(0, source.geometry_id, source.topology_id, "serial"),
        TopologyEpoch(1, target.geometry_id, target.topology_id, "serial"),
    )
    initial, successor = transition.source, transition.target
    initial_artifacts = FiniteVolumeTopologyArtifacts(initial, source.prepared_id)
    successor_artifacts = FiniteVolumeTopologyArtifacts(successor, target.prepared_id)
    source_metrics = lower_static_unstructured_stage_metrics(
        source, topology_epoch_id=initial.epoch_id
    )
    target_metrics = lower_static_unstructured_stage_metrics(
        target, topology_epoch_id=successor.epoch_id
    )

    def content(
        averages: Array,
        geometry: UnstructuredFiniteVolumeDiscretization,
        metrics: FiniteVolumeStageMetrics,
        epoch: TopologyEpoch,
        time: float,
    ) -> FiniteVolumeConservativeContentState:
        return FiniteVolumeConservativeContentState(
            averages * geometry.cell_volumes[:, None],
            geometry.cell_volumes,
            jnp.ones(geometry.cell_count, dtype=jnp.bool_),
            time,
            topology_epoch_id=epoch.epoch_id,
            geometry_family_id=metrics.geometry_family_id,
            geometry_layout_id=metrics.geometry_layout_id,
            geometry_version=metrics.geometry_version,
            evidence_policy_id=metrics.evidence.policy_id,
            evidence_version=metrics.evidence.evidence_version,
            precision=FiniteVolumePrecisionPolicy("float64"),
        )

    accepted = content(density, source, source_metrics, initial, 0.0)
    journal = FiniteVolumeTopologyEventJournal.allocate(
        initial, initial_artifacts, capacity=2, time=0.0
    )

    def transact(
        prepared_remap: PreparedUnstructuredConservativeRemap,
    ) -> FiniteVolumeTopologyEventTransactionResult:
        scheduler = FiniteVolumeTopologyEventScheduler(journal)
        scheduler.submit(
            FiniteVolumeTopologyEventRequest(
                TopologyEventKind.REMESH, initial.epoch_id, "sphere-chart-flip"
            ),
            1,
            0.1,
        )
        return scheduler.transact(
            source_geometry=source,
            target_geometry=target,
            source_content=accepted,
            candidate_epoch=successor,
            candidate_artifacts=successor_artifacts,
            remap=prepared_remap,
            coverage_tolerance=0.0,
            transfer=lambda state, plan: content(
                plan.apply(state.cell_average()), target, target_metrics, successor, 0.1
            ),
        )

    committed = transact(remap)
    assert committed.committed
    assert committed.receipt is not None
    assert committed.receipt.published
    assert committed.journal.current_epoch_id == successor.epoch_id
    (transport,) = committed.receipt.transports
    assert transport.target_content is not None
    assert transport.source_content is not None
    assert transport.content_tolerance is not None
    assert (
        abs(transport.target_content[0] - transport.source_content[0])
        <= transport.content_tolerance[0]
    )
    np.testing.assert_allclose(
        committed.content_state.conservative_content,
        target_content,
        rtol=0,
        atol=MAPPED_MEASURE_ATOL,
    )
    stale_source, stale_target, stale_core, stale_carrier = curved_surface_chart_fv(
        radius=1.1
    )
    stale_remap = prepare_unstructured_conservative_remap(
        stale_source,
        stale_target,
        provenance="independent-larger-sphere-chart",
        source_geometry=stale_core.source_geometry,
        target_geometry=stale_core.target_geometry,
        surface_chart_deformation=stale_core,
    )
    rejected = transact(stale_remap)
    assert (
        not rejected.committed
        and rejected.failure is TopologyEventStatus.FAILED_STALE_EPOCH
    )
    assert rejected.content_state is accepted
    assert rejected.journal.epoch_table == journal.epoch_table
    assert rejected.journal.artifact_table == journal.artifact_table


def test_curved_chart_fv_rejects_stale_incomplete_and_mismatched_bindings(
    chart_fv_case: _ChartFVCase,
) -> None:
    source, target, core, carrier, _remap = chart_fv_case
    _, _, stale_core, _ = curved_surface_chart_fv(radius=1.1)
    with pytest.raises(ValueError, match="stale"):
        prepare_unstructured_conservative_remap(
            source,
            target,
            provenance="reject-wrong-core",
            source_geometry=core.source_geometry,
            target_geometry=core.target_geometry,
            surface_chart_deformation=stale_core,
        )
    with pytest.raises(ValueError, match="coverage|omits"):
        prepare_unstructured_conservative_remap(
            source,
            target,
            provenance="reject-incomplete-core",
            source_geometry=core.source_geometry,
            target_geometry=core.target_geometry,
            surface_chart_deformation=replace(core, occurrences=()),
        )
    with pytest.raises(
        ValueError, match=r"prepare_unstructured_conservative_remap\.geometry_transition"
    ):
        prepare_unstructured_conservative_remap(
            source,
            target,
            provenance="reject-stale-chart-carrier",
            source_geometry=core.source_geometry,
            target_geometry=core.target_geometry,
            geometry_transition=replace(carrier, source_geometry_id="foreign"),
        )
