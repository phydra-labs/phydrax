#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from phydrax.discretization import CellBlock, CellMesh, UnstructuredFiniteVolumePlan
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from phydrax.discretization._cell_geometry_transfer import (
    _certified_cell_measures,
    CellGeometryTransition,
    transition_nested_cell_geometry,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._surface_chart_deformation import (
    PreparedSurfaceChartDeformation,
)
from phydrax.discretization._topology_epoch import TopologyEpoch
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.discretization.fem._surface_chart_transfer import (
    PreparedSurfaceChartFiniteVolumeContents,
)
from phydrax.discretization.finite_volume._automatic_remap import (
    prepare_unstructured_conservative_remap,
)
from phydrax.discretization.finite_volume._geometry_protocol import (
    FiniteVolumeStageMetrics,
)
from phydrax.discretization.finite_volume._remap_evidence import (
    MappedSurfaceChartRemapEvidence,
    PreparedUnstructuredConservativeRemap,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from phydrax.discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
)
from phydrax.lifecycle import CompositionEntry
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh, MixedAdaptationOutcome
from phydrax.meshing._topology_edit import assemble_topology_edit
from phydrax.solver._finite_volume_topology_events import (
    FiniteVolumeTopologyEventTransactionResult,
)


def _require_plan(
    remap: PreparedUnstructuredConservativeRemap,
) -> UnstructuredConservativeRemapPlan:
    if remap.plan is None:
        raise ValueError(f"Expected successful remap, got {remap.status}: {remap.reason}")
    return remap.plan


def _require_chart_evidence(
    remap: PreparedUnstructuredConservativeRemap,
) -> MappedSurfaceChartRemapEvidence:
    evidence = remap.evidence
    if not isinstance(evidence, MappedSurfaceChartRemapEvidence):
        raise TypeError("Expected actual mapped surface-chart evidence.")
    return evidence


def _require_chart_contents(
    plan: UnstructuredConservativeRemapPlan,
) -> PreparedSurfaceChartFiniteVolumeContents:
    contents = plan.surface_chart_contents
    if not isinstance(contents, PreparedSurfaceChartFiniteVolumeContents):
        raise TypeError("Expected actual mapped surface-chart contents.")
    return contents


def _map(points: NDArray[np.float64]) -> NDArray[np.float64]:
    points = np.asarray(points, dtype=np.float64).copy()
    points[:, 1] += 0.05 * points[:, 1] ** 2 + 0.05 * points[:, 0] * points[:, 2]
    return points


def _fixture(kind: str) -> tuple[CellMesh, CellGeometrySpec]:
    if kind == "connected":
        corners = np.concatenate(
            (
                np.asarray(reference_cell_topology("hexahedron").vertices),
                [[0.5, 0.5, 2.0]],
            )
        )
        blocks = (
            CellBlock(
                "hex", "hexahedron", np.arange(8)[None], global_ids=np.asarray([10])
            ),
            CellBlock(
                "cap",
                "pyramid",
                np.asarray([[4, 5, 6, 7, 8]]),
                global_ids=np.asarray([11]),
            ),
        )
    else:
        corners = np.asarray(reference_cell_topology(kind).vertices)
        blocks = (
            CellBlock(
                "volume", kind, np.arange(len(corners))[None], global_ids=np.asarray([17])
            ),
        )
    mesh = CellMesh(_map(corners), blocks)
    elements, routes, coefficients = {}, {}, []
    offset = 0
    for block in blocks:
        element = coordinate_lagrange_element(block.cell_kind, 2)
        nodes = np.asarray(element.reference_nodes).copy()
        if kind == "connected" and block.cell_kind == "pyramid":
            nodes[:, 2] += 1
        elements[block.name] = element
        routes[block.name] = np.arange(offset, offset + len(nodes))[None]
        coefficients.append(_map(nodes))
        offset += len(nodes)
    return mesh, CellGeometrySpec(elements, routes, np.concatenate(coefficients))


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
    target, _, _ = assemble_topology_edit(mesh, outcome.edit, numeric_version="fine")
    transition = transition_nested_cell_geometry(
        mesh,
        geometry,
        target,
        CellGeometrySpec.affine(target),
        refinement=outcome.edit.refinement,
    )
    return target, transition, outcome


@pytest.mark.parametrize(
    "kind,total",
    [
        ("hexahedron", 1.05),
        ("prism", 31 / 60),
        ("pyramid", 0.35),
        ("tetrahedron", 41 / 240),
        ("connected", 1.4),
    ],
)
def test_curved_nested_inventory_refine_coarsen_and_transpose(
    kind: str, total: float
) -> None:
    mesh, geometry = _fixture(kind)
    source = _prepare(mesh, geometry)
    np.testing.assert_allclose(np.sum(source.cell_volumes), total, rtol=0, atol=2e-14)
    fine_mesh, transition, outcome = _refine(mesh, geometry)
    fine = _prepare(fine_mesh, transition.geometry)
    remap = prepare_unstructured_conservative_remap(
        source,
        fine,
        provenance="curved-refine",
        source_geometry=geometry,
        target_geometry=transition.geometry,
        geometry_transition=transition,
    )
    assert remap.refinement is None and remap.route == "mapped-nested"
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
        atol=3e-14,
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
        transferred_content, fine.cell_volumes[:, None] * transferred, rtol=0, atol=3e-14
    )
    np.testing.assert_allclose(
        np.sum(transferred_content), np.sum(content), rtol=0, atol=3e-14
    )
    probe = jnp.sin(jnp.arange(fine.cell_count, dtype=jnp.float64))[:, None]
    np.testing.assert_allclose(
        jnp.vdot(transferred, probe),
        jnp.vdot(values, _require_plan(remap)._transpose_averages(probe)),
        rtol=0,
        atol=3e-14,
    )
    measured, errors, _ = _certified_cell_measures(fine_mesh, transition.geometry)
    np.testing.assert_array_equal(fine.cell_volumes, measured)
    np.testing.assert_array_equal(fine.cell_volume_error_bounds, errors)
    np.testing.assert_array_equal(_require_plan(remap).target_volumes, measured)
    ids = np.concatenate([np.asarray(block.global_ids) for block in fine_mesh.blocks])
    coarsened = adapt_mixed_mesh(
        fine_mesh,
        refine_cell_ids=np.asarray([], dtype=np.int64),
        coarsen_cell_ids=ids,
        hierarchy=outcome.hierarchy,
    )
    restored_mesh, _, _ = assemble_topology_edit(
        fine_mesh, coarsened.edit, numeric_version="restored"
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
    # Nonconstant fine inventory exercises true many-to-one weighted aggregation.
    fine_values = jnp.arange(fine.cell_count, dtype=jnp.float64)[:, None] / 7 + 1
    fine_content = fine_values * fine.cell_volumes[:, None]
    coarse_content = _require_plan(reverse).apply_content(fine_content)
    np.testing.assert_allclose(
        coarse_content,
        _require_plan(reverse).apply(fine_values) * restored.cell_volumes[:, None],
        rtol=0,
        atol=4e-14,
    )
    np.testing.assert_allclose(
        np.sum(coarse_content), np.sum(fine_content), rtol=0, atol=4e-14
    )
    coarse_probe = jnp.arange(restored.cell_count, dtype=jnp.float64)[:, None] + 0.7
    np.testing.assert_allclose(
        jnp.vdot(_require_plan(reverse).apply(fine_values), coarse_probe),
        jnp.vdot(fine_values, _require_plan(reverse)._transpose_averages(coarse_probe)),
        rtol=0,
        atol=4e-14,
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
        atol=4e-14,
    )
    np.testing.assert_array_equal(
        restored.cell_volumes, source.cell_volumes[restored_rows]
    )


def test_stale_wrong_and_incomplete_mapped_witnesses_are_rejected() -> None:
    mesh, geometry = _fixture("hexahedron")
    source = _prepare(mesh, geometry)
    fine_mesh, transition, _ = _refine(mesh, geometry)
    fine = _prepare(fine_mesh, transition.geometry)
    wrong = replace(transition, source_topology_id="foreign")
    incomplete = eqx.tree_at(
        lambda x: (x.target_cell_ids, x.parent_cell_ids, x.parent_reference_vertices),
        transition,
        (
            transition.target_cell_ids[:-1],
            transition.parent_cell_ids[:-1],
            transition.parent_reference_vertices[:-1],
        ),
    )
    for witness in (wrong, incomplete):
        with pytest.raises(ValueError):
            prepare_unstructured_conservative_remap(
                source,
                fine,
                provenance="reject",
                source_geometry=geometry,
                target_geometry=transition.geometry,
                geometry_transition=witness,
            )
    stale = CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        np.asarray(geometry.coordinates) + 0.01,
    )
    with pytest.raises(ValueError, match="bound"):
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
    """Actual affine reference partition, with both children using the root map."""
    from phydrax.discretization._cell_geometry import RestrictedCellGeometryElement

    source_mesh, source_geometry = _fixture("hexahedron")
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
        raise TypeError("Represented children require a tabulatable coordinate root.")
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
    target_mesh = CellMesh(_map(np.asarray(points, dtype=np.float64)) + shift, blocks)
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


def test_actual_mapped_transaction_rolls_back_stale_remap_geometry() -> None:
    from phydrax.discretization import (
        FiniteVolumePrecisionPolicy,
        lower_static_unstructured_stage_metrics,
    )
    from phydrax.solver import FiniteVolumeConservativeContentState
    from phydrax.solver._finite_volume_topology_events import (
        FiniteVolumeTopologyArtifacts,
        FiniteVolumeTopologyEventJournal,
        FiniteVolumeTopologyEventRequest,
        FiniteVolumeTopologyEventScheduler,
        TopologyEventKind,
        TopologyEventStatus,
    )

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
        committed.content_state.volume_integral(), 3.15, rtol=0, atol=2e-14
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


def _curved_surface_chart_fv_fixture(
    radius: float = 1.0,
) -> tuple[
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumeDiscretization,
    PreparedSurfaceChartDeformation,
    CellGeometryTransition,
]:
    from phydrax.discretization._cell_geometry_transfer import (
        CellGeometryTransitionPolicy,
        reconstruct_parametric_surface_cell_geometry,
        transition_chart_deformed_cell_geometry,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._surface_chart_deformation import (
        prepare_surface_chart_deformation,
        SurfaceChartWitness,
    )
    from phydrax.geometry._meshing_domain import (
        MeshingDomain,
        MeshingDomainCurve,
        MeshingSurfacePatch,
        PatchCurveUse,
    )
    from phydrax.geometry.brep._patches import LineCurve, SpherePatch

    uv = np.asarray([[0.125, 0.125], [0.25, 0.125], [0.25, 0.25], [0.125, 0.25]])
    surface = SpherePatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], radius)
    uses = tuple(
        PatchCurveUse(i, LineCurve(uv[i], uv[(i + 1) % 4] - uv[i]), 0.0, 1.0)
        for i in range(4)
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        tuple(MeshingDomainCurve(i, (i + 1) % 4) for i in range(4)),
        4,
        source_id="regular-patch",
        source_revision="r1",
        source_occurrences=(
            tuple(("assembly", f"corner-{i}") for i in range(4)),
            tuple(("assembly", f"curve-{i}") for i in range(4)),
            (("assembly", "face-7"),),
        ),
    )
    xyz = domain.evaluate(np.zeros(4, dtype=np.int32), uv)
    old = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    new = np.asarray([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
    # The accepted old represented surface is within its declared CAD fidelity
    # budget but has genuinely smaller area than the reconstructed target.
    source_mesh = CellMesh.from_triangles(
        0.99 * xyz, old, cell_global_ids=np.asarray([107, 13])
    )
    target_mesh = CellMesh.from_triangles(xyz, new, cell_global_ids=np.asarray([59, 211]))
    element = coordinate_lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes)
    barycentric = np.column_stack((1 - nodes.sum(axis=1), nodes))

    def layout(cells: NDArray[np.int32], radial_scale: float = 1.0) -> CellGeometrySpec:
        charts = np.einsum("nk,ckd->cnd", barycentric, uv[cells])
        unique, inverse = np.unique(charts.reshape((-1, 2)), axis=0, return_inverse=True)
        return CellGeometrySpec(
            {"triangles": element},
            {"triangles": inverse.reshape((2, -1))},
            radial_scale * domain.evaluate(np.zeros(len(unique), dtype=np.int32), unique),
        )

    source_geometry = layout(old, radial_scale=0.99)
    entities, paths = (
        (domain.entity_id(2, 0),) * 2,
        (domain.source_occurrences[2][0],) * 2,
    )
    reconstruction = reconstruct_parametric_surface_cell_geometry(
        source_mesh,
        source_geometry,
        target_mesh,
        layout(new),
        domain,
        domain_id=domain.domain_id,
        cell_ids=np.asarray([59, 211]),
        cell_patches=np.zeros(2, dtype=np.int32),
        cell_charts=uv[new],
        cell_geometry_entity_ids=entities,
        cell_occurrence_paths=paths,
        maximum_fidelity=0.5,
    )
    source_witness = SurfaceChartWitness(
        geometry_id=cell_geometry_id(source_geometry),
        topology_id=source_mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=jnp.asarray([107, 13]),
        patches=jnp.zeros(2, dtype=jnp.int32),
        charts=jnp.asarray(uv[old]),
        geometry_entity_ids=entities,
        occurrence_paths=paths,
    )
    target_witness = SurfaceChartWitness(
        geometry_id=cell_geometry_id(reconstruction.geometry),
        topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=jnp.asarray([59, 211]),
        patches=jnp.zeros(2, dtype=jnp.int32),
        charts=jnp.asarray(uv[new]),
        geometry_entity_ids=entities,
        occurrence_paths=paths,
    )
    core = prepare_surface_chart_deformation(
        source_mesh,
        source_geometry,
        target_mesh,
        reconstruction,
        domain,
        source_witness=source_witness,
        target_witness=target_witness,
        maximum_fidelity=0.5,
        maximum_displacement=1.0,
    )
    carrier = transition_chart_deformed_cell_geometry(
        source_mesh,
        source_geometry,
        target_mesh,
        reconstruction,
        core,
        policy=CellGeometryTransitionPolicy(
            reconstruction="bounded_chart_deformation", reconstruction_tolerance=1.0
        ),
    )
    return (
        _prepare(source_mesh, source_geometry),
        _prepare(target_mesh, reconstruction.geometry),
        core,
        carrier,
    )


def test_actual_curved_chart_fv_density_content_and_transpose() -> None:
    source, target, core, carrier = _curved_surface_chart_fv_fixture()
    remap = prepare_unstructured_conservative_remap(
        source,
        target,
        provenance="actual-sphere-chart-density",
        source_geometry=core.source_geometry,
        target_geometry=core.target_geometry,
        geometry_transition=carrier,
    )
    assert remap.route == "mapped-surface-chart" and remap.refinement is None
    assert _require_chart_evidence(remap).passed
    assert np.sum(source.cell_volumes) < np.sum(target.cell_volumes)
    assert np.max(_require_plan(remap).apply(jnp.ones((2, 1)))) < 0.99
    density = jnp.asarray([[2.0], [3.0]])
    source_content = density * source.cell_volumes[:, None]
    target_density = _require_plan(remap).apply(density)
    target_content = _require_plan(remap).apply_content(source_content)
    np.testing.assert_allclose(
        target_content, target_density * target.cell_volumes[:, None], rtol=0, atol=2e-14
    )
    content_bound = jnp.vdot(
        _require_chart_contents(_require_plan(remap)).source_content_defect_bounds,
        jnp.abs(density[:, 0]),
    )
    assert abs(np.sum(target_content) - np.sum(source_content)) <= content_bound
    np.testing.assert_array_equal(_require_plan(remap).source_cell_global_ids, [107, 13])
    np.testing.assert_array_equal(_require_plan(remap).target_cell_global_ids, [59, 211])
    probe = jnp.asarray([[0.7], [-0.3]])
    np.testing.assert_allclose(
        jnp.vdot(target_density, probe),
        jnp.vdot(density, _require_plan(remap)._transpose_averages(probe)),
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(
        _require_plan(remap).source_volumes, source.cell_volumes
    )
    np.testing.assert_array_equal(
        _require_plan(remap).target_volumes, target.cell_volumes
    )
    transition = _require_plan(remap).epoch_transition(
        source.cell_space,
        target.cell_space,
        TopologyEpoch(0, source.geometry_id, source.topology_id, "serial"),
        TopologyEpoch(1, target.geometry_id, target.topology_id, "serial"),
    )
    assert transition.transfer.properties.conservative
    assert not transition.transfer.properties.constant_preserving
    assert bool(transition.apply(density).successful)
    with pytest.raises(ValueError, match="density bounds"):
        _require_plan(remap).apply_bounded(jnp.full((2, 1), 0.5))

    from phydrax.discretization import (
        FiniteVolumePrecisionPolicy,
        lower_static_unstructured_stage_metrics,
    )
    from phydrax.solver import FiniteVolumeConservativeContentState
    from phydrax.solver._finite_volume_topology_events import (
        FiniteVolumeTopologyArtifacts,
        FiniteVolumeTopologyEventJournal,
        FiniteVolumeTopologyEventRequest,
        FiniteVolumeTopologyEventScheduler,
        TopologyEventKind,
        TopologyEventStatus,
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
        committed.content_state.conservative_content, target_content, rtol=0, atol=2e-14
    )
    stale_source, stale_target, stale_core, stale_carrier = (
        _curved_surface_chart_fv_fixture(radius=1.1)
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
    with pytest.raises(ValueError, match="stale"):
        prepare_unstructured_conservative_remap(
            source,
            target,
            provenance="reject-stale-chart-carrier",
            source_geometry=core.source_geometry,
            target_geometry=core.target_geometry,
            geometry_transition=replace(carrier, source_geometry_id="foreign"),
        )
