from __future__ import annotations

import jax.numpy as jnp
import pytest

from phydrax.applications.cellular_mechanics._vertex_tissue import (
    commit_vertex_tissue_topology,
    evaluate_vertex_tissue_topology,
    polygonal_vertex_tissue_plan,
    polyhedral_vertex_tissue_plan,
    propose_vertex_tissue_topology,
    rollback_vertex_tissue_topology,
    VertexTissueDynamicsPlan,
    VertexTissueEventKind,
    VertexTissueState,
    VertexTissueStatus,
    VertexTissueTopologyEvent,
)


_SQUARE = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))


def _two_triangle_plan(*, flipped: bool = False, remove_first: bool = False):
    if flipped:
        edge_vertices = jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0), (1, 3)))
        cell_edges = jnp.asarray(((1, 2, 4), (4, 3, 0)))
        orientations = jnp.asarray(((1, 1, -1), (1, 1, 1)))
        interfaces = jnp.asarray(((1, -1), (0, -1), (0, -1), (1, -1), (0, 1)))
        vertex_ids = jnp.arange(4)
        edge_ids = jnp.arange(5)
        cell_ids = jnp.asarray((10, 11))
        cell_types = jnp.asarray((0, 0))
        parents = jnp.asarray((10, 11))
        generations = jnp.asarray((0, 0))
    elif remove_first:
        edge_vertices = jnp.asarray(((-1, -1), (-1, -1), (2, 3), (3, 0), (0, 2)))
        cell_edges = jnp.asarray(((-1, -1, -1), (4, 2, 3)))
        orientations = jnp.asarray(((0, 0, 0), (1, 1, 1)))
        interfaces = jnp.asarray(((-1, -1), (-1, -1), (1, -1), (1, -1), (1, -1)))
        vertex_ids = jnp.asarray((0, -1, 2, 3))
        edge_ids = jnp.asarray((-1, -1, 2, 3, 4))
        cell_ids = jnp.asarray((-1, 11))
        cell_types = jnp.asarray((-1, 0))
        parents = jnp.asarray((-1, 11))
        generations = jnp.asarray((-1, 0))
    else:
        edge_vertices = jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)))
        cell_edges = jnp.asarray(((0, 1, 4), (4, 2, 3)))
        orientations = jnp.asarray(((1, 1, -1), (1, 1, 1)))
        interfaces = jnp.asarray(((0, -1), (0, -1), (1, -1), (1, -1), (0, 1)))
        vertex_ids = jnp.arange(4)
        edge_ids = jnp.arange(5)
        cell_ids = jnp.asarray((10, 11))
        cell_types = jnp.asarray((0, 0))
        parents = jnp.asarray((10, 11))
        generations = jnp.asarray((0, 0))
    return polygonal_vertex_tissue_plan(
        vertex_ids,
        edge_ids,
        edge_vertices,
        cell_ids,
        cell_edges,
        orientations,
        interfaces,
        jnp.asarray((0.5, 0.5)),
        jnp.ones((2,)),
        jnp.full((2,), 2.0 + jnp.sqrt(2.0)),
        jnp.ones((2,)),
        cell_types=cell_types,
        cell_parent_ids=parents,
        cell_generation=generations,
        field_names=("morphogen",),
    )


def _t1_plan(*, exchanged: bool):
    old_positions = jnp.asarray(
        (
            (-1.0, 1.0),
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-0.1, 0.0),
            (0.1, 0.0),
        )
    )
    new_positions = (
        old_positions.at[4]
        .set(jnp.asarray((0.0, 0.1)))
        .at[5]
        .set(jnp.asarray((0.0, -0.1)))
    )
    if exchanged:
        edge_vertices = jnp.asarray(
            (
                (4, 5),
                (4, 0),
                (4, 1),
                (5, 2),
                (5, 3),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
            )
        )
        cell_edges = jnp.asarray(
            (
                (2, 5, 1, -1),
                (4, 7, 3, -1),
                (1, 8, 4, 0),
                (0, 3, 6, 2),
            )
        )
        orientations = jnp.asarray(
            (
                (1, -1, -1, 0),
                (1, -1, -1, 0),
                (1, -1, -1, -1),
                (1, 1, -1, -1),
            )
        )
        interfaces = jnp.asarray(
            (
                (2, 3),
                (0, 2),
                (0, 3),
                (1, 3),
                (1, 2),
                (0, -1),
                (3, -1),
                (1, -1),
                (2, -1),
            )
        )
        positions = new_positions
    else:
        edge_vertices = jnp.asarray(
            (
                (4, 5),
                (4, 0),
                (5, 1),
                (5, 2),
                (4, 3),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
            )
        )
        cell_edges = jnp.asarray(
            (
                (0, 2, 5, 1),
                (4, 7, 3, 0),
                (1, 8, 4, -1),
                (3, 6, 2, -1),
            )
        )
        orientations = jnp.asarray(
            (
                (1, 1, -1, -1),
                (1, -1, -1, -1),
                (1, -1, -1, 0),
                (1, -1, -1, 0),
            )
        )
        interfaces = jnp.asarray(
            (
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 3),
                (1, 2),
                (0, -1),
                (3, -1),
                (1, -1),
                (2, -1),
            )
        )
        positions = old_positions
    plan = polygonal_vertex_tissue_plan(
        jnp.arange(6),
        jnp.arange(9),
        edge_vertices,
        jnp.asarray((10, 11, 12, 13)),
        cell_edges,
        orientations,
        interfaces,
        jnp.ones((4,)),
        jnp.zeros((4,)),
        jnp.ones((4,)),
        jnp.zeros((4,)),
        field_names=("morphogen",),
    )
    return plan, positions


def _division_plan(*, divided: bool):
    if divided:
        edge_ids = jnp.arange(5)
        edge_vertices = jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)))
        cell_ids = jnp.asarray((10, 12))
        cell_edges = jnp.asarray(((0, 1, 4, -1), (4, 2, 3, -1)))
        orientations = jnp.asarray(((1, 1, -1, 0), (1, 1, 1, 0)))
        interfaces = jnp.asarray(((0, -1), (0, -1), (1, -1), (1, -1), (0, 1)))
        cell_types = jnp.asarray((0, 0))
        parents = jnp.asarray((10, 10))
        generations = jnp.asarray((0, 1))
        target_area = jnp.asarray((0.5, 0.5))
        target_perimeter = jnp.full((2,), 2.0 + jnp.sqrt(2.0))
    else:
        edge_ids = jnp.asarray((0, 1, 2, 3, -1))
        edge_vertices = jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0), (-1, -1)))
        cell_ids = jnp.asarray((10, -1))
        cell_edges = jnp.asarray(((0, 1, 2, 3), (-1, -1, -1, -1)))
        orientations = jnp.asarray(((1, 1, 1, 1), (0, 0, 0, 0)))
        interfaces = jnp.asarray(((0, -1), (0, -1), (0, -1), (0, -1), (-1, -1)))
        cell_types = jnp.asarray((0, -1))
        parents = jnp.asarray((10, -1))
        generations = jnp.asarray((0, -1))
        target_area = jnp.asarray((1.0, 1.0))
        target_perimeter = jnp.asarray((4.0, 4.0))
    return polygonal_vertex_tissue_plan(
        jnp.arange(4),
        edge_ids,
        edge_vertices,
        cell_ids,
        cell_edges,
        orientations,
        interfaces,
        target_area,
        jnp.ones((2,)),
        target_perimeter,
        jnp.ones((2,)),
        cell_types=cell_types,
        cell_parent_ids=parents,
        cell_generation=generations,
        field_names=("morphogen",),
    )


def _division_with_neighbor(*, divided: bool):
    positions = jnp.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        )
    )
    if divided:
        edge_ids = jnp.arange(8)
        edge_vertices = jnp.asarray(
            (
                (0, 1),
                (1, 2),
                (2, 5),
                (5, 4),
                (4, 3),
                (3, 0),
                (1, 4),
                (0, 4),
            )
        )
        cell_ids = jnp.asarray((10, 11, 12))
        cell_edges = jnp.asarray(((0, 6, 7, -1), (1, 2, 3, 6), (7, 4, 5, -1)))
        orientations = jnp.asarray(((1, 1, -1, 0), (1, 1, 1, -1), (1, 1, 1, 0)))
        interfaces = jnp.asarray(
            (
                (0, -1),
                (1, -1),
                (1, -1),
                (1, -1),
                (2, -1),
                (2, -1),
                (0, 1),
                (0, 2),
            )
        )
        parents = jnp.asarray((10, 11, 10))
        generations = jnp.asarray((0, 0, 1))
    else:
        edge_ids = jnp.asarray((0, 1, 2, 3, 4, 5, 6, -1))
        edge_vertices = jnp.asarray(
            (
                (0, 1),
                (1, 2),
                (2, 5),
                (5, 4),
                (4, 3),
                (3, 0),
                (1, 4),
                (-1, -1),
            )
        )
        cell_ids = jnp.asarray((10, 11, -1))
        cell_edges = jnp.asarray(((0, 6, 4, 5), (1, 2, 3, 6), (-1, -1, -1, -1)))
        orientations = jnp.asarray(((1, 1, 1, 1), (1, 1, 1, -1), (0, 0, 0, 0)))
        interfaces = jnp.asarray(
            (
                (0, -1),
                (1, -1),
                (1, -1),
                (1, -1),
                (0, -1),
                (0, -1),
                (0, 1),
                (-1, -1),
            )
        )
        parents = jnp.asarray((10, 11, -1))
        generations = jnp.asarray((0, 0, -1))
    plan = polygonal_vertex_tissue_plan(
        jnp.arange(6),
        edge_ids,
        edge_vertices,
        cell_ids,
        cell_edges,
        orientations,
        interfaces,
        jnp.ones((3,)),
        jnp.zeros((3,)),
        jnp.ones((3,)),
        jnp.zeros((3,)),
        cell_types=jnp.where(cell_ids >= 0, 0, -1),
        cell_parent_ids=parents,
        cell_generation=generations,
        field_names=("morphogen",),
    )
    return plan, positions


def _evaluate_event(source, state, kind, target, transfer, positions=_SQUARE):
    event = VertexTissueTopologyEvent(
        kind, source.prepared_id, target, positions, transfer
    )
    candidate = propose_vertex_tissue_topology(source, state, event)
    evaluation = evaluate_vertex_tissue_topology(source, state, candidate)
    return candidate, evaluation


def test_t1_commits_only_a_four_cell_neighbor_exchange():
    source_plan, source_positions = _t1_plan(exchanged=False)
    target_plan, target_positions = _t1_plan(exchanged=True)
    source = source_plan.prepare(source_positions)
    state = source.initialize_state(jnp.asarray(((1.0,), (2.0,), (3.0,), (4.0,))))
    candidate, evaluation = _evaluate_event(
        source,
        state,
        VertexTissueEventKind.T1,
        target_plan,
        jnp.eye(4),
        target_positions,
    )
    result = commit_vertex_tissue_topology(source, state, candidate, evaluation)

    assert evaluation.passed
    assert result.committed
    assert result.prepared.prepared_id != source.prepared_id
    assert jnp.array_equal(result.state.cell_fields, state.cell_fields)


def test_t3_commits_a_vertex_edge_rearrangement_but_is_not_mislabeled_t1():
    source = _two_triangle_plan().prepare(_SQUARE)
    state = source.initialize_state(jnp.asarray(((4.0,), (6.0,))))
    target = _two_triangle_plan(flipped=True)
    t3_candidate, t3_evaluation = _evaluate_event(
        source, state, VertexTissueEventKind.T3, target, jnp.eye(2)
    )
    t1_candidate, t1_evaluation = _evaluate_event(
        source, state, VertexTissueEventKind.T1, target, jnp.eye(2)
    )
    result = commit_vertex_tissue_topology(source, state, t3_candidate, t3_evaluation)

    assert t3_evaluation.passed
    assert result.committed
    assert not t1_evaluation.kind_valid
    assert not commit_vertex_tissue_topology(
        source, state, t1_candidate, t1_evaluation
    ).committed


def test_t3_rejects_identifier_only_relabeling():
    source = _two_triangle_plan().prepare(_SQUARE)
    state = source.initialize_state(jnp.asarray(((4.0,), (6.0,))))
    plan = source.plan
    relabeled = polygonal_vertex_tissue_plan(
        plan.vertex_ids + 100,
        plan.edge_ids + 100,
        plan.edge_vertex_indices,
        plan.cell_ids,
        plan.cell_edge_indices,
        plan.cell_edge_orientations,
        plan.interface_cell_indices,
        plan.target_cell_measure,
        plan.cell_measure_stiffness,
        plan.target_boundary_measure,
        plan.boundary_stiffness,
        cell_types=plan.cell_types,
        cell_parent_ids=plan.cell_parent_ids,
        cell_generation=plan.cell_generation,
        field_names=plan.field_names,
    )
    candidate, evaluation = _evaluate_event(
        source,
        state,
        VertexTissueEventKind.T3,
        relabeled,
        jnp.eye(2),
    )

    assert not evaluation.kind_valid
    assert not commit_vertex_tissue_topology(
        source, state, candidate, evaluation
    ).committed


def test_candidate_is_bound_to_exact_source_state_and_commit_epoch():
    source_plan = _two_triangle_plan()
    source = source_plan.prepare(_SQUARE)
    state = source.initialize_state(jnp.asarray(((4.0,), (6.0,))))
    target = _two_triangle_plan(flipped=True)
    candidate, evaluation = _evaluate_event(
        source, state, VertexTissueEventKind.T3, target, jnp.eye(2)
    )
    changed_state = VertexTissueState(
        state.positions,
        jnp.asarray(((6.0,), (4.0,))),
        state.time,
        source.prepared_id,
    )
    changed_evaluation = evaluate_vertex_tissue_topology(source, changed_state, candidate)
    changed_result = commit_vertex_tissue_topology(
        source, changed_state, candidate, evaluation
    )

    shifted_positions = _SQUARE + jnp.asarray((0.05, 0.0))
    other_epoch = source_plan.prepare(shifted_positions)
    other_state = other_epoch.initialize_state(state.cell_fields)
    other_result = commit_vertex_tissue_topology(
        other_epoch, other_state, candidate, evaluation
    )

    assert evaluation.passed
    assert not changed_evaluation.source_state_valid
    assert not changed_evaluation.mapping_valid
    assert not changed_result.committed
    assert changed_result.status == int(VertexTissueStatus.STALE_EPOCH)
    assert jnp.array_equal(changed_result.state.cell_fields, changed_state.cell_fields)
    assert not other_result.committed
    assert other_result.status == int(VertexTissueStatus.STALE_EPOCH)
    assert other_result.prepared.prepared_id == other_epoch.prepared_id


@pytest.mark.parametrize(
    "kind",
    (
        VertexTissueEventKind.T2,
        VertexTissueEventKind.EXTRUSION,
        VertexTissueEventKind.APOPTOSIS,
    ),
)
def test_removal_events_redistribute_conserved_field_and_preserve_survivor_lineage(kind):
    source = _two_triangle_plan().prepare(_SQUARE)
    state = source.initialize_state(jnp.asarray(((4.0,), (6.0,))))
    target = _two_triangle_plan(remove_first=True)
    candidate, evaluation = _evaluate_event(
        source,
        state,
        kind,
        target,
        jnp.asarray(((0.0, 0.0), (1.0, 1.0))),
    )
    result = commit_vertex_tissue_topology(source, state, candidate, evaluation)

    assert evaluation.passed
    assert result.committed
    assert result.state.cell_fields[1, 0] == pytest.approx(10.0)
    assert jnp.sum(result.state.cell_fields) == pytest.approx(jnp.sum(state.cell_fields))
    assert result.prepared.plan.cell_ids[1] == 11
    assert result.prepared.plan.cell_parent_ids[1] == 11


def test_division_records_parent_generation_and_splits_conserved_field():
    source = _division_plan(divided=False).prepare(_SQUARE)
    state = source.initialize_state(jnp.asarray(((10.0,), (0.0,))))
    target = _division_plan(divided=True)
    candidate, evaluation = _evaluate_event(
        source,
        state,
        VertexTissueEventKind.DIVISION,
        target,
        jnp.asarray(((0.4, 0.0), (0.6, 0.0))),
    )
    result = commit_vertex_tissue_topology(source, state, candidate, evaluation)

    assert evaluation.passed
    assert evaluation.lineage_valid
    assert evaluation.conservation_valid
    assert result.committed
    assert jnp.allclose(result.state.cell_fields[:, 0], jnp.asarray((4.0, 6.0)))
    assert result.prepared.plan.cell_parent_ids[1] == 10
    assert result.prepared.plan.cell_generation[1] == 1


def test_division_rejects_transfer_from_a_cell_other_than_declared_parent():
    source_plan, positions = _division_with_neighbor(divided=False)
    target_plan, _ = _division_with_neighbor(divided=True)
    source = source_plan.prepare(positions)
    state = source.initialize_state(jnp.asarray(((4.0,), (6.0,), (0.0,))))
    candidate, evaluation = _evaluate_event(
        source,
        state,
        VertexTissueEventKind.DIVISION,
        target_plan,
        jnp.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 0.5, 0.0),
            )
        ),
        positions,
    )
    result = commit_vertex_tissue_topology(source, state, candidate, evaluation)

    assert evaluation.conservation_valid
    assert not evaluation.lineage_transfer_valid
    assert not evaluation.passed
    assert not result.committed


def test_inactive_cell_fields_and_rates_cannot_be_accepted():
    plan = _division_plan(divided=False)
    tissue = plan.prepare(_SQUARE)
    with pytest.raises(ValueError, match="Inactive cell field"):
        tissue.initialize_state(jnp.asarray(((1.0,), (2.0,))))

    invalid_state = VertexTissueState(
        _SQUARE,
        jnp.asarray(((1.0,), (2.0,))),
        0.0,
        tissue.prepared_id,
    )
    invalid_evaluation = tissue.evaluate(invalid_state)
    state = tissue.initialize_state(jnp.asarray(((1.0,), (0.0,))))
    dynamics = VertexTissueDynamicsPlan(1.0e-3).prepare(tissue)
    step = dynamics.step(state, cell_field_rates=jnp.asarray(((0.0,), (1.0,))))

    assert not invalid_evaluation.inactive_fields_valid
    assert not invalid_evaluation.valid
    assert not step.field_rate_valid
    assert not step.accepted
    assert jnp.array_equal(step.state.cell_fields, state.cell_fields)


def test_failed_quality_guard_and_explicit_rollback_leave_epoch_unchanged():
    source = _two_triangle_plan().prepare(_SQUARE)
    state = source.initialize_state(jnp.asarray(((4.0,), (6.0,))))
    target = _two_triangle_plan(flipped=True)
    collapsed = _SQUARE.at[3].set(_SQUARE[2])
    candidate, evaluation = _evaluate_event(
        source,
        state,
        VertexTissueEventKind.T3,
        target,
        jnp.eye(2),
        collapsed,
    )
    rejected = commit_vertex_tissue_topology(source, state, candidate, evaluation)
    rolled_back = rollback_vertex_tissue_topology(source, state, candidate, evaluation)

    assert not evaluation.passed
    assert not evaluation.quality_valid
    assert not rejected.committed
    assert rejected.prepared.prepared_id == source.prepared_id
    assert jnp.array_equal(rejected.state.positions, state.positions)
    assert not rolled_back.committed
    assert rolled_back.prepared.prepared_id == source.prepared_id


def test_capacity_change_is_rejected_before_topology_commit():
    source = _two_triangle_plan().prepare(_SQUARE)
    state = source.initialize_state(jnp.asarray(((4.0,), (6.0,))))
    target = polygonal_vertex_tissue_plan(
        jnp.arange(4),
        jnp.arange(4),
        jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0))),
        jnp.asarray((10,)),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.ones((1, 4), dtype=jnp.int32),
        jnp.asarray(((0, -1), (0, -1), (0, -1), (0, -1))),
        1.0,
        1.0,
        4.0,
        1.0,
        field_names=("morphogen",),
    )
    candidate, evaluation = _evaluate_event(
        source,
        state,
        VertexTissueEventKind.T3,
        target,
        jnp.asarray(((1.0, 1.0),)),
    )
    result = commit_vertex_tissue_topology(source, state, candidate, evaluation)

    assert not evaluation.capacity_valid
    assert not evaluation.passed
    assert not result.committed
    assert result.prepared.prepared_id == source.prepared_id


def _tetrahedron_epoch(*, transitioned: bool):
    positions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.2),
        )
    )
    apex = 4 if transitioned else 3
    vertex_ids = (
        jnp.asarray((0, 1, 2, -1, 4)) if transitioned else jnp.asarray((0, 1, 2, 3, -1))
    )
    edges = jnp.asarray(((0, 1), (0, 2), (0, apex), (1, 2), (1, apex), (2, apex)))
    faces = jnp.asarray(((1, 2, apex), (0, apex, 2), (0, 1, apex), (0, 2, 1)))
    plan = polyhedral_vertex_tissue_plan(
        vertex_ids,
        jnp.arange(6),
        edges,
        jnp.arange(4),
        faces,
        jnp.asarray((20,)),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.ones((1, 4), dtype=jnp.int32),
        jnp.asarray(((0, -1), (0, -1), (0, -1), (0, -1))),
        1.0 / 6.0 if not transitioned else 0.2,
        1.0,
        2.0,
        0.0,
        field_names=("mass",),
    )
    return plan, positions


def test_three_dimensional_edge_transition_cannot_be_mislabeled_as_face_transition():
    source_plan, positions = _tetrahedron_epoch(transitioned=False)
    target_plan, _ = _tetrahedron_epoch(transitioned=True)
    source = source_plan.prepare(positions)
    state = source.initialize_state(jnp.asarray(((7.0,),)))
    edge_candidate, edge_evaluation = _evaluate_event(
        source,
        state,
        VertexTissueEventKind.EDGE_TRANSITION,
        target_plan,
        jnp.ones((1, 1)),
        positions,
    )
    face_candidate, face_evaluation = _evaluate_event(
        source,
        state,
        VertexTissueEventKind.FACE_TRANSITION,
        target_plan,
        jnp.ones((1, 1)),
        positions,
    )
    result = commit_vertex_tissue_topology(source, state, edge_candidate, edge_evaluation)

    assert edge_evaluation.passed
    assert edge_evaluation.manifold
    assert edge_evaluation.orientation_valid
    assert result.committed
    assert result.prepared.plan.dimension == 3
    assert result.prepared.prepared_id != source.prepared_id
    assert result.state.cell_fields[0, 0] == pytest.approx(7.0)
    assert not face_evaluation.kind_valid
    assert not commit_vertex_tissue_topology(
        source, state, face_candidate, face_evaluation
    ).committed
