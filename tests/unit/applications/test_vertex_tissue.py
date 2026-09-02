from __future__ import annotations

import jax.numpy as jnp
import pytest

from phydrax.applications.cellular_mechanics._vertex_tissue import (
    couple_vertex_tissue_particles,
    polygonal_vertex_tissue_plan,
    polyhedral_vertex_tissue_plan,
    vertex_tissue_potential_energy,
    VertexTissueDynamicsPlan,
    VertexTissueState,
)


def _square_tissue(
    *,
    target_area=1.0,
    target_perimeter=4.0,
    area_stiffness=2.0,
    perimeter_stiffness=3.0,
    active_contractility=0.0,
    interface_tension=0.0,
    cell_traction=None,
    field_names=(),
):
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    plan = polygonal_vertex_tissue_plan(
        jnp.arange(4),
        jnp.arange(4),
        jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0))),
        jnp.asarray((10,)),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.ones((1, 4), dtype=jnp.int32),
        jnp.asarray(((0, -1), (0, -1), (0, -1), (0, -1))),
        target_area,
        area_stiffness,
        target_perimeter,
        perimeter_stiffness,
        active_contractility=active_contractility,
        interface_tension=interface_tension,
        cell_traction=cell_traction,
        field_names=field_names,
    )
    return plan.prepare(positions)


def _two_square_tissue():
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
    plan = polygonal_vertex_tissue_plan(
        jnp.arange(6),
        jnp.arange(7),
        jnp.asarray(((0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0), (1, 4))),
        jnp.asarray((10, 11)),
        jnp.asarray(((0, 6, 4, 5), (1, 2, 3, 6))),
        jnp.asarray(((1, 1, 1, 1), (1, 1, 1, -1))),
        jnp.asarray(((0, -1), (1, -1), (1, -1), (1, -1), (0, -1), (0, -1), (0, 1))),
        jnp.ones((2,)),
        jnp.zeros((2,)),
        jnp.full((2,), 4.0),
        jnp.zeros((2,)),
        cell_types=jnp.asarray((0, 1)),
        adhesion_matrix=jnp.asarray(((0.0, 0.25), (0.25, 0.0))),
    )
    return plan.prepare(positions)


def _tetrahedral_tissue():
    positions = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    plan = polyhedral_vertex_tissue_plan(
        jnp.arange(4),
        jnp.arange(6),
        jnp.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))),
        jnp.arange(4),
        jnp.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))),
        jnp.asarray((20,)),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.ones((1, 4), dtype=jnp.int32),
        jnp.asarray(((0, -1), (0, -1), (0, -1), (0, -1))),
        1.0 / 6.0,
        5.0,
        0.5 * (3.0 + jnp.sqrt(3.0)),
        2.0,
    )
    return plan.prepare(positions)


def test_regular_polygon_and_polyhedron_have_target_energy():
    square = _square_tissue()
    square_evaluation = square.evaluate(square.initialize_state())
    tetrahedron = _tetrahedral_tissue()
    tetrahedron_evaluation = tetrahedron.evaluate(tetrahedron.initialize_state())

    square_clone = _square_tissue()
    assert square_evaluation.valid
    assert square_evaluation.cell_measure[0] == pytest.approx(1.0)
    assert square_evaluation.boundary_measure[0] == pytest.approx(4.0)
    assert square_evaluation.potential_energy == pytest.approx(0.0, abs=2.0e-6)
    assert square.plan.plan_id == square_clone.plan.plan_id
    assert square.prepared_id == square_clone.prepared_id
    assert tetrahedron_evaluation.valid
    assert tetrahedron_evaluation.cell_measure[0] == pytest.approx(1.0 / 6.0)
    assert tetrahedron_evaluation.boundary_measure[0] == pytest.approx(
        0.5 * (3.0 + 3.0**0.5)
    )
    assert tetrahedron_evaluation.potential_energy == pytest.approx(0.0, abs=2.0e-6)


def test_line_tension_and_contractility_contribute_declared_scalar_energy():
    tissue = _square_tissue(interface_tension=0.25, active_contractility=0.125)
    evaluation = tissue.evaluate(tissue.initialize_state())

    assert evaluation.interface_energy == pytest.approx(1.0)
    assert evaluation.contractile_energy == pytest.approx(1.0)
    assert evaluation.potential_energy == pytest.approx(2.0)


def test_cell_traction_is_distributed_as_a_cell_resultant():
    tissue = _square_tissue(cell_traction=jnp.asarray(((2.0, -1.0),)))
    evaluation = tissue.evaluate(tissue.initialize_state())

    assert evaluation.valid
    assert jnp.allclose(evaluation.active_forces, jnp.asarray(((0.5, -0.25),) * 4))
    assert jnp.allclose(
        jnp.sum(evaluation.active_forces, axis=0), jnp.asarray((2.0, -1.0))
    )


def test_conservative_force_matches_centered_finite_difference():
    tissue = _square_tissue(target_area=0.82, target_perimeter=3.7)
    positions = tissue.reference_positions.at[2].set(jnp.asarray((1.15, 0.92)))
    state = VertexTissueState(positions, jnp.zeros((1, 0)), 0.0, tissue.prepared_id)
    evaluation = tissue.evaluate(state)
    step = 2.0e-4
    finite_difference = jnp.zeros_like(positions)
    for vertex in range(positions.shape[0]):
        for component in range(2):
            perturbation = jnp.zeros_like(positions).at[vertex, component].set(step)
            derivative = (
                vertex_tissue_potential_energy(tissue, positions + perturbation)
                - vertex_tissue_potential_energy(tissue, positions - perturbation)
            ) / (2.0 * step)
            finite_difference = finite_difference.at[vertex, component].set(-derivative)

    assert evaluation.valid
    assert jnp.allclose(
        evaluation.conservative_forces, finite_difference, rtol=3.0e-3, atol=3.0e-3
    )


def test_energy_is_translation_and_rotation_invariant():
    tissue = _square_tissue(
        target_area=0.8,
        target_perimeter=3.5,
        active_contractility=0.2,
    )
    positions = tissue.reference_positions.at[2].set(jnp.asarray((1.12, 0.91)))
    angle = jnp.asarray(0.63)
    rotation = jnp.asarray(
        ((jnp.cos(angle), -jnp.sin(angle)), (jnp.sin(angle), jnp.cos(angle)))
    )
    moved = positions @ rotation.T + jnp.asarray((3.0, -1.5))
    original_state = VertexTissueState(
        positions, jnp.zeros((1, 0)), 0.0, tissue.prepared_id
    )
    moved_state = VertexTissueState(moved, jnp.zeros((1, 0)), 0.0, tissue.prepared_id)
    original_evaluation = tissue.evaluate(original_state)
    moved_evaluation = tissue.evaluate(moved_state)

    assert vertex_tissue_potential_energy(tissue, moved) == pytest.approx(
        vertex_tissue_potential_energy(tissue, positions), rel=2.0e-6, abs=2.0e-6
    )
    assert jnp.allclose(
        moved_evaluation.conservative_forces,
        original_evaluation.conservative_forces @ rotation.T,
        rtol=3.0e-6,
        atol=3.0e-6,
    )


def test_cell_type_adhesion_routes_only_the_shared_interface():
    tissue = _two_square_tissue()
    evaluation = tissue.evaluate(tissue.initialize_state())

    assert evaluation.adhesion_routed_tension[-1] == pytest.approx(-0.25)
    assert jnp.allclose(evaluation.adhesion_routed_tension[:-1], 0.0)
    assert evaluation.interface_energy == pytest.approx(-0.25)


def test_overdamped_step_dissipates_passive_energy():
    tissue = _square_tissue(target_area=0.72, target_perimeter=3.55)
    state = tissue.initialize_state()
    dynamics = VertexTissueDynamicsPlan(1.0e-3, maximum_displacement=0.1).prepare(tissue)
    result = dynamics.step(state)

    assert result.accepted
    assert result.dissipation_rate > 0.0
    assert result.energy_descent
    assert result.energy_change < 0.0
    assert result.after.potential_energy < result.before.potential_energy


def test_nonfinite_state_fails_closed_with_zero_loads():
    tissue = _square_tissue()
    positions = tissue.reference_positions.at[0, 0].set(jnp.nan)
    state = VertexTissueState(positions, jnp.zeros((1, 0)), 0.0, tissue.prepared_id)
    evaluation = tissue.evaluate(state)

    assert not evaluation.finite
    assert not evaluation.valid
    assert jnp.all(evaluation.total_forces == 0.0)


def test_cell_field_and_particle_coupling_is_force_conservative():
    tissue = _square_tissue(field_names=("signal", "mass"))
    state = tissue.initialize_state(jnp.asarray(((2.0, 3.0),)))
    coupling = couple_vertex_tissue_particles(
        tissue,
        state,
        jnp.asarray((0, 0, -1)),
        jnp.asarray(((1.0, 0.0), (0.0, 2.0), (0.0, 0.0))),
    )
    vertex_fields = tissue.interpolate_cell_fields(state)
    spread = tissue.spread_vertex_field_sources(jnp.ones((4, 2)))

    assert coupling.valid
    assert jnp.allclose(coupling.particle_fields[:2], jnp.asarray(((2.0, 3.0),) * 2))
    assert jnp.allclose(jnp.sum(coupling.vertex_forces, axis=0), jnp.asarray((1.0, 2.0)))
    assert coupling.force_conservation_residual == pytest.approx(0.0)
    assert jnp.allclose(vertex_fields, jnp.asarray(((2.0, 3.0),) * 4))
    assert jnp.allclose(spread, jnp.asarray(((4.0, 4.0),)))


def test_polygonal_orientation_and_undirected_edge_uniqueness_are_enforced():
    with pytest.raises(ValueError, match="traverse"):
        polygonal_vertex_tissue_plan(
            jnp.arange(4),
            jnp.arange(4),
            jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0))),
            jnp.asarray((10, 11)),
            jnp.asarray(((0, 1, 2, 3), (0, 1, 2, 3))),
            jnp.ones((2, 4), dtype=jnp.int32),
            jnp.asarray(((0, 1), (0, 1), (0, 1), (0, 1))),
            jnp.ones((2,)),
            jnp.ones((2,)),
            jnp.full((2,), 4.0),
            jnp.ones((2,)),
        )

    with pytest.raises(ValueError, match="distinct undirected"):
        polygonal_vertex_tissue_plan(
            jnp.arange(4),
            jnp.arange(5),
            jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0), (1, 0))),
            jnp.asarray((10,)),
            jnp.asarray(((0, 1, 2, 3),)),
            jnp.ones((1, 4), dtype=jnp.int32),
            jnp.asarray(((0, -1),) * 5),
            1.0,
            1.0,
            4.0,
            1.0,
        )


def _concave_prism_tissue():
    base = (
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (0.0, 2.0),
    )
    positions = jnp.asarray(
        tuple((x, y, 0.0) for x, y in base) + tuple((x, y, 1.0) for x, y in base)
    )
    edges = []
    for offset in (0, 6):
        edges.extend((offset + index, offset + (index + 1) % 6) for index in range(6))
    edges.extend((index, index + 6) for index in range(6))
    faces = [
        (5, 4, 3, 2, 1, 0),
        (6, 7, 8, 9, 10, 11),
    ]
    faces.extend(
        (index, (index + 1) % 6, (index + 1) % 6 + 6, index + 6) for index in range(6)
    )
    face_rows = jnp.full((8, 6), -1, dtype=jnp.int32)
    for face, vertices in enumerate(faces):
        face_rows = face_rows.at[face, : len(vertices)].set(
            jnp.asarray(vertices, dtype=jnp.int32)
        )
    plan = polyhedral_vertex_tissue_plan(
        jnp.arange(12),
        jnp.arange(18),
        jnp.asarray(edges),
        jnp.arange(8),
        face_rows,
        jnp.asarray((20,)),
        jnp.arange(8, dtype=jnp.int32)[None, :],
        jnp.ones((1, 8), dtype=jnp.int32),
        jnp.asarray(((0, -1),) * 8),
        3.0,
        1.0,
        14.0,
        1.0,
    )
    return plan.prepare(positions)


def test_concave_polyhedral_faces_use_oriented_polygon_area():
    tissue = _concave_prism_tissue()
    evaluation = tissue.evaluate(tissue.initialize_state())

    assert evaluation.valid
    assert evaluation.cell_measure[0] == pytest.approx(3.0)
    assert evaluation.boundary_measure[0] == pytest.approx(14.0)
    assert evaluation.potential_energy == pytest.approx(0.0, abs=2.0e-6)


def test_duplicate_polyhedral_faces_are_rejected():
    with pytest.raises(ValueError, match="distinct polygonal"):
        polyhedral_vertex_tissue_plan(
            jnp.arange(4),
            jnp.arange(6),
            jnp.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))),
            jnp.arange(5),
            jnp.asarray(
                (
                    (1, 2, 3),
                    (0, 3, 2),
                    (0, 1, 3),
                    (0, 2, 1),
                    (2, 3, 1),
                )
            ),
            jnp.asarray((20,)),
            jnp.asarray(((0, 1, 2, 3),)),
            jnp.ones((1, 4), dtype=jnp.int32),
            jnp.asarray(((0, -1),) * 5),
            1.0 / 6.0,
            1.0,
            1.0,
            1.0,
        )
