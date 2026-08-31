#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


def _compiled_dem(
    *,
    barriers=(),
    friction=0.5,
    restitution=0.8,
    fixed_mask=None,
    gravity=None,
    neighborhood=None,
    execution=None,
):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]),
        jnp.asarray([1.0, 1.0]),
        ambient_dimension=2,
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([0, 0]),
        fixed_mask=fixed_mask,
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[restitution]]),
        jnp.asarray([[friction]]),
    )
    contact = phx.discretization.DEMContactModelPlan(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
        tangential=phx.discretization.CundallStrackTangentialPlan(2.5e3),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        contact, maximum_overlap_fraction=0.25
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "two-sphere-dem",
        materials,
        gravity=jnp.zeros((2,)) if gravity is None else gravity,
        barriers=barriers,
    )
    neighborhood_ = (
        phx.discretization.DenseParticleNeighborhoodPlan(1)
        if neighborhood is None
        else neighborhood
    )
    return phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=neighborhood_,
        execution=execution,
    )


def test_linear_sphere_contact_has_action_reaction_and_torque_balance():
    compiled = _compiled_dem()
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.9, 0.0]]),
        jnp.zeros((2, 2)),
    )
    diagnostics = compiled.diagnostics(0.0, state)

    assert state.loads.total.force[0, 0] < 0.0
    assert state.loads.total.force[1, 0] > 0.0
    assert jnp.allclose(jnp.sum(state.loads.total.force, axis=0), 0.0, atol=1.0e-12)
    assert jnp.allclose(diagnostics.net_internal_force, 0.0, atol=1.0e-12)
    assert jnp.allclose(diagnostics.net_internal_torque, 0.0, atol=1.0e-12)
    assert diagnostics.active_contacts == 1
    assert jnp.isclose(diagnostics.maximum_overlap_fraction, 0.2)


def test_cundall_strack_history_advances_and_respects_coulomb_limit():
    compiled = _compiled_dem(friction=0.2)
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.9, 0.0]]),
        jnp.asarray([[0.0, 1.0], [0.0, 0.0]]),
    )
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-4),
        None,
    )
    response = detail.evaluation.particle_contact
    tangential_norm = jnp.linalg.norm(response.tangential_force[0])
    normal_norm = jnp.linalg.norm(response.normal_force[0])

    assert detail.successful
    assert (
        jnp.linalg.norm(
            detail.candidate_state.particle_history.tangential.displacement[0]
        )
        > 0.0
    )
    assert tangential_norm <= 0.2 * normal_norm + 1.0e-10
    assert response.friction_defect[0] <= 1.0e-10


def test_exact_sdf_container_returns_equal_opposite_wall_reaction():
    barrier = phx.discretization.ImplicitDEMBarrier(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile(),
        phx.discretization.DEMBarrierSide.INTERIOR,
        0,
        barrier_id="square-container",
    )
    compiled = _compiled_dem(barriers=(barrier,))
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.9, 0.0], [0.0, 0.0]]),
        jnp.zeros((2, 2)),
    )
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(0.0), None
    )
    boundary = evaluation.boundaries[0]

    assert evaluation.successful
    assert boundary.particle_force[0, 0] < 0.0
    assert jnp.allclose(
        boundary.reaction_force,
        -jnp.sum(boundary.particle_force, axis=0),
        atol=1.0e-12,
    )


def test_dem_qualification_uses_contact_specific_residuals_and_margins():
    compiled = _compiled_dem()
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.9, 0.0]]),
        jnp.zeros((2, 2)),
    )
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(0.0), None
    )
    residuals = phx.discretization.dem_constraint_residuals(evaluation.diagnostics)
    profile = phx.discretization.DEMQualificationProfile(maximum_overlap_fraction=0.25)
    margins = phx.discretization.dem_differentiability_margins(evaluation.diagnostics)

    assert profile.constraints_satisfied(residuals)
    assert margins.contact_activation > 0.0
    assert margins.route_capacity_successful


def test_periodic_dense_and_cell_contacts_have_identical_force_and_history():
    box = phx.discretization.ParticleBox(
        jnp.asarray([0.0, -1.0]),
        jnp.asarray([2.0, 1.0]),
        periodic_axes=(True, False),
    )
    dense = _compiled_dem(
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1, box=box)
    )
    cell = _compiled_dem(
        neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(1.0, 2, 1, box),
        execution=phx.discretization.ParticleExecutionPolicy(
            realization="cell_edge_list", accumulation="deterministic"
        ),
    )
    position = jnp.asarray([[0.45, 0.0], [1.55, 0.0]])
    velocity = jnp.asarray([[0.0, 0.2], [0.0, -0.2]])

    dense_state = dense.initialize_state(0.0, position, velocity)
    cell_state = cell.initialize_state(0.0, position, velocity)
    dense_step = dense.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        dense_state,
        jnp.asarray(1.0e-5),
        None,
    )
    cell_step = cell.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        cell_state,
        jnp.asarray(1.0e-5),
        None,
    )

    assert dense_step.successful
    assert cell_step.successful
    assert jnp.allclose(
        dense_step.candidate_state.loads.total.force,
        cell_step.candidate_state.loads.total.force,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        dense_step.candidate_state.particle_history.tangential.displacement,
        cell_step.candidate_state.particle_history.tangential.displacement,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_cell_occupancy_overflow_rejects_initial_dem_state():
    box = phx.discretization.ParticleBox(
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        periodic_axes=(False, False),
    )
    compiled = _compiled_dem(
        neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(1.0, 1, 1, box),
        execution=phx.discretization.ParticleExecutionPolicy(
            realization="cell_edge_list", accumulation="deterministic"
        ),
    )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="Initial DEM contact state"
    ):
        compiled.initialize_state(
            0.0,
            jnp.asarray([[0.0, 0.0], [0.1, 0.0]]),
            jnp.zeros((2, 2)),
        )


def test_fixed_sphere_retains_reaction_while_mobile_sphere_follows_gravity():
    compiled = _compiled_dem(
        fixed_mask=jnp.asarray([True, False]),
        gravity=jnp.asarray([0.0, -1.0]),
    )
    initial_position = jnp.asarray([[-0.75, 0.0], [0.75, 0.0]])
    state = compiled.initialize_state(0.0, initial_position, jnp.zeros((2, 2)))
    step_size = jnp.asarray(1.0e-3)
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        step_size,
        None,
    )

    assert detail.successful
    assert jnp.array_equal(
        detail.candidate_state.kinematics.position[0], initial_position[0]
    )
    assert jnp.array_equal(detail.candidate_state.kinematics.velocity[0], jnp.zeros((2,)))
    assert jnp.allclose(
        detail.candidate_state.kinematics.position[1],
        initial_position[1] + jnp.asarray([0.0, -0.5e-6]),
        atol=1.0e-15,
    )
    assert jnp.allclose(
        detail.candidate_state.kinematics.velocity[1],
        jnp.asarray([0.0, -1.0e-3]),
        atol=1.0e-15,
    )
