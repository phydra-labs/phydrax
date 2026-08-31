#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _compiled_collision(*, restitution=1.0):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([10, 20]),
        jnp.asarray([1.0, 1.0]),
        ambient_dimension=2,
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[restitution]]),
        jnp.asarray([[0.0]]),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        ),
        maximum_overlap_fraction=0.1,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "elastic-collision",
        materials,
        gravity=jnp.zeros((2,)),
    )
    return phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )


def _solve(compiled, initial, t0, t1):
    problem = phx.solver.FixedStepProblem(
        phx.solver.DEMFixedStepMethod(compiled.dynamics),
        initial,
        t0=t0,
        t1=t1,
        step_size=1.0e-4,
        state_geometry=compiled.dynamics.state_geometry,
        discretization_bundle=compiled.discretization_bundle,
    )
    return phx.solver.solve_fixed_step(problem)


def test_dem_fixed_step_rollout_is_finite_conservative_and_restart_equivalent():
    compiled = _compiled_collision()
    initial = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.99, 0.0]]),
        jnp.asarray([[0.1, 0.0], [-0.1, 0.0]]),
    )

    direct = _solve(compiled, initial, 0.0, 0.001)
    first = _solve(compiled, initial, 0.0, 0.0005)
    midpoint = jax.tree.map(lambda leaf: leaf[-1], first.states)
    second = _solve(compiled, midpoint, 0.0005, 0.001)
    direct_final = jax.tree.map(lambda leaf: leaf[-1], direct.states)
    split_final = jax.tree.map(lambda leaf: leaf[-1], second.states)

    assert direct.successful
    assert first.successful
    assert second.successful
    assert jnp.allclose(
        direct_final.kinematics.position,
        split_final.kinematics.position,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        direct_final.kinematics.velocity,
        split_final.kinematics.velocity,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    initial_momentum = jnp.sum(initial.kinematics.velocity, axis=0)
    final_momentum = jnp.sum(direct_final.kinematics.velocity, axis=0)
    assert jnp.allclose(final_momentum, initial_momentum, atol=1.0e-10)


def test_damped_head_on_collision_recovers_requested_restitution():
    target_restitution = 0.8
    compiled = _compiled_collision(restitution=target_restitution)
    initial = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [1.02, 0.0]]),
        jnp.asarray([[0.5, 0.0], [-0.5, 0.0]]),
    )

    solution = _solve(compiled, initial, 0.0, 0.06)
    final = jax.tree.map(lambda leaf: leaf[-1], solution.states)
    separation_speed = final.kinematics.velocity[1, 0] - final.kinematics.velocity[0, 0]

    assert solution.successful
    assert final.kinematics.position[1, 0] - final.kinematics.position[0, 0] > 1.0
    assert jnp.isclose(separation_speed, target_restitution, atol=3.0e-2)
    assert final.energy.cumulative_contact_balance_loss > 0.0
