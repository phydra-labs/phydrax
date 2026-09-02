#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


def _compile_liquid(*, initial_film=5.0e-4, evaporation_flux=0.0):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([10, 20]), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    bridge = phx.discretization.BagheriCapillaryBridgePlan(
        0.072,
        0.1,
        1.0e-3,
        conserve_liquid=True,
    )
    liquid = phx.discretization.ConservedLiquidBridgeProcessPlan(
        initial_film,
        evaporation_flux=evaporation_flux,
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
            cohesion=bridge,
        ),
        liquid_process=liquid,
        maximum_overlap_fraction=0.3,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "wet-granular", materials, gravity=jnp.zeros((2,))
    )
    return phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )


def test_bridge_birth_draws_from_films_and_rupture_returns_liquid():
    compiled = _compile_liquid()
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.99, 0.0]]),
        jnp.zeros((2, 2)),
    )
    component = state.particle_history.cohesion.components[0]
    assert component.active[0]
    assert jnp.isclose(component.bridge_volume[0], 1.0e-3)
    assert jnp.allclose(state.liquid.film_volume, 0.0)
    assert jnp.abs(state.liquid.balance_residual) < 1.0e-12

    characteristic_radius = 0.5
    dimensionless_volume = 1.0e-3 / characteristic_radius**3
    critical_gap = (
        characteristic_radius
        * (1.0 + 0.05)
        * (
            dimensionless_volume ** (1.0 / 3.0)
            + 0.1 * dimensionless_volume ** (2.0 / 3.0)
        )
    )
    position = state.kinematics.position.at[1, 0].set(1.01 + critical_gap)
    separated = eqx.tree_at(lambda value: value.kinematics.position, state, position)
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), separated, jnp.asarray(0.0), None
    )
    assert evaluation.successful
    assert evaluation.particle_contact.cohesion_ruptures[0]
    assert jnp.allclose(evaluation.liquid.next_state.film_volume, 5.0e-4)
    assert jnp.isclose(jnp.sum(evaluation.liquid.bridge_volume), 0.0)
    assert jnp.abs(evaluation.liquid.next_state.balance_residual) < 1.0e-12


def test_limited_inventory_allocates_deterministically_without_creating_liquid():
    compiled = _compile_liquid(initial_film=2.0e-4)
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.99, 0.0]]),
        jnp.zeros((2, 2)),
    )
    component = state.particle_history.cohesion.components[0]
    assert component.active[0]
    assert jnp.isclose(component.bridge_volume[0], 4.0e-4)
    assert jnp.allclose(state.liquid.film_volume, 0.0)
    assert jnp.isclose(state.liquid.initial_total_volume, 4.0e-4)
    assert jnp.abs(state.liquid.balance_residual) < 1.0e-12


def test_surface_area_evaporation_is_conservative_and_replay_safe():
    compiled = _compile_liquid(evaporation_flux=1.0e6)
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.99, 0.0]]),
        jnp.zeros((2, 2)),
    )
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-5),
        None,
    )
    next_component = detail.accepted_state.particle_history.cohesion.components[0]
    assert detail.successful
    assert detail.evaluation.particle_contact.bridge_surface_area[0] > 0.0
    assert jnp.isclose(
        detail.evaluation.particle_contact.bridge_evaporation_loss[0], 1.0e-3
    )
    assert detail.evaluation.liquid.evaporated_ruptures[0]
    assert not next_component.active[0]
    assert jnp.isclose(detail.accepted_state.liquid.cumulative_evaporated_volume, 1.0e-3)
    assert jnp.abs(detail.accepted_state.liquid.balance_residual) < 1.0e-12
    assert jnp.abs(detail.evaluation.diagnostics.liquid_balance_residual) < 1.0e-12

    replay = phx.discretization.checkpointed_dem_rollout(
        compiled.dynamics,
        state,
        t0=0.0,
        step_size=1.0e-5,
        step_count=1,
        checkpoint=phx.discretization.DEMCheckpointPolicy(1),
    )
    assert replay.successful
    assert jnp.abs(replay.final_state.liquid.balance_residual) < 1.0e-12


def test_barrier_reservoir_allocation_is_permutation_independent_and_balanced():
    barrier = phx.discretization.DEMBarrierCapillaryPlan(
        "wall",
        geometry_policy="planar",
        particle_liquid_fraction=0.5,
        initial_barrier_film_volume=1.0,
    )
    process = phx.discretization.ConservedLiquidBridgeProcessPlan(
        jnp.asarray((0.5, 0.5)),
        barrier_capillaries=(barrier,),
    )
    state = process.initialize(2, jnp.float64, jnp.asarray((True, True)))
    particle = jnp.asarray((0, 1), dtype=jnp.int32)
    barriers = jnp.asarray((0, 0), dtype=jnp.int32)
    request = jnp.asarray((1.0, 1.0))
    minimum = jnp.zeros((2,))
    births = jnp.ones((2,), dtype=bool)
    allocation = process.allocate_barriers(
        state,
        particle,
        barriers,
        request,
        minimum,
        births,
        2,
    )
    permuted = process.allocate_barriers(
        state,
        particle[::-1],
        barriers[::-1],
        request[::-1],
        minimum[::-1],
        births[::-1],
        2,
    )

    assert bool(allocation.successful)
    assert bool(permuted.successful)
    assert jnp.allclose(
        jnp.sort(allocation.bridge_volume),
        jnp.sort(permuted.bridge_volume),
    )
    evaluation = process.advance_barriers(
        state,
        allocation,
        particle,
        barriers,
        allocation.bridge_volume,
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        minimum,
        jnp.asarray(1.0),
        2,
    )
    assert bool(evaluation.successful)
    assert jnp.abs(evaluation.next_state.balance_residual) < 1.0e-12
