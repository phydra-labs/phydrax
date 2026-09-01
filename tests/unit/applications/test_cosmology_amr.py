import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def test_two_level_prolong_restrict_average_down_and_reflux():
    plan = cosmology.TwoLevelAMRPlan((2,), 2)
    coarse = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    state = plan.initialize(coarse, jnp.asarray([True, False]), 0.5)
    assert state.fine_cell_average.shape == (4, 2)
    np.testing.assert_allclose(plan.restrict(state.fine_cell_average), coarse)
    modified_fine = state.fine_cell_average.at[:2].set(
        jnp.asarray([[2.0, 4.0], [2.0, 4.0]])
    )
    averaged = plan.average_down(
        cosmology.TwoLevelAMRState(
            coarse, modified_fine, state.refined_parent_mask, state.scale_factor
        )
    )
    np.testing.assert_allclose(averaged.coarse_cell_average[0], [2.0, 4.0])
    np.testing.assert_allclose(averaged.coarse_cell_average[1], coarse[1])

    register = cosmology.CoarseFineFluxRegister(
        jnp.asarray([[2.0, 4.0]]), jnp.asarray([[1.0, 2.0]])
    )
    refluxed = register.reflux(
        coarse,
        jnp.asarray([0]),
        jnp.asarray([1]),
        jnp.asarray([1.0, 1.0]),
    )
    np.testing.assert_allclose(refluxed, [[0.0, 0.0], [4.0, 6.0]])


def test_amr_particle_routing_and_atomic_epoch_commit():
    plan = cosmology.TwoLevelAMRPlan((2, 2), 1)
    mask = jnp.asarray([[True, False], [False, False]])
    previous = plan.initialize(jnp.ones((2, 2, 1)), mask, 0.5)
    candidate = cosmology.TwoLevelAMRState(
        previous.coarse_cell_average,
        2.0 * previous.fine_cell_average,
        mask,
        jnp.asarray(0.6),
    )
    routing = cosmology.TwoLevelParticleRoutingPlan(plan, (1.0, 1.0))
    particles = cosmology.CosmologicalParticleState(
        jnp.asarray([[0.1, 0.1], [0.8, 0.8]]),
        jnp.zeros((2, 2)),
        jnp.asarray(0.5),
    )
    candidate_particles = cosmology.CosmologicalParticleState(
        particles.positions,
        particles.canonical_momenta,
        jnp.asarray(0.6),
    )
    assignment = routing.route(particles.positions, mask)
    np.testing.assert_array_equal(assignment.levels, [1, 0])
    register = cosmology.CoarseFineFluxRegister(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    result = cosmology.TwoLevelAMREpochPlan(plan, routing).commit(
        previous,
        candidate,
        particles,
        candidate_particles,
        register,
        jnp.asarray(True),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.state.scale_factor, 0.6)
    np.testing.assert_allclose(result.particles.scale_factor, 0.6)
