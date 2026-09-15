#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_weighted_moments_and_block_uncertainty_are_physical():
    species = phx.discretization.dsmc.DSMCSpeciesPlan(
        ("A",),
        jnp.asarray((2.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
    )
    cells = phx.discretization.dsmc.DSMCStructuredCellPlan(
        jnp.asarray((0.0,)), jnp.asarray((1.0,)), (1,)
    )
    plan = phx.discretization.dsmc.DSMCMomentPlan(
        species,
        cells,
        minimum_particles_per_cell=2,
        block_size=1,
        minimum_blocks=2,
        maximum_relative_standard_error=1.0,
    )
    particles = phx.discretization.dsmc.DSMCParticleState(
        jnp.asarray(((0.25,), (0.75,))),
        jnp.asarray(((1.0,), (-1.0,))),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.asarray((3.0, 3.0)),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((True, True)),
        jnp.zeros((2,), dtype=jnp.int32),
    )

    moments = plan.evaluate(particles)
    np.testing.assert_allclose(moments.number_density, 6.0)
    np.testing.assert_allclose(moments.mass_density, 12.0)
    np.testing.assert_allclose(moments.velocity, 0.0)
    np.testing.assert_allclose(moments.pressure_tensor, [[[12.0]]])
    assert bool(moments.header.globally_eligible)

    accumulator = plan.initialize_accumulator()
    first = plan.accumulate(accumulator, moments)
    assert not bool(plan.statistical_evidence(first).header.globally_eligible)
    second = plan.accumulate(first, moments)
    evidence = plan.statistical_evidence(second)
    assert bool(evidence.header.globally_eligible)
    np.testing.assert_allclose(evidence.covariance, 0.0)
