#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_continuum_conversion_matches_extensive_moments_exactly():
    plan = phx.solver.ContinuumToDSMCConversionPlan(8, 3)
    result = plan.convert(
        jax.random.key(1),
        jnp.asarray(10.0),
        jnp.asarray((2.0, -1.0, 0.5)),
        jnp.asarray(5.0),
        jnp.asarray(2.0),
    )

    assert bool(result.header.globally_eligible)
    np.testing.assert_allclose(result.represented_mass, 10.0, atol=1.0e-12)
    np.testing.assert_allclose(
        result.represented_momentum, (20.0, -10.0, 5.0), atol=1.0e-12
    )
    np.testing.assert_allclose(result.represented_thermal_energy, 5.0, atol=1.0e-12)


def test_particle_reduction_preserves_species_mass_momentum_energy_and_covariance():
    species = phx.discretization.dsmc.DSMCSpeciesPlan(
        ("A", "B"),
        jnp.asarray((1.0, 2.0)),
        jnp.ones((2,)),
        jnp.full((2,), 0.5),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
    )
    particles = phx.discretization.dsmc.DSMCParticleState(
        jnp.zeros((3, 2)),
        jnp.asarray(((1.0, 0.0), (0.0, 1.0), (9.0, 9.0))),
        jnp.asarray((0, 1, 0), dtype=jnp.int32),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.asarray((2.0, 3.0, 0.0)),
        jnp.asarray((0, 0, -1), dtype=jnp.int32),
        jnp.asarray((True, True, False)),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    result = phx.solver.DSMCToContinuumReductionPlan(
        species, 2, minimum_particles=2
    ).reduce(particles)

    assert bool(result.header.globally_eligible)
    np.testing.assert_allclose(result.species_mass, (2.0, 6.0))
    np.testing.assert_allclose(result.momentum, (2.0, 6.0))
    np.testing.assert_allclose(result.total_energy, 4.0)
    assert result.covariance.shape == (5, 5)


def test_hybrid_epoch_refuses_failed_required_conversion():
    plan = phx.solver.HybridOwnershipEpochPlan(
        enter_threshold=0.05,
        leave_threshold=0.02,
        minimum_dwell_steps=0,
        buffer_layers=0,
    )
    state = plan.initialize(jnp.asarray((False,)))
    admitted = phx.AdmissibilityHeader(
        jnp.asarray((1.0,)),
        jnp.zeros((1,), dtype=jnp.uint32),
        "breakdown",
        "breakdown-evidence",
    )
    request = plan.classify(
        state,
        jnp.asarray((0.1,)),
        admitted,
        jnp.asarray(((False,),)),
        2,
        2,
    )
    failed = phx.AdmissibilityHeader(
        jnp.asarray(-1.0),
        jnp.asarray(int(phx.AdmissibilityReason.OUTSIDE_SUPPORT), dtype=jnp.uint32),
        "conversion",
        "conversion-evidence",
    )
    neutral = phx.AdmissibilityHeader(
        jnp.asarray(1.0),
        jnp.asarray(0, dtype=jnp.uint32),
        "reduction",
        "reduction-evidence",
    )

    with pytest.raises(ValueError, match="conversion"):
        plan.transition_epoch(state, request, failed, neutral)
