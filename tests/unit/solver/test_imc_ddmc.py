#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def test_imc_absorption_commits_equal_packet_to_material_energy():
    plan = phx.solver.HybridIMCDDMCPlan(
        jnp.asarray((1.0, 1.0)),
        jnp.full((2, 1), 1.0e-3),
        jnp.zeros((2, 1)),
        jnp.full((2,), 1000.0),
        maximum_packet_energy=1000.0,
    )
    state = plan.initialize(
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((100.0,)),
        jnp.asarray((True,)),
        jnp.asarray((300.0, 300.0)),
    )
    result = plan.advance(state, 1.0e-3, jr.key(1))

    assert bool(result.successful)
    np.testing.assert_allclose(result.evidence.energy_residual, 0.0, atol=1e-9)
    np.testing.assert_allclose(
        jnp.sum(result.accepted.material_energy) + jnp.sum(result.accepted.packet_energy),
        jnp.sum(state.material_energy) + jnp.sum(state.packet_energy),
    )
    assert result.accepted.material_energy[0] >= state.material_energy[0]


def test_ddmc_classification_and_failed_step_rollback_are_explicit():
    plan = phx.solver.HybridIMCDDMCPlan(
        jnp.asarray((1.0,)),
        jnp.asarray(((1.0e-8,),)),
        jnp.asarray(((10.0,),)),
        jnp.asarray((1000.0,)),
        maximum_packet_energy=1000.0,
        ddmc_optical_depth=3.0,
    )
    state = plan.initialize(
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((100.0,)),
        jnp.asarray((True,)),
        jnp.asarray((300.0,)),
    )
    result = plan.advance(state, 1.0e-9, jr.key(2))
    rejected = plan.advance(state, -1.0, jr.key(3))

    assert bool(result.successful)
    assert int(result.evidence.ddmc_packet_count) == 1
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.accepted.packet_energy, state.packet_energy)
    np.testing.assert_array_equal(
        rejected.accepted.material_energy, state.material_energy
    )
    np.testing.assert_array_equal(rejected.accepted.time, state.time)
