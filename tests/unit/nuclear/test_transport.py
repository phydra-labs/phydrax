#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_slab_sn_fixed_source_is_positive_symmetric_and_converged() -> None:
    plan = phx.nuclear.SlabSNTransportPlan(
        jnp.full((8,), 0.125),
        jnp.full((8, 1), 1.0),
        jnp.full((8, 1, 1), 0.2),
        ordinates=8,
        relative_tolerance=1.0e-10,
    )

    result = plan.solve_fixed_source(jnp.ones((8, 1)))

    assert bool(result.converged)
    assert jnp.all(result.scalar_flux > 0.0)
    assert jnp.allclose(result.scalar_flux[:, 0], result.scalar_flux[::-1, 0], atol=1.0e-10)
    assert result.residual_norm < 1.0e-10


def test_bateman_depletion_conserves_two_member_chain() -> None:
    rate = 0.2
    plan = phx.nuclear.BatemanDepletionPlan(
        jnp.asarray([[-rate, 0.0], [rate, 0.0]])
    )

    result = plan.evolve(jnp.asarray([1.0, 0.0]), 3.0)

    expected_parent = jnp.exp(-rate * 3.0)
    assert jnp.allclose(result.inventory, jnp.asarray([expected_parent, 1.0 - expected_parent]), atol=1.0e-12)
    assert bool(result.nonnegative)
    assert result.conservation_defect < 1.0e-12


def test_decay_heat_uses_activity_times_recoverable_energy() -> None:
    heat = phx.nuclear.decay_heat_w(
        jnp.asarray((2.0, 3.0)),
        jnp.asarray((0.5, 0.25)),
        jnp.asarray((4.0, 8.0)),
    )

    assert jnp.isclose(heat, 10.0)
