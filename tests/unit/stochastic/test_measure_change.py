#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_diffusion_girsanov_recovers_shifted_gaussian_expectation():
    num_paths = 32_768
    drift_shift = 0.7
    increments = jr.normal(jr.key(41), (num_paths, 1, 1))
    controls = jnp.full_like(increments, drift_shift)

    change = eqx.filter_jit(phx.stochastic.diffusion_measure_change)(
        controls,
        increments,
        jnp.asarray([1.0]),
        proposal_model_id="zero-drift",
        target_model_id="shifted-drift",
    )
    target = phx.stochastic.measure_changed_target(
        increments[:, 0, 0],
        change,
        independent=True,
    )
    estimate = phx.integration.integrate(lambda value: value, target)

    assert change.kind == "diffusion"
    assert change.proposal_model_id == "zero-drift"
    assert jnp.allclose(
        change.log_likelihood_ratio,
        drift_shift * increments[:, 0, 0] - 0.5 * drift_shift**2,
    )
    assert estimate.successful
    assert jnp.allclose(estimate.value.data, drift_shift, atol=0.03)
    assert jnp.allclose(estimate.diagnostics.normalizer_estimate, 1.0, atol=0.03)


def test_diffusion_measure_change_rejects_partial_invalid_paths():
    controls = jnp.ones((2, 2, 1))
    increments = jnp.ones((2, 2, 1))
    valid = jnp.asarray([[True, True], [True, False]])

    change = phx.stochastic.diffusion_measure_change(
        controls,
        increments,
        jnp.asarray([0.5, 0.5]),
        valid=valid,
    )

    assert jnp.array_equal(change.valid, jnp.asarray([True, False]))
    assert jnp.isneginf(change.log_likelihood_ratio[1])


def _jump_events():
    return phx.stochastic.JumpEventBatch(
        jnp.asarray([[0.0, 0.0], [0.2, 0.7], [0.5, 0.0]]),
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.zeros((3, 2)),
        jnp.asarray([[False, False], [True, True], [True, False]]),
        jnp.zeros((3,), dtype=jnp.int32),
    )


def test_jump_measure_change_matches_poisson_likelihood_ratio():
    events = _jump_events()
    proposal = jnp.full((3, 1, 1), 2.0)
    target = jnp.full((3, 1, 1), 3.0)

    change = eqx.filter_jit(phx.stochastic.jump_measure_change)(
        events,
        jnp.asarray([0.0, 1.0]),
        proposal,
        target,
        proposal_model_id="rate-two",
        target_model_id="rate-three",
    )
    expected = jnp.asarray([0.0, 2.0, 1.0]) * jnp.log(1.5) - 1.0

    assert change.kind == "jump"
    assert jnp.allclose(change.log_likelihood_ratio, expected)
    assert jnp.allclose(change.compensator, 1.0)
    assert jnp.all(change.valid)
    assert jnp.all(change.support_valid)


def test_jump_measure_change_reports_support_failure_and_zero_target_density():
    events = _jump_events()
    proposal = jnp.full((3, 1, 1), 2.0)
    target = jnp.zeros((3, 1, 1))
    zero_density = phx.stochastic.jump_measure_change(
        events,
        jnp.asarray([0.0, 1.0]),
        proposal,
        target,
    )
    unsupported = phx.stochastic.jump_measure_change(
        events,
        jnp.asarray([0.0, 1.0]),
        jnp.zeros_like(proposal),
        jnp.ones_like(target),
    )

    assert zero_density.log_likelihood_ratio[0] == 2.0
    assert jnp.isneginf(zero_density.log_likelihood_ratio[1])
    assert jnp.all(zero_density.support_valid)
    assert not jnp.any(unsupported.support_valid)
    weighted = phx.stochastic.measure_changed_target(
        jnp.arange(3.0),
        unsupported,
        independent=True,
    )
    estimate = phx.integration.integrate(lambda value: value, weighted)
    assert estimate.status == int(
        phx.integration.IntegrationStatus.PROPOSAL_SUPPORT_FAILURE
    )
