#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.sampling._exact_learned import (
    DelayedAcceptanceHMCPlan,
    freeze_learned_coarse_space,
    freeze_learned_metric,
    GaugeFlowProposalPlan,
    initialize_delayed_acceptance_hmc,
    initialize_gauge_flow_chain,
    LearnedSupportTuple,
    prepare_delayed_acceptance_hmc,
    prepare_gauge_flow_proposal,
    sample_delayed_acceptance_hmc,
    sample_gauge_flow_proposal,
    ScalarGaugeEquivariantFlow,
)


def _support():
    return LearnedSupportTuple(
        target_id="standard-normal",
        geometry_id="euclidean-r1",
        configuration_shape=(1,),
        coordinate_dtype="float32",
        parameter_names=("coupling",),
        parameter_lower=jnp.asarray([0.0]),
        parameter_upper=jnp.asarray([1.0]),
    )


def test_exact_delayed_acceptance_invariance_survives_bad_surrogate():
    support = _support()
    metric = freeze_learned_metric(jnp.eye(1, dtype=jnp.float32), support)
    plan = DelayedAcceptanceHMCPlan(support, step_size=0.35, leapfrog_steps=4)
    kernel = prepare_delayed_acceptance_hmc(
        plan,
        lambda x: -0.5 * jnp.sum(x**2),
        lambda x: -0.02 * jnp.sum((x - 5.0) ** 2),
        metric,
        target_id="standard-normal",
        surrogate_id="deliberately-bad-shifted-wide-normal",
        geometry_id="euclidean-r1",
        parameter_values=jnp.asarray([0.5]),
    )
    key = jax.random.key(11)
    initial = jax.random.normal(jax.random.fold_in(key, 1), (2048, 1))
    state = initialize_delayed_acceptance_hmc(kernel, initial)
    result = sample_delayed_acceptance_hmc(
        kernel, state, key=jax.random.fold_in(key, 2), num_draws=4
    )
    terminal = result.samples[:, -1, 0]

    assert jnp.abs(jnp.mean(terminal)) < 0.08
    assert jnp.abs(jnp.mean(terminal**2) - 1.0) < 0.12
    assert jnp.any(result.surrogate_accepted)
    assert jnp.any(result.surrogate_accepted & ~result.accepted)
    assert result.claim == "exact-target-delayed-acceptance-with-frozen-surrogate"


def test_learned_artifacts_are_frozen_and_out_of_domain_is_refused():
    support = _support()
    metric = freeze_learned_metric(jnp.eye(1, dtype=jnp.float32), support)
    coarse = freeze_learned_coarse_space(jnp.ones((1, 1), dtype=jnp.float32), support)
    assert metric.frozen and coarse.frozen
    assert metric.artifact_id != coarse.artifact_id
    coarse_value = coarse.restrict(jnp.asarray([2.0], dtype=jnp.float32))
    assert jnp.allclose(coarse.prolong(coarse_value), jnp.asarray([2.0]))
    with pytest.raises(AttributeError):
        metric.mass_matrix = jnp.asarray([[2.0]], dtype=jnp.float32)

    plan = DelayedAcceptanceHMCPlan(support, step_size=0.1, leapfrog_steps=2)
    with pytest.raises(ValueError, match="out-of-domain"):
        prepare_delayed_acceptance_hmc(
            plan,
            lambda x: -jnp.sum(x**2),
            lambda x: -jnp.sum(x**2),
            metric,
            target_id="standard-normal",
            surrogate_id="same",
            geometry_id="euclidean-r1",
            parameter_values=jnp.asarray([1.5]),
        )


def test_nontrivial_gauge_flow_reports_jacobian_and_uses_exact_mh_ratio():
    support = _support()
    scale = 1.7
    flow = ScalarGaugeEquivariantFlow(support, scale=scale)
    plan = GaugeFlowProposalPlan(support, equivariance_tolerance=1e-6)
    proposal = prepare_gauge_flow_proposal(
        plan,
        flow,
        lambda x: -0.5 * jnp.sum(x**2),
        lambda key: jax.random.normal(key, (1,), dtype=jnp.float32),
        lambda x: -0.5 * jnp.sum(x**2),
        target_id="standard-normal",
        geometry_id="euclidean-r1",
        base_id="standard-normal-base",
        parameter_values=jnp.asarray([0.5]),
    )
    state = initialize_gauge_flow_chain(proposal, jnp.zeros((3, 1), dtype=jnp.float32))
    result = sample_gauge_flow_proposal(
        proposal, state, key=jax.random.key(9), num_draws=5
    )
    first_proposal = result.proposed_samples[:, 0, 0]
    expected_first_log_ratio = (
        0.5 * (first_proposal / scale) ** 2 - 0.5 * first_proposal**2
    )

    assert jnp.allclose(result.forward_log_abs_det_jacobian, jnp.log(scale), rtol=1e-6)
    assert jnp.allclose(result.inverse_log_abs_det_jacobian, -jnp.log(scale), rtol=1e-6)
    assert jnp.allclose(
        result.log_acceptance_ratio[:, 0], expected_first_log_ratio, rtol=1e-6
    )
    assert jnp.allclose(
        result.acceptance_probability[:, 0],
        jnp.exp(jnp.minimum(expected_first_log_ratio, 0.0)),
        rtol=1e-6,
    )
    assert jnp.all(result.proposal_valid)


def test_learned_sampler_states_reject_different_prepared_owners():
    support = _support()
    metric = freeze_learned_metric(jnp.eye(1, dtype=jnp.float32), support)
    hmc_plan = DelayedAcceptanceHMCPlan(support, step_size=0.1, leapfrog_steps=2)

    def hmc(surrogate_id):
        return prepare_delayed_acceptance_hmc(
            hmc_plan,
            lambda x: -0.5 * jnp.sum(x**2),
            lambda x: -0.5 * jnp.sum(x**2),
            metric,
            target_id="standard-normal",
            surrogate_id=surrogate_id,
            geometry_id="euclidean-r1",
            parameter_values=jnp.asarray([0.5]),
        )

    first_hmc = hmc("surrogate-a")
    second_hmc = hmc("surrogate-b")
    hmc_state = initialize_delayed_acceptance_hmc(
        first_hmc,
        jnp.zeros((1, 1), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="another prepared kernel"):
        sample_delayed_acceptance_hmc(
            second_hmc,
            hmc_state,
            key=jax.random.key(30),
            num_draws=1,
        )

    flow = ScalarGaugeEquivariantFlow(support, scale=1.1)
    flow_plan = GaugeFlowProposalPlan(support, equivariance_tolerance=1e-6)

    def proposal(base_id):
        return prepare_gauge_flow_proposal(
            flow_plan,
            flow,
            lambda x: -0.5 * jnp.sum(x**2),
            lambda key: jax.random.normal(key, (1,), dtype=jnp.float32),
            lambda x: -0.5 * jnp.sum(x**2),
            target_id="standard-normal",
            geometry_id="euclidean-r1",
            base_id=base_id,
            parameter_values=jnp.asarray([0.5]),
        )

    first_proposal = proposal("base-a")
    second_proposal = proposal("base-b")
    flow_state = initialize_gauge_flow_chain(
        first_proposal,
        jnp.zeros((1, 1), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="another prepared proposal"):
        sample_gauge_flow_proposal(
            second_proposal,
            flow_state,
            key=jax.random.key(31),
            num_draws=1,
        )
