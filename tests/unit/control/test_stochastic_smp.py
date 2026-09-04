#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.control.stochastic._evaluation import (
    ControlledPathBatch,
    ControlledTransitionProblem,
    PreparedControlledNoise,
)
from phydrax.control.stochastic._smp import (
    evaluate_stochastic_maximum_principle,
    StochasticMaximumPrincipleProblem,
    StochasticMaximumPrincipleStatus,
)
from phydrax.dynamics import TimeGrid


def _paths(states, actions, noise, *, clusters=None, problem_id="smp-test"):
    states = jnp.asarray(states, dtype=float)
    actions = jnp.asarray(actions, dtype=float)
    noise = jnp.asarray(noise, dtype=float)
    count, steps = actions.shape[:2]
    grid = TimeGrid(jnp.arange(steps + 1, dtype=float), time_id="smp-grid")
    controlled = ControlledTransitionProblem(
        lambda context, state, action, increment, args: state,
        grid,
        states[0, 0],
        state_shape=states.shape[2:],
        action_shape=actions.shape[2:],
        noise_shape=noise.shape[2:],
        stage_cost=lambda context, state, action, args: 0.0,
        terminal_cost=lambda time, state, args: 0.0,
        problem_id=problem_id,
    )
    prepared = PreparedControlledNoise(
        noise,
        valid=jnp.ones((count,), dtype=bool),
        realization_ids=tuple(f"path:{index}" for index in range(count)),
        coupling_id="common-private-replay",
        independence_labels=(
            jnp.arange(count, dtype=jnp.int32)
            if clusters is None
            else jnp.asarray(clusters, dtype=jnp.int32)
        ),
        noise_shape=noise.shape[2:],
    )
    return ControlledPathBatch(
        problem=controlled,
        prepared_noise=prepared,
        states=states,
        actions=actions,
        valid=jnp.ones((count,), dtype=bool),
        status=jnp.zeros((count,), dtype=jnp.int32),
        stage_costs=jnp.zeros((count, steps)),
        terminal_costs=jnp.zeros((count,)),
        returns=jnp.zeros((count,)),
        policy_id="supplied-open-loop-action-path",
    )


def _problem(*, sigma_action=0.0, terminal_gradient=None, bad_derivative=False):
    grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="smp-grid")
    terminal = (
        (lambda time, state, args: state)
        if terminal_gradient is None
        else terminal_gradient
    )
    return StochasticMaximumPrincipleProblem(
        grid,
        lambda context, state, action, args: action,
        lambda context, state, action, args: action[:, None] * sigma_action,
        lambda context, state, action, args: jnp.asarray(
            [[jnp.where(bad_derivative & (state[0] > 0.75), jnp.nan, 0.0)]]
        ),
        lambda context, state, action, args: jnp.ones((1, 1)),
        lambda context, state, action, args: jnp.zeros((1, 1, 1)),
        lambda context, state, action, args: jnp.full((1, 1, 1), sigma_action),
        lambda context, state, action, args: jnp.zeros((1,)),
        lambda context, state, action, args: action,
        terminal,
        state_shape=(1,),
        action_shape=(1,),
        noise_shape=(1,),
        problem_id="smp-test",
    )


def _evaluate(problem, paths, adjoints, integrands, labels=None, **kwargs):
    count = paths.path_count
    return evaluate_stochastic_maximum_principle(
        problem,
        paths,
        adjoints,
        integrands,
        jnp.zeros((count, 1), dtype=jnp.int32) if labels is None else labels,
        information_id="pre-increment-filtration",
        predictor_id="supplied-adjoint-pair",
        sample_id="smp-sample-17",
        causal_information_checked=True,
        **kwargs,
    )


def test_stochastic_smp_reduces_to_deterministic_pmp():
    paths = _paths(
        states=[[[1.0], [0.5]], [[1.0], [0.5]]],
        actions=[[[-0.5]], [[-0.5]]],
        noise=[[[0.0]], [[0.0]]],
        clusters=[0, 1],
    )
    adjoints = jnp.full((2, 2, 1), 0.5)
    integrands = jnp.zeros((2, 1, 1, 1))

    result = _evaluate(_problem(), paths, adjoints, integrands)

    assert result.certificate == "OPEN_LOOP_SMP_STATIONARY"
    assert jnp.all(result.stationary)
    assert jnp.allclose(result.forward_residuals, 0.0)
    assert jnp.allclose(result.terminal_adjoint_residuals, 0.0)
    assert jnp.allclose(result.backward_martingale_residuals, 0.0)
    assert jnp.allclose(result.conditional_stationarity_residuals, 0.0)
    assert int(result.path_evidence.valid_path_count) == 2
    assert int(result.path_evidence.independent_cluster_count) == 2
    assert result.path_evidence.sample_role == "holdout"
    assert not result.sufficient
    assert not result.population_stationarity_claim
    assert not result.feedback_claim
    assert not result.markov_perfect_claim
    assert not result.global_optimality_claim


@pytest.mark.parametrize("sample_role", ["training", "holdout"])
def test_zero_empirical_smp_residual_does_not_claim_global_optimality(sample_role):
    paths = _paths(
        states=[[[1.0], [0.5]], [[1.0], [0.5]]],
        actions=[[[-0.5]], [[-0.5]]],
        noise=[[[0.0]], [[0.0]]],
        clusters=[0, 1],
    )
    result = _evaluate(
        _problem(),
        paths,
        jnp.full((2, 2, 1), 0.5),
        jnp.zeros((2, 1, 1, 1)),
        convexity_checked=True,
        convexity_evidence="jointly checked convex running and terminal costs",
        sample_role=sample_role,
    )

    assert result.certificate == "OPEN_LOOP_SMP_STATIONARY"
    assert jnp.all(result.stationary)
    assert jnp.allclose(result.conditional_stationarity_residuals, 0.0)
    assert jnp.allclose(result.maximum_residual_norms, 0.0)
    assert result.path_evidence.sample_role == sample_role
    assert result.convexity_checked
    assert result.convexity_evidence is not None
    assert not result.sufficient
    assert not result.population_stationarity_claim
    assert not result.global_optimality_claim
    assert not result.feedback_claim


def test_stochastic_smp_includes_q_sigma_action_term():
    paths = _paths(
        states=[[[0.0], [0.0]], [[0.0], [0.0]]],
        actions=[[[0.0]], [[0.0]]],
        noise=[[[0.0]], [[0.0]]],
    )
    adjoints = jnp.zeros((2, 2, 1))
    integrands = jnp.ones((2, 1, 1, 1))

    controlled_diffusion = _evaluate(
        _problem(sigma_action=1.0), paths, adjoints, integrands
    )
    mutated_derivative = _evaluate(
        _problem(sigma_action=0.0), paths, adjoints, integrands
    )

    assert jnp.allclose(controlled_diffusion.hamiltonian_action_gradients, 1.0)
    assert jnp.allclose(controlled_diffusion.conditional_stationarity_residuals, 1.0)
    assert jnp.allclose(mutated_derivative.hamiltonian_action_gradients, 0.0)
    assert not jnp.any(controlled_diffusion.stationary)
    assert jnp.all(mutated_derivative.stationary)


def test_stochastic_smp_reports_terminal_adjoint_mismatch():
    paths = _paths(
        states=[[[0.0], [2.0]]],
        actions=[[[2.0]]],
        noise=[[[0.0]]],
    )
    adjoints = jnp.zeros((1, 2, 1))
    integrands = jnp.zeros((1, 1, 1, 1))

    result = _evaluate(_problem(), paths, adjoints, integrands)

    assert bool(result.valid[0])
    assert not bool(result.stationary[0])
    assert jnp.allclose(result.terminal_adjoint_residuals[0], -2.0)
    assert float(result.terminal_adjoint_rms_norms[0]) == 2.0


def test_stochastic_smp_quarantines_nonfinite_derivative_evidence():
    paths = _paths(
        states=[[[0.0], [0.0]], [[1.0], [1.0]]],
        actions=[[[0.0]], [[0.0]]],
        noise=[[[0.0]], [[0.0]]],
    )
    adjoints = jnp.zeros((2, 2, 1))
    integrands = jnp.zeros((2, 1, 1, 1))
    labels = jnp.asarray([[0], [1]], dtype=jnp.int32)

    result = _evaluate(_problem(bad_derivative=True), paths, adjoints, integrands, labels)

    assert bool(result.valid[0])
    assert not bool(result.valid[1])
    assert int(result.status[1]) == int(
        StochasticMaximumPrincipleStatus.NONFINITE_DERIVATIVE
    )
    assert jnp.isinf(result.maximum_residual_norms[1])
    assert int(result.path_evidence.valid_path_count) == 1
