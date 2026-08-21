#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optimistix as optx
import pytest

import phydrax as phx


def _scenario_problem(*, sampling=None, risk=None, chance_constraints=()):
    sampling = (
        phx.optim.FixedSampling(jnp.array([-1.0, 1.0, 3.0]))
        if sampling is None
        else sampling
    )
    return phx.optim.StochasticProblem(
        lambda parameter, scenario, _: 0.5 * jnp.sum((parameter - scenario) ** 2),
        sampling,
        risk=risk,
        chance_constraints=chance_constraints,
        problem_id="scenario-quadratic",
    )


def _termination(*, tolerance=1e-4, steps=200):
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def test_sampling_policies_have_reproducible_declared_refresh_semantics():
    def sampler(key, size):
        return jr.normal(key, (size,))

    fixed = phx.optim.MonteCarloSampling(sampler, 8, refresh="fixed")
    refreshing = phx.optim.MonteCarloSampling(sampler, 8, refresh="per_iteration")
    key = jr.key(7)

    np.testing.assert_array_equal(
        fixed.sample(key, 0).scenarios,
        fixed.sample(key, 9).scenarios,
    )
    assert not jnp.array_equal(
        refreshing.sample(key, 0).scenarios,
        refreshing.sample(key, 1).scenarios,
    )
    compiled_fixed_sample = eqx.filter_jit(
        lambda iteration: fixed.sample(key, iteration).scenarios
    )
    compiled_refreshing_sample = eqx.filter_jit(
        lambda iteration: refreshing.sample(key, iteration).scenarios
    )
    np.testing.assert_array_equal(
        compiled_fixed_sample(jnp.asarray(0)),
        compiled_fixed_sample(jnp.asarray(9)),
    )
    assert not jnp.array_equal(
        compiled_refreshing_sample(jnp.asarray(0)),
        compiled_refreshing_sample(jnp.asarray(1)),
    )
    integer_batch = phx.optim.SampleBatch(jnp.array([0, 1, 2]))
    np.testing.assert_allclose(integer_batch.weights, jnp.full(3, 1.0 / 3.0))


def test_risk_measures_match_weighted_definitions_and_are_finite():
    losses = jnp.array([1.0, 2.0, 8.0])
    weights = jnp.full(3, 1.0 / 3.0)

    np.testing.assert_allclose(
        phx.optim.ExpectationRisk().evaluate(losses, weights),
        11.0 / 3.0,
    )
    np.testing.assert_allclose(
        phx.optim.MeanVarianceRisk(0.5).evaluate(losses, weights),
        76.0 / 9.0,
    )
    np.testing.assert_allclose(
        phx.optim.CVaRRisk(0.5).evaluate(losses, weights),
        6.0,
    )
    assert jnp.isfinite(phx.optim.EntropicRisk(0.2).evaluate(losses, weights))


@pytest.mark.parametrize("method", [phx.optim.StochasticAdam(0.05)])
def test_stochastic_gradient_baseline_optimizes_fixed_expectation(method):
    result = phx.optim.minimize_stochastic(
        _scenario_problem(),
        jnp.array([0.0]),
        method=method,
        termination=_termination(tolerance=1e-5),
        seed=4,
    )

    np.testing.assert_allclose(result.parameters, jnp.array([1.0]), atol=2e-4)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    np.testing.assert_allclose(result.objective, 4.0 / 3.0, atol=1e-7)
    assert result.provenance.backend == "optax"


def test_stochastic_adam_staged_large_budget_agrees_eager_and_jit():
    problem = _scenario_problem()
    method = phx.optim.StochasticAdam(0.05)
    termination = _termination(tolerance=1e-5, steps=50_000)
    key = jr.key(4)

    def solve(initial):
        return phx.optim.minimize_stochastic(
            problem,
            initial,
            method=method,
            termination=termination,
            key=key,
        )

    eager = solve(jnp.array([0.0]))
    compiled = eqx.filter_jit(solve)(jnp.array([0.0]))

    np.testing.assert_allclose(compiled.parameters, eager.parameters, atol=1e-7)
    np.testing.assert_allclose(compiled.objective, eager.objective, atol=1e-7)
    np.testing.assert_array_equal(compiled.status, eager.status)
    np.testing.assert_array_equal(
        compiled.diagnostics.iterations,
        eager.diagnostics.iterations,
    )
    np.testing.assert_array_equal(
        compiled.diagnostics.objective_evaluations,
        eager.diagnostics.objective_evaluations,
    )
    np.testing.assert_array_equal(
        compiled.diagnostics.gradient_evaluations,
        eager.diagnostics.gradient_evaluations,
    )
    assert compiled.status == phx.optim.OptimizationStatus.SUCCESS


def test_stochastic_adam_reuses_iteration_batch_for_accepted_result():
    def sampler(key, size):
        return jr.uniform(key, (size,), minval=-3.0, maxval=3.0)

    sampling = phx.optim.MonteCarloSampling(
        sampler,
        16,
        refresh="per_iteration",
    )
    problem = _scenario_problem(sampling=sampling)
    key = jr.key(29)
    result = phx.optim.minimize_stochastic(
        problem,
        jnp.array([0.0]),
        method=phx.optim.StochasticAdam(0.03),
        termination=_termination(tolerance=0.0, steps=3),
        key=key,
    )

    accepted_batch = sampling.sample(key, 2)
    next_batch = sampling.sample(key, 3)
    expected_objective = problem.value(result.parameters, accepted_batch)
    fresh_objective = problem.value(result.parameters, next_batch)
    np.testing.assert_allclose(result.objective, expected_objective, atol=1e-7)
    assert not np.isclose(result.objective, fresh_objective)
    assert result.diagnostics.iterations == 3
    assert result.diagnostics.accepted_steps == 3
    assert result.diagnostics.objective_evaluations == 4
    assert result.diagnostics.gradient_evaluations == 4


def test_stochastic_adam_evaluation_budget_gates_complete_iterations():
    result = phx.optim.minimize_stochastic(
        _scenario_problem(),
        jnp.array([0.0]),
        method=phx.optim.StochasticAdam(0.05),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=0.0,
            relative_optimality=0.0,
            maximum_steps=100,
            maximum_evaluations=2,
        ),
        key=jr.key(3),
    )

    assert result.status == phx.optim.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
    assert result.diagnostics.iterations == 2
    assert result.diagnostics.accepted_steps == 2
    # Final result packaging evaluates the accepted point outside the iteration gate.
    assert result.diagnostics.objective_evaluations == 3
    assert result.diagnostics.gradient_evaluations == 3


def test_stochastic_adam_rejects_nonfinite_input_before_staged_iteration():
    problem = phx.optim.StochasticProblem(
        lambda _parameter, scenario, _: jnp.square(scenario),
        phx.optim.FixedSampling(jnp.array([-1.0, 1.0])),
    )
    result = phx.optim.minimize_stochastic(
        problem,
        jnp.array([jnp.nan]),
        method=phx.optim.StochasticAdam(),
        termination=_termination(steps=10),
        key=jr.key(0),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_INPUT
    assert jnp.isnan(result.parameters[0])
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.accepted_steps == 0
    assert result.diagnostics.objective_evaluations == 1
    assert result.diagnostics.gradient_evaluations == 1


def test_stochastic_adam_rejects_nonfinite_candidate_without_mutating_state():
    problem = phx.optim.StochasticProblem(
        lambda parameter, _scenario, _: parameter[0],
        phx.optim.FixedSampling(jnp.array([0.0])),
    )
    initial = jnp.array([-3.0e38], dtype=jnp.float32)
    result = phx.optim.minimize_stochastic(
        problem,
        initial,
        method=phx.optim.StochasticAdam(3.0e38),
        termination=_termination(tolerance=0.0, steps=2),
        key=jr.key(0),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_EVALUATION
    np.testing.assert_array_equal(result.parameters, initial)
    assert jnp.isfinite(result.objective)
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.accepted_steps == 0
    assert result.diagnostics.rejected_steps == 1
    assert result.diagnostics.final_step_norm == 0.0
    assert result.diagnostics.objective_evaluations == 2
    assert result.diagnostics.gradient_evaluations == 2


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.ProgressiveHedging(inner_maximum_steps=20),
        phx.optim.ConsensusADMM(inner_maximum_steps=20),
    ],
)
def test_scenario_consensus_methods_recover_expected_value_solution(method):
    result = phx.optim.minimize_stochastic(
        _scenario_problem(),
        jnp.array([0.0]),
        method=method,
        termination=_termination(tolerance=1e-4, steps=25),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([1.0]), atol=1e-4)
    assert result.status == phx.optim.OptimizationStatus.SUCCESS
    assert result.diagnostics.primal_feasibility < 1e-4
    assert result.scenario_parameters.shape == (3, 1)
    assert result.duals.shape == (3, 1)


@pytest.mark.parametrize(
    "method",
    [
        phx.optim.ProgressiveHedging(inner_maximum_steps=2),
        phx.optim.ConsensusADMM(inner_maximum_steps=2),
    ],
)
def test_scenario_consensus_rejects_nonfinite_input_at_workflow_boundary(method):
    result = phx.optim.minimize_stochastic(
        _scenario_problem(),
        jnp.array([jnp.nan]),
        method=method,
        termination=_termination(steps=2),
        key=jr.key(0),
    )

    assert result.status == phx.optim.OptimizationStatus.NONFINITE_INPUT
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.accepted_steps == 0
    assert result.diagnostics.objective_evaluations == 1
    assert jnp.isnan(result.parameters[0])


def test_scenario_consensus_rejects_incomplete_inner_evaluation_counts():
    method = phx.optim.ConsensusADMM(
        maximum_outer_steps=2,
        inner_maximum_steps=4,
        inner_method=phx.optim.OptimistixMethod(optx.BFGS(rtol=1e-6, atol=1e-6)),
    )

    with pytest.raises(ValueError, match="diagnostic counts are incomplete"):
        phx.optim.minimize_stochastic(
            _scenario_problem(sampling=phx.optim.FixedSampling(jnp.array([0.0]))),
            jnp.array([0.0]),
            method=method,
            termination=phx.optim.OptimizationTermination(
                maximum_steps=2,
                maximum_evaluations=4,
            ),
            key=jr.key(0),
        )


def test_chance_constraint_separates_empirical_and_smooth_estimators():
    batch = phx.optim.SampleBatch(jnp.array([-1.0, 1.0]))
    constraint = phx.optim.ChanceConstraint(
        lambda parameter, scenario, _: parameter[0] + scenario,
        maximum_probability=0.5,
        smoothing_temperature=0.1,
    )
    empirical, smooth = constraint.probabilities(jnp.array([0.0]), batch)

    np.testing.assert_allclose(empirical, 0.5)
    np.testing.assert_allclose(smooth, 0.5, atol=1e-7)
    problem = _scenario_problem(
        sampling=phx.optim.FixedSampling(batch.scenarios),
        chance_constraints=(constraint,),
    )
    frozen, frozen_batch = problem.frozen(jr.key(0))
    assert frozen_batch.size == 2
    assert len(frozen.constraints) == 1
    np.testing.assert_allclose(frozen.constraints[0].value(jnp.array([0.0])), 0.5)

    with pytest.raises(ValueError, match="does not silently penalize"):
        phx.optim.minimize_stochastic(
            problem,
            jnp.array([0.0]),
            method=phx.optim.StochasticAdam(),
        )


def test_stochastic_adam_supports_jvp_vmap_and_pytree_parameters():
    problem = phx.optim.StochasticProblem(
        lambda parameters, scenario, target: (
            0.5 * jnp.square(parameters["state"][0] - target + scenario)
        ),
        phx.optim.FixedSampling(jnp.array([-0.5, 0.5])),
    )
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=0.0,
        relative_optimality=0.0,
        maximum_steps=4,
    )

    def solution(target):
        return phx.optim.minimize_stochastic(
            problem,
            {"state": jnp.array([0.0])},
            method=phx.optim.StochasticAdam(0.05),
            termination=termination,
            key=jr.key(5),
            args=target,
        ).parameters["state"][0]

    targets = jnp.array([1.0, 1.5])
    mapped = jax.vmap(solution)(targets)
    independent = jnp.stack([solution(target) for target in targets])
    value, derivative = jax.jvp(
        solution,
        (jnp.array(1.25),),
        (jnp.array(0.3),),
    )
    step = 1e-4
    finite_difference = (
        0.3
        * (solution(jnp.array(1.25 + step)) - solution(jnp.array(1.25 - step)))
        / (2.0 * step)
    )

    np.testing.assert_allclose(mapped, independent, atol=1e-10)
    assert jnp.isfinite(value)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-4, atol=2e-6)
