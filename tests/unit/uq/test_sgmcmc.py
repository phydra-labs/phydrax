#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any, cast

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from blackjax.sgmcmc import diffusions
from blackjax.sgmcmc.sgnht import init as init_sgnht

import phydrax as phx
import phydrax.uq._sgmcmc as sgmcmc_module


def _regression_problem(*, batch_size=3, seed=7):
    inputs = jnp.linspace(-1.5, 1.5, 8)
    targets = 0.4 + 1.7 * inputs
    data = {"input": inputs, "target": targets}
    source = phx.uq.ArrayMinibatchSource(data, batch_size=batch_size, seed=seed)
    space = phx.uq.ParameterSpace(
        jnp.asarray([0.0, 0.0]),
        priors=phx.uq.Normal(0.0, 3.0),
    )

    def factors(parameters, batch):
        prediction = parameters[0] + parameters[1] * batch.data["input"]
        return -0.5 * ((batch.data["target"] - prediction) / 0.5) ** 2

    def full_likelihood(parameters):
        prediction = parameters[0] + parameters[1] * inputs
        return jnp.sum(-0.5 * ((targets - prediction) / 0.5) ** 2)

    problem = phx.uq.MinibatchPosteriorProblem(
        space,
        factors,
        num_factors=source.num_factors,
        full_log_likelihood=full_likelihood,
        predict=lambda parameters, values: parameters[0] + parameters[1] * values,
    )
    return problem, source


def _assert_tree_equal(left, right):
    comparisons = jax.tree_util.tree_map(jnp.array_equal, left, right)
    assert all(jax.tree_util.tree_leaves(comparisons))


def _transition_keys(chain_keys, update):
    return jax.vmap(lambda key: jr.fold_in(jr.fold_in(key, 1), update))(chain_keys)


def test_sgld_first_update_matches_blackjax_diffusion_convention():
    problem, source = _regression_problem()
    initial = jnp.asarray([[-0.3, 0.2], [0.4, -0.1]])
    step_size = 2.0e-4
    result = phx.uq.sample_sgld(
        problem,
        source,
        key=jr.key(20),
        step_size=step_size,
        num_chains=2,
        num_burnin=1,
        num_samples=4,
        initial_positions=initial,
    )
    batch = next(source.epoch(0))
    gradients = jax.vmap(
        lambda position: jax.grad(problem.log_density_estimate)(position, batch)
    )(initial)
    keys = _transition_keys(result.chain_keys, 0)
    integrator = diffusions.overdamped_langevin()
    expected = jax.vmap(
        lambda key, position, gradient: integrator(
            key, position, gradient, step_size, 1.0
        )
    )(keys, initial, gradients)

    assert jnp.allclose(
        result.burnin_states,
        expected,
        rtol=0.0,
        atol=jnp.finfo(expected.dtype).eps,
    )


def test_sgnht_first_update_matches_blackjax_diffusion_convention():
    problem, source = _regression_problem()
    initial = jnp.asarray([[-0.2, 0.3], [0.25, -0.15]])
    step_size = 1.0e-4
    diffusion = 0.02
    thermostat = 0.03
    result = phx.uq.sample_sgnht(
        problem,
        source,
        key=jr.key(21),
        step_size=step_size,
        diffusion=diffusion,
        initial_thermostat=thermostat,
        num_chains=2,
        num_burnin=1,
        num_samples=4,
        initial_positions=initial,
    )
    initialization_keys = jax.vmap(lambda key: jr.fold_in(key, 0))(result.chain_keys)
    states = jax.vmap(lambda position, key: init_sgnht(position, key, thermostat))(
        initial, initialization_keys
    )
    batch = next(source.epoch(0))
    gradients = jax.vmap(
        lambda position: jax.grad(problem.log_density_estimate)(position, batch)
    )(states.position)
    keys = _transition_keys(result.chain_keys, 0)
    integrator = diffusions.sgnht(diffusion, 0.0)
    position, momentum, xi = jax.vmap(
        lambda key, state_position, state_momentum, state_xi, gradient: integrator(
            key,
            state_position,
            state_momentum,
            state_xi,
            gradient,
            step_size,
            1.0,
        )
    )(keys, states.position, states.momentum, states.xi, gradients)

    assert jnp.array_equal(result.burnin_states.position, position)
    assert jnp.array_equal(result.burnin_states.momentum, momentum)
    assert jnp.array_equal(result.burnin_states.xi, xi)


@pytest.mark.parametrize("sample", [phx.uq.sample_sgld, phx.uq.sample_sgnht])
def test_sgmcmc_sequential_and_vectorized_chains_replay_exactly(sample):
    problem, source = _regression_problem(batch_size=4)
    common: dict[str, Any] = {
        "key": jr.key(22),
        "step_size": 1.0e-4,
        "num_chains": 2,
        "num_burnin": 3,
        "num_samples": 6,
        "steps_per_sample": 2,
        "initial_positions": jnp.asarray([[-0.4, 0.1], [0.5, -0.2]]),
    }
    vectorized = sample(problem, source, **common, chain_method="vectorized")
    sequential = sample(problem, source, **common, chain_method="sequential")

    _assert_tree_equal(vectorized.samples, sequential.samples)
    _assert_tree_equal(vectorized.final_states, sequential.final_states)
    assert jnp.array_equal(vectorized.gradient_norm, sequential.gradient_norm)
    assert not jnp.array_equal(vectorized.samples[0], vectorized.samples[1])


def test_control_variate_is_exact_at_center_and_rejects_other_sources():
    problem, source = _regression_problem(batch_size=3)
    center = jnp.asarray([0.2, 1.5])
    control = phx.uq.build_sgmcmc_control_variate(problem, source, center)
    expected = jax.grad(problem.full_log_density)(center)
    estimator = sgmcmc_module._gradient_estimator(problem, control)
    ordinary = jax.grad(problem.log_density_estimate)
    batches = tuple(source.epoch(0))
    controlled = jnp.stack([estimator(center, batch) for batch in batches])
    uncontrolled = jnp.stack([ordinary(center, batch) for batch in batches])

    assert jnp.allclose(control.full_gradient, expected)
    assert jnp.allclose(controlled, jnp.broadcast_to(expected, controlled.shape))
    assert jnp.max(jnp.var(controlled, axis=0)) < jnp.max(jnp.var(uncontrolled, axis=0))
    assert control.construction_gradient_evaluations == source.batches_per_epoch + 2

    incompatible_source = phx.uq.ArrayMinibatchSource(
        source.data, batch_size=3, seed=source.configuration()["seed"] + 1
    )
    with pytest.raises(ValueError, match="different source"):
        phx.uq.sample_sgld(
            problem,
            incompatible_source,
            key=jr.key(23),
            step_size=1.0e-4,
            num_chains=2,
            num_burnin=1,
            num_samples=4,
            control_variate=control,
        )


def test_sgmcmc_preserves_nested_constrained_parameter_samples():
    values = jnp.linspace(-1.0, 1.0, 6)
    source = phx.uq.ArrayMinibatchSource(values, batch_size=4, seed=9)
    space = phx.uq.ParameterSpace(
        {"location": jnp.asarray(0.0), "scale": jnp.asarray(0.0)},
        priors={
            "location": phx.uq.Normal(0.0, 2.0),
            "scale": phx.uq.LogNormal(0.0, 0.5),
        },
        bijectors={
            "location": phx.uq.IdentityBijector(),
            "scale": phx.uq.ExpBijector(),
        },
    )

    def factors(parameters, batch):
        return -0.5 * ((batch.data - parameters["location"]) / parameters["scale"]) ** 2

    problem = phx.uq.MinibatchPosteriorProblem(
        space,
        factors,
        num_factors=source.num_factors,
        predict=lambda parameters, x: cx.Field(
            parameters["location"] + parameters["scale"] * x,
            dims=("point",),
        ),
    )
    result = phx.uq.sample_sgld(
        problem,
        source,
        key=jr.key(24),
        step_size=1.0e-5,
        num_chains=2,
        num_burnin=2,
        num_samples=4,
    )

    assert result.samples["location"].shape == (2, 4)
    assert result.samples["scale"].shape == (2, 4)
    assert jnp.all(result.samples["scale"] > 0.0)
    prediction = result.predict(jnp.asarray([1.0, 2.0]))
    assert isinstance(prediction, phx.uq.PredictiveField)
    assert prediction.samples.shape == (2, 4, 2)


def test_sgmcmc_result_exposes_honest_diagnostics_and_mixing_gates():
    problem, source = _regression_problem(batch_size=3)
    result = phx.uq.sample_sgld(
        problem,
        source,
        key=jr.key(25),
        step_size=1.0e-4,
        num_chains=2,
        num_burnin=3,
        num_samples=8,
    )
    report = result.mixing_report(
        max_rhat=2.0,
        min_bulk_ess=1.0e9,
        min_tail_ess=1.0e9,
    )

    assert result.log_density is not None
    assert result.approximation == "unadjusted_fixed_step"
    assert result.log_density.shape == (2, 8)
    assert result.thermostat is None
    assert result.momentum_norm is None
    assert result.diagnostics.min_active_factors == 2
    assert result.diagnostics.max_active_factors == 3
    assert "bulk_ess" in report.failures
    assert report.as_dict()["approximation"] == "unadjusted_fixed_step"
    with pytest.raises(phx.uq.SGMCMCMixingError):
        report.raise_for_failure()


def test_sgmcmc_rejects_invalid_controls_and_reports_nonfinite_locations():
    problem, source = _regression_problem()
    common: dict[str, Any] = {
        "key": jr.key(26),
        "step_size": 1.0e-4,
        "num_chains": 2,
        "num_burnin": 1,
        "num_samples": 4,
    }
    with pytest.raises(ValueError, match="step_size"):
        phx.uq.sample_sgld(
            problem,
            source,
            **cast(dict[str, Any], common | {"step_size": 0.0}),
        )
    with pytest.raises(ValueError, match="num_chains"):
        phx.uq.sample_sgld(
            problem,
            source,
            **cast(dict[str, Any], common | {"num_chains": 1}),
        )
    with pytest.raises(ValueError, match="num_samples"):
        phx.uq.sample_sgld(
            problem,
            source,
            **cast(dict[str, Any], common | {"num_samples": 3}),
        )
    with pytest.raises(FloatingPointError, match=r"chain\[1\]"):
        phx.uq.sample_sgld(
            problem,
            source,
            **common,
            initial_positions=jnp.asarray([[0.0, 0.0], [jnp.nan, 0.0]]),
        )
