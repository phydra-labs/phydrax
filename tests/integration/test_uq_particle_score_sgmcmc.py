#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _parameterized_problem() -> Any:
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[0.4], [0.8]]),
        case_ids=("only",),
        sequence_id="particle-sgmcmc",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="state-prior",
    )

    def offset(t0: Any, t1: Any, context: Any) -> Any:
        del t0, t1
        return jnp.asarray([context.args["drift"]])

    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        offset=offset,
        process_id="drifting-transition",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    state_problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="particle-sgmcmc-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="particle-sgmcmc-problem",
        args={"drift": jnp.asarray(0.0)},
    )
    parameter_space = phx.uq.ParameterSpace(
        {"drift": jnp.asarray(0.0)},
        priors={"drift": phx.uq.Normal(0.0, 1.0)},
    )
    parameterized = phx.uq.ParameterizedStateSpaceProblem(
        state_problem,
        parameter_space,
        lambda physical, _: {"drift": physical["drift"]},
        parameterization_id="drift",
    )
    stochastic_problem = phx.uq.MinibatchPosteriorProblem(
        parameter_space,
        lambda physical, batch: jnp.zeros(batch.factor_mask.shape),
        num_factors=1,
    )
    source = phx.uq.ArrayMinibatchSource(jnp.zeros((1, 1)), batch_size=1)
    return stochastic_problem, source, parameterized


def test_uq_particle_score_sgmcmc_scenario_1() -> None:
    problem, source, parameterized = _parameterized_problem()
    estimator = phx.uq.ParticleGenealogicalGradientEstimator(
        parameterized,
        num_particles=8,
        resampling_policy="never",
    )
    result = phx.uq.sample_sgld(
        problem,
        source,
        key=jax.random.key(1),
        step_size=1e-4,
        num_chains=2,
        num_burnin=2,
        num_samples=4,
        gradient_estimator=estimator,
    )

    assert result.gradient_estimator_id == estimator.estimator_id
    assert result.samples["drift"].shape == (2, 4)
    assert jnp.all(jnp.isfinite(result.samples["drift"]))
    assert jnp.all(jnp.isfinite(result.gradient_norm))
    problem, source, parameterized = _parameterized_problem()
    control = phx.uq.build_sgmcmc_control_variate(
        problem,
        source,
        problem.initial_position,
    )
    estimator = phx.uq.ParticleGenealogicalGradientEstimator(
        parameterized,
        num_particles=4,
    )

    with pytest.raises(ValueError, match="does not support"):
        phx.uq.sample_sgld(
            problem,
            source,
            key=jax.random.key(3),
            step_size=1e-4,
            num_chains=2,
            num_burnin=1,
            num_samples=4,
            control_variate=control,
            gradient_estimator=estimator,
        )
    inputs = jnp.linspace(-1.0, 1.0, 6)
    source = phx.uq.ArrayMinibatchSource(inputs, batch_size=3, seed=2)
    space = phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.MinibatchPosteriorProblem(
        space,
        lambda parameter, batch: -0.5 * jnp.square(batch.data - parameter),
        num_factors=source.num_factors,
    )
    settings = dict(
        key=jax.random.key(2),
        step_size=1e-4,
        num_chains=2,
        num_burnin=2,
        num_samples=4,
    )
    # ty: ignore[invalid-argument-type]
    default = phx.uq.sample_sgld(problem, source, **settings)
    explicit = phx.uq.sample_sgld(
        problem,
        source,
        # ty: ignore[invalid-argument-type]
        **settings,
        gradient_estimator=phx.uq.AutodiffStochasticGradientEstimator(),
    )

    assert jnp.array_equal(default.unconstrained_samples, explicit.unconstrained_samples)
    assert jnp.array_equal(default.gradient_norm, explicit.gradient_norm)
