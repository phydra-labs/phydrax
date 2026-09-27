#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx


def _problem() -> Any:
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[0.4], [0.8]]),
        case_ids=("only",),
        sequence_id="parameterized",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="prior",
    )

    def offset(t0: Any, t1: Any, context: Any) -> Any:
        del t0, t1
        return jnp.asarray([context.args["drift"]])

    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        offset=offset,
        process_id="drifting",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="parameterized-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="parameterized-problem",
        args={"fixed": jnp.asarray(2.0)},
    )


def test_parameterized_state_space_binds_physical_parameters_into_context() -> None:
    problem = _problem()
    parameterized = phx.uq.ParameterizedStateSpaceProblem(
        problem,
        phx.uq.ParameterSpace(
            {"drift": jnp.asarray(0.0)},
            priors={"drift": phx.uq.Normal(0.0, 1.0)},
        ),
        lambda physical, fixed: {
            "drift": physical["drift"],
            "fixed": fixed["fixed"],
        },
        parameterization_id="drift-parameterization",
    )
    bound = parameterized.bind({"drift": jnp.asarray(0.3)})

    assert jnp.allclose(bound.args["drift"], 0.3)
    assert jnp.allclose(bound.args["fixed"], 2.0)
    context = bound.step_context(0, 0)
    assert jnp.allclose(
        # ty: ignore[unresolved-attribute]
        bound.model.transition.parameters(0.0, 0.5, context).offset,
        jnp.asarray([0.3]),
    )


def test_parameterized_path_density_has_finite_parameter_gradient() -> None:
    parameterized = phx.uq.ParameterizedStateSpaceProblem(
        _problem(),
        phx.uq.ParameterSpace(
            {"drift": jnp.asarray(0.0)},
            priors={"drift": phx.uq.Normal(0.0, 1.0)},
        ),
        lambda physical, _: {"drift": physical["drift"]},
    )
    states = jnp.asarray([[0.0], [0.3], [0.7]])

    def log_density(drift: Any) -> Any:
        return parameterized.path_log_density(
            {"drift": drift},
            states,
        ).log_density

    result = parameterized.path_log_density(
        {"drift": jnp.asarray(0.2)},
        states,
    )
    gradient = jax.jit(jax.grad(log_density))(jnp.asarray(0.2))

    assert bool(result.valid)
    assert jnp.isfinite(result.log_density)
    assert jnp.isfinite(gradient)
    assert result.approximation_id == "parameterized-state-space"
