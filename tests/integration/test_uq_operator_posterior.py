#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _batch() -> phx.nn.operator.OperatorBatch:
    coordinates = jnp.asarray([[[0.0], [0.5], [1.0]]])
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        mask=jnp.asarray([[True, True, False]]),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((1, 3)),
        coordinates=coordinates,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(1,),
    )


def test_operator_map_laplace_and_geometry_reattachment_match_gaussian_reference():
    batch = _batch()
    spec = phx.nn.operator.OperatorOutputSpec("scalar")
    target = jnp.asarray([[1.0, 2.0, jnp.nan]])
    observation_scale = 0.5
    prior_scale = 2.0

    def operator_prediction(parameters):
        values = jnp.broadcast_to(parameters["level"], spec.expected_shape(batch))
        return phx.nn.operator.OperatorPrediction.from_field(
            "output",
            values,
            batch.single_query_name(),
            batch.require_single_query(),
            spec=spec,
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    term = phx.uq.FixedOperatorObservationLikelihood(
        operator_prediction,
        batch,
        target,
        phx.uq.GaussianLikelihood(observation_scale),
        output_spec=spec,
        field_name="output",
        query_name="query",
    )
    parameter_space = phx.uq.ParameterSpace(
        {"level": jnp.asarray(0.0)},
        priors={"level": phx.uq.Normal(0.0, prior_scale)},
    )
    problem = phx.uq.PosteriorProblem.from_terms(
        parameter_space,
        (term,),
        predict=lambda parameters: phx.uq.operator_prediction_field(
            operator_prediction(parameters),
            field_name="output",
        ),
        gauss_newton_residual=lambda parameters: term.standardized_residual(parameters),
    )

    observed = target[0, :2]
    precision = observed.size / observation_scale**2 + 1.0 / prior_scale**2
    analytic_mean = jnp.sum(observed) / observation_scale**2 / precision
    analytic_variance = 1.0 / precision

    map_result = phx.uq.find_map(
        problem,
        max_steps=80,
        gradient_tolerance=1e-8,
    )
    laplace = phx.uq.fit_laplace(
        problem,
        map_result.position,
        stationarity_tolerance=1e-7,
    )

    assert map_result.converged
    assert jnp.allclose(map_result.parameters["level"], analytic_mean, atol=1e-7)
    assert jnp.allclose(laplace.covariance[0, 0], analytic_variance, atol=1e-8)
    residual = problem.gauss_newton_residual(map_result.position)
    assert residual.shape == target.shape
    assert residual[0, 2] == 0.0

    predictive = laplace.predict(
        jr.key(0),
        num_samples=32,
        sample_dim="posterior_draw",
    )
    operator_predictive = phx.uq.OperatorPredictiveField.from_predictive(
        predictive,
        batch,
        spec,
        field_name="output",
        query_name="query",
    )

    assert operator_predictive.predictive.samples.dims == (
        "posterior_draw",
        "case",
        "__phydra_operator_point",
    )
    assert operator_predictive.query is batch.require_single_query()
    assert operator_predictive.case_axes == batch.case_axes
    assert jnp.all(operator_predictive.predictive.samples.data[:, 0, 2] == 0.0)
