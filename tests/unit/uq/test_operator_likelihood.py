#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _batch(*, quadrature_scale: float = 1.0) -> phx.nn.OperatorBatch:
    coordinates = jnp.asarray(
        [
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.5], [1.0]],
        ]
    )
    query = phx.nn.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=quadrature_scale
        * jnp.asarray([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]),
        mask=jnp.asarray([[True, True, False], [True, True, True]]),
    )
    source = phx.nn.FunctionSamples(
        values=jnp.ones((2, 3)),
        coordinates=coordinates,
    )
    return phx.nn.OperatorBatch(inputs={"forcing": source}, queries={"query": query}, case_axes=("case",),
    case_shape=(2,),)


def _prediction(parameters, batch, spec):
    values = jnp.broadcast_to(parameters["level"], spec.expected_shape(batch))
    return phx.nn.OperatorPrediction.from_field(
        "output",
        values,
        "query",
        batch.require_single_query(),
        spec=spec,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _target_and_mask():
    target = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [jnp.nan, jnp.nan]],
            [[2.0, 3.0], [4.0, jnp.nan], [6.0, 7.0]],
        ]
    )
    observation_mask = jnp.asarray(
        [
            [[True, False], [True, True], [True, True]],
            [[True, True], [True, False], [False, True]],
        ]
    )
    return target, observation_mask


def test_operator_likelihood_matches_manual_sum_and_is_jittable():
    batch = _batch()
    spec = phx.nn.OperatorOutputSpec(2, component_names=("u", "v"))
    target, observation_mask = _target_and_mask()
    scale = 0.5
    term = phx.uq.FixedOperatorObservationLikelihood(
        lambda parameters: _prediction(parameters, batch, spec),
        batch,
        target,
        phx.uq.GaussianLikelihood(scale),
        output_spec=spec,
        field_name="output",
        query_name="query",
        observation_mask=observation_mask,
    )
    parameters = {"level": jnp.asarray(1.5)}

    combined = batch.require_single_query().mask_array(
        case_shape=batch.case_shape
    )[..., None]
    combined = jnp.broadcast_to(combined, target.shape) & observation_mask
    safe_target = jnp.where(combined, target, 0.0)
    elements = (
        -0.5 * ((safe_target - parameters["level"]) / scale) ** 2
        - jnp.log(scale)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    expected = jnp.sum(jnp.where(combined, elements, 0.0), axis=(1, 2))

    assert jnp.allclose(term.per_case_log_prob(parameters), expected)
    assert jnp.allclose(term.log_prob(parameters), jnp.sum(expected))
    compiled = eqx.filter_jit(
        lambda likelihood_term, value: likelihood_term.log_prob(value)
    )
    assert jnp.allclose(compiled(term, parameters), jnp.sum(expected))


def test_operator_likelihood_gradient_and_standardized_residual():
    batch = _batch()
    spec = phx.nn.OperatorOutputSpec(2)
    target, observation_mask = _target_and_mask()
    scale = 0.75
    term = phx.uq.FixedOperatorObservationLikelihood(
        lambda level: _prediction({"level": level}, batch, spec),
        batch,
        target,
        phx.uq.GaussianLikelihood(scale),
        output_spec=spec,
        field_name="output",
        query_name="query",
        observation_mask=observation_mask,
    )
    level = jnp.asarray(0.25)
    combined = term.observation_mask
    safe_target = jnp.where(combined, target, 0.0)

    gradient = jax.grad(term.log_prob)(level)
    expected_gradient = jnp.sum(
        jnp.where(combined, (safe_target - level) / scale**2, 0.0)
    )
    residual = term.standardized_residual(level)

    assert jnp.allclose(gradient, expected_gradient)
    assert jnp.all(residual[~combined] == 0.0)
    assert jnp.allclose(
        residual[combined],
        ((safe_target - level) / scale)[combined],
    )


def test_operator_likelihood_is_independent_of_quadrature_weights():
    first = _batch(quadrature_scale=1.0)
    second = _batch(quadrature_scale=17.0)
    spec = phx.nn.OperatorOutputSpec(2)
    target, observation_mask = _target_and_mask()

    def make_term(batch):
        return phx.uq.FixedOperatorObservationLikelihood(
            lambda parameters: _prediction(parameters, batch, spec),
            batch,
            target,
            phx.uq.GaussianLikelihood(0.4),
            output_spec=spec,
            field_name="output",
            query_name="query",
            observation_mask=observation_mask,
        )

    parameters = {"level": jnp.asarray(0.7)}
    assert jnp.array_equal(
        make_term(first).per_case_log_prob(parameters),
        make_term(second).per_case_log_prob(parameters),
    )


def test_operator_likelihood_handles_nonfinite_values_by_observation_status():
    batch = _batch()
    spec = phx.nn.OperatorOutputSpec(2)
    target, observation_mask = _target_and_mask()
    finite_term = phx.uq.FixedOperatorObservationLikelihood(
        lambda parameters: _prediction(parameters, batch, spec),
        batch,
        target,
        phx.uq.GaussianLikelihood(1.0),
        output_spec=spec,
        field_name="output",
        query_name="query",
        observation_mask=observation_mask,
    )

    def invalid_prediction(parameters):
        prediction = _prediction(parameters, batch, spec)
        values = prediction.field("output").values.at[0, 0, 0].set(jnp.nan)
        values = values.at[0, 2, 0].set(jnp.nan)
        return phx.nn.OperatorPrediction.from_field(
            "output",
            values,
            "query",
            batch.require_single_query(),
            spec=spec,
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    invalid_term = phx.uq.FixedOperatorObservationLikelihood(
        invalid_prediction,
        batch,
        target,
        phx.uq.GaussianLikelihood(1.0),
        output_spec=spec,
        field_name="output",
        query_name="query",
        observation_mask=observation_mask,
    )
    parameters = {"level": jnp.asarray(1.0)}

    assert jnp.isfinite(finite_term.log_prob(parameters))
    assert invalid_term.log_prob(parameters) == -jnp.inf


def test_operator_likelihood_rejects_empty_cases_and_contract_mismatches():
    batch = _batch()
    spec = phx.nn.OperatorOutputSpec(2)
    target, _ = _target_and_mask()
    empty_case_mask = jnp.ones_like(target, dtype=bool).at[1].set(False)

    with pytest.raises(ValueError, match="at least one observation"):
        phx.uq.FixedOperatorObservationLikelihood(
            lambda parameters: _prediction(parameters, batch, spec),
            batch,
            target,
            phx.uq.GaussianLikelihood(1.0),
            output_spec=spec,
            field_name="output",
            query_name="query",
            observation_mask=empty_case_mask,
        )

    term = phx.uq.FixedOperatorObservationLikelihood(
        lambda parameters: phx.nn.OperatorPrediction.from_field(
            "output",
            jnp.ones((2, 3)),
            "query",
            batch.require_single_query(),
            spec=phx.nn.OperatorOutputSpec("scalar"),
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        ),
        batch,
        jnp.ones((2, 3, 2)),
        phx.uq.GaussianLikelihood(1.0),
        output_spec=spec,
        field_name="output",
        query_name="query",
    )
    with pytest.raises(ValueError, match="fixed batch contract"):
        term.log_prob({"level": jnp.asarray(0.0)})
