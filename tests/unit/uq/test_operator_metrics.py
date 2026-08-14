#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _batch(
    coordinates,
    weights,
    *,
    cases: int = 2,
    mask=None,
) -> phx.nn.operator.OperatorBatch:
    nodes = jnp.asarray(coordinates, dtype=float)
    query_coordinates = jnp.broadcast_to(nodes[None, :, None], (cases, nodes.size, 1))
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=query_coordinates,
        quadrature_weights=jnp.broadcast_to(
            jnp.asarray(weights, dtype=float),
            (cases, nodes.size),
        ),
        mask=None if mask is None else jnp.asarray(mask, dtype=bool),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((cases, nodes.size)),
        coordinates=query_coordinates,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(cases,),
    )


def _predictive(samples, batch):
    return phx.uq.operator_predictive_from_samples(
        jnp.asarray(samples, dtype=float),
        batch,
        phx.nn.operator.OperatorOutputSpec("scalar"),
        sample_axes=(phx.uq.SampleAxis("member", "epistemic"),),
        field_name="output",
        query_name="query",
    )


def test_operator_crps_matches_weighted_pointwise_reference():
    batch = _batch(
        [0.0, 0.5, 1.0],
        [0.2, 0.3, 0.5],
        mask=[[True, True, False], [True, True, True]],
    )
    samples = jnp.asarray(
        [
            [[0.0, 1.0, 100.0], [1.0, 2.0, 3.0]],
            [[1.0, 2.0, -50.0], [2.0, 3.0, 4.0]],
            [[2.0, 3.0, jnp.nan], [3.0, 4.0, 5.0]],
        ]
    )
    target = jnp.asarray([[1.0, 1.5, jnp.nan], [2.0, 2.0, 4.0]])
    prediction = _predictive(samples, batch)

    pointwise = phx.uq.ensemble_crps(
        prediction.predictive.samples.data,
        jnp.where(prediction.output_mask(), target, 0.0),
        sample_axis=0,
    )
    weights = batch.require_single_query().weights(case_shape=batch.case_shape)
    expected = jnp.sum(pointwise * weights, axis=-1) / jnp.sum(weights, axis=-1)

    actual = phx.uq.operator_ensemble_crps(
        prediction,
        target,
        reduction="none",
    )
    uniform = phx.uq.operator_ensemble_crps(
        prediction,
        target,
        measure="uniform",
        reduction="none",
    )

    assert jnp.allclose(actual, expected)
    assert not jnp.allclose(actual, uniform)


def test_operator_energy_score_matches_whole_field_reference():
    batch = _batch([0.0, 1.0], [0.25, 0.75])
    samples = jnp.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 2.0], [2.0, 1.0]],
            [[2.0, 3.0], [3.0, 2.0]],
        ]
    )
    target = jnp.asarray([[1.0, 2.0], [2.0, 0.5]])
    prediction = _predictive(samples, batch)
    normalized_weights = batch.require_single_query().weights(
        case_shape=batch.case_shape,
        normalized=True,
    )
    expected = jnp.stack(
        tuple(
            phx.uq.energy_score(
                samples[:, index, :] * jnp.sqrt(normalized_weights[index]),
                target[index] * jnp.sqrt(normalized_weights[index]),
            )
            for index in range(2)
        )
    )

    actual = phx.uq.operator_energy_score(
        prediction,
        target,
        reduction="none",
    )

    assert jnp.allclose(actual, expected)


def test_operator_scores_are_invariant_to_equal_weight_point_splitting():
    base_batch = _batch([0.0, 1.0], [0.4, 0.6])
    split_batch = _batch([0.0, 1.0, 1.0], [0.4, 0.3, 0.3])
    base_samples = jnp.asarray(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
            [[2.0, 3.0], [3.0, 4.0]],
        ]
    )
    split_samples = base_samples[..., [0, 1, 1]]
    base_target = jnp.asarray([[0.5, 1.5], [1.5, 2.5]])
    split_target = base_target[..., [0, 1, 1]]
    base = _predictive(base_samples, base_batch)
    split = _predictive(split_samples, split_batch)

    assert jnp.allclose(
        phx.uq.operator_ensemble_crps(base, base_target, reduction="none"),
        phx.uq.operator_ensemble_crps(split, split_target, reduction="none"),
    )
    assert jnp.allclose(
        phx.uq.operator_energy_score(base, base_target, reduction="none"),
        phx.uq.operator_energy_score(split, split_target, reduction="none"),
    )


def test_operator_interval_pointwise_and_simultaneous_coverage_differ():
    batch = _batch([0.0, 0.5, 1.0], [1.0, 1.0, 1.0])
    spec = phx.nn.operator.OperatorOutputSpec("scalar")
    lower = phx.nn.operator.OperatorPrediction.from_field(
        "output",
        -jnp.ones((2, 3)),
        "query",
        batch.require_single_query(),
        spec=spec,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    upper = phx.nn.operator.OperatorPrediction.from_field(
        "output",
        jnp.ones((2, 3)),
        "query",
        batch.require_single_query(),
        spec=spec,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    interval = phx.uq.OperatorPredictionInterval(
        lower,
        upper,
        nominal_coverage=0.9,
        simultaneous=True,
        calibrated=True,
    )
    target = jnp.asarray([[0.0, 0.5, 1.0], [0.0, 2.0, 0.5]])

    pointwise = phx.uq.operator_interval_coverage(
        interval,
        target,
        field_name="output",
        mode="pointwise",
        reduction="none",
    )
    simultaneous = phx.uq.operator_interval_coverage(
        interval,
        target,
        field_name="output",
        mode="simultaneous",
        reduction="none",
    )
    width = phx.uq.operator_interval_width(
        interval,
        field_name="output",
        reduction="none",
    )

    assert jnp.allclose(pointwise, jnp.asarray([1.0, 2.0 / 3.0]))
    assert jnp.array_equal(simultaneous, jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(width, 2.0)


def test_operator_scores_reject_invalid_predictive_draws():
    batch = _batch([0.0, 1.0], [0.5, 0.5])
    samples = jnp.ones((3, 2, 2)).at[1, 0, 0].set(jnp.nan)
    prediction = _predictive(samples, batch)

    with pytest.raises(ValueError, match="every requested predictive draw"):
        phx.uq.operator_ensemble_crps(prediction, jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="every requested predictive draw"):
        phx.uq.operator_energy_score(prediction, jnp.ones((2, 2)))
