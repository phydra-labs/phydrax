#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.nn.operator import AbstractOperatorModel


class _MaskedSourceMeanOperator(AbstractOperatorModel):
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, /, *, key=None):
        del key
        source = batch.input("forcing")
        values = jnp.asarray(source.values)
        mask = source.mask_array(case_shape=batch.case_shape)
        source_mean = jnp.sum(jnp.where(mask, values, 0.0), axis=-1) / jnp.sum(
            mask,
            axis=-1,
        )
        return jnp.broadcast_to(
            source_mean[..., None],
            batch.case_shape + batch.require_single_query().sample_shape,
        )

    def __call__(self, x, /, *, key=None):
        if not isinstance(x, phx.nn.operator.OperatorBatch):
            raise TypeError("_MaskedSourceMeanOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


def _batch(value: float, *, source_points: int) -> phx.nn.operator.OperatorBatch:
    query_coordinates = jnp.asarray(
        [
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.5], [1.0]],
        ]
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=query_coordinates,
        mask=jnp.asarray([[True, True, False], [True, True, True]]),
    )
    source_coordinates = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, source_points)[None, :, None],
        (2, source_points, 1),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.full((2, source_points), value),
        coordinates=source_coordinates,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )


def test_input_function_prediction_matches_explicit_ragged_draw_loop():
    model = _MaskedSourceMeanOperator()
    batches = (_batch(1.0, source_points=2), _batch(3.0, source_points=4))
    stacked = phx.nn.operator.stack_operator_batches(batches, case_axis="input_draw")

    prediction = phx.uq.operator_input_predictive(
        model.predict(stacked),
        input_sample_axes=("input_draw",),
        field_name="output",
    )
    explicit = jnp.stack(
        tuple(model.predict(batch).field("output").values for batch in batches)
    )
    expected = jnp.where(prediction.output_mask()[None, ...], explicit, 0.0)

    assert prediction.predictive.samples.dims == (
        "input_draw",
        "case",
        "__phydra_operator_point",
    )
    assert jnp.allclose(jnp.asarray(prediction.predictive.samples.data), expected)
    assert prediction.predictive.sample_axes[0].source == "input"
    input_variance = prediction.input_variance()
    assert isinstance(input_variance, phx.nn.operator.OperatorPrediction)
    assert jnp.allclose(
        prediction.mean().field("output").values[prediction.output_mask()],
        2.0,
    )
    assert jnp.allclose(
        input_variance.field("output").values[prediction.output_mask()],
        1.0,
    )


def _weighted_batch(value: float, *, source_points: int) -> phx.nn.operator.OperatorBatch:
    query_axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, 3),
        quadrature_weights=jnp.asarray([0.25, 0.5, 0.25]),
    )
    source_axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, source_points),
        quadrature_weights=jnp.full((source_points,), 1.0 / source_points),
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(query_axis,),
        mask=jnp.asarray([[True, True, False], [True, True, True]]),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.full((2, source_points), value),
        axes=(source_axis,),
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )


def test_operator_linearized_covariance_preserves_geometry_and_masks():
    batch = _batch(2.0, source_points=4)
    linearization = phx.nn.operator.training.linearize_operator(
        _MaskedSourceMeanOperator(),
        batch,
        "forcing",
    )
    source_variance = jnp.full((2, 4), 0.25)
    result = phx.uq.propagate_operator_linearized(
        linearization,
        phx.uq.DiagonalCovariance(source_variance),
    )
    mapping = jnp.zeros((6, 8))
    mapping = mapping.at[0:2, 0:4].set(0.25)
    mapping = mapping.at[3:6, 4:8].set(0.25)
    expected_covariance = 0.25 * mapping @ mapping.T
    mask = jnp.asarray([[True, True, False], [True, True, True]])

    assert result.mean.dims == ("case", "__phydra_operator_point")
    assert jnp.array_equal(jnp.asarray(result.mean.data), jnp.where(mask, 2.0, 0.0))
    assert jnp.allclose(
        result.materialize_covariance().matrix,
        expected_covariance,
    )
    assert jnp.allclose(
        jnp.asarray(result.exact_variance().data),
        jnp.diag(expected_covariance).reshape((2, 3)),
    )


def test_operator_hilbert_covariance_requires_measure_and_stays_operator_valued():
    unweighted = phx.nn.operator.training.linearize_operator(
        _MaskedSourceMeanOperator(),
        _batch(1.0, source_points=4),
        "forcing",
    )
    covariance = phx.uq.DiagonalCovariance(jnp.full((2, 4), 0.25))
    with pytest.raises(ValueError, match="physical quadrature"):
        phx.uq.propagate_operator_linearized(
            unweighted,
            covariance,
            geometry="hilbert",
        )

    weighted = phx.nn.operator.training.linearize_operator(
        _MaskedSourceMeanOperator(),
        _weighted_batch(1.0, source_points=4),
        "forcing",
    )
    result = phx.uq.propagate_operator_linearized(
        weighted,
        covariance,
        geometry="hilbert",
    )
    mask = jnp.asarray([[True, True, False], [True, True, True]])
    cotangent = cx.Field(
        mask.astype(float),
        dims=result.mean.dims,
    )
    expected_input = 0.25 * weighted.adjoint(cotangent.data)
    expected_output = jnp.where(mask, weighted.pushforward(expected_input), 0.0)

    assert jnp.allclose(
        jnp.asarray(result.covariance_vector_product(cotangent).data),
        expected_output,
    )
    with pytest.raises(ValueError, match="only vector products"):
        result.exact_variance()
    with pytest.raises(ValueError, match="only vector products"):
        result.estimate_variance(jnp.asarray([0, 1], dtype=jnp.uint32), num_probes=8)
    with pytest.raises(ValueError, match="only vector products"):
        result.materialize_covariance()
