#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx
from phydrax.nn.models.core._base import _AbstractOperatorModel


class _MaskedSourceMeanOperator(_AbstractOperatorModel):
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator_architecture_contract("DeepONet")

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
        if not isinstance(x, phx.nn.OperatorBatch):
            raise TypeError("_MaskedSourceMeanOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


def _batch(value: float, *, source_points: int) -> phx.nn.OperatorBatch:
    query_coordinates = jnp.asarray(
        [
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.5], [1.0]],
        ]
    )
    query = phx.nn.FunctionSamples(
        values=None,
        coordinates=query_coordinates,
        mask=jnp.asarray([[True, True, False], [True, True, True]]),
    )
    source_coordinates = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, source_points)[None, :, None],
        (2, source_points, 1),
    )
    source = phx.nn.FunctionSamples(
        values=jnp.full((2, source_points), value),
        coordinates=source_coordinates,
    )
    return phx.nn.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )


def test_input_function_prediction_matches_explicit_ragged_draw_loop():
    model = _MaskedSourceMeanOperator()
    batches = (_batch(1.0, source_points=2), _batch(3.0, source_points=4))
    stacked = phx.nn.stack_operator_batches(batches, case_axis="input_draw")

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
    assert jnp.allclose(prediction.predictive.samples.data, expected)
    assert prediction.predictive.sample_axes[0].source == "input"
    assert jnp.allclose(
        prediction.mean().field("output").values[prediction.output_mask()],
        2.0,
    )
    assert jnp.allclose(
        prediction.input_variance().field("output").values[prediction.output_mask()],
        1.0,
    )
