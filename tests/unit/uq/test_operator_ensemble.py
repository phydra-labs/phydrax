#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

import phydrax as phx
from phydrax.nn.operator import AbstractOperatorModel


class _KeyedOperator(AbstractOperatorModel):
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    scale: Array

    def __init__(self, scale: float):
        self.in_size = "scalar"
        self.out_size = "scalar"
        self.scale = jnp.asarray(scale)

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, /, *, key=None):
        coordinates = batch.require_single_query().coordinates_array(
            case_shape=batch.case_shape
        )
        noise = 0.0 if key is None else jr.normal(key, ())
        return self.scale * coordinates[..., 0] + noise

    def __call__(self, x, /, *, key=None):
        if not isinstance(x, phx.nn.operator.OperatorBatch):
            raise TypeError("_KeyedOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


def _point_batch(*, shift: float = 0.0) -> phx.nn.operator.OperatorBatch:
    coordinates = jnp.asarray(
        [
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.5], [1.0]],
        ]
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates + shift,
        quadrature_weights=jnp.asarray([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]),
        mask=jnp.asarray([[True, True, False], [True, True, True]]),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((2, 3)),
        coordinates=coordinates,
        mask=jnp.asarray([[True, True, False], [True, True, True]]),
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )


def _tensor_batch() -> phx.nn.operator.OperatorBatch:
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, 8),
        periodic=True,
    )
    source = phx.nn.operator.FunctionSamples(values=jnp.ones((2, 8)), axes=(axis,))
    query = phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
    return phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )


def test_homogeneous_operator_ensemble_matches_explicit_member_loop():
    members = tuple(_KeyedOperator(scale) for scale in (1.0, 2.0, 3.0))
    ensemble = phx.uq.HomogeneousFunctionEnsemble.from_members(
        members,
        source_dim="member",
    )
    batch = _point_batch()
    key = jr.key(10)

    prediction = ensemble.predict_operator(
        batch,
        key=key,
        field_name="output",
        query_name="query",
    )
    explicit = jnp.stack(
        tuple(
            member.predict(batch, key=member_key).field("output").values
            for member, member_key in zip(
                members, jr.split(key, len(members)), strict=True
            )
        )
    )

    assert prediction.predictive.samples.dims == (
        "member",
        "case",
        "__phydra_operator_point",
    )
    assert jnp.array_equal(
        prediction.predictive.samples.data,
        jnp.where(prediction.output_mask()[None, ...], explicit, 0.0),
    )


def test_heterogeneous_operator_ensemble_matches_explicit_member_loop():
    members = (_KeyedOperator(1.0), _KeyedOperator(-2.0))
    ensemble = phx.uq.HeterogeneousFunctionEnsemble(
        members,
        source_dim="member",
    )
    batch = _point_batch()
    key = jr.key(11)

    prediction = ensemble.predict_operator(
        batch,
        key=key,
        field_name="output",
        query_name="query",
    )
    explicit = jnp.stack(
        tuple(
            member.predict(batch, key=member_key).field("output").values
            for member, member_key in zip(
                members, jr.split(key, len(members)), strict=True
            )
        )
    )

    assert jnp.array_equal(
        prediction.predictive.samples.data,
        jnp.where(prediction.output_mask()[None, ...], explicit, 0.0),
    )


def test_keyed_operator_sampling_is_reproducible_and_chunk_invariant():
    model = _KeyedOperator(1.0)
    batch = _point_batch()
    key = jr.key(12)

    unchunked = phx.uq.sample_operator_predictive(
        model,
        batch,
        num_samples=5,
        key=key,
        field_name="output",
        query_name="query",
        sample_dim="draw",
    )
    chunked = phx.uq.sample_operator_predictive(
        model,
        batch,
        num_samples=5,
        key=key,
        field_name="output",
        query_name="query",
        sample_dim="draw",
        sample_batch_size=2,
    )
    changed = phx.uq.sample_operator_predictive(
        model,
        batch,
        num_samples=5,
        key=jr.key(13),
        field_name="output",
        query_name="query",
        sample_dim="draw",
    )

    assert jnp.array_equal(
        unchunked.predictive.samples.data,
        chunked.predictive.samples.data,
    )
    assert not jnp.array_equal(
        unchunked.predictive.samples.data,
        changed.predictive.samples.data,
    )
    first_draw = unchunked.predictive.samples.data[0]
    difference = first_draw - model.__call_operator_batch__(batch, key=None)
    assert jnp.allclose(
        difference[unchunked.output_mask()],
        difference[unchunked.output_mask()][0],
    )


def test_operator_ensemble_preserves_crossed_input_sample_axis():
    stacked = phx.nn.operator.stack_operator_batches(
        (_point_batch(), _point_batch()),
        case_axis="input_draw",
    )
    members = (_KeyedOperator(1.0), _KeyedOperator(2.0))
    ensemble = phx.uq.HeterogeneousFunctionEnsemble(
        members,
        source_dim="member",
    )

    prediction = ensemble.predict_operator(
        stacked,
        key=jr.key(14),
        field_name="output",
        query_name="query",
        input_sample_axes=("input_draw",),
    )

    assert prediction.predictive.samples.dims == (
        "member",
        "input_draw",
        "case",
        "__phydra_operator_point",
    )
    assert tuple(axis.source for axis in prediction.predictive.sample_axes) == (
        "epistemic",
        "input",
    )
    assert prediction.case_axes == ("case",)


def test_stochastic_prediction_rejects_draw_dependent_query_geometry():
    stacked = phx.nn.operator.stack_operator_batches(
        (_point_batch(), _point_batch(shift=0.1)),
        case_axis="input_draw",
    )

    with pytest.raises(ValueError, match="geometry varies along input sample axis"):
        phx.uq.sample_operator_predictive(
            _KeyedOperator(1.0),
            stacked,
            num_samples=2,
            key=jr.key(15),
            field_name="output",
            query_name="query",
            input_sample_axes=("input_draw",),
        )


def test_fno_mc_dropout_and_inference_mode_operator_sampling():
    batch = _tensor_batch()
    model = phx.nn.operator.architectures.FNO(
        n_modes=(3,),
        width=6,
        depth=2,
        dropout=0.5,
        key=jr.key(16),
    )
    stochastic = phx.uq.sample_operator_predictive(
        model,
        batch,
        num_samples=6,
        key=jr.key(17),
        field_name="output",
        query_name="query",
    )
    deterministic = phx.uq.sample_operator_predictive(
        phx.nn.layers.inference_mode(model),
        batch,
        num_samples=3,
        key=jr.key(18),
        field_name="output",
        query_name="query",
    )

    assert jnp.max(stochastic.epistemic_variance().field("output").values) > 0.0
    assert jnp.allclose(
        deterministic.epistemic_variance().field("output").values,
        0.0,
    )
