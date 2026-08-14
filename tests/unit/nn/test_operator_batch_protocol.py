#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


class _CoordinateFeature(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int):
        self.in_size = int(in_size)
        self.out_size = int(out_size)

    def __call__(self, value, *, key=None):
        del key
        return jnp.full((self.out_size,), value[0] + value[-1])


class _Trunk(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value, *, key=None):
        del key
        return jnp.array([1.0 + value[0]])


class _SourceValueKernel(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 4
        self.out_size = 1

    def __call__(self, value, *, key=None):
        del key
        return value[:1]


def _point_batch(source_coordinates, source_values, query_coordinates):
    return phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.asarray(source_values),
                coordinates=jnp.asarray(source_coordinates),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.asarray(query_coordinates),
            )
        },
    )


def test_function_samples_values_are_one_array_or_none():
    samples = phx.nn.operator.FunctionSamples(values=[[1.0], [2.0]])
    assert samples.values.shape == (2, 1)
    with pytest.raises(TypeError, match="one array or None"):
        phx.nn.operator.FunctionSamples(values={"u": jnp.ones((2,))})


def test_per_case_geometry_weights_and_metrics_reduce_independently():
    coordinates = jnp.array(
        [
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.25], [0.75]],
        ]
    )
    quadrature = jnp.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
    mask = jnp.array([[True, True, False], [True, True, True]])
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=quadrature,
        mask=mask,
    )

    assert query.sample_shape == (3,)
    assert query.geometry_case_shape == (2,)
    assert jnp.allclose(
        jnp.sum(query.weights(normalized=True), axis=-1),
        jnp.ones((2,)),
    )

    prediction = jnp.array([[1.0, 1.0, 1000.0], [1.0, 2.0, 3.0]])
    target = jnp.zeros_like(prediction)
    error = phx.nn.operator.operator_l2_loss(
        prediction,
        target,
        query,
        reduction="none",
    )
    assert jnp.allclose(error, jnp.array([1.0, jnp.sqrt(5.9)]))


def test_stack_operator_batches_pads_ragged_points_and_slices_cases():
    first = _point_batch(
        [[0.0], [1.0]],
        [1.0, 2.0],
        [[0.0], [0.5], [1.0]],
    )
    second = _point_batch(
        [[0.0], [0.3], [0.6], [1.0]],
        [3.0, 4.0, 5.0, 6.0],
        [[0.2], [0.8]],
    )

    batch = phx.nn.operator.stack_operator_batches((first, second), case_axis="case")
    assert batch.case_axes == ("case",)
    assert batch.case_shape == (2,)
    assert batch.input("u").sample_shape == (4,)
    assert batch.query("query").sample_shape == (3,)
    assert jnp.array_equal(
        batch.input("u").mask,
        jnp.array([[True, True, False, False], [True, True, True, True]]),
    )
    assert jnp.array_equal(
        batch.query("query").mask,
        jnp.array([[True, True, True], [True, True, False]]),
    )

    selected = batch.take(1, axis="case")
    assert selected.case_axes == ()
    assert selected.case_shape == ()
    assert selected.input("u").values.shape == (4,)
    assert jnp.array_equal(selected.query("query").mask, jnp.array([True, True, False]))


def test_per_case_deeponet_uses_case_specific_source_and_query_geometry():
    source_coordinates = jnp.array(
        [
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.25], [0.75]],
        ]
    )
    query_coordinates = jnp.array(
        [
            [[0.0], [0.5]],
            [[0.25], [0.75]],
        ]
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((2, 3)),
        coordinates=source_coordinates,
        quadrature_weights=jnp.full((2, 3), 1.0 / 3.0),
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=query_coordinates,
        mask=jnp.array([[True, True], [True, False]]),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"u": source},
        queries={"query": query},
        case_axes=("case",),
    )
    branch = phx.nn.operator.architectures.IntegralBranchEncoder(
        feature_model=_CoordinateFeature(2, 1),
        latent_size=1,
        coord_dim=1,
    )
    model = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=_Trunk(),
        coord_dim=1,
        latent_size=1,
    )

    prediction = model.predict(batch)
    output = prediction.field("output")
    assert output.values.shape == (2, 2)
    assert output.spec.channels == "scalar"
    assert prediction.case_axes == ("case",)
    assert output.values[1, 1] == 0.0
    assert not jnp.allclose(output.values[0], output.values[1])


def test_local_integral_operator_supports_per_case_ragged_geometry():
    source_coordinates = jnp.array(
        [
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.2], [0.0]],
        ]
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.array([[1.0, 2.0, 3.0], [4.0, 6.0, 1000.0]]),
        coordinates=source_coordinates,
        quadrature_weights=jnp.array([[0.2, 0.3, 0.5], [0.5, 0.5, 0.0]]),
        mask=jnp.array([[True, True, True], [True, True, False]]),
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.array(
            [
                [[0.25], [0.75]],
                [[0.1], [0.0]],
            ]
        ),
        mask=jnp.array([[True, True], [True, False]]),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"u": source},
        queries={"query": query},
        case_axes=("case",),
    )
    model = phx.nn.operator.architectures.LocalIntegralOperator(
        kernel_model=_SourceValueKernel(),
        coord_dim=1,
    )

    output = model(batch)
    assert output.shape == (2, 2)
    assert jnp.allclose(output[0], jnp.array([2.3, 2.3]))
    assert jnp.allclose(output[1], jnp.array([5.0, 0.0]))


def test_operator_batch_rejects_mismatched_case_layouts():
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((3, 4)),
        coordinates=jnp.ones((2, 4, 1)),
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.ones((2, 5, 1)),
    )

    with pytest.raises(ValueError, match="inconsistent case shapes"):
        phx.nn.operator.OperatorBatch(
            inputs={"u": source},
            queries={"query": query},
            case_axes=("case",),
        )


def test_case_slicing_preserves_shared_coordinates_with_per_case_masks():
    coordinates = jnp.linspace(0.0, 1.0, 5)[:, None]
    source = phx.nn.operator.FunctionSamples(
        values=jnp.arange(20.0).reshape((4, 5)),
        coordinates=coordinates,
        mask=jnp.asarray(
            [
                [True, True, True, True, True],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, False, False, False, False],
            ]
        ),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"u": source},
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, coordinates=coordinates)
        },
        case_axes=("case",),
        case_shape=(4,),
    )
    sliced = batch.take(jnp.asarray([3, 1]))
    assert sliced.input("u").coordinates.shape == (5, 1)
    assert sliced.input("u").mask.shape == (2, 5)
    assert jnp.array_equal(sliced.input("u").coordinates, coordinates)
