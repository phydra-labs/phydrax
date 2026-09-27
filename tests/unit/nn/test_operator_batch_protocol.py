#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


class _CoordinateFeature(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int) -> None:
        self.in_size = int(in_size)
        self.out_size = int(out_size)

    def __call__(self, value: Any, *, key: Any = None) -> Any:
        del key
        return jnp.full((self.out_size,), value[0] + value[-1])


class _Trunk(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self) -> None:
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value: Any, *, key: Any = None) -> Any:
        del key
        return jnp.array([1.0 + value[0]])


class _SourceValueKernel(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self) -> None:
        self.in_size = 4
        self.out_size = 1

    def __call__(self, value: Any, *, key: Any = None) -> Any:
        del key
        return value[:1]


def _point_batch(
    source_coordinates: Any, source_values: Any, query_coordinates: Any
) -> Any:
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


def test_operator_batch_protocol_scenario_1() -> None:
    # ty: ignore[invalid-argument-type]
    samples = phx.nn.operator.FunctionSamples(values=[[1.0], [2.0]])
    # ty: ignore[unresolved-attribute]
    assert samples.values.shape == (2, 1)
    with pytest.raises(TypeError, match="one array or None"):
        # ty: ignore[invalid-argument-type]
        phx.nn.operator.FunctionSamples(values={"u": jnp.ones((2,))})
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
        # ty: ignore[invalid-argument-type]
        feature_model=_CoordinateFeature(2, 1),
        latent_size=1,
        coord_dim=1,
    )
    model = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        # ty: ignore[invalid-argument-type]
        trunk=_Trunk(),
        coord_dim=1,
        latent_size=1,
    )

    prediction = model.evaluate(batch)
    output = prediction.field("output")
    assert output.values.shape == (2, 2)
    assert output.spec.channels == "scalar"
    assert prediction.case_axes == ("case",)
    assert output.values[1, 1] == 0.0
    assert not jnp.allclose(output.values[0], output.values[1])
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
    input_mask = batch.input("u").mask
    query_mask = batch.query("query").mask
    assert input_mask is not None
    assert query_mask is not None
    assert batch.case_axes == ("case",)
    assert batch.case_shape == (2,)
    assert batch.input("u").sample_shape == (4,)
    assert batch.query("query").sample_shape == (3,)
    assert jnp.array_equal(
        input_mask,
        jnp.array([[True, True, False, False], [True, True, True, True]]),
    )
    assert jnp.array_equal(
        query_mask,
        jnp.array([[True, True, True], [True, True, False]]),
    )

    selected = batch.take(1, axis="case")
    selected_values = selected.input("u").values
    selected_mask = selected.query("query").mask
    assert selected_values is not None
    assert selected_mask is not None
    assert selected.case_axes == ()
    assert selected.case_shape == ()
    assert selected_values.shape == (4,)
    assert jnp.array_equal(selected_mask, jnp.array([True, True, False]))


def _fixed_grid_case_loader(*, mask: Any = None) -> Any:
    op = phx.nn.operator
    axis = op.OperatorAxis(
        "x",
        jnp.asarray([0.0, 0.5, 1.0]),
        quadrature_weights=jnp.asarray([0.25, 0.5, 0.25]),
    )
    batch = op.OperatorBatch(
        inputs={
            "u": op.FunctionSamples(
                values=jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axes=(axis,)
            )
        },
        queries={"query": op.FunctionSamples(values=None, axes=(axis,), mask=mask)},
        case_axes=("case",),
        case_shape=(2,),
    )
    task = op.OperatorTask(
        "fixed-grid-collation",
        fields=(op.OperatorFieldSpec("u", query_name="query"),),
        queries=(
            op.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                quadrature="physical_required",
                fixed_geometry=True,
            ),
        ),
        problem=op.OperatorProblemSpec(
            source_query_relation="coincident", query_is_fixed=True
        ),
    )
    dataset = op.training.OperatorDataset(
        batch,
        # ty: ignore[invalid-argument-type]
        op.OperatorTargetBatch.from_arrays({"u": batch.input("u").values}, batch),
    )
    loader = op.training.OperatorBatchLoader(
        dataset, batch_size=2, shuffle=False, prefetch=0
    )
    return task, tuple(loader.epoch(0))[0].batch


def test_operator_batch_protocol_scenario_2() -> None:
    task, batch = _fixed_grid_case_loader()
    task.validate_batch(batch)
    integrated = phx.nn.operator.training.operator_integral(
        batch.input("u").values,
        batch.query("query"),
        case_shape=batch.case_shape,
    )
    assert jnp.allclose(integrated, jnp.asarray([2.0, 5.0]))
    for mask, expected in (
        (jnp.ones((2, 3), dtype="bool"), jnp.asarray([2.0, 5.0])),
        (
            jnp.asarray([[True, False, True], [True, True, False]]),
            jnp.asarray([1.0, 3.5]),
        ),
    ):
        task, batch = _fixed_grid_case_loader(mask=mask)
        with pytest.raises(ValueError, match="geometry shared by every case"):
            task.validate_batch(batch)
        integrated = phx.nn.operator.training.operator_integral(
            batch.input("u").values,
            batch.query("query"),
            case_shape=batch.case_shape,
        )
        assert jnp.allclose(integrated, expected)
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
    # ty: ignore[unresolved-attribute]
    assert sliced.input("u").coordinates.shape == (5, 1)
    # ty: ignore[unresolved-attribute]
    assert sliced.input("u").mask.shape == (2, 5)
    # ty: ignore[invalid-argument-type]
    assert jnp.array_equal(sliced.input("u").coordinates, coordinates)
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
        # ty: ignore[invalid-argument-type]
        kernel_model=_SourceValueKernel(),
        coord_dim=1,
    )

    output = model(batch)
    assert output.shape == (2, 2)
    assert jnp.allclose(output[0], jnp.array([2.3, 2.3]))
    assert jnp.allclose(output[1], jnp.array([5.0, 0.0]))
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
