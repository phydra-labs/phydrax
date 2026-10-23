#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit geometry dtype of operator samples, axes and latent grids with x64.

Omitted dtype keeps canonical float64 geometry. An explicit dtype is the only
geometry cast, values are stored as given, and every derived or reconstructed
geometry array keeps the selected precision.
"""

from typing import Any

import jax.numpy as jnp
import pytest

from phydrax.geometry.operator import TensorGridLatentGeometry
from phydrax.nn.operator import FunctionSamples, OperatorAxis, OperatorBatch
from phydrax.nn.operator.data import (
    function_samples_with_values,
    pad_function_samples,
    slice_operator_batch,
    stack_operator_batches,
)
from tests._support.assertions import assert_tree_equal


pytestmark = pytest.mark.strict_jax

_COORDINATES = [[0.0], [0.3], [0.7], [1.0]]
_WEIGHTS = [0.2, 0.3, 0.0, 0.5]


def _points(dtype: Any, /, **kwargs: Any) -> FunctionSamples:
    return FunctionSamples(
        values=jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        coordinates=jnp.asarray(_COORDINATES, dtype=jnp.float32),
        quadrature_weights=jnp.asarray(_WEIGHTS, dtype=jnp.float32),
        mask=jnp.asarray([True, True, True, False]),
        dtype=dtype,
        **kwargs,
    )


def _geometry_dtypes(samples: FunctionSamples, /) -> set[Any]:
    arrays = [
        samples.coordinates_array(),
        samples.quadrature(),
        samples.weights(),
        samples.weights(normalized=True),
    ]
    if samples.quadrature_weights is not None:
        arrays.append(samples.quadrature_weights)
    for axis in samples.axes:
        arrays.append(axis.nodes)
        if axis.quadrature_weights is not None:
            arrays.append(axis.quadrature_weights)
    return {array.dtype for array in arrays}


def test_omitted_geometry_dtype_keeps_canonical_float64() -> None:
    default = _points(None)
    assert default.geometry_dtype == "float64"
    assert _geometry_dtypes(default) == {jnp.dtype(jnp.float64)}
    # Values are caller data: stored as given, never cast.
    assert default.values is not None and default.values.dtype == jnp.float32
    assert_tree_equal(_points("float64"), default)
    axis = OperatorAxis("x", jnp.asarray([0.0, 0.5], dtype=jnp.float32))
    assert axis.nodes.dtype == jnp.float64
    assert_tree_equal(
        OperatorAxis("x", jnp.asarray([0.0, 0.5], dtype=jnp.float32), dtype="float64"),
        axis,
    )


def test_explicit_float32_geometry_reaches_every_derived_array() -> None:
    selected = _points(jnp.float32)
    assert selected.geometry_dtype == "float32"
    assert _geometry_dtypes(selected) == {jnp.dtype(jnp.float32)}
    unweighted = FunctionSamples(
        values=None,
        coordinates=jnp.asarray(_COORDINATES, dtype=jnp.float64),
        dtype=jnp.float32,
    )
    assert _geometry_dtypes(unweighted) == {jnp.dtype(jnp.float32)}
    values64 = FunctionSamples(
        values=jnp.ones((4,), dtype=jnp.float64),
        coordinates=jnp.asarray(_COORDINATES, dtype=jnp.float32),
        dtype=jnp.float32,
    )
    assert values64.values is not None and values64.values.dtype == jnp.float64


def test_float32_axes_select_tensor_grid_geometry() -> None:
    axes = (
        OperatorAxis("x", jnp.asarray([0.0, 0.5, 1.0]), dtype=jnp.float32),
        OperatorAxis(
            "y",
            jnp.asarray([0.0, 1.0]),
            quadrature_weights=jnp.asarray([0.5, 0.5]),
            dtype=jnp.float32,
        ),
    )
    grid = FunctionSamples(values=None, axes=axes)
    assert grid.geometry_dtype == "float32"
    assert _geometry_dtypes(grid) == {jnp.dtype(jnp.float32)}
    assert grid.quadrature().shape == (3, 2)

    with pytest.raises(ValueError, match="selected geometry dtype"):
        FunctionSamples(values=None, axes=axes, dtype=jnp.float64)
    with pytest.raises(ValueError, match="share one geometry dtype"):
        FunctionSamples(
            values=None, axes=(axes[0], OperatorAxis("y", jnp.asarray([0.0, 1.0])))
        )


@pytest.mark.parametrize(
    ("dtype", "message"),
    ((jnp.int32, "'int32'"), (jnp.complex64, "not real floating-point")),
)
def test_non_real_geometry_dtype_is_refused(dtype: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _points(dtype)
    with pytest.raises(ValueError, match=message):
        OperatorAxis("x", jnp.asarray([0.0, 1.0]), dtype=dtype)
    with pytest.raises(ValueError, match=message):
        TensorGridLatentGeometry((2, 3), dtype=dtype)


def test_geometry_dtype_requires_geometry() -> None:
    with pytest.raises(ValueError, match="requires axes or coordinates"):
        FunctionSamples(values=jnp.ones((3,), dtype=jnp.float32), dtype=jnp.float32)


def test_reconstructed_samples_keep_selected_geometry_dtype() -> None:
    selected = _points(jnp.float32)
    replaced = function_samples_with_values(selected, jnp.zeros((4,), dtype=jnp.float32))
    padded = pad_function_samples(selected, 6)
    assert replaced.support_id == selected.support_id
    assert _geometry_dtypes(replaced) == {jnp.dtype(jnp.float32)}
    assert _geometry_dtypes(padded) == {jnp.dtype(jnp.float32)}
    assert padded.sample_shape == (6,)

    def batch(count: int) -> OperatorBatch:
        return OperatorBatch(
            inputs={
                "u": FunctionSamples(
                    values=jnp.ones((count,), dtype=jnp.float32),
                    coordinates=jnp.linspace(0.0, 1.0, count, dtype=jnp.float32)[:, None],
                    quadrature_weights=jnp.full((count,), 1.0 / count, dtype=jnp.float32),
                    dtype=jnp.float32,
                )
            },
            queries={"query": selected},
        )

    stacked = stack_operator_batches((batch(3), batch(4)))
    assert stacked.case_shape == (2,)
    for samples in (*stacked.inputs.values(), *stacked.queries.values()):
        assert _geometry_dtypes(samples) == {jnp.dtype(jnp.float32)}
    sliced = slice_operator_batch(stacked, 1)
    for samples in (*sliced.inputs.values(), *sliced.queries.values()):
        assert _geometry_dtypes(samples) == {jnp.dtype(jnp.float32)}


def test_latent_grid_geometry_dtype_and_case_bounds_promotion() -> None:
    default = TensorGridLatentGeometry((3, 2))
    assert default.global_bounds.dtype == jnp.float64
    assert_tree_equal(TensorGridLatentGeometry((3, 2), dtype="float64"), default)

    selected = TensorGridLatentGeometry(
        (3, 2), bounds=[[0.0, 2.0], [-1.0, 1.0]], dtype=jnp.float32
    )
    assert selected.global_bounds.dtype == jnp.float32
    assert selected.coordinates((2,)).dtype == jnp.float32
    quadrature = selected.quadrature((2,))
    assert quadrature.dtype == jnp.float32
    assert jnp.allclose(jnp.sum(quadrature, axis=-1), 4.0)
    for axis in selected.axes():
        assert axis.nodes.dtype == jnp.float32
        assert axis.quadrature_weights is not None
        assert axis.quadrature_weights.dtype == jnp.float32
    assert FunctionSamples(values=None, axes=selected.axes()).geometry_dtype == "float32"

    boxed = TensorGridLatentGeometry((2, 2), bounds_policy="case_bbox", dtype=jnp.float32)
    source = jnp.asarray([[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]], dtype=jnp.float32)
    assert boxed.coordinates(source_coordinates=source).dtype == jnp.float32
    # Float64 sources promote the float32 grid; they are never downcast.
    widened = boxed.coordinates(source_coordinates=source.astype(jnp.float64))
    assert widened.dtype == jnp.float64
    assert boxed.quadrature(source_coordinates=source).dtype == jnp.float32
