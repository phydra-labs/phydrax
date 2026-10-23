#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._numerics._compensated import compensated_sum, compensated_sum_chunks


pytestmark = pytest.mark.strict_jax


def test_compensated_contracts() -> None:
    for dtype, large in ((jnp.float32, 1.0e8), (jnp.float64, 1.0e16)):
        values = jnp.asarray((large, 1.0, -large), dtype=dtype)
        reduced = jax.jit(compensated_sum)(values)
        gradient = jax.grad(lambda candidate: compensated_sum(candidate))(values)

        assert reduced == jnp.asarray(1.0, dtype=dtype)
        np.testing.assert_array_equal(gradient, jnp.ones_like(values))
    values = jnp.arange(24.0).reshape((2, 3, 4))
    expected = jnp.sum(values, axis=(0, 2), keepdims=True)
    actual = compensated_sum(values, axis=(0, 2), keepdims=True)
    empty = compensated_sum(jnp.empty((0, 3), dtype=jnp.float64), axis=0)
    integers = compensated_sum(jnp.arange(6).reshape((2, 3)), axis=0)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(empty, jnp.zeros((3,), dtype=jnp.float64))
    np.testing.assert_array_equal(integers, jnp.asarray((3, 5, 7)))
    complex_values = jnp.asarray((1.0e16 + 1.0e16j, 1.0 + 1.0j, -1.0e16 - 1.0e16j))
    nonfinite = jnp.asarray(((jnp.inf, 1.0), (-jnp.inf, 2.0)))

    assert compensated_sum(complex_values) == jnp.asarray(1.0 + 1.0j)
    np.testing.assert_allclose(
        compensated_sum(nonfinite, axis=0),
        jnp.sum(nonfinite, axis=0),
        equal_nan=True,
    )
    values = jnp.asarray(
        (
            (1.0e16, 1.0, -1.0e16),
            (-1.0e16, 2.0, 1.0e16),
        )
    )
    actual = jax.jit(jax.vmap(compensated_sum))(values)

    np.testing.assert_array_equal(actual, jnp.asarray((1.0, 2.0)))
    left = jnp.asarray(
        (
            (1.0e16, 1.0e8),
            (1.0, 3.0),
            (-1.0e16, -1.0e8),
        )
    )
    right = jnp.asarray(((2.0, -4.0), (-2.0, 4.0)))
    chunks = (left, right)

    actual = jax.jit(lambda values: compensated_sum_chunks(values, output_ndim=1))(chunks)
    combined = np.concatenate(tuple(np.asarray(value) for value in chunks), axis=0)
    expected = np.asarray(
        [math.fsum(combined[:, index].tolist()) for index in range(combined.shape[1])]
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype,large",
    [
        pytest.param(np.float32, 2**28, id="float32"),
        pytest.param(np.float64, 2**56, id="float64"),
        pytest.param(np.complex64, 2**28, id="complex64"),
        pytest.param(np.complex128, 2**56, id="complex128"),
    ],
)
def test_long_chunked_cancellation_retains_residuals(
    dtype: type[np.floating] | type[np.complexfloating], large: int
) -> None:
    block = np.zeros((32, 3), dtype=dtype)
    block[0] = large
    block[1] = (1, 2, -3)
    block[2] = -large
    if np.issubdtype(dtype, np.complexfloating):
        block = block + (1j * block).astype(dtype)
    host = np.tile(block, (137, 1))
    chunks = tuple(jnp.asarray(value) for value in np.split(host, (31, 2049)))
    expected = np.asarray(
        [
            math.fsum(host[:, column].real.tolist())
            + (
                1j * math.fsum(host[:, column].imag.tolist())
                if np.issubdtype(dtype, np.complexfloating)
                else 0
            )
            for column in range(3)
        ],
        dtype=dtype,
    )
    actual = jax.jit(lambda parts: compensated_sum_chunks(parts, output_ndim=1))(chunks)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])
def test_long_reduction_nonlinear_derivatives_and_batching(
    dtype: type[jnp.floating],
) -> None:
    values = jnp.arange(4096, dtype=dtype).reshape((2, 2048)) / 4096
    direction = jnp.full_like(values, 0.25)

    def loss(candidate: jax.Array) -> jax.Array:
        return compensated_sum(candidate * candidate)

    def derivatives(candidate: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        gradient = jax.grad(loss)(candidate)
        _, tangent = jax.jvp(loss, (candidate,), (direction,))
        _, hvp = jax.jvp(jax.grad(loss), (candidate,), (direction,))
        return gradient, tangent, hvp

    gradient, tangent, hvp = jax.jit(derivatives)(values)
    np.testing.assert_array_equal(gradient, 2 * values)
    np.testing.assert_array_equal(hvp, 2 * direction)
    np.testing.assert_array_equal(tangent, jnp.sum(2 * values * direction))
    batched = jax.jit(jax.vmap(compensated_sum))(values)
    np.testing.assert_array_equal(batched, jnp.sum(values, axis=1))


def test_long_reduction_empty_output_lanes() -> None:
    values = jnp.empty((4096, 0, 3), dtype=jnp.float64)
    result = jax.jit(lambda x: compensated_sum(x, axis=0, keepdims=True))(values)
    assert result.shape == (1, 0, 3)
    assert result.dtype == jnp.float64


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])
def test_long_reduction_nonfinite_lanes_remain_explicit(
    dtype: type[jnp.floating],
) -> None:
    values = jnp.ones((4096, 3), dtype=dtype)
    values = values.at[19, 0].set(jnp.inf).at[37, 1].set(jnp.nan)
    result = jax.jit(lambda x: compensated_sum(x, axis=0))(values)
    assert bool(jnp.isnan(result[0]))
    assert bool(jnp.isnan(result[1]))
    assert result[2] == 4096


@pytest.mark.parametrize("length", [0, 65, 4097], ids=["empty", "tail", "long"])
@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.complex64, np.complex128],
    ids=["float32", "float64", "complex64", "complex128"],
)
def test_seeded_recurrence_preserves_reference_components(
    length: int, dtype: type[np.floating] | type[np.complexfloating]
) -> None:
    from phydrax._numerics._compensated import _accumulate, _accumulate_reference

    rng = np.random.default_rng(713)
    values = rng.normal(size=(length, 4)).astype(dtype)
    high = np.asarray([-0.0, 0.0, 1e8, -1e8], dtype=dtype)
    correction = np.asarray([-0.0, 0.0, 3.0, -2.0], dtype=dtype)
    values[:, :2] = 0
    if np.issubdtype(dtype, np.complexfloating):
        values = values + (1j * values).astype(dtype)
        high = high + (1j * high).astype(dtype)
        correction = correction + (1j * correction).astype(dtype)
    if length:
        values[-1, 3] = np.nan
    arguments = (jnp.asarray(values), (jnp.asarray(high), jnp.asarray(correction)))
    actual = jax.jit(_accumulate)(*arguments)
    expected = jax.jit(_accumulate_reference)(*arguments)
    for observed, reference in zip(actual, expected, strict=True):
        observed_host, reference_host = np.asarray(observed), np.asarray(reference)
        real_dtype = observed_host.real.dtype
        unsigned = np.uint32 if real_dtype == np.dtype(np.float32) else np.uint64
        observed_components = observed_host.view(real_dtype)
        reference_components = reference_host.view(real_dtype)
        np.testing.assert_array_equal(
            np.isnan(observed_components), np.isnan(reference_components)
        )
        comparable = ~np.isnan(reference_components)
        np.testing.assert_array_equal(
            observed_components.view(unsigned)[comparable],
            reference_components.view(unsigned)[comparable],
        )


def test_reduction_compiler_identity_preserves_foreign_refusal() -> None:
    from jax.extend import core

    from phydrax._identity import execution_metadata_payload
    from phydrax._numerics._compensated_native import _forward_p

    values = jnp.ones(96, dtype=jnp.float64)
    baseline = execution_metadata_payload(jax.make_jaxpr(compensated_sum)(values))
    assert baseline == execution_metadata_payload(
        jax.make_jaxpr(compensated_sum)(values * 2)
    )
    assert baseline != execution_metadata_payload(
        jax.make_jaxpr(compensated_sum)(values[:65])
    )
    foreign = core.Primitive(_forward_p.name)
    foreign.def_impl(lambda value: value)
    foreign.def_abstract_eval(lambda value: value)
    with pytest.raises(TypeError, match="Foreign compiler primitive"):
        execution_metadata_payload(jax.make_jaxpr(foreign.bind)(values))


def test_reduction_structural_jacobian_preserves_lane_independence() -> None:
    from phydrax.sparse._structural_trace import trace_sparse_pattern

    def function(values: jax.Array, args: None) -> jax.Array:
        del args
        return compensated_sum(values.reshape((128, 3)) ** 2, axis=0)

    values = jnp.ones(384, dtype=jnp.float64)
    pattern = trace_sparse_pattern(
        function,
        values,
        None,
        source_size=384,
        target_size=3,
        derivative_kind="jacobian",
    )
    actual = set(
        zip(np.asarray(pattern.rows).tolist(), np.asarray(pattern.cols).tolist())
    )
    assert actual == {(column % 3, column) for column in range(384)}


def test_reduction_structural_hessian_retains_diagonal() -> None:
    from phydrax.sparse._structural_trace import trace_sparse_pattern

    def function(values: jax.Array, args: None) -> jax.Array:
        del args
        return compensated_sum(values * values)

    values = jnp.zeros(96, dtype=jnp.float64)
    pattern = trace_sparse_pattern(
        function,
        values,
        None,
        source_size=96,
        target_size=96,
        derivative_kind="hessian",
    )
    np.testing.assert_array_equal(pattern.rows, np.arange(96))
    np.testing.assert_array_equal(pattern.cols, np.arange(96))


@pytest.mark.parametrize(
    "dtype",
    [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
    ids=["float32", "float64", "complex64", "complex128"],
)
def test_zero_primal_retains_curvature_through_constant_chunks(
    dtype: type[jnp.floating] | type[jnp.complexfloating],
) -> None:
    values = jnp.zeros(96, dtype=dtype)
    constant = jnp.asarray([2.0, 1.0, -2.0], dtype=dtype)
    scalar = 1.0 + 0.25j if jnp.issubdtype(dtype, jnp.complexfloating) else 0.25
    direction = jnp.full_like(values, scalar)

    def loss(candidate: jax.Array) -> jax.Array:
        # The last chunk varies only through the incoming high/correction seed.
        return jnp.real(compensated_sum_chunks((candidate * candidate, constant)))

    def evaluate(candidate: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        value, gradient = jax.value_and_grad(loss)(candidate)
        _, hvp = jax.jvp(jax.grad(loss), (candidate,), (direction,))
        return value, gradient, hvp

    value, gradient, hvp = jax.jit(evaluate)(values)
    assert value == 1
    np.testing.assert_array_equal(gradient, jnp.zeros_like(values))
    np.testing.assert_array_equal(hvp, 2 * direction)
