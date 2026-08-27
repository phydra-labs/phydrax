#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._numerics._compensated import compensated_sum, compensated_sum_chunks


@pytest.mark.parametrize(
    ("dtype", "large"),
    ((jnp.float32, 1.0e8), (jnp.float64, 1.0e16)),
)
def test_compensated_sum_recovers_cancellation_and_derivative(dtype, large):
    values = jnp.asarray((large, 1.0, -large), dtype=dtype)
    reduced = jax.jit(compensated_sum)(values)
    gradient = jax.grad(lambda candidate: compensated_sum(candidate))(values)

    assert reduced == jnp.asarray(1.0, dtype=dtype)
    np.testing.assert_array_equal(gradient, jnp.ones_like(values))


def test_compensated_chunks_match_accurate_signed_component_sums():
    left = jnp.asarray(
        (
            (1.0e16, 1.0e8),
            (1.0, 3.0),
            (-1.0e16, -1.0e8),
        )
    )
    right = jnp.asarray(((2.0, -4.0), (-2.0, 4.0)))
    chunks = (left, right)

    actual = jax.jit(
        lambda values: compensated_sum_chunks(values, output_ndim=1)
    )(chunks)
    combined = np.concatenate(tuple(np.asarray(value) for value in chunks), axis=0)
    expected = np.asarray(
        [math.fsum(combined[:, index].tolist()) for index in range(combined.shape[1])]
    )

    np.testing.assert_array_equal(actual, expected)


def test_compensated_sum_preserves_axes_shapes_and_empty_identity():
    values = jnp.arange(24.0).reshape((2, 3, 4))
    expected = jnp.sum(values, axis=(0, 2), keepdims=True)
    actual = compensated_sum(values, axis=(0, 2), keepdims=True)
    empty = compensated_sum(jnp.empty((0, 3), dtype=jnp.float64), axis=0)
    integers = compensated_sum(jnp.arange(6).reshape((2, 3)), axis=0)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(empty, jnp.zeros((3,), dtype=jnp.float64))
    np.testing.assert_array_equal(integers, jnp.asarray((3, 5, 7)))


def test_compensated_sum_preserves_complex_and_nonfinite_semantics():
    complex_values = jnp.asarray(
        (1.0e16 + 1.0e16j, 1.0 + 1.0j, -1.0e16 - 1.0e16j)
    )
    nonfinite = jnp.asarray(((jnp.inf, 1.0), (-jnp.inf, 2.0)))

    assert compensated_sum(complex_values) == jnp.asarray(1.0 + 1.0j)
    np.testing.assert_allclose(
        compensated_sum(nonfinite, axis=0),
        jnp.sum(nonfinite, axis=0),
        equal_nan=True,
    )


def test_compensated_sum_vmaps_independent_reductions():
    values = jnp.asarray(
        (
            (1.0e16, 1.0, -1.0e16),
            (-1.0e16, 2.0, 1.0e16),
        )
    )
    actual = jax.jit(jax.vmap(compensated_sum))(values)

    np.testing.assert_array_equal(actual, jnp.asarray((1.0, 2.0)))
