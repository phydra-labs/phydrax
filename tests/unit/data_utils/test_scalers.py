#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.data_utils.scalers import (
    AffineScaler,
    MaxAbsScaler,
    MinMaxScaler,
    NormScaler,
    scaler_transform_fn,
    StdScaler,
)


def test_affine_scaler_defaults_and_roundtrip():
    scaler = AffineScaler()
    assert jnp.array_equal(scaler.reference_value, jnp.asarray(0.0))
    assert jnp.array_equal(scaler.scale_value, jnp.asarray(1.0))
    assert jnp.array_equal(scaler.alpha, jnp.asarray(1.0))
    assert jnp.array_equal(scaler.beta, jnp.asarray(0.0))

    custom = AffineScaler(
        reference_value=5.0,
        scale_value=2.0,
        alpha=3.0,
        beta=1.0,
    )
    x = jnp.asarray([7.0, 9.0, 11.0])
    transformed = custom.transform(x)
    assert jnp.allclose(transformed, jnp.asarray([4.0, 7.0, 10.0]))
    assert jnp.allclose(custom.inverse_transform(transformed), x)


def test_minmax_scaler_default_custom_and_axis_scaling():
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    default_scaler = MinMaxScaler(x)
    assert jnp.allclose(
        default_scaler.transform(x), jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    )

    custom_scaler = MinMaxScaler(x, min=-1.0, max=1.0)
    transformed = custom_scaler.transform(x)
    assert jnp.allclose(transformed, jnp.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]))
    assert jnp.allclose(custom_scaler.inverse_transform(transformed), x)

    matrix = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    axis0 = MinMaxScaler(matrix, axis=0)
    assert jnp.allclose(
        axis0.transform(matrix), jnp.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    )

    axis1 = MinMaxScaler(matrix, axis=1)
    assert jnp.allclose(
        axis1.transform(matrix), jnp.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    )


def test_minmax_scaler_constant_input_and_degenerate_target_range():
    x = jnp.asarray([3.0, 3.0, 3.0])
    scaler = MinMaxScaler(x)
    assert jnp.allclose(scaler.transform(x), jnp.zeros_like(x))

    with pytest.raises(ValueError, match="requires `min` and `max` to differ"):
        MinMaxScaler(x, min=1.0, max=1.0)


def test_maxabs_scaler_scaling_axis_and_zero_input():
    x = jnp.asarray([-4.0, -2.0, 0.0, 2.0, 4.0])
    scaler = MaxAbsScaler(x)
    transformed = scaler.transform(x)
    assert jnp.allclose(transformed, jnp.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]))
    assert jnp.allclose(scaler.inverse_transform(transformed), x)

    matrix = jnp.asarray([[-4.0, -3.0], [0.0, 0.0], [2.0, 6.0]])
    axis0 = MaxAbsScaler(matrix, axis=0)
    assert jnp.allclose(
        axis0.transform(matrix), jnp.asarray([[-1.0, -0.5], [0.0, 0.0], [0.5, 1.0]])
    )

    zeros = jnp.zeros((3,))
    zero_scaler = MaxAbsScaler(zeros)
    assert jnp.allclose(zero_scaler.transform(zeros), zeros)
    assert jnp.all(jnp.isfinite(zero_scaler.transform(zeros)))


def test_std_scaler_uses_requested_axis():
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = StdScaler(x)
    transformed = scaler.transform(x)
    assert jnp.allclose(jnp.mean(transformed), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.std(transformed), 1.0, atol=1e-6)
    assert jnp.allclose(scaler.inverse_transform(transformed), x)

    matrix = jnp.asarray([[1.0, 3.0], [2.0, 6.0]])
    axis1 = StdScaler(matrix, axis=1)
    transformed_axis1 = axis1.transform(matrix)
    assert jnp.allclose(jnp.mean(transformed_axis1, axis=1), jnp.zeros((2,)), atol=1e-6)
    assert jnp.allclose(jnp.std(transformed_axis1, axis=1), jnp.ones((2,)), atol=1e-6)

    constant = jnp.asarray([3.0, 3.0, 3.0])
    constant_scaler = StdScaler(constant)
    assert jnp.allclose(constant_scaler.transform(constant), jnp.zeros_like(constant))


def test_norm_scaler_scaling_axis_and_zero_input():
    x = jnp.asarray([3.0, 4.0])
    scaler = NormScaler(x)
    transformed = scaler.transform(x)
    assert jnp.allclose(transformed, jnp.asarray([0.6, 0.8]))
    assert jnp.allclose(jnp.linalg.norm(transformed), 1.0)
    assert jnp.allclose(scaler.inverse_transform(transformed), x)

    l1_scaler = NormScaler(x, ord=1)
    assert jnp.allclose(l1_scaler.transform(x), jnp.asarray([3.0 / 7.0, 4.0 / 7.0]))

    matrix = jnp.asarray([[3.0, 4.0], [6.0, 8.0]])
    axis1 = NormScaler(matrix, axis=1)
    assert jnp.allclose(axis1.transform(matrix), jnp.asarray([[0.6, 0.8], [0.6, 0.8]]))

    zeros = jnp.zeros((2,))
    zero_scaler = NormScaler(zeros)
    zero_transformed = zero_scaler.transform(zeros)
    assert jnp.allclose(zero_transformed, zeros)
    assert jnp.all(jnp.isfinite(zero_transformed))


def test_scaler_transform_fn_composes_input_and_output_scaling():
    input_scaler = AffineScaler(scale_value=2.0)
    output_scaler = AffineScaler(scale_value=10.0)

    def fn(x):
        return x + 1.0

    transformed_fn = scaler_transform_fn(
        fn,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
    )

    assert jnp.allclose(transformed_fn(jnp.asarray(4.0)), jnp.asarray(30.0))
