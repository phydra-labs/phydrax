import jax
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


def test_scalers_scenario_1() -> None:
    default = AffineScaler()
    assert jnp.array_equal(default.reference_value, jnp.asarray(0.0))
    assert jnp.array_equal(default.scale_value, jnp.asarray(1.0))
    assert jnp.array_equal(default.alpha, jnp.asarray(1.0))
    assert jnp.array_equal(default.beta, jnp.asarray(0.0))

    custom = AffineScaler(reference_value=5.0, scale_value=2.0, alpha=3.0, beta=1.0)
    values = jnp.asarray([7.0, 9.0, 11.0])
    transformed = custom.transform(values)
    assert jnp.allclose(transformed, jnp.asarray([4.0, 7.0, 10.0]))
    assert jnp.allclose(custom.inverse_transform(transformed), values)

    invalid = (
        {"reference_value": jnp.nan},
        {"scale_value": jnp.inf},
        {"scale_value": 0.0},
        {"alpha": 0.0},
        {"beta": -jnp.inf},
    )
    for arguments in invalid:
        with pytest.raises(ValueError):
            AffineScaler(**arguments)
    values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    default = MinMaxScaler(values)
    assert jnp.allclose(
        default.transform(values),
        jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
    )
    custom = MinMaxScaler(values, min=-1.0, max=1.0)
    transformed = custom.transform(values)
    assert jnp.allclose(transformed, jnp.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]))
    assert jnp.allclose(custom.inverse_transform(transformed), values)

    matrix = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert jnp.allclose(
        MinMaxScaler(matrix, axis=0).transform(matrix),
        jnp.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
    )
    assert jnp.allclose(
        MinMaxScaler(matrix, axis=1).transform(matrix),
        jnp.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    )

    constant = jnp.asarray([3.0, 3.0, 3.0])
    assert jnp.allclose(MinMaxScaler(constant).transform(constant), 0.0)
    with pytest.raises(ValueError, match="requires `min` and `max` to differ"):
        MinMaxScaler(constant, min=1.0, max=1.0)
    maxabs_values = jnp.asarray([-4.0, -2.0, 0.0, 2.0, 4.0])
    maxabs = MaxAbsScaler(maxabs_values)
    maxabs_transformed = maxabs.transform(maxabs_values)
    assert jnp.allclose(maxabs_transformed, jnp.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]))
    assert jnp.allclose(maxabs.inverse_transform(maxabs_transformed), maxabs_values)
    maxabs_matrix = jnp.asarray([[-4.0, -3.0], [0.0, 0.0], [2.0, 6.0]])
    assert jnp.allclose(
        MaxAbsScaler(maxabs_matrix, axis=0).transform(maxabs_matrix),
        jnp.asarray([[-1.0, -0.5], [0.0, 0.0], [0.5, 1.0]]),
    )

    standard_values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    standard = StdScaler(standard_values)
    standard_transformed = standard.transform(standard_values)
    assert jnp.allclose(jnp.mean(standard_transformed), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.std(standard_transformed), 1.0, atol=1e-6)
    assert jnp.allclose(standard.inverse_transform(standard_transformed), standard_values)
    standard_matrix = jnp.asarray([[1.0, 3.0], [2.0, 6.0]])
    standardized = StdScaler(standard_matrix, axis=1).transform(standard_matrix)
    assert jnp.allclose(jnp.mean(standardized, axis=1), jnp.zeros((2,)), atol=1e-6)
    assert jnp.allclose(jnp.std(standardized, axis=1), jnp.ones((2,)), atol=1e-6)

    norm_values = jnp.asarray([3.0, 4.0])
    norm = NormScaler(norm_values)
    norm_transformed = norm.transform(norm_values)
    assert jnp.allclose(norm_transformed, jnp.asarray([0.6, 0.8]))
    assert jnp.allclose(norm.inverse_transform(norm_transformed), norm_values)
    assert jnp.allclose(
        NormScaler(norm_values, ord=1).transform(norm_values),
        jnp.asarray([3.0 / 7.0, 4.0 / 7.0]),
    )
    norm_matrix = jnp.asarray([[3.0, 4.0], [6.0, 8.0]])
    assert jnp.allclose(
        NormScaler(norm_matrix, axis=1).transform(norm_matrix),
        jnp.asarray([[0.6, 0.8], [0.6, 0.8]]),
    )

    for scaler_type in (MaxAbsScaler, StdScaler, NormScaler):
        zeros = jnp.zeros((3,))
        transformed_zeros = scaler_type(zeros).transform(zeros)
        assert jnp.allclose(transformed_zeros, zeros), scaler_type
        assert jnp.all(jnp.isfinite(transformed_zeros)), scaler_type


def test_scaler_transform_composes_input_and_output_scaling() -> None:
    def function(value: jax.Array) -> jax.Array:
        return value + 1.0

    transformed = scaler_transform_fn(
        function,
        input_scaler=AffineScaler(scale_value=2.0),
        output_scaler=AffineScaler(scale_value=10.0),
    )
    assert jnp.allclose(transformed(jnp.asarray(4.0)), jnp.asarray(30.0))
