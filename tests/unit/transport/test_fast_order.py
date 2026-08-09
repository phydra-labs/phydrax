#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_fast_soft_sort_preserves_value_contract(dtype):
    values = jnp.asarray([3.0, -1.0, 2.0, 0.5, 4.0], dtype=dtype)
    result = phx.transport.fast_soft_sort(values, temperature=0.8)

    tolerance = 2e-5 if dtype == jnp.float32 else 1e-11
    assert result.dtype == dtype
    assert result.shape == values.shape
    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(jnp.diff(result) >= -tolerance)
    assert jnp.min(result) >= jnp.min(values) - tolerance
    assert jnp.max(result) <= jnp.max(values) + tolerance
    assert jnp.allclose(jnp.sum(result), jnp.sum(values), atol=tolerance)
    assert not jnp.array_equal(result, jnp.sort(values))


def test_fast_soft_order_is_affine_and_permutation_equivariant():
    values = jnp.asarray([2.0, -0.5, 4.0, 1.0, 0.2])
    permutation = jnp.asarray([3, 0, 4, 1, 2])
    sorted_values = phx.transport.fast_soft_sort(values, temperature=0.9)
    ranks = phx.transport.fast_soft_rank(values, temperature=2.5)

    assert jnp.allclose(
        phx.transport.fast_soft_sort(3.0 * values + 7.0, temperature=0.9),
        3.0 * sorted_values + 7.0,
    )
    assert jnp.allclose(
        phx.transport.fast_soft_sort(values[permutation], temperature=0.9),
        sorted_values,
    )
    assert jnp.allclose(
        phx.transport.fast_soft_sort(-2.0 * values + 1.0, temperature=0.9),
        -2.0
        * phx.transport.fast_soft_sort(
            values,
            temperature=0.9,
            descending=True,
        )
        + 1.0,
    )
    assert jnp.allclose(
        phx.transport.fast_soft_rank(
            3.0 * values + 7.0,
            temperature=2.5,
        ),
        ranks,
    )
    assert jnp.allclose(
        phx.transport.fast_soft_rank(
            values[permutation],
            temperature=2.5,
        ),
        ranks[permutation],
    )


def test_fast_soft_rank_is_zero_based_and_tie_symmetric():
    values = jnp.asarray([3.0, 1.0, 1.0, 5.0, 2.0])
    ranks = phx.transport.fast_soft_rank(values, temperature=3.0)
    descending = phx.transport.fast_soft_rank(
        values,
        temperature=3.0,
        descending=True,
    )

    assert ranks.shape == values.shape
    assert jnp.all(ranks >= 0.0)
    assert jnp.all(ranks <= values.size - 1)
    assert jnp.allclose(jnp.sum(ranks), values.size * (values.size - 1) / 2)
    assert jnp.allclose(ranks[1], ranks[2])
    assert jnp.allclose(descending, values.size - 1 - ranks)
    assert jnp.array_equal(jnp.argsort(ranks), jnp.argsort(values, stable=True))


def test_fast_soft_order_handles_axes_fields_constants_and_singletons():
    values = jnp.asarray([[3.0, 1.0, 2.0], [4.0, -1.0, 0.0]])
    field = cx.Field(values, dims=("case", "sample"))
    named = phx.transport.fast_soft_sort(
        field,
        axis="sample",
        temperature=0.8,
    )
    plain = phx.transport.fast_soft_sort(values, axis=1, temperature=0.8)
    constant_ranks = phx.transport.fast_soft_rank(
        jnp.ones((2, 5)),
        axis=1,
        temperature=1.0,
    )

    assert named.dims == field.dims
    assert jnp.allclose(named.data, plain)
    assert jnp.allclose(constant_ranks, 2.0)
    assert jnp.array_equal(
        phx.transport.fast_soft_sort(jnp.asarray([4.0])),
        jnp.asarray([4.0]),
    )
    assert jnp.array_equal(
        phx.transport.fast_soft_rank(jnp.asarray([4.0])),
        jnp.asarray([0.0]),
    )


def test_fast_soft_order_supports_jit_vmap_forward_and_reverse_ad():
    values = jnp.asarray([-0.9, -0.2, 0.1, 0.8, 1.4])
    direction = jnp.asarray([0.2, -0.1, 0.4, -0.3, 0.5])
    coefficients = jnp.asarray([0.4, -0.7, 0.3, 0.8, -0.2])

    def operation(candidate):
        return phx.transport.fast_soft_sort(
            candidate,
            temperature=1.2,
        )

    eager = operation(values)
    compiled = jax.jit(operation)(values)
    _, tangent = jax.jvp(operation, (values,), (direction,))
    finite_difference = (
        operation(values + 1e-4 * direction)
        - operation(values - 1e-4 * direction)
    ) / 2e-4
    forward_jacobian = jax.jacfwd(operation)(values)
    reverse_jacobian = jax.jacrev(operation)(values)
    gradient = jax.grad(lambda candidate: jnp.dot(operation(candidate), coefficients))(
        values
    )
    batched = jax.vmap(operation)(jnp.stack((values, values + 0.3)))
    temperature_gradient = jax.grad(
        lambda temperature: jnp.sum(
            phx.transport.fast_soft_sort(values, temperature=temperature) ** 2
        )
    )(jnp.asarray(1.2))
    rank_temperature_gradient = jax.grad(
        lambda temperature: jnp.dot(
            phx.transport.fast_soft_rank(values, temperature=temperature),
            coefficients,
        )
    )(jnp.asarray(2.0))

    assert jnp.allclose(compiled, eager)
    assert jnp.allclose(tangent, finite_difference, rtol=3e-3, atol=3e-3)
    assert jnp.allclose(forward_jacobian, reverse_jacobian)
    assert batched.shape == (2, values.size)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 1e-6)
    assert jnp.isfinite(temperature_gradient)
    assert jnp.isfinite(rank_temperature_gradient)
    assert jnp.abs(rank_temperature_gradient) > 1e-6


def test_fast_soft_rank_has_informative_gradients_for_nearby_values():
    values = jnp.asarray([-0.2, -0.1, 0.0, 0.1, 0.2])
    jacobian = jax.jacfwd(
        lambda candidate: phx.transport.fast_soft_rank(
            candidate,
            temperature=4.0,
        )
    )(values)
    reverse = jax.grad(
        lambda candidate: jnp.dot(
            phx.transport.fast_soft_rank(candidate, temperature=4.0),
            jnp.asarray([0.1, 0.4, -0.3, 0.8, -0.2]),
        )
    )(values)

    assert jnp.all(jnp.isfinite(jacobian))
    assert jnp.all(jnp.isfinite(reverse))
    assert jnp.any(jnp.abs(jacobian) > 1e-6)


def test_fast_soft_order_temperature_controls_hard_approximation():
    values = jnp.asarray([3.0, -1.0, 2.0, 0.5, 4.0])
    order = jnp.argsort(values, stable=True)
    hard_ranks = jnp.zeros_like(values).at[order].set(
        jnp.arange(values.size, dtype=values.dtype)
    )
    low_sort = phx.transport.fast_soft_sort(values, temperature=0.03)
    high_sort = phx.transport.fast_soft_sort(values, temperature=1.5)
    low_ranks = phx.transport.fast_soft_rank(values, temperature=0.03)
    high_ranks = phx.transport.fast_soft_rank(values, temperature=6.0)

    assert jnp.linalg.norm(low_sort - jnp.sort(values)) < jnp.linalg.norm(
        high_sort - jnp.sort(values)
    )
    assert jnp.linalg.norm(low_ranks - hard_ranks) < jnp.linalg.norm(
        high_ranks - hard_ranks
    )



def test_fast_soft_order_rejects_invalid_inputs_eagerly_and_under_jit():
    values = jnp.asarray([3.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="at least one dimension"):
        phx.transport.fast_soft_sort(1.0)
    with pytest.raises(ValueError, match="nonempty"):
        phx.transport.fast_soft_sort(jnp.empty((0,)))
    with pytest.raises(TypeError, match="real-valued"):
        phx.transport.fast_soft_sort(jnp.asarray([1.0 + 2.0j]))
    with pytest.raises(ValueError, match="scalar"):
        phx.transport.fast_soft_sort(values, temperature=jnp.ones((2,)))
    with pytest.raises(TypeError, match="integer axis"):
        phx.transport.fast_soft_sort(values, axis="sample")
    with pytest.raises(TypeError, match="named axis"):
        phx.transport.fast_soft_sort(
            cx.Field(values, dims=("sample",)),
            axis=0,
        )

    invalid_temperature = eqx.filter_jit(
        lambda temperature: phx.transport.fast_soft_sort(
            values,
            temperature=temperature,
        )
    )
    invalid_values = eqx.filter_jit(phx.transport.fast_soft_rank)
    for temperature in (0.0, -1.0, jnp.inf, jnp.nan):
        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError),
            match="temperature must be finite and positive",
        ):
            jax.block_until_ready(invalid_temperature(jnp.asarray(temperature)))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="values must contain only finite values",
    ):
        jax.block_until_ready(invalid_values(jnp.asarray([0.0, jnp.nan, 1.0])))
