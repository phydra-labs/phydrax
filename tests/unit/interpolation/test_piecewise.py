#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax._interpolation import (
    cubic_hermite_interpolate,
    linear_interpolate,
    local_cubic_slopes,
    nearest_interpolate,
)


def test_nearest_ties_and_fill_support_are_explicit():
    nodes = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.arange(4.0)
    query = jnp.asarray([0.5, 1.5, -1.0, 4.0])

    lower = nearest_interpolate(
        nodes, values, query, tie_policy="lower", bounds="fill", fill_value=-1.0
    )
    round_even = nearest_interpolate(
        nodes, values, query, tie_policy="round_even", bounds="clip"
    )

    assert jnp.allclose(lower.values, jnp.asarray([0.0, 1.0, -1.0, -1.0]))
    assert jnp.array_equal(lower.support, jnp.asarray([True, True, False, False]))
    assert jnp.allclose(round_even.values, jnp.asarray([0.0, 2.0, 0.0, 3.0]))


def test_linear_interpolation_is_affine_exact_and_has_exact_derivative():
    nodes = jnp.asarray([-1.0, 0.5, 2.0])
    values = 3.0 * nodes - 2.0
    query = jnp.asarray([-0.25, 1.25])

    value = linear_interpolate(nodes, values, query).values
    derivative = linear_interpolate(nodes, values, query, derivative_order=1).values

    assert jnp.allclose(value, 3.0 * query - 2.0)
    assert jnp.allclose(derivative, 3.0)
    assert jnp.allclose(
        jax.jacfwd(lambda q: linear_interpolate(nodes, values, q).values)(query),
        jnp.eye(2) * 3.0,
    )


def test_local_cubic_slopes_use_one_sided_and_secant_average_rules():
    nodes = jnp.asarray([0.0, 1.0, 3.0, 6.0])
    values = jnp.asarray([0.0, 2.0, 8.0, 20.0])

    slopes = local_cubic_slopes(nodes, values)

    assert jnp.allclose(slopes, jnp.asarray([2.0, 2.5, 3.5, 4.0]))


def test_cubic_hermite_value_and_derivatives_match_interior_quadratic():
    nodes = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = nodes**2
    query = jnp.asarray([1.25, 1.5, 1.75])

    interpolated = cubic_hermite_interpolate(nodes, values, query).values
    first = cubic_hermite_interpolate(nodes, values, query, derivative_order=1).values
    second = cubic_hermite_interpolate(nodes, values, query, derivative_order=2).values

    assert jnp.allclose(interpolated, query**2)
    assert jnp.allclose(first, 2.0 * query)
    assert jnp.allclose(second, 2.0)
    assert jnp.allclose(
        jax.jacfwd(lambda q: cubic_hermite_interpolate(nodes, values, q).values)(query),
        jnp.diag(2.0 * query),
    )


def test_single_node_piecewise_methods_are_constant_and_complex_safe():
    nodes = jnp.asarray([2.0])
    values = jnp.asarray([[1.0 + 2.0j, 3.0 - 4.0j]])
    query = jnp.asarray([-1.0, 2.0, 5.0])

    linear = linear_interpolate(nodes, values, query).values
    cubic = cubic_hermite_interpolate(nodes, values, query).values
    derivative = cubic_hermite_interpolate(
        nodes, values, query, derivative_order=2
    ).values

    assert linear.shape == (3, 2)
    assert jnp.allclose(linear, values[0])
    assert jnp.allclose(cubic, values[0])
    assert jnp.allclose(derivative, 0.0)
