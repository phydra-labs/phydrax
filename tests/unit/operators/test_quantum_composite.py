#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax.numpy as jnp
import pytest

import phydrax as phx


SIGMA_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def test_tensor_product_constructs_vector_and_matrix_products():
    time = phx.domain.TimeInterval(0.0, 1.0)
    zero = time.Function()(jnp.asarray([1.0, 0.0], dtype=complex))
    one = time.Function()(jnp.asarray([0.0, 1.0], dtype=complex))
    sigma_x = time.Function()(SIGMA_X)
    sigma_z = time.Function()(SIGMA_Z)

    zero_one_zero = phx.operators.tensor_product(zero, one, zero)
    xz = phx.operators.tensor_product(sigma_x, sigma_z)

    assert jnp.allclose(
        zero_one_zero.func(),
        jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert jnp.allclose(xz.func(), jnp.kron(SIGMA_X, SIGMA_Z))


def test_tensor_product_joins_compatible_function_domains():
    time = phx.domain.TimeInterval(0.0, 1.0)

    @time.Function("t")
    def rotating(t):
        return jnp.asarray([jnp.cos(t), jnp.sin(t)])

    fixed = time.Function()(jnp.asarray([1.0, 0.0]))
    product = phx.operators.tensor_product(rotating, fixed)
    point = 0.3

    assert product.deps == ("t",)
    assert jnp.allclose(product.func(point), jnp.kron(rotating.func(point), fixed.func()))


def test_embed_operator_places_local_operator_on_selected_subsystem():
    time = phx.domain.TimeInterval(0.0, 1.0)
    sigma_x = time.Function()(SIGMA_X)

    first = phx.operators.embed_operator(
        sigma_x,
        subsystem=0,
        subsystem_dims=(2, 3),
    )
    second = phx.operators.embed_operator(
        sigma_x,
        subsystem=1,
        subsystem_dims=(3, 2),
    )

    assert jnp.allclose(first.func(), jnp.kron(SIGMA_X, jnp.eye(3)))
    assert jnp.allclose(second.func(), jnp.kron(jnp.eye(3), SIGMA_X))


def test_partial_trace_recovers_product_density_factors():
    time = phx.domain.TimeInterval(0.0, 1.0)
    factor_a = time.Function()(jnp.asarray([[1.0, 0.2], [0.0, 0.7]], dtype=complex))
    factor_b = time.Function()(
        jnp.asarray(
            [[1.0, 0.0], [0.1j, 0.8], [0.0, 0.4]],
            dtype=complex,
        )
    )
    density_a = phx.operators.density_from_factor(factor_a)
    density_b = phx.operators.density_from_factor(factor_b)
    product = phx.operators.tensor_product(density_a, density_b)

    reduced_a = phx.operators.partial_trace(
        product,
        subsystem_dims=(2, 3),
        trace_out=1,
    )
    reduced_b = phx.operators.partial_trace(
        product,
        subsystem_dims=(2, 3),
        trace_out=0,
    )
    unchanged = phx.operators.partial_trace(
        product,
        subsystem_dims=(2, 3),
        trace_out=(),
    )
    total_trace = phx.operators.partial_trace(
        product,
        subsystem_dims=(2, 3),
        trace_out=(0, 1),
    )

    assert jnp.allclose(reduced_a.func(), density_a.func(), atol=1e-12)
    assert jnp.allclose(reduced_b.func(), density_b.func(), atol=1e-12)
    assert jnp.allclose(unchanged.func(), product.func(), atol=1e-12)
    assert jnp.allclose(total_trace.func(), 1.0, atol=1e-12)


def test_partial_trace_preserves_untraced_subsystem_order():
    time = phx.domain.TimeInterval(0.0, 1.0)
    densities = []
    for population in (0.2, 0.4, 0.7):
        factor = jnp.diag(jnp.sqrt(jnp.asarray([population, 1.0 - population])))
        densities.append(phx.operators.density_from_factor(time.Function()(factor)))
    product = phx.operators.tensor_product(*densities)
    middle = phx.operators.partial_trace(
        product,
        subsystem_dims=(2, 2, 2),
        trace_out=(2, 0),
    )

    assert jnp.allclose(middle.func(), densities[1].func(), atol=1e-12)


def test_composite_operators_reject_ambiguous_or_invalid_shapes():
    time = phx.domain.TimeInterval(0.0, 1.0)
    vector = time.Function()(jnp.ones((2,)))
    matrix = time.Function()(jnp.eye(2))
    rectangular = time.Function()(jnp.ones((2, 3)))
    larger = time.Function()(jnp.eye(3))
    invalid_factor: Any = object()

    with pytest.raises(ValueError, match="at least one"):
        phx.operators.tensor_product()
    with pytest.raises(TypeError, match="only DomainFunctions"):
        phx.operators.tensor_product(vector, invalid_factor)
    with pytest.raises(ValueError, match="all be vector-valued or all be matrix-valued"):
        phx.operators.tensor_product(vector, matrix).func()
    with pytest.raises(ValueError, match="square matrices"):
        phx.operators.tensor_product(matrix, rectangular).func()
    with pytest.raises(ValueError, match="product of subsystem_dims"):
        phx.operators.partial_trace(matrix, subsystem_dims=(2, 2), trace_out=1).func()
    with pytest.raises(ValueError, match="must be unique"):
        phx.operators.partial_trace(matrix, subsystem_dims=(2,), trace_out=(0, 0))
    with pytest.raises(ValueError, match=r"must be in \[0, 2\)"):
        phx.operators.partial_trace(matrix, subsystem_dims=(1, 2), trace_out=2)
    with pytest.raises(ValueError, match="positive integers"):
        phx.operators.partial_trace(matrix, subsystem_dims=(2, 0), trace_out=1)
    with pytest.raises(ValueError, match="selected subsystem"):
        phx.operators.embed_operator(larger, subsystem=0, subsystem_dims=(2, 2)).func()
