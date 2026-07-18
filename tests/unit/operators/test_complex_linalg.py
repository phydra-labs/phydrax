#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_complex_pointwise_transforms():
    time = phx.domain.TimeInterval(0.0, 1.0)

    @time.Function("t")
    def matrix(t):
        return jnp.asarray([[1.0 + 2.0j * t, 3.0j], [4.0, 5.0 - 1.0j]])

    point = 0.25
    value = matrix.func(point)
    assert jnp.allclose(phx.operators.conjugate(matrix).func(point), jnp.conj(value))
    assert jnp.allclose(phx.operators.adjoint(matrix).func(point), jnp.conj(value.T))
    assert jnp.allclose(phx.operators.real_part(matrix).func(point), jnp.real(value))
    assert jnp.allclose(phx.operators.imag_part(matrix).func(point), jnp.imag(value))


def test_adjoint_reverses_matrix_product():
    time = phx.domain.TimeInterval(0.0, 1.0)
    a_value = jnp.asarray([[1.0 + 1.0j, 2.0], [0.0, -1.0j]])
    b_value = jnp.asarray([[0.5, 1.0j], [2.0 - 1.0j, 3.0]])
    a = time.Function()(a_value)
    b = time.Function()(b_value)

    left = phx.operators.adjoint(a @ b)
    right = phx.operators.adjoint(b) @ phx.operators.adjoint(a)
    assert jnp.allclose(left.func(), right.func())


def test_complex_transforms_are_jittable_and_differentiable():
    time = phx.domain.TimeInterval(0.0, 1.0)

    def value(theta):
        @time.Function("t")
        def field(t):
            return theta * jnp.exp(1j * t)

        transformed = phx.operators.real_part(phx.operators.conjugate(field))
        return transformed.func(0.3)

    assert jnp.isfinite(jax.jit(jax.grad(value))(2.0))


def test_adjoint_rejects_vector_values():
    time = phx.domain.TimeInterval(0.0, 1.0)
    vector = time.Function()(jnp.asarray([1.0, 2.0j]))

    with pytest.raises(ValueError, match="at least two matrix axes"):
        phx.operators.adjoint(vector).func()
