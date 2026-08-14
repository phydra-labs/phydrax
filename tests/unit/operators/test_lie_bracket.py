#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_lie_bracket_matches_analytic_vector_fields():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def x_field(x):
        return jnp.asarray([1.0, 0.0])

    @geom.Function("x")
    def y_field(x):
        return jnp.asarray([0.0, x[0]])

    bracket = phx.operators.lie_bracket(x_field, y_field)
    value = eqx.filter_jit(bracket.func)(jnp.asarray([0.2, -0.3]))
    assert jnp.allclose(value, jnp.asarray([0.0, 1.0]), atol=1e-12)


def test_lie_bracket_is_antisymmetric_and_satisfies_jacobi_identity():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def x_field(x):
        return jnp.asarray([x[1], 0.0])

    @geom.Function("x")
    def y_field(x):
        return jnp.asarray([0.0, x[0]])

    @geom.Function("x")
    def z_field(x):
        return jnp.asarray([x[0], x[1]])

    point = jnp.asarray([0.3, -0.4])
    xy = phx.operators.lie_bracket(x_field, y_field)
    yx = phx.operators.lie_bracket(y_field, x_field)
    jacobi = (
        phx.operators.lie_bracket(x_field, phx.operators.lie_bracket(y_field, z_field))
        + phx.operators.lie_bracket(y_field, phx.operators.lie_bracket(z_field, x_field))
        + phx.operators.lie_bracket(z_field, phx.operators.lie_bracket(x_field, y_field))
    )
    assert jnp.allclose(xy.func(point), -yx.func(point), atol=1e-12)
    assert jnp.allclose(jacobi.func(point), jnp.zeros((2,)), atol=1e-12)


def test_lie_bracket_rejects_nonvector_value_shape():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def scalar(x):
        return x[0]

    @geom.Function("x")
    def vector(x):
        return x

    bracket = phx.operators.lie_bracket(scalar, vector)
    with pytest.raises(ValueError, match="left operand must be a vector"):
        bracket.func(jnp.asarray([0.1, 0.2]))
