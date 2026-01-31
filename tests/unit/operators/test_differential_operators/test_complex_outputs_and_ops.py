#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Square
from phydrax.operators.differential import directional_derivative, div, grad, laplacian


def test_grad_complex_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2 + 1j * x[1] ** 2

    g = grad(u)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(g(pts).data)
    assert jnp.allclose(out, jnp.array([4.0, 6.0j]))


def test_directional_derivative_complex_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2 + 1j * x[1] ** 2

    v = geom.Function()(jnp.array([1.0, 1.0]))
    dd = directional_derivative(u, v)

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(dd(pts).data)
    assert jnp.allclose(out, 4.0 + 6.0j)


def test_div_complex_vector_field_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], 1j * x[1]])

    div_u = div(u)
    pts = frozendict({"x": cx.Field(jnp.array([0.3, -1.1]), dims=(None,))})
    out = jnp.asarray(div_u(pts).data)
    assert jnp.allclose(out, 1.0 + 1.0j)


def test_laplacian_complex_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2 + 1j * x[1] ** 2

    lap_u = laplacian(u)
    pts = frozendict({"x": cx.Field(jnp.array([0.3, -1.1]), dims=(None,))})
    out = jnp.asarray(lap_u(pts).data)
    assert jnp.allclose(out, 2.0 + 2.0j)
