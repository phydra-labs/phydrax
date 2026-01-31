#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.linalg import norm


def test_norm_simple_vector_function():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    norm_u = norm(u)
    pts = frozendict({"x": cx.Field(jnp.array([3.0, 4.0]), dims=(None,))})
    result = jnp.asarray(norm_u(pts).data)

    expected = 5.0  # sqrt(3^2 + 4^2)
    assert jnp.allclose(result, expected)


def test_norm_custom_order():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    norm_u = norm(u, order=1)
    pts = frozendict({"x": cx.Field(jnp.array([3.0, -4.0]), dims=(None,))})
    result = jnp.asarray(norm_u(pts).data)

    expected = 7.0  # |3| + |-4| = 7
    assert jnp.allclose(result, expected)


def test_norm_time_dependent_function():
    dom = Square(center=(0.0, 0.0), side=2.0) @ TimeInterval(0.0, 2.0)

    @dom.Function("x", "t")
    def u(x, t):
        return jnp.array([x[0] * t, x[1] * t])

    norm_u = norm(u)
    pts = frozendict(
        {
            "x": cx.Field(jnp.array([3.0, 4.0]), dims=(None,)),
            "t": cx.Field(jnp.array(2.0), dims=()),
        }
    )
    result = jnp.asarray(norm_u(pts).data)

    expected = 10.0  # 2*sqrt(3^2 + 4^2) = 2*5 = 10
    assert jnp.allclose(result, expected)


def test_norm_complex_function():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], 1j * x[1]])

    norm_u = norm(u)
    pts = frozendict({"x": cx.Field(jnp.array([3.0, 4.0]), dims=(None,))})
    result = jnp.asarray(norm_u(pts).data)

    expected = 5.0  # sqrt(3^2 + 4^2)
    assert jnp.allclose(result, expected)


def test_norm_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom,
        deps=("x",),
        func=lambda x: jnp.array([1.0, 2.0]),
        metadata={"tag": 1},
    )
    out = norm(u)
    assert out.metadata == u.metadata
