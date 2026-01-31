#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square
from phydrax.operators.linalg import einsum


def test_einsum_simple_dot_product():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    @geom.Function("x")
    def v(x):
        return jnp.array([x[1], x[0]])

    einsum_uv = einsum("i,i->", u, v)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(einsum_uv(pts).data)

    expected = 2 * 3 + 2 * 3  # 2*3 + 3*2 = 12
    assert jnp.allclose(result, expected)


def test_einsum_outer_product():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0], x[1]])

    @geom.Function("x")
    def v(x):
        return jnp.array([x[1], x[0]])

    einsum_uv = einsum("i,j->ij", u, v)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(einsum_uv(pts).data)

    expected = jnp.array([[6.0, 4.0], [9.0, 6.0]])  # outer product
    assert result.shape == (2, 2)
    assert jnp.allclose(result, expected)


def test_einsum_matrix_vector_product():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def A(x):
        return jnp.array([[x[0], 0], [0, x[1]]])

    @geom.Function("x")
    def v(x):
        return jnp.array([x[1], x[0]])

    einsum_Av = einsum("ij,j->i", A, v)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(einsum_Av(pts).data)

    expected = jnp.array([6.0, 6.0])  # [2*3, 3*2]
    assert result.shape == (2,)
    assert jnp.allclose(result, expected)


def test_einsum_metadata_only_preserved_when_all_match():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom,
        deps=("x",),
        func=lambda x: jnp.array([x[0], x[1]]),
        metadata={"m": 1},
    )
    v = DomainFunction(
        domain=geom,
        deps=("x",),
        func=lambda x: jnp.array([x[1], x[0]]),
        metadata={"m": 2},
    )
    out = einsum("i,i->", u, v)
    assert out.metadata == {}
