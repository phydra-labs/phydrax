#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import pytest

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


def test_einsum_constant_matrix_and_domain_vector():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def v(x):
        return jnp.array([x[0], x[1], x[0] - x[1]])

    k_mat = jnp.array(
        [
            [3.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 3.0],
        ]
    )

    out = einsum("ij,...j->...i", k_mat, v)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(out(pts).data)

    expected = jnp.asarray(jnp.einsum("ij,j->i", k_mat, jnp.array([2.0, 3.0, -1.0])))
    assert result.shape == (3,)
    assert jnp.allclose(result, expected)


def test_einsum_requires_domain_function_operand():
    with pytest.raises(ValueError, match="at least one DomainFunction operand"):
        einsum("ij,j->i", jnp.eye(2), jnp.array([1.0, 2.0]))
