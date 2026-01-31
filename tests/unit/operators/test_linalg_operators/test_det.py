#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Cube, DomainFunction, Square, TimeInterval
from phydrax.operators.linalg import det


def test_det_simple_matrix_function():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([[x[0], 0], [0, x[1]]])

    det_u = det(u)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(det_u(pts).data)

    expected = 6.0  # 2*3
    assert jnp.allclose(result, expected)


def test_det_time_dependent_matrix_function():
    dom = Square(center=(0.0, 0.0), side=2.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return jnp.array([[x[0] * t, 0.0], [0.0, x[1] * t]])

    det_u = det(u)
    pts = frozendict(
        {
            "x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,)),
            "t": cx.Field(jnp.array(0.5), dims=()),
        }
    )
    result = jnp.asarray(det_u(pts).data)

    expected = 1.5  # 2*3*0.5^2
    assert jnp.allclose(result, expected)


def test_det_complex_function():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([[x[0], 0], [0, 1j * x[1]]])

    det_u = det(u)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(det_u(pts).data)

    expected = 6.0j  # i*2*3
    assert jnp.allclose(result, expected)


def test_det_non_diagonal_matrix():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([[x[0], x[1]], [x[1], x[0]]])

    det_u = det(u)
    pts = frozendict({"x": cx.Field(jnp.array([3.0, 2.0]), dims=(None,))})
    result = jnp.asarray(det_u(pts).data)

    expected = 5.0  # 3*3 - 2*2 = 9 - 4 = 5
    assert jnp.allclose(result, expected)


def test_det_3x3_matrix():
    geom = Cube(center=(0.0, 0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([[x[0], x[1], x[2]], [x[2], x[0], x[1]], [x[1], x[2], x[0]]])

    det_u = det(u)
    pts = frozendict({"x": cx.Field(jnp.array([1.0, 2.0, 3.0]), dims=(None,))})
    result = jnp.asarray(det_u(pts).data)

    # For the matrix [[1, 2, 3], [3, 1, 2], [2, 3, 1]], the determinant is 18.
    expected = 18.0
    assert jnp.allclose(result, expected)


def test_det_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: jnp.eye(2), metadata={"tag": "keep"}
    )
    out = det(u)
    assert out.metadata == u.metadata
