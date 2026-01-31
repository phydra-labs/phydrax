#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square
from phydrax.operators.differential import div_K_grad, div_k_grad


def test_div_k_grad_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2 + x[1] ** 2

    @geom.Function("x")
    def k(x):
        return x[0] + 2.0 * x[1]

    op = div_k_grad(u, k)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, -1.5]), dims=(None,))})
    out = jnp.asarray(op(pts).data)
    expected = 6.0 * 2.0 + 12.0 * (-1.5)
    assert jnp.allclose(out, expected)


def test_div_k_grad_vector_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    @geom.Function("x")
    def k(x):
        return x[0] + 2.0 * x[1]

    op = div_k_grad(u, k)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, -1.5]), dims=(None,))})
    out = jnp.asarray(op(pts).data)
    expected = jnp.array([4.0 * 2.0 + 4.0 * (-1.5), 2.0 * 2.0 + 8.0 * (-1.5)])
    assert jnp.allclose(out, expected)


def test_div_k_grad_coord_separable(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def u(x):
        x, y = x
        return x**2 + y**2

    @geom.Function("x")
    def k(x):
        x, y = x
        return x + 2.0 * y

    out = jnp.asarray(div_k_grad(u, k)(batch).data)
    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    expected = 6.0 * X + 12.0 * Y
    assert jnp.allclose(out, expected, atol=1e-6)


def test_div_K_grad_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2 + x[1] ** 2

    @geom.Function("x")
    def K(x):
        return jnp.array([[x[0], x[1]], [x[1], x[0]]])

    op = div_K_grad(u, K)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, -1.5]), dims=(None,))})
    out = jnp.asarray(op(pts).data)
    assert jnp.allclose(out, 8.0 * 2.0)


def test_div_K_grad_vector_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    @geom.Function("x")
    def K(x):
        return jnp.array([[x[0], x[1]], [x[1], x[0]]])

    op = div_K_grad(u, K)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, -1.5]), dims=(None,))})
    out = jnp.asarray(op(pts).data)
    expected = jnp.array([6.0 * 2.0, 2.0 * 2.0])
    assert jnp.allclose(out, expected)


def test_div_K_grad_coord_separable(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (5, 4)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def u(x):
        x, y = x
        return x**2 + y**2

    @geom.Function("x")
    def K(x):
        x, y = x
        row0 = jnp.stack([x, y], axis=-1)
        row1 = jnp.stack([y, x], axis=-1)
        return jnp.stack([row0, row1], axis=-2)

    out = jnp.asarray(div_K_grad(u, K)(batch).data)
    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    X, _ = jnp.meshgrid(xs, ys, indexing="ij")
    expected = 8.0 * X
    assert jnp.allclose(out, expected, atol=1e-6)


def test_div_k_grad_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: x[0] ** 2, metadata={"scale": 1}
    )
    assert div_k_grad(u, 1.0).metadata == u.metadata
