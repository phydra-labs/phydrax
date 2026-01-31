#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import div_diag_k_grad


def test_div_diag_k_grad_scalar_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2 + x[1] ** 2

    @geom.Function("x")
    def k_vec(x):
        return jnp.array([x[0], 2.0 * x[1]])

    op = div_diag_k_grad(u, k_vec)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, -1.5]), dims=(None,))})
    out = jnp.asarray(op(pts).data)
    expected = 4.0 * 2.0 + 8.0 * (-1.5)
    assert jnp.allclose(out, expected)


def test_div_diag_k_grad_vector_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    @geom.Function("x")
    def k_vec(x):
        return jnp.array([x[0], 2.0 * x[1]])

    op = div_diag_k_grad(u, k_vec)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, -1.5]), dims=(None,))})
    out = jnp.asarray(op(pts).data)
    expected = jnp.array([4.0 * 2.0, 8.0 * (-1.5)])
    assert jnp.allclose(out, expected)


def test_div_diag_k_grad_coord_separable(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def u(x):
        x, y = x
        return x**2 + y**2

    @geom.Function("x")
    def k_vec(x):
        x, y = x
        return jnp.stack([x, 2.0 * y], axis=-1)

    out = jnp.asarray(div_diag_k_grad(u, k_vec)(batch).data)
    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    expected = 4.0 * X + 8.0 * Y
    assert jnp.allclose(out, expected, atol=1e-6)


def test_div_diag_k_grad_spacetime_k_depends_on_t(sample_batch):
    geom = Square(center=(0.0, 0.0), side=2.0)
    dom = geom @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return x[0] ** 2 + x[1] ** 2 + jnp.sin(t)

    @dom.Function("x", "t")
    def k_vec(x, t):
        k1 = 1.0 + x[0] + 3.0 * jnp.cos(t)
        k2 = 2.0 + 2.0 * x[1]
        return jnp.stack([k1, k2], axis=-1)

    op = div_diag_k_grad(u, k_vec, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=1)
    x = jnp.asarray(batch.points["x"].data)
    t = jnp.asarray(batch.points["t"].data)
    out = jnp.asarray(op(batch).data)

    X = x[..., 0:1]
    Y = x[..., 1:2]
    expected = 4.0 * X + 8.0 * Y + 6.0 + 6.0 * jnp.cos(t)[None, :]
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected)


def test_div_diag_k_grad_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: x[0] ** 2, metadata={"tag": 2}
    )
    k_vec = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: jnp.array([1.0 + x[0], 1.0 + x[1]])
    )
    assert div_diag_k_grad(u, k_vec).metadata == u.metadata
