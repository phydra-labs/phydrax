#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.domain import TimeInterval
from phydrax.operators.differential import grad


def test_grad_scalar_function_point():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    g = grad(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(g(pts).data)
    assert jnp.allclose(out, jnp.array([4.0, 6.0]))


def test_grad_vector_function_point():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    g = grad(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(g(pts).data)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[4.0, 0.0], [0.0, 6.0]]))


def test_grad_spacetime_var_x_ignores_t(sample_batch):
    dom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    ) @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def f(x, t):
        return x[0] ** 2 + x[1] ** 2 + t

    g = grad(f, var="x")
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 4), key=0)
    out = jnp.asarray(g(batch).data)
    assert out.shape == (3, 4, 2)
    x = jnp.asarray(batch.points["x"].data)
    expected = jnp.stack([2.0 * x[..., 0], 2.0 * x[..., 1]], axis=-1)
    assert jnp.allclose(out, expected[:, None, :])


def test_grad_coord_separable_matches_expected(sample_grid):
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component()
    batch = sample_grid(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def f(x):
        x, y = x
        return x**2 + y**2

    g = grad(f)
    out = jnp.asarray(g(batch).data)  # (nx, ny, 2)
    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    expected = jnp.stack([2.0 * X, 2.0 * Y], axis=-1)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_grad_preserves_metadata():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    u = geom.Function("x")(lambda x: x[0] ** 2).with_metadata(**{"scale": 3})
    g = grad(u)
    assert g.metadata == u.metadata
