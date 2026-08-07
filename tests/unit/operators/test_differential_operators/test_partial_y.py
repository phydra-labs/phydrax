#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.operators.differential import partial_y


def test_partial_y_point():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    py = partial_y(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(py(pts).data)
    assert jnp.allclose(out, 6.0)


def test_partial_y_coord_separable(sample_grid):
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component()
    batch = sample_grid(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def f(x):
        x, y = x
        return x**2 + y**2

    py = partial_y(f)
    out = jnp.asarray(py(batch).data)
    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    _, Y = jnp.meshgrid(xs, ys, indexing="ij")
    assert jnp.allclose(out, 2.0 * Y, atol=1e-6)


def test_partial_y_preserves_metadata():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    u = geom.Function("x")(lambda x: x[1] ** 2).with_metadata(**{"tag": 1})
    out = partial_y(u)
    assert out.metadata == u.metadata
