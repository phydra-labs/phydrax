#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square
from phydrax.operators.differential import partial_x


def test_partial_x_point():
    geom = Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    px = partial_x(f)
    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(px(pts).data)
    assert jnp.allclose(out, 4.0)


def test_partial_x_coord_separable(sample_coord_separable):
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    batch = sample_coord_separable(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def f(x):
        x, y = x
        return x**2 + y**2

    px = partial_x(f)
    out = jnp.asarray(px(batch).data)
    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    X, _ = jnp.meshgrid(xs, ys, indexing="ij")
    assert jnp.allclose(out, 2.0 * X, atol=1e-6)


def test_partial_x_preserves_metadata():
    geom = Square(center=(0.0, 0.0), side=2.0)
    u = DomainFunction(
        domain=geom, deps=("x",), func=lambda x: x[0] ** 2, metadata={"tag": 1}
    )
    out = partial_x(u)
    assert out.metadata == u.metadata
