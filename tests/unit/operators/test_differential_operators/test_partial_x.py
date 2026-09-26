#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp

import phydrax as phx
import phydrax.axes as cx
from phydrax._frozendict import frozendict
from phydrax.operators.differential import partial_x


def test_partial_x_point() -> None:
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def f(x: Any) -> Any:
        return x[0] ** 2 + x[1] ** 2

    px = partial_x(f)
    pts = frozendict({"x": cx.AxisArray(jnp.array([2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(px(pts).data)
    assert jnp.allclose(out, 4.0)


def test_partial_x_coord_separable(sample_grid: Any) -> None:
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component()
    batch = sample_grid(component, {"x": (6, 5)}, dense_blocks=(), key=0)

    @geom.Function("x")
    def f(x: Any) -> Any:
        x, y = x
        return x**2 + y**2

    px = partial_x(f)
    out = jnp.asarray(px(batch).data)
    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    X, _ = jnp.meshgrid(xs, ys, indexing="ij")
    assert jnp.allclose(out, 2.0 * X, atol=1e-6)


def test_partial_x_preserves_metadata() -> None:
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    u = geom.Function("x")(lambda x: x[0] ** 2).with_metadata(**{"tag": 1})
    out = partial_x(u)
    assert out.metadata == u.metadata
