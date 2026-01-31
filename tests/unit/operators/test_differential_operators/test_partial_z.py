#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, TimeInterval
from phydrax.operators.differential import partial_z


def test_partial_z_point(box3d):
    @box3d.Function("x")
    def f(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0, 4.0]), dims=(None,))})
    out = jnp.asarray(partial_z(f)(pts).data)
    assert jnp.allclose(out, 8.0)


def test_partial_z_vector_output(box3d):
    @box3d.Function("x")
    def f(x):
        return jnp.array([x[0] ** 2, x[1] ** 2, x[2] ** 2])

    pts = frozendict({"x": cx.Field(jnp.array([2.0, 3.0, 4.0]), dims=(None,))})
    out = jnp.asarray(partial_z(f)(pts).data)
    assert jnp.allclose(out, jnp.array([0.0, 0.0, 8.0]))


def test_partial_z_spacetime_ignores_t(box3d):
    dom = box3d @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def f(x, t):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + t

    pts = frozendict(
        {
            "x": cx.Field(jnp.array([2.0, 3.0, 4.0]), dims=(None,)),
            "t": cx.Field(jnp.array(0.3), dims=()),
        }
    )
    out = jnp.asarray(partial_z(f)(pts).data)
    assert jnp.allclose(out, 8.0)


def test_partial_z_coord_separable(sample_coord_separable, box3d):
    component = box3d.component()
    batch = sample_coord_separable(component, {"x": (4, 5, 6)}, dense_blocks=(), key=0)

    @box3d.Function("x")
    def f(x):
        x, y, z = x
        return x**2 + y**2 + z**2

    out = jnp.asarray(partial_z(f)(batch).data)
    xs = jnp.asarray(batch.points["x"][0].data)
    ys = jnp.asarray(batch.points["x"][1].data)
    zs = jnp.asarray(batch.points["x"][2].data)
    _, _, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    assert jnp.allclose(out, 2.0 * Z, atol=1e-6)


def test_partial_z_preserves_metadata(box3d):
    u = DomainFunction(
        domain=box3d, deps=("x",), func=lambda x: x[2] ** 2, metadata={"tag": 1}
    )
    assert partial_z(u).metadata == u.metadata
