#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, TimeInterval
from phydrax.operators.differential import curl


def test_curl_point(box3d):
    @box3d.Function("x")
    def u(x):
        return jnp.array([x[1], -x[0], 0.0])

    curl_u = curl(u, var="x")
    pts = frozendict({"x": cx.Field(jnp.array([1.0, 2.0, 3.0]), dims=(None,))})
    out = jnp.asarray(curl_u(pts).data)
    assert jnp.allclose(out, jnp.array([0.0, 0.0, -2.0]))


def test_curl_spacetime_depends_on_t(sample_batch, box3d):
    dom = box3d @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return jnp.array([x[1] * t, -x[0] * t, 0.0])

    curl_u = curl(u, var="x")
    pts = frozendict(
        {
            "x": cx.Field(jnp.array([1.0, 2.0, 3.0]), dims=(None,)),
            "t": cx.Field(jnp.array(0.5), dims=()),
        }
    )
    out = jnp.asarray(curl_u(pts).data)
    assert jnp.allclose(out, jnp.array([0.0, 0.0, -1.0]))

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(4, 5), key=0)
    out_b = jnp.asarray(curl_u(batch).data)
    assert out_b.shape == (4, 5, 3)
    t = jnp.asarray(batch.points["t"].data)
    z = -2.0 * jnp.broadcast_to(t[None, :], out_b.shape[:2])
    expected = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z], axis=-1)
    assert jnp.allclose(out_b, expected)


def test_curl_coord_separable(sample_coord_separable, box3d):
    component = box3d.component()
    batch = sample_coord_separable(component, {"x": (3, 4, 2)}, dense_blocks=(), key=0)

    @box3d.Function("x")
    def u(x):
        x, y, z = x
        return jnp.stack([y, -x, jnp.zeros_like(z)], axis=-1)

    curl_u = curl(u, var="x")
    out = jnp.asarray(curl_u(batch).data)
    assert out.shape[-1] == 3
    z = -2.0 * jnp.ones_like(jnp.asarray(out[..., 0]))
    expected = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z], axis=-1)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_curl_preserves_metadata(box3d):
    u = DomainFunction(
        domain=box3d,
        deps=("x",),
        func=lambda x: jnp.array([x[1], -x[0], 0.0]),
        metadata={"k": 1},
    )
    assert curl(u, var="x").metadata == u.metadata
