#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, TimeInterval
from phydrax.operators.differential import dt


def test_dt_time_only_vector():
    tdom = TimeInterval(0.0, 1.0)

    @tdom.Function("t")
    def y(t):
        return t**2

    t = jnp.linspace(0.0, 1.0, 7)
    out = jnp.asarray(dt(y)(frozendict({"t": cx.Field(t, dims=("t",))})).data)
    assert out.shape == (t.shape[0],)
    assert jnp.allclose(out, 2.0 * t)


def test_dt_spacetime_broadcasts_over_space(sample_batch):
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def u(t):
        return jnp.sin(t)

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(5, 6), key=0)
    out = jnp.asarray(dt(u, var="t")(batch).data)

    assert out.shape == (5, 6)
    t = jnp.asarray(batch.points["t"].data)
    assert jnp.allclose(out, jnp.cos(t)[None, :])


def test_dt_spacetime_depends_on_x_and_t(sample_batch):
    dom = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("x", "t")
    def u(x, t):
        return x[..., 0] * jnp.exp(t)

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(4, 5), key=1)
    x = jnp.asarray(batch.points["x"].data)
    t = jnp.asarray(batch.points["t"].data)
    out = jnp.asarray(dt(u, var="t")(batch).data)
    assert out.shape == (4, 5)
    assert jnp.allclose(out, x[..., 0:1] * jnp.exp(t)[None, :])
