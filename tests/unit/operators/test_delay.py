#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, TimeInterval
from phydrax.operators.delay import delay


def test_delay_time_only_scalar_point():
    dom = TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return t

    u_delay = delay(u, 0.1)
    y = jnp.asarray(u_delay(frozendict({"t": cx.Field(jnp.array(1.0), dims=())})).data)
    assert jnp.isclose(y, 0.9)


def test_delay_time_only_vectorized_points():
    dom = TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return t

    u_delay = delay(u, 0.25)
    t = jnp.array([0.50, 1.00, 1.50])
    y = jnp.asarray(u_delay(frozendict({"t": cx.Field(t, dims=("t",))})).data)
    assert jnp.allclose(y, jnp.array([0.25, 0.75, 1.25]))


def test_delay_spacetime_time_only_tau_broadcasts(sample_batch):
    dom = Interval1d(-1.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def u(t):
        return t

    u_delay = delay(u, 0.1)

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(3, 5), key=0)
    y = jnp.asarray(u_delay(batch).data)

    t = jnp.asarray(batch.points["t"].data)
    assert y.shape == (3, 5)
    assert jnp.allclose(y, (t - 0.1)[None, :])


def test_delay_spacetime_space_dependent_tau(sample_batch):
    dom = Interval1d(-1.0, 1.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def u(t):
        return t

    @dom.Function("x")
    def tau(x):
        return 0.1 * (1.0 + x[..., 0])

    u_delay = delay(u, tau)

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(2, 3), key=1)
    x = jnp.asarray(batch.points["x"].data)
    t = jnp.asarray(batch.points["t"].data)
    y = jnp.asarray(u_delay(batch).data)

    assert y.shape == (2, 3)
    assert jnp.allclose(y, t[None, :] - 0.1 * (1.0 + x[..., 0:1]))


def test_delay_clip_time_min():
    dom = TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return t

    u_delay = delay(u, 0.2, clip_time_min=0.0)
    y = jnp.asarray(u_delay(frozendict({"t": cx.Field(jnp.array(0.1), dims=())})).data)
    assert jnp.isclose(y, 0.0)


def test_delay_vector_valued_time_only_point():
    dom = TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return jnp.stack([t, 2.0 * t], axis=-1)

    u_delay = delay(u, 0.25)
    y = jnp.asarray(u_delay(frozendict({"t": cx.Field(jnp.array(1.0), dims=())})).data)
    assert y.shape == (2,)
    assert jnp.allclose(y, jnp.array([0.75, 1.5]))
