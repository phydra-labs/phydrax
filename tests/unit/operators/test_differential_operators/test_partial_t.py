#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import DomainFunction, Square, TimeInterval
from phydrax.operators.differential import partial_t


def test_partial_t_time_only_scalar():
    dom = TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def f(t):
        return t**2

    t = jnp.linspace(0.0, 1.0, 7)
    out = jnp.asarray(partial_t(f)(frozendict({"t": cx.Field(t, dims=("t",))})).data)
    assert out.shape == (t.shape[0],)
    assert jnp.allclose(out, 2.0 * t)


def test_partial_t_time_only_vector():
    dom = TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def f(t):
        return jnp.stack([t**2, t**3], axis=-1)

    t = jnp.linspace(0.0, 1.0, 5)
    out = jnp.asarray(partial_t(f)(frozendict({"t": cx.Field(t, dims=("t",))})).data)
    expected = jnp.stack([2.0 * t, 3.0 * t**2], axis=-1)
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected)


def test_partial_t_spacetime_broadcasts_over_space(sample_batch):
    dom = Square(center=(0.0, 0.0), side=2.0) @ TimeInterval(0.0, 1.0)

    @dom.Function("t")
    def u(t):
        return jnp.sin(t)

    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(4, 6), key=0)
    out = jnp.asarray(partial_t(u)(batch).data)

    assert out.shape == (4, 6)
    t = jnp.asarray(batch.points["t"].data)
    assert jnp.allclose(out, jnp.cos(t)[None, :])


def test_partial_t_preserves_metadata():
    dom = TimeInterval(0.0, 1.0)
    u = DomainFunction(
        domain=dom, deps=("t",), func=lambda t: t**2, metadata={"scale": 3}
    )
    assert partial_t(u).metadata == u.metadata
