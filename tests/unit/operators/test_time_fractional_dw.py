#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr

from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, TimeInterval
from phydrax.operators.differential import caputo_time_fractional_dw


def test_caputo_time_fractional_dw_time_only_smoke():
    dom = TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return jnp.sin(t)

    D = caputo_time_fractional_dw(u, alpha=1.5)
    y = jnp.asarray(
        D(frozendict({"t": cx.Field(jnp.array(1.0), dims=())}), key=jr.key(0)).data
    )
    assert jnp.ndim(y) == 0
    assert jnp.isfinite(y)


def test_caputo_time_fractional_dw_broadcasts_over_space(sample_batch):
    dom = Interval1d(-1.0, 1.0) @ TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return jnp.cos(t)

    D = caputo_time_fractional_dw(u, alpha=1.25, M=64)
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(2, 5), key=1)
    Y = jnp.asarray(D(batch, key=jr.key(1)).data)
    assert Y.shape == (2, 5)
    assert jnp.all(jnp.isfinite(Y))
