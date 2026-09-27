#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.conditions import Initial
from phydrax.domain import FixedStart, Interval1d, TimeInterval
from phydrax.terms import ResidualPenalty


def test_initial_constraints_scenario_1() -> None:
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    component = dom.component({"t": FixedStart()})
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.MonteCarloPlan(8),
    )

    u = dom.Function()(1.0)
    condition = Initial("u", component, target=1.0)
    term = ResidualPenalty(condition, source)
    loss = term.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    dom = geom @ time

    component = dom.component()
    dom.Function()(0.0)

    with pytest.raises(ValueError):
        Initial("u", component, target=0.0)
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time
    component = domain.component({"t": FixedStart()})
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.MonteCarloPlan(8),
    )
    u = domain.Function("x", "t")(lambda x, t: x[0])
    condition = Initial("u", component, target=lambda x: x[0])
    term = ResidualPenalty(condition, source)

    loss = term.loss({"u": u}, key=jr.key(0), iter_=jnp.asarray(1))

    assert jnp.allclose(loss, 0.0)
