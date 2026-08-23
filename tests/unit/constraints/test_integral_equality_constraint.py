#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.conditions import Moment
from phydrax.domain import Interval1d
from phydrax.terms import RandomizedMomentPenalty


def test_integral_equal_penalty_matches_exact_constant_integral():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    source = phx.integration.per_step(
        phx.integration.over(component),
        phx.integration.MonteCarloPlan(32),
    )

    one = geom.Function()(1.0)

    condition = Moment("u", component, lambda u: u, target=2.0)
    term = RandomizedMomentPenalty(condition, source)
    loss = term.loss({"u": one}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)

    condition2 = Moment("u", component, lambda u: u, target=0.0)
    term2 = RandomizedMomentPenalty(condition2, source)
    loss2 = term2.loss({"u": one}, key=jr.key(0))
    assert jnp.allclose(loss2, 4.0)
