#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx

# Enforced pipeline integration tests.
from phydrax.domain import Interval1d
from phydrax.operators.differential import grad


def test_residual_penalty_mean_jit():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    component = geom.component()
    condition = phx.conditions.Residual(
        "u",
        component,
        lambda f: grad(f, var="x"),
    )
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(condition.on),
            phx.integration.MonteCarloPlan(16),
        ),
    )

    loss_fn = eqx.filter_jit(lambda: term.loss({"u": u}))
    out = loss_fn()
    assert jnp.isfinite(out)
