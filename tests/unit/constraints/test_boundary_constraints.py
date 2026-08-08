#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.domain import Boundary, Interval1d


def _penalty(condition, *, num_samples=8):
    source = phx.integration.per_step(
        phx.integration.mean_over(condition.on),
        phx.integration.MonteCarloPlan(num_samples),
    )
    return phx.terms.ResidualPenalty(condition, source)


def test_dirichlet_boundary_constraint_zero_when_satisfied():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})
    u = geom.Function()(2.0)
    c = _penalty(phx.conditions.Dirichlet("u", component, target=2.0))
    loss = c.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)


def test_neumann_boundary_constraint_zero_when_satisfied():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})

    u = geom.Function()(0.0)
    c = _penalty(phx.conditions.Neumann("u", component, target=0.0))
    loss = c.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)


def test_robin_boundary_constraint_zero_when_satisfied():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})

    u = geom.Function()(0.0)
    c = _penalty(
        phx.conditions.Robin(
            "u",
            component,
            dirichlet_coefficient=1.0,
            neumann_coefficient=1.0,
            target=0.0,
        )
    )
    loss = c.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)
