#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints import FunctionalConstraint
from phydrax.domain import Interval1d, ProductStructure


def test_functional_constraint_mean_and_integral_reductions():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    structure = ProductStructure((("x",),))

    u = geom.Function()(0.0)

    def operator(u_fn):
        return u_fn - 1.0

    c_mean = FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars="u",
        num_points=16,
        structure=structure,
        reduction="mean",
        weight=3.0,
    )
    loss_mean = c_mean.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 3.0)

    c_int = FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars="u",
        num_points=16,
        structure=structure,
        reduction="integral",
        weight=3.0,
    )
    loss_int = c_int.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_int, 6.0)


def test_functional_constraint_domainfunction_weight():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    structure = ProductStructure((("x",),))

    u = geom.Function()(0.0)

    @geom.Function("x")
    def w(x):
        return x[0] + 1.0

    def operator(u_fn):
        return u_fn - 1.0

    c_mean = FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars="u",
        num_points=4096,
        structure=structure,
        reduction="mean",
        weight=w,
    )
    loss_mean = c_mean.loss({"u": u}, key=jr.key(0))
    # E[x + 1] on [0, 2] equals 2.
    assert jnp.allclose(loss_mean, 2.0, rtol=5e-2, atol=5e-2)

    c_int = FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars="u",
        num_points=4096,
        structure=structure,
        reduction="integral",
        weight=w,
    )
    loss_int = c_int.loss({"u": u}, key=jr.key(1))
    # Integral of (x + 1) on [0, 2] equals 4.
    assert jnp.allclose(loss_int, 4.0, rtol=5e-2, atol=5e-2)
