#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp

# Enforced pipeline integration tests.
from phydrax.constraints import FunctionalConstraint
from phydrax.domain import Interval1d, ProductStructure
from phydrax.operators.differential import grad


def test_functional_constraint_mean_jit():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    component = geom.component()
    structure = ProductStructure((("x",),))

    constraint = FunctionalConstraint.from_operator(
        component=component,
        operator=lambda f: grad(f, var="x"),
        constraint_vars="u",
        num_points=16,
        structure=structure,
        reduction="mean",
    )

    loss_fn = eqx.filter_jit(lambda: constraint.loss({"u": u}))
    out = loss_fn()
    assert jnp.isfinite(out)
