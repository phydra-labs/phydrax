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
