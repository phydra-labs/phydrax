#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints import PointSetConstraint
from phydrax.domain import Interval1d


def test_pointset_penalty_mean_and_sum():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    points = {"x": jnp.array([[0.0], [0.5], [1.0]], dtype=float)}
    u = geom.Function()(0.0)

    def residual(functions):
        return functions["u"] - 1.0

    c_mean = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        reduction="mean",
        weight=2.0,
    )
    loss_mean = c_mean.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 2.0)

    c_sum = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        reduction="sum",
        weight=2.0,
    )
    loss_sum = c_sum.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_sum, 6.0)
