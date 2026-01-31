#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints import IntegralEqualityConstraint
from phydrax.domain import Interval1d, ProductStructure


def test_integral_equal_penalty_matches_exact_constant_integral():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    structure = ProductStructure((("x",),))

    one = geom.Function()(1.0)

    c = IntegralEqualityConstraint.from_integrand(
        component=component,
        integrand=one,
        equal_to=2.0,
        num_points=32,
        structure=structure,
    )
    loss = c.loss({}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)

    c2 = IntegralEqualityConstraint.from_integrand(
        component=component,
        integrand=one,
        equal_to=0.0,
        num_points=32,
        structure=structure,
    )
    loss2 = c2.loss({}, key=jr.key(0))
    assert jnp.allclose(loss2, 4.0)
