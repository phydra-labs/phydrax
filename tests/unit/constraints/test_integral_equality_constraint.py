#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.constraints import IntegralEqualityConstraint
from phydrax.domain import Interval1d, SampleLayout


def test_integral_equal_penalty_matches_exact_constant_integral():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    structure = SampleLayout((("x",),))

    one = geom.Function()(1.0)

    c = IntegralEqualityConstraint.from_integrand(
        component=component,
        integrand=one,
        equal_to=2.0,
        sampling=phx.domain.PointSampling(32, layout=structure),
    )
    loss = c.loss({}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0)

    c2 = IntegralEqualityConstraint.from_integrand(
        component=component,
        integrand=one,
        equal_to=0.0,
        sampling=phx.domain.PointSampling(32, layout=structure),
    )
    loss2 = c2.loss({}, key=jr.key(0))
    assert jnp.allclose(loss2, 4.0)
