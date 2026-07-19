#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _objective(integrand):
    geom = phx.domain.Interval1d(0.0, 1.0)
    return geom, phx.objectives.IntegralFunctional(
        component=geom.component(),
        integrand=integrand,
        num_points={"x": phx.domain.LegendreAxisSpec(8)},
        structure=phx.domain.ProductStructure((("x",),)),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(0),
    )


def test_integral_functional_rejects_implicit_complex_to_real_cast():
    geom = phx.domain.Interval1d(0.0, 1.0)
    density = geom.Function()(1.0 + 2.0j)
    _, objective = _objective(density)

    with pytest.raises(TypeError, match="requires a real scalar integrand"):
        objective.loss({"density": density}, key=jr.key(1))


def test_integral_functional_accepts_explicit_real_part():
    geom = phx.domain.Interval1d(0.0, 1.0)
    complex_density = geom.Function()(1.0 + 2.0j)
    density = phx.operators.real_part(complex_density)
    _, objective = _objective(density)

    value = objective.loss({"density": density}, key=jr.key(2))
    assert jnp.allclose(value, 1.0, atol=1e-12)
