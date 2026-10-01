#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    PointCloudPlan,
    PointDiffusionOperator,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import HyperviscosityPlan, LocalStencilPolicy


def _cloud() -> tuple[Array, PreparedPointCloudDiscretization]:
    x = jnp.linspace(-1.0, 1.0, 11)
    cloud = PointCloudPlan(
        x[:, None],
        jnp.linspace(0.5, 1.5, 11),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=5,
    ).prepare()
    return x, cloud


@pytest.mark.parametrize("order", [1, 2, 3])
def test_repeated_dissipative_laplacian_sign_and_weighted_energy(order: int) -> None:
    x, cloud = _cloud()
    operator = HyperviscosityPlan(0.02, order=order).prepare(cloud)
    diffusion = PointDiffusionOperator(cloud, form="dissipative")
    u = jnp.sin(2 * x)
    expected = u
    for _ in range(order):
        expected = -diffusion.mv(expected)
    expected = -0.02 * expected
    np.testing.assert_allclose(operator.mv(u), expected, rtol=2e-12, atol=1e-8)
    assert float(operator.energy_rate(u)) <= 1e-8
    np.testing.assert_allclose(operator.mv(jnp.ones_like(x)), 0.0, atol=2e-7)
    v = jnp.cos(3 * x)
    mass = cloud.quadrature_weights
    np.testing.assert_allclose(
        jnp.vdot(u, mass * operator.mv(v)),
        jnp.vdot(operator.mv(u), mass * v),
        rtol=2e-11,
        atol=1e-8,
    )


def test_coefficient_scaling_and_spectral_estimate() -> None:
    x, cloud = _cloud()
    first = HyperviscosityPlan(0.01, order=2).prepare(cloud)
    second = HyperviscosityPlan(0.02, order=2).prepare(cloud)
    np.testing.assert_allclose(
        second.mv(jnp.sin(x)), 2 * first.mv(jnp.sin(x)), atol=1e-10
    )
    evidence = first.evidence
    assert float(evidence.spectral_radius_estimate) >= 0.0
    np.testing.assert_allclose(
        evidence.hyperviscosity_radius_estimate,
        0.01 * evidence.spectral_radius_estimate**2,
    )
    np.testing.assert_allclose(
        HyperviscosityPlan(0.0).prepare(cloud).mv(x), 0.0, atol=0.0
    )


@pytest.mark.parametrize(
    "coefficient,order", [(-0.1, 2), (float("nan"), 2), (0.1, 0), (0.1, 1.5)]
)
def test_invalid_explicit_parameters_refused(
    coefficient: float, order: int | float
) -> None:
    with pytest.raises(ValueError):
        HyperviscosityPlan(coefficient, order=order)
