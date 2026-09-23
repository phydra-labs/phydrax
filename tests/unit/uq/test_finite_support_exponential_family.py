#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_finite_support_warm_mean_inversion_recovers_moments_and_probabilities():
    family = phx.uq.FiniteSupportExponentialFamily(
        jnp.asarray(((-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 4.0))),
        jnp.asarray((0.2, 0.3, 0.3, 0.2)),
        family_id="finite-support-test",
    )
    natural = family.natural(jnp.asarray((0.12, -0.08)))
    target = family.mean_from_natural(natural)
    cold = phx.uq.solve_finite_support_mean(family, target)
    warm = phx.uq.solve_finite_support_mean(
        family,
        target,
        initial=jnp.asarray((0.1, -0.05)),
    )
    portable = phx.uq.solve_finite_support_mean(
        family,
        target,
        plan=phx.uq.FiniteSupportNaturalSolvePlan(portable=True),
    )

    assert bool(cold.evidence.successful)
    assert bool(warm.evidence.successful)
    assert bool(portable.evidence.successful)
    np.testing.assert_allclose(cold.conversion.natural.values, natural.values, atol=2e-9)
    np.testing.assert_allclose(warm.conversion.natural.values, natural.values, atol=2e-9)
    np.testing.assert_allclose(
        portable.conversion.natural.values,
        natural.values,
        atol=2e-9,
    )
    np.testing.assert_allclose(jnp.sum(cold.probabilities), 1.0, atol=2e-14)
    np.testing.assert_allclose(
        cold.probabilities @ family.statistics,
        target.values,
        atol=2e-10,
    )


def test_finite_support_exterior_target_fails_without_plausible_coordinates():
    family = phx.uq.FiniteSupportExponentialFamily(
        jnp.asarray(((-1.0,), (0.0,), (1.0,))),
        jnp.ones(3),
        family_id="finite-support-exterior",
    )
    result = phx.uq.solve_finite_support_mean(family, jnp.asarray((2.0,)))

    assert not bool(result.evidence.successful)
    assert not bool(result.conversion.valid)
    assert jnp.all(jnp.isnan(result.conversion.natural.values))
