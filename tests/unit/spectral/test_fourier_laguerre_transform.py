#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mpmath as mp
import numpy as np
import pytest

from phydrax.discretization import (
    FourierLaguerrePlan,
    RadialLaguerrePlan,
    SphericalHarmonicPlan,
)


def _generalized_laguerre(degree: int, alpha: int, value: mp.mpf) -> mp.mpf:
    if degree == 0:
        return mp.mpf(1)
    previous = mp.mpf(1)
    current = alpha + 1 - value
    for order in range(1, degree):
        following = (
            (2 * order + alpha + 1 - value) * current - (order + alpha) * previous
        ) / (order + 1)
        previous, current = current, following
    return current


def _valid_spherical_modes(plan: SphericalHarmonicPlan) -> jax.Array:
    degree = jnp.arange(plan.bandlimit)[:, None]
    order = jnp.arange(-(plan.bandlimit - 1), plan.bandlimit)[None, :]
    return (jnp.abs(order) <= degree) & (degree >= abs(plan.spin))


def test_radial_laguerre_matches_order_two_basis_columns_and_exact_small_cases():
    tau = 1.7
    one = RadialLaguerrePlan(1, tau=tau)
    two = RadialLaguerrePlan(2, tau=tau)

    assert jnp.allclose(one.dimensionless_nodes, jnp.array([3.0]))
    assert jnp.allclose(one.nodes, jnp.array([3.0 * tau]))
    assert jnp.allclose(
        one.quadrature_weights,
        jnp.array([2.0 * tau**3 * np.exp(3.0)]),
        rtol=1e-13,
        atol=1e-13,
    )
    assert jnp.allclose(two.dimensionless_nodes, jnp.array([2.0, 6.0]))
    assert jnp.allclose(
        two.quadrature_weights,
        tau**3 * jnp.array([1.5 * np.exp(2.0), 0.5 * np.exp(6.0)]),
        rtol=1e-13,
        atol=1e-13,
    )

    plan = RadialLaguerrePlan(8, tau=tau)
    nodes = [mp.mpf(str(value)) for value in np.asarray(plan.dimensionless_nodes)]
    expected = np.zeros((8, 8), dtype=float)
    for node_index, node in enumerate(nodes):
        terminal = _generalized_laguerre(9, 2, node)
        gauss_weight = 10 * node / (9 * terminal**2)
        for degree in range(8):
            normalization = mp.sqrt(mp.factorial(degree) / mp.factorial(degree + 2))
            expected[node_index, degree] = float(
                mp.sqrt(gauss_weight)
                * normalization
                * _generalized_laguerre(degree, 2, node)
            )

    assert jnp.allclose(plan.balanced_basis, expected, rtol=2e-11, atol=2e-11)
    assert plan.orthogonality_defect <= 256 * 8 * np.finfo(float).eps


def test_radial_laguerre_roundtrips_parseval_tau_and_channel_contracts():
    first = RadialLaguerrePlan(8, tau=1.0)
    second = RadialLaguerrePlan(8, tau=2.5)
    coefficients = jr.normal(jr.key(1), (8,)) + 1j * jr.normal(jr.key(2), (8,))
    first_values = first.synthesis(coefficients)
    second_values = second.synthesis(coefficients)

    assert jnp.allclose(
        first.analysis(first_values), coefficients, rtol=1e-11, atol=1e-11
    )
    assert jnp.allclose(
        jnp.sum(first.quadrature_weights * jnp.abs(first_values) ** 2),
        jnp.sum(jnp.abs(coefficients) ** 2),
        rtol=1e-11,
        atol=1e-11,
    )
    assert jnp.allclose(second.nodes, 2.5 * first.nodes, rtol=1e-13, atol=1e-13)
    assert jnp.allclose(
        second.quadrature_weights,
        2.5**3 * first.quadrature_weights,
        rtol=1e-12,
        atol=1e-12,
    )
    assert jnp.allclose(
        second_values,
        2.5 ** (-1.5) * first_values,
        rtol=1e-11,
        atol=1e-11,
    )
    assert jnp.allclose(
        second.analysis(first_values),
        2.5**1.5 * first.analysis(first_values),
        rtol=1e-11,
        atol=1e-11,
    )

    channels = jnp.stack((first_values, -0.5 * first_values), axis=-1)[None, ...]
    transformed = eqx.filter_jit(lambda plan, values: plan.analysis(values))(
        first, channels
    )
    reconstructed = eqx.filter_jit(lambda plan, modes: plan.synthesis(modes))(
        first, transformed
    )
    gradient = jax.grad(lambda values: jnp.sum(jnp.abs(first.analysis(values)) ** 2))(
        jnp.real(first_values)
    )

    assert transformed.shape == (1, 8, 2)
    assert jnp.allclose(reconstructed, channels, rtol=1e-11, atol=1e-11)
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize("sampling", ("mw", "mwss", "dh", "gl"))
def test_fourier_laguerre_matches_explicit_separable_transform(sampling):
    radial = RadialLaguerrePlan(4, tau=0.8)
    angular = SphericalHarmonicPlan(4, sampling=sampling, reality=False)
    plan = FourierLaguerrePlan(radial, angular)
    valid = _valid_spherical_modes(angular)
    coefficients = (
        jr.normal(jr.key(3), plan.coefficient_shape)
        + 1j * jr.normal(jr.key(4), plan.coefficient_shape)
    ) * valid[None, ...]

    values = plan.synthesis(coefficients)
    actual = plan.analysis(values)
    angular_modes = angular.analysis(values)
    expected = jnp.moveaxis(
        radial.analysis(jnp.moveaxis(angular_modes, -3, -1)),
        -1,
        -3,
    )

    assert values.shape == plan.sample_shape
    assert actual.shape == plan.coefficient_shape
    assert jnp.allclose(actual, coefficients, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_fourier_laguerre_preserves_spin_batch_channel_and_execution_identity():
    radial = RadialLaguerrePlan(3)
    recursive_angular = SphericalHarmonicPlan(
        4, spin=1, reality=False, execution="recursive"
    )
    precomputed_angular = SphericalHarmonicPlan(
        4, spin=1, reality=False, execution="precomputed"
    )
    recursive = FourierLaguerrePlan(radial, recursive_angular)
    precomputed = FourierLaguerrePlan(radial, precomputed_angular)
    valid = _valid_spherical_modes(recursive_angular)
    coefficients = (
        jr.normal(jr.key(5), (2, *recursive.coefficient_shape, 2))
        + 1j * jr.normal(jr.key(6), (2, *recursive.coefficient_shape, 2))
    ) * valid[None, None, ..., None]

    values = eqx.filter_jit(lambda plan, modes: plan.synthesis(modes))(
        recursive, coefficients
    )
    actual = eqx.filter_jit(lambda plan, fields: plan.analysis(fields))(recursive, values)

    assert actual.shape == coefficients.shape
    assert jnp.allclose(actual, coefficients, rtol=1e-10, atol=1e-10)
    assert recursive.transform_id == precomputed.transform_id
    assert recursive.execution_id != precomputed.execution_id
    assert recursive.layout_id == precomputed.layout_id


def test_laguerre_plans_reject_invalid_configuration_shapes_and_resources():
    with pytest.raises(ValueError, match="positive"):
        RadialLaguerrePlan(0)
    with pytest.raises(ValueError, match="tau"):
        RadialLaguerrePlan(4, tau=0.0)
    with pytest.raises(ValueError, match="max_precompute_bytes"):
        RadialLaguerrePlan(16, max_precompute_bytes=1)
    radial = RadialLaguerrePlan(4)
    with pytest.raises(ValueError, match="Radial analysis expects"):
        radial.analysis(jnp.ones((5,)))
    with pytest.raises(ValueError, match="Radial synthesis expects"):
        radial.synthesis(jnp.ones((5,)))

    angular = SphericalHarmonicPlan(4)
    plan = FourierLaguerrePlan(radial, angular)
    with pytest.raises(ValueError, match="Fourier-Laguerre analysis expects"):
        plan.analysis(jnp.ones((4, 4, 8)))
    with pytest.raises(ValueError, match="Fourier-Laguerre synthesis expects"):
        plan.synthesis(jnp.ones((4, 4, 8)))
