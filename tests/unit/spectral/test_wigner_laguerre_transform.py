#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.discretization import (
    RadialLaguerrePlan,
    SphericalHarmonicPlan,
    WignerLaguerrePlan,
    WignerTransformPlan,
)


def _wigner_valid(plan: WignerTransformPlan) -> jax.Array:
    order_n = jnp.arange(-(plan.directional_bandlimit - 1), plan.directional_bandlimit)[
        :, None, None
    ]
    degree = jnp.arange(plan.bandlimit)[None, :, None]
    order_m = jnp.arange(-(plan.bandlimit - 1), plan.bandlimit)[None, None, :]
    return (
        (jnp.abs(order_n) <= degree)
        & (jnp.abs(order_m) <= degree)
        & (degree >= plan.lower_bandlimit)
    )


def test_wigner_plan_locks_raw_haar_and_spherical_normalization():
    plan = WignerTransformPlan(4, 2)
    coefficients = plan.analysis(jnp.ones(plan.sample_shape))
    center_n = plan.directional_bandlimit - 1
    center_m = plan.bandlimit - 1
    haar_mass = (
        jnp.sum(plan.alpha_quadrature_weights)
        * jnp.sum(plan.beta_quadrature_weights)
        * jnp.sum(plan.gamma_quadrature_weights)
    )

    expected = (
        jnp.zeros(plan.coefficient_shape, dtype=complex)
        .at[center_n, 0, center_m]
        .set(8.0 * jnp.pi**2)
    )
    assert jnp.allclose(haar_mass, 8.0 * jnp.pi**2, rtol=1e-13, atol=1e-13)
    assert jnp.allclose(coefficients, expected, rtol=1e-11, atol=1e-11)

    angular = SphericalHarmonicPlan(4, reality=False)
    spherical_coefficients = (
        jnp.zeros(angular.coefficient_shape, dtype=jnp.complex128)
        .at[2, 4]
        .set(1.0 + 0.3j)
    )
    spherical_values = angular.synthesis(spherical_coefficients)
    gamma_invariant = jnp.broadcast_to(
        spherical_values,
        (plan.sample_shape[0], *spherical_values.shape),
    )
    actual = plan.analysis(gamma_invariant)
    degree_factor = (
        2.0 * jnp.pi * jnp.sqrt(4.0 * jnp.pi / (2.0 * jnp.arange(plan.bandlimit) + 1.0))
    )
    expected_n_zero = degree_factor[:, None] * spherical_coefficients

    assert jnp.allclose(actual[center_n], expected_n_zero, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(actual[:center_n], 0.0, rtol=0.0, atol=1e-11)
    assert jnp.allclose(actual[center_n + 1 :], 0.0, rtol=0.0, atol=1e-11)


def test_wigner_plan_roundtrips_masks_inactive_capacity_and_real_conjugacy():
    plan = WignerTransformPlan(5, 3)
    valid = _wigner_valid(plan)
    coefficients = (
        jr.normal(jr.key(1), plan.coefficient_shape)
        + 1j * jr.normal(jr.key(2), plan.coefficient_shape)
    ) * valid

    actual = plan.analysis(plan.synthesis(coefficients))
    poisoned = jnp.where(valid, 0.0, jnp.nan + 1j * jnp.inf)
    sanitized_values = plan.synthesis(poisoned)
    real_values = jr.normal(jr.key(3), plan.sample_shape)
    real_coefficients = plan.analysis(real_values)
    center = plan.directional_bandlimit - 1
    negative_n = jnp.arange(-(plan.directional_bandlimit - 1), 0)
    order_m = jnp.arange(-(plan.bandlimit - 1), plan.bandlimit)
    expected_negative = jnp.conj(jnp.flip(real_coefficients[center + 1 :], axis=(0, -1)))
    expected_negative *= (-1.0) ** jnp.abs(negative_n)[:, None, None]
    expected_negative *= (-1.0) ** jnp.abs(order_m)[None, None, :]

    assert jnp.allclose(actual, coefficients, rtol=1e-10, atol=1e-10)
    assert jnp.all(jnp.isfinite(sanitized_values))
    assert jnp.allclose(sanitized_values, 0.0, rtol=0.0, atol=1e-13)
    assert jnp.allclose(
        real_coefficients[:center], expected_negative, rtol=1e-10, atol=1e-10
    )


def test_recursive_and_precomputed_wigner_plans_share_semantics():
    recursive = WignerTransformPlan(3, 2, execution="recursive")
    precomputed = WignerTransformPlan(3, 2, execution="precomputed")
    valid = _wigner_valid(recursive)
    coefficients = (
        jr.normal(jr.key(4), recursive.coefficient_shape)
        + 1j * jr.normal(jr.key(5), recursive.coefficient_shape)
    ) * valid
    values = recursive.synthesis(coefficients)

    recursive_coefficients = recursive.analysis(values)
    precomputed_coefficients = precomputed.analysis(values)

    assert recursive.transform_id == precomputed.transform_id
    assert recursive.layout_id == precomputed.layout_id
    assert recursive.execution_id != precomputed.execution_id
    assert jnp.allclose(
        precomputed_coefficients, recursive_coefficients, rtol=1e-10, atol=1e-10
    )
    assert jnp.allclose(
        precomputed.synthesis(precomputed_coefficients), values, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("sampling", ("mw", "mwss", "dh", "gl"))
def test_wigner_laguerre_roundtrips_all_exact_samplings(sampling):
    radial = RadialLaguerrePlan(3, tau=0.7)
    wigner = WignerTransformPlan(3, 2, sampling=sampling)
    plan = WignerLaguerrePlan(radial, wigner)
    valid = _wigner_valid(wigner)
    coefficients = (
        jr.normal(jr.key(6), plan.coefficient_shape)
        + 1j * jr.normal(jr.key(7), plan.coefficient_shape)
    ) * valid[None, ...]

    values = plan.synthesis(coefficients)
    actual = plan.analysis(values)

    assert values.shape == plan.sample_shape
    assert actual.shape == plan.coefficient_shape
    assert jnp.allclose(actual, coefficients, rtol=1e-10, atol=1e-10)


def test_wigner_laguerre_handles_batch_channels_jit_and_gradients():
    radial = RadialLaguerrePlan(3)
    wigner = WignerTransformPlan(3, 2)
    plan = WignerLaguerrePlan(radial, wigner)
    valid = _wigner_valid(wigner)
    coefficients = (
        jr.normal(jr.key(8), (2, *plan.coefficient_shape, 2))
        + 1j * jr.normal(jr.key(9), (2, *plan.coefficient_shape, 2))
    ) * valid[None, None, ..., None]

    values = eqx.filter_jit(lambda transform, modes: transform.synthesis(modes))(
        plan, coefficients
    )
    actual = eqx.filter_jit(lambda transform, fields: transform.analysis(fields))(
        plan, values
    )
    gradient = jax.grad(lambda field: jnp.sum(jnp.abs(plan.analysis(field)) ** 2))(
        jnp.real(values[0, ..., 0])
    )

    assert actual.shape == coefficients.shape
    assert jnp.allclose(actual, coefficients, rtol=1e-10, atol=1e-10)
    assert jnp.all(jnp.isfinite(gradient))


def test_wigner_plans_reject_invalid_configuration_shapes_and_resources():
    with pytest.raises(ValueError, match="1 <= N <= L"):
        WignerTransformPlan(4, 5)
    with pytest.raises(ValueError, match="lower_bandlimit"):
        WignerTransformPlan(4, 2, lower_bandlimit=4)
    with pytest.raises(ValueError, match="certified"):
        WignerTransformPlan(8, 8, execution="recursive")
    with pytest.raises(ValueError, match="max_precompute_bytes"):
        WignerTransformPlan(4, 2, max_precompute_bytes=1)

    plan = WignerTransformPlan(4, 2)
    with pytest.raises(ValueError, match="Wigner analysis expects"):
        plan.analysis(jnp.ones((3, 4, 8)))
    with pytest.raises(ValueError, match="Wigner synthesis expects"):
        plan.synthesis(jnp.ones((3, 4, 8)))
