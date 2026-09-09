#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import s2fft
from s2fft.transforms import spherical as s2fft_spherical
from scipy.special import sph_harm_y

from phydrax.discretization import SphericalHarmonicPlan


def _real_bandlimited_field(plan):
    theta, phi = np.meshgrid(
        np.asarray(plan.theta),
        np.asarray(plan.phi),
        indexing="ij",
    )
    return jnp.asarray(
        np.real(sph_harm_y(2, 1, theta, phi))
        + 0.3 * np.real(sph_harm_y(3, -2, theta, phi))
        - 0.2 * np.real(sph_harm_y(1, 0, theta, phi))
    )


@pytest.mark.parametrize("sampling", ("mw", "mwss", "dh", "gl"))
def test_spherical_plan_roundtrips_sampling_theorems_and_integrates_constants(sampling):
    plan = SphericalHarmonicPlan(4, sampling=sampling)
    values = _real_bandlimited_field(plan)

    coefficients = plan.analysis(values)
    reconstructed = plan.synthesis(coefficients)
    sphere_measure = jnp.sum(plan.theta_quadrature_weights) * jnp.sum(
        plan.phi_quadrature_weights
    )

    assert values.shape == plan.sample_shape
    assert coefficients.shape == plan.coefficient_shape
    assert jnp.allclose(reconstructed, values, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(sphere_measure, 4.0 * jnp.pi, rtol=1e-12, atol=1e-12)


def test_spherical_plan_roundtrips_complex_spin_coefficients():
    plan = SphericalHarmonicPlan(5, spin=1, reality=False)
    degree = jnp.arange(plan.bandlimit)[:, None]
    order = jnp.arange(-(plan.bandlimit - 1), plan.bandlimit)[None, :]
    valid = (jnp.abs(order) <= degree) & (degree >= abs(plan.spin))
    coefficients = (
        jr.normal(jr.key(1), plan.coefficient_shape)
        + 1j * jr.normal(jr.key(2), plan.coefficient_shape)
    ) * valid

    actual = plan.analysis(plan.synthesis(coefficients))

    assert jnp.allclose(actual, coefficients, rtol=1e-11, atol=1e-11)


def test_spherical_plan_matches_s2fft_and_handles_batch_channel_axes():
    plan = SphericalHarmonicPlan(4, sampling="mw")
    first = _real_bandlimited_field(plan)
    values = jnp.stack(
        (
            jnp.stack((first, -0.5 * first), axis=-1),
            jnp.stack((0.25 * first, 2.0 * first), axis=-1),
        )
    )
    forward_precomputes = tuple(s2fft.generate_precomputes_jax(4, 0, "mw", None, True))
    inverse_precomputes = tuple(s2fft.generate_precomputes_jax(4, 0, "mw", None, False))
    expected_first = s2fft_spherical.forward_jax(
        first,
        4,
        0,
        None,
        "mw",
        True,
        forward_precomputes,
    )

    actual = eqx.filter_jit(lambda transform, field: transform.analysis(field))(
        plan, values
    )
    reconstructed = eqx.filter_jit(
        lambda transform, coefficients: transform.synthesis(coefficients)
    )(plan, actual)
    expected_reconstruction = s2fft_spherical.inverse_jax(
        expected_first,
        4,
        0,
        None,
        "mw",
        True,
        inverse_precomputes,
    )
    gradient = jax.grad(lambda field: jnp.sum(jnp.abs(plan.analysis(field)) ** 2))(first)

    assert actual.shape == (2, *plan.coefficient_shape, 2)
    assert jnp.allclose(actual[0, ..., 0], expected_first, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(reconstructed, values, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(expected_reconstruction, first, rtol=1e-11, atol=1e-11)
    assert jnp.all(jnp.isfinite(gradient))


def test_recursive_and_precomputed_spherical_plans_share_semantic_identity():
    recursive = SphericalHarmonicPlan(4, execution="recursive")
    precomputed = SphericalHarmonicPlan(4, execution="precomputed")
    values = _real_bandlimited_field(recursive)

    expected = recursive.analysis(values)
    actual = precomputed.analysis(values)

    assert recursive.fingerprint == precomputed.fingerprint
    assert recursive.layout_id == precomputed.layout_id
    assert recursive.transform_id == precomputed.transform_id
    assert recursive.execution_id != precomputed.execution_id
    assert recursive.precompute_bytes > 0
    assert precomputed.precompute_bytes > 0
    assert jnp.allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(
        precomputed.synthesis(actual),
        recursive.synthesis(expected),
        rtol=1e-11,
        atol=1e-11,
    )


def test_spherical_plan_rejects_invalid_configuration_shapes_and_memory():
    with pytest.raises(ValueError, match="exceed the absolute spin"):
        SphericalHarmonicPlan(2, spin=2, reality=False)
    with pytest.raises(ValueError, match="spin-zero"):
        SphericalHarmonicPlan(4, spin=1, reality=True)
    with pytest.raises(ValueError, match="sampling"):
        SphericalHarmonicPlan(4, sampling="healpix")
    with pytest.raises(ValueError, match="execution"):
        SphericalHarmonicPlan(4, execution="dense")
    with pytest.raises(ValueError, match="max_precompute_bytes"):
        SphericalHarmonicPlan(4, max_precompute_bytes=1)

    plan = SphericalHarmonicPlan(4)
    with pytest.raises(ValueError, match="Spherical analysis expects"):
        plan.analysis(jnp.ones((4, 8)))
    with pytest.raises(ValueError, match="Spherical synthesis expects"):
        plan.synthesis(jnp.ones((4, 8)))
    with pytest.raises(TypeError, match="requires real values"):
        plan.analysis(jnp.ones(plan.sample_shape, dtype=complex))


@pytest.mark.parametrize("execution", ("recursive", "precomputed"))
@pytest.mark.parametrize(("spin", "reality"), ((0, True), (1, False)))
def test_fixed_spherical_transform_jvp_and_real_linear_adjoint(execution, spin, reality):
    plan = SphericalHarmonicPlan(
        4, sampling="mwss", execution=execution, spin=spin, reality=reality
    )
    shape = (2, *plan.sample_shape, 2)
    values = jr.normal(jr.key(31), shape)
    direction = jr.normal(jr.key(32), shape)
    if not reality:
        values = values + 1j * jr.normal(jr.key(33), shape)
        direction = direction + 1j * jr.normal(jr.key(34), shape)
    coefficients = plan.analysis(values)
    coefficient_direction = plan.analysis(direction)

    for transform, primal, tangent in (
        (plan.analysis, values, direction),
        (plan.synthesis, coefficients, coefficient_direction),
    ):
        output, actual = eqx.filter_jit(
            lambda value, delta: jax.jvp(transform, (value,), (delta,))
        )(primal, tangent)
        np.testing.assert_allclose(actual, transform(tangent), atol=3e-11, rtol=3e-11)
        step = 1e-4
        difference = (
            transform(primal + step * tangent) - transform(primal - step * tangent)
        ) / (2 * step)
        np.testing.assert_allclose(actual, difference, atol=3e-9, rtol=3e-9)
        cotangent = jr.normal(jr.key(35), output.shape)
        if jnp.iscomplexobj(output):
            cotangent = cotangent + 1j * jr.normal(jr.key(36), output.shape)
        _, pullback = jax.vjp(transform, primal)
        # JAX complex cotangents use the real part of the bilinear pairing,
        # not a Hermitian pairing. This also exercises real/complex interfaces.
        np.testing.assert_allclose(
            jnp.real(jnp.sum(actual * cotangent)),
            jnp.real(jnp.sum(tangent * pullback(cotangent)[0])),
            atol=3e-10,
            rtol=3e-10,
        )


def test_recursive_transform_forward_over_reverse_quadratic_derivative():
    plan = SphericalHarmonicPlan(4, sampling="mwss", spin=1, reality=False)
    coefficients = jr.normal(jr.key(41), plan.coefficient_shape) + 1j * jr.normal(
        jr.key(42), plan.coefficient_shape
    )
    tangent = jr.normal(jr.key(43), plan.coefficient_shape) + 1j * jr.normal(
        jr.key(44), plan.coefficient_shape
    )
    gradient = jax.grad(lambda modal: jnp.sum(jnp.abs(plan.synthesis(modal)) ** 2))
    _, action = eqx.filter_jit(
        lambda modal, delta: jax.jvp(gradient, (modal,), (delta,))
    )(coefficients, tangent)
    np.testing.assert_allclose(action, gradient(tangent), atol=3e-10, rtol=3e-10)


def test_recursive_numeric_preparation_derivatives_are_explicitly_rejected():
    plan = SphericalHarmonicPlan(4)
    values = _real_bandlimited_field(plan)
    table = plan.transform.forward_precomputes[0]

    def alter_table(replacement):
        changed = eqx.tree_at(
            lambda prepared: prepared.transform.forward_precomputes[0],
            plan,
            replacement,
        )
        return jnp.sum(jnp.abs(changed.analysis(values)) ** 2)

    with pytest.raises(ValueError, match="fixed preparation state"):
        jax.jvp(alter_table, (table,), (jnp.ones_like(table),))
    with pytest.raises(ValueError, match="fixed preparation state"):
        jax.grad(alter_table)(table)
