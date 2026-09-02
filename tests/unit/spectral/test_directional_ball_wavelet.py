#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from opt_einsum import contract

from phydrax.discretization import (
    BallWaveletCoefficients,
    DirectionalBallWaveletPlan,
    FourierLaguerrePlan,
    RadialLaguerrePlan,
    SphericalHarmonicPlan,
)


def _plan(
    *,
    bandlimit: int = 4,
    radial_bandlimit: int = 4,
    directional_bandlimit: int = 2,
    reality: bool = False,
    **wavelet_options,
) -> DirectionalBallWaveletPlan:
    radial = RadialLaguerrePlan(radial_bandlimit, tau=0.8)
    angular = SphericalHarmonicPlan(bandlimit, reality=reality)
    fourier = FourierLaguerrePlan(radial, angular)
    return DirectionalBallWaveletPlan(
        fourier,
        directional_bandlimit=directional_bandlimit,
        **wavelet_options,
    )


def _valid_spherical_modes(plan: DirectionalBallWaveletPlan) -> jax.Array:
    angular = plan.fourier_laguerre.angular
    degree = jnp.arange(angular.bandlimit)[:, None]
    order = jnp.arange(-(angular.bandlimit - 1), angular.bandlimit)[None, :]
    return jnp.abs(order) <= degree


def _complex_field(plan: DirectionalBallWaveletPlan) -> jax.Array:
    valid = _valid_spherical_modes(plan)
    shape = plan.fourier_laguerre.coefficient_shape
    modes = (jr.normal(jr.key(1), shape) + 1j * jr.normal(jr.key(2), shape)) * valid[
        None, ...
    ]
    return plan.fourier_laguerre.synthesis(modes)


def test_wavelet_filters_are_directional_admissible_and_hybrid():
    plan = _plan()
    angular_energy = jnp.sum(plan.angular_windows**2, axis=0)
    radial_energy = jnp.sum(plan.radial_windows**2, axis=0)
    detail_energy = radial_energy[:, None] * angular_energy[None, :]
    directionality_norm = jnp.sum(jnp.abs(plan.directionality[1:]) ** 2, axis=1)

    assert plan.scale_pairs == tuple(sorted(plan.scale_pairs))
    assert plan.scale_count == len(plan.detail_shapes)
    assert jnp.allclose(directionality_norm, 1.0, rtol=0.0, atol=1e-12)
    assert jnp.allclose(
        plan.scaling_window**2 + detail_energy,
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert jnp.allclose(plan.scaling_window[0, -1], 1.0)
    assert jnp.allclose(plan.scaling_window[-1, 0], 1.0)
    assert plan.admissibility_defect <= 1e-12


def test_multiresolution_detail_modes_match_full_coefficient_filtering():
    plan = _plan()
    values = _complex_field(plan)
    full_modes = plan.fourier_laguerre.analysis(values)
    coefficients = plan.analysis(values)

    for detail, scale in zip(coefficients.details, plan._scales, strict=True):
        actual = plan._detail_plan(scale).analysis(detail)
        m_stop = scale.full_m_start + 2 * scale.angular_bandlimit - 1
        n_stop = scale.full_n_start + 2 * scale.directional_bandlimit - 1
        subset = full_modes[
            : scale.radial_bandlimit,
            : scale.angular_bandlimit,
            scale.full_m_start : m_stop,
        ]
        radial_window = plan.radial_windows[scale.radial_window, : scale.radial_bandlimit]
        angular_window = plan.angular_windows[
            scale.angular_window, : scale.angular_bandlimit
        ]
        zeta = plan.directionality[
            : scale.angular_bandlimit,
            scale.full_n_start : n_stop,
        ]
        factor = jnp.sqrt(
            8.0 * jnp.pi**2 / (2.0 * jnp.arange(scale.angular_bandlimit) + 1.0)
        )
        expected = contract(
            "plm,p,l,ln->pnlm",
            subset,
            radial_window,
            angular_window,
            jnp.conj(zeta) * factor[:, None],
        )

        assert actual.shape == expected.shape
        assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_directional_ball_wavelet_roundtrips_complex_real_and_non_dyadic_fields():
    complex_plan = _plan()
    complex_values = _complex_field(complex_plan)
    complex_coefficients = complex_plan.analysis(complex_values)
    complex_reconstructed = complex_plan.synthesis(complex_coefficients)

    real_plan = _plan(reality=True)
    real_modes = (
        jnp.zeros(real_plan.fourier_laguerre.coefficient_shape, dtype=jnp.complex128)
        .at[:, 0, real_plan.fourier_laguerre.angular.bandlimit - 1]
        .set(jnp.arange(1, 5))
    )
    real_values = real_plan.fourier_laguerre.synthesis(real_modes)
    real_reconstructed = real_plan.synthesis(real_plan.analysis(real_values))

    non_dyadic = _plan(
        bandlimit=5,
        radial_bandlimit=5,
        angular_dilation=1.7,
        radial_dilation=2.3,
        angular_minimum_scale=1,
        radial_minimum_scale=1,
    )
    non_dyadic_values = _complex_field(non_dyadic)
    non_dyadic_reconstructed = non_dyadic.synthesis(
        non_dyadic.analysis(non_dyadic_values)
    )

    assert jnp.allclose(complex_reconstructed, complex_values, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(real_reconstructed, real_values, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(
        non_dyadic_reconstructed,
        non_dyadic_values,
        rtol=1e-10,
        atol=1e-10,
    )


def test_wavelet_coefficients_validate_every_ragged_leaf_and_transform_identity():
    plan = _plan()
    values = _complex_field(plan)
    coefficients = plan.analysis(values)
    zero_scaling = jnp.zeros_like(coefficients.scaling)

    for selected, scale in enumerate(plan._scales):
        detail_plan = plan._detail_plan(scale)
        radial_active = np.flatnonzero(
            np.asarray(plan.radial_windows[scale.radial_window, : scale.radial_bandlimit])
            > 0.0
        )
        angular_active = np.flatnonzero(
            np.asarray(
                plan.angular_windows[scale.angular_window, : scale.angular_bandlimit]
            )
            > 0.0
        )
        n_stop = scale.full_n_start + 2 * scale.directional_bandlimit - 1
        local_directionality = np.asarray(
            plan.directionality[
                : scale.angular_bandlimit,
                scale.full_n_start : n_stop,
            ]
        )
        degree = next(
            int(value)
            for value in angular_active
            if np.any(np.abs(local_directionality[value]) > 0.0)
        )
        n_index = int(np.flatnonzero(np.abs(local_directionality[degree]) > 0.0)[0])
        detail_modes = (
            jnp.zeros(detail_plan.coefficient_shape, dtype=jnp.complex128)
            .at[
                int(radial_active[0]),
                n_index,
                degree,
                scale.angular_bandlimit - 1,
            ]
            .set(1.0)
        )
        selected_detail = detail_plan.synthesis(detail_modes)
        details = tuple(
            selected_detail if index == selected else jnp.zeros_like(detail)
            for index, detail in enumerate(coefficients.details)
        )
        contribution = plan.synthesis(
            coefficients.with_coefficients(scaling=zero_scaling, details=details)
        )
        assert float(jnp.max(jnp.abs(contribution))) > 0.0

    missing = BallWaveletCoefficients(
        coefficients.scaling,
        coefficients.details[:-1],
        scale_pairs=coefficients.scale_pairs[:-1],
        transform_id=coefficients.transform_id,
    )
    with pytest.raises(ValueError, match="scale ordering"):
        plan.synthesis(missing)

    malformed_details = list(coefficients.details)
    malformed_details[0] = malformed_details[0][..., :-1]
    malformed = coefficients.with_coefficients(details=malformed_details)
    with pytest.raises(ValueError, match="invalid core transform shape"):
        plan.synthesis(malformed)

    foreign_plan = _plan(angular_dilation=2.5)
    with pytest.raises(ValueError, match="another transform"):
        foreign_plan.synthesis(coefficients)


def test_wavelet_handles_batch_channels_jit_gradients_and_resource_admission():
    plan = _plan(bandlimit=3, radial_bandlimit=3)
    first = _complex_field(plan)
    values = jnp.stack(
        (
            jnp.stack((first, -0.5 * first), axis=-1),
            jnp.stack((0.25 * first, 2.0 * first), axis=-1),
        )
    )
    coefficients = eqx.filter_jit(lambda transform, fields: transform.analysis(fields))(
        plan, values
    )
    reconstructed = eqx.filter_jit(lambda transform, modes: transform.synthesis(modes))(
        plan, coefficients
    )
    gradient = jax.grad(
        lambda field: (
            jnp.sum(jnp.abs(plan.analysis(field).scaling) ** 2)
            + sum(
                jnp.sum(jnp.abs(detail) ** 2) for detail in plan.analysis(field).details
            )
        )
    )(jnp.real(first))
    actual_bytes = int(coefficients.scaling.size * coefficients.scaling.dtype.itemsize)
    actual_bytes += sum(
        int(detail.size * detail.dtype.itemsize) for detail in coefficients.details
    )

    assert reconstructed.shape == values.shape
    assert jnp.allclose(reconstructed, values, rtol=1e-10, atol=1e-10)
    assert jnp.all(jnp.isfinite(gradient))
    assert plan.output_bytes(4) == actual_bytes
    assert plan.estimated_peak_bytes(4) >= actual_bytes

    constrained = _plan(
        bandlimit=3,
        radial_bandlimit=3,
        max_runtime_bytes=1,
    )
    with pytest.raises(ValueError, match="max_runtime_bytes"):
        constrained.analysis(_complex_field(constrained))
    with pytest.raises(ValueError, match="max_scale_pairs"):
        _plan(max_scale_pairs=1)
    with pytest.raises(ValueError, match="max_precompute_bytes"):
        _plan(max_precompute_bytes=1)
