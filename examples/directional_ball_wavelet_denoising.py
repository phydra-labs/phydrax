#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic directional ball-wavelet denoising on a bandlimited field."""

import json

import jax.numpy as jnp
import jax.random as jr

from phydrax.discretization import (
    BallWaveletCoefficients,
    DirectionalBallWaveletPlan,
    FourierLaguerrePlan,
    RadialLaguerrePlan,
    SphericalHarmonicPlan,
)


def _weighted_error(values, reference, radial, angular):
    weights = (
        radial.quadrature_weights[:, None, None]
        * angular.theta_quadrature_weights[None, :, None]
        * angular.phi_quadrature_weights[None, None, :]
    )
    return jnp.sum(weights * jnp.abs(values - reference) ** 2)


def main() -> None:
    radial = RadialLaguerrePlan(4, tau=0.8)
    angular = SphericalHarmonicPlan(4, reality=False)
    fourier_laguerre = FourierLaguerrePlan(radial, angular)
    wavelets = DirectionalBallWaveletPlan(
        fourier_laguerre,
        directional_bandlimit=2,
    )

    scaling = jnp.zeros(fourier_laguerre.sample_shape, dtype=jnp.complex128)
    details = [jnp.zeros(shape, dtype=jnp.complex128) for shape in wavelets.detail_shapes]
    selected = len(details) // 2
    details[selected] = details[selected].at[1, 1, 1, 2].set(1.0)
    clean_coefficients = BallWaveletCoefficients(
        scaling,
        details,
        scale_pairs=wavelets.scale_pairs,
        transform_id=wavelets.transform_id,
    )
    clean = wavelets.synthesis(clean_coefficients)

    degree = jnp.arange(angular.bandlimit)[:, None]
    order = jnp.arange(-(angular.bandlimit - 1), angular.bandlimit)[None, :]
    valid = jnp.abs(order) <= degree
    modal_noise = (
        0.03
        * (
            jr.normal(jr.key(50), fourier_laguerre.coefficient_shape)
            + 1j * jr.normal(jr.key(51), fourier_laguerre.coefficient_shape)
        )
        * valid[None, ...]
    )
    noisy = clean + fourier_laguerre.synthesis(modal_noise)

    analyzed = wavelets.analysis(noisy)
    threshold = 3.0e-4
    thresholded = tuple(
        jnp.where(jnp.abs(detail) >= threshold, detail, 0.0)
        for detail in analyzed.details
    )
    denoised = wavelets.synthesis(analyzed.with_coefficients(details=thresholded))
    no_threshold = wavelets.synthesis(analyzed)

    roundtrip_error = float(jnp.max(jnp.abs(no_threshold - noisy)))
    noisy_error = float(_weighted_error(noisy, clean, radial, angular))
    denoised_error = float(_weighted_error(denoised, clean, radial, angular))
    if roundtrip_error > 1.0e-10:
        raise RuntimeError("directional ball-wavelet round trip exceeded tolerance.")
    if denoised_error >= noisy_error:
        raise RuntimeError("the deterministic threshold did not improve weighted error.")
    print(
        json.dumps(
            {
                "roundtrip_error": roundtrip_error,
                "noisy_weighted_error": noisy_error,
                "denoised_weighted_error": denoised_error,
                "threshold": threshold,
                "scale_count": wavelets.scale_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
