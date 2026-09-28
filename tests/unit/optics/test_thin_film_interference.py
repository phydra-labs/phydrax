#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.optics.geometric import evaluate_refractive_interface
from phydrax.optics.wave import ThinFilmInterferencePlan, ThinFilmInterferenceStatus


VISIBLE = np.arange(380.0, 781.0, 5.0) / 1.0e9
SOAP = 1.33


def _interface_coefficients(
    incident_index: float, transmitted_index: float, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    single = evaluate_refractive_interface(
        jnp.asarray([np.sin(angle), 0.0, np.cos(angle)]),
        jnp.asarray([0.0, 0.0, 1.0]),
        incident_index,
        transmitted_index,
    )
    return np.asarray(single.reflectance), np.asarray(single.transmittance)


def test_vanishing_and_index_matched_films_reduce_to_one_fresnel_interface() -> None:
    bare = ThinFilmInterferencePlan(VISIBLE, 1.0, 1.5, SOAP).evaluate(0.0, 1.0)
    expected = ((SOAP - 1.0) / (SOAP + 1.0)) ** 2
    np.testing.assert_allclose(bare.reflectance, expected, rtol=1.0e-13)
    np.testing.assert_allclose(bare.transmittance, 1.0 - expected, rtol=1.0e-13)

    angle = np.deg2rad(50.0)
    matched = ThinFilmInterferencePlan(VISIBLE, 1.0, 1.0, 1.52).evaluate(
        400.0e-9, np.cos(angle)
    )
    reflectance, transmittance = _interface_coefficients(1.0, 1.52, angle)
    np.testing.assert_allclose(
        matched.reflectance, np.broadcast_to(reflectance, (VISIBLE.size, 2)), rtol=1e-12
    )
    np.testing.assert_allclose(
        matched.transmittance,
        np.broadcast_to(transmittance, (VISIBLE.size, 2)),
        rtol=1e-12,
    )
    assert int(matched.status) == ThinFilmInterferenceStatus.SUCCESS


@pytest.mark.parametrize(
    ("ambient", "substrate"),
    [(1.0, 1.52), (1.0, 0.05 + 3.3j), (1.0, 4.0 + 0.03j), (1.5, 1.0)],
)
def test_lossless_film_conserves_flux_for_every_passive_substrate(
    ambient: float, substrate: complex
) -> None:
    plan = ThinFilmInterferencePlan(VISIBLE, ambient, SOAP, substrate)
    thickness = jnp.linspace(0.0, 2.0e-6, 9)[:, None]
    cosine = jnp.cos(jnp.deg2rad(jnp.linspace(0.0, 85.0, 7)))[None, :]
    result = plan.evaluate(thickness, cosine)

    assert float(jnp.max(result.energy_residual)) <= 1.0e-12
    assert bool(jnp.all(result.evidence.energy_conserved))
    assert bool(jnp.all(result.reflectance >= 0.0))
    assert bool(jnp.all(result.transmittance >= 0.0))


def test_total_internal_reflection_at_the_substrate_transmits_no_flux() -> None:
    plan = ThinFilmInterferencePlan(VISIBLE, 1.5, SOAP, 1.0)
    result = plan.evaluate(jnp.asarray([0.0, 1.0e-7, 1.0e-6]), np.cos(np.deg2rad(70.0)))

    np.testing.assert_allclose(result.reflectance, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(result.transmittance, 0.0, atol=1.0e-15)
    assert bool(jnp.all(result.evidence.substrate_evanescent))
    assert bool(jnp.all(result.evidence.film_evanescent))
    assert bool(jnp.all(result.accepted))


def test_symmetric_film_extrema_follow_the_optical_path_condition() -> None:
    angle = np.deg2rad(35.0)
    wavelength = 550.0e-9
    film_cosine = np.sqrt(1.0 - (np.sin(angle) / SOAP) ** 2)
    order = np.asarray([1.0, 2.0, 3.0])
    optical_unit = wavelength / (2.0 * SOAP * film_cosine)
    plan = ThinFilmInterferencePlan(np.asarray([wavelength]), 1.0, SOAP, 1.0)

    destructive = plan.evaluate(order * optical_unit, np.cos(angle))
    constructive = plan.evaluate((order + 0.5) * optical_unit, np.cos(angle))

    single, _ = _interface_coefficients(1.0, SOAP, angle)
    peak = 4.0 * single / (1.0 + single) ** 2
    assert float(jnp.max(destructive.reflectance)) <= 1.0e-24
    np.testing.assert_allclose(
        constructive.reflectance, np.broadcast_to(peak, (3, 1, 2)), rtol=1.0e-11
    )


def test_fringe_sampling_evidence_rejects_undersampled_spectra() -> None:
    thickness = 2.0e-6
    coarse_step = 20.0e-9
    coarse = np.arange(400.0, 701.0, 20.0) / 1.0e9
    fine = np.arange(400.0, 701.0, 2.0) / 1.0e9

    undersampled = ThinFilmInterferencePlan(coarse, 1.0, SOAP, 1.0).evaluate(
        thickness, 1.0
    )
    samples = float(undersampled.evidence.minimum_samples_per_fringe)
    exact = coarse[0] * coarse[1] / (2.0 * SOAP * thickness * coarse_step)
    fringe_period = coarse[0] ** 2 / (2.0 * SOAP * thickness)
    assert samples == pytest.approx(exact, rel=1.0e-12)
    assert samples == pytest.approx(fringe_period / coarse_step, rel=0.06)
    assert samples < 4.0
    assert int(undersampled.status) == ThinFilmInterferenceStatus.SPECTRAL_UNDERSAMPLED
    assert not bool(undersampled.accepted)
    assert bool(jnp.all(jnp.isnan(undersampled.unpolarized_reflectance)))
    assert bool(jnp.all(jnp.isnan(undersampled.energy_residual)))
    assert bool(jnp.all(jnp.isnan(undersampled.reflection_amplitudes)))
    assert bool(jnp.all(jnp.isnan(undersampled.transmittance)))

    resolved = ThinFilmInterferencePlan(fine, 1.0, SOAP, 1.0).evaluate(thickness, 1.0)
    assert float(resolved.evidence.minimum_samples_per_fringe) >= 4.0
    assert int(resolved.status) == ThinFilmInterferenceStatus.SUCCESS

    lines = ThinFilmInterferencePlan(
        coarse, 1.0, SOAP, 1.0, required_samples_per_fringe=None
    ).evaluate(thickness, 1.0)
    assert int(lines.status) == ThinFilmInterferenceStatus.SUCCESS
    assert float(lines.evidence.minimum_samples_per_fringe) == pytest.approx(samples)


def test_energy_residual_rejection_masks_spectra_but_retains_evidence() -> None:
    tolerance = 1.0e-30
    result = ThinFilmInterferencePlan(
        VISIBLE,
        1.0,
        SOAP,
        1.52,
        required_samples_per_fringe=None,
        energy_tolerance=tolerance,
    ).evaluate(300.0e-9, 0.73)

    assert int(result.status) == ThinFilmInterferenceStatus.ENERGY_RESIDUAL_EXCEEDED
    assert float(result.evidence.maximum_energy_residual) > tolerance
    assert bool(jnp.isfinite(result.evidence.maximum_energy_residual))
    assert bool(jnp.all(jnp.isnan(result.unpolarized_reflectance)))
    assert bool(jnp.all(jnp.isnan(result.energy_residual)))
    assert bool(jnp.all(jnp.isnan(result.reflection_amplitudes)))
    assert bool(jnp.all(jnp.isnan(result.transmittance)))


def test_reflectance_derivatives_match_central_differences() -> None:
    plan = ThinFilmInterferencePlan(VISIBLE, 1.0, SOAP, 1.52)
    index = 40

    def by_thickness(thickness: jax.Array) -> jax.Array:
        return plan.evaluate(thickness, 0.8).unpolarized_reflectance[index]

    def by_film_index(film_index: jax.Array) -> jax.Array:
        updated = eqx.tree_at(lambda value: value.film_index, plan, film_index)
        return updated.evaluate(310.0e-9, 0.8).unpolarized_reflectance[index]

    thickness = jnp.asarray(310.0e-9)
    step = 1.0e-12
    difference = (by_thickness(thickness + step) - by_thickness(thickness - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(jax.grad(by_thickness)(thickness), difference, rtol=1e-6)

    film_index = plan.film_index
    direction = jnp.ones_like(film_index)
    epsilon = 1.0e-7
    central = (
        by_film_index(film_index + epsilon * direction)
        - by_film_index(film_index - epsilon * direction)
    ) / (2.0 * epsilon)
    tangent = jax.jvp(by_film_index, (film_index,), (direction,))[1]
    np.testing.assert_allclose(tangent, central, rtol=1.0e-6)


def test_invalid_samples_fail_closed_with_status_bits() -> None:
    plan = ThinFilmInterferencePlan(VISIBLE, 1.0, SOAP, 1.0)
    result = plan.evaluate(
        jnp.asarray([-1.0e-9, jnp.nan, 1.0e-7, 1.0e-7]),
        jnp.asarray([0.5, 0.5, 1.5, 0.0]),
    )

    np.testing.assert_array_equal(
        result.status,
        [
            ThinFilmInterferenceStatus.INVALID_THICKNESS,
            ThinFilmInterferenceStatus.INVALID_THICKNESS,
            ThinFilmInterferenceStatus.INVALID_INCIDENCE,
            ThinFilmInterferenceStatus.INVALID_INCIDENCE,
        ],
    )
    assert not bool(jnp.any(result.accepted))
    assert bool(jnp.all(jnp.isnan(result.reflectance)))


def test_plan_refuses_gain_media_complex_films_and_unordered_wavelengths() -> None:
    with pytest.raises(ValueError, match="passive"):
        ThinFilmInterferencePlan(VISIBLE, 1.0, SOAP, 1.5 - 0.1j)
    with pytest.raises(TypeError, match="film_index"):
        ThinFilmInterferencePlan(VISIBLE, 1.0, SOAP + 0.01j, 1.0)
    with pytest.raises(ValueError, match="increasing"):
        ThinFilmInterferencePlan(VISIBLE[::-1], 1.0, SOAP, 1.0)
    with pytest.raises(ValueError, match="one value per wavelength"):
        ThinFilmInterferencePlan(VISIBLE, 1.0, np.ones(3), 1.0)
