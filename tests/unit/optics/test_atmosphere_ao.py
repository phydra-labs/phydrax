#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave._atmosphere import (
    AtmosphericLayer,
    frozen_flow_phase_screen,
    LayeredAtmosphere,
    sample_von_karman_phase_screen,
    VonKarmanPhaseScreenPlan,
)
from phydrax.optics.wave._fields import IntensityPlane, PlaneFieldSpace
from phydrax.optics.wave._imaging import normalized_otf_mtf
from phydrax.optics.wave._statistical_ao import (
    long_exposure_otf,
    StatisticalResidualAOPlan,
)


def _periodic_space(count=32, extent=8.0):
    grid = TensorGridPlan(
        (
            UniformAxisSpec(count, endpoint=False, periodic=True),
            UniformAxisSpec(count, endpoint=False, periodic=True),
        ),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-0.5 * extent, -0.5 * extent), (0.5 * extent, 0.5 * extent))))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "periodic-cell")


def _prepared_screen(count=32):
    return VonKarmanPhaseScreenPlan(
        _periodic_space(count),
        0.2,
        10.0,
        inner_scale=0.02,
    ).prepare()


def test_von_karman_preparation_has_nonnegative_decaying_psd_and_no_piston():
    prepared = _prepared_screen()
    psd = prepared.power_spectral_density

    assert bool(jnp.all(psd >= 0.0))
    assert float(psd[0, 0]) == 0.0
    assert float(psd[0, 1]) > float(psd[0, 2]) > 0.0
    assert bool(jnp.all(psd[~prepared.supported_modes] == 0.0))
    assert float(prepared.predicted_variance) > 0.0


def test_phase_screen_is_hermitian_reproducible_and_parseval_consistent():
    prepared = _prepared_screen()
    key = jax.random.key(17)
    first = sample_von_karman_phase_screen(prepared, key)
    second = sample_von_karman_phase_screen(prepared, key)

    np.testing.assert_array_equal(first.phase, second.phase)
    np.testing.assert_array_equal(
        first.spectral_coefficients,
        second.spectral_coefficients,
    )
    assert float(first.evidence.hermitian_error) < 1e-5
    assert float(first.evidence.parseval_relative_error) < 1e-5
    assert abs(float(first.evidence.piston)) < 1e-5
    assert bool(first.valid)


def test_phase_screen_ensemble_covariance_matches_prepared_spectrum():
    prepared = _prepared_screen(count=24)
    keys = jax.random.split(jax.random.key(29), 96)
    samples = jax.vmap(lambda key: sample_von_karman_phase_screen(prepared, key).phase)(
        keys
    )
    empirical_zero = jnp.mean(samples * samples)
    empirical_lag = jnp.mean(samples * jnp.roll(samples, -1, axis=1))
    count = prepared.plan.space.size
    domain_area = prepared.lengths[0] * prepared.lengths[1]
    covariance = jnp.fft.ifft2(prepared.power_spectral_density).real * count / domain_area

    np.testing.assert_allclose(empirical_zero, covariance[0, 0], rtol=0.25)
    np.testing.assert_allclose(empirical_lag, covariance[1, 0], rtol=0.3)


def test_exact_spectral_frozen_flow_matches_one_cell_periodic_shift():
    prepared = _prepared_screen()
    original = sample_von_karman_phase_screen(prepared, jax.random.key(3))
    velocity = jnp.asarray((prepared.spacings[0], 0.0))
    translated = frozen_flow_phase_screen(original, 1.0, velocity)

    np.testing.assert_allclose(
        translated.phase,
        jnp.roll(original.phase, 1, axis=0),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        translated.evidence.realized_variance,
        original.evidence.realized_variance,
        rtol=2e-5,
    )
    assert bool(translated.valid)


def _layered_atmosphere():
    screen = VonKarmanPhaseScreenPlan(_periodic_space(), 0.18, 20.0)
    return LayeredAtmosphere(
        (
            AtmosphericLayer(
                screen,
                0.0,
                (3.0, 0.0),
                0.6,
                layer_id="ground",
            ),
            AtmosphericLayer(
                screen,
                8000.0,
                (-1.0, 4.0),
                0.4,
                layer_id="high",
            ),
        )
    ).prepare()


def test_layered_atmosphere_sampling_preserves_records_and_is_reproducible():
    atmosphere = _layered_atmosphere()
    first = atmosphere.sample(jax.random.key(41))
    repeated = atmosphere.sample(jax.random.key(41))
    stationary = first.advect(0.0)

    assert tuple(layer.layer.layer.layer_id for layer in first.layers) == (
        "ground",
        "high",
    )
    np.testing.assert_array_equal(first.phase, repeated.phase)
    np.testing.assert_allclose(stationary.phase, first.phase, atol=1e-6)
    for realized in first.layers:
        expected = (
            realized.layer.layer.strength_fraction
            * realized.layer.screen.predicted_variance
        )
        np.testing.assert_allclose(
            realized.screen.evidence.predicted_variance,
            expected,
        )
    assert bool(first.valid)


def test_statistical_ao_residual_psd_and_error_budget_close_exactly():
    atmosphere = _layered_atmosphere()
    residual = StatisticalResidualAOPlan(
        0.6,
        correction_gain=0.8,
        loop_delay=0.01,
        measurement_phase_variance=0.02,
        aliasing_phase_variance=0.01,
    ).prepare(atmosphere)
    budget = residual.error_budget
    component_sum = (
        budget.fitting_variance
        + budget.servo_lag_variance
        + budget.measurement_variance
        + budget.aliasing_variance
    )

    np.testing.assert_allclose(budget.total_residual_variance, component_sum, rtol=2e-6)
    np.testing.assert_allclose(budget.measurement_variance, 0.02, rtol=2e-6)
    np.testing.assert_allclose(budget.aliasing_variance, 0.01, rtol=2e-6)
    np.testing.assert_allclose(
        residual.total_residual_psd,
        residual.fitting_psd
        + residual.servo_lag_psd
        + residual.measurement_noise_psd
        + residual.aliasing_psd,
        rtol=2e-6,
    )
    assert bool(jnp.all(residual.total_residual_psd >= 0.0))
    assert bool(residual.valid)


def test_perfect_statistical_correction_has_zero_residual_psd():
    atmosphere = _layered_atmosphere()
    residual = StatisticalResidualAOPlan(
        100.0,
        correction_gain=1.0,
    ).prepare(atmosphere)

    np.testing.assert_allclose(residual.total_residual_psd, 0.0, atol=1e-8)
    np.testing.assert_allclose(residual.error_budget.total_residual_variance, 0.0)
    np.testing.assert_allclose(residual.error_budget.marechal_strehl, 1.0)
    assert float(residual.error_budget.atmospheric_variance) > 0.0


def test_long_exposure_otf_is_normalized_and_reduced_away_from_origin():
    atmosphere = _layered_atmosphere()
    residual = StatisticalResidualAOPlan(
        0.35,
        correction_gain=0.7,
        loop_delay=0.02,
    ).prepare(atmosphere)
    space = atmosphere.layers[0].screen.plan.space
    coordinates = space.transverse_coordinates
    centered = coordinates - jnp.mean(coordinates, axis=(0, 1), keepdims=True)
    intensity = IntensityPlane(
        space,
        jnp.exp(-0.2 * jnp.sum(centered * centered, axis=-1)),
        1.0,
        0.0,
    )
    diffraction_limited = normalized_otf_mtf(intensity)
    result = long_exposure_otf(diffraction_limited, residual, 2.0)
    misaligned = long_exposure_otf(diffraction_limited, residual, 1.0)
    center = (space.shape[0] // 2, space.shape[1] // 2)

    np.testing.assert_allclose(result.optical_transfer_function[center], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        result.atmospheric_transfer_function[center],
        1.0,
        atol=1e-6,
    )
    assert bool(result.sampling.aligned)
    assert not bool(misaligned.valid)
    assert not bool(misaligned.sampling.aligned)
    np.testing.assert_allclose(misaligned.optical_transfer_function, 0.0)
    assert float(result.atmospheric_transfer_function[0, 0]) < 1.0
    assert bool(jnp.all(result.residual_structure_function >= 0.0))
    assert bool(result.valid)
