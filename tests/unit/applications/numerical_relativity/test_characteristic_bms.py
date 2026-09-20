#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax.applications.numerical_relativity._bms import (
    BMSFrameTransformation,
    BMSQuadraturePlan,
    BMSScriData,
)
from phydrax.applications.numerical_relativity._characteristic import (
    CharacteristicEvolutionPlan,
    CharacteristicWorldtubeHistory,
)


def _octahedral_quadrature():
    directions = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    weights = np.full((6,), 4.0 * np.pi / 6.0)
    basis = np.concatenate((np.ones((1, 6)), directions.T), axis=0)
    gram = np.diag((4.0 * np.pi,) + (4.0 * np.pi / 3.0,) * 3)
    return BMSQuadraturePlan(
        directions,
        weights,
        basis,
        gram,
        quadrature_tolerance=2.0e-6,
        supported_bandlimit=1,
    )


def test_characteristic_minkowski_and_analytic_outgoing_wave_controls():
    nine = 9
    times = np.linspace(-1.0, 1.0, nine)
    inverse_radius = np.asarray((0.25, 0.125, 0.0))
    mode_l = np.asarray((2,))
    mode_m = np.asarray((2,))
    plan = CharacteristicEvolutionPlan(nine, 3, 1)

    minkowski = CharacteristicWorldtubeHistory(
        times,
        inverse_radius,
        mode_l,
        mode_m,
        np.zeros((nine, 1), dtype="complex128"),
        np.zeros((nine, 3, 1), dtype="complex128"),
        history_name="minkowski-worldtube",
    )
    flat = plan.evolve(minkowski)
    np.testing.assert_allclose(flat.radial_shear_coefficient, 0.0)
    np.testing.assert_allclose(flat.waveform.news_modes, 0.0)
    np.testing.assert_allclose(flat.waveform.radiated_energy, 0.0)
    assert bool(flat.qualified)
    assert flat.waveform.strain_modes.shape == (nine, 1)

    frequency = 0.7
    amplitude = 0.03 - 0.01j
    radial_slope = -0.02 + 0.04j
    phase = np.exp(-1j * frequency * times)[:, None]
    expected_scri = amplitude * phase
    boundary = expected_scri + inverse_radius[0] * radial_slope * phase
    source = np.broadcast_to(
        radial_slope * phase[:, None, :], (nine, inverse_radius.size, 1)
    )
    wave_history = CharacteristicWorldtubeHistory(
        times,
        inverse_radius,
        mode_l,
        mode_m,
        boundary,
        source,
        history_name="analytic-linearized-bondi-wave",
    )
    wave = plan.evolve(wave_history)
    np.testing.assert_allclose(
        wave.waveform.strain_modes, expected_scri, rtol=2.0e-6, atol=2.0e-7
    )
    expected_news = -1j * frequency * expected_scri
    np.testing.assert_allclose(
        wave.waveform.news_modes[1:-1],
        expected_news[1:-1],
        rtol=6.0e-3,
        atol=2.0e-6,
    )
    expected_psi4 = -(frequency**2) * expected_scri
    np.testing.assert_allclose(
        wave.waveform.psi4_modes[2:-2],
        expected_psi4[2:-2],
        rtol=1.3e-2,
        atol=3.0e-6,
    )
    assert float(wave.waveform.maximum_hypersurface_residual) < 1.0e-7
    assert bool(wave.qualified)


def test_bms_quadrature_rejects_asymmetric_l_one_moments():
    directions = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    weights = np.asarray((np.pi, np.pi / 3.0) + (2.0 * np.pi / 3.0,) * 4)
    basis = np.concatenate((np.ones((1, 6)), directions.T), axis=0)
    declared_gram = np.diag((4.0 * np.pi,) + (4.0 * np.pi / 3.0,) * 3)
    with pytest.raises(ValueError, match="l=0,1"):
        BMSQuadraturePlan(
            directions,
            weights,
            basis,
            declared_gram,
            supported_bandlimit=1,
        )


def test_bms_boost_and_translation_controls_preserve_poincare_laws():
    plan = _octahedral_quadrature()
    times = np.asarray((0.0, 1.0))
    mass = 2.0
    data = BMSScriData(
        times,
        np.full((2, plan.direction_capacity), mass),
        np.zeros((2, plan.direction_capacity), dtype="complex128"),
        np.zeros((2, plan.direction_capacity, 4, 4)),
        data_name="rest-bondi-data",
    )
    charges = plan.charges(data)
    np.testing.assert_allclose(charges.four_momentum[:, 0], mass, rtol=2.0e-6)
    np.testing.assert_allclose(charges.four_momentum[:, 1:], 0.0, atol=2.0e-6)
    assert bool(charges.qualified)

    translation = np.asarray((1.5, 0.25, -0.5, 0.75))
    translated_frame = BMSFrameTransformation(np.zeros(3), translation)
    mapped = translated_frame.map_null_infinity(plan, times)
    expected_cut = translation[0] - plan.directions @ translation[1:]
    np.testing.assert_allclose(mapped.directions, plan.directions, atol=2.0e-6)
    np.testing.assert_allclose(mapped.retarded_times, times[:, None] - expected_cut)
    translated = translated_frame.transform_charges(charges)
    np.testing.assert_allclose(translated.four_momentum, charges.four_momentum)
    expected_lorentz = np.zeros((2, 4, 4))
    expected_lorentz[:, 0, 1:] = translation[1:] * mass
    expected_lorentz[:, 1:, 0] = -translation[1:] * mass
    np.testing.assert_allclose(translated.lorentz_charges, expected_lorentz, atol=2.0e-6)

    speed = 0.3
    boost = BMSFrameTransformation(np.asarray((speed, 0.0, 0.0)), np.zeros(4))
    boosted = boost.transform_charges(charges)
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    np.testing.assert_allclose(boosted.four_momentum[:, 0], gamma * mass, rtol=2.0e-6)
    np.testing.assert_allclose(
        boosted.four_momentum[:, 1], -gamma * speed * mass, rtol=2.0e-6
    )
    np.testing.assert_allclose(boosted.four_momentum[:, 2:], 0.0, atol=2.0e-6)
    assert bool(boosted.qualified)
