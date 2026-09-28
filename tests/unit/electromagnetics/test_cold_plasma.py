#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import ElectromagneticScaleContract
from phydrax.electromagnetics import (
    ColdPlasmaDielectric,
    ColdPlasmaWaveStatus,
    PlasmaWaveMode,
)


SCALE = ElectromagneticScaleContract.si()
E = float(SCALE.elementary_charge)
M_E = float(SCALE.electron_mass)
EPS0 = float(SCALE.vacuum_permittivity)
C = float(SCALE.speed_of_light)
PROTON_RATIO = 1836.152673426

pytestmark = pytest.mark.strict_jax


def _plasma_frequency(
    density: float, charge_number: float = 1.0, ratio: float = 1.0
) -> float:
    return np.sqrt(density * (charge_number * E) ** 2 / (EPS0 * ratio * M_E))


def _cyclotron_frequency(field: float, charge_number: float, ratio: float) -> float:
    return charge_number * E * field / (ratio * M_E)


def _electron_plasma(
    density: float, field: float, *, collision: float = 0.0
) -> ColdPlasmaDielectric:
    return ColdPlasmaDielectric(
        SCALE,
        densities=[density],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, field],
        collision_frequencies=[collision],
    )


def _reference_dielectric(
    omega: float,
    densities: np.ndarray,
    charges: np.ndarray,
    ratios: np.ndarray,
    field: float,
    collisions: np.ndarray,
) -> np.ndarray:
    """Stix tensor built independently, species by species."""
    right = 1.0 + 0.0j
    left = 1.0 + 0.0j
    plasma = 1.0 + 0.0j
    for density, charge, ratio, nu in zip(
        densities, charges, ratios, collisions, strict=True
    ):
        wp2 = density * (charge * E) ** 2 / (EPS0 * ratio * M_E)
        gyro = charge * E * field / (ratio * M_E)
        shifted = omega + 1j * nu
        right -= wp2 / (omega * (shifted + gyro))
        left -= wp2 / (omega * (shifted - gyro))
        plasma -= wp2 / (omega * shifted)
    s = 0.5 * (right + left)
    d = 0.5 * (right - left)
    return np.asarray([[s, -1j * d, 0.0], [1j * d, s, 0.0], [0.0, 0.0, plasma]])


def _wave_normal(theta: float) -> np.ndarray:
    return np.asarray([np.sin(theta), 0.0, np.cos(theta)])


def _determinant_roots(epsilon: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Roots in ``n²`` of ``det(n² (κκ − I) + ε) = 0``.

    The transverse projector ``I − κκ`` has rank two, so the determinant is an
    exact quadratic in ``n²``; it is interpolated from three dense determinants.
    """
    projector = np.eye(3) - np.outer(kappa, kappa)
    step = float(np.max(np.abs(epsilon)))
    center, forward, backward = (
        np.linalg.det(epsilon - value * projector) for value in (0.0, step, -step)
    )
    curvature = 0.5 * (forward + backward) - center
    slope = 0.5 * (forward - backward)
    return step * np.roots([curvature, slope, center])


@pytest.mark.parametrize("seed", [0, 1, 2], ids=["seed0", "seed1", "seed2"])
def test_roots_match_determinant_of_wave_operator(seed: int) -> None:
    rng = np.random.default_rng(seed)
    density = 1.0e16
    densities = np.asarray([density, 0.7 * density, 0.3 * density])
    charges = np.asarray([-1.0, 1.0, 2.0])
    ratios = np.asarray([1.0, PROTON_RATIO, 4.0 * PROTON_RATIO])
    collisions = np.asarray([2.0e8, 5.0e6, 3.0e6])
    field = 0.02
    medium = ColdPlasmaDielectric(
        SCALE,
        densities=densities,
        charge_numbers=charges,
        mass_ratios=ratios,
        magnetic_field=[0.0, 0.0, field],
        collision_frequencies=collisions,
    )
    omega = rng.uniform(0.2, 3.0, size=5) * _plasma_frequency(density)
    theta = rng.uniform(0.0, np.pi, size=5)
    result = medium.refractive_indices(omega, theta)

    for index in range(5):
        epsilon = _reference_dielectric(
            omega[index], densities, charges, ratios, field, collisions
        )
        kappa = _wave_normal(theta[index])
        expected = np.sort_complex(_determinant_roots(epsilon, kappa))
        actual = np.sort_complex(np.asarray(result.n_squared[index]))
        np.testing.assert_allclose(actual, expected, rtol=1.0e-11)
        operator = (
            np.asarray(result.n_squared[index])[:, None, None]
            * (np.outer(kappa, kappa) - np.eye(3))
            + epsilon
        )
        for root in range(2):
            vector = np.asarray(result.polarization[index, root])
            assert abs(np.linalg.norm(vector) - 1.0) < 1.0e-12
            assert np.linalg.norm(operator[root] @ vector) < 1.0e-12 * np.linalg.norm(
                operator[root]
            )
        assert np.all(np.asarray(result.n_squared[index]).imag > 0.0)


def test_unmagnetized_limit_is_isotropic_with_explicit_degeneracy() -> None:
    density = 1.0e16
    wp = _plasma_frequency(density)
    medium = _electron_plasma(density, 0.0)
    omega = np.asarray([0.5 * wp, 2.0 * wp])
    result = medium.refractive_indices(omega, 0.7)

    expected = 1.0 - (wp / omega) ** 2
    np.testing.assert_allclose(
        np.asarray(result.n_squared),
        np.broadcast_to(expected[:, None], (2, 2)),
        rtol=1.0e-14,
    )
    status = np.asarray(result.status)
    evanescent = status & ColdPlasmaWaveStatus.EVANESCENT.value
    assert np.all(evanescent[0] != 0) and np.all(evanescent[1] == 0)
    assert np.all(status & ColdPlasmaWaveStatus.POLARIZATION_UNDEFINED.value)
    assert np.all(status & ColdPlasmaWaveStatus.PARALLEL_LABEL_AMBIGUOUS.value)
    assert np.all(status & ColdPlasmaWaveStatus.PERPENDICULAR_LABEL_AMBIGUOUS.value)
    assert np.all(np.asarray(result.parallel_branch_separation) == 0.0)
    index = np.asarray(result.refractive_index[0])
    assert np.all(index.real == 0.0) and np.all(index.imag == np.sqrt(-expected[0]))


def test_parallel_propagation_gives_labeled_right_and_left_closed_forms() -> None:
    density, field = 1.0e16, 0.01
    wp = _plasma_frequency(density)
    gyro = _cyclotron_frequency(field, 1.0, 1.0)
    omega = 3.0 * wp
    x, y = (wp / omega) ** 2, gyro / omega
    result = _electron_plasma(density, field).refractive_indices(omega, 0.0)

    right = complex(result.select(PlasmaWaveMode.RIGHT, result.n_squared))
    left = complex(result.select(PlasmaWaveMode.LEFT, result.n_squared))
    assert right == pytest.approx(1.0 - x / (1.0 - y), rel=1.0e-13)
    assert left == pytest.approx(1.0 - x / (1.0 + y), rel=1.0e-13)
    assert complex(result.select(PlasmaWaveMode.RIGHT, result.transverse_ratio)) == (
        pytest.approx(1.0, abs=1.0e-12)
    )
    assert complex(result.select(PlasmaWaveMode.LEFT, result.transverse_ratio)) == (
        pytest.approx(-1.0, abs=1.0e-12)
    )
    np.testing.assert_allclose(
        np.asarray(result.longitudinal_component), 0.0, atol=1.0e-15
    )
    assert np.all(np.asarray(result.status) == 0)
    assert sorted(np.asarray(result.parallel_mode).tolist()) == [0, 1]


def test_perpendicular_propagation_gives_ordinary_and_extraordinary_closed_forms() -> (
    None
):
    density, field = 1.0e16, 0.01
    wp = _plasma_frequency(density)
    gyro = _cyclotron_frequency(field, 1.0, 1.0)
    omega = 3.0 * wp
    x, y = (wp / omega) ** 2, gyro / omega
    result = _electron_plasma(density, field).refractive_indices(omega, 0.5 * np.pi)

    ordinary = complex(result.select(PlasmaWaveMode.ORDINARY, result.n_squared))
    extraordinary = complex(result.select(PlasmaWaveMode.EXTRAORDINARY, result.n_squared))
    assert ordinary == pytest.approx(1.0 - x, rel=1.0e-13)
    assert extraordinary == pytest.approx(
        1.0 - x * (1.0 - x) / (1.0 - x - y * y), rel=1.0e-13
    )
    ordinary_vector = np.asarray(
        result.select(PlasmaWaveMode.ORDINARY, result.polarization)
    )
    extraordinary_vector = np.asarray(
        result.select(PlasmaWaveMode.EXTRAORDINARY, result.polarization)
    )
    np.testing.assert_allclose(ordinary_vector, [0.0, 0.0, 1.0], atol=1.0e-14)
    assert abs(extraordinary_vector[2]) < 1.0e-14
    assert abs(extraordinary_vector[1]) > 0.99
    longitudinal = complex(
        result.select(PlasmaWaveMode.EXTRAORDINARY, result.longitudinal_component)
    )
    assert abs(longitudinal) == pytest.approx(abs(extraordinary_vector[0]), rel=1.0e-12)
    assert np.all(np.asarray(result.status) == 0)


def test_whistler_branch_and_resonance_cone_evanescence() -> None:
    density, field = 1.0e18, 0.1
    wp = _plasma_frequency(density)
    gyro = _cyclotron_frequency(field, 1.0, 1.0)
    omega = 0.05 * gyro
    x, y = (wp / omega) ** 2, gyro / omega
    medium = _electron_plasma(density, field)
    cone = medium.resonance_cone(omega)
    assert bool(cone.exists)
    cone_angle = float(cone.angle)
    exact = (x - 1.0) * (y * y - 1.0) / (x + y * y - 1.0)
    assert np.tan(cone_angle) ** 2 == pytest.approx(exact, rel=1.0e-12)

    angles = np.asarray([0.0, 0.4, 0.8, 1.2, cone_angle + 0.05])
    result = medium.refractive_indices(omega, angles)
    whistler = np.asarray(result.select(PlasmaWaveMode.RIGHT, result.n_squared))
    status = np.asarray(result.select(PlasmaWaveMode.RIGHT, result.status))
    quasi_longitudinal = 1.0 - x / (1.0 - y * np.cos(angles[:-1]))
    np.testing.assert_allclose(whistler[:-1].real, quasi_longitudinal, rtol=1.0e-2)
    assert whistler[0].real == pytest.approx(1.0 - x / (1.0 - y), rel=1.0e-13)
    assert np.all(whistler[:-1].real > 1.0)
    assert np.all(status[:-1] & ColdPlasmaWaveStatus.EVANESCENT.value == 0)
    assert whistler[-1].real < 0.0
    assert status[-1] & ColdPlasmaWaveStatus.EVANESCENT.value
    assert np.all(
        np.asarray(result.select(PlasmaWaveMode.LEFT, result.n_squared)).real < 0.0
    )
    ambiguous = (
        ColdPlasmaWaveStatus.PARALLEL_LABEL_AMBIGUOUS.value
        | ColdPlasmaWaveStatus.PERPENDICULAR_LABEL_AMBIGUOUS.value
    )
    assert not np.any(np.asarray(result.status) & ambiguous)


def test_quasi_longitudinal_and_quasi_transverse_limits() -> None:
    density, field = 1.0e16, 0.01
    wp = _plasma_frequency(density)
    gyro = _cyclotron_frequency(field, 1.0, 1.0)
    omega = 4.0 * wp
    x, y = (wp / omega) ** 2, gyro / omega
    medium = _electron_plasma(density, field)

    theta = 0.2
    ql = medium.refractive_indices(omega, theta)
    assert float(ql.quasi_longitudinal_term) > 10.0 * float(ql.quasi_transverse_term)
    right = complex(ql.select(PlasmaWaveMode.RIGHT, ql.n_squared))
    left = complex(ql.select(PlasmaWaveMode.LEFT, ql.n_squared))
    assert right == pytest.approx(1.0 - x / (1.0 - y * np.cos(theta)), rel=1.0e-4)
    assert left == pytest.approx(1.0 - x / (1.0 + y * np.cos(theta)), rel=1.0e-4)
    # Above every cutoff (CMA region 1) the R branch connects to X and L to O.
    assert int(ql.select(PlasmaWaveMode.RIGHT, ql.perpendicular_mode)) == (
        PlasmaWaveMode.EXTRAORDINARY.value
    )
    assert int(ql.select(PlasmaWaveMode.LEFT, ql.perpendicular_mode)) == (
        PlasmaWaveMode.ORDINARY.value
    )

    theta = 0.5 * np.pi - 1.0e-3
    qt = medium.refractive_indices(omega, theta)
    assert float(qt.quasi_transverse_term) > 10.0 * float(qt.quasi_longitudinal_term)
    ordinary = complex(qt.select(PlasmaWaveMode.ORDINARY, qt.n_squared))
    extraordinary = complex(qt.select(PlasmaWaveMode.EXTRAORDINARY, qt.n_squared))
    sin2 = np.sin(theta) ** 2
    assert ordinary == pytest.approx(1.0 - x, rel=1.0e-6)
    assert extraordinary == pytest.approx(
        1.0 - x * (1.0 - x) / (1.0 - x - y * y * sin2), rel=1.0e-6
    )


def test_branch_labels_are_continuous_and_refinement_stable() -> None:
    density, field = 1.0e16, 0.05
    wp = _plasma_frequency(density)
    omega = 0.8 * wp
    coarse = ColdPlasmaDielectric(
        SCALE,
        densities=[density, density],
        charge_numbers=[-1.0, 1.0],
        mass_ratios=[1.0, PROTON_RATIO],
        magnetic_field=[0.0, 0.0, field],
        collision_frequencies=[3.0e8, 1.0e6],
        continuation_steps=8,
    )
    fine = ColdPlasmaDielectric(
        SCALE,
        densities=[density, density],
        charge_numbers=[-1.0, 1.0],
        mass_ratios=[1.0, PROTON_RATIO],
        magnetic_field=[0.0, 0.0, field],
        collision_frequencies=[3.0e8, 1.0e6],
        continuation_steps=512,
    )
    angles = np.linspace(0.0, 0.5 * np.pi, 41)
    coarse_result = coarse.refractive_indices(omega, angles)
    fine_result = fine.refractive_indices(omega, angles)

    assert not np.any(
        np.asarray(fine_result.status)
        & (
            ColdPlasmaWaveStatus.PARALLEL_LABEL_AMBIGUOUS.value
            | ColdPlasmaWaveStatus.PERPENDICULAR_LABEL_AMBIGUOUS.value
        )
    )
    right = np.asarray(fine_result.select(PlasmaWaveMode.RIGHT, fine_result.n_squared))
    left = np.asarray(fine_result.select(PlasmaWaveMode.LEFT, fine_result.n_squared))
    stix = fine_result.stix
    assert right[0] == pytest.approx(complex(stix.right_term[0]), rel=1.0e-12)
    assert left[0] == pytest.approx(complex(stix.left_term[0]), rel=1.0e-12)
    jumps = np.abs(np.diff(right)) + np.abs(np.diff(left))
    swaps = np.abs(right[1:] - left[:-1]) + np.abs(left[1:] - right[:-1])
    assert np.all(jumps < swaps)
    ordinary = complex(
        fine_result.select(PlasmaWaveMode.ORDINARY, fine_result.n_squared)[-1]
    )
    assert ordinary == pytest.approx(complex(stix.plasma_term[-1]), rel=1.0e-12)
    for mode in (PlasmaWaveMode.RIGHT, PlasmaWaveMode.ORDINARY):
        coarse_values = np.asarray(coarse_result.select(mode, coarse_result.n_squared))
        fine_values = np.asarray(fine_result.select(mode, fine_result.n_squared))
        np.testing.assert_array_equal(coarse_values, fine_values)


def test_cutoff_and_resonance_frequencies_match_analytic_values() -> None:
    density, field = 1.0e16, 0.01
    wpe = _plasma_frequency(density)
    gyro_e = _cyclotron_frequency(field, 1.0, 1.0)
    electrons = _electron_plasma(density, field).characteristic_frequencies()

    right_cutoff = 0.5 * (gyro_e + np.sqrt(gyro_e**2 + 4.0 * wpe**2))
    left_cutoff = 0.5 * (-gyro_e + np.sqrt(gyro_e**2 + 4.0 * wpe**2))
    np.testing.assert_allclose(
        np.asarray(electrons.right_cutoffs), [-left_cutoff, right_cutoff], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(electrons.left_cutoffs), [-right_cutoff, left_cutoff], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(electrons.plasma_cutoffs), [-wpe, wpe], rtol=1.0e-12
    )
    upper_hybrid = np.sqrt(wpe**2 + gyro_e**2)
    np.testing.assert_allclose(
        np.asarray(electrons.hybrid_resonances),
        [-upper_hybrid, upper_hybrid],
        rtol=1.0e-12,
    )
    assert np.all(np.asarray(electrons.right_residuals) < 1.0e-10)
    assert np.all(np.asarray(electrons.hybrid_residuals) < 1.0e-10)

    two_species = ColdPlasmaDielectric(
        SCALE,
        densities=[density, density],
        charge_numbers=[-1.0, 1.0],
        mass_ratios=[1.0, PROTON_RATIO],
        magnetic_field=[0.0, 0.0, field],
    ).characteristic_frequencies()
    wpi = _plasma_frequency(density, 1.0, PROTON_RATIO)
    gyro_i = _cyclotron_frequency(field, 1.0, PROTON_RATIO)
    total = wpe**2 + wpi**2 + gyro_e**2 + gyro_i**2
    product = gyro_e**2 * gyro_i**2 + wpe**2 * gyro_i**2 + wpi**2 * gyro_e**2
    hybrids = np.sqrt(
        0.5 * (total + np.asarray([-1.0, 1.0]) * np.sqrt(total**2 - 4.0 * product))
    )
    np.testing.assert_allclose(
        np.asarray(two_species.hybrid_resonances),
        np.concatenate((-hybrids[::-1], hybrids)),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(two_species.plasma_cutoffs),
        np.asarray([-1.0, 1.0]) * np.sqrt(wpe**2 + wpi**2),
        rtol=1.0e-12,
    )
    # Charge neutrality cancels the ω = 0 pole of R; the remaining cutoffs solve
    # ω² + (Ω_e + Ω_i) ω + Ω_e Ω_i − ω_pe² − ω_pi² = 0 with signed Ω_e = −|Ω_e|.
    right_expected = np.sort(
        np.roots([1.0, gyro_i - gyro_e, -gyro_e * gyro_i - wpe**2 - wpi**2])
    )
    np.testing.assert_allclose(
        np.asarray(two_species.right_cutoffs), right_expected, rtol=1.0e-12
    )
    assert np.all(np.asarray(two_species.right_residuals) < 1.0e-10)


def test_collisional_cutoffs_are_complex_with_small_residuals() -> None:
    medium = _electron_plasma(1.0e16, 0.01, collision=5.0e8)
    frequencies = medium.characteristic_frequencies()

    assert np.all(np.asarray(frequencies.plasma_cutoffs).imag != 0.0)
    assert np.all(np.asarray(frequencies.plasma_residuals) < 1.0e-10)
    assert np.all(np.asarray(frequencies.left_residuals) < 1.0e-10)
    cone = medium.resonance_cone(0.5 * _cyclotron_frequency(0.01, 1.0, 1.0))
    assert not bool(cone.exists)
    assert complex(cone.tangent_squared).imag != 0.0


def test_faraday_rotation_matches_high_frequency_limit_along_field() -> None:
    density, field = 1.0e16, 0.01
    wp = _plasma_frequency(density)
    gyro = _cyclotron_frequency(field, 1.0, 1.0)
    omega = 60.0 * wp
    x, y = (wp / omega) ** 2, gyro / omega
    assert y < 1.0e-2
    faraday = _electron_plasma(density, field).faraday_coefficients(omega, 0.0)

    expected = E**3 * density * field / (2.0 * EPS0 * M_E**2 * C * omega**2)
    rotation = complex(faraday.rotation)
    assert rotation.real == pytest.approx(expected, rel=1.0e-3)
    assert rotation.imag == 0.0
    # Q and U vanish exactly along B₀; their rounding floor is ε/(XY) relative to V,
    # the conditioning of the circular mode vectors.
    floor = np.finfo(np.float64).eps / (x * y) * expected
    assert abs(complex(faraday.conversion)) < floor
    assert abs(complex(faraday.coefficients[1])) < floor
    assert float(faraday.mode_overlap) < np.finfo(np.float64).eps / (x * y)
    np.testing.assert_allclose(np.asarray(faraday.transverse_fraction), 1.0, atol=1.0e-14)


def test_faraday_conversion_matches_high_frequency_limit_across_field() -> None:
    density, field = 1.0e16, 0.05
    wp = _plasma_frequency(density)
    gyro = _cyclotron_frequency(field, 1.0, 1.0)
    omega = 20.0 * wp
    x, y = (wp / omega) ** 2, gyro / omega
    faraday = _electron_plasma(density, field).faraday_coefficients(omega, 0.5 * np.pi)

    ordinary = np.sqrt(1.0 - x)
    extraordinary = np.sqrt(1.0 - x * (1.0 - x) / (1.0 - x - y * y))
    exact = (omega / (2.0 * C)) * (extraordinary - ordinary)
    conversion = complex(faraday.conversion)
    assert conversion.real == pytest.approx(exact, rel=1.0e-10)
    leading_order = -(omega / (4.0 * C)) * x * y * y
    assert conversion.real == pytest.approx(leading_order, rel=2.0 * (x + y * y))
    assert abs(complex(faraday.rotation)) < 1.0e-10 * abs(leading_order)
    assert float(faraday.mode_overlap) < 1.0e-12


def test_collisions_damp_both_modes_and_sign_is_absorption() -> None:
    density, field = 1.0e16, 0.01
    wp = _plasma_frequency(density)
    omega = 2.5 * wp
    lossless = _electron_plasma(density, field).refractive_indices(omega, 0.6)
    lossy = _electron_plasma(density, field, collision=2.0e8).refractive_indices(
        omega, 0.6
    )

    assert np.all(np.asarray(lossless.n_squared).imag == 0.0)
    assert np.all(np.asarray(lossy.n_squared).imag > 0.0)
    assert np.all(np.asarray(lossy.refractive_index).imag > 0.0)
    assert np.all(np.asarray(lossy.refractive_index).real > 0.0)
    assert np.all(np.asarray(lossy.stix.plasma_term).imag > 0.0)
    assert np.all(np.asarray(lossy.status) == 0)


def test_float32_and_complex_arguments_are_refused() -> None:
    medium = _electron_plasma(1.0e16, 0.01)
    wp = _plasma_frequency(1.0e16)
    with pytest.raises(TypeError, match="float64"):
        medium.refractive_indices(jnp.asarray(2.0 * wp, dtype=jnp.float32), 0.0)
    with pytest.raises(TypeError, match="float64"):
        medium.refractive_indices(2.0 * wp, jnp.asarray(0.3, dtype=jnp.float32))
    with pytest.raises(TypeError, match="float64"):
        medium.stix_parameters(jnp.asarray(2.0 * wp + 0.0j))
    result = medium.refractive_indices(2.0 * wp, 0)
    assert result.n_squared.dtype == jnp.complex128
    assert result.angle.dtype == jnp.float64


def test_constructor_refuses_invalid_species_tables() -> None:
    with pytest.raises(ValueError, match="densities"):
        ColdPlasmaDielectric(
            SCALE,
            densities=[-1.0],
            charge_numbers=[-1.0],
            mass_ratios=[1.0],
            magnetic_field=[0.0, 0.0, 0.1],
        )
    with pytest.raises(ValueError, match="charge_numbers"):
        ColdPlasmaDielectric(
            SCALE,
            densities=[1.0e16],
            charge_numbers=[0.0],
            mass_ratios=[1.0],
            magnetic_field=[0.0, 0.0, 0.1],
        )
    with pytest.raises(ValueError):
        ColdPlasmaDielectric(
            SCALE,
            densities=[1.0e16, 1.0e16],
            charge_numbers=[-1.0],
            mass_ratios=[1.0, 2.0],
            magnetic_field=[0.0, 0.0, 0.1],
        )
    with pytest.raises(ValueError, match="collision_frequencies"):
        ColdPlasmaDielectric(
            SCALE,
            densities=[1.0e16],
            charge_numbers=[-1.0],
            mass_ratios=[1.0],
            magnetic_field=[0.0, 0.0, 0.1],
            collision_frequencies=[-1.0],
        )
    with pytest.raises(TypeError):
        ColdPlasmaDielectric(
            SCALE.relativity,  # ty: ignore[invalid-argument-type]
            densities=[1.0e16],
            charge_numbers=[-1.0],
            mass_ratios=[1.0],
            magnetic_field=[0.0, 0.0, 0.1],
        )
