from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spectral import BrillouinZonePlan, LatticeHarmonicPlan


def _lattice():
    return LatticeHarmonicPlan.parallelogramic((3,), (9,)).prepare(
        jnp.asarray(((2.0, 0.0),))
    )


def test_reciprocal_duality_and_pixel_centers() -> None:
    lattice = _lattice()
    np.testing.assert_allclose(
        np.asarray(lattice.primitive_vectors @ lattice.reciprocal_vectors.T),
        np.asarray([[2.0 * np.pi]]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(lattice.fractional_coordinates[0]),
        np.asarray([0.5 / 9.0]),
    )
    assert float(lattice.cell_measure) == pytest.approx(2.0)


def test_selected_fft_round_trip_and_convolution() -> None:
    lattice = _lattice()
    coefficients = jnp.asarray((1.0 + 0.2j, -0.3 + 0.5j, 0.7 - 0.1j))
    reconstructed = lattice.synthesis(coefficients)
    np.testing.assert_allclose(
        np.asarray(lattice.analysis(reconstructed)),
        np.asarray(coefficients),
        rtol=1e-12,
        atol=1e-12,
    )
    coordinate = lattice.fractional_coordinates[..., 0]
    material = 2.0 + 0.3 * jnp.cos(2.0 * jnp.pi * coordinate)
    matrix = lattice.convolution_matrix(material)
    np.testing.assert_allclose(
        np.asarray(matrix), np.asarray(matrix.conj().T), atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(jnp.diag(matrix)), 2.0, atol=1e-12)


def test_translation_is_covariant_and_periodic() -> None:
    lattice = _lattice()
    coordinate = lattice.fractional_coordinates[..., 0]
    material = 2.0 + 0.3 * jnp.cos(2.0 * jnp.pi * coordinate)
    matrix = lattice.convolution_matrix(material)
    displacement = jnp.asarray((0.37, 0.0))
    translated = lattice.translate_convolution(matrix, displacement)
    restored = lattice.translate_convolution(translated, -displacement)
    periodic = lattice.translate_convolution(matrix, jnp.asarray((2.0, 0.0)))
    np.testing.assert_allclose(np.asarray(restored), np.asarray(matrix), atol=1e-12)
    np.testing.assert_allclose(np.asarray(periodic), np.asarray(matrix), atol=1e-12)


def test_brillouin_rule_contains_gamma_and_normalizes() -> None:
    lattice = _lattice()
    rule = BrillouinZonePlan((4,)).prepare(lattice)
    assert np.any(np.all(np.asarray(rule.wavevectors) == 0.0, axis=-1))
    assert float(jnp.sum(rule.weights)) == pytest.approx(1.0)


def test_layout_rejects_underresolved_grid_and_nonclosed_custom_set() -> None:
    with pytest.raises(ValueError, match="minimum"):
        LatticeHarmonicPlan.parallelogramic((3,), (3,))
    with pytest.raises(ValueError, match="conjugation"):
        LatticeHarmonicPlan(((0,), (1,)), (5,))
