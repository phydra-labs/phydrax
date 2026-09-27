from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spectral import BrillouinZonePlan, LatticeHarmonicPlan


def _lattice() -> Any:
    return LatticeHarmonicPlan.parallelogramic((3,), (9,)).prepare(
        jnp.asarray(((2.0, 0.0),))
    )


def test_lattice_harmonics_scenario_1() -> None:
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


def test_lattice_harmonics_scenario_2() -> None:
    lattice = _lattice()
    rule = BrillouinZonePlan((4,)).prepare(lattice)
    assert np.any(np.all(np.asarray(rule.wavevectors) == 0.0, axis=-1))
    assert float(jnp.sum(rule.weights)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="minimum"):
        LatticeHarmonicPlan.parallelogramic((3,), (3,))
    with pytest.raises(ValueError, match="conjugation"):
        # ty: ignore[invalid-argument-type]
        LatticeHarmonicPlan(((0,), (1,)), (5,))
