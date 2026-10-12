#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

import phydrax as phx


pytestmark = pytest.mark.strict_jax


def _parameters() -> phx.applications.solid_mechanics.MPMPhaseFieldParameters:
    return phx.applications.solid_mechanics.MPMPhaseFieldParameters(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0),
        1.0,
        0.1,
        1e-6,
    )


def _independent_energy(deformation: NDArray[np.float64], damage: float) -> float:
    eigenvalues = np.linalg.eigvalsh(deformation.T @ deformation)
    hencky = 0.5 * np.log(eigenvalues)
    positive = (
        2.0 * np.sum(np.maximum(hencky, 0.0) ** 2) + 4.0 * max(np.sum(hencky), 0.0) ** 2
    )
    negative = (
        2.0 * np.sum(np.minimum(hencky, 0.0) ** 2) + 4.0 * min(np.sum(hencky), 0.0) ** 2
    )
    return float(((1 - damage) ** 2 + 1e-6) * positive + negative)


@pytest.mark.parametrize("dimension", [2, 3])
def test_fracture_rest_has_finite_primal_and_uncertified_tangent(dimension: int) -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(
        dimension
    )
    initial = material.initialize_state((), jnp.float64)
    response = material.evaluate_linearized(
        jnp.eye(dimension, dtype=jnp.float64), initial, 1.0, _parameters(), 0.0, 0.1
    )
    assert bool(response.response.successful)
    assert not bool(response.tangent_successful)
    assert not bool(response.response.diagnostics["split_derivative_valid"])
    np.testing.assert_allclose(response.response.first_piola, 0.0, atol=0.0)
    assert bool(jnp.all(jnp.isfinite(response.algorithmic_tangent)))
    np.testing.assert_array_equal(response.response.trial_state, initial)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("stretch", [0.9, 1.1])
def test_fracture_repeated_strict_spectrum_tangent(
    dimension: int, stretch: float
) -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(
        dimension
    )
    deformation = stretch * jnp.eye(dimension, dtype=jnp.float64)
    state = jnp.asarray((0.3, 0.1), dtype=jnp.float64)
    response = eqx.filter_jit(material.evaluate_linearized)(
        deformation, state, 1.0, _parameters(), 0.0, 0.1
    )
    assert bool(response.response.successful)
    assert bool(response.tangent_successful)
    direction = (
        jnp.arange(1, dimension**2 + 1, dtype=jnp.float64).reshape((dimension, dimension))
        / 10.0
    )
    epsilon = 1e-6
    plus = material.evaluate(
        deformation + epsilon * direction, state, 1.0, _parameters(), 0.0, 0.1
    )
    minus = material.evaluate(
        deformation - epsilon * direction, state, 1.0, _parameters(), 0.0, 0.1
    )
    predicted = phx.ein.contract("ijkl,kl->ij", response.algorithmic_tangent, direction)
    np.testing.assert_allclose(
        predicted,
        (plus.first_piola - minus.first_piola) / (2 * epsilon),
        rtol=2e-7,
        atol=1e-8,
    )


def test_fracture_piola_matches_independent_energy_gradient_and_objectivity() -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(3)
    deformation = np.asarray(
        ((1.18, 0.07, 0.02), (0.0, 0.91, 0.03), (0.01, 0.0, 1.07)), dtype=np.float64
    )
    state = jnp.asarray((0.4, 0.0), dtype=jnp.float64)
    response = material.evaluate(deformation, state, 1.0, _parameters(), 0.0, 0.1)
    expected = np.zeros((3, 3), dtype=np.float64)
    epsilon = 1e-6
    for row in range(3):
        for column in range(3):
            direction = np.zeros((3, 3), dtype=np.float64)
            direction[row, column] = epsilon
            expected[row, column] = (
                _independent_energy(deformation + direction, 0.4)
                - _independent_energy(deformation - direction, 0.4)
            ) / (2 * epsilon)
    np.testing.assert_allclose(response.first_piola, expected, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(
        response.reference_energy_density,
        _independent_energy(deformation, 0.4),
        rtol=1e-12,
    )
    rotation = jnp.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=jnp.float64
    )
    rotated = material.evaluate(
        rotation @ deformation, state, 1.0, _parameters(), 0.0, 0.1
    )
    np.testing.assert_allclose(
        rotated.first_piola, rotation @ response.first_piola, atol=1e-12
    )
    np.testing.assert_allclose(rotated.trial_state, response.trial_state, atol=1e-12)


@pytest.mark.parametrize(
    "stretches",
    [(1.0, 1.2, 0.8), (2.0, 2.0, 0.25)],
    ids=["principal-zero", "trace-zero"],
)
def test_fracture_exact_split_kinks_are_reported(
    stretches: tuple[float, float, float],
) -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(3)
    response = material.evaluate_linearized(
        jnp.diag(jnp.asarray(stretches, dtype=jnp.float64)),
        jnp.asarray((0.4, 0.0)),
        1.0,
        _parameters(),
        0.0,
        0.1,
    )
    assert bool(response.response.successful)
    assert not bool(response.tangent_successful)
    assert bool(jnp.all(jnp.isfinite(response.response.first_piola)))
    assert bool(jnp.all(jnp.isfinite(response.algorithmic_tangent)))


def test_fracture_does_not_turn_near_kinks_into_kinks() -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(3)
    deformation = jnp.diag(jnp.asarray((1.0 + 1e-10, 1.2, 1.2), dtype=jnp.float64))
    response = material.evaluate_linearized(
        deformation, jnp.asarray((0.4, 0.0)), 1.0, _parameters(), 0.0, 0.1
    )
    assert bool(response.tangent_successful)


@pytest.mark.parametrize("kind", ["folded", "singular", "nonfinite"])
def test_fracture_invalid_deformation_retains_history_and_finite_response(
    kind: str,
) -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(3)
    first = {"folded": -1.0, "singular": 0.0, "nonfinite": np.nan}[kind]
    state = jnp.asarray((0.3, 0.8), dtype=jnp.float64)
    response = material.evaluate(
        jnp.diag(jnp.asarray((first, 1.0, 1.0))), state, 1.0, _parameters(), 0.0, 0.1
    )
    assert not bool(response.successful)
    assert not bool(response.admissible)
    np.testing.assert_array_equal(response.trial_state, state)
    assert bool(jnp.all(jnp.isfinite(response.first_piola)))
    assert bool(jnp.isfinite(response.reference_energy_density))


@pytest.mark.parametrize("state", [(-0.1, 0.0), (1.1, 0.0), (0.2, -1.0)])
def test_fracture_invalid_history_is_not_clipped(state: tuple[float, float]) -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(3)
    initial = jnp.asarray(state, dtype=jnp.float64)
    response = material.evaluate(
        1.1 * jnp.eye(3, dtype=jnp.float64), initial, 1.0, _parameters(), 0.0, 0.1
    )
    assert not bool(response.successful)
    np.testing.assert_array_equal(response.trial_state, initial)


def test_fracture_history_is_irreversible_under_unloading() -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(3)
    initial = jnp.asarray((0.4, 0.0), dtype=jnp.float64)
    loading = material.evaluate(
        1.2 * jnp.eye(3, dtype=jnp.float64), initial, 1.0, _parameters(), 0.0, 0.1
    )
    unloading = material.evaluate(
        0.9 * jnp.eye(3, dtype=jnp.float64),
        loading.trial_state,
        1.0,
        _parameters(),
        0.1,
        0.1,
    )
    assert float(loading.trial_state[1]) > 0.0
    np.testing.assert_array_equal(unloading.trial_state, loading.trial_state)
    np.testing.assert_allclose(
        unloading.diagnostics["tensile_energy_density"], 0.0, atol=0.0
    )
