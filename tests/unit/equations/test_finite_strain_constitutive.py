# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.applications.solid_mechanics import (
    DruckerPragerMPMConstitutivePlan,
    DruckerPragerParameters,
    FiniteStrainJ2MPMConstitutivePlan,
    FiniteStrainJ2Parameters,
)
from phydrax.applications.solid_mechanics._principal_hencky import (
    initialize_metric_state,
    pack_metric,
    unpack_metric,
)


def test_plastic_metric_packing_round_trip() -> None:
    metric = jnp.asarray(
        [[2.0, 0.1, 0.2], [0.1, 3.0, 0.3], [0.2, 0.3, 4.0]], dtype=jnp.float64
    )
    np.testing.assert_allclose(pack_metric(metric), [2.0, 3.0, 4.0, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(unpack_metric(pack_metric(metric)), metric)
    state = initialize_metric_state((2,), jnp.float64)
    assert state.shape == (2, 7)
    np.testing.assert_allclose(
        unpack_metric(state[:, :6]), np.broadcast_to(np.eye(3), (2, 3, 3))
    )


@pytest.mark.parametrize(
    "strain",
    [(0.12, 0.0, 0.0), (0.10, -0.10, 0.0)],
    ids=["uniaxial-log-strain", "principal-shear"],
)
def test_j2_closed_form_principal_return(strain: tuple[float, float, float]) -> None:
    material = FiniteStrainJ2MPMConstitutivePlan()
    parameters = FiniteStrainJ2Parameters(10.0, 30.0, 0.5, 2.0)
    epsilon = np.asarray(strain)
    deformation = jnp.diag(jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64)))
    state = material.initialize_state((), jnp.float64)
    response = material.evaluate(
        deformation, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    )
    s_trial = 20.0 * (epsilon - epsilon.mean())
    q_trial = np.sqrt(1.5 * np.sum(s_trial**2))
    gamma = (q_trial - 0.5) / 32.0
    tau = (1.0 - 30.0 * gamma / q_trial) * s_trial + 30.0 * epsilon.sum()
    assert bool(response.successful)
    assert int(response.branch_code) == 1
    np.testing.assert_allclose(
        response.first_piola @ deformation.T, np.diag(tau), atol=1e-10
    )
    np.testing.assert_allclose(response.trial_state[-1], gamma, atol=1e-12)
    np.testing.assert_allclose(response.dissipation_increment, 0.5 * gamma, atol=1e-12)
    np.testing.assert_allclose(response.diagnostics["yield_residual"], 0.0, atol=1e-12)
    metric = unpack_metric(response.trial_state[:6])
    np.testing.assert_allclose(np.linalg.det(np.asarray(metric)), 1.0, atol=1e-12)


@pytest.mark.parametrize("stretch", [1.0, 0.999])
def test_j2_repeated_spectrum_tangent_matches_finite_difference(stretch: float) -> None:
    material = FiniteStrainJ2MPMConstitutivePlan()
    parameters = FiniteStrainJ2Parameters(10.0, 30.0, 100.0, 2.0)
    state = material.initialize_state((), jnp.float64)
    deformation = jnp.diag(jnp.asarray([1.0, 1.0, stretch], dtype=jnp.float64))
    direction = jnp.asarray(
        [[0.2, 0.3, -0.1], [0.1, -0.1, 0.2], [0.2, 0.0, 0.1]], dtype=jnp.float64
    )
    linearized = material.evaluate_linearized(
        deformation, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    )
    assert bool(linearized.tangent_successful)
    assert bool(jnp.all(jnp.isfinite(linearized.algorithmic_tangent)))
    h = 1e-6
    plus = material.evaluate(
        deformation + h * direction, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    ).first_piola
    minus = material.evaluate(
        deformation - h * direction, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    ).first_piola
    predicted = jnp.einsum("ijkl,kl->ij", linearized.algorithmic_tangent, direction)
    np.testing.assert_allclose(
        predicted, (plus - minus) / (2.0 * h), rtol=2e-6, atol=2e-7
    )


def test_j2_objectivity_and_plastic_branch_tangent() -> None:
    material = FiniteStrainJ2MPMConstitutivePlan()
    parameters = FiniteStrainJ2Parameters(10.0, 30.0, 0.5, 2.0)
    state = material.initialize_state((), jnp.float64)
    deformation = jnp.asarray(
        [[1.1, 0.15, 0.0], [0.0, 0.96, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64
    )
    angle = 0.7
    rotation = jnp.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    response = material.evaluate(
        deformation, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    )
    rotated = material.evaluate(
        rotation @ deformation, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    )
    np.testing.assert_allclose(
        rotated.first_piola, rotation @ response.first_piola, atol=1e-10
    )
    np.testing.assert_allclose(rotated.trial_state, response.trial_state, atol=1e-10)
    direction = jnp.asarray(
        [[0.1, 0.2, 0.0], [-0.1, 0.03, 0.1], [0.02, 0.0, -0.1]], dtype=jnp.float64
    )
    linearized = material.evaluate_linearized(
        deformation, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    )
    h = 1e-6
    plus = material.evaluate(
        deformation + h * direction, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    ).first_piola
    minus = material.evaluate(
        deformation - h * direction, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    ).first_piola
    assert bool(linearized.tangent_successful)
    np.testing.assert_allclose(
        jnp.einsum("ijkl,kl->ij", linearized.algorithmic_tangent, direction),
        (plus - minus) / (2.0 * h),
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.parametrize("failure", ["inverted", "singular", "indefinite-metric"])
def test_j2_failure_retains_committed_history(failure: str) -> None:
    material = FiniteStrainJ2MPMConstitutivePlan()
    parameters = FiniteStrainJ2Parameters(10.0, 30.0, 0.5, 2.0)
    state = material.initialize_state((), jnp.float64).at[-1].set(0.2)
    deformation = jnp.eye(3, dtype=jnp.float64)
    if failure == "inverted":
        deformation = deformation.at[2, 2].set(-1.0)
    elif failure == "singular":
        deformation = deformation.at[2, 2].set(0.0)
    else:
        state = state.at[0].set(-1.0)
    response = material.evaluate(
        deformation, state, jnp.asarray(1.0), parameters, 0.0, 0.01
    )
    assert not bool(response.successful)
    assert int(response.branch_code) == -1
    np.testing.assert_array_equal(response.trial_state, state)


def test_j2_prestress_round_trip_and_traced_admissibility() -> None:
    material = FiniteStrainJ2MPMConstitutivePlan()
    parameters = FiniteStrainJ2Parameters(10.0, 30.0, 5.0, 2.0)
    stress = jnp.asarray(
        [[-3.0, 0.2, 0.0], [0.2, -2.0, 0.1], [0.0, 0.1, -4.0]], dtype=jnp.float64
    )

    def initialize(value: Array) -> Array:
        return material.initialize_state_from_stress(value, parameters).admissible

    assert bool(jax.jit(initialize)(stress))
    initialized = material.initialize_state_from_stress(
        stress, parameters, equivalent_plastic_strain=0.2
    )
    response = material.evaluate(
        jnp.eye(3, dtype=jnp.float64),
        initialized.state,
        jnp.asarray(1.0),
        parameters,
        0.0,
        0.01,
    )
    np.testing.assert_allclose(response.first_piola, stress, atol=1e-10)
    inadmissible = material.initialize_state_from_stress(
        jnp.diag(jnp.asarray([20.0, -20.0, 0.0], dtype=jnp.float64)), parameters
    )
    assert not bool(inadmissible.admissible)


@pytest.mark.strict_jax
@pytest.mark.parametrize("material_kind", ("j2", "dp"))
def test_batched_hydrostatic_stress_reverse_derivative(material_kind: str) -> None:
    j2 = FiniteStrainJ2MPMConstitutivePlan()
    j2_parameters = FiniteStrainJ2Parameters(10.0, 30.0, 1.0, 2.0)
    dp = DruckerPragerMPMConstitutivePlan()
    dp_parameters = DruckerPragerParameters(10.0, 30.0, 1.0, 0.5, 0.2)
    history = initialize_metric_state((2,), jnp.float64)
    deformation = jnp.stack(
        (jnp.eye(3, dtype=jnp.float64), 0.9 * jnp.eye(3, dtype=jnp.float64))
    )

    def stress(value: Array) -> Array:
        density = jnp.ones((2,), dtype=jnp.float64)
        if material_kind == "j2":
            return j2.evaluate(
                value, history, density, j2_parameters, 0.0, 0.01
            ).first_piola
        return dp.evaluate(value, history, density, dp_parameters, 0.0, 0.01).first_piola

    forward = jax.jacfwd(stress)(deformation)
    reverse = jax.jacrev(stress)(deformation)
    assert jnp.all(jnp.isfinite(reverse))
    np.testing.assert_allclose(reverse, forward, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("scale", (1.0, 1000.0), ids=("base-units", "scaled-units"))
def test_j2_yield_boundary_tangent_uncertified(scale: float) -> None:
    material = FiniteStrainJ2MPMConstitutivePlan()
    parameters = FiniteStrainJ2Parameters(10 * scale, 30 * scale, 0.5 * scale, 2 * scale)
    # q=2G*(epsilon_1-epsilon_2)=yield strength, independently of units.
    deformation = jnp.diag(jnp.exp(jnp.asarray((1 / 60, -1 / 120, -1 / 120))))
    result = material.evaluate_linearized(
        deformation,
        material.initialize_state((), jnp.float64),
        jnp.asarray(1.0),
        parameters,
        0.0,
        0.01,
    )
    assert bool(result.response.successful)
    assert not bool(result.tangent_successful)
