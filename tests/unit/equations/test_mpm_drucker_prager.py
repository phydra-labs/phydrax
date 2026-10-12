#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


pytestmark = pytest.mark.strict_jax


def test_drucker_prager_return_and_tangent() -> None:
    deformation = jnp.asarray([[[1.0, 0.16, 0.0], [0.0, 0.92, 0.0], [0.0, 0.0, 1.08]]])
    density = jnp.asarray((1.0,))
    dp = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    dp_parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 0.05, 0.5, 0.2, 1.0
    )
    dp_response = dp.evaluate(
        deformation,
        dp.initialize_state((1,), jnp.float64),
        density,
        dp_parameters,
        0.0,
        0.01,
    )
    assert bool(dp_response.successful[0])
    assert dp_response.dissipation_increment[0] >= 0.0
    assert jnp.isfinite(dp_response.diagnostics["yield_residual"][0])
    assert jnp.isfinite(
        dp.evaluate_linearized(
            deformation,
            dp.initialize_state((1,), jnp.float64),
            density,
            dp_parameters,
            0.0,
            0.01,
        ).algorithmic_tangent
    ).all()


def _trial_deformation(p: float, q: float, g: float = 10.0, k: float = 30.0) -> Array:
    principal = jnp.asarray((2 * q / 3, -q / 3, -q / 3), dtype=jnp.float64)
    return jnp.diag(jnp.exp(principal / (2 * g) - p / (3 * k)))


@pytest.mark.parametrize("stretch", (1.0, 0.999))
def test_drucker_prager_repeated_spectrum_tangent_matches_fd(stretch: float) -> None:
    deformation = jnp.diag(jnp.asarray((1.0, 1.0, stretch), dtype=jnp.float64))
    plan = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 1.0, 0.5, 0.2
    )
    state = plan.initialize_state((), jnp.float64)
    result = plan.evaluate_linearized(deformation, state, 1.0, parameters, 0.0, 0.01)
    direction = jnp.asarray(((0.2, 0.1, 0.0), (0.1, -0.3, 0.1), (0.0, 0.1, 0.4)))
    step = 1e-6
    fd = (
        plan.evaluate(
            deformation + step * direction, state, 1.0, parameters, 0.0, 0.01
        ).first_piola
        - plan.evaluate(
            deformation - step * direction, state, 1.0, parameters, 0.0, 0.01
        ).first_piola
    ) / (2 * step)
    assert bool(result.tangent_successful)
    np.testing.assert_allclose(
        jnp.einsum("ijkl,kl->ij", result.algorithmic_tangent, direction),
        fd,
        rtol=1e-6,
        atol=1e-7,
    )


def test_drucker_prager_confinement_dilation_and_closed_form() -> None:
    plan = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 0.05, 0.5, 0.2, 1.0
    )
    state = plan.initialize_state((), jnp.float64)
    responses = [
        plan.evaluate(_trial_deformation(p, 1.0), state, 1.0, parameters, 0.0, 0.01)
        for p in (0.0, 2.0)
    ]
    m = 6 * np.sin(0.5) / (3 - np.sin(0.5))
    dilation = 6 * np.sin(0.2) / (3 - np.sin(0.2))
    strength = 6 * 0.05 * np.cos(0.5) / (3 - np.sin(0.5))
    gamma = (1 - strength) / (30 + 30 * m * dilation + 1)
    np.testing.assert_allclose(responses[0].trial_state[6], gamma, rtol=1e-10)
    assert responses[0].branch_code == 1
    assert responses[1].branch_code == 0
    # det(Cp^-1)=exp(-2 tr(epsilon_p)); dilation increases plastic volume.
    metric = responses[0].trial_state[:6]
    determinant = metric[0] * metric[1] * metric[2]
    assert determinant < 1
    hydrostatic = plan.evaluate(
        _trial_deformation(10.0, 0.0), state, 1.0, parameters, 0.0, 0.01
    )
    assert hydrostatic.branch_code == 0


@pytest.mark.parametrize(
    ("p", "q", "cohesion", "tension", "branch"),
    ((-2.0, 0.1, 0.05, None, 2), (-0.1, 0.01, 1.0, 0.02, 4), (-1.0, 1.0, 0.05, 0.02, 5)),
)
def test_drucker_prager_apex_tension_corner(
    p: float, q: float, cohesion: float, tension: float | None, branch: int
) -> None:
    plan = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, cohesion, 0.5, 0.2, 1.0, tensile_strength=tension
    )
    response = plan.evaluate(
        _trial_deformation(p, q),
        plan.initialize_state((), jnp.float64),
        1.0,
        parameters,
        0.0,
        0.01,
    )
    assert bool(response.successful)
    assert response.branch_code == branch
    assert response.diagnostics["yield_residual"] <= 1e-8
    m = 6 * np.sin(0.5) / (3 - np.sin(0.5))
    dilation = 6 * np.sin(0.2) / (3 - np.sin(0.2))
    strength = 6 * cohesion * np.cos(0.5) / (3 - np.sin(0.5))
    if branch == 2:
        volume = -(strength + m * p) / (1 / dilation + m * 30)
        expected_p, expected_q, alpha = p + 30 * volume, 0.0, volume / dilation
    elif branch == 4:
        assert tension is not None
        expected_p, expected_q, alpha = -tension, q, 0.0
        volume = (-p - tension) / 30
    else:
        assert tension is not None
        alpha = (q - strength + m * tension) / 31
        expected_p, expected_q = -tension, q - 30 * alpha
        volume = (-p - tension) / 30
    deformation = _trial_deformation(p, q)
    tau = np.asarray(response.first_piola @ deformation.T)
    actual_p = -np.trace(tau) / 3
    actual_q = np.sqrt(1.5 * np.sum((tau + actual_p * np.eye(3)) ** 2))
    np.testing.assert_allclose(
        (actual_p, actual_q, response.trial_state[6]),
        (expected_p, expected_q, alpha),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.prod(response.trial_state[:3]), np.exp(-2 * volume), atol=1e-10, rtol=1e-10
    )


def test_drucker_prager_prestress_objectivity_and_failure_retention() -> None:
    plan = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 1.0, 0.5, 0.2
    )
    stress = jnp.diag(jnp.asarray((-1.0, -2.0, -2.0)))
    prestress = plan.initialize_state_from_stress(stress, parameters)
    assert bool(prestress.admissible)
    response = plan.evaluate(jnp.eye(3), prestress.state, 1.0, parameters, 0.0, 0.01)
    np.testing.assert_allclose(response.first_piola, stress, atol=1e-10)
    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    deformation = _trial_deformation(0.0, 1.0)
    original = plan.evaluate(deformation, prestress.state, 1.0, parameters, 0.0, 0.01)
    rotated = plan.evaluate(
        rotation @ deformation, prestress.state, 1.0, parameters, 0.0, 0.01
    )
    np.testing.assert_allclose(
        rotated.first_piola, rotation @ original.first_piola, atol=1e-10
    )
    failed = plan.evaluate(
        jnp.diag(jnp.asarray((-1.0, 1.0, 1.0))),
        prestress.state,
        1.0,
        parameters,
        0.0,
        0.01,
    )
    assert not bool(failed.successful)
    np.testing.assert_array_equal(failed.trial_state, prestress.state)


def test_drucker_prager_zero_dilation_beyond_apex_refuses() -> None:
    plan = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 0.05, 0.5, 0.0
    )
    history = plan.initialize_state((), jnp.float64)
    response = plan.evaluate(
        _trial_deformation(-2.0, 0.1), history, 1.0, parameters, 0.0, 0.01
    )
    assert not bool(response.successful)
    assert response.branch_code == -1
    np.testing.assert_array_equal(response.trial_state, history)


def test_drucker_prager_negative_history_rejected() -> None:
    plan = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 1.0, 0.5, 0.2, 1.0
    )
    stress = -jnp.eye(3, dtype=jnp.float64)
    prestress = plan.initialize_state_from_stress(
        jnp.stack((stress, stress)), parameters, equivalent_plastic_strain=-0.1
    )
    assert not bool(jnp.any(prestress.admissible))
    response = plan.evaluate(
        jnp.stack((jnp.eye(3), jnp.eye(3))), prestress.state, 1.0, parameters, 0.0, 0.01
    )
    assert not bool(jnp.any(response.successful))
    np.testing.assert_array_equal(response.trial_state, prestress.state)


@pytest.mark.parametrize("linearized", (False, True), ids=("evaluate", "linearized"))
@pytest.mark.parametrize("batch", ((), (4,)), ids=("single", "particles"))
def test_drucker_prager_history_shape_rejected(
    linearized: bool, batch: tuple[int, ...]
) -> None:
    plan = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 1.0, 0.5, 0.2
    )
    deformation = jnp.broadcast_to(jnp.eye(3), batch + (3, 3))
    history = plan.initialize_state((1,) if not batch else (2, 2), jnp.float64)
    method = plan.evaluate_linearized if linearized else plan.evaluate
    with pytest.raises(ValueError, match="history shape"):
        method(deformation, history, 1.0, parameters, 0.0, 0.01)


@pytest.mark.parametrize("scale", (1.0, 1000.0), ids=("base-units", "scaled-units"))
@pytest.mark.parametrize("boundary", ("yield", "cutoff", "apex", "corner"))
def test_drucker_prager_boundary_tangent_uncertified(scale: float, boundary: str) -> None:
    plan = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    m = 6 * np.sin(0.5) / (3 - np.sin(0.5))
    dilation = 6 * np.sin(0.2) / (3 - np.sin(0.2))
    strength = 6 * np.cos(0.5) / (3 - np.sin(0.5))
    tension = 0.02 if boundary in ("cutoff", "corner") else None
    p, q = 0.0, strength
    if boundary == "cutoff":
        p, q = -0.02, 0.01
    elif boundary == "apex":
        q = 0.1
        p = (q - strength - q / 30 * (30 + 30 * m * dilation + 1)) / m
    elif boundary == "corner":
        q = strength
        gamma = (q - strength + m * 0.02) / 31
        p = -0.02 - 30 * dilation * gamma
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10 * scale,
        30 * scale,
        scale,
        0.5,
        0.2,
        scale,
        tensile_strength=None if tension is None else tension * scale,
    )
    response = plan.evaluate_linearized(
        _trial_deformation(p, q),
        plan.initialize_state((), jnp.float64),
        1.0,
        parameters,
        0.0,
        0.01,
    )
    assert bool(response.response.successful)
    assert not bool(response.tangent_successful)
