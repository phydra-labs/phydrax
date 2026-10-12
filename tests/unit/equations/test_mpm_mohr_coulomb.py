# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from itertools import permutations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.optimize import LinearConstraint, minimize, nnls

import phydrax as phx


def _parameters(
    *, psi: float = 0.5, h: float = 0.0, phi: float = 0.5, cohesion: float = 1.0
) -> phx.applications.solid_mechanics.MohrCoulombParameters:
    return phx.applications.solid_mechanics.MohrCoulombParameters(
        10.0, 30.0, cohesion, phi, psi, h
    )


def _deformation(trial: tuple[float, float, float]) -> Array:
    stress = jnp.asarray(trial, dtype=jnp.float64)
    strain = (stress - jnp.mean(stress)) / 20 + jnp.mean(stress) / 90
    return jnp.diag(jnp.exp(strain))


def _normals(angle: float) -> np.ndarray:
    identity = np.eye(3, dtype=np.float64)
    return np.stack(
        [
            (1 + np.sin(angle)) * identity[i] - (1 - np.sin(angle)) * identity[j]
            for i, j in ((0, 2), (1, 2), (0, 1), (2, 0), (2, 1), (1, 0))
        ]
    )


@pytest.mark.parametrize(
    ("trial", "branch", "side"),
    [
        ((0.2, 0.0, -0.2), 0, 0),
        ((2.0, 0.0, -2.0), 1, 0),
        ((4.0, 4.0, -2.0), 2, -1),
        ((2.0, -4.0, -4.0), 2, 1),
        ((8.0, 7.0, 6.0), 3, 0),
    ],
    ids=("elastic", "face", "left-edge", "right-edge", "apex"),
)
@pytest.mark.parametrize("psi", (0.5, 0.2), ids=("associated", "nonassociated"))
@pytest.mark.strict_jax
def test_all_planes_kkt_and_hardening(
    trial: tuple[float, float, float], branch: int, side: int, psi: float
) -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    parameters = _parameters(psi=psi, h=0.4)
    deformation = _deformation(trial)
    response = material.evaluate(
        deformation,
        material.initialize_state((), jnp.float64),
        1.0,
        parameters,
        0.0,
        0.01,
    )
    assert bool(response.successful)
    assert int(response.branch_code) == branch
    assert int(response.diagnostics["edge_side"]) == side
    assert response.trial_state.shape == (7,)
    returned = np.sort(np.diag(np.asarray(response.first_piola @ deformation.T)))[::-1]
    alpha = float(response.trial_state[6])
    values = _normals(0.5) @ returned - 2 * (1 + 0.4 * alpha) * np.cos(0.5)
    assert np.max(values) < 1e-9
    compliance = (
        np.eye(3, dtype=np.float64) / 20
        + (1 / 90 - 1 / 20) * np.ones((3, 3), dtype=np.float64) / 3
    )
    increment = compliance @ (np.asarray(trial) - returned)
    matrix = np.vstack((_normals(psi).T, np.ones((1, 6), dtype=np.float64)))
    multipliers, residual = nnls(
        matrix, np.concatenate((increment, np.asarray((alpha,), dtype=np.float64)))
    )
    assert residual < 1e-10
    assert np.min(multipliers) >= 0
    np.testing.assert_allclose(multipliers * values, 0, atol=1e-9)
    np.testing.assert_allclose(
        response.diagnostics["ordered_plastic_increment"], increment, atol=1e-10
    )
    np.testing.assert_allclose(
        response.dissipation_increment, returned @ increment, atol=1e-10
    )


@pytest.mark.parametrize(
    ("trial", "branch"),
    [
        ((2.0, 0.0, -2.0), 1),
        ((4.0, 4.0, -2.0), 2),
        ((2.0, -4.0, -4.0), 2),
        ((8.0, 7.0, 6.0), 3),
    ],
    ids=("face", "left-edge", "right-edge", "apex"),
)
@pytest.mark.strict_jax
def test_associated_closest_point_energy_projection(
    trial: tuple[float, float, float], branch: int
) -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    deformation = _deformation(trial)
    response = material.evaluate(
        deformation,
        material.initialize_state((), jnp.float64),
        1.0,
        _parameters(),
        0.0,
        0.01,
    )
    assert bool(response.successful)
    assert int(response.branch_code) == branch
    reference = np.asarray(trial, dtype=np.float64)
    compliance = (
        np.eye(3, dtype=np.float64) / 20
        + (1 / 90 - 1 / 20) * np.ones((3, 3), dtype=np.float64) / 3
    )
    normals = _normals(0.5)

    def objective(stress: np.ndarray) -> float:
        difference = stress - reference
        return float(0.5 * difference @ compliance @ difference)

    def gradient(stress: np.ndarray) -> np.ndarray:
        return compliance @ (stress - reference)

    projected = minimize(
        objective,
        reference,
        jac=gradient,
        constraints=LinearConstraint(normals, -np.inf, 2 * np.cos(0.5)),
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 100},
    )
    assert projected.success
    np.testing.assert_allclose(
        np.diag(response.first_piola @ deformation.T), projected.x, atol=3e-7, rtol=1e-7
    )


@pytest.mark.parametrize(
    "trial", ((0.0, 0.0, 0.0), (4.0, 4.0, -2.0), (2.0, -4.0, -4.0), (8.0, 8.0, 8.0))
)
@pytest.mark.strict_jax
def test_repeated_spectrum_strict_tangent_matches_fd(
    trial: tuple[float, float, float],
) -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    state = material.initialize_state((), jnp.float64)
    deformation = _deformation(trial)
    parameters = _parameters()
    linearized = material.evaluate_linearized(
        deformation, state, 1.0, parameters, 0.0, 0.01
    )
    assert bool(linearized.tangent_successful)
    direction = jnp.asarray(
        ((0.13, 0.04, -0.02), (0.03, -0.07, 0.06), (-0.01, 0.02, 0.05)), dtype=jnp.float64
    )
    tangent = (
        linearized.algorithmic_tangent.reshape((9, 9)) @ direction.reshape((9,))
    ).reshape((3, 3))
    epsilon = 1e-6
    plus = material.evaluate(
        deformation + epsilon * direction, state, 1.0, parameters, 0.0, 0.01
    )
    minus = material.evaluate(
        deformation - epsilon * direction, state, 1.0, parameters, 0.0, 0.01
    )
    np.testing.assert_allclose(
        tangent,
        (plus.first_piola - minus.first_piola) / (2 * epsilon),
        atol=2e-7,
        rtol=2e-6,
    )


@pytest.mark.strict_jax
def test_tresca_limit() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    state = material.initialize_state((), jnp.float64)
    parameters = _parameters(phi=0.0, psi=0.0)
    deformation = _deformation((3.0, 0.0, -3.0))
    response = material.evaluate(deformation, state, 1.0, parameters, 0.0, 0.01)
    np.testing.assert_allclose(
        np.diag(response.first_piola @ deformation.T), (1.0, 0.0, -1.0), atol=1e-10
    )


@pytest.mark.strict_jax
def test_objectivity() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    state = material.initialize_state((), jnp.float64)
    parameters = _parameters(h=0.4)
    deformation = _deformation((3.0, 0.0, -3.0))
    response = material.evaluate(deformation, state, 1.0, parameters, 0.0, 0.01)
    rotation = jnp.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=jnp.float64
    )
    rotated = material.evaluate(rotation @ deformation, state, 1.0, parameters, 0.0, 0.01)
    np.testing.assert_allclose(
        rotated.first_piola, rotation @ response.first_piola, atol=1e-10
    )


@pytest.mark.parametrize("permutation", tuple(permutations(range(3))))
@pytest.mark.strict_jax
def test_permutation_equivariance(permutation: tuple[int, ...]) -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    state = material.initialize_state((), jnp.float64)
    parameters = _parameters(h=0.4)
    deformation = _deformation((3.0, 0.0, -3.0))
    matrix = jnp.eye(3, dtype=jnp.float64)[jnp.asarray(permutation)]
    value = material.evaluate(
        matrix @ deformation @ matrix.T, state, 1.0, parameters, 0.0, 0.01
    )
    original = material.evaluate(deformation, state, 1.0, parameters, 0.0, 0.01)
    np.testing.assert_allclose(
        value.first_piola, matrix @ original.first_piola @ matrix.T, atol=1e-10
    )
    np.testing.assert_allclose(
        value.diagnostics["plane_yield_values"],
        original.diagnostics["plane_yield_values"],
        atol=1e-10,
    )


@pytest.mark.strict_jax
def test_prestress_round_trip_and_admissibility() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    parameters = _parameters(h=0.4)
    stress = jnp.diag(jnp.asarray((-1.0, -2.0, -2.0), dtype=jnp.float64))
    prestress = material.initialize_state_from_stress(
        stress, parameters, equivalent_plastic_strain=0.2
    )
    assert bool(prestress.admissible)
    restored = material.evaluate(
        jnp.eye(3, dtype=jnp.float64), prestress.state, 1.0, parameters, 0.0, 0.01
    )
    np.testing.assert_allclose(restored.first_piola, stress, atol=1e-10)
    assert not bool(
        material.initialize_state_from_stress(
            8 * jnp.eye(3, dtype=jnp.float64), parameters
        ).admissible
    )


@pytest.mark.strict_jax
def test_invalid_deformation_retains_history() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    state = material.initialize_state((), jnp.float64)
    invalid = material.evaluate(
        jnp.diag(jnp.asarray((-1.0, 1.0, 1.0), dtype=jnp.float64)),
        state,
        1.0,
        _parameters(),
        0.0,
        0.01,
    )
    assert not bool(invalid.successful)
    assert int(invalid.branch_code) == -1
    np.testing.assert_array_equal(invalid.trial_state, state)


@pytest.mark.strict_jax
def test_zero_dilation_apex_requires_hardening() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    state = material.initialize_state((), jnp.float64)
    failed = material.evaluate(
        _deformation((8.0, 8.0, 8.0)), state, 1.0, _parameters(psi=0.0), 0.0, 0.01
    )
    assert not bool(failed.successful)
    np.testing.assert_array_equal(failed.trial_state, state)
    hardened = material.evaluate(
        _deformation((8.0, 8.0, 8.0)), state, 1.0, _parameters(psi=0.0, h=1.0), 0.0, 0.01
    )
    assert bool(hardened.successful)
    assert int(hardened.branch_code) == 3


@pytest.mark.strict_jax
def test_stress_unit_invariance_and_batched_prestress() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    deformation = _deformation((4.0, 4.0, -2.0))
    state = material.initialize_state((), jnp.float64)
    parameters = _parameters(h=0.4)
    original = material.evaluate(deformation, state, 1.0, parameters, 0.0, 0.01)
    scaled = material.evaluate(
        deformation,
        state,
        1.0,
        phx.applications.solid_mechanics.MohrCoulombParameters(
            10000.0, 30000.0, 1000.0, 0.5, 0.5, 400.0
        ),
        0.0,
        0.01,
    )
    np.testing.assert_allclose(scaled.trial_state, original.trial_state, atol=1e-10)
    np.testing.assert_allclose(scaled.first_piola, 1000 * original.first_piola, atol=1e-8)
    stresses = jnp.stack(
        (-jnp.eye(3, dtype=jnp.float64), -2 * jnp.eye(3, dtype=jnp.float64))
    )
    prestress = material.initialize_state_from_stress(stresses, parameters)
    assert jnp.all(prestress.admissible)
    np.testing.assert_allclose(
        material.evaluate(
            jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (2, 3, 3)),
            prestress.state,
            1.0,
            parameters,
            0.0,
            0.01,
        ).first_piola,
        stresses,
        atol=1e-10,
    )


@pytest.mark.strict_jax
def test_implicit_mpm_mc_step_from_prestressed_rest() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    parameters = _parameters()
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(4, periodic=True, endpoint=False)
            for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=jnp.float64))
    position = jnp.asarray(((0.27, 0.31, 0.35), (0.43, 0.38, 0.44)), dtype=jnp.float64)
    volume = jnp.full((2,), 0.01, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2, dtype=jnp.int32), volume, ambient_dimension=3
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("mc-prestress", material),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=jnp.float64),
            periodic=(True, True, True),
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(parameters)
    initial = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    prestress = material.initialize_state_from_stress(
        -0.1 * jnp.eye(3, dtype=jnp.float64), parameters
    )
    initial = eqx.tree_at(
        lambda state: state.particles.material_state,
        initial,
        jnp.broadcast_to(prestress.state, (2, 7)),
    )
    detailed = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).step_detailed(
        initial, 1e-5, arguments
    )
    assert bool(detailed.successful)
    assert bool(detailed.diagnostics.tangent_successful)
    assert detailed.accepted_state.particles.material_state.shape == (2, 7)


@pytest.mark.strict_jax
def test_jitted_response_and_parameter_derivative() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    deformation = _deformation((4.0, 4.0, -2.0))
    state = material.initialize_state((), jnp.float64)
    parameters = _parameters(h=0.4)
    response = jax.jit(material.evaluate)(deformation, state, 1.0, parameters, 0.0, 0.01)
    assert bool(response.successful)

    def objective(value: phx.applications.solid_mechanics.MohrCoulombParameters) -> Array:
        return jnp.sum(
            material.evaluate(deformation, state, 1.0, value, 0.0, 0.01).first_piola
        )

    derivative = eqx.filter_grad(objective)(parameters)
    assert jnp.isfinite(derivative.cohesion)


@pytest.mark.strict_jax
def test_yield_boundary_reports_invalid_tangent() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    parameters = _parameters()
    # Tresca-like pure shear on the frictional face: sum(tau1,tau3)=0.
    magnitude = float(jnp.cos(parameters.friction_angle))
    response = material.evaluate_linearized(
        _deformation((magnitude, 0.0, -magnitude)),
        material.initialize_state((), jnp.float64),
        1.0,
        parameters,
        0.0,
        0.01,
    )
    assert bool(response.response.successful)
    assert not bool(response.tangent_successful)


@pytest.mark.strict_jax
def test_nonassociated_repeated_edge_tangent() -> None:
    material = phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan()
    parameters = _parameters(psi=0.2, h=0.4)
    state = material.initialize_state((), jnp.float64)
    deformation = _deformation((4.0, 4.0, -2.0))
    response = material.evaluate_linearized(
        deformation, state, 1.0, parameters, 0.0, 0.01
    )
    assert bool(response.tangent_successful)
    direction = jnp.asarray(
        ((0.03, 0.04, 0.01), (0.02, -0.02, 0.03), (0.01, -0.04, 0.01)), dtype=jnp.float64
    )
    epsilon = 1e-6
    plus = material.evaluate(
        deformation + epsilon * direction, state, 1.0, parameters, 0.0, 0.01
    )
    minus = material.evaluate(
        deformation - epsilon * direction, state, 1.0, parameters, 0.0, 0.01
    )
    np.testing.assert_allclose(
        (response.algorithmic_tangent.reshape((9, 9)) @ direction.reshape((9,))).reshape(
            (3, 3)
        ),
        (plus.first_piola - minus.first_piola) / (2 * epsilon),
        atol=2e-7,
        rtol=2e-6,
    )
