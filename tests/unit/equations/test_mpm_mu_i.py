# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.applications.solid_mechanics import MuIMPMConstitutivePlan
from phydrax.equations import MPMConstitutiveResponse
from phydrax.rheology import MuIParameters


def _parameters() -> MuIParameters:
    return MuIParameters(0.38, 0.64, 0.28, 0.001, 2500.0)


def _deformation(shear: float, volume: float = -0.03) -> Array:
    return jnp.diag(
        jnp.exp(
            jnp.asarray(
                [volume / 3 + shear, volume / 3 - shear, volume / 3], dtype=jnp.float64
            )
        )
    )


def _response(
    f: Array,
    *,
    state: Array | None = None,
    parameters: MuIParameters | None = None,
    step: float = 0.01,
) -> MPMConstitutiveResponse:
    material = MuIMPMConstitutivePlan(1000.0, 3000.0)
    return material.evaluate(
        f,
        material.initialize_state((), f.dtype) if state is None else state,
        1500.0,
        _parameters() if parameters is None else parameters,
        0.0,
        step,
    )


@pytest.mark.parametrize(
    "shear,volume,branch",
    [(0.001, -0.03, 0), (0.08, -0.03, 1), (0.08, 0.03, 2), (0.0, 0.0, 2)],
)
def test_branches_and_independent_paper_return(
    shear: float, volume: float, branch: int
) -> None:
    f = _deformation(shear, volume)
    result = _response(f)
    assert bool(result.successful)
    assert int(result.branch_code) == branch
    assert result.dissipation_increment >= 0
    tau = np.asarray(result.first_piola @ f.T)
    if branch == 2:
        np.testing.assert_allclose(tau, 0.0, atol=1e-12)
        np.testing.assert_allclose(_unpack(result.trial_state), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(
            result.dissipation_increment,
            2000 * shear**2 + 1500 * volume**2,
            rtol=1e-10,
            atol=1e-12,
        )
    else:
        pressure = -3000 * volume
        trial_t = 2000 * abs(shear)
        if branch == 1:
            # Paper Eq. 3.15, stable smaller quadratic root; independently
            # adapt the paper's Cauchy stiffness to Kirchhoff with physical J.
            alpha = (
                0.28
                / (0.001 * np.sqrt(2500))
                * 1000
                * 0.01
                * np.sqrt(pressure / np.linalg.det(np.asarray(f)))
            )
            b = 0.64 * pressure + trial_t + alpha
            h = 0.64 * pressure * trial_t + 0.38 * pressure * alpha
            expected_t = 2 * h / (b + np.sqrt(b**2 - 4 * h))
        else:
            expected_t = trial_t
        expected_tau = np.diag([expected_t - pressure, -expected_t - pressure, -pressure])
        np.testing.assert_allclose(tau, expected_tau, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            np.linalg.det(_unpack(result.trial_state)),
            np.linalg.det(np.asarray(f)) ** 2,
            atol=1e-11,
        )
        if branch == 1:
            rate = float(result.diagnostics["plastic_shear_rate"])
            np.testing.assert_allclose(
                result.dissipation_increment, expected_t * 0.01 * rate, rtol=1e-9
            )
            assert 0 < float(result.suggested_step) < np.inf


def _unpack(state: Array) -> np.ndarray:
    a, b, c, d, e, f = np.asarray(state[:6])
    return np.asarray([[a, f, e], [f, b, d], [e, d, c]])


def test_disconnection_forgets_stress_and_reconnects_without_memory() -> None:
    flowing = _response(_deformation(0.08))
    separated_f = _deformation(0.1, 0.2)
    separated = _response(separated_f, state=flowing.trial_state)
    assert int(separated.branch_code) == 2
    stationary = _response(separated_f, state=separated.trial_state)
    np.testing.assert_allclose(stationary.first_piola, 0, atol=1e-10)
    reconnected_f = 0.999 * separated_f
    reconnected = _response(reconnected_f, state=separated.trial_state)
    assert bool(reconnected.successful)
    tau = reconnected.first_piola @ reconnected_f.T
    pressure = -9000 * np.log(0.999)
    np.testing.assert_allclose(tau, -pressure * np.eye(3), atol=1e-9)


@pytest.mark.parametrize(
    "shear,volume",
    [(0.001, -0.03), (0.08, -0.03), (0.02, 0.03), (0.0, -0.03), (0.0, 0.0)],
)
def test_tangent_against_finite_difference(shear: float, volume: float) -> None:
    material = MuIMPMConstitutivePlan(1000.0, 3000.0)
    f = _deformation(shear, volume)
    state = material.initialize_state((), f.dtype)
    linearized = material.evaluate_linearized(f, state, 1500.0, _parameters(), 0.0, 0.01)
    assert jnp.all(jnp.isfinite(linearized.algorithmic_tangent))
    if volume == 0.0:
        assert not bool(linearized.tangent_successful)
        return  # zero pressure is the nonsmooth connectivity boundary.
    assert bool(linearized.tangent_successful)
    direction = jnp.asarray(
        [[0.2, 0.3, -0.2], [0.3, -0.1, 0.4], [0.1, -0.2, 0.15]], dtype=f.dtype
    )
    epsilon = 1e-6
    finite_difference = (
        _response(f + epsilon * direction).first_piola
        - _response(f - epsilon * direction).first_piola
    ) / (2 * epsilon)
    action = jnp.einsum("ijkl,kl->ij", linearized.algorithmic_tangent, direction)
    np.testing.assert_allclose(action, finite_difference, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("field", ["mu_s", "mu_2", "I_0"])
def test_dynamic_parameter_gradient(field: str) -> None:
    f = _deformation(0.08)
    parameters = _parameters()
    selector = {
        "mu_s": lambda p: p.mu_s,
        "mu_2": lambda p: p.mu_2,
        "I_0": lambda p: p.I_0,
    }[field]
    initial = selector(parameters)

    def objective(value: Array) -> Array:
        updated = eqx.tree_at(selector, parameters, value)
        return _response(f, parameters=updated).first_piola[0, 0]

    gradient = jax.grad(objective)(initial)
    epsilon = 1e-5
    finite_difference = (objective(initial + epsilon) - objective(initial - epsilon)) / (
        2 * epsilon
    )
    assert jnp.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-6, atol=1e-7)


def test_objectivity_jit_batch_and_failure_retains_history() -> None:
    f = _deformation(0.08)
    rotation = jnp.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=f.dtype
    )
    np.testing.assert_allclose(
        _response(rotation @ f).first_piola,
        rotation @ _response(f).first_piola,
        atol=1e-9,
    )
    material = MuIMPMConstitutivePlan(1000.0, 3000.0)
    history = material.initialize_state((2,), f.dtype)
    batch = jnp.stack((f, _deformation(0.001)))
    response = eqx.filter_jit(material.evaluate)(
        batch, history, jnp.full((2,), 1500.0), _parameters(), 0.0, 0.01
    )
    assert jnp.all(response.successful)
    invalid = _response(f, step=0.0)
    assert not bool(invalid.successful)
    np.testing.assert_array_equal(invalid.trial_state, history[0])


def _gravity(time: Array, position: Array, velocity: Array, arguments: None) -> Array:
    del velocity, time, arguments
    return jnp.broadcast_to(
        jnp.asarray([0.0, 0.0, -9.81], dtype=position.dtype), position.shape
    )


def test_explicit_small_granular_column_step() -> None:
    bounds = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(11) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-0.25, -0.25, -0.25], [1.25, 1.25, 1.25]], dtype=jnp.float64))
    position = jnp.asarray(
        [[0.46, 0.47, z] for z in (0.26, 0.36, 0.46, 0.56)], dtype=jnp.float64
    )
    volume = jnp.full((4,), 0.001, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), 1500 * volume, ambient_dimension=3
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.MultilinearSplatAssignment()
    ).prepare(particles)
    material = MuIMPMConstitutivePlan(1000.0, 3000.0)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "mu-i-small-column",
            material,
            external_acceleration=_gravity,
            external_acceleration_id="gravity-9.81-z",
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(bounds, support_margin=0.2),
    )
    arguments = phx.equations.MaterialPointArguments(_parameters())
    velocity = jnp.zeros_like(position).at[:, 2].set(-0.01 * position[:, 2])
    # Admit a densely connected column; a stress-free column's near-zero
    # pressure produces a much shorter viscoplastic relaxation restriction.
    state = compiled.initialize_state(
        position,
        velocity,
        volume,
        arguments,
        deformation_gradient=jnp.broadcast_to(
            0.99 * jnp.eye(3, dtype=position.dtype), (4, 3, 3)
        ),
    )
    detail = compiled.dynamics.step_detailed(state, 1e-4, arguments)
    assert bool(detail.successful), (detail.rejection_reasons, detail.suggested_step)
    assert jnp.all(jnp.isfinite(detail.accepted_state.particles.material_state))
    assert detail.accepted_state.particles.material_state.shape == (4, 15)


@pytest.mark.strict_jax
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_strict_precision_return(precision: str) -> None:
    dtype = jnp.float32 if precision == "float32" else jnp.float64
    material = MuIMPMConstitutivePlan(
        jnp.asarray(1000.0, dtype=dtype), jnp.asarray(3000.0, dtype=dtype)
    )
    parameters = MuIParameters(
        *(jnp.asarray(value, dtype=dtype) for value in (0.38, 0.64, 0.28, 0.001, 2500.0))
    )
    f = jnp.diag(jnp.exp(jnp.asarray([0.07, -0.09, -0.01], dtype=dtype)))
    response = material.evaluate(
        f,
        material.initialize_state((), dtype),
        jnp.asarray(1500.0, dtype=dtype),
        parameters,
        0.0,
        0.01,
    )
    assert bool(response.successful)
    assert int(response.branch_code) == 1
    assert jnp.all(jnp.isfinite(response.first_piola))


def test_prestress_initializes_elastic_metric_and_previous_identity() -> None:
    material = MuIMPMConstitutivePlan(1000.0, 3000.0)
    stress = jnp.asarray(
        [[-80.0, 10.0, 0.0], [10.0, -100.0, 0.0], [0.0, 0.0, -120.0]],
        dtype=jnp.float64,
    )
    prepared = material.initialize_state_from_stress(
        jnp.stack((stress, 2.0 * stress)), _parameters()
    )
    assert jnp.all(prepared.admissible)
    assert prepared.state.shape == (2, 15)
    identity = jnp.broadcast_to(jnp.eye(3, dtype=stress.dtype), (2, 3, 3))
    np.testing.assert_array_equal(prepared.state[:, 6:].reshape((2, 3, 3)), identity)
    response = material.evaluate(
        identity, prepared.state, 1500.0, _parameters(), 0.0, 0.01
    )
    assert jnp.all(response.successful)
    np.testing.assert_allclose(response.first_piola, jnp.stack((stress, 2.0 * stress)))
    overstress = material.initialize_state_from_stress(
        jnp.diag(jnp.asarray((100.0, -100.0, -300.0))), _parameters()
    )
    assert not bool(overstress.admissible)
    expected_strain = np.asarray((100.0, -100.0, -300.0)) / 2000.0
    expected_strain += 100.0 / 2000.0 - 100.0 / 9000.0
    np.testing.assert_allclose(
        _unpack(overstress.state), np.diag(np.exp(2.0 * expected_strain))
    )


def test_relative_history_objectivity_and_failed_previous_deformation() -> None:
    material = MuIMPMConstitutivePlan(1000.0, 3000.0)
    f_previous = _deformation(0.08)
    previous = _response(f_previous)
    f = f_previous.at[0, 1].set(0.02)
    reference = _response(f, state=previous.trial_state)
    rotation = jnp.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=f.dtype
    )
    # Independent observer rotations at n and n+1: f transforms as R_next f R_n^T.
    old_rotation = rotation.T
    old_metric = (
        old_rotation @ jnp.asarray(_unpack(previous.trial_state)) @ old_rotation.T
    )
    observed = jnp.concatenate(
        (
            jnp.asarray(
                (
                    old_metric[0, 0],
                    old_metric[1, 1],
                    old_metric[2, 2],
                    old_metric[1, 2],
                    old_metric[0, 2],
                    old_metric[0, 1],
                )
            ),
            (old_rotation @ f_previous).reshape((9,)),
        )
    )
    rotated = _response(rotation @ f, state=observed)
    assert bool(rotated.successful)
    np.testing.assert_allclose(
        rotated.first_piola, rotation @ reference.first_piola, atol=1e-9
    )
    np.testing.assert_allclose(
        _unpack(rotated.trial_state),
        rotation @ _unpack(reference.trial_state) @ rotation.T,
        atol=1e-12,
    )
    invalid = previous.trial_state.at[6:15].set(0.0)
    refused = material.evaluate(f, invalid, 1500.0, _parameters(), 0.0, 0.01)
    assert not bool(refused.successful)
    np.testing.assert_array_equal(refused.trial_state, invalid)


def test_steady_simple_shear_to_ten_thousand_without_metric_conditioning_failure() -> (
    None
):
    """Independent Jop steady-rate inversion; q/p = sqrt(3) mu, not mu."""
    material = MuIMPMConstitutivePlan(200000.0, 600000.0)
    parameters = _parameters()
    pressure, friction = 100.0, 0.5
    inertial = 0.28 * (friction - 0.38) / (0.64 - friction)
    rate = inertial / 0.001 * np.sqrt(pressure / 2500.0)
    increment, steps = 0.1, 100000
    step = increment / rate
    initial = material.initialize_state_from_stress(
        -pressure * jnp.eye(3, dtype=jnp.float64), parameters
    )
    assert bool(initial.admissible)

    def advance(
        index: Array, carry: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        history, admitted, minimum_work, maximum_condition = carry
        deformation = (
            jnp.eye(3, dtype=history.dtype).at[0, 1].set((index + 1) * increment)
        )
        response = material.evaluate(deformation, history, 1500.0, parameters, 0.0, step)
        return (
            response.trial_state,
            admitted & response.successful & response.diagnostics["tangent_successful"],
            jnp.minimum(minimum_work, response.dissipation_increment),
            jnp.maximum(
                maximum_condition, response.diagnostics["trial_elastic_metric_condition"]
            ),
        )

    history, admitted, minimum_work, maximum_condition = eqx.filter_jit(
        lambda state: jax.lax.fori_loop(
            0,
            steps,
            advance,
            (state, jnp.asarray(True), jnp.asarray(jnp.inf), jnp.asarray(1.0)),
        )
    )(initial.state)
    assert bool(admitted)
    assert minimum_work > 0.0
    assert maximum_condition < 1.3
    final_f = jnp.eye(3, dtype=history.dtype).at[0, 1].set(steps * increment)
    np.testing.assert_array_equal(history[6:].reshape((3, 3)), final_f)
    linearized = material.evaluate_linearized(
        final_f.at[0, 1].add(increment), history, 1500.0, parameters, 0.0, step
    )
    assert bool(linearized.tangent_successful)
    response = linearized.response
    tau = np.asarray(response.first_piola @ final_f.at[0, 1].add(increment).T)
    measured_pressure = -np.trace(tau) / 3.0
    measured_t = np.sqrt(np.sum((tau + measured_pressure * np.eye(3)) ** 2) / 2.0)
    np.testing.assert_allclose(measured_pressure, pressure, rtol=1e-6)
    np.testing.assert_allclose(measured_t / measured_pressure, friction, rtol=6e-4)
    np.testing.assert_allclose(
        response.diagnostics["plastic_shear_rate"], rate, rtol=6e-4
    )
    # The obsolete pulled-back metric is ill-conditioned although B_e is benign.
    elastic = _unpack(history)
    host_f = np.asarray(final_f)
    pulled_back = np.linalg.solve(host_f, np.linalg.solve(host_f, elastic).T).T
    assert np.linalg.cond(pulled_back) > 1e15
    assert np.linalg.cond(host_f) < 1.01e8
    assert np.linalg.cond(elastic) < 1.01
    assert float(response.diagnostics["previous_deformation_condition"]) < 1.01e8
